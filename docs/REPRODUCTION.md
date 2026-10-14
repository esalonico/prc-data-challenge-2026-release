# Reproduction

Two levels:

- **Serve path (verified).** `pipeline/` rebuilds the final file from the stored models, byte for byte. It was rerun
  from this repository in fresh workspaces before the release and reproduced F1, F2E, F2F, F2EP and the final
  F2FP byte for byte ([`manifests/final.json`](../manifests/final.json); the F0 variant was not rerun).
- **Training (documented, not rerun).** The code and run manifest of every stored model are included (section 6).
  The models were trained during the challenge and were not refitted for this release.

## 1. Hardware and time

|              |                                                                                                                      |
| ------------ | -------------------------------------------------------------------------------------------------------------------- |
| Machine used | 8-core Apple-silicon laptop, 32 GB RAM, macOS; torch legs on MPS                                                     |
| Serve path   | 85 min for`pipeline/run_all.sh` (step timings in section 5)                                                        |
| Disk         | inputs about 40 GB; two copy-on-write workspaces need about 25 GB of new disk                                        |
| Training     | CatBoost legs about 8 h each on 4 threads; the full set took several days on one workstation plus rented CPU servers |

## 2. Environment

```bash
uv sync                       # Python 3.13; every version pinned in uv.lock
```

The step scripts call `.venv/bin/python` of the checkout (override with `PY=`) and set `PYTHONPATH=src` themselves.

## 3. Data

Every input, its source and licence: [`DATA.md`](DATA.md).

```bash
# organiser files -> data/raw/ (needs the team credentials file minio-s3-credentials.json from the organisers)
uv run python scripts/data/download_data.py
shasum -a 256 -c manifests/organiser_files.sha256

# stored models and open-data snapshots from the public bucket -> artifacts/, data/external/ (SHA-256 checked)
uv run python scripts/release/fetch_artifacts.py
```

The remaining inputs are keyed by the challenge's row IDs or derived from ADS-B positions, so they are not
distributed. Section 4 rebuilds them.

## 4. Rebuilding the 2025 inputs

Not rerun for this release: the stored inputs were built during the challenge with the commands below, in this
order. All commands run from the repository root with `PYTHONPATH=src`. The 2025 rows they produce are what steps
10–30 then check byte for byte (section 5).

**4.1 ADS-B store** (ADSB.lol `globe_history`, ODbL). The raw archive is 2–4 GB per day (about 1.2 TB for the
days used). It is streamed and filtered on an ingest host: positions within 10 km of the ten airports and below field
elevation + 5,000 ft; 8 columns; ground samples kept, airborne samples thinned to one per 10 s (about 7.7 MB/day).

```bash
# on the ingest host (needs data/external/ourairports/airports.csv); about 44 days per hour
python scripts/data/adsb/batch.py --days-file days.txt --assets-json release_index.json --out-dir ~/adsb/out --workers 4
python scripts/data/adsb/slim.py
# copy the slim files to data/external/adsb_lol/slim/ (scripts/data/adsb/sync_slim.sh does it with rsync)
```

`days.txt` lists the UTC days of 2025, of January, February, June and July 2026, and 2026-05-31.
`release_index.json` is the GitHub release listing of `adsblol/globe_history_2025` and `adsblol/globe_history_2026`
(one release per day and instance). Archive days with no release: 2025-05-28 to 2025-06-10 and a few single days.

**4.2 Open data.** Either the snapshots from the bucket (`fetch_artifacts.py`, the exact versions used) or fresh
downloads, which can differ slightly:

```bash
python scripts/data/download_metar.py          # IEM METAR, 10 stations -> data/external/iem_metar/
python scripts/data/download_metar_alti.py     # altimeter series -> data/external/iem_alti/
python scripts/data/download_metar_raw.py      # raw METAR text -> data/external/iem_raw_metar/
python scripts/data/download_ourairports.py    # -> data/external/ourairports/
python scripts/data/download_osm_stands.py     # OSM parking positions and gates -> data/external/osm/
```

**4.3 Base caches** (planning, calendar and METAR base; temporal context; completed-arrival state; expected
occupancy), month by month, for the twelve 2025 months and the public ranking months:

```bash
python scripts/legs/s4_allyear.py --stage build-caches
```

**4.4 Feature blocks** (label-free on the rows they describe, built month by month):

```bash
for b in m1 m2; do
  python scripts/features/build_matched_context_blocks.py --block $b --source training --variant v1s             # months 1-10
  python scripts/features/build_matched_context_blocks.py --block $b --source allyear_plus_ranking --variant v1s
done
for s in training ranking combine; do python scripts/features/build_m4_blocks.py --block p --variant v1 --source $s; done
python scripts/features/build_aobt_local_features.py --scope train
python scripts/features/build_aobt_local_features.py --scope all --promote-owner-ruling D-017
```

**4.5 ADS-B blocks.** The ADS-B commands of `pipeline/10_regress.sh` on the full 2025 store: callsign dictionary,
stage 1 in both variants (with and without `features/callsign_dictionary.parquet`), the prize blocks
(`build_adsb_features.py prize`), the travel-time map (`build_adsb_coverage.py kin`, `votes`, `block`, `lean`) and the
D-pool features (`build_adsb_features.py features`). Then the trajectory store:

```bash
python scripts/legs/traj/extract.py --days <every 2025 day of the store, comma-separated>
```

**4.6 Unmatched chain.** The lineage frame is rebuilt by step 10 and the state files by step 20. The models are
stored (section 6).

After 4.1–4.6, `pipeline/run_all.sh` runs as in section 5.

## 5. Serving the final file

```bash
pipeline/run_all.sh
```

`run_all.sh` runs the steps below in two copy-on-write workspaces of this checkout (`pipeline/env.sh`; override
their locations with `SERVE_WS` and `REGRESS_WS`; logs in `artifacts/experiments/Final-serve4m-logs/` or `LOGS=`).
Every step can also be run on its own, in order. Nothing is refitted.

| Step                   | Workspace | What it does                                                                                                                                                     | Checks                                                                                                                                        |                      Time |
| ---------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------: |
| `00_workspaces.sh`   | both      | fresh copy-on-write copies; serve gets`final_ranking.parquet` as `ranking.parquet`; regress hides the ADS-B days the stored build lacked                     | inputs present                                                                                                                                |                    <1 min |
| `10_regress.sh`      | regress   | re-assembles the stored all-year matrices; rebuilds ADS-B stage 1 in both variants (with and without the callsign dictionary) and the months 1–10 lineage frame | train`c67a87cf…`, eval `b5b8212d…`, v3-lean `e7b81caa…`, prize dfit `0b39f62c…`, prize all-year value-identical to `70cfd2e5…` | 22 min (parallel with 20) |
| `20_blocks.sh`       | serve     | caches for the four months; stand-sequence, neighbour, planned-taxi and AOBT blocks; unmatched state files (`steps/unm_state.py`)                              | 2025 rows identical to the stored blocks                                                                                                      |                     8 min |
| `30_adsb.sh`         | serve     | ADS-B prize block (no dictionary), D-pool blocks and v3-lean block (dictionary) for the 120 serve days, on the 2025 pool of step 10                              | 2025 rows kept byte for byte (`steps/splice_block.py`)                                                                                      |                     9 min |
| `35_trajectories.sh` | serve     | trajectory extraction of the serve days, merged into the stored store                                                                                            | 0 join mismatches against stage 1                                                                                                             |                     3 min |
| `40_assemble.sh`     | serve     | the four-month matrices (month-isolated V3L; contiguous for T1)                                                                                                  | train rows equal the legs' fit matrices (`c67a87cf…`, `c98bb06a…`)                                                                      |                     3 min |
| `50_checks.sh`       | serve     | parity screen per month pair, dither and coverage table                                                                                                          | report only                                                                                                                                   |                    <1 min |
| `60_legs.sh`         | serve     | the fourteen legs with the stored models (GBM and torch legs in separate processes)                                                                              |                                                                                                                                               |      49 min (GBM legs 43) |
| `70_blend.sh`        | serve     | cell-weighted blend → T1 → unmatched chain (`scripts/compose/unm_serve.py`) → LOBT window → clip: F1                                                       | 670,790 rows, IDs =`final_submitting.parquet`, finite, ≥ 0                                                                                 |                    <1 min |
| `75_lirf_rule.sh`    | serve     | LIRF label-mechanism rule (`steps/lirf_rule.py`): F2F                                                                                                          | as step 70                                                                                                                                    |                    <1 min |
|                        |           |                                                                                                                                                                  |                                                                                                                                               |                           |

Why two ADS-B stage-1 variants and a hidden day set

The stored ADS-B artifacts were built from two variants of stage 1 (`configs/final/adsb_shipped_build.json`): the
prize all-year block (used by every tabular leg) **without** the callsign dictionary; the v3-lean block, the D-pool
blocks (unmatched sensor) and the trajectory store **with** it. They were also built from a day set that lacks seven
days now present in the ADS-B store. Rebuilding the 2025 pool with that day set reproduces the stored blocks exactly
(step 10); the 2026 serve days use the full store.

## 6. Training the stored models

All commands run from the repository root with `PYTHONPATH=src` and the inputs of section 4 in place. Each run's manifest (recipe, parameters, features, rounds, matrix SHA-256) is copied to `manifests/models/`.

**Matrices.** Every leg is fitted on the same all-year matrix (train `c67a87cf…`, 2,062,622 rows, 265 columns):

```bash
python scripts/legs/s4_allyear.py --stage fit --train-months 1-12 --eval ranking --tag allyear_iso_v3l \
  --extra-features artifacts/experiments/Matched-features-20260925/m1_stand_v1s/allyear_plus_ranking.parquet \
  --extra-features artifacts/experiments/Matched-features-20260925/m2_follow_v1s/allyear_plus_ranking.parquet \
  --extra-features artifacts/experiments/Matched-features-20260925/m4_p_v1/s4_all.parquet \
  --extra-features artifacts/experiments/AOBT-local-only-20260925/features/aobt_features_allyear_plus_ranking.parquet \
  --extra-features artifacts/experiments/ADSB-features-20260925/prize/adsb_block_prize_allyear.parquet \
  --extra-features artifacts/experiments/ADSB-coverage-20260926/v3/adsb_v3lean_block_allyear.parquet \
  --temporal-mode isocap_v1 --threads 8 --assemble-only
```

**CatBoost legs** (`scripts/legs/s4_allyear.py`; 7,994 trees, parameters `artifacts/experiments/E37-optuna/shortlist_t31.json` from the bucket):

```bash
python scripts/legs/s4_allyear.py --stage fit --train-months 1-12 --eval ranking --tag allyear_iso_v3l_cb8k_plain <the six --extra-features above> \
  --temporal-mode isocap_v1 --iterations 7994 --threads 4
python scripts/legs/s4_allyear.py --stage fit --train-months 1-12 --eval ranking --tag allyear_iso_v3l_cb8k_delta <the six --extra-features above> \
  --temporal-mode isocap_v1 --iterations 7994 --threads 4 --target-offset-col takeoff_minus_aobt_sec --target-sign -1 --target-offset-fill-col unimpeded_p10 --target-winsor 7200
python scripts/legs/s4_allyear.py --stage fit --train-months 1-12 --eval ranking --tag allyear_iso_v3l_cb8k_eobtres <the six --extra-features above> \
  --temporal-mode isocap_v1 --iterations 7994 --threads 4 --target-offset-col takeoff_minus_eobt_sec --target-offset-fill-col unimpeded_p10 --target-winsor 7200
```

**LightGBM legs** (`scripts/legs/train_lgbm_matrix.py`; learning rate 0.01, plain leaves, rounds fixed from an early-stopped holdout run):

```bash
python scripts/legs/train_lgbm_matrix.py --run-dir artifacts/experiments/S4-allyear-rebuild-20260925/runs/allyear_iso_v3l --tag allyear_iso_v3l_lgbm_plain_lr01 --fixed-rounds 9993 \
  --learning-rate 0.01 --linear-tree 0 --threads 16
python scripts/legs/train_lgbm_matrix.py --run-dir artifacts/experiments/S4-allyear-rebuild-20260925/runs/allyear_iso_v3l --tag allyear_iso_v3l_lgbm_delta_lr01 --fixed-rounds 8754 \
  --learning-rate 0.01 --linear-tree 0 --threads 16 --target-offset-col takeoff_minus_aobt_sec --target-sign -1 --target-offset-fill-col unimpeded_p10 --target-winsor 7200
python scripts/legs/train_lgbm_matrix.py --run-dir artifacts/experiments/S4-allyear-rebuild-20260925/runs/allyear_iso_v3l --tag allyear_iso_v3l_lgbm_eobtres_lr01 --fixed-rounds 8427 \
  --learning-rate 0.01 --linear-tree 0 --threads 16 --target-offset-col takeoff_minus_eobt_sec --target-offset-fill-col unimpeded_p10 --target-winsor 7200
python scripts/legs/train_lgbm_matrix.py --run-dir artifacts/experiments/S4-allyear-rebuild-20260925/runs/allyear_iso_v3l --tag allyear_iso_v3l_lgbm_lobtoff_lr01 --fixed-rounds 9961 \
  --learning-rate 0.01 --linear-tree 0 --threads 16 --target-offset-col takeoff_minus_lobt_sec --target-offset-fill-col unimpeded_p10 --target-winsor 7200
```

The F0 variant's LightGBM legs (`allyear_iso_v3l_lgbm_*` without `_lr01`) are the same commands at learning rate 0.03 with the rounds of their manifests.

**MLP legs** (`scripts/legs/train_nn_matrix.py`; 8 seeds each, recipe in the manifest):

```bash
python scripts/legs/train_nn_matrix.py --run-dir artifacts/experiments/S4-allyear-rebuild-20260925/runs/allyear_iso_v3l --tag nn_v14_iso_v3l_allyear --target delta --fixed-epochs 10 --seeds 0,1,2,3,4,5,6,7
python scripts/legs/train_nn_matrix.py --run-dir artifacts/experiments/S4-allyear-rebuild-20260925/runs/allyear_iso_v3l --tag nn_v15_iso_v3l_allyear --target resid --fixed-epochs 10 --seeds 0,1,2,3,4,5,6,7
python scripts/legs/train_nn_matrix.py --run-dir artifacts/experiments/S4-allyear-rebuild-20260925/runs/allyear_iso_v3l --tag nn_v16_iso_v3l_allyear --target eobtres --fixed-epochs 10 --seeds 0,1,2,3,4,5,6,7
```

**PLR and ctx-GRU legs** (`scripts/legs/nnx/`): the split `ay3liso` cache is prepared by `prep.py`, then each architecture is fitted with its stored config:

```bash
python scripts/legs/nnx/prep.py ay3liso
python scripts/legs/nnx/train.py --split ay3liso --stage final --save-models \
  --config "$(python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["cfg"]))' manifests/models/nnx_search_ay3liso/plr_s03_resnet_f3/result.json)"
python scripts/legs/nnx/train.py --split ay3liso --stage final --save-models \
  --config "$(python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["cfg"]))' manifests/models/nnx_search_ay3liso/plr_s03_eobtres_f3/result.json)"
python scripts/legs/nnx/train.py --split ay3liso --stage final --save-models \
  --config "$(python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["cfg"]))' manifests/models/nnx_search_ay3liso/ctx_gru_plr_f2/result.json)"
```

**Trajectory leg** (`scripts/legs/traj/`): tracks for the split, then the fit:

```bash
python scripts/legs/traj/tracks.py split ay3liso --matrices artifacts/experiments/S4-allyear-rebuild-20260925/runs/allyear_iso_v3l/train_matrix.parquet artifacts/experiments/S4-allyear-rebuild-20260925/runs/allyear_iso_v3l/eval_matrix.parquet --pool AY
python scripts/legs/traj/train_traj.py --split ay3liso --stage final --save-models \
  --config "$(python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["cfg"]))' manifests/models/traj_search_ay3liso/traj_conv_resnet_f3/result.json)"
```

**Blend weights** (`scripts/compose/oof_blend_weights.py`). Every leg above is also fitted on the four D folds
({2, 8}, {3, 9}, {4, 10}, {5, 6}; same commands with the D-fold matrices), giving the out-of-fold bank
`artifacts/experiments/Dfold-OOF-V3L-20260928`. Then:

```bash
python scripts/compose/oof_blend_weights.py --json results/blend/lofo.json
```

`configs/final/blend_weights_d.json` is the method-d solution on all four folds (`final_weights.d` of that output).

**Tail model T1** (`scripts/compose/build_matched_tail_ranking.py`): the all-year `p_long` / `mu_long` boosters on
the contiguous all-year matrix (`artifacts/experiments/Matched-tail-20260926/models/`). Its design was selected on
the D folds with `scripts/compose/run_matched_tail.py`.

```bash
python scripts/compose/build_matched_tail_ranking.py --arm T1
```

**Unmatched chain** (`scripts/compose/`, months 1–10 of 2025): the P(copy) model and lineage frame
(`build_unmatched_noncopy_ranking.py --phase lineage --phase features --phase fit`), the non-copy expert
(`build_unmatched_noncopy_ranking.py --phase fit --fix-october-sidecar`, capped by `build_unmatched_cap_ranking.py`),
the LIRF stand model (`build_unmatched_p1_ranking.py`) and the ADS-B sensor expert (`run_unmatched_sensor.py`,
`build_unmatched_sensor_ranking.py --usens-g`). The models they write, and how `unm_serve.py` combines them, are listed
in the docstring of `scripts/compose/unm_serve.py`.

**LIRF rule** (`pipeline/steps/lirf_rule.py`): refits `configs/final/lirf_rule_params.json` from the 2025 files and
checks it against the stored one.

```bash
python pipeline/steps/lirf_rule.py fit --check
```

## 7. Verification

```bash
shasum -a 256 $SERVE_WS/artifacts/experiments/Final-serve4m/{F0,F1,F2E,F2F,F2EP,F2FP}.parquet
python -m json.tool manifests/final.json
```

Each file's SHA-256 must equal the one in `manifests/final.json`. The step logs carry the intermediate checks
(section 5); `<file>.validation.json` next to each file records rows, ID order against `final_submitting.parquet`,
finiteness and the minimum. Tested with the versions of `uv.lock` on macOS (Apple silicon). Other platforms can differ
in the last bits of floating-point results (BLAS, MPS vs CPU torch); compare with a tolerance there.
