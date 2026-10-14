# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-V4 (D-072): Nov/Dec 2025 predictions of the stored D-fit legs.

Every leg below was fitted on the D months (2-6, 8-10) of an ADS-B D-fit matrix.  This script
loads each stored model, predicts

* the Nov/Dec 2025 evaluation matrix ``runs/v4_novdec_dfit_adsb/eval_matrix.parquet``
  (``train_s4_allyear.py --eval novdec --assemble-only``: D-month references, as the JAN_JUL
  eval rows of the D-fit were built), and
* the leg's own JAN_JUL eval matrix, as a reproduction check against the stored
  ``predictions_<tag>_janjul.parquet`` (``s4_raw_prediction``),

inverts each leg's target exactly as its trainer does (``s4.TargetSpec.predict``; LightGBM:
categorical levels fixed on the D-fit training rows; MLP: ``train_nn_matrix.Preprocessor`` fitted
on the D-fit training rows, seed mean of the standardised prediction, then inverted), and writes
``artifacts/experiments/V4-novdec-20260927/legs/<leg>_novdec.parquet`` with
``MVT_ID_mvt, month, AIRPORT, nm_unmatched, label, s4_raw_prediction, prediction``.

``prediction`` is the trainer's ranking-side composition (``s4.compose_ranking_matched``:
E17 -> E18 -> clip >= 0).  Unmatched rows carry the same composition of the leg's own raw
prediction; the blend tool gives every unmatched row one fixed treatment (see ``v4_gate.py``).

Nov/Dec labels are copied into the output for scoring only; nothing here is fitted.

Usage::

  PYTHONPATH=src .venv/bin/python scripts/legs/leg_models.py --legs cb,cbd,cbe,lgb,lgbd,lgbe,lgbl,nn14,nn15,nn16
  PYTHONPATH=src .venv/bin/python scripts/legs/leg_models.py --legs cb --swap-check   # ADS-B pool sensitivity on JAN_JUL

Bake (D-076 gap: Nov/Dec predictions of stage-B legs as driver jobs; every option is off by default)::

  ... --leg-spec lgb_plain_s=<bake>/lgb_plain_s/lgb_plain_s_dfit:lgbm:model_fixed_rounds.txt \
      --dfit-run <runs>/bake_dfit_iso_v3l --novdec-run <runs>/bake_novdec_iso_v3l --out-dir <bake>/novdec \
      --legs lgb_plain_s --jj-repro-only

``--leg-spec KEY=RUN_DIR:KIND:MODEL_FILE`` registers a leg that is not in ``LEGS`` (kinds as ``LEGS``);
``--dfit-run`` is the D-fit matrix run the legs were fitted on (default ``dfit_full_adsb_historder``),
``--out-dir`` replaces ``V4-novdec-20260927`` (leg files go to ``<out-dir>/legs/``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_THREADS = os.environ.get("V4_THREADS", "4")
for _v in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from prc_challenge import matrix_arms as ma  # noqa: E402
from prc_challenge import s4_allyear as s4  # noqa: E402

MAIN = Path(os.environ.get("PRC_MAIN_REPO", Path(__file__).resolve().parents[2]))
RUNS = MAIN / "artifacts/experiments/S4-allyear-rebuild-20260925/runs"
OUT = MAIN / "artifacts/experiments/V4-novdec-20260927"
NOVDEC = RUNS / "v4_novdec_dfit_adsb"
DFIT = RUNS / "dfit_full_adsb_historder"
D_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)

# leg -> (run dir, kind, model file(s))
LEGS: dict[str, tuple[str, str, str]] = {
    "cb": ("dfit_full_adsb_historder", "catboost", "model.cbm"),
    "cbd": ("dfit_full_adsb_delta_historder", "catboost", "model.cbm"),
    "cbe": ("vm_v11_cb_eobtres_adsb_dfit", "catboost", "model.cbm"),
    "lgb": ("vm_v6_lgbm_adsb_dfit", "lgbm", "model_stageB_allD.txt"),
    "lgbd": ("vm_v9_lgbm_delta_adsb_dfit", "lgbm", "model_stageB_allD.txt"),
    "lgbe": ("vm_v10_lgbm_eobtres_adsb_dfit", "lgbm", "model_stageB_allD.txt"),
    "lgbl": ("vm_v12_lgbm_lobtoff_adsb_dfit", "lgbm", "model_stageB_allD.txt"),
    "nn14": ("nn_v14_mlp_adsb_dfit_s8", "mlp", "model_stageB_seed{s}.pt"),
    "nn15": ("nn_v15_mlp_resid_adsb_dfit_s8", "mlp", "model_stageB_seed{s}.pt"),
    "nn16": ("nn_v16_mlp_eobtres_adsb_dfit_s8", "mlp", "model_stageB_seed{s}.pt"),
    # SA-PERAP candidate 14th leg (per-airport-scope LightGBM, AOBT-delta target); boosters under per_airport/<AIRPORT>/
    "lgbm_peraptdelta": ("lgbm_peraptdelta_dfit", "lgbm_apt", "model_stageB_allD.txt"),
}
DEFAULT_LEGS = [k for k, v in LEGS.items() if v[1] != "lgbm_apt"]  # the thirteen SA-V4 legs; candidates are named explicitly
LEG_COLUMNS = ("MVT_ID_mvt", "month", "AIRPORT", "nm_unmatched", "label", "s4_raw_prediction", "prediction")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg} (peak_rss={s4.peak_rss_gb():.2f} GB)", flush=True)


def target_of(manifest: dict[str, Any]) -> s4.TargetSpec:
    """The leg's fit target as its run manifest states it (None / 'default ...' = S4 residual)."""

    t = manifest.get("target")
    if t is None and isinstance(manifest.get("fit"), dict):
        t = manifest["fit"].get("target")
    if isinstance(manifest.get("recipe"), dict) and "target" in manifest["recipe"]:
        t = manifest["recipe"]["target"]
    if t is None or isinstance(t, str):
        return s4.TargetSpec()
    return s4.TargetSpec(offset_col=t["offset_col"], sign=int(t["sign"]), fill_col=t.get("fill_col"), winsor_sec=t.get("winsor_sec"))


def eval_matrix_for(run: Path) -> Path:
    """The JAN_JUL eval matrix the leg was scored on (VM / NN runs used the D-fit run's matrix)."""

    own = run / "eval_matrix.parquet"
    if own.exists():
        return own
    man = json.loads((run / "run_manifest.json").read_text())
    path = Path(man["matrix"]["eval_matrix"]["path"])
    sha = man["matrix"]["eval_matrix"]["sha256"]
    local = DFIT / "eval_matrix.parquet"
    if s4.sha256_file(local) != sha:
        raise RuntimeError(f"{run.name}: eval matrix {path} (sha {sha[:12]}) is not the local D-fit eval matrix")
    return local


def train_matrix_for(run: Path) -> Path:
    own = run / "train_matrix.parquet"
    if own.exists():
        return own
    man = json.loads((run / "run_manifest.json").read_text())
    sha = man["matrix"]["train_matrix"]["sha256"]
    local = DFIT / "train_matrix.parquet"
    if s4.sha256_file(local) != sha:
        raise RuntimeError(f"{run.name}: train matrix sha {sha[:12]} is not the local D-fit train matrix")
    return local


# --------------------------------------------------------------------------- predictors


class CatBoostLeg:
    def __init__(self, run: Path, model_file: str):
        from catboost import CatBoostRegressor

        self.model = CatBoostRegressor()
        self.model.load_model(str(run / model_file))

    def t_hat(self, frame: pl.DataFrame, cats: list[str], nums: list[str]) -> np.ndarray:
        from prc_challenge.campaign.models import to_pool

        return np.asarray(self.model.predict(to_pool(frame, cats, nums), thread_count=int(_THREADS)), dtype=np.float64)


class LgbmLeg:
    def __init__(self, run: Path, model_file: str, levels: dict[str, list[str]]):
        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=str(run / model_file))
        self.levels = levels

    def t_hat(self, frame: pl.DataFrame, cats: list[str], nums: list[str]) -> np.ndarray:
        X = ma.numpy_matrix(frame, cats, nums, self.levels)
        return np.asarray(self.booster.predict(X, num_threads=int(_THREADS)), dtype=np.float64)


class PerAirportLgbmLeg:
    """Per-airport-scope LightGBM (``train_lgbm_matrix.py --per-airport``): each row by its own airport's booster."""

    def __init__(self, run: Path, model_file: str, levels: dict[str, list[str]]):
        from prc_challenge import per_airport_lgbm as pa

        self.pa = pa
        self.boosters = pa.load_boosters(run / "per_airport", model_file)
        self.levels = levels

    def t_hat(self, frame: pl.DataFrame, cats: list[str], nums: list[str]) -> np.ndarray:
        X = ma.numpy_matrix(frame, cats, nums, self.levels)
        return self.pa.predict(self.boosters, X, self.pa.airports_of(frame), num_threads=int(_THREADS))


class MlpLeg:
    """V14/V15/V16: ``train_nn_matrix`` preprocessing refitted on the D-fit train rows (deterministic)."""

    def __init__(self, run: Path, model_file: str, train: pl.DataFrame, cats: list[str], nums: list[str], target: s4.TargetSpec):
        import torch

        import train_nn_matrix as tnm

        man = json.loads((run / "run_manifest.json").read_text())
        recipe = man["recipe"]
        self.seeds = list(recipe["seeds"])
        self.pre = tnm.Preprocessor(cats, nums).fit(train)
        stored = man["preprocessor"]
        if self.pre.describe() != stored:
            raise RuntimeError(f"{run.name}: refitted preprocessor differs from the stored description")
        tnm.TARGET = target
        y = target.with_fit_label(train)[s4.RESIDUAL_LABEL].to_numpy()
        self.mu, self.sd = float(y.mean()), float(y.std())
        sb = man["stage_b"]
        if abs(self.mu - sb["target_mean"]) > 1e-6 or abs(self.sd - sb["target_sd"]) > 1e-6:
            raise RuntimeError(f"{run.name}: target mean/sd {self.mu:.6f}/{self.sd:.6f} != stored {sb['target_mean']}/{sb['target_sd']}")
        self.device = torch.device("cpu")
        self.models = []
        for s in self.seeds:
            net = tnm.build_model(self.pre.sizes, self.pre.n_dense, tuple(recipe["widths"]), recipe["dropout"])
            net.load_state_dict(torch.load(run / model_file.format(s=s), map_location="cpu"))
            net.eval()
            self.models.append(net)
        self.run = run

    def seed_t_hats(self, frame: pl.DataFrame) -> list[np.ndarray]:
        import torch

        cat, dense = self.pre.transform(frame)
        ct = torch.from_numpy(cat.astype(np.int64))
        dt = torch.from_numpy(dense)
        out = []
        with torch.no_grad():
            for net in self.models:
                parts = [net(ct[a : a + 65536], dt[a : a + 65536].float()).float().numpy() for a in range(0, ct.shape[0], 65536)]
                out.append(np.concatenate(parts).astype(np.float64) * self.sd + self.mu)
        return out

    def t_hat(self, frame: pl.DataFrame, cats: list[str], nums: list[str]) -> np.ndarray:
        # V14 recipe: mean of the standardised predictions (equivalently of t_hat), then inverted
        return np.mean(np.vstack(self.seed_t_hats(frame)), axis=0)


# --------------------------------------------------------------------------- composition / io


def compose(evaluation: pl.DataFrame, raw: np.ndarray) -> pl.DataFrame:
    """Ranking-side composition on every row: E17 -> E18 -> clip >= 0 (``s4.compose_ranking_matched``)."""

    composed = s4.compose_ranking_matched(evaluation, raw)
    return composed.select(
        "MVT_ID_mvt",
        "month",
        "AIRPORT",
        "nm_unmatched",
        pl.col(s4.LABEL_COLUMN).cast(pl.Float64).alias("label"),
        "s4_raw_prediction",
        "prediction",
    )


def write_leg(frame: pl.DataFrame, leg: str, extra: dict[str, Any], suffix: str = "novdec") -> dict[str, Any]:
    path = OUT / "legs" / f"{leg}_{suffix}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    s4.atomic_parquet(frame.select(*LEG_COLUMNS), path)
    info = {"leg": leg, "path": str(path), "sha256": s4.sha256_file(path), "rows": frame.height, **extra}
    s4.atomic_json(OUT / "legs" / f"{leg}_{suffix}.manifest.json", info)
    return info


def jj_check(leg: str, run: Path, raw_jj: np.ndarray, ids: np.ndarray, with_rmse: bool = True) -> dict[str, Any]:
    if not with_rmse:
        # reproduction only: predictions vs the stored JAN_JUL predictions; no label is read
        stored = pl.read_parquet(run / f"predictions_{run.name}_janjul.parquet").select("MVT_ID_mvt", "s4_raw_prediction")
        j = stored.join(pl.DataFrame({"MVT_ID_mvt": ids, "mine": raw_jj}), on="MVT_ID_mvt", how="inner", validate="1:1")
        if j.height != min(stored.height, ids.size):
            raise RuntimeError(f"{leg}: JAN_JUL ids do not match ({j.height} vs {stored.height})")
        a, b = j["mine"].to_numpy(), j["s4_raw_prediction"].to_numpy()
        return {"n": int(j.height), "max_abs_diff": float(np.max(np.abs(a - b))), "mean_abs_diff": float(np.mean(np.abs(a - b))),
                "n_bit_identical": int((a == b).sum()), "pearson_r": float(np.corrcoef(a, b)[0, 1]), "labels_read": False}
    stored = pl.read_parquet(run / f"predictions_{run.name}_janjul.parquet").select("MVT_ID_mvt", "s4_raw_prediction", s4.LABEL_COLUMN, "nm_unmatched")
    mine = pl.DataFrame({"MVT_ID_mvt": ids, "mine": raw_jj})
    j = stored.join(mine, on="MVT_ID_mvt", how="inner", validate="1:1")
    if j.height != min(stored.height, ids.size):
        raise RuntimeError(f"{leg}: JAN_JUL ids do not match ({j.height} vs {stored.height})")
    a, b = j["mine"].to_numpy(), j["s4_raw_prediction"].to_numpy()
    m = j["nm_unmatched"].to_numpy() == 0
    y = j[s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
    return {
        "n": int(j.height),
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
        "n_bit_identical": int((a == b).sum()),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "matched_raw_rmse_mine": float(np.sqrt(np.mean((y[m] - a[m]) ** 2))),
        "matched_raw_rmse_stored": float(np.sqrt(np.mean((y[m] - b[m]) ** 2))),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legs", default=",".join(DEFAULT_LEGS))
    p.add_argument("--report-name", help="report json under legs/ (default predict_report[_<suffix>].json; candidates use their own so SA-V4's report is untouched)")
    p.add_argument("--no-jj-check", action="store_true")
    p.add_argument("--jj-repro-only", action="store_true", help="JAN_JUL reproduction without the RMSE fields (no label read; predictions-only legs)")
    p.add_argument("--swap-check", action="store_true", help="also predict JAN_JUL on the final-ADS-B-block eval matrix (pool sensitivity)")
    p.add_argument("--novdec-run", default=NOVDEC.name, help="run dir (under runs/) holding the Nov/Dec eval matrix")
    p.add_argument("--suffix", default="novdec", help="leg file suffix: legs/<leg>_<suffix>.parquet")
    p.add_argument("--swap-only", action="store_true", help="only write <leg>_janjul_swap_raw.npy (JAN_JUL on the final-pool eval matrix)")
    p.add_argument("--swap-kinds", default="catboost,lgbm,mlp", help="leg kinds that get the swap check")
    p.add_argument("--jj-sample", type=int, default=0, help="LightGBM legs: reproduce JAN_JUL on a fixed random subset of this many rows (0 = all)")
    p.add_argument("--leg-spec", action="append", default=[], help="bake: KEY=RUN_DIR:KIND:MODEL_FILE, a leg outside LEGS (repeatable)")
    p.add_argument("--dfit-run", help="bake: the D-fit matrix run of the legs (default dfit_full_adsb_historder)")
    p.add_argument("--out-dir", help="bake: output root instead of V4-novdec-20260927 (leg files under <out-dir>/legs/)")
    a = p.parse_args(argv)
    global DFIT, OUT
    if a.dfit_run:
        DFIT = Path(a.dfit_run)
    if a.out_dir:
        OUT = Path(a.out_dir)
        (OUT / "legs").mkdir(parents=True, exist_ok=True)
    for spec in a.leg_spec:
        key, _, rest = spec.partition("=")
        run_dir, kind, model_file = rest.rsplit(":", 2)
        if kind not in ("catboost", "lgbm", "lgbm_apt", "mlp"):
            raise SystemExit(f"--leg-spec {spec}: unknown kind {kind}")
        LEGS[key] = (run_dir, kind, model_file)  # an absolute run dir: RUNS / run_dir is run_dir
    legs = [x for x in a.legs.split(",") if x]
    nd_run = RUNS / a.novdec_run
    cats, nums = ma.feature_lists_for_run(nd_run)
    if (cats, nums) != ma.feature_lists_for_run(DFIT):
        raise SystemExit("Nov/Dec matrix feature lists differ from the D-fit matrix")
    nd = pl.read_parquet(nd_run / "eval_matrix.parquet")
    if sorted(nd["month"].unique().to_list()) != [11, 12]:
        raise SystemExit("Nov/Dec eval matrix holds other months")
    log(f"Nov/Dec eval rows {nd.height} (matched {(nd['nm_unmatched'] == 0).sum()})")
    report_path = OUT / "legs" / (a.report_name or ("predict_report.json" if a.suffix == "novdec" else f"predict_report_{a.suffix}.json"))
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    train_cache: dict[str, pl.DataFrame] = {}
    jj_cache: dict[str, pl.DataFrame] = {}

    def train_frame(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
        key = f"{path}|{','.join(columns) if columns else '*'}"
        if key not in train_cache:
            train_cache.clear()
            train_cache[key] = pl.read_parquet(path, columns=columns)
        return train_cache[key]

    def jj_frame(path: Path) -> pl.DataFrame:
        if str(path) not in jj_cache:
            jj_cache.clear()
            jj_cache[str(path)] = pl.read_parquet(path)
        return jj_cache[str(path)]

    for leg in legs:
        t0 = time.perf_counter()
        run_name, kind, model_file = LEGS[leg]
        run = RUNS / run_name
        man = json.loads((run / "run_manifest.json").read_text())
        target = target_of(man)
        tpath = train_matrix_for(run)
        if kind == "catboost":
            model = CatBoostLeg(run, model_file)
        elif kind in ("lgbm", "lgbm_apt"):
            levels = ma.category_levels(train_frame(tpath, cats), cats)
            stored_levels = man.get("categorical_levels")
            if stored_levels and {c: len(v) for c, v in levels.items()} != stored_levels:
                raise RuntimeError(f"{leg}: category level counts differ from the run manifest")
            model = LgbmLeg(run, model_file, levels) if kind == "lgbm" else PerAirportLgbmLeg(run, model_file, levels)
        else:
            keep = list(dict.fromkeys([s4.LABEL_COLUMN, "month", *target.columns(), *cats, *nums]))
            model = MlpLeg(run, model_file, train_frame(tpath, keep), cats, nums, target)
        if a.swap_only:
            # JAN_JUL rows with the final-D-pool ADS-B block (as the Nov/Dec rows get it); nothing else
            alt = RUNS / "dfit_full_adsb_delta_historder" / "eval_matrix.parquet"
            altf = jj_frame(alt)
            raw_alt = target.predict(altf, model.t_hat(altf, cats, nums))
            np.save(OUT / "legs" / f"{leg}_janjul_swap_raw.npy", raw_alt)
            s4.atomic_json(OUT / "legs" / f"{leg}_janjul_swap.json", {"leg": leg, "matrix": str(alt), "matrix_sha256": s4.sha256_file(alt),
                                                                       "ids_sha256": s4.json_sha256(altf["MVT_ID_mvt"].to_list()), "rows": altf.height})
            log(f"{leg}: JAN_JUL swap prediction written")
            del model
            continue
        raw_nd = target.predict(nd, model.t_hat(nd, cats, nums))
        out = compose(nd, raw_nd)
        entry: dict[str, Any] = {
            "run": run_name,
            "kind": kind,
            "model_sha256": (model.pa.model_hashes(run / "per_airport", model_file, s4.sha256_file) if kind == "lgbm_apt"
                             else {f: s4.sha256_file(run / f) for f in ([model_file] if kind != "mlp" else [model_file.format(s=s) for s in model.seeds])}),
            "target": target.as_dict(),
            "train_matrix_sha256": s4.sha256_file(tpath),
            "novdec_eval_matrix": {"path": str(nd_run / "eval_matrix.parquet"), "sha256": s4.sha256_file(nd_run / "eval_matrix.parquet")},
        }
        if not a.no_jj_check:
            jpath = eval_matrix_for(run)
            jj = jj_frame(jpath)
            if kind in ("lgbm", "lgbm_apt") and a.jj_sample and a.jj_sample < jj.height:
                jj = jj.sample(n=a.jj_sample, seed=20260927)
                entry["janjul_reproduction_subset_rows"] = a.jj_sample
            raw_jj = target.predict(jj, model.t_hat(jj, cats, nums))
            entry["janjul_reproduction"] = jj_check(leg, run, raw_jj, jj["MVT_ID_mvt"].to_numpy(), with_rmse=not a.jj_repro_only)
            entry["janjul_eval_matrix_sha256"] = s4.sha256_file(jpath)
            if kind == "mlp":
                seeds = model.seed_t_hats(jj)
                entry["janjul_seed_t_hat_vs_stored"] = {
                    str(s): float(np.max(np.abs(seeds[i] - np.load(run / f"eval_that_seed{s}.npy")))) for i, s in enumerate(model.seeds)
                }
            if a.swap_check and kind in a.swap_kinds.split(","):
                alt = RUNS / "dfit_full_adsb_delta_historder" / "eval_matrix.parquet"
                altf = pl.read_parquet(alt)
                if not np.array_equal(altf["MVT_ID_mvt"].to_numpy(), jj["MVT_ID_mvt"].to_numpy()):
                    raise RuntimeError("swap matrix row order differs")
                raw_alt = target.predict(altf, model.t_hat(altf, cats, nums))
                y = jj[s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
                m = jj["nm_unmatched"].to_numpy() == 0
                entry["adsb_pool_swap_janjul"] = {
                    "note": "JAN_JUL raw matched RMSE with the eval rows' ADS-B block from the interim D pool (as fitted) vs the final D pool (as the Nov/Dec rows)",
                    "own_matrix": str(jpath),
                    "swap_matrix": str(alt),
                    "matched_rmse_own": float(np.sqrt(np.mean((y[m] - raw_jj[m]) ** 2))),
                    "matched_rmse_swap": float(np.sqrt(np.mean((y[m] - raw_alt[m]) ** 2))),
                    "mean_abs_diff": float(np.mean(np.abs(raw_alt[m] - raw_jj[m]))),
                }
                np.save(OUT / "legs" / f"{leg}_janjul_swap_raw.npy", raw_alt)
                del altf
            np.save(OUT / "legs" / f"{leg}_janjul_repred_raw.npy", raw_jj)
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        info = write_leg(out, leg, entry, a.suffix)
        report[leg] = info
        s4.atomic_json(report_path, report)
        rep = entry.get("janjul_reproduction", {})
        log(f"{leg}: wrote {info['path']} sha {info['sha256'][:12]}; JJ max|d| {rep.get('max_abs_diff')} r {rep.get('pearson_r')} swap {entry.get('adsb_pool_swap_janjul', {}).get('matched_rmse_swap')}")
        del model


if __name__ == "__main__":
    main()
