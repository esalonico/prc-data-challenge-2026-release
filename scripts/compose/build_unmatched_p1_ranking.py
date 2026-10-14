# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""P1 ranking overlay: 0.5 * A3b_raw + 0.5 * T on the ranking LIRF-unmatched rows. Never uploads.

Recipe (fixed by the blocked-D gates and the single January/July confirmation of
Unmatched-polish-20260925; lambda = 0.5 is not a parameter):

1. Gate-delay model: U3 "stand" variant (U6 CatBoost params, 400 trees, target
   G = takeoff - SCHED - TAXITIME) fitted on every LIRF 2025 months 1-10 departure,
   matched and unmatched. Features: ``run_lirf_label_source._u3_frame`` (E08b cache +
   raw LIRF DEP/ARR features incl. same-stand inbound, month-isolated). Months 11-12
   stay locked (``load_movements`` refuses them).
2. T = max(sd - G_hat, 0) on the ranking LIRF-unmatched rows. Features from the
   S4-allyear ranking base caches (identical schema to the E08b cache, label-free) +
   ranking LIRF raw DEP/ARR rows with departure labels blanked, per ranking month.
3. Blend, before E17/E18: ``prediction = 0.5 * raw_R-A3b-nodest + 0.5 * T`` (rows with
   a missing T keep the A3b value). ``raw_R-A3b-nodest`` is the v18 pre-composition
   mixture from Unmatched-noncopy-20260925/ranking/ranking_unmatched_predictions.parquet.

Outputs (artifacts/experiments/Unmatched-polish-20260925/ranking/):

- ``p1_unmatched_overlay.parquet``   MVT_ID_mvt + ``prediction`` (blended, before E17/E18)
  for the LIRF-unmatched ranking rows only, plus ``prediction_composed`` (E17/E18 + clip),
  the A3b raw/v18 values, T and G_hat for inspection.
- ``p1_v18_unmatched_submission.parquet``   v18 with those rows replaced by
  ``prediction_composed`` (full submission shape): a ready ``--unmatched`` input for
  ``scripts/compose/compose_candidate.py``. Matched rows are byte-identical to v18/v14.
- ``p1_unmatched_overlay.manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("run_lirf_label_source", ROOT / "scripts" / "compose" / "run_lirf_label_source.py")
R = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(R)

from prc_challenge.io import reporting_airport_expr, scan_submitting  # noqa: E402
from prc_challenge.lirf_label_source import serve_gate_delay  # noqa: E402
from prc_challenge.unmatched_recovery.harness import RANKING_N, RANKING_UNMATCHED_N, overlay_unmatched_submission, verify_arm_submission  # noqa: E402
from prc_challenge.unmatched_recovery.noncopy_expert import delay_bucket, distribution  # noqa: E402
from prc_challenge.unmatched_recovery.polish import LAMBDA, blend_on_mask  # noqa: E402
from prc_challenge.unmatched_recovery.schema import LOCKED_MONTHS, compose_experimental_unmatched  # noqa: E402

MAIN = R.MAIN
POLISH = MAIN / "artifacts" / "experiments" / "Unmatched-polish-20260925"
OUT = POLISH / "ranking"
TRACKED = ROOT / "progress-review" / "unmatched-polish-20260925" / "ranking"
S4 = MAIN / "artifacts" / "experiments" / "S4-allyear-rebuild-20260925" / "features"
NONCOPY = MAIN / "artifacts" / "experiments" / "Unmatched-noncopy-20260925" / "ranking"
V18 = MAIN / "submissions" / "unbelievable-snowflake_v18.parquet"
RANKING_RAW = MAIN / "data" / "raw" / "ranking.parquet"
V14_MATCHED_CHECKSUM = "1bf0c6dee84454f4858421d09ad36c719d2e5f032fa1013066225ccc303e3031"
FIT_MONTHS = tuple(range(1, 11))
RANK_PERIODS = ("2026-01", "2026-07")
LABEL_COLS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt")
GATE_COLS = ("MVT_ID_mvt", "AIRPORT", "nm_unmatched", "type_missing", "takeoff_minus_sched_days", "takeoff_minus_sched_sec")
MIN_FREE_MEMORY_PCT = 25.0


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_memory(context: str) -> float | None:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"free percentage:\s*(\d+)%", out)
    free = float(match.group(1)) if match else None
    if free is not None and free < MIN_FREE_MEMORY_PCT:
        raise RuntimeError(f"{context}: only {free:.0f}% memory free; refusing to fit now")
    _log(f"{context}: memory free {free}%")
    return free


def _ranking_lirf(phase: str) -> pl.DataFrame:
    """Ranking LIRF rows of one phase with departure labels blanked (as build_lirf_u2_ranking.py)."""

    frame = (
        pl.scan_parquet(RANKING_RAW)
        .filter(pl.col("PHASE_mvt") == phase)
        .with_columns(reporting_airport_expr(), pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8).alias("month"))
        .filter(pl.col("AIRPORT") == "LIRF")
        .collect()
    )
    drop = [c for c in LABEL_COLS if c in frame.columns and not (phase == "ARR" and c == "BLOCK_TIME_UTC_mvt")]
    if phase == "DEP" and frame["BLOCK_TIME_UTC_mvt"].null_count() != frame.height:
        raise RuntimeError("ranking DEP BLOCK is not empty")
    return frame.drop(drop)


def training_frame() -> pl.DataFrame:
    frame = R._u3_frame(FIT_MONTHS)
    months = sorted(int(m) for m in frame["month"].unique().to_list())
    if months != list(FIT_MONTHS) or set(months) & set(LOCKED_MONTHS):
        raise RuntimeError(f"training months {months}")
    return frame


def ranking_frame() -> pl.DataFrame:
    base = pl.concat([pl.read_parquet(S4 / "base" / f"base_{p}.parquet") for p in RANK_PERIODS], how="vertical_relaxed").filter(pl.col("AIRPORT") == "LIRF")
    bad = [c for c in LABEL_COLS if c in base.columns]
    if bad:
        raise RuntimeError(f"ranking base carries {bad}")
    dep = _ranking_lirf("DEP")
    arr = _ranking_lirf("ARR")
    raw = R.lirf_raw_features(dep, arr)
    frame = base.join(raw, on="MVT_ID_mvt", how="left")
    if frame.height != base.height or frame["flight_prefix"].null_count():
        raise RuntimeError("ranking LIRF raw join missed rows")
    frame = frame.with_columns(
        pl.col("RUNWAY_mvt").is_null().cast(pl.Int8).alias("runway_missing"),
        pl.col("STAND_mvt").is_null().cast(pl.Int8).alias("stand_missing"),
    )
    return frame.filter(pl.col("nm_unmatched") == 1)


def _aligned(base: pl.DataFrame, other: pl.DataFrame, col: str) -> np.ndarray:
    joined = base.select("MVT_ID_mvt").join(other.select("MVT_ID_mvt", col), on="MVT_ID_mvt", how="left", maintain_order="left")
    if joined.height != base.height or joined[col].null_count():
        raise RuntimeError(f"{col}: alignment failed")
    return joined[col].cast(pl.Float64).to_numpy()


def _t_reference(name: str, path: Path, mask_col: str | None) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    frame = pl.read_parquet(path)
    if mask_col is not None:
        frame = frame.filter(pl.col(mask_col) == "LIRF")
    t = frame["T"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    return {"available": True, "source": str(path), **distribution(t[np.isfinite(t)])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    TRACKED.mkdir(parents=True, exist_ok=True)
    _require_memory("p1 overlay")

    # 1) Gate-delay model on all LIRF months 1-10 departures.
    train = training_frame()
    nums = R.U3_VARIANTS["stand"]
    _log(f"gate-delay fit: {train.height} LIRF departures, months {list(FIT_MONTHS)}, {len(R.U6_CATS) + len(nums)} features, {R.G_PARAMS['iterations']} trees")
    model, n_fit = R._fit_g(train, nums)
    model_path = OUT / "models" / "p1_g_stand_m1_10.cbm"
    model.save_model(str(model_path))
    n_train_unmatched = int((train["nm_unmatched"] == 1).sum())
    del train

    # 2) T on the ranking LIRF-unmatched rows.
    rank_u = ranking_frame()
    ref = pl.read_parquet(NONCOPY / "frames" / "ranking_unmatched.parquet").filter(pl.col("AIRPORT") == "LIRF")
    if set(ref["MVT_ID_mvt"].to_list()) != set(rank_u["MVT_ID_mvt"].to_list()):
        raise RuntimeError(f"ranking LIRF unmatched rows differ: {rank_u.height} vs {ref.height} in the v18 build")
    rank_u = ref.select("MVT_ID_mvt").join(rank_u, on="MVT_ID_mvt", how="left", maintain_order="left")
    g_hat = np.asarray(model.predict(R._pool(rank_u, list(R.U6_CATS), nums)), dtype=np.float64)
    sd = rank_u["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    t = serve_gate_delay(sd, g_hat)
    _log(f"T on {rank_u.height} ranking LIRF-unmatched rows (Jan {int((rank_u['month'] == 1).sum())}, Jul {int((rank_u['month'] == 7).sum())}); T missing {int(np.isnan(t).sum())}")

    # 3) Blend with the v18 pre-composition mixture; compose; identity checks against v18.
    preds = pl.read_parquet(NONCOPY / "ranking_unmatched_predictions.parquet")
    a3b_raw = _aligned(rank_u, preds, "raw_R-A3b-nodest")
    a3b_deployed = _aligned(rank_u, preds, "R-A3b-nodest")
    sd_ref = _aligned(rank_u, preds, "takeoff_minus_sched_sec")
    if float(np.nanmax(np.abs(sd - sd_ref))) > 1e-6:
        raise RuntimeError("takeoff - SCHED differs between the ranking feature paths")
    gate = ref.select(list(GATE_COLS))

    def deployed(pred: np.ndarray) -> np.ndarray:
        return compose_experimental_unmatched(gate.with_columns(pl.Series("prediction", np.asarray(pred, dtype=np.float64))))["prediction"].to_numpy()

    v18 = pl.read_parquet(V18)
    v18_rows = _aligned(rank_u, v18, "TAXITIME_SEC_mvt")
    identity = {
        "compose_a3b_raw_vs_R-A3b-nodest": float(np.max(np.abs(deployed(a3b_raw) - a3b_deployed))),
        "R-A3b-nodest_vs_v18": float(np.max(np.abs(a3b_deployed - v18_rows))),
    }
    if max(identity.values()) > 1e-6:
        raise RuntimeError(f"v18 composition does not reproduce on the LIRF rows: {identity}")
    blended = blend_on_mask(a3b_raw, t, np.ones(rank_u.height, dtype=bool), LAMBDA)
    composed = deployed(blended)
    override = np.abs(composed - np.clip(blended, 0.0, None)) > 1e-9
    overlay = rank_u.select("MVT_ID_mvt", "month", "AIRPORT", "takeoff_minus_sched_sec", "type_missing", "takeoff_minus_sched_days").with_columns(
        pl.Series("prediction", blended),
        pl.Series("prediction_composed", composed),
        pl.Series("a3b_raw", a3b_raw),
        pl.Series("v18", v18_rows),
        pl.Series("T", t),
        pl.Series("g_hat", g_hat),
        pl.Series("e17_e18_override", override),
    )
    overlay_path = OUT / "p1_unmatched_overlay.parquet"
    overlay.write_parquet(overlay_path)

    # 4) v18 with the LIRF-unmatched rows replaced (ready --unmatched input for compose_candidate.py).
    all_unmatched_ids = pl.read_parquet(NONCOPY / "frames" / "ranking_unmatched.parquet")["MVT_ID_mvt"]
    submission = overlay_unmatched_submission(v18, overlay.select("MVT_ID_mvt", pl.col("prediction_composed").alias("prediction")))
    template = scan_submitting(MAIN / "data" / "raw").collect()
    verify = verify_arm_submission(submission, template=template, parent=v18, unmatched_ids=all_unmatched_ids, expected_unmatched_n=RANKING_UNMATCHED_N, expected_n=RANKING_N)
    if not verify["ok"] or verify["matched_checksum"] != V14_MATCHED_CHECKSUM:
        raise RuntimeError(f"submission verification failed: {verify['errors']} / matched checksum {verify['matched_checksum']}")
    sub_path = OUT / "p1_v18_unmatched_submission.parquet"
    submission.write_parquet(sub_path)
    change = composed - v18_rows
    changed = np.abs(change) > 1e-9
    bucket = delay_bucket(sd)

    manifest = {
        "name": "P1 unmatched overlay (gate-delay blend, lambda 0.5) on ranking LIRF-unmatched rows",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {
            "overlay": {"path": str(overlay_path), "sha256": _sha256(overlay_path), "rows": int(overlay.height), "columns": overlay.columns, "prediction_is": "blended, before E17/E18 and clip"},
            "v18_plus_p1_unmatched_submission": {"path": str(sub_path), "sha256": _sha256(sub_path), "rows": int(submission.height), "use": "--unmatched input of scripts/compose/compose_candidate.py (rows already E17/E18 + clip composed)"},
        },
        "lambda": LAMBDA,
        "rows": {"ranking_lirf_unmatched": int(rank_u.height), "jan": int((rank_u["month"] == 1).sum()), "jul": int((rank_u["month"] == 7).sum()), "changed_vs_v18": int(changed.sum()), "e17_e18_overrides": int(override.sum()), "t_missing": int(np.isnan(t).sum())},
        "model": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "variant": "stand",
            "params": {k: v for k, v in R.G_PARAMS.items()},
            "target": "G = takeoff_minus_sched_sec - TAXITIME_SEC_mvt on all LIRF departures (matched + unmatched)",
            "fit_months": list(FIT_MONTHS),
            "n_fit": int(n_fit),
            "n_fit_unmatched": n_train_unmatched,
            "cats": list(R.U6_CATS),
            "nums": list(nums),
            "confirmation_model_d_fit": str(POLISH / "models" / "confirm_g_stand_D.cbm"),
        },
        "ranking_features": {
            "base": [str(S4 / "base" / f"base_{p}.parquet") for p in RANK_PERIODS],
            "raw": "ranking.parquet LIRF DEP/ARR rows, departure labels blanked; daily ARR copy rate and same-stand inbound per ranking month (month-isolated)",
        },
        "a3b_source": {"path": str(NONCOPY / "ranking_unmatched_predictions.parquet"), "column": "raw_R-A3b-nodest", "identity_checks_max_abs_diff": identity},
        "v18": {"path": str(V18), "sha256": _sha256(V18)},
        "validator": {"ok": verify["ok"], "errors": verify["errors"], "matched_checksum": verify["matched_checksum"], "matched_checksum_equals_v14": verify["matched_checksum"] == V14_MATCHED_CHECKSUM, "matched_changed": verify["matched_changed"], "unmatched_checksum": verify["unmatched_checksum"]},
        "change_vs_v18": {
            "mean": float(change.mean()),
            "mean_abs": float(np.abs(change).mean()),
            "max_abs": float(np.abs(change).max()),
            "sum_sq": float(np.sum(change**2)),
            "by_delay_bucket": {b: {"n": int((bucket == b).sum()), "mean_change": float(change[bucket == b].mean()) if (bucket == b).any() else None, "mean_v18": float(v18_rows[bucket == b].mean()) if (bucket == b).any() else None, "mean_T": float(np.nanmean(t[bucket == b])) if (bucket == b).any() else None} for b in ("<1h", "1-2h", "2-4h", ">=4h", "missing")},
        },
        "t_distribution": {
            "ranking": distribution(t[np.isfinite(t)]),
            "janjul_2025_confirmation": _t_reference("janjul", POLISH / "confirm" / "janjul_unmatched_components.parquet", "AIRPORT"),
            "d_oof": _t_reference("D", POLISH / "oof" / "arms_D.parquet", "AIRPORT"),
            "a3b_raw_ranking": distribution(a3b_raw),
        },
        "composition_note": "compose_candidate.py order: matched -> unmatched (this submission's unmatched rows) -> U2 -> hull projection -> clip. Blending before or after E17/E18 is equivalent (the gates override the same rows either way).",
        "forbidden_check": {"aobt": False, "lobt": False, "arvt3": False, "dep_block_or_taxi_as_feature": False, "v4_months": False, "adsb": False, "opensky": False, "opdi": False},
        "uploaded": False,
    }
    (OUT / "p1_unmatched_overlay.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (TRACKED / "p1_unmatched_overlay.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _log(f"overlay: {overlay.height} rows, {int(changed.sum())} changed vs v18, mean change {change.mean():+.1f} s, max |change| {np.abs(change).max():.0f} s; matched checksum == v14: {verify['matched_checksum'] == V14_MATCHED_CHECKSUM}")


if __name__ == "__main__":
    main()
