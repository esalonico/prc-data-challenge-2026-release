# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Predict-only serving of stored all-year legs on any assembled eval matrix (final phase, 4 months).

Loads each leg's saved model (CatBoost ``model.cbm``, LightGBM ``model_fixed_rounds.txt``, MLP
``model_stageB_seed{s}.pt``), predicts the rows of ``<eval-run>/eval_matrix.parquet``, inverts the leg's
fit target exactly as its trainer does (``s4.TargetSpec.predict``; LightGBM categorical levels and the MLP
preprocessor are refitted on ``<train-run>/train_matrix.parquet``, which must be the matrix the leg was
fitted on), composes the ranking side (``s4.compose_ranking_matched``: E17 -> E18 -> clip >= 0) and writes
``<out-dir>/ranking_<leg>.parquet`` with ``matrix_arms.RANKING_COLUMNS``. Nothing is fitted.

``--check KEY=PATH`` compares a leg's raw prediction with a stored ranking file (regression gate on the
old Jan/Jul inputs: the stored all-year legs must be reproduced).

Run GBM legs and MLP legs in separate invocations: torch deadlocks in libomp after CatBoost/LightGBM ran in
the same process (macOS, observed 2026-10-11).

Usage::

  PYTHONPATH=src python scripts/legs/predict_legs.py --train-run <runs>/X --eval-run <runs>/Y \
      --out-dir <dir> --legs cb8k,cbd8k,cbe8k,lgb,lgbd,lgbe,lgbl,nn14,nn15,nn16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("V4_THREADS", "8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import leg_models as v4  # noqa: E402  (CatBoostLeg, LgbmLeg, MlpLeg, target_of)
from prc_challenge import matrix_arms as ma  # noqa: E402
from prc_challenge import s4_allyear as s4  # noqa: E402

RUNS_REL = Path("artifacts/experiments/S4-allyear-rebuild-20260925/runs")

# leg key -> (run dir under RUNS, kind, model file)
LEGS: dict[str, tuple[str, str, str]] = {
    "cb": ("allyear_iso_v3l_cb_plain", "catboost", "model.cbm"),
    "cbd": ("allyear_iso_v3l_cb_delta", "catboost", "model.cbm"),
    "cbe": ("allyear_iso_v3l_cb_eobtres", "catboost", "model.cbm"),
    "cb8k": ("allyear_iso_v3l_cb8k_plain", "catboost", "model.cbm"),
    "cbd8k": ("allyear_iso_v3l_cb8k_delta", "catboost", "model.cbm"),
    "cbe8k": ("allyear_iso_v3l_cb8k_eobtres", "catboost", "model.cbm"),
    "lgb": ("allyear_iso_v3l_lgbm_plain", "lgbm", "model_fixed_rounds.txt"),
    "lgbd": ("allyear_iso_v3l_lgbm_delta", "lgbm", "model_fixed_rounds.txt"),
    "lgbe": ("allyear_iso_v3l_lgbm_eobtres", "lgbm", "model_fixed_rounds.txt"),
    "lgbl": ("allyear_iso_v3l_lgbm_lobtoff", "lgbm", "model_fixed_rounds.txt"),
    # LAST_TRY (D-115): the four LightGBM legs at lr 0.01, rounds early-stopped on October with the D months
    "lgb01": ("allyear_iso_v3l_lgbm_plain_lr01", "lgbm", "model_fixed_rounds.txt"),
    "lgbd01": ("allyear_iso_v3l_lgbm_delta_lr01", "lgbm", "model_fixed_rounds.txt"),
    "lgbe01": ("allyear_iso_v3l_lgbm_eobtres_lr01", "lgbm", "model_fixed_rounds.txt"),
    "lgbl01": ("allyear_iso_v3l_lgbm_lobtoff_lr01", "lgbm", "model_fixed_rounds.txt"),
    "nn14": ("nn_v14_iso_v3l_allyear", "mlp", "model_stageB_seed{s}.pt"),
    "nn15": ("nn_v15_iso_v3l_allyear", "mlp", "model_stageB_seed{s}.pt"),
    "nn16": ("nn_v16_iso_v3l_allyear", "mlp", "model_stageB_seed{s}.pt"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg} (peak_rss={s4.peak_rss_gb():.2f} GB)", flush=True)


def leg_target(run: Path) -> s4.TargetSpec:
    for name in ("run_manifest.json", "fit_info.json"):
        p = run / name
        if p.exists():
            man = json.loads(p.read_text())
            if "target" in man or "recipe" in man or "fit" in man:
                return v4.target_of(man)
    raise RuntimeError(f"{run}: no manifest with a target")


def compose(evaluation: pl.DataFrame, raw: np.ndarray) -> pl.DataFrame:
    out = s4.compose_ranking_matched(evaluation, raw)
    if "period" not in out.columns:
        out = out.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("period"))
    return out.select(*ma.RANKING_COLUMNS)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", default=os.environ.get("PRC_SOURCE_ROOT", str(HERE.parents[1])))
    p.add_argument("--train-run", required=True, help="run dir holding train_matrix.parquet + matrix_identity.json")
    p.add_argument("--eval-run", required=True, help="run dir holding the eval_matrix.parquet to predict")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--legs", required=True)
    p.add_argument("--check", action="append", default=[], help="KEY=PATH stored ranking leg to compare with")
    a = p.parse_args(argv)
    root = Path(a.source_root)
    runs = root / RUNS_REL
    train_run, eval_run, out = Path(a.train_run), Path(a.eval_run), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cats, nums = ma.feature_lists_for_run(train_run)
    if (cats, nums) != ma.feature_lists_for_run(eval_run):
        raise SystemExit("train and eval matrices have different feature lists")
    ev = pl.read_parquet(eval_run / "eval_matrix.parquet")
    tpath = train_run / "train_matrix.parquet"
    log(f"eval rows {ev.height} (matched {(ev['nm_unmatched'] == 0).sum()}); months {sorted(ev['month'].unique().to_list())}")
    checks = dict(c.split("=", 1) for c in a.check)
    report_path = out / "predict_serve_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    train_cache: dict[str, pl.DataFrame] = {}

    def train_frame(columns: list[str]) -> pl.DataFrame:
        key = ",".join(columns)
        if key not in train_cache:
            train_cache.clear()
            train_cache[key] = pl.read_parquet(tpath, columns=columns)
        return train_cache[key]

    for leg in [x for x in a.legs.split(",") if x]:
        t0 = time.perf_counter()
        run_name, kind, model_file = LEGS[leg]
        run = runs / run_name
        target = leg_target(run)
        if kind == "catboost":
            model = v4.CatBoostLeg(run, model_file)
        elif kind == "lgbm":
            levels = ma.category_levels(train_frame(cats), cats)
            stored = json.loads((run / "run_manifest.json").read_text()).get("categorical_levels")
            if stored and {c: len(v) for c, v in levels.items()} != stored:
                raise RuntimeError(f"{leg}: category level counts differ from the run manifest")
            model = v4.LgbmLeg(run, model_file, levels)
        else:
            keep = list(dict.fromkeys([s4.LABEL_COLUMN, "month", *target.columns(), *cats, *nums]))
            model = v4.MlpLeg(run, model_file, train_frame(keep), cats, nums, target)
        raw = target.predict(ev, model.t_hat(ev, cats, nums))
        frame = compose(ev, raw)
        path = out / f"ranking_{leg}.parquet"
        s4.atomic_parquet(frame, path)
        entry = {"leg": leg, "run": run_name, "kind": kind, "target": target.as_dict(), "rows": frame.height,
                 "path": str(path), "sha256": s4.sha256_file(path),
                 "eval_matrix_sha256": s4.sha256_file(eval_run / "eval_matrix.parquet"),
                 "train_matrix_sha256": s4.sha256_file(tpath), "seconds": round(time.perf_counter() - t0, 1)}
        if leg in checks:
            stored = pl.read_parquet(checks[leg]).select("MVT_ID_mvt", pl.col("s4_raw_prediction").alias("stored"))
            j = stored.join(frame.select("MVT_ID_mvt", "s4_raw_prediction"), on="MVT_ID_mvt", how="inner", validate="1:1")
            x, y = j["s4_raw_prediction"].to_numpy(), j["stored"].to_numpy()
            entry["check"] = {"stored": checks[leg], "n": j.height, "max_abs_diff": float(np.max(np.abs(x - y))),
                              "mean_abs_diff": float(np.mean(np.abs(x - y))), "n_bit_identical": int((x == y).sum()),
                              "pearson_r": float(np.corrcoef(x, y)[0, 1])}
        report[leg] = entry
        s4.atomic_json(report_path, report)
        log(f"{leg}: {frame.height} rows -> {path.name}; check {entry.get('check')}")
        del model


if __name__ == "__main__":
    main()
