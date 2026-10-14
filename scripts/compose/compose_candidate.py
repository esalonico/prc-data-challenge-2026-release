# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Compose a full ranking candidate from frozen components (orchestrator, 2026-09-25).

Order (the deployed system since v19/v20):
  1. matched rows   <- ``--matched`` (e.g. an S4 all-year run's ``prediction``: raw S4 + E18 + clip)
  2. unmatched rows <- ``--unmatched`` (a composed submission: A3b-nodest + E17/E18 + clip)
  3. optional U2    <- ``--u2-target`` rows: p*max(sd,0) + (1-p)*(0.5*expert + 0.5*parent),
                       parent = step-1 matched prediction, p=0 where a copy is infeasible
  4. NM matching-window projection on matched rows (``--window-mode hull`` = IOBT/EOBT_1 hull;
     ``lobt`` = LOBT +/- 3606, tighter; allowed since owner ruling D-017)
  5. clip at 0
Writes the candidate, a manifest with SHA-256, per-stage row counts and diffs against a reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from prc_challenge.lirf_label_source import copy_mixture
from prc_challenge.matching_window import project_predictions
from prc_challenge.submission import validate_submission

ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("PRC_SOURCE_ROOT"):  # the checkout holding data/raw (a worktree or the bake VM); default: this checkout
    ROOT = Path(os.environ["PRC_SOURCE_ROOT"])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", type=Path, required=True, help="parquet with MVT_ID_mvt + prediction column")
    ap.add_argument("--matched-col", default="prediction")
    ap.add_argument("--unmatched", type=Path, required=True, help="submission parquet whose unmatched rows are kept")
    ap.add_argument("--u2-target", type=Path, default=None)
    ap.add_argument("--no-projection", action="store_true")
    ap.add_argument("--window-mode", default="hull", choices=("hull", "lobt"), help="matching-window fields (D-017 allows lobt)")
    ap.add_argument("--reference", type=Path, default=None, help="submission to diff against (e.g. current best)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    template = pl.read_parquet(ROOT / "data" / "raw" / "submitting.parquet").select("MVT_ID_mvt")
    raw = (
        pl.scan_parquet(ROOT / "data" / "raw" / "ranking.parquet")
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select("MVT_ID_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt")
        .collect()
    )
    matched = pl.read_parquet(args.matched).select("MVT_ID_mvt", pl.col(args.matched_col).cast(pl.Float64).alias("m"))
    unmatched = pl.read_parquet(args.unmatched).select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("u"))
    df = template.join(raw, on="MVT_ID_mvt", how="left", maintain_order="left")
    df = df.join(matched, on="MVT_ID_mvt", how="left", maintain_order="left").join(unmatched, on="MVT_ID_mvt", how="left", maintain_order="left")
    is_matched = df["FLIGHT_ID_mvt"].is_not_null()
    if df.filter(is_matched & pl.col("m").is_null()).height:
        raise SystemExit("matched prediction missing on matched rows")
    if df.filter(~is_matched & pl.col("u").is_null()).height:
        raise SystemExit("unmatched prediction missing on unmatched rows")
    df = df.with_columns(pl.when(is_matched).then(pl.col("m")).otherwise(pl.col("u")).alias("prediction"))
    stages = {"matched_rows": int(is_matched.sum()), "unmatched_rows": int((~is_matched).sum())}
    if args.u2_target:
        tgt = pl.read_parquet(args.u2_target)
        df = df.join(tgt, on="MVT_ID_mvt", how="left", maintain_order="left")
        on = df["p_copy"].is_not_null() & is_matched
        parent = df["prediction"].to_numpy()
        mix = copy_mixture(
            df["p_copy"].fill_null(0.0).to_numpy(),
            df["sd"].fill_null(np.nan).to_numpy(),
            0.5 * df["expert"].fill_null(0.0).to_numpy() + 0.5 * parent,
            df["feasible"].fill_null(False).to_numpy(),
        )
        df = df.with_columns(pl.Series("prediction", np.where(on.to_numpy(), mix, parent)))
        stages["u2_rows"] = int(on.sum())
    if not args.no_projection:
        before = df["prediction"].to_numpy().copy()
        df = project_predictions(df, "prediction", args.window_mode)
        stages["projection_moved"] = int((np.abs(df["prediction"].to_numpy() - before) > 1e-9).sum())
    df = df.with_columns(pl.col("prediction").clip(lower_bound=0.0))
    cand = df.select("MVT_ID_mvt", pl.col("prediction").alias("TAXITIME_SEC_mvt"))
    errors = validate_submission(cand, pl.read_parquet(ROOT / "data" / "raw" / "submitting.parquet"))
    if errors:
        raise SystemExit(f"validator: {errors}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(args.out)
    man = {
        "out": str(args.out),
        "sha256": _sha(args.out),
        "inputs": {
            "matched": [str(args.matched), _sha(args.matched)],
            "unmatched": [str(args.unmatched), _sha(args.unmatched)],
            "u2_target": [str(args.u2_target), _sha(args.u2_target)] if args.u2_target else None,
        },
        "order": ["matched", "unmatched", "u2" if args.u2_target else None, None if args.no_projection else f"{args.window_mode}_projection", "clip0"],
        "stages": stages,
        "pred_mean": float(cand["TAXITIME_SEC_mvt"].mean()),
        "pred_max": float(cand["TAXITIME_SEC_mvt"].max()),
    }
    if args.reference:
        ref = pl.read_parquet(args.reference).rename({"TAXITIME_SEC_mvt": "ref"})
        d = cand.join(ref, on="MVT_ID_mvt").join(raw.select("MVT_ID_mvt", "FLIGHT_ID_mvt"), on="MVT_ID_mvt")
        diff = (d["TAXITIME_SEC_mvt"] - d["ref"]).to_numpy()
        m = d["FLIGHT_ID_mvt"].is_not_null().to_numpy()
        man["vs_reference"] = {
            "reference": str(args.reference),
            "rows_changed": int((np.abs(diff) > 1e-9).sum()),
            "matched_changed": int((np.abs(diff[m]) > 1e-9).sum()),
            "unmatched_changed": int((np.abs(diff[~m]) > 1e-9).sum()),
            "mean_abs_diff": float(np.abs(diff).mean()),
            "max_abs_diff": float(np.abs(diff).max()),
            "corr": float(np.corrcoef(d["TAXITIME_SEC_mvt"].to_numpy(), d["ref"].to_numpy())[0, 1]),
        }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
