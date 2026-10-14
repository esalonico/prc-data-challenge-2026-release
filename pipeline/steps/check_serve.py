# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Checks for the four-month serve matrix (JOINED_PLAN §5.3).

1. parity : standardised mean shift and null-rate change per numeric column, month pair by month pair
            (2025-01 vs 2026-01, 2025-02 vs 2026-02, 2025-06 vs 2026-06, 2025-07 vs 2026-07).
2. dither : Jan/Jul 2026 rows of the new serve matrix vs the shipped Jan/Jul ranking matrix (same MVT_IDs):
            per column share of rows changed, mean |diff|, correlation.
3. coverage: ADS-B coverage class share per airport x month (2025 train months vs the four 2026 months).

Usage:
  python check_serve.py --train <train_matrix.parquet> --serve <serve_matrix.parquet> [--shipped <old ranking matrix>]
                        --out <dir> [--coverage-col adsb_coverage]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

PAIRS = [(1, 1), (2, 2), (6, 6), (7, 7)]


def month_col(df: pl.DataFrame) -> pl.Series:
    if "month" in df.columns:
        return df["month"]
    return df["MVT_TIME_UTC_mvt"].dt.month()


def numeric_cols(df: pl.DataFrame) -> list[str]:
    skip = {"MVT_ID_mvt", "month", "label", "TAXITIME_SEC_mvt"}
    return [c for c, t in df.schema.items() if c not in skip and t.is_numeric()]


def parity(train: pl.DataFrame, serve: pl.DataFrame, cols: list[str]) -> dict:
    tm, sm = month_col(train), month_col(serve)
    out = {}
    for a, b in PAIRS:
        A, B = train.filter(tm == a), serve.filter(sm == b)
        if A.height == 0 or B.height == 0:
            continue
        rows = []
        for c in cols:
            if c not in B.columns:
                rows.append({"col": c, "missing_in_serve": True}); continue
            x = A[c].cast(pl.Float64); y = B[c].cast(pl.Float64)
            xs, ys = x.drop_nulls().drop_nans(), y.drop_nulls().drop_nans()
            sd = float(xs.std()) if xs.len() > 1 else float("nan")
            shift = (float(ys.mean()) - float(xs.mean())) / sd if sd and sd > 0 and ys.len() else float("nan")
            rows.append({"col": c, "shift_sd": round(shift, 3),
                         "null_2025": round(float(x.is_null().mean() + x.is_nan().fill_null(False).mean()), 4),
                         "null_2026": round(float(y.is_null().mean() + y.is_nan().fill_null(False).mean()), 4)})
        df = pl.DataFrame(rows)
        if "shift_sd" in df.columns:
            df = df.with_columns((pl.col("null_2026") - pl.col("null_2025")).alias("null_delta"))
            flag = df.filter((pl.col("shift_sd").abs() > 0.25) | (pl.col("null_delta").abs() > 0.01))
        else:
            flag = df
        out[f"2025-{a:02d}_vs_2026-{b:02d}"] = {"n_2025": A.height, "n_2026": B.height, "n_cols": len(cols),
                                               "flagged": flag.sort("col").to_dicts()}
    return out


def dither(serve: pl.DataFrame, shipped: pl.DataFrame, cols: list[str]) -> dict:
    j = shipped.select(["MVT_ID_mvt", *[c for c in cols if c in shipped.columns]]).join(
        serve.select(["MVT_ID_mvt", *[c for c in cols if c in shipped.columns]]), on="MVT_ID_mvt", suffix="__new")
    res = {"rows_joined": j.height, "rows_shipped": shipped.height}
    rows = []
    for c in cols:
        if c not in shipped.columns or c + "__new" not in j.columns:
            continue
        a = j[c].cast(pl.Float64).to_numpy(); b = j[c + "__new"].cast(pl.Float64).to_numpy()
        both = ~(np.isnan(a) | np.isnan(b))
        changed = float(np.mean(np.where(both, np.abs(a - b) > 1e-6, np.isnan(a) != np.isnan(b))))
        corr = float(np.corrcoef(a[both], b[both])[0, 1]) if both.sum() > 10 and np.std(a[both]) > 0 else float("nan")
        rows.append({"col": c, "share_changed": round(changed, 4), "mean_abs_diff": round(float(np.nanmean(np.abs(a - b))), 3),
                     "corr": round(corr, 5)})
    df = pl.DataFrame(rows).sort("share_changed", descending=True)
    res["most_changed"] = df.head(40).to_dicts()
    res["n_cols_changed_gt_5pct"] = int(df.filter(pl.col("share_changed") > 0.05).height)
    res["n_cols_corr_lt_0p99"] = int(df.filter(pl.col("corr") < 0.99).height)
    return res


def coverage(train: pl.DataFrame, serve: pl.DataFrame, col: str) -> dict:
    out = {}
    for name, df in [("2025", train), ("2026", serve)]:
        if col not in df.columns:
            out[name] = f"no column {col}"; continue
        g = (df.with_columns(month_col(df).alias("_m"))
               .group_by(["AIRPORT", "_m", col]).len()
               .with_columns((pl.col("len") / pl.col("len").sum().over(["AIRPORT", "_m"])).round(3).alias("share"))
               .sort(["AIRPORT", "_m", col]))
        out[name] = g.to_dicts()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--serve", required=True)
    ap.add_argument("--shipped"); ap.add_argument("--out", required=True)
    ap.add_argument("--coverage-col", default="adsb_coverage")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train = pl.read_parquet(a.train); serve = pl.read_parquet(a.serve)
    cols = sorted(set(numeric_cols(train)) & set(numeric_cols(serve)))
    R = {"serve_rows": serve.height, "serve_months": month_col(serve).value_counts().sort("month" if "month" in serve.columns else month_col(serve).name).to_dicts(),
         "only_in_train": sorted(set(train.columns) - set(serve.columns)), "only_in_serve": sorted(set(serve.columns) - set(train.columns)),
         "parity": parity(train, serve, cols)}
    if a.shipped:
        R["dither"] = dither(serve, pl.read_parquet(a.shipped), cols)
    if "AIRPORT" in train.columns:
        R["coverage"] = coverage(train, serve, a.coverage_col)
    (out / "check_serve.json").write_text(json.dumps(R, indent=1, default=str))
    for k, v in R["parity"].items():
        print(k, "flagged:", len(v["flagged"]))
    if "dither" in R:
        print("dither: cols changed >5%:", R["dither"]["n_cols_changed_gt_5pct"], "| corr<0.99:", R["dither"]["n_cols_corr_lt_0p99"])


if __name__ == "__main__":
    main()
