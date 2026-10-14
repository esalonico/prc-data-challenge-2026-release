# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""D-only unmatched subgroup forensics, buckets, and hierarchical conditional means.

January/July labels never enter group definitions, support cutoffs, or shrinkage.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from prc_challenge.campaign.runtime import FORBIDDEN_FEATURE_COLS, LOCKED_MONTHS
from prc_challenge.campaign.scoring import assert_no_forbidden_features, assert_no_locked_months
from prc_challenge.features import CAT_FILL, add_flight_prefix
from prc_challenge.unmatched_recovery.schema import (
    D_MONTHS,
    NM_CATEGORICALS,
    NM_PLANNING_RAW,
    PRIMARY_MONTHS,
    REGIME_EDGES,
    present,
)

COARSE_REGIMES = ("ordinary", "long", "extreme")
FINE_BANDS = (
    ("R0", None, 1800.0),
    ("R1", 1800.0, 3600.0),
    ("R2", 3600.0, 7200.0),
    ("R3", 7200.0, 21600.0),
    ("R4", 21600.0, 43200.0),
    ("R5", 43200.0, 86400.0),
    ("R6", 86400.0, None),
)
DELAY_BUCKETS = ("early", "d0_30m", "d30_60m", "d1_2h", "d2_6h", "d6h_plus", "extra_day", "unknown")
HOUR_BUCKETS = (0, 4, 8, 12, 16, 20)
ALPHA_GRID = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
STABLE_MIN_ROWS = 100
STABLE_MIN_DATES = 30
STABLE_MIN_MONTHS = 3
MISSINGNESS_NM_FIELDS = NM_CATEGORICALS + NM_PLANNING_RAW

HIERARCHIES: tuple[tuple[str, ...], ...] = (
    ("AIRPORT",),
    ("AIRPORT", "type_missing"),
    ("AIRPORT", "ADES_mvt"),
    ("AIRPORT", "flight_prefix"),
    ("AIRPORT", "delay_bucket"),
    ("AIRPORT", "hour_bucket"),
    ("AIRPORT", "type_missing", "delay_bucket"),
    ("AIRPORT", "takeoff_minus_sched_days"),
    ("AIRPORT", "runway_missing"),
    ("AIRPORT", "stand_missing"),
    ("type_missing", "delay_bucket"),
    ("flight_prefix", "ADES_mvt"),
    ("flight_family", "AIRPORT"),
    ("flight_family", "ADES_mvt"),
)


def assert_design_months(months: Sequence[int]) -> None:
    """Refuse January/July or locked V4 months in any design/fit universe."""

    blocked = set(PRIMARY_MONTHS) | set(LOCKED_MONTHS)
    hit = [int(m) for m in months if int(m) in blocked]
    if hit:
        raise RuntimeError(f"design months include primary/locked months: {sorted(set(hit))}")
    extra = [int(m) for m in months if int(m) not in D_MONTHS]
    if extra:
        raise RuntimeError(f"design months outside D: {sorted(set(extra))}")


def shrinkage_weight(n: float, alpha: float) -> float:
    n = float(n)
    alpha = float(alpha)
    if n < 0 or alpha < 0:
        raise ValueError("n and alpha must be nonnegative")
    denom = n + alpha
    if denom <= 0:
        return 0.0
    return n / denom


def empirical_bayes_mean(group_mean: float, parent_mean: float, n: float, alpha: float) -> float:
    w = shrinkage_weight(n, alpha)
    return float(w * group_mean + (1.0 - w) * parent_mean)


def normalize_flight_mvt(expr: pl.Expr | None = None) -> pl.Expr:
    col = expr if expr is not None else pl.col("FLIGHT_mvt")
    return (
        col.cast(pl.String)
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9]", "")
        .fill_null(CAT_FILL)
        .alias("flight_norm")
    )


def flight_family_token(expr: pl.Expr | None = None) -> pl.Expr:
    """Deterministic family: 2–3 letter prefix plus numeric body without leading zeros."""

    norm = expr if expr is not None else pl.col("FLIGHT_mvt").cast(pl.String).str.to_uppercase().str.replace_all(
        r"[^A-Z0-9]", ""
    )
    prefix = norm.str.extract(r"^([A-Z]{2,3})", 1)
    numeric = norm.str.extract(r"^[A-Z]{2,3}0*([0-9]+)$", 1)
    family = pl.when(prefix.is_not_null() & numeric.is_not_null()).then(pl.concat_str([prefix, numeric]))
    family = pl.when(family.is_not_null()).then(family).otherwise(prefix.fill_null(norm.fill_null(CAT_FILL)))
    return family.fill_null(CAT_FILL).alias("flight_family")


def delay_bucket_expr() -> pl.Expr:
    days = pl.col("takeoff_minus_sched_days") if True else pl.lit(None)
    sec = pl.col("takeoff_minus_sched_sec")
    return (
        pl.when(days.abs() >= 1)
        .then(pl.lit("extra_day"))
        .when(sec.is_null())
        .then(pl.lit("unknown"))
        .when(sec < 0)
        .then(pl.lit("early"))
        .when(sec < 1800)
        .then(pl.lit("d0_30m"))
        .when(sec < 3600)
        .then(pl.lit("d30_60m"))
        .when(sec < 7200)
        .then(pl.lit("d1_2h"))
        .when(sec < 21600)
        .then(pl.lit("d2_6h"))
        .otherwise(pl.lit("d6h_plus"))
        .alias("delay_bucket")
    )


def hour_bucket_expr() -> pl.Expr:
    return ((pl.col("local_hour").cast(pl.Int32).fill_null(0) // 4) * 4).clip(0, 20).alias("hour_bucket")


def coarse_regime_expr(col: str = "TAXITIME_SEC_mvt") -> pl.Expr:
    y = pl.col(col)
    return (
        pl.when(y < REGIME_EDGES[0])
        .then(pl.lit("ordinary"))
        .when(y < REGIME_EDGES[1])
        .then(pl.lit("long"))
        .otherwise(pl.lit("extreme"))
        .alias("coarse_regime")
    )


def fine_regime_expr(col: str = "TAXITIME_SEC_mvt") -> pl.Expr:
    y = pl.col(col)
    expr = pl.lit("R6")
    for name, lo, hi in reversed(FINE_BANDS):
        if lo is None:
            expr = pl.when(y < hi).then(pl.lit(name)).otherwise(expr)
        elif hi is None:
            expr = pl.when(y >= lo).then(pl.lit(name)).otherwise(expr)
        else:
            expr = pl.when((y >= lo) & (y < hi)).then(pl.lit(name)).otherwise(expr)
    return expr.alias("fine_regime")


def missingness_taxonomy_expr(columns: Sequence[str] | None = None) -> pl.Expr:
    present_cols = set(columns or MISSINGNESS_NM_FIELDS)
    nm_null: pl.Expr | None = None
    for col in MISSINGNESS_NM_FIELDS:
        piece = (
            pl.col(col).is_null() | pl.col(col).cast(pl.String).is_in(["", CAT_FILL, "None"])
            if col in present_cols
            else pl.lit(True)
        )
        nm_null = piece if nm_null is None else (nm_null & piece)
    if nm_null is None:
        nm_null = pl.lit(True)
    extra_day = (
        pl.col("takeoff_minus_sched_days").abs() >= 1
        if "takeoff_minus_sched_days" in present_cols
        else pl.lit(False)
    )
    return (
        pl.when(extra_day)
        .then(pl.lit("schedule_extra_day"))
        .when(nm_null)
        .then(pl.lit("missing_link_movement_present"))
        .otherwise(pl.lit("other_input_regime"))
        .alias("missingness_taxonomy")
    )


def attach_identity_columns(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    if "flight_prefix" not in out.columns and "FLIGHT_mvt" in out.columns:
        out = add_flight_prefix(out)
    extras: list[pl.Expr] = []
    if "FLIGHT_mvt" in out.columns:
        extras.append(normalize_flight_mvt())
        extras.append(flight_family_token())
    if extras:
        out = out.with_columns(extras)
    buckets: list[pl.Expr] = []
    if "takeoff_minus_sched_sec" in out.columns:
        if "takeoff_minus_sched_days" not in out.columns:
            out = out.with_columns((pl.col("takeoff_minus_sched_sec") / 86400.0).floor().alias("takeoff_minus_sched_days"))
        buckets.append(delay_bucket_expr())
    if "local_hour" in out.columns:
        buckets.append(hour_bucket_expr())
    if buckets:
        out = out.with_columns(buckets)
    flag_exprs: list[pl.Expr] = []
    for src, dest in (
        ("AIRCRAFT_TYPE_mvt", "type_missing"),
        ("RUNWAY_mvt", "runway_missing"),
        ("STAND_mvt", "stand_missing"),
        ("ADES_mvt", "ades_missing"),
    ):
        if dest not in out.columns and src in out.columns:
            flag_exprs.append(pl.col(src).is_null().cast(pl.Int8).alias(dest))
    if flag_exprs:
        out = out.with_columns(flag_exprs)
    out = out.with_columns(missingness_taxonomy_expr(out.columns))
    if "event_date" not in out.columns:
        time_col = "MVT_TIME_UTC_mvt" if "MVT_TIME_UTC_mvt" in out.columns else (
            "SCHED_TIME_UTC_mvt" if "SCHED_TIME_UTC_mvt" in out.columns else None
        )
        if time_col is not None:
            out = out.with_columns(pl.col(time_col).cast(pl.Datetime(time_zone="UTC")).dt.date().alias("event_date"))
    return out


def attach_target_regimes(frame: pl.DataFrame, col: str = "TAXITIME_SEC_mvt") -> pl.DataFrame:
    if col not in frame.columns:
        return frame
    return frame.with_columns(coarse_regime_expr(col), fine_regime_expr(col))


def bimodality_coefficient(values: np.ndarray) -> float | None:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 20:
        return None
    centered = x - np.mean(x)
    sd = float(np.std(centered, ddof=1))
    if sd < 1e-12:
        return 0.0
    z = centered / sd
    g = float(np.mean(z**3))
    excess = float(np.mean(z**4) - 3.0)
    kurtosis = excess + 3.0
    denom = kurtosis + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    if abs(denom) < 1e-12:
        return None
    return float((g * g + 1.0) / denom)


def histogram_peak_count(values: np.ndarray, bins: int = 24) -> int:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return 0
    counts, _ = np.histogram(x, bins=bins)
    peaks = 0
    for i in range(1, len(counts) - 1):
        if counts[i] >= counts[i - 1] and counts[i] >= counts[i + 1] and counts[i] > 0:
            peaks += 1
    return int(peaks)


def _quantile_map(x: np.ndarray) -> dict[str, float]:
    qs = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    return {f"p{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs}


def summarize_target(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    logx = np.log1p(np.clip(x, 0, None))
    out: dict[str, Any] = {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "sse_to_mean": float(np.sum((x - np.mean(x)) ** 2)),
        "bimodality_coefficient": bimodality_coefficient(x),
        "log_bimodality_coefficient": bimodality_coefficient(logx),
        "histogram_peaks": histogram_peak_count(x),
        "log_histogram_peaks": histogram_peak_count(logx),
        "bimodality_threshold": 5.0 / 9.0,
    }
    out.update(_quantile_map(x))
    for name, lo, hi in (
        ("ordinary", None, REGIME_EDGES[0]),
        ("long", REGIME_EDGES[0], REGIME_EDGES[1]),
        ("extreme", REGIME_EDGES[1], None),
    ):
        mask = np.ones(x.size, dtype=bool)
        if lo is not None:
            mask &= x >= lo
        if hi is not None:
            mask &= x < hi
        out[f"n_{name}"] = int(mask.sum())
        out[f"rate_{name}"] = float(mask.mean())
    for name, lo, hi in FINE_BANDS:
        mask = np.ones(x.size, dtype=bool)
        if lo is not None:
            mask &= x >= lo
        if hi is not None:
            mask &= x < hi
        out[f"n_{name}"] = int(mask.sum())
        out[f"rate_{name}"] = float(mask.mean())
    return out


def target_distribution(frame: pl.DataFrame, col: str = "TAXITIME_SEC_mvt") -> dict[str, Any]:
    assert_no_locked_months(frame)
    y = frame[col].to_numpy().astype(np.float64)
    payload: dict[str, Any] = {"overall": summarize_target(y)}
    if "month" in frame.columns:
        payload["by_month"] = {}
        for month in sorted(frame["month"].unique().to_list()):
            part = frame.filter(pl.col("month") == month)
            payload["by_month"][str(int(month))] = summarize_target(part[col].to_numpy())
    hist, edges = np.histogram(y[np.isfinite(y)], bins=40)
    payload["histogram"] = {"counts": hist.tolist(), "edges": edges.tolist()}
    logy = np.log1p(np.clip(y[np.isfinite(y)], 0, None))
    lhist, ledges = np.histogram(logy, bins=40)
    payload["log_histogram"] = {"counts": lhist.tolist(), "edges": ledges.tolist()}
    xs = np.sort(y[np.isfinite(y)])
    if xs.size:
        cdf_p = np.linspace(0, 1, min(201, xs.size))
        payload["empirical_cdf"] = {
            "p": cdf_p.tolist(),
            "y": np.quantile(xs, cdf_p).tolist(),
        }
    return payload


def global_centers(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("empty target vector")
    lo, hi = np.quantile(x, [0.10, 0.90])
    trimmed = x[(x >= lo) & (x <= hi)]
    winsor = np.clip(x, lo, hi)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "trimmed_mean_10": float(np.mean(trimmed) if trimmed.size else np.mean(x)),
        "winsorized_mean_10": float(np.mean(winsor)),
        "n": float(x.size),
    }


def group_stat_table(
    train: pl.DataFrame,
    keys: Sequence[str],
    *,
    y_col: str = "TAXITIME_SEC_mvt",
) -> pl.DataFrame:
    use = [k for k in keys if k in train.columns]
    if not use:
        return train.clear()
    labeled = train.filter(pl.col(y_col).is_not_null())
    date_col = "event_date" if "event_date" in labeled.columns else None
    aggs = [
        pl.len().alias("n"),
        pl.col(y_col).mean().alias("mean"),
        pl.col(y_col).median().alias("median"),
        pl.col(y_col).std().alias("std"),
        pl.col("month").n_unique().alias("n_months") if "month" in labeled.columns else pl.lit(0).alias("n_months"),
        (pl.col(y_col) < REGIME_EDGES[0]).mean().alias("ordinary_rate"),
        ((pl.col(y_col) >= REGIME_EDGES[0]) & (pl.col(y_col) < REGIME_EDGES[1])).mean().alias("long_rate"),
        (pl.col(y_col) >= REGIME_EDGES[1]).mean().alias("extreme_rate"),
        ((pl.col(y_col) - pl.col(y_col).mean()) ** 2).sum().alias("target_sse"),
    ]
    if date_col:
        aggs.append(pl.col(date_col).n_unique().alias("n_dates"))
    for name, lo, hi in FINE_BANDS:
        mask = pl.col(y_col) >= (lo if lo is not None else -1e18)
        if hi is not None:
            mask = mask & (pl.col(y_col) < hi)
        aggs.append(mask.mean().alias(f"rate_{name}"))
    return labeled.group_by(list(use)).agg(aggs)


def is_stable_group(n: int, n_dates: int, n_months: int) -> bool:
    return n >= STABLE_MIN_ROWS and n_dates >= STABLE_MIN_DATES and n_months >= STABLE_MIN_MONTHS


def lookup_shrunk_mean(
    frame: pl.DataFrame,
    table: pl.DataFrame,
    keys: Sequence[str],
    *,
    parent_mean: float,
    alpha: float,
    stat: str = "mean",
) -> np.ndarray:
    use = [k for k in keys if k in frame.columns and k in table.columns]
    if not use or table.height == 0:
        return np.full(frame.height, parent_mean, dtype=np.float64)
    joined = frame.join(table.select([*use, stat, "n"]), on=list(use), how="left")
    n = joined["n"].fill_null(0).to_numpy().astype(np.float64)
    raw = joined[stat].to_numpy().astype(np.float64)
    out = np.empty(frame.height, dtype=np.float64)
    for i in range(frame.height):
        if not np.isfinite(raw[i]) or n[i] <= 0:
            out[i] = parent_mean
        else:
            out[i] = empirical_bayes_mean(float(raw[i]), parent_mean, float(n[i]), alpha)
    return out


def hierarchical_predict(
    frame: pl.DataFrame,
    tables: Sequence[tuple[tuple[str, ...], pl.DataFrame]],
    *,
    parent_mean: float,
    alpha: float,
    stat: str = "mean",
) -> np.ndarray:
    pred = np.full(frame.height, parent_mean, dtype=np.float64)
    filled = np.zeros(frame.height, dtype=bool)
    for keys, table in tables:
        if table.height == 0:
            continue
        use = [k for k in keys if k in frame.columns and k in table.columns]
        if not use:
            continue
        joined = frame.with_row_index("_row").join(table.select([*use, stat, "n"]), on=list(use), how="left")
        n = joined["n"].to_numpy()
        raw = joined[stat].to_numpy()
        rows = joined["_row"].to_numpy()
        for i, row in enumerate(rows):
            if filled[row]:
                continue
            if n[i] is None or not np.isfinite(float(n[i] or 0)) or float(n[i]) <= 0:
                continue
            if raw[i] is None or not np.isfinite(float(raw[i])):
                continue
            pred[row] = empirical_bayes_mean(float(raw[i]), parent_mean, float(n[i]), alpha)
            filled[row] = True
    return np.clip(pred, 0, None)


def fallback_parent_chain(
    frame: pl.DataFrame,
    train: pl.DataFrame,
    key_chain: Sequence[Sequence[str]],
    *,
    alpha: float,
    y_col: str = "TAXITIME_SEC_mvt",
    stat: str = "mean",
) -> np.ndarray:
    assert_design_months(sorted(set(int(m) for m in train["month"].to_list()))) if "month" in train.columns else None
    parent = float(train[y_col].mean())
    tables = []
    for keys in key_chain:
        tables.append((tuple(keys), group_stat_table(train, keys, y_col=y_col)))
    return hierarchical_predict(frame, tables, parent_mean=parent, alpha=alpha, stat=stat)


def subgroup_metrics(
    frame: pl.DataFrame,
    keys: Sequence[str],
    *,
    pred_col: str | None = "u6_prediction",
    y_col: str = "TAXITIME_SEC_mvt",
) -> pl.DataFrame:
    use = [k for k in keys if k in frame.columns]
    if not use:
        return frame.clear()
    y = pl.col(y_col)
    aggs = [
        pl.len().alias("n"),
        y.mean().alias("mean_taxi"),
        y.median().alias("median_taxi"),
        y.std().alias("std_taxi"),
        y.var().alias("var_taxi"),
        pl.col("month").n_unique().alias("n_months") if "month" in frame.columns else pl.lit(0).alias("n_months"),
        (y < REGIME_EDGES[0]).mean().alias("ordinary_rate"),
        ((y >= REGIME_EDGES[0]) & (y < REGIME_EDGES[1])).mean().alias("long_rate"),
        (y >= REGIME_EDGES[1]).mean().alias("extreme_rate"),
        ((y - y.mean()) ** 2).sum().alias("target_sse"),
    ]
    if "event_date" in frame.columns:
        aggs.append(pl.col("event_date").n_unique().alias("n_dates"))
    else:
        aggs.append(pl.lit(0).alias("n_dates"))
    for name, lo, hi in FINE_BANDS:
        mask = y >= (lo if lo is not None else -1e18)
        if hi is not None:
            mask = mask & (y < hi)
        aggs.append(mask.mean().alias(f"rate_{name}"))
    if pred_col and pred_col in frame.columns:
        se = (y - pl.col(pred_col)) ** 2
        aggs.extend(
            [
                se.sum().alias("model_sse"),
                se.mean().sqrt().alias("model_rmse"),
                (pl.col(pred_col) - y).mean().alias("mean_residual"),
            ]
        )
    table = frame.group_by(list(use)).agg(aggs).with_columns(
        pl.lit("|".join(use)).alias("group_keys"),
        pl.concat_str([pl.col(k).cast(pl.String).fill_null("null") for k in use], separator="|").alias("group_value"),
    )
    total_sse = float(table["model_sse"].sum()) if "model_sse" in table.columns else float(table["target_sse"].sum())
    sse_col = "model_sse" if "model_sse" in table.columns else "target_sse"
    global_ext = float((frame[y_col] >= REGIME_EDGES[1]).mean()) if frame.height else 0.0
    return table.with_columns(
        (pl.col(sse_col) / max(total_sse, 1e-12)).alias("sse_share"),
        (pl.col("extreme_rate") / max(global_ext, 1e-12)).alias("extreme_rate_lift"),
        (
            (pl.col("n") >= STABLE_MIN_ROWS)
            & (pl.col("n_dates") >= STABLE_MIN_DATES)
            & (pl.col("n_months") >= STABLE_MIN_MONTHS)
        ).alias("stable"),
    )


def attach_forensic_columns(frame: pl.DataFrame, u6_pred: np.ndarray | None = None) -> pl.DataFrame:
    out = attach_target_regimes(attach_identity_columns(frame))
    if u6_pred is not None:
        pred = np.asarray(u6_pred, dtype=np.float64)
        if pred.shape[0] != out.height:
            raise RuntimeError("U6 prediction length does not match forensic frame")
        out = out.with_columns(pl.Series("u6_prediction", pred))
        if "TAXITIME_SEC_mvt" in out.columns:
            out = out.with_columns(
                (pl.col("TAXITIME_SEC_mvt") - pl.col("u6_prediction")).alias("u6_residual"),
                (pl.col("TAXITIME_SEC_mvt") - pl.col("u6_prediction")).abs().alias("u6_abs_residual"),
                ((pl.col("TAXITIME_SEC_mvt") - pl.col("u6_prediction")) ** 2).alias("u6_squared_error"),
            )
    if {"TAXITIME_SEC_mvt", "takeoff_minus_sched_sec"}.issubset(out.columns):
        delay = pl.col("takeoff_minus_sched_sec")
        y = pl.col("TAXITIME_SEC_mvt")
        out = out.with_columns(
            (y - delay).abs().alias("abs_taxi_minus_delay"),
            (y - (delay + 86400.0)).abs().alias("abs_taxi_minus_delay_plus_day"),
            (y - (delay - 86400.0).abs()).abs().alias("abs_taxi_minus_abs_delay_minus_day"),
            (y % 86400.0).alias("taxi_mod_day"),
            (delay.abs() % 86400.0).alias("delay_mod_day"),
        )
    leaked = FORBIDDEN_FEATURE_COLS.intersection(
        [c for c in out.columns if c not in {"TAXITIME_SEC_mvt", "u6_residual", "u6_abs_residual", "u6_squared_error"}]
    )
    # Target/residual columns are diagnostics, not model features.
    _ = leaked
    return out


def compact_score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y, dtype=np.float64)
    hat = np.asarray(pred, dtype=np.float64)
    err = hat - truth
    sse = float(np.dot(err, err))
    n = int(truth.size)
    payload = {
        "n": n,
        "sse": sse,
        "rmse_seconds": float(np.sqrt(sse / n)) if n else float("nan"),
        "mae_seconds": float(np.mean(np.abs(err))) if n else float("nan"),
        "bias_seconds": float(np.mean(err)) if n else float("nan"),
    }
    for name, lo, hi in (
        ("ordinary", None, REGIME_EDGES[0]),
        ("long", REGIME_EDGES[0], REGIME_EDGES[1]),
        ("extreme", REGIME_EDGES[1], None),
    ):
        mask = np.ones(n, dtype=bool)
        if lo is not None:
            mask &= truth >= lo
        if hi is not None:
            mask &= truth < hi
        if not mask.any():
            continue
        se = err[mask]
        payload[f"{name}_n"] = int(mask.sum())
        payload[f"{name}_sse"] = float(np.dot(se, se))
        payload[f"{name}_rmse"] = float(np.sqrt(np.dot(se, se) / mask.sum()))
    return payload


def assert_feature_names(names: Sequence[str]) -> None:
    assert_no_forbidden_features(list(names))
