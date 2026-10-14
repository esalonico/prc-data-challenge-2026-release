# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Native completed-arrival state with fold-safe taxi-in normalisation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from prc_challenge.io import CHALLENGE_AIRPORTS

NATIVE_STATE_FEATURES = (
    "native_arr_available",
    "native_month_boundary",
    "native_arr_age_sec",
    "native_arr_taxi_in_sec",
    "native_arr_taxi_in_expected_sec",
    "native_arr_taxi_in_excess_sec",
    "native_same_stand_available",
    "native_same_stand_age_sec",
    "native_same_stand_taxi_in_excess_sec",
    "native_same_runway_available",
    "native_same_runway_age_sec",
    "native_same_runway_taxi_in_excess_sec",
    "native_same_runway_recent_excess_sec",
)


def _runway_region(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.to_uppercase()
        .str.strip_chars()
        .str.replace(r"[LCR]$", "")
        .replace("", None)
    )


def _stand_region(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.to_uppercase()
        .str.strip_chars()
        .str.extract(r"^([A-Z]+)", 1)
    )


def _scan_source(source: str | Path | pl.LazyFrame) -> pl.LazyFrame:
    """Accept a raw parquet path or an already month-restricted lazy frame."""

    if isinstance(source, pl.LazyFrame):
        return source
    return pl.scan_parquet(source)


def load_completed_arrivals(
    month_paths: dict[int, str | Path | pl.LazyFrame],
) -> pl.DataFrame:
    """Load completed ARR observations only, retaining their source month."""

    parts: list[pl.DataFrame] = []
    for month, path in sorted(month_paths.items()):
        part = (
            _scan_source(path)
            .filter(
                (pl.col("PHASE_mvt") == "ARR")
                & pl.col("ADES_mvt").is_in(list(CHALLENGE_AIRPORTS))
                & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()
            )
            .select(
                pl.col("MVT_ID_mvt").alias("native_arr_mvt_id"),
                pl.col("ADES_mvt").str.to_uppercase().alias("native_arr_airport"),
                pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias(
                    "native_arr_mvt_epoch"
                ),
                pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias(
                    "native_arr_inblock_epoch"
                ),
                pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(
                    "native_arr_taxi_in_sec"
                ),
                pl.col("RUNWAY_mvt").alias("native_arr_runway"),
                pl.col("STAND_mvt").alias("native_arr_stand"),
                pl.col("ADEP_mvt").str.to_uppercase().alias("native_arr_origin"),
                pl.lit(month).cast(pl.Int8).alias("native_arr_month"),
            )
            .with_columns(
                _runway_region("native_arr_runway").alias("native_arr_runway_region"),
                _stand_region("native_arr_stand").alias("native_arr_stand_region"),
            )
            .filter(
                (pl.col("native_arr_inblock_epoch") >= pl.col("native_arr_mvt_epoch"))
                & pl.col("native_arr_taxi_in_sec").is_between(0, 7200)
            )
            .collect()
        )
        parts.append(part)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="vertical_relaxed")


def _strict_previous_asof(
    departures: pl.DataFrame,
    arrivals: pl.DataFrame,
    *,
    left_by: list[str],
    right_by: list[str],
    prefix: str,
) -> pl.DataFrame:
    right_time = f"{prefix}inblock_epoch"
    rename = {
        column: f"{prefix}{column.removeprefix('native_arr_')}"
        for column in arrivals.columns
    }
    right = arrivals.rename(rename).sort([*right_by, right_time])
    left = departures.with_columns(
        (pl.col("native_dep_epoch") - 1).alias("_native_query_epoch")
    ).sort([*left_by, "_native_query_epoch"])
    return left.join_asof(
        right,
        left_on="_native_query_epoch",
        right_on=right_time,
        by_left=left_by,
        by_right=right_by,
        strategy="backward",
    ).drop("_native_query_epoch")


def build_native_completed_arrival_sidecar(
    raw_path: str | Path | pl.LazyFrame, *, month: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build one native-state row per departure within an isolated month.

    Arrival selection uses strict completed in-block chronology. No departure
    taxi label, departure block time, AOBT, prediction, or residual is read.
    ``raw_path`` may be a lazy frame already restricted to one isolated month
    (the ranking file holds two disjoint months).
    """

    departures = (
        _scan_source(raw_path)
        .filter(
            (pl.col("PHASE_mvt") == "DEP")
            & pl.col("ADEP_mvt").is_in(list(CHALLENGE_AIRPORTS))
        )
        .select(
            "MVT_ID_mvt",
            pl.col("ADEP_mvt").str.to_uppercase().alias("native_airport"),
            pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias(
                "native_dep_epoch"
            ),
            pl.col("RUNWAY_mvt").alias("native_current_runway"),
            pl.col("STAND_mvt").alias("native_current_stand"),
            pl.lit(month).cast(pl.Int8).alias("month"),
        )
        .with_columns(
            _runway_region("native_current_runway").alias(
                "native_current_runway_region"
            ),
            _stand_region("native_current_stand").alias("native_current_stand_region"),
        )
        .collect()
    )
    arrivals = load_completed_arrivals({month: raw_path})
    base = _strict_previous_asof(
        departures,
        arrivals,
        left_by=["native_airport"],
        right_by=["native_last_airport"],
        prefix="native_last_",
    )

    same_stand_arr = arrivals.filter(pl.col("native_arr_stand").is_not_null())
    base = _strict_previous_asof(
        base,
        same_stand_arr,
        left_by=["native_airport", "native_current_stand"],
        right_by=["native_same_stand_airport", "native_same_stand_stand"],
        prefix="native_same_stand_",
    )

    same_runway_arr = arrivals.filter(
        pl.col("native_arr_runway_region").is_not_null()
    )
    base = _strict_previous_asof(
        base,
        same_runway_arr,
        left_by=["native_airport", "native_current_runway_region"],
        right_by=["native_same_runway_airport", "native_same_runway_runway_region"],
        prefix="native_same_runway_",
    )

    out = base.with_columns(
        # Polars omits right-side as-of grouping keys. Reconstruct the matched
        # keys explicitly so downstream baseline ancestry remains auditable.
        pl.col("native_airport").alias("native_last_airport"),
        pl.col("native_airport").alias("native_same_stand_airport"),
        pl.col("native_current_stand").alias("native_same_stand_stand"),
        pl.col("native_airport").alias("native_same_runway_airport"),
        pl.col("native_current_runway_region").alias(
            "native_same_runway_runway_region"
        ),
        pl.col("native_last_inblock_epoch").is_not_null().cast(pl.Int8).alias(
            "native_arr_available"
        ),
        pl.col("native_last_inblock_epoch").is_null().cast(pl.Int8).alias(
            "native_month_boundary"
        ),
        (pl.col("native_dep_epoch") - pl.col("native_last_inblock_epoch")).alias(
            "native_arr_age_sec"
        ),
        pl.col("native_last_taxi_in_sec").alias("native_arr_taxi_in_sec"),
        pl.col("native_same_stand_inblock_epoch")
        .is_not_null()
        .cast(pl.Int8)
        .alias("native_same_stand_available"),
        (
            pl.col("native_dep_epoch") - pl.col("native_same_stand_inblock_epoch")
        ).alias("native_same_stand_age_sec"),
        pl.col("native_same_runway_inblock_epoch")
        .is_not_null()
        .cast(pl.Int8)
        .alias("native_same_runway_available"),
        (
            pl.col("native_dep_epoch") - pl.col("native_same_runway_inblock_epoch")
        ).alias("native_same_runway_age_sec"),
    )
    if out.height != departures.height or out["MVT_ID_mvt"].n_unique() != out.height:
        raise RuntimeError("native completed-arrival sidecar lost or duplicated departures")
    return out, arrivals


def _fit_arrival_baseline(arrivals: pl.DataFrame) -> dict[str, object]:
    if arrivals.is_empty():
        raise ValueError("cannot fit arrival baseline on no rows")
    global_mean = float(arrivals["native_arr_taxi_in_sec"].mean())
    tables: list[tuple[list[str], pl.DataFrame, float]] = []
    specs = (
        (["native_arr_airport", "native_arr_runway_region", "native_arr_stand_region"], 30.0),
        (["native_arr_airport", "native_arr_runway_region"], 50.0),
        (["native_arr_airport"], 100.0),
    )
    for keys, shrinkage in specs:
        table = arrivals.group_by(keys).agg(
            pl.col("native_arr_taxi_in_sec").mean().alias("_mean"),
            pl.len().alias("_n"),
        ).with_columns(
            (
                (pl.col("_mean") * pl.col("_n") + global_mean * shrinkage)
                / (pl.col("_n") + shrinkage)
            ).alias("_estimate")
        )
        tables.append((keys, table, shrinkage))
    return {"global_mean": global_mean, "tables": tables}


def _predict_arrival_baseline(
    frame: pl.DataFrame,
    fitted: dict[str, object],
    *,
    prefix: str,
    output: str,
) -> pl.DataFrame:
    out = frame
    candidates: list[str] = []
    for index, (keys, table, _) in enumerate(fitted["tables"]):
        left_keys = [f"{prefix}{key.removeprefix('native_arr_')}" for key in keys]
        candidate = f"_{output}_{index}"
        candidates.append(candidate)
        out = out.join(
            table.select([*keys, pl.col("_estimate").alias(candidate)]),
            left_on=left_keys,
            right_on=keys,
            how="left",
        )
    out = out.with_columns(
        pl.coalesce(*[pl.col(c) for c in candidates], pl.lit(fitted["global_mean"]))
        .cast(pl.Float64)
        .alias(output)
    )
    return out.drop(candidates)


def _attach_for_fit_months(
    frame: pl.DataFrame, arrivals: pl.DataFrame, fit_months: Iterable[int]
) -> pl.DataFrame:
    fitted = _fit_arrival_baseline(
        arrivals.filter(pl.col("native_arr_month").is_in(list(fit_months)))
    )
    out = _predict_arrival_baseline(
        frame, fitted, prefix="native_last_", output="native_arr_taxi_in_expected_sec"
    )
    out = _predict_arrival_baseline(
        out,
        fitted,
        prefix="native_same_stand_",
        output="_native_same_stand_expected",
    )
    out = _predict_arrival_baseline(
        out,
        fitted,
        prefix="native_same_runway_",
        output="_native_same_runway_expected",
    )
    return out.with_columns(
        (pl.col("native_last_taxi_in_sec") - pl.col("native_arr_taxi_in_expected_sec")).alias(
            "native_arr_taxi_in_excess_sec"
        ),
        (
            pl.col("native_same_stand_taxi_in_sec")
            - pl.col("_native_same_stand_expected")
        ).alias("native_same_stand_taxi_in_excess_sec"),
        (
            pl.col("native_same_runway_taxi_in_sec")
            - pl.col("_native_same_runway_expected")
        ).alias("native_same_runway_taxi_in_excess_sec"),
    ).with_columns(
        (
            pl.col("native_same_runway_taxi_in_excess_sec")
            / (1.0 + pl.col("native_same_runway_age_sec") / 1800.0)
        ).alias("native_same_runway_recent_excess_sec")
    ).drop("_native_same_stand_expected", "_native_same_runway_expected")


def attach_cross_fitted_arrival_state(
    train: pl.DataFrame,
    val: pl.DataFrame,
    arrivals: pl.DataFrame,
    *,
    train_months: Iterable[int],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cross-fit ARR baselines by month for train and fit them on D for val."""

    months = sorted(int(m) for m in train_months)
    train_parts: list[pl.DataFrame] = []
    for holdout in months:
        part = train.filter(pl.col("month") == holdout)
        train_parts.append(
            _attach_for_fit_months(part, arrivals, [m for m in months if m != holdout])
        )
    train_out = pl.concat(train_parts, how="vertical_relaxed")
    val_out = _attach_for_fit_months(val, arrivals, months)
    if train_out.height != train.height or val_out.height != val.height:
        raise RuntimeError("arrival-state cross-fit changed row counts")
    return train_out, val_out
