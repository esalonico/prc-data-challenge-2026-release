# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Fold-safe building blocks for the bounded Agent-3 experiment queue.

The helpers in this module deliberately avoid validation labels.  Fitted
objects contain only summaries learned from the supplied training frame and
can therefore be reused by Q1, Q2, and Q3 without duplicating leakage-prone
encoding code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

TARGET = "TAXITIME_SEC_mvt"
MISSING_CATEGORY = "__MISSING__"

UNIMPEDED_LEVELS: tuple[tuple[str, ...], ...] = (
    ("AIRPORT", "STAND_mvt", "RUNWAY_mvt", "WK_TBL_CAT_flt"),
    ("AIRPORT", "STAND_mvt", "RUNWAY_mvt"),
    ("AIRPORT", "STAND_mvt"),
    ("AIRPORT", "RUNWAY_mvt", "WK_TBL_CAT_flt"),
    ("AIRPORT", "RUNWAY_mvt"),
    ("AIRPORT",),
)


def _normalise_categories(
    frame: pl.DataFrame, cols: tuple[str, ...] | list[str]
) -> pl.DataFrame:
    return frame.with_columns(
        [pl.col(c).fill_null(MISSING_CATEGORY).cast(pl.String).alias(c) for c in cols]
    )


@dataclass(frozen=True)
class HierarchicalBaseline:
    """Training-only hierarchical route baseline with global shrinkage."""

    statistic: str
    quantile: float | None
    min_count: int
    shrinkage: float
    global_value: float
    levels: tuple[tuple[str, ...], ...]
    tables: tuple[pl.DataFrame, ...]
    load_column: str | None = None
    load_cutoff: float | None = None


def fit_hierarchical_baseline(
    train: pl.DataFrame,
    *,
    statistic: str = "median",
    quantile: float | None = None,
    min_count: int = 40,
    shrinkage: float = 80.0,
    low_load_column: str | None = None,
    low_load_quantile: float = 0.25,
    levels: tuple[tuple[str, ...], ...] = UNIMPEDED_LEVELS,
) -> HierarchicalBaseline:
    """Fit one of the predeclared Q1 baseline families.

    ``statistic`` is either ``median`` or ``quantile``.  When
    ``low_load_column`` is provided, its cutoff is learned on ``train`` and all
    route summaries are fitted only on rows at or below that cutoff.
    """

    if TARGET not in train.columns:
        raise ValueError(f"training frame must contain {TARGET}")
    if statistic not in {"median", "quantile"}:
        raise ValueError(f"unsupported statistic {statistic!r}")
    if statistic == "quantile" and (quantile is None or not 0 < quantile < 1):
        raise ValueError("quantile baseline requires 0 < quantile < 1")
    required = sorted({c for level in levels for c in level})
    missing = [c for c in required if c not in train.columns]
    if missing:
        raise ValueError(f"baseline frame missing columns: {missing}")

    source = _normalise_categories(train, required)
    load_cutoff: float | None = None
    if low_load_column is not None:
        if low_load_column not in source.columns:
            raise ValueError(f"low-load column missing: {low_load_column}")
        load_cutoff = source.select(
            pl.col(low_load_column)
            .cast(pl.Float64, strict=False)
            .quantile(low_load_quantile, interpolation="linear")
        ).item()
        if load_cutoff is None or not np.isfinite(float(load_cutoff)):
            raise ValueError("cannot learn finite low-load cutoff")
        load_cutoff = float(load_cutoff)
        source = source.filter(
            pl.col(low_load_column).cast(pl.Float64, strict=False) <= load_cutoff
        )
    if source.height == 0:
        raise ValueError("empty baseline source")

    y = pl.col(TARGET).cast(pl.Float64)
    stat_expr = y.median() if statistic == "median" else y.quantile(float(quantile))
    global_value = float(source.select(stat_expr.alias("value")).item())
    tables: list[pl.DataFrame] = []
    for idx, keys in enumerate(levels):
        raw = source.group_by(list(keys)).agg(
            stat_expr.alias(f"_raw_{idx}"),
            pl.len().alias(f"_count_{idx}"),
        )
        weight = pl.col(f"_count_{idx}") / (pl.col(f"_count_{idx}") + float(shrinkage))
        table = raw.with_columns(
            (
                weight * pl.col(f"_raw_{idx}") + (1.0 - weight) * pl.lit(global_value)
            ).alias(f"_value_{idx}")
        ).drop(f"_raw_{idx}")
        tables.append(table)
    return HierarchicalBaseline(
        statistic=statistic,
        quantile=quantile,
        min_count=int(min_count),
        shrinkage=float(shrinkage),
        global_value=global_value,
        levels=levels,
        tables=tuple(tables),
        load_column=low_load_column,
        load_cutoff=load_cutoff,
    )


def predict_hierarchical_baseline(
    frame: pl.DataFrame,
    fitted: HierarchicalBaseline,
    *,
    output_column: str = "unimpeded_baseline",
) -> pl.DataFrame:
    """Apply a fitted hierarchy, choosing the first adequately supported level."""

    required = sorted({c for level in fitted.levels for c in level})
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame missing columns: {missing}")
    out = _normalise_categories(frame, required).with_row_index("_baseline_row")
    candidates: list[pl.Expr] = []
    support: list[pl.Expr] = []
    for idx, (keys, table) in enumerate(zip(fitted.levels, fitted.tables, strict=True)):
        out = out.join(table, on=list(keys), how="left", validate="m:1")
        valid = pl.col(f"_count_{idx}") >= fitted.min_count
        candidates.append(pl.when(valid).then(pl.col(f"_value_{idx}")))
        support.append(pl.when(valid).then(pl.col(f"_count_{idx}")))
    out = out.with_columns(
        pl.coalesce([*candidates, pl.lit(fitted.global_value)])
        .cast(pl.Float64)
        .alias(output_column),
        pl.coalesce([*support, pl.lit(0)])
        .cast(pl.Int64)
        .alias(f"{output_column}_support"),
    )
    drop = ["_baseline_row"]
    for idx in range(len(fitted.levels)):
        drop.extend([f"_count_{idx}", f"_value_{idx}"])
    return out.drop(drop).with_columns(
        pl.col(output_column)
        .is_finite()
        .cast(pl.Int8)
        .alias(f"{output_column}_available")
    )


def month_blocked_baseline(
    train: pl.DataFrame,
    *,
    fit_kwargs: dict[str, Any],
    output_column: str,
) -> pl.DataFrame:
    """Create leave-one-month-out baseline values for every training row."""

    if "month" not in train.columns:
        raise ValueError("month-blocked baseline requires month")
    parts: list[pl.DataFrame] = []
    for month in train["month"].unique().sort().to_list():
        fit = train.filter(pl.col("month") != month)
        hold = train.filter(pl.col("month") == month)
        fitted = fit_hierarchical_baseline(fit, **fit_kwargs)
        parts.append(
            predict_hierarchical_baseline(hold, fitted, output_column=output_column)
        )
    if not parts:
        raise ValueError("no months available for blocked baseline")
    out = pl.concat(parts)
    if out.height != train.height or out["MVT_ID_mvt"].n_unique() != train.height:
        raise RuntimeError("blocked baseline did not preserve one row per training ID")
    return out


@dataclass(frozen=True)
class CategoricalParityEncoder:
    """Training-fitted category codes and frequencies for Q3 tree parity."""

    categorical: tuple[str, ...]
    numeric: tuple[str, ...]
    vocabularies: dict[str, dict[str, int]]
    frequencies: dict[str, dict[str, float]]
    medians: dict[str, float]


def fit_parity_encoder(
    train: pl.DataFrame,
    categorical: list[str] | tuple[str, ...],
    numeric: list[str] | tuple[str, ...],
) -> CategoricalParityEncoder:
    vocabularies: dict[str, dict[str, int]] = {}
    frequencies: dict[str, dict[str, float]] = {}
    for col in categorical:
        values = train[col].fill_null(MISSING_CATEGORY).cast(pl.String)
        counts = pl.DataFrame({col: values}).group_by(col).len(name="n").sort(col)
        labels = counts[col].to_list()
        vocabularies[col] = {str(value): idx for idx, value in enumerate(labels)}
        total = max(1, train.height)
        frequencies[col] = {
            str(value): float(count) / total
            for value, count in zip(labels, counts["n"].to_list(), strict=True)
        }
    medians: dict[str, float] = {}
    for col in numeric:
        value = train.select(pl.col(col).cast(pl.Float64, strict=False).median()).item()
        medians[col] = (
            0.0 if value is None or not np.isfinite(float(value)) else float(value)
        )
    return CategoricalParityEncoder(
        categorical=tuple(categorical),
        numeric=tuple(numeric),
        vocabularies=vocabularies,
        frequencies=frequencies,
        medians=medians,
    )


def transform_parity_encoder(
    frame: pl.DataFrame,
    fitted: CategoricalParityEncoder,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Return dense float32 values plus categorical-code column indices.

    Every categorical field contributes both a training-vocabulary code and a
    training-only frequency.  Unknown values receive code ``-1`` and frequency
    zero.  LightGBM may treat the code columns as categorical; XGBoost receives
    the identical matrix and therefore the identical focal information.
    """

    arrays: list[np.ndarray] = []
    names: list[str] = []
    categorical_indices: list[int] = []
    for col in fitted.categorical:
        values = frame[col].fill_null(MISSING_CATEGORY).cast(pl.String).to_list()
        vocab = fitted.vocabularies[col]
        freq = fitted.frequencies[col]
        categorical_indices.append(len(arrays))
        arrays.append(
            np.asarray([vocab.get(str(v), -1) for v in values], dtype=np.float32)
        )
        names.append(f"{col}__code")
        arrays.append(
            np.asarray([freq.get(str(v), 0.0) for v in values], dtype=np.float32)
        )
        names.append(f"{col}__frequency")
    for col in fitted.numeric:
        series = (
            frame[col].cast(pl.Float32, strict=False).fill_null(fitted.medians[col])
        )
        values = np.asarray(series.to_numpy(), dtype=np.float32).copy()
        values[~np.isfinite(values)] = np.float32(fitted.medians[col])
        arrays.append(values)
        names.append(col)
    matrix = (
        np.column_stack(arrays).astype(np.float32, copy=False)
        if arrays
        else np.empty((frame.height, 0), np.float32)
    )
    return matrix, categorical_indices, names


def support_shrinkage_weight(
    n_train: int | np.ndarray, *, prior_rows: float = 20_000.0
):
    """Training-support-only shrinkage weight for specialist predictions."""

    n = np.asarray(n_train, dtype=np.float64)
    weight = n / (n + float(prior_rows))
    return float(weight) if weight.ndim == 0 else weight
