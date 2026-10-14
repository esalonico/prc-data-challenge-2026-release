# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Train-only categorical encodings and numeric scaling for neural baselines."""

from __future__ import annotations

import numpy as np
import polars as pl

UNK_INDEX = 0


def fit_vocabs(df: pl.DataFrame, cats: list[str]) -> dict[str, dict[str, int]]:
    vocabs: dict[str, dict[str, int]] = {}
    for col in cats:
        mapping: dict[str, int] = {}
        for i, value in enumerate(df[col].cast(pl.String).unique(maintain_order=True).to_list()):
            mapping[str(value)] = i + 1
        vocabs[col] = mapping
    return vocabs


def encode_cats(df: pl.DataFrame, vocabs: dict[str, dict[str, int]]) -> np.ndarray:
    cols = []
    for col, mapping in vocabs.items():
        encoded = np.fromiter(
            (mapping.get(str(v), UNK_INDEX) for v in df[col].cast(pl.String).to_list()),
            dtype=np.int64,
            count=df.height,
        )
        cols.append(encoded)
    if not cols:
        return np.zeros((df.height, 0), dtype=np.int64)
    return np.column_stack(cols)


def vocab_sizes(vocabs: dict[str, dict[str, int]]) -> list[int]:
    return [len(mapping) + 1 for mapping in vocabs.values()]


def fit_num_scaler(df: pl.DataFrame, nums: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in nums:
        series = df[col].cast(pl.Float64)
        median = float(series.median() or 0.0)
        filled = series.fill_null(median)
        mean = float(filled.mean())
        std = float(filled.std() or 0.0)
        stats[col] = {"median": median, "mean": mean, "std": std if std > 1e-6 else 1.0}
    return stats


SEQ_NUMS = [
    "takeoff_minus_sched_sec",
    "takeoff_minus_eobt_sec",
    "local_hour",
    "sknt",
    "vsby",
    "tmpc",
    "n_delayed_dep_ctr_1800s",
    "sec_since_prev_dep",
]


def add_sec_since_prev_dep(df: pl.DataFrame) -> pl.DataFrame:
    """Seconds since the previous takeoff at the same airport. Does not use taxi labels."""
    ordered = df.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    return ordered.with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").diff().over("AIRPORT").alias("sec_since_prev_dep")
    )


def neighbor_sequence(
    df: pl.DataFrame,
    cols: list[str],
    k: int,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Previous-k same-airport flights, oldest-to-newest. Lag >= 1, so the focal row is excluded.

    Ranking-safe: `cols` must not include TAXITIME / BLOCK / AOBT.
    Returns the time-sorted frame aligned with `seq` and `mask`.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    forbidden = {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"}
    if forbidden.intersection(cols):
        raise ValueError("neighbor sequence cannot include taxi/block/AOBT labels")
    ordered = df.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    exprs: list[pl.Expr] = []
    for lag in range(1, k + 1):
        for col in cols:
            exprs.append(pl.col(col).shift(lag).over("AIRPORT").alias(f"{col}__l{lag}"))
        exprs.append(pl.col("MVT_ID_mvt").shift(lag).over("AIRPORT").is_not_null().cast(pl.Float32).alias(f"mask__l{lag}"))
    wide = ordered.select(exprs)
    n = ordered.height
    seq = np.zeros((n, k, len(cols)), dtype=np.float32)
    mask = np.zeros((n, k), dtype=np.float32)
    for i, lag in enumerate(range(k, 0, -1)):
        mask[:, i] = wide[f"mask__l{lag}"].to_numpy()
        for j, col in enumerate(cols):
            seq[:, i, j] = wide[f"{col}__l{lag}"].fill_null(0.0).cast(pl.Float32).to_numpy()
        seq[:, i, :] *= mask[:, i : i + 1]
    return ordered, seq, mask


def transform_nums(df: pl.DataFrame, nums: list[str], stats: dict[str, dict[str, float]]) -> np.ndarray:
    cols = []
    for col in nums:
        spec = stats[col]
        series = df[col].cast(pl.Float64).fill_null(spec["median"]).fill_nan(spec["median"])
        values = (series.to_numpy() - spec["mean"]) / spec["std"]
        cols.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False))
    if not cols:
        return np.zeros((df.height, 0), dtype=np.float32)
    return np.column_stack(cols)
