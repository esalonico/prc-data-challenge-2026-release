# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Load challenge Parquet files with a stable schema."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

CHALLENGE_AIRPORTS = (
    "EDDF",
    "EDDM",
    "EGLL",
    "EHAM",
    "LEBL",
    "LEMD",
    "LFPG",
    "LIRF",
    "LTFM",
    "LSZH",
)

AIRPORT_TIMEZONES = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LTFM": "Europe/Istanbul",
    "LSZH": "Europe/Zurich",
}

FORBIDDEN_DEPARTURE_LABEL_COLS = (
    "BLOCK_TIME_UTC_mvt",
    "TAXITIME_SEC_mvt",
    "AOBT_3_flt",
)

DATETIME_COLS = (
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
)

RAW_DIR = Path("data/raw")


def training_paths(raw_dir: Path | str = RAW_DIR) -> list[Path]:
    root = Path(raw_dir)
    paths = sorted(root.glob("training_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no training parquet files in {root}")
    return paths


def ranking_path(raw_dir: Path | str = RAW_DIR) -> Path:
    path = Path(raw_dir) / "ranking.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def submitting_path(raw_dir: Path | str = RAW_DIR) -> Path:
    path = Path(raw_dir) / "submitting.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _scan(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def scan_training(raw_dir: Path | str = RAW_DIR) -> pl.LazyFrame:
    return pl.concat([_scan(p) for p in training_paths(raw_dir)], how="vertical")


def scan_ranking(raw_dir: Path | str = RAW_DIR) -> pl.LazyFrame:
    return _scan(ranking_path(raw_dir))


def scan_submitting(raw_dir: Path | str = RAW_DIR) -> pl.LazyFrame:
    return _scan(submitting_path(raw_dir))


def reporting_airport_expr() -> pl.Expr:
    """DEP reports at ADEP; ARR reports at ADES."""
    return (
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt"))
        .when(pl.col("PHASE_mvt") == "ARR")
        .then(pl.col("ADES_mvt"))
        .otherwise(None)
        .alias("AIRPORT")
    )


def with_reporting_airport(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(reporting_airport_expr())


def taxi_identity_expr() -> pl.Expr:
    """Seconds identity used by the challenge for both phases."""
    return (
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds()
        )
        .when(pl.col("PHASE_mvt") == "ARR")
        .then(
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds()
        )
        .otherwise(None)
        .alias("TAXI_IDENTITY_SEC")
    )
