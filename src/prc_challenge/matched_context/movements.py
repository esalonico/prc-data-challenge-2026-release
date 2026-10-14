# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Masked movement table shared by the M1/M2 builders.

``mask_movements`` is the only place that touches raw BLOCK_TIME_UTC_mvt.  It
keeps the in-block time of **arrival** rows (``arr_inblock``) and drops the
departure outcome columns (BLOCK/TAXITIME, AOBT_3, ARVT_3) for every row before
anything else is derived.  Downstream builders call
``reject_forbidden_departure_columns`` on their input as a second guard.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from prc_challenge.operational_v2.surface_reconstruction import (
    FORBIDDEN_DEPARTURE_COLUMNS,
    MISSING_STANDS,
    reject_forbidden_departure_columns,
)

RAW_DIR = Path("data/raw")
EPOCH_NULL_COLS = ("sched", "eobt", "iobt")
MOVEMENT_COLUMNS = (
    "mvt_id",
    "window",
    "phase",
    "airport",
    "t",
    "sched",
    "eobt",
    "iobt",
    "plan",
    "plan_source",
    "runway",
    "stand",
    "aircraft_type",
    "airline",
    "ades",
    "arr_inblock",
    "arr_taxiin",
    "flight_id",
    "arr_same_flight_dep",
)


def training_file(month: int, raw_dir: Path = RAW_DIR) -> Path:
    end = f"2025-{month + 1:02d}-01" if month < 12 else "2026-01-01"
    return raw_dir / f"training_2025-{month:02d}-01_{end}.parquet"


def _text(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.String, strict=False).str.to_uppercase().str.strip_chars().replace("", None)


def _epoch(col: str) -> pl.Expr:
    return pl.col(col).dt.epoch("s").cast(pl.Int64)


def mask_movements(raw: pl.DataFrame, *, window: str | None = None) -> pl.DataFrame:
    """Return the label-free movement table for one source file.

    ``window`` names the context scope.  ``None`` uses the calendar month of
    the movement time (used for the ranking file, which holds two disjoint
    months); a string pins every row to that scope (one training month file).
    """

    required = ("MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt")
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"raw movements missing {missing}")
    phase = pl.col("PHASE_mvt").cast(pl.String).str.to_uppercase()
    # Arrival completion is the only BLOCK information that survives, and only on ARR rows.
    if "BLOCK_TIME_UTC_mvt" in raw.columns:
        arr_block = pl.when(phase == "ARR").then(_epoch("BLOCK_TIME_UTC_mvt")).otherwise(None)
    else:
        arr_block = pl.lit(None, dtype=pl.Int64)
    frame = raw.with_columns(arr_block.alias("__arr_inblock")).drop(
        [c for c in FORBIDDEN_DEPARTURE_COLUMNS if c in raw.columns]
    )
    reject_forbidden_departure_columns(frame)

    def opt_epoch(col: str) -> pl.Expr:
        return _epoch(col) if col in frame.columns else pl.lit(None, dtype=pl.Int64)

    def opt_text(col: str) -> pl.Expr:
        return _text(col) if col in frame.columns else pl.lit(None, dtype=pl.String)

    flight = opt_text("FLIGHT_mvt")
    callsign = opt_text("CALLSIGN_flt")
    airline = pl.coalesce(
        flight.str.extract(r"^([A-Z]{3}|[A-Z0-9]{2})", 1),
        callsign.str.extract(r"^([A-Z]{3}|[A-Z0-9]{2})", 1),
    )
    stand_txt = opt_text("STAND_mvt")
    airport = pl.when(phase == "DEP").then(_text("ADEP_mvt")).otherwise(_text("ADES_mvt"))
    if window is None:
        win = pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m")
    else:
        win = pl.lit(window)
    out = frame.select(
        pl.col("MVT_ID_mvt").alias("mvt_id"),
        win.alias("window"),
        phase.alias("phase"),
        airport.alias("airport"),
        _epoch("MVT_TIME_UTC_mvt").alias("t"),
        opt_epoch("SCHED_TIME_UTC_mvt").alias("sched"),
        opt_epoch("EOBT_1_flt").alias("eobt"),
        opt_epoch("IOBT_flt").alias("iobt"),
        opt_text("RUNWAY_mvt").alias("runway"),
        pl.when(stand_txt.is_null() | stand_txt.is_in(list(MISSING_STANDS)))
        .then(None)
        .otherwise(stand_txt)
        .alias("stand"),
        opt_text("AIRCRAFT_TYPE_mvt").alias("aircraft_type"),
        airline.alias("airline"),
        pl.when(phase == "DEP").then(_text("ADES_mvt")).otherwise(None).alias("ades"),
        pl.col("__arr_inblock").alias("arr_inblock"),
        (pl.col("FLIGHT_ID_mvt") if "FLIGHT_ID_mvt" in frame.columns else pl.lit(None, dtype=pl.Float64)).alias("flight_id"),
    ).with_columns(
        pl.coalesce("eobt", "iobt", "sched").alias("plan"),
        pl.when(pl.col("eobt").is_not_null())
        .then(0)
        .when(pl.col("iobt").is_not_null())
        .then(1)
        .when(pl.col("sched").is_not_null())
        .then(2)
        .otherwise(3)
        .cast(pl.Int8)
        .alias("plan_source"),
        (pl.col("arr_inblock") - pl.col("t")).alias("arr_taxiin"),
    )
    # ARR rows that share FLIGHT_ID_mvt with a DEP of the same airport and window (air
    # returns).  Variants >= v2 keep them out of every focal context (D-005: no focal
    # copies through the same FLIGHT_ID_mvt).
    dep_keys = (
        out.filter((pl.col("phase") == "DEP") & pl.col("flight_id").is_not_null())
        .select("window", "airport", "flight_id")
        .unique()
        .with_columns(pl.lit(True).alias("arr_same_flight_dep"))
    )
    out = out.join(dep_keys, on=["window", "airport", "flight_id"], how="left", maintain_order="left").with_columns(
        ((pl.col("phase") == "ARR") & pl.col("arr_same_flight_dep").fill_null(False)).alias("arr_same_flight_dep")
    )
    if out["mvt_id"].null_count():
        raise RuntimeError("movement input contains null MVT_ID_mvt")
    if out["mvt_id"].n_unique() != out.height:
        raise RuntimeError("movement input contains duplicate MVT_ID_mvt")
    bad = out.filter(~pl.col("phase").is_in(["ARR", "DEP"])).height
    if bad:
        raise RuntimeError(f"{bad} rows with unsupported phase")
    return out.select(MOVEMENT_COLUMNS)


def load_training_month(month: int, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return mask_movements(pl.read_parquet(training_file(month, raw_dir)), window=f"file-2025-{month:02d}")


def load_ranking(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return mask_movements(pl.read_parquet(raw_dir / "ranking.parquet"), window=None)


def check_movement_table(mov: pl.DataFrame) -> None:
    reject_forbidden_departure_columns(mov)
    missing = [c for c in MOVEMENT_COLUMNS if c not in mov.columns]
    if missing:
        raise RuntimeError(f"movement table missing {missing}; build it with mask_movements")
    leaked = mov.filter((pl.col("phase") == "DEP") & pl.col("arr_inblock").is_not_null()).height
    if leaked:
        raise RuntimeError("departure rows carry an in-block time")
