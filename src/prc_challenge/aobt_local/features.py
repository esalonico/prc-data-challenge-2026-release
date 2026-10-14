# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""AOBT/LOBT feature block keyed by ``MVT_ID_mvt`` (LOCAL_ONLY, source_mode aobt_local).

Inputs are ranking-visible DEP fields only: takeoff (``MVT_TIME_UTC_mvt``), the
NM timestamps ``AOBT_3_flt``, ``LOBT_flt``, ``EOBT_1_flt``, ``IOBT_flt`` and the
schedule. The DEP labels ``BLOCK_TIME_UTC_mvt`` / ``TAXITIME_SEC_mvt`` are never
read, and ``ARVT_3_flt`` is never read.

``AOBT_3_flt`` is the NM current-tactical-model off-block time. On a large share of
rows it is not an observed off-block but ``takeoff - default taxi`` (the value of
``takeoff - AOBT`` piles up on one or two minute values per airport and runway;
e.g. 36-43% of LTFM rows sit on 19 or 24 min). The two label-free "mode" columns
let a model tell that derived AOBT apart from an observed one. They are computed
from the same (airport, runway, calendar month) group the row belongs to, using
only AOBT and takeoff, so they work the same way for training months and for the
ranking months.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

SOURCE_MODE = "aobt_local"

TIMESTAMP_INPUTS = (
    "MVT_ID_mvt",
    "FLIGHT_ID_mvt",
    "PHASE_mvt",
    "ADEP_mvt",
    "RUNWAY_mvt",
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "AOBT_3_flt",
)
LABEL_COLUMNS = frozenset({"BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "ARVT_3_flt"})

AOBT_FEATURE_COLUMNS = (
    "takeoff_minus_aobt_sec",
    "aobt_minus_eobt_sec",
    "aobt_minus_lobt_sec",
    "aobt_minus_sched_sec",
    "lobt_minus_iobt_sec",
    "lobt_minus_eobt_sec",
    "takeoff_minus_lobt_sec",
    "lobt_eq_aobt",
    "aobt_eq_eobt",
    "aobt_missing",
    "aobt_has_seconds",
    "aobt_after_takeoff",
    "aobt_lobt_gt_2h",
    "aobt_taxi_value_share",
    "aobt_taxi_is_mode",
)
# Output names must not collide with the forbidden raw columns of the screen harness.
assert not LABEL_COLUMNS.intersection(AOBT_FEATURE_COLUMNS)
assert "AOBT_3_flt" not in AOBT_FEATURE_COLUMNS


def load_dep_timestamps(paths: Iterable[Path]) -> pl.DataFrame:
    """Read DEP rows with the timestamp inputs only (labels are never loaded)."""
    frames = []
    for path in paths:
        cols = [c for c in TIMESTAMP_INPUTS if c in pl.read_parquet_schema(path)]
        frames.append(pl.read_parquet(path, columns=cols).filter(pl.col("PHASE_mvt") == "DEP"))
    out = pl.concat(frames, how="vertical_relaxed").unique("MVT_ID_mvt", keep="first")
    if LABEL_COLUMNS.intersection(out.columns):
        raise RuntimeError("label columns leaked into the AOBT feature inputs")
    return out


def _sec(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)


def build_aobt_features(dep: pl.DataFrame) -> pl.DataFrame:
    """Return one row per ``MVT_ID_mvt`` with :data:`AOBT_FEATURE_COLUMNS`."""
    bad = LABEL_COLUMNS.intersection(dep.columns)
    if bad:
        raise ValueError(f"label columns must not be passed to the AOBT builder: {sorted(bad)}")
    aobt = pl.col("AOBT_3_flt")
    df = dep.with_columns(
        pl.col("ADEP_mvt").alias("_airport"),
        (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("_ym"),
        (
            (pl.col("MVT_TIME_UTC_mvt").dt.truncate("1m") - aobt.dt.truncate("1m")).dt.total_minutes()
        ).alias("_taxi_min"),
    )
    keys = ["_airport", "RUNWAY_mvt", "_ym"]
    present = df.filter(aobt.is_not_null())
    counts = present.group_by([*keys, "_taxi_min"]).agg(pl.len().alias("_k"))
    totals = counts.group_by(keys).agg(pl.col("_k").sum().alias("_n"), pl.col("_k").max().alias("_kmax"))
    counts = counts.join(totals, on=keys, how="left", nulls_equal=True).with_columns(
        (pl.col("_k") / pl.col("_n")).alias("aobt_taxi_value_share"),
        (pl.col("_k") == pl.col("_kmax")).cast(pl.Int8).alias("aobt_taxi_is_mode"),
    )
    df = df.join(
        counts.select([*keys, "_taxi_min", "aobt_taxi_value_share", "aobt_taxi_is_mode"]),
        on=[*keys, "_taxi_min"],
        how="left",
        nulls_equal=True,
    )
    missing = aobt.is_null()
    out = df.select(
        "MVT_ID_mvt",
        _sec("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("takeoff_minus_aobt_sec"),
        _sec("AOBT_3_flt", "EOBT_1_flt").alias("aobt_minus_eobt_sec"),
        _sec("AOBT_3_flt", "LOBT_flt").alias("aobt_minus_lobt_sec"),
        _sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("aobt_minus_sched_sec"),
        _sec("LOBT_flt", "IOBT_flt").alias("lobt_minus_iobt_sec"),
        _sec("LOBT_flt", "EOBT_1_flt").alias("lobt_minus_eobt_sec"),
        _sec("MVT_TIME_UTC_mvt", "LOBT_flt").alias("takeoff_minus_lobt_sec"),
        pl.when(missing).then(None).otherwise((aobt == pl.col("LOBT_flt")).cast(pl.Int8)).alias("lobt_eq_aobt"),
        pl.when(missing).then(None).otherwise((aobt == pl.col("EOBT_1_flt")).cast(pl.Int8)).alias("aobt_eq_eobt"),
        missing.cast(pl.Int8).alias("aobt_missing"),
        pl.when(missing).then(None).otherwise((aobt.dt.second() != 0).cast(pl.Int8)).alias("aobt_has_seconds"),
        pl.when(missing).then(None).otherwise((aobt > pl.col("MVT_TIME_UTC_mvt")).cast(pl.Int8)).alias("aobt_after_takeoff"),
        pl.when(missing)
        .then(None)
        .otherwise((_sec("AOBT_3_flt", "LOBT_flt") > 7200).cast(pl.Int8))
        .alias("aobt_lobt_gt_2h"),
        pl.col("aobt_taxi_value_share").cast(pl.Float64),
        pl.col("aobt_taxi_is_mode").cast(pl.Int8),
    )
    if out.height != dep.height:
        raise RuntimeError("AOBT builder changed the row count")
    return out
