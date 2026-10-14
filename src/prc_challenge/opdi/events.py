# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""OPDI v0.0.2 flight-event schema normalization and projected reads."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

# Departure-relevant event types retained for D1 linkage / surface proxies.
DEPARTURE_EVENT_TYPES: tuple[str, ...] = (
    "take-off",
    "exit-parking_position",
    "entry-taxiway",
    "exit-taxiway",
    "entry-runway",
)

# Predeclared D1a sample: OPDI rolling 10-day intervals (not calendar month starts).
# Jan: 20250105_20250115 covers 2025-01-05 .. 2025-01-14 (inclusive).
# Jul: corresponding fixed-position slot 20250704_20250714 covers 2025-07-04 .. 2025-07-13.
D1A_SAMPLE_CHUNKS: tuple[dict[str, str], ...] = (
    {"label": "jan_2025", "start": "20250105", "end": "20250115"},
    {"label": "jul_2025", "start": "20250704", "end": "20250714"},
)


def event_chunk_filename(start: str, end: str) -> str:
    return f"flight_events_{start}_{end}.parquet"

EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "event_id": ("id", "event_id", "evt_id"),
    "flight_id": ("flight_id", "opdi_flight_id", "flt_id"),
    "event_type": ("type", "event_type", "evt_type"),
    "event_time": ("event_time", "time", "timestamp", "ts"),
    "info": ("info", "event_info", "metadata"),
}


def _resolve_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def resolve_event_columns(columns: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in EVENT_ALIASES.items():
        col = _resolve_column(columns, aliases)
        if col is None and canonical != "event_id":
            missing.append(canonical)
        elif col is not None:
            out[canonical] = col
    if missing:
        raise RuntimeError(f"OPDI flight_events missing expected fields: {missing}; columns={columns}")
    return out


def _event_time_epoch(col: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    if dtype.is_temporal():
        return col.dt.epoch("s").cast(pl.Int64)
    return col.cast(pl.Int64, strict=False)


def scan_flight_events(
    path: Path | str,
    *,
    flight_ids: Iterable[int] | None = None,
    event_types: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Project OPDI events to canonical columns with optional pushdown filters."""
    path = Path(path)
    scan = pl.scan_parquet(path)
    schema = scan.collect_schema()
    mapping = resolve_event_columns(schema.names())
    time_col = pl.col(mapping["event_time"])
    time_dtype = schema[mapping["event_time"]]
    lf = scan.select(
        pl.col(mapping["flight_id"]).cast(pl.UInt64).alias("opdi_id"),
        pl.col(mapping["event_type"]).cast(pl.Utf8, strict=False).str.to_lowercase().alias("event_type"),
        time_col.alias("event_time_ts"),
        _event_time_epoch(time_col, time_dtype).alias("event_epoch"),
        pl.col(mapping["info"]).cast(pl.Utf8, strict=False).alias("info"),
        *(
            [pl.col(mapping["event_id"]).alias("event_id")]
            if "event_id" in mapping
            else []
        ),
    )
    if event_types is not None:
        types = [t.lower() for t in event_types]
        lf = lf.filter(pl.col("event_type").is_in(types))
    if flight_ids is not None:
        ids = list(flight_ids)
        if ids:
            lf = lf.filter(pl.col("opdi_id").is_in(ids))
    return lf


def load_flight_events(
    paths: Sequence[Path | str],
    *,
    flight_ids: Iterable[int] | None = None,
    event_types: Sequence[str] | None = None,
) -> pl.DataFrame:
    frames = [
        scan_flight_events(p, flight_ids=flight_ids, event_types=event_types).collect()
        for p in paths
    ]
    if not frames:
        return pl.DataFrame(
            schema={
                "opdi_id": pl.UInt64,
                "event_type": pl.Utf8,
                "event_time_ts": pl.Datetime(time_unit="us", time_zone="UTC"),
                "event_epoch": pl.Int64,
                "info": pl.Utf8,
            }
        )
    return pl.concat(frames, how="vertical_relaxed")


def takeoff_events_by_flight(events: pl.DataFrame) -> pl.DataFrame:
    """One take-off row per flight; if duplicated, keep earliest event time."""
    takeoffs = events.filter(pl.col("event_type") == "take-off")
    if takeoffs.is_empty():
        return takeoffs.select(["opdi_id", "event_epoch", "event_time_ts", "info"])
    return (
        takeoffs.sort(["opdi_id", "event_epoch"])
        .group_by("opdi_id")
        .agg(
            pl.col("event_epoch").first().alias("takeoff_event_epoch"),
            pl.col("event_time_ts").first().alias("takeoff_event_time_ts"),
            pl.col("info").first().alias("takeoff_event_info"),
        )
    )
