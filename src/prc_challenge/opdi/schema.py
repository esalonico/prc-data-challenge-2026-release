# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""OPDI flight-list schema discovery and field normalization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl

# OPDI v0.0.2 flight-list columns (verified after download; aliases tried in order).
FLIGHT_LIST_ALIASES: dict[str, tuple[str, ...]] = {
    "opdi_id": ("id", "flight_id", "flt_id", "opdi_flight_id"),
    "icao24": ("icao24", "ICAO24"),
    "callsign": ("flt_id", "callsign", "flight_id", "flight"),
    "date_of_flight": ("dof", "date_of_flight", "flight_date", "day"),
    "adep": ("adep", "ADEP", "origin"),
    "ades": ("ades", "ADES", "destination"),
    "registration": ("registration", "reg", "acr"),
    "aircraft_type": ("typecode", "aircraft_type", "actype", "type", "ac_type", "model"),
    "operator": ("icao_operator", "operator", "operating_airline", "airline"),
    "first_seen": ("first_seen", "firstseen", "first_seen_ts"),
    "last_seen": ("last_seen", "lastseen", "last_seen_ts"),
}


def _resolve_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def resolve_flight_list_columns(columns: list[str]) -> dict[str, str]:
    """Map canonical names to actual OPDI flight-list column names."""
    out: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in FLIGHT_LIST_ALIASES.items():
        col = _resolve_column(columns, aliases)
        if col is None:
            missing.append(canonical)
        else:
            out[canonical] = col
    if missing:
        raise RuntimeError(f"OPDI flight_list missing expected fields: {missing}; columns={columns}")
    return out


def normalize_callsign(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    if not s or s in {"NULL", "NAN", "NONE", "NA"}:
        return None
    s = re.sub(r"\s+", "", s)
    return s or None


def _series_stats(s: pl.Series) -> dict[str, Any]:
    n = s.len()
    nulls = int(s.null_count())
    finite = s.drop_nulls()
    out: dict[str, Any] = {"n": n, "nulls": nulls, "null_rate": nulls / n if n else None}
    if finite.dtype in (pl.Utf8, pl.String):
        out["n_unique"] = int(finite.n_unique())
        sample = finite.head(5).to_list()
        out["sample"] = sample
        return out
    if finite.dtype in (pl.Datetime, pl.Date):
        out["min"] = str(finite.min()) if finite.len() else None
        out["max"] = str(finite.max()) if finite.len() else None
        return out
    num = finite.cast(pl.Float64, strict=False)
    num = num.filter(num.is_finite())
    if num.len():
        out["mean"] = float(num.mean())
        out["std"] = float(num.std()) if num.len() > 1 else 0.0
        out["min"] = float(num.min())
        out["max"] = float(num.max())
    return out


def inspect_flight_list_schema(path: Path | str, *, sample_rows: int = 5000) -> dict[str, Any]:
    """Summarize dtypes, null rates and airport coverage for one flight-list parquet."""
    path = Path(path)
    scan = pl.scan_parquet(path)
    columns = scan.collect_schema().names()
    mapping = resolve_flight_list_columns(columns)
    head = scan.head(sample_rows).collect()
    fields: dict[str, Any] = {}
    for canonical, col in mapping.items():
        fields[canonical] = {"column": col, **_series_stats(head[col])}
    adep_col = mapping["adep"]
    ades_col = mapping["ades"]
    adep_counts = (
        head.group_by(adep_col)
        .len()
        .sort("len", descending=True)
        .head(20)
        .rename({adep_col: "airport", "len": "n"})
        .to_dicts()
    )
    return {
        "path": str(path),
        "n_sample_rows": head.height,
        "column_mapping": mapping,
        "fields": fields,
        "top_adep": adep_counts,
        "top_ades": head.group_by(ades_col)
        .len()
        .sort("len", descending=True)
        .head(20)
        .rename({ades_col: "airport", "len": "n"})
        .to_dicts(),
    }


def standardize_flight_list(path: Path | str) -> pl.LazyFrame:
    """Return OPDI flight list with canonical column names and helper fields."""
    path = Path(path)
    scan = pl.scan_parquet(path)
    schema = scan.collect_schema()
    mapping = resolve_flight_list_columns(schema.names())
    first = pl.col(mapping["first_seen"])
    last = pl.col(mapping["last_seen"])
    dof = pl.col(mapping["date_of_flight"])
    first_dtype = schema[mapping["first_seen"]]
    last_dtype = schema[mapping["last_seen"]]
    dof_dtype = schema[mapping["date_of_flight"]]
    if first_dtype.is_temporal():
        first_epoch = first.dt.epoch("s").cast(pl.Int64)
    else:
        first_epoch = first.cast(pl.Int64, strict=False)
    if last_dtype.is_temporal():
        last_epoch = last.dt.epoch("s").cast(pl.Int64)
    else:
        last_epoch = last.cast(pl.Int64, strict=False)
    if dof_dtype == pl.Date:
        flight_date = dof
    elif dof_dtype.is_temporal():
        flight_date = dof.dt.date()
    else:
        flight_date = dof.cast(pl.Utf8, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    callsign_raw = pl.col(mapping["callsign"]).cast(pl.Utf8, strict=False)
    return scan.select(
        pl.col(mapping["opdi_id"]).alias("opdi_id"),
        pl.col(mapping["icao24"]).alias("icao24"),
        callsign_raw.alias("callsign_raw"),
        callsign_raw.map_elements(normalize_callsign, return_dtype=pl.Utf8).alias("callsign_norm"),
        flight_date.alias("flight_date"),
        pl.col(mapping["adep"]).cast(pl.Utf8, strict=False).str.to_uppercase().alias("adep"),
        pl.col(mapping["ades"]).cast(pl.Utf8, strict=False).str.to_uppercase().alias("ades"),
        pl.col(mapping["registration"]).cast(pl.Utf8, strict=False).alias("registration"),
        pl.col(mapping["aircraft_type"]).cast(pl.Utf8, strict=False).str.to_uppercase().alias("aircraft_type"),
        pl.col(mapping["operator"]).cast(pl.Utf8, strict=False).alias("operator"),
        first.alias("first_seen_ts"),
        last.alias("last_seen_ts"),
        first_epoch.alias("first_seen_epoch"),
        last_epoch.alias("last_seen_epoch"),
    ).filter(pl.col("callsign_norm").is_not_null() & pl.col("adep").is_not_null())
