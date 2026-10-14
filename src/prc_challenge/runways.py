# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""OurAirports runway headings for wind-relative features."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from prc_challenge.io import CHALLENGE_AIRPORTS

DEFAULT_PATH = Path("data/external/ourairports/runways.csv")


def load_runway_headings(path: Path | str = DEFAULT_PATH) -> pl.DataFrame:
    raw = pl.read_csv(
        path,
        infer_schema_length=10000,
        schema_overrides={
            "le_heading_degT": pl.Float64,
            "he_heading_degT": pl.Float64,
            "le_ident": pl.String,
            "he_ident": pl.String,
            "airport_ident": pl.String,
            "closed": pl.Int64,
        },
    )
    raw = raw.filter(
        pl.col("airport_ident").is_in(list(CHALLENGE_AIRPORTS)) & (pl.col("closed") == 0)
    )
    le = raw.select(
        [
            pl.col("airport_ident").alias("AIRPORT"),
            pl.col("le_ident").cast(pl.String).str.to_uppercase().alias("RUNWAY_mvt"),
            pl.col("le_heading_degT").cast(pl.Float64, strict=False).alias("runway_heading_deg"),
        ]
    )
    he = raw.select(
        [
            pl.col("airport_ident").alias("AIRPORT"),
            pl.col("he_ident").cast(pl.String).str.to_uppercase().alias("RUNWAY_mvt"),
            pl.col("he_heading_degT").cast(pl.Float64, strict=False).alias("runway_heading_deg"),
        ]
    )
    return pl.concat([le, he]).drop_nulls(["RUNWAY_mvt", "runway_heading_deg"]).unique(
        ["AIRPORT", "RUNWAY_mvt"]
    )


def load_runway_thresholds(path: Path | str = DEFAULT_PATH) -> pl.DataFrame:
    """One lat/lon per airport+runway ident (the labeled threshold)."""
    raw = pl.read_csv(
        path,
        infer_schema_length=10000,
        schema_overrides={
            "le_ident": pl.String,
            "he_ident": pl.String,
            "airport_ident": pl.String,
            "le_latitude_deg": pl.Float64,
            "le_longitude_deg": pl.Float64,
            "he_latitude_deg": pl.Float64,
            "he_longitude_deg": pl.Float64,
            "closed": pl.Int64,
        },
    )
    raw = raw.filter(
        pl.col("airport_ident").is_in(list(CHALLENGE_AIRPORTS)) & (pl.col("closed") == 0)
    )
    le = raw.select(
        [
            pl.col("airport_ident").alias("AIRPORT"),
            pl.col("le_ident").cast(pl.String).str.to_uppercase().alias("RUNWAY_mvt"),
            pl.col("le_latitude_deg").alias("rwy_lat"),
            pl.col("le_longitude_deg").alias("rwy_lon"),
        ]
    )
    he = raw.select(
        [
            pl.col("airport_ident").alias("AIRPORT"),
            pl.col("he_ident").cast(pl.String).str.to_uppercase().alias("RUNWAY_mvt"),
            pl.col("he_latitude_deg").alias("rwy_lat"),
            pl.col("he_longitude_deg").alias("rwy_lon"),
        ]
    )
    return (
        pl.concat([le, he])
        .drop_nulls(["RUNWAY_mvt", "rwy_lat", "rwy_lon"])
        .unique(["AIRPORT", "RUNWAY_mvt"])
    )


def add_runway_heading(df: pl.DataFrame, headings: pl.DataFrame | None = None) -> pl.DataFrame:
    if headings is None:
        headings = load_runway_headings()
    rwy = df.with_columns(pl.col("RUNWAY_mvt").cast(pl.String).str.to_uppercase().alias("_rwy_key"))
    mapped = headings.rename({"RUNWAY_mvt": "_rwy_key"})
    return rwy.join(mapped, on=["AIRPORT", "_rwy_key"], how="left").drop("_rwy_key")
