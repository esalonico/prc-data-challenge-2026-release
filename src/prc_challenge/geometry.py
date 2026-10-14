# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Stand tokens and stand-to-runway distance. Does not use departure labels."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from prc_challenge.runways import load_runway_thresholds

OSM_PATH = Path("data/external/osm/parking_positions.parquet")
EARTH_RADIUS_M = 6_371_000.0
GEO_CATS = ("stand_prefix",)
GEO_NUMS = (
    "stand_number",
    "stand_to_rwy_m",
    "stand_coord_missing",
    "rwy_coord_missing",
)


def normalize_stand_key_expr(col: str = "STAND_mvt") -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.String)
        .str.to_uppercase()
        .str.replace_all(r"[\s\-]", "")
    )


def add_stand_tokens(df: pl.DataFrame) -> pl.DataFrame:
    key = normalize_stand_key_expr()
    stripped = key.str.replace(r"^([A-Z]*)0+(\d+)([A-Z]*)$", "$1$2$3")
    prefix = stripped.str.extract(r"^([A-Z]*)", 1).fill_null("__UNK__")
    prefix = pl.when(prefix == "").then(pl.lit("__NONE__")).otherwise(prefix)
    number = stripped.str.extract(r"(\d+)", 1).cast(pl.Int32, strict=False)
    return df.with_columns(
        [
            stripped.alias("_stand_key"),
            prefix.alias("stand_prefix"),
            number.alias("stand_number"),
        ]
    )


def _strip_leading_zeros(col: str = "stand_key") -> pl.Expr:
    """J05 -> J5, 010 -> 10. Unchanged when there is no extra zero."""
    return pl.col(col).str.replace(r"^([A-Z]*)0+(\d+)([A-Z]*)$", "$1$2$3")


def load_stand_coords(path: Path | str = OSM_PATH) -> pl.DataFrame:
    raw = pl.read_parquet(path)
    raw = raw.with_columns(normalize_stand_key_expr("stand_ref").alias("stand_key"))
    raw = raw.filter(pl.col("stand_key").is_not_null() & (pl.col("stand_key") != ""))
    raw = raw.with_columns((pl.col("aeroway") == "parking_position").cast(pl.Int8).alias("_pref"))
    primary = (
        raw.sort("_pref", descending=True)
        .unique(["AIRPORT", "stand_key"], keep="first")
        .select(
            [
                "AIRPORT",
                "stand_key",
                pl.col("lat").alias("stand_lat"),
                pl.col("lon").alias("stand_lon"),
            ]
        )
    )
    alt = (
        primary.with_columns(pl.col("stand_key").alias("_orig"), _strip_leading_zeros().alias("stand_key"))
        .filter(pl.col("stand_key") != pl.col("_orig"))
        .drop("_orig")
        .join(primary.select(["AIRPORT", "stand_key"]), on=["AIRPORT", "stand_key"], how="anti")
    )
    return pl.concat([primary, alt], how="vertical").unique(["AIRPORT", "stand_key"])


def haversine_m(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    rad = math.pi / 180.0
    phi1 = lat1 * rad
    phi2 = lat2 * rad
    dphi = (lat2 - lat1) * rad
    dlmb = (lon2 - lon1) * rad
    a = (dphi / 2).sin().pow(2) + phi1.cos() * phi2.cos() * (dlmb / 2).sin().pow(2)
    return (2.0 * EARTH_RADIUS_M) * a.sqrt().arcsin()


def add_geometry_features(
    df: pl.DataFrame,
    stands: pl.DataFrame | None = None,
    thresholds: pl.DataFrame | None = None,
    osm_path: Path | str = OSM_PATH,
) -> pl.DataFrame:
    """Prefix/number tokens plus haversine stand → assigned-runway threshold."""
    out = add_stand_tokens(df)
    if stands is None:
        path = Path(osm_path)
        stands = load_stand_coords(path) if path.exists() else pl.DataFrame(
            schema={
                "AIRPORT": pl.String,
                "stand_key": pl.String,
                "stand_lat": pl.Float64,
                "stand_lon": pl.Float64,
            }
        )
    if thresholds is None:
        thresholds = load_runway_thresholds()
    rwy = out.with_columns(pl.col("RUNWAY_mvt").cast(pl.String).str.to_uppercase().alias("_rwy_key"))
    mapped = rwy.join(stands, left_on=["AIRPORT", "_stand_key"], right_on=["AIRPORT", "stand_key"], how="left")
    mapped = mapped.join(
        thresholds.rename({"RUNWAY_mvt": "_rwy_key"}),
        on=["AIRPORT", "_rwy_key"],
        how="left",
    )
    dist = haversine_m(
        pl.col("stand_lat"),
        pl.col("stand_lon"),
        pl.col("rwy_lat"),
        pl.col("rwy_lon"),
    )
    return mapped.with_columns(
        [
            dist.alias("stand_to_rwy_m"),
            pl.col("stand_lat").is_null().cast(pl.Int8).alias("stand_coord_missing"),
            pl.col("rwy_lat").is_null().cast(pl.Int8).alias("rwy_coord_missing"),
        ]
    ).drop(["_stand_key", "_rwy_key", "stand_lat", "stand_lon", "rwy_lat", "rwy_lon"], strict=False)
