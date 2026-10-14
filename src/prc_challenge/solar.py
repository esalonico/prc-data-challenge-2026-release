# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Solar elevation at takeoff from OurAirports coordinates. No AOBT/BLOCK."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from prc_challenge.io import CHALLENGE_AIRPORTS

AIRPORTS_CSV = Path("data/external/ourairports/airports.csv")
SOLAR_NUMS = ("solar_elev_deg", "is_night")


def load_airport_latlon(path: Path | str = AIRPORTS_CSV) -> pl.DataFrame:
    raw = pl.read_csv(path, infer_schema_length=5000)
    ident = "ident" if "ident" in raw.columns else "icao_code"
    return (
        raw.filter(pl.col(ident).is_in(list(CHALLENGE_AIRPORTS)))
        .select(
            [
                pl.col(ident).alias("AIRPORT"),
                pl.col("latitude_deg").cast(pl.Float64).alias("apt_lat"),
                pl.col("longitude_deg").cast(pl.Float64).alias("apt_lon"),
            ]
        )
        .unique("AIRPORT")
    )


def solar_elevation_deg(lat: np.ndarray, lon: np.ndarray, unix_sec: np.ndarray) -> np.ndarray:
    """Approximate solar elevation in degrees. Vectorized; UTC timestamps."""
    lat_r = np.deg2rad(lat)
    # Julian century from J2000
    jc = (unix_sec / 86400.0 - 10957.5) / 36525.0
    # Geometric mean longitude and anomaly (deg)
    gml = (280.46646 + 36000.76983 * jc) % 360.0
    gma = 357.52911 + 35999.05029 * jc
    gma_r = np.deg2rad(gma)
    eoc = (1.914602 - 0.004817 * jc) * np.sin(gma_r) + 0.019993 * np.sin(2 * gma_r)
    sun_long = np.deg2rad((gml + eoc) % 360.0)
    # Obliquity
    obl = np.deg2rad(23.439 - 0.0000004 * unix_sec / 86400.0)
    decl = np.arcsin(np.sin(obl) * np.sin(sun_long))
    # Equation of time (minutes) — simplified
    y = np.tan(obl / 2.0) ** 2
    eqt = 4 * np.rad2deg(
        y * np.sin(2 * np.deg2rad(gml))
        - 2 * 0.0167 * np.sin(gma_r)
        + 4 * 0.0167 * y * np.sin(gma_r) * np.cos(2 * np.deg2rad(gml))
    )
    true_solar = ((unix_sec % 86400) / 60.0 + eqt + 4 * lon) % 1440
    ha = np.deg2rad(true_solar / 4.0 - 180.0)
    cos_zen = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    elev = np.rad2deg(np.arcsin(np.clip(cos_zen, -1.0, 1.0)))
    return elev


def add_solar_features(df: pl.DataFrame, airports: pl.DataFrame | None = None) -> pl.DataFrame:
    """Solar elevation and night flag at takeoff. Ranking-safe."""
    if airports is None:
        airports = load_airport_latlon()
    out = df.join(airports, on="AIRPORT", how="left")
    lat = out["apt_lat"].fill_null(0.0).to_numpy()
    lon = out["apt_lon"].fill_null(0.0).to_numpy()
    unix = out["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.float64)
    elev = solar_elevation_deg(lat, lon, unix)
    missing = out["apt_lat"].is_null().to_numpy()
    elev = np.where(missing, np.nan, elev)
    night = np.where(missing | np.isnan(elev), 0, (elev < 0).astype(np.int8))
    return out.with_columns(
        [
            pl.Series("solar_elev_deg", elev),
            pl.Series("is_night", night),
        ]
    ).drop(["apt_lat", "apt_lon"])
