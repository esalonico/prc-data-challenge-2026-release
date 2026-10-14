# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Copernicus ERA5 / ERA5-Land and NASA POWER airport-time weather features.

Hourly reanalysis is as-of joined to takeoff (backward, UTC). Rolling disruption
indicators are computed on the hourly series, not from taxi labels.
"""

from __future__ import annotations

from typing import Any, Mapping

import polars as pl

ERA5_NUMS = (
    "era5_t2m_c",
    "era5_d2m_c",
    "era5_spread_c",
    "era5_wind10_ms",
    "era5_gust_ms",
    "era5_msl_hpa",
    "era5_sp_hpa",
    "era5_tp_mm_1h",
    "era5_sf_cm_1h",
    "era5_sd_m",
    "era5_cloud_pct",
    "era5_soil_t_c",
    "era5_tp_mm_6h",
    "era5_tp_mm_24h",
    "era5_sf_cm_6h",
    "era5_sf_cm_24h",
    "era5_gust_max_6h",
    "era5_msl_drop_3h",
    "era5_freeze_precip",
    "era5_deice",
    "era5_wind_disrupt",
    "era5_snow_on_ground",
    "era5_age_sec",
    "era5_missing",
)

POWER_NUMS = (
    "power_t2m_c",
    "power_wind10_ms",
    "power_precip_mm_1h",
    "power_ps_hpa",
    "power_precip_mm_6h",
    "power_precip_mm_24h",
    "power_age_sec",
    "power_missing",
)

ERA5_HOURLY_RENAME = {
    "temperature_2m": "era5_t2m_c",
    "dew_point_2m": "era5_d2m_c",
    "precipitation": "era5_tp_mm_1h",
    "snowfall": "era5_sf_cm_1h",
    "snow_depth": "era5_sd_m",
    "pressure_msl": "era5_msl_hpa",
    "surface_pressure": "era5_sp_hpa",
    "wind_speed_10m": "era5_wind10_ms",
    "wind_gusts_10m": "era5_gust_ms",
    "cloud_cover": "era5_cloud_pct",
    "soil_temperature_0cm": "era5_soil_t_c",
}

POWER_HOURLY_RENAME = {
    "T2M": "power_t2m_c",
    "WS10M": "power_wind10_ms",
    "PRECTOTCORR": "power_precip_mm_1h",
    "PS": "power_ps_hpa",
}

FORBIDDEN_WX_INPUTS = frozenset({"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"})


def reject_weather_labels(frame: pl.DataFrame) -> None:
    bad = sorted(FORBIDDEN_WX_INPUTS.intersection(frame.columns))
    if bad:
        raise RuntimeError(f"weather features cannot read {bad}")


def normalize_era5_hourly(raw: pl.DataFrame) -> pl.DataFrame:
    """Canonical hourly ERA5/ERA5-Land table at airport points."""

    frame = raw
    if "AIRPORT" not in frame.columns:
        raise RuntimeError("ERA5 hourly table requires AIRPORT")
    time_col = "time" if "time" in frame.columns else "era5_time"
    if time_col not in frame.columns:
        raise RuntimeError("ERA5 hourly table requires time")
    frame = frame.rename({time_col: "era5_time"})
    if frame["era5_time"].dtype != pl.Datetime:
        frame = frame.with_columns(pl.col("era5_time").str.to_datetime(time_zone="UTC").alias("era5_time"))
    elif str(frame["era5_time"].dtype) in {"Datetime(time_unit='us', time_zone=None)", "Datetime(time_unit='ns', time_zone=None)"}:
        frame = frame.with_columns(pl.col("era5_time").dt.replace_time_zone("UTC"))
    renames = {src: dst for src, dst in ERA5_HOURLY_RENAME.items() if src in frame.columns and dst not in frame.columns}
    if renames:
        frame = frame.rename(renames)
    if "era5_gust_ms" in frame.columns:
        # Open-Meteo default gust unit is km/h unless windspeed_unit=ms.
        gust = pl.col("era5_gust_ms")
        frame = frame.with_columns(
            pl.when(gust.median() > 40).then(gust / 3.6).otherwise(gust).alias("era5_gust_ms")
        )
    if "era5_wind10_ms" in frame.columns:
        wind = pl.col("era5_wind10_ms")
        frame = frame.with_columns(
            pl.when(wind.median() > 40).then(wind / 3.6).otherwise(wind).alias("era5_wind10_ms")
        )
    frame = frame.unique(["AIRPORT", "era5_time"], keep="last").sort(["AIRPORT", "era5_time"])
    extras: list[pl.Expr] = []
    if "era5_tp_mm_1h" in frame.columns:
        extras.extend(
            [
                pl.col("era5_tp_mm_1h").rolling_sum(window_size=6, min_samples=3).over("AIRPORT").alias("era5_tp_mm_6h"),
                pl.col("era5_tp_mm_1h").rolling_sum(window_size=24, min_samples=12).over("AIRPORT").alias("era5_tp_mm_24h"),
            ]
        )
    if "era5_sf_cm_1h" in frame.columns:
        extras.extend(
            [
                pl.col("era5_sf_cm_1h").rolling_sum(window_size=6, min_samples=3).over("AIRPORT").alias("era5_sf_cm_6h"),
                pl.col("era5_sf_cm_1h").rolling_sum(window_size=24, min_samples=12).over("AIRPORT").alias("era5_sf_cm_24h"),
            ]
        )
    if "era5_gust_ms" in frame.columns:
        extras.append(pl.col("era5_gust_ms").rolling_max(window_size=6, min_samples=3).over("AIRPORT").alias("era5_gust_max_6h"))
    if "era5_msl_hpa" in frame.columns:
        extras.append(
            (pl.col("era5_msl_hpa").shift(3).over("AIRPORT") - pl.col("era5_msl_hpa")).alias("era5_msl_drop_3h")
        )
    if extras:
        frame = frame.with_columns(extras)
    return frame


def add_era5_features(dep: pl.DataFrame, hourly: pl.DataFrame) -> pl.DataFrame:
    """Last ERA5 hour ≤ takeoff plus rolling disruption flags.

    Taxi labels may be present for scoring; they are never joined or transformed.
    """

    right = normalize_era5_hourly(hourly).unique(["AIRPORT", "era5_time"], keep="last").sort(["AIRPORT", "era5_time"])
    left = dep.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    joined = left.join_asof(
        right,
        left_on="MVT_TIME_UTC_mvt",
        right_on="era5_time",
        by="AIRPORT",
        strategy="backward",
    )
    if joined.height != left.height:
        raise RuntimeError(f"ERA5 asof join exploded rows {left.height} -> {joined.height}")
    t2m = pl.col("era5_t2m_c") if "era5_t2m_c" in joined.columns else pl.lit(None)
    tp = pl.col("era5_tp_mm_1h") if "era5_tp_mm_1h" in joined.columns else pl.lit(0.0)
    sf = pl.col("era5_sf_cm_1h") if "era5_sf_cm_1h" in joined.columns else pl.lit(0.0)
    sd = pl.col("era5_sd_m") if "era5_sd_m" in joined.columns else pl.lit(None)
    gust = pl.col("era5_gust_ms") if "era5_gust_ms" in joined.columns else pl.lit(None)
    dwpc = pl.col("era5_d2m_c") if "era5_d2m_c" in joined.columns else pl.lit(None)
    extras = [
        (t2m - dwpc).alias("era5_spread_c"),
        ((t2m <= 0.5) & (tp.fill_null(0) > 0.1)).cast(pl.Int8).alias("era5_freeze_precip"),
        ((t2m <= 3.0) & ((tp.fill_null(0) > 0.1) | (sf.fill_null(0) > 0.05))).cast(pl.Int8).alias("era5_deice"),
        (gust.fill_null(0) >= 15.0).cast(pl.Int8).alias("era5_wind_disrupt"),
        (sd.fill_null(0) > 0.01).cast(pl.Int8).alias("era5_snow_on_ground"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("era5_time")).dt.total_seconds().alias("era5_age_sec"),
        pl.col("era5_time").is_null().cast(pl.Int8).alias("era5_missing"),
    ]
    present = set(joined.columns) | {
        "era5_spread_c",
        "era5_freeze_precip",
        "era5_deice",
        "era5_wind_disrupt",
        "era5_snow_on_ground",
        "era5_age_sec",
        "era5_missing",
    }
    for name in ERA5_NUMS:
        if name not in present:
            extras.append(pl.lit(None).cast(pl.Float64).alias(name))
    return joined.with_columns(extras)


def normalize_power_hourly(raw: pl.DataFrame) -> pl.DataFrame:
    frame = raw
    if "time" in frame.columns:
        frame = frame.rename({"time": "power_time"})
    if frame["power_time"].dtype != pl.Datetime:
        frame = frame.with_columns(pl.col("power_time").str.to_datetime(time_zone="UTC"))
    renames = {src: dst for src, dst in POWER_HOURLY_RENAME.items() if src in frame.columns and dst not in frame.columns}
    if renames:
        frame = frame.rename(renames)
    frame = frame.unique(["AIRPORT", "power_time"], keep="last").sort(["AIRPORT", "power_time"])
    if "power_precip_mm_1h" in frame.columns:
        frame = frame.with_columns(
            [
                pl.col("power_precip_mm_1h").rolling_sum(window_size=6, min_samples=3).over("AIRPORT").alias("power_precip_mm_6h"),
                pl.col("power_precip_mm_1h").rolling_sum(window_size=24, min_samples=12).over("AIRPORT").alias("power_precip_mm_24h"),
            ]
        )
    return frame


def add_power_features(dep: pl.DataFrame, hourly: pl.DataFrame) -> pl.DataFrame:
    right = normalize_power_hourly(hourly).unique(["AIRPORT", "power_time"], keep="last").sort(["AIRPORT", "power_time"])
    left = dep.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    joined = left.join_asof(
        right,
        left_on="MVT_TIME_UTC_mvt",
        right_on="power_time",
        by="AIRPORT",
        strategy="backward",
    )
    if joined.height != left.height:
        raise RuntimeError(f"NASA POWER asof join exploded rows {left.height} -> {joined.height}")
    return joined.with_columns(
        [
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("power_time")).dt.total_seconds().alias("power_age_sec"),
            pl.col("power_time").is_null().cast(pl.Int8).alias("power_missing"),
        ]
    )


def weather_coverage(frame: pl.DataFrame, prefix: str = "era5") -> dict[str, Any]:
    miss = f"{prefix}_missing"
    if miss not in frame.columns:
        return {"n": frame.height, "joined_rate": 0.0}
    joined = 1.0 - float(frame[miss].mean())
    payload: dict[str, Any] = {"n": frame.height, "joined_rate": joined}
    for col in frame.columns:
        if col.startswith(prefix) and frame[col].dtype.is_numeric():
            payload[f"{col}_null_rate"] = float(frame[col].null_count() / frame.height)
    return payload
