# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Join the most recent eligible METAR before takeoff."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

METAR_DIR = Path("data/external/iem_metar")
WX_COLS = ["tmpc", "dwpc", "sknt", "gust", "vsby", "wxcodes", "p01i"]


def load_metar(metar_dir: Path | str = METAR_DIR) -> pl.DataFrame:
    root = Path(metar_dir)
    frames = []
    for path in sorted(root.glob("*.csv")):
        frames.append(
            pl.read_csv(
                path,
                null_values=["null", "M", ""],
                schema_overrides={
                    "station": pl.String,
                    "valid": pl.String,
                    "tmpc": pl.Float64,
                    "dwpc": pl.Float64,
                    "relh": pl.Float64,
                    "drct": pl.Float64,
                    "sknt": pl.Float64,
                    "gust": pl.Float64,
                    "vsby": pl.Float64,
                    "wxcodes": pl.String,
                    "p01i": pl.Float64,
                    "skyc1": pl.String,
                    "skyl1": pl.Float64,
                },
            )
        )
    metar = pl.concat(frames, how="diagonal_relaxed")
    return metar.with_columns(
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("metar_time")
    ).sort(["station", "metar_time"])


def add_metar_features(dep: pl.DataFrame, metar: pl.DataFrame) -> pl.DataFrame:
    left = dep.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    right = metar.rename({"station": "AIRPORT"}).sort(["AIRPORT", "metar_time"])
    joined = left.join_asof(
        right,
        left_on="MVT_TIME_UTC_mvt",
        right_on="metar_time",
        by="AIRPORT",
        strategy="backward",
    )
    return joined.with_columns(
        [
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("metar_time")).dt.total_seconds().alias("metar_age_sec"),
            pl.col("wxcodes").is_not_null().cast(pl.Int8).alias("metar_has_wx"),
            pl.col("wxcodes").fill_null("").str.contains("SN|FZ|PL").cast(pl.Int8).alias("metar_frozen"),
            pl.col("wxcodes").fill_null("").str.contains("FG|BR").cast(pl.Int8).alias("metar_fog"),
            pl.col("wxcodes").fill_null("").str.contains("TS").cast(pl.Int8).alias("metar_ts"),
            pl.col("wxcodes").fill_null("").str.contains("RA|DZ|SH").cast(pl.Int8).alias("metar_precip"),
            (pl.col("vsby") < 1.0).cast(pl.Int8).alias("metar_low_vis"),
        ]
    )


DEICE_NUMS = ("dewpoint_spread_c", "relh", "wx_deice")


def add_wx_interact_features(df: pl.DataFrame) -> pl.DataFrame:
    """De-icing / saturation proxies from already-joined origin METAR. No AOBT/BLOCK."""
    cold = pl.col("tmpc") <= 3.0
    wet = (pl.col("metar_precip") == 1) | (pl.col("metar_frozen") == 1) | (pl.col("p01i").fill_null(0) > 0)
    extras: list[pl.Expr] = [
        (pl.col("tmpc") - pl.col("dwpc")).alias("dewpoint_spread_c"),
        (cold & wet).cast(pl.Int8).alias("wx_deice"),
    ]
    if "relh" in df.columns:
        extras.append(pl.col("relh"))
    return df.with_columns(extras)


def add_dest_metar_features(dep: pl.DataFrame, metar: pl.DataFrame) -> pl.DataFrame:
    """Last METAR at ADES ≤ takeoff, only when destination is a challenge station."""
    if "ADES_mvt" not in dep.columns:
        raise ValueError("ADES_mvt required for destination METAR")
    left = dep.with_columns(pl.col("ADES_mvt").fill_null("__NONE__")).sort(["ADES_mvt", "MVT_TIME_UTC_mvt"])
    right = (
        metar.rename({"station": "ADES_mvt"})
        .select(["ADES_mvt", "metar_time", "tmpc", "sknt", "vsby", "wxcodes"])
        .rename(
            {
                "metar_time": "dest_metar_time",
                "tmpc": "dest_tmpc",
                "sknt": "dest_sknt",
                "vsby": "dest_vsby",
                "wxcodes": "dest_wxcodes",
            }
        )
        .sort(["ADES_mvt", "dest_metar_time"])
    )
    joined = left.join_asof(
        right,
        left_on="MVT_TIME_UTC_mvt",
        right_on="dest_metar_time",
        by="ADES_mvt",
        strategy="backward",
    )
    return joined.with_columns(
        [
            pl.col("dest_metar_time").is_not_null().cast(pl.Int8).alias("dest_metar_joined"),
            pl.col("dest_wxcodes").fill_null("").str.contains("SN|FZ|PL").cast(pl.Int8).alias("dest_metar_frozen"),
            pl.col("dest_wxcodes").fill_null("").str.contains("RA|DZ|SH").cast(pl.Int8).alias("dest_metar_precip"),
            (pl.col("dest_vsby") < 1.0).cast(pl.Int8).alias("dest_metar_low_vis"),
        ]
    )


def add_metar_trend(dep: pl.DataFrame, metar: pl.DataFrame, lag_sec: int = 10800) -> pl.DataFrame:
    """Deltas versus the last METAR at or before takeoff minus lag.

    Current METAR columns (`tmpc`, `sknt`, `vsby`, `gust`, `wxcodes`, and the
    `metar_*` flags) must already be joined. Lagged reports are as-of ≤ the
    lagged timestamp.
    """
    left = dep.with_columns(
        (pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=lag_sec)).alias("_lag_time")
    ).sort(["AIRPORT", "_lag_time"])
    right = (
        metar.rename({"station": "AIRPORT"})
        .select(["AIRPORT", "metar_time", "tmpc", "sknt", "vsby", "gust", "wxcodes"])
        .rename(
            {
                "metar_time": "metar_time_lag",
                "tmpc": "tmpc_lag",
                "sknt": "sknt_lag",
                "vsby": "vsby_lag",
                "gust": "gust_lag",
                "wxcodes": "wxcodes_lag",
            }
        )
        .sort(["AIRPORT", "metar_time_lag"])
    )
    joined = left.join_asof(
        right,
        left_on="_lag_time",
        right_on="metar_time_lag",
        by="AIRPORT",
        strategy="backward",
    )
    lag_precip = pl.col("wxcodes_lag").fill_null("").str.contains("RA|DZ|SH").cast(pl.Int8)
    lag_frozen = pl.col("wxcodes_lag").fill_null("").str.contains("SN|FZ|PL").cast(pl.Int8)
    lag_fog = pl.col("wxcodes_lag").fill_null("").str.contains("FG|BR").cast(pl.Int8)
    return joined.with_columns(
        [
            (pl.col("tmpc") - pl.col("tmpc_lag")).alias("tmpc_d3h"),
            (pl.col("sknt") - pl.col("sknt_lag")).alias("sknt_d3h"),
            (pl.col("vsby") - pl.col("vsby_lag")).alias("vsby_d3h"),
            (pl.col("gust").fill_null(pl.col("sknt")) - pl.col("gust_lag").fill_null(pl.col("sknt_lag"))).alias(
                "gust_d3h"
            ),
            (pl.col("metar_precip") - lag_precip).alias("precip_onset_3h"),
            (pl.col("metar_frozen") - lag_frozen).alias("frozen_onset_3h"),
            (pl.col("metar_fog") - lag_fog).alias("fog_onset_3h"),
            pl.col("metar_time_lag").is_not_null().cast(pl.Int8).alias("metar_lag_joined"),
        ]
    ).drop("_lag_time")


def relative_wind_deg(wind_from_deg: pl.Expr, heading_deg: pl.Expr) -> pl.Expr:
    """Signed angle from runway heading to METAR wind-from, in (-180, 180]."""
    return ((wind_from_deg - heading_deg + 180) % 360) - 180


def add_runway_wind(df: pl.DataFrame, headings: pl.DataFrame | None = None) -> pl.DataFrame:
    """Headwind/crosswind using METAR wind-from and OurAirports runway heading."""
    from prc_challenge.runways import add_runway_heading

    out = add_runway_heading(df, headings)
    # METAR drct is direction the wind is FROM, degrees true.
    rel = relative_wind_deg(pl.col("drct"), pl.col("runway_heading_deg"))
    speed = pl.col("sknt")
    rad = rel * (3.141592653589793 / 180.0)
    return out.with_columns(
        [
            rel.alias("wind_rel_deg"),
            (speed * rad.cos()).alias("headwind_kt"),
            (speed * rad.sin()).alias("crosswind_kt"),
            (speed.abs() * rad.sin().abs()).alias("crosswind_abs_kt"),
            pl.col("runway_heading_deg").is_null().cast(pl.Int8).alias("runway_heading_missing"),
        ]
    )


ALTI_DIR = Path("data/external/iem_alti")
ALTI_NUMS = ("alti_inhg", "alti_missing")


def load_alti(alti_dir: Path | str = ALTI_DIR) -> pl.DataFrame:
    root = Path(alti_dir)
    frames = []
    for path in sorted(root.glob("*.csv")):
        frames.append(
            pl.read_csv(
                path,
                null_values=["null", "M", ""],
                schema_overrides={"station": pl.String, "valid": pl.String, "alti": pl.Float64},
            )
        )
    if not frames:
        raise FileNotFoundError(f"no altimeter csv in {root}")
    alti = pl.concat(frames, how="diagonal_relaxed")
    return alti.with_columns(
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("alti_time")
    ).sort(["station", "alti_time"])


def add_alti_features(dep: pl.DataFrame, alti: pl.DataFrame) -> pl.DataFrame:
    """Last altimeter setting (inches Hg) at or before takeoff. Ranking-safe METAR."""
    left = dep.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    right = (
        alti.rename({"station": "AIRPORT"})
        .select(["AIRPORT", "alti_time", "alti"])
        .rename({"alti": "alti_inhg"})
        .sort(["AIRPORT", "alti_time"])
    )
    joined = left.join_asof(
        right,
        left_on="MVT_TIME_UTC_mvt",
        right_on="alti_time",
        by="AIRPORT",
        strategy="backward",
    )
    return joined.with_columns(pl.col("alti_inhg").is_null().cast(pl.Int8).alias("alti_missing"))


RAW_METAR_DIR = Path("data/external/iem_raw_metar")
RVR_NUMS = ("rvr_min_m", "rvr_missing", "rvr_lvp", "metar_cavok")
TREND_NUMS = ("metar_nosig", "metar_tempo", "metar_becmg", "metar_recent_wx")
_RVR_TOKEN = re.compile(r"\bR\d{2}[LCR]?/(P|M)?(\d{4})(?:V(?:P|M)?(\d{4}))?")


def parse_rvr_min_m(metar: str | None) -> float | None:
    """Minimum runway visual range in metres from a raw METAR. None if absent."""
    if not metar:
        return None
    vals: list[int] = []
    for match in _RVR_TOKEN.finditer(metar):
        vals.append(int(match.group(2)))
        if match.group(3):
            vals.append(int(match.group(3)))
    if not vals:
        return None
    return float(min(vals))


def load_raw_metar(raw_dir: Path | str = RAW_METAR_DIR) -> pl.DataFrame:
    root = Path(raw_dir)
    frames = []
    for path in sorted(root.glob("*.csv")):
        frames.append(
            pl.read_csv(
                path,
                null_values=["null", "M", ""],
                schema_overrides={"station": pl.String, "valid": pl.String, "metar": pl.String},
            )
        )
    if not frames:
        raise FileNotFoundError(f"no raw METAR csv in {root}")
    raw = pl.concat(frames, how="diagonal_relaxed")
    return raw.with_columns(
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("raw_metar_time")
    ).sort(["station", "raw_metar_time"])


def add_rvr_features(dep: pl.DataFrame, raw_metar: pl.DataFrame) -> pl.DataFrame:
    """Last raw METAR ≤ takeoff: min RVR metres, LVP flag, CAVOK. Ranking-safe."""
    right = raw_metar.rename({"station": "AIRPORT"}).select(["AIRPORT", "raw_metar_time", "metar"]).sort(
        ["AIRPORT", "raw_metar_time"]
    )
    joined = dep.sort(["AIRPORT", "MVT_TIME_UTC_mvt"]).join_asof(
        right,
        left_on="MVT_TIME_UTC_mvt",
        right_on="raw_metar_time",
        by="AIRPORT",
        strategy="backward",
    )
    rvr = pl.col("metar").map_elements(parse_rvr_min_m, return_dtype=pl.Float64)
    metar_s = pl.col("metar").fill_null("")
    return joined.with_columns(
        [
            rvr.alias("rvr_min_m"),
            rvr.is_null().cast(pl.Int8).alias("rvr_missing"),
            (rvr <= 550.0).fill_null(False).cast(pl.Int8).alias("rvr_lvp"),
            metar_s.str.contains("CAVOK").cast(pl.Int8).alias("metar_cavok"),
            metar_s.str.contains("NOSIG").cast(pl.Int8).alias("metar_nosig"),
            metar_s.str.contains(r"\bTEMPO\b").cast(pl.Int8).alias("metar_tempo"),
            metar_s.str.contains(r"\bBECMG\b").cast(pl.Int8).alias("metar_becmg"),
            metar_s.str.contains(r"\bRE[A-Z]{2,}").cast(pl.Int8).alias("metar_recent_wx"),
        ]
    ).drop("metar", strict=False)
