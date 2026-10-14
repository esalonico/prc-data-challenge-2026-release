# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Open IEM weather / disruption and OSM/OurAirports geometry features.

Ranking-safe: no AOBT, BLOCK, TAXITIME, or Open-Meteo. January/July labels are
never used to build these columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from prc_challenge.campaign.scoring import assert_no_forbidden_features
from prc_challenge.geometry import GEO_CATS, GEO_NUMS, add_geometry_features, load_stand_coords
from prc_challenge.solar import SOLAR_NUMS, add_solar_features
from prc_challenge.unmatched_recovery.schema import present, unmatched_observed_features
from prc_challenge.weather import (
    ALTI_NUMS,
    DEICE_NUMS,
    RVR_NUMS,
    TREND_NUMS,
    add_alti_features,
    add_metar_features,
    add_metar_trend,
    add_rvr_features,
    add_runway_wind,
    add_wx_interact_features,
    load_alti,
    load_metar,
    load_raw_metar,
)

WX_TREND_NUMS = (
    "tmpc_d3h",
    "sknt_d3h",
    "vsby_d3h",
    "gust_d3h",
    "precip_onset_3h",
    "frozen_onset_3h",
    "fog_onset_3h",
    "metar_lag_joined",
)
WX_WIND_NUMS = (
    "wind_rel_deg",
    "headwind_kt",
    "crosswind_kt",
    "crosswind_abs_kt",
    "runway_heading_missing",
)
WEATHER_EXTRA_NUMS = DEICE_NUMS + WX_TREND_NUMS + RVR_NUMS + TREND_NUMS + ALTI_NUMS + WX_WIND_NUMS
WEATHER_EXTRA_CATS: tuple[str, ...] = ()
GEO_EXTRA_CATS = GEO_CATS
GEO_EXTRA_NUMS = GEO_NUMS + SOLAR_NUMS

COMBINED_VARIANTS = ("combined_full", "combined_no_raw_metar")


def _repo_path(repo: Path, rel: str) -> Path:
    return (repo / rel) if not Path(rel).is_absolute() else Path(rel)


def _join_metar_extra(dep: pl.DataFrame, metar: pl.DataFrame) -> pl.DataFrame:
    extras = [c for c in ("drct", "relh") if c not in dep.columns and c in metar.columns]
    if not extras or "MVT_TIME_UTC_mvt" not in dep.columns:
        return dep
    right = metar
    if "station" in right.columns and "AIRPORT" not in right.columns:
        right = right.rename({"station": "AIRPORT"})
    keep = ["AIRPORT", "metar_time", *extras]
    right = right.select([c for c in keep if c in right.columns]).sort(["AIRPORT", "metar_time"])
    left = dep.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    return left.join_asof(
        right,
        left_on="MVT_TIME_UTC_mvt",
        right_on="metar_time",
        by="AIRPORT",
        strategy="backward",
    )


def attach_open_weather_features(
    frame: pl.DataFrame,
    *,
    repo: Path,
    metar: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """IEM METAR disruption extras. Does not call Open-Meteo or Copernicus."""

    if "MVT_TIME_UTC_mvt" not in frame.columns or "AIRPORT" not in frame.columns:
        return frame
    metar_dir = _repo_path(repo, "data/external/iem_metar")
    if metar is None and metar_dir.exists():
        metar = load_metar(metar_dir)
    out = frame
    if metar is not None:
        if "tmpc" not in out.columns:
            out = add_metar_features(out, metar)
        out = _join_metar_extra(out, metar)
        if {"tmpc", "dwpc"}.issubset(out.columns):
            out = add_wx_interact_features(out)
        if "metar_precip" in out.columns:
            out = add_metar_trend(out, metar)
        if "drct" in out.columns and "sknt" in out.columns:
            headings_path = _repo_path(repo, "data/external/ourairports/runways.csv")
            if headings_path.exists():
                out = add_runway_wind(out)
        alti_dir = _repo_path(repo, "data/external/iem_alti")
        if alti_dir.exists() and any(alti_dir.glob("*.csv")):
            out = add_alti_features(out, load_alti(alti_dir))
        raw_dir = _repo_path(repo, "data/external/iem_raw_metar")
        if raw_dir.exists() and any(raw_dir.glob("*.csv")):
            out = add_rvr_features(out, load_raw_metar(raw_dir))
    assert_no_forbidden_features([c for c in out.columns if c in WEATHER_EXTRA_NUMS])
    return out


def _stands_from_osm_json(osm_dir: Path) -> pl.DataFrame | None:
    rows: list[dict[str, Any]] = []
    for path in sorted(osm_dir.glob("*_parking.json")):
        payload = json.loads(path.read_text())
        airport = path.name.split("_")[0].upper()
        for el in payload.get("elements") or []:
            tags = el.get("tags") or {}
            if "lat" in el and "lon" in el:
                lat, lon = el["lat"], el["lon"]
            else:
                center = el.get("center") or {}
                lat, lon = center.get("lat"), center.get("lon")
            if lat is None or lon is None:
                continue
            refs: list[str] = []
            for key in ("ref", "local_ref", "name", "ref:icao"):
                val = tags.get(key)
                if not val:
                    continue
                for part in str(val).split(";"):
                    part = part.strip()
                    if part:
                        refs.append(part)
            for ref in dict.fromkeys(refs):
                rows.append(
                    {
                        "AIRPORT": airport,
                        "stand_ref": ref,
                        "lat": float(lat),
                        "lon": float(lon),
                        "aeroway": tags.get("aeroway"),
                    }
                )
    if not rows:
        return None
    tmp = Path(osm_dir) / "_stands_from_json.parquet"
    pl.DataFrame(rows).write_parquet(tmp)
    return load_stand_coords(tmp)


def attach_physical_geometry_features(frame: pl.DataFrame, *, repo: Path) -> pl.DataFrame:
    """OSM parking positions + OurAirports thresholds/coordinates. ODbL + public domain."""

    out = frame
    osm_parquet = _repo_path(repo, "data/external/osm/parking_positions.parquet")
    osm_dir = _repo_path(repo, "data/external/osm")
    stands = None
    if osm_parquet.exists():
        stands = load_stand_coords(osm_parquet)
    elif osm_dir.exists():
        stands = _stands_from_osm_json(osm_dir)
    if "STAND_mvt" in out.columns and "RUNWAY_mvt" in out.columns:
        out = add_geometry_features(out, stands=stands)
    airports_csv = _repo_path(repo, "data/external/ourairports/airports.csv")
    if airports_csv.exists() and "MVT_TIME_UTC_mvt" in out.columns:
        out = add_solar_features(out)
    return out


def attach_open_data_features(frame: pl.DataFrame, *, repo: Path, metar: pl.DataFrame | None = None) -> pl.DataFrame:
    return attach_physical_geometry_features(
        attach_open_weather_features(frame, repo=repo, metar=metar),
        repo=repo,
    )


def recipe_feature_lists(frame: pl.DataFrame, recipe: str) -> tuple[list[str], list[str]]:
    """Return (cats, nums) for a frozen unmatched recipe. Unknown columns are dropped."""

    cats, nums = unmatched_observed_features(frame)
    weather_nums = present(frame, WEATHER_EXTRA_NUMS)
    weather_cats = present(frame, WEATHER_EXTRA_CATS)
    geo_nums = present(frame, GEO_EXTRA_NUMS)
    geo_cats = present(frame, GEO_EXTRA_CATS)
    if recipe in {"S0", "u6", "control"}:
        extra_c, extra_n = [], []
    elif recipe in {"S1", "open_weather"}:
        extra_c, extra_n = weather_cats, weather_nums
    elif recipe in {"S2", "physical_geometry"}:
        extra_c, extra_n = geo_cats, geo_nums
    elif recipe in {"combined_full", "S3"}:
        extra_c, extra_n = [*weather_cats, *geo_cats], [*weather_nums, *geo_nums]
    elif recipe == "combined_no_raw_metar":
        skip = set(RVR_NUMS + TREND_NUMS + ALTI_NUMS)
        extra_c = [*weather_cats, *geo_cats]
        extra_n = [c for c in weather_nums if c not in skip] + geo_nums
    else:
        raise ValueError(f"unknown unmatched recipe {recipe}")
    cats = list(dict.fromkeys([*cats, *extra_c]))
    nums = list(dict.fromkeys([*nums, *extra_n]))
    assert_no_forbidden_features(cats + nums)
    return cats, nums


def provenance_for(recipe: str) -> dict[str, Any]:
    native = {
        "native_challenge_movements": True,
        "iem_metar": True,
        "open_meteo": False,
        "era5": False,
        "opdi": False,
        "opensky_research": False,
        "aobt": False,
        "v4": False,
        "matched_parent": "S4 expected occupancy / ranking v12",
    }
    if recipe in {"S0", "u6", "control"}:
        return {**native, "unmatched_route": "U6 observed-feature CatBoost", "eligibility": "native_only"}
    if recipe in {"S1", "open_weather"}:
        return {
            **native,
            "unmatched_route": "U6 + IEM METAR disruption extras",
            "sources": ["IEM METAR", "IEM raw METAR", "IEM altimeter", "OurAirports runway headings"],
            "eligibility": "native_open_weather",
        }
    if recipe in {"S2", "physical_geometry"}:
        return {
            **native,
            "unmatched_route": "U6 + OSM/OurAirports physical geometry",
            "sources": ["OSM parking_position ODbL", "OurAirports runways/airports public domain"],
            "eligibility": "native_open_geometry",
        }
    return {
        **native,
        "unmatched_route": "internally selected U6 + open weather + OSM/OurAirports",
        "sources": ["IEM METAR", "OSM ODbL", "OurAirports"],
        "eligibility": "native_open_combined",
        "combined_variant": recipe,
    }
