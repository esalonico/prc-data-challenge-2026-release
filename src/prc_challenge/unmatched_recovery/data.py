# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Load cache, raw unmatched sidecar, S4 parent predictions, and OPDI lists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from prc_challenge.campaign.data import locate_cache
from prc_challenge.campaign.runtime import CampaignConfig
from prc_challenge.campaign.scoring import assert_no_locked_months
from prc_challenge.features import add_calendar_features, add_flight_prefix, add_planning_features, add_reporting_airport
from prc_challenge.io import RAW_DIR, ranking_path, training_paths
from prc_challenge.unmatched_recovery.schema import (
    D_MONTHS,
    LOCKED_MONTHS,
    PRIMARY_MONTHS,
    RAW_SIDECAR_COLS,
    ensure_unmatched_aux,
)

S4_PRED_REL = (
    Path("artifacts")
    / "experiments"
    / "Overnight-20260919-agent2"
    / "predictions"
    / "s4_expected_occupancy"
    / "predictions_overnight_s4_expected_occupancy_V1.parquet"
)
S4_MODEL_REL = (
    Path("artifacts")
    / "experiments"
    / "Overnight-20260919-agent2"
    / "models"
    / "primary_s4_expected_occupancy.cbm"
)
E33B_PARENT_REL = Path("artifacts") / "experiments" / "E33b-matched-splice_V1" / "val_predictions.parquet"
OPDI_FLIGHT_LIST = Path("data") / "external" / "opdi" / "v002" / "flight_list"


def locate_existing(repo: Path, rel: Path, *extra: Path) -> Path | None:
    candidates = [repo / rel, *extra]
    tmp = Path("/private/tmp/prc-overnight-agent2") / rel
    candidates.append(tmp)
    for path in candidates:
        if path.exists():
            return path
    return None


def load_base_cache(cfg: CampaignConfig) -> pl.DataFrame:
    path = locate_cache(cfg)
    df = pl.read_parquet(path)
    if "month" not in df.columns and "MVT_TIME_UTC_mvt" in df.columns:
        df = df.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
    assert_no_locked_months(df)
    bad = df.filter(pl.col("month").is_in(list(LOCKED_MONTHS)))
    if bad.height:
        raise RuntimeError("refusing locked V4 months")
    return df


def build_raw_sidecar(repo: Path, dest: Path, months: tuple[int, ...] | None = None) -> pl.DataFrame:
    """Join movement-side fields omitted from the reduced e08b cache."""

    if dest.exists():
        return pl.read_parquet(dest)
    raw_dir = repo / RAW_DIR if (repo / RAW_DIR).exists() else Path(RAW_DIR)
    want = list(RAW_SIDECAR_COLS)
    keep_months = set(months or tuple(range(1, 11)))
    paths = [
        path
        for path in training_paths(raw_dir)
        if "2025-11-01" not in path.name and "2025-12-01" not in path.name
    ]
    if not paths:
        raise FileNotFoundError(f"no month 1–10 training files in {raw_dir}")
    lf = pl.concat([pl.scan_parquet(p) for p in paths], how="vertical_relaxed").filter(pl.col("PHASE_mvt") == "DEP")
    names = lf.collect_schema().names()
    cols = [c for c in want if c in names]
    if "MVT_TIME_UTC_mvt" in names and "MVT_TIME_UTC_mvt" not in cols:
        cols.append("MVT_TIME_UTC_mvt")
    frame = lf.select(cols).collect()
    if "month" not in frame.columns and "MVT_TIME_UTC_mvt" in frame.columns:
        frame = frame.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
    if "month" in frame.columns:
        frame = frame.filter(pl.col("month").is_in(list(keep_months)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(dest)
    return frame


def attach_raw_sidecar(cache: pl.DataFrame, sidecar: pl.DataFrame) -> pl.DataFrame:
    extra = [c for c in sidecar.columns if c not in cache.columns or c == "MVT_ID_mvt"]
    slim = sidecar.select(extra)
    joined = cache.join(slim, on="MVT_ID_mvt", how="left")
    if joined.height != cache.height:
        raise RuntimeError("raw sidecar join changed cache row count")
    return ensure_unmatched_aux(joined)


def load_s4_parent(repo: Path) -> pl.DataFrame:
    path = locate_existing(repo, S4_PRED_REL)
    if path is None:
        raise FileNotFoundError(f"need S4 parent predictions at {S4_PRED_REL}")
    pred = pl.read_parquet(path)
    need = {"MVT_ID_mvt", "prediction", "TAXITIME_SEC_mvt", "nm_unmatched", "month"}
    missing = need - set(pred.columns)
    if missing:
        raise RuntimeError(f"S4 parent missing {sorted(missing)}")
    if pred.height != 344_419:
        raise RuntimeError(f"S4 parent row count {pred.height} != 344419")
    return pred


def load_e33b_parent(repo: Path) -> pl.DataFrame | None:
    path = locate_existing(repo, E33B_PARENT_REL)
    if path is None:
        return None
    return pl.read_parquet(path)


def load_working_table(cfg: CampaignConfig, repo: Path, sidecar_path: Path) -> pl.DataFrame:
    cache = load_base_cache(cfg)
    sidecar = build_raw_sidecar(repo, sidecar_path)
    return attach_raw_sidecar(cache, sidecar)


def month_filter(frame: pl.DataFrame, months: tuple[int, ...]) -> pl.DataFrame:
    return frame.filter(pl.col("month").is_in(list(months)))


def d_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return month_filter(frame, D_MONTHS)


def primary_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return month_filter(frame, PRIMARY_MONTHS)


def load_ranking_departures(repo: Path) -> pl.DataFrame:
    """Ranking DEP rows with ranking-visible planning/calendar features. No hidden Y."""

    raw = ranking_path(repo / RAW_DIR if (repo / RAW_DIR).exists() else RAW_DIR)
    frame = pl.scan_parquet(raw).filter(pl.col("PHASE_mvt") == "DEP").collect()
    frame = add_reporting_airport(frame)
    frame = add_planning_features(frame)
    frame = add_calendar_features(frame)
    if "FLIGHT_mvt" in frame.columns:
        frame = add_flight_prefix(frame)
    if "month" not in frame.columns and "MVT_TIME_UTC_mvt" in frame.columns:
        frame = frame.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
    if "year" not in frame.columns and "MVT_TIME_UTC_mvt" in frame.columns:
        frame = frame.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.year().alias("year"))
    extras: list[pl.Expr] = []
    if "ADES_mvt" in frame.columns and "ades_missing" not in frame.columns:
        extras.append(pl.col("ADES_mvt").is_null().cast(pl.Int8).alias("ades_missing"))
    if "RUNWAY_mvt" in frame.columns and "runway_missing" not in frame.columns:
        extras.append(pl.col("RUNWAY_mvt").is_null().cast(pl.Int8).alias("runway_missing"))
    if "STAND_mvt" in frame.columns and "stand_missing" not in frame.columns:
        extras.append(pl.col("STAND_mvt").is_null().cast(pl.Int8).alias("stand_missing"))
    if extras:
        frame = frame.with_columns(extras)
    forbidden = {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"}
    drop = [c for c in forbidden if c in frame.columns]
    if drop:
        frame = frame.drop(drop)
    return frame


def load_ranking_unmatched(repo: Path) -> pl.DataFrame:
    """Unlabeled 2026 ranking unmatched covariates. Taxi labels are not present."""

    from prc_challenge.unmatched_recovery.subgroups import attach_identity_columns

    frame = attach_identity_columns(load_ranking_departures(repo))
    if "nm_unmatched" not in frame.columns:
        raise RuntimeError("ranking unmatched requires nm_unmatched from FLIGHT_ID_mvt")
    return unmatched_only(frame)


def unmatched_only(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("nm_unmatched") == 1)


def matched_only(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("nm_unmatched") == 0)


def opdi_flight_list_paths(repo: Path, months: tuple[int, ...]) -> list[Path]:
    root = repo / OPDI_FLIGHT_LIST
    paths = []
    for month in months:
        path = root / f"flight_list_2025{month:02d}.parquet"
        if path.exists():
            paths.append(path)
    return paths


def load_opdi_flight_list(repo: Path, months: tuple[int, ...]) -> pl.DataFrame | None:
    paths = opdi_flight_list_paths(repo, months)
    if not paths:
        return None
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")


def source_inventory(frame: pl.DataFrame) -> dict[str, Any]:
    unmatched = unmatched_only(frame)
    return {
        "n": frame.height,
        "n_unmatched": unmatched.height,
        "months": sorted(frame["month"].unique().to_list()) if "month" in frame.columns else [],
        "unmatched_type_missing": int(unmatched["type_missing"].sum()) if "type_missing" in unmatched.columns else None,
        "has_flight_mvt": "FLIGHT_mvt" in frame.columns,
        "has_ades": "ADES_mvt" in frame.columns,
        "has_flight_prefix": "flight_prefix" in frame.columns,
    }
