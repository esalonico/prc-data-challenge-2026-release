# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Target-independent OPDI flight-list matcher with frozen confidence tiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from prc_challenge.io import CHALLENGE_AIRPORTS
from prc_challenge.opdi.schema import normalize_callsign, standardize_flight_list

# Frozen matching policy (2025 calibration only; do not retune on 2026 before freeze).
TIER_A_MIN_TAKEOFF_MINUS_FIRST_SEEN_SEC = 60
TIER_A_MAX_TAKEOFF_MINUS_FIRST_SEEN_SEC = 7200
TIER_B_MIN_TAKEOFF_MINUS_FIRST_SEEN_SEC = 0
TIER_B_MAX_TAKEOFF_MINUS_FIRST_SEEN_SEC = 10800
TIER_A_MAX_LASTSEEN_MINUS_TAKEOFF_SEC = 900
TIER_B_MAX_LASTSEEN_MINUS_TAKEOFF_SEC = 1800


@dataclass(frozen=True)
class MatchConfig:
    tier_a_min_takeoff_minus_first_seen_sec: int = TIER_A_MIN_TAKEOFF_MINUS_FIRST_SEEN_SEC
    tier_a_max_takeoff_minus_first_seen_sec: int = TIER_A_MAX_TAKEOFF_MINUS_FIRST_SEEN_SEC
    tier_b_min_takeoff_minus_first_seen_sec: int = TIER_B_MIN_TAKEOFF_MINUS_FIRST_SEEN_SEC
    tier_b_max_takeoff_minus_first_seen_sec: int = TIER_B_MAX_TAKEOFF_MINUS_FIRST_SEEN_SEC
    tier_a_max_lastseen_minus_takeoff_sec: int = TIER_A_MAX_LASTSEEN_MINUS_TAKEOFF_SEC
    tier_b_max_lastseen_minus_takeoff_sec: int = TIER_B_MAX_LASTSEEN_MINUS_TAKEOFF_SEC


def _challenge_callsign_expr() -> pl.Expr:
    return pl.coalesce(
        [
            pl.col("FLIGHT_mvt").cast(pl.Utf8, strict=False),
            pl.col("CALLSIGN_flt").cast(pl.Utf8, strict=False),
        ]
    ).map_elements(normalize_callsign, return_dtype=pl.Utf8)


def load_challenge_departures(months: tuple[int, ...], raw_dir: str | Path = "data/raw") -> pl.DataFrame:
    """Load labeled DEP rows for selected UTC months from raw training parquets."""
    month_to_path = {
        1: "training_2025-01-01_2025-02-01.parquet",
        7: "training_2025-07-01_2025-08-01.parquet",
    }
    root = Path(raw_dir)
    frames = []
    for m in months:
        rel = month_to_path.get(m)
        if rel is None:
            raise ValueError(f"no cached raw path for month {m}")
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(
            pl.scan_parquet(path)
            .filter(pl.col("PHASE_mvt") == "DEP")
            .with_columns(pl.lit(m).alias("month"))
            .collect()
        )
    dep = pl.concat(frames, how="vertical")
    return dep.with_columns(
        [
            pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("takeoff_epoch"),
            pl.col("MVT_TIME_UTC_mvt").dt.date().alias("flight_date"),
            pl.col("ADEP_mvt").cast(pl.Utf8, strict=False).str.to_uppercase().alias("adep"),
            pl.col("ADES_mvt").cast(pl.Utf8, strict=False).str.to_uppercase().alias("ades"),
            pl.col("AIRCRAFT_TYPE_mvt").cast(pl.Utf8, strict=False).str.to_uppercase().alias("aircraft_type"),
            _challenge_callsign_expr().alias("callsign_norm"),
        ]
    ).filter(pl.col("callsign_norm").is_not_null() & pl.col("adep").is_in(list(CHALLENGE_AIRPORTS)))


def _build_candidates(challenge: pl.DataFrame, opdi: pl.DataFrame, cfg: MatchConfig) -> pl.DataFrame:
    op = opdi.select(
        [
            "opdi_id",
            "callsign_norm",
            "flight_date",
            "adep",
            "ades",
            "aircraft_type",
            "first_seen_epoch",
            "last_seen_epoch",
        ]
    )
    joined = challenge.with_columns(pl.col("flight_date").cast(pl.Utf8).alias("flight_date_key")).join(
        op.with_columns(pl.col("flight_date").cast(pl.Utf8).alias("flight_date_key")),
        on=["callsign_norm", "flight_date_key", "adep"],
        how="left",
        suffix="_opdi",
    ).with_columns(
        [
            (pl.col("takeoff_epoch") - pl.col("first_seen_epoch")).alias("takeoff_minus_first_seen_sec"),
            (pl.col("last_seen_epoch") - pl.col("takeoff_epoch")).alias("lastseen_minus_takeoff_sec"),
            (pl.col("ades") == pl.col("ades_opdi")).fill_null(False).alias("ades_match"),
            (pl.col("aircraft_type") == pl.col("aircraft_type_opdi")).fill_null(False).alias("type_match"),
        ]
    )
    a_time = (
        pl.col("takeoff_minus_first_seen_sec").is_not_null()
        & (pl.col("takeoff_minus_first_seen_sec") >= cfg.tier_a_min_takeoff_minus_first_seen_sec)
        & (pl.col("takeoff_minus_first_seen_sec") <= cfg.tier_a_max_takeoff_minus_first_seen_sec)
        & pl.col("lastseen_minus_takeoff_sec").is_not_null()
        & (pl.col("lastseen_minus_takeoff_sec").abs() <= cfg.tier_a_max_lastseen_minus_takeoff_sec)
    )
    b_time = (
        pl.col("takeoff_minus_first_seen_sec").is_not_null()
        & (pl.col("takeoff_minus_first_seen_sec") >= cfg.tier_b_min_takeoff_minus_first_seen_sec)
        & (pl.col("takeoff_minus_first_seen_sec") <= cfg.tier_b_max_takeoff_minus_first_seen_sec)
        & pl.col("lastseen_minus_takeoff_sec").is_not_null()
        & (pl.col("lastseen_minus_takeoff_sec").abs() <= cfg.tier_b_max_lastseen_minus_takeoff_sec)
    )
    return joined.with_columns(
        pl.when(a_time & pl.col("ades_match") & pl.col("type_match"))
        .then(pl.lit("A"))
        .when(b_time & (pl.col("ades_match") | pl.col("type_match")))
        .then(pl.lit("B"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("tier_candidate")
    )
    return joined


def _pick_unique_tier(candidates: pl.DataFrame, tier: str) -> pl.DataFrame:
    tier_rows = candidates.filter(pl.col("tier_candidate") == tier)
    counts = tier_rows.group_by("MVT_ID_mvt").agg(pl.len().alias("tier_candidate_count"))
    unique_ids = counts.filter(pl.col("tier_candidate_count") == 1).select("MVT_ID_mvt")
    picked = tier_rows.join(unique_ids, on="MVT_ID_mvt", how="inner").with_columns(
        [
            pl.lit(tier).alias("confidence_tier"),
            pl.lit(False).alias("ambiguous_match"),
            pl.lit(1).alias("tier_candidate_count"),
        ]
    )
    return picked


def match_departures_to_opdi(
    challenge: pl.DataFrame,
    opdi_paths: dict[int, str | Path],
    *,
    cfg: MatchConfig | None = None,
) -> pl.DataFrame:
    """Match challenge departures to OPDI flights without using taxi labels."""
    cfg = cfg or MatchConfig()
    opdi = pl.concat(
        [standardize_flight_list(path).collect() for path in opdi_paths.values()],
        how="vertical",
    )
    challenge = challenge.select(
        [
            "MVT_ID_mvt",
            "month",
            "flight_date",
            "takeoff_epoch",
            "adep",
            "ades",
            "aircraft_type",
            "callsign_norm",
            "TAXITIME_SEC_mvt",
        ]
    )
    candidates = _build_candidates(challenge, opdi, cfg)
    candidate_counts = candidates.group_by("MVT_ID_mvt").agg(
        pl.col("opdi_id").filter(pl.col("opdi_id").is_not_null()).len().alias("opdi_candidate_count"),
        pl.col("tier_candidate").filter(pl.col("tier_candidate") == "A").len().alias("tier_a_candidates"),
        pl.col("tier_candidate").filter(pl.col("tier_candidate") == "B").len().alias("tier_b_candidates"),
    )
    matched_a = _pick_unique_tier(candidates, "A")
    matched_b = _pick_unique_tier(
        candidates.join(matched_a.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti"),
        "B",
    )
    matched = pl.concat([matched_a, matched_b], how="vertical")
    ambiguous = (
        candidates.filter(pl.col("tier_candidate").is_not_null())
        .group_by("MVT_ID_mvt", "tier_candidate")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .select(["MVT_ID_mvt", "tier_candidate", pl.col("n").alias("ambiguous_tier_count")])
    )
    base = challenge.select(
        [
            "MVT_ID_mvt",
            "month",
            "adep",
            "ades",
            "aircraft_type",
            "callsign_norm",
            "TAXITIME_SEC_mvt",
            "takeoff_epoch",
        ]
    )
    out = (
        base.join(candidate_counts, on="MVT_ID_mvt", how="left")
        .join(
            matched.select(
                [
                    "MVT_ID_mvt",
                    "confidence_tier",
                    "opdi_id",
                    "takeoff_minus_first_seen_sec",
                    "lastseen_minus_takeoff_sec",
                    "ades_match",
                    "type_match",
                    "ambiguous_match",
                    "tier_candidate_count",
                ]
            ),
            on="MVT_ID_mvt",
            how="left",
        )
        .join(
            ambiguous.select(["MVT_ID_mvt", pl.col("ambiguous_tier_count")]),
            on="MVT_ID_mvt",
            how="left",
        )
    )
    return out.with_columns(
        [
            pl.col("opdi_candidate_count").fill_null(0),
            pl.col("tier_a_candidates").fill_null(0),
            pl.col("tier_b_candidates").fill_null(0),
            pl.col("ambiguous_match").fill_null(pl.col("ambiguous_tier_count").is_not_null()),
            pl.when(pl.col("confidence_tier").is_not_null())
            .then(pl.col("takeoff_minus_first_seen_sec"))
            .otherwise(None)
            .alias("opdi_first_seen_proxy_sec"),
            pl.lit(True).alias("callsign_match"),
        ]
    )
