# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Frozen native-ATC OPDI rotation sidecars for one isolated month."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from prc_challenge.io import CHALLENGE_AIRPORTS
from prc_challenge.opdi.arr_bridge import link_previous_arrivals
from prc_challenge.opdi.ceiling import (
    build_candidate_universe,
    load_challenge_population,
    load_opdi_flights,
)
from prc_challenge.opdi.rotation import (
    aircraft_history_aggregates,
    link_previous_aircraft_flights,
)
from prc_challenge.opdi.schema import normalize_callsign

ROTATION_NUMERIC_FEATURES = (
    "rotation_l1_broad",
    "rotation_l1_strict",
    "rotation_high_confidence",
    "rotation_arr_usable",
    "rotation_missing",
    "takeoff_minus_previous_arrival_sec",
    "takeoff_minus_previous_inblock_sec",
    "previous_arrival_taxi_in_sec",
    "previous_inbound_arrival_delay_sec",
    "scheduled_turnaround_sec",
    "seconds_since_previous_flight",
    "prior_flights_6h",
    "prior_flights_12h",
    "prior_flights_24h",
    "prior_duration_sec_12h",
    "prior_duration_sec_24h",
    "same_stand",
    "rotation_late_inbound_short_ground",
)

ROTATION_CATEGORICAL_FEATURES = (
    "rotation_failure_class",
    "arr_link_status",
    "previous_arrival_runway",
    "previous_flight_origin",
    "turnaround_duration_bucket",
)


def _select_native_l1(challenge: pl.DataFrame, opdi: pl.DataFrame) -> pl.DataFrame:
    challenge = challenge.with_columns(
        pl.col("callsign_atc").alias("callsign_link"),
        pl.when(pl.col("callsign_atc").is_not_null())
        .then(pl.lit("atc"))
        .otherwise(None)
        .alias("callsign_source"),
    )
    candidates = build_candidate_universe(challenge, opdi)
    eligible = candidates.filter(
        (pl.col("actual_adep_match") | pl.col("planned_adep_match"))
        & pl.col("temporal_plausible")
        & (pl.col("route_match") | pl.col("type_match"))
    )
    selected = (
        eligible.sort("MVT_ID_mvt", "opdi_id")
        .group_by("MVT_ID_mvt", maintain_order=True)
        .agg(
            pl.len().alias("l1_candidate_count"),
            pl.col("opdi_id").first().alias("current_opdi_id"),
            pl.col("route_match").first().alias("current_route_match"),
            pl.col("type_match").first().alias("current_type_match"),
        )
        .with_columns(
            (pl.col("l1_candidate_count") == 1).alias("L1_broad"),
            (
                (pl.col("l1_candidate_count") == 1)
                & pl.col("current_route_match")
                & pl.col("current_type_match")
            ).alias("L1_strict"),
        )
    )
    metadata = opdi.select(
        pl.col("opdi_id").alias("current_opdi_id"),
        pl.col("icao24").alias("current_icao24"),
        pl.col("registration").alias("current_registration"),
        pl.col("callsign_norm").alias("current_callsign"),
        pl.col("opdi_adep").alias("current_opdi_adep"),
        pl.col("opdi_ades").alias("current_opdi_ades"),
        pl.col("opdi_ades_p").alias("current_opdi_ades_p"),
        pl.col("opdi_type").alias("current_opdi_type"),
        pl.col("opdi_operator").alias("current_opdi_operator"),
        pl.col("first_seen_epoch").alias("current_first_seen_epoch"),
        pl.col("last_seen_epoch").alias("current_last_seen_epoch"),
    )
    return (
        challenge.select(
            "MVT_ID_mvt",
            "month",
            "flight_date",
            "takeoff_epoch",
            "adep",
            "ades",
            "SCHED_TIME_UTC_mvt",
            "callsign_atc",
            "callsign_link",
            "callsign_source",
        )
        .join(selected, on="MVT_ID_mvt", how="left")
        .with_columns(
            pl.col("L1_broad").fill_null(False),
            pl.col("L1_strict").fill_null(False),
            pl.col("l1_candidate_count").fill_null(0),
        )
        .join(metadata, on="current_opdi_id", how="left")
    )


def _arrival_identity(raw_path: str | Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(raw_path)
        .filter(
            (pl.col("PHASE_mvt") == "ARR")
            & pl.col("ADES_mvt").is_in(list(CHALLENGE_AIRPORTS))
        )
        .select(
            pl.col("MVT_ID_mvt").alias("arr_mvt_id"),
            pl.col("ADES_mvt").str.to_uppercase().alias("arr_airport"),
            pl.col("CALLSIGN_flt")
            .map_elements(normalize_callsign, return_dtype=pl.Utf8)
            .alias("arr_callsign"),
            pl.col("ADEP_mvt").str.to_uppercase().alias("arr_origin"),
            pl.col("AIRCRAFT_TYPE_mvt").str.to_uppercase().alias("arr_type"),
            pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias(
                "arr_mvt_epoch"
            ),
        )
        .collect()
    )


def _allowed_arrival_features(raw_path: str | Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    arrivals = (
        pl.scan_parquet(raw_path)
        .filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            pl.col("MVT_ID_mvt").alias("selected_arr_mvt_id"),
            pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias(
                "previous_arrival_mvt_epoch"
            ),
            pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias(
                "previous_arrival_inblock_epoch"
            ),
            pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias(
                "previous_arrival_sched_epoch"
            ),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(
                "previous_arrival_taxi_in_sec"
            ),
            pl.col("STAND_mvt").alias("previous_arrival_stand"),
            pl.col("RUNWAY_mvt").alias("previous_arrival_runway"),
            pl.col("ADEP_mvt").alias("previous_flight_origin"),
        )
        .collect()
    )
    departures = (
        pl.scan_parquet(raw_path)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select(
            "MVT_ID_mvt",
            pl.col("STAND_mvt").alias("current_departure_stand"),
            pl.col("RUNWAY_mvt").alias("current_departure_runway"),
            pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias(
                "current_sched_epoch"
            ),
        )
        .collect()
    )
    return arrivals, departures


def _failure_class(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(~pl.col("L1_broad"))
        .then(pl.lit("current_not_native_l1"))
        .when(~pl.col("previous_in_lookback"))
        .then(pl.lit("no_previous_in_36h"))
        .when(pl.col("intervening_same_aircraft_flight"))
        .then(pl.lit("intervening_flight"))
        .when(~pl.col("previous_ends_at_current_adep"))
        .then(pl.lit("previous_ends_elsewhere"))
        .when(pl.col("rotation_ambiguous"))
        .then(pl.lit("ambiguous_or_impossible"))
        .when(pl.col("registration_inconsistent") | pl.col("type_inconsistent"))
        .then(pl.lit("identity_inconsistent"))
        .otherwise(pl.lit("high_confidence"))
        .alias("rotation_failure_class")
    )


def build_rotation_month(
    *,
    month: int,
    raw_path: str | Path,
    current_opdi_path: str | Path,
    previous_opdi_path: str | Path,
) -> pl.DataFrame:
    """Build a target-independent native-callsign rotation sidecar for a month."""

    challenge = load_challenge_population({month: raw_path})
    current_opdi = load_opdi_flights({month: current_opdi_path})
    history = load_opdi_flights(
        {month - 1: previous_opdi_path, month: current_opdi_path}
    ).unique("opdi_id")
    identities = _select_native_l1(challenge, current_opdi)
    linked = link_previous_aircraft_flights(
        identities.filter(pl.col("L1_broad")), history, max_lookback_seconds=36 * 3600
    )
    rotations = identities.join(
        linked.select(
            "MVT_ID_mvt",
            *[c for c in linked.columns if c.startswith("previous_")],
            "latest_adep_previous_opdi_id",
            "latest_adep_previous_last_seen_epoch",
            "seconds_since_previous_flight",
            "intervening_same_aircraft_flight",
            "overlapping_or_impossible",
            "registration_inconsistent",
            "type_inconsistent",
            "rotation_ambiguous",
            "rotation_high_confidence",
        ),
        on="MVT_ID_mvt",
        how="left",
    ).with_columns(
        pl.col("previous_in_lookback").fill_null(False),
        pl.col("previous_ends_at_current_adep").fill_null(False),
        pl.col("intervening_same_aircraft_flight").fill_null(False),
        pl.col("registration_inconsistent").fill_null(False),
        pl.col("type_inconsistent").fill_null(False),
        pl.col("rotation_ambiguous").fill_null(False),
        pl.col("rotation_high_confidence").fill_null(False),
    )
    rotations = _failure_class(rotations)
    history_agg = aircraft_history_aggregates(rotations.filter(pl.col("L1_broad")), history)
    rotations = rotations.join(history_agg, on="MVT_ID_mvt", how="left")
    links = link_previous_arrivals(rotations, _arrival_identity(raw_path))
    arrivals, departures = _allowed_arrival_features(raw_path)
    featured = links.join(arrivals, on="selected_arr_mvt_id", how="left").join(
        departures, on="MVT_ID_mvt", how="left"
    )
    featured = featured.with_columns(
        (
            pl.col("arr_link_status").is_in(["ARR-A", "ARR-B"])
            & (pl.col("previous_arrival_mvt_epoch") < pl.col("takeoff_epoch"))
            & (pl.col("previous_arrival_inblock_epoch") < pl.col("takeoff_epoch"))
            & (pl.col("previous_arrival_inblock_epoch") >= pl.col("previous_arrival_mvt_epoch"))
        )
        .fill_null(False)
        .alias("rotation_arr_usable")
    )
    allowed = pl.col("rotation_arr_usable")
    featured = featured.with_columns(
        pl.col("L1_broad").cast(pl.Int8).alias("rotation_l1_broad"),
        pl.col("L1_strict").cast(pl.Int8).alias("rotation_l1_strict"),
        pl.col("rotation_high_confidence").cast(pl.Int8),
        pl.col("rotation_arr_usable").cast(pl.Int8),
        (~allowed).cast(pl.Int8).alias("rotation_missing"),
        pl.when(allowed)
        .then(pl.col("takeoff_epoch") - pl.col("previous_arrival_mvt_epoch"))
        .otherwise(None)
        .alias("takeoff_minus_previous_arrival_sec"),
        pl.when(allowed)
        .then(pl.col("takeoff_epoch") - pl.col("previous_arrival_inblock_epoch"))
        .otherwise(None)
        .alias("takeoff_minus_previous_inblock_sec"),
        pl.when(allowed)
        .then(pl.col("previous_arrival_taxi_in_sec"))
        .otherwise(None)
        .alias("previous_arrival_taxi_in_sec"),
        pl.when(allowed)
        .then(pl.col("previous_arrival_mvt_epoch") - pl.col("previous_arrival_sched_epoch"))
        .otherwise(None)
        .alias("previous_inbound_arrival_delay_sec"),
        pl.when(allowed)
        .then(pl.col("current_sched_epoch") - pl.col("previous_arrival_inblock_epoch"))
        .otherwise(None)
        .alias("scheduled_turnaround_sec"),
        pl.when(allowed & pl.col("previous_arrival_stand").is_not_null() & pl.col("current_departure_stand").is_not_null())
        .then(pl.col("previous_arrival_stand") == pl.col("current_departure_stand"))
        .otherwise(None)
        .cast(pl.Int8)
        .alias("same_stand"),
    ).with_columns(
        (
            (pl.col("previous_inbound_arrival_delay_sec") > 900)
            & (pl.col("scheduled_turnaround_sec") < 3600)
        )
        .fill_null(False)
        .cast(pl.Int8)
        .alias("rotation_late_inbound_short_ground"),
        pl.when(pl.col("takeoff_minus_previous_inblock_sec") < 3600)
        .then(pl.lit("lt_1h"))
        .when(pl.col("takeoff_minus_previous_inblock_sec") < 7200)
        .then(pl.lit("1h_2h"))
        .when(pl.col("takeoff_minus_previous_inblock_sec") < 14400)
        .then(pl.lit("2h_4h"))
        .when(pl.col("takeoff_minus_previous_inblock_sec").is_not_null())
        .then(pl.lit("ge_4h"))
        .otherwise(None)
        .alias("turnaround_duration_bucket"),
    )
    keep = [
        "MVT_ID_mvt",
        "month",
        *ROTATION_NUMERIC_FEATURES,
        *ROTATION_CATEGORICAL_FEATURES,
        "current_departure_stand",
        "previous_arrival_stand",
        "arr_candidate_count",
        "arr_a_candidate_count",
        "intervening_same_aircraft_flight",
        "registration_inconsistent",
        "type_inconsistent",
    ]
    out = featured.select(keep)
    if out.height != challenge.height or out["MVT_ID_mvt"].n_unique() != out.height:
        raise RuntimeError("rotation sidecar lost or duplicated departures")
    return out
