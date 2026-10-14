# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Target-independent OPDI physical-aircraft rotation linkage.

ICAO24 is used only as an aircraft identity key.  The functions in this module
reject departure labels, block times, model predictions, and residuals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from prc_challenge.opdi.ceiling import (
    FORBIDDEN_COLUMNS,
    assert_target_independent,
    learn_strict_callsign_map,
)

ROTATION_FORBIDDEN_COLUMNS = FORBIDDEN_COLUMNS | frozenset(
    {"parent_prediction", "r2_prediction", "model_prediction"}
)


def assert_rotation_target_independent(columns: list[str] | tuple[str, ...]) -> None:
    assert_target_independent(columns)
    bad = sorted(ROTATION_FORBIDDEN_COLUMNS.intersection(columns))
    if bad:
        raise ValueError(f"target-dependent columns are forbidden in rotation linkage: {bad}")


def temporal_callsign_bridge_validation(
    challenge: pl.DataFrame,
    *,
    train_month: int,
    evaluate_month: int,
    min_observations: int = 3,
    min_share: float = 0.90,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    """Train the IATA-route bridge in one month and evaluate it in another."""
    assert_rotation_target_independent(challenge.columns)
    train = challenge.filter(pl.col("month") == train_month)
    evaluate = challenge.filter(
        (pl.col("month") == evaluate_month)
        & pl.col("callsign_atc").is_not_null()
        & pl.col("flight_iata").is_not_null()
        & pl.col("adep").is_not_null()
        & pl.col("ades").is_not_null()
    ).with_columns(
        pl.col("flight_iata")
        .str.extract(r"^([A-Z]{2,3})", 1)
        .fill_null("UNKNOWN")
        .alias("flight_family")
    )
    mapping = learn_strict_callsign_map(
        train, min_observations=min_observations, min_share=min_share
    )
    scored = evaluate.join(
        mapping.select(
            "flight_iata",
            "adep",
            "ades",
            "inferred_callsign",
            "mapping_observations",
            "mapping_share",
        ),
        on=["flight_iata", "adep", "ades"],
        how="left",
    ).with_columns(
        pl.col("inferred_callsign").is_not_null().alias("bridge_covered"),
        (pl.col("callsign_atc") == pl.col("inferred_callsign"))
        .fill_null(False)
        .alias("bridge_correct"),
    )
    covered = scored.filter(pl.col("bridge_covered"))
    summary = {
        "train_month": train_month,
        "evaluate_month": evaluate_month,
        "min_observations": min_observations,
        "min_share": min_share,
        "n_train_rows": train.height,
        "n_mappings": mapping.height,
        "n_evaluable_known_callsigns": evaluate.height,
        "n_covered": covered.height,
        "coverage": covered.height / evaluate.height if evaluate.height else None,
        "n_correct": int(covered["bridge_correct"].sum()) if covered.height else 0,
        "precision": float(covered["bridge_correct"].mean()) if covered.height else None,
    }
    by_airport = (
        scored.group_by("adep")
        .agg(
            pl.len().alias("n_evaluable"),
            pl.col("bridge_covered").sum().alias("n_covered"),
            pl.col("bridge_correct").filter(pl.col("bridge_covered")).sum().alias("n_correct"),
        )
        .with_columns(
            (pl.col("n_covered") / pl.col("n_evaluable")).alias("coverage"),
            pl.when(pl.col("n_covered") > 0)
            .then(pl.col("n_correct") / pl.col("n_covered"))
            .otherwise(None)
            .alias("precision"),
            pl.lit(train_month).alias("train_month"),
            pl.lit(evaluate_month).alias("evaluate_month"),
        )
        .sort("adep")
    )
    failures = (
        covered.filter(~pl.col("bridge_correct"))
        .group_by("flight_family", "adep", "callsign_atc", "inferred_callsign")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    return summary, by_airport, failures


def _history_with_prefix(history: pl.DataFrame, prefix: str = "previous_") -> pl.DataFrame:
    rename = {column: f"{prefix}{column}" for column in history.columns}
    return history.rename(rename)


def _asof_excluding_current(
    current: pl.DataFrame,
    history: pl.DataFrame,
    *,
    left_by: list[str],
    right_by: list[str],
    prefix: str,
) -> pl.DataFrame:
    """Latest strict-prior row, retrying once when the as-of hit is the current OPDI row."""
    right = _history_with_prefix(history, prefix).sort(
        [*[f"{prefix}{c}" for c in right_by], f"{prefix}last_seen_epoch"]
    )
    left = current.with_columns((pl.col("takeoff_epoch") - 1).alias("_rotation_query_epoch")).sort(
        [*left_by, "_rotation_query_epoch"]
    )
    first = left.join_asof(
        right,
        left_on="_rotation_query_epoch",
        right_on=f"{prefix}last_seen_epoch",
        by_left=left_by,
        by_right=[f"{prefix}{c}" for c in right_by],
        strategy="backward",
    )
    self_hits = first.filter(pl.col(f"{prefix}opdi_id") == pl.col("current_opdi_id"))
    if self_hits.is_empty():
        return first.drop("_rotation_query_epoch")
    retry_left = self_hits.select(current.columns).with_columns(
        (pl.col("current_last_seen_epoch") - 1).alias("_rotation_query_epoch")
    ).sort([*left_by, "_rotation_query_epoch"])
    retry = retry_left.join_asof(
        right,
        left_on="_rotation_query_epoch",
        right_on=f"{prefix}last_seen_epoch",
        by_left=left_by,
        by_right=[f"{prefix}{c}" for c in right_by],
        strategy="backward",
    ).drop("_rotation_query_epoch")
    non_self = first.filter(
        (pl.col(f"{prefix}opdi_id") != pl.col("current_opdi_id"))
        | pl.col(f"{prefix}opdi_id").is_null()
    ).drop("_rotation_query_epoch")
    return pl.concat([non_self, retry], how="diagonal_relaxed")


def link_previous_aircraft_flights(
    current: pl.DataFrame,
    history: pl.DataFrame,
    *,
    max_lookback_seconds: int = 36 * 3600,
) -> pl.DataFrame:
    """Link each accepted current flight to its immediate prior ICAO24 flight.

    The latest prior flight is selected before airport filtering.  Therefore an
    intervening sector ending elsewhere invalidates an older apparent return to
    the current departure airport instead of being silently skipped.
    """
    assert_rotation_target_independent(current.columns)
    required = {
        "MVT_ID_mvt",
        "takeoff_epoch",
        "adep",
        "current_opdi_id",
        "current_icao24",
        "current_first_seen_epoch",
        "current_last_seen_epoch",
        "current_registration",
        "current_opdi_type",
    }
    missing = sorted(required.difference(current.columns))
    if missing:
        raise ValueError(f"current rotation frame missing columns: {missing}")
    history_required = {
        "opdi_id",
        "icao24",
        "callsign_norm",
        "opdi_adep",
        "opdi_ades",
        "opdi_ades_p",
        "registration",
        "opdi_type",
        "opdi_operator",
        "first_seen_epoch",
        "last_seen_epoch",
    }
    missing_history = sorted(history_required.difference(history.columns))
    if missing_history:
        raise ValueError(f"OPDI history missing columns: {missing_history}")

    eligible_current = current.filter(pl.col("current_icao24").is_not_null())
    clean_history = history.filter(
        pl.col("icao24").is_not_null() & pl.col("last_seen_epoch").is_not_null()
    )
    latest = _asof_excluding_current(
        eligible_current,
        clean_history,
        left_by=["current_icao24"],
        right_by=["icao24"],
        prefix="previous_",
    )

    destination_index = pl.concat(
        [
            clean_history.filter(pl.col("opdi_ades").is_not_null()).with_columns(
                pl.col("opdi_ades").alias("history_destination")
            ),
            clean_history.filter(pl.col("opdi_ades_p").is_not_null()).with_columns(
                pl.col("opdi_ades_p").alias("history_destination")
            ),
        ],
        how="vertical_relaxed",
    ).unique(["opdi_id", "history_destination"])
    latest_at_adep = _asof_excluding_current(
        eligible_current,
        destination_index,
        left_by=["current_icao24", "adep"],
        right_by=["icao24", "history_destination"],
        prefix="airport_previous_",
    ).select(
        "MVT_ID_mvt",
        pl.col("airport_previous_opdi_id").alias("latest_adep_previous_opdi_id"),
        pl.col("airport_previous_last_seen_epoch").alias("latest_adep_previous_last_seen_epoch"),
    )
    tie_counts = clean_history.group_by("icao24", "last_seen_epoch").agg(
        pl.len().alias("previous_end_tie_count")
    )
    out = (
        latest.join(latest_at_adep, on="MVT_ID_mvt", how="left")
        .join(
            tie_counts,
            left_on=["current_icao24", "previous_last_seen_epoch"],
            right_on=["icao24", "last_seen_epoch"],
            how="left",
        )
        .with_columns(
            (pl.col("takeoff_epoch") - pl.col("previous_last_seen_epoch")).alias(
                "seconds_since_previous_flight"
            ),
            pl.any_horizontal(
                (pl.col("previous_opdi_ades") == pl.col("adep")).fill_null(False),
                (pl.col("previous_opdi_ades_p") == pl.col("adep")).fill_null(False),
            ).alias("previous_ends_at_current_adep"),
            (
                pl.col("latest_adep_previous_opdi_id").is_not_null()
                & (pl.col("latest_adep_previous_opdi_id") != pl.col("previous_opdi_id"))
                & (
                    pl.col("latest_adep_previous_last_seen_epoch")
                    < pl.col("previous_last_seen_epoch")
                )
            )
            .fill_null(False)
            .alias("intervening_same_aircraft_flight"),
            (
                pl.col("previous_last_seen_epoch").is_not_null()
                & pl.col("current_first_seen_epoch").is_not_null()
                & (pl.col("previous_last_seen_epoch") >= pl.col("current_first_seen_epoch"))
            )
            .fill_null(False)
            .alias("overlapping_or_impossible"),
            (
                pl.col("current_registration").is_not_null()
                & pl.col("previous_registration").is_not_null()
                & (pl.col("current_registration") != pl.col("previous_registration"))
            )
            .fill_null(False)
            .alias("registration_inconsistent"),
            (
                pl.col("current_opdi_type").is_not_null()
                & pl.col("previous_opdi_type").is_not_null()
                & (pl.col("current_opdi_type") != pl.col("previous_opdi_type"))
            )
            .fill_null(False)
            .alias("type_inconsistent"),
        )
        .with_columns(
            pl.col("seconds_since_previous_flight")
            .is_between(1, max_lookback_seconds)
            .fill_null(False)
            .alias("previous_in_lookback"),
            (
                (pl.col("previous_end_tie_count").fill_null(0) > 1)
                | pl.col("overlapping_or_impossible")
            ).alias("rotation_ambiguous"),
        )
        .with_columns(
            (
                pl.col("previous_in_lookback")
                & pl.col("previous_ends_at_current_adep")
                & ~pl.col("intervening_same_aircraft_flight")
                & ~pl.col("rotation_ambiguous")
                & ~pl.col("registration_inconsistent")
                & ~pl.col("type_inconsistent")
            ).alias("rotation_high_confidence")
        )
    )
    return current.join(
        out.select(
            [
                "MVT_ID_mvt",
                *[c for c in out.columns if c.startswith("previous_")],
                "latest_adep_previous_opdi_id",
                "latest_adep_previous_last_seen_epoch",
                "seconds_since_previous_flight",
                "intervening_same_aircraft_flight",
                "overlapping_or_impossible",
                "registration_inconsistent",
                "type_inconsistent",
                "rotation_ambiguous",
                "rotation_high_confidence",
            ]
        ),
        on="MVT_ID_mvt",
        how="left",
    ).with_columns(
        pl.col("previous_in_lookback").fill_null(False),
        pl.col("previous_ends_at_current_adep").fill_null(False),
        pl.col("intervening_same_aircraft_flight").fill_null(False),
        pl.col("overlapping_or_impossible").fill_null(False),
        pl.col("registration_inconsistent").fill_null(False),
        pl.col("type_inconsistent").fill_null(False),
        pl.col("rotation_ambiguous").fill_null(False),
        pl.col("rotation_high_confidence").fill_null(False),
    )


def aircraft_history_aggregates(
    current: pl.DataFrame,
    history: pl.DataFrame,
) -> pl.DataFrame:
    """Compute prior-sector counts and durations without exposing raw ICAO24 as a feature."""
    assert_rotation_target_independent(current.columns)
    left = current.select(
        "MVT_ID_mvt",
        "current_icao24",
        "takeoff_epoch",
        "current_opdi_id",
        "current_first_seen_epoch",
        "current_last_seen_epoch",
    ).sort("current_icao24", "takeoff_epoch")
    right = history.filter(
        pl.col("icao24").is_not_null()
        & pl.col("last_seen_epoch").is_not_null()
        & pl.col("first_seen_epoch").is_not_null()
    ).select(
        "opdi_id",
        "icao24",
        "first_seen_epoch",
        "last_seen_epoch",
    ).sort("icao24", "last_seen_epoch")

    history_icao = right["icao24"].to_numpy()
    history_times = right["last_seen_epoch"].to_numpy()
    history_duration = (
        right["last_seen_epoch"].to_numpy() - right["first_seen_epoch"].to_numpy()
    ).clip(min=0)
    unique_icao, starts, counts = np.unique(
        history_icao, return_index=True, return_counts=True
    )
    ranges = {
        str(key): (int(start), int(start + count))
        for key, start, count in zip(unique_icao, starts, counts, strict=True)
    }
    result: dict[str, list[int | float]] = {
        "MVT_ID_mvt": [],
        "prior_flights_6h": [],
        "prior_flights_12h": [],
        "prior_flights_24h": [],
        "prior_duration_sec_12h": [],
        "prior_duration_sec_24h": [],
    }
    for row in left.iter_rows(named=True):
        result["MVT_ID_mvt"].append(row["MVT_ID_mvt"])
        bounds = ranges.get(str(row["current_icao24"]))
        if bounds is None:
            for column in result:
                if column != "MVT_ID_mvt":
                    result[column].append(0)
            continue
        start, end = bounds
        times = history_times[start:end]
        durations = history_duration[start:end]
        takeoff = int(row["takeoff_epoch"])
        stop = int(np.searchsorted(times, takeoff, side="left"))
        prefix = np.concatenate(([0], np.cumsum(durations[:stop], dtype=np.int64)))
        values: dict[int, tuple[int, int]] = {}
        for hours in (6, 12, 24):
            begin = int(np.searchsorted(times[:stop], takeoff - hours * 3600, side="left"))
            values[hours] = (stop - begin, int(prefix[stop] - prefix[begin]))

        current_last = row["current_last_seen_epoch"]
        current_first = row["current_first_seen_epoch"]
        if current_last is not None and current_first is not None and int(current_last) < takeoff:
            current_duration = max(0, int(current_last) - int(current_first))
            gap = takeoff - int(current_last)
            for hours in (6, 12, 24):
                if 0 < gap <= hours * 3600:
                    count, duration = values[hours]
                    values[hours] = (max(0, count - 1), max(0, duration - current_duration))

        result["prior_flights_6h"].append(values[6][0])
        result["prior_flights_12h"].append(values[12][0])
        result["prior_flights_24h"].append(values[24][0])
        result["prior_duration_sec_12h"].append(values[12][1])
        result["prior_duration_sec_24h"].append(values[24][1])
    return pl.DataFrame(result)
