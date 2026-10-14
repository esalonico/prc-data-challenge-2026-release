# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Target-independent bridge from a previous OPDI flight to a challenge ARR row."""

from __future__ import annotations

import polars as pl

from prc_challenge.opdi.rotation import assert_rotation_target_independent


def link_previous_arrivals(
    rotations: pl.DataFrame,
    arrivals: pl.DataFrame,
    *,
    tight_window_seconds: int = 15 * 60,
    wide_window_seconds: int = 90 * 60,
) -> pl.DataFrame:
    """Freeze ARR identity without taxi-in, in-block, or departure labels.

    ARR-A requires exactly one route-consistent candidate in the tight window.
    ARR-B requires exactly one plausible candidate in the wide window with route
    or type support. Multiple remaining candidates stay ambiguous.
    """
    assert_rotation_target_independent(rotations.columns)
    required_rotations = {
        "MVT_ID_mvt",
        "adep",
        "previous_callsign_norm",
        "previous_opdi_adep",
        "previous_opdi_ades",
        "previous_opdi_ades_p",
        "previous_opdi_type",
        "previous_last_seen_epoch",
        "rotation_high_confidence",
    }
    missing = sorted(required_rotations.difference(rotations.columns))
    if missing:
        raise ValueError(f"rotation frame missing ARR bridge columns: {missing}")
    required_arrivals = {
        "arr_mvt_id",
        "arr_airport",
        "arr_callsign",
        "arr_origin",
        "arr_type",
        "arr_mvt_epoch",
    }
    missing_arr = sorted(required_arrivals.difference(arrivals.columns))
    if missing_arr:
        raise ValueError(f"arrival frame missing identity columns: {missing_arr}")

    base = rotations.filter(pl.col("rotation_high_confidence"))
    candidates = (
        base.join(
            arrivals,
            left_on=["adep", "previous_callsign_norm"],
            right_on=["arr_airport", "arr_callsign"],
            how="inner",
        )
        .with_columns(
            (pl.col("arr_mvt_epoch") - pl.col("previous_last_seen_epoch")).alias(
                "arr_minus_previous_last_seen_sec"
            ),
            (
                pl.col("previous_opdi_adep").is_null()
                | pl.col("arr_origin").is_null()
                | (pl.col("previous_opdi_adep") == pl.col("arr_origin"))
            ).alias("arr_route_consistent"),
            (
                pl.col("previous_opdi_type").is_null()
                | pl.col("arr_type").is_null()
                | (pl.col("previous_opdi_type") == pl.col("arr_type"))
            ).alias("arr_type_consistent"),
            (
                pl.col("previous_opdi_adep").is_not_null()
                & pl.col("arr_origin").is_not_null()
                & (pl.col("previous_opdi_adep") == pl.col("arr_origin"))
            ).alias("arr_route_match"),
            (
                pl.col("previous_opdi_type").is_not_null()
                & pl.col("arr_type").is_not_null()
                & (pl.col("previous_opdi_type") == pl.col("arr_type"))
            ).alias("arr_type_match"),
        )
        .with_columns(pl.col("arr_minus_previous_last_seen_sec").abs().alias("arr_abs_delta_sec"))
        .filter(
            (pl.col("arr_abs_delta_sec") <= wide_window_seconds)
            & pl.col("arr_route_consistent")
            & pl.col("arr_type_consistent")
        )
        .with_columns(
            (
                (pl.col("arr_abs_delta_sec") <= tight_window_seconds)
                & pl.col("arr_route_match")
            ).alias("arr_a_candidate"),
            (pl.col("arr_route_match") | pl.col("arr_type_match")).alias("arr_b_supported"),
        )
    )
    if candidates.is_empty():
        return rotations.with_columns(
            pl.lit(None).cast(pl.Float64).alias("selected_arr_mvt_id"),
            pl.lit("unmatched").alias("arr_link_status"),
            pl.lit(0).cast(pl.UInt32).alias("arr_candidate_count"),
            pl.lit(0).cast(pl.UInt32).alias("arr_a_candidate_count"),
        )

    grouped = candidates.sort("MVT_ID_mvt", "arr_abs_delta_sec").group_by(
        "MVT_ID_mvt", maintain_order=True
    ).agg(
        pl.len().alias("arr_candidate_count"),
        pl.col("arr_a_candidate").sum().alias("arr_a_candidate_count"),
        pl.col("arr_b_supported").sum().alias("arr_supported_candidate_count"),
        pl.col("arr_mvt_id").filter(pl.col("arr_a_candidate")).first().alias("arr_a_mvt_id"),
        pl.col("arr_mvt_id").filter(pl.col("arr_b_supported")).first().alias("arr_b_mvt_id"),
    ).with_columns(
        pl.when(pl.col("arr_a_candidate_count") == 1)
        .then(pl.col("arr_a_mvt_id"))
        .when(
            (pl.col("arr_a_candidate_count") == 0)
            & (pl.col("arr_candidate_count") == 1)
            & (pl.col("arr_supported_candidate_count") == 1)
        )
        .then(pl.col("arr_b_mvt_id"))
        .otherwise(None)
        .alias("selected_arr_mvt_id"),
        pl.when(pl.col("arr_a_candidate_count") == 1)
        .then(pl.lit("ARR-A"))
        .when(
            (pl.col("arr_a_candidate_count") == 0)
            & (pl.col("arr_candidate_count") == 1)
            & (pl.col("arr_supported_candidate_count") == 1)
        )
        .then(pl.lit("ARR-B"))
        .otherwise(pl.lit("ambiguous"))
        .alias("arr_link_status"),
    )
    selected_details = candidates.join(
        grouped.select("MVT_ID_mvt", "selected_arr_mvt_id"),
        left_on=["MVT_ID_mvt", "arr_mvt_id"],
        right_on=["MVT_ID_mvt", "selected_arr_mvt_id"],
        how="inner",
    ).select(
        "MVT_ID_mvt",
        "arr_mvt_id",
        "arr_mvt_epoch",
        "arr_minus_previous_last_seen_sec",
        "arr_route_match",
        "arr_type_match",
    ).rename({"arr_mvt_id": "selected_arr_mvt_id"})
    return (
        rotations.join(grouped, on="MVT_ID_mvt", how="left")
        .join(selected_details, on=["MVT_ID_mvt", "selected_arr_mvt_id"], how="left")
        .with_columns(
            pl.col("arr_link_status").fill_null("unmatched"),
            pl.col("arr_candidate_count").fill_null(0),
            pl.col("arr_a_candidate_count").fill_null(0),
            pl.col("arr_supported_candidate_count").fill_null(0),
        )
    )
