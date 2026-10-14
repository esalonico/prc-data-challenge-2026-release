# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Target-independent helpers for estimating the OPDI linkage ceiling.

This module deliberately excludes taxi labels, block times, model predictions, and
residuals.  It keeps every challenge departure so that missing identifiers are
measured instead of disappearing during loading.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from prc_challenge.io import CHALLENGE_AIRPORTS
from prc_challenge.opdi.schema import normalize_callsign

FORBIDDEN_COLUMNS = frozenset(
    {
        "TAXITIME_SEC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "AOBT_3_flt",
        "taxi_label",
        "prediction",
        "residual",
    }
)

CHALLENGE_LINKAGE_COLUMNS = (
    "MVT_ID_mvt",
    "FLIGHT_ID_mvt",
    "FLIGHT_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "CALLSIGN_flt",
    "ADEP_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "AIRCRAFT_TYPE_flt",
    "AIRCRAFT_OPERATOR_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "LOBT_flt",
    "ARVT_1_flt",
)


def assert_target_independent(columns: Iterable[str]) -> None:
    bad = sorted(FORBIDDEN_COLUMNS.intersection(columns))
    if bad:
        raise ValueError(f"target-dependent columns are forbidden in linkage analysis: {bad}")


def _norm_text(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_uppercase()
        .replace({"": None, "NULL": None, "NAN": None, "NONE": None, "NA": None})
    )


def load_challenge_population(
    month_paths: dict[int, str | Path],
) -> pl.DataFrame:
    """Load all hub departures while excluding target and post-departure columns."""
    frames: list[pl.DataFrame] = []
    for month, raw_path in sorted(month_paths.items()):
        schema = pl.scan_parquet(raw_path).collect_schema()
        available = [c for c in CHALLENGE_LINKAGE_COLUMNS if c in schema.names()]
        assert_target_independent(available)
        frames.append(
            pl.scan_parquet(raw_path)
            .select(available)
            .filter(
                (pl.col("PHASE_mvt") == "DEP")
                & pl.col("ADEP_mvt").is_in(list(CHALLENGE_AIRPORTS))
            )
            .with_columns(pl.lit(month).cast(pl.Int8).alias("month"))
            .collect()
        )
    out = pl.concat(frames, how="vertical_relaxed").with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("flight_date"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("takeoff_epoch"),
        _norm_text("ADEP_mvt").alias("adep"),
        _norm_text("ADES_mvt").alias("ades"),
        _norm_text("CALLSIGN_flt")
        .map_elements(normalize_callsign, return_dtype=pl.Utf8)
        .alias("callsign_atc"),
        _norm_text("FLIGHT_mvt")
        .map_elements(normalize_callsign, return_dtype=pl.Utf8)
        .alias("flight_iata"),
        _norm_text("AIRCRAFT_TYPE_mvt").alias("aircraft_type_mvt_norm"),
        _norm_text("AIRCRAFT_TYPE_flt").alias("aircraft_type_flt_norm"),
        _norm_text("ADES_flt").alias("ades_flt_norm"),
        _norm_text("ADES_FILED_flt").alias("ades_filed_norm"),
        _norm_text("ADEP_flt").alias("adep_flt_norm"),
    )
    assert_target_independent(out.columns)
    return out


def load_opdi_flights(month_paths: dict[int, str | Path]) -> pl.DataFrame:
    """Load raw OPDI flight-list fields, including planned airports."""
    frames: list[pl.DataFrame] = []
    for month, path in sorted(month_paths.items()):
        schema = pl.scan_parquet(path).collect_schema()

        def as_datetime(name: str) -> pl.Expr:
            expr = pl.col(name)
            if schema[name] == pl.String:
                expr = expr.str.to_datetime(strict=False)
            return expr.cast(pl.Datetime("us"), strict=False)

        frames.append(
            pl.scan_parquet(path)
            .select(
                pl.col("id").cast(pl.UInt64).alias("opdi_id"),
                _norm_text("icao24").alias("icao24"),
                _norm_text("flt_id")
                .map_elements(normalize_callsign, return_dtype=pl.Utf8)
                .alias("callsign_norm"),
                as_datetime("dof").dt.date().alias("opdi_dof"),
                _norm_text("adep").alias("opdi_adep"),
                _norm_text("ades").alias("opdi_ades"),
                _norm_text("adep_p").alias("opdi_adep_p"),
                _norm_text("ades_p").alias("opdi_ades_p"),
                _norm_text("registration").alias("registration"),
                _norm_text("typecode").alias("opdi_type"),
                _norm_text("icao_operator").alias("opdi_operator"),
                as_datetime("first_seen").dt.epoch("s").cast(pl.Int64).alias("first_seen_epoch"),
                as_datetime("last_seen").dt.epoch("s").cast(pl.Int64).alias("last_seen_epoch"),
                pl.lit(month).cast(pl.Int8).alias("opdi_file_month"),
            )
            .collect()
        )
    return pl.concat(frames, how="vertical_relaxed")


def learn_strict_callsign_map(
    challenge: pl.DataFrame,
    *,
    min_observations: int = 3,
    min_share: float = 0.90,
) -> pl.DataFrame:
    """Learn a conservative IATA-flight/route to ATC-callsign bridge.

    The map only uses challenge rows with an NM callsign and is independent of
    taxi labels.  Route is included because the same public flight number can be
    reused on different services.
    """
    known = challenge.filter(
        pl.col("callsign_atc").is_not_null()
        & pl.col("flight_iata").is_not_null()
        & pl.col("adep").is_not_null()
        & pl.col("ades").is_not_null()
    )
    counts = known.group_by("flight_iata", "adep", "ades", "callsign_atc").agg(
        pl.len().alias("callsign_count")
    )
    totals = counts.group_by("flight_iata", "adep", "ades").agg(
        pl.col("callsign_count").sum().alias("mapping_observations"),
        pl.col("callsign_count").max().alias("modal_count"),
        pl.len().alias("n_callsigns"),
    )
    return (
        counts.join(totals, on=["flight_iata", "adep", "ades"], how="inner")
        .filter(
            (pl.col("callsign_count") == pl.col("modal_count"))
            & (pl.col("mapping_observations") >= min_observations)
            & ((pl.col("modal_count") / pl.col("mapping_observations")) >= min_share)
        )
        .sort(
            ["flight_iata", "adep", "ades", "callsign_atc"],
        )
        .group_by("flight_iata", "adep", "ades", maintain_order=True)
        .first()
        .rename({"callsign_atc": "inferred_callsign"})
        .with_columns(
            (pl.col("modal_count") / pl.col("mapping_observations")).alias("mapping_share")
        )
    )


def apply_callsign_bridge(challenge: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    return (
        challenge.join(mapping, on=["flight_iata", "adep", "ades"], how="left")
        .with_columns(
            pl.coalesce("callsign_atc", "inferred_callsign").alias("callsign_link"),
            pl.when(pl.col("callsign_atc").is_not_null())
            .then(pl.lit("atc"))
            .when(pl.col("inferred_callsign").is_not_null())
            .then(pl.lit("inferred_iata_route"))
            .otherwise(None)
            .alias("callsign_source"),
        )
    )


def build_candidate_universe(challenge: pl.DataFrame, opdi: pl.DataFrame) -> pl.DataFrame:
    """Generate callsign candidates across UTC date offsets -1/0/+1.

    Origin, route, type, and temporal checks are retained as flags so strategies
    can be compared without rebuilding the join.
    """
    left = challenge.select(
        "MVT_ID_mvt",
        "month",
        "flight_date",
        "takeoff_epoch",
        "adep",
        "ades",
        "ades_flt_norm",
        "ades_filed_norm",
        "aircraft_type_mvt_norm",
        "aircraft_type_flt_norm",
        "callsign_link",
        "callsign_source",
    ).filter(pl.col("callsign_link").is_not_null())
    joined = left.join(
        opdi.filter(pl.col("callsign_norm").is_not_null()),
        left_on="callsign_link",
        right_on="callsign_norm",
        how="inner",
    ).with_columns(
        (pl.col("opdi_dof") - pl.col("flight_date")).dt.total_days().cast(pl.Int16).alias(
            "date_delta_days"
        ),
        (pl.col("takeoff_epoch") - pl.col("first_seen_epoch")).alias("takeoff_minus_first_seen"),
        (pl.col("last_seen_epoch") - pl.col("takeoff_epoch")).alias("last_seen_minus_takeoff"),
    )
    route_values = [pl.col("ades"), pl.col("ades_flt_norm"), pl.col("ades_filed_norm")]
    type_values = [pl.col("aircraft_type_mvt_norm"), pl.col("aircraft_type_flt_norm")]
    return joined.filter(pl.col("date_delta_days").abs() <= 1).with_columns(
        (pl.col("opdi_adep") == pl.col("adep")).fill_null(False).alias("actual_adep_match"),
        (pl.col("opdi_adep_p") == pl.col("adep")).fill_null(False).alias("planned_adep_match"),
        pl.any_horizontal(
            [
                (op == challenge_dest).fill_null(False)
                for op in (pl.col("opdi_ades"), pl.col("opdi_ades_p"))
                for challenge_dest in route_values
            ]
        ).alias("route_match"),
        pl.any_horizontal(
            [(pl.col("opdi_type") == challenge_type).fill_null(False) for challenge_type in type_values]
        ).alias("type_match"),
        (
            pl.col("takeoff_minus_first_seen").is_between(-300, 14_400)
            & pl.col("last_seen_minus_takeoff").is_between(-180, 72_000)
        ).fill_null(False).alias("temporal_plausible"),
    )


def bridge_validation(challenge: pl.DataFrame, mapping: pl.DataFrame) -> dict[str, float | int | None]:
    """Leave-one-row-out precision indicator for the empirical callsign bridge."""
    known = challenge.filter(pl.col("callsign_atc").is_not_null()).join(
        mapping, on=["flight_iata", "adep", "ades"], how="inner"
    )
    eligible = known.with_columns(
        (
            pl.col("modal_count")
            - (pl.col("callsign_atc") == pl.col("inferred_callsign")).cast(pl.Int64)
        ).alias("loo_modal_count"),
        (pl.col("mapping_observations") - 1).alias("loo_observations"),
    ).filter(
        (pl.col("loo_observations") >= 2)
        & ((pl.col("loo_modal_count") / pl.col("loo_observations")) >= 0.90)
    )
    if eligible.is_empty():
        return {"n_known": known.height, "n_leave_one_out_eligible": 0, "accuracy": None}
    correct = int((eligible["callsign_atc"] == eligible["inferred_callsign"]).sum())
    return {
        "n_known": known.height,
        "n_leave_one_out_eligible": eligible.height,
        "correct": correct,
        "accuracy": correct / eligible.height,
    }
