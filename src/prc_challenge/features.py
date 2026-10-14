# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Deterministic features. Forbidden departure labels must already be masked."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import polars as pl

from prc_challenge.io import AIRPORT_TIMEZONES

CAT_FILL = "__UNK__"

LOCATION_CATS = (
    "AIRPORT",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_OPERATOR_flt",
    "WK_TBL_CAT_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt",
)


def add_reporting_airport(df: pl.DataFrame) -> pl.DataFrame:
    from prc_challenge.io import reporting_airport_expr

    if "AIRPORT" not in df.columns:
        df = df.with_columns(reporting_airport_expr())
    return df


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """Local hour / dow / month from takeoff, plus scheduled local hour when SCHED exists."""
    df = add_reporting_airport(df)
    parts = []
    remaining = df
    has_sched = "SCHED_TIME_UTC_mvt" in df.columns
    for icao, tz in AIRPORT_TIMEZONES.items():
        chunk = remaining.filter(pl.col("AIRPORT") == icao)
        remaining = remaining.filter(pl.col("AIRPORT") != icao)
        if chunk.height == 0:
            continue
        local = pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone(tz)
        exprs = [
            local.dt.hour().alias("local_hour"),
            local.dt.weekday().alias("local_dow"),
            local.dt.month().alias("local_month"),
        ]
        if has_sched:
            sched = pl.col("SCHED_TIME_UTC_mvt").dt.convert_time_zone(tz)
            exprs.append(sched.dt.hour().alias("sched_local_hour"))
        parts.append(chunk.with_columns(exprs))
    if remaining.height:
        utc = pl.col("MVT_TIME_UTC_mvt")
        exprs = [
            utc.dt.hour().alias("local_hour"),
            utc.dt.weekday().alias("local_dow"),
            utc.dt.month().alias("local_month"),
        ]
        if has_sched:
            exprs.append(pl.col("SCHED_TIME_UTC_mvt").dt.hour().alias("sched_local_hour"))
        parts.append(remaining.with_columns(exprs))
    if not parts:
        return df
    return pl.concat(parts, how="vertical")


def add_planning_features(df: pl.DataFrame) -> pl.DataFrame:
    """Permitted planning timestamps vs known takeoff. Does not use AOBT/BLOCK/TAXITIME."""
    mvt = pl.col("MVT_TIME_UTC_mvt")
    return df.with_columns(
        [
            (mvt - pl.col("IOBT_flt")).dt.total_seconds().alias("takeoff_minus_iobt_sec"),
            (mvt - pl.col("EOBT_1_flt")).dt.total_seconds().alias("takeoff_minus_eobt_sec"),
            (mvt - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("takeoff_minus_sched_sec"),
            (pl.col("EOBT_1_flt") - pl.col("IOBT_flt")).dt.total_seconds().alias("eobt_minus_iobt_sec"),
            (pl.col("SCHED_TIME_UTC_mvt") - pl.col("IOBT_flt")).dt.total_seconds().alias(
                "sched_minus_iobt_sec"
            ),
            pl.col("IOBT_flt").is_null().cast(pl.Int8).alias("iobt_missing"),
            pl.col("EOBT_1_flt").is_null().cast(pl.Int8).alias("eobt_missing"),
            pl.col("FLIGHT_ID_mvt").is_null().cast(pl.Int8).alias("nm_unmatched"),
            (pl.col("AIRCRAFT_TYPE_mvt").is_null()).cast(pl.Int8).alias("type_missing"),
            ((pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds() / 86400)
            .floor()
            .alias("takeoff_minus_sched_days"),
            ((pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds() % 86400).alias(
                "takeoff_minus_sched_mod_sec"
            ),
        ]
    )


LIRF_DELAY_AS_TAXI_SEC = 14400


def lirf_delay_as_taxi_gate() -> pl.Expr:
    """LIRF unmatched type-missing same-day takeoff with ≥4 h takeoff−SCHED.

    In that cell, recorded taxi ≈ delay (BLOCK stuck near SCHED). Ranking-safe:
    uses only AIRPORT, NM match, type missing, SCHED, takeoff. Not LFPG (there
    delay≥4 h has P(y≥2 h)=0). Does not use AOBT/BLOCK/TAXITIME.
    """
    return (
        (pl.col("AIRPORT") == "LIRF")
        & (pl.col("nm_unmatched") == 1)
        & (pl.col("type_missing") == 1)
        & (pl.col("takeoff_minus_sched_days") == 0)
        & (pl.col("takeoff_minus_sched_sec") >= LIRF_DELAY_AS_TAXI_SEC)
    )


def apply_lirf_delay_as_taxi(df: pl.DataFrame, pred_col: str = "prediction") -> pl.DataFrame:
    delay = pl.col("takeoff_minus_sched_sec").clip(lower_bound=0)
    return df.with_columns(
        pl.when(lirf_delay_as_taxi_gate()).then(delay).otherwise(pl.col(pred_col)).alias(pred_col)
    )


def lirf_extraday_as_taxi_gate() -> pl.Expr:
    """LIRF extra calendar day of takeoff−SCHED: recorded taxi ≈ delay.

    All three 2025 LIRF extra-day rows have y≥2 h and taxi≈delay. Other
    airports with extra-day≥1 have P(y≥2 h)=0 — do not copy. Ranking-safe:
    AIRPORT and SCHED vs takeoff only. Does not use AOBT/BLOCK/TAXITIME.
    """
    return (pl.col("AIRPORT") == "LIRF") & (pl.col("takeoff_minus_sched_days") >= 1)


def apply_lirf_extraday_as_taxi(df: pl.DataFrame, pred_col: str = "prediction") -> pl.DataFrame:
    delay = pl.col("takeoff_minus_sched_sec").clip(lower_bound=0)
    return df.with_columns(
        pl.when(lirf_extraday_as_taxi_gate()).then(delay).otherwise(pl.col(pred_col)).alias(pred_col)
    )


def apply_lirf_stuck_block_gates(df: pl.DataFrame, pred_col: str = "prediction") -> pl.DataFrame:
    """E17 same-day ≥4 h unmatched type-missing gate, then E18 extra-day gate."""
    return apply_lirf_extraday_as_taxi(apply_lirf_delay_as_taxi(df, pred_col), pred_col)


def lirf_leftover_killer_gate() -> pl.Expr:
    """LIRF unmatched type-missing extra-day=0 and not E17 (delay < 4 h).

    Mixed recording regime (E22/E26/E27). Ranking-safe: AIRPORT, NM match,
    type missing, SCHED vs takeoff. Disjoint from E17 and E18. Does not use
    AOBT, BLOCK, or TAXITIME.
    """
    return (
        (pl.col("AIRPORT") == "LIRF")
        & (pl.col("nm_unmatched") == 1)
        & (pl.col("type_missing") == 1)
        & (pl.col("takeoff_minus_sched_days") == 0)
        & (~lirf_delay_as_taxi_gate())
    )


def tail_stratum_expr() -> pl.Expr:
    """Disjoint error-cell labels for TabPFN tail experiments.

    Leftover and E17 win over generic unmatched-long. LIRF delay-variance is
    extra-day=0 matched with takeoff−SCHED ≥ 30 min and taxi < 1 h. Does not
    use AOBT, BLOCK, or predicted taxi.
    """
    y = pl.col("TAXITIME_SEC_mvt")
    return (
        pl.when(lirf_leftover_killer_gate())
        .then(pl.lit("leftover_killer"))
        .when(lirf_delay_as_taxi_gate())
        .then(pl.lit("e17"))
        .when(lirf_extraday_as_taxi_gate())
        .then(pl.lit("e18"))
        .when((pl.col("nm_unmatched") == 1) & (y >= 3600))
        .then(pl.lit("unmatched_y_ge_1h"))
        .when((pl.col("nm_unmatched") == 0) & (y >= 7200))
        .then(pl.lit("matched_y_ge_2h"))
        .when((pl.col("nm_unmatched") == 0) & (y >= 3600))
        .then(pl.lit("matched_y_1h_2h"))
        .when(
            (pl.col("AIRPORT") == "LIRF")
            & (pl.col("nm_unmatched") == 0)
            & (pl.col("takeoff_minus_sched_days") == 0)
            & (pl.col("takeoff_minus_sched_sec") >= 1800)
        )
        .then(pl.lit("lirf_matched_delay_var"))
        .when(
            (pl.col("AIRPORT") == "LIRF")
            & (pl.col("nm_unmatched") == 0)
            & (pl.col("takeoff_minus_sched_days") == 0)
        )
        .then(pl.lit("lirf_matched_ordinary"))
        .when(pl.col("nm_unmatched") == 0)
        .then(pl.lit("matched_ordinary"))
        .otherwise(pl.lit("unmatched_other"))
    )


def apply_leftover_parent_freeze(
    df: pl.DataFrame,
    parent_col: str = "parent_prediction",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Keep parent predictions on leftover-killer rows; use candidate elsewhere."""
    return df.with_columns(
        pl.when(lirf_leftover_killer_gate())
        .then(pl.col(parent_col))
        .otherwise(pl.col(pred_col))
        .alias(pred_col)
    )


def apply_unmatched_parent_freeze(
    df: pl.DataFrame,
    parent_col: str = "parent_prediction",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Keep parent predictions on NM-unmatched rows; use candidate on matched.

    Ranking-safe: FLIGHT_ID_mvt null. Covers leftover, E17, and unmatched-not-gated.
    """
    if "nm_unmatched" not in df.columns:
        df = add_planning_features(df)
    return df.with_columns(
        pl.when(pl.col("nm_unmatched") == 1)
        .then(pl.col(parent_col))
        .otherwise(pl.col(pred_col))
        .alias(pred_col)
    )


def lirf_matched_ordinary_gate() -> pl.Expr:
    """LIRF NM-matched same-day takeoff. Disjoint from E17 (unmatched) and E18 (extra-day)."""
    return (
        (pl.col("AIRPORT") == "LIRF")
        & (pl.col("nm_unmatched") == 0)
        & (pl.col("takeoff_minus_sched_days") == 0)
    )


def matched_ordinary_gate() -> pl.Expr:
    """NM-matched extra-day=0 at any airport. Disjoint from E17 and E18."""
    return (pl.col("nm_unmatched") == 0) & (pl.col("takeoff_minus_sched_days") == 0)


NM_EXTRA_NUMS = (
    "planned_block_sec",
    "takeoff_minus_arvt1_sec",
    "arvt1_missing",
    "dest_filed_mismatch",
)


def add_nm_extra_features(df: pl.DataFrame) -> pl.DataFrame:
    """Unused NM planning leftovers. Does not use AOBT, BLOCK, TAXITIME, or ARVT_3."""
    extras: list[pl.Expr] = [
        pl.col("ARVT_1_flt").is_null().cast(pl.Int8).alias("arvt1_missing"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("ARVT_1_flt")).dt.total_seconds().alias("takeoff_minus_arvt1_sec"),
    ]
    if "EOBT_1_flt" in df.columns:
        extras.append(
            (pl.col("ARVT_1_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().alias("planned_block_sec")
        )
    if "ADES_FILED_flt" in df.columns and "ADES_mvt" in df.columns:
        extras.append(
            (
                pl.col("ADES_FILED_flt").is_not_null()
                & pl.col("ADES_mvt").is_not_null()
                & (pl.col("ADES_FILED_flt") != pl.col("ADES_mvt"))
            )
            .cast(pl.Int8)
            .alias("dest_filed_mismatch")
        )
    return df.with_columns(extras)


AIRPORT_HOLIDAY_COUNTRY = {
    "EDDF": "DE",
    "EDDM": "DE",
    "EGLL": "GB",
    "EHAM": "NL",
    "LEBL": "ES",
    "LEMD": "ES",
    "LFPG": "FR",
    "LIRF": "IT",
    "LTFM": "TR",
    "LSZH": "CH",
}
HOLIDAY_YEARS = (2025, 2026)
HOLIDAY_NUMS = ("is_holiday", "is_holiday_eve", "is_holiday_next")


@lru_cache(maxsize=1)
def holiday_flag_table() -> pl.DataFrame:
    """Country-level public holidays for 2025–2026, keyed by airport local date."""
    import holidays as hd

    flags: dict[tuple[str, date], list[int]] = {}
    for icao, cc in AIRPORT_HOLIDAY_COUNTRY.items():
        for hol in hd.country_holidays(cc, years=HOLIDAY_YEARS):
            flags.setdefault((icao, hol), [0, 0, 0])[0] = 1
            eve = hol - timedelta(days=1)
            nxt = hol + timedelta(days=1)
            flags.setdefault((icao, eve), [0, 0, 0])[1] = 1
            flags.setdefault((icao, nxt), [0, 0, 0])[2] = 1
    if not flags:
        raise RuntimeError("holiday calendar empty")
    rows = [
        {
            "AIRPORT": icao,
            "local_date": d,
            "is_holiday": vals[0],
            "is_holiday_eve": vals[1],
            "is_holiday_next": vals[2],
        }
        for (icao, d), vals in flags.items()
    ]
    return pl.DataFrame(rows).with_columns(pl.col("local_date").cast(pl.Date))


def add_local_date(df: pl.DataFrame) -> pl.DataFrame:
    df = add_reporting_airport(df)
    parts = []
    remaining = df
    for icao, tz in AIRPORT_TIMEZONES.items():
        chunk = remaining.filter(pl.col("AIRPORT") == icao)
        remaining = remaining.filter(pl.col("AIRPORT") != icao)
        if chunk.height == 0:
            continue
        local = pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone(tz)
        parts.append(chunk.with_columns(local.dt.date().alias("local_date")))
    if remaining.height:
        parts.append(
            remaining.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("local_date"))
        )
    if not parts:
        return df
    return pl.concat(parts, how="vertical")


def add_holiday_features(df: pl.DataFrame) -> pl.DataFrame:
    """Public-holiday / adjacent-day flags from takeoff local date. No AOBT/BLOCK."""
    dated = add_local_date(df)
    out = dated.join(holiday_flag_table(), on=["AIRPORT", "local_date"], how="left")
    return out.with_columns(
        [pl.col(c).fill_null(0).cast(pl.Int8).alias(c) for c in HOLIDAY_NUMS]
    ).drop("local_date")


def prepare_model_frame(df: pl.DataFrame, include_planning: bool = False) -> pl.DataFrame:
    out = add_calendar_features(df)
    if include_planning:
        out = add_planning_features(out)
    for col in LOCATION_CATS:
        if col in out.columns:
            out = out.with_columns(pl.col(col).fill_null(CAT_FILL).cast(pl.String).alias(col))
    return out


def e03_feature_lists(include_planning: bool = False) -> tuple[list[str], list[str]]:
    cats = list(LOCATION_CATS)
    nums = ["local_hour", "local_dow", "local_month"]
    if include_planning:
        nums.extend(
            [
                "takeoff_minus_iobt_sec",
                "takeoff_minus_eobt_sec",
                "takeoff_minus_sched_sec",
                "eobt_minus_iobt_sec",
                "sched_minus_iobt_sec",
                "iobt_missing",
                "eobt_missing",
                "nm_unmatched",
                "type_missing",
                "takeoff_minus_sched_days",
                "takeoff_minus_sched_mod_sec",
            ]
        )
    return cats, nums


TAIL_NUMS = (
    "unmatched_type_missing",
    "takeoff_minus_sched_abs_days",
)


def add_flight_prefix(df: pl.DataFrame) -> pl.DataFrame:
    """2–3 letter prefix of `FLIGHT_mvt`. Recovers an operator proxy when NM is unmatched.

    Full callsign is high-cardinality identity (E04e skip). Does not use AOBT/BLOCK/TAXITIME.
    """
    return df.with_columns(
        pl.col("FLIGHT_mvt")
        .cast(pl.String)
        .str.to_uppercase()
        .str.extract(r"^([A-Z]{2,3})", 1)
        .fill_null(CAT_FILL)
        .alias("flight_prefix")
    )


def add_tail_features(df: pl.DataFrame) -> pl.DataFrame:
    """Ranking-safe disruption flags. Does not use AOBT/BLOCK/TAXITIME."""
    if "nm_unmatched" not in df.columns:
        df = add_planning_features(df)
    return df.with_columns(
        [
            (pl.col("nm_unmatched") * pl.col("type_missing")).cast(pl.Int8).alias("unmatched_type_missing"),
            pl.col("takeoff_minus_sched_days").abs().alias("takeoff_minus_sched_abs_days"),
        ]
    )


UNIMPEDED_NUMS = ("unimpeded_q10", "unimpeded_missing")


def unimpeded_tables(src: pl.DataFrame, q: float = 0.10) -> dict[str, pl.DataFrame]:
    """Train-only lower-quantile taxi-out lookups. Requires labels on `src` only."""
    labeled = src.drop_nulls("TAXITIME_SEC_mvt")
    y = pl.col("TAXITIME_SEC_mvt")
    return {
        "stand": labeled.group_by(["AIRPORT", "STAND_mvt"]).agg(y.quantile(q).alias("unimpeded_stand_q10")),
        "runway": labeled.group_by(["AIRPORT", "RUNWAY_mvt"]).agg(y.quantile(q).alias("unimpeded_rwy_q10")),
        "airport": labeled.group_by("AIRPORT").agg(y.quantile(q).alias("unimpeded_apt_q10")),
    }


def add_unimpeded_features(df: pl.DataFrame, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Join precomputed unimpeded lookups. Does not read focal TAXITIME/AOBT/BLOCK."""
    out = (
        df.join(tables["stand"], on=["AIRPORT", "STAND_mvt"], how="left")
        .join(tables["runway"], on=["AIRPORT", "RUNWAY_mvt"], how="left")
        .join(tables["airport"], on="AIRPORT", how="left")
    )
    q10 = pl.coalesce("unimpeded_stand_q10", "unimpeded_rwy_q10", "unimpeded_apt_q10")
    return out.with_columns(
        [
            q10.alias("unimpeded_q10"),
            q10.is_null().cast(pl.Int8).alias("unimpeded_missing"),
        ]
    ).drop(["unimpeded_stand_q10", "unimpeded_rwy_q10", "unimpeded_apt_q10"])


def e13_feature_lists() -> tuple[list[str], list[str]]:
    cats, nums = e03_feature_lists(include_planning=True)
    nums = nums + list(TAIL_NUMS)
    return cats, nums
