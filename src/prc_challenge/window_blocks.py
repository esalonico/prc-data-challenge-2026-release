# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""M4 label-free feature blocks for NM-matched departures (SA-M4, 2026-09-25).

Three blocks, each keyed by ``MVT_ID_mvt`` (DEP rows only):

* **W (window)** — the organizer NM-matching window (hull of IOBT_flt/EOBT_1_flt
  +/- 3606 s, see :mod:`prc_challenge.matching_window`) as explicit features.
* **P (planned taxi)** — NM filed-plan residuals from ``ARVT_1_flt - EOBT_1_flt``
  against cross-fitted route medians.
* **C (cheap record/format signals)** — timestamp seconds, record agreement
  flags and the normalized service string.

Label hygiene: raw files are read through :func:`read_masked`, which selects a
**whitelist** of columns. BLOCK_TIME_UTC_mvt, TAXITIME_SEC_mvt, LOBT_flt,
AOBT_3_flt and ARVT_3_flt are never read. The builders also call
:func:`reject_forbidden` on their input, so a frame that still carries those
columns fails loudly. The P-block reference medians use only takeoff times,
landing times (ARR ``MVT_TIME_UTC_mvt``) and filed plan times of other flights;
no departure outcome of any flight enters any feature.

Month isolation: every per-row feature uses only the row itself, except the P
medians, which come from a *reference* set of months passed in explicitly
(leave-fold-out inside D for the screen, all training months for ranking).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from prc_challenge.io import CHALLENGE_AIRPORTS
from prc_challenge.matching_window import WINDOW_SEC

RAW_DIR = Path("data/raw")
FORBIDDEN = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "LOBT_flt", "AOBT_3_flt", "ARVT_3_flt")
WHITELIST = (
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
    "ADES_flt",
    "ADES_FILED_flt",
    "AIRCRAFT_TYPE_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
)
D_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)
D_FOLDS = ((2, 8), (3, 9), (4, 10), (5, 6))
MIN_KEY_ROWS = 20
C_CATS = ("c_service_ap", "c_carrier_ap")


# --------------------------------------------------------------------------- IO


def training_file(month: int, raw_dir: Path = RAW_DIR) -> Path:
    end = f"2025-{month + 1:02d}-01" if month < 12 else "2026-01-01"
    return Path(raw_dir) / f"training_2025-{month:02d}-01_{end}.parquet"


def reject_forbidden(frame: pl.DataFrame) -> None:
    bad = [c for c in FORBIDDEN if c in frame.columns]
    if bad:
        raise RuntimeError(f"M4 builders must not see departure-outcome / quarantined columns: {bad}")


def read_masked(path: Path) -> pl.DataFrame:
    """Read only whitelisted columns of one raw file (never BLOCK/TAXITIME/LOBT/AOBT_3/ARVT_3)."""

    avail = pl.read_parquet_schema(path)
    cols = [c for c in WHITELIST if c in avail]
    frame = pl.read_parquet(path, columns=cols)
    reject_forbidden(frame)
    return frame


def load_training_month(month: int, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return read_masked(training_file(month, raw_dir)).with_columns(pl.lit(month).cast(pl.Int8).alias("src_month"))


def load_ranking(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    frame = read_masked(Path(raw_dir) / "ranking.parquet")
    return frame.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8).alias("src_month"))


def _text(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.String, strict=False).str.to_uppercase().str.replace_all(r"\s+", "").replace("", None)


def _secs(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)


def departures(frame: pl.DataFrame) -> pl.DataFrame:
    reject_forbidden(frame)
    return frame.filter(pl.col("PHASE_mvt").cast(pl.String).str.to_uppercase() == "DEP")


# --------------------------------------------------------------------------- W


def build_window_block(frame: pl.DataFrame, variant: str = "v1") -> pl.DataFrame:
    """W block. Deterministic in the row's own MVT/IOBT/EOBT_1; no fitted parameter.

    v1: hull bounds, width, binding flag, signed slacks.
    v2: v1 + tail-resolution transforms (log1p of the floored bound, bound/lead share,
        EOBT-vs-IOBT source of the tight bound, capped upper bound).
    """

    dep = departures(frame)
    lead_i = _secs("MVT_TIME_UTC_mvt", "IOBT_flt")
    lead_e = _secs("MVT_TIME_UTC_mvt", "EOBT_1_flt")
    matched = pl.col("FLIGHT_ID_mvt").is_not_null() & pl.col("IOBT_flt").is_not_null() & pl.col("EOBT_1_flt").is_not_null()
    lead_min = pl.min_horizontal(lead_i, lead_e)
    lead_max = pl.max_horizontal(lead_i, lead_e)
    w = WINDOW_SEC

    def m(expr: pl.Expr) -> pl.Expr:
        return pl.when(matched).then(expr).otherwise(None)

    out = dep.select(
        "MVT_ID_mvt",
        m(pl.max_horizontal(lead_min - w, pl.lit(0.0))).alias("w_lo"),
        m(lead_max + w).alias("w_hi"),
        m(lead_i - w).alias("w_slack_iobt"),
        m(lead_e - w).alias("w_slack_eobt"),
        m(lead_min - w).alias("w_slack_hull"),
    ).with_columns(
        (pl.col("w_hi") - pl.col("w_lo")).alias("w_width"),
        (pl.col("w_lo") > 0).cast(pl.Int8).alias("w_binding"),
    )
    if variant == "v2":
        out = out.join(
            dep.select(
                "MVT_ID_mvt",
                m((lead_e < lead_i).cast(pl.Float64)).alias("w_lo_from_eobt"),
                m(lead_min).alias("__lead_min"),
            ),
            on="MVT_ID_mvt",
            how="left",
        ).with_columns(
            pl.col("w_lo").log1p().alias("w_lo_log1p"),
            (pl.col("w_lo") / pl.col("__lead_min").clip(lower_bound=1.0)).alias("w_lo_share"),
            pl.col("w_hi").clip(upper_bound=4 * 3600.0).alias("w_hi_cap4h"),
        ).drop("__lead_min")
    elif variant != "v1":
        raise ValueError(f"unknown W variant {variant}")
    return out


# --------------------------------------------------------------------------- C


_SERVICE_RE = r"^([A-Z]{3}|[A-Z0-9]{2})0*([0-9]+)([A-Z]*)$"


def normalize_service(col: str = "FLIGHT_mvt") -> pl.Expr:
    """Upper-case, strip blanks, drop leading zeros of the number: 'lh 01276' -> 'LH1276'."""

    txt = _text(col)
    parts = txt.str.extract_groups(_SERVICE_RE)
    rebuilt = pl.concat_str(
        [parts.struct.field("1"), parts.struct.field("2"), parts.struct.field("3")], separator=""
    )
    return pl.when(parts.struct.field("1").is_not_null()).then(rebuilt).otherwise(txt)


def build_record_block(frame: pl.DataFrame, variant: str = "v1") -> pl.DataFrame:
    """C block. Own-row record/format signals only.

    SCHED_TIME seconds are constant zero in every 2025 file and the ranking file,
    so they are not emitted (checked in the builder manifest).

    v1: numeric flags + service@airport and carrier@airport categoricals.
    v2: v1 numeric flags only (no high-cardinality categoricals).
    v3: v1 numeric flags + carrier@airport + a season-robust timetable-slot key
        ``AIRPORT:carrier:ADES:sched-UTC-hour`` instead of the service string (ATC
        alphanumeric callsigns are reassigned each IATA season; see the M4 report).
    """

    dep = departures(frame)
    flight = _text("FLIGHT_mvt")
    callsign = _text("CALLSIGN_flt")
    service = normalize_service("FLIGHT_mvt")
    carrier = flight.str.extract(r"^([A-Z]{3}|[A-Z0-9]{2})", 1)
    airport = _text("ADEP_mvt")
    sec = pl.col("MVT_TIME_UTC_mvt").dt.second().cast(pl.Float64)

    def neq(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return pl.when(a.is_null() | b.is_null()).then(None).otherwise((a != b).cast(pl.Float64))

    num = flight.str.extract(r"^(?:[A-Z]{3}|[A-Z0-9]{2})0*([0-9]+)", 1).cast(pl.Float64, strict=False)
    out = dep.select(
        "MVT_ID_mvt",
        sec.alias("c_mvt_sec"),
        (sec == 0).cast(pl.Int8).alias("c_mvt_sec0"),
        pl.when(flight.is_null() | callsign.is_null())
        .then(None)
        .otherwise((flight == callsign).cast(pl.Float64))
        .alias("c_flight_eq_callsign"),
        neq(_text("ADES_mvt"), _text("ADES_flt")).alias("c_ades_mm_flt"),
        neq(_text("ADES_mvt"), _text("ADES_FILED_flt")).alias("c_ades_mm_filed"),
        neq(_text("ADES_flt"), _text("ADES_FILED_flt")).alias("c_ades_flt_ne_filed"),
        neq(_text("AIRCRAFT_TYPE_mvt"), _text("AIRCRAFT_TYPE_flt")).alias("c_type_mm"),
        flight.str.contains(r"^[A-Z]{3}").cast(pl.Float64).alias("c_flight_icao3"),
        flight.str.contains(r"[0-9][A-Z]+$").cast(pl.Float64).alias("c_flight_alnum"),
        num.alias("c_flight_num"),
        pl.concat_str([airport, service], separator=":").alias("c_service_ap"),
        pl.concat_str([airport, carrier], separator=":").alias("c_carrier_ap"),
    )
    if variant == "v2":
        out = out.drop(list(C_CATS))
    elif variant == "v3":
        slot = pl.concat_str(
            [airport, carrier, _text("ADES_mvt"), pl.col("SCHED_TIME_UTC_mvt").dt.strftime("%H")], separator=":"
        )
        out = out.drop("c_service_ap").join(
            dep.select("MVT_ID_mvt", slot.alias("c_slot_h_ap")), on="MVT_ID_mvt", how="left"
        )
    elif variant != "v1":
        raise ValueError(f"unknown C variant {variant}")
    return out


# --------------------------------------------------------------------------- P


def _route_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Matched DEP rows with plan block, takeoff-minus-ARVT_1 and (when the ARR row exists) airborne time.

    The airborne proxy is landing (ARR ``MVT_TIME_UTC_mvt`` of the same NM flight in the
    same source window) minus takeoff; it exists only for challenge-airport destinations.
    """

    reject_forbidden(frame)
    phase = pl.col("PHASE_mvt").cast(pl.String).str.to_uppercase()
    arr = (
        frame.filter((phase == "ARR") & pl.col("FLIGHT_ID_mvt").is_not_null())
        .group_by("src_month", "FLIGHT_ID_mvt")
        .agg(pl.col("MVT_TIME_UTC_mvt").min().alias("__land"), pl.len().alias("__n_arr"))
        .filter(pl.col("__n_arr") == 1)
        .drop("__n_arr")
    )
    dep = frame.filter((phase == "DEP") & pl.col("FLIGHT_ID_mvt").is_not_null())
    dep = dep.join(arr, on=["src_month", "FLIGHT_ID_mvt"], how="left")
    air = _secs("__land", "MVT_TIME_UTC_mvt")
    return dep.select(
        "MVT_ID_mvt",
        "src_month",
        _text("ADEP_mvt").alias("k_adep"),
        _text("ADES_mvt").alias("k_ades"),
        pl.coalesce(_text("AIRCRAFT_TYPE_flt"), _text("AIRCRAFT_TYPE_mvt")).alias("k_type"),
        _secs("ARVT_1_flt", "EOBT_1_flt").alias("plan_block"),
        _secs("MVT_TIME_UTC_mvt", "ARVT_1_flt").alias("to_minus_arvt1"),
        # plausible airborne only (0.25 h .. 20 h); ARR rows of a different leg are dropped
        pl.when((air > 900) & (air < 72000)).then(air).otherwise(None).alias("air"),
    )


def _backoff_median(ref: pl.DataFrame, value: str, keys_fine: list[str], keys_coarse: list[str], name: str):
    """Return (fine, coarse) median tables with >= MIN_KEY_ROWS support."""

    src = ref.filter(pl.col(value).is_not_null())
    fine = (
        src.group_by(keys_fine)
        .agg(pl.col(value).median().alias(f"{name}_f"), pl.len().alias("__n"))
        .filter(pl.col("__n") >= MIN_KEY_ROWS)
        .drop("__n")
    )
    coarse = (
        src.group_by(keys_coarse)
        .agg(pl.col(value).median().alias(f"{name}_c"), pl.len().alias("__n"))
        .filter(pl.col("__n") >= MIN_KEY_ROWS)
        .drop("__n")
    )
    return fine, coarse


def fit_plan_reference(ref_routes: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Median tables from reference-month route rows (no departure labels involved)."""

    fine3 = ["k_adep", "k_ades", "k_type"]
    coarse2 = ["k_adep", "k_ades"]
    air_f, air_c = _backoff_median(ref_routes, "air", fine3, coarse2, "med_air")
    blk_f, blk_c = _backoff_median(ref_routes, "plan_block", fine3, coarse2, "med_blk")
    res_c = (
        ref_routes.filter(pl.col("to_minus_arvt1").is_not_null())
        .group_by(coarse2)
        .agg(pl.col("to_minus_arvt1").median().alias("med_tma"), pl.len().alias("__n"))
        .filter(pl.col("__n") >= MIN_KEY_ROWS)
        .drop("__n")
    )
    # airport-level typical NM taxi (level-1 residual) to put level-2 rows on the same scale
    lvl1 = _apply_air(ref_routes, air_f, air_c)
    ap_taxi = (
        lvl1.filter(pl.col("__base_air").is_not_null() & pl.col("plan_block").is_not_null())
        .group_by("k_adep")
        .agg((pl.col("plan_block") - pl.col("__base_air")).median().alias("med_ap_nm_taxi"))
    )
    return {"air_f": air_f, "air_c": air_c, "blk_f": blk_f, "blk_c": blk_c, "res_c": res_c, "ap_taxi": ap_taxi}


def _apply_air(routes: pl.DataFrame, air_f: pl.DataFrame, air_c: pl.DataFrame) -> pl.DataFrame:
    out = routes.join(air_f, on=["k_adep", "k_ades", "k_type"], how="left").join(air_c, on=["k_adep", "k_ades"], how="left")
    return out.with_columns(pl.coalesce("med_air_f", "med_air_c").alias("__base_air")).drop("med_air_f", "med_air_c")


def apply_plan_reference(routes: pl.DataFrame, ref: dict[str, pl.DataFrame], variant: str = "v1") -> pl.DataFrame:
    """P block for ``routes`` (output of :func:`_route_frame`) given reference medians.

    ``plan_nm_taxi``: level 1 = plan_block - airborne median (destination is a challenge
    airport: landing - takeoff proxy); level 2 = (plan_block - plan-block median of the
    route) + the airport's median level-1 NM taxi, so both levels share one scale.
    """

    out = _apply_air(routes, ref["air_f"], ref["air_c"])
    out = (
        out.join(ref["blk_f"], on=["k_adep", "k_ades", "k_type"], how="left")
        .join(ref["blk_c"], on=["k_adep", "k_ades"], how="left")
        .join(ref["res_c"], on=["k_adep", "k_ades"], how="left")
        .join(ref["ap_taxi"], on="k_adep", how="left")
        .with_columns(pl.coalesce("med_blk_f", "med_blk_c").alias("__base_blk"))
    )
    lvl1 = pl.col("plan_block") - pl.col("__base_air")
    dev = pl.col("plan_block") - pl.col("__base_blk")
    lvl2 = dev + pl.col("med_ap_nm_taxi")
    feats = out.select(
        "MVT_ID_mvt",
        pl.col("plan_block").alias("p_plan_block"),
        pl.coalesce(lvl1, lvl2).alias("plan_nm_taxi"),
        pl.when(lvl1.is_not_null()).then(1).when(lvl2.is_not_null()).then(2).otherwise(None).cast(pl.Float64).alias("p_nm_level"),
        dev.alias("p_plan_block_dev"),
        (pl.col("to_minus_arvt1") - pl.col("med_tma")).alias("plan_taxi_res"),
    )
    if variant == "v2":
        # own-flight airborne actual (challenge destinations only): plan_block - actual airborne
        feats = feats.join(
            out.select("MVT_ID_mvt", (pl.col("plan_block") - pl.col("air")).alias("p_nm_taxi_own_air")),
            on="MVT_ID_mvt",
            how="left",
        )
    elif variant != "v1":
        raise ValueError(f"unknown P variant {variant}")
    return feats


def build_plan_block(
    target: pl.DataFrame,
    reference: pl.DataFrame,
    variant: str = "v1",
) -> pl.DataFrame:
    """P block for all DEP rows of ``target`` using medians from ``reference`` rows.

    Both frames need ``src_month``. Unmatched DEP rows get nulls.
    """

    reject_forbidden(target)
    reject_forbidden(reference)
    ref = fit_plan_reference(_route_frame(reference))
    feats = apply_plan_reference(_route_frame(target), ref, variant)
    ids = departures(target).select("MVT_ID_mvt")
    return ids.join(feats, on="MVT_ID_mvt", how="left")


def build_plan_block_crossfit(frames: dict[int, pl.DataFrame], variant: str = "v1") -> pl.DataFrame:
    """Screen-safe P block for training months: D months use leave-fold-out D references,
    all other months use every D month present in ``frames``."""

    parts = []
    d_avail = [m for m in D_MONTHS if m in frames]
    for month, frame in sorted(frames.items()):
        if month in D_MONTHS:
            fold = next(f for f in D_FOLDS if month in f)
            ref_months = [m for m in d_avail if m not in fold]
        else:
            ref_months = d_avail
        if not ref_months:
            raise RuntimeError(f"no reference months for month {month}")
        reference = pl.concat([frames[m] for m in ref_months], how="vertical_relaxed")
        parts.append(build_plan_block(frame, reference, variant))
    return pl.concat(parts, how="vertical")


def challenge_destination() -> pl.Expr:
    return pl.col("ADES_mvt").is_in(list(CHALLENGE_AIRPORTS))
