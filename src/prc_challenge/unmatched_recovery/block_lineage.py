# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Hidden BLOCK source lattice, fingerprints, flight families, and legal links.

Training BLOCK/TAXITIME define the source-class *label* only. Deployable
features are ranking-visible. AOBT and ARVT_3 are forensic diagnostics and
never enter the feature graph, teacher targets, or candidate choice.
January, July, and V4 months are not development data.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from prc_challenge.features import CAT_FILL, add_calendar_features, add_flight_prefix, add_planning_features
from prc_challenge.io import reporting_airport_expr
from prc_challenge.unmatched_recovery.schema import BLOCKS, D_MONTHS, LOCKED_MONTHS, PRIMARY_MONTHS
from prc_challenge.unmatched_recovery.subgroups import flight_family_token

DAY_SEC = 86400.0
TOL_SEC = 60.0
EXACT_MATCH_SEC = 1.0
RADIUS_SEC = (0.0, 1.0, 60.0, 300.0, 900.0)
EXACT_OFFSETS_SEC = np.array([7200.0, 10800.0, 14400.0, 18000.0, 21600.0, 28800.0, 43200.0, 86400.0])
EXTREME_SEC = 7200.0
LONG_SEC = 3600.0
DAY_SCALE_SEC = 43200.0
SHRINK_N0_GRID = (5.0, 20.0, 50.0)
DEFAULT_N0 = 20.0
LEAVE_FAMILY_MOD = 5

CANDIDATE_FIELDS = ("schedule", "eobt", "iobt", "lobt")
CANDIDATE_CLASS = {
    "schedule": "schedule_copy",
    "eobt": "eobt_copy",
    "iobt": "iobt_copy",
    "lobt": "lobt_copy",
}
CLASSES = (
    "schedule_copy",
    "iobt_copy",
    "eobt_copy",
    "lobt_copy",
    "day_wrap_or_stale",
    "relation_copy",
    "exact_offset",
    "actual_like_unexplained",
    "ambiguous_multiple_candidates",
    "no_candidate",
)
DIRECT_COPY_CLASSES = frozenset({"schedule_copy", "iobt_copy", "eobt_copy", "lobt_copy", "day_wrap_or_stale", "ambiguous_multiple_candidates"})

FIELD_ROLE = {
    "MVT_ID_mvt": "movement_id",
    "FLIGHT_ID_mvt": "movement",
    "FLIGHT_mvt": "movement",
    "FLIGHT_RULE_mvt": "movement",
    "ADEP_mvt": "movement",
    "ADES_mvt": "movement",
    "PHASE_mvt": "movement",
    "MVT_TIME_UTC_mvt": "actual_takeoff_or_landing",
    "BLOCK_TIME_UTC_mvt": "hidden_departure_label",
    "SCHED_TIME_UTC_mvt": "planning",
    "AIRCRAFT_TYPE_mvt": "movement",
    "RUNWAY_mvt": "movement",
    "STAND_mvt": "movement",
    "TAXITIME_SEC_mvt": "hidden_departure_label",
    "LOBT_flt": "planning",
    "CALLSIGN_flt": "flight",
    "ADEP_flt": "flight",
    "ADES_flt": "flight",
    "ADES_FILED_flt": "flight",
    "MARKET_SEGMENT_flt": "flight",
    "IOBT_flt": "planning",
    "FLIGHT_RULE_flt": "flight",
    "FLIGHT_TYPE_flt": "flight",
    "AIRCRAFT_TYPE_flt": "flight",
    "WK_TBL_CAT_flt": "flight",
    "AIRCRAFT_OPERATOR_flt": "flight",
    "EOBT_1_flt": "planning",
    "ARVT_1_flt": "planning",
    "AOBT_3_flt": "forensic_only_quarantined",
    "ARVT_3_flt": "forensic_only_forbidden",
}

TIMESTAMP_FIELDS = (
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
)

RANKING_HIDDEN_ON_DEP = frozenset({"BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"})
FEATURE_FORBIDDEN = frozenset({"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt"})

CORE_CATS = ("AIRPORT", "FLIGHT_RULE_mvt")
CORE_NUMS = (
    "iobt_missing",
    "eobt_missing",
    "lobt_missing",
    "type_missing",
    "stand_missing",
    "runway_missing",
    "ades_missing",
    "flight_missing",
    "operator_missing",
    "sched_local_hour",
    "local_dow",
    "sched_second_zero",
    "mvt_second_zero",
    "arr_prior_n",
    "arr_prior_block_eq_sched_rate",
    "arr_prior_iobt_missing_rate",
    "arr_prior_second_zero_rate",
    "n_dest_arr",
    "dest_arr_unique",
    "linked_arr_block_eq_sched",
)
DELAY_EXTRA_NUMS = ("takeoff_minus_sched_sec",)


def assert_development_months(months: Sequence[int], *, context: str) -> None:
    bad = sorted({int(m) for m in months if int(m) not in D_MONTHS})
    if bad:
        raise RuntimeError(f"{context} used months outside D: {bad}")


def training_start_month(path: Path) -> int | None:
    match = re.search(r"training_(\d{4})-(\d{2})-01_", path.name)
    if match is None or int(match.group(1)) != 2025:
        return None
    return int(match.group(2))


def resolve_raw_dir(repo: Path) -> Path:
    local = repo / "data" / "raw"
    if local.exists():
        return local
    sibling = repo.parent / "prc-challenge-2026" / "data" / "raw"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(f"raw parquet not found from {repo}")


def month_paths(raw_dir: Path, months: Sequence[int]) -> list[Path]:
    """Resolve training files by start month. October's name contains the next month and must still match."""

    found: list[Path] = []
    for path in sorted(raw_dir.glob("training_2025-*.parquet")):
        month = training_start_month(path)
        if month in set(int(m) for m in months):
            found.append(path)
    if not found:
        raise FileNotFoundError(f"no training files for months {list(months)} in {raw_dir}")
    return found


def load_movements(raw_dir: Path, months: Sequence[int], phase: str) -> pl.DataFrame:
    """Load one phase. October is included via the file *start* month, not a name substring."""

    if set(int(m) for m in months) & set(LOCKED_MONTHS):
        raise RuntimeError("refusing V4 months")
    frames = [pl.scan_parquet(path) for path in month_paths(raw_dir, months)]
    frame = pl.concat(frames, how="vertical_relaxed").filter(pl.col("PHASE_mvt") == phase).collect()
    frame = frame.with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
        reporting_airport_expr(),
    )
    frame = frame.filter(pl.col("month").is_in([int(m) for m in months]))
    if frame.filter(pl.col("month").is_in(list(LOCKED_MONTHS))).height:
        raise RuntimeError("V4 rows survived the load filter")
    return frame


def _null_rate(col: str) -> pl.Expr:
    return pl.col(col).is_null().mean()


def inventory_field_rows(
    frame: pl.DataFrame,
    ranking: pl.DataFrame | None,
    *,
    population: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ranking_null = {}
    if ranking is not None and ranking.height:
        for name in frame.columns:
            if name in ranking.columns:
                ranking_null[name] = float(ranking[name].is_null().mean())
    for name, dtype in frame.schema.items():
        if name in {"month", "AIRPORT"}:
            continue
        null_rate = float(frame[name].is_null().mean()) if frame.height else float("nan")
        second_rate = float("nan")
        micro_rate = float("nan")
        if str(dtype).startswith("Datetime") and frame.height and frame[name].null_count() < frame.height:
            second_rate = float(frame[name].dt.second().eq(0).drop_nulls().mean() or 0.0)
            micro_rate = float(frame[name].dt.microsecond().eq(0).drop_nulls().mean() or 0.0)
        ranking_rate = ranking_null.get(name, float("nan"))
        rows.append(
            {
                "field": name,
                "dtype": str(dtype),
                "role": FIELD_ROLE.get(name, "other"),
                "population": population,
                "slice_type": "overall",
                "slice_value": "all",
                "n": int(frame.height),
                "null_rate": null_rate,
                "second_zero_rate": second_rate,
                "microsecond_zero_rate": micro_rate,
                "ranking_null_rate": ranking_rate,
                "ranking_available": bool(name not in RANKING_HIDDEN_ON_DEP and ranking_rate < 1.0)
                if name in ranking_null
                else name not in RANKING_HIDDEN_ON_DEP,
                "survives_on_unmatched": None,
            }
        )
    return rows


def raw_timestamp_inventory(dep: pl.DataFrame, ranking_dep: pl.DataFrame | None = None) -> pl.DataFrame:
    """Null rates and timestamp precision. BLOCK is described, never used as a feature."""

    unmatched = dep.filter(pl.col("FLIGHT_ID_mvt").is_null()) if "FLIGHT_ID_mvt" in dep.columns else dep.clear()
    matched = dep.filter(pl.col("FLIGHT_ID_mvt").is_not_null()) if "FLIGHT_ID_mvt" in dep.columns else dep.clear()
    rows = inventory_field_rows(dep, ranking_dep, population="all_dep")
    for row in rows:
        field = row["field"]
        if field in unmatched.columns and unmatched.height:
            row["survives_on_unmatched"] = bool(float(unmatched[field].is_null().mean()) < 0.999)
        else:
            row["survives_on_unmatched"] = False
    extra: list[dict[str, Any]] = []
    slices = [("unmatched", unmatched), ("matched", matched)]
    for label, part in slices:
        for row in inventory_field_rows(part, ranking_dep, population=label):
            row["slice_type"] = "matched_status"
            row["slice_value"] = label
            extra.append(row)
    if "AIRPORT" in dep.columns:
        for airport in sorted(dep["AIRPORT"].drop_nulls().unique().to_list()):
            part = dep.filter(pl.col("AIRPORT") == airport)
            for row in inventory_field_rows(part, None, population="airport"):
                row["slice_type"] = "airport"
                row["slice_value"] = airport
                extra.append(row)
    if "month" in dep.columns:
        for month in sorted(dep["month"].unique().to_list()):
            part = dep.filter(pl.col("month") == month)
            for row in inventory_field_rows(part, None, population="month"):
                row["slice_type"] = "month"
                row["slice_value"] = str(int(month))
                extra.append(row)
    return pl.DataFrame(rows + extra)


def _epoch(col: str) -> pl.Expr:
    return pl.col(col).dt.epoch("us").cast(pl.Float64) / 1_000_000.0


def attach_candidate_epochs(frame: pl.DataFrame) -> pl.DataFrame:
    """Continuous BLOCK-to-candidate gaps. AOBT/ARVT_3 stay diagnostic columns."""

    out = frame
    stamp = pl.Datetime(time_unit="us", time_zone="UTC")
    casts = []
    for name in ("IOBT_flt", "EOBT_1_flt", "LOBT_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"):
        if name in out.columns and out.schema[name] == pl.Null:
            casts.append(pl.col(name).cast(stamp))
    if casts:
        out = out.with_columns(casts)
    if "AIRPORT" not in out.columns:
        out = out.with_columns(reporting_airport_expr())
    if "nm_unmatched" not in out.columns:
        out = add_planning_features(out)
    else:
        needed = [c for c in ("takeoff_minus_sched_sec", "takeoff_minus_iobt_sec", "takeoff_minus_eobt_sec") if c not in out.columns]
        if needed:
            out = add_planning_features(out)
    if "flight_prefix" not in out.columns and "FLIGHT_mvt" in out.columns:
        out = add_flight_prefix(out)
    if "flight_family" not in out.columns and "FLIGHT_mvt" in out.columns:
        out = out.with_columns(flight_family_token())
    if "sched_local_hour" not in out.columns:
        out = add_calendar_features(out)
    exprs = [
        _epoch("BLOCK_TIME_UTC_mvt").alias("block_epoch"),
        _epoch("MVT_TIME_UTC_mvt").alias("mvt_epoch"),
        _epoch("SCHED_TIME_UTC_mvt").alias("epoch_schedule"),
        _epoch("EOBT_1_flt").alias("epoch_eobt"),
        _epoch("IOBT_flt").alias("epoch_iobt"),
        _epoch("LOBT_flt").alias("epoch_lobt"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().alias("takeoff_minus_lobt_sec"),
    ]
    if "AOBT_3_flt" in out.columns:
        exprs.append((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().abs().alias("diag_aobt_gap_sec"))
    if "ARVT_3_flt" in out.columns:
        exprs.append((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("ARVT_3_flt")).dt.total_seconds().abs().alias("diag_arvt3_gap_sec"))
    if "ARVT_1_flt" in out.columns:
        exprs.append((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("ARVT_1_flt")).dt.total_seconds().abs().alias("diag_arvt1_gap_sec"))
    out = out.with_columns(exprs)
    wrap_gaps = []
    for name in CANDIDATE_FIELDS:
        epoch = pl.col(f"epoch_{name}")
        block = pl.col("block_epoch")
        wrap_gaps.append(pl.min_horizontal((block - (epoch + DAY_SEC)).abs(), (block - (epoch - DAY_SEC)).abs()))
    out = out.with_columns(pl.min_horizontal(wrap_gaps).alias("gap_day_wrap"))
    flag_exprs = []
    for src, dest in (
        ("LOBT_flt", "lobt_missing"),
        ("FLIGHT_mvt", "flight_missing"),
        ("AIRCRAFT_OPERATOR_flt", "operator_missing"),
        ("RUNWAY_mvt", "runway_missing"),
        ("STAND_mvt", "stand_missing"),
        ("ADES_mvt", "ades_missing"),
    ):
        if src in out.columns and dest not in out.columns:
            flag_exprs.append(pl.col(src).is_null().cast(pl.Int8).alias(dest))
    if "SCHED_TIME_UTC_mvt" in out.columns:
        flag_exprs.append((pl.col("SCHED_TIME_UTC_mvt").dt.second() == 0).cast(pl.Int8).alias("sched_second_zero"))
    if "MVT_TIME_UTC_mvt" in out.columns:
        flag_exprs.append((pl.col("MVT_TIME_UTC_mvt").dt.second() == 0).cast(pl.Int8).alias("mvt_second_zero"))
    if "sched_local_hour" in out.columns:
        hour = pl.col("sched_local_hour")
        flag_exprs.append(
            pl.when(hour.is_null())
            .then(pl.lit("unk"))
            .when(hour < 6)
            .then(pl.lit("night"))
            .when(hour < 12)
            .then(pl.lit("morning"))
            .when(hour < 18)
            .then(pl.lit("afternoon"))
            .otherwise(pl.lit("evening"))
            .alias("sched_hour_bucket")
        )
    if "local_dow" in out.columns:
        flag_exprs.append(
            pl.when(pl.col("local_dow").is_null())
            .then(pl.lit("unk"))
            .when(pl.col("local_dow") >= 6)
            .then(pl.lit("weekend"))
            .otherwise(pl.lit("weekday"))
            .alias("weekday_class")
        )
    if flag_exprs:
        out = out.with_columns(flag_exprs)
    return out


def classify_block_sources(
    candidate_epoch: np.ndarray,
    block_epoch: np.ndarray,
    day_wrap_gap: np.ndarray,
    exact_gap: np.ndarray,
    *,
    tol: float = TOL_SEC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutually exclusive source classes from continuous gaps.

    Two candidates within ``tol`` of BLOCK are aliases when their timestamps
    also lie within ``tol`` of each other. They are ambiguous only when both
    match BLOCK and still disagree with each other. Precedence on a tie is
    column order: schedule, EOBT, IOBT, LOBT. The tolerance is the predeclared
    one-minute clock quantum, not a threshold fit on January/July.
    """

    epoch = np.asarray(candidate_epoch, dtype=np.float64)
    block = np.asarray(block_epoch, dtype=np.float64)
    if epoch.ndim != 2:
        raise ValueError("candidate_epoch must be (n, k)")
    n, k = epoch.shape
    gaps = np.abs(epoch - block[:, None])
    gaps[~np.isfinite(epoch) | ~np.isfinite(block)[:, None]] = np.inf
    finite_close = gaps <= tol
    n_close = finite_close.sum(axis=1).astype(np.int16)
    masked = np.where(finite_close, epoch, np.nan)
    span = np.zeros(n, dtype=np.float64)
    multi = n_close >= 2
    if np.any(multi):
        span[multi] = np.nanmax(masked[multi], axis=1) - np.nanmin(masked[multi], axis=1)
    ambiguous = (n_close >= 2) & (span > tol)
    best_idx = np.argmin(gaps, axis=1)
    best_gap = gaps[np.arange(n), best_idx]
    best_gap = np.where(np.isfinite(best_gap), best_gap, np.inf)
    names = np.array([CANDIDATE_CLASS[name] for name in CANDIDATE_FIELDS[:k]], dtype=object)
    any_finite = np.isfinite(epoch).any(axis=1) & np.isfinite(block)
    day_wrap = np.asarray(day_wrap_gap, dtype=np.float64)
    exact = np.asarray(exact_gap, dtype=np.float64)
    classes = np.full(n, "no_candidate", dtype=object)
    classes = np.where(any_finite, "actual_like_unexplained", classes)
    classes = np.where((best_gap > tol) & (exact <= tol), "exact_offset", classes)
    classes = np.where((best_gap > tol) & (day_wrap <= tol), "day_wrap_or_stale", classes)
    classes = np.where(best_gap <= tol, names[best_idx], classes)
    classes = np.where((best_gap <= tol) & ambiguous, "ambiguous_multiple_candidates", classes)
    reported_gap = np.where(np.isfinite(best_gap), best_gap, np.nan)
    return classes, reported_gap, n_close


def exact_offset_gap(mvt_epoch: np.ndarray, block_epoch: np.ndarray) -> np.ndarray:
    taxi = np.asarray(mvt_epoch, dtype=np.float64) - np.asarray(block_epoch, dtype=np.float64)
    return np.min(np.abs(taxi[:, None] - EXACT_OFFSETS_SEC[None, :]), axis=1)


def attach_source_class(frame: pl.DataFrame) -> pl.DataFrame:
    out = attach_candidate_epochs(frame)
    epoch = np.column_stack([out[f"epoch_{name}"].to_numpy().astype(np.float64) for name in CANDIDATE_FIELDS])
    block = out["block_epoch"].to_numpy().astype(np.float64)
    mvt = out["mvt_epoch"].to_numpy().astype(np.float64)
    classes, best_gap, n_close = classify_block_sources(
        epoch,
        block,
        out["gap_day_wrap"].to_numpy().astype(np.float64),
        exact_offset_gap(mvt, block),
    )
    sched_gap = np.abs(epoch[:, 0] - block)
    sched_gap = np.where(np.isfinite(epoch[:, 0]) & np.isfinite(block), sched_gap, np.nan)
    out = out.with_columns(
        pl.Series("source_class", classes),
        pl.Series("best_gap_sec", best_gap),
        pl.Series("n_close_candidates", n_close),
        pl.Series("gap_schedule", sched_gap),
        pl.Series("taxi_from_block", mvt - block),
    )
    return out


def apply_relation_class(frame: pl.DataFrame) -> pl.DataFrame:
    """Promote unexplained rows when a unique legal link reproduces BLOCK.

    Direct planning copies keep their class. Same-departure duplicates never
    promote, because that would copy the focal departure through another row.
    """

    if "diag_relation_gap_sec" not in frame.columns:
        return frame
    eligible = ~pl.col("source_class").is_in(list(DIRECT_COPY_CLASSES))
    unique = pl.col("n_dest_arr").fill_null(0) == 1
    close = pl.col("diag_relation_gap_sec") <= TOL_SEC
    leak = pl.col("same_departure_duplicate").fill_null(0) == 1 if "same_departure_duplicate" in frame.columns else pl.lit(False)
    return frame.with_columns(
        pl.when(eligible & unique & close & ~leak)
        .then(pl.lit("relation_copy"))
        .otherwise(pl.col("source_class"))
        .alias("source_class")
    )


def _slice_share(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns((pl.col("n") / pl.col("n").sum().over(["month", "AIRPORT", "nm_unmatched"])).alias("share_within_slice"))


def taxonomy_table(frame: pl.DataFrame) -> pl.DataFrame:
    work = frame.with_columns(
        [
            (pl.col("best_gap_sec") <= radius).cast(pl.Float64).alias(f"within_{int(radius)}")
            for radius in RADIUS_SEC
            if radius > 0 or True
        ]
    )
    grouped = work.group_by(["month", "AIRPORT", "nm_unmatched", "source_class"]).agg(
        pl.len().alias("n"),
        pl.col("best_gap_sec").median().alias("median_best_gap_sec"),
        (pl.col("best_gap_sec") <= EXACT_MATCH_SEC).mean().alias("exact_1s_rate"),
        (pl.col("best_gap_sec") <= TOL_SEC).mean().alias("within_60_rate"),
        (pl.col("gap_schedule") <= TOL_SEC).mean().alias("schedule_within_60_rate"),
        (pl.col("gap_schedule") <= 300).mean().alias("schedule_within_300_rate"),
        (pl.col("gap_schedule") <= EXACT_MATCH_SEC).mean().alias("schedule_exact_1s_rate"),
    )
    return _slice_share(grouped).sort(["month", "AIRPORT", "nm_unmatched", "source_class"])


def attach_family_keys(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    miss_cols = [
        c
        for c in (
            "iobt_missing",
            "eobt_missing",
            "lobt_missing",
            "operator_missing",
            "type_missing",
            "stand_missing",
            "runway_missing",
            "ades_missing",
            "flight_missing",
        )
        if c in out.columns
    ]
    if miss_cols:
        out = out.with_columns(
            pl.concat_str([pl.col(c).cast(pl.Int8).cast(pl.Utf8) for c in miss_cols], separator="").alias("missingness_bits")
        )
    else:
        out = out.with_columns(pl.lit("").alias("missingness_bits"))
    pieces = [
        pl.col("AIRPORT").fill_null(CAT_FILL),
        pl.col("FLIGHT_RULE_mvt").fill_null(CAT_FILL) if "FLIGHT_RULE_mvt" in out.columns else pl.lit(CAT_FILL),
        pl.col("flight_prefix").fill_null(CAT_FILL) if "flight_prefix" in out.columns else pl.lit(CAT_FILL),
        pl.col("sched_hour_bucket").fill_null("unk") if "sched_hour_bucket" in out.columns else pl.lit("unk"),
        pl.col("weekday_class").fill_null("unk") if "weekday_class" in out.columns else pl.lit("unk"),
        pl.col("missingness_bits"),
    ]
    out = out.with_columns(pl.concat_str(pieces, separator="|").alias("fingerprint"))
    flight = pl.col("FLIGHT_mvt").fill_null(CAT_FILL) if "FLIGHT_mvt" in out.columns else pl.lit(CAT_FILL)
    ades = pl.col("ADES_mvt").fill_null(CAT_FILL) if "ADES_mvt" in out.columns else pl.lit(CAT_FILL)
    hour = pl.col("sched_hour_bucket").fill_null("unk") if "sched_hour_bucket" in out.columns else pl.lit("unk")
    weekday = pl.col("weekday_class").fill_null("unk") if "weekday_class" in out.columns else pl.lit("unk")
    prefix = pl.col("flight_prefix").fill_null(CAT_FILL) if "flight_prefix" in out.columns else pl.lit(CAT_FILL)
    airport = pl.col("AIRPORT").fill_null(CAT_FILL)
    out = out.with_columns(
        airport.alias("key_airport"),
        pl.col("fingerprint"),
        pl.concat_str([prefix, ades, hour, weekday], separator="|").alias("key4"),
        pl.concat_str([airport, flight], separator="|").alias("key1"),
        pl.concat_str([airport, flight, ades], separator="|").alias("key2"),
        pl.concat_str([airport, flight, ades, hour], separator="|").alias("key3"),
    )
    return out


def fingerprint_summary(frame: pl.DataFrame, *, min_n: int = 30) -> pl.DataFrame:
    work = attach_family_keys(frame) if "fingerprint" not in frame.columns else frame
    grouped = (
        work.group_by("fingerprint")
        .agg(
            pl.len().alias("n"),
            (pl.col("nm_unmatched") == 1).sum().alias("n_unmatched"),
            (pl.col("source_class") == "schedule_copy").mean().alias("schedule_copy_rate"),
            pl.col("month").n_unique().alias("n_months"),
            pl.col("AIRPORT").n_unique().alias("n_airports"),
            pl.col("source_class").mode().first().alias("modal_class"),
        )
        .filter(pl.col("n") >= min_n)
        .sort("n", descending=True)
    )
    monthly = (
        work.group_by(["fingerprint", "month"])
        .agg((pl.col("source_class") == "schedule_copy").mean().alias("rate"), pl.len().alias("n"))
        .group_by("fingerprint")
        .agg(pl.col("rate").std().alias("schedule_copy_rate_std_across_months"))
    )
    return grouped.join(monthly, on="fingerprint", how="left")


def _class_count_exprs() -> list[pl.Expr]:
    return [(pl.col("source_class") == name).sum().alias(f"n_{name}") for name in CLASSES] + [pl.len().alias("n")]


def count_table(frame: pl.DataFrame, key: str) -> pl.DataFrame:
    return frame.group_by(key).agg(_class_count_exprs())


def hierarchical_probabilities(
    target: pl.DataFrame,
    source: pl.DataFrame,
    *,
    n0: float,
    exclude_self: bool,
) -> pl.DataFrame:
    """Shrink class probabilities coarse → fine. ``source`` is the only count pool."""

    if target.height == 0:
        return target
    drop_cols = [c for c in target.columns if c.startswith("p_fam_") or c in {"unseen_family", "_row"}]
    if drop_cols:
        target = target.drop(drop_cols)
    target = target.with_columns(pl.int_range(0, target.height).alias("_row"))
    levels = ("key_airport", "fingerprint", "key4", "key1", "key2", "key3")
    tables = {level: count_table(source, level) for level in levels}
    n = target.height
    global_counts = np.array([(source["source_class"] == name).sum() for name in CLASSES], dtype=np.float64)
    total = float(global_counts.sum())
    own = np.column_stack([(target["source_class"] == name).to_numpy() for name in CLASSES]).astype(np.float64)
    if exclude_self:
        denom = max(total - 1.0, 1.0)
        parent = (global_counts[None, :] - own) / denom
    else:
        parent = np.broadcast_to(global_counts / max(total, 1.0), (n, len(CLASSES))).copy()
    out = target
    unseen_exact = np.ones(n, dtype=np.int8)
    for level in levels:
        table = tables[level]
        rename = {f"n_{name}": f"{level}__n_{name}" for name in CLASSES}
        rename["n"] = f"{level}__n"
        joined = out.join(table.rename(rename), on=level, how="left").sort("_row")
        seen = joined[f"{level}__n"].is_not_null().to_numpy()
        raw_n = joined[f"{level}__n"].fill_null(0).to_numpy().astype(np.float64)
        raw_c = np.column_stack(
            [joined[f"{level}__n_{name}"].fill_null(0).to_numpy().astype(np.float64) for name in CLASSES]
        )
        if exclude_self:
            adj_n = np.where(seen, np.maximum(raw_n - 1.0, 0.0), 0.0)
            adj_c = np.where(seen[:, None], np.maximum(raw_c - own, 0.0), 0.0)
        else:
            adj_n = np.where(seen, raw_n, 0.0)
            adj_c = np.where(seen[:, None], raw_c, 0.0)
        parent = (adj_c + n0 * parent) / (adj_n[:, None] + n0)
        if level == "key3":
            unseen_exact = (~seen).astype(np.int8)
        out = joined.drop([f"{level}__n", *[f"{level}__n_{name}" for name in CLASSES]])
    prob_frame = pl.DataFrame({f"p_fam_{name}": parent[:, i] for i, name in enumerate(CLASSES)}).with_columns(
        pl.Series("unseen_family", unseen_exact)
    )
    if out.height != target.height:
        raise RuntimeError("shrinkage join changed row count")
    ordered = out.sort("_row").drop("_row")
    return ordered.hstack(prob_frame)


def _true_class_probability(frame: pl.DataFrame, prefix: str = "p_fam_") -> np.ndarray:
    idx = frame["source_class"].to_numpy()
    cols = np.column_stack([frame[f"{prefix}{name}"].to_numpy() for name in CLASSES])
    lookup = {name: i for i, name in enumerate(CLASSES)}
    chosen = np.array([lookup[str(label)] for label in idx], dtype=np.int32)
    return np.clip(cols[np.arange(cols.shape[0]), chosen], 1e-6, 1.0)


def multiclass_log_loss(frame: pl.DataFrame, prefix: str = "p_fam_") -> float:
    if frame.height == 0:
        return float("nan")
    return float(-np.mean(np.log(_true_class_probability(frame, prefix))))


def multiclass_brier(frame: pl.DataFrame, prefix: str = "p_fam_") -> float:
    if frame.height == 0:
        return float("nan")
    probs = np.column_stack([frame[f"{prefix}{name}"].to_numpy() for name in CLASSES])
    onehot = np.column_stack([(frame["source_class"] == name).to_numpy() for name in CLASSES]).astype(np.float64)
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def class_accuracy(frame: pl.DataFrame, prefix: str = "p_fam_") -> float:
    if frame.height == 0:
        return float("nan")
    probs = np.column_stack([frame[f"{prefix}{name}"].to_numpy() for name in CLASSES])
    pred = np.array(CLASSES, dtype=object)[np.argmax(probs, axis=1)]
    return float(np.mean(pred == frame["source_class"].to_numpy()))


def null_log_loss(source: pl.DataFrame, target: pl.DataFrame) -> float:
    if source.height == 0 or target.height == 0:
        return float("nan")
    rates = np.array([(source["source_class"] == name).mean() for name in CLASSES], dtype=np.float64)
    rates = np.clip(rates, 1e-6, 1.0)
    rates = rates / rates.sum()
    lookup = {name: rates[i] for i, name in enumerate(CLASSES)}
    p = np.array([lookup[str(label)] for label in target["source_class"].to_list()])
    return float(-np.mean(np.log(p)))


def family_recurrence_rows(frame: pl.DataFrame, *, population: str = "unmatched") -> list[dict[str, Any]]:
    """Blocked-month family shrinkage. Counts never include the holdout months."""

    work = frame.filter(pl.col("nm_unmatched") == 1) if population == "unmatched" else frame
    work = attach_family_keys(work) if "key3" not in work.columns else work
    rows: list[dict[str, Any]] = []
    if work.height == 0:
        return rows
    best_n0 = None
    best_loss = np.inf
    grid_loss: dict[float, list[float]] = {n0: [] for n0 in SHRINK_N0_GRID}
    for n0 in SHRINK_N0_GRID:
        for holdout in BLOCKS:
            fit = work.filter(~pl.col("month").is_in(list(holdout)))
            hold = work.filter(pl.col("month").is_in(list(holdout)))
            if fit.height == 0 or hold.height == 0:
                continue
            pred = hierarchical_probabilities(hold, fit, n0=n0, exclude_self=False)
            grid_loss[n0].append(multiclass_log_loss(pred))
        mean_loss = float(np.mean(grid_loss[n0])) if grid_loss[n0] else np.inf
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_n0 = n0
    chosen = float(best_n0 if best_n0 is not None else DEFAULT_N0)
    for holdout in BLOCKS:
        fit = work.filter(~pl.col("month").is_in(list(holdout)))
        hold = work.filter(pl.col("month").is_in(list(holdout)))
        if fit.height == 0 or hold.height == 0:
            continue
        pred = hierarchical_probabilities(hold, fit, n0=chosen, exclude_self=False)
        dropped = fit.filter((pl.col("key2").hash() % LEAVE_FAMILY_MOD) == 0)
        kept = fit.filter((pl.col("key2").hash() % LEAVE_FAMILY_MOD) != 0)
        lfo = hierarchical_probabilities(dropped, kept, n0=chosen, exclude_self=False) if dropped.height and kept.height else None
        rows.append(
            {
                "population": population,
                "holdout": ",".join(str(m) for m in holdout),
                "n_fit": int(fit.height),
                "n_hold": int(hold.height),
                "n0": chosen,
                "log_loss": multiclass_log_loss(pred),
                "null_log_loss": null_log_loss(fit, hold),
                "brier": multiclass_brier(pred),
                "accuracy": class_accuracy(pred),
                "unseen_family_rate": float(pred["unseen_family"].mean()),
                "leave_family_out_log_loss": multiclass_log_loss(lfo) if lfo is not None else float("nan"),
                "leave_family_out_n": int(lfo.height) if lfo is not None else 0,
                "schedule_copy_rate": float((hold["source_class"] == "schedule_copy").mean()),
                "selected": True,
            }
        )
    for n0, losses in grid_loss.items():
        rows.append(
            {
                "population": population,
                "holdout": "grid",
                "n_fit": int(work.height),
                "n_hold": int(work.height),
                "n0": float(n0),
                "log_loss": float(np.mean(losses)) if losses else float("nan"),
                "null_log_loss": float("nan"),
                "brier": float("nan"),
                "accuracy": float("nan"),
                "unseen_family_rate": float("nan"),
                "leave_family_out_log_loss": float("nan"),
                "leave_family_out_n": 0,
                "schedule_copy_rate": float("nan"),
                "selected": bool(n0 == chosen),
            }
        )
    return rows


def attach_dest_arr_links(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Unique same-flight destination arrival on the same UTC schedule date.

    The link is a different phase at the destination. Same-airport departure
    duplicates are flagged and cannot become ``relation_copy``.
    """

    dep_work = dep.with_columns(pl.col("SCHED_TIME_UTC_mvt").dt.date().alias("sched_date"))
    arr_work = arr.filter(pl.col("PHASE_mvt") == "ARR").with_columns(
        pl.col("SCHED_TIME_UTC_mvt").dt.date().alias("sched_date"),
        (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs().alias("arr_block_sched_gap"),
    )
    arr_keys = (
        arr_work.select(
            pl.col("MVT_ID_mvt").alias("arr_id"),
            "FLIGHT_mvt",
            "ADEP_mvt",
            "ADES_mvt",
            "sched_date",
            "arr_block_sched_gap",
            _epoch("SCHED_TIME_UTC_mvt").alias("arr_sched_epoch"),
            _epoch("BLOCK_TIME_UTC_mvt").alias("arr_block_epoch"),
            _epoch("IOBT_flt").alias("arr_iobt_epoch"),
            _epoch("EOBT_1_flt").alias("arr_eobt_epoch"),
            _epoch("LOBT_flt").alias("arr_lobt_epoch"),
        )
        .group_by(["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "sched_date"])
        .agg(
            pl.len().alias("n_arr_key"),
            pl.col("arr_id").first(),
            pl.col("arr_block_sched_gap").first(),
            pl.col("arr_sched_epoch").first(),
            pl.col("arr_block_epoch").first(),
            pl.col("arr_iobt_epoch").first(),
            pl.col("arr_eobt_epoch").first(),
            pl.col("arr_lobt_epoch").first(),
        )
    )
    linked = dep_work.join(arr_keys, on=["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "sched_date"], how="left")
    linked = linked.with_columns(
        pl.min_horizontal(
            (pl.col("block_epoch") - pl.col("arr_sched_epoch")).abs(),
            (pl.col("block_epoch") - pl.col("arr_block_epoch")).abs(),
            (pl.col("block_epoch") - pl.col("arr_iobt_epoch")).abs(),
            (pl.col("block_epoch") - pl.col("arr_eobt_epoch")).abs(),
            (pl.col("block_epoch") - pl.col("arr_lobt_epoch")).abs(),
        ).alias("diag_link_gap")
    )
    if linked.height != dep.height:
        raise RuntimeError("destination-arrival link changed the departure row count")
    linked = linked.with_columns(
        pl.col("n_arr_key").fill_null(0).alias("n_dest_arr"),
        pl.when(pl.col("n_arr_key").fill_null(0) == 1).then(pl.col("diag_link_gap")).otherwise(None).alias("diag_relation_gap_sec"),
        pl.when(pl.col("n_arr_key").fill_null(0) == 1).then(pl.col("arr_block_sched_gap")).otherwise(None).alias("linked_arr_block_sched_gap"),
        pl.when(pl.col("n_arr_key").fill_null(0) == 1).then(pl.col("arr_sched_epoch")).otherwise(None).alias("arr_sched_epoch_unique"),
    )
    dup = (
        dep_work.group_by(["AIRPORT", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"])
        .agg(pl.len().alias("n_same_sched_dep"))
    )
    out = linked.join(dup, on=["AIRPORT", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"], how="left")
    if out.height != dep.height:
        raise RuntimeError("same-departure duplicate join changed the departure row count")
    out = out.with_columns(
        (pl.col("n_same_sched_dep").fill_null(1) > 1).cast(pl.Int8).alias("same_departure_duplicate"),
        (pl.col("n_dest_arr").fill_null(0) == 1).cast(pl.Int8).alias("dest_arr_unique"),
        pl.when(pl.col("n_dest_arr").fill_null(0) == 1)
        .then((pl.col("linked_arr_block_sched_gap") <= TOL_SEC).cast(pl.Float64))
        .otherwise(None)
        .alias("linked_arr_block_eq_sched"),
        pl.when(pl.col("n_dest_arr").fill_null(0) == 1)
        .then(pl.col("arr_sched_epoch_unique"))
        .otherwise(None)
        .alias("arr_sched_epoch_unique"),
    )
    return apply_relation_class(out)


def attach_arr_prior_day(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Completed-ARR source health from the previous UTC day. No focal BLOCK."""

    arr_day = arr.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("arr_date")).group_by(["AIRPORT", "arr_date"]).agg(
        pl.len().alias("arr_prior_n"),
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= TOL_SEC)
        .mean()
        .alias("arr_prior_block_eq_sched_rate"),
        pl.col("IOBT_flt").is_null().mean().alias("arr_prior_iobt_missing_rate"),
        (pl.col("BLOCK_TIME_UTC_mvt").dt.second() == 0).mean().alias("arr_prior_second_zero_rate"),
    )
    out = dep.with_columns((pl.col("MVT_TIME_UTC_mvt").dt.date() - pl.duration(days=1)).alias("prior_date"))
    out = out.join(arr_day, left_on=["AIRPORT", "prior_date"], right_on=["AIRPORT", "arr_date"], how="left")
    if out.height != dep.height:
        raise RuntimeError("ARR prior-day join changed the departure row count")
    return out.drop("prior_date")


def relation_summary(frame: pl.DataFrame) -> pl.DataFrame:
    unmatched = frame.filter(pl.col("nm_unmatched") == 1)
    def _row(name: str, part: pl.DataFrame) -> dict[str, Any]:
        covered = part.filter(pl.col("n_dest_arr").fill_null(0) > 0) if "n_dest_arr" in part.columns else part.clear()
        ambiguous = covered.filter(pl.col("n_dest_arr") > 1) if covered.height else covered
        return {
            "relation": name,
            "population": "unmatched" if part is unmatched else "all_dep",
            "n": int(part.height),
            "coverage": float((part["n_dest_arr"].fill_null(0) > 0).mean()) if part.height and "n_dest_arr" in part.columns else 0.0,
            "ambiguity_given_link": float((covered["n_dest_arr"] > 1).mean()) if covered.height else float("nan"),
            "unique_link_rate": float((part["dest_arr_unique"] == 1).mean()) if part.height and "dest_arr_unique" in part.columns else 0.0,
            "relation_copy_rate": float((part["source_class"] == "relation_copy").mean()) if part.height else float("nan"),
            "same_departure_duplicate_rate": float(part["same_departure_duplicate"].mean()) if part.height and "same_departure_duplicate" in part.columns else float("nan"),
            "n_ambiguous": int(ambiguous.height) if ambiguous is not None else 0,
        }
    rows = [
        _row("dest_arr_same_flight_date", frame),
        _row("dest_arr_same_flight_date", unmatched),
    ]
    if "arr_prior_n" in unmatched.columns and unmatched.height:
        rows.append(
            {
                "relation": "completed_arr_prior_utc_day",
                "population": "unmatched",
                "n": int(unmatched.height),
                "coverage": float(unmatched["arr_prior_n"].is_not_null().mean()),
                "ambiguity_given_link": float("nan"),
                "unique_link_rate": float("nan"),
                "relation_copy_rate": float("nan"),
                "same_departure_duplicate_rate": float(unmatched["same_departure_duplicate"].mean()) if "same_departure_duplicate" in unmatched.columns else float("nan"),
                "n_ambiguous": 0,
            }
        )
    return pl.DataFrame(rows)


def aobt_forensic_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """Training-only AOBT agreement. Must not be read by feature selection."""

    if "diag_aobt_gap_sec" not in frame.columns:
        return pl.DataFrame({"note": ["AOBT column absent"]})
    return (
        frame.group_by(["nm_unmatched", "source_class"])
        .agg(
            pl.len().alias("n"),
            pl.col("diag_aobt_gap_sec").is_not_null().mean().alias("aobt_present_rate"),
            (pl.col("diag_aobt_gap_sec") <= TOL_SEC).mean().alias("block_within_60s_of_aobt"),
            pl.col("diag_aobt_gap_sec").median().alias("median_block_aobt_gap_sec"),
        )
        .sort(["nm_unmatched", "source_class"])
    )


def assert_feature_names(names: Sequence[str]) -> None:
    hit = FEATURE_FORBIDDEN.intersection(names)
    if hit:
        raise RuntimeError(f"feature graph includes quarantined columns: {sorted(hit)}")
    leaked = [name for name in names if name.startswith("diag_") or name in {"source_class", "best_gap_sec", "taxi_from_block", "block_epoch"}]
    if leaked:
        raise RuntimeError(f"feature graph includes label diagnostics: {leaked}")


def feature_names(kind: str) -> tuple[list[str], list[str]]:
    nums = list(CORE_NUMS) + [f"p_fam_{name}" for name in CLASSES]
    if kind == "delay":
        nums = nums + list(DELAY_EXTRA_NUMS)
    elif kind != "core":
        raise ValueError(kind)
    cats = list(CORE_CATS)
    assert_feature_names(cats + nums)
    return cats, nums
