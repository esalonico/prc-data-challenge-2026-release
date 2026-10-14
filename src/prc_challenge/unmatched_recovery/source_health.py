# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Airport movement source-health timelines and ranking-safe ARR sensors.

Rolling features are label-free: they never read departure TAXITIME/BLOCK.
Backward-only windows exclude the focal row and every later event.
ARR taxi-in / in-block are used only after the arrival has completed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl

from prc_challenge.campaign.scoring import assert_no_forbidden_features, assert_no_locked_months
from prc_challenge.features import CAT_FILL, add_flight_prefix
from prc_challenge.io import CHALLENGE_AIRPORTS, RAW_DIR, ranking_path, training_paths
from prc_challenge.unmatched_recovery.schema import D_MONTHS, LOCKED_MONTHS, present
from prc_challenge.unmatched_recovery.subgroups import assert_design_months, flight_family_token

COUNT_WINDOWS = (5, 10, 20)
TIME_WINDOWS_SEC = (900, 1800, 3600, 7200, 21600)
TIME_WINDOW_NAMES = {
    900: "15m",
    1800: "30m",
    3600: "1h",
    7200: "2h",
    21600: "6h",
}
ARR_WINDOWS_SEC = (1800, 3600, 10800, 21600)
ARR_WINDOW_NAMES = {1800: "30m", 3600: "1h", 10800: "3h", 21600: "6h"}
LAG_DAYS = (1,)

TIMELINE_RAW_COLS = (
    "MVT_ID_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "AIRCRAFT_TYPE_mvt",
    "STAND_mvt",
    "RUNWAY_mvt",
    "FLIGHT_mvt",
    "FLIGHT_ID_mvt",
    "FLIGHT_RULE_mvt",
    "TAXITIME_SEC_mvt",
)

RANKING_AUDIT_FIELDS = (
    "TAXITIME_SEC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "MVT_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "STAND_mvt",
    "RUNWAY_mvt",
    "FLIGHT_mvt",
    "FLIGHT_ID_mvt",
    "ADES_mvt",
    "ADEP_mvt",
    "FLIGHT_RULE_mvt",
)

SOURCE_HEALTH_NUMS = (
    "sh_unmatched_streak",
    "sh_sec_since_matched",
    "sh_sec_since_unmatched",
    "sh_sec_since_prev",
    *[f"sh_n_k{k}" for k in COUNT_WINDOWS],
    *[f"sh_unmatched_rate_k{k}" for k in COUNT_WINDOWS],
    *[f"sh_type_miss_rate_k{k}" for k in COUNT_WINDOWS],
    *[f"sh_stand_miss_rate_k{k}" for k in COUNT_WINDOWS],
    *[f"sh_rwy_miss_rate_k{k}" for k in COUNT_WINDOWS],
    *[f"sh_sched_miss_rate_k{k}" for k in COUNT_WINDOWS],
    *[f"sh_flight_miss_rate_k{k}" for k in COUNT_WINDOWS],
    *[f"sh_ades_miss_rate_k{k}" for k in COUNT_WINDOWS],
    *[f"sh_completeness_k{k}" for k in COUNT_WINDOWS],
    *[f"sh_prefix_entropy_k{k}" for k in COUNT_WINDOWS],
    *[f"sh_n_{TIME_WINDOW_NAMES[w]}" for w in TIME_WINDOWS_SEC],
    *[f"sh_unmatched_rate_{TIME_WINDOW_NAMES[w]}" for w in TIME_WINDOWS_SEC],
    *[f"sh_type_miss_rate_{TIME_WINDOW_NAMES[w]}" for w in TIME_WINDOWS_SEC],
    *[f"sh_stand_miss_rate_{TIME_WINDOW_NAMES[w]}" for w in TIME_WINDOWS_SEC],
    *[f"sh_rwy_miss_rate_{TIME_WINDOW_NAMES[w]}" for w in TIME_WINDOWS_SEC],
    *[f"sh_completeness_{TIME_WINDOW_NAMES[w]}" for w in TIME_WINDOWS_SEC],
    *[f"sh_n_dep_{TIME_WINDOW_NAMES[w]}" for w in TIME_WINDOWS_SEC],
    *[f"sh_n_arr_{TIME_WINDOW_NAMES[w]}" for w in TIME_WINDOWS_SEC],
    "sh_n_dtd",
    "sh_unmatched_rate_dtd",
    "sh_completeness_dtd",
    "sh_n_dep_dtd",
    "sh_n_arr_dtd",
    "fam_recent_n_6h",
    "fam_days_since_prev",
    "fam_unmatched_rate_hist",
    "fam_hour_std_hist",
    "fam_dest_nunique_hist",
    "fam_type_nunique_hist",
)

ARR_HEALTH_NUMS = (
    *[f"arr_n_{ARR_WINDOW_NAMES[w]}" for w in ARR_WINDOWS_SEC],
    *[f"arr_taxi_median_{ARR_WINDOW_NAMES[w]}" for w in ARR_WINDOWS_SEC],
    *[f"arr_taxi_p90_{ARR_WINDOW_NAMES[w]}" for w in ARR_WINDOWS_SEC],
    *[f"arr_taxi_std_{ARR_WINDOW_NAMES[w]}" for w in ARR_WINDOWS_SEC],
    *[f"arr_delay_median_{ARR_WINDOW_NAMES[w]}" for w in ARR_WINDOWS_SEC],
    *[f"arr_block_eq_sched_rate_{ARR_WINDOW_NAMES[w]}" for w in ARR_WINDOWS_SEC],
    *[f"arr_type_miss_rate_{ARR_WINDOW_NAMES[w]}" for w in ARR_WINDOWS_SEC],
    *[f"arr_completeness_{ARR_WINDOW_NAMES[w]}" for w in ARR_WINDOWS_SEC],
    "arr_age_sec",
    "arr_n_dtd",
    "arr_taxi_median_dtd",
    "arr_n_prior_day",
    "arr_taxi_median_prior_day",
)

RETRO_NUMS = (
    "retro_unmatched_rate_dtd",
    "retro_completeness_dtd",
    "retro_n_dtd",
)


def _is_missing_expr(col: str) -> pl.Expr:
    return (
        pl.col(col).is_null()
        | (pl.col(col).cast(pl.String).str.strip_chars() == "")
        | pl.col(col).cast(pl.String).is_in([CAT_FILL, "None", "null"])
    )


def source_health_feature_names(*, include_arr: bool = True, include_retro: bool = False) -> list[str]:
    names = list(SOURCE_HEALTH_NUMS)
    if include_arr:
        names.extend(ARR_HEALTH_NUMS)
    if include_retro:
        names.extend(RETRO_NUMS)
    return names


def assert_source_health_features(names: Sequence[str]) -> None:
    forbidden = {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt"}
    hit = forbidden.intersection(names)
    if hit:
        raise RuntimeError(f"source-health features include forbidden labels: {sorted(hit)}")
    assert_no_forbidden_features(list(names))


def month_isolated_training_paths(repo: Path, months: Sequence[int] = tuple(range(1, 11))) -> list[Path]:
    raw_dir = repo / RAW_DIR if (repo / RAW_DIR).exists() else Path(RAW_DIR)
    keep = set(int(m) for m in months)
    blocked = set(LOCKED_MONTHS)
    out = []
    for path in training_paths(raw_dir):
        name = path.name
        if "2025-11-01" in name or "2025-12-01" in name:
            continue
        frame_month = None
        for month in range(1, 13):
            token = f"2025-{month:02d}-01"
            if token in name:
                frame_month = month
                break
        if frame_month is None or frame_month in blocked or frame_month not in keep:
            continue
        out.append(path)
    if not out:
        raise FileNotFoundError(f"no month-isolated training files in {raw_dir}")
    return out


def load_movement_timeline(
    repo: Path,
    *,
    months: Sequence[int] = tuple(range(1, 11)),
    dep_unmatched: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Load DEP+ARR challenge movements for source-health. DEP taxi/block are dropped."""

    paths = month_isolated_training_paths(repo, months)
    parts: list[pl.DataFrame] = []
    for path in paths:
        lf = pl.scan_parquet(path)
        names = lf.collect_schema().names()
        cols = [c for c in TIMELINE_RAW_COLS if c in names]
        frame = (
            lf.select(cols)
            .filter(
                pl.col("PHASE_mvt").is_in(["DEP", "ARR"])
                & (
                    ((pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(list(CHALLENGE_AIRPORTS)))
                    | ((pl.col("PHASE_mvt") == "ARR") & pl.col("ADES_mvt").is_in(list(CHALLENGE_AIRPORTS)))
                )
            )
            .collect()
        )
        parts.append(frame)
    raw = pl.concat(parts, how="vertical_relaxed") if parts else pl.DataFrame()
    if raw.height == 0:
        raise RuntimeError("empty movement timeline")
    airport = (
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt"))
        .otherwise(pl.col("ADES_mvt"))
        .str.to_uppercase()
        .alias("AIRPORT")
    )
    timeline = raw.with_columns(
        airport,
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("epoch"),
        pl.when(pl.col("PHASE_mvt") == "ARR")
        .then(pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64))
        .otherwise(None)
        .alias("inblock_epoch"),
        (pl.col("PHASE_mvt") == "DEP").cast(pl.Int8).alias("is_dep"),
        (pl.col("PHASE_mvt") == "ARR").cast(pl.Int8).alias("is_arr"),
        pl.col("FLIGHT_ID_mvt").is_null().cast(pl.Int8).alias("id_missing"),
        _is_missing_expr("AIRCRAFT_TYPE_mvt").cast(pl.Int8).alias("type_missing_tl"),
        _is_missing_expr("STAND_mvt").cast(pl.Int8).alias("stand_missing_tl"),
        _is_missing_expr("RUNWAY_mvt").cast(pl.Int8).alias("rwy_missing_tl"),
        _is_missing_expr("SCHED_TIME_UTC_mvt").cast(pl.Int8).alias("sched_missing_tl"),
        _is_missing_expr("FLIGHT_mvt").cast(pl.Int8).alias("flight_missing_tl"),
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(_is_missing_expr("ADES_mvt"))
        .otherwise(_is_missing_expr("ADEP_mvt"))
        .cast(pl.Int8)
        .alias("ades_missing_tl"),
        pl.when(pl.col("PHASE_mvt") == "ARR")
        .then(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
        .otherwise(None)
        .alias("arr_taxi_sec"),
        pl.when(pl.col("PHASE_mvt") == "ARR")
        .then(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64)
        )
        .otherwise(None)
        .alias("arr_mvt_minus_sched_sec"),
        pl.when(pl.col("PHASE_mvt") == "ARR")
        .then(
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
            .dt.total_seconds()
            .abs()
            .cast(pl.Float64)
        )
        .otherwise(None)
        .alias("arr_abs_block_minus_sched_sec"),
    )
    if timeline["month"].is_in(list(LOCKED_MONTHS)).any():
        raise RuntimeError("V4 months leaked into movement timeline")
    if "FLIGHT_mvt" in timeline.columns:
        timeline = add_flight_prefix(timeline)
        timeline = timeline.with_columns(flight_family_token())
    else:
        timeline = timeline.with_columns(
            pl.lit(CAT_FILL).alias("flight_prefix"),
            pl.lit(CAT_FILL).alias("flight_family"),
        )
    unmatched = pl.col("id_missing")
    if dep_unmatched is not None and "MVT_ID_mvt" in dep_unmatched.columns:
        flag = dep_unmatched.select(
            "MVT_ID_mvt",
            pl.col("nm_unmatched").cast(pl.Int8).alias("_cache_unmatched"),
        )
        timeline = timeline.join(flag, on="MVT_ID_mvt", how="left")
        unmatched = pl.coalesce([pl.col("_cache_unmatched"), pl.col("id_missing")])
    timeline = timeline.with_columns(unmatched.cast(pl.Int8).alias("tl_unmatched"))
    if "_cache_unmatched" in timeline.columns:
        timeline = timeline.drop("_cache_unmatched")
    miss_cols = [
        "type_missing_tl",
        "stand_missing_tl",
        "rwy_missing_tl",
        "sched_missing_tl",
        "flight_missing_tl",
        "ades_missing_tl",
    ]
    completeness = 1.0 - pl.mean_horizontal([pl.col(c).cast(pl.Float64) for c in miss_cols])
    timeline = timeline.with_columns(
        completeness.alias("completeness"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("utc_date"),
        (pl.col("arr_abs_block_minus_sched_sec") <= 300.0).cast(pl.Int8).alias("arr_block_eq_sched"),
    )
    assert_no_locked_months(timeline)
    return timeline.sort(["AIRPORT", "month", "epoch", "MVT_ID_mvt"])


def _entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    _, counts = np.unique(values, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _rate(flag: np.ndarray) -> float:
    if flag.size == 0:
        return float("nan")
    return float(np.mean(flag))


def _safe_median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


def _safe_p90(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, 0.90))


def _safe_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def _annotate_streaks(timeline: pl.DataFrame) -> pl.DataFrame:
    """Consecutive unmatched run length using only prior movements at the airport-month."""

    parts: list[pl.DataFrame] = []
    for keys, part in timeline.group_by(["AIRPORT", "month"], maintain_order=True):
        unmatched = part["tl_unmatched"].to_numpy().astype(np.int8)
        epoch = part["epoch"].to_numpy().astype(np.int64)
        streak = np.zeros(part.height, dtype=np.int32)
        since_matched = np.full(part.height, np.nan, dtype=np.float64)
        since_unmatched = np.full(part.height, np.nan, dtype=np.float64)
        since_prev = np.full(part.height, np.nan, dtype=np.float64)
        run = 0
        last_matched = None
        last_unmatched = None
        last_epoch = None
        for i, flag in enumerate(unmatched):
            if last_epoch is not None:
                since_prev[i] = float(epoch[i] - last_epoch)
            if last_matched is not None:
                since_matched[i] = float(epoch[i] - last_matched)
            if last_unmatched is not None:
                since_unmatched[i] = float(epoch[i] - last_unmatched)
            streak[i] = run
            if flag == 1:
                run += 1
                last_unmatched = int(epoch[i])
            else:
                run = 0
                last_matched = int(epoch[i])
            last_epoch = int(epoch[i])
        parts.append(
            part.with_columns(
                pl.Series("sh_unmatched_streak", streak),
                pl.Series("sh_sec_since_matched", since_matched),
                pl.Series("sh_sec_since_unmatched", since_unmatched),
                pl.Series("sh_sec_since_prev", since_prev),
            )
        )
    return pl.concat(parts, how="vertical_relaxed") if parts else timeline


def attach_backward_source_health(focals: pl.DataFrame, timeline: pl.DataFrame) -> pl.DataFrame:
    """Backward-only, month-isolated source-health features for focal DEPs."""

    if focals.height == 0:
        return focals
    need = {"MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt"}
    missing = need - set(focals.columns)
    if missing:
        raise RuntimeError(f"focals missing {sorted(missing)}")
    tl = _annotate_streaks(timeline)
    tgt = focals.select(
        "MVT_ID_mvt",
        "AIRPORT",
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("epoch"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8).alias("month"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("utc_date"),
        pl.col("flight_family") if "flight_family" in focals.columns else pl.lit(CAT_FILL).alias("flight_family"),
    )
    feature_rows: list[pl.DataFrame] = []
    grouped_tl: dict[tuple[str, int], pl.DataFrame] = {}
    for part in tl.partition_by(["AIRPORT", "month"], maintain_order=True):
        grouped_tl[(str(part["AIRPORT"][0]), int(part["month"][0]))] = part
    for foc in tgt.partition_by(["AIRPORT", "month"], maintain_order=True):
        ap = str(foc["AIRPORT"][0])
        month_i = int(foc["month"][0])
        part = grouped_tl.get((ap, month_i))
        empty = {
            "sh_unmatched_streak": np.zeros(foc.height, dtype=np.int32),
            "sh_sec_since_matched": np.full(foc.height, np.nan),
            "sh_sec_since_unmatched": np.full(foc.height, np.nan),
            "sh_sec_since_prev": np.full(foc.height, np.nan),
        }
        if part is None or part.height == 0:
            feature_rows.append(_blank_source_health(foc, extra=empty))
            continue
        epoch = part["epoch"].to_numpy().astype(np.int64)
        unmatched = part["tl_unmatched"].to_numpy().astype(np.float64)
        type_m = part["type_missing_tl"].to_numpy().astype(np.float64)
        stand_m = part["stand_missing_tl"].to_numpy().astype(np.float64)
        rwy_m = part["rwy_missing_tl"].to_numpy().astype(np.float64)
        sched_m = part["sched_missing_tl"].to_numpy().astype(np.float64)
        flight_m = part["flight_missing_tl"].to_numpy().astype(np.float64)
        ades_m = part["ades_missing_tl"].to_numpy().astype(np.float64)
        complete = part["completeness"].to_numpy().astype(np.float64)
        is_dep = part["is_dep"].to_numpy().astype(np.float64)
        is_arr = part["is_arr"].to_numpy().astype(np.float64)
        prefixes = part["flight_prefix"].to_numpy()
        families = part["flight_family"].to_numpy() if "flight_family" in part.columns else np.array([CAT_FILL] * part.height)
        hours = (
            part["MVT_TIME_UTC_mvt"].dt.hour().to_numpy().astype(np.float64)
            if "MVT_TIME_UTC_mvt" in part.columns
            else np.full(part.height, np.nan)
        )
        ades = part["ADES_mvt"].fill_null(CAT_FILL).to_numpy() if "ADES_mvt" in part.columns else np.array([CAT_FILL] * part.height)
        types = part["AIRCRAFT_TYPE_mvt"].fill_null(CAT_FILL).cast(pl.String).to_numpy()
        dates = part["utc_date"].to_numpy()
        streak = part["sh_unmatched_streak"].to_numpy()
        since_m = part["sh_sec_since_matched"].to_numpy()
        since_u = part["sh_sec_since_unmatched"].to_numpy()
        since_p = part["sh_sec_since_prev"].to_numpy()
        t_tgt = foc["epoch"].to_numpy().astype(np.int64)
        d_tgt = foc["utc_date"].to_numpy()
        fam_tgt = foc["flight_family"].to_numpy()
        ids = foc["MVT_ID_mvt"].to_numpy()
        n = foc.height
        out: dict[str, Any] = {"MVT_ID_mvt": ids}
        for name in SOURCE_HEALTH_NUMS:
            out.setdefault(name, np.full(n, np.nan))
        right = np.searchsorted(epoch, t_tgt, side="left")
        out["sh_unmatched_streak"] = np.array(
            [streak[r - 1] if r > 0 else 0 for r in right], dtype=np.float64
        )
        out["sh_sec_since_matched"] = np.array(
            [since_m[r - 1] if r > 0 else np.nan for r in right], dtype=np.float64
        )
        out["sh_sec_since_unmatched"] = np.array(
            [since_u[r - 1] if r > 0 else np.nan for r in right], dtype=np.float64
        )
        out["sh_sec_since_prev"] = np.array(
            [since_p[r - 1] if r > 0 else np.nan for r in right], dtype=np.float64
        )
        for k in COUNT_WINDOWS:
            n_k = np.zeros(n, dtype=np.float64)
            rates = {name: np.full(n, np.nan) for name in (
                "unmatched", "type", "stand", "rwy", "sched", "flight", "ades", "complete", "entropy"
            )}
            for i, r in enumerate(right):
                left = max(0, r - k)
                sl = slice(left, r)
                cnt = r - left
                n_k[i] = cnt
                if cnt <= 0:
                    continue
                rates["unmatched"][i] = _rate(unmatched[sl])
                rates["type"][i] = _rate(type_m[sl])
                rates["stand"][i] = _rate(stand_m[sl])
                rates["rwy"][i] = _rate(rwy_m[sl])
                rates["sched"][i] = _rate(sched_m[sl])
                rates["flight"][i] = _rate(flight_m[sl])
                rates["ades"][i] = _rate(ades_m[sl])
                rates["complete"][i] = float(np.mean(complete[sl]))
                rates["entropy"][i] = _entropy(prefixes[sl].astype(object))
            out[f"sh_n_k{k}"] = n_k
            out[f"sh_unmatched_rate_k{k}"] = rates["unmatched"]
            out[f"sh_type_miss_rate_k{k}"] = rates["type"]
            out[f"sh_stand_miss_rate_k{k}"] = rates["stand"]
            out[f"sh_rwy_miss_rate_k{k}"] = rates["rwy"]
            out[f"sh_sched_miss_rate_k{k}"] = rates["sched"]
            out[f"sh_flight_miss_rate_k{k}"] = rates["flight"]
            out[f"sh_ades_miss_rate_k{k}"] = rates["ades"]
            out[f"sh_completeness_k{k}"] = rates["complete"]
            out[f"sh_prefix_entropy_k{k}"] = rates["entropy"]
        for w in TIME_WINDOWS_SEC:
            name = TIME_WINDOW_NAMES[w]
            n_w = np.zeros(n, dtype=np.float64)
            u_rate = np.full(n, np.nan)
            t_rate = np.full(n, np.nan)
            s_rate = np.full(n, np.nan)
            r_rate = np.full(n, np.nan)
            c_rate = np.full(n, np.nan)
            n_dep = np.zeros(n, dtype=np.float64)
            n_arr = np.zeros(n, dtype=np.float64)
            left = np.searchsorted(epoch, t_tgt - w, side="left")
            for i in range(n):
                sl = slice(int(left[i]), int(right[i]))
                cnt = int(right[i] - left[i])
                n_w[i] = cnt
                if cnt <= 0:
                    continue
                u_rate[i] = _rate(unmatched[sl])
                t_rate[i] = _rate(type_m[sl])
                s_rate[i] = _rate(stand_m[sl])
                r_rate[i] = _rate(rwy_m[sl])
                c_rate[i] = float(np.mean(complete[sl]))
                n_dep[i] = float(np.sum(is_dep[sl]))
                n_arr[i] = float(np.sum(is_arr[sl]))
            out[f"sh_n_{name}"] = n_w
            out[f"sh_unmatched_rate_{name}"] = u_rate
            out[f"sh_type_miss_rate_{name}"] = t_rate
            out[f"sh_stand_miss_rate_{name}"] = s_rate
            out[f"sh_rwy_miss_rate_{name}"] = r_rate
            out[f"sh_completeness_{name}"] = c_rate
            out[f"sh_n_dep_{name}"] = n_dep
            out[f"sh_n_arr_{name}"] = n_arr
        n_dtd = np.zeros(n, dtype=np.float64)
        u_dtd = np.full(n, np.nan)
        c_dtd = np.full(n, np.nan)
        dep_dtd = np.zeros(n, dtype=np.float64)
        arr_dtd = np.zeros(n, dtype=np.float64)
        fam_n = np.zeros(n, dtype=np.float64)
        fam_days = np.full(n, np.nan)
        fam_u = np.full(n, np.nan)
        fam_hstd = np.full(n, np.nan)
        fam_dest = np.full(n, np.nan)
        fam_type = np.full(n, np.nan)
        left_6h = np.searchsorted(epoch, t_tgt - 21600, side="left")
        for i in range(n):
            r = int(right[i])
            sl = slice(0, r)
            same_day = dates[sl] == d_tgt[i]
            n_dtd[i] = float(same_day.sum())
            if same_day.any():
                u_dtd[i] = _rate(unmatched[sl][same_day])
                c_dtd[i] = float(np.mean(complete[sl][same_day]))
                dep_dtd[i] = float(np.sum(is_dep[sl][same_day]))
                arr_dtd[i] = float(np.sum(is_arr[sl][same_day]))
            fam_mask = families[sl] == fam_tgt[i]
            if fam_mask.any():
                fam_u[i] = _rate(unmatched[sl][fam_mask])
                fam_hstd[i] = _safe_std(hours[sl][fam_mask])
                fam_dest[i] = float(np.unique(ades[sl][fam_mask]).size)
                fam_type[i] = float(np.unique(types[sl][fam_mask]).size)
                last = int(np.where(fam_mask)[0][-1])
                fam_days[i] = float(max(t_tgt[i] - epoch[last], 0) / 86400.0)
            sl6 = slice(int(left_6h[i]), r)
            fam_n[i] = float(np.sum(families[sl6] == fam_tgt[i])) if r > left_6h[i] else 0.0
        out["sh_n_dtd"] = n_dtd
        out["sh_unmatched_rate_dtd"] = u_dtd
        out["sh_completeness_dtd"] = c_dtd
        out["sh_n_dep_dtd"] = dep_dtd
        out["sh_n_arr_dtd"] = arr_dtd
        out["fam_recent_n_6h"] = fam_n
        out["fam_days_since_prev"] = fam_days
        out["fam_unmatched_rate_hist"] = fam_u
        out["fam_hour_std_hist"] = fam_hstd
        out["fam_dest_nunique_hist"] = fam_dest
        out["fam_type_nunique_hist"] = fam_type
        feature_rows.append(pl.DataFrame(out))
    feats = pl.concat(feature_rows, how="diagonal_relaxed") if feature_rows else focals.select("MVT_ID_mvt")
    joined = focals.join(feats, on="MVT_ID_mvt", how="left")
    if joined.height != focals.height:
        raise RuntimeError("source-health join changed row count")
    present_feats = [c for c in SOURCE_HEALTH_NUMS if c in joined.columns]
    assert_source_health_features(present_feats)
    return joined


def _blank_source_health(foc: pl.DataFrame, extra: dict[str, Any] | None = None) -> pl.DataFrame:
    n = foc.height
    out: dict[str, Any] = {"MVT_ID_mvt": foc["MVT_ID_mvt"]}
    if extra:
        out.update(extra)
    for name in SOURCE_HEALTH_NUMS:
        if name not in out:
            out[name] = np.full(n, np.nan)
    return pl.DataFrame(out)


def attach_arr_source_health(focals: pl.DataFrame, timeline: pl.DataFrame) -> pl.DataFrame:
    """Completed-arrival sensors: in-block strictly before focal takeoff, month-isolated."""

    if focals.height == 0:
        return focals
    arr = timeline.filter(pl.col("is_arr") == 1).drop_nulls("inblock_epoch").sort(["AIRPORT", "month", "inblock_epoch"])
    tgt = focals.select(
        "MVT_ID_mvt",
        "AIRPORT",
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("epoch"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8).alias("month"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("utc_date"),
    )
    grouped_arr: dict[tuple[str, int], pl.DataFrame] = {}
    for part in arr.partition_by(["AIRPORT", "month"], maintain_order=True):
        grouped_arr[(str(part["AIRPORT"][0]), int(part["month"][0]))] = part
    frames: list[pl.DataFrame] = []
    for foc in tgt.partition_by(["AIRPORT", "month"], maintain_order=True):
        ap = str(foc["AIRPORT"][0])
        month = int(foc["month"][0])
        part = grouped_arr.get((ap, month))
        n = foc.height
        out: dict[str, Any] = {"MVT_ID_mvt": foc["MVT_ID_mvt"]}
        if part is None or part.height == 0:
            for name in ARR_HEALTH_NUMS:
                out[name] = np.full(n, np.nan)
            frames.append(pl.DataFrame(out))
            continue
        block = part["inblock_epoch"].to_numpy().astype(np.int64)
        taxi = part["arr_taxi_sec"].to_numpy().astype(np.float64)
        delay = part["arr_mvt_minus_sched_sec"].to_numpy().astype(np.float64)
        eq = part["arr_block_eq_sched"].fill_null(0).to_numpy().astype(np.float64)
        type_m = part["type_missing_tl"].to_numpy().astype(np.float64)
        complete = part["completeness"].to_numpy().astype(np.float64)
        dates = part["utc_date"].to_numpy()
        t_tgt = foc["epoch"].to_numpy().astype(np.int64)
        d_tgt = foc["utc_date"].to_numpy()
        right = np.searchsorted(block, t_tgt, side="left")
        age = np.full(n, np.nan)
        for i, r in enumerate(right):
            if r > 0:
                age[i] = float(t_tgt[i] - block[r - 1])
        out["arr_age_sec"] = age
        for w in ARR_WINDOWS_SEC:
            name = ARR_WINDOW_NAMES[w]
            left = np.searchsorted(block, t_tgt - w, side="left")
            n_w = np.zeros(n, dtype=np.float64)
            med = np.full(n, np.nan)
            p90 = np.full(n, np.nan)
            std = np.full(n, np.nan)
            dmed = np.full(n, np.nan)
            eqr = np.full(n, np.nan)
            tm = np.full(n, np.nan)
            comp = np.full(n, np.nan)
            for i in range(n):
                sl = slice(int(left[i]), int(right[i]))
                cnt = int(right[i] - left[i])
                n_w[i] = cnt
                if cnt <= 0:
                    continue
                med[i] = _safe_median(taxi[sl])
                p90[i] = _safe_p90(taxi[sl])
                std[i] = _safe_std(taxi[sl])
                dmed[i] = _safe_median(delay[sl])
                eqr[i] = _rate(eq[sl])
                tm[i] = _rate(type_m[sl])
                comp[i] = float(np.mean(complete[sl]))
            out[f"arr_n_{name}"] = n_w
            out[f"arr_taxi_median_{name}"] = med
            out[f"arr_taxi_p90_{name}"] = p90
            out[f"arr_taxi_std_{name}"] = std
            out[f"arr_delay_median_{name}"] = dmed
            out[f"arr_block_eq_sched_rate_{name}"] = eqr
            out[f"arr_type_miss_rate_{name}"] = tm
            out[f"arr_completeness_{name}"] = comp
        n_dtd = np.zeros(n, dtype=np.float64)
        med_dtd = np.full(n, np.nan)
        n_prior = np.zeros(n, dtype=np.float64)
        med_prior = np.full(n, np.nan)
        for i in range(n):
            r = int(right[i])
            sl = slice(0, r)
            same = dates[sl] == d_tgt[i]
            n_dtd[i] = float(same.sum())
            if same.any():
                med_dtd[i] = _safe_median(taxi[sl][same])
            if r > 0:
                prior = dates[sl] < d_tgt[i]
                # prior local day only, not all history
                if prior.any():
                    last_prior = dates[sl][prior][-1]
                    day_mask = dates[sl] == last_prior
                    n_prior[i] = float(day_mask.sum())
                    med_prior[i] = _safe_median(taxi[sl][day_mask])
        out["arr_n_dtd"] = n_dtd
        out["arr_taxi_median_dtd"] = med_dtd
        out["arr_n_prior_day"] = n_prior
        out["arr_taxi_median_prior_day"] = med_prior
        frames.append(pl.DataFrame(out))
    feats = pl.concat(frames, how="diagonal_relaxed")
    joined = focals.join(feats, on="MVT_ID_mvt", how="left")
    if joined.height != focals.height:
        raise RuntimeError("ARR source-health join changed row count")
    assert_source_health_features([c for c in ARR_HEALTH_NUMS if c in joined.columns])
    return joined


def attach_retrospective_day_features(focals: pl.DataFrame, timeline: pl.DataFrame) -> pl.DataFrame:
    """Same-day features excluding the focal. Labeled retro_*; not mixed with backward-only."""

    if focals.height == 0:
        return focals
    tgt = focals.select(
        "MVT_ID_mvt",
        "AIRPORT",
        pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8).alias("month"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("utc_date"),
    )
    day = (
        timeline.group_by(["AIRPORT", "month", "utc_date"])
        .agg(
            pl.len().alias("_day_n"),
            pl.col("tl_unmatched").mean().alias("_day_unmatched"),
            pl.col("completeness").mean().alias("_day_complete"),
        )
    )
    joined = tgt.join(day, on=["AIRPORT", "month", "utc_date"], how="left").with_columns(
        (pl.col("_day_n") - 1).clip(lower_bound=0).alias("retro_n_dtd"),
        pl.col("_day_unmatched").alias("retro_unmatched_rate_dtd"),
        pl.col("_day_complete").alias("retro_completeness_dtd"),
    )
    return focals.join(joined.select("MVT_ID_mvt", *RETRO_NUMS), on="MVT_ID_mvt", how="left")


def fit_airport_baselines(train: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """D-fit airport mean/std for z-scores. Train months must be inside D."""

    if "month" in train.columns:
        assert_design_months(sorted(set(int(m) for m in train["month"].to_list())))
    use = [c for c in columns if c in train.columns]
    if not use or "AIRPORT" not in train.columns:
        return pl.DataFrame({"AIRPORT": []})
    aggs = []
    for col in use:
        aggs.extend(
            [
                pl.col(col).mean().alias(f"{col}__mean"),
                pl.col(col).std().alias(f"{col}__std"),
            ]
        )
    return train.group_by("AIRPORT").agg(aggs)


def apply_airport_zscores(frame: pl.DataFrame, baselines: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    use = [c for c in columns if c in frame.columns and f"{c}__mean" in baselines.columns]
    if not use:
        return frame
    joined = frame.join(baselines, on="AIRPORT", how="left")
    exprs = []
    for col in use:
        std = pl.col(f"{col}__std")
        exprs.append(
            pl.when(std.is_not_null() & (std.abs() > 1e-9))
            .then((pl.col(col) - pl.col(f"{col}__mean")) / std)
            .otherwise(None)
            .alias(f"{col}_z")
        )
    out = joined.with_columns(exprs).drop([c for c in joined.columns if c.endswith("__mean") or c.endswith("__std")])
    return out


def attach_source_state(
    focals: pl.DataFrame,
    timeline: pl.DataFrame,
    *,
    include_arr: bool = True,
    include_retro: bool = False,
) -> pl.DataFrame:
    out = attach_backward_source_health(focals, timeline)
    if include_arr:
        out = attach_arr_source_health(out, timeline)
    if include_retro:
        out = attach_retrospective_day_features(out, timeline)
    return out


def _null_rate_table(frame: pl.LazyFrame, cols: Sequence[str]) -> dict[str, float]:
    existing = [c for c in cols if c in frame.collect_schema().names()]
    if not existing:
        return {}
    row = frame.select([pl.col(c).is_null().mean().alias(c) for c in existing]).collect().to_dicts()[0]
    return {k: float(v) for k, v in row.items()}


def ranking_field_availability_audit(repo: Path) -> dict[str, Any]:
    """Field-by-field ranking vs training availability for ARR source-health sensors."""

    rank = pl.scan_parquet(ranking_path(repo / RAW_DIR if (repo / RAW_DIR).exists() else RAW_DIR))
    train_paths = month_isolated_training_paths(repo, tuple(range(1, 11)))
    train = pl.concat([pl.scan_parquet(p) for p in train_paths], how="vertical_relaxed")
    payload: dict[str, Any] = {
        "ranking_arr_taxitime_retained": True,
        "ranking_dep_taxitime_blank": True,
        "contract": "ARR TAXITIME/BLOCK retained in ranking; DEP TAXITIME/BLOCK blank. Backward completed-arrival as-of only.",
    }
    for phase, label in (("ARR", "arr"), ("DEP", "dep")):
        r = rank.filter(pl.col("PHASE_mvt") == phase)
        t = train.filter(pl.col("PHASE_mvt") == phase)
        payload[f"ranking_{label}_n"] = int(r.select(pl.len()).collect().item())
        payload[f"train_{label}_n"] = int(t.select(pl.len()).collect().item())
        payload[f"ranking_{label}_null_rate"] = _null_rate_table(r, RANKING_AUDIT_FIELDS)
        payload[f"train_{label}_null_rate"] = _null_rate_table(t, RANKING_AUDIT_FIELDS)
    r_arr = payload["ranking_arr_null_rate"]
    r_dep = payload["ranking_dep_null_rate"]
    payload["arr_taxi_ranking_available"] = float(r_arr.get("TAXITIME_SEC_mvt", 1.0)) < 0.05
    payload["arr_block_ranking_available"] = float(r_arr.get("BLOCK_TIME_UTC_mvt", 1.0)) < 0.05
    payload["dep_taxi_ranking_blank"] = float(r_dep.get("TAXITIME_SEC_mvt", 0.0)) > 0.95
    payload["dep_block_ranking_blank"] = float(r_dep.get("BLOCK_TIME_UTC_mvt", 0.0)) > 0.95
    payload["usable_arr_sensors"] = [
        field
        for field, rate in r_arr.items()
        if rate < 0.20
    ]
    payload["forbidden_for_dep_features"] = ["TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"]
    return payload


def source_health_bucket_expr(col: str = "sh_unmatched_rate_1h") -> pl.Expr:
    x = pl.col(col)
    return (
        pl.when(x.is_null())
        .then(pl.lit("unknown"))
        .when(x < 0.02)
        .then(pl.lit("low"))
        .when(x < 0.08)
        .then(pl.lit("normal"))
        .when(x < 0.20)
        .then(pl.lit("elevated"))
        .otherwise(pl.lit("high"))
        .alias("source_health_bucket")
    )


def streak_bucket_expr(col: str = "sh_unmatched_streak") -> pl.Expr:
    x = pl.col(col).fill_null(0)
    return (
        pl.when(x <= 0)
        .then(pl.lit("s0"))
        .when(x == 1)
        .then(pl.lit("s1"))
        .when(x <= 3)
        .then(pl.lit("s2_3"))
        .otherwise(pl.lit("s4plus"))
        .alias("unmatched_streak_bucket")
    )


def traffic_bucket_expr(col: str = "sh_n_dep_1h") -> pl.Expr:
    x = pl.col(col)
    return (
        pl.when(x.is_null())
        .then(pl.lit("unknown"))
        .when(x <= 4)
        .then(pl.lit("low"))
        .when(x <= 12)
        .then(pl.lit("normal"))
        .otherwise(pl.lit("high"))
        .alias("traffic_load_bucket")
    )


def arr_health_bucket_expr(col: str = "arr_block_eq_sched_rate_1h") -> pl.Expr:
    x = pl.col(col)
    return (
        pl.when(x.is_null())
        .then(pl.lit("unknown"))
        .when(x < 0.05)
        .then(pl.lit("low"))
        .when(x < 0.20)
        .then(pl.lit("normal"))
        .otherwise(pl.lit("high"))
        .alias("arr_source_health_bucket")
    )
