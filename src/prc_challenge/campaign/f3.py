# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""F3 ranking-safe airport operational-state features. Reuses traffic.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from prc_challenge.campaign.runtime import CampaignConfig, LOCKED_MONTHS
from prc_challenge.campaign.scoring import assert_no_forbidden_features
from prc_challenge.nn_encode import add_sec_since_prev_dep
from prc_challenge.traffic import (
    add_delayed_bunching,
    add_forward_counts,
    add_planned_demand,
    add_prev_same_rwy_wake,
    add_runway_config_features,
    add_runway_counts,
    add_traffic_counts,
)

F3_WINDOWS = (120, 300, 600, 1200, 1800, 3600)
ARR_CONTEXT_COLS = (
    "MVT_ID_mvt",
    "AIRPORT",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "RUNWAY_mvt",
    "SCHED_TIME_UTC_mvt",
    "month",
)


def f3_numeric_columns() -> list[str]:
    cols: list[str] = []
    for w in F3_WINDOWS:
        cols.extend(
            [
                f"n_dep_{w}s",
                f"n_arr_{w}s",
                f"n_mvt_{w}s",
                f"n_dep_rwy_{w}s",
                f"n_arr_rwy_{w}s",
                f"dep_arr_ratio_{w}s",
            ]
        )
    cols.extend(
        [
            "sec_since_prev_dep",
            "gap_roll3",
            "gap_roll5",
            "gap_roll10",
            "gap_mean_5",
            "gap_std_5",
            "burst_size_5",
            "n_rwy_used_3600s",
            "rwy_share_3600s",
            "rwy_hhi_3600s",
            "rwy_modal_changed_1800s",
            "sec_since_prev_rwy_dep",
            "sec_since_prev_rwy_arr",
            "sec_until_next_sched_dep",
        ]
    )
    cols.extend([f"n_delayed_dep_ctr_{w}s" for w in (900, 1800, 3600)])
    cols.extend(
        [
            "n_same_stand_5",
            "n_same_type_5",
            "n_heavy_5",
            "burst_pos_5",
            "n_short_gap_5",
            "gap_median_5",
            "n_dep_fwd_1800s",
            "n_arr_fwd_1800s",
            "n_sched_dep_1800s",
            "n_sched_dep_3600s",
            "demand_vs_throughput_3600s",
        ]
    )
    return cols


def add_burst_gaps(df: pl.DataFrame) -> pl.DataFrame:
    ordered = df.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    if "sec_since_prev_dep" not in ordered.columns:
        ordered = add_sec_since_prev_dep(ordered)
    return ordered.with_columns(
        [
            pl.col("sec_since_prev_dep").shift(1).over("AIRPORT").alias("gap_l1"),
            pl.col("sec_since_prev_dep").shift(2).over("AIRPORT").alias("gap_l2"),
            pl.col("sec_since_prev_dep").shift(3).over("AIRPORT").alias("gap_l3"),
            pl.col("sec_since_prev_dep").shift(4).over("AIRPORT").alias("gap_l4"),
            pl.col("sec_since_prev_dep").shift(5).over("AIRPORT").alias("gap_l5"),
            pl.col("sec_since_prev_dep").shift(9).over("AIRPORT").alias("gap_l9"),
            pl.col("sec_since_prev_dep").rolling_mean(window_size=5, min_samples=1).over("AIRPORT").alias("gap_mean_5"),
            pl.col("sec_since_prev_dep").rolling_std(window_size=5, min_samples=2).over("AIRPORT").alias("gap_std_5"),
            pl.col("sec_since_prev_dep").rolling_median(window_size=5, min_samples=1).over("AIRPORT").alias("gap_median_5"),
            (pl.col("sec_since_prev_dep") < 120).cast(pl.Int32).rolling_sum(window_size=5, min_samples=1).over("AIRPORT").alias("burst_size_5"),
            (pl.col("sec_since_prev_dep") < 120).cast(pl.Int32).rolling_sum(window_size=5, min_samples=1).over("AIRPORT").alias("burst_pos_5"),
            (pl.col("sec_since_prev_dep") < 90).cast(pl.Int32).rolling_sum(window_size=5, min_samples=1).over("AIRPORT").alias("n_short_gap_5"),
        ]
    ).with_columns(
        (pl.col("sec_since_prev_dep").fill_null(0) + pl.col("gap_l1").fill_null(0) + pl.col("gap_l2").fill_null(0)).alias(
            "gap_roll3"
        ),
        (
            pl.col("sec_since_prev_dep").fill_null(0)
            + pl.col("gap_l1").fill_null(0)
            + pl.col("gap_l2").fill_null(0)
            + pl.col("gap_l3").fill_null(0)
        ).alias("gap_roll5"),
        (
            pl.col("sec_since_prev_dep").fill_null(0)
            + pl.col("gap_l1").fill_null(0)
            + pl.col("gap_l2").fill_null(0)
            + pl.col("gap_l3").fill_null(0)
            + pl.col("gap_l4").fill_null(0)
            + pl.col("gap_l5").fill_null(0)
            + pl.col("gap_l9").fill_null(0)
        ).alias("gap_roll10"),
    )


def add_prev_arr_and_next_sched(df: pl.DataFrame) -> pl.DataFrame:
    """Same-runway previous ARR gap and next scheduled DEP gap. Ranking-safe SCHED/ARR times only."""
    ordered = df.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    if "PHASE_mvt" in ordered.columns and "RUNWAY_mvt" in ordered.columns:
        arr = (
            ordered.filter(pl.col("PHASE_mvt") == "ARR")
            .select(
                [
                    "AIRPORT",
                    "RUNWAY_mvt",
                    pl.col("MVT_TIME_UTC_mvt").alias("_prev_arr_time"),
                ]
            )
            .sort(["AIRPORT", "RUNWAY_mvt", "_prev_arr_time"])
        )
        if arr.height:
            left = ordered.sort(["AIRPORT", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
            ordered = left.join_asof(
                arr,
                left_on="MVT_TIME_UTC_mvt",
                right_on="_prev_arr_time",
                by=["AIRPORT", "RUNWAY_mvt"],
                strategy="backward",
            )
            ordered = ordered.with_columns(
                (pl.col("MVT_TIME_UTC_mvt").dt.epoch("s") - pl.col("_prev_arr_time").dt.epoch("s")).alias(
                    "sec_since_prev_rwy_arr"
                )
            ).drop(["_prev_arr_time"], strict=False)
        else:
            ordered = ordered.with_columns(pl.lit(None).cast(pl.Float64).alias("sec_since_prev_rwy_arr"))
    else:
        ordered = ordered.with_columns(pl.lit(None).cast(pl.Float64).alias("sec_since_prev_rwy_arr"))
    if "SCHED_TIME_UTC_mvt" in ordered.columns:
        nxt = (
            ordered.filter(pl.col("SCHED_TIME_UTC_mvt").is_not_null())
            .select(["AIRPORT", pl.col("SCHED_TIME_UTC_mvt").alias("_sched")])
            .sort(["AIRPORT", "_sched"])
        )
        left = ordered.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
        ordered = left.join_asof(
            nxt,
            left_on="MVT_TIME_UTC_mvt",
            right_on="_sched",
            by="AIRPORT",
            strategy="forward",
        )
        ordered = ordered.with_columns(
            (pl.col("_sched").dt.epoch("s") - pl.col("MVT_TIME_UTC_mvt").dt.epoch("s")).alias(
                "sec_until_next_sched_dep"
            )
        ).drop(["_sched"], strict=False)
    else:
        ordered = ordered.with_columns(pl.lit(None).cast(pl.Float64).alias("sec_until_next_sched_dep"))
    return ordered


def add_mix_and_stand_context(df: pl.DataFrame) -> pl.DataFrame:
    """Previous-5 same-airport mix/stand pressure. Uses only earlier takeoffs."""
    ordered = df.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    if "WK_TBL_CAT_flt" in ordered.columns:
        heavy = pl.col("WK_TBL_CAT_flt").cast(pl.String).str.to_uppercase().is_in(["H", "J", "SUPER", "HEAVY"])
    else:
        heavy = pl.lit(0)
    same_stand = pl.sum_horizontal(
        [(pl.col("STAND_mvt") == pl.col("STAND_mvt").shift(i).over("AIRPORT")).cast(pl.Int32).fill_null(0) for i in range(1, 6)]
    )
    same_type = pl.sum_horizontal(
        [
            (pl.col("AIRCRAFT_TYPE_mvt") == pl.col("AIRCRAFT_TYPE_mvt").shift(i).over("AIRPORT")).cast(pl.Int32).fill_null(0)
            for i in range(1, 6)
        ]
    )
    heavy_prev = pl.sum_horizontal(
        [heavy.cast(pl.Int32).fill_null(0).shift(i).over("AIRPORT").fill_null(0) for i in range(1, 6)]
    )
    return ordered.with_columns(
        [
            same_stand.alias("n_same_stand_5"),
            same_type.alias("n_same_type_5"),
            heavy_prev.alias("n_heavy_5"),
        ]
    )


def _nearby_months(month: int | None) -> list[int] | None:
    if month is None:
        return None
    m = int(month)
    return [m - 1, m, m + 1]


def _month_context(dep_ctx: pl.DataFrame, arr: pl.DataFrame | None, month: int | None) -> pl.DataFrame:
    """DEP+ARR rows for `month` and ±1 month so 3600s windows are not truncated at month edges."""
    months = _nearby_months(month)
    dep_s = dep_ctx
    arr_s = arr
    if months is not None and "month" in dep_ctx.columns:
        dep_s = dep_ctx.filter(pl.col("month").is_in(months))
    if arr_s is not None and arr_s.height and months is not None and "month" in arr_s.columns:
        arr_s = arr_s.filter(pl.col("month").is_in(months))
    if arr_s is not None and arr_s.height:
        have = [c for c in ARR_CONTEXT_COLS if c in dep_s.columns and c in arr_s.columns]
        if have:
            return pl.concat([dep_s.select(have), arr_s.select(have)], how="diagonal")
    return dep_s


def build_f3(dep: pl.DataFrame, context: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-month F3 with ±1 month of ranking-safe context for hour-scale windows."""
    ctx = context if context is not None else dep
    if "PHASE_mvt" not in ctx.columns:
        ctx = ctx.with_columns(pl.lit("DEP").alias("PHASE_mvt"))
    parts: list[pl.DataFrame] = []
    months = dep["month"].unique().sort().to_list()
    for month in months:
        dep_m = dep.filter(pl.col("month") == month)
        nearby = _nearby_months(month)
        if nearby is not None and "month" in ctx.columns:
            ctx_m = ctx.filter(pl.col("month").is_in(nearby))
        else:
            ctx_m = ctx
        out = add_traffic_counts(dep_m, ctx_m, windows_sec=F3_WINDOWS)
        out = add_runway_counts(out, ctx_m, windows_sec=F3_WINDOWS)
        if "n_delayed_dep_ctr_1800s" not in out.columns and "EOBT_1_flt" in ctx_m.columns:
            out = add_delayed_bunching(out, ctx_m)
        if "n_rwy_used_3600s" not in out.columns:
            out = add_runway_config_features(out, ctx_m)
        if "sec_since_prev_rwy_dep" not in out.columns and "WK_TBL_CAT_flt" in ctx_m.columns:
            out = add_prev_same_rwy_wake(out, ctx_m)
        out = add_burst_gaps(out)
        out = add_prev_arr_and_next_sched(out)
        out = add_mix_and_stand_context(out)
        if "SCHED_TIME_UTC_mvt" in out.columns:
            out = add_planned_demand(out, ctx_m)
        try:
            out = add_forward_counts(out, ctx_m, windows_sec=(1800,))
        except Exception:
            pass
        for w in F3_WINDOWS:
            dep_c = pl.col(f"n_dep_{w}s").cast(pl.Float64)
            arr_c = pl.col(f"n_arr_{w}s").cast(pl.Float64)
            out = out.with_columns((dep_c / (arr_c + 1.0)).alias(f"dep_arr_ratio_{w}s"))
        if "n_sched_dep_3600s" in out.columns and "n_dep_3600s" in out.columns:
            out = out.with_columns(
                (pl.col("n_sched_dep_3600s").cast(pl.Float64) / (pl.col("n_dep_3600s").cast(pl.Float64) + 1.0)).alias(
                    "demand_vs_throughput_3600s"
                )
            )
        parts.append(out)
    frame = pl.concat(parts, how="diagonal")
    assert_no_forbidden_features([c for c in f3_numeric_columns() if c in frame.columns])
    return frame


def f3_cache_path(cfg: CampaignConfig) -> Path:
    return cfg.cache_root / "f3_airport_state_months_1_10.parquet"


def arr_context_path(cfg: CampaignConfig) -> Path:
    return cfg.cache_root / "arr_context_months_1_10.parquet"


def read_cached_f3(cfg: CampaignConfig) -> pl.DataFrame | None:
    """F3 payload only (MVT_ID + numeric cols). None if missing, forced, smoke, or ARR-stale.

    GPU nights (`allow_rebuild_cpu_caches=False`) must use an existing F3 file even when
    `n_arr_*` is ~0. Returning None there raises a retryable `need F3` skip forever.
    """
    path = f3_cache_path(cfg)
    if not path.exists() or cfg.smoke:
        return None
    if cfg.force_rerun and cfg.allow_rebuild_cpu_caches:
        return None
    cached = pl.read_parquet(path)
    needed = list(f3_numeric_columns())
    if not all(c in cached.columns for c in needed):
        return None
    arr_cols = [c for c in needed if c.startswith("n_arr_")]
    arr_mean = (
        float(np.mean([float(cached[c].fill_null(0).mean() or 0.0) for c in arr_cols])) if arr_cols else 0.0
    )
    if arr_mean < 0.05 and arr_context_path(cfg).exists():
        if not cfg.allow_rebuild_cpu_caches:
            print("F3 n_arr ~0 with ARR sidecar present; GPU will use existing F3", flush=True)
            return cached
        return None
    return cached


def _dep_context(dep: pl.DataFrame) -> pl.DataFrame:
    out = dep
    if "PHASE_mvt" not in out.columns:
        out = out.with_columns(pl.lit("DEP").alias("PHASE_mvt"))
    cols = [c for c in ARR_CONTEXT_COLS if c in out.columns]
    return out.select(cols)


def load_or_build_arr_context(cfg: CampaignConfig) -> pl.DataFrame | None:
    """Ranking-safe ARR movements at the reporting airport, months 1–10 only. No taxi/AOBT labels."""
    path = arr_context_path(cfg)
    if path.exists() and not (cfg.force_rerun and cfg.allow_rebuild_cpu_caches):
        frame = pl.read_parquet(path)
        if "month" in frame.columns:
            locked = frame.filter(pl.col("month").is_in(list(LOCKED_MONTHS)))
            if locked.height:
                print("ARR context sidecar has locked V4 months; ignoring", flush=True)
                return None
        if "PHASE_mvt" in frame.columns and int((frame["PHASE_mvt"] == "ARR").sum()) == 0:
            return None
        return frame
    if cfg.smoke:
        return None
    if not cfg.allow_rebuild_cpu_caches:
        print("ARR context sidecar missing; GPU will not scan data/raw", flush=True)
        return None
    raw = cfg.repo_root / "data" / "raw"
    try:
        from prc_challenge.io import CHALLENGE_AIRPORTS, scan_training, training_paths, with_reporting_airport

        training_paths(raw)
    except FileNotFoundError:
        print("ARR context skipped: no data/raw training parquet", flush=True)
        return None
    from prc_challenge.bundles import CACHE_MONTHS

    lf = (
        with_reporting_airport(scan_training(raw))
        .filter(pl.col("PHASE_mvt") == "ARR")
        .with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
        .filter(pl.col("month").is_in(list(CACHE_MONTHS)))
        .filter(pl.col("AIRPORT").is_in(list(CHALLENGE_AIRPORTS)))
    )
    names = lf.collect_schema().names()
    cols = [c for c in ARR_CONTEXT_COLS if c in names]
    if "TAXITIME_SEC_mvt" in cols or "AOBT_3_flt" in cols or "BLOCK_TIME_UTC_mvt" in cols:
        raise RuntimeError("ARR context refused forbidden label columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"building ARR context sidecar -> {path}", flush=True)
    lf.select(cols).sink_parquet(path)
    from prc_challenge.campaign.runtime import persist_repo_file

    persist_repo_file(cfg, path)
    return pl.read_parquet(path)


def load_or_build_f3(cfg: CampaignConfig, dep: pl.DataFrame) -> pl.DataFrame:
    cached = read_cached_f3(cfg)
    if cached is not None:
        return dep.join(cached, on="MVT_ID_mvt", how="left")
    if not cfg.allow_rebuild_cpu_caches:
        raise FileNotFoundError(
            f"need {f3_cache_path(cfg).as_posix()} from the CPU campaign; GPU will not rebuild F3"
        )
    path = f3_cache_path(cfg)
    if path.exists() and not cfg.force_rerun and not cfg.smoke:
        print("F3 cache has ~0 n_arr; rebuilding with ARR sidecar", flush=True)
    arr = load_or_build_arr_context(cfg)
    dep_ctx = _dep_context(dep)
    n_arr = 0 if arr is None else int(arr.height)
    print(
        f"F3 context DEP={dep_ctx.height} ARR_sidecar={n_arr} "
        f"{'DEP-only' if n_arr == 0 else 'month-padded, not fully concatenated'}",
        flush=True,
    )
    extra_needed = list(f3_numeric_columns())
    months = dep["month"].unique().sort().to_list() if "month" in dep.columns else [None]
    parts: list[pl.DataFrame] = []
    sidecar = arr_context_path(cfg)
    for month in months:
        dep_m = dep.filter(pl.col("month") == month) if month is not None else dep
        shard = cfg.cache_root / f"f3_month_{int(month):02d}.parquet" if month is not None else None
        if shard is not None and shard.exists() and not cfg.force_rerun and not cfg.smoke:
            part = pl.read_parquet(shard)
            arr_cols = [c for c in extra_needed if c.startswith("n_arr_") and c in part.columns]
            arr_mean = (
                float(np.mean([float(part[c].fill_null(0).mean() or 0.0) for c in arr_cols])) if arr_cols else 0.0
            )
            if "MVT_ID_mvt" in part.columns and "n_dep_120s" in part.columns and (
                arr_mean >= 0.05 or not sidecar.exists()
            ):
                print(f"F3 resume skip month={month} n={part.height}", flush=True)
                parts.append(part)
                continue
            print(f"F3 month={month} shard stale (n_arr_mean={arr_mean:.3f}); rebuilding", flush=True)
        ctx_m = _month_context(dep_ctx, arr, month)
        built_m = build_f3(dep_m, context=ctx_m)
        extra = [c for c in extra_needed if c in built_m.columns]
        payload_m = built_m.select(["MVT_ID_mvt", *extra])
        if shard is not None and not cfg.smoke:
            shard.parent.mkdir(parents=True, exist_ok=True)
            payload_m.write_parquet(shard)
            from prc_challenge.campaign.runtime import persist_repo_file

            persist_repo_file(cfg, shard)
        print(f"F3 month={month} n={payload_m.height}", flush=True)
        parts.append(payload_m)
    payload = pl.concat(parts, how="diagonal") if parts else dep.select(["MVT_ID_mvt"])
    if not cfg.smoke:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload.write_parquet(path)
        from prc_challenge.campaign.runtime import persist_repo_file

        persist_repo_file(cfg, path)
    return dep.join(payload, on="MVT_ID_mvt", how="left")
