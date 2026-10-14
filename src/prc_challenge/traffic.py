# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Airport movement counts ending at the focal takeoff. Ranking-safe: no DEP labels."""

from __future__ import annotations

import numpy as np
import polars as pl


def add_traffic_counts(
    dep_targets: pl.DataFrame,
    context: pl.DataFrame,
    windows_sec: tuple[int, ...] = (900, 1800, 3600),
) -> pl.DataFrame:
    """Count other movements with MVT_TIME in (t - w, t] at the same airport.

    Uses prefix sums (no pairwise loops). The focal departure is excluded.
    `context` must already be ranking-masked. Month isolation is the caller's job.
    """
    ctx = context.select(["AIRPORT", "PHASE_mvt", "MVT_TIME_UTC_mvt", "MVT_ID_mvt"]).drop_nulls(
        ["AIRPORT", "MVT_TIME_UTC_mvt"]
    )
    tgt = dep_targets.select(["MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt"])
    out_parts: list[pl.DataFrame] = []
    airports = tgt["AIRPORT"].unique().to_list()
    for ap in airports:
        tgt_ap = tgt.filter(pl.col("AIRPORT") == ap)
        ctx_ap = ctx.filter(pl.col("AIRPORT") == ap)
        if ctx_ap.height == 0:
            cols = {"MVT_ID_mvt": tgt_ap["MVT_ID_mvt"]}
            for w in windows_sec:
                z = np.zeros(tgt_ap.height, dtype=np.int32)
                cols[f"n_dep_{w}s"] = z
                cols[f"n_arr_{w}s"] = z
                cols[f"n_mvt_{w}s"] = z
            out_parts.append(pl.DataFrame(cols))
            continue
        times_all = ctx_ap["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
        phase = ctx_ap["PHASE_mvt"].to_numpy()
        order = np.argsort(times_all, kind="mergesort")
        times_all = times_all[order]
        is_dep = (phase[order] == "DEP").astype(np.int32)
        is_arr = (phase[order] == "ARR").astype(np.int32)
        cdep = np.concatenate(([0], np.cumsum(is_dep)))
        carr = np.concatenate(([0], np.cumsum(is_arr)))
        t_tgt = tgt_ap["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
        cols = {"MVT_ID_mvt": tgt_ap["MVT_ID_mvt"]}
        for w in windows_sec:
            left = np.searchsorted(times_all, t_tgt - w, side="left")
            right = np.searchsorted(times_all, t_tgt, side="left")
            n_dep = (cdep[right] - cdep[left]).astype(np.int32)
            n_arr = (carr[right] - carr[left]).astype(np.int32)
            cols[f"n_dep_{w}s"] = n_dep
            cols[f"n_arr_{w}s"] = n_arr
            cols[f"n_mvt_{w}s"] = n_dep + n_arr
        out_parts.append(pl.DataFrame(cols))
    counted = pl.concat(out_parts)
    return dep_targets.join(counted, on="MVT_ID_mvt", how="left")


def add_runway_counts(
    dep_targets: pl.DataFrame,
    context: pl.DataFrame,
    windows_sec: tuple[int, ...] = (900, 1800, 3600),
) -> pl.DataFrame:
    """Same-runway movement counts in [t-w, t). Observed use, not declared config."""
    ctx = context.select(["AIRPORT", "RUNWAY_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]).drop_nulls(
        ["AIRPORT", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]
    )
    tgt = dep_targets.select(["MVT_ID_mvt", "AIRPORT", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
    parts: list[pl.DataFrame] = []
    keys = tgt.select(["AIRPORT", "RUNWAY_mvt"]).unique()
    for row in keys.iter_rows(named=True):
        ap, rwy = row["AIRPORT"], row["RUNWAY_mvt"]
        tgt_g = tgt.filter((pl.col("AIRPORT") == ap) & (pl.col("RUNWAY_mvt") == rwy))
        ctx_g = ctx.filter((pl.col("AIRPORT") == ap) & (pl.col("RUNWAY_mvt") == rwy))
        t_tgt = tgt_g["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
        cols: dict[str, object] = {"MVT_ID_mvt": tgt_g["MVT_ID_mvt"]}
        if ctx_g.height == 0:
            for w in windows_sec:
                z = np.zeros(tgt_g.height, dtype=np.int32)
                cols[f"n_dep_rwy_{w}s"] = z
                cols[f"n_arr_rwy_{w}s"] = z
            parts.append(pl.DataFrame(cols))
            continue
        times = ctx_g["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
        phase = ctx_g["PHASE_mvt"].to_numpy()
        order = np.argsort(times, kind="mergesort")
        times = times[order]
        is_dep = (phase[order] == "DEP").astype(np.int32)
        is_arr = (phase[order] == "ARR").astype(np.int32)
        cdep = np.concatenate(([0], np.cumsum(is_dep)))
        carr = np.concatenate(([0], np.cumsum(is_arr)))
        for w in windows_sec:
            left = np.searchsorted(times, t_tgt - w, side="left")
            right = np.searchsorted(times, t_tgt, side="left")
            cols[f"n_dep_rwy_{w}s"] = (cdep[right] - cdep[left]).astype(np.int32)
            cols[f"n_arr_rwy_{w}s"] = (carr[right] - carr[left]).astype(np.int32)
        parts.append(pl.DataFrame(cols))
    return dep_targets.join(pl.concat(parts), on="MVT_ID_mvt", how="left")


def add_forward_counts(
    dep_targets: pl.DataFrame,
    context: pl.DataFrame,
    windows_sec: tuple[int, ...] = (900, 1800, 3600),
) -> pl.DataFrame:
    """Count other movements with MVT_TIME in (t, t + w] at the same airport.

    Retrospective reconstruction: ranking months still contain those later takeoffs.
    Month isolation is the caller's job. Focal row is excluded by using a right-open
    start at t (searchsorted side='right' on t).
    """
    ctx = context.select(["AIRPORT", "PHASE_mvt", "MVT_TIME_UTC_mvt"]).drop_nulls(
        ["AIRPORT", "MVT_TIME_UTC_mvt"]
    )
    tgt = dep_targets.select(["MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt"])
    out_parts: list[pl.DataFrame] = []
    for ap in tgt["AIRPORT"].unique().to_list():
        tgt_ap = tgt.filter(pl.col("AIRPORT") == ap)
        ctx_ap = ctx.filter(pl.col("AIRPORT") == ap)
        t_tgt = tgt_ap["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
        cols: dict[str, object] = {"MVT_ID_mvt": tgt_ap["MVT_ID_mvt"]}
        if ctx_ap.height == 0:
            for w in windows_sec:
                z = np.zeros(tgt_ap.height, dtype=np.int32)
                cols[f"n_dep_fwd_{w}s"] = z
                cols[f"n_arr_fwd_{w}s"] = z
            out_parts.append(pl.DataFrame(cols))
            continue
        times_all = ctx_ap["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
        phase = ctx_ap["PHASE_mvt"].to_numpy()
        order = np.argsort(times_all, kind="mergesort")
        times_all = times_all[order]
        is_dep = (phase[order] == "DEP").astype(np.int32)
        is_arr = (phase[order] == "ARR").astype(np.int32)
        cdep = np.concatenate(([0], np.cumsum(is_dep)))
        carr = np.concatenate(([0], np.cumsum(is_arr)))
        for w in windows_sec:
            left = np.searchsorted(times_all, t_tgt, side="right")
            right = np.searchsorted(times_all, t_tgt + w, side="right")
            cols[f"n_dep_fwd_{w}s"] = (cdep[right] - cdep[left]).astype(np.int32)
            cols[f"n_arr_fwd_{w}s"] = (carr[right] - carr[left]).astype(np.int32)
        out_parts.append(pl.DataFrame(cols))
    return dep_targets.join(pl.concat(out_parts), on="MVT_ID_mvt", how="left")


def add_planned_demand(
    dep_targets: pl.DataFrame,
    context: pl.DataFrame,
    windows_sec: tuple[int, ...] = (1800, 3600),
) -> pl.DataFrame:
    """Count other departures whose SCHED falls in [t - w, t + w] at the airport.

    Uses scheduled time, not observed takeoff. Focal SCHED inside the window is
    subtracted so the count is of *other* planned departures.
    """
    ctx = (
        context.filter(pl.col("PHASE_mvt") == "DEP")
        .select(["AIRPORT", "SCHED_TIME_UTC_mvt"])
        .drop_nulls(["AIRPORT", "SCHED_TIME_UTC_mvt"])
    )
    tgt = dep_targets.select(["MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    out_parts: list[pl.DataFrame] = []
    for ap in tgt["AIRPORT"].unique().to_list():
        tgt_ap = tgt.filter(pl.col("AIRPORT") == ap)
        ctx_ap = ctx.filter(pl.col("AIRPORT") == ap)
        t_tgt = tgt_ap["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
        sched_self = tgt_ap["SCHED_TIME_UTC_mvt"].dt.epoch("s").fill_null(-10**18).to_numpy()
        cols: dict[str, object] = {"MVT_ID_mvt": tgt_ap["MVT_ID_mvt"]}
        if ctx_ap.height == 0:
            for w in windows_sec:
                cols[f"n_sched_dep_{w}s"] = np.zeros(tgt_ap.height, dtype=np.int32)
            out_parts.append(pl.DataFrame(cols))
            continue
        sched = ctx_ap["SCHED_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
        sched = np.sort(sched, kind="mergesort")
        c = np.concatenate(([0], np.arange(1, sched.size + 1, dtype=np.int32)))
        for w in windows_sec:
            left = np.searchsorted(sched, t_tgt - w, side="left")
            right = np.searchsorted(sched, t_tgt + w, side="right")
            n = (c[right] - c[left]).astype(np.int32)
            own = ((sched_self >= t_tgt - w) & (sched_self <= t_tgt + w)).astype(np.int32)
            cols[f"n_sched_dep_{w}s"] = np.maximum(n - own, 0)
        out_parts.append(pl.DataFrame(cols))
    return dep_targets.join(pl.concat(out_parts), on="MVT_ID_mvt", how="left")


def add_runway_config_features(
    dep_targets: pl.DataFrame,
    context: pl.DataFrame,
    window_sec: int = 3600,
) -> pl.DataFrame:
    """Observed departure-runway mix in [t-w, t). Not declared ATIS config.

    Features: unique runways used, share on the focal runway, Herfindahl
    concentration, and whether the modal runway flipped between the older and
    newer halves of the window. Month isolation is the caller's job. Focal
    takeoff is excluded by using a right-open end at t.
    """
    ctx = (
        context.filter(pl.col("PHASE_mvt") == "DEP")
        .select(["AIRPORT", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
        .drop_nulls(["AIRPORT", "MVT_TIME_UTC_mvt"])
    )
    tgt = dep_targets.select(["MVT_ID_mvt", "AIRPORT", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
    half = window_sec // 2
    parts: list[pl.DataFrame] = []
    for ap in tgt["AIRPORT"].unique().to_list():
        tgt_ap = tgt.filter(pl.col("AIRPORT") == ap)
        ctx_ap = ctx.filter(pl.col("AIRPORT") == ap)
        n = tgt_ap.height
        n_used = np.zeros(n, dtype=np.int32)
        share = np.full(n, np.nan, dtype=np.float64)
        hhi = np.full(n, np.nan, dtype=np.float64)
        changed = np.zeros(n, dtype=np.int8)
        if ctx_ap.height:
            times = ctx_ap["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
            rwys = ctx_ap["RUNWAY_mvt"].fill_null("__NA__").to_numpy()
            order = np.argsort(times, kind="mergesort")
            times = times[order]
            rwys = rwys[order]
            t_tgt = tgt_ap["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
            focal = tgt_ap["RUNWAY_mvt"].fill_null("__NA__").to_numpy()
            left = np.searchsorted(times, t_tgt - window_sec, side="left")
            mid = np.searchsorted(times, t_tgt - half, side="left")
            right = np.searchsorted(times, t_tgt, side="left")
            for i in range(n):
                lo, hi = int(left[i]), int(right[i])
                if hi <= lo:
                    continue
                window = rwys[lo:hi]
                n_used[i] = int(np.unique(window).size)
                counts = {}
                for r in window:
                    counts[r] = counts.get(r, 0) + 1
                total = float(hi - lo)
                share[i] = counts.get(focal[i], 0) / total
                hhi[i] = sum((c / total) ** 2 for c in counts.values())
                old, new = rwys[lo : int(mid[i])], rwys[int(mid[i]) : hi]
                if old.size and new.size:
                    changed[i] = np.int8(_modal(old) != _modal(new))
        parts.append(
            pl.DataFrame(
                {
                    "MVT_ID_mvt": tgt_ap["MVT_ID_mvt"],
                    "n_rwy_used_3600s": n_used,
                    "rwy_share_3600s": share,
                    "rwy_hhi_3600s": hhi,
                    "rwy_modal_changed_1800s": changed,
                }
            )
        )
    return dep_targets.join(pl.concat(parts), on="MVT_ID_mvt", how="left")


def _modal(values: np.ndarray) -> object:
    uniq, counts = np.unique(values, return_counts=True)
    return uniq[int(np.argmax(counts))]


def add_prev_same_rwy_wake(dep_targets: pl.DataFrame, context: pl.DataFrame) -> pl.DataFrame:
    """Wake category of the previous departure on the same airport+runway."""
    ctx = (
        context.filter(pl.col("PHASE_mvt") == "DEP")
        .select(["AIRPORT", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "WK_TBL_CAT_flt"])
        .drop_nulls(["AIRPORT", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
    )
    tgt = dep_targets.select(["MVT_ID_mvt", "AIRPORT", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
    parts: list[pl.DataFrame] = []
    keys = tgt.select(["AIRPORT", "RUNWAY_mvt"]).unique()
    for row in keys.iter_rows(named=True):
        ap, rwy = row["AIRPORT"], row["RUNWAY_mvt"]
        tgt_g = tgt.filter((pl.col("AIRPORT") == ap) & (pl.col("RUNWAY_mvt") == rwy))
        ctx_g = ctx.filter((pl.col("AIRPORT") == ap) & (pl.col("RUNWAY_mvt") == rwy))
        n = tgt_g.height
        wake = np.array(["__NONE__"] * n, dtype=object)
        gap = np.full(n, np.nan, dtype=np.float64)
        if ctx_g.height:
            times = ctx_g["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
            cats = ctx_g["WK_TBL_CAT_flt"].fill_null("__UNK__").to_numpy()
            order = np.argsort(times, kind="mergesort")
            times = times[order]
            cats = cats[order]
            t_tgt = tgt_g["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
            idx = np.searchsorted(times, t_tgt, side="left") - 1
            ok = idx >= 0
            wake[ok] = cats[idx[ok]]
            gap[ok] = t_tgt[ok] - times[idx[ok]]
        parts.append(
            pl.DataFrame(
                {
                    "MVT_ID_mvt": tgt_g["MVT_ID_mvt"],
                    "prev_rwy_wake": wake.astype(str),
                    "sec_since_prev_rwy_dep": gap,
                }
            )
        )
    if not parts:
        return dep_targets
    return dep_targets.join(pl.concat(parts), on="MVT_ID_mvt", how="left")


def add_prev_dep_gap(df: pl.DataFrame) -> pl.DataFrame:
    """Seconds since the previous departure takeoff at the same airport."""
    dep = df.filter(pl.col("PHASE_mvt") == "DEP").sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    dep = dep.with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").diff().over("AIRPORT").alias("sec_since_prev_dep")
    )
    extra = dep.select(["MVT_ID_mvt", "sec_since_prev_dep"])
    return df.join(extra, on="MVT_ID_mvt", how="left")


DELAY_BUNCH_WINDOWS = (900, 1800, 3600)
DELAY_BUNCH_THRESHOLD_SEC = 1800
DELAY_BUNCH_NUMS = tuple(f"n_delayed_dep_ctr_{w}s" for w in DELAY_BUNCH_WINDOWS)


def add_delayed_bunching(
    dep_targets: pl.DataFrame,
    context: pl.DataFrame,
    windows_sec: tuple[int, ...] = DELAY_BUNCH_WINDOWS,
    delay_sec: int = DELAY_BUNCH_THRESHOLD_SEC,
) -> pl.DataFrame:
    """Count other delayed departures with takeoff in (t-w, t+w] at the same airport.

    Delayed means takeoff−EOBT ≥ `delay_sec`. Uses ranking-available takeoff and
    EOBT only. The focal row is excluded. Arrivals are ignored. Month isolation
    is the caller's job. Centered windows are retrospective reconstruction, not
    an operational forecast.
    """
    ctx = context
    if "PHASE_mvt" in ctx.columns:
        ctx = ctx.filter(pl.col("PHASE_mvt") == "DEP")
    ctx = ctx.select(["AIRPORT", "MVT_TIME_UTC_mvt", "EOBT_1_flt", "MVT_ID_mvt"]).drop_nulls(
        ["AIRPORT", "MVT_TIME_UTC_mvt"]
    )
    delay_s = (
        ctx["MVT_TIME_UTC_mvt"].dt.epoch("s").cast(pl.Float64)
        - ctx["EOBT_1_flt"].dt.epoch("s").cast(pl.Float64)
    )
    ctx = ctx.with_columns(
        (delay_s.is_not_null() & delay_s.is_finite() & (delay_s >= delay_sec))
        .fill_null(False)
        .cast(pl.Int32)
        .alias("_delayed")
    )
    tgt = dep_targets.select(["MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt"])
    self_flag = tgt.join(ctx.select(["MVT_ID_mvt", "_delayed"]), on="MVT_ID_mvt", how="left").with_columns(
        pl.col("_delayed").fill_null(0)
    )
    parts: list[pl.DataFrame] = []
    for ap in tgt["AIRPORT"].unique().to_list():
        tgt_ap = tgt.filter(pl.col("AIRPORT") == ap)
        ctx_ap = ctx.filter(pl.col("AIRPORT") == ap)
        self_ap = self_flag.filter(pl.col("AIRPORT") == ap)["_delayed"].to_numpy().astype(np.int32)
        cols: dict[str, object] = {"MVT_ID_mvt": tgt_ap["MVT_ID_mvt"]}
        t_tgt = tgt_ap["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
        if ctx_ap.height == 0:
            for w in windows_sec:
                cols[f"n_delayed_dep_ctr_{w}s"] = np.zeros(tgt_ap.height, dtype=np.int32)
            parts.append(pl.DataFrame(cols))
            continue
        t = ctx_ap["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
        d = ctx_ap["_delayed"].to_numpy().astype(np.int32)
        order = np.argsort(t, kind="mergesort")
        t_s = t[order]
        csum = np.concatenate(([0], np.cumsum(d[order])))
        for w in windows_sec:
            left = np.searchsorted(t_s, t_tgt - w, side="left")
            right = np.searchsorted(t_s, t_tgt + w, side="right")
            n = (csum[right] - csum[left]).astype(np.int32) - self_ap
            cols[f"n_delayed_dep_ctr_{w}s"] = np.clip(n, 0, None)
        parts.append(pl.DataFrame(cols))
    return dep_targets.join(pl.concat(parts), on="MVT_ID_mvt", how="left")
