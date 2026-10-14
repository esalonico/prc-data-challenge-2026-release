# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Ranking-safe temporal context features for the independent next-step campaign.

This module intentionally keeps context construction separate from supervised
row selection.  It never reads taxi-out, block-time or AOBT labels.  Departure
rows that are excluded from a supervised fit may still contribute observable
movement context.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl

from prc_challenge.campaign.scoring import assert_no_forbidden_features

WINDOWS_SEC = (300, 900, 1800, 3600)

TEMPORAL_V2_NUMS = tuple(
    [f"ctx_dep_past_{w}s" for w in WINDOWS_SEC]
    + [f"ctx_arr_past_{w}s" for w in WINDOWS_SEC]
    + [f"ctx_dep_rwy_past_{w}s" for w in WINDOWS_SEC]
    + [f"ctx_arr_rwy_past_{w}s" for w in WINDOWS_SEC]
    + [f"ctx_sched_dep_next_{w}s" for w in WINDOWS_SEC]
    + [
        "ctx_sec_since_prev_dep",
        "ctx_sec_since_prev_arr",
        "ctx_sec_since_prev_rwy_dep",
        "ctx_sec_since_prev_rwy_arr",
        "ctx_eobt_delay_mean_1800s",
        "ctx_eobt_delay_mean_3600s",
        "ctx_eobt_delay_gt900_count_1800s",
        "ctx_arr_context_available",
    ]
)

_REQUIRED_DEP = ("MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt")


def _require(frame: pl.DataFrame, cols: Iterable[str], *, label: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def _epoch_seconds(frame: pl.DataFrame, col: str) -> np.ndarray:
    if col not in frame.columns:
        return np.full(frame.height, np.nan, dtype=np.float64)
    dtype = frame.schema[col]
    if getattr(dtype, "is_temporal", lambda: False)():
        arr = frame.select(pl.col(col).dt.epoch("s").cast(pl.Float64).alias("_x"))["_x"].to_numpy()
    else:
        arr = frame.select(pl.col(col).cast(pl.Float64, strict=False).alias("_x"))["_x"].to_numpy()
    return np.asarray(arr, dtype=np.float64)


def _numeric(frame: pl.DataFrame, col: str) -> np.ndarray:
    if col not in frame.columns:
        return np.full(frame.height, np.nan, dtype=np.float64)
    arr = frame.select(pl.col(col).cast(pl.Float64, strict=False).alias("_x"))["_x"].to_numpy()
    return np.asarray(arr, dtype=np.float64)


def _schedule_epoch(dep: pl.DataFrame, actual_epoch: np.ndarray) -> np.ndarray:
    """Get scheduled time without requiring the raw timestamp in reduced caches.

    `takeoff_minus_sched_sec` is already a ranking-safe engineered feature.  If
    both representations exist, verify their identity before using the compact
    one so a stale/broken cache fails loudly.
    """
    derived = None
    if "takeoff_minus_sched_sec" in dep.columns:
        derived = actual_epoch - _numeric(dep, "takeoff_minus_sched_sec")
    raw = _epoch_seconds(dep, "SCHED_TIME_UTC_mvt") if "SCHED_TIME_UTC_mvt" in dep.columns else None
    if raw is not None and derived is not None:
        good = np.isfinite(raw) & np.isfinite(derived)
        if good.any():
            max_abs = float(np.max(np.abs(raw[good] - derived[good])))
            if max_abs > 1.0:
                raise RuntimeError(f"SCHED timestamp/delta identity mismatch max_abs={max_abs:.3f}s")
        return np.where(np.isfinite(raw), raw, derived)
    if raw is not None:
        return raw
    if derived is not None:
        return derived
    return np.full(dep.height, np.nan, dtype=np.float64)


def _past_count(target: np.ndarray, events: np.ndarray, window: int, *, include_equal: bool) -> np.ndarray:
    if events.size == 0:
        return np.zeros(target.shape[0], dtype=np.float32)
    ev = np.sort(events[np.isfinite(events)])
    if ev.size == 0:
        return np.zeros(target.shape[0], dtype=np.float32)
    left = np.searchsorted(ev, target - float(window), side="left")
    right = np.searchsorted(ev, target, side="right" if include_equal else "left")
    return (right - left).astype(np.float32)


def _future_count(target: np.ndarray, events: np.ndarray, window: int) -> np.ndarray:
    ev = np.sort(events[np.isfinite(events)])
    if ev.size == 0:
        return np.zeros(target.shape[0], dtype=np.float32)
    left = np.searchsorted(ev, target, side="right")
    right = np.searchsorted(ev, target + float(window), side="right")
    return (right - left).astype(np.float32)


def _previous_gap(target: np.ndarray, events: np.ndarray, *, include_equal: bool) -> np.ndarray:
    ev = np.sort(events[np.isfinite(events)])
    out = np.full(target.shape[0], np.nan, dtype=np.float32)
    if ev.size == 0:
        return out
    pos = np.searchsorted(ev, target, side="right" if include_equal else "left") - 1
    good = pos >= 0
    if good.any():
        out[good] = (target[good] - ev[pos[good]]).astype(np.float32)
    return out


def _window_mean_count(
    target: np.ndarray,
    event_time: np.ndarray,
    values: np.ndarray,
    window: int,
    *,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean and optional count(values > threshold) over [t-window, t).

    `event_time` and `values` correspond to the same departure table.  Sorting
    plus cumulative sums makes this O(n log n) rather than a Python row loop.
    """
    good = np.isfinite(event_time) & np.isfinite(values)
    t = event_time[good]
    v = values[good]
    if t.size == 0:
        mean = np.full(target.shape[0], np.nan, dtype=np.float32)
        count = np.zeros(target.shape[0], dtype=np.float32) if threshold is not None else None
        return mean, count
    order = np.argsort(t, kind="mergesort")
    t, v = t[order], v[order]
    cs = np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])
    cc = np.arange(t.size + 1, dtype=np.int64)
    left = np.searchsorted(t, target - float(window), side="left")
    right = np.searchsorted(t, target, side="left")
    n = cc[right] - cc[left]
    sums = cs[right] - cs[left]
    mean = np.full(target.shape[0], np.nan, dtype=np.float32)
    nz = n > 0
    mean[nz] = (sums[nz] / n[nz]).astype(np.float32)
    if threshold is None:
        return mean, None
    flag = (v > float(threshold)).astype(np.int64)
    fc = np.concatenate([[0], np.cumsum(flag, dtype=np.int64)])
    cnt = (fc[right] - fc[left]).astype(np.float32)
    return mean, cnt


def _as_string(frame: pl.DataFrame, col: str, fill: str = "__NA__") -> np.ndarray:
    if col not in frame.columns:
        return np.full(frame.height, fill, dtype=object)
    return frame[col].cast(pl.String).fill_null(fill).to_numpy()


def build_temporal_context_v2(dep: pl.DataFrame, arr: pl.DataFrame | None = None) -> pl.DataFrame:
    """Build explicit airport/runway temporal state for every supplied DEP row.

    Semantics:
    - departure history uses actual takeoff times strictly before the target;
    - arrival history may include an arrival at exactly the target timestamp;
    - scheduled future demand uses only scheduled DEP times after target takeoff;
    - planning-delay history uses previous departures only;
    - target taxi labels, block times and AOBT are never consumed.

    The output is keyed one-to-one by `MVT_ID_mvt` and contains only numeric
    ranking-safe context features.
    """
    _require(dep, _REQUIRED_DEP, label="DEP context")
    forbidden = {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"}
    consumed = {
        "MVT_ID_mvt",
        "AIRPORT",
        "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt",
        "takeoff_minus_sched_sec",
        "takeoff_minus_eobt_sec",
        "RUNWAY_mvt",
    }
    assert_no_forbidden_features(sorted((consumed & set(dep.columns)) - forbidden))

    n = dep.height
    feat = {name: np.full(n, np.nan, dtype=np.float32) for name in TEMPORAL_V2_NUMS}
    for name in TEMPORAL_V2_NUMS:
        if name.startswith("ctx_dep_") or name.startswith("ctx_arr_") or name.startswith("ctx_sched_"):
            if "sec_since" not in name and "delay_mean" not in name:
                feat[name] = np.zeros(n, dtype=np.float32)
    feat["ctx_eobt_delay_gt900_count_1800s"] = np.zeros(n, dtype=np.float32)
    feat["ctx_arr_context_available"] = np.zeros(n, dtype=np.float32)

    dep_w = dep.with_row_index("_ctx_row")
    actual_all = _epoch_seconds(dep_w, "MVT_TIME_UTC_mvt")
    sched_all = _schedule_epoch(dep_w, actual_all)
    eobt_delay_all = _numeric(dep_w, "takeoff_minus_eobt_sec")
    dep_air = _as_string(dep_w, "AIRPORT")
    dep_rwy = _as_string(dep_w, "RUNWAY_mvt")

    arr_w = arr
    if arr_w is not None and arr_w.height:
        _require(arr_w, ("AIRPORT", "MVT_TIME_UTC_mvt"), label="ARR context")
        arr_air = _as_string(arr_w, "AIRPORT")
        arr_rwy = _as_string(arr_w, "RUNWAY_mvt")
        arr_time = _epoch_seconds(arr_w, "MVT_TIME_UTC_mvt")
    else:
        arr_air = np.empty(0, dtype=object)
        arr_rwy = np.empty(0, dtype=object)
        arr_time = np.empty(0, dtype=np.float64)

    for airport in np.unique(dep_air):
        dep_idx = np.flatnonzero(dep_air == airport)
        target_t = actual_all[dep_idx]
        dep_times = actual_all[dep_idx]
        arr_idx = np.flatnonzero(arr_air == airport) if arr_time.size else np.empty(0, dtype=np.int64)
        arr_times = arr_time[arr_idx] if arr_idx.size else np.empty(0, dtype=np.float64)
        sched_times = sched_all[dep_idx]

        for w in WINDOWS_SEC:
            feat[f"ctx_dep_past_{w}s"][dep_idx] = _past_count(target_t, dep_times, w, include_equal=False)
            feat[f"ctx_arr_past_{w}s"][dep_idx] = _past_count(target_t, arr_times, w, include_equal=True)
            feat[f"ctx_sched_dep_next_{w}s"][dep_idx] = _future_count(target_t, sched_times, w)

        feat["ctx_sec_since_prev_dep"][dep_idx] = _previous_gap(target_t, dep_times, include_equal=False)
        feat["ctx_sec_since_prev_arr"][dep_idx] = _previous_gap(target_t, arr_times, include_equal=True)
        if arr_times.size:
            feat["ctx_arr_context_available"][dep_idx] = 1.0

        mean18, gt18 = _window_mean_count(
            target_t,
            dep_times,
            eobt_delay_all[dep_idx],
            1800,
            threshold=900.0,
        )
        mean36, _ = _window_mean_count(target_t, dep_times, eobt_delay_all[dep_idx], 3600)
        feat["ctx_eobt_delay_mean_1800s"][dep_idx] = mean18
        feat["ctx_eobt_delay_mean_3600s"][dep_idx] = mean36
        if gt18 is not None:
            feat["ctx_eobt_delay_gt900_count_1800s"][dep_idx] = gt18

        # Same-runway context. Missing runway is kept as its own observable state.
        for runway in np.unique(dep_rwy[dep_idx]):
            q_idx = dep_idx[dep_rwy[dep_idx] == runway]
            q_t = actual_all[q_idx]
            dep_r_idx = dep_idx[dep_rwy[dep_idx] == runway]
            dep_r_t = actual_all[dep_r_idx]
            if arr_idx.size:
                arr_r_idx = arr_idx[arr_rwy[arr_idx] == runway]
                arr_r_t = arr_time[arr_r_idx]
            else:
                arr_r_t = np.empty(0, dtype=np.float64)
            for w in WINDOWS_SEC:
                feat[f"ctx_dep_rwy_past_{w}s"][q_idx] = _past_count(q_t, dep_r_t, w, include_equal=False)
                feat[f"ctx_arr_rwy_past_{w}s"][q_idx] = _past_count(q_t, arr_r_t, w, include_equal=True)
            feat["ctx_sec_since_prev_rwy_dep"][q_idx] = _previous_gap(q_t, dep_r_t, include_equal=False)
            feat["ctx_sec_since_prev_rwy_arr"][q_idx] = _previous_gap(q_t, arr_r_t, include_equal=True)

    out = dep.select("MVT_ID_mvt")
    out = out.with_columns([pl.Series(name, feat[name]) for name in TEMPORAL_V2_NUMS])
    if out["MVT_ID_mvt"].n_unique() != out.height:
        raise RuntimeError("temporal context output requires unique MVT_ID_mvt")
    assert_no_forbidden_features(list(TEMPORAL_V2_NUMS))
    return out
