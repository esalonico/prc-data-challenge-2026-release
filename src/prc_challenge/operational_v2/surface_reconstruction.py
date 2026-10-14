# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Native movement-relationship features for surface reconstruction.

The builders in this module are deliberately label-free.  Departure taxi,
departure block and AOBT are rejected before an event table can be built.
Arrival completion is retained as a separate, phase-validated projection.
Every temporal operation is scoped to one observable month supplied by the
caller, so disjoint January and July windows cannot be connected accidentally.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl


FORBIDDEN_DEPARTURE_COLUMNS = frozenset(
    {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt"}
)
MISSING_STANDS = frozenset({"", "NA", "N/A", "NONE", "NULL", "UNKNOWN", "UNK", "-", "?"})
WINDOWS_SEC = (300, 900, 1800)
OCCUPANCY_OFFSETS_SEC = (0, 300, 900, 1800)


def _require(frame: pl.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def reject_forbidden_departure_columns(frame: pl.DataFrame) -> None:
    bad = sorted(FORBIDDEN_DEPARTURE_COLUMNS.intersection(frame.columns))
    if bad:
        raise RuntimeError(f"departure feature input contains forbidden columns: {bad}")


def learned_feature_cache_identity(
    *,
    model: str,
    split_id: str,
    source_sha256: str,
    feature_config: Mapping[str, Any],
    month: int,
) -> dict[str, Any]:
    """Identity for fold-learned S3/S4 caches. A mismatch must be rejected, not reused."""
    return {
        "model": str(model),
        "split_id": str(split_id),
        "source_sha256": str(source_sha256),
        "feature_config": dict(feature_config),
        "month": int(month),
    }


def learned_feature_cache_hash(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def assert_learned_cache_identity(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    stored_hash = learned_feature_cache_hash(stored)
    expected_hash = learned_feature_cache_hash(expected)
    if stored_hash != expected_hash:
        raise RuntimeError(
            "learned feature cache/recipe mismatch: "
            f"stored={stored_hash} expected={expected_hash}"
        )


def write_learned_cache_identity(path: Path, identity: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**dict(identity), "identity_sha256": learned_feature_cache_hash(identity)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_learned_cache_or_reject(parquet_path: Path, expected: Mapping[str, Any]) -> pl.DataFrame:
    identity_path = parquet_path.with_suffix(".identity.json")
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if not identity_path.exists():
        raise RuntimeError(f"learned cache missing identity sidecar: {identity_path}")
    stored = json.loads(identity_path.read_text())
    stored.pop("identity_sha256", None)
    assert_learned_cache_identity(stored, expected)
    return pl.read_parquet(parquet_path)


def _normalised_text(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.to_uppercase()
        .str.strip_chars()
        .replace("", None)
    )


def _stand_identity(column: str, airport_column: str = "airport") -> pl.Expr:
    value = _normalised_text(column)
    return (
        pl.when(value.is_null() | value.is_in(list(MISSING_STANDS)))
        .then(None)
        .otherwise(pl.concat_str([pl.col(airport_column), value], separator="::"))
    )


def _runway_region(column: str) -> pl.Expr:
    return _normalised_text(column).str.replace(r"[LCR]$", "")


def build_event_table(raw: pl.DataFrame, *, source_month: int) -> pl.DataFrame:
    """Build a compact movement table without hidden departure outcomes.

    The caller must remove the forbidden departure columns first.  Arrival
    completion can be supplied separately as ``ARR_BLOCK_TIME_UTC`` and is
    ignored on departure rows.  This makes phase provenance mechanically
    visible instead of relying on a later feature builder to remember it.
    """

    reject_forbidden_departure_columns(raw)
    _require(
        raw,
        (
            "MVT_ID_mvt",
            "PHASE_mvt",
            "MVT_TIME_UTC_mvt",
            "ADEP_mvt",
            "ADES_mvt",
        ),
        label="movement input",
    )
    if raw["MVT_ID_mvt"].null_count():
        raise RuntimeError("movement input contains null exact IDs")
    phase = pl.col("PHASE_mvt").cast(pl.String).str.to_uppercase()
    airport = pl.when(phase == "DEP").then(_normalised_text("ADEP_mvt")).otherwise(
        _normalised_text("ADES_mvt")
    )
    columns: list[pl.Expr] = [
        pl.col("MVT_ID_mvt").alias("event_id"),
        phase.alias("phase"),
        airport.alias("airport"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("event_epoch"),
        pl.lit(int(source_month)).cast(pl.Int8).alias("source_month"),
    ]
    optional = {
        "RUNWAY_mvt": "runway_raw",
        "STAND_mvt": "stand_raw",
        "AIRCRAFT_TYPE_mvt": "aircraft_type",
        "WK_TBL_CAT_flt": "wake_class",
        "SCHED_TIME_UTC_mvt": "schedule_time",
        "EOBT_1_flt": "eobt_time",
        "FLIGHT_ID_mvt": "flight_id",
    }
    for source, target in optional.items():
        if source in raw.columns:
            if source.endswith("TIME_UTC_mvt") or source == "EOBT_1_flt":
                columns.append(pl.col(source).dt.epoch("s").cast(pl.Int64).alias(f"{target}_epoch"))
            else:
                columns.append(pl.col(source).alias(target))
        else:
            missing_name = (
                f"{target}_epoch"
                if source.endswith("TIME_UTC_mvt") or source == "EOBT_1_flt"
                else target
            )
            columns.append(pl.lit(None).alias(missing_name))
    if "ARR_BLOCK_TIME_UTC" in raw.columns:
        columns.append(
            pl.when(phase == "ARR")
            .then(pl.col("ARR_BLOCK_TIME_UTC").dt.epoch("s").cast(pl.Int64))
            .otherwise(None)
            .alias("arrival_complete_epoch")
        )
    else:
        columns.append(pl.lit(None).cast(pl.Int64).alias("arrival_complete_epoch"))
    out = raw.select(columns).with_columns(
        _normalised_text("runway_raw").alias("runway"),
        _runway_region("runway_raw").alias("runway_region"),
        _stand_identity("stand_raw").alias("stand_id"),
        _normalised_text("aircraft_type").alias("aircraft_type"),
        _normalised_text("wake_class").alias("wake_class"),
    )
    if out.filter(~pl.col("phase").is_in(["ARR", "DEP"])).height:
        raise RuntimeError("event table contains an unsupported movement phase")
    if out["event_id"].n_unique() != out.height:
        raise RuntimeError("event table contains conflicting duplicate exact IDs")
    return out


def _group_indices(values: np.ndarray) -> dict[object, np.ndarray]:
    groups: dict[object, list[int]] = {}
    for index, value in enumerate(values):
        groups.setdefault(value, []).append(index)
    return {key: np.asarray(indexes, dtype=np.int64) for key, indexes in groups.items()}


def _counts(target: np.ndarray, events: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    events = np.sort(events[np.isfinite(events)])
    if events.size == 0:
        z = np.zeros(target.size, dtype=np.float32)
        return z, z.copy()
    past = np.searchsorted(events, target, side="left") - np.searchsorted(
        events, target - window, side="left"
    )
    future = np.searchsorted(events, target + window, side="right") - np.searchsorted(
        events, target, side="right"
    )
    return past.astype(np.float32), future.astype(np.float32)


def _gaps(target: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    events = np.sort(events[np.isfinite(events)])
    prev = np.full(target.size, np.nan, dtype=np.float32)
    nxt = np.full(target.size, np.nan, dtype=np.float32)
    if events.size == 0:
        return prev, nxt
    left = np.searchsorted(events, target, side="left") - 1
    right = np.searchsorted(events, target, side="right")
    good = left >= 0
    prev[good] = (target[good] - events[left[good]]).astype(np.float32)
    good = right < events.size
    nxt[good] = (events[right[good]] - target[good]).astype(np.float32)
    return prev, nxt


def build_retrospective_runway_features(events: pl.DataFrame) -> pl.DataFrame:
    """Create S1 actual future, symmetric and service-episode features."""

    _require(events, ("event_id", "phase", "airport", "event_epoch", "source_month"), label="events")
    dep = events.filter(pl.col("phase") == "DEP")
    n = dep.height
    ids = dep["event_id"]
    target = dep["event_epoch"].cast(pl.Float64).to_numpy()
    airports = dep["airport"].cast(pl.String).fill_null("__AIRPORT_MISSING__").to_numpy()
    runways = dep["runway_region"].cast(pl.String).fill_null("__RUNWAY_MISSING__").to_numpy()
    event_airports = events["airport"].cast(pl.String).fill_null("__AIRPORT_MISSING__").to_numpy()
    event_runways = events["runway_region"].cast(pl.String).fill_null("__RUNWAY_MISSING__").to_numpy()
    event_phase = events["phase"].to_numpy()
    event_time = events["event_epoch"].cast(pl.Float64).to_numpy()
    event_wake = events["wake_class"].cast(pl.String).fill_null("__WAKE_MISSING__").to_numpy()
    event_schedule = events["schedule_time_epoch"].cast(pl.Float64, strict=False).to_numpy()
    values: dict[str, np.ndarray] = {}
    for window in WINDOWS_SEC:
        for name in ("dep_past", "dep_future", "arr_past", "arr_future", "all_past", "all_future", "rwy_past", "rwy_future"):
            values[f"s1_{name}_{window}s"] = np.zeros(n, dtype=np.float32)
    for name in (
        "s1_sec_since_prev_movement",
        "s1_sec_to_next_movement",
        "s1_sec_since_prev_runway_movement",
        "s1_sec_to_next_runway_movement",
        "s1_prev_runway_headway_sec",
        "s1_next_runway_headway_sec",
        "s1_runway_change_age_sec",
        "s1_arrival_interruption_next_sec",
    ):
        values[name] = np.full(n, np.nan, dtype=np.float32)
    values["s1_equal_timestamp_other_count"] = np.zeros(n, dtype=np.float32)
    values["s1_future_wake_heavy_share_1800s"] = np.zeros(n, dtype=np.float32)
    values["s1_planned_dep_next_1800s"] = np.zeros(n, dtype=np.float32)
    values["s1_runway_config_agrees"] = np.zeros(n, dtype=np.float32)
    values["s1_left_coverage_1800s"] = np.zeros(n, dtype=np.float32)
    values["s1_right_coverage_1800s"] = np.zeros(n, dtype=np.float32)

    for airport, query_index in _group_indices(airports).items():
        source_index = np.flatnonzero(event_airports == airport)
        qtime = target[query_index]
        stime = event_time[source_index]
        sphase = event_phase[source_index]
        for window in WINDOWS_SEC:
            all_p, all_f = _counts(qtime, stime, window)
            dep_p, dep_f = _counts(qtime, stime[sphase == "DEP"], window)
            arr_p, arr_f = _counts(qtime, stime[sphase == "ARR"], window)
            values[f"s1_all_past_{window}s"][query_index] = all_p
            values[f"s1_all_future_{window}s"][query_index] = all_f
            values[f"s1_dep_past_{window}s"][query_index] = dep_p
            values[f"s1_dep_future_{window}s"][query_index] = dep_f
            values[f"s1_arr_past_{window}s"][query_index] = arr_p
            values[f"s1_arr_future_{window}s"][query_index] = arr_f
        prev, nxt = _gaps(qtime, stime)
        values["s1_sec_since_prev_movement"][query_index] = prev
        values["s1_sec_to_next_movement"][query_index] = nxt
        if stime.size:
            values["s1_left_coverage_1800s"][query_index] = (
                qtime - np.nanmin(stime) >= 1800
            ).astype(np.float32)
            values["s1_right_coverage_1800s"][query_index] = (
                np.nanmax(stime) - qtime >= 1800
            ).astype(np.float32)
            left = np.searchsorted(np.sort(stime), qtime, side="left")
            right = np.searchsorted(np.sort(stime), qtime, side="right")
            values["s1_equal_timestamp_other_count"][query_index] = np.maximum(
                0, right - left - 1
            ).astype(np.float32)
        arr_times = stime[sphase == "ARR"]
        _, next_arr = _gaps(qtime, arr_times)
        values["s1_arrival_interruption_next_sec"][query_index] = next_arr
        dep_source = source_index[sphase == "DEP"]
        if dep_source.size:
            _, planned = _counts(qtime, event_schedule[dep_source], 1800)
            values["s1_planned_dep_next_1800s"][query_index] = planned

        valid_config = event_runways[source_index] != "__RUNWAY_MISSING__"
        config_time = stime[valid_config]
        config_runway = event_runways[source_index][valid_config]
        if config_time.size:
            order_config = np.argsort(config_time, kind="mergesort")
            config_time = config_time[order_config]
            config_runway = config_runway[order_config]
            segment_start = np.empty(config_time.size, dtype=np.float64)
            start = config_time[0]
            for position in range(config_time.size):
                if position and config_runway[position] != config_runway[position - 1]:
                    start = config_time[position]
                segment_start[position] = start
            previous = np.searchsorted(config_time, qtime, side="left") - 1
            good_previous = previous >= 0
            target_rows = query_index[good_previous]
            previous_position = previous[good_previous]
            values["s1_runway_change_age_sec"][target_rows] = (
                qtime[good_previous] - segment_start[previous_position]
            ).astype(np.float32)
            values["s1_runway_config_agrees"][target_rows] = (
                config_runway[previous_position] == runways[target_rows]
            ).astype(np.float32)

        for runway in np.unique(runways[query_index]):
            qidx = query_index[runways[query_index] == runway]
            rsource = source_index[event_runways[source_index] == runway]
            rtime = event_time[rsource]
            for window in WINDOWS_SEC:
                rp, rf = _counts(target[qidx], rtime, window)
                values[f"s1_rwy_past_{window}s"][qidx] = rp
                values[f"s1_rwy_future_{window}s"][qidx] = rf
            prev, nxt = _gaps(target[qidx], rtime)
            values["s1_sec_since_prev_runway_movement"][qidx] = prev
            values["s1_sec_to_next_runway_movement"][qidx] = nxt
            values["s1_prev_runway_headway_sec"][qidx] = prev
            values["s1_next_runway_headway_sec"][qidx] = nxt

        # Input-only wake mixture in the strictly future 30 minute window.
        order = np.argsort(stime, kind="mergesort")
        st, sw = stime[order], event_wake[source_index][order]
        for out_index, q in zip(query_index, qtime, strict=True):
            lo = np.searchsorted(st, q, side="right")
            hi = np.searchsorted(st, q + 1800, side="right")
            if hi > lo:
                wake = sw[lo:hi]
                values["s1_future_wake_heavy_share_1800s"][out_index] = float(
                    np.mean(np.isin(wake, ["H", "J", "HEAVY", "SUPER"]))
                )

    out = dep.select(pl.col("event_id").alias("MVT_ID_mvt"))
    out = out.with_columns([pl.Series(name, value) for name, value in values.items()])
    out = out.with_columns(
        *[
            (pl.col(f"s1_all_future_{window}s") - pl.col(f"s1_all_past_{window}s")).alias(
                f"s1_rate_imbalance_{window}s"
            )
            for window in WINDOWS_SEC
        ],
        (
            pl.col("s1_all_past_300s") + pl.col("s1_all_future_300s")
        ).alias("s1_symmetric_actual_300s"),
        (
            pl.col("s1_all_past_900s") + pl.col("s1_all_future_900s")
        ).alias("s1_symmetric_actual_900s"),
        (
            pl.col("s1_all_past_1800s") + pl.col("s1_all_future_1800s")
        ).alias("s1_symmetric_actual_1800s"),
    )
    return out


@dataclass(frozen=True)
class ServiceReference:
    idle_threshold_sec: float
    restart_threshold_sec: float
    median_headway_by_airport: Mapping[str, float]


def fit_service_reference(
    features: pl.DataFrame,
    airports: Sequence[str],
    *,
    idle_quantile: float = 0.90,
    restart_quantile: float = 0.25,
) -> ServiceReference:
    prev = features["s1_prev_runway_headway_sec"].cast(pl.Float64).to_numpy()
    finite = prev[np.isfinite(prev)]
    if finite.size == 0:
        return ServiceReference(1800.0, 120.0, {})
    medians: dict[str, float] = {}
    airport_values = np.asarray(airports, dtype=object)
    for airport in np.unique(airport_values):
        cell = prev[airport_values == airport]
        cell = cell[np.isfinite(cell)]
        if cell.size >= 500:
            medians[str(airport)] = float(np.median(cell))
    return ServiceReference(
        idle_threshold_sec=float(np.quantile(finite, idle_quantile)),
        restart_threshold_sec=float(np.quantile(finite, restart_quantile)),
        median_headway_by_airport=medians,
    )


def apply_service_reference(
    features: pl.DataFrame, airports: Sequence[str], reference: ServiceReference
) -> pl.DataFrame:
    airport_values = np.asarray(airports, dtype=object)
    global_median = float(np.median(list(reference.median_headway_by_airport.values()))) if reference.median_headway_by_airport else 120.0
    denominator = np.asarray(
        [reference.median_headway_by_airport.get(str(x), global_median) for x in airport_values],
        dtype=np.float64,
    )
    prev = features["s1_prev_runway_headway_sec"].cast(pl.Float64).to_numpy()
    nxt = features["s1_next_runway_headway_sec"].cast(pl.Float64).to_numpy()
    return features.with_columns(
        pl.Series("s1_prev_headway_relative", prev / np.maximum(denominator, 1.0)),
        pl.Series("s1_next_headway_relative", nxt / np.maximum(denominator, 1.0)),
        pl.Series("s1_long_idle", (prev >= reference.idle_threshold_sec).astype(np.int8)),
        pl.Series(
            "s1_restart_burst",
            ((prev >= reference.idle_threshold_sec) & (nxt <= reference.restart_threshold_sec)).astype(np.int8),
        ),
        (
            (pl.col("s1_prev_runway_headway_sec") >= reference.idle_threshold_sec)
            & ((pl.col("s1_planned_dep_next_1800s") >= 3) | (pl.col("s1_arr_future_1800s") >= 3))
        ).cast(pl.Int8).alias("s1_apparent_capacity_loss"),
        (
            (pl.col("s1_prev_runway_headway_sec") >= reference.idle_threshold_sec)
            & (pl.col("s1_planned_dep_next_1800s") < 3)
            & (pl.col("s1_arr_future_1800s") < 3)
        ).cast(pl.Int8).alias("s1_idle_low_demand"),
    )


def build_stand_turnover_features(events: pl.DataFrame, *, max_history_sec: int = 12 * 3600) -> pl.DataFrame:
    """Construct soft stand-occupancy intervals without consulting taxi labels."""

    dep = events.filter(pl.col("phase") == "DEP")
    arrivals = events.filter(
        (pl.col("phase") == "ARR")
        & pl.col("stand_id").is_not_null()
        & pl.col("arrival_complete_epoch").is_not_null()
    )
    n = dep.height
    t = dep["event_epoch"].cast(pl.Float64).to_numpy()
    dep_stand = dep["stand_id"].cast(pl.String).fill_null("__NO_STAND__").to_numpy()
    arr_stand = arrivals["stand_id"].cast(pl.String).to_numpy()
    arr_time = arrivals["arrival_complete_epoch"].cast(pl.Float64).to_numpy()
    lower = np.full(n, np.nan, dtype=np.float32)
    upper = np.full(n, np.nan, dtype=np.float32)
    width = np.full(n, np.nan, dtype=np.float32)
    latest_age = np.full(n, np.nan, dtype=np.float32)
    previous_age = np.full(n, np.nan, dtype=np.float32)
    next_gap = np.full(n, np.nan, dtype=np.float32)
    support = np.zeros(n, dtype=np.float32)
    ambiguous = np.zeros(n, dtype=np.int8)
    missing = (dep_stand == "__NO_STAND__").astype(np.int8)

    for stand, qidx in _group_indices(dep_stand).items():
        if stand == "__NO_STAND__":
            continue
        times = np.sort(arr_time[arr_stand == stand])
        if times.size == 0:
            continue
        for i in qidx:
            pos = int(np.searchsorted(times, t[i], side="left"))
            history = times[max(0, pos - 3) : pos]
            history = history[t[i] - history <= max_history_sec]
            future = times[pos : pos + 1]
            support[i] = float(history.size)
            if history.size:
                latest_age[i] = float(t[i] - history[-1])
                upper[i] = latest_age[i]
            if history.size >= 2:
                previous_age[i] = float(t[i] - history[-2])
                # Consecutive arrivals define the informative soft interval.
                lower[i] = latest_age[i]
                upper[i] = previous_age[i]
                ambiguous[i] = int(history.size >= 3)
            elif history.size == 1:
                lower[i] = 0.0
            if future.size:
                next_gap[i] = float(future[0] - t[i])
            if np.isfinite(lower[i]) and np.isfinite(upper[i]):
                width[i] = max(0.0, float(upper[i] - lower[i]))
                if upper[i] < lower[i]:
                    ambiguous[i] = 1

    return dep.select(pl.col("event_id").alias("MVT_ID_mvt")).with_columns(
        pl.Series("s2_bound_lower_sec", lower),
        pl.Series("s2_bound_upper_sec", upper),
        pl.Series("s2_bound_width_sec", width),
        pl.Series("s2_latest_arrival_age_sec", latest_age),
        pl.Series("s2_previous_arrival_age_sec", previous_age),
        pl.Series("s2_next_arrival_gap_sec", next_gap),
        pl.Series("s2_chain_support", support),
        pl.Series("s2_chain_ambiguous", ambiguous),
        pl.Series("s2_missing_stand", missing),
        pl.Series(
            "s2_chain_confident",
            (
                (support >= 2)
                & np.isfinite(width)
                & (width >= 300)
                & (width <= max_history_sec)
                & (ambiguous == 0)
            ).astype(np.int8),
        ),
    )


@dataclass(frozen=True)
class DurationReference:
    global_duration_sec: float
    airport_duration_sec: Mapping[str, float]
    service_headway_sec: Mapping[str, float]
    residuals_sec: np.ndarray


def fit_duration_reference(
    labelled_departures: pl.DataFrame,
    *,
    min_airport_support: int = 500,
) -> DurationReference:
    _require(labelled_departures, ("airport", "duration_sec"), label="duration reference")
    clean = labelled_departures.filter(pl.col("duration_sec").is_between(0, 7200))
    if clean.is_empty():
        raise RuntimeError("duration reference has no valid training labels")
    global_duration = float(clean["duration_sec"].median())
    airport_duration: dict[str, float] = {}
    for row in clean.group_by("airport").agg(
        pl.len().alias("n"), pl.col("duration_sec").median().alias("median")
    ).iter_rows(named=True):
        if int(row["n"]) >= min_airport_support:
            airport_duration[str(row["airport"])] = float(row["median"])
    predicted = np.asarray(
        [airport_duration.get(str(x), global_duration) for x in clean["airport"].to_list()]
    )
    residuals = clean["duration_sec"].cast(pl.Float64).to_numpy() - predicted
    service: dict[str, float] = {}
    if "event_epoch" in clean.columns:
        for airport, part in clean.partition_by("airport", as_dict=True).items():
            times = np.sort(part["event_epoch"].cast(pl.Float64).to_numpy())
            gaps = np.diff(times)
            gaps = gaps[(gaps > 0) & (gaps <= 1800)]
            if gaps.size >= 100:
                key = airport[0] if isinstance(airport, tuple) else airport
                service[str(key)] = float(np.quantile(gaps, 0.25))
    return DurationReference(global_duration, airport_duration, service, residuals.astype(np.float32))


def _duration_prediction(airports: np.ndarray, reference: DurationReference) -> np.ndarray:
    return np.asarray(
        [reference.airport_duration_sec.get(str(x), reference.global_duration_sec) for x in airports],
        dtype=np.float64,
    )


def build_virtual_workload_features(events: pl.DataFrame, reference: DurationReference) -> pl.DataFrame:
    """Build an inverse, partially observed virtual runway workload."""

    dep = events.filter(pl.col("phase") == "DEP")
    n = dep.height
    times = dep["event_epoch"].cast(pl.Float64).to_numpy()
    airports = dep["airport"].cast(pl.String).fill_null("__AIRPORT_MISSING__").to_numpy()
    eobt = dep["eobt_time_epoch"].cast(pl.Float64, strict=False).to_numpy()
    schedule = dep["schedule_time_epoch"].cast(pl.Float64, strict=False).to_numpy()
    predicted = _duration_prediction(airports, reference)
    base_ready = np.where(np.isfinite(eobt), eobt, schedule)
    readiness = base_ready + predicted
    backlog = np.zeros(n, dtype=np.float32)
    service_units = np.zeros(n, dtype=np.float32)
    growth = np.zeros(n, dtype=np.float32)
    cleared_age = np.full(n, np.nan, dtype=np.float32)
    coverage = np.isfinite(readiness).astype(np.float32)

    for airport, qidx in _group_indices(airports).items():
        qtime = times[qidx]
        ready = readiness[qidx]
        takeoff = times[qidx]
        good = np.isfinite(ready)
        r = np.sort(ready[good])
        d = np.sort(takeoff[good])
        current = np.searchsorted(r, qtime, side="right") - np.searchsorted(d, qtime, side="left")
        prior_t = qtime - 900
        prior = np.searchsorted(r, prior_t, side="right") - np.searchsorted(d, prior_t, side="right")
        current = np.maximum(0, current - good.astype(np.int64))
        backlog[qidx] = current.astype(np.float32)
        growth[qidx] = (current - np.maximum(0, prior)).astype(np.float32)
        headway = reference.service_headway_sec.get(str(airport), 120.0)
        service_units[qidx] = current.astype(np.float32) * float(headway)
        # Conservative approximation: age since the most recent zero workload
        # in the event sweep, capped at the isolated month boundary.
        order = np.argsort(qtime, kind="mergesort")
        last_zero = np.nan
        for local in order:
            if current[local] <= 0:
                last_zero = qtime[local]
            if np.isfinite(last_zero):
                cleared_age[qidx[local]] = float(qtime[local] - last_zero)

    return dep.select(pl.col("event_id").alias("MVT_ID_mvt")).with_columns(
        pl.Series("s3_virtual_backlog_aircraft", backlog),
        pl.Series("s3_virtual_backlog_service_sec", service_units),
        pl.Series("s3_virtual_backlog_growth_900s", growth),
        pl.Series("s3_time_since_backlog_cleared_sec", cleared_age),
        pl.Series("s3_readiness_coverage", coverage),
        pl.Series("s3_preliminary_duration_sec", predicted.astype(np.float32)),
        pl.Series("s3_readiness_uncertainty_sec", np.full(n, float(np.std(reference.residuals_sec)), dtype=np.float32)),
    )


def _survival_probability(threshold: np.ndarray, residuals: np.ndarray, prediction: float) -> np.ndarray:
    if residuals.size == 0:
        return (threshold < prediction).astype(np.float64)
    sorted_residuals = np.sort(residuals.astype(np.float64))
    cut = threshold - float(prediction)
    return 1.0 - np.searchsorted(sorted_residuals, cut, side="right") / sorted_residuals.size


def build_expected_occupancy_features(
    events: pl.DataFrame,
    reference: DurationReference,
    *,
    horizon_sec: int = 5400,
) -> pl.DataFrame:
    """Sum expected taxi presence from nonfocal uncertain intervals."""

    dep = events.filter(pl.col("phase") == "DEP")
    n = dep.height
    ids = dep["event_id"].to_numpy()
    times = dep["event_epoch"].cast(pl.Float64).to_numpy()
    airports = dep["airport"].cast(pl.String).fill_null("__AIRPORT_MISSING__").to_numpy()
    runways = dep["runway_region"].cast(pl.String).fill_null("__RUNWAY_MISSING__").to_numpy()
    predicted = _duration_prediction(airports, reference)
    sorted_residuals = np.sort(reference.residuals_sec.astype(np.float64))
    values: dict[str, np.ndarray] = {}
    for offset in OCCUPANCY_OFFSETS_SEC:
        values[f"s4_expected_airport_occ_tminus_{offset}s"] = np.zeros(n, dtype=np.float32)
        values[f"s4_point_airport_occ_tminus_{offset}s"] = np.zeros(n, dtype=np.float32)
        values[f"s4_expected_runway_occ_tminus_{offset}s"] = np.zeros(n, dtype=np.float32)
    omitted = np.zeros(n, dtype=np.float32)

    for airport, qidx in _group_indices(airports).items():
        context_t = times[qidx]
        context_p = predicted[qidx]
        context_rwy = runways[qidx]
        context_id = ids[qidx]
        order = np.argsort(context_t, kind="mergesort")
        ct, cp = context_t[order], context_p[order]
        cr, ci = context_rwy[order], context_id[order]
        for out_index in qidx:
            for offset in OCCUPANCY_OFFSETS_SEC:
                u = times[out_index] - offset
                lo = int(np.searchsorted(ct, u, side="right"))
                hi = int(np.searchsorted(ct, u + horizon_sec, side="right"))
                if hi <= lo:
                    continue
                takeoff = ct[lo:hi]
                duration = cp[lo:hi]
                context_ids = ci[lo:hi]
                keep = context_ids != ids[out_index]
                takeoff, duration = takeoff[keep], duration[keep]
                context_runway = cr[lo:hi][keep]
                threshold = takeoff - u
                if sorted_residuals.size:
                    probability = 1.0 - np.searchsorted(
                        sorted_residuals,
                        threshold - duration,
                        side="right",
                    ) / sorted_residuals.size
                else:
                    probability = (threshold < duration).astype(np.float64)
                point = threshold < duration
                values[f"s4_expected_airport_occ_tminus_{offset}s"][out_index] = float(probability.sum())
                values[f"s4_point_airport_occ_tminus_{offset}s"][out_index] = float(point.sum())
                same = context_runway == runways[out_index]
                values[f"s4_expected_runway_occ_tminus_{offset}s"][out_index] = float(probability[same].sum())
            if sorted_residuals.size:
                omitted[out_index] = float(
                    1.0
                    - np.searchsorted(
                        sorted_residuals,
                        horizon_sec - predicted[out_index],
                        side="right",
                    )
                    / sorted_residuals.size
                )
    return dep.select(pl.col("event_id").alias("MVT_ID_mvt")).with_columns(
        *[pl.Series(name, value) for name, value in values.items()],
        pl.Series("s4_omitted_long_duration_probability", omitted),
        pl.Series("s4_preliminary_duration_sec", predicted.astype(np.float32)),
    )


def stand_interval_diagnostics(
    features: pl.DataFrame,
    labels: pl.DataFrame,
) -> dict[str, float | int]:
    joined = features.join(labels, on="MVT_ID_mvt", how="inner", validate="1:1")
    supported = joined.filter(pl.col("s2_chain_confident") == 1)
    if supported.is_empty():
        return {"n": joined.height, "supported": 0, "violation_rate": float("nan")}
    violations = supported.filter(
        (pl.col("duration_sec") < pl.col("s2_bound_lower_sec"))
        | (pl.col("duration_sec") > pl.col("s2_bound_upper_sec"))
    ).height
    return {
        "n": joined.height,
        "supported": supported.height,
        "support_rate": supported.height / max(1, joined.height),
        "violation_rate": violations / supported.height,
        "median_width_sec": float(supported["s2_bound_width_sec"].median()),
    }
