# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Fold-safe latent airport state from completed native arrival movements.

Only completed ARR taxi-in observations enter this module.  State is reset at
each isolated month boundary and queried strictly before a departure epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal

import numpy as np
import polars as pl


FILTERED_STATE_FEATURES = (
    "arrf_state_mean_sec",
    "arrf_state_uncertainty_sec",
    "arrf_state_trend_sec_per_hour",
    "arrf_observation_age_sec",
    "arrf_effective_support",
    "arrf_runway_deviation_sec",
    "arrf_runway_support",
    "arrf_stand_deviation_sec",
    "arrf_stand_support",
    "arrf_localized_state_sec",
    "arrf_cold_start",
    "arrf_measurement_quality",
    "arrf_integrated_est_taxi_sec",
)

KALMAN_STATE_FEATURES = (
    "arrk_state_mean_sec",
    "arrk_state_uncertainty_sec",
    "arrk_state_trend_sec_per_hour",
    "arrk_observation_age_sec",
    "arrk_effective_support",
    "arrk_runway_deviation_sec",
    "arrk_runway_uncertainty_sec",
    "arrk_stand_deviation_sec",
    "arrk_stand_uncertainty_sec",
    "arrk_localized_state_sec",
    "arrk_cold_start",
    "arrk_measurement_quality",
    "arrk_integrated_est_taxi_sec",
)


@dataclass(frozen=True)
class FilterConfig:
    half_life_sec: float = 3600.0
    local_half_life_sec: float = 2700.0
    reference_shrinkage: float = 50.0
    local_shrinkage: float = 8.0
    max_support: float = 40.0


@dataclass(frozen=True)
class KalmanConfig:
    level_half_life_sec: float = 7200.0
    trend_half_life_sec: float = 2700.0
    process_noise_sec2_per_hour: float = 900.0
    trend_noise_sec2_per_hour: float = 36.0
    observation_noise_sec: float = 180.0
    local_half_life_sec: float = 2700.0
    local_process_noise_sec2_per_hour: float = 400.0
    local_observation_noise_sec: float = 300.0
    reference_shrinkage: float = 50.0


REFERENCE_LEVELS = (
    ("native_arr_airport", "native_arr_runway_region", "native_arr_stand_region"),
    ("native_arr_airport", "native_arr_runway_region"),
    ("native_arr_airport",),
)


def fit_arrival_reference(
    arrivals: pl.DataFrame, *, fit_months: tuple[int, ...]
) -> dict[str, object]:
    """Fit a bounded hierarchical taxi-in reference on training months only."""

    fit = arrivals.filter(pl.col("native_arr_month").is_in(fit_months))
    if fit.is_empty():
        raise ValueError("arrival reference needs at least one training row")
    global_mean = float(fit["native_arr_taxi_in_sec"].mean())
    tables: list[tuple[tuple[str, ...], pl.DataFrame, float]] = []
    for keys, shrinkage in zip(REFERENCE_LEVELS, (30.0, 50.0, 100.0), strict=True):
        table = (
            fit.group_by(list(keys))
            .agg(
                pl.col("native_arr_taxi_in_sec").mean().alias("_mean"),
                pl.len().cast(pl.Float64).alias("_support"),
            )
            .with_columns(
                (
                    (pl.col("_mean") * pl.col("_support") + global_mean * shrinkage)
                    / (pl.col("_support") + shrinkage)
                ).alias("_expected")
            )
        )
        tables.append((keys, table, shrinkage))
    return {"global_mean": global_mean, "tables": tables, "fit_months": fit_months}


def attach_arrival_measurements(
    arrivals: pl.DataFrame, reference: dict[str, object]
) -> pl.DataFrame:
    """Attach signed excess, reference support, and quality to ARR events."""

    out = arrivals
    expected_cols: list[str] = []
    support_cols: list[str] = []
    for index, (keys, table, _) in enumerate(reference["tables"]):
        expected = f"_ref_expected_{index}"
        support = f"_ref_support_{index}"
        expected_cols.append(expected)
        support_cols.append(support)
        out = out.join(
            table.select(
                *keys,
                pl.col("_expected").alias(expected),
                pl.col("_support").alias(support),
            ),
            on=list(keys),
            how="left",
        )
    return (
        out.with_columns(
            pl.coalesce(*[pl.col(c) for c in expected_cols], pl.lit(reference["global_mean"]))
            .cast(pl.Float64)
            .alias("arrival_reference_sec"),
            pl.coalesce(*[pl.col(c) for c in support_cols], pl.lit(0.0))
            .cast(pl.Float64)
            .alias("arrival_reference_support"),
        )
        .with_columns(
            (pl.col("native_arr_taxi_in_sec") - pl.col("arrival_reference_sec")).alias(
                "arrival_excess_sec"
            ),
            pl.when(pl.col("arrival_reference_support") >= 30)
            .then(pl.lit(2))
            .when(pl.col("arrival_reference_support") > 0)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("arrival_measurement_quality"),
        )
        .drop(*expected_cols, *support_cols)
    )


def _decay(dt: float, half_life: float) -> float:
    return exp(-max(dt, 0.0) * np.log(2.0) / half_life)


def _filter_predict(state: dict[str, float] | None, when: int, half_life: float) -> dict[str, float]:
    if state is None:
        return {"mean": 0.0, "var": 0.0, "support": 0.0, "trend": 0.0, "last": -1.0}
    decay = _decay(when - state["last"], half_life)
    return {
        "mean": decay * state["mean"],
        "var": decay * state["var"],
        "support": decay * state["support"],
        "trend": decay * state["trend"],
        "last": state["last"],
    }


def _filter_update(
    state: dict[str, float] | None,
    *,
    when: int,
    observation: float,
    reliability: float,
    config: FilterConfig,
    local: bool = False,
) -> dict[str, float]:
    half_life = config.local_half_life_sec if local else config.half_life_sec
    prior = _filter_predict(state, when, half_life)
    weight = max(reliability, 0.05)
    denominator = prior["support"] + weight
    mean = (prior["support"] * prior["mean"] + weight * observation) / denominator
    innovation = observation - prior["mean"]
    var = (
        prior["support"] * prior["var"]
        + weight * innovation * (observation - mean)
    ) / denominator
    return {
        "mean": mean,
        "var": max(var, 0.0),
        "support": min(denominator, config.max_support),
        "trend": mean - prior["mean"],
        "last": float(when),
    }


def _kalman_predict(state: dict[str, object] | None, when: int, config: KalmanConfig) -> dict[str, object]:
    if state is None:
        return {
            "x": np.zeros(2, dtype=float),
            "p": np.diag([40000.0, 3600.0]),
            "last": -1.0,
            "support": 0.0,
        }
    dt = max(float(when) - float(state["last"]), 0.0)
    hours = dt / 3600.0
    level_phi = _decay(dt, config.level_half_life_sec)
    trend_phi = _decay(dt, config.trend_half_life_sec)
    transition = np.array([[level_phi, hours * trend_phi], [0.0, trend_phi]])
    q = np.diag(
        [
            config.process_noise_sec2_per_hour * max(hours, 1.0 / 60.0),
            config.trend_noise_sec2_per_hour * max(hours, 1.0 / 60.0),
        ]
    )
    return {
        "x": transition @ state["x"],
        "p": transition @ state["p"] @ transition.T + q,
        "last": state["last"],
        "support": _decay(dt, config.level_half_life_sec) * float(state["support"]),
    }


def _kalman_update(
    state: dict[str, object] | None,
    *,
    when: int,
    observation: float,
    reliability: float,
    config: KalmanConfig,
) -> dict[str, object]:
    prior = _kalman_predict(state, when, config)
    h = np.array([1.0, 0.0])
    r = config.observation_noise_sec**2 / max(reliability, 0.05)
    innovation = observation - float(h @ prior["x"])
    gain = prior["p"] @ h / (float(h @ prior["p"] @ h) + r)
    x = prior["x"] + gain * innovation
    p = (np.eye(2) - np.outer(gain, h)) @ prior["p"]
    return {
        "x": x,
        "p": 0.5 * (p + p.T),
        "last": float(when),
        "support": min(float(prior["support"]) + reliability, 100.0),
    }


def _local_kalman_predict(
    state: dict[str, float] | None, when: int, config: KalmanConfig
) -> dict[str, float]:
    if state is None:
        return {"mean": 0.0, "var": 40000.0, "support": 0.0, "last": -1.0}
    dt = max(float(when) - state["last"], 0.0)
    phi = _decay(dt, config.local_half_life_sec)
    return {
        "mean": phi * state["mean"],
        "var": phi * phi * state["var"]
        + config.local_process_noise_sec2_per_hour * max(dt / 3600.0, 1.0 / 60.0),
        "support": phi * state["support"],
        "last": state["last"],
    }


def _local_kalman_update(
    state: dict[str, float] | None,
    *,
    when: int,
    observation: float,
    reliability: float,
    config: KalmanConfig,
) -> dict[str, float]:
    prior = _local_kalman_predict(state, when, config)
    r = config.local_observation_noise_sec**2 / max(reliability, 0.05)
    gain = prior["var"] / (prior["var"] + r)
    return {
        "mean": prior["mean"] + gain * (observation - prior["mean"]),
        "var": max((1.0 - gain) * prior["var"], 0.0),
        "support": min(prior["support"] + reliability, 100.0),
        "last": float(when),
    }


def _month_events(
    departures: pl.DataFrame,
    measurements: pl.DataFrame,
    *,
    month: int,
    completion_shift_sec: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    deps = (
        departures.filter(pl.col("month") == month)
        .select(
            "MVT_ID_mvt",
            "native_airport",
            "native_dep_epoch",
            "native_current_runway_region",
            "native_current_stand_region",
        )
        .sort("native_dep_epoch")
        .to_dicts()
    )
    arrs = (
        measurements.filter(pl.col("native_arr_month") == month)
        .select(
            "native_arr_airport",
            "native_arr_inblock_epoch",
            "native_arr_runway_region",
            "native_arr_stand_region",
            "arrival_excess_sec",
            "arrival_reference_support",
            "arrival_measurement_quality",
        )
        .with_columns(
            (pl.col("native_arr_inblock_epoch") + completion_shift_sec).alias(
                "native_arr_inblock_epoch"
            )
        )
        .sort("native_arr_inblock_epoch")
        .to_dicts()
    )
    return deps, arrs


def build_latent_state_sidecar(
    departures: pl.DataFrame,
    arrivals: pl.DataFrame,
    *,
    fit_months: tuple[int, ...],
    estimator: Literal["filtered", "kalman"],
    filter_config: FilterConfig | None = None,
    kalman_config: KalmanConfig | None = None,
    completion_shift_sec: int = 0,
    measurements: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build one fold-safe state row per departure for the months in departures."""

    if measurements is None:
        reference = fit_arrival_reference(arrivals, fit_months=fit_months)
        measured = attach_arrival_measurements(arrivals, reference)
    else:
        measured = measurements
    fcfg = filter_config or FilterConfig()
    kcfg = kalman_config or KalmanConfig()
    output: list[dict[str, object]] = []
    for month in sorted(int(x) for x in departures["month"].unique().to_list()):
        deps, events = _month_events(
            departures, measured, month=month, completion_shift_sec=completion_shift_sec
        )
        global_states: dict[str, object] = {}
        runway_states: dict[tuple[str, str], object] = {}
        stand_states: dict[tuple[str, str], object] = {}
        event_index = 0
        quality_sum: dict[str, float] = {}
        quality_weight: dict[str, float] = {}
        for dep in deps:
            query_time = int(dep["native_dep_epoch"])
            while event_index < len(events) and int(events[event_index]["native_arr_inblock_epoch"]) < query_time:
                event = events[event_index]
                airport = str(event["native_arr_airport"])
                when = int(event["native_arr_inblock_epoch"])
                observation = float(event["arrival_excess_sec"])
                support = float(event["arrival_reference_support"])
                reliability = max(0.10, min(1.0, support / (support + fcfg.reference_shrinkage)))
                quality_sum[airport] = quality_sum.get(airport, 0.0) + reliability * float(
                    event["arrival_measurement_quality"]
                )
                quality_weight[airport] = quality_weight.get(airport, 0.0) + reliability
                if estimator == "filtered":
                    prior = _filter_predict(global_states.get(airport), when, fcfg.half_life_sec)
                    deviation = observation - prior["mean"]
                    global_states[airport] = _filter_update(
                        global_states.get(airport),
                        when=when,
                        observation=observation,
                        reliability=reliability,
                        config=fcfg,
                    )
                    updater = lambda state: _filter_update(
                        state,
                        when=when,
                        observation=deviation,
                        reliability=reliability,
                        config=fcfg,
                        local=True,
                    )
                else:
                    prior = _kalman_predict(global_states.get(airport), when, kcfg)
                    deviation = observation - float(prior["x"][0])
                    global_states[airport] = _kalman_update(
                        global_states.get(airport),
                        when=when,
                        observation=observation,
                        reliability=reliability,
                        config=kcfg,
                    )
                    updater = lambda state: _local_kalman_update(
                        state,
                        when=when,
                        observation=deviation,
                        reliability=reliability,
                        config=kcfg,
                    )
                runway = event["native_arr_runway_region"]
                stand = event["native_arr_stand_region"]
                if runway is not None:
                    key = (airport, str(runway))
                    runway_states[key] = updater(runway_states.get(key))
                if stand is not None:
                    key = (airport, str(stand))
                    stand_states[key] = updater(stand_states.get(key))
                event_index += 1

            airport = str(dep["native_airport"])
            runway_key = (airport, str(dep["native_current_runway_region"]))
            stand_key = (airport, str(dep["native_current_stand_region"]))
            if estimator == "filtered":
                state = _filter_predict(global_states.get(airport), query_time, fcfg.half_life_sec)
                runway = _filter_predict(runway_states.get(runway_key), query_time, fcfg.local_half_life_sec)
                stand = _filter_predict(stand_states.get(stand_key), query_time, fcfg.local_half_life_sec)
                runway_shrink = runway["support"] / (runway["support"] + fcfg.local_shrinkage)
                stand_shrink = stand["support"] / (stand["support"] + fcfg.local_shrinkage)
                prefix = "arrf"
                row = {
                    "MVT_ID_mvt": dep["MVT_ID_mvt"],
                    f"{prefix}_state_mean_sec": state["mean"],
                    f"{prefix}_state_uncertainty_sec": sqrt(state["var"] / max(state["support"], 1.0)),
                    f"{prefix}_state_trend_sec_per_hour": state["trend"] * 2.0,
                    f"{prefix}_observation_age_sec": None if state["last"] < 0 else query_time - state["last"],
                    f"{prefix}_effective_support": state["support"],
                    f"{prefix}_runway_deviation_sec": runway_shrink * runway["mean"],
                    f"{prefix}_runway_support": runway["support"],
                    f"{prefix}_stand_deviation_sec": stand_shrink * stand["mean"],
                    f"{prefix}_stand_support": stand["support"],
                    f"{prefix}_localized_state_sec": state["mean"] + runway_shrink * runway["mean"] + stand_shrink * stand["mean"],
                    f"{prefix}_cold_start": int(state["last"] < 0),
                }
            else:
                state = _kalman_predict(global_states.get(airport), query_time, kcfg)
                runway = _local_kalman_predict(runway_states.get(runway_key), query_time, kcfg)
                stand = _local_kalman_predict(stand_states.get(stand_key), query_time, kcfg)
                prefix = "arrk"
                row = {
                    "MVT_ID_mvt": dep["MVT_ID_mvt"],
                    f"{prefix}_state_mean_sec": float(state["x"][0]),
                    f"{prefix}_state_uncertainty_sec": sqrt(max(float(state["p"][0, 0]), 0.0)),
                    f"{prefix}_state_trend_sec_per_hour": float(state["x"][1]),
                    f"{prefix}_observation_age_sec": None if state["last"] < 0 else query_time - float(state["last"]),
                    f"{prefix}_effective_support": float(state["support"]),
                    f"{prefix}_runway_deviation_sec": runway["mean"],
                    f"{prefix}_runway_uncertainty_sec": sqrt(max(runway["var"], 0.0)),
                    f"{prefix}_stand_deviation_sec": stand["mean"],
                    f"{prefix}_stand_uncertainty_sec": sqrt(max(stand["var"], 0.0)),
                    f"{prefix}_localized_state_sec": float(state["x"][0]) + runway["mean"] + stand["mean"],
                    f"{prefix}_cold_start": int(state["last"] < 0),
                }
            row[f"{prefix}_measurement_quality"] = (
                quality_sum.get(airport, 0.0) / quality_weight[airport]
                if quality_weight.get(airport, 0.0) > 0
                else 0.0
            )
            output.append(row)
    out = pl.DataFrame(output)
    if out.height != departures.height or out["MVT_ID_mvt"].n_unique() != out.height:
        raise RuntimeError("latent state sidecar lost or duplicated departures")
    return out


def add_interval_projection(frame: pl.DataFrame, *, estimator: Literal["filtered", "kalman"]) -> pl.DataFrame:
    """Project state over a training-safe estimated taxi interval."""

    prefix = "arrf" if estimator == "filtered" else "arrk"
    return frame.with_columns(
        (
            pl.col(f"{prefix}_state_mean_sec")
            + 0.5
            * pl.col(f"{prefix}_state_trend_sec_per_hour")
            * pl.col("unimpeded_p10").clip(0, 3600)
            / 3600.0
        ).alias(f"{prefix}_integrated_est_taxi_sec")
    )
