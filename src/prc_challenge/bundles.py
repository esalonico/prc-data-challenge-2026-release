# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Shared E08b / E34 feature bundles and the E37 internal tuning split.

V1/V2/V3 val months and locked V4 months are excluded from hyperparameter
selection. The booster train filter matches E33: drop unmatched and E18.
"""

from __future__ import annotations

import polars as pl

from prc_challenge.features import e03_feature_lists, lirf_extraday_as_taxi_gate
from prc_challenge.traffic import DELAY_BUNCH_NUMS

WX_NUM = [
    "tmpc",
    "dwpc",
    "sknt",
    "gust",
    "vsby",
    "p01i",
    "metar_age_sec",
    "metar_has_wx",
    "metar_frozen",
    "metar_fog",
    "metar_ts",
    "metar_precip",
    "metar_low_vis",
]
WX_CAT = ["skyc1"]

# V1 train months are 2–6, 8–10. Hold month 9 as the only internal tuning month.
# Excludes V1 val (1, 7), V3 val (10), and locked V4 (11, 12).
E37_FIT_MONTHS = (2, 3, 4, 5, 6, 8)
E37_TUNE_MONTHS = (9,)
CACHE_MONTHS = tuple(range(1, 11))

CATBOOST_BASELINE = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 400,
    "depth": 6,
    "learning_rate": 0.08,
    "random_seed": 42,
    "thread_count": 4,
    "verbose": False,
    "allow_writing_files": False,
}


def optuna_winner_params(payload: dict, thread_count: int | None = None) -> dict:
    """CatBoost params for official-view scoring of an internal-split winner.

    Locks tree count to the September `best_iteration` (0-based → +1 trees).
    Does not enable early stopping: V1/V2/V3 must not be the eval set.
    """
    raw = payload.get("params") or payload.get("best_params") or payload
    params = dict(CATBOOST_BASELINE)
    for key in (
        "iterations",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "random_strength",
        "border_count",
        "rsm",
        "bootstrap_type",
        "bagging_temperature",
        "subsample",
    ):
        if key in raw:
            params[key] = raw[key]
    best_iter = payload.get("best_iteration")
    if best_iter is not None:
        params["iterations"] = int(best_iter) + 1
    if thread_count is not None:
        params["thread_count"] = thread_count
    if params.get("bootstrap_type") == "Bayesian":
        params.pop("subsample", None)
    else:
        params.pop("bagging_temperature", None)
    params["use_best_model"] = False
    params["verbose"] = False
    params["allow_writing_files"] = False
    return params


def e08b_feature_lists() -> tuple[list[str], list[str]]:
    cats, nums = e03_feature_lists(include_planning=True)
    cats = cats + list(WX_CAT)
    nums = nums + list(WX_NUM)
    return cats, nums


def e34_feature_lists() -> tuple[list[str], list[str]]:
    cats, nums = e08b_feature_lists()
    nums = nums + list(DELAY_BUNCH_NUMS)
    return cats, nums


def matched_booster_frame(df: pl.DataFrame) -> pl.DataFrame:
    """E33/E34 train cut: NM-matched rows, drop LIRF extra-day gate rows."""
    return df.filter((pl.col("nm_unmatched") == 0) & (~lirf_extraday_as_taxi_gate()))


def sample_priority_rows(
    df: pl.DataFrame,
    *,
    take_all: tuple[str, ...] = (),
    fill: tuple[tuple[str, int], ...] = (),
    cap: int | None = None,
    seed: int = 0,
    stratum_col: str = "stratum",
    id_col: str = "MVT_ID_mvt",
) -> pl.DataFrame:
    """Keep every `take_all` row, then fill remaining quota without replacement.

    `cap` truncates fill only. Rare strata are never downsampled.
    """
    parts = [df.filter(pl.col(stratum_col) == label) for label in take_all]
    parts = [part for part in parts if part.height]
    taken = pl.concat(parts).unique(id_col, keep="first") if parts else df.clear()
    for label, want in fill:
        if want <= 0:
            continue
        if cap is not None:
            room = cap - taken.height
            if room <= 0:
                break
            want = min(want, room)
        part = df.filter(pl.col(stratum_col) == label)
        if taken.height:
            part = part.join(taken.select(id_col), on=id_col, how="anti")
        if part.height == 0:
            continue
        if part.height > want:
            part = part.sample(n=want, seed=seed, shuffle=True)
        taken = pl.concat([taken, part]) if taken.height else part
    return taken
