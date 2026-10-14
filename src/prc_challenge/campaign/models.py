# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Shared CatBoost / LightGBM / XGB helpers for the campaign. No V4. No AOBT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from prc_challenge.bundles import CATBOOST_BASELINE, matched_booster_frame
from prc_challenge.campaign.runtime import CampaignConfig
from prc_challenge.campaign.scoring import assert_no_forbidden_features


def booster_params(cfg: CampaignConfig, **overrides: Any) -> dict[str, Any]:
    params = dict(CATBOOST_BASELINE)
    params.update(
        {
            "iterations": cfg.catboost_iterations,
            "thread_count": cfg.thread_count,
            "random_seed": cfg.seed,
            "allow_writing_files": False,
            "verbose": False,
        }
    )
    params.update(overrides)
    return params


def to_pool(df: pl.DataFrame, cats: list[str], nums: list[str], label: str | None = None) -> Pool:
    assert_no_forbidden_features(cats + nums)
    exprs: list[pl.Expr] = []
    for c in cats:
        exprs.append(pl.col(c).fill_null("__UNK__").cast(pl.String))
    for c in nums:
        exprs.append(pl.col(c).cast(pl.Float32))
    frame = df.select(exprs) if exprs else df.select(cats + nums)
    kwargs: dict[str, Any] = {"data": frame.to_pandas(), "cat_features": cats}
    if label is not None:
        kwargs["label"] = df[label].to_numpy()
    return Pool(**kwargs)


def numeric_matrix(df: pl.DataFrame, nums: list[str]) -> np.ndarray:
    """Float32 numeric block for LightGBM/XGBoost without a pandas copy of the e08b table."""
    if not nums:
        return np.zeros((df.height, 0), dtype=np.float32)
    return df.select([pl.col(c).fill_null(0).cast(pl.Float32) for c in nums]).to_numpy()


def fit_regressor(
    cfg: CampaignConfig,
    train: pl.DataFrame,
    cats: list[str],
    nums: list[str],
    label: str = "TAXITIME_SEC_mvt",
    **overrides: Any,
) -> CatBoostRegressor:
    model = CatBoostRegressor(**booster_params(cfg, **overrides))
    model.fit(to_pool(train, cats, nums, label))
    return model


def predict_regressor(model: CatBoostRegressor, df: pl.DataFrame, cats: list[str], nums: list[str]) -> np.ndarray:
    pred = predict_unclipped(model, df, cats, nums)
    return np.clip(pred, 0, None)


def predict_unclipped(model: CatBoostRegressor, df: pl.DataFrame, cats: list[str], nums: list[str]) -> np.ndarray:
    """Predict without clipping. Use for residual/delta targets that may be negative."""
    return np.asarray(model.predict(to_pool(df, cats, nums)), dtype=np.float64)


def predict_classifier(model: CatBoostClassifier, df: pl.DataFrame, cats: list[str], nums: list[str]) -> np.ndarray:
    proba = np.asarray(model.predict_proba(to_pool(df, cats, nums)), dtype=np.float64)
    if proba.ndim == 2 and proba.shape[1] > 1:
        return proba[:, 1]
    return proba.reshape(-1)


def month_blocked_oof(
    cfg: CampaignConfig,
    train: pl.DataFrame,
    cats: list[str],
    nums: list[str],
    label: str = "TAXITIME_SEC_mvt",
    shard_dir: Path | None = None,
    shard_prefix: str = "oof_month",
    **overrides: Any,
) -> pl.DataFrame:
    """Leave-one-month-out OOF on the train frame. Does not use val months."""
    from prc_challenge.campaign.runtime import persist_repo_file

    months = train["month"].unique().sort().to_list()
    parts: list[pl.DataFrame] = []
    for month in months:
        shard = None
        if shard_dir is not None:
            shard = Path(shard_dir) / f"{shard_prefix}_{int(month):02d}.parquet"
        if shard is not None and shard.exists() and not cfg.force_rerun:
            print(f"OOF resume skip month={month}", flush=True)
            parts.append(pl.read_parquet(shard))
            continue
        fit = train.filter(pl.col("month") != month)
        hold = train.filter(pl.col("month") == month)
        if fit.height == 0 or hold.height == 0:
            continue
        model = fit_regressor(cfg, fit, cats, nums, label, **overrides)
        pred = predict_regressor(model, hold, cats, nums)
        part = hold.select(["MVT_ID_mvt", "month", "AIRPORT", label]).with_columns(
            pl.Series("oof_prediction", pred)
        )
        if shard is not None:
            shard.parent.mkdir(parents=True, exist_ok=True)
            part.write_parquet(shard)
            persist_repo_file(cfg, shard)
        parts.append(part)
    if not parts:
        raise RuntimeError("empty OOF")
    oof = pl.concat(parts)
    from prc_challenge.campaign.scoring import oof_is_held_out

    oof_is_held_out(train["MVT_ID_mvt"], oof["MVT_ID_mvt"])
    return oof


def fit_classifier(
    cfg: CampaignConfig,
    train: pl.DataFrame,
    cats: list[str],
    nums: list[str],
    label: str,
    **overrides: Any,
) -> CatBoostClassifier:
    params = booster_params(cfg, **overrides)
    params["loss_function"] = "Logloss"
    params["eval_metric"] = "Logloss"
    model = CatBoostClassifier(**params)
    model.fit(to_pool(train, cats, nums, label))
    return model


def _t31_oof_overrides(cfg: CampaignConfig) -> dict[str, Any]:
    """Depth/lr/bootstrap of t31; keep overnight OOF tractable (not 2210 trees)."""
    import json

    from prc_challenge.bundles import optuna_winner_params
    from prc_challenge.campaign.data_check import locate

    path = locate("artifacts/experiments/E37-optuna/shortlist_t31.json", cfg)
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    params = optuna_winner_params(payload, thread_count=cfg.thread_count)
    params.pop("iterations", None)
    params.pop("use_best_model", None)
    params.pop("od_type", None)
    params.pop("od_wait", None)
    return params


def load_or_build_base_residual(
    cfg: CampaignConfig, extra_cols: list[str] | tuple[str, ...] | None = None
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, str]:
    """Month-blocked train OOF + official BASE val preds (t31/t6), aligned by MVT_ID.

    When the OOF parquet already exists (CPU campaign wrote it), load only skinny
    cache columns plus `extra_cols` so GPU-01..06 do not hold the full e08b dump.
    """
    from prc_challenge.campaign.data import feature_lists, load_base_preds, load_cache, load_cache_columns, matched_train_val

    cats, nums = feature_lists("e08b")
    view = "V1" if "V1" in cfg.views else cfg.views[0]
    suffix = "_smoke" if cfg.smoke else ""
    oof_path = cfg.cache_root / f"base_oof_{view}_official{suffix}.parquet"
    reuse_oof = oof_path.exists() and not (cfg.force_rerun and cfg.allow_rebuild_cpu_caches)
    if reuse_oof:
        df = load_cache_columns(cfg, extra=extra_cols)
    elif not cfg.allow_rebuild_cpu_caches:
        raise FileNotFoundError(
            f"need {oof_path.as_posix()} from the CPU campaign; GPU will not refit BASE OOF"
        )
    else:
        df = load_cache(cfg)
    train, val = matched_train_val(df, view)
    official = load_base_preds(cfg, view).select(
        ["MVT_ID_mvt", pl.col("prediction").alias("base_prediction")]
    )
    n_val = val.height
    val = val.join(official, on="MVT_ID_mvt", how="inner")
    if val.height != n_val or val["base_prediction"].null_count():
        raise RuntimeError(
            f"official BASE val IDs do not cover the split: val={n_val} joined={val.height}"
        )
    val_base = val["base_prediction"].to_numpy()
    val = val.with_columns(pl.col("base_prediction").alias("oof_prediction"))
    cfg.cache_root.mkdir(parents=True, exist_ok=True)
    if reuse_oof:
        oof = pl.read_parquet(oof_path)
    else:
        overrides = _t31_oof_overrides(cfg)
        oof = month_blocked_oof(
            cfg,
            train,
            cats,
            nums,
            shard_dir=cfg.cache_root,
            shard_prefix=f"base_oof_{view}_official{suffix}_month",
            **overrides,
        )
        oof.write_parquet(oof_path)
        from prc_challenge.campaign.runtime import persist_repo_file

        persist_repo_file(cfg, oof_path)
    train = train.join(oof.select(["MVT_ID_mvt", "oof_prediction"]), on="MVT_ID_mvt")
    train = train.with_columns((pl.col("TAXITIME_SEC_mvt") - pl.col("oof_prediction")).alias("residual"))
    return train, val, val_base, view
