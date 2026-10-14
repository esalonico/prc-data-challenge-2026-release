# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Official RMSE and diagnostic metrics. Always aggregate at flight level."""

from __future__ import annotations

import numpy as np
import polars as pl


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty arrays")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(y_pred - y_true))


def score_frame(
    df: pl.DataFrame,
    target_col: str = "TAXITIME_SEC_mvt",
    pred_col: str = "prediction",
    group_cols: tuple[str, ...] = (),
) -> dict[str, float | int]:
    y = df[target_col].to_numpy()
    p = df[pred_col].to_numpy()
    out: dict[str, float | int] = {
        "n": int(df.height),
        "rmse_seconds": rmse(y, p),
        "mae_seconds": mae(y, p),
        "bias_seconds": bias(y, p),
    }
    if group_cols:
        grouped = []
        for keys, part in df.group_by(list(group_cols), maintain_order=True):
            grouped.append(
                {
                    **{c: k for c, k in zip(group_cols, keys if isinstance(keys, tuple) else (keys,))},
                    **score_frame(part, target_col, pred_col),
                }
            )
        out["groups"] = grouped
    return out


def ranking_weighted_rmse(
    scored: pl.DataFrame,
    ranking_weights: pl.DataFrame,
    group_cols: tuple[str, ...] = ("AIRPORT",),
    target_col: str = "TAXITIME_SEC_mvt",
    pred_col: str = "prediction",
    weight_col: str = "n",
) -> float:
    """sqrt(sum_g mse_g * ranking_n_g / ranking_N). Composition diagnostic, not official RMSE."""
    mse = (
        scored.group_by(list(group_cols))
        .agg(((pl.col(target_col) - pl.col(pred_col)) ** 2).mean().alias("mse"))
        .join(ranking_weights, on=list(group_cols), how="inner")
    )
    total = float(mse[weight_col].sum())
    if total <= 0:
        raise ValueError("empty ranking weights")
    return float(((mse["mse"] * mse[weight_col]).sum() / total) ** 0.5)
