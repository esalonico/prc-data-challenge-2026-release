# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Per-airport-scope LightGBM (SA-PERAP, D-081 item 7): one booster per ``AIRPORT`` on the same matrix.

Shared by ``scripts/legs/train_lgbm_matrix.py --per-airport`` (stage A / stage B / fixed rounds),
``scripts/dfold_oof.py --model lgbm_apt`` (D-fold bank) and ``scripts/legs/leg_models.py`` (Nov/Dec).

Every airport gets the same recipe as the global leg (same parameters, same feature block, same
target); only the row scope differs.  The feature matrix is converted to one float32 block once
(categorical levels fixed on all training rows, as the global trainer does), then sliced per airport,
so the per-airport fits see exactly the columns and codes of the global fit.  ``AIRPORT`` stays in the
block; it is constant within an airport and never split on.

Storage: ``<dir>/<AIRPORT>/<model file>`` plus ``<dir>/rounds.json`` (``{airport: best_iteration}``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

AIRPORT_COLUMN = "AIRPORT"


def airports_of(frame: pl.DataFrame) -> np.ndarray:
    values = frame[AIRPORT_COLUMN].cast(pl.String)
    if values.null_count():
        raise RuntimeError("AIRPORT has nulls")
    return values.to_numpy()


def airport_list(codes: np.ndarray) -> list[str]:
    return sorted(set(codes.tolist()))


def dataset(lgb: Any, X: np.ndarray, y: np.ndarray, feature_names: list[str], cats: list[str], params: dict[str, Any] | None = None,
            reference: Any = None) -> Any:
    return lgb.Dataset(X, label=y, feature_name=feature_names, categorical_feature=cats, params=params, reference=reference, free_raw_data=True)


def fit_stage_a(
    lgb: Any,
    params: dict[str, Any],
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_hold: np.ndarray,
    y_hold: np.ndarray,
    feature_names: list[str],
    cats: list[str],
    rounds_cap: int,
    patience: int,
    log_every: int,
    hold_raw_fn: Callable[[np.ndarray], np.ndarray],
    hold_label: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """Early stopping on the holdout rows; returns the booster and a stage-A info dict (training-side numbers only)."""

    t0 = time.perf_counter()
    d_fit = dataset(lgb, X_fit, y_fit, feature_names, cats)
    d_hold = dataset(lgb, X_hold, y_hold, feature_names, cats, reference=d_fit)
    evals: dict[str, Any] = {}
    booster = lgb.train(
        params,
        d_fit,
        num_boost_round=rounds_cap,
        valid_sets=[d_hold],
        valid_names=["holdout"],
        callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(log_every), lgb.record_evaluation(evals)],
    )
    best = int(booster.best_iteration or booster.current_iteration())
    curve = [float(v) for v in evals["holdout"]["rmse"]]
    hold_raw = hold_raw_fn(np.asarray(booster.predict(X_hold, num_iteration=best), dtype=np.float64))
    info = {
        "fit_rows": int(X_fit.shape[0]),
        "holdout_rows": int(X_hold.shape[0]),
        "best_iteration": best,
        "rounds_run": int(booster.current_iteration()),
        "hit_rounds_cap": bool(best >= rounds_cap),
        "holdout_residual_rmse_at_best": float(curve[best - 1]) if best <= len(curve) else None,
        "holdout_taxi_rmse_at_best": float(np.sqrt(np.mean((hold_label - hold_raw) ** 2))),
        "holdout_label_mean": float(hold_label.mean()),
        "curve_every_100": curve[::100],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return booster, info


def fit_fixed(lgb: Any, params: dict[str, Any], X: np.ndarray, y: np.ndarray, feature_names: list[str], cats: list[str], rounds: int,
              log_every: int) -> Any:
    d_all = dataset(lgb, X, y, feature_names, cats, params=params)
    d_all.construct()
    return lgb.train(params, d_all, num_boost_round=int(rounds), callbacks=[lgb.log_evaluation(log_every)])


def load_rounds(path: Path) -> dict[str, int]:
    """``{airport: rounds}`` from a ``rounds.json`` or a per-airport run manifest / stage-A summary."""

    payload = json.loads(Path(path).read_text())
    if "per_airport_rounds" in payload:
        payload = payload["per_airport_rounds"]
    elif "rounds" in payload and isinstance(payload["rounds"], dict):
        payload = payload["rounds"]
    return {str(k): int(v) for k, v in payload.items()}


def predict(boosters: dict[str, Any], X: np.ndarray, codes: np.ndarray, num_threads: int | None = None) -> np.ndarray:
    """Target-scale prediction: each row by its own airport's booster (every airport must have one)."""

    missing = sorted(set(codes.tolist()) - set(boosters))
    if missing:
        raise RuntimeError(f"no per-airport booster for {missing}")
    out = np.full(codes.shape[0], np.nan, dtype=np.float64)
    kw = {"num_threads": num_threads} if num_threads else {}
    for apt, booster in boosters.items():
        idx = np.flatnonzero(codes == apt)
        if idx.size:
            out[idx] = booster.predict(X[idx], **kw)
    if np.isnan(out).any():
        raise RuntimeError("per-airport prediction left NaN rows")
    return out


def load_boosters(model_dir: Path, model_file: str) -> dict[str, Any]:
    import lightgbm as lgb

    rounds = load_rounds(model_dir / "rounds.json")
    return {apt: lgb.Booster(model_file=str(model_dir / apt / model_file)) for apt in rounds}


def model_hashes(model_dir: Path, model_file: str, sha: Callable[[Path], str]) -> dict[str, str]:
    rounds = load_rounds(model_dir / "rounds.json")
    return {apt: sha(model_dir / apt / model_file) for apt in rounds}


__all__ = [name for name in globals() if not name.startswith("_")]
