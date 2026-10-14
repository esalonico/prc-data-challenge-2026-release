# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Matched long-taxi / schedule-stamp tail arms (SA-TAIL, 2026-09-26).

Pre-registration: ``progress-review/matched-tail-20260926/preregistration.md``.

* T1 (M-LONG): ``pred' = base + a * p_long * (mu_long - base)`` with a fixed ``a = 0.35``.
* T2 (schedule-stamp mixture outside LIRF, U2-M3 form):
  ``yhat = p * max(sd, 0) + (1 - p) * (0.5 * E_nc + 0.5 * parent)``, ``p = 0`` where a stamp
  cannot satisfy the NM matching window.
* Post-processing shared by every arm: U2-M3 at LIRF, the LOBT window, clip at 0.

The stamp label ``|y - sd| <= 60`` (``sd = takeoff - SCHED``) is read on training rows only.
Nothing here reads DEP BLOCK/TAXITIME of the row being predicted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from prc_challenge.lirf_label_source import WINDOW_SEC, copy_feasible, copy_mixture

LONG_SEC = 2700.0
T1_COEF = 0.35
MU_LONG_CLIP = 30000.0
N_NEGATIVES = 600_000
AIRPORT_CAP = 0.02
D_FOLDS: dict[str, tuple[int, int]] = {"28": (2, 8), "39": (3, 9), "410": (4, 10), "56": (5, 6)}

PROXY_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "boosting": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 255,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 1.0,
    "lambda_l2": 0.0,
    "linear_tree": False,
    "max_bin": 255,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
PROXY_ROUNDS = 600
P_LONG_PARAMS: dict[str, Any] = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=127, min_child_samples=100,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
)
MU_LONG_PARAMS: dict[str, Any] = dict(
    n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
)


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------


def negative_sample(is_long: np.ndarray, n_neg: int = N_NEGATIVES, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """All long rows plus ``n_neg`` random negatives; negatives weighted back to the population rate."""

    is_long = np.asarray(is_long, dtype=bool)
    rng = np.random.default_rng(seed)
    neg = np.where(~is_long)[0]
    n_neg = min(n_neg, neg.size)
    idx = np.concatenate([np.where(is_long)[0], rng.choice(neg, n_neg, replace=False)])
    w = np.where(is_long[idx], 1.0, neg.size / n_neg)
    return idx, w


def t1_correct(base: np.ndarray, p_long: np.ndarray, mu_long: np.ndarray, apply: np.ndarray, a: float = T1_COEF) -> np.ndarray:
    """``base + a * p * (mu - base)`` on ``apply`` rows; other rows unchanged."""

    base = np.asarray(base, dtype=np.float64)
    corr = base + a * np.asarray(p_long, dtype=np.float64) * (np.asarray(mu_long, dtype=np.float64) - base)
    return np.where(np.asarray(apply, dtype=bool), corr, base)


# ---------------------------------------------------------------------------
# T2
# ---------------------------------------------------------------------------


def stamp_feasible(sd: np.ndarray, lead_iobt: np.ndarray, lead_eobt: np.ndarray, lead_lobt: np.ndarray, window: float = WINDOW_SEC) -> np.ndarray:
    """U2's hull feasibility, refined by LOBT: a stamp needs ``|SCHED - LOBT| <= window`` when LOBT exists."""

    sd = np.asarray(sd, dtype=np.float64)
    ll = np.asarray(lead_lobt, dtype=np.float64)
    hull = copy_feasible(sd, lead_iobt, lead_eobt, window)
    with np.errstate(invalid="ignore"):
        lobt_ok = ~np.isfinite(ll) | (np.abs(ll - sd) <= window)
    return hull & lobt_ok


def t2_mixture(p: np.ndarray, sd: np.ndarray, e_nc: np.ndarray, parent: np.ndarray, feasible: np.ndarray, apply: np.ndarray) -> np.ndarray:
    """U2-M3 form on ``apply`` rows: ``p*max(sd,0) + (1-p)*(0.5*E_nc + 0.5*parent)``; others keep ``parent``."""

    parent = np.asarray(parent, dtype=np.float64)
    apply = np.asarray(apply, dtype=bool)
    e = np.where(np.isfinite(e_nc), e_nc, parent)
    mix = copy_mixture(np.nan_to_num(p, nan=0.0), sd, 0.5 * e + 0.5 * parent, np.asarray(feasible, dtype=bool) & apply)
    return np.where(apply, mix, parent)


# ---------------------------------------------------------------------------
# shared post-processing
# ---------------------------------------------------------------------------


def lobt_window(pred: np.ndarray, lead_lobt: np.ndarray, window: float = WINDOW_SEC) -> np.ndarray:
    """``matching_window`` mode ``lobt``: clip into ``[max(0, lead - W), lead + W]`` where LOBT exists."""

    pred = np.asarray(pred, dtype=np.float64)
    ll = np.asarray(lead_lobt, dtype=np.float64)
    ok = np.isfinite(ll)
    lo = np.where(ok, np.maximum(ll - window, 0.0), -np.inf)
    hi = np.where(ok, ll + window, np.inf)
    return np.where(ok, np.clip(pred, lo, hi), pred)


def postprocess(pred: np.ndarray, *, lirf_u2: np.ndarray, u2_p: np.ndarray, u2_expert: np.ndarray, sd: np.ndarray,
                u2_feasible: np.ndarray, lead_lobt: np.ndarray, matched: np.ndarray) -> np.ndarray:
    """U2-M3 on ``lirf_u2`` rows (parent = ``pred``) -> LOBT window on matched rows -> clip at 0."""

    pred = np.asarray(pred, dtype=np.float64)
    lirf_u2 = np.asarray(lirf_u2, dtype=bool)
    mix = copy_mixture(np.nan_to_num(u2_p, nan=0.0), sd, 0.5 * np.nan_to_num(u2_expert, nan=0.0) + 0.5 * pred, np.asarray(u2_feasible, dtype=bool) & lirf_u2)
    out = np.where(lirf_u2, mix, pred)
    matched = np.asarray(matched, dtype=bool)
    out = np.where(matched, lobt_window(out, lead_lobt), out)
    return np.clip(out, 0.0, None)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def day_bootstrap_p05_p95(dse: np.ndarray, day: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """Day-block bootstrap of the summed ΔSE, as ``compare_screen_arms.py`` (days sorted, seed 0)."""

    dse = np.asarray(dse, dtype=np.float64)
    keys, inv = np.unique(np.asarray(day), return_inverse=True)
    days = np.bincount(inv, weights=dse, minlength=keys.size)
    rng = np.random.default_rng(seed)
    boots = np.array([days[rng.integers(0, days.size, days.size)].sum() for _ in range(n_boot)])
    return float(np.percentile(boots, 5)), float(np.percentile(boots, 95))


def gate_report(y: np.ndarray, ctl: np.ndarray, cand: np.ndarray, *, fold: np.ndarray, airport: np.ndarray, day: np.ndarray,
                min_folds: int = 3, airport_cap: float = AIRPORT_CAP) -> dict[str, Any]:
    """Absolute ΔSSE < 0, day-bootstrap p95 < 0, >= ``min_folds`` folds improve, no airport > +cap."""

    y = np.asarray(y, dtype=np.float64)
    se_c = (y - ctl) ** 2
    dse = (y - cand) ** 2 - se_c
    sse = float(se_c.sum())
    dsse = float(dse.sum())
    p05, p95 = day_bootstrap_p05_p95(dse, day)
    folds = {}
    for f in sorted(set(np.asarray(fold).tolist())):
        m = fold == f
        folds[str(f)] = {"n": int(m.sum()), "dsse": float(dse[m].sum()), "rel": float(dse[m].sum() / se_c[m].sum())}
    airports = {}
    for a in sorted(set(np.asarray(airport).tolist())):
        m = airport == a
        airports[str(a)] = {"n": int(m.sum()), "dsse": float(dse[m].sum()), "rel": float(dse[m].sum() / se_c[m].sum())}
    n_imp = sum(v["dsse"] < 0 for v in folds.values())
    worst = max(v["rel"] for v in airports.values())
    order = np.argsort(dse)
    gains = {k: float(dse[order[:k]].sum() / dsse) if dsse != 0 else float("nan") for k in (20, 100)}
    g = {
        "G1_abs_dsse_lt_0": dsse < 0,
        "G2_boot_p95_lt_0": p95 < 0,
        "G3_folds_improve_ge_3": n_imp >= min_folds,
        "G4_no_airport_gt_cap": worst <= airport_cap,
    }
    return {
        "n": int(y.size),
        "rmse_control": float(np.sqrt(sse / y.size)),
        "rmse_candidate": float(np.sqrt((sse + dsse) / y.size)),
        "sse_control": sse,
        "dsse": dsse,
        "dsse_rel": dsse / sse,
        "d002_rel_le_-0.005": dsse / sse <= -0.005,
        "day_bootstrap_p05_p95": [p05, p95],
        "days_improved_frac": float(np.mean(np.bincount(np.unique(day, return_inverse=True)[1], weights=dse) < 0)),
        "folds_improved": f"{n_imp}/{len(folds)}",
        "worst_airport_rel": worst,
        "by_fold": folds,
        "by_airport": airports,
        "top20_share_of_net": gains[20],
        "top100_share_of_net": gains[100],
        "rows_better": int((dse < 0).sum()),
        "rows_worse": int((dse > 0).sum()),
        "gates": g,
        "GO": all(g.values()),
    }


__all__ = [name for name in globals() if not name.startswith("_")]
