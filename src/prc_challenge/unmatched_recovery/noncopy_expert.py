# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Conditional non-copy expert for NM-unmatched departures.

The schedule-copy source class comes from the 60 s BLOCK lattice on *training*
labels. It only selects which training rows an expert is fitted on. It never
enters the feature graph, and held-out source classes are read only for
reporting slices.

Mixture: ``yhat = p * max(takeoff - SCHED, 0) + (1 - p) * E_nc``. With the
optional gate, ``p`` is set to zero where ``takeoff - SCHED`` is below the gate
or missing.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

COPY_CLASS = "schedule_copy"
GATE_SEC = 3600.0
ORDINARY_SEC = 3600.0
EXTREME_SEC = 7200.0
TOP_K = 5
EXPERT_MODES = ("all", "noncopy", "noncopy_large")

# Pre-registered GO gates (progress-review/unmatched-noncopy-20260925/preregistration.json).
GATES = {
    "G1_pooled_sse_ratio_max": 0.95,
    "G2_ordinary_rmse_ratio_max": 1.03,
    "G3_non_lirf_sse_ratio_max": 1.00,
    "G4_leave_top5_sse_ratio_max": 0.98,
    "G5_min_improving_folds": 3,
}


def expert_training_mask(source_class: Sequence[Any], delay: np.ndarray, *, mode: str, gate_sec: float = GATE_SEC) -> np.ndarray:
    """Rows an expert is fitted on.

    ``all`` is U6. ``noncopy`` drops every schedule copy. ``noncopy_large`` drops
    only schedule copies whose takeoff - SCHED is at least ``gate_sec``; a
    missing delay never counts as large.
    """

    source = np.asarray(source_class, dtype=object)
    delay = np.asarray(delay, dtype=np.float64)
    if source.shape[0] != delay.shape[0]:
        raise ValueError("source_class and delay lengths differ")
    is_copy = source == COPY_CLASS
    if mode == "all":
        return np.ones(source.shape[0], dtype=bool)
    if mode == "noncopy":
        return ~is_copy
    if mode == "noncopy_large":
        large = np.isfinite(delay) & (delay >= gate_sec)
        return ~(is_copy & large)
    raise ValueError(f"unknown expert mode {mode}")


def copy_candidate(delay: np.ndarray) -> np.ndarray:
    """Schedule-copy taxi candidate, clipped at zero. Missing stays missing."""

    delay = np.asarray(delay, dtype=np.float64)
    return np.where(np.isfinite(delay), np.maximum(delay, 0.0), np.nan)


def mixture_prediction(p: np.ndarray, delay: np.ndarray, expert: np.ndarray, *, gate_sec: float | None = None) -> np.ndarray:
    """``p * max(delay, 0) + (1 - p) * expert``; missing delay or a closed gate gives the expert."""

    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    delay = np.asarray(delay, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    if not (p.shape == delay.shape == expert.shape):
        raise ValueError("mixture inputs must share one shape")
    cand = copy_candidate(delay)
    use = np.isfinite(cand)
    if gate_sec is not None:
        use = use & (np.nan_to_num(delay, nan=-np.inf) >= gate_sec)
    weight = np.where(use, p, 0.0)
    return np.where(use, weight * np.nan_to_num(cand) + (1.0 - weight) * expert, expert)


def slice_scores(y: np.ndarray, pred: np.ndarray, base: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    n = int(y.size)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "rmse": nan, "base_rmse": nan, "sse": nan, "base_sse": nan, "sse_ratio": nan}
    sse = float(np.sum((y - pred) ** 2))
    base_sse = float(np.sum((y - base) ** 2))
    return {
        "n": n,
        "rmse": float(np.sqrt(sse / n)),
        "base_rmse": float(np.sqrt(base_sse / n)),
        "sse": sse,
        "base_sse": base_sse,
        "sse_ratio": sse / base_sse if base_sse > 0 else float("nan"),
    }


def improvement(y: np.ndarray, pred: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Per-row SSE improvement over the baseline (positive = candidate better)."""

    y = np.asarray(y, dtype=np.float64)
    return (y - np.asarray(base, dtype=np.float64)) ** 2 - (y - np.asarray(pred, dtype=np.float64)) ** 2


def leave_top_k_ratio(y: np.ndarray, pred: np.ndarray, base: np.ndarray, k: int = TOP_K) -> dict[str, Any]:
    """Drop the ``k`` rows with the largest SSE improvement, then recompute the ratio."""

    y = np.asarray(y, dtype=np.float64)
    delta = improvement(y, pred, base)
    keep = np.ones(y.size, dtype=bool)
    order = np.argsort(-delta, kind="stable")
    top = order[: min(k, y.size)]
    keep[top] = False
    scores = slice_scores(y[keep], np.asarray(pred)[keep], np.asarray(base)[keep])
    total = float(delta.sum())
    return {
        "k": int(k),
        "sse_ratio": scores["sse_ratio"],
        "top_rows": [int(i) for i in top],
        "top_delta": [float(delta[i]) for i in top],
        "top_share_of_gain": float(delta[top].sum() / total) if total > 0 else float("nan"),
        "top1_share_of_gain": float(delta[order[0]] / total) if total > 0 and y.size else float("nan"),
    }


def arm_report(
    y: np.ndarray,
    pred: np.ndarray,
    base: np.ndarray,
    *,
    airport: Sequence[Any],
    fold: Sequence[Any],
    unseen_family: np.ndarray | None = None,
    k: int = TOP_K,
) -> dict[str, Any]:
    """Every pre-registered metric for one arm against the baseline."""

    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    airport = np.asarray(airport, dtype=object)
    fold = np.asarray(fold, dtype=object)
    if not np.isfinite(pred).all():
        raise ValueError("non-finite predictions")
    ordinary = y < ORDINARY_SEC
    mid = (y >= ORDINARY_SEC) & (y < EXTREME_SEC)
    tail = y >= EXTREME_SEC
    non_lirf = airport != "LIRF"
    report: dict[str, Any] = {
        "pooled": slice_scores(y, pred, base),
        "ordinary_lt_1h": slice_scores(y[ordinary], pred[ordinary], base[ordinary]),
        "mid_1h_2h": slice_scores(y[mid], pred[mid], base[mid]),
        "tail_ge_2h": slice_scores(y[tail], pred[tail], base[tail]),
        "non_lirf": slice_scores(y[non_lirf], pred[non_lirf], base[non_lirf]),
        "lirf": slice_scores(y[~non_lirf], pred[~non_lirf], base[~non_lirf]),
        "leave_top_k": leave_top_k_ratio(y, pred, base, k),
        "airports": {str(a): slice_scores(y[airport == a], pred[airport == a], base[airport == a]) for a in sorted(set(airport.tolist()))},
        "folds": {str(f): slice_scores(y[fold == f], pred[fold == f], base[fold == f]) for f in sorted(set(fold.tolist()))},
    }
    if unseen_family is not None:
        unseen = np.asarray(unseen_family, dtype=np.float64) >= 0.5
        report["unseen_family"] = slice_scores(y[unseen], pred[unseen], base[unseen])
        report["seen_family"] = slice_scores(y[~unseen], pred[~unseen], base[~unseen])
    return report


def gate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the five pre-registered GO gates. Every gate must pass."""

    pooled = report["pooled"]["sse_ratio"]
    ordinary = report["ordinary_lt_1h"]
    ord_ratio = ordinary["rmse"] / ordinary["base_rmse"] if ordinary["base_rmse"] > 0 else float("nan")
    non_lirf = report["non_lirf"]["sse_ratio"]
    top = report["leave_top_k"]["sse_ratio"]
    improving = sum(1 for row in report["folds"].values() if np.isfinite(row["sse_ratio"]) and row["sse_ratio"] < 1.0)
    gates = {
        "G1_pooled_sse_ratio": {"value": pooled, "threshold": GATES["G1_pooled_sse_ratio_max"], "pass": bool(np.isfinite(pooled) and pooled <= GATES["G1_pooled_sse_ratio_max"])},
        "G2_ordinary_rmse_ratio": {"value": ord_ratio, "threshold": GATES["G2_ordinary_rmse_ratio_max"], "pass": bool(np.isfinite(ord_ratio) and ord_ratio <= GATES["G2_ordinary_rmse_ratio_max"])},
        "G3_non_lirf_sse_ratio": {"value": non_lirf, "threshold": GATES["G3_non_lirf_sse_ratio_max"], "pass": bool(np.isfinite(non_lirf) and non_lirf <= GATES["G3_non_lirf_sse_ratio_max"])},
        "G4_leave_top5_sse_ratio": {"value": top, "threshold": GATES["G4_leave_top5_sse_ratio_max"], "pass": bool(np.isfinite(top) and top <= GATES["G4_leave_top5_sse_ratio_max"])},
        "G5_improving_folds": {"value": improving, "threshold": GATES["G5_min_improving_folds"], "pass": bool(improving >= GATES["G5_min_improving_folds"])},
    }
    return {"gates": gates, "go": all(g["pass"] for g in gates.values()), "failed": [name for name, g in gates.items() if not g["pass"]]}


def choose_confirmation_arm(gated: dict[str, dict[str, Any]], reports: dict[str, dict[str, Any]]) -> str | None:
    """Among arms that pass every gate, the lowest pooled SSE ratio. None when none pass."""

    passing = [name for name, g in gated.items() if g.get("go")]
    if not passing:
        return None
    return min(passing, key=lambda name: reports[name]["pooled"]["sse_ratio"])


# ---------------------------------------------------------------------------
# Ranking overlay helpers
# ---------------------------------------------------------------------------

DEST_ARRIVAL_FEATURES = ("n_dest_arr", "dest_arr_unique", "linked_arr_block_eq_sched")
DELAY_BUCKETS = ((-np.inf, 3600.0, "<1h"), (3600.0, 7200.0, "1-2h"), (7200.0, 14400.0, "2-4h"), (14400.0, np.inf, ">=4h"))


def source_feature_lists(*, drop_dest_arrival: bool = False) -> tuple[list[str], list[str]]:
    """P(copy) features: the 09-21 ``core`` family, optionally without the same-flight destination arrival."""

    from prc_challenge.unmatched_recovery.block_lineage import assert_feature_names, feature_names

    cats, nums = feature_names("core")
    if drop_dest_arrival:
        nums = [name for name in nums if name not in DEST_ARRIVAL_FEATURES]
    assert_feature_names(cats + nums)
    return list(cats), list(nums)


def fit_source_model(frame: Any, cats: Sequence[str], nums: Sequence[str]) -> tuple[Any, Any, Any]:
    """``block_source_model.fit_source_logistic`` with explicit feature lists."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from prc_challenge.unmatched_recovery.block_lineage import CLASSES
    from prc_challenge.unmatched_recovery.block_source_model import _matrix

    model_frame = frame.filter(pl.col("source_class").is_not_null())
    labels = model_frame["source_class"].to_numpy()
    if len([name for name in CLASSES if np.any(labels == name)]) < 2:
        raise RuntimeError("source logistic needs two classes")
    x, encoder = _matrix(model_frame, list(cats), list(nums), None, fit=True)
    scaler = StandardScaler()
    x[:, : len(nums)] = scaler.fit_transform(x[:, : len(nums)])
    model = LogisticRegression(solver="lbfgs", max_iter=200, C=1.0)
    model.fit(x, labels)
    return model, encoder, scaler


def delay_bucket(delay: np.ndarray) -> np.ndarray:
    delay = np.asarray(delay, dtype=np.float64)
    out = np.full(delay.shape[0], "missing", dtype=object)
    for low, high, name in DELAY_BUCKETS:
        out = np.where(np.isfinite(delay) & (delay >= low) & (delay < high), name, out)
    return out


def diff_summary(
    old: np.ndarray,
    new: np.ndarray,
    *,
    airport: Sequence[Any],
    delay: np.ndarray,
    p: np.ndarray,
    prefix: Sequence[Any],
    tol: float = 1e-6,
    top: int = 20,
) -> dict[str, Any]:
    """Feature-only comparison of two unmatched prediction vectors (no labels)."""

    old = np.asarray(old, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    airport = np.asarray(airport, dtype=object)
    prefix = np.asarray(prefix, dtype=object)
    delay = np.asarray(delay, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    change = new - old
    absch = np.abs(change)
    changed = absch > tol
    bucket = delay_bucket(delay)

    def _group(keys: np.ndarray) -> dict[str, Any]:
        out = {}
        for key in sorted(set(keys.tolist()), key=str):
            mask = keys == key
            out[str(key)] = {
                "n": int(mask.sum()),
                "changed": int((changed & mask).sum()),
                "mean_change": float(change[mask].mean()) if mask.any() else float("nan"),
                "mean_abs_change": float(absch[mask].mean()) if mask.any() else float("nan"),
                "max_abs_change": float(absch[mask].max()) if mask.any() else float("nan"),
                "mean_old": float(old[mask].mean()) if mask.any() else float("nan"),
                "mean_new": float(new[mask].mean()) if mask.any() else float("nan"),
            }
        return out

    order = np.argsort(-absch, kind="stable")[:top]
    return {
        "n": int(old.size),
        "changed": int(changed.sum()),
        "mean_change": float(change.mean()),
        "mean_abs_change": float(absch.mean()),
        "max_abs_change": float(absch.max()) if absch.size else float("nan"),
        "mean_old": float(old.mean()),
        "mean_new": float(new.mean()),
        "by_airport": _group(airport),
        "by_delay_bucket": _group(bucket),
        "top_changes": [
            {
                "airport": str(airport[i]),
                "flight_prefix": None if prefix[i] is None else str(prefix[i]),
                "takeoff_minus_sched_sec": float(delay[i]),
                "old": float(old[i]),
                "new": float(new[i]),
                "p_copy": float(p[i]),
            }
            for i in order
        ],
    }


def distribution(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q = np.quantile(v, [0.5, 0.9, 0.95, 0.99])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "max": float(v.max()),
        "share_gt_0_5": float((v > 0.5).mean()),
    }


# ---------------------------------------------------------------------------
# Lineage (P(copy) inputs), month isolation, explicit-feature blocked OOF
# ---------------------------------------------------------------------------

LABEL_COLS = ("TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt")


def build_lineage(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Source lattice, destination-arrival links, prior-day ARR health, family keys.

    Only ``source_class`` and the ``diag_*`` columns read the focal DEP BLOCK; they
    are labels/diagnostics and never features.
    """

    from prc_challenge.unmatched_recovery.block_lineage import (
        attach_arr_prior_day,
        attach_dest_arr_links,
        attach_family_keys,
        attach_source_class,
    )
    from prc_challenge.unmatched_recovery.block_source_model import build_taxi_relation

    labeled = attach_source_class(dep)
    labeled = attach_dest_arr_links(labeled, arr)
    labeled = attach_arr_prior_day(labeled, arr)
    labeled = attach_family_keys(labeled)
    return build_taxi_relation(labeled)


def month_isolated_lineage(dep: pl.DataFrame, arr: pl.DataFrame, months: Sequence[int]) -> pl.DataFrame:
    """Build lineage one month at a time: a month's DEP rows only see that month's ARR rows.

    Day 1 therefore has no prior-day ARR (null, filled 0.0 by the source model),
    in training exactly as at ranking time.
    """

    parts = []
    for month in months:
        dep_m = dep.filter(pl.col("month") == month)
        arr_m = arr.filter(pl.col("month") == month)
        if dep_m.height == 0:
            continue
        parts.append(build_lineage(dep_m, arr_m))
    if not parts:
        raise RuntimeError("no months to build")
    return pl.concat(parts, how="diagonal_relaxed")


def null_fill_counts(frame: pl.DataFrame, nums: Sequence[str]) -> dict[str, Any]:
    """Rows whose numeric source-model input is null and will be filled with 0.0."""

    per = {name: int(frame[name].null_count()) if name in frame.columns else None for name in nums}
    any_null = np.zeros(frame.height, dtype=bool)
    for name in nums:
        if name in frame.columns:
            any_null |= frame[name].is_null().to_numpy()
    arr_prior = [n for n in nums if n.startswith("arr_prior_")]
    arr_null = np.zeros(frame.height, dtype=bool)
    for name in arr_prior:
        if name in frame.columns:
            arr_null |= frame[name].is_null().to_numpy()
    absent = [name for name in nums if name not in frame.columns]
    return {"n": int(frame.height), "per_feature": per, "absent_columns": absent, "rows_any_null": int(any_null.sum()), "rows_arr_prior_null": int(arr_null.sum())}


def blocked_source_oof_explicit(frame: pl.DataFrame, cats: Sequence[str], nums: Sequence[str], *, n0: float, blocks: Sequence[Sequence[int]]) -> pl.DataFrame:
    """``block_source_model.blocked_source_oof`` with explicit feature lists."""

    from prc_challenge.unmatched_recovery.block_lineage import CLASSES
    from prc_challenge.unmatched_recovery.block_source_model import attach_probability_columns, predict_source_proba, with_family_features

    parts: list[pl.DataFrame] = []
    for holdout in blocks:
        fit = frame.filter(~pl.col("month").is_in(list(holdout)))
        hold = frame.filter(pl.col("month").is_in(list(holdout)))
        if fit.height == 0 or hold.height == 0:
            continue
        fit_x = with_family_features(fit, fit, n0=n0, exclude_self=True)
        hold_x = with_family_features(hold, fit, n0=n0, exclude_self=False)
        model, encoder, scaler = fit_source_model(fit_x, cats, nums)
        proba = predict_source_proba(model, encoder, hold_x, list(cats), list(nums), scaler)
        parts.append(attach_probability_columns(hold_x, proba))
    if not parts:
        raise RuntimeError("blocked OOF produced no rows")
    out = pl.concat(parts, how="vertical_relaxed")
    if len(CLASSES) and "p_schedule_copy" not in out.columns:
        raise RuntimeError("schedule-copy probability missing")
    return out


def prediction_agreement(a: np.ndarray, b: np.ndarray, *, threshold: float = 500.0) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    return {
        "n": int(a.size),
        "pearson": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and np.std(a) > 0 and np.std(b) > 0 else float("nan"),
        "mean_abs_diff": float(d.mean()),
        "max_abs_diff": float(d.max()) if d.size else float("nan"),
        f"n_abs_diff_gt_{int(threshold)}": int((d > threshold).sum()),
        "n_abs_diff_gt_0": int((d > 1e-6).sum()),
    }
