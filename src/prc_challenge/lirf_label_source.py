# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""LIRF label-source experiments (SA-LIRF, 2026-09-25).

U2: matched LIRF schedule-copy mixture. The copy label ``|y - (takeoff - SCHED)| <= 60`` is
read only on *training* rows: it selects the rows the non-copy expert is fitted on and is
the target of the nested copy classifier. It never enters a feature.

    yhat = p * max(sd, 0) + (1 - p) * E_nc,   sd = takeoff - SCHED

``p`` is forced to zero when a copy is impossible under the NM matching window: a matched
BLOCK lies within 3606 s of IOBT or of EOBT_1, so BLOCK == SCHED needs SCHED within
3606 s of one of them. The hull projection (``matching_window``) is applied afterwards.

U3: gate-delay expert for unmatched LIRF rows. ``G = BLOCK - SCHED = sd - y`` is learned
on all LIRF departures; the served taxi is ``T = max(sd - G_hat, 0)``, blended with the
A3b unmatched prediction at a fixed weight.

Pre-registration: ``progress-review/lirf-label-source-20260925/preregistration.json``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

WINDOW_SEC = 3606.0
COPY_TOL_SEC = 60.0
RATE_N0 = 20.0
U2_GATE_SEC = 1800.0
U3_LAMBDA = 0.25
U3_LAMBDA_DIAGNOSTIC = 0.5
ORDINARY_SEC = 3600.0
TOP_K = 5
SD_BUCKET_EDGES = (
    (-np.inf, 0.0, "neg"),
    (0.0, 900.0, "0-15m"),
    (900.0, 1800.0, "15-30m"),
    (1800.0, 3600.0, "30-60m"),
    (3600.0, 7200.0, "1-2h"),
    (7200.0, 14400.0, "2-4h"),
    (14400.0, np.inf, ">=4h"),
)

STAND_NUMS = ("stand_prev_arr_minus_sched_sec", "takeoff_minus_stand_prev_arr_sec", "stand_prev_arr_delay_sec")

U2_GATES = {
    "G1_lirf_matched_sse_ratio_max": 0.95,
    "G2_all_matched_delta_sse_max": -0.003,
    "G3_min_improving_folds": 3,
    "G4_leave_top5_ratio_max": 0.98,
    "G5_lirf_noncopy_rmse_ratio_max": 1.03,
}
U3_GATES = {
    "G1_unmatched_sse_ratio_max": 0.97,
    "G2_non_lirf_max_abs_change": 0.0,
    "G3_min_improving_folds": 3,
    "G4_leave_top5_ratio_max": 0.99,
    "G5_lirf_ordinary_rmse_ratio_max": 1.03,
}


# ---------------------------------------------------------------------------
# labels, window, mixture
# ---------------------------------------------------------------------------


def copy_label(y: np.ndarray, sd: np.ndarray, tol: float = COPY_TOL_SEC) -> np.ndarray:
    """Training-only schedule-copy label: ``|y - sd| <= tol``. Missing sd is never a copy."""

    y = np.asarray(y, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(sd)
    return ok & (np.abs(np.where(ok, y - sd, np.inf)) <= tol)


def window_from_leads(lead_iobt: np.ndarray, lead_eobt: np.ndarray, window: float = WINDOW_SEC) -> tuple[np.ndarray, np.ndarray]:
    """Hull window in taxi seconds from ``takeoff - IOBT`` and ``takeoff - EOBT_1``.

    Same definition as ``matching_window.window_bounds(mode="hull")``: NaN when either
    lead is missing, lower bound clipped at zero.
    """

    li = np.asarray(lead_iobt, dtype=np.float64)
    le = np.asarray(lead_eobt, dtype=np.float64)
    ok = np.isfinite(li) & np.isfinite(le)
    lo = np.where(ok, np.maximum(np.minimum(li, le) - window, 0.0), np.nan)
    hi = np.where(ok, np.maximum(li, le) + window, np.nan)
    return lo, hi


def project(pred: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Clip into ``[lo, hi]`` where both bounds exist; elsewhere unchanged."""

    pred = np.asarray(pred, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    ok = np.isfinite(lo) & np.isfinite(hi)
    return np.where(ok, np.clip(pred, np.where(ok, lo, -np.inf), np.where(ok, hi, np.inf)), pred)


def copy_feasible(sd: np.ndarray, lead_iobt: np.ndarray, lead_eobt: np.ndarray, window: float = WINDOW_SEC) -> np.ndarray:
    """False when a schedule copy cannot satisfy the matching window.

    ``SCHED - IOBT = (takeoff - IOBT) - (takeoff - SCHED)``. A copy is possible when
    ``|SCHED - IOBT| <= window`` or ``|SCHED - EOBT_1| <= window``. A missing planning
    field gives no constraint from that field; both missing gives no constraint at all.
    A missing sd has no copy candidate.
    """

    sd = np.asarray(sd, dtype=np.float64)
    li = np.asarray(lead_iobt, dtype=np.float64)
    le = np.asarray(lead_eobt, dtype=np.float64)
    fi = np.isfinite(li)
    fe = np.isfinite(le)
    with np.errstate(invalid="ignore"):
        near_i = fi & (np.abs(li - sd) <= window)
        near_e = fe & (np.abs(le - sd) <= window)
    unconstrained = ~fi & ~fe
    return np.isfinite(sd) & (near_i | near_e | unconstrained)


def copy_mixture(p: np.ndarray, sd: np.ndarray, expert: np.ndarray, feasible: np.ndarray, *, gate_sec: float | None = None) -> np.ndarray:
    """``w * max(sd, 0) + (1 - w) * expert`` with ``w = p`` on feasible rows (and sd >= gate), else 0."""

    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    sd = np.asarray(sd, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    if not (p.shape == sd.shape == expert.shape == feasible.shape):
        raise ValueError("mixture inputs must share one shape")
    use = feasible & np.isfinite(sd)
    if gate_sec is not None:
        use = use & (np.nan_to_num(sd, nan=-np.inf) >= gate_sec)
    w = np.where(use, p, 0.0)
    cand = np.maximum(np.nan_to_num(sd, nan=0.0), 0.0)
    return np.where(use, w * cand + (1.0 - w) * expert, expert)


def expert_mask(is_copy: np.ndarray, sd: np.ndarray, *, gate_sec: float | None) -> np.ndarray:
    """Training rows for E_nc: drop copies (all copies, or only copies with sd >= gate)."""

    is_copy = np.asarray(is_copy, dtype=bool)
    if gate_sec is None:
        return ~is_copy
    sd = np.asarray(sd, dtype=np.float64)
    large = np.isfinite(sd) & (sd >= gate_sec)
    return ~(is_copy & large)


def sd_bucket(sd: np.ndarray) -> np.ndarray:
    sd = np.asarray(sd, dtype=np.float64)
    out = np.full(sd.shape[0], "missing", dtype=object)
    for low, high, name in SD_BUCKET_EDGES:
        out = np.where(np.isfinite(sd) & (sd >= low) & (sd < high), name, out)
    return out


# ---------------------------------------------------------------------------
# nested out-of-fold rate encodings
# ---------------------------------------------------------------------------


def smoothed_rate(fit_keys: np.ndarray, fit_label: np.ndarray, apply_keys: np.ndarray, *, n0: float = RATE_N0, prior: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(k + n0 * prior) / (n + n0)`` per key from the fit rows; unseen keys get the prior."""

    fit_keys = np.asarray(fit_keys, dtype=object)
    fit_label = np.asarray(fit_label, dtype=np.float64)
    apply_keys = np.asarray(apply_keys, dtype=object)
    if prior is None:
        prior = float(fit_label.mean()) if fit_label.size else 0.0
    sums: dict[Any, float] = {}
    counts: dict[Any, int] = {}
    for key, lab in zip(fit_keys.tolist(), fit_label.tolist()):
        sums[key] = sums.get(key, 0.0) + lab
        counts[key] = counts.get(key, 0) + 1
    n = np.array([counts.get(k, 0) for k in apply_keys.tolist()], dtype=np.float64)
    s = np.array([sums.get(k, 0.0) for k in apply_keys.tolist()], dtype=np.float64)
    return (s + n0 * prior) / (n + n0), n


def nested_rate_encoding(
    keys: np.ndarray,
    label: np.ndarray,
    months: np.ndarray,
    fit_mask: np.ndarray,
    apply_mask: np.ndarray,
    *,
    n0: float = RATE_N0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rate encodings with no row seeing its own month's labels.

    Fit rows get leave-one-month-out rates (and prior) computed from the *other* fit months only.
    Apply rows get rates from all fit rows. ``label`` is read only on fit rows; rows
    outside both masks come back NaN.
    """

    keys = np.asarray(keys, dtype=object)
    label = np.asarray(label, dtype=np.float64)
    months = np.asarray(months)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    apply_mask = np.asarray(apply_mask, dtype=bool)
    if np.any(fit_mask & apply_mask):
        raise ValueError("fit and apply rows overlap")
    rate = np.full(keys.shape[0], np.nan)
    count = np.full(keys.shape[0], np.nan)
    prior = float(label[fit_mask].mean()) if fit_mask.any() else 0.0
    for month in np.unique(months[fit_mask]):
        inner_fit = fit_mask & (months != month)
        target = fit_mask & (months == month)
        inner_prior = float(label[inner_fit].mean()) if inner_fit.any() else prior
        r, n = smoothed_rate(keys[inner_fit], label[inner_fit], keys[target], n0=n0, prior=inner_prior)
        rate[target] = r
        count[target] = n
    if apply_mask.any():
        r, n = smoothed_rate(keys[fit_mask], label[fit_mask], keys[apply_mask], n0=n0, prior=prior)
        rate[apply_mask] = r
        count[apply_mask] = n
    return rate, count


# ---------------------------------------------------------------------------
# U3 gate delay
# ---------------------------------------------------------------------------


def gate_delay_target(y: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """``G = BLOCK - SCHED = (takeoff - SCHED) - (takeoff - BLOCK) = sd - y`` (training only)."""

    return np.asarray(sd, dtype=np.float64) - np.asarray(y, dtype=np.float64)


def serve_gate_delay(sd: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
    """``T = max(sd - G_hat, 0)``; missing sd gives NaN (caller falls back to the control)."""

    sd = np.asarray(sd, dtype=np.float64)
    g_hat = np.asarray(g_hat, dtype=np.float64)
    return np.where(np.isfinite(sd), np.maximum(sd - g_hat, 0.0), np.nan)


def blend(control: np.ndarray, other: np.ndarray, lam: float) -> np.ndarray:
    """``(1 - lam) * control + lam * other``; rows where ``other`` is missing keep the control."""

    control = np.asarray(control, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    return np.where(np.isfinite(other), (1.0 - lam) * control + lam * np.nan_to_num(other), control)


# ---------------------------------------------------------------------------
# label-free LIRF context (other flights' ARR rows only)
# ---------------------------------------------------------------------------


def arr_daily_copy_rate(arr: pl.DataFrame) -> pl.DataFrame:
    """Per (month, UTC day of the ARR movement) at LIRF: share of ARR rows with |in-block - SCHED| <= 60 s."""

    return (
        arr.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"))
        .group_by(["month", "day"])
        .agg(
            pl.len().alias("arr_day_n"),
            ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60).cast(pl.Float64).mean().alias("arr_day_copy_rate"),
        )
    )


def stand_inbound_features(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Latest same-stand LIRF ARR in-block at or before the departure's takeoff, same month, within 24 h.

    Uses other flights' ARR in-block times only (no DEP BLOCK). Returns MVT_ID_mvt + STAND_NUMS.
    """

    left = (
        dep.select("MVT_ID_mvt", "month", "STAND_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
        .filter(pl.col("STAND_mvt").is_not_null())
        .sort("MVT_TIME_UTC_mvt")
    )
    right = (
        arr.select(
            "month",
            "STAND_mvt",
            pl.col("BLOCK_TIME_UTC_mvt").alias("arr_inblock"),
            pl.col("SCHED_TIME_UTC_mvt").alias("arr_sched"),
        )
        .filter(pl.col("STAND_mvt").is_not_null() & pl.col("arr_inblock").is_not_null())
        .sort("arr_inblock")
    )
    joined = left.join_asof(
        right,
        left_on="MVT_TIME_UTC_mvt",
        right_on="arr_inblock",
        by=["month", "STAND_mvt"],
        strategy="backward",
        tolerance="24h",
    )
    out = joined.select(
        "MVT_ID_mvt",
        (pl.col("arr_inblock") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("stand_prev_arr_minus_sched_sec"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("arr_inblock")).dt.total_seconds().cast(pl.Float64).alias("takeoff_minus_stand_prev_arr_sec"),
        (pl.col("arr_inblock") - pl.col("arr_sched")).dt.total_seconds().cast(pl.Float64).alias("stand_prev_arr_delay_sec"),
    )
    return dep.select("MVT_ID_mvt").join(out, on="MVT_ID_mvt", how="left", maintain_order="left")


# ---------------------------------------------------------------------------
# reports and gates
# ---------------------------------------------------------------------------


def _scores(y: np.ndarray, pred: np.ndarray, base: np.ndarray) -> dict[str, float]:
    from prc_challenge.unmatched_recovery.noncopy_expert import slice_scores

    return slice_scores(y, pred, base)


def u2_report(
    y: np.ndarray,
    cand: np.ndarray,
    parent: np.ndarray,
    *,
    fold: Sequence[Any],
    is_copy: np.ndarray,
    sd: np.ndarray,
    all_matched_parent_sse: float,
    k: int = TOP_K,
) -> dict[str, Any]:
    """LIRF-matched metrics against the parent; the all-matched delta uses the parent's pooled SSE."""

    from prc_challenge.unmatched_recovery.noncopy_expert import leave_top_k_ratio

    y = np.asarray(y, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.float64)
    if not np.isfinite(cand).all():
        raise ValueError("non-finite candidate predictions")
    fold = np.asarray(fold, dtype=object)
    is_copy = np.asarray(is_copy, dtype=bool)
    sd = np.asarray(sd, dtype=np.float64)
    pooled = _scores(y, cand, parent)
    delta = pooled["sse"] - pooled["base_sse"]
    buckets = sd_bucket(sd)
    return {
        "lirf": pooled,
        "all_matched_delta_sse": delta,
        "all_matched_delta_ratio": delta / all_matched_parent_sse,
        "noncopy": _scores(y[~is_copy], cand[~is_copy], parent[~is_copy]),
        "copy": _scores(y[is_copy], cand[is_copy], parent[is_copy]),
        "leave_top_k": leave_top_k_ratio(y, cand, parent, k),
        "folds": {str(f): _scores(y[fold == f], cand[fold == f], parent[fold == f]) for f in sorted(set(fold.tolist()))},
        "sd_buckets": {str(b): _scores(y[buckets == b], cand[buckets == b], parent[buckets == b]) for b in sorted(set(buckets.tolist()))},
        "ordinary_lt_1h": _scores(y[y < ORDINARY_SEC], cand[y < ORDINARY_SEC], parent[y < ORDINARY_SEC]),
    }


def u2_gates(report: dict[str, Any]) -> dict[str, Any]:
    ratio = report["lirf"]["sse_ratio"]
    delta = report["all_matched_delta_ratio"]
    improving = sum(1 for row in report["folds"].values() if np.isfinite(row["sse_ratio"]) and row["sse_ratio"] < 1.0)
    top = report["leave_top_k"]["sse_ratio"]
    nc = report["noncopy"]
    nc_ratio = nc["rmse"] / nc["base_rmse"] if nc["base_rmse"] > 0 else float("nan")
    g = U2_GATES
    gates = {
        "G1_lirf_matched_sse_ratio": {"value": ratio, "threshold": g["G1_lirf_matched_sse_ratio_max"], "pass": bool(np.isfinite(ratio) and ratio <= g["G1_lirf_matched_sse_ratio_max"])},
        "G2_all_matched_delta_sse": {"value": delta, "threshold": g["G2_all_matched_delta_sse_max"], "pass": bool(np.isfinite(delta) and delta <= g["G2_all_matched_delta_sse_max"])},
        "G3_improving_folds": {"value": improving, "threshold": g["G3_min_improving_folds"], "pass": bool(improving >= g["G3_min_improving_folds"])},
        "G4_leave_top5_ratio": {"value": top, "threshold": g["G4_leave_top5_ratio_max"], "pass": bool(np.isfinite(top) and top <= g["G4_leave_top5_ratio_max"])},
        "G5_lirf_noncopy_rmse_ratio": {"value": nc_ratio, "threshold": g["G5_lirf_noncopy_rmse_ratio_max"], "pass": bool(np.isfinite(nc_ratio) and nc_ratio <= g["G5_lirf_noncopy_rmse_ratio_max"])},
    }
    return {"gates": gates, "go": all(v["pass"] for v in gates.values()), "failed": [n for n, v in gates.items() if not v["pass"]]}


def u3_report(
    y: np.ndarray,
    cand: np.ndarray,
    control: np.ndarray,
    *,
    airport: Sequence[Any],
    fold: Sequence[Any],
    k: int = TOP_K,
) -> dict[str, Any]:
    from prc_challenge.unmatched_recovery.noncopy_expert import leave_top_k_ratio

    y = np.asarray(y, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if not np.isfinite(cand).all():
        raise ValueError("non-finite candidate predictions")
    airport = np.asarray(airport, dtype=object)
    fold = np.asarray(fold, dtype=object)
    lirf = airport == "LIRF"
    ordinary = lirf & (y < ORDINARY_SEC)
    return {
        "pooled": _scores(y, cand, control),
        "lirf": _scores(y[lirf], cand[lirf], control[lirf]),
        "lirf_ordinary_lt_1h": _scores(y[ordinary], cand[ordinary], control[ordinary]),
        "lirf_ge_1h": _scores(y[lirf & ~ordinary], cand[lirf & ~ordinary], control[lirf & ~ordinary]),
        "non_lirf_max_abs_change": float(np.max(np.abs(cand[~lirf] - control[~lirf]))) if (~lirf).any() else 0.0,
        "leave_top_k": leave_top_k_ratio(y, cand, control, k),
        "folds": {str(f): _scores(y[fold == f], cand[fold == f], control[fold == f]) for f in sorted(set(fold.tolist()))},
    }


def u3_gates(report: dict[str, Any]) -> dict[str, Any]:
    ratio = report["pooled"]["sse_ratio"]
    moved = report["non_lirf_max_abs_change"]
    improving = sum(1 for row in report["folds"].values() if np.isfinite(row["sse_ratio"]) and row["sse_ratio"] < 1.0)
    top = report["leave_top_k"]["sse_ratio"]
    o = report["lirf_ordinary_lt_1h"]
    o_ratio = o["rmse"] / o["base_rmse"] if o["base_rmse"] > 0 else float("nan")
    g = U3_GATES
    gates = {
        "G1_unmatched_sse_ratio": {"value": ratio, "threshold": g["G1_unmatched_sse_ratio_max"], "pass": bool(np.isfinite(ratio) and ratio <= g["G1_unmatched_sse_ratio_max"])},
        "G2_non_lirf_unchanged": {"value": moved, "threshold": g["G2_non_lirf_max_abs_change"], "pass": bool(moved <= g["G2_non_lirf_max_abs_change"])},
        "G3_improving_folds": {"value": improving, "threshold": g["G3_min_improving_folds"], "pass": bool(improving >= g["G3_min_improving_folds"])},
        "G4_leave_top5_ratio": {"value": top, "threshold": g["G4_leave_top5_ratio_max"], "pass": bool(np.isfinite(top) and top <= g["G4_leave_top5_ratio_max"])},
        "G5_lirf_ordinary_rmse_ratio": {"value": o_ratio, "threshold": g["G5_lirf_ordinary_rmse_ratio_max"], "pass": bool(np.isfinite(o_ratio) and o_ratio <= g["G5_lirf_ordinary_rmse_ratio_max"])},
    }
    return {"gates": gates, "go": all(v["pass"] for v in gates.values()), "failed": [n for n, v in gates.items() if not v["pass"]]}


def choose_arm(gated: dict[str, dict[str, Any]], ratios: dict[str, float]) -> str | None:
    """Among arms passing every gate, the lowest primary SSE ratio."""

    passing = [name for name, g in gated.items() if g.get("go")]
    return min(passing, key=lambda n: ratios[n]) if passing else None
