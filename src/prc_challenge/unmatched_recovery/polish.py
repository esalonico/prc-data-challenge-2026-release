# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Unmatched polish (SA-UNM2, 2026-09-25): P1 gate-delay blend, P2 METAR extras, P12 combination.

Pre-registration: ``progress-review/unmatched-polish-20260925/preregistration.json``.

All three arms are scored on the 14,066 blocked-D unmatched rows against the deployed
A3b-nodest control (E17/E18 + clip on both sides). Every helper here is label-agnostic
apart from the scoring functions, which receive ``y`` explicitly.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from prc_challenge.unmatched_recovery.noncopy_expert import improvement, leave_top_k_ratio, slice_scores

LAMBDA = 0.5
GATE_SEC = 3600.0
ORDINARY_SEC = 3600.0
EXTREME_SEC = 7200.0
TOP_K = 5
ARM_ORDER = ("P1", "P2", "P12")

P1_GATES = {
    "G1_pooled_sse_ratio_max": 0.97,
    "G2_min_improving_folds": 3,
    "G3_leave_top5_ratio_max": 0.99,
    "G4_lirf_ordinary_rmse_ratio_max": 1.03,
    "G5_non_lirf_max_abs_change": 0.0,
}
P2_GATES = {
    "G1_pooled_sse_ratio_max": 0.985,
    "G2_min_improving_folds": 3,
    "G3_leave_top5_ratio_max": 0.995,
    "G4_ordinary_rmse_ratio_max": 1.02,
}
P12_GATES = {
    "G1_pooled_sse_ratio_max": 0.97,
    "G2_min_improving_folds": 3,
    "G3_leave_top5_ratio_max": 0.99,
    "G4_ordinary_rmse_ratio_max": 1.02,
    "G5_lirf_ordinary_rmse_ratio_max": 1.03,
    "G6_non_lirf_max_abs_change_vs_p2": 0.0,
}


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


def blend_on_mask(control: np.ndarray, other: np.ndarray, mask: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """``(1 - lam) * control + lam * other`` where ``mask`` holds and ``other`` is finite; else control."""

    control = np.asarray(control, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not (control.shape == other.shape == mask.shape):
        raise ValueError("blend inputs must share one shape")
    use = mask & np.isfinite(other)
    return np.where(use, (1.0 - lam) * control + lam * np.nan_to_num(other), control)


# ---------------------------------------------------------------------------
# reports and gates
# ---------------------------------------------------------------------------


def _ratio(block: dict[str, float]) -> float:
    return block["rmse"] / block["base_rmse"] if block["base_rmse"] > 0 else float("nan")


def polish_report(
    y: np.ndarray,
    cand: np.ndarray,
    control: np.ndarray,
    *,
    airport: Sequence[Any],
    fold: Sequence[Any],
    reference: np.ndarray | None = None,
    k: int = TOP_K,
) -> dict[str, Any]:
    """Every pre-registered metric of one arm against the control.

    ``reference`` (P12 only) is the P2 vector; the off-LIRF identity is measured against it.
    """

    y = np.asarray(y, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if not np.isfinite(cand).all():
        raise ValueError("non-finite candidate predictions")
    airport = np.asarray(airport, dtype=object)
    fold = np.asarray(fold, dtype=object)
    lirf = airport == "LIRF"
    ordinary = y < ORDINARY_SEC
    mid = (y >= ORDINARY_SEC) & (y < EXTREME_SEC)
    tail = y >= EXTREME_SEC
    out: dict[str, Any] = {
        "pooled": slice_scores(y, cand, control),
        "ordinary_lt_1h": slice_scores(y[ordinary], cand[ordinary], control[ordinary]),
        "mid_1h_2h": slice_scores(y[mid], cand[mid], control[mid]),
        "tail_ge_2h": slice_scores(y[tail], cand[tail], control[tail]),
        "lirf": slice_scores(y[lirf], cand[lirf], control[lirf]),
        "lirf_ordinary_lt_1h": slice_scores(y[lirf & ordinary], cand[lirf & ordinary], control[lirf & ordinary]),
        "non_lirf": slice_scores(y[~lirf], cand[~lirf], control[~lirf]),
        "non_lirf_max_abs_change": float(np.max(np.abs(cand[~lirf] - control[~lirf]))) if (~lirf).any() else 0.0,
        "rows_changed": int(np.sum(np.abs(cand - control) > 1e-9)),
        "leave_top_k": leave_top_k_ratio(y, cand, control, k),
        "folds": {str(f): slice_scores(y[fold == f], cand[fold == f], control[fold == f]) for f in sorted(set(fold.tolist()))},
        "airports": {str(a): slice_scores(y[airport == a], cand[airport == a], control[airport == a]) for a in sorted(set(airport.tolist()))},
    }
    if reference is not None:
        reference = np.asarray(reference, dtype=np.float64)
        out["non_lirf_max_abs_change_vs_reference"] = float(np.max(np.abs(cand[~lirf] - reference[~lirf]))) if (~lirf).any() else 0.0
    return out


def _improving(report: dict[str, Any]) -> int:
    return sum(1 for row in report["folds"].values() if np.isfinite(row["sse_ratio"]) and row["sse_ratio"] < 1.0)


def _gate(value: float, threshold: float, *, at_most: bool = True) -> dict[str, Any]:
    ok = bool(np.isfinite(value) and (value <= threshold if at_most else value >= threshold))
    return {"value": float(value), "threshold": float(threshold), "pass": ok}


def _finish(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"gates": gates, "go": all(g["pass"] for g in gates.values()), "failed": [n for n, g in gates.items() if not g["pass"]]}


def p1_gates(report: dict[str, Any]) -> dict[str, Any]:
    g = P1_GATES
    return _finish(
        {
            "G1_pooled_sse_ratio": _gate(report["pooled"]["sse_ratio"], g["G1_pooled_sse_ratio_max"]),
            "G2_improving_folds": _gate(_improving(report), g["G2_min_improving_folds"], at_most=False),
            "G3_leave_top5_ratio": _gate(report["leave_top_k"]["sse_ratio"], g["G3_leave_top5_ratio_max"]),
            "G4_lirf_ordinary_rmse_ratio": _gate(_ratio(report["lirf_ordinary_lt_1h"]), g["G4_lirf_ordinary_rmse_ratio_max"]),
            "G5_non_lirf_unchanged": _gate(report["non_lirf_max_abs_change"], g["G5_non_lirf_max_abs_change"]),
        }
    )


def p2_gates(report: dict[str, Any]) -> dict[str, Any]:
    g = P2_GATES
    return _finish(
        {
            "G1_pooled_sse_ratio": _gate(report["pooled"]["sse_ratio"], g["G1_pooled_sse_ratio_max"]),
            "G2_improving_folds": _gate(_improving(report), g["G2_min_improving_folds"], at_most=False),
            "G3_leave_top5_ratio": _gate(report["leave_top_k"]["sse_ratio"], g["G3_leave_top5_ratio_max"]),
            "G4_ordinary_rmse_ratio": _gate(_ratio(report["ordinary_lt_1h"]), g["G4_ordinary_rmse_ratio_max"]),
        }
    )


def p12_gates(report: dict[str, Any]) -> dict[str, Any]:
    g = P12_GATES
    if "non_lirf_max_abs_change_vs_reference" not in report:
        raise ValueError("P12 report needs the P2 reference")
    return _finish(
        {
            "G1_pooled_sse_ratio": _gate(report["pooled"]["sse_ratio"], g["G1_pooled_sse_ratio_max"]),
            "G2_improving_folds": _gate(_improving(report), g["G2_min_improving_folds"], at_most=False),
            "G3_leave_top5_ratio": _gate(report["leave_top_k"]["sse_ratio"], g["G3_leave_top5_ratio_max"]),
            "G4_ordinary_rmse_ratio": _gate(_ratio(report["ordinary_lt_1h"]), g["G4_ordinary_rmse_ratio_max"]),
            "G5_lirf_ordinary_rmse_ratio": _gate(_ratio(report["lirf_ordinary_lt_1h"]), g["G5_lirf_ordinary_rmse_ratio_max"]),
            "G6_non_lirf_identical_to_p2": _gate(report["non_lirf_max_abs_change_vs_reference"], g["G6_non_lirf_max_abs_change_vs_p2"]),
        }
    )


# ---------------------------------------------------------------------------
# Expert cap (preregistration_cap.json)
# ---------------------------------------------------------------------------

CAP_SEC = 3606.0
CAP_ALT_SEC = 7200.0
CAP_GATES = {
    "G1_pooled_sse_ratio_max": 1.0,
    "G2_fold_sse_ratio_max": 1.02,
    "G3_ordinary_rmse_ratio_max": 1.02,
    "G4_leave_top5_ratio_max": 1.0,
}
CAP_PRIMARY = ("CAP_A", "CAP_B")
CAP_FALLBACK = ("CAP_A7200", "CAP_B7200")


def cap_expert(expert: np.ndarray, sd: np.ndarray, cap_sec: float = CAP_SEC) -> tuple[np.ndarray, np.ndarray]:
    """``min(E, max(sd, 0) + cap)`` where ``sd`` is finite; elsewhere unchanged. Returns (capped, touched)."""

    expert = np.asarray(expert, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    if expert.shape != sd.shape:
        raise ValueError("expert and sd must share one shape")
    ok = np.isfinite(sd)
    bound = np.where(ok, np.maximum(np.nan_to_num(sd, nan=0.0), 0.0) + cap_sec, np.inf)
    touched = ok & (expert > bound)
    return np.where(touched, bound, expert), touched


def cap_gates(report: dict[str, Any]) -> dict[str, Any]:
    g = CAP_GATES
    worst_fold = max(v["sse_ratio"] for v in report["folds"].values())
    return _finish(
        {
            "G1_pooled_sse_ratio": _gate(report["pooled"]["sse_ratio"], g["G1_pooled_sse_ratio_max"]),
            "G2_worst_fold_sse_ratio": _gate(worst_fold, g["G2_fold_sse_ratio_max"]),
            "G3_ordinary_rmse_ratio": _gate(_ratio(report["ordinary_lt_1h"]), g["G3_ordinary_rmse_ratio_max"]),
            "G4_leave_top5_ratio": _gate(report["leave_top_k"]["sse_ratio"], g["G4_leave_top5_ratio_max"]),
        }
    )


def select_cap_arm(gated: dict[str, dict[str, Any]], ratios: dict[str, float]) -> str | None:
    """Primary constant first (lowest pooled ratio, tie -> CAP_A); the 7200 arms only if no primary arm passes."""

    for group in (CAP_PRIMARY, CAP_FALLBACK):
        passing = [name for name in group if name in gated and gated[name].get("go")]
        if passing:
            return min(passing, key=lambda n: (ratios[n], group.index(n)))
    return None


def select_arm(gated: dict[str, dict[str, Any]], ratios: dict[str, float]) -> str | None:
    """Lowest pooled SSE ratio among passing arms; ties go to the simpler arm (P1 < P2 < P12)."""

    passing = [name for name in ARM_ORDER if name in gated and gated[name].get("go")]
    if not passing:
        return None
    return min(passing, key=lambda n: (ratios[n], ARM_ORDER.index(n)))


def flat_row(name: str, report: dict[str, Any], gates: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "arm": name,
        "pooled_sse_ratio": report["pooled"]["sse_ratio"],
        "rmse": report["pooled"]["rmse"],
        "control_rmse": report["pooled"]["base_rmse"],
        "ordinary_rmse_ratio": _ratio(report["ordinary_lt_1h"]),
        "lirf_ordinary_rmse_ratio": _ratio(report["lirf_ordinary_lt_1h"]),
        "lirf_sse_ratio": report["lirf"]["sse_ratio"],
        "non_lirf_sse_ratio": report["non_lirf"]["sse_ratio"],
        "non_lirf_max_abs_change": report["non_lirf_max_abs_change"],
        "mid_1h_2h_rmse": report["mid_1h_2h"]["rmse"],
        "tail_ge_2h_rmse": report["tail_ge_2h"]["rmse"],
        "leave_top5_sse_ratio": report["leave_top_k"]["sse_ratio"],
        "top1_share_of_gain": report["leave_top_k"]["top1_share_of_gain"],
        "top5_share_of_gain": report["leave_top_k"]["top_share_of_gain"],
        "rows_changed": report["rows_changed"],
    }
    for fold, scores in report["folds"].items():
        row[f"fold_{fold}_sse_ratio"] = scores["sse_ratio"]
    if gates is not None:
        row["go"] = gates["go"]
        row["failed_gates"] = ";".join(gates["failed"])
    return row


# ---------------------------------------------------------------------------
# January/July confirmation helpers (labels are passed in explicitly)
# ---------------------------------------------------------------------------


def day_cluster_bootstrap(
    y: np.ndarray,
    control: np.ndarray,
    cand: np.ndarray,
    day: Sequence[Any],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.90,
) -> dict[str, Any]:
    """Resample calendar days with replacement; interval of the full-RMSE change (cand - control)."""

    y = np.asarray(y, dtype=np.float64)
    se_c = (y - np.asarray(control, dtype=np.float64)) ** 2
    se_k = (y - np.asarray(cand, dtype=np.float64)) ** 2
    day = np.asarray(day, dtype=object)
    keys = sorted(set(day.tolist()), key=str)
    idx = {k: i for i, k in enumerate(keys)}
    code = np.array([idx[d] for d in day.tolist()])
    n_days = len(keys)
    sum_c = np.bincount(code, weights=se_c, minlength=n_days)
    sum_k = np.bincount(code, weights=se_k, minlength=n_days)
    cnt = np.bincount(code, minlength=n_days).astype(np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_days, n_days)
        n = cnt[pick].sum()
        deltas[b] = np.sqrt(sum_k[pick].sum() / n) - np.sqrt(sum_c[pick].sum() / n)
    lo, hi = np.quantile(deltas, [(1 - level) / 2, 1 - (1 - level) / 2])
    point = float(np.sqrt(se_k.mean()) - np.sqrt(se_c.mean()))
    return {
        "n_days": int(n_days),
        "n_boot": int(n_boot),
        "level": level,
        "point_delta_rmse": point,
        "interval": [float(lo), float(hi)],
        "days_improved": int(np.sum(sum_k < sum_c)),
        "days_worse": int(np.sum(sum_k > sum_c)),
        "p_delta_below_0": float(np.mean(deltas < 0)),
    }


def confirmation_summary(
    y: np.ndarray,
    control: np.ndarray,
    cand: np.ndarray,
    *,
    month: np.ndarray,
    unmatched: np.ndarray,
    airport: Sequence[Any],
    k: int = TOP_K,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    month = np.asarray(month)
    unmatched = np.asarray(unmatched, dtype=bool)
    airport = np.asarray(airport, dtype=object)
    lirf_u = unmatched & (airport == "LIRF")

    def block(pred: np.ndarray) -> dict[str, float]:
        se = (y - pred) ** 2
        return {
            "full_rmse": float(np.sqrt(se.mean())),
            "full_sse": float(se.sum()),
            "JAN_rmse": float(np.sqrt(se[month == 1].mean())),
            "JUL_rmse": float(np.sqrt(se[month == 7].mean())),
            "matched_rmse": float(np.sqrt(se[~unmatched].mean())),
            "unmatched_rmse": float(np.sqrt(se[unmatched].mean())),
            "lirf_unmatched_rmse": float(np.sqrt(se[lirf_u].mean())) if lirf_u.any() else float("nan"),
            "non_lirf_unmatched_rmse": float(np.sqrt(se[unmatched & ~lirf_u].mean())),
        }

    gain = improvement(y, cand, control)
    order = np.argsort(-gain)
    total = float(gain.sum())
    top_u = leave_top_k_ratio(y[unmatched], cand[unmatched], control[unmatched], k)
    return {
        "control": block(control),
        "candidate": block(cand),
        "delta_full_rmse": float(np.sqrt(np.mean((y - cand) ** 2)) - np.sqrt(np.mean((y - control) ** 2))),
        "delta_sse": float(-total),
        "jan_delta_sse": float(-gain[month == 1].sum()),
        "jul_delta_sse": float(-gain[month == 7].sum()),
        "rows_changed": int(np.sum(np.abs(cand - control) > 1e-9)),
        "matched_rows_changed": int(np.sum(np.abs(cand[~unmatched] - control[~unmatched]) > 1e-9)),
        "unmatched_sse_ratio": float(np.sum((y[unmatched] - cand[unmatched]) ** 2) / np.sum((y[unmatched] - control[unmatched]) ** 2)),
        "lirf_unmatched_sse_ratio": float(np.sum((y[lirf_u] - cand[lirf_u]) ** 2) / np.sum((y[lirf_u] - control[lirf_u]) ** 2)) if lirf_u.any() else None,
        "unmatched_leave_top5_ratio": top_u["sse_ratio"],
        "top1_share_of_gain": float(gain[order[0]] / total) if total > 0 else None,
        "top5_share_of_gain": float(gain[order[:5]].sum() / total) if total > 0 else None,
        "top20_share_of_gain": float(gain[order[:20]].sum() / total) if total > 0 else None,
        "airports_unmatched": {
            str(a): slice_scores(y[unmatched & (airport == a)], cand[unmatched & (airport == a)], control[unmatched & (airport == a)])
            for a in sorted(set(airport[unmatched].tolist()))
        },
    }


def confirmation_reading(summary: dict[str, Any], bootstrap: dict[str, Any], *, control_projected_rmse: float) -> dict[str, Any]:
    """Pre-registered GO/PARK reading on the projected system."""

    a = summary["candidate"]["full_rmse"] < control_projected_rmse
    b = bootstrap["interval"][1] < 0.0
    c = summary["unmatched_leave_top5_ratio"] < 1.0
    return {
        "a_projected_full_rmse_below_control": bool(a),
        "b_bootstrap_90_interval_below_0": bool(b),
        "c_unmatched_leave_top5_below_1": bool(c),
        "go": bool(a and b and c),
    }
