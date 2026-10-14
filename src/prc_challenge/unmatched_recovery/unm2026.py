# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-UNM5 (2026-09-27): 2026-focused unmatched work on top of ALL + C0.

Step 1 (label-free): ADS-B exposure of the unmatched populations (D, JAN_JUL 2025, ranking 2026).
Step 2/3: a subset gate on ADS-B-observable rows and three arms (U-MAP2, U-NN, U-SENS2).

Everything in the exposure part is label-free. Scoring helpers receive ``y`` explicitly.

ADS-B-derived off-block estimates, expressed as taxi seconds (MVT - off-block):

* ``gate``:  ``adsb_taxi_hat`` (v2 stand gate minus the month-isolated gate bias) where the gate is
  confirmed (``adsb_gate_conf == 1``);
* ``map``:   ``-adsb3_map_first_rel`` (v3-lean travel-time map) where ``adsb3_map_first_sd <= 180``;
* ``first``: ``adsb_taxi_hat_first`` (bias-corrected first-seen channel), where present. The first
  sighting is after off-block for most ground-only rows, so this channel is reported but is not a
  "physical" estimate in the exposure classes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl

from prc_challenge.unmatched_recovery.recovery import ArmSpec, Components, compose, expert_ceiling_d
from prc_challenge.unmatched_recovery.sensor import S_CEILING_G, sensor_blend

MAP_SD_MAX = 180.0
FAR_SEC = 1800.0
BLOCK_COLS = ("adsb_join_src", "adsb_coverage", "adsb_n_seg", "adsb_gate_conf", "adsb_taxi_hat", "adsb_taxi_hat_first", "adsb_seg_first_rel")
LEAN_COLS = ("adsb3_map_first_rel", "adsb3_map_first_sd", "adsb3_map_wmean_rel", "adsb3_map_sd_min", "adsb3_map_first_t", "adsb3_map_n", "adsb3_aobt_minus_map")

# exposure classes, mutually exclusive, in priority order
EXPOSURE_CLASSES = ("gate_conf", "seg_map", "map_only", "seg_only", "join_no_seg", "no_adsb")


def adsb_channels(ids: pl.DataFrame, block: pl.DataFrame, lean: pl.DataFrame) -> pl.DataFrame:
    """Per row: the v2 block columns, the lean map columns and derived channel estimates (label-free)."""

    b = block.select("MVT_ID_mvt", *[c for c in BLOCK_COLS if c in block.columns])
    lv = lean.select("MVT_ID_mvt", *[c for c in LEAN_COLS if c in lean.columns])
    z = ids.select("MVT_ID_mvt").join(b, on="MVT_ID_mvt", how="left", maintain_order="left").join(lv, on="MVT_ID_mvt", how="left", maintain_order="left")
    if z.height != ids.height:
        raise RuntimeError("channel join changed the row count")
    map_ok = pl.col("adsb3_map_first_rel").is_not_null() & pl.col("adsb3_map_first_sd").is_not_null() & (pl.col("adsb3_map_first_sd") <= MAP_SD_MAX)
    gate_ok = (pl.col("adsb_gate_conf") == 1) & pl.col("adsb_taxi_hat").is_not_null()
    seg = pl.col("adsb_n_seg").fill_null(0) > 0
    joined = pl.col("adsb_join_src").fill_null(0) > 0
    z = z.with_columns(
        map_ok.fill_null(False).alias("map_ok"),
        gate_ok.fill_null(False).alias("gate_ok"),
        seg.alias("has_seg"),
        joined.alias("joined"),
        pl.when(map_ok).then(-pl.col("adsb3_map_first_rel")).otherwise(None).alias("taxi_map"),
        pl.when(gate_ok).then(pl.col("adsb_taxi_hat")).otherwise(None).alias("taxi_gate"),
        pl.col("adsb_taxi_hat_first").alias("taxi_first"),
    )
    cls = (
        pl.when(pl.col("gate_ok")).then(pl.lit("gate_conf"))
        .when(pl.col("has_seg") & pl.col("map_ok")).then(pl.lit("seg_map"))
        .when(pl.col("map_ok")).then(pl.lit("map_only"))
        .when(pl.col("has_seg")).then(pl.lit("seg_only"))
        .when(pl.col("joined")).then(pl.lit("join_no_seg"))
        .otherwise(pl.lit("no_adsb"))
    )
    z = z.with_columns(cls.alias("exposure"), (pl.col("gate_ok") | pl.col("map_ok")).alias("physical"))
    # a reliable map without v2 segments: the v3 kinematics use the stage-1 join before the v2 linker
    # precision gate (EHAM take-off-time fallback joins, precision 0.947 < 0.95, are cleared in v2)
    z = z.with_columns((pl.col("map_ok") & ~pl.col("has_seg")).alias("map_without_seg"))
    return z


def far_from_adsb(pred: np.ndarray, ch: pl.DataFrame, *, far_sec: float = FAR_SEC, include_first: bool = False) -> np.ndarray:
    """True where the row has >= 1 physical estimate and the prediction is > far_sec from all of them."""

    pred = np.asarray(pred, dtype=np.float64)
    cols = ["taxi_gate", "taxi_map", *(["taxi_first"] if include_first else [])]
    est = np.column_stack([ch[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in cols])
    have = np.isfinite(est).any(axis=1)
    dist = np.where(np.isfinite(est), np.abs(est - pred[:, None]), np.inf).min(axis=1)
    return have & (dist > far_sec)


def exposure_table(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """frame: one row per unmatched row with the columns of ``adsb_channels`` + fired, far, far_first, by-keys.

    ``phys_gap`` = prediction - physical estimate (map if reliable, else confirmed gate), null elsewhere.
    """

    aggs = [pl.len().alias("n"), pl.col("fired").sum().alias("usens_fired")]
    aggs += [(pl.col("exposure") == k).sum().alias(k) for k in EXPOSURE_CLASSES]
    aggs += [
        pl.col("has_seg").sum().alias("any_seg"),
        pl.col("map_ok").sum().alias("map_reliable"),
        (pl.col("map_ok") & ~pl.col("gate_ok")).sum().alias("map_no_conf_gate"),
        pl.col("physical").sum().alias("physical"),
        (pl.col("physical") & ~pl.col("fired")).sum().alias("physical_not_fired"),
        (pl.col("phys_gap").abs() > 600).sum().alias("gap_gt_600"),
        (pl.col("phys_gap").abs() > 900).sum().alias("gap_gt_900"),
        (pl.col("phys_gap") ** 2).sum().round(0).alias("sum_sq_gap"),
        pl.col("far").sum().alias("far_1800"),
        pl.col("far_first").sum().alias("far_1800_incl_first"),
    ]
    out = frame.group_by(by, maintain_order=False).agg(aggs).sort(by)
    return out.with_columns(
        (pl.col("usens_fired") / pl.col("n")).round(3).alias("fired_share"),
        (pl.col("physical") / pl.col("n")).round(3).alias("physical_share"),
    )


# ---------------------------------------------------------------------------
# composition with a replaced sensor stage
# ---------------------------------------------------------------------------


def sensor_stage(c: Components, arm: ArmSpec) -> tuple[np.ndarray, np.ndarray]:
    """The deployed expert after the arm-D ceiling and the U-SENS-G blend: (E', eligible)."""

    e = np.asarray(c.e_cap, dtype=np.float64)
    if arm.d:
        e = expert_ceiling_d(e, c.sd, c.airport)
    return sensor_blend(e, c.s, c.sd, c.cov, c.airport, c.p, s_ceiling=S_CEILING_G)


def compose_with_expert(c: Components, arm: ArmSpec, e_prime: np.ndarray) -> np.ndarray:
    """compose() with the post-sensor expert replaced by ``e_prime`` (arm C's AOBT blend is not supported)."""

    if arm.c:
        raise ValueError("arm C (AOBT blend) is not supported with an expert override")
    c2 = replace(c, e_cap=np.asarray(e_prime, dtype=np.float64), s=np.full(c.n, np.nan), cov=np.full(c.n, np.nan))
    return compose(c2, replace(arm, d=False))


# ---------------------------------------------------------------------------
# subset gates (step 2)
# ---------------------------------------------------------------------------


def subset_gates(y, ctl, cand, *, day, fold, airport, subset, lirf_copy, n_boot: int = 2000, seed: int = 0, airport_cap: float = 0.02) -> dict[str, Any]:
    """The pre-registered SA-UNM5 gate (preregistration.json, section gate).

    P1: day-bootstrap p95 of dSSE on O < 0; P2: >= 3/4 folds with dSSE on O < 0; P3: no airport's dSSE on O
    > +2% of its control SSE on O; G1: outside O the day-bootstrap p05 of dSSE <= 0; G2: LIRF copy-regime
    rows unchanged.
    """

    from prc_challenge.unmatched_recovery.sensor import delta_sse_bootstrap

    y = np.asarray(y, dtype=np.float64)
    ctl = np.asarray(ctl, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    subset = np.asarray(subset, dtype=bool)
    lirf_copy = np.asarray(lirf_copy, dtype=bool)
    day = np.asarray(day, dtype=object)
    fold = np.asarray(fold, dtype=object)
    airport = np.asarray(airport, dtype=object)
    if not np.isfinite(cand).all():
        raise ValueError("non-finite candidate")
    d = (y - cand) ** 2 - (y - ctl) ** 2
    se_c = (y - ctl) ** 2
    sub_boot = delta_sse_bootstrap(y[subset], ctl[subset], cand[subset], day[subset], n_boot=n_boot, seed=seed)
    non = ~subset
    non_boot = delta_sse_bootstrap(y[non], ctl[non], cand[non], day[non], n_boot=n_boot, seed=seed) if non.any() else None
    folds = {str(f): {"n_subset": int((subset & (fold == f)).sum()), "delta_sse_subset": float(d[subset & (fold == f)].sum()), "ctl_sse_subset": float(se_c[subset & (fold == f)].sum()), "delta_sse_full": float(d[fold == f].sum())} for f in sorted(set(fold.tolist()))}
    aps = {}
    for a in sorted(set(airport[subset].tolist())):
        m = subset & (airport == a)
        aps[str(a)] = {
            "n": int(m.sum()),
            "rows_changed": int((np.abs(cand - ctl)[m] > 1e-9).sum()),
            "ctl_rmse": float(np.sqrt(se_c[m].mean())),
            "cand_rmse": float(np.sqrt(((y - cand) ** 2)[m].mean())),
            "ctl_sse": float(se_c[m].sum()),
            "delta_sse": float(d[m].sum()),
            "delta_frac": float(d[m].sum() / se_c[m].sum()) if se_c[m].sum() > 0 else 0.0,
        }
    n_imp = sum(1 for v in folds.values() if v["delta_sse_subset"] < 0)
    worst = max(v["delta_frac"] for v in aps.values()) if aps else 0.0
    lirf_diff = float(np.max(np.abs(cand - ctl)[lirf_copy])) if lirf_copy.any() else 0.0
    changed = np.abs(cand - ctl) > 1e-9
    gates = {
        "P1_subset_boot_p95_lt_0": {"value": sub_boot["p95"], "pass": bool(sub_boot["p95"] < 0)},
        "P2_subset_folds_ge_3": {"value": n_imp, "pass": bool(n_imp >= 3)},
        "P3_subset_airport_cap_2pct": {"value": worst, "pass": bool(worst <= airport_cap)},
        "G1_nonobs_no_significant_harm": {"value": None if non_boot is None else non_boot["p05"], "changed_rows_outside_O": int((changed & ~subset).sum()), "pass": bool(non_boot is None or non_boot["p05"] <= 0.0)},
        "G2_lirf_copy_unchanged": {"value": lirf_diff, "n": int(lirf_copy.sum()), "pass": bool(lirf_diff == 0.0)},
    }
    return {
        "n": int(y.size),
        "n_subset": int(subset.sum()),
        "rows_changed": int(changed.sum()),
        "rows_changed_subset": int((changed & subset).sum()),
        "ctl_sse_full": float(se_c.sum()),
        "ctl_sse_subset": float(se_c[subset].sum()),
        "ctl_rmse_subset": float(np.sqrt(se_c[subset].mean())) if subset.any() else None,
        "cand_rmse_subset": float(np.sqrt(((y - cand) ** 2)[subset].mean())) if subset.any() else None,
        "delta_sse_full": float(d.sum()),
        "delta_sse_subset": float(d[subset].sum()),
        "delta_sse_outside": float(d[~subset].sum()),
        "sum_sq_change_subset": float(((cand - ctl) ** 2)[subset].sum()),
        "efficiency_subset": float(d[subset].sum() / ((cand - ctl) ** 2)[subset].sum()) if ((cand - ctl) ** 2)[subset].sum() > 0 else None,
        "top5_share_of_delta": float(np.sort(d)[:5].sum() / d.sum()) if d.sum() < 0 else None,
        "subset_bootstrap": sub_boot,
        "nonobs_bootstrap": non_boot,
        "folds": folds,
        "subset_airports": aps,
        "gates": gates,
        "go": all(g["pass"] for g in gates.values()),
        "failed": [k for k, g in gates.items() if not g["pass"]],
    }


# ---------------------------------------------------------------------------
# step 3: channels and arm rules (preregistration.json, section arms)
# ---------------------------------------------------------------------------

PHYSICAL_EXCLUDED = ("LIRF", "LFPG", "LTFM")
CHANNEL_CEILING = 3606.0
RESID_CLIP = 1800.0
W_GRID = tuple(round(0.05 * i, 2) for i in range(21))
TAU_GRID = (50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2400.0, 3600.0, float("inf"))


def physical_estimate(taxi_map: np.ndarray, taxi_gate: np.ndarray, airport: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P = map if present else confirmed gate; null at LIRF/LFPG/LTFM. Returns (P, p_src: 1 map, 2 gate, 0 none)."""

    m = np.asarray(taxi_map, dtype=np.float64)
    g = np.asarray(taxi_gate, dtype=np.float64)
    ok_ap = ~np.isin(np.asarray(airport).astype(str), PHYSICAL_EXCLUDED)
    p = np.where(np.isfinite(m), m, g)
    src = np.where(np.isfinite(m), 1, np.where(np.isfinite(g), 2, 0))
    p = np.where(ok_ap, p, np.nan)
    src = np.where(ok_ap & np.isfinite(p), src, 0)
    return p, src.astype(np.int8)


def cap_channel(v: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """min(max(min(v, 3606), 0), max(sd, 0) + 3606); NaN stays NaN (sd missing: no sd bound)."""

    v = np.asarray(v, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    out = np.clip(np.minimum(v, CHANNEL_CEILING), 0.0, None)
    bound = np.where(np.isfinite(sd), np.maximum(np.nan_to_num(sd, nan=0.0), 0.0) + CHANNEL_CEILING, np.inf)
    return np.where(np.isfinite(v), np.minimum(out, bound), np.nan)


def observable(fired: np.ndarray, p: np.ndarray, lirf_copy: np.ndarray) -> np.ndarray:
    return (np.asarray(fired, dtype=bool) | np.isfinite(np.asarray(p, dtype=np.float64))) & ~np.asarray(lirf_copy, dtype=bool)


def umap2_rows(p_src: np.ndarray, gate_conf: np.ndarray, lirf_copy: np.ndarray) -> np.ndarray:
    """U-MAP2 rows: map channel present (physical airport), no confirmed gate, not LIRF copy regime."""

    return (np.asarray(p_src) == 1) & ~(np.nan_to_num(np.asarray(gate_conf, dtype=np.float64), nan=0.0) == 1) & ~np.asarray(lirf_copy, dtype=bool)


def umap2_expert(e_prime: np.ndarray, m: np.ndarray, rows: np.ndarray, fired: np.ndarray, w_fired: float, w_not_fired: float) -> np.ndarray:
    e = np.asarray(e_prime, dtype=np.float64)
    w = np.where(np.asarray(fired, dtype=bool), w_fired, w_not_fired)
    use = np.asarray(rows, dtype=bool) & np.isfinite(m)
    return np.where(use, e + w * (np.nan_to_num(m) - e), e)


def sens2_rows(p: np.ndarray, lirf_copy: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(p, dtype=np.float64)) & ~np.asarray(lirf_copy, dtype=bool)


def lam_from_tau(tau: float, s2: np.ndarray) -> np.ndarray:
    s2 = np.clip(np.asarray(s2, dtype=np.float64), 0.0, None)
    if not np.isfinite(tau):
        return np.ones_like(s2)
    return tau**2 / (tau**2 + s2)


def physical_expert(e_prime: np.ndarray, e_d: np.ndarray, q: np.ndarray, rows: np.ndarray, lam: np.ndarray | float) -> np.ndarray:
    """E'' = lam Q + (1 - lam) E_D on ``rows`` (Q finite); elsewhere E' (the U-SENS-G blend)."""

    q = np.asarray(q, dtype=np.float64)
    use = np.asarray(rows, dtype=bool) & np.isfinite(q)
    lam = np.broadcast_to(np.asarray(lam, dtype=np.float64), q.shape)
    return np.where(use, lam * np.nan_to_num(q) + (1.0 - lam) * np.asarray(e_d, dtype=np.float64), np.asarray(e_prime, dtype=np.float64))


def crossfit_choice(grid, sse_of, fold: np.ndarray, folds: tuple[str, ...]) -> tuple[dict[str, Any], Any]:
    """Per fold f: argmin over ``grid`` of sse_of(value, mask = fold != f); plus the all-rows choice.

    ``sse_of(value, mask)`` returns the composed SSE of the arm rows inside ``mask``. Ties -> first grid value.
    """

    fold = np.asarray(fold, dtype=object)
    per_fold = {}
    for f in folds:
        mask = fold != f
        scores = [sse_of(v, mask) for v in grid]
        per_fold[f] = grid[int(np.argmin(scores))]
    all_scores = [sse_of(v, np.ones(fold.size, dtype=bool)) for v in grid]
    return per_fold, grid[int(np.argmin(all_scores))]


def change_cells(ctl: np.ndarray, p: np.ndarray) -> np.ndarray:
    """E2 cells of |control - P| (label-free)."""

    gap = np.abs(np.asarray(ctl, dtype=np.float64) - np.asarray(p, dtype=np.float64))
    return np.where(~np.isfinite(gap), "P_missing", np.where(gap < 300, "lt300", np.where(gap < 600, "300_600", np.where(gap < 900, "600_900", "ge900"))))
