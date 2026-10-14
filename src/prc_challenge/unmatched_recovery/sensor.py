# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-UNM3 (2026-09-26): ADS-B sensor expert and matched-neighbour regime features for unmatched rows.

Pre-registration: ``progress-review/unmatched-sensor-20260926/preregistration.json``.

Helpers here are label-free except the scoring functions, which receive ``y`` explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

CAP_SEC = 3606.0
W_GATE = 0.75  # adsb_coverage == 3 (stand gate)
W_GROUND = 0.5  # adsb_coverage == 2 (ground block, no gate)
REGIME_X_MIN, REGIME_X_MAX = -600.0, 14400.0
REGIME_COLS = ("rg_n30", "rg_n60", "rg_med30", "rg_p75_30", "rg_med60", "rg_p75_60", "rg_typ_hour", "rg_ratio60")
SENS_CATS = ["AIRPORT", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]
SENS_NUMS_U6 = ["local_hour", "takeoff_minus_sched_sec", "nm_unmatched", "type_missing"]
ADSB_DROP = ("adsb_aobt_rel", "adsb_aobt_minus_gate", "adsb_aobt_minus_first", "adsb_airport")
SENS_PARAMS = {"depth": 8, "learning_rate": 0.06, "l2_leaf_reg": 12.0, "iterations": 1000, "thread_count": 3}
GATE_AIRPORT_MAX_FRAC = 0.02
S_CEILING_G = 3606.0  # U-SENS-G (amendment 1)
N_BOOT = 2000


def adsb_numeric_features(columns: Sequence[str]) -> list[str]:
    return [c for c in columns if c.startswith("adsb_") and c not in ADSB_DROP]


# ---------------------------------------------------------------------------
# matched-neighbour regime features
# ---------------------------------------------------------------------------


def matched_departures(path: Path) -> pl.DataFrame:
    """NM-matched departures of one raw file: airport, takeoff time, x = MVT - AOBT_3 (s). No labels read."""

    frame = (
        pl.scan_parquet(path)
        .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_not_null() & pl.col("AOBT_3_flt").is_not_null())
        .select("MVT_ID_mvt", pl.col("ADEP_mvt").alias("AIRPORT"), "MVT_TIME_UTC_mvt", "AOBT_3_flt")
        .collect()
    )
    return frame.with_columns(
        ((pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64)).alias("x"),
        pl.col("MVT_TIME_UTC_mvt").dt.year().alias("year"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("cal_month"),
        pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("utc_hour"),
    ).filter(pl.col("x").is_between(REGIME_X_MIN, REGIME_X_MAX)).drop("AOBT_3_flt")


def regime_features(focal: pl.DataFrame, matched: pl.DataFrame) -> pl.DataFrame:
    """Per focal row (MVT_ID_mvt, AIRPORT, MVT_TIME_UTC_mvt): rg_* from matched departures of the same airport and calendar month."""

    focal = focal.select("MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt").with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.year().alias("year"), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("cal_month"), pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("utc_hour")
    )
    out_parts = []
    for (airport, year, month), fgrp in focal.group_by(["AIRPORT", "year", "cal_month"], maintain_order=True):
        mg = matched.filter((pl.col("AIRPORT") == airport) & (pl.col("year") == year) & (pl.col("cal_month") == month)).sort("MVT_TIME_UTC_mvt")
        n = fgrp.height
        res = {c: np.full(n, np.nan) for c in REGIME_COLS}
        if mg.height:
            t = mg["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
            x = mg["x"].to_numpy()
            ids = mg["MVT_ID_mvt"].to_numpy()
            hours = mg["utc_hour"].to_numpy()
            typ = {int(h): float(np.median(x[hours == h])) for h in np.unique(hours)}
            ft = fgrp["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
            fid = fgrp["MVT_ID_mvt"].to_numpy()
            fh = fgrp["utc_hour"].to_numpy()
            for i in range(n):
                for half, suffix in ((1800, "30"), (3600, "60")):
                    lo, hi = np.searchsorted(t, ft[i] - half, "left"), np.searchsorted(t, ft[i] + half, "right")
                    w = x[lo:hi][ids[lo:hi] != fid[i]]
                    res[f"rg_n{suffix}"][i] = w.size
                    if w.size:
                        res[f"rg_med{suffix}"][i] = np.median(w)
                        res[f"rg_p75_{suffix}"][i] = np.quantile(w, 0.75)
                res["rg_typ_hour"][i] = typ.get(int(fh[i]), np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                res["rg_ratio60"] = np.where(res["rg_typ_hour"] > 0, res["rg_med60"] / res["rg_typ_hour"], np.nan)
        out_parts.append(fgrp.select("MVT_ID_mvt").with_columns([pl.Series(c, res[c]) for c in REGIME_COLS]))
    out = pl.concat(out_parts)
    if out.height != focal.height or out["MVT_ID_mvt"].n_unique() != out.height:
        raise RuntimeError("regime features: row mismatch")
    return focal.select("MVT_ID_mvt").join(out, on="MVT_ID_mvt", how="left", maintain_order="left")


# ---------------------------------------------------------------------------
# compositions
# ---------------------------------------------------------------------------


def cap(pred: np.ndarray, sd: np.ndarray, cap_sec: float = CAP_SEC) -> np.ndarray:
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, None)
    sd = np.asarray(sd, dtype=np.float64)
    bound = np.where(np.isfinite(sd), np.maximum(np.nan_to_num(sd, nan=0.0), 0.0) + cap_sec, np.inf)
    return np.minimum(pred, bound)


def lirf_copy_regime(airport: np.ndarray, sd: np.ndarray, p: np.ndarray) -> np.ndarray:
    sd = np.asarray(sd, dtype=np.float64)
    return (np.asarray(airport) == "LIRF") & (~np.isfinite(sd) | (sd >= 3600.0) | (np.asarray(p, dtype=np.float64) >= 0.5))


def sensor_blend(e_cap: np.ndarray, s: np.ndarray, sd: np.ndarray, coverage: np.ndarray, airport: np.ndarray, p: np.ndarray, *, s_ceiling: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """E' = w*S_cap + (1-w)*E_cap on eligible rows; returns (E', eligible).

    ``s_ceiling`` (U-SENS-G, amendment 1): absolute ceiling on S before the blend.
    """

    s = np.asarray(s, dtype=np.float64)
    if s_ceiling is not None:
        s = np.where(np.isfinite(s), np.minimum(s, s_ceiling), s)
    cov = np.nan_to_num(np.asarray(coverage, dtype=np.float64), nan=-9.0)
    eligible = (cov >= 2) & np.isfinite(s) & ~lirf_copy_regime(airport, sd, p)
    w = np.where(cov >= 3, W_GATE, W_GROUND)
    s_cap = cap(np.nan_to_num(s), sd)
    return np.where(eligible, w * s_cap + (1.0 - w) * np.asarray(e_cap, dtype=np.float64), e_cap), eligible


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def delta_sse_bootstrap(y: np.ndarray, ctl: np.ndarray, cand: np.ndarray, day: Sequence[Any], *, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    d = (y - np.asarray(cand, dtype=np.float64)) ** 2 - (y - np.asarray(ctl, dtype=np.float64)) ** 2
    day = np.asarray(day, dtype=object)
    keys = sorted(set(day.tolist()), key=str)
    index = {k: i for i, k in enumerate(keys)}
    code = np.array([index[v] for v in day.tolist()])
    per_day = np.bincount(code, weights=d, minlength=len(keys))
    rng = np.random.default_rng(seed)
    boots = np.array([per_day[rng.integers(0, len(keys), len(keys))].sum() for _ in range(n_boot)])
    return {
        "delta_sse": float(d.sum()),
        "p05": float(np.quantile(boots, 0.05)),
        "p50": float(np.quantile(boots, 0.5)),
        "p95": float(np.quantile(boots, 0.95)),
        "p_delta_lt_0": float((boots < 0).mean()),
        "n_days": len(keys),
        "days_improved": int((per_day < 0).sum()),
        "days_worse": int((per_day > 0).sum()),
    }


def arm_gates(y, ctl, cand, *, day, fold, airport, lirf_copy: np.ndarray | None, reference: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    ctl = np.asarray(ctl, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    if not np.isfinite(cand).all():
        raise ValueError("non-finite candidate")
    se_c, se_k = (y - ctl) ** 2, (y - cand) ** 2
    boot = delta_sse_bootstrap(y, ctl, cand, day)
    fold = np.asarray(fold, dtype=object)
    airport = np.asarray(airport, dtype=object)
    folds = {str(f): {"n": int((fold == f).sum()), "delta_sse": float((se_k - se_c)[fold == f].sum()), "ctl_sse": float(se_c[fold == f].sum())} for f in sorted(set(fold.tolist()))}
    aps = {}
    for a in sorted(set(airport.tolist())):
        m = airport == a
        aps[str(a)] = {
            "n": int(m.sum()),
            "rows_changed": int((np.abs(cand - ctl)[m] > 1e-9).sum()),
            "ctl_rmse": float(np.sqrt(se_c[m].mean())),
            "cand_rmse": float(np.sqrt(se_k[m].mean())),
            "ctl_sse": float(se_c[m].sum()),
            "delta_sse": float((se_k - se_c)[m].sum()),
            "delta_frac": float((se_k - se_c)[m].sum() / se_c[m].sum()) if se_c[m].sum() > 0 else 0.0,
        }
    n_improving = sum(1 for v in folds.values() if v["delta_sse"] < 0)
    worst_ap = max(aps.values(), key=lambda v: v["delta_frac"])["delta_frac"]
    gates = {
        "G1_bootstrap_p95_lt_0": {"value": boot["p95"], "pass": bool(boot["p95"] < 0)},
        "G2_folds_ge_3": {"value": n_improving, "pass": bool(n_improving >= 3)},
        "G3_no_airport_gt_2pct": {"value": worst_ap, "pass": bool(worst_ap <= GATE_AIRPORT_MAX_FRAC)},
    }
    if lirf_copy is not None:
        base = ctl if reference is None else np.asarray(reference, dtype=np.float64)
        diff = float(np.max(np.abs(cand[lirf_copy] - base[lirf_copy]))) if lirf_copy.any() else 0.0
        gates["G4_lirf_copy_unchanged"] = {"value": diff, "n": int(lirf_copy.sum()), "pass": bool(diff == 0.0)}
    changed = np.abs(cand - ctl) > 1e-9
    return {
        "n": int(y.size),
        "rows_changed": int(changed.sum()),
        "ctl_rmse": float(np.sqrt(se_c.mean())),
        "cand_rmse": float(np.sqrt(se_k.mean())),
        "ctl_sse": float(se_c.sum()),
        "cand_sse": float(se_k.sum()),
        "delta_sse": float((se_k - se_c).sum()),
        "sse_ratio": float(se_k.sum() / se_c.sum()),
        "changed_rows_rmse": {"ctl": float(np.sqrt(se_c[changed].mean())) if changed.any() else None, "cand": float(np.sqrt(se_k[changed].mean())) if changed.any() else None},
        "bootstrap": boot,
        "folds": folds,
        "airports": aps,
        "gates": gates,
        "go": all(g["pass"] for g in gates.values()),
        "failed": [k for k, g in gates.items() if not g["pass"]],
    }
