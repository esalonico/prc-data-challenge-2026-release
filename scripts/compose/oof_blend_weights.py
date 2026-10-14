# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-BLEND (D-087): blend-weighting methods on the fourteen-leg V3L D-fold OOF bank, leave-one-fold-out.

Pre-registration: ``progress-review/blend-weights-20260928/preregistration.md`` (committed before the bank
was read for this decision).  Methods:

* ``a``  equal weights over the fourteen legs (the live rule; control);
* ``b``  a fixed family: V15 and V16 dropped; GBM legs 1, neural legs 1.5, trajectory leg 2 (normalised);
* ``c``  simplex weights solved in member-error space (min ||(P - y) w||^2 / (n * mean member MSE),
  w >= 0, sum w = 1) on the fit folds' matched rows with label <= 2700 s, then shrunk 50 % toward equal;
* ``d``  cell weights: airport group (ADS-B-rich seven / LFPG / LIRF / LTFM) x ADS-B coverage class
  (gate / segment / none / nofile); per cell with >= 20,000 fit rows the same solve, shrunk 50 % toward
  the method-c vector of the same fit folds; smaller cells use the method-c vector.

Leave-one-fold-out: for held-out fold k the fitted methods (c, d) are solved on the other three folds only
and applied to fold k.  Every method is scored through the full matched composition on the held-out rows:
blend (clip 0) -> T1 (``pred + 0.35 * p_long * (mu_long - pred)`` with SA-TAIL's fold-internal D-fold
p_long / mu_long) -> LOBT window -> clip 0 (``matched_tail.postprocess`` without U2).

Readouts: pooled / per-fold / per-month / per-airport matched RMSE (plain), the same 2026-mix reweighted
(``gate_stats.ranking_mix_weights``, as ``oof_gate.py``), a seed-noise floor (the seven neural legs'
seeds bootstrap-resampled, every method re-fitted per draw), and a day bootstrap of each Δ (diagnostic).

Usage::

  PYTHONPATH=src .venv/bin/python scripts/compose/oof_blend_weights.py --json results/blend/lofo.json [--draws 30]
  PYTHONPATH=src .venv/bin/python scripts/compose/oof_blend_weights.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prc_challenge import gate_stats as gs  # noqa: E402
from prc_challenge import matched_tail as mt  # noqa: E402

MAIN = Path(os.environ.get("PRC_MAIN_REPO", Path(__file__).resolve().parents[2]))
BANK = MAIN / "artifacts/experiments/Dfold-OOF-V3L-20260928"
RUNS = MAIN / "artifacts/experiments/S4-allyear-rebuild-20260925/runs"
DFIT_MATRIX = RUNS / "dfit_full_adsb_historder" / "train_matrix.parquet"
RANKING_MATRIX = RUNS / "allyear_full_adsb_final" / "eval_matrix.parquet"
T1_OOF = MAIN / "artifacts/experiments/Matched-tail-20260926/oof/d_lgbm.parquet"
BASE_CACHES = MAIN / "artifacts/experiments/S4-allyear-rebuild-20260925/features/base"
FOLDS = ("2_8", "3_9", "4_10", "5_6")

# the live fourteen-leg order (D-092 blend14v3l manifest)
LEGS14 = ["cb_plain", "lgbm_plain", "cb_delta", "lgbm_delta", "lgbm_eobtres", "cb_eobtres", "lgbm_lobtoff",
          "nn_v14", "nn_v15", "nn_v16", "plr_s03_resnet_f3", "plr_s03_eobtres_f3", "ctx_gru_plr_f2", "traj_conv_resnet_f3"]
GBM = LEGS14[:7]
NEURAL = LEGS14[7:]
TRAJ = "traj_conv_resnet_f3"
FAMILY_B = {**{k: 1.0 for k in GBM}, "nn_v14": 1.5, "nn_v15": 0.0, "nn_v16": 0.0, "plr_s03_resnet_f3": 1.5,
            "plr_s03_eobtres_f3": 1.5, "ctx_gru_plr_f2": 1.5, TRAJ: 2.0}
FIT_MAX_LABEL = mt.LONG_SEC  # 2700 s: rows T1 acts on cannot choose the weights
SHRINK = 0.5
MIN_CELL_ROWS = 20_000
ADSB_RICH = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LSZH")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- solver


def solve_simplex(E: np.ndarray, *, tol: float = 1e-7) -> tuple[np.ndarray, dict[str, Any]]:
    """min_w ||E w||^2 / n, w >= 0, sum w = 1, posed in member-error space and rescaled by the mean
    member MSE.  SLSQP from the equal start; guarded (success, nfev > 1, moved from the start) and
    certified by the KKT conditions of the simplex QP."""

    from scipy.optimize import minimize

    n, k = E.shape
    G = E.T @ E / n
    scale = float(np.mean(np.diag(G)))
    Gs = G / scale
    w0 = np.full(k, 1.0 / k)
    res = minimize(lambda w: float(w @ Gs @ w), w0, jac=lambda w: 2.0 * Gs @ w, method="SLSQP",
                   bounds=[(0.0, 1.0)] * k, constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0), "jac": lambda w: np.ones_like(w)}],
                   options={"ftol": 1e-14, "maxiter": 2000})
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    moved = float(np.linalg.norm(w - w0))
    g = Gs @ w  # KKT: g_i = lambda on the support, g_i >= lambda off it
    sup = w > 1e-6
    lam = float(np.mean(g[sup]))
    kkt = float(max(np.max(np.abs(g[sup] - lam)), max(0.0, float(np.max(lam - g[~sup]))) if (~sup).any() else 0.0))
    info = {"success": bool(res.success), "nfev": int(res.nfev), "nit": int(res.nit), "moved_from_equal": moved,
            "objective_equal": float(w0 @ Gs @ w0), "objective": float(w @ Gs @ w), "kkt_residual": kkt, "support": int(sup.sum()),
            "mean_member_mse": scale}
    if not res.success or res.nfev <= 1 or moved <= 1e-6:
        raise RuntimeError(f"solver did not move / failed: {info}")
    if kkt > max(tol, 1e-5 * abs(lam)):
        raise RuntimeError(f"KKT check failed: {info}")
    return w, info


def shrink(w: np.ndarray, toward: np.ndarray, s: float = SHRINK) -> np.ndarray:
    out = (1.0 - s) * np.asarray(w, np.float64) + s * np.asarray(toward, np.float64)
    return out / out.sum()


# --------------------------------------------------------------------------- data


def airport_group(airport: np.ndarray) -> np.ndarray:
    a = np.asarray(airport).astype(str)
    return np.where(np.isin(a, ADSB_RICH), "rich7", a).astype(object)


def cell_keys(airport: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(airport_group(airport).astype(str), "|"), gs.coverage_class(cov).astype(str)).astype(object)


class Data:
    def __init__(self, bank: Path, legs: list[str]):
        frames = []
        base = None
        for leg in legs:
            f = pl.read_parquet(bank / leg / "oof.parquet").select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col("month").cast(pl.Int8), "AIRPORT",
                                                                  pl.col("nm_unmatched").cast(pl.Int8), pl.col("label").cast(pl.Float64),
                                                                  pl.col("s4_raw_prediction").cast(pl.Float64).alias(leg), pl.col("fold").cast(pl.String))
            if base is None:
                base = f.drop(leg)
                frames.append(f.select(leg))
                continue
            j = base.select("MVT_ID_mvt", "label", "fold").join(f.select("MVT_ID_mvt", pl.col("label").alias("_l"), pl.col("fold").alias("_f"), leg),
                                                               on="MVT_ID_mvt", how="left", maintain_order="left", validate="1:1")
            if j[leg].null_count() or j.height != base.height or f.height != base.height:
                raise SystemExit(f"{leg}: rows differ from {legs[0]}")
            if not np.allclose(j["_l"].to_numpy(), j["label"].to_numpy()) or (j["_f"] != j["fold"]).any():
                raise SystemExit(f"{leg}: labels / folds differ from {legs[0]}")
            frames.append(j.select(leg))
        assert base is not None
        side = pl.read_parquet(DFIT_MATRIX, columns=["MVT_ID_mvt", "takeoff_minus_lobt_sec", "adsb_coverage"]).with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
        t1 = pl.read_parquet(T1_OOF, columns=["MVT_ID_mvt", "p_long", "mu_long", "fold"]).with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col("fold").alias("t1_fold"))
        s = base.select("MVT_ID_mvt", "fold").join(side, on="MVT_ID_mvt", how="left", maintain_order="left", validate="1:1") \
                .join(t1.drop("fold"), on="MVT_ID_mvt", how="left", maintain_order="left", validate="1:1")
        if s["p_long"].null_count() or s["takeoff_minus_lobt_sec"].dtype == pl.Null:
            raise SystemExit("T1 / side join incomplete")
        if (s["t1_fold"] != s["fold"].str.replace("_", "")).any():
            raise SystemExit("T1 fold keys differ from the bank folds")
        self.legs = legs
        self.ids = base["MVT_ID_mvt"].to_numpy()
        self.y = base["label"].to_numpy()
        self.month = base["month"].to_numpy()
        self.airport = base["AIRPORT"].to_numpy().astype(str)
        self.fold = base["fold"].to_numpy().astype(str)
        self.matched = base["nm_unmatched"].to_numpy() == 0
        self.P = np.column_stack([fr.to_series().to_numpy() for fr in frames]).astype(np.float64)
        self.cov = s["adsb_coverage"].to_numpy()
        self.cells = cell_keys(self.airport, self.cov)
        self.p_long = s["p_long"].to_numpy().astype(np.float64)
        self.mu_long = s["mu_long"].to_numpy().astype(np.float64)
        self.lead_lobt = s["takeoff_minus_lobt_sec"].cast(pl.Float64).to_numpy()
        if not self.matched.all():
            raise SystemExit("the D-fit bank should hold matched rows only")

    def compose(self, blend: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        r = slice(None) if rows is None else rows
        b = np.clip(blend, 0.0, None)
        m = self.matched[r]
        t1 = np.clip(mt.t1_correct(b, self.p_long[r], self.mu_long[r], m), 0.0, None)
        z = np.zeros(b.shape[0])
        return mt.postprocess(t1, lirf_u2=np.zeros(b.shape[0], bool), u2_p=z, u2_expert=z, sd=z, u2_feasible=np.zeros(b.shape[0], bool),
                              lead_lobt=self.lead_lobt[r], matched=m)


# --------------------------------------------------------------------------- methods


def vec(d: dict[str, float], legs: list[str]) -> np.ndarray:
    w = np.array([d[k] for k in legs], np.float64)
    return w / w.sum()


def fit_c(D: Data, P: np.ndarray, fit_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    r = fit_rows & (D.y <= FIT_MAX_LABEL)
    w_star, info = solve_simplex(P[r] - D.y[r, None])
    eq = np.full(P.shape[1], 1.0 / P.shape[1])
    info["n_fit"] = int(r.sum())
    return shrink(w_star, eq), w_star, info


def fit_d(D: Data, P: np.ndarray, fit_rows: np.ndarray, w_c: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    r = fit_rows & (D.y <= FIT_MAX_LABEL)
    out, info = {}, {}
    for cell in sorted(set(D.cells.tolist())):
        rc = r & (D.cells == cell)
        n = int(rc.sum())
        if n < MIN_CELL_ROWS:
            out[cell] = w_c.copy()
            info[cell] = {"n_fit": n, "fallback": "c"}
            continue
        w_star, si = solve_simplex(P[rc] - D.y[rc, None])
        out[cell] = shrink(w_star, w_c)
        info[cell] = {"n_fit": n, "w_star": w_star.round(5).tolist(), **{k: si[k] for k in ("nfev", "moved_from_equal", "kkt_residual", "support")}}
    return out, info


def cell_blend(D: Data, P: np.ndarray, rows: np.ndarray, wmap: dict[str, np.ndarray], w_default: np.ndarray) -> np.ndarray:
    out = np.empty(int(rows.sum()))
    cells = D.cells[rows]
    Pr = P[rows]
    for cell in np.unique(cells):
        s = cells == cell
        out[s] = Pr[s] @ wmap.get(cell, w_default)
    return out


def lofo(D: Data, P: np.ndarray, keep_info: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Composed held-out predictions of every method (each row predicted by weights not fitted on its fold)."""

    k = P.shape[1]
    w_a = np.full(k, 1.0 / k)
    w_b = vec(FAMILY_B, D.legs)
    preds = {m: np.empty(D.y.size) for m in "abcd"}
    info: dict[str, Any] = {"b": w_b.round(6).tolist(), "folds": {}}
    for f in FOLDS:
        hold = D.fold == f
        fit = ~hold
        w_c, w_star, ci = fit_c(D, P, fit)
        wmap, di = fit_d(D, P, fit, w_c)
        preds["a"][hold] = D.compose(P[hold] @ w_a, hold)
        preds["b"][hold] = D.compose(P[hold] @ w_b, hold)
        preds["c"][hold] = D.compose(P[hold] @ w_c, hold)
        preds["d"][hold] = D.compose(cell_blend(D, P, hold, wmap, w_c), hold)
        if keep_info:
            info["folds"][f] = {"c_w_star": w_star.round(6).tolist(), "c_w": w_c.round(6).tolist(), "c_solver": ci,
                                "d_cells": di, "d_w": {c: v.round(6).tolist() for c, v in wmap.items()}}
    return preds, info


# --------------------------------------------------------------------------- scoring


def score(D: Data, pred: np.ndarray, w_rw: np.ndarray) -> dict[str, Any]:
    e = (D.y - pred) ** 2
    out: dict[str, Any] = {"pooled": float(np.sqrt(e.mean())), "pooled_rw": gs.weighted_rmse(e, w_rw, np.ones_like(D.matched))}
    for f in FOLDS:
        s = D.fold == f
        out[f"fold_{f}"] = float(np.sqrt(e[s].mean()))
        out[f"sse_fold_{f}"] = float(e[s].sum())
        out[f"fold_{f}_rw"] = gs.weighted_rmse(e, w_rw, s)
    for m in sorted(set(D.month.tolist())):
        out[f"M{int(m):02d}"] = float(np.sqrt(e[D.month == m].mean()))
    out["airport"] = {a: float(np.sqrt(e[D.airport == a].mean())) for a in sorted(set(D.airport.tolist()))}
    out["class"] = {c: float(np.sqrt(e[gs.coverage_class(D.cov) == c].mean())) for c in ("gate", "segment", "none", "nofile")}
    return out


def compare(sc: dict[str, dict[str, Any]], ref: str = "a") -> dict[str, dict[str, Any]]:
    out = {}
    for m, c in sc.items():
        if m == ref:
            continue
        r = sc[ref]
        out[m] = {
            "d_pooled": c["pooled"] - r["pooled"],
            "d_pooled_rw": c["pooled_rw"] - r["pooled_rw"],
            "d_fold": {f: c[f"fold_{f}"] - r[f"fold_{f}"] for f in FOLDS},
            "d_fold_rw": {f: c[f"fold_{f}_rw"] - r[f"fold_{f}_rw"] for f in FOLDS},
            "folds_improved": int(sum(c[f"sse_fold_{f}"] < r[f"sse_fold_{f}"] for f in FOLDS)),
            "d_month": {k: c[k] - r[k] for k in c if k.startswith("M")},
            "d_airport": {a: c["airport"][a] - r["airport"][a] for a in c["airport"]},
            "d_class": {a: c["class"][a] - r["class"][a] for a in c["class"]},
        }
    return out


def seed_draw_matrix(D: Data, bank: Path, rng: np.random.Generator, seed_arrays: dict[str, list[np.ndarray]]) -> np.ndarray:
    P = D.P.copy()
    for leg, arrs in seed_arrays.items():
        pick = rng.integers(0, len(arrs), len(arrs))
        P[:, D.legs.index(leg)] = np.mean([arrs[i] for i in pick], axis=0)
    return P


def load_seeds(D: Data, bank: Path) -> dict[str, list[np.ndarray]]:
    q = pl.DataFrame({"MVT_ID_mvt": D.ids})
    out = {}
    for leg in NEURAL:
        files = sorted((bank / leg / "seeds").glob("oof_seed*.parquet"))
        if len(files) < 2:
            raise SystemExit(f"{leg}: {len(files)} seed OOF files")
        arrs = []
        for f in files:
            fr = pl.read_parquet(f, columns=["MVT_ID_mvt", "s4_raw_prediction"]).with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
            v = q.join(fr, on="MVT_ID_mvt", how="left", maintain_order="left")["s4_raw_prediction"]
            if v.null_count():
                raise SystemExit(f"{f}: rows missing")
            arrs.append(v.to_numpy().astype(np.float64))
        out[leg] = arrs
    return out


def decide(cmp: dict[str, Any], bar: float) -> dict[str, Any]:
    rules = {}
    for m, c in cmp.items():
        r = {"pooled_gain": c["d_pooled"] < 0, "folds_3of4": c["folds_improved"] >= 3, "reweighted_not_worse": c["d_pooled_rw"] <= 0,
             "beyond_seed_bar": c["d_pooled"] < -bar}
        r["pass"] = all(r.values())
        rules[m] = r
    passing = [m for m in ("b", "c", "d") if rules[m]["pass"]]
    winner = min(passing, key=lambda m: cmp[m]["d_pooled"]) if passing else None
    return {"rules": rules, "passing": passing, "winner": winner}


def final_weights(D: Data) -> dict[str, Any]:
    allr = np.ones(D.y.size, bool)
    w_c, w_star, ci = fit_c(D, D.P, allr)
    wmap, di = fit_d(D, D.P, allr, w_c)
    return {"b": dict(zip(D.legs, vec(FAMILY_B, D.legs).tolist())), "c": dict(zip(D.legs, w_c.tolist())), "c_w_star": dict(zip(D.legs, w_star.tolist())),
            "c_solver": ci, "d": {c: dict(zip(D.legs, v.tolist())) for c, v in wmap.items()}, "d_cells": di}


# --------------------------------------------------------------------------- self-test


def self_test() -> None:
    """Planted weights: legs = y + independent noise; the member-error optimum is w_i ∝ 1/σ_i²."""

    rng = np.random.default_rng(0)
    n = 400_000
    y = rng.gamma(4.0, 150.0, n)
    sig = np.array([60.0, 80.0, 120.0, 200.0, 400.0])
    P = y[:, None] + rng.normal(0, 1, (n, sig.size)) * sig
    w, info = solve_simplex(P - y[:, None])
    target = (1 / sig**2) / np.sum(1 / sig**2)
    assert np.max(np.abs(w - target)) < 0.01, (w, target)
    assert info["nfev"] > 1 and info["moved_from_equal"] > 0.1
    # a raw-scale objective near 1e6 is where SLSQP stops at nfev = 1; the rescaled one must still move
    P2 = P * 1000.0
    w2, _ = solve_simplex(P2 - 1000.0 * y[:, None])
    assert np.max(np.abs(w2 - target)) < 0.01
    # a dominated leg (3x leg 0's error) gets weight 0 (support < k) and the KKT check passes
    P3 = np.column_stack([P, y + 3.0 * (P[:, 0] - y)])
    w3, i3 = solve_simplex(P3 - y[:, None])
    assert w3[-1] < 1e-4 and i3["support"] == sig.size
    # shrink
    assert np.allclose(shrink(np.array([1.0, 0.0]), np.array([0.5, 0.5])), [0.75, 0.25])
    print(json.dumps({"self_test": "PASS", "w": w.round(4).tolist(), "target": target.round(4).tolist(), "nfev": info["nfev"]}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default=str(BANK))
    ap.add_argument("--draws", type=int, default=30, help="seed-floor redraws (0 = off)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="day-bootstrap replicates of each Δ (diagnostic)")
    ap.add_argument("--json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        self_test()
        return
    bank = Path(a.bank)
    D = Data(bank, LEGS14)
    log(f"bank {bank}: {D.y.size} rows, {len(D.legs)} legs, folds {sorted(set(D.fold.tolist()))}")
    cov = gs.coverage_class(D.cov)
    w_rw, rw_info = gs.ranking_mix_weights(gs.mix_cells(D.airport, D.matched, D.cov), gs.ranking_cells_from_matrix(RANKING_MATRIX))
    preds, info = lofo(D, D.P)
    sc = {m: score(D, p, w_rw) for m, p in preds.items()}
    cmp = compare(sc)
    for m in "abcd":
        c = sc[m]
        log(f"{m}: pooled {c['pooled']:.4f} rw {c['pooled_rw']:.4f} " + " ".join(f"{f} {c[f'fold_{f}']:.3f}" for f in FOLDS))
        if m in cmp:
            x = cmp[m]
            log(f"   d vs a: pooled {x['d_pooled']:+.4f} rw {x['d_pooled_rw']:+.4f} folds improved {x['folds_improved']}/4 " + " ".join(f"{f} {v:+.3f}" for f, v in x["d_fold"].items()))
    floor: dict[str, Any] = {}
    if a.draws:
        seeds = load_seeds(D, bank)
        rng = np.random.default_rng(20260928)
        scores = {m: [] for m in "abcd"}
        for i in range(a.draws):
            Pd = seed_draw_matrix(D, bank, rng, seeds)
            pd_, _ = lofo(D, Pd, keep_info=False)
            for m in "abcd":
                scores[m].append(float(np.sqrt(np.mean((D.y - pd_[m]) ** 2))))
            if (i + 1) % 5 == 0:
                log(f"seed draw {i + 1}/{a.draws}")
        s = {m: np.array(v) for m, v in scores.items()}
        sd_a = float(s["a"].std(ddof=1))
        floor = {"n_draws": a.draws, "seed_counts": {k: len(v) for k, v in seeds.items()}, "scores": {m: v.tolist() for m, v in s.items()},
                 "sd": {m: float(v.std(ddof=1)) for m, v in s.items()},
                 "sd_delta_vs_a": {m: float((s[m] - s["a"]).std(ddof=1)) for m in "bcd"},
                 "mean_delta_vs_a": {m: float((s[m] - s["a"]).mean()) for m in "bcd"},
                 "bar": 2.0 * math.sqrt(2.0) * sd_a}
        log(f"seed floor: sd(a) {sd_a:.4f} bar {floor['bar']:.4f}; sd(Δ) {floor['sd_delta_vs_a']}")
    bar = floor.get("bar", 0.0)
    dec = decide(cmp, bar)
    log(f"decision: {json.dumps(dec)}")
    boot = {}
    if a.bootstrap:
        day = gs.day_keys(D.ids, BASE_CACHES)
        for m in "bcd":
            boot[m] = {"plain": gs.day_bootstrap(D.y, preds["a"], preds[m], day, D.month, n_boot=a.bootstrap),
                       "reweighted": gs.day_bootstrap(D.y, preds["a"], preds[m], day, D.month, w=w_rw, n_boot=a.bootstrap)}
            log(f"bootstrap {m}: pooled dRMSE {boot[m]['plain']['pooled']['drmse']:+.4f} p05/p95 [{boot[m]['plain']['pooled']['drmse_p05']:+.4f}, {boot[m]['plain']['pooled']['drmse_p95']:+.4f}]")
    fw = final_weights(D)
    result = {"bank": str(bank), "legs": D.legs, "rows": int(D.y.size), "composition": "blend clip0 -> T1 (SA-TAIL D-fold p_long/mu_long, a=0.35) -> LOBT window -> clip0; no U2; no E17/E18 (not in the bank)",
              "reweight": {k: v for k, v in rw_info.items() if k != "cells"}, "class_shares": {c: float(np.mean(cov == c)) for c in ("gate", "segment", "none", "nofile")},
              "scores": sc, "delta_vs_a": cmp, "seed_floor": floor, "decision": dec, "day_bootstrap": boot, "lofo_weights": info, "final_weights": fw}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(result, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


if __name__ == "__main__":
    main()
