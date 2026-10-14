# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Leakage-safe OOF blend. Weights never see official V1/V2/V3."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prc_challenge.campaign.data import load_cache_columns, matched_train_val
from prc_challenge.campaign.runtime import CampaignConfig, is_complete
from prc_challenge.campaign.scoring import score_and_write
from prc_challenge.metrics import rmse

CPU_BLEND_IDS = (
    "CPU-01b-e34-shortlist",
    "CPU-02-airport-experts",
    "CPU-04-f3-ablation",
    "CPU-05-route-queue",
    "CPU-06-mixture",
    "CPU-09-lgb-xgb",
    "CPU-10-knn",
    "CPU-11-residual",
)
GPU_BLEND_IDS = (
    "GPU-01-transformer",
    "GPU-02-tcn",
    "GPU-03-gru",
    "GPU-04-fusion",
    "GPU-05-route-seq",
    "GPU-06-mixture",
    "GPU-07-faiss",
)


def _unique_pred_ids(frame: pl.DataFrame) -> pl.DataFrame:
    if "MVT_ID_mvt" in frame.columns:
        return frame.unique("MVT_ID_mvt", keep="first", maintain_order=True)
    return frame


def load_candidate_preds(cfg: CampaignConfig, exp_id: str, view: str) -> pl.DataFrame | None:
    dest = cfg.artifact_root / exp_id
    paths = [dest / f"predictions_{view}.parquet"]
    if view == "V1":
        paths.append(dest / "predictions.parquet")
    for path in paths:
        if path.exists():
            return _unique_pred_ids(pl.read_parquet(path))
    tag = None
    metrics_path = dest / "metrics.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        raw = payload.get("best_tag") or payload.get("best_model")
        if raw:
            tag = str(raw).replace("+", "_")
    if tag:
        tagged = dest / f"predictions_{tag}_{view}.parquet"
        if tagged.exists():
            return _unique_pred_ids(pl.read_parquet(tagged))
    return None


def load_weight_frame(cfg: CampaignConfig, exp_id: str) -> pl.DataFrame | None:
    """Train-only predictions for ensemble weights. Never official V1/V2/V3 val."""
    dest = cfg.artifact_root / exp_id
    for name in ("tune_predictions.parquet", "oof_predictions.parquet"):
        path = dest / name
        if not path.exists():
            continue
        frame = pl.read_parquet(path)
        if "prediction" in frame.columns and "TAXITIME_SEC_mvt" in frame.columns:
            return _unique_pred_ids(frame)
    return None


def simplex_weights(y: np.ndarray, mats: np.ndarray) -> tuple[tuple[float, ...], float]:
    """Non-negative weights summing to 1.

    A 5-point product grid is 5^k. CPU-13 already has 8 candidates and GPU-09
    adds sequence models, so overnight cannot enumerate 5^15 RMSE evaluations
    on the tune set. Keep the exact grid for small k; sample the simplex after that.
    """
    y = np.asarray(y, dtype=np.float64)
    mats = np.asarray(mats, dtype=np.float64)
    if y.ndim != 1:
        y = y.reshape(-1)
    if mats.ndim != 2 or mats.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: y={y.shape} mats={mats.shape}")
    keep_j = [j for j in range(mats.shape[1]) if np.isfinite(mats[:, j]).any()]
    if not keep_j:
        raise RuntimeError("no non-negative weights summing to 1")
    mats = mats[:, keep_j]
    row_ok = np.isfinite(y)
    for j in range(mats.shape[1]):
        row_ok &= np.isfinite(mats[:, j])
    if int(row_ok.sum()) < 1:
        raise RuntimeError("no non-negative weights summing to 1")
    y = y[row_ok]
    mats = mats[row_ok]
    k = int(mats.shape[1])
    if k < 1:
        raise RuntimeError("no non-negative weights summing to 1")
    if k == 1:
        return (1.0,), float(rmse(y, mats[:, 0]))

    candidates: list[np.ndarray] = [np.full(k, 1.0 / k, dtype=np.float64)]
    candidates.extend(np.eye(k, dtype=np.float64))
    if k <= 5:
        grid = np.linspace(0, 1, 5)
        for ws in itertools.product(grid, repeat=k):
            if abs(sum(ws) - 1.0) > 1e-8:
                continue
            candidates.append(np.asarray(ws, dtype=np.float64))
    else:
        rng = np.random.default_rng(42)
        candidates.extend(rng.dirichlet(np.ones(k), size=512))

    best_w, best_r = None, 1e18
    seen: set[tuple[float, ...]] = set()
    for w in candidates:
        key = tuple(round(float(x), 8) for x in w)
        if key in seen:
            continue
        seen.add(key)
        r = rmse(y, mats @ w)
        if np.isfinite(r) and r < best_r:
            best_r, best_w = r, tuple(float(x) for x in w)
    if best_w is None:
        raise RuntimeError("no non-negative weights summing to 1")
    vec = np.asarray(best_w, dtype=np.float64)
    step = 0.05
    for _ in range(12):
        improved = False
        for i in range(k):
            for delta in (-2 * step, -step, step, 2 * step):
                trial = vec.copy()
                trial[i] = max(0.0, trial[i] + delta)
                total = float(trial.sum())
                if total <= 1e-12:
                    continue
                trial = trial / total
                r = rmse(y, mats @ trial)
                if np.isfinite(r) and r + 1e-12 < best_r:
                    best_r = r
                    best_w = tuple(float(x) for x in trial)
                    vec = trial
                    improved = True
        if not improved:
            break
    return best_w, best_r


def _prepare_blend_matrix(
    wbase: pl.DataFrame, names: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Keep tune rows/columns with enough finite predictions for simplex search."""
    kept: list[str] = []
    cols: list[np.ndarray] = []
    for name in names:
        if name not in wbase.columns:
            continue
        arr = np.asarray(wbase[name].to_numpy(), dtype=np.float64)
        if int(np.isfinite(arr).sum()) < 20:
            continue
        kept.append(name)
        cols.append(arr)
    if len(kept) < 2:
        return None
    y = np.asarray(wbase["TAXITIME_SEC_mvt"].to_numpy(), dtype=np.float64)
    mat = np.column_stack(cols)
    mask = np.isfinite(y)
    for j in range(mat.shape[1]):
        mask &= np.isfinite(mat[:, j])
    if int(mask.sum()) < 20:
        return None
    return y[mask], mat[mask], kept


TOO_FEW_BLEND_MEMBERS = "fewer than two blend members have tune/OOF and val predictions"


def _job_payload(cfg: CampaignConfig, exp_id: str) -> dict[str, Any] | None:
    path = cfg.artifact_root / exp_id / "metrics.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _pending_blend_ids(cfg: CampaignConfig, candidate_ids: tuple[str, ...]) -> list[str]:
    """Wait for jobs that can still finish. Failed members are dropped, not blocked on."""
    pending: list[str] = []
    for exp_id in candidate_ids:
        if is_complete(cfg, exp_id):
            continue
        payload = _job_payload(cfg, exp_id)
        if payload is not None and payload.get("status") == "failed":
            continue
        pending.append(exp_id)
    return pending


def blend_completed_candidates(
    cfg: CampaignConfig,
    dest: Path,
    *,
    view: str,
    candidate_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Blend val predictions with weights fit on train-only tune/OOF files."""
    from prc_challenge.campaign.data import load_base_preds, load_parent

    if not cfg.smoke:
        pending = _pending_blend_ids(cfg, candidate_ids)
        if pending:
            return {
                "status": "skipped",
                "reason": "need remaining blend candidates: " + ", ".join(pending),
                "conclusion": "PARK",
                "pending_ids": pending,
            }

    weight_frames: list[tuple[str, pl.DataFrame]] = []
    for exp_id in candidate_ids:
        weights = load_weight_frame(cfg, exp_id)
        vals = load_candidate_preds(cfg, exp_id, view)
        if weights is None or vals is None or "prediction" not in vals.columns:
            continue
        weight_frames.append((exp_id, weights.rename({"prediction": exp_id})))
    if len(weight_frames) < 2:
        return {
            "status": "skipped",
            "reason": TOO_FEW_BLEND_MEMBERS,
            "conclusion": "PARK",
            "candidates_seen": [n for n, _ in weight_frames],
        }
    names = [n for n, _ in weight_frames]
    wbase = weight_frames[0][1].select(["MVT_ID_mvt", "TAXITIME_SEC_mvt", names[0]])
    for name, frame in weight_frames[1:]:
        wbase = wbase.join(frame.select(["MVT_ID_mvt", name]), on="MVT_ID_mvt", how="inner")
    prepared = _prepare_blend_matrix(wbase, names)
    if prepared is None:
        return {
            "status": "skipped",
            "reason": TOO_FEW_BLEND_MEMBERS,
            "conclusion": "PARK",
            "candidates_seen": names,
        }
    y, wmat, names = prepared
    best_w, _best_r = simplex_weights(y, wmat)
    w_map = dict(zip(names, [float(w) for w in best_w]))
    view_metrics: dict[str, Any] = {}
    cache = load_cache_columns(cfg)
    for score_view in cfg.views:
        view_vals: list[tuple[str, pl.DataFrame]] = []
        for name in names:
            frame = load_candidate_preds(cfg, name, score_view)
            if frame is None or "prediction" not in frame.columns:
                continue
            view_vals.append((name, frame.rename({"prediction": name})))
        if len(view_vals) < 2:
            view_metrics[score_view] = {
                "status": "skipped",
                "reason": "fewer than two members have predictions on this view",
                "n_present": len(view_vals),
            }
            continue
        present = [n for n, _ in view_vals]
        w_sub = np.array([w_map[n] for n in present], dtype=np.float64)
        if float(w_sub.sum()) <= 1e-12:
            w_sub = np.full(len(present), 1.0 / len(present))
        else:
            w_sub = w_sub / w_sub.sum()
        vbase = view_vals[0][1]
        keep_val = [c for c in ("MVT_ID_mvt", "AIRPORT", "TAXITIME_SEC_mvt", present[0]) if c in vbase.columns]
        vbase = vbase.select(keep_val)
        for name, frame in view_vals[1:]:
            vbase = vbase.join(frame.select(["MVT_ID_mvt", name]), on="MVT_ID_mvt")
        vmat = np.column_stack([vbase[n].to_numpy() for n in present])
        pred = vmat @ w_sub
        _, val = matched_train_val(cache, score_view)
        blended = vbase.select(["MVT_ID_mvt"]).with_columns(pl.Series("blend", pred))
        val = val.join(blended, on="MVT_ID_mvt", how="left")
        if val["blend"].null_count():
            view_metrics[score_view] = {
                "status": "skipped",
                "reason": "blend IDs do not cover val",
                "n_present": len(present),
                "n_null": int(val["blend"].null_count()),
            }
            continue
        parent = load_parent(cfg, score_view)
        try:
            base = load_base_preds(cfg, score_view)
        except FileNotFoundError:
            base = None
        scored_view = score_and_write(
            val, val["blend"].to_numpy(), dest, view=score_view, parent=parent, base=base
        )
        scored_view["blend_members"] = present
        scored_view["renormalized_weights"] = dict(zip(present, [float(x) for x in w_sub]))
        view_metrics[score_view] = scored_view
    composed = view_metrics.get(view) or next(
        (v for v in view_metrics.values() if isinstance(v, dict) and v.get("rmse_seconds") is not None),
        {},
    )
    if not composed or composed.get("status") == "skipped":
        return {
            "status": "skipped",
            "reason": TOO_FEW_BLEND_MEMBERS,
            "conclusion": "PARK",
            "candidates_seen": names,
            "views": view_metrics,
        }
    return {
        "status": "completed",
        "rmse_seconds": composed.get("rmse_seconds"),
        "weights": dict(zip(names, [float(w) for w in best_w])),
        "weight_fit_n": int(y.shape[0]),
        "delta_sse_vs_base": (composed.get("vs_base") or {}).get("delta_sse_vs_base"),
        "sse_cells": composed.get("sse_cells"),
        "views": {
            vw: {k: part.get(k) for k in ("rmse_seconds", "sse", "vs_base", "status", "reason") if k in part}
            for vw, part in view_metrics.items()
        },
        "conclusion": "ENSEMBLE_CANDIDATE"
        if (composed.get("vs_base") or {}).get("delta_sse_vs_base", 0) > 0
        else "PARK",
        "hypothesis": "complementary OOF models improve composed RMSE",
        "candidate_ids": list(names),
    }
