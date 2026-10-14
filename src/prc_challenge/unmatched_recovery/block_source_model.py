# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Blocked-D source-class model, mixture, hard selection, and freeze gates.

January/July labels are not read here. A candidate is eligible for one frozen
confirmation only after the predeclared D gates pass.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prc_challenge.metrics import rmse
from prc_challenge.unmatched_recovery.block_lineage import (
    CLASSES,
    CORE_CATS,
    DAY_SCALE_SEC,
    EXTREME_SEC,
    FEATURE_FORBIDDEN,
    LONG_SEC,
    TOL_SEC,
    assert_feature_names,
    class_accuracy,
    feature_names,
    hierarchical_probabilities,
    multiclass_brier,
    multiclass_log_loss,
    null_log_loss,
)
from prc_challenge.unmatched_recovery.domain_shift import calibration_slope_intercept
from prc_challenge.unmatched_recovery.schema import BLOCKS

SSE_REDUCTION_MIN = 0.05
MIN_IMPROVING_BLOCKS = 3
MAX_BLOCK_RMSE_WORSEN = 100.0
PRECISION_MIN = 0.90
TOP_SHARE_MAX = 0.50
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)
CALIBRATION_SLOPE = (0.40, 1.60)
MIN_PRECISION_SUPPORT = 20


def population_mask(y: np.ndarray, source: np.ndarray, kind: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    source = np.asarray(source)
    if kind == "unmatched":
        return np.ones(y.shape[0], dtype=bool)
    if kind == "extreme":
        return y >= EXTREME_SEC
    if kind == "ordinary":
        return y < LONG_SEC
    if kind == "mid_3600_7200":
        return (y >= LONG_SEC) & (y < EXTREME_SEC)
    if kind == "tail_7200":
        return y >= EXTREME_SEC
    if kind == "schedule_copy":
        return source == "schedule_copy"
    if kind == "day_scale_non_copy":
        return (y >= DAY_SCALE_SEC) & (source != "schedule_copy")
    raise ValueError(kind)


def score_pair(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if y.size == 0:
        return {"n": 0, "rmse": float("nan"), "baseline_rmse": float("nan"), "sse": float("nan"), "baseline_sse": float("nan"), "sse_ratio": float("nan")}
    sse = float(np.sum((y - pred) ** 2))
    base_sse = float(np.sum((y - baseline) ** 2))
    return {
        "n": int(y.size),
        "rmse": float(rmse(y, pred)),
        "baseline_rmse": float(rmse(y, baseline)),
        "sse": sse,
        "baseline_sse": base_sse,
        "sse_ratio": sse / base_sse if base_sse > 0 else float("nan"),
    }


def binary_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y, dtype=np.int32)
    scores = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    out = {
        "n": int(labels.size),
        "observed_prevalence": float(labels.mean()) if labels.size else float("nan"),
        "predicted_prevalence": float(scores.mean()) if labels.size else float("nan"),
    }
    if labels.size == 0 or labels.min() == labels.max():
        out.update({"roc_auc": float("nan"), "brier": float("nan"), "log_loss": float("nan"), "slope": float("nan"), "intercept": float("nan")})
        return out
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    out["roc_auc"] = float(roc_auc_score(labels, scores))
    out["brier"] = float(brier_score_loss(labels, scores))
    out["log_loss"] = float(log_loss(labels, np.column_stack([1 - scores, scores])))
    out.update(calibration_slope_intercept(labels.astype(np.float64), scores))
    return out


def precision_at(y: np.ndarray, p: np.ndarray, threshold: float, *, large_pred: np.ndarray | None = None) -> dict[str, float]:
    pick = np.asarray(p, dtype=np.float64) >= threshold
    if large_pred is not None:
        pick = pick & np.asarray(large_pred, dtype=bool)
    support = int(pick.sum())
    if support == 0:
        return {"threshold": threshold, "support": 0, "precision": float("nan")}
    return {"threshold": threshold, "support": support, "precision": float(np.asarray(y)[pick].mean())}


def choose_threshold(y: np.ndarray, p: np.ndarray, large_taxi: np.ndarray) -> float | None:
    """Smallest predeclared threshold with >=90% precision on selected and on large selected rows."""

    chosen = None
    for threshold in THRESHOLDS:
        overall = precision_at(y, p, threshold)
        large = precision_at(y, p, threshold, large_pred=large_taxi)
        overall_ok = overall["support"] >= MIN_PRECISION_SUPPORT and overall["precision"] >= PRECISION_MIN
        large_ok = large["support"] < MIN_PRECISION_SUPPORT or (np.isfinite(large["precision"]) and large["precision"] >= PRECISION_MIN)
        if overall_ok and large_ok:
            chosen = threshold
            break
    return chosen


def _matrix(frame: pl.DataFrame, cats: Sequence[str], nums: Sequence[str], encoder: OneHotEncoder | None, *, fit: bool) -> tuple[np.ndarray, OneHotEncoder]:
    assert_feature_names(list(cats) + list(nums))
    num = np.column_stack(
        [frame[name].cast(pl.Float64).fill_null(0.0).to_numpy() if name in frame.columns else np.zeros(frame.height) for name in nums]
    )
    cat_frame = frame.select([pl.col(name).cast(pl.Utf8).fill_null("__UNK__") if name in frame.columns else pl.lit("__UNK__").alias(name) for name in cats])
    values = cat_frame.to_numpy()
    if fit:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(values)
    else:
        if encoder is None:
            raise RuntimeError("encoder missing")
        encoded = encoder.transform(values)
    return np.hstack([num, encoded]), encoder


def _align_proba(model: LogisticRegression, proba: np.ndarray, n: int) -> np.ndarray:
    aligned = np.zeros((n, len(CLASSES)), dtype=np.float64)
    class_to_col = {str(label): i for i, label in enumerate(model.classes_)}
    for i, name in enumerate(CLASSES):
        src = class_to_col.get(name)
        if src is not None:
            aligned[:, i] = proba[:, src]
    row_sum = aligned.sum(axis=1, keepdims=True)
    missing = row_sum[:, 0] <= 0
    aligned[missing, CLASSES.index("actual_like_unexplained")] = 1.0
    row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / row_sum


def fit_source_logistic(frame: pl.DataFrame, kind: str) -> tuple[LogisticRegression, OneHotEncoder, StandardScaler, list[str], list[str]]:
    cats, nums = feature_names(kind)
    model_frame = frame.filter(pl.col("source_class").is_not_null())
    labels = model_frame["source_class"].to_numpy()
    present = [name for name in CLASSES if np.any(labels == name)]
    if len(present) < 2:
        raise RuntimeError("source logistic needs two classes")
    x, encoder = _matrix(model_frame, cats, nums, None, fit=True)
    scaler = StandardScaler()
    x[:, : len(nums)] = scaler.fit_transform(x[:, : len(nums)])
    model = LogisticRegression(solver="lbfgs", max_iter=200, C=1.0)
    model.fit(x, labels)
    return model, encoder, scaler, cats, nums


def predict_source_proba(
    model: LogisticRegression,
    encoder: OneHotEncoder,
    frame: pl.DataFrame,
    cats: Sequence[str],
    nums: Sequence[str],
    scaler: StandardScaler,
) -> np.ndarray:
    x, _ = _matrix(frame, cats, nums, encoder, fit=False)
    x[:, : len(nums)] = scaler.transform(x[:, : len(nums)])
    return _align_proba(model, model.predict_proba(x), frame.height)


def with_family_features(hold: pl.DataFrame, fit: pl.DataFrame, *, n0: float, exclude_self: bool) -> pl.DataFrame:
    return hierarchical_probabilities(hold, fit, n0=n0, exclude_self=exclude_self)


def candidate_taxi(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Inference-visible taxi candidates. Day-wrap sign is not identifiable without BLOCK."""

    n = frame.height
    def col(name: str) -> np.ndarray:
        if name not in frame.columns:
            return np.full(n, np.nan)
        return frame[name].to_numpy().astype(np.float64)

    sched = col("takeoff_minus_sched_sec")
    iobt = col("takeoff_minus_iobt_sec")
    eobt = col("takeoff_minus_eobt_sec")
    lobt = col("takeoff_minus_lobt_sec")
    u6 = col("u6_prediction")
    relation = np.full(n, np.nan)
    if "dest_arr_unique" in frame.columns and "taxi_relation" in frame.columns:
        unique = frame["dest_arr_unique"].fill_null(0).to_numpy() == 1
        relation = np.where(unique, col("taxi_relation"), np.nan)
    return {
        "schedule_copy": sched,
        "iobt_copy": iobt,
        "eobt_copy": eobt,
        "lobt_copy": lobt,
        "relation_copy": relation,
        "day_wrap_or_stale": u6,
        "exact_offset": u6,
        "actual_like_unexplained": u6,
        "ambiguous_multiple_candidates": u6,
        "no_candidate": u6,
    }


def mixture_prediction(proba: np.ndarray, taxis: dict[str, np.ndarray], u6: np.ndarray) -> np.ndarray:
    """Complete mixture. Missing candidates donate their probability to U6."""

    n = proba.shape[0]
    pred = np.zeros(n, dtype=np.float64)
    for i, name in enumerate(CLASSES):
        values = taxis.get(name)
        if values is None:
            values = u6
        use = np.isfinite(values)
        pred = pred + np.where(use, proba[:, i] * values, 0.0)
        pred = pred + np.where(use, 0.0, proba[:, i] * u6)
    return pred


def hard_prediction(proba: np.ndarray, taxis: dict[str, np.ndarray], u6: np.ndarray, threshold: float | None) -> tuple[np.ndarray, np.ndarray]:
    pred = u6.copy()
    selected = np.zeros(u6.shape[0], dtype=bool)
    if threshold is None:
        return pred, selected
    sched_p = proba[:, CLASSES.index("schedule_copy")]
    sched = taxis["schedule_copy"]
    pick = (sched_p >= threshold) & np.isfinite(sched)
    pred = np.where(pick, sched, u6)
    return pred, pick


def top_share(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray, groups: np.ndarray | None = None) -> float:
    delta = (y - baseline) ** 2 - (y - pred) ** 2
    total = float(delta.sum())
    if total <= 0:
        return float("nan")
    if groups is None:
        return float(np.max(delta) / total)
    frame = pl.DataFrame({"g": groups, "delta": delta})
    grouped = frame.group_by("g").agg(pl.col("delta").sum())["delta"].to_numpy()
    return float(np.max(grouped) / total)


def gate_decision(block_rows: list[dict[str, Any]], *, pooled: dict[str, float], row_share: float, family_share: float, precision: float | None, mixture: bool) -> dict[str, Any]:
    reasons: list[str] = []
    sse_ratio = float(pooled.get("sse_ratio", float("nan")))
    if not np.isfinite(sse_ratio) or sse_ratio > 1.0 - SSE_REDUCTION_MIN:
        reasons.append("sse_reduction_below_5pct")
    improving = [row for row in block_rows if np.isfinite(row.get("sse_ratio", float("nan"))) and row["sse_ratio"] < 1.0]
    if len(improving) < MIN_IMPROVING_BLOCKS:
        reasons.append("fewer_than_3_blocks_improve")
    for row in block_rows:
        if np.isfinite(row.get("rmse", float("nan"))) and np.isfinite(row.get("baseline_rmse", float("nan"))):
            if row["rmse"] - row["baseline_rmse"] > MAX_BLOCK_RMSE_WORSEN:
                reasons.append("block_rmse_worsens_over_100s")
                break
    tail = pooled.get("tail_rmse")
    tail_base = pooled.get("tail_baseline_rmse")
    if not (np.isfinite(tail) and np.isfinite(tail_base) and tail <= tail_base + 1e-6):
        reasons.append("tail_7200_not_preserved")
    if mixture:
        if precision is not None and precision < PRECISION_MIN:
            reasons.append("mixture_marked_with_hard_precision")
    else:
        if precision is None or not np.isfinite(precision) or precision < PRECISION_MIN:
            reasons.append("hard_precision_below_90")
    if np.isfinite(row_share) and row_share > TOP_SHARE_MAX:
        reasons.append("one_row_explains_gain")
    if np.isfinite(family_share) and family_share > TOP_SHARE_MAX:
        reasons.append("one_family_explains_gain")
    return {"eligible": len(reasons) == 0, "reasons": reasons}


def _probs_from_frame(frame: pl.DataFrame, prefix: str = "p_") -> np.ndarray:
    return np.column_stack([frame[f"{prefix}{name}"].to_numpy() for name in CLASSES])


def attach_probability_columns(frame: pl.DataFrame, proba: np.ndarray, prefix: str = "p_") -> pl.DataFrame:
    return frame.with_columns([pl.Series(f"{prefix}{name}", proba[:, i]) for i, name in enumerate(CLASSES)])


def blocked_source_oof(frame: pl.DataFrame, *, kind: str, n0: float) -> pl.DataFrame:
    """Out-of-fold source probabilities. Shrinkage counts exclude the holdout months."""

    parts: list[pl.DataFrame] = []
    for holdout in BLOCKS:
        fit = frame.filter(~pl.col("month").is_in(list(holdout)))
        hold = frame.filter(pl.col("month").is_in(list(holdout)))
        if fit.height == 0 or hold.height == 0:
            continue
        fit_x = with_family_features(fit, fit, n0=n0, exclude_self=True)
        hold_x = with_family_features(hold, fit, n0=n0, exclude_self=False)
        model, encoder, scaler, cats, nums = fit_source_logistic(fit_x, kind)
        proba = predict_source_proba(model, encoder, hold_x, cats, nums, scaler)
        parts.append(attach_probability_columns(hold_x, proba))
    if not parts:
        raise RuntimeError("blocked source OOF produced no rows")
    return pl.concat(parts, how="vertical_relaxed")


def schedule_probability(frame: pl.DataFrame) -> np.ndarray:
    return frame["p_schedule_copy"].to_numpy().astype(np.float64)


def evaluate_predictions(frame: pl.DataFrame, pred: np.ndarray, *, model: str) -> list[dict[str, Any]]:
    y = frame["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    u6 = frame["u6_prediction"].to_numpy().astype(np.float64)
    source = frame["source_class"].to_numpy()
    airport = frame["AIRPORT"].to_numpy()
    rows: list[dict[str, Any]] = []
    slices = (
        ("pooled", np.ones(y.shape[0], dtype=bool), None),
        ("ordinary", population_mask(y, source, "ordinary"), None),
        ("mid_3600_7200", population_mask(y, source, "mid_3600_7200"), None),
        ("tail_7200", population_mask(y, source, "tail_7200"), None),
        ("schedule_copy", population_mask(y, source, "schedule_copy"), None),
        ("day_scale_non_copy", population_mask(y, source, "day_scale_non_copy"), None),
    )
    for name, mask, _ in slices:
        stats = score_pair(y[mask], pred[mask], u6[mask])
        rows.append({"model": model, "slice": name, "block": "pooled", "airport": "all", **stats})
    for holdout in BLOCKS:
        mask = np.isin(frame["month"].to_numpy(), list(holdout))
        stats = score_pair(y[mask], pred[mask], u6[mask])
        rows.append({"model": model, "slice": "unmatched", "block": ",".join(map(str, holdout)), "airport": "all", **stats})
        tail = mask & (y >= EXTREME_SEC)
        tail_stats = score_pair(y[tail], pred[tail], u6[tail])
        rows.append({"model": model, "slice": "tail_7200", "block": ",".join(map(str, holdout)), "airport": "all", **tail_stats})
    for airport_name in sorted(set(airport.tolist())):
        mask = airport == airport_name
        stats = score_pair(y[mask], pred[mask], u6[mask])
        rows.append({"model": model, "slice": "unmatched", "block": "pooled", "airport": airport_name, **stats})
    return rows


def _masked(frame: pl.DataFrame, mask: np.ndarray) -> pl.DataFrame:
    return frame.with_columns(pl.Series("_keep", np.asarray(mask, dtype=bool))).filter(pl.col("_keep")).drop("_keep")


def source_metric_rows(frame: pl.DataFrame, *, model: str) -> list[dict[str, Any]]:
    y = (frame["source_class"] == "schedule_copy").to_numpy().astype(np.int32)
    p = schedule_probability(frame)
    rows = []
    populations = {
        "unmatched": np.ones(frame.height, dtype=bool),
        "extreme": frame["TAXITIME_SEC_mvt"].to_numpy() >= EXTREME_SEC,
        "lirf": frame["AIRPORT"].to_numpy() == "LIRF",
        "schedule_copy": frame["source_class"].to_numpy() == "schedule_copy",
        "ordinary": frame["TAXITIME_SEC_mvt"].to_numpy() < LONG_SEC,
    }
    for name, mask in populations.items():
        part = _masked(frame, mask)
        metrics = binary_scores(y[mask], p[mask])
        metrics.update(
            {
                "model": model,
                "population": name,
                "multiclass_log_loss": multiclass_log_loss(part, prefix="p_") if part.height else float("nan"),
                "null_log_loss": null_log_loss(frame, part) if part.height else float("nan"),
                "multiclass_brier": multiclass_brier(part, prefix="p_") if part.height else float("nan"),
                "accuracy": class_accuracy(part, prefix="p_") if part.height else float("nan"),
            }
        )
        rows.append(metrics)
    return rows


def confusion_rows(frame: pl.DataFrame, *, model: str) -> list[dict[str, Any]]:
    proba = _probs_from_frame(frame)
    pred = np.array(CLASSES, dtype=object)[np.argmax(proba, axis=1)]
    truth = frame["source_class"].to_numpy()
    rows = []
    for true_name in CLASSES:
        for pred_name in CLASSES:
            n = int(np.sum((truth == true_name) & (pred == pred_name)))
            if n:
                rows.append({"model": model, "truth": true_name, "predicted": pred_name, "n": n})
    return rows


def build_taxi_relation(frame: pl.DataFrame) -> pl.DataFrame:
    """Unique destination-arrival schedule as a BLOCK stand-in, without reading focal BLOCK."""

    if "arr_sched_epoch_unique" not in frame.columns:
        return frame.with_columns(pl.lit(None).cast(pl.Float64).alias("taxi_relation"))
    return frame.with_columns((pl.col("mvt_epoch") - pl.col("arr_sched_epoch_unique")).alias("taxi_relation"))


def select_n0_from_family_rows(rows: list[dict[str, Any]]) -> float:
    grid = [row for row in rows if row.get("holdout") == "grid" and row.get("population") == "unmatched"]
    if not grid:
        return 20.0
    best = min(grid, key=lambda row: row["log_loss"] if np.isfinite(row["log_loss"]) else np.inf)
    return float(best["n0"])


def run_blocked_candidates(frame: pl.DataFrame, *, n0: float) -> dict[str, Any]:
    """Fit the two prespecified predictors on blocked D. No January/July rows."""

    months = frame["month"].unique().to_list()
    if set(int(m) for m in months) - set(sum(BLOCKS, ())):
        raise RuntimeError(f"candidate fit saw months outside the blocked D scheme: {months}")
    oof = {}
    losses = {}
    for kind in ("core", "delay"):
        pred = blocked_source_oof(frame, kind=kind, n0=n0)
        losses[kind] = multiclass_log_loss(pred, prefix="p_")
        oof[kind] = pred
    chosen_kind = min(losses, key=losses.get)
    chosen = oof[chosen_kind]
    taxis = candidate_taxi(chosen)
    u6 = chosen["u6_prediction"].to_numpy().astype(np.float64)
    proba = _probs_from_frame(chosen)
    y_sched = (chosen["source_class"] == "schedule_copy").to_numpy().astype(np.int32)
    p_sched = proba[:, CLASSES.index("schedule_copy")]
    large = np.isfinite(taxis["schedule_copy"]) & (taxis["schedule_copy"] >= EXTREME_SEC)
    # Nested threshold: each holdout uses a threshold fit on the other blocks only.
    hard = u6.copy()
    hard_pick = np.zeros(chosen.height, dtype=bool)
    month = chosen["month"].to_numpy()
    for holdout in BLOCKS:
        test = np.isin(month, list(holdout))
        train = ~test
        threshold = choose_threshold(y_sched[train], p_sched[train], large[train])
        block_pred, block_pick = hard_prediction(proba[test], {k: v[test] for k, v in taxis.items()}, u6[test], threshold)
        hard[test] = block_pred
        hard_pick[test] = block_pick
    mix = mixture_prediction(proba, taxis, u6)
    slope = binary_scores(y_sched, p_sched).get("slope", float("nan"))
    calibrated = bool(np.isfinite(slope) and CALIBRATION_SLOPE[0] <= slope <= CALIBRATION_SLOPE[1])
    picked_precision = float(y_sched[hard_pick].mean()) if hard_pick.any() else float("nan")
    row_groups = chosen["MVT_ID_mvt"].to_numpy()
    family = chosen["key2"].to_numpy() if "key2" in chosen.columns else None
    y = chosen["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64)
    candidates = {
        "hard_selection": hard,
        "probability_mixture": mix,
    }
    reports = {}
    validation_rows: list[dict[str, Any]] = []
    calibration_rows = source_metric_rows(chosen, model=f"logistic_{chosen_kind}")
    validation_rows.extend(confusion_rows(chosen, model=f"logistic_{chosen_kind}"))
    for name, pred in candidates.items():
        rows = evaluate_predictions(chosen, pred, model=name)
        validation_rows.extend(rows)
        pooled = next(row for row in rows if row["slice"] == "pooled" and row["block"] == "pooled")
        tail = next(row for row in rows if row["slice"] == "tail_7200" and row["block"] == "pooled")
        block_rows = [row for row in rows if row["slice"] == "unmatched" and row["block"] != "pooled"]
        pooled_gate = {
            **pooled,
            "tail_rmse": tail["rmse"],
            "tail_baseline_rmse": tail["baseline_rmse"],
        }
        is_mixture = name == "probability_mixture"
        decision = gate_decision(
            block_rows,
            pooled=pooled_gate,
            row_share=top_share(y, pred, u6),
            family_share=top_share(y, pred, u6, family),
            precision=None if is_mixture else picked_precision,
            mixture=is_mixture,
        )
        if is_mixture and not calibrated:
            decision["eligible"] = False
            decision["reasons"] = ["poor_calibration_mixture_not_forced", *decision["reasons"]]
        reports[name] = {
            "eligible": decision["eligible"],
            "reasons": decision["reasons"],
            "pooled_sse_ratio": pooled["sse_ratio"],
            "pooled_rmse": pooled["rmse"],
            "baseline_rmse": pooled["baseline_rmse"],
            "tail_rmse": tail["rmse"],
            "tail_baseline_rmse": tail["baseline_rmse"],
            "fallback_share": float(1.0 - hard_pick.mean()) if name == "hard_selection" else 0.0,
            "hard_precision": picked_precision,
            "calibration_slope": slope,
            "calibrated": calibrated,
        }
    return {
        "chosen_feature_kind": chosen_kind,
        "feature_log_loss": losses,
        "n0": n0,
        "oof": chosen,
        "reports": reports,
        "validation_rows": validation_rows,
        "calibration_rows": calibration_rows,
        "hard_prediction": hard,
        "mixture_prediction": mix,
        "probabilities": proba,
    }


def prediction_checksum(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(np.round(values.astype(np.float64), 6)).tobytes()
    return hashlib.sha256(payload).hexdigest()


def schema_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def freeze_manifest(report: dict[str, Any], *, feature_kind: str, n0: float, checksum: str, months: Sequence[int]) -> dict[str, Any]:
    if any(int(m) in {1, 7, 11, 12} for m in months):
        raise RuntimeError("freeze manifest refuses primary or V4 months in the fit population")
    cats, nums = feature_names(feature_kind)
    body = {
        "feature_kind": feature_kind,
        "cats": cats,
        "nums": nums,
        "n0": n0,
        "classes": list(CLASSES),
        "tolerance_sec": TOL_SEC,
        "forbidden": sorted(FEATURE_FORBIDDEN),
        "gates": report,
    }
    return {
        "model_hash": schema_hash(body),
        "prediction_checksum": checksum,
        "fit_months": [int(m) for m in months],
        "spec": body,
        "eligible": bool(report.get("eligible")),
    }
