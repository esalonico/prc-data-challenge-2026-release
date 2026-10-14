# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Covariate vs conditional/label shift diagnostics for unmatched rows.

Domain classifiers distinguish *dataset domains* (source vs target), never taxi
labels. Target Y is not read when estimating domain scores or density-ratio
features.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from prc_challenge.campaign.scoring import assert_no_forbidden_features
from prc_challenge.unmatched_recovery.methods import fit_classifier, to_pool
from prc_challenge.unmatched_recovery.schema import LOCKED_MONTHS, present
from prc_challenge.unmatched_recovery.subgroups import compact_score

DOMAIN_CATS = (
    "AIRPORT",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "flight_prefix",
    "ADES_mvt",
    "FLIGHT_RULE_mvt",
    "delay_bucket",
)
DOMAIN_NUMS = (
    "local_hour",
    "local_dow",
    "takeoff_minus_sched_sec",
    "takeoff_minus_sched_days",
    "takeoff_minus_sched_mod_sec",
    "type_missing",
    "ades_missing",
    "runway_missing",
    "stand_missing",
)
# Calendar month is inference-visible but is a domain identifier across D vs Jan/Jul.
# It remains in the unmatched taxi specialist; it is excluded from shift/importance X.
FORBIDDEN_TARGET_COLS = {
    "TAXITIME_SEC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "AOBT_3_flt",
    "prediction",
    "parent_prediction",
    "u6_prediction",
}
DELAY_COPY_GAP_SEC = 300.0
EXTREME_SEC = 7200.0
LONG_SEC = 3600.0


def assert_no_target_columns(columns: Sequence[str], *, context: str) -> None:
    hit = FORBIDDEN_TARGET_COLS.intersection(columns)
    if hit:
        raise RuntimeError(f"{context} used hidden/target columns: {sorted(hit)}")


def domain_feature_lists(frame: pl.DataFrame) -> tuple[list[str], list[str]]:
    cats = present(frame, DOMAIN_CATS)
    nums = present(frame, DOMAIN_NUMS)
    assert_no_forbidden_features(cats + nums)
    assert_no_target_columns(cats + nums, context="domain features")
    return cats, nums


def tag_domain(frame: pl.DataFrame, domain: str) -> pl.DataFrame:
    """Attach a dataset-domain label. This is not a taxi label."""

    if "month" in frame.columns:
        bad = frame.filter(pl.col("month").is_in(list(LOCKED_MONTHS)))
        if bad.height:
            raise RuntimeError("V4 months in domain table")
    return frame.with_columns(pl.lit(domain).alias("dataset_domain"))


def stack_domains(source: pl.DataFrame, target: pl.DataFrame, *, source_name: str, target_name: str) -> pl.DataFrame:
    src = tag_domain(source, source_name)
    tgt = tag_domain(target, target_name)
    cols = sorted(set(src.columns) & set(tgt.columns))
    stacked = pl.concat([src.select(cols), tgt.select(cols)], how="vertical")
    return stacked.with_columns((pl.col("dataset_domain") == target_name).cast(pl.Int8).alias("domain_label"))


def _numeric_matrix(frame: pl.DataFrame, cats: list[str], nums: list[str]) -> np.ndarray:
    """Dense matrix for sklearn domain models. No taxi columns."""

    assert_no_target_columns(cats + nums, context="sklearn domain matrix")
    pieces: list[np.ndarray] = []
    for col in nums:
        val = frame[col].cast(pl.Float64).fill_null(0).to_numpy()
        pieces.append(val.reshape(-1, 1))
    for col in cats:
        tokens = frame[col].cast(pl.String).fill_null("__UNK__")
        top = tokens.value_counts(sort=True).head(24)[col].to_list()
        for token in top:
            pieces.append((tokens == token).cast(pl.Float64).to_numpy().reshape(-1, 1))
    if not pieces:
        return np.zeros((frame.height, 1), dtype=np.float64)
    return np.hstack(pieces)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    a = np.asarray(expected, dtype=np.float64)
    b = np.asarray(actual, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 20 or b.size < 20:
        return float("nan")
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(a, qs))
    if edges.size < 3:
        return 0.0
    e_counts, _ = np.histogram(a, bins=edges)
    a_counts, _ = np.histogram(b, bins=edges)
    e_p = np.clip(e_counts / max(e_counts.sum(), 1), 1e-4, 1)
    a_p = np.clip(a_counts / max(a_counts.sum(), 1), 1e-4, 1)
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))


def ks_distance(expected: np.ndarray, actual: np.ndarray) -> float:
    a = np.sort(np.asarray(expected, dtype=np.float64))
    b = np.sort(np.asarray(actual, dtype=np.float64))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_1d(expected: np.ndarray, actual: np.ndarray) -> float:
    a = np.sort(np.asarray(expected, dtype=np.float64))
    b = np.sort(np.asarray(actual, dtype=np.float64))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = max(a.size, b.size)
    qa = np.quantile(a, np.linspace(0, 1, n))
    qb = np.quantile(b, np.linspace(0, 1, n))
    return float(np.mean(np.abs(qa - qb)))


def x_distribution_distances(source: pl.DataFrame, target: pl.DataFrame) -> dict[str, Any]:
    cats, nums = domain_feature_lists(source)
    cats = [c for c in cats if c in target.columns]
    nums = [c for c in nums if c in target.columns]
    payload: dict[str, Any] = {"n_source": source.height, "n_target": target.height, "numeric": {}, "categorical": {}}
    shifts: list[float] = []
    for col in nums:
        s = source[col].cast(pl.Float64).to_numpy()
        t = target[col].cast(pl.Float64).to_numpy()
        row = {
            "psi": psi(s, t),
            "ks": ks_distance(s, t),
            "wasserstein": wasserstein_1d(s, t),
            "source_mean": float(np.nanmean(s)) if s.size else float("nan"),
            "target_mean": float(np.nanmean(t)) if t.size else float("nan"),
        }
        payload["numeric"][col] = row
        if np.isfinite(row["psi"]):
            shifts.append(float(row["psi"]))
    for col in cats:
        s = source[col].cast(pl.String).fill_null("__UNK__").value_counts()
        t = target[col].cast(pl.String).fill_null("__UNK__").value_counts()
        keys = sorted(set(s[col].to_list()) | set(t[col].to_list()))
        s_map = dict(zip(s[col].to_list(), s["count"].to_list()))
        t_map = dict(zip(t[col].to_list(), t["count"].to_list()))
        s_p = np.array([s_map.get(k, 0) / max(source.height, 1) for k in keys], dtype=np.float64)
        t_p = np.array([t_map.get(k, 0) / max(target.height, 1) for k in keys], dtype=np.float64)
        s_p = np.clip(s_p, 1e-6, 1)
        t_p = np.clip(t_p, 1e-6, 1)
        js = 0.5 * float(np.sum(s_p * np.log(s_p / ((s_p + t_p) / 2)))) + 0.5 * float(
            np.sum(t_p * np.log(t_p / ((s_p + t_p) / 2)))
        )
        payload["categorical"][col] = {"js": js, "n_levels": len(keys)}
        shifts.append(js)
    payload["x_shift"] = float(np.mean(shifts)) if shifts else float("nan")
    return payload


def delay_copy_mask(y: np.ndarray, delay: np.ndarray, *, min_y: float = EXTREME_SEC, gap: float = DELAY_COPY_GAP_SEC) -> np.ndarray:
    taxi = np.asarray(y, dtype=np.float64)
    dly = np.asarray(delay, dtype=np.float64)
    return (taxi >= min_y) & np.isfinite(dly) & (np.abs(taxi - dly) <= gap)


def cell_label_stats(frame: pl.DataFrame, y_col: str = "TAXITIME_SEC_mvt") -> dict[str, Any]:
    if y_col not in frame.columns:
        return {"n": frame.height, "labels_hidden": True}
    y = frame[y_col].cast(pl.Float64).to_numpy()
    delay = (
        frame["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
        if "takeoff_minus_sched_sec" in frame.columns
        else np.full(frame.height, np.nan)
    )
    finite = y[np.isfinite(y)]
    copy = delay_copy_mask(y, delay)
    payload = {
        "n": int(frame.height),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "p_ge_3600": float(np.mean(y >= LONG_SEC)) if y.size else float("nan"),
        "p_ge_7200": float(np.mean(y >= EXTREME_SEC)) if y.size else float("nan"),
        "delay_copy_rate": float(copy.mean()) if y.size else float("nan"),
        "labels_hidden": False,
    }
    if "prediction" in frame.columns:
        payload.update(compact_score(y, frame["prediction"].to_numpy()))
    return payload


def conditional_shift_score(*, x_shift: float, extreme_delta: float, delay_copy_delta: float) -> float:
    """Large label-regime change with small X change => concept/conditional shift."""

    if not np.isfinite(x_shift):
        return float("nan")
    label = abs(float(extreme_delta)) + abs(float(delay_copy_delta))
    return float(label / (abs(x_shift) + 0.02))


def known_cells() -> tuple[tuple[str, Any], ...]:
    return (
        ("all_unmatched", None),
        ("LIRF", lambda df: df["AIRPORT"] == "LIRF"),
        ("LIRF_type_missing", lambda df: (df["AIRPORT"] == "LIRF") & (df["type_missing"] == 1)),
        ("LIRF_d2_6h", lambda df: (df["AIRPORT"] == "LIRF") & (df["delay_bucket"] == "d2_6h")),
        ("LTFM", lambda df: df["AIRPORT"] == "LTFM"),
        ("LSZH", lambda df: df["AIRPORT"] == "LSZH"),
    )


def filter_cell(frame: pl.DataFrame, spec: Any) -> pl.DataFrame:
    if spec is None:
        return frame
    mask = spec(frame)
    return frame.filter(mask)


def classifier_metrics(y: np.ndarray, p: np.ndarray, *, name: str) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int32)
    scores = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    payload: dict[str, Any] = {"name": name, "n": int(labels.size), "base_rate": float(labels.mean()) if labels.size else float("nan")}
    if labels.size and labels.min() != labels.max():
        payload["roc_auc"] = float(roc_auc_score(labels, scores))
        payload["pr_auc"] = float(average_precision_score(labels, scores))
        payload["brier"] = float(brier_score_loss(labels, scores))
        payload["log_loss"] = float(log_loss(labels, np.vstack([1 - scores, scores]).T))
        payload["score_p10"] = float(np.quantile(scores, 0.10))
        payload["score_p50"] = float(np.quantile(scores, 0.50))
        payload["score_p90"] = float(np.quantile(scores, 0.90))
        payload["support_overlap"] = float(np.mean((scores > 0.20) & (scores < 0.80)))
    return payload


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y, dtype=np.float64)
    scores = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    if labels.size < 20 or labels.min() == labels.max():
        return {"slope": float("nan"), "intercept": float("nan")}
    logit = np.log(scores / (1 - scores))
    x = np.column_stack([np.ones(logit.size), logit])
    beta, *_ = np.linalg.lstsq(x, labels, rcond=None)
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def fit_domain_models(
    stacked: pl.DataFrame,
    *,
    iterations: int = 80,
    thread_count: int = 4,
) -> dict[str, Any]:
    """Train domain classifiers. Label is dataset_domain, not taxi."""

    if "domain_label" not in stacked.columns:
        raise RuntimeError("domain_label missing")
    if stacked["domain_label"].n_unique() < 2:
        raise RuntimeError("need both source and target domain rows")
    cats, nums = domain_feature_lists(stacked)
    y = stacked["domain_label"].to_numpy().astype(np.int32)
    x = _numeric_matrix(stacked, cats, nums)
    logistic = LogisticRegression(max_iter=200, solver="liblinear")
    logistic.fit(x, y)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42)
    tree.fit(x, y)
    cb = fit_classifier(
        stacked,
        cats,
        nums,
        "domain_label",
        iterations=iterations,
        thread_count=thread_count,
        depth=4,
    )
    models = {
        "logistic": {"predict": logistic.predict_proba(x)[:, 1], "importance": _logistic_importance(logistic, x.shape[1])},
        "tree": {"predict": tree.predict_proba(x)[:, 1], "importance": tree.feature_importances_.tolist()},
        "catboost": {
            "predict": np.asarray(cb.predict_proba(to_pool(stacked, cats, nums))[:, 1], dtype=np.float64),
            "importance": cb.get_feature_importance().tolist(),
            "feature_names": cats + nums,
        },
    }
    out: dict[str, Any] = {"n": stacked.height, "target_rate": float(y.mean()), "features": {"cats": cats, "nums": nums}}
    for name, payload in models.items():
        metrics = classifier_metrics(y, payload["predict"], name=name)
        metrics["importance"] = payload.get("importance")
        metrics["predict"] = np.asarray(payload["predict"], dtype=np.float64)
        if "feature_names" in payload:
            metrics["feature_names"] = payload["feature_names"]
            ranked = sorted(zip(payload["feature_names"], payload["importance"]), key=lambda r: -abs(float(r[1])))
            metrics["top_features"] = [{"name": n, "importance": float(v)} for n, v in ranked[:12]]
        out[name] = metrics
    out["auc_catboost"] = out["catboost"].get("roc_auc")
    return out


def _logistic_importance(model: LogisticRegression, n_features: int) -> list[float]:
    coef = np.asarray(model.coef_, dtype=np.float64).ravel()
    if coef.size != n_features:
        return coef.tolist()
    return coef.tolist()


def oof_domain_scores(
    source: pl.DataFrame,
    target: pl.DataFrame,
    *,
    source_name: str,
    target_name: str,
    iterations: int = 80,
    thread_count: int = 4,
) -> dict[str, Any]:
    """Fit on source+target X only. Target Y is never attached."""

    src = source.drop([c for c in FORBIDDEN_TARGET_COLS if c in source.columns])
    tgt = target.drop([c for c in FORBIDDEN_TARGET_COLS if c in target.columns])
    stacked = stack_domains(src, tgt, source_name=source_name, target_name=target_name)
    metrics = fit_domain_models(stacked, iterations=iterations, thread_count=thread_count)
    src_mask = stacked["domain_label"].to_numpy() == 0
    tgt_mask = stacked["domain_label"].to_numpy() == 1
    cb_scores = np.asarray(metrics["catboost"]["predict"], dtype=np.float64)
    src_scores = cb_scores[src_mask]
    tgt_scores = cb_scores[tgt_mask]
    metrics["source_scores"] = src_scores
    metrics["target_scores"] = tgt_scores
    metrics["target_score_quantiles"] = {
        "p10": float(np.quantile(tgt_scores, 0.10)),
        "p50": float(np.quantile(tgt_scores, 0.50)),
        "p90": float(np.quantile(tgt_scores, 0.90)),
    }
    metrics["source_score_quantiles"] = {
        "p10": float(np.quantile(src_scores, 0.10)),
        "p50": float(np.quantile(src_scores, 0.50)),
        "p90": float(np.quantile(src_scores, 0.90)),
    }
    metrics["x_distances"] = x_distribution_distances(src, tgt)
    metrics["source_name"] = source_name
    metrics["target_name"] = target_name
    return metrics
