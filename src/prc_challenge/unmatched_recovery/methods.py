# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Unmatched-recovery predictors, imputers, distillation, mixture, and linkage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.metrics import log_loss, roc_auc_score

from prc_challenge.bundles import CATBOOST_BASELINE
from prc_challenge.campaign.scoring import assert_no_forbidden_features
from prc_challenge.opdi.schema import normalize_callsign, resolve_flight_list_columns
from prc_challenge.unmatched_recovery.schema import (
    LATENT_PLANNING,
    NM_CATEGORICALS,
    PRECISION_GATE,
    REGIME_EDGES,
    apply_nm_block_mask,
    derive_planning_from_latents,
    present,
    unmatched_observed_features,
)

CAT_FILL = "__UNK__"


def specialist_params(
    *,
    iterations: int = 250,
    depth: int = 6,
    thread_count: int = 4,
    l2_leaf_reg: float = 12.0,
    learning_rate: float = 0.08,
    nan_mode: str = "Min",
    loss_function: str = "RMSE",
    eval_metric: str | None = None,
) -> dict[str, Any]:
    params = dict(CATBOOST_BASELINE)
    metric = eval_metric if eval_metric is not None else loss_function
    params.update(
        {
            "iterations": iterations,
            "depth": depth,
            "thread_count": thread_count,
            "l2_leaf_reg": l2_leaf_reg,
            "learning_rate": learning_rate,
            "random_strength": 1.0,
            "nan_mode": nan_mode,
            "loss_function": loss_function,
            "eval_metric": metric,
            "allow_writing_files": False,
            "verbose": False,
        }
    )
    return params


def to_pool(
    df: pl.DataFrame,
    cats: list[str],
    nums: list[str],
    label: str | None = None,
    *,
    native_missing: bool = True,
    weight: np.ndarray | None = None,
) -> Pool:
    """CatBoost pool. Native missing keeps numeric NaNs; cats stay nullable strings."""

    assert_no_forbidden_features(cats + nums)
    exprs: list[pl.Expr] = []
    for c in cats:
        # CatBoost categorical None is not hashable; a dedicated missing token is the native category.
        exprs.append(pl.col(c).cast(pl.String).fill_null(CAT_FILL).alias(c))
    for c in nums:
        col = pl.col(c).cast(pl.Float32)
        if not native_missing:
            col = pl.col(c).fill_null(0).cast(pl.Float32)
        exprs.append(col.alias(c))
    frame = df.select(exprs) if exprs else df.select(cats + nums)
    kwargs: dict[str, Any] = {"data": frame.to_pandas(), "cat_features": cats}
    if label is not None:
        kwargs["label"] = df[label].to_numpy()
    if weight is not None:
        kwargs["weight"] = np.asarray(weight, dtype=np.float64)
    return Pool(**kwargs)


def fit_regressor(
    train: pl.DataFrame,
    cats: list[str],
    nums: list[str],
    label: str = "TAXITIME_SEC_mvt",
    *,
    native_missing: bool = True,
    sample_weight: np.ndarray | None = None,
    **params: Any,
) -> CatBoostRegressor:
    model = CatBoostRegressor(**specialist_params(**params))
    model.fit(to_pool(train, cats, nums, label, native_missing=native_missing, weight=sample_weight))
    return model


def predict_regressor(
    model: CatBoostRegressor,
    df: pl.DataFrame,
    cats: list[str],
    nums: list[str],
    *,
    native_missing: bool = True,
) -> np.ndarray:
    pred = np.asarray(model.predict(to_pool(df, cats, nums, native_missing=native_missing)), dtype=np.float64)
    return np.clip(pred, 0, None)


def fit_classifier(
    train: pl.DataFrame,
    cats: list[str],
    nums: list[str],
    label: str,
    *,
    native_missing: bool = True,
    iterations: int = 150,
    **params: Any,
) -> CatBoostClassifier:
    cfg = specialist_params(iterations=iterations, **params)
    cfg["loss_function"] = "MultiClass" if train[label].n_unique() > 2 else "Logloss"
    cfg["eval_metric"] = cfg["loss_function"]
    model = CatBoostClassifier(**cfg)
    model.fit(to_pool(train, cats, nums, label, native_missing=native_missing))
    return model


def feature_matrix(df: pl.DataFrame, cats: list[str], nums: list[str]) -> tuple[list[str], list[str]]:
    return present(df, cats), present(df, nums)


def observed_lists(frame: pl.DataFrame) -> tuple[list[str], list[str]]:
    return unmatched_observed_features(frame)


# ---------------------------------------------------------------------------
# Simple / hierarchical imputation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HierarchicalLookup:
    target: str
    kind: str
    global_value: Any
    tables: tuple[pl.DataFrame, ...]
    levels: tuple[tuple[str, ...], ...]
    min_count: int


def _mode_or_median(series: pl.Series, kind: str) -> Any:
    clean = series.drop_nulls()
    if clean.len() == 0:
        return None
    if kind == "categorical":
        modes = clean.mode()
        return modes[0] if modes.len() else None
    return float(clean.median())


def fit_hierarchical_lookup(
    train: pl.DataFrame,
    target: str,
    levels: tuple[tuple[str, ...], ...],
    *,
    kind: str,
    min_count: int = 20,
) -> HierarchicalLookup:
    labeled = train.filter(pl.col(target).is_not_null())
    global_value = _mode_or_median(labeled[target], kind)
    tables: list[pl.DataFrame] = []
    for level in levels:
        if any(c not in labeled.columns for c in level):
            tables.append(labeled.clear())
            continue
        agg = pl.col(target).drop_nulls()
        stats = labeled.group_by(list(level)).agg(
            (agg.mode().first() if kind == "categorical" else agg.median()).alias("value"),
            pl.len().alias("n"),
        ).filter(pl.col("n") >= min_count)
        tables.append(stats)
    return HierarchicalLookup(target, kind, global_value, tuple(tables), levels, min_count)


def apply_hierarchical_lookup(frame: pl.DataFrame, lookup: HierarchicalLookup) -> pl.Series:
    out = frame.with_columns(pl.lit(lookup.global_value).alias("_imp"))
    for level, table in zip(reversed(lookup.levels), reversed(lookup.tables)):
        if table.height == 0 or any(c not in frame.columns for c in level):
            continue
        tmp = out.join(
            table.select([*level, pl.col("value").alias("_lvl")]),
            on=list(level),
            how="left",
        )
        out = tmp.with_columns(pl.coalesce(pl.col("_lvl"), pl.col("_imp")).alias("_imp")).drop("_lvl")
    return out["_imp"]


CAT_LEVELS: dict[str, tuple[tuple[str, ...], ...]] = {
    "AIRCRAFT_OPERATOR_flt": (
        ("AIRPORT", "flight_prefix"),
        ("flight_prefix",),
        ("AIRPORT",),
    ),
    "WK_TBL_CAT_flt": (
        ("AIRCRAFT_TYPE_mvt",),
        ("AIRPORT", "flight_prefix"),
        ("AIRPORT",),
    ),
    "MARKET_SEGMENT_flt": (
        ("AIRPORT", "flight_prefix", "local_hour"),
        ("AIRPORT", "flight_prefix"),
        ("AIRPORT",),
    ),
    "FLIGHT_TYPE_flt": (
        ("AIRPORT", "flight_prefix"),
        ("AIRPORT",),
    ),
    "AIRCRAFT_TYPE_mvt": (
        ("AIRPORT", "flight_prefix"),
        ("flight_prefix",),
        ("AIRPORT",),
    ),
}
NUM_LEVELS: dict[str, tuple[tuple[str, ...], ...]] = {
    "sched_minus_iobt_sec": (
        ("AIRPORT", "flight_prefix", "local_hour"),
        ("AIRPORT", "local_hour"),
        ("AIRPORT",),
    ),
    "eobt_minus_iobt_sec": (
        ("AIRPORT", "flight_prefix"),
        ("AIRPORT",),
    ),
}


def fit_simple_imputer(train: pl.DataFrame, *, min_count: int = 20) -> dict[str, HierarchicalLookup]:
    fitted: dict[str, HierarchicalLookup] = {}
    for target, levels in CAT_LEVELS.items():
        if target in train.columns:
            fitted[target] = fit_hierarchical_lookup(train, target, levels, kind="categorical", min_count=min_count)
    for target, levels in NUM_LEVELS.items():
        if target in train.columns:
            fitted[target] = fit_hierarchical_lookup(train, target, levels, kind="numeric", min_count=min_count)
    return fitted


def apply_simple_imputer(frame: pl.DataFrame, fitted: dict[str, HierarchicalLookup]) -> pl.DataFrame:
    out = frame
    flags: list[pl.Expr] = []
    for target, lookup in fitted.items():
        if target not in out.columns:
            continue
        hat = apply_hierarchical_lookup(out, lookup)
        filled = pl.when(pl.col(target).is_null()).then(hat).otherwise(pl.col(target))
        out = out.with_columns(filled.alias(target), hat.alias(f"{target}_hat"))
        flags.append(pl.col(target).is_null().cast(pl.Int8).alias(f"imputed_{target}"))
    if flags:
        # Flags must use the *original* missingness; recompute from hats vs original via _hat presence.
        pass
    out = derive_planning_from_latents(out)
    return out.with_columns(
        pl.lit(1).cast(pl.Int8).alias("original_nm_unmatched")
        if "nm_unmatched" in out.columns
        else pl.lit(0).cast(pl.Int8).alias("original_nm_unmatched")
    )


def apply_simple_imputer_masked(frame: pl.DataFrame, fitted: dict[str, HierarchicalLookup]) -> pl.DataFrame:
    """Overwrite NM-block fields with lookups (used after artificial masking)."""

    out = frame
    for target, lookup in fitted.items():
        hat = apply_hierarchical_lookup(out, lookup)
        out = out.with_columns(hat.alias(target), hat.alias(f"{target}_hat"))
    return derive_planning_from_latents(out)


# ---------------------------------------------------------------------------
# Supervised per-field imputation
# ---------------------------------------------------------------------------


@dataclass
class FieldImputer:
    target: str
    kind: str
    cats: list[str]
    nums: list[str]
    model: Any
    residual_q: tuple[float, float] | None = None
    classes_: list[Any] | None = None


def _imputer_inputs(frame: pl.DataFrame, target: str) -> tuple[list[str], list[str]]:
    cats, nums = observed_lists(frame)
    cats = [c for c in cats if c != target]
    nums = [c for c in nums if c != target]
    return cats, nums


def fit_field_imputers(
    train: pl.DataFrame,
    targets: Sequence[str],
    *,
    iterations: int = 120,
    thread_count: int = 4,
    sample: int | None = 80_000,
) -> list[FieldImputer]:
    src = train
    if sample and src.height > sample:
        src = src.sample(n=sample, seed=42, shuffle=True)
    fitted: list[FieldImputer] = []
    for target in targets:
        if target not in src.columns:
            continue
        labeled = src.filter(pl.col(target).is_not_null())
        if labeled.height < 200:
            continue
        cats, nums = _imputer_inputs(labeled, target)
        kind = "categorical" if target in NM_CATEGORICALS or target == "AIRCRAFT_TYPE_mvt" else "numeric"
        if kind == "categorical":
            # Cap rare operator classes to keep MultiClass tractable.
            if target == "AIRCRAFT_OPERATOR_flt":
                counts = labeled.group_by(target).len()
                keep = counts.filter(pl.col("len") >= 30)[target]
                labeled = labeled.filter(pl.col(target).is_in(keep.to_list()))
                if labeled.height < 200 or labeled[target].n_unique() < 2:
                    continue
            model = fit_classifier(labeled, cats, nums, target, iterations=iterations, thread_count=thread_count)
            classes = list(model.classes_) if hasattr(model, "classes_") else None
            fitted.append(FieldImputer(target, kind, cats, nums, model, classes_=classes))
        else:
            model = fit_regressor(labeled, cats, nums, target, iterations=iterations, thread_count=thread_count)
            pred = predict_regressor(model, labeled, cats, nums)
            resid = labeled[target].to_numpy() - pred
            q = (float(np.quantile(resid, 0.1)), float(np.quantile(resid, 0.9)))
            fitted.append(FieldImputer(target, kind, cats, nums, model, residual_q=q))
    return fitted


def apply_field_imputers(
    frame: pl.DataFrame,
    imputers: Sequence[FieldImputer],
    *,
    chained: bool = False,
) -> pl.DataFrame:
    out = frame
    for spec in imputers:
        cats = present(out, spec.cats)
        nums = present(out, spec.nums)
        if spec.kind == "categorical":
            proba = np.asarray(spec.model.predict_proba(to_pool(out, cats, nums)), dtype=np.float64)
            classes = spec.classes_ or list(spec.model.classes_)
            idx = np.argmax(proba, axis=1)
            hat = np.asarray(classes, dtype=object)[idx]
            conf = proba.max(axis=1)
            entropy = -np.sum(np.clip(proba, 1e-12, 1.0) * np.log(np.clip(proba, 1e-12, 1.0)), axis=1)
            out = out.with_columns(
                pl.Series(f"{spec.target}_hat", hat),
                pl.Series(f"{spec.target}_conf", conf),
                pl.Series(f"{spec.target}_entropy", entropy),
            )
            if chained:
                out = out.with_columns(
                    pl.when(pl.col(spec.target).is_null())
                    .then(pl.col(f"{spec.target}_hat"))
                    .otherwise(pl.col(spec.target))
                    .alias(spec.target)
                )
        else:
            hat = np.asarray(spec.model.predict(to_pool(out, cats, nums)), dtype=np.float64)
            out = out.with_columns(pl.Series(f"{spec.target}_hat", hat))
            if chained:
                out = out.with_columns(
                    pl.when(pl.col(spec.target).is_null())
                    .then(pl.col(f"{spec.target}_hat"))
                    .otherwise(pl.col(spec.target))
                    .alias(spec.target)
                )
            if spec.residual_q is not None:
                out = out.with_columns(
                    pl.lit(spec.residual_q[0]).alias(f"{spec.target}_q10"),
                    pl.lit(spec.residual_q[1]).alias(f"{spec.target}_q90"),
                )
    # Hard completion of latents when hats exist.
    for col in LATENT_PLANNING:
        hat = f"{col}_hat"
        if hat in out.columns:
            if col in out.columns:
                out = out.with_columns(pl.coalesce(pl.col(col), pl.col(hat)).alias(col))
            else:
                out = out.with_columns(pl.col(hat).alias(col))
    return derive_planning_from_latents(out)


# ---------------------------------------------------------------------------
# MICE / MissForest-style
# ---------------------------------------------------------------------------


def fit_mice(
    train: pl.DataFrame,
    cols: Sequence[str],
    *,
    sample: int = 25_000,
    max_iter: int = 8,
    estimator: str = "bayes",
) -> tuple[IterativeImputer, list[str], dict[str, float]]:
    use = [c for c in cols if c in train.columns]
    src = train.select(use)
    if src.height > sample:
        src = src.sample(n=sample, seed=42, shuffle=True)
    medians = {c: float(src[c].drop_nulls().median() or 0.0) for c in use}
    arr = src.to_numpy()
    est = BayesianRidge() if estimator == "bayes" else HuberRegressor()
    imputer = IterativeImputer(
        estimator=est,
        max_iter=max_iter,
        sample_posterior=estimator == "bayes",
        random_state=42,
        add_indicator=True,
    )
    imputer.fit(arr)
    return imputer, use, medians


def transform_mice(
    frame: pl.DataFrame,
    imputer: IterativeImputer,
    cols: list[str],
) -> pl.DataFrame:
    arr = frame.select(cols).to_numpy()
    filled = imputer.transform(arr)
    out = frame
    for i, col in enumerate(cols):
        out = out.with_columns(pl.Series(col, np.asarray(filled[:, i], dtype=np.float64)))
        out = out.with_columns(pl.col(col).alias(f"{col}_hat"))
    return derive_planning_from_latents(out)


def sample_mice(
    frame: pl.DataFrame,
    imputer: IterativeImputer,
    cols: list[str],
    *,
    k: int = 5,
) -> list[pl.DataFrame]:
    draws: list[pl.DataFrame] = []
    arr = frame.select(cols).to_numpy()
    for _ in range(k):
        filled = imputer.transform(arr)
        part = frame
        for i, col in enumerate(cols):
            part = part.with_columns(pl.Series(col, np.asarray(filled[:, i], dtype=np.float64)))
        draws.append(derive_planning_from_latents(part))
    return draws


def missforest_impute(
    train: pl.DataFrame,
    apply_to: pl.DataFrame,
    targets: Sequence[str],
    *,
    sample: int = 12_000,
    rounds: int = 3,
    iterations: int = 80,
    thread_count: int = 4,
) -> pl.DataFrame:
    """Bounded mixed-type iterative tree imputation (MissForest approximation)."""

    src = train
    if src.height > sample:
        src = src.sample(n=sample, seed=42, shuffle=True)
    working = apply_to.clone()
    simple = fit_simple_imputer(src, min_count=10)
    working = apply_simple_imputer_masked(working, simple)
    src_filled = apply_simple_imputer_masked(src, simple)
    for _ in range(rounds):
        for target in targets:
            if target not in src_filled.columns:
                continue
            cats, nums = _imputer_inputs(src_filled, target)
            labeled = src_filled.filter(pl.col(target).is_not_null()) if target in train.columns else src_filled
            # After simple fill the train copy is complete; refit against original non-null targets.
            original = src.filter(pl.col(target).is_not_null()) if target in src.columns else src
            if original.height < 100:
                continue
            cats, nums = _imputer_inputs(original, target)
            kind = "categorical" if original[target].dtype in (pl.String, pl.Utf8, pl.Categorical) else "numeric"
            if kind == "categorical" or target in NM_CATEGORICALS or target == "AIRCRAFT_TYPE_mvt":
                model = fit_classifier(original, cats, nums, target, iterations=iterations, thread_count=thread_count)
                hat = model.predict(to_pool(working, cats, nums))
                working = working.with_columns(pl.Series(target, np.asarray(hat).reshape(-1)))
            else:
                model = fit_regressor(original, cats, nums, target, iterations=iterations, thread_count=thread_count)
                hat = predict_regressor(model, working, cats, nums)
                working = working.with_columns(pl.Series(target, hat))
            working = working.with_columns(pl.col(target).alias(f"{target}_hat"))
    return derive_planning_from_latents(working)


# ---------------------------------------------------------------------------
# Distillation / joint MLP / mixture
# ---------------------------------------------------------------------------


def distill_targets(
    y: np.ndarray,
    teacher: np.ndarray,
    lam: float,
) -> np.ndarray:
    if lam <= 0:
        return y
    return (y + lam * teacher) / (1.0 + lam)


class MissingAwareMLP:
    """Compact NeuMiss-inspired taxi network (sklearn, no Torch)."""

    def __init__(
        self,
        n_num: int,
        n_aux: int,
        *,
        hidden: int = 64,
        lr: float = 1e-3,
        aux_weight: float = 0.15,
        seed: int = 42,
    ) -> None:
        from sklearn.neural_network import MLPRegressor

        self.n_aux = n_aux
        self.aux_weight = aux_weight
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden, hidden),
            activation="relu",
            solver="adam",
            learning_rate_init=lr,
            random_state=seed,
            max_iter=20,
            early_stopping=False,
        )

    def _pack(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        filled = np.nan_to_num(x, nan=0.0) * mask
        return np.concatenate([filled, mask], axis=1)

    def fit(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        aux: np.ndarray | None,
        *,
        epochs: int = 12,
        batch: int = 1024,
    ) -> dict[str, Any]:
        packed = self._pack(x, mask)
        if aux is not None and self.aux_weight > 0:
            packed = np.concatenate([packed, np.nan_to_num(aux, nan=0.0)], axis=1)
        self.model.max_iter = max(8, epochs)
        self.model.fit(packed, y)
        return {"epochs": int(self.model.n_iter_), "loss": list(self.model.loss_curve_[-5:])}

    def predict(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        packed = self._pack(x, mask)
        if self.n_aux and self.aux_weight > 0:
            packed = np.concatenate([packed, np.zeros((x.shape[0], self.n_aux), dtype=np.float32)], axis=1)
        pred = np.asarray(self.model.predict(packed), dtype=np.float64)
        return np.clip(pred, 0, None)


def numeric_mask_block(frame: pl.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    use = [c for c in cols if c in frame.columns]
    arr = frame.select([pl.col(c).cast(pl.Float64) for c in use]).to_numpy()
    mask = np.isfinite(arr).astype(np.float32)
    mu = np.nanmean(arr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(~np.isfinite(sd) | (sd < 1e-6), 1.0, sd)
    std = (arr - mu) / sd
    return std.astype(np.float32), mask


def fit_regime_mixture(
    train: pl.DataFrame,
    cats: list[str],
    nums: list[str],
    *,
    iterations: int = 180,
    thread_count: int = 4,
) -> dict[str, Any]:
    y = train["TAXITIME_SEC_mvt"].to_numpy()
    labels = np.zeros(train.height, dtype=np.int32)
    labels[y >= REGIME_EDGES[0]] = 1
    labels[y >= REGIME_EDGES[1]] = 2
    gate_frame = train.with_columns(pl.Series("regime", labels))
    # Collapse rare extreme class into long if support is tiny.
    counts = np.bincount(labels, minlength=3)
    if counts[2] < 25:
        labels = labels.copy()
        labels[labels == 2] = 1
        gate_frame = train.with_columns(pl.Series("regime", labels))
        n_class = 2
    else:
        n_class = 3
    if n_class == 2:
        params = specialist_params(iterations=iterations, thread_count=thread_count)
        params["loss_function"] = "Logloss"
        params["eval_metric"] = "Logloss"
        gate = CatBoostClassifier(**params)
    else:
        params = specialist_params(iterations=iterations, thread_count=thread_count)
        params["loss_function"] = "MultiClass"
        params["eval_metric"] = "MultiClass"
        gate = CatBoostClassifier(**params)
    gate.fit(to_pool(gate_frame, cats, nums, "regime"))
    experts = {}
    labeled = train.with_columns(pl.Series("_reg", labels))
    for cls, name in enumerate(("ordinary", "long", "extreme")[:n_class]):
        part = labeled.filter(pl.col("_reg") == cls).drop("_reg")
        if part.height < 40:
            experts[name] = None
            continue
        experts[name] = fit_regressor(
            part,
            cats,
            nums,
            iterations=max(80, iterations // 2),
            thread_count=thread_count,
            l2_leaf_reg=20.0 if name != "ordinary" else 12.0,
        )
    return {"gate": gate, "experts": experts, "n_class": n_class, "cats": cats, "nums": nums, "counts": counts.tolist()}


def predict_regime_mixture(bundle: dict[str, Any], frame: pl.DataFrame) -> np.ndarray:
    cats, nums = bundle["cats"], bundle["nums"]
    proba = np.asarray(bundle["gate"].predict_proba(to_pool(frame, cats, nums)), dtype=np.float64)
    if proba.ndim == 1:
        proba = np.vstack([1 - proba, proba]).T
    names = ("ordinary", "long", "extreme")[: bundle["n_class"]]
    acc = np.zeros(frame.height, dtype=np.float64)
    for i, name in enumerate(names):
        model = bundle["experts"].get(name)
        if model is None:
            pred = np.full(frame.height, float(REGIME_EDGES[min(i, 1)]), dtype=np.float64)
        else:
            pred = predict_regressor(model, frame, cats, nums)
        acc += proba[:, i] * pred
    return np.clip(acc, 0, None)


# ---------------------------------------------------------------------------
# Missingness classifier + density diagnostics
# ---------------------------------------------------------------------------


def fit_missingness_classifier(
    train: pl.DataFrame,
    *,
    iterations: int = 150,
    thread_count: int = 4,
) -> tuple[CatBoostClassifier, list[str], list[str]]:
    cats, nums = observed_lists(train)
    nums = [c for c in nums if c != "nm_unmatched"]
    model = fit_classifier(train, cats, nums, "nm_unmatched", iterations=iterations, thread_count=thread_count)
    return model, cats, nums


def missingness_metrics(
    model: CatBoostClassifier,
    frame: pl.DataFrame,
    cats: list[str],
    nums: list[str],
) -> dict[str, Any]:
    y = frame["nm_unmatched"].to_numpy().astype(int)
    proba = np.asarray(model.predict_proba(to_pool(frame, cats, nums)), dtype=np.float64)
    if proba.ndim == 2 and proba.shape[1] > 1:
        p = proba[:, 1]
    else:
        p = proba.reshape(-1)
    out: dict[str, Any] = {"n": int(frame.height), "unmatched_rate": float(y.mean())}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    return out


def density_distance_diagnostics(matched: pl.DataFrame, unmatched: pl.DataFrame, nums: Sequence[str]) -> dict[str, Any]:
    use = [c for c in nums if c in matched.columns and c in unmatched.columns]
    if not use or matched.height == 0 or unmatched.height == 0:
        return {"status": "skipped"}
    m = matched.select([pl.col(c).cast(pl.Float64) for c in use]).fill_null(strategy="mean")
    u = unmatched.select([pl.col(c).cast(pl.Float64) for c in use]).fill_null(strategy="mean")
    mu = m.mean().to_numpy().reshape(-1)
    sd = np.maximum(m.std().to_numpy().reshape(-1), 1e-6)
    mz = (m.to_numpy() - mu) / sd
    uz = (u.to_numpy() - mu) / sd
    # Sample matched centroids for a cheap kNN-L2 diagnostic.
    rng = np.random.default_rng(42)
    take = min(4000, mz.shape[0])
    sample = mz[rng.choice(mz.shape[0], size=take, replace=False)]
    u_take = uz if uz.shape[0] <= 3000 else uz[rng.choice(uz.shape[0], size=3000, replace=False)]
    d2 = []
    for row in u_take:
        dist = np.sqrt(((sample - row) ** 2).sum(axis=1))
        d2.append(float(np.partition(dist, 5)[:5].mean()))
    arr = np.asarray(d2)
    matched_self = []
    self_take = sample[: min(800, sample.shape[0])]
    for row in self_take:
        dist = np.sqrt(((sample - row) ** 2).sum(axis=1))
        matched_self.append(float(np.partition(dist, 6)[1:6].mean()))
    return {
        "unmatched_nn5_mean": float(arr.mean()),
        "unmatched_nn5_p90": float(np.quantile(arr, 0.9)),
        "matched_nn5_mean": float(np.mean(matched_self)),
        "ratio": float(arr.mean() / max(np.mean(matched_self), 1e-6)),
        "n_unmatched_scored": int(u_take.shape[0]),
    }


# ---------------------------------------------------------------------------
# External / OPDI blind linkage
# ---------------------------------------------------------------------------


def _norm_flight(series: pl.Series) -> pl.Series:
    return series.cast(pl.String).map_elements(normalize_callsign, return_dtype=pl.String)


def prepare_opdi(opdi: pl.DataFrame) -> pl.DataFrame:
    mapping = resolve_flight_list_columns(opdi.columns)
    renamed = opdi.rename({v: k for k, v in mapping.items() if v in opdi.columns})
    out = renamed.with_columns(
        _norm_flight(pl.col("callsign")).alias("callsign_norm"),
        pl.col("adep").cast(pl.String).str.to_uppercase().alias("adep"),
        pl.col("ades").cast(pl.String).str.to_uppercase().alias("ades"),
        pl.col("aircraft_type").cast(pl.String).str.to_uppercase().alias("aircraft_type"),
        pl.col("operator").cast(pl.String).alias("operator"),
    )
    if "first_seen" in out.columns:
        first = out["first_seen"]
        if str(first.dtype).startswith("Date"):
            pass
        elif first.dtype != pl.Datetime:
            out = out.with_columns(pl.col("first_seen").str.to_datetime(strict=False).alias("first_seen"))
        out = out.with_columns(pl.col("first_seen").dt.epoch("s").cast(pl.Int64).alias("first_seen_epoch"))
    if "last_seen" in out.columns:
        last = out["last_seen"]
        if str(last.dtype).startswith("Date"):
            pass
        elif last.dtype != pl.Datetime:
            out = out.with_columns(pl.col("last_seen").str.to_datetime(strict=False).alias("last_seen"))
        out = out.with_columns(pl.col("last_seen").dt.epoch("s").cast(pl.Int64).alias("last_seen_epoch"))
    if "date_of_flight" in out.columns:
        dof = out["date_of_flight"]
        if dof.dtype == pl.Datetime:
            out = out.with_columns(pl.col("date_of_flight").dt.date().alias("flight_date"))
        elif dof.dtype == pl.Date:
            out = out.with_columns(pl.col("date_of_flight").alias("flight_date"))
        else:
            out = out.with_columns(
                pl.col("date_of_flight")
                .cast(pl.String)
                .str.to_datetime(strict=False)
                .dt.date()
                .alias("flight_date")
            )
    return out


def blind_link_opdi(
    movements: pl.DataFrame,
    opdi: pl.DataFrame,
    *,
    window_sec: int = 7200,
) -> pl.DataFrame:
    """Link using only unmatched-available fields: public flight number, airport, takeoff time."""

    left = movements.with_columns(
        _norm_flight(pl.col("FLIGHT_mvt")).alias("callsign_norm"),
        pl.col("AIRPORT").cast(pl.String).str.to_uppercase().alias("adep"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("takeoff_epoch"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("flight_date"),
    )
    right = prepare_opdi(opdi)
    joined = left.join(right, on=["callsign_norm", "adep", "flight_date"], how="left", suffix="_opdi")
    if "first_seen_epoch" in joined.columns:
        joined = joined.with_columns(
            (pl.col("takeoff_epoch") - pl.col("first_seen_epoch")).alias("takeoff_minus_first_seen_sec")
        )
        joined = joined.with_columns(
            (
                pl.col("callsign_norm").is_not_null()
                & pl.col("opdi_id").is_not_null()
                & pl.col("takeoff_minus_first_seen_sec").is_between(0, window_sec)
            ).alias("link_ok")
        )
    else:
        joined = joined.with_columns(pl.col("opdi_id").is_not_null().alias("link_ok"))
    # Unique-match precision: drop IDs with >1 surviving candidate.
    if "opdi_id" in joined.columns:
        counts = joined.filter(pl.col("link_ok")).group_by("MVT_ID_mvt").len()
        unique = counts.filter(pl.col("len") == 1).select("MVT_ID_mvt")
        joined = joined.join(unique.with_columns(pl.lit(True).alias("unique_match")), on="MVT_ID_mvt", how="left")
        joined = joined.with_columns(pl.col("unique_match").fill_null(False))
        joined = joined.with_columns((pl.col("link_ok") & pl.col("unique_match")).alias("hard_link"))
    return joined


def linkage_precision_report(
    linked: pl.DataFrame,
    *,
    truth_operator: str = "AIRCRAFT_OPERATOR_flt",
    pred_operator: str = "operator",
) -> dict[str, Any]:
    hard = linked.filter(pl.col("hard_link")) if "hard_link" in linked.columns else linked.clear()
    n = linked.select("MVT_ID_mvt").n_unique() if "MVT_ID_mvt" in linked.columns else linked.height
    n_hard = hard.select("MVT_ID_mvt").n_unique() if hard.height else 0
    out: dict[str, Any] = {
        "n_rows": linked.height,
        "n_ids": int(n),
        "n_hard": int(n_hard),
        "coverage": float(n_hard / n) if n else 0.0,
        "precision_gate": PRECISION_GATE,
        "source_mode": "opdi_public",
    }
    if hard.height and truth_operator in hard.columns and pred_operator in hard.columns:
        pair = hard.select(
            pl.col(truth_operator).cast(pl.String).str.to_uppercase().alias("t"),
            pl.col(pred_operator).cast(pl.String).str.to_uppercase().alias("p"),
        ).drop_nulls()
        if pair.height:
            out["operator_accuracy"] = float((pair["t"] == pair["p"]).mean())
            out["precision_pass"] = bool(out["operator_accuracy"] >= PRECISION_GATE)
    return out


def ridge_latent_offsets(train: pl.DataFrame, apply_to: pl.DataFrame) -> pl.DataFrame:
    nums = [c for c in ("local_hour", "local_dow", "takeoff_minus_sched_sec", "sknt", "vsby", "tmpc") if c in train.columns]
    out = apply_to
    for target in LATENT_PLANNING:
        if target not in train.columns:
            continue
        labeled = train.select([*nums, target]).drop_nulls()
        if labeled.height < 5 or not nums:
            continue
        x = labeled.select(nums).fill_null(0).to_numpy()
        y = labeled[target].to_numpy()
        model = Ridge(alpha=5.0)
        model.fit(x, y)
        xa = out.select(nums).fill_null(0).to_numpy()
        hat = model.predict(xa)
        out = out.with_columns(pl.Series(target, hat), pl.Series(f"{target}_hat", hat))
    return derive_planning_from_latents(out)
