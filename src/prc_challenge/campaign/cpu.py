# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""CPU campaign jobs. Isolated, resumable, V4 unused, AOBT quarantined."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prc_challenge.bundles import E37_FIT_MONTHS, E37_TUNE_MONTHS, matched_booster_frame
from prc_challenge.features import LOCATION_CATS

RANKING_IDENTITY_EXTRAS = tuple(
    c for c in LOCATION_CATS if c not in ("AIRPORT", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt")
)
from prc_challenge.campaign.data import (
    E34B,
    T31,
    T6,
    choose_base,
    feature_lists,
    load_base_preds,
    load_cache,
    load_cache_columns,
    load_parent,
    matched_train_val,
)
from prc_challenge.campaign.data_check import hydrate_from_drive, locate
from prc_challenge.campaign.f3 import f3_numeric_columns, load_or_build_f3
from prc_challenge.campaign.models import (
    fit_classifier,
    fit_regressor,
    month_blocked_oof,
    numeric_matrix,
    predict_classifier,
    predict_regressor,
    predict_unclipped,
    to_pool,
)
from prc_challenge.campaign.runtime import (
    CampaignConfig,
    campaign_job_catalog,
    inventory_status,
    extra_view_exception,
    persist_dir,
    persist_repo_file,
    print_campaign_status,
    prior_view_metrics,
    is_terminal_skip,
    run_isolated,
    sync_index_from_job_dirs,
    write_campaign_summary,
)
from prc_challenge.campaign.scoring import (
    assert_no_locked_months,
    assert_prediction_frame,
    assert_protected_equal,
    compare_to_base,
    compose_with_parent,
    oof_is_held_out,
    compact_score,
    promote_predictions,
    required_sse_reductions,
    score_and_write,
    slice_table,
)
from prc_challenge.metrics import rmse


def _score_composed(
    cfg: CampaignConfig,
    view: str,
    val: pl.DataFrame,
    pred: np.ndarray,
    dest: Path,
    tag: str | None = None,
) -> dict[str, Any]:
    parent = load_parent(cfg, view)
    try:
        base = load_base_preds(cfg, view)
    except FileNotFoundError:
        base = None
    return score_and_write(val, pred, dest, view=view, parent=parent, base=base, tag=tag)


def _write_pred_table(path: Path, df: pl.DataFrame, pred: np.ndarray) -> None:
    keep = [c for c in ("MVT_ID_mvt", "AIRPORT", "TAXITIME_SEC_mvt", "month") if c in df.columns]
    df.select(keep).with_columns(pl.Series("prediction", np.asarray(pred, dtype=np.float64))).write_parquet(path)


def _primary_view(cfg: CampaignConfig) -> str:
    return "V1" if "V1" in cfg.views else cfg.views[0]


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _checkpoint_job(cfg: CampaignConfig, dest: Path, name: str, payload: dict[str, Any]) -> None:
    path = dest / name
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    persist_repo_file(cfg, path)


def _compact_score(scored: dict[str, Any]) -> dict[str, Any]:
    return compact_score(scored)


def _pred_array_for(val: pl.DataFrame, path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    tbl = pl.read_parquet(path)
    if "MVT_ID_mvt" not in tbl.columns or "prediction" not in tbl.columns:
        return None
    joined = val.select("MVT_ID_mvt").join(tbl.select(["MVT_ID_mvt", "prediction"]), on="MVT_ID_mvt", how="left")
    if joined["prediction"].null_count():
        return None
    return joined["prediction"].to_numpy()


def _missing_other_view_preds(dest: Path, cfg: CampaignConfig, primary: str) -> list[str]:
    return [v for v in cfg.views if v != primary and not (dest / f"predictions_{v}.parquet").exists()]


def _hydrate_extra_view(
    cfg: CampaignConfig,
    dest: Path,
    df: pl.DataFrame,
    view: str,
    *,
    tag: str | None = None,
) -> dict[str, Any]:
    """Keep composed RMSE after a reconnect. Prefer prior metrics; else re-score the parquet (no refit)."""
    prior = prior_view_metrics(dest, view)
    if isinstance(prior, dict) and prior.get("rmse_seconds") is not None:
        return compact_score(prior)
    path = dest / f"predictions_{view}.parquet"
    if not path.exists():
        return {"status": "failed", "error": f"{view} extra-view parquet does not cover the split"}
    _, val = matched_train_val(df, view)
    pred = _pred_array_for(val, path)
    if pred is None or val.height == 0:
        return {"status": "failed", "error": f"{view} extra-view parquet does not cover the split"}
    try:
        return compact_score(_score_composed(cfg, view, val, pred, dest, tag=tag))
    except Exception as exc:
        print(f"extra view {view} resume re-score failed: {exc!r}", flush=True)
        return {"status": "failed", "error": repr(exc)}


def _resumed_extra_views(
    cfg: CampaignConfig,
    dest: Path,
    df: pl.DataFrame,
    primary: str,
    *,
    tag: str | None = None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for view in cfg.views:
        if view == primary:
            continue
        extra[view] = _hydrate_extra_view(cfg, dest, df, view, tag=tag)
        print(f"extra view {view} resume rmse={extra[view].get('rmse_seconds')}", flush=True)
    return extra


def _score_predict_other_views(
    cfg: CampaignConfig,
    dest: Path,
    df: pl.DataFrame,
    *,
    primary: str,
    tag: str | None = None,
    predict_fn,
) -> dict[str, Any]:
    """Score `predict_fn(train, val, view)` on official views other than the V1 screen."""
    extra: dict[str, Any] = {}
    for view in cfg.views:
        if view == primary:
            continue
        if (dest / f"predictions_{view}.parquet").exists() and not cfg.force_rerun:
            extra[view] = _hydrate_extra_view(cfg, dest, df, view, tag=tag)
            print(f"extra view {view} resume rmse={extra[view].get('rmse_seconds')}", flush=True)
            continue
        train, val = matched_train_val(df, view)
        if train.height < 20 or val.height == 0:
            extra[view] = {"status": "skipped", "reason": "empty split"}
            continue
        try:
            pred = predict_fn(train, val, view)
            scored = _score_composed(cfg, view, val, np.asarray(pred, dtype=np.float64), dest, tag=tag)
            extra[view] = _compact_score(scored)
            persist_repo_file(cfg, dest / f"predictions_{view}.parquet")
            print(f"extra view {view} rmse={extra[view].get('rmse_seconds')}", flush=True)
        except Exception as exc:
            print(f"extra view {view} failed: {exc!r}", flush=True)
            extra[view] = extra_view_exception(exc, view=view)
    return extra


def _refit_score_other_views(
    cfg: CampaignConfig,
    dest: Path,
    df: pl.DataFrame,
    cats: list[str],
    nums: list[str],
    *,
    primary: str,
    tag: str | None = None,
) -> dict[str, Any]:
    """Independent per-view CatBoost fit for official views other than the V1 ablation screen."""
    cats = [c for c in cats if c in df.columns]
    nums = [c for c in nums if c in df.columns]

    def _predict(tr: pl.DataFrame, va: pl.DataFrame, _view: str) -> np.ndarray:
        model = fit_regressor(cfg, tr, cats, nums)
        return predict_regressor(model, va, cats, nums)

    return _score_predict_other_views(cfg, dest, df, primary=primary, tag=tag, predict_fn=_predict)


def job_cpu00_t31(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    df = load_cache(cfg)
    comparison: dict[str, Any] = {}
    for model_name, mapping in (("t31", T31), ("t6", T6), ("e34b", E34B)):
        comparison[model_name] = {}
        for view, rel in mapping.items():
            path = locate(rel, cfg)
            if path is None:
                comparison[model_name][view] = {"status": "missing"}
                continue
            _, val = matched_train_val(df, view)
            pred = pl.read_parquet(path)
            val = val.join(pred.select(["MVT_ID_mvt", "prediction"]), on="MVT_ID_mvt", how="left")
            if val["prediction"].null_count():
                comparison[model_name][view] = {
                    "status": "id_mismatch",
                    "null_preds": int(val["prediction"].null_count()),
                }
                continue
            comparison[model_name][view] = _score_composed(
                cfg, view, val, val["prediction"].to_numpy(), dest, tag=f"{model_name}"
            )
            metrics_sidecar = path.with_name("metrics.json")
            if metrics_sidecar.exists():
                comparison[model_name][view]["sidecar"] = json.loads(metrics_sidecar.read_text())
    base = choose_base(cfg)
    name = "t31" if "t31" in str(base.get("V1", "")) else "t6"
    v1 = ((comparison.get(name) or {}).get("V1") or {})
    keep = (
        "rmse_seconds",
        "sse",
        "row_count",
        "status",
        "delta_rmse_vs_e33b",
        "ordinary",
        "matched_long",
        "sse_cells",
        "lfpg_unmatched_extraday0_y_ge_2h",
        "vs_base",
        "airports",
        "top100_sse_share",
        "top1000_sse_share",
    )
    slim = {
        m: {vw: {k: part.get(k) for k in keep if k in part or part.get("status")} for vw, part in views.items()}
        for m, views in comparison.items()
    }
    ladder = None
    if v1.get("sse") and v1.get("row_count"):
        ladder = required_sse_reductions(
            sse=float(v1["sse"]), n=int(v1["row_count"]), current_rmse=v1.get("rmse_seconds")
        )
    if name in ("t31", "t6"):
        for view in cfg.views:
            promote_predictions(dest, view, name)
    return {
        "status": "completed",
        "hypothesis": "collect outstanding t31 official views and pick BASE",
        "model": name,
        "feature_bundle": "e08b",
        "base_model": name,
        "base_paths": {k: str(v) for k, v in base.items()},
        "comparison": slim,
        "rmse_seconds": v1.get("rmse_seconds"),
        "sse": v1.get("sse"),
        "sse_cells": v1.get("sse_cells"),
        "sse_ladder_q12": ladder,
        "conclusion": f"BASE={name}; V3 t31 {'ready' if locate(T31['V3'], cfg) else 'pending'}",
    }


def optuna_basin_saturated(
    trials: list[Any],
    *,
    min_complete: int = 32,
    window: int = 8,
    spread: float = 0.6,
) -> bool:
    """True when recent complete trials collapsed into one HP family (handoff: stop early)."""
    complete = [
        t
        for t in trials
        if getattr(getattr(t, "state", None), "name", None) == "COMPLETE" and getattr(t, "value", None) is not None
    ]
    if len(complete) < min_complete:
        return False
    recent = sorted(complete, key=lambda t: int(t.number))[-window:]
    if len(recent) < window:
        return False
    values = [float(t.value) for t in recent]
    if max(values) - min(values) > spread:
        return False
    families = {(t.params.get("bootstrap_type"), t.params.get("depth")) for t in recent}
    return len(families) == 1


def job_cpu01_e34_optuna(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    from catboost import CatBoostRegressor

    from prc_challenge.bundles import e34_feature_lists

    df = load_cache(cfg)
    assert_no_locked_months(df)
    cats, nums = e34_feature_lists()
    frame = matched_booster_frame(df)
    fit = frame.filter(pl.col("month").is_in(list(E37_FIT_MONTHS)))
    tune = frame.filter(pl.col("month").is_in(list(E37_TUNE_MONTHS)))
    if fit.height == 0 or tune.height == 0:
        return {"status": "skipped", "reason": "empty internal split", "conclusion": "PARK"}
    y_tune = tune["TAXITIME_SEC_mvt"].to_numpy()
    fit_pool = to_pool(fit, cats, nums, label="TAXITIME_SEC_mvt")
    tune_pool = to_pool(tune, cats, nums, label="TAXITIME_SEC_mvt")
    del df, frame, fit, tune

    import optuna

    def objective(trial: optuna.Trial) -> float:
        bootstrap = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": trial.suggest_int("iterations", 80 if cfg.smoke else 300, 200 if cfg.smoke else 2500),
            "depth": trial.suggest_int("depth", 4, 6 if cfg.smoke else 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03 if cfg.smoke else 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "border_count": trial.suggest_categorical(
                "border_count", [32, 64, 128, 254] if not cfg.smoke else [64, 128]
            ),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),
            "random_seed": cfg.seed,
            "thread_count": cfg.thread_count,
            "verbose": False,
            "allow_writing_files": False,
            "od_type": "Iter",
            "od_wait": 20 if cfg.smoke else 50,
            "use_best_model": True,
            "bootstrap_type": bootstrap,
        }
        if bootstrap == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        class _PruneCallback:
            def __init__(self) -> None:
                self.pruned = False

            def after_iteration(self, info) -> bool:
                metrics = getattr(info, "metrics", None)
                iteration = int(getattr(info, "iteration", 0))
                val = None
                if isinstance(metrics, dict):
                    block = metrics.get("validation") or metrics.get("test") or {}
                    if isinstance(block, dict) and "RMSE" in block and block["RMSE"]:
                        val = block["RMSE"][-1]
                if val is None:
                    return True
                trial.report(float(val), step=iteration)
                if trial.should_prune():
                    self.pruned = True
                    return False
                return True

        cb = _PruneCallback()
        model = CatBoostRegressor(**params)
        model.fit(fit_pool, eval_set=tune_pool, callbacks=[cb])
        if cb.pruned:
            raise optuna.TrialPruned()
        pred = np.clip(np.asarray(model.predict(tune_pool), dtype=np.float64), 0, None)
        trial.set_user_attr("best_iteration", int(model.get_best_iteration() or params["iterations"]))
        return float(rmse(y_tune, pred))

    n_startup = 4 if cfg.smoke else 24
    storage = f"sqlite:///{dest / 'study_e34.db'}"
    study = optuna.create_study(
        direction="minimize",
        study_name="e37-e34-campaign",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.seed, n_startup_trials=min(n_startup, cfg.optuna_trials)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=20 if cfg.smoke else 80),
    )
    if not any(t.state.name in {"COMPLETE", "PRUNED", "RUNNING"} for t in study.trials):
        study.enqueue_trial(
            {
                "iterations": 400 if not cfg.smoke else 80,
                "depth": 6,
                "learning_rate": 0.08,
                "l2_leaf_reg": 3.0,
                "random_strength": 1.0,
                "border_count": 254 if not cfg.smoke else 128,
                "rsm": 1.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 1.0,
            }
        )

    def _dump_best(*, stopped_early: bool = False, stop_reason: str | None = None) -> dict[str, Any]:
        try:
            best_trial = study.best_trial
        except ValueError:
            best_trial = None
        payload = {
            "bundle": "e34",
            "best_value": None if best_trial is None else best_trial.value,
            "best_params": {} if best_trial is None else best_trial.params,
            "best_trial": None if best_trial is None else best_trial.number,
            "n_trials": len(study.trials),
            "best_iteration": None if best_trial is None else best_trial.user_attrs.get("best_iteration"),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
        }
        (dest / "best_e34.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
        return payload

    def _after_trial(study_cb, _trial) -> None:
        _dump_best()
        persist_repo_file(cfg, dest / "best_e34.json")
        persist_repo_file(cfg, dest / "study_e34.db")
        if cfg.smoke:
            return
        if optuna_basin_saturated(study_cb.trials):
            print("CPU-01 TPE basin saturated; stopping remaining trials", flush=True)
            _dump_best(stopped_early=True, stop_reason="TPE basin saturated")
            persist_repo_file(cfg, dest / "best_e34.json")
            persist_repo_file(cfg, dest / "study_e34.db")
            study_cb.stop()

    already = len([t for t in study.trials if t.state.name in {"COMPLETE", "PRUNED"}])
    remaining = max(0, cfg.optuna_trials - already)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True, callbacks=[_after_trial])
    prior = json.loads((dest / "best_e34.json").read_text()) if (dest / "best_e34.json").exists() else {}
    best = _dump_best(
        stopped_early=bool(prior.get("stopped_early")),
        stop_reason=prior.get("stop_reason"),
    )
    return {
        "status": "completed",
        "rmse_seconds": best["best_value"],
        "n_trials": best["n_trials"],
        "best_trial": best["best_trial"],
        "stopped_early": best.get("stopped_early"),
        "stop_reason": best.get("stop_reason"),
        "conclusion": "internal-split only; shortlist next on V1/V2/V3",
        "hypothesis": "E34 bunching + tuned CatBoost stacks",
    }


def _optuna_shortlist_payloads(optuna_dir: Path, n: int = 3) -> list[dict[str, Any]]:
    best_path = optuna_dir / "best_e34.json"
    payloads: list[dict[str, Any]] = []
    study_db = optuna_dir / "study_e34.db"
    if study_db.exists():
        import optuna

        study = optuna.load_study(study_name="e37-e34-campaign", storage=f"sqlite:///{study_db}")
        complete = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
        complete.sort(key=lambda t: t.value)
        for trial in complete[:n]:
            payloads.append(
                {
                    "trial": trial.number,
                    "best_value": trial.value,
                    "best_params": trial.params,
                    "best_iteration": trial.user_attrs.get("best_iteration"),
                }
            )
    if payloads:
        return payloads
    if best_path.exists():
        payload = json.loads(best_path.read_text())
        return [
            {
                "trial": payload.get("best_trial"),
                "best_value": payload.get("best_value"),
                "best_params": payload.get("best_params") or {},
                "best_iteration": payload.get("best_iteration"),
            }
        ]
    return []


def job_cpu01b_shortlist(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    optuna_dir = cfg.artifact_root / "CPU-01-e34-optuna"
    shortlist = _optuna_shortlist_payloads(optuna_dir, n=1 if cfg.smoke else 3)
    if not shortlist:
        prior_path = optuna_dir / "metrics.json"
        prior: dict[str, Any] = {}
        if prior_path.exists():
            try:
                prior = json.loads(prior_path.read_text())
            except json.JSONDecodeError:
                prior = {}
        if is_terminal_skip(prior):
            return {
                "status": "skipped",
                "reason": "CPU-01 parked; E34 shortlist is not justified",
                "conclusion": "PARK",
            }
        return {"status": "skipped", "reason": "CPU-01 not complete", "conclusion": "PARK"}
    from prc_challenge.bundles import e34_feature_lists, optuna_winner_params

    df = load_cache(cfg)
    cats, nums = e34_feature_lists()
    progress = {} if cfg.force_rerun else _load_json_dict(dest / "shortlist_progress.json")
    by_trial: dict[str, Any] = {}
    best_v1 = None
    winner_views: dict[str, Any] = {}
    winner_tag: str | None = None
    for item in shortlist:
        params = optuna_winner_params(
            {"params": item.get("best_params") or {}, "best_iteration": item.get("best_iteration")},
            thread_count=cfg.thread_count,
        )
        if cfg.smoke:
            params["iterations"] = min(int(params.get("iterations", 80)), 80)
        trial_key = str(item.get("trial"))
        tag = f"t{item.get('trial')}"
        slot = progress.setdefault(trial_key, {})
        views_slot = slot.setdefault("views", {})
        view_metrics: dict[str, Any] = {}
        fit_kwargs = {k: v for k, v in params.items() if k != "iterations"} | {"iterations": params["iterations"]}
        for view in cfg.views:
            prior = views_slot.get(view) if isinstance(views_slot.get(view), dict) else None
            tagged = dest / f"predictions_{tag}_{view}.parquet"
            if prior and prior.get("rmse_seconds") is not None and tagged.exists() and not cfg.force_rerun:
                print(f"CPU-01b resume skip {tag} {view} rmse={prior['rmse_seconds']}", flush=True)
                view_metrics[view] = prior
                continue
            train, val = matched_train_val(df, view)
            model = fit_regressor(cfg, train, cats, nums, **fit_kwargs)
            pred = predict_regressor(model, val, cats, nums)
            scored = _score_composed(cfg, view, val, pred, dest, tag=tag)
            compact = _compact_score(scored)
            views_slot[view] = compact
            view_metrics[view] = compact
            _checkpoint_job(cfg, dest, "shortlist_progress.json", progress)
        by_trial[trial_key] = {
            "internal_rmse": item.get("best_value"),
            "views": {k: {"rmse_seconds": v.get("rmse_seconds"), "sse": v.get("sse")} for k, v in view_metrics.items()},
        }
        v1_rmse = (view_metrics.get("V1") or {}).get("rmse_seconds")
        if v1_rmse is not None and (best_v1 is None or v1_rmse < best_v1):
            best_v1 = v1_rmse
            winner_views = view_metrics
            winner_tag = tag
        frame = matched_booster_frame(df)
        fit = frame.filter(pl.col("month").is_in(list(E37_FIT_MONTHS)))
        tune = frame.filter(pl.col("month").is_in(list(E37_TUNE_MONTHS)))
        tune_path = dest / f"tune_predictions_{tag}.parquet"
        if fit.height and tune.height and (cfg.force_rerun or not tune_path.exists()):
            tune_model = fit_regressor(cfg, fit, cats, nums, **fit_kwargs)
            _write_pred_table(tune_path, tune, predict_regressor(tune_model, tune, cats, nums))
            persist_repo_file(cfg, tune_path)
    if winner_tag:
        for view in winner_views:
            promote_predictions(dest, view, winner_tag)
        winner_tune = dest / f"tune_predictions_{winner_tag}.parquet"
        if winner_tune.exists():
            pl.read_parquet(winner_tune).write_parquet(dest / "tune_predictions.parquet")
    v1 = winner_views.get("V1", {})
    return {
        "status": "completed",
        "rmse_seconds": v1.get("rmse_seconds"),
        "delta_sse_vs_base": (v1.get("vs_base") or {}).get("delta_sse_vs_base"),
        "shortlist": by_trial,
        "views": winner_views,
        "conclusion": "ENSEMBLE_CANDIDATE" if v1.get("rmse_seconds") else "PARK",
        "hypothesis": "score E34 Optuna shortlist (top-3 internal) on official views",
    }


def job_cpu02_experts(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    df = load_cache(cfg)
    cats, nums = feature_lists("e08b")
    view = _primary_view(cfg)
    train, val = matched_train_val(df, view)
    highlighted = ["LIRF", "EGLL", "LFPG", "LTFM"]
    counts = train.group_by("AIRPORT").len().rename({"len": "n"})
    min_n = 80 if cfg.smoke else 200
    airports = [a for a in counts.filter(pl.col("n") >= min_n)["AIRPORT"].to_list()]
    for ap in highlighted:
        if ap not in airports and bool((train["AIRPORT"] == ap).any()):
            airports.append(ap)
    progress = {} if cfg.force_rerun else _load_json_dict(dest / "experts_progress.json")
    goof_path = dest / "goof.parquet"
    gpred_path = dest / "gpred.parquet"
    gpred = _pred_array_for(val, gpred_path) if not cfg.force_rerun else None
    goof = pl.read_parquet(goof_path) if goof_path.exists() and not cfg.force_rerun else None
    if gpred is not None and goof is not None and progress.get("global"):
        print("CPU-02 resume skip global CatBoost + OOF", flush=True)
    else:
        global_model = fit_regressor(cfg, train, cats, nums)
        gpred = predict_regressor(global_model, val, cats, nums)
        goof = month_blocked_oof(
            cfg, train, cats, nums, shard_dir=dest, shard_prefix="global_oof_month"
        )
        _write_pred_table(gpred_path, val, gpred)
        goof.write_parquet(goof_path)
        progress["global"] = True
        _checkpoint_job(cfg, dest, "experts_progress.json", progress)
    oof_w: dict[str, float] = {}
    prior_w = progress.get("airports") if isinstance(progress.get("airports"), dict) else {}
    pred_a = _pred_array_for(val, dest / "pred_a.parquet") if not cfg.force_rerun else None
    pred_c = _pred_array_for(val, dest / "pred_c.parquet") if not cfg.force_rerun else None
    if pred_a is None:
        pred_a = gpred.copy()
    if pred_c is None:
        pred_c = gpred.copy()
    oof_c = goof.select(["MVT_ID_mvt", "oof_prediction"])
    oof_done = dest / "oof_predictions.parquet"
    if oof_done.exists() and not cfg.force_rerun:
        oof_tbl = pl.read_parquet(oof_done)
        pred_col = "prediction" if "prediction" in oof_tbl.columns else "oof_prediction"
        if pred_col in oof_tbl.columns:
            oof_c = oof_tbl.select(["MVT_ID_mvt", pl.col(pred_col).alias("oof_prediction")])
    for ap in airports:
        if ap in prior_w and not cfg.force_rerun:
            oof_w[ap] = float(prior_w[ap])
            print(f"CPU-02 resume skip airport={ap} w={oof_w[ap]}", flush=True)
            continue
        tr = train.filter(pl.col("AIRPORT") == ap)
        va = val.filter(pl.col("AIRPORT") == ap)
        if tr.height < min_n or va.height == 0:
            oof_w[ap] = 0.0
            prior_w[ap] = 0.0
            progress["airports"] = prior_w
            _checkpoint_job(cfg, dest, "experts_progress.json", progress)
            continue
        local = fit_regressor(cfg, tr, cats, nums)
        lp = predict_regressor(local, va, cats, nums)
        idx = (val["AIRPORT"] == ap).to_numpy()
        pred_a[idx] = lp
        oof = month_blocked_oof(
            cfg, tr, cats, nums, shard_dir=dest, shard_prefix=f"ap_{ap}_oof_month"
        )
        merged = oof.join(goof.select(["MVT_ID_mvt", pl.col("oof_prediction").alias("g")]), on="MVT_ID_mvt")
        y = merged["TAXITIME_SEC_mvt"].to_numpy()
        a = merged["oof_prediction"].to_numpy()
        g = merged["g"].to_numpy()
        best_w, best_rmse = 0.0, rmse(y, g)
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            mix = w * a + (1 - w) * g
            r = rmse(y, mix)
            if r < best_rmse:
                best_rmse, best_w = r, w
        oof_w[ap] = best_w
        pred_c[idx] = best_w * lp + (1 - best_w) * gpred[idx]
        mix = best_w * merged["oof_prediction"].to_numpy() + (1 - best_w) * merged["g"].to_numpy()
        mix_df = merged.select("MVT_ID_mvt").with_columns(pl.Series("oof_prediction", mix))
        oof_c = (
            oof_c.join(mix_df.rename({"oof_prediction": "_mix"}), on="MVT_ID_mvt", how="left")
            .with_columns(pl.coalesce(pl.col("_mix"), pl.col("oof_prediction")).alias("oof_prediction"))
            .drop("_mix")
        )
        prior_w[ap] = best_w
        progress["airports"] = prior_w
        _write_pred_table(dest / "pred_a.parquet", val, pred_a)
        _write_pred_table(dest / "pred_c.parquet", val, pred_c)
        aligned = goof.select("MVT_ID_mvt").join(oof_c, on="MVT_ID_mvt", how="left")
        _write_pred_table(
            dest / "oof_predictions.parquet",
            goof,
            aligned["oof_prediction"].fill_null(0).to_numpy(),
        )
        _checkpoint_job(cfg, dest, "experts_progress.json", progress)
    (dest / "airport_weights.json").write_text(json.dumps(oof_w, indent=2) + "\n")
    aligned = goof.select("MVT_ID_mvt").join(oof_c, on="MVT_ID_mvt", how="left")
    _write_pred_table(
        dest / "oof_predictions.parquet",
        goof,
        aligned["oof_prediction"].fill_null(0).to_numpy(),
    )
    strategies = {
        "A_airport": _score_composed(cfg, view, val, pred_a, dest, tag="A"),
        "B_global": _score_composed(cfg, view, val, gpred, dest, tag="B"),
        "C_shrinkage": _score_composed(cfg, view, val, pred_c, dest, tag="C"),
    }
    best_tag = min(strategies, key=lambda k: strategies[k]["rmse_seconds"])
    tag_map = {"A_airport": "A", "B_global": "B", "C_shrinkage": "C"}
    promote_predictions(dest, view, tag_map[best_tag])
    composed = strategies[best_tag]
    tag = tag_map[best_tag]

    def _predict_experts(tr: pl.DataFrame, va: pl.DataFrame, _other: str) -> np.ndarray:
        gm = fit_regressor(cfg, tr, cats, nums)
        gp = predict_regressor(gm, va, cats, nums)
        if best_tag == "B_global":
            return gp
        out = gp.copy()
        for ap, w in oof_w.items():
            tr_ap = tr.filter(pl.col("AIRPORT") == ap)
            idx = (va["AIRPORT"] == ap).to_numpy()
            if tr_ap.height < min_n or int(idx.sum()) == 0:
                continue
            local = fit_regressor(cfg, tr_ap, cats, nums)
            lp = predict_regressor(local, va.filter(pl.col("AIRPORT") == ap), cats, nums)
            if best_tag == "A_airport":
                out[idx] = lp
            else:
                out[idx] = w * lp + (1.0 - w) * gp[idx]
        return out

    extra_views: dict[str, Any] = {}
    if _missing_other_view_preds(dest, cfg, view) or cfg.force_rerun:
        extra_views = _score_predict_other_views(cfg, dest, df, primary=view, tag=tag, predict_fn=_predict_experts)
    else:
        print("CPU-02 resume skip extra-view refits", flush=True)
        extra_views = _resumed_extra_views(cfg, dest, df, view, tag=tag)
    return {
        "status": "completed",
        "rmse_seconds": composed["rmse_seconds"],
        "delta_sse_vs_base": (composed.get("vs_base") or {}).get("delta_sse_vs_base"),
        "weights_oof": oof_w,
        "strategies": {k: {"rmse_seconds": v.get("rmse_seconds"), "sse": v.get("sse")} for k, v in strategies.items()},
        "views": {view: {k: composed.get(k) for k in ("rmse_seconds", "sse", "vs_base")}, **extra_views},
        "highlighted_airports": highlighted,
        "airports_fit": airports,
        "conclusion": "A/B/C compared; weights from OOF only",
        "hypothesis": "airport heterogeneity is under-modeled by one global CatBoost",
    }


def job_cpu03_error(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    from prc_challenge.campaign.data import load_cache_columns
    from prc_challenge.campaign.models import load_or_build_base_residual

    train, val, val_base, view = load_or_build_base_residual(cfg, extra_cols=RANKING_IDENTITY_EXTRAS)
    cats, nums = feature_lists("e08b")
    cats = [c for c in cats if c in train.columns]
    nums = [c for c in nums if c in train.columns]
    train = train.with_columns(
        ((pl.col("TAXITIME_SEC_mvt") - pl.col("oof_prediction")) ** 2).log1p().alias("log1p_sq")
    )
    fit = train.filter(pl.col("month").is_in(list(E37_FIT_MONTHS)))
    tune = train.filter(pl.col("month").is_in(list(E37_TUNE_MONTHS)))
    if fit.height < 50 or tune.height < 20:
        fit, tune = train, train
    model_sel = fit_regressor(cfg, fit, cats, nums, label="log1p_sq")
    risk_fit = predict_regressor(model_sel, fit, cats, nums)
    risk_tune = predict_regressor(model_sel, tune, cats, nums)
    pl.concat(
        [
            fit.select(["MVT_ID_mvt"]).with_columns(pl.Series("pred_risk", risk_fit)),
            tune.select(["MVT_ID_mvt"]).with_columns(pl.Series("pred_risk", risk_tune)),
        ]
    ).unique(subset=["MVT_ID_mvt"], keep="last").write_parquet(dest / "risk_train.parquet")
    persist_repo_file(cfg, dest / "risk_train.parquet")
    model = fit_regressor(cfg, train, cats, nums, label="log1p_sq")
    val_sq = np.log1p((val["TAXITIME_SEC_mvt"].to_numpy() - val_base) ** 2)
    risk_val = predict_regressor(model, val, cats, nums)
    corr = float(np.corrcoef(risk_val, val_sq)[0, 1]) if val.height > 1 else None
    val = val.with_columns(pl.Series("pred_risk", risk_val), pl.Series("log1p_sq", val_sq))
    try:
        q = val.with_columns(pl.col("pred_risk").qcut(10, labels=[str(i) for i in range(10)]).alias("decile"))
    except Exception:
        q = val.with_columns(pl.lit("0").alias("decile"))
    deciles = []
    for d, part in q.group_by("decile", maintain_order=True):
        yy = part["TAXITIME_SEC_mvt"].to_numpy()
        deciles.append(
            {
                "decile": d[0] if isinstance(d, tuple) else d,
                "n": part.height,
                "rmse": rmse(yy, part["oof_prediction"].to_numpy()),
                "sse": float(((yy - part["oof_prediction"].to_numpy()) ** 2).sum()),
            }
        )
    pl.DataFrame(deciles).write_parquet(dest / "error_slices.parquet")
    train.select(["MVT_ID_mvt", "oof_prediction", "TAXITIME_SEC_mvt"]).write_parquet(dest / "oof_predictions.parquet")
    val.select(["MVT_ID_mvt"]).with_columns(pl.Series("pred_risk", risk_val)).write_parquet(dest / "risk_predictions.parquet")
    importance = pl.DataFrame(
        {"feature": cats + nums, "importance": model.get_feature_importance()}
    ).sort("importance", descending=True)
    importance.write_parquet(dest / "feature_importance.parquet")
    shap_top: list[dict[str, Any]] = []
    try:
        from catboost import EFstrType

        n_shap = min(1_500 if not cfg.smoke else 80, train.height)
        sample = train.sample(n=n_shap, seed=cfg.seed, shuffle=True)
        shap = np.asarray(
            model.get_feature_importance(to_pool(sample, cats, nums), type=EFstrType.ShapValues)
        )
        mean_abs = np.abs(shap[:, :-1]).mean(axis=0)
        shap_top = (
            pl.DataFrame({"feature": cats + nums, "mean_abs_shap": mean_abs})
            .sort("mean_abs_shap", descending=True)
            .head(15)
            .to_dicts()
        )
        pl.DataFrame(shap_top).write_parquet(dest / "shap_top.parquet")
    except Exception as exc:
        shap_top = [{"error": repr(exc)}]
    lgb_corr = None
    try:
        import lightgbm as lgb

        booster = lgb.LGBMRegressor(
            n_estimators=80 if cfg.smoke else 300,
            random_state=cfg.seed,
            n_jobs=cfg.thread_count,
            verbose=-1,
        )
        booster.fit(numeric_matrix(train, nums), train["log1p_sq"].to_numpy())
        lgb_risk = booster.predict(numeric_matrix(val, nums))
        lgb_corr = float(np.corrcoef(lgb_risk, val_sq)[0, 1])
    except Exception as exc:
        lgb_corr = f"failed:{exc!r}"
    persist_dir(cfg, dest)
    extra_views: dict[str, Any] = {}
    cache = load_cache_columns(cfg, extra=list(cats) + list(nums))
    for other in cfg.views:
        if other == view:
            continue
        if (dest / f"risk_predictions_{other}.parquet").exists() and not cfg.force_rerun:
            prior = prior_view_metrics(dest, other)
            if prior is not None:
                extra_views[other] = prior
                print(f"CPU-03 resume skip extra view {other} corr={extra_views[other].get('risk_corr_val')}", flush=True)
                continue
        try:
            _, va = matched_train_val(cache, other)
            official = load_base_preds(cfg, other).select(
                ["MVT_ID_mvt", pl.col("prediction").alias("base_prediction")]
            )
            va = va.join(official, on="MVT_ID_mvt", how="inner")
            if va.height < 20:
                extra_views[other] = {"status": "skipped", "reason": "empty split"}
                continue
            y_o = va["TAXITIME_SEC_mvt"].to_numpy()
            sq_o = np.log1p((y_o - va["base_prediction"].to_numpy()) ** 2)
            risk_o = predict_regressor(model, va, cats, nums)
            corr_o = float(np.corrcoef(risk_o, sq_o)[0, 1]) if va.height > 1 else None
            extra_views[other] = {"risk_corr_val": corr_o, "n": va.height}
            print(f"extra view {other} risk_corr={corr_o}", flush=True)
            va.select(["MVT_ID_mvt"]).with_columns(pl.Series("pred_risk", risk_o)).write_parquet(
                dest / f"risk_predictions_{other}.parquet"
            )
            persist_repo_file(cfg, dest / f"risk_predictions_{other}.parquet")
        except Exception as exc:
            print(f"extra view {other} failed: {exc!r}", flush=True)
            extra_views[other] = extra_view_exception(exc, view=other)
    return {
        "status": "completed",
        "rmse_seconds": None,
        "risk_corr_val": corr,
        "risk_corr_lgb_val": lgb_corr,
        "deciles": deciles,
        "top_features": importance.head(15).to_dicts(),
        "shap_top": shap_top,
        "views": {view: {"risk_corr_val": corr}, **extra_views},
        "conclusion": (
            "error is predictable on val; risk_train uses a fit-month model so September is held out"
            if corr and corr > 0.15
            else "mostly irreducible on current features; risk_train still held out on E37 tune month"
        ),
        "hypothesis": "BASE squared error is predictable from ranking-safe inputs",
    }


def job_cpu04_f3(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    df = load_or_build_f3(cfg, load_cache(cfg))
    cats, nums0 = feature_lists("e08b")
    f3_nums = [c for c in f3_numeric_columns() if c in df.columns]
    view = _primary_view(cfg)
    train, val = matched_train_val(df, view)
    ablations = {
        "BASE": nums0,
        "BASE+bunching": nums0 + [c for c in f3_nums if "delayed" in c],
        "BASE+traffic": nums0 + [c for c in f3_nums if c.startswith("n_dep_") or c.startswith("n_arr_") or c.startswith("n_mvt_")],
        "BASE+runway": nums0 + [c for c in f3_nums if "rwy" in c],
        "BASE+burst": nums0 + [c for c in f3_nums if c.startswith("gap_") or c.startswith("burst") or c in ("sec_since_prev_dep", "n_short_gap_5")],
        "BASE+mix": nums0 + [c for c in f3_nums if c in ("n_same_type_5", "n_heavy_5")],
        "BASE+stand": nums0 + [c for c in f3_nums if "stand" in c],
        "BASE+demand": nums0 + [c for c in f3_nums if "sched" in c or c.startswith("n_dep_fwd") or c.startswith("n_arr_fwd") or "demand_vs" in c],
        "BASE+F3": nums0 + f3_nums,
    }
    ablations = {k: list(dict.fromkeys(v)) for k, v in ablations.items()}
    progress_path = dest / "ablations.json"
    table: dict[str, Any] = {}
    if progress_path.exists() and not cfg.force_rerun:
        try:
            loaded = json.loads(progress_path.read_text())
            if isinstance(loaded, dict):
                table = loaded
        except json.JSONDecodeError:
            table = {}
    best = None
    for name, nums in ablations.items():
        nums = [c for c in nums if c in train.columns]
        prior = table.get(name) if isinstance(table.get(name), dict) else None
        if prior and prior.get("rmse_seconds") is not None and not cfg.force_rerun:
            print(f"CPU-04 resume skip {name} rmse={prior['rmse_seconds']}", flush=True)
            if best is None or prior["rmse_seconds"] < best[1]:
                best = (name, prior["rmse_seconds"])
            continue
        model = fit_regressor(cfg, train, cats, nums)
        pred = predict_regressor(model, val, cats, nums)
        scored = _score_composed(cfg, view, val, pred, dest, tag=name.replace("+", "_"))
        table[name] = _compact_score(scored)
        _checkpoint_job(cfg, dest, "ablations.json", table)
        if best is None or table[name]["rmse_seconds"] < best[1]:
            best = (name, table[name]["rmse_seconds"])
    extra_views: dict[str, Any] = {}
    if best:
        promote_predictions(dest, view, best[0].replace("+", "_"))
        win_nums = [c for c in ablations[best[0]] if c in df.columns]
        fit = train.filter(pl.col("month").is_in(list(E37_FIT_MONTHS)))
        tune = train.filter(pl.col("month").is_in(list(E37_TUNE_MONTHS)))
        if fit.height and tune.height and (cfg.force_rerun or not (dest / "tune_predictions.parquet").exists()):
            tune_model = fit_regressor(cfg, fit, cats, win_nums)
            _write_pred_table(
                dest / "tune_predictions.parquet",
                tune,
                predict_regressor(tune_model, tune, cats, win_nums),
            )
            persist_repo_file(cfg, dest / "tune_predictions.parquet")
        if _missing_other_view_preds(dest, cfg, view) or cfg.force_rerun:
            extra_views = _refit_score_other_views(
                cfg,
                dest,
                df,
                cats,
                win_nums,
                primary=view,
                tag=best[0].replace("+", "_"),
            )
        else:
            print("CPU-04 resume skip extra-view refits", flush=True)
            extra_views = _resumed_extra_views(cfg, dest, df, view, tag=best[0].replace("+", "_"))
    arr_cols = [c for c in f3_nums if c.startswith("n_arr_")]
    arr_mean = float(np.mean([float(df[c].fill_null(0).mean() or 0.0) for c in arr_cols])) if arr_cols else 0.0
    dep_only = arr_mean < 0.05
    return {
        "status": "completed",
        "rmse_seconds": best[1] if best else None,
        "best_ablation": best[0] if best else None,
        "ablations": table,
        "f3_n_features": len(f3_nums),
        "cache_dep_only": dep_only,
        "n_arr_mean": arr_mean,
        "conclusion": f"best={best}"
        + ("; n_arr occupancy is ~0 (ARR sidecar missing)" if dep_only else ""),
        "hypothesis": "richer airport operational state reduces ordinary matched variance",
        "delta_sse_vs_base": (table.get(best[0], {}).get("vs_base") or {}).get("delta_sse_vs_base") if best else None,
        "views": {view: _compact_score(table.get(best[0], {})) if best else {"rmse_seconds": None}, **extra_views},
    }


def job_cpu05_route_queue(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    df = load_or_build_f3(cfg, load_cache(cfg))
    cats, nums = feature_lists("e08b")
    f3_nums = [c for c in f3_numeric_columns() if c in df.columns]
    view = _primary_view(cfg)
    train, val = matched_train_val(df, view)
    keys = ["AIRPORT", "STAND_mvt", "RUNWAY_mvt"]
    gmed = float(train["TAXITIME_SEC_mvt"].median())

    def hierarchical(frame: pl.DataFrame, stats_from: pl.DataFrame) -> pl.DataFrame:
        cell = stats_from.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_cell"))
        ap = stats_from.group_by("AIRPORT").agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_ap"))
        hod = stats_from.group_by(["AIRPORT", "local_hour"]).agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_hod")) if "local_hour" in stats_from.columns else None
        out = frame.join(cell, on=keys, how="left").join(ap, on="AIRPORT", how="left")
        if hod is not None and "local_hour" in frame.columns:
            out = out.join(hod, on=["AIRPORT", "local_hour"], how="left")
            return out.with_columns(
                pl.coalesce(pl.col("route_cell"), pl.col("route_hod"), pl.col("route_ap"), pl.lit(gmed)).alias("route_base")
            )
        return out.with_columns(pl.coalesce(pl.col("route_cell"), pl.col("route_ap"), pl.lit(gmed)).alias("route_base"))

    oof_parts = []
    for month in train["month"].unique().sort().to_list():
        fit = train.filter(pl.col("month") != month)
        hold = train.filter(pl.col("month") == month)
        if fit.height == 0 or hold.height == 0:
            continue
        oof_parts.append(hierarchical(hold, fit))
    train_oof = pl.concat(oof_parts) if oof_parts else hierarchical(train, train)
    val = hierarchical(val, train)
    train_oof = train_oof.with_columns((pl.col("TAXITIME_SEC_mvt") - pl.col("route_base")).alias("queue_residual"))
    train_oof.select(["MVT_ID_mvt", "route_base"]).write_parquet(dest / "route_oof_train.parquet")
    val.select(["MVT_ID_mvt", "route_base"]).write_parquet(dest / "route_oof_val.parquet")
    persist_repo_file(cfg, dest / "route_oof_train.parquet")
    persist_repo_file(cfg, dest / "route_oof_val.parquet")
    tune = train_oof.filter(pl.col("month").is_in(list(E37_TUNE_MONTHS)))
    fit_oof = train_oof.filter(pl.col("month").is_in(list(E37_FIT_MONTHS)))
    if fit_oof.height < 50:
        fit_oof = train_oof
    fit_train = train.filter(pl.col("month").is_in(list(E37_FIT_MONTHS)))
    if fit_train.height < 50:
        fit_train = train
    progress = {} if cfg.force_rerun else _load_json_dict(dest / "route_queue_progress.json")
    q_model = stacked = route_model = None
    q_cols = f3_nums or nums
    stacked_cols = nums + ["route_base"] + f3_nums
    if (
        isinstance(progress.get("sum"), dict)
        and progress["sum"].get("rmse_seconds") is not None
        and (dest / f"predictions_sum_{view}.parquet").exists()
        and not cfg.force_rerun
    ):
        composed = progress["sum"]
        print(f"CPU-05 resume skip sum rmse={composed['rmse_seconds']}", flush=True)
    else:
        q_sel = fit_regressor(cfg, fit_oof, cats, q_cols, label="queue_residual")
        tune_rmse_sum = None
        if tune.height:
            q_tune = predict_unclipped(q_sel, tune, cats, q_cols)
            pred_tune_sum = np.clip(tune["route_base"].to_numpy() + q_tune, 0, None)
            _write_pred_table(dest / "tune_predictions_sum.parquet", tune, pred_tune_sum)
            tune_rmse_sum = float(rmse(tune["TAXITIME_SEC_mvt"].to_numpy(), pred_tune_sum))
        q_model = fit_regressor(cfg, train_oof, cats, q_cols, label="queue_residual")
        q_hat = predict_unclipped(q_model, val, cats, q_cols)
        pred = np.clip(val["route_base"].to_numpy() + q_hat, 0, None)
        composed = _compact_score(_score_composed(cfg, view, val, pred, dest, tag="sum"))
        composed["tune_rmse"] = tune_rmse_sum
        progress["sum"] = composed
        _checkpoint_job(cfg, dest, "route_queue_progress.json", progress)
    if (
        isinstance(progress.get("stacked"), dict)
        and progress["stacked"].get("rmse_seconds") is not None
        and (dest / f"predictions_stacked_{view}.parquet").exists()
        and not cfg.force_rerun
    ):
        stacked_m = progress["stacked"]
        print(f"CPU-05 resume skip stacked rmse={stacked_m['rmse_seconds']}", flush=True)
    else:
        stacked_sel = fit_regressor(cfg, fit_oof, cats, stacked_cols)
        if tune.height:
            pred_tune_st = predict_regressor(stacked_sel, tune, cats, stacked_cols)
            _write_pred_table(dest / "tune_predictions_stacked.parquet", tune, pred_tune_st)
            tune_rmse_st = float(rmse(tune["TAXITIME_SEC_mvt"].to_numpy(), pred_tune_st))
        else:
            tune_rmse_st = None
        stacked = fit_regressor(cfg, train_oof, cats, stacked_cols)
        pred2 = predict_regressor(stacked, val, cats, stacked_cols)
        stacked_m = _compact_score(_score_composed(cfg, view, val, pred2, dest, tag="stacked"))
        stacked_m["tune_rmse"] = tune_rmse_st
        progress["stacked"] = stacked_m
        _checkpoint_job(cfg, dest, "route_queue_progress.json", progress)
    route_nums = [c for c in ("local_hour", "local_dow") if c in train.columns]
    route_cats = [c for c in ("AIRPORT", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt") if c in cats]
    if (
        isinstance(progress.get("route_cb"), dict)
        and progress["route_cb"].get("rmse_seconds") is not None
        and (dest / f"predictions_route_cb_{view}.parquet").exists()
        and not cfg.force_rerun
    ):
        cb_route_m = progress["route_cb"]
        print(f"CPU-05 resume skip route_cb rmse={cb_route_m['rmse_seconds']}", flush=True)
    else:
        route_sel = fit_regressor(cfg, fit_train, route_cats, route_nums)
        if tune.height:
            pred_tune_r = predict_regressor(route_sel, tune, route_cats, route_nums)
            _write_pred_table(dest / "tune_predictions_route_cb.parquet", tune, pred_tune_r)
            tune_rmse_r = float(rmse(tune["TAXITIME_SEC_mvt"].to_numpy(), pred_tune_r))
        else:
            tune_rmse_r = None
        route_model = fit_regressor(cfg, train, route_cats, route_nums)
        route_hat = predict_regressor(route_model, val, route_cats, route_nums)
        cb_route_m = _compact_score(_score_composed(cfg, view, val, route_hat, dest, tag="route_cb"))
        cb_route_m["tune_rmse"] = tune_rmse_r
        progress["route_cb"] = cb_route_m
        _checkpoint_job(cfg, dest, "route_queue_progress.json", progress)
    tagged = {"sum": composed, "stacked": stacked_m, "route_cb": cb_route_m}

    def _sel_rmse(part: dict[str, Any]) -> float:
        if part.get("tune_rmse") is not None:
            return float(part["tune_rmse"])
        return float(part["rmse_seconds"])

    best_tag = min(tagged, key=lambda tag: _sel_rmse(tagged[tag]))
    promote_predictions(dest, view, best_tag)
    winner_tune = dest / f"tune_predictions_{best_tag}.parquet"
    if winner_tune.exists():
        pl.read_parquet(winner_tune).write_parquet(dest / "tune_predictions.parquet")

    def _predict_winner(tr: pl.DataFrame, va: pl.DataFrame, _view: str) -> np.ndarray:
        local_med = float(tr["TAXITIME_SEC_mvt"].median())

        def hier(frame: pl.DataFrame, stats_from: pl.DataFrame) -> pl.DataFrame:
            cell = stats_from.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_cell"))
            ap = stats_from.group_by("AIRPORT").agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_ap"))
            out = frame.join(cell, on=keys, how="left").join(ap, on="AIRPORT", how="left")
            return out.with_columns(pl.coalesce(pl.col("route_cell"), pl.col("route_ap"), pl.lit(local_med)).alias("route_base"))

        parts = []
        for month in tr["month"].unique().sort().to_list():
            fit = tr.filter(pl.col("month") != month)
            hold = tr.filter(pl.col("month") == month)
            if fit.height == 0 or hold.height == 0:
                continue
            parts.append(hier(hold, fit))
        tr_oof = pl.concat(parts) if parts else hier(tr, tr)
        va_h = hier(va, tr)
        if best_tag == "sum":
            tr_oof = tr_oof.with_columns((pl.col("TAXITIME_SEC_mvt") - pl.col("route_base")).alias("queue_residual"))
            qm = fit_regressor(cfg, tr_oof, cats, q_cols, label="queue_residual")
            return np.clip(va_h["route_base"].to_numpy() + predict_unclipped(qm, va_h, cats, q_cols), 0, None)
        if best_tag == "stacked":
            sm = fit_regressor(cfg, tr_oof, cats, stacked_cols)
            return predict_regressor(sm, va_h, cats, stacked_cols)
        rm = fit_regressor(cfg, tr, route_cats, route_nums)
        return predict_regressor(rm, va, route_cats, route_nums)

    extra_views: dict[str, Any] = {}
    if _missing_other_view_preds(dest, cfg, view) or cfg.force_rerun:
        extra_views = _score_predict_other_views(cfg, dest, df, primary=view, tag=best_tag, predict_fn=_predict_winner)
    else:
        print("CPU-05 resume skip extra-view refits", flush=True)
        extra_views = _resumed_extra_views(cfg, dest, df, view, tag=best_tag)
    return {
        "status": "completed",
        "rmse_seconds": min(composed["rmse_seconds"], stacked_m["rmse_seconds"], cb_route_m["rmse_seconds"]),
        "sum_model": {k: composed.get(k) for k in ("rmse_seconds", "sse", "vs_base")},
        "stacked_model": {k: stacked_m.get(k) for k in ("rmse_seconds", "sse", "vs_base")},
        "route_catboost": {k: cb_route_m.get(k) for k in ("rmse_seconds", "sse", "vs_base")},
        "delta_sse_vs_base": (composed.get("vs_base") or {}).get("delta_sse_vs_base"),
        "views": {view: {"rmse_seconds": tagged[best_tag]["rmse_seconds"]}, **extra_views},
        "conclusion": "OOF hierarchical route + F3 queue vs stacked vs route-only CatBoost",
        "hypothesis": "separating route/base taxi from congestion improves generalization",
    }

def job_cpu06_mixture(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    df = load_cache(cfg)
    cats, nums = feature_lists("e08b")
    view = _primary_view(cfg)
    train, val = matched_train_val(df, view)
    fit = train.filter(pl.col("month").is_in(list(E37_FIT_MONTHS)))
    tune = train.filter(pl.col("month").is_in(list(E37_TUNE_MONTHS)))
    if fit.height == 0:
        fit = train
    if tune.height == 0:
        tune = train
    progress = {} if cfg.force_rerun else _load_json_dict(dest / "mixture_progress.json")
    results: dict[str, Any] = {}
    prior_th = progress.get("thresholds") if isinstance(progress.get("thresholds"), dict) else {}
    models: dict[str, Any] = {}
    for t in (2700, 3600, 7200):
        key = str(t)
        prior = prior_th.get(key) if isinstance(prior_th.get(key), dict) else None
        tagged = dest / f"predictions_mix{t}_{view}.parquet"
        if prior and prior.get("tune_rmse") is not None and tagged.exists() and not cfg.force_rerun:
            print(f"CPU-06 resume skip threshold={t} tune_rmse={prior['tune_rmse']}", flush=True)
            results[key] = prior
            tpath = dest / f"tune_predictions_mix{t}.parquet"
            if tpath.exists():
                models[key] = (None, pl.read_parquet(tpath)["prediction"].to_numpy())
            continue
        lab = (fit["TAXITIME_SEC_mvt"] > t).cast(pl.Int32).alias("is_long")
        tr = fit.with_columns(lab)
        clf = fit_classifier(cfg, tr, cats, nums, "is_long")
        normal = tr.filter(pl.col("TAXITIME_SEC_mvt") <= t)
        long = tr.filter(pl.col("TAXITIME_SEC_mvt") > t)
        mu_n = fit_regressor(cfg, normal, cats, nums) if normal.height else None
        mu_l = fit_regressor(cfg, long, cats, nums) if long.height > 50 else None
        p_tune = predict_classifier(clf, tune, cats, nums)
        p_fit = predict_classifier(clf, fit, cats, nums)
        n_tune = predict_regressor(mu_n, tune, cats, nums) if mu_n else np.full(tune.height, float(fit["TAXITIME_SEC_mvt"].median()))
        l_tune = predict_regressor(mu_l, tune, cats, nums) if mu_l else n_tune
        mix_tune = (1 - p_tune) * n_tune + p_tune * l_tune
        p = predict_classifier(clf, val, cats, nums)
        n_hat = predict_regressor(mu_n, val, cats, nums) if mu_n else np.full(val.height, float(fit["TAXITIME_SEC_mvt"].median()))
        l_hat = predict_regressor(mu_l, val, cats, nums) if mu_l else n_hat
        pred = (1 - p) * n_hat + p * l_hat
        hard = np.where(p >= 0.5, l_hat, n_hat)
        pl.concat(
            [
                fit.select(["MVT_ID_mvt"]).with_columns(pl.Series("p_long", p_fit)),
                tune.select(["MVT_ID_mvt"]).with_columns(pl.Series("p_long", p_tune)),
            ]
        ).write_parquet(dest / f"p_long_{t}.parquet")
        val.select(["MVT_ID_mvt"]).with_columns(pl.Series("p_long", p)).write_parquet(dest / f"p_long_val_{t}.parquet")
        results[key] = {
            "mixture": _compact_score(_score_composed(cfg, view, val, pred, dest, tag=f"mix{t}")),
            "hard_gate": _compact_score(_score_composed(cfg, view, val, hard, dest, tag=f"hard{t}")),
            "p_long_mean": float(np.mean(p)),
            "tune_rmse": float(rmse(tune["TAXITIME_SEC_mvt"].to_numpy(), mix_tune)),
        }
        models[key] = (pred, mix_tune)
        _write_pred_table(dest / f"tune_predictions_mix{t}.parquet", tune, mix_tune)
        progress["thresholds"] = results
        _checkpoint_job(cfg, dest, "mixture_progress.json", progress)
    global_prior = progress.get("global") if isinstance(progress.get("global"), dict) else None
    global_tagged = dest / f"predictions_global_{view}.parquet"
    if global_prior and global_tagged.exists() and not cfg.force_rerun:
        print("CPU-06 resume skip global regressor", flush=True)
        global_score = global_prior
        g_tune_rmse = float(global_prior.get("tune_rmse") or 0.0)
        g_tune_path = dest / "tune_predictions_global.parquet"
        g_tune = pl.read_parquet(g_tune_path)["prediction"].to_numpy() if g_tune_path.exists() else None
    else:
        global_m = fit_regressor(cfg, fit, cats, nums)
        global_pred = predict_regressor(global_m, val, cats, nums)
        global_score = _score_composed(cfg, view, val, global_pred, dest, tag="global")
        g_tune = predict_regressor(global_m, tune, cats, nums)
        g_tune_rmse = float(rmse(tune["TAXITIME_SEC_mvt"].to_numpy(), g_tune))
        _write_pred_table(dest / "tune_predictions_global.parquet", tune, g_tune)
        progress["global"] = {**_compact_score(global_score), "tune_rmse": g_tune_rmse}
        _checkpoint_job(cfg, dest, "mixture_progress.json", progress)
    best_t = min(results, key=lambda k: results[k]["tune_rmse"])
    best_tag = f"mix{best_t}"
    if g_tune_rmse < results[best_t]["tune_rmse"]:
        best_tag = "global"
    promote_predictions(dest, view, best_tag)
    if best_tag.startswith("mix"):
        tpath = dest / f"tune_predictions_mix{best_t}.parquet"
        if best_t in models:
            _write_pred_table(dest / "tune_predictions.parquet", tune, models[best_t][1])
        elif tpath.exists():
            pl.read_parquet(tpath).write_parquet(dest / "tune_predictions.parquet")
        p_long_src = dest / f"p_long_{best_t}.parquet"
        p_long_val_src = dest / f"p_long_val_{best_t}.parquet"
        if p_long_src.exists():
            pl.read_parquet(p_long_src).write_parquet(dest / "p_long_features.parquet")
            persist_repo_file(cfg, dest / "p_long_features.parquet")
        if p_long_val_src.exists():
            pl.read_parquet(p_long_val_src).write_parquet(dest / "p_long_val.parquet")
            persist_repo_file(cfg, dest / "p_long_val.parquet")
    elif g_tune is not None:
        _write_pred_table(dest / "tune_predictions.parquet", tune, g_tune)
    elif (dest / "tune_predictions_global.parquet").exists():
        pl.read_parquet(dest / "tune_predictions_global.parquet").write_parquet(dest / "tune_predictions.parquet")

    def _predict_mix(tr: pl.DataFrame, va: pl.DataFrame, _view: str) -> np.ndarray:
        fit_v = tr.filter(pl.col("month").is_in(list(E37_FIT_MONTHS)))
        if fit_v.height == 0:
            fit_v = tr
        if best_tag == "global":
            gm = fit_regressor(cfg, fit_v, cats, nums)
            return predict_regressor(gm, va, cats, nums)
        t = int(best_t)
        lab = (fit_v["TAXITIME_SEC_mvt"] > t).cast(pl.Int32).alias("is_long")
        fit_l = fit_v.with_columns(lab)
        clf = fit_classifier(cfg, fit_l, cats, nums, "is_long")
        normal = fit_l.filter(pl.col("TAXITIME_SEC_mvt") <= t)
        long = fit_l.filter(pl.col("TAXITIME_SEC_mvt") > t)
        mu_n = fit_regressor(cfg, normal, cats, nums) if normal.height else None
        mu_l = fit_regressor(cfg, long, cats, nums) if long.height > 50 else None
        p = predict_classifier(clf, va, cats, nums)
        n_hat = predict_regressor(mu_n, va, cats, nums) if mu_n else np.full(va.height, float(fit_v["TAXITIME_SEC_mvt"].median()))
        l_hat = predict_regressor(mu_l, va, cats, nums) if mu_l else n_hat
        return (1 - p) * n_hat + p * l_hat

    extra_views: dict[str, Any] = {}
    if _missing_other_view_preds(dest, cfg, view) or cfg.force_rerun:
        extra_views = _score_predict_other_views(cfg, dest, df, primary=view, tag=best_tag, predict_fn=_predict_mix)
    else:
        print("CPU-06 resume skip extra-view refits", flush=True)
        extra_views = _resumed_extra_views(cfg, dest, df, view, tag=best_tag)
    return {
        "status": "completed",
        "rmse_seconds": results[best_t]["mixture"]["rmse_seconds"],
        "best_threshold_sec": int(best_t),
        "thresholds": {
            k: {
                "tune_rmse": v["tune_rmse"],
                "mixture_rmse": v["mixture"]["rmse_seconds"],
                "mixture_sse": v["mixture"]["sse"],
                "hard_gate_rmse": v["hard_gate"]["rmse_seconds"],
                "p_long_mean": v["p_long_mean"],
            }
            for k, v in results.items()
        },
        "global_regressor": {
            "rmse_seconds": global_score["rmse_seconds"],
            "sse": global_score["sse"],
            "tune_rmse": g_tune_rmse,
        },
        "delta_sse_vs_base": (results[best_t]["mixture"].get("vs_base") or {}).get("delta_sse_vs_base"),
        "views": {view: {"rmse_seconds": results[best_t]["mixture"]["rmse_seconds"]}, **extra_views},
        "conclusion": "judge composed SSE not tail RMSE; mixture vs hard-gate vs global",
        "hypothesis": "soft mixture of normal/long regimes beats a single mean",
    }


def _catalog_time_expr(df: pl.DataFrame) -> pl.Expr:
    if "EOBT_1_flt" in df.columns:
        return pl.coalesce([pl.col("EOBT_1_flt"), pl.col("MVT_TIME_UTC_mvt")])
    return pl.col("MVT_TIME_UTC_mvt")


def hide_match_reconstruction(
    matched: pl.DataFrame,
    *,
    seed: int = 42,
    n_sim: int = 8_000,
    k: int = 3,
    window_s: int = 7_200,
) -> dict[str, Any]:
    """Hide a ranking-safe identity and recover it from the NM catalog using MVT-side keys.

    Prefers `FLIGHT_ID_mvt` when the cache has it. The e08b campaign cache does not,
    so the identity is `AIRCRAFT_OPERATOR_flt` (airline/carrier from the handoff list).
    """
    if "MVT_TIME_UTC_mvt" not in matched.columns:
        return {"status": "skipped", "reason": "need timestamps"}
    if "FLIGHT_ID_mvt" in matched.columns:
        id_col = "FLIGHT_ID_mvt"
    elif "AIRCRAFT_OPERATOR_flt" in matched.columns:
        id_col = "AIRCRAFT_OPERATOR_flt"
    else:
        return {"status": "skipped", "reason": "need FLIGHT_ID_mvt or AIRCRAFT_OPERATOR_flt"}
    catalog = matched.filter(pl.col(id_col).is_not_null()).with_columns(_catalog_time_expr(matched).alias("c_time"))
    if id_col == "FLIGHT_ID_mvt":
        catalog = catalog.sort(["AIRPORT", id_col, "c_time"]).unique(["AIRPORT", id_col], keep="first")
    dest_col = "ADES_FILED_flt" if "ADES_FILED_flt" in catalog.columns else ("ADES_mvt" if "ADES_mvt" in catalog.columns else None)
    type_col = "AIRCRAFT_TYPE_mvt"
    keep = ["AIRPORT", "MVT_ID_mvt", id_col, "c_time", type_col]
    if dest_col:
        keep.append(dest_col)
    if "RUNWAY_mvt" in catalog.columns:
        keep.append("RUNWAY_mvt")
    catalog = catalog.select([c for c in keep if c in catalog.columns]).drop_nulls(["c_time"])
    queries = matched.filter(pl.col(id_col).is_not_null() & pl.col("MVT_TIME_UTC_mvt").is_not_null())
    n = min(queries.height, n_sim)
    if n < 20 or catalog.height < 20:
        return {"status": "skipped", "reason": "too few matched rows for hide-match"}
    queries = queries.sample(n=n, seed=seed, shuffle=True)
    hits_top1 = 0
    hits_topk = 0
    type_hits = 0
    dest_hits = 0
    n_scored = 0
    scores: list[float] = []
    correct: list[int] = []
    for ap, q_ap in queries.group_by("AIRPORT"):
        c_ap = catalog.filter(pl.col("AIRPORT") == (ap[0] if isinstance(ap, tuple) else ap))
        if c_ap.height < 2 or q_ap.height == 0:
            continue
        c_time = c_ap["c_time"].dt.epoch("s").to_numpy()
        order = np.argsort(c_time)
        c_time = c_time[order]
        c_ids = c_ap[id_col].to_numpy()[order]
        c_mvt = c_ap["MVT_ID_mvt"].to_numpy()[order] if "MVT_ID_mvt" in c_ap.columns else None
        c_type = c_ap[type_col].cast(pl.String).fill_null("").to_numpy()[order] if type_col in c_ap.columns else None
        c_dest = c_ap[dest_col].cast(pl.String).fill_null("").to_numpy()[order] if dest_col and dest_col in c_ap.columns else None
        q_time = q_ap["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
        q_ids = q_ap[id_col].to_numpy()
        q_mvt = q_ap["MVT_ID_mvt"].to_numpy() if "MVT_ID_mvt" in q_ap.columns else None
        q_type = q_ap[type_col].cast(pl.String).fill_null("").to_numpy() if type_col in q_ap.columns else None
        q_dest = q_ap[dest_col].cast(pl.String).fill_null("").to_numpy() if dest_col and dest_col in q_ap.columns else None
        c_rwy = c_ap["RUNWAY_mvt"].cast(pl.String).fill_null("").to_numpy()[order] if "RUNWAY_mvt" in c_ap.columns else None
        q_rwy = q_ap["RUNWAY_mvt"].cast(pl.String).fill_null("").to_numpy() if "RUNWAY_mvt" in q_ap.columns else None
        idx = np.clip(np.searchsorted(c_time, q_time), 0, len(c_time) - 1)
        offsets = np.arange(-k - 2, k + 3)
        exclude_self = id_col != "FLIGHT_ID_mvt"
        for i in range(len(q_time)):
            cand_i = np.clip(idx[i] + offsets, 0, len(c_time) - 1)
            cand_i = np.unique(cand_i)
            if exclude_self and c_mvt is not None and q_mvt is not None:
                cand_i = cand_i[c_mvt[cand_i] != q_mvt[i]]
            if cand_i.size == 0:
                continue
            dt = np.abs(c_time[cand_i] - q_time[i])
            keep_m = dt <= window_s
            if not keep_m.any():
                continue
            cand_i = cand_i[keep_m]
            dt = dt[keep_m]
            bonus = np.zeros(len(cand_i), dtype=np.float64)
            if c_type is not None and q_type is not None:
                th = (c_type[cand_i] == q_type[i]) & (q_type[i] != "")
                bonus += 3_600.0 * th
            if c_dest is not None and q_dest is not None:
                dh = (c_dest[cand_i] == q_dest[i]) & (q_dest[i] != "")
                bonus += 3_600.0 * dh
            if c_rwy is not None and q_rwy is not None:
                rh = (c_rwy[cand_i] == q_rwy[i]) & (q_rwy[i] != "")
                bonus += 1_800.0 * rh
            rank = np.argsort(dt - bonus)
            ranked_ids = c_ids[cand_i][rank]
            truth = q_ids[i]
            n_scored += 1
            top1 = ranked_ids[0] == truth
            hits_top1 += int(top1)
            hits_topk += int(truth in set(ranked_ids[:k].tolist()))
            best = rank[0]
            if c_type is not None and q_type is not None:
                type_hits += int(c_type[cand_i][best] == q_type[i] and q_type[i] != "")
            if c_dest is not None and q_dest is not None:
                dest_hits += int(c_dest[cand_i][best] == q_dest[i] and q_dest[i] != "")
            score = float(-dt[best] + bonus[best])
            scores.append(score)
            correct.append(int(top1))
    if n_scored == 0:
        return {"status": "skipped", "reason": "no scored hide-match queries"}
    scores_a = np.asarray(scores)
    correct_a = np.asarray(correct)
    threshold = None
    precision_at_t = None
    coverage_at_t = None
    if correct_a.size:
        for t in np.quantile(scores_a, [0.2, 0.4, 0.6, 0.8]):
            sel = scores_a >= t
            if sel.sum() < 20:
                continue
            prec = float(correct_a[sel].mean())
            if precision_at_t is None or prec >= 0.8:
                threshold, precision_at_t, coverage_at_t = float(t), prec, float(sel.mean())
            if prec >= 0.8:
                break
    return {
        "status": "completed",
        "identity": id_col,
        "n_sim": int(n_scored),
        "catalog_n": catalog.height,
        "top1_flight_id": hits_top1 / n_scored,
        "topk_flight_id": hits_topk / n_scored,
        "k": k,
        "window_s": window_s,
        "top1_type_hit_rate": type_hits / n_scored,
        "top1_dest_hit_rate": dest_hits / n_scored if dest_col else None,
        "threshold": threshold,
        "precision_at_threshold": precision_at_t,
        "coverage_at_threshold": coverage_at_t,
        "cache_note": None
        if id_col == "FLIGHT_ID_mvt"
        else "campaign cache has no FLIGHT_ID_mvt; hide-match uses AIRCRAFT_OPERATOR_flt",
    }


LINK_ID_COLS = ("FLIGHT_ID_mvt", "ADES_mvt", "ADES_FILED_flt", "ADEP_mvt")
LINK_SIDECAR = "link_ids_months_1_10.parquet"


def _attach_link_ids(cfg: CampaignConfig, df: pl.DataFrame) -> pl.DataFrame:
    """Join ranking-safe identity columns the e08b cache dropped. Does not rebuild the cache."""
    missing = [c for c in LINK_ID_COLS if c not in df.columns]
    if not missing:
        return df
    sidecar = Path(cfg.cache_root) / LINK_SIDECAR
    path = sidecar if sidecar.exists() else locate(Path("artifacts/cache") / LINK_SIDECAR, cfg)
    extra = None
    if path is not None:
        extra = pl.read_parquet(path)
    else:
        import os

        if os.environ.get("PRC_BUILD_LINK_IDS") != "1":
            print(
                "CPU-07 skip raw scan; hide-match uses cache columns. "
                "Set PRC_BUILD_LINK_IDS=1 on the SSH host to build artifacts/cache/link_ids_months_1_10.parquet",
                flush=True,
            )
            return df
        raw = Path(cfg.repo_root) / "data" / "raw"
        try:
            from prc_challenge.bundles import CACHE_MONTHS
            from prc_challenge.io import scan_training

            lf = scan_training(raw)
            names = set(lf.collect_schema().names())
            cols = [c for c in ("MVT_ID_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt", *missing) if c in names]
            if "MVT_ID_mvt" not in cols:
                return df
            extra_lf = lf.select(cols)
            if "PHASE_mvt" in cols:
                extra_lf = extra_lf.filter(pl.col("PHASE_mvt") == "DEP")
            if "MVT_TIME_UTC_mvt" in cols:
                extra_lf = extra_lf.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month")).filter(
                    pl.col("month").is_in(list(CACHE_MONTHS))
                )
            keep_out = ["MVT_ID_mvt", *[c for c in missing if c in cols]]
            extra = extra_lf.select(keep_out).unique("MVT_ID_mvt").collect()
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            extra.write_parquet(sidecar)
            from prc_challenge.campaign.runtime import persist_repo_file

            persist_repo_file(cfg, sidecar)
            print(f"CPU-07 wrote {sidecar} n={extra.height}", flush=True)
        except Exception as exc:
            print(f"CPU-07 link ids unavailable ({exc!r}); hide-match uses cache columns only", flush=True)
            return df
    keep = ["MVT_ID_mvt", *[c for c in missing if c in extra.columns]]
    if len(keep) <= 1:
        return df
    return df.join(extra.select(keep).unique("MVT_ID_mvt", keep="first"), on="MVT_ID_mvt", how="left")


def job_cpu07_unmatched(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    """Hide known NM matches, recover via ranking-safe catalog; do not splice the prize path."""
    df = load_cache_columns(
        cfg,
        extra=["AIRCRAFT_OPERATOR_flt", "FLIGHT_ID_mvt", "ADES_FILED_flt", "ADES_mvt"],
    )
    df = _attach_link_ids(cfg, df)
    view = _primary_view(cfg)
    if "MVT_TIME_UTC_mvt" not in df.columns:
        return {"status": "skipped", "reason": "no timestamps", "conclusion": "PARK"}
    matched = df.filter(pl.col("nm_unmatched") == 0)
    unmatched = df.filter(pl.col("nm_unmatched") == 1)
    if matched.height < 50:
        return {"status": "skipped", "reason": "too few matched rows", "conclusion": "PARK"}
    sim = hide_match_reconstruction(
        matched,
        seed=cfg.seed,
        n_sim=2_000 if cfg.smoke else 8_000,
    )
    if sim.get("status") != "completed":
        return {**sim, "conclusion": "PARK", "hypothesis": "unmatched rows can be relinked with ranking-safe identifiers"}
    _, val = matched_train_val(df, view)
    parent = load_parent(cfg, view)
    # Diagnostic splice: nearest previous matched taxi at the same airport (labeled hypothesis only).
    right = matched.select(
        ["AIRPORT", pl.col("MVT_TIME_UTC_mvt").alias("nn_time"), pl.col("TAXITIME_SEC_mvt").alias("nn_taxi")]
    ).sort(["AIRPORT", "nn_time"])
    left = val.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    linked = left.join_asof(right, left_on="MVT_TIME_UTC_mvt", right_on="nn_time", by="AIRPORT", strategy="backward")
    linked = linked.with_columns(
        (pl.col("MVT_TIME_UTC_mvt").dt.epoch("s") - pl.col("nn_time").dt.epoch("s")).abs().alias("nn_dt")
    )
    val_u = val.join(linked.select(["MVT_ID_mvt", "nn_taxi", "nn_dt"]), on="MVT_ID_mvt", how="left")
    val_u = val_u.join(parent.select(["MVT_ID_mvt", pl.col("prediction").alias("parent_prediction")]), on="MVT_ID_mvt")
    thresh_dt = 120
    splice = val_u.with_columns(
        pl.when((pl.col("nm_unmatched") == 1) & (pl.col("nn_dt") <= thresh_dt) & pl.col("nn_taxi").is_not_null())
        .then(pl.col("nn_taxi"))
        .otherwise(pl.col("parent_prediction"))
        .alias("prediction")
    )
    unmatched_val = val_u.filter(pl.col("nm_unmatched") == 1)
    recovered = unmatched_val.filter(pl.col("nn_taxi").is_not_null() & (pl.col("nn_dt") <= thresh_dt))
    coverage = float(recovered.height / unmatched_val.height) if unmatched_val.height else 0.0
    subset_rmse = None
    if recovered.height:
        subset_rmse = rmse(recovered["TAXITIME_SEC_mvt"].to_numpy(), recovered["nn_taxi"].to_numpy())
    splice.select(["MVT_ID_mvt", "prediction"]).write_parquet(dest / "predictions_splice_hypothesis.parquet")
    (dest / "hide_match_simulation.json").write_text(json.dumps(sim, indent=2) + "\n")
    frozen = _score_composed(cfg, view, val, val_u["parent_prediction"].to_numpy(), dest, tag="frozen")

    def _predict_parent(_tr: pl.DataFrame, va: pl.DataFrame, other: str) -> np.ndarray:
        parent_o = load_parent(cfg, other)
        joined = va.join(parent_o.select(["MVT_ID_mvt", "prediction"]), on="MVT_ID_mvt", how="left")
        if joined["prediction"].null_count():
            raise RuntimeError(f"{other} parent preds do not cover val")
        return joined["prediction"].to_numpy()

    extra_views = _score_predict_other_views(
        cfg, dest, df, primary=view, tag="frozen", predict_fn=_predict_parent
    )
    conclusion = "do not splice into prize pipeline"
    if (sim.get("top1_flight_id") or 0) >= 0.6 and (sim.get("precision_at_threshold") or 0) >= 0.8:
        conclusion = "high hide-match precision on train; still do not auto-splice unmatched"
    return {
        "status": "completed",
        "rmse_seconds": frozen["rmse_seconds"],
        "unmatched_n": unmatched.height,
        "hide_match": sim,
        "recovery_coverage": coverage,
        "unmatched_subset_rmse": subset_rmse,
        "protected_composed": frozen,
        "views": {view: _compact_score(frozen), **extra_views},
        "identity": sim.get("identity"),
        "conclusion": conclusion,
        "hypothesis": "unmatched/leftover rows can be relinked with ranking-safe identifiers",
    }


def job_cpu08_delta(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    df = load_cache(cfg)
    cats, nums = feature_lists("e08b")
    view = _primary_view(cfg)
    train, val = matched_train_val(df, view)
    anchors = [
        ("eobt", "takeoff_minus_eobt_sec", "eobt_to_block_sec"),
        ("iobt", "takeoff_minus_iobt_sec", "iobt_to_block_sec"),
        ("sched", "takeoff_minus_sched_sec", "sched_to_block_sec"),
    ]
    available = [(name, col, label) for name, col, label in anchors if col in train.columns]
    if not available:
        return {"status": "skipped", "reason": "no EOBT/IOBT/SCHED delta", "conclusion": "PARK"}
    progress_path = dest / "delta_progress.json"
    table: dict[str, Any] = {} if cfg.force_rerun else _load_json_dict(progress_path)
    best_tag = None
    best_rmse = None
    winner_anchor = available[0]
    for name, col, label in available:
        tag = f"{name}_recon"
        prior = table.get(tag) if isinstance(table.get(tag), dict) else None
        if prior and prior.get("rmse_seconds") is not None and not cfg.force_rerun:
            print(f"CPU-08 resume skip {tag} rmse={prior['rmse_seconds']}", flush=True)
            if best_rmse is None or prior["rmse_seconds"] < best_rmse:
                best_tag, best_rmse, winner_anchor = tag, prior["rmse_seconds"], (name, col, label)
            continue
        tr = train.filter(pl.col(col).is_not_null() & pl.col("TAXITIME_SEC_mvt").is_not_null()).with_columns(
            (pl.col(col) - pl.col("TAXITIME_SEC_mvt")).alias(label)
        )
        delta_m = fit_regressor(cfg, tr, cats, nums, label=label)
        delta_hat = predict_unclipped(delta_m, val, cats, nums)
        reconstructed = np.clip(val[col].to_numpy() - delta_hat, 0, None)
        scored = _score_composed(cfg, view, val, reconstructed, dest, tag=tag)
        table[tag] = _compact_score(scored)
        _checkpoint_job(cfg, dest, "delta_progress.json", table)
        if best_rmse is None or table[tag]["rmse_seconds"] < best_rmse:
            best_tag, best_rmse, winner_anchor = tag, table[tag]["rmse_seconds"], (name, col, label)
    prior_direct = table.get("direct") if isinstance(table.get("direct"), dict) else None
    if prior_direct and prior_direct.get("rmse_seconds") is not None and not cfg.force_rerun:
        print(f"CPU-08 resume skip direct rmse={prior_direct['rmse_seconds']}", flush=True)
        direct_m = prior_direct
    else:
        direct = fit_regressor(cfg, train, cats, nums)
        dpred = predict_regressor(direct, val, cats, nums)
        direct_m = _score_composed(cfg, view, val, dpred, dest, tag="direct")
        table["direct"] = _compact_score(direct_m)
        _checkpoint_job(cfg, dest, "delta_progress.json", table)
    if best_rmse is None or (direct_m.get("rmse_seconds") is not None and direct_m["rmse_seconds"] < best_rmse):
        best_tag = "direct"
        best_rmse = direct_m.get("rmse_seconds")
        winner_anchor = None
    promote_predictions(dest, view, best_tag)

    def _predict_delta(tr: pl.DataFrame, va: pl.DataFrame, _other: str) -> np.ndarray:
        if best_tag == "direct" or winner_anchor is None:
            m = fit_regressor(cfg, tr, cats, nums)
            return predict_regressor(m, va, cats, nums)
        _name, col, label = winner_anchor
        tr = tr.with_columns((pl.col(col) - pl.col("TAXITIME_SEC_mvt")).alias(label))
        dm = fit_regressor(cfg, tr, cats, nums, label=label)
        dhat = predict_unclipped(dm, va, cats, nums)
        return np.clip(va[col].to_numpy() - dhat, 0, None)

    extra_views = _score_predict_other_views(cfg, dest, df, primary=view, tag=best_tag, predict_fn=_predict_delta)
    winner = table.get(best_tag) or {}
    return {
        "status": "completed",
        "rmse_seconds": winner.get("rmse_seconds") if winner else best_rmse,
        "best_tag": best_tag,
        "anchors": table,
        "views": {view: winner, **extra_views},
        "conclusion": f"{best_tag} vs direct taxi; AOBT not used",
        "hypothesis": "an underlying planning-delta is easier than raw taxi seconds",
    }


def job_cpu08b_local_aobt(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    """LOCAL_ONLY: AOBT offset + learned correction. Never a prize-path feature."""
    if not cfg.allow_local_aobt:
        return {
            "status": "skipped",
            "reason": "AOBT remains quarantined from the prize path; set allow_local_aobt=True only for a local-only check",
            "conclusion": "LOCAL_ONLY — not in overnight prize campaign",
            "hypothesis": "AOBT-derived baseline + learned correction (organizer ruling pending)",
        }
    df = load_cache(cfg)
    if "AOBT_3_flt" not in df.columns or "MVT_TIME_UTC_mvt" not in df.columns:
        return {"status": "skipped", "reason": "AOBT_3_flt not in cache", "conclusion": "PARK"}
    view = _primary_view(cfg)
    cats, nums = feature_lists("e08b")
    train, val = matched_train_val(df, view)
    def _offset(frame: pl.DataFrame) -> np.ndarray:
        sec = (frame["MVT_TIME_UTC_mvt"] - frame["AOBT_3_flt"]).dt.total_seconds().to_numpy()
        return np.clip(np.nan_to_num(sec.astype(np.float64), nan=0.0), 0, None)

    train_off = _offset(train)
    val_off = _offset(val)
    train = train.with_columns(pl.Series("_aobt_resid", train["TAXITIME_SEC_mvt"].to_numpy() - train_off))
    model = fit_regressor(cfg, train, cats, nums, label="_aobt_resid")
    resid_hat = predict_unclipped(model, val, cats, nums)
    pred = np.clip(val_off + resid_hat, 0, None)
    naive = _score_composed(cfg, view, val, val_off, dest, tag="naive_mvt_minus_aobt")
    composed = _score_composed(cfg, view, val, pred, dest, tag="aobt_plus_correction")
    promote_predictions(dest, view, "aobt_plus_correction")
    return {
        "status": "completed",
        "rmse_seconds": composed["rmse_seconds"],
        "naive_mvt_minus_aobt": {k: naive.get(k) for k in ("rmse_seconds", "sse", "vs_base")},
        "corrected": {k: composed.get(k) for k in ("rmse_seconds", "sse", "vs_base")},
        "conclusion": "LOCAL_ONLY — AOBT offset plus e08b residual; not prize-eligible",
        "hypothesis": "AOBT-derived baseline + learned correction (organizer ruling pending)",
    }


def _complementarity(val: pl.DataFrame, pred: np.ndarray, base_pred: np.ndarray | None) -> dict[str, Any]:
    y = val["TAXITIME_SEC_mvt"].to_numpy()
    out: dict[str, Any] = {}
    if base_pred is None or len(base_pred) != len(pred):
        return {"status": "base_missing"}
    out["pred_corr_with_base"] = float(np.corrcoef(pred, base_pred)[0, 1])
    out["residual_corr_with_base"] = float(np.corrcoef(y - pred, y - base_pred)[0, 1])
    return out


def _train_only_blend_weight(y_tune: np.ndarray, pred_tune: np.ndarray, base_tune: np.ndarray) -> float:
    best_w, best_r = 0.0, rmse(y_tune, base_tune)
    for w in np.linspace(0, 1, 11):
        mix = w * pred_tune + (1 - w) * base_tune
        r = rmse(y_tune, mix)
        if r < best_r:
            best_r, best_w = r, float(w)
    return best_w


def job_cpu09_diversity(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor

    df = load_or_build_f3(cfg, load_cache(cfg))
    cats, nums = feature_lists("e08b")
    f3_nums = [c for c in f3_numeric_columns() if c in df.columns]
    nums = list(dict.fromkeys(nums + f3_nums))
    view = _primary_view(cfg)
    train, val = matched_train_val(df, view)
    fit = train.filter(pl.col("month").is_in(list(E37_FIT_MONTHS)))
    tune = train.filter(pl.col("month").is_in(list(E37_TUNE_MONTHS)))
    if fit.height == 0:
        fit = train
    if tune.height == 0:
        tune = train
    Xfit = numeric_matrix(fit, nums)
    Xtune = numeric_matrix(tune, nums)
    Xva = numeric_matrix(val, nums)
    yfit = fit["TAXITIME_SEC_mvt"].to_numpy()
    ytune = tune["TAXITIME_SEC_mvt"].to_numpy()
    try:
        base_val = load_base_preds(cfg, view)
        val_b = val.join(base_val.select(["MVT_ID_mvt", pl.col("prediction").alias("base_pred")]), on="MVT_ID_mvt")
        bp_val = val_b["base_pred"].to_numpy()
        tune_b = tune.join(base_val.select(["MVT_ID_mvt", pl.col("prediction").alias("base_pred")]), on="MVT_ID_mvt", how="left")
        # tune months are not in official val; use in-fold CatBoost OOF cache if present
        from prc_challenge.campaign.models import load_or_build_base_residual

        oof_train, _, _, _ = load_or_build_base_residual(cfg)
        tune_oof = tune.join(oof_train.select(["MVT_ID_mvt", "oof_prediction"]), on="MVT_ID_mvt", how="left")
        bp_tune = tune_oof["oof_prediction"].to_numpy() if "oof_prediction" in tune_oof.columns else None
    except Exception:
        bp_val = None
        bp_tune = None
    modelspecs: list[tuple[str, Any]] = []
    try:
        import lightgbm as lgb

        modelspecs.append(
            (
                "lightgbm",
                lgb.LGBMRegressor(
                    n_estimators=80 if cfg.smoke else 400,
                    learning_rate=0.08,
                    subsample=0.8,
                    random_state=cfg.seed,
                    n_jobs=cfg.thread_count,
                    verbose=-1,
                ),
            )
        )
    except Exception as exc:
        modelspecs.append(("lightgbm", exc))
    try:
        from xgboost import XGBRegressor

        modelspecs.append(
            (
                "xgboost",
                XGBRegressor(
                    n_estimators=80 if cfg.smoke else 400,
                    learning_rate=0.08,
                    max_depth=6,
                    subsample=0.8,
                    tree_method="hist",
                    n_jobs=cfg.thread_count,
                    random_state=cfg.seed,
                    verbosity=0,
                ),
            )
        )
    except Exception as exc:
        modelspecs.append(("xgboost", exc))
    et_n = min(fit.height, 8_000 if cfg.smoke else 80_000)
    et_idx = np.random.default_rng(cfg.seed).choice(fit.height, size=et_n, replace=False) if fit.height > et_n else np.arange(fit.height)
    modelspecs.append(
        (
            "extratrees",
            ExtraTreesRegressor(
                n_estimators=40 if cfg.smoke else 120,
                max_depth=12,
                random_state=cfg.seed,
                n_jobs=cfg.thread_count,
            ),
        )
    )
    progress = {} if cfg.force_rerun else _load_json_dict(dest / "diversity_progress.json")
    out: dict[str, Any] = dict(progress.get("models") or {}) if isinstance(progress.get("models"), dict) else {}
    best_name, best_rmse = progress.get("best_name"), progress.get("best_rmse")
    tune_by_name: dict[str, np.ndarray] = {}
    for name, model in modelspecs:
        if isinstance(model, Exception):
            out[name] = {"status": "failed", "error": repr(model)}
            continue
        prior = out.get(name) if isinstance(out.get(name), dict) else None
        tagged = dest / f"predictions_{name}_{view}.parquet"
        if prior and prior.get("rmse_seconds") is not None and tagged.exists() and not cfg.force_rerun:
            print(f"CPU-09 resume skip {name} rmse={prior['rmse_seconds']}", flush=True)
            if prior.get("tune_rmse") is not None and (best_rmse is None or prior["tune_rmse"] < best_rmse):
                best_name, best_rmse = name, float(prior["tune_rmse"])
            if prior.get("tune_blend_rmse") is not None and (best_rmse is None or prior["tune_blend_rmse"] < best_rmse):
                best_name, best_rmse = f"{name}_blend", float(prior["tune_blend_rmse"])
            for tname in (name, f"{name}_blend"):
                tpath = dest / f"tune_predictions_{tname}.parquet"
                if tpath.exists():
                    tune_by_name[tname] = pl.read_parquet(tpath)["prediction"].to_numpy()
            continue
        if name == "extratrees":
            model.fit(Xfit[et_idx], yfit[et_idx])
        else:
            model.fit(Xfit, yfit)
        pred = np.clip(model.predict(Xva), 0, None)
        scored = _score_composed(cfg, view, val, pred, dest, tag=name)
        comp = _complementarity(val, pred, bp_val)
        blend_w = None
        blend_rmse = None
        pred_tune = np.clip(model.predict(Xtune), 0, None)
        tune_by_name[name] = pred_tune
        tune_rmse = float(rmse(ytune, pred_tune))
        tune_blend_rmse = None
        if bp_tune is not None and np.isfinite(bp_tune).all():
            blend_w = _train_only_blend_weight(ytune, pred_tune, bp_tune)
            if bp_val is not None:
                blend = blend_w * pred + (1 - blend_w) * bp_val
                blend_scored = _score_composed(cfg, view, val, blend, dest, tag=f"{name}_blend")
                blend_rmse = blend_scored.get("rmse_seconds")
            tune_blend = blend_w * pred_tune + (1 - blend_w) * bp_tune
            tune_by_name[f"{name}_blend"] = tune_blend
            tune_blend_rmse = float(rmse(ytune, tune_blend))
        out[name] = {
            "rmse_seconds": scored.get("rmse_seconds"),
            "sse": scored.get("sse"),
            "vs_base": scored.get("vs_base"),
            "complementarity": comp,
            "train_only_blend_weight": blend_w,
            "blend_composed_rmse": blend_rmse,
            "tune_rmse": tune_rmse,
            "tune_blend_rmse": tune_blend_rmse,
        }
        if best_rmse is None or tune_rmse < best_rmse:
            best_name, best_rmse = name, tune_rmse
        if tune_blend_rmse is not None and tune_blend_rmse < best_rmse:
            best_name, best_rmse = f"{name}_blend", tune_blend_rmse
        for tname, tpred in list(tune_by_name.items()):
            if tname == name or tname == f"{name}_blend":
                _write_pred_table(dest / f"tune_predictions_{tname}.parquet", tune, tpred)
        progress["models"] = out
        progress["best_name"] = best_name
        progress["best_rmse"] = best_rmse
        _checkpoint_job(cfg, dest, "diversity_progress.json", progress)
    official_rmse = best_rmse
    if best_name:
        promote_predictions(dest, view, best_name)
        if best_name in tune_by_name:
            _write_pred_table(dest / "tune_predictions.parquet", tune, tune_by_name[best_name])
        rec = out.get(best_name[: -len("_blend")] if best_name.endswith("_blend") else best_name) or {}
        official_rmse = rec.get("blend_composed_rmse") if best_name.endswith("_blend") else rec.get("rmse_seconds")
    extra_views: dict[str, Any] = {}
    if best_name:
        blend = best_name.endswith("_blend")
        base_name = best_name[: -len("_blend")] if blend else best_name
        blend_w = (out.get(base_name) or {}).get("train_only_blend_weight") or 0.0

        def _new_model():
            if base_name == "lightgbm":
                import lightgbm as lgb

                return lgb.LGBMRegressor(
                    n_estimators=80 if cfg.smoke else 400,
                    learning_rate=0.08,
                    subsample=0.8,
                    random_state=cfg.seed,
                    n_jobs=cfg.thread_count,
                    verbose=-1,
                )
            if base_name == "xgboost":
                from xgboost import XGBRegressor

                return XGBRegressor(
                    n_estimators=80 if cfg.smoke else 400,
                    learning_rate=0.08,
                    max_depth=6,
                    subsample=0.8,
                    tree_method="hist",
                    n_jobs=cfg.thread_count,
                    random_state=cfg.seed,
                    verbosity=0,
                )
            from sklearn.ensemble import ExtraTreesRegressor

            return ExtraTreesRegressor(
                n_estimators=40 if cfg.smoke else 120,
                max_depth=12,
                random_state=cfg.seed,
                n_jobs=cfg.thread_count,
            )

        def _predict_div(tr: pl.DataFrame, va: pl.DataFrame, other: str) -> np.ndarray:
            mdl = _new_model()
            Xtr = numeric_matrix(tr, nums)
            Xva = numeric_matrix(va, nums)
            ytr = tr["TAXITIME_SEC_mvt"].to_numpy()
            if base_name == "extratrees" and Xtr.shape[0] > (8_000 if cfg.smoke else 80_000):
                idx = np.random.default_rng(cfg.seed).choice(Xtr.shape[0], size=8_000 if cfg.smoke else 80_000, replace=False)
                mdl.fit(Xtr[idx], ytr[idx])
            else:
                mdl.fit(Xtr, ytr)
            pred = np.clip(mdl.predict(Xva), 0, None)
            if not blend:
                return pred
            bp = load_base_preds(cfg, other)
            joined = va.join(bp.select(["MVT_ID_mvt", pl.col("prediction").alias("bp")]), on="MVT_ID_mvt", how="left")
            base_p = joined["bp"].fill_null(float(np.median(pred))).to_numpy()
            return blend_w * pred + (1 - blend_w) * base_p

        if _missing_other_view_preds(dest, cfg, view) or cfg.force_rerun:
            extra_views = _score_predict_other_views(
                cfg, dest, df, primary=view, tag=best_name, predict_fn=_predict_div
            )
        else:
            extra_views = _resumed_extra_views(cfg, dest, df, view, tag=best_name)
            print("CPU-09 resume skip extra-view refits", flush=True)
    return {
        "status": "completed",
        "rmse_seconds": official_rmse,
        "best_model": best_name,
        "tune_rmse": best_rmse,
        "models": out,
        "views": {view: {"rmse_seconds": official_rmse}, **extra_views},
        "conclusion": "standalone + complementarity + train-only BASE blend",
        "hypothesis": "non-CatBoost learners on F3 have complementary errors",
    }


def _f3_feature_matrix(frame: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return np.nan_to_num(frame.select(cols).fill_null(0).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)


def _airport_scoped_knn(Xtr, ytr, airports_tr, Xte, airports_te, k: int) -> np.ndarray:
    """Per-airport distance-weighted kNN. Median fallback — never a global kNN on all val rows."""
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    gmed = float(np.median(ytr)) if len(ytr) else 0.0
    pred = np.full(len(Xte), gmed, dtype=np.float64)
    for ap in np.unique(airports_te):
        tr_m = airports_tr == ap
        te_m = airports_te == ap
        if te_m.sum() == 0:
            continue
        if tr_m.sum() < max(k + 1, 20):
            continue
        sc = StandardScaler()
        Xa = sc.fit_transform(Xtr[tr_m])
        Xq = sc.transform(Xte[te_m])
        kk = min(k, int(tr_m.sum()) - 1)
        knn = KNeighborsRegressor(n_neighbors=max(1, kk), weights="distance")
        knn.fit(Xa, ytr[tr_m])
        pred[te_m] = np.clip(knn.predict(Xq), 0, None)
    return pred


def job_cpu10_knn(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    df = load_or_build_f3(cfg, load_cache(cfg))
    analog = [c for c in ("local_hour", "local_dow", "sknt", "takeoff_minus_sched_sec") if c in df.columns]
    nums = list(dict.fromkeys([c for c in f3_numeric_columns() if c in df.columns] + analog))
    if not nums:
        nums = analog
    view = _primary_view(cfg)
    train, val = matched_train_val(df, view)
    fit = train.filter(pl.col("month").is_in(list(E37_FIT_MONTHS)))
    tune = train.filter(pl.col("month").is_in(list(E37_TUNE_MONTHS)))
    if fit.height == 0 or tune.height == 0:
        fit, tune = train, train
    cap = 3_000 if cfg.smoke else 40_000
    if fit.height > cap:
        fit = fit.sample(n=cap, seed=cfg.seed, shuffle=True)
    Xfit = _f3_feature_matrix(fit, nums)
    yfit = fit["TAXITIME_SEC_mvt"].to_numpy()
    afit = fit["AIRPORT"].to_numpy()
    Xtune = _f3_feature_matrix(tune, nums)
    ytune = tune["TAXITIME_SEC_mvt"].to_numpy()
    atune = tune["AIRPORT"].to_numpy()
    ks = (5, 15) if cfg.smoke else (5, 15, 31)
    progress = {} if cfg.force_rerun else _load_json_dict(dest / "knn_progress.json")
    table = progress.get("ks") if isinstance(progress.get("ks"), dict) else {}
    best_k, best_r = ks[0], 1e18
    if progress.get("best_k") is not None and not cfg.force_rerun:
        best_k = int(progress["best_k"])
        best_r = float(progress.get("best_r") or best_r)
    for k in ks:
        prior = table.get(str(k))
        if prior is not None and not cfg.force_rerun:
            print(f"CPU-10 resume skip k={k} tune_rmse={prior}", flush=True)
            table[str(k)] = prior
            if prior < best_r:
                best_k, best_r = k, float(prior)
            continue
        pred_tune = _airport_scoped_knn(Xfit, yfit, afit, Xtune, atune, k)
        table[str(k)] = float(rmse(ytune, pred_tune))
        if table[str(k)] < best_r:
            best_k, best_r = k, table[str(k)]
        progress["ks"] = table
        progress["best_k"] = best_k
        progress["best_r"] = best_r
        _checkpoint_job(cfg, dest, "knn_progress.json", progress)
    Xtr = _f3_feature_matrix(train, nums)
    if train.height > cap:
        sampled = train.sample(n=cap, seed=cfg.seed, shuffle=True)
        Xtr = _f3_feature_matrix(sampled, nums)
        ytr = sampled["TAXITIME_SEC_mvt"].to_numpy()
        atr = sampled["AIRPORT"].to_numpy()
    else:
        ytr = train["TAXITIME_SEC_mvt"].to_numpy()
        atr = train["AIRPORT"].to_numpy()
    Xva = _f3_feature_matrix(val, nums)
    pred_path = dest / "predictions.parquet"
    pred = _pred_array_for(val, pred_path) if not cfg.force_rerun else None
    if pred is None:
        pred = _airport_scoped_knn(Xtr, ytr, atr, Xva, val["AIRPORT"].to_numpy(), best_k)
        composed = _score_composed(cfg, view, val, pred, dest)
        persist_dir(cfg, dest)
    else:
        print("CPU-10 resume skip V1 kNN val", flush=True)
        composed = _score_composed(cfg, view, val, pred, dest)
    best_tune = _airport_scoped_knn(Xfit, yfit, afit, Xtune, atune, best_k)
    _write_pred_table(dest / "tune_predictions.parquet", tune, best_tune)

    def _predict_knn(tr: pl.DataFrame, va: pl.DataFrame, _other: str) -> np.ndarray:
        sampled = tr.sample(n=cap, seed=cfg.seed, shuffle=True) if tr.height > cap else tr
        return _airport_scoped_knn(
            _f3_feature_matrix(sampled, nums),
            sampled["TAXITIME_SEC_mvt"].to_numpy(),
            sampled["AIRPORT"].to_numpy(),
            _f3_feature_matrix(va, nums),
            va["AIRPORT"].to_numpy(),
            best_k,
        )

    extra_views = _score_predict_other_views(cfg, dest, df, primary=view, predict_fn=_predict_knn)
    return {
        "status": "completed",
        "rmse_seconds": composed["rmse_seconds"],
        "row_count": composed.get("row_count"),
        "best_k": best_k,
        "internal_tune_rmse": {int(k): v for k, v in table.items()},
        "delta_sse_vs_base": (composed.get("vs_base") or {}).get("delta_sse_vs_base"),
        "delta_rmse_vs_e33b": composed.get("delta_rmse_vs_e33b"),
        "views": {view: _compact_score(composed), **extra_views},
        "conclusion": "k and distance-weighted kNN selected on E37 tune month; airport-scoped",
        "hypothesis": "similar historical airport states predict taxi-out",
    }


def job_cpu11_residual(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    from prc_challenge.campaign.models import load_or_build_base_residual

    df = load_or_build_f3(cfg, load_cache(cfg))
    cats, nums = feature_lists("e08b")
    f3_nums = [c for c in f3_numeric_columns() if c in df.columns]
    train, val, val_base, view = load_or_build_base_residual(cfg, extra_cols=RANKING_IDENTITY_EXTRAS)
    train = train.join(df.select(["MVT_ID_mvt", *f3_nums]), on="MVT_ID_mvt", how="left")
    val = val.join(df.select(["MVT_ID_mvt", *f3_nums]), on="MVT_ID_mvt", how="left")
    risk_train = cfg.artifact_root / "CPU-03-error-predictability" / "risk_train.parquet"
    risk_val_path = cfg.artifact_root / "CPU-03-error-predictability" / "risk_predictions.parquet"
    if risk_train.exists():
        train = train.join(pl.read_parquet(risk_train), on="MVT_ID_mvt", how="left")
    if risk_val_path.exists():
        val = val.join(pl.read_parquet(risk_val_path), on="MVT_ID_mvt", how="left")
    p_long_train = cfg.artifact_root / "CPU-06-mixture" / "p_long_features.parquet"
    p_long_val = cfg.artifact_root / "CPU-06-mixture" / "p_long_val.parquet"
    if p_long_train.exists():
        train = train.join(pl.read_parquet(p_long_train), on="MVT_ID_mvt", how="left")
    if p_long_val.exists():
        val = val.join(pl.read_parquet(p_long_val), on="MVT_ID_mvt", how="left")
    keys = [c for c in ("AIRPORT", "STAND_mvt", "RUNWAY_mvt") if c in train.columns]
    if keys:
        oof_parts = []
        for month in train["month"].unique().sort().to_list():
            fit = train.filter(pl.col("month") != month)
            hold = train.filter(pl.col("month") == month)
            if fit.height == 0 or hold.height == 0:
                continue
            cell = fit.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_base"))
            oof_parts.append(hold.join(cell, on=keys, how="left"))
        train = pl.concat(oof_parts, how="diagonal") if oof_parts else train
        cell = train.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_base"))
        val = val.join(cell, on=keys, how="left")
        med = float(train["TAXITIME_SEC_mvt"].median())
        train = train.with_columns(pl.col("route_base").fill_null(med))
        val = val.with_columns(pl.col("route_base").fill_null(med))
    extra = [c for c in ("oof_prediction", "route_base", "pred_risk", "p_long") if c in train.columns]
    res_nums = list(dict.fromkeys(nums + f3_nums + extra))
    cats = [c for c in cats if c in train.columns]
    res_nums = [c for c in res_nums if c in train.columns]
    fit = train.filter(pl.col("month").is_in(list(E37_FIT_MONTHS)))
    tune = train.filter(pl.col("month").is_in(list(E37_TUNE_MONTHS)))
    if fit.height < 50 or tune.height < 20:
        fit, tune = train, train
    model_sel = fit_regressor(cfg, fit, cats, res_nums, label="residual")
    r_tune = predict_unclipped(model_sel, tune, cats, res_nums)
    pred_tune_cb = np.clip(tune["oof_prediction"].to_numpy() + r_tune, 0, None)
    tune_y = tune["TAXITIME_SEC_mvt"].to_numpy()
    tune_rmse_cb = float(rmse(tune_y, pred_tune_cb))
    booster_sel = None
    pred_tune_lgb = None
    tune_rmse_lgb = None
    try:
        import lightgbm as lgb

        booster_sel = lgb.LGBMRegressor(
            n_estimators=80 if cfg.smoke else 300,
            random_state=cfg.seed,
            n_jobs=cfg.thread_count,
            verbose=-1,
        )
        booster_sel.fit(numeric_matrix(fit, res_nums), fit["residual"].to_numpy())
        pred_tune_lgb = np.clip(
            tune["oof_prediction"].to_numpy() + booster_sel.predict(numeric_matrix(tune, res_nums)),
            0,
            None,
        )
        tune_rmse_lgb = float(rmse(tune_y, pred_tune_lgb))
    except Exception as exc:
        print(f"CPU-11 LightGBM tune failed: {exc!r}", flush=True)
    best_tag = "lgb" if tune_rmse_lgb is not None and tune_rmse_lgb < tune_rmse_cb else "catboost"
    tune_pred = pred_tune_lgb if best_tag == "lgb" and pred_tune_lgb is not None else pred_tune_cb
    _write_pred_table(dest / "tune_predictions.parquet", tune, tune_pred)
    persist_repo_file(cfg, dest / "tune_predictions.parquet")
    # Held-out E37 tune month only — do not write in-sample residual preds for blending.
    _write_pred_table(dest / "oof_predictions.parquet", tune, tune_pred)
    persist_repo_file(cfg, dest / "oof_predictions.parquet")

    model = fit_regressor(cfg, train, cats, res_nums, label="residual")
    val = val.with_columns(pl.Series("oof_prediction", val_base))
    rhat = predict_unclipped(model, val, cats, res_nums)
    pred = np.clip(val_base + rhat, 0, None)
    composed = _score_composed(cfg, view, val, pred, dest, tag="catboost")
    lgb_m = None
    booster = None
    try:
        import lightgbm as lgb

        booster = lgb.LGBMRegressor(
            n_estimators=80 if cfg.smoke else 300,
            random_state=cfg.seed,
            n_jobs=cfg.thread_count,
            verbose=-1,
        )
        booster.fit(numeric_matrix(train, res_nums), train["residual"].to_numpy())
        lhat = booster.predict(numeric_matrix(val, res_nums))
        lpred = np.clip(val_base + lhat, 0, None)
        lgb_m = _score_composed(cfg, view, val, lpred, dest, tag="lgb")
    except Exception as exc:
        lgb_m = {"status": "failed", "error": repr(exc)}
    best = lgb_m if best_tag == "lgb" and isinstance(lgb_m, dict) and lgb_m.get("rmse_seconds") is not None else composed
    if best_tag == "lgb" and not (isinstance(best, dict) and best.get("rmse_seconds") is not None):
        best_tag = "catboost"
        best = composed
    promote_predictions(dest, view, best_tag)
    extra_views: dict[str, Any] = {}
    persist_dir(cfg, dest)
    for other in cfg.views:
        if other == view:
            continue
        if (dest / f"predictions_{other}.parquet").exists() and not cfg.force_rerun:
            extra_views[other] = _hydrate_extra_view(cfg, dest, df, other)
            print(f"CPU-11 resume extra view {other} rmse={extra_views[other].get('rmse_seconds')}", flush=True)
            continue
        try:
            _, va = matched_train_val(df, other)
            official = load_base_preds(cfg, other).select(
                ["MVT_ID_mvt", pl.col("prediction").alias("base_prediction")]
            )
            missing_f3 = [c for c in f3_nums if c not in va.columns]
            if missing_f3:
                va = va.join(df.select(["MVT_ID_mvt", *missing_f3]), on="MVT_ID_mvt", how="left")
            va = va.join(official, on="MVT_ID_mvt", how="inner")
            if keys:
                tr_o, _ = matched_train_val(df, other)
                cell_o = tr_o.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_base"))
                va = va.join(cell_o, on=keys, how="left").with_columns(
                    pl.col("route_base").fill_null(float(tr_o["TAXITIME_SEC_mvt"].median()))
                )
            va = va.with_columns(pl.col("base_prediction").alias("oof_prediction"))
            if "pred_risk" in res_nums and "pred_risk" not in va.columns:
                risk_other = dest.parent / "CPU-03-error-predictability" / f"risk_predictions_{other}.parquet"
                if risk_other.exists():
                    va = va.join(pl.read_parquet(risk_other), on="MVT_ID_mvt", how="left")
                if "pred_risk" not in va.columns:
                    va = va.with_columns(pl.lit(0.0).alias("pred_risk"))
            if "p_long" in res_nums and "p_long" not in va.columns:
                va = va.with_columns(pl.lit(0.0).alias("p_long"))
            if best_tag == "lgb" and booster is not None:
                rhat_o = booster.predict(numeric_matrix(va, res_nums))
            else:
                rhat_o = predict_unclipped(model, va, cats, res_nums)
            pred_o = np.clip(va["base_prediction"].to_numpy() + rhat_o, 0, None)
            scored_o = _score_composed(cfg, other, va, pred_o, dest, tag=best_tag)
            extra_views[other] = {k: scored_o.get(k) for k in ("rmse_seconds", "sse", "vs_base")}
            persist_repo_file(cfg, dest / f"predictions_{other}.parquet")
            print(f"extra view {other} rmse={extra_views[other].get('rmse_seconds')}", flush=True)
        except Exception as exc:
            print(f"extra view {other} failed: {exc!r}", flush=True)
            extra_views[other] = extra_view_exception(exc, view=other)
    return {
        "status": "completed",
        "rmse_seconds": best["rmse_seconds"],
        "delta_sse_vs_base": (best.get("vs_base") or {}).get("delta_sse_vs_base"),
        "best_tag": best_tag,
        "tune_rmse_cb": tune_rmse_cb,
        "tune_rmse_lgb": tune_rmse_lgb,
        "catboost": {k: composed.get(k) for k in ("rmse_seconds", "sse", "vs_base")},
        "lightgbm": {k: lgb_m.get(k) for k in ("rmse_seconds", "sse", "vs_base", "status", "error") if k in lgb_m} if isinstance(lgb_m, dict) else lgb_m,
        "views": {view: best, **extra_views},
        "conclusion": "residual learner selected on held-out E37 tune month; official views use a full-train refit",
        "hypothesis": "second-stage residual model removes remaining BASE error",
    }


def job_cpu12_geometry(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    osm = locate(Path("data/external/osm/parking_positions.parquet"), cfg)
    if osm is None:
        return {
            "status": "skipped",
            "reason": "OSM parking_positions.parquet is not present; F4 geometry parked",
            "conclusion": "PARK until OSM stands are downloaded",
            "hypothesis": "stand-to-runway geometry adds route signal",
        }
    from prc_challenge.geometry import add_geometry_features
    from prc_challenge.runways import load_runway_thresholds

    df = load_or_build_f3(cfg, load_cache(cfg))
    rwy = locate(Path("data/external/ourairports/runways.csv"), cfg)
    thresholds = load_runway_thresholds(rwy) if rwy is not None else load_runway_thresholds()
    df = add_geometry_features(df, osm_path=osm, thresholds=thresholds)
    cats, nums = feature_lists("e08b")
    f3_nums = [c for c in f3_numeric_columns() if c in df.columns]
    geo = [c for c in ("stand_to_rwy_m", "stand_number", "stand_coord_missing", "rwy_coord_missing") if c in df.columns]
    geo_cat = [c for c in ("stand_prefix",) if c in df.columns]
    view = _primary_view(cfg)
    train, val = matched_train_val(df, view)
    ablations = {
        "BASE+F4": (cats + geo_cat, nums + geo),
        "BASE+F3+F4": (cats + geo_cat, list(dict.fromkeys(nums + f3_nums + geo))),
    }
    table = {}
    progress = {} if cfg.force_rerun else _load_json_dict(dest / "geometry_progress.json")
    best = None
    for name, (c, n) in ablations.items():
        n = [col for col in n if col in train.columns]
        prior = progress.get(name) if isinstance(progress.get(name), dict) else None
        if prior and prior.get("rmse_seconds") is not None and not cfg.force_rerun:
            print(f"CPU-12 resume skip {name} rmse={prior['rmse_seconds']}", flush=True)
            table[name] = prior
            if best is None or prior["rmse_seconds"] < best[1]:
                best = (name, prior["rmse_seconds"])
            continue
        model = fit_regressor(cfg, train, c, n)
        pred = predict_regressor(model, val, c, n)
        table[name] = _compact_score(_score_composed(cfg, view, val, pred, dest, tag=name.replace("+", "_")))
        progress[name] = table[name]
        _checkpoint_job(cfg, dest, "geometry_progress.json", progress)
        if best is None or table[name]["rmse_seconds"] < best[1]:
            best = (name, table[name]["rmse_seconds"])
    extra_views: dict[str, Any] = {}
    if best:
        promote_predictions(dest, view, best[0].replace("+", "_"))
        win_cats, win_nums = ablations[best[0]]
        if _missing_other_view_preds(dest, cfg, view) or cfg.force_rerun:
            extra_views = _refit_score_other_views(
                cfg,
                dest,
                df,
                win_cats,
                [c for c in win_nums if c in df.columns],
                primary=view,
                tag=best[0].replace("+", "_"),
            )
        else:
            print("CPU-12 resume skip extra-view refits", flush=True)
            extra_views = _resumed_extra_views(cfg, dest, df, view, tag=best[0].replace("+", "_"))
    cover = float(df.filter(pl.col("stand_coord_missing") == 0).height / df.height) if "stand_coord_missing" in df.columns else None
    osm_meta = locate(Path("data/external/osm/manifest.json"), cfg)
    source = {
        "parking_positions": str(osm),
        "runways": str(rwy) if rwy is not None else None,
        "osm_manifest": json.loads(osm_meta.read_text()) if osm_meta and osm_meta.exists() else None,
        "note": "F4 is stand→runway haversine. Taxiway shortest-path is parked: OSM extract is parking_position, not a route graph.",
    }
    return {
        "status": "completed",
        "rmse_seconds": best[1] if best else None,
        "best_ablation": best[0] if best else None,
        "ablations": table,
        "coverage": cover,
        "delta_sse_vs_base": (table.get(best[0], {}).get("vs_base") or {}).get("delta_sse_vs_base") if best else None,
        "views": {view: {"rmse_seconds": best[1] if best else None}, **extra_views},
        "conclusion": "F4 geometry vs BASE+F3+F4; taxiway graph parked (stands-only OSM)",
        "hypothesis": "OSM stand-runway distance is a ranking-safe route baseline",
        "source": source,
        "runways": str(rwy) if rwy is not None else None,
    }


from prc_challenge.campaign.ensemble import (
    CPU_BLEND_IDS,
    blend_completed_candidates,
    load_candidate_preds as _load_candidate_preds,
    load_weight_frame as _load_weight_frame,
    simplex_weights as _simplex_weights,
)


def job_cpu13_ensemble(cfg: CampaignConfig, dest: Path) -> dict[str, Any]:
    """CPU-only blend of CPU-01b, CPU-02, CPU-04 F3, CPU-05, CPU-09, CPU-10, CPU-11. GPU-09 repeats after sequences."""
    view = _primary_view(cfg)
    # Weights from _load_weight_frame / _load_candidate_preds (tune_predictions.parquet / oof only).
    out = blend_completed_candidates(cfg, dest, view=view, candidate_ids=CPU_BLEND_IDS)
    out.setdefault("hypothesis", "complementary OOF models improve composed RMSE")
    return out


CPU_JOBS: list[tuple[str, str, Any]] = [
    ("CPU-00-t31-collect", "collect t31 V1/V2/V3 and pick BASE", job_cpu00_t31),
    ("CPU-01-e34-optuna", "E34 bunching Optuna on internal split", job_cpu01_e34_optuna),
    ("CPU-01b-e34-shortlist", "score E34 Optuna winner on V1/V2/V3", job_cpu01b_shortlist),
    ("CPU-02-airport-experts", "airport-specific CatBoost + OOF shrinkage", job_cpu02_experts),
    ("CPU-03-error-predictability", "predict BASE log1p squared error", job_cpu03_error),
    ("CPU-04-f3-ablation", "F3 airport-state feature ablations", job_cpu04_f3),
    ("CPU-05-route-queue", "route/base + congestion residual", job_cpu05_route_queue),
    ("CPU-06-mixture", "soft P(long) mixture of regressors", job_cpu06_mixture),
    ("CPU-07-unmatched-link", "reconstruct unmatched context", job_cpu07_unmatched),
    ("CPU-08-delta-target", "alternate planning-delta target; no AOBT", job_cpu08_delta),
    ("CPU-08b-local-aobt", "LOCAL_ONLY AOBT check; skipped unless allow_local_aobt", job_cpu08b_local_aobt),
    ("CPU-09-lgb-xgb", "LightGBM/XGBoost/ExtraTrees diversity on F3", job_cpu09_diversity),
    ("CPU-10-knn", "historical analog kNN on F3", job_cpu10_knn),
    ("CPU-11-residual", "second-stage residual on month-blocked OOF", job_cpu11_residual),
    ("CPU-12-geometry", "OSM F4 stand-runway distance", job_cpu12_geometry),
    ("CPU-13-ensemble", "OOF non-negative blend of completed candidates", job_cpu13_ensemble),
]


def run_cpu_campaign(cfg: CampaignConfig) -> list[dict[str, Any]]:
    print("=== CPU experiment inventory ===", flush=True)
    for exp_id, hypothesis, _fn in CPU_JOBS:
        print(f"{exp_id:28s} {hypothesis}", flush=True)
    print("hydrate", hydrate_from_drive(cfg), flush=True)
    print(f"index synced n={len(sync_index_from_job_dirs(cfg))}", flush=True)
    results = []
    for exp_id, hypothesis, fn in CPU_JOBS:
        results.append(run_isolated(cfg, exp_id, hypothesis, lambda dest, f=fn: f(cfg, dest)))
    catalog = campaign_job_catalog()
    write_campaign_summary(cfg, "cpu", catalog)
    print_campaign_status(inventory_status(cfg, catalog))
    return results
