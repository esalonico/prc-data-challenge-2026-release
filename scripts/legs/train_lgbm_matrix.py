# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""LightGBM on an assembled S4 matrix (janjul scoring or ranking predictions in the trainer's schema).

Arms V1 / V1b of the VM programme and the LightGBM leg of the all-year blend
(progress-review/vm-arms-20260926/preregistration.md).

  1. Load ``runs/<tag>/{train,eval}_matrix.parquet`` (feature lists from ``matrix_identity.json``).
     The eval mode is detected from the eval matrix: Jan+Jul 2025 -> ``janjul`` (train months may
     not include 1, 7, 11, 12); the ranking months -> ``ranking`` (all twelve months allowed,
     November/December as blind training rows, as the CatBoost all-year fit).
  2. Residual target ``TAXITIME_SEC_mvt - unimpeded_p10`` as the S4 CatBoost.
  3. Either ``--fixed-rounds N`` (one fit on every training row, N rounds; used for the all-year
     ranking fit with the round count found on D) or stage A early stopping on a holdout month
     (``--holdout-month 10``; never Jan/Jul, never Nov/Dec) followed by stage B, a refit on all
     training rows at the found round count.
  4. Raw prediction = ``unimpeded_p10`` + residual, clipped >= 0.  janjul: composed and scored with
     ``compose_janjul_primary`` (E17 -> E18 -> E33b freeze -> clip), card as the trainer prints it.
     ranking: ``compose_ranking_matched`` and the trainer's ranking schema
     (``ranking_lgbm_<tag>.parquet``; ``s4_raw_prediction`` is the raw LightGBM prediction to blend
     on, ``prediction`` the gated one; unmatched rows are flagged and owned by the unmatched overlay).

``--per-airport`` (SA-PERAP, D-081 item 7): one booster per AIRPORT with the same matrix, parameters and
target (``run_per_airport``; stage A per airport on the holdout month, or ``--per-airport-rounds``).
``--no-score`` writes JAN_JUL predictions without the scorecard.

Memory: the training rows live in float32 numpy blocks (categoricals as integer codes); stage B
feeds the fit and holdout blocks to LightGBM as a list, nothing is concatenated.  Peak RSS on the
1.4M x 152 D matrix is about 3.3 GB.

Examples::

  PYTHONPATH=src python scripts/legs/train_lgbm_matrix.py --run-dir <runs>/vm_base --tag vm_v1b_lgbm_plain \
      --threads 2 --linear-tree 0 --control <runs>/dfit_m1v1s_aobt_historder/run_manifest.json
  PYTHONPATH=src python scripts/legs/train_lgbm_matrix.py --run-dir <runs>/allyear_full_m1m2p_aobt \
      --tag vm_allyear_lgbm_full --threads 2 --linear-tree 0 --fixed-rounds 2028
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prc_challenge import matrix_arms as ma
from prc_challenge import s4_allyear as s4

SCRIPT_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message} (peak_rss={s4.peak_rss_gb():.2f} GB)", flush=True)


def lgbm_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": 1.0,
        "lambda_l2": args.lambda_l2,
        "linear_tree": bool(args.linear_tree),
        "linear_lambda": args.linear_lambda,
        "max_bin": args.max_bin,
        "num_threads": args.threads,
        "seed": args.seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }


def run_per_airport(args: argparse.Namespace, lgb: Any, run: Path, out: Path, src: Path, target: Any, fit_label: Any, invert: Any) -> None:
    """SA-PERAP (D-081 item 7): one LightGBM booster per AIRPORT, same matrix / parameters / target as the global leg.

    Stage A (default): per airport, early stopping on that airport's holdout-month rows (patience
    ``--early-stopping``, cap ``--rounds``); every airport's best_iteration goes to
    ``per_airport/rounds.json``.  Stage B: per airport, refit on all of that airport's training rows at
    its own count.  With ``--per-airport-rounds rounds.json`` the holdout stage is skipped and each
    airport is fitted once at the given count (the all-year ranking fits).  No per-airport tuning:
    every airport gets the same parameters.  Evaluation rows are predicted by their own airport's
    booster, then inverted and composed exactly as the global trainer does.
    """

    from prc_challenge import per_airport_lgbm as pa

    if args.fixed_rounds is not None:
        raise SystemExit("--per-airport uses --per-airport-rounds, not --fixed-rounds")
    t0 = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    train, evaluation, cats, nums, matrix_info = ma.load_run_matrices(run)
    mode = matrix_info["eval_mode"]
    if args.max_train_rows and train.height > args.max_train_rows:
        train = train.sample(n=args.max_train_rows, seed=args.seed, shuffle=True)
        log(f"SMOKE TEST: training matrix subsampled to {train.height} rows")
    fixed = pa.load_rounds(Path(args.per_airport_rounds)) if args.per_airport_rounds else None
    use_holdout = fixed is None
    if use_holdout and args.holdout_month not in matrix_info["train_months"]:
        raise SystemExit(f"holdout month {args.holdout_month} not in training months {matrix_info['train_months']}")
    levels = ma.category_levels(train, cats)
    feature_names = [*cats, *nums]
    codes = pa.airports_of(train)
    airports = pa.airport_list(codes)
    eval_airports = pa.airport_list(pa.airports_of(evaluation))
    if set(eval_airports) - set(airports):
        raise SystemExit(f"evaluation airports without training rows: {sorted(set(eval_airports) - set(airports))}")
    if fixed is not None and set(fixed) != set(airports):
        raise SystemExit(f"rounds file airports {sorted(fixed)} != training airports {airports}")
    months = train["month"].cast(pl.Int16).to_numpy()
    y_all = fit_label(train)
    label_all = train[s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
    inv_all = train.select(list(dict.fromkeys(["unimpeded_p10", *(target.columns() if target else [])])))
    X_all = ma.numpy_matrix(train, cats, nums, levels)
    rows_per_airport = {a: int((codes == a).sum()) for a in airports}
    del train
    gc.collect()
    stage_seconds["load_frames"] = time.perf_counter() - t0
    params = lgbm_params(args)
    model_dir = out / "per_airport"
    model_dir.mkdir(parents=True, exist_ok=True)
    log(f"per-airport: mode={mode} X={X_all.shape} airports={airports} rows={rows_per_airport} params={json.dumps(params)}")

    stage_a: dict[str, Any] = {}
    rounds: dict[str, int] = {}
    t2 = time.perf_counter()
    for apt in airports:
        apt_dir = model_dir / apt
        apt_dir.mkdir(exist_ok=True)
        if not use_holdout:
            rounds[apt] = int(fixed[apt])
            continue
        info_path, model_a = apt_dir / "stageA_info.json", apt_dir / "model_stageA_holdout.txt"
        if info_path.exists() and model_a.exists() and not args.force:
            stage_a[apt] = json.loads(info_path.read_text())
            log(f"{apt}: reusing stage A best_iteration={stage_a[apt]['best_iteration']}")
        else:
            in_apt = codes == apt
            fit_idx = np.flatnonzero(in_apt & (months != args.holdout_month))
            hold_idx = np.flatnonzero(in_apt & (months == args.holdout_month))
            if hold_idx.size == 0:
                raise SystemExit(f"{apt}: no holdout-month rows")
            hold_inv = inv_all[hold_idx]
            booster_a, info = pa.fit_stage_a(
                lgb, params, X_all[fit_idx], y_all[fit_idx], X_all[hold_idx], y_all[hold_idx], feature_names, cats,
                args.rounds, args.early_stopping, args.log_every, lambda t, f=hold_inv: invert(f, t), label_all[hold_idx],
            )
            info = {"airport": apt, "holdout_month": args.holdout_month, "rounds_cap": int(args.rounds), "early_stopping_rounds": int(args.early_stopping),
                    "learning_rate": float(args.learning_rate), "num_leaves": int(args.num_leaves), "min_data_in_leaf": int(args.min_data_in_leaf), **info}
            booster_a.save_model(str(model_a))
            s4.atomic_json(info_path, info)
            stage_a[apt] = info
            del booster_a
            gc.collect()
            log(f"{apt}: stage A best_iteration={info['best_iteration']} (run {info['rounds_run']}) holdout rows={info['holdout_rows']} "
                f"holdout_taxi_rmse={info['holdout_taxi_rmse_at_best']:.3f} {info['seconds']:.0f}s")
        rounds[apt] = int(stage_a[apt]["best_iteration"])
    s4.atomic_json(model_dir / "rounds.json", rounds)
    stage_seconds["stage_a"] = time.perf_counter() - t2
    if use_holdout:
        s4.atomic_json(out / "stageA_per_airport.json", {"holdout_month": args.holdout_month, "per_airport_rounds": rounds, "stage_a": stage_a})
    if args.stage_a_only:
        if not use_holdout:
            raise SystemExit("--stage-a-only needs the holdout stage (no --per-airport-rounds)")
        log("stage-A-only run: stopping before stage B / evaluation")
        return

    t3 = time.perf_counter()
    model_file = "model_stageB_allD.txt" if use_holdout else "model_fixed_rounds.txt"
    boosters: dict[str, Any] = {}
    for apt in airports:
        path = model_dir / apt / model_file
        if path.exists() and not args.force:
            boosters[apt] = lgb.Booster(model_file=str(path))
            if boosters[apt].current_iteration() != rounds[apt]:
                raise SystemExit(f"{apt}: stored model has {boosters[apt].current_iteration()} rounds, expected {rounds[apt]}")
            log(f"{apt}: reusing fitted model ({rounds[apt]} rounds)")
            continue
        idx = np.flatnonzero(codes == apt)
        tb = time.perf_counter()
        booster = pa.fit_fixed(lgb, params, X_all[idx], y_all[idx], feature_names, cats, rounds[apt], args.log_every)
        booster.save_model(str(path))
        boosters[apt] = booster
        gc.collect()
        log(f"{apt}: fit {rounds[apt]} rounds on {idx.size} rows, {time.perf_counter() - tb:.0f}s")
    del X_all, y_all
    gc.collect()
    stage_seconds["fit"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    X_eval = ma.numpy_matrix(evaluation, cats, nums, levels)
    t_hat = pa.predict(boosters, X_eval, pa.airports_of(evaluation))
    del X_eval
    raw = invert(evaluation, t_hat)
    manifest: dict[str, Any] = {
        "arm": "per-airport-scope LightGBM on the S4 matrix (plain leaves)" if not params["linear_tree"] else "per-airport-scope LightGBM (linear_tree)",
        "tag": args.tag,
        "target": target.as_dict() if target else "default (label - unimpeded_p10)",
        "created_utc": s4.now_utc(),
        "eval_mode": mode,
        "recipe": "per-airport LightGBM: one booster per AIRPORT on that airport's training rows; "
        + ("early stopping per airport on its holdout-month rows, refit per airport on all its training rows at its own best_iteration"
           if use_holdout else f"fixed per-airport rounds from {args.per_airport_rounds}"),
        "matrix": matrix_info,
        "features": {"cats": cats, "nums": nums, "n": len(cats) + len(nums), "feature_schema_sha256": s4.json_sha256([cats, nums])},
        "categorical_levels": {c: len(v) for c, v in levels.items()},
        "params": params,
        "per_airport": {
            "airports": airports,
            "rows": rows_per_airport,
            "rounds": rounds,
            "rounds_source": "stage A (this run)" if use_holdout else {"path": str(Path(args.per_airport_rounds).resolve()), "sha256": s4.sha256_file(Path(args.per_airport_rounds))},
            "model_dir": str(model_dir),
            "model_file": model_file,
            "model_sha256": pa.model_hashes(model_dir, model_file, s4.sha256_file),
        },
        "rounds_requested": args.rounds if use_holdout else None,
        "early_stopping_rounds": args.early_stopping if use_holdout else None,
        "stage_a": {"holdout_month": args.holdout_month, "per_airport": stage_a} if use_holdout else None,
        "final_rounds": rounds,
        "smoke_subsample": args.max_train_rows,
        "nov_dec_labels_used": matrix_info["nov_dec_labels_used"],
        "nov_dec_role": matrix_info["nov_dec_role"],
        "python": sys.version.split()[0],
        "lightgbm": lgb.__version__,
    }
    if mode == "janjul":
        pred_path = out / f"predictions_{args.tag}_janjul.parquet"
        if args.no_score:
            manifest["predictions"] = ma.write_janjul_predictions(ma.compose_janjul_unscored(evaluation, raw, src), pred_path)
            manifest["scorecard"] = None
            manifest["scoring"] = "not scored (--no-score)"
        else:
            scored, card = ma.score_janjul(evaluation, raw, src)
            manifest["predictions"] = ma.write_janjul_predictions(scored, pred_path)
            manifest["scorecard"] = card
    else:
        composed, summary = ma.compose_ranking(evaluation, raw)
        pred_path = out / f"ranking_lgbm_{args.tag}.parquet"
        s4.atomic_parquet(composed, pred_path)
        manifest["predictions"] = {"path": str(pred_path), "sha256": s4.sha256_file(pred_path), **summary}
        manifest["composition"] = {"order": ["E17", "E18", "clip>=0"], "note": "unmatched rows are owned by the unmatched overlay; blend on s4_raw_prediction of matched rows"}
        if args.compare:
            manifest["vs_reference"] = ma.compare_ranking(composed, Path(args.compare))
    stage_seconds["predict_compose"] = time.perf_counter() - t4
    manifest["stage_seconds"] = {k: round(v, 1) for k, v in stage_seconds.items()}
    manifest["peak_rss_gb"] = s4.peak_rss_gb()
    s4.atomic_json(out / "run_manifest.json", manifest)
    p = manifest["predictions"]
    log(f"predictions: {p['path']} sha {p['sha256'][:16]}")
    if manifest.get("vs_reference"):
        r = manifest["vs_reference"]
        log(f"vs reference (matched): r={r['pearson_r']:.6f} mean|d|={r['mean_abs_diff']:.3f} max|d|={r['max_abs_diff']:.1f}")
    log(f"run manifest written: {out / 'run_manifest.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="runs/<tag> holding train_matrix.parquet + eval_matrix.parquet")
    parser.add_argument("--tag", required=True, help="arm tag (output directory name)")
    parser.add_argument("--out", help="output namespace (default: <run-dir>/../<tag>)")
    parser.add_argument("--source-root", help="checkout holding artifacts/ (E33b parent); default: resolve_source_root")
    parser.add_argument("--control", help="janjul: run_manifest.json of the control fit for the delta columns")
    parser.add_argument("--compare", help="ranking: a trainer-schema ranking parquet to compare matched predictions with")
    parser.add_argument("--fixed-rounds", type=int, help="skip the holdout stage: one fit on all training rows with this many rounds")
    parser.add_argument("--holdout-month", type=int, default=10, help="early-stopping holdout month (default 10; never 1, 7, 11, 12)")
    parser.add_argument("--rounds", type=int, default=4000)
    parser.add_argument("--early-stopping", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=255)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--linear-lambda", type=float, default=0.0)
    parser.add_argument("--linear-tree", type=int, default=1, help="1 = linear_tree (default), 0 = plain leaves")
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int, help="smoke tests only: subsample the training matrix")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--target-offset-col", default=None, help="non-default target t = sign*(y - offset) as the S4 trainer (default: label - unimpeded_p10)")
    parser.add_argument("--target-sign", type=int, choices=(1, -1), default=1)
    parser.add_argument("--target-offset-fill-col", default=None)
    parser.add_argument("--target-winsor", type=float, default=None, help="fit-label winsorization (seconds); never touches evaluation")
    parser.add_argument("--stage-a-only", action="store_true", help="bake: stop after stage A (stageA_info.json); no refit, no evaluation rows")
    parser.add_argument("--per-airport", action="store_true", help="SA-PERAP: one booster per AIRPORT (same matrix, parameters and target); see run_per_airport")
    parser.add_argument("--per-airport-rounds", help="--per-airport: rounds.json ({airport: rounds}) of a stage-A run; skips the holdout stage (the per-airport --fixed-rounds)")
    parser.add_argument("--no-score", action="store_true", help="janjul: write the composed predictions without computing the scorecard (no evaluation label is read)")
    args = parser.parse_args(argv)

    import lightgbm as lgb

    target = (
        s4.TargetSpec(offset_col=args.target_offset_col, sign=args.target_sign, fill_col=args.target_offset_fill_col, winsor_sec=args.target_winsor)
        if args.target_offset_col
        else None
    )

    def fit_label(frame: pl.DataFrame) -> np.ndarray:
        if target is None:
            return ma.residual_target(frame)
        return target.with_fit_label(frame)[s4.RESIDUAL_LABEL].to_numpy()

    def invert(frame: pl.DataFrame, t_hat: np.ndarray) -> np.ndarray:
        if target is None:
            return ma.raw_from_residual(frame, t_hat)
        return target.predict(frame, t_hat)

    src = s4.resolve_source_root(args.source_root)
    run = Path(args.run_dir).resolve()
    out = Path(args.out).resolve() if args.out else run.parent / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if args.holdout_month in ma.FORBIDDEN_TRAIN_MONTHS:
        raise SystemExit(f"holdout month {args.holdout_month} is an evaluation/locked month")
    if args.per_airport or args.per_airport_rounds:
        if not args.per_airport:
            raise SystemExit("--per-airport-rounds needs --per-airport")
        run_per_airport(args, lgb, run, out, src, target, fit_label, invert)
        return
    t0 = time.perf_counter()
    stage_seconds: dict[str, float] = {}

    train, evaluation, cats, nums, matrix_info = ma.load_run_matrices(run)
    mode = matrix_info["eval_mode"]
    if args.max_train_rows and train.height > args.max_train_rows:
        train = train.sample(n=args.max_train_rows, seed=args.seed, shuffle=True)
        log(f"SMOKE TEST: training matrix subsampled to {train.height} rows")
    use_holdout = args.fixed_rounds is None
    if use_holdout and args.holdout_month not in matrix_info["train_months"]:
        raise SystemExit(f"holdout month {args.holdout_month} not in training months {matrix_info['train_months']}")
    log(f"matrices: mode={mode} train={train.shape} eval={evaluation.shape} cats={len(cats)} nums={len(nums)} months={matrix_info['train_months']} nov_dec={matrix_info['nov_dec_role']}")
    stage_seconds["load"] = time.perf_counter() - t0

    # Float32 blocks; the polars training frame is dropped right after.
    t1 = time.perf_counter()
    levels = ma.category_levels(train, cats)
    feature_names = [*cats, *nums]
    if use_holdout:
        is_hold = pl.col("month") == args.holdout_month
        fit_frame = train.filter(~is_hold)
        hold_frame = train.filter(is_hold)
    else:
        fit_frame = train
        hold_frame = train.head(0)
    del train
    gc.collect()
    X_fit = ma.numpy_matrix(fit_frame, cats, nums, levels)
    y_fit = fit_label(fit_frame)
    del fit_frame
    X_hold = ma.numpy_matrix(hold_frame, cats, nums, levels)
    y_hold = fit_label(hold_frame)
    hold_label = hold_frame[s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
    hold_inv = hold_frame.select(list(dict.fromkeys(["unimpeded_p10", *(target.columns() if target else [])])))
    del hold_frame
    gc.collect()
    stage_seconds["frames"] = time.perf_counter() - t1
    params = lgbm_params(args)
    log(f"blocks ready: fit={X_fit.shape} holdout={X_hold.shape} eval_rows={evaluation.height} params={json.dumps(params)}")

    stage_a_info: dict[str, Any] | None = None
    stage_a_path = out / "model_stageA_holdout.txt"
    if use_holdout:
        # Stage A: early stopping on the holdout month.
        t2 = time.perf_counter()
        stage_a_info_path = out / "stageA_info.json"
        if stage_a_path.exists() and stage_a_info_path.exists() and not args.force:
            stage_a_info = json.loads(stage_a_info_path.read_text())
            best_rounds = int(stage_a_info["best_iteration"])
            log(f"reusing stage A: best_iteration={best_rounds}")
        else:
            d_fit = lgb.Dataset(X_fit, label=y_fit, feature_name=feature_names, categorical_feature=cats, free_raw_data=True)
            d_hold = lgb.Dataset(X_hold, label=y_hold, feature_name=feature_names, categorical_feature=cats, reference=d_fit, free_raw_data=True)
            evals: dict[str, Any] = {}
            booster_a = lgb.train(
                params,
                d_fit,
                num_boost_round=args.rounds,
                valid_sets=[d_hold],
                valid_names=["holdout"],
                callbacks=[lgb.early_stopping(args.early_stopping, verbose=True), lgb.log_evaluation(args.log_every), lgb.record_evaluation(evals)],
            )
            best_rounds = int(booster_a.best_iteration or booster_a.current_iteration())
            curve = [float(v) for v in evals["holdout"]["rmse"]]
            hold_pred = booster_a.predict(X_hold, num_iteration=best_rounds)
            hold_raw = invert(hold_inv, np.asarray(hold_pred, dtype=np.float64))
            stage_a_info = {
                "holdout_month": args.holdout_month,
                "fit_rows": int(X_fit.shape[0]),
                "holdout_rows": int(X_hold.shape[0]),
                "best_iteration": best_rounds,
                "rounds_run": int(booster_a.current_iteration()),
                "holdout_residual_rmse_at_best": float(curve[best_rounds - 1]) if best_rounds <= len(curve) else None,
                "holdout_taxi_rmse_at_best": ma.rmse(hold_label, hold_raw),
                "curve_every_100": curve[::100],
                "seconds": round(time.perf_counter() - t2, 1),
                "rounds_cap": int(args.rounds),
                "early_stopping_rounds": int(args.early_stopping),
                "learning_rate": float(args.learning_rate),
                "num_leaves": int(args.num_leaves),
                "min_data_in_leaf": int(args.min_data_in_leaf),
            }
            booster_a.save_model(str(stage_a_path))
            s4.atomic_json(stage_a_info_path, stage_a_info)
            del d_fit, d_hold, booster_a, evals, hold_pred, hold_raw
            gc.collect()
            log(f"stage A done: best_iteration={best_rounds} holdout_taxi_rmse={stage_a_info['holdout_taxi_rmse_at_best']:.4f}")
        stage_seconds["stage_a"] = time.perf_counter() - t2
    else:
        best_rounds = int(args.fixed_rounds)
        log(f"fixed rounds: {best_rounds} (no holdout stage)")
    if args.stage_a_only:
        if not use_holdout:
            raise SystemExit("--stage-a-only needs the holdout stage (no --fixed-rounds)")
        log("stage-A-only run: stopping before stage B / evaluation")
        return

    # Stage B: fit on all training rows at the round count (row blocks passed as a list: no concatenation).
    t3 = time.perf_counter()
    model_path = out / ("model_fixed_rounds.txt" if not use_holdout else "model_stageB_allD.txt")
    if model_path.exists() and not args.force:
        booster = lgb.Booster(model_file=str(model_path))
        log("reusing fitted model")
    else:
        n_all = X_fit.shape[0] + X_hold.shape[0]
        blocks = [X_fit, X_hold] if X_hold.shape[0] else X_fit
        labels = np.concatenate([y_fit, y_hold]) if X_hold.shape[0] else y_fit
        d_all = lgb.Dataset(blocks, label=labels, feature_name=feature_names, categorical_feature=cats, params=params, free_raw_data=True)
        d_all.construct()  # with the training params, so linear_tree is fixed at construction
        del X_fit, X_hold, y_fit, y_hold, blocks, labels
        gc.collect()
        booster = lgb.train(params, d_all, num_boost_round=best_rounds, callbacks=[lgb.log_evaluation(args.log_every)])
        booster.save_model(str(model_path))
        del d_all
        gc.collect()
        log(f"fit done: {booster.current_iteration()} rounds on {n_all} rows")
    stage_seconds["fit"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    X_eval = ma.numpy_matrix(evaluation, cats, nums, levels)
    residual_pred = booster.predict(X_eval)
    del X_eval
    raw = invert(evaluation, residual_pred)
    manifest: dict[str, Any] = {
        "arm": "LightGBM on the S4 matrix" + (" (linear_tree)" if params["linear_tree"] else " (plain leaves)"),
        "tag": args.tag,
        "target": target.as_dict() if target else "default (label - unimpeded_p10)",
        "created_utc": s4.now_utc(),
        "eval_mode": mode,
        "recipe": "lightgbm residual (label - unimpeded_p10); "
        + (f"fixed {best_rounds} rounds on all training rows" if not use_holdout else "early stopping on holdout month; refit on all training rows at best_iteration"),
        "matrix": matrix_info,
        "features": {"cats": cats, "nums": nums, "n": len(cats) + len(nums), "feature_schema_sha256": s4.json_sha256([cats, nums])},
        "categorical_levels": {c: len(v) for c, v in levels.items()},
        "params": params,
        "rounds_requested": args.rounds if use_holdout else best_rounds,
        "early_stopping_rounds": args.early_stopping if use_holdout else None,
        "stage_a": stage_a_info,
        "final_rounds": int(booster.current_iteration()),
        "smoke_subsample": args.max_train_rows,
        "models": {"stage_a": str(stage_a_path) if use_holdout else None, "final": str(model_path), "final_sha256": s4.sha256_file(model_path)},
        "nov_dec_labels_used": matrix_info["nov_dec_labels_used"],
        "nov_dec_role": matrix_info["nov_dec_role"],
        "python": sys.version.split()[0],
        "lightgbm": lgb.__version__,
    }
    if mode == "janjul" and args.no_score:
        pred_path = out / f"predictions_{args.tag}_janjul.parquet"
        manifest["predictions"] = ma.write_janjul_predictions(ma.compose_janjul_unscored(evaluation, raw, src), pred_path)
        manifest["scorecard"] = None
        manifest["scoring"] = "not scored (--no-score)"
    elif mode == "janjul":
        scored, card = ma.score_janjul(evaluation, raw, src)
        pred_path = out / f"predictions_{args.tag}_janjul.parquet"
        manifest["predictions"] = ma.write_janjul_predictions(scored, pred_path)
        control = ma.load_control_card(Path(args.control)) if args.control else None
        manifest["scorecard"] = card
        manifest["control"] = {"path": args.control, "scorecard": control} if control else None
        manifest["vs_control"] = ma.card_vs_control(card, control)
    else:
        composed, summary = ma.compose_ranking(evaluation, raw)
        pred_path = out / f"ranking_lgbm_{args.tag}.parquet"
        s4.atomic_parquet(composed, pred_path)
        manifest["predictions"] = {"path": str(pred_path), "sha256": s4.sha256_file(pred_path), **summary}
        manifest["composition"] = {"order": ["E17", "E18", "clip>=0"], "note": "unmatched rows are owned by the unmatched overlay; blend on s4_raw_prediction of matched rows"}
        if args.compare:
            manifest["vs_reference"] = ma.compare_ranking(composed, Path(args.compare))
    stage_seconds["predict_compose"] = time.perf_counter() - t4
    manifest["stage_seconds"] = {k: round(v, 1) for k, v in stage_seconds.items()}
    manifest["peak_rss_gb"] = s4.peak_rss_gb()
    s4.atomic_json(out / "run_manifest.json", manifest)
    if mode == "janjul" and args.no_score:
        log(f"janjul predictions written unscored: {manifest['predictions']['path']}")
    elif mode == "janjul":
        log(ma.card_line(manifest["scorecard"]))
        if manifest.get("vs_control"):
            log("vs control: " + json.dumps({k: round(v["delta_seconds"], 4) for k, v in manifest["vs_control"].items()}))
    else:
        p = manifest["predictions"]
        log(f"ranking: n={p['n']} matched={p['matched_n']} unmatched={p['unmatched_n']} matched_mean={p['matched_pred_mean']:.3f} max={p['matched_pred_max']:.1f} checksum={p['matched_checksum'][:12]}")
        if manifest.get("vs_reference"):
            r = manifest["vs_reference"]
            log(f"vs reference (matched): r={r['pearson_r']:.6f} mean|d|={r['mean_abs_diff']:.3f} max|d|={r['max_abs_diff']:.1f}")
    log(f"run manifest written: {out / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
