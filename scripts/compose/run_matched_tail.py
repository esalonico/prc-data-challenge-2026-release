# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-TAIL runner: proxy D-fold OOF base, T1/T2 components, D gates, JAN_JUL confirmation.

Pre-registration: progress-review/matched-tail-20260926/preregistration.md

Phases (each caches its outputs under artifacts/experiments/Matched-tail-20260926/):

  d_lgbm     per D fold: three proxy LightGBM legs (plain / AOBT-delta / EOBT-residual), the T2
             non-stamp expert E_nc, and the T1 components p_long / mu_long (fold-internal fits)
  d_t2clf    per D fold: the T2 stamp classifier (CatBoost, U2 recipe form)
  d_score    D gates for T1, T2, T1+T2 (proxy OOF base; secondary aobt_plain readout)
  jj_fit     components fitted on the whole D-fit train matrix, predicted on JAN_JUL rows
  jj_score   JAN_JUL composed confirmation vs the six-leg control (same post-processing)

Usage: PYTHONPATH=src PRC_MAIN_REPO=<main> python scripts/compose/run_matched_tail.py <phase> [--threads 3]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_k, "3")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prc_challenge import matched_tail as mt  # noqa: E402
from prc_challenge import matrix_arms as ma  # noqa: E402
from prc_challenge import s4_allyear as s4  # noqa: E402
from prc_challenge.lirf_label_source import copy_feasible, copy_label  # noqa: E402

MAIN = Path(os.environ.get("PRC_MAIN_REPO", Path(__file__).resolve().parents[2]))
RUNS = MAIN / "artifacts" / "experiments" / "S4-allyear-rebuild-20260925" / "runs"
DFIT = RUNS / "dfit_full_adsb_historder"
OUT = MAIN / "artifacts" / "experiments" / "Matched-tail-20260926"
TRACKED = ROOT / "progress-review" / "matched-tail-20260926"
T2_BLOCK = OUT / "features" / "t2_raw_block.parquet"
U2_OOF = MAIN / "artifacts" / "experiments" / "LIRF-label-source-20260925" / "oof" / "u2_oof_D.parquet"
U2_JJ = MAIN / "artifacts" / "experiments" / "LIRF-label-source-20260925" / "confirm" / "u2_janjul_label_free_predictions.parquet"
AOBT_PLAIN = MAIN / "artifacts" / "experiments" / "Matched-screen-20260925" / "aobt_plain" / "oof.parquet"
E08B_CACHE = MAIN / "artifacts" / "cache" / "e08b_e34_months_1_10.parquet"
LEGS = {
    "cb": "dfit_full_adsb_historder",
    "lgb": "vm_v6_lgbm_adsb_dfit",
    "cbd": "dfit_full_adsb_delta_historder",
    "lgbd": "vm_v9_lgbm_delta_adsb_dfit",
    "lgbe": "vm_v10_lgbm_eobtres_adsb_dfit",
    "cbe": "vm_v11_cb_eobtres_adsb_dfit",
}
LEGS10 = {**LEGS, "lgbl": "vm_v12_lgbm_lobtoff_adsb_dfit", "nn14": "nn_v14_mlp_adsb_dfit",
          "nn15": "nn_v15_mlp_resid_adsb_dfit", "nn16": "nn_v16_mlp_eobtres_adsb_dfit"}  # D-061 ten-leg blend (D-062 re-gate)
TARGETS = {
    "plain": s4.TargetSpec(),
    "delta": s4.TargetSpec(offset_col="takeoff_minus_aobt_sec", sign=-1, fill_col="unimpeded_p10", winsor_sec=7200.0),
    "eobt": s4.TargetSpec(offset_col="takeoff_minus_eobt_sec", sign=1, fill_col="unimpeded_p10", winsor_sec=7200.0),
}
FOLD_MONTHS = {k: v for k, v in mt.D_FOLDS.items()}
D_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)

# T2 classifier inputs (U2 recipe form + airport + NM clocks / ADS-B)
ENC_KEYS = ("flight_prefix", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt")
T2_CATS = ("AIRPORT", "sd_bucket")
T2_NUMS = (
    *(f"enc_{k}" for k in ENC_KEYS),
    *(f"logn_{k}" for k in ENC_KEYS),
    "arr_day_copy_rate", "arr_day_n",
    "sched_second_zero", "mvt_second_zero", "iobt_second_zero", "eobt_second_zero",
    "sched_hour", "local_dow",
    "takeoff_minus_sched_sec", "sched_minus_iobt_sec", "sched_minus_eobt_sec",
    "sched_minus_lobt_sec", "sched_minus_aobt_sec", "aobt_missing", "aobt_has_seconds", "adsb_coverage", "sd_minus_adsb_taxi_hat",
)
T2_CLASSIFIER_PARAMS = {
    "loss_function": "Logloss", "iterations": 400, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0,
    "random_seed": 42, "verbose": False, "allow_writing_files": False,
}
MATRIX_T2_COLS = ("takeoff_minus_sched_sec", "takeoff_minus_iobt_sec", "takeoff_minus_eobt_sec", "takeoff_minus_lobt_sec",
                  "takeoff_minus_aobt_sec", "aobt_missing", "aobt_has_seconds", "adsb_coverage", "adsb_taxi_hat",
                  "local_dow", "STAND_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg} (rss={s4.peak_rss_gb():.2f} GB)", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


# ---------------------------------------------------------------------------
# LightGBM components on the D-fit matrix
# ---------------------------------------------------------------------------


def _fit_components(X_tr, frame_tr: pl.DataFrame, X_ap, frame_ap: pl.DataFrame, cats_idx, feature_names, threads: int, cache: Path,
                    which: tuple[str, ...] = ("legs", "e_nc", "long")) -> pl.DataFrame:
    """Fit the proxy legs, E_nc, p_long, mu_long on (X_tr, frame_tr); predict X_ap. Cached per component."""

    import lightgbm as lgb

    cache.mkdir(parents=True, exist_ok=True)
    y = frame_tr[s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
    sd = frame_tr["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    params = {**mt.PROXY_LGBM_PARAMS, "num_threads": threads}
    out: dict[str, np.ndarray] = {}
    ds = None

    def dataset():
        nonlocal ds
        if ds is None:
            ds = lgb.Dataset(X_tr, label=np.zeros(X_tr.shape[0]), feature_name=feature_names, categorical_feature=cats_idx, params=params, free_raw_data=False)
            ds.construct()
        return ds

    for name, spec in (TARGETS.items() if "legs" in which else ()):
        path = cache / f"leg_{name}.npy"
        if path.exists():
            out[f"leg_{name}"] = np.load(path)
            continue
        t0 = time.time()
        t = spec.with_fit_label(frame_tr.select(list(dict.fromkeys([s4.LABEL_COLUMN, "unimpeded_p10", *spec.columns()]))))[s4.RESIDUAL_LABEL].to_numpy()
        d = dataset()
        d.set_label(t)
        booster = lgb.train(params, d, num_boost_round=mt.PROXY_ROUNDS)
        pred = spec.predict(frame_ap, booster.predict(X_ap))
        np.save(path, pred)
        out[f"leg_{name}"] = pred
        log(f"  leg {name}: {time.time() - t0:.0f}s mean {pred.mean():.1f}")
        del booster
        gc.collect()

    path = cache / "e_nc.npy"
    if "e_nc" not in which:
        pass
    elif path.exists():
        out["e_nc"] = np.load(path)
    else:
        t0 = time.time()
        stamp = copy_label(y, sd)
        keep = np.where(~stamp)[0]
        sub = lgb.Dataset(X_tr[keep], label=(y - frame_tr["unimpeded_p10"].cast(pl.Float64).to_numpy())[keep], feature_name=feature_names,
                          categorical_feature=cats_idx, params=params, free_raw_data=True)
        booster = lgb.train(params, sub, num_boost_round=mt.PROXY_ROUNDS)
        pred = np.clip(frame_ap["unimpeded_p10"].cast(pl.Float64).to_numpy() + booster.predict(X_ap), 0, None)
        np.save(path, pred)
        out["e_nc"] = pred
        log(f"  E_nc on {keep.size} non-stamp rows (stamp rate {stamp.mean():.4f}): {time.time() - t0:.0f}s")
        del booster, sub
        gc.collect()
    ds = None
    gc.collect()

    p_path, m_path = cache / "p_long.npy", cache / "mu_long.npy"
    if "long" not in which:
        pass
    elif p_path.exists() and m_path.exists():
        out["p_long"], out["mu_long"] = np.load(p_path), np.load(m_path)
    else:
        t0 = time.time()
        is_long = y > mt.LONG_SEC
        idx, w = mt.negative_sample(is_long)
        clf = lgb.LGBMClassifier(**mt.P_LONG_PARAMS, n_jobs=threads)
        clf.fit(X_tr[idx], is_long[idx].astype(int), sample_weight=w, categorical_feature=cats_idx)
        p = clf.predict_proba(X_ap)[:, 1]
        li = np.where(is_long)[0]
        reg = lgb.LGBMRegressor(**mt.MU_LONG_PARAMS, n_jobs=threads)
        reg.fit(X_tr[li], np.clip(y[li], 0, mt.MU_LONG_CLIP), categorical_feature=cats_idx)
        mu = reg.predict(X_ap)
        np.save(p_path, p)
        np.save(m_path, mu)
        out["p_long"], out["mu_long"] = p, mu
        log(f"  p_long/mu_long ({int(is_long.sum())} long rows of {y.size}): {time.time() - t0:.0f}s")
    return frame_ap.select("MVT_ID_mvt").with_columns([pl.Series(k, v) for k, v in out.items()])


def _matrix_blocks(frame: pl.DataFrame, cats: list[str], nums: list[str], levels) -> np.ndarray:
    return ma.numpy_matrix(frame, cats, nums, levels)


def phase_d_lgbm(threads: int) -> None:
    cats, nums = ma.feature_lists_for_run(DFIT)
    need = list(dict.fromkeys([*ma.ID_COLUMNS, "takeoff_minus_aobt_sec", "takeoff_minus_eobt_sec", *cats, *nums]))
    train = pl.read_parquet(DFIT / "train_matrix.parquet", columns=need)
    months = sorted(train["month"].unique().to_list())
    if months != list(D_MONTHS) or (train["nm_unmatched"] != 0).any():
        raise RuntimeError(f"unexpected D-fit matrix months {months} or unmatched rows")
    levels = ma.category_levels(train, cats)
    X = _matrix_blocks(train, cats, nums, levels)
    train = train.select(list(dict.fromkeys([*ma.ID_COLUMNS, "takeoff_minus_aobt_sec", "takeoff_minus_eobt_sec"])))
    gc.collect()
    names = [*cats, *nums]
    cats_idx = list(range(len(cats)))
    log(f"D matrix {X.shape}")
    parts = []
    for key, hold in FOLD_MONTHS.items():
        path = OUT / "oof" / f"d_lgbm_{key}.parquet"
        if path.exists():
            parts.append(pl.read_parquet(path))
            continue
        m = train["month"].to_numpy()
        tr = np.where(~np.isin(m, hold))[0]
        ap = np.where(np.isin(m, hold))[0]
        log(f"fold {key}: train {tr.size} hold {ap.size}")
        res = _fit_components(X[tr], train[tr], X[ap], train[ap], cats_idx, names, threads, OUT / "cache" / f"d_{key}")
        res = res.with_columns(pl.lit(key).alias("fold"))
        res.write_parquet(path)
        parts.append(res)
        gc.collect()
    pl.concat(parts).write_parquet(OUT / "oof" / "d_lgbm.parquet")
    log("d_lgbm complete")


def phase_jj_fit(threads: int) -> None:
    cats, nums = ma.feature_lists_for_run(DFIT)
    need = list(dict.fromkeys([*ma.ID_COLUMNS, "takeoff_minus_aobt_sec", "takeoff_minus_eobt_sec", *cats, *nums]))
    train = pl.read_parquet(DFIT / "train_matrix.parquet", columns=need)
    ev = pl.read_parquet(DFIT / "eval_matrix.parquet", columns=need).filter(pl.col("nm_unmatched") == 0)
    if sorted(ev["month"].unique().to_list()) != [1, 7]:
        raise RuntimeError("eval matrix is not JAN_JUL")
    levels = ma.category_levels(train, cats)
    Xt = _matrix_blocks(train, cats, nums, levels)
    Xe = _matrix_blocks(ev, cats, nums, levels)
    keep = list(dict.fromkeys([*ma.ID_COLUMNS, "takeoff_minus_aobt_sec", "takeoff_minus_eobt_sec"]))
    train, ev = train.select(keep), ev.select(keep)
    gc.collect()
    passed = [k for k, v in json.loads((OUT / "metrics" / "d_score.json").read_text())["arms"].items() if v["GO"]]
    which = (("long",) if any("T1" in k for k in passed) else ()) + (("e_nc",) if any("T2" in k for k in passed) else ())
    log(f"arms passing D: {passed}; JAN_JUL components {which}")
    res = _fit_components(Xt, train, Xe, ev, list(range(len(cats))), [*cats, *nums], threads, OUT / "cache" / "jj", which=which)
    res.write_parquet(OUT / "oof" / "jj_lgbm.parquet")
    log("jj_fit complete")


# ---------------------------------------------------------------------------
# T2 classifier
# ---------------------------------------------------------------------------


def t2_frame(matrix_path: Path, rows: pl.DataFrame | None = None) -> pl.DataFrame:
    """Matched non-LIRF rows with the T2 inputs (matrix columns + raw block); label kept for training use only."""

    from prc_challenge.lirf_label_source import sd_bucket

    cols = list(dict.fromkeys(["MVT_ID_mvt", "month", "AIRPORT", s4.LABEL_COLUMN, "nm_unmatched", "takeoff_minus_sched_days", *MATRIX_T2_COLS]))
    frame = pl.read_parquet(matrix_path, columns=[c for c in cols if c in pl.read_parquet_schema(matrix_path)])
    frame = frame.filter((pl.col("nm_unmatched") == 0) & (pl.col("AIRPORT") != "LIRF"))
    if rows is not None:
        frame = frame.join(rows, on="MVT_ID_mvt", how="semi")
    raw = pl.read_parquet(T2_BLOCK)
    frame = frame.join(raw, on="MVT_ID_mvt", how="left", validate="1:1")
    if frame["arr_day_n"].null_count() > 0.01 * frame.height:
        raise RuntimeError(f"raw block join failed ({frame['arr_day_n'].null_count()} nulls)")
    f = lambda c: pl.col(c).cast(pl.Float64)  # noqa: E731
    frame = frame.with_columns(
        (f("takeoff_minus_iobt_sec") - f("takeoff_minus_sched_sec")).alias("sched_minus_iobt_sec"),
        (f("takeoff_minus_eobt_sec") - f("takeoff_minus_sched_sec")).alias("sched_minus_eobt_sec"),
        (f("takeoff_minus_lobt_sec") - f("takeoff_minus_sched_sec")).alias("sched_minus_lobt_sec"),
        pl.when(f("aobt_missing") == 1).then(None).otherwise(f("takeoff_minus_aobt_sec") - f("takeoff_minus_sched_sec")).alias("sched_minus_aobt_sec"),
        (f("takeoff_minus_sched_sec") - f("adsb_taxi_hat")).alias("sd_minus_adsb_taxi_hat"),
    )
    for k in ENC_KEYS:
        frame = frame.with_columns((pl.col("AIRPORT") + "|" + pl.col(k).cast(pl.String).fill_null("__UNK__")).alias(f"key_{k}"))
    return frame.with_columns(pl.Series("sd_bucket", sd_bucket(frame["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy())))


def t2_encode(frame: pl.DataFrame, stamp: np.ndarray, fit_mask: np.ndarray, apply_mask: np.ndarray, enc_month: np.ndarray) -> pl.DataFrame:
    from prc_challenge.lirf_label_source import RATE_N0, nested_rate_encoding

    label = np.where(fit_mask, stamp.astype(np.float64), np.nan)
    cols = []
    for k in ENC_KEYS:
        keys = frame[f"key_{k}"].to_numpy().astype(object)
        rate, count = nested_rate_encoding(keys, label, enc_month, fit_mask, apply_mask, n0=RATE_N0)
        cols += [pl.Series(f"enc_{k}", rate), pl.Series(f"logn_{k}", np.log1p(count))]
    return frame.with_columns(cols)


def t2_pool(frame: pl.DataFrame, label: np.ndarray | None = None):
    from catboost import Pool

    bad = {s4.LABEL_COLUMN, "BLOCK_TIME_UTC_mvt"} & set((*T2_CATS, *T2_NUMS))
    if bad:
        raise RuntimeError(f"forbidden T2 inputs {bad}")
    x = frame.select([pl.col(c).cast(pl.String).fill_null("__UNK__") for c in T2_CATS] + [pl.col(c).cast(pl.Float32) for c in T2_NUMS]).to_pandas()
    return Pool(x, label=label, cat_features=list(range(len(T2_CATS))))


def t2_fit_predict(frame: pl.DataFrame, fit_mask: np.ndarray, apply_mask: np.ndarray, threads: int, model_path: Path) -> np.ndarray:
    from catboost import CatBoostClassifier

    y = frame[s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
    sd = frame["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    stamp = np.where(fit_mask, copy_label(np.where(fit_mask, y, np.nan), sd), False)
    months = frame["month"].to_numpy().astype(np.int64)
    enc_month = np.where(fit_mask, months, 100 + months)
    enc = t2_encode(frame, stamp, fit_mask, apply_mask, enc_month)
    feas = mt.stamp_feasible(sd, frame["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), frame["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy(),
                             frame["takeoff_minus_lobt_sec"].cast(pl.Float64).to_numpy())
    tr = fit_mask & feas
    clf = CatBoostClassifier(**T2_CLASSIFIER_PARAMS, thread_count=threads)
    clf.fit(t2_pool(enc.filter(pl.Series(tr)), stamp[tr].astype(int)))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(model_path))
    p = np.full(frame.height, np.nan)
    p[apply_mask] = clf.predict_proba(t2_pool(enc.filter(pl.Series(apply_mask))))[:, 1]
    log(f"  T2 classifier: {int(tr.sum())} feasible training rows, stamp rate {stamp[tr].mean():.4f}")
    return p


def phase_d_t2clf(threads: int) -> None:
    frame = t2_frame(DFIT / "train_matrix.parquet")
    months = frame["month"].to_numpy()
    parts = []
    for key, hold in FOLD_MONTHS.items():
        path = OUT / "oof" / f"d_t2clf_{key}.parquet"
        if not path.exists():
            t0 = time.time()
            fit_mask = ~np.isin(months, hold)
            apply_mask = np.isin(months, hold)
            p = t2_fit_predict(frame, fit_mask, apply_mask, threads, OUT / "models" / f"d_t2clf_{key}.cbm")
            frame.filter(pl.Series(apply_mask)).select("MVT_ID_mvt").with_columns(pl.Series("p_stamp", p[apply_mask]), pl.lit(key).alias("fold")).write_parquet(path)
            log(f"T2 fold {key}: {time.time() - t0:.0f}s")
        parts.append(pl.read_parquet(path))
    pl.concat(parts).write_parquet(OUT / "oof" / "d_t2clf.parquet")


def phase_jj_t2clf(threads: int) -> None:
    tr = t2_frame(DFIT / "train_matrix.parquet").with_columns(pl.lit(True).alias("_fit"))
    ev = t2_frame(DFIT / "eval_matrix.parquet").with_columns(pl.lit(False).alias("_fit"), pl.lit(None, dtype=pl.Float64).alias(s4.LABEL_COLUMN))
    both = pl.concat([tr, ev.select(tr.columns)], how="vertical_relaxed")
    fit_mask = both["_fit"].to_numpy()
    p = t2_fit_predict(both, fit_mask, ~fit_mask, threads, OUT / "models" / "jj_t2clf.cbm")
    both.filter(pl.Series(~fit_mask)).select("MVT_ID_mvt").with_columns(pl.Series("p_stamp", p[~fit_mask])).write_parquet(OUT / "oof" / "jj_t2clf.parquet")
    log("jj_t2clf complete")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _arms(base: np.ndarray, f: pl.DataFrame, apply: np.ndarray, t2_pop: np.ndarray) -> dict[str, np.ndarray]:
    sd = f["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    feas = mt.stamp_feasible(sd, f["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), f["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy(),
                             f["takeoff_minus_lobt_sec"].cast(pl.Float64).to_numpy())
    p_stamp = f["p_stamp"].cast(pl.Float64).to_numpy()
    e_nc = f["e_nc"].cast(pl.Float64).to_numpy()
    t1 = mt.t1_correct(base, f["p_long"].to_numpy(), f["mu_long"].to_numpy(), apply)
    t2 = mt.t2_mixture(p_stamp, sd, e_nc, base, feas, t2_pop & apply)
    t12 = mt.t2_mixture(p_stamp, sd, e_nc, t1, feas, t2_pop & apply)
    return {"control": base, "T1": t1, "T2": t2, "T1+T2": t12}


def _post(pred: np.ndarray, f: pl.DataFrame, matched: np.ndarray) -> np.ndarray:
    sd = f["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    u2_feas = copy_feasible(sd, f["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), f["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy())
    return mt.postprocess(pred, lirf_u2=f["u2_p"].is_not_null().to_numpy() & matched, u2_p=f["u2_p"].fill_null(0.0).to_numpy(),
                          u2_expert=f["u2_expert"].fill_null(0.0).to_numpy(), sd=sd, u2_feasible=u2_feas,
                          lead_lobt=f["takeoff_minus_lobt_sec"].cast(pl.Float64).to_numpy(), matched=matched)


def _auc_block(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    return {"n": int(mask.sum()), "rate": float(y[mask].mean()), "mean_p": float(p[mask].mean()), "auc": float(roc_auc_score(y[mask], p[mask]))}


def phase_d_score() -> None:
    cols = ["MVT_ID_mvt", "month", "AIRPORT", s4.LABEL_COLUMN, "takeoff_minus_sched_sec", "takeoff_minus_iobt_sec", "takeoff_minus_eobt_sec", "takeoff_minus_lobt_sec", "adsb_coverage"]
    f = pl.read_parquet(DFIT / "train_matrix.parquet", columns=cols)
    lg = pl.read_parquet(OUT / "oof" / "d_lgbm.parquet")
    t2 = pl.read_parquet(OUT / "oof" / "d_t2clf.parquet").drop("fold")
    u2 = pl.read_parquet(U2_OOF).select("MVT_ID_mvt", pl.col("p_copy").alias("u2_p"), pl.col("enc_all").alias("u2_expert"))
    day = pl.read_parquet(E08B_CACHE, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day")).drop("MVT_TIME_UTC_mvt")
    f = (f.join(lg, on="MVT_ID_mvt", how="left", validate="1:1").join(t2, on="MVT_ID_mvt", how="left", validate="1:1")
         .join(u2, on="MVT_ID_mvt", how="left", validate="1:1").join(day, on="MVT_ID_mvt", how="left", validate="1:1"))
    if f["leg_plain"].null_count() or f["day"].null_count():
        raise RuntimeError("OOF join incomplete")
    lirf_rows = (f["AIRPORT"] == "LIRF").sum()
    if f["u2_p"].null_count() != f.height - lirf_rows:
        raise RuntimeError("U2 OOF does not cover exactly the LIRF rows")
    y = f[s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
    sd = f["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    airport = f["AIRPORT"].to_numpy()
    fold = f["fold"].to_numpy()
    dday = f["day"].to_numpy()
    matched = np.ones(f.height, dtype=bool)
    t2_pop = airport != "LIRF"
    base = (f["leg_plain"].to_numpy() + f["leg_delta"].to_numpy() + f["leg_eobt"].to_numpy()) / 3
    result: dict[str, Any] = {"population": f"matched D rows of the D-fit train matrix, n={f.height}; proxy OOF base (3 LightGBM target legs)",
                              "post_processing": "U2-M3 (stored D OOF) at LIRF -> LOBT window -> clip"}
    # diagnostics
    is_long = y > mt.LONG_SEC
    stamp = copy_label(y, sd)
    feas = mt.stamp_feasible(sd, f["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), f["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy(),
                             f["takeoff_minus_lobt_sec"].cast(pl.Float64).to_numpy())
    p_stamp = f["p_stamp"].to_numpy()
    result["diagnostics"] = {
        "proxy_leg_rmse": {k: float(np.sqrt(np.mean((y - f[k].to_numpy()) ** 2))) for k in ("leg_plain", "leg_delta", "leg_eobt", "e_nc")},
        "proxy_base_rmse_raw": float(np.sqrt(np.mean((y - base) ** 2))),
        "p_long": _auc_block(is_long, f["p_long"].to_numpy(), matched),
        "long_rows_base_bias": float((y - base)[is_long].mean()),
        "long_share_by_month": {int(m): float(is_long[f["month"].to_numpy() == m].mean()) for m in D_MONTHS},
        "p_stamp_feasible_nonlirf": _auc_block(stamp, p_stamp, t2_pop & feas),
        "stamp_share_nonlirf_by_month": {int(m): float(stamp[t2_pop & (f["month"].to_numpy() == m)].mean()) for m in D_MONTHS},
        "stamp_rows_nonlirf_sse_share_of_matched": float(((y - base) ** 2)[stamp & t2_pop].sum() / ((y - base) ** 2).sum()),
        "feasible_share_nonlirf": float(feas[t2_pop].mean()),
    }
    auc = result["diagnostics"]["p_stamp_feasible_nonlirf"]["auc"]
    arms = _arms(base, f, matched, t2_pop)
    post = {k: _post(v, f, matched) for k, v in arms.items()}
    result["control_rmse_post"] = float(np.sqrt(np.mean((y - post["control"]) ** 2)))
    result["arms"] = {}
    for name in ("T1", "T2", "T1+T2"):
        rep = mt.gate_report(y, post["control"], post[name], fold=fold, airport=airport, day=dday)
        if name != "T1" and auc < 0.75:
            rep["GO"] = False
            rep["stop_rule"] = f"p_stamp AUC {auc:.3f} < 0.75"
        cov = f["adsb_coverage"].fill_null(-9).to_numpy()
        se_c = (y - post["control"]) ** 2
        dse = (y - post[name]) ** 2 - se_c
        rep["by_adsb_coverage"] = {str(c): float(dse[cov == c].sum() / se_c[cov == c].sum()) for c in np.unique(cov)}
        rep["share_of_gain_on_stamp_rows"] = float(dse[stamp].sum() / dse.sum())
        rep["share_of_gain_on_long_rows"] = float(dse[is_long].sum() / dse.sum())
        result["arms"][name] = rep
        log(f"D {name}: dSSE {rep['dsse_rel']*100:+.3f}% p95 {rep['day_bootstrap_p05_p95'][1]:.3e} folds {rep['folds_improved']} worst airport {rep['worst_airport_rel']*100:+.2f}% GO={rep['GO']}")
    # secondary readout: aobt_plain screen OOF as the base
    sec = pl.read_parquet(AOBT_PLAIN).select("MVT_ID_mvt", pl.col("prediction").alias("aobt_plain"))
    g = f.join(sec, on="MVT_ID_mvt", how="left", maintain_order="left")
    if g["aobt_plain"].null_count() == 0:
        b2 = g["aobt_plain"].to_numpy().astype(np.float64)
        arms2 = _arms(b2, g, matched, t2_pop)
        post2 = {k: _post(v, g, matched) for k, v in arms2.items()}
        result["secondary_aobt_plain"] = {name: {k: v for k, v in mt.gate_report(y, post2["control"], post2[name], fold=fold, airport=airport, day=dday).items()
                                                 if k in ("dsse_rel", "day_bootstrap_p05_p95", "folds_improved", "worst_airport_rel", "GO")}
                                          for name in ("T1", "T2", "T1+T2")}
        result["secondary_aobt_plain"]["control_rmse_post"] = float(np.sqrt(np.mean((y - post2["control"]) ** 2)))
    write_json(OUT / "metrics" / "d_score.json", result)
    write_json(TRACKED / "d_score.json", result)
    log("d_score written")


def _janjul_frame(legs: dict[str, str] = LEGS) -> pl.DataFrame:
    base = None
    for k, run in legs.items():
        d = pl.read_parquet(RUNS / run / f"predictions_{run}_janjul.parquet").select(
            "MVT_ID_mvt", "AIRPORT", s4.LABEL_COLUMN, "nm_unmatched", "month", pl.col("s4_raw_prediction").alias(k), pl.col("prediction").alias(f"c_{k}"))
        base = d if base is None else base.join(d.select("MVT_ID_mvt", k, f"c_{k}"), on="MVT_ID_mvt", how="left", validate="1:1", maintain_order="left")
    ev = pl.read_parquet(DFIT / "eval_matrix.parquet", columns=["MVT_ID_mvt", "takeoff_minus_sched_sec", "takeoff_minus_iobt_sec", "takeoff_minus_eobt_sec", "takeoff_minus_lobt_sec", "adsb_coverage", "takeoff_minus_sched_days"])
    u2 = pl.read_parquet(U2_JJ).select("MVT_ID_mvt", pl.col("p_copy").alias("u2_p"), pl.col("expert").alias("u2_expert"))
    return base.join(ev, on="MVT_ID_mvt", how="left", validate="1:1", maintain_order="left").join(u2, on="MVT_ID_mvt", how="left", validate="1:1", maintain_order="left")


def phase_jj_score(ten: bool = False) -> None:
    legs = LEGS10 if ten else LEGS
    f = _janjul_frame(legs)
    y = f[s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
    month = f["month"].to_numpy()
    matched = f["nm_unmatched"].to_numpy() == 0
    raw = np.column_stack([f[k].to_numpy() for k in legs]).mean(1)
    cb_raw, cb_c = f["cb"].to_numpy(), f["c_cb"].to_numpy()
    gated = matched & (np.abs(cb_c - np.clip(cb_raw, 0, None)) > 1e-6)
    six = np.where(matched & ~gated, np.clip(raw, 0, None), cb_c)

    def card(pred):
        e = (y - pred) ** 2
        return {"pooled": float(np.sqrt(e.mean())), "JAN": float(np.sqrt(e[month == 1].mean())), "JUL": float(np.sqrt(e[month == 7].mean())),
                "matched": float(np.sqrt(e[matched].mean()))}

    result: dict[str, Any] = {"legs": list(legs.values()), "blend_reproduction": card(six),
                              "expected": {"pooled": 317.743, "JAN": 337.409, "JUL": 300.959} if ten else {"pooled": 319.137, "JAN": 338.380, "JUL": 302.740, "matched": 204.106},
                              "gated_rows": int(gated.sum())}
    passed = ["T1"] if ten else [k for k, v in json.loads((OUT / "metrics" / "d_score.json").read_text())["arms"].items() if v["GO"]]
    result["arms_passing_D"] = passed
    lg = pl.read_parquet(OUT / "oof" / "jj_lgbm.parquet")
    g = f.join(lg, on="MVT_ID_mvt", how="left", maintain_order="left")
    if any("T2" in k for k in passed):
        g = g.join(pl.read_parquet(OUT / "oof" / "jj_t2clf.parquet"), on="MVT_ID_mvt", how="left", maintain_order="left")
    for c in ("p_long", "mu_long", "p_stamp", "e_nc"):
        if c not in g.columns:
            g = g.with_columns(pl.lit(0.0 if c != "e_nc" else float("nan")).alias(c))
    apply = matched & ~gated
    t2_pop = apply & (g["AIRPORT"].to_numpy() != "LIRF")
    g = g.with_columns(pl.col("p_long").fill_null(0.0), pl.col("mu_long").fill_null(0.0), pl.col("p_stamp").fill_null(0.0), pl.col("e_nc").fill_null(float("nan")))
    lirf_u2 = matched & ~gated & g["u2_p"].is_not_null().to_numpy()
    g = g.with_columns(pl.when(pl.Series(lirf_u2)).then(pl.col("u2_p")).otherwise(None).alias("u2_p"))
    arms = _arms(six, g, apply, t2_pop)
    post = {k: _post(v, g, matched) for k, v in arms.items()}
    result["control_post"] = card(post["control"])
    result["arms"] = {}
    ymat = y[matched]
    for name in passed:
        c = card(post[name])
        d = {k: c[k] - result["control_post"][k] for k in c}
        se_c = (y - post["control"]) ** 2
        dse = (y - post[name]) ** 2 - se_c
        ap = g["AIRPORT"].to_numpy()
        result["arms"][name] = {
            "card": c, "delta": d,
            "matched_dsse_rel": float(dse[matched].sum() / se_c[matched].sum()),
            "pass_pooled_jan_jul": bool(d["pooled"] < 0 and d["JAN"] < 0 and d["JUL"] < 0),
            "by_airport_rel": {str(a): float(dse[matched & (ap == a)].sum() / se_c[matched & (ap == a)].sum()) for a in np.unique(ap[matched])},
            "top20_share_of_net": float(np.sort(dse)[:20].sum() / dse.sum()),
        }
        log(f"JAN_JUL {name}: {c} delta {d}")
    # season-gap diagnostics (labels of Jan/Jul, matched rows)
    sd = g["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    stamp = copy_label(y, sd)
    result["season_gap"] = {
        "long_share": {int(m): float((y > mt.LONG_SEC)[matched & (month == m)].mean()) for m in (1, 7)},
        "stamp_share_nonlirf": {int(m): float(stamp[t2_pop & (month == m)].mean()) for m in (1, 7)},
        "mean_p_long": {int(m): float(g["p_long"].to_numpy()[apply & (month == m)].mean()) for m in (1, 7)},
        "mean_p_stamp_nonlirf": {int(m): float(g["p_stamp"].to_numpy()[t2_pop & (month == m)].mean()) for m in (1, 7)} if any("T2" in k for k in passed) else None,
        "six_leg_long_row_bias": float((y - six)[matched & (y > mt.LONG_SEC)].mean()),
    }
    _ = ymat
    name = "jj_score_10leg.json" if ten else "jj_score.json"
    write_json(OUT / "metrics" / name, result)
    write_json(TRACKED / name, result)
    log("jj_score written")


def phase_t3_score() -> None:
    """T3 screen gate: sgrid_v1 + M1 + M2 vs the stored mf_m1m2_v1 control (D-002 + no airport > +2%)."""

    ctl = pl.read_parquet(MAIN / "artifacts" / "experiments" / "Matched-screen-20260925" / "mf_m1m2_v1" / "oof.parquet")
    cand = pl.read_parquet(OUT / "screen" / "sgrid_v1_m1m2" / "oof.parquet").select("MVT_ID_mvt", pl.col("prediction").alias("cand"))
    j = ctl.join(cand, on="MVT_ID_mvt", how="inner", validate="1:1")
    if j.height != ctl.height or j.height != cand.height:
        raise RuntimeError("screen arms cover different rows")
    day = pl.read_parquet(E08B_CACHE, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"))
    j = j.join(day.select("MVT_ID_mvt", "day"), on="MVT_ID_mvt", how="left")
    y = j[s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
    rep = mt.gate_report(y, j["prediction"].to_numpy(), j["cand"].to_numpy(), fold=j["fold"].to_numpy(), airport=j["AIRPORT"].to_numpy(), day=j["day"].to_numpy())
    rep["gates"]["G1_D002_rel_le_-0.005"] = rep.pop("d002_rel_le_-0.005")
    rep["gates"].pop("G1_abs_dsse_lt_0")
    rep["GO"] = all(rep["gates"].values())
    rep["population"] = "matched-only D blocked OOF (internal screen); control mf_m1m2_v1 (E08b+E34+temporal-v2 + M1 + M2), candidate + sgrid_v1"
    write_json(OUT / "metrics" / "t3_screen.json", rep)
    write_json(TRACKED / "t3_screen.json", rep)
    log(f"T3 screen: dSSE {rep['dsse_rel']*100:+.3f}% p95 {rep['day_bootstrap_p05_p95'][1]:.3e} folds {rep['folds_improved']} worst {rep['worst_airport_rel']*100:+.2f}% GO={rep['GO']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("d_lgbm", "d_t2clf", "d_score", "jj_fit", "jj_t2clf", "jj_score", "jj_score_10leg", "t3_score"))
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    (OUT / "oof").mkdir(parents=True, exist_ok=True)
    {"d_lgbm": lambda: phase_d_lgbm(a.threads), "d_t2clf": lambda: phase_d_t2clf(a.threads), "d_score": phase_d_score,
     "jj_fit": lambda: phase_jj_fit(a.threads), "jj_t2clf": lambda: phase_jj_t2clf(a.threads), "jj_score": phase_jj_score, "jj_score_10leg": lambda: phase_jj_score(ten=True),
     "t3_score": phase_t3_score}[a.phase]()


if __name__ == "__main__":
    main()
