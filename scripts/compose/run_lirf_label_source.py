# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-LIRF label-source campaign: U2 (matched LIRF copy mixture) and U3 (unmatched LIRF gate delay).

Pre-registered in progress-review/lirf-label-source-20260925/preregistration.json before any fit.

Phases (each resumable, all fits nested inside the blocked D folds {2,8},{3,9},{4,10},{5,6}):

  u2_prepare  harness base frame (E08b+E34+R2 temporal) for all matched D rows, the R2-proxy
              parent OOF, hull windows; LIRF p-classifier inputs from raw DEP/ARR (no labels
              as features).
  u2_fit      per fold: nested copy-rate encodings, CatBoost copy classifier, E_nc (all copies
              dropped) and E_nc_large (copies with sd >= 1800 s dropped).
  u2_score    arms M1/M2/M3 vs the parent after projection on both; gates; selection.
  u2_confirm  single frozen JAN_JUL_2025 look (refused unless a U2 arm passed every gate).
  u3_prepare  LIRF D departures with U6-family features and same-stand inbound features;
              A3b-nodest deployed D control on the 14,066 unmatched rows.
  u3_fit      per fold: gate-delay model G = BLOCK - SCHED on all LIRF training departures.
  u3_score    T = max(sd - G_hat, 0) blended with A3b at lambda = 0.25; gates.
  u3_confirm  single frozen JAN_JUL_2025 look (refused unless U3 passed every gate).

January/July labels are read only by the two confirm phases. Nothing is uploaded and no
ranking file is written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _main_repo() -> Path:
    env = os.environ.get("PRC_MAIN_REPO")
    if env:
        return Path(env)
    sibling = ROOT.parent.parent / "prc-challenge-2026"
    if (sibling / "artifacts" / "cache").exists():
        return sibling
    return ROOT


MAIN = _main_repo()
OUT = MAIN / "artifacts" / "experiments" / "LIRF-label-source-20260925"
TRACKED = ROOT / "progress-review" / "lirf-label-source-20260925"
E08B_CACHE = MAIN / "artifacts" / "cache" / "e08b_e34_months_1_10.parquet"
E08B_META = MAIN / "artifacts" / "cache" / "e08b_e34_months_1_10.json"
TEMPORAL_CACHE = MAIN / "artifacts" / "cache" / "temporal_context_v2.parquet"
PARENT_OOF = MAIN / "artifacts" / "experiments" / "Matched-screen-20260925" / "base_r2proxy" / "oof.parquet"
UNC = MAIN / "artifacts" / "experiments" / "Unmatched-noncopy-20260925"
CONFIRM_A3B = UNC / "confirm" / "confirm_A3b.parquet"

from prc_challenge.lirf_label_source import (  # noqa: E402
    RATE_N0,
    U2_GATE_SEC,
    U3_LAMBDA,
    U3_LAMBDA_DIAGNOSTIC,
    blend,
    choose_arm,
    copy_feasible,
    copy_label,
    copy_mixture,
    expert_mask,
    gate_delay_target,
    nested_rate_encoding,
    project,
    sd_bucket,
    serve_gate_delay,
    u2_gates,
    u2_report,
    u3_gates,
    u3_report,
    window_from_leads,
    arr_daily_copy_rate,
    stand_inbound_features,
    STAND_NUMS,
)
from prc_challenge.unmatched_recovery.schema import BLOCKS, D_MONTHS, LOCKED_MONTHS, PRIMARY_MONTHS  # noqa: E402

THREADS = 2
FORBIDDEN = {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt"}
HARNESS_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 800,
    "depth": 8,
    "learning_rate": 0.08,
    "l2_leaf_reg": 3.0,
    "border_count": 128,
    "random_seed": 42,
    "thread_count": THREADS,
    "verbose": False,
    "allow_writing_files": False,
}
CLASSIFIER_PARAMS = {
    "loss_function": "Logloss",
    "iterations": 400,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "thread_count": THREADS,
    "verbose": False,
    "allow_writing_files": False,
}
G_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 400,
    "depth": 6,
    "learning_rate": 0.08,
    "l2_leaf_reg": 16.0,
    "random_strength": 1.0,
    "nan_mode": "Min",
    "random_seed": 42,
    "thread_count": THREADS,
    "verbose": False,
    "allow_writing_files": False,
}
ENC_KEYS = ("flight_prefix", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt")
P_NUMS = (
    *(f"enc_{k}" for k in ENC_KEYS),
    *(f"logn_{k}" for k in ENC_KEYS),
    "arr_day_copy_rate",
    "arr_day_n",
    "sched_second_zero",
    "mvt_second_zero",
    "iobt_second_zero",
    "eobt_second_zero",
    "takeoff_minus_sched_sec",
    "sched_minus_iobt_sec",
    "sched_minus_eobt_sec",
    "sched_hour",
    "local_dow",
)
P_CATS = ("sd_bucket",)
U6_CATS = ("AIRPORT", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "flight_prefix", "ADES_mvt", "FLIGHT_RULE_mvt", "skyc1")
U6_NUMS = (
    "local_hour", "local_dow", "local_month", "takeoff_minus_sched_sec", "takeoff_minus_sched_days", "takeoff_minus_sched_mod_sec",
    "type_missing", "nm_unmatched", "tmpc", "dwpc", "sknt", "gust", "vsby", "p01i", "metar_age_sec", "metar_has_wx", "metar_frozen",
    "metar_fog", "metar_ts", "metar_precip", "metar_low_vis", "n_delayed_dep_ctr_900s", "n_delayed_dep_ctr_1800s",
    "n_delayed_dep_ctr_3600s", "ades_missing", "runway_missing", "stand_missing",
)
GATE_COLS = ("MVT_ID_mvt", "AIRPORT", "nm_unmatched", "type_missing", "takeoff_minus_sched_days", "takeoff_minus_sched_sec")
CAT_FILL = "__UNK__"


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _state() -> dict[str, Any]:
    path = OUT / "state.json"
    return json.loads(path.read_text()) if path.exists() else {"phases": {}}


def _save_state(state: dict[str, Any]) -> None:
    _write_json(OUT / "state.json", state)


def _fold_name(month: int) -> str:
    for holdout in BLOCKS:
        if int(month) in holdout:
            return f"{holdout[0]}_{holdout[1]}"
    raise ValueError(f"month {month} outside blocked D")


def _assert_months(months: list[int], allowed: tuple[int, ...], context: str) -> None:
    bad = set(int(m) for m in months) - set(allowed)
    if bad or set(int(m) for m in months) & set(LOCKED_MONTHS):
        raise RuntimeError(f"{context}: months {sorted(bad)} outside {allowed}")


def _checksum(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(np.round(np.asarray(values, dtype=np.float64), 6)).tobytes()).hexdigest()


def _pool(frame: pl.DataFrame, cats: list[str], nums: list[str], label: np.ndarray | None = None):
    from catboost import Pool

    bad = FORBIDDEN.intersection(cats + nums)
    if bad:
        raise RuntimeError(f"forbidden features {sorted(bad)}")
    x = frame.select(
        [pl.col(c).cast(pl.String).fill_null(CAT_FILL).alias(c) for c in cats]
        + [pl.col(c).cast(pl.Float32).alias(c) for c in nums]
    ).to_pandas()
    return Pool(x, label=label, cat_features=list(range(len(cats))))


def _raw_dir() -> Path:
    from prc_challenge.unmatched_recovery.block_lineage import resolve_raw_dir

    return resolve_raw_dir(MAIN)


def _raw_lirf(months: tuple[int, ...], phase: str) -> pl.DataFrame:
    """LIRF rows of one phase from the raw files, month-filtered. DEP BLOCK/TAXITIME are dropped."""

    from prc_challenge.unmatched_recovery.block_lineage import load_movements

    frame = load_movements(_raw_dir(), months, phase).filter(pl.col("AIRPORT") == "LIRF")
    if phase == "DEP":
        frame = frame.drop([c for c in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt") if c in frame.columns])
    else:
        frame = frame.drop([c for c in ("AOBT_3_flt", "ARVT_3_flt", "LOBT_flt") if c in frame.columns])
    return frame


def lirf_raw_features(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Label-free LIRF DEP attributes for U2/U3: prefix, ADES, rule, seconds flags, daily ARR copy rate, stand inbound."""

    from prc_challenge.features import add_flight_prefix

    frame = add_flight_prefix(dep)
    frame = frame.with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),
        (pl.col("SCHED_TIME_UTC_mvt").dt.second() == 0).cast(pl.Int8).alias("sched_second_zero"),
        (pl.col("MVT_TIME_UTC_mvt").dt.second() == 0).cast(pl.Int8).alias("mvt_second_zero"),
        pl.when(pl.col("IOBT_flt").is_null()).then(None).otherwise((pl.col("IOBT_flt").dt.second() == 0).cast(pl.Int8)).alias("iobt_second_zero"),
        pl.when(pl.col("EOBT_1_flt").is_null()).then(None).otherwise((pl.col("EOBT_1_flt").dt.second() == 0).cast(pl.Int8)).alias("eobt_second_zero"),
        pl.col("SCHED_TIME_UTC_mvt").dt.hour().cast(pl.Int16).alias("sched_hour"),
        pl.col("ADES_mvt").is_null().cast(pl.Int8).alias("ades_missing"),
    )
    parts = []
    for month in sorted(frame["month"].unique().to_list()):
        d = frame.filter(pl.col("month") == month)
        a = arr.filter(pl.col("month") == month)
        daily = arr_daily_copy_rate(a)
        d = d.join(daily, on=["month", "day"], how="left")
        d = d.join(stand_inbound_features(d, a), on="MVT_ID_mvt", how="left")
        parts.append(d)
    out = pl.concat(parts, how="vertical_relaxed")
    if out.height != frame.height:
        raise RuntimeError("LIRF raw feature joins changed the row count")
    keep = ["MVT_ID_mvt", "flight_prefix", "ADES_mvt", "FLIGHT_RULE_mvt", "ades_missing", "sched_second_zero", "mvt_second_zero", "iobt_second_zero", "eobt_second_zero", "sched_hour", "arr_day_n", "arr_day_copy_rate", *STAND_NUMS]
    return out.select(keep)


def harness_frame(months: tuple[int, ...]) -> tuple[pl.DataFrame, list[str], list[str]]:
    """E08b + E34 cache joined with R2 temporal-v2 context (the screen harness base)."""

    meta = json.loads(E08B_META.read_text())
    cats, nums = list(meta["cats"]), list(meta["nums"])
    base = pl.read_parquet(E08B_CACHE).filter(pl.col("month").is_in(list(months)))
    temporal = pl.read_parquet(TEMPORAL_CACHE)
    tnums = [c for c in temporal.columns if c != "MVT_ID_mvt"]
    base = base.join(temporal, on="MVT_ID_mvt", how="left")
    return base, cats, nums + tnums


def _e18_expr() -> pl.Expr:
    return (pl.col("AIRPORT") == "LIRF") & (pl.col("takeoff_minus_sched_days") >= 1)


def _u2_lirf_frame(base: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    lirf = base.filter((pl.col("AIRPORT") == "LIRF") & (pl.col("nm_unmatched") == 0) & ~_e18_expr())
    lirf = lirf.join(raw.drop([c for c in ("FLIGHT_RULE_mvt",) if c in raw.columns]), on="MVT_ID_mvt", how="left")
    if lirf["arr_day_n"].null_count() == lirf.height:
        raise RuntimeError("ARR daily join failed")
    lirf = lirf.with_columns(
        (pl.col("takeoff_minus_eobt_sec") - pl.col("takeoff_minus_sched_sec")).cast(pl.Float64).alias("sched_minus_eobt_sec"),
    )
    lirf = lirf.with_columns(pl.Series("sd_bucket", sd_bucket(lirf["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy())))
    return lirf


# ---------------------------------------------------------------------------
# U2
# ---------------------------------------------------------------------------


def phase_u2_prepare(state: dict[str, Any]) -> None:
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    base, cats, nums = harness_frame(D_MONTHS)
    _assert_months(base["month"].unique().to_list(), D_MONTHS, "u2 prepare")
    matched = base.filter((pl.col("nm_unmatched") == 0) & ~_e18_expr())
    parent = pl.read_parquet(PARENT_OOF)
    allm = matched.select("MVT_ID_mvt", "AIRPORT", "month", "TAXITIME_SEC_mvt", "takeoff_minus_iobt_sec", "takeoff_minus_eobt_sec").join(
        parent.select("MVT_ID_mvt", pl.col("prediction").alias("parent")), on="MVT_ID_mvt", how="left"
    )
    if allm.height != parent.height or allm["parent"].null_count():
        raise RuntimeError(f"parent OOF does not cover the matched D population ({allm.height} vs {parent.height})")
    lo, hi = window_from_leads(allm["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), allm["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy())
    allm = allm.with_columns(pl.Series("win_lo", lo), pl.Series("win_hi", hi), pl.col("month").map_elements(_fold_name, return_dtype=pl.Utf8).alias("fold"))
    y = allm["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    viol = int(np.sum(np.isfinite(lo) & ((y < lo - 0.5) | (y > hi + 0.5))))
    allm.select("MVT_ID_mvt", "AIRPORT", "month", "fold", "TAXITIME_SEC_mvt", "parent", "win_lo", "win_hi").write_parquet(OUT / "frames" / "u2_all_matched_D.parquet")
    _log("raw LIRF DEP/ARR for D")
    dep = _raw_lirf(D_MONTHS, "DEP")
    arr = _raw_lirf(D_MONTHS, "ARR")
    raw = lirf_raw_features(dep, arr)
    lirf = _u2_lirf_frame(base, raw)
    # Parity of the numeric window with matching_window.window_bounds on the raw timestamps.
    from prc_challenge.matching_window import window_bounds

    wb = window_bounds(dep.select("MVT_ID_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "IOBT_flt", "EOBT_1_flt"))
    chk = lirf.select("MVT_ID_mvt").join(allm.select("MVT_ID_mvt", "win_lo", "win_hi"), on="MVT_ID_mvt", how="left").join(
        wb.select("MVT_ID_mvt", pl.col("win_lo").alias("ref_lo"), pl.col("win_hi").alias("ref_hi")), on="MVT_ID_mvt", how="left"
    )
    parity = float(np.nanmax(np.abs(np.concatenate([(chk["win_lo"] - chk["ref_lo"]).to_numpy(), (chk["win_hi"] - chk["ref_hi"]).to_numpy()]))))
    lirf.write_parquet(OUT / "frames" / "u2_lirf_matched_D.parquet")
    _write_json(OUT / "frames" / "u2_features.json", {"expert_cats": cats, "expert_nums": nums, "p_cats": list(P_CATS), "p_nums": list(P_NUMS)})
    sd = lirf["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    feas = copy_feasible(sd, lirf["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), lirf["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy())
    state["phases"]["u2_prepare"] = {
        "status": "complete",
        "n_all_matched": allm.height,
        "n_lirf": lirf.height,
        "window_label_violations_all_matched": viol,
        "window_parity_vs_matching_window": parity,
        "lirf_copy_infeasible_rows": int((~feas).sum()),
        "arr_day_null": int(lirf["arr_day_copy_rate"].null_count()),
        "stand_prev_null": int(lirf["takeoff_minus_stand_prev_arr_sec"].null_count()),
    }
    _save_state(state)
    _log(f"u2 prepare: {state['phases']['u2_prepare']}")


def _encode(frame: pl.DataFrame, copy: np.ndarray, fit_mask: np.ndarray, apply_mask: np.ndarray) -> pl.DataFrame:
    """Nested OOF copy-rate encodings; ``copy`` is read only where ``fit_mask``."""

    months = frame["month"].to_numpy()
    label = np.where(fit_mask, copy.astype(np.float64), np.nan)
    cols = []
    for key in ENC_KEYS:
        keys = frame[key].cast(pl.String).fill_null(CAT_FILL).to_numpy().astype(object)
        rate, count = nested_rate_encoding(keys, label, months, fit_mask, apply_mask, n0=RATE_N0)
        cols += [pl.Series(f"enc_{key}", rate), pl.Series(f"logn_{key}", np.log1p(count))]
    return frame.with_columns(cols)


def _fit_u2_models(train: pl.DataFrame, copy_train: np.ndarray, feas_train: np.ndarray, cats: list[str], nums: list[str], tag: str, modes: dict[str, float | None]):
    from catboost import CatBoostClassifier, CatBoostRegressor

    sd = train["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    y = train["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    clf = CatBoostClassifier(**CLASSIFIER_PARAMS)
    ctrain = train.filter(pl.Series(feas_train))
    clf.fit(_pool(ctrain, list(P_CATS), list(P_NUMS), copy_train[feas_train].astype(int)))
    clf.save_model(str(OUT / "models" / f"u2_pcopy_{tag}.cbm"))
    experts = {}
    for name, gate in modes.items():
        mask = expert_mask(copy_train, sd, gate_sec=gate)
        model = CatBoostRegressor(**HARNESS_PARAMS)
        model.fit(_pool(train.filter(pl.Series(mask)), cats, nums, y[mask]))
        model.save_model(str(OUT / "models" / f"u2_{name}_{tag}.cbm"))
        experts[name] = (model, int(mask.sum()))
    return clf, experts


U2_EXPERT_MODES = {"enc_all": None, "enc_large": U2_GATE_SEC}


def phase_u2_fit(state: dict[str, Any]) -> None:
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    (OUT / "oof").mkdir(parents=True, exist_ok=True)
    lirf = pl.read_parquet(OUT / "frames" / "u2_lirf_matched_D.parquet").with_columns(pl.int_range(0, pl.len()).alias("_row"))
    feats = json.loads((OUT / "frames" / "u2_features.json").read_text())
    cats, nums = feats["expert_cats"], feats["expert_nums"]
    done = state["phases"].setdefault("u2_fit", {})
    for holdout in BLOCKS:
        name = f"{holdout[0]}_{holdout[1]}"
        part_path = OUT / "oof" / f"u2_oof_{name}.parquet"
        if part_path.exists():
            _log(f"u2 fold {name}: cached")
            continue
        started = time.time()
        months = lirf["month"].to_numpy()
        fit_mask = ~np.isin(months, list(holdout))
        hold_mask = np.isin(months, list(holdout))
        y = lirf["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
        sd = lirf["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
        # Copy label only for training rows; held-out labels are never read by a fit.
        copy = np.where(fit_mask, copy_label(np.where(fit_mask, y, np.nan), sd), False)
        enc = _encode(lirf, copy, fit_mask, hold_mask)
        train = enc.filter(pl.Series(fit_mask))
        hold = enc.filter(pl.Series(hold_mask))
        _assert_months(train["month"].unique().to_list(), tuple(m for m in D_MONTHS if m not in holdout), f"u2 fold {name}")
        feas = copy_feasible(sd, lirf["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), lirf["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy())
        clf, experts = _fit_u2_models(train, copy[fit_mask], feas[fit_mask], cats, nums, name, U2_EXPERT_MODES)
        p = clf.predict_proba(_pool(hold, list(P_CATS), list(P_NUMS)))[:, 1]
        cols = {"p_copy": p}
        for ename, (model, _) in experts.items():
            cols[ename] = np.asarray(model.predict(_pool(hold, cats, nums)), dtype=np.float64)
        part = hold.select("MVT_ID_mvt", "month", "_row").with_columns([pl.Series(k, v) for k, v in cols.items()])
        part.write_parquet(part_path)
        done[name] = {
            "n_train": int(fit_mask.sum()),
            "n_hold": int(hold_mask.sum()),
            "n_classifier_train": int(feas[fit_mask].sum()),
            "train_copy_rate": float(copy[fit_mask].mean()),
            "expert_train_rows": {k: n for k, (_, n) in experts.items()},
            "seconds": time.time() - started,
        }
        _save_state(state)
        _log(f"u2 fold {name}: {done[name]}")
    parts = [pl.read_parquet(OUT / "oof" / f"u2_oof_{h[0]}_{h[1]}.parquet") for h in BLOCKS]
    oof = pl.concat(parts).sort("_row").drop("_row")
    oof.write_parquet(OUT / "oof" / "u2_oof_D.parquet")
    done["status"] = "complete"
    done["checksums"] = {c: _checksum(oof[c].to_numpy()) for c in ("p_copy", *U2_EXPERT_MODES)}
    _save_state(state)


def _u2_arms(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    p = frame["p_copy"].to_numpy()
    sd = frame["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    feas = frame["feasible"].to_numpy()
    e_all = frame["enc_all"].to_numpy()
    e_large = frame["enc_large"].to_numpy()
    parent = frame["parent"].to_numpy()
    return {
        "M1": copy_mixture(p, sd, e_all, feas),
        "M2": copy_mixture(p, sd, e_large, feas, gate_sec=U2_GATE_SEC),
        "M3": copy_mixture(p, sd, 0.5 * e_all + 0.5 * parent, feas),
    }


def _u2_diagnostics(frame: pl.DataFrame, copy: np.ndarray) -> dict[str, np.ndarray]:
    p = frame["p_copy"].to_numpy()
    sd = frame["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    feas = frame["feasible"].to_numpy()
    e_all = frame["enc_all"].to_numpy()
    parent = frame["parent"].to_numpy()
    return {
        "diag_Enc_alone": e_all,
        "diag_parent_mixture": copy_mixture(p, sd, parent, feas),
        "diag_M1_no_window_constraint": copy_mixture(p, sd, e_all, np.isfinite(sd)),
        "diag_oracle_p_LABEL_USING": copy_mixture(copy.astype(np.float64), sd, e_all, feas),
    }


def _u2_scoring_frame() -> tuple[pl.DataFrame, pl.DataFrame]:
    lirf = pl.read_parquet(OUT / "frames" / "u2_lirf_matched_D.parquet")
    oof = pl.read_parquet(OUT / "oof" / "u2_oof_D.parquet")
    allm = pl.read_parquet(OUT / "frames" / "u2_all_matched_D.parquet")
    frame = lirf.select("MVT_ID_mvt", "month", "TAXITIME_SEC_mvt", "takeoff_minus_sched_sec", "takeoff_minus_iobt_sec", "takeoff_minus_eobt_sec").join(
        oof.drop("month"), on="MVT_ID_mvt", how="left", maintain_order="left"
    ).join(allm.select("MVT_ID_mvt", "parent", "win_lo", "win_hi", "fold"), on="MVT_ID_mvt", how="left", maintain_order="left")
    for col in ("p_copy", "enc_all", "enc_large", "parent"):
        if frame[col].null_count():
            raise RuntimeError(f"{col} missing on {frame[col].null_count()} LIRF rows")
    sd = frame["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    feas = copy_feasible(sd, frame["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), frame["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy())
    return frame.with_columns(pl.Series("feasible", feas)), allm


def phase_u2_score(state: dict[str, Any]) -> None:
    from sklearn.metrics import log_loss, roc_auc_score

    frame, allm = _u2_scoring_frame()
    y = frame["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    sd = frame["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    lo = frame["win_lo"].to_numpy()
    hi = frame["win_hi"].to_numpy()
    fold = frame["fold"].to_numpy()
    copy = copy_label(y, sd)  # scoring slices only
    parent = project(frame["parent"].to_numpy(), lo, hi)
    ya = allm["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    pa = project(allm["parent"].to_numpy(), allm["win_lo"].to_numpy(), allm["win_hi"].to_numpy())
    all_sse = float(np.sum((ya - pa) ** 2))
    arms = {k: project(v, lo, hi) for k, v in _u2_arms(frame).items()}
    diags = {k: project(v, lo, hi) for k, v in _u2_diagnostics(frame, copy).items()}
    reports, gated, rows = {}, {}, []
    for name, pred in {**arms, **diags}.items():
        rep = u2_report(y, pred, parent, fold=fold, is_copy=copy, sd=sd, all_matched_parent_sse=all_sse)
        reports[name] = rep
        g = u2_gates(rep)
        if name in arms:
            gated[name] = g
        row = {
            "arm": name,
            "lirf_sse_ratio": rep["lirf"]["sse_ratio"],
            "lirf_rmse": rep["lirf"]["rmse"],
            "parent_lirf_rmse": rep["lirf"]["base_rmse"],
            "all_matched_delta_ratio": rep["all_matched_delta_ratio"],
            "leave_top5_ratio": rep["leave_top_k"]["sse_ratio"],
            "top5_share_of_gain": rep["leave_top_k"]["top_share_of_gain"],
            "noncopy_rmse": rep["noncopy"]["rmse"],
            "parent_noncopy_rmse": rep["noncopy"]["base_rmse"],
            "noncopy_rmse_ratio": rep["noncopy"]["rmse"] / rep["noncopy"]["base_rmse"],
            "copy_rmse": rep["copy"]["rmse"],
            "parent_copy_rmse": rep["copy"]["base_rmse"],
            **{f"fold_{k}_ratio": v["sse_ratio"] for k, v in rep["folds"].items()},
            "go": g["go"] if name in arms else None,
            "failed": ";".join(g["failed"]) if name in arms else "diagnostic",
        }
        rows.append(row)
    chosen = choose_arm(gated, {k: reports[k]["lirf"]["sse_ratio"] for k in arms})
    feas = frame["feasible"].to_numpy()
    p = frame["p_copy"].to_numpy()
    calib = {
        "log_loss_feasible": float(log_loss(copy[feas], np.clip(p[feas], 1e-6, 1 - 1e-6))),
        "auc_feasible": float(roc_auc_score(copy[feas], p[feas])),
        "by_sd_bucket": {},
    }
    buckets = sd_bucket(sd)
    for b in sorted(set(buckets.tolist())):
        m = (buckets == b) & feas
        if m.sum() and 0 < copy[m].mean() < 1:
            calib["by_sd_bucket"][b] = {"n": int(m.sum()), "copy_rate": float(copy[m].mean()), "mean_p": float(p[m].mean()), "auc": float(roc_auc_score(copy[m], p[m]))}
    table = pl.DataFrame(rows)
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    TRACKED.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT / "metrics" / "u2_arm_table.csv")
    table.write_csv(TRACKED / "u2_arm_table.csv")
    payload = {
        "population": "matched LIRF D departures excluding E18, n=%d" % frame.height,
        "parent": str(PARENT_OOF),
        "projection": "hull window (IOBT/EOBT_1 +/- 3606 s) on candidate and parent",
        "all_matched_parent_sse_projected": all_sse,
        "all_matched_n": allm.height,
        "gates": gated,
        "chosen": chosen,
        "decision": "GO" if chosen else "PARK",
        "p_copy_calibration": calib,
        "reports": {k: {kk: vv for kk, vv in v.items() if kk != "leave_top_k"} | {"leave_top_k": {kk: vv for kk, vv in v["leave_top_k"].items() if kk != "top_rows"}} for k, v in reports.items()},
    }
    _write_json(OUT / "metrics" / "u2_score.json", payload)
    _write_json(TRACKED / "u2_score.json", payload)
    pl.DataFrame({"MVT_ID_mvt": frame["MVT_ID_mvt"], "parent_proj": parent, **arms}).write_parquet(OUT / "oof" / "u2_arms_D.parquet")
    state["phases"]["u2_score"] = {"status": "complete", "chosen": chosen, "decision": payload["decision"]}
    _save_state(state)
    print(table.select("arm", "lirf_sse_ratio", "all_matched_delta_ratio", "leave_top5_ratio", "noncopy_rmse_ratio", "go", "failed"))
    _log(f"u2 decision {payload['decision']} chosen {chosen}")


def phase_u2_confirm(state: dict[str, Any]) -> None:
    score = state["phases"].get("u2_score") or {}
    arm = score.get("chosen")
    if score.get("decision") != "GO" or not arm:
        raise RuntimeError("U2 confirmation refused: no arm passed every D gate")
    if (OUT / "confirm" / "u2_confirmation.json").exists():
        raise RuntimeError("U2 confirmation already run once; refusing a second look")
    (OUT / "confirm").mkdir(parents=True, exist_ok=True)
    feats = json.loads((OUT / "frames" / "u2_features.json").read_text())
    cats, nums = feats["expert_cats"], feats["expert_nums"]
    months_all = tuple(sorted(set(D_MONTHS) | set(PRIMARY_MONTHS)))
    base, _, _ = harness_frame(months_all)
    dep = _raw_lirf(months_all, "DEP")
    arr = _raw_lirf(months_all, "ARR")
    raw = lirf_raw_features(dep, arr)
    lirf = _u2_lirf_frame(base, raw)
    months = lirf["month"].to_numpy()
    fit_mask = np.isin(months, list(D_MONTHS))
    apply_mask = np.isin(months, list(PRIMARY_MONTHS))
    y = lirf["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    sd = lirf["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    copy = np.where(fit_mask, copy_label(np.where(fit_mask, y, np.nan), sd), False)
    enc = _encode(lirf, copy, fit_mask, apply_mask)
    train = enc.filter(pl.Series(fit_mask))
    hold = enc.filter(pl.Series(apply_mask))
    _assert_months(train["month"].unique().to_list(), D_MONTHS, "u2 confirm fit")
    feas = copy_feasible(sd, lirf["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), lirf["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy())
    mode = {"M1": "enc_all", "M2": "enc_large", "M3": "enc_all"}[arm]
    clf, experts = _fit_u2_models(train, copy[fit_mask], feas[fit_mask], cats, nums, "confirm_D", {mode: U2_EXPERT_MODES[mode]})
    p = clf.predict_proba(_pool(hold, list(P_CATS), list(P_NUMS)))[:, 1]
    e = np.asarray(experts[mode][0].predict(_pool(hold, cats, nums)), dtype=np.float64)
    sd_h = hold["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    feas_h = feas[apply_mask]
    # Label-free Jan/Jul predictions are fixed here, before any Jan/Jul label is read.
    conf = pl.read_parquet(CONFIRM_A3B)
    s4 = hold.select("MVT_ID_mvt").join(conf.select("MVT_ID_mvt", "prediction"), on="MVT_ID_mvt", how="left", maintain_order="left")["prediction"].to_numpy()
    if np.isnan(s4).any():
        raise RuntimeError("S4 prediction missing for Jan/Jul LIRF matched rows")
    if arm == "M1":
        yhat = copy_mixture(p, sd_h, e, feas_h)
    elif arm == "M2":
        yhat = copy_mixture(p, sd_h, e, feas_h, gate_sec=U2_GATE_SEC)
    else:
        yhat = copy_mixture(p, sd_h, 0.5 * e + 0.5 * s4, feas_h)
    cand_rows = hold.select("MVT_ID_mvt").with_columns(pl.Series("u2_pred", yhat), pl.Series("p_copy", p), pl.Series("expert", e))
    cand_rows.write_parquet(OUT / "confirm" / "u2_janjul_label_free_predictions.parquet")
    _log(f"u2 confirm: {cand_rows.height} Jan/Jul LIRF matched rows predicted (label-free); scoring once")
    # --- single look ---
    windows = base.filter(pl.col("month").is_in(list(PRIMARY_MONTHS))).select("MVT_ID_mvt", "takeoff_minus_iobt_sec", "takeoff_minus_eobt_sec")
    frame = conf.join(windows, on="MVT_ID_mvt", how="left").join(cand_rows.select("MVT_ID_mvt", "u2_pred"), on="MVT_ID_mvt", how="left")
    if frame.height != 344_419:
        raise RuntimeError(f"Jan/Jul population {frame.height}")
    lo, hi = window_from_leads(frame["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), frame["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy())
    matched = frame["nm_unmatched"].to_numpy() == 0
    lo = np.where(matched, lo, np.nan)
    hi = np.where(matched, hi, np.nan)
    ctrl = project(frame["prediction"].to_numpy(), lo, hi)
    cand_raw = np.where(frame["u2_pred"].is_not_null().to_numpy(), frame["u2_pred"].fill_null(0.0).to_numpy(), frame["prediction"].to_numpy())
    cand = project(cand_raw, lo, hi)
    payload = _janjul_scores(frame, ctrl, cand, focus=(frame["AIRPORT"].to_numpy() == "LIRF") & matched, focus_name="lirf_matched")
    payload.update({"arm": arm, "expert_mode": mode, "fit_months": list(D_MONTHS), "n_replaced": int(frame["u2_pred"].is_not_null().sum()), "p_copy_mean": float(p.mean()), "p_copy_share_gt_0_5": float((p > 0.5).mean())})
    frame.select("MVT_ID_mvt", "AIRPORT", "month", "nm_unmatched", "TAXITIME_SEC_mvt").with_columns(pl.Series("control", ctrl), pl.Series("prediction", cand)).write_parquet(OUT / "confirm" / "u2_confirm.parquet")
    _write_json(OUT / "confirm" / "u2_confirmation.json", payload)
    _write_json(TRACKED / "u2_confirmation.json", payload)
    state["phases"]["u2_confirm"] = {"status": "complete", "arm": arm, "full_rmse": payload["candidate"]["full_rmse"], "control_full_rmse": payload["control"]["full_rmse"]}
    _save_state(state)
    _log(f"u2 confirmation {arm}: full {payload['candidate']['full_rmse']:.3f} vs control {payload['control']['full_rmse']:.3f}")


def _janjul_scores(frame: pl.DataFrame, ctrl: np.ndarray, cand: np.ndarray, *, focus: np.ndarray, focus_name: str) -> dict[str, Any]:
    from prc_challenge.unmatched_recovery.noncopy_expert import improvement, leave_top_k_ratio

    y = frame["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    month = frame["month"].to_numpy()
    unmatched = frame["nm_unmatched"].to_numpy() == 1

    def block(pred: np.ndarray) -> dict[str, float]:
        se = (y - pred) ** 2
        out = {"full_rmse": float(np.sqrt(se.mean())), "full_sse": float(se.sum())}
        for m, name in ((1, "JAN"), (7, "JUL")):
            out[f"{name}_rmse"] = float(np.sqrt(se[month == m].mean()))
        out["matched_rmse"] = float(np.sqrt(se[~unmatched].mean()))
        out["unmatched_rmse"] = float(np.sqrt(se[unmatched].mean()))
        out[f"{focus_name}_rmse"] = float(np.sqrt(se[focus].mean()))
        return out

    gain = improvement(y, cand, ctrl)
    order = np.argsort(-gain)
    total = float(gain.sum())
    top = leave_top_k_ratio(y[focus], cand[focus], ctrl[focus], 5)
    return {
        "control": block(ctrl),
        "candidate": block(cand),
        "rows_changed": int(np.sum(np.abs(cand - ctrl) > 1e-9)),
        "focus": focus_name,
        "focus_n": int(focus.sum()),
        "focus_sse_ratio": float(np.sum((y[focus] - cand[focus]) ** 2) / np.sum((y[focus] - ctrl[focus]) ** 2)),
        "focus_leave_top5_ratio": top["sse_ratio"],
        "top1_share_of_gain": float(gain[order[0]] / total) if total > 0 else None,
        "top5_share_of_gain": float(gain[order[:5]].sum() / total) if total > 0 else None,
        "top20_share_of_gain": float(gain[order[:20]].sum() / total) if total > 0 else None,
        "delta_sse": float(-total),
        "jan_delta_sse": float(-gain[month == 1].sum()),
        "jul_delta_sse": float(-gain[month == 7].sum()),
    }


# ---------------------------------------------------------------------------
# U3
# ---------------------------------------------------------------------------


def _u3_frame(months: tuple[int, ...]) -> pl.DataFrame:
    """All LIRF departures in ``months`` with U6-family features and same-stand inbound features."""

    base = pl.read_parquet(E08B_CACHE).filter(pl.col("month").is_in(list(months)) & (pl.col("AIRPORT") == "LIRF"))
    dep = _raw_lirf(months, "DEP")
    arr = _raw_lirf(months, "ARR")
    raw = lirf_raw_features(dep, arr)
    frame = base.join(raw, on="MVT_ID_mvt", how="left")
    if frame["flight_prefix"].null_count():
        raise RuntimeError("raw LIRF join missed rows")
    return frame.with_columns(
        pl.col("RUNWAY_mvt").is_null().cast(pl.Int8).alias("runway_missing"),
        pl.col("STAND_mvt").is_null().cast(pl.Int8).alias("stand_missing"),
    )


def _a3b_control() -> pl.DataFrame:
    """A3b-nodest (month-isolated P(copy)) D OOF on the 14,066 unmatched rows, raw and deployed (E17/E18 + clip)."""

    from prc_challenge.unmatched_recovery.noncopy_expert import GATE_SEC, mixture_prediction
    from prc_challenge.unmatched_recovery.schema import compose_experimental_unmatched

    d_u = pl.read_parquet(UNC / "frames" / "d_unmatched_u6_features.parquet").sort("_row")
    ids = d_u.select("MVT_ID_mvt")
    p = ids.join(pl.read_parquet(UNC / "oof" / "p_copy_oof_nodest_iso.parquet"), on="MVT_ID_mvt", how="left", maintain_order="left")["p_schedule_copy"].to_numpy()
    e = ids.join(pl.read_parquet(UNC / "oof" / "expert_noncopy_large_D.parquet").select("MVT_ID_mvt", "prediction"), on="MVT_ID_mvt", how="left", maintain_order="left")["prediction"].to_numpy()
    delay = d_u["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    raw = mixture_prediction(p, delay, e, gate_sec=GATE_SEC)
    gate = d_u.select(list(GATE_COLS))
    deployed = compose_experimental_unmatched(gate.with_columns(pl.Series("prediction", raw)))["prediction"].to_numpy()
    return d_u.select(*GATE_COLS, "month", "TAXITIME_SEC_mvt").with_columns(pl.Series("a3b_raw", raw), pl.Series("a3b_deployed", deployed))


def phase_u3_prepare(state: dict[str, Any]) -> None:
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    frame = _u3_frame(D_MONTHS)
    _assert_months(frame["month"].unique().to_list(), D_MONTHS, "u3 prepare")
    frame.write_parquet(OUT / "frames" / "u3_lirf_all_D.parquet")
    ctrl = _a3b_control()
    ref = json.loads((UNC / "metrics" / "score_deployed.json").read_text())["headline"]["A3b|nodest_iso"]["rmse"]
    y = ctrl["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    rmse = float(np.sqrt(np.mean((y - ctrl["a3b_deployed"].to_numpy()) ** 2)))
    if abs(rmse - ref) > 1e-6:
        raise RuntimeError(f"A3b-nodest control does not reproduce: {rmse} vs {ref}")
    ctrl.write_parquet(OUT / "frames" / "u3_unmatched_control_D.parquet")
    lirf_u = frame.filter(pl.col("nm_unmatched") == 1)
    state["phases"]["u3_prepare"] = {
        "status": "complete",
        "n_lirf_all": frame.height,
        "n_lirf_unmatched": lirf_u.height,
        "control_rmse": rmse,
        "control_reference_rmse": ref,
        "stand_prev_null_all": int(frame["takeoff_minus_stand_prev_arr_sec"].null_count()),
        "stand_prev_null_unmatched": int(lirf_u["takeoff_minus_stand_prev_arr_sec"].null_count()),
    }
    _save_state(state)
    _log(f"u3 prepare: {state['phases']['u3_prepare']}")


U3_VARIANTS = {"stand": list(U6_NUMS) + list(STAND_NUMS), "nostand": list(U6_NUMS)}


def _fit_g(train: pl.DataFrame, nums: list[str]):
    from catboost import CatBoostRegressor

    y = train["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    sd = train["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    g = gate_delay_target(y, sd)
    ok = np.isfinite(g)
    model = CatBoostRegressor(**G_PARAMS)
    model.fit(_pool(train.filter(pl.Series(ok)), list(U6_CATS), nums, g[ok]))
    return model, int(ok.sum())


def phase_u3_fit(state: dict[str, Any]) -> None:
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    (OUT / "oof").mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(OUT / "frames" / "u3_lirf_all_D.parquet")
    done = state["phases"].setdefault("u3_fit", {})
    parts = []
    for holdout in BLOCKS:
        name = f"{holdout[0]}_{holdout[1]}"
        path = OUT / "oof" / f"u3_oof_{name}.parquet"
        if path.exists():
            parts.append(pl.read_parquet(path))
            _log(f"u3 fold {name}: cached")
            continue
        started = time.time()
        train = frame.filter(~pl.col("month").is_in(list(holdout)))
        hold = frame.filter(pl.col("month").is_in(list(holdout)) & (pl.col("nm_unmatched") == 1))
        _assert_months(train["month"].unique().to_list(), tuple(m for m in D_MONTHS if m not in holdout), f"u3 fold {name}")
        cols = {}
        info = {}
        for variant, nums in U3_VARIANTS.items():
            model, n = _fit_g(train, nums)
            model.save_model(str(OUT / "models" / f"u3_g_{variant}_{name}.cbm"))
            cols[f"g_hat_{variant}"] = np.asarray(model.predict(_pool(hold, list(U6_CATS), nums)), dtype=np.float64)
            info[variant] = n
        part = hold.select("MVT_ID_mvt", "month").with_columns([pl.Series(k, v) for k, v in cols.items()])
        part.write_parquet(path)
        parts.append(part)
        done[name] = {"n_train": train.height, "n_train_unmatched": int((train["nm_unmatched"] == 1).sum()), "n_hold_unmatched": hold.height, "fit_rows": info, "seconds": time.time() - started}
        _save_state(state)
        _log(f"u3 fold {name}: {done[name]}")
    oof = pl.concat(parts)
    oof.write_parquet(OUT / "oof" / "u3_oof_D.parquet")
    done["status"] = "complete"
    _save_state(state)


def phase_u3_score(state: dict[str, Any]) -> None:
    from prc_challenge.unmatched_recovery.schema import compose_experimental_unmatched

    ctrl = pl.read_parquet(OUT / "frames" / "u3_unmatched_control_D.parquet")
    oof = pl.read_parquet(OUT / "oof" / "u3_oof_D.parquet")
    frame = ctrl.join(oof.drop("month"), on="MVT_ID_mvt", how="left", maintain_order="left").with_columns(
        pl.col("month").map_elements(_fold_name, return_dtype=pl.Utf8).alias("fold")
    )
    lirf = frame["AIRPORT"].to_numpy() == "LIRF"
    if frame.filter(pl.col("AIRPORT") == "LIRF")["g_hat_stand"].null_count():
        raise RuntimeError("G_hat missing on LIRF unmatched rows")
    y = frame["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    sd = frame["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    a3b_raw = frame["a3b_raw"].to_numpy()
    control = frame["a3b_deployed"].to_numpy()
    gate = frame.select(list(GATE_COLS))

    def deployed(pred: np.ndarray) -> np.ndarray:
        return compose_experimental_unmatched(gate.with_columns(pl.Series("prediction", pred)))["prediction"].to_numpy()

    arms: dict[str, np.ndarray] = {}
    for variant in U3_VARIANTS:
        t = serve_gate_delay(sd, frame[f"g_hat_{variant}"].fill_null(np.nan).to_numpy())
        t = np.where(lirf, t, np.nan)
        for lam, label in ((U3_LAMBDA, "l025"), (U3_LAMBDA_DIAGNOSTIC, "l050"), (1.0, "T_alone")):
            arms[f"{variant}|{label}"] = deployed(blend(a3b_raw, t, lam))
    primary = f"stand|l025"
    reports, gated, rows = {}, {}, []
    airport = frame["AIRPORT"].to_numpy()
    fold = frame["fold"].to_numpy()
    for name, pred in arms.items():
        rep = u3_report(y, pred, control, airport=airport, fold=fold)
        g = u3_gates(rep)
        reports[name] = rep
        gated[name] = g
        rows.append({
            "arm": name,
            "role": "candidate" if name == primary else "diagnostic",
            "unmatched_sse_ratio": rep["pooled"]["sse_ratio"],
            "unmatched_rmse": rep["pooled"]["rmse"],
            "control_rmse": rep["pooled"]["base_rmse"],
            "lirf_sse_ratio": rep["lirf"]["sse_ratio"],
            "lirf_ordinary_rmse": rep["lirf_ordinary_lt_1h"]["rmse"],
            "control_lirf_ordinary_rmse": rep["lirf_ordinary_lt_1h"]["base_rmse"],
            "lirf_ordinary_ratio": rep["lirf_ordinary_lt_1h"]["rmse"] / rep["lirf_ordinary_lt_1h"]["base_rmse"],
            "lirf_ge_1h_sse_ratio": rep["lirf_ge_1h"]["sse_ratio"],
            "leave_top5_ratio": rep["leave_top_k"]["sse_ratio"],
            "top5_share_of_gain": rep["leave_top_k"]["top_share_of_gain"],
            "non_lirf_max_abs_change": rep["non_lirf_max_abs_change"],
            **{f"fold_{k}_ratio": v["sse_ratio"] for k, v in rep["folds"].items()},
            "all_gates_pass": g["go"],
            "failed": ";".join(g["failed"]),
        })
    decision = "GO" if gated[primary]["go"] else "PARK"
    table = pl.DataFrame(rows)
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    TRACKED.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT / "metrics" / "u3_arm_table.csv")
    table.write_csv(TRACKED / "u3_arm_table.csv")
    lirf_rows = frame.filter(pl.col("AIRPORT") == "LIRF")
    t_primary = serve_gate_delay(lirf_rows["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy(), lirf_rows["g_hat_stand"].to_numpy())
    payload = {
        "population": f"D unmatched departures n={frame.height}; LIRF n={int(lirf.sum())}",
        "control": "A3b-nodest (month-isolated P(copy)) D OOF, E17/E18 + clip",
        "primary": primary,
        "decision": decision,
        "gates": gated,
        "t_distribution_lirf": {"mean": float(np.nanmean(t_primary)), "p50": float(np.nanmedian(t_primary)), "a3b_raw_mean": float(lirf_rows["a3b_raw"].mean())},
        "reports": {k: {kk: vv for kk, vv in v.items() if kk != "leave_top_k"} | {"leave_top_k": {kk: vv for kk, vv in v["leave_top_k"].items() if kk != "top_rows"}} for k, v in reports.items()},
    }
    _write_json(OUT / "metrics" / "u3_score.json", payload)
    _write_json(TRACKED / "u3_score.json", payload)
    state["phases"]["u3_score"] = {"status": "complete", "decision": decision}
    _save_state(state)
    print(table.select("arm", "role", "unmatched_sse_ratio", "lirf_sse_ratio", "lirf_ordinary_ratio", "leave_top5_ratio", "all_gates_pass", "failed"))
    _log(f"u3 decision {decision}")


def phase_u3_confirm(state: dict[str, Any]) -> None:
    from prc_challenge.unmatched_recovery.schema import compose_experimental_unmatched

    if (state["phases"].get("u3_score") or {}).get("decision") != "GO":
        raise RuntimeError("U3 confirmation refused: the candidate did not pass every D gate")
    if (OUT / "confirm" / "u3_confirmation.json").exists():
        raise RuntimeError("U3 confirmation already run once; refusing a second look")
    (OUT / "confirm").mkdir(parents=True, exist_ok=True)
    train = pl.read_parquet(OUT / "frames" / "u3_lirf_all_D.parquet")
    _assert_months(train["month"].unique().to_list(), D_MONTHS, "u3 confirm fit")
    model, _ = _fit_g(train, U3_VARIANTS["stand"])
    model.save_model(str(OUT / "models" / "u3_g_stand_confirm_D.cbm"))
    jj = _u3_frame(PRIMARY_MONTHS).filter(pl.col("nm_unmatched") == 1)
    g_hat = np.asarray(model.predict(_pool(jj, list(U6_CATS), U3_VARIANTS["stand"])), dtype=np.float64)
    t = serve_gate_delay(jj["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy(), g_hat)
    t_rows = jj.select("MVT_ID_mvt", *[c for c in GATE_COLS if c != "MVT_ID_mvt" and c != "AIRPORT"]).with_columns(pl.Series("T", t))
    t_rows.write_parquet(OUT / "confirm" / "u3_janjul_label_free_T.parquet")
    _log(f"u3 confirm: T fixed for {t_rows.height} Jan/Jul LIRF unmatched rows (label-free); scoring once")
    conf = pl.read_parquet(CONFIRM_A3B)
    base = pl.read_parquet(E08B_CACHE, columns=["MVT_ID_mvt", "month", "type_missing", "takeoff_minus_sched_days", "takeoff_minus_sched_sec", "takeoff_minus_iobt_sec", "takeoff_minus_eobt_sec"]).filter(pl.col("month").is_in(list(PRIMARY_MONTHS))).drop("month")
    frame = conf.join(base, on="MVT_ID_mvt", how="left").join(t_rows.select("MVT_ID_mvt", "T"), on="MVT_ID_mvt", how="left")
    if frame.height != 344_419:
        raise RuntimeError(f"Jan/Jul population {frame.height}")
    matched = frame["nm_unmatched"].to_numpy() == 0
    lo, hi = window_from_leads(frame["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), frame["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy())
    lo = np.where(matched, lo, np.nan)
    hi = np.where(matched, hi, np.nan)
    ctrl = project(frame["prediction"].to_numpy(), lo, hi)
    tt = frame["T"].fill_null(np.nan).to_numpy()
    blended = blend(frame["prediction"].to_numpy(), tt, U3_LAMBDA)
    composed = compose_experimental_unmatched(frame.select(*GATE_COLS).with_columns(pl.Series("prediction", blended)))["prediction"].to_numpy()
    # Matched rows keep the control; composition only moves unmatched LIRF rows.
    cand = np.where(matched, ctrl, composed)
    focus = (frame["AIRPORT"].to_numpy() == "LIRF") & ~matched
    payload = _janjul_scores(frame, ctrl, cand, focus=focus, focus_name="lirf_unmatched")
    payload.update({"lambda": U3_LAMBDA, "fit_months": list(D_MONTHS), "n_T_rows": int(np.isfinite(tt).sum()), "matched_rows_changed": int(np.sum(np.abs(cand[matched] - ctrl[matched]) > 1e-9))})
    frame.select("MVT_ID_mvt", "AIRPORT", "month", "nm_unmatched", "TAXITIME_SEC_mvt").with_columns(pl.Series("control", ctrl), pl.Series("prediction", cand)).write_parquet(OUT / "confirm" / "u3_confirm.parquet")
    _write_json(OUT / "confirm" / "u3_confirmation.json", payload)
    _write_json(TRACKED / "u3_confirmation.json", payload)
    state["phases"]["u3_confirm"] = {"status": "complete", "full_rmse": payload["candidate"]["full_rmse"], "control_full_rmse": payload["control"]["full_rmse"]}
    _save_state(state)
    _log(f"u3 confirmation: full {payload['candidate']['full_rmse']:.3f} vs control {payload['control']['full_rmse']:.3f}")


PHASES = {
    "u2_prepare": phase_u2_prepare,
    "u2_fit": phase_u2_fit,
    "u2_score": phase_u2_score,
    "u2_confirm": phase_u2_confirm,
    "u3_prepare": phase_u3_prepare,
    "u3_fit": phase_u3_fit,
    "u3_score": phase_u3_score,
    "u3_confirm": phase_u3_confirm,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=list(PHASES))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state = _state()
    PHASES[args.phase](state)


if __name__ == "__main__":
    main()
