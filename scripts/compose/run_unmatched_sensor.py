# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-UNM3 unmatched sensor campaign (2026-09-26): U-SENS, U-REGIME, U-BOTH.

Pre-registered in progress-review/unmatched-sensor-20260926/preregistration.json before any fit.

Phases (resumable):
  prepare      control (CAP_B + P1 D OOF) reproduction; regime features for D / Jan-Jul / ranking
               unmatched rows and the months 1-10 fit frame; sensor training frame (label-free joins).
  sens_fit     U-SENS per blocked fold (train months only), OOF on the D unmatched rows.
  regime_fit   parity refit (must reproduce expert_octfix_D) and U-REGIME expert per fold.
  score        arms, gates, selection (no labels read before this phase except inside fits).
  confirm      ONE frozen JAN_JUL_2025 look of the selected arm.
  ranking      deliverable (only if GO and CONSISTENT).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_key, "3")

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
OUT = MAIN / "artifacts" / "experiments" / "Unmatched-sensor-20260926"
TRACKED = ROOT / "progress-review" / "unmatched-sensor-20260926"
POLISH = MAIN / "artifacts" / "experiments" / "Unmatched-polish-20260925"
UNC = MAIN / "artifacts" / "experiments" / "Unmatched-noncopy-20260925"
NONCOPY_R = UNC / "ranking"
ADSB = MAIN / "artifacts" / "experiments" / "ADSB-features-20260925" / "features"
E08B = MAIN / "artifacts" / "cache" / "e08b_e34_months_1_10.parquet"
RAW = MAIN / "data" / "raw"
DFIT_JJ = MAIN / "artifacts" / "experiments" / "S4-allyear-rebuild-20260925" / "runs" / "dfit_full_adsb_historder" / "predictions_dfit_full_adsb_historder_janjul.parquet"
V18 = MAIN / "submissions" / "unbelievable-snowflake_v18.parquet"
CAPB_P1_SUB = POLISH / "ranking" / "capb_p1_unmatched_submission.parquet"
V14_MATCHED_CHECKSUM = "1bf0c6dee84454f4858421d09ad36c719d2e5f032fa1013066225ccc303e3031"
GATE_COLS = ("MVT_ID_mvt", "AIRPORT", "nm_unmatched", "type_missing", "takeoff_minus_sched_days", "takeoff_minus_sched_sec")
D_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)
BLOCKS = ((2, 8), (3, 9), (4, 10), (5, 6))
REG_THREADS = 2
REG_D_ITER, REG_CONFIRM_ITER, REG_L2 = 220, 400, 16.0
MIN_FREE = 25.0

from prc_challenge.unmatched_recovery.noncopy_expert import GATE_SEC, expert_training_mask, mixture_prediction  # noqa: E402
from prc_challenge.unmatched_recovery.polish import LAMBDA, blend_on_mask, cap_expert  # noqa: E402
from prc_challenge.unmatched_recovery.schema import compose_experimental_unmatched  # noqa: E402
from prc_challenge.unmatched_recovery.sensor import (  # noqa: E402
    REGIME_COLS,
    S_CEILING_G,
    SENS_CATS,
    SENS_NUMS_U6,
    SENS_PARAMS,
    adsb_numeric_features,
    arm_gates,
    cap,
    delta_sse_bootstrap,
    lirf_copy_regime,
    matched_departures,
    regime_features,
    sensor_blend,
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=1, sort_keys=False) + "\n")


def _state() -> dict[str, Any]:
    p = OUT / "state.json"
    st = json.loads(p.read_text()) if p.exists() else {"phases": {}}
    st.pop("_loaded", None)
    return st


def _save_state(state: dict[str, Any]) -> None:
    """Merge into the file on disk (phases may run concurrently): only phases changed in memory overwrite."""

    disk = _state()
    for name, value in state["phases"].items():
        disk["phases"][name] = value if value != state.get("_loaded", {}).get(name) else disk["phases"].get(name, value)
    _write_json(OUT / "state.json", {"phases": disk["phases"]})


def _free_mem() -> float | None:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[-1]
        return float(out.split(":")[-1].strip().rstrip("%"))
    except Exception:
        return None


def _require_memory(ctx: str) -> None:
    free = _free_mem()
    _log(f"{ctx}: memory free {free}%")
    if free is not None and free < MIN_FREE:
        raise SystemExit(f"{ctx}: memory free {free}% < {MIN_FREE}%")


def _aligned(base: pl.DataFrame, other: pl.DataFrame, col: str, *, allow_null: bool = False) -> np.ndarray:
    j = base.select("MVT_ID_mvt").join(other.select("MVT_ID_mvt", col), on="MVT_ID_mvt", how="left", maintain_order="left")
    if j.height != base.height or (not allow_null and j[col].null_count()):
        raise RuntimeError(f"{col}: alignment failed")
    return j[col].cast(pl.Float64).fill_null(np.nan).to_numpy()


def _fold(month: int) -> str:
    for a, b in BLOCKS:
        if month in (a, b):
            return f"{a}_{b}"
    raise ValueError(month)


def _composer(gate: pl.DataFrame):
    def fn(pred: np.ndarray) -> np.ndarray:
        return compose_experimental_unmatched(gate.with_columns(pl.Series("prediction", np.asarray(pred, dtype=np.float64))))["prediction"].to_numpy()

    return fn


def _raw_files(months: tuple[int, ...]) -> list[Path]:
    files = []
    for m in months:
        nxt = f"2025-{m + 1:02d}-01" if m < 12 else "2026-01-01"
        files.append(RAW / f"training_2025-{m:02d}-01_{nxt}.parquet")
    return files


def _matched_pool(months: tuple[int, ...], ranking: bool = False) -> pl.DataFrame:
    parts = [matched_departures(p) for p in _raw_files(months)]
    if ranking:
        parts.append(matched_departures(RAW / "ranking.parquet"))
    return pl.concat(parts)


def _mvt_times(ids: pl.Series, *, ranking: bool = False, months: tuple[int, ...] = ()) -> pl.DataFrame:
    paths = [RAW / "ranking.parquet"] if ranking else _raw_files(months)
    frames = [pl.scan_parquet(p).filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", pl.col("ADEP_mvt").alias("AIRPORT"), "MVT_TIME_UTC_mvt") for p in paths]
    t = pl.concat(frames).filter(pl.col("MVT_ID_mvt").is_in(ids.implode())).collect()
    out = pl.DataFrame({"MVT_ID_mvt": ids}).join(t, on="MVT_ID_mvt", how="left", maintain_order="left")
    if out["MVT_TIME_UTC_mvt"].null_count():
        raise RuntimeError("takeoff times missing")
    return out


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _control_d() -> pl.DataFrame:
    arms = pl.read_parquet(POLISH / "oof" / "arms_D.parquet")
    ctrl = pl.read_parquet(POLISH / "frames" / "control_D.parquet")
    if arms["MVT_ID_mvt"].to_list() != ctrl["MVT_ID_mvt"].to_list():
        raise RuntimeError("arms/control order differs")
    deployed = _composer(ctrl.select(list(GATE_COLS)))
    sd = ctrl["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    p = arms["p_copy"].to_numpy()
    t = arms["T"].to_numpy()
    lirf = ctrl["AIRPORT"].to_numpy() == "LIRF"
    e_cap, _ = cap_expert(arms["expert_octfix"].to_numpy(), sd, 3606.0)
    raw_capb = mixture_prediction(p, sd, e_cap, gate_sec=GATE_SEC)
    capb = deployed(raw_capb)
    p1_on_a3b = deployed(blend_on_mask(arms["a3b_raw"].to_numpy(), t, lirf, LAMBDA))
    identity = {"capb_vs_arms_CAP_B": float(np.max(np.abs(capb - arms["CAP_B"].to_numpy()))), "p1_vs_arms_P1": float(np.max(np.abs(p1_on_a3b - arms["P1"].to_numpy())))}
    if max(identity.values()) > 1e-6:
        raise RuntimeError(f"control does not reproduce: {identity}")
    raw_ctl = blend_on_mask(raw_capb, t, lirf, LAMBDA)
    ctl = deployed(raw_ctl)
    s1 = pl.read_parquet(POLISH / "frames" / "d_unmatched_s1_features.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"])
    out = ctrl.select(*GATE_COLS, "month", "TAXITIME_SEC_mvt").with_columns(
        pl.Series("p", p), pl.Series("T", t), pl.Series("e_cap", e_cap), pl.Series("raw_ctl", raw_ctl), pl.Series("ctl", ctl), pl.Series("capb", capb)
    ).join(s1, on="MVT_ID_mvt", how="left", maintain_order="left")
    if out["MVT_TIME_UTC_mvt"].null_count():
        raise RuntimeError("takeoff time missing on D control rows")
    return out, identity


def phase_prepare(state: dict[str, Any]) -> None:
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    ctl, identity = _control_d()
    blk_d = pl.read_parquet(ADSB / "adsb_block_D.parquet", columns=["MVT_ID_mvt", "adsb_coverage"])
    ctl = ctl.join(blk_d, on="MVT_ID_mvt", how="left", maintain_order="left")
    ctl.write_parquet(OUT / "frames" / "control_D.parquet")
    _log(f"control CAP_B+P1 D OOF reproduced ({identity}); n {ctl.height}")

    # regime features (label-free: other flights' MVT and AOBT_3 only)
    pool = _matched_pool(tuple(range(1, 11)), ranking=True)
    _log(f"matched pool for regime features: {pool.height} departures")
    rg_d = regime_features(ctl.select("MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt"), pool)
    rg_d.write_parquet(OUT / "frames" / "regime_D_unmatched.parquet")
    jj = pl.read_parquet(POLISH / "frames" / "janjul_unmatched_s1_features_nolabel.parquet", columns=["MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt"])
    regime_features(jj, pool).write_parquet(OUT / "frames" / "regime_janjul_unmatched.parquet")
    rk = pl.read_parquet(NONCOPY_R / "frames" / "ranking_unmatched.parquet", columns=["MVT_ID_mvt", "AIRPORT"])
    rk_t = _mvt_times(rk["MVT_ID_mvt"], ranking=True)
    if (rk_t["AIRPORT"] != rk["AIRPORT"]).any():
        raise RuntimeError("ranking airport mismatch")
    regime_features(rk_t, pool).write_parquet(OUT / "frames" / "regime_ranking_unmatched.parquet")
    fit = pl.read_parquet(NONCOPY_R / "frames" / "expert_fit_m1_10.parquet", columns=["MVT_ID_mvt", "AIRPORT", "month"])
    fit_t = _mvt_times(fit["MVT_ID_mvt"], months=tuple(range(1, 11)))
    regime_features(fit_t, pool).write_parquet(OUT / "frames" / "regime_fit_m1_10.parquet")
    _log("regime features written (D, Jan/Jul, ranking, months 1-10 fit frame)")

    # sensor training frame: e08b U6 columns + ADS-B block, coverage >= 2 (label kept for fitting only)
    e = pl.read_parquet(E08B, columns=["MVT_ID_mvt", "month", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", *SENS_CATS, *SENS_NUMS_U6])
    blk_cols = pl.read_parquet_schema(ADSB / "adsb_block_D.parquet")
    adsb_nums = adsb_numeric_features(list(blk_cols))
    parts = []
    for name, months in (("adsb_block_D.parquet", D_MONTHS), ("adsb_block_janjul.parquet", (1, 7))):
        b = pl.read_parquet(ADSB / name, columns=["MVT_ID_mvt", *adsb_nums]).filter(pl.col("adsb_coverage") >= 2)
        parts.append(e.filter(pl.col("month").is_in(list(months))).join(b, on="MVT_ID_mvt", how="inner"))
    train = pl.concat(parts)
    lin = pl.concat([
        pl.read_parquet(POLISH / "frames" / "d_unmatched_octfix_features.parquet", columns=["MVT_ID_mvt", "source_class"]),
        pl.read_parquet(NONCOPY_R / "frames" / "lineage_fit_m1_10.parquet", columns=["MVT_ID_mvt", "month", "source_class"]).filter(pl.col("month").is_in([1, 7])).drop("month"),
    ])
    train = train.join(lin, on="MVT_ID_mvt", how="left")
    miss = train.filter((pl.col("nm_unmatched") == 1) & pl.col("source_class").is_null()).height
    if miss:
        raise RuntimeError(f"source class missing on {miss} unmatched sensor rows")
    sd = train["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    keep_u = expert_training_mask(train["source_class"].fill_null("__matched__").to_numpy(), sd, mode="noncopy_large")
    train = train.with_columns(pl.Series("sens_train_ok", keep_u))
    train.write_parquet(OUT / "frames" / "sensor_train.parquet")
    info = {
        "status": "complete",
        "control_identity": identity,
        "control_rmse_informational": None,
        "adsb_features": adsb_nums,
        "sensor_rows": {str(m): int(c) for m, c in train.group_by("month").len().sort("month").iter_rows()},
        "sensor_unmatched_rows": int((train["nm_unmatched"] == 1).sum()),
        "sensor_dropped_large_copies": int((~keep_u).sum()),
    }
    state["phases"]["prepare"] = info
    _save_state(state)
    _log(f"sensor frame: {train.height} rows, unmatched {info['sensor_unmatched_rows']}, dropped copies {info['sensor_dropped_large_copies']}, {len(adsb_nums)} adsb features")


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------


def _sens_features() -> tuple[list[str], list[str]]:
    info = _state()["phases"]["prepare"]
    return list(SENS_CATS), [*SENS_NUMS_U6, *info["adsb_features"]]


def _unmatched_sensor_frame(ids_frame: pl.DataFrame, u6: pl.DataFrame, block: Path) -> pl.DataFrame:
    cats, nums = _sens_features()
    adsb_nums = [c for c in nums if c.startswith("adsb_")]
    b = pl.read_parquet(block, columns=["MVT_ID_mvt", *adsb_nums])
    f = ids_frame.select("MVT_ID_mvt").join(u6.select("MVT_ID_mvt", *[c for c in [*cats, *SENS_NUMS_U6] if c in u6.columns]), on="MVT_ID_mvt", how="left", maintain_order="left")
    return f.join(b, on="MVT_ID_mvt", how="left", maintain_order="left")


def _fit_sensor(train: pl.DataFrame, cats: list[str], nums: list[str]):
    from prc_challenge.unmatched_recovery.methods import fit_regressor

    return fit_regressor(train, cats, nums, **SENS_PARAMS)


def phase_sens_fit(state: dict[str, Any]) -> None:
    from prc_challenge.unmatched_recovery.methods import predict_regressor

    _require_memory("sens_fit")
    cats, nums = _sens_features()
    tr_all = pl.read_parquet(OUT / "frames" / "sensor_train.parquet").filter(pl.col("month").is_in(list(D_MONTHS)) & pl.col("sens_train_ok"))
    ctl = pl.read_parquet(OUT / "frames" / "control_D.parquet")
    u6 = pl.read_parquet(E08B, columns=["MVT_ID_mvt", *SENS_CATS, *SENS_NUMS_U6]).filter(pl.col("MVT_ID_mvt").is_in(ctl["MVT_ID_mvt"].implode()))
    frame = _unmatched_sensor_frame(ctl, u6, ADSB / "adsb_block_D.parquet").with_columns(ctl["month"])
    (OUT / "models").mkdir(exist_ok=True)
    done = state["phases"].setdefault("sens_fit", {"folds": {}})
    parts = []
    for hold in BLOCKS:
        name = f"{hold[0]}_{hold[1]}"
        path = OUT / "oof" / f"sens_{name}.parquet"
        if path.exists() and done["folds"].get(name):
            parts.append(pl.read_parquet(path))
            continue
        train = tr_all.filter(~pl.col("month").is_in(list(hold)))
        if set(train["month"].unique().to_list()) & set(hold):
            raise RuntimeError("holdout leaked")
        t0 = time.time()
        model = _fit_sensor(train, cats, nums)
        model.save_model(str(OUT / "models" / f"sens_{name}.cbm"))
        h = frame.filter(pl.col("month").is_in(list(hold)))
        pred = predict_regressor(model, h, cats, nums)
        part = h.select("MVT_ID_mvt").with_columns(pl.Series("S", pred))
        (OUT / "oof").mkdir(exist_ok=True)
        part.write_parquet(path)
        parts.append(part)
        done["folds"][name] = {"n_train": train.height, "n_train_unmatched": int((train["nm_unmatched"] == 1).sum()), "n_hold": h.height, "seconds": round(time.time() - t0, 1)}
        _save_state(state)
        _log(f"sensor fold {name}: train {train.height} (unmatched {done['folds'][name]['n_train_unmatched']}), hold {h.height}, {time.time() - t0:.0f} s")
    oof = pl.concat(parts)
    ctl.select("MVT_ID_mvt").join(oof, on="MVT_ID_mvt", how="left", maintain_order="left").write_parquet(OUT / "oof" / "sens_D.parquet")
    done["status"] = "complete"
    _save_state(state)


def _regime_work() -> tuple[pl.DataFrame, list[str], list[str]]:
    work = pl.read_parquet(POLISH / "frames" / "d_unmatched_octfix_features.parquet").sort("_row")
    rg = pl.read_parquet(OUT / "frames" / "regime_D_unmatched.parquet")
    work = work.join(rg, on="MVT_ID_mvt", how="left", maintain_order="left")
    feats = json.loads((UNC / "frames" / "features.json").read_text())
    return work, feats["cats"], feats["nums"]


def _fit_expert_folds(work: pl.DataFrame, cats: list[str], nums: list[str], variant: str) -> pl.DataFrame:
    from prc_challenge.unmatched_recovery.methods import fit_regressor, predict_regressor

    parts = []
    for hold in BLOCKS:
        fit = work.filter(~pl.col("month").is_in(list(hold)))
        h = work.filter(pl.col("month").is_in(list(hold)))
        mask = expert_training_mask(fit["source_class"].to_numpy(), fit["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy(), mode="noncopy_large")
        train = fit.filter(pl.Series(mask))
        model = fit_regressor(train, cats, nums, iterations=REG_D_ITER, thread_count=REG_THREADS, l2_leaf_reg=REG_L2)
        model.save_model(str(OUT / "models" / f"expert_{variant}_{hold[0]}_{hold[1]}.cbm"))
        parts.append(h.select("MVT_ID_mvt", "_row").with_columns(pl.Series("prediction", predict_regressor(model, h, cats, nums))))
        _log(f"expert {variant} fold {hold}: train {train.height}, hold {h.height}, features {len(cats) + len(nums)}")
    return pl.concat(parts).sort("_row").drop("_row")


def phase_regime_fit(state: dict[str, Any]) -> None:
    _require_memory("regime_fit")
    (OUT / "models").mkdir(exist_ok=True)
    (OUT / "oof").mkdir(exist_ok=True)
    work, cats, nums = _regime_work()
    par = _fit_expert_folds(work, cats, nums, "parity")
    ref = pl.read_parquet(POLISH / "oof" / "expert_octfix_D.parquet")
    gap = float(np.max(np.abs(_aligned(par, ref, "prediction") - par["prediction"].to_numpy())))
    _log(f"parity refit vs expert_octfix_D: max |diff| {gap}")
    if gap > 1e-6:
        raise RuntimeError("parity refit does not reproduce the deployed expert OOF")
    reg = _fit_expert_folds(work, cats, [*nums, *REGIME_COLS], "regime")
    reg.write_parquet(OUT / "oof" / "expert_regime_D.parquet")
    state["phases"]["regime_fit"] = {"status": "complete", "parity_max_abs_diff": gap, "n_features": len(cats) + len(nums) + len(REGIME_COLS)}
    _save_state(state)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def _arms_d() -> dict[str, Any]:
    ctl = pl.read_parquet(OUT / "frames" / "control_D.parquet")
    deployed = _composer(ctl.select(list(GATE_COLS)))
    sd = ctl["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    p, t, e_cap = (ctl[c].to_numpy() for c in ("p", "T", "e_cap"))
    airport = ctl["AIRPORT"].to_numpy()
    lirf = airport == "LIRF"
    cov = ctl["adsb_coverage"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    s = _aligned(ctl, pl.read_parquet(OUT / "oof" / "sens_D.parquet"), "S", allow_null=True)
    e_reg_cap = cap(_aligned(ctl, pl.read_parquet(OUT / "oof" / "expert_regime_D.parquet"), "prediction"), sd)

    def compose(e: np.ndarray) -> np.ndarray:
        return deployed(blend_on_mask(mixture_prediction(p, sd, e, gate_sec=GATE_SEC), t, lirf, LAMBDA))

    ctl_rebuilt = compose(e_cap)
    if float(np.max(np.abs(ctl_rebuilt - ctl["ctl"].to_numpy()))) > 1e-6:
        raise RuntimeError("control does not recompose")
    e_sens, elig = sensor_blend(e_cap, s, sd, cov, airport, p)
    e_both, elig_b = sensor_blend(e_reg_cap, s, sd, cov, airport, p)
    arms = {"U-SENS": compose(e_sens), "U-REGIME": compose(e_reg_cap), "U-BOTH": compose(e_both)}
    return {"ctl": ctl, "y": ctl["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy(), "control": ctl_rebuilt, "arms": arms, "eligible": elig, "s": s, "sd": sd, "p": p, "cov": cov, "airport": airport,
            "e_cap": e_cap, "e_reg_cap": e_reg_cap, "compose": compose,
            "day": ctl["MVT_TIME_UTC_mvt"].dt.date().cast(pl.String).to_numpy(), "fold": np.array([_fold(m) for m in ctl["month"].to_list()], dtype=object),
            "lirf_copy": lirf_copy_regime(airport, sd, p)}


def phase_score(state: dict[str, Any]) -> None:
    a = _arms_d()
    y, control, arms = a["y"], a["control"], a["arms"]
    kw = {"day": a["day"], "fold": a["fold"], "airport": a["airport"]}
    rep = {
        "U-SENS": arm_gates(y, control, arms["U-SENS"], lirf_copy=a["lirf_copy"], **kw),
        "U-REGIME": arm_gates(y, control, arms["U-REGIME"], lirf_copy=None, **kw),
    }
    both_eligible = rep["U-SENS"]["go"] and rep["U-REGIME"]["go"]
    rep["U-BOTH"] = arm_gates(y, control, arms["U-BOTH"], lirf_copy=a["lirf_copy"], reference=arms["U-REGIME"], **kw)
    rep["U-BOTH"]["eligible"] = both_eligible
    passing = [k for k in ("U-SENS", "U-REGIME") if rep[k]["go"]] + (["U-BOTH"] if both_eligible and rep["U-BOTH"]["go"] else [])
    chosen = min(passing, key=lambda k: rep[k]["cand_sse"]) if passing else None
    # diagnostics (never used for selection)
    elig = a["eligible"]
    diag: dict[str, Any] = {"eligible_rows": int(elig.sum()), "eligible_by_airport": {str(x): int((elig & (a["airport"] == x)).sum()) for x in sorted(set(a["airport"].tolist()))}}
    s_cap = cap(np.nan_to_num(a["s"]), a["sd"])
    for label, m in (("cov3", elig & (a["cov"] >= 3)), ("cov2", elig & (a["cov"] == 2))):
        if m.any():
            diag[f"eligible_{label}"] = {
                "n": int(m.sum()),
                "rmse_control": float(np.sqrt(np.mean((y[m] - control[m]) ** 2))),
                "rmse_u_sens": float(np.sqrt(np.mean((y[m] - arms["U-SENS"][m]) ** 2))),
                "rmse_sensor_alone_capped": float(np.sqrt(np.mean((y[m] - s_cap[m]) ** 2))),
                "rmse_e_cap_alone": float(np.sqrt(np.mean((y[m] - a["e_cap"][m]) ** 2))),
            }
    # pooled least-squares weight on eligible rows (diagnostic): minimise over w of SSE(compose(E_cap + w*(S_cap - E_cap)))
    grid = {}
    for w in (0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0):
        e_w = np.where(elig, w * s_cap + (1 - w) * a["e_cap"], a["e_cap"])
        grid[str(w)] = float(np.sum((y - a["compose"](e_w)) ** 2))
    diag["uniform_w_grid_sse"] = grid
    payload = {"control": "CAP_B + P1 D OOF (E17/E18 + clip)", "control_rmse": float(np.sqrt(np.mean((y - control) ** 2))), "reports": rep, "passing": passing, "selected_arm": chosen,
               "decision": "GO" if chosen else "NO-GO", "diagnostics": diag}
    _write_json(OUT / "metrics" / "score.json", payload)
    _write_json(TRACKED / "score.json", payload)
    OUT.joinpath("oof").mkdir(exist_ok=True)
    a["ctl"].select("MVT_ID_mvt", "month", "AIRPORT", "TAXITIME_SEC_mvt").with_columns(
        pl.Series("control", control), *[pl.Series(k, v) for k, v in arms.items()], pl.Series("eligible_sens", elig), pl.Series("S", a["s"])
    ).write_parquet(OUT / "oof" / "arms_D.parquet")
    state["phases"]["score"] = {"status": "complete", "decision": payload["decision"], "selected_arm": chosen, "passing": passing}
    _save_state(state)
    for k, r in rep.items():
        _log(f"{k}: rmse {r['ctl_rmse']:.2f} -> {r['cand_rmse']:.2f}, dSSE {r['delta_sse']:.4g}, boot p95 {r['bootstrap']['p95']:.4g}, folds {[round(v['delta_sse'] / 1e6, 1) for v in r['folds'].values()]} (1e6), worst airport {r['gates']['G3_no_airport_gt_2pct']['value']:+.4f}, changed {r['rows_changed']}, go {r['go']} {r['failed']}")
    _log(f"decision {payload['decision']} selected {chosen}")


# ---------------------------------------------------------------------------
# confirm (Jan/Jul 2025, single look)
# ---------------------------------------------------------------------------


def _janjul_matched() -> dict[str, Any]:
    from prc_challenge.matching_window import project_predictions

    d = pl.read_parquet(DFIT_JJ)
    raw = pl.concat([pl.scan_parquet(p).filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt") for p in _raw_files((1, 7))]).collect()
    j = d.join(raw, on="MVT_ID_mvt", how="left", maintain_order="left")
    if j["MVT_TIME_UTC_mvt"].null_count():
        raise RuntimeError("Jan/Jul raw join failed")
    if ((j["nm_unmatched"] == 0) != j["FLIGHT_ID_mvt"].is_not_null()).any():
        raise RuntimeError("nm_unmatched disagrees with FLIGHT_ID")
    proj = project_predictions(j, "prediction", mode="lobt").with_columns(pl.col("prediction").clip(lower_bound=0))
    return {"frame": proj, "n_projected_changed": int((np.abs(proj["prediction"].to_numpy() - j["prediction"].to_numpy()) > 1e-9).sum())}


def phase_confirm(state: dict[str, Any]) -> None:
    from prc_challenge.unmatched_recovery.methods import fit_regressor, predict_regressor

    sc = state["phases"].get("score") or {}
    arm = sc.get("selected_arm")
    if sc.get("decision") != "GO" or not arm:
        raise SystemExit("confirmation refused: no arm passed the D gates")
    if (OUT / "confirm" / "confirmation.json").exists():
        raise SystemExit("confirmation already run once; refusing a second look")
    _require_memory("confirm")
    (OUT / "confirm").mkdir(parents=True, exist_ok=True)
    jj = pl.read_parquet(POLISH / "frames" / "janjul_unmatched_s1_features_nolabel.parquet").sort("_row")
    if "TAXITIME_SEC_mvt" in jj.columns:
        raise RuntimeError("label in Jan/Jul feature frame")
    capc = pl.read_parquet(POLISH / "confirm" / "janjul_unmatched_components_cap_CAP_B.parquet")
    p1c = pl.read_parquet(POLISH / "confirm" / "janjul_unmatched_components.parquet")
    sd = jj["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    airport = jj["AIRPORT"].to_numpy()
    lirf = airport == "LIRF"
    p = _aligned(jj, capc, "p_copy")
    e_cap = _aligned(jj, capc, "expert_capped")
    raw_capb = _aligned(jj, capc, "raw")
    t = _aligned(jj, p1c, "T", allow_null=True)
    deployed = _composer(jj.select(list(GATE_COLS)))
    ident = {"mixture_vs_saved_raw": float(np.max(np.abs(mixture_prediction(p, sd, e_cap, gate_sec=GATE_SEC) - raw_capb))), "compose_vs_saved_deployed": float(np.max(np.abs(deployed(raw_capb) - _aligned(jj, capc, "deployed"))))}
    if max(ident.values()) > 1e-6:
        raise RuntimeError(f"Jan/Jul components do not reproduce: {ident}")

    def compose(e: np.ndarray) -> np.ndarray:
        return deployed(blend_on_mask(mixture_prediction(p, sd, e, gate_sec=GATE_SEC), t, lirf, LAMBDA))

    ctl_u = compose(e_cap)
    fit_info: dict[str, Any] = {}
    e_new = e_cap
    cov = np.full(jj.height, np.nan)
    s = np.full(jj.height, np.nan)
    if arm in ("U-REGIME", "U-BOTH"):
        work, cats, nums = _regime_work()
        mask = expert_training_mask(work["source_class"].to_numpy(), work["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy(), mode="noncopy_large")
        train = work.filter(pl.Series(mask))
        model = fit_regressor(train, cats, [*nums, *REGIME_COLS], iterations=REG_CONFIRM_ITER, thread_count=REG_THREADS, l2_leaf_reg=REG_L2)
        model.save_model(str(OUT / "models" / "confirm_expert_regime_D.cbm"))
        jr = jj.join(pl.read_parquet(OUT / "frames" / "regime_janjul_unmatched.parquet"), on="MVT_ID_mvt", how="left", maintain_order="left")
        e_new = cap(predict_regressor(model, jr, cats, [*nums, *REGIME_COLS]), sd)
        fit_info["regime_expert"] = {"n_train": train.height, "iterations": REG_CONFIRM_ITER}
    if arm in ("U-SENS", "U-BOTH"):
        cats_s, nums_s = _sens_features()
        tr = pl.read_parquet(OUT / "frames" / "sensor_train.parquet").filter(pl.col("month").is_in(list(D_MONTHS)) & pl.col("sens_train_ok"))
        model_s = _fit_sensor(tr, cats_s, nums_s)
        model_s.save_model(str(OUT / "models" / "confirm_sens_D.cbm"))
        fr = _unmatched_sensor_frame(jj, jj, ADSB / "adsb_block_janjul.parquet")
        s = predict_regressor(model_s, fr, cats_s, nums_s)
        cov = fr["adsb_coverage"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        s = np.where(np.isnan(cov), np.nan, s)
        e_new, elig = sensor_blend(e_new, s, sd, cov, airport, p)
        fit_info["sensor"] = {"n_train": tr.height, "eligible_rows": int(elig.sum()), "eligible_by_airport": {str(x): int((elig & (airport == x)).sum()) for x in sorted(set(airport.tolist()))}}
    cand_u = compose(e_new)

    mt = _janjul_matched()
    frame = mt["frame"]
    y = frame["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()  # labels read here only
    matched = frame["nm_unmatched"].to_numpy() == 0
    ctl_full = np.where(matched, frame["prediction"].to_numpy(), _aligned(frame, jj.select("MVT_ID_mvt").with_columns(pl.Series("v", ctl_u)), "v", allow_null=True))
    cand_full = np.where(matched, frame["prediction"].to_numpy(), _aligned(frame, jj.select("MVT_ID_mvt").with_columns(pl.Series("v", cand_u)), "v", allow_null=True))
    if np.isnan(ctl_full).any() or np.isnan(cand_full).any():
        raise RuntimeError("Jan/Jul rows do not line up")
    um = ~matched
    rm = lambda a_, b_: float(np.sqrt(np.mean((a_ - b_) ** 2)))  # noqa: E731
    month = frame["month"].to_numpy()
    ap = frame["AIRPORT"].to_numpy()
    boot = delta_sse_bootstrap(y[um], ctl_full[um], cand_full[um], frame["MVT_TIME_UTC_mvt"].dt.date().cast(pl.String).to_numpy()[um] if "MVT_TIME_UTC_mvt" in frame.columns else month[um])
    res = {
        "arm": arm,
        "identity": ident,
        "fits": fit_info,
        "lobt_projection_rows_changed": mt["n_projected_changed"],
        "full_rmse": {"control": rm(y, ctl_full), "candidate": rm(y, cand_full)},
        "matched_rmse": rm(y[matched], ctl_full[matched]),
        "unmatched_rmse": {"control": rm(y[um], ctl_full[um]), "candidate": rm(y[um], cand_full[um]), "n": int(um.sum())},
        "unmatched_delta_sse": float(np.sum((y[um] - cand_full[um]) ** 2 - (y[um] - ctl_full[um]) ** 2)),
        "by_month": {str(m): {"full_control": rm(y[month == m], ctl_full[month == m]), "full_candidate": rm(y[month == m], cand_full[month == m]),
                              "unmatched_control": rm(y[um & (month == m)], ctl_full[um & (month == m)]), "unmatched_candidate": rm(y[um & (month == m)], cand_full[um & (month == m)])} for m in (1, 7)},
        "by_airport_unmatched": {str(x): {"n": int((um & (ap == x)).sum()), "control": rm(y[um & (ap == x)], ctl_full[um & (ap == x)]), "candidate": rm(y[um & (ap == x)], cand_full[um & (ap == x)]),
                                          "delta_sse": float(np.sum((y[um & (ap == x)] - cand_full[um & (ap == x)]) ** 2 - (y[um & (ap == x)] - ctl_full[um & (ap == x)]) ** 2))} for x in sorted(set(ap.tolist()))},
        "bootstrap_unmatched_delta_sse": boot,
    }
    res["reading"] = "CONSISTENT" if (res["unmatched_delta_sse"] < 0 and boot["p95"] < 0) else "FLAG"
    _write_json(OUT / "confirm" / "confirmation.json", res)
    _write_json(TRACKED / "confirmation.json", res)
    jj.select("MVT_ID_mvt", "AIRPORT", "month").with_columns(pl.Series("control", ctl_u), pl.Series("candidate", cand_u), pl.Series("S", s), pl.Series("adsb_coverage", cov), pl.Series("e_new", e_new)).write_parquet(OUT / "confirm" / "janjul_unmatched_components.parquet")
    state["phases"]["confirm"] = {"status": "complete", "arm": arm, "reading": res["reading"], "full_rmse": res["full_rmse"], "unmatched_rmse": res["unmatched_rmse"]}
    _save_state(state)
    _log(f"confirmation {arm}: full {res['full_rmse']['control']:.3f} -> {res['full_rmse']['candidate']:.3f}; unmatched {res['unmatched_rmse']['control']:.1f} -> {res['unmatched_rmse']['candidate']:.1f}; boot p95 {boot['p95']:.4g}; {res['reading']}")


def phase_score_g(state: dict[str, Any]) -> None:
    """Amendment 1: U-SENS-G = U-SENS with S <= 3606 s before the blend; D gates only (Jan/Jul spent)."""

    a = _arms_d()
    y, control = a["y"], a["control"]
    e_g, elig = sensor_blend(a["e_cap"], a["s"], a["sd"], a["cov"], a["airport"], a["p"], s_ceiling=S_CEILING_G)
    cand = a["compose"](e_g)
    rep = arm_gates(y, control, cand, day=a["day"], fold=a["fold"], airport=a["airport"], lirf_copy=a["lirf_copy"])
    rep["rows_ceiling_binding"] = int(np.sum(elig & (a["s"] > S_CEILING_G)))
    payload = {"arm": "U-SENS-G", "amendment": "preregistration_amendment_1_usens_g.json", "control": "CAP_B + P1 D OOF (E17/E18 + clip)",
               "control_rmse": float(np.sqrt(np.mean((y - control) ** 2))), "report": rep, "decision": "GO" if rep["go"] else "NO-GO",
               "note": "JAN_JUL_2025 is spent for this arm (ceiling informed by the U-SENS Jan/Jul look); evidence = these D gates + a public verification"}
    _write_json(OUT / "metrics" / "score_usens_g.json", payload)
    _write_json(TRACKED / "score_usens_g.json", payload)
    state["phases"]["score_g"] = {"status": "complete", "decision": payload["decision"]}
    _save_state(state)
    _log(f"U-SENS-G: rmse {rep['ctl_rmse']:.2f} -> {rep['cand_rmse']:.2f}, dSSE {rep['delta_sse']:.4g}, boot p95 {rep['bootstrap']['p95']:.4g}, folds {[round(v['delta_sse'] / 1e6, 1) for v in rep['folds'].values()]} (1e6), worst airport {rep['gates']['G3_no_airport_gt_2pct']['value']:+.4f}, ceiling binds {rep['rows_ceiling_binding']}, go {rep['go']} {rep['failed']}")


PHASES = {"score_g": phase_score_g, "prepare": phase_prepare, "sens_fit": phase_sens_fit, "regime_fit": phase_regime_fit, "score": phase_score, "confirm": phase_confirm}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=list(PHASES))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state = _state()
    state["_loaded"] = json.loads(json.dumps(state["phases"]))
    PHASES[args.phase](state)


if __name__ == "__main__":
    main()
