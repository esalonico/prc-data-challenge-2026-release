# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-UNM5 amendment 2 (D-074): U-MAP2V-NF vs ALL + C0 on the unseen Nov/Dec 2025 unmatched rows.

Every fitted component is the months 1-10 (or D-pool) artifact the ranking build used; none saw a Nov/Dec label:

  non-copy expert  Unmatched-polish-20260925/ranking/models/expert_noncopy_large_m1_10_octfix.cbm (months 1-10)
                   features: working_rows_from_base(S4 base caches + raw sidecar, months 11-12) = the path of the
                   model's own October training rows (final_freeze.dry_run parity on months 1-10)
  P(copy)          Unmatched-noncopy-20260925/ranking/models/pcopy_R-A3b-nodest.pkl (lineage months 1-10, n0 20);
                   Nov/Dec lineage month-isolated from Nov/Dec DEP (labels masked) + ARR rows, family features
                   against lineage_fit_m1_10.parquet (exclude_self False, as ranking)
  U-SENS-G sensor  Unmatched-sensor-20260926/ranking/models/sens_m1_10.cbm (months 1-10) on the Nov/Dec v2 block
                   (scripts/unm2026_v4_adsb.py: D-pool, parity with adsb_block_ranking)
  P1 T (LIRF)      Unmatched-polish-20260925/ranking/models/p1_g_stand_m1_10.cbm (months 1-10), T = max(sd - G_hat, 0)
  arm A            q_ranking (E17 near-copy share of D + JAN_JUL, months 1-10)
  arm B            rate map fitted on months 1-10 (a10, b10); input = the Nov/Dec LIRF ARR BLOCK==SCHED rate (ARR rows)
  arm C0           destination links from Nov/Dec raw clocks (label-free)
  map              adsb_v3lean_block_allyear (vote pool months 1-10; Nov/Dec rows use the whole pool)

Nov/Dec DEP labels are read only by ``score`` (TAXITIME_SEC_mvt of the S4 labels cache), never by a fit.
Phases: parity (ranking re-derivation of p, T, S) | build | score
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_key, "4")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


U = _load("run_unm2026", ROOT / "scripts" / "compose" / "run_unm2026.py")
L = _load("run_lirf_label_source", ROOT / "scripts" / "compose" / "run_lirf_label_source.py")
REC = U.REC

from prc_challenge.io import reporting_airport_expr  # noqa: E402
from prc_challenge.lirf_label_source import serve_gate_delay  # noqa: E402
from prc_challenge.mask import mask_departure_labels  # noqa: E402
from prc_challenge.unmatched_recovery.block_lineage import CLASSES  # noqa: E402
from prc_challenge.unmatched_recovery.block_source_model import predict_source_proba, with_family_features  # noqa: E402
from prc_challenge.unmatched_recovery.final_freeze import S4_FEATURES_REL, FinalFreezeGuard, working_rows_from_base  # noqa: E402
from prc_challenge.unmatched_recovery.methods import predict_regressor  # noqa: E402
from prc_challenge.unmatched_recovery.noncopy_expert import COPY_CLASS, month_isolated_lineage  # noqa: E402
from prc_challenge.unmatched_recovery.polish import cap_expert  # noqa: E402
from prc_challenge.unmatched_recovery.recovery import ArmSpec, Components, arr_eq_rates, attach_links, compose, destination_links, month_logit_shift, rate_target  # noqa: E402
from prc_challenge.unmatched_recovery.sensor import S_CEILING_G, SENS_CATS, SENS_NUMS_U6, delta_sse_bootstrap, lirf_copy_regime, sensor_blend  # noqa: E402
from prc_challenge.unmatched_recovery.unm2026 import CHANNEL_CEILING, adsb_channels, cap_channel, observable, physical_estimate, umap2_expert, umap2_rows  # noqa: E402

MAIN = U.MAIN
EXP = MAIN / "artifacts" / "experiments"
OUT = U.OUT / "v4"
TRACKED = U.TRACKED
RAW = MAIN / "data" / "raw"
S4 = MAIN / S4_FEATURES_REL
NONCOPY_R = EXP / "Unmatched-noncopy-20260925" / "ranking"
POLISH_R = EXP / "Unmatched-polish-20260925" / "ranking"
SENSOR_R = EXP / "Unmatched-sensor-20260926" / "ranking"
MODELS = {
    "expert": POLISH_R / "models" / "expert_noncopy_large_m1_10_octfix.cbm",
    "pcopy": NONCOPY_R / "models" / "pcopy_R-A3b-nodest.pkl",
    "sensor": SENSOR_R / "models" / "sens_m1_10.cbm",
    "t_gate": POLISH_R / "models" / "p1_g_stand_m1_10.cbm",
}
V4_MONTHS = (11, 12)
AMEND2 = TRACKED / "preregistration_amendment_2_nf_v4.json"
W_NF = 0.85
LABEL_COLS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "ARVT_3_flt", "LOBT_flt")
_log, _sha, _write_json = U._log, U._sha, U._write_json


def _guard() -> FinalFreezeGuard:
    # D-072 opened V4 for evaluation; the guard is only used to allow the months 11-12 feature loads.
    return FinalFreezeGuard(tuple(range(1, 13)), final_freeze=True)


def _raw_nd(phase: str | None = None) -> pl.DataFrame:
    files = [next(RAW.glob(f"training_2025-{m:02d}-01_*.parquet")) for m in V4_MONTHS]
    lf = pl.concat([pl.scan_parquet(f) for f in files], how="vertical_relaxed")
    if phase:
        lf = lf.filter(pl.col("PHASE_mvt") == phase)
    return lf.collect()


def _lineage_p(dep: pl.DataFrame, arr: pl.DataFrame, months: tuple[int, ...]) -> pl.DataFrame:
    """P(copy) of the deployed months 1-10 nodest model on month-isolated lineage (DEP labels must be masked)."""

    if dep["BLOCK_TIME_UTC_mvt"].null_count() != dep.height or dep["TAXITIME_SEC_mvt"].null_count() != dep.height:
        raise RuntimeError("DEP labels not masked")
    lin = month_isolated_lineage(dep, arr, months).filter(pl.col("nm_unmatched") == 1)
    lin = lin.select([c for c in NC.LINEAGE_KEEP if c in lin.columns])
    lin_fit = pl.read_parquet(NONCOPY_R / "frames" / "lineage_fit_m1_10.parquet").sort("MVT_ID_mvt")
    if set(lin_fit["month"].unique().to_list()) - set(range(1, 11)):
        raise RuntimeError("lineage fit frame has months outside 1-10")
    hold = with_family_features(lin, lin_fit, n0=20.0, exclude_self=False)
    with MODELS["pcopy"].open("rb") as fh:
        m = pickle.load(fh)
    proba = predict_source_proba(m["model"], m["encoder"], hold, m["cats"], m["nums"], m["scaler"])
    return hold.select("MVT_ID_mvt").with_columns(pl.Series("p", proba[:, CLASSES.index(COPY_CLASS)]))


def _masked(frame: pl.DataFrame) -> pl.DataFrame:
    out = mask_departure_labels(frame, extra_cols=("ARVT_3_flt",))
    return out.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"), reporting_airport_expr())


def _t_frame(base: pl.DataFrame, dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    base = base.filter(pl.col("AIRPORT") == "LIRF")
    bad = [c for c in LABEL_COLS if c in base.columns]
    if bad:
        raise RuntimeError(f"base carries {bad}")
    dep = dep.filter(pl.col("AIRPORT") == "LIRF").drop([c for c in LABEL_COLS if c in dep.columns])
    arr = arr.filter(pl.col("AIRPORT") == "LIRF").drop([c for c in LABEL_COLS if c in arr.columns and c != "BLOCK_TIME_UTC_mvt"])
    raw = L.lirf_raw_features(dep, arr)
    frame = base.join(raw, on="MVT_ID_mvt", how="left")
    frame = frame.with_columns(pl.col("RUNWAY_mvt").is_null().cast(pl.Int8).alias("runway_missing"), pl.col("STAND_mvt").is_null().cast(pl.Int8).alias("stand_missing"))
    return frame.filter(pl.col("nm_unmatched") == 1)


def _t(frame: pl.DataFrame) -> pl.DataFrame:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(str(MODELS["t_gate"]))
    nums = L.U3_VARIANTS["stand"]
    g_hat = np.asarray(model.predict(L._pool(frame, list(L.U6_CATS), nums)), dtype=np.float64)
    t = serve_gate_delay(frame["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy(), g_hat)
    return frame.select("MVT_ID_mvt").with_columns(pl.Series("T", t))


def _sensor(u6: pl.DataFrame, block: pl.DataFrame) -> pl.DataFrame:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(str(MODELS["sensor"]))
    f = u6.select("MVT_ID_mvt", *SENS_CATS, *SENS_NUMS_U6).join(block.select("MVT_ID_mvt", *U.ADSB_NUMS), on="MVT_ID_mvt", how="left", maintain_order="left")
    s = predict_regressor(model, f, list(SENS_CATS), [*SENS_NUMS_U6, *U.ADSB_NUMS])
    cov = f["adsb_coverage"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    return f.select("MVT_ID_mvt").with_columns(pl.Series("S", s), pl.Series("adsb_coverage", cov))


NC = _load("build_unmatched_noncopy_ranking", ROOT / "scripts" / "compose" / "build_unmatched_noncopy_ranking.py")


def phase_parity() -> None:
    """Re-derive the ranking p, T and S with this script's code paths; they must equal the persisted control."""

    ctl = U._ctl("ranking")
    rk = _masked(pl.read_parquet(RAW / "ranking.parquet"))
    dep, arr = rk.filter(pl.col("PHASE_mvt") == "DEP"), rk.filter(pl.col("PHASE_mvt") == "ARR")
    res: dict[str, Any] = {}
    p = ctl.select("MVT_ID_mvt").join(_lineage_p(dep, arr, (1, 7)), on="MVT_ID_mvt", how="left", maintain_order="left")["p"].to_numpy()
    res["p_copy_max_abs_diff"] = float(np.max(np.abs(p - ctl["p_copy"].to_numpy())))
    base = pl.concat([pl.read_parquet(S4 / "base" / f"base_{x}.parquet") for x in ("2026-01", "2026-07")], how="vertical_relaxed")
    tf = _t(_t_frame(base, dep, arr))
    t = ctl.select("MVT_ID_mvt").join(tf, on="MVT_ID_mvt", how="left", maintain_order="left")["T"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    ct = ctl["T"].to_numpy()
    lirf = ctl["AIRPORT"].to_numpy() == "LIRF"
    res["T_max_abs_diff_LIRF"] = float(np.nanmax(np.abs(t[lirf] - ct[lirf])))
    res["T_nan_mismatch_LIRF"] = int((np.isnan(t[lirf]) != np.isnan(ct[lirf])).sum())
    u6 = pl.read_parquet(NONCOPY_R / "frames" / "ranking_unmatched.parquet")
    sv = _sensor(ctl.select("MVT_ID_mvt").join(u6, on="MVT_ID_mvt", how="left", maintain_order="left"), pl.read_parquet(U.ADSB / "adsb_block_ranking.parquet"))
    res["S_max_abs_diff"] = float(np.max(np.abs(sv["S"].to_numpy() - ctl["S"].to_numpy())))
    res["ok"] = res["p_copy_max_abs_diff"] < 1e-6 and res["T_max_abs_diff_LIRF"] < 1e-6 and res["T_nan_mismatch_LIRF"] == 0 and res["S_max_abs_diff"] < 1e-6
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "parity_ranking.json", res)
    _write_json(TRACKED / "v4_parity_ranking.json", res)
    _log(f"parity: {res}")


def phase_build() -> None:
    """Nov/Dec unmatched components and the ALL + C0 control (label-free except nothing: labels are read in score)."""

    blk_path = OUT / "adsb_block_v4.parquet"
    meta = json.loads((OUT / "adsb_block_v4.meta.json").read_text())
    if not meta["parity_ok"]:
        raise SystemExit("v4 ADS-B block parity failed")
    guard = _guard()
    w = working_rows_from_base(S4, RAW, V4_MONTHS, guard, with_labels=False)
    if "TAXITIME_SEC_mvt" in w.columns:
        raise RuntimeError("labels in the working rows")
    w = w.sort("MVT_ID_mvt")
    feats = json.loads((NONCOPY_R / "frames" / "expert_features.json").read_text())
    from catboost import CatBoostRegressor

    em = CatBoostRegressor()
    em.load_model(str(MODELS["expert"]))
    e = predict_regressor(em, w, feats["cats"], feats["nums"])
    sd = w["takeoff_minus_sched_sec"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    e_cap, _ = cap_expert(e, sd)
    raw = _raw_nd()
    rawm = _masked(raw)
    dep, arr = rawm.filter(pl.col("PHASE_mvt") == "DEP"), rawm.filter(pl.col("PHASE_mvt") == "ARR")
    pc = _lineage_p(dep, arr, V4_MONTHS)
    base = pl.concat([pl.read_parquet(S4 / "base" / f"base_2025-{m:02d}.parquet") for m in V4_MONTHS], how="vertical_relaxed")
    tf = _t(_t_frame(base.drop([c for c in LABEL_COLS if c in base.columns]), dep, arr))
    sv = _sensor(w, pl.read_parquet(blk_path))
    x = w.select("MVT_ID_mvt", "month", "AIRPORT", "takeoff_minus_sched_sec", "takeoff_minus_sched_days", "type_missing", "MVT_TIME_UTC_mvt").with_columns(pl.Series("e_raw", e), pl.Series("e_cap", e_cap))
    x = x.join(pc, on="MVT_ID_mvt", how="left", maintain_order="left").rename({"p": "p_copy"})
    x = x.join(sv, on="MVT_ID_mvt", how="left", maintain_order="left").join(tf, on="MVT_ID_mvt", how="left", maintain_order="left")
    if x["p_copy"].null_count() or x["S"].null_count():
        raise RuntimeError(f"missing p ({x['p_copy'].null_count()}) or S ({x['S'].null_count()})")
    # links (arm C0), label-free
    link_raw = raw.select([c for c in dict.fromkeys((*REC.LINK_DEP_COLS, *REC.LINK_ARR_COLS)) if c in raw.columns])
    links, link_counts = destination_links(link_raw)
    # arm B: months 1-10 rate map, Nov/Dec LIRF ARR BLOCK==SCHED rate (ARR rows only)
    manifest = json.loads((REC.OUT / "control" / "manifest.json").read_text())
    data = U.load_all()
    prm = data["prm"]
    arr_nd = arr_eq_rates(raw.filter(pl.col("PHASE_mvt") == "ARR").select("PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"))
    a10, b10 = prm["B"]["ranking_fit_months_1_10"]["a"], prm["B"]["ranking_fit_months_1_10"]["b"]
    c = Components(
        mvt_id=x["MVT_ID_mvt"].to_numpy(),
        month=x["month"].cast(pl.Int64).to_numpy(),
        airport=x["AIRPORT"].cast(pl.String).to_numpy().astype(str),
        sd=x["takeoff_minus_sched_sec"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        days=x["takeoff_minus_sched_days"].cast(pl.Float64).fill_null(0).cast(pl.Int64).to_numpy(),
        type_missing=x["type_missing"].cast(pl.Int64).to_numpy(),
        p=x["p_copy"].to_numpy(),
        e_cap=x["e_cap"].to_numpy(),
        s=x["S"].to_numpy(),
        cov=x["adsb_coverage"].to_numpy(),
        t=x["T"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
    )
    c = attach_links(c, links)
    targets = {m: rate_target(a10, b10, arr_nd[(2025, m)]["rate"]) for m in V4_MONTHS}
    shift, info_b = month_logit_shift(c.p, c.lirf & c.gated, c.month, targets)
    spec = ArmSpec(a_q=prm["A"]["q_ranking"], b_logit_shift=shift, d=True, c0=True)
    ctl = compose(c, spec)
    e_prime, fired = U.sensor_stage(c, spec)
    if float(np.max(np.abs(U.compose_with_expert(c, spec, e_prime) - ctl))) > 1e-6:
        raise RuntimeError("expert-override composition does not reproduce ALL + C0 on Nov/Dec")
    ch = adsb_channels(x.select("MVT_ID_mvt"), pl.read_parquet(blk_path), pl.read_parquet(U.V3 / "adsb_v3lean_block_allyear.parquet"))
    frame = x.with_columns(
        pl.Series("all_c0", ctl), pl.Series("e_prime", e_prime), pl.Series("fired", fired), pl.Series("in_window", c.in_window), pl.Series("lirf_copy_regime", lirf_copy_regime(c.airport, c.sd, c.p)),
        pl.Series("e_d", U.expert_ceiling_d(c.e_cap, c.sd, c.airport)), pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"),
    )
    frame = pl.concat([frame, ch.drop("MVT_ID_mvt", "adsb_coverage")], how="horizontal")
    path = OUT / "control_V4.parquet"
    frame.write_parquet(path)
    man = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": frame.height,
        "rows_by_month": {int(m): int((frame["month"] == m).sum()) for m in V4_MONTHS},
        "file": {"path": str(path), "sha256": _sha(path)},
        "models": {k: {"path": str(v), "sha256": _sha(v)} for k, v in MODELS.items()},
        "model_training_months": {
            "expert": "2025 months 1-10 (build_unmatched_cap_ranking.py octfix: expert_fit_m1_10 + October from the base path)",
            "pcopy": "lineage_fit_m1_10 (2025 months 1-10)",
            "sensor": "sensor_train.parquet months 1-10 (manifest: months [1..10], n_train 1,007,926)",
            "t_gate": "LIRF 2025 months 1-10 (build_unmatched_p1_ranking.py FIT_MONTHS; load_movements refuses 11-12)",
            "armA_q": prm["A"]["q_ranking"],
            "armB_rate_map_months_1_10": {"a": a10, "b": b10, "nov_dec_arr_eq": {m: arr_nd[(2025, m)] for m in V4_MONTHS}, "targets": targets, "shift_info": info_b},
            "adsb_v2_block": "build_blocks(D_MONTHS): centroids/biases from D-month labels only (adsb_block_v4.meta.json, parity with adsb_block_ranking)",
            "lean_map": "adsb_v3lean_block_allyear: vote pool 2025 months 1-10; Nov/Dec rows use the whole pool",
        },
        "links": link_counts,
        "fired": int(fired.sum()),
        "coverage_ge2": int((np.nan_to_num(c.cov, nan=-1) >= 2).sum()),
        "exposure": {k: int((frame["exposure"] == k).sum()) for k in U.EXPOSURE_CLASSES},
        "labels_read": False,
    }
    _write_json(OUT / "control_V4.manifest.json", man)
    _write_json(TRACKED / "v4_control_manifest.json", man)
    _log(f"build: {frame.height} Nov/Dec unmatched rows, fired {int(fired.sum())}, exposure {man['exposure']}")


def phase_score() -> None:
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", str(AMEND2)], capture_output=True, text=True).stdout.strip()
    if not commit:
        raise SystemExit("amendment 2 is not committed")
    path = OUT / "results_V4_NF.json"
    if path.exists():
        raise SystemExit("Nov/Dec look already taken")
    f = pl.read_parquet(OUT / "control_V4.parquet")
    airport = f["AIRPORT"].to_numpy().astype(str)
    sd = f["takeoff_minus_sched_sec"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    pv, src = physical_estimate(f["taxi_map"].cast(pl.Float64).fill_null(np.nan).to_numpy(), f["taxi_gate"].cast(pl.Float64).fill_null(np.nan).to_numpy(), airport)
    fired = f["fired"].to_numpy()
    lirf = f["lirf_copy_regime"].to_numpy()
    o = observable(fired, pv, lirf)
    rows = umap2_rows(src, f["adsb_gate_conf"].cast(pl.Float64).fill_null(0.0).to_numpy(), lirf) & o & ~fired
    raw_map = np.where(src == 1, pv, np.nan)
    rows &= np.isfinite(raw_map) & (np.nan_to_num(raw_map, nan=-1.0) >= 0.0) & (np.nan_to_num(raw_map, nan=1e9) <= CHANNEL_CEILING)
    e2 = umap2_expert(f["e_prime"].to_numpy(), cap_channel(raw_map, sd), rows, fired, 0.0, W_NF)
    # rebuild Components for the composition (same as build)
    ctl = f["all_c0"].to_numpy()
    comp = _components_from_frame(f)
    cand = U.compose_with_expert(comp["c"], comp["spec"], e2)
    same = float(np.max(np.abs(U.compose_with_expert(comp["c"], comp["spec"], f["e_prime"].to_numpy()) - ctl)))
    if same > 1e-6:
        raise RuntimeError(f"control recomposition differs by {same}")
    changed = np.abs(cand - ctl) > 1e-9
    n_changed = int(changed.sum())
    by_ap_changed = {a: int((changed & (airport == a)).sum()) for a in sorted(set(airport.tolist())) if (changed & (airport == a)).any()}
    by_month_changed = {int(m): int((changed & (f["month"].to_numpy() == m)).sum()) for m in V4_MONTHS}
    out: dict[str, Any] = {"amendment_commit": commit, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n": f.height, "arm_rows": int(rows.sum()), "rows_changed": n_changed, "rows_changed_by_airport": by_ap_changed, "rows_changed_by_month": by_month_changed}
    if n_changed < 50:
        out["decision"] = "UNDECIDABLE"
        _write_json(path, out)
        _write_json(TRACKED / "results_V4_NF.json", out)
        _log(f"UNDECIDABLE: {n_changed} rows changed")
        return
    lab = pl.concat([pl.read_parquet(S4 / "labels" / f"labels_2025-{m:02d}.parquet").select("MVT_ID_mvt", "TAXITIME_SEC_mvt") for m in V4_MONTHS])
    y = f.select("MVT_ID_mvt").join(lab, on="MVT_ID_mvt", how="left", maintain_order="left")["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    if not np.isfinite(y).all():
        raise RuntimeError(f"{int((~np.isfinite(y)).sum())} Nov/Dec labels missing")
    d = (y - cand) ** 2 - (y - ctl) ** 2
    se_c = (y - ctl) ** 2
    boot = delta_sse_bootstrap(y, ctl, cand, f["day"].to_numpy())
    aps = {}
    for a in sorted(set(airport.tolist())):
        m = airport == a
        aps[a] = {"n": int(m.sum()), "rows_changed": int((changed & m).sum()), "ctl_sse": float(se_c[m].sum()), "delta_sse": float(d[m].sum()), "delta_frac": float(d[m].sum() / se_c[m].sum()) if se_c[m].sum() > 0 else 0.0, "ctl_sse_O": float(se_c[m & o].sum()), "delta_frac_O_informational": float(d[m].sum() / se_c[m & o].sum()) if se_c[m & o].sum() > 0 else 0.0}
    worst = max(v["delta_frac"] for v in aps.values())
    gates = {
        "dSSE_lt_0": {"value": float(d.sum()), "pass": bool(d.sum() < 0)},
        "day_boot_p95_lt_0": {"value": boot["p95"], "pass": bool(boot["p95"] < 0)},
        "no_airport_gt_2pct": {"value": worst, "pass": bool(worst <= 0.02)},
    }
    sq = float(np.sum((cand - ctl) ** 2))
    decision = "GO" if all(g["pass"] for g in gates.values()) else "NO-GO"
    out.update({
        "decision": decision,
        "gates": gates,
        "delta_sse": float(d.sum()),
        "ctl_sse": float(se_c.sum()),
        "ctl_rmse_unmatched": float(np.sqrt(se_c.mean())),
        "cand_rmse_unmatched": float(np.sqrt(((y - cand) ** 2).mean())),
        "bootstrap": boot,
        "by_month": {int(m): float(d[f["month"].to_numpy() == m].sum()) for m in V4_MONTHS},
        "airports": aps,
        "sum_sq_change": sq,
        "efficiency": float(d.sum() / sq) if sq else None,
        "rows_improved": int((d < 0).sum()),
        "rows_worse": int((d > 0).sum()),
        "top5_share": float(np.sort(d)[:5].sum() / d.sum()) if d.sum() < 0 else None,
        "changed_rows_rmse": {"ctl": float(np.sqrt(se_c[changed].mean())), "cand": float(np.sqrt(((y - cand) ** 2)[changed].mean())), "median_abs_y_minus_P": float(np.median(np.abs(y - raw_map)[changed])), "median_abs_y_minus_ctl": float(np.median(np.abs(y - ctl)[changed]))},
        "expected_public_at_V4_efficiency": {"S_rank": 207421091.67743593, "dSSE": (float(d.sum() / sq) if sq else 0.0) * 207421091.67743593, "seconds": float(np.sqrt(U.PUB_BASE**2 + (float(d.sum() / sq) if sq else 0.0) * 207421091.67743593 / U.PUB_N) - U.PUB_BASE)},
    })
    _write_json(path, out)
    _write_json(TRACKED / "results_V4_NF.json", out)
    _log(f"V4 U-MAP2V-NF: {decision} dSSE {d.sum():.4e} p95 {boot['p95']:.3e} worst {worst:+.4f}; changed {n_changed} {by_ap_changed}")


def _components_from_frame(f: pl.DataFrame) -> dict[str, Any]:
    """Components + ArmSpec of the stored control (arm B shift and links recomputed exactly as in build)."""

    raw = _raw_nd()
    link_raw = raw.select([c for c in dict.fromkeys((*REC.LINK_DEP_COLS, *REC.LINK_ARR_COLS)) if c in raw.columns])
    links, _ = destination_links(link_raw)
    data = U.load_all()
    prm = data["prm"]
    arr_nd = arr_eq_rates(raw.filter(pl.col("PHASE_mvt") == "ARR").select("PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"))
    a10, b10 = prm["B"]["ranking_fit_months_1_10"]["a"], prm["B"]["ranking_fit_months_1_10"]["b"]
    c = Components(
        mvt_id=f["MVT_ID_mvt"].to_numpy(),
        month=f["month"].cast(pl.Int64).to_numpy(),
        airport=f["AIRPORT"].cast(pl.String).to_numpy().astype(str),
        sd=f["takeoff_minus_sched_sec"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        days=f["takeoff_minus_sched_days"].cast(pl.Float64).fill_null(0).cast(pl.Int64).to_numpy(),
        type_missing=f["type_missing"].cast(pl.Int64).to_numpy(),
        p=f["p_copy"].to_numpy(),
        e_cap=f["e_cap"].to_numpy(),
        s=f["S"].to_numpy(),
        cov=f["adsb_coverage"].to_numpy(),
        t=f["T"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
    )
    c = attach_links(c, links)
    targets = {m: rate_target(a10, b10, arr_nd[(2025, m)]["rate"]) for m in V4_MONTHS}
    shift, _ = month_logit_shift(c.p, c.lirf & c.gated, c.month, targets)
    return {"c": c, "spec": ArmSpec(a_q=prm["A"]["q_ranking"], b_logit_shift=shift, d=True, c0=True)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["parity", "build", "score"])
    a = ap.parse_args()
    {"parity": phase_parity, "build": phase_build, "score": phase_score}[a.phase]()
