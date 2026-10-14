# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Predict-only serving of the deployed unmatched chain on the ranking file (final phase: any set of months).

System (as the live ranking file of v40): CAP_B + P1 + U-SENS-G + ALL (A, B, D) + C0, then U-MAP2V-NF
(the ADS-B travel-time map on rows U-SENS-G did not fire on, w_fired 0, w_nf 0.85).  This is
``scripts/compose/unm2026_v4.py`` (the blind Nov/Dec application of the same stored models) pointed at the
ranking file:

  non-copy expert  Unmatched-polish-20260925/ranking/models/expert_noncopy_large_m1_10_octfix.cbm on the ranking
                   unmatched rows of ``frozen_campaign.load_ranking_unmatched_features`` (the live ranking path)
  P(copy)          Unmatched-noncopy-20260925/ranking/models/pcopy_R-A3b-nodest.pkl on month-isolated lineage of the
                   ranking file (``unm2026_v4._lineage_p``; family features against lineage_fit_m1_10.parquet)
  U-SENS-G sensor  Unmatched-sensor-20260926/ranking/models/sens_m1_10.cbm on the D-pool ADS-B block of the ranking
                   rows (``build_adsb_features.py features`` -> features/adsb_block_ranking.parquet)
  P1 T (LIRF)      Unmatched-polish-20260925/ranking/models/p1_g_stand_m1_10.cbm on the S4 base caches of the months
  arm A            q_ranking; arm B the months 1-10 rate map (a10, b10) with the month's LIRF ARR BLOCK==SCHED rate
                   of the ranking file's ARR rows; both from the recorded control manifest (prm_public)
  arm C0           destination links from the raw ranking clocks (label-free)
  map              adsb_v3lean_block_allyear (vote pool 2025 months 1-10)

Inputs not in the model hand-off: ``Unmatched-noncopy-20260925/ranking/frames/lineage_fit_m1_10.parquet``
(``build_unmatched_noncopy_ranking.py --phase lineage``), ``.../frames/expert_features.json`` and
``Unmatched-sensor-20260926/state.json`` (the expert's cats / nums and the sensor's ADS-B columns, read off the stored
models' ``feature_names_`` and cat-feature indices).  On the original Jan/Jul ranking file this script reproduces the
live ``unmatched_NF`` predictions to 1e-12 on all 5,283 rows (2026-10-11).

Writes ``<out>/unmatched_serve.parquet`` (component frame + ``prediction``) and ``<out>/unmatched_serve.json``.
``--check PATH`` compares the predictions with a stored submission-shaped file on the unmatched rows.
Months: ``s4_allyear.SERVE_MONTHS`` (PRC_SERVE_MONTHS; default 1,7 = the public ranking file).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

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


V4 = _load("unm2026_v4", ROOT / "scripts" / "compose" / "unm2026_v4.py")
U, REC = V4.U, V4.REC

from prc_challenge import s4_allyear as s4  # noqa: E402
from prc_challenge.unmatched_recovery.frozen_campaign import load_ranking_unmatched_features  # noqa: E402
from prc_challenge.unmatched_recovery.methods import predict_regressor  # noqa: E402
from prc_challenge.unmatched_recovery.polish import cap_expert  # noqa: E402
from prc_challenge.unmatched_recovery.recovery import ArmSpec, Components, arr_eq_rates, attach_links, compose, destination_links, month_logit_shift, rate_target  # noqa: E402
from prc_challenge.unmatched_recovery.sensor import lirf_copy_regime  # noqa: E402
from prc_challenge.unmatched_recovery.unm2026 import CHANNEL_CEILING, adsb_channels, cap_channel, observable, physical_estimate, umap2_expert, umap2_rows  # noqa: E402

CONTROL_MANIFEST = ROOT / "configs" / "final" / "unmatched_control_params.json"
W_NF = V4.W_NF


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--block", default=str(U.ADSB / "adsb_block_ranking.parquet"), help="D-pool ADS-B block of the ranking rows")
    p.add_argument("--lean", default=str(U.V3 / "adsb_v3lean_block_allyear.parquet"))
    p.add_argument("--check", default=None, help="stored submission-shaped file (MVT_ID_mvt, TAXITIME_SEC_mvt) to compare with")
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    months = tuple(int(m) for m in s4.SERVE_MONTHS)
    prm = json.loads(CONTROL_MANIFEST.read_text())["prm_public"]
    q_ranking = float(prm["A"]["q_ranking"])
    a10, b10 = float(prm["B"]["ranking_fit_months_1_10"]["a"]), float(prm["B"]["ranking_fit_months_1_10"]["b"])

    raw = pl.read_parquet(V4.RAW / "ranking.parquet")
    rawm = V4._masked(raw)
    dep, arr = rawm.filter(pl.col("PHASE_mvt") == "DEP"), rawm.filter(pl.col("PHASE_mvt") == "ARR")
    if sorted(dep["month"].unique().to_list()) != sorted(months):
        raise SystemExit(f"ranking months {sorted(dep['month'].unique().to_list())} != serve months {sorted(months)}")

    # non-copy expert (CAP_B) on the live ranking feature path
    camp = V4.NC._camp()
    w = load_ranking_unmatched_features(camp).sort("MVT_ID_mvt")
    if w.height != s4.RANKING_UNMATCHED_N:
        raise RuntimeError(f"unmatched rows {w.height} != {s4.RANKING_UNMATCHED_N}")
    feats = json.loads((V4.NONCOPY_R / "frames" / "expert_features.json").read_text())
    from catboost import CatBoostRegressor

    em = CatBoostRegressor()
    em.load_model(str(V4.MODELS["expert"]))
    e = predict_regressor(em, w, feats["cats"], feats["nums"])
    sd = w["takeoff_minus_sched_sec"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    e_cap, _ = cap_expert(e, sd)
    pc = V4._lineage_p(dep, arr, months)
    base = pl.concat([pl.read_parquet(V4.S4 / "base" / f"base_2026-{m:02d}.parquet") for m in months], how="vertical_relaxed")
    tf = V4._t(V4._t_frame(base.drop([c for c in V4.LABEL_COLS if c in base.columns]), dep, arr))
    blk = pl.read_parquet(a.block)
    sv = V4._sensor(w, blk)
    x = w.select("MVT_ID_mvt", "month", "AIRPORT", "takeoff_minus_sched_sec", "takeoff_minus_sched_days", "type_missing", "MVT_TIME_UTC_mvt" if "MVT_TIME_UTC_mvt" in w.columns else pl.lit(None).alias("MVT_TIME_UTC_mvt")).with_columns(pl.Series("e_raw", e), pl.Series("e_cap", e_cap))
    x = x.join(pc, on="MVT_ID_mvt", how="left", maintain_order="left").rename({"p": "p_copy"})
    x = x.join(sv, on="MVT_ID_mvt", how="left", maintain_order="left").join(tf, on="MVT_ID_mvt", how="left", maintain_order="left")
    if x["p_copy"].null_count() or x["S"].null_count():
        raise RuntimeError(f"missing p ({x['p_copy'].null_count()}) or S ({x['S'].null_count()})")
    # arm C0 links and arm B targets (label-free: raw clocks, ARR rows)
    link_raw = raw.select([c for c in dict.fromkeys((*REC.LINK_DEP_COLS, *REC.LINK_ARR_COLS)) if c in raw.columns])
    links, link_counts = destination_links(link_raw)
    arr_rk = arr_eq_rates(raw.filter(pl.col("PHASE_mvt") == "ARR").select("PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"))
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
    targets = {m: rate_target(a10, b10, arr_rk[(2026, m)]["rate"]) for m in months}
    shift, info_b = month_logit_shift(c.p, c.lirf & c.gated, c.month, targets)
    spec = ArmSpec(a_q=q_ranking, b_logit_shift=shift, d=True, c0=True)
    ctl = compose(c, spec)
    e_prime, fired = U.sensor_stage(c, spec)
    if float(np.max(np.abs(U.compose_with_expert(c, spec, e_prime) - ctl))) > 1e-6:
        raise RuntimeError("expert-override composition does not reproduce ALL + C0")
    ch = adsb_channels(x.select("MVT_ID_mvt"), blk, pl.read_parquet(a.lean))
    lirf = lirf_copy_regime(c.airport, c.sd, c.p)
    # U-MAP2V-NF (unm2026_v4.phase_score, without labels)
    pv, src = physical_estimate(ch["taxi_map"].cast(pl.Float64).fill_null(np.nan).to_numpy(), ch["taxi_gate"].cast(pl.Float64).fill_null(np.nan).to_numpy(), c.airport)
    o = observable(fired, pv, lirf)
    rows = umap2_rows(src, ch["adsb_gate_conf"].cast(pl.Float64).fill_null(0.0).to_numpy(), lirf) & o & ~fired
    raw_map = np.where(src == 1, pv, np.nan)
    rows &= np.isfinite(raw_map) & (np.nan_to_num(raw_map, nan=-1.0) >= 0.0) & (np.nan_to_num(raw_map, nan=1e9) <= CHANNEL_CEILING)
    e2 = umap2_expert(e_prime, cap_channel(raw_map, c.sd), rows, fired, 0.0, W_NF)
    pred = U.compose_with_expert(c, spec, e2)
    if np.any((np.abs(pred - ctl) > 1e-9) & fired):
        raise RuntimeError("fired rows changed")
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise RuntimeError("non-finite or negative unmatched prediction")
    frame = x.with_columns(pl.Series("all_c0", ctl), pl.Series("e_prime", e_prime), pl.Series("fired", fired),
                           pl.Series("lirf_copy_regime", lirf), pl.Series("nf_rows", rows), pl.Series("prediction", pred))
    path = out / "unmatched_serve.parquet"
    frame.write_parquet(path)
    rep = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "months": list(months), "rows": frame.height,
           "rows_by_month": {int(m): int((frame["month"] == m).sum()) for m in months},
           "rows_by_airport": {k: int(v) for k, v in frame.group_by("AIRPORT").len().iter_rows()},
           "arm_A_q": q_ranking, "arm_B_rate_map": {"a": a10, "b": b10, "arr_eq": {int(m): arr_rk[(2026, m)] for m in months}, "targets": targets, "shift_info": info_b},
           "links": link_counts, "fired": int(fired.sum()), "nf_rows": int(rows.sum()),
           "file": {"path": str(path), "sha256": V4._sha(path)}, "models": {k: {"path": str(v), "sha256": V4._sha(v)} for k, v in V4.MODELS.items()},
           "block": {"path": a.block, "sha256": V4._sha(Path(a.block))}, "lean": {"path": a.lean, "sha256": V4._sha(Path(a.lean))},
           "seconds": round(time.time() - t0, 1), "labels_read": False}
    if a.check:
        st = pl.read_parquet(a.check).select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").alias("stored"))
        j = frame.select("MVT_ID_mvt", "prediction", "AIRPORT").join(st, on="MVT_ID_mvt", how="inner", validate="1:1")
        d = np.abs(j["prediction"].to_numpy() - j["stored"].to_numpy())
        rep["check"] = {"stored": a.check, "n": j.height, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                        "n_gt_1e-6": int((d > 1e-6).sum()), "n_gt_1s": int((d > 1).sum()),
                        "by_airport_n_gt_1s": {k: int(v) for k, v in j.filter(pl.Series(d > 1)).group_by("AIRPORT").len().iter_rows()}}
    V4._write_json(out / "unmatched_serve.json", rep)
    V4._log(f"unmatched serve: {frame.height} rows, fired {int(fired.sum())}, NF rows {int(rows.sum())}; check {rep.get('check')}")


if __name__ == "__main__":
    main()
