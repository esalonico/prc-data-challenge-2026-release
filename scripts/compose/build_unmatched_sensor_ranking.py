# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-UNM3 ranking deliverable: unmatched rows under the selected arm (U-SENS / U-REGIME / U-BOTH). Never uploads.

Refuses unless metrics/score.json says GO and confirm/confirmation.json says CONSISTENT.

Composition on the 5,283 ranking unmatched rows (identical to capb_p1_unmatched_submission.parquet
except the arm's change):
  E_cap   = cap_unmatched_overlay.parquet::expert_capped (October-fixed months 1-10 expert, cap 3606)
  U-REGIME: E_cap <- min(months 1-10 October-fixed refit with rg_* columns (400 trees), max(sd,0)+3606)
  U-SENS:   E' = w*S_cap + (1-w)*E_cap on eligible rows (coverage >= 2, not LIRF copy-regime),
            S trained on all 2025 months 1-10 rows with coverage >= 2 (D + Jan/Jul blocks, pool D months)
  raw = p*max(sd,0) + (1-p)*E' (3600 s gate, v18 P(copy)); LIRF: 0.5*raw + 0.5*T (P1); E17/E18 + clip.
Overlay on v18: matched rows untouched, checksum asserted.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("run_unmatched_sensor", ROOT / "scripts" / "compose" / "run_unmatched_sensor.py")
R = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(R)

from prc_challenge.io import scan_submitting  # noqa: E402
from prc_challenge.unmatched_recovery.harness import RANKING_N, RANKING_UNMATCHED_N, overlay_unmatched_submission, verify_arm_submission  # noqa: E402
from prc_challenge.unmatched_recovery.noncopy_expert import GATE_SEC, distribution, expert_training_mask, mixture_prediction  # noqa: E402
from prc_challenge.unmatched_recovery.polish import LAMBDA, blend_on_mask  # noqa: E402
from prc_challenge.unmatched_recovery.sensor import REGIME_COLS, S_CEILING_G, cap, sensor_blend  # noqa: E402

OUT = R.OUT / "ranking"
INSPECTION = "--inspection" in sys.argv
USENS_G = "--usens-g" in sys.argv  # amendment 1: D gates + public verification (Jan/Jul spent)
INSPECTION_MODEL = R.OUT / "ranking_inspection_not_cleared" / "models" / "sens_m1_10.cbm"
INSPECTION_OVERLAY = R.OUT / "ranking_inspection_not_cleared" / "unmatched_overlay.parquet"
if INSPECTION:
    OUT = R.OUT / "ranking_inspection_not_cleared"
OCT_COLS = ("flight_prefix", "ADES_mvt", "FLIGHT_RULE_mvt", "ades_missing")


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _regime_expert_m1_10(rank: pl.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    from prc_challenge.unmatched_recovery.block_lineage import resolve_raw_dir
    from prc_challenge.unmatched_recovery.final_freeze import S4_FEATURES_REL, FinalFreezeGuard, working_rows_from_base
    from prc_challenge.unmatched_recovery.methods import fit_regressor, predict_regressor

    feats = json.loads((R.NONCOPY_R / "frames" / "expert_features.json").read_text())
    cats, nums = feats["cats"], feats["nums"]
    fit = pl.read_parquet(R.NONCOPY_R / "frames" / "expert_fit_m1_10.parquet").sort("_row")
    october = working_rows_from_base(R.MAIN / S4_FEATURES_REL, resolve_raw_dir(R.MAIN), (10,), FinalFreezeGuard((10,)))
    fixed = fit.join(october.select("MVT_ID_mvt", *[pl.col(c).alias(f"{c}__fix") for c in OCT_COLS]), on="MVT_ID_mvt", how="left", maintain_order="left")
    for c in OCT_COLS:
        fixed = fixed.with_columns(pl.when(pl.col("month") == 10).then(pl.col(f"{c}__fix").cast(fit[c].dtype)).otherwise(pl.col(c)).alias(c)).drop(f"{c}__fix")
    lin = pl.read_parquet(R.NONCOPY_R / "frames" / "lineage_fit_m1_10.parquet").select("MVT_ID_mvt", "source_class")
    work = fixed.join(lin, on="MVT_ID_mvt", how="left", maintain_order="left").join(pl.read_parquet(R.OUT / "frames" / "regime_fit_m1_10.parquet"), on="MVT_ID_mvt", how="left", maintain_order="left")
    mask = expert_training_mask(work["source_class"].to_numpy(), work["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy(), mode="noncopy_large")
    train = work.filter(pl.Series(mask))
    # parity: the same refit without rg_* must reproduce the deployed CAP_B ranking expert
    base = fit_regressor(train, cats, nums, iterations=400, thread_count=2, l2_leaf_reg=16.0)
    parity = predict_regressor(base, rank, cats, nums)
    model = fit_regressor(train, cats, [*nums, *REGIME_COLS], iterations=400, thread_count=2, l2_leaf_reg=16.0)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    path = OUT / "models" / "expert_regime_m1_10_octfix.cbm"
    model.save_model(str(path))
    rr = rank.join(pl.read_parquet(R.OUT / "frames" / "regime_ranking_unmatched.parquet"), on="MVT_ID_mvt", how="left", maintain_order="left")
    return predict_regressor(model, rr, cats, [*nums, *REGIME_COLS]), {"path": str(path), "sha256": R._sha(path), "n_train": train.height, "parity_pred": parity}


def main() -> None:
    from prc_challenge.unmatched_recovery.methods import predict_regressor

    score = json.loads((R.OUT / "metrics" / "score.json").read_text())
    conf = json.loads((R.OUT / "confirm" / "confirmation.json").read_text())
    arm = score.get("selected_arm")
    if USENS_G:
        sg = json.loads((R.OUT / "metrics" / "score_usens_g.json").read_text())
        if sg.get("decision") != "GO":
            raise SystemExit("U-SENS-G did not pass the D gates")
        arm = "U-SENS-G"
    elif not INSPECTION and (score.get("decision") != "GO" or conf.get("reading") != "CONSISTENT" or conf.get("arm") != arm):
        raise SystemExit(f"no deliverable: D {score.get('decision')} / Jan-Jul {conf.get('reading')}")
    R._require_memory("ranking")
    OUT.mkdir(parents=True, exist_ok=True)
    rank = pl.read_parquet(R.NONCOPY_R / "frames" / "ranking_unmatched.parquet").sort("_row")
    capo = pl.read_parquet(R.POLISH / "ranking" / "cap_unmatched_overlay.parquet")
    p1o = pl.read_parquet(R.POLISH / "ranking" / "capb_p1_unmatched_overlay.parquet")
    sd = rank["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    airport = rank["AIRPORT"].to_numpy()
    lirf = airport == "LIRF"
    p = R._aligned(rank, capo, "p_copy")
    e_cap = R._aligned(rank, capo, "expert_capped")
    t = R._aligned(rank, p1o, "T", allow_null=True)
    deployed_file = R._aligned(rank, pl.read_parquet(R.CAPB_P1_SUB), "TAXITIME_SEC_mvt")
    deployed = R._composer(rank.select(list(R.GATE_COLS)))

    def compose(e: np.ndarray) -> np.ndarray:
        return deployed(blend_on_mask(mixture_prediction(p, sd, e, gate_sec=GATE_SEC), t, lirf, LAMBDA))

    ctl = compose(e_cap)
    identity = {"control_vs_capb_p1_submission": float(np.max(np.abs(ctl - deployed_file))), "control_vs_capb_p1_overlay": float(np.max(np.abs(ctl - R._aligned(rank, p1o, "prediction"))))}
    if max(identity.values()) > 1e-6:
        raise RuntimeError(f"control does not reproduce the deployed unmatched rows: {identity}")
    e_new = e_cap
    fit_info: dict[str, Any] = {"arm": arm}
    s = np.full(rank.height, np.nan)
    cov = np.full(rank.height, np.nan)
    elig = np.zeros(rank.height, dtype=bool)
    if arm in ("U-REGIME", "U-BOTH"):
        e_reg, info = _regime_expert_m1_10(rank)
        parity = info.pop("parity_pred")
        identity["regime_parity_refit_vs_capb_expert"] = float(np.max(np.abs(cap(parity, sd) - e_cap)))
        if identity["regime_parity_refit_vs_capb_expert"] > 1e-6:
            raise RuntimeError(f"months 1-10 parity refit does not reproduce the CAP_B expert: {identity}")
        e_new = cap(e_reg, sd)
        fit_info["regime_expert"] = info
    if arm in ("U-SENS", "U-BOTH", "U-SENS-G"):
        cats_s, nums_s = R._sens_features()
        tr = pl.read_parquet(R.OUT / "frames" / "sensor_train.parquet").filter(pl.col("sens_train_ok"))
        if set(tr["month"].unique().to_list()) - set(range(1, 11)):
            raise RuntimeError("sensor training months outside 1-10")
        (OUT / "models").mkdir(parents=True, exist_ok=True)
        mpath = OUT / "models" / "sens_m1_10.cbm"
        if USENS_G and INSPECTION_MODEL.exists():
            from catboost import CatBoostRegressor

            model = CatBoostRegressor()
            model.load_model(str(INSPECTION_MODEL))  # same recipe and training rows; reuse instead of an 18-min refit
            model.save_model(str(mpath))
        else:
            model = R._fit_sensor(tr, cats_s, nums_s)
            model.save_model(str(mpath))
        fr = R._unmatched_sensor_frame(rank, rank, R.ADSB / "adsb_block_ranking.parquet")
        cov = fr["adsb_coverage"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        s = np.where(np.isnan(cov), np.nan, predict_regressor(model, fr, cats_s, nums_s))
        if USENS_G and INSPECTION_OVERLAY.exists():
            identity["sensor_S_vs_inspection_overlay"] = float(np.nanmax(np.abs(s - R._aligned(rank, pl.read_parquet(INSPECTION_OVERLAY), "S", allow_null=True))))
            if identity["sensor_S_vs_inspection_overlay"] > 1e-6:
                raise RuntimeError(f"reloaded sensor does not reproduce the inspection S: {identity}")
        e_new, elig = sensor_blend(e_new, s, sd, cov, airport, p, s_ceiling=S_CEILING_G if USENS_G else None)
        fit_info["sensor_ceiling_sec"] = S_CEILING_G if USENS_G else None
        fit_info["ceiling_binding_rows"] = int(np.sum(elig & (s > S_CEILING_G)))
        fit_info["sensor"] = {"path": str(mpath), "sha256": R._sha(mpath), "n_train": tr.height, "n_train_unmatched": int((tr["nm_unmatched"] == 1).sum()), "months": sorted(tr["month"].unique().to_list())}
    new = compose(e_new)
    change = new - ctl
    changed = np.abs(change) > 1e-9
    overlay = rank.select("MVT_ID_mvt", "month", "AIRPORT", "takeoff_minus_sched_sec").with_columns(
        pl.Series("p_copy", p), pl.Series("e_cap", e_cap), pl.Series("S", s), pl.Series("adsb_coverage", cov), pl.Series("eligible_sens", elig), pl.Series("e_new", e_new),
        pl.Series("T", t), pl.Series("capb_p1", ctl), pl.Series("prediction", new)
    )
    overlay_path = OUT / "unmatched_overlay.parquet"
    overlay.write_parquet(overlay_path)
    v18 = pl.read_parquet(R.V18)
    sub = overlay_unmatched_submission(v18, overlay.select("MVT_ID_mvt", "prediction"))
    template = scan_submitting(R.MAIN / "data" / "raw").collect()
    verify = verify_arm_submission(sub, template=template, parent=v18, unmatched_ids=rank["MVT_ID_mvt"], expected_unmatched_n=RANKING_UNMATCHED_N, expected_n=RANKING_N)
    if not verify["ok"] or verify["matched_checksum"] != R.V14_MATCHED_CHECKSUM:
        raise RuntimeError(f"verification failed: {verify['errors']} / {verify['matched_checksum']}")
    sub_path = OUT / ("unmatched_submission_NOT_CLEARED.parquet" if INSPECTION else "unmatched_submission.parquet")
    sub.write_parquet(sub_path)
    # also: vs the capb_p1 submission, only unmatched rows may differ
    capsub = pl.read_parquet(R.CAPB_P1_SUB)
    both = capsub.join(sub, on="MVT_ID_mvt", suffix="_new")
    diff_ids = both.filter((pl.col("TAXITIME_SEC_mvt") - pl.col("TAXITIME_SEC_mvt_new")).abs() > 1e-9)["MVT_ID_mvt"]
    if not set(diff_ids.to_list()) <= set(rank["MVT_ID_mvt"].to_list()):
        raise RuntimeError("rows outside the unmatched population changed")

    # label-free drift of the new inputs: ranking 2026 vs Jan/Jul 2025 (confirmation) vs D OOF
    jj = pl.read_parquet(R.OUT / "confirm" / "janjul_unmatched_components.parquet")
    d_oof = pl.read_parquet(R.OUT / "oof" / "arms_D.parquet")
    drift: dict[str, Any] = {}
    if arm in ("U-SENS", "U-BOTH", "U-SENS-G"):
        s_cap_r = cap(np.nan_to_num(s), sd)
        drift["eligible_rate"] = {"ranking": float(elig.mean()), "janjul_2025": float((jj["candidate"] != jj["control"]).mean()), "D": float(d_oof["eligible_sens"].mean())}
        drift["S_on_eligible"] = {"ranking": distribution(s[elig]), "janjul_2025": distribution(jj.filter(pl.col("candidate") != pl.col("control"))["S"].to_numpy()), "D": distribution(d_oof.filter(pl.col("eligible_sens"))["S"].to_numpy())}
        drift["S_cap_minus_E_cap_on_eligible_ranking"] = distribution((s_cap_r - e_cap)[elig])
        drift["extreme_S_on_eligible"] = {"ranking": {"n_S_gt_7200": int(np.sum(s[elig] > 7200)), "n_S_gt_14400": int(np.sum(s[elig] > 14400)), "n_change_gt_3600": int(np.sum(np.abs(change[elig]) > 3600)), "max_S": float(np.nanmax(s[elig])) if elig.any() else None},
                                          "janjul_2025": {"n_S_gt_7200": int(np.sum(jj.filter(pl.col("candidate") != pl.col("control"))["S"].to_numpy() > 7200)), "n_S_gt_14400": int(np.sum(jj.filter(pl.col("candidate") != pl.col("control"))["S"].to_numpy() > 14400))},
                                          "D": {"n_S_gt_7200": int(np.sum(d_oof.filter(pl.col("eligible_sens"))["S"].to_numpy() > 7200)), "n_S_gt_14400": int(np.sum(d_oof.filter(pl.col("eligible_sens"))["S"].to_numpy() > 14400))}}
        drift["coverage_by_airport_ranking"] = {str(a): {"n": int((airport == a).sum()), "eligible": int((elig & (airport == a)).sum()), "cov3": int(((cov >= 3) & (airport == a)).sum())} for a in sorted(set(airport.tolist()))}
    if arm in ("U-REGIME", "U-BOTH"):
        rr = pl.read_parquet(R.OUT / "frames" / "regime_ranking_unmatched.parquet")
        rj = pl.read_parquet(R.OUT / "frames" / "regime_janjul_unmatched.parquet")
        rd = pl.read_parquet(R.OUT / "frames" / "regime_D_unmatched.parquet")
        drift["regime"] = {c: {"ranking": distribution(rr[c].to_numpy()), "janjul_2025": distribution(rj[c].to_numpy()), "D": distribution(rd[c].to_numpy())} for c in ("rg_med60", "rg_ratio60", "rg_n60")}
    by_ap = {}
    for a in sorted(set(airport.tolist())):
        m = airport == a
        by_ap[str(a)] = {"n": int(m.sum()), "changed": int((changed & m).sum()), "mean_change": float(change[m].mean()), "mean_abs_change": float(np.abs(change[m]).mean()), "max_abs_change": float(np.abs(change[m]).max()),
                         "mean_old": float(ctl[m].mean()), "mean_new": float(new[m].mean())}
    # EHAM January 3-6 2026 (label-free)
    rk_t = R._mvt_times(rank["MVT_ID_mvt"], ranking=True)
    day = rk_t["MVT_TIME_UTC_mvt"].dt.date().to_numpy()
    eham = {}
    pool = R.matched_departures(R.RAW / "ranking.parquet")
    for d in (dt.date(2026, 1, 3), dt.date(2026, 1, 4), dt.date(2026, 1, 5), dt.date(2026, 1, 6)):
        m = (airport == "EHAM") & (day == d)
        mx = pool.filter((pl.col("AIRPORT") == "EHAM") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == d))["x"]
        eham[str(d)] = {"n_unmatched": int(m.sum()), "changed": int((changed & m).sum()), "eligible_sens": int((elig & m).sum()),
                        "median_old": float(np.median(ctl[m])) if m.any() else None, "median_new": float(np.median(new[m])) if m.any() else None,
                        "mean_old": float(ctl[m].mean()) if m.any() else None, "mean_new": float(new[m].mean()) if m.any() else None,
                        "matched_median_mvt_minus_aobt": float(mx.median()) if mx.len() else None}
    manifest = {
        "name": f"SA-UNM3 {arm} unmatched rows on the CAP_B + P1 route",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {"submission": {"path": str(sub_path), "sha256": R._sha(sub_path), "rows": sub.height, "use": "--unmatched input of scripts/compose/compose_candidate.py (unmatched rows already E17/E18 + clip composed)"},
                  "overlay": {"path": str(overlay_path), "sha256": R._sha(overlay_path), "rows": overlay.height}},
        "inputs": {"capb_p1_unmatched_submission": {"path": str(R.CAPB_P1_SUB), "sha256": R._sha(R.CAPB_P1_SUB)}, "v18": {"path": str(R.V18), "sha256": R._sha(R.V18)},
                   "adsb_block_ranking": {"path": str(R.ADSB / "adsb_block_ranking.parquet"), "sha256": R._sha(R.ADSB / "adsb_block_ranking.parquet")}},
        "fit": fit_info,
        "identity_checks_max_abs_diff": identity,
        "rows": {"unmatched": rank.height, "changed_vs_capb_p1": int(changed.sum()), "eligible_sens": int(elig.sum())},
        "change_vs_capb_p1": {"mean": float(change.mean()), "mean_abs": float(np.abs(change).mean()), "max_abs": float(np.abs(change).max()), "mean_on_changed": float(change[changed].mean()) if changed.any() else None, "by_airport": by_ap},
        "drift_label_free": drift,
        "eham_jan_3_6_2026_label_free": eham,
        "validator": {"ok": verify["ok"], "errors": verify["errors"], "matched_checksum": verify["matched_checksum"], "matched_checksum_equals_v14": verify["matched_checksum"] == R.V14_MATCHED_CHECKSUM, "matched_changed": verify["matched_changed"]},
        "eligibility": "adsb.lol ODbL 1.0 (Contains data from ADSB.lol (https://adsb.lol), licensed under ODbL 1.0); AOBT_3 of other flights (D-017); no OpenSky, no Nov/Dec labels, no DEP BLOCK/TAXITIME features, no destination-arrival row",
        "uploaded": False,
        "cleared": not INSPECTION,
        "status": "INSPECTION ONLY - NOT CLEARED: U-SENS passed the D gates but the pre-registered Jan/Jul reading is FLAG" if INSPECTION else ("CLEARED U-SENS-G (amendment 1): D gates GO; Jan/Jul spent; evidence = D gates + public verification" if USENS_G else "GO + CONSISTENT"),
    }
    R._write_json(OUT / (sub_path.stem + ".manifest.json"), manifest)
    R._write_json(R.TRACKED / ("ranking_inspection_manifest.json" if INSPECTION else "ranking_manifest.json"), manifest)
    _log(f"{arm}: changed {int(changed.sum())} of {rank.height}; mean {change.mean():+.1f} s, max |d| {np.abs(change).max():.0f} s; sha {manifest['files']['submission']['sha256']}")
    for d, v in eham.items():
        _log(f"EHAM {d}: {v}")


if __name__ == "__main__":
    main()
