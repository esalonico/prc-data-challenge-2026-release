# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Ranking-side SA-TAIL arm: six all-year legs -> equal blend -> accepted tail arm -> matched prediction file.

Output schema = ``scripts/compose/blend_matched_legs.py`` (``MVT_ID_mvt``, ``prediction``; all rows of leg 0,
E17/E18-gated rows kept from leg 0, clip at 0) plus a manifest. U2, the LOBT window and the final
clip stay in ``scripts/compose/compose_candidate.py`` (``--u2-target … --window-mode lobt``).

Components are fitted on ``allyear_full_adsb_final/train_matrix.parquet`` (2025 months 1-12, the
all-year legs' training months; November/December only as training rows) and applied to that run's
``eval_matrix.parquet`` (the two ranking months).

  --arm T1 | T2 | T1+T2 | control
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_k, "3")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "compose"))

import run_matched_tail as R  # noqa: E402
from prc_challenge import matched_tail as mt  # noqa: E402
from prc_challenge import matrix_arms as ma  # noqa: E402

MAIN = R.MAIN
ALLYEAR = R.RUNS / "allyear_full_adsb_final"
LEG_FILES = [
    R.RUNS / "allyear_full_adsb_final" / "ranking_s4_allyear_full_adsb_final.parquet",
    MAIN / "artifacts" / "experiments" / "VM-arms-20260926" / "vm_allyear_lgbm_adsb" / "ranking_lgbm_vm_allyear_lgbm_adsb.parquet",
    R.RUNS / "allyear_full_adsb_final_delta" / "ranking_s4_allyear_full_adsb_final_delta.parquet",
    R.RUNS / "allyear_lgbm_delta_adsb_final" / "ranking_lgbm_allyear_lgbm_delta_adsb_final.parquet",
    R.RUNS / "allyear_lgbm_eobtres_adsb_final" / "ranking_lgbm_allyear_lgbm_eobtres_adsb_final.parquet",
    R.RUNS / "allyear_full_adsb_final_eobtres" / "ranking_s4_allyear_full_adsb_final_eobtres.parquet",
]
OUT = R.OUT / "ranking"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def six_leg_blend() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """``blend_matched_legs.py`` with the six legs at 1/6: returns (leg-0 frame, blend, gated)."""

    base = pl.read_parquet(LEG_FILES[0]).select("MVT_ID_mvt", "nm_unmatched", "gate_e17", "gate_e18", "s4_raw_prediction", "prediction")
    blend = base["s4_raw_prediction"].to_numpy().astype(float) / len(LEG_FILES)
    for path in LEG_FILES[1:]:
        other = pl.read_parquet(path).select("MVT_ID_mvt", pl.col("s4_raw_prediction").alias("r"))
        j = base.select("MVT_ID_mvt").join(other, on="MVT_ID_mvt", how="left", maintain_order="left")
        if j["r"].null_count():
            raise SystemExit(f"{path}: missing rows")
        blend += j["r"].to_numpy().astype(float) / len(LEG_FILES)
    gated = (base["gate_e17"].cast(pl.Int8) + base["gate_e18"].cast(pl.Int8)).to_numpy() > 0
    pred = np.clip(np.where(gated, base["prediction"].to_numpy().astype(float), blend), 0.0, None)
    return base, pred, gated


def fit_t1_lean(threads: int) -> pl.DataFrame:
    """T1 components on the all-year matrix, loading only the rows the recipe trains on.

    Row selection and order are those of ``run_matched_tail._fit_components`` (all long rows, then
    600k negatives from ``negative_sample(seed=0)`` over the full matrix), so the fit is the same
    recipe without holding the 2.06M-row feature block in memory.
    """

    import lightgbm as lgb

    cats, nums = ma.feature_lists_for_run(ALLYEAR)
    path = ALLYEAR / "train_matrix.parquet"
    lab = pl.read_parquet(path, columns=["month", "nm_unmatched", R.s4.LABEL_COLUMN])
    if sorted(lab["month"].unique().to_list()) != list(range(1, 13)) or (lab["nm_unmatched"] != 0).any():
        raise RuntimeError("unexpected all-year matrix months / unmatched rows")
    y = lab[R.s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
    is_long = y > mt.LONG_SEC
    idx, w = mt.negative_sample(is_long)
    rows = np.unique(idx)
    sub = (pl.scan_parquet(path).select([*cats, *nums]).with_row_index("__r").filter(pl.col("__r").is_in(pl.Series(rows.astype(np.uint32)))).collect())
    if sub.height != rows.size:
        raise RuntimeError("row subset load failed")
    # category levels as _fit_components (on the full training matrix, label-free)
    levels = {c: sorted(pl.read_parquet(path, columns=[c])[c].fill_null("__UNK__").cast(pl.String).unique().to_list()) for c in cats}
    Xs = ma.numpy_matrix(sub, cats, nums, levels)
    pos = np.searchsorted(rows, idx)
    cats_idx = list(range(len(cats)))
    clf = lgb.LGBMClassifier(**mt.P_LONG_PARAMS, n_jobs=threads)
    clf.fit(Xs[pos], is_long[idx].astype(int), sample_weight=w, categorical_feature=cats_idx)
    li = np.where(is_long)[0]
    reg = lgb.LGBMRegressor(**mt.MU_LONG_PARAMS, n_jobs=threads)
    reg.fit(Xs[np.searchsorted(rows, li)], np.clip(y[li], 0, mt.MU_LONG_CLIP), categorical_feature=cats_idx)
    (R.OUT / "models").mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(R.OUT / "models" / "allyear_p_long.txt"))
    reg.booster_.save_model(str(R.OUT / "models" / "allyear_mu_long.txt"))
    ev = pl.read_parquet(ALLYEAR / "eval_matrix.parquet", columns=list(dict.fromkeys(["MVT_ID_mvt", "nm_unmatched", *cats, *nums]))).filter(pl.col("nm_unmatched") == 0)
    Xe = ma.numpy_matrix(ev, cats, nums, levels)
    print(f"T1 all-year: {int(is_long.sum())} long rows of {y.size} (share {is_long.mean():.5f}); {idx.size} classifier rows", flush=True)
    return ev.select("MVT_ID_mvt").with_columns(pl.Series("p_long", clf.predict_proba(Xe)[:, 1]), pl.Series("mu_long", reg.predict(Xe)))


def fit_components(arm: str, threads: int) -> pl.DataFrame:
    """T1 (p_long, mu_long) and/or T2 (E_nc; p_stamp) on the all-year matrix, predicted on ranking matched rows."""

    cats, nums = ma.feature_lists_for_run(ALLYEAR)
    need = list(dict.fromkeys([*ma.ID_COLUMNS, "takeoff_minus_aobt_sec", "takeoff_minus_eobt_sec", *cats, *nums]))
    train = pl.read_parquet(ALLYEAR / "train_matrix.parquet", columns=need)
    months = sorted(train["month"].unique().to_list())
    if months != list(range(1, 13)) or (train["nm_unmatched"] != 0).any():
        raise RuntimeError(f"all-year matrix months {months} / unmatched rows present")
    ev = pl.read_parquet(ALLYEAR / "eval_matrix.parquet", columns=[c for c in need if c != R.s4.LABEL_COLUMN]).filter(pl.col("nm_unmatched") == 0)
    if R.s4.LABEL_COLUMN in pl.read_parquet_schema(ALLYEAR / "eval_matrix.parquet") and \
            pl.read_parquet(ALLYEAR / "eval_matrix.parquet", columns=[R.s4.LABEL_COLUMN])[R.s4.LABEL_COLUMN].null_count() == 0:
        raise RuntimeError("ranking eval matrix carries labels")
    levels = ma.category_levels(train, cats)
    Xt = ma.numpy_matrix(train, cats, nums, levels)
    Xe = ma.numpy_matrix(ev, cats, nums, levels)
    keep = list(dict.fromkeys([*ma.ID_COLUMNS, "takeoff_minus_aobt_sec", "takeoff_minus_eobt_sec"]))
    train = train.select(keep)
    ev = ev.select([c for c in keep if c in ev.columns])
    gc.collect()
    which = (("long",) if "T1" in arm else ()) + (("e_nc",) if "T2" in arm else ())
    res = R._fit_components(Xt, train, Xe, ev, list(range(len(cats))), [*cats, *nums], threads, R.OUT / "cache" / "allyear", which=which)
    del Xt, Xe
    gc.collect()
    if "T2" in arm:
        tr = R.t2_frame(ALLYEAR / "train_matrix.parquet").with_columns(pl.lit(True).alias("_fit"))
        rk = R.t2_frame(ALLYEAR / "eval_matrix.parquet").with_columns(pl.lit(False).alias("_fit"))
        rk = rk.with_columns(pl.lit(None, dtype=pl.Float64).alias(R.s4.LABEL_COLUMN))
        both = pl.concat([tr, rk.select(tr.columns)], how="vertical_relaxed")
        fit_mask = both["_fit"].to_numpy()
        model_path = R.OUT / "models" / "allyear_t2clf.cbm"
        p = R.t2_fit_predict(both, fit_mask, ~fit_mask, threads, model_path)
        pst = both.filter(pl.Series(~fit_mask)).select("MVT_ID_mvt").with_columns(pl.Series("p_stamp", p[~fit_mask]))
        res = res.join(pst, on="MVT_ID_mvt", how="left")
    return res


BLEND10 = R.RUNS.parent / "candidates" / "blend10_matched.parquet"
LEG_FILES_10 = LEG_FILES + [
    R.RUNS / "allyear_lgbm_lobtoff_adsb_final" / "ranking_lgbm_allyear_lgbm_lobtoff_adsb_final.parquet",
    R.RUNS / "nn_v14_allyear_mlp_adsb_final" / "ranking_nn_nn_v14_allyear_mlp_adsb_final.parquet",
    R.RUNS / "nn_v15_allyear_mlp_resid_adsb_final" / "ranking_nn_nn_v15_allyear_mlp_resid_adsb_final.parquet",
    R.RUNS / "nn_v16_allyear_mlp_eobtres_adsb_final" / "ranking_nn_nn_v16_allyear_mlp_eobtres_adsb_final.parquet",
]


def t1_on_blend10(threads: int, blend10: Path | None = None, leg_files: list[Path] | None = None, out_name: str = "blend10_tail_T1_matched.parquet",
                  weights: list[float] | None = None, weights_json: Path | None = None) -> None:
    """D-062: the same all-year p_long/mu_long boosters (no refit, a = 0.35) applied on top of blend10_matched.

    ``weights`` (bake, optional): pre-registered global leg weights of the blend (normalised).
    ``weights_json`` (SA-BLEND, D-087): cell x leg weights file used by ``blend_matched_legs.py --weights-json``.
    The re-blend check uses whichever is given (weights_json first). Default: equal weights, unchanged."""

    import lightgbm as lgb

    BLEND10_ = blend10 or BLEND10
    LEG_FILES_10_ = leg_files or LEG_FILES_10
    b10 = pl.read_parquet(BLEND10_).select("MVT_ID_mvt", "prediction")
    # re-blend check: ten legs, equal weights, gated rows from leg 0 (blend_matched_legs.py logic)
    leg0 = pl.read_parquet(LEG_FILES_10_[0]).select("MVT_ID_mvt", "nm_unmatched", "gate_e17", "gate_e18", "s4_raw_prediction", "prediction")
    raw = np.zeros(leg0.height)
    w_norm = None if weights is None else np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
    if w_norm is not None and len(w_norm) != len(LEG_FILES_10_):
        raise SystemExit(f"{len(w_norm)} weights for {len(LEG_FILES_10_)} legs")
    if weights_json is not None:
        from blend_matched_legs import weighted_blend

        raw, _winfo = weighted_blend(leg0, list(LEG_FILES_10_), json.loads(Path(weights_json).read_text()), Path(weights_json).resolve().parent)
    else:
        for i, path in enumerate(LEG_FILES_10_):
            j = leg0.select("MVT_ID_mvt").join(pl.read_parquet(path).select("MVT_ID_mvt", pl.col("s4_raw_prediction").alias("r")), on="MVT_ID_mvt", how="left", maintain_order="left")
            if j["r"].null_count():
                raise SystemExit(f"{path}: missing rows")
            raw += j["r"].to_numpy().astype(float) / len(LEG_FILES_10_) if w_norm is None else j["r"].to_numpy().astype(float) * w_norm[i]
    gated = (leg0["gate_e17"].cast(pl.Int8) + leg0["gate_e18"].cast(pl.Int8)).to_numpy() > 0
    reblend = np.clip(np.where(gated, leg0["prediction"].to_numpy().astype(float), raw), 0.0, None)
    base = leg0.select("MVT_ID_mvt", "nm_unmatched").join(b10, on="MVT_ID_mvt", how="left", maintain_order="left")
    blend = base["prediction"].to_numpy().astype(float)
    reblend_max_abs = float(np.abs(reblend - blend).max())
    if reblend_max_abs > 1e-6:
        raise SystemExit(f"blend10_matched differs from the ten-leg re-blend (max |d| {reblend_max_abs})")
    matched = base["nm_unmatched"].to_numpy() == 0
    apply = matched & ~gated
    cats, nums = ma.feature_lists_for_run(ALLYEAR)
    path = ALLYEAR / "train_matrix.parquet"
    levels = {c: sorted(pl.read_parquet(path, columns=[c])[c].fill_null("__UNK__").cast(pl.String).unique().to_list()) for c in cats}
    ev = pl.read_parquet(ALLYEAR / "eval_matrix.parquet", columns=list(dict.fromkeys(["MVT_ID_mvt", "nm_unmatched", "month", *cats, *nums]))).filter(pl.col("nm_unmatched") == 0)
    Xe = ma.numpy_matrix(ev, cats, nums, levels)
    pm, mm = R.OUT / "models" / "allyear_p_long.txt", R.OUT / "models" / "allyear_mu_long.txt"
    p = lgb.Booster(model_file=str(pm)).predict(Xe, num_threads=threads)
    mu = lgb.Booster(model_file=str(mm)).predict(Xe, num_threads=threads)
    comp = ev.select("MVT_ID_mvt", "month").with_columns(pl.Series("p_long", p), pl.Series("mu_long", mu))
    f = base.select("MVT_ID_mvt").join(comp, on="MVT_ID_mvt", how="left", maintain_order="left")
    if f.filter(pl.Series(apply))["p_long"].null_count():
        raise RuntimeError("p_long missing on matched rows")
    pred = np.clip(mt.t1_correct(blend, f["p_long"].fill_null(0.0).to_numpy(), f["mu_long"].fill_null(0.0).to_numpy(), apply), 0.0, None)
    out_path = OUT / out_name
    base.select("MVT_ID_mvt").with_columns(pl.Series("prediction", pred)).write_parquet(out_path)
    delta = pred - blend
    m = f["month"].to_numpy()
    man = {
        "arm": "T1 (D-062 re-gate on the ten-leg blend)",
        "recipe": "candidates/blend10_matched.parquet (ten-leg equal blend, 44 E17/E18 rows from CatBoost) -> T1 pred + 0.35*p_long*(mu_long - pred) "
                  "on matched non-gated rows with the unchanged all-year p_long/mu_long boosters -> clip 0. Downstream: compose_candidate.py U2 + LOBT window + clip.",
        "base": {"path": str(BLEND10_), "sha256": sha(BLEND10_), "reblend_check_max_abs": reblend_max_abs},
        "legs": [{"path": str(q), "sha256": sha(q)} for q in LEG_FILES_10_],
        **({"weights": w_norm.tolist()} if w_norm is not None else {}),
        **({"weights_json": {"path": str(weights_json), "sha256": sha(Path(weights_json))}} if weights_json is not None else {}),
        "models": {pm.name: sha(pm), mm.name: sha(mm)},
        "gated_rows_untouched": int(gated.sum()),
        "matched_rows": int(matched.sum()),
        "rows_changed_vs_blend10": int((np.abs(delta) > 1e-9).sum()),
        "rows_changed_gt_1s": int((np.abs(delta) > 1).sum()), "rows_changed_gt_60s": int((np.abs(delta) > 60).sum()),
        "rows_changed_gt_300s": int((np.abs(delta) > 300).sum()), "rows_changed_gt_900s": int((np.abs(delta) > 900).sum()),
        "unmatched_rows_changed": int((np.abs(delta[~matched]) > 1e-9).sum()),
        "delta_vs_blend10": {"mean": float(delta[matched].mean()), "mean_abs": float(np.abs(delta[matched]).mean()), "max_abs": float(np.abs(delta).max())},
        "mean_p_long_by_month": {int(k): float(np.nanmean(f["p_long"].to_numpy()[apply & (m == k)])) for k in (1, 7)},
        "out": str(out_path),
        "out_sha256": sha(out_path),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    R.write_json(R.TRACKED / "ranking" / out_path.with_suffix(".manifest.json").name, man)
    print(json.dumps({k: man[k] for k in ("rows_changed_vs_blend10", "rows_changed_gt_60s", "delta_vs_blend10", "mean_p_long_by_month", "out_sha256", "base")}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("control", "T1", "T2", "T1+T2"))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--on-blend10", action="store_true", help="D-062: apply the stored all-year T1 models on top of candidates/blend10_matched.parquet")
    ap.add_argument("--nn-s8", action="store_true", help="D-067: use the 8-seed neural legs and candidates/blend10s8_matched.parquet")
    args = ap.parse_args()
    if args.on_blend10:
        if args.arm != "T1":
            raise SystemExit("--on-blend10 is defined for T1 only")
        OUT.mkdir(parents=True, exist_ok=True)
        if args.nn_s8:
            legs = [p if "nn_v1" not in str(p) else p.parent.with_name(p.parent.name + "_s8") / (p.stem + "_s8.parquet") for p in LEG_FILES_10]
            t1_on_blend10(args.threads, BLEND10.with_name("blend10s8_matched.parquet"), legs, "blend10s8_tail_T1_matched.parquet")
        else:
            t1_on_blend10(args.threads)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base, blend, gated = six_leg_blend()
    matched = base["nm_unmatched"].to_numpy() == 0
    apply = matched & ~gated
    pred = blend.copy()
    comp_summary = {}
    if args.arm != "control":
        comp = fit_t1_lean(args.threads) if args.arm == "T1" else fit_components(args.arm, args.threads)
        cols = ["MVT_ID_mvt", "AIRPORT", "takeoff_minus_sched_sec", "takeoff_minus_iobt_sec", "takeoff_minus_eobt_sec", "takeoff_minus_lobt_sec", "month"]
        ev = pl.read_parquet(ALLYEAR / "eval_matrix.parquet", columns=cols)
        f = base.select("MVT_ID_mvt").join(ev, on="MVT_ID_mvt", how="left", maintain_order="left").join(comp, on="MVT_ID_mvt", how="left", maintain_order="left")
        for c in comp.columns[1:]:
            if f.filter(pl.Series(apply))[c].null_count():
                raise RuntimeError(f"component {c} missing on matched rows")
        if "T1" in args.arm:
            pred = mt.t1_correct(pred, f["p_long"].fill_null(0.0).to_numpy(), f["mu_long"].fill_null(0.0).to_numpy(), apply)
        if "T2" in args.arm:
            sd = f["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
            feas = mt.stamp_feasible(sd, f["takeoff_minus_iobt_sec"].cast(pl.Float64).to_numpy(), f["takeoff_minus_eobt_sec"].cast(pl.Float64).to_numpy(),
                                     f["takeoff_minus_lobt_sec"].cast(pl.Float64).to_numpy())
            pop = apply & (f["AIRPORT"].to_numpy() != "LIRF")
            pred = mt.t2_mixture(f["p_stamp"].fill_null(0.0).to_numpy(), sd, f["e_nc"].fill_null(float("nan")).to_numpy(), pred, feas, pop)
        m = f["month"].to_numpy()
        for c in comp.columns[1:]:
            v = f[c].to_numpy()
            comp_summary[c] = {int(mm): float(np.nanmean(v[apply & (m == mm)])) for mm in (1, 7)}
    pred = np.clip(pred, 0.0, None)
    tag = args.arm.replace("+", "")
    out_path = OUT / f"blend6_tail_{tag}_matched.parquet"
    out = base.select("MVT_ID_mvt").with_columns(pl.Series("prediction", pred))
    out.write_parquet(out_path)
    delta = pred - blend
    man = {
        "arm": args.arm,
        "recipe": "six-leg equal blend of s4_raw_prediction (blend_matched_legs.py logic, E17/E18 rows from leg 0) -> "
                  + {"control": "no tail arm", "T1": "T1 M-LONG (a=0.35)", "T2": "T2 stamp mixture outside LIRF (U2-M3 form)",
                     "T1+T2": "T1 then T2"}[args.arm] + " on matched non-gated rows -> clip 0. Downstream: compose_candidate.py U2 + LOBT window + clip.",
        "legs": [{"path": str(p), "weight": 1 / len(LEG_FILES), "sha256": sha(p)} for p in LEG_FILES],
        "component_training": {"matrix": str(ALLYEAR / "train_matrix.parquet"), "months": list(range(1, 13)),
                               "nov_dec_role": "blind training rows only (as the all-year legs)"},
        "models": {p.name: sha(p) for p in sorted((R.OUT / "models").glob("allyear_*")) if (args.arm == "T1") == p.name.endswith(".txt")},
        "gated_rows_kept_from_leg0": int(gated.sum()),
        "matched_rows": int(matched.sum()),
        "rows_changed_vs_six_leg_blend": int((np.abs(delta) > 1e-9).sum()),
        "delta_vs_six_leg_blend": {"mean": float(delta[matched].mean()), "mean_abs": float(np.abs(delta[matched]).mean()), "max_abs": float(np.abs(delta).max())},
        "component_means_by_month": comp_summary,
        "out": str(out_path),
        "out_sha256": sha(out_path),
        "pred_mean": float(pred.mean()),
        "pred_max": float(pred.max()),
        "seconds": round(time.time() - t0, 1),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    R.write_json(R.TRACKED / "ranking" / out_path.with_suffix(".manifest.json").name, man)
    print(json.dumps({k: man[k] for k in ("arm", "rows_changed_vs_six_leg_blend", "delta_vs_six_leg_blend", "out_sha256")}, indent=2))


if __name__ == "__main__":
    main()
