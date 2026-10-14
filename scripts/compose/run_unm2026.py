# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-UNM5 (2026-09-27): 2026-focused unmatched work on top of ALL + C0.

Phases:
  control   rebuild ALL + C0 on D (OOF), JAN_JUL 2025 and ranking from the persisted SA-UNM4 control;
            the ranking recomposition must equal unmatched_submission_ALL_C0.parquet (sha 7227baaa...).
            Writes artifacts/experiments/Unmatched-2026-20260927/control/*.parquet (+ ADS-B channels).
  exposure  label-free step 1: ADS-B exposure tables by airport x month for ranking, D and JAN_JUL 2025.
  (later phases: fit / score / ranking, added with the pre-registration)

Boundaries: no DEP BLOCK/TAXITIME as a feature, no Nov/Dec 2025 labels, no uploads, CPU <= 4 threads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_key, "4")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("run_unmatched_recovery", ROOT / "scripts" / "compose" / "run_unmatched_recovery.py")
REC = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(REC)

from prc_challenge.unmatched_recovery.sensor import lirf_copy_regime  # noqa: E402
from prc_challenge.unmatched_recovery.unm2026 import (  # noqa: E402
    EXPOSURE_CLASSES,
    adsb_channels,
    compose_with_expert,
    exposure_table,
    far_from_adsb,
    sensor_stage,
)

MAIN = REC.MAIN
EXP = MAIN / "artifacts" / "experiments"
OUT = EXP / "Unmatched-2026-20260927"
TRACKED = ROOT / "progress-review" / "unmatched-2026-20260927"
ADSB = EXP / "ADSB-features-20260925" / "features"
V3 = EXP / "ADSB-coverage-20260926" / "v3"
ALL_C0_FILE = REC.OUT / "ranking" / "unmatched_submission_ALL_C0.parquet"
ALL_C0_SHA = "7227baaad07c20f0fc220b0aae34b4a928cca7d32a688fe447c535a730ea056e"
BASE_ARMS = ("A", "B", "D", "C0")
POPS = {
    "D": {"block": ADSB / "adsb_block_D.parquet", "lean": V3 / "adsb_v3lean_block_dfit.parquet"},
    "JAN_JUL_2025": {"block": ADSB / "adsb_block_janjul.parquet", "lean": V3 / "adsb_v3lean_block_allyear.parquet"},
    "ranking": {"block": ADSB / "adsb_block_ranking.parquet", "lean": V3 / "adsb_v3lean_block_allyear.parquet"},
}
LEAN_SHA = {"adsb_v3lean_block_dfit.parquet": "5af992c4ec34c055a1f4cc70549852af4fa2602338e1be9264bc8050872abcbe", "adsb_v3lean_block_allyear.parquet": "e7b81caa02c1c022d4a8bad6d785a521c3d3d4f4e45e1cf80c41b48c513c3334"}

_log, _sha, _write_json, _aligned = REC._log, REC._sha, REC._write_json, REC._aligned


def load_all() -> dict[str, Any]:
    """The three populations with ALL + C0 composed (control) and the U-SENS-G-stage expert."""

    manifest = json.loads((REC.OUT / "control" / "manifest.json").read_text())
    dx, dc = REC._load("D")
    jx, jc = REC._load("JAN_JUL_2025")
    rx, rc = REC._load("ranking")
    prm = REC._params(dx, dc, jx, jc, rc, manifest)
    out: dict[str, Any] = {"prm": prm}
    for name, x, c in (("D", dx, dc), ("JAN_JUL_2025", jx, jc), ("ranking", rx, rc)):
        spec = REC._spec(BASE_ARMS, "JAN_JUL" if name == "JAN_JUL_2025" else name, prm)
        ctl = REC.compose(c, spec)
        e_prime, fired = sensor_stage(c, spec)
        again = compose_with_expert(c, spec, e_prime)
        if float(np.max(np.abs(again - ctl))) > 1e-6:
            raise RuntimeError(f"{name}: expert-override composition does not reproduce ALL + C0")
        out[name] = {"x": x, "c": c, "spec": spec, "ctl": ctl, "e_prime": e_prime, "fired": fired}
    ref = _aligned(rx, pl.read_parquet(ALL_C0_FILE), "TAXITIME_SEC_mvt")
    gap = float(np.max(np.abs(out["ranking"]["ctl"] - ref)))
    if gap > 1e-6:
        raise RuntimeError(f"ranking ALL + C0 recomposition differs from the ALL_C0 file by {gap}")
    out["ranking_identity_max_abs_diff"] = gap
    return out


def channels(name: str, ids: pl.DataFrame) -> pl.DataFrame:
    p = POPS[name]
    return adsb_channels(ids, pl.read_parquet(p["block"]), pl.read_parquet(p["lean"]))


def phase_control() -> None:
    if _sha(ALL_C0_FILE) != ALL_C0_SHA:
        raise RuntimeError("ALL_C0 file sha mismatch")
    for f, s in LEAN_SHA.items():
        if _sha(V3 / f) != s:
            raise RuntimeError(f"{f} sha mismatch")
    t0 = time.time()
    data = load_all()
    (OUT / "control").mkdir(parents=True, exist_ok=True)
    man: dict[str, Any] = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "system": "CAP_B + P1 + U-SENS-G + ALL (A, B, D) + C0; E17/E18 + clip", "files": {}}
    for name in ("D", "JAN_JUL_2025", "ranking"):
        d = data[name]
        x, c = d["x"], d["c"]
        ch = channels(name, x)
        keep = [k for k in ("MVT_ID_mvt", "month", "AIRPORT", "takeoff_minus_sched_sec", "takeoff_minus_sched_days", "type_missing", "p_copy", "e_cap", "S", "adsb_coverage", "T", "TAXITIME_SEC_mvt", "source_class", "MVT_TIME_UTC_mvt", "fold", "day") if k in x.columns]
        frame = x.select(keep).with_columns(
            pl.Series("usens_g_current", x["current"].to_numpy()),
            pl.Series("all_c0", d["ctl"]),
            pl.Series("e_prime", d["e_prime"]),
            pl.Series("fired", d["fired"]),
            pl.Series("in_window", c.in_window),
            pl.Series("lirf_copy_regime", lirf_copy_regime(c.airport, c.sd, c.p)),
        )
        frame = pl.concat([frame, ch.drop("MVT_ID_mvt").drop([k for k in ch.columns if k in frame.columns and k != "MVT_ID_mvt"])], how="horizontal_extend")
        if name == "JAN_JUL_2025" and "day" not in frame.columns:
            frame = frame.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"))
        path = OUT / "control" / f"control_{name}.parquet"
        frame.write_parquet(path)
        man["files"][name] = {"path": str(path), "sha256": _sha(path), "rows": frame.height}
    man["ranking_identity_vs_ALL_C0_file"] = {"max_abs_diff": data["ranking_identity_max_abs_diff"], "file": str(ALL_C0_FILE), "sha256": ALL_C0_SHA}
    man["prm_public"] = {k: v for k, v in data["prm"].items() if k != "_arrays"}
    _write_json(OUT / "control" / "manifest.json", man)
    _write_json(TRACKED / "control_manifest.json", man)
    _log(f"control written in {time.time() - t0:.0f}s: {json.dumps({k: v['rows'] for k, v in man['files'].items()})}")


def _ctl(name: str) -> pl.DataFrame:
    return pl.read_parquet(OUT / "control" / f"control_{name}.parquet")


def phase_exposure() -> None:
    """Step 1: label-free. Never reads TAXITIME_SEC_mvt."""

    TRACKED.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "label_free": True, "definitions": {
        "fired": "U-SENS-G eligible (adsb_coverage >= 2, S finite, not the LIRF copy regime): the sensor blend changed the expert",
        "exposure classes (priority order)": {
            "gate_conf": "confirmed v2 stand gate (adsb_gate_conf == 1)",
            "seg_map": "ADS-B ground segments, no confirmed gate, reliable v3-lean map estimate (adsb3_map_first_sd <= 180 s)",
            "map_only": "reliable v3-lean map estimate but no v2 ground segment (v2 join cleared by the linker precision gate: EHAM take-off-time fallback joins)",
            "seg_only": "ground segments, no confirmed gate, no reliable map",
            "join_no_seg": "aircraft joined, no departure ground segment (coverage 1)",
            "no_adsb": "no ADS-B join (coverage 0 or row absent from the block)",
        },
        "physical": "gate_conf or reliable map (a physical off-block estimate exists)",
        "far_1800": "physical row where the ALL + C0 prediction is > 1800 s from both the confirmed-gate taxi estimate and the reliable-map taxi estimate (whichever exist)",
        "far_1800_incl_first": "same, also counting the bias-corrected first-seen estimate adsb_taxi_hat_first as an estimate",
        "map files": "D: adsb_v3lean_block_dfit (D pool, own month excluded); JAN_JUL 2025 and ranking: adsb_v3lean_block_allyear (months 1-10 pool; own month excluded for 2025 rows). LIRF, LFPG, LTFM masked in the lean map.",
    }}
    tables = {}
    for name in ("ranking", "D", "JAN_JUL_2025"):
        f = _ctl(name)
        if "TAXITIME_SEC_mvt" in f.columns:
            f = f.drop("TAXITIME_SEC_mvt")  # label-free step
        ch = f.select("taxi_gate", "taxi_map", "taxi_first")
        pred = f["all_c0"].to_numpy()
        f = f.with_columns(pl.Series("far", far_from_adsb(pred, ch)), pl.Series("far_first", far_from_adsb(pred, ch, include_first=True)))
        f = f.with_columns((pl.col("all_c0") - pl.coalesce("taxi_map", "taxi_gate")).alias("phys_gap"))
        by_am = exposure_table(f, ["AIRPORT", "month"])
        by_a = exposure_table(f, ["AIRPORT"])
        by_m = exposure_table(f, ["month"])
        tot = exposure_table(f.with_columns(pl.lit("ALL").alias("all")), ["all"])
        tables[name] = {"airport_month": by_am, "airport": by_a, "month": by_m, "total": tot}
        for k, t in tables[name].items():
            t.write_csv(TRACKED / f"exposure_{name}_{k}.csv")
        # label-free change-magnitude diagnostics of the physical rows: |ALL+C0 - estimate|
        phys = f.filter(pl.col("physical"))
        est = phys.select(pl.coalesce("taxi_map", "taxi_gate").alias("est"))["est"].to_numpy()
        gap = phys["all_c0"].to_numpy() - est
        s_gap = phys["S"].to_numpy() - est
        res[name] = {
            "n": f.height,
            "total": tot.to_dicts()[0],
            "class_shares": {k: round(float((f["exposure"] == k).mean()), 4) for k in EXPOSURE_CLASSES},
            "fired_by_class": {k: int((f["fired"] & (f["exposure"] == k)).sum()) for k in EXPOSURE_CLASSES},
            "map_without_seg": int(f["map_without_seg"].sum()),
            "physical_pred_minus_estimate": {"n": int(gap.size), "median": float(np.median(gap)) if gap.size else None, "mean_abs": float(np.mean(np.abs(gap))) if gap.size else None, "p90_abs": float(np.quantile(np.abs(gap), 0.9)) if gap.size else None, "rms": float(np.sqrt(np.mean(gap**2))) if gap.size else None},
            "physical_S_minus_estimate": {"median": float(np.nanmedian(s_gap)) if s_gap.size else None, "mean_abs": float(np.nanmean(np.abs(s_gap))) if s_gap.size else None},
            "physical_by_sd_band": {b: int(v) for b, v in phys.with_columns(pl.col("takeoff_minus_sched_sec").cast(pl.Float64).cut([0, 1800, 3600, 14400], labels=["<0", "0-1800", "1800-3600", "3600-14400", ">=14400"]).cast(pl.String).fill_null("sd_missing").alias("b")).group_by("b").len().iter_rows()},
        }
        _log(f"{name}: n {f.height}, fired {int(f['fired'].sum())}, physical {int(f['physical'].sum())}, far {int(f['far'].sum())}")
    # 2025 -> 2026 shift, per airport: physical and fired shares
    shift = {}
    for a in sorted(set(tables["ranking"]["airport"]["AIRPORT"].to_list()) | set(tables["D"]["airport"]["AIRPORT"].to_list())):
        row = {}
        for name in ("D", "JAN_JUL_2025", "ranking"):
            t = tables[name]["airport"].filter(pl.col("AIRPORT") == a)
            row[name] = {k: (t[k][0] if t.height else None) for k in ("n", "usens_fired", "physical", "physical_not_fired", "gap_gt_600", "gap_gt_900", "sum_sq_gap", "far_1800", "fired_share", "physical_share")}
        shift[a] = row
    res["airport_shift"] = shift
    _write_json(TRACKED / "exposure.json", res)
    _write_md(tables, res)


def _md_table(t: pl.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if i == 0 else "---:" for i, _ in enumerate(cols)) + "|"]
    for r in t.select(cols).iter_rows():
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def _write_md(tables: dict[str, Any], res: dict[str, Any]) -> None:
    cols = ["n", "usens_fired", "gate_conf", "seg_map", "map_only", "seg_only", "join_no_seg", "no_adsb", "map_no_conf_gate", "physical", "physical_not_fired", "gap_gt_600", "gap_gt_900", "sum_sq_gap", "far_1800", "far_1800_incl_first"]
    parts = ["# SA-UNM5 step 1: ADS-B exposure of the unmatched populations (label-free)", "", "Generated by `scripts/compose/run_unm2026.py exposure`. Definitions in `exposure.json`.", ""]
    for name in ("ranking", "D", "JAN_JUL_2025"):
        parts += [f"## {name}", "", "### Total", "", _md_table(tables[name]["total"], ["all", *cols]), "", "### By month", "", _md_table(tables[name]["month"], ["month", *cols]), "", "### By airport", "", _md_table(tables[name]["airport"], ["AIRPORT", *cols]), ""]
        if name == "ranking":
            parts += ["### By airport x month", "", _md_table(tables[name]["airport_month"], ["AIRPORT", "month", *cols]), ""]
    (TRACKED / "exposure_tables.md").write_text("\n".join(parts) + "\n")



# ---------------------------------------------------------------------------
# step 3: frames, fits, scoring (preregistration.json, committed dd7fe10)
# ---------------------------------------------------------------------------

import subprocess  # noqa: E402

from prc_challenge.unmatched_recovery.noncopy_expert import expert_training_mask  # noqa: E402
from prc_challenge.unmatched_recovery.recovery import expert_ceiling_d  # noqa: E402
from prc_challenge.unmatched_recovery.sensor import SENS_CATS, SENS_NUMS_U6  # noqa: E402
from prc_challenge.unmatched_recovery.unm2026 import (  # noqa: E402
    CHANNEL_CEILING,
    LEAN_COLS,
    MAP_SD_MAX,
    RESID_CLIP,
    TAU_GRID,
    W_GRID,
    cap_channel,
    change_cells,
    crossfit_choice,
    lam_from_tau,
    observable,
    physical_estimate,
    physical_expert,
    sens2_rows,
    subset_gates,
    umap2_expert,
    umap2_rows,
)

PREREG = TRACKED / "preregistration.json"
E08B = MAIN / "artifacts" / "cache" / "e08b_e34_months_1_10.parquet"
SENSOR = EXP / "Unmatched-sensor-20260926"
POLISH = EXP / "Unmatched-polish-20260925"
NONCOPY_R = EXP / "Unmatched-noncopy-20260925" / "ranking"
D_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)
BLOCKS = ((2, 8), (3, 9), (4, 10), (5, 6))
FOLDS = ("2_8", "3_9", "4_10", "5_6")
ADSB_NUMS = json.loads((SENSOR / "state.json").read_text())["phases"]["prepare"]["adsb_features"]
R_CATS = list(SENS_CATS)
R_NUMS = [*SENS_NUMS_U6, *ADSB_NUMS, *LEAN_COLS, "P", "p_src"]
R_PARAMS = {"depth": 8, "learning_rate": 0.06, "l2_leaf_reg": 12.0, "iterations": 1000, "thread_count": 4, "loss_function": "RMSEWithUncertainty"}
JJ_BASE = 315.094
JJ_N = 344_419
PUB_BASE = 244.8018
PUB_N = 344_841
KAPPA_CENTRAL = 2.0


def _require_prereg() -> str:
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", str(PREREG)], capture_output=True, text=True).stdout.strip()
    if not commit:
        raise SystemExit("pre-registration is not committed")
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(PREREG)], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("pre-registration has uncommitted changes")
    return commit


def _free_mem_check(ctx: str) -> None:
    REC._require_memory(ctx)


def _lean_path(kind: str) -> Path:
    return V3 / ("adsb_v3lean_block_dfit.parquet" if kind == "dfit" else "adsb_v3lean_block_allyear.parquet")


def _with_p(frame: pl.DataFrame) -> pl.DataFrame:
    """Adds P / p_src (label-free) from the block + lean columns already on ``frame``."""

    m = frame.select(pl.when(pl.col("adsb3_map_first_rel").is_not_null() & (pl.col("adsb3_map_first_sd") <= MAP_SD_MAX)).then(-pl.col("adsb3_map_first_rel")).otherwise(None).cast(pl.Float64).alias("m"))["m"].fill_null(np.nan).to_numpy()
    g = frame.select(pl.when((pl.col("adsb_gate_conf") == 1) & pl.col("adsb_taxi_hat").is_not_null()).then(pl.col("adsb_taxi_hat")).otherwise(None).cast(pl.Float64).alias("g"))["g"].fill_null(np.nan).to_numpy()
    pv, src = physical_estimate(m, g, frame["AIRPORT"].cast(pl.String).to_numpy().astype(str))
    return frame.with_columns(pl.Series("P", pv), pl.Series("p_src", src))


def _source_classes() -> pl.DataFrame:
    return pl.concat([
        pl.read_parquet(POLISH / "frames" / "d_unmatched_octfix_features.parquet", columns=["MVT_ID_mvt", "source_class"]),
        pl.read_parquet(NONCOPY_R / "frames" / "lineage_fit_m1_10.parquet", columns=["MVT_ID_mvt", "month", "source_class"]).filter(pl.col("month").is_in([1, 7])).drop("month"),
    ])


def build_train(months: tuple[int, ...], lean_kind: str) -> pl.DataFrame:
    """Residual-model training rows: departures with P at a physical airport, noncopy_large mask, r = clip(y - P)."""

    e = pl.read_parquet(E08B, columns=["MVT_ID_mvt", "month", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", *SENS_CATS, *SENS_NUMS_U6]).filter(pl.col("month").is_in(list(months)))
    parts = []
    for name, ms in (("adsb_block_D.parquet", D_MONTHS), ("adsb_block_janjul.parquet", (1, 7))):
        use = [m for m in ms if m in months]
        if not use:
            continue
        b = pl.read_parquet(ADSB / name, columns=["MVT_ID_mvt", *ADSB_NUMS])
        parts.append(e.filter(pl.col("month").is_in(use)).join(b, on="MVT_ID_mvt", how="inner"))
    x = pl.concat(parts).join(pl.read_parquet(_lean_path(lean_kind), columns=["MVT_ID_mvt", *LEAN_COLS]), on="MVT_ID_mvt", how="left")
    x = _with_p(x).filter(pl.col("P").is_finite())
    x = x.join(_source_classes(), on="MVT_ID_mvt", how="left")
    miss = x.filter((pl.col("nm_unmatched") == 1) & pl.col("source_class").is_null()).height
    if miss:
        raise RuntimeError(f"source class missing on {miss} unmatched rows")
    keep = expert_training_mask(x["source_class"].fill_null("__matched__").to_numpy(), x["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy(), mode="noncopy_large")
    x = x.filter(pl.Series(keep))
    return x.with_columns((pl.col("TAXITIME_SEC_mvt").cast(pl.Float64) - pl.col("P")).clip(-RESID_CLIP, RESID_CLIP).alias("r"))


def serve_frame(name: str) -> pl.DataFrame:
    """Unmatched rows of one population with the residual-model features (label-free) and P."""

    ids = _ctl(name).select("MVT_ID_mvt", "AIRPORT")
    if name == "ranking":
        u = pl.read_parquet(NONCOPY_R / "frames" / "ranking_unmatched.parquet", columns=["MVT_ID_mvt", *SENS_CATS, *SENS_NUMS_U6])
    else:
        u = pl.read_parquet(E08B, columns=["MVT_ID_mvt", *SENS_CATS, *SENS_NUMS_U6])
    blk = POPS[name]["block"]
    lean = POPS[name]["lean"]
    f = ids.select("MVT_ID_mvt").join(u, on="MVT_ID_mvt", how="left", maintain_order="left")
    f = f.join(pl.read_parquet(blk, columns=["MVT_ID_mvt", *ADSB_NUMS]), on="MVT_ID_mvt", how="left", maintain_order="left")
    f = f.join(pl.read_parquet(lean, columns=["MVT_ID_mvt", *LEAN_COLS]), on="MVT_ID_mvt", how="left", maintain_order="left")
    if f.height != ids.height or (f["AIRPORT"] != ids["AIRPORT"]).any():
        raise RuntimeError(f"{name}: serve frame misaligned")
    return _with_p(f)


def _fit_r(train: pl.DataFrame):
    from prc_challenge.unmatched_recovery.methods import fit_regressor

    return fit_regressor(train, R_CATS, R_NUMS, label="r", **R_PARAMS)


def _predict_r(model, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from prc_challenge.unmatched_recovery.methods import to_pool

    out = np.asarray(model.predict(to_pool(frame, R_CATS, R_NUMS)), dtype=np.float64)
    return out[:, 0], out[:, 1]


def _state_path() -> Path:
    return OUT / "state.json"


def _state() -> dict[str, Any]:
    return json.loads(_state_path().read_text()) if _state_path().exists() else {}


def _save_state(st: dict[str, Any]) -> None:
    _write_json(_state_path(), st)


def phase_fit_sens2() -> None:
    commit = _require_prereg()
    st = _state()
    done = st.setdefault("fit_sens2", {"prereg_commit": commit, "fits": {}})
    (OUT / "sens2").mkdir(parents=True, exist_ok=True)
    serve = {n: serve_frame(n) for n in ("D", "JAN_JUL_2025", "ranking")}
    for n, f in serve.items():
        ref = _ctl(n)
        p_ctl, _ = physical_estimate(ref["taxi_map"].cast(pl.Float64).fill_null(np.nan).to_numpy(), ref["taxi_gate"].cast(pl.Float64).fill_null(np.nan).to_numpy(), ref["AIRPORT"].to_numpy().astype(str))
        a, b = f["P"].to_numpy(), p_ctl
        if not np.array_equal(np.isfinite(a), np.isfinite(b)) or np.nanmax(np.abs(a - b)) > 1e-6:
            raise RuntimeError(f"{n}: P differs between the serve frame and the control frame")
    jobs = [(f"fold_{h[0]}_{h[1]}", tuple(m for m in D_MONTHS if m not in h), "dfit", "D", h) for h in BLOCKS]
    jobs += [("all_D", D_MONTHS, "dfit", "JAN_JUL_2025", None), ("m1_10", tuple(range(1, 11)), "allyear", "ranking", None)]
    cache: dict[str, pl.DataFrame] = {}
    for tag, months, lean_kind, target, hold in jobs:
        path = OUT / "sens2" / f"pred_{tag}.parquet"
        if path.exists() and tag in done["fits"]:
            continue
        _free_mem_check(f"fit_sens2 {tag}")
        key = f"{lean_kind}"
        if key not in cache:
            cache.clear()
            cache[key] = build_train(D_MONTHS if lean_kind == "dfit" else tuple(range(1, 11)), lean_kind)
        tr = cache[key].filter(pl.col("month").is_in(list(months)))
        if hold is not None and tr.filter(pl.col("month").is_in(list(hold))).height:
            raise RuntimeError("holdout leaked")
        t0 = time.time()
        model = _fit_r(tr)
        model.save_model(str(OUT / "sens2" / f"model_{tag}.cbm"))
        sv = serve[target]
        if hold is not None:
            months_col = _ctl("D")["month"].to_numpy()
            sv = sv.filter(pl.Series(np.isin(months_col, hold)))
        sv = sv.filter(pl.col("P").is_finite())
        mu, var = _predict_r(model, sv)
        sv.select("MVT_ID_mvt", "P").with_columns(pl.Series("r_hat", mu), pl.Series("s2", var)).write_parquet(path)
        done["fits"][tag] = {"months": list(months), "lean": lean_kind, "n_train": tr.height, "n_train_unmatched": int((tr["nm_unmatched"] == 1).sum()), "n_pred": sv.height, "seconds": round(time.time() - t0, 1), "best_iteration": model.get_best_iteration(), "r_train_mean": float(tr["r"].mean())}
        _save_state(st)
        _log(f"sens2 {tag}: {done['fits'][tag]}")
    st["fit_sens2"]["status"] = "complete"
    _save_state(st)


def _sens2_q(name: str, prefix: str = "sens2") -> tuple[np.ndarray, np.ndarray]:
    """Q = cap(P + r_hat) and s2 aligned to the control rows of ``name`` (NaN where P is missing)."""

    ids = _ctl(name).select("MVT_ID_mvt")
    tags = {"D": [f"fold_{h[0]}_{h[1]}" for h in BLOCKS], "JAN_JUL_2025": ["all_D"], "ranking": ["m1_10"]}[name]
    pr = pl.concat([pl.read_parquet(OUT / prefix / f"pred_{t}.parquet") for t in tags])
    if pr["MVT_ID_mvt"].n_unique() != pr.height:
        raise RuntimeError("duplicate predictions")
    j = ids.join(pr, on="MVT_ID_mvt", how="left", maintain_order="left")
    sd = _ctl(name)["takeoff_minus_sched_sec"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    pv = j["P"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    r = j["r_hat"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    s2 = j["s2"].cast(pl.Float64).fill_null(np.nan).to_numpy() if "s2" in j.columns else np.full(ids.height, np.nan)
    return cap_channel(pv + r, sd), s2


# ---------------------------------------------------------------------------
# arm evaluation
# ---------------------------------------------------------------------------


def _arrays(data: dict[str, Any], name: str) -> dict[str, Any]:
    f = _ctl(name)
    d = data[name]
    c = d["c"]
    pv, src = physical_estimate(f["taxi_map"].cast(pl.Float64).fill_null(np.nan).to_numpy(), f["taxi_gate"].cast(pl.Float64).fill_null(np.nan).to_numpy(), f["AIRPORT"].to_numpy().astype(str))
    out = {
        "f": f,
        "c": c,
        "spec": d["spec"],
        "ctl": d["ctl"],
        "e_prime": d["e_prime"],
        "e_d": expert_ceiling_d(c.e_cap, c.sd, c.airport) if d["spec"].d else np.asarray(c.e_cap, dtype=np.float64),
        "fired": np.asarray(d["fired"], dtype=bool),
        "lirf_copy": f["lirf_copy_regime"].to_numpy(),
        "P": pv,
        "p_src": src,
        "gate_conf": f["adsb_gate_conf"].cast(pl.Float64).fill_null(0.0).to_numpy(),
        "sd": c.sd,
        "airport": c.airport,
        "month": c.month,
    }
    out["O"] = observable(out["fired"], pv, out["lirf_copy"])
    if "TAXITIME_SEC_mvt" in f.columns and name != "ranking":
        out["y"] = f["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    if name == "D":
        out["fold"], out["day"] = f["fold"].to_numpy(), f["day"].to_numpy()
    elif name == "JAN_JUL_2025":
        out["day"] = f["day"].to_numpy() if "day" in f.columns else f["MVT_TIME_UTC_mvt"].dt.date().cast(pl.String).to_numpy()
    if not np.array_equal(np.isfinite(pv), np.isfinite(pv) & ~np.isin(c.airport, ("LIRF", "LFPG", "LTFM"))):
        raise RuntimeError("P at an excluded airport")
    return out


def _compose(a: dict[str, Any], e2: np.ndarray) -> np.ndarray:
    return compose_with_expert(a["c"], a["spec"], e2)


def _sse_on(a: dict[str, Any], e2: np.ndarray, rows: np.ndarray, mask: np.ndarray) -> float:
    pred = _compose(a, e2)
    m = rows & mask
    return float(np.sum((a["y"][m] - pred[m]) ** 2))


def _arm_expert(arm: str, a: dict[str, Any], prm: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    """E'' and the arm rows for one population and one parameter setting (scalar or per-row)."""

    if arm in ("U-MAP2", "U-MAP2V"):
        rows = umap2_rows(a["p_src"], a["gate_conf"], a["lirf_copy"]) & a["O"]
        if arm == "U-MAP2V":
            raw_map = np.where(a["p_src"] == 1, a["P"], np.nan)
            rows = rows & np.isfinite(raw_map) & (np.nan_to_num(raw_map, nan=-1.0) >= 0.0) & (np.nan_to_num(raw_map, nan=1e9) <= CHANNEL_CEILING)
        m = cap_channel(np.where(a["p_src"] == 1, a["P"], np.nan), a["sd"])
        wf, wn = prm
        return umap2_expert(a["e_prime"], m, rows, a["fired"], wf, wn), rows
    if arm in ("U-SENS2", "U-NN", "U-SENS2+NN"):
        rows = sens2_rows(a["P"], a["lirf_copy"]) & a["O"]
        if arm == "U-SENS2":
            q, s2 = _sens2_q(name, "sens2")
            lam = lam_from_tau(prm, s2)
        elif arm == "U-NN":
            q, _ = _sens2_q(name, "unn")
            lam = prm
        else:
            q1, s2 = _sens2_q(name, "sens2")
            q2, _ = _sens2_q(name, "unn")
            q = 0.5 * (q1 + q2)
            lam = lam_from_tau(prm, s2)
        if np.any(rows & ~np.isfinite(q)):
            raise RuntimeError(f"{arm}/{name}: Q missing on {int(np.sum(rows & ~np.isfinite(q)))} arm rows")
        return physical_expert(a["e_prime"], a["e_d"], q, rows, lam), rows
    raise ValueError(arm)


def _fit_params(arm: str, a: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Cross-fitted parameter choice on D (per fold on the other three folds) and the all-D choice."""

    fold = a["fold"]
    if arm in ("U-MAP2", "U-MAP2V"):
        res = {}
        for k, sel in (("fired", a["fired"]), ("not_fired", ~a["fired"])):
            def sse_of(w, mask, sel=sel):
                prm = (w, 0.0) if k == "fired" else (0.0, w)
                e2, rows = _arm_expert(arm, a, prm, "D")
                return _sse_on(a, e2, rows & sel, mask)
            res[k] = crossfit_choice(list(W_GRID), sse_of, fold, FOLDS)
        per_fold = {f: (res["fired"][0][f], res["not_fired"][0][f]) for f in FOLDS}
        return per_fold, (res["fired"][1], res["not_fired"][1])
    grid = list(W_GRID) if arm == "U-NN" else list(TAU_GRID)

    def sse_of(v, mask):
        e2, rows = _arm_expert(arm, a, v, "D")
        return _sse_on(a, e2, rows, mask)

    return crossfit_choice(grid, sse_of, fold, FOLDS)


def _d_candidate(arm: str, a: dict[str, Any], per_fold: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    e2 = np.array(a["e_prime"], dtype=np.float64)
    rows_all = np.zeros(a["c"].n, dtype=bool)
    for f in FOLDS:
        ef, rows = _arm_expert(arm, a, per_fold[f], "D")
        m = a["fold"] == f
        e2[m] = ef[m]
        rows_all |= rows & m
    return _compose(a, e2), rows_all


def _expected_public(arm: str, rep: dict[str, Any], a_d: dict[str, Any], cand_d: np.ndarray, a_r: dict[str, Any], cand_r: np.ndarray, jj: dict[str, Any] | None) -> dict[str, Any]:
    s_rank = float(np.sum((cand_r - a_r["ctl"]) ** 2))
    eff_d = rep["efficiency_subset"]
    out: dict[str, Any] = {"S_rank_sum_sq_change": s_rank, "rows_changed_rank": int(np.sum(np.abs(cand_r - a_r["ctl"]) > 1e-9)), "eff_D": eff_d}

    def sec(dsse: float) -> float:
        return float(np.sqrt(max(PUB_BASE**2 + dsse / PUB_N, 0.0)) - PUB_BASE)

    if eff_d is not None:
        out["E1_kappa1"] = {"dSSE": eff_d * s_rank, "seconds": sec(eff_d * s_rank)}
        out["E1_kappa2_central"] = {"dSSE": KAPPA_CENTRAL * eff_d * s_rank, "seconds": sec(KAPPA_CENTRAL * eff_d * s_rank)}
    if jj is not None and jj.get("efficiency_subset") is not None:
        e = jj["efficiency_subset"]
        out["E1_JJ_efficiency"] = {"eff_JJ": e, "dSSE": e * s_rank, "seconds": sec(e * s_rank)}
    # E2: per-row D gain by |control - P| cell x ranking O rows in the cell
    yd = a_d["y"]
    dd = (yd - cand_d) ** 2 - (yd - a_d["ctl"]) ** 2
    cd = change_cells(a_d["ctl"], a_d["P"])
    cr = change_cells(a_r["ctl"], a_r["P"])
    cells = {}
    tot = 0.0
    for k in ("lt300", "300_600", "600_900", "ge900", "P_missing"):
        md = a_d["O"] & (cd == k)
        mr = a_r["O"] & (cr == k)
        per = float(dd[md].mean()) if md.any() else 0.0
        cells[k] = {"n_D": int(md.sum()), "dSSE_D": float(dd[md].sum()), "per_row_D": per, "n_rank": int(mr.sum())}
        tot += per * mr.sum()
    out["E2_cells"] = cells
    out["E2"] = {"dSSE": tot, "seconds": sec(tot)}
    return out


def phase_score(arm: str) -> None:
    commit = _require_prereg()
    data = load_all()
    a = _arrays(data, "D")
    per_fold, all_choice = _fit_params(arm, a)
    cand, rows = _d_candidate(arm, a, per_fold)
    rep = subset_gates(a["y"], a["ctl"], cand, day=a["day"], fold=a["fold"], airport=a["airport"], subset=a["O"], lirf_copy=a["lirf_copy"])
    if np.any((np.abs(cand - a["ctl"]) > 1e-9) & ~a["O"]):
        raise RuntimeError("arm changed rows outside O")
    # label-free ranking change for the expected-effect estimate (all-D parameters)
    ar = _arrays(data, "ranking")
    try:
        e2r, rows_r = _arm_expert(arm, ar, all_choice, "ranking")
        cand_r = _compose(ar, e2r)
    except FileNotFoundError:  # ranking fit not finished yet: expected effect follows in the confirm phase
        e2r, rows_r, cand_r = None, np.zeros(ar["c"].n, dtype=bool), None
    res = {
        "arm": arm,
        "prereg_commit": commit,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": "GO" if rep["go"] else "NO-GO",
        "params_per_fold": per_fold,
        "params_all_D": all_choice,
        "arm_rows_D": int(rows.sum()),
        "arm_rows_ranking": int(rows_r.sum()),
        "D": rep,
        "D_by_exposure_class": _by_class(a, cand),
        "expected_public": _expected_public(arm, rep, a, cand, ar, cand_r, None) if cand_r is not None else "pending (ranking fit)",
    }
    tag = arm.replace("+", "_")
    _write_json(OUT / "metrics" / f"score_{tag}.json", res)
    _write_json(TRACKED / f"score_{tag}.json", res)
    g = rep["gates"]
    _log(f"{arm}: {res['decision']} D dSSE_O {rep['delta_sse_subset']:.4e} p95 {g['P1_subset_boot_p95_lt_0']['value']:.3e} folds {g['P2_subset_folds_ge_3']['value']}/4 worst ap {g['P3_subset_airport_cap_2pct']['value']:+.4f}; params {all_choice}; ranking rows {int(rows_r.sum())}, E1k2 {res['expected_public'].get('E1_kappa2_central') if isinstance(res['expected_public'], dict) else res['expected_public']}")


def _by_class(a: dict[str, Any], cand: np.ndarray) -> dict[str, Any]:
    y = a["y"]
    d = (y - cand) ** 2 - (y - a["ctl"]) ** 2
    cls = a["f"]["exposure"].to_numpy()
    cells = change_cells(a["ctl"], a["P"])
    out: dict[str, Any] = {"exposure": {}, "gap_cell": {}}
    for k in sorted(set(cls.tolist())):
        m = a["O"] & (cls == k)
        out["exposure"][k] = {"n": int(m.sum()), "delta_sse": float(d[m].sum())}
    for k in sorted(set(cells.tolist())):
        m = a["O"] & (cells == k)
        out["gap_cell"][k] = {"n": int(m.sum()), "delta_sse": float(d[m].sum())}
    return out


def phase_confirm(arm: str) -> None:
    _require_prereg()
    tag = arm.replace("+", "_")
    sc = json.loads((OUT / "metrics" / f"score_{tag}.json").read_text())
    if sc["decision"] != "GO":
        raise SystemExit(f"{arm} is not GO on D; JAN_JUL look refused")
    path = OUT / "metrics" / f"confirm_{tag}.json"
    if path.exists():
        raise SystemExit(f"{arm}: JAN_JUL look already taken")
    data = load_all()
    a = _arrays(data, "JAN_JUL_2025")
    prm = sc["params_all_D"]
    prm = tuple(prm) if isinstance(prm, list) else prm
    e2, rows = _arm_expert(arm, a, prm, "JAN_JUL_2025")
    cand = _compose(a, e2)
    y = a["y"]
    d = (y - cand) ** 2 - (y - a["ctl"]) ** 2
    from prc_challenge.unmatched_recovery.sensor import delta_sse_bootstrap

    o = a["O"]
    boot = delta_sse_bootstrap(y[o], a["ctl"][o], cand[o], a["day"][o])
    se_c = (y - a["ctl"]) ** 2
    jj = {
        "n": int(y.size),
        "n_subset": int(o.sum()),
        "rows_changed": int(np.sum(np.abs(cand - a["ctl"]) > 1e-9)),
        "delta_sse_subset": float(d[o].sum()),
        "delta_sse_full": float(d.sum()),
        "ctl_sse_subset": float(se_c[o].sum()),
        "efficiency_subset": float(d[o].sum() / np.sum(((cand - a["ctl"]) ** 2)[o])) if np.sum(((cand - a["ctl"]) ** 2)[o]) > 0 else None,
        "subset_bootstrap_informational": boot,
        "by_month": {int(m): float(d[a["month"] == m].sum()) for m in (1, 7)},
        "by_airport": {str(ap): {"delta_sse": float(d[o & (a["airport"] == ap)].sum()), "ctl_sse": float(se_c[o & (a["airport"] == ap)].sum())} for ap in sorted(set(a["airport"][o].tolist()))},
        "approx_full_rmse_change_s": float(np.sqrt(JJ_BASE**2 + d.sum() / JJ_N) - JJ_BASE),
        "top5_share": float(np.sort(d)[:5].sum() / d.sum()) if d.sum() < 0 else None,
    }
    verdict = "CONSISTENT" if jj["delta_sse_subset"] < 0 else "FLAG"
    ad = _arrays(data, "D")
    cand_d, _ = _d_candidate(arm, ad, {f: (tuple(v) if isinstance(v, list) else v) for f, v in sc["params_per_fold"].items()})
    ar = _arrays(data, "ranking")
    e2r, _ = _arm_expert(arm, ar, prm, "ranking")
    cand_r = _compose(ar, e2r)
    out = {"arm": arm, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "verdict": verdict, "params": prm, "JAN_JUL_2025": jj, "expected_public": _expected_public(arm, sc["D"], ad, cand_d, ar, cand_r, jj)}
    _write_json(path, out)
    _write_json(TRACKED / f"confirm_{tag}.json", out)
    _log(f"{arm} JAN_JUL: {verdict} dSSE_O {jj['delta_sse_subset']:.4e} full {jj['delta_sse_full']:.4e} (~{jj['approx_full_rmse_change_s']:+.3f} s)")


def phase_ranking(arm: str) -> None:
    from prc_challenge.io import scan_submitting
    from prc_challenge.unmatched_recovery.harness import RANKING_N, RANKING_UNMATCHED_N, overlay_unmatched_submission, verify_arm_submission

    tag = arm.replace("+", "_")
    if arm == "U-MAP2V":
        rv = json.loads((OUT / "metrics" / "results_U-MAP2V.json").read_text())
        if rv["decision"] != "GO":
            raise SystemExit("U-MAP2V: not GO on JAN_JUL")
        sc = {"D": {"delta_sse_subset": rv["D_disclosure_not_gated"]["delta_sse_subset"], "efficiency_subset": rv["D_disclosure_not_gated"]["efficiency_subset"]}}
        cf = {"params": rv["params_all_D"], "JAN_JUL_2025": rv["JAN_JUL_2025"], "expected_public": rv["expected_public"]}
    else:
        sc = json.loads((OUT / "metrics" / f"score_{tag}.json").read_text())
        cf = json.loads((OUT / "metrics" / f"confirm_{tag}.json").read_text())
        if sc["decision"] != "GO" or cf["verdict"] != "CONSISTENT":
            raise SystemExit(f"{arm}: not GO + CONSISTENT")
    data = load_all()
    ar = _arrays(data, "ranking")
    prm = cf["params"]
    prm = tuple(prm) if isinstance(prm, list) else prm
    e2, rows = _arm_expert(arm, ar, prm, "ranking")
    new = _compose(ar, e2)
    if not np.isfinite(new).all() or (new < 0).any():
        raise RuntimeError("invalid predictions")
    if np.any((np.abs(new - ar["ctl"]) > 1e-9) & ~ar["O"]):
        raise RuntimeError("ranking rows outside O changed")
    parent = pl.read_parquet(ALL_C0_FILE)
    sub = overlay_unmatched_submission(parent, pl.DataFrame({"MVT_ID_mvt": ar["c"].mvt_id, "prediction": new}))
    verify = verify_arm_submission(sub, template=scan_submitting(REC.RAW).collect(), parent=parent, unmatched_ids=pl.Series(ar["c"].mvt_id), expected_unmatched_n=RANKING_UNMATCHED_N, expected_n=RANKING_N)
    if not verify["ok"]:
        raise RuntimeError(f"validator failed {verify['errors']}")
    (OUT / "ranking").mkdir(parents=True, exist_ok=True)
    path = OUT / "ranking" / f"unmatched_submission_{tag}.parquet"
    sub.write_parquet(path)
    ch = REC.changed_by_airport(ar["ctl"], new, ar["airport"])
    man = {
        "name": f"SA-UNM5 {arm} unmatched rows on ALL + C0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file": {"path": str(path), "sha256": _sha(path), "rows": sub.height, "use": "--unmatched input of scripts/compose/compose_candidate.py (unmatched rows already E17/E18 + clip composed)"},
        "parent": {"path": str(ALL_C0_FILE), "sha256": ALL_C0_SHA},
        "params": prm,
        "arm_rows": int(rows.sum()),
        "rows_changed_vs_ALL_C0": ch["changed"],
        "rows_changed_by_month": {int(m): int(np.sum((np.abs(new - ar["ctl"]) > 1e-9) & (ar["month"] == m))) for m in (1, 7)},
        "vs_ALL_C0": ch,
        "D_score": {k: sc["D"][k] for k in ("delta_sse_subset", "efficiency_subset")},
        "JAN_JUL": {k: cf["JAN_JUL_2025"][k] for k in ("delta_sse_subset", "delta_sse_full", "approx_full_rmse_change_s")},
        "expected_public": cf["expected_public"],
        "validator": {"ok": verify["ok"], "errors": verify["errors"], "matched_changed": verify.get("matched_changed")},
        "uploaded": False,
        "eligibility": "adsb.lol ODbL 1.0 (D-017); no DEP BLOCK/TAXITIME features; no Nov/Dec labels",
    }
    _write_json(OUT / "ranking" / f"unmatched_submission_{tag}.manifest.json", man)
    _write_json(TRACKED / f"ranking_manifest_{tag}.json", man)
    _log(f"ranking {arm}: changed {ch['changed']} sha {man['file']['sha256']}")



AMEND1 = TRACKED / "preregistration_amendment_1_umap2v.json"


def phase_amend1() -> None:
    """U-MAP2V: D numbers (disclosure only) and the JAN_JUL 2025 gate J1-J4 (one look)."""

    arm = "U-MAP2V"
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", str(AMEND1)], capture_output=True, text=True).stdout.strip()
    if not commit:
        raise SystemExit("amendment 1 is not committed")
    path = OUT / "metrics" / "results_U-MAP2V.json"
    if path.exists():
        raise SystemExit("U-MAP2V: JAN_JUL look already taken")
    from prc_challenge.unmatched_recovery.sensor import delta_sse_bootstrap

    data = load_all()
    a = _arrays(data, "D")
    per_fold, all_choice = _fit_params(arm, a)
    cand_d, rows_d = _d_candidate(arm, a, per_fold)
    rep_d = subset_gates(a["y"], a["ctl"], cand_d, day=a["day"], fold=a["fold"], airport=a["airport"], subset=a["O"], lirf_copy=a["lirf_copy"])
    j = _arrays(data, "JAN_JUL_2025")
    e2, rows_j = _arm_expert(arm, j, all_choice, "JAN_JUL_2025")
    cand_j = _compose(j, e2)
    y = j["y"]
    o = j["O"]
    d = (y - cand_j) ** 2 - (y - j["ctl"]) ** 2
    se_c = (y - j["ctl"]) ** 2
    boot = delta_sse_bootstrap(y[o], j["ctl"][o], cand_j[o], j["day"][o])
    aps = {str(ap): {"n": int((o & (j["airport"] == ap)).sum()), "rows_changed": int(((np.abs(cand_j - j["ctl"]) > 1e-9) & o & (j["airport"] == ap)).sum()), "ctl_sse": float(se_c[o & (j["airport"] == ap)].sum()), "delta_sse": float(d[o & (j["airport"] == ap)].sum())} for ap in sorted(set(j["airport"][o].tolist()))}
    for v in aps.values():
        v["delta_frac"] = v["delta_sse"] / v["ctl_sse"] if v["ctl_sse"] > 0 else 0.0
    worst = max(v["delta_frac"] for v in aps.values())
    lirf_diff = float(np.max(np.abs(cand_j - j["ctl"])[j["lirf_copy"]])) if j["lirf_copy"].any() else 0.0
    gates = {
        "J1_dSSE_O_lt_0": {"value": float(d[o].sum()), "pass": bool(d[o].sum() < 0)},
        "J2_boot_p95_lt_0": {"value": boot["p95"], "pass": bool(boot["p95"] < 0)},
        "J3_airport_cap_2pct": {"value": worst, "pass": bool(worst <= 0.02)},
        "J4_lirf_copy_unchanged": {"value": lirf_diff, "pass": bool(lirf_diff == 0.0)},
    }
    decision = "GO" if all(g["pass"] for g in gates.values()) else "NO-GO"
    sq = float(np.sum(((cand_j - j["ctl"]) ** 2)[o]))
    jj = {"n": int(y.size), "n_subset": int(o.sum()), "arm_rows": int(rows_j.sum()), "rows_changed": int(np.sum(np.abs(cand_j - j["ctl"]) > 1e-9)), "delta_sse_subset": float(d[o].sum()), "delta_sse_full": float(d.sum()), "efficiency_subset": float(d[o].sum() / sq) if sq > 0 else None, "bootstrap": boot, "airports": aps, "by_month": {int(m): float(d[j["month"] == m].sum()) for m in (1, 7)}, "approx_full_rmse_change_s": float(np.sqrt(JJ_BASE**2 + d.sum() / JJ_N) - JJ_BASE), "top5_share": float(np.sort(d)[:5].sum() / d.sum()) if d.sum() < 0 else None}
    ar = _arrays(data, "ranking")
    e2r, rows_r = _arm_expert(arm, ar, all_choice, "ranking")
    cand_r = _compose(ar, e2r)
    exp = _expected_public(arm, rep_d, a, cand_d, ar, cand_r, jj)
    # E2 with JAN_JUL per-row cell gains (amendment: JAN_JUL-based)
    cj, cr = change_cells(j["ctl"], j["P"]), change_cells(ar["ctl"], ar["P"])
    tot = 0.0
    cells = {}
    for k in ("lt300", "300_600", "600_900", "ge900", "P_missing"):
        mj, mr = o & (cj == k), ar["O"] & (cr == k)
        per = float(d[mj].mean()) if mj.any() else 0.0
        cells[k] = {"n_JJ": int(mj.sum()), "dSSE_JJ": float(d[mj].sum()), "per_row_JJ": per, "n_rank": int(mr.sum())}
        tot += per * mr.sum()
    exp["E2_JJ_cells"] = cells
    exp["E2_JJ"] = {"dSSE": tot, "seconds": float(np.sqrt(PUB_BASE**2 + tot / PUB_N) - PUB_BASE)}
    out = {"arm": arm, "amendment_commit": commit, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "decision": decision, "gates": gates, "params_all_D": all_choice, "params_per_fold": per_fold, "D_disclosure_not_gated": {k: rep_d[k] for k in ("n_subset", "rows_changed", "delta_sse_subset", "efficiency_subset", "subset_bootstrap", "folds", "subset_airports", "gates", "failed")}, "JAN_JUL_2025": jj, "arm_rows_ranking": int(rows_r.sum()), "expected_public": exp}
    _write_json(path, out)
    _write_json(TRACKED / "results_U-MAP2V.json", out)
    _log(f"U-MAP2V: {decision} JJ dSSE_O {d[o].sum():.4e} p95 {boot['p95']:.3e} worst {worst:+.4f}; params {all_choice}; D (disclosure) {rep_d['delta_sse_subset']:.4e} failed {rep_d['failed']}; ranking rows {int(rows_r.sum())}")


def phase_inspect_nf() -> None:
    """LEAD, not cleared: U-MAP2V with w_fired = 0 (map channel only on rows U-SENS-G never fired on:
    the EHAM take-off-time fallback joins). Post hoc (the split was chosen after the U-MAP2 / U-MAP2V results);
    writes an inspection file for the orchestrator's decision. Never uploads."""

    from prc_challenge.io import scan_submitting
    from prc_challenge.unmatched_recovery.harness import RANKING_N, RANKING_UNMATCHED_N, overlay_unmatched_submission, verify_arm_submission
    from prc_challenge.unmatched_recovery.sensor import delta_sse_bootstrap

    rv = json.loads((OUT / "metrics" / "results_U-MAP2V.json").read_text())
    w_nf = float(rv["params_all_D"][1])
    data = load_all()
    stats: dict[str, Any] = {}
    for name in ("D", "JAN_JUL_2025"):
        a = _arrays(data, name)
        if name == "D":
            cand, _ = _d_candidate("U-MAP2V", a, {f: (0.0, float(v[1])) for f, v in rv["params_per_fold"].items()})
        else:
            e2, _ = _arm_expert("U-MAP2V", a, (0.0, w_nf), name)
            cand = _compose(a, e2)
        y = a["y"]
        d = (y - cand) ** 2 - (y - a["ctl"]) ** 2
        o = a["O"]
        b = delta_sse_bootstrap(y[o], a["ctl"][o], cand[o], a["day"][o])
        sq = float(np.sum((cand - a["ctl"]) ** 2))
        stats[name] = {"rows_changed": int(np.sum(np.abs(cand - a["ctl"]) > 1e-9)), "delta_sse": float(d.sum()), "sum_sq_change": sq, "efficiency": float(d.sum() / sq) if sq else None, "boot_p05": b["p05"], "boot_p95": b["p95"], "by_airport": {str(ap): float(d[a["airport"] == ap].sum()) for ap in sorted(set(a["airport"].tolist())) if abs(float(d[a["airport"] == ap].sum())) > 0}}
        if name == "D":
            stats[name]["folds"] = {f: float(d[a["fold"] == f].sum()) for f in FOLDS}
        else:
            stats[name]["months"] = {int(m): float(d[a["month"] == m].sum()) for m in (1, 7)}
    ar = _arrays(data, "ranking")
    e2, rows = _arm_expert("U-MAP2V", ar, (0.0, w_nf), "ranking")
    new = _compose(ar, e2)
    if np.any((np.abs(new - ar["ctl"]) > 1e-9) & ar["fired"]):
        raise RuntimeError("fired rows changed")
    parent = pl.read_parquet(ALL_C0_FILE)
    sub = overlay_unmatched_submission(parent, pl.DataFrame({"MVT_ID_mvt": ar["c"].mvt_id, "prediction": new}))
    verify = verify_arm_submission(sub, template=scan_submitting(REC.RAW).collect(), parent=parent, unmatched_ids=pl.Series(ar["c"].mvt_id), expected_unmatched_n=RANKING_UNMATCHED_N, expected_n=RANKING_N)
    if not verify["ok"]:
        raise RuntimeError(f"validator failed {verify['errors']}")
    d_out = OUT / "ranking_inspection_not_cleared"
    d_out.mkdir(parents=True, exist_ok=True)
    path = d_out / "unmatched_submission_U-MAP2V-NF_NOT_CLEARED.parquet"
    sub.write_parquet(path)
    s_rank = float(np.sum((new - ar["ctl"]) ** 2))
    ch = REC.changed_by_airport(ar["ctl"], new, ar["airport"])

    def sec(x: float) -> float:
        return float(np.sqrt(PUB_BASE**2 + x / PUB_N) - PUB_BASE)

    exp = {"S_rank": s_rank, "eff_D": stats["D"]["efficiency"], "eff_JJ": stats["JAN_JUL_2025"]["efficiency"]}
    for k, e in (("D_k1", stats["D"]["efficiency"]), ("JJ_k1", stats["JAN_JUL_2025"]["efficiency"]), ("D_k2", KAPPA_CENTRAL * stats["D"]["efficiency"]), ("usensg_public_eff_-2.34", -2.34)):
        exp[k] = {"dSSE": e * s_rank, "seconds": sec(e * s_rank)}
    man = {
        "status": "LEAD - NOT CLEARED (post hoc: w_fired = 0 chosen after U-MAP2 / U-MAP2V; D and JAN_JUL both seen)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": f"U-MAP2V map channel on rows with a valid reliable map (0 <= P_map <= 3606), no confirmed gate, NOT fired by U-SENS-G: E'' = E' + {w_nf} (cap(P_map) - E'); everything else = ALL + C0",
        "file": {"path": str(path), "sha256": _sha(path), "rows": sub.height},
        "parent": {"path": str(ALL_C0_FILE), "sha256": ALL_C0_SHA},
        "rows_changed": ch["changed"],
        "rows_changed_by_month": {int(m): int(np.sum((np.abs(new - ar["ctl"]) > 1e-9) & (ar["month"] == m))) for m in (1, 7)},
        "vs_ALL_C0": ch,
        "local": stats,
        "expected_public": exp,
        "validator": {"ok": verify["ok"], "errors": verify["errors"]},
        "uploaded": False,
    }
    _write_json(d_out / "unmatched_submission_U-MAP2V-NF_NOT_CLEARED.manifest.json", man)
    _write_json(TRACKED / "inspection_U-MAP2V-NF.json", man)
    _log(f"inspection U-MAP2V-NF: {ch['changed']} rows, sha {man['file']['sha256']}, expected {exp}")


def phase_disruption() -> None:
    """Label-free: ranking airport-days with >= 5 unmatched physical rows at |ALL+C0 - P| > 600 s, next to the
    same day's NM-matched departures (MVT - AOBT_3 and map taxi medians). No BLOCK/TAXITIME is read."""

    f = _ctl("ranking").drop("TAXITIME_SEC_mvt", strict=False)
    f = f.with_columns((pl.col("all_c0") - pl.coalesce("taxi_map", "taxi_gate")).alias("phys_gap"))
    raw = pl.scan_parquet(REC.RAW / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", pl.col("ADEP_mvt").alias("AIRPORT"), "MVT_TIME_UTC_mvt", "AOBT_3_flt", "FLIGHT_ID_mvt").collect()
    lean = pl.read_parquet(V3 / "adsb_v3lean_block_allyear.parquet", columns=["MVT_ID_mvt", "adsb3_map_first_rel", "adsb3_map_first_sd", "adsb3_aobt_minus_map"])
    raw = raw.join(lean, on="MVT_ID_mvt", how="left").with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("d"))
    f = f.join(raw.select("MVT_ID_mvt", "d"), on="MVT_ID_mvt", how="left")
    big = f.filter(pl.col("physical") & (pl.col("phys_gap").abs() > 600)).group_by("AIRPORT", "d").agg(pl.len().alias("n_unm_gap600"), pl.col("phys_gap").median().alias("unm_pred_minus_P_med"), pl.coalesce("taxi_map", "taxi_gate").median().alias("unm_P_med"), pl.col("all_c0").median().alias("unm_pred_med"))
    big = big.filter(pl.col("n_unm_gap600") >= 5)
    ok = pl.col("adsb3_map_first_sd") <= 180
    m = raw.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).group_by("AIRPORT", "d").agg(
        pl.len().alias("n_matched"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().median().alias("matched_mvt_minus_aobt_med"),
        (-pl.col("adsb3_map_first_rel")).filter(ok).median().alias("matched_map_taxi_med"),
        pl.col("adsb3_aobt_minus_map").filter(ok).median().alias("matched_aobt_minus_map_med"),
    )
    typ = raw.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")).group_by("AIRPORT", "mo").agg((pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().median().alias("month_matched_mvt_minus_aobt_med"))
    out = big.join(m, on=["AIRPORT", "d"], how="left").with_columns(pl.col("d").dt.month().alias("mo")).join(typ, on=["AIRPORT", "mo"], how="left").drop("mo").sort("AIRPORT", "d")
    out.write_csv(TRACKED / "exposure_ranking_disruption_days.csv")
    _log(f"disruption days: {out.height} airport-days, {int(out['n_unm_gap600'].sum())} unmatched rows")
    print(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["control", "exposure", "disruption", "fit_sens2", "score", "confirm", "ranking", "amend1", "inspect_nf"])
    ap.add_argument("--arm", choices=["U-MAP2", "U-MAP2V", "U-SENS2", "U-NN", "U-SENS2+NN"])
    a = ap.parse_args()
    if a.phase in ("score", "confirm", "ranking"):
        {"score": phase_score, "confirm": phase_confirm, "ranking": phase_ranking}[a.phase](a.arm)
    else:
        {"control": phase_control, "exposure": phase_exposure, "disruption": phase_disruption, "fit_sens2": phase_fit_sens2, "amend1": phase_amend1, "inspect_nf": phase_inspect_nf}[a.phase]()
