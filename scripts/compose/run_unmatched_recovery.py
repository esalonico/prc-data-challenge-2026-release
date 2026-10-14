# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-UNM4 unmatched recovery arms (2026-09-26): soft E17 (A), LIRF month regime (B),
destination-arrival clock link (C), expert ceiling (D), combined (ALL).

Pre-registered in progress-review/unmatched-recovery-20260926/preregistration.json before any arm
was scored. No model is fitted; every arm is a deterministic change of persisted components.

Phases:
  prepare   persist the current unmatched system (CAP_B + P1 + U-SENS-G) as files: D OOF, JAN_JUL 2025
            predictions, ranking components; identity checks against the deployed artifacts.
            Label-free: destination-arrival links (months 1-10 and ranking), LIRF ARR BLOCK==SCHED rates.
  score     arms on D (gates) and JAN_JUL 2025, then ALL = accepted arms.
  ranking   one unmatched submission file + manifest per accepted arm (and ALL). Never uploads.
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
EXP = MAIN / "artifacts" / "experiments"
OUT = EXP / "Unmatched-recovery-20260926"
TRACKED = ROOT / "progress-review" / "unmatched-recovery-20260926"
POLISH = EXP / "Unmatched-polish-20260925"
SENSOR = EXP / "Unmatched-sensor-20260926"
NONCOPY = EXP / "Unmatched-noncopy-20260925"
ADSB_D = EXP / "ADSB-features-20260925" / "features" / "adsb_block_D.parquet"
RAW = MAIN / "data" / "raw"
USENS_G_SUB = SENSOR / "ranking" / "unmatched_submission.parquet"
USENS_G_SUB_SHA = "35311cd40c06bad7ab0f16c268522d667268534701543699265adac59636c103"
USENS_G_OVERLAY = SENSOR / "ranking" / "unmatched_overlay.parquet"
MONTHS_1_10 = tuple(range(1, 11))
D_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)
JJ_MONTHS = (1, 7)
MIN_FREE = 25.0
JJ_N_FULL = 344_419  # JAN_JUL 2025 rows (for the approximate full-RMSE conversion)
JJ_BASE_RMSE = 319.137  # six-leg JAN_JUL full RMSE (D-053), used only to convert dSSE to seconds
ARMS = ("A", "B", "C", "D")
D_HARM_TOL = 1.0e7  # Arm D "no harm": dSSE <= +1e7 on D and on JAN_JUL

from prc_challenge.unmatched_recovery.polish import cap_expert  # noqa: E402
from prc_challenge.unmatched_recovery.recovery import (  # noqa: E402
    FOLD_OF_MONTH,
    LINK_ARR_COLS,
    LINK_DEP_COLS,
    ArmSpec,
    Components,
    arm_a_q_crossfit,
    arr_eq_rates,
    attach_links,
    changed_by_airport,
    compose,
    destination_links,
    fit_rate_map,
    month_logit_shift,
    near_copy,
    rate_target,
)
from prc_challenge.unmatched_recovery.sensor import arm_gates, sensor_blend  # noqa: E402


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
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=1) + "\n")


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


def _raw_file(m: int) -> Path:
    nxt = f"2025-{m + 1:02d}-01" if m < 12 else "2026-01-01"
    return RAW / f"training_2025-{m:02d}-01_{nxt}.parquet"


def _aligned(ids: pl.DataFrame, other: pl.DataFrame, col: str, *, allow_null: bool = False) -> np.ndarray:
    j = ids.select("MVT_ID_mvt").join(other.select("MVT_ID_mvt", col), on="MVT_ID_mvt", how="left", maintain_order="left")
    if j.height != ids.height or (not allow_null and j[col].null_count()):
        raise RuntimeError(f"{col}: alignment failed")
    return j[col].cast(pl.Float64).fill_null(np.nan).to_numpy()


# ---------------------------------------------------------------------------
# components of the three populations
# ---------------------------------------------------------------------------

COMP_COLS = ("MVT_ID_mvt", "month", "AIRPORT", "takeoff_minus_sched_sec", "takeoff_minus_sched_days", "type_missing", "p_copy", "e_cap", "S", "adsb_coverage", "T")


def _components(frame: pl.DataFrame) -> Components:
    return Components(
        mvt_id=frame["MVT_ID_mvt"].to_numpy(),
        month=frame["month"].cast(pl.Int64).to_numpy(),
        airport=frame["AIRPORT"].cast(pl.String).to_numpy().astype(str),
        sd=frame["takeoff_minus_sched_sec"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        days=frame["takeoff_minus_sched_days"].cast(pl.Int64).to_numpy(),
        type_missing=frame["type_missing"].cast(pl.Int64).to_numpy(),
        p=frame["p_copy"].cast(pl.Float64).to_numpy(),
        e_cap=frame["e_cap"].cast(pl.Float64).to_numpy(),
        s=frame["S"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        cov=frame["adsb_coverage"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        t=frame["T"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
    )


def _mixture_control(c: Components, e: np.ndarray) -> np.ndarray:
    """CAP_B + P1 route for an arbitrary E' (used for the identity checks)."""

    from prc_challenge.unmatched_recovery.noncopy_expert import GATE_SEC, mixture_prediction
    from prc_challenge.unmatched_recovery.polish import LAMBDA, blend_on_mask
    from prc_challenge.unmatched_recovery.schema import compose_experimental_unmatched

    raw = blend_on_mask(mixture_prediction(c.p, c.sd, e, gate_sec=GATE_SEC), c.t, c.lirf, LAMBDA)
    fr = pl.DataFrame({"AIRPORT": c.airport, "nm_unmatched": np.ones(c.n, dtype=np.int8), "type_missing": c.type_missing, "takeoff_minus_sched_days": c.days, "takeoff_minus_sched_sec": c.sd, "prediction": raw})
    return compose_experimental_unmatched(fr)["prediction"].to_numpy()


def _build_d() -> tuple[pl.DataFrame, dict[str, Any]]:
    a = pl.read_parquet(POLISH / "oof" / "arms_D.parquet")
    g = pl.read_parquet(POLISH / "frames" / "control_D.parquet").select("MVT_ID_mvt", "nm_unmatched", "type_missing", "takeoff_minus_sched_days")
    s = pl.read_parquet(SENSOR / "oof" / "arms_D.parquet").select("MVT_ID_mvt", pl.col("control").alias("sens_control"), pl.col("U-SENS").alias("sens_usens"), "S", "eligible_sens")
    cov = pl.read_parquet(ADSB_D, columns=["MVT_ID_mvt", "adsb_coverage"])
    lin = pl.read_parquet(NONCOPY / "frames" / "d_unmatched_lineage_iso.parquet").select("MVT_ID_mvt", "source_class")
    tm = pl.read_parquet(SENSOR / "frames" / "control_D.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"])
    x = a.join(g, on="MVT_ID_mvt").join(s, on="MVT_ID_mvt").join(cov, on="MVT_ID_mvt", how="left").join(lin, on="MVT_ID_mvt").join(tm, on="MVT_ID_mvt", how="left")
    if x.height != a.height or x["MVT_TIME_UTC_mvt"].null_count() or (x["nm_unmatched"] != 1).any():
        raise RuntimeError("D component join failed")
    sd = x["takeoff_minus_sched_sec"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    e_cap, _ = cap_expert(x["expert_octfix"].to_numpy(), sd)
    x = x.with_columns(pl.Series("e_cap", e_cap))
    c = _components(x)
    ctl = _mixture_control(c, c.e_cap)
    e_us, elig = sensor_blend(c.e_cap, c.s, c.sd, c.cov, c.airport, c.p)
    us = _mixture_control(c, e_us)
    cur = compose(c)
    y = x["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    ident = {
        "capb_p1_vs_sensor_arms_D_control": float(np.max(np.abs(ctl - x["sens_control"].to_numpy()))),
        "usens_vs_sensor_arms_D_USENS": float(np.max(np.abs(us - x["sens_usens"].to_numpy()))),
        "eligible_vs_sensor_arms_D": int(np.sum(elig != x["eligible_sens"].to_numpy())),
    }
    sse = {"capb_p1": float(((ctl - y) ** 2).sum()), "usens": float(((us - y) ** 2).sum()), "current_usens_g": float(((cur - y) ** 2).sum())}
    sse["delta_usens_g_vs_capb_p1"] = sse["current_usens_g"] - sse["capb_p1"]
    out = x.select(*COMP_COLS, "TAXITIME_SEC_mvt", "source_class", "MVT_TIME_UTC_mvt").with_columns(
        pl.col("month").replace_strict(FOLD_OF_MONTH, return_dtype=pl.String).alias("fold"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"),
        pl.Series("capb_p1", ctl),
        pl.Series("current", cur),
    )
    return out, {"identity_max_abs_diff": ident, "sse": sse, "n": out.height}


def _build_jj() -> tuple[pl.DataFrame, dict[str, Any]]:
    c0 = pl.read_parquet(POLISH / "confirm" / "janjul_unmatched_components.parquet").select("MVT_ID_mvt", "AIRPORT", "month", "p_copy", "T")
    cap = pl.read_parquet(POLISH / "confirm" / "janjul_unmatched_components_cap_CAP_B.parquet").select("MVT_ID_mvt", pl.col("expert_capped").alias("e_cap"))
    f = pl.read_parquet(POLISH / "frames" / "janjul_unmatched_s1_features_nolabel.parquet").select("MVT_ID_mvt", "nm_unmatched", "type_missing", "takeoff_minus_sched_days", "takeoff_minus_sched_sec")
    s = pl.read_parquet(SENSOR / "confirm" / "janjul_unmatched_components.parquet").select("MVT_ID_mvt", pl.col("control").alias("sens_control"), pl.col("candidate").alias("sens_usens"), "S", "adsb_coverage")
    raw = pl.scan_parquet([_raw_file(m) for m in JJ_MONTHS]).select("MVT_ID_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt").collect()
    lin = pl.read_parquet(NONCOPY / "ranking" / "frames" / "lineage_fit_m1_10.parquet").select("MVT_ID_mvt", "source_class")
    x = c0.join(cap, on="MVT_ID_mvt").join(f, on="MVT_ID_mvt").join(s, on="MVT_ID_mvt").join(raw, on="MVT_ID_mvt", how="left").join(lin, on="MVT_ID_mvt", how="left")
    if x.height != c0.height or x["TAXITIME_SEC_mvt"].null_count() or (x["nm_unmatched"] != 1).any():
        raise RuntimeError("JAN_JUL component join failed")
    c = _components(x)
    ctl = _mixture_control(c, c.e_cap)
    e_us, _ = sensor_blend(c.e_cap, c.s, c.sd, c.cov, c.airport, c.p)
    us = _mixture_control(c, e_us)
    cur = compose(c)
    y = x["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    ident = {"capb_p1_vs_sensor_confirm_control": float(np.max(np.abs(ctl - x["sens_control"].to_numpy()))), "usens_vs_sensor_confirm_candidate": float(np.max(np.abs(us - x["sens_usens"].to_numpy())))}
    sse = {"capb_p1": float(((ctl - y) ** 2).sum()), "usens": float(((us - y) ** 2).sum()), "current_usens_g": float(((cur - y) ** 2).sum())}
    sse["delta_usens_g_vs_capb_p1"] = sse["current_usens_g"] - sse["capb_p1"]
    out = x.select(*COMP_COLS, "TAXITIME_SEC_mvt", "source_class", "MVT_TIME_UTC_mvt").with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"), pl.Series("capb_p1", ctl), pl.Series("current", cur)
    )
    return out, {"identity_max_abs_diff": ident, "sse": sse, "n": out.height}


def _build_ranking() -> tuple[pl.DataFrame, dict[str, Any]]:
    o = pl.read_parquet(USENS_G_OVERLAY)
    rk = pl.read_parquet(NONCOPY / "ranking" / "frames" / "ranking_unmatched.parquet").select("MVT_ID_mvt", "type_missing", "takeoff_minus_sched_days", "nm_unmatched", pl.col("takeoff_minus_sched_sec").alias("sd_rk"))
    x = o.join(rk, on="MVT_ID_mvt", how="left")
    if x.height != 5283 or x["type_missing"].null_count() or (x["nm_unmatched"] != 1).any():
        raise RuntimeError("ranking component join failed")
    c = _components(x)
    cur = compose(c)
    sub = pl.read_parquet(USENS_G_SUB)
    ident = {
        "current_vs_usens_g_overlay_prediction": float(np.max(np.abs(cur - x["prediction"].to_numpy()))),
        "current_vs_usens_g_submission": float(np.max(np.abs(cur - _aligned(x, sub, "TAXITIME_SEC_mvt")))),
        "sd_overlay_vs_frame": float(np.nanmax(np.abs(x["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy() - x["sd_rk"].cast(pl.Float64).to_numpy()))),
    }
    out = x.select(*COMP_COLS).with_columns(pl.Series("current", cur))
    return out, {"identity_max_abs_diff": ident, "n": out.height, "usens_g_submission_sha256": _sha(USENS_G_SUB)}


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

LINK_READ = sorted(set(LINK_DEP_COLS) | set(LINK_ARR_COLS))
ARR_RATE_READ = ["PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]


def phase_prepare() -> None:
    _require_memory("prepare")
    if _sha(USENS_G_SUB) != USENS_G_SUB_SHA:
        raise RuntimeError("U-SENS-G submission sha mismatch")
    (OUT / "control").mkdir(parents=True, exist_ok=True)
    (OUT / "links").mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "system": "CAP_B + P1 + U-SENS-G (S ceiling 3606), E17/E18 + clip", "files": {}}
    for name, fn in (("D", _build_d), ("JAN_JUL_2025", _build_jj), ("ranking", _build_ranking)):
        frame, info = fn()
        bad = {k: v for k, v in info["identity_max_abs_diff"].items() if v > 1e-6}
        if bad:
            raise RuntimeError(f"{name}: identity check failed {bad}")
        path = OUT / "control" / f"current_{name}.parquet"
        frame.write_parquet(path)
        manifest["files"][name] = {"path": str(path), "sha256": _sha(path), "rows": frame.height, **info}
        _log(f"{name}: {frame.height} rows; identity {info['identity_max_abs_diff']}; sse {info.get('sse')}")
    manifest["inputs"] = {
        str(p.relative_to(MAIN)): _sha(p)
        for p in (
            POLISH / "oof" / "arms_D.parquet",
            POLISH / "frames" / "control_D.parquet",
            SENSOR / "oof" / "arms_D.parquet",
            SENSOR / "frames" / "control_D.parquet",
            ADSB_D,
            NONCOPY / "frames" / "d_unmatched_lineage_iso.parquet",
            POLISH / "confirm" / "janjul_unmatched_components.parquet",
            POLISH / "confirm" / "janjul_unmatched_components_cap_CAP_B.parquet",
            POLISH / "frames" / "janjul_unmatched_s1_features_nolabel.parquet",
            SENSOR / "confirm" / "janjul_unmatched_components.parquet",
            NONCOPY / "ranking" / "frames" / "lineage_fit_m1_10.parquet",
            NONCOPY / "ranking" / "frames" / "ranking_unmatched.parquet",
            USENS_G_OVERLAY,
            USENS_G_SUB,
        )
    }
    # label-free: destination-arrival links (DEP/ARR clock and identity columns only)
    links = {}
    for name, files in (("m1_10", [_raw_file(m) for m in MONTHS_1_10]), ("ranking", [RAW / "ranking.parquet"])):
        raw = pl.scan_parquet(files).select(LINK_READ).collect()
        lk, counts = destination_links(raw)
        path = OUT / "links" / f"destlink_{name}.parquet"
        lk.write_parquet(path)
        links[name] = {"path": str(path), "sha256": _sha(path), "rows": lk.height, "counts": counts}
        _log(f"links {name}: {counts}")
    manifest["links"] = links
    # ARR-only LIRF BLOCK==SCHED month rates (2025 months 1-10; 2026 ranking months)
    r25 = arr_eq_rates(pl.scan_parquet([_raw_file(m) for m in MONTHS_1_10]).select(ARR_RATE_READ).collect())
    r26 = arr_eq_rates(pl.scan_parquet(RAW / "ranking.parquet").select(ARR_RATE_READ).collect())
    manifest["lirf_arr_eq"] = {"2025": {f"{y}-{m:02d}": v for (y, m), v in r25.items() if y == 2025 and m in MONTHS_1_10}, "2026": {f"{y}-{m:02d}": v for (y, m), v in r26.items() if y == 2026}}
    _log(f"arr_eq: {manifest['lirf_arr_eq']}")
    _write_json(OUT / "control" / "manifest.json", manifest)
    _write_json(TRACKED / "control_manifest.json", manifest)


def _load(name: str) -> tuple[pl.DataFrame, Components]:
    x = pl.read_parquet(OUT / "control" / f"current_{name}.parquet")
    lk = pl.read_parquet(OUT / "links" / ("destlink_ranking.parquet" if name == "ranking" else "destlink_m1_10.parquet"))
    c = attach_links(_components(x), lk)
    cur = compose(c)
    if np.max(np.abs(cur - x["current"].to_numpy())) > 1e-6:
        raise RuntimeError(f"{name}: control does not reproduce the persisted file")
    return x, c


def _arr_eq(manifest: dict[str, Any], year: int) -> dict[int, float]:
    return {int(k.split("-")[1]): float(v["rate"]) for k, v in manifest["lirf_arr_eq"][str(year)].items()}


def _copy_rates(frames: list[pl.DataFrame], months: tuple[int, ...]) -> dict[int, float]:
    x = pl.concat([f.select("month", "AIRPORT", "takeoff_minus_sched_sec", "source_class") for f in frames])
    x = x.filter((pl.col("AIRPORT") == "LIRF") & (pl.col("takeoff_minus_sched_sec") >= 3600) & pl.col("month").is_in(months))
    g = x.group_by("month").agg((pl.col("source_class") == "schedule_copy").mean().alias("r"))
    return {int(m): float(r) for m, r in g.iter_rows()}


def _params(dx: pl.DataFrame, dc: Components, jx: pl.DataFrame, jc: Components, rc: Components, manifest: dict[str, Any]) -> dict[str, Any]:
    """Label-derived arm parameters, each fitted only on months outside the population it is applied to."""

    yd = dx["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    yj = jx["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    fold = dx["fold"].to_numpy()
    # Arm A
    q_d, q_by_fold = arm_a_q_crossfit(dc, yd, fold)
    q_jj = float(near_copy(yd, dc.sd)[dc.e17].mean())
    nc_all = np.concatenate([near_copy(yd, dc.sd)[dc.e17], near_copy(yj, jc.sd)[jc.e17]])
    q_rank = float(nc_all.mean())
    # Arm B
    arr25, arr26 = _arr_eq(manifest, 2025), _arr_eq(manifest, 2026)
    rate = _copy_rates([dx, jx], MONTHS_1_10)
    cell_d = dc.lirf & dc.gated
    targets_d = {}
    for m in D_MONTHS:
        a, b = fit_rate_map(arr25, rate, [k for k in D_MONTHS if k != m])
        targets_d[m] = rate_target(a, b, arr25[m])
    shift_d, info_d = month_logit_shift(dc.p, cell_d, dc.month, targets_d)
    a8, b8 = fit_rate_map(arr25, rate, list(D_MONTHS))
    targets_jj = {m: rate_target(a8, b8, arr25[m]) for m in JJ_MONTHS}
    shift_jj, info_jj = month_logit_shift(jc.p, jc.lirf & jc.gated, jc.month, targets_jj)
    a10, b10 = fit_rate_map(arr25, rate, list(MONTHS_1_10))
    targets_rank = {m: rate_target(a10, b10, arr26[m]) for m in JJ_MONTHS}
    shift_rank, info_rank = month_logit_shift(rc.p, rc.lirf & rc.gated, rc.month, targets_rank)
    return {
        "A": {"q_D_crossfit_by_fold": q_by_fold, "q_JAN_JUL": q_jj, "q_ranking": q_rank, "n_cell": {"D": int(dc.e17.sum()), "JAN_JUL": int(jc.e17.sum()), "ranking": int(rc.e17.sum())}},
        "B": {
            "copy_rate_2025": rate,
            "arr_eq_2025": arr25,
            "arr_eq_2026": arr26,
            "D_lomo": info_d,
            "JAN_JUL_fit_D_months": {"a": a8, "b": b8, "months": info_jj},
            "ranking_fit_months_1_10": {"a": a10, "b": b10, "months": info_rank},
            "ranking_alt_fit_D_months_only_(disclosed_not_used)": {m: rate_target(a8, b8, arr26[m]) for m in JJ_MONTHS},
        },
        "_arrays": {"q_d": q_d, "shift_d": shift_d, "shift_jj": shift_jj, "shift_rank": shift_rank},
    }


def _spec(arms: tuple[str, ...], pop: str, prm: dict[str, Any]) -> ArmSpec:
    arr = prm["_arrays"]
    a_q = None
    b = None
    if "A" in arms:
        a_q = {"D": arr["q_d"], "JAN_JUL": prm["A"]["q_JAN_JUL"], "ranking": prm["A"]["q_ranking"]}[pop]
    if "B" in arms:
        b = {"D": arr["shift_d"], "JAN_JUL": arr["shift_jj"], "ranking": arr["shift_rank"]}[pop]
    return ArmSpec(a_q=a_q, b_logit_shift=b, c="C" in arms, d="D" in arms, c0="C0" in arms)


def _scope(arms: tuple[str, ...], c: Components) -> np.ndarray:
    m = np.zeros(c.n, dtype=bool)
    if "A" in arms:
        m |= c.e17
    if "B" in arms:
        m |= c.lirf & c.gated
    if "C" in arms:
        m |= c.linked
    if "D" in arms:
        m |= ~c.lirf
    if "C0" in arms:
        m |= c.in_window
    return m


def _eval(arms: tuple[str, ...], dx, dc, jx, jc, prm) -> dict[str, Any]:
    yd = dx["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    yj = jx["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    ctl_d, ctl_j = dx["current"].to_numpy(), jx["current"].to_numpy()
    cand_d = compose(dc, _spec(arms, "D", prm))
    cand_j = compose(jc, _spec(arms, "JAN_JUL", prm))
    rep = arm_gates(yd, ctl_d, cand_d, day=dx["day"].to_numpy(), fold=dx["fold"].to_numpy(), airport=dc.airport, lirf_copy=None)
    out_d = np.abs(cand_d - ctl_d)[~_scope(arms, dc)]
    rep["gates"]["G4_outside_scope_unchanged"] = {"value": float(out_d.max()) if out_d.size else 0.0, "pass": bool(out_d.size == 0 or out_d.max() == 0.0)}
    dj = (cand_j - yj) ** 2 - (ctl_j - yj) ** 2
    jj = {
        "n": int(yj.size),
        "rows_changed": int((np.abs(cand_j - ctl_j) > 1e-9).sum()),
        "ctl_sse": float(((ctl_j - yj) ** 2).sum()),
        "delta_sse": float(dj.sum()),
        "delta_sse_by_month": {int(m): float(dj[jc.month == m].sum()) for m in JJ_MONTHS},
        "delta_sse_by_airport": {str(a): float(dj[jc.airport == a].sum()) for a in sorted(set(jc.airport.tolist()))},
        "approx_full_rmse_change_s": float(np.sqrt(JJ_BASE_RMSE**2 + dj.sum() / JJ_N_FULL) - JJ_BASE_RMSE),
        "top5_share_of_delta": float(np.sort(dj)[:5].sum() / dj.sum()) if dj.sum() < 0 else None,
    }
    dd = (cand_d - yd) ** 2 - (ctl_d - yd) ** 2
    rep["top5_share_of_delta"] = float(np.sort(dd)[:5].sum() / dd.sum()) if dd.sum() < 0 else None
    rep["approx_public_s"] = None
    g = rep["gates"]
    if arms == ("D",):
        verdict = {
            "D_no_harm": {"value": rep["delta_sse"], "pass": bool(rep["delta_sse"] <= D_HARM_TOL)},
            "JAN_JUL_no_harm": {"value": jj["delta_sse"], "pass": bool(jj["delta_sse"] <= D_HARM_TOL)},
            "G3_no_airport_gt_2pct": g["G3_no_airport_gt_2pct"],
            "G4_outside_scope_unchanged": g["G4_outside_scope_unchanged"],
        }
    else:
        verdict = {
            "G0_delta_sse_lt_0": {"value": rep["delta_sse"], "pass": bool(rep["delta_sse"] < 0)},
            "G1_bootstrap_p95_lt_0": g["G1_bootstrap_p95_lt_0"],
            "G2_folds_ge_3": g["G2_folds_ge_3"],
            "G3_no_airport_gt_2pct": g["G3_no_airport_gt_2pct"],
            "G4_outside_scope_unchanged": g["G4_outside_scope_unchanged"],
            "G5_JAN_JUL_not_worse": {"value": jj["delta_sse"], "pass": bool(jj["delta_sse"] <= 0)},
        }
    decision = "GO" if all(v["pass"] for v in verdict.values()) else "NO-GO"
    return {"arms": list(arms), "decision": decision, "verdict": verdict, "D": rep, "JAN_JUL_2025": jj}


def phase_score() -> None:
    prereg = TRACKED / "preregistration.json"
    if not prereg.exists():
        raise SystemExit("pre-registration missing")
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", str(prereg)], capture_output=True, text=True).stdout.strip()
    if not commit:
        raise SystemExit("pre-registration is not committed")
    manifest = json.loads((OUT / "control" / "manifest.json").read_text())
    dx, dc = _load("D")
    jx, jc = _load("JAN_JUL_2025")
    rx, rc = _load("ranking")
    prm = _params(dx, dc, jx, jc, rc, manifest)
    results: dict[str, Any] = {"preregistration_commit": commit, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "params": {k: v for k, v in prm.items() if k != "_arrays"}, "arms": {}}
    for arm in ARMS:
        results["arms"][arm] = _eval((arm,), dx, dc, jx, jc, prm)
        r = results["arms"][arm]
        _log(f"arm {arm}: {r['decision']} D dSSE {r['D']['delta_sse']:.4e} p95 {r['D']['bootstrap']['p95']:.3e} folds {r['D']['gates']['G2_folds_ge_3']['value']}/4 worst ap {r['D']['gates']['G3_no_airport_gt_2pct']['value']:+.4f} | JJ dSSE {r['JAN_JUL_2025']['delta_sse']:.4e} (~{r['JAN_JUL_2025']['approx_full_rmse_change_s']:+.3f} s)")
    accepted = tuple(a for a in ARMS if results["arms"][a]["decision"] == "GO")
    results["accepted"] = list(accepted)
    if len(accepted) >= 2:
        results["arms"]["ALL"] = _eval(accepted, dx, dc, jx, jc, prm)
        r = results["arms"]["ALL"]
        _log(f"ALL {accepted}: {r['decision']} D dSSE {r['D']['delta_sse']:.4e} p95 {r['D']['bootstrap']['p95']:.3e} | JJ dSSE {r['JAN_JUL_2025']['delta_sse']:.4e} (~{r['JAN_JUL_2025']['approx_full_rmse_change_s']:+.3f} s)")
    _write_json(OUT / "metrics" / "results.json", results)
    _write_json(TRACKED / "results.json", results)


def phase_ranking() -> None:
    from prc_challenge.io import scan_submitting
    from prc_challenge.unmatched_recovery.harness import RANKING_N, RANKING_UNMATCHED_N, overlay_unmatched_submission, verify_arm_submission

    res = json.loads((OUT / "metrics" / "results.json").read_text())
    manifest = json.loads((OUT / "control" / "manifest.json").read_text())
    dx, dc = _load("D")
    jx, jc = _load("JAN_JUL_2025")
    rx, rc = _load("ranking")
    prm = _params(dx, dc, jx, jc, rc, manifest)
    base = pl.read_parquet(USENS_G_SUB)
    template = scan_submitting(RAW).collect()
    ctl = rx["current"].to_numpy()
    deliver = [a for a in res["accepted"]]
    if "ALL" in res["arms"] and res["arms"]["ALL"]["decision"] == "GO":
        deliver.append("ALL")
    (OUT / "ranking").mkdir(parents=True, exist_ok=True)
    index = {}
    for name in deliver:
        arms = tuple(res["accepted"]) if name == "ALL" else (name,)
        cand = compose(rc, _spec(arms, "ranking", prm))
        if not np.isfinite(cand).all() or (cand < 0).any():
            raise RuntimeError(f"{name}: invalid ranking predictions")
        outside = np.abs(cand - ctl)[~_scope(arms, rc)]
        if outside.size and outside.max() > 0:
            raise RuntimeError(f"{name}: rows outside the arm scope changed")
        ov = pl.DataFrame({"MVT_ID_mvt": rc.mvt_id, "prediction": cand})
        sub = overlay_unmatched_submission(base, ov)
        verify = verify_arm_submission(sub, template=template, parent=base, unmatched_ids=pl.Series(rc.mvt_id), expected_unmatched_n=RANKING_UNMATCHED_N, expected_n=RANKING_N)
        j = base.join(sub, on="MVT_ID_mvt", suffix="_new")
        diff = j.filter((pl.col("TAXITIME_SEC_mvt") - pl.col("TAXITIME_SEC_mvt_new")).abs() > 1e-9)
        if not set(diff["MVT_ID_mvt"].to_list()) <= set(rc.mvt_id.tolist()):
            raise RuntimeError(f"{name}: matched rows changed")
        path = OUT / "ranking" / f"unmatched_submission_{name}.parquet"
        sub.write_parquet(path)
        ch = changed_by_airport(ctl, cand, rc.airport)
        month_ch = {int(m): int(((np.abs(cand - ctl) > 1e-9) & (rc.month == m)).sum()) for m in JJ_MONTHS}
        man = {
            "name": f"SA-UNM4 arm {name} unmatched rows on CAP_B + P1 + U-SENS-G",
            "arms": list(arms),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "file": {"path": str(path), "sha256": _sha(path), "rows": sub.height, "use": "--unmatched input of scripts/compose/compose_candidate.py (unmatched rows already E17/E18 + clip composed)"},
            "parent": {"path": str(USENS_G_SUB), "sha256": _sha(USENS_G_SUB)},
            "rows_changed_vs_usens_g": ch["changed"],
            "rows_changed_by_month": month_ch,
            "sum_sq_change": ch["sum_sq_change"],
            "mean_change_by_month": {int(m): float((cand - ctl)[rc.month == m].mean()) for m in JJ_MONTHS},
            "mean_change_on_changed_by_month": {int(m): float((cand - ctl)[(rc.month == m) & (np.abs(cand - ctl) > 1e-9)].mean()) if ((rc.month == m) & (np.abs(cand - ctl) > 1e-9)).any() else 0.0 for m in JJ_MONTHS},
            "by_airport": ch["by_airport"],
            "d_decision": res["arms"][name]["decision"],
            "validator": {"ok": verify["ok"], "errors": verify["errors"], "matched_changed": verify.get("matched_changed")},
            "uploaded": False,
            "eligibility": "D-054 (destination-arrival LOBT/AOBT only, never its BLOCK/taxi-in) for arm C; LIRF ARR BLOCK==SCHED month rates (ARR rows only) for arm B; no DEP BLOCK/TAXITIME features; no Nov/Dec labels",
        }
        if not verify["ok"]:
            raise RuntimeError(f"{name}: validator failed {verify['errors']}")
        _write_json(OUT / "ranking" / f"unmatched_submission_{name}.manifest.json", man)
        index[name] = man
        _log(f"ranking {name}: changed {ch['changed']} sha {man['file']['sha256']}")
    _write_json(TRACKED / "ranking_manifest.json", index)


AMEND1 = TRACKED / "preregistration_amendment_1_c0.json"


def phase_score_c0() -> None:
    """Amendment 1 (D-056): C0 on top of ALL, gated on JAN_JUL_2025 only."""

    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", str(AMEND1)], capture_output=True, text=True).stdout.strip()
    if not commit:
        raise SystemExit("amendment 1 is not committed")
    res = json.loads((OUT / "metrics" / "results.json").read_text())
    if res["arms"].get("ALL", {}).get("decision") != "GO":
        raise SystemExit("ALL is not GO")
    base_arms = tuple(res["accepted"])
    manifest = json.loads((OUT / "control" / "manifest.json").read_text())
    dx, dc = _load("D")
    jx, jc = _load("JAN_JUL_2025")
    rx, rc = _load("ranking")
    prm = _params(dx, dc, jx, jc, rc, manifest)
    yj = jx["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    all_j = compose(jc, _spec(base_arms, "JAN_JUL", prm))
    new_j = compose(jc, _spec((*base_arms, "C0"), "JAN_JUL", prm))
    outside = np.abs(new_j - all_j)[~jc.in_window]
    se0, se1 = (all_j - yj) ** 2, (new_j - yj) ** 2
    d = se1 - se0
    aps = {}
    for a in sorted(set(jc.airport.tolist())):
        m = jc.airport == a
        aps[a] = {"n": int(m.sum()), "rows_changed": int((np.abs(new_j - all_j)[m] > 1e-9).sum()), "all_sse": float(se0[m].sum()), "delta_sse": float(d[m].sum()), "delta_frac": float(d[m].sum() / se0[m].sum()) if se0[m].sum() > 0 else 0.0}
    worst = max(v["delta_frac"] for v in aps.values())
    gates = {
        "J1_JAN_JUL_delta_sse_lt_0": {"value": float(d.sum()), "pass": bool(d.sum() < 0)},
        "J2_no_airport_gt_2pct": {"value": worst, "pass": bool(worst <= 0.02)},
        "sanity_outside_scope_unchanged": {"value": float(outside.max()) if outside.size else 0.0, "pass": bool(outside.size == 0 or outside.max() == 0.0)},
    }
    decision = "GO" if all(g["pass"] for g in gates.values()) else "NO-GO"
    # disclosed only (D was seen before registration): D effect of C0 on top of ALL
    yd = dx["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    all_d = compose(dc, _spec(base_arms, "D", prm))
    new_d = compose(dc, _spec((*base_arms, "C0"), "D", prm))
    dd = (new_d - yd) ** 2 - (all_d - yd) ** 2
    out = {
        "amendment_commit": commit,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "control": f"ALL = {'+'.join(base_arms)}",
        "decision": decision,
        "gates": gates,
        "JAN_JUL_2025": {
            "n": int(yj.size),
            "rows_changed": int((np.abs(new_j - all_j) > 1e-9).sum()),
            "in_window_rows": int(jc.in_window.sum()),
            "all_sse": float(se0.sum()),
            "delta_sse": float(d.sum()),
            "delta_sse_by_month": {int(m): float(d[jc.month == m].sum()) for m in JJ_MONTHS},
            "approx_full_rmse_change_s": float(np.sqrt(JJ_BASE_RMSE**2 + d.sum() / JJ_N_FULL) - JJ_BASE_RMSE),
            "all_plus_c0_vs_usens_g_delta_sse": float((se1 - (jx["current"].to_numpy() - yj) ** 2).sum()),
            "top5_share_of_delta": float(np.sort(d)[:5].sum() / d.sum()) if d.sum() < 0 else None,
            "airports": aps,
        },
        "D_disclosed_not_gated": {"delta_sse": float(dd.sum()), "rows_changed": int((np.abs(new_d - all_d) > 1e-9).sum())},
    }
    _write_json(OUT / "metrics" / "results_c0.json", out)
    _write_json(TRACKED / "results_c0.json", out)
    _log(f"C0 on ALL: {decision} JJ dSSE {d.sum():.4e} (~{out['JAN_JUL_2025']['approx_full_rmse_change_s']:+.3f} s) worst airport {worst:+.4f}; D (disclosed) {dd.sum():.4e}")


def phase_ranking_c0() -> None:
    from prc_challenge.io import scan_submitting
    from prc_challenge.unmatched_recovery.harness import RANKING_N, RANKING_UNMATCHED_N, overlay_unmatched_submission, verify_arm_submission

    r0 = json.loads((OUT / "metrics" / "results_c0.json").read_text())
    if r0["decision"] != "GO":
        raise SystemExit("C0 did not pass")
    res = json.loads((OUT / "metrics" / "results.json").read_text())
    base_arms = tuple(res["accepted"])
    manifest = json.loads((OUT / "control" / "manifest.json").read_text())
    dx, dc = _load("D")
    jx, jc = _load("JAN_JUL_2025")
    rx, rc = _load("ranking")
    prm = _params(dx, dc, jx, jc, rc, manifest)
    all_r = compose(rc, _spec(base_arms, "ranking", prm))
    all_file = OUT / "ranking" / "unmatched_submission_ALL.parquet"
    if np.max(np.abs(all_r - _aligned(rx, pl.read_parquet(all_file), "TAXITIME_SEC_mvt"))) > 1e-6:
        raise RuntimeError("ALL ranking recomposition does not match the ALL file")
    new = compose(rc, _spec((*base_arms, "C0"), "ranking", prm))
    if not np.isfinite(new).all() or (new < 0).any():
        raise RuntimeError("invalid predictions")
    outside = np.abs(new - all_r)[~rc.in_window]
    if outside.size and outside.max() > 0:
        raise RuntimeError("rows outside C0 scope changed")
    base = pl.read_parquet(USENS_G_SUB)
    sub = overlay_unmatched_submission(base, pl.DataFrame({"MVT_ID_mvt": rc.mvt_id, "prediction": new}))
    verify = verify_arm_submission(sub, template=scan_submitting(RAW).collect(), parent=base, unmatched_ids=pl.Series(rc.mvt_id), expected_unmatched_n=RANKING_UNMATCHED_N, expected_n=RANKING_N)
    if not verify["ok"]:
        raise RuntimeError(f"validator failed {verify['errors']}")
    path = OUT / "ranking" / "unmatched_submission_ALL_C0.parquet"
    sub.write_parquet(path)
    vs_all = changed_by_airport(all_r, new, rc.airport)
    vs_usg = changed_by_airport(rx["current"].to_numpy(), new, rc.airport)
    man = {
        "name": "SA-UNM4 ALL + C0 (amendment 1, D-056) unmatched rows on CAP_B + P1 + U-SENS-G",
        "arms": [*base_arms, "C0"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file": {"path": str(path), "sha256": _sha(path), "rows": sub.height, "use": "--unmatched input of scripts/compose/compose_candidate.py (unmatched rows already E17/E18 + clip composed)"},
        "parent": {"path": str(USENS_G_SUB), "sha256": _sha(USENS_G_SUB)},
        "all_file": {"path": str(all_file), "sha256": _sha(all_file)},
        "in_window_rows": int(rc.in_window.sum()),
        "rows_changed_vs_ALL": vs_all["changed"],
        "rows_changed_vs_ALL_by_month": {int(m): int(((np.abs(new - all_r) > 1e-9) & (rc.month == m)).sum()) for m in JJ_MONTHS},
        "vs_ALL": vs_all,
        "rows_changed_vs_usens_g": vs_usg["changed"],
        "vs_usens_g": vs_usg,
        "c0_decision": r0["decision"],
        "validator": {"ok": verify["ok"], "errors": verify["errors"], "matched_changed": verify.get("matched_changed")},
        "uploaded": False,
        "eligibility": "D-054 (destination-arrival LOBT only for C0; never its BLOCK/taxi-in); arm B LIRF ARR BLOCK==SCHED month rates (ARR rows only); no DEP BLOCK/TAXITIME features; no Nov/Dec labels",
    }
    _write_json(OUT / "ranking" / "unmatched_submission_ALL_C0.manifest.json", man)
    _write_json(TRACKED / "ranking_manifest_ALL_C0.json", man)
    _log(f"ranking ALL_C0: changed vs ALL {vs_all['changed']}, vs U-SENS-G {vs_usg['changed']}; sha {man['file']['sha256']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "score", "ranking", "score_c0", "ranking_c0"])
    phase = ap.parse_args().phase
    {"prepare": phase_prepare, "score": phase_score, "ranking": phase_ranking, "score_c0": phase_score_c0, "ranking_c0": phase_ranking_c0}[phase]()
