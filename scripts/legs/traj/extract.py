# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-TRAJ step 1: per-departure raw ADS-B ground/air trajectories (label-free), cached once.

For every DEP row at the ten challenge airports on every stage-1 day (2025 months 1-12 and the
2026 ranking days), the departure is joined to one aircraft with the **unchanged** ADS-B block join
(``scripts/features/build_adsb_features.py``: ``departures(with_labels=False)`` -> ``normalize_departures``
-> ``with_dict_callsign`` -> ``load_samples`` -> ``join_aircraft``; join sources 1 callsign,
2 flight number, 4 dictionary callsign, 3 take-off fallback) and the linker precision gate is
applied as in ``apply_linker_gate``: fallback-joined rows (``join_src`` 3) at airports below the
95 % gate (EGLL, EHAM, LFPG) get no track.

Track = every sample of (ADEP, hex) in [MVT - 120 min, MVT + 2 min] (ground and airborne; the slim
files hold every ground sample and airborne samples thinned to 1 per 10 s).  Samples are thinned
to informative points and, if still longer than ``N_MAX``, subsampled uniformly by index (first and
last point kept):

* a point is kept when >= ``KEEP_DT_MOVE`` s after the last kept point while moving, or >=
  ``KEEP_DT_STILL`` s while stationary (gs <= 1 kt and < 20 m from the last kept point), when the
  on-ground flag or the moving state (gs > 3 kt) changes, when it is > 150 m from the last kept
  point, and always for the first and last sample.

Per point (float16, ``POINT_COLS``): minutes relative to MVT, x/y km in the airport frame (east /
north of the mean runway-threshold position), ground speed kt (NaN kept), on-ground flag, heading
sin/cos from the neighbouring kept points (0/0 if they are < 15 m apart), km to the nearest runway
threshold of the airport, and log1p of the number of raw samples the point stands for.

Per row (``rows.parquet``): MVT_ID_mvt, ADEP_mvt, day, join_src (after the linker gate), stand_key,
runway key, n_raw (samples in the window), n_pts, offset into the day's point block, and the
label-free anchors in the airport frame: OSM stand position and the assigned runway's threshold.

MVT (take-off) is the clock origin, as in the ADS-B block.  No label column is read: departures are
loaded with ``with_labels=False`` (the BLOCK column is dropped before anything else is computed; the
leakage test re-runs days with the label column blanked and asserts bit-identical output).

Output: ``<ART>/days/<day>/{rows.parquet, points.npy}`` + ``meta.json`` per day, and
``<ART>/tracks_manifest.json``.

Usage: PYTHONPATH=src .venv/bin/python <wt>/scripts/legs/traj/extract.py [--days 2025-02-03,...] [--force]

High-resolution profile (final bake, ``traj_hires``; owner decision): ``--profile hires`` keeps a moving point every
5 s (``KEEP_DT_MOVE`` 15 -> 5), a 180-min window (``W_BEFORE_S`` 120 -> 180 min) and up to 256 points (``N_MAX`` 128 ->
256); every other rule is v1's.  It writes to ``$TRAJ_ART`` (a separate store; the v1 store is never touched) and, with
any ``--profile``, ``<ART>/extract_manifest.json``.  Without ``--profile`` nothing changes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "3")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

MAIN = Path(os.environ.get("PRC_MAIN_REPO", Path(__file__).resolve().parents[3]))  # bake boxes: PRC_MAIN_REPO
ART = Path(os.environ["TRAJ_ART"]) if os.environ.get("TRAJ_ART") else MAIN / "artifacts/experiments/ADSB-trajectory-20260927"
EXTRACT_VERSION = 1

W_BEFORE_S = 120 * 60
W_AFTER_S = 120
N_MAX = 128
KEEP_DT_MOVE = 15.0
KEEP_DT_STILL = 300.0
KEEP_DIST_M = 150.0
STILL_DIST_M = 20.0
HDG_MIN_M = 15.0
MOVE_KT = 3.0
STILL_KT = 1.0
POINT_COLS = ["t_min", "x_km", "y_km", "gs_kt", "on_ground", "hdg_sin", "hdg_cos", "d_thr_km", "log_n"]
PROFILES = {
    "v1": {"W_BEFORE_S": 120 * 60, "W_AFTER_S": 120, "N_MAX": 128, "KEEP_DT_MOVE": 15.0, "KEEP_DT_STILL": 300.0, "KEEP_DIST_M": 150.0, "EXTRACT_VERSION": 1},
    "hires": {"W_BEFORE_S": 180 * 60, "W_AFTER_S": 120, "N_MAX": 256, "KEEP_DT_MOVE": 5.0, "KEEP_DT_STILL": 300.0, "KEEP_DIST_M": 150.0, "EXTRACT_VERSION": "hires1"},
}


def set_profile(name: str) -> dict:
    """Module constants of a thinning profile (read at call time by ``thin`` / ``day_tracks``)."""

    prof = PROFILES[name]
    globals().update(prof)
    return prof


def _load_bf():
    """The ADS-B block builder of the main checkout (its ROOT/OUT point at the main data)."""
    spec = importlib.util.spec_from_file_location("build_adsb_features", MAIN / "scripts" / "features" / "build_adsb_features.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(MAIN / "src"))
    spec.loader.exec_module(mod)
    return mod


bf = _load_bf()
from prc_challenge.adsb_features import day_files, join_aircraft, load_samples, normalize_departures  # noqa: E402
from prc_challenge.geometry import load_stand_coords  # noqa: E402


# --------------------------------------------------------------------------- airport frame


def airport_frames() -> dict[str, tuple[float, float]]:
    """Origin (lat0, lon0) per airport: mean of its runway thresholds (label-free, fixed)."""
    thr = bf._thresholds()
    g = thr.group_by("AIRPORT").agg(pl.col("rwy_lat").mean(), pl.col("rwy_lon").mean())
    return {a: (la, lo) for a, la, lo in g.iter_rows()}


FRAMES = airport_frames()


def to_xy(ap: str, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    la0, lo0 = FRAMES[ap]
    x = (np.asarray(lon, float) - lo0) * math.cos(math.radians(la0)) * 111.320
    y = (np.asarray(lat, float) - la0) * 110.574
    return x, y


def threshold_xy() -> dict[str, np.ndarray]:
    thr = bf._thresholds()
    out = {}
    for ap in FRAMES:
        t = thr.filter(pl.col("AIRPORT") == ap)
        x, y = to_xy(ap, t["rwy_lat"].to_numpy(), t["rwy_lon"].to_numpy())
        out[ap] = np.stack([x, y], axis=1)
    return out


THR_XY = threshold_xy()


def runway_anchor_table() -> pl.DataFrame:
    thr = bf._thresholds()
    rows = []
    for ap, rwy, la, lo in thr.select("AIRPORT", "RUNWAY_mvt", "rwy_lat", "rwy_lon").iter_rows():
        x, y = to_xy(ap, np.array([la]), np.array([lo]))
        rows.append((ap, rwy, float(x[0]), float(y[0])))
    return pl.DataFrame(rows, schema={"ADEP_mvt": pl.String, "rwy_key": pl.String, "rwy_x": pl.Float64, "rwy_y": pl.Float64}, orient="row")


def osm_anchor_table() -> pl.DataFrame:
    osm = load_stand_coords(MAIN / "data" / "external" / "osm" / "parking_positions.parquet")
    parts = []
    for ap in FRAMES:
        o = osm.filter(pl.col("AIRPORT") == ap)
        if o.height == 0:
            continue
        x, y = to_xy(ap, o["stand_lat"].to_numpy(), o["stand_lon"].to_numpy())
        parts.append(pl.DataFrame({"ADEP_mvt": [ap] * o.height, "stand_key": o["stand_key"], "osm_x": x, "osm_y": y}))
    return pl.concat(parts)


RWY_ANCHORS = runway_anchor_table()
OSM_ANCHORS = osm_anchor_table()


# --------------------------------------------------------------------------- thinning


def thin(t: np.ndarray, x: np.ndarray, y: np.ndarray, gs: np.ndarray, gnd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of kept samples and the number of raw samples each kept point stands for."""
    n = len(t)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    gsn = np.nan_to_num(gs, nan=-1.0)
    moving = gsn > MOVE_KT
    keep = [0]
    lt, lx, ly, lg, lm = t[0], x[0], y[0], gnd[0], moving[0]
    for i in range(1, n):
        d = math.hypot((x[i] - lx) * 1000.0, (y[i] - ly) * 1000.0)
        still = (gsn[i] <= STILL_KT) and d < STILL_DIST_M
        dt = t[i] - lt
        if (gnd[i] != lg) or (moving[i] != lm) or d > KEEP_DIST_M or (dt >= (KEEP_DT_STILL if still else KEEP_DT_MOVE)) or i == n - 1:
            keep.append(i)
            lt, lx, ly, lg, lm = t[i], x[i], y[i], gnd[i], moving[i]
    k = np.asarray(keep, np.int64)
    cnt = np.diff(np.concatenate([k, [n]]))  # raw samples from each kept point up to the next
    if len(k) > N_MAX:
        sel = np.unique(np.round(np.linspace(0, len(k) - 1, N_MAX)).astype(np.int64))
        # counts of the dropped points are merged into the preceding kept one
        bounds = np.concatenate([k[sel], [n]])
        cnt = np.diff(bounds)
        k = k[sel]
    return k, cnt


def heading(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    s = np.zeros(n, np.float64)
    c = np.zeros(n, np.float64)
    if n < 2:
        return s, c
    xp = np.concatenate([[x[0]], x[:-1]])
    yp = np.concatenate([[y[0]], y[:-1]])
    xn = np.concatenate([x[1:], [x[-1]]])
    yn = np.concatenate([y[1:], [y[-1]]])
    dx, dy = (xn - xp) * 1000.0, (yn - yp) * 1000.0
    d = np.hypot(dx, dy)
    ok = d >= HDG_MIN_M
    s[ok] = dx[ok] / d[ok]  # sin(heading), heading clockwise from north
    c[ok] = dy[ok] / d[ok]
    return s, c


# --------------------------------------------------------------------------- one day


def day_tracks(d: date, keep_airports: set[str], dep_override: pl.DataFrame | None = None) -> tuple[pl.DataFrame, np.ndarray]:
    files = day_files(bf.FILT, d)
    dep = bf.departures(d, with_labels=False) if dep_override is None else dep_override
    if "BLOCK_TIME_UTC_mvt" in dep.columns:
        raise RuntimeError("label column present")
    dep = bf.with_dict_callsign(normalize_departures(dep))
    s = load_samples(files)
    dep = join_aircraft(dep, s)
    gated = (pl.col("join_src") == 3) & ~pl.col("ADEP_mvt").is_in(sorted(keep_airports))
    dep = dep.with_columns(
        pl.when(gated).then(None).otherwise(pl.col("hex")).alias("hex"),
        pl.when(gated).then(0).otherwise(pl.col("join_src")).cast(pl.Int8).alias("join_src"),
        pl.col("RUNWAY_mvt").cast(pl.String).str.to_uppercase().alias("rwy_key"),
    )
    keys = s.select("airport", "hex").to_numpy()
    ts_all = s["ts"].to_numpy()
    lat_all, lon_all = s["lat"].to_numpy(), s["lon"].to_numpy()
    gnd_all = s["on_ground"].to_numpy()
    gs_all = s["gs_kt"].fill_null(np.nan).to_numpy().astype(float)
    index = {}
    if len(keys):
        change = np.flatnonzero((keys[1:, 0] != keys[:-1, 0]) | (keys[1:, 1] != keys[:-1, 1])) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(keys)]))
        index = {(keys[a, 0], keys[a, 1]): (a, b) for a, b in zip(starts, ends)}
    recs, blocks, off = [], [], 0
    for mid, ap, hx, mvt in dep.select("MVT_ID_mvt", "ADEP_mvt", "hex", "mvt").iter_rows():
        rec = {"MVT_ID_mvt": mid, "n_raw": 0, "n_pts": 0, "offset": off}
        span = index.get((ap, hx)) if hx is not None else None
        if span is None or ap not in FRAMES:
            recs.append(rec)
            continue
        a, b = span
        lo = a + np.searchsorted(ts_all[a:b], mvt - W_BEFORE_S)
        hi = a + np.searchsorted(ts_all[a:b], mvt + W_AFTER_S, side="right")
        if hi <= lo:
            recs.append(rec)
            continue
        t = ts_all[lo:hi]
        x, y = to_xy(ap, lat_all[lo:hi], lon_all[lo:hi])
        gs, gnd = gs_all[lo:hi], gnd_all[lo:hi]
        k, cnt = thin(t, x, y, gs, gnd)
        xk, yk = x[k], y[k]
        hs, hc = heading(xk, yk)
        thr = THR_XY[ap]
        dthr = np.sqrt(((xk[:, None] - thr[None, :, 0]) ** 2 + (yk[:, None] - thr[None, :, 1]) ** 2).min(axis=1))
        pts = np.stack([(t[k] - mvt) / 60.0, xk, yk, gs[k], gnd[k].astype(float), hs, hc, dthr, np.log1p(cnt)], axis=1)
        blocks.append(pts.astype(np.float16))
        rec.update({"n_raw": int(hi - lo), "n_pts": int(len(k))})
        off += len(k)
        recs.append(rec)
    rows = pl.DataFrame(recs, schema={"MVT_ID_mvt": pl.Float64, "n_raw": pl.Int32, "n_pts": pl.Int32, "offset": pl.Int64})
    rows = dep.select("MVT_ID_mvt", "ADEP_mvt", "join_src", "stand_key", "rwy_key", "mvt").join(rows, on="MVT_ID_mvt", how="left", maintain_order="left")
    rows = (
        rows.join(OSM_ANCHORS, on=["ADEP_mvt", "stand_key"], how="left", maintain_order="left")
        .join(RWY_ANCHORS, on=["ADEP_mvt", "rwy_key"], how="left", maintain_order="left")
        .with_columns(pl.lit(d.isoformat()).alias("day"))
    )
    points = np.concatenate(blocks) if blocks else np.zeros((0, len(POINT_COLS)), np.float16)
    return rows, points


def all_days() -> list[date]:
    root = bf.OUT / "stage1"
    return sorted(date.fromisoformat(p.name) for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists())


def keep_airports() -> set[str]:
    lp = json.loads((bf.OUT / "features" / "linker_precision.json").read_text())
    return {ap for ap, v in lp["airports"].items() if v["keep_fallback"]}


def stage1_check(d: date, rows: pl.DataFrame, keep_ap: set[str]) -> dict:
    """Agreement of the join with the ADS-B block's stage-1 rows (same code; same linker gate)."""
    st = pl.read_parquet(bf.OUT / "stage1" / d.isoformat() / "rows.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "join_src"])
    st = st.with_columns(pl.when((pl.col("join_src") == 3) & ~pl.col("ADEP_mvt").is_in(sorted(keep_ap))).then(0).otherwise(pl.col("join_src")).cast(pl.Int8).alias("js_gated"))
    j = rows.select("MVT_ID_mvt", "join_src").join(st.select("MVT_ID_mvt", "js_gated"), on="MVT_ID_mvt", how="full", coalesce=True)
    return {"rows": rows.height, "stage1_rows": st.height, "join_src_mismatch": int((j["join_src"].fill_null(-1) != j["js_gated"].fill_null(-1)).sum())}


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--profile", choices=sorted(PROFILES), default=None, help="thinning profile (hires: 5 s, 256 points, 180 min); default: v1, no manifest")
    a = p.parse_args(argv)
    prof = set_profile(a.profile) if a.profile else None
    days = [date.fromisoformat(x) for x in a.days.split(",") if x] or all_days()
    keep_ap = keep_airports()
    root = ART / "days"
    t_all = time.time()
    for d in days:
        out = root / d.isoformat()
        if (out / "meta.json").exists() and not a.force:
            m = json.loads((out / "meta.json").read_text())
            if m.get("version") == EXTRACT_VERSION:
                continue
        t0 = time.time()
        rows, points = day_tracks(d, keep_ap)
        out.mkdir(parents=True, exist_ok=True)
        rows.write_parquet(out / "rows.parquet")
        np.save(out / "points.npy", points)
        chk = stage1_check(d, rows, keep_ap)
        meta = {"version": EXTRACT_VERSION, "day": d.isoformat(), "scope": bf.scope_of(d), **chk,
                "tracked": int((rows["n_pts"].fill_null(0) > 0).sum()), "points": int(points.shape[0]),
                "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(), "secs": round(time.time() - t0, 1)}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{d} {meta['scope']}: rows={meta['rows']} tracked={meta['tracked']} pts={meta['points']} mismatch={chk['join_src_mismatch']} {meta['secs']}s", flush=True)
    print(f"done {len(days)} days in {time.time() - t_all:.0f}s", flush=True)
    if prof is not None:
        metas = [json.loads((root / d.isoformat() / "meta.json").read_text()) for d in days]
        man = {"profile": a.profile, **prof, "art": str(ART), "days": len(days), "first_day": days[0].isoformat() if days else None,
               "last_day": days[-1].isoformat() if days else None, "rows": sum(m["rows"] for m in metas), "tracked": sum(m["tracked"] for m in metas),
               "points": sum(m["points"] for m in metas), "join_src_mismatch": sum(m["join_src_mismatch"] for m in metas),
               "label_read": False, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (ART / "extract_manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
