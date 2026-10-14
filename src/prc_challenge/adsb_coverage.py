# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""ADS-B coverage extension (v3 block): off-block estimates for departures without a stand gate.

Works on the stage-1 departure ground blocks (``seg``: MVT_ID_mvt, t_rel, lat, lon, gs) written
by ``scripts/features/build_adsb_features.py stage1``. Three families of label-free-at-inference columns:

1. **Position-derived kinematics** (broadcast ground speed is unreliable on aprons): speed and
   track from successive positions; motion onset after the last position dwell, first large
   track reversal (pushback turn).
2. **Travel-time map** (backward extrapolation): for each (airport, stand, 50 m cell, heading
   sector) and (airport, cell, heading) the median time since the labelled BLOCK at which
   departing aircraft pass that cell, learned from pool months other than the row's month.
   A row's samples on the taxiway then give off-block estimates ``t_i - elapsed(cell_i)``.
3. **Runway queue**: time the aircraft first came within the holding area of its departure
   runway threshold (label-free), stop counts and stationary time in the last part of the taxi.

Only the map uses labels (other months' D-month BLOCK times, like the stand centroids). Licence:
ODbL 1.0 data, "Contains data from ADSB.lol (https://adsb.lol), licensed under ODbL 1.0".
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

EARTH_R = 6_371_000.0
M_PER_DEG = math.pi * EARTH_R / 180.0
CELL_M = 50.0
HDG_SECTORS = 8
MOVE_MS = 2.0  # derived speed above which a sample counts as moving (~3.9 kt)
DWELL_R_M = 30.0
DWELL_MIN_S = 180.0
MAP_MIN_N_STAND = 4
MAP_MIN_N_AP = 8
MAP_TRAIN_LO_S = -1800.0  # training samples: elapsed since BLOCK in [lo, hi]
MAP_TRAIN_HI_S = 3600.0
HOLD_R_M = 600.0

# airport reference points (lat, lon) for a local metric grid; values only set the grid origin
REF = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}


def _ref_frame() -> pl.DataFrame:
    return pl.DataFrame({"ADEP_mvt": list(REF), "ref_lat": [v[0] for v in REF.values()],
                         "ref_lon": [v[1] for v in REF.values()]})


def kinematics(seg: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    """Per-sample local x/y (m), derived speed (m/s) and track (deg) from the neighbours i-1, i+1.

    ``rows`` supplies ADEP_mvt per MVT_ID_mvt. Output adds x, y, v_ms, trk, cx, cy, hs, moving.
    """
    s = (
        seg.join(rows.select("MVT_ID_mvt", "ADEP_mvt"), on="MVT_ID_mvt", how="inner")
        .join(_ref_frame(), on="ADEP_mvt", how="inner")
        .sort("MVT_ID_mvt", "t_rel")
        .with_columns(
            ((pl.col("lon") - pl.col("ref_lon")) * M_PER_DEG * (pl.col("ref_lat") * math.pi / 180).cos()).alias("x"),
            ((pl.col("lat") - pl.col("ref_lat")) * M_PER_DEG).alias("y"),
        )
    )
    prev = [pl.col(c).shift(1).over("MVT_ID_mvt") for c in ("x", "y", "t_rel")]
    nxt = [pl.col(c).shift(-1).over("MVT_ID_mvt") for c in ("x", "y", "t_rel")]
    px, py, pt = [p.fill_null(pl.col(c)) for p, c in zip(prev, ("x", "y", "t_rel"))]
    nx, ny, nt = [p.fill_null(pl.col(c)) for p, c in zip(nxt, ("x", "y", "t_rel"))]
    dx, dy, dt = nx - px, ny - py, (nt - pt).cast(pl.Float64)
    s = s.with_columns(
        pl.when(dt > 0).then((dx.pow(2) + dy.pow(2)).sqrt() / dt).otherwise(None).alias("v_ms"),
        ((pl.arctan2(dx, dy) * 180 / math.pi) % 360).alias("trk"),
        (dx.pow(2) + dy.pow(2)).sqrt().alias("_disp"),
    )
    sector = 360.0 / HDG_SECTORS
    return s.with_columns(
        (pl.col("x") / CELL_M).floor().cast(pl.Int32).alias("cx"),
        (pl.col("y") / CELL_M).floor().cast(pl.Int32).alias("cy"),
        pl.when(pl.col("_disp") >= 8.0).then(((pl.col("trk") + sector / 2) % 360 / sector).floor())
        .otherwise(None).cast(pl.Int8).alias("hs"),
        pl.when(pl.col("v_ms").is_not_null() & (pl.col("v_ms") > 0)).then(pl.col("v_ms") >= MOVE_MS)
        .otherwise(pl.col("gs").fill_nan(None).fill_null(0.0) > 3.0).alias("moving"),
    ).drop("_disp", "ref_lat", "ref_lon")


def travel_map(k: pl.DataFrame, labels: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(stand-level, airport-level, airport heading-free) maps of elapsed time since BLOCK at moving samples.

    ``k``: kinematics of labelled rows (pool months other than the target month only).
    ``labels``: MVT_ID_mvt, block_rel, stand_key.
    """
    e = (
        k.filter(pl.col("moving"))
        .join(labels.select("MVT_ID_mvt", "block_rel", "stand_key"), on="MVT_ID_mvt", how="inner")
        .with_columns((pl.col("t_rel") - pl.col("block_rel")).alias("el"))
        .filter(pl.col("el").is_between(MAP_TRAIN_LO_S, MAP_TRAIN_HI_S))
    )
    # one vote per (row, key) so dense-sampled rows do not dominate
    def agg(keys: list[str], min_n: int) -> pl.DataFrame:
        return (
            e.group_by(["MVT_ID_mvt", *keys]).agg(pl.col("el").median())
            .group_by(keys)
            .agg(pl.col("el").median().alias("m_el"), pl.len().alias("m_n"),
                 pl.col("el").quantile(0.25).alias("q25"), pl.col("el").quantile(0.75).alias("q75"))
            .filter(pl.col("m_n") >= min_n)
            .with_columns(((pl.col("q75") - pl.col("q25")) / 1.349).alias("m_sd")).drop("q25", "q75")
        )
    st = agg(["ADEP_mvt", "stand_key", "cx", "cy", "hs"], MAP_MIN_N_STAND).filter(pl.col("hs").is_not_null())
    ap = agg(["ADEP_mvt", "cx", "cy", "hs"], MAP_MIN_N_AP).filter(pl.col("hs").is_not_null())
    a3 = agg(["ADEP_mvt", "cx", "cy"], MAP_MIN_N_AP)
    return st, ap, a3


def map_estimates(k: pl.DataFrame, rows: pl.DataFrame, st: pl.DataFrame, ap: pl.DataFrame,
                  a3: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-sample off-block estimates ``t_rel - m_el`` (stand map, then airport (cell, heading),
    then airport cell without heading; ``lvl`` 1/2/3)."""
    s = k.filter(pl.col("moving")).join(
        rows.select("MVT_ID_mvt", "stand_key"), on="MVT_ID_mvt", how="left")
    s = s.join(st.rename({"m_el": "s_el", "m_n": "s_n", "m_sd": "s_sd"}),
               on=["ADEP_mvt", "stand_key", "cx", "cy", "hs"], how="left")
    s = s.join(ap.rename({"m_el": "a_el", "m_n": "a_n", "m_sd": "a_sd"}), on=["ADEP_mvt", "cx", "cy", "hs"], how="left")
    if a3 is not None:
        s = s.join(a3.rename({"m_el": "h_el", "m_n": "h_n", "m_sd": "h_sd"}), on=["ADEP_mvt", "cx", "cy"], how="left")
    else:
        s = s.with_columns(pl.lit(None, pl.Float64).alias("h_el"), pl.lit(None, pl.Float64).alias("h_sd"))
    use_st, use_ap = pl.col("s_el").is_not_null(), pl.col("a_el").is_not_null()
    return s.with_columns(
        pl.when(use_st).then(pl.col("s_el")).when(use_ap).then(pl.col("a_el")).otherwise(pl.col("h_el")).alias("el"),
        pl.when(use_st).then(pl.col("s_sd")).when(use_ap).then(pl.col("a_sd")).otherwise(pl.col("h_sd")).alias("sd"),
        pl.when(use_st).then(1).when(use_ap).then(2).when(pl.col("h_el").is_not_null()).then(3).otherwise(0)
        .cast(pl.Int8).alias("lvl"),
    ).filter(pl.col("lvl") > 0).with_columns((pl.col("t_rel") - pl.col("el")).alias("est"))


# --------------------------------------------------------------------------- runway geometry (OSM, ODbL)

RWY_ZONE_M = 250.0
STOP_MS = 1.0
STOP_MIN_S = 20.0
REV_DEG = 100.0
REV_WINDOW_S = 600.0


def load_runway_lines(surface_dir) -> dict[str, list[tuple[set[str], np.ndarray]]]:
    """Per airport: [(idents, (N, 2) local x/y polyline)] from OSM ``aeroway=runway`` ways."""
    import json
    from pathlib import Path

    out: dict[str, list[tuple[set[str], np.ndarray]]] = {}
    for ap, (rlat, rlon) in REF.items():
        p = Path(surface_dir) / f"{ap.lower()}_surface.json"
        if not p.exists():
            continue
        els = json.loads(p.read_text())["elements"]
        nodes = {e["id"]: (e["lat"], e["lon"]) for e in els if e["type"] == "node" and "lat" in e}
        cosr = math.cos(rlat * math.pi / 180)
        lines = []
        for e in els:
            if e["type"] != "way" or e.get("tags", {}).get("aeroway") != "runway":
                continue
            if "geometry" in e:
                pts = [(g["lat"], g["lon"]) for g in e["geometry"]]
            else:
                pts = [nodes[n] for n in e.get("nodes", []) if n in nodes]
            if len(pts) < 2:
                continue
            arr = np.array([((lo - rlon) * M_PER_DEG * cosr, (la - rlat) * M_PER_DEG) for la, lo in pts])
            ref = e["tags"].get("ref") or ""
            idents = {s.strip().upper().lstrip("0") for s in ref.split("/") if s.strip()}
            lines.append((idents, arr))
        out[ap] = lines
    return out


def _dist_polyline(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)
    tx = ((px[:, None] - a[None, :, 0]) * ab[None, :, 0] + (py[:, None] - a[None, :, 1]) * ab[None, :, 1]) / L2
    tx = np.clip(tx, 0, 1)
    cx = a[None, :, 0] + tx * ab[None, :, 0]
    cy = a[None, :, 1] + tx * ab[None, :, 1]
    return np.sqrt((px[:, None] - cx) ** 2 + (py[:, None] - cy) ** 2).min(1)


def _angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs((a - b) % 360.0)
    return np.minimum(d, 360.0 - d)


ROW_COLS = {
    "MVT_ID_mvt": pl.Float64,
    "dwell_end_rel": pl.Float64, "dwell_dur": pl.Float64, "dwell_next_rel": pl.Float64,
    "rev_rel": pl.Float64, "dep_first_rel": pl.Float64, "dep_first_v": pl.Float64,
    "dep_first_rwy_m": pl.Float64, "rwy_zone_rel": pl.Float64, "rwy_stop_s": pl.Float64,
    "last_stop_end_rel": pl.Float64, "n_stops": pl.Float64, "stop_s": pl.Float64, "path_m": pl.Float64,
    "mov_s": pl.Float64, "dep_n": pl.Float64,
}


def row_features(k: pl.DataFrame, dep: pl.DataFrame, lines: dict) -> pl.DataFrame:
    """Label-free per-row kinematic features from ``kinematics`` output.

    ``dep``: MVT_ID_mvt, ADEP_mvt, RUNWAY_mvt. Times are relative to MVT (s).
    The departure run starts after the last position dwell (>= 180 s within 30 m) that is more
    than 250 m from every runway centreline (a stand or remote-hold dwell, not a runway queue).
    """
    k = k.sort("MVT_ID_mvt", "t_rel")
    ids = k["MVT_ID_mvt"].to_numpy()
    T = k["t_rel"].to_numpy().astype(float)
    X, Y = k["x"].to_numpy(), k["y"].to_numpy()
    V = k["v_ms"].fill_null(np.nan).to_numpy().astype(float)
    TRK = k["trk"].to_numpy().astype(float)
    HS = k["hs"].is_not_null().to_numpy()
    rw = dict(zip(dep["MVT_ID_mvt"].to_list(), zip(dep["ADEP_mvt"].to_list(),
                                                     dep["RUNWAY_mvt"].cast(pl.String).to_list())))
    if len(ids) == 0:
        return pl.DataFrame(schema=ROW_COLS)
    change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts, ends = np.concatenate(([0], change)), np.concatenate((change, [len(ids)]))
    recs = {c: [] for c in ROW_COLS}
    for a, b in zip(starts, ends):
        mid = ids[a]
        ap, rwy = rw.get(mid, (None, None))
        t, x, y, v, trk, hs = T[a:b], X[a:b], Y[a:b], V[a:b], TRK[a:b], HS[a:b]
        n = b - a
        ln = lines.get(ap, [])
        d_any = np.full(n, np.inf)
        d_dep = np.full(n, np.nan)
        key = (rwy or "").strip().upper().lstrip("0")
        for idents, poly in ln:
            d = _dist_polyline(x, y, poly)
            d_any = np.minimum(d_any, d)
            if key and key in idents:
                d_dep = np.fmin(d_dep, d)
        # dwells
        dwell = None
        i = 0
        while i < n:
            j = i
            while j + 1 < n and (x[j + 1] - x[i]) ** 2 + (y[j + 1] - y[i]) ** 2 <= DWELL_R_M ** 2:
                j += 1
            if t[j] - t[i] >= DWELL_MIN_S:
                if d_any[i:j + 1].min() > RWY_ZONE_M:
                    dwell = (i, j)
                i = j + 1
            else:
                i += 1
        rec = dict.fromkeys(ROW_COLS)
        rec["MVT_ID_mvt"] = mid
        s0 = 0
        if dwell is not None:
            i0, j0 = dwell
            rec["dwell_end_rel"], rec["dwell_dur"] = t[j0], t[j0] - t[i0]
            rec["dwell_next_rel"] = t[j0 + 1] if j0 + 1 < n else np.nan
            s0 = j0 + 1
        if s0 < n:
            tt, vv = t[s0:], np.nan_to_num(v[s0:], nan=0.0)
            rec["dep_first_rel"], rec["dep_first_v"] = tt[0], v[s0]
            rec["dep_first_rwy_m"] = d_dep[s0] if not np.isnan(d_dep[s0]) else d_any[s0]
            rec["dep_n"] = float(n - s0)
            # first track reversal after the first moving sample with a track
            mv = np.flatnonzero((vv >= MOVE_MS) & hs[s0:])
            if len(mv):
                trk0 = trk[s0 + mv[0]]
                win = mv[(tt[mv] - tt[mv[0]]) <= REV_WINDOW_S]
                rev = win[_angle_diff(trk[s0 + win], trk0) > REV_DEG]
                rec["rev_rel"] = tt[rev[0]] if len(rev) else np.nan
            # stops (stationary episodes >= 20 s)
            still = vv < STOP_MS
            e = np.diff(np.concatenate(([0], still.astype(np.int8), [0])))
            ss, se = np.flatnonzero(e == 1), np.flatnonzero(e == -1) - 1
            dur = tt[se] - tt[ss]
            ok = dur >= STOP_MIN_S
            rec["n_stops"], rec["stop_s"] = float(ok.sum()), float(dur[ok].sum())
            rec["last_stop_end_rel"] = tt[se[ok][-1]] if ok.any() else np.nan
            dd = np.sqrt(np.diff(x[s0:]) ** 2 + np.diff(y[s0:]) ** 2)
            rec["path_m"] = float(dd.sum())
            dts = np.diff(tt)
            rec["mov_s"] = float(dts[(dd / np.maximum(dts, 1e-3)) >= MOVE_MS].sum()) if len(dts) else 0.0
            dz = d_dep[s0:] if not np.all(np.isnan(d_dep[s0:])) else d_any[s0:]
            z = np.flatnonzero(dz <= RWY_ZONE_M)
            if len(z):
                zi = z[0]
                rec["rwy_zone_rel"] = tt[zi]
                ok_z = ok & (tt[ss] >= tt[zi])
                rec["rwy_stop_s"] = float(dur[ok_z].sum())
        for c in ROW_COLS:
            recs[c].append(rec[c])
    return pl.DataFrame(recs, schema=ROW_COLS).with_columns(
        [pl.col(c).fill_nan(None) for c in ROW_COLS if c != "MVT_ID_mvt"])


def map_row_estimates(e: pl.DataFrame, rowf: pl.DataFrame | None = None, prefix: str = "map") -> pl.DataFrame:
    """Aggregate per-sample map estimates to one row. With ``rowf`` only samples of the
    departure run (t_rel >= dep_first_rel) are used."""
    if rowf is not None:
        e = e.join(rowf.select("MVT_ID_mvt", "dep_first_rel"), on="MVT_ID_mvt", how="left").filter(
            pl.col("dep_first_rel").is_null() | (pl.col("t_rel") >= pl.col("dep_first_rel")))
    w = 1 / (pl.col("sd").pow(2) + 60.0 ** 2)
    good = pl.col("sd") <= 180
    return e.sort("MVT_ID_mvt", "t_rel").group_by("MVT_ID_mvt").agg(
        pl.coalesce(pl.col("est").filter(good).first(), pl.col("est").first()).alias(f"{prefix}_first_rel"),
        pl.col("est").sort_by("sd").first().alias(f"{prefix}_minsd_rel"),
        ((pl.col("est") * w).sum() / w.sum()).alias(f"{prefix}_wmean_rel"),
        pl.col("sd").min().alias(f"{prefix}_sd_min"),
        pl.coalesce(pl.col("sd").filter(good).first(), pl.col("sd").first()).alias(f"{prefix}_first_sd"),
        pl.col("t_rel").filter(good).first().alias(f"{prefix}_first_t"),
        pl.col("lvl").min().cast(pl.Float64).alias(f"{prefix}_lvl"),
        pl.len().cast(pl.Float64).alias(f"{prefix}_n"),
        pl.col("est").std().alias(f"{prefix}_est_sd"),
    )
