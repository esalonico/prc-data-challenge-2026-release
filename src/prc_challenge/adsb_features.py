# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""ADS-B (adsb.lol) off-block sensor features for matched departures (research mode, D-012).

Input: per-day filtered trace parquet written by ``scripts/adsb_lol_ingest.py``
(samples within 10 km of the ten ARPs, on ground or below elevation + 5,000 ft).

Three stages, all keyed by ``MVT_ID_mvt``:

1. ``day_segments`` (label-free): join each departure of UTC day D to one aircraft
   (``CALLSIGN_flt``, fallback ``FLIGHT_mvt``, window [MVT - 2 h, MVT + 20 min]) and
   cut its departure ground block: ground samples at ADEP after its last approach,
   up to MVT, after the last transponder-off gap (>= 600 s). Samples come from
   day D plus the tail of D-1 (last 4 h) and the head of D+1 (first 1 h).
   Also returns simple label-free estimators (first motion, last stationary).
2. ``block_fixes`` (training days only, uses labels): the aircraft's position at
   the labelled BLOCK time (nearest block sample within 120 s). These are the
   only label-derived quantities; they feed stand centroids.
3. ``gate_features``: for each row, the last block sample within R m of (i) a
   stand centroid learned from other months' fixes (month-isolated) and (ii) the
   label-free OSM parking position. Times are reported relative to MVT.

Licence: ODbL 1.0 data, "Contains data from ADSB.lol (https://adsb.lol), licensed
under ODbL 1.0". OpenSky is not used anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

EARTH_R = 6_371_000.0
JOIN_BEFORE_S = 2 * 3600
JOIN_AFTER_S = 20 * 60
LOOKBACK_S = 3 * 3600
PREV_TAIL_S = 4 * 3600
NEXT_HEAD_S = 3600
GAP_S = 600
DWELL_S = 240
MOVE_KT = 3.0
STILL_KT = 1.0
FIX_TOL_S = 120
RADII = (100, 200)
CENTROID_OUTLIER_M = 100.0
MIN_FIXES = 2

SAMPLE_COLS = ["airport", "hex", "ts", "lat", "lon", "on_ground", "gs_kt", "callsign"]
SEG_SCHEMA = {"MVT_ID_mvt": pl.Float64, "t_rel": pl.Float32, "lat": pl.Float64, "lon": pl.Float64,
              "gs": pl.Float32}


@dataclass(frozen=True)
class DayFiles:
    day: date
    prev: Path | None
    cur: Path
    nxt: Path | None


def trace_path(filt_dir: Path, day: date) -> Path | None:
    """Per-day slim sample file ``adsb_lol_<day>_slim.parquet`` (VM-side slimming, 2026-09-25):
    8 columns, all ground samples, airborne samples thinned to one per 10 s per aircraft."""
    p = filt_dir / f"adsb_lol_{day.isoformat()}_slim.parquet"
    return p if p.exists() else None


def day_files(filt_dir: Path, day: date) -> DayFiles | None:
    cur = trace_path(filt_dir, day)
    if cur is None:
        return None
    return DayFiles(day, trace_path(filt_dir, day - timedelta(days=1)), cur,
                    trace_path(filt_dir, day + timedelta(days=1)))


def _epoch(d: date) -> float:
    return datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()


def load_samples(files: DayFiles) -> pl.DataFrame:
    """Samples of day D plus the D-1 tail and D+1 head, sorted by (airport, hex, ts)."""
    t0 = _epoch(files.day)
    parts = [pl.scan_parquet(files.cur).select(SAMPLE_COLS)]
    if files.prev is not None:
        parts.append(pl.scan_parquet(files.prev).select(SAMPLE_COLS).filter(pl.col("ts") >= t0 - PREV_TAIL_S))
    if files.nxt is not None:
        parts.append(pl.scan_parquet(files.nxt).select(SAMPLE_COLS).filter(pl.col("ts") < t0 + 86400 + NEXT_HEAD_S))
    return (
        pl.concat(parts)
        .with_columns(pl.col("callsign").str.strip_chars().str.to_uppercase())
        .unique(["airport", "hex", "ts"], keep="first")
        .sort("airport", "hex", "ts")
        .collect()
    )


def normalize_departures(dep: pl.DataFrame) -> pl.DataFrame:
    """Add ``mvt`` (epoch s), callsign keys and ``stand_key`` (label-free)."""
    stand = (
        pl.col("STAND_mvt").cast(pl.String).str.to_uppercase().str.replace_all(r"[\s\-]", "")
        .str.replace(r"^([A-Z]*)0+(\d+)([A-Z]*)$", "$1$2$3")
    )
    return dep.with_columns(
        (pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0).alias("mvt"),
        pl.col("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("cs_flt"),
        pl.col("FLIGHT_mvt").str.strip_chars().str.to_uppercase().str.replace_all(" ", "").alias("cs_mvt"),
        stand.alias("stand_key"),
    )


def join_aircraft(dep: pl.DataFrame, s: pl.DataFrame) -> pl.DataFrame:
    """Dominant ICAO24 per departure; ``join_src`` 1 = CALLSIGN_flt, 2 = FLIGHT_mvt fallback."""
    cs = s.filter(pl.col("callsign").is_not_null() & (pl.col("callsign") != "")).select(
        "airport", "callsign", "hex", "ts")
    out = dep
    passes = [("cs_flt", "flt"), ("cs_mvt", "mvt")]
    if "cs_dict" in dep.columns:  # IATA -> callsign dictionary (prereg amendment 2), join_src 4
        passes.append(("cs_dict", "dict"))
    for key, tag in passes:
        m = (
            dep.select("MVT_ID_mvt", "ADEP_mvt", key, "mvt")
            .filter(pl.col(key).is_not_null())
            .join(cs, left_on=["ADEP_mvt", key], right_on=["airport", "callsign"], how="inner")
            .filter((pl.col("ts") >= pl.col("mvt") - JOIN_BEFORE_S) & (pl.col("ts") <= pl.col("mvt") + JOIN_AFTER_S))
            .group_by("MVT_ID_mvt", "hex")
            .agg(pl.len().alias("n"))
            .sort(["MVT_ID_mvt", "n", "hex"], descending=[False, True, False])
            .unique("MVT_ID_mvt", keep="first")
            .select("MVT_ID_mvt", pl.col("hex").alias(f"hex_{tag}"))
        )
        out = out.join(m, on="MVT_ID_mvt", how="left")
    if "hex_dict" not in out.columns:
        out = out.with_columns(pl.lit(None, pl.String).alias("hex_dict"))
    out = out.with_columns(
        pl.coalesce("hex_flt", "hex_mvt", "hex_dict").alias("hex"),
        pl.when(pl.col("hex_flt").is_not_null()).then(1)
        .when(pl.col("hex_mvt").is_not_null()).then(2)
        .when(pl.col("hex_dict").is_not_null()).then(4).otherwise(0).cast(pl.Int8).alias("join_src"),
    ).drop("hex_flt", "hex_mvt", "hex_dict")
    return takeoff_match(out, s)


TAKEOFF_TOL_S = 180.0
TAKEOFF_CLAIM_S = 600.0


def takeoff_events(s: pl.DataFrame) -> pl.DataFrame:
    """(airport, hex, t_air): first airborne sample right after a ground sample (<= 120 s apart)."""
    return (
        s.select("airport", "hex", "ts", "on_ground")
        .with_columns(pl.col("hex").shift(-1).alias("h1"), pl.col("ts").shift(-1).alias("t1"),
                      pl.col("on_ground").shift(-1).alias("g1"))
        .filter(pl.col("on_ground") & (pl.col("h1") == pl.col("hex")) & ~pl.col("g1")
                & ((pl.col("t1") - pl.col("ts")) <= 120))
        .select("airport", "hex", pl.col("t1").alias("t_air"))
    )


def takeoff_match(dep: pl.DataFrame, s: pl.DataFrame) -> pl.DataFrame:
    """Fallback join (``join_src`` 3) for departures without a callsign match: the ADS-B take-off
    at ADEP nearest to MVT within 180 s, not claimed by a callsign-joined departure (same hex,
    |MVT - t_air| <= 600 s), each take-off used at most once (greedy by |dt|). MVT is the
    actual take-off time, so this is label-free; it mainly serves NM-unmatched rows."""
    ev = takeoff_events(s)
    todo = dep.filter(pl.col("hex").is_null()).select("MVT_ID_mvt", "ADEP_mvt", "mvt")
    if ev.height == 0 or todo.height == 0:
        return dep
    claimed = (
        dep.filter(pl.col("hex").is_not_null()).select("ADEP_mvt", "hex", "mvt")
        .join(ev, left_on=["ADEP_mvt", "hex"], right_on=["airport", "hex"], how="inner")
        .filter((pl.col("t_air") - pl.col("mvt")).abs() <= TAKEOFF_CLAIM_S)
        .select(pl.col("ADEP_mvt").alias("airport"), "hex", "t_air")
    )
    ev = ev.join(claimed, on=["airport", "hex", "t_air"], how="anti")
    cand = (
        todo.join(ev, left_on="ADEP_mvt", right_on="airport", how="inner")
        .with_columns((pl.col("t_air") - pl.col("mvt")).abs().alias("dt"))
        .filter(pl.col("dt") <= TAKEOFF_TOL_S)
        .sort("dt", "MVT_ID_mvt", "hex")
    )
    used_rows, used_ev, picks = set(), set(), []
    for mid, hx, ta, _dt in cand.select("MVT_ID_mvt", "hex", "t_air", "dt").iter_rows():
        if mid in used_rows or (hx, ta) in used_ev:
            continue
        used_rows.add(mid)
        used_ev.add((hx, ta))
        picks.append((mid, hx))
    if not picks:
        return dep
    pk = pl.DataFrame(picks, schema={"MVT_ID_mvt": pl.Float64, "hex_to": pl.String}, orient="row")
    return dep.join(pk, on="MVT_ID_mvt", how="left").with_columns(
        pl.coalesce("hex", "hex_to").alias("hex"),
        pl.when(pl.col("hex").is_null() & pl.col("hex_to").is_not_null()).then(3)
        .otherwise(pl.col("join_src")).cast(pl.Int8).alias("join_src"),
    ).drop("hex_to")


def _estimate_ab(t: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """(first motion > MOVE_KT after the stand dwell, last stationary sample before it)."""
    if len(t) == 0:
        return math.nan, math.nan
    still = np.nan_to_num(v, nan=np.inf) <= STILL_KT
    moving = np.nan_to_num(v, nan=-1.0) > MOVE_KT
    # stationary runs as (start, end) index pairs
    edges = np.diff(np.concatenate(([0], still.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    dwell = None
    for i, j in zip(starts[::-1], ends[::-1]):
        if t[j] - t[i] >= DWELL_S:
            dwell = (i, j)
            break
    if dwell is None and len(starts) and starts[0] == 0:
        dwell = (starts[0], ends[0])
    begin = dwell[1] + 1 if dwell else 0
    mv = np.flatnonzero(moving[begin:])
    if len(mv) == 0:
        return math.nan, (float(t[dwell[1]]) if dwell else math.nan)
    ia = begin + mv[0]
    if dwell:
        b = float(t[dwell[1]])
    else:
        prev = np.flatnonzero(still[:ia])
        b = float(t[prev[-1]]) if len(prev) else math.nan
    return float(t[ia]), b


def day_segments(dep: pl.DataFrame, s: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-departure label-free stats and the departure ground-block samples.

    ``dep`` must be normalized and joined (``hex``). Returns (rows, segments); all
    times are relative to MVT (seconds, negative = before take-off).
    """
    keys = s.select("airport", "hex").to_numpy()
    ts_all = s["ts"].to_numpy()
    lat_all, lon_all = s["lat"].to_numpy(), s["lon"].to_numpy()
    gnd_all = s["on_ground"].to_numpy()
    gs_all = s["gs_kt"].fill_null(np.nan).to_numpy().astype(float)
    # group boundaries for (airport, hex)
    if len(keys):
        change = np.flatnonzero((keys[1:, 0] != keys[:-1, 0]) | (keys[1:, 1] != keys[:-1, 1])) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(keys)]))
        index = {(keys[a, 0], keys[a, 1]): (a, b) for a, b in zip(starts, ends)}
    else:
        index = {}
    rows, seg_ids, seg_t, seg_lat, seg_lon, seg_gs = [], [], [], [], [], []
    for r in dep.select("MVT_ID_mvt", "ADEP_mvt", "hex", "mvt").iter_rows():
        mid, ap, hx, mvt = r
        rec = {"MVT_ID_mvt": mid, "n_seg": 0, "n_hex_samples": 0}
        span = index.get((ap, hx)) if hx is not None else None
        if span is None:
            rows.append(rec)
            continue
        a, b = span
        lo = a + np.searchsorted(ts_all[a:b], mvt - LOOKBACK_S)
        hi = a + np.searchsorted(ts_all[a:b], mvt + 600, side="right")
        t, g = ts_all[lo:hi], gnd_all[lo:hi]
        rec["n_hex_samples"] = int(hi - lo)
        air_before = np.flatnonzero((~g) & (t < mvt - 600))
        cut = t[air_before[-1]] if len(air_before) else mvt - LOOKBACK_S
        air_after = np.flatnonzero((~g) & (t >= mvt - 600))
        rec["takeoff_rel"] = float(t[air_after[0]] - mvt) if len(air_after) else math.nan
        keep = np.flatnonzero(g & (t > cut) & (t <= mvt)) + lo
        if len(keep):
            gaps = np.flatnonzero(np.diff(ts_all[keep]) >= GAP_S)
            if len(gaps):
                keep = keep[gaps[-1] + 1:]
        if len(keep) == 0:
            rows.append(rec)
            continue
        st, sv = ts_all[keep], gs_all[keep]
        est_a, est_b = _estimate_ab(st, sv)
        rec.update({
            "n_seg": len(keep),
            "seg_first_rel": float(st[0] - mvt),
            "seg_last_rel": float(st[-1] - mvt),
            "seg_first_gs": float(sv[0]) if not np.isnan(sv[0]) else math.nan,
            "seg_max_gap": float(np.diff(st).max()) if len(st) > 1 else 0.0,
            "move_rel": est_a - mvt,
            "still_rel": est_b - mvt,
        })
        rows.append(rec)
        seg_ids.append(np.full(len(keep), mid))
        seg_t.append((st - mvt).astype(np.float32))
        seg_lat.append(lat_all[keep])
        seg_lon.append(lon_all[keep])
        seg_gs.append(sv.astype(np.float32))
    row_schema = {"MVT_ID_mvt": pl.Float64, "n_seg": pl.Int32, "n_hex_samples": pl.Int32,
                  "takeoff_rel": pl.Float64, "seg_first_rel": pl.Float64, "seg_last_rel": pl.Float64,
                  "seg_first_gs": pl.Float64, "seg_max_gap": pl.Float64, "move_rel": pl.Float64,
                  "still_rel": pl.Float64}
    rows_df = pl.DataFrame([{k: x.get(k) for k in row_schema} for x in rows], schema=row_schema)
    rows_df = rows_df.with_columns(
        [pl.col(c).fill_nan(None) for c, t in row_schema.items() if t == pl.Float64 and c != "MVT_ID_mvt"])
    if seg_ids:
        seg = pl.DataFrame({
            "MVT_ID_mvt": np.concatenate(seg_ids), "t_rel": np.concatenate(seg_t),
            "lat": np.concatenate(seg_lat), "lon": np.concatenate(seg_lon), "gs": np.concatenate(seg_gs),
        }, schema=SEG_SCHEMA)
    else:
        seg = pl.DataFrame(schema=SEG_SCHEMA)
    return rows_df, seg


def block_fixes(seg: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Position at the labelled BLOCK time (training days only).

    ``labels``: MVT_ID_mvt, block_rel (BLOCK - MVT seconds), plus identifying columns
    (ADEP_mvt, stand_key, day, month) that are carried through.
    """
    j = seg.join(labels.select("MVT_ID_mvt", "block_rel"), on="MVT_ID_mvt", how="inner")
    fix = (
        j.with_columns((pl.col("t_rel") - pl.col("block_rel")).abs().alias("dt"))
        .sort("MVT_ID_mvt", "dt")
        .unique("MVT_ID_mvt", keep="first")
        .filter(pl.col("dt") <= FIX_TOL_S)
        .select("MVT_ID_mvt", pl.col("lat").alias("fix_lat"), pl.col("lon").alias("fix_lon"),
                pl.col("dt").alias("fix_dt"))
    )
    return labels.drop("block_rel").join(fix, on="MVT_ID_mvt", how="inner")


def _hav_expr(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    r = math.pi / 180.0
    a = ((lat2 - lat1) * r / 2).sin().pow(2) + (lat1 * r).cos() * (lat2 * r).cos() * ((lon2 - lon1) * r / 2).sin().pow(2)
    return 2 * EARTH_R * a.sqrt().arcsin()


def stand_centroids(fixes: pl.DataFrame) -> pl.DataFrame:
    """Robust (airport, stand) centroid: median, drop fixes > 100 m away, median again."""
    if fixes.height == 0:
        return pl.DataFrame(schema={"ADEP_mvt": pl.String, "stand_key": pl.String, "c_lat": pl.Float64,
                                    "c_lon": pl.Float64, "c_n": pl.UInt32})
    keys = ["ADEP_mvt", "stand_key"]
    f = fixes.filter(pl.col("stand_key").is_not_null())
    m1 = f.group_by(keys).agg(pl.col("fix_lat").median().alias("m_lat"), pl.col("fix_lon").median().alias("m_lon"))
    f = f.join(m1, on=keys).filter(
        _hav_expr(pl.col("fix_lat"), pl.col("fix_lon"), pl.col("m_lat"), pl.col("m_lon")) <= CENTROID_OUTLIER_M)
    return (
        f.group_by(keys)
        .agg(pl.col("fix_lat").median().alias("c_lat"), pl.col("fix_lon").median().alias("c_lon"),
             pl.len().alias("c_n"))
        .filter(pl.col("c_n") >= MIN_FIXES)
    )


def last_within(seg: pl.DataFrame, anchors: pl.DataFrame, radius: float, name: str) -> pl.DataFrame:
    """Latest block sample within ``radius`` m of the row's anchor (a_lat, a_lon)."""
    return (
        seg.join(anchors.select("MVT_ID_mvt", "a_lat", "a_lon"), on="MVT_ID_mvt", how="inner")
        .filter(_hav_expr(pl.col("lat"), pl.col("lon"), pl.col("a_lat"), pl.col("a_lon")) <= radius)
        .group_by("MVT_ID_mvt")
        .agg(pl.col("t_rel").max().cast(pl.Float64).alias(name))
    )


def gate_features(rows: pl.DataFrame, seg: pl.DataFrame, centroids: pl.DataFrame | None,
                  osm: pl.DataFrame) -> pl.DataFrame:
    """Final feature block. ``rows`` needs MVT_ID_mvt, ADEP_mvt, stand_key and the stage-1 stats.

    ``centroids`` already excludes the rows' own month (caller's responsibility).
    ``osm``: AIRPORT/stand_key/stand_lat/stand_lon from ``geometry.load_stand_coords``.
    """
    out = rows
    anchors_osm = rows.join(osm.rename({"AIRPORT": "ADEP_mvt", "stand_lat": "a_lat", "stand_lon": "a_lon"}),
                            on=["ADEP_mvt", "stand_key"], how="inner")
    for R in RADII:
        out = out.join(last_within(seg, anchors_osm, R, f"gate_osm{R}_rel"), on="MVT_ID_mvt", how="left")
    if centroids is not None and centroids.height:
        anchors_c = rows.join(centroids.rename({"c_lat": "a_lat", "c_lon": "a_lon"}),
                              on=["ADEP_mvt", "stand_key"], how="inner")
        for R in RADII:
            out = out.join(last_within(seg, anchors_c, R, f"gate_c{R}_rel"), on="MVT_ID_mvt", how="left")
        out = out.join(anchors_c.select("MVT_ID_mvt", pl.col("c_n").cast(pl.Int32).alias("centroid_n")),
                       on="MVT_ID_mvt", how="left")
        cdist = anchors_c.join(
            osm.rename({"AIRPORT": "ADEP_mvt"}), on=["ADEP_mvt", "stand_key"], how="inner"
        ).select("MVT_ID_mvt", _hav_expr(pl.col("a_lat"), pl.col("a_lon"), pl.col("stand_lat"),
                                         pl.col("stand_lon")).alias("centroid_osm_m"))
        out = out.join(cdist, on="MVT_ID_mvt", how="left")
    else:
        out = out.with_columns(*[pl.lit(None, pl.Float64).alias(f"gate_c{R}_rel") for R in RADII],
                               pl.lit(None, pl.Int32).alias("centroid_n"),
                               pl.lit(None, pl.Float64).alias("centroid_osm_m"))
    best = pl.coalesce("gate_c100_rel", "gate_osm100_rel", "gate_c200_rel", "gate_osm200_rel")
    best_src = (
        pl.when(pl.col("gate_c100_rel").is_not_null()).then(1)
        .when(pl.col("gate_osm100_rel").is_not_null()).then(2)
        .when(pl.col("gate_c200_rel").is_not_null()).then(3)
        .when(pl.col("gate_osm200_rel").is_not_null()).then(4).otherwise(0)
    )
    cov = (
        pl.when(pl.col("join_src") == 0).then(0)
        .when(pl.col("n_seg") == 0).then(1)
        .when(best.is_null()).then(2).otherwise(3)
    )
    out = out.with_columns(best.alias("gate_best_rel"), best_src.cast(pl.Int8).alias("gate_best_src"),
                           cov.cast(pl.Int8).alias("coverage"))
    # v2: gate confirmation from the samples after the gate sample (coverage-truncation guard)
    after = (
        seg.join(out.select("MVT_ID_mvt", "gate_best_rel").filter(pl.col("gate_best_rel").is_not_null()),
                 on="MVT_ID_mvt", how="inner")
        .filter(pl.col("t_rel") > pl.col("gate_best_rel"))
        .with_columns((pl.col("t_rel") - pl.col("gate_best_rel")).alias("dt_g"))
        .group_by("MVT_ID_mvt")
        .agg(pl.col("dt_g").min().cast(pl.Float64).alias("q_next_dt"), pl.len().cast(pl.Int32).alias("q_n_after"),
             ((pl.col("dt_g") <= CONF_MOVE_S) & (pl.col("gs").fill_nan(-1.0) > MOVE_KT)).any().alias("q_moved_600"))
    )
    out = out.join(after, on="MVT_ID_mvt", how="left").with_columns(
        pl.when(pl.col("gate_best_rel").is_not_null()).then(pl.col("q_n_after").fill_null(0)).alias("q_n_after"),
        pl.when(pl.col("gate_best_rel").is_not_null()).then(pl.col("q_moved_600").fill_null(False))
        .alias("q_moved_600"),
    ).with_columns(
        (pl.col("gate_best_rel").is_not_null() & (pl.col("q_next_dt") <= CONF_NEXT_S) & pl.col("q_moved_600"))
        .fill_null(False).cast(pl.Int8).alias("gate_conf"),
        pl.col("q_moved_600").cast(pl.Int8),
    )
    # v2: first-seen channel, distance of the first block sample to the stand anchor
    anchor = anchors_osm.select("MVT_ID_mvt", pl.col("a_lat").alias("o_lat"), pl.col("a_lon").alias("o_lon"))
    if centroids is not None and centroids.height:
        anchor = anchor.join(anchors_c.select("MVT_ID_mvt", "a_lat", "a_lon"), on="MVT_ID_mvt", how="full",
                             coalesce=True).select(
            "MVT_ID_mvt", pl.coalesce("a_lat", "o_lat").alias("x_lat"), pl.coalesce("a_lon", "o_lon").alias("x_lon"))
    else:
        anchor = anchor.rename({"o_lat": "x_lat", "o_lon": "x_lon"})
    first = seg.sort("MVT_ID_mvt", "t_rel").unique("MVT_ID_mvt", keep="first").select("MVT_ID_mvt", "lat", "lon")
    fd = first.join(anchor, on="MVT_ID_mvt", how="inner").select(
        "MVT_ID_mvt", _hav_expr(pl.col("lat"), pl.col("lon"), pl.col("x_lat"), pl.col("x_lon")).alias("seg_first_dist_m"))
    return out.join(fd, on="MVT_ID_mvt", how="left")


CONF_NEXT_S = 120.0
CONF_MOVE_S = 600.0

FEATURE_COLS = [
    "join_src", "coverage", "n_seg", "n_hex_samples", "takeoff_rel", "seg_first_rel", "seg_last_rel",
    "seg_first_gs", "seg_max_gap", "move_rel", "still_rel", "gate_c100_rel", "gate_c200_rel",
    "gate_osm100_rel", "gate_osm200_rel", "gate_best_rel", "gate_best_src", "centroid_n", "centroid_osm_m",
    "taxi_hat", "bias_n", "prev_day_present",
    # v2
    "gate_conf", "q_next_dt", "q_n_after", "q_moved_600", "seg_first_dist_m",
    "aobt_rel", "aobt_minus_gate", "aobt_minus_first", "taxi_hat_first", "bias_first_n",
]


def first_bias(feat: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Median (seg_first_rel - BLOCK) per (airport, first-seen moving flag) from labelled rows."""
    return (
        feat.select("MVT_ID_mvt", "ADEP_mvt", "seg_first_rel", "seg_first_gs")
        .join(labels.select("MVT_ID_mvt", "block_rel"), on="MVT_ID_mvt", how="inner")
        .filter(pl.col("seg_first_rel").is_not_null())
        .with_columns((pl.col("seg_first_gs").fill_null(0.0) > MOVE_KT).alias("first_moving"))
        .group_by("ADEP_mvt", "first_moving")
        .agg((pl.col("seg_first_rel") - pl.col("block_rel")).median().alias("bias_first"),
             pl.len().alias("bias_first_n"))
        .filter(pl.col("bias_first_n") >= MIN_BIAS_N)
    )


def add_first_and_aobt(feat: pl.DataFrame, fbias: pl.DataFrame | None, aobt: pl.DataFrame) -> pl.DataFrame:
    """taxi_hat_first = bias_first - seg_first_rel; AOBT offsets (aobt_rel = AOBT_3 - MVT, seconds)."""
    out = feat.join(aobt.select("MVT_ID_mvt", "aobt_rel"), on="MVT_ID_mvt", how="left").with_columns(
        (pl.col("aobt_rel") - pl.col("gate_best_rel")).alias("aobt_minus_gate"),
        (pl.col("aobt_rel") - pl.col("seg_first_rel")).alias("aobt_minus_first"),
        (pl.col("seg_first_gs").fill_null(0.0) > MOVE_KT).alias("first_moving"),
    )
    if fbias is not None and fbias.height:
        out = out.join(fbias, on=["ADEP_mvt", "first_moving"], how="left").with_columns(
            (pl.col("bias_first") - pl.col("seg_first_rel")).alias("taxi_hat_first"),
            pl.col("bias_first_n").cast(pl.Int32)).drop("bias_first")
    else:
        out = out.with_columns(pl.lit(None, pl.Float64).alias("taxi_hat_first"),
                               pl.lit(None, pl.Int32).alias("bias_first_n"))
    return out.drop("first_moving")
MIN_BIAS_N = 30


def gate_bias(feat: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Median (gate_best - BLOCK) per (airport, gate source) from labelled rows.

    ``labels``: MVT_ID_mvt, block_rel. The caller passes rows from other months only.
    """
    return (
        feat.select("MVT_ID_mvt", "ADEP_mvt", "gate_best_src", "gate_best_rel")
        .join(labels.select("MVT_ID_mvt", "block_rel"), on="MVT_ID_mvt", how="inner")
        .filter(pl.col("gate_best_rel").is_not_null() & (pl.col("gate_best_src") > 0))
        .group_by("ADEP_mvt", "gate_best_src")
        .agg((pl.col("gate_best_rel") - pl.col("block_rel")).median().alias("bias"), pl.len().alias("bias_n"))
        .filter(pl.col("bias_n") >= MIN_BIAS_N)
    )


def add_taxi_hat(feat: pl.DataFrame, bias: pl.DataFrame) -> pl.DataFrame:
    """Sensor taxi estimate: MVT - (gate - bias) = bias - gate_best_rel."""
    return feat.join(bias, on=["ADEP_mvt", "gate_best_src"], how="left").with_columns(
        (pl.col("bias") - pl.col("gate_best_rel")).alias("taxi_hat"),
        pl.col("bias_n").cast(pl.Int32),
    ).drop("bias")


def to_block(feat: pl.DataFrame) -> pl.DataFrame:
    """Rename to the ``adsb_`` namespace; rows without a join keep NaN/None features."""
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    return feat.select("MVT_ID_mvt", *[pl.col(c).alias(f"adsb_{c}") for c in cols],
                       pl.col("ADEP_mvt").alias("adsb_airport"))


# --------------------------------------------------------------------------- surface load

SURF_ANCHORS_MIN = (0, 10, 20)  # window ends at T, T-10 min, T-20 min (T = focal MVT)
MOVE_WINDOWS_MIN = (5, 10)
PUSH_WINDOWS_MIN = (10, 20)
QUEUE_WINDOW_MIN = 5
QUEUE_RADIUS_M = 1000.0


def _distinct_counts(pairs_hex: np.ndarray, pairs_min: np.ndarray, n_hex: int, n_min: int,
                     q_lo: np.ndarray, q_hi: np.ndarray, q_focal: np.ndarray) -> np.ndarray:
    """Distinct hexes with any (hex, minute) pair in [q_lo, q_hi), excluding the query's focal hex."""
    m = np.zeros((n_hex, n_min + 1), dtype=np.int32)
    if len(pairs_hex):
        m[pairs_hex, pairs_min + 1] = 1
    c = np.cumsum(m, axis=1)
    lo, hi = np.clip(q_lo, 0, n_min), np.clip(q_hi, 0, n_min)
    hit = (c[:, hi] - c[:, lo]) > 0  # (hex, query)
    total = hit.sum(axis=0)
    own = np.zeros(len(q_lo), dtype=np.int64)
    ok = q_focal >= 0
    own[ok] = hit[q_focal[ok], np.flatnonzero(ok)]
    return (total - own).astype(np.float64)


def pushback_events(g: pl.DataFrame) -> pl.DataFrame:
    """(hex, ts) of off-stand movement starts: in each ground block of an aircraft (split at
    gaps >= GAP_S) whose first sample is stationary (gs <= 1 kt), the first sample with
    gs > 1 kt. Blocks that start moving (landings, taxi-ins) never produce an event."""
    g = g.sort("hex", "ts").with_columns(
        ((pl.col("hex") != pl.col("hex").shift(1)) | ((pl.col("ts") - pl.col("ts").shift(1)) >= GAP_S))
        .fill_null(True).cast(pl.Int32).cum_sum().alias("blk"))
    first = g.group_by("blk").agg(pl.col("gs_kt").first().alias("gs0"))
    starts_still = first.filter(pl.col("gs0").is_not_null() & (pl.col("gs0") <= STILL_KT)).select("blk")
    return (
        g.join(starts_still, on="blk", how="semi")
        .filter(pl.col("gs_kt") > STILL_KT)
        .group_by("blk")
        .agg(pl.col("hex").first(), pl.col("ts").min())
        .select("hex", "ts")
        .sort("ts")
    )


def surface_features(dep: pl.DataFrame, s: pl.DataFrame, thresholds: pl.DataFrame, day0: float) -> pl.DataFrame:
    """Label-free measured surface load around each departure's MVT (focal aircraft excluded).

    ``dep``: MVT_ID_mvt, ADEP_mvt, RUNWAY_mvt, mvt, hex. ``thresholds``: AIRPORT, RUNWAY_mvt, rwy_lat,
    rwy_lon (OurAirports). Columns: surf_move{W}_t{A} (distinct aircraft with a ground sample
    > 3 kt in the W min before T-A), surf_push{W}_t{A} (pushback starts in the W min before T-A),
    surf_queue5_t{A} (distinct aircraft on the ground within 1 km of the departure-runway threshold
    in the 5 min before T-A), day_ground_hex (distinct aircraft seen on the ground at the airport
    that day), day_ground_cov (share of that day's departures with a departure ground block;
    filled by the caller).
    """
    base = day0 - PREV_TAIL_S
    n_min = int((86400 + PREV_TAIL_S + NEXT_HEAD_S) // 60) + 1
    out = []
    thr = thresholds.rename({"AIRPORT": "ADEP_mvt"})
    for ap in dep["ADEP_mvt"].unique().to_list():
        d = dep.filter(pl.col("ADEP_mvt") == ap).join(thr, on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
        g = s.filter((pl.col("airport") == ap) & pl.col("on_ground")).select("hex", "ts", "lat", "lon", "gs_kt")
        rec = {"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()}
        hexes = g["hex"].unique().sort().to_list()
        hidx = {h: i for i, h in enumerate(hexes)}
        focal = np.array([hidx.get(h, -1) if h is not None else -1 for h in d["hex"].to_list()], dtype=np.int64)
        tq = ((d["mvt"].to_numpy() - base) // 60).astype(np.int64)
        gh = np.array([hidx[h] for h in g["hex"].to_list()], dtype=np.int64) if g.height else np.zeros(0, np.int64)
        gm = ((g["ts"].to_numpy() - base) // 60).astype(np.int64) if g.height else np.zeros(0, np.int64)
        inr = (gm >= 0) & (gm < n_min)
        mv = inr & (np.nan_to_num(g["gs_kt"].to_numpy().astype(float), nan=-1.0) > MOVE_KT) if g.height else inr
        for A in SURF_ANCHORS_MIN:
            for W in MOVE_WINDOWS_MIN:
                rec[f"surf_move{W}_t{A}"] = _distinct_counts(gh[mv], gm[mv], len(hexes), n_min,
                                                             tq - A - W, tq - A, focal)
        ev = pushback_events(g) if g.height else pl.DataFrame(schema={"hex": pl.String, "ts": pl.Float64})
        ev_t = ev["ts"].to_numpy()
        ev_h = ev["hex"].to_list()
        own_t: dict[str, np.ndarray] = {}
        for h, t in zip(ev_h, ev_t):
            own_t.setdefault(h, []).append(t)
        mvt = d["mvt"].to_numpy()
        fh = d["hex"].to_list()
        for A in SURF_ANCHORS_MIN:
            for W in PUSH_WINDOWS_MIN:
                lo, hi = mvt - (A + W) * 60, mvt - A * 60
                cnt = np.searchsorted(ev_t, hi) - np.searchsorted(ev_t, lo)
                own = np.array([sum(lo[i] <= t < hi[i] for t in own_t.get(h, ())) if h else 0
                                for i, h in enumerate(fh)])
                rec[f"surf_push{W}_t{A}"] = (cnt - own).astype(np.float64)
        # departure queue near the row's runway threshold
        for A in SURF_ANCHORS_MIN:
            rec[f"surf_queue{QUEUE_WINDOW_MIN}_t{A}"] = np.full(d.height, np.nan)
        for rwy, rd in d.filter(pl.col("rwy_lat").is_not_null()).group_by("RUNWAY_mvt"):
            la, lo_ = rd["rwy_lat"][0], rd["rwy_lon"][0]
            near = g.with_row_index("i").filter(
                _hav_expr(pl.col("lat"), pl.col("lon"), pl.lit(la), pl.lit(lo_)) <= QUEUE_RADIUS_M)["i"].to_numpy()
            sel = np.zeros(g.height, dtype=bool)
            sel[near] = True
            sel &= inr
            pos = np.flatnonzero(np.isin(d["MVT_ID_mvt"].to_numpy(), rd["MVT_ID_mvt"].to_numpy()))
            for A in SURF_ANCHORS_MIN:
                vals = _distinct_counts(gh[sel], gm[sel], len(hexes), n_min, tq[pos] - A - QUEUE_WINDOW_MIN,
                                        tq[pos] - A, focal[pos])
                rec[f"surf_queue{QUEUE_WINDOW_MIN}_t{A}"][pos] = vals
        day_hex = g.filter((pl.col("ts") >= day0) & (pl.col("ts") < day0 + 86400))["hex"].n_unique()
        rec["day_ground_hex"] = np.full(d.height, float(day_hex))
        out.append(pl.DataFrame(rec))
    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame({"MVT_ID_mvt": []})


SURF_COLS = (
    [f"surf_move{W}_t{A}" for A in SURF_ANCHORS_MIN for W in MOVE_WINDOWS_MIN]
    + [f"surf_push{W}_t{A}" for A in SURF_ANCHORS_MIN for W in PUSH_WINDOWS_MIN]
    + [f"surf_queue{QUEUE_WINDOW_MIN}_t{A}" for A in SURF_ANCHORS_MIN]
    + ["day_ground_hex", "day_ground_cov"]
)
FEATURE_COLS += SURF_COLS
