# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-TRAJ step 2: consolidate the per-day tracks and write per-split sequence caches.

``consolidate``: every ``days/<day>/{rows.parquet, points.npy}`` into one store
``store/{rows.parquet, points.npy}`` (global point offsets), plus the month-isolated stand
centroids of the ADS-B block, per row, for both pools:

* pool ``D`` (D months 2-6, 8-10; the ADS-B D-fit block): a 2025 row of month m uses the fixes of
  pool months != m; every other row (2026, Nov/Dec) the whole pool;
* pool ``AY`` (months 1-10; the all-year block): the same rule.

Centroids are ``adsb_features.stand_centroids`` on the stage-1 ``fixes.parquet`` files, exactly as
``build_adsb_features.build_blocks`` computes them.  They are the only label-derived quantity (the
existing block's), converted to the airport frame of ``extract.py``.

``split <name>``: for an SA-NNX split cache (``nnx_cache_<name>``: m10, m5, v4jjnd with pool D;
ay, ay3l with pool AY) write ``<ART>/seq_<name>/{fit,ev}_{trk_off,trk_len,trk_row}.npy`` and
``{fit,ev}_trk_pts.npy`` (a compact CSR block of the side's own tracks, float16 [P, 9]) in the row
order of the cache's ``{fit,ev}_ids.npy``.  ``trk_row`` (float32 [N, 9]): has_track, join_src,
log1p(n_raw), OSM stand x/y, assigned-runway threshold x/y, centroid x/y (NaN when absent).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "3")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MAIN = Path(os.environ.get("PRC_MAIN_REPO", Path(__file__).resolve().parents[3]))  # bake VM: PRC_MAIN_REPO
ART = Path(os.environ["TRAJ_ART"]) if os.environ.get("TRAJ_ART") else MAIN / "artifacts/experiments/ADSB-trajectory-20260927"  # bake: traj_hires store
RUNS = MAIN / "artifacts/experiments/S4-allyear-rebuild-20260925/runs"
STAGE1 = MAIN / "artifacts/experiments/ADSB-features-20260925/stage1"
D_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)
AY_MONTHS = tuple(range(1, 11))
POOLS = {"D": D_MONTHS, "AY": AY_MONTHS}
SPLIT_POOL = {"m10": "D", "m5": "D", "v4jjnd": "D", "jj": "D", "ay": "AY", "ay3l": "AY", "ay3liso": "AY"}
DFIT = RUNS / "dfit_full_adsb_historder"
SPLIT_MATRICES = {
    "m10": [DFIT / "train_matrix.parquet"],
    "m5": [DFIT / "train_matrix.parquet"],
    "jj": [DFIT / "train_matrix.parquet", DFIT / "eval_matrix.parquet"],
    "v4jjnd": [DFIT / "train_matrix.parquet", DFIT / "eval_matrix.parquet", RUNS / "v4_novdec_dfit_adsb" / "eval_matrix.parquet"],
    "ay": [RUNS / "allyear_full_adsb_final" / "train_matrix.parquet", RUNS / "allyear_full_adsb_final" / "eval_matrix.parquet"],
    "ay3l": [RUNS / "vm_allyear_v3l_cb_plain" / "train_matrix.parquet", RUNS / "vm_allyear_v3l_cb_plain" / "eval_matrix.parquet"],
    # D-102 (SA-NNISO): ay3l on the isocap_v1 V3L all-year matrix
    "ay3liso": [RUNS / "allyear_iso_v3l_cb_plain" / "train_matrix.parquet", RUNS / "allyear_iso_v3l_cb_plain" / "eval_matrix.parquet"],
}
# SA-BLEND (D-087): the V3L D-fold splits of scripts/legs/nnx/prep.py (``dfit3l_<a>_<b>``): D-fit matrix rows only,
# pool D (the D-fit matrix's own ADS-B block rule), exactly as the m10 / m5 holdout splits.
for _a, _b in ((2, 8), (3, 9), (4, 10), (5, 6)):
    SPLIT_POOL[f"dfit3l_{_a}_{_b}"] = "D"
    SPLIT_MATRICES[f"dfit3l_{_a}_{_b}"] = [DFIT / "train_matrix.parquet"]


def tabular_adsb(split: str) -> pl.DataFrame:
    """MVT id -> the split matrix's ``adsb_coverage`` (null = the row has no ADS-B block, e.g. the
    D-fit matrix's interim-pool days).  A track is kept only where this is non-null and >= 0, so the
    trajectory channel exists exactly where the tabular ADS-B block does."""
    parts = [pl.read_parquet(p, columns=["MVT_ID_mvt", "adsb_coverage"]).with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col("adsb_coverage").cast(pl.Float64)) for p in SPLIT_MATRICES[split]]
    return pl.concat(parts).unique("MVT_ID_mvt", keep="first")
ROW_COLS = ["has_track", "join_src", "log_n_raw", "osm_x", "osm_y", "rwy_x", "rwy_y", "c_x", "c_y"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def all_fixes() -> pl.DataFrame:
    paths = sorted(STAGE1.glob("*/fixes.parquet"))
    return pl.concat([pl.read_parquet(p) for p in paths])


def centroid_columns(rows: pl.DataFrame, pool: str) -> pl.DataFrame:
    """Per-row month-isolated centroid (frame x/y) of pool ``pool`` (build_blocks' rule)."""
    from prc_challenge.adsb_features import stand_centroids
    import extract as ex

    fixes = all_fixes().filter(pl.col("month").is_in(list(POOLS[pool])))
    rows = rows.with_columns(pl.col("day").str.slice(0, 4).cast(pl.Int32).alias("_y"), pl.col("day").str.slice(5, 2).cast(pl.Int32).alias("_m"))
    parts = []
    for (yy, mm), r in rows.group_by(["_y", "_m"], maintain_order=True):
        own = yy == 2025
        cf = fixes.filter(pl.col("month") != mm) if own else fixes
        cents = stand_centroids(cf)
        j = r.select("MVT_ID_mvt", "ADEP_mvt", "stand_key").join(cents, on=["ADEP_mvt", "stand_key"], how="left", maintain_order="left")
        cx = np.full(j.height, np.nan)
        cy = np.full(j.height, np.nan)
        ok = j["c_lat"].is_not_null().to_numpy()
        aps = j["ADEP_mvt"].to_numpy()
        for ap in np.unique(aps[ok]):
            sel = ok & (aps == ap)
            x, y = ex.to_xy(ap, j["c_lat"].to_numpy()[sel], j["c_lon"].to_numpy()[sel])
            cx[sel], cy[sel] = x, y
        parts.append(pl.DataFrame({"MVT_ID_mvt": j["MVT_ID_mvt"], f"c{pool}_x": cx, f"c{pool}_y": cy}))
        log(f"centroids pool {pool} {yy}-{mm:02d}: rows {r.height} centroids {cents.height} with centroid {ok.mean():.3f}")
    return pl.concat(parts)


def consolidate(_a) -> None:
    days = sorted(p.name for p in (ART / "days").iterdir() if (p / "meta.json").exists())
    out = ART / "store"
    out.mkdir(parents=True, exist_ok=True)
    rows_l, pts_l, base = [], [], 0
    mism = 0
    for d in days:
        m = json.loads((ART / "days" / d / "meta.json").read_text())
        mism += m["join_src_mismatch"]
        r = pl.read_parquet(ART / "days" / d / "rows.parquet")
        p = np.load(ART / "days" / d / "points.npy")
        rows_l.append(r.with_columns((pl.col("offset") + base).alias("offset")))
        pts_l.append(p)
        base += p.shape[0]
    rows = pl.concat(rows_l, how="vertical_relaxed")
    if rows["MVT_ID_mvt"].n_unique() != rows.height:
        dup = rows.height - rows["MVT_ID_mvt"].n_unique()
        log(f"WARNING {dup} duplicate MVT ids; keeping the first")
        rows = rows.unique("MVT_ID_mvt", keep="first", maintain_order=True)
    for pool in POOLS:
        rows = rows.join(centroid_columns(rows, pool), on="MVT_ID_mvt", how="left", maintain_order="left")
    points = np.concatenate(pts_l)
    rows.write_parquet(out / "rows.parquet")
    np.save(out / "points.npy", points)
    man = {
        "days": len(days), "first_day": days[0], "last_day": days[-1], "rows": rows.height,
        "tracked_rows": int((rows["n_pts"].fill_null(0) > 0).sum()), "points": int(points.shape[0]),
        "join_src_mismatch_vs_stage1": mism,
        "rows_sha256": sha(out / "rows.parquet"), "points_sha256": sha(out / "points.npy"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "licence": "ODbL-1.0; Contains data from ADSB.lol (https://adsb.lol), licensed under ODbL 1.0",
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    log(json.dumps(man))


def split_cache(a) -> None:
    name = a.name
    if getattr(a, "matrices", None):  # bake (D-092 leg on the bake's own matrices): a custom split
        SPLIT_POOL[name] = a.pool
        SPLIT_MATRICES[name] = [Path(m) for m in a.matrices]
    pool = SPLIT_POOL[name]
    src = RUNS / f"nnx_cache_{name}"
    out = ART / f"seq_{name}"
    out.mkdir(parents=True, exist_ok=True)
    rows = pl.read_parquet(ART / "store" / "rows.parquet")
    points = np.load(ART / "store" / "points.npy", mmap_mode="r")
    cov = tabular_adsb(name)
    info = {"split": name, "pool": pool, "matrices": [str(p) for p in SPLIT_MATRICES[name]], "nnx_cache_meta_sha256": sha(src / "meta.json"), "store": json.loads((ART / "store" / "manifest.json").read_text())}
    for side in ("fit", "ev"):
        ids = np.load(src / f"{side}_ids.npy")
        q = pl.DataFrame({"MVT_ID_mvt": ids.astype(np.float64)})
        j = q.join(rows, on="MVT_ID_mvt", how="left", maintain_order="left")
        assert j.height == len(ids) and np.array_equal(j["MVT_ID_mvt"].to_numpy(), ids)
        j = j.join(cov, on="MVT_ID_mvt", how="left", maintain_order="left")
        tab_ok = (j["adsb_coverage"].is_not_null() & (j["adsb_coverage"].fill_null(-1) >= 0)).to_numpy()
        n_pts_all = j["n_pts"].fill_null(0).to_numpy().astype(np.int64)
        n_pts = np.where(tab_ok, n_pts_all, 0)
        goff = j["offset"].fill_null(0).to_numpy().astype(np.int64)
        lens = n_pts
        loff = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        total = int(lens.sum())
        pts = np.empty((total, points.shape[1]), np.float16)
        # gather each row's points into the side's own contiguous block
        has = np.flatnonzero(lens > 0)
        src_idx = np.concatenate([np.arange(goff[i], goff[i] + lens[i]) for i in has]) if len(has) else np.zeros(0, np.int64)
        pts[:] = points[src_idx] if total else pts
        rowf = np.stack([
            (lens > 0).astype(np.float32),
            np.where(tab_ok, j["join_src"].fill_null(0).to_numpy(), 0).astype(np.float32),
            np.log1p(np.where(tab_ok, j["n_raw"].fill_null(0).to_numpy(), 0).astype(np.float32)),
            j["osm_x"].to_numpy().astype(np.float32), j["osm_y"].to_numpy().astype(np.float32),
            j["rwy_x"].to_numpy().astype(np.float32), j["rwy_y"].to_numpy().astype(np.float32),
            j[f"c{pool}_x"].to_numpy().astype(np.float32), j[f"c{pool}_y"].to_numpy().astype(np.float32),
        ], axis=1)
        np.save(out / f"{side}_trk_off.npy", loff)
        np.save(out / f"{side}_trk_len.npy", lens.astype(np.int16))
        np.save(out / f"{side}_trk_pts.npy", pts)
        np.save(out / f"{side}_trk_row.npy", rowf)
        info[f"{side}_rows"] = len(ids)
        info[f"{side}_in_store"] = int(j["n_pts"].is_not_null().sum())
        info[f"{side}_tracked_before_tab_rule"] = int((n_pts_all > 0).sum())
        info[f"{side}_tab_adsb_rows"] = int(tab_ok.sum())
        info[f"{side}_tracked"] = int((lens > 0).sum())
        info[f"{side}_points"] = total
        info[f"{side}_centroid_share"] = float(np.isfinite(rowf[:, 7]).mean())
        log(f"{name}/{side}: rows {len(ids)} in store {info[f'{side}_in_store']} tracked {info[f'{side}_tracked']} points {total}")
    info["row_cols"] = ROW_COLS
    import extract as ex

    info["point_cols"] = ex.POINT_COLS
    info["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "meta.json").write_text(json.dumps(info, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("consolidate").set_defaults(fn=consolidate)
    s = sub.add_parser("split")
    s.add_argument("name", help=f"one of {sorted(SPLIT_POOL)}, or any nnx_cache_<name> with --matrices and --pool")
    s.add_argument("--matrices", nargs="+", help="bake: the split's matrices (adsb_coverage availability rule), in cache order")
    s.add_argument("--pool", choices=sorted(POOLS), help="bake: the centroid pool of a custom split (D for D-fit rows, AY for all-year)")
    s.set_defaults(fn=split_cache)
    a = p.parse_args(argv)
    if a.cmd == "split" and a.name not in SPLIT_POOL and not (a.matrices and a.pool):
        p.error(f"split {a.name!r} is not registered: pass --matrices and --pool")
    a.fn(a)


if __name__ == "__main__":
    main()
