# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Append-merge the trajectory store for the four serve months (final phase).

The shipped store (``ADSB-trajectory-20260927/store/{rows.parquet, points.npy}``) holds every 2025 day plus the
public Jan/Jul 2026 days; its per-day directories were not shipped, so ``tracks.py consolidate`` cannot be rerun.
This keeps the shipped rows of 2025 days (offsets unchanged), drops the old 2026 rows, appends the freshly
extracted serve days (``extract.py --days ...`` -> ``days/<day>/``) with their point offsets shifted past the shipped
points, and adds the month-isolated centroid columns of both pools with ``tracks.centroid_columns`` (2026 rows: the
whole pool), exactly as ``consolidate`` does.

Usage (serve workspace, PRC_MAIN_REPO set): python pipeline/steps/traj_store_merge.py --days-glob '2026-*'
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

MAIN = Path(os.environ["PRC_MAIN_REPO"])
sys.path.insert(0, str(MAIN / "scripts" / "legs" / "traj"))
import tracks  # noqa: E402

ART = tracks.ART


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-glob", default="2026-*")
    ap.add_argument("--shipped", default=str(ART / "store.shipped"))
    a = ap.parse_args()
    shipped = Path(a.shipped)
    old_rows = pl.read_parquet(shipped / "rows.parquet")
    old_pts = np.load(shipped / "points.npy", mmap_mode="r")
    keep = old_rows.filter(~pl.col("day").str.starts_with("2026"))
    days = sorted(p.name for p in (ART / "days").glob(a.days_glob) if (p / "meta.json").exists())
    base = old_pts.shape[0]
    rows_l, pts_l, mism = [], [], 0
    for d in days:
        m = json.loads((ART / "days" / d / "meta.json").read_text())
        mism += m["join_src_mismatch"]
        r = pl.read_parquet(ART / "days" / d / "rows.parquet")
        p = np.load(ART / "days" / d / "points.npy")
        rows_l.append(r.with_columns((pl.col("offset") + base).alias("offset")))
        pts_l.append(p)
        base += p.shape[0]
    new = pl.concat(rows_l, how="vertical_relaxed")
    if new["MVT_ID_mvt"].n_unique() != new.height:
        new = new.unique("MVT_ID_mvt", keep="first", maintain_order=True)
    for pool in tracks.POOLS:
        new = new.join(tracks.centroid_columns(new, pool), on="MVT_ID_mvt", how="left", maintain_order="left")
    new = new.select([pl.col(c).cast(t) for c, t in keep.schema.items()])
    rows = pl.concat([keep, new], how="vertical")
    if rows["MVT_ID_mvt"].n_unique() != rows.height:
        raise SystemExit("duplicate MVT_ID_mvt across shipped 2025 rows and serve rows")
    points = np.concatenate([np.asarray(old_pts), *pts_l])
    out = ART / "store"
    out.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(out / "rows.parquet")
    np.save(out / "points.npy", points)
    man = {"merge": "shipped store 2025 rows + serve days (pipeline/steps/traj_store_merge.py)",
           "shipped_rows_kept": keep.height, "shipped_rows_dropped": old_rows.height - keep.height,
           "serve_days": len(days), "first_serve_day": days[0], "last_serve_day": days[-1], "serve_rows": new.height,
           "serve_tracked_rows": int((new["n_pts"].fill_null(0) > 0).sum()), "rows": rows.height,
           "points": int(points.shape[0]), "join_src_mismatch_vs_stage1_serve_days": mism,
           "rows_sha256": sha(out / "rows.parquet"), "points_sha256": sha(out / "points.npy"),
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "licence": "ODbL-1.0; Contains data from ADSB.lol (https://adsb.lol), licensed under ODbL 1.0"}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man))


if __name__ == "__main__":
    main()
