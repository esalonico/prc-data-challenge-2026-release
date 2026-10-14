# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""ADS-B coverage extension: the v3 block (travel-time map, position kinematics, runway queue).

Reads the stage-1 ground blocks of ``scripts/features/build_adsb_features.py`` (main checkout artifacts,
``ADSB-features-20260925/stage1``) and writes
``artifacts/experiments/ADSB-coverage-20260926/``.

Commands
--------
kin      per month: kinematics of every stage-1 sample (label-free), per-row kinematic and
         runway-queue features, and the moving samples keyed by (cell, heading sector).
votes    2025 months 1-10 only: per (row, stand, cell, heading) median time since the labelled
         BLOCK (these are the only label-derived quantities; Nov/Dec 2025 labels are never read).
block    month-isolated v3 blocks: ``dfit`` (map pool = D months) and ``allyear`` (pool = 2025
         months 1-10). Rows of 2025 month m use votes from pool months != m; ranking and Nov/Dec
         rows use the whole pool. Row sets are aligned to the v2 prize files. Also writes the
         Aug-Oct screen block (D-fit rows restricted to the screen rows).
report   forensic accuracy on Jan/Jul 2025 (D-fit block, pool never includes Jan/Jul), D-month
         accuracy by coverage class, 2026 label-free coverage gain, 2026-vs-2025 offset drift.

Licence: ODbL 1.0 data, "Contains data from ADSB.lol (https://adsb.lol), licensed under ODbL 1.0";
runway geometry "(c) OpenStreetMap contributors" (ODbL). OpenSky is not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "features"))

from prc_challenge.adsb_coverage import (  # noqa: E402
    HDG_SECTORS,
    MAP_MIN_N_AP,
    MAP_MIN_N_STAND,
    MAP_TRAIN_HI_S,
    MAP_TRAIN_LO_S,
    kinematics,
    load_runway_lines,
    map_estimates,
    map_row_estimates,
    row_features,
)

MAIN = ROOT
ART = MAIN / "artifacts" if not (ROOT / "artifacts" / "experiments" / "ADSB-features-20260925").exists() \
    else ROOT / "artifacts"
V2 = ART / "experiments" / "ADSB-features-20260925"
OUT = ART / "experiments" / "ADSB-coverage-20260926"
STAGE1 = V2 / "stage1"
D_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)
LABEL_MONTHS = tuple(range(1, 11))  # Nov/Dec 2025 labels are never read
ATTRIBUTION = "Contains data from ADSB.lol (https://adsb.lol), licensed under ODbL 1.0"
PREFIX = "adsb3_"


def month_tags() -> list[str]:
    return sorted({p.name[:7] for p in STAGE1.iterdir() if p.is_dir()})


def raw_file(tag: str) -> Path:
    y, m = int(tag[:4]), int(tag[5:7])
    if y == 2026:
        return ROOT / "data" / "raw" / "ranking.parquet"
    return next((ROOT / "data" / "raw").glob(f"training_{y}-{m:02d}-01_*.parquet"))


def kin(args) -> None:
    lines = load_runway_lines(ROOT / "data" / "external" / "osm_surface")
    d = OUT / "stage3"
    d.mkdir(parents=True, exist_ok=True)
    for tag in month_tags():
        if args.months and tag not in args.months.split(","):
            continue
        if (d / f"rows_{tag}.parquet").exists() and not args.force:
            continue
        t0 = time.time()
        days = sorted(p for p in STAGE1.iterdir() if p.is_dir() and p.name.startswith(tag))
        rows = pl.concat([pl.read_parquet(p / "rows.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "stand_key", "day"])
                          for p in days], how="diagonal_relaxed")
        seg = pl.concat([pl.read_parquet(p / "seg.parquet") for p in days])
        # RUNWAY_mvt is a label-free movement field; BLOCK/TAXITIME are not read here
        rwy = (pl.scan_parquet(raw_file(tag)).filter(pl.col("PHASE_mvt") == "DEP")
               .select("MVT_ID_mvt", "ADEP_mvt", pl.col("RUNWAY_mvt").cast(pl.String).str.to_uppercase())
               .collect().join(rows.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi"))
        k = kinematics(seg, rows)
        rf = row_features(k, rwy, lines)
        rows.join(rf, on="MVT_ID_mvt", how="left").write_parquet(d / f"rows_{tag}.parquet")
        k.filter(pl.col("moving")).select(
            "MVT_ID_mvt", "ADEP_mvt", "t_rel", "cx", "cy", "hs", "moving"
        ).write_parquet(d / f"k_{tag}.parquet", compression="zstd")
        print(f"kin {tag}: days={len(days)} rows={rows.height} samples={seg.height} {time.time() - t0:.0f}s",
              flush=True)


def labels_month(m: int) -> pl.DataFrame:
    if m not in LABEL_MONTHS:
        raise SystemExit("Nov/Dec 2025 labels are locked")
    return (pl.scan_parquet(raw_file(f"2025-{m:02d}"))
            .filter(pl.col("PHASE_mvt") == "DEP")
            .select("MVT_ID_mvt", ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds()
                                   .cast(pl.Float64)).alias("block_rel"))
            .filter(pl.col("block_rel").is_not_null()).collect())


def votes(args) -> None:
    d = OUT / "stage3"
    for m in LABEL_MONTHS:
        tag = f"2025-{m:02d}"
        if not (d / f"k_{tag}.parquet").exists():
            continue
        k = pl.read_parquet(d / f"k_{tag}.parquet")
        rows = pl.read_parquet(d / f"rows_{tag}.parquet", columns=["MVT_ID_mvt", "stand_key"])
        e = (k.join(labels_month(m), on="MVT_ID_mvt", how="inner")
             .join(rows, on="MVT_ID_mvt", how="left")
             .with_columns((pl.col("t_rel") - pl.col("block_rel")).alias("el"))
             .filter(pl.col("el").is_between(MAP_TRAIN_LO_S, MAP_TRAIN_HI_S)))
        v = (e.filter(pl.col("hs").is_not_null())
             .group_by("MVT_ID_mvt", "ADEP_mvt", "stand_key", "cx", "cy", "hs").agg(pl.col("el").median()))
        v3 = e.group_by("MVT_ID_mvt", "ADEP_mvt", "cx", "cy").agg(pl.col("el").median())
        v.write_parquet(d / f"votes_{tag}.parquet")
        v3.write_parquet(d / f"votes3_{tag}.parquet")
        print(f"votes {tag}: rows={e['MVT_ID_mvt'].n_unique()} votes={v.height} votes3={v3.height}", flush=True)


def _agg(v: pl.DataFrame, keys: list[str], min_n: int) -> pl.DataFrame:
    return (v.group_by(keys)
            .agg(pl.col("el").median().alias("m_el"), pl.len().alias("m_n"),
                 pl.col("el").quantile(0.25).alias("q25"), pl.col("el").quantile(0.75).alias("q75"))
            .filter(pl.col("m_n") >= min_n)
            .with_columns(((pl.col("q75") - pl.col("q25")) / 1.349).alias("m_sd")).drop("q25", "q75"))


ROW_KEEP = ["dwell_end_rel", "dwell_dur", "dwell_next_rel", "rev_rel", "dep_first_rel", "dep_first_v",
            "dep_first_rwy_m", "rwy_zone_rel", "rwy_stop_s", "last_stop_end_rel", "n_stops", "stop_s", "path_m",
            "mov_s", "dep_n"]


def aobt_rel(tag: str) -> pl.DataFrame:
    """AOBT_3_flt - MVT (s), label-free planning field cleared under D-017 (as in the v2 block)."""
    return (pl.scan_parquet(raw_file(tag)).filter(pl.col("PHASE_mvt") == "DEP")
            .select("MVT_ID_mvt", (pl.col("AOBT_3_flt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds()
                    .cast(pl.Float64).alias("aobt_rel")).collect())


def build_block(pool: tuple[int, ...]) -> pl.DataFrame:
    d = OUT / "stage3"
    V = {m: pl.read_parquet(d / f"votes_{2025}-{m:02d}.parquet").with_columns(pl.lit(m).alias("_m"))
         for m in pool if (d / f"votes_2025-{m:02d}.parquet").exists()}
    V3 = {m: pl.read_parquet(d / f"votes3_{2025}-{m:02d}.parquet").with_columns(pl.lit(m).alias("_m"))
          for m in pool if (d / f"votes3_2025-{m:02d}.parquet").exists()}
    parts = []
    for tag in month_tags():
        t0 = time.time()
        y, m = int(tag[:4]), int(tag[5:7])
        use = [x for x in V if not (y == 2025 and x == m)]
        v = pl.concat([V[x] for x in use])
        v3 = pl.concat([V3[x] for x in use])
        st = _agg(v, ["ADEP_mvt", "stand_key", "cx", "cy", "hs"], MAP_MIN_N_STAND)
        ap = _agg(v, ["ADEP_mvt", "cx", "cy", "hs"], MAP_MIN_N_AP)
        a3 = _agg(v3, ["ADEP_mvt", "cx", "cy"], MAP_MIN_N_AP)
        k = pl.read_parquet(d / f"k_{tag}.parquet")
        rows = pl.read_parquet(d / f"rows_{tag}.parquet")
        e = map_estimates(k, rows, st, ap, a3)
        main = map_row_estimates(e.filter(pl.col("lvl") <= 2), None, "map")
        l3 = e.filter(pl.col("lvl") == 3).sort("MVT_ID_mvt", "t_rel").group_by("MVT_ID_mvt").agg(
            pl.col("est").first().alias("maph_first_rel"), pl.col("sd").first().alias("maph_first_sd"))
        f = (rows.select("MVT_ID_mvt", *ROW_KEEP)
             .join(main, on="MVT_ID_mvt", how="left").join(l3, on="MVT_ID_mvt", how="left")
             .join(aobt_rel(tag), on="MVT_ID_mvt", how="left")
             .with_columns((pl.col("aobt_rel") - pl.col("map_first_rel")).alias("aobt_minus_map"),
                           (pl.col("map_first_t") - pl.col("map_first_rel")).alias("map_seen_after_s"))
             .drop("aobt_rel"))
        parts.append(f.select("MVT_ID_mvt", *[pl.col(c).cast(pl.Float64).alias(PREFIX + c)
                                              for c in f.columns if c != "MVT_ID_mvt"]))
        print(f"block {tag}: pool={use} map st={st.height} ap={ap.height} a3={a3.height} rows={rows.height} "
              f"map_cov={main.height / max(rows.height, 1):.2f} {time.time() - t0:.0f}s", flush=True)
    out = pl.concat(parts, how="diagonal_relaxed")
    if out["MVT_ID_mvt"].n_unique() != out.height:
        raise SystemExit("duplicate MVT_ID_mvt")
    return out


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def block(args) -> None:
    vdir = OUT / "v3"
    vdir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=False).stdout.strip()
    for tag, pool in (("dfit", D_MONTHS), ("allyear", LABEL_MONTHS)):
        if args.tags and tag not in args.tags.split(","):
            continue
        b = build_block(pool)
        prize = V2 / "prize" / f"adsb_block_prize_{tag}.parquet"
        ids = pl.read_parquet(prize, columns=["MVT_ID_mvt"])
        aligned = ids.join(b, on="MVT_ID_mvt", how="left")
        if aligned.height != ids.height:
            raise SystemExit("row set changed")
        path = vdir / f"adsb_v3_block_{tag}.parquet"
        aligned.write_parquet(path)
        sha = _sha(path)
        man = {
            "source_mode": "adsb_lol", "licence": "ODbL-1.0", "attribution": ATTRIBUTION,
            "osm_attribution": "(c) OpenStreetMap contributors (runway centrelines, ODbL)",
            "decision": "D-017 (adsb.lol cleared for the prize path)", "file": str(path), "sha256": sha,
            "rows": aligned.height, "rows_with_v3": int(aligned[PREFIX + "dep_n"].is_not_null().sum()),
            "rows_with_map": int(aligned[PREFIX + "map_first_rel"].is_not_null().sum()),
            "columns": aligned.columns, "tag": tag, "map_pool_months_2025": list(pool),
            "row_set": f"identical to {prize.name} (sha256 {_sha(prize)})",
            "isolation": "month-isolated: rows of 2025 month m use travel-time votes from pool months != m; "
                         "ranking and Nov/Dec 2025 rows use the whole pool; no ranking row is learned from; "
                         "Nov/Dec 2025 labels never read",
            "label_use": "only the travel-time map (median time since BLOCK per stand/cell/heading) uses labels",
            "params": {"cell_m": 50, "heading_sectors": HDG_SECTORS, "min_n_stand": MAP_MIN_N_STAND,
                       "min_n_airport": MAP_MIN_N_AP, "train_elapsed_window_s": [MAP_TRAIN_LO_S, MAP_TRAIN_HI_S]},
            "code_commit": commit, "stage1_source": str(STAGE1),
        }
        (vdir / f"adsb_v3_block_{tag}.manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        print(f"v3 {tag}: {path} rows={aligned.height} sha256={sha}", flush=True)
        if tag == "dfit":
            rows = pl.read_parquet(V2 / "features" / "screen_rows_aug_oct.parquet", columns=["MVT_ID_mvt"])
            sp = vdir / "adsb_v3_block_screen_aug_oct.parquet"
            rows.join(b, on="MVT_ID_mvt", how="left").write_parquet(sp)
            print(f"screen block {sp} sha256={_sha(sp)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["kin", "votes", "block", "report", "lean"])
    ap.add_argument("--months", default="")
    ap.add_argument("--tags", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.command == "kin":
        kin(args)
    elif args.command == "votes":
        votes(args)
    elif args.command == "block":
        block(args)
    elif args.command == "lean":
        vdir = OUT / "v3"
        for name in [n for n in ("screen_aug_oct", "dfit", "allyear") if not args.tags or n in args.tags.split(",")]:
            sha = lean(vdir / f"adsb_v3_block_{name}.parquet", vdir / f"adsb_v3lean_block_{name}.parquet")
            print(f"lean {name}: sha256={sha}", flush=True)
    elif args.command == "report":
        from adsb_coverage_report import report

        report(OUT, V2)



# --------------------------------------------------------------------------- lean variant (prereg amendment 2)

LEAN_COLS = ["map_first_rel", "map_first_sd", "map_wmean_rel", "map_sd_min", "map_first_t", "map_n", "aobt_minus_map"]
LEAN_MASK_AIRPORTS = ("LIRF", "LFPG", "LTFM")


def lean(src: Path, dst: Path) -> str:
    """Map-core columns only; rows at LIRF/LFPG/LTFM masked to null (weak or absent map)."""
    b = pl.read_parquet(src, columns=["MVT_ID_mvt", *[PREFIX + c for c in LEAN_COLS]])
    ap = pl.concat([
        pl.scan_parquet(p).filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", "ADEP_mvt").collect()
        for p in sorted((ROOT / "data" / "raw").glob("training_2025-*.parquet")) + [ROOT / "data" / "raw" / "ranking.parquet"]
    ]).unique("MVT_ID_mvt")
    mask = pl.col("ADEP_mvt").is_in(list(LEAN_MASK_AIRPORTS)).fill_null(False)
    out = b.join(ap, on="MVT_ID_mvt", how="left").with_columns(
        [pl.when(mask).then(None).otherwise(pl.col(PREFIX + c)).alias(PREFIX + c) for c in LEAN_COLS]
    ).drop("ADEP_mvt")
    if out.height != b.height:
        raise SystemExit("row set changed")
    out.write_parquet(dst)
    return _sha(dst)


if __name__ == "__main__":
    main()
