# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Build the ADS-B (adsb.lol) off-block feature block from locally synced day files.

Research mode (D-012): no ADS-B feature enters a ranking file until an eligibility
decision is logged. OpenSky is not used. Nov/Dec 2025 (V4) departures are never read.

Commands
--------
stage1    per UTC day: join departures to aircraft, cut departure ground blocks
          (label-free), and on D-month days extract BLOCK fixes for centroids.
          Incremental: a day is redone when its D-1 / D+1 neighbour file appears.
features  month-isolated block keyed by MVT_ID_mvt. Stand centroids for rows of
          month m come from D-month fixes of other months only; Jan/Jul 2025 and
          ranking rows use centroids from all D-month fixes.
report    2026 coverage per airport, Jan/Jul 2025 sensor accuracy (forensic only),
          per-airport per-month bias stability on D months.

Outputs: artifacts/experiments/ADSB-features-20260925/{stage1,features,report}/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import UTC, date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from prc_challenge.adsb_features import (
    FEATURE_COLS,
    add_first_and_aobt,
    add_taxi_hat,
    block_fixes,
    day_files,
    day_segments,
    first_bias,
    gate_bias,
    gate_features,
    join_aircraft,
    load_samples,
    normalize_departures,
    stand_centroids,
    surface_features,
    takeoff_match,
    to_block,
)
from prc_challenge.geometry import load_stand_coords
from prc_challenge.io import CHALLENGE_AIRPORTS

FILT = ROOT / "data" / "external" / "adsb_lol" / "slim"
OUT = ROOT / "artifacts" / "experiments" / "ADSB-features-20260925"
D_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)
TRAIN_MONTHS_OK = tuple(range(1, 11))  # Nov/Dec 2025 (V4) never read
STAGE1_VERSION = 5  # 5: dictionary callsigns (join_src 4) for rows without CALLSIGN_flt
ATTRIBUTION = "Contains data from ADSB.lol (https://adsb.lol), licensed under ODbL 1.0"
DEP_COLS = ["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "STAND_mvt",
            "FLIGHT_ID_mvt", "RUNWAY_mvt"]


_THR: list[pl.DataFrame] = []


def _thresholds() -> pl.DataFrame:
    if not _THR:
        from prc_challenge.runways import load_runway_thresholds

        _THR.append(load_runway_thresholds(ROOT / "data" / "external" / "ourairports" / "runways.csv"))
    return _THR[0]


def _epoch_day(d: date) -> float:
    from datetime import datetime

    return datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()


SERVE_MONTHS = tuple(int(m) for m in os.environ.get("PRC_SERVE_MONTHS", "1,7").split(","))  # final phase: 1,2,6,7


def scope_of(d: date) -> str | None:
    if d.year == 2026 and d.month in SERVE_MONTHS:
        return "ranking"
    if d.year == 2025 and d.month in D_MONTHS:
        return "D"
    if d.year == 2025 and d.month in (1, 7):
        return "janjul"
    if d.year == 2025 and d.month in (11, 12):
        return "v4"  # label-free stage1 only; rows appear in the all-year prize file, never in screens
    return None


def training_file(y: int, m: int) -> Path:
    return next((ROOT / "data" / "raw").glob(f"training_{y}-{m:02d}-01_*.parquet"))


_DEP_CACHE: dict[tuple[int, int], pl.DataFrame] = {}


def departures(d: date, with_labels: bool) -> pl.DataFrame:
    key = (d.year, d.month)
    if key not in _DEP_CACHE:
        _DEP_CACHE.clear()
        if d.year == 2026:
            src, cols = ROOT / "data" / "raw" / "ranking.parquet", DEP_COLS
        else:
            if d.month not in TRAIN_MONTHS_OK and with_labels:
                raise SystemExit("V4 months are locked: labels are never read")
            src = training_file(d.year, d.month)
            cols = DEP_COLS + (["BLOCK_TIME_UTC_mvt"] if d.month in TRAIN_MONTHS_OK else [])
        _DEP_CACHE[key] = (
            pl.scan_parquet(src)
            .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(list(CHALLENGE_AIRPORTS)))
            .select(cols)
            .collect()
        )
    dep = _DEP_CACHE[key].filter(pl.col("MVT_TIME_UTC_mvt").dt.date() == d)
    if not with_labels and "BLOCK_TIME_UTC_mvt" in dep.columns:
        dep = dep.drop("BLOCK_TIME_UTC_mvt")
    if d.month in (11, 12) and d.year == 2025 and "BLOCK_TIME_UTC_mvt" in dep.columns:
        raise SystemExit("V4 label column present")
    return dep


_DICT: list[pl.DataFrame | None] = []


def callsign_dictionary() -> pl.DataFrame | None:
    """features/callsign_dictionary.parquet from scripts/features/adsb_callsign_dictionary.py (airports that
    passed the 97% precision gate). None when absent."""
    if not _DICT:
        p = OUT / "features" / "callsign_dictionary.parquet"
        _DICT.append(pl.read_parquet(p) if p.exists() else None)
    return _DICT[0]


def with_dict_callsign(dep: pl.DataFrame) -> pl.DataFrame:
    """cs_dict for rows without CALLSIGN_flt: dictionary callsign of FLIGHT_mvt when the hour fits."""
    dic = callsign_dictionary()
    if dic is None:
        return dep
    hour = pl.col("MVT_TIME_UTC_mvt").dt.hour()
    j = dep.join(dic.rename({"fl": "cs_mvt", "cs": "cs_dict"}), on=["ADEP_mvt", "cs_mvt"], how="left")
    ok = pl.col("cs_flt").is_null() & (hour >= pl.col("hour_lo") - 2) & (hour <= pl.col("hour_hi") + 2)
    return j.with_columns(pl.when(ok).then(pl.col("cs_dict")).otherwise(None).alias("cs_dict")).drop(
        "n", "hour_lo", "hour_hi")


def stage1(args) -> None:
    days = sorted({date.fromisoformat(re.findall(r"\d{4}-\d{2}-\d{2}", p.name)[0])
                   for p in FILT.glob("adsb_lol_*_slim.parquet")})
    todo = [d for d in days if scope_of(d) is not None]
    if args.scope:
        todo = [d for d in todo if scope_of(d) in args.scope.split(",")]
    root = OUT / "stage1"
    for d in todo:
        files = day_files(FILT, d)
        meta_p = root / d.isoformat() / "meta.json"
        state = {"prev": files.prev is not None, "next": files.nxt is not None,
                 "cur_bytes": files.cur.stat().st_size, "source": files.cur.name, "version": STAGE1_VERSION}
        if meta_p.exists() and not args.force:
            old = json.loads(meta_p.read_text())
            if all(old.get(k) == v for k, v in state.items()):
                continue
        t0 = time.time()
        scope = scope_of(d)
        dep = with_dict_callsign(normalize_departures(departures(d, with_labels=(scope in ("D", "janjul")))))
        s = load_samples(files)
        dep = join_aircraft(dep, s)
        rows, seg = day_segments(dep, s)
        surf = surface_features(
            dep.select("MVT_ID_mvt", "ADEP_mvt", pl.col("RUNWAY_mvt").cast(pl.String).str.to_uppercase(), "mvt",
                       "hex"), s, _thresholds(), _epoch_day(d))
        rows = dep.select("MVT_ID_mvt", "ADEP_mvt", "stand_key", "join_src", "mvt").join(
            rows, on="MVT_ID_mvt", how="left"
        ).join(surf, on="MVT_ID_mvt", how="left").with_columns(
            pl.lit(d.isoformat()).alias("day"),
            pl.lit(int(files.prev is not None)).cast(pl.Int8).alias("prev_day_present"),
            (pl.col("n_seg").fill_null(0) > 0).mean().over("ADEP_mvt").alias("day_ground_cov"),
        )
        out = root / d.isoformat()
        out.mkdir(parents=True, exist_ok=True)
        rows.write_parquet(out / "rows.parquet")
        seg.write_parquet(out / "seg.parquet", compression="zstd")
        n_fix = 0
        if scope in ("D", "janjul"):  # janjul fixes feed only the all-year prize pool
            labels = dep.select(
                "MVT_ID_mvt", "ADEP_mvt", "stand_key",
                ((pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("ms") / 1000.0) - pl.col("mvt")).alias("block_rel"),
            ).filter(pl.col("block_rel").is_not_null()).with_columns(
                pl.lit(d.isoformat()).alias("day"), pl.lit(d.month).cast(pl.Int8).alias("month"))
            fixes = block_fixes(seg, labels)
            fixes.write_parquet(out / "fixes.parquet")
            n_fix = fixes.height
        state.update({"scope": scope, "n_dep": dep.height, "n_joined": int((dep["join_src"] > 0).sum()),
                      "n_seg_rows": int((rows["n_seg"].fill_null(0) > 0).sum()), "n_fix": n_fix,
                      "secs": round(time.time() - t0, 1)})
        meta_p.write_text(json.dumps(state, indent=1) + "\n")
        print(f"stage1 {d} {scope}: dep={dep.height} joined={state['n_joined']} seg={state['n_seg_rows']} "
              f"fix={n_fix} {state['secs']}s", flush=True)


def _load_stage1(scope: str) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    root = OUT / "stage1"
    days = sorted(p.name for p in root.iterdir() if p.is_dir() and scope_of(date.fromisoformat(p.name)) == scope)
    if not days:
        return pl.DataFrame(), pl.DataFrame(), []
    rows = pl.concat([pl.read_parquet(root / d / "rows.parquet") for d in days], how="diagonal_relaxed")
    seg = pl.concat([pl.read_parquet(root / d / "seg.parquet") for d in days])
    return rows, seg, days


def _all_fixes() -> pl.DataFrame:
    paths = sorted((OUT / "stage1").glob("*/fixes.parquet"))
    return pl.concat([pl.read_parquet(p) for p in paths]) if paths else pl.DataFrame()


LINKER_MIN_PRECISION = 0.95
LINKER_SAMPLE_DAYS = 4


def aobt_frame(months_2025: tuple[int, ...], ranking: bool) -> pl.DataFrame:
    """aobt_rel = AOBT_3_flt - MVT (seconds), label-free planning field (cleared, D-017)."""
    srcs = [training_file(2025, m) for m in months_2025]
    if ranking:
        srcs.append(ROOT / "data" / "raw" / "ranking.parquet")
    parts = [
        pl.scan_parquet(p).filter(pl.col("PHASE_mvt") == "DEP")
        .select("MVT_ID_mvt", (pl.col("AOBT_3_flt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds()
                .cast(pl.Float64).alias("aobt_rel")).collect()
        for p in srcs
    ]
    return pl.concat(parts).unique("MVT_ID_mvt")


def linker_precision(force: bool = False) -> dict:
    """Per-airport precision of the take-off-time fallback join, measured on callsign-joined
    departures whose hex is hidden and re-matched (label-free). Cached in features/."""
    out = OUT / "features" / "linker_precision.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    days = sorted(p.name for p in (OUT / "stage1").iterdir() if p.is_dir())
    sample = []
    for scope in ("D", "janjul", "ranking"):
        sd = [d for d in days if scope_of(date.fromisoformat(d)) == scope]
        sample += sd[:: max(1, len(sd) // LINKER_SAMPLE_DAYS)][:LINKER_SAMPLE_DAYS]
    tot: dict[str, list[int]] = {}
    for d in sample:
        dd = date.fromisoformat(d)
        dep = with_dict_callsign(normalize_departures(departures(dd, with_labels=False)))
        s = load_samples(day_files(FILT, dd))
        dep = join_aircraft(dep, s)
        # hide a deterministic 10% of the callsign-joined rows; the other 90% keep their
        # hex and claim their take-offs, as they would for a real unmatched row
        hide = (pl.col("MVT_ID_mvt").hash(seed=3) % 10) == 0
        work = dep.select("MVT_ID_mvt", "ADEP_mvt", "mvt", "hex", "join_src").with_columns(
            pl.col("hex").alias("hex_cs"),
            pl.when(hide & (pl.col("join_src") == 1)).then(None).otherwise(pl.col("hex")).alias("hex"),
            pl.when(hide & (pl.col("join_src") == 1)).then(0).otherwise(pl.col("join_src")).cast(pl.Int8)
            .alias("join_src"))
        re = takeoff_match(work, s).filter(hide & (pl.col("hex_cs").is_not_null()) & (pl.col("join_src") == 3))
        for ap, g in re.group_by("ADEP_mvt"):
            t = tot.setdefault(ap[0], [0, 0])
            t[0] += int((g["hex"] == g["hex_cs"]).sum())
            t[1] += g.height
    res = {"sample_days": sample, "min_precision": LINKER_MIN_PRECISION,
           "airports": {ap: {"agree": a, "n": n, "precision": a / n if n else None,
                             "keep_fallback": bool(n and a / n >= LINKER_MIN_PRECISION)}
                        for ap, (a, n) in sorted(tot.items())}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    return res


def apply_linker_gate(feat: pl.DataFrame, keep_airports: set[str]) -> pl.DataFrame:
    """Fallback-joined rows (join_src 3) at airports below the precision gate lose their
    aircraft-specific features; the airport-level surface load columns stay."""
    bad = (pl.col("join_src") == 3) & ~pl.col("ADEP_mvt").is_in(sorted(keep_airports))
    keep_cols = {"join_src", "coverage", "prev_day_present", "day_ground_hex", "day_ground_cov"}
    cols = [c for c in FEATURE_COLS if c in feat.columns and c not in keep_cols and not c.startswith("surf_")]
    return feat.with_columns(
        *[pl.when(bad).then(None).otherwise(pl.col(c)).alias(c) for c in cols],
        pl.when(bad).then(0).otherwise(pl.col("join_src")).cast(pl.Int8).alias("join_src"),
        pl.when(bad).then(0).otherwise(pl.col("coverage")).cast(pl.Int8).alias("coverage"),
    )


def build_blocks(pool_months: tuple[int, ...], summary: dict) -> dict[str, pl.DataFrame]:
    """Month-isolated blocks per scope.

    Stand centroids and the per-airport gate bias behind ``adsb_taxi_hat`` are learned
    from labelled 2025 rows of ``pool_months`` *other than the row's own month*.
    Ranking rows (and pool-external months) use the whole pool. Nothing is ever
    learned from ranking rows.
    """
    osm = load_stand_coords(ROOT / "data" / "external" / "osm" / "parking_positions.parquet")
    fixes = _all_fixes()
    if fixes.height:
        fixes = fixes.filter(pl.col("month").is_in(list(pool_months)))
    summary["fix_months"] = sorted(fixes["month"].unique().to_list()) if fixes.height else []
    summary["pool_months"] = list(pool_months)
    feats: dict[str, dict[str, pl.DataFrame]] = {}
    for scope in ("D", "janjul", "ranking", "v4"):
        rows, seg, days = _load_stage1(scope)
        if not days:
            continue
        summary[f"days_{scope}"] = days
        rows = rows.with_columns(pl.col("day").str.slice(0, 7).alias("ym"))
        feats[scope] = {}
        for ym in sorted(rows["ym"].unique().to_list()):
            m, own = int(ym[5:7]), ym.startswith("2025")
            r = rows.filter(pl.col("ym") == ym)
            cf = fixes.filter(pl.col("month") != m) if (own and fixes.height) else fixes
            cents = stand_centroids(cf)
            sg = seg.join(r.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi")
            feats[scope][ym] = gate_features(r, sg, cents, osm).with_columns(
                pl.lit(m).cast(pl.Int8).alias("_m"))
            summary[ym] = {"rows": r.height, "n_centroids": cents.height}
    pool_feat = pl.concat([f for sc in feats.values() for ym, f in sc.items()
                           if ym.startswith("2025") and int(ym[5:7]) in pool_months], how="diagonal_relaxed") \
        if any(ym.startswith("2025") and int(ym[5:7]) in pool_months for sc in feats.values() for ym in sc) else None
    lab = labels_for(pool_months).filter(pl.col("matched") & pl.col("block_rel").is_not_null()) \
        if pool_feat is not None else None
    lp = linker_precision()
    keep_ap = {ap for ap, v in lp["airports"].items() if v["keep_fallback"]}
    summary["linker_keep_fallback"] = sorted(keep_ap)
    aobt = aobt_frame(tuple(range(1, 13)) if "v4" in feats else TRAIN_MONTHS_OK, ranking="ranking" in feats)
    out = {}
    for scope, per in feats.items():
        parts = []
        for ym, f in per.items():
            m = int(ym[5:7])
            fb = None
            if pool_feat is not None:
                src = pool_feat.filter(pl.col("_m") != m) if ym.startswith("2025") else pool_feat
                f = add_taxi_hat(f, gate_bias(src, lab))
                fb = first_bias(src, lab)
            f = add_first_and_aobt(f, fb, aobt)
            f = apply_linker_gate(f, keep_ap)
            blk = to_block(f)
            summary[ym]["coverage3"] = float((blk["adsb_coverage"] == 3).mean())
            parts.append(blk.with_columns(pl.lit(ym).alias("_ym")))
        out[scope] = pl.concat(parts, how="diagonal_relaxed")
    return out


def features(args) -> None:
    """Screen/report blocks: pool = D months only (Jan/Jul 2025 labels never used)."""
    fout = OUT / "features"
    fout.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    blocks = build_blocks(D_MONTHS, summary)
    for scope, b in blocks.items():
        if scope == "v4":
            continue
        for ym in sorted(b["_ym"].unique().to_list()):
            b.filter(pl.col("_ym") == ym).drop("_ym").write_parquet(fout / f"adsb_block_{ym}.parquet")
        b.drop("_ym").write_parquet(fout / f"adsb_block_{scope}.parquet")
        (fout / f"covered_days_{scope}.json").write_text(json.dumps(summary[f"days_{scope}"]) + "\n")
        print(f"features {scope}: {b.height} rows, {len(summary[f'days_{scope}'])} days", flush=True)
    (fout / "features_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


def prize(args) -> None:
    """Prize-mode files for train_s4_allyear.py --extra-features (D-017).

    ``dfit``    pool = D months: for the D-fit replay scored on Jan+Jul 2025.
    ``allyear`` pool = 2025 months 1-10: for the all-year refit predicting ranking.
    Rows: every 2025 month 1-10 departure on an ingested day, plus both ranking
    months; everything else (Nov/Dec 2025, missing days) is absent = NaN in the join.
    ``adsb_airport`` is dropped (AIRPORT is already in the S4 stack).
    """
    import hashlib
    import subprocess

    pdir = OUT / "prize"
    pdir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    for tag, pool in (("dfit", D_MONTHS), ("allyear", TRAIN_MONTHS_OK)):
        summary: dict = {}
        blocks = build_blocks(pool, summary)
        allb = pl.concat([b.drop("_ym") for b in blocks.values()], how="diagonal_relaxed").drop("adsb_airport")
        n_nofile = 0
        if tag == "allyear":
            # explicit rows for 2025 departures on days with no ADS-B file: all features NaN,
            # adsb_coverage = -1 (a flagged "no file" state instead of an absent row). Label-free read.
            ids = pl.concat([
                pl.scan_parquet(training_file(2025, m)).filter(
                    (pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(list(CHALLENGE_AIRPORTS)))
                .select("MVT_ID_mvt").collect() for m in range(1, 13)])
            missing = ids.join(allb.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti")
            filler = missing.with_columns(
                *[pl.lit(None, allb.schema[c]).alias(c) for c in allb.columns if c not in ("MVT_ID_mvt", "adsb_coverage")],
                pl.lit(-1).cast(allb.schema["adsb_coverage"]).alias("adsb_coverage"),
            ).select(allb.columns)
            n_nofile = filler.height
            allb = pl.concat([allb, filler], how="vertical")
        if allb["MVT_ID_mvt"].n_unique() != allb.height:
            raise SystemExit("duplicate MVT_ID_mvt in prize block")
        path = pdir / f"adsb_block_prize_{tag}.parquet"
        allb.write_parquet(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        days = sorted(d for sc in ("D", "janjul", "ranking", "v4") for d in summary.get(f"days_{sc}", []))
        manifest = {
            "source_mode": "adsb_lol", "licence": "ODbL-1.0",
            "licence_url": "https://opendatacommons.org/licenses/odbl/1-0/", "attribution": ATTRIBUTION,
            "decision": "D-017 (owner cleared adsb.lol ADS-B for the prize path)",
            "file": str(path.resolve()), "sha256": sha, "rows": allb.height, "columns": allb.columns,
            "tag": tag, "centroid_and_bias_pool_months_2025": list(pool),
            "isolation": "month-isolated: rows of 2025 month m use only pool labels from months != m; "
                         "ranking and Nov/Dec 2025 rows use the whole pool; no ranking row is ever learned from; "
                         "Nov/Dec 2025 labels are never read (V4 locked)",
            "days_covered": days, "n_days": len(days), "code_commit": commit,
            "days_per_month": dict(sorted(Counter(d[:7] for d in days).items())),
            "rows_no_file_coverage_minus1": n_nofile,
            "coverage_codes": {"-1": "training day with no ADS-B file (allyear file only)", "0": "no aircraft join",
                               "1": "aircraft joined, no departure ground block", "2": "ground block, no stand gate",
                               "3": "stand gate"},
            "input_manifest": "data/external/adsb_lol/slim/adsb_lol_<day>_slim_manifest.json",
        }
        (pdir / f"adsb_block_prize_{tag}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"prize {tag}: {path.resolve()} rows={allb.height} days={len(days)} sha256={sha}", flush=True)


def _pct(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def _s(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0f}"


EST = [("move_rel", "a first motion"), ("still_rel", "b last still"), ("gate_c100_rel", "c100"),
       ("gate_c200_rel", "c200"), ("gate_osm100_rel", "osm100"), ("gate_osm200_rel", "osm200"),
       ("gate_best_rel", "best"), ("neg_taxi_hat", "taxi_hat"), ("gate_conf_rel", "gate_confirmed"),
       ("gate_unconf_rel", "gate_unconfirmed"), ("neg_taxi_hat_conf", "taxi_hat_conf"),
       ("neg_taxi_hat_first", "taxi_hat_first")]


def coverage_table(blk: pl.DataFrame, dep: pl.DataFrame) -> pl.DataFrame:
    """Coverage on NM-matched departures (the screened population); ``join_all`` is over all rows."""
    j = dep.select("MVT_ID_mvt", pl.col("ADEP_mvt").alias("airport"), "matched").join(
        _with_neg(blk), on="MVT_ID_mvt", how="left")
    allj = j.group_by("airport").agg((pl.col("adsb_join_src").fill_null(0) > 0).mean().alias("join_all"))
    j = j.filter(pl.col("matched"))
    aggs = [pl.len().alias("dep"),
            (pl.col("adsb_join_src") == 1).mean().alias("join_callsign"),
            (pl.col("adsb_join_src") > 0).mean().alias("join_any"),
            (pl.col("adsb_n_seg").fill_null(0) > 0).mean().alias("ground_block"),
            (pl.col("adsb_takeoff_rel").abs() <= 300).mean().alias("takeoff_within_5min")]
    aggs += [pl.col(f"adsb_{c}").is_not_null().mean().alias(f"cov_{lab.split()[0]}") for c, lab in EST]
    return j.group_by("airport").agg(aggs).join(allj, on="airport", how="left").sort("airport")


def _with_neg(blk: pl.DataFrame) -> pl.DataFrame:
    out = blk
    if "adsb_taxi_hat" in blk.columns:
        out = out.with_columns((-pl.col("adsb_taxi_hat")).alias("adsb_neg_taxi_hat"))
    else:
        out = out.with_columns(pl.lit(None, pl.Float64).alias("adsb_neg_taxi_hat"))
    if "adsb_gate_conf" in blk.columns:
        conf = pl.col("adsb_gate_conf") == 1
        out = out.with_columns(
            pl.when(conf).then(pl.col("adsb_gate_best_rel")).alias("adsb_gate_conf_rel"),
            pl.when(conf).then(-pl.col("adsb_taxi_hat")).alias("adsb_neg_taxi_hat_conf"),
            pl.when(~conf).then(pl.col("adsb_gate_best_rel")).alias("adsb_gate_unconf_rel"),
            (-pl.col("adsb_taxi_hat_first")).alias("adsb_neg_taxi_hat_first"),
        )
    else:
        out = out.with_columns(*[pl.lit(None, pl.Float64).alias(c) for c in
                                 ("adsb_gate_conf_rel", "adsb_neg_taxi_hat_conf", "adsb_gate_unconf_rel",
                                  "adsb_neg_taxi_hat_first")])
    return out


def accuracy_table(blk: pl.DataFrame, lab: pl.DataFrame) -> list[dict]:
    """Forensic: error of each estimator vs labelled BLOCK (both relative to MVT).

    For ``taxi_hat`` (bias already removed with other-month medians) the reported
    bias is the residual out-of-month bias and RMSE-bc its spread."""
    j = lab.join(_with_neg(blk), on="MVT_ID_mvt", how="left")
    out = []
    for ap in sorted(CHALLENGE_AIRPORTS):
        g = j.filter(pl.col("airport") == ap)
        for c, name in EST:
            e = (g[f"adsb_{c}"] - g["block_rel"]).drop_nulls().drop_nans().to_numpy()
            rec = {"airport": ap, "est": name, "n": len(e), "cov": len(e) / max(g.height, 1)}
            if len(e) >= 10:
                med = float(np.median(e))
                rec.update({"bias": float(e.mean()), "median": med, "rmse_bc": float(e.std()),
                            "w120_med": float(np.mean(np.abs(e - med) <= 120)),
                            "p10": float(np.percentile(e, 10)), "p90": float(np.percentile(e, 90))})
            out.append(rec)
    return out


def labels_for(scope_months: tuple[int, ...]) -> pl.DataFrame:
    parts = []
    for m in scope_months:
        parts.append(
            pl.scan_parquet(training_file(2025, m))
            .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(list(CHALLENGE_AIRPORTS)))
            .select("MVT_ID_mvt", pl.col("ADEP_mvt").alias("airport"),
                    ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds()
                     .cast(pl.Float64)).alias("block_rel"),
                    pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),
                    pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
                    pl.col("FLIGHT_ID_mvt").is_not_null().alias("matched"))
            .collect()
        )
    return pl.concat(parts)


def report(args) -> None:
    fout, rout = OUT / "features", OUT / "report"
    rout.mkdir(parents=True, exist_ok=True)
    res, md = {}, []
    # 2026 ranking coverage (no labels)
    p = fout / "adsb_block_ranking.parquet"
    if p.exists():
        blk = pl.read_parquet(p)
        days = json.loads((fout / "covered_days_ranking.json").read_text())
        dep = (pl.scan_parquet(ROOT / "data" / "raw" / "ranking.parquet")
               .filter((pl.col("PHASE_mvt") == "DEP")
                       & pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).is_in(days))
               .select("MVT_ID_mvt", "ADEP_mvt", pl.col("FLIGHT_ID_mvt").is_not_null().alias("matched"))
               .collect())
        t = coverage_table(blk, dep)
        res["ranking_coverage"] = {"days": len(days), "rows": t.to_dicts()}
        md.append(f"### 2026 ranking coverage ({len(days)} days, {dep.height:,} departures; columns after "
                  "join_all are over NM-matched departures)\n")
        md.append(_cov_md(t))
    # Jan/Jul 2025 forensic accuracy
    p = fout / "adsb_block_janjul.parquet"
    if p.exists():
        blk = pl.read_parquet(p)
        days = json.loads((fout / "covered_days_janjul.json").read_text())
        lab_all = labels_for((1, 7)).filter(pl.col("day").cast(pl.String).is_in(days))
        lab = lab_all.filter(pl.col("matched") & pl.col("block_rel").is_not_null())
        t = coverage_table(blk, lab_all.rename({"airport": "ADEP_mvt"}))
        acc = accuracy_table(blk, lab)
        res["janjul_coverage"] = {"days": len(days), "rows": t.to_dicts()}
        res["janjul_accuracy"] = acc
        md.append(f"\n### Jan/Jul 2025 sensor accuracy, forensic only ({len(days)} days, {lab.height:,} matched "
                  "departures; centroids from D-month fixes only; nothing fitted on these labels)\n")
        md.append(_cov_md(t))
        md.append("\n" + _acc_md(acc))
    # D months: per-airport, per-month bias stability
    p = fout / "adsb_block_D.parquet"
    if p.exists():
        blk = pl.read_parquet(p)
        days = json.loads((fout / "covered_days_D.json").read_text())
        lab_all = labels_for(D_MONTHS).filter(pl.col("day").cast(pl.String).is_in(days))
        lab = lab_all.filter(pl.col("matched") & pl.col("block_rel").is_not_null())
        j = lab.join(_with_neg(blk), on="MVT_ID_mvt", how="left")
        stab = []
        for c, name in EST[2:]:
            e = j.with_columns((pl.col(f"adsb_{c}") - pl.col("block_rel")).alias("e")).filter(
                pl.col("e").is_not_null() & pl.col("e").is_not_nan())
            g = e.group_by("airport", "month").agg(
                pl.len().alias("n"), pl.col("e").mean().alias("bias"), pl.col("e").median().alias("median"),
                pl.col("e").std().alias("sd")).sort("airport", "month")
            for r in g.to_dicts():
                stab.append({"est": name, **r})
        res["D_bias_stability"] = stab
        acc = accuracy_table(blk, lab)
        res["D_accuracy"] = acc
        md.append(f"\n### D months: coverage and accuracy ({len(days)} days, month-isolated centroids)\n")
        md.append(_cov_md(coverage_table(blk, lab_all.rename({"airport": "ADEP_mvt"}))))
        md.append("\n" + _acc_md(acc))
        md.append("\n" + _stab_md(stab))
    (rout / "report.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (rout / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def _cov_md(t: pl.DataFrame) -> str:
    cols = ["dep", "join_all", "join_callsign", "join_any", "ground_block", "takeoff_within_5min", "cov_a", "cov_b",
            "cov_c100", "cov_c200", "cov_osm100", "cov_osm200", "cov_best"]
    head = "| apt | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1)
    lines = [head]
    for r in t.iter_rows(named=True):
        lines.append(f"| {r['airport']} | {r['dep']} | " + " | ".join(_pct(r[c]) for c in cols[1:]) + " |")
    return "\n".join(lines)


def _acc_md(acc: list[dict]) -> str:
    lines = ["| apt | est | n | cov | bias | median | RMSE bias-corr | ±120 s (median-corr) | p10 | p90 |",
             "|" + "---|" * 10]
    for r in acc:
        if r["n"] < 10:
            continue
        lines.append(f"| {r['airport']} | {r['est']} | {r['n']} | {_pct(r['cov'])} | {_s(r.get('bias'))} | "
                     f"{_s(r.get('median'))} | {_s(r.get('rmse_bc'))} | {_pct(r.get('w120_med'))} | "
                     f"{_s(r.get('p10'))} | {_s(r.get('p90'))} |")
    return "\n".join(lines)


def _stab_md(stab: list[dict]) -> str:
    t = pl.DataFrame(stab)
    if t.height == 0:
        return ""
    lines = ["Per-airport median error (gate − BLOCK, s) by month; n in brackets", ""]
    months = sorted(t["month"].unique().to_list())
    lines.append("| apt | est | " + " | ".join(str(m) for m in months) + " | spread (max−min median) |")
    lines.append("|" + "---|" * (len(months) + 3))
    for (ap, est), g in t.filter(pl.col("n") >= 20).group_by(["airport", "est"], maintain_order=True):
        d = {r["month"]: r for r in g.to_dicts()}
        meds = [d[m]["median"] for m in months if m in d]
        cells = [f"{d[m]['median']:.0f} ({d[m]['n']})" if m in d else "" for m in months]
        lines.append(f"| {ap} | {est} | " + " | ".join(cells) + f" | {max(meds) - min(meds):.0f} |")
    return "\n".join(lines)


def screen_rows(args) -> None:
    """MVT_ID_mvt of all D-month departures on stage1-covered days (same file for both arms)."""
    days = sorted(p.name for p in (OUT / "stage1").iterdir()
                  if p.is_dir() and scope_of(date.fromisoformat(p.name)) == "D")
    if args.months:
        keep_m = {int(m) for m in args.months.split(",")}
        days = [d for d in days if int(d[5:7]) in keep_m]
    ids = []
    for m in D_MONTHS:
        ids.append(
            pl.scan_parquet(training_file(2025, m))
            .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).is_in(days))
            .select("MVT_ID_mvt").collect())
    rows = pl.concat(ids)
    p = OUT / "features" / "screen_rows_D.parquet"
    rows.write_parquet(p)
    months = sorted({int(d[5:7]) for d in days})
    (OUT / "features" / "screen_rows_D.json").write_text(json.dumps({"days": days, "months": months,
                                                                    "rows": rows.height}, indent=1) + "\n")
    print(f"screen rows: {rows.height} departures on {len(days)} D days, months {months}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["stage1", "features", "report", "all", "screenrows", "prize"])
    ap.add_argument("--scope", default="", help="comma list of D,janjul,ranking (stage1)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--months", default="", help="screenrows: restrict to these months (comma list)")
    args = ap.parse_args()
    if args.command in ("stage1", "all"):
        stage1(args)
    if args.command in ("features", "all"):
        features(args)
    if args.command in ("report", "all"):
        report(args)
    if args.command == "screenrows":
        screen_rows(args)
    if args.command == "prize":
        prize(args)


if __name__ == "__main__":
    main()
