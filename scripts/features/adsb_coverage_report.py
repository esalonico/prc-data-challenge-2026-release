# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Forensic accuracy, 2026 coverage gain and drift for the v3 ADS-B coverage block.

Called by ``scripts/features/build_adsb_coverage.py report``. Labels: only D-month 2025 BLOCK labels are read,
for forensic evaluation of month-isolated estimates. Jan/Jul and Nov/Dec 2025 labels are not read here.
2026 numbers are label-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
RELIABLE_SD = 180.0


def _labels(months: tuple[int, ...]) -> pl.DataFrame:
    parts = []
    for m in months:
        assert m in (2, 3, 4, 5, 6, 8, 9, 10), "Nov/Dec 2025 labels are locked"
        f = next((ROOT / "data" / "raw").glob(f"training_2025-{m:02d}-01_*.parquet"))
        parts.append(pl.scan_parquet(f).filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_not_null())
                     .select("MVT_ID_mvt", pl.col("ADEP_mvt").alias("airport"),
                             ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds()
                              .cast(pl.Float64)).alias("block_rel"),
                             pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
                     .filter(pl.col("block_rel").is_not_null()).collect())
    return pl.concat(parts)


EST = [("v2_first", "v2 first-seen (taxi_hat_first)"), ("v2_gate_conf", "v2 confirmed gate (taxi_hat)"),
       ("adsb3_map_first_rel", "v3 map first reliable sample"), ("adsb3_map_wmean_rel", "v3 map weighted mean"),
       ("adsb3_maph_first_rel", "v3 heading-free map (fallback)"), ("adsb3_rwy_zone_rel", "v3 runway-zone entry")]


def _stats(e: np.ndarray, n_all: int) -> dict:
    if len(e) < 20:
        return {"n": int(len(e)), "cov": len(e) / max(n_all, 1)}
    med = float(np.median(e))
    r = e - med
    rt = r[np.abs(r) <= 1200]
    return {"n": int(len(e)), "cov": len(e) / max(n_all, 1), "median": med,
            "rmse_bc": float(np.sqrt(np.mean(r ** 2))), "rmse_bc_trim1200": float(np.sqrt(np.mean(rt ** 2))),
            "w120": float(np.mean(np.abs(r) <= 120)), "w300": float(np.mean(np.abs(r) <= 300))}


def accuracy(j: pl.DataFrame) -> list[dict]:
    out = []
    j = j.with_columns((-pl.col("adsb_taxi_hat_first")).alias("v2_first"),
                       pl.when(pl.col("adsb_gate_conf") == 1).then(-pl.col("adsb_taxi_hat")).alias("v2_gate_conf"))
    for ap in AIRPORTS:
        for cls, flt in (("ground_only", pl.col("adsb_coverage") == 2), ("stand_gate", pl.col("adsb_coverage") == 3)):
            g = j.filter((pl.col("airport") == ap) & flt)
            if g.height < 50:
                continue
            for c, name in EST:
                e = (g[c] - g["block_rel"]).drop_nulls().drop_nans().to_numpy()
                out.append({"airport": ap, "class": cls, "rows": g.height, "est": name, **_stats(e, g.height)})
    return out


def _acc_md(acc: list[dict]) -> str:
    lines = ["| apt | class | rows | estimator | cov | median | RMSE-bc | RMSE-bc trim ±1200 | ±120 s | ±300 s |",
             "|" + "---|" * 10]
    for r in acc:
        if "median" not in r:
            continue
        lines.append(f"| {r['airport']} | {r['class']} | {r['rows']} | {r['est']} | {100 * r['cov']:.0f}% | "
                     f"{r['median']:.0f} | {r['rmse_bc']:.0f} | {r['rmse_bc_trim1200']:.0f} | "
                     f"{100 * r['w120']:.0f}% | {100 * r['w300']:.0f}% |")
    return "\n".join(lines)


def coverage_2026(v2: pl.DataFrame, v3: pl.DataFrame) -> list[dict]:
    dep = (pl.scan_parquet(ROOT / "data" / "raw" / "ranking.parquet")
           .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_not_null())
           .select("MVT_ID_mvt", pl.col("ADEP_mvt").alias("airport"),
                   pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month")).collect())
    j = dep.join(v2, on="MVT_ID_mvt", how="left").join(v3, on="MVT_ID_mvt", how="left")
    conf = pl.col("adsb_gate_conf").fill_null(0) == 1
    rel = pl.col("adsb3_map_first_sd").fill_null(1e9) <= RELIABLE_SD
    t = (j.group_by("airport", "month").agg(
        pl.len().alias("n"),
        (pl.col("adsb_n_seg").fill_null(0) > 0).mean().alias("ground"),
        (pl.col("adsb_gate_best_rel").is_not_null()).mean().alias("any_gate"),
        conf.mean().alias("conf_gate"),
        pl.col("adsb3_map_first_rel").is_not_null().mean().alias("map_any"),
        rel.mean().alias("map_reliable"),
        (conf | rel).mean().alias("conf_or_map_reliable"),
        ((pl.col("adsb_coverage") == 2) & rel).sum().alias("ground_only_rows_gaining"),
    ).sort("airport", "month"))
    return t.to_dicts()


def drift(v3_d: pl.DataFrame, v3_26: pl.DataFrame, v2_d: pl.DataFrame, v2_26: pl.DataFrame,
          d_ids: pl.DataFrame, r_ids: pl.DataFrame) -> list[dict]:
    """Label-free medians of AOBT - map and gate - map, 2025 D (dfit block) vs 2026 (allyear block)."""
    def med(v3, v2, ids):
        j = ids.join(v3, on="MVT_ID_mvt", how="inner").join(v2, on="MVT_ID_mvt", how="left")
        rel = pl.col("adsb3_map_first_sd") <= RELIABLE_SD
        return j.group_by("airport", "period").agg(
            pl.col("adsb3_aobt_minus_map").filter(rel).median().alias("aobt_minus_map"),
            (pl.col("adsb_gate_best_rel") - pl.col("adsb3_map_first_rel")).filter(rel & (pl.col("adsb_gate_conf") == 1))
            .median().alias("gate_minus_map"),
            pl.col("adsb3_map_first_t").filter(rel).median().alias("map_first_t"),
            rel.sum().alias("n_reliable"))
    a = med(v3_d, v2_d, d_ids)
    b = med(v3_26, v2_26, r_ids)
    return pl.concat([a, b]).sort("airport", "period").to_dicts()


def report(out: Path, v2root: Path) -> None:
    vdir = out / "v3"
    rdir = out / "report"
    rdir.mkdir(parents=True, exist_ok=True)
    v2cols = ["MVT_ID_mvt", "adsb_coverage", "adsb_taxi_hat_first", "adsb_taxi_hat", "adsb_gate_conf",
              "adsb_gate_best_rel", "adsb_n_seg"]
    v2d = pl.read_parquet(v2root / "prize" / "adsb_block_prize_dfit.parquet", columns=v2cols)
    v3d = pl.read_parquet(vdir / "adsb_v3_block_dfit.parquet")
    res, md = {}, []
    lab_d = _labels((2, 3, 4, 5, 6, 8, 9, 10))
    jd = lab_d.join(v2d, on="MVT_ID_mvt", how="inner").join(v3d, on="MVT_ID_mvt", how="left")
    res["d_accuracy"] = accuracy(jd)
    md.append(f"### D-month 2025 forensic accuracy (month-isolated maps and biases; {jd.height:,} matched rows "
              "on ADS-B days; estimates relative to BLOCK)\n")
    md.append(_acc_md(res["d_accuracy"]))
    va = vdir / "adsb_v3_block_allyear.parquet"
    if va.exists():
        v2a = pl.read_parquet(v2root / "prize" / "adsb_block_prize_allyear.parquet", columns=v2cols)
        v3a = pl.read_parquet(va)
        cov = coverage_2026(v2a, v3a)
        res["coverage_2026"] = cov
        md.append("\n### 2026 ranking, NM-matched departures: reliable off-block estimate coverage (label-free)\n")
        md.append("| apt | month | n | ground | any gate | confirmed gate | map any | map reliable (sd ≤ 180 s) | "
                  "confirmed gate or reliable map | ground-only rows gaining |")
        md.append("|" + "---|" * 10)
        for r in cov:
            md.append(f"| {r['airport']} | {r['month']} | {r['n']} | {100 * r['ground']:.0f}% | "
                      f"{100 * r['any_gate']:.0f}% | {100 * r['conf_gate']:.0f}% | {100 * r['map_any']:.0f}% | "
                      f"{100 * r['map_reliable']:.0f}% | {100 * r['conf_or_map_reliable']:.0f}% | "
                      f"{r['ground_only_rows_gaining']} |")
        d_ids = (pl.concat([pl.scan_parquet(next((ROOT / "data" / "raw").glob(f"training_2025-{m:02d}-01_*.parquet")))
                            .filter(pl.col("PHASE_mvt") == "DEP")
                            .select("MVT_ID_mvt", pl.col("ADEP_mvt").alias("airport")).collect()
                            for m in (2, 3, 4, 5, 6, 8, 9, 10)])
                 .with_columns(pl.lit("2025-D").alias("period")))
        r_ids = (pl.scan_parquet(ROOT / "data" / "raw" / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
                 .select("MVT_ID_mvt", pl.col("ADEP_mvt").alias("airport"),
                         pl.format("2026-{}", pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("period")).collect())
        v2a_full = pl.read_parquet(v2root / "prize" / "adsb_block_prize_allyear.parquet", columns=v2cols)
        dr = drift(v3d, v3a, v2d, v2a_full, d_ids, r_ids)
        res["drift"] = dr
        md.append("\n### Label-free drift: medians on rows with a reliable map estimate (s)\n")
        md.append("| apt | period | n reliable | AOBT − map | confirmed gate − map | first reliable sample (rel. MVT) |")
        md.append("|---|---|---|---|---|---|")
        f = lambda x: "" if x is None else f"{x:.0f}"
        for r in dr:
            md.append(f"| {r['airport']} | {r['period']} | {r['n_reliable']} | {f(r['aobt_minus_map'])} | "
                      f"{f(r['gate_minus_map'])} | {f(r['map_first_t'])} |")
    (rdir / "report.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (rdir / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
