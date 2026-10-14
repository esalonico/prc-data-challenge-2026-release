# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""FLIGHT_mvt -> CALLSIGN_flt dictionary for NM-unmatched departures (prereg amendment 2).

At EHAM (and partly LSZH) ``FLIGHT_mvt`` is the IATA flight number while ADS-B carries the
ICAO callsign, so unmatched rows there cannot be joined by callsign. This builds, per airport,
a label-free dictionary from NM-matched departures (2025 months 1-10 + ranking matched rows):
keep ``FLIGHT_mvt -> CALLSIGN_flt`` when the flight number maps to one callsign, is seen
>= MIN_SEEN times, and record the local-hour range of its departures (time-of-day check at use).

Precision (month-isolated hide-and-rederive): for each 2025 month m, the dictionary built
from the other months (+ ranking) is applied to month-m matched rows whose FLIGHT_mvt differs
from CALLSIGN_flt; precision = share of derived callsigns equal to the true CALLSIGN_flt.
Airports below MIN_PRECISION are excluded. Output: features/callsign_dictionary.parquet + .json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "experiments" / "ADSB-features-20260925" / "features"
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
MIN_SEEN = 3
HOUR_SLACK = 2
MIN_PRECISION = 0.97
COLS = ["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt"]


def load_rows() -> pl.DataFrame:
    files = [next((ROOT / "data" / "raw").glob(f"training_2025-{m:02d}-01_*.parquet")) for m in range(1, 11)]
    files.append(ROOT / "data" / "raw" / "ranking.parquet")
    parts = [
        pl.scan_parquet(f).filter((pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(list(AIRPORTS)))
        .select(COLS).collect() for f in files
    ]
    df = pl.concat(parts).with_columns(
        pl.col("FLIGHT_mvt").str.strip_chars().str.to_uppercase().str.replace_all(" ", "").alias("fl"),
        pl.col("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("cs"),
        pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour"),
        pl.col("MVT_TIME_UTC_mvt").dt.year().alias("year"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
    )
    return df


def build(matched: pl.DataFrame) -> pl.DataFrame:
    g = (
        matched.filter(pl.col("fl").is_not_null() & pl.col("cs").is_not_null())
        .group_by("ADEP_mvt", "fl", "cs")
        .agg(pl.len().alias("n"), pl.col("hour").min().alias("hour_lo"), pl.col("hour").max().alias("hour_hi"))
    )
    uniq = g.group_by("ADEP_mvt", "fl").agg(pl.len().alias("n_cs"), pl.col("n").sum().alias("n_total"))
    return (
        g.join(uniq, on=["ADEP_mvt", "fl"]).filter((pl.col("n_cs") == 1) & (pl.col("n") >= MIN_SEEN))
        .select("ADEP_mvt", "fl", "cs", "n", "hour_lo", "hour_hi")
    )


def apply(rows: pl.DataFrame, dic: pl.DataFrame) -> pl.DataFrame:
    """Add ``cs_dict`` where the flight number is in the dictionary and the hour fits."""
    j = rows.join(dic.rename({"cs": "cs_dict"}), on=["ADEP_mvt", "fl"], how="left")
    ok = (pl.col("hour") >= pl.col("hour_lo") - HOUR_SLACK) & (pl.col("hour") <= pl.col("hour_hi") + HOUR_SLACK)
    return j.with_columns(pl.when(ok).then(pl.col("cs_dict")).otherwise(None).alias("cs_dict")).drop(
        "n", "hour_lo", "hour_hi")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args()
    df = load_rows()
    matched = df.filter(pl.col("FLIGHT_ID_mvt").is_not_null())
    # month-isolated precision on 2025 matched rows whose flight number differs from the callsign
    prec = []
    for m in range(1, 11):
        dic_m = build(matched.filter(~((pl.col("year") == 2025) & (pl.col("month") == m))))
        test = matched.filter((pl.col("year") == 2025) & (pl.col("month") == m) & (pl.col("fl") != pl.col("cs")))
        got = apply(test, dic_m).filter(pl.col("cs_dict").is_not_null())
        prec.append(got.group_by("ADEP_mvt").agg(pl.len().alias("n"), (pl.col("cs_dict") == pl.col("cs")).sum().alias("ok"),
                                                 pl.lit(m).alias("month")))
        prec.append(test.group_by("ADEP_mvt").agg(pl.len().alias("n_test"), pl.lit(m).alias("month")))
    pr = pl.concat([p for p in prec if "ok" in p.columns]).group_by("ADEP_mvt").agg(pl.col("n").sum(), pl.col("ok").sum())
    nt = pl.concat([p for p in prec if "n_test" in p.columns]).group_by("ADEP_mvt").agg(pl.col("n_test").sum())
    pr = pr.join(nt, on="ADEP_mvt", how="full", coalesce=True).with_columns(
        (pl.col("ok") / pl.col("n")).alias("precision"), (pl.col("n") / pl.col("n_test")).alias("recall")
    ).sort("ADEP_mvt")
    keep = {r["ADEP_mvt"] for r in pr.iter_rows(named=True) if r["precision"] is not None and r["precision"] >= MIN_PRECISION
            and r["n"] >= 100}
    dic = build(matched).filter(pl.col("ADEP_mvt").is_in(sorted(keep)))
    OUT.mkdir(parents=True, exist_ok=True)
    dic.write_parquet(OUT / "callsign_dictionary.parquet")
    # ranking unmatched coverage gained
    unm = df.filter(pl.col("FLIGHT_ID_mvt").is_null() & (pl.col("year") == 2026))
    got = apply(unm, dic)
    gain = got.group_by("ADEP_mvt").agg(pl.len().alias("n_unmatched"), pl.col("cs_dict").is_not_null().sum().alias("n_dict")).sort("ADEP_mvt")
    summary = {"min_seen": MIN_SEEN, "hour_slack": HOUR_SLACK, "min_precision": MIN_PRECISION,
               "dictionary_entries": dic.height, "airports_kept": sorted(keep),
               "precision_month_isolated": pr.to_dicts(), "ranking_unmatched_with_dict_callsign": gain.to_dicts()}
    (OUT / "callsign_dictionary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(pr)
    print(gain)
    print("kept", sorted(keep), "entries", dic.height)


if __name__ == "__main__":
    main()
