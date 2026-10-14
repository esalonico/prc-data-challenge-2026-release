# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Splice a feature block for the four-month serve set.

Keeps every 2025 row of the shipped combined block byte-for-byte (rows whose MVT_ID_mvt is not a DEP of the
old public Jan/Jul ranking file), drops the old Jan/Jul 2026 rows and appends the freshly built rows of the
final ranking file (Jan, Feb, Jun, Jul 2026). Column order and dtypes follow the shipped block.

Usage:
  python splice_block.py --shipped <combined.parquet> --ranking <new ranking-rows.parquet> --out <out.parquet>
         [--old-raw data/raw/ranking_public.parquet] [--new-raw data/raw/ranking.parquet] [--allow-missing]
"""
import argparse
import hashlib
import json
from pathlib import Path

import polars as pl


def dep_ids(path: str) -> pl.Series:
    return (pl.scan_parquet(path).filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt").collect()["MVT_ID_mvt"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-raw", default="data/raw/ranking_public.parquet")
    ap.add_argument("--new-raw", default="data/raw/ranking.parquet")
    ap.add_argument("--allow-missing", action="store_true", help="serve rows absent from the new block are allowed")
    a = ap.parse_args()
    shipped = pl.read_parquet(a.shipped)
    new = pl.read_parquet(a.ranking)
    old_ids, new_ids = dep_ids(a.old_raw).implode(), dep_ids(a.new_raw)
    keep = shipped.filter(~pl.col("MVT_ID_mvt").is_in(old_ids))
    clash = keep.filter(pl.col("MVT_ID_mvt").is_in(new_ids.implode())).height
    if clash:
        raise SystemExit(f"{clash} kept 2025 rows share an MVT_ID_mvt with the serve set")
    if set(new.columns) != set(shipped.columns):
        raise SystemExit(f"column sets differ: only shipped {sorted(set(shipped.columns) - set(new.columns))}, "
                         f"only new {sorted(set(new.columns) - set(shipped.columns))}")
    new = new.filter(pl.col("MVT_ID_mvt").is_in(new_ids.implode())).select(
        [pl.col(c).cast(t) for c, t in shipped.schema.items()])
    n_missing = new_ids.len() - new.height
    if n_missing and not a.allow_missing:
        raise SystemExit(f"{n_missing} serve rows missing from the new block")
    out = pl.concat([keep, new], how="vertical")
    if out["MVT_ID_mvt"].n_unique() != out.height:
        raise SystemExit("duplicate MVT_ID_mvt after splice")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    rep = {"out": a.out, "rows": out.height, "kept_2025": keep.height, "dropped_old_ranking": shipped.height - keep.height,
           "serve_rows": new.height, "serve_missing": n_missing,
           "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}
    Path(a.out).with_suffix(".splice.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
