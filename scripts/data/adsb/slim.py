# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Slim filtered adsb.lol traces for transfer (runs on the ingest host).

Keeps airport, hex, ts, lat, lon, on_ground, gs_kt, callsign; all ground samples;
airborne samples thinned to one per 10 s per aircraft; zstd level 9.
Writes out/../slim/adsb_lol_<day>_slim.parquet next to a copy of the day manifest.
"""
import json, shutil, sys, time
from pathlib import Path
import polars as pl

OUT = Path.home() / "adsb" / "out"
SLIM = Path.home() / "adsb" / "slim"
COLS = ["airport", "hex", "ts", "lat", "lon", "on_ground", "gs_kt", "callsign"]


def slim_one(traces: Path, manifest: Path) -> None:
    day = traces.name.split("_")[2]
    dest = SLIM / f"adsb_lol_{day}_slim.parquet"
    if dest.exists():
        return
    df = pl.read_parquet(traces, columns=COLS).sort("airport", "hex", "ts")
    ground = df.filter(pl.col("on_ground"))
    air = (df.filter(~pl.col("on_ground"))
             .with_columns((pl.col("ts") // 10).alias("_b"))
             .unique(["airport", "hex", "_b"], keep="first", maintain_order=True)
             .drop("_b"))
    out = pl.concat([ground, air]).sort("airport", "hex", "ts")
    tmp = dest.with_suffix(".tmp")
    out.write_parquet(tmp, compression="zstd", compression_level=9)
    tmp.rename(dest)
    meta = json.loads(manifest.read_text())
    meta["slim"] = {"file": dest.name, "rows": out.height, "rows_in": df.height,
                    "columns": COLS, "airborne_thin_s": 10, "bytes": dest.stat().st_size}
    (SLIM / f"adsb_lol_{day}_slim_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    SLIM.mkdir(exist_ok=True)
    loop = "--loop" in sys.argv
    while True:
        for man in sorted(OUT.glob("adsb_lol_*_full_manifest.json")):
            stem = man.name.replace("_full_manifest.json", "")
            tr = OUT / f"{stem}_traces.parquet"
            if tr.exists():
                slim_one(tr, man)
        if not loop:
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
