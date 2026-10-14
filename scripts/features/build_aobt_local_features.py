# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Build the LOCAL_ONLY AOBT/LOBT feature block (source_mode ``aobt_local``).

Writes one parquet per scope under
``artifacts/experiments/AOBT-local-only-20260925/features/``:

- ``aobt_features_months_1_10.parquet``   (training months 1-10; screen input)
- ``aobt_features_months_11_12.parquet``  (``--scope all``)

``--ranking-dryrun`` builds the block for ``ranking.parquet`` in memory and prints
coverage only. Nothing ranking-derived is written until the owner clears AOBT/LOBT.

Promotion (only after a written owner ruling): ``--promote-owner-ruling "<ref>"``
writes ``aobt_features_allyear_plus_ranking.parquet`` (2025 months 1-12 plus both
ranking months, one row per DEP ``MVT_ID_mvt``), the single file the S4 trainer
takes via ``--extra-features``. Its manifest records the ruling reference.

Mode statistics (``aobt_taxi_*``) are computed per (airport, runway, UTC month),
which equals per-raw-file isolation for training (2 boundary rows differ) and
per-month isolation for the ranking file.

Every output has a manifest JSON with ``source_mode: aobt_local``;
``ranking_submission_allowed`` is false unless a ruling reference was passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import polars as pl

from prc_challenge.aobt_local.features import (
    AOBT_FEATURE_COLUMNS,
    SOURCE_MODE,
    build_aobt_features,
    load_dep_timestamps,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "artifacts" / "experiments" / "AOBT-local-only-20260925" / "features"


def _month(path: Path) -> int:
    m = re.search(r"training_2025-(\d{2})-01_", path.name)
    return int(m.group(1)) if m else -1


def _write(frame: pl.DataFrame, name: str, inputs: list[Path], t0: float, ruling: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.parquet"
    frame.write_parquet(path)
    manifest = {
        "source_mode": SOURCE_MODE,
        "local_only": True,
        "ranking_submission_allowed": bool(ruling),
        "owner_ruling": ruling,
        "quarantined_inputs": ["AOBT_3_flt", "LOBT_flt"],
        "never_read": ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "ARVT_3_flt"],
        "key": "MVT_ID_mvt",
        "columns": list(AOBT_FEATURE_COLUMNS),
        "rows": frame.height,
        "coverage_aobt": float(1.0 - frame["aobt_missing"].mean()),
        "inputs": [str(p.relative_to(ROOT)) for p in inputs],
        "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "builder": "scripts/features/build_aobt_local_features.py",
        "runtime_sec": round(time.time() - t0, 1),
    }
    (OUT / f"{name}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("rows", "coverage_aobt", "sha256_16")}), name, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=["train", "all"], default="train")
    ap.add_argument("--ranking-dryrun", action="store_true", help="build ranking block in memory, print coverage, write nothing")
    ap.add_argument("--promote-owner-ruling", default=None, help="written owner ruling reference; writes the all-year + ranking block")
    args = ap.parse_args()
    t0 = time.time()
    files = sorted(RAW.glob("training_2025-*.parquet"))
    first = [p for p in files if 1 <= _month(p) <= 10]
    _write(build_aobt_features(load_dep_timestamps(first)), "aobt_features_months_1_10", first, t0)
    if args.scope == "all":
        late = [p for p in files if _month(p) in (11, 12)]
        _write(build_aobt_features(load_dep_timestamps(late)), "aobt_features_months_11_12", late, t0)
    if args.promote_owner_ruling:
        inputs = [*files, RAW / "ranking.parquet"]
        parts = [build_aobt_features(load_dep_timestamps([p])) for p in inputs]
        block = pl.concat(parts, how="vertical")
        if block["MVT_ID_mvt"].n_unique() != block.height:
            raise SystemExit("duplicate MVT_ID_mvt across training and ranking inputs")
        _write(block, "aobt_features_allyear_plus_ranking", inputs, t0, ruling=args.promote_owner_ruling)
    if args.ranking_dryrun:
        block = build_aobt_features(load_dep_timestamps([RAW / "ranking.parquet"]))
        print(
            "ranking dry run (not written):",
            json.dumps({"rows": block.height, "coverage_aobt": float(1.0 - block["aobt_missing"].mean())}),
        )


if __name__ == "__main__":
    main()
