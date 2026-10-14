# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Build the SA-M4 label-free feature blocks (W window, P planned taxi, C record signals).

Examples::

    # months 1-10 (screen input; P medians leave-fold-out inside D)
    python scripts/features/build_m4_blocks.py --block w --variant v1 --source training --months 1-10
    python scripts/features/build_m4_blocks.py --block p --variant v1 --source training --months 1-10
    python scripts/features/build_m4_blocks.py --block c --variant v1 --source training --months 1-10
    # months 11-12 (P medians from all D months)
    python scripts/features/build_m4_blocks.py --block p --source training --months 11-12
    # ranking (P medians from all twelve 2025 training months)
    python scripts/features/build_m4_blocks.py --block p --source ranking
    # one S4-ready file (months 1-10 + 11-12 + ranking) for train_s4_allyear.py --extra-features
    python scripts/features/build_m4_blocks.py --block p --source combine

Raw files are read through ``prc_challenge.window_blocks.read_masked`` (column
whitelist): DEP BLOCK/TAXITIME, LOBT, AOBT_3 and ARVT_3 are never loaded.
Output: ``artifacts/experiments/Matched-features-20260925/m4_<block>_<variant>/<tag>.parquet``
plus ``<tag>.json`` manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import polars as pl

from prc_challenge.window_blocks import (
    D_MONTHS,
    build_plan_block,
    build_plan_block_crossfit,
    build_record_block,
    build_window_block,
    load_ranking,
    load_training_month,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "artifacts" / "experiments" / "Matched-features-20260925"
SOURCE_MODE = {
    "w": "own row only: MVT_TIME, IOBT_flt, EOBT_1_flt, FLIGHT_ID (hull window +/-3606 s); no fitted parameter",
    "p": "own row ARVT_1/EOBT_1/MVT plus route medians from reference months of other flights' "
    "takeoff, ARR landing (MVT_TIME of ARR rows) and filed plan times; reference = leave-fold-out D "
    "(D months), all D months (months 1,7,11,12), all 12 training months (ranking); no DEP labels",
    "c": "own row only: MVT_TIME seconds, FLIGHT_mvt, CALLSIGN_flt, ADES_*, AIRCRAFT_TYPE_*",
}
CATS = {"w": [], "p": [], "c": ["c_service_ap", "c_carrier_ap", "c_slot_h_ap"]}


def _months(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x]


def _commit() -> str:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "src/prc_challenge/window_blocks.py", "scripts/features/build_m4_blocks.py"],
            cwd=ROOT,
            text=True,
        ).strip()
        return head + ("+dirty" if dirty else "")
    except Exception:  # pragma: no cover
        return "unknown"


def _sched_seconds_constant(frame: pl.DataFrame) -> bool:
    s = frame.filter(pl.col("SCHED_TIME_UTC_mvt").is_not_null())["SCHED_TIME_UTC_mvt"].dt.second()
    return bool(s.max() == 0) if s.len() else True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", choices=("w", "p", "c"), required=True)
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--source", choices=("training", "ranking", "combine"), default="training")
    ap.add_argument("--months", default="1-10")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    t0 = time.time()
    sched_const = []
    if args.source == "combine":
        folder = args.out_root / f"m4_{args.block}_{args.variant}"
        tags = ("months_1_10", "months_11_12", "ranking")
        missing = [t for t in tags if not (folder / f"{t}.parquet").exists()]
        if missing:
            raise SystemExit(f"build {missing} first")
        out = pl.concat([pl.read_parquet(folder / f"{t}.parquet") for t in tags], how="vertical")
        if out["MVT_ID_mvt"].n_unique() != out.height:
            raise RuntimeError("duplicate MVT_ID_mvt across months_1_10 / months_11_12 / ranking")
        path = folder / "s4_all.parquet"
        out.write_parquet(path)
        parts = {t: json.loads((folder / f"{t}.json").read_text()) for t in tags}
        manifest = {
            "block": f"m4_{args.block}",
            "variant": args.variant,
            "path": str(path.resolve()),
            "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            "rows": out.height,
            "columns": [c for c in out.columns if c != "MVT_ID_mvt"],
            "categoricals": parts["months_1_10"]["categoricals"],
            "parts": {t: {"sha256_16": parts[t]["sha256_16"], "rows": parts[t]["rows"], "p_reference": parts[t]["p_reference"]} for t in tags},
            "builder": f"scripts/features/build_m4_blocks.py --block {args.block} --variant {args.variant} --source combine",
            "builder_commit": _commit(),
        }
        (folder / "s4_all.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({k: manifest[k] for k in ("block", "variant", "path", "rows", "sha256_16")}, indent=2))
        return
    if args.source == "training":
        months = _months(args.months)
        tag = f"months_{months[0]}_{months[-1]}" if len(months) > 1 else f"month_{months[0]}"
        frames = {m: load_training_month(m, args.raw_dir) for m in months}
        sched_const = [_sched_seconds_constant(f) for f in frames.values()]
        if args.block == "p":
            if all(m in frames for m in D_MONTHS):
                out = build_plan_block_crossfit(frames, args.variant)
                ref_desc = "leave-fold-out D for D months; all D months otherwise"
            else:
                ref = pl.concat([load_training_month(m, args.raw_dir) for m in D_MONTHS], how="vertical_relaxed")
                out = pl.concat([build_plan_block(frames[m], ref, args.variant) for m in months])
                ref_desc = "all D months"
        else:
            fn = build_window_block if args.block == "w" else build_record_block
            out = pl.concat([fn(frames[m], args.variant) for m in months])
            ref_desc = None
    else:
        tag = "ranking"
        target = load_ranking(args.raw_dir)
        months = sorted(target["src_month"].unique().to_list())  # serve months of the ranking file
        sched_const = [_sched_seconds_constant(target)]
        if args.block == "p":
            ref = pl.concat([load_training_month(m, args.raw_dir) for m in range(1, 13)], how="vertical_relaxed")
            out = build_plan_block(target, ref, args.variant)
            ref_desc = "all 12 training months (2025)"
        else:
            fn = build_window_block if args.block == "w" else build_record_block
            out = fn(target, args.variant)
            ref_desc = None
    if out["MVT_ID_mvt"].n_unique() != out.height:
        raise RuntimeError("duplicate MVT_ID_mvt")
    folder = args.out_root / f"m4_{args.block}_{args.variant}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{tag}.parquet"
    out.write_parquet(path)
    cols = [c for c in out.columns if c != "MVT_ID_mvt"]
    nulls = out.select([pl.col(c).is_null().mean().alias(c) for c in cols]).row(0, named=True)
    manifest = {
        "block": f"m4_{args.block}",
        "variant": args.variant,
        "path": str(path.resolve()),
        "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "rows": out.height,
        "columns": cols,
        "categoricals": [c for c in CATS[args.block] if c in cols],
        "null_fraction": nulls,
        "source": args.source,
        "months": months,
        "p_reference": ref_desc,
        "sched_seconds_all_zero": all(sched_const),
        "source_mode": SOURCE_MODE[args.block],
        "forbidden_never_read": ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "LOBT_flt", "AOBT_3_flt", "ARVT_3_flt"],
        "builder": f"scripts/features/build_m4_blocks.py --block {args.block} --variant {args.variant} --source {args.source}"
        + (f" --months {args.months}" if args.source == "training" else ""),
        "builder_commit": _commit(),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (folder / f"{tag}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("block", "variant", "path", "rows", "sha256_16", "runtime_sec", "sched_seconds_all_zero")}, indent=2))


if __name__ == "__main__":
    main()
