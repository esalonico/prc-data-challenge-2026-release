# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Build the SA-MF matched-context feature blocks (M1 stand sequence, M2 neighbour context).

Examples::

    # months 1-10 (screen input)
    python scripts/features/build_matched_context_blocks.py --block m1 --source training --months 1-10
    python scripts/features/build_matched_context_blocks.py --block m2 --source training --months 1-10
    # months 11-12 or the ranking file (label-free; DEP labels are never read)
    python scripts/features/build_matched_context_blocks.py --block m1 --source training --months 11-12
    python scripts/features/build_matched_context_blocks.py --block m2 --source ranking

Each training month file is its own context window; the ranking file is split by
calendar month of the movement time (January and July never interact).  Output:
``artifacts/experiments/Matched-features-20260925/<block dir>/<tag>.parquet`` keyed by
``MVT_ID_mvt`` plus ``<tag>.json`` (columns, source mode, builder commit, coverage).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import polars as pl

from prc_challenge import arrival_guard
from prc_challenge.matched_context.movements import load_ranking, load_training_month
from prc_challenge.matched_context.neighbor_context import M2_VERSION, build_neighbor_context_features
from prc_challenge.matched_context.stand_sequence import M1_VERSION, build_stand_sequence_features

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "artifacts" / "experiments" / "Matched-features-20260925"
BLOCKS = {
    "m1": ("m1_stand_v1", M1_VERSION, build_stand_sequence_features),
    "m2": ("m2_follow_v1", M2_VERSION, build_neighbor_context_features),
}
SOURCE_MODE = (
    "bidirectional within one context window (calendar-month file; ranking split by MVT month); "
    "consumes other flights' movement/plan times, stand, runway, type, destination, flight-number prefix "
    "and ARR in-block times of ARR rows only (decision D-005); never DEP BLOCK/TAXITIME, AOBT_3 or ARVT_3"
)


def _months(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x]


def _commit() -> str:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "src/prc_challenge/matched_context"], cwd=ROOT, text=True).strip()
        return head + ("+dirty" if dirty else "")
    except Exception:  # pragma: no cover - git missing
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", choices=sorted(BLOCKS), required=True)
    ap.add_argument(
        "--source",
        choices=("training", "ranking", "allyear_plus_ranking"),
        default="training",
        help="allyear_plus_ranking = training months 1-12 + both ranking months in one file",
    )
    ap.add_argument("--months", default="1-10")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--variant", default="v1", help="builder variant (M1: v1|v2; M2: v1|v2)")
    ap.add_argument("--arr-guard", choices=("off", *arrival_guard.MODES), default="off",
                    help="arrival-label sanity guard (taxi-in < 0 or > 3 h -> missing; prc_challenge.arrival_guard); "
                    "new blocks only: the output directory gets the suffix _argd so deployed blocks are never overwritten")
    args = ap.parse_args()

    dirname, version, builder_fn = BLOCKS[args.block]
    dirname = dirname.replace("_v1", f"_{args.variant}")
    version = version.replace("-v1", f"-{args.variant}")
    if args.arr_guard != "off":
        dirname += "_argd"
        version += f"+arrguard-{args.arr_guard}"

    def builder(mov):
        if args.arr_guard != "off":
            mov = arrival_guard.guard_movements(mov, args.arr_guard)
        return builder_fn(mov, variant=args.variant)
    t0 = time.time()
    parts, coverage = [], []
    if args.source == "training":
        months = _months(args.months)
        tag = f"months_{months[0]}_{months[-1]}" if len(months) > 1 else f"month_{months[0]}"
        for m in months:
            mov = load_training_month(m, args.raw_dir)
            feats = builder(mov)
            parts.append(feats)
            coverage.append({"window": f"file-2025-{m:02d}", "dep_rows": feats.height})
            print(f"[{args.block}] month {m}: {feats.height} rows ({time.time() - t0:.0f}s)", flush=True)
    elif args.source == "allyear_plus_ranking":
        tag = "allyear_plus_ranking"
        for m in range(1, 13):
            feats = builder(load_training_month(m, args.raw_dir))
            parts.append(feats)
            coverage.append({"window": f"file-2025-{m:02d}", "dep_rows": feats.height})
            print(f"[{args.block}] month {m}: {feats.height} rows ({time.time() - t0:.0f}s)", flush=True)
        mov = load_ranking(args.raw_dir)
        parts.append(builder(mov))
        for w, cnt in mov.filter(pl.col("phase") == "DEP").group_by("window").len().sort("window").iter_rows():
            coverage.append({"window": f"ranking-{w}", "dep_rows": cnt})
    else:
        tag = "ranking"
        mov = load_ranking(args.raw_dir)
        feats = builder(mov)
        parts.append(feats)
        for w, cnt in mov.filter(pl.col("phase") == "DEP").group_by("window").len().sort("window").iter_rows():
            coverage.append({"window": w, "dep_rows": cnt})
    out = pl.concat(parts, how="vertical")
    if out["MVT_ID_mvt"].n_unique() != out.height:
        raise RuntimeError("duplicate MVT_ID_mvt across windows")
    folder = args.out_root / dirname
    folder.mkdir(parents=True, exist_ok=True)
    identity_checks = {}
    if args.source == "allyear_plus_ranking":
        # The combined file must reproduce the screened pieces exactly.
        for piece in ("months_1_10", "ranking"):
            ref_path = folder / f"{piece}.parquet"
            if not ref_path.exists():
                identity_checks[piece] = "reference missing"
                continue
            ref = pl.read_parquet(ref_path).sort("MVT_ID_mvt")
            sub = out.join(ref.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi").sort("MVT_ID_mvt")
            if sub.height != ref.height or not sub.equals(ref, null_equal=True):
                raise RuntimeError(f"combined {dirname} differs from {ref_path}")
            identity_checks[piece] = f"identical ({ref.height} rows)"
            print(f"[{args.block}] {piece}: identical to {ref_path.name} ({ref.height} rows)", flush=True)
    path = folder / f"{tag}.parquet"
    out.write_parquet(path)
    cols = [c for c in out.columns if c != "MVT_ID_mvt"]
    nulls = out.select([pl.col(c).is_null().mean().alias(c) for c in cols]).row(0, named=True)
    manifest = {
        "block": args.block,
        "version": version,
        "path": str(path),
        "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "identity_checks": identity_checks,
        **({"arrival_guard": {"mode": args.arr_guard, "taxi_in_bounds_sec": [arrival_guard.TAXI_IN_MIN_SEC, arrival_guard.TAXI_IN_MAX_SEC]}} if args.arr_guard != "off" else {}),
        "rows": out.height,
        "columns": cols,
        "null_fraction": nulls,
        "source": args.source,
        "months": _months(args.months) if args.source == "training" else ([1, 7] if args.source == "ranking" else "2025-01..2025-12 + ranking 2026-01, 2026-07"),
        "coverage": coverage,
        "source_mode": SOURCE_MODE,
        "variant": args.variant,
        "builder": f"scripts/features/build_matched_context_blocks.py --block {args.block} --variant {args.variant} --source {args.source}"
        + (f" --months {args.months}" if args.source == "training" else ""),
        "windows": "each training month file is one window; ranking split by calendar month of MVT_TIME_UTC_mvt",
        "builder_commit": _commit(),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (folder / f"{tag}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("block", "path", "rows", "sha256_16", "runtime_sec")}, indent=2))


if __name__ == "__main__":
    main()
