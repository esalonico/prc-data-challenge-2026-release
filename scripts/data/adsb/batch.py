# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Parallel batch driver for the adsb.lol stream-and-filter ingest (runs on the owner VM, D-011).

Reads a list of UTC days, picks the first available instance per day from a
pre-fetched asset index (prod-0, then prod-1, then staging-0), and runs
``ingest.py`` for each day with N workers. Skips days whose per-day
manifest exists, keeps a free-disk floor, and retries a failed day once.
Only filtered parquet and manifests are written; raw archives are streamed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTANCES = ("prod-0", "prod-1", "staging-0")


def tag_for(day: str, instance: str) -> str:
    y, m, d = day.split("-")
    return f"v{y}.{m}.{d}-planes-readsb-{instance}"


def run_day(day: str, index: dict, out_dir: Path, logs: Path, min_free_gb: float, extra: list[str]) -> dict:
    inst = next((i for i in INSTANCES if tag_for(day, i) in index), None)
    if inst is None:
        return {"day": day, "status": "no_release"}
    done = out_dir / f"adsb_lol_{day}_{inst}_full_manifest.json"
    if done.exists():
        return {"day": day, "instance": inst, "status": "skipped_done"}
    while shutil.disk_usage(out_dir).free / 1e9 < min_free_gb:
        time.sleep(60)
    cmd = [sys.executable, str(HERE / "ingest.py"), "--day", day, "--instance", inst,
           "--assets-json", str(index["__path__"]), "--out-dir", str(out_dir), "--no-global-manifest", *extra]
    for attempt in (1, 2):
        t0 = time.time()
        with open(logs / f"{day}_{inst}_try{attempt}.log", "w") as fh:
            rc = subprocess.run(["nice", "-n", "5", *cmd], stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc == 0 and done.exists():
            return {"day": day, "instance": inst, "status": "ok", "secs": round(time.time() - t0), "attempt": attempt}
    return {"day": day, "instance": inst, "status": f"failed_rc{rc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-file", type=Path, required=True)
    ap.add_argument("--assets-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=50.0)
    ap.add_argument("--radius-m", type=float, default=10000.0)
    ap.add_argument("--alt-margin-ft", type=float, default=5000.0)
    args = ap.parse_args()
    days = [d.strip() for d in args.days_file.read_text().split() if d.strip()]
    index = json.loads(args.assets_json.read_text())
    index["__path__"] = args.assets_json.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs = args.out_dir / "logs"
    logs.mkdir(exist_ok=True)
    extra = ["--radius-m", str(args.radius_m), "--alt-margin-ft", str(args.alt_margin_ft)]
    status_path = args.out_dir / "batch_status.jsonl"
    with ThreadPoolExecutor(args.workers) as pool:
        for res in pool.map(lambda d: run_day(d, index, args.out_dir, logs, args.min_free_gb, extra), days):
            with open(status_path, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
