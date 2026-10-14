#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Download the release artifacts (stored models and open-data snapshots) into this checkout and verify them.

    python scripts/release/fetch_artifacts.py                     # everything in manifests/artifacts.sha256
    python scripts/release/fetch_artifacts.py --only artifacts/experiments/S4-allyear-rebuild-20260925/
    python scripts/release/fetch_artifacts.py --verify-only       # check files already present

Each manifest line is ``<sha256>  <bytes>  <repo-relative path>``; the header line ``# base_url: <url>`` gives the
public bucket prefix (override with --base-url or PRC_ARTIFACTS_URL). Every file is placed at its path inside this
checkout, so the pipeline runs unchanged. Partial downloads resume; a file whose size and SHA-256 already match is
skipped. Exit 1 if any file fails verification.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "manifests" / "artifacts.sha256"
USER_AGENT = "prc-challenge-2026-fetch/1.0"  # the bucket host refuses urllib's default agent


def read_manifest(path: Path) -> tuple[str | None, list[tuple[str, int, str]]]:
    base, rows = None, []
    for line in path.read_text().splitlines():
        if line.startswith("# base_url:"):
            base = line.split(":", 1)[1].strip()
        elif line.strip() and not line.startswith("#"):
            sha, size, rel = line.split(None, 2)
            rows.append((sha, int(size), rel))
    return base, rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ok(path: Path, sha: str, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256(path) == sha


def fetch(base: str, sha: str, size: int, rel: str, retries: int = 3) -> str:
    dst = REPO / rel
    if ok(dst, sha, size):
        return f"ok      {rel}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    url = f"{base.rstrip('/')}/{rel}"
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have < size:
            headers = {"User-Agent": USER_AGENT, **({"Range": f"bytes={have}-"} if have else {})}
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            except OSError as e:
                if attempt == retries - 1:
                    return f"FAILED  {rel}: {e}"
                continue
        if part.stat().st_size == size and sha256(part) == sha:
            os.replace(part, dst)
            return f"fetched {rel}"
        part.unlink()
    return f"FAILED  {rel}: size or SHA-256 mismatch"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--base-url", default=os.environ.get("PRC_ARTIFACTS_URL"))
    ap.add_argument("--only", action="append", default=[], help="path prefix to fetch (repeatable)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    base, rows = read_manifest(a.manifest)
    base = a.base_url or base
    rows = [r for r in rows if not a.only or any(r[2].startswith(p) for p in a.only)]
    if a.verify_only:
        bad = [rel for sha, size, rel in rows if not ok(REPO / rel, sha, size)]
        for rel in bad:
            print("MISSING/BAD", rel)
        print(f"{len(rows) - len(bad)} of {len(rows)} files verified")
        return 1 if bad else 0
    if not base:
        raise SystemExit("no base URL: the manifest has none; pass --base-url or set PRC_ARTIFACTS_URL")
    total = sum(r[1] for r in rows)
    print(f"{len(rows)} files, {total / 1e9:.2f} GB from {base}")
    failed = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for fut in as_completed([ex.submit(fetch, base, *r) for r in rows]):
            msg = fut.result()
            failed += msg.startswith("FAILED")
            print(msg, flush=True)
    print(f"{len(rows) - failed} of {len(rows)} files verified")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
