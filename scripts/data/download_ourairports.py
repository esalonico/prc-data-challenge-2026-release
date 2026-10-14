# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Download the OurAirports airport and runway tables (public domain) into data/external/ourairports/.

Usage: python scripts/data/download_ourairports.py
Writes airports.csv, runways.csv and manifest.json (URL, retrieval time, SHA-256). The tables change daily; the exact
snapshot the final build used is in the release bucket (manifests/artifacts.sha256).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "ourairports"
BASE = "https://davidmegginson.github.io/ourairports-data/"
FILES = ("airports.csv", "runways.csv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in FILES:
        url = BASE + name
        print(f"fetch {url}", flush=True)
        with urlopen(Request(url, headers={"User-Agent": "prc-challenge-2026/1.0"}), timeout=120) as r:
            data = r.read()
        (OUT / name).write_bytes(data)
        rows.append({"file": name, "url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"source": "OurAirports (public domain), https://ourairports.com/data/",
                "retrieved_utc": datetime.now(timezone.utc).isoformat(), "files": rows}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for r in rows:
        print(f"  {r['file']}  {r['bytes']}  {r['sha256'][:12]}")


if __name__ == "__main__":
    main()
