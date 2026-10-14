# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Download immutable challenge files into data/raw/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from prc_challenge.s3 import DATA_BUCKET, TEAM_BUCKET, TEAM_NAME, download_challenge_data

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    dest = ROOT / "data" / "raw"
    results = download_challenge_data(dest)
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_bucket": DATA_BUCKET,
        "team_bucket": TEAM_BUCKET,
        "team_name": TEAM_NAME,
        "files": results,
    }
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(item["bytes"] for item in results)
    skipped = sum(1 for item in results if item["skipped"])
    print(f"downloaded {len(results)} files ({total} bytes); skipped {skipped} already present")
    print(f"manifest: {manifest_path}")
    for item in results:
        flag = "skip" if item["skipped"] else "new"
        print(f"  [{flag}] {item['key']}  {item['bytes']}  {item['sha256'][:12]}")


if __name__ == "__main__":
    main()
