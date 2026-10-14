# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Download Iowa Environmental Mesonet METAR/ASOS for challenge airports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

STATIONS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
# Buffered challenge windows (UTC dates inclusive on IEM).
REQUESTS = [
    ("2024-12-31", "2026-08-01"),  # training 2025 + the final phase (Jan, Feb, Jun, Jul 2026) + buffers
]
VARS = ["tmpc", "dwpc", "relh", "drct", "sknt", "gust", "vsby", "wxcodes", "p01i", "skyc1", "skyl1"]
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "external" / "iem_metar"


def fetch(station: str, start: str, end: str) -> bytes:
    y1, m1, d1 = (int(x) for x in start.split("-"))
    y2, m2, d2 = (int(x) for x in end.split("-"))
    params = [
        ("station", station),
        *[("data", v) for v in VARS],
        ("year1", y1),
        ("month1", m1),
        ("day1", d1),
        ("year2", y2),
        ("month2", m2),
        ("day2", d2),
        ("tz", "UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "null"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", "3"),
        ("report_type", "4"),
    ]
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "prc-challenge-2026-research/0.1"})
    with urlopen(req, timeout=180) as resp:
        return resp.read()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for station in STATIONS:
        chunks = []
        for start, end in REQUESTS:
            print(f"fetch {station} {start}..{end}", flush=True)
            chunks.append(fetch(station, start, end).decode("utf-8", errors="replace"))
        header = None
        rows = []
        for chunk in chunks:
            lines = [ln for ln in chunk.splitlines() if ln.strip()]
            if not lines:
                continue
            if header is None:
                header = lines[0]
            rows.extend(ln for ln in lines[1:] if not ln.startswith("station,"))
        text = (header or "station,valid") + "\n" + "\n".join(rows) + "\n"
        path = DEST / f"{station}.csv"
        path.write_text(text)
        digest = hashlib.sha256(text.encode()).hexdigest()
        rec = {
            "station": station,
            "path": str(path),
            "bytes": path.stat().st_size,
            "n_lines": text.count("\n"),
            "sha256": digest,
        }
        records.append(rec)
        print(f"  wrote {path.name} lines={rec['n_lines']} bytes={rec['bytes']}", flush=True)
    manifest = {
        "source": "Iowa Environmental Mesonet ASOS/METAR",
        "url": "https://mesonet.agron.iastate.edu/request/download.phtml",
        "disclaimer": "https://mesonet.agron.iastate.edu/disclaimer.php",
        "license_note": "IEM describes site materials as public domain; verify dataset-specific exceptions.",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "stations": records,
        "variables": VARS,
        "windows": REQUESTS,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("done", DEST)


if __name__ == "__main__":
    main()
