# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Download OSM parking-position nodes/ways for challenge airports (ODbL)."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import polars as pl

from prc_challenge.io import CHALLENGE_AIRPORTS

ROOT = Path(__file__).resolve().parents[2]
OA_DIR = ROOT / "data" / "external" / "ourairports"
OSM_DIR = ROOT / "data" / "external" / "osm"
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
USER_AGENT = "prc-challenge-2026-research/0.1 (OSM parking_position cache)"
BBOX_DEG = 0.07  # ~8 km


def fetch(url: str, data: bytes | None = None, timeout: int = 180) -> bytes:
    req = Request(url, data=data, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def overpass_query(south: float, west: float, north: float, east: float) -> str:
    return f"""[out:json][timeout:90];
(
  node["aeroway"="parking_position"]({south},{west},{north},{east});
  way["aeroway"="parking_position"]({south},{west},{north},{east});
  node["aeroway"="gate"]({south},{west},{north},{east});
);
out center tags;
"""


def parse_elements(payload: dict, airport: str) -> list[dict]:
    rows = []
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        else:
            center = el.get("center") or {}
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue
        refs = []
        for key in ("ref", "local_ref", "name", "ref:icao"):
            val = tags.get(key)
            if val:
                for part in str(val).split(";"):
                    part = part.strip()
                    if part:
                        refs.append(part)
        if not refs:
            continue
        for ref in dict.fromkeys(refs):
            rows.append(
                {
                    "AIRPORT": airport,
                    "osm_id": el.get("id"),
                    "osm_type": el.get("type"),
                    "aeroway": tags.get("aeroway"),
                    "stand_ref": ref,
                    "lat": float(lat),
                    "lon": float(lon),
                }
            )
    return rows


def query_airport(lat0: float, lon0: float) -> tuple[dict, str]:
    south, north = lat0 - BBOX_DEG, lat0 + BBOX_DEG
    west, east = lon0 - BBOX_DEG, lon0 + BBOX_DEG
    body = overpass_query(south, west, north, east).encode()
    last_err: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(3):
            try:
                raw = fetch(endpoint, data=body, timeout=180)
                payload = json.loads(raw.decode("utf-8"))
                return payload, endpoint
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_err = exc
                time.sleep(8 * (attempt + 1))
        time.sleep(5)
    raise RuntimeError(f"Overpass failed: {last_err}")


def main() -> None:
    OA_DIR.mkdir(parents=True, exist_ok=True)
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    airports_path = OA_DIR / "airports.csv"
    print(f"fetch {AIRPORTS_URL}", flush=True)
    airports_bytes = fetch(AIRPORTS_URL, timeout=120)
    airports_path.write_bytes(airports_bytes)
    airports = pl.read_csv(airports_path, infer_schema_length=5000).filter(
        pl.col("ident").is_in(list(CHALLENGE_AIRPORTS))
    )
    needed = {icao: None for icao in CHALLENGE_AIRPORTS}
    for row in airports.select(["ident", "latitude_deg", "longitude_deg"]).iter_rows(named=True):
        needed[row["ident"]] = (float(row["latitude_deg"]), float(row["longitude_deg"]))
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        raise RuntimeError(f"OurAirports missing ICAO coords: {missing}")

    all_rows: list[dict] = []
    per_airport = []
    for icao in CHALLENGE_AIRPORTS:
        lat0, lon0 = needed[icao]
        print(f"overpass {icao}", flush=True)
        payload, endpoint = query_airport(lat0, lon0)
        raw_path = OSM_DIR / f"{icao.lower()}_parking.json"
        raw_path.write_text(json.dumps(payload))
        rows = parse_elements(payload, icao)
        all_rows.extend(rows)
        per_airport.append(
            {
                "AIRPORT": icao,
                "n_elements": len(payload.get("elements", [])),
                "n_ref_rows": len(rows),
                "n_unique_refs": len({r["stand_ref"] for r in rows}),
                "endpoint": endpoint,
                "osm_timestamp": payload.get("osm3s", {}).get("timestamp_osm_base"),
            }
        )
        time.sleep(2)

    table = pl.DataFrame(all_rows) if all_rows else pl.DataFrame(
        schema={
            "AIRPORT": pl.String,
            "osm_id": pl.Int64,
            "osm_type": pl.String,
            "aeroway": pl.String,
            "stand_ref": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
        }
    )
    dest = OSM_DIR / "parking_positions.parquet"
    table.write_parquet(dest)
    manifest = {
        "source": "OpenStreetMap via Overpass API (parking_position + gate)",
        "license": "ODbL; attribution required: © OpenStreetMap contributors",
        "copyright_url": "https://www.openstreetmap.org/copyright",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "bbox_deg": BBOX_DEG,
        "airports_csv_sha256": hashlib.sha256(airports_bytes).hexdigest(),
        "n_ref_rows": table.height,
        "by_airport": per_airport,
        "local_path": str(dest),
        "note": "Current OSM extract, not a historical snapshot. First E10 use.",
    }
    (OSM_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(table.group_by("AIRPORT").agg(n=pl.len(), n_ref=pl.col("stand_ref").n_unique()).sort("AIRPORT"))
    print(f"wrote {dest} n={table.height}", flush=True)


if __name__ == "__main__":
    main()
