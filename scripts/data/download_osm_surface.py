# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Download OSM taxiway/runway/parking graphs for challenge airports (ODbL)."""

from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from prc_challenge.io import CHALLENGE_AIRPORTS
from prc_challenge.opendata.http import USER_AGENT, fetch_bytes, write_manifest
from prc_challenge.opendata.surface import ROUTABLE_AEROWAY, build_airport_graph, parse_overpass_elements

ROOT = Path(__file__).resolve().parents[2]
OA_DIR = ROOT / "data" / "external" / "ourairports"
DEST = ROOT / "data" / "external" / "osm_surface"
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
BBOX_DEG = 0.06


def overpass_query(south: float, west: float, north: float, east: float) -> str:
    return f"""[out:json][timeout:90];
(
  way["aeroway"~"^(taxiway|taxilane|runway|parking_position)$"]({south},{west},{north},{east});
  node["aeroway"~"^(parking_position|gate|holding_position)$"]({south},{west},{north},{east});
);
out geom;
"""


def slim_payload(payload: dict) -> dict:
    """Keep aeroway ways plus the nodes they reference and tagged parking nodes."""

    keep_ids: set[int] = set()
    ways = []
    parking = []
    for el in payload.get("elements") or []:
        tags = el.get("tags") or {}
        if el.get("type") == "way" and tags.get("aeroway") in ROUTABLE_AEROWAY:
            ways.append(el)
            keep_ids.update(int(n) for n in (el.get("nodes") or []))
        elif el.get("type") == "node" and tags.get("aeroway") in {"parking_position", "gate", "holding_position"}:
            parking.append(el)
            keep_ids.add(int(el["id"]))
    nodes = [
        el
        for el in (payload.get("elements") or [])
        if el.get("type") == "node" and int(el.get("id", -1)) in keep_ids
    ]
    out = dict(payload)
    out["elements"] = nodes + parking + ways
    return out


def query_osm_map_tiles(lat0: float, lon0: float, *, radius: float = 0.035, step: float = 0.018) -> dict:
    south, north = lat0 - radius, lat0 + radius
    west, east = lon0 - radius, lon0 + radius
    nodes: dict[int, dict] = {}
    ways: dict[int, dict] = {}
    lat = south
    while lat < north:
        lon = west
        lat2 = min(lat + step, north)
        while lon < east:
            lon2 = min(lon + step, east)
            url = f"https://api.openstreetmap.org/api/0.6/map?bbox={lon:.5f},{lat:.5f},{lon2:.5f},{lat2:.5f}"
            raw = fetch_bytes(url, timeout=60)
            root = ET.fromstring(raw)
            for node in root.findall("node"):
                tags = {t.get("k"): t.get("v") for t in node.findall("tag")}
                nid = int(node.get("id"))
                nodes[nid] = {
                    "type": "node",
                    "id": nid,
                    "lat": float(node.get("lat")),
                    "lon": float(node.get("lon")),
                    "tags": tags,
                }
            for way in root.findall("way"):
                tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
                if tags.get("aeroway") not in ROUTABLE_AEROWAY:
                    continue
                wid = int(way.get("id"))
                ways[wid] = {
                    "type": "way",
                    "id": wid,
                    "nodes": [int(n.get("ref")) for n in way.findall("nd")],
                    "tags": tags,
                }
            time.sleep(0.6)
            lon = lon2
        lat = lat2
    return {"elements": list(nodes.values()) + list(ways.values()), "generator": "osm-api-0.6-map-tiles"}


def query_overpass(lat0: float, lon0: float) -> tuple[dict, str]:
    south, north = lat0 - BBOX_DEG, lat0 + BBOX_DEG
    west, east = lon0 - BBOX_DEG, lon0 + BBOX_DEG
    body = overpass_query(south, west, north, east).encode()
    last_err: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(2):
            try:
                raw = fetch_bytes(
                    endpoint, data=body, timeout=90, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
                )
                payload = json.loads(raw.decode("utf-8"))
                if payload.get("elements"):
                    return payload, endpoint
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"Overpass failed: {last_err}")


def query_airport(lat0: float, lon0: float) -> tuple[dict, str]:
    try:
        return query_overpass(lat0, lon0)
    except Exception as exc:  # noqa: BLE001
        print(f"  overpass failed ({exc}); falling back to OSM map API tiles", flush=True)
        return query_osm_map_tiles(lat0, lon0), "osm-api-0.6-map-tiles"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    OA_DIR.mkdir(parents=True, exist_ok=True)
    airports_path = OA_DIR / "airports.csv"
    if not airports_path.exists():
        airports_path.write_bytes(fetch_bytes(AIRPORTS_URL, timeout=120))
    airports = pl.read_csv(airports_path, infer_schema_length=5000).filter(pl.col("ident").is_in(list(CHALLENGE_AIRPORTS)))
    coords = {}
    for row in airports.select(["ident", "latitude_deg", "longitude_deg"]).iter_rows(named=True):
        coords[row["ident"]] = (float(row["latitude_deg"]), float(row["longitude_deg"]))
    missing = [icao for icao in CHALLENGE_AIRPORTS if icao not in coords]
    if missing:
        raise RuntimeError(f"OurAirports missing ICAO coords: {missing}")

    parking_path = ROOT / "data" / "external" / "osm" / "parking_positions.parquet"
    extra_parking = pl.read_parquet(parking_path) if parking_path.exists() else None

    summaries = []
    for icao in CHALLENGE_AIRPORTS:
        raw_path = DEST / f"{icao.lower()}_surface.json"
        if raw_path.exists() and raw_path.stat().st_size > 1000:
            print(f"reuse {icao}", flush=True)
            payload = slim_payload(json.loads(raw_path.read_text()))
            endpoint = "cache"
            raw_path.write_text(json.dumps(payload))
        else:
            lat0, lon0 = coords[icao]
            print(f"overpass surface {icao}", flush=True)
            payload, endpoint = query_airport(lat0, lon0)
            payload = slim_payload(payload)
            raw_path.write_text(json.dumps(payload))
            time.sleep(2)
        extra = None
        if extra_parking is not None:
            extra = extra_parking.filter(pl.col("AIRPORT") == icao).to_dicts()
        graph = build_airport_graph(payload, airport=icao, extra_parking=extra)
        nodes, ways, parking = parse_overpass_elements(payload)
        summaries.append(
            {
                "AIRPORT": icao,
                "endpoint": endpoint,
                "n_nodes": len(nodes),
                "n_ways": len(ways),
                "n_parking_nodes": len(parking),
                "n_graph_edges": graph.n_edges,
                "n_taxi_nodes": len(graph.taxi_nodes),
                "n_runway_nodes": len(graph.runway_nodes),
                "n_stand_keys": len(graph.parking_nodes),
                "osm_timestamp": payload.get("osm3s", {}).get("timestamp_osm_base"),
                "bytes": raw_path.stat().st_size,
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            }
        )
        print(
            f"  {icao} nodes={len(nodes)} edges={graph.n_edges} taxi={len(graph.taxi_nodes)} rwy={len(graph.runway_nodes)}",
            flush=True,
        )

    write_manifest(
        DEST / "manifest.json",
        {
            "source": "OpenStreetMap via Overpass API, with OSM map API 0.6 tiled fallback",
            "license": "ODbL; attribution required: © OpenStreetMap contributors",
            "copyright_url": "https://www.openstreetmap.org/copyright",
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "bbox_deg": BBOX_DEG,
            "ourairports": "public domain nightly dump used for bboxes",
            "by_airport": summaries,
            "note": "Current OSM extract, not a historical snapshot. Graph is label-free. Tiled OSM API used when Overpass times out.",
        },
    )
    print(pl.DataFrame(summaries).select(["AIRPORT", "n_nodes", "n_graph_edges", "n_taxi_nodes", "n_runway_nodes"]))


if __name__ == "__main__":
    main()
