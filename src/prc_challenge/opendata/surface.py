# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Routable OSM + OurAirports airport-surface model.

Builds a taxiway / taxilane / runway graph, snaps parking stands and runway
thresholds, and derives shortest-path distance, route complexity, junctions,
runway crossings, apron-exit distance and training-safe unimpeded-time proxies.

Straight-line haversine is a fallback only, never the sole route representation.
Unimpeded time uses published taxi speeds, never taxi labels.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from prc_challenge.geometry import haversine_m, load_stand_coords, normalize_stand_key_expr
from prc_challenge.runways import load_runway_thresholds

EARTH_RADIUS_M = 6_371_000.0
TAXIWAY_MPS = 15.0 * 0.514444  # 15 kt
APRON_MPS = 10.0 * 0.514444  # 10 kt
SNAP_STAND_M = 120.0
SNAP_RUNWAY_M = 180.0
APRON_LINK_M = 90.0
TURN_DEG = 45.0
ROUTABLE_AEROWAY = frozenset({"taxiway", "taxilane", "runway", "parking_position"})
TAXI_AEROWAY = frozenset({"taxiway", "taxilane"})

SURFACE_NUMS = (
    "osm_path_m",
    "osm_haversine_m",
    "osm_path_haversine_ratio",
    "osm_n_edges",
    "osm_n_junctions",
    "osm_n_turns",
    "osm_n_runway_crossings",
    "osm_apron_exit_m",
    "osm_unimpeded_sec",
    "osm_graph_confidence",
    "osm_route_missing",
    "osm_stand_snap_m",
    "osm_rwy_snap_m",
    "osm_fallback_haversine",
)

FORBIDDEN_SURFACE_INPUTS = frozenset({"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt"})


def _haversine_np(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rad = math.pi / 180.0
    phi1 = lat1 * rad
    phi2 = lat2 * rad
    dphi = (lat2 - lat1) * rad
    dlmb = (lon2 - lon1) * rad
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rad = math.pi / 180.0
    phi1 = lat1 * rad
    phi2 = lat2 * rad
    dlmb = (lon2 - lon1) * rad
    y = math.sin(dlmb) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlmb)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def reject_forbidden_surface_inputs(frame: Mapping[str, Any] | pl.DataFrame | Sequence[str]) -> None:
    if isinstance(frame, pl.DataFrame):
        cols = set(frame.columns)
    elif isinstance(frame, Mapping):
        cols = set(frame.keys())
    else:
        cols = set(frame)
    bad = sorted(FORBIDDEN_SURFACE_INPUTS.intersection(cols))
    if bad:
        raise RuntimeError(f"surface features cannot read {bad}")


@dataclass
class AirportGraph:
    airport: str
    nodes: dict[int, tuple[float, float]]
    adj: dict[int, list[tuple[int, float, str]]]
    degree: dict[int, int]
    runway_nodes: set[int]
    taxi_nodes: set[int]
    parking_nodes: dict[str, int]
    n_edges: int
    source: str = "osm_overpass"

    def connected(self) -> bool:
        return self.n_edges > 0 and bool(self.nodes)


def parse_overpass_elements(
    payload: Mapping[str, Any],
) -> tuple[dict[int, tuple[float, float]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Split Overpass JSON into node coordinates and routable ways."""

    nodes: dict[int, tuple[float, float]] = {}
    ways: list[dict[str, Any]] = []
    parking_nodes: dict[int, dict[str, Any]] = {}
    for el in payload.get("elements") or []:
        etype = el.get("type")
        eid = el.get("id")
        tags = el.get("tags") or {}
        if etype == "node" and "lat" in el and "lon" in el and eid is not None:
            nodes[int(eid)] = (float(el["lat"]), float(el["lon"]))
            aeroway = tags.get("aeroway")
            if aeroway in {"parking_position", "gate", "holding_position"}:
                parking_nodes[int(eid)] = {"id": int(eid), "aeroway": aeroway, "tags": tags, "lat": float(el["lat"]), "lon": float(el["lon"])}
        elif etype == "way" and eid is not None:
            nds = [int(n) for n in (el.get("nodes") or [])]
            geom = el.get("geometry") or []
            if geom:
                if nds and len(nds) == len(geom):
                    for nid, pt in zip(nds, geom):
                        if pt and "lat" in pt and "lon" in pt:
                            nodes[int(nid)] = (float(pt["lat"]), float(pt["lon"]))
                else:
                    nds = []
                    for i, pt in enumerate(geom):
                        if not pt or "lat" not in pt:
                            continue
                        nid = int(eid) * 10_000 + i
                        nodes[nid] = (float(pt["lat"]), float(pt["lon"]))
                        nds.append(nid)
            if len(nds) < 2:
                continue
            ways.append({"id": int(eid), "aeroway": tags.get("aeroway"), "nodes": nds, "tags": tags})
        elif etype in {"node", "way"} and "center" in el and eid is not None:
            center = el["center"] or {}
            if "lat" in center and "lon" in center:
                nodes[int(eid)] = (float(center["lat"]), float(center["lon"]))
    return nodes, ways, parking_nodes


def _stand_refs(tags: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("ref", "local_ref", "name", "ref:icao"):
        val = tags.get(key)
        if not val:
            continue
        for part in str(val).split(";"):
            part = part.strip().upper().replace(" ", "").replace("-", "")
            if part:
                refs.append(part)
    return list(dict.fromkeys(refs))


def build_airport_graph(
    payload: Mapping[str, Any],
    *,
    airport: str,
    extra_parking: Sequence[Mapping[str, Any]] | None = None,
) -> AirportGraph:
    nodes, ways, parking_nodes = parse_overpass_elements(payload)
    adj: dict[int, list[tuple[int, float, str]]] = {nid: [] for nid in nodes}
    runway_nodes: set[int] = set()
    taxi_nodes: set[int] = set()
    n_edges = 0

    def add_edge(a: int, b: int, length: float, aeroway: str) -> None:
        nonlocal n_edges
        if a not in adj:
            adj[a] = []
        if b not in adj:
            adj[b] = []
        adj[a].append((b, length, aeroway))
        adj[b].append((a, length, aeroway))
        n_edges += 1

    for way in ways:
        aeroway = way.get("aeroway") or ""
        if aeroway not in ROUTABLE_AEROWAY:
            continue
        nds = [n for n in way["nodes"] if n in nodes]
        for u, v in zip(nds, nds[1:]):
            lat1, lon1 = nodes[u]
            lat2, lon2 = nodes[v]
            length = _haversine_np(lat1, lon1, lat2, lon2)
            if length <= 0:
                continue
            add_edge(u, v, length, aeroway)
            if aeroway == "runway":
                runway_nodes.update((u, v))
            if aeroway in TAXI_AEROWAY:
                taxi_nodes.update((u, v))

    stand_to_node: dict[str, int] = {}
    for nid, meta in parking_nodes.items():
        for ref in _stand_refs(meta.get("tags") or {}):
            stand_to_node.setdefault(ref, nid)
        if meta.get("aeroway") in TAXI_AEROWAY:
            taxi_nodes.add(nid)

    if extra_parking:
        next_id = min(nodes.keys(), default=0) - 1
        for row in extra_parking:
            lat = float(row["lat"])
            lon = float(row["lon"])
            ref = str(row.get("stand_ref") or row.get("stand_key") or "").upper().replace(" ", "").replace("-", "")
            if not ref:
                continue
            nid = int(row["osm_id"]) if row.get("osm_id") is not None else next_id
            next_id -= 1
            nodes[nid] = (lat, lon)
            adj.setdefault(nid, [])
            stand_to_node.setdefault(ref, nid)

    taxi_list = list(taxi_nodes) if taxi_nodes else [nid for nid, nbrs in adj.items() if nbrs]
    for nid, (lat, lon) in list(nodes.items()):
        if nid in taxi_nodes or nid in runway_nodes:
            continue
        if nid not in stand_to_node.values() and nid not in parking_nodes:
            continue
        best = None
        best_d = APRON_LINK_M
        for tid in taxi_list:
            tlat, tlon = nodes[tid]
            dist = _haversine_np(lat, lon, tlat, tlon)
            if dist < best_d:
                best_d = dist
                best = tid
        if best is not None:
            add_edge(nid, best, best_d, "apron_link")

    degree = {nid: len(nbrs) for nid, nbrs in adj.items()}
    return AirportGraph(
        airport=airport,
        nodes=nodes,
        adj=adj,
        degree=degree,
        runway_nodes=runway_nodes,
        taxi_nodes=taxi_nodes,
        parking_nodes=stand_to_node,
        n_edges=n_edges,
    )


def _nearest_node(
    graph: AirportGraph,
    lat: float,
    lon: float,
    *,
    allowed: Iterable[int] | None = None,
    max_m: float,
) -> tuple[int | None, float]:
    pool = list(allowed) if allowed is not None else list(graph.nodes)
    best_id = None
    best_d = max_m
    for nid in pool:
        nlat, nlon = graph.nodes[nid]
        dist = _haversine_np(lat, lon, nlat, nlon)
        if dist < best_d:
            best_d = dist
            best_id = nid
    if best_id is None:
        return None, float("nan")
    return best_id, best_d


def _dijkstra(graph: AirportGraph, start: int, goal: int) -> tuple[list[int], list[tuple[int, int, float, str]]] | None:
    if start == goal:
        return [start], []
    heap: list[tuple[float, int]] = [(0.0, start)]
    dist = {start: 0.0}
    prev: dict[int, tuple[int, float, str]] = {}
    seen: set[int] = set()
    while heap:
        cost, node = heapq.heappop(heap)
        if node in seen:
            continue
        seen.add(node)
        if node == goal:
            break
        for nbr, length, aeroway in graph.adj.get(node, []):
            cand = cost + length
            if cand < dist.get(nbr, math.inf):
                dist[nbr] = cand
                prev[nbr] = (node, length, aeroway)
                heapq.heappush(heap, (cand, nbr))
    if goal not in dist:
        return None
    nodes = [goal]
    edges: list[tuple[int, int, float, str]] = []
    cur = goal
    while cur != start:
        parent, length, aeroway = prev[cur]
        edges.append((parent, cur, length, aeroway))
        nodes.append(parent)
        cur = parent
    nodes.reverse()
    edges.reverse()
    return nodes, edges


def _unimpeded_sec(path_m: float, apron_exit_m: float) -> float:
    apron = min(max(apron_exit_m, 0.0), max(path_m, 0.0))
    rest = max(path_m - apron, 0.0)
    return apron / APRON_MPS + rest / TAXIWAY_MPS


def route_stand_to_runway(
    graph: AirportGraph,
    *,
    stand_key: str | None,
    stand_lat: float | None,
    stand_lon: float | None,
    rwy_lat: float | None,
    rwy_lon: float | None,
) -> dict[str, float]:
    """Shortest-path route or haversine fallback. Never reads taxi labels."""

    empty = {
        "osm_path_m": float("nan"),
        "osm_haversine_m": float("nan"),
        "osm_path_haversine_ratio": float("nan"),
        "osm_n_edges": 0.0,
        "osm_n_junctions": 0.0,
        "osm_n_turns": 0.0,
        "osm_n_runway_crossings": 0.0,
        "osm_apron_exit_m": float("nan"),
        "osm_unimpeded_sec": float("nan"),
        "osm_graph_confidence": 0.0,
        "osm_route_missing": 1.0,
        "osm_stand_snap_m": float("nan"),
        "osm_rwy_snap_m": float("nan"),
        "osm_fallback_haversine": 0.0,
    }
    if stand_lat is None or stand_lon is None or rwy_lat is None or rwy_lon is None:
        return empty
    if not (math.isfinite(stand_lat) and math.isfinite(stand_lon) and math.isfinite(rwy_lat) and math.isfinite(rwy_lon)):
        return empty
    haversine = _haversine_np(stand_lat, stand_lon, rwy_lat, rwy_lon)
    empty["osm_haversine_m"] = haversine

    start = None
    start_snap = float("nan")
    key = (stand_key or "").upper().replace(" ", "").replace("-", "")
    if key and key in graph.parking_nodes:
        start = graph.parking_nodes[key]
        slat, slon = graph.nodes[start]
        start_snap = _haversine_np(stand_lat, stand_lon, slat, slon)
    if start is None:
        start, start_snap = _nearest_node(graph, stand_lat, stand_lon, max_m=SNAP_STAND_M)
    goal, goal_snap = _nearest_node(
        graph,
        rwy_lat,
        rwy_lon,
        allowed=graph.runway_nodes or None,
        max_m=SNAP_RUNWAY_M,
    )
    if goal is None:
        goal, goal_snap = _nearest_node(graph, rwy_lat, rwy_lon, max_m=SNAP_RUNWAY_M)

    routed = _dijkstra(graph, start, goal) if start is not None and goal is not None else None
    if routed is None:
        out = dict(empty)
        out.update(
            {
                "osm_path_m": haversine,
                "osm_path_haversine_ratio": 1.0,
                "osm_apron_exit_m": haversine,
                "osm_unimpeded_sec": _unimpeded_sec(haversine, haversine),
                "osm_graph_confidence": 0.25 if graph.connected() else 0.1,
                "osm_route_missing": 1.0,
                "osm_stand_snap_m": start_snap,
                "osm_rwy_snap_m": goal_snap,
                "osm_fallback_haversine": 1.0,
            }
        )
        return out

    node_ids, edges = routed
    path_m = float(sum(length for _, _, length, _ in edges)) if edges else 0.0
    n_junctions = sum(1 for nid in node_ids[1:-1] if graph.degree.get(nid, 0) >= 3)
    n_cross = 0
    prev_aw = None
    for _, _, _, aeroway in edges:
        if aeroway == "runway" and prev_aw != "runway":
            n_cross += 1
        prev_aw = aeroway
    n_turns = 0
    for i in range(1, len(node_ids) - 1):
        a = graph.nodes[node_ids[i - 1]]
        b = graph.nodes[node_ids[i]]
        c = graph.nodes[node_ids[i + 1]]
        b1 = _bearing_deg(a[0], a[1], b[0], b[1])
        b2 = _bearing_deg(b[0], b[1], c[0], c[1])
        if _angle_delta(b1, b2) >= TURN_DEG:
            n_turns += 1
    apron_exit = 0.0
    for _, _, length, aeroway in edges:
        if aeroway in {"apron_link", "parking_position"}:
            apron_exit += length
        else:
            break
    if apron_exit <= 0 and start is not None:
        nearest_taxi, taxi_d = _nearest_node(graph, stand_lat, stand_lon, allowed=graph.taxi_nodes or None, max_m=SNAP_STAND_M * 2)
        apron_exit = taxi_d if nearest_taxi is not None else min(path_m, 80.0)
    taxi_frac = 0.0
    if path_m > 0:
        taxi_frac = sum(length for _, _, length, aw in edges if aw in TAXI_AEROWAY | {"apron_link"}) / path_m
    conf = 1.0
    if start_snap > 80 or goal_snap > 120 or taxi_frac < 0.35:
        conf = 0.6
    if not edges:
        conf = 0.4
    ratio = path_m / haversine if haversine > 1.0 else 1.0
    return {
        "osm_path_m": path_m,
        "osm_haversine_m": haversine,
        "osm_path_haversine_ratio": ratio,
        "osm_n_edges": float(len(edges)),
        "osm_n_junctions": float(n_junctions),
        "osm_n_turns": float(n_turns),
        "osm_n_runway_crossings": float(n_cross),
        "osm_apron_exit_m": float(apron_exit),
        "osm_unimpeded_sec": _unimpeded_sec(path_m, apron_exit),
        "osm_graph_confidence": conf,
        "osm_route_missing": 0.0,
        "osm_stand_snap_m": float(start_snap),
        "osm_rwy_snap_m": float(goal_snap),
        "osm_fallback_haversine": 0.0,
    }


def build_route_lookup(
    graphs: Mapping[str, AirportGraph],
    stands: pl.DataFrame,
    thresholds: pl.DataFrame,
) -> pl.DataFrame:
    """Precompute airport×stand×runway routes. Label-free."""

    rows: list[dict[str, Any]] = []
    stand_rows = stands.select(["AIRPORT", "stand_key", "stand_lat", "stand_lon"]).unique()
    thresh_rows = thresholds.select(["AIRPORT", "RUNWAY_mvt", "rwy_lat", "rwy_lon"]).unique()
    for airport, graph in graphs.items():
        local_stands = stand_rows.filter(pl.col("AIRPORT") == airport)
        local_rwy = thresh_rows.filter(pl.col("AIRPORT") == airport)
        for stand in local_stands.iter_rows(named=True):
            for rwy in local_rwy.iter_rows(named=True):
                feat = route_stand_to_runway(
                    graph,
                    stand_key=stand["stand_key"],
                    stand_lat=stand["stand_lat"],
                    stand_lon=stand["stand_lon"],
                    rwy_lat=rwy["rwy_lat"],
                    rwy_lon=rwy["rwy_lon"],
                )
                feat.update({"AIRPORT": airport, "stand_key": stand["stand_key"], "RUNWAY_mvt": rwy["RUNWAY_mvt"]})
                rows.append(feat)
    if not rows:
        return pl.DataFrame(
            schema={
                "AIRPORT": pl.String,
                "stand_key": pl.String,
                "RUNWAY_mvt": pl.String,
                **{name: pl.Float64 for name in SURFACE_NUMS},
            }
        )
    return pl.DataFrame(rows).unique(["AIRPORT", "stand_key", "RUNWAY_mvt"], keep="first")


def add_surface_features(
    df: pl.DataFrame,
    lookup: pl.DataFrame,
    *,
    stands: pl.DataFrame | None = None,
    thresholds: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join precomputed routes; haversine fallback if the pair is missing.

    Label columns may sit on the frame for later scoring; they are never read.
    """

    out = df.with_columns(
        [
            normalize_stand_key_expr().alias("_stand_key"),
            pl.col("RUNWAY_mvt").cast(pl.String).str.to_uppercase().alias("_rwy_key"),
        ]
    )
    n0 = out.height
    lk = lookup
    if {"AIRPORT", "stand_key", "RUNWAY_mvt"}.issubset(lk.columns):
        lk = lk.unique(["AIRPORT", "stand_key", "RUNWAY_mvt"], keep="first")
    mapped = out.join(
        lk.rename({"stand_key": "_stand_key", "RUNWAY_mvt": "_rwy_key"}),
        on=["AIRPORT", "_stand_key", "_rwy_key"],
        how="left",
    )
    if mapped.height != n0:
        raise RuntimeError(f"OSM route lookup exploded rows {n0} -> {mapped.height}")
    if "osm_haversine_m" not in mapped.columns or mapped["osm_haversine_m"].null_count():
        if stands is None:
            stands = load_stand_coords()
        if thresholds is None:
            thresholds = load_runway_thresholds()
        stands_u = stands.unique(["AIRPORT", "stand_key"], keep="first") if "stand_key" in stands.columns else stands
        thresh_u = thresholds.unique(["AIRPORT", "RUNWAY_mvt"], keep="first") if "RUNWAY_mvt" in thresholds.columns else thresholds
        geo = mapped.join(stands_u, left_on=["AIRPORT", "_stand_key"], right_on=["AIRPORT", "stand_key"], how="left")
        if geo.height != n0:
            raise RuntimeError(f"stand coordinate join exploded rows {n0} -> {geo.height}")
        geo = geo.join(thresh_u.rename({"RUNWAY_mvt": "_rwy_key"}), on=["AIRPORT", "_rwy_key"], how="left")
        if geo.height != n0:
            raise RuntimeError(f"runway threshold join exploded rows {n0} -> {geo.height}")
        dist = haversine_m(pl.col("stand_lat"), pl.col("stand_lon"), pl.col("rwy_lat"), pl.col("rwy_lon"))
        mapped = geo.with_columns(
            [
                pl.coalesce([pl.col("osm_haversine_m"), dist]).alias("osm_haversine_m"),
                pl.coalesce([pl.col("osm_path_m"), dist]).alias("osm_path_m"),
            ]
        ).drop(["stand_lat", "stand_lon", "rwy_lat", "rwy_lon"], strict=False)
    mapped = mapped.with_columns(
        [
            pl.col("osm_route_missing").fill_null(1).cast(pl.Float64).alias("osm_route_missing"),
            pl.col("osm_graph_confidence").fill_null(0).cast(pl.Float64).alias("osm_graph_confidence"),
            pl.col("osm_fallback_haversine").fill_null(1).cast(pl.Float64).alias("osm_fallback_haversine"),
            pl.when(pl.col("osm_unimpeded_sec").is_null() & pl.col("osm_path_m").is_not_null())
            .then(pl.col("osm_path_m") / TAXIWAY_MPS)
            .otherwise(pl.col("osm_unimpeded_sec"))
            .alias("osm_unimpeded_sec"),
        ]
    )
    return mapped.drop(["_stand_key", "_rwy_key"], strict=False)


def graph_coverage_report(lookup: pl.DataFrame) -> dict[str, Any]:
    if lookup.height == 0:
        return {"n_pairs": 0, "routed_rate": 0.0}
    routed = lookup.filter(pl.col("osm_route_missing") == 0)
    return {
        "n_pairs": lookup.height,
        "n_airports": lookup["AIRPORT"].n_unique() if "AIRPORT" in lookup.columns else 0,
        "routed_rate": float(routed.height / lookup.height),
        "median_path_m": float(routed["osm_path_m"].median()) if routed.height else None,
        "median_ratio": float(routed["osm_path_haversine_ratio"].median()) if routed.height else None,
        "mean_confidence": float(lookup["osm_graph_confidence"].mean()),
        "fallback_rate": float(lookup["osm_fallback_haversine"].mean()) if "osm_fallback_haversine" in lookup.columns else None,
        "by_airport": lookup.group_by("AIRPORT")
        .agg(
            pl.len().alias("n"),
            (1 - pl.col("osm_route_missing").mean()).alias("routed_rate"),
            pl.col("osm_graph_confidence").mean().alias("confidence"),
        )
        .sort("AIRPORT")
        .to_dicts(),
    }


def synthetic_l_shaped_airport() -> dict[str, Any]:
    """Stand -- taxiway -- junction -- taxiway -- threshold, with a runway crossing option."""

    nodes = {
        1: {"type": "node", "id": 1, "lat": 0.0, "lon": 0.0, "tags": {"aeroway": "parking_position", "ref": "A1"}},
        2: {"type": "node", "id": 2, "lat": 0.0, "lon": 0.002},
        3: {"type": "node", "id": 3, "lat": 0.0, "lon": 0.004},
        4: {"type": "node", "id": 4, "lat": 0.003, "lon": 0.004},
        5: {"type": "node", "id": 5, "lat": 0.006, "lon": 0.004, "tags": {"aeroway": "holding_position"}},
        6: {"type": "node", "id": 6, "lat": 0.003, "lon": 0.000},
    }
    elements: list[dict[str, Any]] = list(nodes.values())
    elements.extend(
        [
            {"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"aeroway": "taxiway"}},
            {"type": "way", "id": 11, "nodes": [3, 4, 5], "tags": {"aeroway": "taxiway"}},
            {"type": "way", "id": 12, "nodes": [6, 4], "tags": {"aeroway": "runway"}},
        ]
    )
    return {"elements": elements}
