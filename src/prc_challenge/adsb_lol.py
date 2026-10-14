# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Stream-and-filter reader for the adsb.lol historical archive (feasibility only).

Source: https://github.com/adsblol/globe_history_{year} GitHub releases, one release
per day and readsb instance (``v2025.02.12-planes-readsb-prod-0``). Each release is
one tar archive split into ~2 GB parts (``.tar.aa``, ``.tar.ab``, ...). The archive
is a dump of readsb's ``/var/globe_history/YYYY/MM/DD`` directory:

* ``heatmap/NN.bin.ttf`` (48 gzip files, one per half hour) come first in the tar.
  Each is a packed array of 16-byte ``heatEntry`` records (readsb ``globe_index.c``):
  one position per aircraft per heatmap interval, plus callsign records.
* ``traces/xx/trace_full_<hex>.json`` (gzip JSON, one per aircraft) follow. These
  hold every position readsb kept for the day (format: readsb ``README-json.md``).

Licence: ODbL 1.0 (database); feeders' contributions CC0. Attribution string used
here: "Contains data from ADSB.lol (https://adsb.lol), licensed under ODbL 1.0".

Everything here is label-free: filtering keeps low / on-ground samples within
``radius_m`` of the ten challenge airports. OpenSky data is not used anywhere.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tarfile
import time
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
from urllib.request import Request, urlopen

import numpy as np
import polars as pl

from prc_challenge.io import CHALLENGE_AIRPORTS

LICENCE = "ODbL-1.0"
LICENCE_URL = "https://opendatacommons.org/licenses/odbl/1-0/"
ATTRIBUTION = "Contains data from ADSB.lol (https://adsb.lol), licensed under ODbL 1.0"
USER_AGENT = "prc-challenge-2026-research/0.1 (adsb.lol feasibility check)"

EARTH_R = 6_371_000.0
HEAT_DTYPE = np.dtype(
    [("hex", "<i4"), ("lat", "<i4"), ("lon", "<i4"), ("alt", "<i2"), ("gs", "<i2")]
)
HEAT_SLICE_MAGIC = 0xE7F7C9D


@dataclass(frozen=True)
class Airport:
    icao: str
    lat: float
    lon: float
    elev_ft: float


def load_airports(csv_path: Path) -> dict[str, Airport]:
    df = pl.read_csv(csv_path, infer_schema_length=0).filter(
        pl.col("ident").is_in(list(CHALLENGE_AIRPORTS))
    )
    out = {}
    for r in df.iter_rows(named=True):
        out[r["ident"]] = Airport(
            r["ident"], float(r["latitude_deg"]), float(r["longitude_deg"]), float(r["elevation_ft"])
        )
    missing = set(CHALLENGE_AIRPORTS) - set(out)
    if missing:
        raise ValueError(f"airports.csv lacks {sorted(missing)}")
    return out


def release_urls(day: str, instance: str = "prod-0", n_parts: int | None = None) -> tuple[str, str]:
    """Return (repo, tag) for a UTC day ``YYYY-MM-DD``."""
    y, m, d = day.split("-")
    return f"adsblol/globe_history_{y}", f"v{y}.{m}.{d}-planes-readsb-{instance}"


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- sources


def _check_range(resp, offset: int) -> None:
    """A resumed read must be a 206 starting exactly at ``offset``.

    A server that ignores Range answers 200 with the whole file; splicing that into
    the stream corrupts the tar (seen once as a zlib error mid-archive).
    """
    if not offset:
        return
    cr = resp.headers.get("Content-Range", "")
    if resp.status != 206 or not cr.startswith(f"bytes {offset}-"):
        resp.close()
        raise OSError(f"range resume failed: status={resp.status} content-range={cr!r} offset={offset}")


class HashingReader(io.RawIOBase):
    """Read a sequence of byte sources as one stream, hashing each part.

    A source is a local path or an https URL. URLs are streamed with automatic
    resume (HTTP Range) on connection errors, so nothing is written to disk.
    """

    def __init__(self, sources: list[str], chunk: int = 1 << 20, timeout: int = 120):
        self.sources = list(sources)
        self.timeout = timeout
        self.idx = -1
        self.fh: BinaryIO | None = None
        self.pos_in_part = 0
        self.part_bytes: list[int] = []
        self.part_sha: list[str] = []
        self.part_secs: list[float] = []
        self._hash = None
        self._t0 = 0.0
        self.total = 0
        self.retries = 0
        self._next()

    def readable(self) -> bool:
        return True

    def _open(self, src: str, offset: int):
        if src.startswith("http"):
            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            resp = urlopen(Request(src, headers=headers), timeout=self.timeout)
            _check_range(resp, offset)
            return resp
        fh = open(src, "rb")  # noqa: SIM115 - kept open for streaming, closed in _next/close_all
        fh.seek(offset)
        return fh

    def _finish_part(self):
        if self._hash is not None:
            self.part_bytes.append(self.pos_in_part)
            self.part_sha.append(self._hash.hexdigest())
            self.part_secs.append(time.time() - self._t0)

    def _next(self):
        if self.fh is not None:
            self.fh.close()
        self._finish_part()
        self.idx += 1
        self.pos_in_part = 0
        if self.idx >= len(self.sources):
            self.fh = None
            self._hash = None
            return
        self._hash = hashlib.sha256()
        self._t0 = time.time()
        self.fh = self._open(self.sources[self.idx], 0)

    def readinto(self, b) -> int:
        while self.fh is not None:
            try:
                n = self.fh.readinto(b)
            except (OSError, TimeoutError):  # network hiccup: resume
                self.retries += 1
                if self.retries > 50:
                    raise
                time.sleep(min(30, 2 * self.retries))
                self.fh.close()
                self.fh = self._open(self.sources[self.idx], self.pos_in_part)
                continue
            if n:
                self._hash.update(memoryview(b)[:n])
                self.pos_in_part += n
                self.total += n
                return n
            self._next()
        return 0

    def drain(self) -> int:
        """Read to the end of the last part so every part's SHA-256 is complete.

        tarfile stops at the end-of-archive marker; the zero padding after it is
        still part of the published file and must be hashed.
        """
        buf = bytearray(1 << 20)
        n = 0
        while True:
            k = self.readinto(buf)
            if not k:
                return n
            n += k

    def close_all(self):
        """Close early (partial fetch). The current part's hash is then partial."""
        if self.fh is not None:
            self.fh.close()
            self.fh = None


class RangeReader(io.RawIOBase):
    """Stream bytes [start, end) of a split tar (list of (url, size)) as one file."""

    def __init__(self, parts: list[tuple[str, int]], start: int, end: int | None = None, timeout: int = 120):
        self.parts = parts
        self.bounds = np.cumsum([0] + [s for _, s in parts])
        self.pos = start
        self.end = int(self.bounds[-1]) if end is None else end
        self.timeout = timeout
        self.fh = None
        self.cur = -1
        self.total = 0
        self.retries = 0

    def readable(self) -> bool:
        return True

    def _open(self):
        k = int(np.searchsorted(self.bounds, self.pos, side="right") - 1)
        off = self.pos - int(self.bounds[k])
        stop = min(self.end, int(self.bounds[k + 1])) - int(self.bounds[k]) - 1
        req = Request(self.parts[k][0], headers={"User-Agent": USER_AGENT, "Range": f"bytes={off}-{stop}"})
        self.fh = urlopen(req, timeout=self.timeout)
        if off:
            _check_range(self.fh, off)
        self.cur = k

    def readinto(self, b) -> int:
        while self.pos < self.end:
            if self.fh is None:
                self._open()
            try:
                n = self.fh.readinto(b)
            except (OSError, TimeoutError):
                self.retries += 1
                if self.retries > 50:
                    raise
                time.sleep(min(30, 2 * self.retries))
                self.fh.close()
                self.fh = None
                continue
            if n:
                self.pos += n
                self.total += n
                return n
            self.fh.close()
            self.fh = None  # part boundary reached; next _open moves to the next part
        return 0

    def close_all(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def _peek_headers(parts: list[tuple[str, int]], offset: int, n: int = 1 << 17) -> list[tuple[int, str, int]]:
    """Fetch n bytes at a global offset and return the tar headers found (pos, name, size)."""
    bounds = np.cumsum([0] + [s for _, s in parts])
    k = int(np.searchsorted(bounds, offset, side="right") - 1)
    local = offset - int(bounds[k])
    stop = min(local + n, parts[k][1]) - 1
    req = Request(parts[k][0], headers={"User-Agent": USER_AGENT, "Range": f"bytes={local}-{stop}"})
    b = urlopen(req, timeout=60).read()
    out = []
    i = (-offset) % 512
    while i + 512 <= len(b):
        h = b[i : i + 512]
        if h[257:262] == b"ustar":
            name = h[:100].split(b"\0")[0].decode(errors="replace")
            size = int(h[124:136].split(b"\0")[0].strip() or b"0", 8)
            out.append((offset + i, name, size))
            i += 512 + ((size + 511) // 512) * 512
        else:
            i += 512
    return out


def locate_heatmap(parts: list[tuple[str, int]], step: int = 25_000_000) -> tuple[int, int, int]:
    """Find a header offset shortly before the heatmap block and one after it.

    The tar member order differs between days (heatmap first on 2025-02-12, near the
    end on 2026-01-14), so the block is located by reading headers with small HTTP
    Range requests. Returns (start_offset, end_offset, n_requests). Windows that fall
    inside the (large) heatmap members contain no header at all.
    """
    total = sum(s for _, s in parts)
    samples = []
    for off in range(0, total, step):
        samples.append((off, _peek_headers(parts, off)))
    samples.append((total - (1 << 17), _peek_headers(parts, total - (1 << 17))))
    kinds = []
    for off, hs in samples:
        names = [h[1] for h in hs]
        if any("/heatmap/" in x for x in names):
            kinds.append("heat")
        elif not names:
            kinds.append("none")
        elif any("trace_full_" in x for x in names):
            kinds.append("trace")
        else:
            kinds.append("other")
    # the heatmap block is the longest run of consecutive samples that land inside
    # heatmap members ("none": no header in the window) or on a heatmap header;
    # isolated "none" samples are large trace files
    best, cur = (0, -1), None
    for i, k in enumerate(kinds + ["end"]):
        if k in ("heat", "none"):
            cur = (cur[0], i) if cur else (i, i)
        else:
            if cur and cur[1] - cur[0] > best[1] - best[0]:
                best = cur
            cur = None
    first, last = best
    # a day's heatmap is ~0.5-0.8 GB of large members; shorter headerless runs are
    # just big trace files (some releases, e.g. 2025-10-15, carry no heatmap at all)
    if last < first or (last - first + 1) * step < 150_000_000:
        raise RuntimeError("heatmap block not found (release may not contain heatmap/)")
    start = 0
    for i in range(first - 1, -1, -1):
        if samples[i][1]:
            start = samples[i][1][-1][0]
            break
    end = total
    for i in range(last + 1, len(samples)):
        if samples[i][1]:
            end = samples[i][1][0][0]
            break
    return start, end, len(samples)


# --------------------------------------------------------------------------- filters


@dataclass
class FilterConfig:
    radius_m: float = 6000.0
    alt_margin_ft: float = 3000.0  # keep airborne samples below elevation + margin (baro)
    airports: dict[str, Airport] = field(default_factory=dict)

    def box(self, a: Airport) -> tuple[float, float, float, float]:
        dlat = self.radius_m / 111_000.0
        dlon = dlat / math.cos(math.radians(a.lat))
        return a.lat - dlat, a.lat + dlat, a.lon - dlon, a.lon + dlon


def _prefixes(lo: float, hi: float) -> list[str]:
    """Decimal prefixes (one decimal) of all numbers printed by readsb in [lo, hi]."""
    out = set()
    for v in np.arange(lo, hi + 1e-4, 0.001):
        s = "-" if v < 0 else ""
        av = abs(v)
        out.add(f"{s}{int(av)}.{round(av * 1e6) // 100000 % 10}")
    return sorted(out)


def build_prefilter(cfg: FilterConfig) -> re.Pattern:
    """Regex that matches a trace point ``[secs,lat,lon,`` inside any airport box.

    It is a coarse (0.1 degree) prefilter; exact distances are applied after parsing.
    """
    alts = []
    for a in cfg.airports.values():
        la0, la1, lo0, lo1 = cfg.box(a)
        lat_alt = "|".join(re.escape(p) for p in _prefixes(la0, la1))
        lon_alt = "|".join(re.escape(p) for p in _prefixes(lo0, lo1))
        alts.append(f"(?:{lat_alt})\\d*,(?:{lon_alt})\\d*,")
    return re.compile(("\\[[0-9.]+,(?:" + "|".join(alts) + ")").encode())


TRACE_COLS = (
    "airport", "hex", "reg", "type", "ts", "lat", "lon", "on_ground", "alt_ft",
    "gs_kt", "track", "flags", "src", "callsign", "dist_m",
)


def filter_trace(raw: bytes, cfg: FilterConfig, prefilter: re.Pattern, stats: dict) -> list[tuple]:
    """Filter one gzip trace_full JSON to low/on-ground samples near airports."""
    try:
        data = zlib.decompress(raw, 31) if raw[:2] == b"\x1f\x8b" else raw
    except zlib.error:
        stats["bad_members"] = stats.get("bad_members", 0) + 1
        return []
    stats["json_bytes"] += len(data)
    if prefilter.search(data) is None:
        return []
    stats["parsed"] += 1
    try:
        obj = json.loads(data)
    except ValueError:
        stats["bad_members"] = stats.get("bad_members", 0) + 1
        return []
    base = float(obj["timestamp"])
    hexid = obj.get("icao")
    reg = obj.get("r")
    typ = obj.get("t")
    rows = []
    callsign = None
    boxes = [(a, cfg.box(a)) for a in cfg.airports.values()]
    for p in obj.get("trace", ()):
        det = p[8] if len(p) > 8 else None
        if det and isinstance(det, dict) and det.get("flight"):
            callsign = det["flight"].strip().upper() or callsign
        lat, lon = p[1], p[2]
        if lat is None or lon is None:
            continue
        for a, (la0, la1, lo0, lo1) in boxes:
            if la0 <= lat <= la1 and lo0 <= lon <= lo1:
                alt = p[3]
                on_ground = alt == "ground"
                if not on_ground and (alt is None or alt > a.elev_ft + cfg.alt_margin_ft):
                    break
                d = float(haversine_m(a.lat, a.lon, lat, lon))
                if d > cfg.radius_m:
                    break
                rows.append((
                    a.icao, hexid, reg, typ, base + p[0], lat, lon, on_ground,
                    None if on_ground else (float(alt) if alt is not None else None),
                    None if p[4] is None else float(p[4]),
                    None if p[5] is None else float(p[5]),
                    int(p[6]) if p[6] is not None else None,
                    p[9] if len(p) > 9 else None,
                    callsign, d,
                ))
                break
    return rows


def parse_heatmap(raw: bytes, cfg: FilterConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse one heatmap half-hour file into (positions near airports, callsigns)."""
    data = zlib.decompress(raw, 31)
    arr = np.frombuffer(data, dtype=HEAT_DTYPE)
    # the first num_slices records are an index; slice headers carry the magic hex
    is_slice = arr["hex"] == HEAT_SLICE_MAGIC
    slice_pos = np.flatnonzero(is_slice)
    # the index block repeats offsets, the real headers are the ones followed by data;
    # a header's timestamp is lat<<32 | (uint32)lon in ms
    ts_ms = (arr["lat"].astype(np.int64) << 32) | (arr["lon"].astype(np.int64) & 0xFFFFFFFF)
    slice_id = np.cumsum(is_slice) - 1
    first_hdr = slice_pos[0] if len(slice_pos) else len(arr)
    body = np.arange(len(arr)) > first_hdr
    hdr_ts = np.where(is_slice, ts_ms, 0)
    slice_ts = hdr_ts[slice_pos][np.clip(slice_id, 0, None)] / 1000.0
    interval_ms = int(arr["alt"][slice_pos[0]]) if len(slice_pos) else -1

    is_cs = body & ~is_slice & (arr["lat"] >= (1 << 30))
    is_pos = body & ~is_slice & ~is_cs
    hex24 = arr["hex"] & 0x1FFFFFF  # keep the non-ICAO flag bit 24
    lat = arr["lat"] / 1e6
    lon = arr["lon"] / 1e6
    keep = np.zeros(len(arr), dtype=bool)
    ap = np.empty(len(arr), dtype=object)
    for a in cfg.airports.values():
        la0, la1, lo0, lo1 = cfg.box(a)
        m = is_pos & (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        if not m.any():
            continue
        alt = arr["alt"]
        low = (alt == -123) | ((alt != -124) & (alt * 25.0 <= a.elev_ft + cfg.alt_margin_ft))
        m &= low
        d = haversine_m(a.lat, a.lon, lat, lon)
        m &= d <= cfg.radius_m
        keep |= m
        ap[m] = a.icao
    idx = np.flatnonzero(keep)
    pos = pl.DataFrame({
        "airport": ap[idx].astype(str),
        "hexnum": hex24[idx].astype(np.int64),
        "ts": slice_ts[idx],
        "lat": lat[idx],
        "lon": lon[idx],
        "on_ground": arr["alt"][idx] == -123,
        "alt_ft": np.where(arr["alt"][idx] <= -123, np.nan, arr["alt"][idx] * 25.0),
        "gs_kt": np.where(arr["gs"][idx] < 0, np.nan, arr["gs"][idx] / 10.0),
    })
    cidx = np.flatnonzero(is_cs & np.isin(hex24, np.unique(hex24[idx])))
    raw8 = arr[cidx].tobytes()
    cs = [
        raw8[i * 16 + 8 : i * 16 + 16].split(b"\x00")[0].decode("ascii", "replace").strip().upper()
        for i in range(len(cidx))
    ]
    css = pl.DataFrame({
        "hexnum": hex24[cidx].astype(np.int64),
        "ts": slice_ts[cidx],
        "callsign": pl.Series(cs, dtype=pl.String),
    })
    pos = pos.with_columns(pl.lit(interval_ms).alias("interval_ms"))
    return pos, css


def iter_members(reader: io.RawIOBase) -> Iterator[tuple[tarfile.TarInfo, tarfile.TarFile]]:
    buf = io.BufferedReader(reader, buffer_size=4 << 20)
    with tarfile.open(fileobj=buf, mode="r|") as tf:
        for m in tf:
            yield m, tf


def process_day(
    sources: list[str],
    cfg: FilterConfig,
    want_traces: bool = True,
    want_heatmap: bool = True,
    log_every: int = 20000,
    log=print,
    max_bytes: int | None = None,
) -> dict:
    """Stream one day's tar parts and return filtered frames plus cost stats.

    ``sources`` is a list of paths/URLs (full archive), or a ``RangeReader`` for a
    partial heatmap fetch (then ``want_traces`` must be False).
    """
    reader = sources if isinstance(sources, RangeReader) else HashingReader(sources)
    prefilter = build_prefilter(cfg)
    stats = {"json_bytes": 0, "parsed": 0, "trace_files": 0, "heatmap_files": 0, "bad_members": 0}
    rows: list[tuple] = []
    heat_pos, heat_cs = [], []
    t0 = time.time()
    cpu0 = time.process_time()
    t_heat_done = None
    bytes_heat_done = None
    dirs_seen: set[str] = set()
    truncated = False
    for m, tf in iter_members(reader):
        name = m.name
        if not m.isfile():
            continue
        if "/heatmap/" in name or name.startswith("heatmap/"):
            raw = tf.extractfile(m).read()
            if want_heatmap:
                p, c = parse_heatmap(raw, cfg)
                heat_pos.append(p)
                heat_cs.append(c)
            stats["heatmap_files"] += 1
            if stats["heatmap_files"] == 48:  # one file per half hour
                t_heat_done = time.time() - t0
                bytes_heat_done = reader.total
                if not want_traces:
                    reader.close_all()
                    break
            continue
        if stats["heatmap_files"] and t_heat_done is None:
            t_heat_done = time.time() - t0
            bytes_heat_done = reader.total
            if not want_traces:
                reader.close_all()
                break
        if "trace_full_" in name:
            if not want_traces:
                continue  # partial fetch: traces before the heatmap block are skipped
            tdir = name.rsplit("/", 2)[-2]
            if max_bytes is not None and reader.total >= max_bytes and tdir not in dirs_seen:
                # byte budget reached at a directory boundary: the archive is ordered by
                # traces/<last two hex digits>/, so the complete directories read so far
                # are a random subsample of aircraft
                truncated = True
                reader.close_all()
                break
            dirs_seen.add(tdir)
            stats["trace_files"] += 1
            raw = tf.extractfile(m).read()
            rows.extend(filter_trace(raw, cfg, prefilter, stats))
            if stats["trace_files"] % log_every == 0:
                log(
                    f"  traces={stats['trace_files']} parsed={stats['parsed']} rows={len(rows)} "
                    f"read={reader.total / 1e9:.2f} GB wall={time.time() - t0:.0f}s "
                    f"cpu={time.process_time() - cpu0:.0f}s"
                )
    if want_traces and not truncated and isinstance(reader, HashingReader):
        stats["drained_bytes"] = reader.drain()
    stats["trace_dirs"] = sorted(dirs_seen)
    stats["truncated_at_bytes"] = max_bytes if truncated else None
    wall = time.time() - t0
    cpu = time.process_time() - cpu0
    traces = pl.DataFrame(rows, schema={
        "airport": pl.String, "hex": pl.String, "reg": pl.String, "type": pl.String,
        "ts": pl.Float64, "lat": pl.Float64, "lon": pl.Float64, "on_ground": pl.Boolean,
        "alt_ft": pl.Float64, "gs_kt": pl.Float64, "track": pl.Float64, "flags": pl.Int32,
        "src": pl.String, "callsign": pl.String, "dist_m": pl.Float64,
    }, orient="row")
    heat = None
    if heat_pos:
        pos = pl.concat(heat_pos)
        css = pl.concat(heat_cs).sort("hexnum", "ts")
        pos = pos.sort("hexnum", "ts").join_asof(
            css, on="ts", by="hexnum", strategy="backward"
        ).with_columns(
            pl.col("hexnum").map_elements(lambda h: format(h & 0xFFFFFF, "06x") if h < (1 << 24) else "~" + format(h & 0xFFFFFF, "06x"), return_dtype=pl.String).alias("hex")
        )
        heat = pos
    return {
        "traces": traces,
        "heatmap": heat,
        "stats": {
            **stats,
            "bytes_read": reader.total,
            "part_bytes": getattr(reader, "part_bytes", []),
            "part_sha256": getattr(reader, "part_sha", []),
            "part_secs": getattr(reader, "part_secs", []),
            "wall_s": wall,
            "cpu_s": cpu,
            "heatmap_prefix_bytes": bytes_heat_done,
            "heatmap_prefix_wall_s": t_heat_done,
            "retries": reader.retries,
            "partial": (not want_traces) or truncated,
        },
    }
