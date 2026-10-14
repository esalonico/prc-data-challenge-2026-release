# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Stream one adsb.lol globe_history day and keep only samples near the ten airports.

Feasibility check (2026-09-25). Label-free. OpenSky is not used.

Examples
--------
# stream straight from GitHub (nothing raw is written to disk)
python scripts/adsb_lol_ingest.py --day 2025-02-12

# heatmap-only partial fetch (first ~25% of the tar, 10 s positions)
python scripts/adsb_lol_ingest.py --day 2026-01-14 --heatmap-only

# filter parts already downloaded into a scratch dir (then delete them)
python scripts/adsb_lol_ingest.py --day 2025-02-12 --local-dir <raw_tmp>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from prc_challenge.adsb_lol import (
    ATTRIBUTION,
    LICENCE,
    LICENCE_URL,
    USER_AGENT,
    FilterConfig,
    RangeReader,
    load_airports,
    locate_heatmap,
    process_day,
    release_urls,
)

DEFAULT_OUT = ROOT / "artifacts" / "experiments" / "ADSB-feasibility-20260925" / "filtered"
MANIFEST = ROOT / "data" / "external" / "adsb_lol" / "manifest.json"


def release_assets(repo: str, tag: str) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    assets = [
        {"name": a["name"], "size": a["size"], "url": a["browser_download_url"],
         "github_digest": a.get("digest")}
        for a in payload["assets"]
        if ".tar." in a["name"]
    ]
    return sorted(assets, key=lambda a: a["name"])


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def update_manifest(entry: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    doc = {"source": "adsb.lol globe_history", "licence": LICENCE, "licence_url": LICENCE_URL,
           "attribution": ATTRIBUTION, "status": "feasibility only", "entries": []}
    if MANIFEST.exists():
        doc = json.loads(MANIFEST.read_text())
    doc["entries"] = [
        e for e in doc["entries"]
        if not (e["day"] == entry["day"] and e["instance"] == entry["instance"] and e["mode"] == entry["mode"])
    ] + [entry]
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="UTC day YYYY-MM-DD")
    ap.add_argument("--instance", default="prod-0", help="prod-0 | prod-1 | staging-0")
    ap.add_argument("--local-dir", type=Path, default=None, help="read already-downloaded parts")
    ap.add_argument("--heatmap-only", action="store_true", help="stop after heatmap members")
    ap.add_argument("--no-heatmap", action="store_true")
    ap.add_argument("--radius-m", type=float, default=6000.0)
    ap.add_argument("--alt-margin-ft", type=float, default=3000.0)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--assets-json", type=Path, default=None,
                    help="pre-fetched {tag: {repo, assets:[{name,size,url,digest}]}} (avoids the GitHub API limit)")
    ap.add_argument("--no-global-manifest", action="store_true",
                    help="write only the per-day manifest (safe for parallel batch runs)")
    ap.add_argument("--max-bytes", type=float, default=None,
                    help="stop at the first trace-directory boundary after this many bytes "
                         "(random aircraft subsample by ICAO24 suffix)")
    args = ap.parse_args()
    args.out_dir = Path(os.path.abspath(args.out_dir))

    repo, tag = release_urls(args.day, args.instance)
    if args.assets_json:
        index = json.loads(args.assets_json.read_text())
        if tag not in index:
            raise SystemExit(f"{tag} not in {args.assets_json}")
        repo = index[tag]["repo"]
        assets = sorted(
            ({"name": a["name"], "size": a["size"], "url": a["url"], "github_digest": a.get("digest")}
             for a in index[tag]["assets"] if ".tar." in a["name"]),
            key=lambda a: a["name"],
        )
    else:
        assets = release_assets(repo, tag)
    if not assets:
        raise SystemExit(f"no tar parts for {repo}@{tag}")
    if args.local_dir:
        sources = [str(args.local_dir / a["name"]) for a in assets]
        if args.max_bytes:
            # subsample from a partially downloaded first part (curl stopped early)
            sources = [s for s in sources if Path(s).exists()]
        else:
            for s, a in zip(sources, assets):
                if Path(s).stat().st_size != a["size"]:
                    raise SystemExit(f"{s} size != release size {a['size']}")
    else:
        sources = [a["url"] for a in assets]
    cfg = FilterConfig(
        radius_m=args.radius_m,
        alt_margin_ft=args.alt_margin_ft,
        airports=load_airports(ROOT / "data" / "external" / "ourairports" / "airports.csv"),
    )
    started = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"{tag}: {len(assets)} parts, {sum(a['size'] for a in assets) / 1e9:.2f} GB, "
          f"mode={'heatmap-only' if args.heatmap_only else ('subsample<=' + str(int(args.max_bytes)) + 'B' if args.max_bytes else 'full')} source={'local' if args.local_dir else 'stream'}",
          flush=True)
    located = None
    if args.heatmap_only and not args.local_dir:
        t_locate = time.time()
        parts = [(a["url"], a["size"]) for a in assets]
        start, end, n_requests = locate_heatmap(parts)
        located = {"start": start, "end": end, "n_requests": n_requests, "locate_s": time.time() - t_locate}
        print(f"heatmap block within [{start}, {end}) after {n_requests} range requests", flush=True)
        sources = RangeReader(parts, start, end)
    res = process_day(
        sources, cfg, want_traces=not args.heatmap_only, want_heatmap=not args.no_heatmap,
        log=lambda s: print(s, flush=True), max_bytes=int(args.max_bytes) if args.max_bytes else None,
    )
    if located:
        res["stats"]["range_search"] = located
    if args.heatmap_only and res["stats"]["heatmap_files"] == 0:
        raise SystemExit("no heatmap members found in the fetched range")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"adsb_lol_{args.day}_{args.instance}"
    t_write = time.time()
    outputs = {}
    if not args.heatmap_only and res["traces"].height:
        p = args.out_dir / f"{stem}_traces.parquet"
        res["traces"].write_parquet(p, compression="zstd")
        outputs["traces"] = {"path": rel(p), "rows": res["traces"].height,
                             "bytes": p.stat().st_size}
    if res["heatmap"] is not None:
        p = args.out_dir / f"{stem}_heatmap.parquet"
        res["heatmap"].write_parquet(p, compression="zstd")
        outputs["heatmap"] = {"path": rel(p), "rows": res["heatmap"].height,
                              "bytes": p.stat().st_size}
    st = res["stats"]
    st["write_s"] = time.time() - t_write
    full_parts = [i for i in range(len(st["part_sha256"])) if st["part_bytes"][i] == assets[i]["size"]]
    entry = {
        "day": args.day,
        "instance": args.instance,
        "repo": repo,
        "tag": tag,
        "mode": "heatmap-only" if args.heatmap_only else ("subsample" if args.max_bytes else "full"),
        "source": "local" if args.local_dir else "stream",
        "retrieved_utc": started,
        "parts": [
            {**a, "sha256": st["part_sha256"][i] if i in full_parts else None,
             "bytes_read": st["part_bytes"][i] if i < len(st["part_bytes"]) else 0,
             "digest_match": (a.get("github_digest") == f"sha256:{st['part_sha256'][i]}")
             if (i in full_parts and a.get("github_digest")) else None}
            for i, a in enumerate(assets)
        ],
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "filter": {"radius_m": cfg.radius_m, "alt_margin_ft": cfg.alt_margin_ft,
                   "airports": sorted(cfg.airports)},
        "outputs": outputs,
        "stats": {k: v for k, v in st.items() if k not in ("part_sha256",)},
    }
    if not args.no_global_manifest:
        update_manifest(entry)
    (args.out_dir / f"{stem}_{entry['mode']}_manifest.json").write_text(json.dumps(entry, indent=2) + "\n")
    print(json.dumps({"outputs": outputs, "wall_s": round(st["wall_s"], 1), "cpu_s": round(st["cpu_s"], 1),
                      "bytes_read": st["bytes_read"], "parsed": st["parsed"],
                      "trace_files": st["trace_files"], "json_bytes": st["json_bytes"],
                      "heatmap_prefix_bytes": st["heatmap_prefix_bytes"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
