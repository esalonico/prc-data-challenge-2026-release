# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Shared HTTP helpers for open-data downloaders."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

USER_AGENT = "prc-challenge-2026-research/0.1 (open-data unmatched campaign)"


def fetch_bytes(url: str, *, data: bytes | None = None, timeout: int = 180, headers: dict[str, str] | None = None) -> bytes:
    hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        hdrs.update(headers)
    req = Request(url, data=data, headers=hdrs)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_json(url: str, *, data: bytes | None = None, timeout: int = 180) -> dict:
    raw = fetch_bytes(url, data=data, timeout=timeout, headers={"Accept": "application/json"})
    return json.loads(raw.decode("utf-8"))


def fetch_with_retries(
    url: str,
    *,
    data: bytes | None = None,
    timeout: int = 180,
    attempts: int = 4,
    sleep_s: float = 4.0,
    headers: dict[str, str] | None = None,
) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fetch_bytes(url, data=data, timeout=timeout, headers=headers)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
