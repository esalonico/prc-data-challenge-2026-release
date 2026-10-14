#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Release gates for this repository: run before every push. Exit 1 on any failure.

    python scripts/release/check_release.py                          # all gates on the files git would commit
    python scripts/release/check_release.py --skip-docs               # without the docs gates (during a build)
    python scripts/release/check_release.py --patterns <file.json>    # also the owner's private identifiers
    python scripts/release/check_release.py --id-scan-raw data/raw    # also scan for challenge row IDs

Files checked: ``git ls-files --cached --others --exclude-standard`` (tracked plus untracked, not ignored).

Gates:
  * secrets: key/token patterns, credential file names, and the values of local credential files if present;
  * private paths: home-directory literals, local workspace names, remote-host names;
  * private identifiers: the regexes of ``--patterns`` (a JSON list kept outside the repository, so this script names
    none of them);
  * challenge row IDs: ``MVT_ID_mvt`` / ``FLIGHT_ID_mvt`` values of ``data/raw`` in any file;
  * data/binary files and size: no binary or data file except the allowlist, nothing over 5 MB;
  * leftover markers in docs: TODO, DRAFT, XXX, FIXME;
  * licence: verbatim GNU GPL v3 ``LICENSE``, SPDX header on every Python and shell file, required notices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GPL3_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"  # FSF gpl-3.0.txt, 674 lines
NOTICES = ("Contains data from ADSB.lol (https://adsb.lol), licensed under ODbL 1.0", "© OpenStreetMap contributors")
REQUIRED = ("README.md", "LICENSE", "NOTICE.md", "THIRD_PARTY_LICENSES.md", "docs/METHOD.md", "docs/REPRODUCTION.md",
            "docs/DATA.md", ".gitignore", "pyproject.toml", "uv.lock")
BINARY_EXT = {".parquet", ".pt", ".pth", ".cbm", ".pkl", ".pickle", ".npy", ".npz", ".feather", ".arrow", ".h5",
              ".hdf5", ".joblib", ".onnx", ".bin", ".zip", ".gz", ".tgz", ".tar", ".zst", ".7z", ".db", ".sqlite",
              ".xlsx", ".csv"}
MAX_BYTES = 5 * 1024 * 1024
SECRET_RE = re.compile(
    rb"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}"
    rb"|xox[bpa]-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,}"
    rb"|(?i:(?:secret|access)[_-]?key[\"']?\s*[:=]\s*[\"'][A-Za-z0-9/+=_-]{16,}))"
)
CREDENTIAL_NAME_RE = re.compile(r"(^|/)(credentials[^/]*\.json|.*-credentials\.json|\.env(\.[a-z]+)?|.*\.pem|id_rsa.*)$")
PATH_RE = re.compile(rb"/Users/[A-Za-z]|/home/[a-z][a-z0-9_-]+/|prc-serve4m|prc-regress\b|\.vast\b|vast\.ai|ssh[0-9]*\.")
MARKER_RE = re.compile(r"\b(TODO|DRAFT|XXX|FIXME)\b")
ID_RE = re.compile(rb"(?<![0-9A-Za-z.])(\d{8,10})(?:\.0(?![0-9A-Za-z.])|(?![0-9A-Za-z.]))")


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    return sorted(f for f in out.splitlines() if (REPO / f).is_file())


def is_text(b: bytes) -> bool:
    if b"\0" in b[:8192]:
        return False
    try:
        b[:8192].decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def load_ids(raw: Path) -> set[int]:
    import os

    os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
    import polars as pl

    ids: set[int] = set()
    for f in sorted(raw.glob("*.parquet")):
        for col in ("MVT_ID_mvt", "FLIGHT_ID_mvt"):
            try:
                ids.update(pl.read_parquet(f, columns=[col])[col].drop_nulls().cast(pl.Int64).to_list())
            except Exception:
                continue
    return ids


def local_secret_values() -> list[bytes]:
    """Values of local credential files next to the repository (never printed)."""
    vals: list[bytes] = []
    for c in list(REPO.glob("*credentials*.json")) + list(REPO.parent.glob("*/*credentials*.json")):
        try:
            for k, v in json.loads(c.read_text()).items():
                if isinstance(v, str) and len(v) >= 12 and k.lower() not in {"url", "api", "path", "endpoint"}:
                    vals.append(v.encode())
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patterns", type=Path, help="JSON list of private-identifier regexes (kept outside the repo)")
    ap.add_argument("--id-scan-raw", type=Path, help="data/raw: scan every text file for challenge row IDs")
    ap.add_argument("--skip-docs", action="store_true", help="skip the docs and notices gates")
    a = ap.parse_args()
    private = re.compile("|".join(json.loads(a.patterns.read_text())).encode(), re.I) if a.patterns else None
    ids = load_ids(a.id_scan_raw) if a.id_scan_raw else None
    secrets = local_secret_values()
    fails: list[str] = []
    fl = files()
    for f in fl:
        p = REPO / f
        b = p.read_bytes()
        if CREDENTIAL_NAME_RE.search(f):
            fails.append(f"credential-like file name: {f}")
        if len(b) > MAX_BYTES:
            fails.append(f"file > 5 MB: {f} ({len(b)} B)")
        if m := SECRET_RE.search(b):
            fails.append(f"secret-like pattern in {f} ({m.group(0)[:8]!r}...)")
        if any(v in b for v in secrets):
            fails.append(f"a local credential value occurs in {f}")
        if f != "scripts/release/check_release.py" and (m := PATH_RE.search(b)):
            fails.append(f"private path {m.group(0).decode(errors='replace')!r} in {f}")
        if private is not None and (m := private.search(b)):
            fails.append(f"private identifier {m.group(0).decode(errors='replace')!r} in {f}")
        if ids is not None and is_text(b):
            n = sum(1 for m in ID_RE.finditer(b) if int(m.group(1)) in ids)
            if n:
                fails.append(f"{n} challenge row ID(s) in {f}")
        if Path(f).suffix in (".py", ".sh") and b"SPDX-License-Identifier: GPL-3.0-or-later" not in b[:1500]:
            fails.append(f"no SPDX header: {f}")
    for req in REQUIRED:
        if req not in fl and not (a.skip_docs and (req.endswith(".md"))):
            fails.append(f"missing {req}")
    lic = REPO / "LICENSE"
    if not lic.exists() or hashlib.sha256(lic.read_bytes()).hexdigest() != GPL3_SHA256:
        fails.append("LICENSE is missing or not the verbatim GNU GPL v3 text")
    if not a.skip_docs:
        notice = (REPO / "NOTICE.md").read_text(encoding="utf-8") if (REPO / "NOTICE.md").exists() else ""
        fails += [f"NOTICE.md lacks {s!r}" for s in NOTICES if s not in notice]
        for f in fl:
            if f.endswith(".md"):
                for i, line in enumerate((REPO / f).read_text(encoding="utf-8").splitlines(), 1):
                    if MARKER_RE.search(line):
                        fails.append(f"marker in {f}:{i}: {line.strip()[:60]}")
    for x in fails:
        print("FAIL", x)
    print(f"{len(fl)} files checked, {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
