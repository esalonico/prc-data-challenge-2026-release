#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Insert a GPL SPDX licence header into every Python and shell source file (dry run by default).

    python scripts/release/add_license_headers.py --root <dir>                  # dry run: list what would change
    python scripts/release/add_license_headers.py --root <dir> --apply          # write the headers
    python scripts/release/add_license_headers.py --root <dir> --check          # exit 1 if any file lacks one

It is idempotent: a file whose first
15 lines already contain ``SPDX-License-Identifier`` is left alone.

Placement rules:
* a shebang line (``#!``) stays first;
* a PEP 263 encoding cookie (``# -*- coding: ... -*-``) stays directly after it;
* the header goes next, followed by one blank line when the file continues with code or a docstring.
Comments before a module docstring do not change ``__doc__``, so docstrings keep working.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_HOLDER = "the unbelievable-snowflake team (PRC Data Challenge 2026)"
DEFAULT_SPDX = "GPL-3.0-or-later"
SUFFIXES = {".py", ".sh"}
SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "node_modules", "build", "dist"}
CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")


def header_lines(holder: str, spdx: str, year: str) -> list[str]:
    return [
        f"# SPDX-License-Identifier: {spdx}\n",
        f"# Copyright (C) {year} {holder}\n",
        "# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.\n",
    ]


def has_header(lines: list[str]) -> bool:
    return any("SPDX-License-Identifier" in ln for ln in lines[:15])


def insert(text: str, hdr: list[str]) -> str:
    lines = text.splitlines(keepends=True)
    i = 0
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
    if i < len(lines) and CODING_RE.match(lines[i]):
        i += 1
    head, rest = lines[:i], lines[i:]
    if head and not head[-1].endswith("\n"):
        head[-1] += "\n"
    sep = ["\n"] if rest and rest[0].strip() else []
    return "".join(head + hdr + sep + rest)


def candidates(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in SUFFIXES:
            continue
        if SKIP_PARTS.intersection(p.relative_to(root).parts):
            continue
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="tree to process (the repository root)")
    ap.add_argument("--holder", default=DEFAULT_HOLDER)
    ap.add_argument("--spdx", default=DEFAULT_SPDX, choices=("GPL-3.0-or-later", "GPL-3.0-only"))
    ap.add_argument("--year", default="2026")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write the headers (default: dry run)")
    mode.add_argument("--check", action="store_true", help="exit 1 if any source file lacks a header")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    hdr = header_lines(args.holder, args.spdx, args.year)
    todo, done = [], 0
    for p in candidates(root):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"skip (not UTF-8): {p.relative_to(root)}", file=sys.stderr)
            continue
        if has_header(text.splitlines(keepends=True)):
            done += 1
            continue
        todo.append(p)
        if args.apply:
            p.write_text(insert(text, hdr), encoding="utf-8")

    verb = "added" if args.apply else ("missing" if args.check else "would add")
    for p in todo[:20]:
        print(f"{verb}: {p.relative_to(root)}")
    if len(todo) > 20:
        print(f"... and {len(todo) - 20} more")
    print(f"{len(todo)} file(s) {verb}; {done} already had a header; spdx={args.spdx}")
    return 1 if (args.check and todo) else 0


if __name__ == "__main__":
    sys.exit(main())
