# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Label-free train/serve parity screen of an assembled S4 matrix (SA-GATE-TOOLS, D-081).

A team member's train/serve parity check, generalised.  For every feature column and every serve month
it compares the training distribution with the serving distribution:

* numeric columns: standardised mean shift ``(mean_serve - mean_train) / sd_train``, null-rate change,
  and the tail beyond the training support (serve rows above the training max / p99.9);
* categorical columns: the share of serve rows whose level never occurs in the training rows compared.

Pairs (``--pairs``):

* ``month`` (default): training month m vs serving month m for each ``--months`` value
  (2025 Jan/Jul vs 2026 Jan/Jul for the all-year matrix);
* ``all``: all training rows pooled vs each serving month (e.g. the D-fit training rows vs the Nov/Dec
  eval rows: a pipeline-path check).

A (column, month) **fails** when |shift| > ``--shift-sd`` (default 0.25), |Δnull| > ``--null-pp``
(default 0.01), or the unseen-level share > ``--unseen-pp`` (categoricals, default 0.01).  Columns whose
*definition* makes them fragile across frames are flagged separately (name rules + tail evidence):

* ``lookback_horizon``: gaps / ages / "since previous" lookbacks (``since``, ``_age``, ``prev``, ``last``,
  ``lag``, ``_gap``): a lookback can cross the 2025 -> 2026 or Feb–Jun gap on serving rows only;
* ``frame_count``: counts that depend on the frame size (``n_``, ``_n``, ``count``, ``cnt``, ``total``,
  ``cum``): a "within frame" count differs between a 12-month and a 2-month frame;
* ``support_fallback``: support counts of reference encodings (``support``): serving rows that fall back
  to coarse cells show up as a tail beyond the training support.

For flagged columns a tail beyond the training max on more than ``--tail-share`` of serving rows (and at
least ``--tail-min`` rows) is a failure too (``tail`` kind).

Failures matching ``--allowlist`` (JSON: ``{"columns": {name: reason}, "patterns": [{"pattern": glob,
"reason": ..., "kinds": [...]}]}``; every entry needs a written mechanism; ``kinds`` limits an entry to some
failure kinds) are reported as allowlisted.  Output: a JSON
report (``--out``) and optionally a CSV table (``--csv``).  Exit codes: 0 = pass (or ``--report-only``),
3 = unallowlisted failures with ``--enforce`` (the bake driver's "gate says stop" code), 2 = usage/input error.

Label-free by construction: the label and the forbidden departure columns (``TAXITIME_SEC_mvt``,
``BLOCK_TIME_UTC_mvt``, ``AOBT_3_flt``, ``ARVT_3_flt``) are never selected.

Examples::

  PYTHONPATH=src .venv/bin/python scripts/features/parity_screen.py --run-dir <runs>/vm_allyear_v3l_cb_plain \\
      --allowlist configs/bake/parity_allowlist_v2.json --out parity_ay3l.json --csv parity_ay3l.csv
  PYTHONPATH=src .venv/bin/python scripts/features/parity_screen.py --train <runs>/dfit_full_adsb_historder/train_matrix.parquet \\
      --serve <runs>/v4_novdec_dfit_adsb/eval_matrix.parquet --pairs all --months 11,12 --out parity_d_vs_novdec.json
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FORBIDDEN = frozenset({"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt", "label", "_additional_taxi"})
EXIT_OK, EXIT_USAGE, EXIT_GATE_STOP = 0, 2, 3
RULES = {
    "lookback_horizon": re.compile(r"(since|_age(_|$)|(^|_)prev|(^|_)last(_|$)|(^|_)lag(_|$)|_gap(_|$)|lookback|horizon)"),
    "frame_count": re.compile(r"((^|_)n(_|$)|(^|_)n_|count|(^|_)cnt(_|$)|total|(^|_)cum)"),
    "support_fallback": re.compile(r"support"),
}


def suspect_kinds(col: str) -> list[str]:
    return [k for k, rx in RULES.items() if rx.search(col.lower())]


Allow = tuple[str, str, frozenset[str] | None]  # (glob pattern, reason, fail kinds covered or None = all)


def load_allowlist(path: str | None) -> list[Allow]:
    """Allowlist entries.  ``columns``: {name: reason | {"reason", "kinds"}}; ``patterns``: [{"pattern", "reason", "kinds"}].

    ``kinds`` (optional) limits an entry to some failure kinds (shift / null / tail / unseen), so a column
    allowlisted for a genuine mean shift still fails on a new tail beyond its training support.
    """

    if not path:
        return []
    d = json.loads(Path(path).read_text())
    out: list[Allow] = []
    for k, v in d.get("columns", {}).items():
        v = {"reason": v} if isinstance(v, str) else v
        out.append((k, v.get("reason") or v.get("mechanism", ""), frozenset(v["kinds"]) if v.get("kinds") else None))
    for e in d.get("patterns", []):  # "mechanism" is accepted as the reason's key too (allowlist v2's STAND_mvt entry)
        out.append((e["pattern"], e.get("reason") or e.get("mechanism", ""), frozenset(e["kinds"]) if e.get("kinds") else None))
    for name, reason, _ in out:
        if not str(reason).strip():
            raise SystemExit(f"allowlist entry {name!r} has no written mechanism")
    return out


def allow_reason(col: str, kinds: list[str], entries: list[Allow]) -> str | None:
    """The reason covering *every* failure kind of this (column, month), or None."""

    reasons = []
    for kind in kinds:
        hit = next((r for p, r, ks in entries if fnmatch.fnmatchcase(col, p) and (ks is None or kind in ks)), None)
        if hit is None:
            return None
        reasons.append(hit)
    return " | ".join(dict.fromkeys(reasons))


def feature_lists(identity: Path | None, schema: dict[str, Any]) -> tuple[list[str], list[str]]:
    if identity is not None and identity.exists():
        cats, nums = json.loads(identity.read_text())["features"]
    else:
        cats = [c for c, t in schema.items() if t in (pl.String, pl.Categorical)]
        nums = [c for c, t in schema.items() if t.is_numeric() or t == pl.Boolean]
    drop = FORBIDDEN | {"month", "nm_unmatched"}
    return [c for c in cats if c not in drop and c in schema], [c for c in nums if c not in drop and c in schema]


def numeric_stats(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    xa, ya = x[~np.isnan(x)], y[~np.isnan(y)]
    out: dict[str, Any] = {"n_train": int(x.size), "n_serve": int(y.size),
                           "null_train": float(np.isnan(x).mean()) if x.size else None, "null_serve": float(np.isnan(y).mean()) if y.size else None}
    out["null_delta"] = (out["null_serve"] - out["null_train"]) if x.size and y.size else None
    if xa.size and ya.size:
        mt, ms, sd = float(xa.mean()), float(ya.mean()), float(xa.std(ddof=1)) if xa.size > 1 else 0.0
        tmax, p999 = float(xa.max()), float(np.quantile(xa, 0.999))
        out.update(mean_train=mt, mean_serve=ms, sd_train=sd, shift=(ms - mt) / sd if sd > 0 else (0.0 if ms == mt else None),
                   max_train=tmax, max_serve=float(ya.max()), p999_train=p999,
                   n_serve_above_train_max=int((ya > tmax).sum()), share_serve_above_train_max=float((ya > tmax).mean()),
                   share_serve_above_train_p999=float((ya > p999).mean()))
    else:
        out.update(mean_train=None, mean_serve=None, sd_train=None, shift=None)
    return out


def cat_stats(x: pl.Series, y: pl.Series) -> dict[str, Any]:
    seen = set(x.drop_nulls().cast(pl.String).unique().to_list())
    ys = y.cast(pl.String)
    unseen = ys.is_not_null() & ~ys.is_in(list(seen))
    return {"n_train": x.len(), "n_serve": y.len(), "null_train": float(x.null_count() / max(x.len(), 1)), "null_serve": float(y.null_count() / max(y.len(), 1)),
            "null_delta": float(y.null_count() / max(y.len(), 1) - x.null_count() / max(x.len(), 1)),
            "levels_train": len(seen), "unseen_share": float(unseen.mean()) if y.len() else 0.0, "n_unseen": int(unseen.sum())}


def screen(train: pl.LazyFrame, serve: pl.LazyFrame, cats: list[str], nums: list[str], months: list[int], pairs: str, route: str,
           thr: dict[str, float]) -> list[dict[str, Any]]:
    cols = list(dict.fromkeys(["month", "nm_unmatched", *cats, *nums]))
    bad = FORBIDDEN.intersection(cols)
    assert not bad, f"forbidden columns selected: {bad}"

    def rows(lf: pl.LazyFrame) -> pl.LazyFrame:
        lf = lf.select(cols)
        if route == "matched":
            lf = lf.filter(pl.col("nm_unmatched") == 0)
        elif route == "unmatched":
            lf = lf.filter(pl.col("nm_unmatched") != 0)
        return lf

    out = []
    tr_all = rows(train).collect() if pairs == "all" else None
    for m in months:
        a = tr_all if pairs == "all" else rows(train).filter(pl.col("month") == m).collect()
        b = rows(serve).filter(pl.col("month") == m).collect()
        if b.height == 0 or a.height == 0:
            raise SystemExit(f"month {m}: {a.height} training rows vs {b.height} serving rows")
        for c in nums:
            x = a[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
            y = b[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
            st = {"column": c, "month": m, "type": "numeric", **numeric_stats(x, y), "suspect": suspect_kinds(c)}
            kinds = []
            if st.get("shift") is not None and abs(st["shift"]) > thr["shift_sd"]:
                kinds.append("shift")
            if st.get("null_delta") is not None and abs(st["null_delta"]) > thr["null_pp"]:
                kinds.append("null")
            if st["suspect"] and st.get("share_serve_above_train_max") is not None and st["share_serve_above_train_max"] > thr["tail_share"] \
                    and st["n_serve_above_train_max"] >= thr["tail_min"]:
                kinds.append("tail")
            st["fail_kinds"] = kinds
            out.append(st)
        for c in cats:
            st = {"column": c, "month": m, "type": "categorical", **cat_stats(a[c], b[c]), "suspect": suspect_kinds(c)}
            kinds = []
            if abs(st["null_delta"]) > thr["null_pp"]:
                kinds.append("null")
            if st["unseen_share"] > thr["unseen_pp"]:
                kinds.append("unseen")
            st["fail_kinds"] = kinds
            out.append(st)
    return out


def resolve_inputs(a: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    if a.run_dir:
        run = Path(a.run_dir)
        return run / "train_matrix.parquet", run / "eval_matrix.parquet", run / "matrix_identity.json"
    if not (a.train and a.serve):
        raise SystemExit("give --run-dir or both --train and --serve")
    tr, sv = Path(a.train), Path(a.serve)
    ident = Path(a.identity) if a.identity else (tr.parent / "matrix_identity.json")
    return tr, sv, ident


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="run dir with train_matrix.parquet, eval_matrix.parquet, matrix_identity.json")
    p.add_argument("--train")
    p.add_argument("--serve")
    p.add_argument("--identity", help="matrix_identity.json for the feature lists (default: next to --train)")
    p.add_argument("--months", default="1,7")
    p.add_argument("--pairs", choices=("month", "all"), default="month")
    p.add_argument("--route", choices=("matched", "all", "unmatched"), default="matched")
    p.add_argument("--columns", help="comma list or glob patterns restricting the screened columns")
    p.add_argument("--shift-sd", type=float, default=0.25)
    p.add_argument("--null-pp", type=float, default=0.01)
    p.add_argument("--unseen-pp", type=float, default=0.01)
    p.add_argument("--tail-share", type=float, default=0.001)
    p.add_argument("--tail-min", type=int, default=10)
    p.add_argument("--allowlist")
    p.add_argument("--out", required=True)
    p.add_argument("--csv")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--enforce", action="store_true", help="exit 3 when unallowlisted failures remain")
    g.add_argument("--report-only", action="store_true")
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    tr_path, sv_path, ident = resolve_inputs(a)
    for q in (tr_path, sv_path):
        if not q.exists():
            print(f"missing input {q}", file=sys.stderr)
            return EXIT_USAGE
    schema = dict(pl.read_parquet_schema(tr_path))
    cats, nums = feature_lists(ident, schema)
    sv_schema = dict(pl.read_parquet_schema(sv_path))
    missing = [c for c in [*cats, *nums] if c not in sv_schema]
    if missing:
        print(f"serving matrix lacks {len(missing)} feature columns: {missing[:10]}", file=sys.stderr)
        return EXIT_USAGE
    if a.columns:
        pats = [x for x in a.columns.split(",") if x]
        cats = [c for c in cats if any(fnmatch.fnmatchcase(c, q) for q in pats)]
        nums = [c for c in nums if any(fnmatch.fnmatchcase(c, q) for q in pats)]
    months = [int(x) for x in a.months.split(",") if x]
    thr = {"shift_sd": a.shift_sd, "null_pp": a.null_pp, "unseen_pp": a.unseen_pp, "tail_share": a.tail_share, "tail_min": a.tail_min}
    rows = screen(pl.scan_parquet(tr_path), pl.scan_parquet(sv_path), cats, nums, months, a.pairs, a.route, thr)
    entries = load_allowlist(a.allowlist)
    failures, allowed = [], []
    for r in rows:
        if not r["fail_kinds"]:
            continue
        reason = allow_reason(r["column"], r["fail_kinds"], entries)
        item = {k: r.get(k) for k in ("column", "month", "fail_kinds", "shift", "null_delta", "unseen_share", "share_serve_above_train_max",
                                      "n_serve_above_train_max", "max_train", "max_serve", "suspect")}
        if reason:
            allowed.append({**item, "allow_reason": reason})
        else:
            failures.append(item)
    failures.sort(key=lambda r: -abs(r.get("shift") or 0.0))
    suspects: dict[str, list[str]] = {k: sorted({r["column"] for r in rows if k in r["suspect"]}) for k in RULES}
    report = {
        "tool": "scripts/features/parity_screen.py", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"train": str(tr_path), "serve": str(sv_path), "identity": str(ident) if ident else None, "allowlist": a.allowlist},
        "pairs": a.pairs, "months": months, "route": a.route, "thresholds": thr,
        "n_columns": {"numeric": len(nums), "categorical": len(cats)}, "labels_read": False,
        "verdict": "fail" if failures else "pass",
        "n_fail": len(failures), "n_allowlisted": len(allowed),
        "unallowlisted_failures": failures, "allowlisted_failures": allowed,
        "suspect_columns": suspects,
        "suspect_tail": [{k: r.get(k) for k in ("column", "month", "suspect", "share_serve_above_train_max", "n_serve_above_train_max", "max_train", "max_serve", "share_serve_above_train_p999")}
                         for r in rows if r["suspect"] and r["type"] == "numeric" and (r.get("n_serve_above_train_max") or 0) > 0],
        "rows": rows, "seconds": round(time.perf_counter() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=float) + "\n")
    if a.csv:
        flat = [{k: (",".join(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows]
        pl.DataFrame(flat, infer_schema_length=None).write_csv(a.csv)
    print(f"parity screen: {len(nums)} numeric + {len(cats)} categorical columns x months {months} ({a.pairs}, {a.route}): "
          f"{len(failures)} unallowlisted failures, {len(allowed)} allowlisted -> {report['verdict'].upper()} ({a.out})")
    for f in failures[:40]:
        print(f"  FAIL {f['column']:<44} m{f['month']:>2} {','.join(f['fail_kinds']):<12} shift={f['shift'] if f['shift'] is None else round(f['shift'], 3)} "
              f"null_d={f['null_delta'] if f['null_delta'] is None else round(f['null_delta'], 4)} above_train_max={f.get('n_serve_above_train_max')}")
    if failures and a.enforce:
        return EXIT_GATE_STOP
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
