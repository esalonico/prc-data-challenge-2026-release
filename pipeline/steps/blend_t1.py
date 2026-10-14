# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Blend (SA-BLEND method d, cell x leg weights) + T1 tail for a set of 14 ranking legs (final phase serve path).

1. ``blend_matched_legs.py --weights-json`` with a copy of ``weights_d.json`` whose ``cell_matrix`` points at the given
   label-free matrix (AIRPORT + adsb_coverage of the rows being served) -> ``<out>/<tag>_blend_matched_preT1.parquet``.
2. ``build_matched_tail_ranking.t1_on_blend10`` (the stored all-year p_long / mu_long boosters, a = 0.35) with its
   ``ALLYEAR`` run pointed at ``--t1-run`` (the contiguous all-year matrix whose eval rows are the served rows)
   -> ``<out>/<tag>_blend_matched_T1.parquet``.

Legs are given as ``--leg NAME=PATH`` with the 14 names of the weights file (any order).
``--check-pre`` / ``--check-t1``: stored blend files to compare with.

Run from the repository root of the checkout that holds the T1 boosters (PYTHONPATH=src, PRC_MAIN_REPO set).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(os.environ["PRC_MAIN_REPO"])
sys.path.insert(0, str(REPO / "scripts" / "compose"))
sys.path.insert(0, str(REPO / "src"))


def compare(new: Path, stored: Path) -> dict:
    a = pl.read_parquet(new).select("MVT_ID_mvt", pl.col("prediction").alias("new"))
    b = pl.read_parquet(stored).select("MVT_ID_mvt", pl.col("prediction").alias("stored"))
    j = a.join(b, on="MVT_ID_mvt", how="inner", validate="1:1")
    d = np.abs(j["new"].to_numpy() - j["stored"].to_numpy())
    return {"stored": str(stored), "n": j.height, "n_new": a.height, "n_stored": b.height, "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()), "n_gt_1e-6": int((d > 1e-6).sum()), "n_gt_0p01": int((d > 0.01).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="weights_d.json (method d)")
    ap.add_argument("--cell-matrix", required=True)
    ap.add_argument("--leg", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--t1-run", required=True, help="run dir of the contiguous all-year matrix (train + eval)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--check-pre")
    ap.add_argument("--check-t1")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(a.weights).read_text())
    spec["cell_matrix"] = str(Path(a.cell_matrix).resolve())
    wpath = out / f"weights_d_{a.tag}.json"
    wpath.write_text(json.dumps(spec, indent=1) + "\n")
    given = dict(x.split("=", 1) for x in a.leg)
    if set(given) != set(spec["legs"]):
        raise SystemExit(f"legs {sorted(given)} != weights legs {sorted(spec['legs'])}")
    legs = [Path(given[n]) for n in spec["legs"]]
    pre = out / f"{a.tag}_blend_matched_preT1.parquet"
    cmd = [sys.executable, str(REPO / "scripts" / "compose" / "blend_matched_legs.py"), "--weights-json", str(wpath), "--out", str(pre)]
    for p in legs:
        cmd += ["--leg", str(p)]
    subprocess.run(cmd, check=True, cwd=REPO)
    import build_matched_tail_ranking as bmt

    bmt.ALLYEAR = Path(a.t1_run).resolve()
    bmt.OUT.mkdir(parents=True, exist_ok=True)
    (bmt.R.TRACKED / "ranking").mkdir(parents=True, exist_ok=True)
    name = f"{a.tag}_blend_matched_T1.parquet"
    bmt.t1_on_blend10(a.threads, blend10=pre, leg_files=legs, out_name=name, weights_json=wpath)
    t1 = out / name
    shutil.move(str(bmt.OUT / name), t1)
    man_src = bmt.OUT / (Path(name).stem + ".manifest.json")
    rep = {"weights": str(wpath), "legs": dict(zip(spec["legs"], map(str, legs))), "t1_run": str(bmt.ALLYEAR),
           "pre_t1": str(pre), "t1": str(t1)}
    if a.check_pre:
        rep["check_pre"] = compare(pre, Path(a.check_pre))
    if a.check_t1:
        rep["check_t1"] = compare(t1, Path(a.check_t1))
    (out / f"{a.tag}_blend_t1_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep.get(k) for k in ("check_pre", "check_t1")}, indent=1))
    if man_src.exists():
        shutil.move(str(man_src), out / man_src.name)


if __name__ == "__main__":
    main()
