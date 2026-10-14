# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Blend matched-row legs from S4-trainer ranking files into one matched prediction file.

Each leg is a trainer ranking parquet (``MVT_ID_mvt``, ``s4_raw_prediction``,
``prediction``, ``gate_e17``, ``gate_e18``, ``nm_unmatched`` …). The blend is a
fixed weighted mean of ``s4_raw_prediction`` on matched rows; rows where the
first leg's E17/E18 gate fired keep that leg's gated ``prediction`` so the
LIRF delay-as-taxi rule is preserved; the result is clipped at 0. Weights are
fixed in advance (pre-registered), never fitted here.

``--weights-json PATH`` (SA-BLEND, D-087; optional, the default output is unchanged): the weights come
from a pre-registered weights file instead of ``path:weight`` specs.  The file names the legs in order
(``"legs"``: one name per ``--leg``, same order) and either

* ``"weights"``: ``{leg name: weight}`` (one global vector, >= 0, summing to 1), or
* ``"cells"``: ``{cell: {leg name: weight}}`` plus ``"default"`` (the vector for unlisted cells),
  ``"cell_matrix"`` (a label-free parquet with ``MVT_ID_mvt``, ``AIRPORT``, ``adsb_coverage``) and
  ``"rich_airports"``; a row's cell is ``<group>|<coverage class>`` with group ``rich7`` for the rich
  airports, else the airport code, and the class of ``prc_challenge.gate_stats.coverage_class``;
* ``"stacker"`` (final bake stage O, a separately registered arm): ``"default"`` is the method-c vector (the
  stacker's base blend) and ``"stacker"`` = ``{"model": LightGBM model file, "airports": [...]}``; the raw blend is
  ``P @ default + booster(P, airport code, coverage-class code, local_hour)`` (``cell_matrix`` must also hold
  ``local_hour``).  ``"model"`` may be relative to the weights file's directory.

Output: parquet with ``MVT_ID_mvt`` and ``prediction`` for ``compose_candidate.py --matched``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


COV_CODES = {"gate": 0, "segment": 1, "none": 2, "nofile": 3}


def stacker_features(P: np.ndarray, airport: np.ndarray, cov_class: np.ndarray, local_hour: np.ndarray, airports: list[str]) -> np.ndarray:
    """Stage-O stacker design matrix: the legs' raw predictions, airport code (categorical; -1 = unseen), ADS-B
    coverage-class code (categorical) and local hour."""

    ap_code = np.array([airports.index(a) if a in airports else -1 for a in np.asarray(airport).astype(str)], np.float64)
    cv_code = np.array([COV_CODES.get(c, -1) for c in np.asarray(cov_class).astype(str)], np.float64)
    return np.column_stack([np.asarray(P, np.float64), ap_code, cv_code, np.asarray(local_hour, np.float64)])


def blend_matrix(P: np.ndarray, airport: np.ndarray, adsb_coverage, spec: dict, local_hour: np.ndarray | None = None,
                 spec_dir: Path | None = None) -> tuple[np.ndarray, dict]:
    """The raw blend of a leg matrix ``P`` (rows x legs, in ``spec["legs"]`` order) under a cells / stacker spec."""

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from prc_challenge import gate_stats as gs

    names = list(spec["legs"])

    def vector(d: dict) -> np.ndarray:
        if set(d) != set(names):
            raise SystemExit(f"weight keys {sorted(d)} != legs {sorted(names)}")
        w = np.array([float(d[k]) for k in names])
        if (w < 0).any() or abs(w.sum() - 1.0) > 1e-9:
            raise SystemExit("weights must be >= 0 and sum to 1")
        return w

    cov = gs.coverage_class(adsb_coverage).astype(str)
    if "stacker" in spec:
        import lightgbm as lgb

        st = spec["stacker"]
        model = Path(st["model"])
        if not model.is_absolute() and spec_dir is not None:
            model = spec_dir / model
        if local_hour is None:
            raise SystemExit("stacker spec needs local_hour")
        booster = lgb.Booster(model_file=str(model))
        X = stacker_features(P, airport, cov, local_hour, list(st["airports"]))
        out = P @ vector(spec["default"]) + booster.predict(X, num_threads=int(st.get("threads", 2)))
        return out, {"mode": "stacker", "model": str(model)}
    ap = np.asarray(airport).astype(str)
    group = np.where(np.isin(ap, list(spec["rich_airports"])), "rich7", ap)
    cells = np.char.add(np.char.add(group.astype(str), "|"), cov)
    default = vector(spec["default"])
    out = np.empty(P.shape[0])
    counts = {}
    for cell in np.unique(cells):
        s = cells == cell
        w = vector(spec["cells"][cell]) if cell in spec["cells"] else default
        out[s] = P[s] @ w
        counts[str(cell)] = {"rows": int(s.sum()), "listed": bool(cell in spec["cells"])}
    return out, {"mode": "cells", "cells": counts}


def weighted_blend(base: pl.DataFrame, paths: list[Path], spec: dict, spec_dir: Path | None = None) -> tuple[np.ndarray, dict]:
    """Raw blend of ``s4_raw_prediction`` under a weights file (global vector, cell x leg, or the stage-O stacker)."""

    names = list(spec["legs"])
    if len(names) != len(paths):
        raise SystemExit(f"weights file names {len(names)} legs, {len(paths)} --leg given")
    raws = []
    for path in paths:
        other = pl.read_parquet(path).select("MVT_ID_mvt", pl.col("s4_raw_prediction").alias("r"))
        j = base.select("MVT_ID_mvt").join(other, on="MVT_ID_mvt", how="left", maintain_order="left")
        if j["r"].null_count():
            raise SystemExit(f"{path}: missing rows")
        raws.append(j["r"].to_numpy().astype(float))
    P = np.column_stack(raws)

    def vector(d: dict) -> np.ndarray:
        if set(d) != set(names):
            raise SystemExit(f"weight keys {sorted(d)} != legs {sorted(names)}")
        w = np.array([float(d[k]) for k in names])
        if (w < 0).any() or abs(w.sum() - 1.0) > 1e-9:
            raise SystemExit("weights must be >= 0 and sum to 1")
        return w

    info: dict = {"mode": "global" if "weights" in spec else ("stacker" if "stacker" in spec else "cells")}
    if "weights" in spec:
        w = vector(spec["weights"])
        info["weights"] = dict(zip(names, w.tolist()))
        return P @ w, info
    cols = ["MVT_ID_mvt", "AIRPORT", "adsb_coverage", *(["local_hour"] if "stacker" in spec else [])]
    m = pl.read_parquet(spec["cell_matrix"], columns=cols).with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
    c = base.select(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(m, on="MVT_ID_mvt", how="left", maintain_order="left")
    if c["AIRPORT"].null_count():
        raise SystemExit("cell matrix does not cover every row")
    out, binfo = blend_matrix(P, c["AIRPORT"].cast(pl.String).to_numpy(), c["adsb_coverage"], spec,
                              c["local_hour"].cast(pl.Float64).to_numpy() if "local_hour" in c.columns else None, spec_dir)
    info.update({k: v for k, v in binfo.items() if k != "mode"})
    return out, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", action="append", required=True, help="path[:weight]; weights must sum to 1")
    ap.add_argument("--weights-json", type=Path, help="pre-registered weights file (global or cell x leg); --leg paths in its 'legs' order")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    legs = []
    for spec in args.leg:
        path, _, w = spec.rpartition(":") if ":" in spec and not spec.endswith(".parquet") else (spec, "", "")
        legs.append((Path(path), float(w) if w else None))
    if args.weights_json is not None:
        weights_json_main(args, legs)
        return
    if any(w is None for _, w in legs):
        legs = [(p, 1.0 / len(legs)) for p, _ in legs]
    if abs(sum(w for _, w in legs) - 1.0) > 1e-9:
        raise SystemExit("weights must sum to 1")
    base = pl.read_parquet(legs[0][0]).select("MVT_ID_mvt", "nm_unmatched", "gate_e17", "gate_e18", "s4_raw_prediction", "prediction")
    blend = legs[0][1] * base["s4_raw_prediction"].to_numpy().astype(float)
    for path, w in legs[1:]:
        other = pl.read_parquet(path).select("MVT_ID_mvt", pl.col("s4_raw_prediction").alias("r"))
        j = base.select("MVT_ID_mvt").join(other, on="MVT_ID_mvt", how="left", maintain_order="left")
        if j["r"].null_count():
            raise SystemExit(f"{path}: missing rows")
        blend += w * j["r"].to_numpy().astype(float)
    gated = (base["gate_e17"].cast(pl.Int8) + base["gate_e18"].cast(pl.Int8)).to_numpy() > 0
    pred = np.where(gated, base["prediction"].to_numpy().astype(float), blend)
    pred = np.clip(pred, 0.0, None)
    out = base.select("MVT_ID_mvt").with_columns(pl.Series("prediction", pred))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out)
    man = {
        "legs": [{"path": str(p), "weight": w, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p, w in legs],
        "gated_rows_kept_from_leg0": int(gated.sum()),
        "out_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "pred_mean": float(pred.mean()),
        "pred_max": float(pred.max()),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2))


def weights_json_main(args: argparse.Namespace, legs: list[tuple[Path, float | None]]) -> None:
    if any(w is not None for _, w in legs):
        raise SystemExit("--weights-json and path:weight specs are exclusive")
    wspec = json.loads(args.weights_json.read_text())
    base = pl.read_parquet(legs[0][0]).select("MVT_ID_mvt", "nm_unmatched", "gate_e17", "gate_e18", "s4_raw_prediction", "prediction")
    blend, winfo = weighted_blend(base, [p for p, _ in legs], wspec, args.weights_json.resolve().parent)
    gated = (base["gate_e17"].cast(pl.Int8) + base["gate_e18"].cast(pl.Int8)).to_numpy() > 0
    pred = np.clip(np.where(gated, base["prediction"].to_numpy().astype(float), blend), 0.0, None)
    out = base.select("MVT_ID_mvt").with_columns(pl.Series("prediction", pred))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out)
    man = {
        "legs": [{"path": str(p), "name": n, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for (p, _), n in zip(legs, wspec["legs"])],
        "weights_json": {"path": str(args.weights_json), "sha256": hashlib.sha256(args.weights_json.read_bytes()).hexdigest(), **winfo},
        "gated_rows_kept_from_leg0": int(gated.sum()),
        "out_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "pred_mean": float(pred.mean()),
        "pred_max": float(pred.max()),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k != "legs"}, indent=2))


if __name__ == "__main__":
    main()
