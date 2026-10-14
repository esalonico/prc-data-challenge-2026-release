# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Predict-only serving of the saved SA-TRAJ all-year leg on another split (final phase, 4 months).

Loads ``<ART>/search_<src-split>/<name>/model_seed<s>.pt`` and ``result.json`` (config, target mean / sd of the
fit rows), predicts the ``ev`` side of ``nnx_cache_<split>`` + ``<ART>/seq_<split>`` (``scripts/legs/nnx/prep.py`` and
``tracks.py split``) exactly as ``train_traj.run`` does after training (seed mean of the raw inverted, clipped
predictions), composes with ``ma.compose_ranking`` (as ``finalize.py``) and writes ``<out-dir>/ranking_<name>.parquet``.
``--check NAME=PATH`` compares the raw prediction with a stored leg.

Usage::

  PYTHONPATH=src .venv/bin/python scripts/legs/traj/predict_serve.py --split ay3liso_serve4m --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

import train_traj as tt  # noqa: E402  (sets the thread env, ART, Side, predict)
import model as tm  # noqa: E402
import train as nnx_train  # noqa: E402
from prep import TARGETS, cache_dir  # noqa: E402
from prc_challenge import matrix_arms as ma  # noqa: E402
from prc_challenge import s4_allyear as s4  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True)
    p.add_argument("--src-split", default="ay3liso")
    p.add_argument("--name", default="traj_conv_resnet_f3")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--check", default=None, help="PATH of the stored ranking leg to compare with")
    a = p.parse_args(argv)
    device = nnx_train.resolve_device(a.device)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sd_dir = tt.ART / f"search_{a.src_split}" / a.name
    res = json.loads((sd_dir / "result.json").read_text())
    cfg = res["cfg"]
    mu, sd = res["target_mu_sd"]
    root = cache_dir(a.split)
    meta = json.loads((root / "meta.json").read_text())
    spec = TARGETS[cfg["target"]]
    ev = tt.Side(a.split, "ev", device)
    off_ev = np.load(root / f"ev_off_{cfg['target']}.npy")
    m = cfg["model"]
    preds = []
    for seed in cfg["seeds"]:
        model = tm.TrajNet(meta["sizes"], len(meta["nums"]), meta["n_dense"], body=m["body"], widths=m["widths"],
                           head_widths=m["head_widths"], dropout=m["dropout"], plr=m["plr"], traj=m["traj"],
                           resnet=m.get("resnet")).to(device)
        model.load_state_dict(torch.load(sd_dir / f"model_seed{seed}.pt", map_location=device))
        preds.append(nnx_train.invert(spec, off_ev, tt.predict(model, ev, cfg, device) * sd + mu))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        tt.log(f"{a.name}: seed {seed} predicted")
    raw = np.mean(np.vstack(preds), axis=0)
    ids = np.load(root / "ev_ids.npy")
    evaluation = pl.read_parquet(Path(meta["run"]) / "eval_matrix.parquet")
    if not np.array_equal(evaluation["MVT_ID_mvt"].to_numpy(), ids):
        raise SystemExit("row order mismatch between the cache and the evaluation matrix")
    composed, _ = ma.compose_ranking(evaluation, raw)
    path = out / f"ranking_{a.name}.parquet"
    s4.atomic_parquet(composed, path)
    np.save(out / f"raw_{a.name}.npy", raw)
    entry = {"name": a.name, "split": a.split, "src": str(sd_dir), "seeds": cfg["seeds"], "mu_sd": [mu, sd],
             "seq_meta": json.loads((tt.ART / f"seq_{a.split}" / "meta.json").read_text()) | {"store": "..."},
             "rows": composed.height, "path": str(path), "sha256": s4.sha256_file(path),
             "seconds": round(time.perf_counter() - t0, 1)}
    if a.check:
        stored = pl.read_parquet(a.check).select("MVT_ID_mvt", pl.col("s4_raw_prediction").alias("stored"))
        j = stored.join(composed.select("MVT_ID_mvt", "s4_raw_prediction"), on="MVT_ID_mvt", how="inner", validate="1:1")
        x, y = j["s4_raw_prediction"].to_numpy(), j["stored"].to_numpy()
        entry["check"] = {"stored": a.check, "n": j.height, "max_abs_diff": float(np.max(np.abs(x - y))),
                          "mean_abs_diff": float(np.mean(np.abs(x - y))), "pearson_r": float(np.corrcoef(x, y)[0, 1])}
    s4.atomic_json(out / f"predict_serve_{a.name}.json", entry)
    tt.log(f"{a.name}: {composed.height} rows -> {path.name}; check {entry.get('check')}")


if __name__ == "__main__":
    main()
