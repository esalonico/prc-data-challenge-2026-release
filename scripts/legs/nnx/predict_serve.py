# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Predict-only serving of saved SA-NNX all-year legs on another split cache (final phase, 4 months).

Loads ``runs/nnx_search_<src-split>/<name>/model_seed<s>.pt`` and the run's ``result.json`` (config with the
model sizes it was trained with, target mean / sd of the fit rows), predicts the ``ev`` side of the cache
``nnx_cache_<split>`` (``prep.py <split>``: same train matrix, so the same preprocessor; a new evaluation
matrix) exactly as ``train.run`` predicts after training (seed mean of the raw inverted predictions; V14
recipe for ``mlp_v14``), composes the ranking rows with ``ma.compose_ranking`` (as ``finalize.py``) and writes
``<out-dir>/ranking_<name>.parquet``.  ``--check NAME=PATH`` compares the raw prediction with a stored leg.

Usage::

  PYTHONPATH=src .venv/bin/python scripts/legs/nnx/predict_serve.py --split ay3liso_serve4m \
      --names plr_s03_resnet_f3,plr_s03_eobtres_f3,ctx_gru_plr_f2 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NNX_THREADS", "3")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

import models  # noqa: E402
import train as T  # noqa: E402
from prep import RUNS, TARGETS, cache_dir  # noqa: E402
from prc_challenge import matrix_arms as ma  # noqa: E402
from prc_challenge import s4_allyear as s4  # noqa: E402


def predict_leg(name: str, split: str, src_split: str, device) -> tuple[np.ndarray, np.ndarray, dict]:
    sd_dir = RUNS / f"nnx_search_{src_split}" / name
    res = json.loads((sd_dir / "result.json").read_text())
    cfg = res["cfg"]
    mu, sd = res["target_mu_sd"]
    root = cache_dir(split, cfg.get("cache_suffix", ""))
    meta = json.loads((root / "meta.json").read_text())
    spec = TARGETS[cfg["target"]]
    need_ctx = cfg["arch"] == "ctx"
    need_arr = need_ctx and bool(cfg["ctx"].get("arrivals"))
    ev = T.Split(root, "ev", device, need_ctx, need_arr=need_arr)
    if need_ctx and cfg["ctx"].get("compact"):
        sel = torch.tensor(T.compact_columns(meta), dtype=torch.int64, device=device)
        ev.nb_dense = ev.dense[:, sel].contiguous()
        if cfg["model"].get("n_nb_dense") not in (None, int(sel.numel())):
            raise SystemExit(f"{name}: compact column count differs from the trained model")
    cols = T.ctx_cols(cfg["ctx"]["k_airport"], cfg["ctx"]["k_runway"], device, meta.get("k_airport"), meta.get("k_runway")) if need_ctx else None
    n_slots = int(cols.numel()) if need_ctx else 0
    off_ev = np.load(root / f"ev_off_{cfg['target']}.npy")
    preds, stds = [], []
    for seed in cfg["seeds"]:
        model = models.build(cfg, meta["sizes"], len(meta["nums"]), meta["n_dense"], n_slots).to(device)
        model.load_state_dict(torch.load(sd_dir / f"model_seed{seed}.pt", map_location=device))
        t_hat = T.predict(model, ev, cfg, cols, device)
        stds.append(t_hat)
        preds.append(T.invert(spec, off_ev, t_hat * sd + mu))
        del model
        T.empty_cache(device)
    if cfg["arch"] == "mlp_v14":
        raw = T.invert(spec, off_ev, np.mean(np.vstack(stds), axis=0) * sd + mu)
    else:
        raw = np.mean(np.vstack(preds), axis=0)
    info = {"name": name, "src": str(sd_dir), "cache": str(root), "seeds": cfg["seeds"], "target": cfg["target"],
            "mu_sd": [mu, sd], "run": meta["run"]}
    return raw, np.load(root / "ev_ids.npy"), info


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, help="cache to predict (nnx_cache_<split>, built by prep.py)")
    p.add_argument("--src-split", default="ay3liso", help="search dir of the trained models (nnx_search_<src-split>)")
    p.add_argument("--names", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--check", action="append", default=[], help="NAME=PATH stored ranking leg to compare with")
    a = p.parse_args(argv)
    device = T.resolve_device(a.device)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = dict(c.split("=", 1) for c in a.check)
    report_path = out / "predict_serve_nnx_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for name in [x for x in a.names.split(",") if x]:
        t0 = time.perf_counter()
        raw, ids, info = predict_leg(name, a.split, a.src_split, device)
        evaluation = pl.read_parquet(Path(info["run"]) / "eval_matrix.parquet")
        if not np.array_equal(evaluation["MVT_ID_mvt"].to_numpy(), ids):
            raise SystemExit("row order mismatch between the cache and the evaluation matrix")
        composed, summary = ma.compose_ranking(evaluation, raw)
        path = out / f"ranking_{name}.parquet"
        s4.atomic_parquet(composed, path)
        np.save(out / f"raw_{name}.npy", raw)
        entry = {**info, "rows": composed.height, "path": str(path), "sha256": s4.sha256_file(path),
                 "seconds": round(time.perf_counter() - t0, 1)}
        if name in checks:
            stored = pl.read_parquet(checks[name]).select("MVT_ID_mvt", pl.col("s4_raw_prediction").alias("stored"))
            j = stored.join(composed.select("MVT_ID_mvt", "s4_raw_prediction"), on="MVT_ID_mvt", how="inner", validate="1:1")
            x, y = j["s4_raw_prediction"].to_numpy(), j["stored"].to_numpy()
            entry["check"] = {"stored": checks[name], "n": j.height, "max_abs_diff": float(np.max(np.abs(x - y))),
                              "mean_abs_diff": float(np.mean(np.abs(x - y))), "pearson_r": float(np.corrcoef(x, y)[0, 1])}
        report[name] = entry
        s4.atomic_json(report_path, report)
        T.log(f"{name}: {composed.height} rows -> {path.name}; check {entry.get('check')}")


if __name__ == "__main__":
    main()
