# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-TRAJ trainer: PLR tabular trunk + focal-trajectory encoder on an SA-NNX split cache.

Inputs per row: the SA-NNX cache (``nnx_cache_<split>``: V14 preprocessing fitted on the split's fit
rows, clocks, target offsets) and the trajectory cache (``seq_<split>``, ``tracks.py``).  The
training loop, optimiser, schedule, target handling and holdout proxy are SA-NNX's
(``scripts/legs/nnx/train.py``): AdamW, warm-up + cosine, grad-clip 1, standardised target, fixed
epochs, final-epoch predictions, seed mean of the raw (inverted, clipped) predictions.

Holdout splits (m10, m5): the proxy is SA-NNX's month-holdout blend of the 7 base legs **plus the
three SA-NNX picks** (their stored holdout ``pred_mean.npy``), i.e. the ten-leg proxy of the current
thirteen-leg system; ``delta`` = RMSE change when the candidate joins as one more equal leg.
Full-fit splits (v4jjnd, ay, ay3l): predictions only, evaluation labels never loaded; models saved.

Usage::

  PYTHONPATH=src .venv/bin/python <wt>/scripts/legs/traj/train_traj.py --split m10 --config '<json>'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NNX_THREADS", "3")
for _v in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "nnx"))
sys.path.insert(0, str(HERE.parent))

import model as tm  # noqa: E402
import train as nnx_train  # noqa: E402
from prep import RUNS, TARGETS, cache_dir  # noqa: E402

WT = HERE.parents[2]
ART = Path(os.environ["TRAJ_ART"]) if os.environ.get("TRAJ_ART") else Path(os.environ.get("PRC_MAIN_REPO", Path(__file__).resolve().parents[3])) / "artifacts/experiments/ADSB-trajectory-20260927"  # bake VM: PRC_MAIN_REPO; traj_hires: TRAJ_ART
RESULTS = Path(os.environ["TRAJ_RESULTS"]) if os.environ.get("TRAJ_RESULTS") else WT / "results" / "traj" / "traj_results.jsonl"  # bake: its own log
PICKS = ["plr_s03_resnet_f3", "plr_s03_eobtres_f3", "ctx_gru_plr_f2"]
CHUNK_ROWS = 512
ROW_STEP = 128
LEN_STEP = 16
DEFAULTS = {
    "model": {"body": "mlp", "widths": [512, 512], "head_widths": [256], "dropout": 0.1,
              "plr": {"k": 16, "sigma": 0.3, "d": 16}, "traj": {"d": 96, "kind": "conv", "n_blocks": 3}},
    "target": "delta", "epochs": 10, "seeds": [0], "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 4096,
    "warmup_steps": 300, "lr_floor": 0.02, "grad_clip": 1.0, "n_max": 128, "no_track": False, "budget_min": 400,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Side:
    """Device-resident inputs of one cache side (fit or ev): tabular + trajectory."""

    def __init__(self, split: str, side: str, device):
        root, sroot = cache_dir(split), ART / f"seq_{split}"
        ld = lambda r, k: np.load(r / f"{side}_{k}.npy", mmap_mode="r")  # noqa: E731
        self.n = ld(root, "ids").shape[0]
        self.cat = torch.from_numpy(np.asarray(ld(root, "cat")).astype(np.int64)).to(device)
        self.dense = torch.from_numpy(np.asarray(ld(root, "dense"))).to(device)
        self.clock = torch.from_numpy(np.asarray(ld(root, "clock"))).to(device)
        self.off = torch.from_numpy(np.asarray(ld(sroot, "trk_off")).astype(np.int64)).to(device)
        self.len = torch.from_numpy(np.asarray(ld(sroot, "trk_len")).astype(np.int64)).to(device)
        pts = np.asarray(ld(sroot, "trk_pts"))
        if pts.shape[0] == 0:
            pts = np.zeros((1, 9), np.float16)
        self.pts = torch.from_numpy(pts).to(device)
        self.row = torch.from_numpy(np.asarray(ld(sroot, "trk_row"))).to(device)
        ids_seq = self.len.shape[0]
        if ids_seq != self.n:
            raise SystemExit(f"{split}/{side}: trajectory cache rows {ids_seq} != tabular {self.n}")
        self.device = device

    def seq(self, idx: torch.Tensor, n_max: int, no_track: bool) -> dict:
        """Rows of ``idx`` with a track, sorted by length and cut into chunks of ``CHUNK_ROWS`` rows,
        each padded to its own longest track rounded up to a multiple of ``LEN_STEP`` (the encoders are
        exactly padding-invariant).  The last chunk is padded to a multiple of ``ROW_STEP`` rows with
        copies of its last row, which are scattered to a discarded extra row (no gradient).  The bounded set of tensor shapes
        keeps the MPS graph cache small (unbounded shapes leaked ~1 MB of host memory per new shape)."""
        if no_track:
            return {"chunks": []}
        lens_all = self.len[idx].clamp(max=n_max)
        lc = lens_all.cpu().numpy()
        order = np.argsort(lc, kind="stable")
        nz = order[lc[order] > 0]
        chunks = []
        for a in range(0, len(nz), CHUNK_ROWS):
            part = nz[a : a + CHUNK_ROWS]
            k = len(part)
            rows_pad = min(CHUNK_ROWS, -(-k // ROW_STEP) * ROW_STEP)
            if rows_pad > k:
                part = np.concatenate([part, np.full(rows_pad - k, part[-1])])
            n = min(n_max, -(-int(lc[part].max()) // LEN_STEP) * LEN_STEP)
            r = torch.from_numpy(part.astype(np.int64)).to(self.device)
            gi = idx[r]
            lens = lens_all[r]
            ar = torch.arange(n, device=self.device)
            mask = ar[None, :] < lens[:, None]
            pos = torch.where(mask, self.off[gi][:, None] + ar[None, :], torch.zeros_like(mask, dtype=torch.int64))
            pts = torch.index_select(self.pts, 0, pos.reshape(-1)).view(len(part), n, -1).float()
            pts = torch.where(mask[..., None], pts, torch.zeros_like(pts))
            m = mask.float()
            scatter = torch.from_numpy(np.where(np.arange(len(part)) < k, part, len(lc)).astype(np.int64)).to(self.device)
            chunks.append({"gather": r, "scatter": scatter, "feats": tm.point_features(pts, m, self.row[gi], self.clock[gi]), "mask": m, "lens": lens})
        return {"chunks": chunks}


def forward(model, s: Side, idx, cfg):
    return model(s.cat[idx], s.dense[idx].float(), s.seq(idx, cfg["n_max"], cfg["no_track"]))


def predict(model, s: Side, cfg, device, chunk: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, s.n, chunk):
            idx = torch.arange(a, min(a + chunk, s.n), device=device)
            out.append(forward(model, s, idx, cfg).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def merge(base: dict, over: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def run(cfg_in: dict, split: str, stage: str, device, out_root: Path | None = None, save_models: bool = False) -> dict:
    cfg = merge(DEFAULTS, cfg_in)
    name = cfg["name"]
    root = cache_dir(split)
    meta = json.loads((root / "meta.json").read_text())
    if meta["cats"][0] != "AIRPORT":
        raise SystemExit("first categorical must be AIRPORT")
    out = (out_root or ART / f"search_{split}") / name
    out.mkdir(parents=True, exist_ok=True)
    holdout = meta["holdout"] is not None
    spec = TARGETS[cfg["target"]]
    t_start = time.perf_counter()
    y_all = np.load(root / "fit_label.npy")
    off_fit = np.load(root / f"fit_off_{cfg['target']}.npy")
    t_fit = nnx_train.fit_label(spec, y_all, off_fit)
    mu, sd = float(t_fit.mean()), float(t_fit.std())
    y_std = torch.from_numpy(((t_fit - mu) / sd).astype(np.float32)).to(device)
    fit = Side(split, "fit", device)
    ev = Side(split, "ev", device)
    off_ev = np.load(root / f"ev_off_{cfg['target']}.npy")
    label_ev = np.load(root / "ev_label.npy") if holdout else None
    proxy = None
    if holdout and (RUNS / f"nnx_proxy_{split}" / "proxy.json").exists():  # m10 / m5 (D-fold splits: alone only)
        extra = [str(RUNS / f"nnx_search_{split}" / p / "pred_mean.npy") for p in PICKS]
        proxy = nnx_train.Proxy(split, label_ev, extra)
    log(f"{name}: split={split} fit={fit.n} ev={ev.n} target={cfg['target']} mu={mu:.2f} sd={sd:.2f} proxy_base={proxy.base if proxy else None} legs={proxy.n if proxy else 0}")
    n = fit.n
    bs = cfg["batch_size"]
    total = math.ceil(n / bs) * cfg["epochs"]
    seed_preds, curves = [], []
    n_par = None
    aborted = False
    reuse = cfg.get("reuse_seeds", {})  # {"<seed>": "<dir of the identical config's run>"} (SA-NNX precedent)
    for seed in cfg["seeds"]:
        if str(seed) in reuse:
            src = Path(reuse[str(seed)])
            prev = json.loads((src / "result.json").read_text())
            same = {k: v for k, v in prev["cfg"].items() if k not in ("name", "seeds", "stage", "reuse_seeds", "budget_min")} == {k: v for k, v in cfg.items() if k not in ("name", "seeds", "stage", "reuse_seeds", "budget_min")}
            if not same or prev["split"] != split:
                raise SystemExit(f"seed {seed}: {src} is not the identical config on {split}")
            raw = np.load(src / f"pred_seed{seed}.npy")
            np.save(out / f"pred_seed{seed}.npy", raw)
            seed_preds.append(raw)
            curves.append([{"reused_from": str(src)}])
            log(f"{name}: seed {seed} reused from {src}")
            continue
        torch.manual_seed(seed)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        m = cfg["model"]
        model = tm.TrajNet(meta["sizes"], len(meta["nums"]), meta["n_dense"], body=m["body"], widths=m["widths"], head_widths=m["head_widths"], dropout=m["dropout"], plr=m["plr"], traj=m["traj"], resnet=m.get("resnet")).to(device)
        model.traj.checkpoint = bool(cfg.get("traj_checkpoint", False))  # D-109: memory only (same math, same dropout masks)
        n_par = int(sum(p.numel() for p in model.parameters()))
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        def factor(step: int) -> float:
            w = min(1.0, (step + 1) / cfg["warmup_steps"])
            return w * (cfg["lr_floor"] + (1 - cfg["lr_floor"]) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
        curve = []
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.perf_counter()
            model.train()
            perm = torch.randperm(n, generator=gen).to(device)
            running = torch.zeros((), device=device)
            for a in range(0, n, bs):
                idx = perm[a : a + bs]
                if idx.numel() < 2:
                    continue
                pred = forward(model, fit, idx, cfg)
                loss = torch.mean((pred - y_std[idx]) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
                sched.step()
                running += loss.detach() * idx.numel()
                if device.type == "mps" and cfg.get("empty_cache_every") and (a // bs + 1) % cfg["empty_cache_every"] == 0:
                    torch.mps.empty_cache()  # D-109: return the allocator cache to the system (memory only)
            rec = {"epoch": epoch, "train_loss": round(float(running.cpu()) / n, 5), "sec": round(time.perf_counter() - t0, 1)}
            if device.type == "mps":
                torch.mps.empty_cache()
            if holdout and (epoch == cfg["epochs"] or epoch % cfg.get("eval_every", 2) == 0):
                raw = nnx_train.invert(spec, off_ev, predict(model, ev, cfg, device) * sd + mu)
                rec["alone"] = round(nnx_train.rmse(label_ev, raw), 4)
                if proxy:
                    rec["delta"] = round(proxy.delta(raw), 4)
            curve.append(rec)
            log(f"{name} seed {seed} " + json.dumps(rec))
            if not np.isfinite(rec["train_loss"]):
                aborted = True
                break
            if (time.perf_counter() - t_start) / 60 > cfg["budget_min"]:
                log(f"{name}: budget exceeded after epoch {epoch}")
                aborted = True
                break
        raw = nnx_train.invert(spec, off_ev, predict(model, ev, cfg, device) * sd + mu)
        np.save(out / f"pred_seed{seed}.npy", raw)
        seed_preds.append(raw)
        curves.append(curve)
        if save_models:
            torch.save(model.state_dict(), out / f"model_seed{seed}.pt")
        del model, opt
        if device.type == "mps":
            torch.mps.empty_cache()
        if aborted:
            break
    raw_mean = np.mean(np.vstack(seed_preds), axis=0)
    np.save(out / "pred_mean.npy", raw_mean)
    np.save(out / "ids.npy", np.load(root / "ev_ids.npy"))
    res = {"name": name, "stage": stage, "split": split, "params": n_par, "cfg": cfg, "seconds": round(time.perf_counter() - t_start, 1),
           "target_mu_sd": [mu, sd], "curves": curves, "aborted": aborted, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "pred_mean": str(out / "pred_mean.npy"), "device": str(device)}
    if holdout:
        res["alone"] = round(nnx_train.rmse(label_ev, raw_mean), 4)
        res["alone_seeds"] = [round(nnx_train.rmse(label_ev, p), 4) for p in seed_preds]
        if proxy:
            res["proxy_base"] = round(proxy.base, 4)
            res["proxy_legs"] = proxy.names
            res["delta"] = round(proxy.delta(raw_mean), 4)
            res["delta_seeds"] = [round(proxy.delta(p), 4) for p in seed_preds]
    (out / "result.json").write_text(json.dumps(res, indent=1))
    if holdout:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in res.items() if k != "curves"}
        trained = [c for c in curves if c and "epoch" in c[0]]
        slim["sec_per_epoch"] = round(float(np.mean([r["sec"] for r in trained[0]])), 1) if trained else None
        with RESULTS.open("a") as fh:
            fh.write(json.dumps(slim) + "\n")
        log(f"RESULT {name}: alone={res['alone']} delta={res.get('delta')} params={n_par} sec={res['seconds']}")
    return res


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--stage", default="adhoc")
    p.add_argument("--out-root")
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"), help="auto: cuda, else mps, else cpu (on the Mac: the previous choice)")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    device = nnx_train.resolve_device(a.device)
    cfgs = json.loads(Path(a.config).read_text()) if a.config.endswith(".json") else json.loads(a.config)
    for cfg in cfgs if isinstance(cfgs, list) else [cfgs]:
        try:
            run(cfg, a.split, cfg.get("stage", a.stage), device, Path(a.out_root) if a.out_root else None, a.save_models)
        except Exception as exc:  # keep the queue going; record the failure
            import traceback

            traceback.print_exc()
            RESULTS.parent.mkdir(parents=True, exist_ok=True)
            with RESULTS.open("a") as fh:
                fh.write(json.dumps({"name": cfg["name"], "split": a.split, "error": repr(exc)[:500]}) + "\n")
            if device.type == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
