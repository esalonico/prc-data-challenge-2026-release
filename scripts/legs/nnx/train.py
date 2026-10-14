# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-NNX trainer: fit one search configuration on a prepared split cache (``prep.py``).

Usage::

  PYTHONPATH=src .venv/bin/python <wt>/scripts/legs/nnx/train.py --split m10 --config '<json>' [--stage s1]
  PYTHONPATH=src .venv/bin/python <wt>/scripts/legs/nnx/train.py --split m10 --queue <queue.json>

A config is a dict::

  {"name": ..., "arch": "mlp_v14|mlp|resnet|dcn|tabm|ftt|ctx", "model": {...},
   "target": "delta|resid|eobtres|lobtoff", "loss": "mse|huber", "huber_delta": 1.0,
   "epochs": 10, "seeds": [0], "lr": 2e-3, "weight_decay": 1e-4, "batch_size": 4096,
   "warmup_steps": 300, "lr_floor": 0.02, "grad_clip": 1.0, "subsample": 1.0,
   "ctx": {"k_airport": 16, "k_runway": 8}}

Holdout splits (m10, m5): after every epoch the holdout rows are predicted and scored alone and
as one more equal leg of the proxy blend (``runs/nnx_proxy_<split>/``; missing -> alone only).
The run writes ``runs/nnx_search_<split>/<name>/pred_seed<s>.npy`` (raw taxi seconds, final
epoch), ``pred_mean.npy`` (seed mean of the standardised prediction, then inverted, as V14),
``result.json``, and appends one line to ``<worktree>/results/nnx_results.jsonl``.
Full-fit splits (jj, ay): predictions only; the evaluation labels are never loaded.

Bake additions (D-076 code gaps #1 and #3; defaults unchanged on MPS/CPU):

* ``--device auto|cuda|mps|cpu`` (auto: cuda, else mps, else cpu); on CUDA the cache blocks are copied
  through pinned host memory; ``--deterministic`` sets ``torch.use_deterministic_algorithms`` (and
  ``CUBLAS_WORKSPACE_CONFIG``); a config ``"precision": "bf16"`` runs forward passes under bf16 autocast
  on CUDA only (FT-T / attention variants).
* ctx configs may name a non-default cache (``"cache_suffix"``, e.g. ``_k32r16a16`` from
  ``prep.py --k-airport 32 --k-runway 16 --k-arrivals 16``); the neighbour slot layout is read from the
  cache's ``meta.json``.  ``"ctx": {"arrivals": true}`` adds the arrival tokens to the context model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_THREADS = os.environ.get("NNX_THREADS", "2")
for _v in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import contextlib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import models  # noqa: E402
from prep import RUNS, TARGETS, cache_dir  # noqa: E402

WT = HERE.parents[2]
RESULTS = Path(os.environ["NNX_RESULTS"]) if os.environ.get("NNX_RESULTS") else WT / "results" / "nnx_results.jsonl"  # bake: its own log
DEFAULTS = {
    "arch": "mlp_v14",
    "model": {},
    "target": "delta",
    "loss": "mse",
    "huber_delta": 1.0,
    "epochs": 10,
    "seeds": [0],
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "batch_size": 4096,
    "warmup_steps": 300,
    "lr_floor": 0.02,
    "grad_clip": 1.0,
    "subsample": 1.0,
    "ctx": {"k_airport": 16, "k_runway": 8},
    "eval_every": 1,
    "budget_min": 240,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, np.float64) - np.asarray(p, np.float64)) ** 2)))


def invert(spec, offset: np.ndarray, t_hat: np.ndarray) -> np.ndarray:
    """``s4.TargetSpec.predict`` on a stored offset array: max(0, offset + sign * t_hat)."""

    return np.clip(offset + spec.sign * np.asarray(t_hat, np.float64), 0, None)


def fit_label(spec, y: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """``s4.TargetSpec.with_fit_label`` on stored arrays."""

    if spec.is_default:
        return y - offset
    t = spec.sign * (y - offset)
    return np.clip(t, -spec.winsor_sec, spec.winsor_sec) if spec.winsor_sec is not None else t


def resolve_device(name: str = "auto") -> torch.device:
    """auto: cuda, else mps, else cpu (on the Mac: exactly the previous mps/cpu choice)."""

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def to_dev(x: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(x)
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def empty_cache(device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def autocast(cfg: dict, device):
    if cfg.get("precision") == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class Split:
    """Device-resident blocks of one cache side (fit or ev)."""

    def __init__(self, root: Path, side: str, device, need_ctx: bool, rows: np.ndarray | None = None, need_arr: bool = False):
        ld = lambda k: np.load(root / f"{side}_{k}.npy", mmap_mode="r")  # noqa: E731
        self.n_all = ld("ids").shape[0]
        self.cat = to_dev(np.asarray(ld("cat")).astype(np.int64), device)
        self.dense = to_dev(np.asarray(ld("dense")), device)
        self.rows = rows  # subsample of focal rows (None = all)
        if need_ctx:
            self.nb_idx = to_dev(np.asarray(ld("nb_idx")).astype(np.int64), device)
            self.nb_dt = to_dev(np.asarray(ld("nb_dt")), device)
            self.clock = to_dev(np.asarray(ld("clock")), device)
        self.arr = None
        if need_arr:
            has_taxi = (root / f"{side}_arr_taxi.npy").exists()
            self.arr = {
                "dt": to_dev(np.asarray(ld("arr_dt")), device),
                "mask": to_dev(np.asarray(ld("arr_mask")).astype(np.float32), device),
                "rwy": to_dev(np.asarray(ld("arr_rwy")).astype(np.float32), device),
            }
            if has_taxi:
                self.arr["taxi"] = to_dev(np.asarray(ld("arr_taxi")), device)
                self.arr["taxi_ok"] = to_dev(np.asarray(ld("arr_taxi_ok")).astype(np.float32), device)
        self.nb_dense = None
        self.root, self.side = root, side

    def arr(self, key: str) -> np.ndarray:
        return np.load(self.root / f"{self.side}_{key}.npy")


COMPACT_PREFIXES = ("takeoff_minus", "aobt", "lobt", "eobt", "adsb_", "unimpeded_p10", "unimpeded_median", "plan_", "p_plan", "p_nm", "s4_preliminary", "m1_prev_dep_age", "m1_next_dep_gap", "m1_stand_free", "local_hour", "iobt_missing", "eobt_missing")


def compact_columns(meta: dict) -> list[int]:
    """Dense columns (numerics + their NaN flags) used for neighbour tokens in the compact mode."""

    nums, nan_cols = meta["nums"], meta["nan_cols"]
    keep = [i for i, c in enumerate(nums) if c.startswith(COMPACT_PREFIXES)]
    keep += [len(nums) + j for j, c in enumerate(nan_cols) if c.startswith(COMPACT_PREFIXES)]
    return keep


def slog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x) / 60.0) / 3.0


N_ARR_FEAT = 4  # slog(dt), same runway, slog(taxi-in) (0 if absent/guarded), taxi-in present


def gather_arr(s: Split, idx: torch.Tensor) -> dict:
    a = s.arr
    mask = a["mask"][idx]
    zero = torch.zeros_like(mask)
    taxi = a["taxi"][idx] if "taxi" in a else zero
    ok = a["taxi_ok"][idx] if "taxi_ok" in a else zero
    feat = torch.stack([slog(a["dt"][idx]), a["rwy"][idx], slog(taxi) * ok, ok], dim=-1) * mask[..., None]
    return {"feat": feat, "mask": mask}


def gather_ctx(s: Split, idx: torch.Tensor, cols: torch.Tensor, ka: int, kr: int) -> dict:
    nb = s.nb_idx[idx][:, cols]  # [B, S]
    mask = (nb >= 0).float()
    nbc = nb.clamp(min=0)
    dt = s.nb_dt[idx][:, cols]
    cf = s.clock[idx][:, None, :]  # focal clocks [B,1,4]: take-off minus (aobt, eobt, lobt, sched)
    cn = s.clock[nbc]  # [B,S,4]
    rel = torch.stack(
        [
            dt,  # neighbour take-off - focal take-off
            dt - cn[..., 0],  # neighbour AOBT - focal take-off
            dt - cn[..., 0] + cf[..., 0],  # neighbour AOBT - focal AOBT
            dt + cf[..., 0],  # neighbour take-off - focal AOBT
            dt - cn[..., 1] + cf[..., 1],  # neighbour EOBT - focal EOBT
            dt - cn[..., 3] + cf[..., 3],  # neighbour schedule - focal schedule
        ],
        dim=-1,
    )
    return {
        "cat": s.cat[nbc],
        "dense": (s.nb_dense if s.nb_dense is not None else s.dense)[nbc].float(),
        "rel": slog(rel) * mask[..., None],
        "mask": mask,
        "k_airport": ka,
        "k_runway": kr,
    }


def ctx_cols(ka: int, kr: int, device, K_AIRPORT: int | None = None, K_RUNWAY: int | None = None) -> torch.Tensor:
    """Slot columns of the first ka / kr neighbours per side, in a cache laid out with K_AIRPORT / K_RUNWAY."""

    import prep

    K_AIRPORT = prep.K_AIRPORT if K_AIRPORT is None else K_AIRPORT
    K_RUNWAY = prep.K_RUNWAY if K_RUNWAY is None else K_RUNWAY
    if ka > K_AIRPORT or kr > K_RUNWAY:
        raise ValueError(f"config asks for {ka}/{kr} neighbours; the cache holds {K_AIRPORT}/{K_RUNWAY}")
    cols = list(range(ka)) + list(range(K_AIRPORT, K_AIRPORT + ka)) + list(range(2 * K_AIRPORT, 2 * K_AIRPORT + kr)) + list(range(2 * K_AIRPORT + K_RUNWAY, 2 * K_AIRPORT + K_RUNWAY + kr))
    return torch.tensor(cols, dtype=torch.int64, device=device)


def forward(model, s: Split, idx: torch.Tensor, cfg: dict, cols) -> torch.Tensor:
    if cfg["arch"] == "ctx":
        c = cfg["ctx"]
        ctx = gather_ctx(s, idx, cols, c["k_airport"], c["k_runway"])
        if s.arr is not None:
            ctx["arr"] = gather_arr(s, idx)
        return model(s.cat[idx], s.dense[idx].float(), ctx)
    return model(s.cat[idx], s.dense[idx].float())


def predict(model, s: Split, cfg: dict, cols, device, chunk: int | None = None) -> np.ndarray:
    if chunk is None:
        chunk = 16384 if cfg["arch"] in ("mlp_v14", "mlp", "resnet", "dcn") else 2048
        if os.environ.get("NNX_PREDICT_CHUNK"):  # bake smoke: smaller prediction chunks keep context models small on a CPU
            chunk = min(chunk, int(os.environ["NNX_PREDICT_CHUNK"]))
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, s.n_all, chunk):
            idx = torch.arange(a, min(a + chunk, s.n_all), device=device)
            with autocast(cfg, device):
                p = forward(model, s, idx, cfg, cols)
            if p.dim() == 2:
                p = p.mean(dim=1)
            out.append(p.float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


class Proxy:
    """Month-holdout blend proxy: equal mean of the base legs' raw predictions (+ extra legs)."""

    def __init__(self, split: str, label: np.ndarray, extra: list[str] | None = None):
        root = RUNS / f"nnx_proxy_{split}"
        info = json.loads((root / "proxy.json").read_text())
        self.names = list(info["legs"])
        legs = [np.load(root / f"{n}.npy") for n in self.names]
        for e in extra or []:
            legs.append(np.load(e))
            self.names.append(e)
        self.sum = np.sum(np.vstack(legs), axis=0)
        self.n = len(legs)
        self.y = label
        self.base = rmse(label, np.clip(self.sum / self.n, 0, None))

    def delta(self, raw: np.ndarray) -> float:
        return rmse(self.y, np.clip((self.sum + raw) / (self.n + 1), 0, None)) - self.base


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def run(cfg_in: dict, split: str, stage: str, device, out_root: Path | None = None, extra_legs: list[str] | None = None, save_models: bool = False) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    cfg.update(cfg_in)
    name = cfg["name"]
    root = cache_dir(split, cfg.get("cache_suffix", ""))
    meta = json.loads((root / "meta.json").read_text())
    out = (out_root or RUNS / f"nnx_search_{split}") / name
    out.mkdir(parents=True, exist_ok=True)
    holdout = meta["holdout"] is not None
    spec = TARGETS[cfg["target"]]
    need_ctx = cfg["arch"] == "ctx"
    need_arr = need_ctx and bool(cfg["ctx"].get("arrivals"))
    if need_arr and not meta.get("k_arrivals"):
        raise SystemExit(f"{name}: ctx.arrivals needs a cache built with --k-arrivals (cache {root})")
    t_start = time.perf_counter()

    y_all = np.load(root / "fit_label.npy")
    off_fit = np.load(root / f"fit_off_{cfg['target']}.npy")
    n_fit_all = y_all.shape[0]
    rows = None
    if cfg["subsample"] < 1.0:
        rng = np.random.default_rng(12345)
        rows = np.sort(rng.choice(n_fit_all, size=int(round(cfg["subsample"] * n_fit_all)), replace=False))
    fit = Split(root, "fit", device, need_ctx, rows, need_arr)
    ev = Split(root, "ev", device, need_ctx, need_arr=need_arr)
    if need_ctx and cfg["ctx"].get("compact"):
        sel = torch.tensor(compact_columns(meta), dtype=torch.int64, device=device)
        for side in (fit, ev):
            side.nb_dense = side.dense[:, sel].contiguous()
        cfg["model"]["n_nb_dense"] = int(sel.numel())
    t_fit = fit_label(spec, y_all, off_fit)
    if rows is not None:
        t_fit = t_fit[rows]
    mu, sd = float(t_fit.mean()), float(t_fit.std())
    y_std = torch.from_numpy(((t_fit - mu) / sd).astype(np.float32)).to(device)
    rows_t = torch.from_numpy(rows).to(device) if rows is not None else None
    off_ev = np.load(root / f"ev_off_{cfg['target']}.npy")
    label_ev = np.load(root / "ev_label.npy") if holdout else None
    proxy = None
    if holdout and (RUNS / f"nnx_proxy_{split}" / "proxy.json").exists():
        proxy = Proxy(split, label_ev, extra_legs)
    cols = ctx_cols(cfg["ctx"]["k_airport"], cfg["ctx"]["k_runway"], device, meta.get("k_airport"), meta.get("k_runway")) if need_ctx else None
    n_slots = int(cols.numel()) if need_ctx else 0
    if need_arr:
        cfg["model"]["n_arr_slots"] = 2 * int(meta["k_arrivals"])
    log(f"{name}: split={split} fit={len(t_fit)} ev={ev.n_all} target={cfg['target']} mu={mu:.2f} sd={sd:.2f} proxy_base={proxy.base if proxy else None}")

    n = len(t_fit)
    bs = cfg["batch_size"]
    steps_per_epoch = math.ceil(n / bs)
    total = steps_per_epoch * cfg["epochs"]
    seed_preds, curves, seed_std = [], [], []
    n_par = None
    aborted = False
    reuse = cfg.get("reuse_seeds", {})  # {"<seed>": "<dir with pred_seed<seed>.npy of the identical config>"}
    for seed in cfg["seeds"]:
        if str(seed) in reuse:
            raw = np.load(Path(reuse[str(seed)]) / f"pred_seed{seed}.npy")
            np.save(out / f"pred_seed{seed}.npy", raw)
            seed_preds.append(raw)
            seed_std.append((spec.sign * (raw - off_ev) - mu) / sd)  # informational only (clip not undone)
            curves.append([{"reused_from": reuse[str(seed)]}])
            log(f"{name}: seed {seed} reused from {reuse[str(seed)]}")
            continue
        torch.manual_seed(seed)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        model = models.build(cfg, meta["sizes"], len(meta["nums"]), meta["n_dense"], n_slots).to(device)
        n_par = n_params(model)
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
                pi = perm[a : a + bs]
                if pi.numel() < 2:
                    continue
                idx = rows_t[pi] if rows_t is not None else pi
                with autocast(cfg, device):
                    pred = forward(model, fit, idx, cfg, cols)
                pred = pred.float()
                yb = y_std[pi]
                if pred.dim() == 2:
                    yb = yb[:, None].expand_as(pred)
                if cfg["loss"] == "huber":
                    loss = torch.nn.functional.smooth_l1_loss(pred, yb, beta=cfg["huber_delta"])
                else:
                    loss = torch.mean((pred - yb) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
                sched.step()
                running += loss.detach() * pi.numel()
            rec = {"epoch": epoch, "train_loss": round(float(running.cpu()) / n, 5), "sec": round(time.perf_counter() - t0, 1)}
            if holdout and (epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]):
                t_hat = predict(model, ev, cfg, cols, device) * sd + mu
                raw = invert(spec, off_ev, t_hat)
                rec["alone"] = round(rmse(label_ev, raw), 4)
                if proxy:
                    rec["delta"] = round(proxy.delta(raw), 4)
            curve.append(rec)
            log(f"{name} seed {seed} " + json.dumps(rec))
            if not np.isfinite(rec["train_loss"]):
                aborted = True
                break
            if (time.perf_counter() - t_start) / 60 > cfg["budget_min"]:
                log(f"{name}: budget exceeded, stopping after epoch {epoch}")
                aborted = True
                break
        t_hat = predict(model, ev, cfg, cols, device)
        seed_std.append(t_hat)
        raw = invert(spec, off_ev, t_hat * sd + mu)
        np.save(out / f"pred_seed{seed}.npy", raw)
        seed_preds.append(raw)
        curves.append(curve)
        if save_models:
            torch.save(model.state_dict(), out / f"model_seed{seed}.pt")
        del model, opt
        empty_cache(device)
        if aborted:
            break
    if cfg["arch"] == "mlp_v14":  # V14 recipe: seed mean of the standardised prediction, then inverted
        raw_mean = invert(spec, off_ev, np.mean(np.vstack(seed_std), axis=0) * sd + mu)
    else:  # SA-NNX legs: seed mean of the raw (inverted, clipped) predictions
        raw_mean = np.mean(np.vstack(seed_preds), axis=0)
    np.save(out / "pred_mean.npy", raw_mean)
    np.save(out / "ids.npy", np.load(root / "ev_ids.npy"))
    res = {
        "name": name,
        "stage": stage,
        "split": split,
        "arch": cfg["arch"],
        "params": n_par,
        "cfg": cfg,
        "seconds": round(time.perf_counter() - t_start, 1),
        "target_mu_sd": [mu, sd],
        "curves": curves,
        "aborted": aborted,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pred_mean": str(out / "pred_mean.npy"),
    }
    if holdout:
        res["alone"] = round(rmse(label_ev, raw_mean), 4)
        res["alone_seeds"] = [round(rmse(label_ev, p), 4) for p in seed_preds]
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
        slim["curve_alone"] = [r.get("alone") for r in trained[0]] if trained else []
        slim["curve_delta"] = [r.get("delta") for r in trained[0]] if trained else []
        slim["sec_per_epoch"] = round(float(np.mean([r["sec"] for r in trained[0]])), 1) if trained else None
        with RESULTS.open("a") as fh:
            fh.write(json.dumps(slim) + "\n")
        log(f"RESULT {name}: alone={res['alone']} delta={res.get('delta')} params={n_par} sec={res['seconds']}")
    return res


def done_names(split: str) -> set[str]:
    if not RESULTS.exists():
        return set()
    out = set()
    for line in RESULTS.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("split") == split and "alone" in r:
            out.add(r["name"])
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--config")
    p.add_argument("--queue")
    p.add_argument("--stage", default="adhoc")
    p.add_argument("--out-root")
    p.add_argument("--extra-legs", default="", help="comma list of npy raw predictions added to the proxy base")
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--deterministic", action="store_true", help="CUDA: deterministic algorithms (final all-year legs)")
    p.add_argument("--cache-suffix", default=None, help="default cache suffix for configs without 'cache_suffix'")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    device = resolve_device(a.device)
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    log(f"device {device}" + (" (deterministic)" if a.deterministic else ""))
    extra = [e for e in a.extra_legs.split(",") if e]
    out_root = Path(a.out_root) if a.out_root else None
    def with_suffix(cfg: dict) -> dict:
        return {**cfg, "cache_suffix": cfg.get("cache_suffix", a.cache_suffix)} if a.cache_suffix else cfg

    if a.config:
        run(with_suffix(json.loads(a.config)), a.split, a.stage, device, out_root, extra, a.save_models)
    if a.queue:
        queue = [with_suffix(c) for c in json.loads(Path(a.queue).read_text())]
        for cfg in queue:
            if cfg["name"] in done_names(a.split) and out_root is None:
                log(f"skip {cfg['name']} (done)")
                continue
            try:
                run(cfg, a.split, cfg.get("stage", a.stage), device, out_root, extra, a.save_models)
            except Exception as exc:  # keep the queue going; record the failure
                import traceback

                traceback.print_exc()
                with RESULTS.open("a") as fh:
                    fh.write(json.dumps({"name": cfg["name"], "split": a.split, "stage": cfg.get("stage", a.stage), "error": repr(exc)[:500]}) + "\n")
                empty_cache(device)


if __name__ == "__main__":
    main()
