# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Neural leg (V14, SA-NN): embedding MLP on an assembled S4 matrix.

Pre-registration: ``progress-review/neural-leg-20260926/preregistration.md``.  Output contract
mirrors ``scripts/legs/train_lgbm_matrix.py``: janjul runs write
``predictions_<tag>_janjul.parquet`` (``ma.score_janjul``: E17 -> E18 -> E33b freeze -> clip,
``s4_raw_prediction`` = the raw network prediction), ranking runs write
``ranking_nn_<tag>.parquet`` in the trainer's ranking schema (``ma.compose_ranking``), plus a
``run_manifest.json``.

Recipe (fixed by the pre-registration; the defaults below *are* the recipe):

* target: AOBT-delta reparameterisation ``t = BLOCK - AOBT`` (``s4.TargetSpec`` offset
  ``takeoff_minus_aobt_sec``, sign -1, fill ``unimpeded_p10``, winsor 7200 on the fit label),
  standardised with the fit rows' mean / std; prediction ``max(0, offset - t_hat)``.
* categoricals: vocabulary of levels seen >= 20 times in the fit rows (rarer / null / unseen ->
  index 0), embedding dim ``min(24, max(4, round(1.6 * n**0.56)))``.
* numerics: per-column quantile-normal map (1001 quantiles from a 400k fit-row sample, sklearn's
  two-sided interpolation, bounds 1e-7), NaN -> 0, plus a NaN indicator for every numeric with a
  NaN in the fit rows.
* network: [Linear 512, BN, SiLU, Dropout .1] x 2, [Linear 256, BN, SiLU, Dropout .1], Linear 1.
* AdamW lr 2e-3, weight decay 1e-4, batch 4096, 300-step linear warm-up then cosine to 2% of
  peak over the epoch budget, MSE on the standardised target, grad-norm clip 1.0.
* stage A: fit months != holdout (10), budget 20 epochs, holdout taxi RMSE each epoch,
  e* = argmin.  Stage B: refit on every training month for e* epochs (cosine over e*), seeds
  0, 1, 2, t_hat averaged over seeds, then inverted.  ``--fixed-epochs`` skips stage A.

Threads: torch / polars limited to ``--threads`` (default 3); the network runs on MPS when
available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

_THREADS = "3"
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        _THREADS = sys.argv[_i + 1]
for _v in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.special import ndtri  # noqa: E402

from prc_challenge import matrix_arms as ma  # noqa: E402
from prc_challenge import s4_allyear as s4  # noqa: E402

TARGETS = {
    # V14 (default): AOBT-delta, t = BLOCK - AOBT
    "delta": s4.TargetSpec(offset_col="takeoff_minus_aobt_sec", sign=-1, fill_col="unimpeded_p10", winsor_sec=7200.0),
    # V15: S4 plain residual t = y - unimpeded_p10, fit label winsorised at +-7200 as V14's
    "resid": s4.TargetSpec(offset_col="unimpeded_p10", sign=1, fill_col=None, winsor_sec=7200.0),
    # V16: EOBT residual as V10/V11, t = y - (MVT - EOBT)
    "eobtres": s4.TargetSpec(offset_col="takeoff_minus_eobt_sec", sign=1, fill_col="unimpeded_p10", winsor_sec=7200.0),
}
TARGET = TARGETS["delta"]
MIN_LEVEL_COUNT = 20
N_QUANTILES = 1001
QUANTILE_SAMPLE = 400_000
BOUND = 1e-7


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message} (peak_rss={s4.peak_rss_gb():.2f} GB)", flush=True)


# --------------------------------------------------------------------------- preprocessing


class Preprocessor:
    """Train-only categorical vocabularies and quantile-normal numeric maps."""

    def __init__(self, cats: list[str], nums: list[str]):
        self.cats = cats
        self.nums = nums
        self.vocab: dict[str, dict[str, int]] = {}
        self.quantiles: dict[str, np.ndarray] = {}
        self.nan_cols: list[str] = []

    def fit(self, frame: pl.DataFrame, seed: int = 0) -> "Preprocessor":
        for c in self.cats:
            counts = frame.select(pl.col(c).cast(pl.String).alias("v")).group_by("v").len()
            keep = sorted(v for v, n in counts.iter_rows() if v is not None and n >= MIN_LEVEL_COUNT)
            self.vocab[c] = {v: i + 1 for i, v in enumerate(keep)}
        sample = frame.sample(n=min(QUANTILE_SAMPLE, frame.height), seed=seed) if frame.height > QUANTILE_SAMPLE else frame
        refs = np.linspace(0.0, 1.0, N_QUANTILES)
        for c in self.nums:
            v = sample[c].cast(pl.Float64).to_numpy()
            v = v[np.isfinite(v)]
            self.quantiles[c] = np.quantile(v, refs) if v.size else np.zeros(N_QUANTILES)
        null_rates = frame.select([(pl.col(c).cast(pl.Float64).is_null() | pl.col(c).cast(pl.Float64).is_nan()).any().alias(c) for c in self.nums]).row(0)
        self.nan_cols = [c for c, has in zip(self.nums, null_rates) if has]
        return self

    @property
    def sizes(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in self.cats]

    @property
    def n_dense(self) -> int:
        return len(self.nums) + len(self.nan_cols)

    def transform(self, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = frame.height
        cat = np.zeros((n, len(self.cats)), dtype=np.int32)
        for j, c in enumerate(self.cats):
            cat[:, j] = (
                frame.select(pl.col(c).cast(pl.String).replace_strict(self.vocab[c], default=0, return_dtype=pl.Int32))
                .to_series()
                .fill_null(0)
                .to_numpy()
            )
        dense = np.zeros((n, self.n_dense), dtype=np.float16)
        refs = np.linspace(0.0, 1.0, N_QUANTILES)
        nan_index = {c: len(self.nums) + k for k, c in enumerate(self.nan_cols)}
        for j, c in enumerate(self.nums):
            x = frame[c].cast(pl.Float64).to_numpy().copy()
            bad = ~np.isfinite(x)
            q = self.quantiles[c]
            if q[-1] > q[0]:
                xs = np.where(bad, q[0], x)
                u = 0.5 * (np.interp(xs, q, refs) - np.interp(-xs, -q[::-1], -refs[::-1]))
                z = ndtri(np.clip(u, BOUND, 1 - BOUND))
            else:
                z = np.zeros(n)
            z[bad] = 0.0
            dense[:, j] = z.astype(np.float16)
            if c in nan_index:
                dense[:, nan_index[c]] = bad.astype(np.float16)
        return cat, dense

    def describe(self) -> dict[str, Any]:
        return {
            "vocab_sizes": {c: len(self.vocab[c]) + 1 for c in self.cats},
            "n_numeric": len(self.nums),
            "n_nan_indicators": len(self.nan_cols),
            "n_dense": self.n_dense,
            "min_level_count": MIN_LEVEL_COUNT,
            "n_quantiles": N_QUANTILES,
            "quantile_sample": QUANTILE_SAMPLE,
        }


def fit_target(frame: pl.DataFrame) -> np.ndarray:
    return TARGET.with_fit_label(frame)[s4.RESIDUAL_LABEL].to_numpy()


# --------------------------------------------------------------------------- network


def emb_dim(n: int) -> int:
    return int(min(24, max(4, round(1.6 * n**0.56))))


def build_model(sizes: list[int], n_dense: int, widths: tuple[int, ...], dropout: float):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(n, emb_dim(n)) for n in sizes])
            prev = sum(emb_dim(n) for n in sizes) + n_dense
            layers: list[nn.Module] = []
            for w in widths:
                layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.SiLU(), nn.Dropout(dropout)]
                prev = w
            layers.append(nn.Linear(prev, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, cat: "torch.Tensor", dense: "torch.Tensor") -> "torch.Tensor":
            parts = [e(cat[:, i]) for i, e in enumerate(self.embs)]
            return self.mlp(torch.cat([*parts, dense], dim=1)).squeeze(-1)

    return Net()


def resolve_device(name: str):
    """``auto``: cuda if available, else mps, else cpu; an explicit name is used as is."""

    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def empty_cache(device) -> None:
    import torch

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


class Trainer:
    def __init__(self, args: argparse.Namespace, device):
        self.args = args
        self.device = device

    def to_device(self, cat: np.ndarray, dense: np.ndarray):
        import torch

        c, d = torch.from_numpy(cat.astype(np.int64)), torch.from_numpy(dense)
        if self.device.type == "cuda":  # bake: pinned host memory for the one-off block copies
            return c.pin_memory().to(self.device, non_blocking=True), d.pin_memory().to(self.device, non_blocking=True)
        return c.to(self.device), d.to(self.device)

    def predict(self, model, cat_t, dense_t) -> np.ndarray:
        import torch

        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, cat_t.shape[0], 65536):
                out.append(model(cat_t[s : s + 65536], dense_t[s : s + 65536].float()).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)

    def fit(self, sizes, n_dense, cat_t, dense_t, y_std: np.ndarray, epochs: int, seed: int, on_epoch=None):
        import torch

        a = self.args
        torch.manual_seed(seed)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        model = build_model(sizes, n_dense, tuple(a.widths), a.dropout).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        n = cat_t.shape[0]
        steps_per_epoch = math.ceil(n / a.batch_size)
        total = steps_per_epoch * epochs
        warm = a.warmup_steps

        def factor(step: int) -> float:
            w = min(1.0, (step + 1) / warm)
            return w * (a.lr_floor + (1 - a.lr_floor) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
        y_t = torch.from_numpy(y_std.astype(np.float32)).to(self.device)
        history = []
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            model.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            running = torch.zeros((), device=self.device)
            for s in range(0, n, a.batch_size):
                idx = perm[s : s + a.batch_size]
                if idx.numel() < 2:
                    continue
                pred = model(cat_t[idx], dense_t[idx].float())
                loss = torch.mean((pred - y_t[idx]) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                sched.step()
                running += loss.detach() * idx.numel()
            train_mse = float(running.cpu()) / n
            rec = {"epoch": epoch, "train_mse_std": round(train_mse, 5), "seconds": round(time.perf_counter() - t0, 1), "lr_end": opt.param_groups[0]["lr"]}
            if on_epoch is not None:
                rec.update(on_epoch(model))
            history.append(rec)
            log(f"seed {seed} epoch {epoch}/{epochs}: " + json.dumps(rec))
        return model, history


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--target", choices=sorted(TARGETS), default="delta", help="delta = V14 (default), resid = V15, eobtres = V16")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", help="output directory (default: <run-dir>/../<tag>)")
    p.add_argument("--source-root")
    p.add_argument("--control", help="janjul: run_manifest.json of a control fit for the delta columns")
    p.add_argument("--compare", help="ranking: a trainer-schema ranking parquet to compare with")
    p.add_argument("--holdout-month", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=20, help="stage A epoch budget")
    p.add_argument("--fixed-epochs", type=int, help="skip stage A; stage B with this many epochs")
    p.add_argument("--stage-a-only", action="store_true", help="smoke / selection only: no stage B, no evaluation rows touched")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--widths", type=lambda s: [int(x) for x in s.split(",")], default=[512, 512, 256])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-floor", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--device", default="auto", help="auto = cuda, else mps, else cpu (on the Mac: the previous mps/cpu choice)")
    p.add_argument("--deterministic", action="store_true", help="CUDA: torch.use_deterministic_algorithms (final all-year legs)")
    p.add_argument("--max-train-rows", type=int, help="smoke tests only: subsample the training matrix")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    global TARGET
    TARGET = TARGETS[args.target]

    import torch

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    src = s4.resolve_source_root(args.source_root)
    run = Path(args.run_dir).resolve()
    out = Path(args.out).resolve() if args.out else run.parent / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if args.holdout_month in ma.FORBIDDEN_TRAIN_MONTHS:
        raise SystemExit(f"holdout month {args.holdout_month} is an evaluation/locked month")
    seeds = [int(s) for s in args.seeds.split(",")]
    t_start = time.perf_counter()
    stage_seconds: dict[str, float] = {}

    train, evaluation, cats, nums, matrix_info = ma.load_run_matrices(run)
    mode = matrix_info["eval_mode"]
    if args.stage_a_only:
        del evaluation
        evaluation = None
    if args.max_train_rows and train.height > args.max_train_rows:
        train = train.sample(n=args.max_train_rows, seed=0, shuffle=True)
        log(f"SMOKE TEST: training matrix subsampled to {train.height} rows")
    keep_cols = list(dict.fromkeys([s4.LABEL_COLUMN, "month", *TARGET.columns(), *cats, *nums]))
    train = train.select(keep_cols)
    log(f"matrices: mode={mode} train={train.shape} cats={len(cats)} nums={len(nums)} months={matrix_info['train_months']} device={device}")
    stage_seconds["load"] = time.perf_counter() - t_start
    trainer = Trainer(args, device)
    common = {
        "target": TARGET.as_dict() | {"standardised": True, "name": args.target},
        "widths": args.widths,
        "dropout": args.dropout,
        "lr": args.lr,
        "lr_floor": args.lr_floor,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "grad_clip": args.grad_clip,
        "loss": "mse (standardised winsorised target)",
    }

    # ---------------- stage A: epoch selection on the holdout month
    stage_a_info: dict[str, Any] | None = None
    info_path = out / "stageA_info.json"
    if args.fixed_epochs is None:
        if args.holdout_month not in matrix_info["train_months"]:
            raise SystemExit("holdout month not in training months")
        if info_path.exists() and not args.force and not args.stage_a_only:
            stage_a_info = json.loads(info_path.read_text())
            log(f"reusing stage A: best_epoch={stage_a_info['best_epoch']}")
        else:
            t1 = time.perf_counter()
            fit_frame = train.filter(pl.col("month") != args.holdout_month)
            hold_frame = train.filter(pl.col("month") == args.holdout_month)
            pre = Preprocessor(cats, nums).fit(fit_frame)
            y_fit = fit_target(fit_frame)
            mu, sd = float(y_fit.mean()), float(y_fit.std())
            cat_fit, dense_fit = pre.transform(fit_frame)
            cat_hold, dense_hold = pre.transform(hold_frame)
            hold_label = hold_frame[s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
            hold_inv = hold_frame.select(TARGET.columns())
            del fit_frame, hold_frame
            ct, dt = trainer.to_device(cat_fit, dense_fit)
            ch, dh = trainer.to_device(cat_hold, dense_hold)
            del cat_fit, dense_fit, cat_hold, dense_hold
            log(f"stage A blocks: fit={tuple(dt.shape)} holdout={tuple(dh.shape)} target mean={mu:.2f} sd={sd:.2f} pre={json.dumps(pre.describe())}")

            def on_epoch(model) -> dict[str, float]:
                t_hat = trainer.predict(model, ch, dh) * sd + mu
                return {"holdout_taxi_rmse": round(ma.rmse(hold_label, TARGET.predict(hold_inv, t_hat)), 4)}

            model, history = trainer.fit(pre.sizes, pre.n_dense, ct, dt, (y_fit - mu) / sd, args.max_epochs, seeds[0], on_epoch)
            curve = [h["holdout_taxi_rmse"] for h in history]
            best = int(np.argmin(curve)) + 1
            stage_a_info = {
                "holdout_month": args.holdout_month,
                "fit_rows": int(ct.shape[0]),
                "holdout_rows": int(ch.shape[0]),
                "max_epochs": args.max_epochs,
                "seed": seeds[0],
                "best_epoch": best,
                "holdout_taxi_rmse_at_best": curve[best - 1],
                "history": history,
                "preprocessor": pre.describe(),
                "seconds": round(time.perf_counter() - t1, 1),
                "smoke_subsample": args.max_train_rows,
            } | common
            torch.save(model.state_dict(), out / "model_stageA.pt")
            s4.atomic_json(info_path, stage_a_info)
            del model, ct, dt, ch, dh
            empty_cache(device)
            log(f"stage A done: best_epoch={best} holdout_taxi_rmse={curve[best - 1]:.4f}")
            stage_seconds["stage_a"] = time.perf_counter() - t1
        epochs = int(stage_a_info["best_epoch"])
    else:
        epochs = int(args.fixed_epochs)
        log(f"fixed epochs: {epochs} (no stage A)")
    if args.stage_a_only:
        log("stage-A-only run: stopping before stage B / evaluation")
        return

    # ---------------- stage B: all training months, e* epochs, seed ensemble
    t2 = time.perf_counter()
    pre = Preprocessor(cats, nums).fit(train)
    y_all = fit_target(train)
    mu, sd = float(y_all.mean()), float(y_all.std())
    cat_all, dense_all = pre.transform(train)
    n_train = train.height
    del train
    ct, dt = trainer.to_device(cat_all, dense_all)
    del cat_all, dense_all
    cat_ev, dense_ev = pre.transform(evaluation)
    ce, de = trainer.to_device(cat_ev, dense_ev)
    del cat_ev, dense_ev
    log(f"stage B blocks: train={tuple(dt.shape)} eval={tuple(de.shape)} target mean={mu:.2f} sd={sd:.2f}")
    t_hats = []
    seed_info = []
    for seed in seeds:
        model_path = out / f"model_stageB_seed{seed}.pt"
        pred_path_seed = out / f"eval_that_seed{seed}.npy"
        if model_path.exists() and pred_path_seed.exists() and not args.force:
            t_hat = np.load(pred_path_seed)
            log(f"reusing seed {seed}")
            seed_info.append({"seed": seed, "reused": True, "model": str(model_path), "model_sha256": s4.sha256_file(model_path)})
        else:
            model, history = trainer.fit(pre.sizes, pre.n_dense, ct, dt, (y_all - mu) / sd, epochs, seed)
            t_hat = trainer.predict(model, ce, de) * sd + mu
            torch.save(model.state_dict(), model_path)
            np.save(pred_path_seed, t_hat)
            seed_info.append({"seed": seed, "history": history, "model": str(model_path), "model_sha256": s4.sha256_file(model_path)})
            del model
        t_hats.append(t_hat)
    stage_seconds["stage_b"] = time.perf_counter() - t2
    t_mean = np.mean(np.vstack(t_hats), axis=0)
    raw = TARGET.predict(evaluation, t_mean)
    seed_raw = [TARGET.predict(evaluation, t) for t in t_hats]
    seed_pairs = {f"{seeds[i]}|{seeds[j]}": s4.rowwise_comparison(seed_raw[i], seed_raw[j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))}

    manifest: dict[str, Any] = {
        "arm": "V14 neural leg: embedding MLP on the S4 matrix (SA-NN)",
        "tag": args.tag,
        "created_utc": s4.now_utc(),
        "eval_mode": mode,
        "recipe": common | {"seeds": seeds, "epochs": epochs, "epochs_source": "stage A best epoch" if args.fixed_epochs is None else "fixed", "seed_combination": "mean of standardised-target predictions, then inverted"},
        "matrix": matrix_info,
        "features": {"cats": cats, "nums": nums, "n": len(cats) + len(nums), "feature_schema_sha256": s4.json_sha256([cats, nums])},
        "preprocessor": pre.describe(),
        "stage_a": stage_a_info,
        "stage_b": {"rows": n_train, "target_mean": mu, "target_sd": sd, "seeds": seed_info},
        "seed_pairwise_raw": seed_pairs,
        "device": str(device),
        "threads": args.threads,
        "smoke_subsample": args.max_train_rows,
        "nov_dec_labels_used": matrix_info["nov_dec_labels_used"],
        "nov_dec_role": matrix_info["nov_dec_role"],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    if mode == "janjul":
        scored, card = ma.score_janjul(evaluation, raw, src)
        pred_path = out / f"predictions_{args.tag}_janjul.parquet"
        manifest["predictions"] = ma.write_janjul_predictions(scored, pred_path)
        control = ma.load_control_card(Path(args.control)) if args.control else None
        manifest["scorecard"] = card
        manifest["control"] = {"path": args.control, "scorecard": control} if control else None
        manifest["vs_control"] = ma.card_vs_control(card, control)
    else:
        composed, summary = ma.compose_ranking(evaluation, raw)
        pred_path = out / f"ranking_nn_{args.tag}.parquet"
        s4.atomic_parquet(composed, pred_path)
        manifest["predictions"] = {"path": str(pred_path), "sha256": s4.sha256_file(pred_path), **summary}
        manifest["composition"] = {"order": ["E17", "E18", "clip>=0"], "note": "unmatched rows are owned by the unmatched overlay; blend on s4_raw_prediction of matched rows"}
        if args.compare:
            manifest["vs_reference"] = ma.compare_ranking(composed, Path(args.compare))
    manifest["stage_seconds"] = {k: round(v, 1) for k, v in stage_seconds.items()}
    manifest["peak_rss_gb"] = s4.peak_rss_gb()
    s4.atomic_json(out / "run_manifest.json", manifest)
    if mode == "janjul":
        log(ma.card_line(manifest["scorecard"]))
    else:
        pinfo = manifest["predictions"]
        log(f"ranking: n={pinfo['n']} matched={pinfo['matched_n']} matched_mean={pinfo['matched_pred_mean']:.3f} checksum={pinfo['matched_checksum'][:12]}")
        if manifest.get("vs_reference"):
            r = manifest["vs_reference"]
            log(f"vs reference (matched): r={r['pearson_r']:.6f} mean|d|={r['mean_abs_diff']:.3f}")
    log(f"run manifest written: {out / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
