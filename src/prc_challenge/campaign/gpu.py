# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""GPU campaign jobs: sequence residual models, not F0 tabular repeats."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prc_challenge.bundles import E37_FIT_MONTHS, E37_TUNE_MONTHS
from prc_challenge.campaign.data import feature_lists, load_parent, matched_train_val
from prc_challenge.campaign.data_check import hydrate_from_drive, locate
from prc_challenge.campaign.ensemble import CPU_BLEND_IDS, GPU_BLEND_IDS, blend_completed_candidates
from prc_challenge.campaign.f3 import f3_numeric_columns, load_or_build_f3
from prc_challenge.campaign.models import fit_regressor, load_or_build_base_residual, month_blocked_oof, predict_regressor
from prc_challenge.campaign.runtime import (
    CampaignConfig,
    campaign_job_catalog,
    extra_view_exception,
    inventory_status,
    is_terminal_skip,
    persist_dir,
    persist_repo_file,
    print_campaign_status,
    prior_view_metrics,
    run_isolated,
    sync_index_from_job_dirs,
    write_campaign_summary,
)
from prc_challenge.campaign.scoring import compact_score, score_and_write
from prc_challenge.campaign.sequences import gather_sequence, seq_dep_frame
from prc_challenge.metrics import rmse
from prc_challenge.nn_encode import fit_num_scaler, transform_nums


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sanitize_np(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def require_cuda(cfg: CampaignConfig) -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.allow_cpu_smoke_on_gpu_notebook or cfg.smoke:
        return torch.device("cpu")
    raise RuntimeError("GPU notebook requires CUDA. Set allow_cpu_smoke_on_gpu_notebook=True only for local smoke.")


def _residual_frame(
    cfg: CampaignConfig, extra_cols: list[str] | tuple[str, ...] | None = None
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, str]:
    return load_or_build_base_residual(cfg, extra_cols=extra_cols)


def _f3_payload(cfg: CampaignConfig) -> pl.DataFrame:
    """F3 numeric table only — not joined onto the full e08b cache."""
    from prc_challenge.campaign.f3 import f3_cache_path, load_or_build_f3, read_cached_f3

    cached = read_cached_f3(cfg)
    if cached is not None:
        return cached
    path = f3_cache_path(cfg)
    if not cfg.allow_rebuild_cpu_caches:
        if path.exists() and not cfg.smoke:
            print(f"using existing F3 without rebuild -> {path}", flush=True)
            return pl.read_parquet(path)
        raise FileNotFoundError(
            f"need {path.as_posix()} from the CPU campaign; GPU will not rebuild F3"
        )
    from prc_challenge.campaign.data import load_cache_columns

    cache = load_cache_columns(cfg)
    keys = [
        c
        for c in (
            "MVT_ID_mvt",
            "AIRPORT",
            "PHASE_mvt",
            "MVT_TIME_UTC_mvt",
            "RUNWAY_mvt",
            "SCHED_TIME_UTC_mvt",
            "month",
        )
        if c in cache.columns
    ]
    built = load_or_build_f3(cfg, cache.select(keys))
    del cache
    keep = ["MVT_ID_mvt", *[c for c in f3_numeric_columns() if c in built.columns]]
    return built.select(keep)


def _internal_masks(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit on E37 months 2–6+8; early-stop / weights on month 9. Never official V1/V2/V3."""
    n = df.height
    if "month" in df.columns:
        fit_m = df["month"].is_in(list(E37_FIT_MONTHS)).to_numpy()
        tune_m = df["month"].is_in(list(E37_TUNE_MONTHS)).to_numpy()
        if int(fit_m.sum()) >= 20 and int(tune_m.sum()) >= 8:
            return fit_m, tune_m
    cut = max(1, int(n * 0.85))
    fit_m = np.zeros(n, dtype=bool)
    fit_m[:cut] = True
    return fit_m, ~fit_m


def _write_tune_preds(dest: Path, frame: pl.DataFrame, pred: np.ndarray, mask: np.ndarray) -> None:
    if int(mask.sum()) == 0:
        return
    keep = [c for c in ("MVT_ID_mvt", "AIRPORT", "TAXITIME_SEC_mvt", "month") if c in frame.columns]
    frame.select(keep).with_columns(pl.Series("prediction", np.asarray(pred, dtype=np.float64))).filter(
        pl.Series(mask)
    ).write_parquet(Path(dest) / "tune_predictions.parquet")


COMPOSE_FLAG_COLS = (
    "nm_unmatched",
    "takeoff_minus_sched_days",
    "type_missing",
    "takeoff_minus_sched_sec",
    "TAXITIME_SEC_mvt",
    "AIRPORT",
    "month",
    "MVT_TIME_UTC_mvt",
)


def _ensure_compose_cols(cfg: CampaignConfig, val: pl.DataFrame) -> pl.DataFrame:
    """Join E17/E18/unmatched flags if the sequence ordered table dropped them."""
    missing = [c for c in COMPOSE_FLAG_COLS if c not in val.columns]
    if not missing or "MVT_ID_mvt" not in val.columns:
        return val
    from prc_challenge.campaign.data import load_cache_columns

    flags = load_cache_columns(cfg, extra=missing)
    keep = ["MVT_ID_mvt", *[c for c in missing if c in flags.columns]]
    if len(keep) <= 1:
        return val
    return val.join(flags.select(keep).unique("MVT_ID_mvt", keep="first"), on="MVT_ID_mvt", how="left")


def _compose(cfg: CampaignConfig, view: str, val: pl.DataFrame, pred: np.ndarray, dest=None) -> dict[str, Any]:
    parent = load_parent(cfg, view)
    val = _ensure_compose_cols(cfg, val)
    try:
        from prc_challenge.campaign.data import load_base_preds

        base = load_base_preds(cfg, view)
    except FileNotFoundError:
        base = None
    if dest is None:
        from prc_challenge.campaign.scoring import compose_with_parent, slice_table

        scored = val.with_columns(pl.Series("prediction", pred).clip(lower_bound=0))
        scored = compose_with_parent(
            scored.join(parent.select(["MVT_ID_mvt", pl.col("prediction").alias("parent_prediction")]), on="MVT_ID_mvt", how="left")
        )
        return slice_table(scored)
    return score_and_write(val, pred, dest, view=view, parent=parent, base=base)


def _hydrate_extra_view(cfg: CampaignConfig, dest, view: str) -> dict[str, Any]:
    """Keep composed RMSE after a reconnect. Prefer prior metrics; else re-score the parquet (no refit)."""
    dest = Path(dest)
    prior = prior_view_metrics(dest, view)
    if isinstance(prior, dict) and prior.get("rmse_seconds") is not None:
        return compact_score(prior)
    path = dest / f"predictions_{view}.parquet"
    if not path.exists():
        return {"status": "failed", "error": f"{view} extra-view parquet is missing"}
    from prc_challenge.campaign.data import load_cache_columns

    _, val = matched_train_val(load_cache_columns(cfg), view)
    tbl = pl.read_parquet(path)
    if "MVT_ID_mvt" not in tbl.columns or "prediction" not in tbl.columns or val.height == 0:
        return {"status": "failed", "error": f"{view} extra-view parquet does not cover the split"}
    val = val.join(tbl.select(["MVT_ID_mvt", "prediction"]), on="MVT_ID_mvt", how="left")
    if val["prediction"].null_count():
        return {"status": "failed", "error": f"{view} extra-view parquet does not cover the split"}
    pred = val["prediction"].to_numpy()
    try:
        scored = _compose(cfg, view, val, pred, dest=dest)
        return compact_score(scored) if scored.get("rmse_seconds") is not None else scored
    except Exception as exc:
        print(f"extra view {view} resume re-score failed: {exc!r}", flush=True)
        return {"status": "failed", "error": repr(exc)}


def _resumed_extra_views(cfg: CampaignConfig, dest, primary: str) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for view in cfg.views:
        if view == primary:
            continue
        extra[view] = _hydrate_extra_view(cfg, dest, view)
        print(f"extra view {view} resume rmse={extra[view].get('rmse_seconds')}", flush=True)
    return extra


def _extra_view_ready(dest, view: str) -> bool:
    dest = Path(dest)
    if not (dest / f"predictions_{view}.parquet").exists():
        return False
    prior = prior_view_metrics(dest, view)
    return isinstance(prior, dict) and _finite(prior.get("rmse_seconds"))


class _SeqRegressor:
    def __init__(self, kind: str, seq_dim: int, n_tab: int, device, hidden: int = 64, mixture: bool = False):
        import torch
        from torch import nn

        self.device = device
        self.kind = kind
        self.mixture = mixture
        d_model = hidden
        nhead = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else 2)
        if kind == "gru":
            n_layers = 1 if hidden <= 64 else 2
            self.enc = nn.GRU(seq_dim, hidden, num_layers=n_layers, batch_first=True).to(device)
            self.out_dim = hidden
        elif kind == "tcn":
            self.enc = nn.Sequential(
                nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
            ).to(device)
            self.out_dim = hidden
        else:
            self.proj = nn.Linear(seq_dim, d_model).to(device)
            n_layers = 2 if hidden <= 64 else 4
            ffn = 128 if hidden <= 64 else 512
            layer_kw = {"d_model": d_model, "nhead": nhead, "dim_feedforward": ffn, "dropout": 0.1}
            try:
                layer = nn.TransformerEncoderLayer(**layer_kw, batch_first=True)
                self._tf_batch_first = True
            except TypeError:
                layer = nn.TransformerEncoderLayer(**layer_kw)
                self._tf_batch_first = False
            try:
                self.enc = nn.TransformerEncoder(
                    layer, num_layers=n_layers, enable_nested_tensor=False
                ).to(device)
            except TypeError:
                self.enc = nn.TransformerEncoder(layer, num_layers=n_layers).to(device)
            self.out_dim = d_model
        out = 6 if mixture else 1
        self.head = nn.Sequential(
            nn.Linear(self.out_dim + n_tab, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        ).to(device)
        self.hidden = hidden
        self.seq_dim = seq_dim
        self.in_scale = nn.Parameter(torch.ones(seq_dim, device=device))
        self.in_bias = nn.Parameter(torch.zeros(seq_dim, device=device))
        # Component scales (seconds) so the extreme regime can keep a heavier tail.
        self.log_sigma = nn.Parameter(torch.tensor([4.0, 4.7, 5.5], device=device)) if mixture else None

    def encode(self, seq, mask):
        import torch

        seq = torch.nan_to_num(seq.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mu = seq.mean(dim=-1, keepdim=True)
        var = seq.var(dim=-1, keepdim=True, unbiased=False)
        seq = (seq - mu) * torch.rsqrt(var.clamp_min(1e-6)) * self.in_scale + self.in_bias
        if self.kind == "gru":
            packed = seq * mask.unsqueeze(-1)
            _, h = self.enc(packed)
            return h[-1]
        if self.kind == "tcn":
            x = seq.transpose(1, 2)
            h = self.enc(x)
            w = mask.unsqueeze(1).to(dtype=h.dtype)
            return (h * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(1.0)
        x = self.proj(seq)
        pad = mask < 0.5
        if getattr(self, "_tf_batch_first", True):
            h = self.enc(x, src_key_padding_mask=pad)
        else:
            h = self.enc(x.transpose(0, 1), src_key_padding_mask=pad).transpose(0, 1)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def mix_parts(self, seq, mask, tab):
        import torch

        h = self.encode(seq, mask)
        z = self.head(torch.cat([h, tab], dim=1)).float()
        log_p = torch.log_softmax(z[:, :3], dim=1)
        means = z[:, 3:]
        return log_p, means

    def mix_nll(self, seq, mask, tab, y):
        """3-component Gaussian mixture NLL. RMSE still uses the expected value from __call__."""
        import torch

        log_p, means = self.mix_parts(seq, mask, tab)
        yb = y.float().unsqueeze(1)
        sigma = self.log_sigma.exp().clamp(8.0, 800.0)
        log_norm = -0.5 * ((yb - means) ** 2) / (sigma ** 2) - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)
        return -(torch.logsumexp(log_p + log_norm, dim=1)).mean()

    def __call__(self, seq, mask, tab):
        import torch

        if not self.mixture:
            h = self.encode(seq, mask)
            z = self.head(torch.cat([h, tab], dim=1))
            return z.squeeze(-1)
        log_p, means = self.mix_parts(seq, mask, tab)
        return (log_p.exp() * means).sum(dim=1)

    def parameters(self):
        import itertools

        params = [self.head.parameters(), self.enc.parameters(), (self.in_scale, self.in_bias)]
        if hasattr(self, "proj"):
            params.append(self.proj.parameters())
        if self.log_sigma is not None:
            params.append((self.log_sigma,))
        return itertools.chain.from_iterable(params)

    def weights(self) -> dict[str, Any]:
        blob: dict[str, Any] = {
            "kind": self.kind,
            "hidden": self.hidden,
            "enc": self.enc.state_dict(),
            "head": self.head.state_dict(),
            "in_norm": {"scale": self.in_scale.detach().cpu(), "bias": self.in_bias.detach().cpu()},
            "mixture": self.mixture,
            "seq_norm": "layer",
        }
        if hasattr(self, "proj"):
            blob["proj"] = self.proj.state_dict()
        if self.log_sigma is not None:
            blob["log_sigma"] = self.log_sigma.detach().cpu()
        return blob

    def load_weights(self, blob: dict[str, Any]) -> None:
        self.enc.load_state_dict(blob["enc"])
        self.head.load_state_dict(blob["head"])
        if "in_norm" in blob:
            spec = blob["in_norm"]
            if isinstance(spec, dict) and "scale" in spec and "bias" in spec:
                self.in_scale.data.copy_(spec["scale"].to(self.in_scale.device))
                self.in_bias.data.copy_(spec["bias"].to(self.in_bias.device))
        if "proj" in blob and hasattr(self, "proj"):
            self.proj.load_state_dict(blob["proj"])
        if self.log_sigma is not None and blob.get("log_sigma") is not None:
            self.log_sigma.data.copy_(blob["log_sigma"].to(self.log_sigma.device))


def _ensure_seq_tab_cols(frame: pl.DataFrame, tab_cols: list[str]) -> pl.DataFrame:
    """Keep fusion tabular channels aligned on extra official views (BASE pred, not zeros)."""
    if "oof_prediction" in tab_cols and "oof_prediction" not in frame.columns:
        if "base_prediction" in frame.columns:
            frame = frame.with_columns(pl.col("base_prediction").alias("oof_prediction"))
    for col in tab_cols:
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(0.0).alias(col))
    return frame


def _align_numeric(frame: pl.DataFrame, ordered: pl.DataFrame, col: str, fallback: float) -> np.ndarray:
    """Join a DEP column onto mixed sequence rows without a Python id dict."""
    if col not in frame.columns or "MVT_ID_mvt" not in ordered.columns:
        return np.full(ordered.height, float(fallback), dtype=np.float32)
    right = frame.select(["MVT_ID_mvt", pl.col(col).cast(pl.Float64)]).unique("MVT_ID_mvt", keep="first")
    joined = ordered.select("MVT_ID_mvt").join(right, on="MVT_ID_mvt", how="left")
    return joined[col].fill_null(float(fallback)).to_numpy().astype(np.float32, copy=False)


def _tab_rows(
    source: pl.DataFrame,
    tab_cols: list[str],
    scaler,
    ordered: pl.DataFrame,
    rows: np.ndarray,
    id_arr: np.ndarray | None = None,
) -> np.ndarray:
    """Scale tabular features for a minibatch of mixed-sequence rows.

    GPU-04 fusion has ~150 columns. Materializing that matrix for every mixed
    ARR+DEP row (~2.5M) OOMs a 16 GB Colab GPU host. Only DEP target rows need
    the fusion head, and only 64–256 of those are in a batch.
    """
    rows = np.asarray(rows, dtype=np.int64)
    n_tab = len(tab_cols)
    if len(rows) == 0:
        return np.zeros((0, n_tab), dtype=np.float32)
    if not tab_cols:
        return np.zeros((len(rows), 0), dtype=np.float32)
    ids = (id_arr if id_arr is not None else ordered["MVT_ID_mvt"].to_numpy())[rows]
    sub = pl.DataFrame({"MVT_ID_mvt": ids}).join(source, on="MVT_ID_mvt", how="left")
    missing = [c for c in tab_cols if c not in sub.columns]
    if missing:
        sub = sub.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in missing])
    return _sanitize_np(transform_nums(sub, tab_cols, scaler))


def _dep_tab_source(source: pl.DataFrame, tab_cols: list[str]) -> pl.DataFrame:
    """Compact unique DEP table for per-batch fusion joins. Call once per split."""
    if "MVT_ID_mvt" not in source.columns:
        return pl.DataFrame({"MVT_ID_mvt": []})
    exprs: list[pl.Expr] = [pl.col("MVT_ID_mvt")]
    exprs.extend(pl.col(c).cast(pl.Float32) for c in tab_cols if c in source.columns)
    return source.select(exprs).unique("MVT_ID_mvt", keep="first")


def _event_pack(cfg: CampaignConfig, dep: pl.DataFrame, *, past: int, future: int, tag: str = "seq"):
    from prc_challenge.campaign.sequences import load_or_build_event_pack

    return load_or_build_event_pack(cfg, dep, past=past, future=future, tag=tag)


def _minibatches(n: int, batch: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for start in range(0, n, batch):
        yield idx[start : start + batch]


class SeqMinBatchOom(RuntimeError):
    """CUDA OOM at the smallest allowed batch. Park the job; do not retry overnight."""


def _is_cuda_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "out of memory" in text or "outofmemory" in text


def _seq_oom_skip(exc: SeqMinBatchOom) -> dict[str, Any]:
    return {"status": "skipped", "reason": str(exc), "conclusion": "PARK"}


def _raise_min_batch_oom(exc: BaseException, *, batch: int, label: str) -> None:
    if _is_cuda_oom(exc) and batch <= 32:
        raise SeqMinBatchOom(f"{label} CUDA OOM at batch={batch}; sequence job parked") from exc


def _cuda_empty() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _amp_dtype(device):
    """bf16 on CUDA when the card supports it. fp16 overflowed unscaled sequence tokens."""
    import torch

    if getattr(device, "type", None) != "cuda":
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return None


def _make_scaler(device):
    """Fresh AMP scaler. Recreate after CUDA OOM — a poisoned scaler skips every later step."""
    import torch

    if getattr(device, "type", None) != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _predict_pack(
    net,
    pack: dict[str, Any],
    source: pl.DataFrame,
    tab_cols: list[str],
    scaler,
    rows: np.ndarray,
    device,
    cfg: CampaignConfig,
) -> np.ndarray:
    import torch

    batch = 256 if not cfg.smoke else 64
    amp_dtype = _amp_dtype(device)
    use_amp = amp_dtype is not None
    rows = np.asarray(rows, dtype=np.int64)
    ordered = pack["ordered"]
    source = _dep_tab_source(source, tab_cols)
    id_arr = ordered["MVT_ID_mvt"].to_numpy()
    while True:
        hats = []
        try:
            with torch.no_grad():
                for start in range(0, len(rows), batch):
                    sl = rows[start : start + batch]
                    seq_b, mask_b = gather_sequence(pack, sl)
                    tab_b = _tab_rows(source, tab_cols, scaler, ordered, sl, id_arr=id_arr)
                    ac = {"device_type": device.type, "enabled": use_amp}
                    if use_amp:
                        ac["dtype"] = amp_dtype
                    with torch.autocast(**ac):
                        pred = net(
                            torch.tensor(seq_b, device=device),
                            torch.tensor(mask_b, device=device),
                            torch.tensor(tab_b, dtype=torch.float32, device=device),
                        )
                    hats.append(pred.float().cpu().numpy())
            return np.concatenate(hats) if hats else np.zeros(len(rows), dtype=np.float32)
        except Exception as exc:
            _raise_min_batch_oom(exc, batch=batch, label="predict")
            if not _is_cuda_oom(exc):
                raise
            print(f"predict CUDA OOM at batch={batch}; retry {batch // 2}", flush=True)
            _cuda_empty()
            batch = max(32, batch // 2)


def _fit_seq(
    cfg: CampaignConfig,
    kind: str,
    pack: dict[str, Any],
    source: pl.DataFrame,
    tab_cols: list[str],
    scaler,
    y,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    dest,
    *,
    mixture: bool = False,
    y_va: np.ndarray | None = None,
) -> tuple[np.ndarray, float, Any, int, Any, Any, float | None]:
    import copy
    import torch

    try:
        torch.set_num_threads(max(1, int(cfg.thread_count)))
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(4, int(cfg.thread_count))))
    except RuntimeError:
        pass

    device = require_cuda(cfg)
    hidden = 32 if cfg.smoke else 128
    if kind != "transformer":
        hidden = 32 if cfg.smoke else 64
    seq_dim = int(pack["tokens"].shape[-1])
    ordered = pack["ordered"]
    n_tab = len(tab_cols)
    source = _dep_tab_source(source, tab_cols)
    id_arr = ordered["MVT_ID_mvt"].to_numpy()
    model = _SeqRegressor(kind, seq_dim, n_tab, device, hidden=hidden, mixture=mixture)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 2 if cfg.smoke else 12
    patience = 2 if cfg.smoke else 3
    batch = 256 if not cfg.smoke else 64
    rng = np.random.default_rng(cfg.seed)
    best_loss = 1e18
    best_val = 1e18
    best_model = None
    best_epoch = 0
    stall = 0
    start_epoch = 0
    n = int(len(train_rows))
    amp_dtype = _amp_dtype(device)
    use_amp = amp_dtype is not None
    peak_vram_gb = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ckpt_path = dest / "checkpoint.pt"
    ckpt_blob = None
    if ckpt_path.exists() and not cfg.force_rerun:
        try:
            try:
                blob = torch.load(ckpt_path, map_location=device, weights_only=False)
            except TypeError:
                blob = torch.load(ckpt_path, map_location=device)
            resume_ok = (
                blob.get("kind") == kind
                and blob.get("mixture", False) == mixture
                and "enc" in blob
                and blob.get("seq_norm") == "layer"
                and "in_norm" in blob
                and int(blob.get("epoch", -1)) >= 0
                and _finite(blob.get("val_mse"))
            )
            if resume_ok:
                model.load_weights(blob)
                start_epoch = int(blob.get("epoch", -1)) + 1
                best_val = float(blob.get("val_mse", best_val))
                best_loss = float(blob.get("loss", best_loss))
                best_epoch = int(blob.get("epoch", 0))
                stall = int(blob.get("stall", 0))
                if blob.get("batch"):
                    batch = max(32, min(batch, int(blob["batch"])))
                best_model = copy.deepcopy(model)
                if isinstance(blob.get("best"), dict) and "enc" in blob["best"]:
                    best_model.load_weights(blob["best"])
                if stall >= patience:
                    start_epoch = epochs
                ckpt_blob = blob
                if "opt" in blob:
                    try:
                        opt.load_state_dict(blob["opt"])
                    except Exception as exc:
                        print(f"{kind} optimizer resume skipped: {exc!r}", flush=True)
                print(f"{kind} resume checkpoint epoch={best_epoch} stall={stall}", flush=True)
            else:
                print(
                    f"{kind} checkpoint skipped (seq_norm={blob.get('seq_norm')} "
                    f"finite={_finite(blob.get('val_mse')) or _finite(blob.get('loss'))})",
                    flush=True,
                )
        except Exception as exc:
            print(f"{kind} checkpoint load skipped: {exc!r}", flush=True)
    scaler_amp = _make_scaler(device) if amp_dtype is torch.float16 else None
    if ckpt_blob is not None and scaler_amp is not None and "scaler" in ckpt_blob:
        try:
            scaler_amp.load_state_dict(ckpt_blob["scaler"])
        except Exception as exc:
            print(f"{kind} amp scaler resume skipped: {exc!r}", flush=True)

    def _fresh_opt():
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def _save_ckpt(epoch_i: int, *, to_drive: bool | None = None) -> None:
        payload = {
            **model.weights(),
            "epoch": epoch_i,
            "loss": best_loss,
            "mixture": mixture,
            "val_mse": best_val,
            "stall": stall,
            "kind": kind,
            "batch": batch,
        }
        if best_model is not None:
            payload["best"] = best_model.weights()
        try:
            payload["opt"] = opt.state_dict()
        except Exception:
            pass
        if scaler_amp is not None:
            try:
                payload["scaler"] = scaler_amp.state_dict()
            except Exception:
                pass
        torch.save(payload, dest / "checkpoint.pt")
        if to_drive is None:
            to_drive = epoch_i < 0 or stall == 0
        if to_drive:
            persist_repo_file(cfg, dest / "checkpoint.pt")

    def _rows_to_seq(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return gather_sequence(pack, np.asarray(rows, dtype=np.int64))

    def _predict_rows(net, rows: np.ndarray, _tab_all=None) -> np.ndarray:
        nonlocal batch
        rows = np.asarray(rows, dtype=np.int64)
        while True:
            hats = []
            try:
                with torch.no_grad():
                    for start in range(0, len(rows), batch):
                        sl = rows[start : start + batch]
                        seq_b, mask_b = _rows_to_seq(sl)
                        tab_b = _tab_rows(source, tab_cols, scaler, ordered, sl, id_arr=id_arr)
                        ac = {"device_type": device.type, "enabled": use_amp}
                        if use_amp:
                            ac["dtype"] = amp_dtype
                        with torch.autocast(**ac):
                            pred = net(
                                torch.tensor(seq_b, device=device),
                                torch.tensor(mask_b, device=device),
                                torch.tensor(tab_b, dtype=torch.float32, device=device),
                            )
                        hats.append(pred.float().cpu().numpy())
                return np.concatenate(hats) if hats else np.zeros(len(rows), dtype=np.float32)
            except Exception as exc:
                _raise_min_batch_oom(exc, batch=batch, label=kind)
                if not _is_cuda_oom(exc):
                    raise
                print(f"{kind} predict CUDA OOM at batch={batch}; retry {batch // 2}", flush=True)
                _cuda_empty()
                batch = max(32, batch // 2)

    if ckpt_blob is None:
        _save_ckpt(-1)
    epoch = start_epoch
    while epoch < epochs:
        losses = []
        order = rng.permutation(n)
        try:
            for start in range(0, n, batch):
                sl = train_rows[order[start : start + batch]]
                seq_b, mask_b = _rows_to_seq(sl)
                seq_t = torch.tensor(seq_b, device=device)
                mask_t = torch.tensor(mask_b, device=device)
                tab_b = torch.tensor(
                    _tab_rows(source, tab_cols, scaler, ordered, sl, id_arr=id_arr),
                    dtype=torch.float32,
                    device=device,
                )
                y_b = torch.tensor(_sanitize_np(y[sl]), dtype=torch.float32, device=device)
                opt.zero_grad(set_to_none=True)
                ac = {"device_type": device.type, "enabled": use_amp}
                if use_amp:
                    ac["dtype"] = amp_dtype
                with torch.autocast(**ac):
                    if mixture:
                        loss = model.mix_nll(seq_t, mask_t, tab_b, y_b)
                    else:
                        pred = model(seq_t, mask_t, tab_b)
                        loss = torch.mean((pred.float() - y_b) ** 2)
                if not torch.isfinite(loss.detach()).item():
                    opt.zero_grad(set_to_none=True)
                    if scaler_amp is not None:
                        scaler_amp = _make_scaler(device)
                    continue
                if scaler_amp is not None:
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
                    scaler_amp.step(opt)
                    scaler_amp.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
                    opt.step()
                losses.append(float(loss.detach().float().cpu()))
        except Exception as exc:
            _raise_min_batch_oom(exc, batch=batch, label=kind)
            if not _is_cuda_oom(exc):
                raise
            print(f"{kind} CUDA OOM at batch={batch}; retry {batch // 2}", flush=True)
            opt.zero_grad(set_to_none=True)
            _cuda_empty()
            batch = max(32, batch // 2)
            if amp_dtype is torch.float16:
                scaler_amp = _make_scaler(device)
            reloaded = False
            if ckpt_path.exists():
                try:
                    try:
                        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
                    except TypeError:
                        blob = torch.load(ckpt_path, map_location=device)
                    if blob.get("kind") == kind and blob.get("mixture", False) == mixture and "enc" in blob:
                        model.load_weights(blob)
                        opt = _fresh_opt()
                        if "opt" in blob:
                            try:
                                opt.load_state_dict(blob["opt"])
                            except Exception as opt_exc:
                                print(f"{kind} OOM optimizer reload skipped: {opt_exc!r}", flush=True)
                        reloaded = True
                except Exception as reload_exc:
                    print(f"{kind} OOM checkpoint reload skipped: {reload_exc!r}", flush=True)
            if not reloaded:
                opt = _fresh_opt()
            continue
        finite_losses = [v for v in losses if _finite(v)]
        mean_loss = float(np.mean(finite_losses)) if finite_losses else float("nan")
        if _finite(mean_loss):
            best_loss = min(best_loss, mean_loss)
        msg = f"{kind} epoch={epoch} loss={mean_loss:.4f} n={n} batch={batch}"
        if y_va is not None and len(val_rows):
            hat = _predict_rows(model, val_rows)
            val_mse = float(np.mean((hat - y_va) ** 2))
            msg += f" val_mse={val_mse:.4f}"
            if _finite(val_mse) and val_mse < best_val - 1e-8:
                best_val = val_mse
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    print(msg + f" early_stop patience={patience}", flush=True)
                    _save_ckpt(epoch, to_drive=True)
                    break
        _save_ckpt(epoch)
        print(msg, flush=True)
        epoch += 1
    if best_model is not None:
        model = best_model
    rhat = _predict_rows(model, val_rows) if len(val_rows) else np.zeros(0, dtype=np.float32)
    torch.save({**model.weights(), "epoch": best_epoch, "loss": best_loss, "mixture": mixture, "val_mse": best_val}, dest / "model.pt")
    if device.type == "cuda":
        peak_vram_gb = round(float(torch.cuda.max_memory_allocated()) / 1024**3, 3)
    return rhat, best_loss, best_val, device, best_epoch, _predict_rows, model, peak_vram_gb


def _score_seq_official_view(
    cfg: CampaignConfig,
    model,
    dest,
    *,
    view: str,
    extra_tab: list[str] | None,
    tab_cols: list[str],
    scaler,
    past: int,
    future: int,
    target: str,
    device,
) -> dict[str, Any]:
    """Apply a trained sequence model to another official view. Does not refit."""
    from prc_challenge.campaign.data import load_base_preds, load_cache_columns

    _ = extra_tab  # trained tab_cols already includes GPU-04 fusion extras

    f3 = _f3_payload(cfg)
    token_extra = [
        c
        for c in (
            "n_delayed_dep_ctr_1800s",
            "sec_since_prev_dep",
            "n_arr_1800s",
            "n_arr_3600s",
            "n_dep_1800s",
            "dep_arr_ratio_1800s",
        )
        if c in f3.columns
    ]
    # Skinny cache + late fusion join. GPU-04 tab_cols is ~all BASE nums + F3;
    # attaching those onto val before seq_dep_frame OOMs a 16 GB Colab GPU host.
    _, val = matched_train_val(load_cache_columns(cfg, extra=token_extra), view)
    official = load_base_preds(cfg, view).select(
        ["MVT_ID_mvt", pl.col("prediction").alias("base_prediction")]
    )
    n0 = val.height
    val = val.join(official, on="MVT_ID_mvt", how="inner")
    if val.height == 0 or val.height != n0 or val["base_prediction"].null_count():
        return {"status": "skipped", "reason": f"need {view} BASE preds from the CPU campaign"}
    keep = ["MVT_ID_mvt", *token_extra]
    keep = list(dict.fromkeys([c for c in keep if c == "MVT_ID_mvt" or c not in val.columns]))
    if len(keep) > 1:
        val = val.join(f3.select(keep), on="MVT_ID_mvt", how="left")
    fallback = float(val["base_prediction"].median())
    val_seq = seq_dep_frame(val)
    tab_keep = ["MVT_ID_mvt", "base_prediction", *[c for c in tab_cols if c in val.columns]]
    tab_va = val.select([c for c in dict.fromkeys(tab_keep) if c in val.columns])
    dummy = tab_va.head(0)
    tab_va, _ = _attach_missing_tab(cfg, tab_va, dummy, tab_cols, f3)
    del f3, val
    tab_va = _ensure_seq_tab_cols(tab_va, tab_cols)
    tab_src = _dep_tab_source(tab_va, tab_cols)
    del tab_va
    pack_va, is_target_va = _event_pack(cfg, val_seq, past=past, future=future, tag=f"{view}_val")
    del val_seq
    ordered_va = pack_va["ordered"]
    aligned_base = _align_numeric(ordered_va, ordered_va, "base_prediction", fallback)
    val_rows = np.flatnonzero(is_target_va)
    rhat = _predict_pack(model, pack_va, tab_src, tab_cols, scaler, val_rows, device, cfg)
    pred = np.clip(rhat, 0, None) if target == "raw" else np.clip(aligned_base[is_target_va] + rhat, 0, None)
    scored = _compose(
        cfg, view, ordered_va.filter(pl.Series(is_target_va)).with_columns(pl.Series("prediction", pred)), pred, dest=dest
    )
    return compact_score(scored)


def _attach_missing_tab(
    cfg: CampaignConfig,
    tab_tr: pl.DataFrame,
    tab_va: pl.DataFrame,
    tab_cols: list[str],
    f3: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Join fusion extras after seq_dep_frame so GPU-04 does not bloat the residual table."""
    missing = [c for c in tab_cols if c not in tab_tr.columns]
    if not missing:
        return tab_tr, tab_va
    f3_need = [c for c in missing if c in f3.columns]
    cache_need = [c for c in missing if c not in f3.columns]
    if cache_need:
        from prc_challenge.campaign.data import load_selected_columns

        chunk = 40
        for i in range(0, len(cache_need), chunk):
            part = cache_need[i : i + chunk]
            extra = load_selected_columns(cfg, part)
            keep = ["MVT_ID_mvt", *[c for c in part if c in extra.columns]]
            if len(keep) <= 1:
                continue
            extra = extra.select(keep)
            tab_tr = tab_tr.join(extra, on="MVT_ID_mvt", how="left")
            tab_va = tab_va.join(extra, on="MVT_ID_mvt", how="left")
            del extra
    if f3_need:
        payload = f3.select(["MVT_ID_mvt", *f3_need])
        tab_tr = tab_tr.join(payload, on="MVT_ID_mvt", how="left")
        tab_va = tab_va.join(payload, on="MVT_ID_mvt", how="left")
    return tab_tr, tab_va


def _train_seq(
    cfg: CampaignConfig,
    kind: str,
    dest,
    extra_tab: list[str] | None = None,
    *,
    mixture: bool = False,
    target: str = "residual",
) -> dict[str, Any]:
    train, val, val_base, view = _residual_frame(cfg)
    f3 = _f3_payload(cfg)
    token_extra = [
        c
        for c in (
            "n_delayed_dep_ctr_1800s",
            "sec_since_prev_dep",
            "n_arr_1800s",
            "n_arr_3600s",
            "n_dep_1800s",
            "dep_arr_ratio_1800s",
        )
        if c in f3.columns
    ]
    keep = ["MVT_ID_mvt"] + token_extra
    keep = list(dict.fromkeys(keep))
    keep = [c for c in keep if c == "MVT_ID_mvt" or c not in train.columns]
    if len(keep) > 1:
        train = train.join(f3.select(keep), on="MVT_ID_mvt", how="left")
        val = val.join(f3.select(keep), on="MVT_ID_mvt", how="left")
    past = 8 if cfg.smoke else 64
    future = 4 if cfg.smoke else 32
    fallback_base = float(np.median(val_base)) if len(val_base) else 0.0
    if "base_prediction" not in val.columns:
        val = val.with_columns(pl.Series("base_prediction", np.asarray(val_base, dtype=np.float32)))
    train_seq = seq_dep_frame(train)
    val_seq = seq_dep_frame(val)
    tab_cols = extra_tab or [c for c in ("local_hour", "takeoff_minus_sched_sec", "sknt") if c in train.columns]
    tab_keep = ["MVT_ID_mvt", *[c for c in tab_cols if c in train.columns]]
    tab_tr = train.select([c for c in dict.fromkeys(tab_keep) if c in train.columns])
    tab_va = val.select([c for c in dict.fromkeys(tab_keep) if c in val.columns])
    del train, val
    tab_tr, tab_va = _attach_missing_tab(cfg, tab_tr, tab_va, tab_cols, f3)
    del f3
    tab_tr = _ensure_seq_tab_cols(tab_tr, tab_cols)
    tab_va = _ensure_seq_tab_cols(tab_va, tab_cols)
    tab_cols = [c for c in tab_cols if c in tab_tr.columns]
    scaler = fit_num_scaler(tab_tr, tab_cols)
    tab_src_tr = _dep_tab_source(tab_tr, tab_cols)
    tab_src_va = _dep_tab_source(tab_va, tab_cols)
    del tab_tr, tab_va
    pack_tr, is_target_tr = _event_pack(cfg, train_seq, past=past, future=future, tag=f"{view}_train")
    pack_va, is_target_va = _event_pack(cfg, val_seq, past=past, future=future, tag=f"{view}_val")
    del train_seq, val_seq
    ordered_tr = pack_tr["ordered"]
    ordered_va = pack_va["ordered"]
    aligned_base = _align_numeric(ordered_va, ordered_va, "base_prediction", fallback_base)
    aligned_tr_base = _align_numeric(ordered_tr, ordered_tr, "oof_prediction", fallback_base)
    y_col = "TAXITIME_SEC_mvt" if target == "raw" else "residual"
    if y_col in ordered_tr.columns:
        y_all = np.nan_to_num(ordered_tr[y_col].fill_null(0).to_numpy().astype(np.float32), nan=0.0)
    else:
        y_all = _align_numeric(ordered_tr, ordered_tr, y_col, 0.0)
    fit_m, tune_m = _internal_masks(ordered_tr)
    fit_m = fit_m & is_target_tr
    tune_m = tune_m & is_target_tr
    train_rows = np.flatnonzero(fit_m)
    tune_rows = np.flatnonzero(tune_m)
    try:
        rhat_tune, best_loss, best_val, device, best_epoch, _predict_rows, model, peak_vram_gb = _fit_seq(
            cfg,
            kind,
            pack_tr,
            tab_src_tr,
            tab_cols,
            scaler,
            y_all,
            train_rows,
            tune_rows,
            dest,
            mixture=mixture,
            y_va=y_all[tune_m] if int(tune_m.sum()) else None,
        )
        val_rows = np.flatnonzero(is_target_va)
        rhat = _predict_pack(model, pack_va, tab_src_va, tab_cols, scaler, val_rows, device, cfg)
    except SeqMinBatchOom as exc:
        return _seq_oom_skip(exc)
    rhat = _sanitize_np(rhat)
    rhat_tune = _sanitize_np(rhat_tune)
    if target == "raw":
        pred = np.clip(rhat, 0, None)
        pred_tune = np.clip(rhat_tune, 0, None)
        conclusion = f"{kind} raw taxi target (secondary comparison)"
    else:
        pred = np.clip(aligned_base[is_target_va] + rhat, 0, None)
        pred_tune = np.clip(aligned_tr_base[tune_m] + rhat_tune, 0, None)
        conclusion = f"{kind} residual on official BASE val; early-stop on E37 tune month"
    keep = [c for c in ("MVT_ID_mvt", "AIRPORT", "TAXITIME_SEC_mvt", "month") if c in ordered_tr.columns]
    ordered_tr.select(keep).filter(pl.Series(tune_m)).with_columns(pl.Series("prediction", pred_tune)).write_parquet(
        Path(dest) / "tune_predictions.parquet"
    )
    aligned_val = ordered_va.filter(pl.Series(is_target_va)).with_columns(pl.Series("prediction", pred))
    scored = _compose(cfg, view, aligned_val, pred, dest=dest)
    persist_dir(cfg, Path(dest))
    extra_views: dict[str, Any] = {}
    missing_other = [v for v in cfg.views if v != view and not _extra_view_ready(dest, v)]
    if not missing_other and not cfg.force_rerun:
        print(f"{kind} resume skip extra-view sequence scoring", flush=True)
        extra_views = _resumed_extra_views(cfg, dest, view)
    else:
        for other in cfg.views:
            if other == view:
                continue
            if _extra_view_ready(dest, other) and not cfg.force_rerun:
                extra_views[other] = _hydrate_extra_view(cfg, dest, other)
                print(f"{kind} resume extra view {other} rmse={extra_views[other].get('rmse_seconds')}", flush=True)
                continue
            try:
                part = _score_seq_official_view(
                    cfg,
                    model,
                    dest,
                    view=other,
                    extra_tab=extra_tab,
                    tab_cols=tab_cols,
                    scaler=scaler,
                    past=past,
                    future=future,
                    target=target,
                    device=device,
                )
                extra_views[other] = compact_score(part) if part.get("rmse_seconds") is not None else part
                persist_repo_file(cfg, Path(dest) / f"predictions_{other}.parquet")
                print(f"extra view {other} rmse={extra_views[other].get('rmse_seconds')}", flush=True)
            except Exception as exc:
                print(f"extra view {other} failed: {exc!r}", flush=True)
                extra_views[other] = extra_view_exception(exc, view=other)
    return {
        "status": "completed",
        "rmse_seconds": scored.get("rmse_seconds") or scored.get("overall", {}).get("rmse_seconds"),
        "sse": scored.get("sse"),
        "vs_base": scored.get("vs_base"),
        "protected": scored.get("protected"),
        "train_mse": best_loss if best_loss < 1e17 else None,
        "tune_mse": best_val if best_val < 1e17 else None,
        "epochs": 2 if cfg.smoke else 12,
        "best_epoch": best_epoch,
        "device": str(device),
        "seq_len": past,
        "future_len": future,
        "seq_dim": int(pack_tr["tokens"].shape[-1]),
        "hidden": 32 if cfg.smoke else (128 if kind == "transformer" else 64),
        "n_tab": int(len(tab_cols)),
        "tab_cols": tab_cols[:48],
        "mixture": mixture,
        "target": target,
        "early_stop": "e37_tune_month",
        "peak_vram_gb": peak_vram_gb,
        "seq_is_arr_mean": float(ordered_tr["is_arr"].mean()) if "is_arr" in ordered_tr.columns else 0.0,
        "views": {view: compact_score(scored), **extra_views},
        "conclusion": conclusion,
    }


def _gpu01_followup_gate(cfg: CampaignConfig, *, hypothesis: str, followup: str) -> dict[str, Any] | None:
    """Skip secondary GPU jobs unless GPU-01 completed and reduced composed SSE."""
    import json
    from pathlib import Path

    prior_path = Path(cfg.artifact_root) / "GPU-01-transformer" / "metrics.json"
    if not prior_path.exists():
        return {
            "status": "skipped",
            "reason": "need GPU-01 metrics first",
            "conclusion": "PARK",
            "hypothesis": hypothesis,
        }
    try:
        prior = json.loads(prior_path.read_text())
    except json.JSONDecodeError:
        return {
            "status": "skipped",
            "reason": "need GPU-01 metrics first",
            "conclusion": "PARK",
            "hypothesis": hypothesis,
        }
    if prior.get("status") != "completed":
        if is_terminal_skip(prior):
            return {
                "status": "skipped",
                "reason": f"GPU-01 parked; {followup} is not justified",
                "conclusion": "PARK",
                "prior_status": prior.get("status"),
                "hypothesis": hypothesis,
            }
        return {
            "status": "skipped",
            "reason": f"need GPU-01 completed (status={prior.get('status')})",
            "conclusion": "PARK",
            "prior_status": prior.get("status"),
            "hypothesis": hypothesis,
        }
    delta = prior.get("delta_sse_vs_base")
    if delta is None:
        delta = (prior.get("vs_base") or {}).get("delta_sse_vs_base")
    rmse = prior.get("rmse_seconds")
    if rmse is None:
        rmse = (prior.get("views") or {}).get("V1", {}).get("rmse_seconds") if isinstance(prior.get("views"), dict) else None
    if not _finite(delta) or delta <= 0 or (rmse is not None and not _finite(rmse)):
        return {
            "status": "skipped",
            "reason": f"GPU-01 sequence residual did not reduce composed SSE; {followup} is not justified",
            "conclusion": "PARK",
            "prior_delta_sse_vs_base": delta,
            "hypothesis": hypothesis,
        }
    return None


def _sequence_oom_family_park(cfg: CampaignConfig, *, hypothesis: str) -> dict[str, Any] | None:
    """TCN/GRU/mixture would hit the same min-batch OOM as GPU-01. FAISS can still run."""
    import json
    from pathlib import Path

    prior_path = Path(cfg.artifact_root) / "GPU-01-transformer" / "metrics.json"
    if not prior_path.exists():
        return None
    try:
        prior = json.loads(prior_path.read_text())
    except json.JSONDecodeError:
        return None
    reason = str(prior.get("reason") or prior.get("error") or "")
    if is_terminal_skip(prior) and "oom" in reason.lower() and "parked" in reason.lower():
        return {
            "status": "skipped",
            "reason": "GPU-01 parked on CUDA OOM; sequence family is not justified",
            "conclusion": "PARK",
            "hypothesis": hypothesis,
        }
    return None


def job_gpu01(cfg: CampaignConfig, dest) -> dict[str, Any]:
    out = _train_seq(cfg, "transformer", dest)
    out["hypothesis"] = "airport event-sequence Transformer explains BASE residuals"
    return out


def job_gpu01b(cfg: CampaignConfig, dest) -> dict[str, Any]:
    hyp = "same Transformer trained on raw taxi as a secondary comparison"
    gate = _gpu01_followup_gate(cfg, hypothesis=hyp, followup="raw-target secondary")
    if gate is not None:
        return gate
    out = _train_seq(cfg, "transformer", dest, target="raw")
    out["hypothesis"] = hyp
    return out


def job_gpu02(cfg: CampaignConfig, dest) -> dict[str, Any]:
    hyp = "TCN residual vs Transformer: is attention needed?"
    park = _sequence_oom_family_park(cfg, hypothesis=hyp)
    if park is not None:
        return park
    out = _train_seq(cfg, "tcn", dest)
    out["hypothesis"] = hyp
    return out


def job_gpu03(cfg: CampaignConfig, dest) -> dict[str, Any]:
    hyp = "improved GRU on longer ranking-safe event sequences"
    park = _sequence_oom_family_park(cfg, hypothesis=hyp)
    if park is not None:
        return park
    out = _train_seq(cfg, "gru", dest)
    out["hypothesis"] = hyp
    return out


def job_gpu04(cfg: CampaignConfig, dest) -> dict[str, Any]:
    hyp = "sequence encoder + BASE tabular + BASE pred + F3 fusion"
    gate = _gpu01_followup_gate(cfg, hypothesis=hyp, followup="fusion")
    if gate is not None:
        return gate
    _cats, nums = feature_lists("e08b")
    extra = list(dict.fromkeys([*nums, *f3_numeric_columns(), "oof_prediction"]))
    out = _train_seq(cfg, "transformer", dest, extra_tab=extra)
    out["hypothesis"] = hyp
    return out


def _align_mvt_id(frame: pl.DataFrame, like: pl.DataFrame) -> pl.DataFrame:
    if "MVT_ID_mvt" not in frame.columns or "MVT_ID_mvt" not in like.columns:
        return frame
    dtype = like.schema["MVT_ID_mvt"]
    if frame.schema["MVT_ID_mvt"] == dtype:
        return frame
    return frame.with_columns(pl.col("MVT_ID_mvt").cast(dtype, strict=False))


def job_gpu05(cfg: CampaignConfig, dest) -> dict[str, Any]:
    from prc_challenge.campaign.data import load_cache_columns

    hyp = "queue residual is easier for a temporal net than raw taxi"
    park = _sequence_oom_family_park(cfg, hypothesis=hyp)
    if park is not None:
        return park
    cpu05_train = Path(cfg.artifact_root) / "CPU-05-route-queue" / "route_oof_train.parquet"
    cpu05_val = Path(cfg.artifact_root) / "CPU-05-route-queue" / "route_oof_val.parquet"
    cpu05_metrics = Path(cfg.artifact_root) / "CPU-05-route-queue" / "metrics.json"
    if not (cpu05_train.exists() and cpu05_val.exists()) and not cfg.smoke:
        prior = {}
        if cpu05_metrics.exists():
            import json

            try:
                prior = json.loads(cpu05_metrics.read_text())
            except json.JSONDecodeError:
                prior = {}
        if is_terminal_skip(prior):
            return {
                "status": "skipped",
                "reason": "CPU-05 parked; queue residual sequence is not justified",
                "conclusion": "PARK",
                "hypothesis": hyp,
            }
        return {
            "status": "skipped",
            "reason": "need CPU-05 route_oof_{train,val}.parquet",
            "conclusion": "PARK",
            "hypothesis": hyp,
        }
    df = load_cache_columns(cfg)
    cats, nums = feature_lists("e08b")
    view = "V1" if "V1" in cfg.views else cfg.views[0]
    train, val = matched_train_val(df, view)
    keys = ["AIRPORT", "STAND_mvt", "RUNWAY_mvt"]
    gmed = float(train["TAXITIME_SEC_mvt"].median())
    used_cpu05 = False
    if cpu05_train.exists() and cpu05_val.exists():
        tr_oof = _align_mvt_id(pl.read_parquet(cpu05_train).select(["MVT_ID_mvt", "route_base"]), train)
        va_oof = _align_mvt_id(pl.read_parquet(cpu05_val).select(["MVT_ID_mvt", "route_base"]), val)
        train = train.join(tr_oof, on="MVT_ID_mvt", how="left")
        val = val.join(va_oof, on="MVT_ID_mvt", how="left")
        if int(train["route_base"].null_count()) < train.height:
            used_cpu05 = True
            train = train.with_columns(pl.col("route_base").fill_null(gmed))
            val = val.with_columns(pl.col("route_base").fill_null(gmed))
            train = train.with_columns((pl.col("TAXITIME_SEC_mvt") - pl.col("route_base")).alias("residual"))
            print("GPU-05 loaded CPU-05 route_oof", flush=True)
        elif not cfg.smoke:
            return {
                "status": "skipped",
                "reason": "need CPU-05 route_oof IDs to match the campaign cache",
                "conclusion": "PARK",
                "hypothesis": hyp,
            }
    if not used_cpu05:
        oof_parts = []
        for month in train["month"].unique().sort().to_list():
            fit = train.filter(pl.col("month") != month)
            hold = train.filter(pl.col("month") == month)
            if fit.height == 0 or hold.height == 0:
                continue
            cell = fit.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_base"))
            oof_parts.append(hold.join(cell, on=keys, how="left"))
        if oof_parts:
            train = pl.concat(oof_parts, how="diagonal")
        else:
            cell_all = train.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_base"))
            train = train.join(cell_all, on=keys, how="left")
        cell = train.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_base_full"))
        train = train.with_columns(pl.col("route_base").fill_null(gmed))
        train = train.with_columns((pl.col("TAXITIME_SEC_mvt") - pl.col("route_base")).alias("residual"))
        val = val.join(cell.rename({"route_base_full": "route_base"}), on=keys, how="left").with_columns(
            pl.col("route_base").fill_null(gmed)
        )
    past = 8 if cfg.smoke else 64
    future = 4 if cfg.smoke else 32
    tab_cols = [c for c in ("local_hour", "takeoff_minus_sched_sec") if c in train.columns]
    scaler = fit_num_scaler(train, tab_cols)
    tab_src_tr = _dep_tab_source(train, tab_cols)
    tab_src_va = _dep_tab_source(val, tab_cols)
    train_seq = seq_dep_frame(train)
    val_seq = seq_dep_frame(val)
    del df, train, val
    pack_tr, is_target_tr = _event_pack(cfg, train_seq, past=past, future=future, tag=f"{view}_train")
    pack_va, is_target_va = _event_pack(cfg, val_seq, past=past, future=future, tag=f"{view}_val")
    del train_seq, val_seq
    ordered_tr = pack_tr["ordered"]
    ordered_va = pack_va["ordered"]
    y = _align_numeric(ordered_tr, ordered_tr, "residual", 0.0)
    route = _align_numeric(ordered_va, ordered_va, "route_base", gmed)
    route_tr = _align_numeric(ordered_tr, ordered_tr, "route_base", gmed)
    fit_m, tune_m = _internal_masks(ordered_tr)
    fit_m = fit_m & is_target_tr
    tune_m = tune_m & is_target_tr
    train_rows = np.flatnonzero(fit_m)
    tune_rows = np.flatnonzero(tune_m)
    try:
        rhat_tune, best_loss, best_val, device, best_epoch, _predict_rows, model, peak_vram_gb = _fit_seq(
            cfg,
            "transformer",
            pack_tr,
            tab_src_tr,
            tab_cols,
            scaler,
            y,
            train_rows,
            tune_rows,
            dest,
            y_va=y[tune_m] if int(tune_m.sum()) else None,
        )
        val_rows = np.flatnonzero(is_target_va)
        rhat = _sanitize_np(_predict_pack(model, pack_va, tab_src_va, tab_cols, scaler, val_rows, device, cfg))
        rhat_tune = _sanitize_np(rhat_tune)
    except SeqMinBatchOom as exc:
        return {**_seq_oom_skip(exc), "hypothesis": hyp}
    pred = np.clip(route[is_target_va] + rhat, 0, None)
    pred_tune = np.clip(route_tr[tune_m] + rhat_tune, 0, None)
    keep = [c for c in ("MVT_ID_mvt", "AIRPORT", "TAXITIME_SEC_mvt", "month") if c in ordered_tr.columns]
    ordered_tr.select(keep).filter(pl.Series(tune_m)).with_columns(pl.Series("prediction", pred_tune)).write_parquet(
        Path(dest) / "tune_predictions.parquet"
    )
    scored = _compose(
        cfg,
        view,
        ordered_va.filter(pl.Series(is_target_va)).with_columns(pl.Series("prediction", pred)),
        pred,
        dest=dest,
    )
    persist_dir(cfg, Path(dest))
    extra_views: dict[str, Any] = {}
    missing_other = [v for v in cfg.views if v != view and not _extra_view_ready(dest, v)]
    if not missing_other and not cfg.force_rerun:
        print("GPU-05 resume skip extra-view sequence scoring", flush=True)
        extra_views = _resumed_extra_views(cfg, dest, view)
    else:
        extra_cache = load_cache_columns(cfg)
        for other in cfg.views:
            if other == view:
                continue
            if _extra_view_ready(dest, other) and not cfg.force_rerun:
                extra_views[other] = _hydrate_extra_view(cfg, dest, other)
                print(f"GPU-05 resume extra view {other} rmse={extra_views[other].get('rmse_seconds')}", flush=True)
                continue
            try:
                tr_o, va_o = matched_train_val(extra_cache, other)
                cell_o = tr_o.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").median().alias("route_base"))
                va_o = va_o.join(cell_o, on=keys, how="left").with_columns(pl.col("route_base").fill_null(gmed))
                va_o = _ensure_seq_tab_cols(va_o, tab_cols)
                tab_o = _dep_tab_source(va_o, tab_cols)
                va_seq = seq_dep_frame(va_o)
                del tr_o, va_o
                pack_o, is_tgt_o = _event_pack(cfg, va_seq, past=past, future=future, tag=f"{other}_val")
                del va_seq
                ordered_o = pack_o["ordered"]
                route_o = _align_numeric(ordered_o, ordered_o, "route_base", gmed)
                tgt_rows = np.flatnonzero(is_tgt_o)
                rhat_o = _predict_pack(model, pack_o, tab_o, tab_cols, scaler, tgt_rows, device, cfg)
                pred_o = np.clip(route_o[is_tgt_o] + rhat_o, 0, None)
                part = _compose(
                    cfg,
                    other,
                    ordered_o.filter(pl.Series(is_tgt_o)).with_columns(pl.Series("prediction", pred_o)),
                    pred_o,
                    dest=dest,
                )
                extra_views[other] = compact_score(part) if part.get("rmse_seconds") is not None else part
                persist_repo_file(cfg, Path(dest) / f"predictions_{other}.parquet")
                print(f"extra view {other} rmse={extra_views[other].get('rmse_seconds')}", flush=True)
            except Exception as exc:
                print(f"extra view {other} failed: {exc!r}", flush=True)
                extra_views[other] = extra_view_exception(exc, view=other)
    return {
        "status": "completed",
        "rmse_seconds": scored.get("rmse_seconds") or scored.get("overall", {}).get("rmse_seconds"),
        "sse": scored.get("sse"),
        "vs_base": scored.get("vs_base"),
        "protected": scored.get("protected"),
        "train_mse": best_loss if best_loss < 1e17 else None,
        "tune_mse": best_val if best_val < 1e17 else None,
        "device": str(device),
        "best_epoch": best_epoch,
        "peak_vram_gb": peak_vram_gb,
        "views": {view: compact_score(scored), **extra_views},
        "conclusion": (
            "sequence models congestion after CPU-05 route OOF"
            if used_cpu05
            else "sequence models congestion after month-blocked median (smoke fallback)"
        ),
        "hypothesis": hyp,
        "route_base_source": "CPU-05-route-queue" if used_cpu05 else "month_blocked_median",
    }


def job_gpu06(cfg: CampaignConfig, dest) -> dict[str, Any]:
    hyp = "sequence mixture of normal/delayed/extreme residuals"
    park = _sequence_oom_family_park(cfg, hypothesis=hyp)
    if park is not None:
        return park
    out = _train_seq(cfg, "transformer", dest, mixture=True)
    out["hypothesis"] = hyp
    out["conclusion"] = "3-component Gaussian mixture NLL; RMSE uses expected value"
    return out


def job_gpu07(cfg: CampaignConfig, dest) -> dict[str, Any]:
    f3 = _f3_payload(cfg)
    nums = [c for c in f3_numeric_columns() if c in f3.columns]
    if not nums:
        return {"status": "skipped", "reason": "no F3 columns", "conclusion": "PARK"}
    train, val, val_base, view = _residual_frame(cfg)
    train = train.join(f3.select(["MVT_ID_mvt", *nums]), on="MVT_ID_mvt", how="left")
    val = val.join(f3.select(["MVT_ID_mvt", *nums]), on="MVT_ID_mvt", how="left")
    cap = 3_000 if cfg.smoke else 80_000
    n_index = train.height
    if train.height > cap:
        train = train.sample(n=cap, seed=cfg.seed, shuffle=True)
        print(f"GPU-07 index sampled {train.height} of {n_index}", flush=True)
    Xtr = _sanitize_np(train.select(nums).fill_null(0).to_numpy())
    Xva = _sanitize_np(val.select(nums).fill_null(0).to_numpy())
    Xtr = Xtr / np.clip(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-8, None)
    Xva = Xva / np.clip(np.linalg.norm(Xva, axis=1, keepdims=True), 1e-8, None)
    resid = np.nan_to_num(train["residual"].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    airports_tr = train["AIRPORT"].to_numpy()
    airports_va = val["AIRPORT"].to_numpy()
    k = 8 if cfg.smoke else 15
    rhat = np.zeros(len(Xva), dtype=np.float64)
    backend = "sklearn"

    def _weighted_mean(gathered: np.ndarray, scores: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
        valid = np.asarray(neighbor_idx) >= 0
        w = np.clip(np.asarray(scores, dtype=np.float64), 0, None)
        w = np.where(valid, w, 0.0)
        g = np.asarray(gathered, dtype=np.float64)
        g = np.where(valid, g, 0.0)
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-8, None)
        return (g * w).sum(axis=1)

    def _search(xb, xq, yb, kk):
        kk = min(kk, max(1, len(xb)))
        xb = np.ascontiguousarray(xb)
        xq = np.ascontiguousarray(xq)
        try:
            import faiss
        except Exception:
            faiss = None
        if faiss is not None:
            try:
                def _cpu_index():
                    return faiss.IndexFlatIP(int(xb.shape[1]))

                index = _cpu_index()
                label = "faiss"
                try:
                    if hasattr(faiss, "StandardGpuResources") and require_cuda(cfg).type == "cuda":
                        res = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                        label = "faiss-gpu"
                except Exception as exc:
                    print(f"faiss GPU skipped ({exc!r}); CPU IndexFlatIP", flush=True)
                    index = _cpu_index()
                    label = "faiss"
                try:
                    index.add(xb)
                    scores, nidx = index.search(xq, kk)
                    nidx = np.asarray(nidx)
                    gathered = yb[np.clip(nidx, 0, max(len(yb) - 1, 0))]
                    return _weighted_mean(gathered, scores, nidx), label
                except Exception as exc:
                    if label != "faiss-gpu":
                        raise
                    print(f"faiss GPU search failed ({exc!r}); CPU IndexFlatIP", flush=True)
                    _cuda_empty()
                    index = _cpu_index()
                    index.add(xb)
                    scores, nidx = index.search(xq, kk)
                    nidx = np.asarray(nidx)
                    gathered = yb[np.clip(nidx, 0, max(len(yb) - 1, 0))]
                    return _weighted_mean(gathered, scores, nidx), "faiss"
            except Exception as exc:
                print(f"faiss failed ({exc!r}); sklearn", flush=True)
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=kk, metric="cosine", algorithm="auto")
        nn.fit(xb)
        dist, nidx = nn.kneighbors(xq)
        nidx = np.asarray(nidx)
        gathered = yb[np.clip(nidx, 0, max(len(yb) - 1, 0))]
        return _weighted_mean(gathered, 1.0 / np.clip(dist, 1e-8, None), nidx), "sklearn"

    rwy_tr = None
    rwy_va = None
    if "RUNWAY_mvt" in train.columns and "RUNWAY_mvt" in val.columns:
        rwy_tr = train["RUNWAY_mvt"].fill_null("__NA__").cast(pl.String).to_numpy()
        rwy_va = val["RUNWAY_mvt"].fill_null("__NA__").cast(pl.String).to_numpy()

    def retrieve(xb, yb, a_tr, xq, a_q, scope_name, r_tr=None, r_q=None):
        out = np.zeros(len(xq), dtype=np.float64)
        used_local = "sklearn"
        if scope_name == "route" and r_tr is not None and r_q is not None:
            for ap in np.unique(a_q):
                for rwy in np.unique(r_q[a_q == ap]):
                    va_m = (a_q == ap) & (r_q == rwy)
                    if int(va_m.sum()) == 0:
                        continue
                    tr_m = (a_tr == ap) & (r_tr == rwy)
                    if int(tr_m.sum()) < 8:
                        tr_m = a_tr == ap
                    if int(tr_m.sum()) < 5:
                        hat, used_local = _search(xb, xq[va_m], yb, k)
                    else:
                        hat, used_local = _search(xb[tr_m], xq[va_m], yb[tr_m], k)
                    out[va_m] = hat
            return out, used_local
        for ap in np.unique(a_q):
            tr_m = a_tr == ap
            va_m = a_q == ap
            if int(va_m.sum()) == 0:
                continue
            if int(tr_m.sum()) < 5:
                hat, used_local = _search(xb, xq[va_m], yb, k)
            else:
                hat, used_local = _search(xb[tr_m], xq[va_m], yb[tr_m], k)
            out[va_m] = hat
        return out, used_local

    rhat_ap, used = retrieve(Xtr, resid, airports_tr, Xva, airports_va, "airport")
    rhat = rhat_ap
    scope = "airport"
    hat_tune = None
    if rwy_tr is not None:
        rhat_r, used = retrieve(Xtr, resid, airports_tr, Xva, airports_va, "route", rwy_tr, rwy_va)
        fit_m, tune_m = _internal_masks(train)
        if int(fit_m.sum()) >= 20 and int(tune_m.sum()) >= 10:
            hat_ap_t, _ = retrieve(
                Xtr[fit_m], resid[fit_m], airports_tr[fit_m], Xtr[tune_m], airports_tr[tune_m], "airport"
            )
            hat_r_t, _ = retrieve(
                Xtr[fit_m],
                resid[fit_m],
                airports_tr[fit_m],
                Xtr[tune_m],
                airports_tr[tune_m],
                "route",
                rwy_tr[fit_m],
                rwy_tr[tune_m],
            )
            y_t = resid[tune_m]
            if float(np.mean((y_t - hat_r_t) ** 2)) < float(np.mean((y_t - hat_ap_t) ** 2)):
                rhat = rhat_r
                scope = "route"
                hat_tune = hat_r_t
            else:
                hat_tune = hat_ap_t
    backend = used
    pred = np.clip(val_base + rhat, 0, None)
    fit_m, tune_m = _internal_masks(train)
    if int(tune_m.sum()) >= 8 and int(fit_m.sum()) >= 20:
        if hat_tune is None:
            hat_tune, _ = retrieve(
                Xtr[fit_m],
                resid[fit_m],
                airports_tr[fit_m],
                Xtr[tune_m],
                airports_tr[tune_m],
                scope,
                None if rwy_tr is None else rwy_tr[fit_m],
                None if rwy_tr is None else rwy_tr[tune_m],
            )
        pred_tune = np.clip(train["oof_prediction"].to_numpy()[tune_m] + hat_tune, 0, None)
        keep = [c for c in ("MVT_ID_mvt", "AIRPORT", "TAXITIME_SEC_mvt", "month") if c in train.columns]
        train.select(keep).filter(pl.Series(tune_m)).with_columns(pl.Series("prediction", pred_tune)).write_parquet(
            Path(dest) / "tune_predictions.parquet"
        )
    scored = _compose(cfg, view, val.with_columns(pl.Series("prediction", pred)), pred, dest=dest)
    persist_dir(cfg, Path(dest))
    extra_views: dict[str, Any] = {}
    from prc_challenge.campaign.data import load_base_preds as _load_base
    from prc_challenge.campaign.data import load_cache_columns

    missing_other = [v for v in cfg.views if v != view and not _extra_view_ready(dest, v)]
    if not missing_other and not cfg.force_rerun:
        print("GPU-07 resume skip extra-view retrieval", flush=True)
        extra_views = _resumed_extra_views(cfg, dest, view)
    else:
        cache = load_cache_columns(cfg, extra=["RUNWAY_mvt"] if scope == "route" else None)
        for other in cfg.views:
            if other == view:
                continue
            if _extra_view_ready(dest, other) and not cfg.force_rerun:
                extra_views[other] = _hydrate_extra_view(cfg, dest, other)
                print(f"GPU-07 resume extra view {other} rmse={extra_views[other].get('rmse_seconds')}", flush=True)
                continue
            try:
                _, va = matched_train_val(cache, other)
                official = _load_base(cfg, other)
                missing_nums = [c for c in nums if c not in va.columns]
                if missing_nums:
                    va = va.join(f3.select(["MVT_ID_mvt", *missing_nums]), on="MVT_ID_mvt", how="left")
                n0 = va.height
                va = va.join(
                    official.select(["MVT_ID_mvt", pl.col("prediction").alias("bp")]),
                    on="MVT_ID_mvt",
                    how="inner",
                )
                if va.height == 0 or va.height != n0 or va["bp"].null_count():
                    extra_views[other] = {"status": "skipped", "reason": f"need {other} BASE preds from the CPU campaign"}
                    continue
                Xo = _sanitize_np(va.select(nums).fill_null(0).to_numpy())
                Xo = Xo / np.clip(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-8, None)
                ao = va["AIRPORT"].to_numpy()
                r_o = None
                if scope == "route" and "RUNWAY_mvt" in va.columns and rwy_tr is not None:
                    r_o = va["RUNWAY_mvt"].fill_null("__NA__").cast(pl.String).to_numpy()
                rhat_o, used = retrieve(Xtr, resid, airports_tr, Xo, ao, scope, rwy_tr, r_o)
                pred_o = np.clip(va["bp"].to_numpy() + rhat_o, 0, None)
                part = _compose(cfg, other, va.with_columns(pl.Series("prediction", pred_o)), pred_o, dest=dest)
                extra_views[other] = compact_score(part) if part.get("rmse_seconds") is not None else part
                persist_repo_file(cfg, Path(dest) / f"predictions_{other}.parquet")
                print(f"extra view {other} rmse={extra_views[other].get('rmse_seconds')}", flush=True)
            except Exception as exc:
                print(f"extra view {other} failed: {exc!r}", flush=True)
                extra_views[other] = extra_view_exception(exc, view=other)
    return {
        "status": "completed",
        "rmse_seconds": scored.get("rmse_seconds") or scored.get("overall", {}).get("rmse_seconds"),
        "sse": scored.get("sse"),
        "vs_base": scored.get("vs_base"),
        "backend": backend,
        "normalized": True,
        "airport_scoped": True,
        "scope": scope,
        "index_rows": train.height,
        "views": {view: compact_score(scored), **extra_views},
        "conclusion": f"{scope}-scoped normalized residual retrieval in F3 space",
        "hypothesis": "analog residuals beat raw-target kNN",
    }


def job_gpu08(cfg: CampaignConfig, dest) -> dict[str, Any]:
    from pathlib import Path

    osm = locate(Path("data/external/osm/parking_positions.parquet"), cfg)
    if osm is None:
        return {
            "status": "skipped",
            "reason": "F4/OSM parking extract is absent; graph experiment is not justified",
            "conclusion": "SKIP — CPU-12 prerequisite missing",
            "hypothesis": "taxiway graph residual (conditional on F4)",
        }
    geo_metrics = cfg.artifact_root / "CPU-12-geometry" / "metrics.json"
    if not geo_metrics.exists():
        return {
            "status": "skipped",
            "reason": "CPU-12 has not finished; do not run a GNN before static geometry is scored",
            "conclusion": "SKIP — wait for CPU notebook",
            "hypothesis": "taxiway graph residual (conditional on F4)",
        }
    import json

    try:
        prior = json.loads(geo_metrics.read_text())
    except json.JSONDecodeError:
        return {
            "status": "skipped",
            "reason": "need CPU-12 completed (status=unreadable)",
            "conclusion": "SKIP — wait for CPU notebook",
            "hypothesis": "taxiway graph residual (conditional on F4)",
        }
    if prior.get("status") != "completed":
        if is_terminal_skip(prior):
            return {
                "status": "skipped",
                "reason": "CPU-12 parked; a taxiway graph is not justified",
                "conclusion": "SKIP — static geometry already parked",
                "hypothesis": "taxiway graph residual (conditional on F4)",
                "prior_status": prior.get("status"),
            }
        return {
            "status": "skipped",
            "reason": f"need CPU-12 completed (status={prior.get('status')})",
            "conclusion": "SKIP — wait for CPU notebook",
            "hypothesis": "taxiway graph residual (conditional on F4)",
            "prior_status": prior.get("status"),
        }
    delta = prior.get("delta_sse_vs_base")
    if delta is None:
        delta = (prior.get("vs_base") or {}).get("delta_sse_vs_base")
    if not isinstance(delta, (int, float)) or delta <= 0:
        return {
            "status": "skipped",
            "reason": "CPU-12 geometry did not reduce composed SSE; skip graph",
            "conclusion": "SKIP — static geometry already failed",
            "prior": prior,
        }
    return {
        "status": "skipped",
        "reason": "OSM extract has parking positions, not a taxiway intersection graph; a GNN is not justified from stands-only",
        "conclusion": "SKIP — no route graph available even though F4 exists",
        "prior": prior,
        "hypothesis": "route graph embedding + tabular residual",
    }


def job_gpu09(cfg: CampaignConfig, dest) -> dict[str, Any]:
    """Final ensemble after GPU jobs. Weights from tune/OOF only; includes CPU + GPU candidates."""
    view = "V1" if "V1" in cfg.views else cfg.views[0]
    out = blend_completed_candidates(
        cfg, Path(dest), view=view, candidate_ids=CPU_BLEND_IDS + GPU_BLEND_IDS
    )
    out.setdefault("hypothesis", "CPU+GPU complementary OOF models improve composed RMSE")
    return out


GPU_JOBS = [
    ("GPU-01-transformer", "airport event-sequence Transformer residual", job_gpu01),
    ("GPU-01b-raw-target", "same Transformer on raw taxi (secondary)", job_gpu01b),
    ("GPU-02-tcn", "TCN residual on the same sequences", job_gpu02),
    ("GPU-03-gru", "improved GRU residual", job_gpu03),
    ("GPU-04-fusion", "sequence encoder + BASE tabular + BASE pred + F3 fusion", job_gpu04),
    ("GPU-05-route-seq", "sequence congestion residual after route/base", job_gpu05),
    ("GPU-06-mixture", "neural soft mixture / distributional residual", job_gpu06),
    ("GPU-07-faiss", "airport-scoped FAISS residual retrieval in F3 space", job_gpu07),
    ("GPU-08-graph", "optional graph model if F4 geometry works", job_gpu08),
    ("GPU-09-ensemble", "OOF blend of CPU + GPU candidates after sequences", job_gpu09),
]


def run_gpu_campaign(cfg: CampaignConfig) -> list[dict[str, Any]]:
    if not cfg.smoke:
        cfg.allow_rebuild_cpu_caches = False
    print("=== GPU experiment inventory ===", flush=True)
    for exp_id, hypothesis, _fn in GPU_JOBS:
        print(f"{exp_id:28s} {hypothesis}", flush=True)
    print("hydrate", hydrate_from_drive(cfg), flush=True)
    print(f"index synced n={len(sync_index_from_job_dirs(cfg))}", flush=True)
    require_cuda(cfg)
    results = []
    for exp_id, hypothesis, fn in GPU_JOBS:
        results.append(run_isolated(cfg, exp_id, hypothesis, lambda dest, f=fn: f(cfg, dest)))
        _cuda_empty()
    catalog = campaign_job_catalog()
    write_campaign_summary(cfg, "gpu", catalog)
    print_campaign_status(inventory_status(cfg, catalog))
    return results
