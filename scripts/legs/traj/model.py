# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-TRAJ model: PLR tabular trunk (SA-NNX ``models.py``) + an encoder over the focal departure's
own ADS-B trajectory (``extract.py`` points), joined before the head.

Point features built on the device from the stored points, the row anchors and the focal clocks
(``point_features``): relative time (linear and signed-log, to take-off and to the focal AOBT), x/y
in the airport frame, ground speed (+ missing flag), on-ground flag, heading sin/cos, km to the
nearest runway threshold, raw-sample weight, gap since the previous point, and km to the OSM stand,
the month-isolated stand centroid and the assigned runway's threshold (each + missing flag).

Encoders: ``conv`` (residual dilated 1D convolutions, masked) or ``gru`` (left-aligned sequence),
pooled by masked mean, masked max, the last point and (conv) a tabular-query attention pool.  Rows
without a track use a learned "no track" vector.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.utils.checkpoint
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nnx"))
import models as nnxm  # noqa: E402

N_PT_IN = 21


def slog_s(x_sec: torch.Tensor) -> torch.Tensor:
    return torch.sign(x_sec) * torch.log1p(torch.abs(x_sec) / 60.0) / 3.0


def _dist(x, y, ax, ay):
    ok = torch.isfinite(ax) & torch.isfinite(ay)
    d = torch.sqrt((x - torch.nan_to_num(ax)[:, None]) ** 2 + (y - torch.nan_to_num(ay)[:, None]) ** 2)
    d = torch.log1p(d * 10.0) * ok[:, None]
    flag = (~ok).float()[:, None].expand_as(d)
    return d, flag


def point_features(pts: torch.Tensor, mask: torch.Tensor, row: torch.Tensor, clock: torch.Tensor) -> torch.Tensor:
    """pts [B,N,9] (t_min, x_km, y_km, gs_kt, on_ground, hdg_sin, hdg_cos, d_thr_km, log_n);
    row [B,9] (has_track, join_src, log_n_raw, osm_x, osm_y, rwy_x, rwy_y, c_x, c_y); clock [B,4]."""
    t = pts[..., 0]
    x, y = pts[..., 1], pts[..., 2]
    gs = pts[..., 3]
    gs_nan = ~torch.isfinite(gs)
    gs = torch.where(gs_nan, torch.zeros_like(gs), gs).clamp(0, 250) / 30.0
    t_sec = t * 60.0
    t_aobt = t_sec + clock[:, 0:1]  # point time minus focal AOBT (clock0 = take-off - AOBT)
    gap = torch.zeros_like(t)
    gap[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp(min=0)
    d_osm, f_osm = _dist(x, y, row[:, 3], row[:, 4])
    d_rwy, f_rwy = _dist(x, y, row[:, 5], row[:, 6])
    d_c, f_c = _dist(x, y, row[:, 7], row[:, 8])
    feats = torch.stack([
        t / 30.0, slog_s(t_sec), slog_s(t_aobt), x / 5.0, y / 5.0, gs, gs_nan.float(), pts[..., 4],
        pts[..., 5], pts[..., 6], torch.log1p(pts[..., 7] * 10.0) / 3.0, pts[..., 8] / 3.0, torch.log1p(gap) / 3.0,
        d_osm / 3.0, f_osm, d_rwy / 3.0, f_rwy, d_c / 3.0, f_c,
        (row[:, 1:2] == 3).float().expand_as(t), row[:, 2:3].expand_as(t) / 6.0,
    ], dim=-1)
    return torch.nan_to_num(feats) * mask[..., None]


class ConvBlock(nn.Module):
    """Pre-norm residual block: dilated k-tap convolution (as shifted copies + Linear, which is a
    Conv1d with zero padding but much faster on MPS) -> GELU -> 1x1 -> dropout.  Padded positions are
    zero before and after the convolution, so a row's output does not depend on its padding."""

    def __init__(self, d: int, k: int, dil: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.k, self.dil = k, dil
        self.c1 = nn.Linear(k * d, 2 * d)
        self.c2 = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):  # x [B,N,d]
        n = x.shape[1]
        h = self.norm(x) * mask[..., None]
        p = self.dil * (self.k // 2)
        hp = torch.nn.functional.pad(h, (0, 0, p, p))
        cols = torch.cat([hp[:, j * self.dil : j * self.dil + n] for j in range(self.k)], dim=-1)
        h = self.c2(torch.nn.functional.gelu(self.c1(cols)))
        return (x + self.drop(h)) * mask[..., None]


class _RngReplay:
    """D-109: ``torch.utils.checkpoint`` does not restore the MPS generator, so a recomputed chunk would draw new dropout
    masks.  The first call records the device RNG state; a recompute replays from it (same masks) and then puts the
    generator back where it was, so the RNG stream of the rest of training is that of the run without checkpointing."""

    def __init__(self, fn, device):
        self.fn, self.dev, self.state = fn, device, None

    def _get(self):
        return torch.mps.get_rng_state() if self.dev.type == "mps" else (torch.cuda.get_rng_state(self.dev) if self.dev.type == "cuda" else torch.get_rng_state())

    def _set(self, st):
        if self.dev.type == "mps":
            torch.mps.set_rng_state(st)
        elif self.dev.type == "cuda":
            torch.cuda.set_rng_state(st, self.dev)
        else:
            torch.set_rng_state(st)

    def __call__(self, *args):
        if self.state is None:
            self.state = self._get()
            return self.fn(*args)
        cur = self._get()
        self._set(self.state)
        try:
            return self.fn(*args)
        finally:
            self._set(cur)


class TrajEncoder(nn.Module):
    """Encodes the rows that have a track (in length-sorted chunks, see ``train_traj.Side.seq``);
    rows without a track get the learned ``none`` vector."""

    def __init__(self, d_q: int, n_airports: int, d: int = 96, kind: str = "conv", n_blocks: int = 3, k: int = 5, dropout: float = 0.1, heads: int = 4):
        super().__init__()
        self.kind = kind
        self.inp = nn.Sequential(nn.Linear(N_PT_IN, d), nn.GELU(), nn.Linear(d, d))
        self.ap = nn.Embedding(n_airports, d)
        nn.init.normal_(self.ap.weight, std=0.02)
        if kind == "conv":
            self.blocks = nn.ModuleList([ConvBlock(d, k, 2**i, dropout) for i in range(n_blocks)])
            self.q = nn.Linear(d_q, d)
            self.kv = nn.Linear(d, 2 * d)
            self.heads = heads
            self.out_dim = 4 * d
        else:
            self.gru = nn.GRU(d, d, batch_first=True)
            self.out_dim = 3 * d
        self.none = nn.Parameter(torch.zeros(self.out_dim))
        self.norm = nn.LayerNorm(self.out_dim)
        self.d = d

    def encode(self, feats, mask, lens, airport, h):
        """feats [b,n,F] left-aligned, every row with lens >= 1."""
        b, n, _ = feats.shape
        tok = (self.inp(feats) + self.ap(airport)[:, None, :]) * mask[..., None]
        last_idx = (lens - 1).clamp(min=0)
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)
        ar = torch.arange(b, device=feats.device)
        if self.kind == "conv":
            x = tok
            for blk in self.blocks:
                x = blk(x, mask)
            mean = x.sum(1) / denom
            mx = torch.where(mask[..., None] > 0, x, torch.full_like(x, -1e4)).max(1).values
            last = x[ar, last_idx]
            hd = self.d // self.heads
            q = self.q(h).view(b, self.heads, hd)
            kk, vv = self.kv(x).view(b, n, 2, self.heads, hd).unbind(2)
            sc = torch.einsum("bhd,bnhd->bhn", q, kk) / math.sqrt(hd) + (mask[:, None, :] - 1.0) * 1e4
            att = torch.einsum("bhn,bnhd->bhd", torch.softmax(sc, -1), vv).reshape(b, self.d)
            return torch.cat([mean, mx, last, att], dim=1)
        # left-aligned GRU: the output at the last real step depends on the real steps only
        out, _ = self.gru(tok)
        mean = (out * mask[..., None]).sum(1) / denom
        last = out[ar, last_idx]
        return torch.cat([mean, last, tok.sum(1) / denom], dim=1)

    def forward(self, seq, airport, h):
        B = h.shape[0]
        c = self.none[None, :].repeat(B + 1, 1)  # row B collects the padding copies and is dropped
        rows, parts = [], []
        ckpt = self.training and getattr(self, "checkpoint", False)
        for ch in seq["chunks"]:
            g = ch["gather"]
            if ckpt:  # D-109: activation checkpointing per chunk; memory only (same math, same dropout masks)
                parts.append(torch.utils.checkpoint.checkpoint(_RngReplay(self.encode, ch["feats"].device), ch["feats"], ch["mask"], ch["lens"], airport[g], h[g], use_reentrant=False, preserve_rng_state=False))
            else:
                parts.append(self.encode(ch["feats"], ch["mask"], ch["lens"], airport[g], h[g]))
            rows.append(ch["scatter"])
        if parts:
            c = c.index_put((torch.cat(rows),), torch.cat(parts))
        return self.norm(c[:B])


class TrajNet(nn.Module):
    """PLR tabular trunk (MLP or ResNet body) + trajectory encoder -> head."""

    def __init__(self, sizes, n_num, n_dense, body="mlp", widths=(512, 512), head_widths=(256,), dropout=0.1, plr=None, traj=None, resnet=None):
        super().__init__()
        self.enc = nnxm.InputEncoder(sizes, n_num, n_dense, plr)
        if body == "mlp":
            self.body, dh = nnxm.mlp_stack(self.enc.out_dim, list(widths), dropout)
        else:
            r = dict(d=512, n_blocks=4, factor=2.0, dropout1=0.15, dropout2=0.0)
            r.update(resnet or {})
            self.body = nn.Sequential(nn.Linear(self.enc.out_dim, r["d"]), *[nnxm.ResBlock(r["d"], r["factor"], r["dropout1"], r["dropout2"], "bn", "relu") for _ in range(r["n_blocks"])], nnxm.norm("bn", r["d"]), nn.ReLU())
            dh = r["d"]
        self.traj = TrajEncoder(dh, sizes[0], **(traj or {}))
        self.head_body, dhh = nnxm.mlp_stack(dh + self.traj.out_dim, list(head_widths), dropout)
        self.head = nn.Linear(dhh, 1)

    def forward(self, cat, dense, seq):
        h = self.body(self.enc(cat, dense))
        c = self.traj(seq, cat[:, 0], h)
        return self.head(self.head_body(torch.cat([h, c], dim=1))).squeeze(-1)
