# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-NNX model zoo (D-065 search space).

Every model maps ``(cat int64 [B, C], dense float [B, D], ctx)`` to a standardised target
prediction ``[B]`` (TabM: ``[B, k]`` member predictions; the trainer averages them for prediction
and trains on each member's loss).  ``ctx`` is ``None`` for tabular models and a dict of gathered
neighbour tensors for temporal models (see ``train.py``).

Architectures:
  mlp       V14's embedding MLP (Linear-BN-SiLU-Dropout blocks); optional PLR numerics.
  resnet    ResNet-MLP (Gorishniy et al. 2021).
  dcn       DCN-v2 (stacked full/low-rank cross layers + deep MLP).
  tabm      TabM (BatchEnsemble MLP, k members; Gorishniy et al. 2024).
  ftt       FT-Transformer with grouped numeric tokens (tractable token count).
  ctx       temporal/context model: focal MLP trunk + attention (or GRU) over the K preceding /
            following departures at the airport and on the runway.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def emb_dim(n: int) -> int:
    return int(min(24, max(4, round(1.6 * n**0.56))))


def act(name: str) -> nn.Module:
    return {"silu": nn.SiLU(), "relu": nn.ReLU(), "gelu": nn.GELU()}[name]


def norm(name: str, d: int) -> nn.Module:
    return {"bn": nn.BatchNorm1d(d), "ln": nn.LayerNorm(d), "none": nn.Identity()}[name]


class PLR(nn.Module):
    """Periodic embeddings -> Linear -> ReLU per numeric feature (Gorishniy et al. 2022, "PLR").

    ``lite``: the linear layer is shared across features (PLR-lite of TabM)."""

    def __init__(self, n: int, k: int = 16, sigma: float = 0.1, d: int = 16, lite: bool = False):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n, k) * sigma)
        self.lite = lite
        if lite:
            self.lin = nn.Linear(2 * k, d)
        else:
            self.w = nn.Parameter(torch.empty(n, 2 * k, d))
            self.b = nn.Parameter(torch.zeros(n, d))
            nn.init.uniform_(self.w, -1 / math.sqrt(2 * k), 1 / math.sqrt(2 * k))
        self.out_dim = n * d

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B, n]
        v = 2 * math.pi * self.freq[None] * x[..., None]
        v = torch.cat([torch.cos(v), torch.sin(v)], dim=-1)  # [B, n, 2k]
        if self.lite:
            h = self.lin(v)
        else:
            h = torch.einsum("bnk,nkd->bnd", v, self.w) + self.b
        return torch.relu(h).flatten(1)


class InputEncoder(nn.Module):
    """Categorical embeddings (V14 dims) + numerics (raw quantile-normal or PLR) + NaN flags."""

    def __init__(self, sizes: list[int], n_num: int, n_dense: int, plr: dict | None = None):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, emb_dim(n)) for n in sizes])
        self.n_num = n_num
        self.plr = PLR(n_num, **plr) if plr else None
        num_dim = self.plr.out_dim if self.plr else n_num
        self.out_dim = sum(emb_dim(n) for n in sizes) + num_dim + (n_dense - n_num)

    def forward(self, cat: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        parts = [e(cat[:, i]) for i, e in enumerate(self.embs)]
        if self.plr is not None:
            parts += [self.plr(dense[:, : self.n_num]), dense[:, self.n_num :]]
        else:
            parts.append(dense)
        return torch.cat(parts, dim=1)


def mlp_stack(d_in: int, widths: list[int], dropout: float, norm_name: str = "bn", act_name: str = "silu") -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    prev = d_in
    for w in widths:
        layers += [nn.Linear(prev, w), norm(norm_name, w), act(act_name), nn.Dropout(dropout)]
        prev = w
    return nn.Sequential(*layers), prev


class MLP(nn.Module):
    def __init__(self, sizes, n_num, n_dense, widths=(512, 512, 256), dropout=0.1, norm_name="bn", act_name="silu", plr=None):
        super().__init__()
        self.enc = InputEncoder(sizes, n_num, n_dense, plr)
        self.body, d = mlp_stack(self.enc.out_dim, list(widths), dropout, norm_name, act_name)
        self.head = nn.Linear(d, 1)

    def forward(self, cat, dense, ctx=None):
        return self.head(self.body(self.enc(cat, dense))).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, d: int, factor: float, dropout1: float, dropout2: float, norm_name: str, act_name: str):
        super().__init__()
        h = int(d * factor)
        self.f = nn.Sequential(norm(norm_name, d), nn.Linear(d, h), act(act_name), nn.Dropout(dropout1), nn.Linear(h, d), nn.Dropout(dropout2))

    def forward(self, x):
        return x + self.f(x)


class ResNet(nn.Module):
    def __init__(self, sizes, n_num, n_dense, d=512, n_blocks=4, factor=2.0, dropout1=0.15, dropout2=0.0, norm_name="bn", act_name="relu", plr=None):
        super().__init__()
        self.enc = InputEncoder(sizes, n_num, n_dense, plr)
        self.inp = nn.Linear(self.enc.out_dim, d)
        self.blocks = nn.Sequential(*[ResBlock(d, factor, dropout1, dropout2, norm_name, act_name) for _ in range(n_blocks)])
        self.head = nn.Sequential(norm(norm_name, d), act(act_name), nn.Linear(d, 1))

    def forward(self, cat, dense, ctx=None):
        return self.head(self.blocks(self.inp(self.enc(cat, dense)))).squeeze(-1)


class Cross(nn.Module):
    def __init__(self, d: int, rank: int | None):
        super().__init__()
        if rank:
            self.u = nn.Linear(d, rank, bias=False)
            self.v = nn.Linear(rank, d)
        else:
            self.u = None
            self.w = nn.Linear(d, d)

    def forward(self, x0, x):
        wx = self.v(self.u(x)) if self.u is not None else self.w(x)
        return x0 * wx + x


class DCN(nn.Module):
    """DCN-v2, stacked: input -> n_cross cross layers -> deep MLP -> head (or parallel)."""

    def __init__(self, sizes, n_num, n_dense, n_cross=3, rank=None, widths=(512, 512, 256), dropout=0.1, parallel=False, plr=None):
        super().__init__()
        self.enc = InputEncoder(sizes, n_num, n_dense, plr)
        d = self.enc.out_dim
        self.bn0 = nn.BatchNorm1d(d)
        self.cross = nn.ModuleList([Cross(d, rank) for _ in range(n_cross)])
        self.parallel = parallel
        self.deep, dd = mlp_stack(d, list(widths), dropout)
        self.head = nn.Linear(dd + (d if parallel else 0), 1)

    def forward(self, cat, dense, ctx=None):
        x0 = self.bn0(self.enc(cat, dense))
        x = x0
        for c in self.cross:
            x = c(x0, x)
        if self.parallel:
            return self.head(torch.cat([x, self.deep(x0)], dim=1)).squeeze(-1)
        return self.head(self.deep(x)).squeeze(-1)


class BELinear(nn.Module):
    """BatchEnsemble linear: y_i = ((x_i * r_i) W) * s_i + b_i for member i."""

    def __init__(self, d_in: int, d_out: int, k: int, adapters: bool = True):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.adapters = adapters
        if adapters:
            self.r = nn.Parameter(torch.ones(k, d_in))
            self.s = nn.Parameter(torch.ones(k, d_out))
            with torch.no_grad():  # TabM: random-sign init of the first adapter
                self.r.copy_(torch.where(torch.rand(k, d_in) < 0.5, -1.0, 1.0))
        self.b = nn.Parameter(torch.zeros(k, d_out))
        bound = 1 / math.sqrt(d_in)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):  # x [B, k, d_in]
        if self.adapters:
            return self.lin(x * self.r) * self.s + self.b
        return self.lin(x) + self.b


class TabM(nn.Module):
    """TabM: an MLP whose every linear layer is a BatchEnsemble layer; k members, mean prediction.

    ``mini``: adapters only in the first layer (TabM-mini); later layers are shared, with
    member-specific biases/heads."""

    def __init__(self, sizes, n_num, n_dense, k=16, widths=(512, 512, 512), dropout=0.1, mini=False, act_name="relu", plr=None):
        super().__init__()
        self.enc = InputEncoder(sizes, n_num, n_dense, plr)
        self.k = k
        layers = []
        prev = self.enc.out_dim
        for i, w in enumerate(widths):
            layers.append(BELinear(prev, w, k, adapters=(i == 0) or not mini))
            prev = w
        self.layers = nn.ModuleList(layers)
        self.act = act(act_name)
        self.drop = nn.Dropout(dropout)
        self.head_w = nn.Parameter(torch.empty(k, prev))
        nn.init.uniform_(self.head_w, -1 / math.sqrt(prev), 1 / math.sqrt(prev))
        self.head_b = nn.Parameter(torch.zeros(k))

    def forward(self, cat, dense, ctx=None):
        x = self.enc(cat, dense)
        x = x[:, None, :].expand(-1, self.k, -1)
        for layer in self.layers:
            x = self.drop(self.act(layer(x)))
        return (x * self.head_w).sum(-1) + self.head_b  # [B, k]


class FTT(nn.Module):
    """FT-Transformer with grouped numeric tokens: each group of ``group`` consecutive dense
    columns becomes one token through its own linear map; each categorical is one token."""

    def __init__(self, sizes, n_num, n_dense, d=96, n_layers=3, heads=8, group=8, ffn_factor=4 / 3, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, d) for n in sizes])
        self.group = group
        self.n_groups = math.ceil(n_dense / group)
        self.pad = self.n_groups * group - n_dense
        self.gw = nn.Parameter(torch.empty(self.n_groups, group, d))
        nn.init.normal_(self.gw, std=1 / math.sqrt(group))
        self.gb = nn.Parameter(torch.zeros(self.n_groups, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, int(d * ffn_factor * 2), dropout, activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, cat, dense, ctx=None):
        b = dense.shape[0]
        if self.pad:
            dense = torch.nn.functional.pad(dense, (0, self.pad))
        x = dense.view(b, self.n_groups, self.group)
        num_tok = torch.einsum("bgk,gkd->bgd", x, self.gw) + self.gb
        cat_tok = torch.stack([e(cat[:, i]) for i, e in enumerate(self.embs)], dim=1)
        tok = torch.cat([self.cls.expand(b, -1, -1), cat_tok, num_tok], dim=1)
        return self.head(self.tr(tok)[:, 0]).squeeze(-1)


class Context(nn.Module):
    """Temporal/context model.

    Focal: V14-style encoder + MLP trunk -> h.  Neighbours (K airport + K runway departures,
    preceding and following): their dense vector, a few of their categorical embeddings (shared
    tables), relative clocks (signed-log seconds) and a slot embedding -> Linear -> d tokens.
    ``mixer='attn'``: a focal query token plus the neighbour tokens through a Transformer encoder
    (key padding for missing neighbours); ``'gru'``: GRUs over the four nearest-first sequences.
    The context vector joins h in the head.
    """

    def __init__(self, sizes, n_num, n_dense, n_slots, widths=(512, 512), head_widths=(256,), dropout=0.1, d=64, n_layers=2, heads=4, mixer="attn", nb_cats=(1, 2, 3, 4), n_rel=6, nb_dense_proj=64, n_nb_dense=None, plr=None,
                 n_arr_slots=0, n_arr_feat=4):
        super().__init__()
        self.enc = InputEncoder(sizes, n_num, n_dense, plr)
        self.body, dh = mlp_stack(self.enc.out_dim, list(widths), dropout)
        self.nb_cats = list(nb_cats)
        self.nb_proj = nn.Linear(n_nb_dense or n_dense, nb_dense_proj)
        nb_in = nb_dense_proj + sum(emb_dim(sizes[i]) for i in self.nb_cats) + n_rel
        self.tok = nn.Sequential(nn.Linear(nb_in, d), nn.GELU(), nn.Linear(d, d))
        self.slot = nn.Embedding(n_slots, d)
        self.mixer = mixer
        self.heads = heads
        if mixer == "xattn":  # focal-query cross-attention pooling (+ masked mean), no self-attention
            self.q = nn.Linear(dh, d)
            self.kv = nn.Linear(d, 2 * d)
            self.o = nn.Linear(d, d)
            dc = 2 * d
        elif mixer == "attn":
            self.q = nn.Linear(dh, d)
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            dc = d
        else:
            self.gru = nn.GRU(d, d, batch_first=True)
            dc = 4 * d
            if n_arr_slots:  # two more nearest-first sequences: preceding and following landings
                dc += 2 * d
        self.out_norm = nn.LayerNorm(dc)
        self.head_body, dhh = mlp_stack(dh + dc, list(head_widths), dropout)
        self.head = nn.Linear(dhh, 1)
        self.n_arr_slots = int(n_arr_slots)
        if self.n_arr_slots:  # ctx-v2 arrival tokens (created last: the default model's init is unchanged)
            self.arr_tok = nn.Sequential(nn.Linear(n_arr_feat, d), nn.GELU(), nn.Linear(d, d))
            self.arr_slot = nn.Embedding(self.n_arr_slots, d)

    def forward(self, cat, dense, ctx):
        h = self.body(self.enc(cat, dense))
        nb_cat, nb_dense, rel, mask = ctx["cat"], ctx["dense"], ctx["rel"], ctx["mask"]  # [B,S,C] [B,S,D] [B,S,R] [B,S]
        parts = [self.nb_proj(nb_dense)] + [self.enc.embs[i](nb_cat[..., i]) for i in self.nb_cats] + [rel]
        tok = self.tok(torch.cat(parts, dim=-1)) + self.slot.weight[None]
        tok = tok * mask[..., None]
        arr_tok = None
        if self.n_arr_slots and "arr" in ctx:
            am = ctx["arr"]["mask"]
            arr_tok = (self.arr_tok(ctx["arr"]["feat"]) + self.arr_slot.weight[None]) * am[..., None]
            if self.mixer != "gru":
                tok = torch.cat([tok, arr_tok], dim=1)
                mask = torch.cat([mask, am], dim=1)
        if self.mixer == "xattn":
            b, S, d = tok.shape
            hd = d // self.heads
            q = self.q(h).view(b, self.heads, hd)
            k, v = self.kv(tok).view(b, S, 2, self.heads, hd).unbind(2)
            sc = torch.einsum("bhd,bshd->bhs", q, k) / math.sqrt(hd) + (mask[:, None, :] - 1.0) * 1e4
            w = torch.softmax(sc, dim=-1)
            att = self.o(torch.einsum("bhs,bshd->bhd", w, v).reshape(b, d))
            mean = tok.sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
            c = torch.cat([att, mean], dim=1)
        elif self.mixer == "attn":
            q = self.q(h)[:, None, :]
            seq = torch.cat([q, tok], dim=1)
            pad = torch.cat([torch.zeros_like(mask[:, :1], dtype=torch.bool), mask < 0.5], dim=1)
            c = self.tr(seq, src_key_padding_mask=pad)[:, 0]
        else:
            s = tok.shape[1]
            ka, kr = ctx["k_airport"], ctx["k_runway"]
            bounds = [(0, ka), (ka, 2 * ka), (2 * ka, 2 * ka + kr), (2 * ka + kr, s)]
            outs = []
            for lo, hi in bounds:  # nearest-first sequences; feed farthest-first so the last state is the nearest
                seg = torch.flip(tok[:, lo:hi], dims=[1])
                _, hn = self.gru(seg)
                outs.append(hn[0])
            if arr_tok is not None:
                ka_ = arr_tok.shape[1] // 2
                for lo, hi in ((0, ka_), (ka_, 2 * ka_)):
                    _, hn = self.gru(torch.flip(arr_tok[:, lo:hi], dims=[1]))
                    outs.append(hn[0])
            c = torch.cat(outs, dim=1)
        z = torch.cat([h, self.out_norm(c)], dim=1)
        return self.head(self.head_body(z)).squeeze(-1)


def build(cfg: dict, sizes: list[int], n_num: int, n_dense: int, n_slots: int = 0) -> nn.Module:
    arch = cfg["arch"]
    kw = dict(cfg.get("model", {}))
    if arch == "mlp_v14":  # the exact V14 network (train_nn_matrix.build_model)
        import train_nn_matrix as tnm

        return tnm.build_model(sizes, n_dense, tuple(kw.get("widths", (512, 512, 256))), kw.get("dropout", 0.1))
    cls = {"mlp": MLP, "resnet": ResNet, "dcn": DCN, "tabm": TabM, "ftt": FTT, "ctx": Context}[arch]
    if arch == "ctx":
        return cls(sizes, n_num, n_dense, n_slots, **kw)
    return cls(sizes, n_num, n_dense, **kw)
