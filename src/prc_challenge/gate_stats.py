# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Shared gate statistics for ``scripts/v4_gate.py`` and ``scripts/oof_gate.py`` (SA-GATE-TOOLS, D-081).

Everything here is a *readout* helper.  None of it selects a model; the decision rules stay in the
pre-registrations that use them.

1. **2026-mix reweighting** (``ranking_mix_weights``).  Each evaluation row gets the weight
   ``share_ranking(cell) / share_eval(cell)``, where a cell is (route, airport, ADS-B coverage class)
   for matched rows and (route, airport) for unmatched rows (``unmatched_by_coverage=True`` splits the
   unmatched cells by coverage class too).  The ranking shares are **row counts of the ranking
   departures** (label-free).  Coverage classes come from the ADS-B block's ``adsb_coverage``:

   ========  ==========================  ===============================================
   class     ``adsb_coverage``           meaning
   ========  ==========================  ===============================================
   gate      3                           stand-gate sample found (off-block observable)
   segment   2                           ground segment, no gate sample
   none      0, 1                        no aircraft join (0) or joined without a segment (1)
   nofile    null or -1                  no ADS-B file for that day (does not occur in 2026)
   ========  ==========================  ===============================================

   Because the ranking set has no ``nofile`` rows, those evaluation rows get weight 0 (the D-fold
   bank has 31.9 % of them).  The weights are normalised to mean 1 over the evaluation rows, so a
   weighted SSE stays on the row-count scale.  The route cells carry the unmatched share: pooled
   numbers are recombined at the ranking unmatched share per airport.  Ranking cells that have no
   evaluation row cannot be represented; their share is reported as ``uncovered_ranking_share``.
   The reweighted numbers are a **veto input only** (never a selector), reported next to the plain ones.

2. **Day-cluster bootstrap** (``day_bootstrap``) of the paired Δ between two predictions: days are
   resampled with replacement *within each month* (stratified), the per-day SSE sums (optionally
   weighted) are recombined, and p05 / p95 are reported for ΔSSE and ΔRMSE, pooled and per month.

3. **Rows behind the gain** (``rows_behind``): the smallest number of rows whose per-row ΔSE, taken
   in order of benefit, carries ``frac`` (default 90 %) of the net ΔSSE.

4. **Seed-noise floor** (``seed_floor``): the SD of the gated score when only one component's seed is
   redrawn (everything else fixed, same rows: paired).  A gated Δ is a difference of two such draws,
   so its noise SD is √2·σ, and the bar is 2·√2·σ.

5. **Permutation control** (``permutation_control``) for classifier-gated arms: the arm is refitted
   with the key column shuffled within fold; its gain must vanish.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

COVERAGE_CLASSES = ("gate", "segment", "none", "nofile")
ROUTES = ("matched", "unmatched")


# --------------------------------------------------------------------------- 2026-mix weights


def coverage_class(cov: Sequence[Any] | np.ndarray) -> np.ndarray:
    """``adsb_coverage`` (3/2/1/0/-1/null) -> class labels (``COVERAGE_CLASSES``)."""

    s = cov if isinstance(cov, pl.Series) else pl.Series("c", list(cov) if not isinstance(cov, np.ndarray) else cov)
    c = s.cast(pl.Float64, strict=False).fill_nan(None).fill_null(-1.0).to_numpy()
    out = np.full(c.shape, "nofile", dtype=object)
    out[(c == 0) | (c == 1)] = "none"
    out[c == 2] = "segment"
    out[c == 3] = "gate"
    return out


def mix_cells(airport: Sequence[Any] | np.ndarray, matched: Sequence[bool] | np.ndarray, cov: Sequence[Any] | np.ndarray,
              unmatched_by_coverage: bool = False) -> np.ndarray:
    """Cell key per row: ``matched|<AIRPORT>|<class>`` or ``unmatched|<AIRPORT>[|<class>]``."""

    ap = np.asarray(airport).astype(str)
    m = np.asarray(matched, dtype=bool)
    cls = coverage_class(cov)
    route = np.where(m, "matched", "unmatched")
    with_cls = m | bool(unmatched_by_coverage)
    keys = np.char.add(np.char.add(route.astype(str), "|"), ap)
    keys = np.where(with_cls, np.char.add(np.char.add(keys, "|"), cls.astype(str)), keys)
    return keys.astype(object)


def ranking_mix_weights(eval_cells: np.ndarray, ranking_cells: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Row weights ``share_ranking(cell) / share_eval(cell)``, normalised to mean 1 over evaluation rows.

    Returns (weights, info).  ``info`` holds the per-cell table, the ranking share that no evaluation
    row can represent, the evaluation share with weight 0, and Kish's effective sample size.
    """

    ev = np.asarray(eval_cells, dtype=object)
    rk = np.asarray(ranking_cells, dtype=object)
    if ev.size == 0 or rk.size == 0:
        raise ValueError("empty cell arrays")
    e_keys, e_inv, e_cnt = np.unique(ev.astype(str), return_inverse=True, return_counts=True)
    r_keys, r_cnt = np.unique(rk.astype(str), return_counts=True)
    r_share = dict(zip(r_keys.tolist(), (r_cnt / r_cnt.sum()).tolist()))
    e_share = e_cnt / e_cnt.sum()
    ratio = np.array([r_share.get(k, 0.0) / s for k, s in zip(e_keys.tolist(), e_share.tolist())])
    w = ratio[e_inv]
    covered = sum(r_share.get(k, 0.0) for k in e_keys.tolist())
    if covered <= 0:
        raise ValueError("no ranking cell is represented in the evaluation rows")
    w = w / w.mean()
    ess = float(w.sum() ** 2 / np.sum(w**2)) if np.any(w > 0) else 0.0
    cells = {
        k: {"n_eval": int(n), "share_eval": float(s), "share_ranking": float(r_share.get(k, 0.0)), "weight": float(r_share.get(k, 0.0) / s / (covered))}
        for k, n, s in zip(e_keys.tolist(), e_cnt.tolist(), e_share.tolist())
    }
    missing = {k: float(v) for k, v in r_share.items() if k not in cells}
    info = {
        "n_eval": int(ev.size),
        "n_ranking": int(rk.size),
        "cells": cells,
        "uncovered_ranking_cells": missing,
        "uncovered_ranking_share": float(sum(missing.values())),
        "zero_weight_eval_share": float(e_share[[r_share.get(k, 0.0) == 0 for k in e_keys.tolist()]].sum()),
        "effective_n": ess,
        "max_weight": float(w.max()),
    }
    return w, info


def ranking_cells_from_matrix(path: Path | str, unmatched_by_coverage: bool = False) -> np.ndarray:
    """Cells of the ranking departures, from label-free columns of the ranking eval matrix."""

    f = pl.read_parquet(path, columns=["AIRPORT", "nm_unmatched", "adsb_coverage"])
    return mix_cells(f["AIRPORT"].to_numpy(), f["nm_unmatched"].cast(pl.Int8).to_numpy() == 0, f["adsb_coverage"].to_numpy(), unmatched_by_coverage)


def weighted_rmse(err2: np.ndarray, w: np.ndarray | None = None, mask: np.ndarray | None = None) -> float | None:
    e = np.asarray(err2, dtype=np.float64)
    ww = np.ones_like(e) if w is None else np.asarray(w, dtype=np.float64)
    if mask is not None:
        e, ww = e[mask], ww[mask]
    s = ww.sum()
    return float(math.sqrt(float((ww * e).sum()) / s)) if e.size and s > 0 else None


# --------------------------------------------------------------------------- day keys


def day_keys(ids: np.ndarray, base_dir: Path | str, periods: Sequence[str] | None = None) -> np.ndarray:
    """UTC take-off date (``YYYY-MM-DD``) per MVT id, from the S4 base caches (label-free clocks)."""

    base_dir = Path(base_dir)
    files = sorted(base_dir.glob("base_*.parquet")) if periods is None else [base_dir / f"base_{p}.parquet" for p in sorted(set(periods))]
    frames = [pl.read_parquet(f, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]) for f in files if f.exists()]
    if not frames:
        raise FileNotFoundError(f"no base caches under {base_dir}")
    base = pl.concat(frames).unique("MVT_ID_mvt", keep="first").with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
    q = pl.DataFrame({"MVT_ID_mvt": np.asarray(ids, dtype=np.float64)})
    t = q.join(base, on="MVT_ID_mvt", how="left", maintain_order="left")["MVT_TIME_UTC_mvt"]
    if t.null_count():
        raise RuntimeError(f"{t.null_count()} rows without a take-off time in {base_dir}")
    return t.dt.strftime("%Y-%m-%d").to_numpy().astype(object)


# --------------------------------------------------------------------------- bootstrap


def _boot_counts(n_days: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=n_boot).astype(np.float64)


def day_bootstrap(y: np.ndarray, base: np.ndarray, cand: np.ndarray, day: np.ndarray, month: np.ndarray | None = None,
                  w: np.ndarray | None = None, mask: np.ndarray | None = None, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Day-cluster bootstrap of the paired Δ (cand − base): ΔSSE and ΔRMSE p05 / p95, pooled and per month.

    Days are resampled within their month (stratified), so every replicate keeps the month mix.
    ``w`` weights rows (the 2026-mix weights); ``mask`` restricts to a row subset (e.g. matched).
    """

    y = np.asarray(y, np.float64)
    keep = np.ones(y.size, bool) if mask is None else np.asarray(mask, bool)
    ww = np.ones(y.size) if w is None else np.asarray(w, np.float64)
    e_b = (y - np.asarray(base, np.float64)) ** 2
    e_c = (y - np.asarray(cand, np.float64)) ** 2
    mon = np.zeros(y.size, int) if month is None else np.asarray(month)
    d = np.asarray(day).astype(str)
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(seed)
    months = sorted(set(mon[idx].tolist()))
    per: dict[str, Any] = {}
    pooled_parts = []
    for m in months:
        r = idx[mon[idx] == m]
        keys, inv = np.unique(d[r], return_inverse=True)
        wn = np.bincount(inv, weights=ww[r], minlength=keys.size)
        sb = np.bincount(inv, weights=ww[r] * e_b[r], minlength=keys.size)
        sc = np.bincount(inv, weights=ww[r] * e_c[r], minlength=keys.size)
        cnt = _boot_counts(keys.size, n_boot, rng)
        bn, bb, bc = cnt @ wn, cnt @ sb, cnt @ sc
        pooled_parts.append((bn, bb, bc))
        per[str(m)] = _summ(wn.sum(), sb.sum(), sc.sum(), bn, bb, bc, keys.size)
    bn = sum(p[0] for p in pooled_parts)
    bb = sum(p[1] for p in pooled_parts)
    bc = sum(p[2] for p in pooled_parts)
    pooled = _summ(float(ww[idx].sum()), float((ww * e_b)[idx].sum()), float((ww * e_c)[idx].sum()), bn, bb, bc,
                   len(set(zip(mon[idx].tolist(), d[idx].tolist()))))
    return {"pooled": pooled, "per_month": per, "n_boot": n_boot, "seed": seed, "weighted": w is not None}


def _summ(n: float, sb: float, sc: float, bn: np.ndarray, bb: np.ndarray, bc: np.ndarray, n_days: int) -> dict[str, Any]:
    ok = bn > 0
    if n <= 0 or not ok.any():
        return {"n_days": int(n_days), "dsse": float(sc - sb), "drmse": None, "dsse_p05": None, "dsse_p95": None, "drmse_p05": None, "drmse_p95": None}
    drmse = np.sqrt(bc[ok] / bn[ok]) - np.sqrt(bb[ok] / bn[ok])
    dsse = (bc - bb)[ok]
    return {
        "n_days": int(n_days),
        "dsse": float(sc - sb),
        "drmse": float(math.sqrt(sc / n) - math.sqrt(sb / n)),
        "dsse_p05": float(np.percentile(dsse, 5)),
        "dsse_p95": float(np.percentile(dsse, 95)),
        "drmse_p05": float(np.percentile(drmse, 5)),
        "drmse_p95": float(np.percentile(drmse, 95)),
    }


# --------------------------------------------------------------------------- rows behind the gain


def rows_behind(dse: np.ndarray, frac: float = 0.9) -> dict[str, Any]:
    """Rows carrying ``frac`` of the net ΔSSE (per-row ΔSE = cand² − base²).

    For a net gain (ΔSSE < 0) the rows are taken from the most improved down; for a net loss from
    the most worsened down.  ``n_rows`` is the smallest count whose cumulative ΔSE reaches
    ``frac`` × the net ΔSSE.
    """

    d = np.asarray(dse, np.float64)
    net = float(d.sum())
    n_changed = int(np.count_nonzero(d))
    if net == 0.0 or n_changed == 0:
        return {"frac": frac, "net_dsse": net, "direction": "none", "n_rows": 0, "n_changed": n_changed}
    sgn = -1.0 if net < 0 else 1.0
    s = np.sort(sgn * d)[::-1]  # most beneficial (for the net direction) first
    cum = np.cumsum(s)
    target = frac * abs(net)
    k = int(np.argmax(cum >= target * (1 - 1e-12)) + 1)
    return {"frac": frac, "net_dsse": net, "direction": "gain" if net < 0 else "loss", "n_rows": k, "n_changed": n_changed,
            "share_of_changed": k / n_changed, "top20_share": float(cum[min(19, cum.size - 1)] / abs(net))}


# --------------------------------------------------------------------------- seed-noise floor


def seed_draws(per_seed: Sequence[np.ndarray], group: int = 1, n_draws: int = 50, seed: int = 0) -> tuple[list[np.ndarray], str]:
    """Redraws of one component: each seed alone (``group`` 1) or means of ``group`` seeds.

    With at least ``2 * group`` seeds the draws are disjoint groups (independent); otherwise they are
    random subsets that overlap, which *understates* the SD (flagged in the returned note).
    """

    arr = [np.asarray(p, np.float64) for p in per_seed]
    k = len(arr)
    if group <= 1:
        return arr, "single seeds"
    if k < group:
        raise ValueError(f"{k} seeds for groups of {group}")
    rng = np.random.default_rng(seed)
    if k >= 2 * group:
        order = rng.permutation(k)
        groups = [order[i : i + group] for i in range(0, k - group + 1, group)]
        note = f"{len(groups)} disjoint groups of {group}"
    else:
        groups = [rng.choice(k, size=group, replace=False) for _ in range(n_draws)]
        note = f"{n_draws} overlapping subsets of {group} from {k} seeds (SD understated)"
    return [np.mean([arr[i] for i in g], axis=0) for g in groups], note


def seed_floor(score: Callable[[np.ndarray], float], draws: Sequence[np.ndarray], group: int = 1, single_seed_sd: float | None = None) -> dict[str, Any]:
    """SD of ``score(draw)`` over the redraws of one component (paired: same rows, everything else fixed).

    ``bar = 2 * sqrt(2) * sd``: a gated Δ is the difference of two draws (SD √2·σ), read at 2 SD.
    If ``single_seed_sd`` is given and ``group > 1``, the scaled estimate ``single_seed_sd / sqrt(group)``
    is also reported and the bar uses the larger of the two SDs (conservative).
    """

    s = np.array([float(score(d)) for d in draws])
    if s.size < 2:
        raise ValueError("a seed floor needs at least two draws")
    sd = float(s.std(ddof=1))
    pair = np.array([a - b for a, b in combinations(s.tolist(), 2)])
    paired_sd = float(math.sqrt(np.mean(pair**2) / 2.0))
    out: dict[str, Any] = {"n_draws": int(s.size), "scores": s.tolist(), "sd": sd, "paired_sd": paired_sd, "group": group}
    use = sd
    if single_seed_sd is not None and group > 1:
        out["scaled_sd"] = float(single_seed_sd / math.sqrt(group))
        use = max(sd, out["scaled_sd"])
    out["bar"] = float(2.0 * math.sqrt(2.0) * use)
    return out


def beyond_floor(delta: float, floor: dict[str, Any]) -> bool:
    """True when |Δ| exceeds the 2√2·σ bar."""

    return abs(delta) > floor["bar"]


# --------------------------------------------------------------------------- permutation control


def shuffle_within(values: np.ndarray, groups: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """Permute ``values`` within each group (e.g. fold); ``groups`` None = one group."""

    v = np.asarray(values).copy()
    if groups is None:
        return v[rng.permutation(v.size)]
    g = np.asarray(groups)
    for key in np.unique(g):
        idx = np.flatnonzero(g == key)
        v[idx] = v[idx[rng.permutation(idx.size)]]
    return v


def permutation_control(fit_delta: Callable[[np.ndarray], float], column: np.ndarray, groups: np.ndarray | None = None,
                        n_perm: int = 3, seed: int = 0, vanish_frac: float = 0.25) -> dict[str, Any]:
    """Shuffled-key control for a classifier-gated arm.

    ``fit_delta(col)`` refits the arm with the key column replaced by ``col`` and returns its ΔSSE
    (or Δ score; negative = gain) against the arm's control.  It is called once with the real column and
    ``n_perm`` times with the column shuffled within ``groups`` (the folds).  The control passes when the
    mean shuffled Δ keeps less than ``vanish_frac`` of the real gain (or the real Δ is not a gain).
    """

    real = float(fit_delta(np.asarray(column)))
    rng = np.random.default_rng(seed)
    perm = [float(fit_delta(shuffle_within(column, groups, rng))) for _ in range(n_perm)]
    mean_perm = float(np.mean(perm))
    retained = mean_perm / real if real < 0 else float("nan")
    passed = bool(real < 0 and mean_perm > vanish_frac * real)
    return {"real_delta": real, "perm_deltas": perm, "perm_mean": mean_perm, "retained_share": retained,
            "vanish_frac": vanish_frac, "gain_vanishes": passed, "real_is_gain": real < 0}
