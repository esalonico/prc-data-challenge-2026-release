# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Month-isolated, horizon-censored lookback columns (SA-EMA-X1 item 1; D-081 takeaway 1).

The R2 temporal-v2 "seconds since previous event" columns were built over one contiguous
DEP/ARR table per run scope.  On the all-year + ranking scope that table jumps over the
Feb–Jun 2026 gap, and January 2026 reaches back into 2025, so ranking rows see gaps that no
training row can see (retro-audit A §4.1; teammate-review parity screen §5.3).

``isocap_v1`` fixes the construction, identically for every period (train, JAN_JUL, Nov/Dec,
ranking):

* **Month isolation.**  Temporal-v2 context is computed per UTC calendar month of
  ``MVT_TIME_UTC_mvt`` (DEP rows and challenge-airport ARR rows of that month only).  All
  temporal-v2 columns are affected at month edges; the gap columns everywhere.
* **Horizon censoring** of every lookback duration ``g`` with horizon ``H`` (fixed below, chosen
  from the 2025 D-month distribution only):

  - a previous event exists in the month and ``g <= H``  ->  ``g``;
  - a previous event exists and ``g > H``                ->  ``H`` ("none within H");
  - no previous event in the month and ``t - month_start >= H``  ->  ``H`` (known: none within H);
  - no previous event and ``t - month_start < H``        ->  NaN (censored by the month edge).

  The value therefore never depends on how much history the frame holds beyond ``H``: it is the
  same for a row whether it sits in a one-month frame or in a multi-year one.

No label, block time, AOBT or ARVT is read here.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import numpy as np
import polars as pl

MODE = "isocap_v1"

TEMPORAL_LOOKBACK_COLUMNS = (
    "ctx_sec_since_prev_dep",
    "ctx_sec_since_prev_arr",
    "ctx_sec_since_prev_rwy_dep",
    "ctx_sec_since_prev_rwy_arr",
)
NATIVE_LOOKBACK_COLUMNS = ("native_same_runway_age_sec",)

# Rule (pre-registered):
#   H_c = min(86400, ceil_to_hour(p99.9 of column c)) over every 2025 D-month DEP row
#   (months 2,3,4,5,6,8,9,10; month-isolated, uncapped build; label-free).
# Derived by scripts/build_temporal_isocap.py --derive-caps (derive_caps.json): p99.9 = 10,908 / 3,863 /
# 85,797 / 2,373,069 / 2,164,675 s -> caps 4 h / 2 h / 24 h / 24 h / 24 h.  Frozen.
HORIZON_CAPS_V1: dict[str, float] = {
    "ctx_sec_since_prev_dep": 14400.0,
    "ctx_sec_since_prev_arr": 7200.0,
    "ctx_sec_since_prev_rwy_dep": 86400.0,
    "ctx_sec_since_prev_rwy_arr": 86400.0,
    "native_same_runway_age_sec": 86400.0,
}


def month_start_epoch(year: int, month: int) -> float:
    return datetime(year, month, 1, tzinfo=UTC).timestamp()


def derive_cap(values: np.ndarray, *, quantile: float = 0.999, ceiling: float = 86400.0) -> float:
    """``min(ceiling, ceil_to_hour(q-quantile))`` of the finite values."""

    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return ceiling
    q = float(np.quantile(v, quantile))
    return float(min(ceiling, np.ceil(q / 3600.0) * 3600.0))


def censor(values: np.ndarray, t_epoch: np.ndarray, month_start: np.ndarray | float, cap: float) -> np.ndarray:
    """Apply the horizon-censoring rule to one lookback column (float64 in, float64 out)."""

    v = np.asarray(values, dtype=np.float64)
    t = np.asarray(t_epoch, dtype=np.float64)
    elapsed = t - np.asarray(month_start, dtype=np.float64)
    out = v.copy()
    has = np.isfinite(v)
    out[has & (v > cap)] = cap
    out[~has & (elapsed >= cap)] = cap
    return out


def month_start_expr(col: str = "MVT_TIME_UTC_mvt") -> pl.Expr:
    return pl.col(col).dt.truncate("1mo").dt.epoch("s").cast(pl.Float64)


def apply_caps(
    frame: pl.DataFrame,
    *,
    t_epoch: np.ndarray,
    month_start: np.ndarray,
    columns: tuple[str, ...],
    caps: Mapping[str, float] = HORIZON_CAPS_V1,
) -> pl.DataFrame:
    """Return ``frame`` with each present lookback column censored (dtype kept as Float32)."""

    out = []
    for c in columns:
        if c not in frame.columns:
            continue
        v = frame[c].cast(pl.Float64).fill_nan(None).to_numpy()
        out.append(pl.Series(c, censor(v, t_epoch, month_start, float(caps[c])).astype(np.float32)))
    return frame.with_columns(out) if out else frame


# --------------------------------------------------------------------------- support-count caps (D-082 scope addition)
# ADS-B support / sample counts whose ranking values exceed the training support because they depend on the
# reference pool or frame size (reference-cell sizes of the gate/first-seen bias tables and the stand centroid)
# or on the 2026 sampling density (samples after the gate, map points).  isocap_v1 caps each at its 2025 D-fit
# training p99 (``derive_support_caps.json``; label-free), identically for train, eval and ranking rows.
SUPPORT_CAPS_V1: dict[str, float] = {
    "adsb_bias_n": 80764.0,
    "adsb_bias_first_n": 64747.0,
    "adsb_centroid_n": 1119.0,
    "adsb_q_n_after": 271.0,
    "adsb3_map_n": 248.0,
}


def cap_support_counts(frame: pl.DataFrame, caps: Mapping[str, float] = SUPPORT_CAPS_V1) -> pl.DataFrame:
    """``min(count, cap)`` for every present support-count column; nulls stay null; dtype kept."""

    exprs = [pl.when(pl.col(c) > caps[c]).then(pl.lit(caps[c])).otherwise(pl.col(c)).cast(frame.schema[c]).alias(c)
             for c in caps if c in frame.columns]
    return frame.with_columns(exprs) if exprs else frame
