# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Arrival-label sanity guard (SA-GATE-TOOLS, D-081 item 5; SA-EMA-DATA item 3).

Other flights' ARR taxi-in (``TAXITIME_SEC_mvt`` of ARR rows, or in-block minus landing) is a legal
feature input (D-005), but it carries recording failures: negative values, 11–14 h 2026 rows at
LIRF/LFPG, a 7-day LSZH 2025 row.  The guard treats an arrival whose taxi-in is **< 0 s or > 3 h
(10,800 s)** as missing.  3 h keeps EHAM's genuine 1–3 h snow taxi-ins (January 2026).

Two modes:

* ``"value"``: only the taxi-in value becomes missing (the arrival still counts as an event);
* ``"row"`` (default for new blocks): the arrival's taxi-in **and** in-block time become missing, so the
  arrival drops out of every in-block-ordered timeline and every taxi-in window, as if its in-block
  record did not exist (its landing clock is untouched).

The guard is opt-in.  Deployed blocks (native state, M1 ``m1_stand_v1s``, M2 ``m2_follow_v1s``) are
**not** changed by it; new blocks enable it through the bake config (``arrival_guard``) or the
builders' ``--arr-guard`` option.  Copy-rate features that are meant to see BLOCK == SCHED rows must not
use it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

TAXI_IN_MIN_SEC = 0.0
TAXI_IN_MAX_SEC = 10_800.0
MODES = ("row", "value")


def plausible(taxi_in: np.ndarray | pl.Series | list, lo: float = TAXI_IN_MIN_SEC, hi: float = TAXI_IN_MAX_SEC) -> np.ndarray:
    """True where the taxi-in is within [lo, hi]; missing values are *not* implausible (True)."""

    x = np.asarray(pl.Series(taxi_in).cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy() if not isinstance(taxi_in, np.ndarray) else taxi_in, dtype=np.float64)
    return np.isnan(x) | ((x >= lo) & (x <= hi))


def guard_values(taxi_in: np.ndarray, lo: float = TAXI_IN_MIN_SEC, hi: float = TAXI_IN_MAX_SEC) -> np.ndarray:
    """Taxi-in values with implausible entries set to NaN."""

    x = np.asarray(taxi_in, dtype=np.float64).copy()
    x[~plausible(x, lo, hi)] = np.nan
    return x


def implausible_expr(col: str, lo: float = TAXI_IN_MIN_SEC, hi: float = TAXI_IN_MAX_SEC) -> pl.Expr:
    """Polars: True for a non-null taxi-in outside [lo, hi]."""

    c = pl.col(col).cast(pl.Float64)
    return c.is_not_null() & ~c.is_between(lo, hi, closed="both")


def guard_frame(frame: pl.DataFrame, taxi_col: str, *, inblock_col: str | None = None, mode: str = "row", phase_col: str | None = None,
                arr_value: str = "ARR", lo: float = TAXI_IN_MIN_SEC, hi: float = TAXI_IN_MAX_SEC, flag_col: str | None = None) -> pl.DataFrame:
    """Null the implausible arrivals' taxi-in (and in-block in ``"row"`` mode).

    ``phase_col`` restricts the guard to rows whose phase equals ``arr_value`` (departure rows are never
    touched).  ``flag_col`` adds a boolean column marking the guarded rows (for counts / transparency).
    """

    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    bad = implausible_expr(taxi_col, lo, hi)
    if phase_col is not None:
        bad = bad & (pl.col(phase_col).cast(pl.String).str.to_uppercase() == arr_value)
    out = frame.with_columns(bad.fill_null(False).alias("__arr_guard"))
    cols = [taxi_col] + ([inblock_col] if (inblock_col and mode == "row") else [])
    out = out.with_columns([pl.when(pl.col("__arr_guard")).then(None).otherwise(pl.col(c)).alias(c) for c in cols])
    if flag_col:
        out = out.rename({"__arr_guard": flag_col})
    else:
        out = out.drop("__arr_guard")
    return out


def guard_movements(mov: pl.DataFrame, mode: str = "row") -> pl.DataFrame:
    """The guard on the masked movement table of the M1/M2 builders (``matched_context.movements``)."""

    return guard_frame(mov, "arr_taxiin", inblock_col="arr_inblock", mode=mode, phase_col="phase")


def counts(taxi_in: np.ndarray | pl.Series, lo: float = TAXI_IN_MIN_SEC, hi: float = TAXI_IN_MAX_SEC) -> dict[str, Any]:
    x = np.asarray(pl.Series(taxi_in).cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy(), dtype=np.float64)
    return {"n": int(x.size), "n_non_null": int((~np.isnan(x)).sum()), "negative": int((x < lo).sum()), "above_max": int((x > hi).sum()),
            "implausible": int((~plausible(x, lo, hi)).sum())}
