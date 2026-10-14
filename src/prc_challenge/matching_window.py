# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Organizer NM-matching window: a label constraint from dataset construction.

Every NM-matched 2025 departure satisfies ``|BLOCK - LOBT| <= 3606 s``
(0 violations in 2,062,577 rows), and every one lies within 3606 s of IOBT *or*
of EOBT_1 (0 violations). The arrival analogue ``|landing - ARVT_3| <= ~3606``
holds in both 2025 and the 2026 ranking file, so the same matching pipeline is
in force. For a matched departure with takeoff ``MVT`` the taxi label therefore
lies in ``[max(0, MVT - P - W), MVT - P + W]`` for the relevant planning time
``P``. Projecting a prediction onto a convex interval that contains the truth
cannot increase that row's error.

The default ``hull`` mode uses only already-whitelisted planning fields
(IOBT_flt, EOBT_1_flt): the smallest interval covering both windows. No
parameter is fitted. Unmatched rows (no planning times) are untouched.
"""

from __future__ import annotations

import polars as pl

WINDOW_SEC = 3606
FIELD_SETS = {
    "hull": ("IOBT_flt", "EOBT_1_flt"),
    "lobt": ("LOBT_flt",),
}


def window_bounds(frame: pl.DataFrame, mode: str = "hull", window: int = WINDOW_SEC) -> pl.DataFrame:
    """Add ``win_lo``/``win_hi`` (seconds of taxi). Null when any field is missing."""

    fields = FIELD_SETS[mode]
    lows, highs = [], []
    for field in fields:
        lead = (pl.col("MVT_TIME_UTC_mvt") - pl.col(field)).dt.total_seconds()
        lows.append(lead - window)
        highs.append(lead + window)
    lo = pl.min_horizontal(lows) if len(lows) > 1 else lows[0]
    hi = pl.max_horizontal(highs) if len(highs) > 1 else highs[0]
    any_null = pl.any_horizontal([pl.col(f).is_null() for f in fields])
    return frame.with_columns(
        pl.when(any_null).then(None).otherwise(pl.max_horizontal(lo, pl.lit(0))).cast(pl.Float64).alias("win_lo"),
        pl.when(any_null).then(None).otherwise(hi).cast(pl.Float64).alias("win_hi"),
    )


def project_predictions(frame: pl.DataFrame, pred_col: str = "prediction", mode: str = "hull") -> pl.DataFrame:
    """Clip ``pred_col`` into the matching window on matched rows only."""

    out = window_bounds(frame, mode)
    matched = pl.col("FLIGHT_ID_mvt").is_not_null() & pl.col("win_lo").is_not_null()
    return out.with_columns(
        pl.when(matched)
        .then(pl.col(pred_col).clip(pl.col("win_lo"), pl.col("win_hi")))
        .otherwise(pl.col(pred_col))
        .alias(pred_col)
    )


def count_violations(frame: pl.DataFrame, mode: str = "hull") -> int:
    """Training-label check: matched rows whose taxi label falls outside the window."""

    out = window_bounds(frame.filter(pl.col("FLIGHT_ID_mvt").is_not_null()), mode)
    y = pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)
    bad = out.filter(pl.col("win_lo").is_not_null() & ((y < pl.col("win_lo") - 0.5) | (y > pl.col("win_hi") + 0.5)))
    return bad.height
