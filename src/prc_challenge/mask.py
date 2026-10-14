# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Mask ranking-unavailable departure labels in every source row."""

from __future__ import annotations

import polars as pl

from prc_challenge.io import FORBIDDEN_DEPARTURE_LABEL_COLS

QUARANTINED_DEPARTURE_COLS = (
    "AOBT_3_flt",
)

# Conservative default: never feed these to a model as features.
MODEL_FORBIDDEN_COLS = FORBIDDEN_DEPARTURE_LABEL_COLS


def mask_departure_labels(
    df: pl.DataFrame | pl.LazyFrame,
    extra_cols: tuple[str, ...] = (),
) -> pl.DataFrame | pl.LazyFrame:
    """Blank forbidden departure fields, including neighboring DEP rows.

    Arrival rows keep taxi-in / in-block observations. Labels must be stored
    separately before calling this on a modeling frame.
    """
    cols = tuple(dict.fromkeys((*FORBIDDEN_DEPARTURE_LABEL_COLS, *extra_cols)))
    present = [c for c in cols if c in df.collect_schema()]
    if not present:
        return df
    expressions = []
    is_dep = pl.col("PHASE_mvt") == "DEP"
    for col in present:
        expressions.append(
            pl.when(is_dep).then(pl.lit(None).cast(df.collect_schema()[col])).otherwise(pl.col(col)).alias(col)
        )
    return df.with_columns(expressions)


def split_labels(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (label table, masked movement table) for departure scoring."""
    labels = df.filter(pl.col("PHASE_mvt") == "DEP").select(
        ["MVT_ID_mvt", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"]
    )
    masked = mask_departure_labels(df, extra_cols=QUARANTINED_DEPARTURE_COLS)
    return labels, masked
