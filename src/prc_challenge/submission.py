# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Validate a submitting.parquet-shaped prediction file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from prc_challenge.io import submitting_path


def validate_submission(
    pred: pl.DataFrame,
    template: pl.DataFrame | None = None,
    raw_dir: Path | str = "data/raw",
) -> list[str]:
    errors: list[str] = []
    if template is None:
        template = pl.read_parquet(submitting_path(raw_dir))
    required = ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    if pred.columns != required and set(pred.columns) != set(required):
        errors.append(f"columns {pred.columns} != {required}")
    if pred.height != template.height:
        errors.append(f"row count {pred.height} != template {template.height}")
    if pred["MVT_ID_mvt"].n_unique() != pred.height:
        errors.append("duplicate MVT_ID_mvt")
    extra = template.select("MVT_ID_mvt").join(pred.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti").height
    missing = pred.select("MVT_ID_mvt").join(template.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti").height
    if extra or missing:
        errors.append(f"id mismatch extra={extra} missing={missing}")
    y = pred["TAXITIME_SEC_mvt"].to_numpy()
    if pred["TAXITIME_SEC_mvt"].null_count():
        errors.append("null predictions")
    if not np.isfinite(np.asarray(y, dtype=np.float64)).all():
        errors.append("non-finite predictions")
    return errors
