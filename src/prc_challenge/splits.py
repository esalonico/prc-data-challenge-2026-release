# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Locked validation protocol v1. Fold membership is derived from takeoff UTC month."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml

PROTOCOL_VERSION = "v1"

VIEWS = {
    "V1": {
        "train_months": [2, 3, 4, 5, 6, 8, 9, 10],
        "val_months": [1, 7],
        "role": "primary_development",
    },
    "V2": {
        "train_months": [1, 2, 3, 4, 5, 6],
        "val_months": [7],
        "role": "forward_summer",
    },
    "V3": {
        "train_months": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "val_months": [10],
        "role": "forward_autumn",
    },
    "V4": {
        "train_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "val_months": [11, 12],
        "role": "locked_final",
        "locked": True,
    },
}

LOCKED_VIEW = "V4"


@dataclass(frozen=True)
class SplitView:
    name: str
    train_months: tuple[int, ...]
    val_months: tuple[int, ...]
    role: str
    locked: bool = False


def load_protocol(path: Path | str = "configs/validation.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def views_from_config(cfg: dict | None = None) -> dict[str, SplitView]:
    raw = (cfg or VIEWS)
    views_raw = raw.get("views", raw)
    out = {}
    for name, spec in views_raw.items():
        out[name] = SplitView(
            name=name,
            train_months=tuple(spec["train_months"]),
            val_months=tuple(spec["val_months"]),
            role=spec.get("role", ""),
            locked=bool(spec.get("locked", False)),
        )
    return out


def month_col(frame: pl.DataFrame | pl.LazyFrame, time_col: str = "MVT_TIME_UTC_mvt") -> pl.Expr:
    return pl.col(time_col).dt.month().alias("month")


def assign_split(df: pl.DataFrame, view: SplitView, time_col: str = "MVT_TIME_UTC_mvt") -> pl.DataFrame:
    months = df.select(month_col(df, time_col))["month"]
    role = (
        pl.when(months.is_in(list(view.train_months)))
        .then(pl.lit("train"))
        .when(months.is_in(list(view.val_months)))
        .then(pl.lit("val"))
        .otherwise(pl.lit("unused"))
    )
    return df.with_columns(role.alias("split_role"), months.alias("month"))


def filter_departures(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("PHASE_mvt") == "DEP")


def persist_split_ids(
    dep: pl.DataFrame,
    dest: Path | str,
    time_col: str = "MVT_TIME_UTC_mvt",
) -> Path:
    """Write MVT_ID plus month/view membership. Does not store target values."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    base = dep.select(
        [
            pl.col("MVT_ID_mvt"),
            month_col(dep, time_col),
            pl.col(time_col).dt.year().alias("year"),
        ]
    )
    for name, view in views_from_config().items():
        assigned = assign_split(dep, view, time_col)
        base = base.join(
            assigned.select(["MVT_ID_mvt", pl.col("split_role").alias(name)]),
            on="MVT_ID_mvt",
            how="left",
        )
    base.write_parquet(dest_path)
    return dest_path
