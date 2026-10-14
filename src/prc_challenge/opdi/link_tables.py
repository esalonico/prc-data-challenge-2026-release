# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Extended D1a linkage summary tables (no taxi labels)."""

from __future__ import annotations

from typing import Any

import polars as pl


def airport_tier_confirmation_table(
    challenge: pl.DataFrame,
    l1: pl.DataFrame,
    old_tier: pl.DataFrame,
) -> list[dict[str, Any]]:
    merged = (
        challenge.select(["MVT_ID_mvt", "adep"])
        .join(l1.select(["MVT_ID_mvt", "l1_confidence"]), on="MVT_ID_mvt", how="left")
        .join(old_tier.select(["MVT_ID_mvt", "confidence_tier"]), on="MVT_ID_mvt", how="left")
    )
    rows: list[dict[str, Any]] = []
    focus = {"LFPG", "LIRF", "EDDF", "EDDM", "EGLL", "LEBL", "EHAM"}
    for airport, grp in merged.group_by("adep"):
        ap = airport[0] if isinstance(airport, tuple) else airport
        out: dict[str, Any] = {"airport": ap, "n": grp.height}
        for tier in ("A", "B"):
            tier_rows = grp.filter(pl.col("confidence_tier") == tier)
            if tier_rows.height == 0:
                out[f"tier_{tier.lower()}_n"] = 0
                out[f"tier_{tier.lower()}_l1_confirmation_rate"] = None
                continue
            confirmed = tier_rows.filter(pl.col("l1_confidence").is_in(["L1A", "L1B"]))
            out[f"tier_{tier.lower()}_n"] = tier_rows.height
            out[f"tier_{tier.lower()}_l1_confirmation_rate"] = float(confirmed.height) / tier_rows.height
        rows.append(out)
    rows.sort(key=lambda r: (r["airport"] not in focus, -r["n"]))
    return rows


def airport_delta_quantile_table(l1: pl.DataFrame) -> list[dict[str, Any]]:
    confirmed = l1.filter(
        pl.col("l1_confidence").is_in(["L1A", "L1B"]) & pl.col("event_takeoff_delta_sec").is_not_null()
    )
    rows: list[dict[str, Any]] = []
    for airport, grp in confirmed.group_by("adep"):
        ap = airport[0] if isinstance(airport, tuple) else airport
        deltas = grp["event_takeoff_delta_sec"]
        if deltas.len() == 0:
            continue
        quantiles = {f"p{int(q * 100):02d}": float(deltas.quantile(q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)}
        rows.append({"airport": ap, "n_confirmed": grp.height, **quantiles})
    rows.sort(key=lambda r: -r["n_confirmed"])
    return rows


def l2_multiplicity_table(l2: pl.DataFrame) -> dict[str, Any]:
    if l2.is_empty():
        return {}
    out: dict[str, Any] = {"n_l1_confirmed": l2.height}
    for col in ("parking_exit_count",):
        if col not in l2.columns:
            continue
        counts = l2.group_by(col).len().sort(col).to_dicts()
        out[f"{col}_distribution"] = counts
        out[f"{col}_p50"] = float(l2[col].median())
        out[f"{col}_p95"] = float(l2[col].quantile(0.95))
    if "has_entry_runway" in l2.columns:
        runway_events = l2.filter(pl.col("has_entry_runway"))
        if runway_events.height:
            out["entry_runway_present_rate"] = float(runway_events.height) / l2.height
    if "has_taxiway_events" in l2.columns:
        taxi = l2.filter(pl.col("has_taxiway_events"))
        if taxi.height:
            out["taxiway_present_rate"] = float(taxi.height) / l2.height
    if "parking_selection_reason" in l2.columns:
        out["parking_selection_reason_counts"] = (
            l2.group_by("parking_selection_reason").len().sort("len", descending=True).to_dicts()
        )
    return out
