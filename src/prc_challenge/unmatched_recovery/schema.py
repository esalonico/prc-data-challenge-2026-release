# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Shared schema, masking, freeze-free composition, and population scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from prc_challenge.campaign.scoring import slice_table
from prc_challenge.features import add_flight_prefix, apply_lirf_stuck_block_gates
from prc_challenge.metrics import score_frame

D_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)
PRIMARY_MONTHS = (1, 7)
LOCKED_MONTHS = (11, 12)
BLOCKS: tuple[tuple[int, ...], ...] = ((2, 8), (3, 9), (4, 10), (5, 6))

NM_CATEGORICALS = (
    "AIRCRAFT_OPERATOR_flt",
    "WK_TBL_CAT_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt",
)
NM_PLANNING_RAW = ("IOBT_flt", "EOBT_1_flt")
NM_NUMERIC_DERIVED = (
    "takeoff_minus_iobt_sec",
    "takeoff_minus_eobt_sec",
    "eobt_minus_iobt_sec",
    "sched_minus_iobt_sec",
)
NM_MISSING_FLAGS = ("iobt_missing", "eobt_missing")
LATENT_PLANNING = ("sched_minus_iobt_sec", "eobt_minus_iobt_sec")

MASK_REGIMES = ("nm_only", "lirf_like", "planning_only", "categorical_only")

UNMATCHED_CATS = (
    "AIRPORT",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "flight_prefix",
    "ADES_mvt",
    "FLIGHT_RULE_mvt",
    "skyc1",
)
UNMATCHED_NUMS = (
    "local_hour",
    "local_dow",
    "local_month",
    "takeoff_minus_sched_sec",
    "takeoff_minus_sched_days",
    "takeoff_minus_sched_mod_sec",
    "type_missing",
    "nm_unmatched",
    "tmpc",
    "dwpc",
    "sknt",
    "gust",
    "vsby",
    "p01i",
    "metar_age_sec",
    "metar_has_wx",
    "metar_frozen",
    "metar_fog",
    "metar_ts",
    "metar_precip",
    "metar_low_vis",
    "n_delayed_dep_ctr_900s",
    "n_delayed_dep_ctr_1800s",
    "n_delayed_dep_ctr_3600s",
    "ades_missing",
    "runway_missing",
    "stand_missing",
)

RAW_SIDECAR_COLS = (
    "MVT_ID_mvt",
    "FLIGHT_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "FLIGHT_RULE_mvt",
    "SCHED_TIME_UTC_mvt",
    "IOBT_flt",
    "EOBT_1_flt",
    "PHASE_mvt",
)

S4_PARENT_RMSE = 345.066655
S4_PARENT_N = 344_419
S4_PARENT_SSE = 41_010_313_432.0
PRECISION_GATE = 0.99
MASK_RATES = (0.10, 0.30, 0.50)
DISTILL_LAMBDAS = (0.0, 0.25, 1.0)
REGIME_EDGES = (3600.0, 7200.0)
MI_K = 5


def present(frame: pl.DataFrame, names: tuple[str, ...] | list[str]) -> list[str]:
    return [name for name in names if name in frame.columns]


def unmatched_observed_features(frame: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Features that remain observable on genuine unmatched rows."""

    cats = present(frame, UNMATCHED_CATS)
    nums = present(frame, UNMATCHED_NUMS)
    return cats, nums


def ensure_unmatched_aux(frame: pl.DataFrame) -> pl.DataFrame:
    """Add ranking-safe unmatched auxiliaries. Never invents an NM flight ID."""

    out = frame
    if "flight_prefix" not in out.columns and "FLIGHT_mvt" in out.columns:
        out = add_flight_prefix(out)
    extras: list[pl.Expr] = []
    if "ADES_mvt" in out.columns and "ades_missing" not in out.columns:
        extras.append(pl.col("ADES_mvt").is_null().cast(pl.Int8).alias("ades_missing"))
    if "RUNWAY_mvt" in out.columns and "runway_missing" not in out.columns:
        extras.append(pl.col("RUNWAY_mvt").is_null().cast(pl.Int8).alias("runway_missing"))
    if "STAND_mvt" in out.columns and "stand_missing" not in out.columns:
        extras.append(pl.col("STAND_mvt").is_null().cast(pl.Int8).alias("stand_missing"))
    if extras:
        out = out.with_columns(extras)
    return out


def apply_nm_block_mask(frame: pl.DataFrame, regime: str = "nm_only") -> pl.DataFrame:
    """Hide the NM information block. Missingness flags follow the masked state."""

    if regime not in MASK_REGIMES:
        raise ValueError(f"unknown mask regime {regime}")
    out = frame
    hide_cats = regime in {"nm_only", "lirf_like", "categorical_only"}
    hide_plan = regime in {"nm_only", "lirf_like", "planning_only"}
    hide_type = regime == "lirf_like"
    if hide_cats:
        existing = present(out, NM_CATEGORICALS)
        if existing:
            out = out.with_columns([pl.lit(None).cast(out.schema[c]).alias(c) for c in existing])
    if hide_plan:
        existing_num = present(out, NM_NUMERIC_DERIVED)
        if existing_num:
            out = out.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in existing_num])
        existing_raw = present(out, NM_PLANNING_RAW)
        if existing_raw:
            out = out.with_columns([pl.lit(None).cast(out.schema[c]).alias(c) for c in existing_raw])
        flag_exprs: list[pl.Expr] = []
        if "iobt_missing" in out.columns:
            flag_exprs.append(pl.lit(1).cast(pl.Int8).alias("iobt_missing"))
        if "eobt_missing" in out.columns:
            flag_exprs.append(pl.lit(1).cast(pl.Int8).alias("eobt_missing"))
        if flag_exprs:
            out = out.with_columns(flag_exprs)
    if hide_type and "AIRCRAFT_TYPE_mvt" in out.columns:
        out = out.with_columns(
            pl.lit(None).cast(out.schema["AIRCRAFT_TYPE_mvt"]).alias("AIRCRAFT_TYPE_mvt")
        )
        if "type_missing" in out.columns:
            out = out.with_columns(pl.lit(1).cast(pl.Int8).alias("type_missing"))
    if "original_nm_unmatched" not in out.columns and "nm_unmatched" in out.columns:
        out = out.with_columns(pl.col("nm_unmatched").alias("original_nm_unmatched"))
    indicator_map = {
        "imputed_operator": hide_cats,
        "imputed_wake": hide_cats,
        "imputed_segment": hide_cats,
        "imputed_flight_type": hide_cats,
        "imputed_iobt": hide_plan,
        "imputed_eobt": hide_plan,
        "imputed_aircraft_type": hide_type,
        "masked_nm_block": True,
    }
    out = out.with_columns(
        [pl.lit(int(flag)).cast(pl.Int8).alias(name) for name, flag in indicator_map.items()]
    )
    return out


def derive_planning_from_latents(frame: pl.DataFrame) -> pl.DataFrame:
    """Rebuild consistent IOBT/EOBT offsets from the two latent planning quantities."""

    need = {"takeoff_minus_sched_sec", "sched_minus_iobt_sec", "eobt_minus_iobt_sec"}
    if not need.issubset(frame.columns):
        return frame
    return frame.with_columns(
        (pl.col("takeoff_minus_sched_sec") + pl.col("sched_minus_iobt_sec")).alias(
            "takeoff_minus_iobt_sec"
        ),
        (
            pl.col("takeoff_minus_sched_sec")
            + pl.col("sched_minus_iobt_sec")
            - pl.col("eobt_minus_iobt_sec")
        ).alias("takeoff_minus_eobt_sec"),
        pl.lit(1).cast(pl.Int8).alias("iobt_missing"),
        pl.lit(1).cast(pl.Int8).alias("eobt_missing"),
    )


E17_E18_GATE_COLS = (
    "AIRPORT",
    "nm_unmatched",
    "type_missing",
    "takeoff_minus_sched_days",
    "takeoff_minus_sched_sec",
)


def compose_experimental_unmatched(
    df: pl.DataFrame,
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Keep E17/E18 delay-as-taxi gates. Do **not** restore the unmatched parent freeze.

    Raises ``RuntimeError`` when any E17/E18 gate column is missing. Until 2026-09-26 the
    function silently returned the frame un-gated in that case, which scored three
    campaigns without E17/E18 (SA-RETRO-C, retro audit C section 2 / C1).
    """

    if pred_col not in df.columns:
        raise RuntimeError(f"{pred_col} missing from experimental composition")
    missing = [c for c in E17_E18_GATE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"E17/E18 gate columns missing from experimental composition: {missing}")
    out = apply_lirf_stuck_block_gates(df, pred_col=pred_col)
    return out.with_columns(pl.col(pred_col).clip(lower_bound=0))


def splice_unmatched(
    parent: pl.DataFrame,
    candidate: pl.DataFrame,
    *,
    pred_col: str = "prediction",
    candidate_col: str = "prediction",
    id_col: str = "MVT_ID_mvt",
) -> pl.DataFrame:
    """Matched rows keep the parent; unmatched rows take the candidate. Router is ``nm_unmatched``."""

    if "nm_unmatched" not in parent.columns:
        raise RuntimeError("parent frame must include nm_unmatched")
    cand = candidate.select(id_col, pl.col(candidate_col).alias("_candidate_pred"))
    joined = parent.join(cand, on=id_col, how="left")
    if joined.height != parent.height:
        raise RuntimeError("unmatched splice changed row count")
    missing = joined.filter(pl.col("nm_unmatched") == 1)["_candidate_pred"].null_count()
    if missing:
        raise RuntimeError(f"candidate missing {missing} unmatched rows")
    return joined.with_columns(
        pl.when(pl.col("nm_unmatched") == 1)
        .then(pl.col("_candidate_pred"))
        .otherwise(pl.col(pred_col))
        .alias(pred_col)
    ).drop("_candidate_pred")


def _month_slices(frame: pl.DataFrame, pred_col: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "month" not in frame.columns:
        return out
    for month, name in ((1, "JAN_2025"), (7, "JUL_2025")):
        part = frame.filter(pl.col("month") == month)
        if part.height:
            scored = score_frame(part, pred_col=pred_col)
            sse = float(((part["TAXITIME_SEC_mvt"] - part[pred_col]) ** 2).sum())
            out[name] = {**scored, "sse": sse}
    return out


def score_populations(
    scored: pl.DataFrame,
    *,
    pred_col: str = "prediction",
    parent_col: str | None = "parent_prediction",
) -> dict[str, Any]:
    """Full-population and unmatched-only metrics. Never drops genuine unmatched rows."""

    if scored.height == 0:
        raise RuntimeError("cannot score an empty frame")
    if pred_col != "prediction":
        scored = scored.with_columns(pl.col(pred_col).alias("prediction"))
    if "TAXITIME_SEC_mvt" not in scored.columns:
        raise RuntimeError("TAXITIME_SEC_mvt required for scoring")
    slices = slice_table(scored)
    unmatched = scored.filter(pl.col("nm_unmatched") == 1) if "nm_unmatched" in scored.columns else scored.clear()
    matched = scored.filter(pl.col("nm_unmatched") == 0) if "nm_unmatched" in scored.columns else scored
    y = scored["TAXITIME_SEC_mvt"].to_numpy()
    p = scored["prediction"].to_numpy()
    if not np.isfinite(p).all():
        raise RuntimeError("non-finite experimental predictions")
    sse = float(np.dot(p - y, p - y))
    payload: dict[str, Any] = {
        "n": int(scored.height),
        "sse": sse,
        "rmse_seconds": float(np.sqrt(sse / scored.height)),
        "full": slices.get("overall", score_frame(scored)),
        "unmatched": slices.get("unmatched") or (score_frame(unmatched) if unmatched.height else {}),
        "matched": slices.get("matched") or (score_frame(matched) if matched.height else {}),
        "months": _month_slices(scored, "prediction"),
        "unmatched_months": _month_slices(unmatched, "prediction") if unmatched.height else {},
        "airports": slices.get("airports"),
        "sse_cells": slices.get("sse_cells"),
        "top100_sse_share": slices.get("top100_sse_share"),
        "top500_sse_share": slices.get("top500_sse_share"),
        "n_unmatched": int(unmatched.height),
        "n_matched": int(matched.height),
        "unmatched_rows_dropped": 0,
        "freeze_applied": False,
    }
    if unmatched.height:
        uy = unmatched["TAXITIME_SEC_mvt"].to_numpy()
        up = unmatched["prediction"].to_numpy()
        payload["unmatched_sse"] = float(np.dot(up - uy, up - uy))
        payload["unmatched_rmse_seconds"] = float(np.sqrt(payload["unmatched_sse"] / unmatched.height))
        payload["unmatched_bias_seconds"] = float(np.mean(up - uy))
        ranked = unmatched.with_columns(((pl.col("TAXITIME_SEC_mvt") - pl.col("prediction")) ** 2).alias("se"))
        payload["unmatched_top100_sse"] = float(ranked.sort("se", descending=True).head(min(100, ranked.height))["se"].sum())
        payload["unmatched_top500_sse"] = float(ranked.sort("se", descending=True).head(min(500, ranked.height))["se"].sum())
        if "AIRPORT" in unmatched.columns and "month" in unmatched.columns:
            payload["unmatched_airport_month"] = (
                ranked.group_by("AIRPORT", "month")
                .agg(
                    pl.len().alias("n"),
                    pl.col("se").sum().alias("sse"),
                    pl.col("TAXITIME_SEC_mvt").mean().alias("mean_y"),
                    pl.col("prediction").mean().alias("mean_pred"),
                )
                .with_columns((pl.col("sse") / pl.col("n")).sqrt().alias("rmse_seconds"))
                .sort("sse", descending=True)
                .to_dicts()
            )
        for lo, hi, name in (
            (None, 3600.0, "ordinary_lt_3600"),
            (3600.0, 7200.0, "long_3600_7200"),
            (7200.0, None, "extreme_ge_7200"),
        ):
            part = unmatched
            if lo is not None:
                part = part.filter(pl.col("TAXITIME_SEC_mvt") >= lo)
            if hi is not None:
                part = part.filter(pl.col("TAXITIME_SEC_mvt") < hi)
            if part.height:
                payload[f"unmatched_{name}"] = {
                    **score_frame(part),
                    "sse": float(((part["TAXITIME_SEC_mvt"] - part["prediction"]) ** 2).sum()),
                }
    if parent_col and parent_col in scored.columns:
        py = scored[parent_col].to_numpy()
        parent_sse = float(np.dot(py - y, py - y))
        payload["parent_sse"] = parent_sse
        payload["parent_rmse_seconds"] = float(np.sqrt(parent_sse / scored.height))
        payload["delta_sse_vs_parent"] = parent_sse - sse
        payload["improvement_vs_parent_seconds"] = payload["parent_rmse_seconds"] - payload["rmse_seconds"]
        if unmatched.height:
            u_parent = unmatched[parent_col].to_numpy()
            parent_u_sse = float(np.dot(u_parent - uy, u_parent - uy))
            payload["unmatched_parent_sse"] = parent_u_sse
            payload["unmatched_delta_sse_vs_parent"] = parent_u_sse - payload["unmatched_sse"]
            payload["unmatched_improvement_vs_parent_seconds"] = float(
                np.sqrt(parent_u_sse / unmatched.height) - payload["unmatched_rmse_seconds"]
            )
    return payload


def reconstruction_metrics(
    truth: pl.DataFrame,
    imputed: pl.DataFrame,
    *,
    numeric_cols: tuple[str, ...] = LATENT_PLANNING,
    categorical_cols: tuple[str, ...] = NM_CATEGORICALS,
    id_col: str = "MVT_ID_mvt",
) -> dict[str, Any]:
    """Reconstruction quality on artificially masked matched rows."""

    hat_cols = [id_col] + [c for c in imputed.columns if c.endswith("_hat")]
    slim = imputed.select([c for c in hat_cols if c in imputed.columns])
    joined = truth.join(slim, on=id_col, how="inner")
    out: dict[str, Any] = {"n": joined.height}
    for col in numeric_cols:
        hat = f"{col}_hat" if f"{col}_hat" in joined.columns else col
        if col not in joined.columns or hat not in joined.columns:
            continue
        pair = joined.select(pl.col(col).cast(pl.Float64), pl.col(hat).cast(pl.Float64).alias("hat")).drop_nulls()
        if pair.height == 0:
            continue
        err = pair["hat"].to_numpy() - pair[col].to_numpy()
        abs_err = np.abs(err)
        out[col] = {
            "n": pair.height,
            "mae": float(np.mean(abs_err)),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "bias": float(np.mean(err)),
            "p50_abs": float(np.median(abs_err)),
            "p90_abs": float(np.quantile(abs_err, 0.90)),
        }
    for col in categorical_cols:
        hat = f"{col}_hat" if f"{col}_hat" in joined.columns else col
        if col not in joined.columns or hat not in joined.columns:
            continue
        pair = joined.select(pl.col(col).cast(pl.String), pl.col(hat).cast(pl.String).alias("hat")).drop_nulls()
        if pair.height == 0:
            continue
        acc = float((pair[col] == pair["hat"]).mean())
        out[col] = {"n": pair.height, "accuracy": acc}
    return out
