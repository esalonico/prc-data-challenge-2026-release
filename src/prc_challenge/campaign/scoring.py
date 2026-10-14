# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Composed-pipeline scoring, BASE comparison, leakage assertions."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from prc_challenge.campaign.runtime import FORBIDDEN_FEATURE_COLS, LOCKED_MONTHS
from prc_challenge.features import (
    apply_leftover_parent_freeze,
    apply_lirf_stuck_block_gates,
    apply_unmatched_parent_freeze,
    tail_stratum_expr,
)
from prc_challenge.metrics import rmse, score_frame

PARENT_RMSE = {"V1": 363.35347729680757, "V2": 350.3324180670465, "V3": 254.13154149737042}
RMSE_LADDER = (300, 275, 250, 245, 200, 165)


def assert_no_forbidden_features(columns: list[str]) -> None:
    hit = FORBIDDEN_FEATURE_COLS.intersection(columns)
    if hit:
        raise RuntimeError(f"forbidden/quarantined columns used as features: {sorted(hit)}")


def assert_no_locked_months(df: pl.DataFrame, month_col: str = "month") -> None:
    if month_col not in df.columns:
        return
    bad = df.filter(pl.col(month_col).is_in(list(LOCKED_MONTHS)))
    if bad.height:
        raise RuntimeError(f"locked V4 months present: {bad[month_col].unique().to_list()}")


def assert_prediction_frame(pred: pl.DataFrame, expected_ids: pl.Series | None = None) -> None:
    if "MVT_ID_mvt" not in pred.columns or "prediction" not in pred.columns:
        raise RuntimeError("predictions must include MVT_ID_mvt and prediction")
    if pred["MVT_ID_mvt"].null_count():
        raise RuntimeError("null MVT_ID_mvt in predictions")
    n_unique = pred["MVT_ID_mvt"].n_unique()
    if n_unique != pred.height:
        raise RuntimeError(f"duplicate prediction rows: n={pred.height} unique={n_unique}")
    if expected_ids is not None:
        want = pl.DataFrame({"MVT_ID_mvt": expected_ids})
        extra = pred.select("MVT_ID_mvt").join(want, on="MVT_ID_mvt", how="anti")
        missing = want.join(pred.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti")
        if extra.height or missing.height:
            raise RuntimeError(
                f"prediction ID mismatch extra={extra.height} missing={missing.height}"
            )


def assert_protected_equal(
    scored: pl.DataFrame,
    parent_col: str = "parent_prediction",
    pred_col: str = "prediction",
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Unmatched rows must keep the incumbent freeze."""
    if "nm_unmatched" not in scored.columns:
        return {"protected_ok": False, "reason": "nm_unmatched missing"}
    prot = scored.filter(pl.col("nm_unmatched") == 1)
    if prot.height == 0:
        return {"protected_ok": True, "n": 0, "max_abs_diff": 0.0}
    diff = (prot[pred_col] - prot[parent_col]).abs()
    max_abs = float(diff.max())
    n_bad = int((diff > atol).sum())
    if n_bad:
        raise RuntimeError(f"protected unmatched rows changed: n_bad={n_bad} max_abs={max_abs}")
    return {"protected_ok": True, "n": prot.height, "max_abs_diff": max_abs}


def compose_with_parent(df: pl.DataFrame) -> pl.DataFrame:
    """E17/E18 then unmatched freeze then leftover freeze then clip ≥ 0. Parent must already be joined."""
    out = apply_lirf_stuck_block_gates(df)
    out = apply_unmatched_parent_freeze(out)
    leftover_cols = {"AIRPORT", "nm_unmatched", "type_missing", "takeoff_minus_sched_days", "takeoff_minus_sched_sec"}
    if leftover_cols.issubset(set(out.columns)):
        out = apply_leftover_parent_freeze(out)
    return out.with_columns(pl.col("prediction").clip(lower_bound=0))


COMPOSE_RULE_ORDER = (
    "e17_lirf_delay_gate",
    "e18_lirf_extraday_gate",
    "unmatched_parent_freeze",
    "leftover_parent_freeze",
    "clip_nonnegative",
)


def explain_compose_with_parent(
    df: pl.DataFrame,
    *,
    proposed_col: str = "proposed_prediction",
    parent_col: str = "parent_prediction",
    atol: float = 1e-6,
) -> pl.DataFrame:
    """Record the last composition rule that replaced the proposed prediction.

    The scored experiment may restore unmatched and leftover rows to the parent.
    Those replacements are policy, not evidence that a candidate failed there.
    """
    from prc_challenge.features import (
        apply_lirf_delay_as_taxi,
        apply_lirf_extraday_as_taxi,
        apply_leftover_parent_freeze,
        apply_unmatched_parent_freeze,
    )

    if proposed_col not in df.columns:
        raise ValueError(f"missing proposed column {proposed_col}")
    if parent_col not in df.columns:
        raise ValueError(f"missing parent column {parent_col}")
    working = df.with_columns(pl.col(proposed_col).alias("prediction"))
    rule = np.full(working.height, "kept_proposed", dtype=object)
    previous = working["prediction"].to_numpy().astype(np.float64, copy=True)

    def _mark(name: str, updated: pl.DataFrame) -> pl.DataFrame:
        nonlocal previous
        current = updated["prediction"].to_numpy().astype(np.float64)
        changed = np.abs(current - previous) > atol
        rule[changed] = name
        previous = current
        return updated

    working = _mark("e17_lirf_delay_gate", apply_lirf_delay_as_taxi(working))
    working = _mark("e18_lirf_extraday_gate", apply_lirf_extraday_as_taxi(working))
    working = _mark("unmatched_parent_freeze", apply_unmatched_parent_freeze(working, parent_col=parent_col))
    leftover_cols = {"AIRPORT", "nm_unmatched", "type_missing", "takeoff_minus_sched_days", "takeoff_minus_sched_sec"}
    if leftover_cols.issubset(set(working.columns)):
        working = _mark(
            "leftover_parent_freeze",
            apply_leftover_parent_freeze(working, parent_col=parent_col),
        )
    clipped = working.with_columns(pl.col("prediction").clip(lower_bound=0))
    working = _mark("clip_nonnegative", clipped)
    return working.with_columns(
        pl.col(proposed_col).alias("proposed_prediction"),
        pl.Series("compose_rule", rule.tolist()),
        pl.Series(
            "replaced_by_compose",
            (np.abs(working["prediction"].to_numpy() - working[proposed_col].to_numpy()) > atol),
        ),
    )


def required_sse_reductions(*, sse: float, n: int, current_rmse: float | None = None) -> dict[str, Any]:
    """Handoff Q12: SSE that must disappear to hit RMSE 300/275/250/245/200/165."""
    if n <= 0:
        raise ValueError("n must be positive")
    rmse_now = current_rmse if current_rmse is not None else float((sse / n) ** 0.5)
    targets = []
    for t in RMSE_LADDER:
        need = float(n) * (t**2)
        reduction = sse - need
        targets.append(
            {
                "rmse": t,
                "sse_at_target": need,
                "delta_sse_required": reduction,
                "frac_sse_reduction": (reduction / sse) if sse else None,
                "already_there": reduction <= 0,
            }
        )
    return {"n": int(n), "current_sse": sse, "current_rmse": rmse_now, "targets": targets}


def slice_table(scored: pl.DataFrame) -> dict[str, Any]:
    y = "TAXITIME_SEC_mvt"
    p = "prediction"
    out: dict[str, Any] = {"overall": score_frame(scored)}
    ordinary = scored.filter((pl.col("nm_unmatched") == 0) & (pl.col(y) < 7200))
    long_m = scored.filter((pl.col("nm_unmatched") == 0) & (pl.col(y) >= 7200))
    out["ordinary"] = score_frame(ordinary) if ordinary.height else {}
    out["matched_long"] = score_frame(long_m) if long_m.height else {}
    airports = []
    for keys, part in scored.group_by("AIRPORT", maintain_order=True):
        ap = keys[0] if isinstance(keys, tuple) else keys
        yy = part[y].to_numpy()
        pp = part[p].to_numpy()
        sse = float(np.sum((yy - pp) ** 2))
        airports.append(
            {
                "AIRPORT": ap,
                "n": part.height,
                "rmse_seconds": rmse(yy, pp),
                "sse": sse,
            }
        )
    out["airports"] = sorted(airports, key=lambda r: -r["sse"])
    if "nm_unmatched" in scored.columns:
        for flag, name in ((0, "matched"), (1, "unmatched")):
            part = scored.filter(pl.col("nm_unmatched") == flag)
            out[name] = score_frame(part) if part.height else {}
    miss_flags: list[tuple[pl.Expr, str, str]] = []
    if "type_missing" in scored.columns:
        miss_flags.append((pl.col("type_missing") == 1, "type_missing", "type_present"))
    elif "AIRCRAFT_TYPE_mvt" in scored.columns:
        miss_flags.append((pl.col("AIRCRAFT_TYPE_mvt").is_null(), "type_missing", "type_present"))
    if "RUNWAY_mvt" in scored.columns:
        miss_flags.append((pl.col("RUNWAY_mvt").is_null(), "runway_missing", "runway_present"))
    if "STAND_mvt" in scored.columns:
        miss_flags.append((pl.col("STAND_mvt").is_null(), "stand_missing", "stand_present"))
    for expr, missing_name, present_name in miss_flags:
        flagged = scored.with_columns(expr.cast(pl.Boolean).alias("_miss"))
        for flag, name in ((True, missing_name), (False, present_name)):
            part = flagged.filter(pl.col("_miss") == flag)
            if part.height:
                out[name] = score_frame(part)
    se = (scored[y] - scored[p]) ** 2
    ranked = scored.with_columns(se.alias("se")).sort("se", descending=True)
    total_sse = float(se.sum())
    out["sse"] = total_sse
    for k in (100, 500, 1000):
        top = ranked.head(min(k, ranked.height))
        share = float(top["se"].sum() / total_sse) if total_sse else 0.0
        out[f"top{k}_sse_share"] = share
        out[f"top{k}_sse"] = float(top["se"].sum())
    if "month" in scored.columns:
        months = []
        for keys, part in scored.group_by("month", maintain_order=True):
            m = keys[0] if isinstance(keys, tuple) else keys
            months.append({"month": int(m), **score_frame(part), "sse": float(((part[y] - part[p]) ** 2).sum())})
        out["months"] = months
    if "type_missing" in scored.columns:
        for flag, name in ((0, "type_present"), (1, "type_missing")):
            part = scored.filter(pl.col("type_missing") == flag)
            out[name] = score_frame(part) if part.height else {}
    cell_cols = {
        "AIRPORT",
        "nm_unmatched",
        "TAXITIME_SEC_mvt",
        "type_missing",
        "takeoff_minus_sched_days",
        "takeoff_minus_sched_sec",
    }
    if cell_cols.issubset(set(scored.columns)):
        labeled = scored.with_columns(tail_stratum_expr().alias("_sse_cell"))
        cells = []
        for keys, part in labeled.group_by("_sse_cell", maintain_order=True):
            name = keys[0] if isinstance(keys, tuple) else keys
            yy = part[y].to_numpy()
            pp = part[p].to_numpy()
            sse_c = float(np.sum((yy - pp) ** 2))
            cells.append(
                {
                    "cell": name,
                    "n": part.height,
                    "rmse_seconds": rmse(yy, pp),
                    "sse": sse_c,
                    "sse_share": (sse_c / total_sse) if total_sse else 0.0,
                }
            )
        out["sse_cells"] = sorted(cells, key=lambda r: -r["sse"])
        lfpg = scored.filter(
            (pl.col("AIRPORT") == "LFPG")
            & (pl.col("nm_unmatched") == 1)
            & (pl.col("takeoff_minus_sched_days") == 0)
            & (pl.col(y) >= 7200)
        )
        if lfpg.height:
            yy = lfpg[y].to_numpy()
            pp = lfpg[p].to_numpy()
            sse_c = float(np.sum((yy - pp) ** 2))
            out["lfpg_unmatched_extraday0_y_ge_2h"] = {
                "n": lfpg.height,
                "rmse_seconds": rmse(yy, pp),
                "sse": sse_c,
                "sse_share": (sse_c / total_sse) if total_sse else 0.0,
            }
    return out


def compare_to_base(
    scored: pl.DataFrame,
    base: pl.DataFrame,
    id_col: str = "MVT_ID_mvt",
) -> dict[str, Any]:
    joined = scored.join(
        base.select([id_col, pl.col("prediction").alias("base_prediction")]),
        on=id_col,
        how="inner",
    )
    if joined.height != scored.height:
        raise RuntimeError(f"BASE join dropped rows: {scored.height} -> {joined.height}")
    y = joined["TAXITIME_SEC_mvt"].to_numpy()
    p = joined["prediction"].to_numpy()
    b = joined["base_prediction"].to_numpy()
    sse = float(np.sum((y - p) ** 2))
    sse_b = float(np.sum((y - b) ** 2))
    out: dict[str, Any] = {
        "n": joined.height,
        "rmse_seconds": rmse(y, p),
        "sse": sse,
        "base_rmse_seconds": rmse(y, b),
        "base_sse": sse_b,
        "delta_rmse_vs_base": rmse(y, p) - rmse(y, b),
        "delta_sse_vs_base": sse_b - sse,
        "pred_corr_vs_base": float(np.corrcoef(p, b)[0, 1]) if joined.height > 1 else None,
        "resid_corr_vs_base": float(np.corrcoef(y - p, y - b)[0, 1]) if joined.height > 1 else None,
        "bias_seconds": float(np.mean(p - y)),
    }
    if "AIRPORT" in joined.columns and "MVT_TIME_UTC_mvt" in joined.columns:
        day = joined.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("_day"))
        day = day.with_columns(
            ((pl.col("TAXITIME_SEC_mvt") - pl.col("prediction")) ** 2).alias("_se"),
            ((pl.col("TAXITIME_SEC_mvt") - pl.col("base_prediction")) ** 2).alias("_se_b"),
        )
        agg = day.group_by(["AIRPORT", "_day"]).agg(
            pl.col("_se").sum().alias("sse"),
            pl.col("_se_b").sum().alias("sse_base"),
            pl.len().alias("n"),
        ).with_columns((pl.col("sse_base") - pl.col("sse")).alias("delta_sse"))
        out["airport_day_delta_sse_p50"] = float(agg["delta_sse"].median() or 0.0)
        out["airport_day_share_positive"] = float((agg["delta_sse"] > 0).mean())
        out["n_airport_days"] = agg.height
        week = joined.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1w").alias("_week"))
        week = week.with_columns(
            ((pl.col("TAXITIME_SEC_mvt") - pl.col("prediction")) ** 2).alias("_se"),
            ((pl.col("TAXITIME_SEC_mvt") - pl.col("base_prediction")) ** 2).alias("_se_b"),
        )
        wagg = week.group_by(["AIRPORT", "_week"]).agg(
            pl.col("_se").sum().alias("sse"),
            pl.col("_se_b").sum().alias("sse_base"),
        ).with_columns((pl.col("sse_base") - pl.col("sse")).alias("delta_sse"))
        out["airport_week_delta_sse_p50"] = float(wagg["delta_sse"].median() or 0.0)
        out["airport_week_share_positive"] = float((wagg["delta_sse"] > 0).mean())
        out["n_airport_weeks"] = wagg.height
    return out


def oof_is_held_out(train_ids: pl.Series, oof_ids: pl.Series) -> None:
    """OOF rows must be a subset of train IDs (same population), produced without using that row's label in its own fold.

    Callers must also pass a fold column. This only checks ID coverage.
    """
    extra = oof_ids.to_frame("id").join(train_ids.to_frame("id"), on="id", how="anti")
    if extra.height:
        raise RuntimeError(f"OOF contains IDs not in train: {extra.height}")
    missing = train_ids.to_frame("id").join(oof_ids.to_frame("id"), on="id", how="anti")
    if missing.height:
        raise RuntimeError(f"OOF missing train IDs: {missing.height}")


def score_and_write(
    val: pl.DataFrame,
    pred: np.ndarray,
    dest,
    *,
    view: str,
    parent: pl.DataFrame,
    base: pl.DataFrame | None = None,
    write_predictions: bool = True,
    tag: str | None = None,
) -> dict[str, Any]:
    """Compose with frozen E17/E18 + unmatched parent, assert IDs/protected rows, persist artifacts."""
    from pathlib import Path

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    scored = val.with_columns(pl.Series("prediction", np.asarray(pred, dtype=np.float64)).clip(lower_bound=0))
    scored = scored.join(
        parent.select(["MVT_ID_mvt", pl.col("prediction").alias("parent_prediction")]),
        on="MVT_ID_mvt",
        how="left",
    )
    scored = compose_with_parent(scored)
    assert_prediction_frame(scored.select(["MVT_ID_mvt", "prediction"]), val["MVT_ID_mvt"])
    protected = assert_protected_equal(scored)
    slices = slice_table(scored)
    vs_base = compare_to_base(scored, base) if base is not None else {"status": "base_missing"}
    stem = f"predictions_{tag}_{view}" if tag else f"predictions_{view}"
    keep = [c for c in ("MVT_ID_mvt", "AIRPORT", "TAXITIME_SEC_mvt", "prediction", "nm_unmatched", "month") if c in scored.columns]
    if write_predictions:
        scored.select(keep).write_parquet(dest / f"{stem}.parquet")
        scored.select(keep).write_parquet(dest / f"predictions_{view}.parquet")
        if view == "V1" or not (dest / "predictions.parquet").exists():
            scored.select(keep).write_parquet(dest / "predictions.parquet")
        val.select("MVT_ID_mvt").write_parquet(dest / f"val_ids_{view}.parquet")
        if "month" in val.columns:
            val.select(["MVT_ID_mvt", "month"]).write_parquet(dest / f"val_ids_{view}.parquet")
        airports = pl.DataFrame(slices.get("airports") or [])
        if airports.height:
            airports.write_parquet(dest / f"error_slices_{view}.parquet")
            if view == "V1" or not (dest / "error_slices.parquet").exists():
                airports.write_parquet(dest / "error_slices.parquet")
        se = scored.with_columns(((pl.col("TAXITIME_SEC_mvt") - pl.col("prediction")) ** 2).alias("se"))
        top = se.sort("se", descending=True).head(1000).select(keep + ["se"])
        top.write_parquet(dest / f"top1000_sse_{view}.parquet")
        if view == "V1" or not (dest / "top1000_sse.parquet").exists():
            top.write_parquet(dest / "top1000_sse.parquet")
    overall = slices["overall"]
    return {
        "view": view,
        "model": "composed",
        "train_population": "matched_booster_train",
        "validation_population": "official_val_all_rows",
        "row_count": overall.get("n"),
        "rmse_seconds": overall.get("rmse_seconds"),
        "sse": slices["sse"],
        "bias_seconds": overall.get("bias_seconds"),
        "ordinary": slices.get("ordinary"),
        "matched_long": slices.get("matched_long"),
        "type_missing": slices.get("type_missing"),
        "type_present": slices.get("type_present"),
        "runway_missing": slices.get("runway_missing"),
        "runway_present": slices.get("runway_present"),
        "stand_missing": slices.get("stand_missing"),
        "stand_present": slices.get("stand_present"),
        "airports": slices.get("airports"),
        "months": slices.get("months"),
        "top100_sse_share": slices.get("top100_sse_share"),
        "top500_sse_share": slices.get("top500_sse_share"),
        "top1000_sse_share": slices.get("top1000_sse_share"),
        "sse_cells": slices.get("sse_cells"),
        "lfpg_unmatched_extraday0_y_ge_2h": slices.get("lfpg_unmatched_extraday0_y_ge_2h"),
        "protected": protected,
        "vs_base": vs_base,
        "delta_sse_vs_base": vs_base.get("delta_sse_vs_base") if isinstance(vs_base, dict) else None,
        "delta_rmse_vs_e33b": (overall.get("rmse_seconds") or 0.0) - PARENT_RMSE.get(view, 0.0),
    }


COMPACT_SCORE_KEYS = (
    "rmse_seconds",
    "sse",
    "vs_base",
    "ordinary",
    "matched_long",
    "row_count",
    "bias_seconds",
    "delta_rmse_vs_e33b",
    "delta_sse_vs_base",
    "protected",
    "train_population",
    "validation_population",
    "top100_sse_share",
    "top500_sse_share",
    "top1000_sse_share",
    "sse_cells",
    "months",
    "airports_top",
)


def compact_score(scored: dict[str, Any]) -> dict[str, Any]:
    """Keep ingest/review fields without dumping the full per-airport slice table."""
    out = {k: scored.get(k) for k in COMPACT_SCORE_KEYS}
    if out.get("airports_top"):
        return out
    airports = scored.get("airports") or []
    if isinstance(airports, list) and airports:
        ranked = sorted(
            [r for r in airports if isinstance(r, dict)],
            key=lambda r: -float(r.get("sse") or 0),
        )[:10]
        out["airports_top"] = [
            {k: r.get(k) for k in ("AIRPORT", "n", "rmse_seconds", "sse") if k in r} for r in ranked
        ]
    return out


def promote_predictions(dest, view: str, tag: str) -> None:
    """Copy a tagged prediction file to the untagged names CPU-13 / ensemble expect."""
    from pathlib import Path
    import shutil

    dest = Path(dest)
    src = dest / f"predictions_{tag}_{view}.parquet"
    if not src.exists():
        return
    shutil.copy2(src, dest / f"predictions_{view}.parquet")
    shutil.copy2(src, dest / "predictions.parquet")
