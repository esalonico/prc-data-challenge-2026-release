# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Arms that train on an already assembled S4 matrix.

``scripts/legs/s4_allyear.py --assemble-only`` writes ``runs/<tag>/train_matrix.parquet``
and ``eval_matrix.parquet`` (ids, label, ``unimpeded_p10``, gate columns, then the model
features) plus ``matrix_identity.json`` with the ``[cats, nums]`` feature lists.  This
module loads such a run, builds the residual target the S4 CatBoost uses
(``TAXITIME_SEC_mvt - unimpeded_p10``), and turns raw predictions into the trainer's outputs:

* **janjul** runs (eval = Jan+Jul 2025): the historical composition (E17 -> E18 -> E33b
  unmatched freeze -> leftover freeze -> clip) and the pooled / JAN / JUL / matched-only card,
  exactly as ``train_s4_allyear.py`` prints them.
* **ranking** runs (eval = the two ranking months): ``compose_ranking_matched`` and the
  trainer's ranking prediction schema (``MVT_ID_mvt, period, AIRPORT, nm_unmatched, gate_e17,
  gate_e18, gate_leftover, owned_by_unmatched_overlay, s4_raw_prediction, prediction``).

Training-month policy (``s4_allyear.validate_run_months``): a janjul run may not hold January,
July, November or December rows; a ranking run may hold all twelve months (November/December as
blind training rows, as the CatBoost all-year fit does).  Nothing here evaluates on
November/December labels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prc_challenge import s4_allyear as s4

E33B_PARENT = Path("artifacts/experiments/E33b-matched-splice_V1/val_predictions.parquet")
PREDICTION_COLUMNS = ("MVT_ID_mvt", "AIRPORT", s4.LABEL_COLUMN, "prediction", "nm_unmatched", "month")
RANKING_COLUMNS = (
    "MVT_ID_mvt",
    "period",
    "AIRPORT",
    "nm_unmatched",
    "gate_e17",
    "gate_e18",
    "gate_leftover",
    "owned_by_unmatched_overlay",
    "s4_raw_prediction",
    "prediction",
)
ID_COLUMNS = (
    "MVT_ID_mvt",
    "period",
    "month",
    "AIRPORT",
    s4.LABEL_COLUMN,
    "nm_unmatched",
    "type_missing",
    "takeoff_minus_sched_days",
    "takeoff_minus_sched_sec",
    "unimpeded_p10",
)
FORBIDDEN_TRAIN_MONTHS = frozenset(s4.PRIMARY_EVAL_MONTHS) | frozenset(s4.LOCKED_EVAL_MONTHS)  # janjul runs
RANKING_PERIOD_KEYS = frozenset(p.key for p in s4.RANKING_PERIODS)


def feature_lists_for_run(run: Path) -> tuple[list[str], list[str]]:
    """``[cats, nums]`` of the assembled matrix, from the identity file the trainer wrote."""

    identity = run / "matrix_identity.json"
    if identity.exists():
        cats, nums = json.loads(identity.read_text())["features"]
        return list(cats), list(nums)
    for name in ("assemble_manifest.json", "run_manifest.json"):
        path = run / name
        if path.exists():
            features = json.loads(path.read_text())["features"]
            return list(features["cats"]), list(features["nums"])
    raise FileNotFoundError(f"{run} has no matrix_identity.json / assemble_manifest.json / run_manifest.json")


def eval_mode_of(evaluation: pl.DataFrame) -> str:
    """``ranking`` when every eval row belongs to a ranking period, else ``janjul``."""

    periods = set(evaluation["period"].unique().to_list())
    return "ranking" if periods and periods <= RANKING_PERIOD_KEYS else "janjul"


def load_run_matrices(run: Path) -> tuple[pl.DataFrame, pl.DataFrame, list[str], list[str], dict[str, Any]]:
    """Train and eval matrices of ``runs/<tag>`` with their feature lists and a provenance dict."""

    cats, nums = feature_lists_for_run(run)
    train_path, eval_path = run / "train_matrix.parquet", run / "eval_matrix.parquet"
    train = pl.read_parquet(train_path)
    evaluation = pl.read_parquet(eval_path)
    missing = [c for c in (*ID_COLUMNS, *cats, *nums) if c not in train.columns or c not in evaluation.columns]
    if missing:
        raise RuntimeError(f"matrices lack columns: {missing[:10]}")
    mode = eval_mode_of(evaluation)
    train_months = sorted(int(m) for m in train["month"].unique().to_list())
    s4.validate_run_months(train_months, mode)
    if train[s4.LABEL_COLUMN].null_count() or train["unimpeded_p10"].null_count():
        raise RuntimeError("training matrix has null labels or null unimpeded_p10")
    locked = sorted(set(train_months) & set(s4.LOCKED_EVAL_MONTHS))
    info = {
        "run_dir": str(run),
        "eval_mode": mode,
        "train_matrix": {"path": str(train_path), "sha256": s4.sha256_file(train_path), "rows": train.height, "columns": train.width},
        "eval_matrix": {"path": str(eval_path), "sha256": s4.sha256_file(eval_path), "rows": evaluation.height, "columns": evaluation.width},
        "train_months": train_months,
        "nov_dec_labels_used": locked,
        "nov_dec_role": "blind final training rows only" if locked else "unused",
        "eval_periods": sorted(evaluation["period"].unique().to_list()),
        "n_cats": len(cats),
        "n_nums": len(nums),
        "n_features": len(cats) + len(nums),
    }
    identity = run / "matrix_identity.json"
    if identity.exists():
        info["matrix_identity_sha256"] = json.loads(identity.read_text()).get("identity_sha256")
    return train, evaluation, cats, nums, info


def residual_target(frame: pl.DataFrame) -> np.ndarray:
    """The S4 residual label: taxi time minus the Q1 unimpeded p10 baseline."""

    y = (frame[s4.LABEL_COLUMN].cast(pl.Float64) - frame["unimpeded_p10"].cast(pl.Float64)).to_numpy()
    if np.isnan(y).any():
        raise RuntimeError("residual label has nulls")
    return y


def raw_from_residual(frame: pl.DataFrame, residual_prediction: np.ndarray) -> np.ndarray:
    """Raw taxi-time prediction: baseline + residual, clipped at zero (as the trainer does)."""

    base = frame["unimpeded_p10"].cast(pl.Float64).to_numpy()
    return np.clip(base + np.asarray(residual_prediction, dtype=np.float64), 0, None)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2))) if y.size else float("nan")


def _smoke_subset() -> bool:
    """Bake smoke runs (``bake.py --smoke``) evaluate a hash sample of the JAN_JUL rows; nothing else sets this."""

    return os.environ.get("PRC_SMOKE_EVAL_SUBSET") == "1"


def score_janjul(evaluation: pl.DataFrame, raw: np.ndarray, src: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Compose raw Jan/Jul predictions with the frozen E33b parent and return (out, scorecard)."""

    parent_path = src / E33B_PARENT
    parent = pl.read_parquet(parent_path)
    scored = s4.compose_janjul_primary(evaluation, raw, parent)
    if scored.height != s4.PRIMARY_N and not _smoke_subset():
        raise RuntimeError(f"JAN_JUL population {scored.height} != {s4.PRIMARY_N}")
    out = scored.select(*PREDICTION_COLUMNS).with_columns(pl.Series("s4_raw_prediction", np.asarray(raw, dtype=np.float64)))
    card = s4.janjul_scorecard(scored)
    return out, card


def compose_janjul_unscored(evaluation: pl.DataFrame, raw: np.ndarray, src: Path) -> pl.DataFrame:
    """``score_janjul``'s composed output without the scorecard (no label is read; predictions-only runs)."""

    parent = pl.read_parquet(src / E33B_PARENT)
    scored = s4.compose_janjul_primary(evaluation, raw, parent)
    if scored.height != s4.PRIMARY_N and not _smoke_subset():
        raise RuntimeError(f"JAN_JUL population {scored.height} != {s4.PRIMARY_N}")
    return scored.select(*PREDICTION_COLUMNS).with_columns(pl.Series("s4_raw_prediction", np.asarray(raw, dtype=np.float64)))


def compose_ranking(evaluation: pl.DataFrame, raw: np.ndarray) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Ranking-month predictions in the trainer's schema (E17 -> E18 -> clip; unmatched rows flagged)."""

    composed = s4.compose_ranking_matched(evaluation, raw)
    if composed.height != s4.RANKING_N:
        raise RuntimeError(f"ranking population {composed.height} != {s4.RANKING_N}")
    out = composed.select(*RANKING_COLUMNS)
    matched = out.filter(pl.col("nm_unmatched") == 0)
    p = matched["prediction"].to_numpy()
    summary = {
        "n": out.height,
        "matched_n": matched.height,
        "unmatched_n": int((out["nm_unmatched"] == 1).sum()),
        "matched_checksum": s4.payload_sha256(matched, "prediction"),
        "matched_pred_mean": float(p.mean()),
        "matched_pred_min": float(p.min()),
        "matched_pred_max": float(p.max()),
        "gate_e17_rows": int(out["gate_e17"].sum()),
        "gate_e18_rows": int(out["gate_e18"].sum()),
    }
    return out, summary


def compare_ranking(out: pl.DataFrame, reference: Path) -> dict[str, Any]:
    """Row-wise comparison of matched ranking predictions with another trainer-schema ranking file."""

    ref = pl.read_parquet(reference)
    value_col = "prediction" if "prediction" in ref.columns else "TAXITIME_SEC_mvt"
    matched = out.filter(pl.col("nm_unmatched") == 0)
    joined = matched.join(ref.select("MVT_ID_mvt", pl.col(value_col).alias("ref")), on="MVT_ID_mvt", how="inner", validate="1:1")
    result = s4.rowwise_comparison(joined["prediction"].to_numpy(), joined["ref"].to_numpy())
    result["reference"] = str(reference)
    result["reference_sha256"] = s4.sha256_file(reference)
    if "s4_raw_prediction" in ref.columns:
        raw_joined = matched.join(ref.select("MVT_ID_mvt", pl.col("s4_raw_prediction").alias("ref_raw")), on="MVT_ID_mvt", how="inner", validate="1:1")
        result["raw"] = s4.rowwise_comparison(raw_joined["s4_raw_prediction"].to_numpy(), raw_joined["ref_raw"].to_numpy())
    return result


def write_janjul_predictions(out: pl.DataFrame, path: Path) -> dict[str, Any]:
    s4.atomic_parquet(out, path)
    return {"path": str(path), "sha256": s4.sha256_file(path), "rows": out.height, "all_rows_checksum": s4.payload_sha256(out, "prediction")}


write_predictions = write_janjul_predictions


def card_line(card: dict[str, Any]) -> str:
    return (
        f"JAN_JUL pooled={card['JAN_JUL_2025']['rmse_seconds']:.6f} "
        f"jan={card['JAN_2025']['rmse_seconds']:.4f} jul={card['JUL_2025']['rmse_seconds']:.4f} "
        f"matched={card['matched_only']['rmse_seconds']:.4f}"
    )


def card_vs_control(card: dict[str, Any], control: dict[str, Any] | None) -> dict[str, Any] | None:
    if not control:
        return None
    return {
        key: {
            "rmse_seconds": card[key]["rmse_seconds"],
            "control_rmse_seconds": control[key]["rmse_seconds"],
            "delta_seconds": card[key]["rmse_seconds"] - control[key]["rmse_seconds"],
        }
        for key in ("JAN_JUL_2025", "JAN_2025", "JUL_2025", "matched_only")
        if key in card and key in control
    }


def load_control_card(path: Path | None) -> dict[str, Any] | None:
    """Scorecard of a control run manifest (``run_manifest.json`` of the trainer or of an arm)."""

    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    return payload.get("scorecard", payload)


def category_levels(frame: pl.DataFrame, cats: list[str]) -> dict[str, list[str]]:
    """Sorted string levels of every categorical column (nulls as ``__UNK__``), fixed on the training rows."""

    return {c: sorted(frame[c].fill_null("__UNK__").cast(pl.String).unique().to_list()) for c in cats}


def numpy_matrix(frame: pl.DataFrame, cats: list[str], nums: list[str], levels: dict[str, list[str]]) -> np.ndarray:
    """One float32 block ``[cats..., nums...]`` for LightGBM without a pandas copy.

    Categoricals become consecutive integer codes from ``levels`` (unseen or null -> -1, which
    LightGBM treats as missing for a categorical feature); numerics are float32 with NaN for null.
    """

    exprs: list[pl.Expr] = []
    for c in cats:
        mapping = {level: float(code) for code, level in enumerate(levels[c])}
        exprs.append(
            pl.col(c).fill_null("__UNK__").cast(pl.String).replace_strict(mapping, default=-1.0, return_dtype=pl.Float32).alias(c)
        )
    exprs.extend(pl.col(c).cast(pl.Float32).alias(c) for c in nums)
    return frame.select(exprs).to_numpy(order="c")


__all__ = [name for name in globals() if not name.startswith("_")]
