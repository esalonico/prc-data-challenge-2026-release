# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Build the A3b non-copy-expert ranking overlay candidates. Never uploads.

Variants:

- ``R-A3b``         expert + P(copy) with the confirmed 09-21 ``core`` features.
- ``R-A3b-nodest``  same, without the three same-flight destination-arrival features.

Recipe (fixed by the D gates and the single January/July confirmation):

1. Expert: U6 recipe (observed unmatched features, CatBoost 400 trees, depth 6,
   lr 0.08, l2 16, nan_mode Min, seed 42) on 2025 months 1-10 unmatched rows
   minus schedule copies with takeoff - SCHED >= 3600 s (60 s lattice on the
   training BLOCK). Months 11-12 stay locked.
2. P(copy): multinomial logistic, n0 = 20, fitted on the months 1-10 unmatched
   lineage frame; ranking features from ranking DEP+ARR rows, month-isolated.
3. Compose: v12 matched rows unchanged -> unmatched
   yhat = p*max(takeoff - SCHED, 0) + (1 - p)*E_nc, p = 0 below 3600 s or when
   missing -> E17/E18 (compose_experimental_unmatched) -> clip at 0.

Outputs go to artifacts/experiments/Unmatched-noncopy-20260925/ranking/. The
repository ``submissions/`` folder is only read (v12 parent, never written).

All-year blind refit (P3, SA-UNM2, for the final freeze only)::

    --train-months 1-12 --final-freeze [--dry-run]

Months 11-12 are refused without ``--final-freeze`` (``final_freeze.FinalFreezeGuard``);
their rows are blind training rows only and are never evaluated. ``--dry-run`` fits
nothing and writes no ranking file: it reports row counts per month, label presence and
the parity of the months 11-12 feature path (S4-allyear base caches + raw sidecar) against
the months 1-10 working table. All-year outputs go to a separate ``ranking_m1_12_final/``
folder so the v18 files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _main_repo() -> Path:
    env = os.environ.get("PRC_MAIN_REPO")
    if env:
        return Path(env)
    sibling = ROOT.parent.parent / "prc-challenge-2026"
    if (sibling / "artifacts" / "cache").exists():
        return sibling
    return ROOT


MAIN = _main_repo()
CAMPAIGN = MAIN / "artifacts" / "experiments" / "Unmatched-noncopy-20260925"
OUT = CAMPAIGN / "ranking"
TRACKED = ROOT / "progress-review" / "unmatched-noncopy-20260925" / "ranking"
FROZEN = MAIN / "artifacts" / "experiments" / "Frozen-unmatched-20260921"
S0_SUB = FROZEN / "submissions" / "S0.parquet"
S1_SUB = FROZEN / "submissions" / "S1.parquet"
OVERNIGHT = MAIN / "artifacts" / "experiments" / "Overnight-lineage-20260922"
V14_MATCHED_CHECKSUM = "1bf0c6dee84454f4858421d09ad36c719d2e5f032fa1013066225ccc303e3031"

from prc_challenge.unmatched_recovery.block_lineage import (  # noqa: E402
    CLASSES,
    CORE_CATS,
    CORE_NUMS,
    load_movements,
    resolve_raw_dir,
)
from prc_challenge.unmatched_recovery.block_source_model import (  # noqa: E402
    predict_source_proba,
    prediction_checksum,
    schema_hash,
    with_family_features,
)
from prc_challenge.unmatched_recovery.final_freeze import (  # noqa: E402
    S4_FEATURES_REL,
    FinalFreezeGuard,
    dry_run_report,
    load_movements_any,
    parse_months,
    working_rows_from_base,
)
from prc_challenge.unmatched_recovery.noncopy_expert import (  # noqa: E402
    COPY_CLASS,
    GATE_SEC,
    delay_bucket,
    diff_summary,
    distribution,
    expert_training_mask,
    fit_source_model,
    mixture_prediction,
    month_isolated_lineage,
    null_fill_counts,
    source_feature_lists,
)

FIT_MONTHS = tuple(range(1, 11))
D_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)


class RunContext:
    """Training months, the final-freeze guard and the output folders of one build."""

    def __init__(self, train_months: tuple[int, ...] = FIT_MONTHS, *, final_freeze: bool = False, dry_run: bool = False, fix_october: bool = False) -> None:
        self.train_months = tuple(int(m) for m in train_months)
        self.fix_october = bool(fix_october)
        self.guard = FinalFreezeGuard(self.train_months, final_freeze=final_freeze, dry_run=dry_run)
        self.guard.check()
        self.fit_tag = f"m{self.train_months[0]}_{self.train_months[-1]}"
        if self.train_months == FIT_MONTHS:
            self.out = OUT
            self.tracked = TRACKED
        else:
            self.out = CAMPAIGN / f"ranking_{self.fit_tag}_final"
            self.tracked = TRACKED.parent / f"ranking_{self.fit_tag}_final"


CTX = RunContext()
RANKING_MONTHS = (1, 7)
THREADS = 2
N0 = 20.0
VARIANTS = {"R-A3b": False, "R-A3b-nodest": True}
LINEAGE_KEEP = tuple(
    dict.fromkeys(
        (
            "MVT_ID_mvt",
            "month",
            "AIRPORT",
            "nm_unmatched",
            "source_class",
            "takeoff_minus_sched_sec",
            "key_airport",
            "fingerprint",
            "key4",
            "key1",
            "key2",
            "key3",
            *CORE_CATS,
            *CORE_NUMS,
        )
    )
)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keep(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select([c for c in LINEAGE_KEEP if c in frame.columns])


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def phase_lineage() -> None:
    frames = CTX.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    raw_dir = resolve_raw_dir(MAIN)
    CTX.guard.check()
    _log(f"training lineage, 2025 months {list(CTX.train_months)} (DEP + ARR)")
    dep = load_movements_any(raw_dir, CTX.train_months, "DEP", CTX.guard)
    arr = load_movements_any(raw_dir, CTX.train_months, "ARR", CTX.guard)
    if set(dep["month"].unique().to_list()) != set(CTX.train_months):
        raise RuntimeError("training months incomplete")
    # Month-isolated, exactly as at ranking time: a month's DEP rows only see that month's ARR rows,
    # so day 1 has no prior-day ARR (null -> 0.0 in the source model) in training and in ranking.
    lin = month_isolated_lineage(dep, arr, CTX.train_months)
    del dep, arr
    lin_u = _keep(lin.filter(pl.col("nm_unmatched") == 1))
    del lin
    lin_u.write_parquet(frames / f"lineage_fit_{CTX.fit_tag}.parquet")
    _log(f"training lineage unmatched rows {lin_u.height} (month-isolated)")
    _log("ranking lineage, month-isolated (ranking DEP + ARR only)")
    from prc_challenge.io import ranking_path, reporting_airport_expr
    from prc_challenge.mask import mask_departure_labels

    ranking = pl.read_parquet(ranking_path(raw_dir))
    ranking = mask_departure_labels(ranking, extra_cols=("ARVT_3_flt",))
    ranking = ranking.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"), reporting_airport_expr())
    dep_nonnull = ranking.filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col("BLOCK_TIME_UTC_mvt").is_not_null().sum(), pl.col("TAXITIME_SEC_mvt").is_not_null().sum(), pl.col("AOBT_3_flt").is_not_null().sum()
    ).row(0)
    if any(dep_nonnull):
        raise RuntimeError(f"ranking DEP labels not blank after masking: {dep_nonnull}")
    dep_r = ranking.filter(pl.col("PHASE_mvt") == "DEP")
    arr_r = ranking.filter(pl.col("PHASE_mvt") == "ARR")
    lin_r = _keep(month_isolated_lineage(dep_r, arr_r, RANKING_MONTHS).filter(pl.col("nm_unmatched") == 1))
    for month in RANKING_MONTHS:
        _log(f"ranking month {month}: DEP {dep_r.filter(pl.col('month') == month).height} ARR {arr_r.filter(pl.col('month') == month).height} unmatched {lin_r.filter(pl.col('month') == month).height}")
    lin_r.write_parquet(frames / "lineage_ranking.parquet")


def _camp():
    from prc_challenge.campaign.runtime import default_config
    from prc_challenge.unmatched_recovery.frozen_campaign import FrozenCampaign

    import tempfile

    # FrozenCampaign creates empty working folders (including one named "submissions");
    # keep them out of the artifact tree. Nothing is written there by this script.
    scratch = Path(tempfile.mkdtemp(prefix="noncopy-frozen-harness-"))
    cfg = default_config(MAIN)
    return FrozenCampaign(repo=MAIN, output=scratch, cfg=cfg, threads=THREADS, skip_upload=True)


def phase_features() -> None:
    """Expert training rows and ranking rows exactly as the v14/v15 frozen harness built them."""

    from prc_challenge.unmatched_recovery.data import unmatched_only
    from prc_challenge.unmatched_recovery.frozen_campaign import load_ranking_unmatched_features
    from prc_challenge.unmatched_recovery.open_data import recipe_feature_lists

    CTX.guard.check()
    camp = _camp()
    _log("working table (months 1-10) via FrozenCampaign.load_table + enrich")
    table = camp.load_table()
    fit = camp.enrich(unmatched_only(table))
    del table
    fit = fit.filter(pl.col("month").is_in(list(CTX.train_months)))
    cats, nums = recipe_feature_lists(fit, "S0")
    if CTX.fix_october and 10 in CTX.train_months:
        # Recipe deviation (off by default): October rows with their true raw sidecar values instead of the
        # historical working table's '__UNK__' prefix / null ADES / null rule (final_freeze.OCTOBER_SIDECAR_NOTE).
        _log("October rows from the S4-allyear base cache + raw sidecar (--fix-october-sidecar)")
        october = working_rows_from_base(MAIN / S4_FEATURES_REL, resolve_raw_dir(MAIN), (10,), CTX.guard)
        fit = pl.concat([fit.filter(pl.col("month") != 10), october], how="diagonal_relaxed")
    if CTX.guard.uses_locked:
        # Blind final training rows: S4-allyear base caches (identical to the E08b cache on
        # months 1-10) + raw sidecar + labels. Never evaluated (guard.assert_no_evaluation).
        _log(f"locked months {list(CTX.guard.locked_months)} from the S4-allyear base caches (final freeze)")
        extra = working_rows_from_base(MAIN / S4_FEATURES_REL, resolve_raw_dir(MAIN), CTX.guard.locked_months, CTX.guard)
        missing_cols = [c for c in ("MVT_ID_mvt", "month", "AIRPORT", "TAXITIME_SEC_mvt", *cats, *nums) if c not in extra.columns]
        if missing_cols:
            raise RuntimeError(f"locked-month rows lack {missing_cols}")
        fit = pl.concat([fit, extra], how="diagonal_relaxed")
    if sorted(fit["month"].unique().to_list()) != list(CTX.train_months):
        raise RuntimeError(f"fit months {sorted(fit['month'].unique().to_list())}")
    _log(f"fit rows {fit.height}; ranking features")
    ranking_u = load_ranking_unmatched_features(camp)
    keep_fit = list(dict.fromkeys(["MVT_ID_mvt", "month", "AIRPORT", "TAXITIME_SEC_mvt", *cats, *nums]))
    keep_rank = list(dict.fromkeys(["MVT_ID_mvt", "month", "AIRPORT", "nm_unmatched", "type_missing", "takeoff_minus_sched_days", "takeoff_minus_sched_sec", "flight_prefix", *cats, *nums]))
    for col in keep_rank:
        if col not in ranking_u.columns:
            raise RuntimeError(f"ranking feature {col} missing")
    frames = CTX.out / "frames"
    fit.select(keep_fit).with_columns(pl.int_range(0, pl.len()).alias("_row")).write_parquet(frames / f"expert_fit_{CTX.fit_tag}.parquet")
    ranking_u.select(keep_rank).with_columns(pl.int_range(0, pl.len()).alias("_row")).write_parquet(frames / "ranking_unmatched.parquet")
    _write_json(frames / "expert_features.json", {"cats": cats, "nums": nums})
    _log(f"saved fit {fit.height} ranking {ranking_u.height}")


# ---------------------------------------------------------------------------
# fit + compose
# ---------------------------------------------------------------------------


def _source_model_hash(model: Any, cats: list[str], nums: list[str]) -> str:
    return schema_hash({"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(), "classes": [str(c) for c in model.classes_], "cats": cats, "nums": nums})


def phase_fit() -> None:
    from prc_challenge.io import scan_submitting
    from prc_challenge.unmatched_recovery.harness import (
        RANKING_N,
        RANKING_UNMATCHED_N,
        overlay_unmatched_submission,
        payload_sha256,
        prepare_unmatched_preds,
        verify_arm_submission,
    )
    from prc_challenge.unmatched_recovery.methods import fit_regressor, predict_regressor

    CTX.guard.check()
    frames = CTX.out / "frames"
    models = CTX.out / "models"
    models.mkdir(parents=True, exist_ok=True)
    fit = pl.read_parquet(frames / f"expert_fit_{CTX.fit_tag}.parquet").sort("_row")
    if sorted(fit["month"].unique().to_list()) != list(CTX.train_months):
        raise RuntimeError("training frame months differ from the run context")
    # This phase fits and composes; it never scores. Any future scoring must call this first.
    CTX.guard.assert_no_evaluation([], context="phase_fit")
    rank = pl.read_parquet(frames / "ranking_unmatched.parquet").sort("_row")
    feats = json.loads((frames / "expert_features.json").read_text())
    cats, nums = feats["cats"], feats["nums"]
    lin_fit = pl.read_parquet(frames / f"lineage_fit_{CTX.fit_tag}.parquet")
    lin_rank = pl.read_parquet(frames / "lineage_ranking.parquet")
    if rank.height != RANKING_UNMATCHED_N:
        raise RuntimeError(f"ranking unmatched {rank.height}")
    if set(lin_rank["MVT_ID_mvt"].to_list()) != set(rank["MVT_ID_mvt"].to_list()):
        raise RuntimeError("ranking lineage rows differ from ranking unmatched feature rows")
    if set(lin_fit["MVT_ID_mvt"].to_list()) != set(fit["MVT_ID_mvt"].to_list()):
        raise RuntimeError("training lineage rows differ from expert training rows")
    delay_check = rank.select("MVT_ID_mvt", pl.col("takeoff_minus_sched_sec").alias("a")).join(
        lin_rank.select("MVT_ID_mvt", pl.col("takeoff_minus_sched_sec").alias("b")), on="MVT_ID_mvt"
    )
    delay_gap = float((delay_check["a"] - delay_check["b"]).abs().max() or 0.0)
    if delay_gap > 1e-6:
        raise RuntimeError(f"takeoff-SCHED differs between feature frames by {delay_gap}")

    # 1) Parity: U6 (v14 S0) refit must reproduce S0's ranking unmatched values (months 1-10 only;
    #    the all-year freeze has no S0 analogue, so the parity refit is skipped there).
    u6_parity: float | None = None
    if CTX.train_months == FIT_MONTHS:
        _log(f"U6 parity refit on {fit.height} rows")
        u6_model = fit_regressor(fit, cats, nums, iterations=400, thread_count=THREADS, l2_leaf_reg=16.0)
        u6_model.save_model(str(models / f"u6_{CTX.fit_tag}.cbm"))
        u6_scored = prepare_unmatched_preds(rank, predict_regressor(u6_model, rank, cats, nums))
        s0 = pl.read_parquet(S0_SUB).rename({"TAXITIME_SEC_mvt": "s0"})
        par = u6_scored.select("MVT_ID_mvt", "prediction").join(s0, on="MVT_ID_mvt", how="left")
        u6_parity = float((par["prediction"] - par["s0"]).abs().max())
        _log(f"U6 vs S0 max |diff| {u6_parity}")
    else:
        u6_scored = rank.select("MVT_ID_mvt").with_columns(pl.lit(None, dtype=pl.Float64).alias("prediction"))

    # 2) Expert on months 1-10 minus large schedule copies.
    work = fit.join(lin_fit.select("MVT_ID_mvt", "source_class"), on="MVT_ID_mvt", how="left", maintain_order="left")
    if work["source_class"].null_count():
        raise RuntimeError("source class missing on training rows")
    mask = expert_training_mask(work["source_class"].to_numpy(), work["takeoff_minus_sched_sec"].to_numpy(), mode="noncopy_large")
    train = work.filter(pl.Series(mask))
    _log(f"expert train {train.height} (dropped {int((~mask).sum())} large schedule copies)")
    expert = fit_regressor(train, cats, nums, iterations=400, thread_count=THREADS, l2_leaf_reg=16.0)
    expert_path = models / f"expert_noncopy_large_{CTX.fit_tag}.cbm"
    expert.save_model(str(expert_path))
    e_pred = predict_regressor(expert, rank, cats, nums)
    delay = rank["takeoff_minus_sched_sec"].to_numpy().astype(np.float64)

    # 3) P(copy) variants on months 1-10; deterministic row order (MVT id).
    lin_fit_sorted = lin_fit.sort("MVT_ID_mvt")
    fit_x = with_family_features(lin_fit_sorted, lin_fit_sorted, n0=N0, exclude_self=True)
    hold_x = with_family_features(lin_rank, lin_fit_sorted, n0=N0, exclude_self=False)
    parent = pl.read_parquet(MAIN / "submissions" / "unbelievable-snowflake_v12.parquet")
    parent_sha = _sha256(MAIN / "submissions" / "unbelievable-snowflake_v12.parquet")
    template = scan_submitting(MAIN / "data" / "raw").collect()
    class_counts = {str(r["source_class"]): int(r["count"]) for r in lin_fit["source_class"].value_counts().iter_rows(named=True)}
    preds_frame = rank.select("MVT_ID_mvt", "AIRPORT", "month", "flight_prefix", "takeoff_minus_sched_sec", "type_missing", "takeoff_minus_sched_days")
    preds_frame = preds_frame.with_columns(pl.Series("expert", e_pred), pl.Series("u6_refit", u6_scored["prediction"].to_numpy()))
    for variant, drop in VARIANTS.items():
        s_cats, s_nums = source_feature_lists(drop_dest_arrival=drop)
        model, encoder, scaler = fit_source_model(fit_x, s_cats, s_nums)
        with (models / f"pcopy_{variant}.pkl").open("wb") as handle:
            pickle.dump({"model": model, "encoder": encoder, "scaler": scaler, "cats": s_cats, "nums": s_nums, "n0": N0}, handle)
        proba = predict_source_proba(model, encoder, hold_x, s_cats, s_nums, scaler)
        p_frame = hold_x.select("MVT_ID_mvt").with_columns(pl.Series("p", proba[:, CLASSES.index(COPY_CLASS)]))
        p = rank.select("MVT_ID_mvt").join(p_frame, on="MVT_ID_mvt", how="left", maintain_order="left")["p"].to_numpy().astype(np.float64)
        if not np.isfinite(p).all():
            raise RuntimeError("P(copy) missing on ranking rows")
        raw_pred = mixture_prediction(p, delay, e_pred, gate_sec=GATE_SEC)
        scored = prepare_unmatched_preds(rank, raw_pred)
        submission = overlay_unmatched_submission(parent, scored)
        verify = verify_arm_submission(submission, template=template, parent=parent, unmatched_ids=rank["MVT_ID_mvt"], expected_unmatched_n=RANKING_UNMATCHED_N, expected_n=RANKING_N)
        if not verify["ok"]:
            raise RuntimeError(f"{variant} verification failed: {verify['errors']}")
        if verify["matched_checksum"] != V14_MATCHED_CHECKSUM:
            raise RuntimeError(f"{variant} matched checksum {verify['matched_checksum']} != v14 {V14_MATCHED_CHECKSUM}")
        dest = CTX.out / f"{variant}.parquet"
        submission.write_parquet(dest)
        u_vals = scored["prediction"].to_numpy().astype(np.float64)
        manifest = {
            "variant": variant,
            "prize_candidate": variant == "R-A3b-nodest",
            "lineage_isolation": {"training": "per 2025 month 1-10, DEP sees only same-month ARR", "ranking": "per ranking month (Jan, Jul 2026), DEP sees only same-month ARR"},
            "null_fill_0": {"ranking": null_fill_counts(hold_x, s_nums), "training": null_fill_counts(fit_x, s_nums)},
            "file": str(dest),
            "sha256": _sha256(dest),
            "n": int(submission.height),
            "unmatched_n": int(rank.height),
            "matched_checksum": verify["matched_checksum"],
            "matched_checksum_equals_v14": verify["matched_checksum"] == V14_MATCHED_CHECKSUM,
            "unmatched_checksum": verify["unmatched_checksum"],
            "parent": {"file": "submissions/unbelievable-snowflake_v12.parquet", "sha256": parent_sha},
            "validator": {"function": "prc_challenge.submission.validate_submission via harness.verify_arm_submission", "ok": verify["ok"], "errors": verify["errors"], "matched_changed": verify["matched_changed"], "nulls": verify["nulls"], "finite": verify["finite"]},
            "pred_all": {"min": float(submission["TAXITIME_SEC_mvt"].min()), "mean": float(submission["TAXITIME_SEC_mvt"].mean()), "max": float(submission["TAXITIME_SEC_mvt"].max())},
            "pred_unmatched": {"min": float(u_vals.min()), "mean": float(u_vals.mean()), "max": float(u_vals.max())},
            "p_copy": distribution(p),
            "gate_open_rows": int((np.nan_to_num(delay, nan=-np.inf) >= GATE_SEC).sum()),
            "models": {
                "expert": {"path": str(expert_path), "sha256": _sha256(expert_path), "mode": "noncopy_large", "iterations": 400, "depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 16.0, "nan_mode": "Min", "random_seed": 42, "thread_count": THREADS, "n_train": int(train.height), "n_dropped_large_copies": int((~mask).sum())},
                "p_copy": {"path": str(models / f"pcopy_{variant}.pkl"), "sha256": _sha256(models / f"pcopy_{variant}.pkl"), "coef_hash": _source_model_hash(model, s_cats, s_nums), "kind": "core" + ("-nodest" if drop else ""), "n0": N0, "solver": "lbfgs", "max_iter": 200, "C": 1.0, "row_order": "MVT_ID_mvt"},
            },
            "training_months": list(CTX.train_months),
            "final_freeze": CTX.guard.describe(),
            "october_sidecar": "base cache + raw (--fix-october-sidecar)" if CTX.fix_october else "historical working table (frozen recipe; '__UNK__' prefix, null ADES/rule)",
            "training_unmatched_rows": int(fit.height),
            "training_unmatched_rows_per_month": {str(r["month"]): int(r["len"]) for r in fit.group_by("month").len().sort("month").iter_rows(named=True)},
            "training_source_class_counts": class_counts,
            "ranking_months": list(RANKING_MONTHS),
            "ranking_feature_isolation": "lineage per ranking month from that month's DEP+ARR rows only; U6 features from frozen_campaign.load_ranking_unmatched_features",
            "features": {"expert_cats": cats, "expert_nums": nums, "p_copy_cats": s_cats, "p_copy_nums": s_nums},
            "composition": "v12 matched unchanged -> p*max(takeoff-SCHED,0)+(1-p)*E_nc with p=0 below 3600 s or missing -> E17/E18 compose_experimental_unmatched -> clip >= 0",
            "gate_sec": GATE_SEC,
            "u6_parity_max_abs_diff_vs_S0": u6_parity,
            "forbidden_check": {"aobt": False, "arvt3": False, "dep_block_or_taxi": False, "v4_months": CTX.guard.uses_locked, "v4_role": "blind training rows only, never evaluated" if CTX.guard.uses_locked else "unused", "opensky": False, "opdi": False},
            "uploaded": False,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json(CTX.out / f"{variant}.manifest.json", manifest)
        _write_json(CTX.tracked / f"{variant}.manifest.json", manifest)
        preds_frame = preds_frame.with_columns(pl.Series(f"p_{variant}", p), pl.Series(f"raw_{variant}", raw_pred), pl.Series(variant, u_vals))
        _log(f"{variant}: sha256 {manifest['sha256'][:12]} matched ok, unmatched mean {manifest['pred_unmatched']['mean']:.1f}")
    preds_frame.write_parquet(CTX.out / "ranking_unmatched_predictions.parquet")


# ---------------------------------------------------------------------------
# diff (feature-only; ranking labels are hidden)
# ---------------------------------------------------------------------------


def _janjul_2025_p() -> dict[str, np.ndarray]:
    """P(copy) of a D-fit model on Jan/Jul 2025 unmatched rows and on ranking rows (features only, no labels)."""

    # D lineage built month-isolated (as deployed); the Jan/Jul 2025 rows come from the month-isolated 1-10 frame.
    lin_d = pl.read_parquet(CAMPAIGN / "frames" / "d_unmatched_lineage_iso.parquet")
    mf = OVERNIGHT / "model_frame.parquet"
    if mf.exists():
        order = pl.read_parquet(mf, columns=["MVT_ID_mvt"]).with_columns(pl.int_range(0, pl.len()).alias("_o"))
        lin_d = lin_d.join(order, on="MVT_ID_mvt", how="left").sort("_o", nulls_last=True).drop("_o")
    lin_fit = pl.read_parquet(CTX.out / "frames" / "lineage_fit_m1_10.parquet")
    lin_jj = lin_fit.filter(pl.col("month").is_in([1, 7]))
    lin_rank = pl.read_parquet(CTX.out / "frames" / "lineage_ranking.parquet")
    fit_x = with_family_features(lin_d, lin_d, n0=N0, exclude_self=True)
    out: dict[str, np.ndarray] = {}
    for variant, drop in VARIANTS.items():
        s_cats, s_nums = source_feature_lists(drop_dest_arrival=drop)
        model, encoder, scaler = fit_source_model(fit_x, s_cats, s_nums)
        for name, target in (("janjul2025", lin_jj), ("ranking", lin_rank)):
            hold = with_family_features(target, lin_d, n0=N0, exclude_self=False)
            proba = predict_source_proba(model, encoder, hold, s_cats, s_nums, scaler)
            out[f"{variant}|dfit|{name}"] = proba[:, CLASSES.index(COPY_CLASS)]
            out[f"{variant}|dfit|{name}|delay"] = hold["takeoff_minus_sched_sec"].to_numpy().astype(np.float64)
            out[f"{variant}|dfit|{name}|airport"] = hold["AIRPORT"].to_numpy()
    return out


def _p_block(p: np.ndarray, delay: np.ndarray, airport: np.ndarray) -> dict[str, Any]:
    gate = np.nan_to_num(delay, nan=-np.inf) >= GATE_SEC
    bucket = delay_bucket(delay)
    return {
        "all": distribution(p),
        "gate_open": distribution(p[gate]),
        "gate_open_share": float(gate.mean()),
        "by_delay_bucket": {b: {"n": int((bucket == b).sum()), "mean_p": float(p[bucket == b].mean()) if (bucket == b).any() else None} for b in ("<1h", "1-2h", "2-4h", ">=4h", "missing")},
        "lirf": distribution(p[airport == "LIRF"]),
        "lirf_share": float((airport == "LIRF").mean()),
    }


def phase_diff() -> None:
    frame = pl.read_parquet(CTX.out / "ranking_unmatched_predictions.parquet")
    s0 = pl.read_parquet(S0_SUB).rename({"TAXITIME_SEC_mvt": "v14_S0"})
    s1 = pl.read_parquet(S1_SUB).rename({"TAXITIME_SEC_mvt": "v15_S1"})
    frame = frame.join(s0, on="MVT_ID_mvt", how="left", maintain_order="left").join(s1, on="MVT_ID_mvt", how="left", maintain_order="left")
    airport = frame["AIRPORT"].to_numpy()
    delay = frame["takeoff_minus_sched_sec"].to_numpy().astype(np.float64)
    prefix = frame["flight_prefix"].to_numpy()
    v14 = frame["v14_S0"].to_numpy().astype(np.float64)
    same_both = (np.abs(frame["R-A3b"].to_numpy() - v14) < 1e-6) & (np.abs(frame["R-A3b-nodest"].to_numpy() - v14) < 1e-6)
    u6_parity = float(np.abs(frame["u6_refit"].to_numpy() - v14).max())
    out: dict[str, Any] = {"n_ranking_unmatched": frame.height, "rows_equal_to_v14_in_both_variants": int(same_both.sum()), "u6_refit_vs_v14_max_abs_diff": u6_parity}
    for variant in VARIANTS:
        new = frame[variant].to_numpy().astype(np.float64)
        p = frame[f"p_{variant}"].to_numpy().astype(np.float64)
        out[variant] = {
            "vs_v14_S0": diff_summary(frame["v14_S0"].to_numpy(), new, airport=airport, delay=delay, p=p, prefix=prefix),
            "vs_v15_S1": diff_summary(frame["v15_S1"].to_numpy(), new, airport=airport, delay=delay, p=p, prefix=prefix),
        }
    a = frame["R-A3b"].to_numpy().astype(np.float64)
    b = frame["R-A3b-nodest"].to_numpy().astype(np.float64)
    out["R-A3b_vs_nodest"] = {
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "n_abs_diff_gt_500": int((np.abs(a - b) > 500).sum()),
        "n_abs_diff_gt_0": int((np.abs(a - b) > 1e-6).sum()),
        "max_abs_diff": float(np.abs(a - b).max()),
        "mean_abs_diff": float(np.abs(a - b).mean()),
        "p_pearson": float(np.corrcoef(frame["p_R-A3b"].to_numpy(), frame["p_R-A3b-nodest"].to_numpy())[0, 1]),
    }
    dfit = _janjul_2025_p()
    p_dist: dict[str, Any] = {}
    for variant in VARIANTS:
        p_dist[variant] = {
            "ranking_model_m1_10_on_ranking": _p_block(frame[f"p_{variant}"].to_numpy(), delay, airport),
            "dfit_model_on_janjul2025": _p_block(dfit[f"{variant}|dfit|janjul2025"], dfit[f"{variant}|dfit|janjul2025|delay"], dfit[f"{variant}|dfit|janjul2025|airport"]),
            "dfit_model_on_ranking": _p_block(dfit[f"{variant}|dfit|ranking"], dfit[f"{variant}|dfit|ranking|delay"], dfit[f"{variant}|dfit|ranking|airport"]),
        }
    out["p_copy_distribution"] = p_dist
    _write_json(CTX.out / "diff_summary.json", out)
    _write_json(CTX.tracked / "diff_summary.json", out)
    _log(f"diff written; R-A3b vs nodest pearson {out['R-A3b_vs_nodest']['pearson']:.5f}, >500 s {out['R-A3b_vs_nodest']['n_abs_diff_gt_500']}")


def phase_dry_run() -> None:
    """Row counts, label presence and feature parity only. No fit, no model, no ranking file."""

    reference = OUT / "frames" / f"expert_fit_m1_10.parquet"
    feats_path = OUT / "frames" / "expert_features.json"
    reference_fit = pl.read_parquet(reference) if reference.exists() else None
    feature_cols: list[str] = []
    if feats_path.exists():
        feats = json.loads(feats_path.read_text())
        feature_cols = [*feats["cats"], *feats["nums"], "TAXITIME_SEC_mvt"]
    report = dry_run_report(MAIN / S4_FEATURES_REL, resolve_raw_dir(MAIN), CTX.guard, reference_fit=reference_fit, feature_cols=feature_cols)
    report["reference_fit"] = str(reference) if reference_fit is not None else None
    report["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(CTX.out / "dry_run.json", report)
    _write_json(CTX.tracked / "dry_run.json", report)
    print(json.dumps(_jsonable({k: v for k, v in report.items() if k != "parity_months_1_10"}), indent=2))
    par = report["parity_months_1_10"]
    print("parity months 1-10:", {k: v for k, v in par.items() if k != "mismatches"}, "mismatching columns:", {k: v for k, v in par.get("mismatches", {}).items() if v})


def main() -> None:
    global CTX
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", action="append", choices=("lineage", "features", "fit", "diff"), default=None)
    parser.add_argument("--train-months", default="1-10", help="'1-10' (default, the v18 build), '1-12' (final freeze only) or a comma list")
    parser.add_argument("--final-freeze", action="store_true", help="required for any training month in 11-12; V4 rows are blind training rows only")
    parser.add_argument("--dry-run", action="store_true", help="row counts, label presence and feature parity only; fits nothing, writes no ranking file")
    parser.add_argument("--fix-october-sidecar", action="store_true", help="recipe deviation: October rows with their true raw sidecar values (see final_freeze.OCTOBER_SIDECAR_NOTE)")
    args = parser.parse_args()
    months = parse_months(args.train_months)
    # RunContext() calls guard.check(): months 11-12 without --final-freeze raise here, before any I/O.
    CTX = RunContext(months, final_freeze=args.final_freeze, dry_run=args.dry_run, fix_october=args.fix_october_sidecar)
    if args.dry_run:
        if args.phase:
            raise SystemExit("--dry-run takes no --phase")
        CTX.out.mkdir(parents=True, exist_ok=True)
        phase_dry_run()
        return
    if not args.phase:
        raise SystemExit("need --phase (or --dry-run)")
    if CTX.guard.uses_locked and not CTX.guard.final_freeze:
        raise SystemExit("months 11-12 need --final-freeze")
    CTX.out.mkdir(parents=True, exist_ok=True)
    for phase in args.phase:
        {"lineage": phase_lineage, "features": phase_features, "fit": phase_fit, "diff": phase_diff}[phase]()


if __name__ == "__main__":
    main()
