# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Frozen four-arm unmatched submission campaign.

Recipes freeze from 2025 D / internal validation only. Ranking scores never
enter feature, weight, threshold, or later-arm decisions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prc_challenge.campaign.hybrid_retrieval import atomic_json, pooled_score, sha256_file
from prc_challenge.campaign.runtime import CampaignConfig, cap_numeric_threads, default_config
from prc_challenge.io import scan_ranking, scan_submitting, with_reporting_airport
from prc_challenge.mask import mask_departure_labels
from prc_challenge.s3 import TEAM_BUCKET, TEAM_NAME, list_objects, s3_client
from prc_challenge.traffic import add_delayed_bunching
from prc_challenge.unmatched_recovery.data import (
    d_frame,
    load_s4_parent,
    load_working_table,
    primary_frame,
    unmatched_only,
)
from prc_challenge.unmatched_recovery.harness import (
    DAILY_CAP,
    PRIMARY_N,
    PRIMARY_UNMATCHED_N,
    RANKING_N,
    RANKING_UNMATCHED_N,
    assert_cross_arm_matched_identical,
    daily_quota,
    file_sha256,
    jsonable,
    overlay_unmatched_submission,
    payload_sha256,
    prepare_unmatched_preds,
    splice_primary,
    verify_arm_submission,
)
from prc_challenge.unmatched_recovery.methods import fit_regressor, observed_lists, predict_regressor
from prc_challenge.unmatched_recovery.open_data import (
    COMBINED_VARIANTS,
    attach_open_data_features,
    attach_open_weather_features,
    attach_physical_geometry_features,
    provenance_for,
    recipe_feature_lists,
)
from prc_challenge.unmatched_recovery.registry import git_head, now_utc
from prc_challenge.unmatched_recovery.schema import BLOCKS, D_MONTHS, LOCKED_MONTHS, score_populations
from prc_challenge.unmatched_recovery.subgroups import assert_design_months, attach_identity_columns
from prc_challenge.weather import add_metar_features, load_metar

PREV_RECOVERY = Path("artifacts/experiments/Unmatched-recovery-20260920")
U6_PRIMARY_NAME = "finalist_U6.parquet"
PUBLISHED_U6_FULL = 339.272082
PUBLISHED_U6_UNMATCHED = 1910.607593
PUBLISHED_U6_D = 1932.69
PARENT_SUB = f"{TEAM_NAME}_v12.parquet"
ARM_ORDER = ("S0", "S1", "S2", "S3")
SPECIALIST = {"iterations": 400, "depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 16.0, "random_seed": 42}


@dataclass
class FrozenCampaign:
    repo: Path
    output: Path
    cfg: CampaignConfig
    smoke: bool = False
    threads: int = 4
    skip_upload: bool = False
    force: bool = False
    metar: pl.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ("freeze", "metrics", "predictions", "models", "submissions", "verification", "uploads"):
            (self.output / name).mkdir(exist_ok=True)
        cap_numeric_threads(self.threads)

    @property
    def iterations(self) -> int:
        return 20 if self.smoke else int(SPECIALIST["iterations"])

    @property
    def freeze_path(self) -> Path:
        return self.output / "freeze" / "freeze_manifest.json"

    def load_table(self) -> pl.DataFrame:
        sidecar = self.repo / PREV_RECOVERY / "raw_sidecar.parquet"
        if not sidecar.exists():
            sidecar = self.output / "raw_sidecar.parquet"
        frame = load_working_table(self.cfg, self.repo, sidecar)
        frame = attach_identity_columns(frame)
        months = [int(m) for m in frame["month"].unique().to_list()] if "month" in frame.columns else []
        if any(m in LOCKED_MONTHS for m in months):
            raise RuntimeError("V4 months leaked into working table")
        if self.smoke:
            u = unmatched_only(frame)
            keep_u = u.sample(n=min(280, u.height), seed=42, shuffle=True) if u.height > 280 else u
            matched = frame.filter(pl.col("nm_unmatched") == 0).sample(
                n=min(400, frame.filter(pl.col("nm_unmatched") == 0).height), seed=42, shuffle=True
            )
            frame = pl.concat([matched, keep_u], how="vertical_relaxed")
        return frame

    def load_metar(self) -> pl.DataFrame | None:
        if self.metar is not None:
            return self.metar
        path = self.repo / "data" / "external" / "iem_metar"
        if path.exists():
            self.metar = load_metar(path)
        return self.metar

    def enrich(self, frame: pl.DataFrame) -> pl.DataFrame:
        return attach_open_data_features(frame, repo=self.repo, metar=self.load_metar())

    def s4_parent(self) -> pl.DataFrame:
        parent = load_s4_parent(self.repo)
        if "parent_prediction" not in parent.columns:
            parent = parent.with_columns(pl.col("prediction").alias("parent_prediction"))
        return parent

    def ranking_parent(self) -> pl.DataFrame:
        path = self.repo / "submissions" / PARENT_SUB
        if not path.exists():
            raise FileNotFoundError(path)
        return pl.read_parquet(path)

    def template(self) -> pl.DataFrame:
        return scan_submitting(self.repo / "data" / "raw").collect()


def compact_primary(metrics: dict[str, Any]) -> dict[str, Any]:
    months = metrics.get("months") or {}
    u_months = metrics.get("unmatched_months") or {}

    def _rmse(block: dict[str, Any] | None) -> float | None:
        if not block:
            return None
        return block.get("rmse_seconds")

    return {
        "n": metrics.get("n"),
        "rmse_seconds": metrics.get("rmse_seconds"),
        "sse": metrics.get("sse"),
        "n_unmatched": metrics.get("n_unmatched"),
        "unmatched_rmse_seconds": metrics.get("unmatched_rmse_seconds"),
        "unmatched_sse": metrics.get("unmatched_sse"),
        "JAN_2025": _rmse(months.get("JAN_2025")),
        "JUL_2025": _rmse(months.get("JUL_2025")),
        "JAN_2025_unmatched": _rmse(u_months.get("JAN_2025")),
        "JUL_2025_unmatched": _rmse(u_months.get("JUL_2025")),
        "unmatched_rows_dropped": metrics.get("unmatched_rows_dropped", 0),
        "freeze_applied": metrics.get("freeze_applied", False),
        "v4_excluded": True,
    }


def fit_predict(
    train: pl.DataFrame,
    score: pl.DataFrame,
    cats: list[str],
    nums: list[str],
    *,
    iterations: int,
    thread_count: int,
) -> tuple[np.ndarray, Any]:
    model = fit_regressor(
        train,
        cats,
        nums,
        iterations=iterations,
        thread_count=thread_count,
        depth=int(SPECIALIST["depth"]),
        learning_rate=float(SPECIALIST["learning_rate"]),
        l2_leaf_reg=float(SPECIALIST["l2_leaf_reg"]),
    )
    return predict_regressor(model, score, cats, nums), model


def blocked_unmatched(
    unmatched: pl.DataFrame,
    recipe: str,
    *,
    iterations: int,
    thread_count: int,
) -> dict[str, Any]:
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    folds: list[dict[str, Any]] = []
    cats, nums = recipe_feature_lists(unmatched, recipe)
    for holdout in BLOCKS:
        fit = unmatched.filter(~pl.col("month").is_in(list(holdout)))
        hold = unmatched.filter(pl.col("month").is_in(list(holdout)))
        if fit.height == 0 or hold.height == 0:
            continue
        assert_design_months([int(m) for m in fit["month"].unique().to_list()])
        pred, _model = fit_predict(fit, hold, cats, nums, iterations=iterations, thread_count=thread_count)
        y = hold["TAXITIME_SEC_mvt"].to_numpy()
        parts.append((y, pred))
        sse = float(np.dot(pred - y, pred - y))
        folds.append({"holdout": list(holdout), "n": int(hold.height), "rmse_seconds": float(np.sqrt(sse / hold.height))})
    pooled = pooled_score(parts) if parts else {"n": 0, "rmse_seconds": None}
    return {
        "recipe": recipe,
        "cats": cats,
        "nums": nums,
        "folds": folds,
        "n": pooled.get("n"),
        "unmatched_rmse_seconds": pooled.get("rmse_seconds"),
        "sse": pooled.get("sse"),
        "design_months": list(D_MONTHS),
    }


def consume_agent2_handoff(repo: Path) -> dict[str, Any]:
    path = repo / "progress-review" / "2026-09-18-agent2-handoff.json"
    if not path.exists():
        return {"available": False, "used_for_unmatched_selection": False, "reason": "missing"}
    payload = json.loads(path.read_text())
    return {
        "available": True,
        "used_for_unmatched_selection": False,
        "reason": "handoff is OPDI/native operational residual, not an unmatched weather/geometry specialist",
        "status": payload.get("status"),
        "source_base_commit": payload.get("source_base_commit"),
        "candidates": [c.get("model_id") for c in payload.get("candidates") or []],
    }


def screen_d(camp: FrozenCampaign, unmatched: pl.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}
    results["S0"] = {
        "recipe": "S0",
        "cats": recipe_feature_lists(unmatched, "S0")[0],
        "nums": recipe_feature_lists(unmatched, "S0")[1],
        "unmatched_rmse_seconds": PUBLISHED_U6_D if not camp.smoke else None,
        "source": "published U6 blocked-D OOF",
        "refit": False,
    }
    for recipe in ("S1", "S2", *COMBINED_VARIANTS):
        print(f"D-screen {recipe}", flush=True)
        results[recipe] = blocked_unmatched(
            unmatched, recipe, iterations=camp.iterations, thread_count=camp.threads
        )
    pick = min(
        COMBINED_VARIANTS,
        key=lambda name: (
            results[name].get("unmatched_rmse_seconds")
            if results[name].get("unmatched_rmse_seconds") is not None
            else 1e18
        ),
    )
    return {
        "arms": results,
        "s3_selected": pick,
        "selection_rule": "lowest blocked-D unmatched RMSE among combined open-data variants; Jan/Jul unused",
        "agent2": consume_agent2_handoff(camp.repo),
    }


def write_freeze(camp: FrozenCampaign, screen: dict[str, Any], unmatched: pl.DataFrame) -> dict[str, Any]:
    if camp.freeze_path.exists() and not camp.force:
        return json.loads(camp.freeze_path.read_text())
    s3_name = screen["s3_selected"]
    recipes = {
        "S0": {"role": "control", "recipe": "S0", "name": "U6 control"},
        "S1": {"role": "open_weather_disruption", "recipe": "S1", "name": "IEM METAR disruption"},
        "S2": {"role": "osm_ourairports_geometry", "recipe": "S2", "name": "OSM/OurAirports physical geometry"},
        "S3": {
            "role": "combined_open_data",
            "recipe": s3_name,
            "name": "internally selected combined open-data",
            "selected_from": list(COMBINED_VARIANTS),
        },
    }
    for arm, spec in recipes.items():
        cats, nums = recipe_feature_lists(unmatched, spec["recipe"])
        spec["cats"] = cats
        spec["nums"] = nums
        spec["hyperparameters"] = {**SPECIALIST, "iterations": camp.iterations, "thread_count": camp.threads}
        spec["train_months"] = list(D_MONTHS)
        spec["ranking_fit_months"] = list(range(1, 11))
        spec["provenance"] = provenance_for(spec["recipe"])
        d_key = "S0" if arm == "S0" else spec["recipe"]
        spec["d_unmatched_rmse_seconds"] = (screen["arms"].get(d_key) or {}).get("unmatched_rmse_seconds")
    payload = {
        "frozen_utc": now_utc(),
        "source_commit": git_head(camp.repo),
        "protocol": "JAN_JUL_2025",
        "leaderboard_inspected": False,
        "leaderboard_scores": None,
        "adapted_from_ranking": False,
        "v4_used": False,
        "open_meteo_used": False,
        "selection": {
            "universe": "D={2,3,4,5,6,8,9,10} unmatched only",
            "s3_selected": s3_name,
            "rule": screen["selection_rule"],
            "d_screen": {
                name: {
                    "unmatched_rmse_seconds": (row or {}).get("unmatched_rmse_seconds"),
                    "n": (row or {}).get("n"),
                }
                for name, row in (screen.get("arms") or {}).items()
            },
        },
        "agent2_handoff": screen.get("agent2"),
        "parent": {
            "local": "S4 expected occupancy Jan/Jul 2025",
            "ranking": PARENT_SUB,
        },
        "arms": recipes,
        "order": list(ARM_ORDER),
        "specialist": SPECIALIST,
        "gates": "E17 then E18; unmatched parent freeze OFF",
    }
    atomic_json(camp.freeze_path, jsonable(payload))
    if not camp.smoke:
        review = camp.repo / "progress-review" / "2026-09-21-frozen-unmatched-freeze-manifest.json"
        review.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    return payload


def _join_gate_cols(parent: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    need = [
        c
        for c in ("AIRPORT", "type_missing", "takeoff_minus_sched_days", "takeoff_minus_sched_sec", "month", "nm_unmatched")
        if c in source.columns and c not in parent.columns
    ]
    if not need:
        return parent
    return parent.join(source.select(["MVT_ID_mvt", *need]), on="MVT_ID_mvt", how="left")


def score_primary_arm(
    camp: FrozenCampaign,
    arm: str,
    spec: dict[str, Any],
    train: pl.DataFrame,
    score: pl.DataFrame,
    parent: pl.DataFrame,
) -> dict[str, Any]:
    cats, nums = spec["cats"], spec["nums"]
    if arm == "S0" and not camp.smoke:
        path = camp.repo / PREV_RECOVERY / "predictions" / U6_PRIMARY_NAME
        if path.exists():
            scored = pl.read_parquet(path)
            if "parent_prediction" not in scored.columns and "prediction" in scored.columns:
                scored = scored.with_columns(pl.col("prediction").alias("parent_prediction"))
            metrics = score_populations(scored)
            dest = camp.output / "predictions" / f"{arm}_janjul.parquet"
            keep = [c for c in ("MVT_ID_mvt", "month", "AIRPORT", "nm_unmatched", "TAXITIME_SEC_mvt", "prediction", "parent_prediction") if c in scored.columns]
            scored.select(keep).write_parquet(dest)
            return {
                "arm": arm,
                "source": str(path),
                "metrics": compact_primary(metrics),
                "prediction_path": str(dest),
                "n": int(scored.height),
                "n_unmatched": int(scored.filter(pl.col("nm_unmatched") == 1).height) if "nm_unmatched" in scored.columns else None,
            }
    pred, model = fit_predict(train, score, cats, nums, iterations=camp.iterations, thread_count=camp.threads)
    model_path = camp.output / "models" / f"{arm}_d.cbm"
    model.save_model(str(model_path))
    cand = prepare_unmatched_preds(score, pred)
    parent_g = _join_gate_cols(parent, score)
    composed = splice_primary(parent_g, cand)
    if not camp.smoke and composed.height != PRIMARY_N:
        raise RuntimeError(f"{arm} primary n {composed.height} != {PRIMARY_N}")
    metrics = score_populations(composed)
    dest = camp.output / "predictions" / f"{arm}_janjul.parquet"
    keep = [c for c in ("MVT_ID_mvt", "month", "AIRPORT", "nm_unmatched", "TAXITIME_SEC_mvt", "prediction", "parent_prediction") if c in composed.columns]
    composed.select(keep).write_parquet(dest)
    return {
        "arm": arm,
        "source": "d_refit",
        "model_path": str(model_path),
        "metrics": compact_primary(metrics),
        "prediction_path": str(dest),
        "n": int(composed.height),
        "n_unmatched": int(composed.filter(pl.col("nm_unmatched") == 1).height),
        "finite": bool(np.isfinite(composed["prediction"].to_numpy()).all()),
    }


def attach_ranking_bunching(dep: pl.DataFrame, ranking_raw: pl.DataFrame) -> pl.DataFrame:
    raw = ranking_raw
    if "month" not in raw.columns and "MVT_TIME_UTC_mvt" in raw.columns:
        raw = raw.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
    parts: list[pl.DataFrame] = []
    for month in dep["month"].unique().to_list():
        tgt = dep.filter(pl.col("month") == month)
        ctx = raw.filter((pl.col("month") == month) & (pl.col("PHASE_mvt") == "DEP"))
        parts.append(add_delayed_bunching(tgt, ctx))
    return pl.concat(parts) if parts else dep


def load_ranking_unmatched_features(camp: FrozenCampaign) -> pl.DataFrame:
    raw_dir = camp.repo / "data" / "raw"
    ranking = with_reporting_airport(scan_ranking(raw_dir)).collect()
    ranking = mask_departure_labels(ranking, extra_cols=("AOBT_3_flt",)).drop(
        ["BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt"], strict=False
    )
    if "month" not in ranking.columns:
        ranking = ranking.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
    dep = ranking.filter(pl.col("PHASE_mvt") == "DEP")
    from prc_challenge.features import add_calendar_features, add_flight_prefix, add_planning_features

    dep = add_planning_features(dep)
    dep = add_calendar_features(dep)
    if "FLIGHT_mvt" in dep.columns:
        dep = add_flight_prefix(dep)
    dep = attach_identity_columns(dep)
    metar = camp.load_metar()
    if metar is not None:
        dep = add_metar_features(dep, metar)
    dep = attach_ranking_bunching(dep, ranking)
    unmatched = unmatched_only(dep)
    unmatched = camp.enrich(unmatched)
    if not camp.smoke and unmatched.height != RANKING_UNMATCHED_N:
        raise RuntimeError(f"ranking unmatched n {unmatched.height} != {RANKING_UNMATCHED_N}")
    return unmatched


def materialize_ranking_arm(
    camp: FrozenCampaign,
    arm: str,
    spec: dict[str, Any],
    fit: pl.DataFrame,
    ranking_u: pl.DataFrame,
    parent: pl.DataFrame,
    template: pl.DataFrame,
) -> dict[str, Any]:
    cats, nums = spec["cats"], spec["nums"]
    cats = [c for c in cats if c in fit.columns and c in ranking_u.columns]
    nums = [c for c in nums if c in fit.columns and c in ranking_u.columns]
    pred, model = fit_predict(fit, ranking_u, cats, nums, iterations=camp.iterations, thread_count=camp.threads)
    model.save_model(str(camp.output / "models" / f"{arm}_ranking.cbm"))
    scored_u = prepare_unmatched_preds(ranking_u, pred)
    submission = overlay_unmatched_submission(parent, scored_u)
    unmatched_ids = ranking_u["MVT_ID_mvt"]
    verify = verify_arm_submission(
        submission,
        template=template,
        parent=parent,
        unmatched_ids=unmatched_ids,
        expected_unmatched_n=ranking_u.height if camp.smoke else RANKING_UNMATCHED_N,
        expected_n=template.height if camp.smoke else RANKING_N,
    )
    dest = camp.output / "submissions" / f"{arm}.parquet"
    submission.write_parquet(dest)
    checksum = file_sha256(dest)
    verify["file_sha256"] = checksum
    verify["path"] = str(dest)
    verify["provenance"] = spec.get("provenance")
    verify["recipe"] = spec.get("recipe")
    verify["arm"] = arm
    if not verify["ok"] and not camp.smoke:
        raise RuntimeError(f"{arm} ranking verification failed: {verify['errors']}")
    atomic_json(camp.output / "verification" / f"{arm}.json", jsonable(verify))
    sidecar = {
        "arm": arm,
        "recipe": spec.get("recipe"),
        "file": str(dest),
        "sha256": checksum,
        "n": int(submission.height),
        "unmatched_n": int(verify["unmatched_n"]),
        "matched_checksum": verify["matched_checksum"],
        "unmatched_checksum": verify["unmatched_checksum"],
        "finite": verify["finite"],
        "provenance": spec.get("provenance"),
        "parent": PARENT_SUB,
        "freeze_applied": False,
        "v4": False,
        "created_utc": now_utc(),
    }
    atomic_json(camp.output / "submissions" / f"{arm}.sidecar.json", jsonable(sidecar))
    return sidecar | {"verify": verify}


def check_quota(camp: FrozenCampaign) -> dict[str, Any]:
    client = s3_client(camp.repo / "minio-s3-credentials.json")
    remote = list_objects(client, TEAM_BUCKET)
    quota = daily_quota(remote)
    quota["bucket"] = TEAM_BUCKET
    atomic_json(camp.output / "uploads" / "quota.json", jsonable(quota))
    return quota


def upload_arm(camp: FrozenCampaign, arm: str, sidecar: dict[str, Any], version: int) -> dict[str, Any]:
    src = Path(sidecar["file"])
    dest = camp.repo / "submissions" / f"{TEAM_NAME}_v{version}.parquet"
    dest.write_bytes(src.read_bytes())
    sidecar_path = dest.with_suffix("").parent / f"{TEAM_NAME}_v{version}.sidecar.json"
    payload = {
        **{k: v for k, v in sidecar.items() if k != "verify"},
        "version": version,
        "campaign_arm": arm,
        "frozen_before_upload": True,
        "leaderboard_at_upload": None,
    }
    sidecar_path.write_text(json.dumps(jsonable(payload), indent=2) + "\n")
    client = s3_client(camp.repo / "minio-s3-credentials.json")
    key = dest.name
    remote = [obj["Key"] for obj in list_objects(client, TEAM_BUCKET)]
    if key in remote:
        raise RuntimeError(f"{key} already on remote; refuse overwrite")
    client.upload_file(str(dest), TEAM_BUCKET, key)
    receipt = {
        "bucket": TEAM_BUCKET,
        "key": key,
        "bytes": dest.stat().st_size,
        "sha256": sidecar["sha256"],
        "uploaded_utc": now_utc(),
        "local_path": str(dest),
        "arm": arm,
        "scorer": "pending",
    }
    receipt_path = camp.repo / "submissions" / f"{TEAM_NAME}_v{version}.upload.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    atomic_json(camp.output / "uploads" / f"{arm}.json", jsonable(receipt))
    return receipt


def fetch_score(client, key: str, dest: Path) -> dict[str, Any] | None:
    result_key = f"{key}_result.json"
    try:
        client.download_file(TEAM_BUCKET, result_key, str(dest))
    except Exception:
        return None
    payload = json.loads(dest.read_text())
    return payload


def wait_for_scores(camp: FrozenCampaign, receipts: list[dict[str, Any]], *, timeout_sec: int = 900) -> dict[str, Any]:
    client = s3_client(camp.repo / "minio-s3-credentials.json")
    deadline = time.time() + timeout_sec
    scores: dict[str, Any] = {}
    pending = {row["arm"]: row for row in receipts}
    while pending and time.time() < deadline:
        for arm, row in list(pending.items()):
            dest = camp.repo / "submissions" / f"{row['key']}_result.json"
            payload = fetch_score(client, row["key"], dest)
            if payload is None:
                continue
            scores[arm] = {
                "key": row["key"],
                "status": payload.get("status"),
                "score": payload.get("score"),
                "used_pairs": payload.get("used_pairs"),
                "sha256": row.get("sha256"),
            }
            pending.pop(arm)
        if pending:
            time.sleep(20)
    for arm, row in pending.items():
        scores[arm] = {"key": row["key"], "status": "pending", "score": None, "sha256": row.get("sha256")}
    atomic_json(camp.output / "uploads" / "ranking_scores.json", jsonable(scores))
    return scores


def run_campaign(camp: FrozenCampaign) -> dict[str, Any]:
    freeze_path = camp.freeze_path
    table = camp.load_table()
    unmatched = camp.enrich(unmatched_only(table))
    d_u = d_frame(unmatched)
    primary_u = primary_frame(unmatched)
    if freeze_path.exists() and not camp.force:
        freeze = json.loads(freeze_path.read_text())
        screen = json.loads((camp.output / "metrics" / "d_screen.json").read_text()) if (camp.output / "metrics" / "d_screen.json").exists() else {"arms": {}, "agent2": consume_agent2_handoff(camp.repo)}
    else:
        screen = screen_d(camp, d_u)
        atomic_json(camp.output / "metrics" / "d_screen.json", jsonable(screen))
        freeze = write_freeze(camp, screen, d_u)
    if freeze.get("leaderboard_inspected"):
        raise RuntimeError("freeze manifest already marked leaderboard-inspected")
    primary_path = camp.output / "metrics" / "primary.json"
    if primary_path.exists() and not camp.force:
        primary_results = json.loads(primary_path.read_text())
    else:
        parent = camp.s4_parent()
        primary_results = {}
        for arm in ARM_ORDER:
            spec = freeze["arms"][arm]
            print(f"primary {arm}", flush=True)
            primary_results[arm] = score_primary_arm(camp, arm, spec, d_u, primary_u, parent)
            if not camp.smoke:
                n_u = primary_results[arm].get("n_unmatched")
                if n_u not in {None, PRIMARY_UNMATCHED_N}:
                    raise RuntimeError(f"{arm} primary unmatched {n_u} != {PRIMARY_UNMATCHED_N}")
        atomic_json(primary_path, jsonable(primary_results))
    mat_path = camp.output / "metrics" / "materialize.json"
    if mat_path.exists() and not camp.force:
        materialize_summary = json.loads(mat_path.read_text())
        ranking_sidecars = {
            arm: json.loads((camp.output / "submissions" / f"{arm}.sidecar.json").read_text())
            for arm in ARM_ORDER
        }
    else:
        ranking_parent = camp.ranking_parent()
        template = camp.template()
        ranking_u = load_ranking_unmatched_features(camp)
        ranking_sidecars = {}
        matched_checksums: dict[str, str] = {}
        for arm in ARM_ORDER:
            spec = freeze["arms"][arm]
            print(f"ranking {arm}", flush=True)
            ranking_sidecars[arm] = materialize_ranking_arm(
                camp, arm, spec, unmatched, ranking_u, ranking_parent, template
            )
            matched_checksums[arm] = ranking_sidecars[arm]["matched_checksum"]
        shared = assert_cross_arm_matched_identical(matched_checksums)
        materialize_summary = {
            "matched_checksum": shared,
            "arms": {k: {kk: vv for kk, vv in row.items() if kk != "verify"} for k, row in ranking_sidecars.items()},
            "freeze_path": str(freeze_path),
            "leaderboard_inspected": False,
        }
        atomic_json(mat_path, jsonable(materialize_summary))
    uploads: list[dict[str, Any]] = []
    scores: dict[str, Any] = {}
    quota = {"skipped": True}
    if not camp.skip_upload and not camp.smoke:
        quota = check_quota(camp)
        remaining = int(quota.get("remaining_today") or 0)
        version = int(quota["next_version"])
        for arm in ARM_ORDER:
            receipt_path = camp.output / "uploads" / f"{arm}.json"
            if receipt_path.exists() and not camp.force:
                uploads.append(json.loads(receipt_path.read_text()))
                continue
            if remaining <= 0:
                break
            print(f"upload {arm} as v{version}", flush=True)
            uploads.append(upload_arm(camp, arm, ranking_sidecars[arm], version))
            version += 1
            remaining -= 1
        if uploads:
            scores = wait_for_scores(camp, uploads)
    return {
        "freeze": freeze,
        "d_screen": screen,
        "primary": primary_results,
        "ranking": materialize_summary,
        "quota": quota,
        "uploads": uploads,
        "ranking_scores": scores,
    }


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_report(camp: FrozenCampaign, result: dict[str, Any]) -> str:
    freeze = result.get("freeze") or {}
    primary = result.get("primary") or {}
    ranking = result.get("ranking") or {}
    screen = result.get("d_screen") or {}
    quota = result.get("quota") or {}
    scores = result.get("ranking_scores") or {}
    uploads = {row.get("arm"): row for row in (result.get("uploads") or [])}
    arms = freeze.get("arms") or {}
    lines = [
        "# Frozen unmatched four-submission campaign",
        "",
        f"**Date:** 21 September 2026  ",
        f"**Starting commit:** `{freeze.get('source_commit')}`  ",
        f"**Protocol:** JAN_JUL_2025 — D = {{2,3,4,5,6,8,9,10}}; January+July scored after freeze  ",
        "**Boundary:** V4 locked. Open-Meteo unused. Ranking scores did not choose recipes, weights, thresholds, or later arms.",
        "",
        "Freeze manifest (written before any new leaderboard look): "
        "`progress-review/2026-09-21-frozen-unmatched-freeze-manifest.json`.",
        "",
        "## Executive summary",
        "",
        f"S3 internally selected `{freeze.get('selection', {}).get('s3_selected')}` from D-only combined variants. "
        f"Matched ranking checksum (all arms): `{ranking.get('matched_checksum')}`.",
        "",
        "| arm | recipe | D unmatched RMSE | JAN+JUL full | JAN+JUL unmatched | ranking SHA256 | public RMSE |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for arm in ARM_ORDER:
        spec = arms.get(arm) or {}
        met = (primary.get(arm) or {}).get("metrics") or {}
        side = (ranking.get("arms") or {}).get(arm) or {}
        score = (scores.get(arm) or {}).get("score")
        lines.append(
            "| {arm} | `{recipe}` | {d} | {full} | {u} | `{sha}` | {pub} |".format(
                arm=arm,
                recipe=spec.get("recipe"),
                d=_num(spec.get("d_unmatched_rmse_seconds")),
                full=_num(met.get("rmse_seconds")),
                u=_num(met.get("unmatched_rmse_seconds")),
                sha=(side.get("sha256") or "")[:16],
                pub=_num(score),
            )
        )
    lines.extend(
        [
            "",
            "## Freeze (before leaderboard)",
            "",
            f"- Frozen UTC: `{freeze.get('frozen_utc')}`",
            f"- `leaderboard_inspected`: `{freeze.get('leaderboard_inspected')}`",
            f"- `adapted_from_ranking`: `{freeze.get('adapted_from_ranking')}`",
            f"- Agent 2 handoff used for unmatched selection: `{((screen.get('agent2') or {}).get('used_for_unmatched_selection'))}`",
            f"- Open-Meteo used: `{freeze.get('open_meteo_used')}`",
            f"- V4 used: `{freeze.get('v4_used')}`",
            "",
            "S3 selection used only blocked D unmatched RMSE among `combined_full` and `combined_no_raw_metar`.",
            "",
            "## Local January+July verification",
            "",
            "| arm | n | unmatched n | JAN full | JUL full | JAN unmatched | JUL unmatched | freeze applied |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for arm in ARM_ORDER:
        met = (primary.get(arm) or {}).get("metrics") or {}
        lines.append(
            "| {arm} | {n} | {nu} | {jan} | {jul} | {janu} | {julu} | {fr} |".format(
                arm=arm,
                n=met.get("n"),
                nu=met.get("n_unmatched"),
                jan=_num(met.get("JAN_2025")),
                jul=_num(met.get("JUL_2025")),
                janu=_num(met.get("JAN_2025_unmatched")),
                julu=_num(met.get("JUL_2025_unmatched")),
                fr=met.get("freeze_applied"),
            )
        )
    lines.extend(
        [
            "",
            "## Ranking files",
            "",
            f"Parent: `{PARENT_SUB}`. Unmatched overlay only. E17/E18 retained. Unmatched parent freeze off.",
            "",
            "| arm | n | unmatched n | matched checksum | unmatched checksum | file SHA256 | finite |",
            "| --- | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for arm in ARM_ORDER:
        side = (ranking.get("arms") or {}).get(arm) or {}
        lines.append(
            "| {arm} | {n} | {nu} | `{mc}` | `{uc}` | `{sha}` | {fin} |".format(
                arm=arm,
                n=side.get("n"),
                nu=side.get("unmatched_n"),
                mc=(side.get("matched_checksum") or "")[:16],
                uc=(side.get("unmatched_checksum") or "")[:16],
                sha=side.get("sha256"),
                fin=side.get("finite"),
            )
        )
    lines.extend(
        [
            "",
            "## Quota and uploads",
            "",
            f"- UTC date: `{quota.get('utc_date')}`",
            f"- Daily cap: `{quota.get('daily_cap', DAILY_CAP)}`; used today `{quota.get('used_today')}`; remaining `{quota.get('remaining_today')}`",
            f"- Next version at check: `{quota.get('next_version')}`",
            "",
        ]
    )
    if uploads:
        lines.append("| arm | key | sha256 | uploaded_utc | public status | public RMSE |")
        lines.append("| --- | --- | --- | --- | --- | ---: |")
        for arm in ARM_ORDER:
            row = uploads.get(arm) or {}
            sc = scores.get(arm) or {}
            if not row:
                continue
            lines.append(
                f"| {arm} | `{row.get('key')}` | `{row.get('sha256')}` | {row.get('uploaded_utc')} | {sc.get('status')} | {_num(sc.get('score'))} |"
            )
    else:
        lines.append("No ranking upload in this run.")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "- Matched: S4 expected-occupancy ranking v12 (bit-identical across arms).",
            "- S0 unmatched: U6 observed-feature CatBoost (IEM METAR already in U6).",
            "- S1 unmatched: U6 + IEM METAR disruption extras (trend, de-ice, RVR/LVP, runway-relative wind, altimeter). Not Open-Meteo.",
            "- S2 unmatched: U6 + OSM parking geometry (ODbL) + OurAirports thresholds/solar.",
            "- S3 unmatched: D-selected combined open-data recipe.",
            "- No AOBT, no OPDI required, no OpenSky state vectors, no V4.",
            "",
            f"Reusable code: `src/prc_challenge/unmatched_recovery/harness.py`, `open_data.py`, `frozen_campaign.py`, `scripts/run_frozen_unmatched_submission_campaign.py`.",
            "",
        ]
    )
    review_json = camp.repo / "progress-review" / "2026-09-21-frozen-unmatched-verification.json"
    if not camp.smoke:
        atomic_json(
            review_json,
            jsonable(
                {
                    "primary": {arm: (primary.get(arm) or {}).get("metrics") for arm in ARM_ORDER},
                    "ranking": ranking,
                    "quota": quota,
                    "uploads": result.get("uploads"),
                    "ranking_scores": scores,
                    "freeze_utc": freeze.get("frozen_utc"),
                    "leaderboard_inspected_at_freeze": freeze.get("leaderboard_inspected"),
                }
            ),
        )
    return "\n".join(lines) + "\n"
