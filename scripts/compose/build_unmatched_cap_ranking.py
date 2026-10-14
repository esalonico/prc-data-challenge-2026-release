# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Cap overlay for v22: v18 with every unmatched row recomputed under the selected expert + cap. Never uploads.

Reads the D decision from artifacts/experiments/Unmatched-polish-20260925/metrics/cap_score.json
(preregistration_cap.json): the arm, its constant C and its expert.

- CAP_A: the v18 ranking expert's predictions (ranking_unmatched_predictions.parquet::expert).
- CAP_B: one refit of the months 1-10 expert with the October rows fixed (400 trees, same recipe),
  predicting the ranking unmatched rows.

Then E_nc_capped = min(E_nc, max(sd,0) + C) -> yhat = p*max(sd,0) + (1-p)*E_nc_capped (3600 s gate,
p = v18's P(copy)) -> E17/E18 + clip -> overlay on v18. Matched rows stay byte-identical.

Outputs (artifacts/experiments/Unmatched-polish-20260925/ranking/):
  cap_unmatched_submission.parquet   full submission shape (the --unmatched input of compose_candidate.py)
  cap_unmatched_overlay.parquet      per unmatched row: expert, capped expert, hit flag, new/old values
  cap_unmatched_overlay.manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
POLISH = MAIN / "artifacts" / "experiments" / "Unmatched-polish-20260925"
OUT = POLISH / "ranking"
TRACKED = ROOT / "progress-review" / "unmatched-polish-20260925" / "ranking"
NONCOPY = MAIN / "artifacts" / "experiments" / "Unmatched-noncopy-20260925" / "ranking"
V18 = MAIN / "submissions" / "unbelievable-snowflake_v18.parquet"
V14_MATCHED_CHECKSUM = "1bf0c6dee84454f4858421d09ad36c719d2e5f032fa1013066225ccc303e3031"
THREADS = 2
GATE_COLS = ("MVT_ID_mvt", "AIRPORT", "nm_unmatched", "type_missing", "takeoff_minus_sched_days", "takeoff_minus_sched_sec")
OCT_COLS = ("flight_prefix", "ADES_mvt", "FLIGHT_RULE_mvt", "ades_missing")

from prc_challenge.io import scan_submitting  # noqa: E402
from prc_challenge.unmatched_recovery.block_lineage import resolve_raw_dir  # noqa: E402
from prc_challenge.unmatched_recovery.final_freeze import S4_FEATURES_REL, FinalFreezeGuard, column_mismatches, working_rows_from_base  # noqa: E402
from prc_challenge.unmatched_recovery.harness import RANKING_N, RANKING_UNMATCHED_N, overlay_unmatched_submission, verify_arm_submission  # noqa: E402
from prc_challenge.unmatched_recovery.noncopy_expert import GATE_SEC, expert_training_mask, mixture_prediction  # noqa: E402
from prc_challenge.unmatched_recovery.polish import cap_expert  # noqa: E402
from prc_challenge.unmatched_recovery.schema import LOCKED_MONTHS, compose_experimental_unmatched  # noqa: E402


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned(base: pl.DataFrame, other: pl.DataFrame, col: str) -> np.ndarray:
    joined = base.select("MVT_ID_mvt").join(other.select("MVT_ID_mvt", col), on="MVT_ID_mvt", how="left", maintain_order="left")
    if joined.height != base.height or joined[col].null_count():
        raise RuntimeError(f"{col}: alignment failed")
    return joined[col].cast(pl.Float64).to_numpy()


def _by_airport(mask: np.ndarray, airport: np.ndarray) -> dict[str, int]:
    return {str(a): int((mask & (airport == a)).sum()) for a in sorted(set(airport.tolist())) if (mask & (airport == a)).sum()}


def octfix_expert_predictions(rank: pl.DataFrame, cats: list[str], nums: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Months 1-10 expert refit once with the October rows fixed; predicts the ranking unmatched rows."""

    from prc_challenge.unmatched_recovery.methods import fit_regressor, predict_regressor

    fit = pl.read_parquet(NONCOPY / "frames" / "expert_fit_m1_10.parquet").sort("_row")
    if set(fit["month"].unique().to_list()) & set(LOCKED_MONTHS):
        raise RuntimeError("locked months in the fit frame")
    october = working_rows_from_base(MAIN / S4_FEATURES_REL, resolve_raw_dir(MAIN), (10,), FinalFreezeGuard((10,)))
    fixed = fit.join(october.select("MVT_ID_mvt", *[pl.col(c).alias(f"{c}__fix") for c in OCT_COLS]), on="MVT_ID_mvt", how="left", maintain_order="left")
    if fixed.filter((pl.col("month") == 10) & pl.col("flight_prefix__fix").is_null()).height:
        raise RuntimeError("October rows missing from the base-cache path")
    for c in OCT_COLS:
        fixed = fixed.with_columns(pl.when(pl.col("month") == 10).then(pl.col(f"{c}__fix").cast(fit[c].dtype)).otherwise(pl.col(c)).alias(c)).drop(f"{c}__fix")
    others = [c for c in dict.fromkeys([*cats, *nums, "TAXITIME_SEC_mvt"]) if c not in OCT_COLS]
    if any(column_mismatches(fixed, fit, others).values()):
        raise RuntimeError("fixed fit frame differs beyond the four October columns")
    lin = pl.read_parquet(NONCOPY / "frames" / "lineage_fit_m1_10.parquet").select("MVT_ID_mvt", "source_class")
    work = fixed.join(lin, on="MVT_ID_mvt", how="left", maintain_order="left")
    if work["source_class"].null_count():
        raise RuntimeError("source class missing on training rows")
    mask = expert_training_mask(work["source_class"].to_numpy(), work["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy(), mode="noncopy_large")
    train = work.filter(pl.Series(mask))
    _log(f"October-fixed expert refit on months 1-10: {train.height} rows (dropped {int((~mask).sum())}), {len(cats) + len(nums)} features, 400 trees")
    model = fit_regressor(train, cats, nums, iterations=400, thread_count=THREADS, l2_leaf_reg=16.0)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    path = OUT / "models" / "expert_noncopy_large_m1_10_octfix.cbm"
    model.save_model(str(path))
    pred = predict_regressor(model, rank, cats, nums)
    october_changed = {c: int((fixed.filter(pl.col("month") == 10)[c].cast(pl.String).fill_null("__NULL__") != fit.filter(pl.col("month") == 10)[c].cast(pl.String).fill_null("__NULL__")).sum()) for c in OCT_COLS}
    return pred, {"path": str(path), "sha256": _sha256(path), "n_train": int(train.height), "n_dropped_large_copies": int((~mask).sum()), "iterations": 400, "october_rows_changed_per_column": october_changed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    TRACKED.mkdir(parents=True, exist_ok=True)
    score = json.loads((POLISH / "metrics" / "cap_score.json").read_text())
    if score.get("decision") != "GO" or not score.get("selected_arm"):
        raise SystemExit("no cap arm passed the pre-registered D gates; no overlay")
    arm, cap, which = score["selected_arm"], float(score["selected_cap_sec"]), score["selected_expert"]
    _log(f"cap overlay for {arm}: expert={which} C={cap:.0f} s")

    rank = pl.read_parquet(NONCOPY / "frames" / "ranking_unmatched.parquet").sort("_row")
    preds = pl.read_parquet(NONCOPY / "ranking_unmatched_predictions.parquet")
    feats = json.loads((NONCOPY / "frames" / "expert_features.json").read_text())
    cats, nums = feats["cats"], feats["nums"]
    if rank.height != RANKING_UNMATCHED_N:
        raise RuntimeError(f"ranking unmatched {rank.height}")
    e_v18 = _aligned(rank, preds, "expert")
    p = _aligned(rank, preds, "p_R-A3b-nodest")
    raw_v18 = _aligned(rank, preds, "raw_R-A3b-nodest")
    dep_v18 = _aligned(rank, preds, "R-A3b-nodest")
    sd = rank["takeoff_minus_sched_sec"].cast(pl.Float64).to_numpy()
    airport = rank["AIRPORT"].to_numpy()
    month = rank["month"].to_numpy()
    gate = rank.select(list(GATE_COLS))

    def deployed(pred: np.ndarray) -> np.ndarray:
        return compose_experimental_unmatched(gate.with_columns(pl.Series("prediction", np.asarray(pred, dtype=np.float64))))["prediction"].to_numpy()

    v18 = pl.read_parquet(V18)
    v18_rows = _aligned(rank, v18, "TAXITIME_SEC_mvt")
    identity = {
        "mixture_vs_raw_R-A3b-nodest": float(np.max(np.abs(mixture_prediction(p, sd, e_v18, gate_sec=GATE_SEC) - raw_v18))),
        "compose_vs_R-A3b-nodest": float(np.max(np.abs(deployed(raw_v18) - dep_v18))),
        "R-A3b-nodest_vs_v18": float(np.max(np.abs(dep_v18 - v18_rows))),
    }
    if max(identity.values()) > 1e-6:
        raise RuntimeError(f"v18 composition does not reproduce: {identity}")

    fit_info: dict[str, Any] = {"expert": which}
    if which == "deployed":
        e = e_v18
        fit_info["model"] = str(NONCOPY / "models" / "expert_noncopy_large_m1_10.cbm")
    elif which == "octfix":
        e, info = octfix_expert_predictions(rank, cats, nums)
        fit_info.update(info)
    else:
        raise SystemExit(f"unknown expert {which}")
    capped, hit = cap_expert(e, sd, cap)
    raw_new = mixture_prediction(p, sd, capped, gate_sec=GATE_SEC)
    composed = deployed(raw_new)
    overridden = np.abs(deployed(np.zeros_like(sd)) - deployed(np.full_like(sd, 1e6))) < 1e-9
    change = composed - v18_rows
    changed = np.abs(change) > 1e-9
    overlay = rank.select("MVT_ID_mvt", "month", "AIRPORT", "takeoff_minus_sched_sec", "type_missing", "takeoff_minus_sched_days").with_columns(
        pl.Series("p_copy", p), pl.Series("expert_v18", e_v18), pl.Series("expert_used", e), pl.Series("expert_capped", capped), pl.Series("cap_hit", hit),
        pl.Series("raw_v18", raw_v18), pl.Series("raw_new", raw_new), pl.Series("v18", v18_rows), pl.Series("prediction", composed), pl.Series("e17_e18_override", overridden),
    )
    overlay_path = OUT / "cap_unmatched_overlay.parquet"
    overlay.write_parquet(overlay_path)
    submission = overlay_unmatched_submission(v18, overlay.select("MVT_ID_mvt", "prediction"))
    template = scan_submitting(MAIN / "data" / "raw").collect()
    verify = verify_arm_submission(submission, template=template, parent=v18, unmatched_ids=rank["MVT_ID_mvt"], expected_unmatched_n=RANKING_UNMATCHED_N, expected_n=RANKING_N)
    if not verify["ok"] or verify["matched_checksum"] != V14_MATCHED_CHECKSUM:
        raise RuntimeError(f"submission verification failed: {verify['errors']} / matched checksum {verify['matched_checksum']}")
    sub_path = OUT / "cap_unmatched_submission.parquet"
    submission.write_parquet(sub_path)

    # Label-free drift of the cap-hit rate: ranking vs Jan/Jul 2025 (confirmation) vs D (OOF).
    drift: dict[str, Any] = {"ranking": {"rate": float(hit.mean()), "n": int(hit.sum()), "n_rows": int(rank.height), "by_airport": _by_airport(hit, airport), "by_month": {str(m): int((hit & (month == m)).sum()) for m in sorted(set(month.tolist()))}}}
    conf = POLISH / "confirm" / "cap_confirmation.json"
    if conf.exists():
        drift["janjul_2025"] = json.loads(conf.read_text()).get("cap_hit_janjul")
    drift["D_oof"] = {"rate": score["touch"][arm]["touched"] / 14066, "n": score["touch"][arm]["touched"], "n_rows": 14066, "by_airport": score["touch"][arm]["touched_by_airport"]}
    manifest = {
        "name": f"cap overlay {arm}: v18 unmatched rows under the {which} expert + cap {cap:.0f} s",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision_source": {"path": str(POLISH / "metrics" / "cap_score.json"), "arm": arm, "cap_sec": cap, "expert": which},
        "files": {
            "submission": {"path": str(sub_path), "sha256": _sha256(sub_path), "rows": int(submission.height), "use": "--unmatched input of scripts/compose/compose_candidate.py (unmatched rows already E17/E18 + clip composed)"},
            "overlay": {"path": str(overlay_path), "sha256": _sha256(overlay_path), "rows": int(overlay.height), "columns": overlay.columns},
        },
        "fit": fit_info,
        "cap_rule": "E_nc_capped = min(E_nc, max(sd,0) + C) where sd is finite; mixture p*max(sd,0) + (1-p)*E_nc_capped with the 3600 s gate; E17/E18 + clip",
        "rows": {
            "unmatched": int(rank.height),
            "cap_touched": int(hit.sum()),
            "cap_touched_by_airport": _by_airport(hit, airport),
            "cap_touched_by_month": drift["ranking"]["by_month"],
            "cap_touched_but_overridden_by_e17_e18": int((hit & overridden).sum()),
            "changed_vs_v18": int(changed.sum()),
            "changed_by_airport": _by_airport(changed, airport),
        },
        "change_vs_v18": {"mean": float(change.mean()), "mean_abs": float(np.abs(change).mean()), "max_abs": float(np.abs(change).max()), "sum_sq": float(np.sum(change**2)), "mean_on_changed": float(change[changed].mean()) if changed.any() else None, "largest_decreases": sorted(change.tolist())[:5]},
        "expert_change_vs_v18_expert": {"rows": int((np.abs(e - e_v18) > 1e-9).sum()), "mean_abs": float(np.abs(e - e_v18).mean()), "max_abs": float(np.abs(e - e_v18).max())},
        "cap_hit_rate_drift": drift,
        "identity_checks_max_abs_diff": identity,
        "v18": {"path": str(V18), "sha256": _sha256(V18)},
        "validator": {"ok": verify["ok"], "errors": verify["errors"], "matched_checksum": verify["matched_checksum"], "matched_checksum_equals_v14": verify["matched_checksum"] == V14_MATCHED_CHECKSUM, "matched_changed": verify["matched_changed"], "unmatched_checksum": verify["unmatched_checksum"]},
        "forbidden_check": {"aobt": False, "lobt": False, "arvt3": False, "dep_block_or_taxi_as_feature": False, "v4_months": False, "adsb": False, "opensky": False, "opdi": False},
        "uploaded": False,
    }
    (OUT / "cap_unmatched_overlay.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (TRACKED / "cap_unmatched_overlay.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _log(f"cap overlay: touched {int(hit.sum())} rows {manifest['rows']['cap_touched_by_airport']}, changed vs v18 {int(changed.sum())}, mean change {change.mean():+.1f} s, max |change| {np.abs(change).max():.0f} s; matched checksum == v14: {verify['matched_checksum'] == V14_MATCHED_CHECKSUM}")


if __name__ == "__main__":
    main()
