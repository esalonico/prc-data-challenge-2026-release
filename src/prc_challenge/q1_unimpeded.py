# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Q1 unimpeded stand/runway baseline plus congestion residual.

The baseline families are bounded and predeclared.  All training-row baseline
values are leave-one-month-out, validation values are fitted on the complete
outer training fold, and the residual CatBoost receives the same legal E08b and
temporal-v2 information as R2.  V4 and ranking output are refused.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from prc_challenge.bundles import optuna_winner_params
from prc_challenge.campaign.agent3_queue import (
    fit_hierarchical_baseline,
    month_blocked_baseline,
    predict_hierarchical_baseline,
)
from prc_challenge.campaign.data import (
    feature_lists,
    load_cache,
    load_parent,
    matched_train_val,
)
from prc_challenge.campaign.data_check import locate
from prc_challenge.campaign.models import fit_regressor, predict_unclipped
from prc_challenge.campaign.runtime import default_config
from prc_challenge.campaign.scoring import score_and_write
from prc_challenge.campaign.temporal_v2 import TEMPORAL_V2_NUMS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "experiments" / "Agent3-open-queue" / "Q1-unimpeded"
REPORT = ROOT / "progress-review" / "2026-09-16-unimpeded-taxi-decomposition-report.md"
CONTEXT = ROOT / "artifacts" / "cache" / "temporal_context_v2.parquet"
PARAMS = Path("artifacts/experiments/E37-optuna/shortlist_t31.json")
BASELINES = {
    "median": {"statistic": "median", "min_count": 40, "shrinkage": 80.0},
    "p10": {
        "statistic": "quantile",
        "quantile": 0.10,
        "min_count": 40,
        "shrinkage": 80.0,
    },
    "p20": {
        "statistic": "quantile",
        "quantile": 0.20,
        "min_count": 40,
        "shrinkage": 80.0,
    },
    "lowload": {
        "statistic": "median",
        "min_count": 30,
        "shrinkage": 80.0,
        "low_load_column": "ctx_dep_past_1800s",
        "low_load_quantile": 0.25,
    },
}


def _existing_metric(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return {"path": str(path.relative_to(ROOT)), "metrics": payload}


def forensic_audit(frame: pl.DataFrame) -> dict[str, object]:
    route_stats = {}
    for col in (
        "AIRPORT",
        "STAND_mvt",
        "RUNWAY_mvt",
        "AIRCRAFT_TYPE_mvt",
        "WK_TBL_CAT_flt",
    ):
        route_stats[col] = {
            "cardinality": int(frame[col].n_unique()),
            "nulls": int(frame[col].null_count()),
            "coverage": float(frame[col].is_not_null().mean()),
        }
    combos = frame.group_by(["AIRPORT", "STAND_mvt", "RUNWAY_mvt"]).len(name="n")
    route_stats["airport_stand_runway"] = {
        "cardinality": combos.height,
        "n_ge_40": int((combos["n"] >= 40).sum()),
        "row_coverage_n_ge_40": float(
            combos.filter(pl.col("n") >= 40)["n"].sum() / frame.height
        ),
    }
    prior_paths = [
        ROOT / "artifacts" / "experiments" / "E02-unimpeded_V1" / "metrics.json",
        ROOT / "artifacts" / "experiments" / "CPU-05-route-queue" / "metrics.json",
        ROOT / "artifacts" / "experiments" / "CPU-12-geometry" / "metrics.json",
        ROOT / "artifacts" / "experiments" / "R2-temporal-context-v2" / "metrics.json",
    ]
    return {
        "route_fields": route_stats,
        "prior_experiments": [
            m for p in prior_paths if (m := _existing_metric(p)) is not None
        ],
        "novelty": (
            "E02 added a single airport/stand q10 feature to an older direct model. CPU-05 tested a route/queue "
            "campaign path with known headline-selection and teacher limitations. Q1 freezes four fold-safe route "
            "summaries, uses p10 as the physical baseline under the current EUROCONTROL method, and fits the "
            "additional-taxi residual with the independently validated R2 context and outer-view refits."
        ),
    }


def attach_baselines(
    train: pl.DataFrame, val: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    train_out = train
    val_out = val
    diagnostics: dict[str, object] = {}
    for name, kwargs in BASELINES.items():
        col = f"unimpeded_{name}"
        blocked = month_blocked_baseline(
            train, fit_kwargs=kwargs, output_column=col
        ).select("MVT_ID_mvt", col, f"{col}_support", f"{col}_available")
        fitted = fit_hierarchical_baseline(train, **kwargs)
        predicted = predict_hierarchical_baseline(
            val, fitted, output_column=col
        ).select("MVT_ID_mvt", col, f"{col}_support", f"{col}_available")
        train_out = train_out.join(blocked, on="MVT_ID_mvt", how="left", validate="1:1")
        val_out = val_out.join(predicted, on="MVT_ID_mvt", how="left", validate="1:1")
        y = val_out["TAXITIME_SEC_mvt"].to_numpy()
        p = val_out[col].to_numpy()
        diagnostics[name] = {
            "global_value": fitted.global_value,
            "low_load_cutoff": fitted.load_cutoff,
            "validation_baseline_rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "validation_baseline_mae": float(np.mean(np.abs(y - p))),
            "support_nonzero": float((val_out[f"{col}_support"] > 0).mean()),
        }
    return train_out, val_out, diagnostics


def baseline_columns() -> list[str]:
    return [
        f"unimpeded_{name}{suffix}"
        for name in BASELINES
        for suffix in ("", "_support", "_available")
    ]


def cache_matches(frame: pl.DataFrame, cached: pl.DataFrame) -> bool:
    """Require a one-to-one, exact ID match before reusing baseline caches."""

    required = {"MVT_ID_mvt", *baseline_columns()}
    if not required.issubset(cached.columns):
        return False
    if cached.height != frame.height:
        return False
    if cached["MVT_ID_mvt"].n_unique() != cached.height:
        return False
    if frame["MVT_ID_mvt"].n_unique() != frame.height:
        return False
    missing = frame.select("MVT_ID_mvt").join(
        cached.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti"
    )
    extra = cached.select("MVT_ID_mvt").join(
        frame.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti"
    )
    return missing.is_empty() and extra.is_empty()


def load_or_attach_baselines(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    view: str,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    train_path = OUT / f"baseline_train_{view}.parquet"
    val_path = OUT / f"baseline_val_{view}.parquet"
    metrics_path = OUT / f"baseline_metrics_{view}.json"
    if train_path.exists() and val_path.exists() and metrics_path.exists():
        train_base = pl.read_parquet(train_path)
        val_base = pl.read_parquet(val_path)
        if cache_matches(train, train_base) and cache_matches(val, val_base):
            return (
                train.join(train_base, on="MVT_ID_mvt", how="left", validate="1:1"),
                val.join(val_base, on="MVT_ID_mvt", how="left", validate="1:1"),
                json.loads(metrics_path.read_text()),
            )
    train_out, val_out, diagnostics = attach_baselines(train, val)
    keep = ["MVT_ID_mvt", *baseline_columns()]
    train_out.select(keep).write_parquet(train_path)
    val_out.select(keep).write_parquet(val_path)
    metrics_path.write_text(json.dumps(diagnostics, indent=2, default=str) + "\n")
    return train_out, val_out, diagnostics


def _r2_predictions(view: str) -> pl.DataFrame:
    path = (
        ROOT
        / "artifacts"
        / "experiments"
        / "R2-temporal-context-v2"
        / f"predictions_{view}.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"missing R2 parent predictions: {path}")
    return pl.read_parquet(path)


def write_report(audit: dict[str, object], results: dict[str, object]) -> None:
    fields = audit["route_fields"]
    lines = [
        "# Unimpeded taxi decomposition report",
        "",
        "**Date:** 2026-09-16  ",
        "**Phase:** Q1  ",
        "**Boundary:** all route statistics are outer-training-fold-only; training features are leave-one-month-out. V4 and ranking output are refused.",
        "",
        "## Forensic audit",
        "",
        "| Field | Cardinality | Nulls | Coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for col in (
        "AIRPORT",
        "STAND_mvt",
        "RUNWAY_mvt",
        "AIRCRAFT_TYPE_mvt",
        "WK_TBL_CAT_flt",
    ):
        row = fields[col]
        lines.append(
            f"| `{col}` | {row['cardinality']:,} | {row['nulls']:,} | {row['coverage']:.1%} |"
        )
    combo = fields["airport_stand_runway"]
    lines.extend(
        [
            "",
            f"There are **{combo['cardinality']:,}** airport/stand/runway cells. **{combo['n_ge_40']:,}** have at least 40 observations and cover **{combo['row_coverage_n_ge_40']:.1%}** of rows.",
            "",
            audit["novelty"],
            "",
            "## Results",
            "",
        ]
    )
    baseline_screen = results.get("baseline_screen") or {}
    if baseline_screen:
        lines.extend(
            [
                "### Fold-safe baseline screen",
                "",
                "| View | Family | Validation RMSE | Validation MAE | Global value |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for view in ("V1", "V2", "V3"):
            for family in ("median", "p10", "p20", "lowload"):
                row = baseline_screen.get(view, {}).get(family)
                if row is None:
                    continue
                lines.append(
                    f"| {view} | {family} | {row['validation_baseline_rmse']:.2f} | "
                    f"{row['validation_baseline_mae']:.2f} | {row['global_value']:.1f} |"
                )
        lines.extend(
            [
                "",
                "The median is the strongest standalone baseline on every view. The p10 remains the predeclared residual anchor because it matches the operational unimpeded-taxi interpretation, not because it is a competitive standalone predictor.",
                "",
            ]
        )
    views = results.get("views") or {}
    if views:
        lines.extend(
            [
                "| View | Candidate RMSE | R2 RMSE | Improvement seconds | Residual correlation |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for view in ("V1", "V2", "V3"):
            if view not in views:
                continue
            row = views[view]
            lines.append(
                f"| {view} | {row['rmse_seconds']:.2f} | {row['r2_rmse_seconds']:.2f} | {row['improvement_seconds']:+.2f} | {row['residual_correlation']:.4f} |"
            )
        decision = (
            f"**Decision: {results.get('decision', 'pending')}.** "
            f"{results.get('reason', '')}"
        ).rstrip()
        lines.extend(["", decision])
    else:
        lines.append("Heavy residual fits are pending the single local trainer slot.")
    lines.extend(
        [
            "",
            "Machine-readable artifacts are under `artifacts/experiments/Agent3-open-queue/Q1-unimpeded/`.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", default="V1,V2,V3")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2210)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--prepare-baselines", action="store_true")
    args = parser.parse_args()
    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    if "V4" in views:
        raise SystemExit("V4 is locked and refused")

    cfg = default_config(ROOT)
    cfg.thread_count = max(1, args.threads)
    base = load_cache(cfg)
    if not CONTEXT.exists():
        raise FileNotFoundError(f"missing R2 context {CONTEXT}")
    context = pl.read_parquet(CONTEXT)
    frame = base.join(context, on="MVT_ID_mvt", how="left", validate="1:1")
    audit = forensic_audit(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "forensic_audit.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n"
    )
    results_path = OUT / "metrics.json"
    results = (
        json.loads(results_path.read_text())
        if results_path.exists()
        else {"status": "running", "views": {}}
    )
    write_report(audit, results)
    if args.audit_only:
        print(json.dumps(audit["route_fields"], indent=2))
        return

    if args.prepare_baselines:
        prepared = {}
        for view in views:
            train, val = matched_train_val(frame, view)
            _, _, prepared[view] = load_or_attach_baselines(train, val, view=view)
            print(f"{view}: prepared fold-safe baseline families", flush=True)
        results["baseline_screen"] = prepared
        results_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
        write_report(audit, results)
        return

    params_path = locate(PARAMS, cfg)
    if params_path is None:
        raise FileNotFoundError(PARAMS)
    params = optuna_winner_params(
        json.loads(params_path.read_text()), thread_count=cfg.thread_count
    )
    params.pop("use_best_model", None)
    params["iterations"] = int(args.iterations)
    cats, nums = feature_lists("e08b")
    baseline_nums = baseline_columns()
    residual_nums = list(dict.fromkeys([*nums, *TEMPORAL_V2_NUMS, *baseline_nums]))
    results.update(
        {
            "status": "running",
            "phase": "Q1",
            "baseline_anchor": "p10",
            "params": params,
            "views": dict(results.get("views") or {}),
        }
    )
    started_all = time.perf_counter()
    for view in views:
        started = time.perf_counter()
        train, val = matched_train_val(frame, view)
        train, val, baseline_metrics = load_or_attach_baselines(train, val, view=view)
        train = train.with_columns(
            (pl.col("TAXITIME_SEC_mvt") - pl.col("unimpeded_p10")).alias(
                "_additional_taxi"
            )
        )
        model = fit_regressor(
            cfg, train, cats, residual_nums, label="_additional_taxi", **params
        )
        residual = predict_unclipped(model, val, cats, residual_nums)
        pred = np.clip(val["unimpeded_p10"].to_numpy() + residual, 0, None)
        r2 = _r2_predictions(view)
        scored = score_and_write(
            val,
            pred,
            OUT,
            view=view,
            parent=load_parent(cfg, view),
            base=r2,
            tag="unimpeded_residual",
        )
        vs = scored["vs_base"]
        model_path = OUT / f"model_{view}.cbm"
        model.save_model(str(model_path))
        results["views"][view] = {
            "rmse_seconds": scored["rmse_seconds"],
            "r2_rmse_seconds": vs["base_rmse_seconds"],
            "improvement_seconds": -float(vs["delta_rmse_vs_base"]),
            "residual_correlation": vs["resid_corr_vs_base"],
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "baseline_metrics": baseline_metrics,
            "model_path": str(model_path),
        }
        results["runtime_seconds"] = round(time.perf_counter() - started_all, 3)
        results_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
        write_report(audit, results)
        print(
            f"{view}: rmse={scored['rmse_seconds']:.3f} improvement={results['views'][view]['improvement_seconds']:.3f}",
            flush=True,
        )

    v = results["views"]
    robust = (
        all(name in v for name in ("V1", "V2", "V3"))
        and v["V1"]["improvement_seconds"] > 0
        and v["V2"]["improvement_seconds"] > 0
        and v["V3"]["improvement_seconds"] >= -1.0
    )
    specialist = any(
        name in v
        and abs(v[name]["improvement_seconds"]) <= 3
        and v[name]["residual_correlation"] < 0.98
        for name in ("V1", "V2", "V3")
    )
    results["status"] = "completed"
    results["decision"] = (
        "GLOBAL_GO" if robust else ("ENSEMBLE_ONLY_GO" if specialist else "PARK")
    )
    results["reason"] = (
        "Predeclared global and complementarity gates applied independently of later queue phases."
    )
    results_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    write_report(audit, results)


if __name__ == "__main__":
    main()
