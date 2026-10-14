# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Reproducible S4 expected-occupancy trainer (D-fit replay and all-year ranking refit).

Stages (all resumable; bulky outputs go to
``<source-root>/artifacts/experiments/S4-allyear-rebuild-20260925/``):

  build-caches   per-period label-free feature caches for 2025-01..12 and the two
                 ranking months (2026-01, 2026-07) + temporal-v2 scopes + manifest.
  verify-caches  compare months 1-10 caches with the historical S4 inputs.
  fit            assemble the S4 stack for ``--train-months`` and ``--eval``
                 (``janjul`` = D-fit replay scored on Jan+Jul 2025,
                 ``ranking`` = all-year refit predicting the ranking file,
                 ``novdec`` = D-month references, Nov/Dec 2025 eval rows; with
                 ``--assemble-only`` only, for scoring stored D-fit models on V4), fit the
                 2210-tree t31 CatBoost on the matched-only cut, compose, score,
                 and write a manifest.  ``--extra-features PATH`` (repeatable)
                 appends parquet feature blocks keyed by ``MVT_ID_mvt``.
  compose-s1     overlay the frozen S1 unmatched model on rebuilt matched ranking
                 rows with the frozen-campaign harness.  Never uploads.

Examples::

  PYTHONPATH=src python scripts/legs/s4_allyear.py --stage build-caches
  PYTHONPATH=src python scripts/legs/s4_allyear.py --stage fit --train-months D --eval janjul --tag dfit_replay
  PYTHONPATH=src python scripts/legs/s4_allyear.py --stage fit --train-months 1-12 --eval ranking --tag allyear
  PYTHONPATH=src python scripts/legs/s4_allyear.py --stage compose-s1 --tag allyear

Fit target (``--target-*``, see ``s4_allyear.TargetSpec``): the default is the
historical residual ``y - unimpeded_p10``. The AOBT delta reparameterization
(model learns ``BLOCK - AOBT``; needs the AOBT block via ``--extra-features``)::

  ... --stage fit --extra-features <aobt block> --target-offset-col takeoff_minus_aobt_sec \
      --target-sign -1 --target-offset-fill-col unimpeded_p10 --target-winsor 7200
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prc_challenge import s4_allyear as s4
from prc_challenge.bundles import matched_booster_frame
from prc_challenge.operational_v2 import (
    attach_cross_fitted_arrival_state,
    build_expected_occupancy_features,
    fit_duration_reference,
    learned_feature_cache_identity,
    load_learned_cache_or_reject,
    write_learned_cache_identity,
)
from prc_challenge.s4_allyear import (
    ALL_PERIODS,
    CacheLayout,
    Period,
    RANKING_PERIODS,
    TRAIN_PERIODS,
    atomic_json,
    atomic_parquet,
    json_sha256,
    now_utc,
    payload_sha256,
    peak_rss_gb,
    sha256_file,
)

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
D_SCOPE_PERIODS = tuple(Period(2025, m) for m in range(1, 11))
HISTORICAL_E08B_CACHE = Path("artifacts/cache/e08b_e34_months_1_10.parquet")
HISTORICAL_TEMPORAL_CACHE = Path("artifacts/cache/temporal_context_v2.parquet")
HISTORICAL_SIDECARS = Path("artifacts/experiments/Wave2-agent2/sidecars")
HISTORICAL_EVENT_MANIFEST = Path("artifacts/experiments/Overnight-20260919-agent2/event_manifest.json")
T31_PARAMS = Path("artifacts/experiments/E37-optuna/shortlist_t31.json")
E33B_PARENT = Path("artifacts/experiments/E33b-matched-splice_V1/val_predictions.parquet")
V12_FILE = Path("submissions/unbelievable-snowflake_v12.parquet")
V15_FILE = Path("submissions/unbelievable-snowflake_v15.parquet")
S1_MODEL = Path("artifacts/experiments/Frozen-unmatched-20260921/models/S1_ranking.cbm")
S1_FREEZE = Path("artifacts/experiments/Frozen-unmatched-20260921/freeze/freeze_manifest.json")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message} (peak_rss={peak_rss_gb():.2f} GB)", flush=True)


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=SCRIPT_ROOT, text=True).strip()
        except Exception:
            return ""

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty_tracked": bool(run("status", "--porcelain", "--untracked-files=no")),
        "module_sha256": sha256_file(Path(s4.__file__)),
        "script_sha256": sha256_file(Path(__file__)),
    }


def parse_months(value: str) -> tuple[int, ...]:
    text = value.strip().upper()
    if text == "D":
        return s4.D_MONTHS
    if text in {"ALL", "ALLYEAR"}:
        return s4.ALL_2025_MONTHS
    months: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            months.extend(range(int(lo), int(hi) + 1))
        elif part:
            months.append(int(part))
    return tuple(sorted(set(months)))


# ---------------------------------------------------------------------------
# Cache building


def _file_entry(path: Path, out: Path, *, block: str, period: Period | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    schema = pl.read_parquet_schema(path)
    rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    entry = {
        "block": block,
        "period": period.key if period else None,
        "path": str(path.relative_to(out)),
        "sha256": sha256_file(path),
        "rows": int(rows),
        "columns": {k: str(v) for k, v in schema.items()},
    }
    if extra:
        entry.update(extra)
    return entry


def ensure_base_caches(layout: CacheLayout, src: Path, periods: list[Period], *, force: bool) -> None:
    todo = [
        p
        for p in periods
        if force or not layout.base(p).exists() or (not p.is_ranking and not layout.labels(p).exists())
    ]
    if not todo:
        return
    from prc_challenge.weather import load_metar

    log(f"base caches: building {[p.key for p in todo]}")
    metar = load_metar(src / "data" / "external" / "iem_metar")
    deps = s4._dep_rows_by_utc_month(src / "data" / "raw", todo)
    for period in todo:
        started = time.perf_counter()
        features, labels = s4.build_base_period(deps.pop(period), metar)
        atomic_parquet(features, layout.base(period))
        if not period.is_ranking:
            atomic_parquet(labels, layout.labels(period))
        log(f"base {period.key}: rows={features.height} cols={features.shape[1]} t={time.perf_counter() - started:.1f}s")
    del deps, metar
    gc.collect()


def ensure_isolated_caches(layout: CacheLayout, src: Path, periods: list[Period], *, force: bool) -> None:
    raw_dir = src / "data" / "raw"
    for period in periods:
        if force or not layout.native(period).exists() or not layout.arrivals(period).exists():
            started = time.perf_counter()
            native, arrivals = s4.build_native_period(raw_dir, period)
            atomic_parquet(native, layout.native(period))
            atomic_parquet(arrivals, layout.arrivals(period))
            log(f"native {period.key}: rows={native.height} arrivals={arrivals.height} t={time.perf_counter() - started:.1f}s")
        if force or not layout.events(period).exists():
            started = time.perf_counter()
            events = s4.build_events_period(raw_dir, period)
            atomic_parquet(events, layout.events(period))
            log(f"events {period.key}: rows={events.height} t={time.perf_counter() - started:.1f}s")


def ensure_temporal_isocap(layout: CacheLayout, src: Path, periods: list[Period], *, force: bool) -> str:
    """``isocap_v1`` (SA-EMA-X1 item 1): month-isolated temporal-v2, lookback gaps horizon-censored."""

    scope = "isocap_v1"
    for period in sorted(periods):
        if not force and layout.temporal(scope, period).exists():
            continue
        started = time.perf_counter()
        ensure_base_caches(layout, src, [period], force=False)
        dep = pl.read_parquet(layout.base(period))
        arr = s4.arr_context_rows(src / "data" / "raw", [period])
        atomic_parquet(s4.build_temporal_period_isocap(dep, arr, period), layout.temporal(scope, period))
        log(f"temporal isocap_v1 {period.key}: dep={dep.height} arr={arr.height} t={time.perf_counter() - started:.1f}s")
    from prc_challenge import lookback_horizon as lh

    atomic_json(
        layout.features / "temporal_v2" / scope / "scope.json",
        {"scope": scope, "caps": lh.HORIZON_CAPS_V1, "semantics": lh.__doc__},
    )
    return scope


def ensure_temporal_scope(layout: CacheLayout, src: Path, periods: list[Period], *, force: bool, mode: str = "contiguous") -> str:
    if mode == "isocap_v1":
        return ensure_temporal_isocap(layout, src, periods, force=force)
    scope = s4.scope_id(periods)
    if not force and all(layout.temporal(scope, p).exists() for p in periods):
        return scope
    started = time.perf_counter()
    ensure_base_caches(layout, src, periods, force=False)
    dep = pl.concat([pl.read_parquet(layout.base(p)) for p in sorted(periods)], how="vertical_relaxed")
    arr = s4.arr_context_rows(src / "data" / "raw", sorted(periods))
    ctx = s4.build_temporal_scope(dep, arr)
    for period in sorted(periods):
        ids = pl.read_parquet(layout.base(period), columns=["MVT_ID_mvt"])
        part = ids.join(ctx, on="MVT_ID_mvt", how="left", validate="1:1", maintain_order="left")
        atomic_parquet(part, layout.temporal(scope, period))
    log(f"temporal scope {scope}: dep={dep.height} arr={arr.height} t={time.perf_counter() - started:.1f}s")
    atomic_json(
        layout.features / "temporal_v2" / scope / "scope.json",
        {
            "scope": scope,
            "periods": [p.key for p in sorted(periods)],
            "dep_rows": dep.height,
            "arr_rows": arr.height,
            "semantics": (
                "build_temporal_context_v2 over one contiguous table of every DEP row (matched and unmatched) "
                "and every challenge-airport ARR row of these periods; identical to the historical "
                "temporal_context_v2.parquet semantics (months 1-10) when the scope is 2025-01..2025-10."
            ),
        },
    )
    return scope


def write_cache_manifest(layout: CacheLayout, src: Path) -> dict[str, Any]:
    out = layout.root
    entries: list[dict[str, Any]] = []
    for period in ALL_PERIODS:
        for block, path in (
            ("base", layout.base(period)),
            ("labels", layout.labels(period)),
            ("native_state", layout.native(period)),
            ("native_arrivals", layout.arrivals(period)),
            ("events", layout.events(period)),
        ):
            if path.exists():
                entries.append(_file_entry(path, out, block=block, period=period))
    temporal_root = layout.features / "temporal_v2"
    if temporal_root.exists():
        for scope_dir in sorted(p for p in temporal_root.iterdir() if p.is_dir()):
            for path in sorted(scope_dir.glob("temporal_v2_*.parquet")):
                period = Period.parse(path.stem.removeprefix("temporal_v2_"))
                entries.append(_file_entry(path, out, block="temporal_v2", period=period, extra={"scope": scope_dir.name}))
    from prc_challenge.bundles import e34_feature_lists
    from prc_challenge.campaign.temporal_v2 import TEMPORAL_V2_NUMS
    from prc_challenge.operational_v2 import NATIVE_STATE_FEATURES

    cats, nums = e34_feature_lists()
    model_cats, model_nums = s4.s4_feature_lists()
    manifest = {
        "schema_version": "s4-allyear-feature-cache-v1",
        "created_utc": now_utc(),
        "source_root": str(src),
        "raw_inputs_sha256": s4.raw_input_hashes(src / "data" / "raw"),
        "metar": s4.metar_input_hash(src / "data" / "external" / "iem_metar"),
        "git": git_state(),
        "key": "MVT_ID_mvt",
        "periods": [p.key for p in ALL_PERIODS],
        "blocks": {
            "base": {
                "isolation": "UTC month of MVT_TIME_UTC_mvt; delayed-bunching context = same month DEP rows",
                "builder": "prc_challenge.s4_allyear.build_base_period (= scripts/prepare_model_cache.py semantics)",
                "label_free": True,
                "e08b_cats": cats,
                "e34_nums": nums,
            },
            "labels": {"columns": ["MVT_ID_mvt", "TAXITIME_SEC_mvt", "month"], "periods": "2025 only"},
            "temporal_v2": {
                "isolation": "NOT month-isolated historically: one contiguous build per scope; stored sliced per period",
                "scopes": {
                    "2025-01_to_2025-10": "D-fit replay (identical to historical artifacts/cache/temporal_context_v2.parquet)",
                    "2025-01_to_2025-12__2026-01__2026-07": "all-year refit + ranking",
                },
                "columns": list(TEMPORAL_V2_NUMS),
            },
            "native_state": {
                "isolation": "raw training file per 2025 month; UTC month for the ranking file",
                "builder": "operational_v2.build_native_completed_arrival_sidecar (no OPDI rotation columns)",
                "model_columns": list(NATIVE_STATE_FEATURES),
            },
            "native_arrivals": {"note": "completed ARR taxi-in observations used only to fit the fold-safe taxi-in normaliser"},
            "events": {
                "isolation": "raw training file per 2025 month; UTC month for the ranking file",
                "builder": "operational_v2.build_event_table on the permitted projection (no DEP taxi/block/AOBT)",
            },
        },
        "fit_dependent_references": {
            "q1_unimpeded": "prc_challenge.q1_unimpeded.attach_baselines: train rows leave-one-month-out inside train months (matched-booster rows); eval rows fitted on all train months",
            "native_arrival_normaliser": "attach_cross_fitted_arrival_state: train month m fitted on other train months' arrivals; eval fitted on all train months",
            "s4_duration_reference": "fit_duration_reference on every labelled DEP row (matched and unmatched) of the reference months; train month m uses train months without m; eval periods use all train months",
        },
        "model_features": {"cats": model_cats, "nums": model_nums, "feature_schema_sha256": json_sha256([model_cats, model_nums])},
        "files": entries,
    }
    atomic_json(layout.manifest, manifest)
    return manifest


def stage_build_caches(args: argparse.Namespace, layout: CacheLayout, src: Path) -> None:
    started = time.perf_counter()
    periods = [Period.parse(k) for k in args.periods.split(",")] if args.periods else list(ALL_PERIODS)
    ensure_base_caches(layout, src, periods, force=args.force)
    ensure_isolated_caches(layout, src, periods, force=args.force)
    if not args.periods:
        ensure_temporal_scope(layout, src, list(D_SCOPE_PERIODS), force=args.force)
        ensure_temporal_scope(layout, src, list(ALL_PERIODS), force=args.force)
    manifest = write_cache_manifest(layout, src)
    log(f"build-caches done: files={len(manifest['files'])} t={time.perf_counter() - started:.1f}s manifest={layout.manifest}")


# ---------------------------------------------------------------------------
# Cache verification against the historical S4 inputs


def _compare_frames(mine: pl.DataFrame, theirs: pl.DataFrame, columns: list[str]) -> dict[str, Any]:
    joined = mine.select("MVT_ID_mvt", *columns).join(
        theirs.select("MVT_ID_mvt", *[pl.col(c).alias(f"{c}__hist") for c in columns]),
        on="MVT_ID_mvt",
        how="full",
        coalesce=True,
    )
    result: dict[str, Any] = {
        "rows_mine": mine.height,
        "rows_hist": theirs.height,
        "rows_joined": joined.height,
        "ids_only_mine": int(joined.filter(pl.col(f"{columns[0]}__hist").is_null() & pl.col(columns[0]).is_not_null()).height) if columns else None,
        "columns": {},
    }
    ids_mine = set(mine["MVT_ID_mvt"].to_list())
    ids_hist = set(theirs["MVT_ID_mvt"].to_list())
    result["ids_only_mine"] = len(ids_mine - ids_hist)
    result["ids_only_hist"] = len(ids_hist - ids_mine)
    both = joined.filter(pl.col("MVT_ID_mvt").is_in(list(ids_mine & ids_hist)))
    mismatched = 0
    for column in columns:
        a, b = both[column], both[f"{column}__hist"]
        if a.dtype.is_temporal():
            a = a.dt.epoch("us")
        if b.dtype.is_temporal():
            b = b.dt.epoch("us")
        if a.dtype == pl.String or b.dtype == pl.String:
            diff = int((a.cast(pl.String).fill_null("\x00") != b.cast(pl.String).fill_null("\x00")).sum())
            result["columns"][column] = {"n_diff": diff}
        else:
            av = a.cast(pl.Float64).to_numpy()
            bv = b.cast(pl.Float64).to_numpy()
            both_nan = np.isnan(av) & np.isnan(bv)
            neq = ~both_nan & ~(av == bv)
            max_abs = float(np.nanmax(np.abs(av - bv))) if np.isfinite(av - bv).any() else 0.0
            result["columns"][column] = {"n_diff": int(neq.sum()), "max_abs_diff": max_abs}
        mismatched += result["columns"][column]["n_diff"]
    result["total_cell_diffs"] = int(mismatched)
    result["identical"] = bool(mismatched == 0 and result["ids_only_mine"] == 0 and result["ids_only_hist"] == 0)
    return result


def stage_verify_caches(args: argparse.Namespace, layout: CacheLayout, src: Path) -> None:
    report: dict[str, Any] = {"created_utc": now_utc(), "historical_sources": {}}
    months = list(D_SCOPE_PERIODS)
    # base vs e08b cache
    hist = pl.read_parquet(src / HISTORICAL_E08B_CACHE)
    report["historical_sources"]["e08b_cache"] = {"path": str(HISTORICAL_E08B_CACHE), "sha256": sha256_file(src / HISTORICAL_E08B_CACHE)}
    mine = pl.concat([pl.read_parquet(layout.base(p)) for p in months], how="vertical_relaxed")
    labels = pl.concat([pl.read_parquet(layout.labels(p)) for p in months], how="vertical_relaxed")
    mine = mine.join(labels.select("MVT_ID_mvt", "TAXITIME_SEC_mvt"), on="MVT_ID_mvt", how="left")
    cols = [c for c in hist.columns if c != "MVT_ID_mvt" and c in mine.columns]
    report["base_vs_e08b_cache"] = _compare_frames(mine, hist, cols)
    report["base_vs_e08b_cache"]["columns_missing_in_mine"] = [c for c in hist.columns if c not in mine.columns]
    del hist
    gc.collect()
    # temporal
    scope = s4.scope_id(months)
    hist = pl.read_parquet(src / HISTORICAL_TEMPORAL_CACHE)
    mine = pl.concat([pl.read_parquet(layout.temporal(scope, p)) for p in months], how="vertical_relaxed")
    cols = [c for c in hist.columns if c != "MVT_ID_mvt"]
    report["temporal_d_scope_vs_historical"] = _compare_frames(mine, hist, cols)
    allscope = s4.scope_id(ALL_PERIODS)
    if all(layout.temporal(allscope, p).exists() for p in months):
        mine_all = pl.concat([pl.read_parquet(layout.temporal(allscope, p)) for p in months], how="vertical_relaxed")
        report["temporal_allyear_scope_vs_historical_months_1_10"] = _compare_frames(mine_all, hist, cols)
    del hist
    gc.collect()
    # native sidecars
    native_report: dict[str, Any] = {}
    for period in months:
        hist_path = src / HISTORICAL_SIDECARS / f"operational_features_2025{period.month:02d}.parquet"
        hist = pl.read_parquet(hist_path)
        mine = pl.read_parquet(layout.native(period))
        cols = [c for c in mine.columns if c != "MVT_ID_mvt" and c in hist.columns]
        cmp = _compare_frames(mine, hist, cols)
        hist_arr = pl.read_parquet(src / HISTORICAL_SIDECARS / f"completed_arrivals_2025{period.month:02d}.parquet")
        mine_arr = pl.read_parquet(layout.arrivals(period))
        cmp["arrivals_equal"] = bool(mine_arr.equals(hist_arr))
        cmp["columns"] = {k: v for k, v in cmp["columns"].items() if v["n_diff"]}  # compact
        native_report[period.key] = cmp
    report["native_vs_wave2_sidecars"] = native_report
    # events
    manifest = json.loads((src / HISTORICAL_EVENT_MANIFEST).read_text())
    events_report = {}
    for row in manifest:
        period = Period(2025, int(row["month"]))
        mine_sha = sha256_file(layout.events(period))
        events_report[period.key] = {"historical_sha256": row["sha256"], "rebuilt_sha256": mine_sha, "byte_identical": mine_sha == row["sha256"], "rows": row["rows"]}
    report["events_vs_historical_manifest"] = events_report
    report["summary"] = {
        "base_identical": report["base_vs_e08b_cache"]["identical"],
        "temporal_d_scope_identical": report["temporal_d_scope_vs_historical"]["identical"],
        "native_identical_all_months": all(v["identical"] and v["arrivals_equal"] for v in native_report.values()),
        "events_byte_identical_all_months": all(v["byte_identical"] for v in events_report.values()),
    }
    atomic_json(layout.root / "manifests" / "cache_verification.json", report)
    print(json.dumps(report["summary"], indent=2))


# ---------------------------------------------------------------------------
# Fit-dependent S4 learned features


def _s4_builder_sha256() -> str:
    return json_sha256(inspect.getsource(build_expected_occupancy_features))


def s4_learned_for(
    layout: CacheLayout,
    period: Period,
    reference_months: tuple[int, ...],
    label_frames: dict[int, pl.DataFrame],
    ref_cache: dict[tuple[int, ...], Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if reference_months not in ref_cache:
        ref = fit_duration_reference(pl.concat([label_frames[m] for m in reference_months], how="vertical"))
        ref_cache[reference_months] = (ref, s4.reference_fingerprint(ref))
    reference, fingerprint = ref_cache[reference_months]
    path = layout.s4_learned(period, fingerprint)
    identity = learned_feature_cache_identity(
        model=s4.RECIPE,
        split_id=f"ref-{fingerprint[:16]}",
        source_sha256=sha256_file(layout.events(period)),
        feature_config={
            "reference_fingerprint": fingerprint,
            "reference_months": list(reference_months),
            "builder_sha256": _s4_builder_sha256(),
            "columns": list(s4.S4_EXPECTED_COLUMNS),
        },
        month=period.month,
    )
    identity_path = path.with_suffix(".identity.json")
    if path.exists() and identity_path.exists():
        features = load_learned_cache_or_reject(path, identity)
    else:
        features = s4.build_s4_learned(pl.read_parquet(layout.events(period)), reference)
        atomic_parquet(features, path)
        write_learned_cache_identity(identity_path, identity)
    info = {
        "period": period.key,
        "reference_months": list(reference_months),
        "reference_fingerprint": fingerprint,
        "global_duration_sec": reference.global_duration_sec,
        "n_reference_labels": int(reference.residuals_sec.size),
        "path": str(path.relative_to(layout.root)),
        "sha256": sha256_file(path),
    }
    return features, info


# ---------------------------------------------------------------------------
# Extra feature blocks


def load_extra_blocks(paths: list[str], existing: set[str], keep: pl.DataFrame | None = None) -> tuple[list[pl.DataFrame], list[str], list[str], list[dict[str, Any]]]:
    blocks: list[pl.DataFrame] = []
    cats: list[str] = []
    nums: list[str] = []
    info: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw).resolve()
        if keep is None:
            frame = pl.read_parquet(path)
        else:  # bake smoke only (--smoke-train-frac): the block's rows of the kept ids
            lazy = pl.scan_parquet(path)
            ids = keep.with_columns(pl.col("MVT_ID_mvt").cast(lazy.collect_schema()["MVT_ID_mvt"]))
            frame = lazy.join(ids.lazy(), on="MVT_ID_mvt", how="semi").collect()
        if "MVT_ID_mvt" not in frame.columns:
            raise SystemExit(f"extra feature block {path} lacks MVT_ID_mvt")
        if frame["MVT_ID_mvt"].n_unique() != frame.height:
            raise SystemExit(f"extra feature block {path} has duplicate MVT_ID_mvt")
        columns = [c for c in frame.columns if c != "MVT_ID_mvt"]
        s4.assert_no_forbidden_columns(columns, label=f"extra block {path.name}")
        clash = sorted(set(columns) & (existing | set(cats) | set(nums)))
        if clash:
            raise SystemExit(f"extra block {path.name} collides with existing columns: {clash[:10]}")
        block_cats = [c for c in columns if frame.schema[c] in (pl.String, pl.Categorical)]
        block_nums = [c for c in columns if c not in block_cats]
        cats.extend(block_cats)
        nums.extend(block_nums)
        blocks.append(frame.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64)))
        info.append({"path": str(path), "sha256": sha256_file(path), "rows": frame.height, "cats": block_cats, "nums": block_nums})
    return blocks, cats, nums, info


# ---------------------------------------------------------------------------
# Fit


def _read_concat(paths: list[Path]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")


def smoke_keep_ids(layout: CacheLayout, scope_periods: list[Period], train_keys: set[str], eval_keys: set[str], frac: float,
                   eval_frac: float | None = None) -> pl.DataFrame:
    """Bake smoke (``--smoke-train-frac``): a fixed hash sample of the training-period rows (an arithmetic hash of
    MVT_ID_mvt, identical on every machine); every ranking row (a candidate needs them all); every other evaluation row,
    or a hash sample of them with ``--smoke-eval-frac`` (JAN_JUL / Nov/Dec)."""

    def sample(ids: pl.DataFrame, f: float) -> pl.DataFrame:
        return ids.filter(((pl.col("MVT_ID_mvt").cast(pl.Int64) * 2654435761) % 4294967296) < int(f * 4294967296))

    parts = []
    for p in scope_periods:
        ids = pl.read_parquet(layout.base(p), columns=["MVT_ID_mvt"]).with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
        if p.key in eval_keys:
            parts.append(sample(ids, eval_frac) if eval_frac and not p.is_ranking else ids)
        elif p.key in train_keys:
            parts.append(sample(ids, frac))
    return pl.concat(parts).unique("MVT_ID_mvt", maintain_order=True)


def assemble(
    args: argparse.Namespace,
    layout: CacheLayout,
    src: Path,
    train_months: tuple[int, ...],
    eval_periods: tuple[Period, ...],
    scope: str,
    scope_periods: list[Period],
    extra_blocks: list[pl.DataFrame],
    cats: list[str],
    nums: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    train_keys = {p.key for p in s4.periods_for_months(train_months)}
    eval_keys = {p.key for p in eval_periods}
    info: dict[str, Any] = {}
    base = pl.concat(
        [pl.read_parquet(layout.base(p)).with_columns(pl.lit(p.key).alias("period")) for p in scope_periods],
        how="vertical_relaxed",
    )
    ctx = _read_concat([layout.temporal(scope, p) for p in scope_periods])
    native = _read_concat([layout.native(p) for p in scope_periods])
    if getattr(args, "temporal_mode", "contiguous") == "isocap_v1":
        native = s4.cap_native_lookbacks(native)  # SA-EMA-X1 item 1: horizon-censored native lookback
    labels = _read_concat([layout.labels(p) for p in scope_periods if not p.is_ranking]).select(
        "MVT_ID_mvt", s4.LABEL_COLUMN
    )
    ref_base = base  # the S4 duration references always use every labelled row of their months
    keep = getattr(args, "_smoke_keep", None)
    if keep is not None:  # bake smoke only: the kept rows (all evaluation rows, a sample of the training rows)
        base = base.join(keep.with_columns(pl.col("MVT_ID_mvt").cast(base.schema["MVT_ID_mvt"])), on="MVT_ID_mvt", how="semi", maintain_order="left")
        info["smoke_train_frac"] = args.smoke_train_frac
    frame = (
        base.join(ctx, on="MVT_ID_mvt", how="left", validate="1:1", maintain_order="left")
        .join(native, on="MVT_ID_mvt", how="left", validate="1:1", maintain_order="left")
        .join(labels, on="MVT_ID_mvt", how="left", validate="1:1", maintain_order="left")
    )
    del ctx, native
    if args.row_order_reference:
        reference_ids = pl.read_parquet(args.row_order_reference, columns=["MVT_ID_mvt"]).with_row_index("_order")
        frame = frame.join(reference_ids, on="MVT_ID_mvt", how="left", maintain_order="left").with_row_index("_canonical")
        frame = frame.sort(["_order", "_canonical"], nulls_last=True).drop("_order", "_canonical")
        info["row_order_reference"] = {"path": str(args.row_order_reference), "sha256": sha256_file(Path(args.row_order_reference))}
    label_frames = {
        p.month: s4.label_frame(ref_base.filter(pl.col("period") == p.key), pl.read_parquet(layout.labels(p)))
        for p in scope_periods
        if not p.is_ranking and p.month in train_months
    }
    train_all = frame.filter(pl.col("period").is_in(list(train_keys)))
    train = matched_booster_frame(train_all)
    info["train_rows_in_months"] = train_all.height
    info["train_kept"] = train.height
    info["train_dropped"] = train_all.height - train.height
    evaluation = frame.filter(pl.col("period").is_in(list(eval_keys)))
    info["eval_rows"] = evaluation.height
    del train_all, frame
    gc.collect()
    log(f"assembled frame: train_kept={info['train_kept']} dropped={info['train_dropped']} eval={info['eval_rows']}")

    q1 = s4.q1_module()
    started = time.perf_counter()
    train, evaluation, q1_diag = q1.attach_baselines(train, evaluation)
    info["q1"] = {"fit_months_train_rows": "leave-one-month-out within train months", "eval_fit_months": list(train_months), "diagnostics": q1_diag, "seconds": round(time.perf_counter() - started, 1)}
    log("q1 baselines attached")

    arrivals = _read_concat([layout.arrivals(p) for p in scope_periods if not p.is_ranking])
    train, evaluation = attach_cross_fitted_arrival_state(train, evaluation, arrivals, train_months=train_months)
    info["native_normaliser"] = {"train": "cross-fitted by month within train months", "eval_fit_months": list(train_months), "arrival_rows": arrivals.height}
    del arrivals
    log("native arrival state attached")

    ref_cache: dict[tuple[int, ...], Any] = {}
    learned_train: list[pl.DataFrame] = []
    learned_eval: list[pl.DataFrame] = []
    s4_info: list[dict[str, Any]] = []
    for period in scope_periods:
        is_train = period.key in train_keys
        is_eval = period.key in eval_keys
        if not (is_train or is_eval):
            continue
        ids = base.filter(pl.col("period") == period.key).select("MVT_ID_mvt")
        if is_train:
            months = s4.s4_reference_months(period, train_months, cross_fit=True)
            features, meta = s4_learned_for(layout, period, months, label_frames, ref_cache)
            learned_train.append(features.join(ids, on="MVT_ID_mvt", how="semi"))
            s4_info.append({**meta, "role": "train"})
        if is_eval:
            months = s4.s4_reference_months(period, train_months, cross_fit=False)
            features, meta = s4_learned_for(layout, period, months, label_frames, ref_cache)
            learned_eval.append(features.join(ids, on="MVT_ID_mvt", how="semi"))
            s4_info.append({**meta, "role": "eval"})
    info["s4_references"] = s4_info
    del base, ref_base, label_frames, ref_cache
    train = train.join(pl.concat(learned_train), on="MVT_ID_mvt", how="left", validate="1:1", maintain_order="left")
    evaluation = evaluation.join(pl.concat(learned_eval), on="MVT_ID_mvt", how="left", validate="1:1", maintain_order="left")
    log("s4 occupancy attached")

    for block in extra_blocks:
        train = train.join(block, on="MVT_ID_mvt", how="left", validate="1:1", maintain_order="left")
        evaluation = evaluation.join(block, on="MVT_ID_mvt", how="left", validate="1:1", maintain_order="left")
    if getattr(args, "temporal_mode", "contiguous") == "isocap_v1":
        from prc_challenge import lookback_horizon as lh

        # SA-EMA-X1 item 1 (D-082 addition): pool/frame-size dependent ADS-B support counts capped at D-fit p99
        train, evaluation = lh.cap_support_counts(train), lh.cap_support_counts(evaluation)
    keep = list(
        dict.fromkeys(
            [
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
                *cats,
                *nums,
            ]
        )
    )
    missing = [c for c in keep if c not in train.columns or c not in evaluation.columns]
    if missing:
        raise RuntimeError(f"assembled frames lack columns: {missing}")
    train = train.select(keep)
    evaluation = evaluation.select(keep)
    s4.assert_no_forbidden_columns([*cats, *nums], label="model matrix")
    info["coverage"] = {
        "train_s4_nonnull": float(train["s4_expected_airport_occ_tminus_0s"].is_not_null().mean()),
        "eval_s4_nonnull": float(evaluation["s4_expected_airport_occ_tminus_0s"].is_not_null().mean()),
        "train_native_available": float(train["native_arr_available"].fill_null(0).mean()),
        "eval_native_available": float(evaluation["native_arr_available"].fill_null(0).mean()),
    }
    return train, evaluation, info


def target_spec_from_args(args: argparse.Namespace) -> s4.TargetSpec:
    """Fit-target definition; the defaults reproduce the historical S4 residual exactly."""
    return s4.TargetSpec(
        offset_col=getattr(args, "target_offset_col", "unimpeded_p10"),
        sign=getattr(args, "target_sign", 1),
        fill_col=getattr(args, "target_offset_fill_col", None),
        winsor_sec=getattr(args, "target_winsor", None),
    )


def stage_fit(args: argparse.Namespace, layout: CacheLayout, src: Path) -> None:
    from catboost import CatBoostRegressor

    from prc_challenge.campaign.models import predict_unclipped, to_pool

    stage_times: dict[str, float] = {}
    t0 = time.perf_counter()
    git_at_start = git_state()
    train_months = parse_months(args.train_months)
    s4.validate_run_months(train_months, args.eval)
    eval_periods = s4.eval_periods_for(args.eval)
    scope_periods = sorted(set(s4.periods_for_months(train_months)) | set(eval_periods))
    if args.eval == "novdec":
        # D-fit models are fitted elsewhere; this mode only writes the Nov/Dec evaluation matrix.
        if not args.assemble_only:
            raise SystemExit("--eval novdec writes matrices for existing D-fit models: pass --assemble-only")
        # Label-free temporal context over the contiguous 2025 year (Jan/Jul rows are context only:
        # their labels never enter a reference, and they are neither train nor eval rows).
        scope_periods = list(s4.TRAIN_PERIODS)
    run = layout.run_dir(args.tag)
    run.mkdir(parents=True, exist_ok=True)
    log(f"fit tag={args.tag} train_months={train_months} eval={args.eval} scope={[p.key for p in scope_periods]}")

    ensure_base_caches(layout, src, scope_periods, force=False)
    ensure_isolated_caches(layout, src, scope_periods, force=False)
    scope = ensure_temporal_scope(layout, src, scope_periods, force=False, mode=args.temporal_mode)
    stage_times["caches"] = time.perf_counter() - t0

    base_cats, base_nums = s4.s4_feature_lists()
    smoke_frac = getattr(args, "smoke_train_frac", None)
    if smoke_frac:  # bake smoke only: every evaluation row, a fixed sample of the training rows
        eval_frac = getattr(args, "smoke_eval_frac", None)
        args._smoke_keep = smoke_keep_ids(layout, scope_periods, {p.key for p in s4.periods_for_months(train_months)}, {p.key for p in eval_periods},
                                          smoke_frac, eval_frac)
        log(f"SMOKE: training-period rows sampled at {smoke_frac}, non-ranking evaluation rows at {eval_frac or 1.0}; kept ids {args._smoke_keep.height}")
    extra_blocks, extra_cats, extra_nums, extra_info = load_extra_blocks(args.extra_features or [], set(base_cats) | set(base_nums),
                                                                         getattr(args, "_smoke_keep", None))
    cats, nums = s4.s4_feature_lists(extra_cats, extra_nums)

    cache_files = {}
    for p in scope_periods:
        for name, path in (("base", layout.base(p)), ("native", layout.native(p)), ("events", layout.events(p)), ("temporal", layout.temporal(scope, p))):
            cache_files[f"{name}:{p.key}"] = sha256_file(path)
        if not p.is_ranking:
            cache_files[f"labels:{p.key}"] = sha256_file(layout.labels(p))
            cache_files[f"arrivals:{p.key}"] = sha256_file(layout.arrivals(p))
    identity = {
        "train_months": list(train_months),
        "eval": args.eval,
        "scope": scope,
        "cache_files": cache_files,
        "features": [cats, nums],
        "extra": extra_info,
        "row_order_reference": sha256_file(Path(args.row_order_reference)) if args.row_order_reference else None,
        "module_sha256": sha256_file(Path(s4.__file__)),
    }
    if smoke_frac:
        identity["smoke_train_frac"] = smoke_frac
        if getattr(args, "smoke_eval_frac", None):
            identity["smoke_eval_frac"] = args.smoke_eval_frac
    if args.temporal_mode != "contiguous":
        from prc_challenge import lookback_horizon as lh

        identity["temporal_mode"] = {"mode": args.temporal_mode, "caps": lh.HORIZON_CAPS_V1, "support_caps": lh.SUPPORT_CAPS_V1,
                                     "module_sha256": sha256_file(Path(lh.__file__))}
    identity_hash = json_sha256(identity)
    train_path, eval_path, identity_path = run / "train_matrix.parquet", run / "eval_matrix.parquet", run / "matrix_identity.json"
    t1 = time.perf_counter()
    if (
        not args.force
        and train_path.exists()
        and eval_path.exists()
        and identity_path.exists()
        and json.loads(identity_path.read_text()).get("identity_sha256") == identity_hash
    ):
        assembly_info = json.loads((run / "assembly_info.json").read_text())
        log("reusing assembled matrices")
    else:
        train, evaluation, assembly_info = assemble(args, layout, src, train_months, eval_periods, scope, scope_periods, extra_blocks, cats, nums)
        atomic_parquet(train, train_path)
        atomic_parquet(evaluation, eval_path)
        atomic_json(run / "assembly_info.json", assembly_info)
        atomic_json(identity_path, {**identity, "identity_sha256": identity_hash})
        del train, evaluation
        gc.collect()
        log("matrices written")
    stage_times["assemble"] = time.perf_counter() - t1
    del extra_blocks
    if args.assemble_only:
        summary = {
            "tag": args.tag,
            "assemble_only": True,
            "created_utc": now_utc(),
            "train_months": list(train_months),
            "eval": args.eval,
            "eval_periods": [p.key for p in eval_periods],
            "temporal_scope": scope,
            "features": {"cats": cats, "nums": nums, "feature_schema_sha256": json_sha256([cats, nums])},
            "extra_features": extra_info,
            "matrix_identity_sha256": identity_hash,
            "matrices": {"train": {"path": str(train_path), "sha256": sha256_file(train_path)}, "eval": {"path": str(eval_path), "sha256": sha256_file(eval_path)}},
            "row_counts": {k: assembly_info[k] for k in ("train_rows_in_months", "train_kept", "train_dropped", "eval_rows")},
            "references": {k: assembly_info.get(k) for k in ("q1", "native_normaliser", "s4_references")},
            "git": git_at_start,
            "stage_seconds": {k: round(v, 1) for k, v in stage_times.items()},
            "peak_rss_gb": peak_rss_gb(),
        }
        atomic_json(run / "assemble_manifest.json", summary)
        log(f"assemble-only: matrices at {run}")
        return

    t31 = json.loads((src / T31_PARAMS).read_text())
    params = s4.s4_params(t31, iterations=args.iterations, threads=args.threads, seed=args.seed)
    if args.learning_rate is not None:  # bake (D-076 code gap #2b); default: the t31 rate, params (and their sha) unchanged
        params["learning_rate"] = float(args.learning_rate)
    if args.task_type == "GPU":  # bake GPU CatBoost recipe, only after its gate adopted it (D-081)
        params = s4.gpu_recipe(params, args.devices, args.gpu_bootstrap)
    params_sha = json_sha256(params)
    target = target_spec_from_args(args)
    model_path = run / "model.cbm"
    t2 = time.perf_counter()
    fit_info_path = run / "fit_info.json"
    prior = json.loads(fit_info_path.read_text()) if fit_info_path.exists() else {}
    if (
        model_path.exists()
        and not args.force
        and prior.get("params_sha256") == params_sha
        and prior.get("matrix_identity_sha256") == identity_hash
        and prior.get("target", s4.TargetSpec().as_dict()) == target.as_dict()
    ):
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        log("reusing fitted model")
    else:
        mp = os.popen("memory_pressure | tail -1").read().strip()
        log(f"memory before fit: {mp}")
        train = pl.read_parquet(train_path)
        absent = [c for c in target.columns() if c not in train.columns]
        if absent:
            raise RuntimeError(f"target offset columns {absent} are not in the model matrix (add them via --extra-features)")
        train = target.with_fit_label(train)
        if train[s4.RESIDUAL_LABEL].null_count():
            raise RuntimeError(f"fit label has nulls (missing label or target offset {target.as_dict()})")
        log(f"fit target: {target.as_dict()}")
        pool = to_pool(train, cats, nums, label=s4.RESIDUAL_LABEL)
        n_train = train.height
        del train
        gc.collect()
        fit_params = {
            **params,
            # Logging/snapshot options only; verified result-neutral (max diff 0.0).
            "verbose": args.log_every,
            "allow_writing_files": True,
            "train_dir": str(run / "catboost_train"),
            "save_snapshot": True,
            "snapshot_file": "s4_fit.cbsnapshot",
            "snapshot_interval": 600,
        }
        log(f"fitting CatBoost n={n_train} features={len(cats) + len(nums)} params_sha={params_sha[:12]}")
        model = CatBoostRegressor(**fit_params)
        model.fit(pool)
        del pool
        gc.collect()
        model.save_model(str(model_path))
        snapshot = run / "catboost_train" / "s4_fit.cbsnapshot"
        if snapshot.exists():
            snapshot.unlink()  # resume point no longer needed once the model is saved
        atomic_json(run / "fit_info.json", {"params": params, "params_sha256": params_sha, "matrix_identity_sha256": identity_hash, "target": target.as_dict(), "n_train": n_train, "fit_seconds": time.perf_counter() - t2, "model_sha256": sha256_file(model_path)})
        log("model saved")
    stage_times["fit"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    evaluation = pl.read_parquet(eval_path)
    raw = target.predict(evaluation, predict_unclipped(model, evaluation, cats, nums))
    result: dict[str, Any] = {
        "tag": args.tag,
        "recipe": s4.RECIPE,
        "created_utc": now_utc(),
        "train_months": list(train_months),
        "eval": args.eval,
        "eval_periods": [p.key for p in eval_periods],
        "temporal_scope": scope,
        "nov_dec_labels_used": sorted(set(train_months) & set(s4.LOCKED_EVAL_MONTHS)),
        "nov_dec_role": "blind final training rows only" if set(train_months) & set(s4.LOCKED_EVAL_MONTHS) else "unused",
        "row_counts": {k: assembly_info[k] for k in ("train_rows_in_months", "train_kept", "train_dropped", "eval_rows")},
        "coverage": assembly_info.get("coverage"),
        "params": params,
        "params_sha256": params_sha,
        "target": target.as_dict(),
        "features": {"cats": cats, "nums": nums, "n": len(cats) + len(nums), "feature_schema_sha256": json_sha256([cats, nums])},
        "extra_features": extra_info,
        "matrix_identity_sha256": identity_hash,
        "matrices": {"train": {"path": str(train_path), "sha256": sha256_file(train_path)}, "eval": {"path": str(eval_path), "sha256": sha256_file(eval_path)}},
        "model": {"path": str(model_path), "sha256": sha256_file(model_path), "tree_count": int(model.tree_count_)},
        "inputs": {
            "raw_sha256": s4.raw_input_hashes(src / "data" / "raw"),
            "metar": s4.metar_input_hash(src / "data" / "external" / "iem_metar"),
            "t31_params_sha256": sha256_file(src / T31_PARAMS),
            "cache_manifest_sha256": sha256_file(layout.manifest) if layout.manifest.exists() else None,
            "cache_files_sha256": identity["cache_files"],
        },
        "references": {k: assembly_info.get(k) for k in ("q1", "native_normaliser", "s4_references")},
        "git": git_at_start,
        "git_at_end": git_state(),
    }
    if args.eval == "janjul":
        parent = pl.read_parquet(src / E33B_PARENT)
        scored = s4.compose_janjul_primary(evaluation, raw, parent)
        if scored.height != s4.PRIMARY_N:
            raise RuntimeError(f"JAN_JUL population {scored.height} != {s4.PRIMARY_N}")
        pred_path = run / f"predictions_{args.tag}_janjul.parquet"
        out = scored.select("MVT_ID_mvt", "AIRPORT", s4.LABEL_COLUMN, "prediction", "nm_unmatched", "month").with_columns(
            pl.Series("s4_raw_prediction", raw)
        )
        atomic_parquet(out, pred_path)
        result["scorecard"] = s4.janjul_scorecard(scored)
        result["composition"] = {"order": ["clip>=0", "E17", "E18", "unmatched parent freeze", "leftover parent freeze", "clip>=0"], "parent": str(E33B_PARENT), "parent_sha256": sha256_file(src / E33B_PARENT)}
        reference = src / "artifacts" / "experiments" / s4.CAMPAIGN / "reference" / "predictions_overnight_s4_expected_occupancy_V1.parquet"
        if reference.exists():
            hist = pl.read_parquet(reference)
            joined = out.join(hist.select("MVT_ID_mvt", pl.col("prediction").alias("hist")), on="MVT_ID_mvt", how="inner", validate="1:1")
            matched = joined.filter(pl.col("nm_unmatched") == 0)
            result["vs_historical_s4"] = {
                "reference_sha256": sha256_file(reference),
                "historical_pooled_rmse": s4.HISTORICAL_S4_JANJUL_RMSE,
                "delta_pooled_rmse": result["scorecard"]["JAN_JUL_2025"]["rmse_seconds"] - s4.HISTORICAL_S4_JANJUL_RMSE,
                "historical_scorecard": s4.janjul_scorecard(hist),
                "all_rows": s4.rowwise_comparison(joined["prediction"].to_numpy(), joined["hist"].to_numpy()),
                "matched_rows": s4.rowwise_comparison(matched["prediction"].to_numpy(), matched["hist"].to_numpy()),
                "matched_checksum_rebuilt": payload_sha256(matched, "prediction"),
                "matched_checksum_historical": payload_sha256(matched, "hist"),
            }
        result["predictions"] = {"path": str(pred_path), "sha256": sha256_file(pred_path), "all_rows_checksum": payload_sha256(out, "prediction")}
        card = result["scorecard"]
        log(f"JAN_JUL pooled={card['JAN_JUL_2025']['rmse_seconds']:.6f} jan={card['JAN_2025']['rmse_seconds']:.4f} jul={card['JUL_2025']['rmse_seconds']:.4f} matched={card['matched_only']['rmse_seconds']:.4f}")
    else:
        composed = s4.compose_ranking_matched(evaluation, raw)
        if composed.height != s4.RANKING_N:
            raise RuntimeError(f"ranking population {composed.height} != {s4.RANKING_N}")
        pred_path = run / f"ranking_s4_{args.tag}.parquet"
        out = composed.select("MVT_ID_mvt", "period", "AIRPORT", "nm_unmatched", "gate_e17", "gate_e18", "gate_leftover", "owned_by_unmatched_overlay", "s4_raw_prediction", "prediction")
        atomic_parquet(out, pred_path)
        matched = out.filter(pl.col("nm_unmatched") == 0)
        v12 = pl.read_parquet(src / V12_FILE)
        parent = s4.v12_equivalent_parent(out, v12)
        parent_path = run / f"v12_equivalent_{args.tag}.parquet"
        atomic_parquet(parent, parent_path)
        cmp = matched.select("MVT_ID_mvt", "prediction").join(
            v12.select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").alias("v12")), on="MVT_ID_mvt", how="inner", validate="1:1"
        )
        v12_matched = cmp.select("MVT_ID_mvt", pl.col("v12").alias("TAXITIME_SEC_mvt"))
        result["predictions"] = {
            "path": str(pred_path),
            "sha256": sha256_file(pred_path),
            "matched_n": matched.height,
            "unmatched_n": int((out["nm_unmatched"] == 1).sum()),
            "matched_checksum": payload_sha256(matched, "prediction"),
            "v12_equivalent_parent": {"path": str(parent_path), "sha256": sha256_file(parent_path)},
        }
        result["vs_v12_matched"] = {
            "v12_sha256": sha256_file(src / V12_FILE),
            **s4.rowwise_comparison(cmp["prediction"].to_numpy(), cmp["v12"].to_numpy()),
            "matched_checksum_rebuilt": payload_sha256(matched, "prediction"),
            "matched_checksum_v12": payload_sha256(v12_matched, "TAXITIME_SEC_mvt"),
        }
        log(f"ranking matched vs v12: r={result['vs_v12_matched']['pearson_r']:.6f} mean|d|={result['vs_v12_matched']['mean_abs_diff']:.3f} max|d|={result['vs_v12_matched']['max_abs_diff']:.1f}")
    stage_times["predict_compose"] = time.perf_counter() - t3
    result["stage_seconds"] = {k: round(v, 1) for k, v in stage_times.items()}
    result["peak_rss_gb"] = peak_rss_gb()
    atomic_json(run / "run_manifest.json", result)
    atomic_json(layout.root / "manifests" / f"run_{args.tag}.json", result)
    log(f"run manifest written: {run / 'run_manifest.json'}")


# ---------------------------------------------------------------------------
# S1 unmatched overlay (frozen campaign harness)


def stage_compose_s1(args: argparse.Namespace, layout: CacheLayout, src: Path) -> None:
    from catboost import CatBoostRegressor

    from prc_challenge.campaign.runtime import default_config
    from prc_challenge.submission import validate_submission
    from prc_challenge.unmatched_recovery.frozen_campaign import FrozenCampaign, load_ranking_unmatched_features
    from prc_challenge.unmatched_recovery.harness import (
        overlay_unmatched_submission,
        prepare_unmatched_preds,
        verify_arm_submission,
    )
    from prc_challenge.unmatched_recovery.methods import predict_regressor

    run = layout.run_dir(args.tag)
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest["eval"] != "ranking":
        raise SystemExit("compose-s1 needs a ranking run")
    parent_path = Path(manifest["predictions"]["v12_equivalent_parent"]["path"])
    parent = pl.read_parquet(parent_path)
    camp = FrozenCampaign(repo=src, output=layout.root / "compose_s1" / "frozen_campaign_io", cfg=default_config(src), threads=args.threads, skip_upload=True)
    ranking_u = load_ranking_unmatched_features(camp)
    freeze = json.loads((src / S1_FREEZE).read_text())
    spec = freeze["arms"]["S1"]
    model = CatBoostRegressor()
    model.load_model(str(src / S1_MODEL))
    names = list(model.feature_names_)
    cats = [c for c in spec["cats"] if c in names]
    nums = [c for c in spec["nums"] if c in names]
    if cats + nums != names:
        raise RuntimeError(f"S1 model feature order mismatch: {names} vs {cats + nums}")
    pred = predict_regressor(model, ranking_u, cats, nums)
    scored_u = prepare_unmatched_preds(ranking_u, pred)
    template = camp.template()
    submission = overlay_unmatched_submission(parent, scored_u)
    verify = verify_arm_submission(submission, template=template, parent=parent, unmatched_ids=ranking_u["MVT_ID_mvt"])
    errors = validate_submission(submission, template=template)
    v15 = pl.read_parquet(src / V15_FILE)
    v15_unmatched = v15.join(ranking_u.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi")
    s1_matches_v15 = payload_sha256(v15_unmatched, "TAXITIME_SEC_mvt") == verify["unmatched_checksum"]
    out_dir = layout.root / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"s4rebuild_{args.tag}_S1_ranking_candidate.parquet"
    submission.select("MVT_ID_mvt", "TAXITIME_SEC_mvt").write_parquet(dest)
    cmp = submission.join(v15.rename({"TAXITIME_SEC_mvt": "v15"}), on="MVT_ID_mvt", how="inner", validate="1:1")
    matched_cmp = cmp.join(ranking_u.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti")
    y = submission["TAXITIME_SEC_mvt"].to_numpy()
    sidecar = {
        "file": str(dest),
        "sha256": sha256_file(dest),
        "n": submission.height,
        "created_utc": now_utc(),
        "not_uploaded": True,
        "not_in_submissions_dir": True,
        "recipe": "S4 expected occupancy (rebuilt, reproducible trainer) matched + frozen S1 unmatched overlay",
        "matched_source_run": args.tag,
        "matched_run_manifest": str(run / "run_manifest.json"),
        "training_months": manifest["train_months"],
        "s1_model": {"path": str(S1_MODEL), "sha256": sha256_file(src / S1_MODEL), "features": names},
        "matched_checksum": verify["matched_checksum"],
        "unmatched_checksum": verify["unmatched_checksum"],
        "v15_matched_checksum": s4.V15_MATCHED_CHECKSUM,
        "v15_unmatched_checksum": s4.V15_UNMATCHED_CHECKSUM,
        "s1_unmatched_identical_to_v15": bool(s1_matches_v15),
        "matched_identical_to_v15": verify["matched_checksum"] == s4.V15_MATCHED_CHECKSUM,
        "verify": {k: v for k, v in verify.items() if k != "errors"} | {"errors": verify["errors"]},
        "validate_submission_errors": errors,
        "vs_v15_all_rows": s4.rowwise_comparison(cmp["TAXITIME_SEC_mvt"].to_numpy(), cmp["v15"].to_numpy()),
        "vs_v15_matched_rows": s4.rowwise_comparison(matched_cmp["TAXITIME_SEC_mvt"].to_numpy(), matched_cmp["v15"].to_numpy()),
        "pred_mean": float(np.mean(y)),
        "pred_min": float(np.min(y)),
        "pred_max": float(np.max(y)),
        "provenance": {"aobt": False, "arvt3": False, "opdi": False, "opensky": False, "v4_for_selection": False, "sources": ["challenge native movements", "IEM METAR (+raw, altimeter)", "OurAirports runway headings"]},
    }
    atomic_json(dest.with_suffix(".sidecar.json"), sidecar)
    atomic_json(layout.root / "manifests" / f"candidate_{args.tag}_S1.json", sidecar)
    print(json.dumps({k: sidecar[k] for k in ("file", "sha256", "matched_checksum", "unmatched_checksum", "s1_unmatched_identical_to_v15", "matched_identical_to_v15", "validate_submission_errors")}, indent=2))
    if errors or not verify["ok"]:
        raise SystemExit(f"candidate verification failed: {errors or verify['errors']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=("build-caches", "verify-caches", "fit", "compose-s1"))
    parser.add_argument("--source-root", help="checkout holding data/ and bulky artifacts/ (default: main checkout)")
    parser.add_argument("--out", help="artifact namespace (default <source-root>/artifacts/experiments/S4-allyear-rebuild-20260925)")
    parser.add_argument("--periods", help="comma-separated period keys for build-caches (default: all)")
    parser.add_argument("--train-months", default="1-12", help="'D', '1-12', or comma list")
    parser.add_argument(
        "--eval",
        default="ranking",
        choices=("janjul", "ranking", "novdec"),
        help="novdec (V4, D-072): Nov/Dec 2025 rows as evaluation rows of a D-fit matrix; --assemble-only only",
    )
    parser.add_argument("--tag", default="allyear")
    parser.add_argument("--extra-features", action="append", default=[], help="parquet keyed by MVT_ID_mvt; repeatable")
    parser.add_argument("--row-order-reference", help="parquet whose MVT_ID_mvt order fixes training row order (replay diagnostics)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=s4.HISTORICAL_SEED)
    parser.add_argument("--iterations", type=int, default=s4.HISTORICAL_ITERATIONS)
    parser.add_argument("--learning-rate", type=float, default=None, help="override the t31 learning rate (bake stage C at the stage-A count)")
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="CPU", help="GPU = the separately gated bake recipe (s4.gpu_recipe)")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--gpu-bootstrap", choices=("MVS", "Bernoulli"), default="MVS")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--assemble-only", action="store_true", help="write the full-stack train/eval matrices and stop before fitting")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--target-offset-col", default="unimpeded_p10", help="fit t = sign*(y - offset); predict max(0, offset + sign*t_hat)")
    parser.add_argument("--target-sign", type=int, choices=(1, -1), default=1, help="-1 with takeoff_minus_aobt_sec = AOBT delta (BLOCK - AOBT)")
    parser.add_argument("--target-offset-fill-col", default=None, help="offset for rows where --target-offset-col is null (e.g. unimpeded_p10)")
    parser.add_argument("--target-winsor", type=float, default=None, help="clip the fit target to [-W, W]; fit only")
    parser.add_argument("--smoke-train-frac", type=float, default=None,
                        help="bake smoke only: assemble every evaluation row and this hash-sampled share of the training rows (tiny matrices)")
    parser.add_argument("--smoke-eval-frac", type=float, default=None,
                        help="bake smoke only (with --smoke-train-frac): also hash-sample the JAN_JUL / Nov/Dec evaluation rows (ranking rows stay complete)")
    parser.add_argument(
        "--temporal-mode",
        default="contiguous",
        choices=s4.TEMPORAL_MODES,
        help="isocap_v1 (SA-EMA-X1 item 1, for the bake): month-isolated temporal-v2 context with horizon-censored "
        "lookback gaps (ctx_sec_since_prev_*, native_same_runway_age_sec) and ADS-B support counts capped at the D-fit p99 "
        "(adsb_bias_n, adsb_bias_first_n, adsb_centroid_n, adsb_q_n_after, adsb3_map_n); contiguous = historical recipe",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    src = s4.resolve_source_root(args.source_root)
    layout = CacheLayout(Path(args.out).resolve() if args.out else src / "artifacts" / "experiments" / s4.CAMPAIGN)
    layout.root.mkdir(parents=True, exist_ok=True)
    if args.stage == "build-caches":
        stage_build_caches(args, layout, src)
    elif args.stage == "verify-caches":
        stage_verify_caches(args, layout, src)
    elif args.stage == "fit":
        stage_fit(args, layout, src)
    else:
        stage_compose_s1(args, layout, src)


if __name__ == "__main__":
    main()
