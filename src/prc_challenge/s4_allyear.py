# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Reproducible S4 expected-occupancy matched model (D-fit replay and all-year refit).

This module rebuilds, from raw challenge files plus IEM METAR, every input of the
historical S4 recipe (overnight primary 2026-09-19, ranking submission v12):

* ``base``        E08b planning/calendar/location/METAR features plus the E34
                  delayed-bunching counts, per UTC month of ``MVT_TIME_UTC_mvt``
                  (bunching context = same month, as ``prepare_model_cache.py``).
* ``labels``      departure taxi labels, stored apart from every feature cache.
* ``temporal_v2`` R2 temporal context.  Historically it is *not* month
                  isolated: it was computed over one contiguous DEP/ARR table
                  (months 1-10 for the D-fit cache).  We therefore build it per
                  *scope* (the sorted union of periods a run uses) and store it
                  sliced per period.
* ``native``      Wave2 native completed-arrival state, month-isolated per raw
                  file (per UTC month for the ranking file).
* ``events``      Surface-reconstruction event tables, month-isolated per raw
                  file (per UTC month for the ranking file).

Fit-dependent references are built per split and cached with an identity
sidecar: Q1 unimpeded baselines (leave-one-month-out inside the train months,
full train months for evaluation rows), the native arrival taxi-in normaliser
(same cross-fit), and the S4 duration reference (train month ``m`` uses the train
months without ``m``; evaluation periods use every train month).

No departure taxi/block/AOBT/ARVT_3 value of any departure enters a feature
cache.  November/December 2025 labels may only be used as blind training rows of
an all-year fit whose evaluation target is the ranking file.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from prc_challenge.bundles import e34_feature_lists, matched_booster_frame, optuna_winner_params
from prc_challenge.campaign.data import feature_lists as campaign_feature_lists
from prc_challenge.campaign.temporal_v2 import TEMPORAL_V2_NUMS, build_temporal_context_v2
from prc_challenge.features import (
    CAT_FILL,
    apply_leftover_parent_freeze,
    apply_lirf_stuck_block_gates,
    apply_unmatched_parent_freeze,
    lirf_delay_as_taxi_gate,
    lirf_extraday_as_taxi_gate,
    lirf_leftover_killer_gate,
    prepare_model_frame,
)
from prc_challenge.io import CHALLENGE_AIRPORTS, ranking_path, training_paths, with_reporting_airport
from prc_challenge.mask import mask_departure_labels
from prc_challenge.operational_v2 import (
    NATIVE_STATE_FEATURES,
    DurationReference,
    attach_cross_fitted_arrival_state,
    build_event_table,
    build_expected_occupancy_features,
    build_native_completed_arrival_sidecar,
    fit_duration_reference,
    learned_feature_cache_identity,
    load_learned_cache_or_reject,
    reject_forbidden_departure_columns,
    write_learned_cache_identity,
)
from prc_challenge.traffic import DELAY_BUNCH_NUMS, add_delayed_bunching
from prc_challenge.weather import add_metar_features

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "S4-allyear-rebuild-20260925"


def resolve_source_root(explicit: str | Path | None = None) -> Path:
    """Checkout that owns ``data/`` and the bulky ``artifacts/`` (gitignored).

    Order: explicit argument, ``PRC_SOURCE_ROOT``, this checkout when it holds the
    raw data and caches, else the main checkout of the shared git directory (so a
    worktree reads inputs from, and writes bulky outputs to, the main checkout).
    """

    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get("PRC_SOURCE_ROOT")
    if env:
        return Path(env).resolve()
    if (ROOT / "data" / "raw").exists() and (ROOT / "artifacts" / "cache").exists():
        return ROOT
    try:
        import subprocess

        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT, text=True
        ).strip()
        candidate = Path(common).parent
        if (candidate / "data" / "raw").exists():
            return candidate
    except Exception:  # pragma: no cover - git unavailable
        pass
    return ROOT


DEFAULT_OUT = resolve_source_root() / "artifacts" / "experiments" / CAMPAIGN

RECIPE = "s4_expected_occupancy"
D_MONTHS: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 9, 10)
PRIMARY_EVAL_MONTHS: tuple[int, ...] = (1, 7)
ALL_2025_MONTHS: tuple[int, ...] = tuple(range(1, 13))
LOCKED_EVAL_MONTHS: tuple[int, ...] = (11, 12)
HISTORICAL_ITERATIONS = 2210
HISTORICAL_SEED = 42
HISTORICAL_S4_JANJUL_RMSE = 345.06665468615967
HISTORICAL_S4_JANJUL_SHA256 = "6ab6f73b8fe217bc3825c598b7edcbc4d972e8454249f44219f173d192d71321"
V12_SHA256 = "587538524c9bc3bc8e9ea20e38fed9c11fc91daf8538e14571da12b126989cc2"
V15_MATCHED_CHECKSUM = "1bf0c6dee84454f4858421d09ad36c719d2e5f032fa1013066225ccc303e3031"
V15_UNMATCHED_CHECKSUM = "2e17decf02174f8c296c63cb3490b64d06bb1c41eb73c4059f14d263a9bf5f77"
V12_TRAIN_KEPT = 2_062_622
V12_TRAIN_DROPPED = 22_425
PRIMARY_N = 344_419
# Serve set (final phase): the organisers' final file holds Jan, Feb, Jun and Jul 2026. The defaults keep the
# original Jan/Jul ranking file; set PRC_SERVE_MONTHS=1,2,6,7 PRC_RANKING_N=670790 PRC_RANKING_UNMATCHED_N=9057
# (with data/raw/ranking.parquet = final_ranking.parquet) to build and serve the four months.
SERVE_MONTHS: tuple[int, ...] = tuple(int(m) for m in os.environ.get("PRC_SERVE_MONTHS", "1,7").split(","))
RANKING_N = int(os.environ.get("PRC_RANKING_N", "344841"))
RANKING_UNMATCHED_N = int(os.environ.get("PRC_RANKING_UNMATCHED_N", "5283"))

FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt"}
)
LABEL_COLUMN = "TAXITIME_SEC_mvt"
RESIDUAL_LABEL = "_additional_taxi"


@dataclass(frozen=True)
class TargetSpec:
    """Model target ``t = sign * (y - offset)``; prediction ``max(0, offset + sign * t_hat)``.

    The default (``unimpeded_p10``, sign +1, no fill, no winsorization) is the
    historical S4 residual and is computed with the historical expressions, so
    default runs are bit-identical. The AOBT delta reparameterization is
    ``offset_col="takeoff_minus_aobt_sec"``, ``sign=-1`` (the model then learns
    ``delta = BLOCK - AOBT``), ``fill_col="unimpeded_p10"`` for rows without AOBT,
    and ``winsor_sec=7200``. Winsorization touches the fit label only, never the
    evaluation labels or the prediction.
    """

    offset_col: str = "unimpeded_p10"
    sign: int = 1
    fill_col: str | None = None
    winsor_sec: float | None = None

    def __post_init__(self) -> None:
        if self.sign not in (1, -1):
            raise ValueError("target sign must be +1 or -1")
        if self.winsor_sec is not None and self.winsor_sec <= 0:
            raise ValueError("winsor_sec must be positive")

    @property
    def is_default(self) -> bool:
        return self == TargetSpec()

    def as_dict(self) -> dict[str, Any]:
        return {"offset_col": self.offset_col, "sign": self.sign, "fill_col": self.fill_col, "winsor_sec": self.winsor_sec}

    def columns(self) -> list[str]:
        return [self.offset_col] + ([self.fill_col] if self.fill_col else [])

    def offset(self, frame: pl.DataFrame) -> np.ndarray:
        expr = pl.col(self.offset_col).cast(pl.Float64).fill_nan(None)
        if self.fill_col:
            expr = pl.coalesce(expr, pl.col(self.fill_col).cast(pl.Float64).fill_nan(None))
        values = frame.select(expr.alias("_offset"))["_offset"]
        if values.null_count():
            raise RuntimeError(f"target offset {self.offset_col!r} has {values.null_count()} nulls; pass a fill column")
        return values.to_numpy()

    def with_fit_label(self, train: pl.DataFrame) -> pl.DataFrame:
        """Attach ``RESIDUAL_LABEL`` (the fit target) to the training frame."""
        if self.is_default:
            return train.with_columns((pl.col(LABEL_COLUMN) - pl.col("unimpeded_p10")).alias(RESIDUAL_LABEL))
        y = train[LABEL_COLUMN].cast(pl.Float64).to_numpy()
        t = self.sign * (y - self.offset(train))
        if self.winsor_sec is not None:
            t = np.clip(t, -self.winsor_sec, self.winsor_sec)
        return train.with_columns(pl.Series(RESIDUAL_LABEL, t, dtype=pl.Float64))

    def predict(self, evaluation: pl.DataFrame, t_hat: np.ndarray) -> np.ndarray:
        """Invert the target: ``max(0, offset + sign * t_hat)``."""
        if self.is_default:
            return np.clip(evaluation["unimpeded_p10"].to_numpy() + t_hat, 0, None)
        return np.clip(self.offset(evaluation) + self.sign * np.asarray(t_hat, dtype=np.float64), 0, None)

S4_EXPECTED_COLUMNS: tuple[str, ...] = (
    *(f"s4_expected_airport_occ_tminus_{offset}s" for offset in (0, 300, 900, 1800)),
    *(f"s4_expected_runway_occ_tminus_{offset}s" for offset in (0, 300, 900, 1800)),
    "s4_omitted_long_duration_probability",
    "s4_preliminary_duration_sec",
)

# Columns carried by the base cache besides the E34 cats/nums (identical to the
# historical ``prepare_model_cache.py`` keep list minus the label).
BASE_ID_COLUMNS: tuple[str, ...] = (
    "MVT_ID_mvt",
    "month",
    "MVT_TIME_UTC_mvt",
    "nm_unmatched",
    "type_missing",
    "takeoff_minus_sched_sec",
    "takeoff_minus_sched_days",
)
ARR_CONTEXT_COLUMNS: tuple[str, ...] = (
    "MVT_ID_mvt",
    "AIRPORT",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "RUNWAY_mvt",
    "SCHED_TIME_UTC_mvt",
    "month",
)
EVENT_REQUIRED = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt"]
EVENT_OPTIONAL = [
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt",
    "SCHED_TIME_UTC_mvt",
    "EOBT_1_flt",
    "FLIGHT_ID_mvt",
]


# ---------------------------------------------------------------------------
# Periods


@dataclass(frozen=True, order=True)
class Period:
    """One isolated UTC month: a 2025 training month or a 2026 ranking month."""

    year: int
    month: int

    @property
    def key(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def is_ranking(self) -> bool:
        return self.year == 2026

    @classmethod
    def parse(cls, key: str) -> "Period":
        year, month = key.split("-")
        return cls(int(year), int(month))


TRAIN_PERIODS: tuple[Period, ...] = tuple(Period(2025, m) for m in ALL_2025_MONTHS)
RANKING_PERIODS: tuple[Period, ...] = tuple(Period(2026, m) for m in SERVE_MONTHS)
ALL_PERIODS: tuple[Period, ...] = (*TRAIN_PERIODS, *RANKING_PERIODS)


def periods_for_months(months: Iterable[int]) -> tuple[Period, ...]:
    return tuple(Period(2025, int(m)) for m in sorted(set(int(m) for m in months)))


def scope_id(periods: Iterable[Period]) -> str:
    """Readable identifier of a contiguous temporal-context build."""

    keys = sorted(p.key for p in set(periods))
    if not keys:
        raise ValueError("empty scope")
    train = [k for k in keys if k.startswith("2025-")]
    other = [k for k in keys if not k.startswith("2025-")]
    parts: list[str] = []
    if train:
        months = [int(k[5:]) for k in train]
        contiguous = months == list(range(months[0], months[-1] + 1))
        parts.append(f"{train[0]}_to_{train[-1]}" if contiguous else "_".join(train))
    parts.extend(other)
    return "__".join(parts)


def validate_run_months(train_months: Sequence[int], eval_mode: str) -> None:
    """Enforce the validation protocol on a requested fit."""

    months = sorted(set(int(m) for m in train_months))
    if not months:
        raise ValueError("no training months")
    bad = [m for m in months if m not in ALL_2025_MONTHS]
    if bad:
        raise ValueError(f"training months outside 2025: {bad}")
    if eval_mode == "janjul":
        overlap = sorted(set(months) & set(PRIMARY_EVAL_MONTHS))
        if overlap:
            raise RuntimeError(f"JAN_JUL evaluation cannot train on months {overlap}")
        locked = sorted(set(months) & set(LOCKED_EVAL_MONTHS))
        if locked:
            raise RuntimeError(
                f"November/December labels {locked} are only allowed as blind rows of the ranking refit"
            )
    elif eval_mode == "novdec":
        # V4 (D-072): Nov/Dec 2025 rows scored by D-fit models only; their labels are never fit on.
        overlap = sorted(set(months) & (set(PRIMARY_EVAL_MONTHS) | set(LOCKED_EVAL_MONTHS)))
        if overlap:
            raise RuntimeError(f"Nov/Dec (V4) evaluation is for D-fit models; training months {overlap} are not allowed")
    elif eval_mode == "ranking":
        pass
    else:
        raise ValueError(f"unknown eval mode {eval_mode}")


def eval_periods_for(eval_mode: str) -> tuple[Period, ...]:
    if eval_mode == "janjul":
        return periods_for_months(PRIMARY_EVAL_MONTHS)
    if eval_mode == "ranking":
        return RANKING_PERIODS
    if eval_mode == "novdec":
        return periods_for_months(LOCKED_EVAL_MONTHS)
    raise ValueError(eval_mode)


# ---------------------------------------------------------------------------
# Hashing, IO and resource helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def payload_sha256(frame: pl.DataFrame, value_col: str) -> str:
    """Stable checksum of sorted (MVT_ID, value) pairs (same as the frozen campaign)."""

    pairs = frame.select("MVT_ID_mvt", pl.col(value_col).cast(pl.Float64)).sort("MVT_ID_mvt")
    digest = hashlib.sha256()
    digest.update(pairs["MVT_ID_mvt"].to_numpy().tobytes())
    digest.update(np.asarray(pairs[value_col].to_numpy(), dtype=np.float64).tobytes())
    return digest.hexdigest()


def frame_content_sha256(frame: pl.DataFrame, key: str = "MVT_ID_mvt") -> str:
    """Order-independent content hash of a keyed frame (schema + sorted rows)."""

    ordered = frame.sort(key) if key in frame.columns else frame
    digest = hashlib.sha256()
    digest.update(json.dumps({k: str(v) for k, v in ordered.schema.items()}).encode())
    for column in ordered.columns:
        series = ordered[column]
        if series.dtype == pl.String or series.dtype == pl.Categorical:
            digest.update("\x1f".join("\x00" if v is None else v for v in series.cast(pl.String).to_list()).encode())
        elif series.dtype.is_temporal():
            digest.update(series.dt.epoch("us").fill_null(-(2**62)).to_numpy().tobytes())
        else:
            values = series.cast(pl.Float64).to_numpy()
            digest.update(np.nan_to_num(values, nan=-1.2345e308).tobytes())
            digest.update(series.is_null().to_numpy().tobytes())
    return digest.hexdigest()


def atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.write_parquet(temporary)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
    temporary.replace(path)


def peak_rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return float(usage * scale / 1e9)


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def assert_no_forbidden_columns(columns: Iterable[str], *, label: str) -> None:
    bad = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(columns))
    if bad:
        raise RuntimeError(f"{label} contains forbidden departure columns: {bad}")


# ---------------------------------------------------------------------------
# Cache layout


@dataclass(frozen=True)
class CacheLayout:
    root: Path = DEFAULT_OUT

    @property
    def features(self) -> Path:
        return self.root / "features"

    def base(self, period: Period) -> Path:
        return self.features / "base" / f"base_{period.key}.parquet"

    def labels(self, period: Period) -> Path:
        return self.features / "labels" / f"labels_{period.key}.parquet"

    def native(self, period: Period) -> Path:
        return self.features / "native_state" / f"native_{period.key}.parquet"

    def arrivals(self, period: Period) -> Path:
        return self.features / "native_arrivals" / f"arrivals_{period.key}.parquet"

    def events(self, period: Period) -> Path:
        return self.features / "events" / f"events_{period.key}.parquet"

    def temporal(self, scope: str, period: Period) -> Path:
        return self.features / "temporal_v2" / scope / f"temporal_v2_{period.key}.parquet"

    def s4_learned(self, period: Period, reference_hash: str) -> Path:
        return self.features / "s4_learned" / period.key / f"s4_{period.key}_ref-{reference_hash[:16]}.parquet"

    def split_dir(self, split: str) -> Path:
        return self.features / "splits" / split

    @property
    def manifest(self) -> Path:
        return self.root / "manifests" / "feature_cache_manifest.json"

    def run_dir(self, tag: str) -> Path:
        return self.root / "runs" / tag


# ---------------------------------------------------------------------------
# Raw sources


def raw_training_file(raw_dir: Path, month: int) -> Path:
    next_month, next_year = month + 1, 2025
    if next_month == 13:
        next_month, next_year = 1, 2026
    return Path(raw_dir) / f"training_2025-{month:02d}-01_{next_year}-{next_month:02d}-01.parquet"


def raw_source_for_isolated_period(raw_dir: Path, period: Period) -> pl.LazyFrame:
    """Month-isolated raw movements.

    2025 periods use the raw file of that month (the historical event/native
    isolation unit).  2026 ranking periods restrict the single ranking file to one
    UTC month, so January and July 2026 are never connected.
    """

    if period.is_ranking:
        return pl.scan_parquet(ranking_path(raw_dir)).filter(
            (pl.col("MVT_TIME_UTC_mvt").dt.year() == period.year)
            & (pl.col("MVT_TIME_UTC_mvt").dt.month() == period.month)
        )
    return pl.scan_parquet(raw_training_file(raw_dir, period.month))


def raw_input_hashes(raw_dir: Path) -> dict[str, str]:
    paths = [*training_paths(raw_dir), ranking_path(raw_dir), Path(raw_dir) / "submitting.parquet"]
    return {p.name: sha256_file(p) for p in paths if p.exists()}


def metar_input_hash(metar_dir: Path) -> dict[str, Any]:
    files = sorted(Path(metar_dir).glob("*.csv"))
    return {"n_files": len(files), "sha256_of_file_hashes": json_sha256({f.name: sha256_file(f) for f in files})}


# ---------------------------------------------------------------------------
# Base (E08b + bunching) and labels


def _dep_rows_by_utc_month(raw_dir: Path, periods: Sequence[Period]) -> dict[Period, pl.DataFrame]:
    """DEP rows per UTC month of ``MVT_TIME_UTC_mvt`` with a stable raw row index."""

    out: dict[Period, pl.DataFrame] = {}
    wanted_train = sorted(p.month for p in periods if not p.is_ranking)
    if wanted_train:
        frames = [pl.scan_parquet(path) for path in training_paths(raw_dir)]
        train = (
            with_reporting_airport(pl.concat(frames, how="vertical"))
            .with_row_index("_raw_row")
            .filter(pl.col("PHASE_mvt") == "DEP")
            .with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
            .filter(pl.col("month").is_in(wanted_train))
            .collect()
        )
        for period in periods:
            if not period.is_ranking:
                out[period] = train.filter(pl.col("month") == period.month)
        del train
    if any(p.is_ranking for p in periods):
        ranking = (
            with_reporting_airport(pl.scan_parquet(ranking_path(raw_dir)))
            .with_row_index("_raw_row")
            .filter(pl.col("PHASE_mvt") == "DEP")
            .with_columns(
                pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
                pl.col("MVT_TIME_UTC_mvt").dt.year().alias("_year"),
            )
            .collect()
        )
        for period in periods:
            if period.is_ranking:
                out[period] = ranking.filter(
                    (pl.col("_year") == period.year) & (pl.col("month") == period.month)
                ).drop("_year")
    return out


def build_base_period(dep: pl.DataFrame, metar: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """E08b+bunching features for one UTC month (``prepare_model_cache.build_cache`` semantics).

    Returns ``(features, labels)``.  Features never contain departure labels.
    """

    labels = dep.select("MVT_ID_mvt", LABEL_COLUMN, "month")
    masked = mask_departure_labels(dep, extra_cols=("AOBT_3_flt",)).drop(
        [LABEL_COLUMN, "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt"], strict=False
    )
    masked = add_delayed_bunching(masked, masked)
    masked = add_metar_features(masked, metar)
    masked = prepare_model_frame(masked, include_planning=True)
    for column in DELAY_BUNCH_NUMS:
        if column in masked.columns:
            masked = masked.with_columns(pl.col(column).fill_null(0))
    cats, nums = e34_feature_lists()
    for column in cats:
        masked = masked.with_columns(pl.col(column).fill_null(CAT_FILL).cast(pl.String))
    keep: list[str] = []
    for column in ("_raw_row", *BASE_ID_COLUMNS, *cats, *nums):
        if column in masked.columns and column not in keep:
            keep.append(column)
    features = (
        masked.select(keep)
        .unique("MVT_ID_mvt", keep="first", maintain_order=True)
        .sort("_raw_row")
        .drop("_raw_row")
    )
    assert_no_forbidden_columns(features.columns, label="base cache")
    labels = labels.join(features.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi")
    return features, labels


# ---------------------------------------------------------------------------
# Temporal context (contiguous scope)


def arr_context_rows(raw_dir: Path, periods: Sequence[Period]) -> pl.DataFrame:
    """ARR movements at challenge airports for the given UTC months (no labels)."""

    parts: list[pl.DataFrame] = []
    train_months = sorted(p.month for p in periods if not p.is_ranking)
    if train_months:
        frames = [pl.scan_parquet(path) for path in training_paths(raw_dir)]
        lf = (
            with_reporting_airport(pl.concat(frames, how="vertical"))
            .filter(pl.col("PHASE_mvt") == "ARR")
            .with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
            .filter(pl.col("month").is_in(train_months))
            .filter(pl.col("AIRPORT").is_in(list(CHALLENGE_AIRPORTS)))
        )
        parts.append(lf.select(list(ARR_CONTEXT_COLUMNS)).collect())
    for period in periods:
        if period.is_ranking:
            lf = (
                with_reporting_airport(raw_source_for_isolated_period(raw_dir, period))
                .filter(pl.col("PHASE_mvt") == "ARR")
                .with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
                .filter(pl.col("AIRPORT").is_in(list(CHALLENGE_AIRPORTS)))
            )
            parts.append(lf.select(list(ARR_CONTEXT_COLUMNS)).collect())
    arr = pl.concat(parts, how="vertical_relaxed") if parts else pl.DataFrame()
    assert_no_forbidden_columns(arr.columns, label="ARR context")
    return arr


def build_temporal_scope(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """R2 temporal-v2 context computed over one contiguous DEP/ARR table."""

    needed = [
        c
        for c in (
            "MVT_ID_mvt",
            "AIRPORT",
            "MVT_TIME_UTC_mvt",
            "takeoff_minus_sched_sec",
            "takeoff_minus_eobt_sec",
            "RUNWAY_mvt",
        )
        if c in dep.columns
    ]
    return build_temporal_context_v2(dep.select(needed), arr)


TEMPORAL_MODES = ("contiguous", "isocap_v1")


def build_temporal_period_isocap(
    dep: pl.DataFrame, arr: pl.DataFrame, period: Period, *, caps: Mapping[str, float] | None = None
) -> pl.DataFrame:
    """Temporal-v2 context of ONE isolated UTC month, lookback gaps horizon-censored (``isocap_v1``).

    ``dep`` must hold only this period's DEP rows and ``arr`` only this period's ARR rows (both
    by UTC month of ``MVT_TIME_UTC_mvt``); both are checked.  ``caps=None`` uses the frozen
    ``lookback_horizon.HORIZON_CAPS_V1``; ``caps={}`` skips censoring (the uncapped isolated build
    used only to derive the caps).  See ``prc_challenge.lookback_horizon``.
    """

    from prc_challenge import lookback_horizon as lh

    for name, frame in (("DEP", dep), ("ARR", arr)):
        if frame.height:
            ym = frame.select(
                (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).unique()
            ).to_series().to_list()
            if ym != [period.year * 100 + period.month]:
                raise RuntimeError(f"{name} rows of {period.key} span months {sorted(ym)}: not isolated")
    ctx = build_temporal_scope(dep, arr)
    if not ctx["MVT_ID_mvt"].equals(dep["MVT_ID_mvt"]):
        raise RuntimeError("temporal context changed the DEP row order")
    if caps is not None and not caps:
        return ctx
    t = dep.select(pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64)).to_series().to_numpy()
    start = np.full(dep.height, lh.month_start_epoch(period.year, period.month))
    return lh.apply_caps(ctx, t_epoch=t, month_start=start, columns=lh.TEMPORAL_LOOKBACK_COLUMNS,
                         caps=caps or lh.HORIZON_CAPS_V1)


def cap_native_lookbacks(native: pl.DataFrame, *, caps: Mapping[str, float] | None = None) -> pl.DataFrame:
    """Horizon-censor the native (already month-isolated) lookback columns (``isocap_v1``).

    The month start is the UTC calendar month of the departure's takeoff (``native_dep_epoch``).
    """

    from prc_challenge import lookback_horizon as lh

    t = native["native_dep_epoch"].cast(pl.Float64).to_numpy()
    start = native.select(
        pl.from_epoch(pl.col("native_dep_epoch").cast(pl.Int64), time_unit="s")
        .dt.replace_time_zone("UTC").dt.truncate("1mo").dt.epoch("s").cast(pl.Float64)
    ).to_series().to_numpy()
    return lh.apply_caps(native, t_epoch=t, month_start=start, columns=lh.NATIVE_LOOKBACK_COLUMNS,
                         caps=caps or lh.HORIZON_CAPS_V1)


# ---------------------------------------------------------------------------
# Native completed-arrival state and event tables (month-isolated)


def build_native_period(raw_dir: Path, period: Period) -> tuple[pl.DataFrame, pl.DataFrame]:
    source = raw_source_for_isolated_period(raw_dir, period)
    native, arrivals = build_native_completed_arrival_sidecar(source, month=period.month)
    native = native.drop("month")
    assert_no_forbidden_columns(native.columns, label="native sidecar")
    return native, arrivals


def permitted_event_input(source: pl.LazyFrame) -> pl.DataFrame:
    """Raw movement projection used by the historical event builder."""

    names = set(source.collect_schema().names())
    expressions: list[pl.Expr] = [pl.col(c) for c in EVENT_REQUIRED]
    expressions.extend(pl.col(c) for c in EVENT_OPTIONAL if c in names)
    if "BLOCK_TIME_UTC_mvt" in names:
        expressions.append(
            pl.when(pl.col("PHASE_mvt") == "ARR")
            .then(pl.col("BLOCK_TIME_UTC_mvt"))
            .otherwise(None)
            .alias("ARR_BLOCK_TIME_UTC")
        )
    frame = source.select(expressions).collect()
    reject_forbidden_departure_columns(frame)
    return frame


def build_events_period(raw_dir: Path, period: Period) -> pl.DataFrame:
    return build_event_table(
        permitted_event_input(raw_source_for_isolated_period(raw_dir, period)),
        source_month=period.month,
    )


# ---------------------------------------------------------------------------
# Fit-boundary references


def label_frame(base: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Rows for the S4 duration reference: every DEP label (matched or not)."""

    return base.select("MVT_ID_mvt", "AIRPORT", "MVT_TIME_UTC_mvt").join(
        labels.select("MVT_ID_mvt", LABEL_COLUMN), on="MVT_ID_mvt", how="inner", validate="1:1"
    ).select(
        pl.col("AIRPORT").alias("airport"),
        pl.col(LABEL_COLUMN).cast(pl.Float64).alias("duration_sec"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("event_epoch"),
    )


def reference_fingerprint(reference: DurationReference) -> str:
    """Order-independent identity of a duration reference."""

    payload = {
        "global": reference.global_duration_sec,
        "airport": dict(sorted((str(k), float(v)) for k, v in reference.airport_duration_sec.items())),
        "service": dict(sorted((str(k), float(v)) for k, v in reference.service_headway_sec.items())),
        "residuals_sorted_sha256": hashlib.sha256(
            np.sort(reference.residuals_sec.astype(np.float64)).tobytes()
        ).hexdigest(),
        "n_residuals": int(reference.residuals_sec.size),
    }
    return json_sha256(payload)


def s4_reference_months(target: Period, fit_months: Sequence[int], *, cross_fit: bool) -> tuple[int, ...]:
    """Historical rule: training month m uses fit months without m; evaluation uses all."""

    months = tuple(sorted(int(m) for m in fit_months))
    if cross_fit and not target.is_ranking and target.month in months:
        return tuple(m for m in months if m != target.month)
    return months


def build_s4_learned(events: pl.DataFrame, reference: DurationReference) -> pl.DataFrame:
    features = build_expected_occupancy_features(events, reference)
    return features.select("MVT_ID_mvt", *S4_EXPECTED_COLUMNS)


# ---------------------------------------------------------------------------
# Feature lists and parameters


def q1_module():
    """The exact Q1 unimpeded-baseline code used by the historical S4 primary."""

    from prc_challenge import q1_unimpeded as q1

    return q1


def base_feature_lists() -> tuple[list[str], list[str]]:
    """E08b cats, then E08b nums + temporal-v2 + Q1 baselines + native state, deduped in order."""

    cats, nums = campaign_feature_lists("e08b")
    nums = list(
        dict.fromkeys([*nums, *TEMPORAL_V2_NUMS, *q1_module().baseline_columns(), *NATIVE_STATE_FEATURES])
    )
    return list(cats), nums


def s4_feature_lists(
    extra_cats: Sequence[str] = (), extra_nums: Sequence[str] = ()
) -> tuple[list[str], list[str]]:
    cats, nums = base_feature_lists()
    nums = list(dict.fromkeys([*nums, *S4_EXPECTED_COLUMNS]))
    cats = list(dict.fromkeys([*cats, *extra_cats]))
    nums = list(dict.fromkeys([*nums, *extra_nums]))
    assert_no_forbidden_columns([*cats, *nums], label="model features")
    return cats, nums


def s4_params(
    t31_payload: Mapping[str, Any],
    *,
    iterations: int = HISTORICAL_ITERATIONS,
    threads: int = 4,
    seed: int = HISTORICAL_SEED,
) -> dict[str, Any]:
    """Exactly the historical primary parameters (``_params`` + ``booster_params``)."""

    params = optuna_winner_params(dict(t31_payload), thread_count=threads)
    params.pop("use_best_model", None)
    params["iterations"] = int(iterations)
    params["random_seed"] = int(seed)
    params["thread_count"] = int(threads)
    params["allow_writing_files"] = False
    params["verbose"] = False
    return params


def gpu_recipe(params: Mapping[str, Any], devices: str = "0", bootstrap: str = "MVS") -> dict[str, Any]:
    """The bake's GPU CatBoost recipe (D-081): a *different model* from the CPU t31 leg, gated separately.

    ``rsm`` is dropped (not supported on GPU for RMSE); ``border_count`` stays; bootstrap MVS, or Bernoulli
    at the t31 subsample rate.  Never used unless ``--task-type GPU`` is passed.
    """

    p = {k: v for k, v in params.items() if k != "rsm"}
    p.update(task_type="GPU", devices=str(devices))
    if bootstrap == "Bernoulli":
        p["bootstrap_type"] = "Bernoulli"
    return p


# ---------------------------------------------------------------------------
# Composition


def compose_janjul_primary(val: pl.DataFrame, raw: np.ndarray, parent: pl.DataFrame) -> pl.DataFrame:
    """Historical ``score_and_write`` composition for the 345.067 number.

    clip >= 0, then E17 -> E18 gates -> unmatched parent freeze -> leftover
    parent freeze -> clip >= 0, with the E33b splice as the frozen parent.
    """

    from prc_challenge.campaign.scoring import compose_with_parent

    scored = val.with_columns(pl.Series("prediction", np.asarray(raw, dtype=np.float64)).clip(lower_bound=0))
    scored = scored.join(
        parent.select("MVT_ID_mvt", pl.col("prediction").alias("parent_prediction")),
        on="MVT_ID_mvt",
        how="left",
    )
    if scored["parent_prediction"].null_count():
        raise RuntimeError("frozen parent does not cover every January/July row")
    return compose_with_parent(scored)


def compose_ranking_matched(ranking: pl.DataFrame, raw: np.ndarray) -> pl.DataFrame:
    """S4 ranking predictions with the matched-side composition order.

    E17 -> E18 gates, then clip >= 0.  Unmatched rows (``nm_unmatched == 1``,
    including every E17 and leftover row) are flagged ``owned_by_unmatched_overlay``;
    their value here is not used by any submission.
    """

    frame = ranking.with_columns(pl.Series("s4_raw_prediction", np.asarray(raw, dtype=np.float64)))
    frame = frame.with_columns(pl.col("s4_raw_prediction").alias("prediction"))
    frame = apply_lirf_stuck_block_gates(frame).with_columns(pl.col("prediction").clip(lower_bound=0))
    return frame.with_columns(
        lirf_delay_as_taxi_gate().fill_null(False).cast(pl.Int8).alias("gate_e17"),
        lirf_extraday_as_taxi_gate().fill_null(False).cast(pl.Int8).alias("gate_e18"),
        lirf_leftover_killer_gate().fill_null(False).cast(pl.Int8).alias("gate_leftover"),
        (pl.col("nm_unmatched") == 1).cast(pl.Int8).alias("owned_by_unmatched_overlay"),
    )


def v12_equivalent_parent(matched: pl.DataFrame, v12: pl.DataFrame) -> pl.DataFrame:
    """Rebuilt matched rows + v12 unmatched rows (v8 freeze), in template schema."""

    rebuilt = matched.select("MVT_ID_mvt", "prediction", "nm_unmatched")
    joined = v12.select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").alias("_v12")).join(
        rebuilt, on="MVT_ID_mvt", how="left", validate="1:1"
    )
    if joined["prediction"].null_count():
        raise RuntimeError("rebuilt ranking predictions do not cover the template")
    return joined.select(
        "MVT_ID_mvt",
        pl.when(pl.col("nm_unmatched") == 1)
        .then(pl.col("_v12"))
        .otherwise(pl.col("prediction"))
        .cast(pl.Float64)
        .alias("TAXITIME_SEC_mvt"),
    )


def score_block(frame: pl.DataFrame, pred_col: str = "prediction") -> dict[str, Any]:
    y = frame[LABEL_COLUMN].cast(pl.Float64).to_numpy()
    p = frame[pred_col].cast(pl.Float64).to_numpy()
    sse = float(np.sum((y - p) ** 2))
    return {
        "n": int(frame.height),
        "sse": sse,
        "rmse_seconds": float(np.sqrt(sse / frame.height)) if frame.height else None,
        "mae_seconds": float(np.mean(np.abs(y - p))) if frame.height else None,
        "bias_seconds": float(np.mean(p - y)) if frame.height else None,
    }


def janjul_scorecard(scored: pl.DataFrame) -> dict[str, Any]:
    card = {"JAN_JUL_2025": score_block(scored)}
    for month, name in ((1, "JAN_2025"), (7, "JUL_2025")):
        card[name] = score_block(scored.filter(pl.col("month") == month))
    card["matched_only"] = score_block(scored.filter(pl.col("nm_unmatched") == 0))
    card["unmatched_only"] = score_block(scored.filter(pl.col("nm_unmatched") == 1))
    return card


def rowwise_comparison(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = np.abs(a - b)
    return {
        "n": int(a.size),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else None,
        "mean_abs_diff": float(diff.mean()) if a.size else None,
        "median_abs_diff": float(np.median(diff)) if a.size else None,
        "p99_abs_diff": float(np.quantile(diff, 0.99)) if a.size else None,
        "max_abs_diff": float(diff.max()) if a.size else None,
        "n_bit_identical": int((a == b).sum()),
        "mean_a": float(a.mean()) if a.size else None,
        "mean_b": float(b.mean()) if a.size else None,
        "max_a": float(a.max()) if a.size else None,
        "max_b": float(b.max()) if a.size else None,
    }


__all__ = [name for name in globals() if not name.startswith("_")]
