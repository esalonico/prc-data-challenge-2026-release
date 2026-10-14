# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Final-freeze guard and the months 11-12 feature path for the all-year unmatched refit (P3).

The blind refit of the A3b-nodest expert and P(copy) on 2025 months 1-12 is only for the
final ranking file. Months 11-12 (V4) are locked everywhere else, so:

- ``FinalFreezeGuard`` refuses locked months unless ``--final-freeze`` was passed, and it
  refuses any *evaluation* on locked months even then (``assert_no_evaluation``).
- ``load_movements_any`` is the only loader that can read locked raw months, and only
  through a checked guard.
- Months 11-12 working-table rows come from the S4-allyear rebuild ``base`` caches (verified
  identical to the historical E08b cache on months 1-10 in ``manifests/cache_verification.json``)
  plus the raw DEP sidecar columns and the label files. ``working_rows_from_base`` also serves
  months 1-10 so the dry run can check parity against the months 1-10 working table.
- ``dry_run_report`` fits nothing and writes nothing: row counts, label presence and parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from prc_challenge.io import reporting_airport_expr
from prc_challenge.unmatched_recovery.block_lineage import load_movements, month_paths
from prc_challenge.unmatched_recovery.schema import LOCKED_MONTHS, RAW_SIDECAR_COLS, ensure_unmatched_aux
from prc_challenge.unmatched_recovery.subgroups import attach_identity_columns

ALL_MONTHS = tuple(range(1, 13))
OCTOBER_SIDECAR_NOTE = (
    "Known defect of the historical working table: build_raw_sidecar skips the October file "
    "(training_2025-10-01_2025-11-01.parquet matches its '2025-11-01' exclusion), so October rows carry "
    "flight_prefix='__UNK__', ADES_mvt/FLIGHT_RULE_mvt null and ades_missing=1. The deployed A3b-nodest "
    "expert (v18) was trained with that quirk; the base-cache path carries the true values. "
    "--fix-october-sidecar replaces the October rows with the base-cache rows (a recipe deviation, off by default)."
)
S4_FEATURES_REL = Path("artifacts") / "experiments" / "S4-allyear-rebuild-20260925" / "features"


def parse_months(spec: str) -> tuple[int, ...]:
    """``'1-12'``, ``'1-10'``, ``'2,3,4'`` or ``'D'`` -> sorted month tuple."""

    text = spec.strip().upper()
    if text == "D":
        return (2, 3, 4, 5, 6, 8, 9, 10)
    months: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            months.update(range(int(lo), int(hi) + 1))
        else:
            months.add(int(part))
    out = tuple(sorted(months))
    if not out or any(m not in ALL_MONTHS for m in out):
        raise ValueError(f"months outside 1-12 in {spec!r}")
    return out


@dataclass(frozen=True)
class FinalFreezeGuard:
    train_months: tuple[int, ...]
    final_freeze: bool = False
    dry_run: bool = False

    @property
    def locked_months(self) -> tuple[int, ...]:
        return tuple(m for m in self.train_months if m in LOCKED_MONTHS)

    @property
    def uses_locked(self) -> bool:
        return bool(self.locked_months)

    def check(self) -> None:
        bad = [m for m in self.train_months if m not in ALL_MONTHS]
        if bad:
            raise ValueError(f"training months outside 2025: {bad}")
        if self.uses_locked and not self.final_freeze:
            raise RuntimeError(f"months {list(self.locked_months)} are locked (V4); pass --final-freeze for the blind all-year refit")

    def allow_locked_load(self, context: str) -> None:
        """Called by every loader that can touch months 11-12."""

        self.check()
        if not self.final_freeze:
            raise RuntimeError(f"{context}: locked months without --final-freeze")

    def assert_no_evaluation(self, months: Sequence[int], context: str = "evaluation") -> None:
        """Scoring on months 11-12 is refused in every mode, final freeze included."""

        locked = sorted(set(int(m) for m in months) & set(LOCKED_MONTHS))
        if locked:
            raise RuntimeError(f"{context}: months {locked} are blind training rows only; no evaluation on V4 months")

    def describe(self) -> dict[str, Any]:
        return {
            "train_months": list(self.train_months),
            "locked_months_used": list(self.locked_months),
            "final_freeze": self.final_freeze,
            "dry_run": self.dry_run,
            "nov_dec_role": "blind final training rows only" if self.uses_locked else "unused",
        }


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_movements_any(raw_dir: Path, months: Sequence[int], phase: str, guard: FinalFreezeGuard) -> pl.DataFrame:
    """``load_movements`` for months 1-10; locked months only through the guard (final freeze)."""

    wanted = tuple(int(m) for m in months)
    open_months = tuple(m for m in wanted if m not in LOCKED_MONTHS)
    locked = tuple(m for m in wanted if m in LOCKED_MONTHS)
    parts: list[pl.DataFrame] = []
    if open_months:
        parts.append(load_movements(raw_dir, open_months, phase))
    if locked:
        guard.allow_locked_load(f"load_movements_any({phase}, {list(locked)})")
        frames = [pl.scan_parquet(path) for path in month_paths(raw_dir, locked)]
        frame = pl.concat(frames, how="vertical_relaxed").filter(pl.col("PHASE_mvt") == phase).collect()
        frame = frame.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"), reporting_airport_expr())
        parts.append(frame.filter(pl.col("month").is_in(list(locked))))
    if not parts:
        raise ValueError("no months requested")
    return pl.concat(parts, how="vertical_relaxed")


def base_path(features_dir: Path, month: int) -> Path:
    return features_dir / "base" / f"base_2025-{month:02d}.parquet"


def labels_path(features_dir: Path, month: int) -> Path:
    return features_dir / "labels" / f"labels_2025-{month:02d}.parquet"


def raw_dep_sidecar(raw_dir: Path, months: Sequence[int], guard: FinalFreezeGuard) -> pl.DataFrame:
    """The raw DEP columns the reduced cache lacks (``RAW_SIDECAR_COLS``), for the given months.

    Lazy column projection (the full DEP frame is never materialised). Locked months need the guard.
    """

    wanted = tuple(int(m) for m in months)
    locked = tuple(m for m in wanted if m in LOCKED_MONTHS)
    if locked:
        guard.allow_locked_load(f"raw_dep_sidecar({list(locked)})")
    frames = [pl.scan_parquet(path) for path in month_paths(raw_dir, wanted)]
    lf = pl.concat(frames, how="vertical_relaxed").filter(pl.col("PHASE_mvt") == "DEP")
    names = lf.collect_schema().names()
    cols = [c for c in RAW_SIDECAR_COLS if c in names]
    if "MVT_TIME_UTC_mvt" not in cols:
        cols.append("MVT_TIME_UTC_mvt")
    frame = lf.select(cols).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month")).filter(pl.col("month").is_in(list(wanted))).collect()
    if not locked and frame.filter(pl.col("month").is_in(list(LOCKED_MONTHS))).height:
        raise RuntimeError("V4 rows survived the sidecar filter")
    return frame.drop("month", "MVT_TIME_UTC_mvt")


def working_rows_from_base(features_dir: Path, raw_dir: Path, months: Sequence[int], guard: FinalFreezeGuard, *, with_labels: bool = True) -> pl.DataFrame:
    """Unmatched working-table rows for ``months`` from the S4-allyear base caches + raw sidecar (+ labels).

    Same shape as ``FrozenCampaign.load_table`` restricted to unmatched rows: base columns,
    ``RAW_SIDECAR_COLS``, identity columns, missingness flags. Locked months need the guard.
    """

    months = tuple(int(m) for m in months)
    if set(months) & set(LOCKED_MONTHS):
        guard.allow_locked_load(f"working_rows_from_base({list(months)})")
    missing = [str(base_path(features_dir, m)) for m in months if not base_path(features_dir, m).exists()]
    if missing:
        raise FileNotFoundError(f"base caches missing: {missing}")
    base = pl.concat([pl.read_parquet(base_path(features_dir, m)) for m in months], how="vertical_relaxed")
    base = base.filter(pl.col("nm_unmatched") == 1)
    sidecar = raw_dep_sidecar(raw_dir, months, guard)
    extra = [c for c in sidecar.columns if c not in base.columns or c == "MVT_ID_mvt"]
    out = base.join(sidecar.select(extra), on="MVT_ID_mvt", how="left")
    if out.height != base.height:
        raise RuntimeError("raw sidecar join changed the row count")
    if with_labels:
        labels = pl.concat([pl.read_parquet(labels_path(features_dir, m)).select("MVT_ID_mvt", "TAXITIME_SEC_mvt") for m in months], how="vertical_relaxed")
        out = out.join(labels, on="MVT_ID_mvt", how="left")
        if out.height != base.height:
            raise RuntimeError("label join changed the row count")
    out = ensure_unmatched_aux(out)
    return attach_identity_columns(out)


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------


def column_mismatches(a: pl.DataFrame, b: pl.DataFrame, cols: Sequence[str]) -> dict[str, int]:
    """Rows where ``a[c] != b[c]`` (nulls compare equal to nulls). Frames must be aligned."""

    out: dict[str, int] = {}
    for c in cols:
        x, y = a[c], b[c]
        if x.dtype in (pl.Utf8, pl.String, pl.Categorical) or y.dtype in (pl.Utf8, pl.String, pl.Categorical):
            out[c] = int((x.cast(pl.String).fill_null("__NULL__") != y.cast(pl.String).fill_null("__NULL__")).sum())
        else:
            out[c] = int((x.cast(pl.Float64).fill_null(-9e18) != y.cast(pl.Float64).fill_null(-9e18)).sum())
    return out


def dry_run_report(
    features_dir: Path,
    raw_dir: Path,
    guard: FinalFreezeGuard,
    *,
    reference_fit: pl.DataFrame | None,
    feature_cols: Sequence[str],
) -> dict[str, Any]:
    """Row counts per month (DEP / unmatched / label presence), raw ARR counts, and months 1-10 parity.

    Fits nothing, writes nothing, reads no label value (only null counts).
    """

    guard.check()
    per_month: dict[str, Any] = {}
    for month in guard.train_months:
        base = pl.read_parquet(base_path(features_dir, month), columns=["MVT_ID_mvt", "nm_unmatched"])
        labels = pl.read_parquet(labels_path(features_dir, month), columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
        unmatched_ids = base.filter(pl.col("nm_unmatched") == 1).select("MVT_ID_mvt")
        lab_u = unmatched_ids.join(labels, on="MVT_ID_mvt", how="left")
        per_month[str(month)] = {
            "dep_rows": int(base.height),
            "unmatched_rows": int(unmatched_ids.height),
            "unmatched_label_present": int(lab_u["TAXITIME_SEC_mvt"].is_not_null().sum()),
            "locked": month in LOCKED_MONTHS,
        }
    arr_counts: dict[str, int] = {}
    if guard.uses_locked:
        arr = load_movements_any(raw_dir, guard.locked_months, "ARR", guard)
        for month in guard.locked_months:
            arr_counts[str(month)] = int(arr.filter(pl.col("month") == month).height)
    parity: dict[str, Any] = {"checked": False}
    open_months = tuple(m for m in guard.train_months if m not in LOCKED_MONTHS)
    if reference_fit is not None and open_months:
        mine = working_rows_from_base(features_dir, raw_dir, open_months, guard)
        ref = reference_fit.filter(pl.col("month").is_in(list(open_months)))
        cols = [c for c in feature_cols if c in mine.columns and c in ref.columns]
        absent = [c for c in feature_cols if c not in mine.columns]
        joined = ref.select("MVT_ID_mvt", "month", *cols).join(mine.select("MVT_ID_mvt", *[pl.col(c).alias(f"{c}__mine") for c in cols]), on="MVT_ID_mvt", how="inner")
        mism = column_mismatches(joined.select([pl.col(f"{c}__mine").alias(c) for c in cols]), joined.select(cols), cols)
        per_month_mism: dict[str, dict[str, int]] = {}
        for month in open_months:
            sub = joined.filter(pl.col("month") == month)
            counts = column_mismatches(sub.select([pl.col(f"{c}__mine").alias(c) for c in cols]), sub.select(cols), cols)
            per_month_mism[str(month)] = {c: n for c, n in counts.items() if n}
        months_clean = [m for m in open_months if not per_month_mism[str(m)]]
        parity = {
            "checked": True,
            "months": list(open_months),
            "reference_rows": int(ref.height),
            "base_path_rows": int(mine.height),
            "joined_rows": int(joined.height),
            "absent_columns": absent,
            "mismatches": mism,
            "mismatches_per_month": per_month_mism,
            "months_identical": months_clean,
            "ok": bool(joined.height == ref.height == mine.height and not absent and not any(mism.values())),
            "note": OCTOBER_SIDECAR_NOTE if "10" in per_month_mism and per_month_mism["10"] else "",
        }
    total_u = sum(v["unmatched_rows"] for v in per_month.values())
    return {
        "guard": guard.describe(),
        "per_month": per_month,
        "locked_arr_rows": arr_counts,
        "training_unmatched_rows_total": total_u,
        "parity_months_1_10": parity,
        "fitted": False,
        "ranking_file_written": False,
    }
