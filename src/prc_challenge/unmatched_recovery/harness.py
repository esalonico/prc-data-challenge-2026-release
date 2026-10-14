# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Freeze-free unmatched submission harness.

Matched rows stay on the parent submission. Only ``nm_unmatched == 1``
predictions may change. Ranking scores are never inputs to recipes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from prc_challenge.s3 import TEAM_NAME
from prc_challenge.submission import validate_submission
from prc_challenge.unmatched_recovery.schema import compose_experimental_unmatched, splice_unmatched

VERSION_RE = re.compile(rf"^{re.escape(TEAM_NAME)}_v(\d+)\.parquet$")
DAILY_CAP = 5
# final phase: the serve set comes from the environment (as s4_allyear.RANKING_N); defaults = public Jan/Jul file
RANKING_N = int(os.environ.get("PRC_RANKING_N", "344841"))
RANKING_UNMATCHED_N = int(os.environ.get("PRC_RANKING_UNMATCHED_N", "5283"))
PRIMARY_N = 344_419
PRIMARY_UNMATCHED_N = 5_366


def _sorted_pairs(frame: pl.DataFrame, value_col: str) -> pl.DataFrame:
    if "MVT_ID_mvt" not in frame.columns or value_col not in frame.columns:
        raise RuntimeError(f"need MVT_ID_mvt and {value_col}")
    return frame.select("MVT_ID_mvt", pl.col(value_col).cast(pl.Float64)).sort("MVT_ID_mvt")


def payload_sha256(frame: pl.DataFrame, value_col: str) -> str:
    """Stable SHA-256 of sorted id/value pairs. Bit-identical rows hash equally."""

    pairs = _sorted_pairs(frame, value_col)
    digest = hashlib.sha256()
    digest.update(pairs["MVT_ID_mvt"].to_numpy().tobytes())
    digest.update(np.asarray(pairs[value_col].to_numpy(), dtype=np.float64).tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlay_unmatched_submission(
    parent: pl.DataFrame,
    unmatched: pl.DataFrame,
    *,
    pred_col: str = "prediction",
    value_col: str = "TAXITIME_SEC_mvt",
) -> pl.DataFrame:
    """Replace parent values only on unmatched IDs. Matched/ARR rows stay bit-identical."""

    parent = parent.select("MVT_ID_mvt", value_col)
    cand = unmatched.select("MVT_ID_mvt", pl.col(pred_col).cast(pl.Float64).alias("_u"))
    if cand["MVT_ID_mvt"].n_unique() != cand.height:
        raise RuntimeError("duplicate unmatched overlay IDs")
    joined = parent.join(cand, on="MVT_ID_mvt", how="left")
    if joined.height != parent.height:
        raise RuntimeError("unmatched overlay changed row count")
    found = int(joined.filter(pl.col("_u").is_not_null()).height)
    if found != cand.height:
        raise RuntimeError(f"overlay IDs missing from parent: have {found} want {cand.height}")
    out = joined.with_columns(
        pl.when(pl.col("_u").is_not_null()).then(pl.col("_u")).otherwise(pl.col(value_col)).alias(value_col)
    ).drop("_u")
    return out.select("MVT_ID_mvt", value_col)


def matched_view(submission: pl.DataFrame, unmatched_ids: Sequence[float] | pl.Series, value_col: str = "TAXITIME_SEC_mvt") -> pl.DataFrame:
    ids = pl.DataFrame({"MVT_ID_mvt": list(unmatched_ids) if not isinstance(unmatched_ids, pl.Series) else unmatched_ids})
    return submission.join(ids, on="MVT_ID_mvt", how="anti")


def unmatched_view(submission: pl.DataFrame, unmatched_ids: Sequence[float] | pl.Series, value_col: str = "TAXITIME_SEC_mvt") -> pl.DataFrame:
    ids = pl.DataFrame({"MVT_ID_mvt": list(unmatched_ids) if not isinstance(unmatched_ids, pl.Series) else unmatched_ids})
    return submission.join(ids, on="MVT_ID_mvt", how="inner")


def verify_arm_submission(
    submission: pl.DataFrame,
    *,
    template: pl.DataFrame,
    parent: pl.DataFrame,
    unmatched_ids: Sequence[float] | pl.Series,
    expected_unmatched_n: int = RANKING_UNMATCHED_N,
    expected_n: int = RANKING_N,
) -> dict[str, Any]:
    """Full-row, matched-equality, unmatched-count, finite, ID checks."""

    errors = list(validate_submission(submission, template=template))
    n = int(submission.height)
    if n != expected_n:
        errors.append(f"row count {n} != {expected_n}")
    y = np.asarray(submission["TAXITIME_SEC_mvt"].to_numpy(), dtype=np.float64)
    finite = bool(np.isfinite(y).all())
    if not finite:
        errors.append("non-finite predictions")
    unmatched = unmatched_view(submission, unmatched_ids)
    matched = matched_view(submission, unmatched_ids)
    if unmatched.height != expected_unmatched_n:
        errors.append(f"unmatched count {unmatched.height} != {expected_unmatched_n}")
    parent_matched = matched_view(parent.select("MVT_ID_mvt", "TAXITIME_SEC_mvt"), unmatched_ids)
    matched_join = matched.join(
        parent_matched.rename({"TAXITIME_SEC_mvt": "parent"}),
        on="MVT_ID_mvt",
        how="left",
    )
    matched_diff = (matched_join["TAXITIME_SEC_mvt"] - matched_join["parent"]).abs()
    max_matched = float(matched_diff.max() or 0.0)
    n_matched_changed = int((matched_diff > 0).sum())
    if n_matched_changed:
        errors.append(f"matched rows changed vs parent: {n_matched_changed} max_abs={max_matched}")
    payload = {
        "n": n,
        "unmatched_n": int(unmatched.height),
        "matched_n": int(matched.height),
        "finite": finite,
        "nulls": int(submission["TAXITIME_SEC_mvt"].null_count()),
        "matched_checksum": payload_sha256(matched, "TAXITIME_SEC_mvt"),
        "unmatched_checksum": payload_sha256(unmatched, "TAXITIME_SEC_mvt"),
        "parent_matched_checksum": payload_sha256(parent_matched, "TAXITIME_SEC_mvt"),
        "matched_changed": n_matched_changed,
        "matched_max_abs_diff": max_matched,
        "pred_min": float(np.min(y)) if y.size else None,
        "pred_max": float(np.max(y)) if y.size else None,
        "errors": errors,
        "ok": not errors,
    }
    if payload["matched_checksum"] != payload["parent_matched_checksum"]:
        payload["ok"] = False
        payload["errors"] = [*errors, "matched checksum != parent matched checksum"]
    return payload


def assert_cross_arm_matched_identical(checksums: Mapping[str, str]) -> str:
    values = sorted(set(checksums.values()))
    if len(values) != 1:
        raise RuntimeError(f"matched checksums differ across arms: {dict(checksums)}")
    return values[0]


def daily_quota(
    remote_objects: Iterable[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    daily_cap: int = DAILY_CAP,
) -> dict[str, Any]:
    """Count this UTC day's team parquet uploads. Does not read scores."""

    now = now or datetime.now(timezone.utc)
    today = now.astimezone(timezone.utc).date().isoformat()
    used_today: list[dict[str, Any]] = []
    all_versions: list[int] = []
    for obj in remote_objects:
        key = str(obj.get("Key") or obj.get("key") or "")
        name = Path(key).name
        match = VERSION_RE.match(name)
        if not match:
            continue
        version = int(match.group(1))
        all_versions.append(version)
        modified = obj.get("LastModified") or obj.get("last_modified")
        if modified is None:
            continue
        if hasattr(modified, "astimezone"):
            day = modified.astimezone(timezone.utc).date().isoformat()
        else:
            day = str(modified)[:10]
        if day == today:
            used_today.append({"key": name, "version": version, "utc_date": day})
    next_version = (max(all_versions) + 1) if all_versions else 1
    remaining = max(0, daily_cap - len(used_today))
    return {
        "utc_date": today,
        "daily_cap": daily_cap,
        "used_today": len(used_today),
        "remaining_today": remaining,
        "today_keys": [row["key"] for row in sorted(used_today, key=lambda r: r["version"])],
        "next_version": next_version,
        "remote_parquet_versions": sorted(all_versions),
        "ok_to_upload": remaining > 0,
    }


def prepare_unmatched_preds(
    rows: pl.DataFrame,
    pred: np.ndarray | pl.Series,
    *,
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Clip and apply E17/E18 on unmatched rows only. Does not restore the parent freeze."""

    out = rows.with_columns(pl.Series(pred_col, np.asarray(pred, dtype=np.float64)))
    return compose_experimental_unmatched(out, pred_col=pred_col)


def splice_primary(parent: pl.DataFrame, unmatched: pl.DataFrame) -> pl.DataFrame:
    if "parent_prediction" not in parent.columns and "prediction" in parent.columns:
        parent = parent.with_columns(pl.col("prediction").alias("parent_prediction"))
    spliced = splice_unmatched(parent, unmatched)
    return compose_experimental_unmatched(spliced)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
