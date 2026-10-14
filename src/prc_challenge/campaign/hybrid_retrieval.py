# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Leakage-aware utilities for the overnight hybrid and retrieval campaign."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


BLOCKS: tuple[tuple[int, int], ...] = ((2, 8), (3, 9), (4, 10), (5, 6))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    temporary.replace(path)


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError("truth and prediction shapes differ")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("metrics require finite truth and predictions")
    error = pred - truth
    sse = float(np.dot(error, error))
    return {
        "n": int(truth.size),
        "sse": sse,
        "rmse_seconds": float(np.sqrt(sse / truth.size)),
        "mae_seconds": float(np.mean(np.abs(error))),
        "bias_seconds": float(np.mean(error)),
    }


def pooled_score(parts: Iterable[tuple[np.ndarray, np.ndarray]]) -> dict[str, float | int]:
    n = 0
    sse = 0.0
    absolute = 0.0
    signed = 0.0
    for y, prediction in parts:
        truth = np.asarray(y, dtype=np.float64)
        pred = np.asarray(prediction, dtype=np.float64)
        if truth.shape != pred.shape:
            raise ValueError("truth and prediction shapes differ")
        error = pred - truth
        if not np.isfinite(error).all():
            raise ValueError("pooled metrics require finite predictions")
        n += int(error.size)
        sse += float(np.dot(error, error))
        absolute += float(np.abs(error).sum())
        signed += float(error.sum())
    if n <= 0:
        raise ValueError("pooled metrics require at least one row")
    return {
        "n": n,
        "sse": sse,
        "rmse_seconds": float(np.sqrt(sse / n)),
        "mae_seconds": absolute / n,
        "bias_seconds": signed / n,
    }


def fit_blend_weight(
    y: np.ndarray,
    native: np.ndarray,
    neural: np.ndarray,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    """Closed-form constrained weight for native + w * (neural - native)."""

    truth = np.asarray(y, dtype=np.float64)
    parent = np.asarray(native, dtype=np.float64)
    candidate = np.asarray(neural, dtype=np.float64)
    if not (truth.shape == parent.shape == candidate.shape):
        raise ValueError("blend arrays must have identical shapes")
    delta = candidate - parent
    denominator = float(np.dot(delta, delta))
    if denominator <= 0 or not np.isfinite(denominator):
        return float(lower)
    weight = float(np.dot(truth - parent, delta) / denominator)
    return float(np.clip(weight, lower, upper))


def apply_blend(native: np.ndarray, neural: np.ndarray, weight: float) -> np.ndarray:
    parent = np.asarray(native, dtype=np.float64)
    candidate = np.asarray(neural, dtype=np.float64)
    if parent.shape != candidate.shape:
        raise ValueError("blend arrays must have identical shapes")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("blend weight must be in [0, 1]")
    return parent + float(weight) * (candidate - parent)


def exact_uint64(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError("floating IDs are not finite integers")
        if np.any(np.abs(array) > 2**53):
            raise ValueError("floating IDs exceed exact integer range")
    result = array.astype(np.uint64)
    if np.issubdtype(array.dtype, np.floating) and not np.array_equal(
        result.astype(array.dtype), array
    ):
        raise ValueError("floating ID conversion is not lossless")
    return result


@dataclass(frozen=True)
class RetrievalIndexManifest:
    fold_id: str
    train_months: tuple[int, ...]
    feature_hash: str
    encoder_hash: str
    bank_id_hash: str
    teacher_recipe_hash: str

    def validate_query(
        self,
        *,
        fold_id: str,
        query_months: Sequence[int],
        feature_hash: str,
        encoder_hash: str,
        teacher_recipe_hash: str,
    ) -> None:
        if fold_id != self.fold_id:
            raise ValueError("retrieval index fold mismatch")
        if set(query_months).intersection(self.train_months):
            raise ValueError("query holdout months appear in retrieval bank")
        if feature_hash != self.feature_hash:
            raise ValueError("retrieval feature schema mismatch")
        if encoder_hash != self.encoder_hash:
            raise ValueError("retrieval encoder mismatch")
        if teacher_recipe_hash != self.teacher_recipe_hash:
            raise ValueError("retrieval teacher recipe mismatch")


def exclude_self_and_duplicates(
    neighbor_indices: np.ndarray,
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    *,
    query_duplicate_groups: np.ndarray | None = None,
    bank_duplicate_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Mask self IDs and known duplicate movement groups from neighbor results."""

    indices = np.asarray(neighbor_indices, dtype=np.int64)
    qids = exact_uint64(query_ids)
    bids = exact_uint64(bank_ids)
    if indices.ndim != 2 or indices.shape[0] != qids.size:
        raise ValueError("neighbor index rows must match query IDs")
    valid = (indices >= 0) & (indices < bids.size)
    safe = np.where(valid, indices, 0)
    valid &= bids[safe] != qids[:, None]
    if query_duplicate_groups is not None or bank_duplicate_groups is not None:
        if query_duplicate_groups is None or bank_duplicate_groups is None:
            raise ValueError("both query and bank duplicate groups are required")
        qgroups = np.asarray(query_duplicate_groups)
        bgroups = np.asarray(bank_duplicate_groups)
        if qgroups.size != qids.size or bgroups.size != bids.size:
            raise ValueError("duplicate group arrays do not match IDs")
        valid &= bgroups[safe] != qgroups[:, None]
    return valid


def shrunk_neighbor_mean(
    values: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    valid_mask: np.ndarray,
    *,
    prior: np.ndarray | float,
    shrinkage: float,
    distance_weighted: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a bounded local mean and effective neighbor support."""

    bank = np.asarray(values, dtype=np.float64)
    indices = np.asarray(neighbor_indices, dtype=np.int64)
    distances = np.asarray(neighbor_distances, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if not (indices.shape == distances.shape == valid.shape):
        raise ValueError("neighbor arrays must have identical shapes")
    safe = np.where(valid, indices, 0)
    weights = np.where(valid, 1.0, 0.0)
    if distance_weighted:
        weights = np.where(valid, 1.0 / np.maximum(distances, 1e-6), 0.0)
    support = weights.sum(axis=1)
    local_sum = (bank[safe] * weights).sum(axis=1)
    prior_array = np.broadcast_to(np.asarray(prior, dtype=np.float64), support.shape)
    denominator = support + float(shrinkage)
    prediction = np.where(
        denominator > 0,
        (local_sum + float(shrinkage) * prior_array) / denominator,
        prior_array,
    )
    return prediction, support


def validate_fold_ancestry(
    *, train_months: Sequence[int], holdout_months: Sequence[int], rows_months: np.ndarray
) -> None:
    train = set(int(x) for x in train_months)
    holdout = set(int(x) for x in holdout_months)
    if train.intersection(holdout):
        raise ValueError("train and holdout months overlap")
    present = set(int(x) for x in np.unique(rows_months))
    if not present.issubset(train):
        raise ValueError("teacher rows contain months outside declared training set")
