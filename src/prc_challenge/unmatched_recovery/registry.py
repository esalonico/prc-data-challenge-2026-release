# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Atomic experiment registry and campaign state."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prc_challenge.campaign.hybrid_retrieval import atomic_json, sha256_file, stable_hash

SCHEMA_VERSION = "unmatched-recovery-state-v1"
VALID_STATUSES = (
    "pending",
    "running",
    "completed",
    "parked",
    "blocked",
    "failed_implementation",
)


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def git_branch(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()


def code_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def empty_state(repo: Path, output: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "branch": git_branch(repo),
        "source_commit": git_head(repo),
        "host": platform.node(),
        "pid": os.getpid(),
        "created_utc": now_utc(),
        "updated_utc": now_utc(),
        "output": str(output),
        "active_stage": None,
        "status": "pending",
        "jobs": {},
        "finalists": [],
        "winner": None,
    }


def empty_registry() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": now_utc(),
        "experiments": {},
        "stages": {},
    }


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return default


def write_state(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_utc"] = now_utc()
    atomic_json(path, payload)


def record_experiment(
    registry: dict[str, Any],
    experiment_id: str,
    *,
    status: str,
    stage: str,
    imputation_strategy: str,
    downstream_strategy: str,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    outputs: list[Path] | None = None,
    decision: str | None = None,
    error: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status}")
    entry: dict[str, Any] = {
        "id": experiment_id,
        "status": status,
        "stage": stage,
        "imputation_strategy": imputation_strategy,
        "downstream_strategy": downstream_strategy,
        "config": config,
        "config_hash": stable_hash(config),
        "metrics": metrics or {},
        "decision": decision,
        "error": error,
        "provenance": provenance or {"source_mode": "native"},
        "updated_utc": now_utc(),
        "outputs": [],
    }
    for path in outputs or []:
        if path.exists() and path.is_file():
            entry["outputs"].append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    registry.setdefault("experiments", {})[experiment_id] = entry
    return entry
