# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Authorized OpenSky S3 access. Credentials stay on disk; never log them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

DEFAULT_ENDPOINT = "https://s3.opensky-network.org"
DEFAULT_CREDS_PATH = Path("minio-s3-credentials.json")
DATA_BUCKET = "prc-2026-datasets"
TEAM_BUCKET = "prc-2026-unbelievable-snowflake"
TEAM_NAME = "unbelievable-snowflake"

EXPECTED_OBJECTS = [
    "ranking.parquet",
    "submitting.parquet",
    "final_ranking.parquet",  # final phase: Jan, Feb, Jun, Jul 2026 (scored blind)
    "final_submitting.parquet",
    "training_2025-01-01_2025-02-01.parquet",
    "training_2025-02-01_2025-03-01.parquet",
    "training_2025-03-01_2025-04-01.parquet",
    "training_2025-04-01_2025-05-01.parquet",
    "training_2025-05-01_2025-06-01.parquet",
    "training_2025-06-01_2025-07-01.parquet",
    "training_2025-07-01_2025-08-01.parquet",
    "training_2025-08-01_2025-09-01.parquet",
    "training_2025-09-01_2025-10-01.parquet",
    "training_2025-10-01_2025-11-01.parquet",
    "training_2025-11-01_2025-12-01.parquet",
    "training_2025-12-01_2026-01-01.parquet",
]


def load_credentials(path: Path | str = DEFAULT_CREDS_PATH) -> dict[str, Any]:
    creds = json.loads(Path(path).read_text())
    missing = [k for k in ("accessKey", "secretKey") if k not in creds]
    if missing:
        raise ValueError(f"credentials JSON missing keys: {missing}")
    return creds


def s3_client(creds_path: Path | str = DEFAULT_CREDS_PATH, endpoint: str = DEFAULT_ENDPOINT):
    creds = load_credentials(creds_path)
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=creds["accessKey"],
        aws_secret_access_key=creds["secretKey"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def list_objects(client, bucket: str) -> list[dict[str, Any]]:
    paginator = client.get_paginator("list_objects_v2")
    objects: list[dict[str, Any]] = []
    try:
        for page in paginator.paginate(Bucket=bucket):
            objects.extend(page.get("Contents") or [])
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "unknown")
        raise RuntimeError(f"cannot list bucket {bucket!r} ({code})") from exc
    return objects


def download_object(
    client,
    bucket: str,
    key: str,
    dest: Path,
    expected_size: int | None = None,
) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        checksum = sha256_file(dest)
        return {
            "bucket": bucket,
            "key": key,
            "path": str(dest),
            "bytes": dest.stat().st_size,
            "sha256": checksum,
            "skipped": True,
        }
    tmp = dest.with_suffix(dest.suffix + ".partial")
    client.download_file(bucket, key, str(tmp))
    size = tmp.stat().st_size
    if expected_size is not None and size != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"size mismatch for {key}: got {size}, expected {expected_size}")
    tmp.replace(dest)
    return {
        "bucket": bucket,
        "key": key,
        "path": str(dest),
        "bytes": size,
        "sha256": sha256_file(dest),
        "skipped": False,
    }


def download_challenge_data(
    dest_dir: Path | str = "data/raw",
    creds_path: Path | str = DEFAULT_CREDS_PATH,
    endpoint: str = DEFAULT_ENDPOINT,
) -> list[dict[str, Any]]:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    client = s3_client(creds_path, endpoint)
    remote = {obj["Key"]: obj for obj in list_objects(client, DATA_BUCKET)}
    missing = [key for key in EXPECTED_OBJECTS if key not in remote]
    if missing:
        raise RuntimeError(f"missing expected objects in {DATA_BUCKET}: {missing}")
    results = []
    for key in EXPECTED_OBJECTS:
        results.append(
            download_object(
                client,
                DATA_BUCKET,
                key,
                dest / key,
                expected_size=int(remote[key]["Size"]),
            )
        )
    return results
