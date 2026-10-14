# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Locate, verify, and (optionally) hydrate campaign files from the Colab Data Link.

Does not copy secrets, data/raw, or V4 labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from prc_challenge.campaign.runtime import FORBIDDEN_FEATURE_COLS, LOCKED_MONTHS, CampaignConfig
from prc_challenge.campaign.scoring import assert_no_forbidden_features

CACHE_REL = Path("artifacts/cache/e08b_e34_months_1_10.parquet")
CACHE_JSON_REL = Path("artifacts/cache/e08b_e34_months_1_10.json")
MANIFEST_NAME = "MANIFEST.json"

REQUIRED_PREDS = [
    Path("artifacts/experiments/E33b-matched-splice_V1/val_predictions.parquet"),
    Path("artifacts/experiments/E33b-matched-splice_V2/val_predictions.parquet"),
    Path("artifacts/experiments/E33b-matched-splice_V3/val_predictions.parquet"),
    Path("artifacts/experiments/E37-eval-e08b-t6_V1/val_predictions.parquet"),
    Path("artifacts/experiments/E37-eval-e08b-t6_V2/val_predictions.parquet"),
    Path("artifacts/experiments/E37-eval-e08b-t6_V3/val_predictions.parquet"),
]
OPTIONAL_PREDS = [
    Path("artifacts/experiments/E37-eval-e08b-t31_V1/val_predictions.parquet"),
    Path("artifacts/experiments/E37-eval-e08b-t31_V2/val_predictions.parquet"),
    Path("artifacts/experiments/E37-eval-e08b-t31_V3/val_predictions.parquet"),
    Path("artifacts/experiments/E34b-bunching-gated_V1/val_predictions.parquet"),
    Path("artifacts/experiments/E34b-bunching-gated_V2/val_predictions.parquet"),
    Path("artifacts/experiments/E34b-bunching-gated_V3/val_predictions.parquet"),
]
OPTIONAL_SIDECARS = [
    Path("artifacts/experiments/E37-optuna/shortlist_t31.json"),
    Path("artifacts/cache/e08b_e34_months_1_10.json"),
]
OPTIONAL_CACHES = [
    Path("artifacts/cache/f3_airport_state_months_1_10.parquet"),
    Path("artifacts/cache/arr_context_months_1_10.parquet"),
    Path("artifacts/cache/base_oof_V1_official.parquet"),
    Path("artifacts/cache/link_ids_months_1_10.parquet"),
]
SKIP_RESULT_SUFFIX = {".cbm", ".onnx", ".pkl", ".npy", ".db-wal", ".db-shm"}
SKIP_RESULT_NAMES = {"model.pt"}
JOB_INDEX_NAMES = {"campaign_index.json", "campaign_summary.json"}


def is_job_result_rel(rel: str | Path) -> bool:
    """CPU/GPU job artifacts. Mac Drive sync / SSH push must not overwrite a live run."""
    parts = Path(rel).parts
    if len(parts) < 3 or parts[0] != "artifacts" or parts[1] != "experiments":
        return False
    name = parts[2]
    return name.startswith(("CPU-", "GPU-")) or name in JOB_INDEX_NAMES
GEOMETRY = [
    Path("data/external/osm/parking_positions.parquet"),
    Path("data/external/osm/manifest.json"),
    Path("data/external/ourairports/runways.csv"),
]
CODE_REL = [
    Path("src/prc_challenge"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("configs/validation.yaml"),
    Path("notebooks/next_experiments_cpu.ipynb"),
    Path("notebooks/next_experiments_gpu.ipynb"),
    Path("scripts/campaign_doctor.py"),
    Path("scripts/run_next_experiments.py"),
    Path("scripts/remote_campaign.py"),
    Path("scripts/ingest_campaign_results.py"),
    Path("scripts/sync_colab_data_link.py"),
]


def existing_campaign_result_rels(root: Path) -> list[Path]:
    """CPU/GPU job artifacts already on `root` (predictions, metrics, F3/OOF caches). No weights."""
    out: list[Path] = []
    for rel in OPTIONAL_CACHES:
        if (root / rel).exists():
            out.append(rel)
    cache_dir = root / "artifacts" / "cache"
    if cache_dir.exists():
        for pattern in ("f3_month_*.parquet", "base_oof_*_month_*.parquet", "seq_*"):
            for path in sorted(cache_dir.glob(pattern)):
                if path.suffix in SKIP_RESULT_SUFFIX:
                    continue
                rel = path.relative_to(root)
                if rel not in out:
                    out.append(rel)
        for path in sorted(cache_dir.rglob("oof_month_*.parquet")):
            rel = path.relative_to(root)
            if rel not in out:
                out.append(rel)
    exp = root / "artifacts" / "experiments"
    if not exp.exists():
        return out
    for name in ("campaign_index.json", "campaign_summary.json"):
        path = exp / name
        if path.exists():
            out.append(path.relative_to(root))
    for folder in sorted(exp.iterdir()):
        if not folder.is_dir():
            continue
        if not (folder.name.startswith("CPU-") or folder.name.startswith("GPU-")):
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if path.name in SKIP_RESULT_NAMES or path.suffix in SKIP_RESULT_SUFFIX:
                continue
            out.append(path.relative_to(root))
    return out


def campaign_bundle_rels(repo_root: Path | None = None) -> list[Path]:
    """Ranking-safe files to copy to a remote Jupyter host. No secrets, no data/raw, no V4."""
    rels = [CACHE_REL, CACHE_JSON_REL, *REQUIRED_PREDS, *OPTIONAL_PREDS, *OPTIONAL_SIDECARS, *GEOMETRY, *OPTIONAL_CACHES]
    if repo_root is not None:
        rels = [*rels, *existing_campaign_result_rels(Path(repo_root))]
    seen: set[str] = set()
    unique: list[Path] = []
    for rel in rels:
        key = str(rel)
        if key in seen:
            continue
        seen.add(key)
        unique.append(Path(rel))
    return unique

NEVER_SYNC = (
    ".env",
    ".env.local",
    "minio-s3-credentials.json",
    "secrets",
)


def default_drive_link() -> Path | None:
    env = os.environ.get("PRC_COLAB_DATA_LINK")
    if env:
        path = Path(env)
        if path.exists():
            return path
    candidates = [
        Path("/content/drive/MyDrive/Colab Data Link"),
        Path("/content/drive/My Drive/Colab Data Link"),
    ]
    cloud = Path.home() / "Library/CloudStorage"
    if cloud.exists():
        candidates.extend(sorted(cloud.glob("GoogleDrive-*/My Drive/Colab Data Link")))
    for path in candidates:
        if path.exists():
            return path
    return None


def candidate_roots(cfg: CampaignConfig | None = None) -> list[Path]:
    roots: list[Path] = []
    if cfg is not None:
        roots.extend([cfg.repo_root, cfg.cache_root.parent.parent, cfg.artifact_root.parent.parent])
        if cfg.drive_link is not None:
            roots.append(Path(cfg.drive_link))
    drive = default_drive_link()
    if drive is not None:
        roots.append(drive)
    roots.extend(
        [
            Path("/content/prc-challenge-2026"),
            Path("/content/drive/MyDrive/prc-challenge-2026"),
            Path("/content/drive/MyDrive/Colab Data Link"),
        ]
    )
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(root)
    return out


def locate(rel: str | Path, cfg: CampaignConfig | None = None) -> Path | None:
    rel = Path(rel)
    for root in candidate_roots(cfg):
        path = root / rel
        if path.exists():
            return path
        if rel.name and (root / rel.name).exists() and rel.suffix in {".parquet", ".json"}:
            return root / rel.name
    return None


@dataclass
class FileStatus:
    rel: str
    required: bool
    present: bool
    path: str | None = None
    bytes: int | None = None
    note: str = ""


@dataclass
class DataCheckReport:
    ok: bool
    drive_link: str | None
    files: list[FileStatus] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "drive_link": self.drive_link,
            "files": [f.__dict__ for f in self.files],
            "cache": self.cache,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _stat(rel: Path, required: bool, cfg: CampaignConfig | None) -> FileStatus:
    path = locate(rel, cfg)
    if path is None:
        return FileStatus(rel=str(rel), required=required, present=False, note="missing")
    return FileStatus(
        rel=str(rel),
        required=required,
        present=True,
        path=str(path),
        bytes=path.stat().st_size if path.is_file() else None,
        note="ok" if path.is_file() else "directory",
    )


def _inspect_cache(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        info["sidecar"] = json.loads(sidecar.read_text())
        months = info["sidecar"].get("months")
        n = info["sidecar"].get("n")
        info["n"] = n
        info["months"] = months
    else:
        df = pl.scan_parquet(path)
        cols = df.collect_schema().names()
        if "month" in cols:
            months = df.select("month").unique().collect()["month"].sort().to_list()
        elif "MVT_TIME_UTC_mvt" in cols:
            months = (
                df.select(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
                .unique()
                .collect()["month"]
                .sort()
                .to_list()
            )
        else:
            months = []
        n = df.select(pl.len()).collect().item()
        info["n"] = n
        info["months"] = months
    locked = sorted(set(info.get("months") or []) & set(LOCKED_MONTHS))
    info["locked_months"] = locked
    return info


def inspect_predictions(path: Path) -> dict[str, Any]:
    schema = pl.scan_parquet(path).collect_schema().names()
    need = {"MVT_ID_mvt", "prediction"}
    missing = sorted(need - set(schema))
    n = pl.scan_parquet(path).select(pl.len()).collect().item()
    return {"n": n, "columns": schema, "missing_required_cols": missing}


def check_campaign_data(cfg: CampaignConfig | None = None, *, strict: bool = True) -> DataCheckReport:
    drive = cfg.drive_link if cfg is not None and cfg.drive_link else default_drive_link()
    report = DataCheckReport(ok=True, drive_link=str(drive) if drive else None)

    report.files.append(_stat(CACHE_REL, True, cfg))
    for rel in REQUIRED_PREDS:
        report.files.append(_stat(rel, True, cfg))
    for rel in OPTIONAL_PREDS + OPTIONAL_SIDECARS + GEOMETRY + OPTIONAL_CACHES:
        report.files.append(_stat(rel, False, cfg))
    for rel in CODE_REL:
        report.files.append(_stat(rel, rel.parts[0] == "src" or rel.name == "pyproject.toml", cfg))

    cache_file = next((f for f in report.files if f.rel == str(CACHE_REL)), None)
    if cache_file and cache_file.present and cache_file.path:
        try:
            report.cache = _inspect_cache(Path(cache_file.path))
            if report.cache.get("locked_months"):
                report.errors.append(f"cache contains locked V4 months {report.cache['locked_months']}")
            months = report.cache.get("months") or []
            extra = sorted(set(months) - set(range(1, 11)))
            if extra:
                report.errors.append(f"cache months outside 1–10: {extra}")
        except Exception as exc:
            report.errors.append(f"cache inspect failed: {exc!r}")
    else:
        report.errors.append("feature cache missing (e08b_e34_months_1_10.parquet)")

    from prc_challenge.bundles import e08b_feature_lists, e34_feature_lists

    try:
        cats, nums = e08b_feature_lists()
        assert_no_forbidden_features(cats + nums)
        cats34, nums34 = e34_feature_lists()
        assert_no_forbidden_features(cats34 + nums34)
        leak = FORBIDDEN_FEATURE_COLS.intersection(cats + nums + cats34 + nums34)
        if leak:
            report.errors.append(f"forbidden columns in feature lists: {sorted(leak)}")
    except Exception as exc:
        report.errors.append(f"feature-list check failed: {exc!r}")

    for status in report.files:
        if status.required and not status.present:
            report.errors.append(f"required missing: {status.rel}")
        if status.present and status.path and status.rel.endswith("val_predictions.parquet"):
            try:
                pred = inspect_predictions(Path(status.path))
                status.note = f"n={pred['n']}"
                if pred["missing_required_cols"]:
                    report.errors.append(f"{status.rel} missing cols {pred['missing_required_cols']}")
            except Exception as exc:
                report.warnings.append(f"could not inspect {status.rel}: {exc!r}")

    t31_v1 = locate(Path("artifacts/experiments/E37-eval-e08b-t31_V1/val_predictions.parquet"), cfg)
    t31_v2 = locate(Path("artifacts/experiments/E37-eval-e08b-t31_V2/val_predictions.parquet"), cfg)
    t31_v3 = locate(Path("artifacts/experiments/E37-eval-e08b-t31_V3/val_predictions.parquet"), cfg)
    if t31_v3 is None and t31_v1 is not None and t31_v2 is not None:
        report.warnings.append("t31 V3 predictions pending; BASE already uses t31 from V1+V2")
    elif t31_v1 is None or t31_v2 is None:
        report.warnings.append("t31 V1/V2 missing; BASE falls back to t6")

    osm = locate(GEOMETRY[0], cfg)
    if osm is None:
        report.warnings.append("OSM parking_positions.parquet missing; CPU-12/GPU-08 will skip")

    report.ok = not report.errors
    if strict and not report.ok:
        raise RuntimeError("campaign data check failed:\n" + "\n".join(report.errors))
    return report


def print_data_check(report: DataCheckReport) -> None:
    print("=== Campaign data check ===", flush=True)
    print(f"drive_link: {report.drive_link or '(not found)'}", flush=True)
    print(f"ok: {report.ok}", flush=True)
    for row in report.files:
        flag = "REQ" if row.required else "OPT"
        state = "OK " if row.present else "MISS"
        size = f"{row.bytes:,} B" if row.bytes is not None else ""
        print(f"  [{flag} {state}] {row.rel:72s} {size:14s} {row.note}", flush=True)
    if report.cache:
        print(
            f"cache n={report.cache.get('n')} months={report.cache.get('months')} "
            f"locked={report.cache.get('locked_months')}",
            flush=True,
        )
    for warn in report.warnings:
        print(f"WARN {warn}", flush=True)
    for err in report.errors:
        print(f"ERR  {err}", flush=True)


def _copy_file(src: Path, dest: Path, *, keep_existing: bool = False) -> str:
    """Copy `src` → `dest`. Hydrate uses keep_existing so a live remote run is not replaced by stale Drive."""
    if src.suffix == ".npy":
        return "skip_npy"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if keep_existing:
            return "skip_existing"
        if src.stat().st_size == dest.stat().st_size and int(src.stat().st_mtime) <= int(dest.stat().st_mtime) + 2:
            return "skip_same"
    shutil.copy2(src, dest)
    return "copied"


def _copy_tree(
    src: Path,
    dest: Path,
    *,
    ignore_names: tuple[str, ...] = NEVER_SYNC,
    keep_existing: bool = False,
) -> int:
    n = 0
    if src.is_file():
        if src.name in ignore_names or src.suffix in {".env"}:
            return 0
        action = _copy_file(src, dest, keep_existing=keep_existing)
        return 1 if action == "copied" else 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        if path.name in ignore_names or path.suffix == ".pyc" or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(src)
        action = _copy_file(path, dest / rel, keep_existing=keep_existing)
        if action == "copied":
            n += 1
    return n


def sync_to_drive(repo_root: Path, drive_link: Path | None = None) -> dict[str, Any]:
    """Copy ranking-safe campaign inputs onto the Colab Data Link. No secrets, no data/raw, no V4."""
    dest_root = drive_link or default_drive_link()
    if dest_root is None:
        raise FileNotFoundError(
            "Colab Data Link folder not found. Create Google Drive folder "
            "'My Drive/Colab Data Link' and wait for Desktop sync, or set PRC_COLAB_DATA_LINK."
        )
    dest_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []

    rels = campaign_bundle_rels(repo_root)
    for rel in rels:
        src = repo_root / rel
        dest = dest_root / rel
        if not src.exists():
            missing.append(str(rel))
            continue
        action = _copy_file(src, dest, keep_existing=is_job_result_rel(rel))
        (copied if action == "copied" else skipped).append(str(rel))
        if rel == CACHE_REL:
            _copy_file(src, dest_root / src.name)

    metrics_pairs = [p.with_name("metrics.json") for p in [*REQUIRED_PREDS, *OPTIONAL_PREDS]]
    for rel in metrics_pairs:
        src = repo_root / rel
        if src.exists():
            action = _copy_file(src, dest_root / rel)
            (copied if action == "copied" else skipped).append(str(rel))

    n_src = _copy_tree(repo_root / "src" / "prc_challenge", dest_root / "src" / "prc_challenge")
    copied.append(f"src/prc_challenge ({n_src} files)")
    scripts_src = repo_root / "scripts"
    if scripts_src.exists():
        n_scripts = _copy_tree(scripts_src, dest_root / "scripts")
        copied.append(f"scripts ({n_scripts} files)")
    for rel in (Path("pyproject.toml"), Path("configs/validation.yaml")):
        src = repo_root / rel
        if src.exists():
            action = _copy_file(src, dest_root / rel)
            (copied if action == "copied" else skipped).append(str(rel))
    nb_dir = repo_root / "notebooks"
    if nb_dir.exists():
        n_nb = _copy_tree(nb_dir, dest_root / "notebooks")
        copied.append(f"notebooks ({n_nb} files)")

    # keep a flat cache copy for the original Colab cell that looks at the folder root
    cache_src = repo_root / CACHE_REL
    if cache_src.exists():
        _copy_file(cache_src, dest_root / CACHE_REL.name)

    manifest = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(repo_root),
        "drive_link": str(dest_root),
        "copied": copied,
        "skipped_unchanged": skipped,
        "missing_optional_or_pending": missing,
        "never_synced": list(NEVER_SYNC) + ["data/raw", ".env.local", "V4 labels"],
        "colab_path": "/content/drive/MyDrive/Colab Data Link",
        "mac_path_glob": "~/Library/CloudStorage/GoogleDrive-*/My Drive/Colab Data Link",
    }
    (dest_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def hydrate_from_drive(cfg: CampaignConfig) -> dict[str, Any]:
    """Copy Drive files into repo_root so relative artifact paths work on Colab."""
    drive = Path(cfg.drive_link) if cfg.drive_link else default_drive_link()
    if drive is None or not drive.exists():
        return {"status": "no_drive", "copied": []}
    copied = []
    rels = campaign_bundle_rels(drive)
    for rel in rels:
        src = drive / rel
        if not src.exists() and (drive / Path(rel).name).exists() and rel == CACHE_REL:
            src = drive / Path(rel).name
        dest = cfg.repo_root / rel
        if not src.exists():
            continue
        try:
            if src.resolve() == dest.resolve():
                continue
        except OSError:
            pass
        dest.parent.mkdir(parents=True, exist_ok=True)
        action = _copy_file(src, dest, keep_existing=True)
        if action == "copied":
            copied.append(str(rel))
        metrics = src.with_name("metrics.json")
        if metrics.exists() and src.suffix == ".parquet":
            mdest = dest.with_name("metrics.json")
            if _copy_file(metrics, mdest, keep_existing=True) == "copied":
                copied.append(str(Path(rel).with_name("metrics.json")))
    # Nested CPU/GPU job folders and cache shards that persist_dir wrote on Drive.
    exp_src = drive / "artifacts" / "experiments"
    exp_dst = cfg.repo_root / "artifacts" / "experiments"
    if exp_src.exists():
        for child in exp_src.iterdir():
            if child.is_file() and child.name in {"campaign_index.json", "campaign_summary.json"}:
                action = _copy_file(child, exp_dst / child.name, keep_existing=True)
                if action == "copied":
                    copied.append(str(Path("artifacts/experiments") / child.name))
            elif child.is_dir() and (child.name.startswith("CPU-") or child.name.startswith("GPU-")):
                n = _copy_tree(child, exp_dst / child.name, keep_existing=True)
                if n:
                    copied.append(f"artifacts/experiments/{child.name} ({n} files)")
    cache_src = drive / "artifacts" / "cache"
    cache_dst = cfg.repo_root / "artifacts" / "cache"
    if cache_src.exists():
        for path in cache_src.iterdir():
            if not path.is_file():
                continue
            name = path.name
            keep = (
                name.startswith("f3_")
                or name.startswith("seq_")
                or name.startswith("base_oof_")
                or name.startswith("arr_context")
                or name.startswith("link_ids")
                or name == CACHE_REL.name
                or name == CACHE_JSON_REL.name
            )
            if not keep:
                continue
            action = _copy_file(path, cache_dst / name, keep_existing=True)
            if action == "copied":
                copied.append(str(Path("artifacts/cache") / name))
    return {"status": "ok", "copied": copied, "drive": str(drive)}


def sha256_file(path: Path, limit: int = 0) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        if limit:
            digest.update(fh.read(limit))
        else:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()
