# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Campaign configuration, hardware check, artifact IO, resume, experiment index."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

FORBIDDEN_FEATURE_COLS = frozenset(
    {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt"}
)
LOCKED_MONTHS = (11, 12)
INDEX_NAME = "campaign_index.json"
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
)


def cap_numeric_threads(n: int) -> None:
    """Bound BLAS/OpenMP/Polars pools so they do not oversubscribe beside CatBoost."""
    n_s = str(max(1, int(n)))
    for key in THREAD_ENV_KEYS:
        os.environ.setdefault(key, n_s)


@dataclass
class CampaignConfig:
    repo_root: Path
    artifact_root: Path
    cache_root: Path
    force_rerun: bool = False
    seed: int = 42
    views: tuple[str, ...] = ("V1", "V2", "V3")
    thread_count: int = 4
    smoke: bool = False
    max_rows: int | None = None
    optuna_trials: int = 96
    catboost_iterations: int = 400
    allow_cpu_smoke_on_gpu_notebook: bool = False
    persist_drive: Path | None = None
    drive_link: Path | None = None
    allow_local_aobt: bool = False
    allow_rebuild_cpu_caches: bool = True

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root)
        self.artifact_root = Path(self.artifact_root)
        self.cache_root = Path(self.cache_root)
        if self.persist_drive is not None:
            self.persist_drive = Path(self.persist_drive)
        if self.drive_link is not None:
            self.drive_link = Path(self.drive_link)
        if "V4" in self.views:
            raise RuntimeError("V4 is locked; remove it from CampaignConfig.views")
        if self.smoke:
            self.max_rows = self.max_rows or 8_000
            self.optuna_trials = min(self.optuna_trials, 2)
            self.catboost_iterations = min(self.catboost_iterations, 60)
            self.views = ("V1",)
        cap_numeric_threads(self.thread_count)


def resolve_repo_root(start: Path | None = None) -> Path:
    """Walk from cwd / PRC_REPO_ROOT until `src/prc_challenge` exists."""
    env = os.environ.get("PRC_REPO_ROOT")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    cwd = (start or Path.cwd()).resolve()
    candidates.extend([cwd, *cwd.parents])
    for path in candidates:
        if (path / "src" / "prc_challenge").exists():
            return path
    raise FileNotFoundError("cannot find src/prc_challenge; set PRC_REPO_ROOT to the repo root")


def default_config(repo_root: Path | None = None, *, smoke: bool = False) -> CampaignConfig:
    root = Path(repo_root) if repo_root is not None else resolve_repo_root()
    persist = os.environ.get("PRC_PERSIST_DIR")
    ncpu = os.cpu_count() or 4
    return CampaignConfig(
        repo_root=root,
        artifact_root=root / "artifacts" / "experiments",
        cache_root=root / "artifacts" / "cache",
        smoke=smoke,
        thread_count=max(1, min(ncpu, 4)),
        persist_drive=Path(persist) if persist else None,
    )


def hardware_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        report["ram_gb"] = round(vm.total / 1024**3, 2)
        report["ram_available_gb"] = round(vm.available / 1024**3, 2)
    except Exception:
        if sys.platform == "darwin":
            try:
                mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
                report["ram_gb"] = round(int(mem) / 1024**3, 2)
            except Exception:
                report["ram_gb"] = None
        else:
            meminfo = Path("/proc/meminfo")
            if meminfo.exists():
                for line in meminfo.read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        report["ram_gb"] = round(int(line.split()[1]) / 1024**2, 2)
                        break
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = bool(torch.cuda.is_available())
        report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["cuda_vram_gb"] = round(props.total_memory / 1024**3, 2)
            report["cuda_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    except Exception as exc:
        report["torch"] = f"unavailable:{type(exc).__name__}"
        report["cuda"] = False
    for mod_name, key in (
        ("numpy", "numpy"),
        ("polars", "polars"),
        ("pyarrow", "pyarrow"),
        ("catboost", "catboost"),
        ("optuna", "optuna"),
        ("sklearn", "sklearn"),
        ("lightgbm", "lightgbm"),
        ("xgboost", "xgboost"),
        ("faiss", "faiss"),
    ):
        try:
            mod = __import__(mod_name)
            report[key] = getattr(mod, "__version__", "present")
        except Exception:
            continue
    return report


def git_status(repo_root: Path) -> dict[str, str]:
    def sh(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "sha": sh(["git", "rev-parse", "HEAD"]),
        "branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": sh(["git", "status", "--porcelain"])[:2000],
    }


def experiment_dir(cfg: CampaignConfig, exp_id: str) -> Path:
    dest = cfg.artifact_root / exp_id
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _retry_skipped_reason(reason: str) -> bool:
    """Skipped jobs that were blocked on missing inputs should retry after a later job or cache fix."""
    text = (reason or "").lower()
    if "quarantined" in text or "allow_local_aobt" in text:
        return False
    if "did not reduce" in text:
        return False
    if "parked" in text:
        return False
    markers = (
        "need ",
        "no timestamp",
        "no eobt",
        "not complete",
        "has not finished",
        "2 candidates",
        "missing data/",
        "not available",
    )
    return any(m in text for m in markers)


def is_terminal_skip(payload: dict[str, Any] | None) -> bool:
    """True when a job skipped for a scientific/data PARK, not a missing-input wait."""
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status") or "")
    return status.startswith("skip") and not _retry_skipped_reason(str(payload.get("reason") or ""))


def prior_view_metrics(dest: Path, view: str) -> dict[str, Any] | None:
    """Reuse a previously scored official view so a Colab reconnect does not drop RMSE."""
    path = Path(dest) / "metrics.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    views = payload.get("views")
    if not isinstance(views, dict):
        return None
    part = views.get(view)
    if not isinstance(part, dict):
        return None
    if part.get("rmse_seconds") is not None or part.get("risk_corr_val") is not None:
        return part
    return None


def extra_views_block_complete(payload: dict[str, Any], views: tuple[str, ...] | None = None) -> bool:
    """True when a required official view is missing or still waiting on inputs.

    A view that already failed (OOM, exception) or could not be rehydrated
    (`status: resumed` leftover) is terminal: retrying the whole job would
    refit GPU-01 all night and never start GPU-02. Retryable skips
    (`need BASE preds`) still block so a later CPU hydrate can fill them.
    """
    parts = payload.get("views")
    if not isinstance(parts, dict):
        return False
    names = views if views is not None else tuple(parts.keys())
    for vw in names:
        part = parts.get(vw)
        if not isinstance(part, dict):
            return views is not None
        if part.get("status") == "skipped" and _retry_skipped_reason(
            str(part.get("reason") or part.get("error") or "")
        ):
            return True
    return False


def extra_view_exception(exc: BaseException, *, view: str) -> dict[str, Any]:
    """Missing campaign inputs retry; extra-view OOM/bugs stay terminal."""
    if isinstance(exc, FileNotFoundError):
        reason = str(exc) or f"need {view} extra-view inputs"
        if _retry_skipped_reason(reason):
            return {"status": "skipped", "reason": reason}
    return {"status": "failed", "error": repr(exc)}



def _was_smoke_run(payload: dict[str, Any], dest: Path) -> bool:
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    if runtime.get("smoke") is True:
        return True
    cfg_path = dest / "config.json"
    if not cfg_path.exists():
        return False
    try:
        conf = json.loads(cfg_path.read_text())
    except json.JSONDecodeError:
        return False
    return conf.get("smoke") is True


def _payload_has_nonfinite_rmse(payload: dict[str, Any]) -> bool:
    """NaN/Inf RMSE is not a finished experiment — the campaign must retry after a trainer fix."""
    values: list[Any] = [payload.get("rmse_seconds"), payload.get("rmse")]
    views = payload.get("views")
    if isinstance(views, dict):
        for part in views.values():
            if isinstance(part, dict):
                values.append(part.get("rmse_seconds"))
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
            return True
    return False


def is_complete(cfg: CampaignConfig, exp_id: str) -> bool:
    dest = cfg.artifact_root / exp_id
    path = dest / "metrics.json"
    if not path.exists() or cfg.force_rerun:
        return False
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    if _payload_has_nonfinite_rmse(payload):
        return False
    status = payload.get("status")
    if _was_smoke_run(payload, dest) and not cfg.smoke:
        if status == "skipped" and not _retry_skipped_reason(str(payload.get("reason") or "")):
            return True
        return False
    if extra_views_block_complete(payload):
        return False
    if not cfg.smoke and extra_views_block_complete(payload, cfg.views):
        return False
    if status == "failed":
        return False
    if status == "skipped":
        return not _retry_skipped_reason(str(payload.get("reason") or ""))
    return True


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def view_rmses(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull V1/V2/V3 composed RMSE from a job metrics.json."""
    views = payload.get("views")
    if not isinstance(views, dict):
        comparison = payload.get("comparison") or {}
        views = comparison.get("t31") or comparison.get("t6") or {}
        if not isinstance(views, dict):
            views = {}
    out: dict[str, Any] = {}
    for vw in ("V1", "V2", "V3"):
        part = views.get(vw)
        if not isinstance(part, dict):
            continue
        out[f"rmse_{vw}"] = part.get("rmse_seconds")
        vs = part.get("vs_base")
        if isinstance(vs, dict):
            out[f"delta_sse_{vw}"] = vs.get("delta_sse_vs_base")
        if part.get("status") and part.get("rmse_seconds") is None:
            out[f"status_{vw}"] = part.get("status")
        if part.get("risk_corr_val") is not None:
            out[f"risk_corr_{vw}"] = part.get("risk_corr_val")
    return out


def finalize_job_metrics(metrics: dict[str, Any], dest: Path) -> dict[str, Any]:
    """Lift V1 composed fields so ingest/review tables have rows / delta RMSE / SSE."""
    views = metrics.get("views") if isinstance(metrics.get("views"), dict) else {}
    v1 = views.get("V1") if isinstance(views.get("V1"), dict) else {}
    if not v1:
        comparison = metrics.get("comparison")
        if isinstance(comparison, dict):
            t31 = comparison.get("t31") if isinstance(comparison.get("t31"), dict) else {}
            cand = t31.get("V1") if isinstance(t31.get("V1"), dict) else {}
            if cand:
                v1 = cand
    for key in (
        "row_count",
        "sse",
        "rmse_seconds",
        "delta_rmse_vs_e33b",
        "protected",
        "ordinary",
        "matched_long",
        "sse_cells",
        "bias_seconds",
        "train_population",
        "validation_population",
        "top100_sse_share",
        "top500_sse_share",
        "top1000_sse_share",
        "months",
        "airports_top",
    ):
        if metrics.get(key) is None and v1.get(key) is not None:
            metrics[key] = v1[key]
    if metrics.get("delta_sse_vs_base") is None:
        vs = metrics.get("vs_base")
        if not isinstance(vs, dict):
            vs = v1.get("vs_base") if isinstance(v1.get("vs_base"), dict) else None
        if isinstance(vs, dict) and vs.get("delta_sse_vs_base") is not None:
            metrics["delta_sse_vs_base"] = vs["delta_sse_vs_base"]
    if metrics.get("row_count") is None:
        for name in ("predictions_V1.parquet", "predictions.parquet"):
            path = dest / name
            if not path.exists():
                continue
            try:
                import polars as pl

                metrics["row_count"] = int(pl.scan_parquet(path).select(pl.len()).collect().item())
            except Exception:
                pass
            break
    return metrics


def infer_verdict(metrics: dict[str, Any]) -> str:
    """Map a job result onto the handoff keep/reject labels."""
    status = str(metrics.get("status") or "")
    if status == "failed":
        return "REJECT"
    if status.startswith("skip"):
        return "PARK"
    text = str(metrics.get("conclusion") or "")
    for tag in ("PROMOTE", "ENSEMBLE_CANDIDATE", "PARK", "REJECT"):
        if tag in text:
            return tag
    delta = metrics.get("delta_sse_vs_base")
    if isinstance(delta, (int, float)) and delta > 0:
        return "ENSEMBLE_CANDIDATE"
    return "PARK"


def persist_repo_file(cfg: CampaignConfig, path: Path) -> None:
    """Copy a repo-relative file onto Drive so a later GPU/Colab session can hydrate it.

    Sequence index/token `.npy` files are ~1 GB. Copying them through Colab Drive FUSE
    hangs GPU-01 overnight; hydrate rebuilds those arrays from `*_ordered.parquet`.
    """
    import shutil

    path = Path(path)
    if not path.exists():
        return
    if path.suffix == ".npy":
        print(f"drive persist skipped (npy): {path.name}", flush=True)
        return
    dests: list[Path] = []
    if cfg.persist_drive is not None:
        try:
            dests.append(cfg.persist_drive / path.resolve().relative_to(cfg.artifact_root.resolve()))
        except ValueError:
            pass
    if cfg.drive_link is not None:
        try:
            dests.append(Path(cfg.drive_link) / path.resolve().relative_to(cfg.repo_root.resolve()))
        except ValueError:
            pass
    seen: set[str] = set()
    for dest in dests:
        key = str(dest)
        if key in seen:
            continue
        seen.add(key)
        try:
            if dest.resolve() == path.resolve():
                continue
        except OSError:
            pass
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".db":
                try:
                    copy_sqlite_consistent(path, dest)
                    continue
                except Exception as exc:
                    print(f"sqlite backup failed ({path.name}): {exc}; using copy2", flush=True)
            shutil.copy2(path, dest)
        except OSError as exc:
            print(f"drive persist skipped: {exc}", flush=True)


def campaign_job_catalog() -> list[tuple[str, str, Any]]:
    """CPU then GPU inventory. GPU summary must not drop CPU rows after an overnight run."""
    from prc_challenge.campaign.cpu import CPU_JOBS
    from prc_challenge.campaign.gpu import GPU_JOBS

    return list(CPU_JOBS) + list(GPU_JOBS)


def write_campaign_summary(cfg: CampaignConfig, kind: str, jobs: list | None = None) -> None:
    rows = inventory_status(cfg, jobs) if jobs is not None else read_index(cfg)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        extra: dict[str, Any] = {}
        exp_id = row.get("id")
        metrics_path = cfg.artifact_root / str(exp_id) / "metrics.json" if exp_id else None
        if metrics_path is not None and metrics_path.exists():
            try:
                extra = view_rmses(json.loads(metrics_path.read_text()))
            except json.JSONDecodeError:
                extra = {}
        enriched.append({**row, **extra})
    path = cfg.artifact_root / "campaign_summary.json"
    write_json(
        path,
        {
            "kind": kind,
            "n": len(enriched),
            "hardware": hardware_report(),
            "pending_ids": [
                r.get("id")
                for r in enriched
                if r.get("incomplete") or str(r.get("status") or "") == "pending"
            ],
            "rows": enriched,
        },
    )
    persist_repo_file(cfg, path)


def read_index(cfg: CampaignConfig) -> list[dict[str, Any]]:
    path = cfg.artifact_root / INDEX_NAME
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, list) else []


def index_row_from_metrics(exp_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
    runtime = metrics.get("runtime") if isinstance(metrics.get("runtime"), dict) else {}
    vr = view_rmses(metrics)
    return {
        "id": exp_id,
        "hypothesis": metrics.get("hypothesis"),
        "status": metrics.get("status", "completed"),
        "rmse": metrics.get("rmse_seconds") or vr.get("rmse_V1"),
        "delta_sse_vs_base": metrics.get("delta_sse_vs_base"),
        "runtime_seconds": runtime.get("seconds"),
        "reason": metrics.get("reason") or metrics.get("error"),
        "conclusion": metrics.get("conclusion"),
        "verdict": metrics.get("verdict"),
        **vr,
    }


def inventory_status(
    cfg: CampaignConfig,
    jobs: list[tuple[str, str, Any]] | tuple[tuple[str, str, Any], ...],
) -> list[dict[str, Any]]:
    """Merge the job inventory with the index so unstarted experiments show as pending."""
    by_id = {str(r.get("id")): r for r in read_index(cfg) if isinstance(r, dict) and r.get("id")}
    out: list[dict[str, Any]] = []
    for exp_id, hypothesis, _fn in jobs:
        row = dict(by_id.get(exp_id) or {})
        dest = cfg.artifact_root / exp_id
        metrics_path = dest / "metrics.json"
        if (not row.get("status") or row.get("status") == "pending") and metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict) and payload:
                row = {**index_row_from_metrics(exp_id, payload), **row}
        row["id"] = exp_id
        row["hypothesis"] = row.get("hypothesis") or hypothesis
        row["status"] = str(row.get("status") or "pending")
        row["incomplete"] = not is_complete(cfg, exp_id)
        out.append(row)
    return out


def print_campaign_status(rows: list[dict[str, Any]]) -> None:
    print("=== Campaign completion status ===", flush=True)
    counts: dict[str, int] = {}
    pending: list[str] = []
    for row in rows:
        status = str(row.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
        if status == "pending" or row.get("incomplete"):
            pending.append(str(row.get("id")))
        note = str(row.get("reason") or row.get("conclusion") or row.get("hypothesis") or "")[:60]
        print(f"{str(row.get('id')):28s} {status:18s} rmse={row.get('rmse')}  {note}", flush=True)
    print("counts", counts, flush=True)
    if pending:
        print("pending", pending, flush=True)


def sync_index_from_job_dirs(cfg: CampaignConfig) -> list[dict[str, Any]]:
    """Merge CPU-*/GPU-*/metrics.json into campaign_index.json after Drive hydrate.

    Job-folder metrics win for overlapping ids. Index-only rows are kept so a GPU
    host that hydrates a CPU index without CPU folders does not drop those jobs.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for row in read_index(cfg):
        if isinstance(row, dict) and row.get("id"):
            by_id[str(row["id"])] = row
    root = cfg.artifact_root
    if root.exists():
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            name = folder.name
            if not (name.startswith("CPU-") or name.startswith("GPU-")):
                continue
            path = folder / "metrics.json"
            if not path.exists():
                continue
            try:
                metrics = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if not isinstance(metrics, dict):
                continue
            by_id[name] = index_row_from_metrics(name, metrics)
    rows = sorted(by_id.values(), key=lambda r: str(r.get("id") or ""))
    write_json(root / INDEX_NAME, rows)
    if cfg.persist_drive is not None:
        try:
            cfg.persist_drive.mkdir(parents=True, exist_ok=True)
            write_json(cfg.persist_drive / INDEX_NAME, rows)
        except OSError:
            pass
    return rows


def upsert_index(cfg: CampaignConfig, row: dict[str, Any]) -> None:
    rows = [r for r in read_index(cfg) if r.get("id") != row.get("id")]
    rows.append(row)
    write_json(cfg.artifact_root / INDEX_NAME, rows)
    if cfg.persist_drive is not None:
        try:
            cfg.persist_drive.mkdir(parents=True, exist_ok=True)
            write_json(cfg.persist_drive / INDEX_NAME, rows)
        except OSError:
            pass


def copy_sqlite_consistent(src: Path, dest: Path) -> None:
    """Snapshot a possibly-open SQLite file, then copy the closed snapshot to dest.

    Backup must not open a SQLite connection on Colab Drive FUSE (`My Drive/...`).
    """
    import os
    import sqlite3
    import tempfile
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    local_tmp = Path(raw)
    src_conn = dest_conn = None
    try:
        src_conn = sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True)
        dest_conn = sqlite3.connect(local_tmp)
        src_conn.backup(dest_conn)
        dest_conn.close()
        dest_conn = None
        src_conn.close()
        src_conn = None
        shutil.copy2(local_tmp, dest)
    finally:
        for conn in (dest_conn, src_conn):
            if conn is None:
                continue
            try:
                conn.close()
            except Exception:
                pass
        local_tmp.unlink(missing_ok=True)


def persist_dir(cfg: CampaignConfig, dest: Path) -> None:
    if cfg.persist_drive is None:
        return
    import shutil

    try:
        if cfg.persist_drive.resolve() == dest.parent.resolve():
            return
    except OSError:
        pass
    target = cfg.persist_drive / dest.name
    try:
        target.mkdir(parents=True, exist_ok=True)
        from prc_challenge.campaign.data_check import SKIP_RESULT_NAMES, SKIP_RESULT_SUFFIX

        for path in dest.rglob("*"):
            if not path.is_file():
                continue
            if path.name in SKIP_RESULT_NAMES or path.suffix in SKIP_RESULT_SUFFIX:
                continue
            if path.name.endswith((".db-wal", ".db-shm")) or path.suffix in {".db-wal", ".db-shm"}:
                continue
            rel = path.relative_to(dest)
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".db":
                try:
                    copy_sqlite_consistent(path, out)
                    continue
                except Exception as exc:
                    print(f"sqlite backup failed ({path.name}): {exc}; using copy2", flush=True)
            shutil.copy2(path, out)
    except OSError as exc:
        print(f"drive persist skipped: {exc}", flush=True)


def run_isolated(
    cfg: CampaignConfig,
    exp_id: str,
    hypothesis: str,
    fn: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Run one experiment. Failures are recorded; they do not abort the campaign."""
    dest = experiment_dir(cfg, exp_id)
    if is_complete(cfg, exp_id):
        print(f"SKIP {exp_id} (metrics.json exists; FORCE_RERUN=False)", flush=True)
        metrics = json.loads((dest / "metrics.json").read_text())
        row = index_row_from_metrics(exp_id, metrics)
        row["hypothesis"] = hypothesis
        upsert_index(cfg, row)
        return {"status": "skipped_complete", "id": exp_id, "dir": str(dest), "metrics": metrics}
    print(f"RUN  {exp_id} — {hypothesis}", flush=True)
    t0 = time.perf_counter()
    try:
        metrics = fn(dest) or {}
        runtime = {
            "seconds": round(time.perf_counter() - t0, 3),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": hardware_report(),
            "smoke": cfg.smoke,
        }
        metrics = dict(metrics)
        metrics.setdefault("hypothesis", hypothesis)
        metrics.setdefault("experiment_id", exp_id)
        metrics.setdefault("model", metrics.get("model") or exp_id)
        metrics.setdefault("feature_bundle", "e08b")
        metrics = finalize_job_metrics(metrics, dest)
        metrics["runtime"] = runtime
        metrics["verdict"] = metrics.get("verdict") or infer_verdict(metrics)
        write_json(dest / "runtime.json", runtime)
        write_json(dest / "config.json", {**asdict(cfg), "experiment_id": exp_id, "hypothesis": hypothesis})
        write_json(dest / "metrics.json", metrics)
        (dest / "notes.md").write_text(
            f"# {exp_id}\n\n**Hypothesis:** {hypothesis}\n\n**Status:** {metrics.get('status', 'done')}\n\n"
            f"**Verdict:** {metrics.get('verdict')}\n"
        )
        persist_dir(cfg, dest)
        row = index_row_from_metrics(exp_id, metrics)
        upsert_index(cfg, row)
        print(
            f"DONE {exp_id} status={row['status']} rmse={row.get('rmse')} t={runtime['seconds']:.1f}s",
            flush=True,
        )
        return {"status": row["status"], "id": exp_id, "dir": str(dest), "metrics": metrics}
    except Exception as exc:
        if isinstance(exc, FileNotFoundError):
            reason = str(exc)
            if not _retry_skipped_reason(reason):
                reason = f"need {exc}"
            runtime = {
                "seconds": round(time.perf_counter() - t0, 3),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "hardware": hardware_report(),
                "smoke": cfg.smoke,
            }
            metrics = {
                "status": "skipped",
                "reason": reason,
                "hypothesis": hypothesis,
                "conclusion": "PARK",
            }
            write_json(dest / "runtime.json", runtime)
            write_json(dest / "config.json", {**asdict(cfg), "experiment_id": exp_id, "hypothesis": hypothesis})
            write_json(dest / "metrics.json", metrics)
            (dest / "notes.md").write_text(
                f"# {exp_id}\n\n**Hypothesis:** {hypothesis}\n\n**Status:** skipped\n\n**Reason:** {reason}\n"
            )
            persist_dir(cfg, dest)
            row = {
                "id": exp_id,
                "hypothesis": hypothesis,
                "status": "skipped",
                "reason": reason,
                "runtime_seconds": runtime["seconds"],
                "conclusion": "PARK",
                "verdict": "PARK",
            }
            upsert_index(cfg, row)
            print(f"SKIP {exp_id}: {reason}", flush=True)
            return {"status": "skipped", "id": exp_id, "dir": str(dest), "metrics": metrics}
        runtime = {"seconds": round(time.perf_counter() - t0, 3), "error": repr(exc)}
        write_json(dest / "runtime.json", runtime)
        (dest / "notes.md").write_text(f"# {exp_id} FAILED\n\n```\n{traceback.format_exc()}\n```\n")
        metrics = {"status": "failed", "error": repr(exc), "hypothesis": hypothesis}
        write_json(dest / "metrics.json", metrics)
        upsert_index(
            cfg,
            {
                "id": exp_id,
                "hypothesis": hypothesis,
                "status": "failed",
                "error": repr(exc),
                "runtime_seconds": runtime["seconds"],
            },
        )
        persist_dir(cfg, dest)
        print(f"FAIL {exp_id}: {exc!r}", flush=True)
        return {"status": "failed", "id": exp_id, "dir": str(dest), "metrics": metrics}
