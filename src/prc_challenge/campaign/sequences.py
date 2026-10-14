# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Airport event sequences for GPU models. No taxi/block/AOBT labels in tokens."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from prc_challenge.campaign.scoring import assert_no_forbidden_features

# Cache stem tag. Bump when token layout changes so GPU-02..06 do not reuse stale packs.
SEQ_PACK_TAG = "t3"

SEQ_TOKEN_NUMS = [
    "takeoff_minus_sched_sec",
    "takeoff_minus_eobt_sec",
    "local_hour",
    "sknt",
    "vsby",
    "n_delayed_dep_ctr_1800s",
    "sec_since_prev_dep",
    "n_arr_1800s",
    "n_arr_3600s",
    "n_dep_1800s",
    "dep_arr_ratio_1800s",
    "is_arr",
    "airport_hash",
    "runway_hash",
    "stand_hash",
    "type_hash",
    "runway_missing",
    "stand_missing",
    "type_missing",
    "mvt_epoch_s",
]
FORBIDDEN = {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"}

# Actual takeoff / F3-at-neighbor-time. Safe for past movements; not known for future
# scheduled-context rows at ranking time.
FUTURE_ACTUAL_TOKEN_COLS = (
    "takeoff_minus_sched_sec",
    "takeoff_minus_eobt_sec",
    "n_delayed_dep_ctr_1800s",
    "sec_since_prev_dep",
    "n_arr_1800s",
    "n_arr_3600s",
    "n_dep_1800s",
    "dep_arr_ratio_1800s",
)


def _hash_u10(col: str, alias: str) -> pl.Expr:
    return pl.col(col).cast(pl.String).fill_null("__UNK__").hash(seed=0).mod(1024).cast(pl.Float32).alias(alias)


def enrich_sequence_tokens(df: pl.DataFrame) -> pl.DataFrame:
    """Ranking-visible token fields: ARR/DEP, hashed ids, missingness, epoch."""
    exprs: list[pl.Expr] = []
    if "PHASE_mvt" in df.columns:
        exprs.append((pl.col("PHASE_mvt").cast(pl.String) == "ARR").cast(pl.Float32).alias("is_arr"))
    else:
        exprs.append(pl.lit(0.0).alias("is_arr"))
    if "AIRPORT" in df.columns:
        exprs.append(_hash_u10("AIRPORT", "airport_hash"))
    if "RUNWAY_mvt" in df.columns:
        exprs.append(_hash_u10("RUNWAY_mvt", "runway_hash"))
        exprs.append(pl.col("RUNWAY_mvt").is_null().cast(pl.Float32).alias("runway_missing"))
    if "STAND_mvt" in df.columns:
        exprs.append(_hash_u10("STAND_mvt", "stand_hash"))
        exprs.append(pl.col("STAND_mvt").is_null().cast(pl.Float32).alias("stand_missing"))
    if "AIRCRAFT_TYPE_mvt" in df.columns:
        exprs.append(_hash_u10("AIRCRAFT_TYPE_mvt", "type_hash"))
        exprs.append(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Float32).alias("type_missing"))
    if "MVT_TIME_UTC_mvt" in df.columns:
        dtype = df.schema.get("MVT_TIME_UTC_mvt")
        if dtype is not None and getattr(dtype, "is_temporal", lambda: False)():
            exprs.append(pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("mvt_epoch_i64"))
            exprs.append(pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float32).alias("mvt_epoch_s"))
        else:
            exprs.append(pl.col("MVT_TIME_UTC_mvt").cast(pl.Int64).alias("mvt_epoch_i64"))
            exprs.append(pl.col("MVT_TIME_UTC_mvt").cast(pl.Float32).alias("mvt_epoch_s"))
    if "SCHED_TIME_UTC_mvt" in df.columns:
        dtype = df.schema.get("SCHED_TIME_UTC_mvt")
        if dtype is not None and getattr(dtype, "is_temporal", lambda: False)():
            exprs.append(pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("sched_epoch_i64"))
        else:
            exprs.append(pl.col("SCHED_TIME_UTC_mvt").cast(pl.Int64).alias("sched_epoch_i64"))
    return df.with_columns(exprs) if exprs else df


def window_indices(airports: np.ndarray, past: int, future: int) -> np.ndarray:
    """Same-airport past (oldest→newest), the target row, then future (nearest→farthest). Missing = -1.

    Memory is O(n * (past+1+future)) int32, not O(n * T * D) floats.
    """
    n = len(airports)
    t = past + 1 + future
    if n == 0:
        return np.zeros((0, t), dtype=np.int32)
    idx = np.full((n, t), -1, dtype=np.int32)
    rows = np.arange(n, dtype=np.int32)
    for j, lag in enumerate(range(past, 0, -1)):
        src = rows - np.int32(lag)
        valid = src >= 0
        src_clip = np.clip(src, 0, max(n - 1, 0))
        valid &= airports == airports[src_clip]
        idx[:, j] = np.where(valid, src, np.int32(-1))
    idx[:, past] = rows
    for j, lead in enumerate(range(1, future + 1)):
        src = rows + np.int32(lead)
        valid = src < n
        src_clip = np.clip(src, 0, max(n - 1, 0))
        valid &= airports == airports[src_clip]
        idx[:, past + 1 + j] = np.where(valid, src, np.int32(-1))
    return idx


def token_columns(frame: pl.DataFrame) -> list[str]:
    cols = [c for c in SEQ_TOKEN_NUMS if c in frame.columns]
    if not cols:
        cols = [c for c in ("local_hour", "takeoff_minus_sched_sec") if c in frame.columns]
    assert_no_forbidden_features(cols)
    if FORBIDDEN.intersection(cols):
        raise RuntimeError("sequence tokens include forbidden labels")
    return cols


def ensure_sequence_token_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Fixed-width token schema so train/score/extra views share the same seq_dim."""
    out = enrich_sequence_tokens(frame)
    missing = [c for c in SEQ_TOKEN_NUMS if c not in out.columns]
    if missing:
        out = out.with_columns([pl.lit(0.0).alias(c) for c in missing])
    return out


def mix_arrivals(
    dep: pl.DataFrame,
    arr: pl.DataFrame | None,
    *,
    window_sec: int = 7200,
) -> pl.DataFrame:
    """Interleave ranking-safe ARR rows into a DEP timeline. Train/score still uses DEP IDs only.

    ARR more than `window_sec` from the nearest same-airport DEP are dropped so the mixed
    pack stays near the 64-past / 32-future neighborhood instead of the full day.
    """
    dep_out = dep
    if "PHASE_mvt" not in dep_out.columns:
        dep_out = dep_out.with_columns(pl.lit("DEP").alias("PHASE_mvt"))
    if arr is None or arr.height == 0 or "MVT_TIME_UTC_mvt" not in arr.columns:
        return dep_out
    if "AIRPORT" in dep_out.columns and "AIRPORT" in arr.columns:
        arr = arr.join(dep_out.select("AIRPORT").unique(), on="AIRPORT", how="semi")
    if "month" in dep_out.columns and "month" in arr.columns:
        arr = arr.join(dep_out.select("month").unique(), on="month", how="semi")
    if "MVT_ID_mvt" in dep_out.columns and "MVT_ID_mvt" in arr.columns:
        arr = arr.join(dep_out.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti")
    if arr.height == 0:
        return dep_out
    dtype = arr.schema.get("MVT_TIME_UTC_mvt")
    if window_sec and dtype is not None and getattr(dtype, "is_temporal", lambda: False)() and "AIRPORT" in arr.columns:
        dep_t = dep_out.select(["AIRPORT", pl.col("MVT_TIME_UTC_mvt").alias("_dep_t")]).sort(["AIRPORT", "_dep_t"])
        try:
            arr = (
                arr.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
                .join_asof(dep_t, left_on="MVT_TIME_UTC_mvt", right_on="_dep_t", by="AIRPORT", strategy="nearest")
                .filter(
                    (pl.col("MVT_TIME_UTC_mvt").dt.epoch("s") - pl.col("_dep_t").dt.epoch("s")).abs() <= int(window_sec)
                )
                .drop(["_dep_t"], strict=False)
            )
        except Exception:
            arr = arr.drop(["_dep_t"], strict=False)
    if arr.height == 0:
        return dep_out
    return pl.concat([dep_out, arr], how="diagonal")


def target_row_mask(ordered: pl.DataFrame, target_ids) -> np.ndarray:
    ids = target_ids if isinstance(target_ids, pl.Series) else pl.Series("MVT_ID_mvt", target_ids)
    hits = ids.to_frame("MVT_ID_mvt").unique().with_columns(pl.lit(True).alias("_tgt"))
    return (
        ordered.select("MVT_ID_mvt")
        .join(hits, on="MVT_ID_mvt", how="left")["_tgt"]
        .fill_null(False)
        .to_numpy()
        .astype(bool, copy=False)
    )


def build_event_pack(
    df: pl.DataFrame,
    *,
    past: int = 64,
    future: int = 0,
) -> dict[str, Any]:
    """Compact token table + neighbor indices. Does not materialize (n, T, D)."""
    frame = ensure_sequence_token_columns(df)
    cols = list(SEQ_TOKEN_NUMS)
    ordered = frame.sort(["AIRPORT", "MVT_TIME_UTC_mvt"])
    tok = ordered.select([pl.col(c).cast(pl.Float32).fill_null(0).fill_nan(0) for c in cols])
    tokens = np.ascontiguousarray(tok.to_numpy()).astype(np.float32, copy=False)
    airports = ordered["AIRPORT"].cast(pl.Categorical).to_physical().to_numpy()
    idx = window_indices(airports, max(1, past), max(0, future))
    epoch = None
    epoch_j = cols.index("mvt_epoch_s") if "mvt_epoch_s" in cols else None
    if "mvt_epoch_i64" in ordered.columns:
        epoch = ordered["mvt_epoch_i64"].fill_null(0).to_numpy().astype(np.int64, copy=False)
    sched_epoch = None
    if "sched_epoch_i64" in ordered.columns:
        sched_epoch = ordered["sched_epoch_i64"].fill_null(0).to_numpy().astype(np.int64, copy=False)
    return {
        "ordered": ordered,
        "tokens": tokens,
        "idx": idx,
        "cols": cols,
        "epoch": epoch,
        "epoch_j": epoch_j,
        "sched_epoch": sched_epoch,
        "past": max(1, past),
        "future": max(0, future),
    }


def _dep_id_hash(dep: pl.DataFrame) -> str:
    import hashlib

    ids = np.ascontiguousarray(dep["MVT_ID_mvt"].to_numpy())
    return hashlib.sha1(ids.tobytes()).hexdigest()[:16]


def _window_mask(idx: np.ndarray) -> np.ndarray:
    """Per-batch mask from neighbor indices. Do not store a dense (n, T) float32 copy."""
    return (np.asarray(idx) >= 0).astype(np.float32)


def _mmap_pack_arrays(tok_path, idx_path) -> tuple[np.ndarray, np.ndarray]:
    """Keep GPU-01..06 from holding ~1 GB tokens + ~1 GB idx + ~1 GB mask in RSS."""
    return np.load(tok_path, mmap_mode="r"), np.load(idx_path, mmap_mode="r")


def _arrays_from_ordered(ordered: pl.DataFrame, cols: list[str], past: int, future: int) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild compact npy from the lean ordered table. Does not call mix_arrivals."""
    tok = ordered.select([pl.col(c).cast(pl.Float32).fill_null(0).fill_nan(0) for c in cols])
    tokens = np.ascontiguousarray(tok.to_numpy()).astype(np.float32, copy=False)
    airports = ordered["AIRPORT"].cast(pl.Categorical).to_physical().to_numpy()
    idx = window_indices(airports, max(1, past), max(0, future))
    return tokens, idx


def _arr_fingerprint(cfg) -> str:
    """Sidecar size+mtime so packs rebuild if ARR context appears after a DEP-only mix."""
    from prc_challenge.campaign.f3 import arr_context_path

    path = arr_context_path(cfg)
    if not path.exists():
        return "noarr"
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


# Labels / OOF / compose flags only. Copying the full e08b table onto mixed ARR+DEP rows OOMs GPU-04 fusion.
# nm_unmatched + extra-day flags are required for E17/E18 and unmatched freeze at score time.
SEQ_ATTACH_COLS = (
    "TAXITIME_SEC_mvt",
    "residual",
    "queue_residual",
    "oof_prediction",
    "route_base",
    "base_prediction",
    "pred_risk",
    "nm_unmatched",
    "takeoff_minus_sched_days",
)

SEQ_MIX_COLS = (
    "MVT_ID_mvt",
    "AIRPORT",
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "PHASE_mvt",
    "month",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    *SEQ_TOKEN_NUMS,
)


def seq_dep_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Skinny DEP table for pack build: tokens + labels, not the e08b feature dump."""
    cols = [c for c in dict.fromkeys((*SEQ_MIX_COLS, *SEQ_ATTACH_COLS)) if c in frame.columns]
    return frame.select(cols) if cols else frame


def _slim_for_sequence(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep token/id columns only so mix_arrivals does not copy the e08b table."""
    cols = [c for c in dict.fromkeys(SEQ_MIX_COLS) if c in frame.columns]
    return frame.select(cols) if cols else frame


def _attach_dep_columns(ordered: pl.DataFrame, dep: pl.DataFrame) -> pl.DataFrame:
    extra = [c for c in SEQ_ATTACH_COLS if c in dep.columns]
    if not extra or "MVT_ID_mvt" not in ordered.columns or "MVT_ID_mvt" not in dep.columns:
        return ordered
    keep_ordered = [c for c in ordered.columns if c not in extra]
    ordered = ordered.select(keep_ordered)
    right = dep.select(["MVT_ID_mvt", *extra]).unique("MVT_ID_mvt", keep="first")
    if ordered.schema["MVT_ID_mvt"] != right.schema["MVT_ID_mvt"]:
        right = right.with_columns(pl.col("MVT_ID_mvt").cast(ordered.schema["MVT_ID_mvt"], strict=False))
    return ordered.join(right, on="MVT_ID_mvt", how="left")


def load_or_build_event_pack(
    cfg,
    dep: pl.DataFrame,
    *,
    past: int,
    future: int,
    tag: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Cache compact sequence packs so later GPU jobs do not rebuild mix_arrivals."""
    import json
    from pathlib import Path

    from prc_challenge.campaign.f3 import load_or_build_arr_context
    from prc_challenge.campaign.runtime import persist_repo_file

    cfg.cache_root.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
    suffix = "_smoke" if getattr(cfg, "smoke", False) else ""
    stem = cfg.cache_root / f"seq_{safe}_p{int(past)}_f{int(future)}_{SEQ_PACK_TAG}{suffix}"
    meta_path = Path(str(stem) + "_meta.json")
    tok_path = Path(str(stem) + "_tokens.npy")
    idx_path = Path(str(stem) + "_idx.npy")
    ordered_path = Path(str(stem) + "_ordered.parquet")
    want_hash = _dep_id_hash(dep)
    want_arr = _arr_fingerprint(cfg)
    if (
        not getattr(cfg, "force_rerun", False)
        and meta_path.exists()
        and ordered_path.exists()
    ):
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
        cached_arr = meta.get("arr_fp") or "noarr"
        if (
            meta.get("id_hash") == want_hash
            and cached_arr == want_arr
            and int(meta.get("past", -1)) == int(past)
            and int(meta.get("future", -1)) == int(future)
            and list(meta.get("token_schema") or []) == list(SEQ_TOKEN_NUMS)
            and meta.get("tag") == SEQ_PACK_TAG
            and len(meta.get("cols") or []) == len(SEQ_TOKEN_NUMS)
        ):
            ordered = _attach_dep_columns(pl.read_parquet(ordered_path), dep)
            cols = list(SEQ_TOKEN_NUMS)
            if all(c in ordered.columns for c in cols):
                if tok_path.exists() and idx_path.exists():
                    tokens, idx = _mmap_pack_arrays(tok_path, idx_path)
                    if tokens.ndim == 2 and int(tokens.shape[1]) == len(SEQ_TOKEN_NUMS):
                        print(f"seq pack resume {stem.name}", flush=True)
                        epoch = None
                        if "mvt_epoch_i64" in ordered.columns:
                            epoch = ordered["mvt_epoch_i64"].fill_null(0).to_numpy().astype(np.int64, copy=False)
                        sched_epoch = None
                        if "sched_epoch_i64" in ordered.columns:
                            sched_epoch = ordered["sched_epoch_i64"].fill_null(0).to_numpy().astype(
                                np.int64, copy=False
                            )
                        pack = {
                            "ordered": ordered,
                            "tokens": tokens,
                            "idx": idx,
                            "cols": cols,
                            "epoch": epoch,
                            "epoch_j": cols.index("mvt_epoch_s") if "mvt_epoch_s" in cols else None,
                            "sched_epoch": sched_epoch,
                            "past": int(past),
                            "future": int(future),
                        }
                        return pack, target_row_mask(ordered, dep["MVT_ID_mvt"])
                else:
                    print(f"seq pack rebuild npy from ordered {stem.name}", flush=True)
                    tokens, idx = _arrays_from_ordered(ordered, cols, past, future)
                    np.save(tok_path, tokens)
                    np.save(idx_path, idx)
                    tokens, idx = _mmap_pack_arrays(tok_path, idx_path)
                    epoch = None
                    if "mvt_epoch_i64" in ordered.columns:
                        epoch = ordered["mvt_epoch_i64"].fill_null(0).to_numpy().astype(np.int64, copy=False)
                    sched_epoch = None
                    if "sched_epoch_i64" in ordered.columns:
                        sched_epoch = ordered["sched_epoch_i64"].fill_null(0).to_numpy().astype(np.int64, copy=False)
                    pack = {
                        "ordered": ordered,
                        "tokens": tokens,
                        "idx": idx,
                        "cols": cols,
                        "epoch": epoch,
                        "epoch_j": cols.index("mvt_epoch_s") if "mvt_epoch_s" in cols else None,
                        "sched_epoch": sched_epoch,
                        "past": int(past),
                        "future": int(future),
                    }
                    return pack, target_row_mask(ordered, dep["MVT_ID_mvt"])
    arr = load_or_build_arr_context(cfg)
    want_arr = _arr_fingerprint(cfg)
    mixed = mix_arrivals(
        _slim_for_sequence(dep),
        None if arr is None else _slim_for_sequence(arr),
    )
    pack = build_event_pack(mixed, past=past, future=future)
    lean_cols = [
        c
        for c in (
            "MVT_ID_mvt",
            "AIRPORT",
            "MVT_TIME_UTC_mvt",
            "PHASE_mvt",
            "month",
            "mvt_epoch_i64",
            "sched_epoch_i64",
            "is_arr",
        )
        if c in pack["ordered"].columns
    ]
    for c in pack["cols"]:
        if c in pack["ordered"].columns and c not in lean_cols:
            lean_cols.append(c)
    pack["ordered"].select(lean_cols).write_parquet(ordered_path)
    np.save(tok_path, pack["tokens"])
    np.save(idx_path, pack["idx"])
    meta_path.write_text(
        json.dumps(
            {
                "id_hash": want_hash,
                "arr_fp": want_arr,
                "past": int(past),
                "future": int(future),
                "n": int(pack["tokens"].shape[0]),
                "cols": pack["cols"],
                "layout": "past_target_future",
                "tag": SEQ_PACK_TAG,
                "token_schema": list(SEQ_TOKEN_NUMS),
            },
            indent=2,
        )
        + "\n"
    )
    for path in (ordered_path, tok_path, idx_path, meta_path):
        persist_repo_file(cfg, path)
    pack.pop("mask", None)
    pack["tokens"], pack["idx"] = _mmap_pack_arrays(tok_path, idx_path)
    pack["ordered"] = _attach_dep_columns(pack["ordered"].select(lean_cols), dep)
    n_arr = 0 if arr is None else int(arr.height)
    print(
        f"seq mix tag={tag} DEP={dep.height} ARR_sidecar={n_arr} mixed={pack['ordered'].height} "
        f"targets={int(target_row_mask(pack['ordered'], dep['MVT_ID_mvt']).sum())}",
        flush=True,
    )
    return pack, target_row_mask(pack["ordered"], dep["MVT_ID_mvt"])


def gather_sequence(pack: dict[str, Any], rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gather `(batch, T, D)` from a compact pack. `pack['idx']` always indexes the full token table."""
    idx = pack["idx"] if rows is None else pack["idx"][rows]
    mask = _window_mask(idx)
    tokens = pack["tokens"]
    if tokens.shape[0] == 0:
        d = tokens.shape[1] if tokens.ndim == 2 else 0
        t = idx.shape[1] if idx.ndim == 2 else 0
        return np.zeros((0, t, d), dtype=np.float32), mask
    safe = np.where(idx >= 0, idx, 0)
    seq = np.array(tokens[safe], dtype=np.float32, copy=True)
    missing = idx < 0
    seq[missing] = 0.0
    epoch_j = pack.get("epoch_j")
    epoch = pack.get("epoch")
    past = pack.get("past")
    future = pack.get("future") or 0
    if epoch_j is not None and epoch is not None:
        target_rows = np.arange(tokens.shape[0], dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
        target = epoch[target_rows].astype(np.float64).reshape(-1, 1)
        neigh = epoch[safe].astype(np.float64)
        offset = (target - neigh).astype(np.float32)
        offset[missing] = 0.0
        seq[:, :, epoch_j] = offset * mask
        if past is not None and int(future) > 0:
            start = int(past) + 1
            if start < seq.shape[1]:
                sched = pack.get("sched_epoch")
                if sched is not None:
                    neigh_s = sched[safe].astype(np.float64)
                    off_s = (target - neigh_s).astype(np.float32)
                    off_s[missing] = 0.0
                    off_s[neigh_s == 0] = 0.0
                    seq[:, start:, epoch_j] = off_s[:, start:] * mask[:, start:]
                else:
                    seq[:, start:, epoch_j] = 0.0
    cols = pack.get("cols") or []
    if past is not None and int(future) > 0:
        start = int(past) + 1
        leak_j = [i for i, c in enumerate(cols) if c in FUTURE_ACTUAL_TOKEN_COLS]
        if start < seq.shape[1] and leak_j:
            seq[:, start:, leak_j] = 0.0
    np.nan_to_num(seq, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(seq, dtype=np.float32), mask


def build_event_sequence(
    df: pl.DataFrame,
    *,
    past: int = 64,
    future: int = 0,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Previous-`past`, the target row, and optional future scheduled-context same-airport movements.

    Future tokens keep ranking-visible identity/weather/scheduled context.
    Actual takeoff deltas and F3-at-neighbor-time are zeroed on future slots.
    Taxi / block / AOBT are never included. Epoch channels are converted to
    signed seconds from the target movement (past > 0, future < 0).

    Materializes `(n, T, D)` — use `build_event_pack` + `gather_sequence` for full-data GPU jobs.
    """
    pack = build_event_pack(df, past=past, future=future)
    seq, mask = gather_sequence(pack)
    return pack["ordered"], seq, mask
