# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-NNX data prep: preprocessed, memory-mappable caches for the neural search (D-065).

One cache per split.  A split names a run directory (an assembled S4 matrix) and which rows are
"fit" rows and which are "ev" (holdout / evaluation) rows:

* ``m10``: ADS-B D-fit matrix, fit = D months != 10, ev = month 10 (selection holdout).
* ``m5``:  same matrix, fit = D months != 5, ev = month 5 (robustness holdout).
* ``jj``:  same matrix, fit = every D month, ev = the JAN_JUL eval matrix (ONE confirmation look).
* ``ay``:  all-year matrix, fit = every training row, ev = the ranking rows.
* ``ay3l``: the V3L all-year matrix (``vm_allyear_v3l_cb_plain``), fit = every training row, ev = ranking.
* ``ay3liso`` (D-102): as ``ay3l`` on the isocap_v1 V3L all-year matrix (``allyear_iso_v3l_cb_plain``).
* ``dfit3l_<a>_<b>`` (SA-BLEND, D-087): the V3L D-fit matrix = the stored ADS-B D-fit matrix plus the
  v3-lean map block (``adsb_v3lean_block_dfit.parquet``, appended as ``dfold_oof.py --extra`` does),
  fit = D months other than a and b, ev = months a and b (one blocked D fold of the D-066 OOF bank:
  ``dfit3l_2_8``, ``dfit3l_3_9``, ``dfit3l_4_10``, ``dfit3l_5_6``).  Holdout labels are stored (D rows).

The preprocessing is V14's (``scripts/legs/train_nn_matrix.py``'s ``Preprocessor``: vocabularies and
quantile-normal maps fitted on the fit rows' features only).  Per split we write

* ``{fit,ev}_cat.npy`` (int16), ``{fit,ev}_dense.npy`` (float16), ``{fit,ev}_ids.npy`` (MVT_ID),
* ``{fit,ev}_label.npy`` (taxi seconds; ranking rows: NaN) -- stored apart from every input,
* ``{fit,ev}_off_<target>.npy``: the target offset (``s4.TargetSpec.offset``) per target variant,
* ``{fit,ev}_clock.npy`` (float32, N x 4): the row's own raw clocks relative to its take-off
  (``takeoff_minus_{aobt,eobt,lobt,sched}_sec``; NaN -> 0 with the NaN flag in ``dense``),
* temporal context: ``{fit,ev}_nb_idx.npy`` (int32, N x K, -1 = none) and ``{fit,ev}_nb_dt.npy``
  (float32 seconds, neighbour take-off minus focal take-off).  Neighbours are the K_A preceding and
  K_A following matched departures at the same airport and the K_R preceding / following matched
  departures on the same runway, ordered by MVT (take-off) time, **within the same file and the
  same airport-period**.  Indices point into the same split block (fit rows -> fit rows, ev rows ->
  ev rows).  No label or label-derived column is read to build any input: the only non-matrix
  column is ``MVT_TIME_UTC_mvt`` from the base cache (a take-off clock).

The leakage test (``scripts/legs/nnx/test_leakage.py``) rebuilds the inputs with the label column
scrambled and asserts bit-identity.

**ctx-v2 options (bake, D-076 code gap #3; defaults reproduce the caches above byte for byte):**

* ``--k-airport K --k-runway R``: the neighbour window sizes (default 16 / 8).
* ``--k-arrivals A`` (default 0 = none): arrival tokens, the A preceding and A following **landings** at
  the same airport and period relative to the focal take-off (``{fit,ev}_arr_dt`` float32 N x 2A, landing
  minus focal take-off, nearest-first, 0 where none; ``{fit,ev}_arr_mask`` int8; ``{fit,ev}_arr_rwy`` int8,
  same runway as the focal row).  Landing clocks come from the masked movement table
  (``matched_context.movements``: DEP BLOCK/TAXITIME, AOBT_3, ARVT_3 are dropped before anything is read);
  arrivals sharing a FLIGHT_ID with a departure of the same airport-window (air returns) are excluded.
  The tokens are label-free clocks.
* ``--arr-taxi-in``: also store the arrivals' taxi-in (``{fit,ev}_arr_taxi``, seconds, 0 where missing, and
  ``{fit,ev}_arr_taxi_ok``), always through the arrival-label guard (< 0 s or > 3 h -> missing,
  ``prc_challenge.arrival_guard``).  ARR taxi-in is a legal input (D-005) but not label-free for arrivals.
* Non-default options write to ``nnx_cache_<split><suffix>`` (suffix ``_k<K>r<R>[a<A>[t]]``); the
  default cache directory and its files are unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_THREADS = os.environ.get("NNX_THREADS", "3")
for _v in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # scripts/ -> train_nn_matrix

from prc_challenge import matrix_arms as ma  # noqa: E402
from prc_challenge import s4_allyear as s4  # noqa: E402
import train_nn_matrix as tnm  # noqa: E402

MAIN = Path(os.environ.get("PRC_MAIN_REPO", Path(__file__).resolve().parents[3]))  # bake VM: PRC_MAIN_REPO
RUNS = MAIN / "artifacts/experiments/S4-allyear-rebuild-20260925/runs"
BASE = MAIN / "artifacts/experiments/S4-allyear-rebuild-20260925/features/base"
DFIT = RUNS / "dfit_full_adsb_historder"
ALLYEAR = RUNS / "allyear_full_adsb_final"
ALLYEAR_V3L = RUNS / "vm_allyear_v3l_cb_plain"  # D-069: all-year matrix + 7 adsb3_* taxiway-map columns
ALLYEAR_V3L_ISO = RUNS / "allyear_iso_v3l_cb_plain"  # D-102: the V3L all-year matrix, isocap_v1 temporal context
V3L_DFIT_BLOCK = MAIN / "artifacts/experiments/ADSB-coverage-20260926/v3/adsb_v3lean_block_dfit.parquet"  # D-087 V3L bank
D_FOLDS = ((2, 8), (3, 9), (4, 10), (5, 6))  # D-066 blocked folds

TARGETS: dict[str, s4.TargetSpec] = {
    "delta": tnm.TARGETS["delta"],  # V14
    "resid": tnm.TARGETS["resid"],  # V15
    "eobtres": tnm.TARGETS["eobtres"],  # V16
    "lobtoff": s4.TargetSpec(offset_col="takeoff_minus_lobt_sec", sign=1, fill_col="unimpeded_p10", winsor_sec=7200.0),  # V12's
    "p10": s4.TargetSpec(),  # V6's default residual (LightGBM inversion only)
}
CLOCKS = ["takeoff_minus_aobt_sec", "takeoff_minus_eobt_sec", "takeoff_minus_lobt_sec", "takeoff_minus_sched_sec"]
K_AIRPORT = 16
K_RUNWAY = 8
RAW = MAIN / "data" / "raw"

SPLITS = {
    "m10": {"run": DFIT, "holdout": 10},
    "m5": {"run": DFIT, "holdout": 5},
    "jj": {"run": DFIT, "holdout": None},
    "ay": {"run": ALLYEAR, "holdout": None},
    "ay3l": {"run": ALLYEAR_V3L, "holdout": None},
    "ay3liso": {"run": ALLYEAR_V3L_ISO, "holdout": None},  # D-102 (SA-NNISO): ay3l on the isocap_v1 matrix
    # final phase: the ay3liso train matrix with another evaluation matrix (predict-only, scripts/legs/nnx/predict_serve.py)
    "ay3liso_regress": {"run": RUNS / "regress_allyear_iso_v3l", "holdout": None},  # re-assembled Jan/Jul ranking matrix
    "ay3liso_serve4m": {"run": RUNS / "serve4m_allyear_iso_v3l", "holdout": None},  # Jan, Feb, Jun, Jul 2026
    **{f"dfit3l_{a}_{b}": {"run": DFIT, "holdout": (a, b), "extra": V3L_DFIT_BLOCK} for a, b in D_FOLDS},
}


def append_extra(frame: pl.DataFrame, path: Path, nums: list[str]) -> tuple[pl.DataFrame, list[str], dict]:
    """``dfold_oof.py --extra``: left-join a label-free block (MVT_ID_mvt + numeric columns, cast Float32)
    and append its columns to ``nums``.  Refuses label columns and name clashes; keeps the row order."""

    import hashlib

    extra = pl.read_parquet(path)
    cols = [c for c in extra.columns if c != "MVT_ID_mvt"]
    bad = {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "ARVT_3_flt"}.intersection(cols)
    if bad or set(cols) & set(frame.columns):
        raise SystemExit(f"bad extra columns: {sorted(bad or set(cols) & set(frame.columns))}")
    extra = extra.select(pl.col("MVT_ID_mvt").cast(frame.schema["MVT_ID_mvt"]), *[pl.col(c).cast(pl.Float32) for c in cols]).unique("MVT_ID_mvt")
    n0 = frame.height
    out = frame.join(extra, on="MVT_ID_mvt", how="left", maintain_order="left")
    if out.height != n0:
        raise SystemExit("extra join changed the row count")
    return out, nums + cols, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "columns": cols}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg} (peak_rss={s4.peak_rss_gb():.2f} GB)", flush=True)


def cache_suffix(k_airport: int = K_AIRPORT, k_runway: int = K_RUNWAY, k_arrivals: int = 0, arr_taxi_in: bool = False) -> str:
    """'' for the default layout (16/8, no arrivals); otherwise ``_k<K>r<R>[a<A>[t]]``."""

    if (k_airport, k_runway, k_arrivals, arr_taxi_in) == (K_AIRPORT, K_RUNWAY, 0, False):
        return ""
    return f"_k{k_airport}r{k_runway}" + (f"a{k_arrivals}" + ("t" if arr_taxi_in else "") if k_arrivals else "")


def cache_dir(split: str, suffix: str = "") -> Path:
    return RUNS / f"nnx_cache_{split}{suffix}"


# --------------------------------------------------------------------------- temporal context


def mvt_times(ids: np.ndarray, periods: list[str]) -> np.ndarray:
    """MVT (take-off) epoch seconds for the given MVT ids, from the base caches of ``periods``."""

    frames = [
        pl.read_parquet(BASE / f"base_{p}.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"])
        for p in sorted(set(periods))
    ]
    base = pl.concat(frames).unique("MVT_ID_mvt", keep="first")
    q = pl.DataFrame({"MVT_ID_mvt": ids})
    joined = q.join(base, on="MVT_ID_mvt", how="left", maintain_order="left")
    t = joined["MVT_TIME_UTC_mvt"]
    if t.null_count():
        raise RuntimeError(f"{t.null_count()} rows without MVT time")
    if t.dtype == pl.String:
        t = t.str.to_datetime()
    return (t.dt.epoch("ms").cast(pl.Float64) / 1000.0).to_numpy()


def neighbours(group: np.ndarray, t: np.ndarray, pool: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k preceding + k following pool rows of the same group, ordered by time.

    ``group`` int64 group ids (-1 = no group), ``t`` epoch seconds, ``pool`` boolean mask of rows
    that may serve as neighbours.  Returns (idx, dt) of shape (N, 2k): columns 0..k-1 are the
    nearest-first preceding neighbours, k..2k-1 the nearest-first following ones; -1 / 0 where
    there is none.  A row is never its own neighbour.  Ties in time are broken by row order.
    """

    n = t.shape[0]
    idx = np.full((n, 2 * k), -1, dtype=np.int32)
    dt = np.zeros((n, 2 * k), dtype=np.float32)
    rows = np.flatnonzero(group >= 0)
    order = rows[np.lexsort((rows, t[rows], group[rows]))]  # every grouped row, by (group, time, row)
    in_pool = pool[order]
    before = np.concatenate([[0], np.cumsum(in_pool)[:-1]])  # pool rows strictly before each position
    pool_order = order[in_pool]  # pool rows in sorted order
    m = pool_order.size
    g_pool = group[pool_order]
    for side, sgn in ((0, -1), (1, 1)):
        for j in range(1, k + 1):
            if sgn < 0:
                p = before - j
            else:
                p = before + j - np.where(in_pool, 0, 1)  # pool row: skip itself; else the next pool row is `before`
            valid = (p >= 0) & (p < m)
            pc = np.clip(p, 0, max(m - 1, 0))
            same = valid & (g_pool[pc] == group[order]) if m else valid
            focal = order[same]
            nb = pool_order[pc[same]]
            col = side * k + (j - 1)
            idx[focal, col] = nb.astype(np.int32)
            dt[focal, col] = (t[nb] - t[focal]).astype(np.float32)
    return idx, dt


def group_ids(frame: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Dense int64 ids of the column combination (any null -> -1)."""

    g = frame.select([pl.col(c).cast(pl.String) for c in cols])
    has_null = g.select(pl.any_horizontal([pl.col(c).is_null() for c in cols])).to_series().to_numpy()
    codes = g.select(pl.concat_str([pl.col(c).fill_null("") for c in cols], separator="|").alias("k"))["k"]
    ids = codes.cast(pl.Categorical).to_physical().cast(pl.Int64).to_numpy().copy()
    ids[has_null] = -1
    return ids


def context_arrays(frame: pl.DataFrame, t: np.ndarray, k_airport: int = K_AIRPORT, k_runway: int = K_RUNWAY) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour indices / time offsets for every row of ``frame`` (label-free)."""

    pool = frame["nm_unmatched"].cast(pl.Int64).to_numpy() == 0
    g_air = group_ids(frame, ["AIRPORT", "period"])
    g_rwy = group_ids(frame, ["AIRPORT", "period", "RUNWAY_mvt"])
    ia, da = neighbours(g_air, t, pool, k_airport)
    ir, dr = neighbours(g_rwy, t, pool, k_runway)
    return np.concatenate([ia, ir], axis=1), np.concatenate([da, dr], axis=1)


# --------------------------------------------------------------------------- arrival tokens (ctx-v2)


def arrival_table(periods: list[str], raw_dir: Path = RAW) -> pl.DataFrame:
    """Landings (label-free clocks) of the given periods from the masked movement table.

    Columns: airport, period, t (landing epoch s), runway, taxi (in-block - landing, s; not a clock).
    Training periods ``2025-MM`` read the month file (its own context window); ranking periods ``2026-MM``
    read the ranking file split by calendar month.  Air returns (an ARR sharing FLIGHT_ID with a DEP of the
    same airport-window) are excluded.
    """

    from prc_challenge.matched_context.movements import load_ranking, load_training_month

    frames = []
    ps = sorted(set(periods))
    if any(p.startswith("2026") for p in ps):
        r = load_ranking(raw_dir)
        frames.append(r.filter(pl.col("window").is_in([p for p in ps if p.startswith("2026")])).with_columns(pl.col("window").alias("period")))
    for p in ps:
        if p.startswith("2025"):
            frames.append(load_training_month(int(p[-2:]), raw_dir).with_columns(pl.lit(p).alias("period")))
    mov = pl.concat(frames, how="vertical_relaxed")
    arr = mov.filter((pl.col("phase") == "ARR") & ~pl.col("arr_same_flight_dep").fill_null(False) & pl.col("t").is_not_null())
    return arr.select("airport", "period", pl.col("t").cast(pl.Float64), "runway", pl.col("arr_taxiin").cast(pl.Float64).alias("taxi"))


def arrival_neighbours(f_group: np.ndarray, f_t: np.ndarray, a_group: np.ndarray, a_t: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k preceding (landing <= take-off) + k following (landing > take-off) arrivals of the same group.

    Returns (idx into the arrival table, dt = landing - take-off), shape (N, 2k), nearest-first; -1 / 0 where
    there is none.  Groups are int64 ids shared by both sides (-1 = no group).
    """

    n = f_t.shape[0]
    idx = np.full((n, 2 * k), -1, dtype=np.int32)
    dt = np.zeros((n, 2 * k), dtype=np.float32)
    ok_a = a_group >= 0
    a_rows = np.flatnonzero(ok_a)
    a_ord = a_rows[np.lexsort((a_rows, a_t[a_rows], a_group[a_rows]))]
    ag, at = a_group[a_ord], a_t[a_ord]
    rows = np.flatnonzero(f_group >= 0)
    if rows.size == 0 or a_ord.size == 0:
        return idx, dt
    g = f_group[rows]
    lo = np.searchsorted(ag, g, side="left")  # the group's block in the sorted arrival table
    hi = np.searchsorted(ag, g, side="right")
    # position of the first arrival landing strictly after the take-off, within the group block
    pos = np.empty(rows.size, dtype=np.int64)
    for gid in np.unique(g):
        sel = g == gid
        a, b = lo[sel][0], hi[sel][0]
        pos[sel] = a + np.searchsorted(at[a:b], f_t[rows[sel]], side="right")
    for j in range(1, k + 1):
        p = pos - j  # preceding
        v = p >= lo
        idx[rows[v], j - 1] = a_ord[p[v]]
        dt[rows[v], j - 1] = (at[p[v]] - f_t[rows[v]]).astype(np.float32)
        q = pos + j - 1  # following
        v = q < hi
        idx[rows[v], k + j - 1] = a_ord[q[v]]
        dt[rows[v], k + j - 1] = (at[q[v]] - f_t[rows[v]]).astype(np.float32)
    return idx, dt


def arrival_arrays(frame: pl.DataFrame, t: np.ndarray, k: int, arr: pl.DataFrame, taxi_in: bool) -> dict[str, np.ndarray]:
    """Arrival-token arrays for ``frame``'s rows (month-isolated by period)."""

    from prc_challenge import arrival_guard

    keys = pl.concat([frame.select(pl.col("AIRPORT").cast(pl.String).alias("a"), pl.col("period").cast(pl.String).alias("p")),
                      arr.select(pl.col("airport").cast(pl.String).alias("a"), pl.col("period").cast(pl.String).alias("p"))])
    codes = keys.select(pl.concat_str(["a", "p"], separator="|").cast(pl.Categorical).to_physical().cast(pl.Int64)).to_series().to_numpy()
    fg, ag_ = codes[: frame.height].copy(), codes[frame.height :].copy()
    idx, dt = arrival_neighbours(fg, t, ag_, arr["t"].to_numpy(), k)
    has = idx >= 0
    nb = np.where(has, idx, 0)
    rw = pl.concat([frame["RUNWAY_mvt"].cast(pl.String), arr["runway"].cast(pl.String)])
    rc = rw.cast(pl.Categorical).to_physical().cast(pl.Int64).fill_null(-1).to_numpy()
    f_rwy, a_rwy = rc[: frame.height], rc[frame.height :]
    same = has & (a_rwy[nb] == f_rwy[:, None]) & (f_rwy[:, None] >= 0)
    out = {"arr_dt": dt, "arr_mask": has.astype(np.int8), "arr_rwy": same.astype(np.int8)}
    if taxi_in:
        tx = arrival_guard.guard_values(arr["taxi"].to_numpy())[nb]
        ok = has & ~np.isnan(tx)
        out["arr_taxi"] = np.where(ok, tx, 0.0).astype(np.float32)
        out["arr_taxi_ok"] = ok.astype(np.int8)
    return out


# --------------------------------------------------------------------------- inputs


def build_inputs(frame: pl.DataFrame, pre: "tnm.Preprocessor", k_airport: int = K_AIRPORT, k_runway: int = K_RUNWAY,
                 k_arrivals: int = 0, arr_taxi_in: bool = False, arrivals: pl.DataFrame | None = None) -> dict[str, np.ndarray]:
    """Every model input of ``frame``'s rows.  Reads feature / id / clock columns only (+ landing clocks)."""

    cat, dense = pre.transform(frame)
    ids = frame["MVT_ID_mvt"].to_numpy()
    t = mvt_times(ids, frame["period"].unique().to_list())
    nb_idx, nb_dt = context_arrays(frame, t, k_airport, k_runway)
    clock = np.nan_to_num(frame.select([pl.col(c).cast(pl.Float64) for c in CLOCKS]).to_numpy(), nan=0.0).astype(np.float32)
    out = {"cat": cat.astype(np.int16), "dense": dense, "ids": ids, "nb_idx": nb_idx, "nb_dt": nb_dt, "clock": clock}
    if k_arrivals:
        arr = arrivals if arrivals is not None else arrival_table(frame["period"].unique().to_list())
        out.update(arrival_arrays(frame, t, k_arrivals, arr, arr_taxi_in))
    return out


def build_targets(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    out = {f"off_{name}": spec.offset(frame).astype(np.float64) for name, spec in TARGETS.items()}
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("split", choices=sorted(SPLITS))
    p.add_argument("--force", action="store_true")
    p.add_argument("--k-airport", type=int, default=K_AIRPORT)
    p.add_argument("--k-runway", type=int, default=K_RUNWAY)
    p.add_argument("--k-arrivals", type=int, default=0, help="arrival tokens per side (0 = none, the default cache)")
    p.add_argument("--arr-taxi-in", action="store_true", help="also store guarded arrival taxi-in (needs --k-arrivals)")
    a = p.parse_args(argv)
    if a.arr_taxi_in and not a.k_arrivals:
        raise SystemExit("--arr-taxi-in needs --k-arrivals")
    spec = SPLITS[a.split]
    suffix = cache_suffix(a.k_airport, a.k_runway, a.k_arrivals, a.arr_taxi_in)
    out = cache_dir(a.split, suffix)
    if (out / "meta.json").exists() and not a.force:
        log(f"{out} exists")
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    run = spec["run"]
    cats, nums = ma.feature_lists_for_run(run)
    keep = list(dict.fromkeys(["MVT_ID_mvt", "period", "month", "nm_unmatched", s4.LABEL_COLUMN, *CLOCKS, "unimpeded_p10", *cats, *nums]))
    train = pl.read_parquet(run / "train_matrix.parquet", columns=keep)
    extra_info = None
    if spec.get("extra") is not None:
        train, nums, extra_info = append_extra(train, Path(spec["extra"]), nums)
    if spec["holdout"] is not None:
        h = spec["holdout"]
        hs = list(h) if isinstance(h, (tuple, list)) else [h]
        if any(x in ma.FORBIDDEN_TRAIN_MONTHS for x in hs) or not set(hs) <= set(train["month"].unique().to_list()):
            raise SystemExit("holdout must be D months of the matrix")
        fit = train.filter(~pl.col("month").is_in(hs))
        ev = train.filter(pl.col("month").is_in(hs))
        del train
    else:
        fit = train
        ev = pl.read_parquet(run / "eval_matrix.parquet", columns=[c for c in keep])
    log(f"split {a.split}: fit={fit.shape} ev={ev.shape}")
    pre = tnm.Preprocessor(cats, nums).fit(fit)
    meta = {
        "split": a.split,
        "run": str(run),
        "train_sha256": s4.sha256_file(run / "train_matrix.parquet"),
        "eval_sha256": s4.sha256_file(run / "eval_matrix.parquet"),
        "holdout": list(spec["holdout"]) if isinstance(spec["holdout"], tuple) else spec["holdout"],
        "cats": cats,
        "nums": nums,
        "nan_cols": pre.nan_cols,
        "sizes": pre.sizes,
        "n_dense": pre.n_dense,
        "preprocessor": pre.describe(),
        "k_airport": a.k_airport,
        "k_runway": a.k_runway,
        "clocks": CLOCKS,
        "targets": {k: v.as_dict() for k, v in TARGETS.items()},
    }
    if extra_info is not None:
        meta["extra"] = extra_info
    if suffix:
        meta.update(k_arrivals=a.k_arrivals, arr_taxi_in=a.arr_taxi_in, cache_suffix=suffix)
        if a.arr_taxi_in:
            from prc_challenge import arrival_guard

            meta["arrival_guard"] = [arrival_guard.TAXI_IN_MIN_SEC, arrival_guard.TAXI_IN_MAX_SEC]
    for name, frame in (("fit", fit), ("ev", ev)):
        arrays = build_inputs(frame, pre, a.k_airport, a.k_runway, a.k_arrivals, a.arr_taxi_in)
        arrays.update(build_targets(frame))
        if name == "ev" and spec["holdout"] is None:
            # JAN_JUL / ranking rows: never stored (the one confirmation look scores through ma.score_janjul)
            arrays["label"] = np.full(frame.height, np.nan)
        else:
            arrays["label"] = frame[s4.LABEL_COLUMN].cast(pl.Float64).to_numpy()
        arrays["month"] = frame["month"].cast(pl.Int16).to_numpy()
        arrays["unmatched"] = frame["nm_unmatched"].cast(pl.Int8).to_numpy()
        for k, v in arrays.items():
            np.save(out / f"{name}_{k}.npy", np.ascontiguousarray(v))
        meta[f"{name}_rows"] = frame.height
        meta[f"{name}_nb_coverage"] = float((arrays["nb_idx"] >= 0).mean())
        log(f"{name}: rows={frame.height} nb coverage={meta[f'{name}_nb_coverage']:.3f}")
        del arrays
    # the preprocessor itself, so the leakage test and later fits can re-use it
    import pickle

    with open(out / "preprocessor.pkl", "wb") as fh:
        pickle.dump({"cats": pre.cats, "nums": pre.nums, "vocab": pre.vocab, "quantiles": pre.quantiles, "nan_cols": pre.nan_cols}, fh)
    meta["seconds"] = round(time.perf_counter() - t0, 1)
    s4.atomic_json(out / "meta.json", meta)
    log(f"cache {out} written in {meta['seconds']} s")


if __name__ == "__main__":
    main()
