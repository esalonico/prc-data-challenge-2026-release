# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""M1: stand-sequence / turnaround context for departures.

For a departure with takeoff ``T`` at stand ``S`` the true off-block ``B`` lies
between the in-block of its inbound aircraft and the in-block of the next
occupant of ``S``.  Neither ``B`` nor the inbound identity is observable, so the
block exposes the raw stand timeline around ``T`` and lets the model learn the
geometry:

* the latest arrivals at ``S`` that blocked in before ``T`` (ages, taxi-in,
  landing age, same type / same airline as the departure);
* the next arrivals at ``S`` at or after ``T`` (signed gaps, taxi-in, landing);
* an *inbound candidate* (latest earlier arrival at ``S`` with the same type,
  preferring the same airline prefix) and the arrival that followed it.  When
  that follower blocked in before ``T`` it is a candidate next occupant and
  ``T - in_block`` a candidate lower bound on taxi; it is exposed as a feature,
  never used to clip;
* the previous / next departure takeoff from ``S`` and the arrival count in
  between (chain consistency), and stand activity / missingness flags.

The S2 overnight block used the last two arrivals before ``T`` as a
``[lower, upper]`` taxi interval.  That is only valid when the latest arrival is
the next occupant (about 9% of departures); in the other 91% the latest arrival
is the inbound and its age is an *upper* bound.  Here no interval is asserted.

Only the masked movement table is consumed (see ``movements``): arrival in-block
times of ARR rows, movement times, schedule / EOBT / IOBT, stand, type, and
flight-number prefixes.  Departure BLOCK/TAXITIME is never available.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prc_challenge.matched_context.movements import check_movement_table

SCALE = 10**10  # epoch seconds < 1e10, so key * SCALE + epoch is an exact int64 composite
MIN_TURN_SEC = 180
M1_VERSION = "m1-stand-sequence-v1"


def _codes(dep_key: pl.Series, arr_key: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    keys = pl.concat([dep_key, arr_key]).drop_nulls().unique().sort()
    lookup = {k: i for i, k in enumerate(keys.to_list())}
    dcode = np.array([lookup.get(k, -1) if k is not None else -1 for k in dep_key.to_list()], dtype=np.int64)
    acode = np.array([lookup[k] for k in arr_key.to_list()], dtype=np.int64)
    return dcode, acode


def _f(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _take(arr: np.ndarray, idx: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(idx.size, np.nan, dtype=np.float64)
    out[valid] = arr[idx[valid]]
    return out


def _same(a: np.ndarray, idx: np.ndarray, valid: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1/0 equality of integer codes a[idx] and b; NaN when either side is unknown (-1)."""

    out = np.full(idx.size, np.nan, dtype=np.float64)
    if not valid.any():
        return out
    av = a[idx[valid]]
    bv = b[valid]
    out[valid] = np.where((av >= 0) & (bv >= 0), (av == bv).astype(np.float64), np.nan)
    return out


def _text_codes(dep_col: pl.Series, arr_col: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    both = pl.concat([dep_col.cast(pl.String), arr_col.cast(pl.String)])
    codes = both.cast(pl.Categorical).to_physical().cast(pl.Int64).fill_null(-1).to_numpy()
    return codes[: dep_col.len()], codes[dep_col.len() :]


def build_stand_sequence_features(
    mov: pl.DataFrame,
    *,
    lookback: int = 8,
    max_inbound_age_sec: int = 36 * 3600,
    variant: str = "v1",
) -> pl.DataFrame:
    """Return one row per DEP movement keyed by ``MVT_ID_mvt`` with ``m1_*`` features.

    ``variant="v2"`` adds occupant-waiting and chain-ambiguity features: arrival
    taxi-in excess over the airport-window median ARR taxi-in (a next occupant
    that waited for the stand), the second follower of the inbound, the
    occupant's airline match, and the next same-stand departure's plan.

    ``variant="v1s"`` is v1 with one change: for each departure, arrival rows that
    share its ``FLIGHT_ID_mvt`` at the same stand (air returns) are invisible to that
    departure only.  It is implemented by giving each such departure a private copy
    of its stand timeline without those rows; every other departure is untouched.
    """

    if variant not in ("v1", "v1s", "v2"):
        raise ValueError(f"unknown M1 variant {variant!r}")

    check_movement_table(mov)
    skey = pl.when(pl.col("stand").is_not_null()).then(
        pl.concat_str(["window", "airport", "stand"], separator="::")
    )
    m = mov.with_columns(skey.alias("skey"))
    dep = m.filter(pl.col("phase") == "DEP").with_row_index("_row")
    arr = m.filter(
        (pl.col("phase") == "ARR") & pl.col("skey").is_not_null() & pl.col("arr_inblock").is_not_null()
    )
    if variant == "v2":
        arr = arr.filter(~pl.col("arr_same_flight_dep"))
    dep = dep.with_columns(pl.lit(True).alias("_keep"))
    if variant == "v1s":
        own = arr.filter(pl.col("flight_id").is_not_null()).select("skey", "flight_id").unique()
        conflict = dep.join(own, on=["skey", "flight_id"], how="semi")
        if conflict.height:
            ids = conflict["mvt_id"]
            priv_dep, priv_arr = [], []
            for mid, sk, fid in conflict.select("mvt_id", "skey", "flight_id").iter_rows():
                pk = f"{sk}#private#{mid!r}"
                priv_arr.append(
                    arr.filter((pl.col("skey") == sk) & (pl.col("flight_id").is_null() | (pl.col("flight_id") != fid)))
                    .with_columns(pl.lit(pk).alias("skey"))
                )
                priv_dep.append(
                    dep.filter(pl.col("skey") == sk).with_columns(
                        pl.lit(pk).alias("skey"), (pl.col("mvt_id") == mid).alias("_keep")
                    )
                )
            dep = pl.concat(
                [dep.with_columns((~pl.col("mvt_id").is_in(ids.implode())).alias("_keep")), *priv_dep], how="vertical"
            )
            arr = pl.concat([arr, *priv_arr], how="vertical")
    n = dep.height
    dcode, acode = _codes(dep["skey"], arr["skey"])

    # Arrivals sorted by (stand key, in-block).
    a_in = arr["arr_inblock"].to_numpy().astype(np.int64)
    a_comp = acode * SCALE + a_in
    order = np.argsort(a_comp, kind="stable")
    a_comp = a_comp[order]
    A = _f(a_in[order])
    L = _f(arr["t"].to_numpy()[order])
    TAU = _f(arr["arr_taxiin"].to_numpy()[order])
    ASCHED = _f(arr["sched"].fill_null(np.iinfo(np.int64).min).to_numpy()[order])
    ASCHED[ASCHED < 0] = np.nan
    dtype_code, atype_code = _text_codes(dep["aircraft_type"], arr["aircraft_type"])
    dal_code, aal_code = _text_codes(dep["airline"], arr["airline"])
    ATYPE = atype_code[order]
    AAL = aal_code[order]

    T = _f(dep["t"].to_numpy())
    P = dep["plan"].cast(pl.Float64).to_numpy()
    DTYPE = dtype_code
    DAL = dal_code
    has = dcode >= 0
    kc = np.where(has, dcode, 0)
    q = kc * SCALE + dep["t"].to_numpy().astype(np.int64)
    start = np.searchsorted(a_comp, kc * SCALE, side="left")
    end = np.searchsorted(a_comp, (kc + 1) * SCALE, side="left")
    pos = np.searchsorted(a_comp, q, side="left")  # arrivals strictly before T: [start, pos)

    feat: dict[str, np.ndarray] = {}
    feat["m1_stand_missing"] = (~has).astype(np.float64)

    # --- latest arrivals before T and next arrivals at/after T -------------------------
    for k in (1, 2, 3):
        idx = pos - k
        ok = has & (idx >= start)
        feat[f"m1_prev{k}_age"] = T - _take(A, idx, ok)
        if k <= 2:
            feat[f"m1_prev{k}_same_type"] = _same(ATYPE, idx, ok, DTYPE)
            feat[f"m1_prev{k}_same_airline"] = _same(AAL, idx, ok, DAL)
            feat[f"m1_prev{k}_taxiin"] = _take(TAU, idx, ok)
        if k == 1:
            feat["m1_prev1_land_age"] = T - _take(L, idx, ok)
            feat["m1_prev1_minus_plan"] = _take(A, idx, ok) - P
    for k in (1, 2):
        idx = pos + k - 1
        ok = has & (idx < end)
        feat[f"m1_next{k}_gap"] = _take(A, idx, ok) - T
        if k == 1:
            feat["m1_next1_taxiin"] = _take(TAU, idx, ok)
            feat["m1_next1_land_minus_T"] = _take(L, idx, ok) - T
            feat["m1_next1_same_type"] = _same(ATYPE, idx, ok, DTYPE)
            feat["m1_next1_same_airline"] = _same(AAL, idx, ok, DAL)

    # --- inbound candidate: latest earlier arrival with same type (prefer same airline) ---
    inb_ta = np.full(n, -1, dtype=np.int64)
    inb_t = np.full(n, -1, dtype=np.int64)
    for o in range(lookback, 0, -1):  # smaller offsets overwrite: latest match wins
        idx = pos - o
        ok = has & (idx >= start)
        if not ok.any():
            continue
        age = np.full(n, np.nan)
        age[ok] = T[ok] - A[idx[ok]]
        ok &= (age >= MIN_TURN_SEC) & (age <= max_inbound_age_sec)
        st = _same(ATYPE, idx, ok, DTYPE) == 1.0
        sa = _same(AAL, idx, ok, DAL) == 1.0
        inb_t = np.where(ok & st, idx, inb_t)
        inb_ta = np.where(ok & st & sa, idx, inb_ta)
    inb = np.where(inb_ta >= 0, inb_ta, inb_t)
    level = np.where(inb_ta >= 0, 2.0, np.where(inb_t >= 0, 1.0, 0.0))
    level[~has] = np.nan
    ok_inb = inb >= 0
    feat["m1_inb_level"] = level
    feat["m1_inb_age"] = T - _take(A, inb, ok_inb)
    feat["m1_inb_minus_plan"] = _take(A, inb, ok_inb) - P
    feat["m1_inb_arr_delay"] = _take(A, inb, ok_inb) - _take(ASCHED, inb, ok_inb)
    feat["m1_inb_taxiin"] = _take(TAU, inb, ok_inb)
    feat["m1_n_arr_after_inb_before_T"] = np.where(ok_inb, (pos - 1 - inb).astype(np.float64), np.nan)
    occ = inb + 1
    ok_occ = ok_inb & (occ < end)
    occ_gap = _take(A, occ, ok_occ) - T
    feat["m1_occ_gap"] = occ_gap
    feat["m1_occ_lb"] = np.where(np.isfinite(occ_gap), np.maximum(0.0, -occ_gap), np.nan)
    feat["m1_occ_taxiin"] = _take(TAU, occ, ok_occ)
    feat["m1_occ_land_minus_T"] = _take(L, occ, ok_occ) - T
    feat["m1_occ_same_type"] = _same(ATYPE, occ, ok_occ, DTYPE)
    feat["m1_stand_free_window"] = _take(A, occ, ok_occ) - _take(A, inb, ok_occ)
    if variant == "v2":
        ref = (
            mov.filter((pl.col("phase") == "ARR") & pl.col("arr_taxiin").is_not_null() & ~pl.col("arr_same_flight_dep"))
            .group_by("window", "airport")
            .agg(pl.col("arr_taxiin").median().alias("_ref"))
        )
        aref = arr.select("window", "airport").join(ref, on=["window", "airport"], how="left", maintain_order="left")
        XS = TAU - _f(aref["_ref"].to_numpy())[order]
        feat["m1_occ_excess_taxiin"] = _take(XS, occ, ok_occ)
        feat["m1_occ_same_airline"] = _same(AAL, occ, ok_occ, DAL)
        occ2 = occ + 1
        ok_occ2 = ok_occ & (occ2 < end)
        feat["m1_occ2_gap"] = _take(A, occ2, ok_occ2) - T
        ok1 = has & (pos < end)
        feat["m1_next1_excess_taxiin"] = _take(XS, pos, ok1)
        okp1 = has & (pos - 1 >= start)
        feat["m1_prev1_excess_taxiin"] = _take(XS, pos - 1, okp1)

    # --- departure chain at the same stand ---------------------------------------------
    d_comp = kc * SCALE + dep["t"].to_numpy().astype(np.int64)
    d_order = np.lexsort((np.arange(n), d_comp))
    ranked_code = np.where(has, kc, -1)[d_order]
    prev_dep_t = np.full(n, np.nan)
    next_dep_t = np.full(n, np.nan)
    same_prev = np.zeros(n, dtype=bool)
    same_prev[1:] = (ranked_code[1:] == ranked_code[:-1]) & (ranked_code[1:] >= 0)
    same_next = np.zeros(n, dtype=bool)
    same_next[:-1] = same_prev[1:]
    t_sorted = T[d_order]
    tmp = np.full(n, np.nan)
    tmp[1:] = t_sorted[:-1]
    prev_dep_t[d_order] = np.where(same_prev, tmp, np.nan)
    tmp = np.full(n, np.nan)
    tmp[:-1] = t_sorted[1:]
    next_dep_t[d_order] = np.where(same_next, tmp, np.nan)
    if variant == "v2":
        p_sorted = P[d_order]
        tmp = np.full(n, np.nan)
        tmp[:-1] = p_sorted[1:]
        next_dep_p = np.full(n, np.nan)
        next_dep_p[d_order] = np.where(same_next, tmp, np.nan)
        feat["m1_next_dep_plan_minus_T"] = next_dep_p - T
    feat["m1_prev_dep_age"] = T - prev_dep_t
    feat["m1_next_dep_gap"] = next_dep_t - T
    okp = np.isfinite(prev_dep_t)
    since = np.full(n, np.nan)
    pd_comp = kc[okp] * SCALE + prev_dep_t[okp].astype(np.int64)
    since[okp] = pos[okp] - np.searchsorted(a_comp, pd_comp, side="right")
    feat["m1_n_arr_since_prev_dep"] = since

    # --- counts in planning-relative windows -------------------------------------------
    okP = has & np.isfinite(P)
    lo = np.where(okP, np.minimum(np.nan_to_num(P, nan=0.0), T), T).astype(np.int64)
    cnt = np.full(n, np.nan)
    cnt[okP] = pos[okP] - np.searchsorted(a_comp, kc[okP] * SCALE + lo[okP], side="left")
    feat["m1_n_arr_plan_to_T"] = cnt
    for w in (1800, 3600):
        c = np.full(n, np.nan)
        c[has] = pos[has] - np.searchsorted(a_comp, q[has] - w, side="left")
        feat[f"m1_n_arr_last_{w}s"] = c

    # --- stand activity -----------------------------------------------------------------
    spans = mov.group_by("window").agg(
        ((pl.col("t").max() - pl.col("t").min()) / 86400.0).clip(lower_bound=1.0).alias("_span")
    )
    span_days = dep.select("window").join(spans, on="window", how="left", maintain_order="left")["_span"].to_numpy()
    arr_per = np.bincount(acode, minlength=int(max(acode.max(initial=-1), dcode.max(initial=-1)) + 1))
    dep_per = np.bincount(dcode[has], minlength=arr_per.size)
    feat["m1_stand_arr_per_day"] = np.where(has, arr_per[kc] / span_days, np.nan)
    feat["m1_stand_dep_per_day"] = np.where(has, dep_per[kc] / span_days, np.nan)

    out = dep.select(pl.col("mvt_id").alias("MVT_ID_mvt"), "_keep")
    out = out.with_columns([pl.Series(k, v.astype(np.float32)).fill_nan(None) for k, v in feat.items()])
    return out.filter(pl.col("_keep")).drop("_keep")
