# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""M2: following / neighbour NM-planning context for departures.

The focal departure has planned off-block ``P`` (EOBT_1, else IOBT, else
SCHED) and takeoff ``T``; its planning delay is ``D = T - P``.  A long ``D`` is
either gate delay (ATFM slot, late boarding, airline disruption) or taxi.  Other
flights' observable planning and takeoff times help separate the two:

* overtaking: flights planned before ``P`` that took off after ``T`` and
  flights planned after ``P`` that took off before ``T`` (airport and runway);
* planning-order neighbours: delay of the flights immediately before / after
  in planned order (airport and runway);
* windowed delay distribution of flights planned near ``P`` (airport, runway,
  same destination, same airline prefix) and the focal excess over it;
* takeoff-order neighbours on the same runway (delay and next-takeoff gap,
  future departure counts);
* arrival taxi-in times of other flights around ``T`` (surface congestion).

Inputs are the masked movement table only: movement times, plan times, runway,
destination, airline prefix and ARR in-block of arrivals.  Departure
BLOCK/TAXITIME is never available.  Context never crosses a window.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prc_challenge.matched_context.movements import check_movement_table

M2_VERSION = "m2-neighbor-context-v1"


def _groups(keys: np.ndarray) -> list[np.ndarray]:
    """Row indices per distinct integer key (keys >= 0 only)."""

    order = np.argsort(keys, kind="stable")
    k = keys[order]
    cuts = np.flatnonzero(np.diff(k)) + 1
    out = []
    for part in np.split(order, cuts):
        if part.size and keys[part[0]] >= 0:
            out.append(part)
    return out


def _codes(*cols: pl.Series) -> np.ndarray:
    """Dense integer code of the composite key; -1 when any part is null."""

    frame = pl.DataFrame({f"c{i}": c.cast(pl.String) for i, c in enumerate(cols)})
    key = frame.select(pl.concat_str(pl.all(), separator="\x1f").alias("k"))["k"]
    return (key.rank("dense").cast(pl.Int64) - 1).fill_null(-1).to_numpy()


def _planned_window_stats(
    idx: np.ndarray,
    P: np.ndarray,
    T: np.ndarray,
    D: np.ndarray,
    windows: tuple[int, ...],
    feat: dict[str, np.ndarray],
    prefix: str,
    *,
    overtaking: bool,
    neighbours: bool,
    quantiles: bool,
) -> None:
    """Fill planned-time window statistics for one group of departure rows."""

    ok = np.isfinite(P[idx])
    idx = idx[ok]
    if idx.size == 0:
        return
    order = np.lexsort((T[idx], P[idx]))
    g = idx[order]
    pg, tg, dg = P[g], T[g], D[g]
    lo_eq = np.searchsorted(pg, pg, side="left")
    hi_eq = np.searchsorted(pg, pg, side="right")
    if neighbours:
        prev_d = np.full(g.size, np.nan)
        next_d = np.full(g.size, np.nan)
        prev_d[1:] = dg[:-1]
        next_d[:-1] = dg[1:]
        feat[f"{prefix}prevP_delay"][g] = prev_d
        feat[f"{prefix}nextP_delay"][g] = next_d
    for w in windows:
        lo = np.searchsorted(pg, pg - w, side="left")
        hi = np.searchsorted(pg, pg + w, side="right")
        med = np.full(g.size, np.nan)
        q75 = np.full(g.size, np.nan)
        before_after = np.full(g.size, np.nan)
        after_before = np.full(g.size, np.nan)
        same_before = np.full(g.size, np.nan)
        for i in range(g.size):
            a, b = lo[i], hi[i]
            if b - a > 1:
                vals = np.concatenate((dg[a:i], dg[i + 1 : b]))
                med[i] = np.median(vals)
                if quantiles:
                    q75[i] = np.quantile(vals, 0.75)
            if overtaking:
                before_after[i] = np.count_nonzero(tg[a : lo_eq[i]] > tg[i])
                after_before[i] = np.count_nonzero(tg[hi_eq[i] : b] < tg[i])
                same_before[i] = np.count_nonzero(tg[lo_eq[i] : hi_eq[i]] < tg[i])
        feat[f"{prefix}med_delay_pm{w}s"][g] = med
        feat[f"{prefix}n_pm{w}s"][g] = (hi - lo - 1).astype(np.float64)
        if quantiles:
            feat[f"{prefix}q75_delay_pm{w}s"][g] = q75
        if overtaking:
            feat[f"{prefix}n_planned_before_took_after_{w}s"][g] = before_after
            feat[f"{prefix}n_planned_after_took_before_{w}s"][g] = after_before
            if w == windows[0]:
                feat[f"{prefix}n_sameplan_took_before"][g] = same_before


def _spec() -> dict[str, dict]:
    return {
        "m2_": dict(windows=(900, 1800, 3600), overtaking=True, neighbours=True, quantiles=True),
        "m2_rwy_": dict(windows=(1800, 3600), overtaking=True, neighbours=True, quantiles=False),
        "m2_ades_": dict(windows=(10800,), overtaking=False, neighbours=False, quantiles=False),
        "m2_al_": dict(windows=(3600,), overtaking=False, neighbours=False, quantiles=False),
    }


def build_neighbor_context_features(mov: pl.DataFrame, *, variant: str = "v1") -> pl.DataFrame:
    """Return one row per DEP movement keyed by ``MVT_ID_mvt`` with ``m2_*`` features.

    ``variant="v1s"`` is v1 with one change: in the arrival taxi-in statistics of a
    departure, ARR rows that share its ``FLIGHT_ID_mvt`` (air returns) are left out
    for that departure only.  ``variant="v2"`` drops such ARR rows globally and adds
    runway-queue columns.
    """

    if variant not in ("v1", "v1s", "v2"):
        raise ValueError(f"unknown M2 variant {variant!r}")
    check_movement_table(mov)
    dep = mov.filter(pl.col("phase") == "DEP")
    arr = mov.filter((pl.col("phase") == "ARR") & pl.col("arr_taxiin").is_not_null())
    if variant == "v2":
        arr = arr.filter(~pl.col("arr_same_flight_dep"))
    n = dep.height
    T = dep["t"].cast(pl.Float64).to_numpy()
    P = dep["plan"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    D = T - P

    names: list[str] = []
    for prefix, cfg in _spec().items():
        if cfg["neighbours"]:
            names += [f"{prefix}prevP_delay", f"{prefix}nextP_delay"]
        for w in cfg["windows"]:
            names += [f"{prefix}med_delay_pm{w}s", f"{prefix}n_pm{w}s"]
            if cfg["quantiles"]:
                names.append(f"{prefix}q75_delay_pm{w}s")
            if cfg["overtaking"]:
                names += [
                    f"{prefix}n_planned_before_took_after_{w}s",
                    f"{prefix}n_planned_after_took_before_{w}s",
                ]
        if cfg["overtaking"]:
            names.append(f"{prefix}n_sameplan_took_before")
    feat = {k: np.full(n, np.nan) for k in names}

    group_keys = {
        "m2_": _codes(dep["window"], dep["airport"]),
        "m2_rwy_": _codes(dep["window"], dep["airport"], dep["runway"]),
        "m2_ades_": _codes(dep["window"], dep["airport"], dep["ades"]),
        "m2_al_": _codes(dep["window"], dep["airport"], dep["airline"]),
    }
    for prefix, cfg in _spec().items():
        for idx in _groups(group_keys[prefix]):
            _planned_window_stats(idx, P, T, D, feat=feat, prefix=prefix, **cfg)

    # Focal excess over the neighbourhood.
    feat["m2_delay_excess_pm1800s"] = D - feat["m2_med_delay_pm1800s"]
    feat["m2_rwy_delay_excess_pm1800s"] = D - feat["m2_rwy_med_delay_pm1800s"]
    feat["m2_ades_delay_excess_pm10800s"] = D - feat["m2_ades_med_delay_pm10800s"]
    feat["m2_al_delay_excess_pm3600s"] = D - feat["m2_al_med_delay_pm3600s"]
    feat["m2_overtake_net_3600s"] = (
        feat["m2_n_planned_before_took_after_3600s"] - feat["m2_n_planned_after_took_before_3600s"]
    )

    # Takeoff-order neighbours on the same runway (bidirectional).
    for k in ("m2_rwy_prevT_delay", "m2_rwy_nextT_delay", "m2_rwy_next_dep_gap", "m2_rwy_dep_next_300s", "m2_rwy_dep_next_900s", "m2_dep_next_900s"):
        feat[k] = np.full(n, np.nan)
    for idx in _groups(group_keys["m2_rwy_"]):
        order = np.lexsort((idx, T[idx]))
        g = idx[order]
        tg, dg = T[g], D[g]
        prev_d = np.full(g.size, np.nan)
        next_d = np.full(g.size, np.nan)
        gap = np.full(g.size, np.nan)
        prev_d[1:] = dg[:-1]
        next_d[:-1] = dg[1:]
        gap[:-1] = tg[1:] - tg[:-1]
        feat["m2_rwy_prevT_delay"][g] = prev_d
        feat["m2_rwy_nextT_delay"][g] = next_d
        feat["m2_rwy_next_dep_gap"][g] = gap
        for w in (300, 900):
            feat[f"m2_rwy_dep_next_{w}s"][g] = np.searchsorted(tg, tg + w, side="right") - np.searchsorted(tg, tg, side="right")
        if variant == "v2":
            after = np.searchsorted(tg, tg, side="right")
            for w in (600, 1200, 1800):
                feat.setdefault(f"m2_rwy_dep_next_{w}s", np.full(n, np.nan))[g] = np.searchsorted(tg, tg + w, side="right") - after
            feat.setdefault("m2_rwy_dep_pm600s", np.full(n, np.nan))[g] = (
                np.searchsorted(tg, tg + 600, side="right") - np.searchsorted(tg, tg - 600, side="left") - 1
            )
            # queue behind the focal: later takeoffs within 15 min that were already due (plan <= T)
            pg = P[g]
            hi = np.searchsorted(tg, tg + 900, side="right")
            due = np.full(g.size, np.nan)
            for i in range(g.size):
                due[i] = np.count_nonzero(pg[after[i] : hi[i]] <= tg[i])
            feat.setdefault("m2_rwy_due_next_900s", np.full(n, np.nan))[g] = due
    for idx in _groups(group_keys["m2_"]):
        tg = np.sort(T[idx])
        feat["m2_dep_next_900s"][idx] = np.searchsorted(tg, T[idx] + 900, side="right") - np.searchsorted(tg, T[idx], side="right")

    # Arrival taxi-in around the takeoff (other flights' ARR in-block).
    for k in ("m2_arr_taxiin_med_pm1200s", "m2_arr_taxiin_p75_pm1200s", "m2_arr_n_pm1200s", "m2_arr_taxiin_med_prev1800s"):
        feat[k] = np.full(n, np.nan)
    dkey = _codes(dep["window"], dep["airport"])
    akey_frame = pl.concat(
        [dep.select("window", "airport"), arr.select("window", "airport")], how="vertical"
    )
    both = _codes(akey_frame["window"], akey_frame["airport"])
    dcode, acode = both[:n], both[n:]
    aL = arr["t"].cast(pl.Float64).to_numpy()
    aTau = arr["arr_taxiin"].cast(pl.Float64).to_numpy()
    for idx in _groups(dcode):
        sel = np.flatnonzero(acode == dcode[idx[0]])
        if sel.size == 0:
            continue
        o = np.argsort(aL[sel], kind="stable")
        lg, taug = aL[sel][o], aTau[sel][o]
        tq = T[idx]
        lo = np.searchsorted(lg, tq - 1200, side="left")
        hi = np.searchsorted(lg, tq + 1200, side="right")
        lo30 = np.searchsorted(lg, tq - 1800, side="left")
        hi0 = np.searchsorted(lg, tq, side="right")
        med = np.full(idx.size, np.nan)
        p75 = np.full(idx.size, np.nan)
        prev = np.full(idx.size, np.nan)
        for i in range(idx.size):
            if hi[i] > lo[i]:
                v = taug[lo[i] : hi[i]]
                med[i] = np.median(v)
                p75[i] = np.quantile(v, 0.75)
            if hi0[i] > lo30[i]:
                prev[i] = np.median(taug[lo30[i] : hi0[i]])
        feat["m2_arr_taxiin_med_pm1200s"][idx] = med
        feat["m2_arr_taxiin_p75_pm1200s"][idx] = p75
        feat["m2_arr_n_pm1200s"][idx] = (hi - lo).astype(np.float64)
        feat["m2_arr_taxiin_med_prev1800s"][idx] = prev
        if variant == "v2":
            feat.setdefault("m2_arr_next_900s", np.full(n, np.nan))[idx] = (
                np.searchsorted(lg, tq + 900, side="right") - np.searchsorted(lg, tq, side="right")
            )

    if variant == "v1s":
        # Recompute the four arrival columns for departures whose own flight has an ARR row
        # at the same airport and window, leaving that flight's ARR rows out.
        a_fid = arr["flight_id"].to_numpy()
        d_fid = dep["flight_id"].to_numpy()
        own = arr.filter(pl.col("flight_id").is_not_null()).select("window", "airport", "flight_id").unique()
        hit = dep.with_row_index("_i").join(own, on=["window", "airport", "flight_id"], how="semi")["_i"].to_numpy()
        for i in hit:
            sel = np.flatnonzero((acode == dcode[i]) & ~(a_fid == d_fid[i]))
            lg_all, tau_all = aL[sel], aTau[sel]
            o = np.argsort(lg_all, kind="stable")
            lg, taug = lg_all[o], tau_all[o]
            tq = T[i]
            lo, hi = np.searchsorted(lg, tq - 1200, side="left"), np.searchsorted(lg, tq + 1200, side="right")
            lo30, hi0 = np.searchsorted(lg, tq - 1800, side="left"), np.searchsorted(lg, tq, side="right")
            feat["m2_arr_taxiin_med_pm1200s"][i] = np.median(taug[lo:hi]) if hi > lo else np.nan
            feat["m2_arr_taxiin_p75_pm1200s"][i] = np.quantile(taug[lo:hi], 0.75) if hi > lo else np.nan
            feat["m2_arr_n_pm1200s"][i] = float(hi - lo)
            feat["m2_arr_taxiin_med_prev1800s"][i] = np.median(taug[lo30:hi0]) if hi0 > lo30 else np.nan

    out = dep.select(pl.col("mvt_id").alias("MVT_ID_mvt"))
    return out.with_columns(
        [pl.Series(k, np.asarray(v, dtype=np.float32)).fill_nan(None) for k, v in feat.items()]
    )
