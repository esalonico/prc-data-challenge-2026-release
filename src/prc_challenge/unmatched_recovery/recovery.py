# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""SA-UNM4 (2026-09-26): unmatched recovery arms on top of CAP_B + P1 + U-SENS-G.

Pre-registration: ``progress-review/unmatched-recovery-20260926/preregistration.json``.
Spec: ``progress-review/2026-09-26-retro-audit-C.md`` sections 4.4-4.8 and 5.

Arms (each a deterministic post-hoc change to the persisted components; no model is fitted):

* **A, soft E17:** on E17 rows, ``q * max(sd, 0) + (1 - q) * T`` instead of the hard delay copy.
* **B, LIRF month regime:** on LIRF gated rows (sd >= 3600) of month m, shift logit(p) by one
  constant so the month's mean p equals ``clip(a + b * arr_eq_m, 0.05, 0.95)``; ``arr_eq_m`` is the
  share of that month's LIRF arrival rows with |BLOCK - SCHED| <= 60 s (ARR rows only).
* **C, destination-arrival clock link (D-054):** an unmatched departure whose flight lands at
  another challenge airport takes that arrival row's LOBT/AOBT (NM clock fields only; never its
  BLOCK/taxi-in label). p = 0 where |SCHED - LOBT| <= 3606 s; ``E' <- 0.5 * min(MVT - AOBT, 3606)
  + 0.5 * E'`` where MVT - AOBT > 0.
* **D, expert ceiling (insurance):** at non-LIRF airports ``E_cap <- min(E_cap, min(max(sd, 0) + 1800,
  7200))`` before the sensor blend (7200 where sd is missing).

Composition order (fixed): D ceiling -> sensor blend (U-SENS-G, deployed p) -> C AOBT blend ->
p: B shift, then C zero -> mixture (3600 s gate) -> LIRF 0.5 T blend (P1) -> E17/E18 + clip ->
A on E17 rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import brentq

from prc_challenge.unmatched_recovery.noncopy_expert import GATE_SEC, mixture_prediction
from prc_challenge.unmatched_recovery.polish import LAMBDA, blend_on_mask
from prc_challenge.unmatched_recovery.schema import compose_experimental_unmatched
from prc_challenge.unmatched_recovery.sensor import S_CEILING_G, sensor_blend

FOLD_OF_MONTH = {2: "2_8", 8: "2_8", 3: "3_9", 9: "3_9", 4: "4_10", 10: "4_10", 5: "5_6", 6: "5_6"}
FOLDS = ("2_8", "3_9", "4_10", "5_6")

E17_MIN_SD = 14400.0
NEAR_COPY_TOL = 300.0  # Arm A: |y - max(sd, 0)| <= 300 s counts as a delay copy
ARR_EQ_TOL = 60.0  # Arm B: |BLOCK - SCHED| <= 60 s on ARR rows
B_TARGET_CLIP = (0.05, 0.95)
LINK_MIN_SEC, LINK_MAX_SEC = 600.0, 6 * 3600.0  # Arm C: landing 10 min .. 6 h after takeoff
LINK_WINDOW_SEC = 3606.0
AOBT_W = 0.5
AOBT_CAP = 3606.0
D_MARGIN_SEC = 1800.0
D_ABS_SEC = 7200.0


@dataclass
class Components:
    """Label-free inputs of the current unmatched system for one population (D, JAN_JUL or ranking)."""

    mvt_id: np.ndarray
    month: np.ndarray
    airport: np.ndarray
    sd: np.ndarray  # takeoff - SCHED (s), NaN if SCHED missing
    days: np.ndarray  # takeoff_minus_sched_days
    type_missing: np.ndarray
    p: np.ndarray  # deployed P(copy)
    e_cap: np.ndarray  # CAP_B expert (min(E_nc, max(sd,0)+3606))
    s: np.ndarray  # sensor S (NaN if not covered)
    cov: np.ndarray  # adsb_coverage (NaN if none)
    t: np.ndarray  # P1 T expert (LIRF)
    sched_minus_lobt: np.ndarray = field(default=None)  # type: ignore[assignment]  # Arm C link, NaN if no link
    taxi_aobt: np.ndarray = field(default=None)  # type: ignore[assignment]  # MVT - AOBT_dest, NaN if none

    def __post_init__(self) -> None:
        n = len(self.mvt_id)
        for name in ("month", "airport", "sd", "days", "type_missing", "p", "e_cap", "s", "cov", "t"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"component {name} has the wrong length")
        if self.sched_minus_lobt is None:
            self.sched_minus_lobt = np.full(n, np.nan)
        if self.taxi_aobt is None:
            self.taxi_aobt = np.full(n, np.nan)

    @property
    def n(self) -> int:
        return len(self.mvt_id)

    @property
    def sdp(self) -> np.ndarray:
        sd = np.asarray(self.sd, dtype=np.float64)
        return np.where(np.isnan(sd), 0.0, np.maximum(sd, 0.0))

    @property
    def lirf(self) -> np.ndarray:
        return np.asarray(self.airport) == "LIRF"

    @property
    def gated(self) -> np.ndarray:
        sd = np.asarray(self.sd, dtype=np.float64)
        return np.isfinite(sd) & (sd >= GATE_SEC)

    @property
    def e17(self) -> np.ndarray:
        sd = np.asarray(self.sd, dtype=np.float64)
        return self.lirf & (np.asarray(self.type_missing) == 1) & (np.asarray(self.days) == 0) & np.isfinite(sd) & (sd >= E17_MIN_SD)

    @property
    def e18(self) -> np.ndarray:
        return self.lirf & (np.asarray(self.days) >= 1)

    @property
    def linked(self) -> np.ndarray:
        return np.isfinite(np.asarray(self.sched_minus_lobt, dtype=np.float64))

    @property
    def in_window(self) -> np.ndarray:
        sml = np.asarray(self.sched_minus_lobt, dtype=np.float64)
        return np.isfinite(sml) & (np.abs(np.nan_to_num(sml, nan=np.inf)) <= LINK_WINDOW_SEC)

    @property
    def aobt_usable(self) -> np.ndarray:
        ta = np.asarray(self.taxi_aobt, dtype=np.float64)
        return self.linked & np.isfinite(ta) & (np.nan_to_num(ta, nan=-1.0) > 0)


@dataclass(frozen=True)
class ArmSpec:
    """Which arms are on, with their fitted (label-derived) parameters."""

    a_q: float | np.ndarray | None = None  # Arm A: scalar q or per-row q (cross-fitted)
    b_logit_shift: np.ndarray | None = None  # Arm B: per-row logit shift (0 outside the cell)
    c: bool = False
    d: bool = False
    c0: bool = False  # amendment 1 (D-056): arm C's p = 0 on in-window rows only, no AOBT blend

    @property
    def name(self) -> str:
        on = [k for k, v in (("A", self.a_q is not None), ("B", self.b_logit_shift is not None), ("C", self.c), ("C0", self.c0), ("D", self.d)) if v]
        return "+".join(on) if on else "control"


def _logit(q: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(q / (1 - q))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def expert_ceiling_d(e_cap: np.ndarray, sd: np.ndarray, airport: np.ndarray) -> np.ndarray:
    """Arm D: non-LIRF E_cap <= min(max(sd,0)+1800, 7200); 7200 where sd is missing."""

    e = np.asarray(e_cap, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    bound = np.where(np.isfinite(sd), np.minimum(np.maximum(np.nan_to_num(sd, nan=0.0), 0.0) + D_MARGIN_SEC, D_ABS_SEC), D_ABS_SEC)
    return np.where(np.asarray(airport) != "LIRF", np.minimum(e, bound), e)


def compose(c: Components, arm: ArmSpec = ArmSpec()) -> np.ndarray:
    """The current unmatched route (CAP_B + P1 + U-SENS-G, E17/E18, clip) with the selected arms."""

    e = np.asarray(c.e_cap, dtype=np.float64)
    if arm.d:
        e = expert_ceiling_d(e, c.sd, c.airport)
    e, _ = sensor_blend(e, c.s, c.sd, c.cov, c.airport, c.p, s_ceiling=S_CEILING_G)
    p = np.asarray(c.p, dtype=np.float64)
    if arm.c:
        use = c.aobt_usable
        ta = np.nan_to_num(np.asarray(c.taxi_aobt, dtype=np.float64), nan=0.0)
        e = np.where(use, AOBT_W * np.minimum(ta, AOBT_CAP) + (1.0 - AOBT_W) * e, e)
    if arm.b_logit_shift is not None:
        shift = np.asarray(arm.b_logit_shift, dtype=np.float64)
        p = np.where(shift != 0.0, _sigmoid(_logit(p) + shift), p)
    if arm.c or arm.c0:
        p = np.where(c.in_window, 0.0, p)
    raw = blend_on_mask(mixture_prediction(p, c.sd, e, gate_sec=GATE_SEC), c.t, c.lirf, LAMBDA)
    frame = pl.DataFrame(
        {
            "AIRPORT": np.asarray(c.airport).astype(str),
            "nm_unmatched": np.ones(c.n, dtype=np.int8),
            "type_missing": np.asarray(c.type_missing),
            "takeoff_minus_sched_days": np.asarray(c.days),
            "takeoff_minus_sched_sec": np.asarray(c.sd, dtype=np.float64),
            "prediction": raw,
        }
    )
    out = compose_experimental_unmatched(frame)["prediction"].to_numpy().astype(np.float64)
    if arm.a_q is not None:
        m = c.e17
        t = np.asarray(c.t, dtype=np.float64)
        if not np.isfinite(t[m]).all():
            raise RuntimeError("Arm A: T missing on an E17 row")
        q = np.broadcast_to(np.asarray(arm.a_q, dtype=np.float64), (c.n,))
        out = np.where(m, np.maximum(q * c.sdp + (1.0 - q) * np.nan_to_num(t), 0.0), out)
    return out


# ---------------------------------------------------------------------------
# label-derived parameters (fitted only on months outside the evaluated population)
# ---------------------------------------------------------------------------


def near_copy(y: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd = np.asarray(sd, dtype=np.float64)
    sdp = np.where(np.isnan(sd), 0.0, np.maximum(sd, 0.0))
    return np.abs(np.asarray(y, dtype=np.float64) - sdp) <= NEAR_COPY_TOL


def arm_a_q_crossfit(c: Components, y: np.ndarray, fold: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Per-row q on D: the E17-cell near-copy share of the other three folds."""

    m = c.e17
    nc = near_copy(y, c.sd)
    fold = np.asarray(fold, dtype=object)
    q = np.zeros(c.n)
    qs = {}
    for f in FOLDS:
        tr = m & (fold != f)
        qs[f] = float(nc[tr].mean())
        q[m & (fold == f)] = qs[f]
    return q, qs


def month_logit_shift(p: np.ndarray, cell: np.ndarray, month: np.ndarray, targets: dict[int, float]) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
    """Per month m: one logit shift on ``cell & month == m`` so the mean p equals ``targets[m]``."""

    p = np.asarray(p, dtype=np.float64)
    shift = np.zeros(p.size)
    info: dict[int, dict[str, float]] = {}
    lg = _logit(p)
    for m, target in targets.items():
        k = cell & (np.asarray(month) == m)
        if not k.any():
            continue
        delta = brentq(lambda d: _sigmoid(lg[k] + d).mean() - target, -5.0, 5.0)
        shift[k] = delta
        info[int(m)] = {"n": int(k.sum()), "p_mean_before": float(p[k].mean()), "target": float(target), "logit_shift": float(delta), "p_mean_after": float(_sigmoid(lg[k] + delta).mean())}
    return shift, info


def fit_rate_map(arr_eq: dict[int, float], copy_rate: dict[int, float], months: list[int]) -> tuple[float, float]:
    """Linear map rate = a + b * arr_eq over ``months``. Returns (a, b)."""

    b, a = np.polyfit([arr_eq[m] for m in months], [copy_rate[m] for m in months], 1)
    return float(a), float(b)


def rate_target(a: float, b: float, arr_eq_m: float) -> float:
    return float(np.clip(a + b * arr_eq_m, *B_TARGET_CLIP))


def arr_eq_rates(arr: pl.DataFrame, airport: str = "LIRF") -> dict[tuple[int, int], dict[str, float]]:
    """Per (year, month): share of ``airport`` ARR rows with |BLOCK - SCHED| <= 60 s. ARR rows only."""

    a = arr.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == airport)).with_columns(
        year=pl.col("MVT_TIME_UTC_mvt").dt.year(),
        month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
        eq=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= ARR_EQ_TOL),
    )
    g = a.group_by("year", "month").agg(pl.len().alias("n"), pl.col("eq").mean().alias("rate")).sort("year", "month")
    return {(int(r["year"]), int(r["month"])): {"n": int(r["n"]), "rate": float(r["rate"])} for r in g.iter_rows(named=True)}


# ---------------------------------------------------------------------------
# Arm C link (label-free: never reads the arrival row's BLOCK or TAXITIME)
# ---------------------------------------------------------------------------

LINK_DEP_COLS = ("MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
LINK_ARR_COLS = ("FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt")


def destination_links(raw: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Unmatched DEP rows -> their flight's unique ARR row at another challenge airport.

    ``raw`` holds DEP and ARR rows with ``LINK_DEP_COLS | LINK_ARR_COLS`` only. The link: same
    FLIGHT_mvt, same ADEP/ADES pair, landing 10 min to 6 h after takeoff, exactly one such ARR row.
    Returns MVT_ID_mvt, sched_minus_lobt, taxi_aobt (NaN where the clock is missing) and counts.
    """

    forbidden = {"BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"} & set(raw.columns)
    if forbidden:
        raise RuntimeError(f"destination link must not see labels: {sorted(forbidden)}")
    airports = set(raw.filter(pl.col("PHASE_mvt") == "DEP")["ADEP_mvt"].unique().to_list())
    unm = raw.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_null())
    dep = unm.filter(pl.col("ADES_mvt").is_in(sorted(airports)) & pl.col("FLIGHT_mvt").is_not_null())
    arr = raw.filter(pl.col("PHASE_mvt") == "ARR").select(
        "FLIGHT_mvt",
        pl.col("ADEP_mvt").alias("A_ADEP"),
        pl.col("ADES_mvt").alias("A_ADES"),
        pl.col("MVT_TIME_UTC_mvt").alias("A_land"),
        pl.col("FLIGHT_ID_mvt").alias("A_fid"),
        pl.col("LOBT_flt").alias("A_LOBT"),
        pl.col("AOBT_3_flt").alias("A_AOBT"),
    )
    j = dep.join(arr, on="FLIGHT_mvt", how="inner").filter(
        (pl.col("A_ADEP") == pl.col("ADEP_mvt"))
        & (pl.col("A_ADES") == pl.col("ADES_mvt"))
        & (pl.col("A_land") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().is_between(LINK_MIN_SEC, LINK_MAX_SEC)
    )
    cnt = j.group_by("MVT_ID_mvt").len()
    j = j.join(cnt, on="MVT_ID_mvt").filter(pl.col("len") == 1)
    counts = {
        "unmatched_dep": unm.height,
        "dest_in_challenge_set": dep.height,
        "unique_link": j.height,
        "link_arr_nm_matched": j.filter(pl.col("A_fid").is_not_null()).height,
        "link_with_lobt": j.filter(pl.col("A_LOBT").is_not_null()).height,
        "link_with_lobt_arr_unmatched": j.filter(pl.col("A_LOBT").is_not_null() & pl.col("A_fid").is_null()).height,
    }
    j = j.filter(pl.col("A_LOBT").is_not_null())
    out = j.select(
        "MVT_ID_mvt",
        (pl.col("SCHED_TIME_UTC_mvt") - pl.col("A_LOBT")).dt.total_seconds().cast(pl.Float64).alias("sched_minus_lobt"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("A_AOBT")).dt.total_seconds().cast(pl.Float64).alias("taxi_aobt"),
    )
    counts["in_window"] = int((out["sched_minus_lobt"].abs() <= LINK_WINDOW_SEC).sum())
    return out, counts


def attach_links(c: Components, links: pl.DataFrame) -> Components:
    z = pl.DataFrame({"MVT_ID_mvt": c.mvt_id}).join(links, on="MVT_ID_mvt", how="left", maintain_order="left")
    if z.height != c.n:
        raise RuntimeError("link join changed the row count")
    return replace(
        c,
        sched_minus_lobt=z["sched_minus_lobt"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        taxi_aobt=z["taxi_aobt"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
    )


def changed_by_airport(ctl: np.ndarray, cand: np.ndarray, airport: np.ndarray) -> dict[str, Any]:
    d = np.asarray(cand, dtype=np.float64) - np.asarray(ctl, dtype=np.float64)
    ch = np.abs(d) > 1e-9
    out = {}
    for a in sorted(set(np.asarray(airport).tolist())):
        m = np.asarray(airport) == a
        out[str(a)] = {
            "n": int(m.sum()),
            "changed": int((ch & m).sum()),
            "mean_change": float(d[m].mean()),
            "mean_change_on_changed": float(d[ch & m].mean()) if (ch & m).any() else 0.0,
            "max_abs_change": float(np.abs(d[m]).max()),
            "sum_sq_change": float((d[m] ** 2).sum()),
        }
    return {"changed": int(ch.sum()), "sum_sq_change": float((d**2).sum()), "by_airport": out}
