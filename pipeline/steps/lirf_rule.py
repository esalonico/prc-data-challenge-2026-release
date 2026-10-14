# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""LIRF label-mechanism rule (D-116) on LIRF departures without an NM record. 2025 data only; no 2026 label is read.

Mechanisms of such a label, measured on the 2025 files:
  ordinary  BLOCK is the real off-block                                 -> y = real taxi
  copy      BLOCK = SCHED                                               -> y = b = MVT - SCHED
  date bug  BLOCK = local date of SCHED + real off-block clock time     -> y = real taxi + 86,400 k
            with k = local date of the off-block (take-off - 15 min) - local date of SCHED. It can only occur when the
            delay crosses local midnight: all 12 LIRF 24 h-class labels of 2025 have k = 1.

Rule (parameters in configs/final/lirf_rule_params.json):
  A  k >= 1 and 12 h <= b < 30 h:  pred' = pred + q_band (86,400 k + m - pred);  q_band = date-bug share of the 2025 rows of
     the band (12-16 / 16-20 / 20-30 h), shrunk (+0.3 / +1); m = median real-taxi part of the 2025 date-bug labels.
  B  k = 0 and 3 h <= b < 6 h:     pred' = max(0, pred + c);  c_season = minus the shrunk (n + 20) mean residual of the
     cross-fitted unmatched control on the same season's 2025 rows (summer = June-October), months 1-10.
     --b-mode crowd (D-116b, not shipped: v46 +0.45 s vs v45): c = c_cell for the cell season x crowded, shrunk (+10) toward c_season. A row is
     *crowded* when at least 3 other LIRF departures without an NM record take off within +-3 h of it: an NM-matching gap,
     whose rows are ordinary taxis (2025 copy share 28 % with > 5 such neighbours vs 79 % with none). --b-mode season = F2B.
     --b-mode fmt (D-116c, F2D; v47 236.2816 vs v45 236.4358): c = c_cell for the cell season x callsign format of the movement record, shrunk (+10) toward c_season.
     *clean* = FLIGHT_mvt is an ICAO prefix + number (e.g. RYR359); garbled / other formats (e.g. ITYTY680) mark the SOBT-fallback
     path: 2025 summer copy share 0.44 for clean vs 0.79 for other.
     --b-mode fmt_special (D-116d, F2E; v48 235.6516 vs v47 236.2816): rows of *special operators* get the schedule copy, pred' = b; the other rows the fmt shift refit
     without them. Special = the callsign prefix is not a regular scheduled carrier (>= 20 matched 2025 flights at the ten airports,
     mostly Mainline / Lowcost / Regional in NM's market segment, <50 % non-scheduled / cargo / business / GA-military); a missing
     callsign is special. 2025: 34 of 34 special-operator LIRF rows without an NM record and >= 3 h late are schedule copies.
     --b-mode fmt_special13 (D-116e, the final F2F; v49 235.6236 vs v48 235.6516): as fmt_special, and special-operator rows with k = 0 and 1 h <= b < 3 h also get pred' = b
     (2025: copy share 0.94 over 64 rows; LOMO months 1-10 -0.066 s, 9 of 10 months better).

  apply:  python pipeline/steps/lirf_rule.py apply --in <final>.parquet --ranking data/raw/final_ranking.parquet --out <final2>.parquet [--parts B]
  fit:    python pipeline/steps/lirf_rule.py fit [--check]   (refits from the 2025 files and the scorecard tables; --check compares with the config)

Validation (D-116): leave-one-month-out on 2025 months 1-10 vs the v40 system 244.270 -> 241.926; four-month scorecard full +4.83 s,
body +1.92 s, transfer gate PASS; v44 (F2 = A + B on the public IDs) 237.515 vs v43 237.86.
Confirmation split (D-116, second confirmation upload): v45 (B only) 236.4358, so B = -1.42 s and A = +1.08 s on Jan/Jul 2026.
By the pre-registered rule part A is dropped: the final uses --parts B (F2B).
D-116b: part B by season x crowding (LOMO months 1-10: B -0.878 -> -1.005 s, 8 of 10 months better); first shipped without a public reading
(both confirmation uploads used) as F2C; the owner-requested v46 read 236.8877 (+0.45 s vs v45), so the final reverted to
--parts B --b-mode season (F2B).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "final" / "lirf_rule_params.json"
TZ = "Europe/Rome"
DAY = 86400
A_BANDS = ((12, 16), (16, 20), (20, 30))
SUMMER = range(6, 11)
CROWD_H, CROWD_MIN, CELL_SHRINK = 3, 3, 10.0


def with_k_b(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        k=((pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=900)).dt.convert_time_zone(TZ).dt.date()
           - pl.col("SCHED_TIME_UTC_mvt").dt.convert_time_zone(TZ).dt.date()).dt.total_days(),
        b=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
        month=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64),
        unmatched=pl.col("FLIGHT_ID_mvt").is_null())


def lirf_dep(path_or_frame) -> pl.DataFrame:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    df = pl.read_parquet(path_or_frame, columns=cols) if not isinstance(path_or_frame, pl.DataFrame) else path_or_frame
    return with_k_b(df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")))


def clean_callsign(df: pl.DataFrame) -> np.ndarray:
    """FLIGHT_mvt is an ICAO prefix followed by a flight number (RYR359, BAW561D); null and garbled forms (ITYTY680) are not."""
    return df["FLIGHT_mvt"].str.contains(r"^[A-Z]{3}[0-9][0-9A-Z]*$").fill_null(False).to_numpy()


def regular_carriers() -> list[str]:
    """ICAO prefixes of regular scheduled carriers, profiled from their own matched 2025 flights (NM market segment / flight type)."""
    cols = ["FLIGHT_ID_mvt", "CALLSIGN_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt"]
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "training_2025-*.parquet")))
    m = pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed").filter(pl.col("FLIGHT_ID_mvt").is_not_null())
    prof = m.with_columns(pre=pl.col("CALLSIGN_flt").str.slice(0, 3)).group_by("pre").agg(
        n=pl.len(), seg=pl.col("MARKET_SEGMENT_flt").mode().first(),
        special=((pl.col("MARKET_SEGMENT_flt").is_in(["Non-Scheduled", "Cargo", "Business"])) | (pl.col("FLIGHT_TYPE_flt").is_in(["N", "M", "G", "X"]))).mean(),
        nonsched=((pl.col("MARKET_SEGMENT_flt") == "Non-Scheduled") | (pl.col("FLIGHT_TYPE_flt") == "N")).mean(),
        cargo=(pl.col("MARKET_SEGMENT_flt") == "Cargo").mean(), business=(pl.col("MARKET_SEGMENT_flt") == "Business").mean(),
        govmil=pl.col("FLIGHT_TYPE_flt").is_in(["M", "G", "X"]).mean())
    reg = prof.filter((pl.col("n") >= 20) & (pl.col("nonsched") < 0.5) & (pl.col("cargo") < 0.5) & (pl.col("business") < 0.5) & (pl.col("govmil") < 0.5)
                      & pl.col("seg").is_in(["Mainline", "Lowcost", "Regional"]))
    return sorted(reg["pre"].drop_nulls().to_list())


def special_operator(df: pl.DataFrame, regular: list[str]) -> np.ndarray:
    return (~df["FLIGHT_mvt"].str.slice(0, 3).is_in(regular)).fill_null(True).to_numpy()


def in_a(df: pl.DataFrame) -> pl.Series:
    return df["unmatched"] & (df["k"] >= 1) & (df["b"] >= A_BANDS[0][0] * 3600) & (df["b"] < A_BANDS[-1][1] * 3600)


def in_c(df: pl.DataFrame) -> pl.Series:
    return df["unmatched"] & (df["k"] == 0) & (df["b"] >= 3600) & (df["b"] < 3 * 3600)


def in_b(df: pl.DataFrame) -> pl.Series:
    return df["unmatched"] & (df["k"] == 0) & (df["b"] >= 3 * 3600) & (df["b"] < 6 * 3600)


def crowded(rows: pl.DataFrame, pool: pl.DataFrame) -> np.ndarray:
    """True where at least CROWD_MIN *other* no-NM departures of `pool` (the airport's departures) take off within +-CROWD_H h."""
    ts = pool["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy(); un = pool["unmatched"].cast(pl.Int32).to_numpy()
    o = np.argsort(ts); ts, cum = ts[o], np.cumsum(np.r_[0, un[o]])
    t = rows["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
    n = cum[np.searchsorted(ts, t + CROWD_H * 3600, side="right")] - cum[np.searchsorted(ts, t - CROWD_H * 3600)] - rows["unmatched"].cast(pl.Int32).to_numpy()
    return n >= CROWD_MIN


def apply_rule(rows: pl.DataFrame, pred: np.ndarray, prm: dict, b_mode: str = "crowd") -> np.ndarray:
    """rows: every LIRF departure of the ranking file (the crowding count is taken over them)."""
    p = pred.astype(float).copy()
    b, k, months = rows["b"].to_numpy(), rows["k"].to_numpy(), rows["month"].to_numpy()
    cr = crowded(rows, rows) if b_mode == "crowd" else None
    cl = clean_callsign(rows) if b_mode in ("fmt", "fmt_special", "fmt_special13") else None
    sp = special_operator(rows, prm["B"]["regular_carriers"]) if b_mode in ("fmt_special", "fmt_special13") else None
    if b_mode == "fmt_special13" and prm["B"]["summer"]["E"]["c"] != 0.0:     # part B active: also the 1-3 h special rows
        for i in np.where(in_c(rows).to_numpy() & sp)[0]:
            p[i] = float(b[i])
    for i in np.where(in_b(rows).to_numpy())[0]:
        cell = prm["B"]["summer" if int(months[i]) in SUMMER else "other"]
        if b_mode in ("fmt_special", "fmt_special13"):
            if sp[i]:
                p[i] = float(b[i])          # SOBT fallback: the label is take-off - SCHED
                continue
            c = cell["E"]["clean_callsign" if cl[i] else "other_callsign"]["c"]
        elif b_mode == "season":
            c = cell["c"]
        elif b_mode == "crowd":
            c = cell["crowded" if cr[i] else "isolated"]["c"]
        else:
            c = cell["clean_callsign" if cl[i] else "other_callsign"]["c"]
        p[i] = max(0.0, p[i] + c)
    for i in np.where(in_a(rows).to_numpy())[0]:
        q = next(prm["A"]["q"][f"{lo}-{hi}h"] for lo, hi in A_BANDS if lo * 3600 <= b[i] < hi * 3600)
        p[i] = p[i] + q * (DAY * int(k[i]) + prm["A"]["m"] - p[i])
    return p


def fit() -> dict:
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "training_2025-*.parquet")))
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    raw = with_k_b(pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
                   .filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF"))).with_columns(y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    raw = raw.with_columns(datebug=(pl.col("k") >= 1) & ((pl.col("y") - pl.col("k") * DAY).abs() < 4 * 3600))
    pa = raw.filter(in_a(raw))
    q = {}
    for lo, hi in A_BANDS:
        x = pa.filter((pl.col("b") >= lo * 3600) & (pl.col("b") < hi * 3600))
        q[f"{lo}-{hi}h"] = (float(x["datebug"].sum()) + 0.3) / (x.height + 1.0)
    db = raw.filter(pl.col("datebug") & pl.col("unmatched"))
    prm = {"A": {"q": q, "m": float((db["y"] - db["k"] * DAY).median()), "n": pa.height, "n_datebug": int(pa["datebug"].sum())}, "B": {}}
    sc = ROOT / "artifacts" / "experiments" / "Scorecard4m-20261011"
    ctl = pl.read_parquet(sc / "population_m1_10.parquet", columns=["MVT_ID_mvt"]).join(
        pl.read_parquet(sc / "systems" / "v40_m1_10.parquet", columns=["MVT_ID_mvt", "prediction"]), on="MVT_ID_mvt")
    regular = regular_carriers()
    t = raw.with_columns(crowded=pl.Series(crowded(raw, raw)), clean=pl.Series(clean_callsign(raw)),
                         special=pl.Series(special_operator(raw, regular))).join(ctl, on="MVT_ID_mvt")
    t = t.filter(in_b(t))
    for name, months in (("summer", list(SUMMER)), ("other", [m for m in range(1, 13) if m not in SUMMER])):
        g = t.filter(pl.col("month").is_in(months))
        c_s = float(-(g["prediction"] - g["y"]).sum() / (g.height + 20.0))
        prm["B"][name] = {"c": c_s, "n": g.height}
        for col, cname, flag in (("crowded", "crowded", True), ("crowded", "isolated", False), ("clean", "clean_callsign", True), ("clean", "other_callsign", False)):
            h = g.filter(pl.col(col) == flag)
            prm["B"][name][cname] = {"c": float((-(h["prediction"] - h["y"]).sum() + CELL_SHRINK * c_s) / (h.height + CELL_SHRINK)), "n": h.height}
        gn = g.filter(~pl.col("special"))
        c_e = float(-(gn["prediction"] - gn["y"]).sum() / (gn.height + 20.0))
        prm["B"][name]["E"] = {"c": c_e, "n": gn.height, "n_special": int(g["special"].sum()), "copy_share_special": float(((g.filter(pl.col("special"))["y"] - g.filter(pl.col("special"))["b"]).abs() <= 300).mean() or 0.0)}
        for cname, flag in (("clean_callsign", True), ("other_callsign", False)):
            h = gn.filter(pl.col("clean") == flag)
            prm["B"][name]["E"][cname] = {"c": float((-(h["prediction"] - h["y"]).sum() + CELL_SHRINK * c_e) / (h.height + CELL_SHRINK)), "n": h.height}
    prm["B"]["regular_carriers"] = regular
    tc = raw.with_columns(special=pl.Series(special_operator(raw, regular)))
    tc = tc.filter(in_c(tc) & pl.col("special"))
    prm["B"]["C_1to3h_special"] = {"n": tc.height, "copy_share": float(((tc["y"] - tc["b"]).abs() <= 300).mean() or 0.0)}
    prm["B"]["crowd_rule"] = {"window_h": CROWD_H, "min_other_no_nm_departures": CROWD_MIN, "cell_shrink": CELL_SHRINK}
    return prm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--check", action="store_true", help="compare with configs/final/lirf_rule_params.json instead of writing it")
    a = sub.add_parser("apply")
    a.add_argument("--in", dest="inp", required=True, help="final file (MVT_ID_mvt, TAXITIME_SEC_mvt)")
    a.add_argument("--ranking", required=True, help="the ranking file the final file predicts (final_ranking.parquet)")
    a.add_argument("--out", required=True)
    a.add_argument("--parts", default="AB", choices=("AB", "A", "B"), help="which parts of the rule to apply (final: B, D-116)")
    a.add_argument("--b-mode", default="fmt_special", choices=("season", "crowd", "fmt", "fmt_special", "fmt_special13"), help="part B: special operators -> copy + season x callsign format (the final, F2E, D-116d); fmt = F2D; season = F2B; crowd = F2C")
    args = ap.parse_args()
    if args.cmd == "fit":
        prm = fit()
        if args.check:
            ref = json.loads(CONFIG.read_text())["params"]
            same = json.dumps(prm, sort_keys=True) == json.dumps(ref, sort_keys=True)
            print(json.dumps(prm, indent=1)); print("matches config:", same)
            raise SystemExit(0 if same else 1)
        CONFIG.write_text(json.dumps({"rule": "D-116", "params": prm}, indent=1) + "\n")
        print(f"wrote {CONFIG}")
        return
    prm = json.loads(CONFIG.read_text())["params"]
    if "A" not in args.parts:
        prm["A"]["q"] = {k: 0.0 for k in prm["A"]["q"]}
    if "B" not in args.parts:
        for v in (prm["B"]["summer"], prm["B"]["other"]):
            for key in ("crowded", "isolated", "clean_callsign", "other_callsign"):
                v[key]["c"] = 0.0
            v["c"] = 0.0
            v["E"]["c"] = 0.0; v["E"]["clean_callsign"]["c"] = 0.0; v["E"]["other_callsign"]["c"] = 0.0
    base = pl.read_parquet(args.inp)
    rows = lirf_dep(args.ranking).join(base.rename({"TAXITIME_SEC_mvt": "pred"}), on="MVT_ID_mvt")
    new = apply_rule(rows, rows["pred"].to_numpy(), prm, args.b_mode)
    ch = rows.select("MVT_ID_mvt").with_columns(new=pl.Series(new)).filter(pl.Series(np.abs(new - rows["pred"].to_numpy()) > 1e-6))
    out = base.join(ch, on="MVT_ID_mvt", how="left").with_columns(TAXITIME_SEC_mvt=pl.coalesce("new", "TAXITIME_SEC_mvt")).select(base.columns)
    y = out["TAXITIME_SEC_mvt"].to_numpy()
    assert out.height == base.height and out["MVT_ID_mvt"].equals(base["MVT_ID_mvt"]) and np.isfinite(y).all() and y.min() >= 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out)
    rep = {"in": args.inp, "out": args.out, "parts": args.parts, "b_mode": args.b_mode, "rows_changed": ch.height, "A_rows": int(in_a(rows).sum()), "B_rows": int(in_b(rows).sum()),
           "sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest()}
    Path(args.out).with_suffix(".lirf_mech.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
