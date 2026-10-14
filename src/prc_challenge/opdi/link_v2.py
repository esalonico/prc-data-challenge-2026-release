# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""OPDI linkage v2: L0 candidate generation, L1 takeoff-event confirmation, L2 surface coverage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Literal

import polars as pl

from prc_challenge.io import CHALLENGE_AIRPORTS
from prc_challenge.opdi.events import DEPARTURE_EVENT_TYPES, load_flight_events, takeoff_events_by_flight
from prc_challenge.opdi.link_tables import (
    airport_delta_quantile_table,
    airport_tier_confirmation_table,
    l2_multiplicity_table,
)
from prc_challenge.opdi.schema import normalize_callsign, standardize_flight_list

# Frozen 2025 identity thresholds — do not tune on taxi labels.
L1A_MAX_ABS_DELTA_SEC = 60
L1B_MAX_ABS_DELTA_SEC = 180
L1_AMBIGUITY_MARGIN_SEC = 120
L2_DEPARTURE_WINDOW_SEC = 7200

L1_RESULT_SCHEMA: dict[str, pl.DataType] = {
    "MVT_ID_mvt": pl.Float64,
    "adep": pl.Utf8,
    "ades": pl.Utf8,
    "aircraft_type": pl.Utf8,
    "callsign_norm": pl.Utf8,
    "callsign_source": pl.Utf8,
    "takeoff_epoch": pl.Int64,
    "l0_candidate_count": pl.Int64,
    "l0_with_takeoff_event_count": pl.Int64,
    "l1_confidence": pl.Utf8,
    "selected_opdi_id": pl.UInt64,
    "event_takeoff_delta_sec": pl.Int64,
    "second_best_abs_delta_sec": pl.Int64,
    "delta_margin_sec": pl.Int64,
    "metadata_conflict": pl.Boolean,
    "ades_match": pl.Boolean,
    "type_match": pl.Boolean,
}
L1Confidence = Literal["L1A", "L1B", "ambiguous", "unconfirmed"]

FORBIDDEN_MATCH_COLUMNS = frozenset(
    {
        "TAXITIME_SEC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "block_time",
        "taxi_label",
        "residual",
        "t31_pred",
    }
)


@dataclass(frozen=True)
class LinkV2Config:
    l1a_max_abs_delta_sec: int = L1A_MAX_ABS_DELTA_SEC
    l1b_max_abs_delta_sec: int = L1B_MAX_ABS_DELTA_SEC
    ambiguity_margin_sec: int = L1_AMBIGUITY_MARGIN_SEC
    l2_departure_window_sec: int = L2_DEPARTURE_WINDOW_SEC


def _assert_no_forbidden_columns(columns: Iterable[str]) -> None:
    bad = FORBIDDEN_MATCH_COLUMNS.intersection(set(columns))
    if bad:
        raise ValueError(f"matching frame must not contain label/residual columns: {sorted(bad)}")


def _challenge_callsign_columns() -> list[pl.Expr]:
    atc = pl.col("CALLSIGN_flt").cast(pl.Utf8, strict=False).map_elements(
        normalize_callsign, return_dtype=pl.Utf8
    )
    iata = pl.col("FLIGHT_mvt").cast(pl.Utf8, strict=False).map_elements(
        normalize_callsign, return_dtype=pl.Utf8
    )
    return [
        pl.coalesce([atc, iata]).alias("callsign_norm"),
        pl.when(atc.is_not_null())
        .then(pl.lit("atc"))
        .when(iata.is_not_null())
        .then(pl.lit("iata_fallback"))
        .otherwise(None)
        .alias("callsign_source"),
    ]


def load_challenge_departures_for_linkage(
    months: tuple[int, ...],
    *,
    raw_dir: str | Path = "data/raw",
    date_start: date | None = None,
    date_end: date | None = None,
) -> pl.DataFrame:
    """Load DEP rows for linkage without taxi/block labels."""
    month_to_path = {
        1: "training_2025-01-01_2025-02-01.parquet",
        7: "training_2025-07-01_2025-08-01.parquet",
    }
    root = Path(raw_dir)
    frames = []
    for m in months:
        rel = month_to_path.get(m)
        if rel is None:
            raise ValueError(f"no cached raw path for month {m}")
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(
            pl.scan_parquet(path)
            .filter(pl.col("PHASE_mvt") == "DEP")
            .with_columns(pl.lit(m).alias("month"))
            .collect()
        )
    dep = pl.concat(frames, how="vertical")
    out = dep.with_columns(
        [
            pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("takeoff_epoch"),
            pl.col("MVT_TIME_UTC_mvt").alias("takeoff_ts"),
            pl.col("MVT_TIME_UTC_mvt").dt.date().alias("flight_date"),
            pl.col("ADEP_mvt").cast(pl.Utf8, strict=False).str.to_uppercase().alias("adep"),
            pl.col("ADES_mvt").cast(pl.Utf8, strict=False).str.to_uppercase().alias("ades"),
            pl.col("AIRCRAFT_TYPE_mvt")
            .cast(pl.Utf8, strict=False)
            .str.to_uppercase()
            .alias("aircraft_type"),
            *_challenge_callsign_columns(),
        ]
    ).filter(pl.col("callsign_norm").is_not_null() & pl.col("adep").is_in(list(CHALLENGE_AIRPORTS)))
    if date_start is not None:
        out = out.filter(pl.col("flight_date") >= date_start)
    if date_end is not None:
        out = out.filter(pl.col("flight_date") <= date_end)
    out = out.select(
        [
            "MVT_ID_mvt",
            "month",
            "flight_date",
            "takeoff_ts",
            "takeoff_epoch",
            "adep",
            "ades",
            "aircraft_type",
            "callsign_norm",
            "callsign_source",
        ]
    )
    _assert_no_forbidden_columns(out.columns)
    return out


def load_opdi_flight_list(paths: dict[int, str | Path]) -> pl.DataFrame:
    return pl.concat(
        [standardize_flight_list(path).collect() for path in paths.values()],
        how="vertical",
    )


def build_l0_candidates(challenge: pl.DataFrame, opdi: pl.DataFrame) -> pl.DataFrame:
    """L0: same normalized callsign, ADEP, UTC takeoff date / OPDI DOF."""
    _assert_no_forbidden_columns(challenge.columns)
    op = opdi.select(
        [
            "opdi_id",
            "callsign_norm",
            "flight_date",
            "adep",
            "ades",
            "aircraft_type",
            "first_seen_epoch",
            "last_seen_epoch",
        ]
    )
    joined = challenge.with_columns(pl.col("flight_date").cast(pl.Utf8).alias("flight_date_key")).join(
        op.with_columns(pl.col("flight_date").cast(pl.Utf8).alias("flight_date_key")),
        on=["callsign_norm", "flight_date_key", "adep"],
        how="left",
        suffix="_opdi",
    )
    return joined.filter(pl.col("opdi_id").is_not_null()).with_columns(
        [
            (pl.col("ades") == pl.col("ades_opdi")).fill_null(False).alias("ades_match"),
            (pl.col("aircraft_type") == pl.col("aircraft_type_opdi")).fill_null(False).alias("type_match"),
            pl.col("ades").is_not_null()
            & pl.col("ades_opdi").is_not_null()
            & (pl.col("ades") != pl.col("ades_opdi"))
            .alias("ades_conflict"),
            pl.col("aircraft_type").is_not_null()
            & pl.col("aircraft_type_opdi").is_not_null()
            & (pl.col("aircraft_type") != pl.col("aircraft_type_opdi"))
            .alias("type_conflict"),
        ]
    )


def _metadata_conflict(row: dict[str, Any]) -> bool:
    return bool(row.get("ades_conflict")) and bool(row.get("type_conflict"))


def _downgrade_confidence(conf: L1Confidence) -> L1Confidence:
    if conf == "L1A":
        return "L1B"
    if conf == "L1B":
        return "unconfirmed"
    return conf


def classify_l1_for_group(
    rows: list[dict[str, Any]],
    *,
    cfg: LinkV2Config,
) -> dict[str, Any]:
    """Resolve L1 for one challenge MVT from candidate rows with takeoff deltas."""
    eligible = [
        r
        for r in rows
        if r.get("takeoff_event_epoch") is not None
        and abs(r["event_takeoff_delta_sec"]) <= cfg.l1b_max_abs_delta_sec
    ]
    if not eligible:
        return {
            "l1_confidence": "unconfirmed",
            "selected_opdi_id": None,
            "event_takeoff_delta_sec": None,
            "second_best_abs_delta_sec": None,
            "delta_margin_sec": None,
            "metadata_conflict": False,
        }
    eligible.sort(key=lambda r: abs(r["event_takeoff_delta_sec"]))
    best = eligible[0]
    second = eligible[1] if len(eligible) > 1 else None
    best_abs = abs(best["event_takeoff_delta_sec"])
    second_abs = abs(second["event_takeoff_delta_sec"]) if second is not None else None
    margin = (second_abs - best_abs) if second_abs is not None else None
    ambiguous = second is not None and second_abs is not None and margin is not None and margin < cfg.ambiguity_margin_sec
    if ambiguous:
        conf: L1Confidence = "ambiguous"
        selected = None
        delta = None
        second_best = second_abs
        meta_conflict = False
    else:
        if best_abs <= cfg.l1a_max_abs_delta_sec:
            conf = "L1A"
        else:
            conf = "L1B"
        selected = best["opdi_id"]
        delta = best["event_takeoff_delta_sec"]
        second_best = second_abs
        if _metadata_conflict(best):
            conf = _downgrade_confidence(conf)
            if conf == "unconfirmed":
                selected = None
                delta = None
        meta_conflict = _metadata_conflict(best)
    return {
        "l1_confidence": conf,
        "selected_opdi_id": selected,
        "event_takeoff_delta_sec": delta,
        "second_best_abs_delta_sec": second_best,
        "delta_margin_sec": margin,
        "metadata_conflict": meta_conflict if not ambiguous else False,
        "ades_match": best.get("ades_match") if selected is not None else None,
        "type_match": best.get("type_match") if selected is not None else None,
        "callsign_source": best.get("callsign_source") if selected is not None else None,
    }


def confirm_l1_matches(
    l0: pl.DataFrame,
    takeoffs: pl.DataFrame,
    *,
    cfg: LinkV2Config | None = None,
) -> pl.DataFrame:
    """Attach L1 confidence by comparing OPDI take-off events to challenge takeoff."""
    cfg = cfg or LinkV2Config()
    if l0.is_empty():
        return pl.DataFrame(schema=L1_RESULT_SCHEMA)
    enriched = l0.join(takeoffs, on="opdi_id", how="left").with_columns(
        (pl.col("takeoff_event_epoch") - pl.col("takeoff_epoch")).alias("event_takeoff_delta_sec")
    )
    groups: list[dict[str, Any]] = []
    for mvt_id in enriched["MVT_ID_mvt"].unique(maintain_order=True).to_list():
        group = enriched.filter(pl.col("MVT_ID_mvt") == mvt_id)
        rows = group.to_dicts()
        resolved = classify_l1_for_group(rows, cfg=cfg)
        base = rows[0]
        groups.append(
            {
                "MVT_ID_mvt": mvt_id,
                "adep": base["adep"],
                "ades": base["ades"],
                "aircraft_type": base["aircraft_type"],
                "callsign_norm": base["callsign_norm"],
                "callsign_source": base.get("callsign_source"),
                "takeoff_epoch": base["takeoff_epoch"],
                "l0_candidate_count": len(rows),
                "l0_with_takeoff_event_count": sum(
                    1 for r in rows if r.get("takeoff_event_epoch") is not None
                ),
                **{k: v for k, v in resolved.items() if k not in {"l0_candidate_count", "l0_with_takeoff_event_count"}},
            }
        )
    if not groups:
        return pl.DataFrame(schema=L1_RESULT_SCHEMA)
    return pl.DataFrame(groups, schema=L1_RESULT_SCHEMA)


def select_parking_exit_event(
    events: pl.DataFrame,
    *,
    takeoff_epoch: int,
    cfg: LinkV2Config | None = None,
) -> dict[str, Any]:
    """Pick latest valid exit-parking-position before departure chain (no label use)."""
    cfg = cfg or LinkV2Config()
    empty = {
        "selected_parking_exit_epoch": None,
        "selected_parking_exit_info": None,
        "parking_exit_count": 0,
        "parking_selection_reason": "no_events_in_window",
    }
    if events.is_empty() or "event_epoch" not in events.columns:
        return empty
    window_start = takeoff_epoch - cfg.l2_departure_window_sec
    subset = events.filter(
        (pl.col("event_epoch") >= window_start) & (pl.col("event_epoch") <= takeoff_epoch)
    ).sort("event_epoch")
    if subset.is_empty():
        return {
            "selected_parking_exit_epoch": None,
            "selected_parking_exit_info": None,
            "parking_exit_count": 0,
            "parking_selection_reason": "no_events_in_window",
        }
    parking = subset.filter(pl.col("event_type") == "exit-parking_position")
    if parking.is_empty():
        return {
            "selected_parking_exit_epoch": None,
            "selected_parking_exit_info": None,
            "parking_exit_count": 0,
            "parking_selection_reason": "no_parking_exit",
        }
    chain_types = {"entry-taxiway", "exit-taxiway", "entry-runway", "take-off"}
    chain = subset.filter(pl.col("event_type").is_in(list(chain_types)))
    if chain.is_empty():
        return {
            "selected_parking_exit_epoch": None,
            "selected_parking_exit_info": None,
            "parking_exit_count": parking.height,
            "parking_selection_reason": "no_departure_chain",
        }
    first_chain_epoch = int(chain["event_epoch"].min())
    valid = parking.filter(pl.col("event_epoch") <= first_chain_epoch)
    if valid.is_empty():
        return {
            "selected_parking_exit_epoch": None,
            "selected_parking_exit_info": None,
            "parking_exit_count": parking.height,
            "parking_selection_reason": "no_parking_before_chain",
        }
    chosen = valid.sort("event_epoch").tail(1)
    return {
        "selected_parking_exit_epoch": int(chosen["event_epoch"][0]),
        "selected_parking_exit_info": chosen["info"][0],
        "parking_exit_count": parking.height,
        "parking_selection_reason": "latest_before_departure_chain",
    }


def inspect_l2_surface_events(
    l1_confirmed: pl.DataFrame,
    events: pl.DataFrame,
    *,
    cfg: LinkV2Config | None = None,
) -> pl.DataFrame:
    """Summarize surface-event availability for L1-confirmed flights."""
    cfg = cfg or LinkV2Config()
    confirmed = l1_confirmed.filter(pl.col("l1_confidence").is_in(["L1A", "L1B"]))
    if confirmed.is_empty():
        return confirmed
    rows: list[dict[str, Any]] = []
    events_by_flight: dict[int, pl.DataFrame] = {}
    for keys, grp in events.group_by("opdi_id", maintain_order=True):
        opdi_id = keys[0] if isinstance(keys, tuple) else keys
        events_by_flight[int(opdi_id)] = grp.sort("event_epoch")
    for row in confirmed.iter_rows(named=True):
        opdi_id = int(row["selected_opdi_id"])
        takeoff_epoch = int(row["takeoff_epoch"])
        flight_events = events_by_flight.get(opdi_id, pl.DataFrame())
        parking = select_parking_exit_event(flight_events, takeoff_epoch=takeoff_epoch, cfg=cfg)
        window_start = takeoff_epoch - cfg.l2_departure_window_sec
        if flight_events.is_empty() or "event_epoch" not in flight_events.columns:
            window = flight_events
            has_entry_runway = False
            has_taxiway = False
            has_takeoff = False
            ordered = False
        else:
            window = flight_events.filter(
                (pl.col("event_epoch") >= window_start) & (pl.col("event_epoch") <= takeoff_epoch)
            )
            has_entry_runway = window.filter(pl.col("event_type") == "entry-runway").height > 0
            has_taxiway = (
                window.filter(pl.col("event_type").is_in(["entry-taxiway", "exit-taxiway"])).height > 0
            )
            has_takeoff = window.filter(pl.col("event_type") == "take-off").height > 0
            ordered = _physically_ordered_chain(window)
        rows.append(
            {
                **row,
                **parking,
                "has_exit_parking_position": parking["selected_parking_exit_epoch"] is not None,
                "has_entry_runway": has_entry_runway,
                "has_taxiway_events": has_taxiway,
                "has_takeoff_event": has_takeoff,
                "physically_ordered_chain": ordered,
            }
        )
    return pl.DataFrame(
        rows,
        schema_overrides={
            "selected_parking_exit_epoch": pl.Int64,
            "parking_exit_count": pl.Int64,
        },
        infer_schema_length=None,
    )


def _physically_ordered_chain(window: pl.DataFrame) -> bool:
    parking = window.filter(pl.col("event_type") == "exit-parking_position")
    taxi = window.filter(pl.col("event_type").is_in(["entry-taxiway", "exit-taxiway"]))
    runway = window.filter(pl.col("event_type") == "entry-runway")
    takeoff = window.filter(pl.col("event_type") == "take-off")
    if parking.is_empty() or takeoff.is_empty():
        return False
    p = int(parking.sort("event_epoch").tail(1)["event_epoch"][0])
    t0 = int(takeoff.sort("event_epoch").tail(1)["event_epoch"][0])
    if p > t0:
        return False
    if not taxi.is_empty():
        t_min = int(taxi["event_epoch"].min())
        t_max = int(taxi["event_epoch"].max())
        if t_min < p or t_max > t0:
            return False
    if not runway.is_empty():
        r = int(runway.sort("event_epoch").tail(1)["event_epoch"][0])
        if r < p or r > t0:
            return False
    return True


def l0_candidate_counts(challenge: pl.DataFrame, l0: pl.DataFrame) -> pl.DataFrame:
    counts = (
        l0.group_by("MVT_ID_mvt")
        .agg(pl.len().alias("l0_candidate_count"))
        if not l0.is_empty()
        else pl.DataFrame(schema={"MVT_ID_mvt": pl.Float64, "l0_candidate_count": pl.Int64})
    )
    return challenge.select("MVT_ID_mvt").join(counts, on="MVT_ID_mvt", how="left").with_columns(
        pl.col("l0_candidate_count").fill_null(0)
    )


def linkage_summary(
    challenge: pl.DataFrame,
    l0_counts: pl.DataFrame,
    l1: pl.DataFrame,
    l2: pl.DataFrame | None = None,
    *,
    old_tier: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Aggregate D1a linkage metrics without taxi labels."""
    n = challenge.height
    l0_cov = int((l0_counts["l0_candidate_count"] > 0).sum()) / n if n else 0.0
    l1a_cov = int((l1["l1_confidence"] == "L1A").sum()) / n if n else 0.0
    l1b_cov = int((l1["l1_confidence"] == "L1B").sum()) / n if n else 0.0
    confirmed_cov = int(l1["l1_confidence"].is_in(["L1A", "L1B"]).sum()) / n if n else 0.0
    amb_cov = int((l1["l1_confidence"] == "ambiguous").sum()) / n if n else 0.0
    unconf_cov = int((l1["l1_confidence"] == "unconfirmed").sum()) / n if n else 0.0
    fallback_rate = (
        float((challenge["callsign_source"] == "iata_fallback").sum()) / n if n else 0.0
    )
    deltas = l1.filter(pl.col("event_takeoff_delta_sec").is_not_null())["event_takeoff_delta_sec"]
    quantiles = {}
    if deltas.len():
        for q in (0.01, 0.05, 0.5, 0.95, 0.99):
            quantiles[f"p{int(q * 100):02d}"] = float(deltas.quantile(q))
    cand_dist = (
        l0_counts.group_by("l0_candidate_count")
        .len()
        .sort("l0_candidate_count")
        .to_dicts()
        if l0_counts.height
        else []
    )
    takeoff_avail = (
        float(l1["l0_with_takeoff_event_count"].sum()) / float(l1["l0_candidate_count"].sum())
        if l1.height and float(l1["l0_candidate_count"].sum()) > 0
        else 0.0
    )
    ades_agree = l1.filter(pl.col("l1_confidence").is_in(["L1A", "L1B"]) & pl.col("ades_match") == True)
    type_agree = l1.filter(pl.col("l1_confidence").is_in(["L1A", "L1B"]) & pl.col("type_match") == True)
    confirmed_n = int(l1["l1_confidence"].is_in(["L1A", "L1B"]).sum())
    summary: dict[str, Any] = {
        "n_challenge_rows": n,
        "l0_candidate_coverage": l0_cov,
        "l0_candidate_count_distribution": cand_dist,
        "takeoff_event_availability_among_candidates": takeoff_avail,
        "l1a_coverage": l1a_cov,
        "l1b_coverage": l1b_cov,
        "confirmed_coverage": confirmed_cov,
        "ambiguous_rate": amb_cov,
        "unconfirmed_rate": unconf_cov,
        "event_takeoff_delta_quantiles": quantiles,
        "ades_agreement_rate_confirmed": float(ades_agree.height) / confirmed_n if confirmed_n else 0.0,
        "type_agreement_rate_confirmed": float(type_agree.height) / confirmed_n if confirmed_n else 0.0,
        "callsign_fallback_rate": fallback_rate,
    }
    if l2 is not None and l2.height:
        summary["l2_exit_parking_coverage"] = float(l2["has_exit_parking_position"].sum()) / l2.height
        summary["l2_entry_runway_coverage"] = float(l2["has_entry_runway"].sum()) / l2.height
        summary["l2_taxiway_coverage"] = float(l2["has_taxiway_events"].sum()) / l2.height
        summary["l2_ordered_chain_coverage"] = float(l2["physically_ordered_chain"].sum()) / l2.height
    if old_tier is not None and old_tier.height:
        merged = old_tier.join(
            l1.select(["MVT_ID_mvt", "l1_confidence"]),
            on="MVT_ID_mvt",
            how="left",
        )
        for tier in ("A", "B"):
            tier_rows = merged.filter(pl.col("confidence_tier") == tier)
            if tier_rows.height:
                confirmed = tier_rows.filter(pl.col("l1_confidence").is_in(["L1A", "L1B"]))
                summary[f"old_tier_{tier.lower()}_l1_confirmation_rate"] = (
                    float(confirmed.height) / tier_rows.height
                )
                summary[f"old_tier_{tier.lower()}_n"] = tier_rows.height
    summary["by_airport"] = airport_linkage_breakdown(challenge, l1)
    summary["by_airport_tier_confirmation"] = (
        airport_tier_confirmation_table(challenge, l1, old_tier) if old_tier is not None else []
    )
    summary["by_airport_delta_quantiles"] = airport_delta_quantile_table(l1)
    summary["l2_multiplicity"] = l2_multiplicity_table(l2) if l2 is not None else {}
    summary["go_park"] = linkage_go_park_decision(summary)
    return summary


def airport_linkage_breakdown(challenge: pl.DataFrame, l1: pl.DataFrame) -> list[dict[str, Any]]:
    merged = challenge.select(["MVT_ID_mvt", "adep"]).join(l1, on="MVT_ID_mvt", how="left")
    rows = []
    for airport, grp in merged.group_by("adep"):
        ap = airport[0] if isinstance(airport, tuple) else airport
        n = grp.height
        rows.append(
            {
                "airport": ap,
                "n": n,
                "l0_candidate_coverage": float((grp["l0_candidate_count"].fill_null(0) > 0).sum()) / n,
                "l1a_coverage": float((grp["l1_confidence"] == "L1A").sum()) / n,
                "l1b_coverage": float((grp["l1_confidence"] == "L1B").sum()) / n,
                "confirmed_coverage": float(grp["l1_confidence"].is_in(["L1A", "L1B"]).sum()) / n,
                "ambiguous_rate": float((grp["l1_confidence"] == "ambiguous").sum()) / n,
                "unconfirmed_rate": float((grp["l1_confidence"] == "unconfirmed").sum()) / n,
            }
        )
    return sorted(rows, key=lambda r: r["n"], reverse=True)


def linkage_go_park_decision(summary: dict[str, Any]) -> str:
    confirmed = summary.get("confirmed_coverage", 0.0)
    ambiguous = summary.get("ambiguous_rate", 0.0)
    if confirmed >= 0.50 and ambiguous <= 0.05:
        return "strong_go"
    if confirmed >= 0.25 or any(
        r.get("confirmed_coverage", 0) >= 0.25
        for r in summary.get("by_airport", [])
        if r.get("airport") in {"LFPG", "LIRF", "EDDF", "EDDM", "EGLL", "LEBL"}
    ):
        return "moderate_go"
    if confirmed < 0.10:
        return "park"
    return "moderate_go"


def run_linkage_pipeline(
    challenge: pl.DataFrame,
    opdi_paths: dict[int, str | Path],
    event_paths: list[Path | str],
    *,
    cfg: LinkV2Config | None = None,
) -> dict[str, pl.DataFrame]:
    """End-to-end L0 → L1 → L2 linkage for one challenge sample."""
    cfg = cfg or LinkV2Config()
    opdi = load_opdi_flight_list(opdi_paths)
    l0 = build_l0_candidates(challenge, opdi)
    candidate_ids = set(l0["opdi_id"].unique().to_list()) if not l0.is_empty() else set()
    events = load_flight_events(
        event_paths,
        flight_ids=candidate_ids,
        event_types=DEPARTURE_EVENT_TYPES,
    )
    takeoffs = takeoff_events_by_flight(events)
    l1 = confirm_l1_matches(l0, takeoffs, cfg=cfg)
    l0_counts = l0_candidate_counts(challenge, l0)
    l1 = (
        challenge.select(["MVT_ID_mvt", "adep", "ades", "aircraft_type", "callsign_norm", "callsign_source", "takeoff_epoch"])
        .join(l1, on="MVT_ID_mvt", how="left", suffix="_l1")
        .join(l0_counts, on="MVT_ID_mvt", how="left")
        .with_columns(pl.col("l1_confidence").fill_null("unconfirmed"))
    )
    confirmed = l1.filter(pl.col("l1_confidence").is_in(["L1A", "L1B"]))
    confirmed_events = events.filter(pl.col("opdi_id").is_in(confirmed["selected_opdi_id"].drop_nulls()))
    l2 = inspect_l2_surface_events(confirmed, confirmed_events, cfg=cfg)
    return {
        "challenge": challenge,
        "l0": l0,
        "l0_counts": l0_counts,
        "events": events,
        "l1": l1,
        "l2": l2,
    }
