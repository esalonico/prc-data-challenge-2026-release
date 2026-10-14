# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Load months 1–10 cache, parent preds, and choose BASE."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from prc_challenge.bundles import e08b_feature_lists, e34_feature_lists, matched_booster_frame
from prc_challenge.campaign.data_check import default_drive_link, locate
from prc_challenge.campaign.runtime import CampaignConfig
from prc_challenge.campaign.scoring import assert_no_forbidden_features, assert_no_locked_months
from prc_challenge.splits import assign_split, views_from_config

CACHE_NAME = "e08b_e34_months_1_10.parquet"
PARENT = {
    "V1": "artifacts/experiments/E33b-matched-splice_V1/val_predictions.parquet",
    "V2": "artifacts/experiments/E33b-matched-splice_V2/val_predictions.parquet",
    "V3": "artifacts/experiments/E33b-matched-splice_V3/val_predictions.parquet",
}
T31 = {
    "V1": "artifacts/experiments/E37-eval-e08b-t31_V1/val_predictions.parquet",
    "V2": "artifacts/experiments/E37-eval-e08b-t31_V2/val_predictions.parquet",
    "V3": "artifacts/experiments/E37-eval-e08b-t31_V3/val_predictions.parquet",
}
T6 = {
    "V1": "artifacts/experiments/E37-eval-e08b-t6_V1/val_predictions.parquet",
    "V2": "artifacts/experiments/E37-eval-e08b-t6_V2/val_predictions.parquet",
    "V3": "artifacts/experiments/E37-eval-e08b-t6_V3/val_predictions.parquet",
}
E34B = {
    "V1": "artifacts/experiments/E34b-bunching-gated_V1/val_predictions.parquet",
    "V2": "artifacts/experiments/E34b-bunching-gated_V2/val_predictions.parquet",
    "V3": "artifacts/experiments/E34b-bunching-gated_V3/val_predictions.parquet",
}


def locate_cache(cfg: CampaignConfig) -> Path:
    found = locate(Path("artifacts/cache") / CACHE_NAME, cfg)
    if found is not None:
        return found
    candidates = [
        cfg.cache_root / CACHE_NAME,
        Path("/content") / CACHE_NAME,
        Path("/content/drive/MyDrive/Colab Data Link") / CACHE_NAME,
        Path("/content/drive/MyDrive/Colab Data Link/artifacts/cache") / CACHE_NAME,
        cfg.repo_root / "artifacts" / "cache" / CACHE_NAME,
    ]
    drive = cfg.drive_link or default_drive_link()
    if drive is not None:
        candidates.extend([drive / CACHE_NAME, drive / "artifacts" / "cache" / CACHE_NAME])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"need {CACHE_NAME} in artifacts/cache or Colab Data Link"
    )


# Split, compose, and sequence-pack columns. Extra-view GPU jobs must not materialize the
# full e08b dump while transformer packs are still mapped.
SKINNY_CACHE_COLS = (
    "MVT_ID_mvt",
    "AIRPORT",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "month",
    "nm_unmatched",
    "type_missing",
    "takeoff_minus_sched_days",
    "takeoff_minus_sched_sec",
    "takeoff_minus_eobt_sec",
    "takeoff_minus_iobt_sec",
    "TAXITIME_SEC_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "local_hour",
    "local_dow",
    "sknt",
    "vsby",
    "n_delayed_dep_ctr_1800s",
    "sec_since_prev_dep",
)


def load_cache(cfg: CampaignConfig) -> pl.DataFrame:
    path = locate_cache(cfg)
    df = pl.read_parquet(path)
    if "month" not in df.columns and "MVT_TIME_UTC_mvt" in df.columns:
        df = df.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
    assert_no_locked_months(df)
    if cfg.max_rows and df.height > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, seed=cfg.seed, shuffle=True)
    return df


def load_cache_columns(cfg: CampaignConfig, extra: list[str] | tuple[str, ...] | None = None) -> pl.DataFrame:
    """Load only split/score/sequence columns plus `extra`. Does not fit models."""
    path = locate_cache(cfg)
    names = set(pl.scan_parquet(path).collect_schema().names())
    want = list(dict.fromkeys([*SKINNY_CACHE_COLS, *(extra or ())]))
    cols = [c for c in want if c in names]
    if "MVT_ID_mvt" not in cols:
        raise RuntimeError(f"{CACHE_NAME} missing MVT_ID_mvt")
    if "month" not in cols and "MVT_TIME_UTC_mvt" in names:
        cols.append("MVT_TIME_UTC_mvt")
    df = pl.scan_parquet(path).select(cols).collect()
    if "month" not in df.columns and "MVT_TIME_UTC_mvt" in df.columns:
        df = df.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
    assert_no_locked_months(df)
    if cfg.max_rows and df.height > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, seed=cfg.seed, shuffle=True)
    return df


def load_selected_columns(cfg: CampaignConfig, cols: list[str] | tuple[str, ...]) -> pl.DataFrame:
    """Load `MVT_ID_mvt` plus exactly `cols`. No skinny bundle, no max_rows sample.

    GPU-04 fusion joins extras onto a compact DEP table; pulling SKINNY_CACHE_COLS
    again would hold two copies of the split keys while sequence packs are live.
    """
    path = locate_cache(cfg)
    names = set(pl.scan_parquet(path).collect_schema().names())
    keep = ["MVT_ID_mvt"]
    keep.extend(c for c in cols if c in names and c != "MVT_ID_mvt")
    if len(keep) == 1:
        return pl.DataFrame({"MVT_ID_mvt": pl.Series([], dtype=pl.String)})
    return pl.scan_parquet(path).select(keep).collect()


def feature_lists(bundle: str) -> tuple[list[str], list[str]]:
    if bundle == "e08b":
        cats, nums = e08b_feature_lists()
    elif bundle == "e34":
        cats, nums = e34_feature_lists()
    else:
        raise ValueError(bundle)
    assert_no_forbidden_features(cats + nums)
    return cats, nums


def load_parent(cfg: CampaignConfig, view: str) -> pl.DataFrame:
    rel = Path(PARENT[view])
    path = locate(rel, cfg)
    if path is None:
        raise FileNotFoundError(f"need parent preds {rel}")
    return pl.read_parquet(path)


def choose_base(cfg: CampaignConfig) -> dict[str, Path]:
    """t31 if V1 and V2 exist; else t6. V3 optional."""
    t31_v1 = locate(T31["V1"], cfg)
    t31_v2 = locate(T31["V2"], cfg)
    mapping = T31 if (t31_v1 is not None and t31_v2 is not None) else T6
    out: dict[str, Path] = {}
    for view, rel in mapping.items():
        found = locate(rel, cfg)
        if found is not None:
            out[view] = found
        else:
            out[view] = cfg.repo_root / rel
    return out


def load_base_preds(cfg: CampaignConfig, view: str) -> pl.DataFrame:
    paths = choose_base(cfg)
    path = paths[view]
    if not path.exists():
        raise FileNotFoundError(f"need BASE preds for {view}: {path}")
    return pl.read_parquet(path)


def split_view(df: pl.DataFrame, view_name: str) -> pl.DataFrame:
    views = views_from_config()
    view = views[view_name]
    if view.locked:
        raise RuntimeError(f"refusing locked view {view_name}")
    return assign_split(df, view)


def matched_train_val(df: pl.DataFrame, view_name: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    assigned = split_view(df, view_name)
    train = matched_booster_frame(assigned.filter(pl.col("split_role") == "train"))
    val = assigned.filter(pl.col("split_role") == "val")
    return train, val
