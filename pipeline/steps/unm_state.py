# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""The two small state files the unmatched serve path (scripts/compose/unm_serve.py) reads, rebuilt from the stored models.

* ``Unmatched-noncopy-20260925/ranking/frames/expert_features.json``: the non-copy expert's categorical and numeric
  feature lists, in model order (``expert_noncopy_large_m1_10_octfix.cbm``);
* ``Unmatched-sensor-20260926/state.json``: the U-SENS-G sensor's ADS-B columns (``sens_m1_10.cbm`` numeric features
  after the U6 columns).

Both were written by the original training runs and are not part of the model hand-off; they follow from the model
files. Run from the workspace root (PYTHONPATH=src).
"""
from __future__ import annotations

import json
from pathlib import Path

from catboost import CatBoostRegressor

from prc_challenge.unmatched_recovery.sensor import SENS_CATS, SENS_NUMS_U6

E = Path("artifacts/experiments")
NOTE = "rebuilt by pipeline/steps/unm_state.py from the stored model's feature names"


def split(path: Path) -> tuple[list[str], list[str]]:
    m = CatBoostRegressor()
    m.load_model(str(path))
    names, cat_idx = m.feature_names_, set(m.get_cat_feature_indices())
    return [n for i, n in enumerate(names) if i in cat_idx], [n for i, n in enumerate(names) if i not in cat_idx]


def main() -> None:
    ec, en = split(E / "Unmatched-polish-20260925/ranking/models/expert_noncopy_large_m1_10_octfix.cbm")
    sc, sn = split(E / "Unmatched-sensor-20260926/ranking/models/sens_m1_10.cbm")
    if sc != list(SENS_CATS) or sn[: len(SENS_NUMS_U6)] != list(SENS_NUMS_U6):
        raise SystemExit("sensor model features do not start with the U6 columns")
    q = E / "Unmatched-noncopy-20260925/ranking/frames/expert_features.json"
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(json.dumps({"note": NOTE, "cats": ec, "nums": en}, indent=1) + "\n")
    s = E / "Unmatched-sensor-20260926/state.json"
    s.write_text(json.dumps({"note": NOTE, "phases": {"prepare": {"adsb_features": sn[len(SENS_NUMS_U6):]}}}, indent=1) + "\n")
    print(f"expert features {len(ec)} + {len(en)}; sensor ADS-B columns {len(sn) - len(SENS_NUMS_U6)}")


if __name__ == "__main__":
    main()
