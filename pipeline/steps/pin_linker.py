# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Write a pinned linker_precision.json for build_adsb_features.py.

The shipped all-year prize block (sha 70cfd2e5...) keeps take-off-fallback joins (adsb_join_src = 3) at exactly
EDDF, EDDM, LEBL, LEMD, LIRF and LSZH; EGLL, EHAM, LFPG and LTFM are gated out. The original json (sample days drawn
from the stage1 days present at build time) was not shipped; recomputing it on a different set of days could move an
airport across the 95% gate and change the serve rows, so the gate is pinned to the shipped behaviour.
"""
import json
import sys
from pathlib import Path

KEEP = ("EDDF", "EDDM", "LEBL", "LEMD", "LIRF", "LSZH")
ALL = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"pinned": "from the shipped adsb_block_prize_allyear.parquet (sha 70cfd2e5): airports with "
                                     "adsb_join_src == 3 rows keep the take-off fallback",
                           "min_precision": 0.95,
                           "airports": {ap: {"keep_fallback": ap in KEEP} for ap in ALL}}, indent=1) + "\n")
print("pinned", out)
