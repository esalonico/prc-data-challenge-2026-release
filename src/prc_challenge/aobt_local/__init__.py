# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""LOCAL_ONLY AOBT/LOBT research block (quarantined from ranking files).

``AOBT_3_flt`` and ``LOBT_flt`` stay out of every ranking artifact until the
owner clears them. Everything in this package writes ``source_mode: aobt_local``
into its manifests so downstream code can refuse it.
"""

from prc_challenge.aobt_local.features import (
    AOBT_FEATURE_COLUMNS,
    SOURCE_MODE,
    build_aobt_features,
    load_dep_timestamps,
)

__all__ = ["AOBT_FEATURE_COLUMNS", "SOURCE_MODE", "build_aobt_features", "load_dep_timestamps"]
