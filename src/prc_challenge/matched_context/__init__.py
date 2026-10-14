# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Label-free context blocks for NM-matched departures (SA-MF campaign, 2026-09-25).

M1 = stand sequence / turnaround context (``stand_sequence``).
M2 = following / neighbour NM planning context (``neighbor_context``).

Every builder consumes the masked movement table from ``movements`` only.  That
table never carries departure BLOCK/TAXITIME or AOBT_3/ARVT_3 of any flight;
arrival in-block times are kept for ARR rows only.  Context is computed inside
one context window (a calendar-month source file, or one calendar month of the
ranking file) at a time, so disjoint windows are never bridged.
"""
