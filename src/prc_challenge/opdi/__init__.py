# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""OPDI v0.0.2 flight-list matching, event linkage, and feasibility diagnostics."""

from prc_challenge.opdi.events import D1A_SAMPLE_CHUNKS, DEPARTURE_EVENT_TYPES, load_flight_events
from prc_challenge.opdi.link_v2 import LinkV2Config, run_linkage_pipeline
from prc_challenge.opdi.match import MatchConfig, match_departures_to_opdi
from prc_challenge.opdi.schema import inspect_flight_list_schema, normalize_callsign

__all__ = [
    "D1A_SAMPLE_CHUNKS",
    "DEPARTURE_EVENT_TYPES",
    "LinkV2Config",
    "MatchConfig",
    "inspect_flight_list_schema",
    "load_flight_events",
    "match_departures_to_opdi",
    "normalize_callsign",
    "run_linkage_pipeline",
]
