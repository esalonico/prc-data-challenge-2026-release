# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""Ranking-aligned operational-information features.

The package keeps target-independent identity sidecars separate from learned
arrival-state normalisation and from local-only AOBT sensitivity work.
"""

from prc_challenge.operational_v2.native_state import (
    NATIVE_STATE_FEATURES,
    attach_cross_fitted_arrival_state,
    build_native_completed_arrival_sidecar,
    load_completed_arrivals,
)
from prc_challenge.operational_v2.latent_arrival_state import (
    FILTERED_STATE_FEATURES,
    KALMAN_STATE_FEATURES,
    FilterConfig,
    KalmanConfig,
    add_interval_projection,
    attach_arrival_measurements,
    build_latent_state_sidecar,
    fit_arrival_reference,
)
from prc_challenge.operational_v2.rotation_features import (
    ROTATION_CATEGORICAL_FEATURES,
    ROTATION_NUMERIC_FEATURES,
    build_rotation_month,
)
from prc_challenge.operational_v2.surface_reconstruction import (
    DurationReference,
    ServiceReference,
    apply_service_reference,
    assert_learned_cache_identity,
    build_event_table,
    build_expected_occupancy_features,
    build_retrospective_runway_features,
    build_stand_turnover_features,
    build_virtual_workload_features,
    fit_duration_reference,
    fit_service_reference,
    learned_feature_cache_identity,
    load_learned_cache_or_reject,
    reject_forbidden_departure_columns,
    stand_interval_diagnostics,
    write_learned_cache_identity,
)

__all__ = [
    "NATIVE_STATE_FEATURES",
    "FILTERED_STATE_FEATURES",
    "KALMAN_STATE_FEATURES",
    "ROTATION_CATEGORICAL_FEATURES",
    "ROTATION_NUMERIC_FEATURES",
    "attach_cross_fitted_arrival_state",
    "add_interval_projection",
    "attach_arrival_measurements",
    "build_latent_state_sidecar",
    "build_native_completed_arrival_sidecar",
    "build_rotation_month",
    "load_completed_arrivals",
    "fit_arrival_reference",
    "FilterConfig",
    "KalmanConfig",
    "DurationReference",
    "ServiceReference",
    "apply_service_reference",
    "build_event_table",
    "build_expected_occupancy_features",
    "build_retrospective_runway_features",
    "build_stand_turnover_features",
    "build_virtual_workload_features",
    "fit_duration_reference",
    "fit_service_reference",
    "reject_forbidden_departure_columns",
    "stand_interval_diagnostics",
    "learned_feature_cache_identity",
    "assert_learned_cache_identity",
    "load_learned_cache_or_reject",
    "write_learned_cache_identity",
]
