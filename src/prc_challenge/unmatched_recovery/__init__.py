# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

"""NM-unmatched recovery: freeze-free experimental path for ``nm_unmatched == 1``.

The incumbent ranking pipeline still freezes unmatched rows onto the parent
through ``apply_unmatched_parent_freeze``. This package is the research path
that removes that policy, compares native missing-value handling, imputation,
pattern models, distillation, mixture, and external linkage, and scores the
full population while retaining every genuine unmatched row.
"""

from prc_challenge.unmatched_recovery.schema import (
    BLOCKS,
    D_MONTHS,
    NM_CATEGORICALS,
    NM_NUMERIC_DERIVED,
    PRIMARY_MONTHS,
    apply_nm_block_mask,
    compose_experimental_unmatched,
    derive_planning_from_latents,
    score_populations,
    splice_unmatched,
    unmatched_observed_features,
)
from prc_challenge.unmatched_recovery.subgroups import (
    assert_design_months,
    attach_identity_columns,
    empirical_bayes_mean,
)
from prc_challenge.unmatched_recovery.source_health import (
    attach_source_state,
    ranking_field_availability_audit,
)

__all__ = [
    "BLOCKS",
    "D_MONTHS",
    "NM_CATEGORICALS",
    "NM_NUMERIC_DERIVED",
    "PRIMARY_MONTHS",
    "apply_nm_block_mask",
    "assert_design_months",
    "attach_identity_columns",
    "attach_source_state",
    "compose_experimental_unmatched",
    "derive_planning_from_latents",
    "empirical_bayes_mean",
    "ranking_field_availability_audit",
    "score_populations",
    "splice_unmatched",
    "unmatched_observed_features",
]
