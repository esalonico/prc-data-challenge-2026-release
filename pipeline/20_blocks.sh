#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Step 20 (SERVE_WS): label-free caches and the non-ADS-B feature blocks for the four serve months.
#   * S4 base / native / events caches for 2026-01, -02, -06, -07 (month-isolated);
#   * M1 stand sequence, M2 neighbour context: full rebuild (the builder checks 2025 months 1-10 against the shipped
#     months_1_10 reference; months 11-12 are deterministic per month file);
#   * M4-P planned taxi: ranking rows rebuilt, spliced into the shipped block (2025 rows kept byte for byte);
#   * AOBT / LOBT clocks: full rebuild (per input file, so 2025 rows are unchanged);
#   * the two state files of the unmatched serve path (pipeline/steps/unm_state.py).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
in_serve
M=$A/Matched-features-20260925
log "caches"
"$PY" scripts/legs/s4_allyear.py --stage build-caches --periods 2026-01,2026-02,2026-06,2026-07 --force
log "M1 / M2"
for b in m1 m2; do "$PY" scripts/features/build_matched_context_blocks.py --block $b --source allyear_plus_ranking --variant v1s; done
log "M4-P"
[ -f $M/m4_p_v1/s4_all.shipped.parquet ] || cp $M/m4_p_v1/s4_all.parquet $M/m4_p_v1/s4_all.shipped.parquet
"$PY" scripts/features/build_m4_blocks.py --block p --variant v1 --source ranking
"$PY" "$STEPS/splice_block.py" --shipped $M/m4_p_v1/s4_all.shipped.parquet --ranking $M/m4_p_v1/ranking.parquet --out $M/m4_p_v1/s4_all.parquet \
  --old-raw data/raw/ranking_public.parquet
log "AOBT"
"$PY" scripts/features/build_aobt_local_features.py --scope all --promote-owner-ruling "D-017"
log "unmatched state files"
"$PY" "$STEPS/unm_state.py"
log "serve blocks done"
