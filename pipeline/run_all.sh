#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# The final file from this checkout in one command (inputs and run time: docs/REPRODUCTION.md).
#   pipeline/run_all.sh                  # -> $SERVE_WS/artifacts/experiments/Final-serve4m/F2FP.parquet (the final)
#   REF=<file> pipeline/run_all.sh       # also compare the step-70 file with a reference
# Intermediate files in the same folder: F1 (step 70), F2F (step 75). Other variants: docs/REPRODUCTION.md.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export NAME=${NAME:-F1} LGB_SET=${LGB_SET-01}
LOGS="${LOGS:-$REPO/artifacts/experiments/Final-serve4m-logs}"
mkdir -p "$LOGS"
step() { local name=$1; shift; log "start $name"; "$@" > "$LOGS/$name.log" 2>&1 || { log "$name FAILED (log: $LOGS/$name.log)"; tail -20 "$LOGS/$name.log"; exit 1; }; log "done $name"; }
step 00_workspaces "$PIPE/00_workspaces.sh"
# 10 (REGRESS_WS) and 20 (SERVE_WS) are independent
log "start 10_regress + 20_blocks (parallel)"
"$PIPE/10_regress.sh" > "$LOGS/10_regress.log" 2>&1 & p10=$!
"$PIPE/20_blocks.sh" > "$LOGS/20_blocks.log" 2>&1 & p20=$!
wait $p20 || { log "20_blocks FAILED"; tail -20 "$LOGS/20_blocks.log"; exit 1; }
wait $p10 || { log "10_regress FAILED"; tail -20 "$LOGS/10_regress.log"; exit 1; }
log "done 10 + 20"
step 30_adsb "$PIPE/30_adsb.sh"
step 35_trajectories "$PIPE/35_trajectories.sh"
step 40_assemble "$PIPE/40_assemble.sh"
step 50_checks "$PIPE/50_checks.sh"
step 60_legs_gbm "$PIPE/60_legs.sh" gbm
step 60_legs_mlp "$PIPE/60_legs.sh" mlp
step 60_legs_nnx "$PIPE/60_legs.sh" nnx
step 60_legs_traj "$PIPE/60_legs.sh" traj
step 70_blend "$PIPE/70_blend.sh"
step 75_lirf_rule env IN="$NAME" OUT=F2F "$PIPE/75_lirf_rule.sh"
log "ALL DONE: $OUTF/F2FP.parquet"
cat "$OUTF/F2FP.validation.json"
