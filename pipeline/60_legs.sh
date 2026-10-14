#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Step 60 (SERVE_WS, after steps 35 and 40): the fourteen legs on the four-month matrix, predicted with the stored
# all-year models (nothing is refitted). GBM legs and torch legs run in separate processes: torch deadlocks in libomp
# after CatBoost / LightGBM ran in the same process. Usage: 60_legs.sh [all|gbm|mlp|nnx|traj]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
in_serve
L="$OUTF/legs"
M=$RUNS/serve4m_allyear_iso_v3l
STEP=${1:-all}
G=${LGB_SET:-}   # "" = the first LightGBM legs (F0); "01" = the lr 0.01 legs (F1 and the final)
if [[ $STEP == all || $STEP == gbm ]]; then
  "$PY" scripts/legs/predict_legs.py --train-run $M --eval-run $M --out-dir "$L" --legs cb8k,cbd8k,cbe8k,lgb$G,lgbd$G,lgbe$G,lgbl$G
fi
if [[ $STEP == all || $STEP == mlp ]]; then
  "$PY" scripts/legs/predict_legs.py --train-run $M --eval-run $M --out-dir "$L" --legs nn14,nn15,nn16
fi
if [[ $STEP == all || $STEP == nnx ]]; then
  NNX_THREADS=3 "$PY" scripts/legs/nnx/prep.py ay3liso_serve4m
  "$PY" scripts/legs/nnx/predict_serve.py --split ay3liso_serve4m --names plr_s03_resnet_f3,plr_s03_eobtres_f3,ctx_gru_plr_f2 --out-dir "$L"
fi
if [[ $STEP == all || $STEP == traj ]]; then
  "$PY" scripts/legs/traj/tracks.py split ay3liso_serve4m --matrices $M/train_matrix.parquet $M/eval_matrix.parquet --pool AY
  "$PY" scripts/legs/traj/predict_serve.py --split ay3liso_serve4m --out-dir "$L"
fi
log "legs done ($STEP)"
