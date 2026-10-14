#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Step 50 (SERVE_WS, after step 40): label-free checks of the four-month serve matrix.
#   parity screen (2025 month m vs 2026 month m, m = 1, 2, 6, 7; allowlist v2), report only;
#   dither / coverage table against the shipped Jan/Jul ranking matrix (REGRESS_WS step 10).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
in_serve
mkdir -p "$OUTF/checks"
"$PY" scripts/features/parity_screen.py --run-dir $RUNS/serve4m_allyear_iso_v3l --months 1,2,6,7 --allowlist configs/bake/parity_allowlist_v2.json \
  --out "$OUTF/checks/parity_screen_4m.json" --csv "$OUTF/checks/parity_screen_4m.csv" --report-only
"$PY" "$STEPS/check_serve.py" --train $RUNS/serve4m_allyear_iso_v3l/train_matrix.parquet --serve $RUNS/serve4m_allyear_iso_v3l/eval_matrix.parquet \
  --shipped "$REGRESS_WS/$RUNS/regress_allyear_iso_v3l/eval_matrix.parquet" --out "$OUTF/checks" --coverage-col adsb_coverage
log "checks done"
