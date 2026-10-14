#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Step 40 (SERVE_WS, after steps 20 and 30): the four-month matrices.
#   serve4m_allyear_iso_v3l : isocap V3L all-year matrix; its train rows must equal the legs' fit matrix (c67a87cf...)
#   serve4m_t1              : contiguous all-year matrix without v3-lean (T1 p_long / mu_long); train rows c98bb06a...
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
in_serve
B=(--extra-features "$A/Matched-features-20260925/m1_stand_v1s/allyear_plus_ranking.parquet"
   --extra-features "$A/Matched-features-20260925/m2_follow_v1s/allyear_plus_ranking.parquet"
   --extra-features "$A/Matched-features-20260925/m4_p_v1/s4_all.parquet"
   --extra-features "$A/AOBT-local-only-20260925/features/aobt_features_allyear_plus_ranking.parquet"
   --extra-features "$F/prize/adsb_block_prize_allyear.parquet")
log "isocap V3L"
"$PY" scripts/legs/s4_allyear.py --stage fit --train-months 1-12 --eval ranking --tag serve4m_allyear_iso_v3l \
  "${B[@]}" --extra-features "$C/v3/adsb_v3lean_block_allyear.parquet" --temporal-mode isocap_v1 --threads 8 --assemble-only
log "contiguous (T1)"
"$PY" scripts/legs/s4_allyear.py --stage fit --train-months 1-12 --eval ranking --tag serve4m_t1 "${B[@]}" --threads 8 --assemble-only
"$PY" - <<'EOF'
import hashlib
R = "artifacts/experiments/S4-allyear-rebuild-20260925/runs"
for run, want in (("serve4m_allyear_iso_v3l", "c67a87cf"), ("serve4m_t1", "c98bb06a")):
    got = hashlib.sha256(open(f"{R}/{run}/train_matrix.parquet", "rb").read()).hexdigest()
    print(run, "train", got[:16])
    if not got.startswith(want):
        raise SystemExit(f"{run}: train rows differ from the legs' fit matrix ({want}...)")
EOF
log "assembly done"
