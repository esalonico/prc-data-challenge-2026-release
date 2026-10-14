#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Step 35 (SERVE_WS, after step 30): trajectory store for the serve days. extract.py on every 2026 stage-1 day (the
# dictionary variant left in $F/stage1 by step 30), then append-merge into the shipped store (2025 rows kept, the old
# Jan/Jul rows replaced). Re-extracting shipped days reproduces the stored rows and points.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
in_serve
[ -d $T/store.shipped ] || mv $T/store $T/store.shipped
rm -rf $T/days/2026-*
DAYS=$(ls $F/stage1 | grep '^2026-' | paste -sd, -)
log "serve days: $(echo "$DAYS" | tr ',' '\n' | wc -l | tr -d ' ')"
"$PY" scripts/legs/traj/extract.py --days "$DAYS"
"$PY" "$STEPS/traj_store_merge.py" --days-glob '2026-*'
log "trajectory store done"
