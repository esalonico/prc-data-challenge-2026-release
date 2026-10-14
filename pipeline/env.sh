# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Sourced by pipeline/*.sh (the four-month serve path).
#
# Two workspaces, both copy-on-write copies of this checkout (code, data/ and artifacts/):
#   SERVE_WS    data/raw/ranking.parquet = final_ranking.parquet (Jan, Feb, Jun, Jul 2026); builds every 2026 serve row
#   REGRESS_WS  the public Jan/Jul ranking file; rebuilds the 2025 ADS-B pool exactly as shipped and checks it
# Override the locations with SERVE_WS / REGRESS_WS; PY defaults to this checkout's .venv.
PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS="$PIPE/steps"
REPO="$(cd "$PIPE/.." && pwd)"
SERVE_WS="${SERVE_WS:-$HOME/prc-ws/serve}"
REGRESS_WS="${REGRESS_WS:-$HOME/prc-ws/regress}"
PY="${PY:-$REPO/.venv/bin/python}"
A=artifacts/experiments
RUNS=$A/S4-allyear-rebuild-20260925/runs
F=$A/ADSB-features-20260925
C=$A/ADSB-coverage-20260926
T=$A/ADSB-trajectory-20260927
OUT_REL=$A/Final-serve4m
OUTF="$SERVE_WS/$OUT_REL"
CFG="$REPO/configs/final"

common_env() {
  unset VIRTUAL_ENV
  export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH=src PYTHONUNBUFFERED=1
}
in_regress() {
  cd "$REGRESS_WS" || exit 1
  common_env
  export PRC_SOURCE_ROOT="$PWD" PRC_MAIN_REPO="$PWD"
  unset PRC_SERVE_MONTHS PRC_RANKING_N PRC_RANKING_UNMATCHED_N
}
in_serve() {
  cd "$SERVE_WS" || exit 1
  common_env
  export PRC_SOURCE_ROOT="$PWD" PRC_MAIN_REPO="$PWD"
  export PRC_SERVE_MONTHS=1,2,6,7 PRC_RANKING_N=670790 PRC_RANKING_UNMATCHED_N=9057
}
cow_copy() {  # copy-on-write where the filesystem supports it (APFS, btrfs, xfs); a full copy otherwise
  if [ "$(uname)" = Darwin ]; then cp -Rc "$1" "$2"; else cp -a --reflink=auto "$1" "$2"; fi
}
log() { echo "[$(date +%H:%M:%S)] $*"; }
