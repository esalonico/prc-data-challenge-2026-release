#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Step 30 (SERVE_WS, after step 10): ADS-B blocks for the four serve months.
# The shipped artifacts come from two stage-1 variants (configs/final/adsb_shipped_build.json):
#   no dictionary -> the prize all-year block (the matrix block of every tabular leg);
#   dictionary    -> the v3-lean block, the D-pool feature blocks (unmatched sensor + channels), the trajectory store.
# Each variant's 2025 pool is copied from REGRESS_WS; the 120 serve days are built from final_ranking.parquet with a
# callsign dictionary over 2025 months 1-10 + the final file (the recipe of the shipped one). The shipped 2025 rows of
# the prize and lean blocks are kept byte for byte (splice_block.py). The dictionary stage 1 is left in $F/stage1 for
# the trajectory extraction (step 35).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
in_serve
DICT=$F/features/callsign_dictionary.parquet
pool() { rm -rf "$2" && mkdir -p "$2" && for d in "$REGRESS_WS/$F/$1"/2025-*; do cow_copy "$d" "$2/"; done; }
[ -d "$REGRESS_WS/$F/stage1_dict" ] && [ -d "$REGRESS_WS/$F/stage1_nodict" ] || { echo "run step 10 first"; exit 1; }
[ -f $F/prize/adsb_block_prize_allyear.shipped.parquet ] || cp $F/prize/adsb_block_prize_allyear.parquet $F/prize/adsb_block_prize_allyear.shipped.parquet
[ -f $C/v3/adsb_v3lean_block_allyear.shipped.parquet ] || cp $C/v3/adsb_v3lean_block_allyear.parquet $C/v3/adsb_v3lean_block_allyear.shipped.parquet
mkdir -p $F/features
"$PY" scripts/features/adsb_callsign_dictionary.py
"$PY" "$STEPS/pin_linker.py" $F/features/linker_precision.json

log "no-dictionary variant: prize block"
pool stage1_nodict $F/stage1
mv $DICT $DICT.off
"$PY" scripts/features/build_adsb_features.py stage1 --scope ranking
"$PY" scripts/features/build_adsb_features.py prize
mv $DICT.off $DICT
cp $F/prize/adsb_block_prize_allyear.parquet $F/prize/adsb_block_prize_allyear.rebuilt.parquet
"$PY" "$STEPS/splice_block.py" --shipped $F/prize/adsb_block_prize_allyear.shipped.parquet \
  --ranking $F/prize/adsb_block_prize_allyear.rebuilt.parquet --out $F/prize/adsb_block_prize_allyear.parquet
rm -rf $F/stage1_nodict && mv $F/stage1 $F/stage1_nodict

log "dictionary variant: D-pool features, v3-lean"
pool stage1_dict $F/stage1
"$PY" scripts/features/build_adsb_features.py stage1 --scope ranking
"$PY" scripts/features/build_adsb_features.py features
rm -rf $C/stage3 && mkdir -p $C/stage3
for f in "$REGRESS_WS/$C/stage3"/*2025-*; do cow_copy "$f" $C/stage3/; done
"$PY" scripts/features/build_adsb_coverage.py kin --months 2026-01,2026-02,2026-06,2026-07
"$PY" scripts/features/build_adsb_coverage.py block --tags allyear
"$PY" scripts/features/build_adsb_coverage.py lean --tags allyear
cp $C/v3/adsb_v3lean_block_allyear.parquet $C/v3/adsb_v3lean_block_allyear.rebuilt.parquet
"$PY" "$STEPS/splice_block.py" --shipped $C/v3/adsb_v3lean_block_allyear.shipped.parquet \
  --ranking $C/v3/adsb_v3lean_block_allyear.rebuilt.parquet --out $C/v3/adsb_v3lean_block_allyear.parquet

log "unmatched chain: lineage frame from REGRESS_WS (2025 months 1-10 only)"
mkdir -p $A/Unmatched-noncopy-20260925/ranking/frames
cp "$REGRESS_WS/$A/Unmatched-noncopy-20260925/ranking/frames/lineage_fit_m1_10.parquet" $A/Unmatched-noncopy-20260925/ranking/frames/
log "serve ADS-B done"
