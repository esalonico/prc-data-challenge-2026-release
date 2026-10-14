#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Step 10 (REGRESS_WS, the public Jan/Jul ranking file): rebuild what the serve path takes from 2025, and check it
# against the shipped artifacts.
#   * the isocap V3L all-year matrices (must equal train c67a87cf... / eval b5b8212d...);
#   * ADS-B stage 1 in both variants with the shipped day set:
#       dictionary    -> v3 stage 3 (kinematics + travel-time votes) and the lean block (must equal e7b81caa...);
#       no dictionary -> prize blocks (all-year must be value-identical to 70cfd2e5...; dfit byte-identical 0b39f62c...);
#   * the months 1-10 P(copy) lineage frame (unmatched chain).
# Outputs used by the serve steps: $F/stage1_dict, $F/stage1_nodict, $C/stage3, the lineage frame.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
in_regress
CHK="$A/Final-regress"
mkdir -p "$CHK"
DICT=$F/features/callsign_dictionary.parquet
EXTRA=(--extra-features "$A/Matched-features-20260925/m1_stand_v1s/allyear_plus_ranking.parquet"
       --extra-features "$A/Matched-features-20260925/m2_follow_v1s/allyear_plus_ranking.parquet"
       --extra-features "$A/Matched-features-20260925/m4_p_v1/s4_all.parquet"
       --extra-features "$A/AOBT-local-only-20260925/features/aobt_features_allyear_plus_ranking.parquet"
       --extra-features "$F/prize/adsb_block_prize_allyear.parquet"
       --extra-features "$C/v3/adsb_v3lean_block_allyear.parquet")
log "assemble the shipped all-year isocap V3L matrices"
"$PY" scripts/legs/s4_allyear.py --stage fit --train-months 1-12 --eval ranking --tag regress_allyear_iso_v3l "${EXTRA[@]}" \
  --temporal-mode isocap_v1 --threads 8 --assemble-only
cp -c "$F/prize/adsb_block_prize_allyear.parquet" "$CHK/prize_allyear.shipped.parquet" 2>/dev/null || cp "$F/prize/adsb_block_prize_allyear.parquet" "$CHK/prize_allyear.shipped.parquet"
cp "$F/prize/adsb_block_prize_dfit.parquet" "$CHK/prize_dfit.shipped.parquet"
cp "$C/v3/adsb_v3lean_block_allyear.parquet" "$CHK/v3lean_allyear.shipped.parquet"

log "ADS-B, dictionary variant"
"$PY" scripts/features/adsb_callsign_dictionary.py
"$PY" "$STEPS/pin_linker.py" "$F/features/linker_precision.json"
rm -rf "$F/stage1" "$C/stage3"
"$PY" scripts/features/build_adsb_features.py stage1 --scope D,janjul,ranking,v4
"$PY" scripts/features/build_adsb_coverage.py kin
"$PY" scripts/features/build_adsb_coverage.py votes
"$PY" scripts/features/build_adsb_features.py prize            # the v3 block aligns its rows to the prize file
"$PY" scripts/features/build_adsb_coverage.py block --tags allyear
"$PY" scripts/features/build_adsb_coverage.py lean --tags allyear
mv "$F/stage1" "$F/stage1_dict"
cp "$CHK/prize_allyear.shipped.parquet" "$F/prize/adsb_block_prize_allyear.parquet"
cp "$CHK/prize_dfit.shipped.parquet" "$F/prize/adsb_block_prize_dfit.parquet"

log "ADS-B, no-dictionary variant"
mv "$DICT" "$DICT.off"
"$PY" scripts/features/build_adsb_features.py stage1 --scope D,janjul,ranking,v4
"$PY" scripts/features/build_adsb_features.py prize
mv "$DICT.off" "$DICT"
mv "$F/stage1" "$F/stage1_nodict"

log "unmatched chain: months 1-10 lineage frame"
"$PY" scripts/compose/build_unmatched_noncopy_ranking.py --phase lineage

log "checks"
"$PY" - "$CHK" <<'EOF'
import hashlib, json, sys
from pathlib import Path
import numpy as np, polars as pl
chk = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
R = "artifacts/experiments/S4-allyear-rebuild-20260925/runs/regress_allyear_iso_v3l"
F = "artifacts/experiments/ADSB-features-20260925/prize"
res = {"train_matrix": sha(f"{R}/train_matrix.parquet")[:16], "eval_matrix": sha(f"{R}/eval_matrix.parquet")[:16],
       "v3lean_allyear": sha("artifacts/experiments/ADSB-coverage-20260926/v3/adsb_v3lean_block_allyear.parquet")[:16],
       "prize_dfit": sha(f"{F}/adsb_block_prize_dfit.parquet")[:16]}
a = pl.read_parquet(chk / "prize_allyear.shipped.parquet").sort("MVT_ID_mvt")
b = pl.read_parquet(f"{F}/adsb_block_prize_allyear.parquet").sort("MVT_ID_mvt")
res["prize_allyear_value_identical"] = bool(a.height == b.height and a.equals(b, null_equal=True))
expect = {"train_matrix": "c67a87cfbedba013", "eval_matrix": "b5b8212d", "v3lean_allyear": "e7b81caa02c1c022", "prize_dfit": "0b39f62caef299b6"}
res["ok"] = all(res[k].startswith(v) for k, v in expect.items()) and res["prize_allyear_value_identical"]
(chk / "regress_checks.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res))
if not res["ok"]:
    raise SystemExit("regression checks failed")
EOF
log "regress done"
