#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Step 0: fresh SERVE_WS and REGRESS_WS (copy-on-write copies of this checkout).
# Pipeline outputs are deleted from both so that every one of them is rebuilt by the later steps.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
for f in ranking.parquet submitting.parquet final_ranking.parquet final_submitting.parquet; do
  [ -f "$REPO/data/raw/$f" ] || { echo "missing data/raw/$f (scripts/data/download_data.py)"; exit 1; }
done
rm -rf "$REGRESS_WS" "$SERVE_WS"
mkdir -p "$(dirname "$REGRESS_WS")" "$(dirname "$SERVE_WS")"
log "copy $REPO -> $REGRESS_WS, $SERVE_WS"
cow_copy "$REPO" "$REGRESS_WS"
cow_copy "$REPO" "$SERVE_WS"
for ws in "$REGRESS_WS" "$SERVE_WS"; do
  rm -rf "$ws/$F/stage1" "$ws/$F/stage1_dict" "$ws/$F/stage1_nodict" "$ws/$F/features" "$ws/$C/stage3" "$ws/$T/days" \
         "$ws/$T"/seq_* "$ws/$A/Unmatched-noncopy-20260925/ranking/frames" "$ws/$A/Unmatched-sensor-20260926/state.json" \
         "$ws/$OUT_REL" "$ws/$A/Final-regress" "$ws/$RUNS"/serve4m_* "$ws/$RUNS"/nnx_cache_ay3liso_* "$ws/$RUNS"/regress_allyear_iso_v3l
done
# serve: the final files take the ranking / submitting slots; 2026 caches built from the public file are purged
(cd "$SERVE_WS/data/raw" && mv ranking.parquet ranking_public.parquet && mv submitting.parquet submitting_public.parquet \
   && cp final_ranking.parquet ranking.parquet && cp final_submitting.parquet submitting.parquet)
(cd "$SERVE_WS/$A/S4-allyear-rebuild-20260925/features" && rm -f base/base_2026-* native_state/native_2026-* \
   native_arrivals/arrivals_2026-* events/events_2026-* temporal_v2/*/temporal_v2_2026-* && rm -rf s4_learned/2026-*)
# regress: hide the slim days the shipped ADS-B build did not have, and the 2026 days other than Jan, Jul and 06-30
"$PY" - "$REGRESS_WS/data/external/adsb_lol" "$CFG/adsb_shipped_build.json" <<'EOF'
import json, re, shutil, sys
from pathlib import Path
root, cfg = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
hide = set(cfg["extra_days_hidden_for_the_2025_pool"])
(root / "slim_hidden").mkdir(exist_ok=True)
n = 0
for p in sorted((root / "slim").glob("adsb_lol_*_slim*")):
    d = re.findall(r"\d{4}-\d{2}-\d{2}", p.name)[0]
    if d in hide or (d.startswith("2026") and not (d[5:7] in ("01", "07") or d == "2026-06-30")):
        shutil.move(str(p), root / "slim_hidden" / p.name)
        n += 1
print(f"regress: {n} slim files hidden")
EOF
log "workspaces ready"
