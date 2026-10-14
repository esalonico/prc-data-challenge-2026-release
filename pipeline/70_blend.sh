#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Step 70 (SERVE_WS, after steps 20, 30, 60): the final file.
#   method-d cell-weighted blend of the fourteen legs (CB8K CatBoost legs) -> T1 (stored all-year boosters) ->
#   unmatched chain (scripts/compose/unm_serve.py: CAP_B + P1 + U-SENS-G + ALL + C0 + U-MAP2V-NF; no LIRF24) ->
#   LOBT window (+-3606 s) -> clip >= 0, validated against final_submitting.parquet.
# Output: $OUTF/<NAME>.parquet (+ .manifest.json, .validation.json). NAME defaults to F0; REF=<parquet> adds a comparison.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
in_serve
NAME=${NAME:-F0}
L="$OUTF/legs"
WEIGHTS=${WEIGHTS:-$CFG/blend_weights_d.json}
LGB_SET=${LGB_SET:-}   # "" = the first LightGBM legs (F0); "01" = the lr 0.01 legs (F1 and the final)
LEGS=(cb_plain=$L/ranking_cb8k.parquet cb_delta=$L/ranking_cbd8k.parquet cb_eobtres=$L/ranking_cbe8k.parquet
      lgbm_plain=$L/ranking_lgb$LGB_SET.parquet lgbm_delta=$L/ranking_lgbd$LGB_SET.parquet
      lgbm_eobtres=$L/ranking_lgbe$LGB_SET.parquet lgbm_lobtoff=$L/ranking_lgbl$LGB_SET.parquet
      nn_v14=$L/ranking_nn14.parquet nn_v15=$L/ranking_nn15.parquet nn_v16=$L/ranking_nn16.parquet
      plr_s03_resnet_f3=$L/ranking_plr_s03_resnet_f3.parquet plr_s03_eobtres_f3=$L/ranking_plr_s03_eobtres_f3.parquet
      ctx_gru_plr_f2=$L/ranking_ctx_gru_plr_f2.parquet traj_conv_resnet_f3=$L/ranking_traj_conv_resnet_f3.parquet)
ARGS=(); for x in "${LEGS[@]}"; do ARGS+=(--leg "$x"); done
log "blend + T1"
"$PY" "$STEPS/blend_t1.py" --weights "$WEIGHTS" --cell-matrix $RUNS/serve4m_allyear_iso_v3l/eval_matrix.parquet "${ARGS[@]}" \
  --t1-run $RUNS/serve4m_t1 --out-dir "$OUTF" --tag "$NAME"
log "unmatched chain"
"$PY" scripts/compose/unm_serve.py --out-dir "$OUTF/unmatched"
"$PY" -c "
import polars as pl
u = pl.read_parquet('$OUTF/unmatched/unmatched_serve.parquet').select('MVT_ID_mvt', pl.col('prediction').alias('TAXITIME_SEC_mvt'))
u.write_parquet('$OUTF/unmatched/unmatched_for_compose.parquet'); print('unmatched rows', u.height)"
log "compose"
"$PY" scripts/compose/compose_candidate.py --matched "$OUTF/${NAME}_blend_matched_T1.parquet" --unmatched "$OUTF/unmatched/unmatched_for_compose.parquet" \
  --window-mode lobt --out "$OUTF/$NAME.parquet" > "$OUTF/$NAME.compose.log"
"$PY" - "$OUTF/$NAME.parquet" "${REF:-}" <<'EOF'
import hashlib, json, sys
from pathlib import Path
import numpy as np, polars as pl
path, ref = Path(sys.argv[1]), sys.argv[2]
f = pl.read_parquet(path)
t = pl.read_parquet("data/raw/final_submitting.parquet")
y = f["TAXITIME_SEC_mvt"].to_numpy()
rep = {"rows": f.height, "ids_equal_final_submitting_order": bool(f["MVT_ID_mvt"].equals(t["MVT_ID_mvt"])),
       "finite": bool(np.isfinite(y).all()), "min": float(y.min()), "mean": float(y.mean()),
       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
if ref:
    r = pl.read_parquet(ref).rename({"TAXITIME_SEC_mvt": "ref"})
    j = f.join(r, on="MVT_ID_mvt")
    d = np.abs(j["TAXITIME_SEC_mvt"].to_numpy() - j["ref"].to_numpy())
    rep["vs_ref"] = {"ref": ref, "n": j.height, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "n_gt_1e-6": int((d > 1e-6).sum())}
path.with_suffix(".validation.json").write_text(json.dumps(rep, indent=1) + "\n")
print(json.dumps(rep))
if not (rep["rows"] == 670790 and rep["ids_equal_final_submitting_order"] and rep["finite"] and rep["min"] >= 0):
    raise SystemExit("final file failed validation")
EOF
log "final file: $OUTF/$NAME.parquet"
