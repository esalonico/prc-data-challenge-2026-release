#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Step 75 (SERVE_WS, after step 70): the LIRF label-mechanism rule on LIRF departures without a Network Manager record.
#   IN=<step-70 file, default F1>  OUT=<default F2F>  PARTS=<AB|A|B, default B>
#   B_MODE=<fmt_special13|fmt_special|fmt|season|crowd, default fmt_special13>
#     fmt_special13  special operators -> schedule copy, including those 1-3 h late; others season x callsign format (F2F, the final)
#     fmt_special    as fmt_special13 without the 1-3 h rows (F2E)
#     fmt            season x callsign format (F2D); season = season only (F2B); crowd = season x NM-gap crowding (F2C)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
in_serve
IN=${IN:-F1}; OUT=${OUT:-F2F}; PARTS=${PARTS:-B}; B_MODE=${B_MODE:-fmt_special13}
"$PY" "$STEPS/lirf_rule.py" apply --in "$OUTF/$IN.parquet" --ranking data/raw/ranking.parquet --out "$OUTF/$OUT.parquet" --parts "$PARTS" --b-mode "$B_MODE"
log "final file: $OUTF/$OUT.parquet"
