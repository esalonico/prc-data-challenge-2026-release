#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Pull slim adsb.lol days from the ingest VM every 10 min until its batch + slim loops are done.
#   ADSB_VM=<user@host> scripts/data/adsb/sync_slim.sh     (DEST defaults to data/external/adsb_lol/slim)
set -u
: "${ADSB_VM:?set ADSB_VM=user@host of the ingest VM}"
DEST="${DEST:-$(cd "$(dirname "$0")/../../.." && pwd)/data/external/adsb_lol/slim}"
mkdir -p "$DEST"
while true; do
  rsync -a --partial "$ADSB_VM:adsb/slim/" "$DEST/" --exclude='*.tmp' 2>&1 | tail -2
  rsync -a "$ADSB_VM:adsb/out/batch_status.jsonl" "$DEST/" 2>/dev/null
  n=$(ls "$DEST"/*_slim.parquet 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date -u +%H:%M:%S) slim days here=$n"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$ADSB_VM" 'tmux list-windows -t adsb 2>/dev/null | grep -q batch'; then
    sleep 300
    rsync -a --partial "$ADSB_VM:adsb/slim/" "$DEST/" --exclude='*.tmp'
    echo "batch finished; final sync done"; break
  fi
  sleep 600
done
