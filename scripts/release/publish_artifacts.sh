#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the unbelievable-snowflake team (PRC Data Challenge 2026)
# This file is part of the team's PRC Data Challenge 2026 solution; see LICENSE (GNU GPL v3) and NOTICE.md.

# Owner only: upload the release artifacts to the public bucket and write manifests/artifacts.sha256.
#
#   RCLONE_REMOTE=<rclone remote for R2> BUCKET=<bucket> PUBLIC_BASE_URL=<https://pub-....r2.dev> \
#     scripts/release/publish_artifacts.sh [manifests/artifacts.list]
#
# The list holds one repo-relative path per line: stored models (artifacts/...) and unmodified open-data snapshots
# (data/external/...). It never holds challenge data, files keyed by MVT_ID_mvt or ADS-B-derived files; the script
# refuses any path outside those two roots and any prediction or feature file name. Sources are read from this
# checkout; objects go to <bucket>/<PREFIX>/<path> (PREFIX default: final), next to MANIFEST.sha256.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIST="${1:-$REPO/manifests/artifacts.list}"
PREFIX="${PREFIX:-final}"
: "${RCLONE_REMOTE:?set RCLONE_REMOTE}" "${BUCKET:?set BUCKET}" "${PUBLIC_BASE_URL:?set PUBLIC_BASE_URL}"
OUT="$REPO/manifests/artifacts.sha256"

while IFS= read -r p; do
  [ -z "$p" ] && continue
  case "$p" in
    artifacts/*|data/external/iem_*/*|data/external/osm/*|data/external/ourairports/*) ;;
    *) echo "refused (outside the allowed roots): $p"; exit 1 ;;
  esac
  case "$(basename "$p")" in
    *ranking*.parquet|*predictions*|*oof*|*matrix*.parquet|*block*.parquet|*features*.parquet|*slim*)
      echo "refused (keyed by row or ADS-B-derived): $p"; exit 1 ;;
  esac
  [ -f "$REPO/$p" ] || { echo "missing: $p"; exit 1; }
done < "$LIST"

size() { if [ "$(uname)" = Darwin ]; then stat -f%z "$1"; else stat -c%s "$1"; fi; }
{
  echo "# base_url: ${PUBLIC_BASE_URL%/}/$PREFIX"
  while IFS= read -r p; do
    [ -z "$p" ] && continue
    echo "$(shasum -a 256 "$REPO/$p" | cut -d' ' -f1)  $(size "$REPO/$p")  $p"
  done < "$LIST"
} > "$OUT"
echo "wrote $OUT ($(grep -vc '^#' "$OUT") files)"

rclone copy --s3-no-check-bucket --files-from "$LIST" --transfers 8 --progress "$REPO" "$RCLONE_REMOTE:$BUCKET/$PREFIX"
rclone copyto --s3-no-check-bucket "$OUT" "$RCLONE_REMOTE:$BUCKET/$PREFIX/MANIFEST.sha256"
echo "published to $RCLONE_REMOTE:$BUCKET/$PREFIX; verify with: scripts/release/fetch_artifacts.py into a fresh clone"
