#!/usr/bin/env bash
# Nightly routine: generate -> score -> digest for one date (default: today).
# Safe to rerun: each step only fills in what is missing for that date.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs
# shellcheck disable=SC1091
source .venv/bin/activate
DATE="${1:-$(date +%F)}"
echo "[nightly] $(date -Is) start date=$DATE"
python 02_generate.py --date "$DATE"
python 03_score.py --date "$DATE"
python 04_digest.py --date "$DATE"
echo "[nightly] $(date -Is) done -> digests/$DATE.md"
