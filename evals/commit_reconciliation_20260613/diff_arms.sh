#!/bin/bash
# Diff two arm dirs log-by-log with --strict. $1=preDir $2=postDir $3=outFile
set -u
cd /home/quant/pokerbot
source .venv/bin/activate
PRE="$1"; POST="$2"; OUTF="$3"
: > "$OUTF"
for f in "$PRE"/*.jsonl; do
  L=$(basename "$f" .jsonl)
  echo "=== $L ===" >> "$OUTF"
  python scripts/replay_make_decision_diff.py diff --pre "$f" \
    --post "$POST/$L.jsonl" --strict >> "$OUTF" 2>&1
done
echo "diff done -> $OUTF"
