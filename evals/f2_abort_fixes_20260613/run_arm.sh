#!/bin/bash
# G1 arm runner: replay all 17 raw-record logs through make_decision at
# a given code tree, flag-OFF, sample mode, seed 42 (the D2/CR gate-1
# protocol). $1=code tree root  $2=output dir  [$3=extra flags]
set -u
TREE="$1"; OUTDIR="$2"; EXTRA="${3:-}"
PY=/home/quant/pokerbot/.venv/bin/python
CKPT=/home/quant/pokerbot/runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt
ABS=/home/quant/pokerbot/runs/abstraction_20260521_223018_retrofit/abstraction.pkl
mkdir -p "$OUTDIR"
cd "$TREE"
for f in \
  live_dryrun_20260608_152756 live_dryrun_20260609_154557 \
  live_dryrun_20260609_192543 live_dryrun_20260611_163815 \
  live_dryrun_20260611_181249 live_dryrun_20260611_201230 \
  live_dryrun_20260611_204751 live_dryrun_20260611_222532 \
  live_dryrun_20260612_035454 live_dryrun_20260612_042058 \
  live_dryrun_20260612_125441 live_dryrun_20260612_130846 \
  live_dryrun_20260612_161347 live_dryrun_20260612_183701 \
  live_dryrun_20260612_220101 live_dryrun_20260612_230149 \
  live_dryrun_verify1_20260609_202137; do
  nice -n 19 taskset -c 4-7 "$PY" scripts/replay_make_decision_diff.py run \
    --log /home/quant/pokerbot/logs/$f.jsonl \
    --out "$OUTDIR/$f.jsonl" \
    --checkpoint "$CKPT" --abstraction "$ABS" --seed 42 $EXTRA \
    > "$OUTDIR/$f.stdout" 2>&1 || echo "FAIL $f" >> "$OUTDIR/FAILURES.txt"
done
echo done > "$OUTDIR/ARM_DONE.flag"
