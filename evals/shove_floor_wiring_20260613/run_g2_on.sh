#!/bin/bash
# G2 flag-ON arm: replay all 17 logs with --shove-defense-floor armed
# (registered tau=0.0, default battery range table), seed 42, sample mode.
set -u
cd /home/quant/pokerbot
source .venv/bin/activate
OUT="$1"
mkdir -p "$OUT"
LOGS="live_dryrun_20260608_152756 live_dryrun_20260609_154557 live_dryrun_20260609_192543 live_dryrun_20260611_163815 live_dryrun_20260611_181249 live_dryrun_20260611_201230 live_dryrun_20260611_204751 live_dryrun_20260611_222532 live_dryrun_20260612_035454 live_dryrun_20260612_042058 live_dryrun_20260612_125441 live_dryrun_20260612_130846 live_dryrun_20260612_161347 live_dryrun_verify1_20260609_202137 live_dryrun_20260612_183701 live_dryrun_20260612_220101 live_dryrun_20260612_230149"
for L in $LOGS; do
  nice -n 19 taskset -c 4-7 python scripts/replay_make_decision_diff.py run \
    --log "logs/${L}.jsonl" --out "${OUT}/${L}.jsonl" \
    --checkpoint runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt \
    --abstraction runs/abstraction_20260521_223018_retrofit/abstraction.pkl \
    --shove-defense-floor --shove-defense-tau 0.0 \
    --seed 42 || echo "FAILED ${L}"
done
echo "G2 ON ARM DONE -> ${OUT}"
