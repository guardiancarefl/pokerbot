#!/bin/bash
# Armed-arm runner: $1 = outdir, $2 = mode (sample|argmax), $3.. = extra flags
set -u
cd /home/quant/pokerbot
source .venv/bin/activate
OUT="$1"; MODE="$2"; shift 2
mkdir -p "$OUT"
LOGS="live_dryrun_20260608_152756 live_dryrun_20260609_154557 live_dryrun_20260609_192543 live_dryrun_20260611_163815 live_dryrun_20260611_181249 live_dryrun_20260611_201230 live_dryrun_20260611_204751 live_dryrun_20260611_222532 live_dryrun_20260612_035454 live_dryrun_20260612_042058 live_dryrun_20260612_125441 live_dryrun_20260612_130846 live_dryrun_20260612_161347 live_dryrun_verify1_20260609_202137 live_dryrun_20260612_183701 live_dryrun_20260612_220101 live_dryrun_20260612_230149"
for L in $LOGS; do
  nice -n 19 taskset -c 4-5 python scripts/replay_make_decision_diff.py run \
    --log "logs/${L}.jsonl" --out "${OUT}/${L}.jsonl" \
    --checkpoint runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt \
    --abstraction runs/abstraction_20260521_223018_retrofit/abstraction.pkl \
    --seed 42 --mode "$MODE" "$@" || echo "FAILED ${L}"
done
echo "ARM DONE -> ${OUT}"
