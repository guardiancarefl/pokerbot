#!/bin/bash
# EXP_H2 verdict battery (spec §5). Run AFTER runs/H2_PROBE.DONE exists.
# Usage: bash scripts/h2_verdict_battery.sh
set -u
cd ~/pokerbot
PROBE=runs/h2_probe_league_v1_resume/checkpoints/ckpt_iter_2000.pt
CHAMP=runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt
ABS=runs/abstraction_20260521_223018_retrofit/abstraction.pkl
D=evals/h2_verdict_20260612
mkdir -p $D
[ -f "$PROBE" ] || { echo "ABORT: $PROBE missing"; exit 1; }
if ps axww | grep "python scripts/run_live_dryrun.py" | grep -v grep >/dev/null; then
  echo "ABORT: live listener active"; exit 1; fi

# F-M1: battery grade (cores 4-5, ~10 min)
nice -n 19 taskset -c 4-5 .venv/bin/python -m scripts.fold_vs_shove_battery grade \
  --battery evals/h2_battery/battery_v1.json --ckpt $PROBE \
  --out $D/m1_probe.json > $D/m1_probe.log 2>&1 &
M1=$!

# F-M2a/b rows: CRN 2000 games each (cores 6-11, parallel)
C=6
for P in killphilmtt ticketmaster sng tighttom; do
  nice -n 19 taskset -c $C .venv/bin/python -m scripts.sng_baseline \
    --ckpt $PROBE --abstraction $ABS --profiles $P --games 2000 \
    --master-seed 2026 --out-dir $D/$P > $D/$P.log 2>&1 &
  C=$((C+1))
done
# F-M2b(c) self-anchor: probe vs champion, standard mode, 2000 games (cores 10-11)
nice -n 19 taskset -c 10-11 .venv/bin/python -m scripts.attacker_extraction_eval \
  --attacker-ckpt $PROBE --champion-ckpt $CHAMP --abstraction $ABS \
  --games 2000 --floors off --mode standard \
  --out $D/self_anchor.jsonl --log-every 500 > $D/self_anchor.log 2>&1 &

wait $M1
wait
touch $D/BATTERY.DONE
echo "H2 VERDICT BATTERY COMPLETE"
grep -h "shanky:" $D/*.log 2>/dev/null | tail -5
grep "RESULT" $D/self_anchor.log 2>/dev/null
cat $D/m1_probe.json 2>/dev/null
