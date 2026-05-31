#!/usr/bin/env bash
# X6 Arm 1 — STAGED, MANUAL-LAUNCH ONLY. Does NOT auto-start; run this by hand only
# on explicit go (cores must be free; Step-2 br arms must have landed). Arm 2 (d=4)
# and its d=4 timing check are DELETED: the grounded pre-estimate (~37h for 120 hands;
# the 20-hand check itself ~6h) made d=4 oracle both unshippable and unrunnable, so
# confirming a 37h number with a 6h run is pure waste. This collapses the original
# three-way decision rule to TWO-WAY (see commit message / DECISIONS).
#
# Arm 1 (variance probe): BEST_RESPONSE leaves, n_samples=32, --no-icm-short-circuit,
# opponent_prior=None (harness default), max_action_depth=3, 120 hands, condition A,
# CRN-paired to J2/J3 (killphil 2026 + ticketmaster 3026), NATURAL sampling.
#   >= blueprint-alone -> constraint was VARIANCE -> cheap config fix (raise M, drop sc).
#   <  blueprint-alone -> variance NOT the constraint; does NOT distinguish unsafe-solve
#      from biased-continuation (Arm 2 was the separator, dead on cost) -> report as
#      "deep rebuild required, fork unresolved".
set -u
cd ~/pokerbot
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CKPT=runs/six_max_20260530_034023_phase4f_dcfr_candC_k200/checkpoints/ckpt_iter_2000.pt
ABS=runs/abstraction_20260521_223018_retrofit/abstraction.pkl
STRUCT=configs/ignition_double_up_6max_turbo.yaml
SDIR=data/shanky_profiles
OUTDIR=evals/resolver_shards

run_resolver() {  # tag profile seed
  tag=$1; p=$2; seed=$3
  out=$OUTDIR/${tag}.jsonl; log=$OUTDIR/${tag}.log
  echo "[x6-arm1 start $(date +%H:%M:%S)] $tag p=$p seed=$seed d=3 M=32 sc-off n=120"
  python -m scripts.eval_resolver_vs_shanky \
    --ckpt "$CKPT" --abstraction "$ABS" --structure "$STRUCT" \
    --shanky-dir "$SDIR" --conditions A --only "$p" \
    --leaf-mode br --n-samples 32 --no-icm-short-circuit \
    --max-action-depth 3 --hands 120 --seed "$seed" \
    --out "$out" > "$log" 2>&1
  echo "[x6-arm1 done  $(date +%H:%M:%S)] $tag rc=$?"
}
export CKPT ABS STRUCT SDIR OUTDIR

echo "[x6-arm1] $(date) launching Arm 1 (both profiles, parallel)"
run_resolver X6_var_A_killphilmtt   killphilmtt  2026 &
run_resolver X6_var_A_ticketmaster  ticketmaster 3026 &
wait
echo "X6_ARM1_DONE $(date)" > "$OUTDIR/X6_ARM1_DONE.flag"
echo "=== x6 Arm 1 complete: $(date) ==="
