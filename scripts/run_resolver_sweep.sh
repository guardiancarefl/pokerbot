#!/usr/bin/env bash
# Resolver gate-widening sweep driver (curated worst-opponents pool, parallel).
# 5 worst legit profiles x 3 gate conditions x 600 hands, sharded 8-wide.
# Plus a blueprint-alone baseline (for ICM-lift column). Writes one jsonl per
# shard + a baseline json; sets SWEEP_DONE.flag on completion.
set -u
cd ~/pokerbot
source .venv/bin/activate
# Single-threaded per process so 8 concurrent solvers don't thrash 12 vCPU.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CKPT=runs/six_max_20260530_034023_phase4f_dcfr_candC_k200/checkpoints/ckpt_iter_2000.pt
ABS=runs/abstraction_20260521_223018_retrofit/abstraction.pkl
STRUCT=configs/ignition_double_up_6max_turbo.yaml
SDIR=data/shanky_profiles
HANDS=600
SHARD_DIR=evals/resolver_shards
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/SWEEP_DONE.flag

# Profiles (index drives the per-profile seed; SAME seed across conditions => CRN).
PROFILES=(killphilmtt ticketmaster thefixersng minestackermttv.7.3 gushansenmtt)

echo "=== $(date) baseline: blueprint-alone vs 5 profiles ==="
python -m scripts.eval_shanky_vs_dcfr \
  --challenger-ckpt "$CKPT" --challenger-name candC-2000 \
  --shanky-dir "$SDIR" --abstraction "$ABS" --structure "$STRUCT" \
  --only "${PROFILES[@]}" --hands "$HANDS" --seed 2026 \
  --output "$SHARD_DIR/baseline_blueprint.json" \
  > "$SHARD_DIR/baseline.log" 2>&1
echo "=== baseline done: $(date) ==="

# Build the 15 resolver jobs as "cond|profile|seed". Order: killphil (slowest)
# conditions first so they claim pool slots at t0.
JOBS=()
for c in C B A; do
  j=0
  for p in "${PROFILES[@]}"; do
    seed=$((2026 + j * 1000))
    JOBS+=("$c|$p|$seed")
    j=$((j + 1))
  done
done

run_one() {
  job="$1"
  c="${job%%|*}"; rest="${job#*|}"; p="${rest%%|*}"; seed="${rest##*|}"
  out="evals/resolver_shards/shard_${c}_${p}.jsonl"
  log="evals/resolver_shards/shard_${c}_${p}.log"
  echo "[start $(date +%H:%M:%S)] cond=$c profile=$p seed=$seed"
  python -m scripts.eval_resolver_vs_shanky \
    --ckpt "$CKPT_E" --abstraction "$ABS_E" --structure "$STRUCT_E" \
    --shanky-dir "$SDIR_E" --conditions "$c" --only "$p" \
    --hands "$HANDS_E" --seed "$seed" --out "$out" > "$log" 2>&1
  echo "[done  $(date +%H:%M:%S)] cond=$c profile=$p rc=$?"
}
export -f run_one
export CKPT_E="$CKPT" ABS_E="$ABS" STRUCT_E="$STRUCT" SDIR_E="$SDIR" HANDS_E="$HANDS"

echo "=== $(date) launching ${#JOBS[@]} resolver jobs, 8-wide ==="
printf '%s\n' "${JOBS[@]}" | xargs -I{} -P 8 bash -c 'run_one "$@"' _ {}

echo "ALL_DONE $(date)" > "$SHARD_DIR/SWEEP_DONE.flag"
echo "=== sweep complete: $(date) ==="
