#!/bin/bash
# Eval every checkpoint in a DCFR overnight run against the standard opponent pool.
#
# Usage:
#   scripts/eval_overnight.sh                  # auto-finds latest overnight run
#   scripts/eval_overnight.sh runs/six_max_...phase4f_dcfr_linear_overnight/
#
# Idempotent: skips checkpoints whose JSON already exists. Safe to re-run while
# overnight is still producing checkpoints.

set -e
cd /workspace/pokerbot

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
    RUN_DIR=$(ls -1dt runs/six_max_*phase4f_dcfr_linear_overnight*/ 2>/dev/null | head -1)
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: no overnight run directory found. Pass one as arg 1."
    exit 1
fi
RUN_DIR="${RUN_DIR%/}"   # strip trailing slash
CKPT_DIR="$RUN_DIR/checkpoints"

ABS=runs/abstraction_20260521_223018_retrofit/abstraction.pkl
STRUCT=configs/ignition_double_up_6max_turbo.yaml
HANDS="${HANDS:-5000}"
SEED="${SEED:-2026}"

# Standard opponent pool. Stable across all overnight checkpoints so each
# challenger gets the same yardstick.
OPPONENTS=(
    "vanilla-100=runs/six_max_20260523_205154_phase4f_shakedown/checkpoints/ckpt_iter_0100.pt"
    "vanilla-200=runs/six_max_20260523_224646_phase4f_overnight/checkpoints/ckpt_iter_0200.pt"
    "vanilla-400=runs/six_max_20260523_224646_phase4f_overnight/checkpoints/ckpt_iter_0400.pt"
    "dcfr-100=runs/six_max_20260524_005853_phase4f_dcfr_linear_shakedown/checkpoints/ckpt_iter_0100.pt"
    "dcfr-200=runs/six_max_20260524_005853_phase4f_dcfr_linear_shakedown/checkpoints/ckpt_iter_0200.pt"
    "random=__RANDOM__"
)

echo "==== Pool eval against overnight run ===="
echo "Run dir: $RUN_DIR"
echo "Hands per matchup: $HANDS"
echo "Seed: $SEED"
echo "Checkpoints to eval:"
ls "$CKPT_DIR"/ckpt_iter_*.pt 2>/dev/null | sort -V || { echo "  (none yet)"; exit 0; }
echo ""

mkdir -p evals

CKPT_LIST=$(ls "$CKPT_DIR"/ckpt_iter_*.pt 2>/dev/null | sort -V)
for ckpt in $CKPT_LIST; do
    iter=$(basename "$ckpt" .pt | sed 's/ckpt_iter_0*//')
    tag="dcfr-overnight-$iter"
    out="evals/${tag}_vs_pool_${HANDS}.json"

    if [ -f "$out" ]; then
        echo "SKIP  $tag  (already evaluated -> $out)"
        continue
    fi

    echo ""
    echo "==== Challenger: $tag ===="
    python -m scripts.eval_pool \
        --challenger "$tag=$ckpt" \
        --opponents "${OPPONENTS[@]}" \
        --abstraction "$ABS" \
        --structure "$STRUCT" \
        --hands "$HANDS" \
        --seed "$SEED" \
        --log-every 1000 \
        --output "$out"
done

echo ""
echo "==== Summary across overnight checkpoints ===="
python - <<'PYEOF'
import json, glob, os, re
files = sorted(glob.glob("evals/dcfr-overnight-*_vs_pool_*.json"),
               key=lambda p: int(re.search(r"overnight-(\d+)", p).group(1)))
if not files:
    print("(no eval JSONs found)")
    raise SystemExit(0)

# Print one row per challenger, one column per opponent.
opponents = None
rows = []
for f in files:
    d = json.load(open(f))
    tag = d["challenger"]["name"]
    by_opp = {r["opponent"]: (r["diff"], r["stderr"]) for r in d["results"]}
    if opponents is None:
        opponents = list(by_opp.keys())
    rows.append((tag, [by_opp.get(op, (None, None)) for op in opponents]))

# Header
col_w = 14
print(f"{'challenger':<22}", end="")
for op in opponents:
    print(f"{op:>{col_w}}", end="")
print()
print("-" * (22 + col_w * len(opponents)))

# Rows
for tag, results in rows:
    print(f"{tag:<22}", end="")
    for diff, stderr in results:
        if diff is None:
            print(f"{'—':>{col_w}}", end="")
        else:
            print(f"{diff:>+9.4f}±{stderr:.3f}"[:col_w].rjust(col_w), end="")
    print()
PYEOF
