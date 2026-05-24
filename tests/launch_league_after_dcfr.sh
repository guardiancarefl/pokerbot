#!/bin/bash
# Wait for the DCFR overnight training to finish, then build the league
# registry from its checkpoints and launch the league overnight in a
# new tmux session.
#
# Designed to be fire-and-forget. Typical usage:
#
#   tmux new -s league_launcher
#   ./scripts/launch_league_after_dcfr.sh
#   # Ctrl+B D to detach
#
# Polls every $POLL_INTERVAL seconds. Exits cleanly if DCFR has ALREADY
# finished (so safe to invoke at any time). Refuses to launch if the
# league tmux session already exists, if GPU is still busy, or if the
# registry validation fails.
#
# Configurable via environment variables (defaults shown):
#   DCFR_TMUX=dcfr_overnight        # tmux session running DCFR
#   LEAGUE_TMUX=league_overnight    # tmux session for the launched league
#   POLL_INTERVAL=60                # seconds between checks while waiting
#   REGISTRY_PATH=runs/league/registry.json
#   LEAGUE_CONFIG=configs/six_max_phase4f_dcfr_league_overnight.yaml
#   LEAGUE_LOG=/tmp/league_overnight.log
#   LAUNCHER_LOG=/tmp/league_launcher.log
#
# Override example:
#   DCFR_TMUX=my_session ./scripts/launch_league_after_dcfr.sh
#
# Flags:
#   --no-launch   Build the registry but do not launch the league.
#                 Useful for verifying the registry build path manually
#                 before committing to the chain.

set -euo pipefail

DCFR_TMUX="${DCFR_TMUX:-dcfr_overnight}"
LEAGUE_TMUX="${LEAGUE_TMUX:-league_overnight}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
REGISTRY_PATH="${REGISTRY_PATH:-runs/league/registry.json}"
LEAGUE_CONFIG="${LEAGUE_CONFIG:-configs/six_max_phase4f_dcfr_league_overnight.yaml}"
LEAGUE_LOG="${LEAGUE_LOG:-/tmp/league_overnight.log}"
LAUNCHER_LOG="${LAUNCHER_LOG:-/tmp/league_launcher.log}"

NO_LAUNCH=0
for arg in "$@"; do
    case "$arg" in
        --no-launch) NO_LAUNCH=1 ;;
        --help|-h)
            sed -n '2,/^set -e/p' "$0" | sed -e 's/^# \{0,1\}//' -e '$d'
            exit 0
            ;;
        *) echo "unknown flag: $arg"; exit 2 ;;
    esac
done

log() {
    local msg="[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
    echo "$msg" | tee -a "$LAUNCHER_LOG"
}

die() {
    log "ERROR: $*"
    exit 1
}

cd /workspace/pokerbot

log "==== league launcher starting ===="
log "dcfr_tmux=$DCFR_TMUX  league_tmux=$LEAGUE_TMUX  no_launch=$NO_LAUNCH"

# Refuse if league tmux session already exists.
if tmux has-session -t "$LEAGUE_TMUX" 2>/dev/null; then
    die "tmux session '$LEAGUE_TMUX' already exists. Kill it (tmux kill-session -t $LEAGUE_TMUX) or override LEAGUE_TMUX."
fi

# Find the most-recent DCFR overnight run dir.
DCFR_RUN=$(ls -1dt runs/six_max_*phase4f_dcfr_linear_overnight*/ 2>/dev/null | head -1)
DCFR_RUN="${DCFR_RUN%/}"
[ -n "$DCFR_RUN" ] || die "no DCFR overnight run dir found under runs/"
log "DCFR run dir: $DCFR_RUN"

# Read the iteration count from the run's config.json so we don't hardcode 3400.
DCFR_CONFIG_JSON="$DCFR_RUN/config.json"
[ -f "$DCFR_CONFIG_JSON" ] || die "DCFR config.json not found at $DCFR_CONFIG_JSON"
TARGET_ITER=$(python3 -c "import json; print(json.load(open('$DCFR_CONFIG_JSON'))['n_iterations'])")
TARGET_CKPT="$DCFR_RUN/checkpoints/$(printf 'ckpt_iter_%04d.pt' "$TARGET_ITER")"
log "target iter=$TARGET_ITER  final ckpt=$TARGET_CKPT"

# Poll until BOTH:
#   (a) the target checkpoint file exists
#   (b) the DCFR tmux session has exited
# Either alone isn't sufficient: ckpt could be mid-write; tmux gone with no
# final ckpt means DCFR crashed.
log "Polling every ${POLL_INTERVAL}s..."
LAST_STATUS_PRINT=0
while true; do
    have_ckpt=0
    [ -f "$TARGET_CKPT" ] && have_ckpt=1

    tmux_alive=0
    tmux has-session -t "$DCFR_TMUX" 2>/dev/null && tmux_alive=1

    if [ $have_ckpt -eq 1 ] && [ $tmux_alive -eq 0 ]; then
        log "DCFR overnight finished. Final ckpt present, tmux session gone."
        break
    fi

    if [ $have_ckpt -eq 0 ] && [ $tmux_alive -eq 0 ]; then
        die "DCFR tmux session is gone but final ckpt does not exist. Run probably crashed. Inspect $DCFR_RUN/checkpoints/ manually before launching league."
    fi

    # Status line every ~5 polls (so log isn't spammed).
    NOW_S=$(date +%s)
    if [ $((NOW_S - LAST_STATUS_PRINT)) -ge $((POLL_INTERVAL * 5)) ]; then
        LATEST=$(ls -1 "$DCFR_RUN/checkpoints/" 2>/dev/null | grep -E '^ckpt_iter_[0-9]+\.pt$' | sort -V | tail -1 || echo "(none)")
        log "  waiting: ckpt=$have_ckpt tmux=$tmux_alive latest_ckpt=$LATEST"
        LAST_STATUS_PRINT=$NOW_S
    fi
    sleep "$POLL_INTERVAL"
done

# GPU sanity: should be free now that DCFR is done.
log "Checking GPU is free..."
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n' || true)
    if [ -n "$GPU_PIDS" ]; then
        die "GPU still has compute processes: $GPU_PIDS. Aborting to avoid contention."
    fi
    log "  GPU clear."
else
    log "  (nvidia-smi not available; skipping GPU check)"
fi

# Build the league registry from the completed overnight + shakedown runs.
log "Building league registry at $REGISTRY_PATH..."
python -m scripts.build_league_registry \
    --run-dir "$DCFR_RUN" \
    --name-prefix dcfr-overnight \
    --tags dcfr overnight \
    --output "$REGISTRY_PATH" 2>&1 | tee -a "$LAUNCHER_LOG"

SHAKEDOWN_RUN=$(ls -1dt runs/six_max_*phase4f_dcfr_linear_shakedown*/ 2>/dev/null | head -1)
SHAKEDOWN_RUN="${SHAKEDOWN_RUN%/}"
if [ -n "$SHAKEDOWN_RUN" ] && [ -d "$SHAKEDOWN_RUN/checkpoints" ]; then
    log "Also registering shakedown checkpoints from $SHAKEDOWN_RUN..."
    python -m scripts.build_league_registry \
        --run-dir "$SHAKEDOWN_RUN" \
        --name-prefix dcfr-shake \
        --tags dcfr shakedown \
        --output "$REGISTRY_PATH" 2>&1 | tee -a "$LAUNCHER_LOG"
else
    log "  (no shakedown run found; skipping)"
fi

# Validate registry: at least one entry tagged 'dcfr', and roughly the
# expected number of overnight checkpoints (target_iter / 200).
log "Validating registry..."
EXPECTED_OVERNIGHT=$((TARGET_ITER / 200))
python3 - <<PYEOF 2>&1 | tee -a "$LAUNCHER_LOG"
import sys
from src.nlhe.checkpoint_registry import CheckpointRegistry
r = CheckpointRegistry.load("$REGISTRY_PATH")
overnight = [n for n in r.names() if n.startswith("dcfr-overnight-")]
total = len(r)
dcfr_tagged = sum(1 for n in r.names() if "dcfr" in r.get(n).tags)
print(f"  total entries: {total}")
print(f"  dcfr-tagged:   {dcfr_tagged}")
print(f"  dcfr-overnight:{len(overnight)}  (expected >= $EXPECTED_OVERNIGHT)")
if dcfr_tagged == 0:
    print("ERROR: no dcfr-tagged entries; league_tag_filter=[dcfr] would yield empty pool", file=sys.stderr)
    sys.exit(1)
if len(overnight) < $EXPECTED_OVERNIGHT:
    print("ERROR: too few overnight checkpoints", file=sys.stderr)
    sys.exit(1)
PYEOF

if [ $NO_LAUNCH -eq 1 ]; then
    log "--no-launch flag set. Registry is built and validated; not launching league."
    log "To launch manually: tmux new -d -s $LEAGUE_TMUX 'python -m scripts.train_6max --config $LEAGUE_CONFIG 2>&1 | tee $LEAGUE_LOG'"
    log "==== league launcher done (no-launch mode) ===="
    exit 0
fi

# Launch league in fresh tmux session.
log "Launching league overnight in tmux session '$LEAGUE_TMUX'..."
log "  config: $LEAGUE_CONFIG"
log "  log:    $LEAGUE_LOG"
tmux new-session -d -s "$LEAGUE_TMUX" \
    "cd /workspace/pokerbot && python -m scripts.train_6max --config $LEAGUE_CONFIG 2>&1 | tee $LEAGUE_LOG"

# Sanity check: session should be alive a few seconds after launch.
sleep 5
if ! tmux has-session -t "$LEAGUE_TMUX" 2>/dev/null; then
    die "League tmux session died within 5s of launch. Inspect $LEAGUE_LOG."
fi

log "League overnight launched."
log "  tmux session: $LEAGUE_TMUX  (attach with: tmux attach -t $LEAGUE_TMUX)"
log "  training log: $LEAGUE_LOG   (tail with: tail -f $LEAGUE_LOG)"
log "First lines of training log:"
sleep 2
head -10 "$LEAGUE_LOG" 2>/dev/null | sed 's/^/    /' | tee -a "$LAUNCHER_LOG" || log "  (log not yet populated)"

log "==== league launcher done ===="
