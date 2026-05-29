#!/bin/bash
# Production training runner: launch scripts/train_6max.py in tmux with tee'd
# logs, preflight checks, and operator-friendly next-step instructions.
#
# Usage:
#   ./scripts/run_training.sh <config_path> [<extra_train_args>...]
#
# Example:
#   ./scripts/run_training.sh configs/six_max_phase4f_dcfr_overnight.yaml
#   ./scripts/run_training.sh configs/baseline_seq.yaml --resume runs/.../ckpt.pt
#
# What it does:
#   1. Validates the config file exists and is parseable YAML.
#   2. Preflight-imports the config to check abstraction_path,
#      tournament_structure_path, league_registry_path, and any mini_eval
#      anchors point at real files.
#   3. Creates runs/<tag>_<YYYYMMDD_HHMMSS>/, refuses to overwrite an
#      existing dir.
#   4. Writes launch_info.txt with timestamp, abs config path, git HEAD,
#      hostname (for forensics).
#   5. Launches a tmux session named train_<tag>_<short_hash>; inside it,
#      runs `python -u -m scripts.train_6max ...` tee'd to training.log.
#   6. Prints tail / attach / stop instructions.
#
# After launch, the operator can:
#   - Detach from any SSH session (script returns immediately).
#   - tail -f runs/.../training.log    (full training output)
#   - tail -f runs/.../lift.log         (if mini_eval_enabled — single-line lift records)
#   - tmux attach -t <session>           (live observe)
#   - tmux send-keys -t <session> C-c    (graceful stop — saves final checkpoint)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_path> [<extra_train_args>...]"
    exit 2
fi

CONFIG_PATH="$1"
shift
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config not found: $CONFIG_PATH"
    exit 1
fi
if [ ! -r "$CONFIG_PATH" ]; then
    echo "ERROR: config not readable: $CONFIG_PATH"
    exit 1
fi
CONFIG_ABS="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"

# ---------- preflight ----------
echo "Preflight: parsing config + checking referenced paths..."
PREFLIGHT_OUT="$(PYTHONPATH="$REPO_ROOT" .venv/bin/python - << PYEOF
import sys, os, yaml
cfg = yaml.safe_load(open("$CONFIG_PATH"))
tag = cfg.get("tag") or os.path.splitext(os.path.basename("$CONFIG_PATH"))[0]
missing = []
def need(path, label):
    if path is None:
        return
    if not os.path.exists(path):
        missing.append(f"{label}={path}")
need(cfg.get("abstraction_path"), "abstraction_path")
need(cfg.get("tournament_structure_path"), "tournament_structure_path")
need(cfg.get("league_registry_path"), "league_registry_path")
if cfg.get("mini_eval_enabled"):
    for spec in (cfg.get("mini_eval_anchors") or []):
        path = spec.split("=", 1)[1] if "=" in spec else spec
        need(path.strip(), "mini_eval_anchor")
    for spec in (cfg.get("mini_eval_shanky_rotation") or []):
        path = spec.split("=", 1)[1] if "=" in spec else spec
        need(path.strip(), "mini_eval_shanky")
if missing:
    print("MISSING:" + "|".join(missing), file=sys.stderr)
    sys.exit(1)
print(f"TAG={tag}")
PYEOF
)" || { echo "Preflight failed."; echo "$PREFLIGHT_OUT"; exit 1; }

TAG=$(echo "$PREFLIGHT_OUT" | sed -n 's/^TAG=//p')
[ -n "$TAG" ] || { echo "ERROR: could not extract tag from config"; exit 1; }

# ---------- run dir ----------
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="runs/${TAG}_${TS}"
if [ -e "$RUN_DIR" ]; then
    echo "ERROR: run dir already exists: $RUN_DIR"
    exit 1
fi
mkdir -p "$RUN_DIR"
RUN_DIR_ABS="$(cd "$RUN_DIR" && pwd)"

GIT_HEAD=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "(not a git repo)")
HOSTNAME_FULL=$(hostname -f 2>/dev/null || hostname)
cat > "$RUN_DIR/launch_info.txt" << INFOEOF
timestamp:   $(date -Iseconds)
config:      $CONFIG_ABS
run_dir:     $RUN_DIR_ABS
git_head:    $GIT_HEAD
hostname:    $HOSTNAME_FULL
extra_args:  ${EXTRA_ARGS[*]:-(none)}
INFOEOF
echo "Wrote $RUN_DIR/launch_info.txt"

# ---------- tmux launch ----------
SHORT_HASH=$(echo -n "$RUN_DIR" | sha256sum | cut -c1-6)
SESSION="train_${TAG}_${SHORT_HASH}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session $SESSION already exists; refusing to clobber"
    exit 1
fi

EXTRA_ARGS_QUOTED=""
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    for a in "${EXTRA_ARGS[@]}"; do
        EXTRA_ARGS_QUOTED+=" $(printf '%q' "$a")"
    done
fi

# SIGINT (Ctrl+C) is sent to the whole pipeline. Wrap the tee in a subshell
# that ignores INT/TERM so python's cleanup log lines + final metrics.json
# write reach training.log before the pipeline tears down. Python's own
# KeyboardInterrupt handler still fires normally — the trap only applies to
# the tee process.
tmux new-session -d -s "$SESSION" \
  "cd '$REPO_ROOT' && source .venv/bin/activate && \
   python -u -m scripts.train_6max --config '$CONFIG_ABS' --out '$RUN_DIR_ABS'$EXTRA_ARGS_QUOTED 2>&1 | { trap '' INT TERM; tee '$RUN_DIR_ABS/training.log'; }"

cat << ENDMSG

Run launched.

  session:       $SESSION
  run dir:       $RUN_DIR_ABS
  config:        $CONFIG_ABS
  git HEAD:      $GIT_HEAD

Operator commands:
  tail -f $RUN_DIR/training.log        # full training output
  tail -f $RUN_DIR/lift.log            # mini-eval lift lines (if mini_eval_enabled)
  tmux attach -t $SESSION              # live observe (Ctrl+B D to detach)
  tmux send-keys -t $SESSION C-c       # graceful stop (saves checkpoint, writes metrics.json)
  tmux list-sessions                   # see all live runs

ENDMSG
