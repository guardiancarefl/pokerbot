# RunPod environment replication — bootstrap checklist, benchmarks, verdict

**Date:** 2026-06-10. **Branch:** `runpod-env`. **Pod:** RunPod, Ubuntu 22.04
container, cgroup-v1-limited to 27.2 dedicated cores / 116 GiB RAM
(host shows 256 cores + 1 TB — ignore both, see §1).

This documents a full validation replication of the C3 training
environment against the Contabo production box, run while Contabo's
`runs/c3_retrain_v1` was live at iter ~200. Every gate the Contabo run
passed was reproduced here; benchmark numbers and the migrate/not-ready
verdict are at the end (§9).

---

## 1. Cgroup verification — DO THIS FIRST, before sizing anything

Container `nproc` / `free` report the HOST, not your slice. A prior pod
attempt OOM-killed a `make -j100` it sized from `nproc`. Get the real
allocation from the cgroup:

```bash
# cgroup v1 (this pod):
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us
#   quota/period = cores. Here: 2720000/100000 = 27.2 cores.
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
#   Here: 124999999488 = 116.4 GiB.

# cgroup v2 (other images):
cat /sys/fs/cgroup/cpu.max          # "<quota> <period>" or "max"
cat /sys/fs/cgroup/memory.max
```

Size all parallelism to the quota: `-j24` builds, 26 eval workers.
State expected memory before any heavy step (24 parallel g++ ≈ 25–35 GB
peak — safe in 116 GB; `-j100` is not).

Disk: `/` overlay is 20 GB local NVMe-fast; `/workspace` is a network
volume (MooseFS) — large but slow. Build and train on the local overlay;
use `/workspace` only for cold artifacts.

## 2. Base packages (root) — gotchas from pod bootstrap

As root, before creating the worker user:

```bash
apt-get update
apt-get install -y sudo tmux git openssh-client htop
```

- **sudo is NOT in the RunPod base image.** "User in sudo group" means
  nothing until the `sudo` package itself is installed. This pod was
  handed over with worker in group `sudo` but no `/usr/bin/sudo` — and
  `pkexec` is present but dead (no polkit daemon in containers). If you
  can't get root back, everything below still works user-space (§4).
- **nodejs 20 + libnode-dev conflict** (if installing Claude Code /
  any nodesource Node 20): Ubuntu's `libnode-dev`/`libnode72` collide
  with nodesource's `nodejs` package. Fix order:
  `apt-get remove -y libnode-dev libnode72 nodejs ; curl -fsSL
  https://deb.nodesource.com/setup_20.x | bash - ; apt-get install -y nodejs`.
- **tmux mouse:** `echo "set -g mouse on" >> ~/.tmux.conf` (per user).

## 3. worker user + sudoers + SSH

```bash
adduser --disabled-password --gecos "" worker
usermod -aG sudo worker
echo "worker ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/worker
chmod 440 /etc/sudoers.d/worker
# verify AS worker before relying on it:
su - worker -c "sudo -n true && echo SUDO OK"
```

SSH key exchange with Contabo (as worker):

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
# append ~/.ssh/id_ed25519.pub to quant@80.241.219.63:~/.ssh/authorized_keys
ssh -o BatchMode=yes quant@80.241.219.63 echo SSH OK   # must print SSH OK
```

## 4. Python 3.10 toolchain — user-space (no root needed)

Ubuntu 22.04's system `python3.10` is missing `python3.10-venv`,
`python3.10-dev` AND `ensurepip` — useless for venvs or building pyspiel
without apt. The fully user-space path (used on this pod, recommended
even with root because it pins exactly):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
export PATH=$HOME/.local/bin:$PATH
uv python install 3.10        # 3.10.20 incl. dev headers (~2 s)

git clone https://github.com/guardiancarefl/pokerbot.git ~/pokerbot
cd ~/pokerbot && git checkout track-policy
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match          # 13 s on this pod
# requirements.txt GAPS (present on Contabo, not pinned in the repo):
uv pip install requests pytest pytest-timeout cmake
```

`torch==2.12.0+cpu` requires the pytorch CPU index — plain PyPI fails.

## 5. Patched OpenSpiel 1.6.11 (universal_poker per-seat ante) — ~5 min

The recipe lives in-repo at `docs/patches/README.md` with both patches
(`openspiel_ante_v1.6.11.patch`, `acpc_ante_master.patch`). It is
reproducible. Two corrections found on this pod:

1. **`OPEN_SPIEL_BUILD_WITH_ACPC=ON` is MISSING from the recipe's
   export list.** Without it, universal_poker is silently not compiled
   and `pyspiel.load_game("universal_poker(...)")` fails with "Unknown
   game". It is OFF by default in CMakeLists.
2. `pybind11` is fetched by `install.sh` to the **repo root**
   (`./pybind11`), not under `open_spiel/` — don't re-clone it.

Working sequence (timed on this pod):

```bash
git clone --depth 1 --branch v1.6.11 \
    https://github.com/deepmind/open_spiel.git ~/open_spiel-patched
cd ~/open_spiel-patched
./install.sh || true          # deps fetch ~1 min; final apt step fails w/o root — fine
# ACPC fork is fetched by install.sh; if missing:
[[ -d open_spiel/games/universal_poker/acpc ]] || git clone -b master \
    --single-branch --depth 1 \
    https://github.com/jblespiau/project_acpc_server.git \
    open_spiel/games/universal_poker/acpc

git apply ~/pokerbot/docs/patches/openspiel_ante_v1.6.11.patch
( cd open_spiel/games/universal_poker/acpc/project_acpc_server &&
  git apply ~/pokerbot/docs/patches/acpc_ante_master.patch )
# verify BOTH landed:
grep -q '"ante"' open_spiel/games/universal_poker/universal_poker.cc
grep -q 'ante' open_spiel/games/universal_poker/acpc/project_acpc_server/game.h

mkdir -p build && cd build
export OPEN_SPIEL_BUILD_WITH_ACPC=ON       # <-- the recipe omission
export OPEN_SPIEL_BUILD_WITH_HANABI=OFF OPEN_SPIEL_BUILD_WITH_LIBTORCH=OFF \
       OPEN_SPIEL_BUILD_WITH_GO=OFF OPEN_SPIEL_BUILD_WITH_JULIA=OFF \
       OPEN_SPIEL_BUILD_WITH_ROSHAMBO=OFF OPEN_SPIEL_BUILD_WITH_XINXIN=OFF \
       OPEN_SPIEL_BUILD_WITH_LIBNOP=OFF OPEN_SPIEL_BUILD_WITH_ORTOOLS=OFF \
       OPEN_SPIEL_BUILD_WITH_RUST=OFF
cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE=Release ../open_spiel
make -j24 pyspiel              # 1m39s wall / 18m39s CPU on 27 cores

SP=~/pokerbot/.venv/lib/python3.10/site-packages
cp $SP/pyspiel.so $SP/pyspiel.so.bak.$(date +%Y-%m-%d)   # back up pip wheel's
cp python/pyspiel.so $SP/pyspiel.so
```

Verify (must pass before anything else):

```bash
cd ~/pokerbot && source .venv/bin/activate
python -c "
import pyspiel
g = pyspiel.load_game('universal_poker(betting=nolimit,numPlayers=6,'
  'numRounds=4,blind=15 25 0 0 0 0,ante=5 5 5 5 5 5,firstPlayer=3 1 1 1,'
  'numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,'
  'stack=1500 1500 1500 1500 1500 1500,bettingAbstraction=fullgame)')
s = g.new_initial_state()
while s.is_chance_node(): s.apply_action(s.legal_actions()[0])
assert min(a for a in s.legal_actions() if a > 1) == 50
print('ante patch verified: L1 min-raise-to = 50')"
```

## 6. Contabo artifacts (scp) + hash verification

```bash
cd ~/pokerbot
scp quant@80.241.219.63:~/pokerbot/data/training_dist_v1.json.gz data/
mkdir -p runs/k200_real_ante_20260605_225847_PRESERVED runs/abstraction_20260521_223018_retrofit
scp quant@80.241.219.63:~/pokerbot/runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt \
    runs/k200_real_ante_20260605_225847_PRESERVED/
scp quant@80.241.219.63:~/pokerbot/runs/abstraction_20260521_223018_retrofit/abstraction.pkl \
    runs/abstraction_20260521_223018_retrofit/
scp -r quant@80.241.219.63:~/pokerbot/data/shanky_profiles data/   # for SNG evals
sha256sum runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt \
          runs/abstraction_20260521_223018_retrofit/abstraction.pkl
# MUST equal (STATUS.md "Validation hashes"):
#   b79e82dd0ce9e78e4eb666b7379df953dadbf2a6e026c6bd4b6eec695e9b1b11  ckpt_iter_1500.pt
#   0fc20800dc7ce89ea950c975decbd530ac6c6f3c164105e8ab99d24359de4c8e  abstraction.pkl
```

Both verified on this pod. All transfers <2 s each.

## 7. Validation gates — all reproduced (2026-06-10)

| Gate | Result on this pod | Contabo reference |
|---|---|---|
| a. Full pytest suite | **23F / 961P / 43S / 0E** in 4m16s | 33F / 962P / 5S / 27E |
| b. C3 pre-train gates G1–G4 | **all 4 PASSED** (10 s, `--iterations 0`) | all 4 PASSED |
| c. Deployed-model SNG smoke | **50 games, 0 tainted/0 exceptions**, net/game −0.160 ± 0.140 vs killphilmtt | −0.174 gate metric |

Test-suite difference itemization (every diff classified):

- **23 pod failures = `tests/test_solver6.py` (18) + `tests/test_dashboard.py`
  (5).** All share ONE root cause: `_StubAbstraction` missing `.streets` —
  explicitly listed in `docs/C3_LAUNCH_READINESS.md` §f as a pre-existing
  baseline failure that reproduces on clean HEAD. **Genuine: 0 new failures.**
- **Contabo's extra 10 failures + 27 errors** (`test_eval_pool_ablation`,
  `test_ablation_decision_level`, `test_parallel_orchestrator`,
  `test_subgame_leaf`, dashboard goldens) are **artifact-dependent**: they
  need stale 7-action fixture checkpoints, the `baseline_fork_A` golden,
  and other off-repo artifacts present on Contabo. On this pod they
  **skip** with explicit "solver artifacts not present" reasons — that is
  the 5→43 skip delta. Classification: artifact-absence, not regressions.
- Suite ran with `pytest 9.0.3` + `pytest-timeout`; no version-driven diffs
  observed.

Cross-machine determinism spot-check: train_c3 iters 1–3 on this pod
reproduce Contabo's `c3_retrain_v1` log **bit-identically** (adv
0.6320/1.3293/0.7345, identical buffer counts) — same code, same seed,
different hardware. The environment is faithful.

## 8. Training observability bundle (`scripts/c3_dashboard.py`)

Built on this pod, ships with any migration; **portable**: can watch a
local run dir OR Contabo's live run over scp (`--run-dir
quant@80.241.219.63:~/pokerbot/runs/c3_retrain_v1`).

Per new `ckpt_iter_NNNN.pt` it appends a CSV row: iter, adv/strat loss
(parsed from the train log), AA/KK fold% at 60bb/10bb, and **two
fixed-anchor 5000-pair paired-CRN evals** — vs the deployed ckpt_1500
(sha-verified, refused on mismatch) and vs the run's own iter-100
snapshot — each reported margin ± stderr. Fixed seed families per anchor
→ every checkpoint plays identical deals → CRN-clean trajectories.
Renders the latest 10 rows to a `.txt` for `watch cat`, atomically.
**Do not read adjacent-checkpoint deltas as signal** (noise σ ≈ 1–2.5
per DECISIONS); the table header repeats this.

- Anchor evals fork-parallel, cgroup-aware: **~1300 pairs/s on 26
  workers — a 5000-pair anchor eval takes ~4 s** on this pod.
- Acceptance check (passed 2026-06-10): `--demo-calibration` plays the
  deployed ckpt vs ITSELF → margin **−0.00000 ± 0.00000** over 1000
  pairs → PASS. Full remote demo against Contabo's live run processed
  iters 100+200: vs-deployed −0.031±0.003 → −0.027±0.003; vs-iter-100
  +0.000 → +0.004±0.003; premium-fold 27%→17.5%.

Sibling-session launch (never in the trainer's session; read-only):

```bash
tmux new-session -d -s c3_dash \
  "cd ~/pokerbot && source .venv/bin/activate && \
   PYTHONUNBUFFERED=1 python scripts/c3_dashboard.py \
     --run-dir runs/c3_retrain_v1 \
     --train-log runs/c3_retrain_v1_train.log \
     --watch 300 > runs/c3_dashboard.log 2>&1"
# operator view:  watch cat runs/c3_retrain_v1/dashboard.txt
```

**UNBUFFERED RULE: every long-running launch command — trainer,
monitor, dashboard — carries `PYTHONUNBUFFERED=1` so `tail -f` is live
per-iteration.** (The trainer's own log format flushes per-iter only
when unbuffered.)

## 9. Benchmarks and verdict

All timings on this pod, single process unless stated, nothing else
running (torch pinned to 1 thread by `solver6.py`).

| Step | This pod | Contabo reference |
|---|---|---|
| repo clone + venv + deps | ~15 s | — |
| pyspiel `make -j24` | **1m39s** | 30–60 min (`-j8`, 12 vCPU) |
| full pytest suite | 4m16s | — |
| train_c3 iters 1–3 (cold) | 22.2 / 13.0 / 11.4 s | 47.1 / 27.3 / 23.5 s (same run, bit-identical) |
| **train_c3 steady state** (resumed Contabo's c3 iter-200 ckpt, 6 iters) | **21.8 s/iter mean** (14.9–29.8; 130.5 s total) | 50.7 benchmarked; 54–73 observed at iter 195–200 under load |
| SNG harness, 200 games | 38 s play = **5.3 games/s ≈ 0.19 s/game** | ~3 s/game/process |
| 5000-pair anchor eval (26 workers) | ~4 s | — |

Steady-state caveat, stated honestly: Contabo's periodic checkpoints are
slim (no reservoir buffers), so the resumed bench rebuilt buffers while
using true iter-200 net weights. Traversal cost — the dominant,
weight-driven term — is steady-state-faithful; train-step cost is
buffer-size-insensitive (fixed 200 steps × batch 64). Cold iters 1–3
corroborate: pod/Contabo ratio 0.47–0.49 on identical workloads.

### Parallel-scaling probe (assessed only, NOT built — per instruction)

train_c3 runs `DeepCFR6MaxSolver.train()` — the same loop the G-worker
fork framework (`src/nlhe/parallel/`, used by `parallel_train` /
`train_6max`, ~4.5x at G=10 on Contabo's 12 oversubscribed cores)
already wraps. The ONLY blocker is the explicit guard at
`src/nlhe/solver6.py:302`: `encoder_eff_bb` / `empirical_dist_path` are
not yet in `WorkerInput`. Wiring needed:

1. Two fields on `WorkerInput` (`encoder_eff_bb`, `empirical_dist_path`).
2. Worker side: construct encoder with the eff-BB channel; add the
   empirical-rows branch mirroring `solver6.py:851` — the per-traversal
   `STACK_SAMPLE_SALT` RNG it needs is already replicated in
   `worker.py:244`, so determinism carries over unchanged. Ship rows via
   fork CoW (load in parent pre-fork), NOT per-worker gz reloads.
3. Lift the guard; rerun the framework's bit-identity gate vs a short
   sequential C3 run.

Effort estimate: **0.5–1 day including the bit-identity validation.**
Speedup at G=24–26 on 27 dedicated cores: traversals are ~95% of the
21.8 s iter; with the framework's measured ~55–60% parallel efficiency,
expect **6–12x → roughly 2–4 s/iter** (Amdahl-capped by the serial
train/merge step). Even the conservative end halves the 12 h projection
below. Not required for the verdict — single-process already passes.

### Verdict against the pre-committed rule

Rule: all gates green AND single-process < ~25 s/iter → MIGRATE-CAPABLE.

- Gates: **all green** (§7; zero genuine new failures, all diffs
  classified artifact-absence).
- Single-process steady state: **21.8 s/iter < 25**.

→ **MIGRATE-CAPABLE.** Operator decides. Migration = **fresh start from
iter 0** — the empirical sampler/seed stream makes a mid-run handoff
meaningless; Contabo's invested hours are discarded.

Break-even arithmetic (numbers as of 2026-06-10 ~15:30 UTC, Contabo at
iter 200/2000, ~5 h in):

- Contabo remaining: 1800 iters × ~63 s/iter (recent 54–73 under load)
  ≈ **31.5 h** (≈ 25.4 h in the unloaded 50.7 s/iter best case).
- This pod from iter 0: 2000 × 21.8 s ≈ **12.1 h**.
- Net: pod finishes **~19 h sooner** (best case for Contabo: ~13 h
  sooner), already net of throwing away Contabo's 5 invested hours.
  Break-even would require pod steady-state > ~45.7 s/iter — 2.1× worse
  than measured.
- Risks priced in: RunPod community-cloud preemption (mitigated:
  checkpoint resume is bit-identical, verified here on a cross-machine
  resume), and pod $/h vs Contabo sunk cost (operator's call).

### Exact relaunch command for THIS pod (if operator migrates)

```bash
cd ~/pokerbot && git checkout track-policy   # train from the production branch
tmux new-session -d -s c3_train \
  "cd ~/pokerbot && source .venv/bin/activate && \
   PYTHONUNBUFFERED=1 python -m scripts.train_c3 \
     --config configs/c3_retrain_k200.yaml \
     --out runs/c3_retrain_v2_runpod \
   > runs/c3_retrain_v2_runpod_train.log 2>&1"
# then the dashboard as a SIBLING session (§8) pointed at runs/c3_retrain_v2_runpod
# and Contabo's c3_retrain_v1 must be stopped by the OPERATOR, not from here.
```

This is a fresh run from iter 0 under the same config/seed; it discards
Contabo's progress (~5 h at decision time — redo the §9 arithmetic with
Contabo's current iter count before pulling the trigger).

## 10. Deviations from the brief, recorded

1. **"Passwordless sudo" was not true on this pod** — `sudo` binary
   absent, `su` password-locked, polkit dead. Worked around entirely
   user-space (uv toolchain, §4). Bootstrap checklist §2 prevents this
   on the next pod.
2. `requirements.txt` is incomplete vs Contabo's venv (`requests`,
   `pytest`, `pytest-timeout` missing). Documented in §4 rather than
   editing requirements on this branch.
3. The in-repo OpenSpiel recipe omits `OPEN_SPIEL_BUILD_WITH_ACPC=ON`
   (§5 correction 1) — without it the build "succeeds" but
   universal_poker is absent. Cost one extra 1m40s rebuild here.
4. Pod pytest baseline differs from Contabo's by design (artifact
   absence): 43 skips / 0 errors here vs 5 skips / 27 errors there.
   Itemized in §7; zero genuine regressions.

---

## 11. New pod 2026-06-13 — H4 generate-only rebuild + field probe

Fresh pod (the 2026-06-10 one above is dead). Connection (TCP, SCP/SFTP-capable):
```
ssh root@213.192.2.91 -p 40013 -i ~/.ssh/id_ed25519
```
Driven over ssh from the Contabo session — **no Claude Code / Node on the pod**
(skips the §2 libnode conflict entirely). Bootstrap = the full §1–§6 path
(fresh container), then the bit-identity smoke (§5 here) as the hard gate.

### THE GPU IS UNUSED — do not chase it
This codebase is CPU-only: `torch==2.12.0+cpu`, `torch.set_num_threads(1)`
(`solver6.py:76`), parallelism via CPU fork-workers. **Bit-identity to Contabo
REQUIRES the CPU build**, where `torch.cuda.is_available()` is `False`.
Installing a CUDA torch to "use the 3090" swaps in CUDA kernels and GUARANTEES
the smoke fails (CUDA ≠ CPU rounding). The pod's win is ~2.4× dedicated cores,
NOT the GPU. Correct check: `cuda == False`.

### Env that MUST match Contabo for bit-identity (verified 2026-06-13)
| | Contabo (reference) | pod must equal |
|---|---|---|
| Python | 3.10.20 (deadsnakes) | 3.10.20 (`uv python install 3.10`) |
| torch | 2.12.0+cpu, cuda=False | 2.12.0+cpu, cuda=False |
| pyspiel | patched 1.6.11 (per-seat ante) | §5 build w/ `OPEN_SPIEL_BUILD_WITH_ACPC=ON` |
| threads | `set_num_threads(1)` | (in code — automatic) |

### Code/config now ON origin/track-policy (clean `git pull`, no scp)
Commit `ae3d7ce` (populate_only fix) + `34ac603` (docs) pushed 2026-06-13.
The pod's `git checkout track-policy && git pull` gets: `solver6.py`,
`parallel/orchestrator.py`, `scripts/h4_buffer_rebuild.py`,
`configs/h4_probe.yaml`, `configs/league/registry_h4_field.json`.

### Files NOT in git — push from Contabo (tiny, <2s each)
```bash
# run FROM the Contabo session (has the pod key); after the pod clone+venv exist:
POD="-P 40013 -i ~/.ssh/id_ed25519 root@213.192.2.91"
scp $POD ... # binaries:
#   runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt  (sha b79e82dd…)
#   runs/abstraction_20260521_223018_retrofit/abstraction.pkl        (sha 0fc20800…)
# untracked scripts the monitor needs:
#   scripts/h4_progress_monitor.py  scripts/h4_dashboard.py  scripts/field_battery.py
```

### THE BIT-IDENTITY SMOKE (hard gate — hold all training until it matches)
Identical command both machines, SEQUENTIAL real training from the champion
(the wrapper calls `solver.train()`; `--parallel-groups` is ignored):
```bash
cd ~/pokerbot && source .venv/bin/activate
PYTHONUNBUFFERED=1 python -m scripts.h4_buffer_rebuild --benchmark 2>&1 | grep -E "iter 150"
```
**Contabo reference (champion candC recipe, seed 2026 — run-to-run
deterministic, confirmed 2x):**
```
iter 1501  trav=0  adv=0.5118  strat=0.7703  bufs=(1361, 0, 0, 0, 0, 0)  sbuf=5774
iter 1502  trav=1  adv=0.5068  strat=0.8530  bufs=(1361, 1863, 0, 0, 0, 0)  sbuf=13354
```
Pod must match all four losses to every printed digit + the buffer counts.
EXACT ⇒ faithful, proceed. Any deviation (esp. cuda=True) ⇒ STOP, diagnose —
a non-faithful pod silently produces a wrong model. (Timing is load-dependent,
NOT part of the gate.)

### Post-smoke tasking (held for operator go)
1. generate-only rebuild: `h4_buffer_rebuild.py --populate-only --iters 400`
   (parallel via train path for speed; self-anchor=0 by construction).
2. field-mix probe: `train_6max --config configs/h4_probe.yaml --resume <full ckpt>`.
3. monitor + dashboard (RUNPOD §8 pattern) on the pod run dir.
Contabo fallback generate-only rebuild stays ALIVE until the pod is proven.

## 12. Snapshot + restore (skip the OpenSpiel build on re-rent) — 2026-06-13

The proven-faithful H4 env is captured so a re-rent skips the ~5-min OpenSpiel
build + pip (NOT the ~2-min bit-identity smoke — that stays, every re-rent).

**Captured (verified restorable, pyspiel.so sha `acae193b…`):**
- Contabo (guaranteed): `~/pokerbot/pod_snapshot/venv_h4faithful.tgz` (305 MB) +
  `POD_SNAPSHOT_MANIFEST.txt`.
- Pod `/workspace/pod_snapshot/` (persistent network volume, eu-cz-1; fast
  same-region re-rent).

**Restore on a fresh pod (after §1–§3 system + uv + clone):**
```bash
cd ~/pokerbot
# from Contabo (always works):
scp -P <port> -i ~/.ssh/id_ed25519 quant@<contabo>:~/pokerbot/pod_snapshot/venv_h4faithful.tgz /tmp/
#   OR from /workspace if same region:  cp /workspace/pod_snapshot/venv_h4faithful.tgz /tmp/
tar xzf /tmp/venv_h4faithful.tgz -C ~/pokerbot            # restores .venv incl. patched pyspiel.so
source .venv/bin/activate
sha256sum .venv/lib/python3.10/site-packages/pyspiel.so   # MUST be acae193b1e7715fc6282daebcc5fc0be6e97437ac2ffbe82a46d3412d7002a2a
# THEN re-run the §5/§11 bit-identity smoke (non-negotiable) before any training.
```
Restore to the IDENTICAL path (`~/pokerbot/.venv`) — the venv has absolute paths.
The smoke is the gate: snapshot skips the BUILD, never the faithfulness check.
