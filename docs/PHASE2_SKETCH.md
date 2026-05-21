# Phase 2 sketch: heads-up no-limit hold'em

This is a forward-looking sketch, not a commitment. Wrote it during the Phase 1
training run in Session 2 so Session 3 has a starting point. Firm up the details
at the start of Session 3 once Phase 1 is officially closed.

## Goal

Train a Deep CFR policy on heads-up no-limit Texas hold'em (HUNL), evaluate
it against Slumbot via API, get a measurable bb/100 win rate.

This is the bridge between Leduc (toy, tabular, Phase 1) and 6-max NLHE SNG
(real target, Phase 4). HUNL is real poker but only 2 players, so the game
tree is smaller and we can use it to validate the training pipeline scales
from toy to real before adding 6 players.

## What carries over from Phase 1

- `src/leduc/config.py` pattern → `src/nlhe/config.py`. Same dataclass + YAML loader.
- `src/leduc/checkpoint.py` → move to `src/common/checkpoint.py` (shared).
- `src/leduc/solver.py` structure → same skeleton, different solver inside.
- `scripts/train_*.py` pattern → unchanged.
- `tests/` pattern → unchanged.
- Run directory format → unchanged.

## What changes

### Game
Switch from `pyspiel.load_game("leduc_poker")` to HUNL via OpenSpiel's
universal_poker, or use a dedicated HUNL loader if available. Information
state encoding becomes much larger: hole cards, board across 4 streets,
betting history, stack sizes.

### Game size
Leduc: ~3,000 information sets. HUNL without abstraction: ~10^17. Cannot
enumerate. Drives the next two changes.

### Card abstraction (NEW MODULE)
Required. Group hands by strategic equivalence so the network sees ~10^4
buckets instead of 10^17 raw hands. Standard approach: Earth Mover's Distance
(EMD) clustering on equity distributions. Target ~200 buckets per street per
ARCHITECTURE.md.

New file: `src/nlhe/abstraction.py`. This is a non-trivial chunk of work on
its own — clustering equity distributions correctly needs its own design
and validation.

### Action abstraction (NEW MODULE)
Required. HUNL has continuous bet sizes. Discretize to:
`{check/fold, call, 0.33pot, 0.66pot, 1pot, 2pot, all-in}` per ARCHITECTURE.md.
Also need action translation: when an opponent bets a size not in our discrete
set (e.g., 0.5 pot), map it to the nearest tree action.

New file: `src/nlhe/actions.py`.

### Exploitability evaluation
Tabular exploitability via full game tree traversal becomes infeasible. Options:
- Local Best Response (LBR) — fast, approximate, standard in literature.
- Head-to-head vs Slumbot via API — primary signal.
- vs held-out archetypes — secondary (becomes available in Phase 3).

For Phase 2: head-to-head vs Slumbot is the primary metric. LBR as secondary.

### Compute
Leduc converged on CPU. HUNL won't. Expected training time on Contabo CPU for
a real coarse-abstraction run: days to weeks. Phase 2 is when we first rent
cloud GPU (Vast.ai / RunPod / Vultr GPU).

## Genuinely new (not just bigger versions)

1. Card abstraction — research-flavored subtask with its own validation.
2. Slumbot API integration — external service, rate limits, hand history parsing.
3. Possibly custom Deep CFR — OpenSpiel's reference may not scale; profile first.
4. First real GPU dependency — PyTorch CUDA wheel, CUDA driver matching.
5. Resumable training — runs are long enough that "lose progress on kill" hurts.

## Suggested sub-phase structure

- **2a. Game + abstraction.** Load HUNL, build card abstraction, build action
  abstraction, validate via equity-histogram sanity checks. No training.
- **2b. Solver works on tiny HUNL.** Tiny stacks (20bb max), coarse
  abstraction (~50 buckets), train enough to verify the solver doesn't
  explode. Leduc-but-real.
- **2c. Slumbot eval harness.** API client, head-to-head runner, bb/100
  calculator. Test with a dummy random policy before plugging in our bot.
- **2d. Real training run.** Full coarse-abstraction HUNL, real iteration
  counts. Multi-day GPU run. Result: a bot that plays Slumbot at some bb/100.
- **2e. Iterate.** Better abstraction, more iterations, deeper networks,
  until bb/100 vs Slumbot stops improving.

## Decisions to make before Session 3 starts

- GPU provider — Vast.ai, RunPod, or Vultr GPU. Need pricing comparison.
- Card abstraction approach — EMD clustering vs simpler (KMeans on equity, OCHS).
  ARCHITECTURE.md says EMD; confirm before committing.
- Fork OpenSpiel's Deep CFR vs use as-is — depends on profiling we haven't done.
- Reference bot for the Slumbot harness — maybe an OpenSpiel CFR-trained
  HUNL policy from the literature, as a known baseline to validate the
  eval pipeline before plugging in our own bot.

## What this does NOT include

- The 42 bought-bot profiles. Those come in Phase 3.
- The within-match exploitation layer (Layer 4). That's Phase 6.
- ICM value function. That's Phase 4 (when we add 6 players and the SNG
  payout structure).
- 6-max anything. That's also Phase 4.

Phase 2's job is: get a working HUNL bot. Just one more building block.
