# C3 RETRAIN — LAUNCH-READINESS REPORT (2026-06-10)

Status: **READY — awaiting operator launch.** Nothing has been launched.
Scope: retrain bundle build + pre-train gates per the operator-approved
spec. Bridge-session surfaces untouched except the two additive live-entry
convention gates noted in (c).

## a. H1 harvest gate — PASS

20,000 self-play SNGs (hero ckpt `b79e82dd…` in all six seats,
calibration-row configuration, per-seat RNG, hpl=5, master seed 2026,
8 tmux workers), 322,546 hand starts, **0 tainted games**.
Artifact: `data/training_dist_v1.json.gz` (sha256 `f87aaab2…`, manifest
entry; regenerable deterministically). Raw shards:
`evals/c3_harvest_20260610/`.

| level | hand starts | share |   | n_alive | share |
|---|---|---|---|---|---|
| L1 | 99,492 | 30.8% |   | 6-handed | 40.8% |
| L2 | 90,827 | 28.2% |   | 5-handed | 29.2% |
| L3 | 67,448 | 20.9% |   | 4-handed | 30.0% |
| L4 | 39,186 | 12.1% |   |          |       |
| L5 | 17,745 |  5.5% |   |          |       |
| L6–L9 | 7,848 | 2.43% |  |          |       |

**PRE-GATE H1: L6+ mass = 2.43% ≤ 10% threshold → PASS** (consistent
with the live envelope: live corpus has zero hands above L5; the v1
parametric sampler's vs-field games had 19.1% at L6+).

## b. Encoder 236 → 237 — GREEN, deployed model unaffected

`InfosetEncoder6Max(include_eff_bb=True)` appends ONE channel at the end
of the layout: `min(hero_stack, max alive opp stack) / big_blind`,
scaled 1/100 (`EFF_BB_SCALE`) — the floors' `_hero_eff_bb` quantity, and
the test asserts equality against `scripts/short_stack_floor_ab._hero_eff_bb`
itself across levels 1/3/5/8 and 4/5/6-handed states.
Dispatch: `TrainConfig6Max.encoder_eff_bb` (default False) is stamped in
checkpoint `config_dict`; `_load_solver` reads it with `saved.get(...,
False)`, so pre-C3 checkpoints reconstruct the 236-d encoder exactly.
Tests (all in `tests/test_c3_bundle.py`, 22/22 green): legacy-prefix
byte-identity (first 236 features bit-equal), old-ckpt round-trip (C3
keys stripped → 236-d, params bit-equal), new-ckpt round-trip (237-d),
deployed checkpoint loads + serves (see (c) replay gate).

## c. Ghost-ante — both replay gates GREEN (with one spec deviation)

**Deviation from spec, reported plainly:** the spec assumed the Option B
bug-match still lives in `invariant.openspiel_to_scraper_view`. It does
not — the bridge has been a clean real-ante 1:1 identity since `ae8307e`
(2026-06-05), and the library's `to_inner_game_string_for_state` already
posts per-seat native antes (empty/busted seats post NOTHING). The
"source fix" is therefore already canonical; what was missing was the
checkpoint-conditional convention machinery, which this bundle adds:

- `TrainConfig6Max.ante_convention` ("real" | "inflated_bb") stamped into
  every new checkpoint's config_dict.
- `src/nlhe/conventions.py`: `resolve_ante_convention` (stamp → sha256
  whitelist → unknown) + `require_live_servable`, wired into BOTH live
  entry points (`run_live_dryrun.py`, `run_logonly_resolver.py`).
  Unstamped checkpoints are NOT assumed "real" — the deployed model is
  whitelisted by sha256 (`b79e82dd…`); anything else unstamped or
  stamped `inflated_bb` is refused loudly. Per-checkpoint dispatch,
  never a global flip. The bug-match view was NOT re-introduced for old
  inflated checkpoints: it no longer exists in the bridge, no inflated
  checkpoint is deployable today, and resurrecting dead code on the
  live path contradicts the byte-identity requirement; serving such a
  checkpoint now requires an explicit operator decision instead of a
  silent wrong view.

Gates:
- **Old-model byte-identity:** `replay_make_decision_diff` over ALL NINE
  raw-record dry-run logs (standing rule: never a subset), deployed
  checkpoint, pre = clean HEAD `55cbda5` worktree, post = this bundle:
  **0 behavioral changes on every log.**
- **New-convention correctness:** `test_game_string_blinds_antes_all_levels`
  sweeps every schedule level × {4,5,6}-handed: every alive seat posts
  exactly the level ante, busted seats post nothing, SB/BB post raw
  (un-inflated) blinds, dead money = sb + bb + n_alive×ante. Plus the
  L1 min-raise = 50 gate. All green; the same checks run as pre-train
  gates G1/G2 in `train_c3.py`.

Related queued item NOT bundled (per spec scope): the heads-up SB/BB
convention bug (DECISIONS.md companion entry) — unreachable in the
deployment format (terminates at 3 alive) and in training (all harvest
rows n_alive ≥ 4, enforced by gate G3).

## d. Action space — DROPPED (spec's named drop-trigger conditions are real)

The ~2.2×BB open (9 → 10) was evaluated and **dropped**:
1. `N_DISCRETE_ACTIONS = len(DiscreteAction)` is a global that sizes
   every advantage/strategy net head (`networks6.py`), the HUNL solver,
   archetype weight arrays, and the parallel worker protocol. Widening
   the enum makes the deployed 9-dim checkpoint unloadable without an
   out-dim dispatch layer across all of them — the deployed model must
   remain live-servable throughout, so this is disqualifying on its own.
2. Floor masks: both `apply_short_stack_floor` implementations build
   masks sized to the enum on the live path (spec trigger: "floor
   masks").
3. Bridge translation: the click planner / `client_action_bb_mult`
   mapping and `discretize_legal_actions` are pot-fraction-based; a
   BB-multiple open is a new sizing family through the pseudo-harmonic
   translation (spec trigger: "bridge translation").
The 7-dim-fixture failures in `tests/test_subgame_leaf.py` (pre-existing
at HEAD) are a live demonstration of exactly this enum-coupling failure
mode. Action space stays 9.

## e. One-iteration benchmark + projection

`python -m scripts.train_c3 --iterations 1` (sequential, Contabo 12 vCPU,
quiesced): all four pre-train gates pass, **50.7 sec/iter**
(traversals=150, train_steps=200, 237-d nets, empirical sampling).
Projection for 2000 iters: **~28.2 h wall-clock** (~1.2 days).
Caveat per CLAUDE.md: Contabo is oversubscribed and per-iteration
timings vary up to ~10×; the deployed k200_real_ante run (same recipe
shape) took ~2 days to its iter-1500 stop, consistent with this number.
Checkpoint cadence 100 ⇒ 20 checkpoints (~full, not slim — the same
resume-safety tradeoff as the deployed run).

## f. Test suite — no new failures vs baseline

- New: `tests/test_c3_bundle.py` 22/22 green.
- Full suite (this tree): 962 passed / 33 failed / 27 errors / 5 skipped.
- Baseline (clean HEAD `55cbda5` worktree with the SAME artifacts
  symlinked): identical failing sets, verified by per-file diff
  (`test_solver6`, `test_dashboard*`, `test_eval_pool_ablation`,
  `test_ablation_decision_level`, `test_parallel_orchestrator`,
  `test_subgame_leaf` — all reproduce at HEAD; causes are stale 7-action
  fixture checkpoints, the `baseline_fork_A` golden predating later
  solver changes, and a `_StubAbstraction` missing `.streets`). These
  pre-existing failures are masked in routine runs by artifact-dependent
  skips and surface only when run with the full artifact set present.
- Replay gate (the live-path test that actually matters): green, see (c).

## g. Launch command (DO NOT RUN until operator approves)

```bash
tmux new-session -d -s c3_train \
  "cd ~/pokerbot && source .venv/bin/activate && \
   python -m scripts.train_c3 --config configs/c3_retrain_k200.yaml \
     --out runs/c3_retrain_v1 \
   > runs/c3_retrain_v1_train.log 2>&1"
```

Monitoring (start after launch, both read-only):

```bash
tmux new-window -t c3_train -n monitor \
  "cd ~/pokerbot && source .venv/bin/activate && \
   python scripts/monitor_k200_convergence.py \
     --run-dir runs/c3_retrain_v1 \
     --train-log runs/c3_retrain_v1_train.log"
tmux new-window -t c3_train -n tell-alert \
  "cd ~/pokerbot && source .venv/bin/activate && \
   python scripts/c3_premium_fold_alert.py \
     --csv runs/c3_retrain_v1/convergence.csv --watch 600"
```

Stop discipline: the iter-1500 pattern (DECISIONS.md "k200_real_ante
iter-1500 stop") — from ~iter 1000, run `scripts/slope_h2h_k200.py`
5,000-hand paired evals every ~250 iters (latest vs latest−500); stop
when the slope window includes zero twice consecutively and pin the peak
by H2H.
