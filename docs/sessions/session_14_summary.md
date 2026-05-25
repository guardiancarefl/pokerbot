# Session 14 — 2026-05-25

Focus: B1c sub-step 2 (the depth-limited subgame LEAF evaluator), implemented
across Stages A–E, plus a load-bearing ICM busted-seat fix that surfaced
mid-implementation. Three separate "load-bearing number in the design doc was
wrong" findings — the meta-lesson of the session.

## What was done

- **Stage A — `src/nlhe/fast_view.py`** (`994b587`). Sorted-legal-actions fast path
  for the rollout hot loop: `min_bet`/`max_bet`/fold/call via bisect on the
  ascending `legal_actions()` (no 9,803-element `set`/list-comp/min/max), one
  shared `legal_actions()` call for view+discretize, field-identical to canonical
  `_build_view_6max` + `discretize_legal_actions`. 10 tests (50 random rollouts +
  edge cases incl. facing-all-in chip-0 alias; chance/terminal raise). Benchmark:
  **0.046 ms/step vs 0.28 ms canonical = ~6×**; end-to-end loop 0.084 ms/step.
- **Stage B — `src/nlhe/subgame_leaf.py` scaffold** (`cbc047a`). Q9 contracts:
  `LeafEvalMode`, `BlueprintProvider` (`@runtime_checkable`, verified `isinstance`
  against a real loaded `DeepCFR6MaxSolver`), `LeafEvalContext`, `evaluate_leaf` /
  `evaluate_leaves` (NotImplementedError), `SubgameNode.leaf_value` field. 8 tests.
- **Stage C — PROFILE_SAMPLE mode** (`8320103`). Per-rollout opponent biases drawn
  from the prior; M rollouts → `icm_adjust_returns` mean; ITM short-circuit;
  wall-clock guard; NaN-sample drop. Returns `LeafEvalResult` (value + `degraded`,
  a documented deviation from Q9's bare-tuple sketch). 8 tests incl. the
  bias-differentiation check (deterministic, replacing the spec's too-noisy
  hero-value-direction MC test).
- **Stage D — BEST_RESPONSE mode** (`dd9cfc3`). Per-opponent independent BR: CRN
  eval rollouts (others=blueprint) → argmax bias (lowest-index tie-break) → value
  pass under the composed profile. Common random numbers give exact-tie stability.
  Busted-seat / heads-up fixtures via `TournamentStructure.to_inner_game_string_for_state`.
  All Q10 invariants pass, including optional #11 (hero ≤ uniform, heads-up).
- **ICM busted-seat fix** (`ae8e1b5`, load-bearing — see findings). Opt-in
  `eligible` parameter on `icm_equity`; `icm_adjust_returns` / `_option_a` pass
  alive-at-start seats. Default unchanged (backward-compatible). 8 new ICM tests;
  existing 40 unchanged.
- **Stage E — `evaluate_leaves` batch path** (`fd7fb88`). Cache-sharing (one bucket-
  cache reset for the whole tree via `manage_cache_externally`); mutates
  `node.leaf_value` in place; `LeafBatchResult` summary with `partial_eval_degraded`;
  tree-wide wall-clock guard (between-leaves). 4 new batch tests (24 + 10 subtests
  total). Default `n_samples` dropped 8→5.
- **Doc + tracking** (`e5f9469`). Q4 superseded, Q12 (Stage E.5) filed,
  Q4.5 meta-discipline note added, NEXT_SESSION Stage-E.5 entry.

## What was decided

(See `docs/DECISIONS.md` for the locked leaf-evaluator architecture entry from
session 13; this session's decisions are recorded inline in `docs/SUBGAME_LEAF_DESIGN.md`
Q4/Q4.5/Q12 and the commit bodies.)

- **Default `n_samples` 8 → 5.** Measured BR M=8 = ~44 s / M=5 = ~30 s per 64-leaf
  tree (>> the 1.5 s budget). Per the Stage-E rule (M=8 > 12 s), dropped to 5. M is
  NOT the budget lever; Stage E.5 is.
- **Cache-sharing is the shipped Stage-E lever** (3× PROFILE, ~10× single-leaf BR).
  It does not fully close the budget — that needs Stage E.5.
- **`partial_eval_degraded`** named distinctly from the per-leaf `degraded` (a value
  fallback) and `eval_pool`'s `exceeded_cap`; it denotes an incomplete batch.
- **Lockstep network batching NOT built** — the measurement showed it optimizes a
  non-bottleneck.

## What was learned / measured

Three foundational-primitive findings, **same pattern each time** (a load-bearing
number asserted without measurement-with-attribution, caught only by re-measuring
at the start of the work depending on it):

1. **Parse-attribution misdiagnosis.** Q4 said the ~0.9 ms/step floor was
   `parse_state_6max`'s regex; re-measurement showed `parse` = 0.008 ms and the
   cost was `_build_view_6max`/`discretize` over the ~9,803-element fullgame
   `legal_actions()`. Corrected in `dc09617` + Stage A `fast_view`.
2. **ICM busted-seat bug** (`icm.py` busted-handling). `icm_equity` split the bottom
   payouts among ALL stack-0 seats, so a mid-rollout bust spuriously enriched
   already-eliminated (pre-busted) seats. The bug had existed since Phase 4b under
   40 passing tests (none covered pre-busted-vs-newly-busted); it was caught by
   Stage D's realistic busted-seat ITM fixture (Option-A vs rollout disagreed by
   ~0.44). Fixed `ae8e1b5`. **Implication:** `dcfr-overnight-3000` was trained with
   the buggy ICM, so its late-game/post-bubble value function is systematically
   biased (over-valuing busting) in states that START with busted seats. Retrain
   decision deferred to sub-step 6's measurement.
3. **Encoder bucket-MC bottleneck.** Q4 claimed the per-step bottleneck was the
   network forward (~0.13 ms GPU). Reality (Stage E): `encode_from_parsed`
   cache-miss = **10.77 ms** (treys MC, `bucket_runouts=20`), ~43× the network
   (0.25 ms single / 0.45 µs/row batched). Filed as Stage E.5 / Q12.

**Meta-lesson (recorded in design doc Q4.5):** three wrong load-bearing numbers in
three stages, same root cause. Going forward, every numerical claim in any design
doc is a HYPOTHESIS pending fresh re-measurement at the start of the work that
depends on it; stage prompts must schedule the re-measurement explicitly.

Other measured facts: GPU IS available on this box (`torch.cuda.is_available()=True`,
contradicting CLAUDE.md) but irrelevant — the bottleneck is CPU/treys-bound
bucket-MC. `evaluate_leaves` (64-leaf depth-3 tree): PROFILE M=8 ≈ 3.5 s, BR M=8 ≈
44 s, BR M=5 ≈ 30 s. Cache-sharing freezes each `(hero,board)` bucket-MC draw, so a
few leaves differ ~0.5 from per-leaf-fresh evaluation (a bucket-assignment flip
under `bucket_runouts=20` noise, not a bug; batch values are self-consistent).

Tests at close: leaf evaluator 24 + 10 subtests; no regression across
icm/icm_returns/subgame/fast_view/cfr6 (95/95).

## State at close

- **Done:** sub-step 2 Stages A–E (fast_view, scaffold, PROFILE_SAMPLE,
  BEST_RESPONSE, evaluate_leaves) + the ICM busted-seat fix. The leaf evaluator is
  correct and usable for sub-step 3 development; it is NOT yet fast enough for
  sub-step 6's pool evaluation.
- **Deferred:**
  - **Stage E.5** (encoder bucket-MC precompute, design doc Q12) — budget-closing;
    MUST land before sub-step 6.
  - **Stages F / G** — Q11 Level-1 (leaf-only) and Level-2 (decision-level stub
    solver) ablations; sub-step 2 closure.
  - **Fold `fast_view` into canonical `_build_view_6max`** after sub-step 2 closes
    (tracked deliverable, NEXT_SESSION, with the dcfr-overnight-3000 fp-repro gate).
  - **Retrain `dcfr-overnight-3000` with fixed ICM** — decide after sub-step 6
    measures the practical impact of the busted-seat bias.
- **Open questions for next session (all filed in design doc Q12):** where the
  Stage E.5 precompute hooks in; encoder API change vs pre-populating
  `_bucket_cache`; precompute cost vs off-table miss rate; interaction with the
  shipped cache-sharing.

## Next session opens with

**Stage E.5 implementation.** Read the encoder (`infoset6.py`), abstraction
(`abstraction.py`), and bucket-clustering modules end-to-end before designing.
Re-read design doc **Q12** (the four design questions) and this summary. Then
**propose the Stage E.5 design before implementing** (per the measurement-discipline
note: re-measure the bottleneck and the expected precompute cost first).
