# Session 20 — 2026-05-26

Focus: **Close sub-steps 4 & 5 — policy extraction + the `SubgamePolicy` wrapper —
making the full subgame-solving stack a drop-in `eval_pool` challenger.** Two
foundational findings surfaced and were fixed cleanly mid-stream (bringing the
project total to **eight**); the session ended with sub-step 6 proven feasible
end-to-end (~3.8 h Contabo-parallel).

## What was done

- **Sub-step 4 — policy extraction** (`9798832`). `subgame_solver.extract_action(
  result, state, rng, mode)`: maps `root_policy` → a played chip action (sample
  default / argmax), reuses the tree's own `_discretize_at_decision` map, and applies
  the `ALLIN`→CALL(1) chip-0 alias (`b2dded5`) at the translation boundary. 8 tests
  (sample convergence 0.596 vs 0.60; alias returns CALL not FOLD).
- **Sub-step 5 Stage 5-A — scaffold + gate + instrumentation** (`6ab60be`).
  `SubgamePolicy` conforming to `eval_pool.Policy`; gate (≥3 legal actions AND
  blueprint max-prob <0.95); `_reconstruct_starting_stacks` = money+contribution
  (the Policy interface omits starting_stacks; reconstruction test-verified across
  preflop/mid-hand/all-in). Measured **empirical gate fire-rate f = 0.271** (below the
  0.55 hypothesis → "pleasant surprise" branch) and the **regime-asymmetry finding**:
  98.2% of decisions are preflop, opposite where Stage F/G's BR signal is strongest
  (`4cf3fe7`).
- **Sub-step 5 Stage 5-B — full pipeline** (`e3f2218`). `_solve_action`:
  build→evaluate_leaves→solve→(degraded?)→extract. Integration surfaced finding #7.
- **Finding #7 — chance-leaf parse crash, fixed** (`03576eb`). `_best_response_biases`
  / `_option_a` crashed on chance-node leaves (`observation_string(current_player()==
  -1)`); now parse chance-safely (fixed observer). Measured **22/25 chance leaves are
  bias-INACTIVE**.
- **Finding #8 — tree-builder leaf explosion, fixed** (`9ff106d`). A round-closing
  depth-3 tree blew up to **2560 leaves (2048 chance)** because chance branched ×8
  without consuming the action-depth budget and compounded across streets; the
  chance-leaf fix made them all evaluable → a single solve took **>666 s**. Two
  coordinated changes: (A) chance collapses to a transparent LEAF (the rollout draws
  the board) → **2560 → 5–12 leaves**; (B) chance leaves use **blueprint-only**
  evaluation (skip `v×k` BR, justified by 88% bias-inactivity).
- **Sub-step 5 Stage 5-C — degraded diagnostics + reproducibility + closure**
  (this session's `feat` commit). WARNING + `n_degraded` on degraded (no back-off);
  `stats()` adds gate_skip_rate / gate_solve_rate / degraded_rate; reproducibility,
  gate-determinism, distribution, and degraded-path tests.

## What was decided

- **Sub-step-4 decisions (4.1–4.5)** all as designed: sample default, reuse the
  discretize map, alias at chip-translation, explicit rng, no in-step-4 degraded
  fallback (deferred to sub-step 5, which owns the blueprint).
- **Sub-step-5 decisions (5.1–5.4)** as designed: Option-B gate, degraded→blueprint
  + log + count + **no temporal back-off**, stateless across decisions, constructor
  defaults (K=1000, leaf BR/M=8, depth-3, gate (3, 0.95)).
- **Chance is transparent to the subgame tree** (mitigation Change A) — a
  depth-limited solver should defer the board runout to leaf evaluation, not expand
  it into the tree. This is a correctness improvement, not only a cost fix.
- **Chance leaves get blueprint-only evaluation** (Change B) — a cost optimization at
  the architectural cost of ~12% of chance-leaf BR signal, itself a small slice of
  total BR signal.

## What was learned / measured

- **Empirical f = 0.271** (3000 self-play decisions); preflop is 98.2% of decisions.
- **Per-solve cost (production M=8 depth-3 K=1000):** chance-free ~12–22 s (≈ Q13),
  chance-reaching ~5 s, **blended ~6.7 s** over a 50-decision benchmark; per-skip
  ~0.9 ms; **0 degraded** in 50 decisions. Bit-identity to the Stage-G stub preserved.
- **The discipline pattern caught two more foundational findings** (chance-leaf parse,
  tree-builder cost) — both in the round-closing/all-in regime that Stage F/G
  (hand-built leaves / depth-1) and Q13 (chance-free first-decision trees) never
  exercised, and both caught at Stage-5-B integration *before* a 100 h+ sub-step-6 run.
- **LOAD-BEARING for sub-step 6:** the deployment mix (98% preflop + chance leaves
  blueprint-only/88%-bias-inactive + shallow round-closing solves) concentrates the BR
  lift in the **minority of chance-free decision-bearing solves**. Stage F/G's +3.4σ /
  +3.07σ were measured on a **different mix**; sub-step 6's bb/100 may be substantially
  below that naive projection. Not a defect — the demonstration regime ≠ the
  deployment regime.

## State at close

- **Done: sub-steps 4 & 5 CLOSED** (`9798832`, `6ab60be`→`9ff106d`). The full stack
  (tree → leaf-eval → solver → extract → SubgamePolicy) is a drop-in `eval_pool`
  challenger. Full suite green throughout (the one failure is the pre-existing,
  UNTRACKED `tests/test_pushfold.py:274`).
- **Open / next: sub-step 6 — Level-3 pool ablation** (the strength go/no-go). Needs
  hand-level multiprocessing (eval_pool is sequential). Projected ~3.8 h
  Contabo-parallel at the measured ~6.7 s/solve, f≈0.27.
- **Unchanged carry-forward:** fold `fast_view` into the canonical path; the
  `dcfr-overnight-3000` ICM-retrain decision after sub-step 6.

## Next session opens with

**Sub-step 6 design proposal** — the three-challenger pool ablation (subgame-BR vs
subgame-PROFILE vs blueprint over `league-v2-600` × 5,000 hands, BR-vs-blueprint
σ > 2 to lock BR), the hand-level multiprocessing wrapper `eval_pool` needs, and the
regime-asymmetry lens for reading the measured bb/100. This is the first measured
strength delta from subgame solving — the milestone the whole B1c line has built toward.
