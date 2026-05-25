# Session 19 — 2026-05-25

Focus: **Sub-step 3 — build the real subgame CFR solver, end to end.**
Design proposal → review-gate approval → five implementation stages (3-A…3-E).
Sub-step 3 **CLOSED**: `src/nlhe/subgame_solver.py` is the multi-iteration vanilla
weighted CFR loop over the depth-limited tree, production K=1000. Unlike Stages F/G
(ablation *gates*), sub-step 3 closes by **execution-and-measurement** — implementation
stages, each with a commit and a stop point — same discipline, different shape.

## What was done

- **Design proposal** (`512039c`, `docs/SUBSTEP_3_DESIGN.md`, ~2750 words). Interface,
  algorithm, seven pre-committed decisions, cost model, test plan, staged plan, scope.
  Surfaced two deviations from the documented plan for sign-off.
- **Stage 3-A — scaffold** (`3ef06e4`). `SubgameSolveContext` / `SubgameSolveResult`,
  warm-up caching (`_build_warmup`: per-decision-node adv/mask/σ from the blueprint —
  all network forwards here), and the **K=0** blueprint-passthrough path. Plus the
  approved doc corrections: STATUS/NEXT_SESSION "external-sampling"→"vanilla weighted
  CFR", and an ARCHITECTURE.md Layer-3 paragraph on BR-as-safety.
- **Stage 3-B — K=1 bit-identity** (`aeabae1`). The single-iteration regret update;
  `solve_subgame(K=1).root_policy` is **bit-identical** (`np.array_equal`, max|diff|
  0.0) to the Stage-G stub `stub_root_policy` on a real 6-action root — the
  seed-to-plant continuity gate. `_regret_matched_policy` mirrors the stub line-for-line
  (copied not imported, to avoid a `src→scripts` dependency; the test is the guard).
- **Stage 3-C — the K>1 loop (the heart)** (`b415517`). `_run_cfr`: vanilla weighted
  traversal — chance weighted by `chance_prob`, opponents by their FIXED blueprint
  strategy, hero accumulates RM+ regret; output = linear-LCFR average root policy.
  Leaf/terminal hero values precomputed once. Convergence anchor (synthetic
  chance-weighted tree → analytical pure BR) and the convergence-direction test both
  pass.
- **Stage 3-D — diagnostic enrichment** (`10f7e45`). Five supplementary
  `SubgameSolveResult` fields (n_iterations_run, convergence_history, root_q_values,
  root_advantages_blueprint, root_advantages_refined) + `summarize_solve_result`
  (JSON-serializable, adds `policy_shift_l1`). `root_policy` stays the sub-step-5
  contract; K=1 bit-identity preserved under enrichment.
- **Stage 3-E — production K locked** (this session's first commit below). K=1000
  chosen from 3-C measurements; design doc cost model updated with measured numbers;
  Stage-3-E closure section added; three lock tests (default K=1000, converged_l1_tail
  < 1e-5, CFR loop < 1 s).

## What was decided

- **Production K = 1000** — a convergence-quality decision, not a cost one. `converged_l1_tail`
  4.4e-2 → 4.4e-5 → 4.4e-8 across K=10/100/1000 (stable at 1000); the loop is 0.109 s
  at K=1000 (~250× under the 27 s/decision budget), linear in K, so escalation is free.
- **Decision 2 (vanilla weighted CFR, not external sampling) — APPROVED.** The tree
  builder's per-branch `chance_prob` is a full-traversal design; regret units are
  sampling-variant-independent; the small depth-limited tree makes vanilla
  deterministic and faster-converging. Methodologically stronger than the documented
  spec — same pattern as Stage F/G's approved deviations.
- **Decision 6 (BR-as-safety, no HUNL CFV gadget) — APPROVED.** Pluribus-style safety
  in 6-max ICM is the BR-over-biased-continuations leaf mode (Stage G +3.07σ confirms);
  the CFV gadget is a 2p0s construction, structurally inapplicable. ARCHITECTURE.md
  Layer 3 now states this.
- **`converged_l1_tail` tracks the AVERAGE (output) root policy, not the current
  strategy.** Under fixed opponents (Decision 4) the subgame is a best-response
  problem; the current strategy collapses to its pure BR in ~7 iterations, so only the
  average-policy movement is a usable convergence signal.

## What was learned / measured

- **Cost is ~15× under the design estimate.** Design guessed ~0.6–2 s for the K=1000
  loop (~10 µs/node); measured **0.109 s** (0.109 ms/iter, ≈0.5 µs/node) on a 216-node
  depth-3 tree. The loop is pure arithmetic over cached values (no in-loop forwards or
  rollouts; Decision 1), so W is a non-factor — Z (leaf eval, ~10–20 s) dominates the
  budget by ~100×. Implication: K selection is purely about convergence quality.
- **The current-strategy convergence metric is useless here** (collapses to exactly 0
  by iteration ~7), a direct consequence of the fixed-opponent best-response structure.
  Caught and resolved by shipping the average-policy metric instead — the same
  "measure the right thing, deviate from the literal spec when it's stronger" pattern
  as Stage F/G.
- **`traverse_6max` is not reusable** as a subgame solver (it is a training-time
  reservoir-buffer writer); sub-step 3 is a new tabular traversal that reuses only
  `_strategy_from_advantages`, `icm_adjust_returns`, and the regret formula.
- **Diagnostic advantages are raw 7-vectors** (illegal actions carry the net's raw
  output; RM+ masks them). `policy_shift_l1` is the clean "pull off blueprint"
  magnitude; the refined-advantage vector is read for direction only (it grows ~linearly
  in K).

## State at close

- **Done: sub-step 3 CLOSED** (`3ef06e4`→`10f7e45` + the 3-E commit). 32 solver tests
  green; full suite 635 passed + 10 subtests (the 1 failure is the pre-existing,
  UNTRACKED pushfold experiment `tests/test_pushfold.py:274`, unrelated throughout).
- **Open / next: sub-step 4 — policy extraction** (`root_policy` → played action;
  argmax/sampling; `ALLIN`→CALL(1) chip-0-alias translation surfaces here).
- **Unchanged carry-forward:** fold `fast_view` into the canonical path (now both
  sub-steps 2 & 3 are closed); `dcfr-overnight-3000` ICM-retrain decision after
  sub-step 6; the sub-step-5 `eval_pool.Policy` wrapper.

## Next session opens with

**Sub-step 4 — policy extraction.** A small follow-up: map the refined `root_policy`
7-vector to a chosen action (argmax and sampling modes), translate the `DiscreteAction`
back to an OpenSpiel chip action via the solver's discretize map, and handle the
`DiscreteAction.ALLIN` → CALL(1) alias (`b2dded5`) at this extraction boundary. Then
sub-step 5 (the `Policy` wrapper that chains build→evaluate_leaves→solve→extract) and
sub-step 6 (the Level-3 pool ablation — the first measured strength delta from subgame
solving).
