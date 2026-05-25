# Project Status

**Last updated:** 2026-05-26 (Session 20)
**Current phase:** B1c — depth-limited subgame solving (real-time), sub-step 6
(sub-steps 4 & 5 CLOSED)

> Note: this file was badly stale before Session 13 (it still read "Pre-Phase-1
> setup, 2026-05-21"). Rewritten from git history + the docs. Cross-check
> `git log --oneline` before trusting any single line here — STATUS can lag the
> last commit.

## Done
- **Phase 1 — Leduc Deep CFR** (OpenSpiel-wrapped), validated.
- **Phase 2 — custom NLHE Deep CFR:** EMD card abstraction, external-sampling
  solver with bit-identical checkpoint resume, Slumbot eval client.
- **Phase 4 — 6-max SNG:** parametric game strings, ICM value function
  (Malmuth-Harville) + ICM-adjusted returns wired into training, 6-max
  external-sampling CFR (`cfr6.traverse_6max`), PSRO league play (v1 + v2);
  `dcfr-overnight-3000` blueprint trained ICM-correct.
- **Phase 5 — Shanky bot-profile runtime:** parser, predicate evaluator, policy
  adapter; 36 profiles loadable; league-v2 + Shanky eval baselines measured.
- **B1c sub-step 1.5 — subgame tree builder:** correct discretized enumeration,
  20/20 tests vs the production game (`2be87df`). Internal-node descendants
  invariant verified on decision + chance trees.
- **B1c sub-step 2 — leaf evaluator DESIGN approved**
  (`docs/SUBGAME_LEAF_DESIGN.md`, best-response form, after two revisions +
  Q4.5 / Q11 additions).
- **B1c sub-step 2 — leaf evaluator IMPLEMENTED** (Stages A–E, `994b587`→`fd7fb88`;
  ICM busted-seat fix `ae8e1b5`; M=8 restore `db89145`; cache-reset guard
  `3109fb0`). Correct, tested, production-ready.
- **Q13 — leaf-eval budget RESOLVED** (`b092480`, session 16): no optimization
  needed, Stage E.5/E.6 shelved (`docs/STAGE_E_BUDGET_REDERIVATION.md`).
- **B1c sub-step 2 — Stage F (Q11 Level 1 leaf ablation) CLOSED** via
  SUBSTANTIVE_PASS_AGGREGATE (session 17, `e939bce`→`9dbbfd4`;
  `docs/sessions/session_17_summary.md`). The per-pair opponent-own-value
  resolution gate is structurally intractable (55% of leaf-opp pairs have zero
  bias effect → max resolution ~45% at any M), but the architecture is confirmed
  by the aggregate hero-direction signal (+3.4/+3.6σ), 94% differentiation among
  resolved pairs, and a non-degenerate menu.
- **B1c sub-step 2 — Stage G (Q11 Level 2 decision-level stub ablation) CLOSED** via
  SUBSTANTIVE_PASS_AGGREGATE (session 18, `cb82072`→`b4f85dd`;
  `docs/sessions/session_18_summary.md`). **Sub-step 2 is now complete.** The M=16
  gate was a clean 2–3σ near-miss (value_suppression +2.45σ, policy_divergence
  sig@~99.5%); the one design-sanctioned M=32 escalation cleared both load-bearing
  bars (value_suppression +3.07σ; policy_divergence significant, obs L1 0.330 > null
  p99.7 0.297; differentiation 71.4%, 6 distinct shifted actions). Finding: the
  "BR is flatter" prior was **wrong** (BR −0.29σ on entropy — BR *shifts mass*, does
  not flatten); entropy was correctly demoted to non-load-bearing before the run.
- **B1c sub-step 3 — the real subgame CFR solver CLOSED** (session 19,
  `3ef06e4`→`10f7e45`; `src/nlhe/subgame_solver.py`, `docs/SUBSTEP_3_DESIGN.md`,
  `docs/sessions/session_19_summary.md`). Vanilla weighted multi-iteration CFR over
  the depth-limited tree (Decision 2, approved): opponents fixed at blueprint, chance
  weighted by `chance_prob`, hero accumulates RM+ regret, linear-LCFR average root
  policy. Stages 3-A scaffold/warm-up/K=0 → 3-B K=1 bit-identical to the Stage-G stub
  → 3-C K>1 loop → 3-D diagnostics (`summarize_solve_result`) → 3-E **production
  K=1000** locked. Closed by execution-and-measurement (implementation stages), not
  an ablation gate. Measured: K=1000 loop 0.109 s (~15× under estimate, ~250× under
  the 27 s budget); convergence `converged_l1_tail` 4.4e-2→4.4e-8 over K=10→1000.
  Safety = BR leaf mode, no CFV gadget (Decision 6, approved).
- **B1c sub-step 4 — policy extraction CLOSED** (session 20, `9798832`;
  `subgame_solver.extract_action`). root_policy 7-vector → played chip action
  (sample default / argmax), reuses the tree's discretize map, applies the
  `ALLIN`→CALL(1) chip-0 alias (`b2dded5`) at the translation boundary.
- **B1c sub-step 5 — SubgamePolicy wrapper CLOSED** (session 20, `6ab60be`→`9ff106d`;
  `src/nlhe/subgame_policy.py`, `docs/SUBSTEP_5_DESIGN.md`,
  `docs/sessions/session_20_summary.md`). Drop-in `eval_pool.Policy`: gate (≥3 actions
  AND blueprint max-prob <0.95; empirical f≈0.27) → SKIP (blueprint) / SOLVE
  (build→evaluate_leaves→solve→extract) → degraded → blueprint fall-through (WARNING +
  `n_degraded`, no back-off). Two foundational findings caught + fixed this session:
  **chance-leaf parse crash** (`03576eb`) and **tree-builder leaf explosion**
  (`9ff106d`, 2560→5–12 leaves; chance now collapses to a transparent leaf, chance
  leaves use blueprint-only eval — 88% bias-inactive). Per-solve ~6.7 s blended
  (≈ Q13); sub-step-6 projected **~3.8 h Contabo-parallel** — feasible.

## In progress
- B1c **sub-step 6 — Level-3 pool ablation** (subgame-BR vs subgame-PROFILE vs
  blueprint over `league-v2-600` × 5,000 hands). Sub-steps 4 & 5 are closed; the
  full subgame-solving stack (tree → leaf-eval → solver → extract → SubgamePolicy)
  routes through `eval_pool` unchanged.

## Next up (sub-step 6 + handoff)
1. **Sub-step 6 design proposal** (next session): the go/no-go strength measurement.
   Needs **hand-level multiprocessing** (`eval_pool` is sequential; ~3.8 h
   Contabo-parallel only with it). **Load-bearing interpretation context:** the
   deployment mix (98% preflop, chance leaves blueprint-only/bias-inactive,
   round-closing solves shallow) concentrates the BR lift in the minority of
   chance-free decision-bearing solves — sub-step 6's bb/100 may be **well below**
   the Stage F/G aggregate-signal projection (see SUBSTEP_5_DESIGN Stage-5-C closure).
2. View/discretize fast path is shipped (`src/nlhe/fast_view.py`); fold into the
   canonical path now that sub-steps 2–5 have closed (NEXT_SESSION.md tracked
   deliverable + acceptance criteria).

## Then (later B1c sub-steps)
- Decide the `dcfr-overnight-3000` ICM-retrain after sub-step 6 measures the
  busted-seat-bias impact (`ae8e1b5`).

## Known issues / open questions
- The ~0.9 ms/step state-prep floor is `_build_view_6max` (0.64 ms) + `discretize`
  (0.10–0.24 ms) doing O(n) Python ops over the ~9,803-element fullgame
  `legal_actions()` — NOT the regex parse (0.008 ms; earlier attribution corrected
  2026-05-24). The fix is the sorted-legal-actions fast path (`fast_view.py`,
  sub-step 2 Stage A), gate ≤ 0.30 ms/step; folding it into canonical
  `_build_view_6max` is filed as a follow-up (see `NEXT_SESSION.md`).
- BR-vs-blueprint robustness gain is an empirical bet, unproven until the Q11
  Level-3 pool ablation runs (post-sub-step-5).
- `SESSION_LOG.md` documents through Session 9 only; Sessions 10–12 live in commit
  messages / STATUS, not back-filled. Session 13+ summarized in `docs/sessions/`.

## Decisions deferred
- Fast-path fallback knob (cut L / cut M / raise X) — measure `fast_view.py` first.
- α (bias strength, default 3.0) and k tuning — revisit if the Q11 Level-3 ablation
  underwhelms.
