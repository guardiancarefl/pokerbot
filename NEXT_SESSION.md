# Next Session Pickup Notes

## Runtime state (added session 16 — read before any eval/timing planning)

The project's **nominal** runtime is the Contabo VPS (12 vCPU, no GPU) per the
original architectural decision (CLAUDE.md / DECISIONS.md). **Actual development
since ~session 13 runs on whatever RunPod instance is currently active** — session
16's measurement box was a 128-core + RTX PRO 4000 Blackwell machine, not Contabo.
CLAUDE.md and the body of this file still describe the Contabo host; treat that as
the nominal/fallback target, not a statement of the live box. For sub-step 6 eval
planning: the Q13 conclusion (**current evaluator is adequate, no optimization
needed**) holds across hardware — on Contabo (12 vCPU, parallelism Y≈10) the full
Level-3 ablation is **~21 h** wall-clock; on a many-core box (Y≈24+) it is
**~10.5 h or less**. Both fit an overnight-to-weekend budget. Note the Q13
hardware surprise: this evaluator runs **faster on CPU than GPU** (single-row
`[64,64]` forward is launch-bound), so a GPU pod is not required — many CPU cores
are what help. Full reasoning: `docs/STAGE_E_BUDGET_REDERIVATION.md`.

## Where the project stands (start of session 20)

B1c (depth-limited subgame solving):

- **Sub-step 1.5 — tree builder: DONE** (`2be87df`). Production-game tested.
- **Sub-step 2 — leaf evaluator: IMPLEMENTATION COMPLETE** (Stages A–E,
  `994b587`→`fd7fb88`) plus the load-bearing ICM busted-seat fix (`ae8e1b5`) and
  the post-Q13 cleanups (`db89145` restore M=8; `3109fb0` cache-reset guard). The
  evaluator is correct, tested, and production-ready.
- **Q13 — budget RESOLVED** (`b092480`, session 16). No optimization needed;
  Stage E.5/E.6 **shelved**. (`docs/STAGE_E_BUDGET_REDERIVATION.md`.)
- **Stage F (Q11 Level 1) — CLOSED via SUBSTANTIVE_PASS_AGGREGATE** (session 17,
  `e939bce`→`9dbbfd4`). See `docs/sessions/session_17_summary.md`. The per-pair
  opponent-own-value resolution gate is **structurally intractable** in this
  shallow SNG (55% of leaf-opp pairs have zero bias effect → max resolution ~45%
  at any M), but the architecture is confirmed by the **aggregate** signal:
  hero-direction +3.4σ (BASELINE−BR) / +3.6σ (PROFILE−BR), 94% differentiation
  among resolved pairs, non-degenerate menu (biases 1/2/3 all selected).
- **Stage G (Q11 Level 2) — CLOSED via SUBSTANTIVE_PASS_AGGREGATE** (session 18,
  `cb82072`→`b4f85dd`). See `docs/sessions/session_18_summary.md`. The M=16 gate was
  a clean 2–3σ near-miss; the one design-sanctioned M=32 escalation cleared both
  load-bearing bars (value_suppression +3.07σ; policy_divergence significant, obs L1
  0.330 > null p99.7 0.297; differentiation 71.4%, 6 distinct shifted actions). BR
  demonstrably moves hero's *root decision*, not just leaf values. Finding: the
  "BR flatter" prior was wrong (BR shifts mass, does not flatten) — entropy was
  correctly demoted to non-load-bearing before the run.
- **Sub-step 3 — subgame CFR solver: CLOSED** (session 19, `3ef06e4`→`10f7e45`;
  `src/nlhe/subgame_solver.py`, `docs/SUBSTEP_3_DESIGN.md` Stage-3-E closure,
  `docs/sessions/session_19_summary.md`). Vanilla weighted multi-iteration CFR over
  the depth-limited tree: opponents fixed at blueprint, chance weighted by
  `chance_prob`, hero accumulates RM+ regret, output = linear-LCFR average root
  policy. **Production K=1000.** Closed by execution-and-measurement (implementation
  Stages 3-A…3-E), not an ablation gate. Both flagged decisions approved: vanilla
  CFR (Decision 2) and BR-as-safety / no CFV gadget (Decision 6).
- **Path: Path A (confirmation-first). Sub-steps 2 & 3 are CLOSED.** Next is sub-step 4.

## Current deliverable — Sub-step 4 (policy extraction)

Turn `SubgameSolveResult.root_policy` (the refined masked 7-vector from
`subgame_solver.solve_subgame`) into a **played action**. Scope:
- Map the 7-vector `DiscreteAction` distribution to a chosen action — support both
  argmax (eval/exploit) and sampling (mixed-strategy play) modes, mirroring the
  blueprint's own action-selection (`eval_6max_self_play._sample_action_from_policy`,
  `eval_pool` `mode` arg).
- Translate the chosen `DiscreteAction` back to an OpenSpiel chip action via the same
  `discretize_legal_actions` map the solver/tree used — and handle the
  **`DiscreteAction.ALLIN` → chip-0 FOLD alias**: when ALLIN is selected facing a
  shove with no re-raise room, emit **CALL (1)**, not the chip-0 fold (`b2dded5`).
  (This is the carry-forward formerly filed under sub-step 5; it surfaces here at the
  extraction boundary.)
- Keep it a thin, pure-ish function over `SubgameSolveResult` + the root state; the
  `eval_pool.Policy` wrapper that *calls* build→evaluate_leaves→solve→extract is
  sub-step 5.

Sub-step 3's diagnostic fields (`summarize_solve_result`) are available for the
extraction layer / sub-step 6 to log.

## Ablation gate reference (Stages F & G methodology, for sub-step 3 / Level 3 to reuse)

Both gates use a split metric (`verdict()`), status in
{PASS, SUBSTANTIVE_PASS_AGGREGATE, FAIL}, with **resolution** ("is the effect
detectable at >3σ on a CRN-paired difference") separated from **differentiation**
("does BR actually move the decision"), plus a `SUBSTANTIVE_PASS_AGGREGATE` fallback
for the resolution-intractable-but-aggregate-confirmed case this SNG produces:
- **Stage F** (`scripts/ablation_leaf_eval.py`): per-(leaf,opp) resolution;
  differentiation = BR picks non-blueprint bias.
- **Stage G** (`scripts/ablation_decision_level.py`): per-root resolution on the
  most mode-deviating action; differentiation = material L1 root-policy shift
  (direction-agnostic); load-bearing aggregate = value_suppression (≥3σ) +
  policy_divergence (per-deal mode-label **permutation null**, significant @99.7% —
  approved over the literal "bootstrap CI" spec, see SUBGAME_LEAF_DESIGN.md Q11 L2).

## B1c roadmap (post-Q13)

1. **Sub-step 2 — leaf evaluator — CLOSED** (Stages A–E DONE; **Stage F CLOSED**
   session 17, **Stage G CLOSED** session 18, both via SUBSTANTIVE_PASS_AGGREGATE).
2. ~~Stage E.5 — bucket-MC precompute~~ — **SHELVED (Q13)**, no longer on the path.
3. **Sub-step 3 — subgame CFR solver — CLOSED** (session 19, vanilla weighted CFR,
   production K=1000; `src/nlhe/subgame_solver.py`).
4. **Sub-step 4 — policy extraction (root_policy → action; ALLIN→CALL alias) — NEXT.**
5. Sub-step 5 — SubgamePolicy wrapper (conform to `eval_pool.py` `Policy`; calls
   build→evaluate_leaves→solve→extract).
6. Sub-step 6 — Level-3 pool ablation (BR vs PROFILE_SAMPLE vs blueprint),
   the full `league-v2-600` pool × 5,000 hands (Q13: ~10.5 h at Y≈24, feasible).

## Carry-forward for sub-step 5 (SubgamePolicy)

- **`DiscreteAction.ALLIN` → chip 0 = FOLD when facing a shove with no re-raise
  room** (documented `b2dded5`). When SubgamePolicy selects ALLIN and translates
  back to a game action in that state, it must map to **CALL (1)**, not the chip-0
  fold. Do not let the alias ship a fold where the policy meant all-in.

## TRACKED DELIVERABLE — fold fast_view into the canonical path (after sub-step 2 closes)

Sub-step 2 ships the view/discretize optimization as a parallel `src/nlhe/fast_view.py`
to contain blast radius (Stage A). Measured 6× faster (0.046 vs 0.28 ms/step) with
field-identical output. After sub-step 2 closes (Stages F/G done), fold it into the
canonical `cfr6._build_view_6max` + `actions.discretize_legal_actions` and re-point
all consumers: `traverse_6max` (TRAINING hot path), `subgame.py`,
`pushfold_policy.py`, `scripted_bots/policy.py`, `solver.py`, `policy_adapter.py`,
`scripts/eval_pool.py`, `scripts/eval_6max_self_play.py`.

**Acceptance for the fold-in (not optional):**
1. The Stage A exact-equality tests (`tests/test_fast_view.py`) become the
   regression guard and must stay green after the canonical path is swapped.
2. **Reproducibility against `dcfr-overnight-3000`:** run a small fixed training
   step (same seed, same data) on the blueprint *before* and *after* the swap and
   confirm the produced advantages / network outputs are identical to
   floating-point tolerance. If outputs diverge beyond tolerance, the fold-in is
   wrong — do not land it.

## Also still queued (unchanged by Q13)

- Decide the `dcfr-overnight-3000` ICM-retrain after sub-step 6 measures the
  practical impact of the busted-seat bias (`ae8e1b5`).

## Docs map

- `docs/SUBGAME_LEAF_DESIGN.md` — the sub-step 2 design + Q11 Stage F/G outcomes +
  Q13 resolution (read first).
- `docs/STAGE_G_DESIGN.md` — Stage G design proposal (stub, regime-aware gate).
- `docs/STAGE_E_BUDGET_REDERIVATION.md` — full Q13 budget reasoning + measurements.
- `docs/sessions/session_18_summary.md` — most recent session (Stage G closure,
  sub-step 2 complete).
- `docs/sessions/session_17_summary.md` — Stage F closure.
- `docs/sessions/session_16_summary.md` — Q13 budget re-derivation.
- `docs/sessions/README.md` — the per-session-summary convention.
- `docs/STATUS.md` — current snapshot. `docs/DECISIONS.md` — locked choices.
