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

## Where the project stands (start of session 18)

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
- **Path: Path A (confirmation-first).** Stage F done; Stage G (decision-level)
  remains before sub-step 3.

## Current deliverable — Stage G (Q11 Level 2, decision-level ablation)

Wrap the **simplest possible CFR around the leaf evaluator: a one-iteration regret
update at the root infoset only** (no full tree traversal). Measure hero's root
action distribution under each leaf-eval mode across ~50 root decisions.
Hypothesis: BR yields a flatter / more-mixed root policy than PROFILE_SAMPLE. The
stub is a **sub-step 2 deliverable**, superseded by the real solver in sub-step 3.

**Carry the Stage-F lesson into Stage G's success criterion (load-bearing):** do
NOT gate on a uniform expectation across all ~50 root decisions. Stage F showed
the bias effect is *absent* in the majority of states (opponents with no
bias-sensitive decision) and *concentrated* in late-street / bias-active spots —
a uniform per-decision gate would hit the same structural-intractability wall and
spuriously FAIL. Design Stage G's gate around the regime where the architecture
predicts an effect (root decisions where ≥1 live opponent faces real
post-flop / committed latitude), and treat the rest as expected no-ops. The Stage-F
split-metric pattern (resolution vs differentiation, CRN-paired stderr,
`SUBSTANTIVE_PASS_AGGREGATE` for the intractable-but-confirmed case) is the model.

## Stage F gate reference (the methodology, for Stage G to reuse)

`scripts/ablation_leaf_eval.py` `verdict()` — split metric, status in
{PASS, SUBSTANTIVE_PASS_AGGREGATE, FAIL}:
- **resolution_rate** = fraction of (leaf,opp) pairs where the most blueprint-
  deviating bias beats blueprint by > 3σ on the **CRN-paired** difference (cancels
  per-deal variance — the only way the small bias signal clears noise).
- **differentiation_rate** = among resolved pairs, fraction where BR picks a
  non-blueprint bias.
- PASS iff resolution ≥ 0.50 AND (differentiation ≥ 0.30 OR resolution ≥ 0.90);
  **SUBSTANTIVE_PASS_AGGREGATE** iff resolution < 0.25 AND both aggregate hero
  deltas ≥ 3σ (positive) AND differentiation ≥ 0.90 AND ≥ 3 distinct biases among
  resolved (W2 non-degenerate-menu guard); else FAIL.

## B1c roadmap (post-Q13)

1. **Sub-step 2 — leaf evaluator** (Stages A–E DONE; **Stage F CLOSED** session 17
   via SUBSTANTIVE_PASS_AGGREGATE; **Stage G remains** — gates the formal sub-step
   2 close).
2. ~~Stage E.5 — bucket-MC precompute~~ — **SHELVED (Q13)**, no longer on the path.
3. Sub-step 3 — subgame CFR loop (replaces the Stage-G Level-2 stub).
4. Sub-step 4 — policy extraction (hero's refined root action distribution).
5. Sub-step 5 — SubgamePolicy wrapper (conform to `eval_pool.py` `Policy`;
   handle the ALLIN→CALL translation below).
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

- `docs/SUBGAME_LEAF_DESIGN.md` — the sub-step 2 design + Q13 resolution (read first).
- `docs/STAGE_E_BUDGET_REDERIVATION.md` — full Q13 budget reasoning + measurements.
- `docs/sessions/session_17_summary.md` — most recent session (Stage F closure).
- `docs/sessions/session_16_summary.md` — Q13 budget re-derivation.
- `docs/sessions/README.md` — the per-session-summary convention.
- `docs/STATUS.md` — current snapshot. `docs/DECISIONS.md` — locked choices.
