# Project Status

**Last updated:** 2026-05-24 (Session 13)
**Current phase:** B1c — depth-limited subgame solving (real-time), sub-step 2

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

## In progress
- B1c sub-step 2 implementation — `src/nlhe/subgame_leaf.py` (next deliverable).

## Next up (sub-step 2 deliverables)
1. `subgame_leaf.py`: BEST_RESPONSE (default) + PROFILE_SAMPLE leaf evaluator;
   6-vector ICM-equity values; `SubgameNode.leaf_value` field; `LeafEvalContext`.
2. Path B incremental parsing (`ParsedStateDelta`) — in scope per Q4.5; record the
   fallback knob chosen if it underperforms.
3. Stub one-iteration root regret update (for Q11 Level 2).
4. Q11 Level 1 (leaf-only) + Level 2 (decision-level) ablations — both gate
   sub-step 2 completion.

## Then (later B1c sub-steps)
- Sub-step 3: subgame CFR loop. Sub-step 4: policy extraction.
- Sub-step 5: SubgamePolicy wrapper — must map `DiscreteAction.ALLIN` → CALL(1)
  when facing a shove with no re-raise room (the chip-0 alias, `b2dded5`).
- Sub-step 6: Level-3 pool ablation (BR vs PROFILE_SAMPLE vs blueprint).

## Known issues / open questions
- `parse_state_6max` regex parse is ~0.9 ms/step (CPU-bound, GPU-invariant) — the
  binding constraint on the BR-mode decision budget. Path B (sub-step 2) targets
  ~0.15 ms/step; full Path A rewrite deferred (see `NEXT_SESSION.md`).
- BR-vs-blueprint robustness gain is an empirical bet, unproven until the Q11
  Level-3 pool ablation runs (post-sub-step-5).
- `SESSION_LOG.md` documents through Session 9 only; Sessions 10–12 live in commit
  messages / STATUS, not back-filled. Session 13+ summarized in `docs/sessions/`.

## Decisions deferred
- Path B fallback knob (cut L / cut M / raise X) — measure incremental parsing first.
- α (bias strength, default 3.0) and k tuning — revisit if the Q11 Level-3 ablation
  underwhelms.
