# Next Session Pickup Notes

## Where the project stands at end of session 13

B1c (depth-limited subgame solving) is mid-build:

- **Sub-step 1.5 — tree builder: DONE and correct** (`2be87df`). Enumerates the
  `DiscreteAction` set via the same path as `cfr6.traverse_6max`; 20/20 tests
  pass against the production game `six_max_sng(starting_stack=10000)`.
- **Sub-step 2 — leaf evaluator: DESIGN APPROVED, not yet implemented.** Full
  design in `docs/SUBGAME_LEAF_DESIGN.md` (commits 89ec957 → 4029757 → 98edf14).

## Sub-step 2 is the next deliverable — implement `src/nlhe/subgame_leaf.py`

Build to the approved design. Load-bearing points:

1. **BEST_RESPONSE is the production form**; PROFILE_SAMPLE is the fallback /
   ablation mode (`LeafEvalContext.mode`). Both must be implementable; default is
   BEST_RESPONSE. Leaf value = 6-vector ICM-equity deltas (drop-in for
   `SubgameNode.terminal_returns`, stored on a new `SubgameNode.leaf_value` field).
2. **Best-response is computed vs hero's BLUEPRINT at the root** (Brown/Sandholm
   2018 single-pass approximation), NOT vs hero's iteration-k subgame strategy —
   this is what keeps leaf values cacheable-exact across CFR iterations
   (compute once per decision, not Z×W).
3. **Parse optimization (Path B) is IN SCOPE** (Q4.5). Implement `ParsedStateDelta`:
   parse once at the leaf, then incrementally update only the mutable fields
   (pot / contributions / history / current player / board on chance steps /
   folded+all-in sets) after each rollout step. Target ~0.9 → ~0.15 ms/step. If it
   lands only at ~0.3–0.4 ms/step, apply the Q4.5(c) fallback **in order** (cut L
   50→30, then M 8→5, then raise X 6→8 s) and **record the chosen knob in the
   implementation commit message.**
4. **Q11 two-level ablation gates sub-step 2 completion.** Level 1 (leaf-only) and
   Level 2 (decision-level via a **stub one-iteration root regret update** — itself
   a sub-step 2 deliverable; checks BR yields a flatter / more-mixed root policy
   than PROFILE_SAMPLE on ~50 root decisions). Level 3 (full league-v2-600 pool,
   5,000 hands) waits until after sub-step 5.
5. **ICM Option B** (rollout → `state.returns()` → `icm_adjust_returns`) with the
   `is_itm()` Option-A short-circuit. **BR ties → lowest bias index.**

## Carry-forward for sub-step 5 (SubgamePolicy)

- **`DiscreteAction.ALLIN` → chip 0 = FOLD when facing a shove with no re-raise
  room** (documented `b2dded5`). When SubgamePolicy selects ALLIN and translates
  back to a game action in that state, it must map to **CALL (1)**, not the chip-0
  fold. Do not let the alias ship a fold where the policy meant all-in.

## Remaining B1c roadmap

1. **Sub-step 2 — leaf evaluator** (NEXT; this session's deliverable).
2. Sub-step 3 — subgame CFR loop (replaces the Level-2 stub with the real solver).
3. Sub-step 4 — policy extraction (hero's refined root action distribution).
4. Sub-step 5 — SubgamePolicy wrapper (conform to `eval_pool.py` `Policy`;
   handle the ALLIN→CALL translation above).
5. Sub-step 6 — Level-3 pool ablation (BR vs PROFILE_SAMPLE vs blueprint).

## Future optimization (NOT sub-step 2): Path A parse rewrite

Bypass `information_state_string` / observation-string parsing entirely and read
fields directly from native OpenSpiel state methods. Benefits `traverse_6max` at
TRAINING time too, not just decision-time leaf eval — so it's a separate task with
its own validation surface. Pick up when blueprint-training compute (not
decision-time latency) becomes the bottleneck.

## Docs map

- `docs/SUBGAME_LEAF_DESIGN.md` — the approved sub-step 2 design (read first).
- `docs/sessions/session_13_summary.md` — this session.
- `docs/sessions/README.md` — the per-session-summary convention.
- `docs/STATUS.md` — current snapshot. `docs/DECISIONS.md` — locked choices.
