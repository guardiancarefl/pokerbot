# Next Session Pickup Notes

## Where the project stands at end of session 12

**Committed and pushed to origin (`phase4f-league`):**
- Phase 5 archetype runtime: parser + runtime + policy adapter, 147 tests, 36 Shanky profiles loadable
- League v2: LeaguePool with Shanky-dispatch, 28-entry registry, league-v2-600 baseline measured (tie with DCFR-matched-iter)
- Shanky tournament eval vs DCFR-3000 (37 archetype matchups, all losses; 5 statistical ties)
- **Subgame tree builder (Track B1c sub-step 1)** at `920e95f` — depth-limited tree construction, 15 tests passing

## What's NOT yet working (the gap to fix next session)

The subgame tree builder passes its own tests but has a critical bug exposed when used with the real production game:

**Bug:** `build_subgame_tree()` iterates raw `state.legal_actions()` to enumerate children at decision nodes. With OpenSpiel's `bettingAbstraction=fullgame` (what `six_max_sng()` uses), this returns ~10,000 chip-action integers — one per possible chip amount. The tree therefore explodes to 600+ leaves at depth=1 instead of the expected 3-5.

**Fix:** The existing CFR walker in `src/nlhe/cfr6.py` (lines 290-350) handles this correctly by calling `discretize_legal_actions(legal_chip, view)` from `src/nlhe/actions.py`. That maps the ~10,000 raw chip actions to the 7-action `DiscreteAction` enum. The tree builder needs to use the same discretization at decision nodes.

**What needs to change in `src/nlhe/subgame.py`:**
1. At decision nodes, call `discretize_legal_actions` to get the discrete action set (not raw `state.legal_actions()`)
2. Branch on each discrete action's corresponding chip action
3. Store the DiscreteAction enum value (not the raw chip int) as `action_from_parent` for clarity in CFR
4. Re-test against the real game: `pyspiel.load_game(six_max_sng(starting_stack=10000))`, NOT against a hand-written ACPC gamedef

**What `tests/test_subgame.py` needs:**
- Existing tests use `load_universal_poker_from_acpc_gamedef` with a hand-written gamedef. They passed but didn't catch the bug because that gamedef has different action structure than the production wrapper.
- New tests must use `pyspiel.load_game(six_max_sng(...))` to validate against the actual game.

## Why the leaf evaluator was reverted

The leaf evaluator (`src/nlhe/subgame_leaf.py`, sub-step 2) was built but reverted because it depends on the tree builder being correct for the real game. Once the tree builder fix is in and validated, the leaf evaluator can be rebuilt — its logic was mostly right (Brown/Sandholm-style k=4 biased continuation strategies, MC rollouts, ICM-adjusted payoffs). The rebuild will be faster because the architecture is already understood.

## Roadmap from here

1. **Fix tree builder** to use `discretize_legal_actions`. Test against real game. Commit.
2. **Rebuild leaf evaluator** (sub-step 2). Test against real leaf states.
3. **Subgame CFR loop** (sub-step 3). Standard regret matching over the tree, with leaf values from the leaf evaluator.
4. **Policy extraction** (sub-step 4). Read out hero's refined action distribution at the root infoset.
5. **SubgamePolicy wrapper** (sub-step 5). Integration with the existing eval pipeline.
6. **Eval against baseline** (sub-step 6). Measure if subgame-augmented dcfr-overnight-3000 beats pure dcfr-overnight-3000 on the standard pool.

Each sub-step is its own session deliverable. None should be attempted without first reading the relevant existing code end-to-end.

## Key lessons from session 12

1. **Read before writing.** Don't assume what's in the codebase. Grep first, read the relevant files end-to-end, then write code. Two architectural mistakes this session were rooted in skipping that step.
2. **Test against the production game, not stand-ins.** Tests that use simpler game definitions can pass while the code is broken for the real game.
3. **Runtime CFR with MC equity is not deployable.** Push/fold work hit this wall. Subgame solver must follow the same lesson: precompute where possible, never run unbounded MC inside the hot path.
4. **ICM-correct training is already in the codebase** via `src/nlhe/icm_returns.py` integrated with `src/nlhe/cfr6.py`. dcfr-overnight-3000 was trained ICM-correct. Don't claim ICM is missing without verifying.

## Files to read end-to-end before touching subgame code

- `src/nlhe/cfr6.py` (especially `traverse_6max` lines 207-417, the existing tree walker)
- `src/nlhe/actions.py` (the `DiscreteAction` enum + `discretize_legal_actions`)
- `src/nlhe/biased_policy.py` (the BiasedBlueprint that leaf eval will use)
- `src/nlhe/icm_returns.py` (ICM transformation already wired into training)
- `scripts/eval_pool.py` (CheckpointPolicy interface — what subgame policy must conform to)

