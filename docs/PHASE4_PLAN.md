# Phase 4: 6-max NLHE SNG Blueprint with ICM

**Status:** Not started. Design phase opens after Session 7.5/8 close, in or before Session 9.
**Estimated:** 4-6 weeks of focused work across multiple sessions.
**Goal:** Port the HUNL training/eval stack to 6-max, add ICM-adjusted value functions for the SNG payout structure, train the blueprint, validate against the 42 bought-bot profiles and SNG simulations.

This is the largest single-phase undertaking in the project. It's where everything we've built so far stops being "infrastructure validation" and becomes the actual research bot.

## Why 6-max + ICM is genuinely different

HUNL training has converged-to-deployed pipelines that work today. 6-max with SNG ICM has none of that established. The differences from HUNL aren't cosmetic:

1. **Game tree explodes.** 6 players, each with their own decision points and stack-state, multiplies branching at every betting node. Pluribus's blueprint training was 12,400 CPU-hours on hardware in a 2019-era datacenter. Our compute budget is orders of magnitude smaller. Action abstraction, card abstraction, sampling efficiency, network architecture all need re-tuning, not just re-running.

2. **Convergence theory weakens.** Nash equilibrium in 2-player zero-sum games is uniquely defined; CFR converges to it. In N≥3 player games (and especially in SNG where the payoff structure isn't even chip-EV), there's no unique Nash — there are families of "reasonable" strategies, and CFR-style algorithms converge to some local equilibrium that depends on initialization. The behavior we want is "robust play against varied opponents," not "exact Nash." Self-play CFR + archetype training (already in A2) becomes more important, not less.

3. **ICM changes value semantics fundamentally.** In chip-EV, the value of a chip is constant. In ICM, the value of a chip depends on every player's stack and the payout structure. A player on the bubble (4 remaining, top 3 paid) playing for 30bb has very different value-per-chip than a player on the chip lead with the same 30bb. This is the entire point of the SNG specialization but it's also where the training pipeline has to be carefully refactored.

4. **No more Slumbot benchmark.** Slumbot is HUNL-only. For 6-max we either need to (a) train a separate 6-max bot as our own evaluation opponent, (b) measure against the 42 bought-bot profiles, (c) self-play tournaments, or (d) some combination. The eval infrastructure is its own multi-session piece of work.

## Subphase structure

### Phase 4a — Game representation and action abstraction

**COMPLETE** (commits accaf14, 2b80578, f1c393c).

src/nlhe/game_strings.py builds OpenSpiel universal_poker game strings parametrically. PokerGameConfig dataclass accepts num_players (2-10), starting_stack, big_blind, small_blind. Convenience constructors: hunl_200bb, hunl_20bb, six_max_200bb, six_max_sng. Action abstraction (src/nlhe/actions.py) unchanged from HUNL: 7 discrete actions (fold, call, 0.33pot, 0.66pot, pot, 2pot, allin) plus the action-translation layer for opponent off-tree sizings.

All 5 HUNL configs migrated from hardcoded game_str to structured fields. Backward compat: legacy game_str still works as fallback.

End-to-end verified: 6-max games load in OpenSpiel (num_players=6, observation_tensor_shape=[116], max_utility scales with N*starting_stack), a random-action 6-max game completes in 25 steps with zero-sum returns. 15 dedicated tests in tests/test_game_strings.py, 83 → 112 total tests passing after Phase 4b also landed.

**Important finding:** observation_tensor_shape changes from [108] (HUNL) to [116] (6-max). +8 features for +4 players. This is the next concrete refactor target — Phase 4d (InfosetEncoder for 6-player state).

### Phase 4b — ICM value function

**Math module: COMPLETE** (commit 02663e3). src/nlhe/icm.py + 29 passing tests.

The bot targets two Ignition 6-max formats:
  - **Double Up**: top-3 each get 2x buy-in (equal payouts to ITM). Bubble at 4 active.
  - **Standard**: top-2 paid 65/35 of prize pool. Bubble at 3 active. No degenerate ITM phase.

Per ARCHITECTURE.md the value function takes the payout vector as input and computes per-player equity via Malmuth-Harville. Below ~15bb effective, push/fold is analytically solvable and we switch to precomputed Nash ICM lookup tables. For Double Up specifically, at 3 players with equal payouts, all remaining chips have zero marginal EV — switch to "fold non-premium" mode. Standard mode doesn't have this degenerate ITM phase.

**Solver integration: PENDING.** The math module returns per-player equity given stacks and payouts. The integration step rewires the solver's terminal utility from chip-EV to ICM-EV. That requires the 6-max solver refactor (4d/4e) to be in place first, since the existing HUNL solver assumes 2-player and chip-EV — there's nothing to integrate ICM into yet at the multi-player level.

### Phase 4c — Card abstraction for 6-max (Session 12)

The HUNL EMD-on-equity-histograms approach uses 2-player equity calculations. 6-max needs range-vs-range-vs-range, which is computationally heavier and produces different histogram shapes. Options:

1. **Reuse HUNL abstraction.** Treat range-vs-one-opponent equity as the histogram even in 6-max. Loses some information but no new code. Pluribus actually used a 2-player abstraction for 6-max.
2. **Range-vs-2-random-opponents.** Computationally tractable, more representative.
3. **Range-vs-N-1-random-opponents.** Most representative, slowest.

Recommendation: start with (1) per Pluribus precedent. The 6-max blueprint trains on top of whatever abstraction we ship; if eval shows it's the weak link, revisit.

### Phase 4d — 6-max infoset encoder (Session 13)

Currently `src/nlhe/infoset.py` encodes HUNL state (2 player stacks, 2 hole-card sets unseen). 6-max needs:
- All 6 player stacks
- Position information (button, blinds, etc.)
- Who's still in the hand vs folded
- Pot odds with multiple opponents

This is a meaningful refactor. The encoder produces the input tensor to the strategy and advantage networks. Network input dimension changes. Trained HUNL networks cannot be reused — 6-max blueprint starts from scratch.

### Phase 4e — 6-max self-play loop + trajectory generation (Sessions 14-15)

The DCFR solver in `src/nlhe/solver.py` is mostly player-count-agnostic. Per-player advantage networks generalize from 2 to 6. The trajectory generator needs to handle 6-player rotation, opponent sampling from the (now larger) opponent pool (self-play + archetypes + bought-bots + league archives).

This is where the 42 bought-bot profiles get integrated as per the original Phase 3 plan. They've been waiting for a 6-max training environment to plug into.

### Phase 4f — Initial training run (Sessions 16-17)

Rent GPU (likely RTX 4090 or A100 from RunPod / Vast.ai depending on pricing at the time). Start a multi-day training run with the ICM-adjusted value function, 6-max self-play + archetype + bought-bot mix, modest network size.

Validation gates during training:
- Iter 1: trajectories complete, no crashes
- Iter 100: per-player advantage networks training (loss decreasing)
- Iter 500: strategy network producing varied policies across diverse infosets
- Iter 1000: bb/100 vs random baseline positive
- Iter 5000+: bb/100 vs archetype mix positive, vs bought-bots positive

If any gate fails, debug and either restart or continue. Don't burn GPU credits on hopeful runs.

### Phase 4g — 6-max evaluation harness (Session 18)

Build a 6-max tournament simulator. Plays N tournaments to completion, tracks finishes, computes:
- Top-3 finish rate (the headline goal metric, target 70%+)
- ROI in chip-EV and ICM-EV
- Performance breakdown by stack-depth phase (early, mid, late, bubble, ITM)
- Head-to-head vs each archetype and each of the 42 bought-bots

Without Slumbot we need our own internal benchmark. The bought-bot profiles are the standardized benchmark per the original architecture.

### Phase 4h — Eval and decision (Session 19)

Run a full eval suite. Three real outcomes:

1. **Blueprint beats archetypes + bought-bots by target margins.** Phase 4 done. Move to Phase 5 (league play) and Phase 6 (within-match adaptation = C1).
2. **Blueprint loses or marginally beats.** Debug. Likely culprits: abstraction granularity, action set, network capacity, training duration. Iterate.
3. **Blueprint plays sensibly but doesn't meet targets.** Same as (2) but the bar is "improve, not redesign." More training, possibly larger network, possibly add bias toward archetype-aware play.

Conservative estimate: 4-6 weeks of focused work end-to-end. Could be longer if we hit foundational bugs (we found four in Session 7.5).

## What gets reused from HUNL work

The infrastructure isn't wasted:

- `src/nlhe/actions.py` (DiscreteAction enum + translation) — used directly.
- `src/nlhe/abstraction.py` (post-Session 7.5: deterministic bucket lookup, retrofit script) — likely reused with caveats from Phase 4c.
- `src/nlhe/solver.py` (DCFR loop, advantage net training, strategy net training, checkpoint) — minor refactor for 6-player state.
- `src/nlhe/archetypes.py` (Track A2) — directly applicable as 6-max training opponents; A2 was designed player-count-agnostic.
- `src/nlhe/biased_policy.py` (Track B1a) — directly applicable as leaf strategies for 6-max subgame solving.
- Training rig, config patterns, run-directory format, tests/ scaffolding — all reusable.

What doesn't transfer:
- `src/nlhe/infoset.py` — needs major refactor for 6-player state.
- `src/nlhe/policy_adapter.py` and `src/nlhe/slumbot_client.py` — HUNL-specific; not relevant for 6-max.
- Phase 2d trained checkpoint — HUNL only. The 6-max blueprint is a new training run from scratch.

## ICM correctness — the critical technical risk

ICM math is well-published but easy to get wrong in code. The Malmuth-Harville formula recursively computes finish-position probabilities, which means small bugs compound. Risks:

1. **Bubble dynamics asymmetry.** Short-stack at 4-handed bubble has very different optimal play than chip-leader. The ICM value differential is huge and easy to miscompute.
2. **Below-15bb push/fold tables.** Need to match published values exactly. If our tables disagree with Sklansky-Chubukov or similar by even small amounts, the late-game play is wrong.
3. **3-player ITM degenerate case.** Equal payouts mean fold-everything-non-premium is the optimal strategy. Easy to miss if not explicitly modeled.

Mitigation: extensive testing of `src/nlhe/icm.py` against published numbers before any training relies on it. Make this a session of its own (Session 11 per the subphase plan).

## Compute budget

Pluribus blueprint: 12,400 CPU-hours equivalent. Modern training is more efficient (DCFR vs vanilla CFR, GPU acceleration, better abstraction). Realistic for a research-grade 6-max blueprint:

- **Smoke training run:** RTX 4090 or A100 for ~24-48 hours. $20-50 in GPU time. Validates the pipeline.
- **First real blueprint:** RTX 4090 for 5-7 days. $50-100 in GPU time. Produces something measurable.
- **Refined blueprint:** A100 for 7-10 days. $200-400 in GPU time. Production-grade artifact.

Total Phase 4 GPU budget: $300-600 across 2-3 training runs. Inside the Vultr credit + reasonable additional spend.

## Open questions

1. **Eval opponent for Phase 4.** Bought-bot profiles (when integrated) provide a standardized benchmark. But we have no live human comparison until Phase 6+. Self-play tournaments give us a relative metric (improving over training) but not absolute strength.

2. **B1 (subgame solver) integration timing.** B1_PLAN.md sketched B1 as HUNL-first. With the reprioritization toward 6-max, the question is whether to:
   - (a) Train the 6-max blueprint first, add B1 as a separate later phase.
   - (b) Build B1 into the 6-max blueprint from the start.
   
   (a) is faster to a working bot. (b) produces a stronger bot. Decide in Session 9 once we see how Phase 4a goes.

3. **42 bought-bot profile integration.** Original Phase 3 plan, blocked on having a 6-max training environment. Phase 4e is the natural integration point.

4. **GPU provider for Phase 4.** RunPod RTX 4090 (per Phase 2d), Vultr A40/L40S (uses up the $93 credit), Vast.ai (cheaper but more variance). Decide at the time based on current pricing.

## What this does NOT include

- **Phase 5 (league play).** Self-play CFR + bought-bot opponents during Phase 4 training are enough. League play (PSRO with archived self) is a Phase 5 strength-improvement layer.
- **Phase 6 (C1 within-match adaptation).** The blueprint is the prior; C1 nudges it at decision time. Separate phase.
- **Real-money deployment.** Out of scope for the project; the deliverable is the trained bot and the infrastructure that produced it.
- **Human evaluation.** Requires a platform decision (private home games, real-money sites with legal/policy nuance, simulation against human-style bots). Phase 7 if at all.

## References

- Brown, Sandholm (2019). *Superhuman AI for multiplayer poker.* Science. Pluribus — the canonical 6-max NLHE bot.
- Malmuth, Harville. ICM formulation. Standard in poker literature.
- DECISIONS.md, "ICM value function in CFR training" entry. Established the requirement.
- ARCHITECTURE.md, "Specialization decisions for SNG format." Constraint set.
