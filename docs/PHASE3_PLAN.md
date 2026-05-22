# Phase 3 Plan: The Ambitious Path to a Pluribus-Class SNG Bot

**Authored:** 2026-05-22 (end of Session 5)
**Scope:** Phase 3 through Phase 6, roughly 6-10 weeks of focused work.

This document supersedes the original phase plan in ARCHITECTURE.md for Phase 3 onward. The original architecture treated Phases 3-6 as sequential. This plan parallelizes the work across three tracks running concurrently from Week 1, then converges at integration milestones.

## Project goal (sharpened from Session 5)

Build the strongest publicly-known 6-max NLHE SNG bot with correct ICM. Specifically:
- Beat Slumbot in HUNL by 1-5 bb/100 (architectural sanity check)
- Beat each of the 42 bought-bot profiles by 10-30 bb/100 in SNG format
- Achieve 70%+ top-3 finish rate in SNG simulations against archetype-based opponent pools
- Sub-second decision latency via real-time subgame solving
- Withstand a 100k-hand test without exploitation pattern emerging
- Stretch target: plausibly beat Pluribus head-to-head in 6-max NLHE cash via architectural improvements + ICM-correct value function + within-match adaptation

## Architectural commitment

We build a Pluribus-class architecture: blueprint (Deep CFR trained offline) + subgame solving (depth-limited continual resolving at decision time) + within-match opponent modeling (Bayesian belief over continuous archetype representation). With ICM-correct value function for SNG format.

## Three parallel tracks

### Track A: Algorithm + training improvements
Owner-track on Contabo CPU with occasional GPU bursts for validation.

**A1. Linear/Discounted CFR (DCFR) — Week 1**
- Implement DCFR weighting in src/nlhe/solver.py
- Add per-buffer-entry timestamp/iteration tracking
- Weight strategy training loss by iter index (linear) or by configurable exponent (discounted)
- Smoke test against current Phase 2d v2 config to confirm convergence improvement
- Expected outcome: 1.5-3x faster convergence to similar bb/100 baseline, or higher final bb/100 in same iter count

**A2. Hand-engineered archetype framework — Weeks 1-2**
- Continuous parameterization (tightness ∈ [0,1] × aggression ∈ [0,1])
- Sample archetype instances from this 2D space during training
- Behavioral rule sets: preflop range tightness, postflop aggression, sizing patterns, c-bet frequencies
- Used as training opponents alongside self-play in a configurable mix

**A3. OCHS card abstraction investigation — Weeks 2-3**
- Literature review of Opponent Cluster Hand Strength (Johanson 2013 et al.)
- Reference implementation comparing EMD-clustered vs OCHS-clustered abstractions
- A/B comparison: train two identical solvers, one with each abstraction, eval against Slumbot
- Decision point: adopt OCHS for Phase 4 if measurable improvement

**A4. Bought-bot profile integration — Weeks 2-3**
- Identify the 42 bought-bot profile format (text / XML / JSON / binary)
- Write profile loader and runtime player
- Integrate into training opponent pool

### Track B: Subgame solver engineering
Design + initial implementation on Contabo CPU, integration testing requires the Phase 4 blueprint.

**B1. Subgame extractor — Week 2**
- Given an OpenSpiel game state, extract the depth-limited subgame as a new game tree
- Define depth limit policy (e.g., to end of current street, or n-actions ahead)
- Validate via roundtrip: extract → solve → check returned policy matches expectation

**B2. Fast online CFR variant — Weeks 2-3**
- Standard Deep CFR is too slow for sub-second use
- Implement table-based CFR (no networks) for small subgames, possibly with action-abstraction
- Optimization target: solve typical subgame in <500ms on Contabo CPU

**B3. Belief state estimation — Weeks 3-4**
- Given observation history (betting actions so far), compute opponent's likely range at subgame root
- Bayesian update using blueprint as prior + observed actions as evidence
- Handle the multi-opponent case for 6-max

**B4. Leaf value function from blueprint — Week 4**
- At subgame depth limit, use blueprint's strategy network to estimate equity
- Validate values are sane (not negative, not absurdly large)

**B5. End-to-end subgame solving integration — Week 5**
- Plug the trained Phase 4 blueprint into the subgame solver pipeline
- Validate the solver produces strategies, doesn't crash, meets latency budget
- First eval: blueprint + subgame solving vs blueprint alone, head-to-head 1000 hands

### Track C: Within-match opponent modeling
Design work in Weeks 1-2, implementation Weeks 3-4 in parallel with Tracks A and B.

**C1. Continuous archetype representation — Week 1 (design)**
- 2D space: tightness × aggression
- Each opponent has a belief distribution (e.g., Gaussian) over this space
- Update method: Bayesian, with action likelihoods from archetype mixture model

**C2. Bayesian update from actions — Weeks 2-3**
- Action likelihood model: P(action | archetype, infoset)
- Sequential update: P(archetype | actions) ∝ P(actions | archetype) × P(archetype)
- Numerical implementation (variational or particle-based)

**C3. Population priors per stake level — Week 3**
- Default archetype distribution for new opponent: weighted by stake level
- Without cross-match identification (anonymity preserved)

**C4. Policy response from belief — Week 4**
- Bot's policy at each decision = integral over belief distribution of (archetype-specific best response)
- Implementation: sample archetypes from belief, compute responses, average

## Phase 4: ICM-adjusted blueprint training (Weeks 4-6)

Once Tracks A and C are largely complete, Phase 4 trains the full SNG blueprint:

- 6-max NLHE SNG with ICM-adjusted value function
- ICM equity computed per node based on remaining player chip stacks and prize structure
- Training pool mix: self-play (40%) + 42 bought bots (30%) + archetype-instances (30%)
- DCFR algorithm with OCHS abstraction (if A3 confirmed beneficial)
- Larger networks ([1024, 1024]) with 1M+ buffer
- Multi-day GPU training run

Phase 4 produces the blueprint that goes into the subgame solver.

## Phase 5: Integration + first evaluation (Weeks 6-8)

- Plug Phase 4 blueprint into Track B's subgame solver
- Add Track C's within-match opponent modeling
- End-to-end testing: bot plays SNG simulations against archetype pools
- Slumbot HUNL eval (architectural sanity check)
- Closed tournaments against the 42 bought bots
- Track each phase's marginal contribution (blueprint alone, blueprint + subgame, blueprint + subgame + opponent modeling)

## Phase 6: League play and final tuning (Weeks 8-10)

- PSRO-style league play: archive checkpoints, train new generations as best responses to league
- Strength-curve tracking
- Final benchmark vs Slumbot, 42 bought bots, archetype opponents
- 100k-hand validation
- Documentation of final architecture and results

## Risks and mitigations

**Risk 1: DCFR implementation has subtle bugs that aren't obvious in smoke tests.**
- Mitigation: validate against Leduc poker first (where exploitability can be computed exactly), then HUNL.

**Risk 2: Subgame solver is harder than expected (belief estimation in particular).**
- Mitigation: start with HUNL subgames (well-understood from DeepStack literature), extend to multi-opponent only after HUNL works.

**Risk 3: OCHS abstraction doesn't help as much as theory suggests.**
- Mitigation: A/B test before committing. Fall back to EMD with augmented features if needed.

**Risk 4: ICM value function is harder to integrate than planned.**
- Mitigation: implement ICM as a wrapper around terminal-state returns. Test with known-Nash 3-handed pushfold scenarios.

**Risk 5: 6-10 weeks slips to 12-15 weeks.**
- Mitigation: each Phase has explicit deliverables. If a track stalls, the others continue.

## Cost projection

- Track A development: mostly Contabo CPU. ~$0 in marginal cost.
- Track B development: Contabo CPU. ~$0 marginal.
- Track C development: Contabo CPU. ~$0 marginal.
- Phase 2d continuation (current run + occasional eval checkpoints): ~$30-50.
- Phase 4 blueprint training: 5-10 GPU-days on RTX 4090 or PRO 4000 Blackwell. ~$200-500.
- Phase 5 integration testing: occasional GPU bursts for fast simulation. ~$50-100.
- Phase 6 league play: 10-20 GPU-days. ~$300-700.

**Total estimated compute: $600-1400.**

## Success criteria for the project (final)

When all six criteria are met, the project is "done":

1. ✓ Beats Slumbot in HUNL by ≥1 bb/100 over 10,000 hands
2. ✓ Beats each of the 42 bought-bot profiles by ≥10 bb/100 in SNG format
3. ✓ Top-3 finish rate ≥70% in SNG simulations vs archetype opponent pool
4. ✓ Decision latency <1 second per action via subgame solving
5. ✓ No detectable exploitation pattern in 100k-hand test
6. ✓ Within-match opponent modeling demonstrably adds ≥1 bb/100 over blueprint+subgame alone

## Next session (Session 6) priorities

1. Decide whether to continue the Phase 2d GPU training overnight (check ckpt_iter_0200+ next morning)
2. Implement DCFR (Track A1) — first concrete Phase 3 deliverable
3. Begin archetype framework design (Track A2)
4. Begin subgame extractor design (Track B1)
5. Begin opponent modeling design (Track C1)

These five items represent the first 1-2 days of Phase 3 work.
