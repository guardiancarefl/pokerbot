# Track B1: Subgame Solver (Real-Time Refinement)

**Status:** Not started. Design phase opens once Track A3 (card abstraction)
is solid enough to support meaningful subgame solving.
**Estimated:** 3-4 weeks of focused work across multiple sessions.
**Goal:** Bring the bot from "blueprint-only play" to "blueprint + real-time
subgame refinement" — the architectural distinction between a strong CFR bot
(Phase 2d, +31.45 bb/100 vs Slumbot) and a Pluribus-class bot.

## Background

The Phase 2d HUNL bot plays its blueprint policy directly. Every decision is
"look up the infoset bucket, sample from the policy network." This is the
floor for what trained Deep CFR produces.

Pluribus-class bots add a second layer: at decision time, re-solve a
limited-lookahead subgame using the blueprint as a value-substitute at the
leaves. This produces a finer-grained policy for the current situation than
the blueprint alone can express, while staying within real-time decision
budgets.

The technical challenge: in imperfect-information games, states do not have
well-defined values, so leaf substitution as used in chess/Go does not apply
directly. The Brown/Sandholm/Amos (NeurIPS 2018) result is the principled
fix: at leaves, the opponent chooses among k pre-computed strategies, which
makes leaf values well-defined and robust to opponent adaptation.

## What this delivers

A `SubgameSolver` module that, given a current game state and the trained
blueprint, returns a refined action distribution for the current decision.
The solver budgets a configurable amount of CFR iterations (target:
sub-second for casual play, multi-second for evaluation).

Initial scope: HUNL 2-player. The same algorithmic core extends to 6-max in
Phase 4 with more bookkeeping but no new conceptual machinery.

## Algorithm: depth-limited solving with multi-valued states

### High-level

1. At decision time, identify the current public node.
2. Construct the subgame: a limited-lookahead game rooted at the start of the
   current betting round (per Pluribus), terminating at the end of the round.
3. At each leaf of the subgame, the opponent picks from k continuation
   strategies. Hero also picks from k continuation strategies (per Pluribus's
   extension over the original NeurIPS-18 formulation).
4. Solve the subgame with CFR for a fixed iteration budget.
5. Return the policy at the current decision node.

### Constructing the leaf strategy set (the bias approach)

The set of k continuation strategies at each leaf is built once at training
time, not at decision time:

- Strategy 0: the blueprint itself.
- Strategy 1: blueprint biased toward folding (multiply fold probabilities by
  alpha > 1, renormalize).
- Strategy 2: blueprint biased toward calling.
- Strategy 3: blueprint biased toward raising.

Bias factor alpha is a hyperparameter; Brown 2018 used alpha=5 with k=4.

### Why search starts BEFORE the current decision (Pluribus)

Two reasons:
1. Reduces exploitability of unsafe subgame solving by giving opponents a
   chance to deviate within the subgame from blueprint behavior.
2. Allows opponent actions in the current betting round to inform our
   subgame solution rather than treating them as fixed history.

Hero's actions before the current decision are fixed (we already played
them); opponent's actions enter the subgame model just like they would have
been observed in the full game.

## Integration with existing code

### What's reused

- The trained DeepCFR blueprint (current: Phase 2d's ckpt_iter_0100; future:
  whatever blueprint Track A3 + further training produces).
- `InfosetEncoder` (src/nlhe/infoset.py) for bucket lookups and infoset
  feature vectors at subgame nodes.
- The Abstraction (`src/nlhe/abstraction.py`) now that bucket_of is
  deterministic — same hand always maps to same bucket in the subgame.
- OpenSpiel's game representation to construct the subgame tree.

### What's new

- `src/nlhe/subgame.py`: `SubgameSolver` class. Takes a current
  OpenSpiel state, runs limited CFR, returns refined action distribution.
- `src/nlhe/leaf_strategies.py`: code that constructs the k biased
  continuation strategies from a blueprint. Run once per blueprint.
- `src/nlhe/biased_policy.py`: wrapper that applies the bias multipliers to
  a blueprint policy at infoset query time.

### What changes minimally

- `src/nlhe/policy_adapter.py`: the Slumbot bridge currently calls
  blueprint directly. Add an opt-in `--use-subgame-solver` flag and the
  corresponding code path that invokes the SubgameSolver at decision time.

## Sub-phase structure

**B1a — leaf strategy construction.** Implement `biased_policy.py` and
`leaf_strategies.py`. Validate: a fold-biased blueprint folds more often;
a call-biased one calls more often; etc. Unit tests on small example
infosets. No subgame solving yet.

**B1b — subgame tree construction.** Given a current OpenSpiel state,
build the limited-lookahead subgame: root at start of current betting
round, leaves at end of round. Validate: tree has correct number of
nodes, no missing infosets, every leaf is reachable, opponent action
spaces are correct.

**B1c — subgame CFR solver.** CFR loop on the constructed subgame. At
leaves, evaluate the multi-valued state by treating the opponent's
choice over k strategies as an additional decision. Validate: solver
converges on simple subgames; produces sensible exploitability decrease
on Leduc-sized subgames before scaling to HUNL.

**B1d — decision-time integration.** Wire SubgameSolver into the
PolicyAdapter as an optional path. Run a small Slumbot eval (200 hands)
to confirm nothing's broken before launching a full comparison.

**B1e — measurement.** Two evals against Slumbot at matched budget:
blueprint alone (current +31.45 reference) vs blueprint + subgame solver.
Expect meaningful gain; if not, debug.

This is ~5 focused sessions. Comfortable inside the 3-4 week budget.

## Open design questions

1. **Subgame solve time budget.** Pluribus used "sub-second on a 4-core CPU"
   for casual play. Our Contabo CPU is shared/oversubscribed, so per-decision
   time will vary. For Slumbot eval (no time pressure) we can afford
   multi-second solves. Real-time human play is out of scope for the project,
   so this is a measurement-quality knob, not a deployment constraint.

2. **Number of CFR iterations per subgame.** Bias toward more iterations
   given we're not real-time. Start with 1000, scale up if exploitability is
   high.

3. **Card abstraction at subgame leaves.** The subgame's leaves require
   evaluating the blueprint's policy. The blueprint uses Abstraction's
   bucket_of, which is now deterministic. Same bucket lookup machinery used
   in the subgame, no special handling needed.

4. **Continual re-solving vs solve-from-blueprint.** Continual re-solving
   (DeepStack) re-solves at every decision using the previous subgame's
   solution as the next subgame's input. Pluribus only solves at the start
   of each betting round and reuses within the round. Pluribus's approach
   is simpler and we should start there. Continual re-solving is a
   potential B2 extension.

5. **6-max extension.** The algorithmic core is the same. The main 6-max
   complication: opponent strategy sets become more expensive (4 strategies
   per opponent x 5 opponents = 20 leaf policies to bias and evaluate).
   Pluribus handled this. Real concern but not a blocker for 2-player B1.

## Risks

1. **Subgame solver convergence in our codebase.** OpenSpiel's CFR
   implementation is mature; using it directly is the safe path. Writing
   our own depth-limited CFR for HUNL would be a multi-week side project.
2. **Bias-factor tuning.** alpha=5 worked for Brown 2018 on HUNL with
   their abstraction. May need re-tuning for our abstraction.
3. **Subgame size at deep stacks.** With 200bb starting stacks the
   subgame tree at the start of a betting round can be large. Phase 2d
   trained at 200bb (20000 chips / 100 BB). May need to constrain action
   abstraction within the subgame to keep it tractable.

## What this does NOT include

- Continual re-solving (DeepStack-style; potential B2).
- Within-match opponent modeling (Track C1).
- ICM-adjusted subgame solving (Phase 4 when we move to 6-max SNG).
- 6-max subgame solving (Phase 4).

## References

- Brown, Sandholm, Amos (2018). *Depth-Limited Solving for Imperfect-
  Information Games.* NeurIPS-18. arXiv:1805.08195. Multi-valued states.
- Brown, Sandholm (2019). *Superhuman AI for multiplayer poker.* Science.
  Pluribus. Multi-strategy leaves for all players, subgame rooted before
  current decision.
- Moravcik et al. (2017). *DeepStack: Expert-level artificial intelligence
  in heads-up no-limit poker.* Science. Continual re-solving alternative.
- Brown, Bakhtin, Lerer, Gong (2020). *Combining Deep Reinforcement Learning
  and Search for Imperfect-Information Games.* NeurIPS-20. ReBeL — combines
  value-network-substituted-leaves with depth-limited solving. Potential
  reference for later refinement once B1 baseline is in.
