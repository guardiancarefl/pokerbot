# Track C1: Within-Match Opponent Modeling (Layer 4)

**Status:** Not started. Design phase opens after B1 (subgame solver) is in place — C1 plugs into the subgame solver, so it needs B1 as its substrate.
**Estimated:** 2-3 weeks of focused work across multiple sessions.
**Goal:** Add Layer 4 of the four-layer architecture — within-match opponent observation + blueprint-anchored range nudging — turning the bot from "plays optimally against an unknown opponent" into "plays optimally against an unknown opponent and refines its read as the match progresses."

## Constraints inherited from prior decisions

These are non-negotiable and shape the entire design (see DECISIONS.md):

1. **Opponent anonymity.** No persistent identity for any opponent across matches. No real-world hand-history training data. The bot's information state mirrors a competent human at an anonymous online table.
2. **All within-match state wiped at match end.** When a match concludes (we bust, win, or finish), every derived statistic, every observation buffer entry, every range estimate is deleted. The next match starts blind.
3. **Position 2: blueprint-anchored.** Adaptation is *light-to-medium* online statistics nudging the subgame solver's opponent range estimates. If reads are noisy or wrong, behavior falls back to the unexploitable blueprint, never to bad play.
4. **No exploitation EV at the cost of robustness.** This is a values-driven design — we accept lower peak EV against weak opponents to guarantee no catastrophic failure mode against any opponent. The bot is robust by construction.
5. **Sub-second decision budget shared with subgame solver.** C1 cannot compete with B1 for compute. Online statistics are cheap-to-update; the work happens at the C1-feeds-into-B1 boundary, not in dedicated long-running modules.

## What C1 actually does

At any decision point during a match, the subgame solver (B1) needs an estimate of the opponent's range — the probability distribution over hands the opponent could hold given the public action history. The blueprint provides a *prior* on this range derived from equilibrium analysis. C1 nudges that prior toward what the observed opponent actually seems to be doing.

Three concrete mechanisms:

### 1. Online statistics per anonymous seat

A flat per-seat record of basic poker statistics over the current match. Indexed by Seat 1, Seat 2, etc. — never by persistent identity. For each seat, track:

- **VPIP** (voluntarily put money in pot): fraction of hands where the seat invested chips preflop.
- **PFR** (preflop raise): fraction of hands where the seat raised preflop.
- **Aggression frequency by street**: fraction of decisions on flop/turn/river that are bets-or-raises (vs checks-or-calls).
- **Fold-to-bet by street**: fraction of times this seat folded facing a bet on each street.
- **Showdown win rate**: fraction of showdowns this seat won. Sample-limited but useful late in matches.
- **Average bet sizing relative to pot**: rough sizing tells.

These are cheap to update (O(1) per observed action) and form the observation substrate the next two layers consume.

### 2. Range biasing for the subgame solver

The subgame solver (B1) takes an opponent range estimate as input — the range from the blueprint's equilibrium. C1 produces a *bias multiplier* on that range based on the online stats:

- If the seat's VPIP is much higher than blueprint-typical, the bot weights the opponent's range toward "looser" — more hands than blueprint equilibrium would put there.
- If the seat's aggression frequency is much higher than blueprint, weights toward more bluffs in the range.
- If fold-to-bet is high, weights the seat's range toward "calling with weak made hands less often" — they fold those.

The multiplier is *anchored to the blueprint as a safety floor*: the maximum bias factor is bounded (per Pluribus, typical factors are 1.5x-3x at most), so even a heavily-biased read produces a range still recognizable as approximately-Nash.

### 3. Decay and confidence

Online stats become more confident as the match progresses (more samples). C1 weights the bias multiplier by a confidence factor that grows with sample count — early in a match the bias is near 1.0 (blueprint dominates), later it grows toward the bound.

Sample-count thresholds (initial values, tune empirically):
- < 20 observed actions: confidence 0, blueprint only.
- 20-100 actions: confidence ramps linearly from 0 to 0.5.
- 100-300 actions: confidence ramps from 0.5 to 1.0.
- 300+ actions: confidence 1.0, max bias applied.

## What C1 explicitly does NOT do

- Estimate opponent's exact hand probabilities online (Position 3 in DECISIONS.md). Too noisy, too compute-expensive, fights with B1 for decision budget.
- Train a network on observed opponent behavior. Would require labeled data, conflicts with anonymity, and overfits to noise.
- Persist anything across matches. State wipe at match boundary is structural, not configurable.
- Categorize opponents into archetypes online. The bot already trains against archetypes (Track A2); within-match it observes statistics, not labels.
- Compute best response to estimated opponent strategy. Best-response computation is fragile against noisy estimates; blueprint anchoring is safer.

## Integration with B1

The clean interface: B1's subgame solver currently takes a hand `state` and produces a refined policy. It uses the blueprint to estimate opponent ranges at every infoset. C1 extends this by passing a `range_bias_fn(opponent_seat, infoset) -> bias_multiplier` callable into B1's subgame construction.

B1's subgame node construction calls this for each opponent infoset; if no bias function is provided (e.g., very early in a match), B1 behaves identically to its standalone behavior. C1 is strictly additive at the API surface.

## Module structure

- `src/nlhe/within_match.py`: the central module.
  - `class MatchObserver`: records actions per seat throughout a match, exposes `get_stats(seat) -> SeatStats`.
  - `class SeatStats`: VPIP, PFR, aggression-by-street, fold-to-bet-by-street, etc.
  - `class RangeBiaser`: given a `SeatStats` and a confidence factor, produces a `range_bias_fn` callable suitable for B1.
  - `wipe_match_state()`: explicit wipe call invoked at match boundaries by the match-orchestration layer.
- `src/nlhe/policy_adapter.py`: extended to instantiate and wipe a `MatchObserver` per match, feed observed actions in, and pass the resulting `range_bias_fn` to B1's subgame solver.
- `tests/test_within_match.py`: per-seat stat tracking, decay/confidence behavior, range-biasing correctness against synthetic opponents, and the critical no-persistence test (observers from two matches share no state).

## Sub-phase structure

**C1a — observer + stats.** Build `MatchObserver` and `SeatStats`. Wire it into the Slumbot eval loop so it observes every Slumbot hand. Verify: stats update correctly, no persistence across hands when wipe is called. No range biasing yet — this is pure observation.

**C1b — range biaser.** Build `RangeBiaser` with the multiplier-and-confidence design. Validate against synthetic opponents (a maniac-like or nit-like simulated opponent should produce predictable bias multipliers). No B1 integration yet.

**C1c — B1 integration.** Plug the `range_bias_fn` into B1's subgame construction. Validate on toy subgames that the resulting opponent range estimates reflect the bias.

**C1d — Slumbot eval with C1+B1.** Run a Slumbot eval with `MatchObserver` enabled and `RangeBiaser` feeding B1. Compare bb/100 against blueprint-alone and blueprint+B1-alone baselines.

**C1e — bought-bot eval.** The bought-bot profiles (Track A2 already, full integration Phase 3) provide a controlled environment where opponent "style" is known. C1 should produce measurably better results against the style-extreme profiles (maniac, nit) than blueprint+B1 alone, because that's where online observation has the most to learn.

**C1f — wipe-test.** End-to-end test that no information leaks across match boundaries: run two consecutive matches, verify the second match's observer has zero data from the first.

This is ~4-5 focused sessions inside the 2-3 week budget.

## Open design questions

1. **Bias factor bounds.** Pluribus reportedly used 1.5x-3x ranges. Our values-driven robustness preference may push us toward the lower end (1.5x-2x). Decide empirically during C1b-c.
2. **What counts as "an action" for sample counting?** Every public decision the opponent makes, or every street-completing action? Affects how fast confidence ramps. Probably "every public decision" — more samples, smoother ramp.
3. **Per-street vs aggregate biases.** Should the bias multiplier vary by street (more bias on river where stakes are higher) or be uniform? Start uniform, add per-street if measurements justify.
4. **6-max scaling.** Five opponents instead of one. Memory and update cost are linear, no algorithmic issue. The compute cost per decision is the worry — five seat-bias-multipliers feeding into B1's subgame at every infoset. Same conceptual machinery, but B1's per-decision cost goes up. Real concern but not a C1 blocker; handle when porting to 6-max in Phase 4.

## Risks

1. **Online stats noisy in short matches.** SNG matches can be short (~30-100 hands). Stats may not stabilize before the match ends. Mitigation: low confidence early, blueprint dominant. Acceptable failure mode.
2. **Opponent gameplay style shift mid-match.** Humans bluff-frequency varies with mood, fatigue. Online stats can't catch this fast enough. Acceptable — blueprint anchoring is the safety net.
3. **Bias direction wrong against weird opponents.** A LAG opponent might look passive in a small early sample. Real loss possibility, but bounded by the bias factor cap.

## What this does NOT include

- Per-opponent persistent profiles (forbidden by anonymity decision).
- Multi-match learning, league play, or PSRO (those are Phase 5).
- Deep-RL-trained opponent modeler (forbidden by the no-real-world-data principle).
- ICM-adjusted opponent modeling (Phase 4 — same conceptual machinery, ICM-aware bias factors).

## References

- Brown, Sandholm (2019). *Superhuman AI for multiplayer poker.* Science. Pluribus. Section on online opponent adaptation describes the bias approach.
- DECISIONS.md, entries "Opponent anonymity (core design principle)" and "Within-match adaptation: Position 2."
- ARCHITECTURE.md, Section "Layer 4: Within-match adaptation."
