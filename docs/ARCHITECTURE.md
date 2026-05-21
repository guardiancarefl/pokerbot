# Architecture

## Core design principle: opponent anonymity

The bot plays anonymous tables. It has no persistent identity for any opponent across matches and no access to pre-collected real-world opponent data. Within a single match, it observes opponents' actions and adapts. When the match ends (we bust, win, or the match concludes), the observation buffer is wiped. The next match starts blind. This mirrors the information state of a competent human at an anonymous online table.

This principle constrains the architecture in specific ways, called out below.

## The four-layer stack

### Layer 1: Game representation and abstraction
- **Card abstraction:** Earth Mover's Distance (EMD) clustering of hands by equity distribution. Target ~200 buckets per street.
- **Action abstraction:** discretized bet sizes — {check/fold, call, 0.33pot, 0.66pot, 1pot, 2pot, all-in}. Action translation for opponents' off-tree sizings.
- **Information set encoding:** hole card bucket, board features, betting history, stack sizes, position, street indicator, pot odds, range-vs-range equity features.

### Layer 2: Blueprint strategy (Deep CFR)
- Advantage network: predicts counterfactual regret for each action at a given infoset.
- Strategy network: predicts action distribution; this is the deployed policy.
- Linear/Discounted CFR variants for faster convergence.
- External sampling for trajectory generation.
- Reservoir buffer for average strategy training.
- **ICM-adjusted value function**: terminal utilities computed via ICM equity in remaining prize pool, not chip count. This is the key departure from standard cash-game CFR.

### Layer 3: Real-time refinement (subgame solving)
- Continual resolving: at decision time, re-solve a finer-grained subgame using blueprint values at the boundary.
- Safe subgame solving (Pluribus-style) to avoid exploitable inconsistencies.
- Budget: target sub-second per decision, more during evaluation.

### Layer 4: Within-match adaptation (Position 2)
**Scope: within current match only. All state wiped at match end.**

- Within-match observation buffer: logs every observed action by every opponent in the current match. Indexed by anonymous seat (Seat 1, Seat 2, ...), never by persistent identity.
- Simple online statistics per anonymous seat: aggression frequency, fold-to-bet by street, showdown tendencies, sizing patterns.
- These statistics nudge opponent range estimates inside the subgame solver for the remaining hands of the current match.
- All adaptation anchored to the blueprint as a safety floor — if reads are noisy or wrong, behavior falls back to unexploitable play, not to bad play.
- **Strict no-persistence rule:** when a match ends, the buffer and all derived statistics are deleted. There is no opponent identity database, no cross-match memory, no profile lookup.

## Specialization decisions for SNG format

### Stack-depth focus
Training emphasizes effective stacks in the 8-80bb range, where SNG gameplay actually occurs. Deep-stack postflop trees (150bb+) are de-prioritized.

### ICM payout structure handling
Value function takes payout vector as input. For 6-max top-3-pays format: payout vector is [0.333, 0.333, 0.333, 0, 0, 0] across finish positions. ICM equity computed per node based on remaining player chip stacks.

### Late-game compression
Below 15bb effective, switch from learned policy to precomputed Nash ICM push/fold lookup tables. Massively reduces compute requirements for the most common late-game decisions.

### In-the-money degeneracy
With 3 players remaining and equal payouts, all remaining chips have zero EV. Policy switches to "fold-everything-non-premium" mode. Trivial to handle, easy to miss if not explicitly modeled.

## Training opponent pool (all anonymous)

The bot improves through training against a diverse pool of anonymous opponents:

- **Self-play CFR:** the bot plays itself.
- **Hand-engineered archetypes:** maniac, nit, station, LAG, and similar style profiles. These provide behavioral diversity during training. At deployment, no opponent comes with a tag.
- **Bought-bot profiles (42 of them):** used as anonymous training opponents who happen to play in their characteristic ways. The bot learns to handle the variety they generate without ever knowing it is facing a specific bot.
- **League play (PSRO):** archived versions of the bot rotate into the opponent pool as training progresses.

**No real-world hand history data enters training.** No data scraped from human play, no observed-opponent population priors, nothing that would require identifying real people. Training opponent diversity comes entirely from self-play, hand-engineered archetypes, and the 42 bought-bot behavior generators.

## Training pipeline phases

### Phase 1: Infrastructure validation (Leduc poker)
- OpenSpiel installation and Deep CFR loop validation
- Verify convergence to known Nash equilibrium on toy game
- Duration: 2-3 days

### Phase 2: Heads-up NLHE prototype
- Scale to real poker with coarse abstraction
- Evaluate against Slumbot via API
- Establish baseline training pipeline
- Duration: ~1 week including iteration

### Phase 3: Archetype and bought-bot integration
- Integrate hand-engineered archetypes (maniac, nit, station, LAG)
- Plug in the 42 bought-bot profiles as anonymous training opponents
- Validate that training against style diversity produces a more robust bot than self-play alone
- Duration: 1-2 weeks

### Phase 4: Specialized 6-max SNG blueprint
- Full ICM-adjusted Deep CFR training run
- Card and action abstraction tuning
- Main blueprint training: ~2 weeks of compute + iteration

### Phase 5: League play / PSRO
- Iteratively add archived bot versions to opponent pool
- Train new generations as best responses to expanding league
- Track pairwise win rate matrix and Nash mixture weights
- Duration: 2-4 weeks (5-10 league iterations)

### Phase 6: Within-match adaptation layer
- Build the within-match observation buffer and online statistics
- Wire stats into subgame solver range estimation
- Validate adaptation improves win rate against varied training opponents
- Verify the no-persistence rule: confirm that no opponent state survives match boundaries
- Duration: 1-2 weeks

Total estimated calendar time: 2-3 months on RTX 3060 Laptop with serious effort.

## Technology stack
- **Engine:** OpenSpiel (DeepMind, open source) — has CFR, Deep CFR, PSRO, poker environments built in
- **ML framework:** PyTorch
- **Language:** Python 3.10+
- **Hardware:** RTX 3060 Laptop GPU (6GB VRAM) local for development and main training — NOTE: this is the 6GB laptop variant, not the 12GB desktop variant. Batch sizes and network widths in Phase 4+ must account for this.
- **Runtime environment:** WSL2 + Ubuntu 22.04 on Windows host (OpenSpiel does not officially support Windows native)
- **Optional later:** Cloud GPU burst (Vast.ai / RunPod) for final blueprint at finer abstraction
- **Optional later:** VPS (12 vCPU) as parallel self-play generator
