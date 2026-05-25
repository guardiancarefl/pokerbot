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

**How "safe subgame solving" is realized here (B1c sub-step 2/3).** In this 6-max
ICM setting, Pluribus-style safety is realized by the **`BEST_RESPONSE` leaf mode**,
not by a HUNL-style counterfactual-value (CFV) gadget. At the depth limit each live
opponent independently best-responds among the k=4 biased continuation strategies
(`biased_policy.py`), so hero's refined strategy is solved against an opponent who
adversarially picks the most damaging continuation — structurally the same
worst-case-over-opponent-continuations mechanism Pluribus used, with the bias menu
discretizing what Pluribus's CFV gadget handled continuously. The DeepStack/Libratus
CFV gadget is a 2-player-zero-sum construction (scalar exploitability, root opponent
CFVs) and is **structurally inapplicable** to multiplayer ICM, so it is not
implemented. Empirical confirmation: Stage G measured BR suppressing hero value by
**+3.07σ** vs profile-sampling and moving hero's root policy above the per-deal noise
floor (`docs/sessions/session_18_summary.md`, `docs/SUBGAME_LEAF_DESIGN.md` Q11
Level 2). The sub-step-3 CFR loop therefore runs plain vanilla weighted CFR over the
already-robust leaf values and adds no separate safety gadget (`docs/SUBSTEP_3_DESIGN.md`
Decision 6).

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

Total estimated calendar time: 2-3 months with serious effort, assuming GPU compute is rented in Phase 4+.

## Technology stack

### Engine and ML framework
- **Engine:** OpenSpiel (DeepMind, open source) — has CFR, Deep CFR, PSRO, poker environments built in
- **Phase 1 algorithm implementation:** `open_spiel.python.pytorch.deep_cfr` (reference implementation, thin-wrapped with logging and checkpointing)
- **ML framework:** PyTorch
- **Language:** Python 3.10 (via deadsnakes PPA on Ubuntu 24.04; system Python 3.12 untouched). Pinned to 3.10 because OpenSpiel's officially-tested Python range is 3.7–3.10.

### Compute, by phase

The project runs on different hardware at different phases. The transition between phases is a hardware decision driven by measured throughput needs, not a project goal in itself. Code stays portable; data and checkpoints transit via git for source and via SCP / object storage for large artifacts.

- **Phases 1–3 (development, validation, prototype):** Contabo VPS running Ubuntu 24.04. 12 vCPU AMD EPYC (oversubscribed — shared with other tenants), 48 GB RAM, ~300 GB free disk, no GPU. Sufficient for Leduc Deep CFR, heads-up NLHE prototype, and archetype integration. Throughput from this box is not a benchmark for Phase 4+ planning.
- **Phase 4+ (full blueprint training, league play):** Rented cloud GPU. Provider chosen at the time based on price and availability (Vast.ai, RunPod, Vultr GPU, or similar). Likely an A100, A40, or 4090-class GPU. Rented per-session, stopped when not actively training.
- **Always available as fallback:** Windows 11 host with RTX 3060 Laptop GPU (6 GB VRAM). Currently unusable due to Windows component store corruption blocking WSL2. Could be repaired and brought back in if needed — not a priority since the cloud GPU path is cleaner.

### Project location and version control
- **Working directory:** `~/pokerbot/` on the Contabo VPS.
- **Repo:** github.com/guardiancarefl/pokerbot (private). SSH key auth.
- **Migration model:** when hardware changes (e.g., spinning up a GPU rental for Phase 4), the new instance does `git clone` and a fresh venv install. Large artifacts (model checkpoints, replay buffers) live outside git — either on the persistent disk of the GPU rental or in cheap object storage.
