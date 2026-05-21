# Decision Log

## Format specialization: 6-max NLHE SNG (top-3 equal payout)
**Decided:** 2026-05-21
**Why:** Smaller strategic space than 6-max cash; ICM dynamics provide large exploit edge against humans who play chip-EV instincts; bounded variance; late-game compresses to analytically-solvable push/fold; softer typical populations. Specialized bot likely stronger in its niche with same compute than a generalist would be.
**Alternative considered:** General-purpose 6-max NLHE cash bot.
**Reason rejected:** ~3x larger compute requirement, no specialization advantage, much larger strategic space to cover.

## Engine: OpenSpiel
**Decided:** 2026-05-21
**Why:** Battle-tested implementations of Deep CFR, PSRO, league play. Actively maintained by DeepMind. Has poker environments built in. Reduces "build from scratch" risk.
**Alternative considered:** PokerRL, custom build.
**Reason rejected:** PokerRL is less maintained; custom build adds months of infrastructure work that has no research payoff.

## Runtime environment: WSL2 + Ubuntu 22.04
**Decided:** 2026-05-21
**Why:** OpenSpiel does not officially support Windows native (pip wheels are Linux/macOS only). The local hardware is a Windows 11 machine. WSL2 provides a real Linux environment inside Windows with CUDA passthrough to the GPU, preserving every other architectural decision while avoiding Windows-native build friction.
**Alternative considered:** Dual-boot Linux; Docker Desktop with NVIDIA container toolkit; switch engines to a Windows-native option.
**Reason rejected:** Dual-boot is disruptive and unnecessary. Docker adds interactive-development friction. Switching engines re-opens the OpenSpiel decision and costs months of infrastructure work.

## Hardware: RTX 3060 Laptop GPU (6GB VRAM), defer cloud/upgrade decision
**Decided:** 2026-05-21
**Why:** Local hardware confirmed as RTX 3060 Laptop variant with 6GB VRAM (not the 12GB desktop variant originally assumed). Sufficient for Phase 1-3 development without modification. Phase 4+ will require careful batch sizing and network width tuning to fit within VRAM. Iteration cycles are cheaper on owned hardware than rented. Cloud bursting makes sense only for the final blueprint training run at finer abstraction, and only if needed.
**Alternative considered:** Buy 4090, rent A100 from start.
**Reason rejected:** Premature optimization. Validate pipeline first, then make compute decisions with real throughput data.
**Note:** The 6GB VRAM constraint is a real limitation for Phase 4+ and must be accounted for in network architecture choices.

## Training approach: hybrid (self-play CFR + anonymous opponent diversity + league play)
**Decided:** 2026-05-21
**Why:** Pure self-play converges to Nash within the self-play distribution but can leave the bot vulnerable to styles it never sees. Anonymous diverse training opponents (hand-engineered archetypes + 42 bought-bot behavior generators) force exposure to style variety. League play (PSRO with archived self) provides strength-level diversity and protects against strategic collapse.
**Alternative considered:** Pure self-play; pure imitation learning from bought bots.
**Reason rejected:** Pure self-play risks blind spots against unseen styles; pure imitation copies leaks and lacks theoretical grounding.

## ICM value function in CFR training
**Decided:** 2026-05-21
**Why:** SNG format has equal payouts at top 3 — chip-EV training produces fundamentally wrong strategy because chips above survival threshold have zero marginal value. ICM-adjusted value function is the correct optimization target for this format.
**Alternative considered:** Chip-EV training with post-hoc ICM adjustment at play time.
**Reason rejected:** Post-hoc adjustment cannot fully correct chip-EV-trained policies; the bubble pressure and in-the-money dynamics need to be baked into training.

## Opponent anonymity (core design principle)
**Decided:** 2026-05-21
**Why:** The bot's information state must mirror that of a competent human at an anonymous online table. No persistent identity is maintained for any opponent across matches. No pre-collected real-world opponent data is used in training. Within a single match, the bot observes opponents and adapts; when the match ends, all derived state is wiped.

This is a values-driven decision (robustness and fairness over peak exploitation EV), not purely a technical one. The architectural consequence is that Layer 4 of the original design — persistent per-opponent statistics tracking and population-level priors derived from observed hand history data — is removed entirely and replaced with within-match-only adaptation (see next entry).

**Alternative considered:** Original design with persistent opponent identity, hand-history-informed population priors, per-opponent stat tracking across sessions.
**Reason rejected:** Inconsistent with anonymous-table information state. Adds complexity in service of EV extraction that the project values less than robustness.

## Within-match adaptation: Position 2 (light-to-medium online reads, blueprint-anchored)
**Decided:** 2026-05-21
**Why:** Three positions were considered for within-match adaptation: (1) zero adjustment, pure Nash play; (2) light online statistics nudging subgame solver ranges, anchored to blueprint; (3) full real-time opponent modeling with range estimation and best-response calculation. Position 2 captures most of the practical exploitation EV against varied opponents while remaining robust — if reads are wrong, behavior falls back to unexploitable blueprint play, not to bad play. Position 3 is more theoretically powerful but practically worse: it tries to do too much with too little within-match data, risks overfitting to noise, and competes with subgame solving for the sub-second decision budget. This is the approach Pluribus used.
**Alternative considered:** Pure GTO (Position 1); full real-time opponent modeling (Position 3).
**Reason rejected:** Position 1 leaves too much EV against weak opponents. Position 3 is fragile and compute-expensive on the available hardware.

## Bought-bot profiles remain frozen; opponent strength grows via league play
**Decided:** 2026-05-21
**Why:** The 42 bought-bot profiles serve two roles: training opponent diversity (style variety) and stable benchmark targets. Evolving them would compromise both — benchmarks need to be stable to be meaningful, and 42x parallel evolution would waste compute that's better spent on the main bot. Strength-level diversity in the training pool is instead provided by league play, which adds archived versions of our own bot to the opponent pool over time. This separates style diversity (42 frozen profiles + archetypes) from strength diversity (league archives).
**Alternative considered:** Co-evolving the 42 profiles alongside the main bot.
**Reason rejected:** 42x compute cost, loss of benchmark stability, redundant with league play.

## Scope: training only, no deployment layer
**Decided:** 2026-05-21
**Why:** This is a research/training project. Deliverable is the trained model and the infrastructure that produced it, evaluated offline against benchmarks and in closed environments.
