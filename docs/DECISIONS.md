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

## Runtime environment: WSL2 + Ubuntu 22.04 (SUPERSEDED in Session 2)
**Decided:** 2026-05-21
**Superseded:** 2026-05-21 (Session 2) — see "Runtime environment: Contabo VPS" below.
**Why (at the time):** OpenSpiel does not officially support Windows native (pip wheels are Linux/macOS only). The local hardware is a Windows 11 machine. WSL2 provides a real Linux environment inside Windows with CUDA passthrough to the GPU, preserving every other architectural decision while avoiding Windows-native build friction.
**Alternative considered:** Dual-boot Linux; Docker Desktop with NVIDIA container toolkit; switch engines to a Windows-native option.
**Reason rejected (at the time):** Dual-boot is disruptive and unnecessary. Docker adds interactive-development friction. Switching engines re-opens the OpenSpiel decision and costs months of infrastructure work.
**Why this was superseded:** WSL2 install failed at the DISM step with error 14098 (Windows component store corruption). DISM /RestoreHealth and sfc /scannow could not repair the store enough to enable VirtualMachinePlatform. The remaining repair path (ISO-source DISM, or in-place Windows reinstall) would cost more time than just using a clean Linux machine that was already available.

## Hardware: RTX 3060 Laptop GPU (6GB VRAM), defer cloud/upgrade decision (SUPERSEDED in Session 2)
**Decided:** 2026-05-21
**Superseded:** 2026-05-21 (Session 2) — see "Hardware: Contabo VPS (CPU) + future cloud GPU" below.
**Why (at the time):** Local hardware confirmed as RTX 3060 Laptop variant with 6GB VRAM (not the 12GB desktop variant originally assumed). Sufficient for Phase 1-3 development without modification. Phase 4+ will require careful batch sizing and network width tuning to fit within VRAM. Iteration cycles are cheaper on owned hardware than rented. Cloud bursting makes sense only for the final blueprint training run at finer abstraction, and only if needed.
**Alternative considered:** Buy 4090, rent A100 from start.
**Reason rejected (at the time):** Premature optimization. Validate pipeline first, then make compute decisions with real throughput data.
**Why this was superseded:** The Windows host became unavailable due to the same component store corruption that blocked WSL2. Rather than repair Windows, switched to existing Contabo VPS for development.

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

## Runtime environment: Contabo VPS (Ubuntu 24.04)
**Decided:** 2026-05-21 (Session 2)
**Supersedes:** "Runtime environment: WSL2 + Ubuntu 22.04" above.
**Why:** WSL2 install on Windows 11 failed at DISM with error 14098 (component store corruption). Standard repair (StartComponentCleanup, RestoreHealth, sfc /scannow) did not enable the required features. Continuing the Windows repair would have required ISO-source DISM repair or an in-place Windows reinstall — both costly in time for a project that doesn't depend on the Windows host. An existing Contabo VPS (Ubuntu 24.04, 12 vCPU AMD EPYC, 48GB RAM, ~300GB free) was already paid for and available. Switching took ~5 minutes of SSH versus an unknown-length Windows repair.
**Note on Python version:** Ubuntu 24.04 ships Python 3.12 as system default. OpenSpiel's officially-tested Python range is 3.7–3.10. Installed Python 3.10 via deadsnakes PPA to use as the project's interpreter; system Python 3.12 untouched.
**Alternative considered:** Continue Windows repair via ISO-source DISM or in-place reinstall; rent a Vultr instance (the user has credit there); use Vast.ai or RunPod from day one.
**Reason rejected:** Windows repair is open-ended time on a non-project problem. Vultr would have worked but costs credit we'd rather preserve for GPU training later. Vast.ai / RunPod likewise — better saved for when GPU compute actually matters. The Contabo box was already paid for and idle for this project's purposes.

## Hardware: Contabo VPS (CPU) for Phase 1–3, rented cloud GPU for Phase 4+
**Decided:** 2026-05-21 (Session 2)
**Supersedes:** "Hardware: RTX 3060 Laptop GPU" above.
**Why:** Contabo box has no GPU, but Phase 1–3 don't need one. Leduc Deep CFR (Phase 1) is small enough that Python/CFR overhead dominates network compute — CPU is fine. Heads-up NLHE prototype (Phase 2) benefits from GPU but isn't blocked without one. Phase 3 (archetype + bought-bot integration) is CPU-bound on trajectory generation. By the time Phase 4 (full ICM-adjusted blueprint training) starts, we'll have real throughput numbers to size GPU rental correctly, and we'll rent on whichever provider has the best price at that time (Vast.ai, RunPod, Vultr GPU, etc.).
**Note:** The RTX 3060 Laptop on the Windows host is not gone — it could be brought back into the picture if the Windows component store gets repaired later. Not a priority. Cloud GPU is cleaner anyway and matches the long-term shape of the project.
**Alternative considered:** Get a GPU instance from day one to "build momentum."
**Reason rejected:** Pre-Phase-1 throughput is dominated by Python overhead and CPU-bound trajectory generation. Renting a GPU now wastes credit on cycles that won't be used. Rent when measurements show it's needed.

## Phase 1 implementation: OpenSpiel reference Deep CFR, not custom
**Decided:** 2026-05-21 (Session 2)
**Why:** Phase 1's goal is pipeline validation — confirm that the train-eval loop works end-to-end on a known game (Leduc) and converges to known Nash. Reimplementing CFR adds risk and time without research payoff at this stage. `open_spiel.python.pytorch.deep_cfr` is a working reference implementation maintained by the OpenSpiel team. We thin-wrap it with logging, checkpointing, and exploitability evaluation rather than reimplementing the algorithm.
**Alternative considered:** Custom Deep CFR implementation in PyTorch from scratch for learning value.
**Reason rejected:** Learning value of reimplementation is real but better captured later (Phase 2+) when the deviations from textbook Deep CFR (ICM value function, action abstraction) demand custom code anyway. Phase 1 should de-risk infrastructure, not algorithms.

## Git workflow: two-commit migration
**Decided:** 2026-05-21 (Session 2)
**Why:** Session 2's changes are large — runtime, hardware, project location, Python version. Committing them on top of the original Windows-era docs in one merged commit would obscure what changed. Instead: first commit lands the verbatim Session-1 state (all docs as they existed at end of Session 1). Second commit updates STATUS, SESSION_LOG, DECISIONS, ARCHITECTURE to reflect Session 2. Anyone reading the git history can see the two states cleanly.
**Alternative considered:** Single combined commit; rewrite history later if needed.
**Reason rejected:** Single commits lose information. History rewrites are error-prone and shouldn't be a planned step.
