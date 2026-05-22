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

## GPU provider for Phase 2d: RunPod Community Cloud RTX 4090
**Decided:** 2026-05-21 (Session 3)
**Why:** Phase 2d needs a GPU for the first time in the project — coarse-abstraction HUNL training won't converge on the Contabo CPU in tolerable wall-clock time. Three constraints drive the choice. First, the planned [256, 256] networks fit comfortably in 24GB VRAM, so a 4090 is sufficient and an A40/A100/L40S buys nothing useful for Phase 2. Second, Deep CFR's bottleneck on small networks is CPU-bound trajectory generation, not GPU throughput, so paying for a bigger or faster GPU mostly buys idle GPU time. Third, this is the first cloud GPU run in the project, so cheap-tier hours are valuable for the inevitable PyTorch/CUDA/checkpoint debugging cycle. RunPod Community Cloud 4090 at $0.34/hr is the cheapest reliable option — same price tier as Vast.ai but with a curated host pool instead of an open marketplace, and per-second billing that matches an iterative dev workflow. Resumable training (planned for Phase 2b) makes interruption risk on Community Cloud a minor cost rather than a run-killer.
**Alternative considered:** Vast.ai RTX 4090 interruptible (~$0.29/hr); Vultr L40S or A40 (~$1.67–1.71/hr, $93 credit available); rent today vs defer to Phase 2d.
**Reason rejected:** Vast.ai is marginally cheaper but its open-marketplace model produces more variable host quality (a 95% uptime host interrupts roughly one in twenty 10-hour jobs); the small savings vs RunPod Community aren't worth the variance for a first GPU run. Vultr's A40/L40S cost ~5x RunPod for compute Phase 2 doesn't need — the 48GB VRAM isn't used, and Vultr doesn't stock RTX 4090s. Spending the $93 Vultr credit now buys ~55 hours on L40S vs ~270 hours on RunPod for the same dollar; the credit is more valuable held for Phase 4 (multi-day ICM blueprint training, where dedicated no-interrupt hardware actually matters) or as a cheap CPU bolt-on for Phase 3 if hosting the 42 bought-bot opponents alongside the trainer on Contabo gets tight.
**Action:** RunPod account created in Session 3. No pod rented yet — Phase 2a and 2b run on Contabo CPU. First rental triggered at start of Phase 2d.
**Open follow-up:** Check whether the $93 Vultr credit has an expiry date. If it expires before Phase 4 (realistically 6–10 weeks out), revisit and consider using it for a Phase 3 helper instance instead of letting it lapse.

## Card abstraction: EMD on equity histograms, not OCHS or simpler
**Decided:** 2026-05-21 (Session 3)
**Why:** PHASE2_SKETCH flagged this as a Session 3 decision: EMD vs OCHS vs raw-equity bucketing. Going with EMD on equity histograms over Monte Carlo runouts, ~200 buckets per postflop street, 20 buckets preflop. The choice is values-driven (Reading 2 of the three framings considered in session): EMD is the gold-standard technique in the poker AI literature (Pluribus, Libratus), and even though the *module* will be partially superseded in Phase 4 (6-max abstraction will be a different module due to range-vs-range-vs-range equity), the *technique* compounds — implementing EMD properly now means the team has done it once when the harder Phase 4 abstraction needs it. The cost is real (~2-3x the code of OCHS, ~6.8 min training run on Contabo CPU vs. an estimated ~2 min for OCHS), but it's a one-time cost. Inspection of the trained preflop and flop buckets confirms EMD is doing real strategic clustering: different surface hands with similar equity-histogram shapes (e.g., "drawing dead on coordinated board" hands) correctly land in the same bucket regardless of which specific cards are involved.
**Alternative considered:** OCHS (Opponent Cluster Hand Strength, K-means on 8-dim equity-vs-opponent-cluster vectors); raw mean-equity quantile bucketing.
**Reason rejected:** OCHS is simpler and was the analytical recommendation when optimizing for "fastest path to Slumbot bb/100 measurement." Raw-equity bucketing is incorrect (hands with same mean equity can have very different playstyle — flush draw vs top-pair-weak-kicker both ~50% but play differently). The session deliberately chose the harder/more thorough technique knowing the trade-off.
**Sample sizes:** preflop=169 hands × 400 runouts × 50 histogram bins → k=20 medoids. Postflop streets=1500 sampled (hand, board) combos × 200 runouts × 50 histogram bins → k=200 medoids. Total training time: 6.8 minutes on Contabo CPU. Embarrassingly parallel; multiprocessing could reduce this to ~1-2 min when needed.
**Followup deferred to Phase 2d:** If Slumbot evaluation suggests abstraction quality is the bottleneck (rather than other Phase 2 components), retrain with more sampled hands per street (5000+) before increasing bucket count. Bucket count of 200 is generous for 1500 sampled hands; sample coverage is the more likely limiting factor.

## PolicyAdapter for Slumbot evaluation: bridge module at `src/nlhe/policy_adapter.py`
**Decided:** 2026-05-22 (Session 4)
**Why:** Phase 2c shipped the Slumbot client and eval script, but no interface existed between a trained `DeepCFRSolver` checkpoint and the `eval_vs_slumbot.py` harness. `PolicyAdapter` is that interface. It reconstructs a fresh OpenSpiel state by replaying the hand from `new_initial_state()` (forcing hero's hole cards, dealing arbitrary opponent hole cards, forcing board cards, walking the Slumbot action history), encodes the resulting infoset via the same `InfosetEncoder` used during training, runs the strategy network for the hero seat, samples a `DiscreteAction` over legal actions, and translates the result back to a Slumbot wire token. Design constraints: **eager init** (fail-fast on bad config / checkpoint mismatch at adapter construction, not at the first Slumbot hand); **loud assertions** on state-reconstruction failures (no graceful degradation — if replay can't reach the decision point, that's a bug that has to surface); and an **explicit warning when training-stack ≠ eval-stack** so plumbing-test runs against a smoke checkpoint at the wrong depth don't get confused for production eval. The plumbing-test pattern this enables — train a small smoke model at the cheap stack, build the adapter against it, run a 20-hand eval to surface protocol bugs *before* paying for serious compute — caught the per-street/per-hand bet-translation bug (see next entry) within minutes.
**Alternative considered:** Lazy init (load checkpoint on first `choose_action`); graceful fallback on bad action translation (route a default fold or check on translation error).
**Reason rejected:** Lazy init would fail at the first Slumbot hand rather than at startup, mid-eval — costly because the failure point is far from the configuration mistake. Graceful fallbacks on action translation would mask protocol bugs exactly like the per-street/per-hand bug this adapter surfaced; we want loud failures during plumbing testing, not silent skipping.

## Wire-format translation: Slumbot `b<N>` (per-street) ≠ OpenSpiel `universal_poker` int N (per-hand)
**Decided:** 2026-05-22 (Session 4)
**Why:** The two conventions are identical preflop (no prior commitment exists) but diverge as soon as any chips have entered the pot on a prior street. Empirically verified by direct OpenSpiel probe: at a flop decision node with both players having committed 300 chips preflop, OpenSpiel's `legal_actions()` minimum-bet action is **int 400** (300 prior + 100 new street min-bet), and `action_to_string(400)` returns `'player=0 move=Bet400'` — i.e., the bet integer is "total chips committed by the actor across the **whole hand**." Slumbot's wire `b<N>` is the per-street total. The translation helpers `slumbot_token_to_openspiel_action` and `openspiel_action_to_slumbot_token` take a `prior_streets_committed_by_actor: int = 0` kwarg and the adapter's replay loop maintains a per-player dict refreshed at each postflop street transition by parsing the `[Money: X Y]` field of `state.information_state_string()`. The default value of 0 keeps preflop callers and existing unit tests unmodified. This is locked-in protocol knowledge that any future bot-vs-protocol-X integration (other ACPC servers, future heads-up poker benchmarks) should reuse — the failure mode (5 of 20 hands rejected with "Bet size too big") was inscrutable without tracing a specific postflop example.
**Alternative considered:** Treat the two conventions as identity (the original assumption, which survived design review); patch only Slumbot-token decoding by inferring prior commitment from the OpenSpiel state's pot field.
**Reason rejected:** Identity was empirically false (the plumbing test rejected 5/20 hands). Inferring from pot is unreliable because the pot mixes both players' commitments and doesn't separately track per-player per-street contributions; the `[Money: X Y]` field is the clean per-player source, and refreshing it once per street boundary is both correct and cheap.
