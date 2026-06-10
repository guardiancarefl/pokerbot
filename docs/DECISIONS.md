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

## Project goal sharpened: strongest publicly-known SNG bot, Pluribus-class
**Decided:** 2026-05-22 (Session 5)
**Why:** Original goal in PROJECT_OVERVIEW.md was broadly "specialized 6-max NLHE SNG bot." Session 5 discussion clarified the actual ambition: not just specialized, but the strongest publicly-known SNG bot. Concrete targets: beat Slumbot in HUNL by 1-5 bb/100, beat each of the 42 bought-bot profiles by 10-30 bb/100 in SNG format, achieve 70%+ top-3 finish rate in SNG simulations, sub-second decisions via subgame solving, withstand 100k-hand test without exploitation. Stretch: plausibly beat Pluribus head-to-head in 6-max cash via architectural improvements + ICM-correct value function + within-match adaptation.

Drivers for the sharpened goal:
- At today's compute prices, replicating Pluribus's training compute is ~$125, not millions. The hard part is algorithmic correctness, not compute.
- AI-assisted engineering (this collaboration) partially replaces the 6-researcher team that Pluribus had, especially for translating published papers to working code.
- SNG format with ICM has no published Pluribus-equivalent. This is genuinely empty territory in the literature — we'd be the first to build a Pluribus-class bot for the SNG-with-ICM problem.
- Within-match continuous opponent modeling (Bayesian updating, integration over belief distribution) is genuinely beyond what Pluribus did.

**Estimated total project: 6-10 weeks of focused work, $500-2000 in GPU compute.**

**Alternative considered:** keep the goal modest ("specialized SNG bot that beats most humans"). Reasonable, more certain, but less interesting. The math on compute and the AI-assisted engineering capability changed the calculus.
**Reason rejected:** the gap between "modest" and "ambitious" is actually engineering hours and care, not compute. With the goal sharpened, every implementation decision becomes more careful, which is the right pressure for the project.

## Phase 3 expanded to three parallel tracks (DCFR + subgame solver + archetype modeling)
**Decided:** 2026-05-22 (Session 5)
**Supersedes:** original ARCHITECTURE.md Phase 3 (archetype/bought-bot integration only).
**Why:** The original sequential phase plan was Phase 3 (archetype) → Phase 4 (full ICM blueprint) → Phase 5 (league play) → Phase 6 (within-match adaptation), with subgame solving deferred or implicit. Session 5 reanalysis: subgame solving is not a "polish on top of finished blueprint" — it's a different architectural choice that shapes the blueprint's role. Building it in parallel with Phase 3 means it's ready when the Phase 4 blueprint completes, instead of being 3-4 weeks of additional sequential work after Phase 4.

**Track A (Algorithm + training):**
1. Implement Linear/Discounted CFR (DCFR) — published improvement, weights later iters more in average-policy. 1.5-3x faster convergence in literature. ~1 day implementation.
2. Hand-engineered archetype framework (maniac, nit, station, LAG, TAG) parameterized by tightness × aggression, used as training opponents.
3. Investigate OCHS card abstraction — replaces EMD clustering with opponent-cluster-based equity. Fixes the AA = QQ = TT collapse confirmed in Session 5.

**Track B (Subgame solver engineering):**
1. Subgame extractor (given OpenSpiel state, define depth-limited subgame).
2. Fast online CFR variant for sub-second solving at decision time.
3. Belief state estimation for opponent ranges at subgame root.
4. Leaf value function integration with blueprint.

**Track C (Within-match adaptation — moves up from Phase 6):**
1. Continuous archetype representation (2D tightness × aggression, not categorical).
2. Bayesian updating from observed actions.
3. Population priors per stake level (no cross-match identification, anonymity preserved).
4. Policy response as integration over belief distribution.

**Alternative considered:** keep subgame solving and within-match adaptation in their original later phases.
**Reason rejected:** subgame solving in particular has long ramp-up time and benefits from being designed alongside the blueprint that feeds it. Sequential ordering means more total weeks. Parallel ordering compresses timeline by 2-3 weeks.

## Bigger buffer is the lever, not bigger network
**Decided:** 2026-05-22 (Session 5)
**Why:** Phase 2d GPU validation confirmed that the [64,64] CPU and [512,512] GPU runs both plateau at similar loss when buffer is fixed at 100K. When buffer expanded to 500K, the [512,512] run pushed past to a meaningfully lower plateau (strategy loss ~0.85 vs ~1.00). Final Slumbot evaluation: +31.45 baseline-adj bb/100 (vs CPU's -14.8 = +46 improvement). Bigger network alone produced no measurable gain; bigger network + bigger buffer was the actual upgrade.

This finding propagates forward: Phase 4's ICM blueprint training should use 1M+ buffer with the network sized to fit it (probably [1024, 1024]), not the other way around. Compute budget allocation should favor more traversals/iter and larger buffers over deeper networks.

**Alternative considered:** continue testing deeper / wider networks at current 100K buffer.
**Reason rejected:** empirical evidence in this session showed buffer is the binding constraint. No reason to spend GPU hours on a hypothesis that's been disconfirmed.

## DCFR: simplified single-exponent form (Track A1)
**Decided:** 2026-05-22 (Session 6)
**Why:** Brown & Sandholm 2019 (Discounted CFR) defines three separate exponents — α discounts positive regrets, β discounts negative regrets, γ discounts the strategy average. Implementing all three faithfully in Deep CFR means treating advantage-net and strategy-net training differently *and* splitting advantage-net targets by sign. That's three knobs to tune and three places for the implementation to be subtly wrong, in service of a Phase 3 capability whose role for us is convergence speed-up — not research-grade DCFR reproduction.

The simplified form: one exponent governs both nets. `cfr_variant="linear"` is exponent=1 (every sample weighted by iteration of origin, divided by current iter). `cfr_variant="discounted"` exposes the exponent as a configurable knob. Vanilla CFR remains the default and the regression baseline.

Single exponent captures the main mechanism — late-iteration samples contribute more than early-iteration samples — and that mechanism is what produces the convergence speedup in the published results. Splitting α from β from γ adds tuning surface without changing the core dynamic.

**Alternative considered:** Full three-exponent DCFR per the paper, with per-sample positive/negative regret split on the advantage net.
**Reason rejected:** Three times the implementation surface, three times the tuning surface, and the final SNG bot doesn't need paper-grade DCFR — it needs faster blueprint convergence. Revisit if measurements show single-exponent DCFR is leaving meaningful convergence speed on the table.

## DCFR backward compatibility: refuse non-vanilla resume from pre-DCFR checkpoints
**Decided:** 2026-05-22 (Session 6)
**Why:** Old checkpoints (the Session 5 GPU run included) don't carry per-sample iteration tags in their buffers. Resuming `cfr_variant="linear"` from one would have to invent iter values for every existing buffer entry — and any default is wrong (all-old underweights real training; all-current overweights stale samples). The error path is clean: refuse the load, point the user at either resume-vanilla or start-fresh. The Session 5 checkpoint isn't worth approximating around — Phase 3 starts a new training run anyway.
**Alternative considered:** Default missing iters to 1 (treat as oldest) with a warning; default missing iters to current iter (treat as newest) with a warning.
**Reason rejected:** Both options silently corrupt the weighting math in ways the loss curves wouldn't necessarily flag. A loud refusal at load time is cheaper than a quiet degradation across training.

## Archetype training opponents (Track A2): data-derived thresholds + designed aggression
**Decided:** 2026-05-22 (Session 7)
**Why:** Track A2 needed five hand-engineered training opponents (NIT, TAG, LAG, STATION, MANIAC) to provide style diversity beyond self-play. Two ways to define them:
- Hand-picked equity thresholds ("nit folds below 0.78") — fast to write but unanchored. Initial drafts of this used made-up numbers that would have caused nit to fold AA preflop given the EMD abstraction's actual bucket equities. Caught by data-first design.
- Data-derived quantile thresholds ("nit plays the top 15% of hands") — derived from the empirical bucket-equity distribution. Robust to the bucket-equity table's actual shape and self-correcting against the EMD ordering errors documented in STATUS.

Chose the data-derived approach. Each archetype gets per-street play quantiles. The actual equity threshold for "fold below quantile X" is computed from a one-shot population sample of 5000 preflop hands + 2000 per postflop street. The result lives in runs/archetype_design/bucket_equity_analysis.json and is regenerated by scripts/analyze_bucket_equity.py.

Aggression is a separate dial governing "given I'm in the pot, bet/raise vs call/check." Tightness is data-derived; aggression is a designed parameter because there's no labeled-action dataset to derive it from — and per the opponent anonymity principle, there never should be. Aggression values (NIT 0.25, TAG 0.65, LAG 0.85, STATION 0.20, MANIAC 0.95) come from poker convention and are flagged as such in the code.

**Alternative considered:** Hand-picked absolute equity thresholds; full ML-trained policy per archetype.
**Reason rejected:** Hand-picked thresholds silently broke against the EMD bucket equities (would have folded AA). ML-trained archetypes per profile would require labeled human-action data, which the project doesn't have and doesn't want under opponent anonymity. Decision tables on data-derived quantiles get most of the benefit at none of the cost.

**Side effect:** Archetypes auto-adapt to abstraction changes. When A3 swaps EMD for OCHS, scripts/analyze_bucket_equity.py reruns against the new abstraction, the JSON regenerates, archetype behavior shifts accordingly — no code change to archetypes.py needed.

## Archetype opponents do not write to the strategy buffer
**Decided:** 2026-05-22 (Session 7)
**Why:** Deep CFR's strategy buffer absorbs the bot's *own* current policy at opponent nodes during self-play, used to train the strategy net (the deployed policy). When the opponent is an archetype, those decisions are *not the bot's* — they're the archetype's. Writing the archetype policy to the strategy buffer would teach the bot to imitate the archetype. A bot that learned to play like a maniac in some range of infosets is a worse bot, not a more diverse one.

The fix: at opponent nodes, if self._current_archetype is set, skip the strategy buffer write. The archetype still drives the action so the bot's traverser-side learning sees archetype behavior in the opponent's moves — exactly what we want for style diversity. The advantage net (regret learning) is unaffected because regrets are computed at traverser nodes, where the bot is making decisions and the opponent's response is just a sampled action.

Verified by smoke test: at archetype_mix=1.0 with 4188 opponent decisions across 20 trajectories, zero strategy buffer writes occurred. Pre-patch behavior at archetype_mix=0.0 is numerically identical to before A2 landed.

**Alternative considered:** Write archetype policies to the strategy buffer with a flag/weight indicating they're archetype-driven.
**Reason rejected:** Adds complexity and a weighting knob with no clear answer on what the weight should be. The clean separation (archetype decisions are not training data for the strategy net) is conceptually correct and easier to reason about.

**Known consequence:** at archetype_mix=1.0 the strategy buffer never fills, so the strategy net can't train. Recommended range for production: 0.2-0.7. Documented on TrainConfig.archetype_mix.

## A3 finding: Abstraction.bucket_of() is non-deterministic (lookup-side, not training-side)
**Discovered:** 2026-05-22 (Session 7.5)
**Why this matters:** Same (hero, board) call to bucket_of() returns different bucket IDs across calls. The function does ~30 Monte Carlo equity-rollout simulations on every call to compute a query histogram, then snaps to the nearest medoid. The sampling variance in those rollouts is enough to flip the nearest-medoid answer when medoids are close together in EMD space. At k=20 (the current abstraction) most hands stably hit "their" bucket because medoids are spread far apart. At k=169 (Option 1's lossless preflop) the medoids are tightly packed and the noise flips the answer.

Concrete evidence: at k=169 preflop, AA mapped to bucket 12 in one call. JJ also mapped to bucket 12. On a second pass through the same probe set: AA still 12 but KK became 4, QQ also 4, with the same reported equity 0.8443 for both. Two distinct hands collapsing to the same bucket with the same equity reading is impossible if the lookup is deterministic and the abstraction is genuinely lossless.

This means: (1) more buckets without fixing the lookup gives more granular noise, not better play. Option 1 as originally designed cannot deliver. (2) Phase 2d's HUNL training run was silently subject to bucket noise, though at k=20 the noise was small enough not to dominate. (3) KrwEmd (Option 4) would inherit the same bug — it also uses Monte Carlo at lookup time.

**Alternatives considered:** ignore it and accept the noise; treat the noise as an implicit regularizer.
**Reason rejected:** silent non-determinism in a foundational primitive is the wrong place to "let it ride." Two identical infosets producing different bucket assignments mean the network sees two different infosets and tries to learn two different policies for the same actual situation. Hard to bound how badly this corrupts training; easier to just fix the lookup.

**Fix (deferred to Session 8):** make bucket_of() deterministic. Verified path:
- Preflop: precompute the bucket for all 169 isomorphism classes once, store in a dict keyed by canonical hand representation. Lookup is O(1) and exact.
- Postflop: same idea — abstraction stores a canonical-hand -> bucket lookup table built at training time using exact equity. Query-time lookup is a dict, not a distance comparison.

**Crucial follow-up finding (same session, stronger than originally documented):** verified the noise empirically at runouts=200, 800, 2000, and 5000. At every level, multiple pocket pairs and high-card hands flip buckets across trials with fresh rng. Even 5000 runouts (25x production) does not stabilize KK or QQ on the k=169 abstraction. The medoid distances between adjacent buckets are smaller than the irreducible MC noise floor at any reasonable runout budget. "More runouts" does not solve this. The previous claim that "k=20 noise was below medoid separation" was also disproven: at runouts=200 on the original k=20 abstraction, AKs flipped between buckets 18, 10, 16 across three trials. The noise is foundational, not granularity-dependent.

This means: Phase 2d HUNL training was meaningfully noisy at the bucket-assignment layer. The +31.45 bb/100 result is genuine but achieved despite this. Algorithm robustness is higher than I would have estimated.

A3 cannot meaningfully compare abstractions until the lookup is deterministic — otherwise we are comparing measurement noise, not abstraction quality. Session 8 starts with the deterministic-lookup redesign, not with new abstraction implementations.

Once fixed, an additional small win: retrain the existing k=20 abstraction's *lookup table* (not the medoids) deterministically, rerun Slumbot eval. Expect the result to be slightly better than +31.45 because the bot now sees consistent buckets at decision time. This is a free measurement on whether the deterministic-lookup fix alone moves bb/100, before any new abstraction algorithm is implemented.

## A3 update: preflop deterministic lookup landed (Session 7.5 late close)
**Date:** 2026-05-22 (Session 7.5, late close)
**Status of the non-determinism finding above:** fixed for preflop, pending for postflop.

Commits f274c6f (infrastructure) and ae2a1e7 (trainer integration) added an optional `preflop_lookup: dict[str, int]` field on `StreetAbstraction`. `bucket_of()` uses it as a deterministic O(1) fast path when present; old pickles without it fall back to the unchanged MC histogram path. The trainer builds the dict from k-medoids labels during preflop training.

End-to-end verified at k=169: 5 trials × 5 different rng seeds × 50 runouts = same bucket every time for AA, KK, QQ, AKs, 72o. Before the change, the same probes returned 3-4 different buckets each.

Adjacent fix that came out of the same investigation: `_kmeans_plus_plus_init` was sampling with replacement (commit a22af38). The "lossless" k=169 preflop trainer had been silently producing only 168 distinct HoleClass strings, with 87o duplicated and J7o missing. Now provably correct on the lossless case.

**What's next (Session 9):** postflop deterministic lookup. Harder problem — (hero, board) tuples aren't enumerable. Likely path: deterministic rng seeded from `hash(canonical(hero, board))` inside `compute_hand_histogram`, so same query always produces same histogram. This fixes the noise but leaves the lossy-distance-metric question open. Acceptable for A3 to proceed; a deeper exact-equity refit would be a Phase 4 polish.

**Outstanding measurement opportunity:** once postflop determinism lands, retrain the original k=20 abstraction with the new deterministic-lookup machinery (no algorithm change), rerun Slumbot eval. Phase 2d's +31.45 bb/100 was achieved against a noisy lookup; deterministic lookup alone may move bb/100 measurably before any new abstraction algorithm is introduced. This is a free measurement and should be the first datapoint in the A3 comparison harness.

## A3 update: retrofit succeeds end-to-end, four foundational bugs documented
**Date:** 2026-05-23 (Session 7.5/8 close, post-eval)
**Result:** Eval C (retrofit abstraction + Phase 2d checkpoint vs Slumbot, 1000 hands) measured +78.35 bb/100, baseline-adjusted. Comparison points: Eval A (noisy original) +15.05 in the same session; Phase 2d's historical eval was +31.45. The deterministic-lookup retrofit is worth ~+50 bb/100 to the trained bot.

### Four foundational bugs found in this session (chronologically)

1. **bucket_of() Monte Carlo non-determinism.** Same (hero, board) call returns different buckets across calls. Verified irreducible at runouts 200, 800, 2000, 5000 — medoid distances smaller than MC noise floor. Fixed via deterministic seeding from hash(canonical(hero, board)) on postflop path (commit 5333a4a), and via the preflop_lookup dict for preflop (commits f274c6f, ae2a1e7).

2. **kmedoids sampling with replacement.** _kmeans_plus_plus_init used rng.choices(range(n), weights=probs) without zeroing already-picked indices, so duplicate medoid picks were possible. At k=n=169 the "lossless preflop" trainer produced only 168 distinct HoleClass strings (87o duplicated, J7o missing). Fixed via short-circuit when k==n and probability zeroing of picked indices (commit a22af38).

3. **Fresh-retrain bucket-id instability across training runs.** k-medoids has random init; the same algorithm and config produces different bucket IDs across runs. Phase 2d's trained network is indexed by the original training's bucket IDs. Eval B (fresh deterministic retrain) regressed to -16.75 bb/100 because preflop bucket IDs changed for 7/12 probe hands. Fixed via the retrofit script (commit 129dda7) which adds preflop_lookup to existing abstractions in-place, preserving original bucket IDs.

4. **bucket_of() suit-dependence on canonically-equivalent literals.** AsKs and AcKc are the same strategic class but the MC sampler draws from different remaining-deck card sets, producing different histograms and potentially different bucket IDs. The retrofit canonicalizes to one literal per HoleClass via hole_class_to_cards() and stores that bucket — same canonical class always returns the same bucket regardless of which suit-permutation the bot is dealt. (Postflop suit-permutation is genuinely strategy-relevant via board interactions, so no analogous fix needed there.)

### Why the retrofit works where fresh retrain didn't

Same bot, same abstraction medoids, only the *lookup path* changed:
- Fresh retrain (Eval B): same algorithm + config produced *different* bucket IDs for ~58% of preflop hands. The bot's network learned policies for the original bucket IDs; remapped IDs mean the network sees the wrong bucket for those hands. -16.75 bb/100.
- Retrofit (Eval C): preserved all original bucket IDs (medoid_histograms unchanged, medoid_hands unchanged), added only a canonical-class -> bucket-id lookup table derived from the original abstraction's MC modal answers. The bot sees the same bucket IDs at decision time that it saw during training, just consistently instead of with noise. +78.35 bb/100.

### Implications

- **The Phase 2d bot's effective skill is higher than +31.45 was capturing.** Lookup noise at decision time was suppressing its actual quality by ~+50 bb/100.
- **Future training runs benefit automatically** because ae2a1e7 has the trainer populate preflop_lookup from labels.
- **Track A3 KrwEmd / Option 4 / comparison harness work is now DE-PRIORITIZED.** The current k=20 abstraction with retrofit produces +78.35 bb/100; further A3 algorithm work has diminishing returns vs. the bigger pieces missing (B1, C1, 6-max port, ICM).
- **For 6-max SNG: the next priority should be either B1 implementation or 6-max port** — both of which add capability the current bot lacks, vs A3 further work which polishes a layer that's already working.

## Target payout structures: Ignition 6-max Double Up + Standard
**Date:** 2026-05-23 (Session 8)
**Why:** Original PROJECT_OVERVIEW.md described the project as "6-max SNG with top-3 of 6 finishers each receiving 33% of the prize pool" — implicitly equal-split top-3. Session 8 clarified the actual target rooms (Ignition specifically) and found that Ignition's 6-max formats are:

1. **Double Up** (top-3 paid, each gets 2x buy-in). Equal in-the-money payouts. Matches what was originally called "triple-up" in older docs but Ignition reserves "Triple Up" for a 9-handed format where 3 of 9 get 3x buy-in.

2. **Standard** (top-2 paid, ~65/35 split of prize pool). 3rd through 6th get nothing. Default at most rooms for traditional 6-max SNGs.

The bot trains/plays both. The mode is configurable; the same trained network can in principle play either if it's been trained on both, or we train two specialists if cross-mode transfer turns out to be weak (likely — the strategic shapes differ).

Strategic differences:
- **Double Up**: bubble at 4 active. Degenerate ITM phase at 3 active (equal payouts → all marginal chip EV is zero → fold non-premium). Below-15bb push/fold tables differ from Standard because the ITM ceiling is fixed.
- **Standard**: bubble at 3 active. No degenerate ITM phase (strict 1st > 2nd payout ordering). Late-game pressure concentrates on 3rd-place avoider/seeker.

**Alternative considered:** Top-3 paid 50/30/20 of total pool (PokerStars-style). This was assumed in earlier docs but is not actually a structure Ignition offers in 6-max. Kept as a function (`sng_payouts_6max`) for backward compat with a DeprecationWarning.

**Reason rejected:** The bot's target rooms include Ignition; the bot has to play the structures those rooms actually offer. Training on a hypothetical 50/30/20 structure would produce a bot whose ICM-aware play is wrong for the real games.
## CFR6MaxContext: API extension over the Session 9 prompt's literal signature
**Decided:** 2026-05-23 (Session 9, Phase 4e.3b)
**Why:** The Session 9 prompt sketched `traverse_6max(state, traversing_player, policy_nets, abstraction, encoder, rng)` — a 6-arg positional signature. The literal form is insufficient for what 4e.3b actually has to do:

  - ICM-adjusting terminal returns requires `starting_stacks` and `payouts`. Neither is present in the sketched signature, and neither lives on any of the existing args.
  - DCFR-future-compatibility requires tagging each regret sample with the current iteration. The training loop in 4e.3c needs to write `ctx.iteration` into reservoir entries; a recursive function that doesn't know its iteration can't do that.
  - `abstraction` is accessible via `encoder.abstraction`. Keeping it as a separate positional arg has the recursive call lie about its dependencies (the function uses `encoder`, never `abstraction` directly).

Final signature: `traverse_6max(state, traversing_player, ctx, rng, depth=0)` where `ctx` is a `CFR6MaxContext` dataclass bundling `policy_nets`, `encoder`, `starting_stacks`, `payouts`, `iteration`, `max_depth`. This both honors the spirit of the sketch (the dependencies are exactly the same) and keeps the recursion clean for the training loop's hot path.

**Alternative considered:** Honor the prompt's literal 6-arg form with `starting_stacks` / `payouts` / `iteration` added as keyword-only arguments.
**Reason rejected:** 8+ args between positional and keyword-only get noisy on every recursive call site. The context-bundle pattern is the standard fix for "these deps don't change across a single traversal." Same pattern is used in PyTorch dataloaders, in DeepMind's OpenSpiel solvers, and in most production CFR code.

## 6-max regret normalization: not divided by starting_stack
**Decided:** 2026-05-23 (Session 9, Phase 4e.3b)
**Why:** The HUNL solver in `src/nlhe/solver.py` normalizes regrets by `starting_stack` before writing to the buffer: `regrets = (values_per_action - ev) * legal_mask / max(self.cfg.starting_stack, 1)`. This was added because chip-EV regrets at 200bb stacks ran into the thousands; without the divide, MSE losses are O(chip²) = O(10⁶) and gradients are ill-conditioned for [64,64] networks.

For 6-max with ICM-EV, the equivalent quantities are bounded by the payout structure:
  - Double Up payouts `[2.0, 2.0, 2.0]`: max per-player equity = 2.0 buy-ins.
  - Standard payouts (65/35): max per-player equity = 3.9 buy-ins.

Counterfactual values inherit this bounded scale through every internal-node backup (weighted sums and differences preserve equity-space interpretation). Regrets are differences between counterfactual values, also bounded — in practice, in the range [-4.0, +4.0] for the worst case under Standard payouts, and typically much tighter. They are already on O(1) scale.

Adding a `/ cfg.starting_stack` divide on top would shrink regrets to ~0.0001 to 0.003 in chip units. MSE losses would be O(10⁻⁶) and gradients vanishingly small. Net effect: training stalls.

So: 6-max regret samples are added to the buffer in their natural equity-space scale, no normalization. Documented in `src/nlhe/cfr6.py`'s module docstring at point 5.

**Alternative considered:** Match HUNL's pattern verbatim and divide by `starting_stack`.
**Reason rejected:** Wrong scaling problem. If 6-max ICM MSE training actually shows pathology in practice (gradients exploding or vanishing on real runs), re-introduce the divide — but only with measured evidence that it's needed.

## 6-max blueprint training: minimum-viable first cut
**Decided:** 2026-05-23 (Session 9, Phase 4e.3c)
**Why:** Four enhancements that exist in the HUNL pipeline are intentionally OUT of the 6-max first cut in `src/nlhe/solver6.py`:

  1. **No strategy net / no average-strategy approximation.** `PlayerNetworks6Max` (Phase 4e.3a) only carries advantage nets. At deployment time, the deployed policy is the regret-matched current strategy from the most recent advantage net — not the average strategy across iterations. Average-strategy approximation is its own subphase (likely Phase 4e.4) and only worth doing once the baseline trains. **(SUPERSEDED 2026-05-27, Session 22: the strategy net was added — see "Strategy net for 6-max: single shared net with seat in features" below. All four deferrals SUPERSEDED 2026-05-27 — see entries dated 2026-05-27 in this file and the Step 5 design-decisions entry below. The Session-9 'minimum-viable first cut' restriction is fully retired.)**

  2. **No DCFR weighting yet (vanilla CFR only).** DCFR (Brown & Sandholm 2019) was added to HUNL in Phase 3 Track A1 (commit 41e2fa3). It provides faster convergence by per-sample iteration weighting. For 6-max first-cut, vanilla CFR (uniform weights) is the baseline. Mechanical to add later once the vanilla loss trajectory is established. **(SUPERSEDED 2026-05-27: DCFR weighting is wired in — `cfr_variant`/`_dcfr_weights` apply to both advantage-net (`solver6.py:393`) and strategy-net (`solver6.py:444`) training; the `dcfr-overnight-3000` production baseline used `cfr_variant="linear"`, `dcfr_exponent=1.0`. Implementation predates Session 22; the docs never propagated. See the 2026-05-27 audit-findings entry below.)**

  3. **No archetype mix yet.** The archetype framework (`src/nlhe/archetypes.py`) is HUNL-specific — it uses `derive_in_position` (HU position-from-current-player) and decision tables sized for two-player betting trees. Porting to 6-max requires position derivation for all 6 seats and decision tables sized for the multiway 6-max pot. Its own subphase, not a 4e.3c concern.

  4. **Uniform starting stacks per traversal.** Real SNG hands have stacks that evolve across hands (chip leader, short stack, equal stacks, bubble situations). The 4e.3c training loop uses `[cfg.starting_stack] * 6` for every traversal — every hand looks like Hand 1, with all stacks equal. The ICM transformation still operates correctly per terminal (deltas in equity are computed against equal-stack baselines), but the bot doesn't see bubble-pressure asymmetry from a 4-handed game with widely differing stacks. Stack-distribution sampling is its own subphase. Until it's added, the bot trains on a degenerate slice of the SNG state space. **(SUPERSEDED 2026-05-27: stack-distribution sampling is implemented — `src/nlhe/stack_sampler.py`'s `sample_starting_state` is wired into `solver6.py:518-551` (the `sampled["stacks"]` path), activated when `tournament_structure_path` is set; 6 of 8 phase4f configs set it and the `dcfr-overnight-3000` production baseline used it. Implemented 2026-05-23, commit 4739c1a; the docs never propagated. See the 2026-05-27 audit-findings entry below.)**

First-cut scope is "the smallest 6-max thing that trains without diverging, in time to evaluate on benchmark before adding complexity." This staging is what landed +78.35 bb/100 on HUNL — vanilla Deep CFR first, then DCFR, then archetypes, then determinism retrofits. Same logic applies to 6-max: ship a baseline that trains, measure it, then add enhancements with each one's contribution measurable.

**Alternative considered:** Ship 6-max with the full Phase 3 enhancement stack (strategy net + DCFR + archetypes + stack sampling) from the start, since the HUNL precedent exists and the code patterns are known.
**Reason rejected:** Five interacting moving pieces in one ship is uninvestigable when something breaks. The HUNL track added enhancements incrementally across 5+ sessions, with each step's contribution measurable separately. The same incremental approach is the right one for 6-max — even if the individual additions are "easier" the second time around, the joint debugging cost is the same.

## Session-summary convention: per-session files in docs/sessions/
**Decided:** 2026-05-24 (Session 13)
**Why:** `docs/SESSION_LOG.md` grew to 480+ lines with internal ordering that drifted out of sync with session numbering — it documents through Session 9 while commits already reference Sessions 11–12. Going forward each session gets its own `docs/sessions/session_<N>_summary.md`: self-contained, exactly one new file per close, no churn in an ever-growing file, greppable filename, easy to diff in review. `docs/sessions/README.md` explains the convention and its relationship to STATUS / DECISIONS / NEXT_SESSION. `SESSION_LOG.md` is retained as the historical record for Sessions 1–9; Sessions 10–12 (captured only in commit messages / STATUS at the time) are not back-filled.

**Alternative considered:** Keep appending to the single `SESSION_LOG.md`.
**Reason rejected:** The drift already happened once (numbering desync), and a single growing file makes each session-close a noisy multi-section diff. Per-file is the standard fix and costs nothing.

## Subgame leaf evaluator: BEST_RESPONSE form (Brown/Sandholm 2018 blueprint-reference approximation) (SUPERSEDED in Session 22)
**Superseded:** 2026-05-27 (Session 22) — see "Subgame leaf evaluator: ship PROFILE_SAMPLE for production" below. The Q11 / sub-step 6 pool ablation that this decision was explicitly gated on returned `PASS_BR_EQUIVALENT_TO_PROFILE`: BR beats blueprint (σ=4.56) but is **not** statistically distinguishable from PROFILE (σ=0.56, under the σ≥1.5 bar), so the "if BR doesn't earn its v×k cost empirically, revert to profile-sampling" condition written into this entry fired.
**Decided:** 2026-05-24 (Session 13, B1c sub-step 2 design — `docs/SUBGAME_LEAF_DESIGN.md`)
**Why:** The leaf evaluator assigns each depth-limited leaf a value by having opponents continue with biased blueprint strategies (the k=4 configs in `biased_policy.py`). Two forms were considered: (a) **profile-sampling** — leaf value = prior-weighted average over opponents' biases; (b) **best-response** — each opponent independently picks the bias maximizing its own value. We chose (b). Profile-sampling solves hero against an opponent assumed to play a fixed mixture, which produces hero strategies **brittle to whichever single bias actually exploits hero's chosen line** — the opposite of what real-time solving is for. The maximization is the robustness mechanism: it forces hero's subgame strategy to hold up against an opponent who adversarially selects the most damaging continuation. We use the **Brown/Sandholm 2018 single-pass approximation** — best-response computed against hero's BLUEPRINT strategy at the root, not hero's iteration-k subgame strategy. This recovers most of the robustness of full alternating best-response while keeping the leaf value a function of only `(leaf state, fixed biased continuations, menu, fixed blueprint)`, so leaf values are cacheable EXACTLY across CFR iterations (compute once per decision, not Z×W). Same tradeoff Pluribus made. Cost: per-leaf rollouts rise from `M` to `(v×k+1)×M`, and the production path moves to a rented GPU (decision budget X raised 4→6 s); the binding constraint is the measured ~0.9 ms/step state-prep cost, which a fresh decomposition (2026-05-24) attributes to `_build_view_6max` (0.64 ms) + `discretize` (0.10–0.24 ms) doing O(n) Python ops over the ~9,803-element fullgame `legal_actions()` — NOT `parse_state_6max`'s regex (0.008 ms). The fix is the sorted-legal-actions fast path (`fast_view.py`, sub-step 2 Stage A, gate ≤ 0.30 ms/step), Q4.5. Profile-sampling is retained as a `PROFILE_SAMPLE` fallback / ablation mode.

**Alternative considered:** Profile-sampling under a (Layer-4-tunable) prior — cheaper (`M` rollouts/leaf, cost independent of v) and the original Session-12 sketch.
**Reason rejected:** Averaging over opponent biases produces non-robust hero strategies; it does not capture the Pluribus mechanism. Kept only as the `PROFILE_SAMPLE` comparison baseline. The BR choice is itself gated on the Q11 Level-3 pool ablation (lock BR only if it beats blueprint at σ>2 and ≥ profile-sample); if BR doesn't earn its v×k cost empirically, revert to profile-sampling.

## Subgame solver: vanilla full weighted traversal, NOT external sampling (decision-time only)
**Decided:** 2026-05-25 (Session 19, B1c sub-step 3 — Decision 2 in `docs/SUBSTEP_3_DESIGN.md`, landed Stage 3-C `b415517`)
**Why:** The subgame CFR solver (`src/nlhe/subgame_solver.py`) runs **vanilla full weighted traversal**: every iteration visits every node of the depth-limited tree, weighting chance children by `chance_prob` (subgame.py:114-117) and opponent children by their fixed blueprint strategy (Decision 4 — opponents do not update). It does NOT Monte-Carlo-sample trajectories. Four reasons, in order: (a) **the tree is small** — depth-3 is ~150–220 nodes, so full traversal is cheap; Stage 3-C measured **0.109 s for K=1000 iterations**, ~15× under the design estimate and ~250× under the 27 s/decision budget. (b) **The tree builder is constructed for full-weighted traversal** — the per-branch `chance_prob` field exists precisely so the solver can weight children and "multiply these along the path to recover reach" (subgame.py docstring); external sampling would ignore those weights and re-sample. (c) **Regret semantics are sampling-variant-independent in expectation** — vanilla CFR computes the *exact* expected regret `v(a) − ev` while external sampling computes a Monte-Carlo *estimate* of the same expectation, so the units match either way and the warm-start from blueprint advantages stays commensurate. (d) **Vanilla has zero MC variance** and converges faster (deterministically) on a small tree than a sampled variant. **This is a decision-time-only choice for the subgame solver; blueprint *training* (`cfr6.traverse_6max`) still uses external sampling — that decision is unchanged.** External sampling is the right tool when the tree is too large to enumerate per iteration (the full-game training tree); the depth-limited subgame at decision time is not that regime. Empirical confirmation: Stage 3-C's analytical math anchor (synthetic chance-weighted tree converges to the analytical best response within 1% L1) plus Stage 3-B's bit-identity gate against the Stage-G stub (itself validated by Stage G's SUBSTANTIVE_PASS_AGGREGATE at M=32, session 18).

**Alternative considered:** External-sampling CFR in the subgame, as the pre-session-19 plans documented (`STATUS.md` / `NEXT_SESSION.md`, since corrected) — matching the blueprint training traversal.
**Reason rejected:** The "match training's sampling variant or regret units diverge" premise is false — units come from the regret formula and ICM leaf/terminal values, not the sampling scheme. With the units concern removed, external sampling buys nothing on a small enumerable tree and costs determinism + convergence speed + MC variance. The tree builder's `chance_prob` design confirms full-weighted traversal was the intended path. Methodologically stronger than the documented spec, the same shape as Stage F/G's approved deviations.

## Subgame leaf evaluator: ship PROFILE_SAMPLE for production
**Decided:** 2026-05-27 (Session 22, B1c sub-step 6 verdict — `docs/SUBSTEP_6_DESIGN.md` Decision 6.3)
**Supersedes:** the Session-13 "Subgame leaf evaluator: BEST_RESPONSE form" decision above.
**Why:** Sub-step 6's pool ablation returned verdict `PASS_BR_EQUIVALENT_TO_PROFILE`, measured on 75,000 hands (5,000 × 5 opponents × 3 challengers, `base_seed=2026`, `workers=64`, 19h29m wall). The architecture lifts strength decisively — `L(BR−blueprint) = +0.0087, σ=4.56`, with all 5/5 opponents positive — but BR is **not statistically distinguishable from PROFILE**: `L(BR−PROFILE) = −0.0009, σ=0.56`. BR's `v×k` rollout cost is therefore not justified by the data.
**Implication:** deploy the `PROFILE_SAMPLE` leaf evaluator at decision time. The H5 argmax-collapse mechanism documented in the diagnostic spike (close-EV preflop decisions where BR's small leaf-value perturbations flip the CFR argmax) is real, but its net effect averages to zero across the opponent pool at full statistical power. BR code is retained in `src/nlhe/subgame_leaf.py` for future regime testing — deeper stacks and postflop-heavier formats where the leaf/decision mix differs from this format's ≈98%-preflop profile — but is **not invoked at decision time**.

**Alternative considered:** Continue with BR (the Session-13 production choice).
**Reason rejected:** The locked verdict criterion (Decision 6.3, written pre-data) was explicit that BR earns its complexity only if statistically distinguishable from PROFILE at σ ≥ 1.5. Measured σ=0.56 is well under that bar — BR buys no measurable strength over PROFILE while costing `v×k` rollouts per leaf.

## Strategy net for 6-max: single shared net with seat in features
**Decided:** 2026-05-27 (Session 22, Scenario 3 step 3 — commit 78917f8)
**Supersedes:** resolves deferral #1 ("No strategy net") of "6-max blueprint training: minimum-viable first cut" (Session 9) above; and supersedes the mid-Session-22 "6 separate per-seat strategy nets" framing (held only in the Step-A/B working plan, before the HUNL no-filtering finding).
**Why:** HUNL's `_train_strategy_net` samples from the full shared strategy buffer with no per-player filtering (`solver.py:453`). Its two strat_nets therefore train on identical data and converge to the same function — the 2-net split is cosmetic. Mirrored naively to 6-max this would give 6 redundant nets. Pluribus precedent + data efficiency favor ONE shared strategy net with seat conditioning via the existing 236-dim feature vector (seat/position is already encoded there). `PlayerNetworks6Max` carries one `strat_net` + one shared `strat_buffer`; `cfr6.traverse_6max` writes the acting seat's current policy at non-traverser opp nodes; `solver6._train_strategy_net` trains it each iteration (KL, `_dcfr_weights`).

**Alternative considered:** (a) 6 separate strategy nets matching the advantage-side topology. **Reason rejected:** redundant once it was clear HUNL doesn't filter per-seat — all 6 would converge to the same position-conditioned function. (b) Seat-filtered training of 6 separate nets (each net trains only on its own seat's samples). **Reason rejected:** data inefficiency — each net would see ~1/6 the samples, for no benefit the shared net + seat-in-features doesn't already provide.

## Checkpoint schema policy: two-tier load (accept all, refuse strat-net deploy on v1)
**Decided:** 2026-05-27 (Session 22, Scenario 3 step 3 Step E — commit 78917f8)
**Supersedes:** the Step-B-era "refuse v1 at the load boundary" choice (locked mid-Session 22 in code + commit drafts, never formalized in this file; reversed in the Step-E recon). Builds on the precedent of "DCFR backward compatibility: refuse non-vanilla resume from pre-DCFR checkpoints" above, but lands on a softer policy for this case.
**Why:** The Step-B refuse-on-load policy (`load_state_dict` raises on any non-v2 dict) would have made the existing 36 `six_max_*` checkpoints — including `dcfr-overnight-3000` — unloadable. That contradicted: (1) Step E's "fall back on v1" intent; (2) the commit message's "remain loadable for adv-only paths" claim; (3) sub-step-6 reproduction continuity (the verdict was measured on v1 checkpoints); and (4) the Scenario 3 plan of comparing v2 retraining against the v1 baseline.
**Decision:** `PlayerNetworks6Max.load_state_dict` accepts BOTH v1 and v2 dicts. A v1 dict (no `schema_version`) loads the advantage nets only, leaves `strat_net` at fresh init, and records `loaded_schema_version="v1"` on the container. A v2 dict ("v2_with_strategy") loads adv + strat and records v2. The refuse-on-mismatch moves to the DEPLOYMENT boundary: `PlayerNetworks6Max.inference_policy(seat, features, legal_mask)` dispatches v2 → `strat_net` masked softmax (HUNL `policy_adapter.py:349-372` pattern); v1 → regret-matched advantage-net policy (`_strategy_from_advantages`, legacy behavior).
**Implication:** existing v1 checkpoints stay loadable for advantage-net-only inference + eval; they cannot be deployed with strategy-net policy (no trained strat_net to deploy). Scenario 3 step 7's v2-era retraining produces v2 checkpoints capable of full strategy-net deployment.

**Alternative considered:** (a) refuse-on-load (Step B's initial choice). **Reason rejected:** the four-way contradiction above. (b) silent backward-compat (load v1, auto-instantiate a fresh strat_net, no schema marker). **Reason rejected:** silent quality-degradation risk — a caller could load v1 and unknowingly deploy a fresh-initialized (untrained) strat_net. The explicit `loaded_schema_version` marker + deployment-time dispatch makes the v1/v2 distinction visible and safe.

## SubgamePolicy two-signal architecture: gate on adv-net RM+, inference on schema-dispatched policy
**Decided:** 2026-05-27 (Session 22, Scenario 3 step 3 Step E — commit 78917f8)
**Why:** `SubgamePolicy` makes two policy reads with different purposes. The GATE (`_evaluate_gate`) decides WHEN to invoke the subgame solver — a heuristic threshold on "blueprint max action prob". The PLAYED action decides WHAT to play — the deployed policy. These are allowed to be different signals without inconsistency. Sub-step 6's gate was calibrated against the adv-net RM+ policy (f≈0.27 SOLVE rate); routing the gate through `inference_policy` would change the SOLVE rate on v2 checkpoints and break measurement continuity with sub-step 6. Routing only the played action through `inference_policy` captures the deployment improvement without touching the gate.
**Decision:** `_evaluate_gate` stays on adv-net RM+ regardless of `schema_version`. `_sample_action_from_policy` (the played-action path, shared by SubgamePolicy fall-through, `eval_pool`, and self-play) goes through `PlayerNetworks6Max.inference_policy` and dispatches by schema.

**Alternative considered:** Unify the gate and the played action through `inference_policy`. **Reason rejected:** would change the sub-step-6-calibrated SOLVE rate on v2 checkpoints, sacrificing measurement continuity for no clear benefit — the gate is a "should I bother solving" heuristic, not the deployed policy.

## FINDING — v1 inference_policy behavior deltas vs the legacy adv-net read
**Recorded:** 2026-05-27 (Session 22, Scenario 3 step 3 Step E — commit 78917f8)
**Status:** a finding, not a deferred fix. Two narrow behavior deltas exist between the legacy direct adv-net read (`_sample_action_from_policy` pre-Step-E) and the new `inference_policy` v1 dispatch, BOTH confined to the all-legal-advantages-≤0 edge case:
  - **(a) Sample mode:** legacy used `rng.choice(legal_indices)` (uniform over legal); new uses `rng.choices(weights=uniform)`. Different rng draw pattern, same statistical distribution.
  - **(b) Argmax mode:** legacy picks the "least-negative" advantage (`argmax` over raw advantages); the new path (`_strategy_from_advantages` → uniform when all ≤0 → `argmax`) picks the "first legal" action. Deterministically different action when all legal advantages ≤ 0.
**Confirmed immaterial in the ablation regime:** `tests/test_eval_pool_ablation.py` + `tests/test_ablation_decision_level.py` (22 tests exercising the eval path that now routes through `inference_policy`) all passed at 6m52s in Step E. The edge cases are either too rare to surface in 75,000-hand statistical regimes, or both branches produce sufficiently-similar outcomes that the harness absorbs the difference.
**Revisit if:** future high-precision determinism work surfaces these (e.g. byte-identical sub-step-6 reproduction across the v1→v2 era).

## Doc-vs-code audit: stack sampling + DCFR weighting + league play already done in production
**Dated:** 2026-05-27 (Session 22 close-out)

**Background:** While preparing to start Scenario 3 step 4 (stack-distribution sampling), the recon-first audit discovered the work was already implemented and in production. A broader audit found two additional items in the same state. This entry records the findings durably so future sessions don't redo this work.

**Audit method:** For each item claimed "deferred" or "future" in DECISIONS.md / NEXT_SESSION.md, grep the codebase for implementation evidence and check whether the production baseline (`dcfr-overnight-3000`) used it.

**Findings:**

1. **Stack sampling: DONE 2026-05-23 (commit 4739c1a).** Full implementation in `src/nlhe/stack_sampler.py` + `solver6.py:518-551`. Used in the `dcfr-overnight-3000` production baseline. Docs marked deferral #4 "open" through Session 22.

2. **DCFR weighting: DONE prior to `dcfr-overnight-3000`** (the training run is literally named `dcfr_linear_overnight`). `_dcfr_weights` helper in `solver6.py:338` applied to both advantage- and strategy-net training. Production baseline config: `cfr_variant="linear"`, `dcfr_exponent=1.0`. Docs marked deferral #2 "open" through Session 22.

3. **League play infrastructure: DONE.** `league_pool.py`, `checkpoint_registry.py`, `_maybe_sample_league_opponent` threaded through both traversal branches. Six config knobs on `TrainConfig6Max`. Default `league_mix=0.0` means no-op when unconfigured. League v1 ran to ~iter 1700 (killed early; `PHASE4F_LEAGUE_V1_FINDINGS.md` verdict: NEGATIVE — trails DCFR every iter, loses to `dcfr-3000` by 9.9σ at iter 1400). League v2 config exists (`configs/six_max_phase4f_dcfr_league_v2.yaml`: mix 0.15, recency-weighted, `[peak,archetype]` anchors) and ran a 900-iter shakedown that completed cleanly. The shakedown's 600-iter checkpoint (`league-v2-600`) is in active use as an eval-pool member for sub-step 6 ablations. Open work for league: extend past 900 iters and produce a verdict OR (per Scenario 3 revised plan) bundle the v2-schema league run into step 7's production training.

4. **Archetype mix for 6-max: GENUINELY OPEN.** `archetypes.py` exists but is HUNL-only. 6-max opponent diversity currently sources from league pool + `scripted_bots/` (a different mechanism). Porting the in-traversal archetype framework to 6-max remains as Scenario 3 step 5.

5. **Layer 4 within-match adaptation: GENUINELY NOT STARTED.** ARCHITECTURE.md describes the design; zero implementation in `src/nlhe/`. Scheduled for Scenario 3 step 8+.

**Naming clarification:** "v2" is now overloaded in the project. "League v2" means the second league config (vs league v1). "v2-schema" or "v2_with_strategy" means the strategy-net checkpoint schema introduced in commit 78917f8. These are unrelated; do not conflate. The existing league-v2 shakedown predates the strategy net by 3 days, so it is a v1-schema league run.

**Impact on Scenario 3 plan:** The remaining work is shorter than session-22 estimates. Of the originally-planned steps 4-7, only steps 5 (archetype mix) and 7 (production training) are genuinely-open implementation work. Step 6 (league) becomes "bundle v2-schema league training into step 7's production run" rather than a separate step. `league-v2-600` remains in the eval pool as a v1-schema opponent for measuring whether v2-schema training helps.

**Process finding:** the recon-first discipline saved ~1-2 weeks of redundant work on stack sampling. Future sessions should not trust DECISIONS.md "deferred" claims without code verification.

**Alternative considered:** (a) silently update the planning docs without an audit entry — rejected on the durability argument (this kind of doc-vs-code drift will recur, and a recorded audit makes the next investigation cheaper). (b) skip the audit and just supersede deferrals #2 and #4 — rejected because that wouldn't preserve the league-v2 / v2-schema overloading clarification or the process finding.

## Archetype mix for 6-max: wrap-not-port via ArchetypePolicy + ArchetypePool
**Dated:** 2026-05-27 (Session 23)
**Supersedes:** Session-9 "minimum-viable first cut" deferral #3 (archetype mix).

**Why:** 6-max training needs in-traversal style diversity beyond what the league pool (Shanky bots) provides. League diversity is rich-but-realistic; archetype diversity is parametric-and-extreme (NIT/TAG/LAG/STATION/MANIAC span style space deliberately). Both contribute different shapes of opponent variety to the strategy net's learned average.

**Decision:** wrap HUNL's `archetype_policy` (`archetypes.py`, unmodified) as a 6-max `ArchetypePolicy` adapter rather than porting. Recon confirmed `archetype_policy` is game-agnostic (zero 2-player assumptions; `in_position` is the only positional input, used as a bool for a 10% nudge). Adapter lives in `src/nlhe/archetype6.py` following the project's HUNL-paired-module naming convention.

**Mechanism:**
- `ArchetypePolicy.select_action(parsed, state, rng, mode)` mirrors `ShankyProfilePolicy`'s view+discretize+evaluate pipeline.
- `ArchetypePool.sample_opponent(rng)` mirrors `LeaguePool` — no internal mix gate; the three-way roll already decided we're in the archetype band.
- Per-hand-shared dispatch: one archetype profile per traversal, used across all opp seats (matches league override semantics).
- Strategy-buffer write-suppression is automatic via the existing cfr6 short-circuit at line ~299 (returns before `strat_buffer.add` at line ~381).

**6-max-specific adaptations (in the adapter only, `archetype_policy` untouched):**
- `in_position` derivation: binary (postflop BTN=in; preflop BB=in) via `infoset6.position_for_seat_with_dealer`.
- `dealer_seat` sourced from `parsed['dealer_seat']` (present in `parse_state_repeated_6max`, the tournament training path).
- Defensive fallback if `dealer_seat` missing: `in_position=False` with one-time warning log.
- `bucket_of` called with `cfg.bucket_runouts` (small MC noise on postflop, by-design per the noise-tolerant archetype framework).
- Loads `runs/archetype_design/bucket_equity_analysis_6max.json` (Phase 5-pre artifact, commit d093abd).

**Three-way combined override-slot sampler (Phase 5-A, commit 470beb7):**
- Roll uniform `r` on shared `self.rng` once per traversal.
- Band ordering: `r in [0, archetype_mix)` → archetype; `r in [archetype_mix, archetype_mix + league_mix)` → league; `r >= total` → self-play.
- Constraint: `archetype_mix + league_mix <= 1.0` (validated in `TrainConfig6Max.__post_init__`).
- Bit-identity by construction at `archetype_mix=0.0`: archetype band `[0, 0.0)` never fires; league band reduces to the exact pre-Phase-5 condition `r < league_mix`.

**Implementation phases:**
- Phase 5-pre (d093abd): regenerated EquityCalibration for the 6-max abstraction.
- Phase 5-A (470beb7): three-way sampler scaffolding (archetype band placeholder).
- Phase 5-B (48b5eae): `ArchetypePolicy` + `ArchetypePool`, archetype band filled.
- Phase 5-C (this commit): closure + documentation reconciliation.

**Acceptance gates passed (Phase 5-B):**
- F1 bit-identity at `archetype_mix=0.0`: byte-identical to `/tmp/fast_view_smoke_pre` baseline.
- F2 pool-built-but-unused: zero side effects from pool construction at `archetype_mix=0.0`.
- F3 functional at `archetype_mix=0.5`: `strat_buffer ≈ 0.59×` baseline (DECISIONS.md:216 invariant verified).
- F4 full project test suite: 167 passed across 10 modules.

**Findings recorded for Step 7:**
- Passive archetypes (STATION, NIT) produce longer hands than self-play due to more calling, so adv-buffer growth scales with hand-decision count, not trajectory count. Effective sample efficiency at `archetype_mix>0` differs from pure self-play.
- Phase 5-pre's calibration regeneration surfaced a recon-quality finding: `analyze_bucket_equity.py` uses a shared rng across preflop+postflop sampling, so deterministic preflop bucketing shifts postflop sampling position. Postflop quantiles differ from HUNL by 0.01–0.03 MC noise (provably bit-identical bucketing). By-design; noise-tolerant framework.

**Alternatives considered:**
- Option 2 (port `archetype_policy` as 6-max-native): rejected — recon confirmed `archetype_policy` is game-agnostic; wrapping is minimum risk.
- Per-seat archetype dispatch (different archetype per opp seat per hand): rejected — mechanism cost (per-seat policy map vs single override slot) doesn't justify the marginal coverage gain. Across many trajectories, per-hand-shared still sees every archetype at every seat.
- ICM-aware archetypes (adjust thresholds for bubble pressure): rejected as first cut — archetypes are deliberately suboptimal; ICM-aware archetypes converge toward equilibrium. The league/Shanky path provides realistic-ICM-aware-but-flawed diversity; `archetypes.py` provides deliberately-flawed-style-extreme diversity. They complement.

**Implication for remaining Scenario 3 work:** only Step 7 (multi-day production training) remains as genuine implementation work. Step 7's training run exercises the full v2-schema stack (strategy net + DCFR + stack sampling + archetype mix + league play) at production scale (24–72h GPU).

## Integration testing required for cross-module wiring (Phase 5-B post-mortem)
**Dated:** 2026-05-27 (Session 23, Step 7 pre-flight)

Phase 5-B (commit 48b5eae) shipped the archetype Policy adapter with the `dealer_seat` fallback as a defensive safety net. The Phase 5-B test suite included unit tests that hand-fed `dealer_seat` to `_in_position` and verified the contract, and a separate test that deliberately removed `dealer_seat` and verified the fallback fired with a warning. Both passed.

What the tests did not cover: running archetype dispatch through the actual training path (`parse_state_6max` in tournament mode) and verifying `dealer_seat` reaches the adapter. The Step 7 benchmark (real production path) found that `parse_state_6max` never produces `dealer_seat`, so the defensive fallback fired for every archetype hand in production training, silently degrading archetype behavior to "always out of position."

The Phase 5-B recon assumed `parse_state_repeated_6max` was the training path (which DOES carry `dealer_seat`). The actual training path uses `parse_state_6max` (which does NOT — the tournament-mode inner games are single-hand `universal_poker` states with no `dealer_seat()` method). The unit tests validated the contract by hand-feeding the input the real wiring never supplied.

**Lesson:** contract-mocking unit tests can pass while integration wiring is broken. Future practice for any cross-module wiring change requires at least one end-to-end integration test that exercises the actual production path, not just hand-fed input/output contracts.

This commit (Step 7-fix) restores the intended Phase 5-B behavior by threading `dealer_seat` through `CFR6MaxContext` into the parsed dict at the cfr6 parse site, and adds the integration tests Phase 5-B should have had (`test_dealer_seat_reaches_adapter_in_tournament_path`, `test_seat_indexing_alignment_in_tournament_path`).

**Seat-indexing crux, empirically confirmed:** `current_player()` uses the same original-seat numbering as `sample_starting_state`'s `dealer_seat` — verified on full-ring hands (first preflop actor == `(dealer+3) % 6` = UTG, 16/16). So `position_for_seat_with_dealer(current_player, dealer_seat)` is consistent. Known approximation retained: on busted-seat hands (alive < 6, ~60% of sampled states) the full-ring position helper gives an approximate position — already-documented behavior driving only `archetype_policy`'s 10% in-position nudge, not a correctness defect. The ~60% busted-seat prevalence observed in tournament-mode sampling (Step 7 fix verification, 2026-05-27) is higher than 'edge case' implies and suggests busted-seat-aware position semantics could be a meaningful Phase C/D investment if archetype-mix style differentiation becomes a strength bottleneck. Recorded but not addressed here — the 10% in_position nudge is small enough that approximate positions on multi-bust hands don't block Step 7's strength claim.

## Diversity-mix experiment shelved; Layer 3 (subgame solving) promoted to next priority
**Decided:** 2026-05-29 (Session 5 close)
**Why:** The anchor run (DCFR self-play at k=200, 2000 iters) completed cleanly at 12:19 UTC and produced enough lift trajectory data to identify the real bottleneck. The bot reached its strength ceiling around iter 500-1000:
- strat_loss decreased 0.945 → 0.815 across the full 2000 iters (refinement continued)
- Head-to-head self-anchor lift mean ≈ zero after iter 500 (CFR equilibrium oscillation around the abstraction's Nash, not strength gain)
- 15 of 19 Shanky-rotation readings negative — bot losing to most commercial scripted bots even at iter 2000
- Same Shanky opponent (KillPhilMTT) at iter 200 / 1100 / 2000: -0.0435 / -0.0256 / -0.0302 (improved then plateaued)

This pattern is consistent with abstraction being the bottleneck, not iteration count. CFR converges to Nash *within the abstraction*; k=200 buckets compress ~1.3 trillion strategically-distinct hand+board combinations into 200 cells per street, so within-bucket distinctions (top pair good kicker vs weak kicker, draw-type) are invisible to the bot. A Shanky rule-based bot has effectively much finer perception, which lets it exploit the k=200 Nash bot.

This shifts the project priorities. The original plan was: complete the diversity-mix experiment (control vs treatment at k=200), then train production blueprint at k=500 with the winning strategy. The honest read is that the diversity-mix question at k=200 answers a narrow comparison that may not transfer to k=500, and the bigger unlock is Layer 3 (real-time subgame solving) which is the architectural keystone for commercial-grade play.

**Decision:**
- Diversity-mix control + treatment runs are SHELVED. Three configs (anchor/control/treatment at 2000 iters with parallel_groups=10) remain in the repo for future use.
- Layer 3 (real-time subgame solving) is the next session's focus. ARCHITECTURE.md originally placed Layer 3 in Phase 3+; promoting to immediate next work.
- The anchor checkpoint (runs/dcfr_anchor_2000/checkpoints/ckpt_iter_2000.pt) is preserved as a k=200 blueprint baseline. When Layer 3 lands, the blueprint-only vs blueprint+subgame-solving comparison will use this exact checkpoint.

**Alternative considered:** Continue with the planned diversity-mix experiment (~5h) and produce a small answer; or train one k=500 self-play run (~6-8h) to test the "is abstraction the bottleneck" hypothesis directly.
**Reason rejected:** Both are real work; both would produce some incremental information; but neither addresses the architectural keystone the project actually needs. Layer 3 is the unlock that moves the bot from "Nash within an abstraction" to "Nash + real-time refinement at decision time" — the difference between Pluribus-level play and blueprint-only play.

## Blueprint-alone is the reference agent; real-time resolver is experiment-only
**Decided:** 2026-05-31 (Session 6, resolver diagnosis J0–J3 + depth sweep)
**Superseded in part:** 2026-05-31 (Session 6, post-bubble-slice) — see
"Foundation pivot: retire real-time resolver; build trained-adaptive-policy with Leduc proof
first" below. The floor rule (blueprint-alone is the shipped reference) CARRIES FORWARD; the
rebuild-the-resolver fork (Step 3 leaf probe → Step 4 path A/B) is RETIRED.
**Conclusion (verbatim):**
Blueprint-alone is the reference agent. The real-time resolver is proven net-negative at every
shippable configuration (X0 lift ≈ 0 vs a correct-model opponent; J2 condition-A −2.7 to −4.2
ICM-pts vs blueprint on all 4 legit profiles; J3 robust BR-leaves help partially but never reach
blueprint-alone, and are unshippable at action p95 82–99s / 14.5% >15s, d=3; depth sweep: d=3 is
the worst point, non-monotone. Root cause under diagnosis: leaf-value precision/bias vs unsafe
solve. Resolver remains experiment-only / opt-in pending rebuild — do not enable in any
non-experimental path.

**Floor rule (absolute):** no change may make the shipped/reference agent worse than blueprint-alone.
There is no production deployment layer (see "Scope: training only, no deployment layer" above);
the resolver (`SubgamePolicy`) is instantiated ONLY by experiment/measurement harnesses
(`eval_resolver_vs_shanky.py`, `eval_pool_ablation.py`, `measure_*`), never on a default path.
This entry records that opt-in status as the locked floor pending the rebuild decision (Step 3
leaf probe → Step 4 path A safe-resolve or path B learned leaf value net).
**Do NOT** wire `SubgamePolicy` as a default agent anywhere until a rebuild beats blueprint-alone.

## Foundation pivot: retire real-time resolver; build trained-adaptive-policy with Leduc proof first
**Decided:** 2026-05-31 (Session 6, post-bubble-slice)
**Supersedes:** the resolver-rebuild fork (Step 3 leaf probe → Step 4 path A safe-resolve / path B
learned leaf value net) recorded in the prior entry ("Blueprint-alone is the reference agent;
real-time resolver is experiment-only"). The floor rule from that entry (blueprint-alone is the
shipped reference; no change may make the shipped agent worse than blueprint-alone) CARRIES
FORWARD. Only the rebuild-the-resolver direction is retired.

**Verbatim conclusion:**

FOUNDATION PIVOT (supersedes the resolver-rebuild fork). The real-time resolver is RETIRED as a
development path — proven net-negative at every shippable config (X0 lift≈0 vs correct-model
opponent; J2 −2.7..−4.2 ICM-pts; J3 robust BR-leaves never reach blueprint and are unshippable
at 80-100s/decision; depth sweep d=3 worst, non-monotone; bubble slice: profile-leaf harm
concentrated >15BB, break-even ≤15BB). Even a perfectly fixed resolver is low-upside (X0).
New foundation = a TRAINED ADAPTIVE POLICY (StratFormer-style): a sequence model anchored to a
GTO baseline that shifts toward opponent exploitation WITHIN a match, regularized for safety
(never more exploitable than the anchor). Blueprint-alone remains the reference floor. Goal:
crush exploitable opponents (most humans + chip-EV GTO bots via ICM); NOT to beat optimal play
(no exploitable edge exists there).
PLAN: validate the architecture with a CPU-only LEDUC PROOF on Contabo BEFORE any 6-max/GPU
work. Step 0: tabular CFR+ near-Nash Leduc anchor (validate <5 mbb/g via leduc/evaluate.py).
Step 1: ~6 rule-based Leduc archetypes. Step 2: small adaptive sequence-model (2 heads: policy +
opponent-model). Step 3: two-phase training (distill-to-anchor, then regularized exploit-shift
vs archetypes). Step 4 PASS BAR (both required): positive exploitation gain on exploitable
archetypes AND exploitability stays near the CFR+ anchor. If it clears the bar → scale to 6-max
NLHE on RunPod, anchored to the Deep CFR ICM blueprint. If not → architecture reworks before any
GPU spend. The resolver code stays in-tree but dormant; do not invest in fixing it.

**Execution status (2026-05-31):**
- **Step 0 — DONE.** Tabular CFR+ Nash anchor landed in commit `019d486`. Final exploitability
  **0.1286 mbb/g** at 1000 iters (Nash bar < 5 mbb/g; ~39× margin). 936 info states. Artifact:
  `runs/leduc_cfr_anchor_20260531_144405/{anchor_table.json, avg_policy_arrays.pkl, metrics.json}`.
  Code: `src/leduc/cfr_anchor.py`, `scripts/train_leduc_cfr_anchor.py`.
- Steps 1–4: pending. Step 1 (rule-based Leduc archetypes) is the next-session start; Step 2 and
  Step 3 require explicit approval after their respective design proposals; Step 4 is the
  go/no-go gate before any GPU/6-max scale-up.

**Resolver code disposition:** `src/nlhe/subgame_solver.py`, `src/nlhe/subgame_policy.py`,
`scripts/eval_resolver_vs_shanky.py`, and `evals/resolver_shards/*` stay in-tree but DORMANT.
Do not invest in fixing them. The bubble-slice BR arms currently running may finish — log their
≤15BB / >15BB buckets when they land, for completeness — but no further resolver work follows.

**For any fresh session reading this file first:** the previously-queued resolver-rebuild work
is OFF. The active program is the foundation-pivot Steps 0–4 above; see
`docs/NEXT_SESSION.md` top-of-file banner and `docs/STATUS.md` "Foundation pivot" banner.

## Leduc proof complete — S1 verdict; move to 6-max scaffold (CPU-smoke before GPU)
**Decided:** 2026-06-01 (Session 6 late, post-S2(a) cotrain)
**Supersedes:** the "Leduc proof FIRST" gate in the foundation-pivot entry above. The Leduc
proof is now DONE; its verdict reroutes the program to a 6-max adaptive-training scaffold, with
the S2(a) negative as design input. The architecture, leakage invariants, and read mechanism
carry forward unchanged; what is retired is "use Leduc to validate per-hand strength resolution
before scaling."

**Verbatim conclusion:**

LEDUC PROOF COMPLETE. Verdict: **S1 confirmed** — Leduc is too information-poor (~2–3 opp
decisions per hand) to per-hand-resolve opponent STRENGTH, even with direct cell-classification
supervision and a transferable head-derived read.

**Falsification chain — four hypotheses tested and ruled out by experiment:**
1. **Imbalance (A+B)** — recipe-A cell-sampling weights + recipe-B inverse-freq class CE.
   Maniac D̄ moved 0.13 → 0.59; maniac action acc capped at 0.60. Cell-distribution rebalance
   insufficient. (commit `e4c9a9a`)
2. **Feature smearing (C)** — STATS_MANIFEST_v3 per-street opp_stats split (12→20 dims). No
   movement on watch metric (0.60→0.61); E0 broke (4.09→5.66). Reverted.
   (commit `08b59eb`, revert `2d004ca`)
3. **Read primitive (D-pivot)** — KL → argmax-disagreement × sharpness margin. Ratio
   1.17 → 1.35. Maniac choice-reveal acc 0.583. Disagreement was 74–86% on LOPSIDED GTO spots,
   not spurious near-tie firing — the head was confidently wrong, not noise-firing. Read
   mechanism was not the bottleneck. (commit `8f8f20b`)
4. **Training target (S2(a))** — cell-classifier head as primary aux signal, head-derived KL-
   vs-anchor read (oracle-free, Test D verified). Ratio 1.25; maniac CELL-classification acc
   0.580; misclassification breakdown 90% raise-cluster but only 58% strength-resolved to
   s1.00 (32% spilled to s0.50 raise-mid). E0 broke (6.24, predicted trunk overload from triple
   loss). Decision-level cell acc 1.000 vs hand-end 0.58 = per-hand-info-poverty signature.
   (commit `9e5c07f`)

**What was validated — carry forward to 6-max:**
- **Architecture mechanism is sound.** Transformer trunk + `opp_head_cell` 9-way classifier +
  head-derived KL-vs-anchor read is the right shape. The head reads maniac raise-tendency 90%
  reliably and NEVER confuses it for calling-station / always-fold / over-folder — the call-
  default leak is ruled out. The mechanism is identity-aware; it cannot resolve fine-grained
  strength on Leduc's thin signal.
- **Leakage invariant + Tests A/B/C/D are the reusable correctness surface.** §8 tokens /
  opp_stats counterfactual bit-identity, plus Test D's "cell label not in forward inputs"
  guarantee, transfer to 6-max as the audit harness for any new head architecture.
- **Confidence-gated blend is the safety mechanism.** `gate(0)=0` (no evidence → π = anchor
  exactly), `g ∈ [0, 1)`, blend `π_final = (1−g)·anchor + g·raw`. By construction
  `E0_final ≤ E0_anchor` when the gate is closed. Transfers unchanged.
- **Read path is transferable, NOT oracle-dependent.** Inputs enumerated and verified: head
  outputs + universal anchor only; archetype identities and δ tables never enter the read. Same
  primitive deploys at 6-max vs unseen opponents.

**Why 6-max is the next proving ground (deliberate scope decision, not a tweak):**
6-max NLHE provides ~10–20× more opp decisions per hand than Leduc (4 streets × ~3 decisions/
street + richer action space + multiple opponents). This directly relieves the strength-
resolution bottleneck. The information density that capped Leduc at "reads tendency, loses
strength" is precisely what 6-max provides natively.

**Open questions for the 6-max scaffold (flag, do NOT solve in scaffold step):**
- **(a) Cell-classifier generalization story.** Leduc S2(a) used 9 enumerable archetypes; real
  6-max opponents are not discrete cells. The 6-max adaptive head needs a generalization design:
  continuous opponent embedding (cell head becomes regression to a learned embedding), archetype-
  mixture (posterior over many discrete profiles, accepting mixture as natural state), or some
  hybrid. Design surface, not solved here.
- **(b) Trunk capacity under triple-loss budget.** E0 broke at Leduc under
  `L_distill + 0.3·L_cell + 0.3·L_action` on a 78k-param net. The 6-max net is larger, but the
  distill-vs-aux gradient competition must be managed; first lever per S2(a) result is lowering
  `λ_action` (action carryover) while keeping `λ_cell` as the primary identity signal. Settings
  to be re-derived at scale; flag as a known dial.

**Next step (gated; NOT GPU yet):**
Build + CPU-smoke-test a 6-max adaptive-training scaffold on Contabo BEFORE renting GPU:
1. 6-max adaptive net (trunk shape derived from S2(a), scaled to NLHE token/feature surface).
2. Deep CFR ICM blueprint as anchor (existing in-tree; not Leduc CFR+).
3. Training loop wiring: distill + aux (cell-or-embedding head per (a)) + safety blend.
4. CPU smoke: tiny epoch count, verify gradients flow / losses decrease / §8-analog leakage tests
   green on 6-max tokens / E0-analog bounded.
GPU spend follows ONLY after the scaffold runs correctly on CPU. Scaffold spec is the next
session's design surface, not tonight's work.

**Artifact pointers:**
- Leduc CFR+ anchor: `runs/leduc_cfr_anchor_20260531_144405/`.
- A+B reference (E0 = 4.0936 baseline): `runs/leduc_phase1cotrain_20260531_222313/`.
- D-pivot probe result: `runs/leduc_phase1cotrain_20260531_222313/d_pivot_probe_argmax.json`.
- S2(a) terminal experiment: `runs/leduc_phase1s2a_20260531_234542/`.
- Code: `src/leduc/adaptive/model.py` (`AdaptivePolicyNet`, `AdaptivePolicyNetS2a`),
  `scripts/leduc_phase1_*cotrain*.py`, `scripts/leduc_d_pivot_probe.py`,
  `tests/test_leduc_token_no_leak.py` (A/B/C/D, 7/7 GREEN).

**For any fresh session reading this file first:** the Leduc proof is DONE. The active program
is the 6-max adaptive scaffold per "Next step" above. See `docs/STATUS.md` "Leduc proof complete"
banner and `docs/NEXT_SESSION.md` top-of-file banner.

## Integration ghost-ante bug-match workaround (Option B, deliberate; FIX QUEUED)
**Decided:** 2026-06-04 (Option B integration build, scraper-state replay phase)
**Why:** `src/nlhe/game_strings.py:to_inner_game_string_for_state` has a latent bug — it
passes `self.num_players` (always 6) to `BlindLevel.inflated_big_blind(n)` regardless of
the actual alive-seat count. The result: at shorthanded tables (`n_alive < 6`), the library
posts a "ghost ante" for each empty seat into the BB's contribution, over-stating both the
BB contribution and the pot by `(NUM_SEATS - n_alive) * ante` chips. The bug is exposed by
the strict invariant check in the live-corpus replay (`scripts/test_integration_handstart.py`):
all 6-handed frames pass 100% bit-exactly; all shorthanded frames fail by exactly the
ghost-ante amount.

**Why we didn't fix the library directly:** the existing ship-candidate (`rebel_value_net_full.pt`)
and the k=200 blueprint (`runs/six_max_20260530_034023_phase4f_dcfr_candC_k200/checkpoints/ckpt_iter_2000.pt`)
were trained via `sample_starting_state` → `to_inner_game_string_for_state` (the buggy path),
and `_sample_alive_count` samples 4/5/6-handed at 40-55% rates in mid/short stages. So the
ghost-ante distribution is BAKED INTO their training distributions. Fixing the library now
without retraining would shift inference-time per-seat normalized features by up to
`(NUM_SEATS - n_alive) * ante / starting_stack` — as much as 24% at level 10 with 4 alive —
on the exact shorthanded late-game states that decide the format. That would invalidate
the `candidate_bakeoff` numbers used to pick the ship checkpoint.

**Workaround applied (Option B):** in `src/nlhe/integration/invariant.openspiel_to_scraper_view`,
match the library's ante convention by using `NUM_SEATS=6` in the BB stack/bet conversions
(instead of `n_alive`) and subtracting `(NUM_SEATS - n_alive) * ante` "ghost antes" from the
reconstructed pot. The resolver/integration thus sees state values that match the model's
training distribution; the invariant check passes on shorthanded states by matching the
library's bug. Documented as a deliberate bug-match in the invariant.py module docstring.

**Alternative considered (Option A) — fix the library at the source:** clean semantics
(antes paid by alive seats only) but requires retraining the blueprint and rebel value-net
on correctly-sampled chips. ~weeks of training. Plus updates to one test that hard-codes
the buggy inflated_bb value (`tests/test_game_strings.py:test_for_state_sb_bb_rotate_over_alive_seats`).

**Reason Option B chosen now:** preserves model in-distribution at inference; bake-off
numbers stay valid; contained 3-line workaround; trivially removable when models are
retrained.

**QUEUED for the next training cycle:** apply Option A — fix `to_inner_game_string_for_state`
to use `n_alive` not `num_players` in `inflated_big_blind`; update
`test_for_state_sb_bb_rotate_over_alive_seats`; retrain blueprint + value-net on the
corrected chip distribution; THEN remove the bug-match from `invariant.py` (the conversion
becomes the formula in the module docstring's commented-out "correct" alternative). The
workaround and its removal are paired — do not remove one without the other.

**Pointer to the workaround:** `src/nlhe/integration/invariant.py:openspiel_to_scraper_view`
(module docstring + inline `BUG-MATCHED` comments at the conversion lines).

## Heads-up SB/BB convention library bug (latent; FIX QUEUED with ghost-ante fix)
**Decided:** 2026-06-04 (Option B integration build, scraper-state replay phase)
**Why this is noted, not fixed:** `src/nlhe/game_strings.py:to_inner_game_string_for_state`
(line ~397-417) computes SB/BB positions in heads-up play with the convention "dealer=BB,
non-dealer=SB" — opposite to standard real-poker heads-up (dealer=SB, non-dealer=BB; SB acts
first preflop). Concretely, for `n_alive=2` the function returns `sb_seat=alive[(dpos+1)%2]`
and `bb_seat=alive[(dpos+2)%2]=alive[dpos]=dealer_seat`. The comment in the source claims
"BB acts first preflop" for heads-up which is also opposite to standard rules.

**Exposed by:** the strict invariant check on `live9.jsonl` line 265 — a heads-up frame where
the scraper sees `dealer=seat3` with `bets={seat1: 100, seat3: 50}` (dealer is SB), but the
library's replay produces `bets={seat1: 50, seat3: 100}` (dealer is BB). Pre Option B with
the ghost-ante fixed in the integration, the per-seat stack/bet are swapped between the two
positions.

**Why we don't fix this now:** out of scope for the ship integration:

1. `_sample_alive_count` (`src/nlhe/stack_sampler.py:149`) requires
   `alive_count >= num_paid + 1 = 4`, so the rebel value-net + k=200 blueprint have ZERO
   training exposure to 2-alive states. Even if the convention were fixed, the model's
   decisions on heads-up are unreliable.
2. The Double-Up top-3 deployment format terminates at 3-alive (the bubble is at 4, the
   match ends when only 3 remain — all 3 have cashed). So n_alive=2 is NEVER reached during
   play in the deployment context.
3. Fixing the convention is a `to_inner_game_string_for_state` change that's coupled with
   the ghost-ante fix (same function, different shorthanded-accounting bug); see the
   companion "Integration ghost-ante bug-match workaround" entry above. Both fixes should
   land in the same training cycle to avoid retraining twice.

**Mitigation in the integration:** the parser drops `n_alive < 4` frames as
`ScraperDataQuality` (matches both the model's training-distribution lower bound AND the
deployment format's playable range). See `scraper_schema.parse_frame` near the alive[]
derivation.

**QUEUED for the next training cycle:** when the ghost-ante fix lands (Option A from the
companion entry), ALSO fix the heads-up SB/BB convention to match real-poker rules:
`sb_seat = dealer_seat` and `bb_seat = alive[(dpos+1)%2]` for `n_alive=2`, with
`preflop_actor = sb_seat + 1` and `postflop_actor = bb_seat + 1`. Update the
`if n_alive == 2:` branch's comment to "SB acts first preflop, BB first postflop" (the
standard rules). Add tests for the heads-up case. Remove the `n_alive < 4` drop if the
retrained model is exposed to heads-up samples; keep it otherwise. Document the test
expectations alongside the ghost-ante test updates so both library bugs are fixed and
verified together.

## Phase 2 integration: chip-translation BRIDGE chosen over library rewrite + retrain

**Date:** 2026-06-04. **Status:** chosen path for log-only deployment; library
rewrite queued as fallback if drift evidence proves bad.

### Context — three paths considered

After the bug-match workaround (the entry above) carried Phase 1 hand-start
integration to 100% on the live9 corpus, Phase 2 (mid-hand hero-to-act replay)
opened with 28.87% pass rate. Diagnosed root cause: OpenSpiel min-raise =
2 × `inflated_BB` = 110 chips at level 1, but real-poker min-raise = 2 × `BB`
= 50. Any scraper frame containing a real min-open (chip_int ∈ [50, 109]) was
rejected as an illegal action by OpenSpiel during action replay.

Three paths considered:
1. **Option A — fix library + retrain.** Rip out the entire inflated_BB
   convention from `game_strings.py`, `cfr6.py`, `stack_sampler.py`, the
   blueprint, and the rebel value net. Retrain everything. ~1–2 weeks.
   Clean but expensive; invalidates existing checkpoint bake-off.
2. **Option B — restore the original workaround.** Keep the bug-match,
   accept the 28% mid-hand pass rate. Cheap but Phase 2 unviable.
3. **Option C — BRIDGE** (chosen). Add a translation layer in the
   integration that maps real-poker chip_ints ↔ OpenSpiel chip_ints. Cost:
   1–2 days. Trade-off: bet-sizing drift on the model's output (see below).

### Why the bridge

The trained model and OpenSpiel game-string convention agree on:
- starting_chips (1500 real = 1500 inflated)
- pre-action pot (designed-preserved: 70 chips at level 1 in either space)
- pot-relative bet fractions (BET_33, BET_50, ..., BET_200 are denominated
  in `pot * fraction`, not in BB multiples)
- stack-relative ALLIN (denominated in `view.max_bet` = hero's stack)

They disagree only on:
- BB-denominated quantities: `min_bet`, `min_raise` (each = `inflated_BB`
  or 2 × `inflated_BB`, ≈ 2x the real-poker equivalent)
- The "ante" line item, which the inflated game string folds into the BB
  (see entry "Integration ghost-ante bug-match workaround")

Because pot and stack anchors are identical, the resolver's qualitative
decisions translate correctly across the inflation. The drift lives in
one specific place: bet sizing on the model's RAISE outputs (chip_int
≥ 2). See "Drift profile" below.

### Bridge layer (`src/nlhe/integration/translate.py`)

Forward (real → OpenSpiel, for action replay):
- `real_to_openspiel_action(real_chip_int, legal_actions)`:
  - fold (0) and call (1) pass through unchanged
  - raise (≥ 2): if `real_chip_int ≥ openspiel_min_raise`, pass through
    (clamping to max raise); if below, bump up to `openspiel_min_raise`
  - degenerate states (no fold/call/raise legal) raise `ValueError`

Reverse (OpenSpiel → real, for client action):
- `openspiel_to_real_action(openspiel_chip_int, scraper_min_raise,
    scraper_max_raise, scraper_facing_bet)`:
  - fold and call/check: emit kind + null chip_amount
  - raise: clamp `openspiel_chip_int` to `[scraper_min_raise,
    scraper_max_raise]`. The model's chip_int is used directly as the
    real-table chip amount. This is the bet-sizing drift channel.

### Mid-hand invariant (bridge-aware)

The strict per-seat chip-equality check from Phase 1 is no longer
applicable: by construction, OpenSpiel's reconstructed chip values are
inflated relative to the scraper's. The mid-hand invariant
(`check_mid_hand_invariant`) instead requires:
1. `current_player == hero_seat` (right player to act)
2. `private_cards == scraper.hero_cards` (right hero hand)
3. `public_cards == scraper.board` (right board)
4. `legal_actions` consistent with `hero_facing_bet` (if facing, FOLD
   must be legal)
5. **Scraper self-consistency**: scraper.pot_total agrees (within an
   integer-divide tolerance of `n_anf - 1` chips) with the chip-
   conservation arithmetic implied by the frame's own per-seat fields

Tolerance band reasoning: the simple-model preflop_commit derivation
integer-divides residual chips among alive non-folded seats. Small
remainders (≤ n_anf - 1 chips) mean alive seats didn't all match at
exactly the same preflop level — fine for the bridge since exact chip
matching is already drift-tolerant. Larger gaps indicate real scraper
inconsistency and trip the check.

### Actual drift profile (NOT the 2x straw figure)

The "2x" drift commonly miscited is the **min-raise constraint**
inflation (110 vs 50 chips at level 1), not typical bet-sizing drift.
Pot-relative and stack-relative sizing have NEAR-ZERO drift because the
underlying anchors agree.

Drift channels, ordered by impact:

1. **Preflop opening sizing — DOMINANT.** At level 1, pot=70 and
   min_bet=110. Every pot fraction up through 1.5pot (=105 chips) falls
   below min_bet → unavailable to the model. The model's smallest legal
   open is BET_200 (140 chips ≈ 5.6x real BB) or ALLIN. The model never
   learned to open 2-3x BB (those actions never existed in its training
   action set). When the model opens, the client opens to ~140 chips.
   This is over-large vs typical real-poker opens (2-3x BB), but it's
   what the model was trained to do.

2. **Min-bet on small postflop pots.** Model can't bet less than 55
   chips (1 × inflated_BB). For flops with pot ≥ ~165, this is below
   0.33pot and the constraint disappears.

3. **Postflop pot-fraction bets — NEAR-ZERO DRIFT.** OpenSpiel pot
   matches real pot exactly along the action path (1:1 chip-int flow
   for any action ≥ min_bet). 0.66pot c-bet = 0.66 × actual_pot in
   both spaces.

4. **Pot inflation downstream of an over-large preflop open.** If
   preflop saw a min-open (140 chips), the flop pot is ~50-100%
   larger than it would be with a real-poker 2.5x open. Bet
   FRACTIONS of that larger pot are still correct; bets just look
   absolutely big because the pot got bigger.

### Why this is acceptable for double-up format

- Uncontested over-opens cost ~0 EV (collect 70 chips of dead money
  with any winning frequency; 140 vs 50 doesn't matter when uncalled)
- Late-game SNG already plays push/fold-style, where over-sized opens
  vanish into all-in repertoire (BET_200 ≈ ALLIN for short stacks)
- Top-3-equal-pay incentivizes survival over max-EV pots; over-opens
  that fold out marginal callers are slightly +EV in ICM terms

The cost concentrates on early levels (1-3) when called: the bot opens
~140 chips, gets called by a tighter range than the model trained
against, and plays a bloated OOP pot vs a strong range.

### Deployment as a MEASUREMENT step (log-only)

The bridge is shipping log-only — predict, log, do NOT click. The
log-only deployment is specifically designed to measure whether
preflop-open drift is tolerable.

**Watch-list (the single specific drift indicator):**
> Track preflop open-fold-frequency-when-called at levels 1-3. If
> the bot's preflop opens are getting called at a high rate AND
> the bot then folds/checks-to-give-up on the flop at a high rate,
> THAT is the signal that the over-large opens are inflating pots
> vs tight callers — i.e., the bridge's drift is materially costly
> and the library rewrite + retrain trigger has fired.

If logs show the opposite (opens fold out / hold up vs callers /
postflop play looks reasonable), the bridge ships permanently and
the rewrite is never built.

### Library rewrite queued as fallback (NOT chosen now)

The full rewrite stays on deck:
- Remove `inflated_big_blind` from `BlindLevel`
- Add real `ante` parameter to game-string builder (universal_poker
  supports it; we need to thread it through)
- Update `cfr6.py`, `stack_sampler.py`, `infoset6.py`,
  `to_inner_game_string_for_state` to use the real ante
- Retrain blueprint + rebel_value_net
- Delete the bug-match workaround in `invariant.py` AND delete
  `src/nlhe/integration/translate.py`
- Update `derive_action_sequence` to skip translation

Trigger conditions (any one):
- Log-only shows preflop open-fold-frequency-when-called significantly
  worse than self-play baseline at levels 1-3
- Specific bad-decision pattern in logs (e.g., model min-opens AKo,
  gets called, c-bets 0.66pot into bloated pot, loses to JJ that
  3-bets pre in a non-inflated world)
- User decides the qualitative bet-sizing artifacts are intolerable

### Validation

Bridge implemented in `src/nlhe/integration/translate.py` (16 unit
tests). `derive_action_sequence` uses forward translation. Mid-hand
invariant rewritten as 4 load-bearing checks + scraper self-consistency.

Corpus results (live9.jsonl, 288 records, all 6-handed):
- Phase 1 hand-start: 100% (32/32) — unchanged regression check
- Phase 2 mid-hand: 80.41% raw (78/97). Below the 99% gate, but the
  remaining 19 failures decompose as:
  - 13: scraper-side `hero_cards` empty at hero-to-act state (data
    quality, not bridge — would shift to `data_quality` skip if the
    classification were updated, giving 92.86%)
  - 3: `scraper_self_consistency:pot` (real scraper chip-arithmetic
    inconsistency, safely rejected)
  - 2: action sequence terminated before hero's decision (edge case)
  - 1: `legal_actions:fold_when_facing_bet` (state mismatch, safely
    rejected)
- All 50 integration unit tests pass

ALL FAILURES ARE SAFE REJECTIONS (return fold/check via `safe_action`),
not wrong decisions. The bridge's correctness mechanism is intact: when
we can't reconstruct the state with confidence, we reject rather than
guess.

## Process learning: iter_500 throwaway probes are NOT valid for absolute deployment judgments

**Date:** 2026-06-05. **Context:** the 8-cycle depth-distinction
investigation (depth-blindness → sampler rebalance → depth feature →
combined fix → capacity wall scare → trajectory-average → apples-to-
apples → sample-mode probe → convergence probe).

### The error

Throughout the investigation, every "diagnosis" of model character
(over-jamming, depth-blindness, polarization, oscillation, premium-
folding) was based on iter_500 throwaway models. The production-validated
k200 was trained to iter_2000. The throwaways were ~25% of production
training depth.

When the converged k200 iter_2000 model was finally probed in sample
mode at the end of the arc, it showed:
- Premium-fold rates of 3-7% (vs the iter_500 throwaways' 17-28%)
- A clean depth-distinct mixed strategy: AA at 60bb plays 64% normal-
  raise / 19% jam / 14% call / 4% fold; at 10bb plays 48% normal-raise
  / 26% jam / 20% call / 7% fold
- No signs of the "polarization" or "depth-blindness" that drove the
  middle six cycles of the investigation

**The iter_500 throwaway pathologies (premium-fold rates, oscillation,
flattened depth-distinction) were not properties of the architecture,
features, distribution, or capacity. They were artifacts of an
under-converged Deep CFR policy.** The strategy buffer in Deep CFR
saturates around iter_700-800 in this configuration; below that, the
average policy is still being shaped by early-training noise, and any
probe of "what the model has learned" reads transient regret dynamics
rather than the converged equilibrium.

### What's actually valid at iter_500

iter_500 throwaways are valid for:
- **Relative comparisons across configurations**, IF the comparison is
  at the same training stage AND the metric being compared is robust to
  oscillation magnitude (e.g., "does the gap exist directionally" is
  more robust than "is the gap > N pp")
- **Smoke tests** that the training pipeline runs end-to-end
- **Sanity checks** that a code change didn't introduce a catastrophic
  regression (model still trains, still produces a policy)

iter_500 throwaways are NOT valid for:
- **Absolute deployment judgments** ("can we ship this model?")
- **Architecture/feature ablations** that depend on converged behavior
  (the converged model's behavior differs qualitatively from the
  iter_500 transient)
- **Gate-against-thresholds** anchored on production model behavior
  (the threshold reflects a converged model; the test reads an
  unconverged one)

### How to recognize the failure mode

Symptoms of "reading an unconverged Deep CFR policy as if it were the
converged equilibrium":
- High premium-fold rates (>10% on AA/KK is a strong tell)
- Large checkpoint-to-checkpoint oscillation in argmax behavior
- Trajectory-average values significantly different from single-ckpt
  reads
- Sample-mode play that looks qualitatively wrong (e.g., folds best
  hands, raises worst hands) even when the configuration matches a
  validated production recipe

When these appear in a throwaway probe, the first hypothesis should
be undertraining, not architectural or distributional failure. The
cheapest discriminator: probe the production iter_N model (where N
matches the validated convergence point) under the same protocol. If
the converged model behaves correctly under that protocol but the
throwaway doesn't, the gap is convergence.

### Cost

The 8-cycle arc consumed ~20+ hours of session time + ~10 hours of
training compute, all chasing what turned out to be the same root
cause: iter_500 was the wrong observable. Two cheap probes at the start
would have prevented it:
1. Probe the production k200 ckpt under sample mode at the same depths
   before launching any throwaway investigation
2. Train a single throwaway to iter_2000 (full convergence) before
   trying to diagnose anything from iter_500 behavior

### Carryover for future investigations

When a future probe shows model behavior that "looks broken" on a
throwaway:
- **First check**: probe the production-scale (convergence-time) model
  under the same protocol. If it's fine, the throwaway is undertrained
- **Then**: only after the throwaway has been trained to convergence,
  treat the probe's output as load-bearing
- **Default assumption**: "this iter_500 model looks weird" is a
  convergence problem, not an architecture/distribution problem,
  until proven otherwise

The single real bug surfaced by the 8-cycle arc was the deployment
flag (argmax → sample, see this file's `fix(deploy)` commit). The
inflated_BB action-set patch was a different, earlier finding. Both
fixes are independent of distribution tuning or depth features — both
of which the data ultimately said weren't needed.

## k=1000 convergence proxy: compare against k=200 committed baseline

**Date:** 2026-06-05. **Context:** decision deferred for the eventual
k=1000 RunPod run, not the current k=200 retrain. Recording now so it's
available when we pull onto RunPod and not re-derived under time pressure.

### The k=1000 proxy strategy

The k=1000 convergence proxy must compare **k1000-current vs the
COMMITTED k200 baseline** — not just adjacent k1000 checkpoints. Log
per checkpoint: margin in chips/hand of k1000-current vs k200-committed,
played over a fixed CRN hand set.

This is the cumulative "is k=1000 actually better than the reference"
signal. Read two ways:

- **(a) margin positive** = k1000 is beating the baseline at all (i.e.,
  the larger card abstraction is buying something at this point in
  training; if it's not, k=1000 isn't earning its compute)
- **(b) margin stopped growing** = k1000 has converged to its
  equilibrium edge over k200; further iterations don't add advantage

Keep the **adjacent-delta proxy** (current ckpt vs the ckpt at half its
iter) from the k=200 monitor recipe — that pulse tells you "still
moving" vs "stuck/oscillating." But the **vs-baseline margin** is the
load-bearing signal for both questions that matter:
- Was k=1000 worth running at all? (sign of the margin)
- When is k=1000 done? (the margin's trajectory flattens)

### How to READ all margins (process discipline)

Read ALL margins as **trajectories / column trends**, never single
deltas. The 8-cycle investigation observed ±15pp swings in the
depth-gap metric between adjacent checkpoints under unchanged
training — single readings carry that much noise. **One reading
lies; the shrinking/flattening series is the real signal.**

Practical rule: for any "is it converged" question, plot the last
5+ checkpoints and look for plateau in the linear-fit slope.
Single-point reads of "margin = 0.18" or "depth_gap = 11pp" tell
you almost nothing about convergence; the slope over many points does.

### Why fold rates lie at k=1000 in particular

The previous k=1000 run was undertrained at iter_527. The fold rates
on premium hands hit their competence floor (≈ correct, not catastrophically
folding aces) much earlier than the full strategy converges — likely
because the policy-net's first learning task is "don't fold strong
preflop hands," which gets handled relatively quickly even on a
large card abstraction. Subsequent training is doing finer-grained work
(postflop value, mixed-bet sizing, blocker-aware ranges) which doesn't
show up in fold rates at all.

**Implication: at k=1000, watch the vs-baseline MARGIN, not the fold
rates, for when to stop.** Fold rates hit the competence floor early
and falsely look done. The margin keeps growing until the policy is
genuinely converged. **k=1000 needs MORE iterations than k=200 to
converge** because the abstraction is finer and the strategy space
larger.

Concretely: do not stop k=1000 just because the iter_500 / iter_800
behavioral probe looks like the k=200 baseline's converged behavior.
Stop only when the vs-k200-baseline margin trajectory has flattened
across multiple consecutive checkpoints (3+).

### Why this entry is here

This decision matters because we're about to pull the validated k=200
recipe to RunPod for the k=1000 run. The convergence-monitoring tool
built for the k=200 retrain (`scripts/monitor_k200_convergence.py` and
its `convergence_log.csv`) becomes the template for the k=1000
monitor. The k=1000 monitor needs ONE additional column:
`margin_vs_k200_committed_chips_per_hand`, populated by playing the
current k1000 ckpt vs the shipped k200 ckpt over a fixed CRN seed set.

The earlier k=1000 run that stopped at iter_527 lacked this proxy. It
saw "fold rates look fine, behavioral probes look reasonable" and
called it done. Without the vs-baseline margin trajectory, there was
no way to see that the run was still climbing. Don't repeat that.

---

## Depth-confusion in the deployed model — short-stack floor SHIPPED, retrain QUEUED

**Date:** 2026-06-08
**Model:** ckpt_iter_1500.pt from runs/k200_real_ante_20260605_225847_PRESERVED/
**Hash:** sha256 = b79e82dd0ce9e78e4eb666b7379df953dadbf2a6e026c6bd4b6eec695e9b1b11

### Finding

The 236-d feature vector normalizes every chip quantity by
`starting_stack = 1500` and never uses `big_blind`. The strategically
load-bearing quantity in tournament poker is effective stack in BB —
but the encoder cannot represent this directly. The YAML comment in
`configs/ignition_double_up_6max_turbo.yaml:56-66` already identified
this gap ("the missing piece is the depth-conditioning FEATURE").

### Probe — what the gap actually does

`scripts/depth_invariance_probe.py` measured how the model's strategy
varies across blind levels at fixed BB-depth (it should be constant)
and at fixed chip count (it should vary):

- **Fixed-BB sweep** (169 hands × 3 positions × 2 scenarios × 3
  depths × levels {L2, L5, L8}, true depth identical): median TV
  distance **0.53–0.78**, primary action flips **17–99%**.
- **Fixed-chips mirror** (750 chips at L1/L5/L8 = 30/3.75/1.25 BB,
  depth varies 24×): median TV **0.20–0.37**, flips **2–39%**.

Ratio fixed-BB ÷ fixed-chips: **1.55–3.16**. A depth-aware model
should have ratio << 1; depth-blind would be ≈ 1. Observed **> 1**
means the model is **depth-CONFUSED, not just depth-blind** — it
encodes level via the chip-magnitude smear (pot/to_call/contribution
all `/1500`) and the smear correlates wrongly with strategy.

Concrete example: AsAh BTN-unopened at true 4.8 BB. At L2 (250 chips)
the model plays jam-heavy push/fold (1.1% fold). At L8 (3000 chips,
identical true depth) the model folds AA **20.4%** of the time — the
same anomaly the seq=48 / seq=227 AA-fold probes hit live.

### A/B — what the gap costs in real ICM games

`scripts/short_stack_floor_ab.py`: paired ICM games (V0 = raw,
V1 = mask intermediate bet sizes at ≤6 BB) against 5×V0 opponents in
full escalating-blind matches. 24,000 paired games per arm. The
**hpl=5 arm matches the observed live Ignition turbo cadence** (Games
2 + 3 of 2026-06-08 dry-runs averaged 5 hands/level); hpl=3 was kept
as a faster-escalation sensitivity check.

| metric | hpl=3 (stress) | **hpl=5 (live-matched)** |
|---|---|---|
| floor-firing rate (% V0 hero decisions) | 14.40% | **7.53%** |
| % games diverged | 6.03% | **3.91%** |
| all-games paired Δ V1−V0 ± SE | +0.01842 ± 0.00317 (z=5.81) | **+0.01000 ± 0.00255 (z=3.92)** |
| diverged-only paired Δ ± SE | +0.305 ± 0.052 (z=5.88) | **+0.256 ± 0.065 (z=3.95)** |

Diverged-only delta is escalation-invariant within noise — the
per-firing physics is escalation-independent. All-games delta scales
with firing rate, which scales with how much time hero spends at
≤6 BB. **Live-field headline: +0.0100 ± 0.00255 ICM/game** (z=3.92,
95% CI [+0.005, +0.015]). ≈1% ICM/game, ~6σ from noise.

V0 incoherence at floor-fire spots (mass on intermediate bet sizes
that the floor masks): mean **6.6%**, but **11.81% of decisions
have >20% intermediate-bet mass** (seq=461-class errors — the model
emitting min-raise / pot bets at 5-BB push/fold depth). 3.08% have
>40% (strong); 0.79% >60% (severe). The headline gain comes from a
minority of decisions being severely incoherent.

H7b dose-response confirms the effect is concentrated where the floor
fires multiple times: 0-fires games have delta=0 by construction;
≥4-fires games have mean delta **+0.0153 ± 0.0044**. Low-fire games
are no-ops (delta ≈ 0 within noise) — so there's **no downside risk**
to enabling the floor universally.

### Decision

**Ship now (this commit):** short-stack floor + check-when-free floor
added to `src/nlhe/integration/live_loop.py`, chained after the
existing AA/KK preflop floor:

  1. **AA/KK preflop floor** (existing): mask FOLD when hero holds AA/KK
     and street_idx == 0.
  2. **Check-when-free floor** (NEW): mask FOLD when CHECK is legal
     (to_call == 0 with CALL in legal_actions). Universal — any street,
     any depth, any hand. Strictly dominated action.
  3. **Short-stack floor** (NEW): when hero's effective stack
     (= min(my_stack, max alive opp stack)) ÷ BB ≤ 6.0 (configurable
     `short_stack_floor_bb` kwarg, default 6.0):
        - facing action (to_call > 0)            → keep {FOLD, CALL, ALLIN}
        - to_call == 0 with CHECK legal          → keep {CALL, ALLIN}
        - to_call == 0 with CHECK illegal (rare) → keep {FOLD, ALLIN}

Each floor logs to stdout when it fires (level, eff-BB, current_player,
street, pre/post argmax) for live dry-run audit. Default off in
training and eval; the live path's composed filter is the only consumer.

**Queued (next major workstream, NOT before next dry-run):** retrain
with `eff_stack_in_BB` added to the encoder. The encoder change is a
one-feature addition: drop the depth-confused smear as the model's only
depth signal, provide a clean BB-normalized depth channel. This is the
real fix; the floor is a deployment patch that lower-bounds the cost
until then.

### Falsification test for the retrain

Re-run `scripts/short_stack_floor_ab.py` against the retrained model
with the SAME paired-seed harness. A depth-aware model should:

  - **firing-divergence rate drops toward zero**: V0 and V1 produce
    identical actions because V0 already plays push/fold at ≤6 BB
    (the floor's mask becomes a no-op);
  - **diverged-only delta drops toward zero with widening SE**: the
    few divergent decisions left should be approximately random
    (V0's residual errors don't predict V1's wins);
  - **all-games delta indistinguishable from zero** (within SE).

If the retrained model still shows diverged-only delta near +0.25, the
encoder change didn't deliver depth-awareness — investigate before
shipping the retrain.

### Why a floor is enough for live dry-runs

The validated model still won the 23/24 ICM bake-off — depth confusion
costs strategic correctness on the marginal short-stack spots, not the
median spot. The seq=461-class error rate at field scale is **0.89%**
of all V0 decisions (11.81% of 7.53%). The floor turns those incoherent
0.89% into push/fold, which we know is the locally-correct strategy at
that depth. Everything else is left untouched.

### What's NOT decided

Whether to add other "categorically never +EV" floors (QQ-vs-non-3bet
preflop, 4-bet stacks, etc.) before retraining. Each requires its own
empirical justification — the AA/KK rule had a clear "no opponent
range" argument; check-when-free has a strict-dominance argument; the
short-stack floor has the depth-invariance + A/B evidence. Other
floors should clear the same bar.

---

## Bridge reconstruction (Issue 2) — closed

**Date:** 2026-06-08
**Session log:** `logs/live_dryrun_20260608_152756.jsonl` (200 frames)

### Background

Live dry-run safe-folded AhAs on two spots (seq=170 preflop and seq=192
turn). User flagged these as a class of bridge bug blocking the bot
from playing good hands. Re-replay against the saved log via
`scripts/replay_session_diff.py` confirmed exactly **2 invariant
failures** out of 200 frames — both AA spots, no other reconstruction
failures hiding in the data.

### Two distinct defects, same symptom class

**seq=170 — derive-side: blind-seat misclassified as voluntary limper.**
`src/nlhe/integration/scraper_schema.py:derive_action_sequence`'s
`limper_after_me_unemitted` check treated the BB sitting at exactly
`bb_amount` as a voluntary limper waiting for emission. Combined with
`raiser_after_me_exists`, this caused early-position raisers
(UTG at target=100) to defer their open and emit `chip_int=1` (call)
instead of `chip_int=100` (raise) — attribution flipped onto a later
seat (CO at target=100 became the apparent raiser). OpenSpiel
contribution off by exactly the BB amount (50 chips) on UTG.

**Fix:** exclude blind seats sitting at exactly their forced post from
the `limper_after_me_unemitted` set when `raise_above_bb` is False. A
real (non-blind) limper still defers correctly — verified by
`test_blind_seat_exclusion_preserves_real_limper_behavior`.

**seq=192 — invariant-view-side: postflop matched_all_in branch used
wrong subtractor.**
`src/nlhe/integration/invariant.py:openspiel_to_scraper_view`'s
matched_all_in branch on postflop subtracted `preflop_max_chip_int` to
isolate current-street voluntary. That subtractor equals the busted
seat's full pre_hand (= ante + preflop_carry + current-street voluntary)
when the all-in happens postflop, over-subtracting the current
voluntary to 0. The diagnostic
`scripts/diag_seq192_legal_actions.py` confirmed
OpenSpiel's `legal_actions()` at BB's turn decision collapses to
`{0, 1, pre_hand}` — the chip_int=pre_hand convention is the only
legal all-in, and the convention reintroduces the ante into `contrib`.

**Fix:** postflop matched_all_in branch subtracts
`preflop_commit_per_alive + ante`, structurally consistent with the
existing preflop matched_all_in branch (which already subtracts
`ante`). The `- ante` term is motivated by the `chip_int=pre_hand`
convention, not by the replay-fallback inflation — it applies whether
the chip_int=pre_hand was emitted by the forced-all-in fallback
(seq=192-style) OR by `derive_action_sequence`'s `busted_mid_hand`
emission path.

**The `chip_int=1515` value is NOT in OpenSpiel's legal_actions** — not a
fallback-detection bug. The forced-all-in regime collapses legal to
fold/call/full-stack only. The fallback's behavior is correct; the
fix lives in the view.

### Re-replay gates

Pre-fix:
  invariant_pass: 31, invariant_fail: 2 (seq=170, seq=192)

Post-fix:
  invariant_pass: 33 (+2 = exactly seq=170 + seq=192)
  invariant_fail: 0

Diff (`scripts/replay_session_diff.py diff`):
  196 / 200 frames bit-identical (status, invariant_ok, deltas,
    action_seq_hash, state_hash all match).
  2 frames invariant FIX: seq=170, seq=192 → invariant_pass.
  2 frames (seq=81, seq=83): action_seq_hash changed but state_hash
    unchanged — alternate equivalent emission sequence reaching the
    same OpenSpiel state (same policy, same bot decision). Walked
    the logic: at these spots the limper-rule change reorders the
    chronological emission but `should_defer` evaluates to the same
    final value, so the final chip distribution is bit-identical.
  0 invariant regressions (no PASS → FAIL).

### AA decisions verified

seq=170 (preflop SB AsAh, dealer s5, facing UTG-open to 100): real
mixed-strategy raise/call/fold. With the AA/KK preflop floor active
the residual fold mass is masked. Not a safe_fold.

seq=192 (turn SB AsAh on QdQsJc8c, facing BB shove for 1015, hero
stack 250): forced-all-in regime, legal collapses to
{FOLD, CALL, ALLIN}. Policy: 86.7% CALL, 6.7% FOLD, 6.7% ALLIN. Bot
calls the all-in for 250 chips with AA. Not a safe_fold.

### Bridge vs. scraper split — what this fixes, what it doesn't

Of the 200 frames in the session log:

  126 (63%) — `not_hero_to_act`: between hands, opponent acting; normal.
   33 (16.5%) — `invariant_pass`: hero decision frames; bridge works.
   41 (20.5%) — `scraper_data_quality`: scraper-side issues (dealer
                missing/empty, dealer points to non-alive seat, blinds
                parse).
   17 (8.5%) — `scraper_suspect`: scraper marked the frame as suspect
                (image rendering / OCR confidence below threshold).
    3 (1.5%) — `parse_error`: scraper output didn't conform to schema.

**Pre-fix invariant failures: 2 (seq=170, seq=192). Post-fix: 0. The
bridge cleared 2/2 = 100% of its known failures.**

The **61 non-reconstructing frames (30.5% of the session)** that remain
are ALL **scraper-side**, not bridge:
  - 41 scraper_data_quality (data missing or inconsistent from scraper)
  - 17 scraper_suspect (scraper itself marked the frame unreliable)
  - 3 parse_error (schema violation in scraper output)

These are a **Windows-side workstream** (the scraper runs on Windows;
the bot reads its output over a socket). Improving the scraper is
out of scope for the bridge code at `src/nlhe/integration/*`.

### Tests

`tests/test_bridge_seq170_seq192_fixes.py` (6 new):
  - test_seq170_utg_open_attribution_after_fix
  - test_seq170_replay_invariant_passes
  - test_blind_seat_exclusion_preserves_real_limper_behavior
    (real non-blind limpers still defer correctly)
  - test_seq192_postflop_matched_all_in_bet_recovered
  - test_seq192_preflop_matched_all_in_unchanged
  - test_postflop_no_all_in_unchanged

All 64 tests on touched modules (scraper_schema, invariant, replay,
premium_floor, short_stack_floor, bridge fixes) pass.

---

## Correction — bridge-vs-scraper split: 41 frames, not 61

**Date:** 2026-06-08
**Re:** prior entry "Bridge reconstruction (Issue 2) — closed" and commit f1a412b

The prior entry and commit `f1a412b`'s message both stated the session's
non-reconstructing frames as "**41 scraper_data_quality + 17
scraper_suspect + 3 parse_error = 61**" — that math double-counts.
The original `live_dryrun_20260608_152756.jsonl` used `skip_data_quality`
as a single UNION bucket containing all three exception types
(`ScraperSuspect`, `ScraperDataQuality`, `ScraperParseError`). The
correct partition is:

| sub-reason | count | % of session |
|---|---|---|
| `scraper_suspect` (image render / OCR confidence below threshold) | 17 | 8.5% |
| `data_quality: dealer field missing/empty` | 17 | 8.5% |
| `data_quality: dealer points to non-alive seat` (transient mid-hand) | 4 | 2.0% |
| `parse_error: blinds string parse failure` (level transitions) | 3 | 1.5% |
| **total scraper-side frames** | **41** | **20.5%** |

Live dryrun's broader status breakdown remains:

```
not_hero_to_act       126   63.0%   (normal: between hands, opponent acting)
invariant_pass         33   16.5%   (post-fix; was 31 pre-fix)
scraper_side total     41   20.5%   (the 41 above)
invariant_fail          0    0.0%   (post-fix; was 2 pre-fix: seq=170, seq=192)
```

(`63 + 16.5 + 20.5 + 0 = 100`. The bridge cleared 2/2 = 100% of its
known failures.)

### Hand-touch analysis — 0 hands sat out due to scraper

The session segments into **10 hand-segments** (by `(dealer_seat, level)`
across the 9.1-minute dry-run). Distribution:

| metric | count |
|---|---|
| Total hand-segments | 10 |
| Hands with ≥1 `invariant_pass` (bot got a real decision) | **9 pre-fix / 10 post-fix** |
| Hands with ≥1 scraper-fail frame | 9 |
| Hands BOTH played fine AND had scraper-fail frames (= scraper-fail was noise around played decisions) | 9 |
| **Hands with 0 `invariant_pass` and ≥1 scraper-fail (= bot SAT OUT due to scraper)** | **0** |

**In this session, the 41 scraper-side frames cost zero playable hands.**
The redundancy of polling-rate captures absorbed every OCR/render
glitch as long as at least one frame within each hand was clean —
which it always was. The bridge's "drop and let the next frame retry"
behavior was the right fallback for this noise level.

The one hand where the bot DID effectively sit out pre-fix was the
last AA hand (level 2, dealer=5, containing seq=170 + seq=192) — but
that was a **bridge** failure (now fixed in `f1a412b`), not a scraper
failure. That hand had 0 scraper-fail frames; the issue was the
invariant rejecting valid frames due to the two reconstruction bugs.

### Scraper prioritization (if/when)

If scraper improvements are ever scheduled, **dealer-button detection
is the highest-leverage target**: 21 of 41 failures (51% of scraper
issues, 10.5% of all session frames) are dealer-related —
`dealer field missing/empty` (17) + `dealer points to non-alive seat`
(4). Fixing the dealer-OCR path would roughly halve the scraper
failure count.

The 17 `scraper_suspect` failures are heterogeneous (whatever triggers
the scraper's "suspect" flag — likely a mix of rendering and OCR
confidence reasons; not characterized here). The 3 `parse_error`
failures are blinds-string parsing at level transitions —
self-correcting on the next frame.

**That said, scraper work has zero urgency for the next dry-run.** The
present scraper noise floor did not block any hands in this session,
and the redundancy-tolerant bridge handles it correctly.


## Bridge reconstruction (Issue 3) — closed: seq=246 + seq=364 dead-SB + all-in-for-less

**Date:** 2026-06-09
**Session log:** `logs/live_dryrun_20260609_154557.jsonl` (571 frames)

### Background

Second live dry-run surfaced two new reconstruction failures the Issue
2 fixes did not cover: **seq=246** (invariant_fail, BTN all-in 2166 with
SB all-in for less 615, deltas off by ~2000 chips on BTN's
contribution) and **seq=364** (replay_error, hero is BB in an Ignition
dead-SB rotation where the natural-SB-this-hand seat busted between the
prior hand and this one). Re-replay against the 571-frame log via
`scripts/replay_session_diff.py --log` confirmed exactly those 2
failures out of 571 frames, both in the same all-in-for-less family.

### Four distinct defects, one user-visible frame each

**seq=364 — dead-SB rotation not modeled (frame-only detection required).**
Ignition uses forward-moving button: each hand the button advances one
position; SB is whoever was BB last hand; BB is the next alive seat
after that. When the player-who-was-BB-last-hand busts between hands,
this hand's SB position is dead (no SB posted, only BB). The bridge's
prior rule (`SB = next alive after dealer`) cannot distinguish this
case from a long-stable empty seat past the rotation
(seq=55: dealer=2, idx 3 empty for many hands, SB=idx 4 LIVE) — same
geometry, opposite correct answer.

**Fix:** `_detect_blinds_from_bb_post` (Predicate 1) in
`src/nlhe/integration/scraper_schema.py`. Walks alive seats clockwise
from `(dealer+1)` absolute, tracks the SB poster (bet==sb_amount) and
BB poster (bet==bb_amount). If a BB poster is found with no SB poster
between them and the dealer AND the absolute `(dealer+1) % NUM_SEATS`
seat is empty, dead-SB confirmed. Returns `(None, bb_seat)` for
caller to thread through. Falls back to the H1 default (next-alive-
after-dealer) when posting evidence is ambiguous (mid-hand frame where
SB has acted past the forced post) — those frames either reconstruct
under H1 or invariant_fail loudly (safe-fold).

**Frame-only design rationale:** the user rejected a session-history
button-position tracker because the scraper drops hands, and a tracker
would desync silently on a missed hand and produce a wrong-but-confident
SB/BB assignment — the worst failure mode. Predicate 1 reads the
posted-blind evidence actually on the table, so it self-corrects across
missed hands; ambiguous frames fall to the invariant → safe-fold.

**Threading discipline (load-bearing):**
`game_strings.to_inner_game_string_for_state` accepts `sb_seat` /
`bb_seat` parameters and uses them verbatim — it MUST NOT recompute
SB/BB from `(dealer, alive_seats)` independently. A second derivation
site would re-introduce the dead-SB misclassification at the OpenSpiel
game-string layer (the seq=55-class regression we observed when only
`scraper_schema` was patched). `_derive_blinds_and_action_order` in
`scraper_schema.py` is the single source of truth; `replay.py` threads
the resolved `(sb_seat, bb_seat)` through to game_strings.
`test_seq364_single_source_of_truth_game_string_consumes_resolved_sb`
guards this — asserts blind_array on the OpenSpiel game string matches
the Predicate 1 resolution, not an independent recomputation.

**seq=246 layer 1 — defer-for-smaller-raise triggered on a seat that
couldn't actually raise.** `defer_for_smaller_raise` in
`derive_action_sequence` would defer the BTN's all-in (chip_int=2166)
because SB had `target=615 > running_max=100` — but SB's max possible
voluntary commit was `pre - ante = 615`, which equals their target
(they're all-in for less). The defer caused BTN to emit `chip_int=1`
(call) instead of the raise, under-attributing BTN's contribution by
~2000 chips.

**Fix (Predicate 2):** add discriminator
`int(pre[s]) - ante > int(target[s])` to `defer_for_smaller_raise`.
Defer only to seats that could keep raising past their current target
(= max voluntary STRICTLY greater than target). Seats fully committed
at target (`pre-ante == target`) are all-in-for-less and chronologically
subordinate to a larger raise above their target. Refinement post-
seq=428 audit: `> target`, not `>= t` — the initial discriminator
incorrectly excluded partial-raise seats whose pre-ante was below t
but above their target (seq=428 hero 3-bet to 1200 with 281 chips
remaining = partial raise, not all-in).

**seq=246 layer 2 — call-for-less seat folded instead of called.** After
Predicate 2 unblocked BTN's raise, SB visited the loop with
`target=615 < running_max=2166` and took the `t < running_max → fold`
branch, emitting `chip_int=0` — losing SB's 615 committed chips from
the OpenSpiel pot. The branch collapsed two semantically distinct
cases (true fold with no voluntary commit vs. all-in-for-less call
below the running_max).

**Fix:** surgical discriminator in the `t < running_max` branch of
`derive_action_sequence`. If `pre[seat] - ante == t` AND `t > 0` AND
`frame.stack[seat] == 0` (= committed every chip available, stack-
capped at target), emit `chip_int=1` (call; OpenSpiel caps at remaining
stack so the committed chips land in the pot). Otherwise the original
defensive fold. The earlier proposal to broaden `busted_mid_hand` to
include alive-but-all-in seats was REJECTED in audit — it would have
routed raising-all-in seats (seq=97/106/170/427/428 currently passing)
through the `chip_int=pre_hand` convention, regressing the Class B
`chip_int=bet` contract they rely on.

**seq=246 layer 3 — OpenSpiel ante-absorption asymmetric between RAISE
and CALL all-in actions.** With the call-for-less emission correct, the
OpenSpiel state was chip-arithmetically right but the invariant's
`openspiel_to_scraper_view` compared off by 15 chips (= ante). Empirical
finding: the patched universal_poker absorbs ante (credits back into
`money[i]`, subtracts from `contrib[i]`) for RAISE all-in
(seq=170 BTN: `money=15`, `contrib=2809` excluding ante), but NOT for
CALL all-in-for-less (seq=246 SB: `money=0`, `contrib=630` including
ante). The existing `matched_all_in` branch already handles "spent
includes ante" via the no-ante-subtract formula, but its detection
`contrib >= preflop_max_chip_int` only caught the chip_int=pre_hand
convention.

**Fix:** extend `matched_all_in` detection in
`src/nlhe/integration/invariant.py:openspiel_to_scraper_view` to also
flag `frame_alive[i] AND money[i] == 0` — the unambiguous signature of
an all-in-via-call seat where the ante wasn't credited back.
Empirically verified safe by the 12-frame audit
(`/tmp/money_zero_audit.py` — 0 misfires: raising-all-in seats all
have `money >= ante`, only call-for-less seats have `money == 0`).

### Re-replay gate

Baseline (`logs/live_dryrun_20260609_154557.jsonl`, pre-fix):
- invariant_pass: 59
- invariant_fail: 1 (seq=246)
- replay_error: 1 (seq=364)

Post-fix (all five pieces together):
- invariant_pass: **61** (+2: seq=246, seq=364 both reconstruct)
- invariant_fail: 0
- replay_error: 0
- bit-identical to baseline: **448 / 450** non-changed frames
- changed frames: exactly 2 (both target fixes flipping FAIL/ERROR → PASS)
- ZERO regressions

Money==0 trigger audit across all 12 effectively-all-in frames in the
corpus: 0 misfires, correctly fires only on the 2 call-for-less seats
(seq=246/247 SB).

Test suite: 145 integration tests pass (7 new in
`tests/test_bridge_seq246_seq364_fixes.py` plus the synthetic Class B
test updated to assert the semantically correct chip_int=1 for the
call-for-less emission).

### Process lesson — end-to-end chronology trace BEFORE predicate fixes

**The principle:** when a frame fails at multiple pipeline layers,
trace its full real-game chronology end-to-end and diff against the
bridge's emission at every stage BEFORE writing predicates. Each
layer's corruption masks the next, and predicate-first fixing discovers
them serially.

**What happened on seq=246 (the cautionary tale):** four bugs in the
same frame, discovered one at a time. Each fix unblocked a new code
path that revealed the next defect:
1. Fix Predicate 2 (defer) → BTN now raises → revealed SB takes the
   fold branch (call-for-less emission gap)
2. Fix call-for-less branch → SB now emits call → revealed OpenSpiel's
   ante-absorption asymmetry (invariant view gap)
3. Fix invariant extension → revealed Predicate 2 was too strict (broke
   seq=428's partial-raise hero)
4. Refine Predicate 2 (`> target` instead of `>= t`) → all four
   pieces work together

**What would have surfaced all four in one analysis pass:** trace
seq=246's actual real-game chronology FIRST — hero BB posts, UTG/MP
fold, BTN all-in 2166, SB all-in for-less 615, hero to-act facing
2166 — then walk the bridge stage by stage:
- Action-sequence emitter: what does it emit for each seat in pf_order?
- OpenSpiel state after apply_action: what's contrib/money for each seat?
- Invariant view: how does it translate contrib → scraper_bet?
- Compare each stage's output to what the real frame shows.

Every divergence is a bug. All four would have appeared in this
trace before any predicate was written.

**Why predicate-first failed here:** I started at the defer rule
because it was the most visibly-suspicious code, fixed it, and assumed
that closed the failure. But the failure mode was a chain — the defer
bug was the topmost of four, each one masking the next. Without
having the full chronology vs. emission diff in hand, there was no way
to see the chain.

**The standard going forward:** for any multi-layer reconstruction
failure (= any frame where a single fix to the most-visible layer
doesn't immediately produce invariant_pass), do the full chronology
trace before writing or refining any predicate. The trace is cheaper
than a series of predicate iterations, each of which requires its own
audit/re-replay round-trip and risks regressing a previously-passing
frame.



## Bridge reconstruction (Issue 4) — closed: seq=101 postflop hero-stop / cycle-index-shift

**Date:** 2026-06-09
**Session log:** `logs/live_dryrun_20260609_192543.jsonl` (402 frames, first
post-5f24ed7 dry-run, first to exercise dealer=seat1 postflop frames).

### Defect

`derive_action_sequence`'s postflop emission loop mutated `pf_alive_post`
mid-iteration (`pf_alive_post.remove(seat)` on defensive fold). The
cycle's modular index (`cs_pos % len(pf_alive_post)`) shifted under
the removal and skipped past hero — and any seat between the removed
position and hero — on the next iteration. Two distinct symptoms:
hero-stop check missed (loop continued past hero's turn) AND a
fabricated downstream action emitted (the seat the cycle landed on
after the wrap, typically SB folding to its re-opened decision —
chronologically future relative to hero's decision).

seq=101 instance: 6-handed flop, dealer=hero=BTN, UTG/MP folded
preflop, SB/BB/CO/BTN limped. Flop: SB check, BB bet 98, CO defensive
fold. Correct sequence stops at hero (BTN). Pre-fix sequence emitted
an extra `(1, 0)` SB fold and replay failed with
`action_seq[9] expects seat 1 but state.current_player()=0`.

### Root cause vs. preflop equivalent

The preflop loop (lines 1499-1518) uses `folded_emitted: set` as a
per-seat filter; pf_order_alive is never mutated, the cycle's index
walk stays stable across folds. The postflop loop diverged from this
pattern by removing seats from the cycle list — a structural defect.

### Fix

`cs_folded: set` replaces `pf_alive_post.remove()`. The main loop's
top-of-iteration check (`if seat in cs_folded: continue`),
`cs_round_closed()`, the `unacted_eligible` comprehension, and the
hero-stop's `later_has_bet` scan all filter via the set. Mirrors the
preflop pattern. No mid-loop list mutation.

### Generality

The bug fired whenever any postflop seat defensive-folded at cycle
position k while hero was at position k+1 or later in the original
`pf_alive_post`. With multiple folds, the skip cascades — intermediate
folds get dropped from the action sequence, producing OpenSpiel state
divergence (seats appear "alive" in OpenSpiel that the scraper shows
folded). Affected positions: hero=BTN with any fold before, hero=CO/MP
with folds before, hero=UTG with folds before — i.e., any non-SB hero
facing defensive folds upstream in the postflop walk.

### Why it surfaced now

The seat1 zone fix (committed earlier this session) unlocked
dealer=seat1=hero-on-BTN frames. With hero at the END of the postflop
cycle, ANY upstream defensive fold triggers the bug. Prior corpora
had no dealer=BTN postflop frames; the bug was latent in every such
frame requiring a defensive-fold-before-hero — just hadn't been
exercised.

### Re-replay gate

Pre-fix on the verify-game log: 65 invariant_pass, 1 replay_error
(seq=101).
Post-fix: **66 invariant_pass, 0 replay_error**, 401/401 non-changed
frames bit-identical, zero regressions. seq=101 the only changed
frame (replay_error → invariant_pass).

Test coverage (4 class-coverage tests in
`tests/test_bridge_seq101_postflop_loop_fix.py`):
- seq=101 reconstructs (hero=BTN single fold)
- hero=CO with earlier defensive fold
- cascading: 3 folds before hero, ALL emitted in correct order
  (explicit assertion on the fold-seats-in-order, not just hero-stop)
- no defensive folds → bit-identical no-op guard

### Process principle — masking layers

This is the SECOND time in this project that removing an upstream
mask exposed a latent downstream bug:

1. The scraper's noise masked the original bridge reconstruction bugs
   (Issue 2 + Issue 3 seq=170/seq=192/seq=246/seq=364) — once frames
   reached the bridge cleanly, the bridge's own defects became visible.
2. The seat1 zone fix masked the postflop loop's `pf_alive_post.remove`
   bug (Issue 4 seq=101) — once dealer=seat1 postflop frames reached
   the loop, the iterator's structural defect produced wrong action
   sequences.

**The principle**: fixing a layer exposes the next layer's dormant
bugs. Code paths that were never exercised carry latent defects that
only surface when upstream stops masking them. A clean kick-risk
ratio on game N reflects the position space that game N happened to
sample — NOT a verified upper bound on the bridge's correctness.

**Operational consequence**: verification runs must span MULTIPLE
games before any kick-risk rate is trusted for autoclicker gating.
One game's kick-risk is a sample, not a measurement. Until the
position space is broadly exercised (= multiple games covering
varying dealer rotations, alive-seat configurations, and postflop
action shapes), each "clean" report is provisional.


## k200_real_ante iter-1500 stop — deliberate operator convergence call

**Date:** 2026-06-07 (recorded 2026-06-09). **Run:** `runs/k200_real_ante_20260605_225847` (target 2000 iters).

Training did NOT crash at 1500: the train log shows it ran to iter
1538/2000 and was killed externally at ~02:07 UTC with no traceback.
`ckpt_iter_1500.pt` is simply the last checkpoint boundary before the
kill; iters 1501–1538 were trained and discarded.

The stop was a convergence call backed by two evals run that night:
- `slope_1500_vs_1000` (5,000 paired-CRN hands): +0.0056 chips/hand,
  t=2.02 — barely-significant residual improvement over 500 iterations.
- `peakpin_1100_vs_1500`: 1500 beats 1100 (t=−2.76) — no earlier peak.
Plus flat adv_loss (0.636–0.644 over iters 1100–1500) in
`convergence_log.csv`.

Evidence preserved (copied from volatile /tmp on 2026-06-09):
`runs/k200_real_ante_20260605_225847_PRESERVED/stop_evidence/` (slope +
peakpin logs, train-log tail showing iter 1538, launch script). The
24-profile bake-off that used ckpt_1500 as its baseline is preserved at
`evals/bakeoff_20260607/`. Fuller doc corrections are a separately
queued task.


## Correction — hands WERE sat out due to scraper (sat-out count was a segmentation artifact)

**Date:** 2026-06-09. **Re:** the "Correction — bridge-vs-scraper split:
41 frames, not 61" entry (this file, ~line 1418) and the former
STATUS.md "Zero hands sat out due to scraper" claim (now corrected
inline in STATUS.md).

The hand-touch analysis segmented hands by `(dealer_seat, level)` —
blind to any hand whose EVERY frame lost the dealer button, which is
precisely what the dominant skip class does. Re-analysis using
`raw_record` hero-card/button evidence:

- `logs/live_dryrun_20260608_152756.jsonl` (the session the claim was
  made about): **1/11 hands sat out** — the 8dKc hand: seq=92 shows
  hero action buttons `['CALL','FOLD','RAISE']` but was killed by
  `dealer field missing/empty`; seq=97 (`['BET','CHECK']`) killed by
  `ScraperSuspect`; the hand produced 0 decision frames.
- `logs/live_dryrun_20260609_154557.jsonl`: **14/41 hands lost (34%)**
  — 12 missed-decision hands (incl. AdKd fully lost to suspect frames,
  seqs 106–118) + 2 safe-fold-only hands.

Root cause (positional, not stage-related): dealer-button OCR failed
whenever the button sat at the HERO's seat — raw `dealer` parsed as
seat1 ZERO times in every session through 154557 (e.g. 154557 raw
counts: `<empty>`:138, seat2:100, seat3:94, seat5:136, seat6:102,
seat1:0), so ~every hero-BTN orbit slot produced an invisible hand.
The apparent early/mid/bubble skip gradient (9.4→18.6→29.1% in 004502;
0→21.5→39.8% in 154557) is mechanically explained by orbit shrinkage
(hero is BTN 1/6 of hands at 6-alive but 1/4 at 4-alive) plus
dealer-on-dead-seat requiring a busted seat — not by bubble-specific
UI behavior.

## Scraper dealer-at-hero-seat fix — landed Windows-side 2026-06-09 (recorded here; fix lives outside this repo)

**Date:** 2026-06-09. The Windows scraper's dealer-button OCR could not
read the button at the hero's own seat (bottom-center). A fix landed
between 16:09 and 19:25 UTC on 2026-06-09 — i.e., after session
`live_dryrun_20260609_154557.jsonl` and before
`live_dryrun_20260609_192543.jsonl`. Evidence (log-derived; the scraper
code is not in this repo):

- raw `dealer == seat1` appears **76×** in 192543 and 16× in verify1,
  vs **0×** in every prior session;
- dealer-OCR skip rate collapsed **24.0% → 2.2%** of frames
  (154557 → 192543; verify1: 3.3%);
- first hero-BTN decision frames ever recorded (192543:
  `dealer_seat=0` on 10 decision frames).

This entry exists so the repo's record is not silent about a
load-bearing change that happened outside it; the bridge-side Issue 4
fix (`2c0b71b`) was surfaced by exactly these newly-arriving
dealer=seat1 frames.

## Correction — KillPhilMTT loss figures: two unrecorded caveats

**Date:** 2026-06-09. **Re:** SHIP_BAKEOFF.md:87-103 ("It is
structural; no non-adaptive fix exists in our constraint stack") and
the 24-profile bake-off table (evals/bakeoff_20260607/,
killphilmtt −0.0153 ± 0.0012, 12.7σ). SHIP_BAKEOFF.md itself is left
unedited as a historical record.

1. **All bake-off killphilmtt numbers were measured against a
   misconfigured opponent.** SHIP_BAKEOFF.md:194-196's own follow-up
   records that `opponentsattable` was NOT wired to the live
   alive-count for Shanky bots. KillPhilMTT's fold-or-jam mode gating
   is predicate-dependent (`FoldOrGoAllInWhenOpponentsAtTableLessThan
   = 5`, and its preflop tiers branch on `OpponentsAtTable`), so the
   −0.0153 figure is against a profile whose table-size predicates did
   not see the real table. Re-measurement with bubble-aware Shanky
   opponents is queued (C1a).
2. **Transfer to real Ignition opponents is unmeasured.** No
   population evidence exists anywhere in this repo that
   killphilmtt-style play occurs at measurable frequency among real
   Ignition opponents, nor that the loss was ever observed against a
   non-Shanky opponent. SHIP_BAKEOFF's "Real recreational poker
   populations are dominated by loose play" is an unsourced assertion.
   The structural claim that DOES generalize from the diagnosis
   (REBEL_KILLPHIL_DIAG_FINDINGS.md): any opponent whose shove range
   is materially tighter than the model's effective belief collects
   the same <10bb call-vs-shove EV.

## Deployment reversal record — deployed agent is the k200_real_ante blueprint, superseding SHIP_BAKEOFF's "ship rebel" verdict

**Date:** recorded 2026-06-09 (events 2026-06-04 → 2026-06-07). Closes
a gap in the record: SHIP_BAKEOFF.md (2026-06-04) verdicts "Ship
`rebel_value_net_full` (d3k150 resolver)", but the deployed agent is
the k200_real_ante blueprint
(`runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt`).
The reversal chain, reconstructed from the existing record:

1. **The Jun-4 bake-off was invalidated the next day.**
   HANDOFF_RETRAIN.md (2026-06-05): the inflated-BB action-set
   distortion was verified by direct policy query (2.4% BTN open rate,
   0% CO at deep stacks — "correct would be 40-50% / 25-35%"), and the
   handoff explicitly reopened the ship decision: "Re-decide the ship
   model from scratch: The prior winner (rebel_d3k150 …) is REOPENED —
   no longer the default ship candidate" (HANDOFF_RETRAIN.md:222-228).
2. **Only the blueprint was retrained under the corrected (real-ante)
   convention** (`scripts/train_k200_real_ante.py`, 2026-06-05→07).
   The rebel value net's retrain was queued "after blueprint
   stabilizes" (HANDOFF_RETRAIN.md:154) and has not happened; the
   resolver had independently been retired as a development path
   ("Blueprint-alone is the reference agent", this file ~line 520;
   "Foundation pivot", ~line 543; bubble-slice closure net-negative).
3. **The re-decided candidate validated:** convergence-monitored
   training, deliberate iter-1500 stop (this file, entry above), and
   the Jun-7 24-profile bake-off (wins 23/24;
   evals/bakeoff_20260607/). The iter_500-probe process learning
   (~line 964) confirmed the earlier "model looks broken" reads were
   under-convergence artifacts, not architecture failures.

**Honesty note:** no contemporaneous document says in one sentence
"k200_real_ante replaces rebel as the ship agent" — the rationale was
not fully recorded at the time; this entry reconstructs it from
HANDOFF_RETRAIN.md, the resolver-retirement entries, and the
validation chain, all of which are in-repo.


## 2026-06-09 wired bake-off — dual result: CRN reproducibility gate + surgical partial re-baseline

**Date:** 2026-06-09/10. **Artifacts:** `evals/bakeoff_20260609_wired/`
(json + log). **Context:** `78c71f9` wired `opponentsattable` to the live
alive-count (was hardcoded 5); this rerun replays the Jun-7 24-profile
bake-off (same master seed 2026, 10,000 hands/profile, sample mode,
real-ante, ckpt sha b79e82dd…) against the corrected opponents.

**Result 1 — CRN reproducibility gate PASSED.** 18/24 rows bit-exact vs
the Jun-7 table (diff AND stderr identical to full float precision):
zero harness nondeterminism. Every zero-table-size-predicate profile is
exactly unchanged.

**Result 2 — surgical partial re-baseline.** The 6 profiles with live
`OpponentsAtTable <= 3 / <= 4 / = 3` rules moved, all against hero:
mtt −0.0037, millenniummttv.49 −0.0025, modernmikemtt −0.0025,
itmstrikeC −0.0005, itmstrikeA −0.0003, thefixersng −0.0001. Panel mean
+0.0109 → **+0.0105**. The wired table replaces Jun-7 as the
single-hand-methodology baseline.

**killphilmtt: bit-exact at −0.0153 ± 0.0012** — the loss is NOT a
dead-predicate artifact. Its 4-6-handed OpponentsAtTable blocks are
mutual clones; its 3-handed/heads-up blocks are unreachable in the
double-up format (hands never start below 4 alive). The figure
graduates to a C3 gate metric, with this AMENDMENT: **the simulated
killphilmtt remains unfaithful to the real profile — the
`FoldOrGoAllInWhenOpponentsAtTableLessThan` header setting (its actual
bubble shift) is unimplemented in our parser, so the −0.0153 gate
metric is measured against a half-faithful clone.** The same applies to
every profile's settings header (header lines are parsed for no one).

**Known harness limitation (motivates the SNG baseline):** each hand is
an i.i.d. `sample_starting_state` draw at a fixed 6-seat game string —
shorthanded states ARE sampled (54% of queries at n_alive 4-5), but no
busts occur within a match and no bubble trajectory exists.
