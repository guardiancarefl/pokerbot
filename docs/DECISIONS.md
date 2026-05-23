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

  1. **No strategy net / no average-strategy approximation.** `PlayerNetworks6Max` (Phase 4e.3a) only carries advantage nets. At deployment time, the deployed policy is the regret-matched current strategy from the most recent advantage net — not the average strategy across iterations. Average-strategy approximation is its own subphase (likely Phase 4e.4) and only worth doing once the baseline trains.

  2. **No DCFR weighting yet (vanilla CFR only).** DCFR (Brown & Sandholm 2019) was added to HUNL in Phase 3 Track A1 (commit 41e2fa3). It provides faster convergence by per-sample iteration weighting. For 6-max first-cut, vanilla CFR (uniform weights) is the baseline. Mechanical to add later once the vanilla loss trajectory is established.

  3. **No archetype mix yet.** The archetype framework (`src/nlhe/archetypes.py`) is HUNL-specific — it uses `derive_in_position` (HU position-from-current-player) and decision tables sized for two-player betting trees. Porting to 6-max requires position derivation for all 6 seats and decision tables sized for the multiway 6-max pot. Its own subphase, not a 4e.3c concern.

  4. **Uniform starting stacks per traversal.** Real SNG hands have stacks that evolve across hands (chip leader, short stack, equal stacks, bubble situations). The 4e.3c training loop uses `[cfg.starting_stack] * 6` for every traversal — every hand looks like Hand 1, with all stacks equal. The ICM transformation still operates correctly per terminal (deltas in equity are computed against equal-stack baselines), but the bot doesn't see bubble-pressure asymmetry from a 4-handed game with widely differing stacks. Stack-distribution sampling is its own subphase. Until it's added, the bot trains on a degenerate slice of the SNG state space.

First-cut scope is "the smallest 6-max thing that trains without diverging, in time to evaluate on benchmark before adding complexity." This staging is what landed +78.35 bb/100 on HUNL — vanilla Deep CFR first, then DCFR, then archetypes, then determinism retrofits. Same logic applies to 6-max: ship a baseline that trains, measure it, then add enhancements with each one's contribution measurable.

**Alternative considered:** Ship 6-max with the full Phase 3 enhancement stack (strategy net + DCFR + archetypes + stack sampling) from the start, since the HUNL precedent exists and the code patterns are known.
**Reason rejected:** Five interacting moving pieces in one ship is uninvestigable when something breaks. The HUNL track added enhancements incrementally across 5+ sessions, with each step's contribution measurable separately. The same incremental approach is the right one for 6-max — even if the individual additions are "easier" the second time around, the joint debugging cost is the same.
