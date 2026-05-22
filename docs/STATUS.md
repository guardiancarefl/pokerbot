# Project Status

**Last updated:** 2026-05-22 (Session 5 close)
**Current phase:** Phase 2d closed. Phase 3 (DCFR + archetype + subgame solver design in parallel) is next.

## Done
- Architecture designed (four-layer stack with opponent anonymity as core principle)
- Format target chosen (6-max NLHE SNG, top-3 equal payout)
- Engine selected (OpenSpiel)
- Scope, non-goals, opponent anonymity, within-match adaptation approach defined
- Runtime: Contabo VPS (Ubuntu 24.04, 12 vCPU AMD EPYC, 48GB RAM); GitHub repo at github.com/guardiancarefl/pokerbot
- Phase 1 closed: Leduc Deep CFR pipeline validated (exploitability 1187 -> 434 mbb/g across 25 iters)
- Phase 2a closed: HUNL game representation, card abstraction (EMD clustering, k=20 preflop/k=200 postflop), action abstraction
- Phase 2b closed: custom Deep CFR solver in src/nlhe/solver.py with reservoir buffers, traversal-driven training, GameStateView abstraction layer
- Phase 2c closed: Slumbot client with full b<N> action token translation (bet-translation bug fixed for per-street vs per-hand semantics)
- Phase 2d closed: GPU validation completed. Headline result: **+31.45 baseline-adjusted bb/100 vs Slumbot at iter 100** (1000 hands, seed 2026, 200bb stack). +46 bb/100 improvement over CPU's -14.8 baseline-adj result.
- GPU device support patches in solver.py and policy_adapter.py (auto-detect CUDA, networks to device, .cpu() before numpy conversion, map_location on checkpoint load, RNG state try/except for cross-version compatibility)
- requirements.txt completed (treys added — was missing, surfaced during pod setup)
- RunPod RTX PRO 4000 Blackwell pod (Secure Cloud, $0.57/hr) provisioned and validated with PyTorch 2.11.0+cu128 + sm_120 support
- Phase 3 Track A1 closed: DCFR (Linear / Discounted CFR) implemented in src/nlhe/solver.py. Per-sample iteration weighting in advantage and strategy net training, single-exponent simplification of Brown & Sandholm 2019. Vanilla path byte-identical to pre-patch behavior at fixed seed; weighting math verified standalone; real-loop smoke (vanilla / linear / discounted on 2-iter HUNL) confirms iter-1 losses match across variants (T=1 weights collapse to uniform) and iter-2 losses diverge (weighting actually applied). Checkpoint format change: buffers now persist per-entry iteration. Pre-DCFR checkpoints (Session 5 GPU artifacts) resume only under cfr_variant="vanilla"; non-vanilla resume raises with a clear message.

## Key findings from Phase 2d
1. **Buffer size matters more than network size at this scale.** 100K buffer + [64,64] CPU vs 500K buffer + [512,512] GPU: the +46 bb/100 improvement is driven primarily by the bigger buffer, not the bigger network. Strategy loss plateau dropped from ~0.95 (100K buffer) to ~0.85 (500K buffer).
2. **EMD card abstraction loses real information.** Premium pairs (AA, QQ, TT) deterministically map to the same preflop bucket regardless of bucket count (k=20 vs k=169 both collide). This is a representation limitation; OCHS would resolve it. Not a query-noise issue (confirmed by multiple bucket_runouts tests).
3. **Per-iter time at 200bb is dominated by deep-trajectory variance.** 100 traversals/iter at 200bb stack produces iter times from 1.0s to 326s depending on randomly-drawn trajectory depth. Average ~95s/iter on RTX PRO 4000 Blackwell with [512,512] and 500K buffer.
4. **Loss plateau != policy plateau.** Loss flattens around iter 80-100, but the deployed (average) strategy can keep improving past that as buffer churns.

## In progress
- Session 6 wrap-up: DCFR shipped (commit 41e2fa3). Next track to develop: A2 archetype framework, B1 subgame extractor design, or C1 archetype-belief design — pick at start of Session 7.

## Next up (Phase 3 — revised, more ambitious than original architecture)
The original plan treated subgame solving as Phase 5-6. Revised plan moves subgame solver engineering into Phase 3 as a parallel track, alongside DCFR and archetype framework. This is the path to a Pluribus-class SNG bot in 6-10 weeks.

**Track A: Algorithm + training improvements (Contabo CPU + occasional GPU bursts)**
1. ~~Implement Linear / Discounted CFR (DCFR)~~ — done in Session 6 (commit 41e2fa3). Capability shipped; actual deployment in a Phase 3 training run is a separate work item.
2. Build hand-engineered archetype framework: maniac, nit, station, LAG, TAG behavioral profiles parameterized by tightness × aggression. Use as training opponents alongside self-play.
3. Investigate OCHS card abstraction (Opponent Cluster Hand Strength) — replacement for EMD that distinguishes AA from QQ properly. Real research-flavored work, 1-2 weeks.

**Track B: Subgame solver engineering (Contabo CPU is fine for design + testing)**
1. Subgame extractor: given an OpenSpiel game state, define the depth-limited subgame to solve.
2. Fast online solver: CFR variant that converges sub-second. Different optimization target than blueprint solver.
3. Belief state estimation: opponent range at subgame root, given observation history.
4. Leaf value function: blueprint policy used to terminate subgame tree at depth limit.

**Track C: Within-match opponent modeling design (Phase 6 originally, moves up too)**
1. Continuous archetype representation (2D: tightness × aggression, not categorical).
2. Bayesian updating from observed actions within current match.
3. Population priors per stake level (compatible with anonymity — no cross-match identity).
4. Response policy as integration over belief distribution.

## Project goal (revised)
**Strongest publicly-known 6-max NLHE SNG bot with correct ICM**, evaluated to:
- Beat Slumbot in HUNL by 1-5 bb/100 (architectural sanity check)
- Beat each of the 42 bought-bot profiles by 10-30 bb/100 in SNG format
- Finish top-3 in SNG simulations against archetype opponents at 70%+ rate
- Sub-second decision latency via subgame solving
- Withstand 100k-hand test without exploitation pattern emerging
- Plausibly beat Pluribus head-to-head in 6-max cash (architectural improvements + ICM-correct value function + within-match adaptation), though this is a stretch target

Estimated total time: 6-10 weeks. Estimated total compute cost: $500-2000.

## Known issues / open questions
- 42 bought-bot profile format still unidentified (text/XML/JSON/binary) — defer until Phase 3 archetype work begins
- train_leduc.py writes config.json and metrics.json into checkpoints/ subdirectory, not run-dir root (Session 2 carryover, minor)
- _build_game_state_view in src/nlhe/solver.py has private-by-convention underscore but is imported by src/nlhe/policy_adapter.py (Session 4 carryover, cosmetic)
- Subgame solver requires careful handling of belief states — well-trodden but easy to get wrong
- Pre-DCFR checkpoints (e.g., runs/gpu_phase2d_artifacts/ from Session 5) can only be resumed under cfr_variant="vanilla". Non-vanilla resume raises by design — no silent approximation of missing per-entry iter data.

## Decisions deferred
- OCHS vs simpler abstraction extensions (decide during Phase 3 after literature review)
- Exact mix percentages for training opponent pool (self-play vs archetypes vs bought-bots vs league archives) — tune empirically during Phase 4-5
- Whether to use multiple GPU pods in parallel during Phase 4 — depends on Phase 3 throughput findings

## Session log
- **2026-05-21 (Session 1):** Project bootstrapped on Windows. Foundational docs created. Major design decisions: opponent anonymity, Position 2 within-match adaptation, league play for strength diversity, 42 profiles frozen.
- **2026-05-21 (Session 2):** WSL2 install blocked by Windows component store corruption. Switched runtime to Contabo VPS. Built Phase 1 scaffold. Completed Phase 1 run: exploitability 1187 -> 434 mbb/g across 25 iterations.
- **2026-05-22 morning (Session 3):** Phase 2a HUNL game representation, EMD card abstraction, action abstraction. Phase 2b custom Deep CFR solver. Phase 2c Slumbot client foundations.
- **2026-05-22 early (Session 4):** Phase 2c closure (bet-translation bug fix), PolicyAdapter built with 28 unit tests, CPU overnight training (275 iters at 200bb), Slumbot evaluation showing -14.8 baseline-adj bb/100 (45 bb/100 better than random's -60).
- **2026-05-22 afternoon/evening (Session 5):** GPU device support patches landed and tested on CPU. Multiple failed RunPod deployments (CUDA driver/container incompatibility on Community Cloud pool). Eventually working pod on Secure Cloud RTX PRO 4000 Blackwell. GPU smoke + 50-iter benchmark + 119+ iter v2 run with 500K buffer. ckpt_iter_0100 evaluated against Slumbot: +31.45 baseline-adj bb/100, +46 over CPU. Architecture plan revised to be more ambitious based on Pluribus compute math (we have more compute than they did, plus AI-assisted engineering). Phase 3 redefined to start subgame solver + DCFR + archetype work in parallel. Target: best publicly-known SNG bot in 6-10 weeks.
- **2026-05-22 (Session 6):** Phase 3 Track A1 (DCFR) implemented and shipped. Three sequential patches in src/nlhe/solver.py: TrainConfig fields, ReservoirBuffer iteration tagging + checkpoint format, weighted training loss with simplified single-exponent form. Vanilla regression gate held at every step (32/32 tests green). Standalone math verification matched hand-calculated weights exactly. Real-loop smoke confirmed wiring (iter-1 variants converge to same loss, iter-2 variants diverge as expected, discounted(exponent=1.0) ≡ linear exactly). Commit 41e2fa3. DECISIONS.md updated with two new entries: simplified single-exponent rationale and refuse-non-vanilla-resume backward-compat rationale.
