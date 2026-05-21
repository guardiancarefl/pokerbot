# Project Status

**Last updated:** 2026-05-21 (Session 3 close)
**Current phase:** Phase 2a closed. Phase 2b (tiny-HUNL solver smoke) is next.

## Done
- Architecture designed (four-layer stack with opponent anonymity as core principle)
- Format target chosen (6-max NLHE SNG, top-3 equal payout)
- Engine selected (OpenSpiel)
- Scope and non-goals defined
- Opponent anonymity principle established (no persistent identity across matches, no real-world hand history data, no pre-collected opponent profiles)
- Within-match adaptation approach decided (Position 2: light-to-medium online reads anchored to blueprint)
- League play role clarified (provides strength diversity; the 42 bought-bot profiles stay frozen to preserve benchmark stability)
- Foundational docs created (PROJECT_OVERVIEW, ARCHITECTURE, DECISIONS, STATUS, SESSION_LOG)
- Runtime environment migrated from WSL2-on-Windows to Contabo VPS (Ubuntu 24.04, 12 vCPU AMD EPYC, 48GB RAM) after Windows component store corruption blocked WSL2 install
- GitHub repo at github.com/guardiancarefl/pokerbot (private), SSH key registered
- Python 3.10 installed alongside system 3.12 via deadsnakes PPA
- venv created at ~/pokerbot/.venv with PyTorch 2.12 CPU + OpenSpiel 1.6.11
- Project structure established (src/leduc/, scripts/, tests/, configs/, runs/)
- Phase 1 pipeline scaffold built with modular structure (config dataclass + YAML, solver wrapper, checkpoint, eval, tests)
- Per-iteration logging and eval callback added to solver (after Phase 1 default config ran 30+ min with no visibility)
- **Phase 1 closed: Leduc Deep CFR pipeline validated.** Exploitability trajectory 1187 (uniform) -> 502 (iter 10) -> 447 (iter 20) -> 434 (iter 25). Configured for time-on-Contabo, not for record exploitability. Pipeline confirmed to train networks, evaluate exploitability, save checkpoints, and produce sensible downward-trending results on a known-Nash game.
- GPU provider for Phase 2d chosen: RunPod Community Cloud RTX 4090 at $0.34/hr. RunPod account created Session 3; no pod rented yet. $93 Vultr credit held for Phase 4 (or Phase 3 helper if needed).
- HUNL game representation validated in OpenSpiel via universal_poker (Session 3). Configuration: 2 players, 4 rounds, blinds 100/50, stacks 20000, fullgame betting abstraction. Confirmed: 760-dim information state tensor, max_game_length 218, ~20000 actions per node before abstraction, dynamic shrinking of action space with effective stack.
- Equity calculator built (src/nlhe/equity.py, 191 lines) over the `treys` library. 169-class canonical hole enumeration, Monte Carlo equity-vs-random and equity-vs-range, full card-removal handling. Validated against literature (AA preflop 0.851, AA vs JJ 0.809, 72o preflop 0.348).
- **EMD card abstraction built (src/nlhe/abstraction.py, 326 lines).** Equity-histogram-based, with `compute_hand_histogram`, pairwise EMD via scipy.stats.wasserstein_distance, custom k-medoids (PAM) clustering, and Abstraction container with bucket-of/save/load. Validated end-to-end via smoke test (preflop k=5 in ~10s). Decision recorded in DECISIONS.md (EMD over OCHS, values-driven).
- Trained abstraction artifact produced (runs/abstraction_20260521_223018/): preflop k=20, flop/turn/river k=200 each. Total training time 6.8 minutes. Inspection confirms EMD clustering is strategically coherent — different surface hands with similar equity-histogram shapes correctly group together (e.g., "drawing dead on coordinated board" hands across different specific cards land in one bucket).
- Action abstraction built (src/nlhe/actions.py, ~230 lines). DiscreteAction enum {fold, call, 0.33pot, 0.66pot, 1pot, 2pot, allin}; `policy_to_game_action` (discrete → OpenSpiel integer); `game_to_policy_action` (OpenSpiel integer → distribution over discrete via pseudo-harmonic translation, Ganzfried & Sandholm 2013); `discretize_legal_actions` for filtering at decision time. Handles sub-min-bet illegality cleanly (returns None for unavailable bet sizes rather than aliasing to min_bet).
- **Phase 2a closed: HUNL game loaded, equity calculator + card abstraction + action abstraction all built and validated.** Foundations for Phase 2b training pipeline are in place.

## In progress
- Session 3 docs wrap-up (this STATUS update + SESSION_LOG entry)

## Next up (Phase 2b–2d)
1. Phase 2b: build NLHE solver wrapper following the Leduc solver pattern. Drives OpenSpiel DeepCFRSolver with per-iteration logging, accepts abstraction artifact + action module as inputs. Information state encoding must combine card bucket (from abstraction) with betting history features.
2. Phase 2b: train Deep CFR on tiny HUNL (small stacks ~20bb, coarse abstraction, [64,64] networks) on Contabo CPU. Validate the pipeline scales from Leduc to real poker. Target: clean training run, sensible loss curves, no NaN/inf in advantage or strategy networks.
3. Phase 2b: implement resumable training (checkpoint every N iterations, idempotent resume). Prerequisite for using RunPod Community Cloud in Phase 2d without interrupt-cost panic.
4. Phase 2c: build Slumbot API harness (src/nlhe/slumbot_client.py). Test with a random-policy baseline before plugging in our bot. Includes hand-history parsing and bb/100 calculator.
5. Phase 2d: rent RunPod 4090, run full coarse-abstraction HUNL training with [256,256] networks. Iterate to a measurable bb/100 vs Slumbot.

## Known issues / open questions
- 42 bought-bot profile format still unidentified (text/XML/JSON/binary) — defer until Phase 3
- Contabo per-iteration time varies wildly (13s to 151s observed) due to oversubscribed vCPUs. Adequate for Phase 1-3 algorithm validation; not a benchmark for Phase 4+ compute planning.
- The original Windows host (RTX 3060 Laptop, 6GB VRAM) still exists. Could be brought back if Windows component store gets repaired. Not a priority.
- Workflow note from Session 1 still applies: when exact file contents are provided between BEGIN/END markers, write them verbatim; concerns are questions before writing, not silent edits.
- train_leduc.py writes config.json and metrics.json into the checkpoints/ subdirectory, not at the run-dir root as runs/README.md claims. Deferred from Session 2; not blocking.
- $93 Vultr credit: check whether it has an expiry date. If it expires before Phase 4 (~6-10 weeks out), revisit and consider Phase 3 helper use instead of letting it lapse.
- runs/ directory contains five Leduc attempts from Session 2 (not just the winning one). Keeping all of them is intentional (failed attempts have diagnostic value via their config.json), but worth noting that SESSION_LOG/STATUS only narrate the final run.
- Pairwise EMD in abstraction training is single-threaded on a 12-vCPU box. ~6.8 min training run could be ~1-2 min with multiprocessing.Pool. Not worth optimizing until we retrain abstraction in Phase 2d/2e.
- Action abstraction uses 5 bet sizes {0.33pot, 0.66pot, 1pot, 2pot, allin}. Opponent bets between 2pot and allin all snap to 2pot. Acceptable for Phase 2; revisit if Slumbot shows our policy is exploitable on overbet sizing.

## Decisions deferred
- Specific card abstraction granularity refinement (decided 200 buckets postflop / 20 preflop for Phase 2a; refine in Phase 2d if Slumbot says abstraction is the bottleneck)
- Exact league play schedule (decide after Phase 4 blueprint exists)
- Whether to use VPS as parallel self-play worker (decide if single-instance throughput becomes a bottleneck)
- Exact mix percentages for training opponent pool composition (self-play vs. archetypes vs. bought-bots vs. league archives) — tune empirically during Phase 4-5

## Session log
- **2026-05-21 (Session 1):** Project bootstrapped on Windows. Foundational docs created. Major design decisions: opponent anonymity, Position 2 within-match adaptation, league play for strength diversity, 42 profiles frozen.
- **2026-05-21 (Session 2):** WSL2 install blocked by Windows component store corruption. Switched runtime to existing Contabo VPS. Set up GitHub repo + SSH. Built Phase 1 scaffold as modular code (config dataclass, solver wrapper, eval, checkpoint, tests). Initial training run timed out at 30 min with no visibility into progress; rewrote solver to drive iterations manually with per-iteration logging and periodic exploitability eval. Completed Phase 1 run on a deliberately undersized config in ~38 min: exploitability 1187 -> 434 mbb/g across 25 iterations. Pipeline validated. Phase 2 (HUNL prototype) is next.
- **2026-05-21 (Session 3):** Opened Phase 2. GPU provider decision locked in (RunPod Community Cloud 4090, $93 Vultr credit held for Phase 4). HUNL game representation validated in OpenSpiel (760-dim info state, ~20000-action node before abstraction). Built equity calculator over `treys`. Decided EMD over OCHS for card abstraction (values-driven, Reading 2). Built and trained EMD abstraction artifact (preflop k=20, postflop k=200, 6.8 min). Inspection confirmed strategically coherent clustering. Built action abstraction module with discrete bet sizes and pseudo-harmonic opponent-bet translation. Phase 2a closed. Phase 2b is next.
