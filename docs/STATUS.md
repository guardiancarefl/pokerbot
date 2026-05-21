# Project Status

**Last updated:** 2026-05-21 (Session 3, in progress)
**Current phase:** Phase 1 closed. Phase 2 (heads-up NLHE prototype) is next.

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

## In progress
- Session 3 docs updates (this STATUS update + DECISIONS entry for GPU choice)

## Next up (Phase 2)
1. Phase 2a: load HUNL in OpenSpiel via universal_poker, validate game representation
2. Phase 2a: build card abstraction module (src/nlhe/abstraction.py) using EMD clustering
3. Phase 2a: build action abstraction module (src/nlhe/actions.py) with discretized bet sizes
4. Phase 2b: train Deep CFR on tiny HUNL (small stacks, coarse abstraction) on Contabo CPU. Validate the pipeline scales from Leduc to real poker.
5. Phase 2c: build Slumbot API harness (src/nlhe/slumbot_client.py), test with a random policy baseline
6. Phase 2d: rent RunPod 4090, run real HUNL training with [256, 256] networks. Get bb/100 vs Slumbot.

## Known issues / open questions
- 42 bought-bot profile format still unidentified (text/XML/JSON/binary) — defer until Phase 3
- Contabo per-iteration time varies wildly (13s to 151s observed) due to oversubscribed vCPUs. Adequate for Phase 1-3 algorithm validation; not a benchmark for Phase 4+ compute planning.
- The original Windows host (RTX 3060 Laptop, 6GB VRAM) still exists. Could be brought back if Windows component store gets repaired. Not a priority.
- Workflow note from Session 1 still applies: when exact file contents are provided between BEGIN/END markers, write them verbatim; concerns are questions before writing, not silent edits.
- train_leduc.py writes config.json and metrics.json into the checkpoints/ subdirectory, not at the run-dir root as runs/README.md claims. Minor — Session 3 cleanup.
- $93 Vultr credit: check whether it has an expiry date. If it expires before Phase 4 (~6-10 weeks out), revisit and consider Phase 3 helper use instead of letting it lapse.

## Decisions deferred
- Specific card abstraction granularity for Phase 2 (decide during Phase 2a based on equity-histogram experiments)
- Exact league play schedule (decide after Phase 4 blueprint exists)
- Whether to use VPS as parallel self-play worker (decide if single-instance throughput becomes a bottleneck)
- Exact mix percentages for training opponent pool composition (self-play vs. archetypes vs. bought-bots vs. league archives) — tune empirically during Phase 4-5

## Session log
- **2026-05-21 (Session 1):** Project bootstrapped on Windows. Foundational docs created. Major design decisions: opponent anonymity, Position 2 within-match adaptation, league play for strength diversity, 42 profiles frozen.
- **2026-05-21 (Session 2):** WSL2 install blocked by Windows component store corruption. Switched runtime to existing Contabo VPS. Set up GitHub repo + SSH. Built Phase 1 scaffold as modular code (config dataclass, solver wrapper, eval, checkpoint, tests). Initial training run timed out at 30 min with no visibility into progress; rewrote solver to drive iterations manually with per-iteration logging and periodic exploitability eval. Completed Phase 1 run on a deliberately undersized config in ~38 min: exploitability 1187 -> 434 mbb/g across 25 iterations. Pipeline validated. Phase 2 (HUNL prototype) is next.
- **2026-05-21 (Session 3, in progress):** Opened Phase 2. GPU provider decision locked in: RunPod Community Cloud RTX 4090 at $0.34/hr; Vultr $93 credit preserved for Phase 4. RunPod account created (no pod rented). Next: Phase 2a code work.
