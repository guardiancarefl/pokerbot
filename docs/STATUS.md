# Project Status

**Last updated:** 2026-05-22 (Session 3 extended, very late)
**Current phase:** Phase 2b closed. Phase 2c (Slumbot harness) is next, then Phase 2d on RunPod.

## Done
- Architecture designed (four-layer stack with opponent anonymity as core principle)
- Format target chosen (6-max NLHE SNG, top-3 equal payout)
- Engine selected (OpenSpiel)
- Opponent anonymity principle, Position 2 within-match adaptation, league play, 42 frozen profiles — all from Session 1
- Runtime: Contabo VPS (Ubuntu 24.04, 12 vCPU, 48GB RAM)
- GitHub repo, SSH key, Python 3.10 venv, PyTorch 2.12 CPU + OpenSpiel 1.6.11
- **Phase 1 closed.** Leduc Deep CFR pipeline validated. Exploitability 1187 → 434 mbb/g across 25 iters.
- GPU provider for Phase 2d chosen: RunPod Community Cloud RTX 4090 at $0.34/hr.
- HUNL game representation validated in OpenSpiel via universal_poker.
- Equity calculator (`src/nlhe/equity.py`, 191 lines) over `treys`. Validated against literature.
- **EMD card abstraction** (`src/nlhe/abstraction.py`, 326 lines). Trained artifact at `runs/abstraction_20260521_223018/`: preflop k=20, postflop k=200, 6.8 min training. Inspection confirmed strategically coherent clustering.
- Action abstraction (`src/nlhe/actions.py`, ~230 lines). Discrete action space + pseudo-harmonic opponent-bet translation.
- **Phase 2a closed.**
- **Custom Deep CFR solver** (`src/nlhe/solver.py`, 452 lines). External sampling, regret-matching+, reservoir buffers, dual networks per player. Regrets normalized by starting_stack for sane MSE conditioning.
- **Resumable checkpointing.** Save/load full solver state (networks, optimizers, buffers, all RNGs). **Bit-identical resume correctness verified** on 10-iter test (max param diff 0.00e+00 across all 4 networks).
- Custom info-state encoder (`src/nlhe/infoset.py`, 229 lines). Bucket one-hot + street + position + pot/stack/to_call + betting features = 214-dim vector. Per-traversal bucket cache.
- Training entry point (`scripts/train_nlhe.py`, ~95 lines). YAML-driven config, mirrors Leduc pattern. Configs in `configs/nlhe_smoke.yaml` and `configs/nlhe_phase2b.yaml`.
- **Phase 2b closed.** Solver runs end-to-end, losses sensibly scaled, resumable training works exactly. Phase 2d ready (gated only by Slumbot harness for evaluation).

## In progress
- Session 3 extended docs wrap-up.

## Next up (Phase 2c, 2d)
1. Phase 2c: build Slumbot API harness (`src/nlhe/slumbot_client.py`). HTTP client, hand-history parsing, bb/100 calculator. Test with a random-policy baseline before plugging in our bot.
2. Phase 2c: write evaluation script that runs N hands of our trained policy vs Slumbot and reports bb/100.
3. Phase 2b run-it-for-real (optional, on Contabo CPU): execute `configs/nlhe_phase2b.yaml` (100 iters × 100 traversals) to get a real trained baseline policy before Phase 2d. Estimated ~3-5 hours wall-clock on Contabo. Could overlap with Phase 2c work.
4. Phase 2d: rent RunPod 4090, scale up to [256,256] networks + larger sample counts. Get bb/100 vs Slumbot.

## Known issues / open questions
- Buffer asymmetry between players observed in early smoke runs (player 0 buffer fills ~4x faster than player 1's at 20bb stacks). Likely structural — SB folds preflop often → fewer SB decision nodes. Monitor in longer runs; if it persists with worse-than-expected Slumbot results, investigate.
- 42 bought-bot profile format still unidentified — defer until Phase 3.
- Contabo per-iteration time varies wildly (~3-160s observed) due to oversubscribed vCPUs. Adequate for development; meaningless as benchmark for Phase 4+ planning.
- `train_leduc.py` writes config.json and metrics.json into `checkpoints/` not at run-dir root as `runs/README.md` claims. Deferred from Session 2. `train_nlhe.py` does it correctly (root-level), so the inconsistency widens.
- $93 Vultr credit: expiry date unchecked. If it expires before Phase 4, revisit for Phase 3 helper instance.
- Pairwise EMD in abstraction training is single-threaded. ~7 min run could be ~1-2 min with multiprocessing. Not worth optimizing until we retrain.
- Action abstraction: opponent bets between 2pot and 0.9×stack all snap to 2pot. Coarse but acceptable; revisit if Slumbot shows overbet exploits.

## Decisions deferred
- Card abstraction refinement (decided 200/20 buckets for Phase 2a; refine in Phase 2d if Slumbot says abstraction is the bottleneck).
- League play schedule (decide after Phase 4 blueprint).
- VPS as parallel self-play worker (decide if throughput bottleneck).
- Training opponent pool mix percentages (tune empirically Phase 4-5).

## Session log
- **2026-05-21 (Session 1):** Project bootstrapped on Windows. Foundational docs created.
- **2026-05-21 (Session 2):** WSL2 blocked by Windows corruption, migrated to Contabo. Phase 1 Leduc Deep CFR pipeline validated (1187 → 434 mbb/g).
- **2026-05-21 (Session 3):** Phase 2a built: GPU provider chosen (RunPod 4090), HUNL game validated in OpenSpiel, equity calculator + EMD card abstraction + action abstraction shipped and validated.
- **2026-05-22 (Session 3 extended, into the early hours):** Phase 2b built: custom info-state encoder, external-sampling Deep CFR solver with regret normalization, resumable checkpointing with bit-identical correctness verified, YAML-driven training script. Phase 2b closed; Phase 2c (Slumbot harness) is next.
