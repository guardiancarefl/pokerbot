# Project Status

**Last updated:** 2026-05-22 (Session 3 extended extended, very late)
**Current phase:** Phase 2c closed. Phase 2d (RunPod 4090 training + Slumbot eval) is next.

## Done
- Architecture, format target, engine, opponent anonymity, within-match adaptation, league play, 42 frozen profiles — all Session 1
- Runtime: Contabo VPS, GitHub repo, Python 3.10 venv with PyTorch CPU + OpenSpiel
- **Phase 1 closed:** Leduc Deep CFR pipeline validated (1187 → 434 mbb/g across 25 iters)
- GPU provider for Phase 2d: RunPod Community Cloud RTX 4090 at $0.34/hr, account created
- **Phase 2a closed:** HUNL game validated in OpenSpiel; `src/nlhe/equity.py` (191 lines), `src/nlhe/abstraction.py` (326 lines, EMD-based card abstraction), `src/nlhe/actions.py` (~230 lines, discrete action space + pseudo-harmonic translation). Trained abstraction at `runs/abstraction_20260521_223018/`: preflop k=20, postflop k=200, 6.8 min training. Inspection confirmed strategic coherence.
- **Phase 2b closed:** `src/nlhe/infoset.py` (229 lines, 214-dim feature vector), `src/nlhe/solver.py` (452 lines, custom External Sampling Deep CFR), regret normalization, resumable checkpointing **bit-identical** verified, YAML-driven `scripts/train_nlhe.py`.
- **Phase 2c closed:** `src/nlhe/slumbot_client.py` (~210 lines, HTTP client + action-language parser + RandomPolicy baseline), `scripts/eval_vs_slumbot.py` (~110 lines, raw and baseline-adjusted bb/100 reporter). 50-hand validation run: 0 errors, raw=-13 bb/100, baseline-adjusted=-6 bb/100. Slumbot's `baseline_winnings` field used for variance reduction (the headline eval metric).

## In progress
- Session 3 extended extended docs wrap-up.

## Next up (Phase 2d)
1. Write a `PolicyAdapter` that wraps a trained `DeepCFRSolver` into a `choose_action(SlumbotState) -> str` callable. Translates Slumbot's 200bb chip-action-string state into our `InfosetEncoder` view, runs the strategy network, picks an action via the discrete action space, translates back to Slumbot's action string format.
2. Optional Phase 2b real-run on Contabo CPU before going to GPU: execute `configs/nlhe_phase2b.yaml` (100 iter × 100 trav, est 3-5 hours) to get a real trained policy. Could overlap with `PolicyAdapter` work.
3. Phase 2d: rent RunPod 4090, scale config to [256, 256] networks + larger traversal counts, train to convergence-ish. Eval against Slumbot, get measurable bb/100. This is the headline Phase 2 deliverable.

## Known issues / open questions
- Buffer asymmetry (player 0 ~4x more entries than player 1 at 20bb). Probably structural to short-stack HUNL; monitor across longer runs.
- 42 bought-bot profile format still unknown — defer until Phase 3.
- Contabo per-iter time variance (3-160s) due to oversubscribed vCPUs. Adequate for dev; meaningless for compute planning.
- `train_leduc.py` writes config.json/metrics.json into checkpoints/ subdir; `train_nlhe.py` writes them at run-dir root; `runs/README.md` claims root. Three-way inconsistency. Reconcile during Phase 2d setup.
- $93 Vultr credit: expiry date unchecked.
- Pairwise EMD abstraction training single-threaded; embarrassingly parallelizable.
- Action abstraction: opponent bets between 2pot and 0.9×stack snap to 2pot. Coarse but acceptable for now.
- Slumbot eval: 50 hands far too few for statistically meaningful bb/100; need 500-5000+ for stable estimates of small edges.

## Decisions deferred
- Card abstraction granularity refinement (decide Phase 2d if Slumbot says abstraction is the bottleneck)
- League play schedule (decide post Phase 4)
- VPS as parallel self-play worker (decide if throughput bound)
- Training opponent pool mix percentages (Phase 4-5)

## Session log
- **2026-05-21 (Session 1):** Project bootstrapped, foundational docs created.
- **2026-05-21 (Session 2):** WSL2 blocked, migrated to Contabo. Phase 1 Leduc Deep CFR validated.
- **2026-05-21 (Session 3):** Phase 2a — HUNL validated in OpenSpiel, equity + EMD abstraction + action abstraction shipped.
- **2026-05-22 (Session 3 extended, early hours):** Phase 2b — custom Deep CFR solver, resumable checkpointing with bit-identical correctness, YAML training script. Closed.
- **2026-05-22 (Session 3 extended extended, ~00:50):** Phase 2c — Slumbot API client built, action-language parser, baseline-winnings variance reduction wired, 50-hand random-policy validation run clean. Closed.
