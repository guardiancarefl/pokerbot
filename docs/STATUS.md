# Project Status

**Last updated:** 2026-05-22 (Session 4 close)
**Current phase:** Phase 2d step 1 closed. 200bb overnight CPU training running unattended; Session 5 will evaluate.

## Done
- Architecture, format target, engine, opponent anonymity, within-match adaptation, league play, 42 frozen profiles — all Session 1
- Runtime: Contabo VPS, GitHub repo, Python 3.10 venv with PyTorch CPU + OpenSpiel
- **Phase 1 closed:** Leduc Deep CFR pipeline validated (1187 → 434 mbb/g across 25 iters)
- GPU provider for Phase 2d: RunPod Community Cloud RTX 4090 at $0.34/hr, account created
- **Phase 2a closed:** HUNL game validated in OpenSpiel; `src/nlhe/equity.py` (191 lines), `src/nlhe/abstraction.py` (326 lines, EMD-based card abstraction), `src/nlhe/actions.py` (~230 lines, discrete action space + pseudo-harmonic translation). Trained abstraction at `runs/abstraction_20260521_223018/`: preflop k=20, postflop k=200, 6.8 min training. Inspection confirmed strategic coherence.
- **Phase 2b closed:** `src/nlhe/infoset.py` (229 lines, 214-dim feature vector), `src/nlhe/solver.py` (452 lines, custom External Sampling Deep CFR), regret normalization, resumable checkpointing **bit-identical** verified, YAML-driven `scripts/train_nlhe.py`.
- **Phase 2c closed:** `src/nlhe/slumbot_client.py` (~210 lines, HTTP client + action-language parser + RandomPolicy baseline), `scripts/eval_vs_slumbot.py` (~110 lines, raw and baseline-adjusted bb/100 reporter). 50-hand validation run: 0 errors, raw=-13 bb/100, baseline-adjusted=-6 bb/100. Slumbot's `baseline_winnings` field used for variance reduction (the headline eval metric).
- **Phase 2d step 1 closed:** `src/nlhe/policy_adapter.py` (454 lines, `PolicyAdapter` glue between trained `DeepCFRSolver` and the Slumbot client) + `tests/test_policy_adapter.py` (28 unit tests) + `scripts/eval_vs_slumbot.py` refactor for `--policy {random,adapter}`. Validated end-to-end against Slumbot: 20/20 plumbing-test hands clean. Wire-format bet-translation bug (Slumbot `b<N>` per-street vs OpenSpiel int per-hand) caught by the plumbing test and fixed. Latest commit `c07fd9c`.
- **Session 4 cleanup:** `CLAUDE.md` written via `/init`, `scripts/verify_abstraction_stack_invariance.py` committed (confirms `Abstraction.bucket_of()` takes no stack args), `configs/nlhe_smoke_iter1.yaml` deleted. Cleanup commit `9e7be8b`.
- **200bb training pipeline validated at iter-1 timing benchmark** on Contabo CPU: 48.9s/iter at `[64,64]` / 50 trav / 100 steps. Scaled-up `[128,128]` / 100 trav / 200 steps variant hung at 100% CPU single-threaded; killed at 4+ min. Reverted to bench config for the overnight run.

## In progress
- **Overnight 200bb training run** launched in separate SSH session (PID 730653, log `/tmp/200bb_overnight.log`, run dir `runs/nlhe_20260522_043341_phase2d_200bb_overnight/`, 300 iters at `[64,64]`/50/100, ETA ~07:15 Contabo). Independent of this Claude Code session.

## Next up (Phase 2d)
1. **Review overnight training artifacts** at start of Session 5: final iter losses, full loss trajectory across 300 iters, all 12 checkpoints saved, buffer behavior.
2. **Plumbing-test the latest checkpoint against Slumbot at matched 200bb stack depth** (e.g., `ckpt_iter_0300.pt`). This is the actual Phase 2d headline test — the earlier 20-hand plumbing run used the 20bb checkpoint and was stack-mismatched by design.
3. **Decide on RunPod rental for `[256,256]` scale-up training.** Decision is conditional on overnight result: if 300-iter CPU run produces measurable positive learning signal in Slumbot eval, GPU rental becomes a "make it stronger" task rather than a "make it work at all" task. If overnight produces garbage or hangs, GPU rental is the next debugging move with concrete data.

## Known issues / open questions
- 200bb training with `hidden_dim: [128, 128]` + `traversals_per_iter: 100` + `train_steps_per_iter: 200` hung at 100% CPU single-threaded during iter-1 timing test. Mechanism unknown — torch thread starvation, GIL contention on deeper trees, or interaction effect. Reverted to bench-proven `[64,64]`/50/100 for overnight. Worth diagnosing before any GPU run where we'll definitely want bigger networks.
- Preflop abstraction k=20 groups some adjacent premium hands (e.g., AA and KK in same bucket). Cheap to revisit if Slumbot eval reveals preflop weakness — retraining preflop to k=169 (lossless) costs ~1 min and only changes preflop bucket count, not postflop.
- Buffer asymmetry from Session 3 (4:1 at 20bb) **did not reproduce** — Session 4 smoke run measured 1.13:1 at iter 20, well within sampling noise. The 4:1 was probably a small-sample artifact from the earlier short Phase 2b dry-run, not a structural property of short-stack HUNL. **Closed.**
- Three-way `--run-name` CLI inconsistency: `train_leduc.py` accepts `--run-name`, `train_nlhe.py` only uses YAML `tag:`. Not a bug — different scripts, different conventions — but worth either unifying or documenting at end of Phase 2.
- `train_leduc.py` config.json/metrics.json location inconsistency (Session 2 carryover) is still open. `train_nlhe.py` writes them at run-dir root (correct per `runs/README.md`). Only `train_leduc.py` is wrong (writes into `checkpoints/`).
- `_build_game_state_view` in `src/nlhe/solver.py` is module-private by underscore convention but now imported by `src/nlhe/policy_adapter.py`. Either rename to drop the underscore (it's now public-by-usage) or wrap in a thin public function. Code hygiene, not blocking anything.
- 42 bought-bot profile format still unknown — defer until Phase 3.
- Contabo per-iter time variance (3-160s on Phase 2b smoke, 30-49s on 200bb) due to oversubscribed vCPUs. Adequate for dev; meaningless for compute planning.
- $93 Vultr credit: expiry date unchecked.
- Pairwise EMD abstraction training single-threaded; embarrassingly parallelizable.
- Action abstraction: opponent bets between 2pot and 0.9×stack snap to 2pot. Coarse but acceptable for now.
- Slumbot eval: 20-50 hands far too few for statistically meaningful bb/100; need 500-5000+ for stable estimates of small edges.

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
- **2026-05-22 (Session 4, ~02:00-04:35):** Phase 2d step 1 — PolicyAdapter shipped; bet-translation bug caught by plumbing test and fixed; 20/20 plumbing hands clean; 200bb iter-1 timing benchmark; overnight 300-iter run launched in separate SSH and left running.
