# Prompt for Session 4 (paste this verbatim at the start of the next conversation)

---

Continuing the pokerbot project. Session 4.

**Status:** Phase 2a closed in Session 3. HUNL game representation validated in OpenSpiel; equity calculator (`src/nlhe/equity.py`), EMD card abstraction (`src/nlhe/abstraction.py`), and action abstraction (`src/nlhe/actions.py`) all built, validated, and committed. Trained abstraction artifact lives at `runs/abstraction_20260521_223018/abstraction.pkl` (preflop k=20, flop/turn/river k=200, 6.8 min training time, inspection confirmed strategically coherent clustering — different surface hands with similar equity-histogram shapes correctly group together).

**Runtime:** Contabo VPS, Ubuntu 24.04, 12 vCPU AMD EPYC, 48GB RAM, no GPU. Project at `~/pokerbot/` on the Contabo box. Repo at github.com/guardiancarefl/pokerbot (private). Latest commit `5e8aef9` ("Phase 2a: HUNL abstraction"). Working in Python 3.10 venv with PyTorch 2.12 CPU + OpenSpiel 1.6.11 + scipy + treys.

**GPU decision locked:** Phase 2d will use RunPod Community Cloud RTX 4090 at $0.34/hr. RunPod account created in Session 3; no pod rented yet. $93 Vultr credit held for Phase 4 (or Phase 3 helper if needed).

**Today's session opens Phase 2b: tiny-HUNL solver smoke test.** The next-up items from STATUS.md:

1. Phase 2b: build NLHE solver wrapper following the Leduc solver pattern (per-iteration logging, eval callback, NaN-safe loss capture). Information state encoding must combine card bucket (from `Abstraction.bucket_of`) with betting history features. The solver wraps OpenSpiel's `DeepCFRSolver` thinly, same approach as Phase 1.
2. Phase 2b: train Deep CFR on tiny HUNL (20bb stacks, coarse abstraction, [64,64] networks) on Contabo CPU. Goal is "the pipeline doesn't explode," not "good HUNL strategy." Sensible loss curves, no NaN/inf, sensible bucket assignments at decision nodes.
3. Phase 2b: implement resumable training (checkpoint every N iterations, idempotent resume). Hard prerequisite for using RunPod Community Cloud in Phase 2d without interrupt-cost panic.
4. Optional Phase 2c: Slumbot API harness (`src/nlhe/slumbot_client.py`) with random-policy baseline. Independent of 2b — could be done in parallel.

**Two small Session 2 cleanups still open:**
- `runs/README.md` claims `train_leduc.py` writes config.json and metrics.json at run-dir root; the script actually writes them into `checkpoints/`. Reconcile.
- Sanity-check the climbing advantage loss across the Phase 1 run against a published Deep CFR Leduc reference, to confirm it's the expected reinitialize-from-scratch pattern.

Please read the project knowledge docs in this order: STATUS.md (entry point, has the full state), SESSION_LOG.md Session 3 entry (for context on Phase 2a build), PHASE2_SKETCH.md (forward plan, may need a refresh now that 2a is done), DECISIONS.md (latest two entries: GPU provider, EMD abstraction). Confirm state matches what STATUS says, then we'll start with Phase 2b solver wrapper.

**Workflow rules carrying over from previous sessions:**
- When exact file contents are given between BEGIN/END markers, those markers are *labels* describing where content goes, not shell commands to paste. Always wrap content in a heredoc (`cat > path << 'DISTINCTIVE_EOF'` ... `DISTINCTIVE_EOF`), single-quoted delimiter to prevent variable expansion.
- Verify file writes with `wc -l` / `head` / `tail` / `grep`. Terminal echo sometimes visually mangles heredoc pastes; the file on disk is usually fine.
- Two-SSH-session workflow on the Contabo box: one for long-running training, one for editing/committing. (Tmux would be cleaner but isn't worth the setup time.)
- Benchmark one iteration (or dry-run on a small config) before committing to any long training run. Don't guess wall-clock times.
- Kill criterion: kill a run when the *type* of problem changes (need visibility, need to reconfigure, broken assumption), not when "this is taking longer than I hoped."
- When patching existing files, prefer `sed`/`python heredoc with assert` over manual editing. The assert catches silent failure modes.

**Phase 2b will probably need 1-2 sessions.** The solver wrapper is the equivalent work to Session 2's Leduc solver wrapper (a few hundred lines), plus the new piece of wiring up `Abstraction.bucket_of` into the OpenSpiel information state pipeline. The training run itself, on Contabo CPU at tiny scale, will probably take ~30-60 minutes — long enough to need the per-iteration logging we learned to add in Session 2. Then resumable checkpointing is a separate chunk on top of that.
