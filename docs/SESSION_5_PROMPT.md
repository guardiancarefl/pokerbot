# Prompt for Session 5 (paste verbatim at start of next conversation)

Continuing the pokerbot project. Session 5.

**Status:** Phase 2b closed (Session 3 ran long, into the early morning of 2026-05-22). HUNL game validated, equity calculator + EMD card abstraction + action abstraction built (Phase 2a). Custom Deep CFR solver + info-state encoder + resumable checkpointing + YAML training script built (Phase 2b). **Bit-identical resume correctness verified** (max param diff 0.00e+00). Phase 2b training pipeline is ready; the gate for Phase 2d is no longer the solver — it's the Slumbot evaluation harness.

**Runtime:** Contabo VPS, Ubuntu 24.04, Python 3.10 venv, latest commit `563793f` ("Phase 2b: HUNL Deep CFR solver with resumable training"). Trained abstraction at `runs/abstraction_20260521_223018/abstraction.pkl`.

**GPU decision still standing:** Phase 2d uses RunPod Community Cloud RTX 4090 at $0.34/hr. Account created Session 3, no pod rented yet.

**Today's session opens Phase 2c: Slumbot evaluation harness.** Next-up items:

1. Phase 2c: build `src/nlhe/slumbot_client.py` — HTTP client against the Slumbot 2017 API, hand-history parsing, bb/100 calculator.
2. Phase 2c: write an eval script that runs N hands of a policy vs Slumbot and reports win rate. Test with random-policy baseline first to validate the harness before plugging in our trained Deep CFR policy.
3. (Optional) Phase 2b real-run on Contabo CPU: execute `configs/nlhe_phase2b.yaml` (100 iter × 100 trav, estimated 3-5 hours). Gets a real baseline policy before going to GPU. Could overlap with 2c work in the second SSH session.
4. Phase 2d: rent RunPod 4090, scale to [256,256] networks, get measurable bb/100 vs Slumbot.

**Cleanups still open** (originally Session 2 deferred): `train_leduc.py` writes config.json/metrics.json into checkpoints/, `runs/README.md` claims root. `train_nlhe.py` does it correctly. Reconcile.

Please read STATUS.md (entry point), then SESSION_LOG.md Session 3 extended entry (for context on Phase 2b design decisions), then DECISIONS.md latest entries. Confirm state matches STATUS, then start with Phase 2c.

**Workflow rules carrying:** Single-quoted distinctive heredoc delimiters (`SOMETHING_EOF` not `EOF`). Verify file writes with `wc -l`/`head`/`tail`/`grep`. When patching: verify each patch landed before moving to the next step. Benchmark before committing to long runs. Kill criterion: kill when the *type* of problem changes, not when "taking longer than hoped."
