# Prompt for Session 4 (paste verbatim at start of next conversation)

Continuing the pokerbot project. Session 4.

**Status:** Phases 2a, 2b, 2c all closed in one very long Session 3 that ran into 2026-05-22. Trained card abstraction at `runs/abstraction_20260521_223018/`. Custom Deep CFR solver with bit-identical resumable checkpointing (`src/nlhe/solver.py`). Slumbot client and eval harness validated (`src/nlhe/slumbot_client.py`, `scripts/eval_vs_slumbot.py`). Latest commit `796868b` ("Phase 2c: Slumbot evaluation harness").

**Runtime:** Contabo VPS, Python 3.10 venv. RunPod account created for Phase 2d (no pod rented yet, $0.34/hr 4090). $93 Vultr credit held for Phase 4.

**Today's session opens Phase 2d.** Three things on the front burner:

1. **PolicyAdapter** (deliberately deferred at end of Session 3, time to pick up): write the glue that wraps a trained `DeepCFRSolver` into a `choose_action(SlumbotState) -> str` callable for the eval harness. Translates Slumbot's chip-action-string state into our `InfosetEncoder` view, runs the strategy network, picks an action via the discrete action space, translates back to Slumbot's action string format.

2. **Phase 2b real run at 200bb (mandatory before eval, not optional)**: our Phase 2b smoke runs were all at 20bb stacks. Slumbot is 200bb. Going to 200bb is a *different training regime*, not a "scale up" — different action-relative-to-stack situations, longer betting trees, different equity distributions in postflop spots. Plan: verify the existing abstraction works at 200bb (EMD on equity histograms should be stack-agnostic, but spot-check the bucket assignments first), make a `configs/nlhe_200bb.yaml`, run an initial CPU training pass on Contabo to validate the pipeline at the new stack depth, then move to GPU.

3. **Phase 2d (main): rent RunPod 4090, train at 200bb with [256, 256] networks, eval against Slumbot.** Headline Phase 2 deliverable — a measurable bb/100 against the public benchmark.

**Smaller cleanups still open** (deferred from Session 2): `train_leduc.py` writes config.json/metrics.json into `checkpoints/`, `train_nlhe.py` writes them at run-dir root, `runs/README.md` claims root. Three-way inconsistency. Reconcile.

Read STATUS.md (entry point), SESSION_LOG.md latest entry, then DECISIONS.md for the EMD-abstraction and RunPod-provider entries. Confirm state, then start with the abstraction-validation-at-200bb spot check (cheap; informs whether we need to retrain abstraction before training the solver), then PolicyAdapter, then `configs/nlhe_200bb.yaml`.

**Workflow rules carrying over:**
- Single-quoted distinctive heredoc delimiters (`SOMETHING_EOF`).
- Verify file writes with `wc -l` / `head` / `tail` / `grep`.
- When patching: verify each patch landed before the next step.
- Benchmark before committing to long runs.
- Kill criterion: kill when the *type* of problem changes, not when "taking longer than hoped."
- Probe APIs with curl before writing client code (Session 3 lesson).
- Project files in /mnt/project/ may be stale; trust the live repo on Contabo as ground truth.
