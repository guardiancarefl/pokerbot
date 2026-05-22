# Prompt for Session 5 (paste verbatim at start of next conversation)

Continuing the pokerbot project. Session 5.

**Status:** Phase 2d step 1 closed in Session 4. `PolicyAdapter` shipped (`src/nlhe/policy_adapter.py`, 454 lines) plus 28 unit tests plus `eval_vs_slumbot.py` refactored for `--policy {random,adapter}`. End-to-end plumbing test validated: 20/20 hands clean against Slumbot at 20bb-checkpoint vs 200bb-game (intentional stack mismatch — validated wire-protocol round-trip, not strategy). The big find of the session was a real wire-format bug: Slumbot's `b<N>` is per-street, OpenSpiel `universal_poker`'s bet int is per-hand. Identical preflop, divergent the moment any chip enters the pot on a prior street. Fixed via a `prior_streets_committed_by_actor` kwarg on both translation helpers and a per-player dict refreshed at each postflop street boundary. Latest commit `c07fd9c`; cleanup `9e7be8b`.

Also during Session 4: the **200bb overnight CPU training run** was launched in a separate SSH session and ran unattended overnight (PID 730653 at session close, log `/tmp/200bb_overnight.log`, run dir `runs/nlhe_20260522_043341_phase2d_200bb_overnight/`, 300 iters at `[64,64]`/50/100, ETA ~07:15 Contabo). **First task of this session is to look at the results.**

**Runtime:** Contabo VPS, Python 3.10 venv. RunPod account in standby (no pod rented yet, $0.34/hr 4090). $93 Vultr credit held for Phase 4.

**Today's session opens by reviewing overnight artifacts.** Three things on the front burner, in priority order:

1. **Review overnight 200bb training results.** Final iter losses (adv0/adv1/strat0/strat1), full loss trajectory across 300 iters, confirm all 12 checkpoints saved (`ckpt_iter_0025.pt` through `ckpt_iter_0300.pt`), buffer behavior (`adv_buf_0`, `adv_buf_1`, `strat_buf` across iters). Sanity-check that losses decreased and didn't NaN. The 20bb smoke ended at adv=1.63 strat0=0.81 strat1=0.83 — 300 iters of 200bb at the same network should land lower on advantage at least.

2. **Plumbing-test the latest checkpoint against Slumbot at matched 200bb stack depth.** Run `python -m scripts.eval_vs_slumbot --policy adapter --checkpoint runs/nlhe_20260522_043341_phase2d_200bb_overnight/checkpoints/ckpt_iter_0300.pt --abstraction-path runs/abstraction_20260521_223018/abstraction.pkl --game-str "<200bb game_str>" --hands 100 --seed <fresh>`. **This is the actual Phase 2d headline test** — Session 4's 20-hand plumbing run used the 20bb checkpoint and was stack-mismatched by design. Now train and eval stacks match, the network is in-distribution, and the bb/100 number is meaningful (modulo small-sample variance — 100 hands is still low).

3. **Decide on RunPod rental for `[256,256]` scale-up training.** Conditional on the above:
   - If 300-iter CPU run produces measurable positive learning signal in Slumbot eval, GPU rental becomes a **"make it stronger"** task — a confidence-bearing scale-up rather than a debug effort.
   - If 300-iter CPU run produces garbage or hangs late, GPU rental is the **next debugging move with concrete data** on what didn't work at smaller scale.
   - If results are middling and bb/100 is within noise of zero, decide whether more CPU iters or jumping to GPU is the better next step.

**Carryover doc/code-hygiene items** (smaller cleanups, can land in any commit boundary):
- `train_leduc.py` config.json/metrics.json location: writes into `checkpoints/`; `train_nlhe.py` writes them at run-dir root (correct per `runs/README.md`). Three-way inconsistency. Reconcile.
- `_build_game_state_view` in `src/nlhe/solver.py` is module-private by underscore convention but now imported by `src/nlhe/policy_adapter.py`. Either rename to drop the underscore (it's now public-by-usage) or wrap in a thin public function.
- Three-way `--run-name` CLI inconsistency: `train_leduc.py` accepts `--run-name`, `train_nlhe.py` only uses YAML `tag:`. Unify or document.
- Possibly diagnose the **`[128,128]` / 100 trav / 200 steps hang** from the Session 4 iter-1 bench — process alive at 100% CPU but single-threaded, no parallelization. Unknown mechanism. Worth understanding before GPU run where we'll definitely want bigger networks.

**Workflow rules carrying over:**
- **Use tmux for long-running anything** (lesson from Session 4's two SSH dropouts — recoverable but tmux would have been cleaner up front).
- Single-quoted distinctive heredoc delimiters (`STATUS_EOF`, not `EOF`).
- Verify file writes with `wc -l` / `head` / `tail` / `grep`.
- When patching: verify each patch landed before the next step.
- **Benchmark one knob at a time before committing to a long run** (Session 4 lesson: tried to triple-bump network + traversals + train steps on top of a single-knob bench, hung at 100% CPU single-threaded).
- Kill criterion: kill when the *type* of problem changes, not when "taking longer than hoped."
- Probe APIs with curl (or direct library probe) before writing client code — Session 3's Slumbot lessons applied verbatim to Session 4's OpenSpiel bet-int probe and caught the per-hand/per-street bug.
- Project files in `/mnt/project/` may be stale; trust the live repo on Contabo as ground truth.
