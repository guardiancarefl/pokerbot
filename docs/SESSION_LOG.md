# Session Log

Append-only record of what happened in each working session and why. STATUS.md tracks current state; this file tracks history. When something in STATUS.md changes, the reason should be findable here.

Format: most recent session at the top. Each session block notes date, what was done, what was decided, what was learned, and what's queued for next time.

---

## Session 2 — 2026-05-21
**Focus:** Move from planning to execution. Get a working Linux environment with PyTorch + OpenSpiel, write the Phase 1 Leduc Deep CFR scaffold, complete a training run that validates the pipeline.

### What was done
- Attempted WSL2 + Ubuntu 22.04 install on Windows 11 host. WSL itself (2.7.3) installed but DISM failed at `Enabling feature VirtualMachinePlatform` with error 14098 (component store corrupted). Same failure for `Microsoft-Windows-Subsystem-Linux`.
- Ran `DISM /Online /Cleanup-Image /StartComponentCleanup` (no effect), then `DISM /Online /Cleanup-Image /RestoreHealth` (succeeded but didn't fix the feature install), then `sfc /scannow` (found and repaired corrupt files but features still wouldn't enable). After reboot both features still `State : Disabled`. The remaining repair paths (ISO-source DISM, in-place Windows reinstall) would have cost more time than just using a clean Linux machine that was already available.
- Pivoted to existing Contabo VPS: Ubuntu 24.04 (Noble), 12 vCPU AMD EPYC (oversubscribed), 48GB RAM, ~300GB free, no GPU.
- Created GitHub account `guardiancarefl`. Created private repo `pokerbot`. Generated ed25519 SSH key on Contabo, registered with GitHub.
- Installed system packages: `cmake`, `build-essential`, `software-properties-common`. Added `ppa:deadsnakes/ppa`. Installed `python3.10`, `python3.10-venv`, `python3.10-dev` alongside system Python 3.12. (OpenSpiel's officially-tested Python range is 3.7-3.10, so 3.10 is the safer target.)
- Created `~/pokerbot/` on the Contabo box. Initialized git, set remote, wrote `.gitignore` and `README.md`. Wrote all five foundational docs verbatim from Session 1 versions. Committed and pushed (`c756764`).
- Updated STATUS.md, SESSION_LOG.md, DECISIONS.md, ARCHITECTURE.md to reflect the runtime migration. Committed and pushed (`52c3fd5`).
- Created venv at `~/pokerbot/.venv`. Installed PyTorch CPU build (2.12.0+cpu) and OpenSpiel (1.6.11). Smoke tests confirmed both import and Leduc loads.
- Built Phase 1 scaffold as a modular structure designed for reuse in Phase 2+: `src/leduc/config.py` (TrainConfig dataclass + YAML loader), `src/leduc/solver.py` (wraps OpenSpiel's DeepCFRSolver), `src/leduc/evaluate.py` (exploitability in mbb/g), `src/leduc/checkpoint.py`, `configs/leduc_default.yaml` + `configs/leduc_smoke.yaml`, `scripts/train_leduc.py` (~180 lines), `tests/test_pipeline.py` with 4 fast tests. Committed and pushed (`4f5c2d8`).
- Pipeline tests passed but the first real training run with `leduc_default.yaml` (100 iters x 40 traversals x 500 adv steps) ran for 30+ minutes with **no per-iteration visibility** because OpenSpiel's `solve()` loops internally with no callback hooks. Killed it.
- Tried a smaller config (`leduc_phase1.yaml`, 40 iters x 200 adv steps). First iteration came back at 67 seconds. Estimated total 45+ minutes. Killed it because the lack of visibility was the real problem, not the runtime.
- Rewrote `src/leduc/solver.py` to drive training one iteration at a time: construct solver with `num_iterations=1`, call `solve()` in a Python loop, accept optional `logger` and `eval_callback`. Cost: one extra policy-network training pass per loop iteration (since OpenSpiel's `solve()` includes policy training as its final step). Benefit: per-iteration progress logging and periodic exploitability eval.
- Updated `scripts/train_leduc.py` to pass the logger and an exploitability eval_callback (every 10 iterations, plus final iteration).
- Caught a None-handling bug in the new solver: OpenSpiel's `_learn_strategy_network()` can return None when the strategy reservoir buffer is too small, same way `_learn_advantage_network()` can. Both now coerced to NaN.
- Patched `runs/` gitignore: switched from `runs/` to `runs/*` so the negation rule `!runs/README.md` could take effect (git can't re-include files under an excluded directory). Wrote `runs/README.md` documenting the run directory format. Wrote `docs/PHASE2_SKETCH.md` as a forward-looking Phase 2 plan to give Session 3 a starting point. Committed and pushed (`d0688f9`).
- Ran the third training attempt with `leduc_phase1.yaml` cut to 25 iterations. Per-iteration timing was very noisy on the oversubscribed Contabo CPU: individual iterations ranged from 13s to 151s, averaging 91s/iter. Total runtime 38 minutes.
- **Phase 1 result: exploitability 1187 (uniform random) -> 502 (iter 10) -> 447 (iter 20) -> 434 (iter 25) mbb/g.** Clear downward trend, decelerating as expected. Final checkpoint saved at `runs/leduc_20260521_210552_phase1_take2/checkpoints/final.pt`. metrics.json and config.json saved as companions.

### What was decided
- **Runtime moved from WSL2-on-Windows to Contabo VPS** (recorded in DECISIONS.md, supersedes original WSL2 entry).
- **GPU work deferred to rented cloud hardware in Phase 4+** (or earlier if Phase 2d needs it — see open questions). The local RTX 3060 isn't part of the plan anymore.
- **OpenSpiel reference implementation for Phase 1**, wrapped not reimplemented. The goal of Phase 1 is pipeline validation, not algorithmic research.
- **Per-iteration solver loop** instead of single `solve(num_iterations=N)` call. Trades small compute overhead for the visibility needed to make informed decisions during runs. Acceptable for Leduc; may revisit for Phase 2+ if profiling shows it matters.
- **Phase 1 config sized for time-on-Contabo, not for Leduc record exploitability.** 25 iterations with 200/400 train steps and 64-unit networks. Result (~434 mbb/g) is well above published Leduc Deep CFR levels (50-250 mbb/g typical) but well below uniform random (1187), demonstrating the pipeline works. Phase 1's job was infrastructure validation, not Leduc benchmarking.
- **42 bought-bot profiles stay in Phase 3.** Brief discussion this session about skipping ahead to integrate them after Phase 1; the conclusion was that without a working NLHE training environment (Phase 2), there's no integration point for them. Phase 2 is the necessary bridge.

### What was learned / surprises
- The Windows component store corruption was the kind of environmental problem that no amount of project-specific planning would have caught. Worth flagging: operating systems can be broken in ways that look like "your install command failed" but are actually "your system needs an in-place repair." Knowing when to bail vs. push through is a real skill.
- **My time estimates for ML iterations were repeatedly wrong tonight.** Estimated "3-10 minutes," then "30-90 minutes," then "~18s/iter" (actually 67s+). The lesson: benchmark one iteration of the real config before committing to a full run. We should have run a single-iteration timing check before each config change.
- **A throwaway micro-benchmark gave wildly inconsistent numbers** on Contabo (17.9s vs 42.1s for same workload variants in opposite of the expected direction). Contabo's oversubscribed vCPUs mean per-iteration time varies by ~10x depending on neighbor activity. Adequate for development; useless for benchmarking. Phase 4+ compute planning must be done on dedicated hardware.
- **OpenSpiel's `solve()` returns None for losses when buffers are too small** for either advantage or strategy training. Came up in two places (advantage net + policy net), both initially crashed our code with `TypeError: float() argument must be a string or a real number, not 'NoneType'`. Now handled by NaN coercion. Worth defensive-coding for in any wrapper around library code.
- **Three killed training runs in one session.** The first kill was justified (no visibility). The second was justified (no visibility + bad timing estimate). The third would have been the impatience tax — we didn't kill that one and got the result. Pattern noted: kill when the *type* of problem changes (need visibility, need to reconfigure), not when "this is taking longer than I hoped."
- **Visual mash from heredoc + command echoes** kept making terminal pastes look corrupted. Every time, the actual file contents were correct (proven by verify commands). Lesson: trust `wc -l` / `head` / `tail` / `grep` over visual scan of terminal echo.

### Workflow notes for next time
- Run a single-iteration timing benchmark on the real config before kicking off a long training run. Saves wall-clock time on misjudgments.
- The 30-minute "is this hung or running" anxiety is the worst use of time in the project. Per-iteration logging eliminates it; future custom solvers (Phase 2+) need to preserve this.
- Two SSH sessions to the Contabo box was the productive workflow (one for training, one for editing/committing). Tmux next session would be cleaner than juggling SSH windows.
- The Session 1 verbatim-not-silent-edit rule held up well. Heredoc-based file writes were the right pattern.

### Phase 1 cleanup deferred to Session 3
- `train_leduc.py` writes its companion config.json and metrics.json into the `checkpoints/` subdirectory, not at the run-dir root as `runs/README.md` claims. Two options: update the README to match reality, or update the script to write both root and companion copies. Either is fine; not blocking Phase 2.

### Queued for next session
1. Decide GPU provider for Phase 2d training. Compare current pricing for Vast.ai 4090, RunPod 4090, Vultr A40. Pick a winner.
2. Phase 2a: load HUNL in OpenSpiel via universal_poker, validate the game representation, confirm action and information state encoding.
3. Phase 2a: build card abstraction module (`src/nlhe/abstraction.py`) using EMD clustering on equity distributions. Target ~200 buckets per street.
4. Phase 2a: build action abstraction module (`src/nlhe/actions.py`) with discretized bet sizes {check/fold, call, 0.33pot, 0.66pot, 1pot, 2pot, all-in} and translation of off-tree opponent sizes.
5. (Small) Fix the Phase 1 cleanup deferred item above.
6. (Small) Investigate why advantage loss kept climbing through the run. Per the OpenSpiel pattern with `reinitialize_advantage_networks=True`, climbing loss reflects growing-buffer complexity not network failure, but worth a sanity check against a published reference.

---

## Session 1 — 2026-05-21
**Focus:** Project bootstrap. Establish foundational docs and local infrastructure for Claude Code.

### What was done
- Identified that the local machine is Windows 11 with an RTX 3060 Laptop GPU (6GB VRAM), not the 12GB desktop variant originally assumed in project docs. Updated ARCHITECTURE.md and DECISIONS.md to reflect this.
- Confirmed OpenSpiel does not officially support Windows native. Decided to use WSL2 + Ubuntu 22.04 as the runtime environment.
- Installed Claude Code on the Windows host. Resolved PATH issue so `claude` is callable from any PowerShell.
- Created project directory at C:\Users\Ngior\pokerbot\ with docs/ subfolder.
- Created the four foundational docs verbatim from contents provided in the planning chat: PROJECT_OVERVIEW.md, ARCHITECTURE.md, DECISIONS.md, STATUS.md.

### What was decided
- **Runtime environment: WSL2 + Ubuntu 22.04** on the Windows host. (Superseded in Session 2.)
- **Opponent anonymity as a core design principle.** No persistent identity for any opponent across matches. No pre-collected real-world hand history data feeds training. The bot's information state mirrors a competent human at an anonymous online table. Values-driven decision — robustness and fairness over peak exploitation EV.
- **Within-match adaptation at Position 2.** Light-to-medium online statistics nudge subgame solver ranges within the current match, anchored to the blueprint as a safety floor. When a match ends, all derived state is wiped.
- **The 42 bought-bot profiles stay frozen.** Style diversity in training and stable benchmark targets. Strength diversity comes from league play (PSRO with archived self).

### What was learned / surprises
- The original ARCHITECTURE.md mixed two design goals — anonymous Nash-leaning play and identified-opponent exploitation. Removing the exploitation layer made the project cleaner.
- Claude Code defaulted to its own judgment when given file contents to write verbatim — it removed a line on its own and announced the edit after the fact. A hard rule was set: when exact contents are given between BEGIN/END markers, write them exactly; concerns must be raised as questions before writing, not as silent edits. Noted in STATUS.md under Known issues.

### Workflow notes for next time
- Always verify Claude Code's output, not just its self-reports. "Done" doesn't always mean done correctly.
- The relay workflow (Claude planning chat ↔ Claude Code execution) works, but only if outputs are actually read end to end before being passed onward.

### Queued for next session
1. Install WSL2 + Ubuntu 22.04 on Windows host
2. Verify CUDA passthrough from WSL2 to the RTX 3060 Laptop GPU
3. Set up Python 3.10 environment with venv inside WSL2
4. Install PyTorch (CUDA build) and OpenSpiel inside the venv
5. Confirm PyTorch sees the GPU and OpenSpiel loads Leduc poker
6. Move the project directory from C:\Users\Ngior\pokerbot\ to the WSL2 Linux filesystem (better I/O than the Windows mount)
7. Install Claude Code inside WSL2 for ongoing project work
8. Begin Phase 1 scaffold: Leduc Deep CFR training script

---

## Session 3 — 2026-05-21
**Focus:** Open Phase 2. Decide GPU provider for Phase 2d. Load HUNL in OpenSpiel, validate game representation. Build card abstraction (EMD-based) and action abstraction. End-state goal: Phase 2a closed, foundations in place for Phase 2b training pipeline.

### What was done
- Researched current pricing for Vast.ai, RunPod, and Vultr GPU options. Recommendation made, committed to project docs, RunPod account created (no pod rented yet — Phase 2a and 2b run on Contabo CPU; first rental triggered at start of Phase 2d). DECISIONS.md entry locks the choice.
- Validated HUNL game loads in OpenSpiel via `universal_poker` with ACPC-style parameter string. Numbers confirmed: 760-dim information state tensor, max_game_length 218, ~20000-action node before abstraction with dynamic shrinking as effective stack drops. Imperfect-information encoding verified (each player sees only their hole cards).
- Walked a HUNL game from initial state under uniform-random play, validating that the engine correctly handles chance nodes, decision nodes, terminal returns. Returns are zero-sum chips (`[20000, -20000]`), to be wrapped with ICM in Phase 4.
- Installed `treys` (Python poker hand evaluator). `pokerkit` would have been preferred for capability but requires Python 3.11+; we're pinned to 3.10 by OpenSpiel's tested range. Treys imported, sanity-checked AA>KK, AA-vs-random equity at 0.847 vs literature 0.85.
- Built `src/nlhe/equity.py` (191 lines): wraps `treys`, exposes 169-class canonical hole enumeration, string-based card I/O, Monte Carlo `equity_vs_random` and `equity_vs_range`. Smoke-tested at 6 levels: AA at 0.856, 72o at 0.348, paired-board AA at 0.888, wet-board AA at 0.701. All match literature.
- **Decision: EMD over OCHS for card abstraction.** PHASE2_SKETCH had flagged this as a session decision. After laying out three readings of the question (will-need-eventually vs values-driven vs default-to-harder-on-autopilot), confirmed Reading 2 — values-driven choice for the gold-standard technique. Logged in DECISIONS.md.
- Built `src/nlhe/abstraction.py` (326 lines): equity-histogram generation via Monte Carlo runouts, pairwise EMD via `scipy.stats.wasserstein_distance`, custom PAM (k-medoids) clustering, `Abstraction` container with `bucket_of`/`save`/`load`. Validated end-to-end on a tiny-scale smoke test (preflop k=5 in ~10 seconds). Numbers checked: AA in different bucket from 72o, d(AA, 72o) = 0.54, kmedoids cost decreases monotonically and converges in <5 iterations.
- Built `scripts/train_abstraction.py` (158 lines) for the production training run + `scripts/inspect_abstraction.py` (77 lines) for post-hoc analysis. Dry-run on preflop only first; clean. Then full run across all four streets in 6.8 minutes wall-clock. Bucket distributions are sensible (preflop median 6 hands/bucket on k=20, postflop median 7 hands/bucket on k=200, max 19–26 across streets — no degenerate one-giant-bucket clustering).
- Inspected the trained flop abstraction in detail. EMD is doing real strategic clustering: different surface hands with similar histogram shapes group together. Specific verifications:
  - "Drawing dead on coordinated board" hands (e.g., `4h2s` on `8s6dTh`, `5h2c` on `Kc3d9h`) cluster despite different surface cards — same equity-histogram shape.
  - Set on dry board (`3h3d` on `3cKd7h`) at equity 0.96 sits in its own bucket above two-pair-no-improvement clusters.
  - Pocket pair tiers (99 / JJ / QQ / AA) end up in separate buckets even where mean equities are close — EMD's histogram-shape sensitivity is the value-add over OCHS, and it works.
- Built `src/nlhe/actions.py` (~230 lines after patches): `DiscreteAction` enum {fold, call, 0.33pot, 0.66pot, 1pot, 2pot, allin}, `policy_to_game_action` (discrete → OpenSpiel integer), `game_to_policy_action` (OpenSpiel integer → distribution over discrete via pseudo-harmonic translation from Ganzfried & Sandholm 2013), `discretize_legal_actions` (filter at decision time). Smoke-tested with mid-pot, small-pot, large-pot views.
- Caught and patched two bugs in the action module on the first smoke run. First: `policy_to_game_action` was clamping sub-min-bet sizes up to min_bet, causing 0.33pot/0.66pot/1pot to alias to the same chip count on small pots. Fix: return `None` for sub-min-bet sizes, treat them as unavailable in that state. Second: `_legal_discrete_bet_sizes` was deduping at the same chip count by keeping the first (smallest-label) action; it should keep the largest. Fix: use a dict keyed by chip count, so later (larger) actions overwrite earlier ones at the same count. Re-run after both patches: clean.
- Updated DECISIONS.md (EMD abstraction entry), STATUS.md (Phase 2a closure + Phase 2b queue), SESSION_LOG.md (this entry).

### What was decided
- **GPU provider for Phase 2d: RunPod Community Cloud RTX 4090** at $0.34/hr. Vast.ai marginally cheaper but more variable; Vultr 5x the cost for unneeded VRAM. $93 Vultr credit held for Phase 4 (multi-day blueprint training where reliability matters) or Phase 3 helper if needed.
- **Card abstraction: EMD on equity histograms, not OCHS.** Values-driven choice: EMD is the gold-standard technique and implementing it once now means we've done it properly when Phase 4 needs an even harder abstraction. Trade-off acknowledged (2-3x the code of OCHS, no measurable Slumbot-cycle validation against simpler alternative).
- **EMD sample sizes for Phase 2a:** preflop=169 canonical hands × 400 runouts → k=20. Postflop=1500 sampled (hand, board) combos × 200 runouts → k=200. 50 histogram bins per equity distribution. 6.8 min total training time, well within budget.
- **Action abstraction: 5 bet sizes** {0.33pot, 0.66pot, 1pot, 2pot, allin} + fold/call. Confirmed as ARCHITECTURE.md's target. Acknowledged that opponent bets between 2pot and 0.9×stack all snap to 2pot — coarse but acceptable for Phase 2; revisit if Slumbot shows overbet-sizing exploits.
- **Action illegality is signal, not noise.** When a discrete action's intended chip target falls below the legal min_bet, return None rather than aliasing to min_bet. The policy network will softmax only over legal discrete actions in that state.

### What was learned / surprises
- **Visual mash from heredoc echo continues to mislead, including misleading the assistant.** Twice this session: once after the abstraction.py write where my eye saw the closing `STATUS_EOF` fused to fragments of later content and I worried the file was truncated; once after the STATUS.md write where `tail -3` showed three log entries and I momentarily thought Session 3 had been appended twice. Both times the file on disk was correct, verified by `wc -l` and `grep`. Session 2's lesson holds: trust `wc -l` / `head` / `tail` / `grep` over visual scan of terminal echo.
- **Pasting large content directly into a bash prompt fails noisily.** Early in the session, the first DECISIONS.md content was pasted without the surrounding `cat << 'EOF'` heredoc wrapper. Bash tried to execute each line as a command, producing a wall of `command not found` errors. No actual file damage, but a real reminder that BEGIN/END markers in the assistant's instructions are *labels* describing where content goes, not shell commands to type. The fix going forward: always wrap content in a heredoc, single-quote the delimiter to prevent variable expansion, give the delimiter a distinctive name (`STATUS_EOF` not `EOF`).
- **The "trust the test, don't trust my expectation" lesson, again.** On the equity calculator's flop test, AA on `2c7d9s` came back at 0.86, and I'd written "expected 0.88-0.92" — looked like a bug. Higher-precision MC (20k trials) confirmed 0.859. The function was correct; my expected range was wrong. The same instinct would have wasted an hour chasing a non-existent bug if I'd trusted my prior over the measurement.
- **EMD really does what the literature says it does, and inspection makes it visible.** The flop bucket inspection was the most satisfying moment of the session — seeing different surface hands with the same strategic situation correctly clustered, and seeing pocket pair tiers correctly separated by histogram *shape* not just *mean*. Not abstract theory; visible in the output table.
- **First-attempt bugs in action translation were the right ones to catch.** Both bugs I caught (sub-min-bet aliasing, dedupe-by-keeping-first) would have produced subtle problems downstream — the policy would have learned weird patterns in small-pot states. Catching them in the smoke test before any training run is cheap; catching them after a multi-hour Phase 2d Slumbot evaluation would have been expensive.
- **Time estimates were better this session than last.** Predicted 12-15 minutes for abstraction training; actual was 6.8. Predicted ~120s for pairwise EMD per postflop street; actual 107-125s. Predicted ~10s for preflop smoke test; actual ~4s. Session 2's "always benchmark one iteration before committing to a full run" approach (here: dry-run on preflop before full four-street run) is paying off.

### Workflow notes for next time
- The two-SSH-session pattern was the right shape: one window for the training run, one for editing, one for inspection. Tmux would still be cleaner — defer until it becomes a friction point.
- Heredoc paste failures continue to be the dominant low-level annoyance. Single-quoted distinctive delimiter is the rule; not optional.
- When patching existing files, sed-via-python-heredoc with `assert old in src` is safer than raw sed. The assert catches silent failure modes where sed would otherwise no-op on a near-miss.
- The "kill criterion" from Session 2 (kill when the *type* of problem changes, not when "this is taking longer than hoped") held this session — we never killed a run.

### Phase 2a cleanup that's still open
- `runs/README.md` vs `train_leduc.py` config/metrics location mismatch (deferred from Session 2). Not blocking Phase 2b.
- Advantage loss sanity check against published Deep CFR Leduc reference (deferred from Session 2). Also not blocking. Both are good candidates for "warm-up" tasks at start of Session 4.

### Queued for next session
1. Phase 2b: build NLHE solver wrapper. Should follow the Leduc solver pattern (per-iteration logging, eval callback, NaN-safe loss capture). Information state encoding has to combine card bucket (from `Abstraction.bucket_of`) with betting history features.
2. Phase 2b: train Deep CFR on tiny HUNL (20bb stacks, coarse abstraction, [64,64] networks) on Contabo CPU. Goal is "the pipeline doesn't explode," not "good HUNL strategy."
3. Phase 2b: implement resumable training (checkpoint every N iterations, idempotent resume). Hard prerequisite for RunPod Community Cloud in Phase 2d.
4. Optionally Phase 2c: start the Slumbot harness. Independent of 2b work, could be done in parallel.
5. Session 2 cleanups (the two open items above) if time permits.
