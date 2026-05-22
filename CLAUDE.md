# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A 6-max NLHE SNG poker bot, top-3 equal payout format. Deep CFR blueprint with an ICM-adjusted value function, PSRO league play, and a within-match adaptation layer. Built around **opponent anonymity** as a core constraint — no persistent per-opponent state across matches, no real-world hand-history data in training. See `docs/ARCHITECTURE.md` for the four-layer stack and `docs/DECISIONS.md` for why each piece was chosen.

The project is staged in phases. Current state and what's open lives in `docs/STATUS.md`; the chronological "why" lives in `docs/SESSION_LOG.md`. **Read those two files first when picking up work** — they're authoritative over anything you might infer from the code.

**Then cross-check with git.** Run `git log --oneline -10` before designing or coding anything. STATUS.md can lag the last commit, and trusting a stale STATUS will cost a full session of rebuilt work. If STATUS says a track is "next up" but the most recent commits land that track, believe the commits. Session 7.5 (2026-05-22) burned hours on this exact failure mode.

## Runtime

- **Host:** Contabo VPS, Ubuntu 24.04, 12 vCPU AMD EPYC (oversubscribed — per-iteration timings vary ~10×), 48 GB RAM, no GPU. Project lives at `~/pokerbot/`.
- **Python:** 3.10 from the `deadsnakes` PPA (system Python is 3.12 — do not use it). Venv at `.venv/`. OpenSpiel's officially-tested range is 3.7–3.10, which pins us.
- **Deps:** PyTorch CPU build, OpenSpiel 1.6.11, `treys` for hand evaluation, `scipy` for EMD. Pinned in `requirements.txt`.
- **GPU:** none on Contabo. RunPod Community Cloud RTX 4090 ($0.34/hr) is the chosen provider for Phase 2d+; account exists but no pod rented yet.

Activate the venv before running anything: `source .venv/bin/activate`.

## Common commands

```bash
# Tests (fast smoke tests, ~30s — not algorithm validation)
python -m pytest tests/ -v
python tests/test_pipeline.py          # also works; inserts repo root on sys.path

# Phase 1 — Leduc (validated, OpenSpiel-wrapped)
python scripts/train_leduc.py --config configs/leduc_smoke.yaml
python scripts/train_leduc.py --config configs/leduc_default.yaml --run-name foo
python scripts/train_leduc.py --iterations 200 --learning-rate 0.0005   # CLI overrides

# Phase 2a — train card abstraction (EMD k-medoids; ~7 min on Contabo CPU)
python scripts/train_abstraction.py
python scripts/inspect_abstraction.py runs/abstraction_<ts>/abstraction.pkl

# Phase 2b — train NLHE Deep CFR solver. MUST run as module (uses `src.nlhe.*` imports).
python -m scripts.train_nlhe --config configs/nlhe_smoke.yaml
python -m scripts.train_nlhe --config configs/nlhe_phase2b.yaml --resume runs/.../ckpt_iter_0050.pt

# Phase 2c — eval against Slumbot's public API (no API key needed)
python -m scripts.eval_vs_slumbot --policy random --hands 100
```

Run-dir layout is documented in `runs/README.md`. **Known inconsistency:** `train_leduc.py` writes `config.json`/`metrics.json` into `checkpoints/`, `train_nlhe.py` writes them at run-dir root, and `runs/README.md` claims root. Tracked in STATUS.md.

## Code architecture

Two parallel training stacks, intentionally — Phase 1 was de-risking infrastructure with a thin wrapper; Phase 2 is the real engineering.

### `src/leduc/` (Phase 1, closed)
Thin wrapper around `open_spiel.python.pytorch.deep_cfr.DeepCFRSolver`. Single-iteration solve loop (built `num_iterations=1` inside a Python `for`) so we get per-iteration logging and periodic exploitability eval — OpenSpiel's stock `solve(num_iterations=N)` has no callback hooks. Both `_learn_advantage_network` and `_learn_strategy_network` can return `None` when reservoirs are small; coerce to NaN.

### `src/nlhe/` (Phase 2, in progress) — custom implementation, not wrapping OpenSpiel
- `equity.py` — `treys`-backed Monte Carlo equity (`equity_vs_random`, `equity_vs_range`), 169-class canonical hole enumeration. AA-vs-random sanity = 0.847.
- `abstraction.py` — EMD on equity histograms + custom PAM/k-medoids clustering. **EMD, not OCHS** — values-driven choice locked in DECISIONS.md. Preflop k=20, postflop k=200. Trained artifact lives in `runs/abstraction_*/abstraction.pkl` and is referenced by path from each NLHE training config.
- `actions.py` — `DiscreteAction` enum `{fold, call, 0.33pot, 0.66pot, 1pot, 2pot, allin}`. `policy_to_game_action` returns **`None`** for sub-min-bet sizes (illegality is signal, not noise — don't alias up to min_bet). `game_to_policy_action` does pseudo-harmonic translation (Ganzfried & Sandholm 2013) for off-tree opponent sizings.
- `infoset.py` — 214-dim feature vector: bucket one-hot (200) + street (4) + position (2) + pot/to_call/eff_stack (3) + betting features (5). **Normalization is parametric via `InfosetEncoder.starting_stack`** (dataclass field, default 20000). When changing stack depth across training/eval, pass the right value at construction.
- `solver.py` — custom external-sampling Deep CFR, RM+, 2 nets per player (advantage + strategy), reservoir buffers. **Regrets are normalized by `starting_stack`** — without this, MSE on chip-scale regrets is in the millions. Checkpointing serializes all 4 networks + 4 optimizers + 3 reservoir buffers + Python/torch RNG state and is **bit-identical on resume** (verified `max param diff = 0.00e+00`). Don't break this — RunPod Community Cloud preemption recovery depends on it.
- `slumbot_client.py` — HTTP client + action-language parser. Slumbot's wire language uses `k` (check) and `c` (call) **strictly** — never interchangeable. Preflop SB faces the implicit BB even with empty action string. Response includes `baseline_winnings` (Slumbot's built-in variance-reduction signal — use as the headline eval metric, not raw `winnings`).

### Config / artifact flow
NLHE training config (`configs/nlhe_*.yaml`) points to a pre-trained abstraction by path (`abstraction_path:`), then the solver loads that pickle and constructs the `InfosetEncoder` and the game. Stack depth lives in two places that must agree: the `stack=NNNN NNNN` field of the OpenSpiel `game_str` and `starting_stack:`. Existing configs are all 20bb (`stack=2000`); Slumbot is 200bb. See SESSION_4_PROMPT.md.

## Project workflow rules (carried session-to-session)

These come from repeated lessons in `docs/SESSION_LOG.md`:

- **Heredocs for file writes:** single-quoted, distinctive delimiter (`STATUS_EOF`, not `EOF`). Prevents shell expansion and prevents bash from trying to execute pasted content as commands.
- **Verify file writes with `wc -l` / `head` / `tail` / `grep`, never visual scan of terminal echo.** Heredoc echoes look corrupted constantly; the file on disk is almost always fine. Trust the verification commands.
- **When patching:** when a patch sequence has multiple steps, gate each step on verification of the previous one. Don't bundle "apply patch + grep to verify" in one shot — silent no-op patches are a real failure mode here.
- **Benchmark one iteration of any new config before committing to a long run.** Session 2 lost hours to "I estimated 10 min, actual was 60 min."
- **Kill criterion:** kill a run when the *type* of problem changes (no visibility, wrong config, hardware misallocated) — not when "this is taking longer than I hoped." Impatience-killed runs cost more than they save.
- **Probe APIs with curl before writing client code.** Session 3's Slumbot integration found 3 undocumented action-language quirks this way; Session 2's lessons were learned the harder way.
- **Per-iteration logging is non-negotiable** for any solver, custom or wrapped. The "is this hung or running" anxiety is the worst use of time in this project.
- **`/mnt/project/` may be stale** — the live repo on Contabo at `~/pokerbot/` is ground truth.

## Docs hierarchy (read in this order when orienting)

1. `docs/STATUS.md` — current phase, what's open, known issues. Single source of truth for "where are we."
2. `docs/SESSION_LOG.md` — append-only history. Each session block has *what was done*, *what was decided*, *what was learned*. Anything in STATUS has its reason here.
3. `docs/DECISIONS.md` — locked-in choices with alternatives considered and why-rejected. Don't relitigate without reading the relevant entry first.
4. `docs/ARCHITECTURE.md` — the four-layer system design and phase plan.
5. `docs/SESSION_4_PROMPT.md` (or the most recent `SESSION_N_PROMPT.md`) — handoff brief for the current session.
6. `docs/PROJECT_OVERVIEW.md` and `docs/PHASE2_SKETCH.md` — background context, no longer load-bearing.

When a session closes, update STATUS, append to SESSION_LOG, and add to DECISIONS for any new locked-in choice. Commit docs separately from code when the boundary is clean.
