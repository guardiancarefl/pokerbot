# NEXT_SESSION — Session 22 close-out → Session 23 pickup

**Date written:** 2026-05-27 (Session 22 close)
**Pickup target:** Scenario 3 **Step 4 — stack-distribution sampling**

This handoff is written so the next session's pickup is mechanical: branch state,
what landed, where we are in the master sequence, and the exact design surface
Step 4 opens with.

---

## 1. Branch state

- **Branch:** `phase4f-league`
- **HEAD:** `e144092` (`docs(decisions): record Scenario 3 step 3 design decisions`)
- **Local == remote:** yes (`git rev-parse HEAD origin/phase4f-league` match).
- **Tracked working tree:** clean.
- **Pre-existing untracked files remain out of scope:** `pushfold_*.py`, `*.zip`
  (phase5_train_v1, pushfold_*, subgame_v*), and the loose `evals/*.json`
  (leaf_eval_ablation_session17, league-v2-600_vs_pool_5000, subgame_ablation_smoke6c).
  These predate Session 22; left untouched.

---

## 2. What landed this session (four commits)

| SHA | Subject |
|-----|---------|
| `0bbdb86` | feat(b1c): sub-step 6 CLOSED — verdict PASS_BR_EQUIVALENT_TO_PROFILE |
| `4046caf` | refactor(cfr6,actions): fast_view fold-in (delegation pattern, bit-identical training output) |
| `78917f8` | feat(solver6): shared strategy net (Pattern A, HUNL template, single-net design) |
| `e144092` | docs(decisions): record Scenario 3 step 3 design decisions |

---

## 3. Scenario 3 progress (the master sequence)

- ✓ **Step 1 — Sub-step 6 close** (`0bbdb86`) — PASS_BR_EQUIVALENT_TO_PROFILE; ship PROFILE_SAMPLE.
- ✓ **Step 2 — fast_view fold-in** (`4046caf`) — canonical `_build_view_6max` + `discretize_legal_actions` delegate to `fast_view`; training output bit-identical.
- ✓ **Step 3 — Strategy net for 6-max** (`78917f8` + `e144092`) — single shared strat net, v2 checkpoint schema, two-tier load, two-signal SubgamePolicy.
- → **Step 4 — Stack-distribution sampling** — *next pickup* (see §5).
- **Step 5 — Archetype mix for 6-max** — port HUNL archetype framework (position derivation for all 6 seats).
- **Step 6 — League v2 config + execution.**
- **Step 7 — Multi-day production blueprint training** (24–72h GPU) — produces the v2-era checkpoints; must beat `dcfr-overnight-3000` head-to-head.
- **Step 8+ — Within-match observation framework, Slumbot HUNL bridge, 42 shanky bots.**

**3 of 7 core steps complete.**

---

## 4. Step 4 design surface (what Session 23 opens with)

**Goal:** replace the uniform `[cfg.starting_stack] * 6` per-traversal stacks with
sampled stack distributions representing real mid-tournament states (chip leader,
short stack, bubble asymmetry). This is **deferral #4** of the Session-9
"6-max blueprint training: minimum-viable first cut" entry (`docs/DECISIONS.md:339`;
deferral #1 was resolved this session, #2 DCFR / #3 archetypes still open).

**Design questions outstanding — each needs a decision before code:**

- **(a) Distribution shape:** uniform-over-plausible / late-stage-weighted /
  empirical-trajectory / hybrid.
- **(b) Sampling cadence:** per-traversal / per-iteration / per-N-iterations.
- **(c) Constraints:** reachable-state-only or unconstrained positive 6-vector
  (sum = 6 × starting_stack, or free?).
- **(d) ICM payout interaction:** does the existing `src/nlhe/icm.py` handle
  arbitrary positive 6-vectors, or does it need updating? (Recon item.)
- **(e) Acceptance gate:** **no bit-identity** — we are changing the stack
  distribution by design, so the advantage-net trajectory *will* change. The gate
  is "training still converges sensibly" (loss trajectory healthy, no divergence)
  **plus** a head-to-head: stack-sampled training vs the uniform-stack baseline.

**Open with a read-only recon prompt** (drafted in the Session-22 transcript):
inspect `state6.py` / stack-handling in `solver6.py`, `icm.py`'s 6-vector support,
the config knobs (`TrainConfig6Max`), and any prior art (`sample_starting_state`
already exists in `scripts/eval_6max_self_play.py` / `eval_pool.py` for eval-time
state sampling — check whether it's reusable for training-time stack sampling).

---

## 5. Tracked follow-ups (carried forward, NOT blocking Step 4)

- **Stale solver6 docstring note #3** ("No DCFR weighting yet") in
  `src/nlhe/solver6.py` — now inaccurate (DCFR exists: `cfr_variant` /
  `dcfr_exponent` / `_dcfr_weights`). One-line doc fix; can ride along with
  Step 4's first commit if convenient.
- **`paired_lifts` / `per_challenger` JSON schema gap** in
  `evals/subgame_ablation_v1_5000.json` — the verdict was applied correctly
  *internally* but those keys serialize empty. The serialization site was **not
  found** in `scripts/` or `src/` during sub-step-6 recon. Worth ~30 min of
  `git grep` when convenient. Not blocking anything.
- **Untracked `pushfold_*.py` files** — predate Session 22, out of scope for the
  fast_view fold-in and the strategy net. If pushfold becomes production-relevant,
  BOTH the fold-in (canonical view/discretize) and the strategy-net inference
  dispatch need to be extended there.
- **`_legal_discrete_bet_sizes`** lives in `actions.py` but is now reached only via
  `fast_view` — code-organization observation, not a bug.
- **Two mid-stream design reversals this session** (6 nets → 1 shared; refuse-on-load
  → two-tier) are recorded in `DECISIONS.md`. Worth being aware: design pivots
  happened twice in one session because recon-first catches gaps *after* an initial
  lock — keep doing recon-first (see §6).

---

## 6. Workflow patterns validated this session (carry forward)

- **Recon-first before any implementation step** caught real issues twice
  (6-nets-redundant; v1-unloadable). Pattern is paying dividends — don't skip it.
- **Bit-identity acceptance gate** via fixed-seed smoke training + tensor-hash
  extraction is the load-bearing safety net for any change in the training hot
  path. Use again for archetype-mix integration (Step 5) if it touches
  `cfr6.traverse_6max`. (Note: Step 4 deliberately changes the trajectory, so it
  uses the convergence + head-to-head gate instead — see §4(e).)
- **`.copy()` on numpy arrays before buffer writes** is non-negotiable (the
  strategy write mirrored the adv-buffer convention).
- **Independent RNG instances for new buffers** prevent unintended perturbation of
  the training trajectory (`strat_rng` = `config.seed + 100`, independent of the
  adv-buffer rng). Apply the same pattern for any future buffer addition.
- **Two-SSH-session + tmux + nohup** is the production long-run pattern;
  **workers=64** is the validated ceiling on this MooseFS filesystem
  (`mfs#euro.runpod.net:9421`; 126 stalls on metadata contention).
- **Idempotent verify-don't-redo** when a prompt re-sends by mistake — saves a
  duplicate commit (happened twice in Session 22).

---

## 7. State snapshot for Session 23

- **Python env:** `/usr/bin/python` (system; no active `.venv` — `source .venv/bin/activate`
  silently no-ops). `torch 2.11.0+cu128`.
- **Device:** ⚠️ **verify at pickup.** This session's training smokes logged
  `device=cuda` (GPU was used for training). The older "Q13: CPU forced for harness
  work (Session 13)" note appears **stale or context-specific** — it did not hold for
  Session-22 training runs. Don't assume CPU-forcing; check `torch.cuda.is_available()`
  and the actual `DeepCFR6MaxSolver` device log line.
- **36 v1 `six_max_*` checkpoints** on disk remain loadable for advantage-net-only
  inference (two-tier load, Step 3). They cannot be deployed with strategy-net policy.
- **Production baseline:** `dcfr-overnight-3000` →
  `runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/ckpt_iter_3000.pt`.
  The v2-era retrain (Step 7) must beat it head-to-head.
- **Bit-identity reference:** `/tmp/fast_view_smoke_pre` persists as the pre-Step-3
  baseline for any future RNG-discipline question (⚠️ `/tmp` — may not survive a host
  restart; regenerate from `/tmp/fast_view_smoke.yaml` at seed 12345 if gone).
- **Smoke config:** `/tmp/fast_view_smoke.yaml` (seed 12345, n_iterations 5,
  checkpoint_every 1) — the reusable fixed-seed fixture for hot-path gates.
