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
- ✓ **Step 4 — Stack-distribution sampling** — discovered already in production (commit `4739c1a`, 2026-05-23). See the DECISIONS.md audit entry dated 2026-05-27.
- ✓ **Step 5 — Archetype mix for 6-max** (`d093abd` + `470beb7` + `48b5eae` + Phase 5-C closure commit) — wrap-not-port `ArchetypePolicy` + `ArchetypePool`, three-way combined sampler, bit-identity-validated at default. See the DECISIONS.md Step 5 design-decisions entry.
- → **Step 7 — Multi-day production blueprint training** (24–72h GPU) — the capstone. **ONLY genuinely-open implementation work remaining.** See §4.
- **Step 8+ — Within-match observation framework (Layer 4), Slumbot HUNL bridge, 42 shanky bots integration evaluation.**

**Note:** Step 6 (league play) is folded into Step 7 — `league_mix` is a config knob, no longer a separate step.

**6 of 7 core steps complete (Steps 1–5 + the Step-6-into-7 fold). Genuinely-open implementation work: Step 7 only.**

---

## 3.5 Naming clarification: "v2" is overloaded

- **"League v2"** = the second league config (vs league v1). Currently means
  `configs/six_max_phase4f_dcfr_league_v2.yaml`, the 900-iter shakedown's config.
- **"v2-schema" / "v2_with_strategy"** = the strategy-net checkpoint schema
  introduced in commit `78917f8` (Session 22).

These are unrelated. The existing league-v2 shakedown is a v1-schema run (it predates
the strategy net by 3 days). Step 7's "produces v2-schema checkpoints with league
play" means strategy-net schema + league mechanism, not "league v3."

**Step 5 closure (2026-05-27):** v2-schema is now fully capable. Strategy net
(`v2_with_strategy` checkpoint schema), DCFR weighting (`cfr_variant='linear'`),
stack-distribution sampling (`tournament_structure_path`), archetype mix
(`archetype_mix` + `archetype_calibration_path`), and league play (`league_mix` +
`league_registry_path`) are all wired, unit-tested, and bit-identity-validated at
their respective zero/default settings. Step 7's production training run exercises
the full stack.

---

## 4. Step 7 — multi-day production blueprint training (the capstone)

**Goal:** train a v2-schema blueprint at production scale, with the full diversity
stack active (self-play + archetype mix + league play). This is the bot whose strength
claim Scenario 3 has been building toward.

**Open design questions for Step 7's recon:**

- **(a) Training duration target:** 24h / 48h / 72h. Trades off compute cost vs
  convergence quality. The `dcfr-overnight-3000` baseline ran ~12h.
- **(b) Configuration choices:** `archetype_mix` value (0.2–0.7 recommended per
  `DECISIONS.md:216`), `league_mix` value (`PHASE4F_LEAGUE_V1_FINDINGS.md` suggested
  0.15), profile subset (all 5 or weighted), league registry anchors (peak vs
  peak+archetype tags).
- **(c) Checkpoint cadence:** every N iterations. `dcfr-overnight-3000` used
  `checkpoint_every=300`. Tradeoffs: smaller N = more granular eval-pool members, more
  storage; larger N = faster training, fewer checkpoints.
- **(d) Validation gates during the run:** heartbeat logging, periodic eval against
  `dcfr-overnight-3000` (the v1 baseline the v2 run must beat), strat-loss monitoring.
- **(e) Compute provisioning:** GPU type (RTX 4090 vs H100), worker count (`workers=64`
  confirmed safe on this MooseFS filesystem), nohup + tmux pattern.
- **(f) Acceptance:** head-to-head vs `dcfr-overnight-3000` in the eval pool (target:
  positive lift at σ > 2.0). What other measurements (Slumbot bridge if available,
  Shanky pool comparison)?

**Recon-first opener:** read `scripts/train_6max.py` for the runtime API,
`configs/six_max_phase4f_dcfr_linear_overnight.yaml` for the baseline config shape, and
`PHASE4F_LEAGUE_V1_FINDINGS.md` + the sub-step 6 verdict JSON for the strength
references the v2 run must clear.

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
  `cfr6.traverse_6max`. (Note: Some Scenario 3 steps deliberately change the
  training trajectory by design — e.g. Step 5's archetype-mix path. For those,
  the bit-identity gate applies only to the pure-self-play branch where
  archetype_mix=0.0 is bit-equivalent to the prior implementation; the archetype
  path itself uses a functional gate. See §4(e) for the Step 5 specific
  acceptance gate.)
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
