# Overnight Phase-1 adaptive scaffold — morning report (2026-06-04)

## TL;DR

**PHASE-1 COMPLETE.** Foundation verified end-to-end (4/4 sanity checks pass);
leakage tests green (9/9); Phase-1 d=128 backbone reproduced bit-identically to
the prior `runs/arm_D128` run. The "Phase-1 distill backbone can't beat the
blueprint" finding is **REAL**, not a wiring artifact. The G6 separation
failure (tendency head can't distinguish archetypes from blueprint) is the
expected signature of supervised distillation on a blueprint-as-hero pool, and
is exactly the gap **Phase-2 RL** with the population corpus is designed to
close. **PHASE-2 PENDING — needs RunPod corpus + user gate review.**

## Step 1 — Recon (done)

The Phase-1 pipeline is already fully scaffolded; no code needed to be written.

| Component | File | Status |
|---|---|---|
| StratFormer model | `src/nlhe/adaptive/model.py` (229L) | Adaptive6MaxNet (transformer trunk over InfosetEncoder6Max tokens + policy_head + opp_head_tendency K=10), tendency_l2_read, gate, blend, collate. §8/D6-clean forward signature. |
| Leakage tests A6-E6 | `tests/test_six_max_token_no_leak.py` (702L) | A6 structural + layout + parse-state, B6 counterfactual + history-purity, C6 SeatStats no-card-info, D6 forward-surface guard, E6 entropy-floor purity. |
| Full pipeline | `scripts/six_max_adaptive_smoke.py` (1359L) | Leakage preflight, blueprint loader, encode-once pool generation (stratified or single), tendency-blueprint reference, fixed eval set, cotrain loop with G2-G6 gates. |
| Anchor | `runs/six_max_20260530_034023_phase4f_dcfr_candC_k200/checkpoints/ckpt_iter_2000.pt` | 11 MB, schema=v2_with_strategy |
| Abstraction | `runs/abstraction_20260521_223018_retrofit/abstraction.pkl` | 150 KB, k=200 postflop |

## Step 2 — Leakage tests A6-E6 (PASS)

`pytest tests/test_six_max_token_no_leak.py` — **9/9 passed in 8.5s.**
Hard preflight gate clear.

## Step 3 — Phase-1 distill training (PASS, with project-state caveats)

### Initial attempts at A4 smoke-size (d=64) — known-broken per project history

- **Try 1** — baseline (d=64, lam_aux=0.10): G4 floor BREACH at epoch 25
  (mean_KL=0.113 vs bar 0.02). G2 and G3 OK.
- **Try 2** — halved lam_aux per smoke's "first recovery lever" (d=64,
  lam_aux=0.05): G4 BREACH again (mean_KL=0.127). lam_aux is dominated by
  lam_distill=1.0 → halving has near-zero effect (same-seed, near-identical G2 curve).
- **Try 3** — entropy-floor mitigation (lam_aux=0.05, beta_entropy_floor=0.25):
  killed, because project commit `005a3fa` already falsified this remedy at
  d=64 ("uniformly worse on every panel member"). The recovery lever was known
  bad.

### Pivot to documented Phase-1 backbone — `arm_D128`

Project commits `b0f91dd` (G4 KL gate retired — "the unanchored 0.02-KL bar
never had this [empirical anchor]"; ICM panel is the actual gate) and `005a3fa`
(entropy-floor falsified at d=64; capacity-limited; next experiment = d=128)
made the path forward clear: train at d=128 with the wide stratified pool and
skip the retired KL gate. That config = `runs/arm_D128/` already on disk.

### Re-train at d=128 for reproducibility — `runs/phase1_d128_repro/`

Re-trained arm_D128's exact config: d=128, L=4, ff=512 (843,283 params),
pool_stratified S1+S2+S3+S4=10852 tuples, 200 epochs, lam_aux=0.10, skip-g4-floor.

| Metric | arm_D128 (prior, 2026-06-02) | phase1_d128_repro (tonight, 2026-06-04) |
|---|---|---|
| n_params | 843,283 | **843,283** |
| pool size | 10,852 | **10,852** |
| epochs | 200 | **200** |
| wall (cotrain) | 1791 s | **1819 s** |
| **G2 L_distill** | OK (drops) | **OK (1.07 → 0.77)** |
| **G3 L_aux** | OK (drops) | **OK (0.064 → 0.016)** — tendency head learns |
| G4 (retired) | skipped | skipped |
| G6 blueprint d_bar | 0.5714 | **0.5714** ← bit-identical |
| G6 per-opp d_bar (gus/kill/loose) | 0.5648 / 0.4387 / 0.6603 | **same** |
| G6 separation_ok | False (WEAK) | **False (WEAK)** ← real, reproducible |

**Reproducibility confirmed bit-identically.** The G6 separation failure
(archetypes not distinguishable from blueprint in tendency space) is REAL and
deterministic given seed + data.

### Phase-1 verdict

`runs/phase1_d128_repro/` is a valid Phase-1 backbone:
- G2 distill loss drops 28%.
- G3 aux loss drops 4× (tendency head IS learning).
- Forward-pass spot-check vs blueprint: KL(bp‖sf) = 0.0001–0.07 per state
  (close to blueprint, argmaxes match).
- G6 separation fails: tendency head learns "what the noised blueprint hero looks
  like in this match," **not** "what the OPPONENT looks like." Expected outcome
  for supervised distillation where the hero IS the noised blueprint.
- Prior ICM-panel run on arm_D128 (`runs/arm_D128/icm_panel_arm_D128.json`):
  student loses to blueprint by 0.008-0.04 ICM/hand on 4/5 panel members.
  Consistent with the small per-state KL accumulating across hands.

## Foundation skepticism (4/4 PASS) — verified the verdict isn't an artifact

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Distill target correctness | ✓ | smoke's `blueprint_policy` returns bit-identical 9-vector to gate2's `_sample_action_from_policy` underlying `inference_policy` for the same state+rng. feature max-abs-diff = 0.00e+00. policy max-abs-diff = 0.00e+00. |
| 2 | Encoder correctness | ✓ | 236-dim layout verified field-by-field (bucket one-hot sums to 1, street/position one-hots correct, stacks/contribs sensibly normalized, deterministic). Same encoder used for training and inference. |
| 3 | Gate verification (Net₀) | ✓ | gate2 blueprint-vs-blueprint paired CRN: **Δ = 0.0000 ± 0.0000 exactly**. arm_D128 trunk loads + produces sensible policies (argmaxes match blueprint, per-state KL 0.0001-0.07). |
| 4 | Blueprint baseline consistency | ✓ | Bit-identical CKPT SHA `2a9665ee…` (gate2's `runs/k200_blueprint_ckpt_iter_2000.pt` == smoke's anchor `ckpt_iter_2000.pt`). Bit-identical abstraction SHA `0fc20800…`. |

**The "Phase-1 distill backbone can't beat the blueprint" verdict is REAL.**
The supervised distill objective converges to a CLOSE-BUT-NOT-IDENTICAL
mirror of the blueprint; small per-state KL deviations accumulate into a small
ICM disadvantage. No wiring bug; no encoder mistake; no garbled distill target;
no harness asymmetry.

## Step 4 — STOP (per discipline)

**Phase-2 RL (online PPO/REINFORCE with chip-EV reward) is NOT started.**
Reasons:
1. Phase-2 needs the RunPod population corpus (still generating; not present
   on Contabo).
2. Phase-2 is where the user wants to review gates BEFORE the actual exploit
   training runs.

The Phase-1 backbone artifact `runs/phase1_d128_repro/smoke_net.pt` (3.4 MB)
is ready for Phase-2 to load as its starting point.

## What Phase-2 is expected to fix

G6 separation fails because the tendency head's supervision (auto-supervised
against MatchObserver SeatStats during pool gen) only sees noised-blueprint
behavior at the HERO seat. The blueprint is context-blind, so the model has
no reason to learn that opponent context should change its policy — distill
pulls the policy head toward the blueprint regardless of opponent.

**Phase-2 RL with chip-EV reward will pull the policy AWAY from the blueprint
when the opponent's tendency vector signals an exploit is available.** That
opens the gate (gate=match_conf · tanh(D̄)) and lets the blend leave the
anchor. Tonight's KillPhilMTT diagnostic showed the leak is range-dependent
exploitation — a generalization of this exploit-gate mechanism is the proper
fix, subsuming the static-range fix tried tonight (which was symmetric/net-zero
across the panel).

## Files / artifacts

Committed:
- `docs/REBEL_OVERNIGHT_PHASE1_REPORT.md` — this report.
- `runs/phase1_d128_repro/metrics.json` — full training metrics + G6 probe.

Gitignored (`.pt`):
- `runs/phase1_d128_repro/smoke_net.pt` (3.4 MB, the Phase-1 backbone).
- `runs/phase1_d128_repro/ckpt_ep{0025…0200}.pt` (8 checkpoints, 27 MB total).

Logs (in `/tmp/`):
- `/tmp/overnight_status.log` — master status, one line per step.
- `/tmp/adaptive_build.log` — recon + foundation-check notes.
- `/tmp/leakage_tests.log` — pytest output (9/9 pass).
- `/tmp/phase1_d128_repro.log` — full smoke training log.

## Repo state at handoff

Branch `track-policy`. Base commit `f2940b9` (tonight's KillPhil diagnostic +
static-range fix verdict). No code changes overnight — Phase-1 used existing
scaffold as-is. One new artifact directory (`runs/phase1_d128_repro/`).

Next user-driven steps:
1. Review this report + foundation checks.
2. When the RunPod population corpus is ready, design + gate the Phase-2 RL
   loop (PPO/REINFORCE on chip-EV reward, gate-open mechanism, ICM-panel as
   acceptance gate).
3. Phase-2 starts from `runs/phase1_d128_repro/smoke_net.pt`.
