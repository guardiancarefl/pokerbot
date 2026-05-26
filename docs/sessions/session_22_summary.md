# Session 22 — 2026-05-26

Focus: **Run Stage 6-D — the load-bearing strength measurement of the entire B1c
line — and diagnose the alarming sub-case it surfaced.** A 200-hand calibration
against `dcfr-shake-200` fired the Branch-C revert-gate trigger (BR significantly
worse than PROFILE), so the full launch was held for a Path-A diagnostic spike. The
spike identified the mechanism — **H5 "argmax-collapse"** — and cleared the
architecture: PROFILE lifts the blueprint, only the BR leaf-eval mode underperforms,
and that underperformance is regime-dependent rather than a bug. A torch
thread-pinning fix and the session summary land before the full ablation launches.

## What was done

- **Stage 6-D calibration** (n=193 vs `dcfr-shake-200`, M=8, workers=64, base_seed=7).
  Three-way pooled diffs: **blueprint +0.0203** (σ=1.08), **sg-profile +0.0490**
  (σ=1.98), **sg-br +0.0064** (σ=0.25). Lifts: **PROFILE−blueprint +0.0287** (σ=1.41,
  n_opp_positive=1/1); **BR−blueprint −0.0139** (σ=0.60, noise as predicted at 200
  hands); **BR−PROFILE −0.0426** (σ=2.05, **significant — BR worse than PROFILE**).
  The Branch-C trigger ("alarming sub-case") fired; full launch held for diagnosis.
- **Path-A diagnostic spike** (`scripts/subgame_6d_diag_v1.py`, 50 solves, M=128,
  workers=64, wall=**25m55s** / 25.97 h CPU). Side-by-side BR vs PROFILE on
  **identical** subgame trees, M=128 to suppress rollout Monte-Carlo noise; captured
  per-solve leaf values, root policies, convergence tails, and gate decisions.
  Output: `evals/subgame_6d_diag_v1.json`.
- **Diagnostic analysis (2 passes).** Aggregate bimodality scan + a verbatim deep-dive
  on the four genuinely-concerning cases, discriminating H2 (single-pass breakdown)
  and H4 (gate artifact) from the eventual H5 explanation.

## What was decided

- **Ship PROFILE, document BR's regime-dependence.** The architecture (subgame solving
  on the blueprint) works — PROFILE lifts the blueprint by **+0.0287** in calibration.
  The BR leaf-eval mode specifically is the underperformer.
- **The full ablation will route to the PROFILE-favors branch of the locked verdict** —
  this is *anticipated from the calibration + diagnosis*, not a finding deferred to the
  run. The full run measures the effect size; the verdict is then read off mechanically.
- **Two fixes commit before the full run** (torch thread pin + the session summary),
  bundled with the diagnostic artifacts under the `phase4f-league` branch.

## What was learned / measured

- **KEY FINDING — H5 "argmax-collapse".** BR-vs-PROFILE `root_policy_l1` is **bimodal**:
  median **1.9e-05** (≈78% of solves bit-identical root policy), but **9/50 (18%) at
  L1=2.0** — hard one-hot disagreement. In all four genuinely-concerning cases
  (`h33/dec1`, `h45/dec1`, `h36/dec2`, `h30/dec1`): both modes **converge cleanly**
  (conv_tail ≤ 1e-6, `degraded=False`, full 1000 iters); per-leaf differences are
  **tiny** (`leaf_value_max_abs_diff` 0.18–0.34); both root policies collapse to a hard
  one-hot **on different actions**; and BR systematically picks actions the blueprint
  weights low/zero (h30: BR picks CALL where blueprint = [FOLD:0.34, ALLIN:0.66]; h33:
  BR picks CALL where the blueprint is near-uniform). Tail fingerprint: **smaller trees**
  (median 13 vs 27 leaves), **100% preflop**, hero in seats {1,2,3}, gate scores
  **overlap the body** (no gate-regime artifact). **H1 (over-fitting) and H4 (gate
  artifact) ruled out** by the bimodal shape + gate-score overlap.
- **Mechanism.** In close-EV preflop decisions, the small leaf-value perturbations from
  BR's adversarial bias-selection flip the CFR argmax; CFR then amplifies a tiny EV
  difference into a hard one-hot policy. The Brown–Sandholm 2018 mechanism is
  **correctly implemented**, but its theoretical benefit does not manifest in this
  deployment regime (≈98% preflop, small trees, blueprint near-uniform on close calls).
- **Torch thread oversubscription.** Each worker spawned ~65 torch intra-op threads ×
  64 workers ≈ **4160 threads on 128 cores** — contention that inflated the diagnostic
  wall-clock from a ~5–15 min projection to 25m55s. Fix: `torch.set_num_threads(1)` per
  worker init (64 workers × 1 thread → 64 threads, no oversubscription).
- **Wall-clock projection drift.** The M=128 diagnostic ran 25m55s on 64 workers vs the
  5–15 min estimate. The full M=8 production run is ~16× cheaper per leaf, but the scaling
  factor differs; **per-decision cost on the deployment regime must be re-measured from
  the full run, not extrapolated** from the M=128 spike.

## State at close

- **Done: calibration + diagnosis complete; H5 mechanism identified.** Session summary
  written. Torch thread pin applied to `scripts/eval_pool_ablation.py` and
  `scripts/subgame_6d_diag_v1.py`; test suite re-verified green
  (`tests/test_eval_pool_ablation.py` + `tests/test_ablation_decision_level.py`).
- **Full Stage 6-D ablation: _pending launch_** (5000 hands × 5 opp × 3 challengers,
  workers=126, base_seed=2026 → `evals/subgame_ablation_v1_5000.json`). _Status updated
  on launch/completion in Step D._
- **Carry-forward (post-verdict cleanup):** the `torch.set_num_threads(1)` fix lands in
  `scripts/eval_pool_ablation.py` so the Stage 6-D run incorporates it; decision-time
  complexity recommendation **ship PROFILE, document BR's regime-dependence**.

## Next session opens with

**The full Stage 6-D verdict.** Read the canonical verdict block from harness stdout,
record pooled diffs + the three lifts (with σ) + per-opponent breakdown, and apply the
locked four-branch criterion mechanically. Expected route: **PROFILE-favors** (PROFILE
> BR statistically significant; PROFILE > blueprint at σ ≥ 1.5). This is the first
measured strength delta from subgame solving — the milestone the whole B1c line has
built toward.
