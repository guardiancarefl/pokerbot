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

## Stage 6-D — full ablation verdict

- **Wall-clock: 19h29m at workers=64** (after the 126-way MooseFS metadata stall;
  see Findings). 75,000 hands total (5000 × 5 opponents × 3 challengers, base_seed=2026).
- **Verdict: `PASS_BR_EQUIVALENT_TO_PROFILE`** — read off mechanically per
  `docs/SUBSTEP_6_DESIGN.md` Decision 6.3 (locked pre-data).
- **Pooled diffs:** blueprint **+0.0331**, sg-profile **+0.0427**, sg-br **+0.0418**.
- **Lifts:** **L(BR−blueprint) = +0.0087 σ=4.56** (subgame solving demonstrably lifts
  strength); **L(BR−PROFILE) = −0.0009 σ=0.56** (BR not statistically distinguishable
  from PROFILE).
- **n_opp_positive = 5/5** — every opponent positive, per-matchup σ ranging **4.95–11.08**.
- **ordering_ok = False** — sub-clinical: PROFILE pooled 0.0009 *above* BR, but the
  BR-vs-PROFILE comparison is only σ=0.56, well under the 1.5σ bar. Consistent with the
  two being indistinguishable; the flag is an artifact of the pooled point estimates, not
  a real ordering signal.
- **Recommendation: ship PROFILE_SAMPLE leaf-eval for production; retire BR at decision
  time** (BR code retained for future regime testing).
- **Diagnostic vs full-run reconciliation:** the H5 argmax-collapse mechanism is **real**
  but **smaller in net magnitude** than the 200-hand calibration suggested. Calibration
  read BR−PROFILE at σ=2.05 (n=193, single opponent — variance-inflated); at full power
  (n≈4830/matchup, 5-opponent pool) the effect averages to **noise, σ=0.56**. The
  mechanism flips individual close-EV preflop argmaxes but nets to zero across the pool.

### Findings

- **MooseFS metadata-contention ceiling.** workers=126 stalls — ~756 concurrent file
  opens (126 workers × 6 checkpoints) against the MooseFS master saturate the metadata
  server; workers=64 (~384 concurrent opens) runs cleanly. Filesystem
  `mfs#euro.runpod.net:9421`. The concrete ceiling lives somewhere between 64 and 126;
  future harness runs at >64 workers on this filesystem need either an in-main
  checkpoint-load-and-broadcast refactor (load once, broadcast to workers) or empirical
  worker-count tuning. workers=64 is the validated configuration for now.
- **Workflow trap — `pkill -f` self-match.** `pkill -f "eval_pool_ablation"` matches its
  own command line and SIGKILLs the wrapper shell. Use `pgrep -P <parent-PID>` identification
  instead — never string-pattern kill on the eval command name.
- **Schema gaps in `subgame_ablation_v1_5000.json`.** `paired_lifts` and `per_challenger`
  blocks are empty though the verdict applied correctly internally. Worth populating for
  downstream analysis (separate cleanup).
- **Recurring heartbeat log not persisted.** `/tmp/heartbeat_recurring.log` never landed
  on disk. Re-arming the heartbeat needs `--output` redirect verification before the next
  long run.
- **Wall-clock projection drift.** The 2–5 hour estimate became 19h29m actual.
  Calibration-naive linear extrapolation was systematically optimistic: calibration
  sampled preflop-heavy decisions, while the full run hit the real-game postflop
  distribution where solves are more expensive (larger trees, deeper rollouts). Future
  runs should benchmark on a real-distribution decision mix before projecting wall-clock.
