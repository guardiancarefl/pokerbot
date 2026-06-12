# ICM Correction Fit + Consumer Re-Price Audit — Registration (cycle-2 item #2)

**Status: DRAFT — pending manager review. Nothing here has been run.**
Written 2026-06-12, triggered by the c1 MATERIAL verdict
(`evals/c1_icm_gap_20260612/TIER1_REPORT.txt`, design
`docs/research_program/ICM_GAP_STUDY_DESIGN.md` §6 decision matrix,
actions 1–4). Once approved this document is binding; deviations are
logged in `EXPERIMENT_LOG.md` before results are looked at. All
probability units; 0.01 prob = 0.02 buy-in-equity.

## 0. What c1 measured (inputs to this spec)

Tier-1 (n_alive=4, levels 3–4, CV tertiles; N=250 states × M=100
rollouts per cell; 750/750 states scored, caps=0, taints=0):

| cell | disp | d_short mean | Bonferroni 98.33% CI | σ_bias |
|---|---|---|---|---|
| t1 | low CV | −0.0034 | [−0.0131, +0.0062] | 0.0405 |
| t2 | mid CV | +0.0148 | [+0.0051, +0.0245] | 0.0407 |
| t3 | high CV | **+0.0412** | [+0.0286, +0.0539] | **0.0683** |

Structure (results.json secondary maps): the error loads on
micro-stacks — depth band [0,5)bb mean d = **+0.0652** in t3 (n=148)
and +0.0529 in t2 (n=21); [10,20)bb ≈ −0.0205 (t3); 20+bb ≈ −0.009.
By stack rank (t3): rank1 +0.041, rank2 −0.023, rank3 −0.022,
rank4 (leader) +0.003 — the mid stacks pay for the short stack's extra
survival; sum over ranks ≈ 0 (conservation: Σᵢ dᵢ(s) = 0 exactly per
state, since itm_counts sum to 3 and q sums to 3). t1 (low CV) is
clean: bias ≈ 0 everywhere. σ_bias(t3)=0.068 > 0.05 — the error is
also HETEROGENEOUS: a mean-level correction cannot remove all of it;
the residual σ_bias is itself a registered deliverable (§2.5).

Scope of the evidence: **Tier-1 covers ONLY n_alive=4, levels 3–4.**
n_alive and level are constants in this data — they are NOT
identifiable as fit features. Correction v1 is therefore
**bubble-cell-scoped**: applied only when `n_alive == 4`; identity
elsewhere. Extension to 4-alive other levels / 5- and 6-alive needs
Tier-2 (already licensed by the parent design, not run; separate
registration).

## 1. Correction model

### 1.1 Fit data (existing — zero new rollouts)

`evals/c1_icm_gap_20260612/tier1_t{1,2,3}.jsonl`: 750 states, each row
has `stacks, level, dealer, itm_counts[6], M_eff`. Per alive seat i:
`p̂ᵢ = itm_counts[i]/M_eff`, `qᵢ = icm_equity(stacks, [2,2,2],
eligible=alive)[i] / 2.0` (recomputed at fit time; rows don't store q),
residual `dᵢ = p̂ᵢ − qᵢ`. That is **3,000 seat-level residuals from 750
states**, with exact within-state sum-zero and cross-seat dependence —
all SEs are state-clustered; never treat seats as independent.

### 1.2 Candidates

- **M1 (chosen): per-seat additive correction with conservation
  projection.** `Δᵢ = g(depth_bbᵢ) · h(CV(s))`, where `depth_bbᵢ` =
  seat i's stack in current big blinds, `g` = continuous piecewise-linear
  with knots at the registered band edges {0, 5, 10, 20} (4–5 params),
  `h` = linear in alive-stack CV, anchored so h→0 as CV→CV(t1 median)
  ≈ 0.27 region (t1 says no correction at low CV; exact anchor form
  chosen at fit time and logged). Then per state: `p_corr,i =
  clip(qᵢ + Δᵢ, 0, 1)`, renormalized over alive seats to Σ p_corr = 3
  (proportional shrink of the corrections, not of p, so seats with
  Δ=0 move minimally). Fit: weighted least squares on the 3,000
  residuals, weights `1/(p̂(1−p̂)/M + 1e−4)`, state-clustered SEs.
  ≤ 10 parameters for 750 independent states. Monotone in q is free:
  Δ does not depend on q, so ∂p_corr/∂q = 1 > 0 (the design's
  monotonicity requirement holds by construction).
- **M2 (alternative, rejected as primary): isotonic regression of p̂ on
  q within depth-band × CV-tertile bins.** Rejected because (a) bins go
  thin exactly where the signal is (t2 [0,5)bb has n=21); (b) no
  smooth extrapolation off-support; (c) piecewise-constant output makes
  step discontinuities in ICM *differences* — and the most exposed
  consumers (battery oracle, resolver leaves) consume gradients between
  nearby stack vectors, so steps are actively harmful; (d) per-bin
  isotonic does not conserve Σp = 3. Kept as the registered fallback if
  M1 fails the holdout gate.
- **M3 (alternative, rejected): logistic/beta calibration**
  `logit(p) = a·logit(q) + b + c·features`. Rejected: the measured
  error is additive in probability and loads on depth, not on the q
  level; the logit link misbehaves as q→1 (chip leader), where the
  measured bias is ≈ 0 (+0.003).
- **M4 (degenerate baseline, reported not deployed): rank × CV-tertile
  lookup of cell means.** Used only as a floor for holdout comparison.

Model selection within M1 (nested: depth-only → depth×CV → + rank
dummies): pick the simplest variant within 1 SE of the best holdout
weighted MSE. Variant set frozen here; no other features may be added
post-hoc.

### 1.3 Fit/holdout split (existing Tier-1 data only)

Stratified 80/20 **by state** within each cell: 200 fit / 50 holdout
states per cell (600/150 total), drawn by `random.Random(20260613)`
over sorted `state_idx`. This implements the parent design §9.4
("hold out 20% of Tier-1 states"). Holdout is opened once, after the
M1-variant choice is frozen on fit-set cross-validation.

**Holdout gate (must pass before any validation rollouts are spent):**
- G1: holdout-t3 |mean corrected d_short| ≤ 0.5 × |raw| (raw on the
  same 50 states; target ≤ ~0.0206).
- G2: holdout-t1 mean corrected d_short ∈ (−0.01, +0.01).
- G3: holdout [0,5)bb pooled corrected mean ≤ +0.03 (from +0.065).
Fail any → fall back to M2; if M2 also fails, NO-ADOPT and escalate
(the bias is not capturable by these features; Tier-2 + rethink).

### 1.4 Artifact

`data/icm_correction_v1.json`: knots, coefficients, CV anchor, scope
predicate (`n_alive==4` only), fit-set hashes, seed, git head. Applied
by a NEW module (proposed `src/nlhe/icm_correction.py`,
`corrected_itm_probs(stacks, payouts, eligible) -> list[float]` plus an
equity wrapper). **`src/nlhe/icm.py` is not edited** — MH stays frozen
and tests keep pinning it; the correction is a separate opt-in layer
behind explicit flags at each consumer (§3/§4). Nothing in this spec
writes code yet; build happens after manager approval.

## 2. Validation pre-registration (the only new compute)

### 2.1 Fresh states

Sampled from the same frame (`data/training_dist_v1.json.gz`,
sha f87aaab2…), distinct-tuple rule, same cell definitions and tertile
cuts as Tier-1, **excluding all 750 Tier-1 tuples** (fit and holdout
both — the holdout was opened in §1.3, so it can no longer serve as
confirmation). Seed 20260614 for state sampling; rollout CRN seed =
sha256-hash `(2026, cell, "val", state_idx, m)` (disjoint from Tier-1's
stream). Same instrument as c1: `scripts/icm_gap_probe.py` rollout
semantics, corrected ITM definition (D2), champion ckpt b79e82dd…,
M=100, caps/taints rules unchanged.

Cells: fresh t3 (N=150), fresh t1 (N=150), micro-depth oversample
V3 = 100 high-CV bubble states filtered to shortest-stack depth < 5bb
(sampled from the t2∪t3 CV range), M=100.

### 2.2 Bars (frozen now)

- **V1 (primary, t3):** corrected |mean d_short| ≤ 0.5 × raw |mean
  d_short| measured on the SAME fresh states, AND corrected mean CI95
  contains 0 or |mean| ≤ 0.0206. (Raw fresh mean also reported as a
  replication check against +0.0412.)
- **V2 (no-degradation, t1):** corrected mean CI95 ⊂ (−0.015, +0.015)
  AND |corrected mean| ≤ |raw mean| + 0.005. The correction must be a
  near-no-op where MH was already right.
- **V3 (micro-depth):** corrected pooled mean over depth<5bb seats
  ≤ +0.03 (raw ≈ +0.065).
- All three pass → **ADOPT**. V2 fails alone → one pre-registered
  retry: shrink the correction globally by λ ∈ (0,1) chosen on the fit
  set, re-run V1+V2 on the same fresh states (no new sampling); else
  NO-ADOPT. V1 or V3 fail → NO-ADOPT, report, escalate to Tier-2.
  No other re-rolls, no bar shopping.

### 2.3 Cost

Measured rate (Tier-0 rate_benchmark.json, exactly this state class):
**0.065 s/rollout**. Validation = (150+150+100) states × 100 rollouts
= 40,000 rollouts ≈ 2,600 s ≈ **0.72 core-h** (≈1.4 core-h at 2× safety
margin; trivially night-shift schedulable, nice'd). Fit + holdout:
~zero compute (<1 min, 3,000 weighted residuals, ≤10 params).

### 2.4 Deliverables

`evals/c1_icm_correction_<date>/` JSONLs + run headers (e2 style);
verdict appended to `EXPERIMENT_LOG.md`; on ADOPT:
`data/icm_correction_v1.json` committed + DECISIONS.md entry;
RESEARCH_MAP c1 updated.

### 2.5 Heterogeneity honesty clause

σ_bias(t3) = 0.068. The correction removes the feature-explained mean;
report the holdout/validation **residual σ_bias** and the fraction of
σ²_bias explained (no hard bar — exploratory). Whatever remains bounds
the per-decision accuracy any corrected oracle label can claim; this
number goes verbatim into any future battery re-label report.

## 3. Consumer re-price audit (exhaustive)

Grep basis (2026-06-12, this audit): every in-repo call site of
`icm_equity` / `icm_adjust_returns` / `icm_adjust_trajectory` /
`icm_equity_normalized` outside docstrings.

**Direct `icm_equity` call sites**
1. `src/nlhe/icm_returns.py:76–77` — core of `icm_adjust_returns` (all
   of group B inherits).
2. `src/nlhe/icm.py:163` — inside `icm_equity_normalized` (consumers:
   tests only).
3. `src/nlhe/subgame_leaf.py:413–414` — resolver Option-A leaf diff.
4. `scripts/fold_vs_shove_battery.py:273` (via `oracle_ev` :210,:220)
   — H2 battery oracle.
5. `scripts/sng_baseline.py:226,236` (stage_acc deltas), `:257`
   (capped-game hero_net).
6. `scripts/tg4_verdict.py:73` — TG4 bubble-battery start_eq.
7. `scripts/f1_ensemble_probe.py:266` — ensemble-probe baseline eq.
8. `scripts/rebel_gate2.py:418` — resolver gate diagnostics.
9. `scripts/icm_gap_probe.py:283` — the c1 instrument's q definition.

**`icm_adjust_returns` consumers**
10. `src/nlhe/cfr6.py:266` (import :85; applied via `solver6.py`) —
    training terminal utility.
11. `src/nlhe/subgame_solver.py:363,404,554` — resolver terminals.
12. `src/nlhe/subgame_leaf.py:350` — resolver leaf terminal.
13. `scripts/eval_icm_panel_floor.py:176` — H1/floor panel pricing.
14. `scripts/eval_6max_self_play.py:272` — single-hand A/B harness.
15. `scripts/eval_pool.py:164`; 16. `scripts/eval_pool_ablation.py:129`;
17. `scripts/measure_layer4_cheap.py:127`; 18. `scripts/rebel_gate2.py:523`;
19. `scripts/rebel_diag_killphil.py:175`;
20. `scripts/ablation_decision_level.py:311`.
21. `src/nlhe/icm_returns.py:92` — `icm_adjust_trajectory` wrapper
    (callers of the trajectory path inherit via cfr6).
22. `src/nlhe/mini_eval.py` — indirect (consumes icm deltas produced
    upstream; no direct call).
23. Tests: `tests/test_icm.py`, `test_icm_returns.py`,
    `test_cfr6.py:217,236`, `test_subgame_solver.py:270`,
    `test_subgame_leaf.py` — pin MH semantics; untouched (correction is
    a separate layer).

**Verified NON-consumers (the live path):** `scripts/run_live_dryrun.py`,
`src/nlhe/within_match.py`, `src/nlhe/layer4_factory.py`,
`src/nlhe/adaptive/*` — zero icm call sites. Also NOT MH: the TG3
chain (`scripts/tail_floor_ab.py:188` ← `short_stack_floor_ab.py:379`)
scores via `cfr6.compute_icm_payouts` (cfr6.py:181) — exact terminal
cash/bust ±1, despite the name. See row T3 below.

### Per-consumer checklist (dispersion regime / sensitivity / cheap re-check / pre-committed expectation)

| consumer | regime | sensitivity | cheap re-check | pre-committed expectation |
|---|---|---|---|---|
| **H2 battery oracle** (#4; `evals/h2_battery/battery_v1.json`, 8,112 spots) | start = **6-alive equal stacks** 5–15bb L3/5/7; counterfactuals = 6-alive dispersed or 5-alive after bust | **Zero under v1** — n_alive∈{5,6} is outside Tier-1 support; correction v1 is identity there. The decision-relevant quantity is the ICM *gradient* at micro-depth, which Tier-2b (not run) prices | Run the re-labeler anyway (pure math, minutes, 0 rollouts) and **assert 0/8112 label flips** as a scope-consistency gate | 0 flips. Honest note: TIER1_REPORT action (3) "re-label the battery" overstates what Tier-1 alone licenses — the real re-label is contingent on Tier-2 (n_alive 5/6 cells) + the Tier-2b flip rate; queue both |
| **TG4 verdicts** (#6; `reports/EXP_H1_tail_floor.md` §7) | bubble battery start_eq = MH on 4-alive states from the same bubble artifact — exactly the measured high-bias regime | **None on the verdict**: start_eq is subtracted identically in both CRN-paired arms (asserted seed schedule), so the kill-bar delta cancels it exactly; and hero is uniform over alive seats where Σᵢbᵢ(s)=0 exactly, so even the level means are seat-averaged-unbiased | Re-read only (this row is the re-read) | TG4 verdict stands; no re-run. The bubble paired z=−1.96 is borderline for sample-size reasons, not ICM-bias reasons |
| **TG3 24k gate** (T3; tail_floor_ab → short_stack_floor_ab → `compute_icm_payouts`) | full games, symmetric start → terminal ≤3 alive | **Not an MH consumer.** Per-game hero_icm is exact cash/bust scoring; with payouts [2,2,2] the ≤3-alive terminal is exact and the symmetric start is exact, so completed games carry zero MH exposure. Capped games (>3 alive at max_hands) use the same non-MH rule in both arms (a separate, CRN-cancelled overpay artifact — footnote only) | Re-read only (this row) | TG3 PASS (+0.1227 ± 0.0073) stands untouched. The brief's worry "TG3 used ICM deltas" resolves cleanly — and the equal-stack regime it lives in is t1-clean anyway |
| **H1 floor panel** (#13, `eval_icm_panel_floor.py`) | sampled tournament states incl. 4-alive high-CV; paired student-vs-blueprint per-hand MH deltas | Level bias cancels on non-diverged hands (delta ≡ 0); diverged hands are **gradient-priced** — exposed up to ~2×0.065 equity on micro-stack bust/double events | Offline re-price of the stored per-hand records' 4-alive subset with the corrected map; recompute pooled σ per member. 0 rollouts, minutes | Floor sign-off read (|σ|<2 every member) expected to hold; if any member crosses 2σ under re-price, STOP and escalate to manager — this is decision-matrix action (4) |
| **sng_baseline** (#5) | capped-game hero_net: any; stage_acc: all stages | Capped-game frequency measured **0** in 16k e2 games → hero_net exposure ≈ nil. stage_acc is diagnostic-only; bubble rows shift ≤ the measured bias | `grep` capped counts in existing e2/baseline JSONLs (expect 0); no re-score | **Zero diffs** on all existing headline hero_net numbers. stage_acc re-derived only if a future analysis leans on its bubble levels |
| **eval_pool / eval_pool_ablation / eval_6max_self_play / measure_layer4_cheap / rebel_gate2 / rebel_diag_killphil / ablation_decision_level / f1_ensemble_probe** (#7,8,14–20) | per-hand MH deltas, mixed regimes | All are comparative A-vs-B on shared/CRN states: level bias largely cancels; gradient bias does not, but every verdict they produced is closed and comparative | None retroactive. Forward rule: any NEW gate run on these harnesses after adoption runs with the correction flag and reports both numbers | No historical verdict is re-issued from this group |
| **Resolver** (#3,11,12; + indirect `eval_resolver_vs_shanky.py`) | leaf/terminal one-hand deltas at the bubble — micro-stack gradient error distorts resolve targets directly | HIGH for future resolver work; **not in the live path** | None now (ReBeL track gated separately). Precondition added: the correction flag must reach subgame_leaf/subgame_solver before any future resolver gate is run | — |
| **Training** (#10) | every regret in any retrain | TOTAL, but frozen: the deployed champion was trained on MH and is not retroactively wrong-er | Build the flag (default OFF) when the apply layer lands | Flag flips ON only in the next league/blueprint retrain config — **pod-relevant** (RunPod), decision-matrix action (2), HIGH-EVoI branch; queue in COMPUTE_QUEUE |
| **c1 instrument** (#9) | — | — | — | **Never corrected** — it defines q as MH; correcting it would make the study circular |

## 4. What does NOT get touched

1. **Live path** — `run_live_dryrun.py`, listener, Layer-4/within-match:
   zero icm call sites (verified above). No behavior change is possible
   from this work; tail-floor arming (OQ-2) and all operator gates
   unchanged.
2. **`src/nlhe/icm.py`** — MH math frozen; tests keep pinning it. The
   correction is a separate artifact + opt-in apply layer.
3. **Training returns** (`cfr6.py:266`) — can only change with a
   retrain; pod-relevant. This spec builds the flag OFF; flipping it is
   a separate, manager-gated retrain decision.
4. **Historical verdicts and baselines** (e2 rebaseline, H1 TG1–TG4,
   H2 CLOSED-FAIL, pool yardsticks) — never rewritten; any re-price is
   filed as an addendum showing both numbers. The battery stays the H2
   "permanent panel instrument" with its v1 labels until the
   Tier-2-licensed re-label.
5. **The c1 probe's q definition** (§3 last row).

## 5. Decision matrix + costs

| Stage | Outcome | Action |
|---|---|---|
| Holdout gate (§1.3) | any of G1–G3 fail for M1 | fallback M2; both fail → NO-ADOPT, escalate, Tier-2 rethink. **No validation rollouts spent** |
| Fresh validation (§2.2) | V1+V2+V3 pass | **ADOPT** `data/icm_correction_v1.json` for offline consumers behind flags; run the §3 checklist top-to-bottom; queue Tier-2 + Tier-2b registration (battery re-label path) and the retrain-flag pod item |
| | V2 fails alone | one λ-shrink retry on the same states; else NO-ADOPT |
| | V1 or V3 fail | NO-ADOPT; report; Tier-2 escalation |

Costs (measured 0.065 s/rollout, Tier-0 benchmark):

| item | rollouts | core-h |
|---|---|---|
| fit + holdout (existing 750 states) | 0 | ~0 (<1 min) |
| battery 0-flip assertion, floor-panel re-price, TG3/TG4 re-reads | 0 | ~0 (minutes, pure math) |
| fresh validation V1+V2+V3 | 40,000 | **0.72** (≤1.4 at 2× margin) |
| — deferred, separate registrations — | | |
| Tier-2 27-cell map (n_alive/level extension; enables real battery re-label) | 172,800 | ~3 at bubble rate; up to ~10 blended (6-alive rate 0.39 s) |
| Tier-2b gradient/sign-flip probe (prices battery re-label aggressiveness) | 60,000 | ~1.1 at bubble rate |
| retrain with corrected returns | — | pod-scale (out of scope here) |

Run placement: night shift, nice'd, free cores per the core ledger;
per-state JSONL logging (CLAUDE.md per-iteration-logging rule);
benchmark-one-shard-first rule applies before the validation launch.
