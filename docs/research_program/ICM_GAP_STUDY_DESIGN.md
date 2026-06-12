# ICM Gap Study — Pre-Registered Design (RESEARCH_MAP c1)

**Status: DESIGN ONLY — nothing in this document has been run.**
Written 2026-06-12 under Addendum 5 slate item (2e). Pre-registered per
Addendum 1 (hypothesis, method, falsification criteria, cost — written
BEFORE running). Runner, configs, and seeds specified here are binding
for the eventual run; deviations must be logged in EXPERIMENT_LOG.md
before results are looked at.

---

## 1. Question and why it matters here

**c1: How wrong is Malmuth-Harville (MH) ICM for THIS format?**
Format: 6-max SNG, top-3 equal payout (each survivor gets 2.0 buy-ins),
escalating blinds per `configs/ignition_double_up_6max_turbo.yaml`,
level clock = 5 hands/level in all our simulation harnesses
(`hands_per_level=5`).

MH ICM (`src/nlhe/icm.py::icm_equity`) maps a stack vector to per-seat
finish equity using chip fractions only. Known failure modes, all
concentrated exactly where our gates concentrate (the 4-alive bubble):

- **Ignores position and the blind clock.** A 3bb stack about to post
  the BB at level 6 is strictly worse than the same 3bb stack with the
  button; MH prices them identically. Escalating blinds + 5-hands-per-
  level make this time-dependence strong in our format.
- **Ignores skill / policy asymmetry.** MH assumes finish order is
  determined by chip-proportional survival.
- **Equal-payout flattening.** With payouts [2,2,2] the only quantity
  that matters is P(finish top-3); MH's known rank-level distortions
  (P(2nd), P(3rd) biases) partially cancel under the sum. Whether they
  cancel *enough* is precisely what this study measures.

**Everything downstream inherits any MH error.** Audited consumers
(grep `icm_equity|icm_adjust` 2026-06-12):

| Consumer | Where | What inherits the error |
|---|---|---|
| Training terminal utility | `src/nlhe/icm_returns.py::icm_adjust_returns`, used by `cfr6.py`/`solver6.py` | every regret the blueprint ever learned |
| Resolver leaf values | `src/nlhe/subgame_leaf.py`, `subgame_solver.py` | depth-limited resolve quality |
| H2 battery oracle | `scripts/fold_vs_shove_battery.py` (`ev_fold = ICM(fold stacks)`, `ev_call = eq*ICM(win) + (1-eq)*ICM(lose)`) | every oracle label in `evals/h2_battery/battery_v1.json` (8,112 spots) |
| Eval scoring of capped games + per-hand stage deltas | `scripts/sng_baseline.py` (`stage_acc`, capped-game `hero_net`) | baseline yardstick tails |
| Tail-floor panel pricing | `scripts/eval_icm_panel_floor.py` | H1 floor calibration |

Note a critical distinction: the battery oracle and the resolver use
ICM **differences between nearby stack vectors** (one hand's outcomes),
while training uses per-hand deltas too. So the decision-relevant
quantity is the local *gradient* error as much as the absolute equity
error. The design measures both (§2.4).

## 2. Estimand and estimator

### 2.1 Estimand

For a mid-tournament state `s = (stacks, level, dealer)` with alive set
`A(s)`, define for each alive seat `i`:

- `q_i(s)` = MH ICM prediction of P(seat i finishes in the money)
  = `icm_equity(stacks, [2,2,2], eligible=A(s))[i] / 2.0` ∈ [0,1].
  (Equal payouts make equity = 2·P(ITM); we work in probability units
  throughout. **0.01 in probability units = 0.02 buy-in-equity units.**)
- `p_i(s)` = the TRUE probability seat i finishes in the money **under
  champion self-play continuation** of this exact tournament (same
  blind schedule, same 5-hands/level clock, same dealer rotation).

The per-seat bias is `b_i(s) = p_i(s) − q_i(s)`. The study estimates
`B = E_s[b_·(s)]` per stratum and per stack-rank/position slice, plus
the dispersion of `b` across states.

Scope statement (pre-registered): `p` is defined relative to champion
self-play, not to optimal or field play. That is the right reference
for our consumers — training, resolver leaves, and the battery all
price continuations that the champion itself plays out. Field-play
sensitivity is explicitly out of scope (Future Work).

### 2.2 Rollout measurement

Harness already exists: `scripts/sng_baseline.py::play_sng_game` with
the fixed-start seam added 2026-06-12 (`starting_stacks`,
`starting_level`, `starting_dealer`). One rollout = one call:

```
play_sng_game(champ, champ, structure,
              seed=rollout_seed(s, m),
              starting_stacks=s.stacks, starting_level=s.level,
              starting_dealer=s.dealer,
              seat_to_policy=[champion]*6,   # champion in ALL seats
              hands_per_level=5, max_hands=200, mode="sample")
```

- **Champion in all seats** (`seat_to_policy=[champion]*6`) removes the
  skill asymmetry — by symmetry, any per-seat deviation from `q` is
  attributable to stacks/position/blind-clock, not skill.
- Champion ckpt pinned:
  `runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt`
  (sha256 b79e82dd…, same as the e2 rebaseline header), `mode="sample"`.
- Outcome per rollout `m`: `Y_im = 1` if seat i has chips when the loop
  breaks at `n_alive <= 3` (the format's terminal: ITM = survived).
  `p̂_i(s) = (1/M) Σ_m Y_im`.
- **Capped rollouts** (`max_hands=200` reached before 3-alive — should
  be near-impossible from mid states given blind escalation): excluded
  from `p̂` and counted; a state with >2% capped rollouts is dropped and
  logged. **Tainted rollouts** (exception path): same treatment; >0
  taints triggers investigation before any analysis (e2 rebaseline had
  tainted=0 across 16k games, so expect none).
- Chip-conservation note: harvested stack vectors occasionally sum to
  slightly under 9000 (the documented placeholder pot leak upstream of
  the harvest; 947/1000 sampled records sum to exactly 9000). MH is
  scale-invariant in chip fractions and the rollout conserves whatever
  it starts with, so this is benign; record the total anyway.

### 2.3 Error metric and CI treatment — separating rollout noise from ICM bias

The M rollouts of one state share that state, so they estimate `p_i(s)`
with binomial noise; they tell us nothing *individually* about the
stratum. The unit of analysis is therefore the **state**, with the
per-state paired difference

```
d_i(s) = p̂_i(s) − q_i(s),    E[d_i(s)] = b_i(s),
Var[d_i(s) | s] = p_i(s)(1 − p_i(s)) / M        (binomial, given s)
```

Within one rollout the seat indicators are dependent (exactly 3 of the
alive seats survive, so `Σ_i Y_im = 3` and `Σ_i d_i(s) = 0` exactly).
We avoid the cross-seat dependence by analyzing **one scalar per state
per question**, pre-registered:

- **Primary scalar:** `d_short(s)` — the error for the shortest alive
  stack (ties → lowest seat index). The bubble decisions our gates care
  about are short-stack survival decisions, and the known MH failure
  modes load on the short stack.
- **Secondary scalars (same machinery, reported as a map):** error by
  stack rank (1=shortest … n=biggest), error by seat-relative-to-button
  (BTN / SB / BB / others at hand start), error vs depth-in-BB bands.

For a stratum with N states (i.i.d. draws from the harvest cell):

```
B̂ = (1/N) Σ_s d(s)
Var(B̂) = [ σ²_bias + E_s p(1−p)/M ] / N
```

where `σ²_bias = Var_s[b(s)]` is real heterogeneity of the ICM error
across states. CI: t-interval over the N per-state `d(s)` values
(states are independent; each `d(s)` already contains its own rollout
noise, so the empirical variance across states is the correct total).
Robustness check: BCa bootstrap over states. **Never** bootstrap over
rollouts pooled across states — that would treat shared-state noise as
independent and understate the CI.

**Noise/bias decomposition (reported per stratum):**

```
σ̂²_total = sample variance of d(s) over states
σ̂²_noise = (1/N) Σ_s p̂_s(1−p̂_s) / (M−1)      (unbiased binomial plug-in)
σ̂²_bias  = max(0, σ̂²_total − σ̂²_noise)
```

`σ̂_bias` is itself a headline output: even with mean bias ≈ 0, large
σ_bias means ICM is wrong state-by-state in compensating directions —
which still corrupts per-decision oracle labels.

**Multiplicity:** the confirmatory claim is tested on 3 pre-registered
primary cells (§3) with Bonferroni (α = 0.05/3 each). The full 27-cell
gap map uses Benjamini-Hochberg FDR at 0.05, labeled exploratory.

### 2.4 Secondary analysis — decision-relevant gradient error

The battery oracle compares `ICM(fold stacks)` vs
`eq·ICM(win) + (1−eq)·ICM(lose)`. To test whether MH error *changes
decisions* (not just levels), for a 200-state subsample of the primary
cells we additionally roll out the two counterfactual stack vectors of
a canonical short-stack jam/call event (fold-stacks and call-win /
call-lose stacks, constructed exactly as `fold_vs_shove_battery.py`
does), M=100 each, **sharing the rollout seed list across the three
arms** (common random numbers — the deal stream after the first hand
diverges, but CRN still cancels the level-clock and seat-rotation
noise). Output: rollout-EV(fold) − rollout-EV(call) vs ICM-EV(fold) −
ICM-EV(call), and the **sign-flip rate** — the fraction of probe states
where the corrected EVs reverse the oracle's fold/call label. This is
the number that directly prices a battery re-label.

## 3. Sampling frame and stratification

Frame: `data/training_dist_v1.json.gz` — 322,546 harvested hand-start
records `{stacks, level, dealer, n_alive}` from 20,000 champion-vs-pool
games. This IS the deployed state distribution; sampling from it makes
the measured gap representative of states the bot actually inhabits.

Measured composition (computed from the artifact 2026-06-12, light
read on cores 8-9):

| n_alive | levels 1-2 | levels 3-4 | levels 5+ | row total |
|---|---|---|---|---|
| 6 | 116,558 | 14,663 | 491 | 131,712 |
| 5 | 51,884 | 37,807 | 4,394 | 94,085 |
| 4 (bubble) | 21,877 | 54,164 | 20,708 | 96,749 |

Level histogram (raw): L1 99,492 / L2 90,827 / L3 67,448 / L4 39,186 /
L5 17,745 / L6 6,189 / L7 1,485 / L8 167 / L9 7.

**Stratification: n_alive {4,5,6} × level band {1-2, 3-4, 5+} ×
stack-dispersion tertile = 27 cells.** Dispersion metric: coefficient
of variation (CV) of alive-stack chip fractions; tertile cuts computed
*within* each (n_alive, level-band) cell so each tertile is exactly a
third of the cell. Measured per-cell tertile cuts (low/high):

| cell | T1/T2 CV cuts | median CV |
|---|---|---|
| (4, 1-2) | 0.378 / 0.568 | 0.451 |
| (4, 3-4) | 0.369 / 0.535 | 0.446 |
| (4, 5+) | 0.360 / 0.532 | 0.442 |
| (5, 1-2) | 0.350 / 0.425 | 0.376 |
| (5, 3-4) | 0.362 / 0.504 | 0.427 |
| (5, 5+) | 0.376 / 0.524 | 0.449 |
| (6, 1-2) | 0.037 / 0.125 | 0.072 |
| (6, 3-4) | 0.243 / 0.434 | 0.329 |
| (6, 5+) | 0.383 / 0.524 | 0.451 |

Sampling rule: uniform without replacement over **distinct**
`(stacks, level, dealer)` tuples within the cell (the harvest contains
heavy duplicates near the symmetric start; duplicates add no
between-state information). Cell (6, 5+) has only 491 records — cap N
at the distinct-tuple count and report the shortfall.

**Primary confirmatory cells (pre-registered): the three dispersion
tertiles of (n_alive=4, levels 3-4)** — the modal bubble mass (54,164
records, 17% of all hand starts) and exactly where the H1/H2 gates
concentrate.

**Control cell C0 (negative control + position probe):** the exact
symmetric start `stacks=[1500]*6, level=1, dealer=0`, M=2,000 rollouts.
By symmetry the true seat-averaged P(ITM) is 0.5 = ICM's prediction, so
the *seat-averaged* error must be ≈ 0 — any failure is a harness bug
(falsifies the instrument, halts the study). Meanwhile the per-seat
deviations from 0.5 *by seat-relative-to-button* are a clean, pure
measurement of the position effect ICM ignores — a known-direction
effect that validates the instrument's sensitivity.

## 4. Power analysis — N×M to resolve a 0.01 bias at 95%

All in probability units (0.01 here = 0.02 buy-in-equity).

Binomial building block: one rollout's indicator has variance
`p(1−p)`. At the bubble the short stack's typical `q` is in the 0.5-0.8
range; we budget with `p(1−p) = 0.1875` (p = 0.75) and check worst case
0.25. Per-state noise after M rollouts: `p(1−p)/M`.

Total variance of the stratum mean over N states:

```
Var(B̂) = (σ²_bias + p(1−p)/M) / N
```

Detecting bias δ = 0.01, two-sided α = 0.05, power 80% requires

```
SE(B̂) ≤ δ / (z_{0.975} + z_{0.80}) = 0.01 / (1.96 + 0.84) = 0.00357
```

We budget `σ_bias = 0.03` (no prior measurement exists — this is the
planning value; the C0 + first-cell data will update it, see the
adaptive rule below). With **M = 100, N = 250**:

```
Var(B̂) = (0.03² + 0.1875/100) / 250 = (9.0e-4 + 1.875e-3) / 250
        = 1.11e-5   →   SE = 0.00333  ≤ 0.00357  ✓
MDE(80% power) = 2.8 × 0.00333 = 0.0093 < 0.01  ✓
Worst case p(1−p)=0.25: SE = 0.00368, MDE = 0.0103 (marginal — accepted)
```

**Why not M = 1?** For the *pooled* bias alone, fixed budget T = N·M
gives `Var = σ²_bias·M/T + p(1−p)/T`, minimized at M = 1, N = T — pure
bias estimation wants all-states-no-repeats. We deliberately spend
M = 100 because the design's other deliverables need per-state `p̂`:
(i) the σ²_bias decomposition (§2.3) is unidentifiable at M = 1;
(ii) the gap-map / correction-curve fit (§6) regresses `p̂ − q` on state
features and needs per-state SE ≈ `sqrt(0.1875/100)` = 0.043;
(iii) the gradient probe (§2.4) needs per-state EV differences.
This trade-off is pre-registered, not an oversight.

Tier-2 mapping cells use **N = 120, M = 60**:

```
Var = (9.0e-4 + 0.1875/60) / 120 = 3.34e-5 → SE = 0.0058
MDE(80%) = 0.016 ≈ 0.02
```

i.e. the map resolves 0.02-probability biases per cell — adequate for
triage; any cell flagged ≥ 0.02 can be promoted to Tier-1 sampling
later.

**Adaptive rule (pre-registered):** after the first primary cell
completes, recompute σ̂_bias. If σ̂_bias > 0.05, increase N (not M) to
restore MDE ≤ 0.01 — at σ_bias = 0.05, N = 345 suffices — and log the
change before unblinding the other cells.

## 5. Run plan and cost

### 5.1 Measured rate basis

`evals/e2_rebaseline_20260612/*.log`, full games from level 1
(~28-36 hands/game), single worker per profile:

| log | s/game |
|---|---|
| sng | 263s/2000 = 0.13 |
| ticketmaster | 398s/2000 = 0.20 |
| tighttom | 646s/2000 = 0.32 |
| killphilmtt | 1076s/2000 = 0.54 |
| gushansenmtt | ~1.6 (still running at log read) |

Planning number **0.5 s/game** (the task's ~0.5 figure ≈ the killphil
rate; conservative for this study because (a) all-champion seats means
6 NN policies — more NN forwards per hand than 1-champion-5-shanky, but
(b) rollouts start mid-tournament, so games are far shorter than 35
hands — bubble starts end in a handful of hands). Sensitivity bracket
[0.15, 0.8] s/game. **Binding pre-step per CLAUDE.md: benchmark one
worker × 50 rollouts from a bubble start and a 6-alive start before
committing the full schedule; re-plan if outside the bracket.**

### 5.2 Tiers

| Tier | What | Rollouts | Core-h @0.5s | @0.2s |
|---|---|---|---|---|
| 0 | C0 control (1 state × M=2000, full games) + rate benchmark | 2,100 | 0.3 (full-game rate ~0.5s measured) | 0.3 |
| 1 | 3 primary cells × N=250 × M=100 | 75,000 | 10.4 | 4.2 |
| 2 | 24 map cells × N=120 × M=60 | 172,800 | 24.0 | 9.6 |
| 2b | gradient probe: 200 states × 3 arms × M=100 | 60,000 | 8.3 | 3.3 |
| | **Total** | **310k** | **~43** | **~17** |

Mid-state rollouts are shorter than full games, so true cost should sit
near the low column for Tiers 1-2b; the RESEARCH_MAP's 6-10 h CPU
estimate corresponds to Tier 0+1+2 at the observed short-game rates.
**Night-shift schedulable (Addendum 5.4):** sharded by cell across the
free cores (heavy-job taskset honored per the core ledger; nice'd),
one night at ~9 workers covers ~25-65 core-h. Schedule: Tier 0 → gate →
Tier 1 → (analysis of primaries) → Tier 2 + 2b the same or next night.
Tier 2 degrades gracefully: if the rate comes in slow, drop M to 40
(MDE 0.017, still triage-grade) rather than dropping cells.

### 5.3 Seeds, artifacts, runner

- Master seed 2026. State sampling: `random.Random(2026)` over sorted
  distinct tuples per cell. Rollout seed = stable hash
  `(2026, cell_id, state_idx, m)` — reproducible by a stranger.
- New runner script `scripts/icm_gap_study.py` (to be written at run
  time): loads the artifact, samples cells, loops `play_sng_game` with
  the fixed-start seam, writes one JSONL row per state:
  `{cell, state_idx, stacks, level, dealer, total_chips, q[6],
  survive_counts[6], M_eff, n_capped, n_tainted, mean_hands}` plus a
  run header (ckpt sha, artifact sha, git head) in the e2 style.
- Output dir: `evals/icm_gap_<date>/`. Analysis notebook-free: a
  `scripts/icm_gap_report.py` producing the per-cell table + map.
- No changes to `sng_baseline.py` needed — the seam is sufficient.

## 6. Decision matrix — what each outcome drives

Let `B*` = the largest |bias| among the 3 primary bubble cells (short-
stack scalar), with its Bonferroni-corrected CI; `flip` = oracle
sign-flip rate from §2.4.

| Outcome | Verdict | Action |
|---|---|---|
| `B* ≥ 0.02` (CI excludes 0.01), any primary cell | **MATERIAL** | (1) Fit a calibrated correction `p_corr = f(q, depth_bb, position, level)` (monotone in q) from the Tier-1/2 per-state data; (2) wrap `icm_adjust_returns` / `subgame_leaf` behind a flag for the next league retrain (re-prices every ICM-adjusted return — the RESEARCH_MAP's HIGH-EVoI branch); (3) re-label the H2 battery oracle with corrected equities and re-issue verdicts; (4) re-examine H1 tail-floor pricing against the corrected panel. |
| `0.01 ≤ B* < 0.02`, or `B* < 0.01` but `flip > 2%` | **MARGINAL** | Cheap fixes only: battery-oracle re-label (one offline pass over 8,112 spots) + a correction note on capped-game scoring in `sng_baseline.py`. Training/resolver repricing deferred; add a RESEARCH_MAP follow-up with the measured effect size. |
| All primary-cell CIs ⊂ (−0.01, +0.01) and `flip ≤ 2%` | **NULL / VALIDATION** | Standing validation note: DECISIONS.md entry "MH ICM validated for this format at ±0.01 probability (±0.02 equity) under champion self-play"; RESEARCH_MAP c1 → CLOSED (still report σ̂_bias and the position-effect size from C0 as standing caveats). |
| σ̂_bias > 0.05 with mean ≈ 0 | **HETEROGENEOUS** | Mean-zero but state-dependent error: oracle/per-decision consumers are still exposed. Treat as MARGINAL minimum; the correction-curve fit decides whether a feature-conditional correction recovers it. |
| C0 seat-averaged error ≠ 0 (CI excludes 0) | **INSTRUMENT FAILURE** | Halt; debug harness (seam, blind guard, capping) before any inference. |

The NULL branch is explicitly valuable (Addendum 1.2): it converts an
unexamined assumption load-bearing for the whole stack into a measured
one, and the C0 position-effect number is novel format-specific
knowledge either way.

## 7. Pre-registered falsification framing (Addendum 1)

- **H_c1 (the hypothesis under test):** "MH ICM finish-probabilities
  match champion-self-play finish frequencies to within ±0.01
  (probability units) in the deployed bubble distribution."
- **Falsified if:** any primary cell's short-stack bias CI
  (Bonferroni 0.05/3) lies entirely outside (−0.01, +0.01).
- **Confirmed (equivalence, not just non-rejection) if:** all three
  primary-cell 90% CIs lie inside (−0.01, +0.01) — TOST at α = 0.05.
  Powered for this: §4 gives MDE ≈ 0.0093 at the planned N×M.
- **Neither** (CIs straddle a bound): report as INCONCLUSIVE with the
  exact CI, and the adaptive-N rule (§4) governs whether to extend.
- Analysis plan, scalars, cells, and the decision matrix above are
  frozen at design time. Any post-hoc slice (e.g. a surprising seat
  pattern) is reported as exploratory, clearly labeled.

## 8. Threats to validity (pre-registered, with dispositions)

1. **`p` is champion-relative, not field-relative.** Accepted by
   design (§2.1): our ICM consumers price champion continuations. A
   field-policy sensitivity arm (pool opponents in 5 seats) is listed
   as Future Work, not folded in — it changes the estimand.
2. **Champion quality contaminates "truth".** If the champion plays the
   bubble badly, rollout frequencies reflect that. Mitigation: this is
   the *self-consistency* gap that training/resolver actually need;
   plus the C0 control bounds harness-level artifacts. Noted in report.
3. **5-hands-per-level clock vs real 5-minute levels.** The harness
   clock is itself an approximation of the live format; the study
   measures ICM-vs-rollout under the harness's own clock, which is the
   clock every consumer uses. Live-clock mismatch is a separate
   question (RESEARCH_MAP e-pillar candidate).
4. **Duplicate/near-duplicate states** reduce effective N if sampled —
   handled by distinct-tuple sampling (§3).
5. **Capped/tainted rollouts** bias `p̂` if dropped non-randomly —
   handled by per-state caps and exclusion logging (§2.2).
6. **Oversubscribed vCPUs** change wall-time, not estimates; per-
   iteration logging (CLAUDE.md rule) on every shard.

## 9. Deliverables

1. `evals/icm_gap_<date>/` raw JSONL + run headers.
2. `docs/research_program/reports/EXP_C1_icm_gap.md` — full Addendum-1
   report: per-cell bias table with CIs, σ_bias decomposition, gap map
   (n_alive × level × dispersion × stack-rank × position), C0 position
   effect, gradient/flip-rate result, verdict per §6/§7.
3. RESEARCH_MAP c1 update + DECISIONS.md entry on the verdict branch.
4. If MATERIAL/MARGINAL: the fitted correction artifact
   (`data/icm_correction_v1.json`) with its own validation holdout
   (20% of Tier-1 states never used in the fit).
