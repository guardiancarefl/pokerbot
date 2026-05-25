# Session 18 — 2026-05-25

Focus: **Stage G (Q11 Level 2, decision-level ablation) — finish, run, and close.**
A reconnect-after-disconnect session: the orphaned Stage G implementation was found
already in its corrected state; the M=16 production gate landed a clean 2–3σ
near-miss FAIL; the one design-sanctioned M=32 escalation cleared both load-bearing
bars, closing Stage G via **SUBSTANTIVE_PASS_AGGREGATE** — the same verdict shape as
Stage F. **Sub-step 2 is now complete.**

## What was done

- **Recovered the disconnect-orphaned Stage G implementation (Path X).** The script
  (`scripts/ablation_decision_level.py`) was already in the post-correction state:
  the `kl_floor → l1_floor` rename was complete (zero remnants) and
  `differentiation_rate` was already direction-agnostic (L1 policy-shift magnitude
  `>= L1_FLOOR`, **not** the flatness test), with entropy demoted to a soft,
  reported-only signal (Addition 2). Re-running the N=5/M=8 smoke reproduced the
  pre-existing artifact **bit-identically**, confirming it was generated under the
  corrected code. Committed implementation + 5 green G-F tests + smoke (`cb82072`).
- **M=16 production gate** (`b3c1011`). N=50, M=16, seed=18, 3074 s. **FAIL**, but a
  clean 2–3σ near-miss on **both** load-bearing axes: value_suppression **+2.45σ**
  (bar 3σ) and policy_divergence significant at ~99.5% but not the required 99.7%
  (`frac_null_ge_obs` 0.005). resolution 18.0% (9/50), differentiation 77.8% (7/9),
  n_distinct_shifted 4, filter_rate 21.9%. Per design **G-D**, a 2–3σ near-miss is
  exactly the measurement-budget case that permits one M-escalation.
- **M=32 escalation** (`b4f85dd`). Same 50 roots (the battery is seed-driven and
  M-independent — a clean apples-to-apples comparison), M=32, 6230 s (≈ the design's
  ~108 min estimate). **SUBSTANTIVE_PASS_AGGREGATE**: value_suppression **+3.07σ**,
  policy_divergence **significant** (observed L1 0.330 > null p99.7 0.297,
  `frac_null_ge_obs` 0.000), resolution 28.0% (14/50), differentiation 71.4% (10/14),
  n_distinct_shifted 6. Entropy −0.29σ (reversed) — surfaced, does not gate.

## What was decided

- **Stage G (Q11 Level 2) is CLOSED via SUBSTANTIVE_PASS_AGGREGATE; sub-step 2 is
  complete.** Strict PASS is unreachable (per-root resolution is structurally capped,
  same wall as Stage F), but the load-bearing aggregate signals — BR demonstrably
  suppresses hero value (+3.07σ) and demonstrably moves the root policy above the
  per-deal noise floor (policy_divergence significant) — confirm the BR mechanism
  moves hero's *decision*, not just leaf values. No new `DECISIONS.md` lock: this
  closes an ablation gate, not an architectural choice.
- **Honored the pre-committed rule mechanically, including its escalation clause.**
  The rule has two clauses: sub-2σ misses → FAIL-and-stop; 2–3σ near-misses → one
  M-escalation. The M=16 result triggered the second clause; M=32 then resolved it.
  No threshold was renegotiated.
- **Methodology deviation approved: the per-deal mode-label permutation null
  (B=400) supersedes the design's literal "bootstrap CI-lower > 0" spec for
  policy_divergence.** The permutation null tests the meaningful question ("does the
  BR/PROFILE label assignment matter") whereas a bootstrap CI on L1 distances is
  near-trivial (L1 ≥ 0, so the lower bound is ~always > 0). Same pattern as Stage F's
  absolute-deviation / CRN-paired-stderr deviations — methodologically stronger than
  the literal spec. Documented in `docs/SUBGAME_LEAF_DESIGN.md` Q11 Level 2.

## What was learned / measured

- **The 2–3σ near-miss escalation clause worked exactly as designed.** From M=16 to
  M=32, value_suppression went 2.45σ → 3.07σ and policy_divergence not-sig → sig. The
  mechanism: the **null shrinks faster than the signal**. The permutation-null mean
  L1 fell 0.375 → 0.297 (more samples → less per-deal noise → less spurious
  label-permuted divergence) while observed L1 barely moved (0.374 → 0.330), opening
  the significance gap. This is the textbook signature of a real-but-small effect
  separating from noise as M grows — not a threshold-shopping artifact.
- **The "BR is flatter" hypothesis was WRONG at the decision level — and the gate
  correctly did not depend on it.** BR was slightly *less* flat than PROFILE at both
  M (entropy_delta −0.29σ; ~36% of resolved roots BR-flatter, i.e. a minority). The
  Pluribus-style "best-responding opponent pushes hero toward mixing" intuition did
  not hold in this shallow ICM SNG. What BR actually does is **shift mass** (material
  L1 on 71% of resolved roots, 6 distinct shifted actions) **without flattening**.
  Demoting entropy to non-load-bearing (Addition 2) before the run was the
  load-bearing design call — had it gated, a correct architecture would have spuriously
  FAILed on a wrong theoretical prior.
- **Same street/decision-latitude gradient as Stage F.** Resolution concentrates on
  the river (M=32: preflop 3/13, flop 2/13, turn 1/12, **river 8/12**). The effect
  lives where opponents face genuine committed decisions; the late-street
  oversampling in the battery (G-A) was the right design.
- **~22% of "real decisions" are still degenerate fold/shove** (filter_rate 21.9%,
  14/64) even after the ≥1-live-opp and non-ITM filters — under the 30% finding
  threshold, but a real property of this turbo SNG: a fifth of nominal decisions
  collapse to <3 actions.
- **Cost is linear in M, on estimate.** ≈61 s/root at M=16, ≈125 s/root at M=32; full
  N=50 runs 51 min / 104 min sequential (design G-D predicted ~54 / ~108). The host
  was a 128-core box with a GPU, but the solver loads on CUDA only for the
  launch-bound single-row forward; the cost is bucket/rollout-bound, matching Q13.

## State at close

- **Done:** **Stage G (Q11 Level 2) CLOSED** via SUBSTANTIVE_PASS_AGGREGATE; **sub-step
  2 complete.** Implementation + tests + smoke (`cb82072`), M=16 production
  (`b3c1011`), M=32 escalation (`b4f85dd`). All 5 G-F tests green. One minor gap: the
  degraded-child → root-degraded G-F #4 *test* is unwritten (the marking *logic* is
  present and exercised in `evaluate_root`); file under sub-step 3 hardening if the
  stub path is reused.
- **Open / next:** **Sub-step 3 — the real subgame CFR loop** (replaces the Stage-G
  one-iteration root stub). The stub is now discarded/superseded per design.
- **Unchanged carry-forward:** fold `fast_view` into the canonical path (now that
  sub-step 2 closes — see `NEXT_SESSION.md` tracked deliverable + acceptance);
  `dcfr-overnight-3000` ICM-retrain decision after sub-step 6; ALLIN→CALL(1) alias
  for SubgamePolicy (sub-step 5).

## Next session opens with

**Sub-step 3 — subgame CFR loop design.** Build the real multi-iteration
external-sampling CFR over the depth-limited subgame tree, with leaf values from the
(now-confirmed) BR evaluator, replacing the Level-2 stub. Carry the validated
methodology forward: late-street/decision-latitude is where the BR effect lives, the
split-metric + SUBSTANTIVE_PASS_AGGREGATE pattern is the right instrument for this
shallow SNG, and the "BR flatter" prior is **not** something to design around — BR
shifts mass, it does not flatten.
