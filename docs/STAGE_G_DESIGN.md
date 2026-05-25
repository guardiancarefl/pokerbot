# Stage G — Q11 Level 2 (decision-level ablation) — design proposal

**Status:** PROPOSAL (review gate). No implementation, no measurement runs yet.
**Predecessor:** Stage F (Q11 Level 1) CLOSED via `SUBSTANTIVE_PASS_AGGREGATE`
(`docs/sessions/session_17_summary.md`; `scripts/ablation_leaf_eval.py`).
**Question:** when BR-mode leaf values feed a regret update, does hero's *root
policy* actually change vs PROFILE_SAMPLE, in the predicted direction — before we
build the real sub-step-3 CFR loop?

This is a sub-step-2 deliverable: the stub one-iteration root regret update is
superseded by the real solver in sub-step 3 (design doc Q11, lines 658–677).

---

## What the stub is, mechanically

The "one-iteration root regret update at the root infoset only" reuses three pieces
already in the tree — it reimplements none of them:

- **The depth-1 subgame tree** (`subgame.build_subgame_tree`, `max_action_depth=1`).
  At depth 1 every child hits the depth limit and becomes a LEAF (or TERMINAL)
  before any chance/decision expansion (`subgame.py:281–293`). The root therefore
  has ≤7 children, one per `DiscreteAction`, built through the *identical*
  discretize path the CFR walker uses (`subgame.py:373–396`, mirroring
  `cfr6.traverse_6max`). **No tree traversal / backup is required** — each action's
  value comes straight from leaf evaluation.
- **The leaf evaluator** (`subgame_leaf.evaluate_leaf`, `:614`). For each child it
  returns the per-seat ICM-equity-delta 6-vector under `ctx.mode` ∈
  {`BEST_RESPONSE`, `PROFILE_SAMPLE`} (`subgame_leaf.py:81–91`). Hero's action value
  is `q_M[a] = value[hero]`. TERMINAL children carry no `leaf_value`; their hero
  value is `icm_adjust_returns(child.returns(), starting_stacks, payouts)[hero]`
  (same units, `subgame_leaf.py:335–337`).
- **The regret math** (`cfr6.py:371–385`) and **RM+** (`solver._strategy_from_advantages`,
  `solver.py:143`). Both are imported, not re-derived.

Blueprint root strategy: `σ0 = RM+(adv)`, `adv = policy_nets.predict_advantages(hero, feat)`
via the encoder path already used at `subgame_leaf.py:256–282` and `cfr6.py:348–353`.

**The update (one iteration, warm-started at the blueprint):**

```
ev_M   = Σ_a σ0[a] · q_M[a]                  # blueprint-mix value under mode-M leaves
r_M[a] = (q_M[a] − ev_M) · legal_mask[a]     # instantaneous regret (cfr6.py:378)
σ_M    = RM+(adv + r_M)                       # blueprint regret + one fresh iteration
```

It borrows the regret formula and RM+ from `cfr6`/`solver`; it **skips** the full
recursive traversal, the multi-iteration accumulation, the buffer writes, and the
average-strategy net. This is the minimal mechanism that yields a real signal:

- `r_M = 0` (q flat across legal actions) ⇒ `σ_M = RM+(adv) = σ0` — reduces to
  blueprint when the subgame is non-informative.
- `q_BR ≡ q_PROFILE` (bias-inactive root) ⇒ `r_BR ≡ r_PROFILE` ⇒ `σ_BR ≡ σ_PROFILE`
  — a correct cross-mode no-op (note: this is *cross-mode identity*, **not**
  `σ = σ0`; see G-E).
- Adding `r_M` to `adv` (not replacing) is scale-correct because the advantage net
  regresses *instantaneous* O(1) ICM-unit regrets (`cfr6.py:374–377`), so `adv` and
  `r_M` are commensurate — one iteration is a ~real perturbation, not one part in
  thousands.

Smallest footprint that gives real signal: a `scripts/ablation_decision_level.py`
that reuses Stage F's battery sampler, CRN seeding, per-sample rollout collection
(`ablation_leaf_eval.py:232,333`), `_resolution` (`:299`), and JSON/verdict
scaffolding — applied to root *children* instead of a single leaf — plus a ~30-line
stub solver as above.

---

## Part 2 findings — the regime where the architecture predicts an effect

Recomputed from the M=32 production data
(`evals/leaf_eval_ablation_session17_M32_20260525T041122Z.json`, 50 leaves, 220
pairs):

| signal | preflop | flop | turn | river | overall |
|---|---|---|---|---|---|
| hero-delta-active (|Δ|>0) | 62% | 92% | 100% | 92% | **86%** (43/50) |
| ≥1 opp BR-bias ≠ blueprint | 46% | 100% | 92% | 100% | **84%** (42/50) |
| ≥1 RESOLVED opp pair (>3σ) | 8% | 15% | 42% | 75% | **34%** (17/50) |
| mean |baseline−BR| hero-Δ | 0.052 | 0.068 | 0.079 | **0.128** | — |

**Conclusion: most root decisions are Stage-G-active.** Only 7/50 leaves (14%) are
fully bias-inactive (hero-Δ exactly 0). Because a *root* has ~5 action-children and
is active if *any* child is, the root-active fraction is **≥ 85%, almost certainly
> 90%**. The 55%-dead-*pairs* structural wall that made Stage F's per-pair gate
intractable (`diagnostic_findings.structural_intractability`) is a per-(leaf,opp)
phenomenon; it does **not** recur at the root level. Stage G is *not* doomed to the
same wall.

**But the Stage-F lesson still applies in its magnitude form.** Detectability and
effect size track the street/decision-latitude gradient (resolved-any 8%→75%;
|Δ| 0.052→0.128 ≈ 2.5×). A *uniform* sample would be dominated by low-magnitude
preflop roots and would dilute the signal. So Stage G samples broadly **but
stratifies toward where the magnitude lives** and gates on the active regime, with
an aggregate fallback for the residual no-ops — the Stage-F pattern, softened to fit
the milder decision-level regime.

---

## G-A — Sampling strategy

**Chosen: stratified-by-street with late-street oversampling (option 2), behind a
"real-decision" inclusion pre-screen (option 3 mechanics).** Justified by Part 2:
filtering for activity (option 3 alone) is unnecessary (>85% active); pure uniform
(option 1) dilutes into low-magnitude preflop; the magnitude gradient is the axis
that matters, so stratify on street.

Inclusion filter at battery construction (pre-committed; the *definition* of the
test regime, not cherry-picking):

1. Hero has **≥ 3 discrete legal actions** — a real decision with mixing latitude,
   not fold-or-shove. This directly excludes the collapsed preflop-short-stack
   pushes the architecture cannot move, the decision-level analog of Stage F's
   non-ITM filter.
2. **≥ 1 live opponent with chips behind** (post-decision latitude) — same filter as
   Stage F (`ablation_leaf_eval.py:163`).
3. **Non-ITM** root (ITM short-circuits to a bias-insensitive Option-A value,
   `subgame_leaf.py:520,579`).

Reuse `build_battery` (`ablation_leaf_eval.py:178`) with the added ≥3-action check
and `min_late_roots = 20` (turn+river), heavier than Stage F's 10 because Stage G's
magnitude lives there. **Target N = 50 roots** (≈ 250 child-evaluations ≈ 5× Stage
F's data) — keeps street strata ≥ ~12 each and gives the aggregate ample power. The
≥3-action filter naturally co-selects deeper-stack/later-street roots, reinforcing
the stratification.

---

## G-B — What is measured

For each root, run the stub under **{BR, PROFILE, baseline}** (baseline = all
opponents pinned to bias 0 via a degenerate prior). **CRN is paired across modes**
(shared per-deal seeds, the Stage-F necessity, `ablation_leaf_eval.py:347`): for each
legal action `a`, collect per-sample hero values `q_BR[a][m]`, `q_PROFILE[a][m]`,
`q_baseline[a][m]` over the same M deals using the rollout primitives already in
`_collect_samples` (`:232`). Means → `q̄_M` → `σ_M` via the update above.

Primary observable: **`σ_BR` vs `σ_PROFILE`**. Per-root metrics:

- **L1(σ_BR, σ_PROFILE)** = Σ_a |σ_BR[a] − σ_PROFILE[a]| — does BR move the policy?
- **Entropy** H(σ_BR), H(σ_PROFILE) — the design predicts BR is *flatter/more-mixed*
  (Q11 hypothesis), so report the signed Δ = H(σ_BR) − H(σ_PROFILE).
- **KL(σ_BR ‖ σ0)** — how far BR pulls hero off the unrefined blueprint.
- **Most-shifted action** = argmax_a |σ_BR[a] − σ_PROFILE[a]| — for the
  non-degeneracy guard (is it always the same action, or context-sensitive?).
- **CRN-paired action-value difference** d[a] = q_PROFILE[a] − q_BR[a], for the
  value-suppression continuity check and the resolution test below.

These are the right four because L1 answers the core Stage-G question (policy moves),
entropy tests the *stated directional* hypothesis, KL anchors the move relative to
the blueprint, and the most-shifted action guards against a one-action artifact.

---

## G-C — Success criterion (pre-committed; not negotiable post-results)

Split-metric, mirroring Stage F (`ablation_leaf_eval.py:617`), with the
`SUBSTANTIVE_PASS_AGGREGATE` branch adapted. **These thresholds are fixed before the
run** (the load-bearing Stage-F lesson; numbers are hypotheses calibrated to Stage
F's measured effect sizes, to be re-confirmed by fresh measurement, not retuned to
the result):

Metrics:
- `resolution_rate` — frac of roots where the most mode-deviating action's CRN-paired
  d[a] clears 3σ (Stage F's `_resolution`, `:299`, applied to hero-action-values).
- `differentiation_rate` — among resolved roots, frac with H(σ_BR) > H(σ_PROFILE)
  (predicted flatter direction).
- `value_suppression_sigma` — aggregate of CRN-paired d[a] over all (root, action),
  mean/stderr. Continuity with Stage F (which hit +3.4σ at M=32); confirms the leaf
  mechanism fires at the children.
- `entropy_delta_sigma` — aggregate of H(σ_BR) − H(σ_PROFILE) over roots.
- `policy_divergence` — mean L1(σ_BR, σ_PROFILE) with a CRN-paired bootstrap CI
  (B=500 resamples of the M deals, recomputing both policies from the *same*
  resampled deals so the noise floor is paired).
- `n_distinct_shifted_actions` — distinct most-shifted actions across resolved roots
  (non-degeneracy / W2 analog).

**PASS (strict):** `resolution_rate ≥ 0.50` AND (`differentiation_rate ≥ 0.60` OR
`resolution_rate ≥ 0.90`) AND `value_suppression_sigma ≥ 3`.

**SUBSTANTIVE_PASS_AGGREGATE** (the realistically expected outcome, by analogy to
Stage F): `resolution_rate < 0.50` AND **all** of —
- `value_suppression_sigma ≥ 3` (mechanism fires at the children),
- `policy_divergence` bootstrap CI lower bound > 0 (BR demonstrably moves the policy
  in aggregate — the *core* Stage-G claim),
- `entropy_delta_sigma ≥ 2` in the predicted (flatter) direction,
- `differentiation_rate ≥ 0.55`,
- `n_distinct_shifted_actions ≥ 2` (non-degenerate).

**FAIL:** otherwise → **STOP and surface to the human; do NOT proceed to sub-step 3**
(Path A discipline, `ablation_leaf_eval.py:48–56`).

*Effect-size grounding.* Stage F's resolved-pair mean gap was ~0.28 and its aggregate
hero signal +3.4σ at M=32; the per-action analog d[a] is ~0.05–0.13 (Part 2 table).
The 3σ value-suppression bar is reachable because the Stage-G aggregate pools ~250
(root, action) pairs (5× Stage F's 50 leaves), shrinking the stderr. The directional
entropy bar is set softer (2σ) because "BR flatter" is a genuinely less certain
prediction than Stage F's "BR picks non-blueprint" (a negative entropy-Δ is a flag to
surface, not an auto-pass).

---

## G-D — Computational cost

Cost is **dominated by the BR mechanism** (per-opponent bias selection: v live opps
× k=4 biases × M rollouts), not the value rollouts. From Stage F, one leaf's full
3-condition cost ≈ **6.4 s at M=8 / 26 s at M=32** (323 s and 1301 s ÷ 50 leaves).

Stage G evaluates ~5 children per root, each ≈ one Stage-F leaf:

| setting | per-root | N=50 sequential | parallel (Y≈24) |
|---|---|---|---|
| M=8 | ~32 s | ~27 min | ~1–2 min |
| **M=16 (default)** | ~64 s | **~54 min** | ~2–3 min |
| M=32 (escalation) | ~130 s | ~108 min | ~5 min |

All settings fit the budget; even M=32 sequential (~1.8 h) is under the 2 h CPU
threshold, and each root is independent so it parallelizes trivially (Q13: many CPU
cores, not GPU — single-row forward is launch-bound). **Default M=16** — Stage F
needed M≥16 for any per-pair resolution (1.8% at M=8), and the hero-action-value
signal is the noisy one. One escalation to M=32 is permitted *only* if the aggregate
lands in a 2–3σ near-miss band (a measurement-budget result); below 2σ at M=16 with
low resolution is a genuine FAIL, not an escalation trigger (Stage F's
"no-endless-escalation" lesson). **No scope-down needed.**

---

## G-E — Failure modes and responses

1. **All leaf values identical (no bias-active children).** `q_BR ≡ q_PROFILE` ⇒
   `r_BR ≡ r_PROFILE` ⇒ `σ_BR ≡ σ_PROFILE` ⇒ L1 = 0, entropy-Δ = 0. The correct
   no-op is **cross-mode identity**, *not* `σ = σ0`. (`σ = σ0` holds only in the
   stronger case where q is flat *across actions*, `r_M = 0`.) Both are correct
   behavior, not bugs; the test plan asserts the *right* invariant for each
   (G-F #1). Such roots are unresolved and counted as expected no-ops.
2. **Degraded leaf** (`LeafEvalResult.degraded`, `subgame_leaf.py:220`: budget
   breach / all-samples-failed → Option-A or zero-vector). A zero-vector child would
   silently corrupt q and the policy. **Response: if any child is degraded, mark the
   whole root degraded and exclude it from the gate aggregates** (Stage F's `usable`
   filter, `ablation_leaf_eval.py:430,517`). Conservative; one bad child cannot ship
   a corrupted root policy.
3. **Hero is the only live player.** Cannot occur: the ≥1-live-opponent inclusion
   filter (G-A #2) excludes it at construction, and a hero-already-won state is
   terminal and never sampled.
4. **ITM children** (a child where an action busts/commits a seat into the money).
   Short-circuits to a bias-insensitive Option-A value → identical q across modes for
   that action → contributes to the no-op direction. Not a bug; documented.

---

## G-F — Test plan (must pass before the production run; production game per discipline)

1. **`test_stub_no_signal_invariants`.** (a) Force the mode to return constant q
   across actions ⇒ assert `σ_M == RM+(adv) == σ0` within fp tolerance (`r_M = 0`).
   (b) Force `q_BR == q_PROFILE` ⇒ assert L1(σ_BR, σ_PROFILE) == 0 and entropy-Δ == 0
   (cross-mode no-op). Asserts the *correct* invariant for each case (G-E #1).
2. **`test_stub_synthetic_bias_pulls_toward_boosted_action`.** Boost one legal
   action's q far above the rest ⇒ assert `σ_M[a*] > σ0[a*]` and `σ_M[a*]` is the
   argmax. Confirms the regret update moves mass toward higher-value actions.
3. **`test_stub_reproducibility`.** Same root, same seed, two runs ⇒ `σ_BR`,
   `σ_PROFILE` identical to fp tolerance (relies on the per-call cache-reset
   determinism, `subgame_leaf.py:621–636`).
4. **`test_stub_degraded_child_marks_root_degraded`.** Stub one child's
   `evaluate_leaf` to return `degraded=True` ⇒ assert the root record is flagged
   degraded and dropped from the aggregate (G-E #2).
5. **`test_stub_root_structure_production_game`.** Build the depth-1 tree from a real
   `sample_starting_state` root ⇒ assert 3 ≤ n_children ≤ 7, all children
   LEAF/TERMINAL, and `σ_BR` sums to 1 on legal actions / 0 on illegal.

---

## Pre-commitment

The G-C thresholds and the G-A inclusion filter are fixed as of this proposal. After
the run, the verdict is read off them mechanically; results are not used to retune
the gate (Stage F lesson). A FAIL stops and surfaces; it does not get rationalized
into a pass.
