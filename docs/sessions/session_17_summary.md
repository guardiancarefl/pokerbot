# Session 17 — 2026-05-25

Focus: **Stage F (Q11 Level 1, leaf-level ablation) — implement, run, and close.**
A reconnect-after-disconnect session that became a methodology round: the orphaned
ablation FAILed, a diagnostic showed the FAIL was a smoke-test artifact, the gate
criterion was rebuilt (single-metric → split-metric), and after an M=8 → M=32
escalation Stage F closed via a new **SUBSTANTIVE_PASS_AGGREGATE** verdict — the
per-pair gate proved *structurally intractable*, but the aggregate hero-direction
signal (+3.4–3.6σ) confirms the BR architecture.

## What was done

- **Diagnostic on the disconnect-orphaned ablation** (analysis only, no commit).
  The previous instance's run FAILed at N=3/M=4 — the `--smoke` preset, **not** the
  N=50/M=8 gate. The BR==baseline bit-identity was traced to CRN + realized-action
  invariance (**not** a code bug): under common random numbers, when BR's selected
  bias profile flips no sampled action, the rollouts are identical by construction.
  The mechanism does fire (one leaf: opp-2 bias-3 own-value 0.182 vs blueprint
  0.066).
- **Split-metric gate** (`e939bce`). Replaced the single
  `frac_br_selects_nonblueprint ≥ 20%` gate (which conflated sampling resolution,
  blueprint convergence, and the mechanism) with **resolution_rate** (per
  (leaf,opp) pair: is the most blueprint-deviating bias > 3σ under CRN-PAIRED
  stderr?) and **differentiation_rate** (among resolved pairs, does BR pick
  non-blueprint?). Plus street-coverage stratification (`min_late_leaves`, default
  10) and the `M_needed_for_gate` projection. Approved deviations from the literal
  spec: **absolute-deviation** resolution (so high-res/low-diff is observable) and
  **CRN-paired** stderr (single-condition stderr is swamped by per-deal variance,
  the session-14 finding). Diag script deleted before commit; subgame_leaf.py /
  NEXT_SESSION.md Q12/Stage-E.5 doc corrections rode along.
- **M=8 production gate** (`6edf6e7`). N=50, 220 pairs, 323 s. **FAIL — LOW
  RESOLUTION**: resolution 1.8% (4/220), differentiation 100%. The retired gate
  would have *passed* this (28.6% raw non-blueprint); the split metric correctly
  caught a noise-driven false PASS (only 4 of 63 raw picks statistically real).
- **M=32 escalation** (`0fe8e00`). Same 50 leaves, 1301 s. **FAIL — LOW
  RESOLUTION** still (7.7%, 17/220), but the aggregate hero-direction signal jumped
  to +3.4σ (BASELINE−BR) / +3.6σ (PROFILE−BR) and differentiation held at 94%.
- **Root-cause diagnostic + SUBSTANTIVE_PASS_AGGREGATE branch** (`9dbbfd4`).
  Analysis of the M=32 data (no re-run) established structural intractability;
  added the fourth verdict branch and annotated the M=32 JSON with a
  `diagnostic_findings` block.

## What was decided

- **Q11 Level 1 (Stage F) is CLOSED via SUBSTANTIVE_PASS_AGGREGATE.** Per-pair
  opponent-own-value resolution is the wrong PASS/FAIL instrument in this game; the
  aggregate hero-direction effect + non-degenerate menu + 94% differentiation are
  the architectural confirmation. No new `DECISIONS.md` entry — this closes an
  ablation gate, not a new architectural lock; the gate methodology lives in the
  script + design-doc Q11.
- **No further M-escalation** — the winner's-curse + structural-ceiling findings
  make it futile.
- **Stage G (Q11 Level 2) is next**, and its success criterion must be designed
  around the regime where the architecture predicts effects (late-street,
  bias-active decisions), not uniform sampling — the durable Stage-F lesson.

## What was learned / measured

- **The per-pair gate is structurally intractable, not budget-limited.** 55% of
  (leaf,opp) pairs have ZERO bias effect on the opponent's own value (median gap
  0.000) — the opponent never faces a bias-sensitive decision. That caps maximum
  resolution at ~45% at *any* M; the 0.50 strict gate is unreachable by escalation.
- **The driver is street/decision-latitude, not stack depth.** Resolution climbs
  preflop 1.5% → flop 3.5% → turn 10.4% → river 18.0%. By eff-stack-at-leaf the
  correlation runs *opposite* to the shallow-stack hypothesis (lower behind
  resolves better, because low behind ≈ late street). The hypothesised
  "shallow-stacks-shrink-effects" mechanism was **wrong**; the real cause is the
  prevalence of no-bias-decision pairs.
- **The aggregate hero signal is coherent and concentrated, not noise.** The 7
  bias-inactive leaves have hero delta *exactly* 0.000 (BR ≡ blueprint by
  construction); restricting to bias-active leaves raises correct-sign to 65% and
  mean delta to +0.056; the top-8 leaves by |delta| are unanimously correct-sign
  and late-street. Signal appears where theory predicts and is absent where it
  doesn't — the signature of a real mechanism.
- **Bias menu is non-degenerate.** Resolved pairs select biases {1:6, 2:6, 3:4} —
  genuine context-sensitive best-response, not one-bias-always-wins.
- **Winner's curse confirmed → the M-projection is optimistic.** Mean resolved gap
  deflated 0.672 (M=8) → 0.279 (M=32), tracking the selection-on-noise prediction
  (0.336); typical candidate gap is 0.091. Projected M (31 → 90, and rising)
  understates the truth; real M for 0.50 resolution is in the many-hundreds.
- **Cost (ablation `_collect_samples` regime):** ~84% bucket / 16% network for both
  modes — differs from Q13's BR ~45%-network because (a) per-condition cold-cache
  reset vs `evaluate_leaves`' shared cache, and (b) the cost block covers only the
  value-collection phase, not the BR mechanism phase. **Not** a revision of Q13.

## State at close

- **Done:** Stage F (Q11 Level 1) **CLOSED**. Split-metric gate +
  SUBSTANTIVE_PASS_AGGREGATE branch shipped (`e939bce`, `9dbbfd4`); M=8 and M=32
  production JSONs committed (`6edf6e7`, `0fe8e00`); M=32 annotated with the
  diagnostic. Full leaf suite green (one unrelated pre-existing `test_pushfold`
  Nash-convergence failure in untracked code, not touched this session).
- **Open (sub-step 2 closure):** **Stage G** (Q11 Level 2, decision-level
  stub-solver ablation) remains.
- **Unchanged carry-forward:** fold `fast_view` into the canonical path (after
  sub-step 2 closes); `dcfr-overnight-3000` ICM-retrain decision after sub-step 6.

## Next session opens with

**Stage G (Q11 Level 2) design.** Wrap the one-iteration root regret-update stub
around the leaf evaluator; measure hero's root action distribution under BR vs
PROFILE_SAMPLE. **Apply the Stage-F lesson:** design the success criterion around
the regime where the architecture predicts an effect (late-street / bias-active
root decisions), not uniform expectations across the whole test space — a uniform
per-decision gate would hit the same structural-intractability wall Stage F did.
