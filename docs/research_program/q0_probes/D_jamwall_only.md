# PROBE D — Jam-wall-only training (control: isolates the imperfect-pool factor)

**Run order: 2nd (cheapest control). Cost: ~40 min on pod.**

## Hypothesis (falsifiable)
H4's secondary factor was that the pool is jam-wall-faithful but limp-style-
IMPERFECT (registered limitation 1), so training against the imperfect
dimensions taught blunt over-aggression. **If we train ONLY on the jam-wall-
faithful dimensions, the self-anchor collapse should disappear** (or shrink) vs
the full-pool H4 baseline.

## Method
- Mask the limp/entry-style spots; restrict league-RNR training to the spots
  where the pool is moment-matched faithful (the jam wall).
- Otherwise identical to the H4 probe (`league_mix=0.25`, same recipe, ~500
  iters). One delta vs H4: the dimension mask.

## Gates — IDENTICAL to the canonical block (BURST_PLAN G1–G4)
Same self-anchor (G1), ΔFIELD (G2), aggression (G3) bars. No softening.

## Probe-specific falsifier (what D decides — informative either way)
This is a CONTROL; both outcomes are valuable:
- **If self-anchor STILL collapses on jam-wall-only** → the imperfect pool was
  NOT the culprit → confirms the H4 root cause (the missing explicit anchor, §3)
  as primary, and says the pool fidelity is a red herring for the anchor problem.
  (Strengthens A/B over pool-fixing.)
- **If self-anchor HOLDS on jam-wall-only** (and ΔFIELD trends with flat aggr) →
  the pool imperfection WAS a major factor → the data-application fix may be as
  simple as scoping training to trustworthy dimensions, and it informs any future
  pool construction.

D is FALSIFIED as a *standalone fix* only if it neither holds self-anchor nor
yields field gain — in which case its value is purely diagnostic (which is still
useful).

## Kill criterion
Runs to completion (it is a controlled comparison; the comparison itself is the
deliverable). No early kill unless it trips the live collapse-watch, in which
case the collapse point is the datum.
