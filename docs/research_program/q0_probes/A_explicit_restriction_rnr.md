# PROBE A — Explicit-restriction RNR (the direct fix for H4's root cause)

**Run order: 3rd. Cost: 3 λ × ~40 min parallel on pod.**

## Hypothesis (falsifiable)
The RNR *method* is sound; our anchor was fake. H4 collapsed because the 75%
self-play "anchor" tracks the DRIFTING current policy, not the champion. An
**explicit restriction to the FROZEN champion** — a KL/interpolation penalty of
strength λ to `b79e82dd` in the strategy-net loss — gives a real restoring force,
so some λ holds self-anchor while still adapting to the field.

## Method
- Load the frozen champion as a FIXED reference policy.
- Add `λ · KL(π_θ ‖ π_champion)` (or an interpolation toward champion) to the
  strategy-net objective. λ controls restriction strength.
- Sweep **λ ∈ {small, med, large}** as 3 parallel short probes (~500 iters each,
  the H4 length), `league_mix=0.25`, same pool.
- One delta vs H4: the explicit restriction term. Nothing else changes.

## Gates — IDENTICAL to the canonical block (BURST_PLAN G1–G4)
The self-anchor bar (G1) is THE test — it caught H4 and A must clear it where H4
failed. ΔFIELD (G2) must trend toward +0.05 with FLAT aggression (G3); a λ that
hits +0.05 only via Δaggr inflation is the artifact, scored FAIL.

## Probe-specific falsifier (the thing that kills A)
**Is there a sweet-spot λ?** A passes iff SOME λ holds self-anchor |z|<2 WHILE
ΔFIELD trends positive with flat aggression. A is FALSIFIED if the trade-off has
no sweet spot:
- small λ → collapses self-anchor like H4 (restriction too weak), AND
- large λ → pins the policy so hard ΔFIELD ≈ 0 (no exploitation room).
If every λ is either H4-collapse or zero-gain, the blueprint cannot be both
anchored and exploit — and the answer is B (move exploitation off the blueprint).

## Kill criterion
If the med-λ probe collapses self-anchor (G1 fail) AND the large-λ probe yields
ΔFIELD ≈ 0, stop the sweep — no sweet spot exists.
