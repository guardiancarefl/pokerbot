# PROBE C — Data-as-targeting (diagnostic, not gradient)

**Run order: FIRST / always. Cost: ~3–5 h read-only, $0 training, Contabo.**

## Hypothesis (falsifiable)
The correct use of the field data is **diagnostic + eval-oracle**, not a training
distribution. The champion is already strong; its highest-value improvement is a
*surgical* fix at the 2–3 spots where the field is exploitable AND the champion
mis-plays — found by the data, fixed by targeted solving, with the data never
entering a gradient.

## Method
1. **Read-only targeting (no compute on the policy).** Over the frozen field
   portrait + the 517 hands + the field battery, rank spots by
   `(field-exploitability) × (champion-deviation-from-the-exploiting-line)`.
   Output: a ranked leak map (position × depth × spot).
2. **Optional surgical fix.** For the top 2–3 spots only, a depth-limited subgame
   solve (Layer-3) at those spots; everything else = champion verbatim.

## Gates — IDENTICAL to the canonical block (BURST_PLAN G1–G4)
- **G1 self-anchor:** surgical edits touch ONLY the identified spots, so
  self-anchor must read ≈0 (verify like generate-only — bit-identical to champion
  everywhere off the patched spots). Any broader drift = FAIL.
- **G2/G3:** the patched spots must show real field gain with FLAT aggression.
- **G4:** survivor → ≥2 out-of-pool holds.

## Probe-specific falsifier (the thing that kills C) — OPERATIONAL, pre-registered
**Metric (already built):** `grade_battery()` — champion mean ICM-EV-loss vs the
frozen per-spot jam-wall ORACLE (oracle = best of {call, fold} vs the
pool-as-implemented; the loss = oracle_EV − champion_EV). This IS the BR-gap: EV
the champion forgoes by not best-responding to the field. It is an UPPER BOUND
(per-spot best response, field fixed), so a small value is dispositive — "we
can't find it" cannot be confused with "it isn't there."
**Threshold (pre-registered, R6 no-go):** `R6_NOGO_EVLOSS = 0.02`/spot.
- champion `ev_loss_mean < 0.02` (with CI) ⇒ **HALT THE PROGRAM** — field not
  meaningfully exploitable beyond the champion; reconsider everything before any
  burst. This is the believable program-level negative.
- `≥ 0.02` ⇒ headroom exists; proceed to the leak map.

**STATUS: already computed (`evals/h4_field_battery/champion_baselines.json`,
P2_DONE).** Champion = **0.0866/spot (±0.0006), 16,224 spots, HALT=false — 4.3×
the threshold.** So the program-negative did NOT fire: **the headroom is real.**
The headroom is DEFENSIVE (oracle is call/fold vs the field's shoves), while the
H4 probe collapsed via OFFENSIVE over-aggression — i.e. global-RNR learned the
wrong adjustment. C therefore pivots from "does headroom exist" (answered: yes)
to the live deliverable below.

## Live deliverable (given R6 already passed): the leak map + reachability
Decompose the 0.087/spot champion loss by (position × depth × oracle-direction)
to find WHERE the headroom concentrates and confirm it is DEFENSIVE
(call/fold-vs-shove), not offensive. Output ranks the top 2–3 spot-clusters for
A/B to target. Reachability check: a surgical fix at those clusters must keep
self-anchor ≈0 (G1) — capturing defensive headroom should NOT require the
offensive over-aggression that collapsed H4.

## Kill criterion
Program-negative (HALT) if a future re-pool drops champion ev_loss < 0.02. With
the current pool it is 0.087 (>>0.02), so C proceeds to the leak map. If the leak
map shows the headroom is UNreachable without self-anchor cost (high-headroom
spots coincide with high champion-deviation that only over-aggression captures),
that itself bounds what A/B can achieve.

## Why first
Free, safe, and it tells A and B WHERE to focus (or that there's nothing to
focus on). It cannot degrade the champion (no gradient).
