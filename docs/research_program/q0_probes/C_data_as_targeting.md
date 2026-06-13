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

## Probe-specific falsifier (the thing that kills C)
**Do exploitable spots even exist beyond the champion's current play?** If the
targeting analysis finds NO spot with both (field-exploitability > threshold) AND
(champion materially deviates from the exploiting line), then **the field is not
exploitable beyond what the champion already does** — C is an informative
NEGATIVE that would also cast doubt on A/B/D (there may be nothing to win). This
is high-value either way.

## Kill criterion
Stop at the read-only stage if no spot clears both thresholds. Report
"field not exploitable beyond champion" — a program-level finding.

## Why first
Free, safe, and it tells A and B WHERE to focus (or that there's nothing to
focus on). It cannot degrade the champion (no gradient).
