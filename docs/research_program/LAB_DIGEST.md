# LAB DIGEST — autonomous run (check on your schedule)

_Updated: 2026-06-13. Mode: autonomous. Gates: canonical (self-anchor Δ-from-champion
baseline ≈ −0.06 / ΔFIELD ≥ +0.05 / flat aggression). Probe-before-program, gates
never weakened, negatives fully reported, verify-don't-assume._

---

## ⚠️ NEEDS NICK — one coordination action (not a hard block)

**Route the Windows scraper briefs (OQ-4 capture / OQ-1 folded-flag) to a Windows
CC session.** The live-vs-sim gap is diagnosed (below) as ~entirely scraper data
quality, and the dominant fixes are Windows-side — which this Contabo bench can't
touch. The briefs are already written and awaiting your routing. This is the
single highest-EV action right now. (Not pod/arm/deploy — a coordination call.)

_Hard buttons that pause+flag: pod-spend (≥3 pod-ready methods); any arm/deploy to
live; another genuine strategic fork; gate-can't-decide ambiguity._

## DECISION (operator, 2026-06-13): Q0 SHELVED → pivot to OPERATIONAL
Q0 banked as a SUCCESSFUL outcome ("champion near-optimal vs this field; don't
retrain"). Sizing (Q0b) shelved with it. Revisit only after operational EV is
captured, or vs a tougher field. Now full-focus operational.

---

## OPERATIONAL TRACK (active focus)

### ✅ #1 live-vs-sim gap — DIAGNOSED (`reports/LIVE_VS_SIM_GAP_DIAGNOSIS.md`)
A near-optimal bot goes 9W/13L live because **it's the SCRAPER, not the model.**
~30% of hero-to-act moments are lost to scraper data quality (skipped/force-folded
incl. **pocket aces folded on invariant_fail**). Dominant causes: scraper-suspect
frames 56% (capture/OQ-4) + dealer-button detection 33% + reconstruction failures.
**Fix = Windows scraper reliability (OQ-4 + OQ-1).** Worth far more than any model
tweak. → needs operator routing (flag above).

### #2 capture reliability (OQ-4) — same root cause; scoping next
The wrong-window capture (session-6) IS the dominant skip cause (suspect frames).
Next free-bench step: read the OQ-4 brief + session-6 audit, sharpen the Windows
brief with the triage frame-evidence so the Windows session has exact targets.

### #3 Stage-2 autoclicker readiness — mapping next
N=5 clean-sessions gate reportedly CONFIRMED (OPERATOR_QUEUE). Next: read
STAGE2_LEDGER + DEPLOYMENT_PACKAGE to map what's gated-in vs blocking, and confirm
the SanityChecker (binding Stage-2 constraint) status against the live evidence.

_Live deploy / arm / autoclicker-go = your button. I do free analysis + scoping._

---

## VERDICTS SO FAR (Q0 — convert field data → winning policy)

| exp | method | verdict | ΔFIELD | notes |
|-----|--------|---------|--------|-------|
| H4 | global league-RNR | FAIL (collapse) | +0.065* | *artifact; self-anchor −0.18/z−3.7 |
| C/C2/C3/R6 | battery leak-map | headroom = ARTIFACT | — | 0.087/spot didn't transfer live |
| F | un-gated oracle overlay | FAIL | +0.015 | position-blind, washed out |
| B (broad) | gated learned head | FAIL | −0.077 | over-applied (over-fold harm) |
| B (heads-up) | gated, exact domain | FAIL | −0.011 ≈ 0 | **oracle yields ~0/game even in-domain** |

| tilt | crude global directions | NO edge | all ≤0 | aggro_more −0.004, call_more −0.16, fold_more −0.04, aggro_less −0.21, tight_pre −0.06 |

| tilt (postflop) | crude postflop directions | NO edge | all ≤0 | aggro_more +0.004, rest ≤0 |

**Tilt sweeps (global + postflop): champion locally ROBUST on both** — no crude
direction beats it on any street; deviating either way costs (calibration good).
Suggestive of near-optimal vs this field (not proof — a subtle state-conditioned edge
would need a learned BR / A; sizing is data-gated). THREE dry signals → program-pivot
flag raised above.

**JAM-WALL EXHAUSTED** (4 methods ≤0). The battery's "headroom" was an enumerated/
heads-up/ICM artifact that doesn't realize as a live per-game edge. Champion proven
near-optimal **on the jam wall only** — sizing/postflop/strong-opp UNTESTED (a
tooling/data gap, not an edge gap). Full writeup: `q0_probes/C2_C3_RESULTS.md`.

## INSTRUMENTS BANKED (reusable, all free)
generate-only buffer rebuild (`populate_only`) · jam-wall training gate (`jam_wall_only`) ·
exploit-head module (`exploit_head.py`) · gate dashboard/monitor · field battery ·
proven-faithful pod snapshot (Contabo `pod_snapshot/`, restorable) · the bit-identical
champion full-rebuild ckpt.
