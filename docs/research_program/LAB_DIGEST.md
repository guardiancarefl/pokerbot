# LAB DIGEST — autonomous run (check on your schedule)

_Updated: 2026-06-13. Mode: autonomous. Gates: canonical (self-anchor Δ-from-champion
baseline ≈ −0.06 / ΔFIELD ≥ +0.05 / flat aggression). Probe-before-program, gates
never weakened, negatives fully reported, verify-don't-assume._

---

## ⚠️ NEEDS NICK — PROGRAM PIVOT (model bench PAUSED, awaiting your fork)

**Three independent "no live edge" signals are now in. I've paused the model-
exploitation queue and the fork is yours:**
1. Jam-wall: 4 methods ≤0 (battery headroom = artifact).
2. Global tilt sweep: no crude direction beats champion.
3. Postflop tilt sweep: no crude postflop direction beats champion.

→ The champion is near-optimal vs THIS WEAK FIELD on every dimension we can
faithfully test. No live model edge found. (Caveat held: this is "testable
dimensions" — a subtle state-conditioned edge or the data-gated SIZING dimension
remain genuinely untested.)

**THE FORK (your call — I won't pick it):**
- **Option 1 — PIVOT TO OPERATIONAL** (my rec by EV). Stop model-vs-this-field
  exploitation; actualize the proven +0.746 edge — autoclicker/capture reliability
  (the session-6 wrong-window failure is a direct EV leak), volume, live-vs-sim gap.
  Free code/analysis is within my autonomy; live deploy/arm is your button.
- **Option 2 — BUILD A SIZING-FAITHFUL POOL** (the one untested model dimension).
  Real data effort to replicate the field's limp/min-raise (T4/T6) → unlocks sizing,
  where the field actually reveals itself. Bigger build, likely pod-scale eventually.
- **Option 3 — BANK "champion near-optimal vs this field" as the Q0 verdict** (a
  successful outcome: "don't retrain vs this field"), decide operational vs sizing later.

My honest recommendation: **Option 1 now (highest realized-EV), Option 2 as the
parallel model track if you want a model iron in the fire.** But it's your strategic call.

_Other buttons that pause+flag: pod-spend (≥3 pod-ready methods); any arm/deploy;
gate-can't-decide ambiguity._

---

## RUNNING NOW
- **PAUSED** — model-exploitation queue stopped pending the program-pivot fork above.
  No probe running; no pod; no live anything.

## QUEUE (resumes per your fork)
- Option 1 (operational): free analysis tracks (capture-failure root-cause, live-vs-sim
  gap) are within my autonomy; live deploy/arm is your button.
- Option 2 (sizing pool): a real data build — I'd scope it, you approve.
- Strong-opponent robustness: a free probe, but needs a strong opponent set we don't
  have (would build one first).
- Frontier tier (bbnorm/novel-obj/MoE): premature until a target is located.

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
