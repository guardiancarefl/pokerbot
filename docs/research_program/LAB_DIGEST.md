# LAB DIGEST — autonomous run (check on your schedule)

_Updated: 2026-06-13. Mode: autonomous. Gates: canonical (self-anchor Δ-from-champion
baseline ≈ −0.06 / ΔFIELD ≥ +0.05 / flat aggression). Probe-before-program, gates
never weakened, negatives fully reported, verify-don't-assume._

---

## ⚠️ NEEDS NICK
**Nothing BLOCKING right now** — lab running clean on the free bench, no pod, no arm/deploy.

**Standing strategic option (non-urgent, your call when you want it):** the evidence
is trending toward a PROGRAM PIVOT to operational EV (you've seen the roadmap). Not
flagged as blocking — I'm keeping the free model queue producing (postflop is the
biggest untested dimension, no data gate) while you hold the fork. I'll escalate this
to a hard flag IF the postflop probe also comes up empty (that would make
"model-vs-this-field is diminishing-returns" decisive).

_Buttons that will PAUSE the lab and flag here: (1) pod-spend when ≥3 pod-needing
methods are training-ready; (2) any arm/deploy to live; (3) a forced program pivot;
(4) genuine gate-can't-decide ambiguity._

---

## RUNNING NOW
- **Postflop tilt sweep** (`probe_tilt_sweep.py --streets postflop`, 5 dirs × 800
  games) — isolates postflop-specific crude edge (the global sweep mixed streets).

## NEXT IN QUEUE (autonomous, free, cheapest-first)
1. Strong-opponent robustness check (only beat a weak scripted field so far).
3. Frontier tier (bbnorm organ / novel objective / MoE) — only after a target is located.
4. Sizing (Q0b) — **data-gated**; needs a sizing-faithful pool = a real data build =
   PROGRAM PIVOT → will flag, not start unilaterally.

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

**Tilt sweep (global): champion locally ROBUST** — no crude direction beats it; deviating either way costs (calibration is good). Suggestive of near-optimal vs this field on crude directions (not proof — a subtle state-conditioned edge would need a learned BR / A to find). Postflop isolation running.

**JAM-WALL EXHAUSTED** (4 methods ≤0). The battery's "headroom" was an enumerated/
heads-up/ICM artifact that doesn't realize as a live per-game edge. Champion proven
near-optimal **on the jam wall only** — sizing/postflop/strong-opp UNTESTED (a
tooling/data gap, not an edge gap). Full writeup: `q0_probes/C2_C3_RESULTS.md`.

## INSTRUMENTS BANKED (reusable, all free)
generate-only buffer rebuild (`populate_only`) · jam-wall training gate (`jam_wall_only`) ·
exploit-head module (`exploit_head.py`) · gate dashboard/monitor · field battery ·
proven-faithful pod snapshot (Contabo `pod_snapshot/`, restorable) · the bit-identical
champion full-rebuild ckpt.
