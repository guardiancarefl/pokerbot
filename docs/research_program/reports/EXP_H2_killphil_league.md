# EXP_H2 — killphil-league probe — FINAL REPORT (2026-06-12): FAIL at probe

## Verdict

**H2 FAILS at probe stage.** Every load-bearing pre-registered bar
broken (spec §5, frozen before launch at commit aab1f52):

| metric | probe ckpt_2000 | baseline | bar | verdict |
|---|---|---|---|---|
| F-M1 battery EV-loss | 0.1375 ± 0.0008 | 0.0867 ± 0.0007 | ≤ 0.0650 | **FAIL** (+59% worse) |
| F-M2a killphil row | −0.2360 ± 0.0217 | −0.0800 ± 0.0223 | ≥ −0.0300 @2σ | **FAIL** (−0.156, ≈5σ) |
| F-M2b ticketmaster | +0.3940 ± 0.0206 | +0.5400 ± 0.0188 | > −2σ_diff | **FAIL** (−5.2σ) |
| F-M2b sng | +0.8060 ± 0.0132 | +0.8360 ± 0.0123 | > −2σ_diff | hold (−1.7σ) |
| F-M2b tighttom | +0.7790 ± 0.0140 | +0.7780 ± 0.0140 | > −2σ_diff | hold |
| F-M2b self-anchor | −0.2160 ± 0.0218 | 0 | |z| < 2 | **FAIL** (z = −9.9) |

Probe call mass on the battery went 35.9% → 52.1% — the league
exposure made the over-calling defect WORSE, not better.

## The interruption question (resolved against relaunch)

The run was interrupted by a live window at iter 1800 and resumed from
a slim checkpoint (second reservoir rebuild). The pre-registered
infrastructure-relaunch clause was considered and REJECTED on evidence:
**ckpt_1800 (pre-interruption, 300 contiguous league iters) is WORSE
than the final checkpoint** (M1 0.1565 vs 0.1375; call mass 57.5% vs
52.1%). The collapse was fully developed before the interruption; the
post-resume segment partially recovered. Relaunching "because of the
interruption" would be results-motivated — forbidden by the clause's
own text.

## Mechanism (hypothesis for any successor, NOT a verdict)

The probe cannot distinguish (a) "league exposure doesn't teach
shove-defense" from (b) "fine-tuning from a SLIM checkpoint (empty
reservoirs) with 30% scripted-opponent traversals collapses the
policy." Three observations point at (b) as a live possibility:
1. The self-anchor collapse (−0.216 vs its own initialization) is far
   too broad for a shove-defense-specific update.
2. M1 trajectory IMPROVED 1800→2000 as the reservoir matured
   (0.1565 → 0.1375) — consistent with early-training-on-thin-buffer
   damage, partially healing.
3. v2's independent failure had the same broad-regression signature
   (a3 report) under a different treatment — fine-tuning fragility is
   a repeated pattern on this stack.
A successor hypothesis (H2b: full-buffer continuation or
fresh-run-with-league-mix-from-iter-0, new registration, fresh
falsification bars) is a REFILL-pass candidate, to be judged on EVoI —
it is NOT entitled to H2's slot.

## What survives

- The fold-vs-shove battery + oracle (permanent panel instrument).
- The post-fix baselines and b2's checkpoint-stability calibration
  (spread 0.022 ≪ bars — this FAIL is not checkpoint noise either).
- The league-mix infrastructure (pool, config, launcher) — reusable.
- Champion remains the deployed reference; nothing here touches live.

## Costs

Probe ~2.6 h compute (interrupted + resumed), battery ~35 min,
diagnostics ~10 min. Within estimate.
