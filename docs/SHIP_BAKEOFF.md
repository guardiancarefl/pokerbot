# Pre-ship candidate bake-off — final ICM/hand table + verdict

**Date:** 2026-06-04
**Branch:** `track-policy`
**Verdict (one-line):** **Ship `rebel_value_net_full` (d3k150 resolver).** It
wins panel-mean ICM/hand by **+0.022** over the canonical k200 blueprint
(0.1604 vs 0.1383, ~16% relative), with three statistically significant
wins (NIT, STATION, MANIAC) and only one statistically-borderline
regression (KillPhilMTT, the known range-dependent leak). The KillPhil
hit (Δ = −0.028 ± 0.034 vs blueprint) is within 2 SE of zero but
directionally real; we accept it because it's structural (closing it
requires per-opponent range modelling — gated out by `PHASE2_GATE_REPORT`).

## Setup

Shared-CRN across all four candidates: each candidate sees the SAME
sampled starting state per hand index AND the SAME `play_hand` seed
(common opponent RNG and common chance dealing). Action divergence at
decision points causes the actual game trajectory to diverge across
candidates, but the environmental setup (cards, dealer, alive seats,
opp rng seed) is shared.

- **500 hands per (candidate, matchup) = 10,000 hand-evals.**
- Total wall: **18 min** on Contabo CPU (one process; CFR resolver
 dominated the time inside the rebel candidate).
- Opponent factories built once per matchup, reused across all
 candidates.
- Probe script: `scripts/candidate_bakeoff.py`.

## Full ICM/hand table (mean ± 2 SE per cell)

```
matchup           k200_iter2000     k200_iter2800    k1000_iter0527    rebel_d3k150
KillPhilMTT       -0.0149 ±0.0281  -0.0147 ±0.0258  -0.0222 ±0.0300  -0.0431 ±0.0346
NIT               -0.0176 ±0.0308  -0.0193 ±0.0341  -0.0102 ±0.0345  +0.0278 ±0.0202
STATION           +0.2233 ±0.0517  +0.2179 ±0.0525  +0.1619 ±0.0574  +0.2882 ±0.0382
MANIAC            +0.3332 ±0.0435  +0.3257 ±0.0447  +0.3236 ±0.0449  +0.3744 ±0.0377
LAG               +0.1677 ±0.0404  +0.1570 ±0.0397  +0.1111 ±0.0448  +0.1546 ±0.0372
```

### Panel summary

```
metric                     k200_iter2000   k200_iter2800   k1000_iter0527   rebel_d3k150
panel_mean ICM/hand          +0.1383         +0.1333         +0.1128         +0.1604
worst_matchup ICM/hand       -0.0176         -0.0193         -0.0222         -0.0431
```

- **Best panel-mean: `rebel_d3k150` (+0.1604).** 16% relative advantage
 over k200_iter2000; smaller but consistent advantage over the
 late-iter k200_iter2800 and the partial k1000_iter0527.
- **Best worst-matchup: `k200_iter2000` (−0.0176)** — the rebel regression
 against KillPhil is the largest negative cell anywhere in the table.

### ReBeL head-to-head vs k200_iter2000 (Δ = rebel − blueprint, CRN paired)

```
matchup           Δ (rebel − k200_iter2000)    ±2 SE       verdict
KillPhilMTT       -0.0282                       0.0338     ~tie (directional regression)
NIT               +0.0453                       0.0338     rebel+ (2.7 SE)
STATION           +0.0648                       0.0480     rebel+ (2.7 SE)
MANIAC            +0.0413                       0.0379     rebel+ (2.2 SE)
LAG               -0.0131                       0.0420     ~tie

Loose-archetype slice (STATION + MANIAC + LAG):
  mean k200      +0.2414
  mean rebel     +0.2724
  Δ              +0.0310 (rebel beats blueprint by ~13% on the loose row)

Tight slice (KillPhilMTT + NIT):
  mean k200      -0.0163
  mean rebel     -0.0077
  Δ              +0.0086 (NIT win outweighs KillPhil loss; rebel still ahead)
```

## ReBeL question — settled

**Does the d3k150 ReBeL resolver beat the bare k200_iter2000 blueprint
on the deployment panel?**

**Yes — directionally and statistically.** Rebel wins on 3 of 5 matchups
with deltas at or above 2 SE (NIT, STATION, MANIAC), ties on LAG,
underperforms on KillPhilMTT within 2 SE. Both the loose-archetype slice
(+0.031) and the tight slice (+0.009) favour rebel. The panel-mean
advantage (+0.022) is meaningful given the per-matchup SEs (~0.04).

**Where rebel costs:** the KillPhilMTT regression (−0.028 vs blueprint).
The prior diagnosis (`f2940b9`, `docs/REBEL_KILLPHIL_DIAG_FINDINGS.md`)
established that this is range-dependent exploitation by KillPhil's
literal-7%-shove range against the resolver's wider effective belief.
Three independent attempts to close it have all hit the same Pareto wall
(12-lever sweep, policy-net warm-start, static range fix, ICM-EV rule —
the last documented in `docs/ICM_CALL_DIAG.md`). It is structural; no
non-adaptive fix exists in our constraint stack.

**Why we ship rebel anyway:** the KillPhil cost is bounded
(−0.028 ICM/hand) and statistically borderline (~1.6 SE). The
loose-archetype wins are larger (+0.04 to +0.06) and more significant.
Real recreational poker populations are dominated by loose play, not
literal-KillPhil shove-fold bots; the EV-weighted expectation is
strongly in rebel's favour. If the deployment population shifts toward
tight-shanky-style opponents, the calculus changes — flag for
re-evaluation post-launch.

## k1000 — asymmetric read (the user-specified framing)

**k1000_iter_0527 is UNDERTRAINED.** Training was interrupted at iter 527
(`metrics.json`: `"interrupted": True, "last_iteration": 527`); all
per-iteration metric logs are empty. k200 was trained to iter 2000+ on
a 5× smaller abstraction; k1000 needs proportionally more iterations
than k200 to converge per-bucket, so 527/2000 ≈ 26% of an
abstraction-equivalent training budget.

**A loss vs k200 is therefore uninformative.** The asymmetric read:
*where does partial-k1000 sit, and does finishing look worthwhile?*

| Matchup | k200_iter2000 | k1000_iter0527 | gap |
|---|---|---|---|
| KillPhilMTT | −0.0149 | −0.0222 | k1000 behind by 0.007 |
| **NIT** | **−0.0176** | **−0.0102** | **k1000 AHEAD by 0.007 (tight row)** |
| STATION | +0.2233 | +0.1619 | k1000 behind by 0.061 (largest gap) |
| MANIAC | +0.3332 | +0.3236 | k1000 behind by 0.010 |
| LAG | +0.1677 | +0.1111 | k1000 behind by 0.057 |

Panel mean: k1000 at +0.1128 vs k200 at +0.1383, a gap of −0.026 ICM/hand
on the **partial** run.

**Is finishing worth pursuing?** Mixed signal:

- **Positive evidence for finishing**: at only ~26% of training budget,
 k1000 already MATCHES or BEATS k200 on the tight row (NIT: ahead;
 KillPhilMTT: within noise). The richer postflop abstraction is
 expected to help most where bucket coarseness costs the most —
 KillPhil-style hand-strength-sensitive spots. The fact that k1000 is
 already competitive there suggests the richer abstraction is doing
 some of what it was designed to do.
- **Negative evidence**: the largest gaps are vs loose opponents
 (STATION −0.061, LAG −0.057). Loose opponents play multi-way pots
 where postflop abstraction richness should help most. If those gaps
 are still −0.06 at iter 2000+, the k1000 hypothesis weakens.
- **Net read**: finishing the k1000 run is worth pursuing — the cost
 is bounded (resume training to iter ~2000+ on RunPod or Contabo)
 and the tight-row early-signal is genuinely encouraging. But it's
 not a sure thing; the loose-row gap needs to close, and we can't
 predict that without finishing.

**Recommendation on k1000:** queue the resume run as a Phase-4f
follow-up. Do NOT block the rebel ship on it.

## k200_iter_2800 read

Session 5's close note said the anchor "plateaus around iter 500-1000".
This bake-off confirms: `k200_iter_2800` is essentially indistinguishable
from `k200_iter_2000` on every matchup (max diff 0.011, all within 2
SE). Training to iter_2800 added nothing measurable. Iter_2000 remains
the canonical k200 ship point.

## Files / artifacts

Committed this turn:
- `scripts/candidate_bakeoff.py` — shared-CRN bake-off probe (253L).
- `docs/SHIP_BAKEOFF.md` — this report.
- `runs/phase1_d128_repro/candidate_bakeoff.json` — full per-hand
 numerical results (force-added per metrics.json precedent).

Inputs unchanged:
- All four candidate checkpoints (frozen)
- `configs/ignition_double_up_6max_turbo.yaml`
- `runs/abstraction_k1000_retrofit_20260530_221744/abstraction.pkl`
- `runs/k200_abstraction.pkl`

## Ship decision

**Ship: `rebel_value_net_full.pt` (the d3k150 value-only ReBeL resolver).**

**Reason:** Best panel-mean ICM/hand on the deployment-relevant opponent
panel, with statistically significant wins against three of five
matchups including both loose archetypes. The one regression
(KillPhilMTT, −0.028 ICM/hand) is structural — no situational-only fix
exists for it (confirmed by three independent prior diagnoses), and
adaptive within-match exploitation is gated out by
`PHASE2_GATE_REPORT.md`. The integration cost (live depth-3 CFR search
per decision) is bounded; we have the gate2 framework already
demonstrating it works at deployment latencies.

**Known live-data caveat:** stage-blind opponents inflate hero edge on
the loose row (per `STAGE_AWARENESS_AUDIT` — stamp not re-derived here).
Treat the +0.16 panel mean as an upper bound; live deployment numbers
will be discounted by however much real humans bubble-tighten.

**Follow-ups (not blocking ship):**
1. Resume k1000 training from iter 527 to iter ≥ 2000 on RunPod;
 re-bake-off afterwards.
2. Wire `opponentsattable` to live alive-count for Shanky bots
 (~1-2 hour fix per scoping report); re-measure KillPhil/timidtom
 numbers with bubble-aware shanky opponents.
3. Stage-aware archetype design (half-day to a day; opens up real
 panel measurement on bubble-tightening loose opps).
