# ICM-EV short-stack call rule — symmetry test verdict

**Date:** 2026-06-04
**Branch:** `track-policy`
**Verdict:** **Situational-layer premise FAILS.** ICM-EV-aware calling at <10bb
with the deployable tight16 range assumption is asymmetric in the WRONG
direction: small positive vs KillPhil (within noise), neutral vs STATION,
and a significant negative against MANIAC (−0.040 ICM/hand aggregate,
A+ verdict at 2 SE). The rule is "tightening in disguise" against
sufficiently loose shovers.

## What was tested

Hypothesis: an ICM-EV-aware fold/call threshold (three-branch
Malmuth-Harville diff with payouts [2,2,2] and current stack distribution)
at <10bb commit-call spots, leaving all >10bb play untouched, would close
the KillPhil leak without costing against loose opponents — because hero's
equity against loose shovers should be high enough to clear even a tighter
ICM threshold.

Range assumption fixed at **`tight16`** (deployable population prior; the
panel-average tight shove range). The experiment isolates ICM-EV-vs-chip-EV
at fixed range — the only thing varied is the threshold rule.

## Setup

Paired-CRN gate2 framework (`scripts/rebel_gate2.py`):
- Hero A = value-only ReBeL resolver (`rebel_value_net_full.pt`, d=3, k=150),
 no override.
- Hero B = same resolver + `--ss-rule icm_ev --ss-range tight16
 --short-stack-fix-b 1` (override fires when start stack <10bb, to_call >0,
 to_call >= 70% of remaining).
- Common-random-numbers: identical sampled tournament starting states,
 hole-card deal, opponent RNG; only the hero policy differs between A and B.
- 500 paired hands per matchup. KillPhil + STATION run together (seed
 2026); MANIAC run separately (same seed).
- Per-stack-bucket break-out (<10bb, 10–20bb, 20+bb) for the spot-level
 attribution the user requested.

## ICM-EV branch (the swap)

In `_short_stack_override`, when the spot has exactly one opponent with
`contribution > hero.contribution` (clean shover, no side-pot):

```
eq        = equity(hero_cards, tight16, board)
s_fold    = monies; s_fold[shover] += pot
s_win     = monies; s_win[hero]   += pot
s_lose    = monies; s_lose[hero]   = max(0, monies[hero] - effective_call)
                     s_lose[shover] += pot + effective_call
icm_call  = eq * icm_equity(s_win, [2,2,2])[hero]
          + (1-eq) * icm_equity(s_lose, [2,2,2])[hero]
icm_fold  = icm_equity(s_fold, [2,2,2])[hero]
call iff icm_call >= icm_fold
```

Multi-shover (side-pot) spots fall through to the existing chip-EV rule.

## Numbers (500 paired hands per matchup, CRN, ±2 SE)

### Aggregate (all stack buckets)

```
matchup       A baseline      B (icm_ev)       Δ (B-A)        ±2 SE    verdict
KillPhilMTT    -0.0431         -0.0269         +0.0162        0.0290   ~tie
STATION        +0.2882         +0.2901         +0.0019        0.0221   ~tie
MANIAC         +0.3744         +0.3341         -0.0403        0.0349   A+
```

### <10bb bucket (where the leak lives)

```
matchup        n     A_bucket        B_bucket        Δ_bucket        ±2 SE
KillPhilMTT   289   -0.0771         -0.0491         +0.0280         0.0500
STATION       289   +0.2722         +0.2756         +0.0034         0.0381
MANIAC        289   +0.2881         +0.2183         -0.0697         0.0601
```

The 10–20bb and 20+bb buckets show Δ=0 by construction (override is
gated to <10bb start stacks).

### Override telemetry

```
KillPhil + STATION run (1000 paired hands):
  ss-fix B fires=179  calls=45  folds=134
  icm_ev branch decided: 62 (35%)
  multi-shover chip_ev fall-through: 117 (65%)
  mean eq=0.350  mean threshold=0.617

MANIAC run (500 paired hands):
  ss-fix B fires=150  calls=60  folds=90
  icm_ev branch decided: 21 (14%)
  multi-shover chip_ev fall-through: 129 (86%)
  mean eq=0.355  mean threshold=0.503
```

The icm_ev branch's actual fire rate is **35% vs KillPhil/STATION,
14% vs MANIAC** — the rest is chip_ev fall-through. So the experimental
result is a mixture: 14–35% icm_ev + 65–86% chip_ev-with-tight16. The icm_ev
rule itself has limited isolated impact.

## Interpretation

### What this tells us about the original hypothesis

The user's prediction: an ICM-aware rule would help vs KillPhil and not
hurt vs loose, because hero's equity vs loose shovers is high enough to
clear the (tighter) ICM threshold.

The data falsifies that prediction in the MANIAC case:
- KillPhil <10bb: +0.028 ± 0.050 — directionally positive (helps), not
 statistically distinct from zero at 2 SE.
- STATION <10bb: +0.003 ± 0.038 — clean tie (the prediction holds here).
- **MANIAC <10bb: −0.070 ± 0.060 — significantly negative. The override
 costs vs MANIAC about as much as it recovers vs KillPhil.**

### Why MANIAC breaks the asymmetry

Two compounding reasons:

1. **Range assumption mismatch (the larger driver).** The override
 computes `equity(hero, tight16, board)`. Against MANIAC's actually-wide
 shove range, hero's TRUE equity is much higher than `equity_vs_tight16`
 indicates. The threshold (icm_ev or chip_ev) folds spots that are +EV
 against MANIAC's true range. This is the same range-dependence trap the
 prior static fix (`killphil7`) hit, just rotated: there, KillPhil's
 actually-tight range was used vs all opponents and the symmetry cost was
 on STATION; here, tight16 (panel-average) leaves the cost on MANIAC.

2. **Multi-shover fall-through dominates.** Of MANIAC's 150 override
 fires, 86% (129) fall through to the chip_ev path with tight16 range
 — that's identical to the prior static fix's behaviour on those spots.
 So we're partly re-testing the prior static fix, not the pure ICM-EV
 rule. The pure ICM-EV branch only ran 21 times against MANIAC.

### Why KillPhil isn't dramatically rescued either

The KillPhil delta is +0.016 aggregate, +0.028 in the <10bb bucket — both
within 2 SE. The prior static-range fix using `killphil7` recovered +0.027
KillPhil aggregate (63% of the −0.043 baseline leak). `tight16` + icm_ev
recovers only the +0.016 aggregate. The prior fix was tighter (killphil7
is narrower than tight16, so equity vs killphil7 is lower → more folds at
the same threshold). Switching to a wider range assumption costs recovery
on the tight side.

### The deeper structural point

The original f2940b9 commit's verdict stands and is now confirmed from a
second angle:

> The leak is range-dependent. A single static call range cannot beat
> the resolver across the panel.

Adding payout-structure (ICM) information to the threshold doesn't change
this. The threshold still consumes `equity_vs_fixed_range` as input; if
the range is wrong, the threshold is wrong direction. ICM math gives a
better fold criterion CONDITIONAL on a correct equity input — but the
equity input requires a per-opponent range estimate, which is precisely
what the anonymity-+-short-format constraint denies us.

The situational layer (stack-depth + payout structure, no opponent
modelling) CANNOT close the leak. Any single fixed-range choice is locally
Pareto: helps some opponents, hurts equal-magnitude on opposite-type
opponents. ICM-EV doesn't break this Pareto; it shifts where the cost
lands.

## Verdict for project decisioning

- **Do not ship the icm_ev override.** It's net-negative across the
 KillPhil + STATION + MANIAC slice that this experiment covered
 (panel-summed Δ = +0.016 + 0.002 − 0.040 = **−0.022 ICM/hand**).
- **Do not pursue further situational-layer-only refinements at <10bb
 call-vs-shove.** The Pareto wall confirmed here is structural, not a
 tuning issue.
- **Stack-situation adaptation may still be valuable in other spots**
 (preflop range tightening, ICM-pressure betting/raising on the bubble),
 but NOT at the <10bb call-vs-shove window the KillPhil diagnosis
 identified. That spot needs opponent-range information.
- This closes the third independent track confirming the f2940b9 verdict:
 the 12-lever sweep, the policy-net warm-start, the killphil7 static
 range, and now the ICM-EV rule have all hit the same wall. The
 anonymity-+-short-format constraint forces either: (a) accept the
 blueprint and stop, or (b) the Phase-2 RL-style adaptive layer — and the
 prior PHASE2_GATE_REPORT already gates that out for the 10–22 hand
 window.

This experiment is the cleanest closure on situational-only fixes that we
have. Both the symmetric trap (f2940b9) and the situational generalisation
(this report) are now logged.

## Files / artifacts

Committed this turn:
- `scripts/rebel_gate2.py` — adds `--ss-rule {chip_ev, icm_ev}` flag,
 the three-branch ICM diff inside `_short_stack_override`,
 multi-shover-fallthrough telemetry, per-stack-bucket CRN delta breakout,
 and `MANIAC` in the matchup PANEL.
- `docs/ICM_CALL_DIAG.md` — this report.
- `evals/icm_ev_gate2_killphil_station.log` — KillPhil+STATION run log.
- `evals/icm_ev_gate2_maniac.log` — MANIAC run log.

Inputs unchanged:
- `rebel_value_net_full.pt` (frozen value-net resolver)
- `src/nlhe/icm.py` (Malmuth-Harville, used as-is)
- `configs/ignition_double_up_6max_turbo.yaml` (turbo schedule)
- prior baseline `evals/diag_killphil_1k.json` (trusted, not re-run)
