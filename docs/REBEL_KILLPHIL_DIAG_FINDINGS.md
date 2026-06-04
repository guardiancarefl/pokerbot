# KillPhilMTT leak: diagnosis + static-fix verdict (track-policy, 2026-06-04)

## TL;DR

The persistent −0.05 ICM/hand leak vs KillPhilMTT — a *fixed, non-adaptive* Shanky
bot — is **CONCENTRATED** in a narrow set of spots: short-stack
(<10bb) call-vs-shove decisions where the resolver assumes a generic (wider)
opponent shove range than KillPhil's actually-tight shove. A surgical static-range
fix CLOSES the KillPhil leak by 63%, but COSTS an equal-magnitude amount vs
STATION, producing a net-zero panel result. **The leak is range-dependent. A
single static call range cannot beat the resolver across the panel.** Closing
it requires opponent-range modeling — the adaptive within-match layer.

## What was tried (and not)

| Track | Build | Plateau-check | Verdict |
|---|---|---|---|
| Track-policy increment 1 | Policy-net warm-start at hero root (ReBeL value+policy) — trained on existing schema-2 shards (271k rows, ~40k-param MLP), wired via `SubgameSolveContext.root_policy_prior`, L1-matched mass to blueprint warm-start. | KillPhil 500 hands paired, A=−0.043, B=−0.046, Δ=−0.003 ± 0.021. timidtom and NIT also tie. | **NULL.** Policy-net val_KL=1.20 was barely better than uniform-over-legal (~1.43); prior shape too weak to redirect CFR. Data/capacity bound. |
| Track-policy increment 2 (this work) | Per-hand-instrumented leak diagnostic vs KillPhilMTT (`rebel_diag_killphil.py`): slice by ending street, stack bucket, position, vpip, all-in, blind level. | 1000 hands, mean ICM=−0.035 ± 0.024 — matches trusted baseline. | **Concentrated.** |
| Static-range fix (Option A) | `<10bb` call-vs-shove override (`ReBeLHero.short_stack_fix`): bypass resolver, compute hero equity vs hardcoded shove range, fold below chip-EV-pot-odds + ICM padding. Two ranges: `tight16` (~16% 6-max push, generic) and `killphil7` (~7%, KillPhil's actual literal allin range parsed from the Shanky profile). | 500-hand × 7-matchup paired gate. | **Symmetric / net-zero.** |

## The diagnostic findings (1000 hands vs KillPhilMTT, value-only ReBeL hero)

```
=== by ended_street ===
river            341    -0.134     93.2% of loss
preflop          598    +0.009      6.8%
flop/turn         61    +0.09        ~0% (profit)

=== by starting stack ===
<10bb            576    -0.067     81.0% of loss
10-20bb          246    -0.011     15.6%
20-40bb          178    +0.034      3.4% (profit)

=== by voluntary-in ===
vpip=Y           354    -0.153     94.3% of loss
vpip=N (folded preflop)  +0.029   profit

=== by hero's last action ===
pass (check/call) 237   -0.257     88.1% of loss
aggr (bet/raise)  130   +0.083     profit
fold              633   +0.024     profit

=== by all-in ===
all_in=Y         176    -0.191     59.6% of loss
all_in=N         824    -0.002     40.4%
```

Top-4 cross buckets `{position}/river/<10bb` = 179 hands (18% of total), but
43.5% of all ICM loss. **The leak is short-stack call-vs-shove pots that run
to showdown.**

One-sentence diagnosis: **when hero is <10bb, voluntarily in, gets to the river
(all-in), and the last action was a check/call, hero loses −0.26 ICM/hand**;
everywhere else hero is break-even or profitable. Hero is **calling too wide
against KillPhil's actually-tight shove range.**

## Why a fixed bot can do this (the range-mismatch story)

KillPhilMTT (per `data/shanky_profiles/KillPhilMTT.txt`) shoves *literally*:
`AA, KK, QQ, JJ, TT, 99, 88, 77, 66, AKs, AKo, AQs, AQo, AJs, ATs` — 94 combos
= **7.1% of all hands**. The resolver's belief-block at the leaf is uniform-over-
buckets (i.e., a much wider effective range). So when the resolver computes hero
equity vs the opponent's *assumed* range, it over-estimates equity, and hero
calls hands that are −EV vs KillPhil's actually-tight push.

GTO would defend appropriately to opponent ranges; we're defending against the
panel-average range, which is too wide for KillPhil and too narrow for STATION.

## Static-range fix attempt + panel verdict

Override fires when (a) hero start-of-hand stack < 10bb, (b) to_call > 0, and
(c) to_call ≥ 70% of remaining (i.e., "calling effectively commits me all-in").
In those spots: compute hero equity vs `killphil7` (94 combos), compare to
chip-EV pot odds `to_call / (pot + to_call)` (+ optional ICM padding).

Full panel, 500 paired hands per matchup, A=value-only ReBeL, B=A + override:

| Matchup | A | B | Δ (B−A) | noise ± |
|---|---|---|---|---|
| **KillPhilMTT** | −0.0431 | **−0.0161** | **+0.0270** | 0.0271 |
| timidtom | +0.0469 | +0.0430 | −0.0038 | 0.0044 |
| NIT | +0.0278 | +0.0249 | −0.0029 | 0.0093 |
| TAG | +0.0189 | +0.0262 | +0.0073 | 0.0160 |
| LAG | +0.1546 | +0.1717 | +0.0170 | 0.0281 |
| mixed | +0.1077 | +0.1111 | +0.0034 | 0.0245 |
| **STATION** | +0.2882 | **+0.2640** | **−0.0241** | 0.0300 |

Override fires 459 times across 3500 hand-pairs; mean equity 0.287 vs mean
threshold 0.335. **KillPhil leak shrank from −0.043 to −0.016 (63% closed)**;
**STATION cost ≈ equal magnitude (−0.024)**. Sum of deltas across panel
= +0.023 ICM, mean = **+0.003/hand — within noise**.

## Verdict — why this matters for the project, not just for KillPhil

1. The leak is real and **CONCENTRATED**, not diffuse: the diagnostic
   methodology (instrument loss per spot, slice by feature, identify
   concentration ratio) is the load-bearing artifact here, not the fix.
2. The leak is **range-dependent**: closing it for one opponent type opens it
   for another of opposite tightness. The resolver's panel-average calling
   range is locally Pareto: any static tilt that helps tight bots hurts loose
   bots by ≈ equal magnitude.
3. **No single static range fix can shift the panel.** Three independent
   tracks have now landed in the same place (12-lever sweep + policy-net
   warm-start + static range fix) — they all confirm that the value-only ReBeL
   is approximately the right *static* policy across the panel.
4. **Closing the leak requires opponent-range modeling.** Either within-hand
   action-sequence inference (StratFormer-style adaptive policy, the
   foundation-pivot direction in DECISIONS.md) or per-match adaptation
   (forbidden by the anonymity constraint).

This is consistent with the RunPod track's read on the same question and with
the Phase-2 plan: adaptive within-match layer subsumes the range-detection
that a hand-rolled "Option B" range-adaptive override would have built in
narrower form.

## Files / artifacts

- `scripts/rebel_diag_killphil.py` — per-hand instrumented harness vs KillPhilMTT;
  slice tables by street/stack/position/vpip/allin/last-action/blind-level
  plus cross-bucket top-10. Re-usable for any matchup.
- `evals/diag_killphil_1k.{jsonl,json}` — 1000-hand record + summary used here.
- `scripts/rebel_gate2.py` — adds `--short-stack-fix-{a,b}`, `--ss-range`,
  `--ss-bb-thresh`, `--ss-commit-thresh`, `--ss-icm-tighten`, `--matchups`.
  Shove ranges: `tight16` (generic 6-max ~16%), `killphil7` (KP allin ~7%).
  *Labelled: static fix, symmetric trade-off, superseded by adaptive layer.*
- `scripts/rebel_train_policy_net.py` — ReBeL policy-net trainer (masked-KL on
  schema-2 shards). Fixes: `0·(-inf)=nan` in masked-KL on illegal slots;
  save-on-improve checkpoint persistence.
- `src/nlhe/subgame_solver.py` — `SubgameSolveContext.root_policy_prior`
  field; hero-root warm-start replacement in `_run_cfr` (L1-matched to
  blueprint warm-start mass).
- `runs/rebel_policy_net_m3box2.pt` — gitignored (`*.pt`); recipe in train
  script.

## Next

Contabo's Phase-2 role per STATUS: build the trained adaptive policy (per the
foundation-pivot in DECISIONS.md). The adaptive policy is the proper home for
the range-detection capability the static fix could only fake; the RunPod
track is generating the population corpus with rewards + train/held-out split.
