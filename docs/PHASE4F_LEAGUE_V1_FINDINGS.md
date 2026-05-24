# Phase 4f League v1 — Findings

**Run:** `runs/six_max_20260524_090852_phase4f_dcfr_league_overnight/`
**Killed at:** iter ~1700/3400 (Session 11, 2026-05-24)
**Eval reference:** `evals/league/league-{200,800,1400}_vs_pool_5000.json`

## Config under test

Built on `phase4f-league` (commits 88c28fd → 06c9d8d).

| field | value |
|---|---|
| `cfr_variant` | `linear` (DCFR) |
| `n_iterations` | 3400 (killed early) |
| `league_mix` | 0.30 |
| `league_sample_strategy` | `uniform` |
| `league_registry_path` | `configs/league/registry.json` |
| `league_tag_filter` | `[dcfr]` |
| Pool composition | 17 `dcfr-overnight-*` (iter 200→3400) + 2 `dcfr-shake-*` (iter 100,200) |
| Hidden dim | `[256, 256]` |
| Hyperparams | matched to `phase4f_dcfr_linear_overnight` (3400 iter, traversals=150) |

## Three checkpoints scored

Pool: `vanilla-100/200/400`, `dcfr-100/200`, `dcfr-overnight-3000`, `random`. Same six baseline opponents as the DCFR pool, plus `dcfr-overnight-3000` (peak DCFR checkpoint) as the strength benchmark.

```
                  vanilla-100  vanilla-200  vanilla-400  dcfr-100  dcfr-200  dcfr-3000  random
league-200        +0.0098      -0.0103      -0.0067      -0.0020   -0.0148   -0.0396    +0.1479
league-800        +0.0273      +0.0189      +0.0120      +0.0173   +0.0058   -0.0210    +0.1613
league-1400       +0.0273      +0.0218      +0.0223      +0.0193   +0.0153   -0.0203    +0.1655
```

vs the matched-iter DCFR-only baseline (from `dcfr-overnight-*` evals):

| matchup           | DCFR-200 | League-200 | DCFR-800 | League-800 | DCFR-1400 | League-1400 |
|---                |---       |---         |---       |---         |---        |---          |
| vanilla-200       | +0.0134  | -0.0103    | +0.0144  | +0.0189    | +0.0240   | +0.0218     |
| vanilla-400       | +0.0079  | -0.0067    | +0.0214  | +0.0120    | +0.0240   | +0.0223     |
| dcfr-200          | -0.0099  | -0.0148    | +0.0073  | +0.0058    | +0.0224   | +0.0153     |
| **aggregate (4)** | **+0.0055** | **-0.0085** | **+0.0171** | **+0.0110** | **+0.0228** | **+0.0197** |

League aggregate trails DCFR aggregate at every measured iter.

## Conclusions

**1. v1 league config is suboptimal at matched compute.** A 30% uniform mix slows convergence at every measurement point compared to pure self-play DCFR. The early gap is largest (−0.014 aggregate at iter 200); the gap narrows but doesn't close by iter 1400.

**2. League is still climbing, just slower.** vs `dcfr-overnight-3000`: −0.0396 (iter 200) → −0.0210 (iter 800) → −0.0203 (iter 1400). The big improvement was 200→800; 800→1400 was small (within seed noise). League's marginal-return curve has the same concave shape DCFR's has, but starts lower and rises slower in this config.

**3. League v1 has not produced a stronger bot than DCFR-3000.** At iter 1400, league loses to `dcfr-overnight-3000` by 9.9σ. Linear extrapolation from the trajectory does not project a crossover within the remaining 2000 iters. Killed at iter ~1700 to conserve GPU for better-targeted experiments.

## What v1 does NOT rule out

The league *architecture* (LeaguePool, CheckpointPolicy, opponent override threading) is sound — the implementation works. What's failing is this specific *config*. Configs worth testing next:

- **Lower mix (0.10–0.15).** v1's 30% disrupts self-play convergence more than league diversity compensates. The PSRO literature typically uses 0.15–0.20.
- **Recency-weighted sampling instead of uniform.** v1 sampled overtrained dcfr-3400 at the same rate as peak dcfr-3000. The new `dcfr-peak-3000` alias (tag `peak`) enables single-checkpoint training against the strongest available anchor.
- **Initialize from `dcfr-overnight-3000` weights instead of fresh.** Use league as fine-tuning on top of DCFR's peak rather than retraining from scratch. Architecturally distinct from v1 (would require a `--init-from` flag on `train_6max.py`).

## Recommended v2 config

```yaml
league_mix: 0.15
league_sample_strategy: recency
league_tag_filter: [peak]
league_recency_halflife: 5.0
```

This pairs the lower mix with the `dcfr-peak-3000` tagged checkpoint, training only against the single peak. ~7 hours on GPU. Hypothesis: this will at least match DCFR-3000 and likely exceed it through diversity-against-noise gains.

## Status

- v1 league run killed early at ~iter 1700.
- Three checkpoint evals committed: `evals/league/league-{200,800,1400}_vs_pool_5000.json`.
- Registry alias `dcfr-peak-3000` (tag `peak`) added in commit 06c9d8d, ready for v2.
- v2 config not yet written; will be created as `configs/six_max_phase4f_dcfr_league_v2.yaml` when v2 is queued.
