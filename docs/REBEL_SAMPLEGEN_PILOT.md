# Phase 2 sample-gen PILOT — measured rate + ETA (HOLD for hardware decision)

Wiring validated end-to-end with the k200 artifacts: load blueprint → self-play
to a decision PBS → `build_subgame_tree` → `evaluate_leaves` → `solve_subgame`
→ (encode PBS, CFR root value) sample. Script: `scripts/rebel_samplegen_pilot.py`.

## The dominant cost is leaf evaluation, NOT the CFR solve

Per-sample stage breakdown (depth-3 subgame, single core):

| stage | cost |
|---|---|
| build_subgame_tree | ~0.00 s |
| **evaluate_leaves (rollout)** | **0.5–20 s** (mode/M dependent) |
| solve_subgame (150 CFR iters) | ~0.05 s |

Leaf-eval cost by mode (depth 3, 38 infosets): PROFILE_SAMPLE M=1 = 0.50 s,
M=5 = 2.08 s; BEST_RESPONSE M=1 = 5.04 s, M=5 = 20.6 s. The CFR solve is trivial.
**This is exactly what the PBS value net fixes** — it turns leaf-eval into a ~ms
GPU forward, so post-net generation rounds run ~100× faster on the leaf and are
then bounded by tree-build + solve (~0.1 s).

## Measured rate (real 13.6-core slice)

13 CPU workers (CPU-per-worker to avoid single-GPU serialization), depth 3,
PROFILE_SAMPLE M=1 (the round-1 bootstrap leaf):

    1187 samples in 81.3 s  →  14.6 samples/s  →  ~52,500 samples/hour

## ETA to a usable (not paper-grade) value-net sample count

Target for a *usable* 6-max value net to test at GATE 2: ~1–5M samples (paper-
grade ReBeL/DeepStack used far more; we only need enough to beat-or-not k200).

| samples | this box (13.6 cores, ~52.5k/h) | 64-core box (~4.7×) | 4×64-core (~19×) |
|---|---|---|---|
| 1M  | ~19 h        | ~4 h   | ~1 h   |
| 2M  | ~38 h (1.6d) | ~8 h   | ~2 h   |
| 5M  | ~95 h (4.0d) | ~20 h  | ~5 h   |
| 10M | ~190 h (7.9d)| ~40 h  | ~10 h  |

(Linear in cores; generation is embarrassingly parallel across independent
boxes. Round-1 bootstrap rate shown; later value-net-leaf rounds are faster.)

## Conclusion / recommendation (HOLD)

On this 13.6-core slice, even 1M samples is ~a day and 5M is ~4 days — so the
user's plan is right: **generate on bigger / multiple CPU boxes, reserve this GPU
box for Phase-3 training**. Samples are designed shardable/portable (independent
per-worker shards, mergeable). HOLDING for the hardware/go decision before any
full generation run.

## Remaining before a full run (small)
  - Sample persistence: encode PBS (public block + 6×k belief, see
    `docs/REBEL_PBS_6MAX.md`) + target, write portable shards. The pilot measured
    the dominant search cost; serialization adds ~ms/sample.
  - Depth choice: depth 3 ≈ partial multi-street lookahead; depth 4–5 (fuller
    lookahead, the stated goal) is slower (more leaves) — a Phase-2/3 tradeoff to
    set with the value net in the loop.
