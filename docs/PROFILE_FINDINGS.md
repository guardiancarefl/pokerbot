# Solver bottleneck profile findings

**Date:** 2026-05-24
**Branch:** phase4f-bucket-cache (later partially merged: scripts only, no code changes)
**Config:** configs/six_max_profile.yaml (5 iters, hidden=[32,32], 20 traversals, CPU forced)

## Headline result

In a 5-iteration CPU profile of the DCFR 6-max solver:

- **47.7 seconds total runtime**
- **44.1 seconds (92.5%) in `Abstraction._bucket_of_uncached`**
- **40.9 seconds (85.7%) in `scipy.stats.wasserstein_distance`**
- **0.8 seconds (1.8%) in advantage-net training**

The solver is not CPU-bound on neural-net work. It is bound on
Earth Mover's Distance computation in the card abstraction lookup.

## Why naive caching does not help

`Abstraction.bucket_of` is deterministic by design: same (hero, board)
always produces the same bucket. An LRU cache on it would seem to be the
obvious fix.

It is not. Measured by direct instrumentation:

- 3,440 `bucket_of` calls across the 5-iter profile run
- 3,348 unique `(sorted(hero), sorted(board), runouts)` combinations
- **2.7% hit rate**

The work is genuinely distinct queries. There is no redundant computation
hiding inside `bucket_of` itself.

A patch adding `Abstraction`-level caching was tried, validated to work
correctly via unit tests (5,000× speedup on repeated identical queries),
and **reverted** when the in-training hit rate measurement showed
~0% impact. The patch added cache-lookup overhead with no real benefit
because the encoder layer above already does the same caching at the
correct level — see `src/nlhe/infoset6.py::_get_bucket`.

## The encoder cache works but resets per-iteration

`InfosetEncoder6Max._bucket_cache` is the right caching layer. In the
profile run, it contained 592 unique entries at end-of-training. But
`bucket_of` was still called 3,440 times — meaning the encoder cache
is being cleared between iterations (per docstring: "Reset its bucket
cache between iterations to bound memory").

3440 / 592 ≈ 5.8 — every unique encoder query is hit ~6 times across
the 5 training iterations, with the cache being thrown away each iter.

**If the encoder cache were not reset, we would save ~5x bucket_of
calls for free with zero algorithmic change.** This is the cheapest
plausible optimization and should be the next investigation.

## Real optimization paths, in order of effort/payoff

### 1. Don't reset the encoder cache between iters (cheap)
**Estimated win:** ~5x bucket_of calls cut, possibly ~80% of profile
runtime returned.
**Risk:** memory growth. The cache size in the profile was 592 entries
after 5 iters; at this rate a full 3400-iter run could reach ~400k
entries. With ~16 bytes per entry (key + int) that's ~6 MB, negligible.
A bounded LRU would cap it cleanly.
**Effort:** half a day to implement, test, profile, and confirm
correctness across the existing test suite.

### 2. Precompute bucket lookup table (Pluribus-style)
**Estimated win:** essentially eliminate `bucket_of` cost. Lookups
become O(1) dict reads.
**Risk:** offline compute time, disk space, and we need to settle on
the canonical key format (suit-isomorphism matters for postflop).
**Effort:** 1-2 days. Worth it once Phase 4 / blueprint training
becomes the dominant compute cost.

### 3. Custom NumPy EMD
**Estimated win:** 10-20x per-call speedup on `wasserstein_distance`.
**Risk:** the existing implementation is correct and battle-tested in
scipy. Replacing it introduces numerical-correctness risk.
**Effort:** 2-3 days including correctness validation against scipy.

The current recommendation is to do (1) first, measure, and only escalate
to (2) or (3) if the residual bottleneck still dominates.

## Artifacts preserved

- `scripts/profile_solver.py` — cProfile harness, CPU-forced
- `configs/six_max_profile.yaml` — small-but-representative 5-iter config
- `profiles/profile_baseline_5iter_cpu.prof` — raw cProfile output
- `profiles/profile_baseline_5iter_cpu.txt` — text version of the profile run

To re-profile after any change:

    python -m scripts.profile_solver --config configs/six_max_profile.yaml

Future optimization PRs should include before/after profile output in
the commit message, sourced from this script.
