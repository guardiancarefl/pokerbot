# Encoder LRU bucket cache: tried, didn't help

**Date:** 2026-05-24 (immediate follow-up to docs/PROFILE_FINDINGS.md)
**Branch:** phase4f-encoder-cache-persist (no source changes merged; this doc only)

## What we tried

docs/PROFILE_FINDINGS.md projected ~5x bucket_of call reduction from
letting the encoder bucket cache persist across iterations instead of
clearing it every iter. Implemented as a bounded LRU:
- `InfosetEncoder6Max._bucket_cache` → `collections.OrderedDict`
- `bucket_cache_size: int = 100_000` config field
- `_get_bucket`: move_to_end on hit, popitem(last=False) on insert past cap
- `solver6.train()`: remove the per-iter `self.encoder.reset_cache()` call

## Direct measurement (5 iters CPU, configs/six_max_profile.yaml)

Three runs each, wall-clock from `solver.train()` start to finish.
No cProfile wrapper this time — cProfile's per-call overhead masks
real differences.

| run | LRU      | baseline |
|-----|----------|----------|
|  1  | 29.7s    | 30.9s    |
|  2  | 29.8s    | 29.8s    |
|  3  | 32.2s    | 29.6s    |
| **mean** | **30.6s** | **30.1s** |

LRU is 1.6% *slower* in the mean. Run-to-run noise (~2.5s) is larger
than any effect. **No measurable speedup.**

## Why the projection was wrong

Direct instrumentation during the LRU run:
- `_get_bucket` calls: 6,844
- `bucket_of` calls (actual cache misses): 3,348
- LRU hit rate: 51.1%

The cache did cut bucket_of calls roughly in half. But:
- Saved EMD work per cache hit: small (each `bucket_of` is fast on a
  warm scipy state; the slow scipy initialization runs once and is
  amortized across all calls).
- Cost added per `_get_bucket` call: OrderedDict.move_to_end on hit,
  len() check + possible popitem on miss. These are O(1) but Python-
  function-call overhead is real.

Net: the two roughly cancel.

The earlier 47.7s number in docs/PROFILE_FINDINGS.md was a
cProfile-wrapped measurement. Direct training (no cProfile) was
~30s baseline all along. The "92.5% of runtime in bucket_of" stat
is true *under cProfile* — cProfile's function-call timing inflates
hot functions vastly more than cold ones, exaggerating the EMD
share. The reported share has limited bearing on real wall-clock.

## What this means for the next optimization step

docs/PROFILE_FINDINGS.md ranked three optimization paths:
1. ~~Don't reset encoder cache (this attempt)~~  → no win
2. Precompute bucket lookup table (Pluribus-style)  → still plausible
3. Custom NumPy EMD                                 → still plausible

But the right reframing now is: **at the current scale of training
(traversals_per_iter=20, 5 iters, ~3k unique queries), the solver's
wall-clock is dominated by per-query EMD cost on genuinely distinct
queries.** Caching can't help that; only attacking the per-query cost
itself can.

For a real overnight training run (traversals_per_iter=150, 3400 iters)
the unique-query count will be much larger and bucket_of becomes the
dominant cost in *absolute* terms even though cProfile already says so
for the small config.

**Path 2 (precomputed lookup table) is the right next investigation.**
Bucket the entire HoleClass × board space offline once (probably 6-12
hours of one-time compute), save as a dict next to the abstraction.pkl,
load at startup. Runtime `bucket_of` becomes O(1) lookup. No cache, no
LRU, no overhead — just a single hash table read.

Estimated win: the ~38s of EMD per 5 iters drops to ~0s. That's the
real ~10× wall-clock speedup the profile suggests is available.

## Recommendation

Do NOT merge phase4f-encoder-cache-persist. The patch is correct and
tested but has no measurable benefit. Leave the branch on origin as
a record of the attempt; don't pollute main with neutral changes.

Tonight's optimization-direction lesson: **always measure direct
wall-clock before and after a perf change, with the same config and
no profiler attached.** cProfile numbers are diagnostic, not
ground-truth, and chasing cProfile percentages without wall-clock
validation is exactly how you ship neutral patches that look like wins.
