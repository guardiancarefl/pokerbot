# Layer 4 — Empirical Verdict (deferred for now)

**Run date:** 2026-05-29
**Measurement:** scripts/measure_layer4_cheap.py at n=23/30 (killed early after
verdict was clear; full log at /tmp/layer4_full.log)
**Config:** match_len=15, n_iterations=100, base_seed=12345, archetype rotation
(NIT/TAG/LAG/STATION/MANIAC), hero seat rotation
**Blueprint:** runs/dcfr_anchor_2000/checkpoints/ckpt_iter_2000.pt (k=200, 2000 iters)

## Result
- Raw path lift vs baseline: ~-0.27 per match (ICM equity, pool units)
- Archetype path lift vs baseline: ~-0.23 per match
- Both paths net-negative; high single-match variance dominates
- Layer 4 only fires on ~30-40% of matches (confidence threshold gating)
- Biggest losses against STATION opponent — bias function over-aggressive against
  high-VPIP-low-aggression style

## Architecture status: GREEN
- C1a/b/c/d-1 committed, 41/41 tests passing
- Bit-identity invariant locked at confidence=0
- All plumbing reusable for future Layer 4 iterations
- The "what failed" was decision logic + parameter setting, not architecture

## Why deferred
Layer 4 is the lowest-leverage of available improvements. Bigger blueprint,
variance-reduced CFR, and dynamic action abstraction (RL-CFR) all rank higher
per recent literature. Layer 4 may become meaningful after those improvements
land; the infrastructure stays in place for that revisit.

## Conditions under which to revisit
1. After a stronger blueprint exists (k_postflop >= 1000, ICM-trained)
2. After match_len >= 40 measurements show Layer 4 can accumulate sufficient
   confidence within a match
3. After α_C1 ablation (try α=1.5 and α=1.3 to see if tighter cap reduces variance)
4. If range tracking is built (Phase 5) — Layer 4's category errors against STATION
   would likely be fixed by hand-range information the current system lacks

## Files preserved
- src/nlhe/within_match.py (C1a)
- src/nlhe/bias_configs.py (C1b)
- src/nlhe/layer4_factory.py (C1d-1)
- src/nlhe/subgame_policy.py (C1c bias_factory hook)
- src/nlhe/subgame_leaf.py (C1c per-seat dispatch)
- tests/test_within_match.py, tests/test_bias_configs.py,
  tests/test_c1c_integration.py, tests/test_layer4_factory.py (41 tests)
- scripts/measure_layer4_cheap.py (measurement script)
