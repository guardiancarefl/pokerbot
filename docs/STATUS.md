# Project Status

**Last updated:** 2026-05-24 (Session 10 close)
**Current phase:** Phase 4f (6-max DCFR blueprint training).

## Done

### Foundations & infrastructure
- Architecture designed (four-layer stack, opponent anonymity principle, ICM-adjusted training)
- Format target chosen (6-max NLHE SNG, top-3 equal payout)
- Engine selected (OpenSpiel; reference Deep CFR thin-wrapped for Phase 1)
- Runtime: Contabo VPS (Phase 1–3) → rented GPU (Phase 4+). Currently training on rented RunPod RTX PRO 4000 Blackwell.
- GitHub repo at github.com/guardiancarefl/pokerbot with full session-by-session history.

### Phase 1 (Leduc) — closed Session 2
- Leduc Deep CFR pipeline validated. Exploitability 1187 → 434 mbb/g across 25 iters on a deliberately undersized config. Pipeline proved correct; not a Leduc record.

### Phase 2 (HUNL) — closed Sessions 3–8
- HUNL game representation in OpenSpiel validated; ICM not yet applicable (heads-up cash mode).
- src/nlhe/abstraction.py: EMD-based card abstraction, 200 buckets/street, k-medoid clustering on equity distributions. Production artifact at runs/abstraction_20260521_223018_retrofit/abstraction.pkl.
- src/nlhe/actions.py: discretized bet sizes {check/fold, call, 0.33pot, 0.66pot, 1pot, 2pot, all-in} + off-tree action translation.
- HUNL Deep CFR solver, advantage net + average-policy net per player, reservoir buffer, DCFR weighting.
- scripts/eval_vs_slumbot.py: Slumbot API harness with bb/100 over N hands.
- **HUNL bot validated at +78.35 bb/100 vs Slumbot over 10,000 hands.** This is the canonical "the pipeline works on real poker" milestone.

### Phase 3 (archetypes + biased policy) — closed Session 7
- src/nlhe/archetypes.py: NIT / TAG / LAG / STATION / MANIAC archetype profiles + per-street equity quantile thresholds + OpponentPool.
- src/nlhe/biased_policy.py: blueprint policy biasing for archetype-flavored play.
- src/nlhe/policy_adapter.py: maps between OpenSpiel actions, Slumbot tokens, discrete action enum.

### Phase 4 (6-max scaffolding + first blueprint) — Sessions 8–9
- src/nlhe/icm.py: Malmuth-Harville ICM equity calculation.
- src/nlhe/icm_returns.py: chip-returns → ICM-equity-delta for terminal nodes.
- src/nlhe/infoset6.py: 236-dim feature encoding with dealer-aware position for repeated_poker games.
- src/nlhe/networks6.py: PlayerNetworks6Max — 6 advantage nets, 6 buffers, container-owns-state pattern. State_dict serializes all 6 in one call.
- src/nlhe/trajectory6.py: pure trajectory walker (separated from CFR for clean unit testing).
- src/nlhe/cfr6.py: traverse_6max + CFR6MaxContext. External-sampling CFR for 6-max with ICM at terminals.
- src/nlhe/solver6.py: DeepCFR6MaxSolver + TrainConfig6Max. 6-seat traverser cycling, bit-identical checkpoint round-trip.
- scripts/train_6max.py: training entry point with config.json + metrics.json + checkpoint cadence.
- scripts/eval_6max_self_play.py: 2-checkpoint head-to-head with ICM-equity-delta scoring.
- src/nlhe/stack_sampler.py + game_strings.py: tournament-state sampling for variable blind levels and active counts.

### Phase 4f (DCFR + reinit + diagnostics) — Session 10
- src/nlhe/solver6.py: DCFR (linear + discounted) variants ported to 6-max solver. `cfr_variant` + `dcfr_exponent` config fields.
- configs/six_max_phase4f_dcfr_overnight.yaml: full DCFR-linear overnight (3400 iters).
- configs/six_max_phase4f_dcfr_reinit_overnight.yaml: DCFR-linear + reinit every 4 per-seat iters. Staged on phase4f-reinit branch, not launched.
- scripts/eval_pool.py: multi-baseline pool evaluator. `Policy` protocol (CheckpointPolicy, UniformRandomPolicy). 5000 hands/matchup default.
- scripts/eval_overnight.sh: morning driver that auto-evals every checkpoint in the latest overnight run against a stable opponent pool. Idempotent.
- tests/test_reinit.py: 13 tests for periodic advantage-net reinitialization (Brown 2019 sec 4.3) on branch phase4f-reinit.
- scripts/profile_solver.py + configs/six_max_profile.yaml: CPU-forced cProfile harness. Re-runnable for before/after measurements.
- profiles/profile_baseline_5iter_cpu.{prof,txt}: baseline diagnostic measurement.
- docs/PROFILE_FINDINGS.md + docs/PROFILE_FINDINGS_LRU_ATTEMPT.md: two negative-result writeups documenting why two perf attempts (Abstraction.bucket_of cache, encoder LRU) didn't help.

### Session 10 validated results
- DCFR-200 beats vanilla-200 at matched compute: **+0.0134 ± 0.0027 (4.9σ) over 5000 hands**. Reproduced across two seeds.
- DCFR-200 still climbing iter 100 → 200: **+0.0108 ± 0.0028 (3.9σ)**, suggesting the algorithm hasn't plateaued where vanilla did at iter 200.
- Random calibration anchor: **+0.1567 ± 0.0034 (46σ)**. All future bots must clear this by ≥0.15 ICM.
- CUDA noise floor: **~0.01 ICM**. Two identical-config DCFR runs at iter 200 differ by ~0.01. Future effect-size claims must clear this.

## In progress

- **DCFR linear overnight** at iter ~617/3400 as of Session 10 close. ~6-7 hours remaining. Checkpoints landing every 200 iters into runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/. tmux session: dcfr_overnight.
- All Session 10 work pushed to origin across four branches: phase4f-dcfr-6max (DCFR + eval + overnight driver), phase4f-reinit (reinit code + tests + config), phase4f-bucket-cache (profile harness + first findings doc), phase4f-encoder-cache-persist (LRU attempt + second findings doc).

## Next up (Session 11)

1. **Eval the overnight when it finishes.** `scripts/eval_overnight.sh` against the completed run. Shape of the DCFR-vs-iter curve decides next direction.
2. **Decision tree based on overnight shape:**
   - If DCFR keeps climbing past iter 2000 → league play (Pillar 5) is next focus. CheckpointRegistry + LeaguePool implementation.
   - If DCFR plateaus before iter 1500 → launch configs/six_max_phase4f_dcfr_reinit_overnight.yaml. Reinit branch is tested and ready.
3. **Average-strategy approximation** (deferred from Session 9). Either per-seat strategy net or on-the-fly average. Required before any deployment-quality eval.
4. **Precomputed bucket lookup table** (the path-2 finding from PROFILE_FINDINGS_LRU_ATTEMPT.md). Estimated 6-12 hours one-time offline compute; runtime bucket_of becomes O(1). Worth picking up as a focused session.
5. **Subgame solving prototype** (Pillar 3 from the original architecture). Still on the roadmap, multi-week effort. Don't start until blueprint training is mature.

## Known issues / open questions

- **CUDA training non-determinism at ~0.01 ICM scale.** Same code, same seed, different GPU runs produce policies that differ by ~0.01 ICM. Effect-size claims below this floor are not meaningful.
- **Encoder bucket cache is cleared per-iter,** giving ~2.7% hit rate during training. The LRU fix (Session 10) didn't help in wall-clock because cache machinery cost ≈ EMD work saved at current scale. Real fix path is offline precompute, not runtime caching. Documented in PROFILE_FINDINGS_LRU_ATTEMPT.md.
- **42 bought-bot profile format still unidentified.** Defer until Phase 5 / league play.
- **Reinit branch phase4f-reinit is unmerged.** Decision waits for overnight DCFR plateau data.
- **Two small cleanups from Sessions 2 and 4 still pending:** train_leduc.py writes config.json/metrics.json into checkpoints/ instead of run-dir root; _build_game_state_view has a minor privacy inconsistency. Neither blocks anything.

## Decisions deferred

- **GPU provider for next overnight** (RunPod current, Vast.ai / Vultr GPU as alternatives). Decide when current pod expires or we need different specs.
- **Specific card abstraction granularity for any future re-train.** Current is 20 preflop / 200 postflop. Could go finer (Pluribus-style ~10,000 per street) once precomputed lookup makes runtime cost O(1).
- **Archetype port to 6-max.** archetypes.py is HUNL-focused; 6-max positional play differs significantly. Needed before bot-vs-archetype evals.
- **Slumbot replacement for 6-max evaluation.** Slumbot is HUNL-only. Options for 6-max eval: self-play tournaments, bot-vs-archetype tournaments, head-to-head against a Nash push/fold ICM solver for the late game.
- **Whether to merge any Session 10 perf-attempt branches.** Recommendation: don't merge phase4f-bucket-cache or phase4f-encoder-cache-persist source changes. The diagnostic docs (PROFILE_FINDINGS*.md) and tooling (profile_solver.py, six_max_profile.yaml, evals/) should land on main eventually as part of a "diagnostics" merge.

## Session log

See docs/SESSION_LOG.md for per-session detail. Most recent: Session 10 (2026-05-24 overnight).
