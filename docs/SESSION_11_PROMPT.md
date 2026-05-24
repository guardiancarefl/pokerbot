# Prompt for Session 11 (paste this verbatim at the start of the next conversation)

---

Continuing the pokerbot project. Session 11.

**Status:** Phase 4f DCFR-linear overnight kicked off in Session 10 at 01:43 (UTC, 2026-05-24), targeting 3400 iters at ~8-9s/iter on the rented RunPod RTX PRO 4000 Blackwell. At Session 10 close (~04:00) it was at iter ~617/3400 with checkpoints landing every 200 iters. By the time Session 11 starts the overnight should be finished or near-finished — first task is to check status, eval the result, and decide where to go next.

Session 10 validated three things on the existing iter-200 DCFR checkpoint at 5000 hands:
- DCFR-200 beats vanilla-200 at matched compute: **+0.0134 ± 0.0027 (4.9 sigma)**. Reproduced across two seeds.
- DCFR-200 still climbing iter 100 -> 200: **+0.0108 ± 0.0028 (3.9 sigma)**.
- Random calibration: **+0.1567 ± 0.0034 (46 sigma)**. Every future bot must clear this.

And surfaced one critical caveat: **CUDA training is non-deterministic at the ~0.01 ICM scale**. Two identical-config DCFR runs at iter 200 produce policies that differ by ~0.01 ICM. Future effect-size claims must clear this floor by a wide margin to be meaningful.

**Runtime:** Rented RunPod RTX PRO 4000 Blackwell. Project at /workspace/pokerbot on the pod. Repo at github.com/guardiancarefl/pokerbot. Branches active on origin:
- phase4f-dcfr-6max (main DCFR line, eval harness, overnight driver, doc updates)
- phase4f-reinit (Brown 2019 reinit, 13 tests, staged config, **not launched**)
- phase4f-bucket-cache (profile harness + diagnostic doc, no source changes)
- phase4f-encoder-cache-persist (LRU attempt + diagnostic doc, no source changes)

**Session 11 priorities:**

1. **Check overnight status and eval.**
   Quick status:
       tmux ls
       tail -5 /tmp/dcfr_overnight.log
       ls runs/six_max_*phase4f_dcfr_linear_overnight*/checkpoints/ | sort -V
   If finished, run the morning eval driver:
       git checkout phase4f-dcfr-6max
       ./scripts/eval_overnight.sh
   That auto-discovers the latest overnight run and evals every checkpoint against a stable opponent pool (vanilla-100/200/400, dcfr-100/200, random). Output: per-checkpoint JSON in evals/, plus a summary table.

2. **Read the DCFR-vs-iter curve and decide next direction.**
   - **If DCFR keeps climbing past iter 2000** (diff vs vanilla-200 still trending up at 5000 hands): the algorithm hasn't plateaued at the scale we're training. League play (Pillar 5) becomes the higher-priority next investment. Session 10 sketched the design: CheckpointRegistry + LeaguePool, with the Policy protocol from scripts/eval_pool.py as the integration contract. Implementation is the next overnight-prep target.
   - **If DCFR plateaus before iter 1500** (diff vs vanilla-200 flat or declining): reinit becomes the immediate next overnight. Branch phase4f-reinit is fully tested (13/13 tests pass, 295 total pass). Launch with:
         git checkout phase4f-reinit
         python -m scripts.train_6max --config configs/six_max_phase4f_dcfr_reinit_overnight.yaml
   - **If results are ambiguous** (e.g. DCFR climbs but slowly, or curve is noisy): get more hands on the strongest checkpoint. 10,000 hands vs same pool. The CUDA noise floor (~0.01 ICM) is a real lower bound on effect sizes; if our best checkpoint's lead over vanilla-200 is in the 0.005-0.015 range it could be noise, and we need more data before deciding.

3. **Average-strategy approximation** (deferred from Sessions 9 and 10). Either add a strategy net per seat to PlayerNetworks6Max (HUNL pattern) or implement on-the-fly average via Brown 2019's trick. Required before any deployment-quality eval. Not blocking but earning interest.

4. **Precomputed bucket lookup table** (the remaining plausible perf win from PROFILE_FINDINGS_LRU_ATTEMPT.md). Both runtime caching attempts failed in Session 10. Offline precomputation of HoleClass x board -> bucket map is the path forward. ~6-12 hours of one-time compute; runtime bucket_of becomes O(1) hash lookup. Pick this up as a focused session, not a side-quest.

**Key files to reference:**
- docs/STATUS.md - current state (rewritten in Session 10 to reflect actual phase)
- docs/SESSION_LOG.md - full Session 10 entry covers everything that happened overnight 2026-05-24
- docs/PROFILE_FINDINGS.md + docs/PROFILE_FINDINGS_LRU_ATTEMPT.md - two negative-result writeups; second one has the "real fix path is offline precompute" finding
- scripts/eval_pool.py - multi-baseline pool evaluator with the Policy protocol
- scripts/eval_overnight.sh - morning driver, auto-discovers latest overnight, idempotent
- scripts/profile_solver.py + configs/six_max_profile.yaml - diagnostic harness, re-runnable for any perf claim
- tests/test_reinit.py - 13 tests on the reinit branch; if launching reinit overnight, these should pass first
- configs/six_max_phase4f_dcfr_reinit_overnight.yaml - staged reinit config, same hyperparams as DCFR-only otherwise

**Workflow rules carrying over from Session 10:**
- **Direct wall-clock before/after for any perf claim.** cProfile percentages are diagnostic, not ground-truth. The 47.7s "baseline" that motivated two reverted Session 10 patches was a cProfile-wrapped measurement; real baseline was ~30s. Always measure outside cProfile.
- **Branch-per-experiment.** Each independent direction (reinit, bucket cache, encoder LRU) got its own branch. Main DCFR work stayed on phase4f-dcfr-6max. Kept the optimization detours from polluting the main line.
- **Heredoc + Python patch scripts work cleanly if and only if there are no triple-quoted docstrings inside.** String concatenation with \n escapes is the reliable pattern. Use cat -A to see exact whitespace bytes when anchors silently fail - invisible blank lines matter.
- **Negative results are results.** Two of Session 10's experiments (bucket cache, encoder LRU) didn't work and are documented as such. Branches preserved on origin as evidence; source changes not merged.
- **Pool eval is the standard yardstick now.** Every checkpoint from this point on gets scored against the stable opponent pool. Don't go back to ad-hoc 2-checkpoint evals.
- **CUDA noise floor is ~0.01 ICM.** Effect-size claims below this scale are not meaningful. Demand wider margins.
- **User paces sleep/food invisibly between messages.** Don't suggest breaks or ask about stopping. Just keep working.

Please read the project knowledge docs (STATUS.md as entry point, then SESSION_LOG.md Session 10 entry for context, then PROFILE_FINDINGS*.md for the perf-attempt history). Confirm state matches what STATUS says. Then start with item 1: check the overnight status and run the morning eval.
