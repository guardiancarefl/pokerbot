# Session 21 — 2026-05-26

Focus: **Build sub-step 6 — the Level-3 pool ablation (the strength go/no-go) —
through Stage 6-C.** The final design proposal of the project landed (with a
variance-grounded revert-gate clarification), then the parallel harness (6-A) and the
3-challenger CLI + locked verdict (6-B), validated end-to-end by a clean smoke (6-C).
Sub-step 6 is ready for the real run (Stage 6-D), pending a quiet-box per-solve
re-measurement.

## What was done

- **Sub-step 6 design proposal** (`455df14`, `docs/SUBSTEP_6_DESIGN.md`, ~2250 words).
  3-way comparison (Q11 pre-commit), hand-level multiprocessing with SHA256 per-hand
  seeding + CRN, and a four-branch verdict. **Major methodology correction surfaced:**
  the harness metric is **ICM-equity-delta, not bb/100**, with stderr ~0.0026/matchup
  at 5,000 hands (a 0.02 effect → σ8) — far tighter than the prompt's bb/100
  assumption, so the thresholds were re-grounded to ICM-equity-delta.
- **Revert-gate variance clarification** (`0e4ab0e`). The BR-vs-PROFILE gate now gates
  on **measured σ(L_BRvsP) ≥ 1.5**, not a fixed +0.001 absolute (at the conservative
  unpaired pooled stderr ~0.0016, +0.001 is only ~0.6σ — would "revert" on noise). New
  branch **PASS_BR_EQUIVALENT_TO_PROFILE** (architecture lifts, BR not statistically
  distinguishable from PROFILE → recommend PROFILE) distinct from a confident revert.
- **Stage 6-A — parallel harness** (`2ad48c1`, `scripts/eval_pool_ablation.py`).
  `hand_seed` = SHA256-mod-2³¹ (index-derived, order-independent); `_play_one_hand`
  with split RNG (chance_rng shared across challengers → CRN-paired deal/board/seats;
  policy_rng diverges); `ProcessPoolExecutor` stride-shards, **fail-loud** on worker
  exception; reduce → per-matchup diff/stderr/σ + CRN-paired lift. **Worker-count
  bit-identity** (aggregate sums hands in sorted order → identical for workers ∈
  {1,4,8}). 9 tests.
- **Stage 6-B — 3-challenger CLI + verdict** (`b0dfa11`). `compute_verdict` (all five
  locked branches), `run_three_way` (blueprint / sg-profile / sg-br from one blueprint
  ckpt → pooled diffs + ordering + verdict), CLI `main()`, JSON schema (per-matchup +
  pooled_diff + paired lifts + per-challenger `stats()` + verdict + git_rev/md5/
  wall-clock). 10 tests (all five verdict routes + orchestration/JSON smoke).
- **Stage 6-C — clean smoke** (492 s, alone). Full pipeline + verdict + JSON validated
  end-to-end on the real blueprint.

## What was decided

- **3-way comparison** (Decision 6.1, per Q11): blueprint / sg-profile / sg-br, with
  the BR-vs-PROFILE delta as the architecturally-decisive revert gate.
- **Verdict LOCKED** (Decision 6.3, ICM-equity-delta on the CRN-paired lift): PASS
  (L≥+0.005, σ≥2.0, ordering, σ(BRvsP)≥1.5); SUBSTANTIVE_PASS (L≥+0.002, σ≥1.5, ≥4/5
  opponents positive, σ(BRvsP)≥1.5); PASS_BR_EQUIVALENT_TO_PROFILE (BR-vs-blueprint
  passes but σ(BRvsP)<1.5 → recommend PROFILE); AMBIGUOUS (positive, undermeasured);
  FAIL (L≤0). Read off mechanically post-run.
- **Real-torch tests use workers=1 in-suite** (the multiprocessing fan-out is validated
  separately by the cheap-policy determinism test) — combining real torch models + a
  process pool is contention-fragile in the full suite.

## What was learned / measured

- **Stage 6-C smoke (100 hands × 1 opp × 3 challengers, workers=4, alone): 492 s** —
  within the expected 460–500 s window. Gate rates **0.31 / 0.32** (match f≈0.27–0.32),
  **0 degraded**. VERDICT **FAIL — but a NOISE artifact** (σ(L)=0.90 < 1, exactly as
  predicted at 100 hands; the verdict *function* correctly routed FAIL on the noisy
  L=−0.033 ≤ 0). The real verdict is Stage 6-D at 5,000 hands × 5 opponents. CRN
  confirmed: paired BR-vs-PROFILE σ (1.54) tighter than the unpaired BR-vs-blueprint σ.
- **Wall-clock projection drift.** Scaling the clean smoke → **~13.7 h at Y=10 / ~5.7 h
  at Y=24** for 6-D, ABOVE the design's §E 5.8 h-at-Y=10 (measured ~10 s/solve blended
  vs the assumed ~5 s). **Most likely box-load** — the full suite ran 786 s (≈2× its
  usual ~360–390 s) in the same window. **Re-measure per-solve on a quiet box before
  6-D launch.** Feasible regardless (<24 h at Y≥10; ~6 h many-core).
- **Two measurement-discipline reinforcements:** (1) the Stage-6-A contention slip
  (running the timing smoke concurrently with the full suite → two ProcessPoolExecutor+
  torch jobs oversubscribed, inflating the smoke and failing TestAblationSmoke in-suite)
  — fixed by workers=1 for the real-policy smoke + the no-concurrent-compute rule; (2)
  aggregation must sum in canonical (sorted) order for true worker-count bit-identity.

## State at close

- **Done: Stages 6-A & 6-B landed; 6-C smoke validated** (`455df14`→`b0dfa11`). HEAD
  `b0dfa11`, tree committed-clean. Full suite 681 passed + 10 subtests, 1 skipped; the
  one failure is the pre-existing, UNTRACKED `tests/test_pushfold.py:274`.
- **Project total: 9 foundational findings** caught by the discipline pattern across
  sessions (this session: the ICM-equity-delta-vs-bb/100 metric correction).
- **Open / next: Stage 6-D — the real 5,000-hand × 5-opponent run + verdict.** Ready
  pending a quiet-box per-solve re-measurement to confirm the wall-clock window.
- **Unchanged carry-forward:** fold `fast_view` into the canonical path; the
  `dcfr-overnight-3000` ICM-retrain decision after sub-step 6.

## Next session opens with

**Stage 6-D — the real eval run + verdict.** Verify quiet-box state, re-measure
per-solve (100 hands × 1 opp, sg-br only), then launch the full 3-way × 5-opponent ×
5,000-hand ablation under `tmux` (no concurrent compute), and apply the locked verdict
mechanically once results land. This is the first measured strength delta from subgame
solving — the milestone the whole B1c line has built toward. Read the number through
the regime-asymmetry lens (`docs/SUBSTEP_5_DESIGN.md` Stage-5-C closure).
