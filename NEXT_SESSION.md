# Next Session Pickup Notes

## Runtime state (added session 16 — read before any eval/timing planning)

The project's **nominal** runtime is the Contabo VPS (12 vCPU, no GPU) per the
original architectural decision (CLAUDE.md / DECISIONS.md). **Actual development
since ~session 13 runs on whatever RunPod instance is currently active** — session
16's measurement box was a 128-core + RTX PRO 4000 Blackwell machine, not Contabo.
CLAUDE.md and the body of this file still describe the Contabo host; treat that as
the nominal/fallback target, not a statement of the live box. For sub-step 6 eval
planning: the Q13 conclusion (**current evaluator is adequate, no optimization
needed**) holds across hardware — on Contabo (12 vCPU, parallelism Y≈10) the full
Level-3 ablation is **~21 h** wall-clock; on a many-core box (Y≈24+) it is
**~10.5 h or less**. Both fit an overnight-to-weekend budget. Note the Q13
hardware surprise: this evaluator runs **faster on CPU than GPU** (single-row
`[64,64]` forward is launch-bound), so a GPU pod is not required — many CPU cores
are what help. Full reasoning: `docs/STAGE_E_BUDGET_REDERIVATION.md`.

## Where the project stands (start of session 22)

B1c (depth-limited subgame solving):

- **Sub-step 1.5 — tree builder: DONE** (`2be87df`). Production-game tested.
- **Sub-step 2 — leaf evaluator: IMPLEMENTATION COMPLETE** (Stages A–E,
  `994b587`→`fd7fb88`) plus the load-bearing ICM busted-seat fix (`ae8e1b5`) and
  the post-Q13 cleanups (`db89145` restore M=8; `3109fb0` cache-reset guard). The
  evaluator is correct, tested, and production-ready.
- **Q13 — budget RESOLVED** (`b092480`, session 16). No optimization needed;
  Stage E.5/E.6 **shelved**. (`docs/STAGE_E_BUDGET_REDERIVATION.md`.)
- **Stage F (Q11 Level 1) — CLOSED via SUBSTANTIVE_PASS_AGGREGATE** (session 17,
  `e939bce`→`9dbbfd4`). See `docs/sessions/session_17_summary.md`. The per-pair
  opponent-own-value resolution gate is **structurally intractable** in this
  shallow SNG (55% of leaf-opp pairs have zero bias effect → max resolution ~45%
  at any M), but the architecture is confirmed by the **aggregate** signal:
  hero-direction +3.4σ (BASELINE−BR) / +3.6σ (PROFILE−BR), 94% differentiation
  among resolved pairs, non-degenerate menu (biases 1/2/3 all selected).
- **Stage G (Q11 Level 2) — CLOSED via SUBSTANTIVE_PASS_AGGREGATE** (session 18,
  `cb82072`→`b4f85dd`). See `docs/sessions/session_18_summary.md`. The M=16 gate was
  a clean 2–3σ near-miss; the one design-sanctioned M=32 escalation cleared both
  load-bearing bars (value_suppression +3.07σ; policy_divergence significant, obs L1
  0.330 > null p99.7 0.297; differentiation 71.4%, 6 distinct shifted actions). BR
  demonstrably moves hero's *root decision*, not just leaf values. Finding: the
  "BR flatter" prior was wrong (BR shifts mass, does not flatten) — entropy was
  correctly demoted to non-load-bearing before the run.
- **Sub-step 3 — subgame CFR solver: CLOSED** (session 19, `3ef06e4`→`10f7e45`;
  `src/nlhe/subgame_solver.py`, `docs/SUBSTEP_3_DESIGN.md` Stage-3-E closure,
  `docs/sessions/session_19_summary.md`). Vanilla weighted multi-iteration CFR over
  the depth-limited tree: opponents fixed at blueprint, chance weighted by
  `chance_prob`, hero accumulates RM+ regret, output = linear-LCFR average root
  policy. **Production K=1000.** Closed by execution-and-measurement (implementation
  Stages 3-A…3-E), not an ablation gate. Both flagged decisions approved: vanilla
  CFR (Decision 2) and BR-as-safety / no CFV gadget (Decision 6).
- **Sub-step 4 — policy extraction: CLOSED** (session 20, `9798832`;
  `subgame_solver.extract_action`). root_policy → played chip action (sample/argmax),
  reuses the tree's discretize map, `ALLIN`→CALL(1) chip-0 alias at translation.
- **Sub-step 5 — SubgamePolicy wrapper: CLOSED** (session 20, `6ab60be`→`9ff106d`;
  `src/nlhe/subgame_policy.py`, `docs/SUBSTEP_5_DESIGN.md` Stage-5-C closure,
  `docs/sessions/session_20_summary.md`). Drop-in `eval_pool.Policy`; gate (empirical
  f≈0.27) → SKIP/SOLVE → degraded→blueprint. Two foundational findings caught + fixed
  (chance-leaf parse crash `03576eb`; tree-builder leaf explosion `9ff106d`). Per-solve
  ~6.7 s blended; sub-step-6 ~3.8 h Contabo-parallel — feasible.
- **Sub-step 6 — Stages 6-A & 6-B LANDED, 6-C smoke validated** (session 21,
  `455df14`→`b0dfa11`; `scripts/eval_pool_ablation.py`, `docs/SUBSTEP_6_DESIGN.md`,
  `docs/sessions/session_21_summary.md`). Parallel CRN harness (per-hand SHA256 seeding,
  worker-count bit-identity, fail-loud), 3-challenger CLI, locked four-branch verdict.
  6-C smoke: 492 s, gate 0.31/0.32, 0 degraded, FAIL=noise-as-predicted at 100 hands.
- **Path: Path A (confirmation-first). Sub-steps 2–5 CLOSED; sub-step 6 harness ready.**
  Next is **Stage 6-D — the real run + verdict.**

## Current deliverable — Stage 6-D (the real Level-3 eval run + verdict)

The harness is built and validated. Stage 6-D runs the full 3-way ablation at scale and
applies the **locked verdict** mechanically. **Run on a QUIET box, under `tmux`, with NO
concurrent compute** (the carried-over contention discipline — session 21 lost a clean
timing read to a concurrent job).

**Procedure (in order):**

1. **Verify quiet-box state.** No other heavy jobs (`ps`/`top`); on shared infra,
   confirm no other tenant is hammering CPU. The per-solve cost is CPU-bound and
   contention doubles it.

2. **Re-measure per-solve on the quiet box** (the 13.7 h vs 5.8 h question). Quick:
   ```
   python -m scripts.eval_pool_ablation \
     --blueprint-ckpt runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/ckpt_iter_3000.pt \
     --abstraction runs/abstraction_20260521_223018/abstraction.pkl \
     --opponents dcfr-shake-200=runs/six_max_20260524_005853_phase4f_dcfr_linear_shakedown/checkpoints/ckpt_iter_0200.pt \
     --hands 100 --workers <cpu-2> --base-seed 7 --output evals/subgame_6d_calibration.json
   ```
   - **If ≈ design (~5 s/solve, smoke ≈ 250–300 s at workers≈10 → 6-D ≈ 5.8 h at Y=10):**
     launch full-scale, expect a ~6 h window.
   - **If ≈ box-loaded (~10 s/solve, smoke ≈ 490 s → 6-D ≈ 13.7 h at Y=10):** still
     launch, but plan the longer window (or use a many-core box: ~5.7 h at Y=24).

3. **Launch the real run** (5,000 hands × 5 opponents × 3 challengers) under `tmux`:
   ```
   python -m scripts.eval_pool_ablation \
     --blueprint-ckpt runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/ckpt_iter_3000.pt \
     --abstraction runs/abstraction_20260521_223018/abstraction.pkl \
     --opponents \
       vanilla-200=runs/six_max_20260523_224646_phase4f_overnight/checkpoints/ckpt_iter_0200.pt \
       vanilla-400=runs/six_max_20260523_224646_phase4f_overnight/checkpoints/ckpt_iter_0400.pt \
       dcfr-shake-100=runs/six_max_20260524_005853_phase4f_dcfr_linear_shakedown/checkpoints/ckpt_iter_0100.pt \
       dcfr-shake-200=runs/six_max_20260524_005853_phase4f_dcfr_linear_shakedown/checkpoints/ckpt_iter_0200.pt \
       dcfr-overnight-600=runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/ckpt_iter_0600.pt \
     --hands 5000 --workers <cpu-2> --base-seed 2026 \
     --output evals/subgame_ablation_v1_5000.json
   ```
   These are the **baseline `league-v2-600` pool's 5 opponents**, paths verified present
   (vanilla-200/400 from `eval_overnight.sh`; dcfr-shake-100/200 + dcfr-overnight-600
   from `configs/league/registry.json`). Monitoring: the CLI prints a header then the
   final verdict; `tail` the tmux pane / the run's stdout. No per-matchup progress log
   yet — if a progress heartbeat is wanted, add one before launch (small).

4. **Apply the LOCKED verdict** (computed automatically by the harness; in
   `docs/SUBSTEP_6_DESIGN.md` Decision 6.3, on the CRN-paired lift in ICM-equity-delta):
   - **PASS:** L ≥ +0.005 AND σ(L) ≥ 2.0 AND ordering BR ≥ PROFILE ≥ blueprint AND
     σ(L_BRvsP) ≥ 1.5.
   - **SUBSTANTIVE_PASS:** L ≥ +0.002 AND σ(L) ≥ 1.5 AND ≥ 4/5 opponents positive AND
     σ(L_BRvsP) ≥ 1.5.
   - **PASS_BR_EQUIVALENT_TO_PROFILE:** BR-vs-blueprint passes strict/substantive but
     σ(L_BRvsP) < 1.5 → recommend PROFILE for production.
   - **AMBIGUOUS:** L > 0 but σ(L) < 1.5 OR L < +0.002 → re-run with more hands.
   - **FAIL:** L ≤ 0 → surface for diagnosis (Path A), do not lock.
   Read off mechanically; do **not** retune thresholds to the result.

**REGIME-ASYMMETRY LENS (load-bearing for interpretation).** Deployment is **98%
preflop**; chance leaves are blueprint-only / 88% bias-inactive; round-closing solves
are shallow. The BR lift concentrates in the **minority of chance-free decision-bearing
solves** — a different mix than Stage F/G's +3.4σ / +3.07σ. **The measured ICM-equity-
delta lift is expected ~+0.002 to +0.005, NOT a Stage-F/G-scaled large effect.**
SUBSTANTIVE_PASS (+0.002) is genuine validation; PASS (+0.005) is strong; AMBIGUOUS
(positive, σ<1.5) means the architecture lifts in a smaller regime than projected — a
real finding, not a failure. Full reasoning: `docs/SUBSTEP_5_DESIGN.md` Stage-5-C
closure.

The full stack (build_subgame_tree → evaluate_leaves → solve_subgame → extract_action
→ SubgamePolicy) routes through `eval_pool_ablation` unchanged; `summarize_solve_result`
and `SubgamePolicy.stats()` (in the JSON) supply per-challenger diagnostics.

## Ablation gate reference (Stages F & G methodology, for sub-step 3 / Level 3 to reuse)

Both gates use a split metric (`verdict()`), status in
{PASS, SUBSTANTIVE_PASS_AGGREGATE, FAIL}, with **resolution** ("is the effect
detectable at >3σ on a CRN-paired difference") separated from **differentiation**
("does BR actually move the decision"), plus a `SUBSTANTIVE_PASS_AGGREGATE` fallback
for the resolution-intractable-but-aggregate-confirmed case this SNG produces:
- **Stage F** (`scripts/ablation_leaf_eval.py`): per-(leaf,opp) resolution;
  differentiation = BR picks non-blueprint bias.
- **Stage G** (`scripts/ablation_decision_level.py`): per-root resolution on the
  most mode-deviating action; differentiation = material L1 root-policy shift
  (direction-agnostic); load-bearing aggregate = value_suppression (≥3σ) +
  policy_divergence (per-deal mode-label **permutation null**, significant @99.7% —
  approved over the literal "bootstrap CI" spec, see SUBGAME_LEAF_DESIGN.md Q11 L2).

## B1c roadmap (post-Q13)

1. **Sub-step 2 — leaf evaluator — CLOSED** (Stages A–E DONE; **Stage F CLOSED**
   session 17, **Stage G CLOSED** session 18, both via SUBSTANTIVE_PASS_AGGREGATE).
2. ~~Stage E.5 — bucket-MC precompute~~ — **SHELVED (Q13)**, no longer on the path.
3. **Sub-step 3 — subgame CFR solver — CLOSED** (session 19, vanilla weighted CFR,
   production K=1000; `src/nlhe/subgame_solver.py`).
4. **Sub-step 4 — policy extraction — CLOSED** (session 20, `9798832`;
   `extract_action`, ALLIN→CALL alias).
5. **Sub-step 5 — SubgamePolicy wrapper — CLOSED** (session 20, `6ab60be`→`9ff106d`;
   + chance-leaf fix `03576eb` and tree-builder cost mitigation `9ff106d`).
6. **Sub-step 6 — Level-3 pool ablation (BR vs PROFILE_SAMPLE vs blueprint).**
   Harness LANDED + validated (Stages 6-A/6-B, session 21, `455df14`→`b0dfa11`;
   `scripts/eval_pool_ablation.py` — hand-level multiprocessing, CRN, locked verdict;
   6-C smoke clean at 492 s). **NEXT: Stage 6-D — the real run + verdict** (5,000 hands ×
   5 opponents × 3 challengers; ~5.8–13.7 h at Y=10, re-measure per-solve on a quiet box
   first). See "Current deliverable — Stage 6-D" above.

## Carry-forward for sub-step 5 (SubgamePolicy)

- **`DiscreteAction.ALLIN` → chip 0 = FOLD when facing a shove with no re-raise
  room** (documented `b2dded5`). When SubgamePolicy selects ALLIN and translates
  back to a game action in that state, it must map to **CALL (1)**, not the chip-0
  fold. Do not let the alias ship a fold where the policy meant all-in.

## TRACKED DELIVERABLE — fold fast_view into the canonical path (after sub-step 2 closes)

Sub-step 2 ships the view/discretize optimization as a parallel `src/nlhe/fast_view.py`
to contain blast radius (Stage A). Measured 6× faster (0.046 vs 0.28 ms/step) with
field-identical output. After sub-step 2 closes (Stages F/G done), fold it into the
canonical `cfr6._build_view_6max` + `actions.discretize_legal_actions` and re-point
all consumers: `traverse_6max` (TRAINING hot path), `subgame.py`,
`pushfold_policy.py`, `scripted_bots/policy.py`, `solver.py`, `policy_adapter.py`,
`scripts/eval_pool.py`, `scripts/eval_6max_self_play.py`.

**Acceptance for the fold-in (not optional):**
1. The Stage A exact-equality tests (`tests/test_fast_view.py`) become the
   regression guard and must stay green after the canonical path is swapped.
2. **Reproducibility against `dcfr-overnight-3000`:** run a small fixed training
   step (same seed, same data) on the blueprint *before* and *after* the swap and
   confirm the produced advantages / network outputs are identical to
   floating-point tolerance. If outputs diverge beyond tolerance, the fold-in is
   wrong — do not land it.

## Also still queued (unchanged by Q13)

- Decide the `dcfr-overnight-3000` ICM-retrain after sub-step 6 measures the
  practical impact of the busted-seat bias (`ae8e1b5`).

## Docs map

- `docs/SUBGAME_LEAF_DESIGN.md` — the sub-step 2 design + Q11 Stage F/G outcomes +
  Q13 resolution (read first).
- `docs/STAGE_G_DESIGN.md` — Stage G design proposal (stub, regime-aware gate).
- `docs/STAGE_E_BUDGET_REDERIVATION.md` — full Q13 budget reasoning + measurements.
- `docs/sessions/session_18_summary.md` — most recent session (Stage G closure,
  sub-step 2 complete).
- `docs/sessions/session_17_summary.md` — Stage F closure.
- `docs/sessions/session_16_summary.md` — Q13 budget re-derivation.
- `docs/sessions/README.md` — the per-session-summary convention.
- `docs/STATUS.md` — current snapshot. `docs/DECISIONS.md` — locked choices.
