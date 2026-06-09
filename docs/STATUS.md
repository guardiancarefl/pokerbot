# Project Status

**Last updated:** 2026-06-09
**Current phase:** Live deployment readiness. Validated k200 blueprint
  (`runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt`).
  Bridge fixed end-to-end; three deployment-time policy floors shipped;
  depth-confusion characterized and a retrain queued.

> Note: this file historically lagged commits badly. Always cross-check
> `git log --oneline` before trusting any single line. The "Live deployment
> readiness" entry below is the current load-bearing summary; the older
> Leduc / sub-step 6 entries describe a research workstream that's
> superseded by the deployment focus.

## Suspect-frame stack recovery (Layer 1) — 2026-06-09 — CLOSED (`7d47e86`)

The verify1 blackout (seq 83-94: stable stuck-digit hero-stack OCR →
SanityChecker suspect-flagged 12 consecutive otherwise-clean frames → 4
hero-to-act moments dropped → 38.4s freeze → hero busted) is closed
bridge-side. When a suspect frame's only flag is ONE seat's stack jump,
the bridge derives that stack from the clean hand-start anchor via chip
conservation and re-validates through the unchanged replay+invariant gate
(`status=decision_recovered`, `recovered_fields` audit trail; every gate
failure drops exactly as before, reason annotated). Gate evidence: 692
frames / 3 dry-run logs replayed → 0 behavioral diffs on non-suspect
frames; the 3 recoveries are exactly verify1 seq 88/92/94 with derived
values confirmed against the seq=95 post-hand ground truth. 15 new tests
(`tests/test_suspect_recovery.py`); verification harness
`scripts/replay_make_decision_diff.py`.

NOTE the diagnosis correction recorded in SESSION_LOG 2026-06-09: the
prior session's "second unflagged bad field (seat5)" claim was an
analysis artifact — the incident was pure single-field corruption, and
layer 1 alone rescues all three real decisions.

**Open, in priority order:**

- **Layer 2 — Windows scraper (route to Windows CC):** the
  SanityChecker's last-good reference froze during the suspect run and
  kept rejecting the *correct* OCR (`150` at seq 93/94 flagged as "jump
  1260->150"). Spec: decay the reference, or re-accept a value stable
  for N consecutive frames; per-seat suspect flagging would also help.
  Binding sub-case: **anchor starvation** — a suspect run surviving into
  the next hand's hand-start frame leaves the bridge unable to anchor
  (observe() is deliberately blind to suspect frames), disabling layer-1
  recovery for that entire hand. Scraper-side repair is the root-cause
  fix.
- **Layer 3 — contribution ledger: DESIGNED, NOT BUILT, build gated on
  user approval.** `docs/LAYER3_CONTRIBUTION_LEDGER_DESIGN.md`. With the
  corrected evidence it is no longer kick-risk-critical (0 additionally
  rescuable hero-to-act frames in any replayed corpus); the build
  trigger is observable in dry-run logs as a hero-to-act suspect frame
  declined for "no in-range closure candidate" / "ambiguous" / "no
  candidate passed replay+invariant".

## Live deployment readiness — 2026-06-08 — read this first

The validated k200 blueprint is live-deployable. Live dry-run on
`logs/live_dryrun_20260608_152756.jsonl` (200 frames, 9.1 min, 10 hands)
exposed two distinct classes of problem; both have been characterized,
the bridge bugs are fixed, the model gaps have deployment-time
mitigations in place, and the underlying model fix is queued as the
next major workstream.

### Bridge — Issue 2 CLOSED (`f1a412b`)

Two distinct invariant-failure bugs identified, root-caused, fixed, tested.
Re-replay gate green: pre-fix 2 `invariant_fail` → post-fix 0; 196/200
frames bit-identical; 2 FIX (seq=170, seq=192 — both AA hands); 2
state-identical action-seq reorderings (seq=81, seq=83) with provably
unchanged policy decisions; 0 regressions.

- **seq=170 — derive-side**: blind-seat-as-limper misclassification in
  `scraper_schema.py:derive_action_sequence` caused UTG opens to defer
  when BB sat at exactly `bb_amount`. Fix: exclude blind seats at their
  forced post from `limper_after_me_unemitted` when no raise above bb.
- **seq=192 — view-side**: `openspiel_to_scraper_view`'s `matched_all_in`
  branch on postflop used `preflop_max_chip_int` as subtractor, wrong
  for postflop all-ins (over-subtracted current voluntary to 0). Fix:
  subtract `preflop_commit_per_alive + ante`. The `- ante` is motivated
  by the chip_int=pre_hand convention (invariant across both the
  replay-fallback path AND the derive's `busted_mid_hand` emission
  path), not by fallback inflation. Diagnostic at
  `scripts/diag_seq192_legal_actions.py` confirmed chip_int=1515 is not
  in OpenSpiel's `legal_actions`; only `[0, 1, 1525]` are legal at the
  forced-all-in moment.

Tests: `tests/test_bridge_seq170_seq192_fixes.py` (6 tests, all pass).
64/64 tests pass on touched modules (scraper_schema, invariant, replay,
plus all three floors).

### Model — depth-confusion characterized + 3 deployment-time floors shipped

The 236-d encoder normalizes chip features by `starting_stack=1500` and
never uses `big_blind`. Depth-invariance probe
(`scripts/depth_invariance_probe.py`) measured median TV distance 0.53–
0.78 across blind levels at **fixed true BB-depth** (should be ≈ noise)
vs 0.20–0.37 at **fixed chips** (should be high). Ratio 1.55–3.16 in
the wrong direction: the model uses the chip-magnitude smear as a
proxy for BB-depth and the proxy doesn't reconstruct depth correctly.

Live-field paired A/B (`scripts/short_stack_floor_ab.py`,
24,000 paired games at hpl=5 live-matched escalation):
**paired ICM delta V1−V0 = +0.0100 ± 0.00255** (z=3.92, 95% CI
[+0.005, +0.015]). Per-firing (diverged games only) **+0.256 ± 0.065**
(z=3.95). Floor fires on 7.53% of V0 decisions live-matched.

Three deployment-time floors now chain in
`src/nlhe/integration/live_loop.py` (composed via
`make_live_policy_filter`, default off in training and eval):

1. **`apply_aa_kk_preflop_floor`** (`3999d58`) — mask FOLD when hero
   holds AA/KK preflop. Categorically never +EV. 15 tests.
2. **`apply_check_when_free_floor`** (`dfc37c2`) — mask FOLD when CHECK
   is legal (to_call==0 with CALL in `legal_actions`). Strictly
   dominated action: folding for free gives up free equity for zero
   gain. Universal — any street, any depth, any hand. Covers seq=48
   (BB 92o ~55% fold) and analogous spots at all depths.
3. **`apply_short_stack_floor`** (`dfc37c2`) — at hero eff-stack ≤ 6 BB
   (configurable `short_stack_floor_bb`, default 6.0):
   facing action → keep {FOLD, CALL, ALLIN}; check spot → {CALL, ALLIN};
   true unopened (no CALL legal) → {FOLD, ALLIN}. Intermediate bet
   sizes masked, mass redistributed. Covers seq=461 (5.3 BB SB
   min-raising to 2 BB facing action).

Each fire logs to stdout for live dry-run audit:
`[FLOOR] fired=[…]  eff_bb=X.XX  cp=N  street=S  pre_argmax=…  post_argmax=…`.

Default `[OOD-WARN]` runtime log fires when blind level ≥ 8 OR hero
effective stack < 2 BB. Precision (corrected 2026-06-09): training DOES
sample L8 (5.1% weight) and L9 (2.04%) per the yaml `training_weights`;
only L10+ and stacks < ~2 BB are genuinely unsampled. The warning fires
inside the thin sampled tail by design. (The warning STRING in
`live_loop.py` still says "out-of-training-support" — queued nit, code
change out of scope for the doc pass.)

### Scraper — characterized as NON-BLOCKING for this session

Of the 200 session frames, **41 (20.5%) are scraper-side failures**
(the original log used `skip_data_quality` as a UNION bucket — see the
"Correction" entry in `DECISIONS.md`):
- 17 `scraper_suspect` (image render / OCR confidence below threshold)
- 17 `data_quality: dealer field missing/empty`
-  4 `data_quality: dealer points to non-alive seat` (mid-hand transient)
-  3 `parse_error: blinds string parse` (level-transition garbling)

**CORRECTED 2026-06-09 — hands WERE sat out due to scraper.** The
original "zero hands sat out" claim segmented hands by
`(dealer_seat, level)`, which is blind to any hand whose EVERY frame
lost the dealer button — exactly the failure mode of the dominant skip
class. Re-analysis using `raw_record` hero-card/button evidence:
- This session (152756): **1/11 hands sat out** — the 8dKc hand
  (seq=92 shows hero action buttons CALL/FOLD/RAISE, killed by
  `dealer field missing/empty`; seq=97 killed by `ScraperSuspect`;
  0 decision frames in the hand).
- Session 154557: **14/41 hands lost (34%)** — 12 missed decisions
  (incl. an AdKd hand fully lost to suspect frames, seqs 106–118)
  + 2 safe-fold-only hands.
Root cause: dealer-button OCR failed on every hero-button hand — the
raw `dealer` field NEVER parsed as seat1 (hero) in any session through
154557. See DECISIONS.md "Correction — hands sat out" + "Scraper
dealer-at-hero-seat fix" entries (2026-06-09).

Dealer-button detection WAS the highest-leverage target (understated:
55% of 154557's data-quality frames were dealer-related, 24% of all
its frames). A Windows-side fix landed 2026-06-09 between sessions
154557 and 192543; dealer-OCR skips collapsed 24.0% → 2.2% of frames.

### Next major workstream — model retrain with `eff_stack_in_BB`

Queued, **NOT before the next dry-run**: add a BB-normalized depth
channel to the encoder (`eff_stack_in_BB = min(hero_stack, max alive
opp stack) / big_blind`) and retrain. The model should learn to read
BB-depth directly rather than from the chip-magnitude smear, and the
short-stack floor should become a no-op.

**Falsification test for the retrain:** re-run
`scripts/short_stack_floor_ab.py` against the retrained checkpoint
with the same paired-seed harness. A depth-aware model should drive
both the firing-divergence rate AND the diverged-only delta toward
zero. If the diverged-only delta stays near +0.25 after retrain, the
feature didn't land — investigate before shipping.

### Commits load-bearing for this state

- `3999d58` feat(deploy): AA/KK preflop FOLD floor
- `dfc37c2` feat(deploy): short-stack + check-when-free floors
- `f1a412b` fix(bridge): close Issue 2 (seq=170, seq=192)

Validation hashes (the deployed model):
- `ckpt_iter_1500.pt`  sha256 = `b79e82dd0ce9e78e4eb666b7379df953dadbf2a6e026c6bd4b6eec695e9b1b11`
- `abstraction.pkl`    sha256 = `0fc20800dc7ce89ea950c975decbd530ac6c6f3c164105e8ab99d24359de4c8e`

---

## Leduc proof complete — S1 verdict (2026-06-01, Session 6 late) — read this first
The Leduc CPU proof is DONE. Verdict: **S1 confirmed** — Leduc is too information-poor
(~2–3 opp decisions per hand) to per-hand-resolve opponent STRENGTH, even with direct
cell-classification supervision and a transferable head-derived read. The architecture,
leakage invariants (Tests A/B/C/D), and read mechanism are VALIDATED and carry forward to
6-max. See `docs/DECISIONS.md` → "Leduc proof complete — S1 verdict; move to 6-max scaffold"
for the verbatim verdict, the full falsification chain (imbalance → smearing → read primitive
→ training target), what's validated, the open scaffold questions, and the gated next step.

- **Falsification chain — four independent hypotheses ruled out by experiment:**
  imbalance (A+B, `e4c9a9a`) → feature smearing (C, `08b59eb`, reverted `2d004ca`) →
  read primitive (D-pivot, `8f8f20b`) → training target (S2(a), `9e5c07f`). Maniac mis-
  classification breakdown was the decisive diagnostic: 90% raise-cluster, 58% strength-resolved
  to s1.00, 32% spilled to s0.50 = "reads tendency, loses strength" — the predicted
  information-poverty signature.
- **What was validated (carry forward to 6-max):** transformer trunk + opp_head_cell
  classifier + head-derived KL-vs-anchor read; §8 A/B/C/D leakage invariant as the reusable
  correctness surface; gate(0)=0 confidence-gated blend as the safety mechanism;
  oracle-free read path (transferable by construction to unseen opponents).
- **Next step (gated; NOT GPU yet):** build + CPU-smoke-test the 6-max adaptive-training
  scaffold on Contabo (net + DCFR ICM blueprint anchor + training loop + 6-max-analog
  leakage tests + E0-analog bounded). GPU spend only after the scaffold runs correctly on
  CPU. Scaffold spec is next-session design surface.
- **Open scaffold questions (flag, not solved):** (a) cell-classifier generalization
  story (continuous embedding vs archetype-mixture vs hybrid — 6-max opponents are not
  discrete cells); (b) trunk capacity under triple-loss budget (E0 broke at Leduc;
  first lever is lowering `λ_action` while keeping `λ_cell` primary).
- **Resolver code disposition:** unchanged from the foundation-pivot entry —
  in-tree but DORMANT; do not invest in fixing it. Bubble-slice BR arms have closed
  (both shards DONE; killphil ≤15BB diff −0.039 σ−2.5, ticket >15BB diff −0.044 σ−2.9;
  net-negative overall on both, resolver path closed).

The floor rule from the prior entry CARRIES FORWARD (no change may make the shipped agent
worse than blueprint-alone). The "Floor lock" block below is preserved for history.

## Foundation pivot (2026-05-31, Session 6, post-bubble-slice) — superseded by the entry above
The real-time resolver is RETIRED as a development path. New foundation = a TRAINED ADAPTIVE
POLICY (StratFormer-style) anchored to a GTO baseline, validated first by a CPU-only Leduc
proof on Contabo before any 6-max/GPU work. The "Leduc proof FIRST" gate from this entry was
honored and cleared above; the four-experiment falsification chain finished with S1.
See `docs/DECISIONS.md` → "Foundation pivot: retire real-time resolver; build trained-
adaptive-policy with Leduc proof first" for the original verbatim plan + Steps 0–4.
- **Step 0 DONE:** tabular CFR+ Nash anchor at **0.1286 mbb/g** (Nash bar < 5 mbb/g; ~39× margin).
  Commit `019d486`; artifact `runs/leduc_cfr_anchor_20260531_144405/`.
- **Steps 1–4 DONE (Leduc proof closed):** see "Leduc proof complete" block above.
- **Resolver code:** in-tree but dormant; do not invest in fixing it. Bubble-slice BR arms
  closed (final numbers in "Leduc proof complete" block above).

## Floor lock (2026-05-31, Session 6) — blueprint-alone is the reference agent
The real-time resolver is proven net-negative at every shippable config (X0 lift ≈ 0 vs a
correct-model opponent; J2 condition-A −2.7 to −4.2 ICM-pts vs blueprint on all 4 legit profiles;
J3 BR-leaves help partially but never reach blueprint-alone and are unshippable at action p95
82–99s / 14.5% >15s, d=3; depth sweep: d=3 is the worst point, non-monotone). Root cause under
diagnosis: leaf-value precision/bias vs unsafe solve. **Resolver is experiment-only / opt-in** —
do not enable in any non-experimental path. See DECISIONS.md "Blueprint-alone is the reference
agent". Open diagnostic: X5 bubble slice (running), X6 two-arm leaf probe (M=32; Arm-1 d3 + gated
Arm-2 d4) → feeds the Step-4 rebuild choice (A safe-resolve / B learned leaf value net).

## Done
- **Phase 1 — Leduc Deep CFR** (OpenSpiel-wrapped), validated.
- **Phase 2 — custom NLHE Deep CFR:** EMD card abstraction, external-sampling
  solver with bit-identical checkpoint resume, Slumbot eval client.
- **Phase 4 — 6-max SNG:** parametric game strings, ICM value function
  (Malmuth-Harville) + ICM-adjusted returns wired into training, 6-max
  external-sampling CFR (`cfr6.traverse_6max`), PSRO league play (v1 + v2);
  `dcfr-overnight-3000` blueprint trained ICM-correct.
- **Phase 5 — Shanky bot-profile runtime:** parser, predicate evaluator, policy
  adapter; 36 profiles loadable; league-v2 + Shanky eval baselines measured.
- **B1c sub-step 1.5 — subgame tree builder:** correct discretized enumeration,
  20/20 tests vs the production game (`2be87df`). Internal-node descendants
  invariant verified on decision + chance trees.
- **B1c sub-step 2 — leaf evaluator DESIGN approved**
  (`docs/SUBGAME_LEAF_DESIGN.md`, best-response form, after two revisions +
  Q4.5 / Q11 additions).
- **B1c sub-step 2 — leaf evaluator IMPLEMENTED** (Stages A–E, `994b587`→`fd7fb88`;
  ICM busted-seat fix `ae8e1b5`; M=8 restore `db89145`; cache-reset guard
  `3109fb0`). Correct, tested, production-ready.
- **Q13 — leaf-eval budget RESOLVED** (`b092480`, session 16): no optimization
  needed, Stage E.5/E.6 shelved (`docs/STAGE_E_BUDGET_REDERIVATION.md`).
- **B1c sub-step 2 — Stage F (Q11 Level 1 leaf ablation) CLOSED** via
  SUBSTANTIVE_PASS_AGGREGATE (session 17, `e939bce`→`9dbbfd4`;
  `docs/sessions/session_17_summary.md`). The per-pair opponent-own-value
  resolution gate is structurally intractable (55% of leaf-opp pairs have zero
  bias effect → max resolution ~45% at any M), but the architecture is confirmed
  by the aggregate hero-direction signal (+3.4/+3.6σ), 94% differentiation among
  resolved pairs, and a non-degenerate menu.
- **B1c sub-step 2 — Stage G (Q11 Level 2 decision-level stub ablation) CLOSED** via
  SUBSTANTIVE_PASS_AGGREGATE (session 18, `cb82072`→`b4f85dd`;
  `docs/sessions/session_18_summary.md`). **Sub-step 2 is now complete.** The M=16
  gate was a clean 2–3σ near-miss (value_suppression +2.45σ, policy_divergence
  sig@~99.5%); the one design-sanctioned M=32 escalation cleared both load-bearing
  bars (value_suppression +3.07σ; policy_divergence significant, obs L1 0.330 > null
  p99.7 0.297; differentiation 71.4%, 6 distinct shifted actions). Finding: the
  "BR is flatter" prior was **wrong** (BR −0.29σ on entropy — BR *shifts mass*, does
  not flatten); entropy was correctly demoted to non-load-bearing before the run.
- **B1c sub-step 3 — the real subgame CFR solver CLOSED** (session 19,
  `3ef06e4`→`10f7e45`; `src/nlhe/subgame_solver.py`, `docs/SUBSTEP_3_DESIGN.md`,
  `docs/sessions/session_19_summary.md`). Vanilla weighted multi-iteration CFR over
  the depth-limited tree (Decision 2, approved): opponents fixed at blueprint, chance
  weighted by `chance_prob`, hero accumulates RM+ regret, linear-LCFR average root
  policy. Stages 3-A scaffold/warm-up/K=0 → 3-B K=1 bit-identical to the Stage-G stub
  → 3-C K>1 loop → 3-D diagnostics (`summarize_solve_result`) → 3-E **production
  K=1000** locked. Closed by execution-and-measurement (implementation stages), not
  an ablation gate. Measured: K=1000 loop 0.109 s (~15× under estimate, ~250× under
  the 27 s budget); convergence `converged_l1_tail` 4.4e-2→4.4e-8 over K=10→1000.
  Safety = BR leaf mode, no CFV gadget (Decision 6, approved).
- **B1c sub-step 4 — policy extraction CLOSED** (session 20, `9798832`;
  `subgame_solver.extract_action`). root_policy 7-vector → played chip action
  (sample default / argmax), reuses the tree's discretize map, applies the
  `ALLIN`→CALL(1) chip-0 alias (`b2dded5`) at the translation boundary.
- **B1c sub-step 5 — SubgamePolicy wrapper CLOSED** (session 20, `6ab60be`→`9ff106d`;
  `src/nlhe/subgame_policy.py`, `docs/SUBSTEP_5_DESIGN.md`,
  `docs/sessions/session_20_summary.md`). Drop-in `eval_pool.Policy`: gate (≥3 actions
  AND blueprint max-prob <0.95; empirical f≈0.27) → SKIP (blueprint) / SOLVE
  (build→evaluate_leaves→solve→extract) → degraded → blueprint fall-through (WARNING +
  `n_degraded`, no back-off). Two foundational findings caught + fixed this session:
  **chance-leaf parse crash** (`03576eb`) and **tree-builder leaf explosion**
  (`9ff106d`, 2560→5–12 leaves; chance now collapses to a transparent leaf, chance
  leaves use blueprint-only eval — 88% bias-inactive). Per-solve ~6.7 s blended
  (≈ Q13); sub-step-6 projected **~3.8 h Contabo-parallel** — feasible.

## In progress
- B1c **sub-step 6 — Level-3 pool ablation** (subgame-BR vs subgame-PROFILE vs
  blueprint over `league-v2-600` × 5,000 hands). Sub-steps 4 & 5 are closed; the
  full subgame-solving stack (tree → leaf-eval → solver → extract → SubgamePolicy)
  routes through `eval_pool` unchanged.

## Next up (sub-step 6 + handoff)
1. **Sub-step 6 design proposal** (next session): the go/no-go strength measurement.
   Needs **hand-level multiprocessing** (`eval_pool` is sequential; ~3.8 h
   Contabo-parallel only with it). **Load-bearing interpretation context:** the
   deployment mix (98% preflop, chance leaves blueprint-only/bias-inactive,
   round-closing solves shallow) concentrates the BR lift in the minority of
   chance-free decision-bearing solves — sub-step 6's bb/100 may be **well below**
   the Stage F/G aggregate-signal projection (see SUBSTEP_5_DESIGN Stage-5-C closure).
2. View/discretize fast path is shipped (`src/nlhe/fast_view.py`); fold into the
   canonical path now that sub-steps 2–5 have closed (NEXT_SESSION.md tracked
   deliverable + acceptance criteria).

## Then (later B1c sub-steps)
- Decide the `dcfr-overnight-3000` ICM-retrain after sub-step 6 measures the
  busted-seat-bias impact (`ae8e1b5`).

## Known issues / open questions
- The ~0.9 ms/step state-prep floor is `_build_view_6max` (0.64 ms) + `discretize`
  (0.10–0.24 ms) doing O(n) Python ops over the ~9,803-element fullgame
  `legal_actions()` — NOT the regex parse (0.008 ms; earlier attribution corrected
  2026-05-24). The fix is the sorted-legal-actions fast path (`fast_view.py`,
  sub-step 2 Stage A), gate ≤ 0.30 ms/step; folding it into canonical
  `_build_view_6max` is filed as a follow-up (see `NEXT_SESSION.md`).
- BR-vs-blueprint robustness gain is an empirical bet, unproven until the Q11
  Level-3 pool ablation runs (post-sub-step-5).
- `SESSION_LOG.md` documents through Session 9 only; Sessions 10–12 live in commit
  messages / STATUS, not back-filled. Session 13+ summarized in `docs/sessions/`.

## Decisions deferred
- Fast-path fallback knob (cut L / cut M / raise X) — measure `fast_view.py` first.
- α (bias strength, default 3.0) and k tuning — revisit if the Q11 Level-3 ablation
  underwhelms.

## Session 5 close (2026-05-29)

### What landed
- Parallel framework wired into production training path (commit b2571aa). G=10 measured at 4.5x speedup, BLAS-pinned, parallel mode validated end-to-end with mini_eval support inside the orchestrator.
- 24 Shanky profiles ingested at data/shanky_profiles/ (gitignored), curated 9-profile rotation, league registry at configs/league/registry_experiment.json.
- Three experiment configs committed (anchor / control / treatment at 2000 iters each, k=200, parallel_groups=10). Configs remain in repo for future use.
- Anchor run completed cleanly: runs/dcfr_anchor_2000 symlink → phase4f_dcfr_anchor_2000_20260529_061615/, iter 2000 checkpoint preserved.
- 17 session commits, all pushed to origin/phase4f-league.

### Pivot (mid-session)
Anchor's lift trajectory revealed the bot plateaus around iter 500-1000 at k=200; mean self-anchor lift ≈ zero after iter 500, with strat_loss continuing to refine (0.945 → 0.815) but no measurable head-to-head strength gain. The bot loses to 15 of 19 sampled Shanky scripted bots. This is consistent with k=200 abstraction being the bottleneck, not training iterations. See DECISIONS.md for the full reasoning.

### Decisions
- Diversity-mix control + treatment runs shelved (configs preserved).
- Layer 3 (real-time subgame solving) promoted to next session's focus.
- Anchor checkpoint preserved as k=200 blueprint baseline for future blueprint-vs-blueprint+subgame comparison.

### Queued for Session 6
1. Layer 3 design recon: read the published subgame-solving literature relevant to our setup (CPU-only, 6-max not HUNL, ICM-adjusted value function). Pluribus's continual resolving, safe subgame solving, depth-limited solving — which variant fits us?
2. Layer 3 implementation plan: integration points with the blueprint we already have, computational budget per decision, memory footprint, action abstraction during resolve.
3. Decision: build Layer 3 on top of current k=200 blueprint (cheaper, faster), or train a k=500 blueprint first and add Layer 3 on top (longer overall but stronger). The right call depends on Layer 3's expected lift over blueprint alone.
