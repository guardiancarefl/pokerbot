# Sub-step 6 — Level-3 Pool Ablation — Design Proposal (B1c)

**Status:** PROPOSAL (review gate). No implementation lands with this doc. The final
design proposal of the B1c line: the strength measurement that decides whether subgame
solving makes the bot meaningfully stronger. Line numbers cite files as read 2026-05-26
(session 21). Predecessors: the full subgame stack (sub-steps 1.5–5, CLOSED) is a
drop-in `eval_pool.Policy` challenger (`src/nlhe/subgame_policy.py`).

---

## A. Goal

Measure the **strength lift from subgame solving** over the blueprint, against the
established `league-v2-600` opponent pool, at 5,000 hands/matchup — the Q11 Level-3
go/no-go. Three challengers, all built on the `dcfr-overnight-3000` blueprint:
**blueprint** (plain `CheckpointPolicy`), **subgame-PROFILE** (`SubgamePolicy`,
`leaf_mode=PROFILE_SAMPLE`), **subgame-BR** (`SubgamePolicy`,
`leaf_mode=BEST_RESPONSE`, the production form). At stake: whether the `v×k` BR cost
and the whole real-time-solving architecture earn their keep, or we revert to
PROFILE / blueprint. This is the milestone the B1c line was built toward.

The headline metric is the harness's native **ICM-equity-delta diff** (challenger
per-seat equity minus opponent per-seat equity, averaged over hands), with stderr and
σ — exactly what `evals/league-v2-600_vs_pool_5000.json` reports. **Not bb/100**
(see Finding 1).

---

## B. Comparison structure — Decision 6.1: **3-way** (blueprint, PROFILE, BR)

**Chosen: 3-way.** Q11 Level 3 pre-committed it ("run all three challengers …
subgame-BR ≥ subgame-PROFILE ≥ blueprint, BR-vs-blueprint σ > 2") and the BR-vs-PROFILE
contrast is the *explicit architectural go/no-go*: it isolates whether BR's lift comes
from the **BR mechanism** (adversarial bias maximization) versus generic
from-blueprint refinement that PROFILE would also deliver. Stage G decomposed this at
the *decision* level (+3.07σ value-suppression); sub-step 6 decomposes it at the
*strength* level, which is the dimension that actually decides the architecture.

Trade-off: 3-way costs more compute than 2-way, but the blueprint challenger does **no
solving** (it is just `CheckpointPolicy` — minutes), and PROFILE is ~2× cheaper per
solve than BR (no `v×k`), so 3-way is only ~1.5× the BR-only cost, not 3× (§E). The
information — independent lift of each architecture over blueprint, plus the BR-vs-
PROFILE delta that gates the revert decision — is worth it. 2-way would forfeit the
revert criterion that Q11 made the whole point of the gate.

---

## C. Parallelism — Decision 6.2: hand-level multiprocessing with per-hand seeding

`eval_pool.evaluate_matchup` is **sequential**: one `rng = random.Random(seed)` drives
all 5,000 hands of a matchup in a single loop (eval_pool.py:173-200). The ~3.8 h
Contabo-parallel projection requires parallelism. Design:

**New `scripts/eval_pool_ablation.py`** wrapping `eval_pool`'s per-hand logic; it does
**not** modify `eval_pool.py`.

- **Per-hand seeding (the enabler).** Replace the one-stream-per-matchup RNG with a
  deterministic per-hand seed `seed_h = SHA256(base_seed, opp_idx, hand_idx) mod 2^31`.
  Each hand constructs `random.Random(seed_h)` and drives `sample_starting_state` +
  `play_one_hand`. This makes the result **order-independent** → identical regardless
  of worker count, and lets all three challengers replay the **same hand seed** for
  CRN pairing (Decision: this is the lift-variance reducer; see §D).
- **Worker unit = a hand-index shard.** Workers = `min(n_shards,
  multiprocessing.cpu_count() - 2)` (Q13: many CPU cores, single-row forward is
  launch-bound). Each worker constructs its **own** `SubgamePolicy` + opponent
  `CheckpointPolicy` instances (loads the checkpoint once per worker — `SubgamePolicy`
  is stateless across hands except diagnostic counters, verified; a torch model can't
  be shared across processes). It processes its hand-seed shard and, for each seed,
  plays the hand **three times** (blueprint / PROFILE / BR in the challenger seats,
  same opponent seats), returning the three per-seat equity vectors + that worker's
  `SubgamePolicy.stats()` counters.
- **Reduce.** The parent concatenates per-hand results across workers (keyed by
  `(opp_idx, hand_idx)`) and aggregates: per-(challenger, opponent) diff/stderr/σ
  (matching the baseline format) **plus** the CRN-paired lift (§D). `stats()` counters
  summed across workers.
- **Determinism.** Same `base_seed` + same hand count → **bit-identical** aggregate,
  independent of worker count (per-hand seeds are computed from indices, not execution
  order). This is the reproducibility gate (test, §F).
- **Failure mode — FAIL LOUD, no silent drop.** Use
  `concurrent.futures.ProcessPoolExecutor`; if any worker raises, cancel the pool and
  fail the whole run with the traceback. No auto-retry (a crash is a bug to fix, and
  silently dropping a shard would bias the aggregate). A `--resume` from a per-shard
  checkpoint file is an optional robustness add (out of scope unless a crash recurs).

---

## D. Pre-committed verdict — Decision 6.3 (LOCKED before any compute)

**Metric.** The **lift** `L = mean over the 5 opponents of (diff_BR,o −
diff_blueprint,o)` in ICM-equity-delta, computed **CRN-paired** (each opponent's BR and
blueprint diffs over the *same* per-hand seeds), with the paired stderr → `σ(L)`.
Likewise `L_PROFILE` (PROFILE − blueprint) and `L_BRvsP` (BR − PROFILE). CRN pairs the
starting conditions (deal, stacks, seat assignment); play diverges after the first
action where the challengers differ, so the pairing is **partial** — the paired stderr
is measured, not assumed (Finding 3).

**Variance grounding (empirical, not the prompt's bb/100 assumption).** The baseline
shows per-matchup stderr **~0.0026 at 5,000 hands**, uniform across opponents; pooled
over 5 opponents the mean-diff stderr is **~0.0012**, and the CRN-paired lift stderr is
**≤ that** (pairing only helps). So a lift of ~+0.0024 is already ~2σ — the variance
regime is **far** tighter than "3–5 bb/100, undetectable at 2σ." The magnitude floors
below are set relative to this measured variance and the blueprint's own pool edge
(mean +0.0098 in the baseline): a "meaningful" lift is a non-trivial fraction of that.

**Four-branch verdict (thresholds in ICM-equity-delta, LOCKED):**

- **PASS (strict):** `L ≥ +0.005` **AND** `σ(L) ≥ 2.0` **AND** ordering
  `BR ≥ PROFILE ≥ blueprint` holds (per-opponent means, within noise). A lift ≥ ~half
  the blueprint's mean pool edge, clearly significant — strong architectural
  validation.
- **SUBSTANTIVE_PASS:** `L ≥ +0.002` **AND** `σ(L) ≥ 1.5` **AND** directional
  consistency (`≥ 4 / 5` opponents show positive `diff_BR − diff_blueprint`). A real
  but smaller lift — the regime-asymmetry-reduced outcome the §H lens predicts as most
  likely.
- **AMBIGUOUS:** `L > 0` but (`σ(L) < 1.5` **OR** `L < +0.002`). Directionally
  correct, undermeasured — recommend re-running with more hands (the variance is tight
  enough that 2–4× hands is feasible), or accept the architecture lifts in a regime
  smaller than projected. Does **not** auto-pass or auto-fail; surfaces to the human.
- **FAIL:** `L ≤ 0` **OR** `σ(L) ≥ 2.0 in the wrong direction` (BR significantly worse
  than blueprint). The architecture does not lift strength as measured → do not lock
  BR; surface for diagnosis (the Path-A discipline).

**BR-vs-PROFILE (the revert gate, per Q11) — gated on SIGNIFICANCE, not absolute
magnitude.** Variance grounding for `L_BRvsP = mean_o(diff_BR,o − diff_PROFILE,o)`:
per-opponent diffs have stderr ~0.0026 (5,000 hands); **unpaired**, the difference has
stderr ~0.0026·√2 ≈ 0.0037/opponent, pooled over 5 ≈ **0.0016**. Under that
conservative bound, `+0.001` is only **~0.6σ — within noise**: a fixed `+0.001`
absolute gate would "revert to PROFILE" on noise. CRN pairing makes the real stderr
*lower* — BR and PROFILE play **identically on the ~73% gated-SKIP decisions** (both
fall through to the same blueprint action) and differ only at the ~27% solves where
the leaf-eval mode changes the refined policy, so per-hand paired differences are
sparse → the paired stderr could be several× below 0.0016. But that correlation is
**unmeasured** (Finding: re-measure at Stage 6-C). So the gate is on the **measured
significance**, not a magnitude guess:
- At the conservative unpaired pooled stderr ~0.0016: `L_BRvsP` reaches **1.5σ at
  ≈ +0.0024**, **2σ at ≈ +0.0032**. Under CRN pairing these magnitudes drop in
  proportion to the (measured) paired stderr.
- **Full PASS requires PASS-strict on `L` AND `σ(L_BRvsP) ≥ 1.5` in BR's favor**
  (BR statistically distinguishable from PROFILE) — using the *measured* paired
  `σ(L_BRvsP)`, with a `+0.001` absolute floor only as a triviality guard.
- **If `L` passes (strict or substantive) but `σ(L_BRvsP) < 1.5`:** verdict
  **PASS_BR_EQUIVALENT_TO_PROFILE** — the architecture lifts strength, but BR is **not
  statistically distinguishable from PROFILE at this sample size**, so BR's `v×k`
  complexity is **not justified by the data** → recommend **PROFILE for production**.
  (This is distinct from a confident "revert": it says *we lack the data to prefer BR*,
  not *PROFILE is provably better*.)
- **If `σ(L_BRvsP) ≥ 1.5 in BR's DISfavor** (PROFILE significantly beats BR):** PROFILE
  is the better production choice — recommend reverting to PROFILE.

These are pre-committed: after the run, the verdict is read off mechanically; numbers
are not retuned to the result (the Stage F/G discipline that caught eight findings).

---

## E. Cost model (empirical f≈0.27, per-solve from Stage 5-C)

Per challenger: 5 opponents × 5,000 hands × ~3.04 challenger-decisions/hand ≈ **76,000
decisions**; gate fires on f≈0.27 → ~20,600 solves. Measured per-solve (M=8 depth-3
K=1000): **BR ~6.7 s blended; PROFILE ~3–4 s** (no `v×k`; chance leaves blueprint-only
either way); **blueprint 0 solves** (skips are ~ms).

| challenger | solves | per-solve | sequential | Y=10 (Contabo) | Y=24 (many-core) |
|---|---|---|---|---|---|
| blueprint | 0 | — | ~minutes | ~minutes | ~minutes |
| subgame-PROFILE | ~20,600 | ~3.5 s | ~20 h | ~2.0 h | ~0.8 h |
| subgame-BR | ~20,600 | ~6.7 s | ~38 h | ~3.8 h | ~1.6 h |
| **3-way total** | | | **~58 h** | **~5.8 h** | **~2.4 h** |

The 3-way CRN sweep plays each hand-seed 3× (blueprint+PROFILE+BR) in one worker pass;
the table sums the three. **~5.8 h Contabo-parallel (Y≈10), well under 24 h.** All
numbers are HYPOTHESES pending the Stage-6-C smoke re-measurement (the project rule);
`f` and per-solve will be re-confirmed on the real pool before the full run.

---

## F. Test plan (before the real ablation)

1. **Parallel ≡ sequential (determinism).** On 1 matchup × 100 hands, the parallel
   harness with `--workers 4` produces a **bit-identical** aggregate (diff, stderr, σ,
   counters) to the same harness with `--workers 1`. Locks order-independence /
   per-hand seeding.
2. **CRN pairing is real.** The same per-hand seed yields the same starting state
   (stacks/board/seat assignment/hole cards) across the three challengers (assert on a
   handful of seeds) — the precondition for the paired lift.
3. **Aggregation correctness.** Hand-built per-hand equity arrays → assert the diff,
   stderr (`sqrt(var/n)`), σ, and the CRN-paired lift stderr match a reference NumPy
   computation. Edge cases: `n_capped` hands excluded, single-hand stderr = 0.
4. **End-to-end smoke.** 1 matchup × 100 hands per challenger (mock or real blueprint
   if artifacts) completes, no worker crash, produces a valid result JSON with the lift
   fields and `stats()` per challenger.
5. **Verdict logic.** Unit-test `verdict(L, sigma_L, per_opp, L_brvsp, sigma_brvsp)`
   against hand-fed numbers hitting each branch (PASS / SUBSTANTIVE_PASS / AMBIGUOUS /
   FAIL / PASS_BR_EQUIVALENT_TO_PROFILE), including the σ(L_BRvsP) ≥ 1.5 gate. No game
   needed.

All game-touching tests use the production `six_max_sng` / tournament structure.

---

## G. Staged implementation

- **Stage 6-A — parallelism wrapper + determinism test.** `scripts/eval_pool_ablation.py`:
  per-hand seeding, `ProcessPoolExecutor` shards, reduce. Tests 1–3. Fail-loud worker
  handling. One commit + stop.
- **Stage 6-B — ablation harness + verdict logic.** Wire the 3 challengers + CRN-paired
  lift + the four-branch `verdict()` + JSON output (baseline format + lift fields +
  per-challenger `stats()`). Test 5. One commit + stop.
- **Stage 6-C — smoke run (gate before the real run).** 1 matchup × 100 hands/challenger
  on the real `dcfr-overnight-3000` + pool. Confirm end-to-end (Test 4), **re-measure f
  and per-solve on the real pool**, and re-derive the full-run wall-clock. If f or
  per-solve materially exceed the §E hypotheses, surface before committing the full run.
  One commit (smoke results) + stop.
- **Stage 6-D — the real run + verdict.** Full 3-way × 5 opponents × 5,000 hands under
  `tmux` (the long-compute rule). Apply the locked §D verdict mechanically; write the
  results JSON + a verdict summary; surface AMBIGUOUS/FAIL to the human. This closes
  B1c sub-step 6 and produces the first measured strength delta from subgame solving.

---

## H. Interpretation lens (LOAD-BEARING — read the number through this)

The deployment regime is **not** the regime where Stage F/G measured the BR signal:
**98% of decisions are preflop**; **chance leaves are blueprint-only / 88% bias-
inactive**; **round-closing solves are shallow** (5–12 leaves, blueprint-dominated).
So the BR architecture's measurable lift **concentrates in the minority of chance-free,
decision-bearing solves**. Stage F/G's +3.4σ / +3.07σ were measured on a **different
leaf/decision mix** than deployment produces. **The expected sub-step-6 lift is roughly
+0.002 to +0.005 ICM-equity-delta, NOT a Stage-F/G-scaled large effect** (a naive
"+3–8 bb/100" projection does not survive the regime correction). Accordingly:
**SUBSTANTIVE_PASS (+0.002, σ≥1.5) is genuine architectural validation**; PASS-strict
(+0.005) is strong validation; AMBIGUOUS (positive, σ<1.5) means the architecture
lifts in a smaller regime than projected — a real finding, not a failure. This is why
the verdict has a SUBSTANTIVE branch and an AMBIGUOUS branch, not a single high gate
that would false-FAIL a working architecture.

---

## I. Out of scope

- **Production training run** (100k+ iterations) — that follows the verdict, not part
  of sub-step 6.
- **Abstraction upgrade / hand-specific bet sizing** — separate roadmap items.
- **Layer-4 within-match adaptation** (Track C1) — later phase.
- **`fast_view` canonical fold-in** — separate tracked deliverable.
- **Modifying `eval_pool.py`** — the ablation wraps it; the legacy sequential harness
  stays for the existing baselines.
- **The `dcfr-overnight-3000` ICM-retrain decision** — taken after sub-step 6 measures
  the busted-seat-bias impact (`ae8e1b5`).

---

## J. Failure modes the design handles

1. **Worker crash mid-ablation** — fail-loud (cancel pool, surface traceback); no
   silent shard drop (would bias the aggregate). Per-shard checkpoint/`--resume` is an
   optional add only if crashes recur.
2. **Degraded-solve count outside the expected range** — `SubgamePolicy.stats()`
   `degraded_rate` is aggregated and reported; a degraded_rate above a flag threshold
   (say > 2%, vs the measured 0%) is **surfaced as a finding**, not an auto-fail (the
   degraded path falls through to blueprint, so high degradation biases the lift toward
   0 — a measurement caveat to note, not a crash).
3. **bb/100 / ICM-equity-delta variance much higher than projected** — if the measured
   `σ(L)` at 25,000 hands lands the verdict in AMBIGUOUS, the design's recommendation is
   **more hands** (variance is tight, so 2–4× is feasible in budget) or fewer opponents
   focused on the peers where the lift is most detectable; the verdict does not silently
   resolve it.

---

## Part 4 — findings surfaced

1. **The metric is ICM-equity-delta, not bb/100, and the variance regime is far tighter
   than the prompt assumed.** The harness reports ICM-equity-delta diff with **stderr
   ~0.0026/matchup at 5,000 hands** (a 0.02 effect → σ8), not "3–5 bb/100, undetectable
   at 2σ." The verdict is therefore locked in **ICM-equity-delta + σ** (Q11's native
   metric); bb/100 is at most a loose informal rescaling and is **not** used for the
   gate. The prompt's bb/100 thresholds (+1.5 / +0.5) were re-grounded to ICM-equity-
   delta (+0.005 / +0.002) against the measured variance and the blueprint's pool edge.
2. **Q11 Level 3 pre-committed 3-way + the BR-vs-PROFILE revert gate** — so the design
   honors it (Decision 6.1) rather than the prompt's 2-way option; the BR-vs-PROFILE
   delta is the architecturally-decisive contrast, added as a verdict sub-clause.
3. **CRN pairing requires re-seeding `eval_pool` per hand** (not one stream per
   matchup) — which is also what makes parallelism deterministic. The parallel harness's
   "bit-identical vs sequential" test compares against a **per-hand-seeded** sequential
   run, not the legacy single-stream `eval_pool` (whose RNG semantics differ). CRN is
   **partial** (play diverges after the challengers first differ); the paired lift
   stderr is measured, not assumed.
4. **No existing infra covers the cross-challenger paired lift.** `eval_pool` reports
   per-matchup diffs only; the ablation harness adds the CRN-paired `diff_BR −
   diff_blueprint` (and PROFILE, BR-vs-PROFILE) lift + its paired σ — the quantity the
   verdict actually gates on. `SubgamePolicy.stats()` and `summarize_solve_result`
   supply the per-challenger diagnostics.
