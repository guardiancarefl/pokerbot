# Sub-step 5 — SubgamePolicy Wrapper — Design Proposal (B1c)

**Status:** PROPOSAL (review gate). No implementation lands with this doc. Line
numbers cite files as read 2026-05-26 (session 20). Predecessors: the subgame solver
(`subgame_solver.solve_subgame` / `extract_action`, sub-steps 3–4, CLOSED), the leaf
evaluator (`subgame_leaf.evaluate_leaves`, sub-step 2), and the tree builder
(`subgame.build_subgame_tree`, sub-step 1.5).

**What sub-step 5 is:** `SubgamePolicy`, a class conforming to the
`eval_pool.Policy` protocol so it is a drop-in challenger in the existing pool
harness (`scripts/eval_pool.py`) with **no harness changes**. On each hero decision
it chains build → `evaluate_leaves` → `solve_subgame` → `extract_action`, behind a
**gate** that falls through to the blueprint when refinement can't matter — the gate
is the load-bearing knob that makes sub-step 6's wall-clock feasible.

---

## A. Interface

```python
class SubgamePolicy:                       # conforms to eval_pool.Policy
    name: str
    def __init__(self, name, ckpt_path, abstraction, structure, *,
                 leaf_mode=LeafEvalMode.BEST_RESPONSE, n_samples=8,
                 n_iterations=1000, max_action_depth=3, chance_samples_per_node=8,
                 min_legal_actions=3, max_blueprint_prob=0.95,
                 payouts=None, num_paid=3): ...
    def select_action(self, parsed, state, rng, mode: str = "sample") -> int: ...
    def stats(self) -> dict: ...           # diagnostics (Decision 5.2)
```

`__init__` mirrors `CheckpointPolicy` (eval_pool.py:69-72): `_load_solver(ckpt_path,
abstraction, structure)` → `self.solver` (the `BlueprintProvider`: `.encoder` +
`.policy_nets`). It also builds `self.biased = BiasedBlueprint()` (k=4 configs for
leaf eval) and `self.payouts = payouts or sng_payouts_6max_double_up()`.

`select_action` is the exact `Policy` contract (eval_pool.py:62-64): takes the parsed
dict + live OpenSpiel `state` + caller `rng` + `mode`, returns one legal chip int.
`mode` ("sample"/"argmax") flows to **both** the blueprint fall-through and
`extract_action` (so a `--mode argmax` pool run is greedy end-to-end).

For sub-step 6's three challengers: **subgame-BR** = `leaf_mode=BEST_RESPONSE`,
**subgame-PROFILE** = `leaf_mode=PROFILE_SAMPLE` (same gate, same K — only the leaf
semantics differ), and **blueprint** = the existing `CheckpointPolicy` (no wrapper).

---

## B. Decision pipeline (`select_action`)

```
1. cp = parsed["current_player"]                       # hero seat for this solve
2. discrete_to_chip = _discretize_at_decision(state)   # legal DiscreteActions
   blueprint_probs  = RM+(predict_advantages(cp, encode(parsed)))  masked  # σ0
3. GATE (Decision 5.1): if len(discrete_to_chip) < min_legal_actions
                          OR max(blueprint_probs) >= max_blueprint_prob:
        self.n_skipped += 1
        return _blueprint_action(parsed, state, rng, mode)   # fall through
4. starting_stacks = [money[i] + contribution[i] for i in range(6)]   # Finding 1
5. tree  = build_subgame_tree(state, max_action_depth, chance_samples_per_node, rng)
6. batch = evaluate_leaves(tree, LeafEvalContext(self.solver, self.biased,
              starting_stacks, self.payouts, hero_seat=cp, mode=leaf_mode,
              n_samples=n_samples, rng=rng, num_paid=num_paid))
7. result = solve_subgame(tree, SubgameSolveContext(self.solver, starting_stacks,
              self.payouts, hero_seat=cp, n_iterations=n_iterations, rng=rng,
              num_paid=num_paid))
8. DEGRADED (Decision 5.2): if result.degraded or batch.partial_eval_degraded:
        self.n_degraded += 1; log.warning(...); 
        return _blueprint_action(parsed, state, rng, mode)   # fall through
9. self.n_solved += 1
   return extract_action(result, state, rng, mode)
```

`_blueprint_action` is `_sample_action_from_solver(self.solver, parsed, state, rng,
mode)` — the **identical** selection the opponent `CheckpointPolicy` uses
(eval_pool.py:75), so a skipped/degraded decision is played exactly as the pure
blueprint would. Step 2's `blueprint_probs` reuses the encode→predict→RM+ path
(`eval_6max_self_play._sample_action_from_policy`, lines 137-162); the solver's
warm-up recomputes σ0 at the root (one extra forward, negligible vs Z).

---

## C. Pre-committed decisions

**Decision 5.1 — Gate: solve iff `≥ min_legal_actions (=3)` legal discrete actions
AND blueprint `max action prob < max_blueprint_prob (=0.95)`.** This is Option B.
Justification: Stage G measured `filter_rate = 21.9%` of contested roots have `<3`
legal actions (forced fold/shove) — so ~78% clear the first condition; the
`max-prob < 0.95` condition drops decisions where the blueprint is already
near-deterministic (a clear fold, a clear value bet) and a refined policy would
return essentially the same action. Both are cheap, code-checkable conditions
computed from `discrete_to_chip` + the one blueprint forward the wrapper needs
anyway. We pre-commit to Option B (not A "always solve", not C "late-street/pot
gating") for two reasons: A is wall-clock-marginal on Contabo at M=8 (§D), and C
over-engineers before sub-step 6 tells us whether the simpler gate suffices. **Door
open to C:** if sub-step 6's measured runtime is too slow, add a late-street
condition (`street_idx >= 2`) — a one-line tightening, not a redesign. The gate is
applied **identically** to the BR and PROFILE challengers so their comparison is
apples-to-apples.

**Decision 5.2 — Degraded → fall through to blueprint; log + count; no back-off.**
When `solve_subgame(...).degraded` (a leaf lacked a value) or
`evaluate_leaves(...).partial_eval_degraded` (budget cut the batch — won't fire at
the default `time_budget_s=None`, included defensively), `SubgamePolicy` returns the
blueprint action (sub-step 4 deferred this; sub-step 5 owns the blueprint, so it is
the right place — see SUBSTEP_4_DESIGN Decision 4.5). It **logs** the incident at
WARNING and increments `self.n_degraded`; `stats()` returns
`{n_solved, n_skipped, n_degraded}` per matchup for sub-step 6's interpretation. **No
temporal back-off** — each decision is independent, degraded is rare, and back-off
would add per-hand state for no benefit. The next decision solves normally.

**Decision 5.3 — No across-decision caching; the wrapper is stateless between
decisions (except diagnostic counters).** Each hero decision faces a different game
state, so each builds a fresh tree, runs its own `evaluate_leaves` (which resets the
encoder bucket cache once, Stage E) and its own solve. Considered and rejected:
caching trees/leaf-values across decisions — the leaf value is keyed on `(leaf state,
biased menu, blueprint)` and every decision's tree is a different state, so there is
nothing to reuse (SUBGAME_LEAF_DESIGN Q6 already rejected cross-decision leaf caching
under Track-C1 menu staleness). The encoder's bucket cache persists across decisions
as a *deterministic* perf cache (`bucket_of` is seeded by card hash, rng discarded —
session-15 finding), which is harmless and correctness-neutral. The counters are the
only mutable state.

**Decision 5.4 — Constructor parameters & defaults.** `name`, `ckpt_path`,
`abstraction`, `structure` (blueprint, as `CheckpointPolicy`). Tree:
`max_action_depth=3` (the validated Z-cost regime — Q13/Stage-3-C measured depth-3;
the leaf-eval cost knob), `chance_samples_per_node=8` (builder default). Solve:
`n_iterations=1000` (**locked Stage 3-E**). Leaf eval: `leaf_mode=BEST_RESPONSE`,
`n_samples=8` (**Stage F/G production config**, `ablation_leaf_eval._make_ctx:226-229`).
Gate: `min_legal_actions=3`, `max_blueprint_prob=0.95`. Format:
`payouts=sng_payouts_6max_double_up()`, `num_paid=3`. `extract`/blueprint `mode`
comes from `select_action`'s arg (the harness's `--mode`), not the constructor.

---

## D. Cost model & sub-step-6 projection

Per **solved** decision (Q13 / Stage-3-C measured, re-measure at the M=8 config):

| Phase | Cost | Source |
|---|---|---|
| Tree build (depth-3) | ~0.1–0.5 s | `state.child()` calls |
| **Leaf eval (`evaluate_leaves`, BR M=8)** | **~15 s** (M=5 was 10 s, ×1.6 for M=8) | Q13 `STAGE_E_BUDGET_REDERIVATION` |
| CFR solve (K=1000) | ~0.11 s | Stage 3-C measured |
| Extraction | ~0 | one discretize + draw |
| **Total / solved decision** | **~15 s** | leaf-eval-dominated |

A **skipped** decision is one blueprint forward (~ms). PROFILE leaf eval is ~3.5 s
(no `v×k` BR multiplier), so the PROFILE challenger is ~4× cheaper per solve.

**Sub-step-6 projection (per challenger):** 5 opponents × 5000 hands × ~3.04
challenger-decisions/hand ≈ **76,000 challenger decisions**. Gate fire-rate `f`:
~78% clear `≥3 actions` (Stage G); the `max-prob<0.95` condition is **unmeasured** —
estimate it passes ~70% of those, so **`f ≈ 0.55`** (HYPOTHESIS — instrument in
Stage 5-A, per the project's "numbers are hypotheses pending re-measurement" rule).
Solved decisions ≈ 76,000 × 0.55 ≈ 42,000 × ~15 s ≈ **627,000 s**:

| | f = 0.55 (Option B) | f = 1 (Option A) |
|---|---|---|
| **Sequential** | **174 h — INFEASIBLE** | 317 h |
| Parallel Y=10 (Contabo) | **17.4 h** ✓ | 31.7 h ✗ (>24 h) |
| Parallel Y=24 (many-core) | **7.3 h** ✓ | 13.2 h |

**Two load-bearing takeaways** (both surfaced in Part 4): (1) the projection is only
feasible **with parallelism**, which `eval_pool.py` does **not** currently have
(sequential 174 h) — sub-step 6 must add a parallel harness. (2) Under parallelism,
the **gate is what keeps Contabo under 24 h at M=8** (17.4 h gated vs 31.7 h
ungated), validating Option B over Option A. The BR challenger (~17.4 h) + PROFILE
(~4 h) + blueprint (minutes) ≈ the Q13 ~10.5 h estimate at Y≈24.

---

## E. Test plan (production game where a state is needed)

1. **Protocol conformance.** `SubgamePolicy` has a `name: str` and
   `select_action(parsed, state, rng, mode) -> int` with the right arg names/return
   type; `isinstance` against the `Policy` protocol passes; it can be dropped into a
   `seat_to_policy` slot.
2. **Gate fires correctly.** A constructed/mock blueprint that is near-deterministic
   at the root (max prob ≥ 0.95) ⇒ gate SKIPS (solver not invoked — assert via a spy
   that `build_subgame_tree`/`evaluate_leaves` are not called); a mixed blueprint
   with ≥3 actions and max prob < 0.95 ⇒ gate SOLVES. Also: `<3` legal actions ⇒ skip.
3. **Degraded falls through to blueprint.** Force `solve_subgame` to return
   `degraded=True` (monkeypatch) ⇒ `select_action` returns the blueprint action and
   `n_degraded` increments.
4. **Reproducibility.** Same `SubgamePolicy` config + same state + same `rng` seed ⇒
   bit-identical chip across two calls (vanilla solve is deterministic; the only rng
   use is leaf-eval sampling + extraction, both seeded).
5. **Mock-blueprint integration.** Full pipeline (gate→build→eval→solve→extract) on a
   real `six_max_sng` decision with a mock blueprint ⇒ returns a legal chip
   (∈ `discrete_to_chip.values()`).
6. **`starting_stacks` reconstruction (Finding 1 guard).** On a state from
   `sample_starting_state` with known sampled stacks, assert
   `[money[i]+contribution[i]]` equals the sampled hand-start stacks. Locks the
   reconstruction the whole ICM path depends on.
7. **End-to-end smoke.** Play one full hand with `SubgamePolicy` (real, small
   blueprint if artifacts present; else mock) vs `CheckpointPolicy`/random through
   `play_one_hand_two_policies` ⇒ completes without crash, no invalid action,
   terminal reached (`exceeded_cap == False`).

Tests 1–6 are unit/fast; 7 is the integration smoke. Tests needing a real solver skip
gracefully off-host (the `_find_artifacts` pattern); the gate/degraded/reconstruction
tests use a mock blueprint and run anywhere.

---

## F. Out of scope

- **Strength measurement** (BR vs PROFILE vs blueprint diffs/σ) — **sub-step 6**.
- **Parallelism for the sub-step-6 run** — sub-step 6's harness (but it is a hard
  *prerequisite* for feasibility; see §D / Part 4).
- **Layer-4 within-match adaptation** (Track C1 menu nudging) — later.
- **New abstraction layers** — the production 7-action set is fixed.
- **`fast_view` fold-in** — separate tracked deliverable.
- **A real-time `time_budget_s`** — default `None` (offline throughput regime, Q13);
  a deployment-latency budget is a later concern.

---

## G. Staged implementation plan

- **Stage 5-A — scaffold + gate + conformance.** The class, `__init__`
  (load solver, biased, params, counters), `select_action` with the gate +
  blueprint fall-through only (no solve yet — gate-skip returns blueprint), `stats()`.
  Tests 1, 2, 6 (conformance, gate, `starting_stacks` reconstruction). Instrument the
  gate over a sample of production decisions to **measure `f`** (replaces the §D
  hypothesis). One commit.
- **Stage 5-B — full pipeline + smoke.** Wire build→`evaluate_leaves`→`solve_subgame`
  →`extract_action` into the solve branch. Tests 5, 7 (mock integration, end-to-end
  one-hand smoke). One commit + stop.
- **Stage 5-C — degraded handling + reproducibility + diagnostics.** Degraded
  fall-through, the `n_degraded`/`stats()` plumbing, tests 3, 4. Full suite for 0
  tracked regressions. One commit + stop.

---

## Part 4 — findings surfaced

1. **Interface gap — `starting_stacks` is not in `Policy.select_action`.** The
   blueprint (`CheckpointPolicy`) never needs it (a one-shot policy query), but the
   subgame solver + leaf evaluator need the hand-start per-seat stacks for ICM.
   `parse_state_6max` exposes `money` (current) and `contribution` ("chips committed
   this hand"), so `starting_stacks[i] = money[i] + contribution[i]` reconstructs it
   with **no harness change**. This is non-obvious and load-bearing for every ICM
   value in the solve — Test 6 verifies it against `sample_starting_state` before any
   solve is trusted (do not assume the `contribution` semantics; measure).
2. **`eval_pool.py` is SEQUENTIAL — sub-step 6 is infeasible without parallelism.**
   The §D projection (17.4 h on Contabo) assumes Q13's parallelism Y≈10–24; the
   *current* harness runs one hand at a time (≈174 h sequential at the gated rate,
   ≈317 h ungated). Parallelism is explicitly sub-step 6's job (out of sub-step 5
   scope), but the wall-clock budget **depends** on it, so sub-step 6 must add a
   parallel wrapper (process-shard hands/matchups) or the run does not complete in any
   reasonable budget. Flagged now so it is planned, not discovered.
3. **The gate fire-rate `f` is a hypothesis.** `≥3 actions` is grounded in Stage G
   (~78%); `max-prob<0.95` is not measured. Stage 5-A instruments the gate over real
   decisions to replace the `f≈0.55` estimate before the sub-step-6 budget is
   committed.
4. **No existing subgame-policy infra to reuse** — `SubgamePolicy` is new; the
   pattern to match is `CheckpointPolicy`, and the blueprint fall-through reuses
   `_sample_action_from_solver` verbatim. The `CheckpointPolicy` interface needs
   nothing beyond what's above.

---

## Stage 5-A measurement findings and regime asymmetry

Measured at Stage 5-A close (`scripts/measure_gate_rate.py`, 500 self-play hands on
the `dcfr-3000` blueprint, 3000 decisions):

- **Empirical gate fire-rate `f = 0.271`** — below the §D `f≈0.55` hypothesis (the
  `max-prob<0.95` half was over-estimated; many turbo-SNG spots are forced shove/fold
  or near-deterministic blueprint). Below the [0.45, 0.65] re-derivation band, the
  "pleasant surprise" case: **proceed with the gate as-is.** Sub-step-6 wall-clock
  re-derived at `f=0.271`: ~20,600 solves/challenger × ~15 s ≈ **~8.6 h Contabo-
  parallel (Y≈10) / ~3.6 h many-core (Y≈24)** — comfortably inside budget (was 17.4 h
  / 7.3 h at `f=0.55`). Sequential is still ~86 h — the parallelism prerequisite
  (Finding 2) stands.

- **98.2% of natural decisions are PREFLOP** (preflop 2946 / flop 41 / turn 9 /
  river 4, of 3000). This turbo SNG at the sampled blind levels is short-stacked and
  push/fold-dominated — most hands resolve before a flop. The gate's solves are
  correspondingly ~97% preflop (791 of 813; per-street solve rates preflop 0.268,
  flop 0.463, turn 0.222, river 0.250).

**The regime asymmetry (load-bearing for sub-step-6 interpretation).** This deployment
distribution is the **opposite** of where Stages F/G found the strongest BR signal.
Stage F per-(leaf,opp) resolution climbed **preflop 1.5% → flop 3.5% → turn 10.4% →
river 18.0%**; Stage G's resolution concentrated on the river (M=32: river 8/12 vs
preflop 3/13). The architecture's measurable lift is largest late-street; the bot
overwhelmingly *fires* the solver preflop. **Implication:** sub-step 6's measured
strength delta will be dominated by the **preflop** regime — the weakest-signal regime
in the validation. A rough expectation: **half to a quarter** of what the Stage G
aggregate signal would project under uniform regime weighting. **This is not a failure
mode** — the architecture works directionally (Stage F +3.4σ, Stage G +3.07σ); the
regime where it most clearly *demonstrates* lift simply differs from the regime where
the bot most often *deploys* it. Both are true at once. The finding informs how to
read sub-step 6's number (and may later motivate revisiting tree depth or the gate for
preflop spots); it does **not** block Stage 5-B — the implementation proceeds as
designed.

---

## Stage 5-C closure — sub-step 5 COMPLETE (session 20)

Sub-step 5 is closed. `SubgamePolicy` (`src/nlhe/subgame_policy.py`) is a drop-in
`eval_pool.Policy` challenger: gate → SKIP (blueprint) / SOLVE
(build→evaluate_leaves→solve→extract) → degraded → blueprint fall-through with a
WARNING + `n_degraded` (no temporal back-off, decisions are independent). `stats()`
exposes the four counters + `gate_skip_rate` / `gate_solve_rate` / `degraded_rate`.

**Stage-5 findings (all commits):**
- **Regime asymmetry** (`4cf3fe7`): deployment is **98.2% preflop**; Stage F/G's
  strongest BR signal is late-street. Tempers sub-step-6 magnitude expectations.
- **Gate fire-rate `f ≈ 0.271`** measured (`6ab60be`, `scripts/measure_gate_rate.py`),
  below the 0.55 hypothesis (the "pleasant surprise" branch → proceed).
- **Chance-leaf parse crash** (finding #7) fixed (`03576eb`): `_best_response_biases`
  / `_option_a` crashed on chance-node leaves (`observation_string(-1)`); now parse
  chance-safely. Surfaced at Stage-5-B integration; Stage F/G never exercised it.
- **Chance leaves ~88% bias-INACTIVE** (`03576eb`, measured 22/25): BR adds ~zero
  signal there.
- **Tree-builder leaf explosion** (finding #8) fixed (`9ff106d`): chance branched ×8
  without consuming the action-depth budget, compounding across streets → a
  round-closing depth-3 tree blew up to **2560 leaves**, and the chance-leaf fix made
  them all evaluable → a single solve took **>666 s**. Two coordinated changes:
  (A) chance collapses to a LEAF (transparent; the rollout draws the board) →
  **2560 → 5–12 leaves**, and (B) chance leaves use **blueprint-only** evaluation
  (skip the `v×k` BR, justified by 88% bias-inactivity). Also a correctness fix — a
  depth-limited solver should not expand chance into the tree.
- **Per-solve cost (production M=8 depth-3 K=1000):** chance-free ~12–22 s (≈ Q13),
  chance-reaching ~5 s; **blended ~6.7 s** over a 50-decision benchmark (per-skip
  ~0.9 ms; 0 degraded). Bit-identity to the Stage-G stub preserved throughout.
- **Sub-step-6 wall-clock at f≈0.27:** ~20,600 solves/challenger × ~6.7 s ≈
  **~3.8 h Contabo-parallel (Y≈10) / ~1.6 h many-core (Y≈24)** — comfortably <24 h.
  Feasible end-to-end.

**LOAD-BEARING context for sub-step 6 interpretation.** The compound of (98% preflop)
+ (chance leaves 88% bias-inactive, blueprint-only) + (round-closing solves are
shallow, blueprint-dominated) means the BR architecture's measurable lift **in
deployment** is concentrated in the **minority of chance-free, decision-bearing
solves**. Stage F/G's aggregate signal magnitudes (+3.4σ / +3.07σ) were measured on a
**different leaf/decision mix** than deployment produces. Sub-step 6's measured bb/100
is the **deployment-mix reality** and may be **substantially below** what the Stage
F/G aggregate would naively project. This is not a defect — the architecture works
directionally; the regime where it most clearly demonstrates lift differs from the
regime where the bot most often deploys it. Read sub-step 6's number through this lens.

**Next: sub-step 6** — the Level-3 pool ablation (subgame-BR vs subgame-PROFILE vs
blueprint over `league-v2-600` × 5,000 hands), which needs hand-level multiprocessing
(`eval_pool` is sequential — Finding 2). The first measured strength delta from
subgame solving.
