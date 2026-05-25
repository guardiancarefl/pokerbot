# Sub-step 3 — Subgame CFR Loop — Design Proposal (B1c)

**Status:** IMPLEMENTED & CLOSED (session 19) — see the Stage 3-E closure section at
the end. Proposal approved at review gate (both flagged deviations signed off);
implemented in `src/nlhe/subgame_solver.py` (commits `3ef06e4`→`10f7e45` + the 3-E
locking commit). The body below is the as-approved design; line numbers cite files as
read on 2026-05-25 (session 19). Predecessors: tree builder (`subgame.py`, sub-step
1.5, `2be87df`), leaf evaluator (`subgame_leaf.py`, sub-step 2, Stages A–E + Stage
F/G ablation closed via `SUBSTANTIVE_PASS_AGGREGATE`). The Stage-G one-iteration
root stub (`scripts/ablation_decision_level.py:120`, `stub_root_policy`) is the seed
this plant grows from.

**What sub-step 3 is:** the real multi-iteration CFR loop over the depth-limited
subgame tree, warm-started at the blueprint, consuming the validated `BEST_RESPONSE`
leaf values, producing hero's refined root policy. **What it is not:** policy
extraction/sampling (sub-step 4), the `eval_pool.py` `Policy` wrapper (sub-step 5),
the strength measurement (sub-step 6), opponent-range/belief estimation (Track B
item 3), or Layer-4 within-match adaptation (Track C1).

---

## A. Interface

One new module, `src/nlhe/subgame_solver.py`, exposing one function plus two
dataclasses (the `CFR6MaxContext` / `LeafEvalContext` pattern, cfr6.py:82,
subgame_leaf.py:122):

```python
@dataclass
class SubgameSolveContext:
    blueprint: BlueprintProvider          # encoder + policy_nets (subgame_leaf.py:98)
    starting_stacks: Sequence[int]        # length 6, icm_adjust_returns baseline
    payouts: Sequence[float]              # sng_payouts_6max_double_up()
    hero_seat: int                        # the seat whose decision we refine
    n_iterations: int = 1000              # K — see Decision 1
    rng: Optional[random.Random] = None   # only used if tie-break/sampling needed
    num_paid: int = 3
    average_weighting: str = "linear"     # LCFR linear averaging (Decision 5)

@dataclass
class SubgameSolveResult:
    root_policy: np.ndarray               # 7-vector avg strategy, masked, sums to 1 (CONTRACT)
    root_blueprint: np.ndarray            # σ0 = RM+(adv) at the root, for diagnostics
    n_iterations: int
    degraded: bool                        # any degraded leaf reached → mark for sub-step 5 fallback
    converged_l1_tail: float              # mean per-iter root-policy L1 change over last 10% (diagnostic)

def solve_subgame(tree: SubgameTree, ctx: SubgameSolveContext) -> SubgameSolveResult: ...
```

**Pre-condition:** `subgame_leaf.evaluate_leaves(tree, leaf_ctx)` has already been
called, so every `LEAF` node carries `leaf_value` (subgame.py:124) and every
`TERMINAL` carries `terminal_returns` (subgame.py:120). `solve_subgame` reads these;
it never calls the leaf evaluator itself. This is the load-bearing separation that
keeps leaf eval a one-time cost (Decision 1).

**Calling pattern from the eventual `SubgamePolicy` (sub-step 5):**

```python
tree   = build_subgame_tree(state, max_action_depth=K_d, chance_samples_per_node=8, rng=r)
evaluate_leaves(tree, LeafEvalContext(blueprint, biased, stacks, payouts, hero, mode=BEST_RESPONSE))
result = solve_subgame(tree, SubgameSolveContext(blueprint, stacks, payouts, hero))
# sub-step 4 turns result.root_policy into an action; sub-step 5 handles ALLIN→CALL alias
```

The output `root_policy` is a drop-in for the blueprint's own 7-vector action
distribution, so a `SubgamePolicy` conforms to `eval_pool.py`'s `Policy` protocol
(eval_pool.py:62-75) the same way `CheckpointPolicy` does.

---

## B. Algorithm

The solver is **tabular RM+ CFR over the small, finite, pre-built subgame tree**,
with **only hero accumulating regret** (opponents and chance are fixed; Decision 4)
and **full weighted traversal** rather than Monte-Carlo sampling at solve time
(Decision 2). Counterfactual values are backed up as **hero's scalar value** at each
node (we only need hero's value to form hero's regret; the 6-vector leaf values are
sliced to `[hero_seat]`).

**Warm-up (once, before the loop).** Walk `tree.all_nodes`. For each `DECISION`
node, query the blueprint exactly as the stub did (ablation_decision_level.py:295-297;
eval_6max_self_play.py:137-143): `feat = encoder.encode_from_parsed(parsed)`,
`adv = policy_nets.predict_advantages(cp, feat)`, `mask = legal mask from
node.children`. Cache, per node: `adv`, `mask`, and — for **opponent** nodes — the
**fixed** strategy `σ_opp = RM+(adv)` (`solver._strategy_from_advantages`,
solver.py:143). For **hero** infosets, initialise cumulative regret `R[I] = adv`
(the blueprint warm-start — exactly the stub's `adv` term, so K=1 reduces to the
stub, Decision 5 / test 3-B) and average accumulator `S[I] = 0`. All network
forwards happen here; the iteration loop touches no network and no rollout.

**Per-iteration recursion** `value(node) -> float` (hero's counterfactual value):

- **TERMINAL:** return `icm_adjust_returns(node.terminal_returns, starting_stacks,
  payouts)[hero]` (cfr6.py:278-285) — or, when the tournament bubble has burst,
  `compute_icm_payouts(...)[hero]` (cfr6.py:266-275). Cache this per terminal node
  (it is iteration-invariant).
- **LEAF:** return `node.leaf_value[hero]` (subgame.py:124). Iteration-invariant
  (Decision 1 / Q6) — cache the hero slice. If `leaf_value is None` (budget-truncated
  batch, subgame_leaf.py:697) treat as degraded: set `result.degraded`, value 0.0.
- **CHANCE:** `Σ_c node.chance_prob_of(c) · value(c)`, weighting each subsampled
  child by its renormalised `chance_prob` (subgame.py:114-117, 327). No resampling —
  the tree builder already subsampled chance at build time.
- **OPPONENT decision** (`cp != hero`): `Σ_a σ_opp[a] · value(child_a)`, opponent's
  **fixed** blueprint strategy. Opponents never update (Decision 4).
- **HERO decision** (`cp == hero`): for each legal `a`, `v[a] = value(child_a)`;
  `σ = RM+(R[I])`; `ev = Σ_a σ[a]·v[a]`; `r[a] = (v[a] − ev)·mask` (cfr6.py:378 —
  the same instantaneous-regret formula, same ICM-equity units, **not** normalised
  by `starting_stack`, cfr6.py correctness note #5); accumulate `R[I] = max(R[I] +
  r, 0)` (RM+ clamp) and `S[I] += w_t · σ` (w_t = t for linear averaging); return
  `ev`.

After K iterations, `root_policy = S[root_I] / Σ w_t` (the **average** strategy,
not the final iterate — Decision 5). At the root, hero's reach is 1 every iteration,
so the root average is the plain (linearly weighted) mean of `σ_t` — no reach
re-weighting needed for the one infoset we ship.

**Reused vs new.** Reused directly: `_strategy_from_advantages` (RM+, solver.py:143),
`icm_adjust_returns` + `is_tournament_terminal` + `compute_icm_payouts`
(cfr6.py:180-204, 266-285), `encode_from_parsed`/`predict_advantages` (warm-up only),
`build_subgame_tree`, `evaluate_leaves`, and the regret formula already distilled in
`stub_root_policy` (ablation_decision_level.py:120-134). **New** (thin): the tabular
per-infoset `R`/`S` accumulators and the weighted recursion above. `traverse_6max`
itself is **not** reused — it is a training-time, reservoir-buffer-writing,
external-sampling traversal (cfr6.py:380-385), structurally wrong for tabular subgame
CFR (Finding 1).

---

## C. Pre-committed decisions

**Decision 1 — Iteration count: K = 1000 baseline; leaf values cached EXACT across
iterations.** Verified, not assumed: Q6 (SUBGAME_LEAF_DESIGN.md:389-417) and the
code both confirm the leaf value is a function of `(leaf state, k fixed biased
continuations, menu, hero's fixed blueprint)` — none of which the CFR loop mutates
(the Brown/Sandholm 2018 single-pass approximation: BR computed against hero's
**blueprint**, not the iteration-k subgame strategy). So one `evaluate_leaves` call
serves all K iterations; iterations do **no** rollouts and **no** network forwards
(all forwards are warm-up). Iterations are pure arithmetic over a tiny tree, so the
budget allows hundreds–thousands, not Pluribus's ~400 floor as a ceiling. Baseline
**K=1000** (comfortable margin); the **real K is set by a convergence curve in Stage
3-E** (root-policy L1 change per iteration → pick the knee), capped by the W budget.
This pre-commits the *method*, measures the *number* — the project's measurement
discipline (SUBGAME_LEAF_DESIGN.md:246-283).

**Decision 2 — Solve-time traversal: VANILLA full weighted traversal over the built
tree, NOT Monte-Carlo external sampling. (DEVIATES from the documented plan — flagged
for review, see Finding 6.)** The docs say "external-sampling CFR" (STATUS.md:55,
NEXT_SESSION.md:51). I recommend deviating, for three grounded reasons: (i) the
"sampling-variant mismatch ⇒ different regret units" premise is incorrect — regrets
are in ICM-equity units by the formula `v(a)−ev` (cfr6.py:378), independent of how
chance/opponent nodes are visited; external sampling was a *training-time necessity*
(the full-game tree is astronomical), not a units requirement. (ii) The tree builder
stores renormalised `chance_prob` per branch *specifically so the solver can weight
children and "multiply these along the path to recover reach"* (subgame.py:114-117) —
that is a full-traversal design; external sampling would ignore those weights and
re-sample. (iii) The depth-limited tree is *tiny* (~64 leaves / ~150 nodes at depth-3,
the measured Q13 shape), so full traversal is cheap, **deterministic** (free
reproducibility, a test requirement), and converges in tens of iterations vs the
hundreds external sampling needs to average out its variance. Either variant targets
the same fixed point in the same units; vanilla is strictly better here on every axis
except "matches the training traversal's shape," which does not affect correctness.

**Decision 3 — Action abstraction: the SAME 7-action `DiscreteAction` menu as
blueprint training. No expansion.** The tree builder already enforces this — its
decision-node children come from `_discretize_at_decision` (subgame.py:401-425),
which mirrors `cfr6.traverse_6max` (cfr6.py:333-336) exactly. Sub-step 3 enumerates
the children the builder produced; it adds no bet sizes. Pluribus-style decision-time
action *refinement* (new sizings) and pseudo-harmonic translation of off-tree
opponent sizings (actions.py `game_to_policy_action`) are **out of scope** — the
latter lives at the wrapper/eval boundary (sub-step 5), not in the solve.

**Decision 4 — Resolve against the blueprint, not against itself. Opponents fixed;
only hero updates.** Leaf values come from `BEST_RESPONSE` evaluated against hero's
**blueprint** (subgame_leaf.py:475-501) and are frozen for the decision (Decision 1).
Consistent with that, **interior opponent nodes play their fixed blueprint
current-strategy** `RM+(adv)` and never accumulate regret; **only hero** accumulates
regret and refines. This is the exact depth-K generalisation of the Stage-G stub,
which updated hero only (ablation_decision_level.py:120-134). Letting hero's
leaf-reference blueprint drift mid-solve would shift leaf values per iteration and
break the cache/closure (Q6); letting interior opponents *also* update regret is full
multiplayer subgame CFR — a richer variant that needs opponent regret tables, is
*inconsistent* with the blueprint-reference leaf approximation, and is **deferred /
out of scope** (Finding 7).

**Decision 5 — Output: hero's AVERAGE strategy at the root, a masked 7-vector.**
Standard CFR convergence is in the average strategy, not the final iterate, so we
return `S[root]/Σw_t` (linearly weighted, Pluribus used Linear CFR; uniform averaging
is the fallback). At K=1 this equals the Stage-G stub's `σ_M = RM+(adv + r)` because
`R_0 = adv` and a single linear-weighted iterate is just `σ_1` (test 3-B). The richer
artifacts (full per-infoset regret/average tables, per-iteration root trajectory) are
*available for debugging* but are **not** the contract — sub-step 4 reads
`root_policy` only.

**Decision 6 — "Safe subgame solving" is already realised by the BR leaf mechanism;
NO HUNL-style CFV gadget. (Resolves the ARCHITECTURE.md commitment — see Finding 5.)**
ARCHITECTURE.md:26 commits to "safe subgame solving (Pluribus-style)." The honest
reading of *Pluribus-style* in a 6-max ICM setting: the safety mechanism **is** the
depth-limit's multiple biased continuation strategies with the opponent
best-responding among them (Brown/Sandholm 2018 depth-limited solving), which
**sub-step 2 already ships and validated** (Stage F +3.4σ, Stage G +3.07σ,
SUBGAME_LEAF_DESIGN.md:621-719). The DeepStack/Libratus *CFV-gadget* form of safe
subgame solving (Burch et al. 2014; Brown/Sandholm 2017) is a **2-player-zero-sum
construction** — it relies on a scalar exploitability and opponent counterfactual
values at the root, neither of which is well-defined for 6-player ICM. It does **not**
transfer, so it is not "deferred," it is **inapplicable**. Sub-step 3 therefore runs
plain CFR over the already-robust leaf values and adds no root gadget. This honors the
user's instinct (ship the simpler loop, validate strength at sub-step 6) while
satisfying the documented commitment — because the robustness lives in the leaves,
which are done. The *residual* approximation (the opponent's true best response may
lie outside the k=4 biases) is the documented Pluribus-level limitation; shrinking it
(larger k / richer biases) is a leaf-evaluator concern gated by sub-step 6, not a
sub-step 3 deliverable.

**Decision 7 — Budget: comfortably under the 27 s/decision throughput target (Q13).**
See §D. The CFR loop (W) is not the binding term — leaf eval (Z) is, and Q13 already
proved Z fits. The old W=2 s real-time sub-budget (Q4) is superseded by Q13's 27 s
*throughput* framing, leaving ample room.

---

## D. Cost model (per decision)

| Phase | Symbol | Estimate | Provenance |
|---|---|---|---|
| Tree build | Y | ~0.1–0.5 s | `state.child()` calls only; not separately measured (Q4 budgeted 0.5 s) |
| **Leaf eval (one `evaluate_leaves`)** | **Z** | **~10–14 s (BR M=5) / ~20 s (BR M=8), depth-3 CPU** | subgame_leaf.py:151-160; Q13 (`STAGE_E_BUDGET_REDERIVATION.md`) |
| CFR loop (K iterations) | W | **0.109 s at K=1000** (MEASURED, Stage 3-C) | warm-up 1.2 ms + loop 0.109 ms/iter, 216-node depth-3 tree |
| Policy extraction | — | ~0 (accumulated in-loop; one normalise) | — |
| **Total** | X | **~10–14 s (M=5) / ~20 s (M=8)** | < 27 s with wide margin |

**W — measured (Stage 3-C), supersedes the design hypothesis.** The design guessed
~0.6–2 s at K=1000 (~10 µs/node). Measured on a real 216-node depth-3 production tree:
**warm-up (all blueprint forwards) 1.2 ms; CFR loop K=1000 = 0.109 s (0.109 ms/iter,
≈0.5 µs/node) — ~15× UNDER the design estimate.** The loop is pure arithmetic over
cached values (no network forwards / rollouts; Decision 1), so it is far cheaper than
the per-node guess assumed. W is therefore a non-factor: Z (leaf eval) dominates the
27 s budget by ~100×, and Z already fits (Q13 / Finding 4). The measurement is locked
by `tests/test_subgame_solver.py::TestProductionKGated::test_production_k_cost_gate_loop_under_1s`.

---

## E. Test plan (must pass before sub-step 3 is complete; production game per discipline)

1. **`test_k1_reduces_to_stub`.** Build a depth-1 tree from a sampled production root;
   run `solve_subgame` with `K=1`. Assert `result.root_policy` equals
   `ablation_decision_level.stub_root_policy(adv, q, mask)[0]` (the validated Stage-G
   oracle) within `1e-9`. This pins the seed→plant continuity.
2. **`test_trivial_tree_converges_to_best_response`.** Hand-built single-hero-decision,
   2-action subgame, leaf values fixed (opponents fixed): (a) action A strictly
   dominant ⇒ `root_policy[A] → 1` as K grows (assert `> 0.99` at K=1000); (b) A and B
   exactly equal value ⇒ `root_policy` → uniform on legal (assert `|p−0.5| < 1e-2`).
   This is the convergence test; because opponents are fixed (Decision 4) the subgame
   optimum is hero's best response, so "Nash" here = pure-on-dominant / uniform-on-tie.
3. **`test_reproducibility`.** Same seed through build → evaluate_leaves → solve, two
   runs ⇒ `np.array_equal(root_policy_a, root_policy_b)`. Vanilla CFR is deterministic
   given the tree+leaf values, so this is exact, not within-tolerance.
4. **`test_cost_budget`.** Representative depth-3 production tree (~64 leaves), real
   `evaluate_leaves` (BR), `K=1000`. Assert the **CFR-loop** wall-clock (excluding leaf
   eval) `< W_GATE` (2 s) and total decision `< 27 s`. Number re-measured per discipline.
5. **`test_integration_production_game`.** Sampled production root → full pipeline →
   assert `root_policy` is a valid distribution: `sum≈1` on legal, `==0` on illegal,
   all finite, length 7.
6. **`test_leaf_eval_called_once`.** Spy/count: `solve_subgame` performs **zero**
   `evaluate_leaf`/rollout calls and **zero** opponent/leaf network forwards inside the
   iteration loop (all forwards in warm-up). Guards the Q6 caching contract — the
   load-bearing premise of Decision 1.
7. **`test_degraded_leaf_marks_result`.** Stub one leaf's `leaf_value=None` (or a
   degraded eval) ⇒ `result.degraded is True` and `root_policy` is still a valid vector
   (so sub-step 5 can choose blueprint fallback). Mirrors Stage G's degraded handling
   (ablation_decision_level.py:368, G-E #2).

Tests 1, 3, 6 are deterministic gates; 2 is the convergence gate; 4 the cost gate;
5, 7 the integration/safety envelope.

---

## F. Staged implementation plan (sub-step 2's Stages A–E pattern; each = one commit + stop)

- **Stage 3-A — scaffold.** `src/nlhe/subgame_solver.py`: `SubgameSolveContext`,
  `SubgameSolveResult`, `solve_subgame` signature, the per-infoset `R`/`S` table types,
  warm-up that caches `adv`/`mask`/`σ_opp` per node. No iteration loop yet (returns
  `root_blueprint` only). Tests: dataclass + warm-up sanity (σ_opp sums to 1, masks
  correct). One session.
- **Stage 3-B — K=1 reproduces the stub.** Wire the single-iteration regret update;
  pass `test_k1_reduces_to_stub`. This is the bit-identical-to-stub gate — the cheapest
  proof the math is wired right before any multi-iteration logic.
- **Stage 3-C — K>1 vanilla CFR.** The real weighted recursion (chance/opp/hero/leaf/
  terminal) with `R` accumulation + RM+ clamp across iterations and cached
  leaf/terminal values. Pass `test_trivial_tree_converges_to_best_response`,
  `test_leaf_eval_called_once`, `test_degraded_leaf_marks_result`. The heart of
  sub-step 3 — likely more than one session.
- **Stage 3-D — average-strategy extraction.** Linear-weighted `S` accumulation and
  `root_policy` output; `test_integration_production_game`, `test_reproducibility`.
- **Stage 3-E — convergence + cost gate.** Measure the root-policy convergence curve to
  **set K** (the knee), run `test_cost_budget`, confirm W and total fit the 27 s budget
  on a representative decision. Record the measured per-node/per-iteration cost and the
  chosen K in the commit message (the project's measure-before-committing rule).

Sub-step 3 is **complete** when 3-A…3-E are committed and all seven tests are green —
at which point the handoff is to sub-step 4 (policy extraction), not sub-step 6.

---

## G. Explicitly out of scope

- **Policy extraction / action sampling** (turning `root_policy` into a played action,
  incl. the `ALLIN→CALL(1)` alias, NEXT_SESSION.md:96-100) — **sub-step 4/5**.
- **`eval_pool.py` `Policy` wrapper** (`SubgamePolicy`) — **sub-step 5**.
- **Strength measurement** vs the `league-v2-600` pool — **sub-step 6** (Q11 Level 3).
- **HUNL-style safe-subgame CFV gadget** — inapplicable to 6-max ICM (Decision 6).
- **Full multiplayer interior CFR** (interior opponents also updating regret) —
  deferred; inconsistent with the blueprint-reference leaf approximation (Decision 4).
- **Opponent-range / belief-state modeling.** The tree is a **single-deal** forward
  lookahead built from one OpenSpiel state (subgame.py:189-251); opponents' private
  cards are fixed, not a range. Belief-state estimation is **Track B item 3**
  (DECISIONS.md:152), and the "don't solve with knowledge of opponent cards" concern it
  raises is a **sub-step 5** wrapper concern. Sub-step 3 solves the tree as built — the
  same conditioning Stages F/G ran under and accepted. (Finding 8.)
- **Layer-4 within-match adaptation** (Track C1 menu nudging) — later.
- **Re-deriving blueprint regret semantics** — those stay in `cfr6.py`/`solver.py`;
  sub-step 3 reuses `_strategy_from_advantages` and the regret formula, unchanged.
- **Folding `fast_view` into the canonical path** — a separate tracked deliverable
  (NEXT_SESSION.md:102-119), not gated by sub-step 3.

---

## Pre-commitment

The seven Decisions and the Stage 3-A…3-E gates are fixed as of this proposal. Two
deviations from the documented plan are surfaced for explicit sign-off at the review
gate (not designed around): **Decision 2** (vanilla full-traversal CFR instead of the
documented external-sampling CFR) and **Decision 6** (no CFV gadget; Pluribus-style
safety is the already-shipped BR leaf mechanism). The remaining findings (1, 4, 7, 8)
are reported for awareness; none blocks the design.

---

## Stage 3-E closure — sub-step 3 COMPLETE (session 19)

**Production K = 1000.** Chosen from the Stage-3-C measurements, not a fresh sweep
(Stage 3-C already produced both signals): convergence — `converged_l1_tail` falls
4.4e-2 → 4.4e-5 → 4.4e-8 across K=10/100/1000 (the average root policy is stable at
K=1000, ~6 orders of magnitude below any reasonable threshold); cost — the CFR loop
is 0.109 s at K=1000 (§D), negligible against the ~10–20 s leaf-eval Z and ~250×
under the 27 s/decision budget. K=1000 gives comfortable convergence margin, and the
loop is linear in K so escalation is essentially free (K=10000 ≈ 1 s) — the choice is
a convergence-quality decision, not a cost one. Locked by three tests
(`TestProductionK*`): default K=1000, `converged_l1_tail < 1e-5` at K=1000, and the
CFR loop < 1 s at K=1000.

Both review-gate deviations were **approved** in session 19: Decision 2 (vanilla
weighted CFR — methodologically stronger than the documented external-sampling spec,
same pattern as Stage F/G's approved deviations) and Decision 6 (no HUNL CFV gadget;
BR-as-safety, confirmed by Stage G's +3.07σ). One additional resolution: the
`converged_l1_tail` metric tracks the **average (output) root-policy** movement, not
the current strategy — under fixed opponents (Decision 4) the subgame is a
best-response problem and the current strategy collapses to its pure BR in ~7
iterations, so only the average-policy movement is a usable convergence signal.

**This closes sub-step 3** (the real subgame CFR solver: `src/nlhe/subgame_solver.py`,
Stages 3-A scaffold → 3-B K=1 stub-identity → 3-C K>1 loop → 3-D diagnostics → 3-E
K-locking). **Next: sub-step 4 — policy extraction** (turn `SubgameSolveResult.root_policy`
into a played action: argmax / sampling, and the `ALLIN→CALL(1)` translation when
the chosen `DiscreteAction` maps to the chip-0 fold alias), then sub-step 5
(`SubgamePolicy` `eval_pool.Policy` wrapper) and sub-step 6 (the Level-3 pool ablation
— the first measured strength delta from subgame solving).
