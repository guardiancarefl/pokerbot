# Subgame Leaf Evaluator — Design Proposal (B1 sub-step 2)

Status: DESIGN ONLY. No implementation lands with this doc. This is the
review gate before `src/nlhe/subgame_leaf.py` is written. Every decision below
is grounded in code already in the repo; line numbers cite the file as read on
2026-05-24.

## Context and the session-12 tension (read first)

`NEXT_SESSION.md` calls the reverted leaf evaluator "mostly right —
Brown/Sandholm-style k=4 biased continuation strategies, MC rollouts,
ICM-adjusted payoffs." But the same file's lesson #3 is blunt: *"Runtime CFR
with MC equity is not deployable. Push/fold work hit this wall. Subgame solver
must follow the same lesson: precompute where possible, never run unbounded MC
inside the hot path."* These two statements are in tension. This design
resolves it explicitly: we keep the k=4 biased-continuation framework, but the
MC is **bounded, batched, computed once per decision (not per CFR iteration),
and guarded by a wall-clock budget with a non-MC fallback**. The numbers in Q4
are chosen to prove the hot path stays bounded.

The tree builder (sub-step 1.5, commit `2be87df`) hands us `SubgameNode`s of
kind `LEAF` that carry the live OpenSpiel `state`, `depth`, `current_player`,
and `chance_prob`, but `terminal_returns=None` (subgame.py SubgameNode
docstring lines 70–72, 256–269). The leaf evaluator's job is to fill that gap:
assign each leaf a per-player value vector in the *same units* the walker
already backs up at true terminals.

---

## Q1 — Leaf value shape

**Return a 6-vector `tuple[float, ...]` of ICM-equity deltas, one per seat,
length `NUM_SEATS_6MAX`.**

Justification, tied to code:

- The walker's terminal value is `icm_adjust_returns(state.returns(),
  starting_stacks, payouts)` (cfr6.py:278–285), which is length-6 (icm_returns.py
  returns `len(chip_returns)` entries, line 50/68). The tournament-terminal path
  (cfr6.py:261–275) likewise produces a length-6 `icm_payouts`. So at every
  natural stopping point the walker deals in 6-vectors of ICM-equity.
- `SubgameNode.terminal_returns` is already `list[float]` of length 6
  (subgame.py:103, asserted by the test `test_terminal_returns_present`). A leaf
  value must be a **drop-in substitute** for a terminal so sub-step 3's backups
  treat LEAF and TERMINAL nodes identically. That forces the same shape.
- This is multi-player CFR (Pluribus, not HUNL). A scalar "value to the
  traverser" is insufficient because sub-step 3 will run traversals for
  different seats and an opponent-continuation choice at the leaf changes *every*
  player's payoff. A full 6-vector lets one cached leaf value serve every
  traverser.

Units: ICM-equity *delta* vs the hand's starting stacks (`e_end - e_start`,
icm_returns.py:68), **not** raw chips and **not** absolute ICM equity. This
matches cfr6.py:278–285 exactly, so internal-node backups remain in a single
consistent equity space (cfr6.py correctness note #4, lines 47–50). The vector
sums to ~0 (prize pool conserved; icm.py guarantees `sum(icm_equity)=sum(payouts)`,
so the start/end difference telescopes to ~0) — a free invariant for testing (Q10).

---

## Q2 — Continuation strategies: already exposed

**They exist. Do not construct them.** `biased_policy.py:standard_bias_configs()`
(lines 47–78) returns exactly k=4 `BiasConfig`s:

1. `"blueprint"` — identity multipliers `np.ones(n)` (line 74).
2. `"fold-biased"` — FOLD ×α, all bets ×(1/α) (line 75).
3. `"call-biased"` — CALL ×α, all bets ×(1/α) (line 76).
4. `"raise-biased"` — FOLD and CALL ×(1/α), bets unchanged (line 77).

`α` defaults to 3.0 (line 28; comment notes Brown-2018 used 5, we chose 3.0 for
the project's "values-driven robustness" preference, tune in B1b). `BiasedBlueprint`
wraps them and exposes `action_probs(blueprint_probs, legal_mask, strategy_idx)`
(lines 131–140), which calls `apply_bias` (lines 81–109): multiply → mask →
renormalize, with a fallback to uniform-over-legal when the bias zeros all legal
mass (lines 104–108).

Critically, the class **does not run the network** — the docstring (lines
121–123) states the caller "is responsible for running the network once per leaf
infoset and feeding the result here for each of the k strategy choices." So the
leaf evaluator owns the blueprint forward pass and reuses the exact path from
`eval_6max_self_play._sample_action_from_policy` (lines 137–143):
`solver.encoder.encode_from_parsed(parsed)` → `solver.policy_nets.predict_advantages(cp, feat)`
→ RM+ normalize over the legal mask → that 7-vector is `blueprint_probs`.

One caveat to record: there is no distinct "all-in-biased" config; ALLIN sits in
the `BETS` group (line 63) and is left at blueprint weight in raise-biased. Given
the ALLIN→FOLD chip-0 alias documented in `actions.py` (commit `b2dded5`),
raise-biased does not over-weight ALLIN, which is fine — the alias only bites at
translation time, not in the probability vector.

---

## Q3 — Combinatorics

**Hero plays blueprint (bias fixed at index 0). Each still-active opponent at
the leaf independently draws one of k=4 biases. The enumeration count is `4^v`
where `v` = number of opponents with a future decision at this leaf — but we do
not enumerate; we profile-sample, so the cost is `M` rollouts, independent of v.**

The back-of-envelope `4^5 = 1024` is the worst-case enumeration (v=5, all
opponents live). The real v is much smaller and we never pay the enumeration:

- **Exclude folded seats.** A seat that has folded before the leaf has no future
  action; its bias is irrelevant. `parse_state_6max` exposes per-seat
  contribution/active info (used by `_build_view_6max`, cfr6.py:140–166) so the
  evaluator can read which seats are still in.
- **Exclude all-in seats.** A seat already all-in has no decision left in the
  hand; only chance (cards) remains. Its bias never applies. Detect via the view's
  effective-stack / contribution data (cfr6.py:154–159).
- **In late-game SNG / push-fold spots, contested leaves are usually heads-up or
  3-way**, so v∈{1,2} and even full enumeration would be 4–16. With v=2, 4²=16.
- **We sample, not enumerate.** Because the prior factorizes across opponents
  (Q7), drawing each active opponent's bias independently per rollout yields an
  unbiased estimator of the prior-weighted average leaf value. Cost is `M`
  rollouts *total*, not `4^v × M`. This is the single most important
  simplification — it severs leaf cost from v entirely and kills the 1024×
  blow-up that sank the earlier attempt.

So: enumeration count = `4^v`, v = live-opponent count (typically 1–2, ≤5);
actual evaluator cost = `M` sampled rollouts per leaf, prior-weighted in
expectation.

---

## Q4 — MC sample budget, with the multiplication

**Decision-time budget X = 4.0 s per SubgamePolicy decision**, split:

| Phase | Symbol | Budget |
|---|---|---|
| Tree build | Y | 0.5 s |
| Leaf eval (one-time precompute) | Z | 1.5 s |
| Subgame CFR loop | W | 1.5 s |
| Headroom | — | 0.5 s |

X=4 s is defensible for an online 6-max SNG bot: a human tanks several seconds
on hard spots, and the format is not hyper-turbo at the decision level. Y=0.5 s
is backed by the sub-step-1.5 probes (depth-3 tree from a real state built in
0.03 s; a 11k-node depth-4 near-flop tree in ~2 s — so we keep depth ≤3 and
subsample chance to stay well under 0.5 s).

**Leaf eval is a one-time precompute (Q6), so Z is paid once, then the W CFR
loop reuses cached leaf values for free.** The multiplication:

```
Z = L_leaves × M_samples × (avg_rollout_steps × per_step_network_cost)
```

Conservative per-piece values for Contabo CPU:
- `per_step_network_cost` ≈ 0.3 ms (236-dim MLP forward, the dominant cost; the
  encoder + RM+ math is cheap relative to it).
- `avg_rollout_steps` ≈ 8 decisions from a mid-hand leaf to terminal (plus
  near-free chance steps).
- `L_leaves` capped at 50 by tree-build params (depth ≤3 + chance subsample ≤4).
- `M_samples` = 10.

Naïve sequential cost:
```
Z = 50 × 10 × (8 × 0.3 ms) = 50 × 10 × 2.4 ms = 1.2 s   ✓ under 1.5 s
```

That already fits. But we additionally **batch the M rollouts of a single leaf in
lockstep**: all M samples start at the *same* leaf state, so step 1 is one
batched forward of size M; only after actions/cards diverge do later steps spread
to ≤M distinct infosets, which PyTorch batches in one call. Batching turns `M`
forwards-per-step into ~1, cutting Z by roughly M×:
```
Z_batched ≈ 50 × 8 × (one batched forward ≈ 0.5 ms) = 0.2 s   ✓ large margin
```

So the budget holds with margin even before batching, and comfortably with it.
A **hard wall-clock guard** (`time_budget_s`, Q7) degrades `M` (and, last resort,
short-circuits remaining leaves to the Option-A ICM estimate, Q5) rather than
overrunning. This is the explicit answer to the session-12 wall: M is bounded,
the work is one-time per decision, and overruns degrade instead of hanging.

---

## Q5 — ICM at leaf vs projected through rollout

**Recommend Option B (MC-rollout to terminal, then ICM at terminal stacks), with
an Option-A short-circuit for ITM-degenerate leaves.**

Reasoning tied to `icm_returns.py`/`icm.py`:

- `icm_adjust_returns` is *built* for Option B: it consumes `chip_returns` =
  `state.returns()` from a **completed** hand (icm_returns.py:38–48, 59) and
  computes `e_end - e_start`. A rollout produces exactly that terminal
  `state.returns()`, so Option B is a direct, zero-friction call into existing,
  tested code.
- Option A (ICM on *current* stacks at the leaf, no rollout) has no clean way to
  use `state.returns()` (the hand isn't over) and, worse, must attribute the
  **live contested pot** to someone. Mid-hand the pot belongs to no player yet;
  feeding "stack behind, pot excluded" to `icm_equity` systematically *undercounts*
  the all-in upside, and feeding "stack + committed" double-counts across players.
  There is no correct static attribution, so Option A is biased on contested
  leaves.
- **What Option B loses if we used A:** correct resolution of the contested pot
  and the realization variance of the remaining streets — exactly the cases where
  a subgame solve is supposed to add value, so Option A would defeat the purpose.
- **Where they converge:** once `is_itm(stacks, paid_positions)` is true (icm.py:229–237),
  remaining chips have ~0 marginal ICM EV (the "ITM degenerate" note, icm.py:160–163),
  so A and B agree within MC noise. The evaluator short-circuits such leaves to
  the cheap deterministic Option-A estimate `icm_equity(current_stacks) -
  icm_equity(starting_stacks)`. This is a speed win with no accuracy loss and is
  testable (Q10).

`icm_equity` cost is 120 permutations for 6-max/3-paid (icm.py:20–22) — sub-ms,
negligible against the network forwards.

---

## Q6 — Caching strategy

**Compute leaf values once per decision, cache across all CFR iterations within
that decision, recompute fresh on the next decision. Do not cache across
decisions.**

Correctness argument (this is the strong point): the leaf value is a function of
`(leaf state, the k fixed biased continuation strategies, the prior)` — **none of
which the CFR loop mutates.** Sub-step 3 only changes the *root-region* strategy;
the biased continuations below the leaf are frozen blueprint-derived policies.
Therefore the leaf value is invariant across CFR iterations, and caching it across
iterations is **exact, not an approximation**. This is what makes Z a one-time
cost (Q4) instead of `Z × W_iterations`.

- **Invalidation within a decision:** none required. The cache is valid for the
  entire subgame solve; it is discarded when the decision ends.
- **Across decisions — rejected.** Each decision has a different board, stacks,
  and betting line, so the leaf states differ; a cross-decision cache would need a
  key over an equivalence class of leaf states *and* the prior. Staleness risk:
  within-match adaptation (Track C1) nudges the prior between decisions (Q7), so a
  value cached under an old prior is silently wrong. Not worth the correctness
  hazard for a one-time 0.2–1.2 s cost.

Implementation: store the value on the node (proposed new optional field
`leaf_value`, Q9) during the precompute pass; CFR reads it during backups.

---

## Q7 — Opponent prior parameterization

The prior over the k=4 biases is a **parameter, not a constant**. Default uniform
(`1/k` each), matching the uniform weighting the question describes and the
even-handed robustness intent. Track C1 will pass a nudged prior without touching
the evaluator.

Shape: factorized per opponent — either a shared `(k,)` vector or a per-seat
`(NUM_SEATS_6MAX, k)` matrix (per-seat lets C1 model individual opponents). `None`
→ uniform. Sketch:

```python
def evaluate_leaf(
    node: SubgameNode,
    blueprint: BlueprintProvider,          # encode + predict_advantages (see Q9)
    biased: BiasedBlueprint,               # the k=4 configs (biased_policy.py)
    starting_stacks: Sequence[int],
    payouts: Sequence[float],
    *,
    hero_seat: int,
    opponent_prior: np.ndarray | None = None,   # (k,) or (NUM_SEATS_6MAX, k); None=uniform
    n_samples: int = 10,
    rng: random.Random,
    icm_short_circuit: bool = True,
    time_budget_s: float | None = None,
    num_paid: int = 3,
) -> tuple[float, ...]:                          # length NUM_SEATS_6MAX, ICM-equity deltas
```

Per rollout, each active opponent's bias is `rng.choices(range(biased.k),
weights=opponent_prior[seat])`. Sampling from the prior (rather than enumerating)
is what makes cost independent of v (Q3).

---

## Q8 — Failure modes

Leaf-eval failure is contained per-leaf with graceful degradation; only a *total*
failure escalates to sub-step 5's blueprint fallback. Concrete cases:

- **Bias zeros all legal mass** → already handled inside `apply_bias`
  (biased_policy.py:104–108): falls back to uniform-over-legal. No action needed
  here; just don't re-mask after.
- **`icm_equity` with zero total chips** → returns all-zeros (icm.py:69–71).
  Only reachable if every stack is 0 (impossible mid-hand); treated as a neutral
  0-vector if it ever occurs.
- **NaN in a network forward / advantages** → detect with `np.isfinite`; if a
  single sample produces NaN, **drop that sample** and average over the survivors.
- **Rollout doesn't terminate** → OpenSpiel hands always terminate, but guard
  with a `max_steps` cap mirroring `eval_pool.py:122` (500). On cap-exceed for a
  sample, fall back to the Option-A ICM estimate *for that sample* and log at
  WARNING (mirrors `eval_pool.py`'s `exceeded_cap` handling, lines 141–146).
- **All samples for a leaf fail** → fall back to Option-A ICM-at-leaf for the
  whole leaf; log WARNING with the leaf depth + infoset key.
- **Option-A also impossible** (degenerate stacks) → leaf value = 0-vector and set
  a `leaf_eval_degraded` flag on the result so sub-step 5 can decide to abandon the
  solve and play pure blueprint for the decision.
- **Misconfigured blueprint** (wrong-shape advantages, missing weights) → validate
  at evaluator construction, raise there, **never** mid-rollout.

The distinction the question asks for: sub-step 5 falls back to blueprint on
*solver* failure; here, per-leaf failures degrade locally (drop sample → Option-A
→ zero-vector) and never crash the solve. Only the escalation flag connects the
two.

---

## Q9 — Integration point with subgame CFR (sub-step 3)

Pin two entry points:

```python
# Single leaf — used by tests and as the inner call.
def evaluate_leaf(node: SubgameNode, ...) -> tuple[float, ...]   # length 6

# Batch — used by sub-step 3 once, before the CFR loop. Enables lockstep
# batching across leaves and writes the cached value onto each leaf node.
def evaluate_leaves(tree: SubgameTree, ctx: LeafEvalContext) -> None
```

`evaluate_leaves` iterates `iter_leaf_nodes(tree)` (subgame.py:406–410) and sets a
**new optional field** `SubgameNode.leaf_value: Optional[list[float]]` (added in
sub-step 2's commit, alongside the evaluator — it is a data field, not behavior, and
keeps leaf values co-located with the node the same way `terminal_returns` is). Sub-step
3 then backs up uniformly:

```python
value_vec = node.terminal_returns if node.is_terminal else node.leaf_value
```

A `LeafEvalContext` dataclass bundles the fixed deps (blueprint provider, biased
configs, starting_stacks, payouts, prior, n_samples, rng, budget) so the signature
stays short — same pattern as `CFR6MaxContext` (cfr6.py:82–124).

`BlueprintProvider` is a tiny protocol the existing solver already satisfies (it
has `.encoder` and `.policy_nets.predict_advantages`, exercised at
eval_6max_self_play.py:137–139). Defining it as a protocol — not importing the
concrete `DeepCFR6MaxSolver` — keeps `subgame_leaf.py` aligned with the
`Policy`-protocol style of `eval_pool.py:62–64` and makes the eventual
`SubgamePolicy` (sub-step 5) trivially conformable to `CheckpointPolicy`'s
`select_action(parsed, state, rng, mode) -> int` contract (eval_pool.py:74–75).

---

## Q10 — Test plan (before sub-step 3 relies on it)

1. **Determinism / reproducibility.** Same `rng` seed → bit-identical 6-vector.
   For a leaf with no remaining chance (all live players already all-in, only the
   board left but board fixed in the leaf), the value is fully determined; assert
   exact equality across two runs.
2. **ICM extreme stacks.** Build a leaf where one seat holds ~all chips and the
   rest are near-0; assert that seat's equity-delta drives it toward the top
   payout, matching a hand-computed `icm_equity` (use icm.py examples lines 54–61
   and `sng_payouts_6max_double_up`). Tolerance = MC stderr.
3. **Conservation.** The 6-vector sums to ~0 (prize pool conserved by the ICM
   map). Assert `abs(sum(value)) < 1e-6` for the deterministic case and `< few×stderr`
   for sampled cases.
4. **Prior = point-mass on blueprint ≡ pure-blueprint rollout.** With
   `opponent_prior` a delta on index 0, the sampled value must match a direct
   blueprint-only rollout (within stderr). Confirms the bias plumbing.
5. **Permutation invariance.** With symmetric stacks/cards across two opponent
   seats and a symmetric prior, swapping which seat is which leaves the (symmetrized)
   value unchanged within stderr — the "tree-irrelevant permutation of biased
   strategies" check.
6. **ITM short-circuit equivalence.** In an `is_itm()` leaf (icm.py:229), Option-A
   and Option-B agree within stderr; assert the short-circuit doesn't move the value
   beyond MC noise.
7. **Budget guard.** With `time_budget_s` tiny, the call returns a (higher-variance)
   value, respects the budget within a small slack, and never raises.
8. **Failure degradation.** Inject a NaN-producing blueprint stub; assert the
   result degrades to Option-A / zero-vector with `leaf_eval_degraded=True` and a
   logged warning, not an exception.

Tests 1–3 and 6 are deterministic or low-variance and gate sub-step 3; 4–5 validate
the bias/prior semantics; 7–8 validate the safety envelope.

---

## Key decisions (summary)

- **Leaf value = 6-vector of ICM-equity deltas** (`tuple[float, ...]`,
  `NUM_SEATS_6MAX`), identical units to `icm_adjust_returns` at cfr6.py:278–285, so
  LEAF and TERMINAL back up identically.
- **k=4 continuation strategies already exist** in `biased_policy.py`
  (`standard_bias_configs`, blueprint/fold/call/raise-biased, α=3.0); the evaluator
  only supplies the blueprint forward pass and the prior.
- **Combinatorics = `4^v`** (v = live opponents, usually 1–2, ≤5) **but we
  profile-sample**, so real cost is `M` rollouts independent of v — this kills the
  1024× blow-up.
- **Budget X=4 s = Y 0.5 (build) + Z 1.5 (leaf) + W 1.5 (CFR) + 0.5 headroom.**
  Leaf math: `50 leaves × 10 samples × 8 steps × 0.3 ms = 1.2 s` sequential, ~0.2 s
  with lockstep batching. Bounded, with a wall-clock guard that degrades M.
- **ICM via Option B** (rollout → `state.returns()` → `icm_adjust_returns`), with
  an Option-A short-circuit when `is_itm()` (no accuracy loss, pure speed).
- **Cache leaf values per decision, across CFR iterations** — *exact*, since
  continuations are frozen; **never across decisions** (prior staleness under
  Track C1).
- **Prior is a parameter** (`opponent_prior`, `(k,)` or `(seats,k)`, default
  uniform); Track C1 nudges it without touching the evaluator.
- **Failures degrade locally** (drop NaN sample → Option-A → zero-vector +
  `leaf_eval_degraded` flag), distinct from sub-step 5's whole-decision blueprint
  fallback.
- **Integration:** `evaluate_leaves(tree, ctx)` populates a new
  `SubgameNode.leaf_value` field once before the CFR loop; `evaluate_leaf(node,
  ...)` is the testable inner call; deps bundled in a `LeafEvalContext` (mirrors
  `CFR6MaxContext`).
- **Blueprint accessed via a small protocol** the existing solver already
  satisfies (`.encoder` + `.policy_nets.predict_advantages`), keeping
  `SubgamePolicy` conformable to `eval_pool.py`'s `Policy`/`CheckpointPolicy`
  contract.
- **Test gate before sub-step 3:** determinism, ICM-extreme vs hand-computed,
  conservation (sum≈0), prior-point-mass ≡ blueprint rollout, permutation
  invariance, ITM short-circuit equivalence, budget guard, failure degradation.
