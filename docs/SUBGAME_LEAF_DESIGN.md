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

**Revision (best-response form).** This version replaces the original
profile-sampling leaf value with an **opponent-best-response** leaf value, using
the Brown/Sandholm 2018 single-pass depth-limited approximation: best-response is
computed against hero's *blueprint* strategy at the root, **not** hero's
iteration-k subgame strategy. The maximization — not a prior-weighted average —
is the mechanism that makes the solve robust. This raises per-leaf cost and moves
the production path onto a **rented GPU (A100/H100)**; the CPU path becomes the
fallback / ablation / debug path. Q3, Q4, Q6, Q7 carry the consequences and Q8–Q11
add the new machinery; Q1, Q2, Q5 are unchanged.

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

**Per leaf the evaluator enumerates `v × k` continuation evaluations — `v` live
opponents each independently trying their `k=4` biases — NOT `k^v` (joint) and
NOT a single averaged profile. Hero plays blueprint (bias index 0, fixed).**

This is the Brown/Sandholm 2018 independent-best-response approximation. Rather
than jointly optimizing all opponents' biases (`k^v`, intractable: v=5 → 1024) or
averaging over a prior (the original profile-sampling proposal), **each opponent
independently best-responds among its k biases** while the other opponents and
hero are held at blueprint. The `4^5 = 1024` back-of-envelope was the *joint*
enumeration; independent BR collapses it to `v × k` (v=5 → 20).

For each live opponent i and each bias b∈[0,k): estimate opponent i's own leaf
value `V_i(b) = E[icm_delta_i | hero=BP, opp_i=b, others=BP]` by M rollouts, then
`b_i* = argmax_b V_i(b)`. The returned leaf 6-vector is the rollout EV with hero
at blueprint and every opponent at its `b_i*` (one final M-rollout pass under the
composed profile). So per leaf:

- evaluations: `v × k` (one per opponent-bias) + 1 (composed-profile pass);
- rollouts: `(v × k + 1) × M`.

The live-opponent reduction still applies and now matters more (it scales the
whole `v×k` term):

- **Exclude folded seats** — no future action; bias irrelevant. Read per-seat
  active/contribution from `parse_state_6max` (used by `_build_view_6max`,
  cfr6.py:140–166).
- **Exclude all-in seats** — no decision left, only chance; bias never applies
  (cfr6.py:154–159).
- In late-game SNG / push-fold, contested leaves are typically heads-up or 3-way,
  so v∈{1,2}: `v×k` = 4–8 evaluations, `(v×k+1)×M` ≈ 50–90 rollouts at M=10.

**Profile-sampling** (the original `M`-rollouts-per-leaf, prior-weighted average,
cost independent of v) is retained only as the `PROFILE_SAMPLE` mode (Q9) — a
CPU-fallback / ablation path, not production. It is cheaper but non-robust (Q6).

---

## Q4 — MC sample budget, with the multiplication

Best-response costs `(v×k+1) × M` rollouts per leaf (Q3) vs profile-sampling's
`M`. The original doc's network-only math (`Z = 50 × 10 × 8 × 0.3 ms = 1.2 s`)
becomes, for v=2 (v×k=8), `≈ 9.6 s` on CPU — the figure the reviewer cited. That
does not fit a 1.5 s Z, so **the production path moves to a rented GPU and the
CPU path becomes fallback / ablation**.

**Measured per-step floor (this matters more than the network cost).** A sandbox
measurement (representative of Contabo CPU) of one rollout decision-step's
state-prep — `parse_state_6max` + `_build_view_6max` + `discretize_legal_actions`
— is **≈ 0.9 ms/step**, dominated by `parse_state_6max`'s regex parse of the
observation string. `state.child()` is ≈ 0.1 ms; `information_state_string` is
≈ 0.001 ms. This state-prep is **CPU-bound and GPU-invariant**: a GPU drops the
network forward (~0.3 ms CPU → ~0.05 ms A100/H100) but does nothing for the
0.9 ms parse. The original 0.3 ms-only budget undercounted real per-step cost by
~4×. Honest per-step cost is `state_prep + forward`:

- CPU: ~0.9 + 0.3 = **~1.2 ms/step**
- GPU (network batched/cheap, parse unchanged): ~0.9 + ~0 = **~0.9 ms/step**

So **GPU alone barely helps** — the parse dominates. BR-form total work is
`total_steps = L × (v×k+1) × M × avg_steps`. For L=50, v=2, M=10, avg_steps=8:
`50 × 90 × 8 = 36,000 steps`. At 0.9 ms that is ~32 s even on GPU. **The required
optimization is cutting per-step state-prep, not just the network:**

1. **Cache `parse_state_6max` by state** within the precompute (the v×k passes
   revisit the same early-street infosets).
2. **Batch the encoder forward** across a leaf's live rollouts (lockstep), so the
   GPU forward is ~free per step.
3. A lighter rollout-only parse that skips fields the bias path does not read.

Target per-step ≈ **0.15 ms** (parse cached/lightened + batched GPU forward).

**Revised budget — X raised to 6.0 s** (production = rented GPU, BR leaves):

| Phase | Symbol | Budget |
|---|---|---|
| Tree build | Y | 0.5 s |
| Leaf eval (BR, one-time precompute) | Z | 3.0 s |
| Subgame CFR loop | W | 2.0 s |
| Headroom | — | 0.5 s |

The increase from 4 s is driven by (a) the `v×k` BR multiplier and (b) the
measured CPU-bound parse floor that GPU does not remove. 6 s/decision is still
acceptable for non-turbo online 6-max SNG. Production-fit arithmetic (Z ≤ 3.0 s),
keeping the tree shallow so `L≤30` with `M=8`:

```
total_steps = L × (v×k+1) × avg_steps × M        (v=2)
            = 30 ×    9    ×    8     × 8  = 17,280 steps
Z = 17,280 × 0.15 ms (optimized GPU step) ≈ 2.6 s   ✓ under 3.0 s
```

Without the state-prep optimization (0.9 ms/step): `17,280 × 0.9 ms ≈ 15.6 s` —
does not fit; that is the work item the implementation must close.

**CPU fallback / ablation cost** (same config at 1.2 ms/step): BR-on-CPU is
~15 s (debug-only); `PROFILE_SAMPLE` drops the `v×k` factor —
`30 × 1 × 8 × 8 = 1,920 steps × 1.2 ms ≈ 2.3 s`, so the CPU fallback uses
PROFILE_SAMPLE, not BR. A **hard wall-clock guard** (`time_budget_s`) degrades M,
then falls back to Option-A ICM (Q5) for unfinished leaves rather than
overrunning — the session-12 safeguard, preserved.

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
that decision, recompute fresh next decision. Do not cache across decisions.**

**The Brown/Sandholm 2018 approximation, stated explicitly:** the opponent
best-response at each leaf (Q3) is computed against hero's **blueprint strategy at
the root**, *not* against hero's iteration-k subgame strategy. Full *alternating*
best-response would recompute leaf values as hero's subgame strategy evolves each
CFR iteration — multiplying Z by W and destroying the per-decision cache. The
published depth-limited approximation freezes the reference point at the
blueprint. The loss vs full alternating BR is **small in practice
(Brown/Sandholm 2018), large in compute terms (it would multiply Z by W)** — the
same tradeoff Pluribus made.

This is precisely what preserves caching exactness: the leaf value is a function
of `(leaf state, the k fixed biased continuation strategies, the menu, hero's
fixed blueprint)` — **none of which the CFR loop mutates.** Sub-step 3 only
changes the root-region iteration-k strategy, which the leaf value (under the
approximation) does not depend on. So caching the leaf value across CFR
iterations is **exact, not an approximation of the approximation**, and Z stays a
one-time cost (Q4) rather than `Z × W`.

- **Invalidation within a decision:** none. Valid for the whole subgame solve;
  discarded when the decision ends.
- **Across decisions — rejected.** Board/stacks/line differ each decision; a
  cross-decision cache would need a key over a leaf-state equivalence class *and*
  the opponent menu, which Track C1 changes between decisions (Q7) — silent
  staleness. Not worth it for a one-time precompute.

Store the value on the node (`leaf_value`, Q9) during the precompute; CFR reads it
during backups.

---

## Q7 — Opponent menu parameterization

The semantics change with the move to best-response. The parameter is no longer
"a prior over which bias the opponent plays" but **a weighting on the opponent's
best-response menu** — which biases the opponent may choose from when maximizing,
and how they are tilted. Track C1 (Layer 4 within-match adaptation) can
**downweight a bias or remove it from the menu** based on observed behavior,
instead of nudging a sampling distribution.

Concretely, per (opponent seat, bias) the parameter carries a non-negative weight:

- weight `0` → that bias is **excluded from the opponent's argmax menu** (the
  opponent may not play that continuation);
- positive weight → bias is **in the menu**; relative magnitudes act as an
  additive tilt on the bias's value before the argmax (a soft prior belief that
  the opponent leans toward some continuation) and as the tie-tilt input (Q8).

The API signature is unchanged in shape — same `(k,)` or `(NUM_SEATS_6MAX, k)`
array, `None` → uniform full menu (all biases in, no tilt). Only the *meaning*
differs by mode (Q9):

- **BEST_RESPONSE:** menu membership + argmax tilt (above).
- **PROFILE_SAMPLE:** the old prior — the sampling distribution over biases.

```python
opponent_prior: np.ndarray | None = None
# BEST_RESPONSE : per-(seat,bias) menu weight; 0 excludes the bias, positive
#                 includes it and tilts the argmax. None -> uniform full menu.
# PROFILE_SAMPLE: per-(seat,bias) sampling distribution. None -> uniform.
```

Default uniform full menu = pure best-response over all k biases, no Layer-4
adaptation.

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

**Best-response ties.** When `argmax_b V_i(b)` has two or more biases within a
numerical epsilon, break deterministically by **lowest bias index** (reproducible,
seed-independent). A tie means the opponent is *indifferent among those
continuations* — the maximized quantity (the opponent's own value) is by
definition equal across them, so the choice is value-irrelevant *for the
opponent*. Caveat worth recording: equal *opponent* value does **not** imply equal
*hero* value across the tied biases, so lowest-index is not provably the most
conservative choice for hero. If the Q11 ablation shows sensitivity to the
tie-break, switch to a pessimistic rule (pick the tied bias that minimizes hero's
value). For v1 we use lowest-index and add a tie test (Q10 #10).

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
configs, starting_stacks, payouts, opponent menu/prior, n_samples, rng, budget,
**mode**) so the signature stays short — same pattern as `CFR6MaxContext`
(cfr6.py:82–124). The `mode` field selects the leaf semantics:

```python
class LeafEvalMode(Enum):
    BEST_RESPONSE  = "best_response"   # default, production / GPU target
    PROFILE_SAMPLE = "profile_sample"  # CPU fallback / ablation (Q3, Q11)
```

Both paths must be implementable; `BEST_RESPONSE` is the default and the
production target, `PROFILE_SAMPLE` is the cheaper non-robust comparison baseline
used by the Q11 ablation and CPU debugging.

`BlueprintProvider` is a tiny protocol the existing solver already
satisfies (it has `.encoder` and `.policy_nets.predict_advantages`, exercised at
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
4. **Prior = point-mass on blueprint ≡ pure-blueprint rollout.** In
   `PROFILE_SAMPLE` mode with `opponent_prior` a delta on index 0 (and in
   `BEST_RESPONSE` mode with a menu of only bias 0), the value must match a direct
   blueprint-only rollout (within stderr). Confirms the bias plumbing.
5. **Permutation invariance.** With symmetric stacks/cards across two opponent
   seats and a symmetric menu/prior, swapping which seat is which leaves the
   (symmetrized) value unchanged within stderr — the "tree-irrelevant permutation
   of biased strategies" check.
6. **ITM short-circuit equivalence.** In an `is_itm()` leaf (icm.py:229), Option-A
   and Option-B agree within stderr; assert the short-circuit doesn't move the value
   beyond MC noise.
7. **Budget guard.** With `time_budget_s` tiny, the call returns a (higher-variance)
   value, respects the budget within a small slack, and never raises.
8. **Failure degradation.** Inject a NaN-producing blueprint stub; assert the
   result degrades to Option-A / zero-vector with `leaf_eval_degraded=True` and a
   logged warning, not an exception.
9. **Maximization fires (BR ≥ uniform on the opponent's own value).** For each
   live opponent seat i, its own value entry under `BEST_RESPONSE` ≥ under
   `PROFILE_SAMPLE` with a uniform menu — algebraically `max_b V_i(b) ≥ mean_b V_i(b)`,
   the signature that the argmax is actually running. Equivalently, **hero's**
   value entry under `BEST_RESPONSE` ≤ under uniform `PROFILE_SAMPLE`: an opponent
   best-responding can only hurt hero relative to one drawn from a uniform prior.
   This is the test that pins the central design change in code.
10. **Tie-break determinism.** Mock blueprint values to force two biases to give an
    opponent equal value; assert the chosen bias is the lowest index and the result
    is identical across runs/seeds (Q8).

Tests 1–3 and 6 are deterministic or low-variance and gate sub-step 3; 4, 5, 9
validate the bias / menu / best-response semantics; 7, 8, 10 validate the safety
envelope and tie-break.

---

## Q11 — Ablation plan (acceptance gate for the BR design choice)

The best-response-vs-profile-sample decision is an empirical bet; it must be
measured, not assumed. The acceptance check is a three-way comparison against the
standard pool:

1. **pure blueprint** — `dcfr-overnight-3000`, no subgame solving (baseline);
2. **subgame + PROFILE_SAMPLE leaves** — the cheaper averaged form;
3. **subgame + BEST_RESPONSE leaves** — the production form.

**Staging (honest dependency).** The full pool comparison needs the SubgamePolicy
(sub-steps 3–5) to route through `scripts/eval_pool.py`, so it cannot run at
sub-step 2 in isolation. Two gates:

- **Sub-step-2 micro-ablation (runs now, leaf-level):** on a fixed battery of
  ~50 hand-built leaf states spanning streets / stack-depths / live-opponent
  counts, compute BR and PROFILE_SAMPLE(uniform) leaf 6-vectors and confirm the
  Q10 #9 ordering holds (BR ≥ uniform on opponent value, ≤ on hero) and quantify
  the per-leaf hero-value gap. This validates the mechanism before the pipeline
  exists.
- **Post-sub-step-5 pool ablation (acceptance gate before locking BR):** run all
  three challengers through `scripts/eval_pool.py` against the **same pool used
  for the `league-v2-600` baseline** (`evals/league-v2-600_vs_pool_5000.json`),
  at **5,000 hands/matchup** (the established sample size in that file). Report
  the ICM-equity-delta diff ± stderr and σ per matchup (the script's headline
  metric; the bb/100 framing is this delta scaled by the big blind).

**Success criterion:** subgame-BR ≥ subgame-PROFILE_SAMPLE ≥ blueprint, with the
**BR-vs-blueprint gap statistically significant (σ > 2)** on the pool. If BR does
*not* beat PROFILE_SAMPLE by more than noise, the `v×k` cost and the X→6 s budget
increase are not buying robustness, and we revert to PROFILE_SAMPLE (or revisit
α / k). This is the explicit go/no-go on the architectural change.

---

## Key decisions (summary)

- **Leaf value = 6-vector of ICM-equity deltas** (`tuple[float, ...]`,
  `NUM_SEATS_6MAX`), identical units to `icm_adjust_returns` at cfr6.py:278–285, so
  LEAF and TERMINAL back up identically. *(unchanged)*
- **k=4 continuation strategies already exist** in `biased_policy.py`
  (`standard_bias_configs`, blueprint/fold/call/raise-biased, α=3.0); the evaluator
  only supplies the blueprint forward pass and the menu. *(unchanged)*
- **Best-response, not prior-average.** Each live opponent independently
  best-responds among its k biases against hero's **blueprint** (Brown/Sandholm
  2018 single-pass approximation). Profile-sampling is demoted to a
  `PROFILE_SAMPLE` fallback / ablation mode.
- **Combinatorics = `v × k`** per leaf (independent BR), not `k^v` (joint) and not
  1 (average); `(v×k+1) × M` rollouts; v is typically 1–2 after excluding folded
  and all-in seats.
- **Budget X raised to 6 s** (GPU production): Y 0.5 + Z 3.0 + W 2.0 + 0.5
  headroom. **Measured** per-step state-prep ≈ 0.9 ms (parse-dominated,
  GPU-invariant) is the real floor — GPU speeds the network but not the parse, so
  the implementation must cache/lighten the parse and batch forwards to reach
  ~0.15 ms/step. CPU (BR ≈ 15 s) is debug-only; the CPU fallback uses
  PROFILE_SAMPLE (~2.3 s).
- **ICM via Option B** (rollout → `state.returns()` → `icm_adjust_returns`), with
  an Option-A short-circuit when `is_itm()` (no accuracy loss, pure speed).
  *(unchanged)*
- **Cache leaf values per decision, across CFR iterations — *exact***, because the
  leaf value is computed against the *fixed blueprint* (the Brown/Sandholm
  approximation), not the iteration-k strategy. Never cache across decisions
  (menu staleness under Track C1).
- **Opponent parameter is now a best-response menu** (membership + argmax tilt),
  not a play-prior; same array shape, `None` → uniform full menu; Layer-4 / C1
  drops or downweights biases. In `PROFILE_SAMPLE` mode it keeps the old
  sampling-prior meaning.
- **BR ties → lowest index** (deterministic). Opponent-indifferent; hero-value
  caveat recorded; tie test added (Q10 #10).
- **Failures degrade locally** (drop NaN sample → Option-A → zero-vector +
  `leaf_eval_degraded` flag), distinct from sub-step 5's whole-decision blueprint
  fallback. *(unchanged)*
- **Integration:** `evaluate_leaves(tree, ctx)` populates a new
  `SubgameNode.leaf_value` field once before the CFR loop; `evaluate_leaf(node,
  ...)` is the testable inner call; `LeafEvalContext` adds a `mode` field
  (`BEST_RESPONSE` default / `PROFILE_SAMPLE`), mirrors `CFR6MaxContext`.
- **Blueprint accessed via a small protocol** the existing solver already
  satisfies (`.encoder` + `.policy_nets.predict_advantages`), keeping
  `SubgamePolicy` conformable to `eval_pool.py`'s `Policy`/`CheckpointPolicy`
  contract. *(unchanged)*
- **Test gate before sub-step 3:** determinism, ICM-extreme vs hand-computed,
  conservation (sum≈0), blueprint-only ≡ pure rollout, permutation invariance,
  ITM short-circuit equivalence, budget guard, failure degradation, **BR ≥ uniform
  on opponent value (maximization fires)**, **tie-break determinism**.
- **Ablation gate (Q11):** BR vs PROFILE_SAMPLE vs blueprint on the
  `league-v2-600` pool at 5,000 hands/matchup; lock BR only if BR-vs-blueprint
  σ > 2 and BR ≥ PROFILE_SAMPLE.
