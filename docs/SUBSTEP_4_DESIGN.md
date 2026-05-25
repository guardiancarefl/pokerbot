# Sub-step 4 — Policy Extraction — Design Proposal (B1c)

**Status:** PROPOSAL (review gate). No implementation lands with this doc. Line
numbers cite files as read 2026-05-26 (session 20). Predecessor: sub-step 3 (the
subgame CFR solver, `src/nlhe/subgame_solver.py`, CLOSED session 19), which produces
`SubgameSolveResult.root_policy` — hero's refined, masked, length-7 `DiscreteAction`
distribution at the root.

**What sub-step 4 is:** the thin, pure-ish step that turns `root_policy` into a
concrete OpenSpiel chip action ready for `state.apply_action(...)` — select a
`DiscreteAction` (argmax or weighted-sample), translate it to a chip amount via the
same discretize map the solver used, and apply the `ALLIN→CALL` alias at the
translation boundary. It is the subgame-solver analog of the blueprint's eval-time
`_sample_action_from_policy` (eval_6max_self_play.py:120-167), substituting the
refined policy for the blueprint's RM+-from-advantages.

---

## A. Interface

One function, added to `src/nlhe/subgame_solver.py` (it operates on
`SubgameSolveResult` and reuses the module's `_parse`):

```python
def extract_action(result: SubgameSolveResult, state, rng: random.Random,
                   mode: str = "sample") -> int:
    """Map result.root_policy to an OpenSpiel chip action for state.apply_action().
    mode: "sample" (weighted by root_policy) | "argmax" (greedy). Returns the chip int."""
```

`state` is the **root** OpenSpiel state the subgame was built from (hero to act).
The signature deliberately mirrors `eval_pool.Policy.select_action(parsed, state,
rng, mode) -> int` (eval_pool.py:62-64), minus `parsed` (re-derived internally,
0.008 ms) and plus `result`. Sub-step 5's `SubgamePolicy.select_action` becomes a
one-liner: build tree → `evaluate_leaves` → `solve_subgame` → `extract_action(result,
state, rng, mode)`.

**Output contract for sub-step 5:** a single legal chip int. That is all sub-step 5
needs to play through to the next state.

---

## B. Algorithm

```
1. discrete_to_chip = subgame._discretize_at_decision(state)   # the SAME map the
                                                               # tree/solver used
2. p = result.root_policy                                      # masked 7-vector
3. select a DiscreteAction index:
     mode == "argmax":  idx = argmax over legal of p  (mask via result.legal_mask)
     mode == "sample":  idx = rng.choices(range(7), weights=p)   # p is 0 on illegal
     (defensive) if p has no legal mass: uniform over discrete_to_chip.keys()
4. da = DiscreteAction(idx);  chip = discrete_to_chip[da]
5. ALLIN→CALL alias: if da == ALLIN and chip == 0:   # no-re-raise-room shove
        chip = discrete_to_chip[DiscreteAction.CALL]  # == 1
6. return int(chip)
```

Step 1 reuses `subgame._discretize_at_decision` (subgame.py:401-425 — itself
`_build_view_6max` + `discretize_legal_actions`), so the chip map's keys are exactly
the actions `root_policy` was masked over (the tree built its children from the same
call). No re-derivation of bet sizes, no risk of the action set drifting from the
mask. Step 3 mirrors the blueprint pattern (eval_6max_self_play.py:145-162): argmax
over the masked vector, else `rng.choices` weighted by the (already-masked)
distribution, else uniform fallback. Step 5 is the only sub-step-4-specific logic
(Decision 4.3).

---

## C. Pre-committed decisions

**Decision 4.1 — Default mode = `"sample"` (weighted), `"argmax"` available.** The
blueprint at eval time defaults to `"sample"`: `_sample_action_from_solver`'s `mode`
defaults to `"sample"` (eval_pool.py:74), `eval_pool` CLI defaults `--mode sample`
(eval_pool.py:255). Matching it is load-bearing — sub-step 6 compares subgame-BR vs
blueprint on the **same** pool, so both must select actions the same way or the
comparison conflates the architectural delta with a selection-mode delta.
Weighted-sample also preserves the mixing the refined policy encodes (argmax discards
it). Argmax stays available for greedy/diagnostic runs.

**Decision 4.2 — Reuse `subgame._discretize_at_decision` for `DiscreteAction → chip`;
do not reimplement.** It is the exact map the tree builder and solver used, so
selection and translation share one source of truth. (`actions.policy_to_game_action`
exists but translates one action against a `view` and would re-derive the view — using
the tree's own map is both cheaper and guaranteed-consistent.)

**Decision 4.3 — The `ALLIN→CALL` alias fires at the `DiscreteAction → chip`
translation step, inside `extract_action` (step 5).** This is "wherever
DiscreteAction → chip happens." Detection (per `b2dded5`): the chosen action is
`ALLIN` and its mapped chip is `0`. In the no-re-raise-room shove the map is
`{FOLD:0, CALL:1, ALLIN:0}` (since `max_bet == 0`), so an `ALLIN` resolving to chip 0
is unambiguously the alias case (chip 0 is FOLD) — remap to `CALL` (chip 1, always
legal when facing a bet). Putting it here keeps sub-step 5 (and any other caller)
from re-implementing the alias.

**Decision 4.4 — RNG is an explicit parameter** (`rng: random.Random`), threaded by
the caller — matching the `Policy.select_action(parsed, state, rng, mode)` protocol
sub-step 5 conforms to. Not `numpy` global state, and not `SubgameSolveContext.rng`
(that seeds the solve; extraction sampling is a separate, caller-controlled draw — the
same `rng` `eval_pool` already threads through the rollout).

**Decision 4.5 — On `result.degraded`, `extract_action` extracts normally and does
NOT itself fall back to blueprint; the fallback decision belongs to sub-step 5.**
Rationale: `root_policy` is **always a valid distribution** (RM+ normalizes over legal
actions; the K=0 path returns the blueprint policy itself), so extraction always
yields a legal action — there is nothing to raise on. And sub-step 4 has **no
blueprint reference** to fall back to (it is a pure function of `result` + `state`);
the blueprint lives in the solver/wrapper. So sub-step 5 — which owns the solver and
the blueprint — reads `result.degraded` and decides whether to play the extracted
action or the raw blueprint. `extract_action` neither raises nor returns `None`.
*(This refines the prompt's fall-back/raise/None menu: the honest boundary is "neither
in sub-step 4" — surfaced in Part 4.)*

---

## D. Test plan (production game where a state is needed)

1. **Argmax on a known distribution.** `root_policy` with a clear max on a legal
   action ⇒ `mode="argmax"` returns that action's chip. Assert exact.
2. **Weighted-sample converges to the distribution.** Fixed `root_policy` (e.g.
   60/40 over two legal actions), N=20k samples with a seeded `rng` ⇒ empirical
   frequencies within ~2% of `root_policy`.
3. **Masked elements never selected.** `root_policy` with 0 on illegal actions ⇒
   over many samples and under argmax, no illegal `DiscreteAction`'s chip is ever
   returned (chip ∈ `discrete_to_chip.values()` for legal keys).
4. **`ALLIN→CALL` alias fires.** A hand-built / sampled state facing an all-in with
   no re-raise room (`discrete_to_chip == {FOLD:0, CALL:1, ALLIN:0}`); force
   selection of `ALLIN` ⇒ `extract_action` returns `1` (CALL), not `0` (FOLD). And
   the negative: a normal state where `ALLIN` maps to `max_bet > 0` ⇒ returns
   `max_bet` unchanged.
5. **Degraded handling.** `result.degraded == True` with a valid `root_policy` ⇒
   `extract_action` still returns a legal chip action (does not raise / return None);
   pairs with a sub-step-5 test (later) that the wrapper reads `degraded`.
6. **Reproducibility.** Same `result` + same `rng` seed + `mode="sample"` ⇒
   identical chip action across two calls.

Tests 1, 3, 4, 6 are deterministic; 2 is statistical (seeded). All states come from
the production `six_max_sng` game (the project test convention), reusing the
`_first_decision_state` / tree helpers already in `tests/test_subgame_solver.py`.

---

## E. Staged implementation

**Single stage (Stage 4-A).** The scope is ~30–40 lines + tests; it does not split
naturally. One commit: `extract_action` + the six tests above, run against the full
suite for 0 tracked regressions. (The `ALLIN→CALL` alias is a 2-line branch inside the
same function, not a separable stage.)

---

## F. Out of scope

- **The `eval_pool.Policy` / `SubgamePolicy` wrapper** (chaining
  build→evaluate_leaves→solve→extract, conforming to `select_action`) — **sub-step 5**.
- **Computing the solver result** (`solve_subgame`) — **sub-step 3** (done).
- **Multi-decision sequencing / re-solving each decision in a hand** — **sub-step 5/6**.
- **Strength measurement** (BR vs PROFILE vs blueprint on the pool) — **sub-step 6**.
- **`degraded`-driven blueprint fallback** — **sub-step 5** (Decision 4.5).
- **Opponent off-tree sizing translation** (`actions.game_to_policy_action`,
  pseudo-harmonic) — that is an *input*-side concern at the wrapper/eval boundary, not
  hero's action *output*.

---

## Part 4 — findings surfaced

1. **Decision 4.5 deviates from the prompt's literal menu** (fall-back / raise /
   None): the correct boundary is *none of those in sub-step 4*. `root_policy` is
   always a valid distribution and sub-step 4 has no blueprint reference, so it
   extracts normally and defers the `degraded` fallback to sub-step 5 (which owns the
   blueprint). Flagged for sign-off.
2. **No other surprises.** The blueprint's eval-time selection matches the design
   assumption (default `sample`, argmax available, explicit `rng`). The `ALLIN→CALL`
   alias is exactly as `b2dded5` documents (chip-0 collision, remap to CALL) — a clean
   2-line check, not more complex. The `DiscreteAction → chip` translation has one
   edge case (the alias); `policy_to_game_action`'s "None for sub-min-bet" case cannot
   arise here because the tree only created children for legal discrete actions, so
   `root_policy` carries no mass there. Sub-step 5's `CheckpointPolicy` interface needs
   nothing from sub-step 4 beyond the returned chip int.
