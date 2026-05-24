# Session 13 — 2026-05-24

Focus: B1c (depth-limited subgame solving). Fixed the tree builder, verified its
invariants, and designed + twice-revised the leaf evaluator (sub-step 2) to a
review-approved state. No leaf-evaluator implementation landed — the session
ends at the implementation gate.

## What was done

- **Subgame tree builder fix** (`2be87df`). `_build_node` iterated raw
  `state.legal_actions()` (~9,800 chip ints under `bettingAbstraction=fullgame`),
  exploding a depth-1 subgame to 600+ leaves. Replaced with
  `discretize_legal_actions` via the exact `cfr6.traverse_6max` path
  (parse → `_build_view_6max` → discretize); decision nodes now enumerate the
  same ≤7 `DiscreteAction` set the walker does (depth-1 root: 6 children).
  `action_from_parent` / `action_at_child` store the DiscreteAction value, not
  the chip int. `_compute_descendants` converted to iterative post-order. Test
  fixtures moved off a hand-written ACPC gamedef onto
  `pyspiel.load_game(six_max_sng(starting_stack=10000))` and gained 5 invariant
  tests (child-set == walker-set, recursively; restricted facing-all-in count;
  fold-ends-hand → TERMINAL). 20/20 tests pass.

- **ALLIN→FOLD chip-0 alias documented** (`b2dded5`). Diagnosed that, facing an
  all-in with no re-raise room, `discretize_legal_actions` returns
  `{FOLD:0, CALL:1, ALLIN:0}` — `ALLIN` aliases to chip 0 (FOLD) because
  `view.max_bet == 0`. It is a label-only alias (no real all-in action exists in
  `legal_actions`); `apply_action(0)` folds. Documented in
  `_legal_discrete_bet_sizes`; the correct all-in-equivalent there is CALL (1).
  No behavior change.

- **Internal-node descendants invariant verified** (no code change).
  `n.n_descendants == len(n.children) + sum(c.n_descendants for c in children)`
  confirmed on every internal node of a depth-3 post-flop tree (34 decision
  nodes) and a near-flop tree with 21 chance internal nodes (2,332 internal nodes
  total).

- **Leaf evaluator design proposal** (`89ec957`), answering 10 review questions
  grounded in `biased_policy.py`, `icm_returns.py`, `icm.py`, `cfr6.py`,
  `eval_pool.py`.

- **Design revised to best-response form** (`4029757`). Switched the leaf value
  from prior-weighted profile-sampling to opponent best-response (Brown/Sandholm
  2018 single-pass approximation: BR vs hero's blueprint at root). Combinatorics
  v×k (not k^v, not averaged); budget redone on GPU with X raised 4→6 s; menu
  semantics; tie-break; `mode` field; maximization-fires test; ablation gate.

- **Parse optimization scoped + Q11 split** (`98edf14`). Q4.5 makes incremental
  parsing (Path B) explicit sub-step 2 scope; Q11 split into Level 1 (leaf-only,
  gates sub-step 2), Level 2 (decision-level via a stub one-iteration root regret
  update — itself a sub-step 2 deliverable), Level 3 (full pool, post-sub-step-5).

## What was decided

Two entries appended to `docs/DECISIONS.md`:

1. **Session-summary convention** — per-session files in `docs/sessions/`.
2. **Leaf evaluator architecture = BEST_RESPONSE** (Brown/Sandholm 2018
   blueprint-reference approximation), not profile-sampling under a prior.

## What was learned / measured

- Raw `legal_actions()` at a first-decision state = 9,803 chip ints; the
  discretized set = 6. This is the bug the old ACPC-gamedef fixture hid (it used
  OpenSpiel's default smaller abstraction, and no test asserted child count).
- **Per-step state-prep ≈ 0.9 ms/step on CPU** (`parse_state_6max` regex), vs
  `state.child()` ≈ 0.1 ms and `information_state_string` ≈ 0.001 ms. It is
  CPU-bound and GPU-invariant, so it — not the network forward — is the binding
  constraint on the BR-mode decision budget. This drove Q4.5 (Path B incremental
  parsing) into sub-step 2 scope.

## State at close

- **Done:** subgame tree builder (sub-step 1.5) correct + tested; leaf evaluator
  design approved (after two revisions + the Q4.5 / Q11 additions).
- **Open / next:** implement `src/nlhe/subgame_leaf.py` — BEST_RESPONSE +
  PROFILE_SAMPLE, Path B incremental parsing, and the stub one-iteration root
  regret update for the Q11 Level-2 ablation. See `NEXT_SESSION.md`.
