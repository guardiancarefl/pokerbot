"""SubgamePolicy — an `eval_pool.Policy` wrapper around the subgame solver (B1c sub-step 5).

A drop-in challenger for `scripts/eval_pool.py`: conforms to the `Policy` protocol
(`name` + `select_action(parsed, state, rng, mode) -> int`) so the existing pool
harness uses it with no changes. On each hero decision it would chain
build_subgame_tree → evaluate_leaves → solve_subgame → extract_action, behind a GATE
(Decision 5.1) that falls through to the blueprint when refinement can't matter. Full
design + the four pre-committed decisions: `docs/SUBSTEP_5_DESIGN.md`.

Two design findings carried in the code:
  - starting_stacks is NOT in the Policy interface, but the solver/leaf-eval need the
    hand-start per-seat stacks for ICM. Reconstructed as money + contribution from the
    parsed state (`_reconstruct_starting_stacks`; Finding 1). Test-verified.
  - SUB-STEP-6 PREREQUISITE (Finding 2): `eval_pool.py` is sequential; the sub-step-6
    wall-clock budget assumes hand-level multiprocessing (independent hands, each with
    its own SubgamePolicy). Not designed here — noted so it is planned.

STATUS: STAGE 5-C — sub-step 5 CLOSED. `select_action` evaluates the gate; on SKIP it
falls through to the blueprint (`_sample_action_from_policy`); on SOLVE it runs the
full pipeline (`_solve_action`: build_subgame_tree → evaluate_leaves → solve_subgame →
extract_action), falling through to the blueprint on a degraded result with a WARNING
+ `n_degraded` increment (no temporal back-off). `stats()` exposes the four counters
plus gate_skip_rate / gate_solve_rate / degraded_rate. Empirical f≈0.27, per-solve
~6.7s blended (`scripts/measure_gate_rate.py`); see `docs/SUBSTEP_5_DESIGN.md`.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from src.nlhe.actions import DiscreteAction
from src.nlhe.biased_policy import BiasedBlueprint
from src.nlhe.icm import sng_payouts_6max_double_up
from src.nlhe.solver import _strategy_from_advantages
from src.nlhe.subgame import _discretize_at_decision, build_subgame_tree, iter_leaf_nodes
from src.nlhe.subgame_leaf import LeafEvalMode, LeafEvalContext, evaluate_leaves
from src.nlhe.subgame_solver import (
    SubgameSolveContext, solve_subgame, extract_action,
)

log = logging.getLogger("subgame_policy")

_N_ACTIONS = len(DiscreteAction)  # 7
_NUM_SEATS = 6


class SubgamePolicy:
    """Subgame-solving challenger conforming to `eval_pool.Policy`."""

    def __init__(self, name: str, ckpt_path: str, abstraction, structure, *,
                 leaf_mode: LeafEvalMode = LeafEvalMode.BEST_RESPONSE,
                 n_samples: int = 8, n_iterations: int = 1000,
                 max_action_depth: int = 3, chance_samples_per_node: int = 8,
                 min_legal_actions: int = 3, max_blueprint_prob: float = 0.95,
                 payouts: Optional[Sequence[float]] = None, num_paid: int = 3):
        # Lazy import: src/ must not depend on scripts/ at module load (the project's
        # lazy-import pattern, mirroring subgame._discretize_at_decision -> cfr6).
        from scripts.eval_6max_self_play import _load_solver
        self.name = name
        self.ckpt_path = ckpt_path
        self.solver = _load_solver(ckpt_path, abstraction, structure)  # the BlueprintProvider
        self.biased = BiasedBlueprint()  # k=4 configs (Stage 5-B leaf eval)
        # config (Decision 5.4)
        self.leaf_mode = leaf_mode
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.max_action_depth = max_action_depth
        self.chance_samples_per_node = chance_samples_per_node
        self.min_legal_actions = min_legal_actions
        self.max_blueprint_prob = max_blueprint_prob
        self.payouts = (list(payouts) if payouts is not None
                        else list(sng_payouts_6max_double_up()))
        self.num_paid = num_paid
        # diagnostics (Decision 5.2)
        self.n_decisions_total = 0
        self.n_gated_skip = 0
        self.n_gated_solve = 0
        self.n_degraded = 0

    # ---- gate (Decision 5.1) ----
    def _evaluate_gate(self, parsed, state, rng) -> dict:
        """Gate decision + diagnostics for one hero decision (no solve).

        solve iff (>= min_legal_actions legal discrete actions) AND
        (blueprint max action prob < max_blueprint_prob). Returns dict with
        solve / n_legal / max_prob / street_idx (the last for per-street analysis)."""
        cp = parsed["current_player"]
        discrete_to_chip = _discretize_at_decision(state)
        n_legal = len(discrete_to_chip)
        street = int(parsed.get("street_idx", 0))
        if n_legal == 0:
            return {"solve": False, "n_legal": 0, "max_prob": 1.0, "street_idx": street}
        legal_mask = np.zeros(_N_ACTIONS, dtype=np.float32)
        for da in discrete_to_chip:
            legal_mask[int(da)] = 1.0
        feat = np.asarray(self.solver.encoder.encode_from_parsed(parsed, rng=rng),
                          dtype=np.float32)
        # TWO-SIGNAL design (Step E): the GATE stays on the adv-net RM+ policy
        # regardless of checkpoint schema, preserving sub-step-6 gate calibration
        # (f≈0.27). The PLAYED action goes through inference_policy, which uses
        # the v2 strategy net when available. Gate = WHEN to solve; inference =
        # WHAT to play. Do NOT route this read through inference_policy.
        adv = np.asarray(self.solver.policy_nets.predict_advantages(cp, feat),
                         dtype=np.float32)
        probs = _strategy_from_advantages(adv, legal_mask)  # RM+ masked blueprint policy
        max_prob = float(probs.max())
        solve = (n_legal >= self.min_legal_actions) and (max_prob < self.max_blueprint_prob)
        return {"solve": solve, "n_legal": n_legal, "max_prob": max_prob,
                "street_idx": street}

    def _blueprint_action(self, parsed, state, rng, mode) -> int:
        """Fall-through: the IDENTICAL blueprint selection opponents use
        (eval_pool.py:75), so a skipped/degraded decision plays exactly as the pure
        blueprint would."""
        from scripts.eval_6max_self_play import _sample_action_from_policy  # lazy
        return int(_sample_action_from_policy(self.solver, parsed, state, rng, mode=mode))

    @staticmethod
    def _reconstruct_starting_stacks(parsed) -> Sequence[int]:
        """Hand-start per-seat stacks = chips behind (money) + chips committed this
        hand (contribution). The Policy interface does not carry starting_stacks, but
        the solver/leaf-eval need them for ICM (Finding 1). Chip conservation makes
        this exact at every point in the hand (behind + committed = start), including
        folded and all-in seats. Operates on the parsed dict (already length-6,
        original-seat-indexed by both parse_state_6max and parse_state_repeated_6max —
        the latter remaps its contracted contributions), so it is parse-agnostic.
        Test-verified against sample_starting_state across streets / all-in states."""
        money = parsed["money"]
        contribution = parsed["contribution"]
        return [int(money[i]) + int(contribution[i]) for i in range(_NUM_SEATS)]

    # ---- solve branch (Stage 5-B pipeline + Stage 5-C degraded diagnostics) ----
    def _solve_action(self, parsed, state, rng, mode, gate) -> int:
        """The full subgame-solve pipeline for a gated-SOLVE decision:
        build_subgame_tree -> evaluate_leaves -> solve_subgame -> extract_action.

        On a degraded result (a leaf lacked a value, or the leaf-eval budget was cut
        short) fall through to the blueprint (Decision 5.2), increment `n_degraded`,
        and log the decision context at WARNING. NO temporal back-off: decisions are
        independent, so the next decision solves normally — backing off would only
        forfeit refinement on subsequent (unrelated) decisions for no benefit."""
        cp = parsed["current_player"]
        starting_stacks = self._reconstruct_starting_stacks(parsed)
        tree = build_subgame_tree(
            state, max_action_depth=self.max_action_depth,
            chance_samples_per_node=self.chance_samples_per_node, rng=rng)
        batch = evaluate_leaves(tree, LeafEvalContext(
            blueprint=self.solver, biased_blueprint=self.biased,
            starting_stacks=starting_stacks, payouts=self.payouts, hero_seat=cp,
            mode=self.leaf_mode, n_samples=self.n_samples, rng=rng,
            num_paid=self.num_paid))
        result = solve_subgame(tree, SubgameSolveContext(
            blueprint=self.solver, starting_stacks=starting_stacks,
            payouts=self.payouts, hero_seat=cp, n_iterations=self.n_iterations,
            rng=rng, num_paid=self.num_paid))
        if result.degraded or batch.partial_eval_degraded:
            self.n_degraded += 1
            n_chance = sum(1 for lf in iter_leaf_nodes(tree)
                           if lf.state is not None and lf.state.is_chance_node())
            reason = ("solve result.degraded (a leaf lacked a value)"
                      if result.degraded
                      else "evaluate_leaves partial_eval_degraded (budget cut short)")
            log.warning(
                "SubgamePolicy degraded -> blueprint fall-through: street=%s n_legal=%s "
                "blueprint_max_prob=%.3f leaves=%d chance_leaves=%d reason=%s",
                gate["street_idx"], gate["n_legal"], gate["max_prob"],
                tree.n_leaf_nodes, n_chance, reason)
            return self._blueprint_action(parsed, state, rng, mode)
        return extract_action(result, state, rng, mode)

    # ---- Policy contract ----
    def select_action(self, parsed, state, rng, mode: str = "sample") -> int:
        self.n_decisions_total += 1
        gate = self._evaluate_gate(parsed, state, rng)
        if not gate["solve"]:
            self.n_gated_skip += 1
            return self._blueprint_action(parsed, state, rng, mode)
        self.n_gated_solve += 1
        return self._solve_action(parsed, state, rng, mode, gate)

    def stats(self) -> dict:
        """Diagnostic counters + computed rates (Stage 5-C). degraded_rate is over
        SOLVED decisions (the denominator that can degrade), not all decisions."""
        n = self.n_decisions_total
        solve = self.n_gated_solve
        return {
            "n_decisions_total": n,
            "n_gated_skip": self.n_gated_skip,
            "n_gated_solve": solve,
            "n_degraded": self.n_degraded,
            "gate_skip_rate": (self.n_gated_skip / n) if n else 0.0,
            "gate_solve_rate": (solve / n) if n else 0.0,
            "degraded_rate": (self.n_degraded / max(1, solve)),
        }
