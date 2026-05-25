"""Stage 5-A tests for src.nlhe.subgame_policy.SubgamePolicy (B1c sub-step 5 scaffold).

Stage 5-A ships the class + the gate + instrumentation + starting_stacks
reconstruction. The SOLVE branch raises NotImplementedError (Stage 5-B), so the gate
tests verify the gate correctly ROUTES (skip -> blueprint fall-through; solve ->
counted + NotImplementedError). A mock blueprint drives the gate/conformance/
fall-through tests; the starting_stacks reconstruction test walks REAL production
states (no checkpoint needed) and asserts the invariant everywhere, including all-in.
"""
from __future__ import annotations

import inspect
import random
import unittest

import numpy as np


def _open_spiel_available() -> bool:
    try:
        import pyspiel  # noqa: F401
        return True
    except Exception:
        return False


_HAS_OPEN_SPIEL = _open_spiel_available()


# ---- mock blueprint (replaces the network only) ----

class _MockEncoder:
    def encode_from_parsed(self, parsed, rng=None):
        return np.zeros(8, dtype=np.float32)

    def reset_cache(self):
        pass


class _MockNets:
    def __init__(self, adv):
        self._adv = np.asarray(adv, dtype=np.float32)

    def predict_advantages(self, seat, features):
        return self._adv.copy()


class _MockSolver:
    def __init__(self, adv):
        self.encoder = _MockEncoder()
        self.policy_nets = _MockNets(adv)


def _make_policy(adv, **kw):
    """Construct a SubgamePolicy with a mock blueprint (patches the lazy _load_solver)."""
    from unittest import mock
    from src.nlhe.subgame_policy import SubgamePolicy
    with mock.patch("scripts.eval_6max_self_play._load_solver",
                    return_value=_MockSolver(adv)):
        return SubgamePolicy("sg", "fake.pt", None, None, **kw)


def _first_decision_state(starting_stack: int = 10000, seed: int = 42):
    import pyspiel
    from src.nlhe.game_strings import six_max_sng
    state = pyspiel.load_game(six_max_sng(starting_stack=starting_stack)).new_initial_state()
    r = random.Random(seed)
    while state.is_chance_node():
        actions, probs = zip(*state.chance_outcomes())
        state = state.child(int(r.choices(actions, weights=probs, k=1)[0]))
    return state


# ============================================================
# Conformance + gate routing (mock blueprint)
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestSubgamePolicyGate(unittest.TestCase):
    DECISIVE = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # all mass on CALL -> max-prob ~1.0
    MIXED = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]       # uniform over legal -> max-prob small

    def test_conformance_interface(self):
        from src.nlhe.subgame_policy import SubgamePolicy
        p = _make_policy(self.MIXED)
        self.assertIsInstance(p.name, str)
        params = list(inspect.signature(SubgamePolicy.select_action).parameters)
        self.assertEqual(params, ["self", "parsed", "state", "rng", "mode"])
        self.assertTrue(callable(p.select_action))
        # stats() shape (Stage 5-C added computed rates)
        s = p.stats()
        self.assertEqual(set(s), {"n_decisions_total", "n_gated_skip",
                                  "n_gated_solve", "n_degraded",
                                  "gate_skip_rate", "gate_solve_rate", "degraded_rate"})

    def _parsed(self, state):
        from src.nlhe.infoset6 import parse_state_6max
        return parse_state_6max(state)

    def test_gate_skip_on_decisive_blueprint(self):
        state = _first_decision_state()
        p = _make_policy(self.DECISIVE)
        chip = p.select_action(self._parsed(state), state, random.Random(0), mode="sample")
        self.assertIn(chip, set(state.legal_actions()))   # legal action via fall-through
        self.assertEqual(p.n_gated_skip, 1)
        self.assertEqual(p.n_gated_solve, 0)
        self.assertEqual(p.n_decisions_total, 1)

    def test_gate_solve_on_mixed_blueprint(self):
        # A mixed blueprint with >=3 legal actions gates to SOLVE. (The full solve
        # pipeline is exercised by TestSubgamePolicyPipeline with small params; here
        # we just assert the gate routing decision.)
        state = _first_decision_state()
        p = _make_policy(self.MIXED)
        g = p._evaluate_gate(self._parsed(state), state, random.Random(0))
        self.assertTrue(g["solve"])
        self.assertGreaterEqual(g["n_legal"], 3)
        self.assertLess(g["max_prob"], 0.95)

    def test_gate_skip_when_fewer_than_three_actions(self):
        # Monkeypatch the discretize to a 2-action map (the no-re-raise-room shape);
        # even a mixed blueprint must SKIP for <3 legal actions.
        from unittest import mock
        from src.nlhe.actions import DiscreteAction
        state = _first_decision_state()
        p = _make_policy(self.MIXED, min_legal_actions=3)
        two = {DiscreteAction.FOLD: 0, DiscreteAction.CALL: 1}
        with mock.patch("src.nlhe.subgame_policy._discretize_at_decision", return_value=two):
            chip = p.select_action(self._parsed(state), state, random.Random(0), mode="sample")
        self.assertEqual(p.n_gated_skip, 1)
        self.assertEqual(p.n_gated_solve, 0)

    def test_fall_through_returns_legal_action(self):
        state = _first_decision_state()
        p = _make_policy(self.DECISIVE)
        for md in ("sample", "argmax"):
            chip = p.select_action(self._parsed(state), state, random.Random(1), mode=md)
            self.assertIn(chip, set(state.legal_actions()))


# ============================================================
# starting_stacks reconstruction (Finding 1 guard) — real states, all scenarios
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestStartingStacksReconstruction(unittest.TestCase):
    def test_reconstruction_invariant_over_walk(self):
        import pyspiel
        from src.nlhe.subgame_policy import SubgamePolicy
        from src.nlhe.game_strings import TournamentStructure
        from src.nlhe.stack_sampler import sample_starting_state
        from src.nlhe.infoset6 import parse_state_6max, parse_state_repeated_6max

        structure = TournamentStructure.from_yaml(
            "configs/ignition_double_up_6max_turbo.yaml")
        recon = SubgamePolicy._reconstruct_starting_stacks
        master = random.Random(2026)
        checked = saw_preflop = saw_postflop = saw_allin = saw_multibet = 0

        for _ in range(150):
            sampled = sample_starting_state(structure, master, num_paid=3)
            stacks = list(sampled["stacks"])
            gs = structure.to_inner_game_string_for_state(
                blind_level=sampled["blind_level"], stacks=stacks,
                dealer_seat=sampled["dealer_seat"])
            state = pyspiel.load_game(gs).new_initial_state()
            rng = random.Random(master.randrange(2 ** 31))
            steps = 0
            while not state.is_terminal() and steps < 200:
                steps += 1
                if state.is_chance_node():
                    outs = state.chance_outcomes()
                    state.apply_action(int(rng.choices(
                        [o[0] for o in outs], weights=[o[1] for o in outs], k=1)[0]))
                    continue
                parsed = (parse_state_repeated_6max(state)
                          if hasattr(state, "dealer_seat") else parse_state_6max(state))
                rec = recon(parsed)
                for i in range(6):
                    self.assertLessEqual(
                        abs(int(rec[i]) - int(stacks[i])), 1,
                        f"seat {i}: reconstructed {rec[i]} != sampled {stacks[i]} "
                        f"(money={parsed['money']}, contribution={parsed['contribution']}, "
                        f"street={parsed.get('street_idx')})")
                checked += 1
                st = int(parsed.get("street_idx", 0))
                if st == 0:
                    saw_preflop += 1
                else:
                    saw_postflop += 1
                # all-in this hand: a seat that started with chips now has 0 behind
                if any(stacks[i] > 0 and int(parsed["money"][i]) == 0 for i in range(6)):
                    saw_allin += 1
                if max(int(c) for c in parsed["contribution"]) > min(
                        int(c) for c in parsed["contribution"] if c) if any(parsed["contribution"]) else False:
                    saw_multibet += 1
                state.apply_action(int(rng.choice(state.legal_actions())))

        # invariant held everywhere; confirm we actually exercised the scenarios
        self.assertGreater(checked, 200, "too few decision nodes exercised")
        self.assertGreater(saw_preflop, 0, "no preflop (after-blinds) states")
        self.assertGreater(saw_postflop, 0, "no mid-hand multi-street states")
        self.assertGreater(saw_allin, 0, "no all-in states exercised — reconstruction "
                                         "not covered for the all-in / partial-all-in scenario")


# ============================================================
# Stage 5-B: full pipeline (build -> eval -> solve -> extract)
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestSubgamePolicyPipeline(unittest.TestCase):
    MIXED = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]       # -> gate SOLVE
    DECISIVE = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # -> gate SKIP
    SMALL = dict(n_samples=2, max_action_depth=2, n_iterations=10)  # fast mock solves

    def _parsed(self, state):
        from src.nlhe.infoset6 import parse_state_6max
        return parse_state_6max(state)

    def test_pipeline_executes_returns_legal(self):
        state = _first_decision_state()
        p = _make_policy(self.MIXED, **self.SMALL)
        chip = p.select_action(self._parsed(state), state, random.Random(0), mode="sample")
        self.assertIn(chip, set(state.legal_actions()))
        self.assertEqual(p.n_gated_solve, 1)
        self.assertEqual(p.n_gated_skip, 0)
        self.assertEqual(p.n_degraded, 0)

    def test_solve_branch_taken_not_fallthrough(self):
        from unittest import mock
        state = _first_decision_state()
        p = _make_policy(self.MIXED, **self.SMALL)
        with mock.patch.object(p, "_blueprint_action",
                               wraps=p._blueprint_action) as spy:
            chip = p.select_action(self._parsed(state), state, random.Random(0), mode="argmax")
        self.assertIn(chip, set(state.legal_actions()))
        self.assertEqual(p.n_gated_solve, 1)
        self.assertEqual(p.n_degraded, 0)
        spy.assert_not_called()  # non-degraded solve returns via extract_action, not fall-through

    def test_gated_skip_matches_blueprint_exactly(self):
        from scripts.eval_6max_self_play import _sample_action_from_policy
        state = _first_decision_state()
        p = _make_policy(self.DECISIVE, **self.SMALL)
        parsed = self._parsed(state)
        chip_sg = p.select_action(parsed, state, random.Random(0), mode="argmax")
        chip_bp = _sample_action_from_policy(p.solver, parsed, state, random.Random(0),
                                             mode="argmax")
        self.assertEqual(chip_sg, chip_bp)
        self.assertEqual(p.n_gated_skip, 1)
        self.assertEqual(p.n_gated_solve, 0)

    def test_end_to_end_single_hand_smoke(self):
        from scripts.eval_pool import play_one_hand_two_policies, UniformRandomPolicy
        from src.nlhe.game_strings import TournamentStructure
        structure = TournamentStructure.from_yaml(
            "configs/ignition_double_up_6max_turbo.yaml")
        challenger = _make_policy(self.MIXED, **self.SMALL)
        challenger.name = "subgame"
        opponent = UniformRandomPolicy("rand")
        result = play_one_hand_two_policies(
            challenger, opponent, structure, random.Random(3), mode="sample")
        self.assertFalse(result["exceeded_cap"], "hand did not reach terminal")
        self.assertEqual(len(result["seat_to_equity_delta"]), 6)


# ============================================================
# Stage 5-C: degraded diagnostics + reproducibility
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestSubgamePolicyStageC(unittest.TestCase):
    MIXED = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    DECISIVE = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    SMALL = dict(n_samples=2, max_action_depth=2, n_iterations=10)

    def _parsed(self, state):
        from src.nlhe.infoset6 import parse_state_6max
        return parse_state_6max(state)

    def test_reproducible_same_seed(self):
        state = _first_decision_state()
        parsed = self._parsed(state)
        p = _make_policy(self.MIXED, **self.SMALL)
        a1 = p.select_action(parsed, state, random.Random(5), mode="sample")
        a2 = p.select_action(parsed, state, random.Random(5), mode="sample")
        self.assertEqual(a1, a2)  # full pipeline deterministic given the seed

    def test_gate_decision_seed_independent(self):
        # The gate (encode -> predict -> RM+ max_prob) has no rng-dependent sampling,
        # so the SOLVE/SKIP decision is identical across seeds.
        state = _first_decision_state()
        parsed = self._parsed(state)
        p = _make_policy(self.MIXED, **self.SMALL)
        decisions = {p._evaluate_gate(parsed, state, random.Random(s))["solve"]
                     for s in (1, 2, 3, 99)}
        self.assertEqual(len(decisions), 1)  # all seeds agree

    def test_action_distribution_matches_solved_policy(self):
        # Isolate the SAMPLING layer: mock build/eval/solve so each select_action just
        # samples a FIXED solved root policy (0.7 FOLD / 0.3 CALL); confirm the action
        # frequencies match within ~3% (the per-decision pipeline cost is irrelevant
        # here and is covered by the timing benchmark).
        from unittest import mock
        from src.nlhe.subgame_solver import SubgameSolveResult
        from src.nlhe.subgame_leaf import LeafBatchResult
        state = _first_decision_state()
        parsed = self._parsed(state)
        p = _make_policy(self.MIXED, **self.SMALL)
        pol = np.zeros(7, np.float32); pol[0] = 0.7; pol[1] = 0.3
        mask = np.zeros(7, np.float32); mask[0] = 1.0; mask[1] = 1.0
        fixed = SubgameSolveResult(root_policy=pol, root_blueprint=pol, legal_mask=mask,
                                   hero_seat=parsed["current_player"], n_iterations=10,
                                   n_decision_nodes_cached=1, degraded=False)
        batch = LeafBatchResult(n_leaves=0, n_evaluated=0, partial_eval_degraded=False)
        rng = random.Random(2024); N = 2000; fold = 0
        with mock.patch("src.nlhe.subgame_policy.build_subgame_tree", return_value=None), \
             mock.patch("src.nlhe.subgame_policy.evaluate_leaves", return_value=batch), \
             mock.patch("src.nlhe.subgame_policy.solve_subgame", return_value=fixed):
            for _ in range(N):
                if p.select_action(parsed, state, rng, mode="sample") == 0:  # FOLD chip
                    fold += 1
        self.assertAlmostEqual(fold / N, 0.7, delta=0.03)

    def test_degraded_falls_through_with_diagnostics(self):
        from unittest import mock
        from src.nlhe.subgame_solver import SubgameSolveResult
        state = _first_decision_state()
        parsed = self._parsed(state)
        p = _make_policy(self.MIXED, **self.SMALL)
        degraded = SubgameSolveResult(
            root_policy=np.zeros(7, np.float32), root_blueprint=np.zeros(7, np.float32),
            legal_mask=np.zeros(7, np.float32), hero_seat=parsed["current_player"],
            n_iterations=10, n_decision_nodes_cached=1, degraded=True)
        with mock.patch("src.nlhe.subgame_policy.solve_subgame", return_value=degraded):
            with self.assertLogs("subgame_policy", level="WARNING") as cm:
                chip = p.select_action(parsed, state, random.Random(0), mode="sample")
        self.assertIn(chip, set(state.legal_actions()))     # legal blueprint fall-through
        self.assertEqual(p.n_degraded, 1)
        self.assertEqual(p.n_gated_solve, 1)
        self.assertTrue(any("degraded -> blueprint fall-through" in m for m in cm.output))

    def test_stats_rates(self):
        p = _make_policy(self.MIXED, **self.SMALL)
        p.n_decisions_total = 100
        p.n_gated_skip = 73
        p.n_gated_solve = 27
        p.n_degraded = 3
        s = p.stats()
        self.assertAlmostEqual(s["gate_skip_rate"], 0.73)
        self.assertAlmostEqual(s["gate_solve_rate"], 0.27)
        self.assertAlmostEqual(s["degraded_rate"], 3 / 27)


if __name__ == "__main__":
    unittest.main()
