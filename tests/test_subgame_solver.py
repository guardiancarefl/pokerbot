"""Stage 3-A tests for src.nlhe.subgame_solver (B1c sub-step 3 scaffold).

Stage 3-A ships the solver module, warm-up caching, and the K=0 blueprint-
passthrough path. These tests pin: the dataclasses construct/validate, K=0 returns
the blueprint's masked root policy BIT-IDENTICALLY, the result is reproducible and a
valid distribution, and the K>=1 boundary is an explicit NotImplementedError.

The K=0 / reproducibility / sanity tests build a REAL production decision state via
six_max_sng (no checkpoint needed) and drive a MOCK blueprint that replaces only the
network — the game, the tree, and the masks are real.
"""
from __future__ import annotations

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


# ============================================================
# Mock blueprint (replaces only the network; game/tree/masks are real)
# ============================================================

class _MockEncoder:
    """Encoder stub: returns a fixed feature vector (the mock net ignores it) and
    a no-op cache reset. Shape is irrelevant — predict_advantages does not read it."""

    def encode_from_parsed(self, parsed, rng=None):
        return np.zeros(8, dtype=np.float32)

    def reset_cache(self):
        pass


class _MockNets:
    """Advantage net stub: deterministic 7-vector per seat (some negative, so RM+
    zeros them and the resulting distribution is non-uniform — a meaningful sanity
    check). Identical inputs → identical outputs, so the solve is reproducible."""

    _BASE = np.array([0.5, -0.3, 0.2, 0.1, -0.1, 0.4, 0.05], dtype=np.float32)

    def predict_advantages(self, seat, features):
        return (self._BASE + np.float32(0.01) * np.float32(seat)).astype(np.float32)


class _MockBlueprint:
    def __init__(self):
        self.encoder = _MockEncoder()
        self.policy_nets = _MockNets()


def _first_decision_state(starting_stack: int = 10000, seed: int = 42):
    """Real six_max_sng decision state, past hole-card chance (mirrors
    tests/test_subgame.py)."""
    import pyspiel
    from src.nlhe.game_strings import six_max_sng
    state = pyspiel.load_game(six_max_sng(starting_stack=starting_stack)).new_initial_state()
    r = random.Random(seed)
    while state.is_chance_node():
        actions, probs = zip(*state.chance_outcomes())
        state = state.child(int(r.choices(actions, weights=probs, k=1)[0]))
    return state


def _make_ctx(tree, n_iterations=0, rng=None, hero_seat=None):
    from src.nlhe.icm import sng_payouts_6max_double_up
    from src.nlhe.subgame_solver import SubgameSolveContext
    hero = tree.root.current_player if hero_seat is None else hero_seat
    return SubgameSolveContext(
        blueprint=_MockBlueprint(),
        starting_stacks=[10000] * 6,
        payouts=sng_payouts_6max_double_up(),
        hero_seat=hero,
        n_iterations=n_iterations,
        rng=rng,
    )


# ============================================================
# Sandbox-runnable: dataclass construction / validation
# ============================================================

class TestContextValidation(unittest.TestCase):
    def _ctx(self, **over):
        from src.nlhe.icm import sng_payouts_6max_double_up
        from src.nlhe.subgame_solver import SubgameSolveContext
        kw = dict(blueprint=object(), starting_stacks=[10000] * 6,
                  payouts=sng_payouts_6max_double_up(), hero_seat=0)
        kw.update(over)
        return SubgameSolveContext(**kw)

    def test_defaults(self):
        ctx = self._ctx()
        self.assertEqual(ctx.n_iterations, 1000)
        self.assertEqual(ctx.num_paid, 3)
        self.assertEqual(ctx.average_weighting, "linear")
        self.assertIsNone(ctx.rng)

    def test_rejects_bad_stack_len(self):
        with self.assertRaises(ValueError):
            self._ctx(starting_stacks=[10000] * 5)

    def test_rejects_empty_payouts(self):
        with self.assertRaises(ValueError):
            self._ctx(payouts=[])

    def test_rejects_bad_hero_seat(self):
        with self.assertRaises(ValueError):
            self._ctx(hero_seat=6)

    def test_rejects_negative_iterations(self):
        with self.assertRaises(ValueError):
            self._ctx(n_iterations=-1)

    def test_rejects_bad_weighting(self):
        with self.assertRaises(ValueError):
            self._ctx(average_weighting="quadratic")


class TestResultFields(unittest.TestCase):
    def test_result_constructs(self):
        from src.nlhe.subgame_solver import SubgameSolveResult
        res = SubgameSolveResult(
            root_policy=np.zeros(7, dtype=np.float32),
            root_blueprint=np.zeros(7, dtype=np.float32),
            legal_mask=np.zeros(7, dtype=np.float32),
            hero_seat=0, n_iterations=0, n_decision_nodes_cached=1,
        )
        self.assertFalse(res.degraded)
        self.assertTrue(np.isnan(res.converged_l1_tail))


# ============================================================
# K=0 path (real game + mock blueprint)
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestK0BlueprintPassthrough(unittest.TestCase):
    def _tree(self, depth=1, seed=42):
        from src.nlhe.subgame import build_subgame_tree
        state = _first_decision_state(seed=seed)
        return build_subgame_tree(state, max_action_depth=depth,
                                  rng=random.Random(seed + 1))

    def test_k0_matches_blueprint_masked_policy_bit_identical(self):
        from src.nlhe.solver import _strategy_from_advantages
        from src.nlhe.subgame_solver import solve_subgame, _mask_from_children
        tree = self._tree(depth=1)
        ctx = _make_ctx(tree, n_iterations=0)
        res = solve_subgame(tree, ctx)

        # Independently recompute the blueprint's masked policy at the root the
        # exact same way warm-up does, and require BIT-identity.
        hero = tree.root.current_player
        adv = _MockNets().predict_advantages(hero, None)
        mask = _mask_from_children(tree.root)
        expected = _strategy_from_advantages(adv.astype(np.float32), mask)

        self.assertTrue(np.array_equal(res.root_policy, expected),
                        f"{res.root_policy} != {expected}")
        # K=0: refined == blueprint, and nothing degraded (no leaves read).
        self.assertTrue(np.array_equal(res.root_policy, res.root_blueprint))
        self.assertEqual(res.n_iterations, 0)
        self.assertFalse(res.degraded)
        self.assertGreaterEqual(res.n_decision_nodes_cached, 1)

    def test_warmup_covers_interior_decision_nodes(self):
        # A depth-2 tree has interior decision nodes beyond the root; warm-up must
        # cache all of them (count matches the tree's decision-node count).
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=2)
        res = solve_subgame(tree, _make_ctx(tree, n_iterations=0))
        self.assertEqual(res.n_decision_nodes_cached, tree.n_decision_nodes)

    def test_root_policy_is_valid_distribution(self):
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=1)
        res = solve_subgame(tree, _make_ctx(tree, n_iterations=0))
        p, mask = res.root_policy, res.legal_mask
        self.assertEqual(p.shape, (7,))
        self.assertTrue(np.all(p >= 0.0))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        # mass only on legal actions
        self.assertEqual(float(p[mask == 0].sum()), 0.0)
        self.assertGreaterEqual(float(mask.sum()), 1.0)

    def test_reproducible_same_seed(self):
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=1, seed=7)
        r1 = solve_subgame(tree, _make_ctx(tree, n_iterations=0, rng=random.Random(123)))
        tree2 = self._tree(depth=1, seed=7)
        r2 = solve_subgame(tree2, _make_ctx(tree2, n_iterations=0, rng=random.Random(123)))
        self.assertTrue(np.array_equal(r1.root_policy, r2.root_policy))

    def test_root_not_hero_raises(self):
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=1)
        wrong = (tree.root.current_player + 1) % 6
        with self.assertRaises(ValueError):
            solve_subgame(tree, _make_ctx(tree, n_iterations=0, hero_seat=wrong))

    def test_k1_raises_not_implemented(self):
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=1)
        with self.assertRaises(NotImplementedError):
            solve_subgame(tree, _make_ctx(tree, n_iterations=1))


if __name__ == "__main__":
    unittest.main()
