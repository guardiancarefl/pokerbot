"""Structural tests for src.nlhe.subgame_leaf (B1c sub-step 2, Stage B scaffold).

Stage B ships only the Q9 interface contracts (no evaluation behavior). These
tests pin the scaffold: the context constructs with its documented defaults, the
BlueprintProvider protocol accepts a real solver, SubgameNode has the new
leaf_value field defaulting None, and the entry points accept the right argument
types (raising NotImplementedError until Stages C/D/E land).
"""
from __future__ import annotations

import glob
import os
import unittest


def _open_spiel_available() -> bool:
    try:
        import pyspiel  # noqa: F401
        return True
    except Exception:
        return False


_HAS_OPEN_SPIEL = _open_spiel_available()


def _find_solver_artifacts():
    """Locate an abstraction, a 6-max checkpoint, and the tournament structure.

    Returns (abstraction_path, checkpoint_path, structure_path) or None if any
    is absent (so the real-solver test skips gracefully off-host)."""
    abstr = sorted(glob.glob("runs/abstraction_*/abstraction.pkl"))
    ckpts = sorted(glob.glob("runs/six_max_*/checkpoints/ckpt_iter_*.pt"))
    struct = "configs/ignition_double_up_6max_turbo.yaml"
    if not abstr or not ckpts or not os.path.exists(struct):
        return None
    return abstr[0], ckpts[0], struct


def _make_minimal_context(blueprint=None):
    """A LeafEvalContext with only the required fields supplied."""
    from src.nlhe.subgame_leaf import LeafEvalContext
    from src.nlhe.biased_policy import BiasedBlueprint
    from src.nlhe.icm import sng_payouts_6max_double_up
    return LeafEvalContext(
        blueprint=blueprint if blueprint is not None else object(),
        biased_blueprint=BiasedBlueprint(),
        starting_stacks=[10000] * 6,
        payouts=sng_payouts_6max_double_up(),
        hero_seat=0,
    )


# ============================================================
# Mode + protocol (sandbox-runnable)
# ============================================================

class TestLeafEvalMode(unittest.TestCase):
    def test_two_modes_distinct(self):
        from src.nlhe.subgame_leaf import LeafEvalMode
        self.assertNotEqual(LeafEvalMode.BEST_RESPONSE, LeafEvalMode.PROFILE_SAMPLE)
        self.assertEqual(LeafEvalMode.BEST_RESPONSE.value, "best_response")
        self.assertEqual(LeafEvalMode.PROFILE_SAMPLE.value, "profile_sample")


class TestBlueprintProviderProtocol(unittest.TestCase):
    def test_runtime_checkable_stub(self):
        from src.nlhe.subgame_leaf import BlueprintProvider

        class _Stub:
            encoder = object()
            policy_nets = object()

        class _Missing:
            encoder = object()
            # no policy_nets

        self.assertIsInstance(_Stub(), BlueprintProvider)
        self.assertNotIsInstance(_Missing(), BlueprintProvider)
        self.assertNotIsInstance(object(), BlueprintProvider)


class TestLeafEvalContextDefaults(unittest.TestCase):
    def test_constructs_with_defaults(self):
        from src.nlhe.subgame_leaf import LeafEvalMode
        ctx = _make_minimal_context()
        # documented defaults
        self.assertEqual(ctx.mode, LeafEvalMode.BEST_RESPONSE)
        self.assertEqual(ctx.n_samples, 8)            # provisional design M
        self.assertIsNone(ctx.opponent_prior)
        self.assertIsNone(ctx.rng)
        self.assertTrue(ctx.icm_short_circuit)
        self.assertIsNone(ctx.time_budget_s)
        self.assertEqual(ctx.num_paid, 3)
        # required fields preserved
        self.assertEqual(ctx.hero_seat, 0)
        self.assertEqual(list(ctx.starting_stacks), [10000] * 6)


# ============================================================
# SubgameNode.leaf_value field
# ============================================================

class TestSubgameNodeLeafValueField(unittest.TestCase):
    def test_leaf_value_defaults_none_and_settable(self):
        from src.nlhe.subgame import SubgameNode, NodeKind
        n = SubgameNode(kind=NodeKind.LEAF, state=None, depth=4)
        self.assertIsNone(n.leaf_value)
        n.leaf_value = [0.1, -0.2, 0.0, 0.1, 0.0, 0.0]
        self.assertEqual(len(n.leaf_value), 6)

    def test_existing_node_construction_unbroken(self):
        # The field is appended with a default, so positional/keyword construction
        # used elsewhere keeps working.
        from src.nlhe.subgame import SubgameNode, NodeKind
        n = SubgameNode(kind=NodeKind.DECISION, state=None, depth=0,
                        current_player=2)
        self.assertIsNone(n.leaf_value)
        self.assertEqual(n.current_player, 2)


# ============================================================
# Entry-point signatures (scaffold raises NotImplementedError)
# ============================================================

class TestEntryPointSignatures(unittest.TestCase):
    def test_evaluate_leaf_accepts_node_and_ctx(self):
        from src.nlhe.subgame_leaf import evaluate_leaf
        from src.nlhe.subgame import SubgameNode, NodeKind
        node = SubgameNode(kind=NodeKind.LEAF, state=None, depth=4)
        ctx = _make_minimal_context()
        with self.assertRaises(NotImplementedError):
            evaluate_leaf(node, ctx)

    def test_evaluate_leaves_accepts_tree_and_ctx(self):
        from src.nlhe.subgame_leaf import evaluate_leaves
        from src.nlhe.subgame import SubgameNode, SubgameTree, NodeKind
        root = SubgameNode(kind=NodeKind.LEAF, state=None, depth=0)
        tree = SubgameTree(root=root, all_nodes=[root], n_leaf_nodes=1)
        ctx = _make_minimal_context()
        with self.assertRaises(NotImplementedError):
            evaluate_leaves(tree, ctx)


# ============================================================
# Protocol accepts a REAL solver (requires artifacts on host)
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestBlueprintProviderRealSolver(unittest.TestCase):
    def test_real_solver_satisfies_protocol(self):
        artifacts = _find_solver_artifacts()
        if artifacts is None:
            self.skipTest("solver artifacts (abstraction/checkpoint/structure) not present")
        abstr_path, ckpt_path, struct_path = artifacts
        from src.nlhe.abstraction import Abstraction
        from src.nlhe.game_strings import TournamentStructure
        from scripts.eval_6max_self_play import _load_solver
        from src.nlhe.subgame_leaf import BlueprintProvider

        abstr = Abstraction.load(abstr_path)
        structure = TournamentStructure.from_yaml(struct_path)
        solver = _load_solver(ckpt_path, abstr, structure)

        self.assertIsInstance(solver, BlueprintProvider)
        # the attributes the leaf evaluator will actually use exist:
        self.assertTrue(hasattr(solver.encoder, "encode_from_parsed"))
        self.assertTrue(hasattr(solver.policy_nets, "predict_advantages"))

        # And it slots into a context as the blueprint.
        ctx = _make_minimal_context(blueprint=solver)
        self.assertIs(ctx.blueprint, solver)


if __name__ == "__main__":
    unittest.main()
