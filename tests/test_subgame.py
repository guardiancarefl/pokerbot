"""Tests for src.nlhe.subgame.

Two test categories:
  - Pure-data tests: SubgameNode/SubgameTree dataclass behavior, helpers.
    Sandbox-runnable.
  - Tree-build tests: actually constructs a subgame tree from a real
    OpenSpiel state. Requires open_spiel and our existing 6-max game
    machinery (runs on pod).
"""
from __future__ import annotations

import unittest


def _open_spiel_available() -> bool:
    try:
        import pyspiel  # noqa: F401
        return True
    except Exception:
        return False


_HAS_OPEN_SPIEL = _open_spiel_available()


# ============================================================
# Pure-data tests (sandbox)
# ============================================================

class TestNodeKinds(unittest.TestCase):
    def test_kinds_are_distinct(self):
        from src.nlhe.subgame import NodeKind
        kinds = [NodeKind.DECISION, NodeKind.CHANCE,
                 NodeKind.TERMINAL, NodeKind.LEAF]
        self.assertEqual(len(kinds), len(set(kinds)))


class TestSubgameNodeBoolHelpers(unittest.TestCase):
    def test_decision_node_flags(self):
        from src.nlhe.subgame import SubgameNode, NodeKind
        n = SubgameNode(kind=NodeKind.DECISION, state=None, depth=0)
        self.assertTrue(n.is_decision)
        self.assertFalse(n.is_chance)
        self.assertFalse(n.is_terminal)
        self.assertFalse(n.is_leaf)

    def test_chance_node_flags(self):
        from src.nlhe.subgame import SubgameNode, NodeKind
        n = SubgameNode(kind=NodeKind.CHANCE, state=None, depth=2)
        self.assertTrue(n.is_chance)
        self.assertFalse(n.is_decision)

    def test_terminal_node_flags(self):
        from src.nlhe.subgame import SubgameNode, NodeKind
        n = SubgameNode(kind=NodeKind.TERMINAL, state=None, depth=5,
                       terminal_returns=[1.0, -1.0, 0.0])
        self.assertTrue(n.is_terminal)
        self.assertEqual(n.terminal_returns, [1.0, -1.0, 0.0])

    def test_leaf_node_flags(self):
        from src.nlhe.subgame import SubgameNode, NodeKind
        n = SubgameNode(kind=NodeKind.LEAF, state=None, depth=4)
        self.assertTrue(n.is_leaf)
        self.assertIsNone(n.terminal_returns)


class TestSubgameTreeDataclass(unittest.TestCase):
    def test_empty_tree_summary(self):
        from src.nlhe.subgame import SubgameNode, SubgameTree, NodeKind
        root = SubgameNode(kind=NodeKind.LEAF, state=None, depth=0)
        tree = SubgameTree(root=root, all_nodes=[root], n_leaf_nodes=1)
        s = tree.summary()
        self.assertIn("0 decisions", s)
        self.assertIn("1 leaves", s)


# ============================================================
# Tree-build tests (require open_spiel)
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestTreeBuilderRequirements(unittest.TestCase):
    """Sanity-check that we cannot build subgames from chance or terminal states."""

    def _make_initial_state(self):
        """Load 6-max NLHE and return a state at a decision point."""
        import pyspiel
        # Universal poker, 6 players, betsize=4 abstraction, 100 BB starting
        gdef = (
            "universal_poker(betting=nolimit,numPlayers=6,numRounds=4,"
            "blind=50 100 0 0 0 0,firstPlayer=3 1 1 1,"
            "numSuits=4,numRanks=13,numHoleCards=2,"
            "numBoardCards=0 3 1 1,stack=10000 10000 10000 10000 10000 10000,"
            "bettingAbstraction=fullgame)"
        )
        game = pyspiel.universal_poker.load_universal_poker_from_acpc_gamedef(
            "GAMEDEF\nnolimit\nnumPlayers = 6\nnumRounds = 4\nblind = 50 100 0 0 0 0\n"
            "firstPlayer = 4 1 1 1\nnumSuits = 4\nnumRanks = 13\nnumHoleCards = 2\n"
            "numBoardCards = 0 3 1 1\nstack = 10000 10000 10000 10000 10000 10000\n"
            "END GAMEDEF\n"
        )
        state = game.new_initial_state()
        # Advance past chance nodes (deal hole cards)
        import random
        rng = random.Random(42)
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            chosen = rng.choices(actions, weights=probs, k=1)[0]
            state = state.child(int(chosen))
        return state

    def test_cannot_build_from_chance(self):
        """Building a subgame from a chance node should raise ValueError."""
        from src.nlhe.subgame import build_subgame_tree
        import pyspiel
        # Get a brand new game (will be at chance node)
        game = pyspiel.universal_poker.load_universal_poker_from_acpc_gamedef(
            "GAMEDEF\nnolimit\nnumPlayers = 6\nnumRounds = 4\nblind = 50 100 0 0 0 0\n"
            "firstPlayer = 4 1 1 1\nnumSuits = 4\nnumRanks = 13\nnumHoleCards = 2\n"
            "numBoardCards = 0 3 1 1\nstack = 10000 10000 10000 10000 10000 10000\n"
            "END GAMEDEF\n"
        )
        state = game.new_initial_state()
        self.assertTrue(state.is_chance_node(),
                        "expected initial state to be chance node")
        with self.assertRaises(ValueError):
            build_subgame_tree(state, max_action_depth=1)


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestTreeBuilderBasic(unittest.TestCase):
    """Basic tree-shape sanity checks."""

    def _decision_state(self):
        """Get past hole-card dealing to first decision point."""
        import pyspiel
        import random
        game = pyspiel.universal_poker.load_universal_poker_from_acpc_gamedef(
            "GAMEDEF\nnolimit\nnumPlayers = 6\nnumRounds = 4\nblind = 50 100 0 0 0 0\n"
            "firstPlayer = 4 1 1 1\nnumSuits = 4\nnumRanks = 13\nnumHoleCards = 2\n"
            "numBoardCards = 0 3 1 1\nstack = 10000 10000 10000 10000 10000 10000\n"
            "END GAMEDEF\n"
        )
        state = game.new_initial_state()
        rng = random.Random(42)
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            chosen = rng.choices(actions, weights=probs, k=1)[0]
            state = state.child(int(chosen))
        return state

    def test_root_is_decision(self):
        """When built from a decision state, root must be a decision node."""
        from src.nlhe.subgame import build_subgame_tree
        state = self._decision_state()
        tree = build_subgame_tree(state, max_action_depth=1)
        self.assertTrue(tree.root.is_decision)
        self.assertIsNotNone(tree.root.current_player)

    def test_depth_1_has_only_leaves_as_children(self):
        """At max_action_depth=1, the root's children should be either leaves or
        terminals (or chance, if a fold ends the game)."""
        from src.nlhe.subgame import build_subgame_tree
        state = self._decision_state()
        tree = build_subgame_tree(state, max_action_depth=1)
        for child in tree.root.children:
            self.assertIn(child.kind.value, ("leaf", "terminal", "chance"))

    def test_node_counts_consistency(self):
        """all_nodes count = decisions + chance + terminals + leaves."""
        from src.nlhe.subgame import build_subgame_tree
        state = self._decision_state()
        tree = build_subgame_tree(state, max_action_depth=2)
        total = (tree.n_decision_nodes + tree.n_chance_nodes
                 + tree.n_terminal_nodes + tree.n_leaf_nodes)
        self.assertEqual(total, len(tree.all_nodes))

    def test_deeper_tree_has_more_nodes(self):
        """Increasing max_action_depth should monotonically increase node count."""
        from src.nlhe.subgame import build_subgame_tree
        state = self._decision_state()
        tree_d1 = build_subgame_tree(state, max_action_depth=1,
                                     chance_samples_per_node=2)
        tree_d2 = build_subgame_tree(state, max_action_depth=2,
                                     chance_samples_per_node=2)
        self.assertGreater(len(tree_d2.all_nodes), len(tree_d1.all_nodes))

    def test_infoset_groups_populated(self):
        """Tree built at depth>0 should have at least one infoset group
        (from the root decision)."""
        from src.nlhe.subgame import build_subgame_tree
        state = self._decision_state()
        tree = build_subgame_tree(state, max_action_depth=1)
        self.assertGreaterEqual(len(tree.infoset_groups), 1)

    def test_terminal_returns_present(self):
        """Any terminal node found should carry valid per-player returns."""
        from src.nlhe.subgame import build_subgame_tree, iter_terminal_nodes
        state = self._decision_state()
        tree = build_subgame_tree(state, max_action_depth=4)
        for tn in iter_terminal_nodes(tree):
            self.assertIsNotNone(tn.terminal_returns)
            # 6 players in this game
            self.assertEqual(len(tn.terminal_returns), 6)

    def test_descendants_count_correct(self):
        """For a leaf node, n_descendants should be 0."""
        from src.nlhe.subgame import build_subgame_tree, iter_leaf_nodes
        state = self._decision_state()
        tree = build_subgame_tree(state, max_action_depth=2)
        for leaf in iter_leaf_nodes(tree):
            self.assertEqual(leaf.n_descendants, 0)
        # Root descendants = total nodes - 1
        self.assertEqual(tree.root.n_descendants, len(tree.all_nodes) - 1)


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestChanceSubsampling(unittest.TestCase):
    """Tests that chance nodes are sub-sampled, not fully expanded."""

    def _build_through_flop(self):
        """Build a tree that goes past the preflop street and into chance flop."""
        import pyspiel
        import random
        from src.nlhe.subgame import build_subgame_tree
        game = pyspiel.universal_poker.load_universal_poker_from_acpc_gamedef(
            "GAMEDEF\nnolimit\nnumPlayers = 6\nnumRounds = 4\nblind = 50 100 0 0 0 0\n"
            "firstPlayer = 4 1 1 1\nnumSuits = 4\nnumRanks = 13\nnumHoleCards = 2\n"
            "numBoardCards = 0 3 1 1\nstack = 10000 10000 10000 10000 10000 10000\n"
            "END GAMEDEF\n"
        )
        state = game.new_initial_state()
        rng = random.Random(42)
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            chosen = rng.choices(actions, weights=probs, k=1)[0]
            state = state.child(int(chosen))
        # Build a deeper tree to reach flop chance node
        tree = build_subgame_tree(state, max_action_depth=8,
                                  chance_samples_per_node=4,
                                  rng=rng)
        return tree

    def test_chance_sample_count_bounded(self):
        """Every chance node should have at most chance_samples_per_node children."""
        tree = self._build_through_flop()
        if tree.n_chance_nodes == 0:
            self.skipTest("Tree didn't reach any chance nodes — try deeper")
        for n in tree.all_nodes:
            if n.is_chance:
                self.assertLessEqual(len(n.children), 4)


if __name__ == "__main__":
    unittest.main()
