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
# Shared fixtures (production game)
# ============================================================
#
# All tree-build tests load the REAL production game via six_max_sng (the
# universal_poker(...) wrapper with bettingAbstraction=fullgame, ~10k raw chip
# actions per node), NOT a hand-written ACPC gamedef. The old gamedef fixture
# happened to use OpenSpiel's default (smaller) betting abstraction, which hid
# the 600-leaves bug: the tree builder iterated raw legal_actions() and only
# looked correct because that fixture's action set was already small.


def _load_six_max_game(starting_stack: int = 10000):
    import pyspiel
    from src.nlhe.game_strings import six_max_sng
    return pyspiel.load_game(six_max_sng(starting_stack=starting_stack))


def _advance_past_chance(state, seed: int = 42):
    """Advance a state past any chance nodes (e.g. hole-card dealing) to the
    next decision point, sampling chance outcomes deterministically."""
    import random
    rng = random.Random(seed)
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        chosen = rng.choices(actions, weights=probs, k=1)[0]
        state = state.child(int(chosen))
    return state


def _first_decision_state(starting_stack: int = 10000, seed: int = 42):
    return _advance_past_chance(_load_six_max_game(starting_stack).new_initial_state(),
                                seed=seed)


def _discretized_at(state) -> dict:
    """Reproduce — independently of subgame.py — the discrete-action map that
    cfr6.traverse_6max enumerates at this decision state (cfr6.py:307-336).

    This is the reference the invariant tests compare against: it calls cfr6's
    OWN _build_view_6max plus discretize_legal_actions, so a passing assertion
    proves the tree builder enumerated that same set rather than raw chip ints.
    """
    from src.nlhe.actions import discretize_legal_actions
    from src.nlhe.cfr6 import _build_view_6max
    from src.nlhe.infoset6 import parse_state_6max, parse_state_repeated_6max
    if hasattr(state, "dealer_seat"):
        parsed = parse_state_repeated_6max(state)
    else:
        parsed = parse_state_6max(state)
    view = _build_view_6max(state, parsed)
    return discretize_legal_actions(list(state.legal_actions()), view)


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
        """Load the production 6-max NLHE game; return a state at first decision."""
        return _first_decision_state()

    def test_cannot_build_from_chance(self):
        """Building a subgame from a chance node should raise ValueError."""
        from src.nlhe.subgame import build_subgame_tree
        # A brand-new production game starts on a chance node (hole-card deal).
        state = _load_six_max_game().new_initial_state()
        self.assertTrue(state.is_chance_node(),
                        "expected initial state to be chance node")
        with self.assertRaises(ValueError):
            build_subgame_tree(state, max_action_depth=1)


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestTreeBuilderBasic(unittest.TestCase):
    """Basic tree-shape sanity checks."""

    def _decision_state(self):
        """Get past hole-card dealing to first decision point."""
        return _first_decision_state()

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
        """Build a tree that goes past the preflop street and into a flop chance.

        Under fullgame discretization a naive deep (depth-8) build branches ~6
        ways per decision and would balloon before reaching a chance node. So
        we instead fast-forward preflop by calling/checking (action 1) until
        one more call completes the round into the flop deal, then build a
        shallow tree from that last pre-flop decision: the CALL child is the
        flop chance node, reached at depth 1.
        """
        from src.nlhe.subgame import build_subgame_tree
        state = _first_decision_state()
        guard = 0
        while not state.child(1).is_chance_node():
            state = state.child(1)
            guard += 1
            assert guard < 20, "never reached a round-completing call"
        tree = build_subgame_tree(state, max_action_depth=2,
                                  chance_samples_per_node=4)
        return tree

    def test_chance_sample_count_bounded(self):
        """Every chance node should have at most chance_samples_per_node children."""
        tree = self._build_through_flop()
        if tree.n_chance_nodes == 0:
            self.skipTest("Tree didn't reach any chance nodes — try deeper")
        for n in tree.all_nodes:
            if n.is_chance:
                self.assertLessEqual(len(n.children), 4)


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestTreeBuilderDiscretization(unittest.TestCase):
    """Regression guard for the 600-leaves bug.

    The invariant these tests pin down: at every decision node, the tree
    builder enumerates EXACTLY the discrete action set that cfr6.traverse_6max
    enumerates — discretize_legal_actions(legal_chip, view).keys() — not the
    ~10k raw chip actions returned by state.legal_actions() under
    bettingAbstraction=fullgame.
    """

    def test_root_children_bounded_by_action_count(self):
        """A decision node can have at most len(DiscreteAction) children."""
        from src.nlhe.subgame import build_subgame_tree
        from src.nlhe.actions import DiscreteAction
        state = _first_decision_state()
        tree = build_subgame_tree(state, max_action_depth=1)
        self.assertLessEqual(len(tree.root.children), len(DiscreteAction))

    def test_root_children_match_discretized_set(self):
        """Root's children enumerate exactly discretize_legal_actions(...).keys()
        for the same view cfr6 would build."""
        from src.nlhe.subgame import build_subgame_tree
        state = _first_decision_state()
        tree = build_subgame_tree(state, max_action_depth=1)
        expected = {int(da) for da in _discretized_at(state).keys()}
        self.assertEqual(set(tree.root.action_at_child), expected)
        # children list stays parallel to action_at_child
        self.assertEqual(len(tree.root.children), len(tree.root.action_at_child))

    def test_every_decision_node_matches_walker_action_set(self):
        """The invariant, recursively: every decision node in the tree
        enumerates the same action set as the walker would at that state."""
        from src.nlhe.subgame import build_subgame_tree, iter_decision_nodes
        state = _first_decision_state()
        tree = build_subgame_tree(state, max_action_depth=3,
                                  chance_samples_per_node=2)
        checked = 0
        for node in iter_decision_nodes(tree):
            expected = {int(da) for da in _discretized_at(node.state).keys()}
            self.assertEqual(
                set(node.action_at_child), expected,
                f"decision node at depth {node.depth} diverged from walker set")
            self.assertEqual(len(node.children), len(node.action_at_child))
            checked += 1
        self.assertGreater(checked, 0, "expected at least one decision node")

    def test_facing_allin_restricted_child_count(self):
        """Facing an all-in shove, the discrete set is restricted (fold/call,
        plus the all-in alias) — child count matches that restricted set, never
        7 and never the ~10k raw chip actions.

        NOTE: discretize_legal_actions returns {FOLD, CALL, ALLIN} here, not
        just {FOLD, CALL}: with no raise legal, view.max_bet == 0, so ALLIN
        aliases to chip 0 (the fold chip). That is exactly what the walker
        enumerates, so a faithful tree builder produces 3 children here, and
        this test pins to that real set rather than a hardcoded 2.
        """
        from src.nlhe.subgame import build_subgame_tree
        from src.nlhe.actions import DiscreteAction
        state = _first_decision_state()
        allin_chip = _discretized_at(state)[DiscreteAction.ALLIN]
        shoved = _advance_past_chance(state.child(int(allin_chip)))
        restricted = _discretized_at(shoved)
        # Sanity: this really is a restricted node (no middle bet sizes).
        self.assertIn(DiscreteAction.FOLD, restricted)
        self.assertIn(DiscreteAction.CALL, restricted)
        for mid in (DiscreteAction.BET_33, DiscreteAction.BET_66,
                    DiscreteAction.BET_100, DiscreteAction.BET_200):
            self.assertNotIn(mid, restricted)

        tree = build_subgame_tree(shoved, max_action_depth=1)
        self.assertEqual(len(tree.root.children), len(restricted))
        self.assertLess(len(tree.root.children), len(DiscreteAction))
        self.assertEqual(set(tree.root.action_at_child),
                         {int(da) for da in restricted.keys()})

    def test_fold_to_end_hand_is_terminal_not_leaf(self):
        """A fold that ends the hand produces a TERMINAL node with returns
        populated — not a depth-limited LEAF."""
        from src.nlhe.subgame import build_subgame_tree
        from src.nlhe.actions import DiscreteAction
        # Fold around the table until one more fold would end the hand.
        state = _first_decision_state()
        guard = 0
        while not state.child(0).is_terminal():
            state = state.child(0)
            guard += 1
            self.assertLess(guard, 6, "expected hand to end within 6 folds")
        tree = build_subgame_tree(state, max_action_depth=1)
        fold_children = [c for c in tree.root.children
                         if c.action_from_parent == int(DiscreteAction.FOLD)]
        self.assertEqual(len(fold_children), 1)
        fold_child = fold_children[0]
        self.assertTrue(fold_child.is_terminal)
        self.assertFalse(fold_child.is_leaf)
        self.assertIsNotNone(fold_child.terminal_returns)
        self.assertEqual(len(fold_child.terminal_returns), 6)


# ============================================================
# Stage-5-B mitigation: chance collapses into leaves (no expansion)
# ============================================================

def _round_closing_decision(seed: int = 42):
    """A preflop decision where CALLing closes the betting round (child(1) is a
    chance node), so a depth-K tree reaches a board-deal chance node."""
    s = _first_decision_state(seed=seed)
    guard = 0
    while not s.child(1).is_chance_node():
        s = s.child(1)
        guard += 1
        if guard > 12:
            break
    return s


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestChanceCollapsesToLeaf(unittest.TestCase):
    def test_chance_becomes_leaf_no_expansion_no_explosion(self):
        from src.nlhe.subgame import build_subgame_tree, iter_leaf_nodes
        s = _round_closing_decision()
        self.assertTrue(s.child(1).is_chance_node(),
                        "fixture: calling should close the round into a chance node")
        tree = build_subgame_tree(s, max_action_depth=3, chance_samples_per_node=8)
        # Change A: chance is NEVER expanded into the tree -> no CHANCE-kind nodes,
        # and chance states appear only as LEAVES (current_player None).
        self.assertEqual(tree.n_chance_nodes, 0)
        chance_leaves = [lf for lf in iter_leaf_nodes(tree) if lf.state.is_chance_node()]
        self.assertGreater(len(chance_leaves), 0, "tree should reach >=1 chance leaf")
        for lf in chance_leaves:
            self.assertIsNone(lf.current_player)
        # No leaf explosion: bounded leaf count (was 2000+ when chance expanded x8
        # and compounded across streets within the depth budget).
        self.assertLess(tree.n_leaf_nodes, 200,
                        f"leaf explosion not bounded: {tree.n_leaf_nodes} leaves")


if __name__ == "__main__":
    unittest.main()
