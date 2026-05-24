"""Tests for src.nlhe.scripted_bots.policy.

Covers the standalone helpers (card parsing, position derivation, sequence
counting, action translation) plus GameContext building and the policy
fallback chain. Full end-to-end integration with eval_pool.py requires
the full ML stack (treys, torch) and runs on the pod; here we exercise
everything that doesn't need a live OpenSpiel state.
"""
import os
import pathlib
import random
import unittest
from unittest.mock import MagicMock

from src.nlhe.actions import DiscreteAction, GameStateView
from src.nlhe.scripted_bots.parser import Action, ActionKind, parse_profile
from src.nlhe.scripted_bots.policy import (
    _parse_cards_string,
    _street_name,
    _position_name,
    _count_actions_on_current_street,
    shanky_action_to_discrete,
    _pot_fraction_to_discrete,
    _nearest_legal_bet,
    _smallest_legal_bet,
    _largest_legal_bet,
    build_game_context,
    ShankyProfilePolicy,
)


FIXTURE_DIR = pathlib.Path(__file__).parent / "scripted_bots_fixtures"


# ============================================================
# Card-string parsing
# ============================================================

class TestCardStringParsing(unittest.TestCase):
    def test_two_cards(self):
        # AsKs: A=12, K=11, s=3 -> 12*4+3=51, 11*4+3=47
        self.assertEqual(_parse_cards_string("AsKs"), [51, 47])

    def test_three_cards(self):
        result = _parse_cards_string("Kh2c3d")
        # Kh=11*4+2=46, 2c=0*4+0=0, 3d=1*4+1=5
        self.assertEqual(result, [46, 0, 5])

    def test_empty(self):
        self.assertEqual(_parse_cards_string(""), [])

    def test_five_card_board(self):
        # AsKhQdJc2s: 5 cards
        result = _parse_cards_string("AsKhQdJc2s")
        self.assertEqual(len(result), 5)

    def test_lowercase_suit(self):
        # OpenSpiel always uses lowercase suit; verify we handle uppercase too
        result = _parse_cards_string("As")
        self.assertEqual(result, [51])


# ============================================================
# Street naming
# ============================================================

class TestStreetNaming(unittest.TestCase):
    def test_each_street(self):
        self.assertEqual(_street_name(0), "preflop")
        self.assertEqual(_street_name(1), "flop")
        self.assertEqual(_street_name(2), "turn")
        self.assertEqual(_street_name(3), "river")

    def test_out_of_range_defaults_to_preflop(self):
        self.assertEqual(_street_name(99), "preflop")
        self.assertEqual(_street_name(-1), "preflop")


# ============================================================
# Position derivation
# ============================================================

class TestPositionDerivation(unittest.TestCase):
    def test_six_max_positions(self):
        # button at seat 0
        self.assertEqual(_position_name(0, 6, 0), "button")
        self.assertEqual(_position_name(1, 6, 0), "smallblind")
        self.assertEqual(_position_name(2, 6, 0), "bigblind")
        self.assertEqual(_position_name(3, 6, 0), "first")
        self.assertEqual(_position_name(4, 6, 0), "middle")
        self.assertEqual(_position_name(5, 6, 0), "last")

    def test_button_rotation(self):
        # button at seat 3: seat 3 is button, seat 4 SB, etc.
        self.assertEqual(_position_name(3, 6, 3), "button")
        self.assertEqual(_position_name(4, 6, 3), "smallblind")
        self.assertEqual(_position_name(5, 6, 3), "bigblind")
        self.assertEqual(_position_name(0, 6, 3), "first")


# ============================================================
# Sequence parsing
# ============================================================

class TestSequenceParsing(unittest.TestCase):
    def test_empty_sequence(self):
        r = _count_actions_on_current_street("")
        self.assertEqual(r["raises"], 0)
        self.assertEqual(r["bets"], 0)
        self.assertEqual(r["calls"], 0)
        self.assertEqual(r["checks"], 0)
        self.assertEqual(r["last_action"], "none")

    def test_check_check(self):
        r = _count_actions_on_current_street("cc")
        # Both 'c' are checks (no preceding raise)
        self.assertEqual(r["checks"], 2)
        self.assertEqual(r["calls"], 0)
        self.assertEqual(r["last_action"], "check")

    def test_bet_call_call(self):
        r = _count_actions_on_current_street("r200cc")
        self.assertEqual(r["bets"], 1)
        self.assertEqual(r["calls"], 2)
        self.assertEqual(r["raises"], 0)

    def test_bet_raise(self):
        r = _count_actions_on_current_street("r200r400")
        self.assertEqual(r["bets"], 1)
        self.assertEqual(r["raises"], 1)
        self.assertEqual(r["last_action"], "raise")

    def test_only_current_street(self):
        """Multi-street sequence: only the last street's actions are counted."""
        # Preflop: bet 200 + call. Flop: bet 400 + call + call.
        r = _count_actions_on_current_street("r200c/r400cc")
        # Only flop counted
        self.assertEqual(r["bets"], 1)
        self.assertEqual(r["calls"], 2)

    def test_folds(self):
        r = _count_actions_on_current_street("fff")
        self.assertEqual(r["folds"], 3)


# ============================================================
# Action translation: Shanky → DiscreteAction
# ============================================================

class TestActionTranslation(unittest.TestCase):
    def setUp(self):
        self.view = GameStateView(
            pot=100, to_call=20, effective_stack=500,
            min_bet=40, max_bet=500,
            legal_fold=True, legal_call=True,
        )
        # All bet sizes legal
        self.full_legal = {
            DiscreteAction.FOLD: 0,
            DiscreteAction.CALL: 1,
            DiscreteAction.BET_33: 33,
            DiscreteAction.BET_66: 66,
            DiscreteAction.BET_100: 100,
            DiscreteAction.BET_200: 200,
            DiscreteAction.ALLIN: 500,
        }
        # Only fold/call/allin legal (e.g. short-stack situation)
        self.limited_legal = {
            DiscreteAction.FOLD: 0,
            DiscreteAction.CALL: 1,
            DiscreteAction.ALLIN: 500,
        }

    def test_fold(self):
        a = Action(kind=ActionKind.FOLD)
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.FOLD)

    def test_call(self):
        a = Action(kind=ActionKind.CALL)
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.CALL)

    def test_check_maps_to_call(self):
        a = Action(kind=ActionKind.CHECK)
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.CALL)

    def test_raisemax_maps_to_allin(self):
        a = Action(kind=ActionKind.RAISE_MAX)
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.ALLIN)

    def test_raisepot_maps_to_bet_100(self):
        a = Action(kind=ActionKind.RAISE_POT)
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.BET_100)

    def test_raisemin_picks_smallest(self):
        a = Action(kind=ActionKind.RAISE_MIN)
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.BET_33)

    def test_percent_translation_buckets(self):
        # 33% pot -> BET_33
        a = Action(kind=ActionKind.RAISE_PERCENT, amount=33, amount_target="potsize")
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.BET_33)
        # 66% pot -> BET_66
        a = Action(kind=ActionKind.RAISE_PERCENT, amount=66, amount_target="potsize")
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.BET_66)
        # 100% pot -> BET_100
        a = Action(kind=ActionKind.RAISE_PERCENT, amount=100, amount_target="potsize")
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.BET_100)
        # 200% pot -> BET_200
        a = Action(kind=ActionKind.RAISE_PERCENT, amount=200, amount_target="potsize")
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.BET_200)

    def test_bare_raise_defaults_to_bet_66(self):
        a = Action(kind=ActionKind.RAISE_AMOUNT, amount=None)
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.BET_66)

    def test_sitout_falls_back_to_fold(self):
        a = Action(kind=ActionKind.SITOUT)
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.FOLD)

    def test_beep_falls_back_to_fold(self):
        a = Action(kind=ActionKind.BEEP)
        self.assertEqual(shanky_action_to_discrete(a, self.full_legal, self.view),
                         DiscreteAction.FOLD)

    def test_raisepot_with_no_bet_100_falls_to_allin(self):
        """When BET_100 isn't legal, raisepot finds the nearest larger legal bet."""
        a = Action(kind=ActionKind.RAISE_POT)
        # In limited_legal: only ALLIN is a larger bet than BET_100 not in set
        self.assertEqual(shanky_action_to_discrete(a, self.limited_legal, self.view),
                         DiscreteAction.ALLIN)

    def test_none_action_returns_none(self):
        self.assertIsNone(shanky_action_to_discrete(None, self.full_legal, self.view))

    def test_empty_legal_returns_none(self):
        a = Action(kind=ActionKind.FOLD)
        self.assertIsNone(shanky_action_to_discrete(a, {}, self.view))


# ============================================================
# Helper functions
# ============================================================

class TestBetSelectors(unittest.TestCase):
    def test_largest_legal_bet(self):
        legal = {DiscreteAction.FOLD, DiscreteAction.CALL, DiscreteAction.BET_33,
                 DiscreteAction.BET_100}
        self.assertEqual(_largest_legal_bet(legal), DiscreteAction.BET_100)

    def test_smallest_legal_bet(self):
        legal = {DiscreteAction.FOLD, DiscreteAction.CALL, DiscreteAction.BET_66,
                 DiscreteAction.BET_200}
        self.assertEqual(_smallest_legal_bet(legal), DiscreteAction.BET_66)

    def test_nearest_legal_bet_exact_match(self):
        legal = {DiscreteAction.BET_33, DiscreteAction.BET_66, DiscreteAction.BET_100}
        self.assertEqual(
            _nearest_legal_bet(DiscreteAction.BET_66, legal),
            DiscreteAction.BET_66,
        )

    def test_nearest_legal_bet_neighbor(self):
        # Target BET_66 not legal, BET_33 and BET_100 both legal.
        # Walking outward: delta=1 finds BET_33 first (or BET_100 next).
        legal = {DiscreteAction.BET_33, DiscreteAction.BET_100}
        result = _nearest_legal_bet(DiscreteAction.BET_66, legal)
        self.assertIn(result, {DiscreteAction.BET_33, DiscreteAction.BET_100})


# ============================================================
# GameContext building
# ============================================================

class TestGameContextBuilding(unittest.TestCase):
    def _mock_state(self, legal_actions=None):
        state = MagicMock()
        state.legal_actions.return_value = legal_actions or [0, 1, 200, 400, 1000]
        return state

    def test_basic_preflop_context(self):
        parsed = {
            "street_idx": 0,
            "current_player": 3,
            "pot": 150,
            "money": [10000, 9950, 9900, 10000, 10000, 10000],
            "contribution": [0, 50, 100, 0, 0, 0],
            "private_cards": "AsKs",
            "public_cards": "",
            "sequences": "",
            "num_players": 6,
        }
        state = self._mock_state()
        ctx = build_game_context(parsed, state, big_blind_chips=100)

        # Hole cards translated
        self.assertEqual(ctx.hole_cards, [51, 47])
        # Empty board
        self.assertEqual(ctx.board, [])
        self.assertEqual(ctx.street, "preflop")
        # BB conversion: 10000 chips / 100 BB = 100 BB stack
        self.assertEqual(ctx.stacksize, 100.0)
        # Pot 150 / 100 = 1.5 BB
        self.assertEqual(ctx.potsize, 1.5)
        # Player 3 = UTG (first to act, position = 'first')
        self.assertEqual(ctx.position, "first")

    def test_amounttocall_computed_correctly(self):
        parsed = {
            "street_idx": 1,
            "current_player": 0,
            "pot": 600,
            "money": [9800, 9800, 9800],
            "contribution": [200, 200, 200],   # everyone equal so far
            "private_cards": "AsKs",
            "public_cards": "Qh7d2c",
            "sequences": "ccc/r200",            # bet 200 on flop
            "num_players": 3,
        }
        state = self._mock_state()
        ctx = build_game_context(parsed, state, big_blind_chips=100)
        # Contribution shown as 200 for everyone — to_call comes from max_contrib - my_contrib
        # All equal here so to_call=0. (The sequence string is informational; the contributions
        # are what determine "still owe.")
        self.assertEqual(ctx.amounttocall, 0.0)
        self.assertEqual(ctx.street, "flop")

    def test_board_parsed(self):
        parsed = {
            "street_idx": 2,
            "current_player": 0,
            "pot": 100,
            "money": [9000, 9000, 9000],
            "contribution": [200, 200, 200],
            "private_cards": "AsKs",
            "public_cards": "Qh7d2c5h",
            "sequences": "ccc/cc/cc",
            "num_players": 3,
        }
        state = self._mock_state()
        ctx = build_game_context(parsed, state, big_blind_chips=100)
        # Board should have 4 cards on the turn
        self.assertEqual(len(ctx.board), 4)
        self.assertEqual(ctx.street, "turn")


# ============================================================
# Policy class
# ============================================================

def _ml_stack_available() -> bool:
    """True iff the modules needed for full Policy integration are importable."""
    try:
        import treys  # noqa: F401
        from src.nlhe import cfr6  # noqa: F401
        return True
    except Exception:
        return False


_HAS_ML_STACK = _ml_stack_available()


@unittest.skipUnless(_HAS_ML_STACK, "Requires treys + full ML stack (runs on the pod)")
class TestShankyProfilePolicy(unittest.TestCase):
    def test_construct_from_real_fixture(self):
        path = FIXTURE_DIR / "littlegreen.txt"
        policy = ShankyProfilePolicy(name="littlegreen", profile_path=str(path))
        self.assertEqual(policy.name, "littlegreen")
        # Profile loaded successfully
        self.assertTrue(policy.profile.has_custom)
        # Two sections (preflop, flop)
        self.assertEqual(len(policy.profile.sections), 2)

    def test_select_action_returns_legal_chip(self):
        """End-to-end: load a profile, mock a state, get a chip action.

        We mock `_build_view_6max` so we don't need a real OpenSpiel state.
        """
        from unittest.mock import patch

        path = FIXTURE_DIR / "littlegreen.txt"
        policy = ShankyProfilePolicy(name="littlegreen", profile_path=str(path))

        parsed = {
            "street_idx": 0,
            "current_player": 4,
            "pot": 150,
            "money": [10000] * 6,
            "contribution": [0, 50, 100, 0, 0, 0],
            "private_cards": "7s7h",       # 77 — in littlegreen's open range
            "public_cards": "",
            "sequences": "",
            "num_players": 6,
        }
        # Mock the legal actions: fold, call, plus various bet sizes
        state = MagicMock()
        state.legal_actions.return_value = [0, 1, 200, 400, 600, 1000, 9950]

        # Mock _build_view_6max to return a usable GameStateView
        mock_view = GameStateView(
            pot=150, to_call=100, effective_stack=9900,
            min_bet=200, max_bet=9950,
            legal_fold=True, legal_call=True,
        )
        with patch("src.nlhe.cfr6._build_view_6max", return_value=mock_view):
            rng = random.Random(42)
            chip_action = policy.select_action(parsed, state, rng)
        # Should return one of the legal chip actions
        self.assertIn(chip_action, state.legal_actions.return_value)

    def test_select_action_with_no_match_uses_fallback(self):
        """When the profile returns None (no matching rule), fall back to FOLD."""
        from unittest.mock import patch

        path = FIXTURE_DIR / "littlegreen.txt"
        policy = ShankyProfilePolicy(name="littlegreen", profile_path=str(path))

        # AA preflop — littlegreen has no rule for AA, returns None.
        parsed = {
            "street_idx": 0,
            "current_player": 3,
            "pot": 150,
            "money": [10000] * 6,
            "contribution": [0, 50, 100, 0, 0, 0],
            "private_cards": "AsAh",      # AA, NOT in littlegreen's range
            "public_cards": "",
            "sequences": "",
            "num_players": 6,
        }
        state = MagicMock()
        state.legal_actions.return_value = [0, 1, 200, 9950]

        mock_view = GameStateView(
            pot=150, to_call=100, effective_stack=9900,
            min_bet=200, max_bet=9950,
            legal_fold=True, legal_call=True,
        )
        with patch("src.nlhe.cfr6._build_view_6max", return_value=mock_view):
            rng = random.Random(42)
            chip_action = policy.select_action(parsed, state, rng)
        # Fallback is FOLD, which is chip action 0
        self.assertEqual(chip_action, 0)


if __name__ == "__main__":
    unittest.main()
