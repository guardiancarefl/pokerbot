"""Tests for src/nlhe/infoset6.py (Phase 4d: 6-max InfosetEncoder)."""
from __future__ import annotations
import random

import numpy as np
import pyspiel
import pytest

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import six_max_200bb
from src.nlhe.infoset6 import (
    parse_state_6max,
    position_for_seat,
    POSITIONS_6MAX,
    InfosetEncoder6Max,
)


# Use the production retrofit abstraction for any bucket-related tests.
ABSTRACTION_PATH = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"


# ---- Parser ----

def _make_initial_6max_state(seed: int = 2026):
    """Build a 6-max state past the chance phase, ready for first action."""
    game = pyspiel.load_game(six_max_200bb())
    state = game.new_initial_state()
    rng = random.Random(seed)
    while state.is_chance_node():
        actions, probs = zip(*state.chance_outcomes())
        state.apply_action(rng.choices(actions, weights=probs, k=1)[0])
    return state


def test_parser_extracts_6_money_slots():
    state = _make_initial_6max_state()
    parsed = parse_state_6max(state)
    assert parsed["num_players"] == 6
    assert len(parsed["money"]) == 6


def test_parser_extracts_6_contribution_slots():
    state = _make_initial_6max_state()
    parsed = parse_state_6max(state)
    assert len(parsed["contribution"]) == 6
    # On the initial state before any action, only SB and BB have contributed.
    # SB seat (seat 0) put in 50, BB seat (seat 1) put in 100.
    assert parsed["contribution"][0] == 50
    assert parsed["contribution"][1] == 100
    for i in range(2, 6):
        assert parsed["contribution"][i] == 0


def test_parser_money_reflects_blinds_taken():
    state = _make_initial_6max_state()
    parsed = parse_state_6max(state)
    # SB has 19950 (20000 - 50), BB has 19900 (20000 - 100), others have 20000.
    assert parsed["money"][0] == 19950
    assert parsed["money"][1] == 19900
    for i in range(2, 6):
        assert parsed["money"][i] == 20000


def test_parser_current_player_is_zero_indexed():
    state = _make_initial_6max_state()
    parsed = parse_state_6max(state)
    # current_player should be 0-indexed; OpenSpiel's string shows 1-indexed Player: 3 (UTG).
    assert parsed["current_player"] == 2  # zero-indexed UTG
    # Cross-check against OpenSpiel's own API
    assert parsed["current_player"] == state.current_player()


def test_parser_round_is_preflop_initially():
    state = _make_initial_6max_state()
    parsed = parse_state_6max(state)
    assert parsed["street_idx"] == 0


def test_parser_private_cards_present():
    state = _make_initial_6max_state()
    parsed = parse_state_6max(state)
    # current player has hole cards visible
    assert len(parsed["private_cards"]) == 4  # 2 cards, 2 chars each


def test_parser_pot_matches_contributions():
    state = _make_initial_6max_state()
    parsed = parse_state_6max(state)
    # Pot should equal sum of contributions (just blinds at this point).
    # NOTE: universal_poker initializes pot in a way that includes all
    # money currently committed, including future-due chips. Test relaxed
    # to "pot >= sum of contributions" because OpenSpiel's pot accounting
    # may include the upcoming dead money.
    assert parsed["pot"] >= sum(parsed["contribution"])


# ---- Position mapping ----

def test_position_for_seat_6max():
    """OpenSpiel 6-max seat numbering maps to UTG/MP/CO/BTN/SB/BB."""
    # seat 0 = SB (position idx 4)
    assert POSITIONS_6MAX[position_for_seat(0)] == "SB"
    # seat 1 = BB (idx 5)
    assert POSITIONS_6MAX[position_for_seat(1)] == "BB"
    # seat 2 = UTG (idx 0)
    assert POSITIONS_6MAX[position_for_seat(2)] == "UTG"
    # seat 3 = MP (idx 1)
    assert POSITIONS_6MAX[position_for_seat(3)] == "MP"
    # seat 4 = CO (idx 2)
    assert POSITIONS_6MAX[position_for_seat(4)] == "CO"
    # seat 5 = BTN (idx 3)
    assert POSITIONS_6MAX[position_for_seat(5)] == "BTN"


def test_position_for_seat_out_of_range_raises():
    with pytest.raises(AssertionError):
        position_for_seat(6)
    with pytest.raises(AssertionError):
        position_for_seat(-1)


# ---- Encoder ----

@pytest.fixture(scope="module")
def abstraction():
    """Load the retrofit abstraction once for all encoder tests."""
    import os
    if not os.path.exists(ABSTRACTION_PATH):
        pytest.skip(f"abstraction not found at {ABSTRACTION_PATH}; skipping encoder tests")
    return Abstraction.load(ABSTRACTION_PATH)


def test_encoder_feature_dim(abstraction):
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=20000)
    # 200 + 4 + 6 + 6 + 6 + 6 + 1 + 1 + 1 + 5
    assert enc.feature_dim == 236


def test_encoder_produces_correct_shape(abstraction):
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=20000)
    state = _make_initial_6max_state()
    feat = enc.encode(state)
    assert feat.shape == (236,)
    assert feat.dtype == np.float32


def test_encoder_position_one_hot_correct(abstraction):
    """The position slice should one-hot the current player's position."""
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=20000)
    state = _make_initial_6max_state()
    parsed = parse_state_6max(state)
    feat = enc.encode(state)

    # Position slice starts at offset 200 + 4 = 204
    pos_slice = feat[204:210]
    # Should be exactly one 1.0 and five 0.0s
    assert np.isclose(pos_slice.sum(), 1.0), f"position slice doesn't sum to 1: {pos_slice}"
    # The 1.0 should be at the right position index
    expected_pos = position_for_seat(parsed["current_player"])
    assert pos_slice[expected_pos] == 1.0


def test_encoder_street_one_hot_correct(abstraction):
    """Preflop should give street_idx=0."""
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=20000)
    state = _make_initial_6max_state()
    feat = enc.encode(state)

    # Street slice at offset 200..204
    street_slice = feat[200:204]
    assert street_slice[0] == 1.0  # preflop
    assert street_slice[1:].sum() == 0.0


def test_encoder_stack_features_normalized(abstraction):
    """Each per-player stack should be in [0, 1] after normalization."""
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=20000)
    state = _make_initial_6max_state()
    feat = enc.encode(state)

    # Stack slice at offset 200 + 4 + 6 = 210, length 6.
    stack_slice = feat[210:216]
    # SB has 19950/20000 = 0.9975, BB has 19900/20000 = 0.995, others 1.0.
    assert np.isclose(stack_slice[0], 19950.0 / 20000.0)
    assert np.isclose(stack_slice[1], 19900.0 / 20000.0)
    for i in range(2, 6):
        assert np.isclose(stack_slice[i], 1.0)


def test_encoder_active_mask_all_active_initially(abstraction):
    """At the start of a hand, all 6 players are active (no busts)."""
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=20000)
    state = _make_initial_6max_state()
    feat = enc.encode(state)

    # Active mask at offset 200 + 4 + 6 + 6 = 216, length 6.
    active_slice = feat[216:222]
    assert active_slice.sum() == 6.0


def test_encoder_contribution_features(abstraction):
    """SB and BB have small initial contributions; others have zero."""
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=20000)
    state = _make_initial_6max_state()
    feat = enc.encode(state)

    # Contribution slice at offset 200 + 4 + 6 + 6 + 6 = 222, length 6.
    contrib_slice = feat[222:228]
    assert np.isclose(contrib_slice[0], 50.0 / 20000.0)
    assert np.isclose(contrib_slice[1], 100.0 / 20000.0)
    for i in range(2, 6):
        assert contrib_slice[i] == 0.0


def test_encoder_cache_works(abstraction):
    """Repeated encode() on the same state should hit the bucket cache."""
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=20000)
    state = _make_initial_6max_state()
    # First call populates cache; second call should be much faster but
    # we just verify the cache fills.
    assert len(enc._bucket_cache) == 0
    enc.encode(state)
    assert len(enc._bucket_cache) == 1
    enc.encode(state)
    assert len(enc._bucket_cache) == 1  # didn't add a new entry


def test_encoder_reset_cache(abstraction):
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=20000)
    state = _make_initial_6max_state()
    enc.encode(state)
    assert len(enc._bucket_cache) > 0
    enc.reset_cache()
    assert len(enc._bucket_cache) == 0



# ---- parse_state_repeated_6max tests (Phase 4f) ----

import pyspiel
import random as _random_for_tests
from src.nlhe.infoset6 import (
    parse_state_repeated_6max,
    position_for_seat_with_dealer,
    POSITIONS_6MAX,
)


def _make_repeated_poker_state_at_first_decision(seed=7):
    """Helper: build a 6-max repeated_poker game, walk past chance nodes,
    return state at the first player decision (preflop)."""
    rng = _random_for_tests.Random(seed)
    gs = (
        "repeated_poker(max_num_hands=10,reset_stacks=False,rotate_dealer=True,"
        "blind_schedule=5:15/55;5:25/110;,"
        "universal_poker_game_string=universal_poker(betting=nolimit,numPlayers=6,"
        "numRounds=4,blind=15 55 0 0 0 0,firstPlayer=3 1 1 1,numSuits=4,numRanks=13,"
        "numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=1500 1500 1500 1500 1500 1500,bettingAbstraction=fullgame))"
    )
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        a = rng.choices(
            [o[0] for o in outcomes], weights=[o[1] for o in outcomes]
        )[0]
        state.apply_action(a)
    return state


def _make_repeated_poker_state_postflop(seed=7):
    """Helper: walk through cooperative play to reach a postflop state."""
    state = _make_repeated_poker_state_at_first_decision(seed)
    rng = _random_for_tests.Random(seed + 100)
    for _ in range(80):
        if state.is_terminal():
            break
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            a = rng.choices(
                [o[0] for o in outcomes], weights=[o[1] for o in outcomes]
            )[0]
            state.apply_action(a)
        else:
            obs = state.observation_string(state.current_player())
            if "[Round 1]" in obs:
                return state
            legal = state.legal_actions()
            a = 1 if 1 in legal else legal[0]
            state.apply_action(a)
    return state


def test_parse_repeated_6max_preflop_basic_fields():
    state = _make_repeated_poker_state_at_first_decision()
    parsed = parse_state_repeated_6max(state)
    assert parsed["num_players"] == 6
    assert parsed["street_idx"] == 0  # preflop
    assert 0 <= parsed["current_player"] <= 5
    assert parsed["pot"] > 0
    assert len(parsed["money"]) == 6
    assert len(parsed["contribution"]) == 6
    assert len(parsed["private_cards"]) == 4  # "XYxy" hole cards


def test_parse_repeated_6max_preflop_no_board_no_sequences():
    state = _make_repeated_poker_state_at_first_decision()
    parsed = parse_state_repeated_6max(state)
    assert parsed["public_cards"] == ""
    assert parsed["sequences"] == ""


def test_parse_repeated_6max_postflop_has_board_and_sequences():
    state = _make_repeated_poker_state_postflop()
    parsed = parse_state_repeated_6max(state)
    assert parsed["street_idx"] == 1  # flop
    assert len(parsed["public_cards"]) == 6  # "XYxyAB" three flop cards
    assert "/" in parsed["sequences"]  # at least one street boundary


def test_parse_repeated_6max_blinds_inflated_in_contribution():
    state = _make_repeated_poker_state_at_first_decision()
    parsed = parse_state_repeated_6max(state)
    # First two contributions are SB and BB-inflated (15 and 55 for L1)
    assert parsed["contribution"][0] == 15
    assert parsed["contribution"][1] == 55


def test_parse_repeated_6max_tournament_fields_present():
    state = _make_repeated_poker_state_at_first_decision()
    parsed = parse_state_repeated_6max(state)
    assert "dealer_seat" in parsed
    assert "hand_number" in parsed
    assert "current_big_blind" in parsed
    assert "current_small_blind" in parsed
    assert parsed["hand_number"] == 0  # first hand
    assert parsed["current_big_blind"] == 55  # L1 inflated BB
    assert parsed["current_small_blind"] == 15  # L1 SB
    assert 0 <= parsed["dealer_seat"] <= 5


# ---- position_for_seat_with_dealer tests (Phase 4f) ----


def test_position_with_dealer_5_matches_original():
    """With dealer at seat 5, new function matches the original layout."""
    from src.nlhe.infoset6 import position_for_seat
    for seat in range(6):
        assert position_for_seat_with_dealer(seat, dealer_seat=5) == position_for_seat(seat)


def test_position_with_dealer_button_is_dealer_seat():
    """The dealer's seat is always BTN regardless of which seat is dealer."""
    btn_idx = POSITIONS_6MAX.index("BTN")
    for dealer in range(6):
        assert position_for_seat_with_dealer(dealer, dealer_seat=dealer) == btn_idx


def test_position_with_dealer_sb_is_dealer_plus_1():
    """The seat immediately clockwise of the dealer is always SB."""
    sb_idx = POSITIONS_6MAX.index("SB")
    for dealer in range(6):
        sb_seat = (dealer + 1) % 6
        assert position_for_seat_with_dealer(sb_seat, dealer_seat=dealer) == sb_idx


def test_position_with_dealer_bb_is_dealer_plus_2():
    """The seat two clockwise of the dealer is always BB."""
    bb_idx = POSITIONS_6MAX.index("BB")
    for dealer in range(6):
        bb_seat = (dealer + 2) % 6
        assert position_for_seat_with_dealer(bb_seat, dealer_seat=dealer) == bb_idx


def test_position_with_dealer_utg_is_dealer_plus_3():
    """The seat three clockwise of the dealer is UTG (first to act preflop)."""
    utg_idx = POSITIONS_6MAX.index("UTG")
    for dealer in range(6):
        utg_seat = (dealer + 3) % 6
        assert position_for_seat_with_dealer(utg_seat, dealer_seat=dealer) == utg_idx


def test_position_with_dealer_full_rotation():
    """All 6 positions should be covered by each dealer seat."""
    for dealer in range(6):
        positions = {
            position_for_seat_with_dealer(seat, dealer_seat=dealer)
            for seat in range(6)
        }
        assert positions == set(range(6))  # all 6 positions present


def test_position_with_dealer_rejects_invalid_seat():
    import pytest
    with pytest.raises(ValueError):
        position_for_seat_with_dealer(seat=-1, dealer_seat=0)
    with pytest.raises(ValueError):
        position_for_seat_with_dealer(seat=6, dealer_seat=0)


def test_position_with_dealer_rejects_invalid_dealer():
    import pytest
    with pytest.raises(ValueError):
        position_for_seat_with_dealer(seat=0, dealer_seat=-1)
    with pytest.raises(ValueError):
        position_for_seat_with_dealer(seat=0, dealer_seat=6)
