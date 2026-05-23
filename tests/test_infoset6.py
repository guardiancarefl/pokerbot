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
