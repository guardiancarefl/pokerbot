"""Tests for src/nlhe/game_strings.py (Phase 4a: parametric game strings)."""
from __future__ import annotations
import random

import pyspiel
import pytest

from src.nlhe.game_strings import (
    PokerGameConfig,
    hunl_200bb,
    hunl_20bb,
    six_max_200bb,
    six_max_sng,
)


# ---- PokerGameConfig validation ----

def test_config_rejects_too_few_players():
    with pytest.raises(ValueError):
        PokerGameConfig(num_players=1)


def test_config_rejects_too_many_players():
    with pytest.raises(ValueError):
        PokerGameConfig(num_players=11)


def test_config_rejects_nonpositive_stack():
    with pytest.raises(ValueError):
        PokerGameConfig(starting_stack=0)
    with pytest.raises(ValueError):
        PokerGameConfig(starting_stack=-100)


def test_config_rejects_bad_blinds():
    with pytest.raises(ValueError):
        PokerGameConfig(small_blind=0)
    with pytest.raises(ValueError):
        PokerGameConfig(small_blind=100, big_blind=100)  # SB must be < BB
    with pytest.raises(ValueError):
        PokerGameConfig(small_blind=200, big_blind=100)  # SB > BB


def test_config_accepts_valid_defaults():
    cfg = PokerGameConfig()
    assert cfg.num_players == 2
    assert cfg.starting_stack == 20000


# ---- game string format ----

def test_hunl_string_contains_expected_pieces():
    s = hunl_200bb()
    assert "numPlayers=2" in s
    assert "stack=20000 20000" in s
    assert "blind=50 100" in s
    assert "bettingAbstraction=fullgame" in s


def test_six_max_string_contains_six_players():
    s = six_max_200bb()
    assert "numPlayers=6" in s
    # Stack repeated six times.
    assert "stack=20000 20000 20000 20000 20000 20000" in s
    # Blind padded with zeros for non-blind seats.
    assert "blind=50 100 0 0 0 0" in s


def test_first_player_differs_between_2_and_6():
    # HUNL: BB acts first preflop (firstPlayer=2 1 1 1)
    assert "firstPlayer=2 1 1 1" in hunl_200bb()
    # 6-max: UTG acts first preflop (firstPlayer=3 1 1 1)
    assert "firstPlayer=3 1 1 1" in six_max_200bb()


# ---- OpenSpiel can actually load each ----

@pytest.mark.parametrize("name,fn", [
    ("hunl_200bb", hunl_200bb),
    ("hunl_20bb", hunl_20bb),
    ("six_max_200bb", six_max_200bb),
    ("six_max_sng_default", six_max_sng),
])
def test_openspiel_loads(name, fn):
    """OpenSpiel can load each of the convenience configs."""
    game = pyspiel.load_game(fn())
    assert game.num_players() == (2 if "hunl" in name else 6)


def test_six_max_game_walks_to_terminal():
    """A 6-max game with random actions completes with zero-sum returns."""
    game = pyspiel.load_game(six_max_200bb())
    state = game.new_initial_state()
    rng = random.Random(2026)
    steps = 0
    while not state.is_terminal():
        steps += 1
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            a = rng.choices(actions, weights=probs, k=1)[0]
        else:
            a = rng.choice(state.legal_actions())
        state.apply_action(a)
        if steps > 5000:
            pytest.fail(f"6-max game did not reach terminal in 5000 steps")
    returns = state.returns()
    assert len(returns) == 6
    # Zero-sum: returns must sum to 0 (within float tolerance)
    assert abs(sum(returns)) < 1e-6, f"returns sum to {sum(returns)}, expected 0"


def test_six_max_sng_starting_stack_configurable():
    """six_max_sng(starting_stack=N) actually uses N."""
    s = six_max_sng(starting_stack=3000)
    assert "stack=3000 3000 3000 3000 3000 3000" in s


def test_three_handed_game_loads():
    """Edge case: 3-handed game (between HUNL and 6-max). Important for late-tournament play."""
    cfg = PokerGameConfig(num_players=3)
    game = pyspiel.load_game(cfg.to_universal_poker_string())
    assert game.num_players() == 3
