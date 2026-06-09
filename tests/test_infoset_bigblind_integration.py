"""Integration guard: parse_state_6max must expose the raw big blind.

The short-stack floor reads parsed["big_blind"] via _hero_eff_bb_from_parsed
(src/nlhe/integration/live_loop.py) and silently no-ops when the key is
missing or 0 — _hero_eff_bb_from_parsed returns (0.0, 0) and
apply_short_stack_floor returns the input policy unchanged. The floor's unit
tests hand-construct the parsed dict, so they cannot catch a parser that
stops emitting big_blind. These tests build a REAL OpenSpiel state through
the same game-string path the live loop and the A/B harness use
(TournamentStructure.to_inner_game_string_for_state -> pyspiel.load_game ->
parse_state_6max) and assert the whole chain fires.
"""
from __future__ import annotations

import numpy as np
import pyspiel

from src.nlhe.actions import DiscreteAction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.integration.live_loop import (
    _hero_eff_bb_from_parsed,
    apply_short_stack_floor,
)

STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"

N_ACT = len(DiscreteAction)
A_FOLD = int(DiscreteAction.FOLD)
A_CALL = int(DiscreteAction.CALL)
A_ALLIN = int(DiscreteAction.ALLIN)
INTERMEDIATE_BETS = tuple(
    int(a) for a in DiscreteAction
    if int(a) not in (A_FOLD, A_CALL, A_ALLIN)
)


def _real_decision_state(stacks, dealer_seat=0, level=1):
    """Build a single-hand universal_poker state at the first decision node,
    via the same real-ante game-string path the live loop replays through."""
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    bl = structure.level(level)
    game_str = structure.to_inner_game_string_for_state(bl, stacks, dealer_seat)
    game = pyspiel.load_game(game_str)
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(int(state.chance_outcomes()[0][0]))
    assert state.current_player() >= 0
    return state


def test_parse_state_6max_exposes_raw_big_blind():
    state = _real_decision_state(stacks=[1500] * 6)
    parsed = parse_state_6max(state)
    assert "big_blind" in parsed, (
        "parse_state_6max no longer emits 'big_blind' — the short-stack "
        "floor will silently no-op in live play"
    )
    # L1 of the Ignition turbo structure is 15/25/5; raw BB (not inflated).
    assert parsed["big_blind"] == 25


def test_hero_eff_bb_positive_on_real_state():
    state = _real_decision_state(stacks=[1500] * 6)
    parsed = parse_state_6max(state)
    eff_bb, bb = _hero_eff_bb_from_parsed(parsed)
    assert bb == 25
    assert eff_bb > 0.0
    # ~1500 chips at BB=25 is deep: sanity-band the magnitude.
    assert 30.0 < eff_bb < 70.0


def test_short_stack_floor_fires_through_real_parser():
    # 140 chips at L1 (BB=25) is ~5.5 BB effective for every seat —
    # inside the 6.0 BB default threshold, facing the BB preflop.
    state = _real_decision_state(stacks=[140] * 6)
    parsed = parse_state_6max(state)

    eff_bb, bb = _hero_eff_bb_from_parsed(parsed)
    assert bb == 25
    assert 0.0 < eff_bb <= 6.0

    legal_mask = np.ones(N_ACT, dtype=np.float64)
    policy = np.full(N_ACT, 1.0 / N_ACT, dtype=np.float64)

    out = apply_short_stack_floor(policy, legal_mask, parsed, state)

    assert out is not policy, (
        "short-stack floor did not fire on a <=6 BB state built through "
        "the real parser"
    )
    for idx in INTERMEDIATE_BETS:
        assert out[idx] == 0.0
    # Facing the BB's forced post -> keep {FOLD, CALL, ALLIN}.
    assert out[A_FOLD] > 0.0
    assert out[A_CALL] > 0.0
    assert out[A_ALLIN] > 0.0
    assert abs(float(out.sum()) - 1.0) < 1e-6
