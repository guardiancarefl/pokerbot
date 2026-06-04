"""Unit test: 6-max universal_poker hole-card deal order matches the
codified DEAL_ORDER_SEQUENCE in src/nlhe/integration/replay.py.

This locks the empirically-verified order so an OpenSpiel upgrade or any
change to game_strings.to_inner_game_string_for_state can't silently shift
the deal sequence under the Phase 2 forced-card dealer (which would
silently mis-deal hero hole cards to the wrong seat).
"""
from __future__ import annotations

import re

import pyspiel
import pytest

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.replay import DEAL_ORDER_SEQUENCE
from src.nlhe.integration.scraper_schema import NUM_SEATS

_PRIVATE_RE = re.compile(r"\[Private:\s+([^\]]*)\]")


def _private_for(state, seat: int) -> str:
    info = state.information_state_string(seat)
    m = _PRIVATE_RE.search(info)
    return m.group(1) if m else ""


def _observe_deal_order(structure: TournamentStructure, dealer_seat: int,
                         stacks: list[int]) -> list[tuple[int, int]]:
    """Walk the initial chance nodes one at a time, return the actual
    (seat_idx, card_position) sequence."""
    gs = structure.to_inner_game_string_for_state(
        blind_level=structure.level(1),
        stacks=stacks,
        dealer_seat=dealer_seat,
    )
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    order: list[tuple[int, int]] = []
    while state.is_chance_node() and len(order) < NUM_SEATS * 2:
        before = [_private_for(state, p) for p in range(NUM_SEATS)]
        legal = state.legal_actions()
        if not legal:
            break
        state.apply_action(int(legal[0]))
        after = [_private_for(state, p) for p in range(NUM_SEATS)]
        for p in range(NUM_SEATS):
            if len(after[p]) > len(before[p]):
                order.append((p, (len(after[p]) // 2) - 1))
                break
        else:
            # Not a hole-card deal — likely a board card. Stop.
            break
    return order


def test_deal_order_matches_codified_sequence():
    """The actual OpenSpiel deal order at 6-max full table dealer=0 must
    match DEAL_ORDER_SEQUENCE. Locks Phase 2's forced-card dealer."""
    structure = TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml"
    )
    actual = _observe_deal_order(structure, dealer_seat=0, stacks=[1500] * 6)
    expected = list(DEAL_ORDER_SEQUENCE)
    assert actual == expected, (
        f"6-max deal order drifted from codified DEAL_ORDER_SEQUENCE.\n"
        f"  expected (codified): {expected}\n"
        f"  actual   (observed): {actual}\n"
        f"Phase 2 forced-card dealer in src/nlhe/integration/replay.py would "
        f"silently mis-deal hero hole cards. Investigate before any further "
        f"Phase 2 work — re-run scripts/probe_six_max_deal_order.py."
    )


def test_deal_order_invariant_across_dealer_position():
    """Deal order shouldn't depend on dealer_seat — universal_poker deals
    in absolute seat order (not relative to button). Probe at a different
    dealer to confirm."""
    structure = TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml"
    )
    actual_d0 = _observe_deal_order(structure, dealer_seat=0,
                                     stacks=[1500] * 6)
    actual_d3 = _observe_deal_order(structure, dealer_seat=3,
                                     stacks=[1500] * 6)
    assert actual_d0 == actual_d3, (
        f"Deal order should be dealer-independent, but differs: "
        f"d=0 -> {actual_d0}, d=3 -> {actual_d3}"
    )


def test_deal_order_sequence_has_12_slots():
    """Sanity: 6 seats × 2 cards = 12 hole-card chance actions before any
    decision node."""
    assert len(DEAL_ORDER_SEQUENCE) == 12
    seats = [s for s, _ in DEAL_ORDER_SEQUENCE]
    cards = [c for _, c in DEAL_ORDER_SEQUENCE]
    assert set(seats) == set(range(NUM_SEATS))
    assert set(cards) == {0, 1}
    # Each seat appears exactly twice
    from collections import Counter
    assert Counter(seats) == {i: 2 for i in range(NUM_SEATS)}
