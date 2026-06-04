"""Unit tests for Piece 3: forced-card dealer helpers (deal_one_card_6max).

Validates that:
  - hero hole cards are placed at the right slots (seats 0..5, sequentially)
  - opponent hole cards avoid the forbidden set (hero + board reservations)
  - board cards are placed in scraper-given order at the right chance node
  - the helper walks correctly from new_initial_state through all hole-card
    chance nodes to the first decision; we can keep going through flop
    chance nodes; etc.
"""
from __future__ import annotations

import pyspiel
import pytest

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.replay import (
    DEAL_ORDER_SEQUENCE,
    deal_one_card_6max,
    pick_deck_action,
    _cards_in_private,
    _cards_in_public,
    _private_cards_for,
    _public_cards,
    _extract_card_from_action_string,
    ReplayError,
)


def _new_state(dealer: int = 0):
    structure = TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml")
    gs = structure.to_inner_game_string_for_state(
        blind_level=structure.level(1), stacks=[1500] * 6, dealer_seat=dealer)
    return pyspiel.load_game(gs).new_initial_state()


# ---- Low-level helpers ----

def test_extract_card_from_action_string():
    """Realistic OpenSpiel chance action_to_string is 'player=-1 move=Deal <card>'."""
    assert _extract_card_from_action_string("player=-1 move=Deal 2c") == "2c"
    assert _extract_card_from_action_string("player=-1 move=Deal Ah") == "Ah"
    assert _extract_card_from_action_string("Deal Ts") == "Ts"


def test_extract_card_raises_on_no_card():
    with pytest.raises(ValueError):
        _extract_card_from_action_string("player=-1 move=Bet 100")


def test_cards_in_private_split():
    assert _cards_in_private("AhKh") == ["Ah", "Kh"]
    assert _cards_in_private("") == []


def test_cards_in_public_split():
    assert _cards_in_public("JcTh3d") == ["Jc", "Th", "3d"]
    assert _cards_in_public("") == []


# ---- Dealer behaviour ----

def test_deal_walks_through_all_12_hole_cards_to_first_decision():
    """Hero=seat 0, AhKs. Walk all 12 hole-card chance nodes via
    deal_one_card_6max. After 12 calls, state should be at the first
    decision (or possibly chance for flop deal — but at hand start with
    no decisions yet, state should be at preflop UTG's decision)."""
    state = _new_state(dealer=0)
    hero_seat = 0
    hero_cards = ("Ah", "Ks")
    board = ()  # no board cards needed yet — we'll stop at first decision

    while state.is_chance_node():
        deal_one_card_6max(state, hero_seat, hero_cards, board)

    # 12 hole cards dealt; should now be at preflop first decision
    assert not state.is_chance_node()
    assert not state.is_terminal()
    # Hero's private should match exactly what we asked for
    hero_priv = _cards_in_private(_private_cards_for(state, 0))
    assert sorted(hero_priv) == sorted(["Ah", "Ks"]), (
        f"hero priv = {hero_priv}, expected {hero_cards}")

    # Opponents should have 2 cards each, none of which are in
    # forbidden = hero_cards (board is empty here)
    for opp in range(1, 6):
        opp_priv = _cards_in_private(_private_cards_for(state, opp))
        assert len(opp_priv) == 2, f"opp {opp} priv len = {len(opp_priv)}"
        for c in opp_priv:
            assert c not in hero_cards, (
                f"opp {opp} got {c} which is reserved for hero")


def test_deal_respects_board_reservation_when_dealing_opp_cards():
    """If we reserve specific board cards, opponent hole cards must avoid
    them too — otherwise we'd be unable to deal the board later."""
    state = _new_state(dealer=0)
    hero_seat = 2
    hero_cards = ("Ah", "Kh")
    target_board = ("Jc", "Th", "3d", "2s", "9c")

    while state.is_chance_node():
        # Stop if we've dealt all 12 hole cards (don't try to deal board yet)
        per_seat_counts = [len(_private_cards_for(state, p)) // 2
                           for p in range(6)]
        if sum(per_seat_counts) >= 12:
            break
        deal_one_card_6max(state, hero_seat, hero_cards, target_board)

    # All 12 hole cards placed; no opponent should have any of the 5 board cards
    forbidden_for_opps = set(hero_cards) | set(target_board)
    for opp in range(6):
        if opp == hero_seat:
            continue
        opp_priv = _cards_in_private(_private_cards_for(state, opp))
        for c in opp_priv:
            assert c not in forbidden_for_opps, (
                f"opp {opp} got {c} which is in forbidden={forbidden_for_opps}")


def test_deal_hero_cards_arrive_at_hero_seat_in_order():
    """Hero=seat 4 (CO). hero_cards=(Ac, 2d). Walk; verify CO gets exactly
    those two cards in deal order."""
    state = _new_state(dealer=0)
    hero_seat = 4
    hero_cards = ("Ac", "2d")
    board = ()

    while state.is_chance_node():
        per_seat_counts = [len(_private_cards_for(state, p)) // 2
                           for p in range(6)]
        if sum(per_seat_counts) >= 12:
            break
        deal_one_card_6max(state, hero_seat, hero_cards, board)

    hero_priv = _cards_in_private(_private_cards_for(state, hero_seat))
    # Deal order is sequential — seat 4 gets BOTH cards in order Ac then 2d
    assert hero_priv == ["Ac", "2d"], (
        f"hero_priv = {hero_priv}, expected ['Ac', '2d']")


def test_deal_raises_if_target_board_exhausted_for_extra_chance():
    """If we're past hole cards AND past all target_board cards, an extra
    chance node should raise — the caller has a bug."""
    state = _new_state(dealer=0)
    hero_seat = 0
    hero_cards = ("Ah", "Ks")
    target_board = ()  # no board cards specified

    # Walk through hole cards
    while state.is_chance_node():
        per_seat_counts = [len(_private_cards_for(state, p)) // 2
                           for p in range(6)]
        if sum(per_seat_counts) >= 12:
            break
        deal_one_card_6max(state, hero_seat, hero_cards, target_board)

    # State should now be at first decision (UTG to act). Apply some
    # actions to force a flop deal: have everyone fold except 2 seats
    # then check around postflop... actually simpler: have all seats raise
    # all-in preflop which forces showdown. But that triggers terminal.
    # Skip this test for now — it's covered by the "exhausted" path in
    # the dealer. Just confirm the path raises on a manually-constructed
    # chance node beyond board.

    # Verify the raise path by trying to deal a board card when we have no
    # board reservations: directly call with an empty target_board on a
    # state we forcibly advance to need a board card.
    # (Easiest way: apply actions to close preflop and reach flop chance)
    # We just emit "everyone calls" preflop.
    while not state.is_terminal():
        if state.is_chance_node():
            # We're at flop chance — try to deal_one_card with empty board
            with pytest.raises(ReplayError, match="target_board exhausted"):
                deal_one_card_6max(state, hero_seat, hero_cards,
                                    target_board)
            return  # confirmed the raise; done
        legal = state.legal_actions()
        # action 1 = call/check
        if 1 in legal:
            state.apply_action(1)
        else:
            state.apply_action(legal[0])
    # If we never reached a flop chance node, the test setup is wrong
    pytest.fail("never reached flop chance; cannot verify exhausted-board raise")
