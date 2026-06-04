"""Unit tests for Piece 5: check_mid_hand_invariant.

End-to-end: parse hand-crafted ScraperFrame -> replay_to_decision ->
check_mid_hand_invariant. Asserts the invariant PASSES on
self-consistent frames and REJECTS on synthetic mismatches (wrong hero
cards, wrong board, fake stack values).
"""
from __future__ import annotations

import pytest

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.scraper_schema import BlindsLevel, ScraperFrame
from src.nlhe.integration.replay import replay_to_decision
from src.nlhe.integration.invariant import check_mid_hand_invariant


LV1 = BlindsLevel(sb=15, bb=25, ante=5)


def _structure():
    return TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml")


def _frame(*, dealer_seat: int, hero_seat: int,
            stack: tuple, bet: tuple,
            folded: tuple = (False,) * 6, empty: tuple = (False,) * 6,
            board: tuple = (), pot_total: int = 0,
            hero_cards: tuple = ("Ah", "Ks"),
            blinds: BlindsLevel = LV1) -> ScraperFrame:
    alive = tuple(
        (not empty[i]) and (stack[i] > 0 or bet[i] > 0)
        for i in range(6)
    )
    max_opp_bet = max(
        (bet[i] for i in range(6) if i != hero_seat and alive[i]),
        default=0,
    )
    hero_facing_bet = alive[hero_seat] and max_opp_bet > bet[hero_seat]
    return ScraperFrame(
        captured_at="test",
        blinds=blinds,
        dealer_seat=dealer_seat,
        hero_seat=hero_seat,
        hero_cards=hero_cards,
        board=board,
        stack=stack,
        bet=bet,
        folded=folded,
        empty=empty,
        alive=alive,
        pot_total=pot_total,
        controls_present=True,
        hero_facing_bet=hero_facing_bet,
    )


def test_invariant_passes_preflop_hero_utg_handstart():
    """Trivial — hand-start, hero=UTG, no actions. Invariant should pass."""
    bet = (25, 0, 0, 0, 0, 15)
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    frame = _frame(dealer_seat=4, hero_seat=1, stack=stack, bet=bet,
                   pot_total=25 + 15 + 6*5)
    pack = replay_to_decision(frame, _structure())
    res = check_mid_hand_invariant(frame, pack)
    assert res.ok, f"invariant failed: {res.format_deltas()}"


def test_invariant_passes_preflop_limp_around_bb_option():
    """All limped; hero=BB has option. Invariant should pass."""
    bet = (25,) * 6
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    frame = _frame(dealer_seat=4, hero_seat=0, stack=stack, bet=bet,
                   pot_total=25*6 + 6*5)
    pack = replay_to_decision(frame, _structure())
    res = check_mid_hand_invariant(frame, pack)
    assert res.ok, f"invariant failed: {res.format_deltas()}"
    # Hero is NOT facing a bet (option to check or raise)
    assert not frame.hero_facing_bet


def test_invariant_passes_postflop_check_to_hero_bb():
    """Preflop limped; flop dealt; SB checked; hero=BB to act on flop."""
    pre = 1500
    bet = (0,) * 6
    stack = (pre - 25 - 5,) * 6
    board = ("Jc", "Th", "3d")
    pot_total = 25 * 6 + 6 * 5
    frame = _frame(dealer_seat=4, hero_seat=0,
                   stack=stack, bet=bet, board=board,
                   pot_total=pot_total)
    pack = replay_to_decision(frame, _structure())
    res = check_mid_hand_invariant(frame, pack)
    assert res.ok, f"invariant failed: {res.format_deltas()}"


def test_invariant_rejects_wrong_hero_cards():
    """If the scraper says hero has AhKs but we craft a frame claiming
    different cards, the post-replay state's private should still be AhKs
    (we forced those), but the SCRAPER frame's hero_cards field is what
    we diff against — mismatch -> reject."""
    bet = (25, 0, 0, 0, 0, 15)
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    # Real hero_cards passed for replay
    frame_real = _frame(dealer_seat=4, hero_seat=1, stack=stack, bet=bet,
                        hero_cards=("Ah", "Ks"),
                        pot_total=25 + 15 + 6*5)
    pack = replay_to_decision(frame_real, _structure())
    # Now construct an adversarial frame with WRONG hero_cards
    bad_frame = _frame(dealer_seat=4, hero_seat=1, stack=stack, bet=bet,
                       hero_cards=("2c", "3c"),  # wrong
                       pot_total=25 + 15 + 6*5)
    res = check_mid_hand_invariant(bad_frame, pack)
    assert not res.ok
    assert any(d[0] == "hero_cards" for d in res.deltas), (
        f"expected hero_cards delta, got: {res.deltas}")


def test_invariant_rejects_wrong_board():
    """Hand-start frame with empty board, but adversarial frame claims a
    board exists. State (no flop dealt) vs frame (with board) -> mismatch."""
    bet = (25, 0, 0, 0, 0, 15)
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    frame_real = _frame(dealer_seat=4, hero_seat=1, stack=stack, bet=bet,
                        pot_total=25 + 15 + 6*5, board=())
    pack = replay_to_decision(frame_real, _structure())
    bad_frame = _frame(dealer_seat=4, hero_seat=1, stack=stack, bet=bet,
                       pot_total=25 + 15 + 6*5,
                       board=("Jc", "Th", "3d"))  # claim flop
    res = check_mid_hand_invariant(bad_frame, pack)
    assert not res.ok
    # Both "board" and possibly "street_idx" deltas
    fields = [d[0] for d in res.deltas]
    assert "board" in fields or "street_idx" in fields, (
        f"expected board/street_idx delta, got: {fields}")


def test_invariant_rejects_wrong_pot():
    """Adversarial: scraper claims a pot total that doesn't match the
    reconstructed sum-of-contributions. (Per-seat stack/bet checks are
    intentionally dropped for mid-hand invariant due to the inflated-BB
    conversion configuration-dependence; pot total is still strict.)"""
    bet = (25, 0, 0, 0, 0, 15)
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    frame_real = _frame(dealer_seat=4, hero_seat=1, stack=stack, bet=bet,
                        pot_total=25 + 15 + 6*5)
    pack = replay_to_decision(frame_real, _structure())
    # Adversarial: claim pot total is way off
    bad_frame = _frame(dealer_seat=4, hero_seat=1, stack=stack, bet=bet,
                       pot_total=9999)
    res = check_mid_hand_invariant(bad_frame, pack)
    assert not res.ok
    fields = [d[0] for d in res.deltas]
    assert "pot" in fields, f"expected pot delta, got: {fields}"


def test_safe_action_is_fold_when_facing_bet_else_check():
    """On invariant failure, safe_action should be 'fold' if hero faces a
    bet, else 'check'."""
    # UTG hand-start — hero faces "the BB" which counts as facing a bet
    # (UTG can fold). So hero_facing_bet=True here. Safe action = "fold".
    bet = (25, 0, 0, 0, 0, 15)
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    frame_real = _frame(dealer_seat=4, hero_seat=1, stack=stack, bet=bet,
                        pot_total=25 + 15 + 6*5)
    pack = replay_to_decision(frame_real, _structure())
    # Adversarial pot mismatch -> rejected; safe_action depends on facing_bet
    bad_frame_facing = _frame(dealer_seat=4, hero_seat=1, stack=stack,
                               bet=bet, pot_total=9999)
    assert bad_frame_facing.hero_facing_bet  # UTG faces BB
    res = check_mid_hand_invariant(bad_frame_facing, pack)
    assert not res.ok
    assert res.safe_action == "fold"

    # Limp-around BB option scenario — hero=BB, not facing a bet
    bet_limped = (25, 25, 25, 25, 25, 25)
    stack_limped = tuple(1500 - bet_limped[i] - 5 for i in range(6))
    real_limped = _frame(dealer_seat=4, hero_seat=0,
                         stack=stack_limped, bet=bet_limped,
                         pot_total=25*6 + 6*5)
    pack_limped = replay_to_decision(real_limped, _structure())
    assert not real_limped.hero_facing_bet
    # Adversarial pot
    bad_limped = _frame(dealer_seat=4, hero_seat=0, stack=stack_limped,
                        bet=bet_limped, pot_total=9999)
    res2 = check_mid_hand_invariant(bad_limped, pack_limped)
    assert not res2.ok
    assert res2.safe_action == "check"
