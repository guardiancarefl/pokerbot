"""Unit tests for Piece 4: replay_to_decision (full mid-hand replay engine).

Hand-crafted ScraperFrames; assert that replay_to_decision lands at the
right state (current_player == hero, with hero's hole cards placed).
"""
from __future__ import annotations

import pytest

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.scraper_schema import BlindsLevel, ScraperFrame
from src.nlhe.integration.replay import (
    replay_to_decision, MidHandState, ReplayError,
    _private_cards_for, _cards_in_private,
)


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


# ---- PREFLOP scenarios ----

def test_replay_preflop_hero_utg_handstart():
    """Hand-start. Hero=UTG=seat 1 (with dealer=4). No prior actions."""
    bet = (25, 0, 0, 0, 0, 15)  # only blinds posted
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    frame = _frame(dealer_seat=4, hero_seat=1,
                   stack=stack, bet=bet,
                   pot_total=25 + 15 + 6*5)
    result = replay_to_decision(frame, _structure())
    assert isinstance(result, MidHandState)
    assert result.final_current_player == 1  # hero
    assert result.n_actions_applied == 0  # empty preflop sequence
    # Hero's hole cards should be at seat 1
    hero_priv = _cards_in_private(_private_cards_for(result.state, 1))
    assert sorted(hero_priv) == sorted(["Ah", "Ks"])


def test_replay_preflop_hero_bb_facing_raise():
    """Hero=BB=seat 0 (dealer=4). UTG raised to a min-raise-legal amount;
    everyone else folded.

    NOTE: OpenSpiel's inflated-BB convention forces min-raise = 2 *
    inflated_bb = 110 at level 1 (BB=25, ante=5, 6 alive: inflated_bb=55).
    Scraper raises BELOW 110 (e.g., a real-world 3xBB raise to 75) are
    REJECTED by OpenSpiel as illegal actions; replay fails with
    ReplayError on those frames. This is a known consequence of the
    Option B bug-match design (documented in DECISIONS.md): the inflated
    BB changes the betting-rule constraints too, not just the chip
    arithmetic. Corpus frames with sub-min-raise preflop raises will be
    soft-dropped — flag for evaluation against the full corpus.
    """
    bet = (25, 110, 0, 0, 0, 15)
    folded = (False, False, True, True, True, True)
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    frame = _frame(dealer_seat=4, hero_seat=0,
                   stack=stack, bet=bet, folded=folded,
                   pot_total=25 + 110 + 15 + 6*5)
    result = replay_to_decision(frame, _structure())
    assert result.final_current_player == 0  # hero=BB
    # 5 preflop actions: UTG raise, 3 folds, SB fold
    assert result.n_actions_applied == 5
    hero_priv = _cards_in_private(_private_cards_for(result.state, 0))
    assert sorted(hero_priv) == sorted(["Ah", "Ks"])
    # The hand is NOT terminal — hero has a decision
    assert not result.state.is_terminal()
    assert not result.state.is_chance_node()


def test_replay_preflop_limp_around_bb_option():
    """All 5 non-BB seats limp; hero=BB faces option."""
    bet = (25, 25, 25, 25, 25, 25)
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    frame = _frame(dealer_seat=4, hero_seat=0,
                   stack=stack, bet=bet,
                   pot_total=25*6 + 6*5)
    result = replay_to_decision(frame, _structure())
    assert result.final_current_player == 0
    assert result.n_actions_applied == 5  # 5 calls (UTG through SB)


def test_replay_preflop_shorthanded_5alive_with_caller():
    """5-handed (seat 1 empty). One non-hero seat CALLs preflop so the
    hand doesn't go terminal before hero's decision.

    Layout: alive=[0,2,3,4,5]. dealer=4. n_alive=5. dpos=alive.index(4)=3.
      SB = alive[(3+1)%5] = alive[4] = 5
      BB = alive[(3+2)%5] = alive[0] = 0
      UTG = alive[(3+3)%5] = alive[1] = 2
    Preflop order: [2, 3, 4(BTN), 5(SB), 0(BB)].

    Hero=BB=0. We want at least one caller so hero's option is reached.
    Have UTG (seat 2) call BB; everyone else fold."""
    bl1 = BlindsLevel(sb=15, bb=25, ante=5)
    empty = (False, True, False, False, False, False)
    folded = (False, False, False, True, True, True)  # MP/CO/BTN=seat3/4 + SB=5 folded
    # Wait: MP doesn't exist in 5-handed with this dealer. Action order is
    # [UTG=2, MP-equiv=3, CO-equiv=4=BTN, SB=5, BB=0]. We want UTG=2 to call;
    # the others before hero (3, 4=BTN, 5=SB) fold.
    bet = (25, 0, 25, 0, 0, 15)  # BB=25, UTG called=25, SB=15
    pre = 1500
    stack = (
        pre - 25 - 5,  # BB=hero
        0,              # empty
        pre - 25 - 5,  # UTG called
        pre - 5,        # folded (just ante)
        pre - 5,        # folded
        pre - 15 - 5,  # SB (folded but still has bet=15 chips out front? UI
                        # behaviour varies; for simplicity: SB folded, chips
                        # moved to pot, bet=0)
    )
    # actually correct: when SB folds preflop, the 15 chips go to pot, bet=0
    bet = (25, 0, 25, 0, 0, 0)
    stack = (
        pre - 25 - 5,
        0,
        pre - 25 - 5,
        pre - 5,
        pre - 5,
        pre - 15 - 5,
    )
    # pot: 25 (BB) + 25 (UTG call) + 15 (SB folded) + 5*ante = 70
    pot = 25 + 25 + 15 + 5 * 5
    frame = _frame(dealer_seat=4, hero_seat=0, stack=stack, bet=bet,
                   folded=folded, empty=empty, pot_total=pot, blinds=bl1)
    result = replay_to_decision(frame, _structure())
    assert result.final_current_player == 0  # hero=BB
    # Preflop sequence walks WITH-EMPTIES: starts at UTG=2 walking clockwise
    # by absolute seat. Order: [2, 3, 4, 5, 0]. Hero=0 last; emit 4 actions.
    # Seat 2: alive non-folded UTG, called BB (preflop_commit=25) -> CALL=1
    # Seat 3: folded -> 0
    # Seat 4: folded -> 0
    # Seat 5: folded (SB) -> 0
    # (Seat 1 empty is NOT in [2,3,4,5,0] because UTG=2 + 5 absolute offsets
    #  starting from 2 = [2, 3, 4, 5, 0]. Seat 1 is at offset 5 which would
    #  be the next lap; we only walk one lap.)
    assert result.n_actions_applied == 4


# ---- POSTFLOP ----

def test_replay_postflop_flop_check_to_hero_bb():
    """Preflop everyone limped at BB=25; flop dealt. SB checked.
    Hero=BB faces option to check or bet on flop.
    Hero=seat 0 (dealer=4 -> BB=alive[(4+2)%6]=alive[0]=0)."""
    pre = 1500
    bet = (0,) * 6  # nothing in front; flop just started
    stack = (pre - 25 - 5,) * 6  # everyone paid 25 preflop + 5 ante
    board = ("Jc", "Th", "3d")
    pot_total = 25 * 6 + 6 * 5  # 180
    frame = _frame(dealer_seat=4, hero_seat=0,
                   stack=stack, bet=bet, board=board,
                   pot_total=pot_total)
    result = replay_to_decision(frame, _structure())
    assert result.final_current_player == 0
    # 6 preflop actions (5 calls + hero's BB check option must be a CALL=1
    # in OpenSpiel since BB matches the bet) + 1 flop SB check = 7
    assert result.n_actions_applied == 7
    # Should be at a decision node (hero's flop turn)
    assert not result.state.is_chance_node()
    assert not result.state.is_terminal()
    # Board cards placed correctly (board is in information_state_string,
    # not observation_string)
    info = result.state.information_state_string(0)
    for card in board:
        assert card in info, (
            f"board card {card!r} not in info_state: {info}")


def test_replay_returns_replay_error_on_inconsistent_frame():
    """A frame that's structurally impossible: alive seats have chip totals
    that can't be reconstructed should produce a ReplayError."""
    # Wildly mismatched: hero's stack doesn't match any consistent history
    bet = (25, 0, 0, 0, 0, 15)
    # Stack inconsistent with pre_hand=1500 (off by huge amount)
    stack = (10, 0, 0, 0, 0, 0)  # seats 1..5 have stack=0 -> not alive
    # alive[0] only -> n_alive=1 -> ReplayError
    frame = _frame(dealer_seat=4, hero_seat=0, stack=stack, bet=bet,
                   pot_total=999)
    with pytest.raises(ReplayError):
        replay_to_decision(frame, _structure())


# ---- Class B forced-all-in fallback (live dryrun 2026-06-08 seq=228) ----

LV2 = BlindsLevel(sb=25, bb=50, ante=10)


def test_replay_forced_all_in_fallback_to_pre_hand():
    """Live dryrun seq=228 regression: when a seat is forced all-in for less
    than the min-raise increment, OpenSpiel's legal_actions collapse to
    {fold, call, pre_hand}; the voluntary-only chip_int from
    derive_action_sequence (= bet[seat]) is rejected. replay_to_decision
    catches this and retries with chip_int=stacks[seat] (= pre_hand) when
    the failing seat is going all-in.

    Constructed scenario (L2, ante=10): hero=seat 4 (mid-stack). Dealer=5
    → SB=seat 0, BB=seat 1, UTG=seat 2. UTG opens to 200; seat 3 (short
    stack pre_hand=200) is forced all-in for less since their full stack
    < min-raise (2×200=400). bet[3] = 200 - 10 = 190 visible. Action
    returns to hero.
    """
    pre_hand = (1500, 1500, 1500, 200, 1500, 1500)
    bet = (25, 50, 200, 190, 0, 0)
    stack = (1465, 1440, 1290, 0, 1490, 1490)
    pot_total = 10 * 6  # antes only; blinds/raises still in front
    frame = _frame(
        dealer_seat=5, hero_seat=4,
        stack=stack, bet=bet, board=(),
        pot_total=pot_total, hero_cards=("Ah", "Jh"),
        blinds=LV2,
    )
    # Without the replay fallback this raises ReplayError (chip_int=190
    # rejected, legal=[0,1,200]). With the fallback it lands at hero.
    pack = replay_to_decision(
        frame, _structure(), pre_hand_override=pre_hand)
    assert pack.final_current_player == 4
    assert pack.street_idx == 0
