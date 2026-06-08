"""Regression tests for the seq=170 + seq=192 bridge fixes.

seq=170 (derive_action_sequence): a forced-blind BB sitting at exactly
bb_amount before any voluntary raise was being misclassified as a
"limper" by the defer-for-limper rule, causing early-position raisers
(UTG, MP) to defer their open and emit chip_int=1 (call) instead of
chip_int=raise_target. Live-dryrun reproduction: UTG opens 100, bridge
emitted seat 2 chip_int=1 and seat 4 chip_int=100 → contribution off
by 50 on UTG. Fix in scraper_schema.py: exclude blind seats sitting
at their forced post from the limper_after_me_unemitted set when
raise_above_bb is False.

seq=192 (openspiel_to_scraper_view matched_all_in branch): when a seat
goes all-in postflop via chip_int=pre_hand convention, the view used
preflop_max_chip_int as the subtractor — wrong for postflop since
preflop_max_chip_int equals the busted seat's full stack (= ante +
preflop_carry + current-street voluntary), over-subtracting current
voluntary to 0. Fix in invariant.py: postflop matched_all_in branch
subtracts (preflop_commit_per_alive + ante).
"""
from __future__ import annotations

import pyspiel
import pytest

from src.nlhe.actions import DiscreteAction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.invariant import openspiel_to_scraper_view
from src.nlhe.integration.replay import deal_one_card_6max, replay_to_decision
from src.nlhe.integration.scraper_schema import (
    BlindsLevel, ScraperFrame, derive_action_sequence,
)


# --------------------------------------------------------------------------
# seq=170 — limper_after_me_unemitted blind-seat exclusion
# --------------------------------------------------------------------------

def _make_seq170_frame():
    """Reconstruct seq=170: preflop, hero=SB(AhAs), dealer=seat5, UTG
    opens 100, MP folds (bet cleared), CO calls 100, BTN folds (bet
    cleared), action to hero (SB). Level 2: SB=25, BB=50, ante=10."""
    return ScraperFrame(
        hero_cards=("Ah", "As"),
        hero_facing_bet=True,
        board=(),
        pot_total=335,
        controls_present=True,
        hero_seat=0,
        dealer_seat=5,
        blinds=BlindsLevel(sb=25, bb=50, ante=10),
        alive=(True, True, True, True, True, True),
        folded=(False, False, False, False, False, False),
        stack=(1375, 1465, 1255, 1650, 965, 1955),
        bet=(25, 50, 100, 0, 100, 0),
        empty=(False, False, False, False, False, False),
        captured_at="seq170_synthetic",
    )


def test_seq170_utg_open_attribution_after_fix():
    """With the fix: UTG's open is emitted as chip_int=100 (raise to BB×2),
    not chip_int=1 (call)."""
    pre_hand_override = (1410, 1525, 1365, 1660, 1075, 1965)
    frame = _make_seq170_frame()
    actions = derive_action_sequence(
        frame, pre_hand_override=pre_hand_override)
    # Expected: UTG (seat 2) raises to 100, MP folds, CO calls, BTN folds.
    expected = [(2, 100), (3, 0), (4, 1), (5, 0)]
    assert actions == expected, (
        f"expected {expected}, got {actions}.\n"
        f"  Without fix: would emit (2, 1), (3, 0), (4, 100), (5, 0) — "
        f"UTG mis-attributed as caller, CO as raiser."
    )


def test_seq170_replay_invariant_passes():
    """End-to-end: replay the seq=170 frame through OpenSpiel, run the
    invariant — should now PASS (chip distribution matches scraper)."""
    from src.nlhe.integration.invariant import check_mid_hand_invariant
    structure = TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml")
    frame = _make_seq170_frame()
    pre = (1410, 1525, 1365, 1660, 1075, 1965)
    pack = replay_to_decision(frame, structure, pre_hand_override=pre)
    inv = check_mid_hand_invariant(frame, pack)
    assert inv.ok, f"invariant FAILED with deltas {inv.deltas}"


def test_blind_seat_exclusion_preserves_real_limper_behavior():
    """If a NON-blind seat genuinely limps (CO calls BB amount), the
    defer rule should still fire for an early-position raiser. The fix
    only excludes BB-at-bb_amount, not real limpers."""
    # Construct: UTG (seat 2) wants to raise to 100. CO (seat 4) limped
    # to 50 (= BB amount), and another later seat (seat 5 = BTN) has
    # target 150 (raise). UTG should DEFER to let CO limp first and BTN
    # raise. CO's target == bb_amount AND CO is NOT a blind seat, so
    # CO IS a real limper.
    pre = (1500, 1500, 1500, 1500, 1500, 1500)
    frame = ScraperFrame(
        hero_cards=("Ah", "As"),
        hero_facing_bet=True,
        board=(),
        pot_total=400,  # ignored by derive
        controls_present=True,
        hero_seat=0,
        dealer_seat=5,  # SB=0, BB=1, UTG=2, MP=3, CO=4, BTN=5
        blinds=BlindsLevel(sb=25, bb=50, ante=10),
        alive=(True, True, True, True, True, True),
        folded=(False, False, False, False, False, False),
        # SB sits at 25, BB at 50, UTG at 100, MP fold(0), CO limp 50, BTN raise 150
        stack=(1475, 1450, 1400, 1500, 1450, 1350),
        bet=(25, 50, 100, 0, 50, 150),
        empty=(False, False, False, False, False, False),
        captured_at="real_limper_synthetic",
    )
    actions = derive_action_sequence(frame, pre_hand_override=pre)
    # CO IS a real limper (target=50=bb but voluntary, NOT in blind seat).
    # The defer rule should still fire for UTG given a real limper +
    # BTN raise. So UTG defers (chip_int=1), then CO limps (1), then
    # BTN raises (150), then UTG re-decides.
    # The first emission for UTG should be chip_int=1 (defer/call), NOT 100.
    utg_first_emit = next((c for s, c in actions if s == 2), None)
    assert utg_first_emit == 1, (
        f"UTG should defer to real limper CO; first emit chip_int={utg_first_emit}"
    )


# --------------------------------------------------------------------------
# seq=192 — postflop matched_all_in branch in openspiel_to_scraper_view
# --------------------------------------------------------------------------

def test_seq192_postflop_matched_all_in_bet_recovered():
    """With the fix: a postflop all-in via chip_int=pre_hand correctly
    reports the current-street voluntary bet = 1015 (not 0)."""
    # Simulate the post-action OpenSpiel state at seq=192:
    #   hero (SB) bet 650 on turn → contrib 1150 (preflop_carry 500 + 650, ante absorbed)
    #   BB went all-in via chip_int=1525 (pre_hand) → contrib 1525, money 0
    #   others folded with ante posted
    parsed = {
        "contribution": [1150, 1525, 100, 10, 100, 10],
        "money":        [260, 0, 1265, 1650, 975, 1955],
        "current_player": 0,
        "street_idx": 2,  # turn
        "private_cards": "AhAs",
        "public_cards": "QdQsJc8c",
    }
    frame_alive = (True, True, True, True, True, True)
    folded = (False, False, True, True, True, True)
    still_in_hand = tuple(frame_alive[i] and not folded[i]
                           for i in range(6))
    # All seats voluntarily acted → all absorbed (except those that only
    # ante'd). For the test, set absorbed True for seats with contrib > ante.
    absorbed = (True, True, True, False, True, False)
    preflop_commit_per_alive = 500
    busted_mid_hand_exists = True  # BB went all-in via pre_hand convention
    preflop_max_chip_int = 1525    # max all-in chip_int

    view = openspiel_to_scraper_view(
        parsed,
        frame_alive=frame_alive,
        still_in_hand=still_in_hand,
        absorbed=absorbed,
        ante=10,
        preflop_commit_per_alive=preflop_commit_per_alive,
        busted_mid_hand_exists=busted_mid_hand_exists,
        preflop_max_chip_int=preflop_max_chip_int,
    )
    # Without fix: bet[BB] = max(0, 1525 - 1525) = 0
    # With fix:    bet[BB] = max(0, 1525 - 500 - 10) = 1015
    assert view["bet"][1] == 1015, (
        f"bet[BB] expected 1015 (current-street voluntary), got {view['bet'][1]}"
    )
    # Hero unchanged (standard absorbed branch)
    assert view["bet"][0] == 650


def test_seq192_preflop_matched_all_in_unchanged():
    """Sanity: PREFLOP matched_all_in branch is unchanged by the fix."""
    parsed = {
        "contribution": [500, 1525, 1525, 10, 10, 10],
        "money":        [1000, 0, 0, 1490, 1490, 1490],
        "current_player": 0,
        "street_idx": 0,
        "private_cards": "AhAs",
        "public_cards": "",
    }
    frame_alive = (True, True, True, True, True, True)
    folded = (False, False, False, True, True, True)
    still_in_hand = (True, True, True, False, False, False)
    absorbed = (True, True, True, False, False, False)
    view = openspiel_to_scraper_view(
        parsed,
        frame_alive=frame_alive,
        still_in_hand=still_in_hand,
        absorbed=absorbed,
        ante=10,
        preflop_commit_per_alive=0,  # preflop frame
        busted_mid_hand_exists=True,
        preflop_max_chip_int=1525,
    )
    # Preflop matched_all_in: bet = contrib - ante
    assert view["bet"][1] == 1525 - 10
    assert view["bet"][2] == 1525 - 10


def test_postflop_no_all_in_unchanged():
    """Sanity: ordinary postflop (no all-in) goes through the standard
    absorbed branch; behavior unchanged by the fix."""
    # Hero (SB) bet 200 on turn over BB call.
    parsed = {
        "contribution": [700, 500, 100, 10, 100, 10],
        "money":        [800, 1000, 1390, 1490, 1390, 1490],
        "current_player": 1,
        "street_idx": 2,
        "private_cards": "ThTd",
        "public_cards": "QdQsJc8c",
    }
    frame_alive = (True, True, True, True, True, True)
    still_in_hand = (True, True, False, False, False, False)
    absorbed = (True, True, True, False, True, False)
    view = openspiel_to_scraper_view(
        parsed,
        frame_alive=frame_alive,
        still_in_hand=still_in_hand,
        absorbed=absorbed,
        ante=10,
        preflop_commit_per_alive=500,
        busted_mid_hand_exists=False,   # no all-ins
        preflop_max_chip_int=0,
    )
    # Standard absorbed postflop: bet = contrib - preflop_commit_per_alive
    assert view["bet"][0] == 700 - 500   # hero turn bet 200
    assert view["bet"][1] == 500 - 500   # BB hasn't acted on turn = 0
