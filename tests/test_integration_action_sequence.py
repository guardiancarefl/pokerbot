"""Unit tests for derive_action_sequence (Phase 2 Piece 2).

Hand-crafted ScraperFrames covering the tricky cases the user specified:
  (a) preflop BB option (limp-around, hero=BB)
  (b) folded seat preflop (hero is somewhere after the folder)
  (c) facing-bet vs not-facing-bet
  (d) multi-street (preflop-heavy + simple postflop)

The tests assert specific (seat, chip_int) sequences. Where the expectation
depends on the canonical model's simplifying assumptions (e.g., all
prior-street commit attributed to preflop on postflop frames), the test
docstring calls that out.
"""
from __future__ import annotations

import pytest

from src.nlhe.integration.scraper_schema import (
    BlindsLevel,
    ScraperFrame,
    derive_action_sequence,
    preflop_action_order,
    postflop_action_order,
    _street_idx_from_board,
)


# Common blinds: Ignition Double-Up turbo level 1
LV1 = BlindsLevel(sb=15, bb=25, ante=5)


def _frame(*, dealer_seat: int, hero_seat: int,
            stack: tuple, bet: tuple,
            folded: tuple = (False,) * 6, empty: tuple = (False,) * 6,
            board: tuple = (), pot_total: int = 0,
            blinds: BlindsLevel = LV1) -> ScraperFrame:
    """Construct a ScraperFrame for testing.

    NOTE: alive is derived in real parse_frame as (not empty[i]) AND (stack
    OR bet > 0). Mirror that here.
    """
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
        hero_cards=("6h", "7c"),
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


# ---- preflop action order helper tests ----

def test_preflop_order_6max_dealer_4():
    """Sample-frame setup: dealer=seat5 (idx 4), 6-handed.
    Expected order: UTG=idx 1, MP=2, CO=3, BTN=4, SB=5, BB=0."""
    order = preflop_action_order(4, [0, 1, 2, 3, 4, 5])
    assert order == [1, 2, 3, 4, 5, 0]


def test_preflop_order_5handed_seat2_empty():
    """Line 208 setup: dealer=seat3 (idx 2), seat2 (idx 1) empty.
    alive = [0, 2, 3, 4, 5]. dpos=1 -> UTG=alive[4]=5,
    then 0, 2, 3 (=dealer/BTN), then SB=alive[2]=3? No wait —
    let me recompute by the formula: UTG=alive[(1+3)%5]=alive[4]=5,
    next alive[(1+4)%5]=alive[0]=0, next alive[(1+5)%5]=alive[1]=2,
    next alive[(1+6)%5]=alive[2]=3 (=dealer), next alive[(1+7)%5]=alive[3]=4."""
    order = preflop_action_order(2, [0, 2, 3, 4, 5])
    assert order == [5, 0, 2, 3, 4]


def test_postflop_order_6max_one_folded():
    """6-handed, dealer=4. SB=5, BB=0. If seat 2 folded postflop, the
    postflop order is SB->BB->UTG->MP(skip CO=folded)->BTN.
    SB=5, BB=0, UTG=1, MP=2(folded,skip), CO=3, BTN=4. Wait that's confusing.
    Let me just verify against the function: starting from SB clockwise."""
    folded = (False, False, True, False, False, False)  # seat 2 folded
    order = postflop_action_order(4, [0, 1, 2, 3, 4, 5], folded)
    # SB=alive[(4+1)%6]=alive[5]=5. Walk clockwise from there.
    # 5, 0, 1, 2(folded), 3, 4 -> skipping 2:
    assert order == [5, 0, 1, 3, 4]


def test_street_idx_from_board():
    assert _street_idx_from_board(()) == 0
    assert _street_idx_from_board(("Jc", "Th", "3d")) == 1
    assert _street_idx_from_board(("Jc", "Th", "3d", "2s")) == 2
    assert _street_idx_from_board(("Jc", "Th", "3d", "2s", "9c")) == 3


# ---- derive_action_sequence: preflop test cases ----

def test_case_a_preflop_bb_option_limp_around():
    """All seats limp (call BB=25). Hero is BB (seat 0 in the sample's
    dealer=4 setup). Hero faces their BB option (no raise above BB).
    Expected sequence: UTG, MP, CO, BTN, SB all CALL (chip_int=1).
    Hero (BB) is the next slot; we stop there."""
    # 6-handed, dealer=4 -> SB=5, BB=0. Hero=BB=0.
    # All non-blind seats called the BB; SB also completed; BB has option.
    # bet[i]=25 for all alive (all matched BB).
    # stacks: 1500 - bet - ante for each
    bet = (25, 25, 25, 25, 25, 25)
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    frame = _frame(dealer_seat=4, hero_seat=0,  # hero=BB
                   stack=stack, bet=bet, pot_total=180)
    actions = derive_action_sequence(frame)
    # Preflop order with dealer=4: [1, 2, 3, 4, 5, 0]. Hero=0 last; stop at 0.
    expected = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    assert actions == expected


def test_case_b_folded_seat_preflop():
    """Hero is CO (idx 3, with dealer=4). MP (idx 2) folded preflop.
    UTG limped, MP folded, hero next.
    Expected: [(UTG=1, call=1), (MP=2, fold=0)]."""
    bet = (0, 25, 0, 0, 0, 0)  # UTG=1 called; MP/CO/BTN/SB/BB no bets yet
    # MP folded, so folded[2]=True
    folded = (False, False, True, False, False, False)
    # SB=5, BB=0 — but bet[0]=0 and bet[5]=0 in this test, meaning blinds
    # not posted yet? That's inconsistent with preflop. Real preflop has
    # SB=15 and BB=25 even with no voluntary action. Fix:
    bet = (25, 25, 0, 0, 0, 15)  # BB=0 posts 25, UTG limped, SB=5 posts 15
    # MP folded
    folded = (False, False, True, False, False, False)
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    # alive: empty[i]=False for all, stack>0 for all -> all alive (incl MP folded)
    frame = _frame(dealer_seat=4, hero_seat=3,  # hero=CO=idx 3
                   stack=stack, bet=bet, folded=folded,
                   pot_total=25 + 25 + 15 + 6*5)
    actions = derive_action_sequence(frame)
    # Preflop order: [1=UTG, 2=MP, 3=CO=hero, 4, 5, 0]. Stop at hero.
    # UTG (seat 1) called: running_max=25, bet[1]=25 -> call (chip_int=1)
    # MP (seat 2) folded: chip_int=0
    expected = [(1, 1), (2, 0)]
    assert actions == expected


def test_case_c_facing_bet_preflop():
    """Hero is BB (idx 0). UTG raised to 75; everyone else folded.
    Hero faces a 75 to call (or fold).
    Expected: [(UTG=1, raise=75), (MP=2, fold=0), (CO=3, fold=0),
               (BTN=4, fold=0), (SB=5, fold=0)]."""
    bet = (25, 75, 0, 0, 0, 15)  # BB posted 25, UTG raised to 75, SB just-blind
    folded = (False, False, True, True, True, True)  # MP/CO/BTN/SB folded
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    frame = _frame(dealer_seat=4, hero_seat=0,  # hero=BB
                   stack=stack, bet=bet, folded=folded,
                   pot_total=25 + 75 + 15 + 6*5)
    actions = derive_action_sequence(frame)
    expected = [(1, 75), (2, 0), (3, 0), (4, 0), (5, 0)]
    assert actions == expected


def test_case_c_not_facing_bet_handstart():
    """Hand-start frame: nobody has voluntarily acted yet. Hero is UTG.
    Expected: empty action sequence (hero is FIRST in preflop order)."""
    bet = (25, 0, 0, 0, 0, 15)  # only blinds posted
    stack = tuple(1500 - bet[i] - 5 for i in range(6))
    frame = _frame(dealer_seat=4, hero_seat=1,  # hero=UTG
                   stack=stack, bet=bet, pot_total=25 + 15 + 6*5,
                   board=())
    actions = derive_action_sequence(frame)
    # Hero=UTG=1 is the FIRST in preflop order [1, 2, 3, 4, 5, 0]. Stop
    # immediately, no actions emitted.
    assert actions == []


# ---- Multi-street (simple postflop) test ----

def test_case_d_multistreet_postflop_check_to_hero():
    """Preflop limped (everyone called BB=25). Flop dealt. SB checked.
    Hero=BB faces the option to check or bet on flop.
    Expected: preflop calls + flop SB check. Hero (BB) next, stop."""
    # All alive paid 25 preflop (called BB). Now on flop. bet[] all 0.
    # Hero is BB=seat 0. Postflop order is SB=5, BB=0, UTG=1, MP=2, CO=3, BTN=4.
    # SB acts first postflop. We should see SB check before hero's slot.
    pre = 1500
    # After preflop (called 25), each seat has 1500 - 25 - 5 = 1470 (ante=5)
    stack = (1470,) * 6
    bet = (0,) * 6
    # board = flop (3 cards)
    board = ("Jc", "Th", "3d")
    pot_total = 25 * 6 + 6 * 5  # 6 seats × 25 chips + 6 antes = 180
    frame = _frame(dealer_seat=4, hero_seat=0,
                   stack=stack, bet=bet, board=board,
                   pot_total=pot_total)
    actions = derive_action_sequence(frame)
    # Preflop actions: order=[1,2,3,4,5,0], all called BB before hero.
    # Hero=0 is LAST in preflop order. So preflop emits 5 calls (chip_int=1)
    # for seats 1,2,3,4,5. Hero is BB at end of preflop -> BB's option.
    # But we're on FLOP. So hero ACTED preflop (checked the option). We need
    # to emit hero's BB-check too on preflop, then continue to flop.
    # However our derive_action_sequence STOPS at hero on preflop frames
    # only. For postflop frames, we walk through ALL preflop seats.
    # Actually re-reading the implementation: street_idx>0 case walks
    # preflop ENTIRELY without the hero-stop. Then walks postflop.
    # Let me trace: preflop loop walks [1,2,3,4,5,0]. street_idx=1 (flop)
    # so the hero-stop only triggers for street_idx==0. So:
    #   seat 1: bet=0 in preflop_commit calculation. Hmm.
    # WAIT — for postflop frame, preflop_commit = total_committed - bet -
    # ante. total_committed = pre_hand - stack = 1500 - 1470 = 30. ante=5.
    # bet=0 (current street). So preflop_commit = 30 - 0 - 5 = 25 for each
    # alive seat. running_max_pf starts at BB=25.
    # seat 1: preflop_commit=25 == running_max -> call (1).
    # seat 2: same. seat 3: same. seat 4: same. seat 5: same. seat 0 (BB):
    # bet[0]=25 in scraper? No — bet[0]=0 on flop. But preflop_commit for
    # seat 0 = 30 - 0 - 5 = 25. == running_max -> call (1).
    # So preflop actions: [(1,1), (2,1), (3,1), (4,1), (5,1), (0,1)].
    # Then intermediate streets (none between preflop and flop). Then
    # current street (flop). postflop_action_order from dealer=4, no folded:
    # SB=5, BB=0, UTG=1, MP=2, CO=3, BTN=4. Walk until hero=0.
    # seat 5 (SB): bet[5]=0 -> check (1). Stop at hero=0.
    expected = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (0, 1),
                 (5, 1)]
    assert actions == expected


def test_postflop_facing_bet_on_flop():
    """Preflop UTG opened to 75, everyone else folded except hero(BB) and SB
    called. Flop dealt. SB led for 50 on flop. Hero(BB) to act.
    Expected: preflop (UTG raise 75, MP/CO/BTN fold, SB call, BB call),
              flop (SB bet 125 = preflop_commit 75 + 50 = 125), hero next."""
    # 6-handed, dealer=4. SB=5, BB=0=hero. UTG=1 raised to 75. MP/CO/BTN folded.
    # SB called 75. BB called 75. Preflop matched at 75.
    # Flop: SB bet 50. Hero=BB to act, facing 50.
    folded = (False, False, True, True, True, False)
    # bet on flop: SB=5 has 50 in front, others 0
    bet = (0, 0, 0, 0, 0, 50)
    # stacks: paid 75 preflop + 5 ante; SB additionally 50 on flop.
    # everyone (alive) paid ante=5 + their preflop commit (75 for callers, 0 for folders)
    stack_default = 1500
    stack = (
        stack_default - 75 - 5,                # seat 0 (BB): 75 preflop + ante
        stack_default - 75 - 5,                # seat 1 (UTG): 75 raised + ante
        stack_default - 5,                     # seat 2 (MP folded, just ante)
        stack_default - 5,                     # seat 3 (CO folded)
        stack_default - 5,                     # seat 4 (BTN folded)
        stack_default - 75 - 5 - 50,           # seat 5 (SB): 75 preflop + 50 flop + ante
    )
    # pot: 75*3 (BB+UTG+SB) + 5*6 (antes) + 50 (SB flop bet) = 305
    pot_total = 75*3 + 5*6 + 50
    board = ("Jc", "Th", "3d")
    frame = _frame(dealer_seat=4, hero_seat=0,
                   stack=stack, bet=bet, board=board,
                   folded=folded, pot_total=pot_total)
    actions = derive_action_sequence(frame)
    # PREFLOP — order=[1,2,3,4,5,0]. preflop_commit derived from
    # total_committed - bet - ante:
    # seat 1: total=1500-1420=80, bet=0, ante=5 -> 75
    # seat 2: total=5, bet=0, ante=5 -> 0 (folded)
    # seat 3: total=5, bet=0, ante=5 -> 0 (folded)
    # seat 4: total=5, bet=0, ante=5 -> 0 (folded)
    # seat 5: total=1500-1370=130, bet=50, ante=5 -> 75
    # seat 0: total=1500-1420=80, bet=0, ante=5 -> 75
    # Walk preflop:
    # seat 1: alive, not folded, preflop_commit=75 > running_max(25) -> raise(75)
    # seat 2: folded -> 0
    # seat 3: folded -> 0
    # seat 4: folded -> 0
    # seat 5: alive, not folded, preflop_commit=75 == running_max(75) -> call(1)
    # seat 0: alive, not folded, preflop_commit=75 == running_max -> call(1)
    # POSTFLOP order (alive non-folded): SB=5, BB=0, UTG=1. Skip [2,3,4 folded].
    # No intermediate streets (preflop -> flop directly). Current street=flop.
    # Walk current street:
    # seat 5: bet=50 != running_max_cs(0) -> raise(preflop_commit + bet = 125)
    # seat 0 (hero): STOP.
    expected = [
        (1, 75), (2, 0), (3, 0), (4, 0), (5, 1), (0, 1),
        (5, 125),
    ]
    assert actions == expected
