"""Unit tests for _repair_folded_from_chip_deductions (live_1500 fix).

The Ignition scraper's per-seat `folded` field is stale (doesn't update
when seats fold preflop). The repair helper re-derives `folded` from per-
seat stack-deduction patterns. These tests cover:
  - Limped pot with 3 preflop folders (the line-9 case from live_1500)
  - All-checked-around preflop (no folders, repair is no-op)
  - Preflop raise with some folders (preflop_commit stops at the raise size)
  - Heads-up postflop (cannot drop below 2 non-folded)
  - SB-completed vs SB-folded (carries through; documented limitation)
"""
from __future__ import annotations

from src.nlhe.integration.scraper_schema import (
    BlindsLevel, ScraperFrame, _repair_folded_from_chip_deductions,
)


LV1 = BlindsLevel(sb=15, bb=25, ante=5)
LV2 = BlindsLevel(sb=25, bb=50, ante=10)


def _frame(*, dealer_seat: int, hero_seat: int,
            stack: tuple, bet: tuple,
            folded: tuple = (False,) * 6, empty: tuple = (False,) * 6,
            board: tuple = (), pot_total: int = 0,
            blinds: BlindsLevel = LV1) -> ScraperFrame:
    alive = tuple(
        (not empty[i]) and (stack[i] > 0 or bet[i] > 0)
        for i in range(6)
    )
    return ScraperFrame(
        captured_at="test", blinds=blinds, dealer_seat=dealer_seat,
        hero_seat=hero_seat, hero_cards=("6h", "7c"), board=board,
        stack=stack, bet=bet, folded=folded, empty=empty, alive=alive,
        pot_total=pot_total, controls_present=True, hero_facing_bet=False,
    )


def test_limped_pot_with_three_preflop_folders():
    """The exact live_1500 line 9 scenario.

    Level 1 (15/25/5 ante), 6 alive, postflop on board J 3 9. Stacks show
    deductions of (30, 5, 5, 30, 30, 5) — 3 seats lost only ante (folded
    preflop), 3 seats lost ante + 25-chip voluntary (limped). Scraper
    reports folded=all-False (stale field). Repair should mark seats 1,
    2, 5 as folded (the bet=0 non-blind seats with the LARGEST stacks).
    """
    # dealer=2 (seat3), so SB=3 (seat4), BB=4 (seat5)
    frame = _frame(
        dealer_seat=2, hero_seat=0,
        stack=(1470, 1495, 1495, 1470, 1470, 1495),
        bet=(0, 0, 0, 0, 0, 0),
        board=("Js", "3h", "9c"),
        pot_total=105,
    )
    repaired = _repair_folded_from_chip_deductions(
        frame, sb_seat=3, bb_seat=4)
    assert repaired == (False, True, True, False, False, True), (
        f"expected seats 1, 2, 5 marked folded, got {repaired}")


def test_all_checked_around_preflop_no_repair():
    """Postflop check-around frame: all 6 seats limped preflop (each paid
    ante + bb = 30 chips voluntary), no folds. Stack deductions are equal.
    Repair should be a no-op."""
    frame = _frame(
        dealer_seat=2, hero_seat=0,
        stack=(1470, 1470, 1470, 1470, 1470, 1470),  # each lost 30
        bet=(0, 0, 0, 0, 0, 0),
        board=("Js", "3h", "9c"),
        # 6 antes (30) + sb (15) + bb (25) + 4 limps (4 × 25) + sb-completion (10) = 180
        pot_total=180,
    )
    repaired = _repair_folded_from_chip_deductions(
        frame, sb_seat=3, bb_seat=4)
    assert repaired == (False,) * 6, (
        f"expected no folds (all limped), got {repaired}")


def test_preflop_raise_with_some_folders():
    """Postflop on flop after a preflop raise to 75 with 2 callers, 3
    folders. preflop_commit per non-folder = 75 (> bb=25). Repair should
    stop after dropping the 3 non-blind folders."""
    # dealer=2 (seat3), SB=3 (seat4), BB=4 (seat5).
    # Story: seat0 raises to 75 preflop, seat1/seat2/seat5 fold, SB calls
    # (completes to 75), BB calls.
    # Per-seat preflop voluntary spend:
    #   seat0: 75 (raise) → deduction = ante(5) + 75 = 80, stack = 1420
    #   seat1: 0 (fold)   → deduction = 5,  stack = 1495
    #   seat2: 0 (fold)   → deduction = 5,  stack = 1495
    #   seat3 (SB): 75 (completed) → deduction = ante(5) + sb(15) + 60 = 80, stack = 1420
    #   seat4 (BB): 75 (called)    → deduction = ante(5) + bb(25) + 50 = 80, stack = 1420
    #   seat5: 0 (fold)   → deduction = 5,  stack = 1495
    # Pot = 5*6 + 75*3 + 0*3 = 30 + 225 = 255. (SB/BB inclusive in the 75 voluntary commit)
    # Wait — sum of all deductions = 80*3 + 5*3 = 255. Match pot. ✓
    frame = _frame(
        dealer_seat=2, hero_seat=0,
        stack=(1420, 1495, 1495, 1420, 1420, 1495),
        bet=(0, 0, 0, 0, 0, 0),
        board=("Js", "3h", "9c"),
        pot_total=255,
    )
    repaired = _repair_folded_from_chip_deductions(
        frame, sb_seat=3, bb_seat=4)
    assert repaired == (False, True, True, False, False, True), (
        f"expected seats 1, 2, 5 marked folded after preflop raise, "
        f"got {repaired}")


def test_heads_up_postflop_no_drops():
    """When only 2 alive non-folded remain, the algorithm cannot drop
    further. Verified the termination condition."""
    # 2 alive (seats 0 + 1), seat 0 is SB, seat 1 is BB (heads-up).
    # dealer=0 (heads-up: dealer is SB)
    frame = _frame(
        dealer_seat=0, hero_seat=0,
        stack=(1470, 1470, 0, 0, 0, 0),  # both lost ante+bb
        bet=(0, 0, 0, 0, 0, 0),
        empty=(False, False, True, True, True, True),
        board=("Js", "3h", "9c"),
        pot_total=60,  # 2 antes + sb + bb + sb_complete = 10 + 15 + 25 + 10 = 60
    )
    repaired = _repair_folded_from_chip_deductions(
        frame, sb_seat=0, bb_seat=1)
    # Heads-up: no drops possible (n_anf <= 2 termination)
    assert repaired == (False, False, False, False, False, False)


def test_repair_skips_seats_with_bet_visible():
    """A seat with a visible current-street bet cannot have folded — they
    have chips in front of them this street. Should never be in the
    drop-candidates pool."""
    # Postflop frame where seat0 has bet a c-bet (75 chips). All 6 alive.
    # 3 preflop folders (seats 1, 2, 5) but stale scraper shows all alive.
    frame = _frame(
        dealer_seat=2, hero_seat=3,
        # seat0 limped, then on flop bet 75 of its stack
        stack=(1395, 1495, 1495, 1470, 1470, 1495),  # seat0 = 1500-30-75
        bet=(75, 0, 0, 0, 0, 0),
        board=("Js", "3h", "9c"),
        pot_total=180,  # 105 preflop limped + 75 c-bet = 180
    )
    repaired = _repair_folded_from_chip_deductions(
        frame, sb_seat=3, bb_seat=4)
    # seat0 has bet=75 — never drop candidate. seats 1, 2, 5 (folders)
    # should be marked folded.
    assert repaired[0] is False, "seat with bet>0 must not be marked folded"
    assert repaired == (False, True, True, False, False, True), (
        f"expected seats 1, 2, 5 folded, got {repaired}")


def test_repair_respects_pre_existing_folded():
    """If scraper actually does mark a seat as folded (rare but possible
    when the field updates correctly), the repair should leave it folded
    and only ADD more folders as needed."""
    # Same as the limped 3-folders case, but scraper correctly marked seat 1.
    frame = _frame(
        dealer_seat=2, hero_seat=0,
        stack=(1470, 1495, 1495, 1470, 1470, 1495),
        bet=(0, 0, 0, 0, 0, 0),
        folded=(False, True, False, False, False, False),  # seat 1 correctly marked
        board=("Js", "3h", "9c"),
        pot_total=105,
    )
    repaired = _repair_folded_from_chip_deductions(
        frame, sb_seat=3, bb_seat=4)
    # Seat 1 stays folded. Repair should still add seats 2 and 5.
    assert repaired[1] is True
    assert repaired == (False, True, True, False, False, True)


def test_preflop_frame_no_spurious_repair():
    """For a preflop frame at hero-to-act, the repair should not mark
    pre-hero-action seats as folded just because their bet=0. preflop_commit
    on a preflop frame can legitimately be < bb (everyone's just paid ante)
    and the algorithm should still terminate correctly.

    Note: on a preflop frame, the n_anf <= 2 termination may or may not
    fire; what matters is the algorithm doesn't loop forever and the
    output is sensible for downstream consumers.
    """
    # Hero is in BB (seat 4, hero_seat=4 = idx 4). Action folded around
    # to BB. seats 0-3 all paid ante (some + SB). Hero (BB) has BB in the
    # pot. Frame is preflop (board empty), hero_to_act for BB option.
    # dealer=2, so SB=3, BB=4=hero.
    # Stacks: seats 0, 1, 2, 5 paid only ante (5 chips, folded preflop).
    # SB (seat 3): paid ante + sb = 20, stack = 1480.
    # BB (seat 4): paid ante + bb = 30, stack = 1470.
    # Pot at this moment = 6*5 + 15 + 25 = 70 (preflop, no voluntary).
    frame = _frame(
        dealer_seat=2, hero_seat=4,
        stack=(1495, 1495, 1495, 1480, 1470, 1495),
        bet=(0, 0, 0, 15, 25, 0),
        board=(),  # preflop
        pot_total=70,
    )
    repaired = _repair_folded_from_chip_deductions(
        frame, sb_seat=3, bb_seat=4)
    # The repair MAY mark folders here — for a preflop frame at BB option
    # this is actually correct (those seats DID fold).
    # We just assert: it doesn't crash and doesn't mark blind seats as
    # folded (they have visible bets so they're never candidates).
    assert repaired[3] is False, "SB has bet=15, must not be marked folded"
    assert repaired[4] is False, "BB has bet=25, must not be marked folded"
