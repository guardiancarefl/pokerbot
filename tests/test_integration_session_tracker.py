"""Regression tests for SessionTracker — focus: live-streaming robustness.

These pin down behaviour the 2026-06-08 live dryrun surfaced as broken:
  - seq=148 anchored a hand at dealer=1, alive=6
  - seq=156 was still in the SAME hand (still dealer=1, board still empty)
    but scraper-derived alive dropped 6→5 because one player had folded
    (Ignition mis-reads a folded-but-still-seated player as no-longer-alive
    on some frames)
  - the OLD hand-key included alive[] verbatim → pre_hand_for() returned
    None for every seq from 156 onward → simple-model fallback → remainder
    rejection → safe-fold the trips-of-Kings river spot at seq=175

The fix drops `alive[]` from the hand-key; chip conservation uses the
seats-alive-at-hand-start projection (= pre_hand[i] > 0) so a mid-hand
alive-shrink doesn't break the closure check either.
"""
from __future__ import annotations

import pytest

from src.nlhe.integration.scraper_schema import (
    BlindsLevel,
    ScraperFrame,
    SessionTracker,
)


LV2 = BlindsLevel(sb=25, bb=50, ante=10)


def _hand_start_frame(dealer_seat: int, alive_count: int = 6,
                       blinds: BlindsLevel = LV2,
                       pre_hand_each: int = 1500) -> ScraperFrame:
    """Construct a hand-start scraper frame: board empty, only blinds +
    antes posted, each alive seat at `pre_hand_each` chips pre-hand.

    Pre-hand stack `pre_hand_each` BEFORE blinds/antes are deducted.
    Stacks after posting: non-blind seats lose ante; SB seat loses sb+ante;
    BB seat loses bb+ante. Pot collects all of that.
    """
    sb, bb, ante = blinds.sb, blinds.bb, blinds.ante
    # SB = seat (dealer+1)%6 IF that seat is alive; BB = (dealer+2)%6 — for
    # this helper we keep the first `alive_count` consecutive seats alive
    # starting from seat 0, and place the dealer within them.
    alive = tuple(i < alive_count for i in range(6))
    assert alive[dealer_seat], "dealer must be alive"
    alive_seats = [i for i in range(6) if alive[i]]
    dpos = alive_seats.index(dealer_seat)
    sb_seat = alive_seats[(dpos + 1) % len(alive_seats)]
    bb_seat = alive_seats[(dpos + 2) % len(alive_seats)]

    stack = [0] * 6
    bet = [0] * 6
    for i in range(6):
        if not alive[i]:
            continue
        if i == sb_seat:
            stack[i] = pre_hand_each - ante - sb
            bet[i] = sb
        elif i == bb_seat:
            stack[i] = pre_hand_each - ante - bb
            bet[i] = bb
        else:
            stack[i] = pre_hand_each - ante
            bet[i] = 0
    # Pot collects only the antes pre-action (sb/bb are still in front of
    # seats as bet, until the postflop collection moves them).
    pot_total = ante * alive_count + sb + bb

    return ScraperFrame(
        captured_at="hand_start",
        blinds=blinds,
        dealer_seat=dealer_seat,
        hero_seat=0,
        hero_cards=("Ks", "8h"),
        board=(),
        stack=tuple(stack),
        bet=tuple(bet),
        folded=(False,) * 6,
        empty=tuple(not a for a in alive),
        alive=alive,
        pot_total=pot_total,
        controls_present=False,  # hand-start: action hasn't begun
        hero_facing_bet=bet[0] < max(bet),
    )


def _midhand_frame_after_one_fold(dealer_seat: int,
                                   scraper_mis_drops_alive_for: int) -> ScraperFrame:
    """Construct a mid-hand frame where ONE seat has folded and Ignition's
    scraper has mis-derived that seat as `alive=False` (empty=True, stack=0).
    Everyone else is still in the hand with the BB-call action having
    completed for the remaining seats.

    Specifically: 5 alive (folded seat now reads alive=False), board still
    empty (preflop continues), pot has absorbed the folder's posted ante
    plus the limped BB calls from non-folders.
    """
    blinds = LV2
    sb, bb, ante = blinds.sb, blinds.bb, blinds.ante
    # Original 6 alive (seats 0..5), dealer=1, SB=seat2, BB=seat3
    sb_seat, bb_seat = 2, 3
    pre_hand_each = 1500
    folded_seat = scraper_mis_drops_alive_for

    # All non-folded non-blind seats limped (paid bb) — they all match BB at 50.
    # Folded seat lost just its ante (= 10).
    # SB seat completed the BB option from sb to bb (+25 more).
    stack = [0] * 6
    for i in range(6):
        if i == folded_seat:
            # Scraper mis-reads this folded seat as alive=False:
            stack[i] = 0  # mis-zero'd; what the live log actually shows
            continue
        if i == sb_seat:
            # Limped: SB paid ante + bb total (= 60). Pre-hand 1500 → 1440.
            stack[i] = pre_hand_each - ante - bb
        elif i == bb_seat:
            # BB: paid ante + bb (= 60) at hand-start; no further commit
            # in a limped-around pot.
            stack[i] = pre_hand_each - ante - bb
        else:
            # Other limpers: paid ante + bb (= 60).
            stack[i] = pre_hand_each - ante - bb

    # All bets cleared (preflop done — bets folded into pot).
    bet = [0] * 6

    # Pot: ante from all 6 (60), plus bb from each of 5 non-folders (250).
    # = 60 + 250 = 310. Folder paid only its ante (already counted).
    pot_total = ante * 6 + bb * 5

    alive = tuple(i != folded_seat for i in range(6))

    return ScraperFrame(
        captured_at="midhand_after_fold",
        blinds=blinds,
        dealer_seat=dealer_seat,
        hero_seat=0,
        hero_cards=("Ks", "8h"),
        board=(),
        stack=tuple(stack),
        bet=tuple(bet),
        folded=(False,) * 6,  # Ignition's stale folded flag — repair handles
        empty=tuple(not a for a in alive),
        alive=alive,
        pot_total=pot_total,
        controls_present=False,
        hero_facing_bet=False,
    )


def test_anchor_survives_within_hand_alive_shrink():
    """Live regression seq=148→seq=156: hand_key must NOT include alive[].

    Hand-start observed at alive=6. Mid-hand the scraper drops one seat to
    alive=False (folded-but-still-seated mis-read). pre_hand_for must
    still return the anchor — the chip conservation check must use seats-
    alive-at-hand-start so the missing seat doesn't break the closure."""
    tracker = SessionTracker()
    start = _hand_start_frame(dealer_seat=1, alive_count=6)
    tracker.observe(start)
    assert tracker.pre_hand_for(start) is not None, \
        "hand-start frame must be self-recoverable"

    # Mid-hand: seat 5 folded; Ignition reports alive=False, stack=0 for it.
    midhand = _midhand_frame_after_one_fold(
        dealer_seat=1, scraper_mis_drops_alive_for=5)
    # Pre-fix: hand_key changed (alive 6→5) → returned None.
    # Post-fix: same hand_key (dealer+blinds), closure check uses alive-
    # at-hand-start so the missing seat's chips are accounted for by the
    # pot.
    pre_hand = tracker.pre_hand_for(midhand)
    assert pre_hand is not None, (
        "SessionTracker must keep the anchor through a within-hand alive "
        "shrink — Ignition's per-frame alive derivation is unstable on "
        "fold transitions but the hand itself is unchanged"
    )
    # Anchor is the original (alive=6, 1500-chip) pre-hand tuple.
    assert pre_hand == (1500,) * 6


def test_anchor_drops_when_dealer_moves_new_hand():
    """The fix only relaxes alive[] from the key — dealer change still
    indicates a new hand, so the prior anchor must be discarded."""
    tracker = SessionTracker()
    h1 = _hand_start_frame(dealer_seat=1, alive_count=6)
    tracker.observe(h1)
    # Next hand: dealer moves to seat 2 — same blinds, same alive.
    h2 = _hand_start_frame(dealer_seat=2, alive_count=6)
    # observe() re-anchors:
    tracker.observe(h2)
    pre = tracker.pre_hand_for(h2)
    assert pre is not None
    # The h1 anchor should NOT be served for h2 (different key now).
    assert tracker.pre_hand_for(h1) is None


def test_anchor_drops_when_blind_level_changes():
    """Level escalation = different key = new anchor required."""
    tracker = SessionTracker()
    h1 = _hand_start_frame(dealer_seat=1, alive_count=6, blinds=LV2)
    tracker.observe(h1)
    LV3 = BlindsLevel(sb=50, bb=100, ante=15)
    h_level_up = _hand_start_frame(
        dealer_seat=1, alive_count=6, blinds=LV3,
    )
    # Mid-hand frame with new blinds: anchor for LV2 must not satisfy LV3.
    assert tracker.pre_hand_for(h_level_up) is None


def test_corrected_pot_uses_alive_at_hand_start():
    """UI-lag pot correction must also tolerate within-hand alive shrink."""
    tracker = SessionTracker()
    start = _hand_start_frame(dealer_seat=1, alive_count=6)
    tracker.observe(start)
    # Mid-hand: alive shrank AND there's a visible bet from one seat (50
    # chips in front, not yet collected to the pot box). UI-lag pattern.
    blinds = LV2
    sb, bb, ante = blinds.sb, blinds.bb, blinds.ante
    # Same midhand setup but with a non-collected 50-chip bet from seat 4.
    stacks = list(_midhand_frame_after_one_fold(
        dealer_seat=1, scraper_mis_drops_alive_for=5).stack)
    bets = [0] * 6
    bets[4] = 50  # seat 4 just bet 50; pot UI hasn't picked it up yet
    stacks[4] -= 50  # the 50 came out of seat 4's stack
    alive = tuple(i != 5 for i in range(6))
    # Pot: original midhand pot was 310 (antes+limps). The 50-chip bet is
    # NOT yet in pot_total — UI-lag.
    frame = ScraperFrame(
        captured_at="ui_lag",
        blinds=blinds,
        dealer_seat=1, hero_seat=0,
        hero_cards=("Ks", "8h"), board=(),
        stack=tuple(stacks), bet=tuple(bets),
        folded=(False,) * 6,
        empty=tuple(not a for a in alive),
        alive=alive,
        pot_total=ante * 6 + bb * 5,  # the stale, pre-bet pot
        controls_present=False, hero_facing_bet=False,
    )
    corrected = tracker.corrected_pot_for(frame)
    assert corrected == ante * 6 + bb * 5 + 50, (
        "corrected pot should fold in the uncollected 50-chip bet"
    )
