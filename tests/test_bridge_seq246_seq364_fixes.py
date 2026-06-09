"""Regression tests for the seq=246 + seq=364 bridge fixes (2026-06-09).

Fixes covered:

Predicate 1 — dead-SB detection (scraper_schema._detect_blinds_from_bb_post):
  Ignition's forward-moving button can leave the (dealer+1) % NUM_SEATS
  seat empty when the player who would-have-been-SB just busted between
  hands. Dead SB → no SB posted, BB walks forward to the next alive seat.
  Detected from posted-blind evidence in frame.bet (NOT from session
  history, which would desync silently on missed-hand frames).

Predicate 2 — defer-for-smaller-raise discriminator (scraper_schema:1554):
  Only defer to a smaller-pending-raise seat whose pre[s]-ante >= my t
  (= they actually have chips to match my raise). All-in-for-less seats
  with pre-ante < my t can't raise past me; deferring to them caused
  the raising seat to call instead of raise (chip-arithmetic broken).

Surgical call-for-less discriminator (scraper_schema:1531):
  When a seat's target < running_max AND they've committed every chip
  they had (pre-ante == t, stack == 0), emit chip_int=1 (call all-in
  for less) instead of chip_int=0 (fold). OpenSpiel caps at remaining
  stack so the committed chips land in the pot.

Single-source-of-truth threading (game_strings.to_inner_game_string_for_state):
  Accepts sb_seat/bb_seat parameters resolved upstream by Predicate 1.
  game_strings does NOT recompute SB/BB when these are passed in —
  recomputing would re-introduce the dead-SB misclassification at a
  second site (the seq=55-class regression).
"""
from __future__ import annotations

import pyspiel
import pytest

from src.nlhe.game_strings import TournamentStructure, BlindLevel
from src.nlhe.integration.invariant import (
    check_mid_hand_invariant, openspiel_to_scraper_view,
)
from src.nlhe.integration.replay import replay_to_decision
from src.nlhe.integration.scraper_schema import (
    BlindsLevel, ScraperFrame, derive_action_sequence,
    _derive_blinds_and_action_order, _detect_blinds_from_bb_post,
)


STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _frame(*, dealer_seat, hero_seat, hero_cards, stack, bet,
            blinds, board=(), folded=None, empty=None,
            pot_total=None, controls_present=True, captured_at="test"):
    if folded is None:
        folded = (False,) * 6
    if empty is None:
        empty = tuple(stack[i] == 0 and bet[i] == 0 for i in range(6))
    alive = tuple(
        (not empty[i]) and (stack[i] > 0 or bet[i] > 0)
        for i in range(6)
    )
    if pot_total is None:
        pot_total = sum(bet) + sum(blinds.ante for _ in alive if _)
    max_opp_bet = max(
        (bet[i] for i in range(6) if i != hero_seat and alive[i]),
        default=0,
    )
    hero_facing_bet = alive[hero_seat] and max_opp_bet > bet[hero_seat]
    return ScraperFrame(
        captured_at=captured_at, blinds=blinds,
        dealer_seat=dealer_seat, hero_seat=hero_seat,
        hero_cards=hero_cards, board=board,
        stack=stack, bet=bet, folded=folded, empty=empty, alive=alive,
        pot_total=pot_total, controls_present=controls_present,
        hero_facing_bet=hero_facing_bet,
    )


# --------------------------------------------------------------------------
# seq=364 — dead-SB end-to-end
# --------------------------------------------------------------------------

LV4 = BlindsLevel(sb=75, bb=150, ante=25)


def _make_seq364_frame():
    """Reconstruct seq=364 (live_dryrun_20260609_154557.jsonl):

    Dealer=seat5 (idx 4). Seat 4 (idx 3) and seat 6 (idx 5) are EMPTY
    (idx 5 just busted between seq=350 and seq=352 → forward-moving
    button → idx 5 is the natural-SB-this-hand → DEAD SB).

    Hero (idx 0 = seat 1) posts BB=150 (no SB posted by anyone).
    Seat 3 (idx 2, MP under dead-SB rotation) opens to 300.
    Hero faces 150 to call from a single 300 raise.
    """
    return ScraperFrame(
        hero_cards=("7c", "Kd"),
        hero_facing_bet=True,
        board=(),
        pot_total=550,
        controls_present=True,
        hero_seat=0,
        dealer_seat=4,
        blinds=LV4,
        alive=(True, True, True, False, True, False),
        folded=(False, False, False, False, False, False),
        stack=(1571, 2275, 1453, 0, 3151, 0),
        bet=(150, 0, 300, 0, 0, 0),
        empty=(False, False, False, True, False, True),
        captured_at="seq364_synthetic",
    )


def test_seq364_predicate1_detects_dead_sb():
    """Predicate 1 reads the frame's posted-blind evidence and concludes
    SB is dead (sb_seat=None), BB=hero(idx 0)."""
    frame = _make_seq364_frame()
    alive_seats = [i for i in range(6) if frame.alive[i]]
    sb_det, bb_det = _detect_blinds_from_bb_post(
        frame, frame.dealer_seat, alive_seats)
    assert sb_det is None, f"expected dead-SB; predicate returned sb={sb_det}"
    assert bb_det == 0, f"expected bb=hero(0); predicate returned bb={bb_det}"


def test_seq364_centralizer_returns_dead_sb_and_correct_pf_order():
    """The centralizer routes Predicate 1's resolution to action-order:
    UTG = first alive after BB(hero=0) = seat 2 (idx 1)."""
    frame = _make_seq364_frame()
    alive_seats = [i for i in range(6) if frame.alive[i]]
    sb_seat, bb_seat, pf_alive, pf_empties, postflop_first = (
        _derive_blinds_and_action_order(
            frame.dealer_seat, alive_seats, frame=frame))
    assert sb_seat is None
    assert bb_seat == 0
    # Preflop order under dead-SB: UTG=1, MP=2, BTN=4, BB=0
    assert pf_alive == [1, 2, 4, 0], (
        f"expected [1,2,4,0]; got {pf_alive}")
    # Postflop first-to-act = BB (since SB is dead)
    assert postflop_first == 0


def test_seq364_end_to_end_reconstructs():
    """Full bridge: derive_action_sequence + replay_to_decision both
    use Predicate 1 via the SAME centralizer. OpenSpiel game string
    has hero=BB (blind=150), no SB blind, and the action sequence walks
    to hero's decision without state mismatch."""
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    frame = _make_seq364_frame()
    # No pre_hand_override available for this frame (no session
    # observation before it). replay_to_decision uses the simple-model
    # derivation path internally.
    pack = replay_to_decision(frame, structure, pre_hand_override=None)
    # Invariant must pass: scraper view and OpenSpiel view agree on
    # per-seat chip distribution AND pot total.
    inv = check_mid_hand_invariant(frame, pack)
    assert inv.ok, f"invariant FAILED for seq=364 dead-SB; deltas={inv.deltas}"


def test_seq364_single_source_of_truth_game_string_consumes_resolved_sb():
    """Single-source-of-truth guard: scraper_schema._derive_blinds_and_action_order
    returns sb_seat=None for seq=364. The game string built via
    to_inner_game_string_for_state with that sb_seat threaded in must
    encode blind[hero]=150 (BB) and blind[everyone-else]=0 (no SB).
    game_strings must NOT recompute SB at next-alive-after-dealer
    (which would put SB=hero, blind[hero]=75)."""
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    frame = _make_seq364_frame()
    alive_seats = [i for i in range(6) if frame.alive[i]]
    sb_seat, bb_seat, *_ = _derive_blinds_and_action_order(
        frame.dealer_seat, alive_seats, frame=frame)
    # Pre-hand stacks for the game string (frame.stack + ante + bet for
    # alive seats; busted/empty are 0).
    pre = [0] * 6
    for i in alive_seats:
        if i == bb_seat:
            pre[i] = frame.stack[i] + frame.blinds.ante + frame.blinds.bb
        else:
            pre[i] = frame.stack[i] + frame.blinds.ante + frame.bet[i]
    bl = BlindLevel(level=1, small_blind=frame.blinds.sb,
                     big_blind=frame.blinds.bb, ante=frame.blinds.ante)
    game_str = structure.to_inner_game_string_for_state(
        blind_level=bl, stacks=pre, dealer_seat=frame.dealer_seat,
        sb_seat=sb_seat, bb_seat=bb_seat,
    )
    # Parse the blind= section. Order matches seat order 0..5.
    blind_section = [
        part for part in game_str.split(",") if part.startswith("blind=")
    ][0]
    blind_values = [int(x) for x in blind_section[len("blind="):].split()]
    assert blind_values[0] == frame.blinds.bb, (
        f"hero(0) must post BB=150 under dead-SB; got blind[0]={blind_values[0]}. "
        f"If 75, game_strings recomputed SB independently — single-source-of-truth "
        f"violation. game_str: {game_str[:300]}"
    )
    # No other seat may post SB (= no seat has blind == sb_amount).
    sb_amt = frame.blinds.sb
    for i in range(6):
        assert blind_values[i] != sb_amt, (
            f"no seat may post SB under dead-SB; seat {i} has blind={blind_values[i]}=sb. "
            f"game_str: {game_str[:300]}"
        )


# --------------------------------------------------------------------------
# seq=55 — live-SB stays unchanged (regression guard for Predicate 1's
# common-case behavior + centralizer's H1 fallback)
# --------------------------------------------------------------------------

LV1 = BlindsLevel(sb=15, bb=25, ante=5)  # level 1 of the structure
LV2 = BlindsLevel(sb=25, bb=50, ante=10)  # seq=55 was at level 2


def _make_seq55_frame():
    """seq=55 from same live_dryrun: dealer=seat3 (idx 2), seat 4 (idx 3)
    empty (long-busted, blinds stabilized past it). LIVE SB.
    SB=seat5 (idx 4)=25, BB=seat6 (idx 5)=50, no other action yet."""
    return ScraperFrame(
        hero_cards=("6h", "7c"),
        hero_facing_bet=False,
        board=(),
        pot_total=125,
        controls_present=True,
        hero_seat=0,
        dealer_seat=2,
        blinds=LV2,
        alive=(True, True, True, False, True, True),
        folded=(False, False, False, False, False, False),
        stack=(1705, 2965, 1793, 0, 1272, 1140),
        bet=(0, 0, 0, 0, 25, 50),
        empty=(False, False, False, True, False, False),
        captured_at="seq55_synthetic",
    )


def test_seq55_stays_live_sb_bit_identical():
    """seq=55 has an empty seat (idx 3) but it's NOT the natural-SB-this-
    hand — the blinds have rotated past it. Predicate 1 must detect the
    live SB (idx 4 with bet=25) and confirm SB=4, BB=5. This must match
    the H1 default exactly (bit-identical common-case behavior)."""
    frame = _make_seq55_frame()
    alive_seats = [i for i in range(6) if frame.alive[i]]
    # H1 default (without Predicate 1): SB=alive_seats[(dpos+1)%n_alive],
    # BB=alive_seats[(dpos+2)%n_alive].
    dpos = alive_seats.index(frame.dealer_seat)
    sb_h1 = alive_seats[(dpos + 1) % len(alive_seats)]
    bb_h1 = alive_seats[(dpos + 2) % len(alive_seats)]
    # Centralizer with frame: Predicate 1 applies.
    sb_seat, bb_seat, *_ = _derive_blinds_and_action_order(
        frame.dealer_seat, alive_seats, frame=frame)
    assert sb_seat == sb_h1 == 4, (
        f"seq=55 is LIVE SB; expected sb=4 from both H1 and Predicate 1; "
        f"got sb_seat={sb_seat}, sb_h1={sb_h1}"
    )
    assert bb_seat == bb_h1 == 5


# --------------------------------------------------------------------------
# seq=246 — Predicate 2 defer + call-for-less discriminator together
# --------------------------------------------------------------------------

LV3 = BlindsLevel(sb=50, bb=100, ante=15)


def _make_seq246_frame():
    """Reconstruct seq=246: hero is BB (idx 0), seat 4 (idx 3) empty.
    BTN (seat5, idx 4) all-in for 2166. SB (seat6, idx 5) all-in for less
    at 615. Hero faces ALLIN/FOLD.

    pre_hand_override is required (session tracker provides per-seat
    hand-start chip counts; without it, simple-model derivation can't
    recover the all-in-for-less seat 5's pre-hand)."""
    return ScraperFrame(
        hero_cards=("Qh", "Ts"),
        hero_facing_bet=True,
        board=(),
        pot_total=2956,
        controls_present=True,
        hero_seat=0,
        dealer_seat=4,
        blinds=LV3,
        alive=(True, True, True, False, True, True),
        folded=(False, False, False, False, False, False),
        stack=(1236, 2765, 2043, 0, 0, 0),
        bet=(100, 0, 0, 0, 2166, 615),
        empty=(False, False, False, True, False, False),
        captured_at="seq246_synthetic",
    )


def test_seq246_reconstructs_with_defer_and_call_for_less_discriminators():
    """seq=246 needs BOTH Predicate 2 (BTN must not defer to SB's
    all-in-for-less — pre[SB]-ante=615 < BTN t=2166) AND the call-for-
    less discriminator (SB target=615 < running_max=2166 → emit
    chip_int=1, NOT chip_int=0 — to capture SB's committed chips in
    the pot)."""
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    frame = _make_seq246_frame()
    # Session-tracker-equivalent override: pre-hand stacks for all alive-
    # at-start seats (including the now-busted SB seat 5).
    pre = (1351, 2780, 2058, 0, 2181, 630)
    actions = derive_action_sequence(frame, pre_hand_override=pre)
    # Action 4 (BTN, idx 4): must RAISE chip_int=2166 (the all-in bet),
    # NOT call chip_int=1 (which would happen under the buggy defer).
    btn_emit = next((c for s, c in actions if s == 4), None)
    assert btn_emit == 2166, (
        f"BTN must raise to 2166 (NOT defer to SB's all-in-for-less); "
        f"got chip_int={btn_emit}. actions={actions}"
    )
    # Seat 5 (SB, busted-mid-hand all-in for less): must emit chip_int=1
    # (call all-in for less), NOT chip_int=0 (which would lose their
    # committed 615 chips from the pot).
    sb_emit = next((c for s, c in actions if s == 5), None)
    assert sb_emit == 1, (
        f"SB (busted mid-hand, all-in for less below BTN) must emit "
        f"chip_int=1 (call); got chip_int={sb_emit}. actions={actions}"
    )
    # Full end-to-end: replay through OpenSpiel + invariant.
    pack = replay_to_decision(frame, structure, pre_hand_override=pre)
    inv = check_mid_hand_invariant(frame, pack)
    assert inv.ok, f"invariant FAILED for seq=246; deltas={inv.deltas}"


# --------------------------------------------------------------------------
# Normal full-ring (no busts, no dead-SB): bit-identical to pre-change
# behavior across all derivations
# --------------------------------------------------------------------------

def test_normal_full_ring_bit_identical():
    """6-handed all-alive hand-start frame with standard live-SB rotation.
    Predicate 1 confirms (matches H1). All centralizer outputs match
    what the H1 default would produce — guards against the centralization
    shifting the common case across call sites.
    """
    # Dealer=5, blinds posted, no actions. SB=0, BB=1, UTG=2, ..., BTN=5.
    frame = ScraperFrame(
        hero_cards=("Ah", "As"),
        hero_facing_bet=False,
        board=(),
        pot_total=15 + 25 + 5 * 10,  # SB + BB + 5 antes (BTN didn't ante? actually all 6)
        controls_present=True,
        hero_seat=2,
        dealer_seat=5,
        blinds=LV1,
        alive=(True, True, True, True, True, True),
        folded=(False, False, False, False, False, False),
        stack=(1480, 1465, 1490, 1490, 1490, 1490),
        bet=(15, 25, 0, 0, 0, 0),
        empty=(False,) * 6,
        captured_at="normal_full_ring",
    )
    alive_seats = [0, 1, 2, 3, 4, 5]
    # Predicate 1 detection — should match H1 exactly.
    sb_det, bb_det = _detect_blinds_from_bb_post(
        frame, frame.dealer_seat, alive_seats)
    assert sb_det == 0 and bb_det == 1, (
        f"normal full-ring must detect sb=0, bb=1; got sb={sb_det}, bb={bb_det}"
    )
    # Centralizer with and without frame: must agree (frame-driven
    # path returns same as H1 default for the common case).
    sb_a, bb_a, pf_alive_a, pf_full_a, post_a = (
        _derive_blinds_and_action_order(
            frame.dealer_seat, alive_seats, frame=frame))
    sb_b, bb_b, pf_alive_b, pf_full_b, post_b = (
        _derive_blinds_and_action_order(
            frame.dealer_seat, alive_seats))  # no frame
    assert sb_a == sb_b == 0
    assert bb_a == bb_b == 1
    assert pf_alive_a == pf_alive_b == [2, 3, 4, 5, 0, 1]
    assert pf_full_a == pf_full_b == [2, 3, 4, 5, 0, 1]
    assert post_a == post_b == 0
