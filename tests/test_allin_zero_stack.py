"""F2 — zero-stack all-in handling (--allin-zero-stack, task #11,
operator-routed 2026-06-13). Parse-level tests with synthetic fixtures
derived from the session-5 pointed-seat kill set
(logs/live_dryrun_20260612_230149.jsonl; postmortem
evals/session5_postmortem_20260613/POSTMORTEM.txt).

None-vs-0 documentation (load-bearing for these fixtures): the
session-5 raw logs mostly show the pointed seat's stack as None because
the Windows ocr_int truthiness bug (F1, shipped Windows-side) dropped
consensus ZEROS — an all-in seat genuinely displays "0". F2 is built
for the POST-F1 world where the literal 0 arrives at the bridge:
  - explicit 0 + occupied  -> all-in candidate (committed test decides)
  - None + empty           -> busted/vacated seat (D2 dead-button class)
The fixtures below therefore patch the historical frames to their
post-F1 shape (0 present, empty=False) where the session-5 ground truth
says the seat was all-in, and keep the uncommitted dead-button frames
as the post-F1 bust shape to prove the D2 routing is preserved.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.nlhe.integration.scraper_schema import (
    parse_frame, ScraperDataQuality, NUM_SEATS,
)


# ── fixtures (derived from session-5 frames; values verbatim) ──────────

def seq472_8s5c():
    """Session-5 seq 472 VERBATIM (hand 8s5c, postmortem pin 2): dealer
    = seat4, which shoved all-in on the flop (stack 2780 at seq 470/471,
    explicit 0 at 472 — this frame already carries the post-F1 shape).
    Pot 628 with main 290 (side pot = all-in). Killed pre-F2 as
    'dealer points to seat4 but that seat is empty/non-alive'."""
    return {
        "captured_at": "20260612_192715_000",
        "blinds": "% 15/25, 5 Ante No Limit Hold'em - TBL#1",
        "dealer": "seat4",
        "hero_cards": ["8s", "5c"],
        "board": ["Jd", "8h", "8d"],
        "stacks": {"seat1": 1042, "seat2": 1380, "seat3": None,
                   "seat4": 0, "seat5": 1640, "seat6": 1530},
        "bets": {"seat1": 338, "seat2": None, "seat3": None,
                 "seat4": None, "seat5": None, "seat6": None},
        "empty": {"seat1": False, "seat2": False, "seat3": True,
                  "seat4": False, "seat5": False, "seat6": False},
        "folded": {"seat1": False, "seat2": True, "seat3": False,
                   "seat4": False, "seat5": True, "seat6": True},
        "pot": {"total": 628, "main": 290},
        "suspect": False,
        "controls": {"present": False},
    }


def seq909_qdkc_postf1():
    """Session-5 seq 909 (hand QdKc, postmortem pin 4) transformed to
    the post-F1 BUST-DISPLAY shape: seat2 busted the PREVIOUS hand, the
    button moved onto its vacated seat (a correctly-scraped DEAD BUTTON
    — D2's class, Windows attribution 2026-06-12), and its box is
    synthesized as still rendering "0" (stack=0, empty=False). Ground
    truth that seat2 is NOT in this hand: ante=25 and the pot reads 100
    = 4 antes — exactly the four chips-visible seats posted; seat2 did
    not. F2's commitment test must REFUSE the promotion so the frame
    keeps routing through D2's dead-button path."""
    return {
        "captured_at": "20260612_194105_000",
        "blinds": "% 75/150, 25 Ante No Limit Hold'em - TBL#1",
        "dealer": "seat2",
        "hero_cards": [],
        "board": [],
        "stacks": {"seat1": 2782, "seat2": 0, "seat3": None,
                   "seat4": 1615, "seat5": 3533, "seat6": 970},
        "bets": {"seat1": None, "seat2": None, "seat3": None,
                 "seat4": None, "seat5": None, "seat6": None},
        "empty": {"seat1": False, "seat2": False, "seat3": True,
                  "seat4": False, "seat5": False, "seat6": False},
        "folded": {"seat1": None, "seat2": False, "seat3": False,
                   "seat4": None, "seat5": None, "seat6": None},
        "pot": {"total": 100, "main": 100},
        "suspect": False,
        "controls": {"present": False},
    }


def seq912_qdkc_postf1():
    """Session-5 seq 912 (the QdKc hero-to-act kill frame, fallback #5)
    in post-F1 shape: blinds posted (SB 75 seat4, BB 150 seat5), pot 325
    = 100 antes + 75 + 225? no — 100 + 75 + 150. seat2 (dead button)
    synthesized as stack=0/empty=False. Commitment test: residual pot
    325 - 225 visible bets = 100 < 25 * (4+1) = 125 -> NOT committed."""
    return {
        "captured_at": "20260612_194112_000",
        "blinds": "% 75/150, 25 Ante No Limit Hold'em - TBL#1",
        "dealer": "seat2",
        "hero_cards": ["Qd", "Kc"],
        "board": [],
        "stacks": {"seat1": 2782, "seat2": 0, "seat3": None,
                   "seat4": 1540, "seat5": 3383, "seat6": 970},
        "bets": {"seat1": None, "seat2": None, "seat3": None,
                 "seat4": 75, "seat5": 150, "seat6": None},
        "empty": {"seat1": False, "seat2": False, "seat3": True,
                  "seat4": False, "seat5": False, "seat6": False},
        "folded": {"seat1": False, "seat2": False, "seat3": False,
                   "seat4": False, "seat5": False, "seat6": True},
        "pot": {"total": 325, "main": 100},
        "suspect": False,
        "controls": {"present": True,
                     "action_buttons": [{"label": "FOLD"},
                                        {"label": "CALL"},
                                        {"label": "RAISE"}]},
    }


def four_handed_one_allin():
    """The 5hTs endgame class (postmortem pin 7 second killer):
    4-handed, one seat all-in reading explicit 0 -> pre-F2 the frame
    dropped as n_alive=3 < 4. Post-F1/F2 it must count the all-in seat
    and parse. Mid-hand flop frame, all-in seat5's full stack in pot."""
    return {
        "captured_at": "20260612_200600_000",
        "blinds": "% 50/100, 15 Ante No Limit Hold'em - TBL#1",
        "dealer": "seat1",
        "hero_cards": ["5h", "Ts"],
        "board": ["2c", "7d", "Jh"],
        "stacks": {"seat1": 2000, "seat2": None, "seat3": 1500,
                   "seat4": None, "seat5": 0, "seat6": 1200},
        "bets": {"seat1": None, "seat2": None, "seat3": None,
                 "seat4": None, "seat5": None, "seat6": None},
        "empty": {"seat1": False, "seat2": True, "seat3": False,
                  "seat4": True, "seat5": False, "seat6": False},
        "folded": {"seat1": False, "seat2": False, "seat3": False,
                   "seat4": False, "seat5": False, "seat6": False},
        "pot": {"total": 1300, "main": 900},
        "suspect": False,
        "controls": {"present": False},
    }


# ── flag OFF: byte-identical pre-F2 behavior ──────────────────────────

def test_flag_off_dealer_on_explicit_zero_seat_still_drops():
    with pytest.raises(ScraperDataQuality):
        parse_frame(seq472_8s5c())


def test_flag_off_n_alive_gate_still_drops_four_handed_allin():
    with pytest.raises(ScraperDataQuality):
        parse_frame(four_handed_one_allin())


def test_flag_off_default_frame_field_is_all_false():
    # A frame that parses normally carries the additive default.
    rec = seq912_qdkc_postf1()
    rec["dealer"] = "seat4"  # point the button at a chips-visible seat
    f = parse_frame(rec)
    assert f.allin_seats == (False,) * NUM_SEATS


# ── flag ON: committed explicit-0 seat = valid all-in ─────────────────

def test_committed_allin_dealer_seat_parses_not_dead(_=None):
    f = parse_frame(seq472_8s5c(), allin_zero_stack=True)
    assert f.allin_seats[3] is True or f.allin_seats[3] == True  # seat4
    assert f.dealer_dead is False           # NOT a dead-button candidate
    assert f.alive[3] is False              # downstream busted-mid-hand model
    assert sum(f.allin_seats) == 1


def test_committed_allin_counts_toward_n_alive_gate():
    f = parse_frame(four_handed_one_allin(), allin_zero_stack=True)
    assert f.allin_seats[4]                 # seat5 promoted
    assert sum(f.alive) == 3                # alive[] unchanged semantics
    # the frame parsed at all == the n_alive>=4 gate counted the all-in


# ── flag ON: uncommitted explicit-0 seat stays a D2 dead button ───────

def test_uncommitted_zero_seat_is_not_promoted_and_d2_routes():
    # F2 alone: refusal preserved (commitment test fails: residual pot
    # 100 < ante*(4+1)=125 — seat2 posted no ante).
    with pytest.raises(ScraperDataQuality):
        parse_frame(seq909_qdkc_postf1(), allin_zero_stack=True)
    # F2 + D2: the dead-button path takes it, allin_seats stays empty.
    f = parse_frame(seq909_qdkc_postf1(), allin_zero_stack=True,
                    dead_button_handling=True)
    assert f.dealer_dead is True
    assert f.allin_seats == (False,) * NUM_SEATS


def test_uncommitted_zero_seat_blinds_posted_frame_too():
    # seq 912 shape: blinds visible; residual pot 325-225=100 < 125.
    with pytest.raises(ScraperDataQuality):
        parse_frame(seq912_qdkc_postf1(), allin_zero_stack=True)
    f = parse_frame(seq912_qdkc_postf1(), allin_zero_stack=True,
                    dead_button_handling=True)
    assert f.dealer_dead is True
    assert f.allin_seats == (False,) * NUM_SEATS


def test_committed_preflop_allin_by_antes_promotes():
    # A seat whose whole stack went in preflop: pot covers its ante.
    rec = seq912_qdkc_postf1()
    rec["pot"] = {"total": 350, "main": 125}   # 5 antes + 75 + 150
    f = parse_frame(rec, allin_zero_stack=True)
    assert f.allin_seats[1]                    # seat2 promoted
    assert f.dealer_dead is False


# ── exclusions: the promotion never fires on these ────────────────────

def test_none_stack_read_is_never_promoted():
    """The None-vs-0 distinction: a missing/None read (pre-F1 logs, or a
    genuinely vacated seat post-F1) is NOT all-in evidence."""
    rec = seq472_8s5c()
    rec["stacks"]["seat4"] = None
    with pytest.raises(ScraperDataQuality):
        parse_frame(rec, allin_zero_stack=True)


def test_folded_zero_seat_is_never_promoted():
    rec = seq472_8s5c()
    rec["folded"]["seat4"] = True   # cannot fold all-in: OCR noise
    with pytest.raises(ScraperDataQuality):
        parse_frame(rec, allin_zero_stack=True)


def test_empty_zero_seat_is_never_promoted():
    rec = seq472_8s5c()
    rec["empty"]["seat4"] = True
    with pytest.raises(ScraperDataQuality):
        parse_frame(rec, allin_zero_stack=True)


def test_zero_ante_level_is_conservative_no_promotion():
    rec = seq472_8s5c()
    rec["blinds"] = "15/25, 0 Ante"
    with pytest.raises(ScraperDataQuality):
        parse_frame(rec, allin_zero_stack=True)


def test_explicit_dealer_dead_key_ignored_for_committed_seat():
    """Coordination with D2's Windows dealer_dead key: a COMMITTED
    0-stack seat must not read as a dead button even when the scraper
    annotates one (the committed read is higher-information)."""
    rec = seq472_8s5c()
    rec["dealer_dead"] = True
    f = parse_frame(rec, allin_zero_stack=True, dead_button_handling=True)
    assert f.allin_seats[3]
    assert f.dealer_dead is False


# ── downstream coherence: busted-mid-hand machinery picks it up ───────

def test_promoted_seat_routes_into_busted_mid_hand_with_anchor():
    from src.nlhe.integration.scraper_schema import derive_action_sequence
    f = parse_frame(four_handed_one_allin(), allin_zero_stack=True)
    # Anchor: seat5 entered the hand with chips (its 1300-pot shove).
    pre = (2115, 0, 1615, 0, 1015, 1315)
    seq = derive_action_sequence(f, pre_hand_override=pre)
    # seat5 (idx 4) must appear with a chip-committing action (its
    # all-in emission), never a fold.
    seat5_actions = [ci for s, ci in seq if s == 4]
    assert seat5_actions and all(ci != 0 for ci in seat5_actions)


def test_rebuild_helpers_preserve_allin_annotation():
    import dataclasses
    from src.nlhe.integration.scraper_schema import (
        _rebuild_frame_with_stack,
    )
    f = parse_frame(seq472_8s5c(), allin_zero_stack=True)
    g = _rebuild_frame_with_stack(f, 0, 999)
    assert g.allin_seats == f.allin_seats
    assert g.alive[3] is False   # recomputed alive keeps promoted seat out
