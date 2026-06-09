"""Layer-1 single-field stack recovery — gate + regression tests.

Pins down the 2026-06-09 verify1 blackout fix: at seq=83-94 the scraper's
SanityChecker suspect-flagged 12 consecutive frames because hero's stack
OCR'd as a stable 11101 (stuck leading digit; actual 1110), dropping 4
hero-to-act moments (seq 88/92/93/94) and busting hero on the 38.4s freeze.
Every OTHER field in those frames was correct (verified against the seq=95
post-hand ground truth), so the single-flagged-stack recovery path must:

  - derive the flagged stack from the clean hand-start anchor via chip
    conservation,
  - re-validate through the full replay + invariant gate,
  - decide (status decision_recovered) when it passes,
  - and decline into the EXACT pre-existing safe-fold drop on every gate
    failure (no anchor, non-stack reasons, multi-seat reasons, out-of-range
    candidate, failed re-validation, strict/uilag ambiguity).

Records below are verbatim raw_records from
logs/live_dryrun_verify1_20260609_202137.jsonl (seq 72, 79, 88), minus the
Windows-side frame-path key.
"""
from __future__ import annotations

import copy
import json
import random

import pytest

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.live_loop import (
    DecisionCache, make_decision, _attempt_suspect_stack_recovery,
)
from src.nlhe.integration.scraper_schema import (
    ScraperSuspect, SessionTracker, parse_frame,
    build_stack_recovery_candidates, recoverable_suspect_seat,
)

STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"

_CONTROLS_FLOP = {
    "present": True, "stable": True,
    "action_buttons": [
        {"label": "BET", "slot": "right",
         "box": [0.668, 0.775, 0.8, 0.857], "amount": 25},
        {"label": "CHECK", "slot": "center",
         "box": [0.515, 0.775, 0.66, 0.857], "amount": None},
    ],
    "bet_input": {"box": [0.705, 0.876, 0.812, 0.922], "value": 25},
    "slider": {"box": [0.43, 0.886, 0.672, 0.916], "present": True},
    "shortcuts": [],
}

# seq=72 — hand-start anchor frame (dealer seat4, 15/25/5, Σpre-hand = 9000)
REC_HAND_START = {
    "blinds": "% 15/25, 5 Ante No Limit Hold'em - Hold'em (Double-Up Turbo) - TBL#1",
    "hero_cards": ["Qs", "4c"], "board": [],
    "pot": {"total": 70, "main": 30},
    "stacks": {"seat1": 1310, "seat2": 1698, "seat3": 1500,
               "seat4": 1610, "seat5": 1382, "seat6": 1430},
    "folded": {f"seat{i}": False for i in range(1, 7)},
    "empty": {f"seat{i}": False for i in range(1, 7)},
    "bets": {"seat1": None, "seat2": None, "seat3": None,
             "seat4": None, "seat5": 15, "seat6": 25},
    "dealer": "seat4",
    "controls": {"present": False, "stable": True, "action_buttons": [],
                 "bet_input": {"box": [0, 0, 0, 0], "value": None},
                 "slider": {"box": [0, 0, 0, 0], "present": False},
                 "shortcuts": []},
    "table_rendered": True, "suspect": False,
    "captured_at": "20260609_162518_092",
}

# seq=79 — clean flop hero-to-act frame, same hand (hero stack truth: 1260)
REC_FLOP_DECISION = {
    "blinds": "% 15/25, 5 Ante No Limit Hold'em - Hold'em (Double-Up Turbo) - TBL#1",
    "hero_cards": ["Qs", "4c"], "board": ["4d", "7d", "9s"],
    "pot": {"total": 180, "main": 180},
    "stacks": {"seat1": 1260, "seat2": 1698, "seat3": 1500,
               "seat4": 1610, "seat5": 1347, "seat6": 1405},
    "folded": {f"seat{i}": False for i in range(1, 7)},
    "empty": {f"seat{i}": False for i in range(1, 7)},
    "bets": {f"seat{i}": None for i in range(1, 7)},
    "dealer": "seat4",
    "controls": _CONTROLS_FLOP,
    "table_rendered": True, "suspect": False,
    "captured_at": "20260609_162551_676",
}

# seq=88 — THE incident frame: turn BET/CHECK, hero stack OCR'd 11101
# (truth 1110), every other field correct. Recovery must rescue this.
REC_TURN_SUSPECT = {
    "blinds": "% 15/25, 5 Ante No Limit Hold'em - Hold'em (Double-Up Turbo) - TBL#1",
    "hero_cards": ["Qs", "4c"], "board": ["4d", "7d", "9s", "Qh"],
    "pot": {"total": 480, "main": 480},
    "stacks": {"seat1": 11101, "seat2": 1698, "seat3": 1500,
               "seat4": 1610, "seat5": 1197, "seat6": 1405},
    "folded": {f"seat{i}": False for i in range(1, 7)},
    "empty": {f"seat{i}": False for i in range(1, 7)},
    "bets": {f"seat{i}": None for i in range(1, 7)},
    "dealer": "seat4",
    "controls": _CONTROLS_FLOP,
    "table_rendered": True, "suspect": True,
    "suspect_reasons": ["seat1 stack jump 1260->11101"],
    "captured_at": "20260609_162607_537",
}


@pytest.fixture(scope="module")
def structure():
    return TournamentStructure.from_yaml(STRUCTURE_YAML)


@pytest.fixture
def anchored_tracker():
    tracker = SessionTracker()
    tracker.observe(parse_frame(REC_HAND_START))
    assert tracker._current_pre_hand is not None, "anchor must register"
    return tracker


@pytest.fixture
def stub_policy(monkeypatch):
    """make_decision imports _sample_action_from_policy at call time —
    stub it so no solver checkpoint is needed. chip_int=1 = CALL/CHECK."""
    import scripts.eval_6max_self_play as ev

    def _stub(solver, parsed, state, rng, mode="sample", policy_filter=None):
        return 1

    monkeypatch.setattr(ev, "_sample_action_from_policy", _stub)
    return _stub


def _suspect_copy(rec, hero_stack, reasons):
    r = copy.deepcopy(rec)
    r["stacks"]["seat1"] = hero_stack
    r["suspect"] = True
    r["suspect_reasons"] = reasons
    return r


def _decide(record, structure, tracker, cache=None):
    return make_decision(record, structure, solver=None, tracker=tracker,
                         rng=random.Random(42), mode="sample", seq=1,
                         decision_cache=cache)


# ── default-path behavior unchanged ─────────────────────────────────────

def test_parse_frame_still_raises_on_suspect_by_default():
    with pytest.raises(ScraperSuspect):
        parse_frame(REC_TURN_SUSPECT)


def test_clean_frame_decides_without_recovery(structure, anchored_tracker,
                                              stub_policy):
    d = _decide(REC_FLOP_DECISION, structure, anchored_tracker)
    assert d.status == "decision"
    assert d.recovered_fields is None
    assert d.hero_stack == 1260


# ── the incident regression: seq=88 must recover to 1110 ────────────────

def test_seq88_incident_frame_recovers(structure, anchored_tracker,
                                       stub_policy):
    d = _decide(REC_TURN_SUSPECT, structure, anchored_tracker)
    assert d.status == "decision_recovered"
    assert d.recovered_fields == ["stack.seat1=1110 (strict)"]
    assert d.hero_stack == 1110
    assert d.client_action is not None


def test_recovered_decision_caches(structure, anchored_tracker, stub_policy):
    cache = DecisionCache()
    d1 = _decide(REC_TURN_SUSPECT, structure, anchored_tracker, cache)
    d2 = _decide(REC_TURN_SUSPECT, structure, anchored_tracker, cache)
    assert d1.status == "decision_recovered"
    assert d2.status == "decision_recovered_cached"
    assert d1.client_action == d2.client_action


def test_recovery_matches_clean_twin_action(structure, stub_policy):
    """The recovered frame must produce the SAME decision the clean frame
    with the true stack produces — recovery reconstructs, never invents."""
    clean = copy.deepcopy(REC_TURN_SUSPECT)
    clean["stacks"]["seat1"] = 1110
    clean["suspect"] = False
    clean.pop("suspect_reasons")

    t1 = SessionTracker(); t1.observe(parse_frame(REC_HAND_START))
    t2 = SessionTracker(); t2.observe(parse_frame(REC_HAND_START))
    d_clean = _decide(clean, structure, t1)
    d_rec = _decide(REC_TURN_SUSPECT, structure, t2)
    assert d_clean.status == "decision"
    assert d_rec.status == "decision_recovered"
    assert d_rec.client_action == d_clean.client_action
    assert d_rec.hero_stack == d_clean.hero_stack == 1110


# ── decline gates: every failure must drop exactly as before ────────────

def _assert_declined(d, why_fragment):
    assert d.status == "skip_data_quality"
    assert "ScraperSuspect" in d.skip_reason
    assert "recovery declined" in d.skip_reason
    assert why_fragment in d.skip_reason
    assert d.recovered_fields is None
    assert d.client_action is None


def test_no_anchor_declines(structure, stub_policy):
    d = _decide(REC_TURN_SUSPECT, structure, SessionTracker())
    _assert_declined(d, "no clean pre-hand anchor")


def test_non_stack_reason_declines(structure, anchored_tracker, stub_policy):
    rec = _suspect_copy(REC_FLOP_DECISION, 11260,
                        ["pot jump 100->900"])
    d = _decide(rec, structure, anchored_tracker)
    _assert_declined(d, "not a single-seat stack jump")


def test_multi_seat_reasons_decline(structure, anchored_tracker, stub_policy):
    rec = _suspect_copy(REC_FLOP_DECISION, 11260,
                        ["seat1 stack jump 1260->11260",
                         "seat5 stack jump 1347->9999"])
    d = _decide(rec, structure, anchored_tracker)
    _assert_declined(d, "not a single-seat stack jump")


def test_not_hero_to_act_declines(structure, anchored_tracker, stub_policy):
    rec = _suspect_copy(REC_FLOP_DECISION, 11260,
                        ["seat1 stack jump 1260->11260"])
    rec["controls"] = {"present": False, "stable": True,
                       "action_buttons": [], "bet_input": {}, "slider": {},
                       "shortcuts": []}
    d = _decide(rec, structure, anchored_tracker)
    _assert_declined(d, "not a hero-to-act frame")


def test_out_of_range_candidate_declines(structure, anchored_tracker,
                                         stub_policy):
    # Second corruption pushes the solved hero stack past pre_hand - ante:
    # seat5 reads 150 LOW -> candidate = 1410 > 1310 -> infeasible -> drop.
    rec = _suspect_copy(REC_FLOP_DECISION, 11260,
                        ["seat1 stack jump 1260->11260"])
    rec["stacks"]["seat5"] = 1197
    d = _decide(rec, structure, anchored_tracker)
    _assert_declined(d, "no in-range closure candidate")


def test_double_corruption_rejected_by_revalidation(structure,
                                                    anchored_tracker,
                                                    stub_policy):
    # Second corruption (seat5 reads 150 HIGH) keeps the solved candidate
    # in-range (1110) but implies a negative contribution from seat5 —
    # the replay/invariant re-validation must reject it. This is the
    # never-feed-the-model-a-wrong-derivation property.
    rec = _suspect_copy(REC_FLOP_DECISION, 11260,
                        ["seat1 stack jump 1260->11260"])
    rec["stacks"]["seat5"] = 1497
    d = _decide(rec, structure, anchored_tracker)
    _assert_declined(d, "no candidate passed replay+invariant")


def test_ambiguous_dual_candidates_decline(structure, anchored_tracker,
                                           monkeypatch, stub_policy):
    # UI-lag-shaped frame (visible uncollected bet) yields TWO in-range
    # candidates (strict 1260, uilag 1110). Force re-validation to approve
    # both: arbitration must refuse to pick and drop the frame.
    rec = _suspect_copy(REC_FLOP_DECISION, 11260,
                        ["seat1 stack jump 1260->11260"])
    rec["stacks"]["seat5"] = 1197
    rec["bets"]["seat5"] = 150
    rec["pot"]["total"] = 330

    import src.nlhe.integration.replay as replay_mod
    import src.nlhe.integration.invariant as inv_mod

    class _Pack:
        pass

    class _Inv:
        ok = True
        deltas = []

    monkeypatch.setattr(replay_mod, "replay_to_decision",
                        lambda *a, **k: _Pack())
    monkeypatch.setattr(inv_mod, "check_mid_hand_invariant",
                        lambda *a, **k: _Inv())
    d = _decide(rec, structure, anchored_tracker)
    _assert_declined(d, "ambiguous")


# ── helper-level checks ──────────────────────────────────────────────────

def test_recoverable_suspect_seat_parsing():
    ok = {"suspect_reasons": ["seat3 stack jump 100->9100"]}
    assert recoverable_suspect_seat(ok) == 2
    same_seat_twice = {"suspect_reasons": ["seat3 stack jump 100->9100",
                                           "seat3 stack jump 100->9101"]}
    assert recoverable_suspect_seat(same_seat_twice) == 2
    assert recoverable_suspect_seat({"suspect_reasons": []}) is None
    assert recoverable_suspect_seat({}) is None
    assert recoverable_suspect_seat(
        {"suspect_reasons": ["seat7 stack jump 1->2"]}) is None
    assert recoverable_suspect_seat(
        {"suspect_reasons": ["board flicker"]}) is None


def test_candidate_solve_strict_and_uilag(anchored_tracker):
    # Strict-only when no bets are on the table.
    frame = parse_frame(REC_TURN_SUSPECT, allow_suspect=True)
    anchor = anchored_tracker.anchor_for(frame)
    assert anchor is not None
    cands = build_stack_recovery_candidates(frame, 0, anchor)
    assert [(c.stack[0], lbl) for c, lbl in cands] == [(1110, "strict")]
    # Recovered frame keeps every other field bit-identical.
    rec_frame = cands[0][0]
    assert rec_frame.stack[1:] == frame.stack[1:]
    assert rec_frame.bet == frame.bet
    assert rec_frame.pot_total == frame.pot_total
    assert rec_frame.board == frame.board


def test_tracker_observe_blind_to_suspect_frames(structure, anchored_tracker,
                                                 stub_policy):
    """A recovered decision must not mutate the tracker's anchor state."""
    pre = anchored_tracker._current_pre_hand
    key = anchored_tracker._current_hand_key
    d = _decide(REC_TURN_SUSPECT, structure, anchored_tracker)
    assert d.status == "decision_recovered"
    assert anchored_tracker._current_pre_hand == pre
    assert anchored_tracker._current_hand_key == key
