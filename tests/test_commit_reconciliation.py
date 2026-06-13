"""CR anchored commit reconciliation — gate + regression tests.

Built from the three 2026-06-12 live casualties in
logs/live_dryrun_20260612_230149.jsonl — the class pre-registered in
evals/p2_session5_validation_20260613/REPORT.txt: the SCRAPER frame is
RIGHT (anchor-conserving; outcome ground truth sided with it 3/3) and
the RECONSTRUCTION under-counts a non-hero seat's committed chips, so
the strict invariant kills a real decision. CR rebuilds the believed
action sequence from anchor-implied per-seat commits and re-validates
the UNCHANGED frame; the frame is never patched (that is P2's
direction, proven backwards for this class).

GROUND-TRUTH CHAINS (regression fixtures; full walkthroughs in
evals/p2_session5_validation_20260613/PER_PIN_ANALYSIS.txt):
  seq 780  8c8d — folded SB/BB sweep. seat4 1630 = 1695-15(ante)-50(SB),
      seat5 2685 = 2800-15-100(BB); uncalled-raise return at seq 781/784
      and the next hand-start (seq 786, every stack exactly -15) confirm
      both blinds were posted and lost. Believed state must be pot 1017,
      stack4 1630, stack5 2685.
  seq 1086 AcAh — limp-then-fold sweep. seat3 limped 25 then folded
      (1707-5-25=1677); operator called 613, turned the As, won 3306
      (seq 1087-1094 chain). Believed state must be pot 2693, stack3
      1677; decision CALL of 613.
  seq 1635 5hTs — ante-bookkeeping on an all-in-for-less hand. seat3
      3464-25(ante)-150(call)=3289; hand checks down, seat3 collects
      exactly the scraper's 509 pot (seq 1646-47). Believed state must
      be pot 509, bet3 150, stack3 3289, CHECK available (facing_bet
      False). NOTE: the policy itself (not CR) chooses the action on the
      recovered state — sample/argmax both currently favor ALLIN here;
      the registration's "decision CHECK" is the ground-truth/fallback
      action and is asserted as CHECK-availability, not as the sample.

The seq-276 dead-SB fixture (G3) is imported from the P2 test module:
CR must STILL refuse it — it has no REGULAR hand-start anchor (a
dead-SB hand has no SB poster, so is_hand_start can never fire), and
the regular anchor is CR's positive blind-structure evidence.

Records below are verbatim raw_records from the live log (Windows-side
frame-path key dropped).
"""
from __future__ import annotations

import copy
import random

import pytest

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.live_loop import (
    DecisionCache, make_decision, _attempt_commit_reconciliation,
)
from src.nlhe.integration.scraper_schema import (
    SessionTracker, parse_frame, is_hand_start,
    classify_commit_reconciliation_deltas,
    build_commit_reconciliation_shadow,
    implied_commits_from_anchor,
)
from src.nlhe.integration.invariant import ReconciledEmission

# The seq-276 dead-SB universe (G3 regression) — same fixtures P2 uses.
from tests.test_bet_closure_recovery import (
    REC_PREBLIND, REC_BB_ONLY, REC_DISPLACED,
)

STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"

_NO_CONTROLS = {
    "present": False, "stable": True, "action_buttons": [],
    "bet_input": {"box": [0, 0, 0, 0], "value": None},
    "slider": {"box": [0, 0, 0, 0], "present": False},
    "shortcuts": [],
}

_CALL_FOLD_CONTROLS = {
    "present": True, "stable": True,
    "action_buttons": [
        {"label": "CALL", "slot": "center",
         "box": [0.515, 0.775, 0.66, 0.857], "amount": None},
        {"label": "FOLD", "slot": "left",
         "box": [0.366, 0.775, 0.505, 0.857], "amount": None},
    ],
    "bet_input": {"box": [0.705, 0.876, 0.812, 0.922], "value": None},
    "slider": {"box": [0.43, 0.886, 0.672, 0.916], "present": False},
    "shortcuts": [],
}


# ── PIN 1: seq 780 (8c8d) — folded SB/BB sweep ───────────────────────────

_BLINDS_50_100 = ("% 50/100, 15 Ante No Limit Hold'em - Hold'em "
                  "(Double-Up Turbo) - TBL#1")

# seq 775 — the hand-start anchor frame (SB 50 on seat4, BB 100 on
# seat5 both visible; pre-hand (2523, 707, 0, 1695, 2800, 1275), sum
# 9000). Hero-to-act at its own first decision, but only observe() is
# exercised by the anchor fixtures.
REC_780_HAND_START = {
    "blinds": _BLINDS_50_100,
    "hero_cards": ["8c", "8d"], "board": [],
    "pot": {"total": 225, "main": 75},
    "stacks": {"seat1": 2508, "seat2": 692, "seat3": None,
               "seat4": 1630, "seat5": 2685, "seat6": 1260},
    "folded": {"seat1": False, "seat2": False, "seat3": False,
               "seat4": False, "seat5": False, "seat6": True},
    "empty": {"seat1": False, "seat2": False, "seat3": True,
              "seat4": False, "seat5": False, "seat6": False},
    "bets": {"seat1": None, "seat2": None, "seat3": None,
             "seat4": 50, "seat5": 100, "seat6": None},
    "dealer": "seat2",
    "controls": _NO_CONTROLS,
    "table_rendered": True, "suspect": False,
    "captured_at": "20260612_193655_682",
}

# seq 780 — THE casualty frame: hero 8c8d limped 100, seat2 raised to
# 692 (all-in), seats 4/5 = FOLDED SB/BB whose posts the recon failed
# to deduct (their bets render null once folded). Scraper is RIGHT:
# 7983 visible + 1017 pot = 9000.
REC_780_PIN = {
    "blinds": _BLINDS_50_100,
    "hero_cards": ["8c", "8d"], "board": [],
    "pot": {"total": 1017, "main": 75},
    "stacks": {"seat1": 2408, "seat2": None, "seat3": None,
               "seat4": 1630, "seat5": 2685, "seat6": 1260},
    "folded": {"seat1": False, "seat2": False, "seat3": False,
               "seat4": True, "seat5": True, "seat6": True},
    "empty": {"seat1": False, "seat2": False, "seat3": True,
              "seat4": False, "seat5": False, "seat6": False},
    "bets": {"seat1": 100, "seat2": 692, "seat3": None,
             "seat4": None, "seat5": None, "seat6": None},
    "dealer": "seat2",
    "controls": _CALL_FOLD_CONTROLS,
    "table_rendered": True, "suspect": False,
    "captured_at": "20260612_193706_536",
}


# ── PIN 2: seq 1086 (AcAh) — limp-then-fold sweep ────────────────────────

_BLINDS_15_25 = ("% 15/25, 5 Ante No Limit Hold'em - Hold'em "
                 "(Double-Up Turbo) - TBL#1")

# seq 1066 — hand-start anchor (SB 15 hero, BB 25 seat2; pre-hand
# (1642, 1163, 1707, 1435, 1435, 1618), sum 9000).
REC_1086_HAND_START = {
    "blinds": _BLINDS_15_25,
    "hero_cards": ["Ac", "Ah"], "board": [],
    "pot": {"total": 70, "main": 30},
    "stacks": {"seat1": 1622, "seat2": 1133, "seat3": 1702,
               "seat4": 1430, "seat5": 1430, "seat6": 1613},
    "folded": {f"seat{i}": False for i in range(1, 7)},
    "empty": {f"seat{i}": False for i in range(1, 7)},
    "bets": {"seat1": 15, "seat2": 25, "seat3": None,
             "seat4": None, "seat5": None, "seat6": None},
    "dealer": "seat6",
    "controls": _NO_CONTROLS,
    "table_rendered": True, "suspect": False,
    "captured_at": "20260612_194913_336",
}

# seq 1086 — THE casualty frame: hero AcAh raised to 1000, seat6 shoved
# 1613, hero faces CALL 613. seat3 limped 25 then FOLDED — the swept
# limp is invisible in the bets row and the recon missed it. Scraper is
# RIGHT: 6307 visible + 2693 pot = 9000.
REC_1086_PIN = {
    "blinds": _BLINDS_15_25,
    "hero_cards": ["Ac", "Ah"], "board": [],
    "pot": {"total": 2693, "main": 30},
    "stacks": {"seat1": 637, "seat2": 1133, "seat3": 1677,
               "seat4": 1430, "seat5": 1430, "seat6": 0},
    "folded": {"seat1": False, "seat2": True, "seat3": True,
               "seat4": True, "seat5": True, "seat6": False},
    "empty": {f"seat{i}": False for i in range(1, 7)},
    "bets": {"seat1": 1000, "seat2": None, "seat3": None,
             "seat4": None, "seat5": None, "seat6": 1613},
    "dealer": "seat6",
    "controls": _CALL_FOLD_CONTROLS,
    "table_rendered": True, "suspect": False,
    "captured_at": "20260612_194941_036",
}


# ── PIN 3: seq 1635 (5hTs) — ante bookkeeping, all-in-for-less hand ──────

_BLINDS_75_150 = ("% 75/150, 25 Ante No Limit Hold'em - Hold'em "
                  "(Double-Up Turbo) - TBL#1")

# seq 1630 — hand-start anchor (SB 75 seat4, BB 150 hero; pre-hand
# (3792, 59, 3464, 1685, 0, 0), sum 9000; seat2 enters with 59).
REC_1635_HAND_START = {
    "blinds": _BLINDS_75_150,
    "hero_cards": ["5h", "Ts"], "board": [],
    "pot": {"total": 325, "main": 100},
    "stacks": {"seat1": 3617, "seat2": 34, "seat3": 3439,
               "seat4": 1585, "seat5": None, "seat6": None},
    "folded": {f"seat{i}": False for i in range(1, 7)},
    "empty": {"seat1": False, "seat2": False, "seat3": False,
              "seat4": False, "seat5": True, "seat6": True},
    "bets": {"seat1": 150, "seat2": None, "seat3": None,
             "seat4": 75, "seat5": None, "seat6": None},
    "dealer": "seat3",
    "controls": _NO_CONTROLS,
    "table_rendered": True, "suspect": False,
    "captured_at": "20260612_200624_444",
}

# seq 1635 — THE casualty frame: hero = BB 150 with CHECK available,
# seat2 all-in 34 (call-for-less), seat3 called 150, seat4 folded SB.
# Scraper is RIGHT (seat3 3464-25-150 = 3289; seat3 later collects
# exactly this 509 pot). The recon under-counted seat3 by the 25 ante:
# the invariant gate's all-in heuristic listed seat2's 59-chip
# call-for-less as a matchable all-in and mis-rendered seat3 through
# the no-ante-subtract branch.
REC_1635_PIN = {
    "blinds": _BLINDS_75_150,
    "hero_cards": ["5h", "Ts"], "board": [],
    "pot": {"total": 509, "main": 100},
    "stacks": {"seat1": 3617, "seat2": None, "seat3": 3289,
               "seat4": 1585, "seat5": None, "seat6": None},
    "folded": {"seat1": False, "seat2": False, "seat3": False,
               "seat4": True, "seat5": False, "seat6": False},
    "empty": {"seat1": False, "seat2": False, "seat3": False,
              "seat4": False, "seat5": True, "seat6": True},
    "bets": {"seat1": 150, "seat2": 34, "seat3": 150,
             "seat4": None, "seat5": None, "seat6": None},
    "dealer": "seat3",
    "controls": {
        "present": True, "stable": True,
        "action_buttons": [
            {"label": "CHECK", "slot": "center",
             "box": [0.515, 0.775, 0.66, 0.857], "amount": None},
            {"label": "RAISE", "slot": "right",
             "box": [0.668, 0.775, 0.8, 0.857], "amount": 300},
        ],
        "bet_input": {"box": [0.705, 0.876, 0.812, 0.922], "value": 300},
        "slider": {"box": [0.43, 0.886, 0.672, 0.916], "present": True},
        "shortcuts": [],
    },
    "table_rendered": True, "suspect": False,
    "captured_at": "20260612_200631_846",
}


# ── plumbing ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def structure():
    return TournamentStructure.from_yaml(STRUCTURE_YAML)


def _cr_tracker(*records):
    tracker = SessionTracker(anchor_sum_floor=True)
    for rec in records:
        tracker.observe(parse_frame(rec))
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


def _decide(record, structure, tracker, cache=None, flag=True, p2=False):
    return make_decision(record, structure, solver=None, tracker=tracker,
                         rng=random.Random(42), mode="sample", seq=1,
                         decision_cache=cache,
                         bet_closure_recovery=p2,
                         commit_reconciliation=flag)


def _assert_refused(d, why_fragment):
    assert d.status == "safe_fold"
    assert d.skip_reason.startswith("invariant_fail")
    assert "commit reconciliation declined" in d.skip_reason
    assert why_fragment in d.skip_reason
    assert d.recovered_fields is None
    assert d.client_action is None
    assert d.invariant_deltas is not None


# ── gating: flag-off byte-identity; CR requires P1 ───────────────────────

def test_flag_off_drop_is_unannotated(structure, stub_policy):
    tracker = _cr_tracker(REC_780_HAND_START)
    d = _decide(REC_780_PIN, structure, tracker, flag=False)
    assert d.status == "safe_fold"
    assert d.skip_reason == "invariant_fail (3 deltas)"
    assert d.recovered_fields is None


def test_p1_not_armed_refuses(structure, stub_policy):
    """Defensive double-gate: even if the flag is passed to
    make_decision, a tracker without the P1 floor refuses."""
    tracker = SessionTracker()  # no sum-floor
    tracker.observe(parse_frame(REC_780_HAND_START))
    d = _decide(REC_780_PIN, structure, tracker)
    _assert_refused(d, "anchor sum-floor guard not armed")


# ── the three pinned casualties: must recover to ground truth ────────────

def test_seq780_folded_blind_sweep_recovers(structure, stub_policy):
    tracker = _cr_tracker(REC_780_HAND_START)
    d = _decide(REC_780_PIN, structure, tracker)
    assert d.status == "decision_recovered"
    assert d.recovered_fields == [
        "commit.seat4=50 (commit_reconciliation)",
        "commit.seat5=100 (commit_reconciliation)",
        "replay=rebuilt_from_anchor_commits "
        "(commit_reconciliation:2 seats restored)",
    ]
    # Ground truth (seq 781-786 chain): the believed state IS the
    # scraper frame — pot 1017, stack4 1630, stack5 2685, hero 2408
    # facing seat2's 692.
    assert d.pot_total == 1017
    assert d.hero_stack == 2408
    assert d.facing_bet is True
    assert d.client_action is not None


def test_seq1086_limp_then_fold_recovers(structure, stub_policy):
    tracker = _cr_tracker(REC_1086_HAND_START)
    d = _decide(REC_1086_PIN, structure, tracker)
    assert d.status == "decision_recovered"
    assert d.recovered_fields == [
        "commit.seat2=25 (commit_reconciliation)",
        "commit.seat3=25 (commit_reconciliation)",
        "replay=rebuilt_from_anchor_commits "
        "(commit_reconciliation:2 seats restored)",
    ]
    # Ground truth (seq 1087-1094 chain): pot 2693 (seat3's swept limp
    # included), hero 637 facing 613. With the stubbed CALL the client
    # action is the registered decision: CALL 613.
    assert d.pot_total == 2693
    assert d.hero_stack == 637
    assert d.facing_bet is True
    assert d.client_action == {"kind": "call", "chip_amount": None,
                               "raw_openspiel_chip_int": 1}


def test_seq1635_ante_bookkeeping_recovers(structure, stub_policy):
    tracker = _cr_tracker(REC_1635_HAND_START)
    d = _decide(REC_1635_PIN, structure, tracker)
    assert d.status == "decision_recovered"
    assert d.recovered_fields == [
        "commit.seat4=75 (commit_reconciliation)",
        "replay=rebuilt_from_anchor_commits "
        "(commit_reconciliation:1 seats restored)",
    ]
    # Ground truth (seq 1646-47: seat3 collects exactly this pot):
    # pot 509, bet3 150 (the UNCHANGED frame carries it), hero = BB
    # with a free CHECK (facing_bet False) — the registered safe
    # decision is available; the stubbed chip_int=1 realizes it.
    assert d.pot_total == 509
    assert d.hero_stack == 3617
    assert d.facing_bet is False
    assert d.client_action == {"kind": "check", "chip_amount": None,
                               "raw_openspiel_chip_int": 1}


def test_recovered_decision_caches(structure, stub_policy):
    tracker = _cr_tracker(REC_1086_HAND_START)
    cache = DecisionCache()
    d1 = _decide(REC_1086_PIN, structure, tracker, cache)
    d2 = _decide(REC_1086_PIN, structure, tracker, cache)
    assert d1.status == "decision_recovered"
    assert d2.status == "decision_recovered_cached"
    assert d1.client_action == d2.client_action


def test_recovery_does_not_mutate_tracker(structure, stub_policy):
    tracker = _cr_tracker(REC_780_HAND_START)
    pre = tracker._current_pre_hand
    key = tracker._current_hand_key
    d = _decide(REC_780_PIN, structure, tracker)
    assert d.status == "decision_recovered"
    assert tracker._current_pre_hand == pre
    assert tracker._current_hand_key == key


def test_cr_runs_after_p2_when_both_armed(structure, stub_policy):
    """With P1+P2+CR all armed (the deployment shape), a mirror-class
    frame still recovers via CR: P2 refuses it first (multi-seat /
    non-closure / re-validation), CR then reconciles."""
    tracker = SessionTracker(anchor_sum_floor=True,
                             bet_closure_recovery=True)
    tracker.observe(parse_frame(REC_780_HAND_START))
    d = _decide(REC_780_PIN, structure, tracker, p2=True)
    assert d.status == "decision_recovered"
    assert any("commit_reconciliation" in f for f in d.recovered_fields)


# ── G3: the seq-276 dead-SB fixture must STILL refuse ────────────────────

def test_seq276_dead_sb_still_refuses_cr_only(structure, stub_policy):
    """The 276 hand has NO regular hand-start anchor (a dead-SB hand has
    no SB poster, so is_hand_start can never fire) — CR's anchor gate,
    which doubles as the positive blind-structure evidence, refuses."""
    tracker = _cr_tracker(REC_PREBLIND, REC_BB_ONLY)
    d = _decide(REC_DISPLACED, structure, tracker)
    _assert_refused(d, "no clean REGULAR hand-start anchor")


def test_seq276_dead_sb_still_refuses_p2_plus_cr(structure, stub_policy):
    """Deployment shape (P1+P2+CR): both recoveries refuse 276 — P2 on
    the bb_only dead-SB guard, CR on the missing regular anchor."""
    tracker = SessionTracker(anchor_sum_floor=True,
                             bet_closure_recovery=True)
    for rec in (REC_PREBLIND, REC_BB_ONLY):
        tracker.observe(parse_frame(rec))
    d = _decide(REC_DISPLACED, structure, tracker, p2=True)
    assert d.status == "safe_fold"
    assert "bet-closure recovery declined" in d.skip_reason
    assert "BB-only posting" in d.skip_reason
    assert "commit reconciliation declined" in d.skip_reason
    assert "no clean REGULAR hand-start anchor" in d.skip_reason
    assert d.recovered_fields is None


# ── refusal classes ──────────────────────────────────────────────────────

def test_no_anchor_refuses(structure, stub_policy):
    tracker = SessionTracker(anchor_sum_floor=True)  # never observed
    d = _decide(REC_780_PIN, structure, tracker)
    _assert_refused(d, "no clean REGULAR hand-start anchor")


def test_stale_anchor_refuses(structure, stub_policy):
    """An anchor from a DIFFERENT hand-key (dealer moved) is never
    served — hand-key mismatch reads as no anchor."""
    stale = copy.deepcopy(REC_780_HAND_START)
    stale["dealer"] = "seat4"
    stale["bets"] = {"seat1": 100, "seat2": None, "seat3": None,
                     "seat4": None, "seat5": 50, "seat6": None}
    stale["stacks"]["seat5"] = 2735
    stale["stacks"]["seat1"] = 2508
    tracker = _cr_tracker(stale)
    d = _decide(REC_780_PIN, structure, tracker)
    _assert_refused(d, "no clean REGULAR hand-start anchor")


def test_non_conserving_frame_refuses(structure, stub_policy):
    """A frame that does not conserve chips against the anchor is not
    this class (could be a scraper misread — P2's direction)."""
    bad = copy.deepcopy(REC_780_PIN)
    bad["stacks"]["seat4"] = 1680   # +50: sum no longer 9000
    tracker = _cr_tracker(REC_780_HAND_START)
    d = _decide(bad, structure, tracker)
    _assert_refused(d, "does not conserve chips against the anchor")


def test_live_seat_commit_mismatch_refuses(structure, stub_policy):
    """A LIVE (non-folded) seat whose implied commit disagrees with its
    visible bet is a mid-collection capture or another class — refuse.
    Built sum-conserving so the conservation gate passes."""
    bad = copy.deepcopy(REC_780_PIN)
    bad["stacks"]["seat1"] = 2358   # hero stack -50: implied 150 > bet 100
    bad["stacks"]["seat6"] = 1310   # seat6 +50 (keeps sum 9000)
    tracker = _cr_tracker(REC_780_HAND_START)
    d = _decide(bad, structure, tracker)
    # hero is checked first in the builder: implied 150 != bet 100.
    # seat6 (+50 hidden) would be caught one line later by the same
    # equality if hero were clean.
    _assert_refused(d, "hero implied commit 150 != visible bet 100")


def test_postflop_frame_refuses(structure, stub_policy):
    """The proven class is preflop-only; a postflop pot/stack/bet
    invariant_fail annotates the street refusal instead of rebuilding."""
    post = copy.deepcopy(REC_1635_PIN)
    post["board"] = ["2c", "7d", "Jh"]
    tracker = _cr_tracker(REC_1635_HAND_START)
    d = _decide(post, structure, tracker)
    assert d.status == "safe_fold"
    if "commit reconciliation declined" in (d.skip_reason or ""):
        assert "not a preflop frame" in d.skip_reason
    assert d.recovered_fields is None


def test_non_signature_invariant_fail_is_untouched(structure, stub_policy,
                                                   monkeypatch):
    """An invariant_fail with any non-pot/stack/bet delta (e.g. the live
    seq-271 scraper_self_consistency family) drops with NO annotation —
    byte-identical to the flag-off path."""
    from src.nlhe.integration import live_loop as ll
    from src.nlhe.integration.invariant import InvariantResult

    real = ll.make_decision  # ensure import side effects unchanged
    import src.nlhe.integration.invariant as inv_mod

    orig = inv_mod.check_mid_hand_invariant

    def fake_inv(frame, pack, reconciled_emission=None):
        r = orig(frame, pack, reconciled_emission=reconciled_emission)
        if reconciled_emission is None and not r.ok:
            r.deltas.append(("scraper_self_consistency:pot", 1017, 900))
        return r

    monkeypatch.setattr(inv_mod, "check_mid_hand_invariant", fake_inv)
    tracker = _cr_tracker(REC_780_HAND_START)
    d = _decide(REC_780_PIN, structure, tracker)
    assert d.status == "safe_fold"
    assert "commit reconciliation declined" not in (d.skip_reason or "")
    assert d.recovered_fields is None


# ── unit: classifier ─────────────────────────────────────────────────────

def test_classify_commit_deltas_candidate():
    assert classify_commit_reconciliation_deltas(
        [("pot", 1017, 867), ("stack[seat4]", 1630, 1680),
         ("stack[seat5]", 2685, 2785)]) == "candidate"
    assert classify_commit_reconciliation_deltas(
        [("pot", 2693, 2668), ("stack[seat3]", 1677, 1702)]) == "candidate"
    assert classify_commit_reconciliation_deltas(
        [("pot", 509, 484), ("stack[seat3]", 3289, 3314),
         ("bet[seat3]", 150, 125)]) == "candidate"


def test_classify_commit_deltas_not_signature():
    assert classify_commit_reconciliation_deltas([]) == "not_signature"
    assert classify_commit_reconciliation_deltas(
        [("pot", 1, 2), ("current_player", 0, 3)]) == "not_signature"
    assert classify_commit_reconciliation_deltas(
        [("pot", 1, 2), ("scraper_self_consistency:pot", 1, 2)]
    ) == "not_signature"
    assert classify_commit_reconciliation_deltas(
        [("street_idx", 0, 1)]) == "not_signature"
    assert classify_commit_reconciliation_deltas(
        [("hero_cards", "AcAh", "AcKc")]) == "not_signature"


# ── unit: implied commits + shadow builder ───────────────────────────────

_ANCHOR_780 = (2523, 707, 0, 1695, 2800, 1275)
_ANCHOR_1086 = (1642, 1163, 1707, 1435, 1435, 1618)
_ANCHOR_1635 = (3792, 59, 3464, 1685, 0, 0)


def test_implied_commits_pin780():
    frame = parse_frame(REC_780_PIN)
    c = implied_commits_from_anchor(frame, _ANCHOR_780)
    assert c == {0: 100, 1: 692, 3: 50, 4: 100, 5: 0}


def test_shadow_pin780_restores_folded_blinds():
    frame = parse_frame(REC_780_PIN)
    shadow, restored, why = build_commit_reconciliation_shadow(
        frame, _ANCHOR_780)
    assert why == "ok"
    assert restored == {3: 50, 4: 100}
    assert shadow.bet == (100, 692, 0, 50, 100, 0)
    assert shadow.stack == frame.stack          # never touched
    assert shadow.pot_total == frame.pot_total  # never touched


def test_shadow_pin1086_restores_swept_limp():
    frame = parse_frame(REC_1086_PIN)
    shadow, restored, why = build_commit_reconciliation_shadow(
        frame, _ANCHOR_1086)
    assert why == "ok"
    # seat2 = folded BB (post restored), seat3 = the swept limp
    assert restored == {1: 25, 2: 25}
    assert shadow.bet == (1000, 25, 25, 0, 0, 1613)


def test_shadow_pin1635_restores_folded_sb_only():
    frame = parse_frame(REC_1635_PIN)
    shadow, restored, why = build_commit_reconciliation_shadow(
        frame, _ANCHOR_1635)
    assert why == "ok"
    assert restored == {3: 75}
    assert shadow.bet == (150, 34, 150, 75, 0, 0)


def test_shadow_refuses_commit_below_bet():
    frame = parse_frame(REC_1635_PIN)
    bad_anchor = (3792, 59, 3464 - 50, 1685, 0, 0)  # seat3 implied 100 < bet 150
    shadow, restored, why = build_commit_reconciliation_shadow(
        frame, bad_anchor)
    assert shadow is None
    # conservation breaks first with a -50 anchor; both gates are
    # acceptable refusals for this corruption — assert it refused.
    assert "conserve" in why or "below visible bet" in why


def test_shadow_refuses_negative_commit():
    frame = parse_frame(REC_780_PIN)
    bad_anchor = (2523, 707, 0, 1695, 2800, 1260)  # seat6 implied -15
    shadow, restored, why = build_commit_reconciliation_shadow(
        frame, bad_anchor)
    assert shadow is None


def test_shadow_refuses_alive_seat_missing_from_anchor():
    frame = parse_frame(REC_780_PIN)
    bad_anchor = (2523, 707, 0, 1695, 2800, 0)  # seat6 alive, pre 0
    shadow, restored, why = build_commit_reconciliation_shadow(
        frame, bad_anchor)
    assert shadow is None
    assert "absent from the anchor" in why


def test_shadow_no_restoration_needed_is_ok():
    """A conserving candidate frame with no swept folded commits returns
    the frame itself (seq-1635-style recoveries can still proceed via
    exact-emission ante bookkeeping)."""
    frame = parse_frame(REC_1635_PIN)
    # An anchor where the folded SB shows zero implied commit beyond the
    # ante (synthetic): seat4 pre = stack + ante.
    anchor = (3792, 59, 3464, 1610, 0, 0)
    # sum changes -> conservation refuses; rebuild a conserving variant
    # by moving the 75 to the pot side is not representable, so instead
    # exercise the no-restoration path directly on a frame whose folded
    # seats committed nothing: strip seat4's fold commit.
    rec = copy.deepcopy(REC_1635_PIN)
    rec["stacks"]["seat4"] = 1660       # only the ante left seat4
    rec["pot"]["total"] = 434           # 509 - 75
    frame2 = parse_frame(rec)
    anchor2 = (3792, 59, 3464, 1685, 0, 0)
    shadow, restored, why = build_commit_reconciliation_shadow(
        frame2, anchor2)
    assert why == "ok"
    assert restored == {}
    assert shadow is frame2


# ── unit: ReconciledEmission default is byte-identical ───────────────────

def test_invariant_default_kwarg_is_identity(structure):
    """check_mid_hand_invariant(frame, pack) with no kwarg must produce
    the exact same result object content as before the CR build — the
    pin-780 deltas are the regression value."""
    from src.nlhe.integration.replay import replay_to_decision
    from src.nlhe.integration.invariant import check_mid_hand_invariant

    frame = parse_frame(REC_780_PIN)
    pack = replay_to_decision(frame, structure,
                              pre_hand_override=_ANCHOR_780)
    inv = check_mid_hand_invariant(frame, pack)
    assert not inv.ok
    assert inv.deltas == [("pot", 1017, 867),
                          ("stack[seat4]", 1630, 1680),
                          ("stack[seat5]", 2685, 2785)]
