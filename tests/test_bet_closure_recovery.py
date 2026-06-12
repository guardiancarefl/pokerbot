"""P2 bet-closure recovery — gate + regression tests.

Built from the 2026-06-12 live seq=276 frame (session
logs/live_dryrun_20260612_161347.jsonl): AcKc dropped as a 3-delta
invariant_fail whose deltas close under a single bet/stack transfer on
seat5 — the "displacement signature" from the 2026-06-11 session-3
postmortem.

GROUND-TRUTH DISPROOF (the central regression here): the seq=276 hand
was NOT a displacement. seat3 busted the hand before (all-in 1093, seqs
255-264), making seat3 the dead SMALL BLIND; the BB fell on seat4 (the
visible 100) and seat5 owed NOTHING. Proof chain, all from the log:
  - seq 270-272 (before hero acted): pot 160 = 60 antes + seat4's 100;
    seat4's stack dropped exactly 115 (ante+BB), seat5's exactly 15
    (ante only). A live-SB seat4 cannot have 100 committed before its
    preflop turn.
  - seq 272 decided cleanly: the per-frame dead-SB detector still saw
    seat4's bet == BB and reconstructed correctly. By seq 276 the raise
    to 400 destroyed that evidence -> next-alive-after-dealer default
    -> phantom BB reconstructed on seat5 -> the 3 deltas.
  - post-hand idle frames (seqs 280-281): hero 2720, seat5 1222 — the
    displaced universe's values, with no bets left to displace.
  - next exact-9000 anchor (seq 301): seat5 pre-hand 1467, reachable
    from 1222 (+245 = won a 260 pot over its own commit in the
    in-between hand) and unreachable from the "corrected" 1122.
Chip conservation CANNOT distinguish the two universes (a displacement
is sum-invariant everywhere), so recovery demands positive
blind-structure evidence; the dead-SB guard must REFUSE seq=276, and
recovering it would have fed the model a provably false state.

A true displacement (BB rendered into a seat's stack) shows the
mirror-image hand start: SB visible, BB missing ("sb_only" evidence —
Ignition never plays a dead BB). The synthetic twin below differs from
the live hand only in that evidence frame, and must recover.

Records below are verbatim raw_records from
logs/live_dryrun_20260612_161347.jsonl (seq 269, 270, 276), minus the
Windows-side frame-path key; REC_SB_ONLY is the synthetic live-SB twin
of seq 270 (seat4 = SB 50, stack 3041 = 3106-15-50, pot 110 = 60+50).
"""
from __future__ import annotations

import copy
import random

import pytest

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.live_loop import (
    DecisionCache, make_decision, _attempt_bet_closure_recovery,
)
from src.nlhe.integration.scraper_schema import (
    SessionTracker, parse_frame, is_hand_start, is_preblind_hand_start,
    preblind_pre_hand_stacks, classify_displacement_deltas,
    build_bet_closure_candidate,
)

STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"

_NO_CONTROLS = {
    "present": False, "stable": True, "action_buttons": [],
    "bet_input": {"box": [0, 0, 0, 0], "value": None},
    "slider": {"box": [0, 0, 0, 0], "present": False},
    "shortcuts": [],
}

_BLINDS = ("% 50/100, 15 Ante No Limit Hold'em - Hold'em "
           "(Double-Up Turbo) - TBL#1")

# seq=269 — ante-only pre-blind hand-start (dealer seat1, 50/100/15,
# pot == 4 alive * 15 ante == 60, no bets). Σ(stack + ante) = 9000.
REC_PREBLIND = {
    "blinds": _BLINDS,
    "hero_cards": [], "board": [],
    "pot": {"total": 60, "main": 60},
    "stacks": {"seat1": 2260, "seat2": None, "seat3": None,
               "seat4": 3091, "seat5": 1222, "seat6": 2367},
    "folded": {f"seat{i}": False for i in range(1, 7)},
    "empty": {"seat1": False, "seat2": True, "seat3": True,
              "seat4": False, "seat5": False, "seat6": False},
    "bets": {f"seat{i}": None for i in range(1, 7)},
    "dealer": "seat1",
    "controls": _NO_CONTROLS,
    "table_rendered": True, "suspect": False,
    "captured_at": "20260612_122632_870",
}

# seq=270 — the post-blinds frame of the REAL hand: ONE visible blind,
# bet == BB on seat4, pot 160 = antes + BB. This is "bb_only" posting
# evidence = dead-SB hand (or displaced SB) — recovery must refuse.
REC_BB_ONLY = {
    "blinds": _BLINDS,
    "hero_cards": [], "board": [],
    "pot": {"total": 160, "main": 60},
    "stacks": {"seat1": 2260, "seat2": None, "seat3": None,
               "seat4": 2991, "seat5": 1222, "seat6": 2367},
    "folded": {f"seat{i}": False for i in range(1, 7)},
    "empty": {"seat1": False, "seat2": True, "seat3": True,
              "seat4": False, "seat5": False, "seat6": False},
    "bets": {"seat1": None, "seat2": None, "seat3": None,
             "seat4": 100, "seat5": None, "seat6": None},
    "dealer": "seat1",
    "controls": _NO_CONTROLS,
    "table_rendered": True, "suspect": False,
    "captured_at": "20260612_122634_184",
}

# Synthetic live-SB twin of seq 270: seat4 posted SB 50 (stack 3041 =
# 3106 - 15 - 50, pot 110 = 60 + 50) and seat5's BB is MISSING from
# bet/pot/stack arithmetic — "sb_only" evidence, a true displaced BB.
REC_SB_ONLY = copy.deepcopy(REC_BB_ONLY)
REC_SB_ONLY["pot"]["total"] = 110
REC_SB_ONLY["stacks"]["seat4"] = 3041
REC_SB_ONLY["bets"]["seat4"] = 50
REC_SB_ONLY["captured_at"] = "20260612_122634_999"

# seq=276 — THE incident frame: AcKc facing seat4's raise to 400.
# Scraper: seat5 stack 1222 / bet 0 / pot 560. The phantom-BB recon
# says 1122 / 100 / 660 — a perfect displacement signature, disproven
# by ground truth (see module docstring). In the synthetic sb_only
# universe the identical frame IS a real displacement and the corrected
# values are the truth.
REC_DISPLACED = {
    "blinds": _BLINDS,
    "hero_cards": ["Ac", "Kc"], "board": [],
    "pot": {"total": 560, "main": 60},
    "stacks": {"seat1": 2160, "seat2": None, "seat3": None,
               "seat4": 2691, "seat5": 1222, "seat6": 2367},
    "folded": {f"seat{i}": False for i in range(1, 7)},
    "empty": {"seat1": False, "seat2": True, "seat3": True,
              "seat4": False, "seat5": False, "seat6": False},
    "bets": {"seat1": 100, "seat2": None, "seat3": None,
             "seat4": 400, "seat5": None, "seat6": None},
    "dealer": "seat1",
    "controls": {
        "present": True, "stable": True,
        "action_buttons": [
            {"label": "CALL", "slot": "center",
             "box": [0.515, 0.775, 0.66, 0.857], "amount": 300},
            {"label": "FOLD", "slot": "left",
             "box": [0.366, 0.775, 0.505, 0.857], "amount": None},
            {"label": "RAISE", "slot": "right",
             "box": [0.668, 0.775, 0.8, 0.857], "amount": 700},
        ],
        "bet_input": {"box": [0.705, 0.876, 0.812, 0.922], "value": 700},
        "slider": {"box": [0.43, 0.886, 0.672, 0.916], "present": True},
        "shortcuts": [],
    },
    "table_rendered": True, "suspect": False,
    "captured_at": "20260612_122700_038",
}


@pytest.fixture(scope="module")
def structure():
    return TournamentStructure.from_yaml(STRUCTURE_YAML)


def _p2_tracker():
    return SessionTracker(anchor_sum_floor=True, bet_closure_recovery=True)


def _observe(tracker, *records):
    for rec in records:
        tracker.observe(parse_frame(rec))
    return tracker


@pytest.fixture
def dead_sb_tracker():
    """The REAL session state at seq 276: pre-blind anchor + the
    bb_only (dead-SB) posting evidence from seq 270."""
    tracker = _observe(_p2_tracker(), REC_PREBLIND, REC_BB_ONLY)
    assert tracker._preblind_pre_hand is not None
    assert tracker._blind_evidence == "bb_only"
    return tracker


@pytest.fixture
def displaced_tracker():
    """The synthetic true-displacement universe: pre-blind anchor +
    sb_only posting evidence (BB missing at the post moment)."""
    tracker = _observe(_p2_tracker(), REC_PREBLIND, REC_SB_ONLY)
    assert tracker._preblind_pre_hand is not None
    assert tracker._blind_evidence == "sb_only"
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


def _decide(record, structure, tracker, cache=None, flag=True):
    return make_decision(record, structure, solver=None, tracker=tracker,
                         rng=random.Random(42), mode="sample", seq=1,
                         decision_cache=cache, bet_closure_recovery=flag)


# ── gating: P2 requires P1; default path byte-identical ──────────────────

def test_tracker_requires_p1_floor():
    with pytest.raises(ValueError, match="gated on P1"):
        SessionTracker(bet_closure_recovery=True)


def test_flag_off_drop_is_unannotated(structure, stub_policy):
    """Default (flag OFF): the displaced frame drops with the exact
    pre-P2 reason — no recovery, no annotation."""
    tracker = SessionTracker()
    _observe(tracker, REC_PREBLIND, REC_SB_ONLY)
    d = _decide(REC_DISPLACED, structure, tracker, flag=False)
    assert d.status == "safe_fold"
    assert d.skip_reason == "invariant_fail (3 deltas)"
    assert d.recovered_fields is None


def test_flag_off_tracker_captures_nothing():
    tracker = SessionTracker(anchor_sum_floor=True)
    _observe(tracker, REC_PREBLIND, REC_SB_ONLY)
    assert tracker._preblind_pre_hand is None
    assert tracker._blind_evidence is None


def test_p1_not_armed_refuses(structure, stub_policy):
    """Defensive double-gate: even if the flag is passed to
    make_decision, a tracker without the P1 floor refuses."""
    d = _decide(REC_DISPLACED, structure, SessionTracker())
    _assert_refused(d, "anchor sum-floor guard not armed")


# ── the seq=276 regression: ground truth says dead-SB -> REFUSE ──────────

def test_seq276_dead_sb_hand_refuses(structure, dead_sb_tracker,
                                     stub_policy):
    """The named live fixture. The displacement signature matches, the
    pre-blind anchor exists, conservation would close — and the hand was
    a dead-SB hand (ground-truth chain in the module docstring), so the
    'corrected' state is false. The dead-SB guard must refuse."""
    d = _decide(REC_DISPLACED, structure, dead_sb_tracker)
    _assert_refused(d, "BB-only posting (dead-SB hand or displaced SB)")


def test_dead_sb_guard_beats_missing_evidence(structure, stub_policy):
    """Pre-blind anchor but NO posting evidence at all (scraper gap):
    refuse — the recon's blind assignment is unverifiable."""
    tracker = _observe(_p2_tracker(), REC_PREBLIND)
    d = _decide(REC_DISPLACED, structure, tracker)
    _assert_refused(d, "no blind-structure evidence")


# ── the true-displacement twin: must recover ─────────────────────────────

def test_true_displacement_recovers(structure, displaced_tracker,
                                    stub_policy):
    d = _decide(REC_DISPLACED, structure, displaced_tracker)
    assert d.status == "decision_recovered"
    assert d.recovered_fields == [
        "stack.seat5=1122 (bet_closure)",
        "bet.seat5=100 (bet_closure)",
        "pot=660 (bet_closure:preblind_anchor)",
    ]
    assert d.pot_total == 660
    assert d.hero_stack == 2160
    assert d.facing_bet is True
    assert d.client_action is not None


def test_recovered_decision_caches(structure, displaced_tracker,
                                   stub_policy):
    cache = DecisionCache()
    d1 = _decide(REC_DISPLACED, structure, displaced_tracker, cache)
    d2 = _decide(REC_DISPLACED, structure, displaced_tracker, cache)
    assert d1.status == "decision_recovered"
    assert d2.status == "decision_recovered_cached"
    assert d1.client_action == d2.client_action


def test_recovery_matches_clean_twin_action(structure, stub_policy):
    """The recovered frame must produce the SAME decision the clean frame
    with the true split produces — recovery reconstructs, never invents."""
    clean = copy.deepcopy(REC_DISPLACED)
    clean["stacks"]["seat5"] = 1122
    clean["bets"]["seat5"] = 100
    clean["pot"]["total"] = 660

    t1 = _observe(_p2_tracker(), REC_PREBLIND, REC_SB_ONLY)
    t2 = _observe(_p2_tracker(), REC_PREBLIND, REC_SB_ONLY)
    d_clean = _decide(clean, structure, t1)
    d_rec = _decide(REC_DISPLACED, structure, t2)
    assert d_clean.status == "decision"
    assert d_rec.status == "decision_recovered"
    assert d_rec.client_action == d_clean.client_action
    assert d_rec.pot_total == d_clean.pot_total == 660


def test_recovery_does_not_mutate_tracker(structure, displaced_tracker,
                                          stub_policy):
    pre = displaced_tracker._preblind_pre_hand
    key = displaced_tracker._preblind_key
    regular = displaced_tracker._current_pre_hand
    evidence = displaced_tracker._blind_evidence
    d = _decide(REC_DISPLACED, structure, displaced_tracker)
    assert d.status == "decision_recovered"
    assert displaced_tracker._preblind_pre_hand == pre
    assert displaced_tracker._preblind_key == key
    assert displaced_tracker._current_pre_hand == regular
    assert displaced_tracker._blind_evidence == evidence


# ── refusal classes: every gate failure drops exactly as before ──────────

def _assert_refused(d, why_fragment, n_deltas=3):
    assert d.status == "safe_fold"
    assert d.skip_reason.startswith(f"invariant_fail ({n_deltas} deltas)")
    assert "bet-closure recovery declined" in d.skip_reason
    assert why_fragment in d.skip_reason
    assert d.recovered_fields is None
    assert d.client_action is None
    assert d.invariant_deltas is not None


def test_no_anchor_refuses(structure, stub_policy):
    """Armed flags but no clean hand-start of any kind observed."""
    d = _decide(REC_DISPLACED, structure, _p2_tracker())
    _assert_refused(d, "no clean hand-start anchor")


def test_guard_refused_anchor_refuses(structure, stub_policy):
    """A pre-blind hand-start whose chip sum breaks the ceiling guard is
    REFUSED and sticky — recovery must surface the refusal, never use it."""
    bad = copy.deepcopy(REC_PREBLIND)
    bad["stacks"]["seat4"] = 13091  # stuck-digit class; sum 19000 > 9000
    tracker = _observe(_p2_tracker(), bad, REC_SB_ONLY)
    assert tracker._preblind_pre_hand is None
    assert tracker._preblind_refused_key is not None
    d = _decide(REC_DISPLACED, structure, tracker)
    _assert_refused(d, "refused by the anchor guard")


def test_preblind_sum_floor_refusal_all_alive():
    """P1 sum-floor semantics carry over: a 6-alive pre-blind frame whose
    sum falls short of chips-in-play (dropped-digit class) is refused."""
    rec = copy.deepcopy(REC_PREBLIND)
    rec["empty"] = {f"seat{i}": False for i in range(1, 7)}
    rec["stacks"] = {"seat1": 1485, "seat2": 1485, "seat3": 1485,
                     "seat4": 1485, "seat5": 1485, "seat6": 685}  # short
    rec["pot"]["total"] = 90  # 6 alive * 15
    tracker = _p2_tracker()
    tracker.observe(parse_frame(rec))
    assert tracker._preblind_pre_hand is None
    assert tracker._preblind_refused_key is not None


def test_extra_stack_corruption_caught_by_conservation(
        structure, displaced_tracker, stub_policy):
    """Stack off by MORE than the displacement: the simple-model recon
    adapts (deltas still close, recon 1132), so the anchor-conservation
    check is the layer that must refuse — and does."""
    rec = copy.deepcopy(REC_DISPLACED)
    rec["stacks"]["seat5"] = 1232
    d = _decide(rec, structure, displaced_tracker)
    _assert_refused(d, "no candidate passed replay+invariant")
    assert "chip conservation" in d.skip_reason


def test_second_seat_corruption_caught_by_conservation(
        structure, displaced_tracker, stub_policy):
    """A second seat's stack also off (+50 on seat6): the simple-model
    recon takes that stack at face value (no extra delta), so only the
    independent conservation check against the anchor can see the 50
    phantom chips — refuse, annotated. (The multi-seat DELTA refusal
    class is covered at classifier level below.)"""
    rec = copy.deepcopy(REC_DISPLACED)
    rec["stacks"]["seat6"] = 2417
    d = _decide(rec, structure, displaced_tracker)
    _assert_refused(d, "no candidate passed replay+invariant")
    assert "chip conservation" in d.skip_reason


def test_conservation_failure_refuses(structure, stub_policy):
    """Anchor disagreeing with the corrected frame's chip total: the
    independent post-patch conservation check must refuse."""
    bad = copy.deepcopy(REC_PREBLIND)
    bad["stacks"]["seat1"] = 2210  # -50: Σpre = 8950; not all alive ->
    tracker = _observe(_p2_tracker(), bad, REC_SB_ONLY)  # floor exempt
    assert tracker._preblind_pre_hand is not None
    d = _decide(REC_DISPLACED, structure, tracker)
    _assert_refused(d, "no candidate passed replay+invariant")
    assert "chip conservation" in d.skip_reason


def test_wrong_anchor_rejected_by_revalidation(structure, stub_policy):
    """A sum-preserving but per-seat-wrong anchor passes conservation;
    the replay + invariant re-validation must reject it. This is the
    never-feed-the-model-a-wrong-derivation property."""
    bad = copy.deepcopy(REC_PREBLIND)
    bad["stacks"]["seat1"] = 2210   # -50
    bad["stacks"]["seat6"] = 2417   # +50 (sum still 9000)
    tracker = _observe(_p2_tracker(), bad, REC_SB_ONLY)
    assert tracker._preblind_pre_hand is not None
    d = _decide(REC_DISPLACED, structure, tracker)
    _assert_refused(d, "no candidate passed replay+invariant")


def test_ambiguous_dual_anchors_refuse(structure, monkeypatch):
    """Regular and pre-blind anchors BOTH present with different pre-hand
    stacks; force re-validation to approve both: arbitration must refuse
    to pick (more than one candidate closure). Exercised at the
    _attempt_bet_closure_recovery level — patching the replay/invariant
    module attributes would also stub make_decision's own step-5/6 calls
    and the frame would never reach the recovery path."""
    tracker = _observe(_p2_tracker(), REC_PREBLIND, REC_SB_ONLY)
    # Plant a conflicting regular anchor under the same hand-key
    # (sum-preserving so build_bet_closure_candidate accepts both).
    pre = list(tracker._preblind_pre_hand)
    pre[0] -= 50
    pre[5] += 50
    tracker._current_pre_hand = tuple(pre)
    tracker._current_hand_key = tracker._preblind_key
    anchors, _ = tracker.bet_closure_anchors_for(parse_frame(REC_DISPLACED))
    assert len(anchors) == 2

    import src.nlhe.integration.replay as replay_mod
    import src.nlhe.integration.invariant as inv_mod

    class _Pack:
        street_idx = 0

    class _Inv:
        ok = True
        deltas = []

    monkeypatch.setattr(replay_mod, "replay_to_decision",
                        lambda *a, **k: _Pack())
    monkeypatch.setattr(inv_mod, "check_mid_hand_invariant",
                        lambda *a, **k: _Inv())

    class _InvFail:
        deltas = [("pot", 560, 660), ("stack[seat5]", 1222, 1122),
                  ("bet[seat5]", 0, 100)]

    frame, pack, recovered, why = _attempt_bet_closure_recovery(
        parse_frame(REC_DISPLACED), _InvFail(), structure, tracker)
    assert frame is None and pack is None and recovered is None
    assert "ambiguous" in why


def test_non_signature_invariant_fail_is_untouched(structure,
                                                   displaced_tracker,
                                                   stub_policy):
    """An invariant failure that is NOT the displacement family (an
    additional scraper_self_consistency:pot delta — foreign type) must
    drop with the exact pre-P2 reason — no annotation at all, even with
    the flag armed."""
    rec = copy.deepcopy(REC_DISPLACED)
    rec["pot"]["total"] = 510  # breaks the frame's own pot arithmetic
    d = _decide(rec, structure, displaced_tracker)
    assert d.status == "safe_fold"
    assert d.skip_reason.startswith("invariant_fail (")
    assert "bet-closure" not in d.skip_reason
    assert d.recovered_fields is None


# ── helper-level checks ──────────────────────────────────────────────────

def test_preblind_hand_start_detection():
    f = parse_frame(REC_PREBLIND)
    assert is_preblind_hand_start(f) is True
    assert is_hand_start(f) is False           # no blinds visible yet
    assert preblind_pre_hand_stacks(f) == (2275, 0, 0, 3106, 1237, 2382)
    # Neither posting frame nor the decision frame is a pre-blind start.
    assert is_preblind_hand_start(parse_frame(REC_BB_ONLY)) is False
    assert is_preblind_hand_start(parse_frame(REC_DISPLACED)) is False


def test_blind_evidence_capture_patterns():
    # bb_only from the real seq-270 frame.
    t = _observe(_p2_tracker(), REC_BB_ONLY)
    assert t._blind_evidence == "bb_only"
    # sb_only from the synthetic twin.
    t = _observe(_p2_tracker(), REC_SB_ONLY)
    assert t._blind_evidence == "sb_only"
    # First capture wins within a hand.
    t = _observe(_p2_tracker(), REC_BB_ONLY, REC_SB_ONLY)
    assert t._blind_evidence == "bb_only"
    # Pot not matching antes + blind -> no evidence (mid-hand frame).
    rec = copy.deepcopy(REC_BB_ONLY)
    rec["pot"]["total"] = 260
    t = _observe(_p2_tracker(), rec)
    assert t._blind_evidence is None
    # Two visible bets -> no evidence.
    rec = copy.deepcopy(REC_BB_ONLY)
    rec["bets"]["seat5"] = 100
    t = _observe(_p2_tracker(), rec)
    assert t._blind_evidence is None


def test_classify_displacement_deltas():
    fixture = [("pot", 560, 660), ("stack[seat5]", 1222, 1122),
               ("bet[seat5]", 0, 100)]
    assert classify_displacement_deltas(fixture) == ("closure", (4, 100))
    # Reverse direction ("vice versa"): stack low / bet+pot high.
    rev = [("pot", 660, 560), ("stack[seat5]", 1122, 1222),
           ("bet[seat5]", 100, 0)]
    assert classify_displacement_deltas(rev) == ("closure", (4, -100))
    # Foreign delta type -> not the signature.
    kind, _ = classify_displacement_deltas(
        fixture + [("street_idx", 0, 1)])
    assert kind == "not_signature"
    # Pot-only -> not the signature.
    assert classify_displacement_deltas(
        [("pot", 560, 660)]) == ("not_signature", None)
    # Two seats -> refused.
    kind, why = classify_displacement_deltas(
        fixture + [("stack[seat6]", 2417, 2367)])
    assert kind == "refused" and "2 seats" in why
    # Missing pot delta -> amounts can't close.
    kind, why = classify_displacement_deltas(
        [("stack[seat5]", 1222, 1122), ("bet[seat5]", 0, 100)])
    assert kind == "refused" and "do not close" in why


def test_build_bet_closure_candidate_bounds():
    frame = parse_frame(REC_DISPLACED)
    anchor = (2275, 0, 0, 3106, 1237, 2382)
    cand, why = build_bet_closure_candidate(frame, 4, 100, anchor)
    assert why == "ok"
    assert cand.stack[4] == 1122 and cand.bet[4] == 100
    assert cand.pot_total == 660
    # Every other field bit-identical.
    assert cand.stack[:4] + cand.stack[5:] == \
        frame.stack[:4] + frame.stack[5:]
    assert cand.bet[:4] + cand.bet[5:] == frame.bet[:4] + frame.bet[5:]
    assert cand.board == frame.board
    # Seat not at hand start per the anchor -> refused.
    cand, why = build_bet_closure_candidate(
        frame, 4, 100, (2275, 0, 0, 3106, 0, 3619))
    assert cand is None and "not at hand start" in why
    # Out-of-range corrected stack (transfer larger than the stack).
    cand, why = build_bet_closure_candidate(frame, 4, 1300, anchor)
    assert cand is None and "out of range" in why
    # Conservation breach (anchor total != corrected total).
    cand, why = build_bet_closure_candidate(
        frame, 4, 100, (2225, 0, 0, 3106, 1237, 2382))
    assert cand is None and "chip conservation" in why


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
