"""D2 dead-button position handling — gate + regression tests.

Built from the 2026-06-12 live session (logs/live_dryrun_20260612_230149
.jsonl), the session-5 #1 hand-killer skip class: the bridge rejected
every frame of hands where the scraped dealer button sat on an
eliminated/empty seat. Windows attribution proved these are CORRECTLY
scraped DEAD BUTTONS — standard short-handed turbo rotation.

THE GOVERNING RULE (poker-rules correctness bar): the dead-button rule
is NOT "SB/BB = next actives clockwise of the button". The invariant is
that the BIG BLIND advances exactly one ACTIVE player per hand; the
small blind and the button derive from it:
  - dead-button hand: the previous SB player busted -> button sits on
    their vacated seat; SB posted by the previous BB player (or dead
    too); BB on the next active.
  - dead-SB hand: the previous BB player busted -> button on an active
    player, SB seat empty, NO SB posted (Ignition never plays a dead
    BB; it does play dead SBs).
  - double elimination: dead button AND dead SB in the same hand — the
    only post is the BB, one active player past the previous BB.
The scraped frames contain the actual posted blinds, so the derivation
is: dealer_seat from the scrape (possibly dead), SB/BB VALIDATED
against the observed posts (Predicate 1 in
scraper_schema._derive_blinds_and_action_order); the rule's clockwise
walk fills in only what is unobservable.

Exhibits (verbatim raw_records in tests/fixtures/dead_button_exhibits
.json, minus the Windows-side frame-path key):
  - seq 912 (QdKc): blinds 75/150/25, dealer on EMPTY seat2, alive
    seats 1/4/5/6, posts SB=seat4 (75) BB=seat5 (150), hero to act
    facing the BB. Whole hand previously unread.
  - seq 1366 (TsAc): blinds 50/100/15, dealer on seat5 (busted the
    previous hand), alive seats 1-4, posts SB=seat1=hero (50)
    BB=seat2 (100), hero to act facing the BB.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.live_loop import DecisionCache, make_decision
from src.nlhe.integration.scraper_schema import (
    ScraperDataQuality, SessionTracker, _derive_blinds_and_action_order,
    derive_action_sequence, is_hand_start, parse_frame, pre_hand_stacks,
    NUM_SEATS,
)

STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"
FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "dead_button_exhibits.json")
    .read_text())

QDKC_SB_POSTED = FIXTURES["qdkc_sb_posted"]["raw_record"]      # seq 910
QDKC_DECISION = FIXTURES["qdkc_decision"]["raw_record"]        # seq 912
TSAC_HAND_START = FIXTURES["tsac_hand_start"]["raw_record"]    # seq 1363
TSAC_DECISION = FIXTURES["tsac_decision"]["raw_record"]        # seq 1366
TSAC_RERAISE = FIXTURES["tsac_reraise_decision"]["raw_record"]  # seq 1370


@pytest.fixture(scope="module")
def structure():
    return TournamentStructure.from_yaml(STRUCTURE_YAML)


@pytest.fixture
def stub_policy(monkeypatch):
    """make_decision imports _sample_action_from_policy at call time —
    stub it so no solver checkpoint is needed. chip_int=1 = CALL/CHECK."""
    import scripts.eval_6max_self_play as ev

    def _stub(solver, parsed, state, rng, mode="sample", policy_filter=None):
        return 1

    monkeypatch.setattr(ev, "_sample_action_from_policy", _stub)
    return _stub


# ── flag OFF: byte-identical pre-D2 drops ────────────────────────────────

def test_flag_off_drops_both_exhibits():
    for rec in (QDKC_DECISION, TSAC_DECISION):
        with pytest.raises(ScraperDataQuality) as ei:
            parse_frame(rec)
        assert "dealer points to seat" in str(ei.value)


def test_flag_off_ignores_additive_dealer_dead_key():
    """Windows adds `dealer_dead` to the schema; flag-OFF parse must
    tolerate (= ignore) it: same drop with the key present."""
    rec = copy.deepcopy(QDKC_DECISION)
    rec["dealer_dead"] = True
    with pytest.raises(ScraperDataQuality):
        parse_frame(rec)


def _alive_dealer_record():
    """TsAc hand-start rewritten with the button on ALIVE seat4 — the
    same chips, a live dealer (for flag-no-op checks)."""
    rec = copy.deepcopy(TSAC_HAND_START)
    rec["dealer"] = "seat4"
    return rec


def test_flag_off_alive_dealer_unchanged():
    """An alive-dealer frame parses identically with and without the
    flag, dealer_dead False both ways."""
    rec = _alive_dealer_record()
    f_off = parse_frame(rec)
    f_on = parse_frame(rec, dead_button_handling=True)
    assert f_off == f_on
    assert f_off.dealer_dead is False


# ── flag ON: dead-button frames parse ────────────────────────────────────

def test_flag_on_parses_qdkc():
    f = parse_frame(QDKC_DECISION, dead_button_handling=True)
    assert f.dealer_dead is True
    assert f.dealer_seat == 1                  # seat2, vacated
    assert f.alive == (True, False, False, True, True, True)
    assert f.controls_present is True


def test_flag_on_parses_tsac():
    f = parse_frame(TSAC_DECISION, dead_button_handling=True)
    assert f.dealer_dead is True
    assert f.dealer_seat == 4                  # seat5, busted prior hand
    assert f.alive == (True, True, True, True, False, False)


def test_additive_key_preferred_true():
    rec = copy.deepcopy(QDKC_DECISION)
    rec["dealer_dead"] = True
    f = parse_frame(rec, dead_button_handling=True)
    assert f.dealer_dead is True


def test_additive_key_false_contradiction_drops():
    """Scraper positively says NOT a dead button while the seat reads
    empty -> OCR-drift class; keep the pre-D2 drop even flag-ON."""
    rec = copy.deepcopy(QDKC_DECISION)
    rec["dealer_dead"] = False
    with pytest.raises(ScraperDataQuality):
        parse_frame(rec, dead_button_handling=True)


def test_additive_key_absent_derives_from_seat():
    assert "dealer_dead" not in QDKC_DECISION
    f = parse_frame(QDKC_DECISION, dead_button_handling=True)
    assert f.dealer_dead is True


def test_additive_key_true_on_alive_dealer_is_annotation_only():
    """dealer_dead=true reported against an alive-reading dealer seat:
    trust the marker; blind derivation (clockwise walk from dealer+1)
    is unchanged either way."""
    rec = _alive_dealer_record()
    rec["dealer_dead"] = True
    f = parse_frame(rec, dead_button_handling=True)
    assert f.dealer_dead is True
    base = parse_frame(_alive_dealer_record(), dead_button_handling=True)
    alive_seats = [i for i in range(NUM_SEATS) if f.alive[i]]
    assert (_derive_blinds_and_action_order(f.dealer_seat, alive_seats,
                                            frame=f)
            == _derive_blinds_and_action_order(base.dealer_seat,
                                               alive_seats, frame=base))


# ── blind placement: observed posts are ground truth ─────────────────────

def test_qdkc_blinds_match_observed_posts():
    f = parse_frame(QDKC_DECISION, dead_button_handling=True)
    alive_seats = [i for i in range(NUM_SEATS) if f.alive[i]]
    sb, bb, pf_alive, _, _ = _derive_blinds_and_action_order(
        f.dealer_seat, alive_seats, frame=f)
    assert (sb, bb) == (3, 4)                  # seat4 posted 75, seat5 150
    assert f.bet[sb] == f.blinds.sb == 75
    assert f.bet[bb] == f.blinds.bb == 150
    # UTG = next active after BB: seat6 (idx 5), then hero (idx 0)
    assert pf_alive == [5, 0, 3, 4]


def test_tsac_blinds_match_observed_posts():
    f = parse_frame(TSAC_DECISION, dead_button_handling=True)
    alive_seats = [i for i in range(NUM_SEATS) if f.alive[i]]
    sb, bb, pf_alive, _, _ = _derive_blinds_and_action_order(
        f.dealer_seat, alive_seats, frame=f)
    assert (sb, bb) == (0, 1)                  # hero posted SB 50, seat2 BB 100
    assert f.bet[sb] == f.blinds.sb == 50
    assert f.bet[bb] == f.blinds.bb == 100
    assert pf_alive == [2, 3, 0, 1]


# ── the governing rotation invariant (synthetic) ─────────────────────────
#
# Each case constructs hand t (pre-elimination) and hand t+1 (scraped
# dealer per the standard rule) and asserts BB(t+1) is exactly ONE
# active player clockwise of BB(t) — never skipping, never repeating —
# with the posts as ground truth.

def _frame_stub(dealer_seat, alive_idx, bets, sb=50, bb=100, ante=15):
    """Minimal raw record -> parsed frame for derivation tests."""
    rec = {
        "blinds": f"% {sb}/{bb}, {ante} Ante No Limit Hold'em",
        "hero_cards": [], "board": [],
        "pot": {"total": ante * len(alive_idx) + sum(bets.values())},
        "stacks": {f"seat{i+1}": (1500 if i in alive_idx else None)
                   for i in range(NUM_SEATS)},
        "bets": {f"seat{i+1}": bets.get(i) for i in range(NUM_SEATS)},
        "folded": {f"seat{i+1}": False for i in range(NUM_SEATS)},
        "empty": {f"seat{i+1}": (i not in alive_idx)
                  for i in range(NUM_SEATS)},
        "dealer": f"seat{dealer_seat+1}",
        "controls": {"present": False},
        "suspect": False,
        "captured_at": "synthetic",
    }
    return parse_frame(rec, dead_button_handling=True)


def _next_active(seat, alive_idx):
    for off in range(1, NUM_SEATS + 1):
        c = (seat + off) % NUM_SEATS
        if c in alive_idx:
            return c
    raise AssertionError("no active seat")


def test_dead_button_single_elimination_rotation():
    """Hand t: actives {0,1,2,3,4}, dealer=0, SB=1, BB=2. Seat1 (the SB
    player) busts. Hand t+1: button DEAD on seat1; SB posted by the
    previous BB player (seat2); BB advances one active to seat3."""
    prev_bb = 2
    alive_t1 = [0, 2, 3, 4]
    f = _frame_stub(dealer_seat=1, alive_idx=alive_t1,
                    bets={2: 50, 3: 100})
    assert f.dealer_dead is True
    sb, bb, _, _, _ = _derive_blinds_and_action_order(
        f.dealer_seat, alive_t1, frame=f)
    assert (sb, bb) == (2, 3)
    assert bb == _next_active(prev_bb, alive_t1)   # BB advanced exactly one


def test_dead_sb_rotation():
    """Hand t: actives {0,1,2,3,4}, dealer=0, SB=1, BB=2. Seat2 (the BB
    player) busts. Hand t+1: button LIVE on seat1 (previous SB player),
    SB DEAD (nobody posts), BB advances one active to seat3."""
    prev_bb = 2
    alive_t1 = [0, 1, 3, 4]
    f = _frame_stub(dealer_seat=1, alive_idx=alive_t1, bets={3: 100})
    assert f.dealer_dead is False                  # button on an active seat
    sb, bb, _, _, _ = _derive_blinds_and_action_order(
        f.dealer_seat, alive_t1, frame=f)
    assert sb is None                              # dead SB: no post
    assert bb == 3 == _next_active(prev_bb, alive_t1)
    # pot arithmetic confirms the dead SB: antes + BB only
    assert f.pot_total == 4 * 15 + 100


def test_double_elimination_dead_button_and_dead_sb():
    """Hand t: actives {0,1,2,3,4,5}, dealer=0, SB=1, BB=2. BOTH blind
    players bust. Hand t+1: button DEAD on seat1, SB DEAD (seat2's
    player gone), the only post is the BB one active past seat2 -> 3.
    The naive 'SB/BB = next two actives clockwise of the button' would
    wrongly assign SB=3 BB=4; the posts refute it."""
    prev_bb = 2
    alive_t1 = [0, 3, 4, 5]
    f = _frame_stub(dealer_seat=1, alive_idx=alive_t1, bets={3: 100})
    assert f.dealer_dead is True
    sb, bb, _, _, _ = _derive_blinds_and_action_order(
        f.dealer_seat, alive_t1, frame=f)
    assert sb is None
    assert bb == 3 == _next_active(prev_bb, alive_t1)
    assert (sb, bb) != (3, 4)                      # naive rule rejected


def test_bb_advances_one_active_across_exhibit_hands():
    """Real-log invariant check: in the TsAc hand the dealer is dead on
    seat5 (busted in the dealer=seat4 hand just before, where seat5 had
    posted the SB). BB lands on seat2 — exactly one active player past
    the previous hand's BB (seat1... the previous hand's posts are not
    in the fixtures, so assert the within-hand consistency instead:
    SB poster (seat1) is exactly the active seat between the dead
    button and the BB, and no active seat between BB and button posted
    nothing)."""
    f = parse_frame(TSAC_DECISION, dead_button_handling=True)
    alive_seats = [i for i in range(NUM_SEATS) if f.alive[i]]
    sb, bb, _, _, _ = _derive_blinds_and_action_order(
        f.dealer_seat, alive_seats, frame=f)
    # walk button -> BB: every active seat in between must be the SB
    between = []
    c = f.dealer_seat
    while True:
        c = (c + 1) % NUM_SEATS
        if c == bb:
            break
        if c in alive_seats:
            between.append(c)
    assert between == [sb]


# ── end-to-end: replay + invariant + decision on the exhibits ────────────

def _replay_and_check(record, structure):
    from src.nlhe.integration.replay import replay_to_decision
    from src.nlhe.integration.invariant import check_mid_hand_invariant
    f = parse_frame(record, dead_button_handling=True)
    pack = replay_to_decision(f, structure, pre_hand_override=None)
    inv = check_mid_hand_invariant(f, pack)
    return f, pack, inv


def test_qdkc_replays_to_hero_invariant_pass(structure):
    f, pack, inv = _replay_and_check(QDKC_DECISION, structure)
    assert inv.ok, inv.deltas
    assert (pack.sb_seat, pack.bb_seat) == (3, 4)
    assert pack.state.current_player() == f.hero_seat
    assert pack.street_idx == 0


def test_tsac_replays_to_hero_invariant_pass(structure):
    f, pack, inv = _replay_and_check(TSAC_DECISION, structure)
    assert inv.ok, inv.deltas
    assert (pack.sb_seat, pack.bb_seat) == (0, 1)
    assert pack.state.current_player() == f.hero_seat
    assert pack.street_idx == 0


def test_hand_start_anchor_on_dead_button_hand():
    """seq 1363 (TsAc hand-start, dealer dead on seat5) must anchor:
    is_hand_start fires on the SB+BB posts, pre-hand sum closes to the
    9000 chips-in-play exactly."""
    f = parse_frame(TSAC_HAND_START, dead_button_handling=True)
    assert is_hand_start(f)
    tracker = SessionTracker(anchor_sum_floor=True)
    assert tracker.observe(f) is True
    pre = pre_hand_stacks(f)
    assert sum(pre) == 9000
    assert pre == (3590, 1023, 2042, 2345, 0, 0)
    f2 = parse_frame(TSAC_DECISION, dead_button_handling=True)
    assert tracker.pre_hand_for(f2) == pre


def test_make_decision_flag_off_unchanged(structure, stub_policy):
    tracker = SessionTracker()
    d = make_decision(QDKC_DECISION, structure, solver=None,
                      tracker=tracker, rng=random.Random(42),
                      mode="sample", seq=912,
                      decision_cache=DecisionCache())
    assert d.status == "skip_data_quality"
    assert "dealer points to seat2" in d.skip_reason


def test_make_decision_qdkc_counterfactual(structure, stub_policy):
    tracker = SessionTracker()
    d = make_decision(QDKC_DECISION, structure, solver=None,
                      tracker=tracker, rng=random.Random(42),
                      mode="sample", seq=912,
                      decision_cache=DecisionCache(),
                      dead_button_handling=True)
    assert d.status == "decision"
    assert d.pre_hand_override_used is True       # anchored from this frame
    assert d.client_action["kind"] == "call"      # stub chip_int=1, facing BB
    assert d.invariant_deltas is None


def test_make_decision_tsac_counterfactual(structure, stub_policy):
    tracker = SessionTracker()
    cache = DecisionCache()
    rng = random.Random(42)
    d0 = make_decision(TSAC_HAND_START, structure, solver=None,
                       tracker=tracker, rng=rng, mode="sample", seq=1363,
                       decision_cache=cache, dead_button_handling=True)
    assert d0.status == "skip_not_hero_to_act"    # hand-start, anchors only
    d = make_decision(TSAC_DECISION, structure, solver=None,
                      tracker=tracker, rng=rng, mode="sample", seq=1366,
                      decision_cache=cache, dead_button_handling=True)
    assert d.status == "decision"
    assert d.pre_hand_override_used is True
    assert d.client_action["kind"] == "call"
    d2 = make_decision(TSAC_DECISION, structure, solver=None,
                       tracker=tracker, rng=rng, mode="sample", seq=1367,
                       decision_cache=cache, dead_button_handling=True)
    assert d2.status == "decision_cached"         # decide-once holds


def test_make_decision_tsac_reraise_counterfactual(structure, stub_policy):
    """seq 1370: seat2's 1008 all-in re-opens the action — a NEW
    decision identity on the same dead-button hand must re-sample."""
    tracker = SessionTracker()
    cache = DecisionCache()
    rng = random.Random(42)
    make_decision(TSAC_HAND_START, structure, solver=None, tracker=tracker,
                  rng=rng, mode="sample", seq=1363, decision_cache=cache,
                  dead_button_handling=True)
    d1 = make_decision(TSAC_DECISION, structure, solver=None,
                       tracker=tracker, rng=rng, mode="sample", seq=1366,
                       decision_cache=cache, dead_button_handling=True)
    d2 = make_decision(TSAC_RERAISE, structure, solver=None,
                       tracker=tracker, rng=rng, mode="sample", seq=1370,
                       decision_cache=cache, dead_button_handling=True)
    assert d1.status == "decision"
    assert d2.status == "decision"                # not cached: new identity
    assert d2.decision_identity != d1.decision_identity


# ── action-sequence sanity on the exhibits ───────────────────────────────

def test_qdkc_action_sequence():
    f = parse_frame(QDKC_DECISION, dead_button_handling=True)
    # UTG=seat6 folded; hero (seat1) next — stop at hero's decision.
    assert derive_action_sequence(f) == [(5, 0)]


def test_tsac_action_sequence():
    f = parse_frame(TSAC_DECISION, dead_button_handling=True)
    # UTG=seat3 fold, seat4 fold, dead seats 5/6 forced-fold, stop at hero.
    assert derive_action_sequence(f) == [(2, 0), (3, 0), (4, 0), (5, 0)]
