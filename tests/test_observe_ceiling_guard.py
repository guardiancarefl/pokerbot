"""observe() chips-in-play ceiling guard — test9907 fixture tests.

Pins the fix for the confirmed live layer-1 defect (2026-06-09,
docs/OBSERVE_CEILING_GUARD_DESIGN.md): SessionTracker.observe() accepted
two poisoned hand-starts (seat6 stuck-digit 9907, sum(pre)=17917 — its
self-consistency check passes because both sides use the same wrong
stack), and layer-1 recovery then consumed the poisoned anchor 7 times,
producing decision_recovered outputs with phantom seat6 stacks
(9897/9822x5/9907) that the invariant validated self-consistently.

Guard (approved Q1-Q3): refuse a new hand-start anchor when any per-seat
pre-hand stack OR the pre-hand sum exceeds the config-derived
chips-in-play ceiling (STARTING_CHIPS x NUM_SEATS = 9000). Upper bounds
only — legitimate under-read hand-starts (mis-alive hidden chips, 30/178
in the corpus) must keep anchoring. Strict `>`.

Real frames are loaded from the committed fixture streams under
tools/scraper_sanity_fixture/streams/ (keyed by captured_at — seq
numbers collide across sender reconnects).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.live_loop import make_decision
from src.nlhe.integration.scraper_schema import (
    NUM_SEATS, STARTING_CHIPS,
    BlindsLevel, ScraperFrame, SessionTracker,
    ScraperDataQuality, ScraperParseError, ScraperSuspect,
    is_hand_start, parse_frame, pre_hand_stacks,
)

REPO = Path(__file__).resolve().parent.parent
STREAMS = REPO / "tools" / "scraper_sanity_fixture" / "streams"
STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"
CEILING = STARTING_CHIPS * NUM_SEATS  # 9000

# The two poisoned hand-starts (must-refuse) and their stream files.
POISONED = [
    ("live_dryrun_20260609_154557__s1.jsonl", "20260609_115039_013"),
    ("live_dryrun_20260609_154557__s2.jsonl", "20260609_120518_206"),
]
# A real under-read hand-start (sum=8200) — must STILL anchor.
UNDER_READ = ("live_dryrun_20260609_154557__s2.jsonl", "20260609_115900_891")
# The first poisoned recovery frame (decision_recovered seat6=9897 before
# the guard) — must decline with no-anchor after the guard.
POISONED_RECOVERY = ("live_dryrun_20260609_154557__s1.jsonl",
                     "20260609_115050_884")


def _record(stream_name: str, captured_at: str) -> dict:
    for line in open(STREAMS / stream_name):
        fr = json.loads(line)
        if fr["captured_at"] == captured_at:
            return fr["record"]
    raise AssertionError(f"{captured_at} not in {stream_name}")


def _synthetic_hand_start(dealer_seat: int, stacks: list[int],
                          blinds=(15, 25, 5)) -> ScraperFrame:
    """Build a self-consistent 6-alive hand-start frame whose pre-hand
    sum is sum(stacks) + 2*sb-ish pot arithmetic — callers pick `stacks`
    to hit the target sum: sum(pre) == sum(stacks) + pot where
    pot = sb + bb + 6*ante."""
    sb, bb, ante = blinds
    sb_seat = (dealer_seat + 1) % NUM_SEATS
    bb_seat = (dealer_seat + 2) % NUM_SEATS
    bet = [0] * NUM_SEATS
    bet[sb_seat], bet[bb_seat] = sb, bb
    pot = sb + bb + NUM_SEATS * ante
    return ScraperFrame(
        captured_at=f"synthetic_d{dealer_seat}",
        blinds=BlindsLevel(sb=sb, bb=bb, ante=ante),
        dealer_seat=dealer_seat, hero_seat=0,
        hero_cards=("As", "Kd"), board=(),
        stack=tuple(stacks), bet=tuple(bet),
        folded=(False,) * NUM_SEATS, empty=(False,) * NUM_SEATS,
        alive=(True,) * NUM_SEATS,
        pot_total=pot, controls_present=False, hero_facing_bet=False,
    )


# ── must-refuse: the two real poisoned hand-starts ──────────────────────

@pytest.mark.parametrize("stream,cap", POISONED)
def test_poisoned_hand_start_refused(stream, cap, capsys):
    rec = _record(stream, cap)
    frame = parse_frame(rec)
    assert is_hand_start(frame)
    assert sum(pre_hand_stacks(frame)) == 17917
    t = SessionTracker()
    assert t.observe(frame) is False
    assert t._current_pre_hand is None
    out = capsys.readouterr().out
    assert "[ANCHOR-REFUSED]" in out
    assert "seat6" in out  # per-seat bound names the offender


# ── must-still-anchor: clean and under-read hand-starts ─────────────────

def test_under_read_hand_start_still_anchors():
    rec = _record(*UNDER_READ)
    frame = parse_frame(rec)
    assert is_hand_start(frame)
    assert sum(pre_hand_stacks(frame)) == 8200  # below ceiling: fine
    t = SessionTracker()
    assert t.observe(frame) is True
    assert t._current_pre_hand is not None


def test_corpus_zero_anchor_regressions():
    """Replay every fixture stream's clean frames through a tracker:
    refusals occur at EXACTLY the two poisoned hand-starts."""
    refused = []
    for stream_path in sorted(STREAMS.glob("*.jsonl")):
        t = SessionTracker()
        for line in open(stream_path):
            fr = json.loads(line)
            try:
                f = parse_frame(fr["record"])
            except (ScraperParseError, ScraperSuspect, ScraperDataQuality):
                continue
            if t.observe(f) is False:
                refused.append((stream_path.name, fr["captured_at"]))
    assert refused == POISONED


# ── boundary: strict `>`, sum bound ─────────────────────────────────────

def test_sum_exactly_at_ceiling_accepted():
    # sum(pre) == 9000 exactly -> accepted (strict >). pot = 15+25+30 = 70.
    stacks = [CEILING - 70 - 5 * 1500] + [1500] * 5  # 1430,1500x5 -> pre 9000
    f = _synthetic_hand_start(0, stacks)
    assert sum(pre_hand_stacks(f)) == CEILING
    assert SessionTracker().observe(f) is True


def test_sum_one_over_ceiling_refused(capsys):
    stacks = [CEILING - 70 - 5 * 1500 + 1] + [1500] * 5  # pre sum 9001
    f = _synthetic_hand_start(0, stacks)
    assert sum(pre_hand_stacks(f)) == CEILING + 1
    assert SessionTracker().observe(f) is False
    assert "sum(pre_hand)" in capsys.readouterr().out


# ── OOD cross-check ─────────────────────────────────────────────────────

def test_ood_fires_once_when_mode_disagrees(capsys):
    t = SessionTracker()
    # 6 accepted anchors all summing to 8200 (distinct hands via dealer).
    for d in range(6):
        stacks = [8200 - 70 - 5 * 1300] + [1300] * 5
        assert t.observe(_synthetic_hand_start(d, stacks)) is True
    out = capsys.readouterr().out
    assert out.count("[ANCHOR-OOD]") == 1


def test_no_ood_on_normal_session(capsys):
    t = SessionTracker()
    for d in range(6):
        stacks = [CEILING - 70 - 5 * 1500] + [1500] * 5
        assert t.observe(_synthetic_hand_start(d, stacks)) is True
    assert "[ANCHOR-OOD]" not in capsys.readouterr().out


# ── integration: poisoned recovery becomes a safe-fold ──────────────────

@pytest.fixture
def stub_policy(monkeypatch):
    import scripts.eval_6max_self_play as ev
    monkeypatch.setattr(ev, "_sample_action_from_policy",
                        lambda *a, **k: 1)


def test_recovery_declines_after_refused_anchor(stub_policy):
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    tracker = SessionTracker()
    hs = _record(*POISONED[0])
    d_hs = make_decision(hs, structure, solver=None, tracker=tracker,
                         rng=random.Random(42), mode="sample", seq=104)
    assert d_hs.anchor_refused is True
    assert d_hs.status == "skip_not_hero_to_act"  # frame output unaffected

    rec = _record(*POISONED_RECOVERY)
    d = make_decision(rec, structure, solver=None, tracker=tracker,
                      rng=random.Random(42), mode="sample", seq=105)
    assert d.status == "skip_data_quality"
    assert "no clean pre-hand anchor" in d.skip_reason
    assert d.recovered_fields is None
    assert d.anchor_refused is False


def test_clean_anchor_recovery_unaffected(stub_policy):
    """The guard is a strict no-op on clean-anchor recoveries: the
    verify1 seq=88 incident recovery (validated in layer 1) must still
    fire identically."""
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    tracker = SessionTracker()
    v1 = "live_dryrun_verify1_20260609_202137__s1.jsonl"
    hs = _record(v1, "20260609_162518_092")   # clean hand-start (seq 72)
    sus = _record(v1, "20260609_162607_537")  # seq 88, hero 11101
    d_hs = make_decision(hs, structure, solver=None, tracker=tracker,
                         rng=random.Random(42), mode="sample", seq=72)
    assert d_hs.anchor_refused is False
    d = make_decision(sus, structure, solver=None, tracker=tracker,
                      rng=random.Random(42), mode="sample", seq=88)
    assert d.status == "decision_recovered"
    assert d.recovered_fields == ["stack.seat1=1110 (strict)"]
