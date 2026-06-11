"""Unit tests for the guaranteed-action fallback watchdog
(src/nlhe/integration/fallback.py). Pure state-machine tests with
injected time — no solver, no sockets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.nlhe.integration.fallback import (
    FallbackWatchdog, is_real_to_act,
)


def raw(cards=("Ah", "Kd"), buttons=("CALL", "FOLD", "RAISE"),
        captured_at="20260611_120000_000"):
    return {
        "hero_cards": list(cards),
        "action": {"buttons": list(buttons), "amounts": [], "raw": ""},
        "controls": {"present": bool(buttons),
                     "action_buttons": list(buttons)},
        "captured_at": captured_at,
    }


def test_real_to_act_predicate():
    assert is_real_to_act(raw(buttons=("CALL", "FOLD", "RAISE")))
    assert is_real_to_act(raw(buttons=("BET", "CHECK")))
    assert is_real_to_act(raw(buttons=("ALLIN", "CALL", "FOLD")))
    # FOLD-only is the muck / fold-ahead UI (23 such frames in session
    # 163815) — not a real to-act spot.
    assert not is_real_to_act(raw(buttons=("FOLD",)))
    assert not is_real_to_act(raw(buttons=()))


def test_requires_positive_seconds():
    with pytest.raises(ValueError):
        FallbackWatchdog(0)


def test_arms_but_does_not_fire_before_deadline():
    w = FallbackWatchdog(7.0)
    assert w.observe(raw(), "skip_data_quality", seq=1, now=100.0) is None
    assert w.observe(raw(), "skip_data_quality", seq=2, now=106.9) is None
    assert w.n_fallbacks == 0


def test_fires_fold_when_no_check_available():
    w = FallbackWatchdog(7.0)
    w.observe(raw(), "skip_data_quality", seq=1, now=100.0)
    plan = w.observe(raw(), "skip_data_quality", seq=5, now=107.0)
    assert plan is not None
    assert plan.action_kind == "fold"
    assert plan.armed_seq == 1
    assert plan.fired_after_seq == 5
    assert plan.waited_seconds == pytest.approx(7.0)
    assert not plan.check_available


def test_fires_check_when_free():
    w = FallbackWatchdog(7.0)
    w.observe(raw(buttons=("BET", "CHECK")), "safe_fold", seq=1, now=0.0)
    plan = w.observe(raw(buttons=("BET", "CHECK")), "safe_fold",
                     seq=2, now=8.0)
    assert plan is not None
    assert plan.action_kind == "check"


def test_deadline_anchored_to_first_frame_not_refreshed():
    w = FallbackWatchdog(7.0)
    w.observe(raw(), "skip_data_quality", seq=1, now=0.0)
    w.observe(raw(), "skip_data_quality", seq=2, now=5.0)
    plan = w.observe(raw(), "skip_data_quality", seq=3, now=7.0)
    assert plan is not None and plan.armed_seq == 1


def test_decision_disarms():
    w = FallbackWatchdog(7.0)
    w.observe(raw(), "skip_data_quality", seq=1, now=0.0)
    w.observe(raw(), "decision", seq=2, now=1.0)
    # Deadline long past, but the episode was disarmed by the decision.
    assert w.observe(raw(), "decision_cached", seq=3, now=60.0) is None
    assert w.poll(now=60.0) is None


def test_one_frame_flicker_keeps_armed_two_frames_disarms():
    w = FallbackWatchdog(7.0, vanish_frames=2)
    w.observe(raw(), "skip_data_quality", seq=1, now=0.0)
    # 1-frame controls flicker: still armed, original deadline.
    w.observe(raw(buttons=()), "skip_not_hero_to_act", seq=2, now=1.0)
    plan = w.observe(raw(), "skip_data_quality", seq=3, now=7.5)
    assert plan is not None and plan.armed_seq == 1

    w2 = FallbackWatchdog(7.0, vanish_frames=2)
    w2.observe(raw(), "skip_data_quality", seq=1, now=0.0)
    w2.observe(raw(buttons=()), "skip_not_hero_to_act", seq=2, now=1.0)
    w2.observe(raw(buttons=()), "skip_not_hero_to_act", seq=3, now=2.0)
    # Disarmed (action taken externally) -> re-arm starts a NEW deadline.
    plan = w2.observe(raw(), "skip_data_quality", seq=4, now=8.0)
    assert plan is None
    assert w2.poll(now=14.9) is None
    assert w2.poll(now=15.0) is not None


def test_fires_once_per_episode():
    w = FallbackWatchdog(7.0)
    w.observe(raw(), "skip_data_quality", seq=1, now=0.0)
    assert w.observe(raw(), "skip_data_quality", seq=2, now=8.0) is not None
    assert w.observe(raw(), "skip_data_quality", seq=3, now=20.0) is None
    assert w.poll(now=30.0) is None
    assert w.n_fallbacks == 1


def test_new_hand_resets_episode():
    w = FallbackWatchdog(7.0)
    w.observe(raw(cards=("Ah", "Kd")), "skip_data_quality", seq=1, now=0.0)
    # New hand dealt before the deadline — old episode dead.
    assert w.observe(raw(cards=("2c", "7d")), "skip_data_quality",
                     seq=2, now=8.0) is None
    # New episode armed at now=8; fires at 15.
    assert w.observe(raw(cards=("2c", "7d")), "skip_data_quality",
                     seq=3, now=15.0) is not None
    assert w.hands_seen == 2


def test_poll_fires_framelessly():
    w = FallbackWatchdog(7.0)
    w.observe(raw(), "skip_data_quality", seq=9, now=0.0)
    assert w.poll(now=6.0) is None
    plan = w.poll(now=7.0)
    assert plan is not None
    assert plan.fired_after_seq is None
    assert plan.armed_seq == 9


def test_muck_only_never_arms():
    w = FallbackWatchdog(7.0)
    w.observe(raw(buttons=("FOLD",)), "skip_not_hero_to_act",
              seq=1, now=0.0)
    assert w.poll(now=100.0) is None


def test_abort_on_consecutive_hands():
    w = FallbackWatchdog(1.0, abort_consecutive=2)
    w.observe(raw(cards=("Ah", "Kd")), "skip_data_quality", seq=1, now=0.0)
    p1 = w.observe(raw(cards=("Ah", "Kd")), "skip_data_quality",
                   seq=2, now=2.0)
    assert p1 is not None and not p1.abort_recommended
    w.observe(raw(cards=("2c", "7d")), "skip_data_quality", seq=3, now=10.0)
    p2 = w.observe(raw(cards=("2c", "7d")), "skip_data_quality",
                   seq=4, now=12.0)
    assert p2 is not None and p2.abort_recommended
    assert "consecutive" in p2.abort_reason


def test_abort_on_window_count():
    w = FallbackWatchdog(1.0, abort_window_hands=10, abort_fallbacks=3,
                         abort_consecutive=99)  # isolate window rule
    hands = [("Ah", "Kd"), ("2c", "7d"), ("Th", "Tc"),
             ("5s", "6s"), ("Qd", "Jd")]
    plans = []
    t = 0.0
    for i, cards in enumerate(hands):
        fallback_hand = i in (0, 2, 4)
        w.observe(raw(cards=cards), "skip_data_quality",
                  seq=10 * i, now=t)
        if fallback_hand:
            plans.append(w.observe(raw(cards=cards), "skip_data_quality",
                                   seq=10 * i + 1, now=t + 2.0))
        t += 10.0
    fired = [p for p in plans if p is not None]
    assert len(fired) == 3
    assert not fired[0].abort_recommended
    assert not fired[1].abort_recommended
    assert fired[2].abort_recommended
    assert "fallback hands in the last" in fired[2].abort_reason


def test_log_record_invisible_to_existing_tooling():
    w = FallbackWatchdog(1.0)
    w.observe(raw(), "skip_data_quality", seq=1, now=0.0)
    plan = w.observe(raw(), "skip_data_quality", seq=2, now=2.0)
    rec = plan.as_log_record()
    assert rec["record_type"] == "fallback"
    # dryrun_triage filters on "seq" in rec; decision_audit and
    # replay_make_decision_diff filter on raw_record presence.
    assert "seq" not in rec
    assert "raw_record" not in rec


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ── watchdog v2 (hand_deadline + click_confirmation_mode) ──────────────

def test_v2_spot_deadline_survives_vanish_disarm_rearm():
    # The 9cAd gap (204751 seqs 1013-1020): intermittent visibility.
    w = FallbackWatchdog(7.0, vanish_frames=2, hand_deadline=True)
    w.observe(raw(cards=("9c", "Ad")), "skip_data_quality", seq=1, now=0.0)
    # buttons vanish for 2 frames -> episode disarmed
    w.observe(raw(cards=("9c", "Ad"), buttons=()), "skip_not_hero_to_act",
              seq=2, now=1.0)
    w.observe(raw(cards=("9c", "Ad"), buttons=()), "skip_not_hero_to_act",
              seq=3, now=2.0)
    # same spot re-appears at t=6.5: v1 would restart the deadline;
    # v2 resumes t0=0 and fires at 7.0
    assert w.observe(raw(cards=("9c", "Ad")), "safe_fold",
                     seq=4, now=6.5) is None
    plan = w.observe(raw(cards=("9c", "Ad")), "safe_fold", seq=5, now=7.0)
    assert plan is not None
    assert plan.waited_seconds == pytest.approx(7.0)


def test_v2_decision_clears_spot_anchor():
    w = FallbackWatchdog(7.0, hand_deadline=True)
    w.observe(raw(cards=("9c", "Ad")), "skip_data_quality", seq=1, now=0.0)
    w.observe(raw(cards=("9c", "Ad")), "decision", seq=2, now=1.0)
    # New to-act episode on the SAME spot gets a fresh deadline.
    assert w.observe(raw(cards=("9c", "Ad")), "skip_data_quality",
                     seq=3, now=8.0) is None
    assert w.poll(now=14.9) is None
    assert w.poll(now=15.0) is not None


def test_v2_fired_spot_reappearing_gets_fresh_deadline():
    w = FallbackWatchdog(7.0, vanish_frames=2, hand_deadline=True)
    w.observe(raw(cards=("9c", "Ad")), "skip_data_quality", seq=1, now=0.0)
    assert w.observe(raw(cards=("9c", "Ad")), "skip_data_quality",
                     seq=2, now=7.5) is not None
    # vanish x2 -> disarm; spot re-appears: fresh deadline, no instant
    # re-fire even though 20s have passed
    w.observe(raw(cards=("9c", "Ad"), buttons=()), "skip_not_hero_to_act",
              seq=3, now=8.0)
    w.observe(raw(cards=("9c", "Ad"), buttons=()), "skip_not_hero_to_act",
              seq=4, now=9.0)
    assert w.observe(raw(cards=("9c", "Ad")), "skip_data_quality",
                     seq=5, now=20.0) is None
    assert w.poll(now=26.9) is None
    assert w.poll(now=27.0) is not None


def test_v2_board_change_is_new_spot():
    w = FallbackWatchdog(7.0, hand_deadline=True)
    r1 = raw(cards=("9c", "Ad"))
    w.observe(r1, "skip_data_quality", seq=1, now=0.0)
    r2 = raw(cards=("9c", "Ad"))
    r2["board"] = ["Kh", "7c", "2d"]
    # flop arrives -> different spot -> fresh deadline
    assert w.observe(r2, "skip_data_quality", seq=2, now=8.0) is None


def test_click_confirmation_mode_decision_does_not_disarm():
    w = FallbackWatchdog(7.0, hand_deadline=True,
                         click_confirmation_mode=True)
    w.observe(raw(), "decision", seq=1, now=0.0)       # arms despite decision
    plan = w.observe(raw(), "decision_cached", seq=2, now=7.0)
    assert plan is not None                            # click never landed
    assert plan.action_kind == "fold"


def test_click_confirmation_mode_vanish_still_disarms():
    w = FallbackWatchdog(7.0, vanish_frames=2,
                         click_confirmation_mode=True)
    w.observe(raw(), "decision", seq=1, now=0.0)
    w.observe(raw(buttons=()), "skip_not_hero_to_act", seq=2, now=1.0)
    w.observe(raw(buttons=()), "skip_not_hero_to_act", seq=3, now=2.0)
    assert w.poll(now=60.0) is None                    # click confirmed


# ── AbortGate (enforced session abort) ─────────────────────────────────

def test_abort_gate_trip_and_manual_reset(tmp_path):
    from src.nlhe.integration.fallback import AbortGate
    reset = tmp_path / "RESUME"
    g = AbortGate(str(reset))
    assert not g.active
    assert g.trip("3 fallback hands in the last 10") is True
    assert g.active and "fallback hands" in g.reason
    assert g.trip("again") is False          # already active; not re-tripped
    assert g.check_reset() is False          # no reset file yet
    reset.write_text("")
    assert g.check_reset() is True           # operator reset
    assert not g.active and not reset.exists()
    assert g.trip("second trip") is True     # can trip again after reset
