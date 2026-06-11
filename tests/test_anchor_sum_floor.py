"""P1 anchor sum-floor guard (seq-1363 poisoned-anchor postmortem).

The guard refuses a hand-start anchor whose pre-hand sum differs from
chips-in-play while every seat reads alive. Flag-gated OFF by default —
default behavior must be byte-identical to pre-P1 (proven session-wide
by the replay gate; unit-level here)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlhe.integration.scraper_schema import SessionTracker
from tests.test_integration_session_tracker import _hand_start_frame, LV2


def _poisoned_frame(dealer_seat=2, drop=800):
    """A hand-start frame where seat5's stack OCR dropped `drop` chips
    (dropped-digit class): internally consistent — pot reflects the
    same wrong universe — so only a sum check can catch it."""
    f = _hand_start_frame(dealer_seat)
    stacks = list(f.stack)
    stacks[5] -= drop
    return dataclasses.replace(f, stack=tuple(stacks))


def test_flag_off_accepts_poisoned_anchor_legacy_behavior():
    tr = SessionTracker()  # default: floor OFF
    assert tr.observe(_poisoned_frame()) is True


def test_flag_on_refuses_poisoned_anchor():
    tr = SessionTracker(anchor_sum_floor=True)
    assert tr.observe(_poisoned_frame()) is False
    assert tr.pre_hand_for(_poisoned_frame()) is None


def test_flag_on_accepts_correct_anchor():
    tr = SessionTracker(anchor_sum_floor=True)
    assert tr.observe(_hand_start_frame(dealer_seat=2)) is True


def test_flag_on_skips_floor_when_a_seat_is_not_alive():
    # Mis-alive exemption: with a non-alive seat, chips can legitimately
    # hide; the floor must not fire (ceiling-only behavior preserved).
    tr = SessionTracker(anchor_sum_floor=True)
    f = _hand_start_frame(dealer_seat=2, alive_count=5)
    # 5 alive x 1500 = 7500 < 9000 ceiling: accepted under both guards.
    assert tr.observe(f) is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
