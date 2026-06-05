"""Unit tests for src/nlhe/integration/translate.py.

Forward and reverse action translations between real-poker chip_ints and
OpenSpiel's inflated-BB chip_ints.
"""
from __future__ import annotations

import pytest

from src.nlhe.integration.translate import (
    real_to_openspiel_action,
    openspiel_to_real_action,
)


# --- Forward translation: real -> openspiel ---

def test_forward_fold_passes_through():
    assert real_to_openspiel_action(0, [0, 1, 110, 111, 112]) == 0


def test_forward_call_passes_through():
    assert real_to_openspiel_action(1, [0, 1, 110, 111, 112]) == 1


def test_forward_raise_above_min_passes_through():
    # Real raise to 150 with openspiel min-raise=110: passes through (150 > 110)
    assert real_to_openspiel_action(150, [0, 1, 110, 111, 112, 113, 114, 150,
                                           200, 1500]) == 150


def test_forward_raise_below_min_bumps_up():
    """Load-bearing: real min-raise=50 below OpenSpiel min-raise=110."""
    # legal: fold, call, raise [110..1500]
    assert real_to_openspiel_action(50, [0, 1, 110, 111, 112, 113, 1500]) == 110


def test_forward_raise_at_exact_min_passes_through():
    """Raise exactly at OpenSpiel min: passes through (no need to bump)."""
    assert real_to_openspiel_action(110, [0, 1, 110, 111, 1500]) == 110


def test_forward_raise_above_max_clamps_to_max():
    """If real raise exceeds hero's stack (= max legal), clamp."""
    # Hero stack legal max = 200. Real raise of 5000 > 200 → clamp to 200.
    assert real_to_openspiel_action(5000, [0, 1, 110, 111, 200]) == 200


def test_forward_fold_when_no_fold_legal_falls_back_to_call():
    """Defensive: scraper says fold, but state has no fold legal (BB-option
    case where check is the costless option). Fall back to call/check (1)."""
    assert real_to_openspiel_action(0, [1, 110, 111, 1500]) == 1


def test_forward_no_call_and_no_fold_raises():
    """Degenerate: neither fold nor call legal — caller should treat as
    ReplayError."""
    with pytest.raises(ValueError):
        real_to_openspiel_action(0, [110, 111])  # only raises legal
    # ditto for explicit call when no call legal
    with pytest.raises(ValueError):
        real_to_openspiel_action(1, [0, 110, 111])  # no call


def test_forward_raise_when_no_raise_legal_raises():
    """If hero can only fold/call (no raise legal — e.g., already all-in
    facing all-in), and scraper asks for raise: caller decides."""
    with pytest.raises(ValueError):
        real_to_openspiel_action(150, [0, 1])  # no raise


# --- Reverse translation: openspiel -> real ---

def test_reverse_fold_passes_through():
    res = openspiel_to_real_action(0, scraper_min_raise=50,
                                    scraper_max_raise=1500,
                                    scraper_facing_bet=True)
    assert res["kind"] == "fold"
    assert res["chip_amount"] is None
    assert res["raw_openspiel_chip_int"] == 0


def test_reverse_call_when_facing_bet():
    res = openspiel_to_real_action(1, scraper_min_raise=50,
                                    scraper_max_raise=1500,
                                    scraper_facing_bet=True)
    assert res["kind"] == "call"
    assert res["chip_amount"] is None


def test_reverse_check_when_not_facing_bet():
    res = openspiel_to_real_action(1, scraper_min_raise=25,
                                    scraper_max_raise=1500,
                                    scraper_facing_bet=False)
    assert res["kind"] == "check"


def test_reverse_raise_within_bounds_passes_through():
    """Model output 200 chips; real legal range 50-1500. Pass through 200."""
    res = openspiel_to_real_action(200, scraper_min_raise=50,
                                    scraper_max_raise=1500,
                                    scraper_facing_bet=True)
    assert res["kind"] == "raise_to"
    assert res["chip_amount"] == 200
    assert res["raw_openspiel_chip_int"] == 200


def test_reverse_raise_below_real_min_bumps_up():
    """Model output 30 chips but real min-raise is 50. Clamp to 50."""
    res = openspiel_to_real_action(30, scraper_min_raise=50,
                                    scraper_max_raise=1500,
                                    scraper_facing_bet=True)
    assert res["kind"] == "raise_to"
    assert res["chip_amount"] == 50
    assert res["raw_openspiel_chip_int"] == 30


def test_reverse_raise_above_real_max_clamps_to_max():
    """Model output 5000 but hero has only 1500 in stack. Clamp."""
    res = openspiel_to_real_action(5000, scraper_min_raise=50,
                                    scraper_max_raise=1500,
                                    scraper_facing_bet=True)
    assert res["kind"] == "raise_to"
    assert res["chip_amount"] == 1500
    assert res["raw_openspiel_chip_int"] == 5000


def test_reverse_drift_documented():
    """The bet-sizing drift: model's openspiel min-raise (= 110 chips
    at level 1) is roughly 2x real min-raise (= 50 chips). We pass the
    inflated number through directly. This test documents the drift
    rather than asserting it's small — the user accepted the cost."""
    # Inflated min-raise 110 at level 1
    res = openspiel_to_real_action(110, scraper_min_raise=50,
                                    scraper_max_raise=1500,
                                    scraper_facing_bet=True)
    assert res["chip_amount"] == 110
    # In real-poker terms, 110 is ~4.4x BB (= 4.4*25), while the model
    # intended "min-raise" (2x BB = 50). Hero will raise 4.4x instead of 2x.
    # This is the documented bridge cost.
