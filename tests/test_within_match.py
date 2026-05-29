"""Tests for src/nlhe/within_match.py (Layer 4 / C1a).

Covers stats correctness, confidence ramp boundaries, wipe idempotency, and
the load-bearing anti-latch isolation check.
"""
from __future__ import annotations

import pytest

from src.nlhe.within_match import MatchObserver, SeatStats


# DiscreteAction integer aliases (kept inline so the tests don't depend on the
# action enum import path — matches the module's own internal constant style).
FOLD = 0
CALL = 1
BET_33 = 2
BET_66 = 3
BET_100 = 4
BET_200 = 5
ALLIN = 6


def _fresh_seat_stats(seat: int) -> SeatStats:
    """Return a SeatStats identical to a freshly-constructed observer's seat."""
    return SeatStats(seat=seat)


def test_fresh_observer_zero_stats():
    obs = MatchObserver()
    for s in range(6):
        assert obs.get_stats(s) == _fresh_seat_stats(s)


def test_vpip_counted():
    obs = MatchObserver()
    # Seat 0 facing a bet (seat 1 put in 100, others 0): each CALL is VPIP.
    parsed_facing = {"street_idx": 0, "contribution": [0, 100, 0, 0, 0, 0]}
    obs.update(None, parsed_facing, action=CALL, seat=0)
    obs.update(None, parsed_facing, action=CALL, seat=0)
    obs.update(None, parsed_facing, action=FOLD, seat=0)
    stats = obs.get_stats(0)
    assert stats.n_preflop_decisions == 3
    assert stats.n_preflop_voluntary == 2


def test_pfr_counted():
    obs = MatchObserver()
    parsed = {"street_idx": 0, "contribution": [0, 0, 0, 0, 0, 0]}
    obs.update(None, parsed, action=BET_66, seat=2)
    obs.update(None, parsed, action=FOLD, seat=2)
    obs.update(None, parsed, action=BET_100, seat=2)
    stats = obs.get_stats(2)
    assert stats.n_preflop_raises == 2
    assert stats.n_preflop_decisions == 3
    # Aggressive preflop actions are also voluntary.
    assert stats.n_preflop_voluntary == 2


def test_aggression_by_street():
    obs = MatchObserver()
    flop_parsed = {"street_idx": 1, "contribution": [0, 0, 0, 0, 0, 0]}
    turn_parsed = {"street_idx": 2, "contribution": [0, 0, 0, 0, 0, 0]}
    obs.update(None, flop_parsed, action=BET_100, seat=3)
    obs.update(None, flop_parsed, action=CALL, seat=3)
    obs.update(None, turn_parsed, action=BET_100, seat=3)
    obs.update(None, turn_parsed, action=CALL, seat=3)
    stats = obs.get_stats(3)
    assert stats.n_postflop_decisions[1] == 2
    assert stats.n_postflop_aggressive[1] == 1
    assert stats.n_postflop_decisions[2] == 2
    assert stats.n_postflop_aggressive[2] == 1
    # Slot 0 (preflop) and slot 3 (river) untouched on postflop arrays.
    assert stats.n_postflop_decisions[0] == 0
    assert stats.n_postflop_decisions[3] == 0


def test_fold_to_bet():
    obs = MatchObserver()
    # Seat 4 on flop facing a bet (seat 5 has put in 100, seat 4 = 0 on flop).
    parsed_facing = {"street_idx": 1, "contribution": [0, 0, 0, 0, 0, 100]}
    obs.update(None, parsed_facing, action=FOLD, seat=4)
    stats_after_fold = obs.get_stats(4)
    assert stats_after_fold.n_facing_bet[1] == 1
    assert stats_after_fold.n_folds_facing_bet[1] == 1

    obs.update(None, parsed_facing, action=CALL, seat=4)
    stats_after_call = obs.get_stats(4)
    assert stats_after_call.n_facing_bet[1] == 2
    assert stats_after_call.n_folds_facing_bet[1] == 1  # unchanged on a call


def test_showdown_outcome():
    obs = MatchObserver()
    obs.note_showdown(seats_reaching_showdown=[1, 3, 4], winner_seats=[3])
    s1 = obs.get_stats(1)
    s3 = obs.get_stats(3)
    s4 = obs.get_stats(4)
    assert s1.n_showdowns == 1 and s1.n_showdown_wins == 0
    assert s3.n_showdowns == 1 and s3.n_showdown_wins == 1
    assert s4.n_showdowns == 1 and s4.n_showdown_wins == 0
    # Untouched seats remain zero.
    for s in (0, 2, 5):
        sx = obs.get_stats(s)
        assert sx.n_showdowns == 0 and sx.n_showdown_wins == 0


def test_confidence_ramp_values():
    """Boundary checks for the locked 20/100/300 ramp.

    Drive n_actions by feeding null updates (FOLD preflop with no bet faced —
    increments n_actions + n_preflop_decisions, no other counters).
    """
    parsed_null = {"street_idx": 0, "contribution": [0] * 6}

    def make_observer_with_n(n: int) -> MatchObserver:
        o = MatchObserver()
        for _ in range(n):
            o.update(None, parsed_null, action=FOLD, seat=0)
        return o

    # n=0 → 0.0 (no updates)
    assert MatchObserver().confidence(0) == 0.0
    # n=19 → 0.0 (still below the 20-action floor)
    assert make_observer_with_n(19).confidence(0) == 0.0
    # n=20 → 0.0 (start of ramp; the first non-floored sample)
    assert make_observer_with_n(20).confidence(0) == pytest.approx(0.0)
    # n=60 → midway through first ramp segment (40/80 * 0.5 = 0.25)
    assert make_observer_with_n(60).confidence(0) == pytest.approx(0.25)
    # n=100 → boundary between first and second segments
    assert make_observer_with_n(100).confidence(0) == pytest.approx(0.5)
    # n=200 → midway through second segment (0.5 + 100/200 * 0.5 = 0.75)
    assert make_observer_with_n(200).confidence(0) == pytest.approx(0.75)
    # n=300 → saturates at 1.0
    assert make_observer_with_n(300).confidence(0) == pytest.approx(1.0)
    # n=500 → still 1.0
    assert make_observer_with_n(500).confidence(0) == pytest.approx(1.0)


def test_wipe_resets_all_state():
    obs = MatchObserver()
    # Mixed populations across seats + a showdown.
    obs.update(None, {"street_idx": 0, "contribution": [0, 100, 0, 0, 0, 0]},
               action=CALL, seat=0)
    obs.update(None, {"street_idx": 1, "contribution": [0, 0, 0, 0, 0, 100]},
               action=BET_66, seat=2)
    obs.update(None, {"street_idx": 2, "contribution": [0, 0, 0, 0, 0, 100]},
               action=FOLD, seat=4)
    obs.note_showdown([1, 3], [1])

    obs.wipe()
    for s in range(6):
        assert obs.get_stats(s) == _fresh_seat_stats(s)

    # Feed exactly one new action: only that counter should increment.
    obs.update(None, {"street_idx": 0, "contribution": [0, 100, 0, 0, 0, 0]},
               action=CALL, seat=0)
    stats0 = obs.get_stats(0)
    assert stats0.n_actions == 1
    assert stats0.n_preflop_decisions == 1
    assert stats0.n_preflop_voluntary == 1
    # No leftover state from any other seat.
    for s in (1, 2, 3, 4, 5):
        assert obs.get_stats(s) == _fresh_seat_stats(s)


def test_wipe_idempotent():
    obs = MatchObserver()
    obs.update(None, {"street_idx": 0, "contribution": [0, 100, 0, 0, 0, 0]},
               action=CALL, seat=0)

    obs.wipe()
    snapshot_after_one = [obs.get_stats(s) for s in range(6)]
    obs.wipe()
    obs.wipe()
    snapshot_after_three = [obs.get_stats(s) for s in range(6)]
    assert snapshot_after_three == snapshot_after_one
    # Both equal the fresh-observer state.
    for s in range(6):
        assert snapshot_after_three[s] == _fresh_seat_stats(s)


def test_match_started_ended_equivalent_to_wipe():
    parsed = {"street_idx": 0, "contribution": [0, 100, 0, 0, 0, 0]}

    # match_ended() resets.
    obs = MatchObserver()
    for _ in range(5):
        obs.update(None, parsed, action=CALL, seat=0)
    obs.match_ended()
    for s in range(6):
        assert obs.get_stats(s) == _fresh_seat_stats(s)

    # match_started() resets the same way.
    obs2 = MatchObserver()
    for _ in range(7):
        obs2.update(None, parsed, action=CALL, seat=0)
    obs2.match_started()
    for s in range(6):
        assert obs2.get_stats(s) == _fresh_seat_stats(s)


def test_get_stats_returns_independent_copy():
    obs = MatchObserver()
    parsed = {"street_idx": 0, "contribution": [0, 100, 0, 0, 0, 0]}
    obs.update(None, parsed, action=CALL, seat=2)
    obs.update(None, parsed, action=CALL, seat=2)

    s_first = obs.get_stats(2)
    assert s_first.n_actions == 2
    # Mutate the returned copy.
    s_first.n_actions = 9999
    s_first.n_preflop_voluntary = 9999
    s_first.n_postflop_decisions[1] = 9999  # list mutation must not leak either

    s_second = obs.get_stats(2)
    assert s_second.n_actions == 2
    assert s_second.n_preflop_voluntary == 2  # both CALLs were VPIP (facing_bet=True)
    assert s_second.n_postflop_decisions[1] == 0


def test_two_instances_independent():
    obs_a = MatchObserver()
    obs_b = MatchObserver()

    parsed = {"street_idx": 0, "contribution": [0, 100, 0, 0, 0, 0]}
    for _ in range(30):
        obs_a.update(None, parsed, action=CALL, seat=1)
    obs_a.note_showdown([1], [1])

    # B is untouched.
    for s in range(6):
        assert obs_b.get_stats(s) == _fresh_seat_stats(s)

    # Wipe A; B remains untouched.
    obs_a.wipe()
    for s in range(6):
        assert obs_b.get_stats(s) == _fresh_seat_stats(s)


def test_no_class_level_mutable_state_after_wipe():
    """Anti-latch test.

    If anyone introduces a class-level latch (cf. archetype6.py:78
    `_warned_no_dealer`), state set on instance A would leak into the behavior
    of a freshly-constructed instance B. This test catches that regression.
    """
    obs_a = MatchObserver()
    parsed = {"street_idx": 0, "contribution": [0, 100, 0, 0, 0, 0]}
    for _ in range(50):
        obs_a.update(None, parsed, action=CALL, seat=0)
    obs_a.note_showdown([0, 1, 2], [2])
    obs_a.wipe()

    obs_b = MatchObserver()  # freshly constructed AFTER A's lifecycle
    for s in range(6):
        assert obs_a.get_stats(s) == obs_b.get_stats(s)
        assert obs_b.get_stats(s) == _fresh_seat_stats(s)
