"""Tests for src/nlhe/icm_returns.py (Phase 4e.2)."""
from __future__ import annotations
import math
import random

import pyspiel
import pytest

from src.nlhe.icm_returns import icm_adjust_returns, icm_adjust_trajectory
from src.nlhe.icm import sng_payouts_6max_double_up, sng_payouts_6max_standard
from src.nlhe.game_strings import six_max_sng
from src.nlhe.trajectory6 import walk_game, uniform_random_policy


# ---- Basic invariants ----

def test_zero_chip_returns_yield_zero_icm():
    """A hand where nobody won or lost any chips has zero ICM delta."""
    chip_returns = [0, 0, 0, 0, 0, 0]
    starting_stacks = [1500, 1500, 1500, 1500, 1500, 1500]
    payouts = sng_payouts_6max_double_up(buy_in=1.0)
    icm_returns = icm_adjust_returns(chip_returns, starting_stacks, payouts)
    for x in icm_returns:
        assert math.isclose(x, 0.0, abs_tol=1e-9)


def test_chip_pool_conserved_yields_zero_sum_icm():
    """Chips conserved (sum=0) implies ICM delta is approximately conserved.

    ICM is not perfectly conservative (it's a nonlinear map of stacks),
    but the START-pool and END-pool prize allocations sum to the same
    total — so the deltas should sum to ~0.
    """
    chip_returns = [+300, -300, 0, 0, 0, 0]  # SB doubles BB up
    starting_stacks = [1500, 1500, 1500, 1500, 1500, 1500]
    payouts = sng_payouts_6max_double_up(buy_in=1.0)
    icm_returns = icm_adjust_returns(chip_returns, starting_stacks, payouts)
    assert math.isclose(sum(icm_returns), 0.0, abs_tol=1e-6), (
        f"ICM deltas sum to {sum(icm_returns)}, expected ~0"
    )


def test_player_who_gained_chips_has_positive_icm():
    """A player who finishes with more chips than they started has positive ICM delta."""
    chip_returns = [+500, -250, -250, 0, 0, 0]
    starting_stacks = [1500] * 6
    payouts = sng_payouts_6max_double_up(buy_in=1.0)
    icm = icm_adjust_returns(chip_returns, starting_stacks, payouts)
    # Player 0 won, players 1 and 2 lost.
    assert icm[0] > 0, f"chip winner has icm {icm[0]}, expected > 0"
    assert icm[1] < 0
    assert icm[2] < 0


def test_busted_player_has_negative_icm():
    """A player who busts loses chips equal to their starting stack and has negative ICM."""
    chip_returns = [-1500, +1500, 0, 0, 0, 0]  # player 0 busts to player 1
    starting_stacks = [1500] * 6
    payouts = sng_payouts_6max_double_up(buy_in=1.0)
    icm = icm_adjust_returns(chip_returns, starting_stacks, payouts)
    # Player 0 is now busted (end_stack=0): they lock the bottom payout.
    # 5 active players, 3 paid. Bottom payouts go to busted players.
    # Player 0 went from 1/6 of pool equity to ~$0 (they're locked into
    # whatever the bottom-payout slot is — which is 0 here since only top-3
    # are paid). So they should be deeply negative.
    assert icm[0] < icm[2], "busted player should have worse ICM than untouched players"
    # The chip winner (player 1) gained chips, should have positive ICM.
    assert icm[1] > 0


# ---- ICM is NOT just chip-proportional ----

def test_icm_compresses_chip_lead():
    """ICM should give the chip leader LESS equity gain than chip-EV would suggest.

    This is the classic ICM property: doubling your chips doesn't double
    your equity (because there's a payout ceiling).
    """
    # Two scenarios where player 0 gains 500 chips, in different overall states.
    starting_stacks_a = [1500] * 6  # all equal
    chip_returns_a = [+500, -100, -100, -100, -100, -100]

    payouts = sng_payouts_6max_double_up(buy_in=1.0)
    icm_a = icm_adjust_returns(chip_returns_a, starting_stacks_a, payouts)

    # In chip-EV, the +500 chip gain is exactly +500.
    # In ICM-EV, it's compressed by the prize-pool cap.
    # Verify icm[0] is LESS than chip-proportional (+500 normalized to prize pool fraction).
    total_pool = sum(payouts)
    total_chips = sum(starting_stacks_a)
    chip_eq_chip_gain = 500.0 * total_pool / total_chips  # what +500 chips would be worth at start
    assert icm_a[0] < chip_eq_chip_gain, (
        f"icm_a[0]={icm_a[0]} should be less than chip-proportional {chip_eq_chip_gain}"
    )


# ---- Argument validation ----

def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        icm_adjust_returns([0, 0, 0], [1500] * 6, sng_payouts_6max_double_up())


def test_negative_starting_stack_raises():
    with pytest.raises(ValueError):
        icm_adjust_returns([0] * 6, [-100] + [1500] * 5, sng_payouts_6max_double_up())


# ---- Real trajectory integration ----

def test_icm_adjust_real_6max_trajectory_double_up():
    """End-to-end: walk a 6-max SNG game, apply ICM, returns are well-formed."""
    game = pyspiel.load_game(six_max_sng(starting_stack=1500))
    rng = random.Random(2026)
    traj = walk_game(game, uniform_random_policy, rng)
    payouts = sng_payouts_6max_double_up(buy_in=1.0)

    icm = icm_adjust_trajectory(traj, starting_stacks=[1500] * 6, payouts=payouts)

    assert len(icm) == 6
    # Sum ~0 (ICM map is approximately conservative for chip-conserved hands).
    assert math.isclose(sum(icm), 0.0, abs_tol=1e-6)


def test_icm_adjust_real_6max_trajectory_standard():
    """Standard 6-max payouts (top-2 paid 65/35) also work end-to-end."""
    game = pyspiel.load_game(six_max_sng(starting_stack=1500))
    rng = random.Random(2026)
    traj = walk_game(game, uniform_random_policy, rng)
    payouts = sng_payouts_6max_standard(buy_in=1.0)

    icm = icm_adjust_trajectory(traj, starting_stacks=[1500] * 6, payouts=payouts)

    assert len(icm) == 6
    assert math.isclose(sum(icm), 0.0, abs_tol=1e-6)


def test_icm_adjust_doesnt_mutate_trajectory():
    """icm_adjust_trajectory returns a new list, doesn't mutate trajectory."""
    game = pyspiel.load_game(six_max_sng(starting_stack=1500))
    rng = random.Random(2026)
    traj = walk_game(game, uniform_random_policy, rng)
    original_returns = list(traj.terminal_returns)

    icm = icm_adjust_trajectory(traj, starting_stacks=[1500] * 6,
                                payouts=sng_payouts_6max_double_up())

    # Trajectory's chip returns unchanged
    assert traj.terminal_returns == original_returns
    # ICM result is a new list
    assert icm is not traj.terminal_returns


# ---- Floating-point edge case ----

def test_handles_floating_point_drift_in_returns():
    """If state.returns() includes tiny floating drift, end-stacks shouldn't go negative."""
    # Simulate a return that would push a stack to -0.001
    chip_returns = [-1500.001, +1500.001, 0, 0, 0, 0]
    starting_stacks = [1500] * 6
    payouts = sng_payouts_6max_double_up()
    # Should not raise (icm.py would otherwise complain about negative stacks)
    icm = icm_adjust_returns(chip_returns, starting_stacks, payouts)
    assert len(icm) == 6


# ---------- Pre-busted-seat handling (load-bearing icm.py fix) ----------

def test_pre_busted_locked_itm_zero_delta():
    """Realistic ITM: 3 alive + 3 PRE-busted, num_paid=3 (double-up). Every alive
    seat is locked at the equal payout, so ANY hand outcome yields ~0 ICM delta,
    and pre-busted seats are never enriched. Regression for the icm.py busted-seat
    bug: pre-fix, a mid-hand bust split the in-money bottom payout among all
    stack-0 seats, giving the 3 pre-busted seats spurious +equity and the
    newly-busted seat a spurious -loss."""
    starting = [4000, 4000, 4000, 0, 0, 0]
    payouts = sng_payouts_6max_double_up()  # [2, 2, 2]
    for cr in (
        [+4000, 0, -4000, 0, 0, 0],      # seat 2 busts to seat 0
        [+2000, +2000, -4000, 0, 0, 0],  # seat 2 busts, split pot
        [-2000, +1000, +1000, 0, 0, 0],  # seat 0 loses but survives
        [+8000, -4000, -4000, 0, 0, 0],  # seats 1 AND 2 both bust this hand
    ):
        delta = icm_adjust_returns(cr, starting, payouts)
        for i, x in enumerate(delta):
            assert math.isclose(x, 0.0, abs_tol=1e-9), f"cr={cr} seat={i} delta={delta}"


def test_pre_busted_not_enriched_on_bubble_bust():
    """Bubble (4 alive + 2 pre-busted, num_paid=3): when an alive seat busts it
    finishes 4th (unpaid) and loses equity; the 2 PRE-busted seats stay at 0
    delta (not enriched), and the pool is conserved."""
    starting = [3000, 3000, 3000, 3000, 0, 0]
    payouts = sng_payouts_6max_double_up()
    cr = [+3000, +1000, -1000, -3000, 0, 0]  # seat 3 busts (finishes 4th)
    delta = icm_adjust_returns(cr, starting, payouts)
    assert math.isclose(delta[4], 0.0, abs_tol=1e-9), f"{delta}"  # pre-busted
    assert math.isclose(delta[5], 0.0, abs_tol=1e-9), f"{delta}"  # pre-busted
    assert delta[3] < -0.5, f"newly-busted seat should lose equity: {delta}"
    assert math.isclose(sum(delta), 0.0, abs_tol=1e-9)  # conserved


def test_all_alive_hand_unchanged_by_fix():
    """Normal all-alive hand: eligible = all seats, so the fix is a no-op vs the
    single-hand semantics (the busted seat locks the bottom payout). Conserved."""
    starting = [1500] * 6
    payouts = sng_payouts_6max_double_up()
    cr = [+1500, 0, 0, 0, 0, -1500]  # seat 5 busts to seat 0
    delta = icm_adjust_returns(cr, starting, payouts)
    assert len(delta) == 6
    assert math.isclose(sum(delta), 0.0, abs_tol=1e-9)
