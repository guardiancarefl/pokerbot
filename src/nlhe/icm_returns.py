"""ICM-adjusted terminal utility for 6-max SNG training (Phase 4e.2).

Bridges between the chip-EV returns OpenSpiel produces and the ICM-EV
returns the CFR solver needs for SNG specialization.

The transformation:
  1. Start state: each player has `starting_stack_i` chips.
  2. End state:   each player has `starting_stack_i + chip_return_i` chips
                  (where chip_return_i comes from state.returns()).
  3. ICM equity at start: e_start_i = icm_equity(starting_stacks, payouts)[i]
  4. ICM equity at end:   e_end_i   = icm_equity(end_stacks, payouts)[i]
  5. ICM utility delta:   icm_return_i = e_end_i - e_start_i

The per-player icm_return is what the CFR solver should treat as the
terminal utility instead of chip_return.

This module does NOT touch the trajectory walker — it operates on the
walker's output. Callers compose: walk_game(...) -> apply_icm(...).
"""
from __future__ import annotations
from typing import Sequence

from src.nlhe.icm import icm_equity
from src.nlhe.trajectory6 import Trajectory


def icm_adjust_returns(
    chip_returns: Sequence[float],
    starting_stacks: Sequence[int],
    payouts: Sequence[float],
) -> list[float]:
    """Transform chip-EV terminal returns into ICM-EV returns.

    Args:
        chip_returns: per-player chip P/L from a single hand
            (state.returns() from OpenSpiel). Length N.
        starting_stacks: per-player chip count at the START of this
            hand (before blinds were posted). Length N.
        payouts: tournament prize pool by finish position. Length K <= N.

    Returns:
        Per-player ICM utility delta (equity gained or lost), length N.
        Sum is ~0 (the prize pool is conserved by the ICM map).

    Example:
        # Two players post-blind: SB lost 50, BB lost 100, BB doubled SB's stack.
        # Starting stacks [1500, 1500, ...], hand played, returns = [-1500, +1500, 0, 0, 0, 0].
        # ICM transformation maps chip delta to equity delta.
    """
    n = len(chip_returns)
    if len(starting_stacks) != n:
        raise ValueError(
            f"starting_stacks length {len(starting_stacks)} != chip_returns length {n}"
        )
    if any(s < 0 for s in starting_stacks):
        raise ValueError("starting_stacks must be non-negative")

    # End stacks after the hand
    end_stacks = [starting_stacks[i] + chip_returns[i] for i in range(n)]

    # Guard: very small floating-point drift in returns can produce stacks
    # slightly below zero (e.g. -1e-9). Clamp to zero before ICM.
    end_stacks = [max(0.0, float(s)) for s in end_stacks]

    e_start = icm_equity(starting_stacks, payouts)
    e_end = icm_equity(end_stacks, payouts)

    return [e_end[i] - e_start[i] for i in range(n)]


def icm_adjust_trajectory(
    trajectory: Trajectory,
    starting_stacks: Sequence[int],
    payouts: Sequence[float],
) -> list[float]:
    """Convenience wrapper: apply ICM to a Trajectory's terminal_returns.

    Doesn't mutate the trajectory — returns the adjusted per-player
    utilities as a new list.
    """
    return icm_adjust_returns(
        chip_returns=trajectory.terminal_returns,
        starting_stacks=starting_stacks,
        payouts=payouts,
    )
