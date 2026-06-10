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

    # Eligible = seats that ENTERED the hand with chips. Seats starting at 0 are
    # PRE-BUSTED (already eliminated, out of the money) and must be excluded from
    # the ICM map for BOTH endpoints — otherwise, when an alive player busts during
    # the hand, the in-money bottom payout gets split among ALL stack-0 seats
    # (pre-busted included), spuriously enriching already-finished seats and
    # mis-pricing the newly-busted one. Passing `eligible` keeps pre-busted seats
    # at 0 and lets the newly-busted seat (eligible but 0 at end) correctly claim
    # its finishing payout. For a normal all-alive hand, eligible = all seats and
    # this is a no-op (behavior unchanged).
    eligible = [i for i in range(n) if starting_stacks[i] > 0]

    e_start = icm_equity(starting_stacks, payouts, eligible=eligible)
    e_end = icm_equity(end_stacks, payouts, eligible=eligible)

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


# ---------------------------------------------------------------------------
# Placeholder pot-misassignment fix (scoring path only — NOT training).
# ---------------------------------------------------------------------------
#
# `game_strings.to_inner_game_string_for_state` models busted seats as
# stack=1 / ante=0 / blind=0 placeholders (universal_poker cannot drop
# seats). A placeholder posts nothing and is never in firstPlayer rotation,
# so it never acts — but universal_poker treats an unacted player as LIVE.
# When all alive players fold preflop and the fold-around lands on a
# placeholder, the placeholder is the "last live player" and wins the pot;
# the chips (all contributed by alive seats) are then destroyed when the
# caller zeroes placeholder stacks. Measured leak: ~1.5-7.7% of shorthanded
# hands depending on opponent fold frequency, always exactly one placeholder.
#
# This corrects the chip-EV return vector by reassigning any placeholder
# winnings to the rightful alive winner, BEFORE icm_adjust_returns. It is a
# pure function of the terminal return vector + the fold record. It does NOT
# touch what any player observed during the hand (parse_state_6max and the
# encoder are upstream and unchanged), so it preserves the deployed model's
# observation distribution exactly. It is deliberately NOT applied in the
# training path (cfr6) — the deployed model was trained on the uncorrected
# convention and is not being retrained.

_LEAK_EPS = 1e-9


def conserving_chip_returns(
    raw_returns: Sequence[float],
    starting_stacks: Sequence[int],
    folded: Sequence[bool],
    fold_order: Sequence[int],
    alive_winners: Sequence[int] | None = None,
) -> list[float]:
    """Redistribute any placeholder (busted-seat) winnings to the rightful
    alive winner so the return vector is chip-conserving among real seats.

    Args:
        raw_returns: state.returns() from the 6-seat placeholder game.
        starting_stacks: pre-hand stacks; seats with stack <= 0 are
            placeholders (busted, must not win chips).
        folded: per-seat fold flag from the play loop (True if the seat
            folded this hand).
        fold_order: seats in the order they folded (for the all-alive-fold
            class, the rightful winner is the last alive seat to fold).
        alive_winners: explicit winner set for the showdown class (>= 2
            non-folded alive seats). Required when that class occurs;
            ignored otherwise.

    Returns:
        A new return vector, equal to raw_returns except that placeholder
        winnings are moved to the rightful alive winner(s). Conserves by
        construction: total is unchanged.

    Raises:
        ValueError if the showdown class occurs and alive_winners is not
        supplied (refuses to guess silently).
    """
    n = len(raw_returns)
    out = [float(x) for x in raw_returns]
    placeholders = [i for i in range(n) if starting_stacks[i] <= 0]
    leak_seats = [i for i in placeholders if out[i] > _LEAK_EPS]
    if not leak_seats:
        return out  # no placeholder won anything — nothing to correct

    alive = [i for i in range(n) if starting_stacks[i] > 0]
    surplus = sum(out[i] for i in leak_seats)
    non_folded_alive = [i for i in alive if not folded[i]]

    if alive_winners is not None:
        winners = list(alive_winners)
    elif len(non_folded_alive) == 1:
        winners = non_folded_alive                      # sole survivor
    elif len(non_folded_alive) == 0:
        # All alive folded; the placeholder was the last "live" seat. The
        # rightful winner is the alive seat that folded last (it would have
        # won uncontested were the placeholder not there).
        last_alive = next((s for s in reversed(fold_order) if s in alive),
                          None)
        if last_alive is None:
            raise ValueError(
                "all-alive-fold leak but no alive seat in fold_order")
        winners = [last_alive]
    else:
        raise ValueError(
            f"showdown leak: {len(non_folded_alive)} non-folded alive seats "
            f"with a placeholder winning, but alive_winners not supplied")

    if not winners:
        raise ValueError("no rightful winner resolved for placeholder leak")

    for i in leak_seats:
        out[i] = 0.0
    share = surplus / len(winners)
    for w in winners:
        out[w] += share
    return out


def _parse_cards(card_str: str) -> list[str]:
    """Parse a universal_poker card run like '2d2c' / 'Td9s' into treys
    tokens ['2d','2c'] (rank uppercased, suit kept)."""
    s = card_str.strip().replace(" ", "")
    out = []
    for i in range(0, len(s) - 1, 2):
        out.append(s[i].upper() + s[i + 1].lower())
    return out


def showdown_winners(state, contenders: Sequence[int]) -> list[int]:
    """Best-hand winner(s) among `contenders` at a terminal state, via treys
    (lower rank = stronger). Splits on exact tie. Used only for the rare
    showdown leak class (>= 2 non-folded alive seats with a placeholder
    also winning)."""
    import re as _re
    from treys import Card, Evaluator

    info0 = state.information_state_string(int(contenders[0]))
    board_m = _re.search(r"\[(?:Public|Board): ([^\]]*)\]", info0)
    board = _parse_cards(board_m.group(1)) if board_m else []
    if len(board) < 3:
        # Not enough board to evaluate (shouldn't reach showdown) — fall
        # back to all contenders splitting (still conserving).
        return list(contenders)
    board_cards = [Card.new(c) for c in board]
    ev = Evaluator()
    scores = {}
    for seat in contenders:
        info = state.information_state_string(int(seat))
        priv_m = _re.search(r"\[Private: ([^\]]*)\]", info)
        hole = _parse_cards(priv_m.group(1)) if priv_m else []
        if len(hole) != 2:
            continue
        scores[seat] = ev.evaluate(board_cards, [Card.new(c) for c in hole])
    if not scores:
        return list(contenders)
    best = min(scores.values())
    return [s for s, sc in scores.items() if sc == best]


def conserving_returns_for_terminal(
    state,
    raw_returns: Sequence[float],
    starting_stacks: Sequence[int],
    folded: Sequence[bool],
    fold_order: Sequence[int],
) -> list[float]:
    """State-aware wrapper around `conserving_chip_returns`: resolves the
    showdown class via treys when needed, then redistributes placeholder
    winnings. Hero observation is untouched (operates only on the terminal
    return vector)."""
    n = len(raw_returns)
    placeholders = [i for i in range(n) if starting_stacks[i] <= 0]
    leak_seats = [i for i in placeholders if float(raw_returns[i]) > _LEAK_EPS]
    if not leak_seats:
        return [float(x) for x in raw_returns]
    alive = [i for i in range(n) if starting_stacks[i] > 0]
    non_folded_alive = [i for i in alive if not folded[i]]
    alive_winners = None
    if len(non_folded_alive) >= 2:
        alive_winners = showdown_winners(state, non_folded_alive)
    return conserving_chip_returns(
        raw_returns, starting_stacks, folded, fold_order,
        alive_winners=alive_winners)
