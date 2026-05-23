"""ICM (Independent Chip Model) value function — Malmuth-Harville.

Phase 4b (docs/PHASE4_PLAN.md). Converts a tournament chip-state into
per-player equity in the prize pool. The key abstraction for training
the 6-max SNG bot: chip-EV is wrong for tournaments where the payout
structure caps how much money any player can win above survival.

Algorithm: Malmuth-Harville recursive formula.

For chip fractions x_1, ..., x_k (the players we're computing the
"finish in this order" probability for), the probability they finish
in positions 1st through kth in that specific order is:

    p = (x_1 · x_2 · ... · x_k) /
        ((1-x_1) · (1-x_1-x_2) · ... · (1-x_1-...-x_{k-1}))

Each player's ICM equity = sum over payout positions p of
(probability player finishes in position p) × payout[p].

Algorithmic complexity: O(N! / (N-K)!) where N is number of players and
K is number of paid positions. For 6-max SNG (N=6, K=3), this is 120
permutations — tractable. For ~10+ paid positions it becomes slow.

References:
- Harville (1973), "Assigning probabilities to the outcomes of
  multi-entry competitions"
- Malmuth (1987), "Gambling Theory and Other Topics"
- arxiv 0911.3100, "The Independent Chip Model and Risk Aversion"

This module is PURE MATH. No solver integration. It computes equities
given stacks and payouts, nothing more.
"""
from __future__ import annotations
from itertools import permutations
from typing import Sequence


def icm_equity(stacks: Sequence[float], payouts: Sequence[float]) -> list[float]:
    """Compute per-player ICM equity via Malmuth-Harville.

    Args:
        stacks: Per-player chip counts (or stack sizes). Length N. All
            entries must be >= 0. Players with stack 0 are "busted" and
            get 0 equity.
        payouts: Prize pool per finish position. Length K, where K is
            the number of paid positions. payouts[0] is 1st-place prize,
            payouts[1] is 2nd-place prize, etc. K can be <= N (the rest
            of the field gets nothing).

    Returns:
        Per-player ICM equity, length N. Sum equals sum of payouts
        (the entire prize pool is distributed).

    Examples:
        >>> # Equal stacks, equal payouts: each gets prize_pool/N
        >>> icm_equity([100, 100, 100], [50, 30, 20])
        [33.333..., 33.333..., 33.333...]

        >>> # One player dominates: gets close to 1st prize
        >>> icm_equity([10000, 1, 1], [50, 30, 20])
        # First player ~50, others split ~50 between 2nd and 3rd
    """
    n = len(stacks)
    if n == 0:
        return []
    if any(s < 0 for s in stacks):
        raise ValueError(f"stacks must be non-negative, got {stacks}")
    total_chips = sum(stacks)
    if total_chips == 0:
        # No chips in play — equity is 0 for everyone.
        return [0.0] * n
    k = len(payouts)
    if k == 0:
        # No payouts — equity is 0 for everyone.
        return [0.0] * n
    if k > n:
        # More paid positions than players — pad payouts to N by
        # ignoring the extras. (Common scenario: K = N for full ITM.)
        payouts = list(payouts[:n])
        k = n

    # Identify active players (stack > 0).
    active_idx = [i for i, s in enumerate(stacks) if s > 0]
    n_active = len(active_idx)
    busted_idx = [i for i, s in enumerate(stacks) if s == 0]
    n_busted = len(busted_idx)

    # Convert stacks to fractions of the active chip pool (so active fractions sum to 1).
    fractions = [s / total_chips for s in stacks]

    equity = [0.0] * n

    # Busted players lock in the BOTTOM payouts (they've already finished).
    # If 2 players busted and there are 3 payouts, payouts[1] and payouts[2]
    # are taken by the busted players (sum split equally — we don't track
    # bust order). The remaining active players compete for payouts[0..n_active-1].
    if n_busted > 0 and k > n_active:
        # The bottom (k - n_active) payouts go to the busted players, split equally.
        bottom_payouts = payouts[n_active:k]
        bottom_total = sum(bottom_payouts)
        per_busted = bottom_total / n_busted if n_busted > 0 else 0.0
        for bi in busted_idx:
            equity[bi] = per_busted

    # Active players compete for the top payouts.
    payouts = list(payouts[:n_active])
    k_effective = len(payouts)

    # For each ordering of active players into the first k_effective
    # finish positions, compute P(this specific ordering) and accumulate
    # each player's contribution to their equity in that position.
    for perm in permutations(active_idx, k_effective):
        # P(perm[0] finishes 1st, perm[1] finishes 2nd, ..., perm[k-1] finishes kth)
        # Numerator: product of chip fractions of players in this order.
        # Denominator: each successive denom is (1 - sum of fractions removed so far).
        prob = 1.0
        removed_sum = 0.0
        for pos, player_idx in enumerate(perm):
            if pos == 0:
                # First place: P(this player wins) = x_i / 1
                prob *= fractions[player_idx]
            else:
                # Conditional probability given previous players removed.
                denom = 1.0 - removed_sum
                if denom <= 0:
                    # Should never happen with active players, but guard.
                    prob = 0.0
                    break
                prob *= fractions[player_idx] / denom
            removed_sum += fractions[player_idx]

        # Each player in this ordering contributes their payout × probability.
        for pos, player_idx in enumerate(perm):
            equity[player_idx] += prob * payouts[pos]

    return equity


def icm_equity_normalized(stacks: Sequence[float], payouts: Sequence[float]) -> list[float]:
    """ICM equity expressed as fractions of the total prize pool.

    Each entry is in [0, 1] and the sum is 1 (with floating-point
    tolerance). Useful for converting equities to "tournament equity
    share" semantics for terminal utility calculation.
    """
    raw = icm_equity(stacks, payouts)
    total = sum(payouts)
    if total == 0:
        return [0.0] * len(raw)
    return [e / total for e in raw]


def sng_payouts_6max_double_up(buy_in: float = 1.0) -> list[float]:
    """Ignition 6-max Double Up: top 3 each get 2x buy-in (equal split).

    The "top-3 equal payout" mode the project targets. Each of the 6
    players pays `buy_in`. Total prize pool = 6 * buy_in. Top 3 finishers
    each get 2 * buy_in. 4th, 5th, 6th get nothing.

    Strategic shape: bubble is at 4 players left (one elimination from
    the money). All in-the-money positions have identical equity, so
    once 3 players remain, all remaining chips have zero marginal EV
    (the "ITM degenerate case").

    Default buy_in=1.0 returns [2.0, 2.0, 2.0].
    """
    return [2.0 * buy_in, 2.0 * buy_in, 2.0 * buy_in]


def sng_payouts_6max_standard(buy_in: float = 1.0,
                              first_share: float = 0.65) -> list[float]:
    """Ignition 6-max Standard: top 2 paid (first_share / 1 - first_share).

    The "1st takes most, 2nd takes less" mode. Default first_share=0.65
    (industry standard 65/35 split of the prize pool). 3rd through 6th
    get nothing.

    Strategic shape: bubble is at 3 players left (one elimination from
    the money). Late-game pressure concentrates on the 3rd-place player.
    NO degenerate ITM phase — there is always strict equity ordering
    between 1st and 2nd payouts.

    Default buy_in=1.0, first_share=0.65 returns [3.90, 2.10].
    """
    if not 0.5 <= first_share <= 1.0:
        raise ValueError(f"first_share must be in [0.5, 1.0], got {first_share}")
    prize_pool = 6.0 * buy_in
    return [first_share * prize_pool, (1.0 - first_share) * prize_pool]


# Backward-compatible alias: the old name still works but points to
# Standard. (Code that called sng_payouts_6max() got 50/30/20 — that
# structure is not actually offered at Ignition. Callers should migrate
# to one of the named functions above.)
def sng_payouts_6max(buy_in: float = 1.0) -> list[float]:
    """DEPRECATED. Use sng_payouts_6max_double_up or sng_payouts_6max_standard.

    Original implementation returned 50/30/20 of prize pool, which is not
    a structure Ignition actually offers in 6-max. Kept temporarily for
    backward compatibility; new code should use the explicit Ignition
    variants.
    """
    import warnings
    warnings.warn(
        "sng_payouts_6max is deprecated; use sng_payouts_6max_double_up "
        "(top-3 equal) or sng_payouts_6max_standard (top-2 65/35) instead",
        DeprecationWarning, stacklevel=2,
    )
    prize_pool = 6.0 * buy_in
    return [0.50 * prize_pool, 0.30 * prize_pool, 0.20 * prize_pool]


def is_bubble(stacks: Sequence[float], paid_positions: int) -> bool:
    """True if exactly `paid_positions + 1` players have non-zero stacks.

    On the bubble = one elimination away from any player making the
    money. Strategic importance is maximum: ICM pressure is highest.

    Args:
        stacks: per-player chip counts.
        paid_positions: number of paid positions in the tournament.
            Double Up 6-max: paid_positions=3 (bubble at 4 left).
            Standard 6-max:  paid_positions=2 (bubble at 3 left).
    """
    active = sum(1 for s in stacks if s > 0)
    return active == paid_positions + 1


def is_itm(stacks: Sequence[float], paid_positions: int) -> bool:
    """True if remaining active players are all in the money.

    Defined as: number of active players <= paid_positions. Once
    we're ITM, all remaining players have at least min-cash guaranteed,
    so any further elimination is for higher payouts.
    """
    active = sum(1 for s in stacks if s > 0)
    return active <= paid_positions
