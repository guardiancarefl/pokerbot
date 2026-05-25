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


def icm_equity(stacks: Sequence[float], payouts: Sequence[float],
               eligible: Sequence[int] | None = None) -> list[float]:
    """Compute per-player ICM equity via Malmuth-Harville.

    Args:
        stacks: Per-player chip counts (or stack sizes). Length N. All
            entries must be >= 0.
        payouts: Prize pool per finish position. Length K, where K is
            the number of paid positions. payouts[0] is 1st-place prize,
            payouts[1] is 2nd-place prize, etc. K can be <= N (the rest
            of the field gets nothing).
        eligible: optional list of seat indices that are "in the running"
            for this calculation. **This is how a stack-0 seat's meaning is
            disambiguated:**

              - eligible=None (default): every seat is in the running. A
                stack-0 seat is treated as a player who JUST busted and is
                locked into a bottom payout (split equally among the stack-0
                seats). This is the correct single-hand-terminal semantics and
                is backward-compatible.
              - eligible given: only those seats participate; seats NOT in
                `eligible` are PRE-BUSTED (entered already eliminated, out of
                the money) and get EXACTLY 0. Among the eligible seats, a
                stack-0 one is a NEWLY busted finisher and still claims a
                bottom payout. Callers that know which seats entered at stack 0
                (e.g. icm_returns.icm_adjust_returns, from starting_stacks) pass
                this to avoid handing pre-busted seats spurious bottom-payout
                equity.

    Returns:
        Per-player ICM equity, length N. With eligible=None the sum equals the
        prize pool; with an eligible set smaller than the number of paid
        positions the sum is the prize awardable among the eligible seats
        (the rest belongs to already-finished players not represented here).

    Examples:
        >>> # Equal stacks, equal payouts: each gets prize_pool/N
        >>> icm_equity([100, 100, 100], [50, 30, 20])
        [33.333..., 33.333..., 33.333...]

        >>> # Pre-busted seats excluded: 3 alive lock the top-3, the rest get 0
        >>> icm_equity([4000, 4000, 4000, 0, 0, 0], [2, 2, 2], eligible=[0, 1, 2])
        [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
    """
    n = len(stacks)
    if n == 0:
        return []
    if any(s < 0 for s in stacks):
        raise ValueError(f"stacks must be non-negative, got {stacks}")
    k = len(payouts)
    if k == 0:
        # No payouts — equity is 0 for everyone.
        return [0.0] * n
    if k > n:
        # More paid positions than players — ignore the extras.
        payouts = list(payouts[:n])
        k = n

    # Eligible seats = those "in the running" for this calculation. Default: all
    # seats (a stack-0 seat is a just-busted finisher; backward-compatible). When
    # given, seats NOT in `eligible` are PRE-BUSTED (already out) and get 0.
    if eligible is None:
        elig = list(range(n))
    else:
        elig = sorted({int(i) for i in eligible if 0 <= int(i) < n})

    equity = [0.0] * n
    if not elig:
        return equity

    # Within the eligible set: `live` seats (stack > 0) compete via Malmuth-
    # Harville for the TOP positions; `zero` seats (stack == 0) are just-busted
    # finishers who split the BOTTOM contested payouts equally. Pre-busted seats
    # (not eligible) keep 0.
    live = [i for i in elig if stacks[i] > 0]
    zero = [i for i in elig if stacks[i] == 0]
    if not live:
        # No chips among the eligible seats — no equity to distribute.
        return equity

    a = len(elig)                       # eligible finishing positions: 1..a
    contested = list(payouts[:a])       # payouts awardable among the eligible
    n_live = len(live)
    top = contested[:n_live]            # competed for by live seats
    bottom = contested[n_live:]         # locked by just-busted (zero) seats

    if zero:
        per_zero = sum(bottom) / len(zero)
        for i in zero:
            equity[i] = per_zero

    k_top = len(top)
    if k_top == 0:
        return equity

    # Malmuth-Harville over the LIVE seats only; fractions are over live chips.
    total_live = sum(stacks[i] for i in live)
    fractions = {i: stacks[i] / total_live for i in live}
    for perm in permutations(live, k_top):
        # P(perm[0] finishes 1st among live, perm[1] 2nd, ...).
        prob = 1.0
        removed_sum = 0.0
        for pos, player_idx in enumerate(perm):
            if pos == 0:
                prob *= fractions[player_idx]
            else:
                denom = 1.0 - removed_sum
                if denom <= 0:
                    prob = 0.0
                    break
                prob *= fractions[player_idx] / denom
            removed_sum += fractions[player_idx]
        for pos, player_idx in enumerate(perm):
            equity[player_idx] += prob * top[pos]

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
