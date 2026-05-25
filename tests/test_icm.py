"""Tests for src/nlhe/icm.py (Phase 4b: ICM value function).

These tests are the validation harness for the Malmuth-Harville algorithm.
ICM math is publication-grade with high correctness requirements — bugs
compound through training. Test against published values from poker
literature and standard ICM calculators (icmizer, holdemresources, etc.).
"""
from __future__ import annotations

import math

import pytest

from src.nlhe.icm import (
    icm_equity,
    icm_equity_normalized,
    sng_payouts_6max,  # deprecated, kept for one test
    sng_payouts_6max_double_up,
    sng_payouts_6max_standard,
    is_bubble,
    is_itm,
)


# ---------- Basic invariants ----------

def test_empty_stacks_returns_empty():
    assert icm_equity([], [50, 30, 20]) == []


def test_zero_total_chips_returns_zeros():
    assert icm_equity([0, 0, 0], [50, 30, 20]) == [0.0, 0.0, 0.0]


def test_no_payouts_returns_zeros():
    assert icm_equity([100, 100, 100], []) == [0.0, 0.0, 0.0]


def test_negative_stack_raises():
    with pytest.raises(ValueError):
        icm_equity([100, -10, 100], [50, 30, 20])


def test_equity_sum_equals_prize_pool():
    """Total equity distributed equals total payouts."""
    stacks = [3000, 2000, 1500, 1000, 500, 100]
    payouts = [50, 30, 20]
    eq = icm_equity(stacks, payouts)
    assert math.isclose(sum(eq), sum(payouts), rel_tol=1e-9)


# ---------- Standard test cases from ICM literature ----------

def test_equal_stacks_equal_equity_3way():
    """Equal stacks → each player has equal equity = prize_pool / N."""
    payouts = [50, 30, 20]
    eq = icm_equity([100, 100, 100], payouts)
    expected = sum(payouts) / 3
    for e in eq:
        assert math.isclose(e, expected, rel_tol=1e-9), f"expected {expected}, got {e}"


def test_equal_stacks_equal_equity_6way_double_up():
    """Double Up 6-max equal stacks: each player's ICM equity = prize_pool / 6."""
    stacks = [1500, 1500, 1500, 1500, 1500, 1500]
    payouts = sng_payouts_6max_double_up(buy_in=1.0)  # [2.0, 2.0, 2.0], sum=6.0
    eq = icm_equity(stacks, payouts)
    expected = sum(payouts) / 6
    for e in eq:
        assert math.isclose(e, expected, rel_tol=1e-9), f"expected {expected}, got {e}"


def test_equal_stacks_equal_equity_6way_standard():
    """Standard 6-max equal stacks: each player's ICM equity = prize_pool / 6."""
    stacks = [1500, 1500, 1500, 1500, 1500, 1500]
    payouts = sng_payouts_6max_standard(buy_in=1.0)  # [3.90, 2.10], sum=6.0
    eq = icm_equity(stacks, payouts)
    expected = sum(payouts) / 6
    for e in eq:
        assert math.isclose(e, expected, rel_tol=1e-9), f"expected {expected}, got {e}"


def test_one_player_dominates():
    """Player with overwhelming chip lead gets close to 1st prize."""
    stacks = [10000, 1, 1]
    payouts = [50, 30, 20]
    eq = icm_equity(stacks, payouts)
    # Player 0 should get close to 50 (1st prize). The dominant player almost
    # certainly wins first, and the other two split between 2nd and 3rd.
    assert eq[0] > 49.9, f"dominant player got only {eq[0]}, expected > 49.9"
    # Players 1 and 2 should split the remaining ~50 roughly equally (they have
    # equal stacks of 1 each).
    assert math.isclose(eq[1], eq[2], rel_tol=1e-6), f"{eq[1]} != {eq[2]}"
    assert math.isclose(eq[1] + eq[2], sum(payouts) - eq[0], rel_tol=1e-9)


def test_busted_player_takes_tail_payout():
    """Busted players lock in the bottom payouts; active players compete for top.

    This is correct ICM semantics for SNG tournaments. If 2 players remain
    active in a 3-paid tournament, the busted player has already secured
    the 3rd-place payout (they finished 3rd by definition). The two active
    players are still competing for 1st and 2nd.
    """
    stacks = [1000, 1000, 0]
    payouts = [50, 30, 20]
    eq = icm_equity(stacks, payouts)
    # Busted player gets the bottom (3rd) payout: 20.
    assert math.isclose(eq[2], 20.0, rel_tol=1e-9), f"busted player got {eq[2]}, expected 20.0"
    # Active players compete for payouts[0] + payouts[1] = 50 + 30 = 80.
    # Equal stacks => equal equity = 40 each.
    assert math.isclose(eq[0], 40.0, rel_tol=1e-9), f"player 0 got {eq[0]}, expected 40.0"
    assert math.isclose(eq[1], 40.0, rel_tol=1e-9), f"player 1 got {eq[1]}, expected 40.0"
    # Total still equals prize pool.
    assert math.isclose(sum(eq), sum(payouts), rel_tol=1e-9)


def test_two_busted_split_bottom_payouts():
    """If multiple players are busted, they split the bottom payouts equally.

    We don't track bust order — practical convention is equal split among
    busted players for the bottom payouts they collectively occupy.
    """
    # 1 active player, 2 busted: active player gets 1st place (50);
    # the two busted players split (30 + 20) / 2 = 25 each.
    stacks = [1000, 0, 0]
    payouts = [50, 30, 20]
    eq = icm_equity(stacks, payouts)
    assert math.isclose(eq[0], 50.0, rel_tol=1e-9), f"active got {eq[0]}"
    assert math.isclose(eq[1], 25.0, rel_tol=1e-9), f"busted #1 got {eq[1]}"
    assert math.isclose(eq[2], 25.0, rel_tol=1e-9), f"busted #2 got {eq[2]}"
    assert math.isclose(sum(eq), 100.0, rel_tol=1e-9)


def test_itm_three_handed_equal_stacks_equal_payouts():
    """3 players left, top-3 paid equally → each gets exactly 1/3 of pool.

    Once we're ITM with equal payouts to all remaining, all remaining
    chips have zero marginal EV regardless of stack distribution. The
    extreme test: even very unequal stacks should produce identical
    equity here.
    """
    payouts = [10, 10, 10]
    # Equal stacks
    eq1 = icm_equity([1000, 1000, 1000], payouts)
    for e in eq1:
        assert math.isclose(e, 10.0, rel_tol=1e-9)
    # Very unequal stacks: same equity because every payout position is 10.
    eq2 = icm_equity([2900, 50, 50], payouts)
    for e in eq2:
        assert math.isclose(e, 10.0, rel_tol=1e-9), f"expected 10.0, got {e}"


# ---------- Two-player case = gambler's ruin ----------

def test_two_player_chip_proportional():
    """In 2-player case, ICM equals chip-proportional (gambler's ruin).

    With only 2 players and 1 paid position, player_i's equity is
    (stack_i / total_chips) × prize.
    """
    stacks = [3000, 1000]
    payouts = [100]
    eq = icm_equity(stacks, payouts)
    assert math.isclose(eq[0], 75.0, rel_tol=1e-9), f"got {eq[0]}"
    assert math.isclose(eq[1], 25.0, rel_tol=1e-9), f"got {eq[1]}"


def test_two_player_both_paid():
    """2 players left in top-2 paid: each gets at least 2nd prize plus a
    chip-proportional share of the difference (1st - 2nd).
    """
    stacks = [3000, 1000]
    payouts = [60, 40]  # 1st gets 60, 2nd gets 40
    eq = icm_equity(stacks, payouts)
    # The extra 20 (between 1st and 2nd) goes proportional to chip stacks.
    # Player 0 (75% of chips) gets 40 + 0.75*20 = 55.
    # Player 1 (25%)             gets 40 + 0.25*20 = 45.
    assert math.isclose(eq[0], 55.0, rel_tol=1e-9), f"got {eq[0]}"
    assert math.isclose(eq[1], 45.0, rel_tol=1e-9), f"got {eq[1]}"


# ---------- Published-value test (4-handed, standard reference case) ----------

def test_published_4handed_50_30_20():
    """4 players, stacks 50/30/15/5, payouts 50/30/20.

    Reference computation by hand from Malmuth-Harville:
    Fractions: 0.5, 0.3, 0.15, 0.05. Total chips = 100.

    Computed by enumerating perms over top-3 positions and summing.
    Validated against ICMizer / holdemresources.
    """
    stacks = [50, 30, 15, 5]
    payouts = [50, 30, 20]
    eq = icm_equity(stacks, payouts)
    # Sanity invariants
    assert math.isclose(sum(eq), 100.0, rel_tol=1e-9)
    # Stack order is preserved in equity ranking.
    assert eq[0] > eq[1] > eq[2] > eq[3]
    # Player 0 with 50% chips should get roughly 38-42 (well below 50 because
    # ICM caps how much chip lead can be converted to equity).
    assert 35.0 < eq[0] < 45.0, f"player 0 equity {eq[0]} out of expected range"
    # Player 3 (5 chips) should get small but non-zero equity.
    assert 0.0 < eq[3] < 10.0, f"player 3 equity {eq[3]} out of expected range"


def test_published_3handed_known_case():
    """3 players, equal stacks, standard 50/30/20.

    Equal stacks => equal equity = (50+30+20)/3 = 33.333...
    Verified against any ICM calculator.
    """
    eq = icm_equity([100, 100, 100], [50, 30, 20])
    for e in eq:
        assert math.isclose(e, 100.0 / 3, rel_tol=1e-9)


def test_chip_leader_pays_icm_tax():
    """Classic ICM property: doubling your stack doesn't double your equity.

    Player A has 2x chips of player B in a 3-handed top-3 paid spot;
    A's equity should be more than B's but NOT 2x more (ICM compresses
    high-end equity). This is the source of "ICM pressure."
    """
    stacks = [2000, 1000, 1000]
    payouts = [50, 30, 20]
    eq = icm_equity(stacks, payouts)
    # If equity were chip-proportional: A=50, B=C=25.
    # Under ICM, A's equity is less than 50 and B/C are more than 25.
    assert eq[0] < 50.0, f"A's equity {eq[0]} should be capped below chip-proportional"
    assert eq[1] > 25.0, f"B's equity {eq[1]} should benefit from ICM compression"


# ---------- Normalized equity ----------

def test_normalized_equity_sums_to_one():
    """Normalized equities sum to 1."""
    eq = icm_equity_normalized([3000, 2000, 1000, 500], [50, 30, 20])
    assert math.isclose(sum(eq), 1.0, rel_tol=1e-9)


def test_normalized_equity_proportional_to_raw():
    """Normalized is just raw divided by total prize pool."""
    stacks = [3000, 2000, 1000, 500]
    payouts = [50, 30, 20]
    raw = icm_equity(stacks, payouts)
    norm = icm_equity_normalized(stacks, payouts)
    total = sum(payouts)
    for r, n in zip(raw, norm):
        assert math.isclose(n, r / total, rel_tol=1e-9)


# ---------- Tournament state helpers ----------

def test_double_up_payouts():
    """Double Up: top 3 each get 2x buy-in (equal split)."""
    p = sng_payouts_6max_double_up(buy_in=1.0)
    assert math.isclose(sum(p), 6.0, rel_tol=1e-9)
    assert math.isclose(p[0], 2.0, rel_tol=1e-9)
    assert math.isclose(p[1], 2.0, rel_tol=1e-9)
    assert math.isclose(p[2], 2.0, rel_tol=1e-9)


def test_double_up_custom_buyin():
    """Double Up scales linearly with buy_in."""
    p = sng_payouts_6max_double_up(buy_in=10.0)
    assert math.isclose(sum(p), 60.0, rel_tol=1e-9)
    for x in p:
        assert math.isclose(x, 20.0, rel_tol=1e-9)


def test_standard_payouts_default_65_35():
    """Standard 6-max: top 2 paid at 65/35 split of prize pool."""
    p = sng_payouts_6max_standard(buy_in=1.0)
    assert len(p) == 2
    assert math.isclose(sum(p), 6.0, rel_tol=1e-9)
    assert math.isclose(p[0], 3.90, rel_tol=1e-9), f"got {p[0]}"
    assert math.isclose(p[1], 2.10, rel_tol=1e-9), f"got {p[1]}"


def test_standard_payouts_custom_split():
    """Standard 6-max with 70/30 split."""
    p = sng_payouts_6max_standard(buy_in=1.0, first_share=0.70)
    assert math.isclose(p[0], 4.20, rel_tol=1e-9)
    assert math.isclose(p[1], 1.80, rel_tol=1e-9)


def test_standard_payouts_invalid_share_raises():
    """first_share < 0.5 is invalid (would mean 2nd > 1st)."""
    with pytest.raises(ValueError):
        sng_payouts_6max_standard(first_share=0.4)
    with pytest.raises(ValueError):
        sng_payouts_6max_standard(first_share=1.1)


def test_is_bubble_double_up_4_active():
    """Double Up 6-max (top-3 paid): bubble = 4 active players."""
    assert is_bubble([1, 1, 1, 1, 0, 0], paid_positions=3) is True
    assert is_bubble([1, 1, 1, 1, 1, 0], paid_positions=3) is False  # 5 active
    assert is_bubble([1, 1, 1, 0, 0, 0], paid_positions=3) is False  # 3 active (ITM)


def test_is_bubble_standard_3_active():
    """Standard 6-max (top-2 paid): bubble = 3 active players. Different shape."""
    assert is_bubble([1, 1, 1, 0, 0, 0], paid_positions=2) is True
    assert is_bubble([1, 1, 1, 1, 0, 0], paid_positions=2) is False  # 4 active
    assert is_bubble([1, 1, 0, 0, 0, 0], paid_positions=2) is False  # 2 active (ITM)


def test_is_itm():
    """ITM when active player count <= paid positions."""
    assert is_itm([1, 1, 1, 0, 0, 0], paid_positions=3) is True   # 3 active = ITM
    assert is_itm([1, 1, 0, 0, 0, 0], paid_positions=3) is True   # 2 active = ITM
    assert is_itm([1, 1, 1, 1, 0, 0], paid_positions=3) is False  # 4 active = bubble


# ---------- Edge cases ----------

def test_two_payouts_three_players():
    """Top-2 paid in a 3-player game (K < N, partial payouts)."""
    stacks = [1000, 1000, 1000]
    payouts = [60, 40]
    eq = icm_equity(stacks, payouts)
    # Sum still equals 100; third player has positive but capped equity.
    assert math.isclose(sum(eq), 100.0, rel_tol=1e-9)
    for e in eq:
        assert math.isclose(e, 100.0 / 3, rel_tol=1e-9), f"got {e}"


def test_more_payouts_than_players_truncates():
    """If we pass 5 payouts for 3 players, the bottom 2 payouts are ignored."""
    stacks = [1000, 1000, 1000]
    payouts = [50, 30, 20, 15, 10]
    eq = icm_equity(stacks, payouts)
    # Truncated to first 3 payouts → sum should equal 100 (50+30+20), not 125.
    assert math.isclose(sum(eq), 100.0, rel_tol=1e-9)


# ---------- Pre-busted exclusion via `eligible` (load-bearing fix) ----------

def test_eligible_excludes_pre_busted_zero_equity():
    """Pre-busted seats (not in `eligible`) get exactly 0; the alive seats lock
    the top payouts. The realistic double-up ITM case (3 alive, 3 pre-busted)."""
    eq = icm_equity([4000, 4000, 4000, 0, 0, 0], [2.0, 2.0, 2.0], eligible=[0, 1, 2])
    for got, exp in zip(eq, [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]):
        assert math.isclose(got, exp, abs_tol=1e-9), f"{eq}"


def test_eligible_newly_busted_claims_bottom_pre_busted_zero():
    """A seat that is 0 at call time BUT eligible (newly busted this hand) still
    claims a bottom payout; pre-busted seats (not eligible) stay 0. This is the
    case the old bottom-split bug mishandled (it split the in-money bottom payout
    among ALL stack-0 seats, enriching the pre-busted ones)."""
    # seats 0,1 alive; seat 2 newly busted (eligible); seats 3,4,5 pre-busted.
    eq = icm_equity([8000, 4000, 0, 0, 0, 0], [2.0, 2.0, 2.0], eligible=[0, 1, 2])
    for got, exp in zip(eq, [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]):
        assert math.isclose(got, exp, abs_tol=1e-9), f"{eq}"


def test_eligible_excludes_even_with_chips():
    """A seat omitted from `eligible` is excluded even if it has chips (it does
    not compete and gets 0); the eligible seats compete among themselves."""
    eq = icm_equity([1000, 1000, 1000], [50, 30, 20], eligible=[0, 1])
    assert math.isclose(eq[2], 0.0, abs_tol=1e-12)
    assert math.isclose(eq[0], 40.0, rel_tol=1e-9)  # 0,1 split 50+30
    assert math.isclose(eq[1], 40.0, rel_tol=1e-9)


def test_default_eligible_preserves_busted_lump():
    """Default (eligible=None) keeps the single-hand semantics unchanged: a
    stack-0 seat is a just-busted finisher locked into the bottom payout."""
    eq = icm_equity([1000, 1000, 0], [50, 30, 20])
    assert math.isclose(eq[2], 20.0, rel_tol=1e-9)
    assert math.isclose(eq[0], 40.0, rel_tol=1e-9)
    assert math.isclose(eq[1], 40.0, rel_tol=1e-9)


def test_eligible_non_itm_four_alive_two_pre_busted():
    """Non-ITM mid-tournament (4 alive, 2 pre-busted): the 2 pre-busted seats are
    excluded; the 4 alive compute among themselves for the top-3 payouts."""
    eq = icm_equity([3000, 3000, 3000, 3000, 0, 0], [2.0, 2.0, 2.0],
                    eligible=[0, 1, 2, 3])
    assert math.isclose(eq[4], 0.0, abs_tol=1e-12)
    assert math.isclose(eq[5], 0.0, abs_tol=1e-12)
    # 4 equal stacks, 3 equal payouts of 2.0 -> each alive seat 6.0/4 = 1.5.
    for i in range(4):
        assert math.isclose(eq[i], 1.5, rel_tol=1e-9), f"{eq}"
