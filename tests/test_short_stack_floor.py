"""Unit tests for the short-stack + check-when-free floors.

Verifies:
  - apply_check_when_free_floor: scoping (to_call==0 + CALL legal), no-op
    when not free, renormalization, identity short-circuit.
  - apply_short_stack_floor: scoping (≤threshold_bb), facing/check/unopened
    regime selection, renormalization, identity short-circuit above threshold.
  - make_live_policy_filter: stacks with AA/KK floor; strictly additive
    (returns input policy reference when no gate fires).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.nlhe.actions import DiscreteAction
from src.nlhe.integration.live_loop import (
    apply_aa_kk_preflop_floor,
    apply_check_when_free_floor,
    apply_short_stack_floor,
    make_live_policy_filter,
)


N_ACT = len(DiscreteAction)
A_FOLD = int(DiscreteAction.FOLD)
A_CALL = int(DiscreteAction.CALL)
A_ALLIN = int(DiscreteAction.ALLIN)
A_BET_33 = int(DiscreteAction.BET_33)
A_BET_50 = int(DiscreteAction.BET_50)
A_BET_66 = int(DiscreteAction.BET_66)
A_BET_100 = int(DiscreteAction.BET_100)
A_BET_150 = int(DiscreteAction.BET_150)
A_BET_200 = int(DiscreteAction.BET_200)


def _mk_policy(fold=0.05, call=0.10, allin=0.20):
    p = np.zeros(N_ACT, dtype=np.float32)
    p[A_FOLD] = fold
    p[A_CALL] = call
    p[A_ALLIN] = allin
    bet_slots = (A_BET_33, A_BET_50, A_BET_66, A_BET_100, A_BET_150, A_BET_200)
    rem = 1.0 - fold - call - allin
    assert rem >= 0
    per = rem / len(bet_slots)
    for b in bet_slots:
        p[b] = per
    assert abs(p.sum() - 1.0) < 1e-5
    return p


def _full_legal():
    return np.ones(N_ACT, dtype=np.float32)


# --------------------------------------------------------------------------
# check-when-free floor
# --------------------------------------------------------------------------

def test_check_free_fires_when_to_call_zero():
    p = _mk_policy(fold=0.30, call=0.20, allin=0.10)
    parsed = {"current_player": 0, "contribution": [50, 50, 50, 50, 50, 50]}
    out = apply_check_when_free_floor(p, _full_legal(), parsed, state=None)
    assert out[A_FOLD] == 0.0
    assert abs(out.sum() - 1.0) < 1e-5
    assert abs(out[A_CALL] - 0.20 / 0.70) < 1e-5
    assert abs(out[A_ALLIN] - 0.10 / 0.70) < 1e-5


def test_check_free_noop_when_to_call_positive():
    p = _mk_policy(fold=0.20)
    parsed = {"current_player": 0,
              "contribution": [25, 50, 100, 0, 100, 0]}  # facing 75 to call
    out = apply_check_when_free_floor(p, _full_legal(), parsed, state=None)
    assert out is p


def test_check_free_noop_when_call_illegal():
    p = _mk_policy(fold=0.20)
    mask = _full_legal()
    mask[A_CALL] = 0.0
    parsed = {"current_player": 0, "contribution": [50] * 6}
    out = apply_check_when_free_floor(p, mask, parsed, state=None)
    assert out is p


def test_check_free_noop_when_fold_mass_zero():
    p = _mk_policy(fold=0.0)
    parsed = {"current_player": 0, "contribution": [50] * 6}
    out = apply_check_when_free_floor(p, _full_legal(), parsed, state=None)
    assert out is p


# --------------------------------------------------------------------------
# short-stack floor
# --------------------------------------------------------------------------

def test_short_stack_noop_above_threshold():
    p = _mk_policy(fold=0.10)
    # 30 BB (BB=50, stack=1500)
    parsed = {"current_player": 0, "money": [1500, 1500, 1500, 1500, 1500, 1500],
              "contribution": [25, 50, 0, 0, 0, 0], "big_blind": 50}
    out = apply_short_stack_floor(p, _full_legal(), parsed, state=None,
                                    threshold_bb=6.0)
    assert out is p


def test_short_stack_fires_facing_action():
    """At 4 BB facing a bet, intermediate sizes masked → only FOLD/CALL/ALLIN."""
    p = _mk_policy(fold=0.10, call=0.15, allin=0.20)
    # 4 BB = 200 chips at BB=50
    parsed = {"current_player": 0, "money": [200, 1500, 1500, 1500, 1500, 1500],
              "contribution": [25, 50, 100, 0, 0, 0],  # facing 75
              "big_blind": 50}
    out = apply_short_stack_floor(p, _full_legal(), parsed, state=None,
                                    threshold_bb=6.0)
    # Intermediate masses → 0
    for b in (A_BET_33, A_BET_50, A_BET_66, A_BET_100, A_BET_150, A_BET_200):
        assert out[b] == 0.0
    # FOLD/CALL/ALLIN kept
    assert out[A_FOLD] > 0
    assert out[A_CALL] > 0
    assert out[A_ALLIN] > 0
    # Sum == 1
    assert abs(out.sum() - 1.0) < 1e-5
    # Renormalization preserved relative shares
    assert abs(out[A_FOLD] / out[A_CALL] - 0.10 / 0.15) < 1e-5
    assert abs(out[A_FOLD] / out[A_ALLIN] - 0.10 / 0.20) < 1e-5


def test_short_stack_fires_check_spot():
    """At 4 BB with CHECK available (to_call==0) → CALL + ALLIN only.
    FOLD is masked too (check-when-free composition handles this; the
    short-stack floor's check-regime is just {CALL, ALLIN})."""
    p = _mk_policy(fold=0.10, call=0.15, allin=0.20)
    parsed = {"current_player": 0, "money": [200, 1500, 1500, 1500, 1500, 1500],
              "contribution": [50, 50, 50, 50, 50, 50],  # all matched (check)
              "big_blind": 50}
    out = apply_short_stack_floor(p, _full_legal(), parsed, state=None,
                                    threshold_bb=6.0)
    # FOLD and intermediates masked
    assert out[A_FOLD] == 0.0
    for b in (A_BET_33, A_BET_50, A_BET_66, A_BET_100, A_BET_150, A_BET_200):
        assert out[b] == 0.0
    assert out[A_CALL] > 0
    assert out[A_ALLIN] > 0
    assert abs(out.sum() - 1.0) < 1e-5


def test_short_stack_fires_unopened_no_check():
    """At 4 BB with no CALL legal (rare; treat as true unopened) → FOLD + ALLIN."""
    p = _mk_policy(fold=0.10, call=0.15, allin=0.20)
    mask = _full_legal()
    mask[A_CALL] = 0.0
    parsed = {"current_player": 0, "money": [200, 1500, 1500, 1500, 1500, 1500],
              "contribution": [50, 50, 50, 50, 50, 50], "big_blind": 50}
    out = apply_short_stack_floor(p, mask, parsed, state=None, threshold_bb=6.0)
    assert out[A_FOLD] > 0
    assert out[A_ALLIN] > 0
    for b in (A_BET_33, A_BET_50, A_BET_66, A_BET_100, A_BET_150, A_BET_200):
        assert out[b] == 0.0
    assert abs(out.sum() - 1.0) < 1e-5


def test_short_stack_uses_min_with_max_opp():
    """Effective stack = min(hero, max(opp)). Hero deep but all opps short."""
    p = _mk_policy(fold=0.10)
    # Hero 1500 (30 BB), max opp 200 (4 BB) → effective = 4 BB
    parsed = {"current_player": 0,
              "money": [1500, 200, 100, 100, 100, 100],
              "contribution": [25, 50, 100, 0, 0, 0], "big_blind": 50}
    out = apply_short_stack_floor(p, _full_legal(), parsed, state=None,
                                    threshold_bb=6.0)
    assert out is not p   # floor fired due to short eff stack


def test_short_stack_threshold_inclusive():
    """≤ threshold means 6.0 BB exactly should fire."""
    p = _mk_policy(fold=0.10)
    parsed = {"current_player": 0, "money": [300, 300, 300, 300, 300, 300],
              "contribution": [25, 50, 100, 0, 0, 0], "big_blind": 50}
    # min(hero, max(opp)) = 300 chips = 6 BB exactly
    out = apply_short_stack_floor(p, _full_legal(), parsed, state=None,
                                    threshold_bb=6.0)
    assert out is not p


def test_short_stack_threshold_strict_above():
    p = _mk_policy(fold=0.10)
    parsed = {"current_player": 0, "money": [301, 301, 301, 301, 301, 301],
              "contribution": [25, 50, 100, 0, 0, 0], "big_blind": 50}
    # 6.02 BB > 6.0 — no fire
    out = apply_short_stack_floor(p, _full_legal(), parsed, state=None,
                                    threshold_bb=6.0)
    assert out is p


# --------------------------------------------------------------------------
# Composed live policy filter
# --------------------------------------------------------------------------

def test_composed_filter_strict_no_op_when_nothing_fires():
    """At a 30 BB preflop spot with 92o facing nothing → no floor fires."""
    p = _mk_policy(fold=0.05, call=0.10, allin=0.10)
    parsed = {"current_player": 0, "money": [1500] * 6,
              "contribution": [25, 50, 100, 0, 0, 0], "big_blind": 50,
              "public_cards": "", "private_cards": "9h2d", "street_idx": 0}
    f = make_live_policy_filter(short_stack_threshold_bb=6.0)
    out = f(p, _full_legal(), parsed, state=None)
    assert out is p


def test_composed_filter_aa_kk_alone():
    """AA preflop, deep stack — only AA/KK floor fires."""
    p = _mk_policy(fold=0.05)
    parsed = {"current_player": 0, "money": [1500] * 6,
              "contribution": [25, 50, 100, 0, 0, 0], "big_blind": 50,
              "public_cards": "", "private_cards": "AsAh", "street_idx": 0}
    f = make_live_policy_filter(short_stack_threshold_bb=6.0)
    out = f(p, _full_legal(), parsed, state=None)
    assert out[A_FOLD] == 0.0
    # Intermediate sizes remain (deep stack)
    assert out[A_BET_100] > 0


def test_composed_filter_short_stack_alone():
    """5 BB facing action, no AA/KK, no check spot — only short-stack floor."""
    p = _mk_policy(fold=0.10, call=0.10, allin=0.30)
    parsed = {"current_player": 0, "money": [250, 1500, 1500, 1500, 1500, 1500],
              "contribution": [25, 50, 100, 0, 0, 0], "big_blind": 50,
              "public_cards": "", "private_cards": "7d2c", "street_idx": 0}
    f = make_live_policy_filter(short_stack_threshold_bb=6.0)
    out = f(p, _full_legal(), parsed, state=None)
    # Intermediate sizes masked
    for b in (A_BET_33, A_BET_50, A_BET_66, A_BET_100, A_BET_150, A_BET_200):
        assert out[b] == 0.0
    # FOLD kept (facing action, not AA/KK)
    assert out[A_FOLD] > 0
    assert abs(out.sum() - 1.0) < 1e-5


def test_composed_filter_aa_short_stack_stacked():
    """AA preflop AT 5 BB facing action — BOTH floors fire (AA → mask FOLD;
    short-stack → mask intermediate sizes)."""
    p = _mk_policy(fold=0.05, call=0.15, allin=0.30)
    parsed = {"current_player": 0, "money": [250, 1500, 1500, 1500, 1500, 1500],
              "contribution": [25, 50, 100, 0, 0, 0], "big_blind": 50,
              "public_cards": "", "private_cards": "AsAh", "street_idx": 0}
    f = make_live_policy_filter(short_stack_threshold_bb=6.0)
    out = f(p, _full_legal(), parsed, state=None)
    # FOLD masked by AA/KK floor
    assert out[A_FOLD] == 0.0
    # Intermediate sizes masked by short-stack floor
    for b in (A_BET_33, A_BET_50, A_BET_66, A_BET_100, A_BET_150, A_BET_200):
        assert out[b] == 0.0
    # Only CALL + ALLIN remain
    assert out[A_CALL] > 0
    assert out[A_ALLIN] > 0
    assert abs(out.sum() - 1.0) < 1e-5


def test_composed_filter_check_free_alone():
    """Deep stack, BB-after-limp (CHECK free, to_call=0) — check-free fires."""
    p = _mk_policy(fold=0.50, call=0.20, allin=0.05)
    # All contribs equal → to_call = 0
    parsed = {"current_player": 0, "money": [1500] * 6,
              "contribution": [50, 50, 50, 50, 50, 50], "big_blind": 50,
              "public_cards": "", "private_cards": "9h2d", "street_idx": 0}
    f = make_live_policy_filter(short_stack_threshold_bb=6.0)
    out = f(p, _full_legal(), parsed, state=None)
    assert out[A_FOLD] == 0.0
    # Intermediate sizes remain (deep)
    assert out[A_BET_100] > 0
    assert abs(out.sum() - 1.0) < 1e-5


def test_composed_filter_threshold_configurable():
    """Bump threshold to 10 BB — fires on 8 BB but not 12 BB."""
    p = _mk_policy(fold=0.10)
    # 8 BB at BB=50 = 400 chips
    parsed8 = {"current_player": 0, "money": [400, 1500, 1500, 1500, 1500, 1500],
               "contribution": [25, 50, 100, 0, 0, 0], "big_blind": 50,
               "public_cards": "", "private_cards": "7d2c", "street_idx": 0}
    # 12 BB at BB=50 = 600 chips
    parsed12 = dict(parsed8)
    parsed12["money"] = [600, 1500, 1500, 1500, 1500, 1500]

    f_thr10 = make_live_policy_filter(short_stack_threshold_bb=10.0)
    out8 = f_thr10(p, _full_legal(), parsed8, state=None)
    out12 = f_thr10(p, _full_legal(), parsed12, state=None)
    assert out8 is not p       # fired at 8 BB
    assert out12 is p          # didn't fire at 12 BB
