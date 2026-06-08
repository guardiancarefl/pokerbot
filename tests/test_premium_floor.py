"""Unit tests for the AA/KK preflop FOLD floor.

Verifies the floor's strict scope and correctness:
  - Fires ONLY on (preflop AND AA-or-KK)
  - Renormalizes the remaining mass so the policy stays a valid distribution
  - Returns the input policy reference unchanged when the gate doesn't fire
    (so callers can rely on identity to short-circuit)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.nlhe.actions import DiscreteAction
from src.nlhe.integration.live_loop import apply_aa_kk_preflop_floor


N_ACT = len(DiscreteAction)


def _mk_policy(fold=0.05, call=0.10, allin=0.30):
    """Build a length-N policy with FOLD=`fold`, CALL=`call`, ALLIN=`allin`,
    and the rest spread evenly over the bet-actions. Always sums to 1."""
    p = np.zeros(N_ACT, dtype=np.float32)
    p[int(DiscreteAction.FOLD)] = fold
    p[int(DiscreteAction.CALL)] = call
    p[int(DiscreteAction.ALLIN)] = allin
    bet_slots = [DiscreteAction.BET_33, DiscreteAction.BET_50,
                 DiscreteAction.BET_66, DiscreteAction.BET_100,
                 DiscreteAction.BET_150, DiscreteAction.BET_200]
    raise_mass = 1.0 - fold - call - allin
    assert raise_mass >= 0.0, f"invalid policy: fold+call+allin={fold+call+allin}"
    per = raise_mass / len(bet_slots)
    for da in bet_slots:
        p[int(da)] = per
    assert abs(p.sum() - 1.0) < 1e-5
    return p


def _full_legal_mask():
    return np.ones(N_ACT, dtype=np.float32)


# --------------------------------------------------------------------------
# (1) AA preflop → FOLD masked, mass renormalized
# --------------------------------------------------------------------------

def test_aa_preflop_masks_fold_and_renormalizes():
    p = _mk_policy(fold=0.05)
    mask = _full_legal_mask()
    parsed = {"private_cards": "AsAh", "public_cards": ""}

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    assert out[int(DiscreteAction.FOLD)] == 0.0
    # Total mass still 1
    assert abs(out.sum() - 1.0) < 1e-5
    # Other actions got scaled UP by 1/(1 - 0.05) = ~1.0526
    expected_call = 0.10 / 0.95
    assert abs(out[int(DiscreteAction.CALL)] - expected_call) < 1e-5
    expected_allin = 0.30 / 0.95
    assert abs(out[int(DiscreteAction.ALLIN)] - expected_allin) < 1e-5


# --------------------------------------------------------------------------
# (2) KK preflop → same treatment
# --------------------------------------------------------------------------

def test_kk_preflop_masks_fold():
    p = _mk_policy(fold=0.08)
    mask = _full_legal_mask()
    parsed = {"private_cards": "KsKh", "public_cards": ""}

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    assert out[int(DiscreteAction.FOLD)] == 0.0
    assert abs(out.sum() - 1.0) < 1e-5


# --------------------------------------------------------------------------
# (3) AA preflop facing 3-bet (still preflop, still AA) → masked
# --------------------------------------------------------------------------

def test_aa_preflop_facing_3bet_still_masked():
    # public_cards still empty preflop regardless of betting depth
    p = _mk_policy(fold=0.12)
    mask = _full_legal_mask()
    parsed = {"private_cards": "AhAd", "public_cards": ""}

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    assert out[int(DiscreteAction.FOLD)] == 0.0


# --------------------------------------------------------------------------
# (4) QQ preflop → UNCHANGED (QQ-folds are sometimes correct)
# --------------------------------------------------------------------------

def test_qq_preflop_unchanged():
    p = _mk_policy(fold=0.17)
    mask = _full_legal_mask()
    parsed = {"private_cards": "QsQh", "public_cards": ""}

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    # Identity short-circuit: returns the same reference
    assert out is p
    assert out[int(DiscreteAction.FOLD)] == 0.17


# --------------------------------------------------------------------------
# (5) AA on a flop → UNCHANGED (postflop AA can be a fold on wet textures)
# --------------------------------------------------------------------------

def test_aa_postflop_unchanged():
    p = _mk_policy(fold=0.10)
    mask = _full_legal_mask()
    parsed = {"private_cards": "AsAh", "public_cards": "QsJsTs"}  # monotone

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    assert out is p
    assert out[int(DiscreteAction.FOLD)] == 0.10


def test_aa_turn_unchanged():
    p = _mk_policy(fold=0.10)
    mask = _full_legal_mask()
    parsed = {"private_cards": "AsAh", "public_cards": "Qs7c2dKd"}

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    assert out is p


def test_aa_river_unchanged():
    p = _mk_policy(fold=0.10)
    mask = _full_legal_mask()
    parsed = {"private_cards": "AsAh", "public_cards": "Qs7c2dKd9s"}

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    assert out is p


# --------------------------------------------------------------------------
# (6) AKs / AKo preflop → UNCHANGED (mixed pre-fold can be correct)
# --------------------------------------------------------------------------

def test_aks_preflop_unchanged():
    p = _mk_policy(fold=0.20)
    mask = _full_legal_mask()
    parsed = {"private_cards": "AsKs", "public_cards": ""}

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    assert out is p


def test_ako_preflop_unchanged():
    p = _mk_policy(fold=0.25)
    mask = _full_legal_mask()
    parsed = {"private_cards": "AsKh", "public_cards": ""}

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    assert out is p


# --------------------------------------------------------------------------
# (7) FOLD already illegal → no-op
# --------------------------------------------------------------------------

def test_aa_preflop_fold_already_illegal_noop():
    p = _mk_policy(fold=0.0)
    mask = _full_legal_mask()
    mask[int(DiscreteAction.FOLD)] = 0.0
    parsed = {"private_cards": "AsAh", "public_cards": ""}

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    # FOLD was already 0 → identity short-circuit
    assert out is p


# --------------------------------------------------------------------------
# (8) Pure-FOLD policy (degenerate, defensive) → uniform on legal non-FOLD
# --------------------------------------------------------------------------

def test_aa_preflop_pure_fold_policy_defensive():
    p = np.zeros(N_ACT, dtype=np.float32)
    p[int(DiscreteAction.FOLD)] = 1.0
    mask = np.zeros(N_ACT, dtype=np.float32)
    mask[int(DiscreteAction.FOLD)] = 1.0
    mask[int(DiscreteAction.CALL)] = 1.0
    mask[int(DiscreteAction.ALLIN)] = 1.0
    parsed = {"private_cards": "AsAh", "public_cards": ""}

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    assert out[int(DiscreteAction.FOLD)] == 0.0
    # Uniform over the 2 remaining legal non-FOLD actions
    assert abs(out[int(DiscreteAction.CALL)] - 0.5) < 1e-5
    assert abs(out[int(DiscreteAction.ALLIN)] - 0.5) < 1e-5


# --------------------------------------------------------------------------
# (9) AA suited (AsAs is impossible) — just confirms gate uses rank, not suit
# --------------------------------------------------------------------------

def test_aa_all_suit_combos_masked():
    """AA across all 6 unordered suit pairs all trigger the mask."""
    for cards in ("AsAh", "AsAd", "AsAc", "AhAd", "AhAc", "AdAc"):
        p = _mk_policy(fold=0.05)
        mask = _full_legal_mask()
        parsed = {"private_cards": cards, "public_cards": ""}
        out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)
        assert out[int(DiscreteAction.FOLD)] == 0.0, (
            f"{cards} preflop should mask FOLD, got "
            f"{out[int(DiscreteAction.FOLD)]}"
        )


# --------------------------------------------------------------------------
# (10) Pair of twos preflop → UNCHANGED (gate is rank-A or rank-K only)
# --------------------------------------------------------------------------

def test_22_preflop_unchanged():
    p = _mk_policy(fold=0.40)
    mask = _full_legal_mask()
    parsed = {"private_cards": "2s2h", "public_cards": ""}

    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)

    assert out is p


# --------------------------------------------------------------------------
# (11) Bad / missing card data → no-op (defensive)
# --------------------------------------------------------------------------

def test_missing_private_cards_noop():
    p = _mk_policy()
    mask = _full_legal_mask()
    parsed = {"private_cards": "", "public_cards": ""}
    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)
    assert out is p


def test_one_card_only_noop():
    p = _mk_policy()
    mask = _full_legal_mask()
    parsed = {"private_cards": "As", "public_cards": ""}
    out = apply_aa_kk_preflop_floor(p, mask, parsed, state=None)
    assert out is p
