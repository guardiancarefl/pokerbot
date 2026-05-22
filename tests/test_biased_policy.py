"""Tests for src/nlhe/biased_policy.py (Track B1a: leaf strategies)."""
from __future__ import annotations

import numpy as np
import pytest

from src.nlhe.actions import DiscreteAction
from src.nlhe.biased_policy import (
    BiasConfig,
    BiasedBlueprint,
    apply_bias,
    standard_bias_configs,
)


# ---- BiasConfig validation ----

def test_bias_config_rejects_wrong_shape():
    with pytest.raises(ValueError):
        BiasConfig(name="bad", multipliers=np.ones(3))  # wrong size


def test_bias_config_rejects_nonpositive():
    bad = np.ones(len(DiscreteAction))
    bad[0] = 0.0
    with pytest.raises(ValueError):
        BiasConfig(name="zero", multipliers=bad)
    bad[0] = -1.0
    with pytest.raises(ValueError):
        BiasConfig(name="neg", multipliers=bad)


def test_bias_config_accepts_valid():
    BiasConfig(name="ok", multipliers=np.ones(len(DiscreteAction)) * 2.5)


# ---- standard_bias_configs shape ----

def test_standard_configs_returns_four():
    cfgs = standard_bias_configs()
    assert len(cfgs) == 4
    names = [c.name for c in cfgs]
    assert names[0] == "blueprint"
    assert "fold-biased" in names
    assert "call-biased" in names
    assert "raise-biased" in names


def test_blueprint_config_is_identity():
    cfgs = standard_bias_configs()
    bp = next(c for c in cfgs if c.name == "blueprint")
    assert np.allclose(bp.multipliers, 1.0)


def test_fold_biased_boosts_fold_only():
    cfgs = standard_bias_configs(alpha=3.0)
    fb = next(c for c in cfgs if c.name == "fold-biased")
    assert fb.multipliers[DiscreteAction.FOLD] == 3.0
    assert fb.multipliers[DiscreteAction.CALL] == 1.0
    for bet in (DiscreteAction.BET_33, DiscreteAction.BET_66,
                DiscreteAction.BET_100, DiscreteAction.BET_200, DiscreteAction.ALLIN):
        assert fb.multipliers[bet] == pytest.approx(1.0 / 3.0)


def test_raise_biased_reduces_fold_and_call():
    cfgs = standard_bias_configs(alpha=3.0)
    rb = next(c for c in cfgs if c.name == "raise-biased")
    assert rb.multipliers[DiscreteAction.FOLD] == pytest.approx(1.0 / 3.0)
    assert rb.multipliers[DiscreteAction.CALL] == pytest.approx(1.0 / 3.0)
    for bet in (DiscreteAction.BET_33, DiscreteAction.BET_66,
                DiscreteAction.BET_100, DiscreteAction.BET_200, DiscreteAction.ALLIN):
        assert rb.multipliers[bet] == 1.0


# ---- apply_bias correctness ----

def _uniform_legal(legal_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Build a (probs, legal_mask) pair where probs is uniform over legal_indices."""
    n = len(DiscreteAction)
    mask = np.zeros(n, dtype=np.float32)
    mask[legal_indices] = 1.0
    probs = mask / mask.sum()
    return probs, mask


def test_apply_bias_blueprint_is_identity():
    probs, mask = _uniform_legal([0, 1, 2, 3, 4])
    cfgs = standard_bias_configs()
    bp = next(c for c in cfgs if c.name == "blueprint")
    out = apply_bias(probs, mask, bp)
    assert np.allclose(out, probs)


def test_apply_bias_renormalizes_to_one():
    probs, mask = _uniform_legal([0, 1, 2, 3, 4])
    cfgs = standard_bias_configs(alpha=3.0)
    for c in cfgs:
        out = apply_bias(probs, mask, c)
        assert out.sum() == pytest.approx(1.0)


def test_apply_bias_zero_on_illegal():
    probs, mask = _uniform_legal([0, 1, 2])  # only fold/call/bet33 legal
    cfgs = standard_bias_configs(alpha=3.0)
    for c in cfgs:
        out = apply_bias(probs, mask, c)
        # illegal indices should be exactly zero
        for illegal in (3, 4, 5, 6):
            assert out[illegal] == 0.0


def test_apply_bias_fold_biased_increases_fold_share():
    probs, mask = _uniform_legal([0, 1, 2, 3, 4])
    cfgs = standard_bias_configs(alpha=3.0)
    fb = next(c for c in cfgs if c.name == "fold-biased")
    out = apply_bias(probs, mask, fb)
    # Fold should now be larger than its blueprint share of 0.2
    assert out[DiscreteAction.FOLD] > 0.2
    # And larger than any single bet share
    for bet in (DiscreteAction.BET_33, DiscreteAction.BET_66, DiscreteAction.BET_100):
        assert out[DiscreteAction.FOLD] > out[bet]


def test_apply_bias_raise_biased_increases_total_bet_share():
    probs, mask = _uniform_legal([0, 1, 2, 3, 4])  # 5 legal; 3 are bets
    cfgs = standard_bias_configs(alpha=3.0)
    rb = next(c for c in cfgs if c.name == "raise-biased")
    out = apply_bias(probs, mask, rb)
    bet_share = out[2] + out[3] + out[4]
    blueprint_bet_share = 0.2 * 3  # 0.6
    assert bet_share > blueprint_bet_share


def test_apply_bias_falls_back_when_bias_zeroes_all_mass():
    # Blueprint has all mass on bets; fold-bias zeros them but fold isn't legal.
    n = len(DiscreteAction)
    mask = np.zeros(n, dtype=np.float32)
    # Only bets legal (no fold/call)
    mask[2] = 1.0
    mask[3] = 1.0
    probs = np.zeros(n, dtype=np.float32)
    probs[2] = 0.5
    probs[3] = 0.5
    # An artificial bias that zeros out the only legal actions
    bad_bias = BiasConfig(
        name="bad",
        multipliers=np.array([1.0, 1.0, 1e-30, 1e-30, 1.0, 1.0, 1.0]),
    )
    out = apply_bias(probs, mask, bad_bias)
    # Should fall back to uniform-over-legal (0.5 each on indices 2, 3)
    assert out[2] == pytest.approx(0.5)
    assert out[3] == pytest.approx(0.5)
    assert out.sum() == pytest.approx(1.0)


# ---- BiasedBlueprint ----

def test_biased_blueprint_has_four_strategies():
    bb = BiasedBlueprint()
    assert bb.k == 4


def test_biased_blueprint_rejects_invalid_strategy_idx():
    bb = BiasedBlueprint()
    probs, mask = _uniform_legal([0, 1])
    with pytest.raises(ValueError):
        bb.action_probs(probs, mask, strategy_idx=-1)
    with pytest.raises(ValueError):
        bb.action_probs(probs, mask, strategy_idx=4)


def test_biased_blueprint_strategy_0_is_blueprint():
    bb = BiasedBlueprint()
    probs, mask = _uniform_legal([0, 1, 2, 3])
    out = bb.action_probs(probs, mask, strategy_idx=0)
    assert np.allclose(out, probs)


def test_biased_blueprint_strategies_differ():
    bb = BiasedBlueprint()
    probs, mask = _uniform_legal([0, 1, 2, 3, 4])
    outs = [bb.action_probs(probs, mask, i) for i in range(bb.k)]
    # All four should differ from at least one other
    for i in range(bb.k):
        for j in range(i + 1, bb.k):
            assert not np.allclose(outs[i], outs[j]),                 f"strategies {i} and {j} ({bb.strategy_name(i)}, {bb.strategy_name(j)}) are identical"
