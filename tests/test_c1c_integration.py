"""Tests for C1c — SubgamePolicy bias_factory wiring (Layer 4).

The C1c bit-identity invariant is the central contract: `bias_factory=None`
must preserve pre-C1c behavior byte-for-byte. These tests pin that contract
plus the per-seat dispatch semantics needed for C1d wiring.

Most tests exercise `build_per_seat_biased_blueprints` directly — it is the
integration logic, and unit-testing it sidesteps the heavy torch.load that
`SubgamePolicy.__init__` would otherwise require.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.nlhe.biased_policy import BiasConfig, BiasedBlueprint, standard_bias_configs
from src.nlhe.subgame_policy import build_per_seat_biased_blueprints
from src.nlhe.within_match import SeatStats
from src.nlhe.bias_configs import (
    stats_to_bias_configs_raw,
    stats_to_bias_configs_archetype,
)


# Default menu captured from Recon 1.4: BiasedBlueprint() == standard_bias_configs(3.0).
# These literal values lock the bit-identity gate; any future change to either
# BiasedBlueprint defaults or standard_bias_configs(3.0) must be intentional and
# break this test loudly.
_DEFAULT_NAMES = ("blueprint", "fold-biased", "call-biased", "raise-biased")
_DEFAULT_MULTS = (
    np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    np.array([3.0, 1.0, 1/3, 1/3, 1/3, 1/3, 1/3]),
    np.array([1.0, 3.0, 1/3, 1/3, 1/3, 1/3, 1/3]),
    np.array([1/3, 1/3, 1.0, 1.0, 1.0, 1.0, 1.0]),
)


def _identity_factory(seat: int) -> list[BiasConfig]:
    """Factory that returns 4 all-ones BiasConfigs (the C1b confidence=0 shape)."""
    return [BiasConfig(name=f"id_k{i}", multipliers=np.ones(7, dtype=np.float64))
            for i in range(4)]


def _distinct_factory(seat: int) -> list[BiasConfig]:
    """Factory whose k=0 entry has FOLD multiplier = 2.0 at seat 1, FOLD=1.5 at
    seat 2, FOLD=1.25 at others — distinctive enough for cross-seat isolation
    and reach-leaf-eval checks."""
    fold_mult = {1: 2.0, 2: 1.5}.get(seat, 1.25)
    base = np.ones(7, dtype=np.float64)
    base[0] = fold_mult  # FOLD index
    # k=0 carries the distinctive multiplier; k=1..3 are near-identity perturbations
    # so the all-identity short-circuit doesn't fire.
    configs = [BiasConfig(name=f"dist_k0_seat{seat}", multipliers=base.copy())]
    for i in range(1, 4):
        mults = np.ones(7, dtype=np.float64)
        mults[0] = 1.0 + 0.01 * i  # tiny perturbation, avoids short-circuit
        configs.append(BiasConfig(name=f"dist_k{i}_seat{seat}", multipliers=mults))
    return configs


# ---------- 1. Bit-identity invariant (bias_factory=None) ----------

def test_bias_factory_none_preserves_default_menu():
    """Locked invariant: the default BiasedBlueprint() that SubgamePolicy
    constructs at __init__ has the exact k=4 menu captured in Recon 1.4
    (standard_bias_configs(alpha=3.0)). C1c must not perturb this; any
    deviation here means a SUBSTEP_6_DESIGN ablation would no longer
    reproduce byte-for-byte."""
    bb = BiasedBlueprint()
    assert len(bb.bias_configs) == 4
    for i, cfg in enumerate(bb.bias_configs):
        assert cfg.name == _DEFAULT_NAMES[i], \
            f"k{i} name expected {_DEFAULT_NAMES[i]!r}, got {cfg.name!r}"
        assert np.array_equal(cfg.multipliers, _DEFAULT_MULTS[i]), \
            f"k{i} multipliers diverged from Recon 1.4 capture"

    # And: when bias_factory is None, the per-seat builder returns None — the
    # consumer (_bias_dist_fn) then uses the singleton biased_blueprint, which
    # IS this default menu. Byte-identical to pre-C1c.
    assert build_per_seat_biased_blueprints(None, hero_seat=0) is None


# ---------- 2. Factory invocation pattern ----------

def test_bias_factory_called_per_seat():
    """When hero=0 and the factory is non-identity, the builder calls it once
    for each of seats {1,2,3,4,5} and never for seat 0."""
    calls: list[int] = []

    def factory(seat: int) -> list[BiasConfig]:
        calls.append(seat)
        return _distinct_factory(seat)

    result = build_per_seat_biased_blueprints(factory, hero_seat=0)
    assert result is not None
    assert sorted(calls) == [1, 2, 3, 4, 5]
    assert 0 not in calls, "factory must not be called for hero seat"
    assert set(result.keys()) == {1, 2, 3, 4, 5}


def test_bias_factory_invoked_once_per_select_action():
    """The builder is per-call (no caching across calls). Within one call,
    exactly num_opp_seats = 5 invocations — not per-leaf."""
    call_count = {"n": 0}

    def factory(seat: int) -> list[BiasConfig]:
        call_count["n"] += 1
        return _distinct_factory(seat)

    # Single build call (one select_action's worth of factory work).
    build_per_seat_biased_blueprints(factory, hero_seat=0)
    assert call_count["n"] == 5  # 6 seats - 1 hero

    # A subsequent build call rebuilds — no caching.
    build_per_seat_biased_blueprints(factory, hero_seat=0)
    assert call_count["n"] == 10


# ---------- 3. Factory output reaches the leaf-eval surface ----------

def test_bias_factory_output_reaches_leaf_eval():
    """A factory with seat 1 → FOLD multiplier 2.0 must surface in the returned
    dict's BiasedBlueprint for seat 1. This is the data path into
    LeafEvalContext.biased_blueprint_per_seat that _bias_dist_fn consumes."""
    result = build_per_seat_biased_blueprints(_distinct_factory, hero_seat=0)
    assert result is not None
    assert 1 in result
    bb1 = result[1]
    assert isinstance(bb1, BiasedBlueprint)
    assert bb1.bias_configs[0].multipliers[0] == 2.0  # FOLD index, seat 1's k=0


# ---------- 4. C1b raw path at confidence=0 → identity short-circuit ----------

def test_bias_factory_zero_confidence_recovers_blueprint():
    """Using the REAL C1b raw builder at confidence=0:

      1. The factory output is all-ones for every seat (locks C1b's invariant
         at the integration boundary).
      2. The per-seat builder detects identity and returns None.
      3. None ⇒ LeafEvalContext.biased_blueprint_per_seat stays None ⇒
         _bias_dist_fn takes the byte-identical pre-C1c code path.

    Together: factory at C1==0 is observationally indistinguishable from
    bias_factory=None. This is the second load-bearing gate.
    """
    def factory(seat: int) -> list[BiasConfig]:
        return stats_to_bias_configs_raw(SeatStats(seat=seat), confidence=0.0)

    # Gate (1): factory output is all-ones for every seat.
    for s in range(6):
        cfgs = factory(s)
        assert len(cfgs) == 4
        for cfg in cfgs:
            assert np.allclose(cfg.multipliers, 1.0, atol=1e-12), \
                f"seat {s} expected all-ones, got {cfg.multipliers}"

    # Gates (2)+(3): builder short-circuits to None.
    result = build_per_seat_biased_blueprints(factory, hero_seat=0)
    assert result is None, \
        "C1b raw at confidence=0 must short-circuit to None (bit-identity gate)"


# ---------- 5. C1b archetype path at confidence=0 → identity short-circuit ----------

def test_archetype_factory_integration():
    """Same bit-identity gate via the C1b archetype path. At confidence=0 the
    archetype path returns all-ones without requiring bucket_id / calibration —
    so the integration boundary is symmetric across the two C1b paths."""
    minimal_parsed = {
        "street_idx": 1,
        "current_player": 0,
        "contribution": [0] * 6,
        "money": [1000] * 6,
        "pot": 100,
        "legal_mask": np.ones(7, dtype=np.float32),
    }

    def factory(seat: int) -> list[BiasConfig]:
        return stats_to_bias_configs_archetype(
            SeatStats(seat=seat), confidence=0.0,
            parsed=minimal_parsed, state=None,
            bucket_id=None, in_position=False,  # allowed at confidence=0
        )

    # Factory output is all-ones for every seat.
    for s in range(6):
        cfgs = factory(s)
        assert len(cfgs) == 4
        for cfg in cfgs:
            assert np.allclose(cfg.multipliers, 1.0, atol=1e-12)

    # Builder short-circuits to None.
    result = build_per_seat_biased_blueprints(factory, hero_seat=0)
    assert result is None


# ---------- 6. Per-seat isolation ----------

def test_per_seat_dispatch_isolation():
    """Different seats receive different multipliers; no leakage between them.
    Seat 1 gets FOLD=2.0, seat 2 gets FOLD=1.5, others get FOLD=1.25 — and the
    returned BiasedBlueprints reflect this distinction at the right slot.
    """
    result = build_per_seat_biased_blueprints(_distinct_factory, hero_seat=0)
    assert result is not None

    # Seat 1 distinct from seat 2 distinct from seat 3.
    assert result[1].bias_configs[0].multipliers[0] == 2.0
    assert result[2].bias_configs[0].multipliers[0] == 1.5
    assert result[3].bias_configs[0].multipliers[0] == 1.25
    assert result[4].bias_configs[0].multipliers[0] == 1.25
    assert result[5].bias_configs[0].multipliers[0] == 1.25

    # Mutating one seat's multipliers must not affect another's (the factory
    # builds fresh arrays per seat; this test catches any accidental aliasing).
    result[1].bias_configs[0].multipliers[0] = 99.0
    assert result[2].bias_configs[0].multipliers[0] == 1.5
    assert result[3].bias_configs[0].multipliers[0] == 1.25
