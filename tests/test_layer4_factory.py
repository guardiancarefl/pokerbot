"""Tests for src/nlhe/layer4_factory.py (Layer 4 / C1d-1).

The factory is a thin adapter — most behavior is verified at the C1a/b/c
layers. These tests pin the integration glue: zero-observation bit-identity,
per-seat isolation, live-read invariant, error paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.nlhe.archetypes import EquityCalibration
from src.nlhe.biased_policy import BiasConfig
from src.nlhe.layer4_factory import make_bias_factory
from src.nlhe.within_match import MatchObserver


# Hand-built parsed dicts mirror those used in test_within_match.py — the
# observer's facing_bet heuristic reads contribution[seat] vs max(contribution).
_PARSED_PRE_FACING = {"street_idx": 0, "contribution": [0, 100, 0, 0, 0, 0]}
_PARSED_FLOP_NULL = {"street_idx": 1, "contribution": [0] * 6}

# DiscreteAction integer aliases (kept inline; matches the test_within_match
# style and avoids importing the enum across test files).
_FOLD = 0
_CALL = 1
_BET_66 = 3


def _minimal_calibration() -> EquityCalibration:
    """Mirrors the helper in test_bias_configs.py — small monotone bucket
    equity ladder + matched quantile table. Used by archetype-path tests
    where the C1b builder needs a calibration at confidence > 0."""
    bucket_eq = {
        street: np.array([0.2, 0.5, 0.8], dtype=np.float32)
        for street in ("preflop", "flop", "turn", "river")
    }
    quantiles = {
        street: {"q05": 0.15, "q25": 0.30, "q50": 0.50,
                 "q75": 0.70, "q95": 0.85}
        for street in ("preflop", "flop", "turn", "river")
    }
    return EquityCalibration(bucket_equity=bucket_eq,
                             quantile_thresholds=quantiles)


def _stub_resolver(seat: int) -> dict:
    """A minimal leaf_context_resolver for archetype-path smoke tests.

    Returns the fields stats_to_bias_configs_archetype requires; at the
    confidence=0 tests below this content is unused (the short-circuit fires
    before any of these fields are inspected)."""
    return {
        "parsed": {
            "street_idx": 1,
            "current_player": seat,
            "contribution": [0] * 6,
            "money": [1000] * 6,
            "pot": 100,
            "legal_mask": np.ones(7, dtype=np.float32),
        },
        "state": None,
        "bucket_id": 0,
        "in_position": False,
        "calibration": _minimal_calibration(),
    }


# ---------- 1. Raw path: zero observations → all-ones ----------

def test_raw_factory_zero_obs_returns_ones():
    """At observer.confidence(seat)==0 the raw factory returns k=4 all-ones
    BiasConfigs — the bit-identity gate that lets the factory be a safe
    drop-in for None."""
    obs = MatchObserver()
    factory = make_bias_factory(obs, "raw")
    cfgs = factory(seat=1)
    assert len(cfgs) == 4
    for cfg in cfgs:
        assert isinstance(cfg, BiasConfig)
        assert cfg.multipliers.shape == (7,)
        assert np.allclose(cfg.multipliers, 1.0, atol=1e-12)


# ---------- 2. Raw path: live observations produce non-trivial bias ----------

def test_raw_factory_with_observations():
    """Feed seat 2 a long stream of aggressive postflop actions; confidence
    crosses past 0 and the stats are far from BLUEPRINT_REF. At least one
    BiasConfig should diverge from the all-ones identity."""
    obs = MatchObserver()
    # 200 flop BETs from seat 2 — saturates the confidence ramp (n>=300 → 1.0;
    # 200 gives 0.75) AND drives aggression_freq to 1.0 vs BLUEPRINT_REF 0.45.
    parsed = {"street_idx": 1, "contribution": [0] * 6}
    for _ in range(200):
        obs.update(None, parsed, action=_BET_66, seat=2)
    assert obs.confidence(2) > 0.5  # sanity: ramp engaged

    factory = make_bias_factory(obs, "raw")
    cfgs = factory(seat=2)
    assert len(cfgs) == 4
    arr = np.stack([c.multipliers for c in cfgs])
    # At least one k entry must differ from ones somewhere.
    assert not np.allclose(arr, 1.0, atol=1e-9), \
        "raw factory at non-zero confidence + extreme stats must produce non-trivial bias"


# ---------- 3. Per-seat isolation ----------

def test_raw_factory_per_seat_isolation():
    """Populate seat 1 only; factory(seat=1) yields non-trivial multipliers,
    factory(seat=2) yields the zero-observation ones-only output."""
    obs = MatchObserver()
    parsed = {"street_idx": 1, "contribution": [0] * 6}
    for _ in range(200):
        obs.update(None, parsed, action=_BET_66, seat=1)

    factory = make_bias_factory(obs, "raw")
    cfgs_1 = factory(seat=1)
    cfgs_2 = factory(seat=2)

    arr_1 = np.stack([c.multipliers for c in cfgs_1])
    arr_2 = np.stack([c.multipliers for c in cfgs_2])
    assert not np.allclose(arr_1, 1.0, atol=1e-9)  # seat 1 non-trivial
    assert np.allclose(arr_2, 1.0, atol=1e-12)     # seat 2 untouched


# ---------- 4. Archetype path needs the resolver ----------

def test_archetype_factory_requires_resolver():
    """Constructing the archetype path without leaf_context_resolver raises
    ValueError at factory build time (not at call time) — fast failure."""
    obs = MatchObserver()
    with pytest.raises(ValueError, match="leaf_context_resolver"):
        make_bias_factory(obs, "archetype")


# ---------- 5. Archetype path: zero observations → all-ones ----------

def test_archetype_factory_zero_obs_returns_ones():
    """At observer.confidence(seat)==0 the archetype path short-circuits to
    all-ones BiasConfigs — bucket_id and resolver content are not consulted."""
    obs = MatchObserver()
    factory = make_bias_factory(obs, "archetype",
                                leaf_context_resolver=_stub_resolver)
    cfgs = factory(seat=1)
    assert len(cfgs) == 4
    for cfg in cfgs:
        assert np.allclose(cfg.multipliers, 1.0, atol=1e-12)


# ---------- 6. Archetype path: resolver is invoked on each call ----------

def test_archetype_factory_resolver_is_called():
    """For each non-trivial confidence factory call, the resolver fires
    exactly once with the queried seat. At zero confidence the C1b
    short-circuit fires BEFORE the resolver is touched, so this test
    populates the observer enough to engage the ramp."""
    calls: list[int] = []

    def resolver(seat: int) -> dict:
        calls.append(seat)
        return _stub_resolver(seat)

    obs = MatchObserver()
    # Engage the ramp for seat 3 so the factory actually hits the resolver
    # (zero-confidence call would short-circuit upstream).
    parsed = {"street_idx": 1, "contribution": [0] * 6}
    for _ in range(150):
        obs.update(None, parsed, action=_BET_66, seat=3)
    assert obs.confidence(3) > 0.5

    factory = make_bias_factory(obs, "archetype",
                                leaf_context_resolver=resolver)
    factory(seat=3)
    assert calls == [3], f"resolver expected called once with seat=3, got {calls}"


# ---------- 7. Invalid path ----------

def test_invalid_path_raises():
    obs = MatchObserver()
    with pytest.raises(ValueError, match="path must be"):
        make_bias_factory(obs, "bogus")


# ---------- 8. Live read invariant ----------

def test_factory_reads_observer_live():
    """The factory is a closure over `observer`; it must read CURRENT state on
    each call. Snapshot the output, mutate the observer, call again — the
    second output reflects the new observations."""
    obs = MatchObserver()
    factory = make_bias_factory(obs, "raw")

    # First snapshot: zero observations → all-ones.
    snap_before = np.stack([c.multipliers for c in factory(seat=1)])
    assert np.allclose(snap_before, 1.0, atol=1e-12)

    # Feed 200 postflop bets to seat 1 — drives confidence > 0 AND stats far
    # from BLUEPRINT_REF.
    parsed = {"street_idx": 1, "contribution": [0] * 6}
    for _ in range(200):
        obs.update(None, parsed, action=_BET_66, seat=1)

    # Second snapshot: factory MUST reflect the new observations.
    snap_after = np.stack([c.multipliers for c in factory(seat=1)])
    assert not np.allclose(snap_after, snap_before, atol=1e-9), \
        "factory must read observer state live (not cache the construction-time snapshot)"
