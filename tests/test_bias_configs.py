"""Tests for src/nlhe/bias_configs.py (Layer 4 / C1b).

Both raw-stats and archetype-Bayesian paths share the bit-identity-at-zero
contract (tests 3, 8, 13) and the alpha-clipping contract (tests 4, 10).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.nlhe.within_match import SeatStats
from src.nlhe.bias_configs import (
    ALPHA_C1_DEFAULT,
    BLUEPRINT_REF,
    compute_seat_ratios,
    _compute_archetype_posterior,
    stats_to_bias_configs_raw,
    stats_to_bias_configs_archetype,
)
from src.nlhe.archetypes import EquityCalibration, NAMED_ARCHETYPES, ArchetypeName


# ---------- helpers ----------

def _stats_with_ratios(
    *, vpip: float | None = None, pfr: float | None = None,
    agg_freq: float | None = None, fold_to_bet: float | None = None,
    avg_bet_over_pot: float | None = None,
    n_preflop: int = 200, n_postflop: int = 200, n_facing: int = 200,
    n_bet_samples: int = 50,
) -> SeatStats:
    """Construct a SeatStats with the requested ratios (within roundoff).

    Counters are sized so the ratios stay well above the confidence ramp's
    n_actions < 20 zero band (every test that touches confidence > 0 starts
    from a well-populated stats vector).
    """
    s = SeatStats(seat=0)
    if vpip is not None:
        s.n_preflop_decisions = n_preflop
        s.n_preflop_voluntary = int(round(vpip * n_preflop))
    if pfr is not None:
        if s.n_preflop_decisions == 0:
            s.n_preflop_decisions = n_preflop
        s.n_preflop_raises = int(round(pfr * s.n_preflop_decisions))
    if agg_freq is not None:
        # Distribute across flop slot (1) for simplicity.
        s.n_postflop_decisions = [0, n_postflop, 0, 0]
        s.n_postflop_aggressive = [0, int(round(agg_freq * n_postflop)), 0, 0]
    if fold_to_bet is not None:
        s.n_facing_bet = [0, n_facing, 0, 0]
        s.n_folds_facing_bet = [0, int(round(fold_to_bet * n_facing)), 0, 0]
    if avg_bet_over_pot is not None:
        s.n_bet_size_samples = n_bet_samples
        s.sum_bet_size_over_pot = float(avg_bet_over_pot) * n_bet_samples
    # n_actions used by confidence(); bump to ensure tests can drive
    # confidence to a known value externally — bias_configs doesn't read it.
    s.n_actions = max(s.n_preflop_decisions, sum(s.n_postflop_decisions), 1)
    return s


def _make_minimal_calibration() -> EquityCalibration:
    """Construct a minimal EquityCalibration for archetype-path tests.

    Three buckets per street with a simple monotone equity ladder + a
    matched quantile table that the archetypes can read.
    """
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


def _minimal_parsed_leaf(*, legal_mask: np.ndarray | None = None) -> dict:
    """A minimal parsed dict suitable for the archetype path's leaf context."""
    if legal_mask is None:
        legal_mask = np.ones(7, dtype=np.float32)
    return {
        "street_idx": 1,           # flop
        "current_player": 0,
        "contribution": [0, 0, 0, 0, 0, 100],  # seat 0 faces a bet
        "money": [1000] * 6,
        "pot": 200,
        "legal_mask": legal_mask,
    }


# ---------- ratios ----------

def test_compute_seat_ratios_empty():
    s = SeatStats(seat=0)
    r = compute_seat_ratios(s)
    assert r == {"vpip": None, "pfr": None, "aggression_freq": None,
                 "fold_to_bet": None, "avg_bet_over_pot": None}


def test_compute_seat_ratios_populated():
    s = SeatStats(
        seat=0,
        n_preflop_decisions=10,
        n_preflop_voluntary=3,
        n_preflop_raises=2,
        n_postflop_decisions=[0, 10, 5, 2],
        n_postflop_aggressive=[0, 4, 2, 1],
        n_facing_bet=[5, 4, 2, 1],
        n_folds_facing_bet=[2, 2, 1, 1],
        sum_bet_size_over_pot=3.0,
        n_bet_size_samples=4,
    )
    r = compute_seat_ratios(s)
    assert r["vpip"] == pytest.approx(3 / 10)
    assert r["pfr"] == pytest.approx(2 / 10)
    assert r["aggression_freq"] == pytest.approx((4 + 2 + 1) / (10 + 5 + 2))
    assert r["fold_to_bet"] == pytest.approx((2 + 2 + 1 + 1) / (5 + 4 + 2 + 1))
    assert r["avg_bet_over_pot"] == pytest.approx(3.0 / 4)


# ---------- raw path ----------

def test_raw_path_confidence_zero():
    """Bit-identity gate: at confidence=0 all k entries == ones, regardless of
    how extreme the underlying observations are."""
    s = _stats_with_ratios(vpip=0.80, agg_freq=0.85)  # extreme deviation
    configs = stats_to_bias_configs_raw(s, confidence=0.0)
    assert len(configs) == 4
    for cfg in configs:
        assert cfg.multipliers.shape == (7,)
        assert np.allclose(cfg.multipliers, 1.0, atol=1e-12)


def test_raw_path_clipped_to_alpha():
    """All multipliers must lie in [1/alpha, alpha] regardless of deviation."""
    s = _stats_with_ratios(
        vpip=0.95, pfr=0.85, agg_freq=0.95,
        fold_to_bet=0.05, avg_bet_over_pot=1.50,
    )
    alpha = ALPHA_C1_DEFAULT  # 2.0
    configs = stats_to_bias_configs_raw(s, confidence=1.0, alpha=alpha)
    lo, hi = 1.0 / alpha, alpha
    for cfg in configs:
        assert cfg.multipliers.min() >= lo - 1e-12
        assert cfg.multipliers.max() <= hi + 1e-12


def test_raw_path_high_aggression_increases_fold_multiplier():
    """High observed opponent aggression_freq → hero defends: FOLD↑, CALL↓.

    Test 5 lock — the W matrix is configured hero-defensive on aggression_freq.
    """
    # aggression_freq=0.75 against BLUEPRINT_REF mean=0.45, sd=0.15 → z=+2.
    # Other stats omitted (None) so only the aggression_freq row activates.
    s = _stats_with_ratios(agg_freq=0.75)
    configs = stats_to_bias_configs_raw(s, confidence=1.0)
    k0 = configs[0].multipliers
    # FOLD index 0, CALL index 1.
    assert k0[0] > 1.0, f"FOLD multiplier expected > 1, got {k0[0]}"
    assert k0[1] < 1.0, f"CALL multiplier expected < 1, got {k0[1]}"


def test_raw_path_k_entries_distinct():
    """At confidence=1 with a non-zero, non-saturating deviation the four k
    entries differ from each other (the menu has graded scaling)."""
    # Mild deviation: aggression_freq=0.55 (z ≈ +0.67 against mean=0.45).
    s = _stats_with_ratios(agg_freq=0.55)
    configs = stats_to_bias_configs_raw(s, confidence=1.0)
    arr = np.stack([c.multipliers for c in configs])  # (4, 7)
    # No two rows should be exactly equal.
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            assert not np.allclose(arr[i], arr[j], atol=1e-9), \
                f"k{i} and k{j} must differ; got identical rows"


def test_raw_path_names():
    s = _stats_with_ratios(vpip=0.40)
    configs = stats_to_bias_configs_raw(s, confidence=0.5)
    names = [c.name for c in configs]
    assert names == ["c1_raw_k0", "c1_raw_k1", "c1_raw_k2", "c1_raw_k3"]


# ---------- archetype path ----------

def test_archetype_path_confidence_zero():
    """Bit-identity gate (archetype path): confidence=0 → all-ones, no
    bucket_id / calibration required."""
    s = _stats_with_ratios(vpip=0.10, agg_freq=0.20, fold_to_bet=0.75)
    configs = stats_to_bias_configs_archetype(
        s, confidence=0.0,
        parsed=_minimal_parsed_leaf(),
        state=None,
        bucket_id=None,  # bypass-allowed at confidence=0
        in_position=False,
    )
    assert len(configs) == 4
    for cfg in configs:
        assert cfg.multipliers.shape == (7,)
        assert np.allclose(cfg.multipliers, 1.0, atol=1e-12)


def test_archetype_path_posterior_concentrates_on_nit():
    """A NIT-like stats profile yields posterior weight concentrated on NIT.

    NIT_REF (vpip=0.15, pfr=0.04, agg_freq=0.25, fold_to_bet=0.68).
    Pick observed ratios within ~0.1 sigma of those, far from the other four.
    """
    s = _stats_with_ratios(
        vpip=0.15, pfr=0.04, agg_freq=0.25, fold_to_bet=0.68,
    )
    posterior = _compute_archetype_posterior(s)
    assert posterior.shape == (len(NAMED_ARCHETYPES),)
    # NAMED_ARCHETYPES order: NIT, TAG, LAG, STATION, MANIAC.
    nit_idx = [i for i, a in enumerate(NAMED_ARCHETYPES)
               if a.name == ArchetypeName.NIT][0]
    assert posterior[nit_idx] == max(posterior), \
        f"NIT should be argmax of posterior; got {posterior.tolist()}"
    # Domination: NIT should hold the majority of mass.
    assert posterior[nit_idx] > 0.5


def test_archetype_path_clipped_to_alpha():
    """All multipliers stay in [1/alpha, alpha] under any observation."""
    s = _stats_with_ratios(vpip=0.95, agg_freq=0.95, fold_to_bet=0.05)
    alpha = ALPHA_C1_DEFAULT
    cal = _make_minimal_calibration()
    configs = stats_to_bias_configs_archetype(
        s, confidence=1.0,
        parsed=_minimal_parsed_leaf(),
        state=None,
        bucket_id=1,
        in_position=True,
        calibration=cal,
        alpha=alpha,
    )
    lo, hi = 1.0 / alpha, alpha
    for cfg in configs:
        assert cfg.multipliers.shape == (7,)
        assert cfg.multipliers.min() >= lo - 1e-12
        assert cfg.multipliers.max() <= hi + 1e-12


def test_archetype_path_requires_bucket_id():
    """At confidence > 0, bucket_id=None must raise ValueError."""
    s = _stats_with_ratios(vpip=0.50)
    cal = _make_minimal_calibration()
    with pytest.raises(ValueError, match="bucket_id"):
        stats_to_bias_configs_archetype(
            s, confidence=0.5,
            parsed=_minimal_parsed_leaf(),
            state=None,
            bucket_id=None,
            in_position=False,
            calibration=cal,
        )


# ---------- cross-path invariants ----------

def test_both_paths_same_output_signature():
    """Both functions return list[BiasConfig] of length k=4, multipliers
    shape (7,)."""
    s = _stats_with_ratios(vpip=0.40, agg_freq=0.50)
    cal = _make_minimal_calibration()

    raw_cfgs = stats_to_bias_configs_raw(s, confidence=0.5)
    arch_cfgs = stats_to_bias_configs_archetype(
        s, confidence=0.5,
        parsed=_minimal_parsed_leaf(),
        state=None,
        bucket_id=1,
        in_position=False,
        calibration=cal,
    )
    assert len(raw_cfgs) == 4
    assert len(arch_cfgs) == 4
    for cfg in raw_cfgs + arch_cfgs:
        assert cfg.multipliers.shape == (7,)
        assert cfg.multipliers.dtype == np.float64
        assert (cfg.multipliers > 0).all()


def test_both_paths_blueprint_recovery():
    """Bit-identity gate: at confidence=0 both paths produce identical
    all-ones multipliers across all k entries. This is the C1c integration
    invariant — disabling C1 must reproduce SUBSTEP_6_DESIGN's recorded BR/
    PROFILE/blueprint verdict bit-for-bit.
    """
    s = _stats_with_ratios(vpip=0.95, agg_freq=0.95)  # extreme; ignored at conf=0
    raw_cfgs = stats_to_bias_configs_raw(s, confidence=0.0)
    arch_cfgs = stats_to_bias_configs_archetype(
        s, confidence=0.0,
        parsed=_minimal_parsed_leaf(),
        state=None,
        bucket_id=None,
        in_position=False,
    )
    assert len(raw_cfgs) == len(arch_cfgs) == 4
    for cfg in raw_cfgs + arch_cfgs:
        assert np.allclose(cfg.multipliers, 1.0, atol=1e-12)
