"""DCFR weighting unit tests for DeepCFR6MaxSolver._dcfr_weights.

Tests target the helper in isolation by bypassing __init__: we only need
self.cfg and self.iteration set, no game / network / buffer needed.
Mirrors the HUNL DCFR smoke-gates pattern from Phase 3 Track A1.
"""
import pytest
import torch

from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max


def _stub(cfr_variant="vanilla", dcfr_exponent=1.0, iteration=10):
    """Build a minimal solver with only the attrs _dcfr_weights reads."""
    cfg = TrainConfig6Max(cfr_variant=cfr_variant, dcfr_exponent=dcfr_exponent)
    s = DeepCFR6MaxSolver.__new__(DeepCFR6MaxSolver)
    s.cfg = cfg
    s.iteration = iteration
    return s


def test_vanilla_returns_none():
    s = _stub(cfr_variant="vanilla")
    assert s._dcfr_weights(torch.tensor([1, 2, 3, 4])) is None


def test_linear_normalizes_to_batch_size():
    s = _stub(cfr_variant="linear", iteration=10)
    w = s._dcfr_weights(torch.tensor([1, 5, 10]))
    # Normalization invariant: sum(weights) == batch_size.
    assert torch.allclose(w.sum(), torch.tensor(3.0), atol=1e-5)
    # Monotonicity: newer iters weight higher than older iters.
    assert w[2] > w[1] > w[0]


def test_discounted_exponent_one_equals_linear():
    iters = torch.tensor([1, 5, 10])
    a = _stub(cfr_variant="linear", iteration=10)._dcfr_weights(iters)
    b = _stub(cfr_variant="discounted", dcfr_exponent=1.0, iteration=10)._dcfr_weights(iters)
    assert torch.allclose(a, b, atol=1e-6)


def test_discounted_exponent_two_more_aggressive_than_linear():
    iters = torch.tensor([1, 5, 10])
    w_lin = _stub(cfr_variant="linear", iteration=10)._dcfr_weights(iters)
    w_dis = _stub(cfr_variant="discounted", dcfr_exponent=2.0, iteration=10)._dcfr_weights(iters)
    # Recent-over-old ratio grows with exponent.
    assert (w_dis[2] / w_dis[0]) > (w_lin[2] / w_lin[0])


def test_iter_1_uniform_iters_collapse_to_ones():
    """When all sampled iters == T, ratio==1 everywhere so normalized w==1."""
    w = _stub(cfr_variant="linear", iteration=1)._dcfr_weights(torch.tensor([1, 1, 1, 1]))
    assert torch.allclose(w, torch.ones(4), atol=1e-6)


def test_unknown_variant_raises():
    s = _stub(cfr_variant="nonsense")
    with pytest.raises(ValueError, match="unknown cfr_variant"):
        s._dcfr_weights(torch.tensor([1, 2]))


def test_config_defaults_preserve_vanilla():
    """A YAML config that omits the new fields must yield vanilla behavior."""
    cfg = TrainConfig6Max()
    assert cfg.cfr_variant == "vanilla"
    assert cfg.dcfr_exponent == 1.0


def test_int_iters_handled_via_float_cast():
    """Buffer returns int64 iters tensor; helper must cast safely."""
    iters = torch.tensor([1, 5, 10], dtype=torch.int64)
    w = _stub(cfr_variant="linear", iteration=10)._dcfr_weights(iters)
    assert w.dtype == torch.float32 or w.dtype == torch.float64
    assert torch.allclose(w.sum(), torch.tensor(3.0), atol=1e-5)
