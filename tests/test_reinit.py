"""Tests for periodic advantage-network reinitialization (Brown 2019 sec 4.3).

Validates the reinit machinery added to:
  - PlayerNetworks6Max.reinit_seat() — fresh weights, preserved buffer
  - DeepCFR6MaxSolver._maybe_reinit_advantage_net cadence formula

The cadence tests duplicate the solver's formula inline so they validate
the math independently of solver state — they're pure unit tests.
"""
import inspect
import pytest
import torch


# ============================================================
# Cadence formula — pure math, no project imports.
# ============================================================

NUM_SEATS = 6


def _per_seat_iter(global_iter: int, num_seats: int = NUM_SEATS) -> int:
    """Mirror of formula in DeepCFR6MaxSolver._maybe_reinit_advantage_net."""
    return ((global_iter - 1) // num_seats) + 1


def _reinit_fires(global_iter: int, every, num_seats: int = NUM_SEATS) -> bool:
    """Whether reinit fires at this global iter."""
    if every is None or every <= 0:
        return False
    per_seat = _per_seat_iter(global_iter, num_seats)
    return per_seat > 0 and per_seat % every == 0


class TestCadenceFormula:
    def test_per_seat_advances_every_6_global(self):
        for g in range(1, 7):
            assert _per_seat_iter(g) == 1
        for g in range(7, 13):
            assert _per_seat_iter(g) == 2
        for g in range(13, 19):
            assert _per_seat_iter(g) == 3

    def test_every_2_fires_at_per_seat_2_and_4(self):
        # per_seat=2 spans iters 7-12, per_seat=4 spans iters 19-24
        fires = [i for i in range(1, 25) if _reinit_fires(i, every=2)]
        assert fires == [7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24]

    def test_every_5_fires_at_per_seat_5(self):
        # per_seat=5 spans iters 25-30
        fires = [i for i in range(1, 31) if _reinit_fires(i, every=5)]
        assert fires == [25, 26, 27, 28, 29, 30]

    def test_every_1_fires_every_iter(self):
        # per_seat=K for any K is a multiple of 1, so every iter fires
        assert all(_reinit_fires(i, every=1) for i in range(1, 50))

    def test_none_disables(self):
        assert not any(_reinit_fires(i, every=None) for i in range(1, 50))

    def test_zero_disables(self):
        assert not any(_reinit_fires(i, every=0) for i in range(1, 50))

    def test_negative_disables(self):
        assert not any(_reinit_fires(i, every=-3) for i in range(1, 50))


# ============================================================
# reinit_seat() behavior — requires constructed PlayerNetworks6Max.
# ============================================================

@pytest.fixture(scope="module")
def nets():
    """Construct PlayerNetworks6Max with introspected kwargs."""
    try:
        from src.nlhe.networks6 import PlayerNetworks6Max
    except ImportError as e:
        pytest.skip(f"PlayerNetworks6Max import failed: {e}")

    sig = inspect.signature(PlayerNetworks6Max.__init__)
    params = sig.parameters
    candidates = {
        "input_dim": 16, "feature_dim": 16, "in_dim": 16,
        "hidden": [32, 32], "hidden_dim": [32, 32], "hidden_dims": [32, 32],
        "buffer_capacity": 1000,
        "learning_rate": 0.001, "lr": 0.001,
        "device": "cpu",
        "num_seats": 6, "n_seats": 6,
    }
    kwargs = {k: v for k, v in candidates.items() if k in params}
    try:
        return PlayerNetworks6Max(**kwargs)
    except TypeError as e:
        pytest.skip(f"could not construct PlayerNetworks6Max with inferred kwargs "
                    f"({kwargs}): {e}")


def _weight_sig(net) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in net.parameters()])


class TestReinitSeat:
    def test_reinit_method_exists(self, nets):
        assert hasattr(nets, "reinit_seat"), \
            "reinit_seat() missing — patch may not have been applied"

    def test_reinit_changes_weights(self, nets):
        before = _weight_sig(nets.nets[0])
        nets.reinit_seat(0)
        after = _weight_sig(nets.nets[0])
        assert before.shape == after.shape
        # Probability of two fresh random inits matching is ~0.
        assert not torch.allclose(before, after, atol=1e-7), \
            "weights unchanged across reinit_seat()"

    def test_reinit_does_not_leak_to_other_seats(self, nets):
        # Capture all OTHER seats first, then reinit seat 0.
        others_before = {s: _weight_sig(nets.nets[s]) for s in range(NUM_SEATS) if s != 0}
        nets.reinit_seat(0)
        for s, before in others_before.items():
            after = _weight_sig(nets.nets[s])
            assert torch.allclose(before, after), \
                f"reinit_seat(0) modified seat {s}'s net"

    def test_reinit_preserves_buffer_object(self, nets):
        # Same buffer object should be referenced before and after.
        # (Patch explicitly says buffers are not touched.)
        buf_before = nets.buffer_for(0)
        nets.reinit_seat(0)
        buf_after = nets.buffer_for(0)
        assert buf_before is buf_after, \
            "buffer was replaced across reinit_seat() — should be preserved"

    def test_reinit_replaces_optimizer_by_default(self, nets):
        opt_before = nets.optimizers[0]
        nets.reinit_seat(0)
        assert nets.optimizers[0] is not opt_before, \
            "optimizer instance unchanged (should be replaced by default)"

    def test_reinit_keeps_optimizer_when_reset_optimizer_false(self, nets):
        opt_before = nets.optimizers[0]
        nets.reinit_seat(0, reset_optimizer=False)
        assert nets.optimizers[0] is opt_before, \
            "optimizer replaced despite reset_optimizer=False"
