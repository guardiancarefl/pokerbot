"""Tests for src/nlhe/networks6.py (Phase 4e.3a)."""
from __future__ import annotations
import random

import numpy as np
import pytest
import torch

from src.nlhe.networks6 import PlayerNetworks6Max, N_DISCRETE_ACTIONS, NUM_SEATS_6MAX


# ---- Basic construction ----

def test_constructs_6_nets():
    pn = PlayerNetworks6Max(input_dim=236, hidden=[64, 64])
    assert len(pn.nets) == NUM_SEATS_6MAX == 6
    assert len(pn.optimizers) == 6
    assert len(pn.buffers) == 6


def test_each_net_has_correct_shape():
    pn = PlayerNetworks6Max(input_dim=236, hidden=[64, 64])
    for seat in range(6):
        net = pn.net_for(seat)
        # Probe with a dummy input
        x = torch.zeros(1, 236)
        y = net(x)
        assert y.shape == (1, N_DISCRETE_ACTIONS), f"seat {seat} output shape {y.shape}"


def test_seat_out_of_range_raises():
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    with pytest.raises(IndexError):
        pn.net_for(6)
    with pytest.raises(IndexError):
        pn.net_for(-1)


# ---- Forward pass ----

def test_predict_advantages_returns_correct_shape():
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    features = np.zeros(236, dtype=np.float32)
    adv = pn.predict_advantages(seat=0, features=features)
    assert adv.shape == (N_DISCRETE_ACTIONS,)
    assert adv.dtype == np.float32 or adv.dtype == np.float64


def test_each_seat_has_independent_parameters():
    """Networks should be initialized independently (not sharing weights)."""
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    # Different seats should produce different outputs for the same input
    # (because they have different random initializations)
    features = np.ones(236, dtype=np.float32)
    outputs = []
    for seat in range(6):
        outputs.append(pn.predict_advantages(seat, features))
    # At least two seats should produce different outputs
    different_pairs = 0
    for i in range(6):
        for j in range(i + 1, 6):
            if not np.allclose(outputs[i], outputs[j]):
                different_pairs += 1
    assert different_pairs > 0, "all 6 seats produced identical outputs — networks share weights"


# ---- Buffer access ----

def test_buffers_start_empty():
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    for seat in range(6):
        assert len(pn.buffer_for(seat)) == 0


def test_buffer_add_works():
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32], buffer_capacity=100,
                            rng=random.Random(42))
    feature = np.zeros(236, dtype=np.float32)
    target = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
    mask = np.ones(N_DISCRETE_ACTIONS, dtype=np.float32)
    pn.buffer_for(0).add(feature, target, mask, iteration=1)
    assert len(pn.buffer_for(0)) == 1
    # Other seats' buffers are unaffected
    for seat in range(1, 6):
        assert len(pn.buffer_for(seat)) == 0


# ---- Optimizer ----

def test_optimizer_is_adam_with_correct_lr():
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32], learning_rate=2e-3)
    for seat in range(6):
        opt = pn.optimizer_for(seat)
        assert isinstance(opt, torch.optim.Adam)
        # All param groups should have the configured LR
        for group in opt.param_groups:
            assert group["lr"] == 2e-3


def test_optimizer_can_step():
    """A backward + optimizer.step() actually updates the net's weights."""
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32], learning_rate=1e-2)
    seat = 0
    # Snapshot the first layer's weight before training
    first_layer = list(pn.net_for(seat).net.parameters())[0]
    initial = first_layer.detach().clone()

    # Do one gradient step
    x = torch.randn(4, 236)
    target = torch.randn(4, N_DISCRETE_ACTIONS)
    out = pn.net_for(seat)(x)
    loss = ((out - target) ** 2).mean()
    pn.optimizer_for(seat).zero_grad()
    loss.backward()
    pn.optimizer_for(seat).step()

    after = list(pn.net_for(seat).net.parameters())[0]
    # Weights should have changed
    assert not torch.allclose(initial, after), "optimizer.step() didn't change weights"


# ---- Device handling ----

def test_to_device_cpu():
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32], device="cpu")
    pn.to("cpu")  # no-op but should work
    for net in pn.nets:
        for param in net.parameters():
            assert param.device.type == "cpu"


# ---- State dict roundtrip ----

def test_state_dict_save_and_load():
    pn1 = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    sd = pn1.state_dict()
    assert "nets" in sd
    assert len(sd["nets"]) == 6
    assert "optimizers" in sd
    assert sd["input_dim"] == 236
    assert sd["hidden"] == [32, 32]

    # Create a fresh container and load
    pn2 = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    pn2.load_state_dict(sd)

    # Verify parameters match across all 6 nets
    for seat in range(6):
        p1 = list(pn1.net_for(seat).parameters())
        p2 = list(pn2.net_for(seat).parameters())
        for a, b in zip(p1, p2):
            assert torch.allclose(a, b), f"seat {seat} parameters didn't roundtrip"


def test_load_state_dict_rejects_wrong_count():
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    sd = pn.state_dict()
    sd["nets"] = sd["nets"][:3]  # truncate
    with pytest.raises(ValueError, match="3 nets"):
        pn.load_state_dict(sd)


# ---- Independence: training seat 0 doesn't affect seat 1 ----

def test_training_one_seat_doesnt_affect_others():
    """Optimizer step on seat 0 leaves seat 1's parameters unchanged."""
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32], learning_rate=1e-2)
    seat1_initial = [p.detach().clone() for p in pn.net_for(1).parameters()]

    # Train seat 0 a bunch
    for _ in range(5):
        x = torch.randn(4, 236)
        target = torch.randn(4, N_DISCRETE_ACTIONS)
        out = pn.net_for(0)(x)
        loss = ((out - target) ** 2).mean()
        pn.optimizer_for(0).zero_grad()
        loss.backward()
        pn.optimizer_for(0).step()

    # Seat 1 should be untouched
    seat1_after = [p.detach().clone() for p in pn.net_for(1).parameters()]
    for a, b in zip(seat1_initial, seat1_after):
        assert torch.allclose(a, b), "seat 1's params changed when training seat 0"


# ---- v2 schema: shared strategy net + buffer ----

def test_strategy_net_present():
    """A single shared strategy net (NOT a per-seat list) with correct shape."""
    from src.nlhe.networks6 import MLP
    pn = PlayerNetworks6Max(input_dim=236, hidden=[64, 64])
    assert isinstance(pn.strat_net, MLP)
    assert not isinstance(pn.strat_net, list), "strat_net must be a single net, not a list"
    # Forward shape matches the action dim.
    y = pn.strat_net(torch.zeros(1, 236))
    assert y.shape == (1, N_DISCRETE_ACTIONS)


def test_strat_buffer_present():
    """A single shared strategy buffer (NOT a per-seat list), correct capacity."""
    from src.nlhe.solver import ReservoirBuffer
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32], buffer_capacity=2000)
    assert isinstance(pn.strat_buffer, ReservoirBuffer)
    assert not isinstance(pn.strat_buffer, list), "strat_buffer must be single, not a list"
    assert pn.strat_buffer.capacity == 2000
    assert len(pn.strat_buffer) == 0


def test_schema_version_in_state_dict():
    """state_dict() carries schema_version + strat net/optimizer, NOT buffer contents."""
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    sd = pn.state_dict()
    assert sd["schema_version"] == "v2_with_strategy"
    assert "strat_net" in sd
    assert "strat_optimizer" in sd
    # Strategy net params are reachable through the dict.
    assert any(isinstance(v, torch.Tensor) for v in sd["strat_net"].values())
    # Optimizer state is reachable.
    assert "param_groups" in sd["strat_optimizer"]
    # Buffer contents must NOT be in state_dict() — they live in solver6 (C1).
    assert "strat_buffer" not in sd


def test_v1_load_succeeds_and_sets_schema_marker():
    """REVISED two-tier design (Step E): a v1 dict (no schema_version) loads the
    advantage nets WITHOUT raising and marks the container "v1"."""
    pn_src = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    sd = pn_src.state_dict()
    v1 = {k: v for k, v in sd.items() if k != "schema_version"}  # strip → v1 shape

    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    pn.load_state_dict(v1)  # must NOT raise
    assert pn.loaded_schema_version == "v1"
    # Advantage nets loaded from the v1 dict (bit-identical to source).
    for seat in range(6):
        for a, b in zip(pn.net_for(seat).parameters(), pn_src.net_for(seat).parameters()):
            assert torch.allclose(a, b), f"seat {seat} adv params not loaded from v1 dict"


def test_v1_load_succeeds_strat_net_fresh():
    """On a v1 load the strategy net is NOT loaded — it keeps its fresh init."""
    pn_src = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    v1 = {k: v for k, v in pn_src.state_dict().items() if k != "schema_version"}

    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    strat_before = [p.detach().clone() for p in pn.strat_net.parameters()]
    pn.load_state_dict(v1)
    for after, before in zip(pn.strat_net.parameters(), strat_before):
        assert torch.allclose(after, before), "strat_net changed on a v1 load"


def test_v2_load_sets_schema_marker():
    pn_src = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    pn.load_state_dict(pn_src.state_dict())  # full v2 dict
    assert pn.loaded_schema_version == "v2_with_strategy"


def test_fresh_construction_defaults_to_v2():
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    assert pn.loaded_schema_version == "v2_with_strategy"


def test_inference_policy_uses_strat_net_when_v2():
    """On v2, inference_policy returns the strat_net's masked softmax."""
    from src.nlhe.solver import _strategy_from_advantages
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])  # fresh → v2
    feat = np.random.RandomState(0).randn(236).astype(np.float32)
    mask = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
    for i in (0, 1, 6):
        mask[i] = 1.0
    out = pn.inference_policy(seat=0, features=feat, legal_mask=mask)
    # Expected: strat_net masked softmax.
    with torch.no_grad():
        logits = pn.strat_net(torch.from_numpy(feat).unsqueeze(0)).squeeze(0).numpy()
    logits = logits - logits.max()
    exp_l = np.exp(logits) * mask
    expected = exp_l / exp_l.sum()
    assert np.allclose(out, expected, atol=1e-6)
    # And it is NOT the adv-net RM+ (a different net / computation).
    rm = _strategy_from_advantages(pn.predict_advantages(0, feat), mask)
    assert not np.allclose(out, rm), "v2 inference unexpectedly matched adv-net RM+"


def test_inference_policy_falls_back_to_regret_matched_on_v1():
    """On a v1-loaded container, inference_policy returns the adv-net RM+."""
    from src.nlhe.solver import _strategy_from_advantages
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    pn.loaded_schema_version = "v1"  # simulate a v1 load
    feat = np.random.RandomState(1).randn(236).astype(np.float32)
    mask = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
    for i in (0, 1, 4, 6):
        mask[i] = 1.0
    out = pn.inference_policy(seat=2, features=feat, legal_mask=mask)
    expected = _strategy_from_advantages(pn.predict_advantages(2, feat), mask)
    assert np.allclose(out, expected, atol=1e-6)


def test_inference_policy_masked_softmax_sums_to_one():
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])  # v2
    legal = (1, 3, 5)
    mask = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
    for i in legal:
        mask[i] = 1.0
    out = pn.inference_policy(seat=0, features=np.zeros(236, dtype=np.float32), legal_mask=mask)
    assert abs(out[list(legal)].sum() - 1.0) < 1e-5
    assert all(out[i] == 0.0 for i in range(N_DISCRETE_ACTIONS) if i not in legal)


def test_inference_policy_rng_neutral():
    """Two calls with identical inputs produce identical output (no rng inside)."""
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    feat = np.random.RandomState(2).randn(236).astype(np.float32)
    mask = np.ones(N_DISCRETE_ACTIONS, dtype=np.float32)
    a = pn.inference_policy(0, feat, mask)
    b = pn.inference_policy(0, feat, mask)
    assert np.array_equal(a, b)


def test_strat_net_shares_device_and_dtype_with_adv():
    """Strategy net lives on the same device + dtype as the advantage nets."""
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32], device="cpu")
    adv_p = next(pn.net_for(0).parameters())
    strat_p = next(pn.strat_net.parameters())
    assert strat_p.device == adv_p.device
    assert strat_p.dtype == adv_p.dtype


def test_strat_rng_independent_from_adv_rng():
    """strat_rng must be a distinct object from the adv-buffer rng (self.rng),
    and the strat buffer must actually use strat_rng (not self.rng) — the
    construction-time guarantee behind advantage-net bit-identity."""
    adv_rng = random.Random(1)
    pn = PlayerNetworks6Max(input_dim=236, hidden=[32, 32],
                            rng=adv_rng, strat_rng=random.Random(42))
    # Distinct objects: the strat rng is neither the container's adv rng nor
    # the adv rng instance we passed in.
    assert pn.strat_rng is not pn.rng, "strat_rng aliases the adv rng — writes would perturb adv stream"
    assert pn.rng is adv_rng, "adv buffers should use exactly the rng we passed (unchanged)"
    # The strat buffer uses strat_rng; adv buffers use self.rng (instance, not seed).
    assert pn.strat_buffer.rng is pn.strat_rng
    assert pn.buffer_for(0).rng is pn.rng
    assert pn.strat_buffer.rng is not pn.buffer_for(0).rng

    # And the None default resolves to a fresh Random (no None leaks to the buffer).
    pn2 = PlayerNetworks6Max(input_dim=236, hidden=[32, 32])
    assert isinstance(pn2.strat_rng, random.Random)
    assert pn2.strat_buffer.rng is pn2.strat_rng
