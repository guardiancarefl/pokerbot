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
