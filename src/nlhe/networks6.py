"""6-player advantage networks for Phase 4e.3 (CFR training).

Phase 4e.3a (docs/PHASE4_PLAN.md). The data structure that holds the
six per-player advantage networks + their optimizers + their reservoir
buffers. The CFR training loop (Phase 4e.3b/c) will use this as the
indexed network container.

In external-sampling CFR, each player needs their own regret signal
because their information sets and strategic incentives differ. For
HUNL (Phase 2d) this was 2 networks. For 6-max it's 6.

Architecture per network: MLP same as HUNL solver, output dim = 7
(matching N_DISCRETE_ACTIONS from the action abstraction). Input dim
matches the InfosetEncoder6Max feature_dim (236 by default).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reuse the existing MLP architecture from the HUNL solver.
from src.nlhe.solver import MLP, ReservoirBuffer
from src.nlhe.actions import DiscreteAction


N_DISCRETE_ACTIONS = len(DiscreteAction)
NUM_SEATS_6MAX = 6


@dataclass
class PlayerNetworks6Max:
    """Container for 6 per-seat advantage networks + optimizers + buffers.

    Args:
        input_dim: feature dim (matches InfosetEncoder6Max.feature_dim;
            default 236).
        hidden: hidden layer sizes for each MLP (e.g. [256, 256]).
        learning_rate: per-network Adam LR.
        buffer_capacity: per-seat reservoir buffer capacity.
        rng: random source for buffer reservoir sampling (one seed
            shared across all 6 buffers).
        device: 'cpu' or 'cuda'.
    """
    input_dim: int = 236
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 1e-3
    buffer_capacity: int = 100_000
    rng: random.Random = field(default_factory=random.Random)
    device: str = "cpu"

    # Populated in __post_init__.
    nets: list[MLP] = field(default_factory=list, init=False)
    optimizers: list[optim.Optimizer] = field(default_factory=list, init=False)
    buffers: list[ReservoirBuffer] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if len(self.nets) != 0:
            # Allow caller to pre-populate (e.g. when loading checkpoints).
            return

        for seat in range(NUM_SEATS_6MAX):
            net = MLP(in_dim=self.input_dim, hidden=self.hidden,
                      out_dim=N_DISCRETE_ACTIONS).to(self.device)
            self.nets.append(net)
            self.optimizers.append(optim.Adam(net.parameters(), lr=self.learning_rate))
            # Each seat gets its own buffer with its own rng — but they share
            # a single rng instance because we want determinism from a single
            # seed. Each call to add() uses the shared rng's state.
            self.buffers.append(ReservoirBuffer(capacity=self.buffer_capacity, rng=self.rng))

    # ---- Access by seat ----

    def net_for(self, seat: int) -> MLP:
        self._check_seat(seat)
        return self.nets[seat]

    def optimizer_for(self, seat: int) -> optim.Optimizer:
        self._check_seat(seat)
        return self.optimizers[seat]

    def buffer_for(self, seat: int) -> ReservoirBuffer:
        self._check_seat(seat)
        return self.buffers[seat]

    def reinit_seat(self, seat: int, reset_optimizer: bool = True) -> None:
        """Reset seat's advantage net to fresh init. Buffer is untouched.

        Brown 2019 (Deep CFR, sec 4.3): periodically resetting the
        advantage network weights and retraining from scratch on the
        accumulated reservoir buffer dramatically improves convergence.
        The buffer carries the regret history; the net just needs to
        relearn the current sampling distribution.

        reset_optimizer defaults to True because keeping Adam's
        momentum/variance estimates across a net reset typically causes
        catastrophic first-gradient updates (huge stale momentum hits a
        fresh net). Almost always you want this True.
        """
        self._check_seat(seat)
        fresh = MLP(in_dim=self.input_dim, hidden=self.hidden,
                    out_dim=N_DISCRETE_ACTIONS).to(self.device)
        self.nets[seat] = fresh
        if reset_optimizer:
            self.optimizers[seat] = optim.Adam(
                fresh.parameters(), lr=self.learning_rate
            )

    def _check_seat(self, seat: int) -> None:
        if not (0 <= seat < NUM_SEATS_6MAX):
            raise IndexError(f"seat {seat} out of range [0, {NUM_SEATS_6MAX})")

    # ---- Device movement ----

    def to(self, device: str) -> "PlayerNetworks6Max":
        """Move all 6 nets to the given device. Optimizer state moves with
        the nets (Adam state is tied to parameter tensors).
        """
        self.device = device
        for net in self.nets:
            net.to(device)
        return self

    # ---- Forward pass convenience ----

    def predict_advantages(self, seat: int, features: np.ndarray) -> np.ndarray:
        """Run a forward pass for the given seat. Returns numpy array of
        shape (n_actions,).
        """
        self._check_seat(seat)
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.nets[seat](x)
        return out.squeeze(0).cpu().numpy()

    # ---- Checkpoint state ----

    def state_dict(self) -> dict:
        """Snapshot all 6 nets' parameters + optimizer states for saving."""
        return {
            "nets": [n.state_dict() for n in self.nets],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "input_dim": self.input_dim,
            "hidden": list(self.hidden),
            "learning_rate": self.learning_rate,
            "buffer_capacity": self.buffer_capacity,
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore from a state_dict produced by self.state_dict()."""
        if len(sd["nets"]) != NUM_SEATS_6MAX:
            raise ValueError(f"state_dict has {len(sd['nets'])} nets, expected {NUM_SEATS_6MAX}")
        for i, net_sd in enumerate(sd["nets"]):
            self.nets[i].load_state_dict(net_sd)
        for i, opt_sd in enumerate(sd["optimizers"]):
            self.optimizers[i].load_state_dict(opt_sd)
