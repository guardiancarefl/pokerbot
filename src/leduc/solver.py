"""Wrapper around OpenSpiel's DeepCFRSolver.

The wrapper exists to (a) accept our TrainConfig dataclass instead of
positional/keyword args, and (b) return a structured result instead of a
3-tuple. The actual Deep CFR algorithm is unchanged — we're not subclassing
or modifying behavior.

If/when Phase 4 needs an ICM value function or other algorithmic changes,
this is where the customization would go.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from open_spiel.python.pytorch import deep_cfr
import pyspiel

from .config import TrainConfig


@dataclass
class TrainResult:
    """Everything the training loop produced.

    Attributes:
        policy_network: trained policy network (PyTorch nn.Module).
        action_probabilities_fn: callable(state) -> dict[action -> prob],
            backed by the policy network. Use this for evaluation.
        advantage_losses: dict mapping player_idx -> list of per-iteration
            advantage network losses.
        policy_loss: final policy network training loss.
        train_seconds: wall-clock time of solve().
    """

    policy_network: torch.nn.Module
    action_probabilities_fn: Any
    advantage_losses: dict[int, list[float]]
    policy_loss: float
    train_seconds: float


def train(game: pyspiel.Game, config: TrainConfig) -> TrainResult:
    """Run Deep CFR on the given game with the given config.

    Sets torch.manual_seed(config.seed) for reproducibility.
    """
    torch.manual_seed(config.seed)

    solver = deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=tuple(config.policy_network_layers),
        advantage_network_layers=tuple(config.advantage_network_layers),
        num_iterations=config.iterations,
        num_traversals=config.num_traversals,
        learning_rate=config.learning_rate,
        batch_size_advantage=config.batch_size_advantage,
        batch_size_strategy=config.batch_size_strategy,
        memory_capacity=config.memory_capacity,
        policy_network_train_steps=config.policy_train_steps,
        advantage_network_train_steps=config.advantage_train_steps,
    )

    t0 = time.time()
    policy_network, advantage_losses, policy_loss = solver.solve()
    train_secs = time.time() - t0

    # Convert defaultdict from OpenSpiel to plain dict, and ensure
    # losses are plain floats (not numpy/tensor objects) for JSON serialization.
    # OpenSpiel's _learn_advantage_network() can return None when the
    # advantage buffer doesn't have enough samples to form a batch. We
    # convert None to NaN so downstream code (JSON, CSV) handles it cleanly
    # without crashing, and the "no training happened" signal is preserved.
    losses_clean: dict[int, list[float]] = {
        int(p): [float(loss) if loss is not None else float("nan") for loss in losses]
        for p, losses in advantage_losses.items()
    }

    return TrainResult(
        policy_network=policy_network,
        action_probabilities_fn=solver.action_probabilities,
        advantage_losses=losses_clean,
        policy_loss=float(policy_loss),
        train_seconds=train_secs,
    )
