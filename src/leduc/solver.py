"""Wrapper around OpenSpiel's DeepCFRSolver.

The wrapper exists to:
  (a) accept our TrainConfig dataclass instead of positional/keyword args
  (b) return a structured result instead of a 3-tuple
  (c) drive training one iteration at a time so we get per-iteration
      progress logging and optional periodic evaluation hooks

The actual Deep CFR algorithm is unchanged — we're not subclassing the
solver or modifying its behavior, we're just calling solve(num_iterations=1)
in a Python loop instead of solve(num_iterations=N) once.

Cost of the per-iteration loop: each call to solve() re-trains the policy
network at the end, so we pay for N policy-network training passes instead
of 1. For Leduc this overhead is small (~10% of total train time) and the
visibility win is worth it. Phase 2+ may want to skip back to a single
solve() call or call internal methods directly for performance.

If/when Phase 4 needs an ICM value function or other algorithmic changes,
this is where the customization would go.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

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
        train_seconds: wall-clock time of the full training loop.
        per_iteration_seconds: list of wall-clock seconds per iteration.
        intermediate_exploitability: list of (iteration, exploitability_mbb)
            tuples for every iteration where the eval callback was invoked.
            Empty if no eval_callback was provided.
    """

    policy_network: torch.nn.Module
    action_probabilities_fn: Any
    advantage_losses: dict[int, list[float]]
    policy_loss: float
    train_seconds: float
    per_iteration_seconds: list[float]
    intermediate_exploitability: list[tuple[int, float]]


def train(
    game: pyspiel.Game,
    config: TrainConfig,
    logger: logging.Logger | None = None,
    eval_callback: Callable[[Any], float] | None = None,
    eval_every: int = 10,
) -> TrainResult:
    """Run Deep CFR on the given game with the given config.

    Args:
        game: an OpenSpiel game.
        config: TrainConfig with hyperparameters.
        logger: optional logger for per-iteration progress lines. If None,
            training is silent.
        eval_callback: optional function called as eval_callback(action_probs_fn)
            every `eval_every` iterations and at the end. Should return a
            scalar (we treat it as exploitability in mbb/g for the log line,
            but any scalar works). Pass None to skip eval entirely.
        eval_every: how often to invoke eval_callback. Ignored if
            eval_callback is None.
    """
    torch.manual_seed(config.seed)

    if logger is None:
        logger = logging.getLogger("leduc_train.silent")
        logger.addHandler(logging.NullHandler())

    # Construct with num_iterations=1; we drive the loop ourselves.
    solver = deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=tuple(config.policy_network_layers),
        advantage_network_layers=tuple(config.advantage_network_layers),
        num_iterations=1,
        num_traversals=config.num_traversals,
        learning_rate=config.learning_rate,
        batch_size_advantage=config.batch_size_advantage,
        batch_size_strategy=config.batch_size_strategy,
        memory_capacity=config.memory_capacity,
        policy_network_train_steps=config.policy_train_steps,
        advantage_network_train_steps=config.advantage_train_steps,
    )

    # Accumulators across iterations.
    advantage_losses_all: dict[int, list[float]] = {}
    per_iter_secs: list[float] = []
    intermediate_expl: list[tuple[int, float]] = []
    last_policy_loss: float = float("nan")
    last_policy_network = None

    overall_t0 = time.time()
    for it in range(1, config.iterations + 1):
        iter_t0 = time.time()
        policy_network, advantage_losses, policy_loss = solver.solve()
        iter_secs = time.time() - iter_t0
        per_iter_secs.append(iter_secs)
        last_policy_loss = float(policy_loss) if policy_loss is not None else float("nan")
        last_policy_network = policy_network

        # OpenSpiel returns a defaultdict keyed by player; the values are
        # 1-element lists when num_iterations=1. Accumulate across our loop.
        for p, losses in advantage_losses.items():
            entry = advantage_losses_all.setdefault(int(p), [])
            for loss in losses:
                entry.append(float(loss) if loss is not None else float("nan"))

        # Build a per-iteration log line.
        loss_str_parts = []
        for p in sorted(advantage_losses_all.keys()):
            val = advantage_losses_all[p][-1]
            loss_str_parts.append(f"p{p}={val:.4f}")
        loss_str = ", ".join(loss_str_parts)

        log_line = (
            f"iter {it:4d}/{config.iterations} | "
            f"adv_loss [{loss_str}] | "
            f"policy_loss={last_policy_loss:.4f} | "
            f"{iter_secs:.2f}s"
        )

        # Eval at the requested cadence and on the final iteration.
        do_eval = (
            eval_callback is not None
            and (it % eval_every == 0 or it == config.iterations)
        )
        if do_eval:
            eval_t0 = time.time()
            metric = eval_callback(solver.action_probabilities)
            eval_secs = time.time() - eval_t0
            intermediate_expl.append((it, float(metric)))
            log_line += f" | eval={metric:.3f} ({eval_secs:.1f}s)"

        logger.info(log_line)

    train_secs = time.time() - overall_t0

    return TrainResult(
        policy_network=last_policy_network,
        action_probabilities_fn=solver.action_probabilities,
        advantage_losses=advantage_losses_all,
        policy_loss=last_policy_loss,
        train_seconds=train_secs,
        per_iteration_seconds=per_iter_secs,
        intermediate_exploitability=intermediate_expl,
    )
