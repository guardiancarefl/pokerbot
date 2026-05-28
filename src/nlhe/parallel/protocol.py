"""Picklable dataclasses for the parallel-traversal worker contract.

Workers receive a WorkerInput, run their assigned traversals against a
read-only copy of the policy nets, and return one WorkerOutput per
traversal_id. The orchestrator collects the WorkerOutputs and replays
their samples into the real ReservoirBuffers in strict ascending t order
(see DESIGN.md).

Everything here must be picklable across a multiprocessing.get_context('fork')
boundary: no pyspiel.Game, no torch.nn.Module, no live ReservoirBuffer. Net
weights cross as state_dicts (CPU tensors); the game crosses as its string;
the abstraction crosses as a filesystem path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class TraversalSample:
    """One buffer-add tuple, matching ReservoirBuffer.add()'s signature
    exactly: (feature, target, legal_mask, iteration). Used for both
    advantage samples (target = regrets) and strategy samples
    (target = strat distribution); the shape is identical, and which list
    a sample belongs to determines its destination buffer at merge time.
    """
    feature: np.ndarray
    target: np.ndarray
    legal_mask: np.ndarray
    iteration: int


@dataclass
class WorkerInput:
    """Everything a fork()ed worker needs to reproduce its assigned
    traversals deterministically. See DESIGN.md "Worker contract" for the
    confirmed-complete input set.
    """
    # ---- RNG / identity ----
    seed: int
    iteration: int
    traverser: int
    traversal_ids: list[int]

    # ---- Read-only advantage nets ----
    # State dicts only; the worker rebuilds MLPs locally. Strategy net is
    # never read during traversal and is intentionally NOT shipped.
    adv_state_dicts: list[dict]
    input_dim: int
    hidden_dim: list[int]

    # ---- Abstraction + encoder construction ----
    abstraction_path: str
    encoder_starting_stack: int
    encoder_max_bucket_dim: int
    encoder_bucket_runouts: int

    # ---- Game ----
    # Worker calls pyspiel.load_game(game_str) locally; pyspiel.Game is not
    # pickled across processes.
    game_str: str

    # ---- CFR6MaxContext scalar fields ----
    starting_stacks: list[int]
    payouts: list[float]
    max_depth: int
    num_paid: int
    dealer_seat: Optional[int] = None


@dataclass
class WorkerOutput:
    """One traversal's collected samples, tagged for fixed-order merge."""
    traversal_id: int
    adv_samples: list[TraversalSample]
    strat_samples: list[TraversalSample]
