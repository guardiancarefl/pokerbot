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

    # ---- Phase 2: override pool construction (picklable scalars/strings) ----
    # Workers reconstruct league_pool / archetype_pool locally from these and
    # sample the override deterministically per (seed, iter, t) using the
    # same OVERRIDE_SALT-keyed rng_override_t the orchestrator uses for the
    # counter bookkeeping. None paths → pool stays None → mix=0 callers
    # are bit-identical to Phase 1 (no override sampling, no rng draw).
    league_registry_path: Optional[str] = None
    league_sample_strategy: str = "uniform"
    league_weights: Optional[dict] = None
    league_recency_halflife: float = 5.0
    league_tag_filter: Optional[list] = None
    league_mix: float = 0.0
    archetype_calibration_path: Optional[str] = None
    archetype_profile_names: Optional[list] = None
    archetype_mix: float = 0.0

    # ---- Phase 3: tournament-mode structure (picklable string path) ----
    # When set, workers load TournamentStructure from this path and sample
    # per-traversal starting stacks/dealer via sample_starting_state. None
    # → legacy fixed-game mode (the existing per-WorkerInput game_str path).
    tournament_structure_path: Optional[str] = None


@dataclass
class WorkerOutput:
    """One traversal's collected samples, tagged for fixed-order merge."""
    traversal_id: int
    adv_samples: list[TraversalSample]
    strat_samples: list[TraversalSample]
