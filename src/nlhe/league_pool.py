"""Opponent sampling pool for league-play training.

Wraps a CheckpointRegistry and exposes a sample_opponent(rng) method that
returns a loaded policy ready to plug into the Policy protocol from
scripts/eval_pool.py (CheckpointPolicy.select_action signature).

Three sampling strategies:
  - "uniform": every registered checkpoint equally likely.
  - "weighted": caller supplies a dict of name -> non-negative weight.
    Weights need not sum to 1; LeaguePool normalizes.
  - "recency": favor most-recently-registered checkpoints with an
    exponentially-decaying weight (configurable half-life in entries).

Solvers are lazy-loaded and cached. First sample of a checkpoint loads
the .pt; subsequent samples reuse the cached CheckpointPolicy. This means
a long training run only pays the load cost once per opponent, not per
sample.

Design notes:
  - LeaguePool does NOT mutate the underlying registry. The registry is
    the source of truth; the pool is a sampling view over it.
  - CheckpointPolicy is imported from scripts.eval_pool to keep a single
    Policy protocol implementation. If that file moves, update the
    import here.
  - The pool requires abstraction + structure objects at construction
    because CheckpointPolicy needs them to load a solver. This couples
    the pool to the training context, which is intentional: a league
    pool is meaningless without the abstraction it was trained against.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from src.nlhe.checkpoint_registry import CheckpointRegistry, CheckpointEntry


_VALID_STRATEGIES = ("uniform", "weighted", "recency")


class LeaguePool:
    """Sampling pool over a CheckpointRegistry."""

    def __init__(
        self,
        registry: CheckpointRegistry,
        abstraction,
        structure,
        sample_strategy: str = "uniform",
        weights: Optional[dict] = None,
        recency_halflife: float = 5.0,
        tag_filter: Optional[list] = None,
    ):
        """
        Args:
            registry: CheckpointRegistry to draw from.
            abstraction: loaded Abstraction object (for CheckpointPolicy).
            structure: loaded TournamentStructure (for CheckpointPolicy).
            sample_strategy: "uniform", "weighted", or "recency".
            weights: required if sample_strategy == "weighted". Dict of
                name -> non-negative float. Names must exist in registry.
            recency_halflife: half-life in entries for "recency" strategy.
                Most-recent gets weight 1; entry k back gets weight
                0.5 ** (k / halflife). Default 5.
            tag_filter: optional list of tags. If provided, only checkpoints
                with at least one matching tag are sampled. Useful for
                e.g. restricting league to "dcfr"-tagged checkpoints.
        """
        if sample_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"sample_strategy must be one of {_VALID_STRATEGIES}, "
                f"got {sample_strategy!r}"
            )
        self.registry = registry
        self.abstraction = abstraction
        self.structure = structure
        self.sample_strategy = sample_strategy
        self.recency_halflife = recency_halflife
        self.tag_filter = list(tag_filter) if tag_filter else None

        # Resolve eligible entries (post-filter) once at construction.
        if self.tag_filter:
            self._eligible = registry.query(tags=self.tag_filter, match="any")
        else:
            self._eligible = list(registry)

        # Validate weighted strategy has the required keys.
        if sample_strategy == "weighted":
            if not weights:
                raise ValueError("weighted strategy requires non-empty weights")
            eligible_names = {e.name for e in self._eligible}
            for name in weights:
                if name not in eligible_names:
                    raise ValueError(
                        f"weight key {name!r} not in eligible registry entries"
                    )
            for w in weights.values():
                if w < 0:
                    raise ValueError("weights must be non-negative")
            self.weights = dict(weights)
        else:
            self.weights = None

        # Lazy-load cache: name -> CheckpointPolicy.
        self._policy_cache: dict = {}

    def __len__(self) -> int:
        return len(self._eligible)

    def eligible_names(self) -> list:
        return [e.name for e in self._eligible]

    def _compute_weights(self) -> list:
        """Return parallel list of weights for self._eligible."""
        if self.sample_strategy == "uniform":
            return [1.0] * len(self._eligible)
        if self.sample_strategy == "weighted":
            # Pull from self.weights; default missing to 0.
            return [self.weights.get(e.name, 0.0) for e in self._eligible]
        if self.sample_strategy == "recency":
            n = len(self._eligible)
            # Most recent (last registered) gets weight 1.0; oldest gets smallest.
            # Index 0 in self._eligible is oldest (registry preserves registration order),
            # index n-1 is newest. weight = 0.5 ** ((n-1-i) / halflife).
            hl = max(self.recency_halflife, 1e-9)
            return [0.5 ** ((n - 1 - i) / hl) for i in range(n)]
        raise RuntimeError(f"unreachable strategy: {self.sample_strategy}")

    def sample_entry(self, rng: random.Random) -> CheckpointEntry:
        """Sample one CheckpointEntry according to the current strategy."""
        if not self._eligible:
            raise RuntimeError("no eligible checkpoints to sample from")
        weights = self._compute_weights()
        total = sum(weights)
        if total <= 0:
            raise RuntimeError(
                "all sampling weights are zero; pool cannot sample"
            )
        return rng.choices(self._eligible, weights=weights, k=1)[0]

    def sample_opponent(self, rng: random.Random):
        """Sample an entry, lazy-load its policy, return a CheckpointPolicy.

        First call for a given entry loads the solver (expensive).
        Subsequent calls return the cached CheckpointPolicy.
        """
        entry = self.sample_entry(rng)
        if entry.name in self._policy_cache:
            return self._policy_cache[entry.name]

        # Lazy import to avoid circular import at module load time.
        from scripts.eval_pool import CheckpointPolicy
        policy = CheckpointPolicy(
            name=entry.name,
            ckpt_path=entry.path,
            abstraction=self.abstraction,
            structure=self.structure,
        )
        self._policy_cache[entry.name] = policy
        return policy

    def cache_info(self) -> dict:
        """Return current cache state for introspection / debugging."""
        return {
            "eligible_count": len(self._eligible),
            "cached_count": len(self._policy_cache),
            "cached_names": list(self._policy_cache.keys()),
        }

    def clear_cache(self) -> None:
        """Drop all cached loaded policies. Frees GPU/CPU memory."""
        self._policy_cache.clear()
