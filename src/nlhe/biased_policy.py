"""Biased blueprint policies for depth-limited subgame solving (Track B1).

A "biased policy" is the trained blueprint policy modulated by per-action-class
bias multipliers, renormalized over the legal-action distribution. Used by the
subgame solver as one of the k continuation strategies that players (hero and
opponents) choose from at leaf infosets, per Brown/Sandholm/Amos NeurIPS-18 +
Pluribus Science-19.

This module is the "leaf strategies" piece of the B1 plan (docs/B1_PLAN.md).

STATUS: SKETCH ONLY. The class is implemented; integration with the subgame
solver lives in src/nlhe/subgame.py (not yet written, B1c).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.nlhe.actions import DiscreteAction


# Per-action bias factors for each of the four standard continuation policies.
# Brown 2018 used alpha=5 as the base multiplier with k=4 strategies. We use
# alpha=3.0 as a starting value, more conservative per our values-driven
# robustness preference. Tune empirically during B1b.
_DEFAULT_ALPHA = 3.0


@dataclass(frozen=True)
class BiasConfig:
    """Per-action bias multipliers. Vector indexed by DiscreteAction value."""
    name: str
    multipliers: np.ndarray  # shape (NUM_ACTIONS,)

    def __post_init__(self) -> None:
        if self.multipliers.shape != (len(DiscreteAction),):
            raise ValueError(
                f"BiasConfig multipliers must be shape ({len(DiscreteAction)},), "
                f"got {self.multipliers.shape}"
            )
        if np.any(self.multipliers <= 0):
            raise ValueError("BiasConfig multipliers must all be positive")


def standard_bias_configs(alpha: float = _DEFAULT_ALPHA) -> list[BiasConfig]:
    """The k=4 continuation-strategy bias configs used at leaf nodes.

    Index 0 is the unbiased blueprint (identity). The other three lean
    toward distinct strategic directions:
      1. Fold-biased: more folds, fewer bets.
      2. Call-biased: more calls, fewer bets.
      3. Raise-biased: more bets at all sizes, fewer folds/calls.

    The bias factor `alpha` controls how strong the bias is; alpha=1.0
    recovers the blueprint everywhere. Default alpha=3.0 is mid-conservative.
    """
    n = len(DiscreteAction)
    # action indices
    F, C = DiscreteAction.FOLD, DiscreteAction.CALL
    BETS = (DiscreteAction.BET_33, DiscreteAction.BET_66,
            DiscreteAction.BET_100, DiscreteAction.BET_200, DiscreteAction.ALLIN)

    def mults(passive_up: list[DiscreteAction], aggressive_down: list[DiscreteAction]) -> np.ndarray:
        m = np.ones(n)
        for a in passive_up:
            m[int(a)] = alpha
        for a in aggressive_down:
            m[int(a)] = 1.0 / alpha
        return m

    return [
        BiasConfig(name="blueprint", multipliers=np.ones(n)),
        BiasConfig(name="fold-biased",  multipliers=mults([F], list(BETS))),
        BiasConfig(name="call-biased",  multipliers=mults([C], list(BETS))),
        BiasConfig(name="raise-biased", multipliers=mults([], [F, C])),  # bets unchanged, F/C reduced
    ]


def apply_bias(
    probs: np.ndarray,
    legal_mask: np.ndarray,
    bias: BiasConfig,
) -> np.ndarray:
    """Apply a BiasConfig to a blueprint action distribution.

    Args:
        probs: blueprint probabilities, shape (NUM_ACTIONS,), masked to legal
          (zero on illegal indices) and summing to 1 over legal.
        legal_mask: 0/1 mask over actions, shape (NUM_ACTIONS,).
        bias: BiasConfig with per-action multipliers.

    Returns:
        Biased+renormalized probability vector, shape (NUM_ACTIONS,), summing
        to 1 over legal actions, zero elsewhere.
    """
    if probs.shape != (len(DiscreteAction),):
        raise ValueError(f"probs must be shape ({len(DiscreteAction)},), got {probs.shape}")
    if legal_mask.shape != probs.shape:
        raise ValueError(f"legal_mask must match probs shape")
    biased = probs * bias.multipliers * legal_mask
    denom = float(biased.sum())
    if denom <= 0:
        # Bias zeroed out all legal probability mass. Fall back to uniform-over-legal.
        # This can happen e.g. if blueprint had ~all mass on bets and bias = fold-biased
        # set those to 1/alpha while fold itself wasn't legal.
        return legal_mask / float(legal_mask.sum())
    return biased / denom


@dataclass
class BiasedBlueprint:
    """Wraps a trained blueprint with a set of k continuation strategies.

    Each continuation strategy is the blueprint modulated by one BiasConfig.
    The subgame solver queries this object at leaf nodes with a chosen
    strategy index in [0, k).

    This class does NOT execute the underlying blueprint network — the caller
    provides masked blueprint probs as input to action_probs(). The subgame
    solver is responsible for running the network once per leaf infoset and
    feeding the result here for each of the k strategy choices.
    """
    bias_configs: list[BiasConfig] = field(default_factory=standard_bias_configs)

    @property
    def k(self) -> int:
        return len(self.bias_configs)

    def action_probs(
        self,
        blueprint_probs: np.ndarray,
        legal_mask: np.ndarray,
        strategy_idx: int,
    ) -> np.ndarray:
        """Given blueprint probs at an infoset, return biased probs for strategy_idx."""
        if not 0 <= strategy_idx < self.k:
            raise ValueError(f"strategy_idx {strategy_idx} out of range [0, {self.k})")
        return apply_bias(blueprint_probs, legal_mask, self.bias_configs[strategy_idx])

    def strategy_name(self, strategy_idx: int) -> str:
        return self.bias_configs[strategy_idx].name
