"""Depth-limited CFR: the validated engine with a leaf-substitution hook.

`DepthLimitedCFR` is `TabularCFR` plus one thing: at any non-terminal state for
which `leaf_predicate(state)` is True, the recursion stops and substitutes
`leaf_value_fn(state)` (a per-player expected-utility vector) instead of
expanding the subtree. That single hook is the whole basis of ReBeL search —
everything beyond the depth limit is replaced by a (learned, or here oracle)
value function.

Correctness contract (the reason this subclasses rather than reimplements):

  * With `leaf_predicate=None` the override is a no-op and the solver is
    *byte-for-byte* the validated full-tree engine — so it must reproduce the
    engine's Leduc number (CFR+ @1000 = 0.1234 mbb/g). That is the Layer-1 gate.

  * The frozen-strategy / deferred-commit / alternating-update discipline that
    makes the engine converge to Nash is inherited unchanged; we only shorten
    the trajectories. A leaf value substituted at `state` must therefore be the
    *counterfactual-reach-independent* expected utility of the subtree under the
    continuation strategy — i.e. exactly what `_cfr` would have returned by
    recursing. The CFR backup multiplies by counterfactual reach itself, so the
    leaf fn must NOT pre-weight by reach.

The leaf value function receives the OpenSpiel `state` (which, for poker, embeds
the private cards). For a *belief-aware* leaf (the PBS value net, or the oracle
re-solve), the value at a public leaf depends on the opponent's range reaching
it; that range is carried by the solve's reach probabilities and is supplied to
belief-aware leaves through `src.rebel.pbs` — see `value_oracle.py`. The plain
hook here stays deliberately minimal and reach-agnostic.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pyspiel

from src.rlcfr.cfr import DCFRParams, TabularCFR

# A leaf predicate decides where the subgame ends; a leaf value fn supplies the
# per-player expected-utility vector there.
LeafPredicate = Callable[[pyspiel.State], bool]
LeafValueFn = Callable[[pyspiel.State], np.ndarray]


class DepthLimitedCFR(TabularCFR):
    """Full-tree (D)CFR with an optional leaf-substitution hook.

    Parameters
    ----------
    game, abstraction_fn, dcfr
        Forwarded to `TabularCFR` unchanged.
    leaf_predicate
        `state -> bool`. When True at a *non-terminal, non-chance* state the
        recursion stops and `leaf_value_fn(state)` is used. `None` (default)
        disables the hook entirely, recovering the exact engine behaviour.
    leaf_value_fn
        `state -> np.ndarray` of shape `(num_players,)`: the expected utility
        per player at the leaf, NOT pre-weighted by reach. Required iff
        `leaf_predicate` is not None.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        abstraction_fn=None,
        dcfr: Optional[DCFRParams] = None,
        leaf_predicate: Optional[LeafPredicate] = None,
        leaf_value_fn: Optional[LeafValueFn] = None,
    ) -> None:
        super().__init__(game, abstraction_fn=abstraction_fn, dcfr=dcfr)
        if leaf_predicate is not None and leaf_value_fn is None:
            raise ValueError("leaf_predicate given but no leaf_value_fn")
        self.leaf_predicate = leaf_predicate
        self.leaf_value_fn = leaf_value_fn
        # Count leaf substitutions per pass — a cheap sanity signal that the cut
        # is actually firing where we think it is.
        self.leaf_hits = 0

    def _cfr(self, state: pyspiel.State, reach: np.ndarray,
             traverser: Optional[int]) -> np.ndarray:
        if state.is_terminal():
            return np.asarray(state.returns(), dtype=np.float64)

        # The single ReBeL hook: substitute a leaf value before any expansion.
        # Chance nodes are eligible too (Leduc's round->round public-card deal is
        # a chance node, and that is exactly the natural depth cut), but the root
        # is never a leaf.
        if self.leaf_predicate is not None and self.leaf_predicate(state):
            self.leaf_hits += 1
            val = np.asarray(self.leaf_value_fn(state), dtype=np.float64)
            if val.shape != (self.num_players,):
                raise ValueError(
                    f"leaf_value_fn returned shape {val.shape}, "
                    f"expected ({self.num_players},)")
            return val

        # Otherwise: identical to the validated engine.
        return super()._cfr(state, reach, traverser)

    def _pass(self, traverser, t) -> None:  # reset the per-pass leaf counter
        self.leaf_hits = 0
        super()._pass(traverser, t)
