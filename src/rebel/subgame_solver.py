"""PBS-rooted subgame solver — the core of ReBeL re-solving.

`SubgameCFR` solves a subgame that starts not at the game root but at a set of
*root states*, each carrying an initial per-player reach (its belief / range
weight). This is the operation ReBeL performs at every public decision point:
take the current public state + both players' ranges, build the subgame, and
solve it (depth-limited, with a value function at the leaves).

It subclasses the validated `TabularCFR` and changes exactly one thing — the
entry point of a CFR pass. The full-tree engine starts every pass at
`new_initial_state()` with reach = all-ones; `SubgameCFR` instead starts each
pass by traversing every supplied root with that root's initial reach. The
frozen-strategy / deferred-commit / alternating-update discipline (and thus the
Nash-convergence guarantee) is inherited unchanged: a single pass over the union
of roots is mathematically one traversal of one subgame.

Correctness reduction (Layer-3a gate): with a single root = the game's initial
state and reach = ones, a `SubgameCFR` pass is identical to an engine pass, so
it must reproduce the engine's Leduc number (0.1234 mbb/g). Validated before any
belief-seeded use.

Depth limiting / leaf values compose via the same `leaf_predicate` /
`leaf_value_fn` hook as `DepthLimitedCFR` (re-used here by inheritance order).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pyspiel

from src.rebel.subgame import DepthLimitedCFR

# A root is a (state, initial_reach) pair. initial_reach has length
# num_players+1 (chance last), matching the engine's reach vector convention.
Root = tuple


class SubgameCFR(DepthLimitedCFR):
    """(D)CFR over a subgame defined by belief-seeded root states.

    Parameters
    ----------
    game
        The OpenSpiel game (used for player count + BR-based eval helpers).
    roots
        Sequence of `(pyspiel.State, np.ndarray)` — each root state and the
        initial reach vector to start it with. The state is cloned per pass so
        the caller's states are never mutated.
    leaf_predicate, leaf_value_fn, abstraction_fn, dcfr
        As in `DepthLimitedCFR`.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        roots: Sequence[Root],
        leaf_predicate: Optional[Callable[[pyspiel.State], bool]] = None,
        leaf_value_fn: Optional[Callable[[pyspiel.State], np.ndarray]] = None,
        abstraction_fn=None,
        dcfr=None,
    ) -> None:
        super().__init__(game, abstraction_fn=abstraction_fn, dcfr=dcfr,
                         leaf_predicate=leaf_predicate, leaf_value_fn=leaf_value_fn)
        if not roots:
            raise ValueError("SubgameCFR needs at least one root")
        self.roots = list(roots)
        for st, reach in self.roots:
            if np.asarray(reach).shape != (self.num_players + 1,):
                raise ValueError(
                    f"root reach must have shape ({self.num_players + 1},), "
                    f"got {np.asarray(reach).shape}")
        # Accumulates per-root traverser value for the most recent pass, so the
        # caller can read leaf/subgame values out (used by the oracle).
        self.root_values: list = []

    def _pass(self, traverser, t) -> None:
        self._pass_id += 1
        self._touched = []
        self.leaf_hits = 0
        self.root_values = []
        for state, reach0 in self.roots:
            v = self._cfr(state.clone(), np.asarray(reach0, dtype=np.float64).copy(),
                          traverser)
            self.root_values.append(v)
        self._commit(t)

    def root_value_vector(self, traverser: int, iters: int = 0) -> list:
        """Per-root expected value (under current avg strategy) for `traverser`.

        Computes a single non-updating traversal of every root reading the
        average strategy, returning the per-player utility vector at each root.
        Used to read subgame values without perturbing accumulators.
        """
        out = []
        for state, reach0 in self.roots:
            out.append(self._eval_avg(state.clone(),
                                      np.asarray(reach0, dtype=np.float64).copy()))
        return out

    def _eval_avg(self, state: pyspiel.State, reach: np.ndarray) -> np.ndarray:
        """Expected per-player utility under the average strategy (no updates)."""
        if state.is_terminal():
            return np.asarray(state.returns(), dtype=np.float64)
        if self.leaf_predicate is not None and self.leaf_predicate(state):
            return np.asarray(self.leaf_value_fn(state), dtype=np.float64)
        if state.is_chance_node():
            ev = np.zeros(self.num_players)
            for action, prob in state.chance_outcomes():
                ev += prob * self._eval_avg(state.child(action), reach)
            return ev
        player = state.current_player()
        info_key = state.information_state_string(player)
        avg = self.average_strategy(info_key)
        legal = self._legal(state)
        if avg is None:
            avg = np.full(len(legal), 1.0 / len(legal))
        ev = np.zeros(self.num_players)
        for i, action in enumerate(legal):
            ev += avg[i] * self._eval_avg(state.child(action), reach)
        return ev
