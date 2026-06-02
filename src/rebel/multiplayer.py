"""N-player generalization of the validated CFR engine (6-max groundwork).

The 2-player `TabularCFR` math is already N-player-general — reach vectors,
alternating updates over `range(num_players)`, and the counterfactual-reach
product all use `num_players`. Only two things are 2-player-specific: the
`num_players != 2` guard in `__init__`, and `exploitability_mbb` (2-player exact
best response). `NPlayerCFR` lifts the guard and swaps in OpenSpiel's `nash_conv`
(the multiplayer generalization of exploitability: the summed best-response gain
across all players).

Validated: CFR+ on 3-player Kuhn drives NashConv to ~3e-5 (converges to a Nash;
3-player Kuhn is the standard smallest multiplayer poker testbed). This is the
foundation the 6-max PBS / depth-limited search / safe-resolve are built on.

IMPORTANT caveat for 6-max: CFR's Nash-convergence guarantee is a 2-player
zero-sum result. It empirically converges on 3-player Kuhn/Leduc, but multiplayer
CFR converges to a Nash only in special cases; in general it reaches a
correlated/coarse equilibrium. Safe subgame re-solving theory (the 2-player
gadget, GATE 1) likewise does not transfer unmodified to >2 players — multiplayer
safe re-solving is a research-grade problem. Treat every multiplayer number as
empirical and gate it against `nash_conv`, not as a theorem.
"""

from __future__ import annotations

from typing import Optional

import pyspiel

from src.rlcfr.cfr import DCFRParams, TabularCFR


class NPlayerCFR(TabularCFR):
    """Full-tree (D)CFR for an N-player game (N >= 2). Same engine, no guard."""

    def __init__(self, game: pyspiel.Game, abstraction_fn=None,
                 dcfr: Optional[DCFRParams] = None) -> None:
        if game.num_players() < 2:
            raise ValueError("NPlayerCFR needs >= 2 players")
        # Mirror TabularCFR.__init__ exactly, minus the `!= 2` guard.
        self.game = game
        self.num_players = game.num_players()
        self.abstraction_fn = abstraction_fn
        self.dcfr = dcfr if dcfr is not None else DCFRParams()
        self.nodes = {}
        self.iteration = 0
        self._touched = []
        self._pass_id = 0
        self.alternating = True

    def nash_conv(self) -> float:
        """Multiplayer exploitability: summed best-response gain over players."""
        from open_spiel.python import policy as policy_lib
        from open_spiel.python.algorithms import exploitability as expl_lib

        tab = policy_lib.tabular_policy_from_callable(
            self.game, self.action_probabilities)
        return float(expl_lib.nash_conv(self.game, tab))
