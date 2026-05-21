"""Exploitability evaluation for Leduc Deep CFR policies.

Exploitability is the gap between a policy and Nash equilibrium: the amount
a best-response opponent could win against this policy, averaged over both
players. Zero exploitability = Nash. For Leduc, published Nash exploitability
is well below 0.01 chips/game; Deep CFR with sufficient iterations gets close.

We compute exploitability via OpenSpiel's `exploitability.exploitability()`
which does a full game-tree traversal — tractable for Leduc, infeasible for
NLHE. This module is Leduc-specific; Phase 2 will need a different approach
(local best response, approximate exploitability, etc.).
"""

from __future__ import annotations

from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability
import pyspiel

# Conversion factor: Leduc utilities are in chips. One big blind = 2 chips
# (small blind 1, big blind 2 in OpenSpiel's leduc_poker). So a unit of
# chip-utility-per-game is 500 milli-big-blinds per game (mbb/g).
LEDUC_CHIPS_TO_MBB = 500.0


def exploitability_mbb(game: pyspiel.Game, action_probabilities_fn) -> float:
    """Compute Nash exploitability of a callable policy.

    Args:
        game: the OpenSpiel game (must be leduc_poker for the unit conversion
            to be correct).
        action_probabilities_fn: callable(state) -> dict[action -> prob],
            typically `solver.action_probabilities`.

    Returns:
        Exploitability in milli-big-blinds per game.
    """
    tabular = policy_lib.tabular_policy_from_callable(game, action_probabilities_fn)
    expl_chips = exploitability.exploitability(game, tabular)
    return expl_chips * LEDUC_CHIPS_TO_MBB
