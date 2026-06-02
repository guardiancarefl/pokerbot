"""Public-belief-state (PBS) representation + belief propagation for Leduc.

A PBS is the object ReBeL search and the value net operate on:

    PBS = (public state, joint belief over private states)

For a 2-player game the joint belief factorises into one *range* per player —
a distribution over that player's private holdings consistent with reaching the
public state under the current strategy. (For 6-max NLHE the joint belief is a
range *per opponent seat*; see `docs/REBEL_PBS_6MAX.md`. This module is the
2-player Leduc instance used to validate the search before NLHE.)

Two responsibilities:

  * **Public partition** — map every history/infoset onto a *public key* that
    contains only what all players observe (round, betting, pot, board), with
    the private card stripped. The invariant we validate (Layer 2 of GATE 1):
    `public_key + acting-player's-private-card` is a bijection with the engine's
    infoset set. If that holds, the public tree is a faithful coarsening of the
    game tree and beliefs can be carried on it.

  * **Belief propagation** — given a strategy, compute each player's range at
    every public state: the reach contribution of each private card. This is the
    `reach` the engine already tracks, regrouped by public state — so the leaf
    value function (oracle re-solve in Layer 3, value net in Phase 3) can be
    handed the opponent range it needs.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyspiel

# `[Observer: 0]` and `[Private: 3]` are the only private-bearing fields in
# Leduc's information_state_string; everything else is public.
_PRIV_RE = re.compile(r"\[Observer: \d+\]\[Private: \d+\]")
_PRIVATE_VAL_RE = re.compile(r"\[Private: (\d+)\]")


def public_key(state: pyspiel.State, player: Optional[int] = None) -> str:
    """Public-state key: the acting player's info string with private info removed.

    Both players' info strings at the same public node collapse to the same key
    (they differ only in the stripped `[Observer]`/`[Private]` fields), so the
    key identifies the *public* node irrespective of who is observing.
    """
    if player is None:
        player = state.current_player()
    info = state.information_state_string(player)
    return _PRIV_RE.sub("", info)


def private_of(state: pyspiel.State, player: int) -> int:
    """The private card index held by `player` at `state`."""
    m = _PRIVATE_VAL_RE.search(state.information_state_string(player))
    if m is None:
        raise ValueError(f"no [Private: N] in info string for player {player}")
    return int(m.group(1))


def num_private(game: pyspiel.Game) -> int:
    """Number of distinct private holdings (Leduc: 6 cards)."""
    return len(game.new_initial_state().chance_outcomes())


@dataclass
class PublicBeliefState:
    """A public node plus a per-player range over private holdings.

    `ranges[p][c]` = unnormalized reach of player p holding private card c at
    this public state, under the strategy beliefs were computed with. Unnormalized
    so that mass is conserved across the tree (the normalized range is
    `ranges[p] / ranges[p].sum()` where defined).
    """

    public_key: str
    round: int
    acting_player: int
    ranges: dict  # player -> np.ndarray[num_private]
    # one representative full state (for the leaf value fn / re-solve to anchor on)
    rep_state: Optional[pyspiel.State] = field(default=None, repr=False)


def build_public_tree(game: pyspiel.Game) -> dict:
    """Structural public partition: public_key -> info about the public node.

    Returns `{public_key: {"acting": p, "round": r, "members": {private_card:
    info_state_key}}}`. No strategy involved — pure tree structure.
    """
    tree: dict = {}

    def walk(state: pyspiel.State) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                walk(state.child(action))
            return
        p = state.current_player()
        pk = public_key(state, p)
        priv = private_of(state, p)
        info_key = state.information_state_string(p)
        node = tree.setdefault(
            pk, {"acting": p, "round": _round_of(state, p), "members": {}})
        # The same (public_key, private_card) can be revisited by different deals
        # of the *opponent's* card; they share the acting player's infoset, which
        # is exactly the bijection we want to confirm.
        prev = node["members"].get(priv)
        if prev is not None and prev != info_key:
            raise ValueError(
                f"public_key+private not unique: {pk!r} priv={priv} "
                f"maps to both {prev!r} and {info_key!r}")
        node["members"][priv] = info_key
        for action in state.legal_actions():
            walk(state.child(action))

    walk(game.new_initial_state())
    return tree


def _round_of(state: pyspiel.State, player: int) -> int:
    m = re.search(r"\[Round (\d+)\]", state.information_state_string(player))
    return int(m.group(1)) if m else -1


def compute_ranges(game: pyspiel.Game, policy_fn) -> dict:
    """Per-public-state, per-player range under `policy_fn`.

    `policy_fn(state) -> {action: prob}` is the average strategy (the engine's
    `action_probabilities` is exactly this). We carry the engine's per-player
    reach vector and regroup it by public state: player p's range at a public
    node is the reach mass that player p (via own actions) and chance (via the
    card deals) contribute, summed over the opponent's holdings.

    Returns `{public_key: {player: np.ndarray[num_private]}}` (unnormalized).
    """
    n = num_private(game)
    nump = game.num_players()
    ranges: dict = defaultdict(lambda: {p: np.zeros(n) for p in range(nump)})

    def walk(state: pyspiel.State, reach: np.ndarray) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                nr = reach.copy()
                nr[nump] *= prob
                walk(state.child(action), nr)
            return
        p = state.current_player()
        pk = public_key(state, p)
        # Each player's range entry for THEIR private card = their own reach
        # times chance reach (the deal probability), independent of the
        # opponent's action reach — the PBS factorisation.
        for q in range(nump):
            priv_q = private_of(state, q)
            ranges[pk][q][priv_q] += reach[q] * reach[nump]
        probs = policy_fn(state)
        for action in state.legal_actions():
            nr = reach.copy()
            nr[p] *= probs.get(action, 0.0)
            walk(state.child(action), nr)

    walk(game.new_initial_state(), np.ones(nump + 1))
    return ranges
