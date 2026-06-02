"""Oracle leaf values + depth-limited re-solving — the substance of GATE 1.

The question GATE 1 answers: *does depth-limited subgame solving, given a correct
value function at the leaves, produce near-optimal play?* If the search logic is
wrong this fails on Leduc (where we know the exact answer), and everything
downstream — value net, NLHE — is invalid. So this is a HARD gate.

We isolate the search from the value net by handing the search a *known-correct*
value function:

  Layer 3b (the gate core) — "correct value function ⇒ optimal play"
    1. Solve full Leduc to Nash with the validated engine.
    2. Extract the exact Nash continuation value at every round-2-entry world
       state (the value of that subtree under the Nash average strategy). This
       is, by definition, the perfect leaf value function.
    3. Run a depth-limited solve of round 1 that cuts at round-2 entry and reads
       those leaf values, then reassemble the full policy (depth-limited round-1
       strategy + Nash round-2 strategy).
    4. Measure exploitability with OpenSpiel's exact best response. It must
       return to ≈ Nash. If depth-limited solving + leaf substitution is correct,
       feeding it the true continuation values reconstructs optimal play.

  Layer 4 (faithful re-solving) — belief-conditioned oracle in a ReBeL loop:
    instead of Nash leaves, re-solve the round-2 subgame for the *current*
    beliefs (PBS-rooted `SubgameCFR`) and iterate. Confirms the full re-solving
    agent — not just the leaf plumbing — is near-optimal.

Both report exploitability in the engine's mbb/g convention (x500), comparable to
the engine's own 0.1234 Leduc number.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np
import pyspiel

from src.rlcfr.cfr import DCFRParams, TabularCFR
from src.rebel.subgame import DepthLimitedCFR

_ROUND_RE = re.compile(r"\[Round (\d+)\]")
_ROUND2_BETS_RE = re.compile(r"\[Round2: ([^\]]*)\]")


def is_round2_entry(state: pyspiel.State) -> bool:
    """True at the first decision node of round 2 (board dealt, no round-2 bets).

    This is the depth-limit cut: round-1 betting (board-blind) and the board
    chance deal are inside the subgame; round-2 play is replaced by a leaf value.
    """
    if state.is_terminal() or state.is_chance_node():
        return False
    info = state.information_state_string(state.current_player())
    rnd = _ROUND_RE.search(info)
    if rnd is None or int(rnd.group(1)) != 2:
        return False
    bets = _ROUND2_BETS_RE.search(info)
    # First round-2 decision: no round-2 action has been taken yet.
    return bets is not None and bets.group(1).strip() == ""


def avg_strategy_subtree_value(game: pyspiel.Game, solver: TabularCFR) -> dict:
    """Per round-2-entry world state, the value vector under the avg strategy.

    Keyed by `state.history_str()` (the full action path — unique per world
    state, and identical between the full game and any subgame that reaches it).
    This is the exact continuation value when `solver` is at Nash.
    """
    out: dict = {}

    def probs(state):
        d = solver.action_probabilities(state)
        return d

    def walk(state: pyspiel.State) -> np.ndarray:
        if state.is_terminal():
            return np.asarray(state.returns(), dtype=np.float64)
        if state.is_chance_node():
            ev = np.zeros(game.num_players())
            for a, p in state.chance_outcomes():
                ev += p * walk(state.child(a))
            return ev
        if is_round2_entry(state):
            v = _value_under(game, solver, state)
            out[state.history_str()] = v
            return v
        ev = np.zeros(game.num_players())
        pr = probs(state)
        for a in state.legal_actions():
            ev += pr.get(a, 0.0) * walk(state.child(a))
        return ev

    walk(game.new_initial_state())
    return out


def _value_under(game, solver, state) -> np.ndarray:
    """Expected value of `state`'s subtree under solver's average strategy."""
    if state.is_terminal():
        return np.asarray(state.returns(), dtype=np.float64)
    if state.is_chance_node():
        ev = np.zeros(game.num_players())
        for a, p in state.chance_outcomes():
            ev += p * _value_under(game, solver, state.child(a))
        return ev
    pr = solver.action_probabilities(state)
    ev = np.zeros(game.num_players())
    for a in state.legal_actions():
        ev += pr.get(a, 0.0) * _value_under(game, solver, state.child(a))
    return ev


def round1_strategy_with_leaves(
    game: pyspiel.Game,
    leaf_value_fn: Callable[[pyspiel.State], np.ndarray],
    dcfr: Optional[DCFRParams] = None,
    iters: int = 1000,
) -> dict:
    """Depth-limited solve of round 1 (cut at round-2 entry); return strategy dict.

    Returns `{info_state_string: {action: prob}}` for every round-1 infoset, read
    off the depth-limited solver's average strategy.
    """
    dl = DepthLimitedCFR(
        game, dcfr=dcfr or DCFRParams(mode="plus"),
        leaf_predicate=is_round2_entry, leaf_value_fn=leaf_value_fn)
    dl.run(iters)
    out = {}
    for info_key, node in dl.nodes.items():
        avg = dl.average_strategy(info_key)
        out[info_key] = {int(a): float(p) for a, p in zip(node.legal, avg)}
    return out, dl


def combined_tabular_policy(game: pyspiel.Game, policy_map: dict):
    """Wrap `{info_state_string: {action: prob}}` as an OpenSpiel TabularPolicy."""
    from open_spiel.python import policy as policy_lib

    def callable_policy(state):
        key = state.information_state_string(state.current_player())
        if key in policy_map:
            return policy_map[key]
        la = state.legal_actions()
        return {a: 1.0 / len(la) for a in la}

    return policy_lib.tabular_policy_from_callable(game, callable_policy)


def exploitability_mbb(game, policy_map, chips_to_mbb: float = 500.0) -> float:
    from open_spiel.python.algorithms import exploitability as expl_lib
    tab = combined_tabular_policy(game, policy_map)
    return float(expl_lib.exploitability(game, tab)) * chips_to_mbb


# ---------------------------------------------------------------------------
# Layer 4 — faithful belief-conditioned re-solving (the GATE).
#
# Fixed leaf values (Layer 3b) are UNSAFE: pinning round-2 to a static value lets
# the trunk strategy become exploitable, because the opponent can no longer
# best-respond in round 2. The fix — and the substance of ReBeL — is to re-solve
# the round-2 subgame for the *current belief* each outer iteration, so the
# opponent's round-2 play adapts to the trunk-induced range. As beliefs converge
# this is a safe refinement and the full policy approaches Nash.
# ---------------------------------------------------------------------------


def _uniform_policy_fn(state):
    la = state.legal_actions()
    return {a: 1.0 / len(la) for a in la}


def compute_round1_beliefs(game: pyspiel.Game, round1_policy_fn) -> dict:
    """Per round-1 betting context, each player's range over private cards.

    Returns `{r1_action_tuple: {player: np.ndarray[num_private]}}` where the
    range is the player's own reach (product of its round-1 action probs) for
    each private card — the belief entering round 2. Deal priors are uniform and
    absorbed into the root enumeration, so only own-action reach is tracked.
    """
    n = len(game.new_initial_state().chance_outcomes())
    nump = game.num_players()
    beliefs: dict = {}

    def walk(state, own_reach, r1_actions):
        if state.is_terminal():
            return
        if state.is_chance_node():
            # Private deals (round 1) vs the public-card deal (round 1->2 cut).
            child0 = state.child(state.chance_outcomes()[0][0])
            entering_round2 = (not child0.is_chance_node()
                               and not child0.is_terminal()
                               and "[Round 2]" in child0.information_state_string(
                                   child0.current_player()))
            if entering_round2:
                # Record beliefs for this betting context, keyed by r1 actions.
                hist = state.history()
                cards = hist[:nump]  # p0, p1 private deals
                key = tuple(r1_actions)
                slot = beliefs.setdefault(
                    key, {p: np.zeros(n) for p in range(nump)})
                for p in range(nump):
                    slot[p][cards[p]] = own_reach[p]
                return
            for a, _ in state.chance_outcomes():
                walk(state.child(a), own_reach, r1_actions)
            return
        p = state.current_player()
        probs = round1_policy_fn(state)
        for a in state.legal_actions():
            nr = own_reach.copy()
            nr[p] *= probs.get(a, 0.0)
            walk(state.child(a), nr, r1_actions + [a])

    walk(game.new_initial_state(), np.ones(nump), [])
    return beliefs


def resolve_round2(game: pyspiel.Game, beliefs: dict,
                   dcfr: Optional[DCFRParams] = None, iters: int = 400):
    """Re-solve every round-2 subgame for the given beliefs.

    Returns `(leaf_values, round2_map)`:
      * `leaf_values[history_str]` — per-player value at each round-2-entry world
        state under the belief-conditioned round-2 equilibrium (the trunk's leaf).
      * `round2_map[info_state_string]` — the re-solved round-2 average strategy.
    """
    from src.rebel.subgame_solver import SubgameCFR

    nump = game.num_players()
    # Collect round-2-entry states, grouped by public node (r1 actions + board).
    groups: dict = {}

    def collect(state):
        if state.is_terminal():
            return
        if is_round2_entry(state):
            hist = state.history()
            gkey = tuple(hist[nump:])  # r1 actions + board card
            groups.setdefault(gkey, []).append(state.clone())
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                collect(state.child(a))
            return
        for a in state.legal_actions():
            collect(state.child(a))

    collect(game.new_initial_state())

    leaf_values: dict = {}
    round2_map: dict = {}
    for gkey, states in groups.items():
        r1_key = gkey[:-1]  # drop the board card -> round-1 betting context
        belief = beliefs.get(r1_key)
        roots = []
        for st in states:
            hist = st.history()
            cards = hist[:nump]
            reach = np.ones(nump + 1)
            if belief is not None:
                for p in range(nump):
                    reach[p] = belief[p][cards[p]]
            roots.append((st, reach))
        sub = SubgameCFR(game, roots=roots, dcfr=dcfr or DCFRParams(mode="plus"))
        sub.run(iters)
        # Round-2 strategy for every infoset touched in this subgame.
        for info_key, node in sub.nodes.items():
            avg = sub.average_strategy(info_key)
            round2_map[info_key] = {int(a): float(p)
                                    for a, p in zip(node.legal, avg)}
        # Leaf value = value at each root under the re-solved strategy.
        for st, _reach in roots:
            leaf_values[st.history_str()] = sub._eval_avg(
                st.clone(), np.ones(nump + 1))
    return leaf_values, round2_map


class Round2Resolver:
    """Persistent round-2 subgame re-solver: structure precomputed once.

    The expensive parts of `resolve_round2` — walking the full tree to find every
    round-2-entry state, cloning them, grouping by public node — depend only on
    the game, not the belief. We do them once here. Each `step(beliefs, k)` then
    only re-seeds the belief reach and runs `k` more CFR iterations on persistent
    per-group solvers (warm start), which is what makes per-trunk-iteration
    re-solving affordable. Same operation NLHE needs, so this is reusable.
    """

    def __init__(self, game: pyspiel.Game, dcfr: Optional[DCFRParams] = None):
        from src.rebel.subgame_solver import SubgameCFR
        self.game = game
        self.nump = game.num_players()
        self.dcfr = dcfr or DCFRParams(mode="plus")
        self._SubgameCFR = SubgameCFR
        self.groups: dict = {}      # gkey -> list of (state_clone, cards_tuple)
        self.solvers: dict = {}     # gkey -> SubgameCFR (persistent)
        self._collect()

    def _collect(self):
        nump = self.nump

        def walk(state):
            if state.is_terminal():
                return
            if is_round2_entry(state):
                hist = state.history()
                gkey = tuple(hist[nump:])        # r1 actions + board
                cards = tuple(hist[:nump])       # (p0 card, p1 card)
                self.groups.setdefault(gkey, []).append((state.clone(), cards))
                return
            if state.is_chance_node():
                for a, _ in state.chance_outcomes():
                    walk(state.child(a))
                return
            for a in state.legal_actions():
                walk(state.child(a))

        walk(self.game.new_initial_state())

    def step(self, beliefs: dict, k: int = 8):
        nump = self.nump
        leaf_values: dict = {}
        round2_map: dict = {}
        for gkey, entries in self.groups.items():
            r1_key = gkey[:-1]
            belief = beliefs.get(r1_key)
            roots = []
            for st, cards in entries:
                reach = np.ones(nump + 1)
                if belief is not None:
                    for p in range(nump):
                        reach[p] = belief[p][cards[p]]
                roots.append((st, reach))
            solver = self.solvers.get(gkey)
            if solver is None:
                solver = self._SubgameCFR(self.game, roots=roots, dcfr=self.dcfr)
                self.solvers[gkey] = solver
            else:
                solver.roots = roots  # re-seed belief reach; states unchanged
            solver.run(k)
            for info_key, node in solver.nodes.items():
                avg = solver.average_strategy(info_key)
                round2_map[info_key] = {int(a): float(p)
                                        for a, p in zip(node.legal, avg)}
            for st, _cards in entries:
                leaf_values[st.history_str()] = solver._eval_avg(
                    st.clone(), np.ones(nump + 1))
        return leaf_values, round2_map


def rebel_resolve(game: pyspiel.Game, iters: int = 400,
                  round2_iters: int = 200, dcfr: Optional[DCFRParams] = None,
                  eval_every: int = 0, verbose: bool = True):
    """Faithful ReBeL re-solving: leaf values recomputed every trunk iteration.

    One persistent trunk solver. Each iteration:
      1. beliefs <- trunk's running-average strategy at the round-2 leaves,
      2. leaf values <- re-solve round-2 for those beliefs (opponent adapts),
      3. one trunk CFR iteration against those leaf values.
    The leaf payoffs are thus functions of the running average (ReBeL's
    convergence regime), so the trunk average approaches a safe Nash trunk.
    Returns the assembled (round-1 trunk + round-2 re-solved) policy map.
    """
    dcfr = dcfr or DCFRParams(mode="plus")

    # Persistent trunk solver; leaf_value_fn is swapped in each iteration via a
    # mutable closure cell so the accumulators (regret/avg-strategy) persist.
    cell = {"lv": {}}

    def leaf_fn(state):
        return cell["lv"][state.history_str()]

    dl = DepthLimitedCFR(game, dcfr=dcfr, leaf_predicate=is_round2_entry,
                         leaf_value_fn=leaf_fn)
    resolver = Round2Resolver(game, dcfr=dcfr)

    def trunk_policy_fn(state):
        key = state.information_state_string(state.current_player())
        node = dl.nodes.get(key)
        if node is None:
            return _uniform_policy_fn(state)
        avg = dl.average_strategy(key)
        return {int(a): float(p) for a, p in zip(node.legal, avg)}

    round2_map = {}
    for t in range(iters):
        beliefs = compute_round1_beliefs(game, trunk_policy_fn)
        cell["lv"], round2_map = resolver.step(beliefs, k=round2_iters)
        dl.run(1)  # one alternating CFR iteration against current leaf values
        if verbose and eval_every and (t + 1) % eval_every == 0:
            combined = dict(round2_map)
            combined.update(_trunk_map(dl))
            e = exploitability_mbb(game, combined)
            print(f"    iter {t+1}/{iters}: exploitability = {e:.5f} mbb/g")

    combined = dict(round2_map)
    combined.update(_trunk_map(dl))
    return combined


def _trunk_map(dl) -> dict:
    out = {}
    for info_key, node in dl.nodes.items():
        avg = dl.average_strategy(info_key)
        out[info_key] = {int(a): float(p) for a, p in zip(node.legal, avg)}
    return out
