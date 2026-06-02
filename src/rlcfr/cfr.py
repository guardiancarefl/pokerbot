"""Tabular (Discounted) CFR engine over an OpenSpiel game tree.

This is the *backbone* of the RL-CFR implementation. RL-CFR (Li, Fang, Huang,
ICML 2024 — arXiv:2403.04344) is a dynamic-action-abstraction layer on top of a
CFR solver; the CFR solver itself is the thing that must converge to a Nash
equilibrium for the whole method to be correct. So we keep this engine simple,
exact, and independently testable.

Design choices, all in service of the Leduc validation gate:

  * **Full-tree vanilla CFR**, not Monte-Carlo sampling. For small games (Leduc:
    a few hundred infosets) a full traversal each iteration is exact and
    deterministic — the cleanest way to certify the engine converges to Nash.
    Sampling would add variance that muddies the "is my engine correct?"
    question the gate is meant to answer.

  * **Discounted CFR (DCFR)** discounting is optional and parameterised. RL-CFR
    uses DCFR(alpha=1.5, beta=0, gamma=2) (Brown & Sandholm 2019). We default to
    those constants but can fall back to vanilla CFR (alpha=beta=gamma absent)
    to cross-check.

  * The engine is **action-abstraction aware**: at each decision node it asks an
    optional `abstraction_fn(state) -> list[action]` for the legal actions to
    expand. For Leduc the abstraction is complete (all legal actions), so the
    engine solves the exact game. For no-limit games this is where RL-CFR's
    dynamically-chosen bet sizes enter. Default = full legal action set.

The average strategy is exposed as a callable compatible with OpenSpiel's
`exploitability.exploitability`, so the same exact-best-response oracle the repo
already uses for Deep CFR validation measures this engine too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pyspiel


AbstractionFn = Callable[[pyspiel.State], list]


@dataclass
class DCFRParams:
    """Regret-update / averaging schedule for the CFR engine.

    `mode` selects the variant:
      * "dcfr"    — Discounted CFR (Brown & Sandholm 2019). On iteration t,
                    accumulated *positive* regret is multiplied by
                    t**alpha/(t**alpha+1), *negative* regret by
                    t**beta/(t**beta+1), and the running average-strategy sum by
                    (t/(t+1))**gamma. RL-CFR uses (1.5, 0, 2).
      * "plus"    — CFR+: regret-matching+ (cumulative regret floored at 0 each
                    update) with linear (weight ∝ t) strategy averaging. Gold
                    standard for Leduc; used to certify the engine plumbing.
      * "linear"  — Linear CFR: discount accumulated regret AND strategy by
                    t/(t+1) each iteration (≡ DCFR(1,1,1)).
      * "vanilla" — plain CFR, no discounting.

    `vanilla=True` is kept as a back-compat shortcut for mode="vanilla".
    """

    alpha: float = 1.5
    beta: float = 0.0
    gamma: float = 2.0
    vanilla: bool = False
    mode: str = "dcfr"

    def resolved_mode(self) -> str:
        return "vanilla" if self.vanilla else self.mode


@dataclass
class _Node:
    """Per-infoset accumulators.

    `regret`/`strat_sum` are the committed cumulative values. The remaining
    fields are per-iteration scratch used to implement *frozen-strategy*
    updates: because an infoset is visited once per history (many times per
    traversal), the regret-matched strategy must be frozen at the first visit of
    each iteration and reused for every later visit that iteration — otherwise
    later visits see a half-updated regret and the solve silently corrupts.
    Deltas are accumulated during the traversal and committed once at the end.
    """

    legal: list  # abstraction action ids at this infoset (stable order)
    regret: np.ndarray  # cumulative (discounted) regret per action
    strat_sum: np.ndarray  # cumulative (discounted) average-strategy weight
    cur_strategy: np.ndarray  # strategy frozen for the current iteration
    regret_delta: np.ndarray  # regret accumulated this iteration (pre-commit)
    strat_delta: np.ndarray  # avg-strategy contribution this iteration
    work_iter: int = 0  # iteration the scratch was last reset for

    @staticmethod
    def make(legal: list) -> "_Node":
        n = len(legal)
        z = lambda: np.zeros(n, dtype=np.float64)
        return _Node(legal=list(legal), regret=z(), strat_sum=z(),
                     cur_strategy=np.full(n, 1.0 / n), regret_delta=z(),
                     strat_delta=z(), work_iter=0)


def regret_matching(regret: np.ndarray) -> np.ndarray:
    """Regret-matching: strategy proportional to positive regret.

    Falls back to uniform when no action has positive regret (standard CFR
    convention).
    """
    pos = np.maximum(regret, 0.0)
    s = pos.sum()
    if s > 0.0:
        return pos / s
    return np.full(len(regret), 1.0 / len(regret))


class TabularCFR:
    """Full-tree (Discounted) CFR solver for a 2-player zero-sum OpenSpiel game.

    Usage:
        solver = TabularCFR(game)
        solver.run(iterations=1000)
        expl = solver.exploitability_mbb()   # uses OpenSpiel exact BR
    """

    def __init__(
        self,
        game: pyspiel.Game,
        abstraction_fn: Optional[AbstractionFn] = None,
        dcfr: Optional[DCFRParams] = None,
    ) -> None:
        if game.num_players() != 2:
            raise ValueError("TabularCFR supports 2-player games only.")
        self.game = game
        self.num_players = game.num_players()
        self.abstraction_fn = abstraction_fn  # None => full legal action set
        self.dcfr = dcfr if dcfr is not None else DCFRParams()
        self.nodes: dict[str, _Node] = {}
        self.iteration = 0
        self._touched: list[_Node] = []
        self._pass_id = 0
        # Alternating (Gauss-Seidel) updates: one pass per player per iteration,
        # each player seeing the other's just-committed regrets. This is what
        # makes CFR+/DCFR converge at ~1/T (matching OpenSpiel) rather than the
        # ~1/sqrt(T) of simultaneous updates. Default on.
        self.alternating = True

    # -- abstraction --------------------------------------------------------
    def _legal(self, state: pyspiel.State) -> list:
        if self.abstraction_fn is None:
            return state.legal_actions()
        legal = self.abstraction_fn(state)
        if not legal:
            # An abstraction must never empty a decision node; that would make
            # the node unreachable and silently corrupt the solve. Fail loud.
            raise ValueError("abstraction_fn returned no actions at a decision node")
        return legal

    def _node(self, info_key: str, legal: list) -> _Node:
        node = self.nodes.get(info_key)
        if node is None:
            node = _Node.make(legal)
            self.nodes[info_key] = node
        return node

    # -- discounting --------------------------------------------------------
    def _discounts(self, t: int) -> tuple[float, float, float]:
        """Return (pos_regret_mult, neg_regret_mult, strat_sum_mult) for iter t.

        These multiply the *accumulated* values before this iteration's
        contribution is added.
        """
        mode = self.dcfr.resolved_mode()
        if mode == "vanilla":
            return 1.0, 1.0, 1.0
        if mode == "linear":  # DCFR(1,1,1): weight everything by t
            d = t / (t + 1.0)
            return d, d, d
        if mode == "dcfr":
            a, b, g = self.dcfr.alpha, self.dcfr.beta, self.dcfr.gamma
            pos = (t ** a) / (t ** a + 1.0)
            neg = (t ** b) / (t ** b + 1.0)
            strat = (t / (t + 1.0)) ** g
            return pos, neg, strat
        raise ValueError(f"unknown CFR mode {mode!r}")

    # -- core recursion -----------------------------------------------------
    def _freeze(self, node: _Node) -> None:
        """On the first visit of the current pass, snapshot the node's strategy
        and reset its delta buffers. Idempotent within a pass.

        Cumulative regret is mutated only at commit (end of pass), so the
        snapshot equals regret_matching(node.regret) on every later visit this
        pass too — this keeps an infoset reached by many histories consistent.
        """
        if node.work_iter != self._pass_id:
            node.cur_strategy = regret_matching(node.regret)
            node.regret_delta = np.zeros_like(node.regret)
            node.strat_delta = np.zeros_like(node.strat_sum)
            node.work_iter = self._pass_id
            self._touched.append(node)

    def _cfr(self, state: pyspiel.State, reach: np.ndarray,
             traverser: Optional[int]) -> np.ndarray:
        """Return the expected utility vector (per player) at `state`.

        `traverser` selects whose regrets are accumulated this pass:
          * None  -- simultaneous: accumulate for every acting player.
          * int p -- alternating: accumulate only for player p; other players
            play their current regret-matched strategy (stable during the pass).

        `reach` holds reach probabilities [pi_player0, pi_player1, pi_chance].
        """
        if state.is_terminal():
            return np.asarray(state.returns(), dtype=np.float64)

        if state.is_chance_node():
            ev = np.zeros(self.num_players, dtype=np.float64)
            for action, prob in state.chance_outcomes():
                child = state.child(action)
                new_reach = reach.copy()
                new_reach[self.num_players] *= prob  # index num_players == chance
                ev += prob * self._cfr(child, new_reach, traverser)
            return ev

        player = state.current_player()
        info_key = state.information_state_string(player)
        legal = self._legal(state)
        node = self._node(info_key, legal)
        updating = traverser is None or player == traverser
        if updating:
            self._freeze(node)
            strategy = node.cur_strategy
        else:
            # Non-updating player: regret is stable this pass, so this is the
            # consistent frozen strategy without needing the delta machinery.
            strategy = regret_matching(node.regret)

        action_util = np.zeros((len(legal), self.num_players), dtype=np.float64)
        node_util = np.zeros(self.num_players, dtype=np.float64)
        for i, action in enumerate(legal):
            child = state.child(action)
            new_reach = reach.copy()
            new_reach[player] *= strategy[i]
            u = self._cfr(child, new_reach, traverser)
            action_util[i] = u
            node_util += strategy[i] * u

        if updating:
            # Counterfactual reach = product of everyone else's reach (+ chance).
            cf_reach = 1.0
            for p in range(self.num_players + 1):
                if p != player:
                    cf_reach *= reach[p]
            node.regret_delta += cf_reach * (action_util[:, player] - node_util[player])
            node.strat_delta += reach[player] * strategy

        return node_util

    def _commit(self, t: int) -> None:
        """Apply this pass's accumulated deltas to all touched nodes."""
        mode = self.dcfr.resolved_mode()
        if mode == "plus":
            for node in self._touched:
                node.regret = np.maximum(node.regret + node.regret_delta, 0.0)
                node.strat_sum = node.strat_sum + t * node.strat_delta
        else:
            pos_d, neg_d, strat_d = self._discounts(t)
            for node in self._touched:
                discounted = np.where(node.regret > 0,
                                      node.regret * pos_d, node.regret * neg_d)
                node.regret = discounted + node.regret_delta
                node.strat_sum = node.strat_sum * strat_d + node.strat_delta
        self._touched = []

    def _pass(self, traverser: Optional[int], t: int) -> None:
        self._pass_id += 1
        self._touched = []
        reach = np.ones(self.num_players + 1, dtype=np.float64)
        self._cfr(self.game.new_initial_state(), reach, traverser)
        self._commit(t)

    def run(self, iterations: int, log_every: int = 0,
            eval_fn: Optional[Callable[[int], None]] = None) -> None:
        """Run `iterations` CFR iterations.

        With `alternating` (default) each iteration runs one commit pass per
        player in turn (Gauss-Seidel), so each player sees the other's updated
        regrets -- the ~1/T regime that matches OpenSpiel CFR+. Otherwise a
        single simultaneous pass per iteration (~1/sqrt(T)). Either way the
        frozen-strategy / deferred-commit discipline keeps multi-history
        infosets consistent.
        """
        for _ in range(iterations):
            self.iteration += 1
            t = self.iteration
            if self.alternating:
                for traverser in range(self.num_players):
                    self._pass(traverser, t)
            else:
                self._pass(None, t)
            if eval_fn is not None and log_every and t % log_every == 0:
                eval_fn(t)

    # -- strategy extraction ------------------------------------------------
    def average_strategy(self, info_key: str) -> Optional[np.ndarray]:
        node = self.nodes.get(info_key)
        if node is None:
            return None
        s = node.strat_sum.sum()
        if s > 0.0:
            return node.strat_sum / s
        return np.full(len(node.legal), 1.0 / len(node.legal))

    def action_probabilities(self, state: pyspiel.State) -> dict:
        """Average-strategy policy as dict{action: prob} over abstraction legal.

        Compatible with OpenSpiel's `tabular_policy_from_callable`.
        """
        player = state.current_player()
        info_key = state.information_state_string(player)
        node = self.nodes.get(info_key)
        if node is None:
            legal = self._legal(state)
            return {a: 1.0 / len(legal) for a in legal}
        avg = self.average_strategy(info_key)
        return {a: float(p) for a, p in zip(node.legal, avg)}

    # -- evaluation ---------------------------------------------------------
    def exploitability_mbb(self, chips_to_mbb: float = 500.0) -> float:
        """Exact Nash exploitability via OpenSpiel best response, in mbb/g.

        The 500x factor matches the repo's Leduc convention (big blind = 2
        chips). Only meaningful when the abstraction is complete (e.g. Leduc),
        because OpenSpiel's BR explores the *full* legal action set — if the
        abstraction drops legal actions, this measures exploitability of the
        abstracted strategy within the full game, which is still the right
        thing for the gate but should be interpreted accordingly.
        """
        from open_spiel.python import policy as policy_lib
        from open_spiel.python.algorithms import exploitability

        tabular = policy_lib.tabular_policy_from_callable(
            self.game, self.action_probabilities)
        expl_chips = exploitability.exploitability(self.game, tabular)
        return float(expl_chips) * chips_to_mbb
