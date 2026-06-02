"""CFR-D re-solving gadget — safe subgame solving (the Gate-1 fix).

Isolated subgame re-solving is UNSAFE: a strategy solved only for the blueprint
range is exploitable by an opponent who deviates to arrive with a different
range (validated on Leduc: 54.6 mbb/g — see docs/REBEL_GATE1_FINDINGS.md). The
re-solving gadget (Burch/Johanson/Bowling, CFR-D 2014; Moravcik et al.,
DeepStack 2017) fixes this by giving each player, at the subgame root, a per-hand
choice:

  * FOLLOW   — enter the subgame and play it out, or
  * TERMINATE — opt out and take a fixed value `w*(hand)` = the player's
    counterfactual value under the *blueprint* (what the hand could guarantee
    without being re-solved against).

A player only enters with hands whose subgame value beats their opt-out, so the
re-solved strategy cannot give the opponent more than their blueprint value for
*any* hand → safe against range deviation.

Implementation reuses the validated `SubgameCFR` as the inner range-conditioned
solver (proven byte-exact). The gadget is an OUTER loop that, each round,
(i) re-solves the subgame for the current ranges, (ii) reads each player's
per-hand counterfactual value, (iii) updates each player's per-hand FOLLOW
probability by regret-matching FOLLOW (subgame CFV) vs TERMINATE (opt-out),
(iv) rescales ranges by the new follow probabilities. At convergence both
players' subgame strategies are safe; combined with the blueprint trunk they
must reach ≈ Nash on Leduc (the gate).
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pyspiel

from src.rlcfr.cfr import DCFRParams, TabularCFR
from src.rebel import oracle
from src.rebel.subgame_solver import SubgameCFR

_ROUND_RE = re.compile(r"\[Round (\d+)\]")


def blueprint_round2(game: pyspiel.Game, blueprint: TabularCFR):
    """From a blueprint solve, extract per round-2 public node:
      - reach ranges r_p[hand] (blueprint reach into the node, per player),
      - opt-out CFVs w_p[hand] (blueprint counterfactual value per player/hand).
    Keyed by the public node (r1 actions + board). Returns (ranges, optouts,
    groups) where groups[gkey] = list of (entry_state, (h0,h1)).
    """
    nump = game.num_players()
    groups: dict = {}
    # reach[gkey][player][hand]  and  value sums for CFV
    reach: dict = {}
    # cfv_num[gkey][player][hand] = sum_{opp} reach_opp * u_player(hand,opp)
    cfv_num: dict = {}

    def walk(state, rch):
        if state.is_terminal():
            return
        if oracle.is_round2_entry(state):
            hist = state.history()
            gkey = tuple(hist[nump:])
            cards = tuple(hist[:nump])
            groups.setdefault(gkey, []).append((state.clone(), cards))
            rslot = reach.setdefault(
                gkey, {p: {} for p in range(nump)})
            for p in range(nump):
                rslot[p][cards[p]] = rch[p]
            # value of this entry under blueprint continuation, per player
            v = _value_under_blueprint(game, blueprint, state)
            cslot = cfv_num.setdefault(gkey, {p: {} for p in range(nump)})
            # counterfactual: player p's value for its hand weighted by opp reach
            for p in range(nump):
                opp = 1 - p
                cslot[p][cards[p]] = cslot[p].get(cards[p], 0.0) + rch[opp] * v[p]
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                walk(state.child(a), rch)
            return
        p = state.current_player()
        probs = blueprint.action_probabilities(state)
        for a in state.legal_actions():
            nr = rch.copy()
            nr[p] *= probs.get(a, 0.0)
            walk(state.child(a), nr)

    walk(game.new_initial_state(), np.ones(nump + 1))

    ranges, optouts = {}, {}
    n = len(game.new_initial_state().chance_outcomes())
    for gkey in groups:
        ranges[gkey] = {p: np.zeros(n) for p in range(nump)}
        optouts[gkey] = {p: np.zeros(n) for p in range(nump)}
        for p in range(nump):
            for h, r in reach[gkey][p].items():
                ranges[gkey][p][h] = r
            for h, cfv in cfv_num[gkey][p].items():
                optouts[gkey][p][h] = cfv  # blueprint CFV (counterfactual)
    return ranges, optouts, groups


def _value_under_blueprint(game, blueprint, state):
    if state.is_terminal():
        return np.asarray(state.returns(), dtype=np.float64)
    if state.is_chance_node():
        ev = np.zeros(game.num_players())
        for a, p in state.chance_outcomes():
            ev += p * _value_under_blueprint(game, blueprint, state.child(a))
        return ev
    pr = blueprint.action_probabilities(state)
    ev = np.zeros(game.num_players())
    for a in state.legal_actions():
        ev += pr.get(a, 0.0) * _value_under_blueprint(game, blueprint,
                                                       state.child(a))
    return ev


def _regret_match2(regret):
    pos = np.maximum(regret, 0.0)
    s = pos.sum()
    return pos / s if s > 0 else np.array([0.5, 0.5])


def _infoset_player(info_key):
    return None  # placeholder (unused)


def gadget_resolve_oneside(game, entries, resolver_player, r_fixed, r_opp_prior,
                           w_opp, iters: int = 1000,
                           dcfr: Optional[DCFRParams] = None):
    """One-sided CFR-D gadget as a SINGLE integrated CFR (the correct form).

    Augmented game (re-solver R fixed range, opponent O gadgeted): O, knowing its
    hand h, chooses FOLLOW (play the subgame) or TERMINATE (take opt-out
    `w_opp[h]`, counterfactual-scaled). FOLLOW and the subgame are solved in ONE
    CFR: a persistent accumulating subgame solver, and the gadget FOLLOW/TERMINATE
    regrets updated EVERY subgame iteration in lockstep (RM+). O's reach into the
    subgame each iteration = `r_opp_prior[h] * follow_t[h]`, so the average
    strategy reflects the gadget-controlled range — exactly the augmented tree.

    Returns `(strategy_map_for_R, value_per_R_hand)`.
    """
    R = resolver_player
    O = 1 - R
    nump = game.num_players()
    dcfr = dcfr or DCFRParams(mode="plus")
    opp_hands = sorted({c[O] for _, c in entries})
    r_hands = sorted({c[R] for _, c in entries})
    g_reg = {h: np.zeros(2) for h in opp_hands}        # [FOLLOW, TERMINATE], RM+
    g_strat_sum = {h: np.zeros(2) for h in opp_hands}  # linear-averaged

    solver = SubgameCFR(game, roots=[(st, np.ones(nump + 1)) for st, _ in entries],
                        dcfr=dcfr)  # persistent / accumulating
    for t in range(1, iters + 1):
        follow = {h: _regret_match2(g_reg[h])[0] for h in opp_hands}
        roots = []
        for st, c in entries:
            reach = np.ones(nump + 1)
            reach[R] = r_fixed[c[R]]
            reach[O] = r_opp_prior[c[O]] * follow[c[O]]
            roots.append((st, reach))
        solver.roots = roots
        solver.run(1)  # one accumulating subgame iteration against current range
        # O's subgame counterfactual value per hand under current avg strategy
        vals = {c: solver._eval_avg(st.clone(), np.ones(nump + 1))
                for st, c in entries}
        cfv_O = {h: 0.0 for h in opp_hands}
        for c, v in vals.items():
            cfv_O[c[O]] += r_fixed[c[R]] * v[O]
        # gadget RM+ update + linear strategy averaging
        for h in opp_hands:
            s = _regret_match2(g_reg[h])
            node_v = s[0] * cfv_O[h] + s[1] * w_opp[h]
            g_reg[h][0] = max(g_reg[h][0] + cfv_O[h] - node_v, 0.0)
            g_reg[h][1] = max(g_reg[h][1] + w_opp[h] - node_v, 0.0)
            g_strat_sum[h] += t * s

    # R's safe strategy = subgame average at R's infosets
    out_map = {}
    for info_key, node in solver.nodes.items():
        m = _ROUND_RE.search(info_key)
        if m and int(m.group(1)) == 2 and info_key.startswith(
                "[Observer: %d]" % R):
            avg = solver.average_strategy(info_key)
            out_map[info_key] = {int(a): float(p)
                                 for a, p in zip(node.legal, avg)}
    # leaf value to R per hand: weight by O's final gadget range
    follow_avg = {h: (g_strat_sum[h][0] / g_strat_sum[h].sum()
                      if g_strat_sum[h].sum() > 0 else 1.0) for h in opp_hands}
    finalvals = {c: solver._eval_avg(st.clone(), np.ones(nump + 1))
                 for st, c in entries}
    valR = {h: 0.0 for h in r_hands}
    for c, v in finalvals.items():
        valR[c[R]] += r_opp_prior[c[O]] * follow_avg[c[O]] * v[R]
    return out_map, valR


def gadget_resolve_all(game, blueprint, iters: int = 1000,
                       dcfr: Optional[DCFRParams] = None):
    """Safe re-solve of every round-2 node via the one-sided gadget, run once per
    player. Returns (round2_map, leaf_values)."""
    ranges, optouts, groups = blueprint_round2(game, blueprint)
    round2_map, leaf_values = {}, {}
    nump = game.num_players()
    for gkey, entries in groups.items():
        r0 = {h: ranges[gkey][0][h] for h in {c[0] for _, c in entries}}
        r1 = {h: ranges[gkey][1][h] for h in {c[1] for _, c in entries}}
        w0 = {h: optouts[gkey][0][h] for h in r0}
        w1 = {h: optouts[gkey][1][h] for h in r1}
        m0, val0 = gadget_resolve_oneside(game, entries, 0, r0, r1, w1,
                                          iters=iters, dcfr=dcfr)
        m1, val1 = gadget_resolve_oneside(game, entries, 1, r1, r0, w0,
                                          iters=iters, dcfr=dcfr)
        round2_map.update(m0)
        round2_map.update(m1)
        for st, c in entries:
            leaf_values[st.history_str()] = np.array(
                [val0.get(c[0], 0.0), val1.get(c[1], 0.0)])
    return round2_map, leaf_values
