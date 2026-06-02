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


def gadget_resolve_node(game, entries, r0_prior, r1_prior, w0, w1,
                        outer: int = 40, inner: int = 60,
                        dcfr: Optional[DCFRParams] = None):
    """Safe re-solve of one round-2 public node via the CFR-D gadget.

    r{0,1}_prior: blueprint reach range per hand. w{0,1}: opt-out CFV per hand.
    Returns (round2_map, leaf_values, cfv0, cfv1) where cfvs are the safe
    counterfactual values per hand (the trunk leaf values).
    """
    nump = game.num_players()
    dcfr = dcfr or DCFRParams(mode="plus")
    cards = [c for _, c in entries]
    hands0 = sorted({c[0] for c in cards})
    hands1 = sorted({c[1] for c in cards})
    # gadget follow regrets per hand (FOLLOW vs TERMINATE)
    g_reg0 = {h: np.zeros(2) for h in hands0}
    g_reg1 = {h: np.zeros(2) for h in hands1}
    follow0 = {h: 1.0 for h in hands0}
    follow1 = {h: 1.0 for h in hands1}

    solver = None
    last_map, last_lv = {}, {}
    cfv0 = {h: 0.0 for h in hands0}
    cfv1 = {h: 0.0 for h in hands1}
    for it in range(outer):
        range0 = {h: r0_prior[h] * follow0[h] for h in hands0}
        range1 = {h: r1_prior[h] * follow1[h] for h in hands1}
        roots = []
        for st, (h0, h1) in entries:
            reach = np.ones(nump + 1)
            reach[0] = range0[h0]
            reach[1] = range1[h1]
            roots.append((st, reach))
        if solver is None:
            solver = SubgameCFR(game, roots=roots, dcfr=dcfr)
        else:
            solver.roots = roots
        solver.run(inner)
        # per-hand counterfactual values under current solved strategy
        vals = {}
        for st, (h0, h1) in entries:
            vals[(h0, h1)] = solver._eval_avg(st.clone(), np.ones(nump + 1))
        cfv0 = {h: 0.0 for h in hands0}
        cfv1 = {h: 0.0 for h in hands1}
        for (h0, h1), v in vals.items():
            cfv0[h0] += range1[h1] * v[0]
            cfv1[h1] += range0[h0] * v[1]
        # gadget regret update: FOLLOW value = cfv, TERMINATE value = w*
        for h in hands0:
            f = _regret_match2(g_reg0[h])
            node_v = f[0] * cfv0[h] + f[1] * w0[h]
            g_reg0[h][0] += cfv0[h] - node_v
            g_reg0[h][1] += w0[h] - node_v
            follow0[h] = _regret_match2(g_reg0[h])[0]
        for h in hands1:
            f = _regret_match2(g_reg1[h])
            node_v = f[0] * cfv1[h] + f[1] * w1[h]
            g_reg1[h][0] += cfv1[h] - node_v
            g_reg1[h][1] += w1[h] - node_v
            follow1[h] = _regret_match2(g_reg1[h])[0]

    # final solved strategy map + leaf values
    for info_key, node in solver.nodes.items():
        m = _ROUND_RE.search(info_key)
        if m and int(m.group(1)) == 2:
            avg = solver.average_strategy(info_key)
            last_map[info_key] = {int(a): float(p)
                                  for a, p in zip(node.legal, avg)}
    for st, _c in entries:
        last_lv[st.history_str()] = solver._eval_avg(
            st.clone(), np.ones(nump + 1))
    return last_map, last_lv, cfv0, cfv1


def gadget_resolve_all(game, blueprint, outer: int = 40, inner: int = 60,
                       dcfr: Optional[DCFRParams] = None):
    """Safe re-solve of every round-2 node against a blueprint. Returns
    (round2_map, leaf_values) assembled across all public nodes."""
    ranges, optouts, groups = blueprint_round2(game, blueprint)
    round2_map, leaf_values = {}, {}
    for gkey, entries in groups.items():
        r0 = {h: ranges[gkey][0][h] for h in {c[0] for _, c in entries}}
        r1 = {h: ranges[gkey][1][h] for h in {c[1] for _, c in entries}}
        w0 = {h: optouts[gkey][0][h] for h in r0}
        w1 = {h: optouts[gkey][1][h] for h in r1}
        m, lv, _c0, _c1 = gadget_resolve_node(
            game, entries, r0, r1, w0, w1, outer=outer, inner=inner, dcfr=dcfr)
        round2_map.update(m)
        leaf_values.update(lv)
    return round2_map, leaf_values
