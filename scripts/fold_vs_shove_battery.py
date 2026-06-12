"""EXP_H2 fold-vs-shove battery (spec §2): frozen oracle battery + grader.

A FIXED set of preflop facing-all-in decision spots (hero 5-15 BB eff,
exactly one all-in raiser, blinds L3-L7) with a KILLPHIL-OPTIMAL oracle
response per spot, frozen to disk once; checkpoints are then graded
against the file (M1 = mean ICM-EV loss vs oracle, in ICM units).

Oracle construction per (shover_pos, depth, level) cell:
  1. The shover's range = the set of 169 hand classes the FIXED
     killphilmtt adapter actually open-shoves when seated in that spot
     (queried through the real ShankyProfilePolicy on the real OpenSpiel
     state — adapter-as-played semantics, post-a8933ea fix).
  2. For each hero hand class: EV(fold) and EV(call) in Malmuth-Harville
     ICM units (double-up payouts), equity vs the weighted range via
     equity_vs_range. Oracle = argmax. Spots where the range is EMPTY
     (killphil never shoves that cell) are dropped.

Approximations (same for every graded checkpoint, cancel in deltas):
  - ties folded into equity at half weight (no explicit chop branch);
  - probe construction gives every seat the same stack (depth == eff
    depth everywhere; pool conservation relaxed exactly as
    depth_invariance_probe does).

Usage:
  # freeze the battery (ONCE, before the probe — spec §2):
  python -m scripts.fold_vs_shove_battery make \
      --out evals/h2_battery/battery_v1.json

  # grade a checkpoint (M1):
  python -m scripts.fold_vs_shove_battery grade \
      --battery evals/h2_battery/battery_v1.json \
      --ckpt runs/.../ckpt_iter_1500.pt \
      --out evals/h2_battery/m1_champion.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.nlhe.game_strings import TournamentStructure  # noqa: E402
from src.nlhe.icm import icm_equity  # noqa: E402
from src.nlhe.equity import equity_vs_range  # noqa: E402
from src.nlhe.actions import DiscreteAction  # noqa: E402
from scripts.depth_invariance_probe import (  # noqa: E402
    enumerate_169_hands, build_preflop_spot, query_policy,
    HERO_SEAT, N_SEATS)
from scripts.sng_baseline import PAYOUTS  # noqa: E402

STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"
KILLPHIL = "data/shanky_profiles/KillPhilMTT.txt"

# Battery cells (spec §2): (hero_pos, shover_pos). The shover always
# open-shoves first-in; everyone else folds to hero.
CELLS = [("BTN", "UTG"), ("SB", "UTG"), ("BB", "UTG"), ("BB", "SB")]
DEPTHS_BB = [5, 8, 11, 15]
LEVELS = [3, 5, 7]

_POS_OFFSET = {"UTG": 3, "MP": 4, "CO": 5, "BTN": 0, "SB": 1, "BB": 2}


def _dealer_for(pos: str) -> int:
    """Dealer seat that puts HERO_SEAT at `pos` (matches
    depth_invariance_probe's convention)."""
    return (HERO_SEAT - _POS_OFFSET[pos]) % N_SEATS


def build_facing_shove_spot(structure, level_num, hero_pos, shover_pos,
                            hero_cards, depth_bb):
    """OpenSpiel state: `shover_pos` open-shoves first-in, action folds
    to hero at `hero_pos`. All seats start at depth_bb * bb chips.
    Returns (state, dealer, bb, shover_seat)."""
    bl = structure.level(level_num)
    bb = bl.big_blind
    chips = int(round(depth_bb * bb))
    dealer = _dealer_for(hero_pos)
    shover_seat = (dealer + _POS_OFFSET[shover_pos]) % N_SEATS
    if shover_seat == HERO_SEAT:
        raise ValueError("shover == hero")

    import pyspiel
    from scripts.depth_invariance_probe import deal_one_card_6max
    stacks = [chips] * N_SEATS
    game_str = structure.to_inner_game_string_for_state(
        blind_level=bl, stacks=stacks, dealer_seat=dealer)
    state = pyspiel.load_game(game_str).new_initial_state()
    safety = 50
    while state.is_chance_node():
        safety -= 1
        if safety < 0:
            raise RuntimeError("chance loop")
        deal_one_card_6max(state, HERO_SEAT, hero_cards, target_board=())

    safety = 30
    while state.current_player() != HERO_SEAT:
        safety -= 1
        if safety < 0 or state.is_terminal():
            raise RuntimeError("never reached hero")
        cp = state.current_player()
        legal = state.legal_actions()
        if cp == shover_seat:
            state.apply_action(max(legal))      # open-shove (max chip int)
        elif 0 in legal:
            state.apply_action(0)               # fold
        else:
            raise RuntimeError(f"seat {cp} can't fold; legal={legal[:5]}")
    return state, dealer, bb, shover_seat


def _build_unopened_spot(structure, level_num, pos, cards, depth_bb):
    """OpenSpiel state with HERO_SEAT at `pos` (any of _POS_OFFSET),
    first-in unopened (everyone before hero folds; SB limps when hero is
    BB, matching build_preflop_spot's convention). All seats equal-stack
    at depth_bb * bb."""
    import pyspiel
    from scripts.depth_invariance_probe import deal_one_card_6max
    bl = structure.level(level_num)
    bb = bl.big_blind
    chips = int(round(depth_bb * bb))
    if chips < bb + bl.ante:
        raise ValueError("cannot post")
    dealer = _dealer_for(pos)
    sb_seat = (dealer + 1) % N_SEATS
    game_str = structure.to_inner_game_string_for_state(
        blind_level=bl, stacks=[chips] * N_SEATS, dealer_seat=dealer)
    state = pyspiel.load_game(game_str).new_initial_state()
    safety = 50
    while state.is_chance_node():
        safety -= 1
        if safety < 0:
            raise RuntimeError("chance loop")
        deal_one_card_6max(state, HERO_SEAT, cards, target_board=())
    safety = 30
    while state.current_player() != HERO_SEAT:
        safety -= 1
        if safety < 0 or state.is_terminal():
            raise RuntimeError("never reached hero")
        cp = state.current_player()
        legal = state.legal_actions()
        if pos == "BB" and cp == sb_seat and 1 in legal:
            state.apply_action(1)
        elif 0 in legal:
            state.apply_action(0)
        elif 1 in legal:
            state.apply_action(1)
        else:
            raise RuntimeError(f"seat {cp} stuck; legal={legal[:5]}")
    return state, dealer, bb


def killphil_shove_range(structure, level_num, shover_pos, depth_bb,
                         hands, profile_path=KILLPHIL):
    """Set of hand labels the fixed killphil adapter open-shoves
    first-in at this cell (queried as the acting player on the real
    OpenSpiel state)."""
    import random as _random
    from src.nlhe.scripted_bots.policy import ShankyProfilePolicy
    from src.nlhe.infoset6 import parse_state_6max
    bl = structure.level(level_num)
    bot = ShankyProfilePolicy(name="killphilmtt", profile_path=profile_path,
                              big_blind_chips=bl.big_blind)
    shove = set()
    for c1, c2, label, _kind in hands:
        try:
            state, _dealer, _bb = _build_unopened_spot(
                structure, level_num, shover_pos, (c1, c2), depth_bb)
        except (ValueError, RuntimeError):
            continue
        parsed = parse_state_6max(state)
        chip = int(bot.select_action(parsed, state, _random.Random(0)))
        legal = state.legal_actions()
        # all-in = the max chip action; treat any raise-to >= 90% of max
        # as a shove (profiles emit RaiseMax; adapter may clip).
        if chip >= 0.9 * max(legal) and chip > 1:
            shove.add(label)
    return shove


def oracle_ev(structure, level_num, hero_pos, shover_pos, depth_bb,
              hero_label, hero_cards, shove_labels):
    """(ev_fold, ev_call) in hero ICM units for the constructed spot."""
    bl = structure.level(level_num)
    sb, bb, ante = bl.small_blind, bl.big_blind, bl.ante
    chips = int(round(depth_bb * bb))
    dealer = _dealer_for(hero_pos)
    shover_seat = (dealer + _POS_OFFSET[shover_pos]) % N_SEATS
    sb_seat, bb_seat = (dealer + 1) % N_SEATS, (dealer + 2) % N_SEATS

    def post(seat):
        p = ante
        if seat == sb_seat:
            p += sb
        elif seat == bb_seat:
            p += bb
        return min(p, chips)

    dead_others = sum(post(i) for i in range(N_SEATS)
                      if i not in (HERO_SEAT, shover_seat))
    # FOLD: hero keeps chips - own post; shover scoops everything posted.
    stacks_fold = [chips - post(i) for i in range(N_SEATS)]
    stacks_fold[shover_seat] = chips + sum(
        post(i) for i in range(N_SEATS) if i != shover_seat)
    ev_fold = _hero_icm(stacks_fold)

    # CALL: both all-in for `chips` total each; dead money from others.
    eq = _equity_vs_labels(hero_cards, shove_labels)
    stacks_win = [chips - post(i) for i in range(N_SEATS)]
    stacks_win[HERO_SEAT] = 2 * chips + dead_others
    stacks_win[shover_seat] = 0
    stacks_lose = [chips - post(i) for i in range(N_SEATS)]
    stacks_lose[HERO_SEAT] = 0
    stacks_lose[shover_seat] = 2 * chips + dead_others
    ev_call = eq * _hero_icm(stacks_win) + (1 - eq) * _hero_icm(stacks_lose)
    return float(ev_fold), float(ev_call)


_SUITS = "cdhs"


def _label_combos(label: str):
    """All literal 2-card combos (treys ints) of a 169-class label."""
    from src.nlhe.equity import cards_from_str
    r1, r2 = label[0], label[1]
    kind = label[2] if len(label) > 2 else "pair"
    combos = []
    if r1 == r2:
        for a in range(4):
            for b in range(a + 1, 4):
                combos.append(tuple(cards_from_str(
                    r1 + _SUITS[a] + r2 + _SUITS[b])))
    elif kind == "s":
        for s in _SUITS:
            combos.append(tuple(cards_from_str(r1 + s + r2 + s)))
    else:
        for s1 in _SUITS:
            for s2 in _SUITS:
                if s1 != s2:
                    combos.append(tuple(cards_from_str(r1 + s1 + r2 + s2)))
    return combos


def _equity_vs_labels(hero_cards, shove_labels, trials=400):
    """Hero equity vs the label range, card-removal filtered, seeded."""
    import random as _random
    from src.nlhe.equity import cards_from_str
    hero = cards_from_str(hero_cards[0] + hero_cards[1])
    blocked = set(hero)
    rng_combos = []
    for lab in sorted(shove_labels):
        for c in _label_combos(lab):
            if c[0] not in blocked and c[1] not in blocked:
                rng_combos.append(list(c))
    if not rng_combos:
        return 0.5
    import zlib
    seed = zlib.crc32("|".join((hero_cards[0], hero_cards[1],
                                *sorted(shove_labels))).encode())
    return float(equity_vs_range(hero, rng_combos, trials=trials,
                                 rng=_random.Random(seed)))


def _hero_icm(stacks):
    eligible = [i for i in range(N_SEATS) if stacks[i] > 0]
    if HERO_SEAT not in eligible:
        return 0.0
    return float(icm_equity(stacks, PAYOUTS, eligible=eligible)[HERO_SEAT])


def make_battery(out_path: str):
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    hands = enumerate_169_hands()
    spots, range_cache = [], {}
    t0 = time.time()
    for level in LEVELS:
        for depth in DEPTHS_BB:
            for hero_pos, shover_pos in CELLS:
                key = (shover_pos, depth, level)
                if key not in range_cache:
                    range_cache[key] = killphil_shove_range(
                        structure, level, shover_pos, depth, hands)
                    print(f"[range] {key}: {len(range_cache[key])}/169 shove",
                          flush=True)
                rng_labels = range_cache[key]
                if not rng_labels:
                    continue
                for c1, c2, label, _kind in hands:
                    try:
                        ev_f, ev_c = oracle_ev(
                            structure, level, hero_pos, shover_pos, depth,
                            label, (c1, c2), rng_labels)
                    except Exception as e:
                        print(f"[skip] {level}/{depth}/{hero_pos}: {e}")
                        continue
                    spots.append({
                        "level": level, "depth_bb": depth,
                        "hero_pos": hero_pos, "shover_pos": shover_pos,
                        "hero_cards": [c1, c2], "hero_label": label,
                        "ev_fold": ev_f, "ev_call": ev_c,
                        "oracle": "call" if ev_c > ev_f else "fold",
                        "ev_gap": abs(ev_c - ev_f),
                    })
    out = {
        "version": "h2_battery_v1",
        "structure": STRUCTURE_YAML, "profile": KILLPHIL,
        "cells": CELLS, "depths_bb": DEPTHS_BB, "levels": LEVELS,
        "n_spots": len(spots),
        "ranges": {f"{k[0]}|{k[1]}|{k[2]}": sorted(v)
                   for k, v in range_cache.items()},
        "spots": spots,
        "elapsed_s": time.time() - t0,
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"[make] {len(spots)} spots -> {p}  ({out['elapsed_s']:.0f}s)")


def grade(battery_path: str, ckpt: str, out_path: str):
    from src.nlhe.abstraction import Abstraction
    from scripts.eval_pool import CheckpointPolicy
    from scripts.throwaway_query_real_ante import _RealAnteStructure
    battery = json.loads(Path(battery_path).read_text())
    structure = _RealAnteStructure(
        TournamentStructure.from_yaml(battery["structure"]))
    abstr = Abstraction.load(
        "runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    pol = CheckpointPolicy("graded", ckpt, abstr, structure)

    losses, n_oracle_call, n_policy_call_mass = [], 0, 0.0
    for s in battery["spots"]:
        state, dealer, _bb, _sh = build_facing_shove_spot(
            structure, s["level"], s["hero_pos"], s["shover_pos"],
            tuple(s["hero_cards"]), s["depth_bb"])
        policy, mask, _parsed, d2c, _f = query_policy(
            pol.solver, state, HERO_SEAT, dealer, rng_seed=0)
        fold_i, call_i = int(DiscreteAction.FOLD), int(DiscreteAction.CALL)
        # facing a shove every non-fold discrete action commits the
        # stack — treat all non-FOLD legal mass as CALL.
        p_fold = policy[fold_i] if mask[fold_i] else 0.0
        legal_mass = float(sum(policy[i] for i in range(len(policy))
                               if mask[i]))
        p_call = max(0.0, legal_mass - p_fold)
        tot = p_fold + p_call
        if tot <= 0:
            continue
        p_call /= tot
        ev_policy = p_call * s["ev_call"] + (1 - p_call) * s["ev_fold"]
        ev_oracle = max(s["ev_call"], s["ev_fold"])
        losses.append(ev_oracle - ev_policy)
        n_oracle_call += (s["oracle"] == "call")
        n_policy_call_mass += p_call

    m1 = float(np.mean(losses))
    out = {
        "battery": battery["version"], "ckpt": ckpt,
        "n_spots_graded": len(losses),
        "m1_mean_ev_loss": m1,
        "m1_se": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
        "oracle_call_rate": n_oracle_call / len(losses),
        "policy_call_mass_mean": n_policy_call_mass / len(losses),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--out", required=True)
    gr = sub.add_parser("grade")
    gr.add_argument("--battery", required=True)
    gr.add_argument("--ckpt", required=True)
    gr.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "make":
        make_battery(args.out)
    else:
        grade(args.battery, args.ckpt, args.out)


if __name__ == "__main__":
    main()
