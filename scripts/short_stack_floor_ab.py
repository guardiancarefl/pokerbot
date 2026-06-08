"""Paired A/B: cost of depth-confusion at short stacks, V0 (raw) vs V1
(short-stack floored). Inference only. No model / encoder changes.

V0: current sample-mode behavior.
V1: identical, except when hero effective stack ≤ THRESHOLD BB:
      facing action  → keep {FOLD, CALL, ALLIN}
      to_call == 0   → keep {CALL, ALLIN}   (avoid free-fold pathology)
      otherwise (true unopened first-to-act) → keep {FOLD, ALLIN}

Paired design: V0 and V1 use the SAME per-game RNG seed → identical
card runouts and identical opponent draws up to the first hero
divergence (after which downstream divergence is the cost being
measured).

Per-decision logging (both arms): blind level, hero eff-stack-BB,
position, facing-action flag, street, full 9-d action distribution,
sampled action, short-stack-coherence flag (fraction of mass on
{FOLD, ALLIN} at ≤ threshold BB).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyspiel

from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.cfr6 import (
    _build_view_6max, compute_icm_payouts, is_tournament_terminal,
)
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.infoset6 import parse_state_6max

CHECKPOINT = "runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt"
ABSTRACTION = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"
N_SEATS = 6
N_ACT = len(DiscreteAction)
HERO_SEAT_DEFAULT = 0
# Action set memberships used by the V1 floor mask
A_FOLD = int(DiscreteAction.FOLD)
A_CALL = int(DiscreteAction.CALL)
A_ALLIN = int(DiscreteAction.ALLIN)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# V1 floor + coherence helpers
# --------------------------------------------------------------------------

def _hero_eff_bb(parsed: dict) -> tuple[float, int]:
    """Return (effective-stack-in-BB, big_blind_amount). 0,0 if unknown."""
    bb = int(parsed.get("big_blind", 0))
    if bb <= 0:
        return 0.0, 0
    cp = parsed["current_player"]
    my_stack = int(parsed["money"][cp])
    opps = [int(parsed["money"][i]) for i in range(N_SEATS)
            if i != cp and parsed["money"][i] > 0]
    eff = min(my_stack, max(opps)) if opps else my_stack
    return float(eff) / bb, bb


def _to_call(parsed: dict) -> int:
    cp = parsed["current_player"]
    contribs = parsed["contribution"]
    mx = max(contribs) if contribs else 0
    return int(mx - contribs[cp])


def short_stack_coherence(policy: np.ndarray, legal_mask: np.ndarray) -> float:
    """Fraction of policy mass on {FOLD, CALL, ALLIN} (the "pure" shove/fold
    actions). A coherent short-stack strategy concentrates here; min-raise
    /BET_50/BET_66/BET_100/BET_150/BET_200 are the depth-confusion tells.
    Returns 0 if any of the three discrete-action slots are illegal AND
    have nonzero policy (shouldn't happen for a properly-masked sampler)."""
    pure_idxs = (A_FOLD, A_CALL, A_ALLIN)
    return float(sum(policy[i] for i in pure_idxs if legal_mask[i] > 0))


def apply_short_stack_floor(policy: np.ndarray, legal_mask: np.ndarray,
                             parsed: dict, threshold_bb: float):
    """V1 mask + renormalize. Returns (new_policy, fired_bool, eff_bb,
    facing_action_bool, regime_str). On fire, regime is 'facing'/'check'/'unopened'."""
    eff_bb, bb = _hero_eff_bb(parsed)
    if bb <= 0 or eff_bb > threshold_bb:
        return policy, False, eff_bb, None, "skip"
    to_call = _to_call(parsed)
    facing = to_call > 0
    # Determine regime + kept actions
    if facing:
        keep = (A_FOLD, A_CALL, A_ALLIN)
        regime = "facing"
    else:
        # to_call == 0. In universal_poker preflop this is usually BB-after-
        # limp (CHECK available — strictly dominates folding). Postflop
        # check spots are similar. Mask FOLD only when CALL (= CHECK) is
        # legal; otherwise fall back to FOLD/ALLIN.
        if legal_mask[A_CALL] > 0:
            keep = (A_CALL, A_ALLIN)
            regime = "check"
        else:
            keep = (A_FOLD, A_ALLIN)
            regime = "unopened"
    keep_mask = np.zeros(N_ACT, dtype=np.float32)
    for a in keep:
        if legal_mask[a] > 0:
            keep_mask[a] = 1.0
    if keep_mask.sum() == 0:
        # No legal actions in kept set — fall back to original distribution
        return policy, False, eff_bb, facing, "fallback_empty_keep"
    new = (policy.astype(np.float64) * keep_mask.astype(np.float64))
    s = new.sum()
    if s > 1e-12:
        new = new / s
    else:
        # All mass was on intermediate sizes — uniform over the kept set
        new = keep_mask.astype(np.float64) / float(keep_mask.sum())
    return new.astype(policy.dtype), True, eff_bb, facing, regime


def position_label(seat: int, dealer_seat: int, n_alive_seats: list[int]) -> str:
    """6-handed position label relative to dealer (BTN). Uses ALIVE rotation
    only — busted seats are skipped — so labels match real-game semantics.
    Defensive against dealer pointing to a busted-seat placeholder by
    advancing clockwise to the next alive seat."""
    if not n_alive_seats or seat not in n_alive_seats:
        return f"UNK({seat})"
    if dealer_seat not in n_alive_seats:
        d = dealer_seat
        for _ in range(N_SEATS):
            d = (d + 1) % N_SEATS
            if d in n_alive_seats:
                dealer_seat = d
                break
        if dealer_seat not in n_alive_seats:
            return f"UNK({seat})"
    k = len(n_alive_seats)
    didx = n_alive_seats.index(dealer_seat)
    sidx = n_alive_seats.index(seat)
    rel = (sidx - didx) % k
    if k == 6:
        names = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    elif k == 5:
        names = ["BTN", "SB", "BB", "UTG", "CO"]
    elif k == 4:
        names = ["BTN", "SB", "BB", "UTG"]
    else:
        return f"REL{rel}/N{k}"
    return names[rel]


# --------------------------------------------------------------------------
# Hand + match driver
# --------------------------------------------------------------------------

def _decide(solver, parsed: dict, state, rng: random.Random,
            hero_seat: int, threshold_bb: Optional[float],
            log_sink: Optional[list], game_idx: int, hand_idx: int,
            blind_level_idx: int, dealer_seat: int):
    """One decision-node action: query policy, optionally apply V1 floor,
    sample, and (optionally) log. Returns chip_int."""
    cp = parsed["current_player"]
    legal_chip = list(state.legal_actions())
    view = _build_view_6max(state, parsed)
    d2c = discretize_legal_actions(legal_chip, view)
    legal_mask = np.zeros(N_ACT, dtype=np.float32)
    for da in d2c:
        legal_mask[int(da)] = 1.0
    feat = solver.encoder.encode_from_parsed(
        parsed, rng=random.Random(cp * 100003 + 7))  # deterministic encoder rng
    raw_policy = np.asarray(
        solver.policy_nets.inference_policy(cp, feat, legal_mask),
        dtype=np.float64,
    )
    fired = False
    eff_bb_decision, _ = _hero_eff_bb(parsed)
    regime = None
    if cp == hero_seat and threshold_bb is not None:
        policy, fired, eff_bb_decision, _facing, regime = \
            apply_short_stack_floor(
                raw_policy, legal_mask, parsed, threshold_bb)
    else:
        policy = raw_policy
    # Sample from `policy` masked to legal
    keys = np.flatnonzero(legal_mask).tolist()
    ws = [float(policy[k]) for k in keys]
    s_ws = sum(ws)
    if s_ws <= 0:
        choice = rng.choice(keys)
    else:
        choice = rng.choices(keys, weights=ws, k=1)[0]
    chip = d2c.get(DiscreteAction(int(choice)))
    if chip is None:
        chip = next(iter(d2c.values()))
    # Log hero decision in both arms
    if cp == hero_seat and log_sink is not None:
        alive_seats = [i for i in range(N_SEATS)
                       if int(parsed["money"][i]) > 0]
        pos = position_label(hero_seat, dealer_seat, alive_seats)
        contribs = parsed["contribution"]
        to_call = max(contribs) - contribs[cp] if contribs else 0
        log_sink.append({
            "game": int(game_idx),
            "hand": int(hand_idx),
            "level": int(blind_level_idx),
            "street": int(parsed.get("street_idx", 0)),
            "position": pos,
            "n_alive": int(len(alive_seats)),
            "hero_eff_bb": float(eff_bb_decision),
            "facing_action": bool(to_call > 0),
            "raw_policy": [float(x) for x in raw_policy],
            "post_filter_policy": [float(x) for x in policy],
            "legal_mask": [int(x) for x in legal_mask],
            "sampled_disc_action": int(choice),
            "sampled_chip_int": int(chip),
            "floor_fired": bool(fired),
            "floor_regime": regime,
            "raw_short_stack_coherence":
                short_stack_coherence(raw_policy, legal_mask),
        })
    return int(chip)


def play_one_hand(solver, structure: TournamentStructure, stacks: list[int],
                  blind_level_idx: int, dealer_seat: int,
                  rng: random.Random, hero_seat: int,
                  threshold_bb: Optional[float], log_sink: Optional[list],
                  game_idx: int, hand_idx: int) -> tuple[list[int], int]:
    """Play one hand. Returns (new_stacks, n_decisions_logged_for_hero)."""
    blind_level = structure.level(blind_level_idx)
    n = structure.num_players
    sb = blind_level.small_blind
    bb_inflated = blind_level.inflated_big_blind(n)

    stacks = list(stacks)
    for i in range(n):
        if 0 < stacks[i] < sb:
            stacks[i] = 0

    alive_seats = [i for i, s in enumerate(stacks) if s > 0]
    if len(alive_seats) <= 3:
        return stacks, 0
    if dealer_seat not in alive_seats:
        i = dealer_seat
        while i not in alive_seats:
            i = (i + 1) % n
        dealer_seat = i

    def sb_bb_seats_for_dealer(d, alive):
        dpos = alive.index(d)
        return alive[(dpos + 1) % len(alive)], alive[(dpos + 2) % len(alive)]

    guard = 0
    while len(alive_seats) > 3 and guard < n:
        sb_s, bb_s = sb_bb_seats_for_dealer(dealer_seat, alive_seats)
        if stacks[sb_s] >= sb and stacks[bb_s] >= bb_inflated:
            break
        if stacks[sb_s] < sb:
            stacks[sb_s] = 0
        if stacks[bb_s] < bb_inflated:
            stacks[bb_s] = 0
        alive_seats = [i for i, s in enumerate(stacks) if s > 0]
        if len(alive_seats) <= 3:
            return stacks, 0
        if dealer_seat not in alive_seats:
            i = dealer_seat
            while i not in alive_seats:
                i = (i + 1) % n
            dealer_seat = i
        guard += 1

    game_str = structure.to_inner_game_string_for_state(
        blind_level, stacks, dealer_seat)
    g = pyspiel.load_game(game_str)
    state = g.new_initial_state()

    n_logged_before = len(log_sink) if log_sink is not None else 0
    while not state.is_terminal():
        if state.is_chance_node():
            outs = state.chance_outcomes()
            a = rng.choices([o for o, _ in outs],
                            weights=[p for _, p in outs], k=1)[0]
            state.apply_action(int(a))
            continue
        cur = state.current_player()
        if cur < 0:
            break
        if stacks[cur] <= 0:
            legal = state.legal_actions()
            state.apply_action(int(legal[0]))
            continue
        parsed = parse_state_6max(state)
        # parse_state_6max also sets "dealer_seat" if available; ensure it
        parsed["dealer_seat"] = dealer_seat
        chip = _decide(solver, parsed, state, rng, hero_seat,
                       threshold_bb, log_sink, game_idx,
                       hand_idx, blind_level_idx, dealer_seat)
        state.apply_action(int(chip))

    returns = list(state.returns())
    new_stacks = list(stacks)
    for i in range(n):
        new_stacks[i] = max(0, stacks[i] + int(returns[i]))
    for i in range(n):
        if stacks[i] == 0:
            new_stacks[i] = 0
    n_logged_after = len(log_sink) if log_sink is not None else n_logged_before
    return new_stacks, n_logged_after - n_logged_before


def play_match(solver, structure: TournamentStructure, *,
                seed: int, hero_seat: int, threshold_bb: Optional[float],
                starting_stack: int, hands_per_level: int, max_hands: int,
                log_sink: Optional[list], game_idx: int) -> dict:
    """Play one full match (to 3-left bubble or max_hands cap)."""
    rng = random.Random(seed)
    n = structure.num_players
    stacks = [starting_stack] * n
    level = 1
    max_level = max(bl.level for bl in structure.blind_schedule)
    dealer = rng.randrange(n)
    hands_played = 0
    hands_in_level = 0

    while True:
        n_alive = sum(1 for s in stacks if s > 0)
        if n_alive <= 3 or hands_played >= max_hands:
            break
        if hands_in_level >= hands_per_level:
            level = min(level + 1, max_level)
            hands_in_level = 0
        try:
            stacks, _ = play_one_hand(
                solver, structure, stacks, level, dealer, rng,
                hero_seat, threshold_bb, log_sink, game_idx, hands_played,
            )
        except Exception as e:
            bl = structure.level(level)
            for i in range(n):
                if 0 < stacks[i] < bl.inflated_big_blind(n):
                    stacks[i] = 0
            print(f"  [warn] game {game_idx} hand {hands_played+1}: "
                  f"{type(e).__name__}: {str(e)[:80]}", flush=True)
        hands_played += 1
        hands_in_level += 1
        dealer = (dealer + 1) % n
        while stacks[dealer] == 0:
            dealer = (dealer + 1) % n

    # ICM at termination (top-3 equal pay)
    icms = compute_icm_payouts(
        stacks=stacks, num_paid=3, payout_per_seat=2.0, buy_in_units=1.0,
    )
    return {
        "hands_played": hands_played,
        "n_alive_end": sum(1 for s in stacks if s > 0),
        "level_end": level,
        "final_stacks": stacks,
        "hero_icm": float(icms[hero_seat]),
        "all_icms": [float(x) for x in icms],
    }


# --------------------------------------------------------------------------
# Paired runner
# --------------------------------------------------------------------------

def play_paired_game(solver, structure, *, seed: int, hero_seat: int,
                      threshold_bb: float, starting_stack: int,
                      hands_per_level: int, max_hands: int,
                      log_v0: list, log_v1: list,
                      game_idx: int) -> dict:
    """Run the same game twice — V0 (raw) and V1 (floored). Same seed in
    both arms → identical card runouts + opponent draws up to first hero
    divergence."""
    v0 = play_match(solver, structure, seed=seed, hero_seat=hero_seat,
                     threshold_bb=None, starting_stack=starting_stack,
                     hands_per_level=hands_per_level, max_hands=max_hands,
                     log_sink=log_v0, game_idx=game_idx)
    v1 = play_match(solver, structure, seed=seed, hero_seat=hero_seat,
                     threshold_bb=threshold_bb,
                     starting_stack=starting_stack,
                     hands_per_level=hands_per_level, max_hands=max_hands,
                     log_sink=log_v1, game_idx=game_idx)
    return {
        "game": game_idx, "seed": seed, "hero_seat": hero_seat,
        "v0_icm": v0["hero_icm"], "v1_icm": v1["hero_icm"],
        "delta": v1["hero_icm"] - v0["hero_icm"],
        "v0_hands": v0["hands_played"], "v1_hands": v1["hands_played"],
        "v0_level_end": v0["level_end"], "v1_level_end": v1["level_end"],
        "v0_n_alive_end": v0["n_alive_end"],
        "v1_n_alive_end": v1["n_alive_end"],
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_solver(structure):
    print(f"[load] sha256(ckpt) = {sha256_of(CHECKPOINT)}")
    print(f"[load] sha256(abstr) = {sha256_of(ABSTRACTION)}")
    print(f"[load] checkpoint path: {CHECKPOINT}")
    print(f"[load] abstraction path: {ABSTRACTION}")
    with open(ABSTRACTION, "rb") as f:
        abstr = pickle.load(f)
    from scripts.eval_6max_self_play import _load_solver
    solver = _load_solver(CHECKPOINT, abstr, structure)
    assert solver.encoder.feature_dim == 236, (
        f"feature_dim {solver.encoder.feature_dim} != 236"
    )
    print(f"[load] feature_dim = {solver.encoder.feature_dim}  ✓")
    return solver


def benchmark(solver, structure, *, threshold_bb, max_hands,
              hands_per_level, starting_stack):
    print("\n[bench] one paired game end-to-end...")
    log_v0, log_v1 = [], []
    t0 = time.time()
    g = play_paired_game(
        solver, structure, seed=0xBEEF, hero_seat=HERO_SEAT_DEFAULT,
        threshold_bb=threshold_bb, starting_stack=starting_stack,
        hands_per_level=hands_per_level, max_hands=max_hands,
        log_v0=log_v0, log_v1=log_v1, game_idx=0,
    )
    elapsed = time.time() - t0
    print(f"  v0: hands={g['v0_hands']} level_end={g['v0_level_end']} "
          f"n_alive_end={g['v0_n_alive_end']} icm={g['v0_icm']:+.1f}")
    print(f"  v1: hands={g['v1_hands']} level_end={g['v1_level_end']} "
          f"n_alive_end={g['v1_n_alive_end']} icm={g['v1_icm']:+.1f}")
    print(f"  delta={g['delta']:+.2f}  hero_decisions: "
          f"v0={len(log_v0)} v1={len(log_v1)}  wall={elapsed:.2f}s")
    n_floor_fired_v1 = sum(1 for r in log_v1 if r["floor_fired"])
    print(f"  v1 floor fired: {n_floor_fired_v1}/{len(log_v1)} hero decisions")
    return elapsed


def run_full(solver, structure, *, n_games: int, threshold_bb: float,
              max_hands: int, hands_per_level: int, starting_stack: int,
              out_dir: Path, base_seed: int = 1):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_v0_path = out_dir / "decisions_v0.jsonl"
    log_v1_path = out_dir / "decisions_v1.jsonl"
    games_path = out_dir / "games.jsonl"
    print(f"\n[run] N={n_games} paired games  threshold_bb={threshold_bb}")
    print(f"      out_dir={out_dir}")
    t0 = time.time()
    with open(log_v0_path, "w") as fv0, \
         open(log_v1_path, "w") as fv1, \
         open(games_path, "w") as fg:
        log_v0, log_v1 = [], []
        for i in range(n_games):
            seed = base_seed + i
            log_v0.clear(); log_v1.clear()
            try:
                g = play_paired_game(
                    solver, structure, seed=seed,
                    hero_seat=HERO_SEAT_DEFAULT,
                    threshold_bb=threshold_bb,
                    starting_stack=starting_stack,
                    hands_per_level=hands_per_level, max_hands=max_hands,
                    log_v0=log_v0, log_v1=log_v1, game_idx=i,
                )
            except Exception as e:
                print(f"  game {i}: FAIL {type(e).__name__}: {str(e)[:80]}",
                      flush=True)
                continue
            for rec in log_v0:
                fv0.write(json.dumps(rec) + "\n")
            for rec in log_v1:
                fv1.write(json.dumps(rec) + "\n")
            fg.write(json.dumps(g) + "\n")
            if (i + 1) % 50 == 0:
                fg.flush(); fv0.flush(); fv1.flush()
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (n_games - i - 1) / max(rate, 1e-9)
                print(f"  game {i+1}/{n_games}  "
                      f"elapsed={el/60:.1f}m  "
                      f"rate={rate:.2f}g/s  eta={eta/60:.1f}m",
                      flush=True)
    print(f"\n[run] DONE  total {(time.time()-t0)/60:.1f}m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("bench", "run"), default="bench")
    ap.add_argument("--n-games", type=int, default=200)
    ap.add_argument("--threshold-bb", type=float, default=6.0)
    ap.add_argument("--max-hands", type=int, default=200)
    ap.add_argument("--hands-per-level", type=int, default=3)
    ap.add_argument("--starting-stack", type=int, default=1500)
    ap.add_argument("--out-dir", type=str, default="evals/short_stack_floor_ab")
    ap.add_argument("--base-seed", type=int, default=1)
    args = ap.parse_args()

    print(f"[args] {vars(args)}")
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    solver = load_solver(structure)

    if args.mode == "bench":
        benchmark(solver, structure,
                   threshold_bb=args.threshold_bb,
                   max_hands=args.max_hands,
                   hands_per_level=args.hands_per_level,
                   starting_stack=args.starting_stack)
        return

    out_dir = Path(args.out_dir)
    run_full(solver, structure,
             n_games=args.n_games, threshold_bb=args.threshold_bb,
             max_hands=args.max_hands,
             hands_per_level=args.hands_per_level,
             starting_stack=args.starting_stack,
             out_dir=out_dir, base_seed=args.base_seed)


if __name__ == "__main__":
    main()
