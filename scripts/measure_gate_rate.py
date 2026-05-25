"""One-off: measure the SubgamePolicy gate fire-rate `f` over natural self-play
traffic (B1c sub-step 5, Stage 5-A). NOT part of the test suite.

Loads the production blueprint, plays N self-play hands (all seats blueprint), and at
EVERY hero decision evaluates the SubgamePolicy gate (Decision 5.1) and tallies
skip/solve by street — then advances the hand with the blueprint action so hands
complete (the Stage-5-A solve branch raises, so we never call full select_action).
Reports the empirical f = n_solve / n_total and the per-street breakdown.

Usage:
    python -m scripts.measure_gate_rate --hands 500 --seed 20
"""
from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict

import pyspiel

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.stack_sampler import sample_starting_state
from src.nlhe.infoset6 import parse_state_6max, parse_state_repeated_6max
from src.nlhe.subgame_policy import SubgamePolicy

ABSTR = "runs/abstraction_20260521_223018/abstraction.pkl"
CKPT = "runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/ckpt_iter_3000.pt"
STRUCT = "configs/ignition_double_up_6max_turbo.yaml"
_STREET = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--abstraction", default=ABSTR)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--structure", default=STRUCT)
    args = ap.parse_args()

    abstr = Abstraction.load(args.abstraction)
    structure = TournamentStructure.from_yaml(args.structure)
    policy = SubgamePolicy("sg-gate", args.ckpt, abstr, structure)
    print(f"blueprint={args.ckpt}\nplaying {args.hands} self-play hands, seed={args.seed}\n")

    master = random.Random(args.seed)
    tot = defaultdict(int)
    solve = defaultdict(int)
    n_total = n_solve = 0
    t0 = time.perf_counter()

    for h in range(args.hands):
        sampled = sample_starting_state(structure, master, num_paid=3)
        gs = structure.to_inner_game_string_for_state(
            blind_level=sampled["blind_level"], stacks=list(sampled["stacks"]),
            dealer_seat=sampled["dealer_seat"])
        state = pyspiel.load_game(gs).new_initial_state()
        rng = random.Random(master.randrange(2 ** 31))
        steps = 0
        while not state.is_terminal() and steps < 300:
            steps += 1
            if state.is_chance_node():
                outs = state.chance_outcomes()
                state.apply_action(int(rng.choices(
                    [o[0] for o in outs], weights=[o[1] for o in outs], k=1)[0]))
                continue
            parsed = (parse_state_repeated_6max(state)
                      if hasattr(state, "dealer_seat") else parse_state_6max(state))
            g = policy._evaluate_gate(parsed, state, rng)
            st = g["street_idx"]
            tot[st] += 1
            n_total += 1
            if g["solve"]:
                solve[st] += 1
                n_solve += 1
            state.apply_action(policy._blueprint_action(parsed, state, rng, "sample"))
        if (h + 1) % 100 == 0:
            print(f"  ... {h + 1}/{args.hands} hands, {n_total} decisions "
                  f"[{time.perf_counter() - t0:.0f}s]")

    f = n_solve / n_total if n_total else float("nan")
    print(f"\n=== gate fire-rate over {args.hands} self-play hands ===")
    print(f"total decisions: {n_total}   solve: {n_solve}   skip: {n_total - n_solve}")
    print(f"EMPIRICAL f = {f:.4f}")
    print(f"\nper-street  (solve / total = rate   |  share of all decisions):")
    for st in sorted(tot):
        t, s = tot[st], solve.get(st, 0)
        print(f"  {_STREET.get(st, st):>8}: {s:>6}/{t:<6} = {s / t:.3f}"
              f"   |  {100 * t / n_total:5.1f}%")
    print(f"\nwall-clock: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
