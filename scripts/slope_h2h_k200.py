"""Read-only slope / head-to-head check for the k=200 real-ante run.

Plays paired-CRN hands between two checkpoints (same dealer/cards across
the pair, seats swapped) and reports mean chip delta + sem + 2σ window.

Convention here: returns A's mean chip delta, so:
  - slope check  (latest as A, earlier as B): >2σ positive  = latest still climbing
  - peak-pin H2H (earlier as A, latest  as B): positive    = earlier is peak

Pure observability — does not modify training, does not touch GPU.
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.nlhe.game_strings import TournamentStructure
from scripts.throwaway_query_real_ante import _RealAnteStructure
from scripts.eval_6max_self_play import _load_solver, play_one_hand


def paired_crn_margins(solver_a, solver_b, structure, n_pairs, seed_base):
    margins = []
    t0 = time.time()
    for i in range(n_pairs):
        seed = seed_base + i
        seats_A = [solver_a, solver_b, solver_a, solver_b, solver_a, solver_b]
        rng = random.Random(seed)
        try:
            result_A = play_one_hand(seats_A, structure, rng,
                                     num_paid=3, mode="sample")
        except Exception:
            continue
        delta_A = sum(result_A["seat_to_equity_delta"][s] for s in (0, 2, 4))

        seats_B = [solver_b, solver_a, solver_b, solver_a, solver_b, solver_a]
        rng = random.Random(seed)
        try:
            result_B = play_one_hand(seats_B, structure, rng,
                                     num_paid=3, mode="sample")
        except Exception:
            continue
        delta_B = sum(result_B["seat_to_equity_delta"][s] for s in (1, 3, 5))

        paired_delta = (delta_A + delta_B) / 2.0
        margins.append(paired_delta)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_pairs - i - 1) / rate
            print(f"  pair {i+1}/{n_pairs}  elapsed={elapsed:.0f}s  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s  "
                  f"running_mean={np.mean(margins):+.4f}",
                  flush=True)
    return np.array(margins, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True, help="A side checkpoint")
    ap.add_argument("--ckpt-b", required=True, help="B side checkpoint")
    ap.add_argument("--abstraction",
                    default="runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    ap.add_argument("--structure",
                    default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--pairs", type=int, default=5000,
                    help="number of paired CRN hand-pairs (each pair = 2 underlying hands)")
    ap.add_argument("--seed-base", type=int, default=20260607,
                    help="seed base disjoint from monitor's seed_base+1000 family")
    ap.add_argument("--label", default="slope")
    args = ap.parse_args()

    print(f"=== {args.label} ===")
    print(f"  A: {args.ckpt_a}")
    print(f"  B: {args.ckpt_b}")
    print(f"  pairs={args.pairs}  seed_base={args.seed_base}")
    print(f"  (positive margin = A beats B)", flush=True)

    print("Loading abstraction...", flush=True)
    with open(args.abstraction, "rb") as f:
        abstraction = pickle.load(f)
    base = TournamentStructure.from_yaml(args.structure)
    structure = _RealAnteStructure(base)

    print("Loading solver A...", flush=True)
    solver_a = _load_solver(args.ckpt_a, abstraction, structure)
    solver_a.tournament_structure = structure
    print("Loading solver B...", flush=True)
    solver_b = _load_solver(args.ckpt_b, abstraction, structure)
    solver_b.tournament_structure = structure

    margins = paired_crn_margins(solver_a, solver_b, structure,
                                 n_pairs=args.pairs,
                                 seed_base=args.seed_base)
    n = len(margins)
    mean = float(np.mean(margins))
    sd = float(np.std(margins, ddof=1))
    sem = sd / np.sqrt(n)
    t_stat = mean / sem if sem > 0 else float("nan")

    print()
    print(f"=== RESULT: {args.label} ===")
    print(f"  n_paired_hands = {n}")
    print(f"  mean A-B chip delta = {mean:+.4f}")
    print(f"  sd = {sd:.4f}   sem = {sem:.4f}")
    print(f"  t = mean / sem = {t_stat:+.3f}")
    print(f"  95% CI ≈ [{mean - 1.96*sem:+.4f}, {mean + 1.96*sem:+.4f}]")
    print(f"  2σ window = [{mean - 2*sem:+.4f}, {mean + 2*sem:+.4f}]")
    if t_stat > 2:
        verdict = "A SIGNIFICANTLY BEATS B (>2σ positive)"
    elif t_stat < -2:
        verdict = "B SIGNIFICANTLY BEATS A (<-2σ)"
    else:
        verdict = "WITHIN 2σ OF ZERO — no significant difference"
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
