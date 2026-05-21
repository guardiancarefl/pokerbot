"""Inspect a trained abstraction: bucket sizes, medoid hands, sample contents.

Usage:
    python -m scripts.inspect_abstraction <path-to-abstraction.pkl>
    python -m scripts.inspect_abstraction runs/abstraction_<timestamp>/abstraction.pkl
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from src.nlhe.abstraction import Abstraction, compute_hand_histogram
from src.nlhe.equity import cards_to_str


def inspect_street(abst: Abstraction, street: str, n_samples: int = 5) -> None:
    sa = abst.streets[street]
    print(f"\n{'='*60}")
    print(f"Street: {street}")
    print(f"  bins: {sa.bins}")
    print(f"  k (medoids): {sa.k}")
    print(f"{'='*60}")

    centers = np.linspace(0.01, 0.99, sa.bins)
    rows = []
    for i in range(sa.k):
        hero, board = sa.medoid_hands[i]
        hist = sa.medoid_histograms[i]
        mean_eq = float((hist * centers).sum())
        # Concentration: how much mass in the top-5 bins vs bottom-5.
        top5 = float(hist[-5:].sum())
        bot5 = float(hist[:5].sum())
        rows.append({
            "idx": i,
            "hero": cards_to_str(hero),
            "board": cards_to_str(board) if board else "(preflop)",
            "mean_eq": mean_eq,
            "top5": top5,
            "bot5": bot5,
        })

    # Sort by mean equity for readability.
    rows.sort(key=lambda r: r["mean_eq"])
    print(f"\n  Medoids sorted by mean equity:")
    print(f"  {'idx':>4} {'hero':>6} {'board':>15} {'mean_eq':>8} {'top5':>6} {'bot5':>6}")
    print(f"  {'-'*4} {'-'*6} {'-'*15} {'-'*8} {'-'*6} {'-'*6}")
    for r in rows:
        print(f"  {r['idx']:>4} {r['hero']:>6} {r['board']:>15} {r['mean_eq']:>8.3f} {r['top5']:>6.3f} {r['bot5']:>6.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="path to abstraction.pkl")
    p.add_argument(
        "--streets",
        default="preflop,flop,turn,river",
        help="comma-separated streets to inspect",
    )
    args = p.parse_args()

    abst = Abstraction.load(args.path)
    print(f"Loaded abstraction from {args.path}")
    print(f"Available streets: {list(abst.streets.keys())}")

    for street in args.streets.split(","):
        if street in abst.streets:
            inspect_street(abst, street)
        else:
            print(f"  (skipping {street}: not trained in this artifact)")


if __name__ == "__main__":
    main()
