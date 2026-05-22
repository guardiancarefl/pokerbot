"""Compute bucket-equity table and population quantiles for archetype design.

Reads a trained card abstraction (.pkl), derives per-bucket mean equity from
medoid histograms, samples random hands to estimate the population equity
distribution per street, and writes the result as JSON for the archetype
module to consume.

Output schema (matches src/nlhe/archetypes.py:EquityCalibration.load):
    {
      "abstraction_path": str,
      "n_samples_preflop": int,
      "n_samples_postflop": int,
      "runouts": int,
      "bucket_equity": {street: list[float]},
      "quantiles": {street: {"q05": float, ..., "q95": float}}
    }

Rerun this script whenever the underlying card abstraction changes (e.g.,
after A3 swaps EMD for OCHS). Archetypes will adapt automatically since they
read thresholds from the JSON at runtime.

Usage:
    python3 scripts/analyze_bucket_equity.py \
        --abstraction runs/abstraction_20260521_223018/abstraction.pkl \
        --out runs/archetype_design/bucket_equity_analysis.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.nlhe.abstraction import Abstraction
from src.nlhe.equity import cards_from_str


# Defaults match the run that produced the locked archetype thresholds in
# Session 6. Change with care -- archetype tests assume this calibration shape.
DEFAULT_N_SAMPLES_PREFLOP = 5000
DEFAULT_N_SAMPLES_POSTFLOP = 2000
DEFAULT_RUNOUTS = 30
DEFAULT_SEED = 2026
QUANTILE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                   0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def compute_bucket_equity_table(abst: Abstraction) -> dict[str, list[float]]:
    """Mean equity per bucket per street, derived from medoid histograms."""
    table = {}
    for street, sa in abst.streets.items():
        bin_midpoints = (np.arange(sa.bins) + 0.5) / sa.bins
        sums = sa.medoid_histograms.sum(axis=1, keepdims=True)
        normalized = sa.medoid_histograms / np.clip(sums, 1e-12, None)
        bucket_equity = (normalized * bin_midpoints).sum(axis=1)
        table[street] = bucket_equity.tolist()
    return table


def sample_preflop_equities(abst, n, rng, bucket_equity, runouts):
    ranks = "23456789TJQKA"
    suits = "shdc"
    deck = [r + s for r in ranks for s in suits]
    out = []
    for _ in range(n):
        pair = rng.sample(deck, 2)
        cards = cards_from_str("".join(pair))
        b = abst.bucket_of(cards, [], runouts, rng)
        out.append(bucket_equity[b])
    return out


def sample_postflop_equities(abst, street, n, rng, bucket_equity, runouts):
    ranks = "23456789TJQKA"
    suits = "shdc"
    deck = [r + s for r in ranks for s in suits]
    board_n = {"flop": 3, "turn": 4, "river": 5}[street]
    out = []
    for _ in range(n):
        all_cards = rng.sample(deck, 2 + board_n)
        hero = cards_from_str("".join(all_cards[:2]))
        board = cards_from_str("".join(all_cards[2:]))
        b = abst.bucket_of(hero, board, runouts, rng)
        out.append(bucket_equity[b])
    return out


def quantile_table(equities):
    arr = np.array(equities)
    return {f"q{int(q * 100):02d}": float(np.quantile(arr, q))
            for q in QUANTILE_LEVELS}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--abstraction", required=True,
                        help="path to abstraction.pkl")
    parser.add_argument("--out", required=True,
                        help="output JSON path")
    parser.add_argument("--n-preflop", type=int, default=DEFAULT_N_SAMPLES_PREFLOP)
    parser.add_argument("--n-postflop", type=int, default=DEFAULT_N_SAMPLES_POSTFLOP)
    parser.add_argument("--runouts", type=int, default=DEFAULT_RUNOUTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading abstraction from {args.abstraction}...")
    abst = Abstraction.load(args.abstraction)
    print(f"Streets: {list(abst.streets.keys())}")
    print()

    bucket_equity = compute_bucket_equity_table(abst)
    print("=== Bucket equity table ===")
    for street in ["preflop", "flop", "turn", "river"]:
        if street not in bucket_equity:
            continue
        eqs = bucket_equity[street]
        print(f"  {street:8} k={len(eqs):3d}  "
              f"min={min(eqs):.3f}  max={max(eqs):.3f}")
    print()

    rng = random.Random(args.seed)
    quantiles = {}
    for street in ["preflop", "flop", "turn", "river"]:
        if street not in bucket_equity:
            continue
        n = args.n_preflop if street == "preflop" else args.n_postflop
        print(f"Sampling {n} {street} hands...")
        if street == "preflop":
            eqs = sample_preflop_equities(abst, n, rng, bucket_equity[street], args.runouts)
        else:
            eqs = sample_postflop_equities(abst, street, n, rng,
                                            bucket_equity[street], args.runouts)
        quantiles[street] = quantile_table(eqs)

    print()
    print("=== Population equity quantiles ===")
    cols = ("q05", "q15", "q25", "q50", "q75", "q85", "q95")
    print(f"{'street':<10} " + " ".join(f"{c:>6}" for c in cols))
    for street in ["preflop", "flop", "turn", "river"]:
        if street not in quantiles:
            continue
        q = quantiles[street]
        print(f"{street:<10} " + " ".join(f"{q[c]:>6.3f}" for c in cols))

    output = {
        "abstraction_path": args.abstraction,
        "n_samples_preflop": args.n_preflop,
        "n_samples_postflop": args.n_postflop,
        "runouts": args.runouts,
        "bucket_equity": bucket_equity,
        "quantiles": quantiles,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
