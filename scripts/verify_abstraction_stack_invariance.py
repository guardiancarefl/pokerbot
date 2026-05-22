"""Verify that the trained card abstraction is stack-agnostic.

EMD on equity histograms is a pure function of (hero cards, board cards):
equity is a property of cards, not chips. This script makes that explicit by:
  1. Inspecting Abstraction.bucket_of()'s signature for stack-related args.
  2. Empirically: pick 20 (hand, board) combinations across all 4 streets,
     print their bucket assignments. The point is just to demonstrate the
     module accepts only card inputs.

Usage:
    python -m scripts.verify_abstraction_stack_invariance \
        --abstraction runs/abstraction_20260521_223018/abstraction.pkl
"""

from __future__ import annotations

import argparse
import inspect
import random

from src.nlhe.abstraction import Abstraction
from src.nlhe.equity import cards_from_str


# 20 (hero, board) combinations — 5 per street.
CASES = [
    # Preflop (no board)
    ("preflop", "AsAh", ""),
    ("preflop", "KdKc", ""),
    ("preflop", "AhKs", ""),
    ("preflop", "7c2d", ""),
    ("preflop", "9s8s", ""),
    # Flop
    ("flop", "AsAh", "2c7d9s"),
    ("flop", "AdKs", "QcJhTd"),       # broadway straight
    ("flop", "JhTh", "9h8h2c"),       # combo draw
    ("flop", "5c5d", "3hKsAd"),       # underpair to overcards
    ("flop", "7s2c", "Kd9h4d"),       # garbage
    # Turn
    ("turn", "AcKc", "2c7c9sTh"),     # nut flush draw + overs
    ("turn", "QdQh", "JsTc4d2s"),     # overpair
    ("turn", "8h7h", "9h6c2dKs"),     # open-ended on dry turn
    ("turn", "AhAd", "5c5d5hKs"),     # full house
    ("turn", "3c2c", "7h9dKsAh"),     # whiff
    # River
    ("river", "AsKs", "QsJsTs2c5h"),  # royal flush
    ("river", "JhJd", "2c5d8s9hKc"),  # one pair, overcard
    ("river", "AdKd", "AsKh7c2d4s"),  # two pair top
    ("river", "9c9d", "AhKsQsJd5c"),  # bricked pair vs board
    ("river", "5h4h", "6h7h8h2dTc"),  # made flush
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--abstraction",
        default="runs/abstraction_20260521_223018/abstraction.pkl",
        help="Path to trained abstraction.pkl",
    )
    ap.add_argument("--runouts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    sig = inspect.signature(Abstraction.bucket_of)
    params = list(sig.parameters.keys())
    print(f"Abstraction.bucket_of signature: {params}")
    stack_params = [p for p in params if "stack" in p.lower() or "chip" in p.lower()
                    or "pot" in p.lower() or "bb" in p.lower()]
    print(f"Stack/chip/pot/bb-related parameters: {stack_params or 'NONE'}")

    abs_obj = Abstraction.load(args.abstraction)
    rng = random.Random(args.seed)

    print(f"\nLoaded abstraction from {args.abstraction}")
    for street, sa in abs_obj.streets.items():
        print(f"  street={street}  k={sa.k}  bins={sa.bins}")

    print(f"\nBucket assignments ({len(CASES)} cases, runouts={args.runouts}, seed={args.seed}):")
    print(f"  {'street':<8} {'hero':<6} {'board':<12} -> bucket")
    for street, hero_s, board_s in CASES:
        hero = cards_from_str(hero_s)
        board = cards_from_str(board_s) if board_s else []
        b = abs_obj.bucket_of(hero, board, runouts=args.runouts, rng=rng)
        print(f"  {street:<8} {hero_s:<6} {board_s:<12} -> {b}")

    if not stack_params:
        print("\nOK: Abstraction.bucket_of() takes only (hero, board, runouts, rng).")
        print("    No stack/chip/pot/bb inputs. Abstraction is card-based and stack-agnostic.")
    else:
        print(f"\nWARNING: found stack-related params: {stack_params}")


if __name__ == "__main__":
    main()
