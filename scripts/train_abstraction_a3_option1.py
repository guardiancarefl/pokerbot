#!/usr/bin/env python3
"""Train the Option 1 baseline abstraction for the Track A3 comparison harness.

Option 1: lossless preflop (k=169), k=500 postflop, EMD-histogram distance.
The same algorithm and code path as the default Session-3 abstraction --
only the bucket counts change.

This is one of three abstractions the A3 comparison harness will evaluate:
  - Baseline (Session 3 default): k=20 preflop, k=200 postflop, EMD.
  - Option 1 (this script):       k=169 preflop, k=500 postflop, EMD.
  - Option 4 (TBD):                KrwEmd state-of-the-art.

Output: runs/abstraction_a3_option1_<timestamp>/abstraction.{pkl,json}

Usage:
    python scripts/train_abstraction_a3_option1.py
"""
from __future__ import annotations
import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.nlhe.abstraction import (
    Abstraction,
    StreetAbstraction,
)
from scripts.train_abstraction import train_street

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("a3_option1")

CONFIG = {
    "preflop":  {"n_hands": 0,    "k": 169, "runouts": 400},
    "flop":     {"n_hands": 5000, "k": 500, "runouts": 200},
    "turn":     {"n_hands": 5000, "k": 500, "runouts": 200},
    "river":    {"n_hands": 5000, "k": 500, "runouts": 200},
    "bins": 50,
    "seed": 2026,
    "kmedoids_max_iter": 100,
    "tag": "a3_option1",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None,
                        help="output directory (default runs/abstraction_a3_option1_<timestamp>)")
    parser.add_argument("--streets", default=None,
                        help="comma-separated street list to train (default all four)")
    args = parser.parse_args()

    if args.out:
        out_dir = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"runs/abstraction_{CONFIG['tag']}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    streets_to_train = (args.streets.split(",") if args.streets
                        else ["preflop", "flop", "turn", "river"])

    log.info(f"output dir: {out_dir}")
    log.info(f"config: {CONFIG}")

    rng = random.Random(CONFIG["seed"])
    abst = Abstraction()
    t_start = time.time()

    for street in streets_to_train:
        sc = CONFIG[street]
        sa = train_street(
            street=street,
            n_hands=sc["n_hands"],
            k=sc["k"],
            runouts=sc["runouts"],
            bins=CONFIG["bins"],
            rng=rng,
            kmedoids_max_iter=CONFIG["kmedoids_max_iter"],
        )
        abst.streets[street] = sa
        # Save incrementally so a crash doesnt lose all progress.
        abst.save(out_dir / "abstraction.pkl")
        log.info(f"  saved partial abstraction ({len(abst.streets)} streets)")

    # Companion JSON for human inspection.
    summary = {
        "config": CONFIG,
        "streets": {
            s: {"k": sa.k, "bins": sa.bins, "n_medoids": len(sa.medoid_hands)}
            for s, sa in abst.streets.items()
        },
        "total_runtime_min": (time.time() - t_start) / 60.0,
    }
    (out_dir / "abstraction.json").write_text(json.dumps(summary, indent=2))
    log.info(f"=== done in {(time.time() - t_start) / 60.0:.1f} min ===")
    log.info(f"output: {out_dir}")


if __name__ == "__main__":
    main()
