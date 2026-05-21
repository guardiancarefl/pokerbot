"""Train HUNL card abstraction (EMD-based, ~200 buckets per street).

Usage:
    python scripts/train_abstraction.py [--config configs/abstraction_default.yaml]

Default config is hardcoded; YAML loader can be added later if we need
parameter sweeps. Output: a single .pkl + companion .json in runs/abstraction_<timestamp>/.
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
    BOARD_SIZE,
    Abstraction,
    StreetAbstraction,
    compute_hand_histogram,
    kmedoids,
    pairwise_emd,
    sample_street_hands,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_abstraction")


# Default training configuration.
DEFAULT_CONFIG = {
    "preflop":  {"n_hands": 0,    "k": 20,  "runouts": 400},
    "flop":     {"n_hands": 1500, "k": 200, "runouts": 200},
    "turn":     {"n_hands": 1500, "k": 200, "runouts": 200},
    "river":    {"n_hands": 1500, "k": 200, "runouts": 200},
    "bins": 50,
    "seed": 2026,
    "kmedoids_max_iter": 100,
}


def train_street(
    street: str,
    n_hands: int,
    k: int,
    runouts: int,
    bins: int,
    rng: random.Random,
    kmedoids_max_iter: int,
) -> StreetAbstraction:
    log.info(f"=== Training {street} (n_hands={n_hands or 169}, k={k}, runouts={runouts}) ===")

    # Sample
    t0 = time.time()
    hands = sample_street_hands(street, n_samples=n_hands, rng=rng)
    log.info(f"  sampled {len(hands)} hands in {time.time()-t0:.1f}s")

    # Cap k at len(hands); preflop only has 169 canonical classes.
    effective_k = min(k, len(hands))
    if effective_k < k:
        log.info(f"  capping k from {k} to {effective_k} (only {len(hands)} hands available)")

    # Histograms
    t0 = time.time()
    histograms = np.stack([
        compute_hand_histogram(hero, board, runouts=runouts, bins=bins, rng=rng)
        for hero, board in hands
    ])
    log.info(f"  computed {len(hands)} histograms ({bins} bins, {runouts} runouts) in {time.time()-t0:.1f}s")

    # Pairwise EMD
    t0 = time.time()
    dist = pairwise_emd(histograms)
    log.info(f"  pairwise EMD ({dist.shape[0]}x{dist.shape[0]}) in {time.time()-t0:.1f}s")

    # K-medoids
    t0 = time.time()
    medoid_idxs, labels, cost = kmedoids(
        dist, k=effective_k, max_iter=kmedoids_max_iter, rng=rng, verbose=False
    )
    log.info(f"  k-medoids (k={effective_k}) converged to cost={cost:.4f} in {time.time()-t0:.1f}s")
    log.info(f"  label distribution: min={np.bincount(labels).min()} max={np.bincount(labels).max()} median={int(np.median(np.bincount(labels)))}")

    return StreetAbstraction(
        street=street,
        bins=bins,
        medoid_histograms=histograms[medoid_idxs],
        medoid_hands=[hands[i] for i in medoid_idxs],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="output directory (default runs/abstraction_<timestamp>)")
    parser.add_argument(
        "--streets",
        default="preflop,flop,turn,river",
        help="comma-separated street list to train (default all four)",
    )
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    rng = random.Random(cfg["seed"])

    if args.out:
        out_dir = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"runs/abstraction_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"output dir: {out_dir}")

    streets_to_train = args.streets.split(",")
    abst = Abstraction()
    total_start = time.time()

    for street in streets_to_train:
        if street not in BOARD_SIZE:
            raise ValueError(f"unknown street: {street}")
        s_cfg = cfg[street]
        sa = train_street(
            street=street,
            n_hands=s_cfg["n_hands"],
            k=s_cfg["k"],
            runouts=s_cfg["runouts"],
            bins=cfg["bins"],
            rng=rng,
            kmedoids_max_iter=cfg["kmedoids_max_iter"],
        )
        abst.streets[street] = sa

    total_t = time.time() - total_start
    log.info(f"=== Total training time: {total_t/60:.1f} min ===")

    # Persist.
    pkl_path = out_dir / "abstraction.pkl"
    abst.save(pkl_path)
    log.info(f"saved: {pkl_path}")
    log.info(f"saved: {pkl_path.with_suffix('.json')}")

    # Save config alongside for reproducibility.
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    log.info(f"saved: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
