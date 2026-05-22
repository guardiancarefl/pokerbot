"""
EMD-based card abstraction for HUNL.

Pipeline per street:
  1. Sample canonical (hand, board) combinations.
  2. For each, compute an equity histogram by Monte Carlo over river runouts.
  3. Cluster the histograms with k-medoids using Earth Mover's Distance.
  4. Persist the medoids + bucket labels for inference-time bucket assignment.

This module is the library. Training is driven by scripts/train_abstraction.py.
"""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import wasserstein_distance
from treys import Card, Deck, Evaluator

from src.nlhe.equity import (
    HoleClass,
    all_hole_classes,
    cards_from_str,
    cards_to_str,
    hole_class_to_cards,
)

_EVAL = Evaluator()

STREETS = ("preflop", "flop", "turn", "river")
BOARD_SIZE = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


# ----- Histogram generation -----

def compute_hand_histogram(
    hero: list[int],
    board: list[int],
    runouts: int = 200,
    bins: int = 50,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Compute the equity histogram of `hero` on `board`.

    For each runout sample, completes the board with random remaining cards,
    samples a random opponent hand from the remaining deck, and records hero's
    equity (1.0 win / 0.5 tie / 0.0 loss). Returns a normalized histogram of
    those equity values.

    Args:
        hero: 2 treys-int cards.
        board: 0, 3, 4, or 5 treys-int board cards already exposed.
        runouts: number of MC samples.
        bins: histogram resolution; equity space [0,1] is divided into this many bins.
        rng: optional random.Random for reproducibility.

    Returns:
        A length-`bins` numpy array summing to 1.0.
    """
    if len(hero) != 2:
        raise ValueError(f"hero must be 2 cards, got {len(hero)}")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board must have 0/3/4/5 cards, got {len(board)}")

    rng = rng or random
    used = set(hero) | set(board)
    deck_remaining = [c for c in Deck.GetFullDeck() if c not in used]
    cards_to_complete = 5 - len(board)

    equities = np.empty(runouts, dtype=np.float32)
    for i in range(runouts):
        # Need 2 villain cards + remaining board cards, all distinct from used
        sample = rng.sample(deck_remaining, 2 + cards_to_complete)
        villain = sample[:2]
        rest_of_board = sample[2:]
        full_board = board + rest_of_board
        h = _EVAL.evaluate(full_board, hero)
        v = _EVAL.evaluate(full_board, villain)
        if h < v:
            equities[i] = 1.0
        elif h == v:
            equities[i] = 0.5
        else:
            equities[i] = 0.0

    # np.histogram returns counts; convert to probability vector.
    hist, _ = np.histogram(equities, bins=bins, range=(0.0, 1.0))
    return hist.astype(np.float32) / runouts


# ----- EMD distance -----

# Precomputed bin centers, used as the support for wasserstein_distance.
def _bin_centers(bins: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, bins + 1)
    return (edges[:-1] + edges[1:]) / 2.0


def emd_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Earth Mover's Distance between two 1-D probability histograms.

    Both histograms are assumed to have the same length and same bin support
    (uniform partition of [0,1]).
    """
    if hist_a.shape != hist_b.shape:
        raise ValueError(f"shape mismatch: {hist_a.shape} vs {hist_b.shape}")
    centers = _bin_centers(len(hist_a))
    # wasserstein_distance with explicit weights computes 1-D EMD directly.
    return float(wasserstein_distance(centers, centers, u_weights=hist_a, v_weights=hist_b))


def pairwise_emd(histograms: np.ndarray) -> np.ndarray:
    """Compute full pairwise EMD distance matrix for a stack of histograms.

    Args:
        histograms: (N, bins) array.

    Returns:
        (N, N) symmetric distance matrix with zero diagonal.
    """
    n = histograms.shape[0]
    centers = _bin_centers(histograms.shape[1])
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = wasserstein_distance(centers, centers,
                                     u_weights=histograms[i],
                                     v_weights=histograms[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


# ----- K-medoids clustering -----

def _kmeans_plus_plus_init(
    dist: np.ndarray, k: int, rng: random.Random
) -> list[int]:
    """k-means++ style spread initialization using precomputed distance matrix.

    Each returned medoid index is guaranteed unique. When k == n, returns
    list(range(n)) directly (every point is its own medoid -- the trivially
    lossless case). Otherwise iteratively picks points with probability
    proportional to squared distance from the nearest existing medoid,
    EXCLUDING points already picked.
    """
    n = dist.shape[0]
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    # Trivial lossless case: every point is its own medoid.
    if k == n:
        return list(range(n))
    medoids = [rng.randrange(n)]
    for _ in range(k - 1):
        # For each point, its squared distance to the nearest existing medoid.
        d_to_nearest = np.min(dist[medoids, :], axis=0) ** 2
        # CRITICAL: exclude already-picked medoids so we never sample with replacement.
        # Without this, rng.choices(weights=probs) can land on an already-picked
        # index, producing duplicate medoids and leaving some points unrepresented.
        d_to_nearest[medoids] = 0.0
        total = d_to_nearest.sum()
        if total == 0.0:
            # All remaining points are duplicates of existing medoids; pick any unused.
            unused = [i for i in range(n) if i not in medoids]
            medoids.append(rng.choice(unused))
            continue
        probs = d_to_nearest / total
        chosen = rng.choices(range(n), weights=probs.tolist(), k=1)[0]
        medoids.append(chosen)
    return medoids


def kmedoids(
    dist: np.ndarray,
    k: int,
    max_iter: int = 100,
    rng: random.Random | None = None,
    verbose: bool = False,
) -> tuple[list[int], np.ndarray, float]:
    """Partitioning Around Medoids (PAM) clustering on a precomputed distance matrix.

    Args:
        dist: (N, N) symmetric distance matrix.
        k: number of clusters.
        max_iter: max iterations of medoid update.
        rng: optional random.Random for init.
        verbose: print per-iteration cost.

    Returns:
        (medoid_indices, labels, total_cost)
            medoid_indices: list of k indices into the input points
            labels: (N,) array of cluster assignment in [0, k)
            total_cost: sum of distances from each point to its assigned medoid
    """
    rng = rng or random
    n = dist.shape[0]
    medoids = _kmeans_plus_plus_init(dist, k, rng)

    def assign_and_cost(medoid_idxs: list[int]) -> tuple[np.ndarray, float]:
        sub = dist[medoid_idxs, :]                # (k, N)
        labels = np.argmin(sub, axis=0)           # (N,)
        cost = float(sub[labels, np.arange(n)].sum())
        return labels, cost

    labels, cost = assign_and_cost(medoids)
    if verbose:
        print(f"  iter 0  cost={cost:.4f}")

    for it in range(1, max_iter + 1):
        improved = False
        # For each cluster, find the point in the cluster that minimizes total
        # intra-cluster distance. That point becomes the new medoid.
        new_medoids = list(medoids)
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            sub_dist = dist[np.ix_(members, members)]
            within_costs = sub_dist.sum(axis=1)
            best_local = members[np.argmin(within_costs)]
            if best_local != new_medoids[c]:
                new_medoids[c] = int(best_local)
                improved = True

        new_labels, new_cost = assign_and_cost(new_medoids)
        if verbose:
            print(f"  iter {it}  cost={new_cost:.4f}  changed={improved}")

        if not improved or new_cost >= cost - 1e-9:
            medoids = new_medoids
            labels = new_labels
            cost = new_cost
            break
        medoids = new_medoids
        labels = new_labels
        cost = new_cost

    return medoids, labels, cost


# ----- Canonical hand sampling per street -----

def sample_street_hands(
    street: str,
    n_samples: int,
    rng: random.Random,
) -> list[tuple[list[int], list[int]]]:
    """Sample (hero, board) pairs for a given street.

    Preflop returns all 169 canonical hole classes (n_samples ignored).
    Postflop streets sample n_samples random (hole, board) combos with proper
    card-removal.
    """
    if street == "preflop":
        return [(list(hole_class_to_cards(c)), []) for c in all_hole_classes()]

    n_board = BOARD_SIZE[street]
    out = []
    full_deck = Deck.GetFullDeck()
    for _ in range(n_samples):
        cards = rng.sample(full_deck, 2 + n_board)
        hero = cards[:2]
        board = cards[2:]
        out.append((hero, board))
    return out


# ----- Top-level Abstraction object -----

@dataclass
class StreetAbstraction:
    """Trained abstraction for one street."""
    street: str
    bins: int                           # histogram bin count
    medoid_histograms: np.ndarray       # (k, bins)
    medoid_hands: list[tuple[list[int], list[int]]]  # (hero, board) pairs for each medoid

    @property
    def k(self) -> int:
        return self.medoid_histograms.shape[0]


@dataclass
class Abstraction:
    """Full HUNL card abstraction across all streets."""
    streets: dict[str, StreetAbstraction] = field(default_factory=dict)

    def bucket_of(
        self,
        hero: list[int],
        board: list[int],
        runouts: int = 200,
        rng: random.Random | None = None,
    ) -> int:
        """Assign a bucket index to a (hero, board) pair on its street."""
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(board)]
        sa = self.streets[street]
        h = compute_hand_histogram(hero, board, runouts=runouts, bins=sa.bins, rng=rng)
        # Distance to each medoid, pick the min.
        best = 0
        best_d = float("inf")
        centers = _bin_centers(sa.bins)
        for i in range(sa.k):
            d = wasserstein_distance(
                centers, centers,
                u_weights=h, v_weights=sa.medoid_histograms[i],
            )
            if d < best_d:
                best_d = d
                best = i
        return best

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pickle is fine for now — tightly coupled to numpy + treys-int cards.
        with open(path, "wb") as f:
            pickle.dump(self, f)
        # Companion metadata in human-readable JSON.
        meta = {
            street: {
                "k": sa.k,
                "bins": sa.bins,
                "medoid_count": len(sa.medoid_hands),
            }
            for street, sa in self.streets.items()
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Abstraction":
        with open(path, "rb") as f:
            return pickle.load(f)
