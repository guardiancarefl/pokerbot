"""
Equity calculator for HUNL card abstraction.

Thin layer over the `treys` library that provides:
  - String-based card I/O ("AsKh") at the API boundary
  - Canonical hole-card class enumeration (169 strategic classes vs 1326 literal combos)
  - Monte Carlo equity calculation with proper card-removal

Used by src/nlhe/abstraction.py to generate equity histograms for EMD clustering.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from treys import Card, Evaluator, Deck

_EVAL = Evaluator()

# ----- Hole card canonicalization -----

RANKS = "23456789TJQKA"
# Rank order index: 0=2, 1=3, ..., 12=A. Higher index = stronger rank.
_RANK_INDEX = {r: i for i, r in enumerate(RANKS)}


@dataclass(frozen=True)
class HoleClass:
    """Strategic equivalence class for a 2-card hold'em hand.

    Three flavors, totaling 169 classes:
      - Pocket pairs:    high == low, suited == False  (13 classes: 22..AA)
      - Suited:          high > low, suited == True    (78 classes: A2s..KQs)
      - Offsuit:         high > low, suited == False   (78 classes: A2o..KQo)
    """
    high: str   # rank char of the higher-or-equal card
    low: str    # rank char of the lower card (or equal rank if pair)
    suited: bool

    def __str__(self) -> str:
        if self.high == self.low:
            return self.high + self.low                  # "AA"
        return self.high + self.low + ("s" if self.suited else "o")  # "AKs" / "AKo"


def all_hole_classes() -> list[HoleClass]:
    """Enumerate the 169 strategically distinct preflop hole-card classes."""
    out: list[HoleClass] = []
    for i, hi in enumerate(RANKS):
        for j, lo in enumerate(RANKS):
            if i < j:
                continue
            if i == j:
                out.append(HoleClass(hi, lo, suited=False))  # pair
            else:
                out.append(HoleClass(hi, lo, suited=True))
                out.append(HoleClass(hi, lo, suited=False))
    assert len(out) == 169, f"expected 169 classes, got {len(out)}"
    return out


def hole_class_to_cards(cls: HoleClass) -> tuple[int, int]:
    """Pick a canonical (card_a, card_b) treys-int representation of a class.

    For pairs: use spades + hearts.
    For suited: both cards are clubs.
    For offsuit: high=spades, low=hearts.
    """
    if cls.high == cls.low:
        return Card.new(cls.high + "s"), Card.new(cls.low + "h")
    if cls.suited:
        return Card.new(cls.high + "c"), Card.new(cls.low + "c")
    return Card.new(cls.high + "s"), Card.new(cls.low + "h")


# ----- Card I/O -----

def hole_class_from_cards(hero: list[int]) -> HoleClass:
    """Inverse of hole_class_to_cards: given 2 treys ints, return the HoleClass.

    Two literal hands may map to the same HoleClass (AsAh and AcAd both -> AA).
    Used by Abstraction.bucket_of() for deterministic preflop lookup.
    """
    if len(hero) != 2:
        raise ValueError(f"hero must be 2 cards, got {len(hero)}")
    cs = [Card.int_to_str(c) for c in hero]
    r1, s1 = cs[0][0], cs[0][1]
    r2, s2 = cs[1][0], cs[1][1]
    # Order by rank descending (high first).
    if _RANK_INDEX[r1] < _RANK_INDEX[r2]:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    suited = (s1 == s2)
    return HoleClass(high=r1, low=r2, suited=suited)


def cards_from_str(s: str) -> list[int]:
    """Parse a string like 'AsKh' or 'AsKh7c' into treys integer cards."""
    if len(s) % 2 != 0:
        raise ValueError(f"cards_from_str: expected even length, got {s!r}")
    return [Card.new(s[i:i + 2]) for i in range(0, len(s), 2)]


def cards_to_str(cards: Iterable[int]) -> str:
    """Inverse of cards_from_str."""
    return "".join(Card.int_to_str(c) for c in cards)


# ----- Equity calculation -----

def equity_vs_random(
    hero: list[int],
    board: list[int] | None = None,
    trials: int = 1000,
    rng: random.Random | None = None,
) -> float:
    """Monte Carlo equity of `hero` vs one uniform-random opponent on `board`.

    Args:
        hero: 2 treys-int cards.
        board: 0, 3, 4, or 5 treys-int board cards already exposed. Defaults to [].
        trials: number of Monte Carlo samples.
        rng: optional random.Random for reproducibility. Defaults to module-global.

    Returns:
        equity in [0, 1]: wins + 0.5 * ties, divided by trials.

    Card removal: hero cards and any exposed board cards are removed from the deck
    before opponent hand and remaining board cards are sampled.
    """
    if board is None:
        board = []
    if len(hero) != 2:
        raise ValueError(f"hero must be 2 cards, got {len(hero)}")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board must have 0/3/4/5 cards, got {len(board)}")

    rng = rng or random
    used = set(hero) | set(board)
    deck_remaining = [c for c in Deck.GetFullDeck() if c not in used]
    cards_to_complete_board = 5 - len(board)

    wins = ties = 0
    for _ in range(trials):
        sample = rng.sample(deck_remaining, 2 + cards_to_complete_board)
        villain = sample[:2]
        rest_of_board = sample[2:]
        full_board = board + rest_of_board
        h = _EVAL.evaluate(full_board, hero)
        v = _EVAL.evaluate(full_board, villain)
        if h < v:
            wins += 1
        elif h == v:
            ties += 1
    return (wins + 0.5 * ties) / trials


def equity_vs_range(
    hero: list[int],
    villain_range: list[tuple[int, int]],
    board: list[int] | None = None,
    trials: int = 1000,
    rng: random.Random | None = None,
) -> float:
    """Monte Carlo equity of `hero` vs uniform-random draw from `villain_range`.

    villain_range is a list of (card_a, card_b) tuples representing the literal
    2-card combos the opponent is presumed to be holding. The MC samples a combo
    uniformly each trial, then a board runout, skipping samples where the chosen
    combo conflicts with the hero or exposed board.

    Used for OCHS-style clustering against representative opponent buckets.
    """
    if board is None:
        board = []
    if len(hero) != 2:
        raise ValueError(f"hero must be 2 cards, got {len(hero)}")
    if not villain_range:
        raise ValueError("villain_range must be non-empty")

    rng = rng or random
    hero_and_board = set(hero) | set(board)
    # Pre-filter range to combos not conflicting with the known cards.
    valid_combos = [
        combo for combo in villain_range
        if combo[0] not in hero_and_board and combo[1] not in hero_and_board
        and combo[0] != combo[1]
    ]
    if not valid_combos:
        raise ValueError("no villain combos compatible with hero+board")

    cards_to_complete_board = 5 - len(board)
    wins = ties = 0
    valid_trials = 0
    for _ in range(trials):
        villain = list(rng.choice(valid_combos))
        used = hero_and_board | set(villain)
        deck_remaining = [c for c in Deck.GetFullDeck() if c not in used]
        rest_of_board = rng.sample(deck_remaining, cards_to_complete_board)
        full_board = board + rest_of_board
        h = _EVAL.evaluate(full_board, hero)
        v = _EVAL.evaluate(full_board, villain)
        if h < v:
            wins += 1
        elif h == v:
            ties += 1
        valid_trials += 1
    return (wins + 0.5 * ties) / valid_trials
