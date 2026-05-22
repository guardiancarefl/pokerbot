"""Tests for src/nlhe/abstraction.py and src/nlhe/equity.py.

Focus: the deterministic preflop_lookup path added in response to the
bucket_of() non-determinism finding (see DECISIONS.md).
"""
from __future__ import annotations
import random

import numpy as np
import pytest

from src.nlhe.abstraction import (
    Abstraction,
    StreetAbstraction,
)
from src.nlhe.equity import (
    HoleClass,
    all_hole_classes,
    cards_from_str,
    hole_class_from_cards,
    hole_class_to_cards,
)


# ---- equity.hole_class_from_cards ----

def test_hole_class_from_cards_pair():
    cards = cards_from_str("AsAh")
    hc = hole_class_from_cards(cards)
    assert str(hc) == "AA"


def test_hole_class_from_cards_suited():
    cards = cards_from_str("AsKs")
    hc = hole_class_from_cards(cards)
    assert str(hc) == "AKs"


def test_hole_class_from_cards_offsuit():
    cards = cards_from_str("AsKh")
    hc = hole_class_from_cards(cards)
    assert str(hc) == "AKo"


def test_hole_class_from_cards_low_first():
    # Cards in low-then-high order should canonicalize to high-then-low.
    cards = cards_from_str("2sAh")
    hc = hole_class_from_cards(cards)
    assert str(hc) == "A2o"


def test_hole_class_from_cards_rejects_wrong_count():
    cards = cards_from_str("AsKhQc")
    with pytest.raises(ValueError):
        hole_class_from_cards(cards)


def test_hole_class_round_trip_all_169():
    """Every canonical HoleClass survives to_cards -> from_cards round trip."""
    for hc in all_hole_classes():
        cards = hole_class_to_cards(hc)
        recovered = hole_class_from_cards(list(cards))
        assert str(recovered) == str(hc), f"round-trip failed for {hc}: got {recovered}"


# ---- StreetAbstraction.preflop_lookup field ----

def _make_lookup_preflop_sa(lookup: dict[str, int]) -> StreetAbstraction:
    """Build a minimal StreetAbstraction for preflop with a lookup table.

    Histograms and medoid_hands are placeholder; the test only exercises the
    preflop_lookup fast path which never touches the histograms.
    """
    return StreetAbstraction(
        street="preflop",
        bins=50,
        medoid_histograms=np.zeros((3, 50), dtype=np.float32),
        medoid_hands=[(cards_from_str("AsAh"), []),
                      (cards_from_str("KsKh"), []),
                      (cards_from_str("2s3h"), [])],
        preflop_lookup=lookup,
    )


def test_streetabstraction_defaults_lookup_to_none():
    sa = StreetAbstraction(
        street="preflop",
        bins=50,
        medoid_histograms=np.zeros((1, 50), dtype=np.float32),
        medoid_hands=[(cards_from_str("AsAh"), [])],
    )
    assert sa.preflop_lookup is None


# ---- Abstraction.bucket_of preflop fast path ----

def test_bucket_of_uses_preflop_lookup_when_present():
    """If preflop_lookup is present, bucket_of returns dict[canonical] directly."""
    lookup = {"AA": 0, "KK": 1, "32o": 2}
    a = Abstraction(streets={"preflop": _make_lookup_preflop_sa(lookup)})

    # AA must always return 0, no matter the rng or runouts.
    assert a.bucket_of(cards_from_str("AsAh"), [], runouts=50) == 0
    assert a.bucket_of(cards_from_str("AcAd"), [], runouts=50) == 0  # same class, different literals
    assert a.bucket_of(cards_from_str("KsKh"), [], runouts=50) == 1
    assert a.bucket_of(cards_from_str("3s2h"), [], runouts=50) == 2  # canonicalizes to 32o


def test_bucket_of_lookup_deterministic_across_calls():
    """The bucket_of non-determinism finding motivated this whole change.
    Verify deterministically: same input -> same output across N calls.
    """
    lookup = {str(hc): i for i, hc in enumerate(all_hole_classes())}
    sa = StreetAbstraction(
        street="preflop",
        bins=50,
        medoid_histograms=np.zeros((169, 50), dtype=np.float32),
        medoid_hands=[(list(hole_class_to_cards(hc)), []) for hc in all_hole_classes()],
        preflop_lookup=lookup,
    )
    a = Abstraction(streets={"preflop": sa})

    for label in ["AsAh", "KsKh", "QsQh", "JsJh", "TsTh", "AsKs", "5h6h", "7s2h"]:
        cards = cards_from_str(label)
        # Different rng seeds and runout counts should not affect the answer.
        buckets = set()
        for trial in range(5):
            rng = random.Random(trial * 31337)
            buckets.add(a.bucket_of(cards, [], runouts=50, rng=rng))
        assert len(buckets) == 1, f"{label}: got multiple buckets {buckets} across trials"


# ---- Abstraction.bucket_of postflop deterministic-MC path ----

def test_bucket_of_postflop_deterministic_across_calls():
    """The MC fallback path is now deterministic: same (hero, board) -> same
    bucket across calls regardless of what rng the caller passes.

    Uses a small synthetic flop abstraction with real (non-zero) histograms so
    the MC distance computation has well-defined v_weights.
    """
    # Build a tiny realistic flop abstraction with 3 medoids.
    bins = 50
    # Generate three real histograms with different MC seeds so they differ.
    from src.nlhe.abstraction import compute_hand_histogram
    rng_a = random.Random(1)
    h1 = compute_hand_histogram(cards_from_str("AsAh"), cards_from_str("2c3d4h"), runouts=50, bins=bins, rng=rng_a)
    h2 = compute_hand_histogram(cards_from_str("KsKh"), cards_from_str("2c3d4h"), runouts=50, bins=bins, rng=random.Random(2))
    h3 = compute_hand_histogram(cards_from_str("7s2h"), cards_from_str("2c3d4h"), runouts=50, bins=bins, rng=random.Random(3))
    medoid_histograms = np.stack([h1, h2, h3])
    sa_flop = StreetAbstraction(
        street="flop",
        bins=bins,
        medoid_histograms=medoid_histograms,
        medoid_hands=[
            (cards_from_str("AsAh"), cards_from_str("2c3d4h")),
            (cards_from_str("KsKh"), cards_from_str("2c3d4h")),
            (cards_from_str("7s2h"), cards_from_str("2c3d4h")),
        ],
    )
    a = Abstraction(streets={"flop": sa_flop})

    # Same query across 5 trials with 5 different caller-supplied rngs:
    # should give the same bucket every time (caller rng is ignored).
    hero = cards_from_str("QsJh")
    board = cards_from_str("2c3d4h")  # different board cards from any medoid
    buckets = set()
    for t in range(5):
        rng = random.Random(t * 31337)
        buckets.add(a.bucket_of(hero, board, runouts=50, rng=rng))
    assert len(buckets) == 1, f"got multiple buckets {buckets}, expected deterministic"
    # And with rng=None too -- should match.
    buckets.add(a.bucket_of(hero, board, runouts=50, rng=None))
    assert len(buckets) == 1, f"rng=None produced different bucket: {buckets}"
