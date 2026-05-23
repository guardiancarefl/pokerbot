"""Tests for src/nlhe/stack_sampler.py (Phase 4f)."""
from __future__ import annotations
import random

import pytest

from src.nlhe.game_strings import TournamentStructure, BlindLevel
from src.nlhe.stack_sampler import sample_starting_state


CONFIG_PATH = "configs/ignition_double_up_6max_turbo.yaml"


@pytest.fixture(scope="module")
def ignition_structure():
    return TournamentStructure.from_yaml(CONFIG_PATH)


def test_sampler_chip_conservation(ignition_structure):
    """sum(stacks) must equal num_players * starting_chips."""
    rng = random.Random(0)
    expected_total = ignition_structure.total_chips_in_play()
    for _ in range(200):
        s = sample_starting_state(ignition_structure, rng, num_paid=3)
        assert sum(s["stacks"]) == expected_total


def test_sampler_alive_count_above_bubble(ignition_structure):
    """alive_count must be > num_paid (we never sample past-bubble states)."""
    rng = random.Random(1)
    for _ in range(200):
        s = sample_starting_state(ignition_structure, rng, num_paid=3)
        alive = sum(1 for x in s["stacks"] if x > 0)
        assert alive > 3
        assert s["alive_count"] == alive


def test_sampler_alive_stacks_meet_bb_floor(ignition_structure):
    """Each alive seat\'s stack must be >= inflated_big_blind."""
    rng = random.Random(2)
    for _ in range(200):
        s = sample_starting_state(ignition_structure, rng, num_paid=3)
        bb_inf = s["blind_level"].inflated_big_blind(ignition_structure.num_players)
        for stack in s["stacks"]:
            assert stack == 0 or stack >= bb_inf


def test_sampler_dealer_is_alive(ignition_structure):
    """dealer_seat must point to a seat with nonzero stack."""
    rng = random.Random(3)
    for _ in range(200):
        s = sample_starting_state(ignition_structure, rng, num_paid=3)
        assert s["stacks"][s["dealer_seat"]] > 0


def test_sampler_blind_level_in_schedule(ignition_structure):
    """Sampled blind_level must be from the structure\'s schedule."""
    rng = random.Random(4)
    valid_levels = {bl.level for bl in ignition_structure.blind_schedule}
    for _ in range(200):
        s = sample_starting_state(ignition_structure, rng, num_paid=3)
        assert s["blind_level"].level in valid_levels


def test_sampler_dealer_spreads_across_seats(ignition_structure):
    """Across many samples, dealer should land on a variety of seats."""
    rng = random.Random(5)
    dealer_counts = {}
    for _ in range(1000):
        s = sample_starting_state(ignition_structure, rng, num_paid=3)
        dealer_counts[s["dealer_seat"]] = dealer_counts.get(s["dealer_seat"], 0) + 1
    # All 6 seats should have been dealer at least once
    assert len(dealer_counts) == 6


def test_sampler_blind_level_distribution_matches_weights(ignition_structure):
    """High-weight levels should appear more often than low-weight ones."""
    rng = random.Random(6)
    level_counts = {}
    for _ in range(2000):
        s = sample_starting_state(ignition_structure, rng, num_paid=3)
        level_counts[s["blind_level"].level] = (
            level_counts.get(s["blind_level"].level, 0) + 1
        )
    # Level 1 has weight ~0.18 (highest), level 9 has ~0.02 (much lower)
    assert level_counts.get(1, 0) > level_counts.get(9, 0)
