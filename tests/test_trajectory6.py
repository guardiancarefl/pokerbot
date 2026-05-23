"""Tests for src/nlhe/trajectory6.py (Phase 4e.1)."""
from __future__ import annotations
import math
import random

import pytest

from src.nlhe.game_strings import six_max_200bb, six_max_sng, hunl_200bb
from src.nlhe.trajectory6 import (
    Trajectory,
    TrajectoryStep,
    walk_game,
    uniform_random_policy,
    always_fold_policy,
)

import pyspiel


# ---- Basic walker mechanics ----

def test_walker_reaches_terminal_6max():
    """Random-policy 6-max walk completes within max_steps."""
    game = pyspiel.load_game(six_max_200bb())
    rng = random.Random(2026)
    traj = walk_game(game, uniform_random_policy, rng, max_steps=5000)
    assert not traj.terminated_at_max_steps
    assert len(traj.terminal_returns) == 6
    assert len(traj.steps) > 0


def test_walker_returns_are_zero_sum_6max():
    """Per-player returns sum to 0 for 6-max games."""
    game = pyspiel.load_game(six_max_200bb())
    for seed in [1, 2, 3, 42, 2026]:
        rng = random.Random(seed)
        traj = walk_game(game, uniform_random_policy, rng)
        assert abs(sum(traj.terminal_returns)) < 1e-6, (
            f"seed={seed}: returns {traj.terminal_returns} sum to "
            f"{sum(traj.terminal_returns)}, expected ~0"
        )


def test_walker_chance_nodes_consumed():
    """Walker consumes all chance nodes (card dealing) automatically."""
    game = pyspiel.load_game(six_max_200bb())
    rng = random.Random(2026)
    traj = walk_game(game, uniform_random_policy, rng)
    # 6-max needs 12 hole cards (6 players × 2) + up to 5 board cards.
    # Minimum: 12 chance samples if all-in preflop. Reasonable upper bound.
    assert traj.n_chance_nodes >= 12, (
        f"only {traj.n_chance_nodes} chance nodes consumed; expected >= 12 "
        f"(6 players × 2 hole cards)"
    )
    assert traj.n_chance_nodes <= 17  # 12 hole + 5 board max


def test_walker_steps_record_correct_metadata():
    """Each step records player, action taken (within legal), and policy."""
    game = pyspiel.load_game(six_max_200bb())
    rng = random.Random(2026)
    traj = walk_game(game, uniform_random_policy, rng)
    for step in traj.steps:
        assert isinstance(step.player, int)
        assert 0 <= step.player < 6
        assert step.action_taken in step.legal_actions
        assert len(step.policy) == len(step.legal_actions)
        # uniform_random_policy → all probs equal
        expected = 1.0 / len(step.legal_actions)
        for p in step.policy:
            assert math.isclose(p, expected, rel_tol=1e-9)


# ---- Walker works for non-6-max games too ----

def test_walker_works_for_hunl_too():
    """Same walker should handle HUNL since the mechanics are player-agnostic."""
    game = pyspiel.load_game(hunl_200bb())
    rng = random.Random(2026)
    traj = walk_game(game, uniform_random_policy, rng)
    assert traj.num_players == 2
    assert len(traj.terminal_returns) == 2
    assert abs(sum(traj.terminal_returns)) < 1e-6


def test_walker_works_for_sng_stacks():
    """6-max SNG with small starting stacks (15bb default) walks to terminal."""
    game = pyspiel.load_game(six_max_sng(starting_stack=1500))
    rng = random.Random(2026)
    traj = walk_game(game, uniform_random_policy, rng)
    assert traj.num_players == 6
    assert abs(sum(traj.terminal_returns)) < 1e-6


# ---- Always-fold semantics: SB folder loses blind ----

def test_sb_always_folds_loses_small_blind_on_average():
    """A small-blind seat that always folds loses ~0.5bb per hand (just their SB).

    SB seat (seat 0) posts the small blind preflop. If they always fold
    (action 0 when facing a bet), they lose their posted SB every hand.
    Action 0 in OpenSpiel universal_poker is Fold when facing a bet
    (verified inline by checking action_to_string).

    Note: UTG and other non-blind seats lose 0 when always-folding because
    they never pay a blind — there's nothing to lose.
    """
    game = pyspiel.load_game(six_max_200bb())
    folder_idx = 0  # SB seat — actually pays a blind

    def mixed_policy(state, cp):
        if cp == folder_idx:
            return always_fold_policy(state, cp)
        return uniform_random_policy(state, cp)

    total_folder_return = 0.0
    n_hands = 30
    for hand in range(n_hands):
        rng_hand = random.Random(hand * 7 + 1)
        traj = walk_game(game, mixed_policy, rng_hand)
        total_folder_return += traj.terminal_returns[folder_idx]
    avg = total_folder_return / n_hands
    # SB is 50 chips; folder loses ~50 per hand on average.
    # Allow for some hands where folder ends up checking (no bet to face).
    assert avg <= 0, f"SB always-folder's avg return is {avg}, expected <= 0"
    assert avg > -200, f"SB always-folder's avg return is {avg}, expected > -200 (one full BB worst-case)"


# ---- Policy contract enforcement ----

def test_walker_rejects_bad_policy_length():
    """Policy returning wrong number of probs raises ValueError."""
    game = pyspiel.load_game(six_max_200bb())
    rng = random.Random(2026)

    def bad_policy(state, cp):
        return [1.0]  # always length 1, regardless of legal_actions

    with pytest.raises(ValueError, match="policy returned"):
        walk_game(game, bad_policy, rng)


def test_walker_rejects_zero_policy():
    """Policy returning zero-sum probabilities raises ValueError."""
    game = pyspiel.load_game(six_max_200bb())
    rng = random.Random(2026)

    def zero_policy(state, cp):
        return [0.0] * len(state.legal_actions())

    with pytest.raises(ValueError, match="zero-sum probs"):
        walk_game(game, zero_policy, rng)


def test_walker_tolerates_small_renorm_drift():
    """Policy that sums to ~0.9999 or ~1.0001 should renormalize without raising."""
    game = pyspiel.load_game(six_max_200bb())
    rng = random.Random(2026)

    def slightly_off_policy(state, cp):
        legal = state.legal_actions()
        n = len(legal)
        # Build probs that sum to 0.999 — should be renormalized
        return [0.999 / n] * n

    traj = walk_game(game, slightly_off_policy, rng)
    assert not traj.terminated_at_max_steps
    assert abs(sum(traj.terminal_returns)) < 1e-6


# ---- Max-steps safety ----

def test_walker_respects_max_steps_safety():
    """Hitting max_steps flags the trajectory but doesn't crash."""
    game = pyspiel.load_game(six_max_200bb())
    rng = random.Random(2026)
    # Set max_steps absurdly low to force the safety break.
    traj = walk_game(game, uniform_random_policy, rng, max_steps=5)
    # We expected to bail out (real games take much more than 5 steps).
    assert traj.terminated_at_max_steps is True


# ---- Determinism with seeded RNG ----

def test_walker_deterministic_with_seed():
    """Same seed produces identical trajectory."""
    game = pyspiel.load_game(six_max_200bb())
    traj1 = walk_game(game, uniform_random_policy, random.Random(99))
    traj2 = walk_game(game, uniform_random_policy, random.Random(99))
    assert traj1.terminal_returns == traj2.terminal_returns
    assert len(traj1.steps) == len(traj2.steps)
    for s1, s2 in zip(traj1.steps, traj2.steps):
        assert s1.player == s2.player
        assert s1.action_taken == s2.action_taken


# ---- Different seeds produce different games ----

def test_walker_different_seeds_different_games():
    """Different seeds → different trajectories (cards/actions differ)."""
    game = pyspiel.load_game(six_max_200bb())
    traj1 = walk_game(game, uniform_random_policy, random.Random(1))
    traj2 = walk_game(game, uniform_random_policy, random.Random(2))
    # Either the step sequence or the terminal returns should differ.
    same_steps = (
        len(traj1.steps) == len(traj2.steps)
        and all(s1.action_taken == s2.action_taken
                for s1, s2 in zip(traj1.steps, traj2.steps))
    )
    same_returns = traj1.terminal_returns == traj2.terminal_returns
    assert not (same_steps and same_returns), (
        "Two different seeds produced identical trajectories — RNG isn't being consumed properly"
    )
