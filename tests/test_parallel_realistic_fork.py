"""Realistic-config regression test for mp.fork() determinism.

The smoke gate (test_parallel_orchestrator.py at baseline_seq.yaml's
[32,32] nets) cannot detect the torch + mp.fork() MKL deadlock — tiny
matmul sizes never engage multi-threaded BLAS, so fork() is incidentally
safe regardless of the fix. This file exercises [256,256] nets, which DO
engage multi-threaded BLAS by default, exposing the deadlock to any
regression of the single-threaded-torch determinism fix in solver6.py.

Each test runs parallel_train with use_processes=True at a small but
non-smoke configuration; if mp.fork() deadlocks at fork-after-MKL-init
time, the pytest-timeout(180) decorator fails the test rather than
hanging the suite. There is no bit-identity comparison — the only gate
is "completes within budget".
"""

from __future__ import annotations

from pathlib import Path

import pyspiel
import pytest
import yaml

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import PokerGameConfig
from src.nlhe.parallel.orchestrator import parallel_train
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max


REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "baseline_seq.yaml"


def _build_realistic_solver(tmp_path):
    """Construct a solver with the smallest config that still triggers
    [256,256] BLAS — small enough to fit two iters in under 180s but big
    enough to engage multi-threaded BLAS heuristics in any unfixed build.
    """
    base_cfg = yaml.safe_load(open(BASELINE_CONFIG_PATH))
    abstraction_path = base_cfg["abstraction_path"]
    if not (REPO_ROOT / abstraction_path).exists():
        pytest.skip(f"abstraction missing at {abstraction_path}")

    tc = TrainConfig6Max(
        starting_stack=1500,
        big_blind=100,
        small_blind=50,
        payout_mode="double_up",
        buy_in=1.0,
        first_share=0.65,
        hidden_dim=[256, 256],   # the threshold where MKL engages multi-threaded BLAS
        n_iterations=2,
        traversals_per_iter=20,
        train_steps_per_iter=10,
        batch_size=32,
        learning_rate=0.001,
        buffer_capacity=20000,
        bucket_runouts=30,
        max_traversal_depth=200,
        seed=2026,
    )

    game_str = PokerGameConfig(
        num_players=6,
        starting_stack=tc.starting_stack,
        big_blind=tc.big_blind,
        small_blind=tc.small_blind,
    ).to_universal_poker_string()
    game = pyspiel.load_game(game_str)
    abst = Abstraction.load(abstraction_path)
    solver = DeepCFR6MaxSolver(game=game, abstraction=abst, config=tc)
    return solver, tc, game_str, abstraction_path


def _run_and_assert_complete(solver, tc, game_str, abstraction_path, n_groups):
    metrics = parallel_train(
        solver,
        game_str=game_str,
        abstraction_path=abstraction_path,
        n_workers=n_groups,
        use_processes=True,
        checkpoint_dir=None,
    )
    # The only gate here is "did it complete without hanging" — the timeout
    # decorator turns a deadlock into a fast test failure rather than
    # hanging the whole suite. Bit-identity is gated in
    # test_parallel_orchestrator.py at smoke config.
    assert len(metrics["iter"]) == tc.n_iterations, (
        f"expected {tc.n_iterations} iters, got {len(metrics['iter'])}"
    )
    assert metrics["iter"] == list(range(1, tc.n_iterations + 1))


@pytest.mark.timeout(180)
def test_realistic_fork_G2_completes(tmp_path):
    """mp.fork at G=2 must complete at [256,256] without deadlocking."""
    solver, tc, game_str, abstraction_path = _build_realistic_solver(tmp_path)
    _run_and_assert_complete(solver, tc, game_str, abstraction_path, n_groups=2)


@pytest.mark.timeout(180)
def test_realistic_fork_G4_completes(tmp_path):
    """mp.fork at G=4 must complete at [256,256] without deadlocking."""
    solver, tc, game_str, abstraction_path = _build_realistic_solver(tmp_path)
    _run_and_assert_complete(solver, tc, game_str, abstraction_path, n_groups=4)
