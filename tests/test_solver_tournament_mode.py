"""Tests for DeepCFR6MaxSolver tournament mode (Phase 4f)."""
from __future__ import annotations
import os
import pytest

ABSTRACTION_PATH = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"


@pytest.fixture(scope="module")
def smoke_abstraction():
    if not os.path.exists(ABSTRACTION_PATH):
        pytest.skip(f"abstraction not found at {ABSTRACTION_PATH}")
    from src.nlhe.abstraction import Abstraction
    return Abstraction.load(ABSTRACTION_PATH)


def test_solver_tournament_mode_loads_structure(smoke_abstraction):
    """Solver constructed with tournament_structure_path should load the structure."""
    import pyspiel
    from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max
    from src.nlhe.game_strings import six_max_sng

    cfg = TrainConfig6Max(
        starting_stack=1500,
        hidden_dim=[32, 32],
        n_iterations=1,
        traversals_per_iter=2,
        train_steps_per_iter=2,
        batch_size=4,
        buffer_capacity=100,
        bucket_runouts=10,
        seed=2026,
        tournament_structure_path="configs/ignition_double_up_6max_turbo.yaml",
    )
    game = pyspiel.load_game(six_max_sng(starting_stack=1500))
    solver = DeepCFR6MaxSolver(game=game, abstraction=smoke_abstraction, config=cfg)
    assert solver.tournament_structure is not None
    assert solver.tournament_structure.format_name == "ignition_double_up_6max_turbo"


def test_solver_legacy_mode_no_structure(smoke_abstraction):
    """Solver without tournament_structure_path should have no structure."""
    import pyspiel
    from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max
    from src.nlhe.game_strings import six_max_sng

    cfg = TrainConfig6Max(
        starting_stack=1500,
        hidden_dim=[32, 32],
        n_iterations=1,
        traversals_per_iter=2,
        train_steps_per_iter=2,
        batch_size=4,
        buffer_capacity=100,
        bucket_runouts=10,
        seed=2026,
    )
    game = pyspiel.load_game(six_max_sng(starting_stack=1500))
    solver = DeepCFR6MaxSolver(game=game, abstraction=smoke_abstraction, config=cfg)
    assert solver.tournament_structure is None


def test_solver_tournament_mode_trains_without_crash(smoke_abstraction):
    """Smoke: tournament-mode solver completes a few iters without crashing."""
    import pyspiel
    from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max
    from src.nlhe.game_strings import six_max_sng

    cfg = TrainConfig6Max(
        starting_stack=1500,
        hidden_dim=[32, 32],
        n_iterations=2,
        traversals_per_iter=5,
        train_steps_per_iter=5,
        batch_size=4,
        buffer_capacity=200,
        bucket_runouts=10,
        seed=2026,
        tournament_structure_path="configs/ignition_double_up_6max_turbo.yaml",
    )
    game = pyspiel.load_game(six_max_sng(starting_stack=1500))
    solver = DeepCFR6MaxSolver(game=game, abstraction=smoke_abstraction, config=cfg)
    solver.train()

    # At least one buffer should have samples after training
    buffer_sizes = [
        len(solver.policy_nets.buffer_for(s)) for s in range(6)
    ]
    assert max(buffer_sizes) > 0


def test_solver_legacy_mode_trains_without_crash(smoke_abstraction):
    """Smoke: legacy-mode solver still works (no regression)."""
    import pyspiel
    from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max
    from src.nlhe.game_strings import six_max_sng

    cfg = TrainConfig6Max(
        starting_stack=1500,
        hidden_dim=[32, 32],
        n_iterations=2,
        traversals_per_iter=5,
        train_steps_per_iter=5,
        batch_size=4,
        buffer_capacity=200,
        bucket_runouts=10,
        seed=2026,
    )
    game = pyspiel.load_game(six_max_sng(starting_stack=1500))
    solver = DeepCFR6MaxSolver(game=game, abstraction=smoke_abstraction, config=cfg)
    solver.train()
    buffer_sizes = [
        len(solver.policy_nets.buffer_for(s)) for s in range(6)
    ]
    assert max(buffer_sizes) > 0
