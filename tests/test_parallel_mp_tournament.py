"""GATE 3: parallel-tournament == sequential-tournament bit-identity.

Phase 3 forks the stack-sampling rng so workers can derive identical
per-traversal starting states from (seed, iter, t) + STACK_SAMPLE_SALT.
This test exercises tournament mode at two configurations:

  - mix=0 (pure tournament training)
  - mix=0.30 (tournament + league override — the diversity-mix endpoint)

Compares per-iter metrics + final solver._override_counts between
sequential train() and parallel_train(use_processes=True) at G=2 and G=4.

Sized small enough to fit two parametrizations under 300s:
  hidden_dim=[32,32], n_iterations=8, traversals_per_iter=20,
  train_steps_per_iter=10, batch_size=8.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyspiel
import pytest
import yaml

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import PokerGameConfig
from src.nlhe.networks6 import NUM_SEATS_6MAX
from src.nlhe.parallel.orchestrator import parallel_train
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max


REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "baseline_seq.yaml"
TOURNAMENT_STRUCTURE_PATH = (
    REPO_ROOT / "configs" / "ignition_double_up_6max_turbo.yaml"
)
SHANKY_FIXTURE = REPO_ROOT / "tests" / "scripted_bots_fixtures" / "littlegreen2.txt"


def _write_test_registry(path):
    registry = {
        "version": 1,
        "entries": [
            {
                "name": "shanky-littlegreen2",
                "path": str(SHANKY_FIXTURE),
                "metadata": {
                    "policy_type": "shanky",
                    "big_blind_chips": 100,
                },
                "tags": ["test", "shanky"],
            }
        ],
    }
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def _build_tournament_tc(**overrides):
    """Compact tournament-mode config sized for ~120s sequential."""
    base = yaml.safe_load(open(BASELINE_CONFIG_PATH))
    tc_kwargs = {
        k: v for k, v in base.items()
        if k not in ("tag", "abstraction_path", "checkpoint_every")
    }
    # Override with tournament-mode shape (small but real).
    tc_kwargs.update({
        "tournament_structure_path": str(TOURNAMENT_STRUCTURE_PATH),
        "hidden_dim": [32, 32],
        "n_iterations": 8,
        "traversals_per_iter": 20,
        "train_steps_per_iter": 10,
        "batch_size": 8,
    })
    tc_kwargs.update(overrides)
    return TrainConfig6Max(**tc_kwargs), base["abstraction_path"]


def _construct_solver(tc, abstraction_path):
    game_str = PokerGameConfig(
        num_players=6,
        starting_stack=tc.starting_stack,
        big_blind=tc.big_blind,
        small_blind=tc.small_blind,
    ).to_universal_poker_string()
    game = pyspiel.load_game(game_str)
    abst = Abstraction.load(abstraction_path)
    solver = DeepCFR6MaxSolver(game=game, abstraction=abst, config=tc)
    return solver, game_str


def _floats_exact(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


def _list_exact(a, b):
    if len(a) != len(b):
        return False, min(len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        if not _floats_exact(x, y):
            return False, i
    return True, -1


def _compare_metrics(actual, expected, label):
    keys = ["iter", "traverser", "adv_loss", "strat_loss", "strat_buf"]
    for s in range(NUM_SEATS_6MAX):
        keys.append(f"buf_{s}")
    for k in keys:
        ok, idx = _list_exact(actual[k], expected[k])
        if not ok:
            ax = actual[k][idx] if idx < len(actual[k]) else "<missing>"
            ex = expected[k][idx] if idx < len(expected[k]) else "<missing>"
            raise AssertionError(
                f"{label}: metrics[{k!r}] diverges first at iter index {idx}: "
                f"actual={ax} expected={ex}"
            )


@pytest.fixture(scope="module")
def tournament_mix0_assets():
    if not TOURNAMENT_STRUCTURE_PATH.exists():
        pytest.skip(f"tournament structure missing at {TOURNAMENT_STRUCTURE_PATH}")
    tc, abstraction_path = _build_tournament_tc()
    seq_solver, _ = _construct_solver(tc, abstraction_path)
    seq_metrics = seq_solver.train(checkpoint_dir=None, checkpoint_every=999)
    seq_counts = dict(seq_solver._override_counts)
    return {
        "tc": tc,
        "abstraction_path": abstraction_path,
        "seq_metrics": seq_metrics,
        "seq_counts": seq_counts,
    }


@pytest.fixture(scope="module")
def tournament_mix_gt_0_assets(tmp_path_factory):
    if not TOURNAMENT_STRUCTURE_PATH.exists():
        pytest.skip(f"tournament structure missing at {TOURNAMENT_STRUCTURE_PATH}")
    if not SHANKY_FIXTURE.exists():
        pytest.skip(f"Shanky fixture missing at {SHANKY_FIXTURE}")
    tmp = tmp_path_factory.mktemp("phase3_tournament_mix_gt_0")
    registry_path = tmp / "test_registry.json"
    _write_test_registry(registry_path)
    tc, abstraction_path = _build_tournament_tc(
        league_mix=0.30,
        league_registry_path=str(registry_path),
        league_sample_strategy="uniform",
    )
    seq_solver, _ = _construct_solver(tc, abstraction_path)
    seq_metrics = seq_solver.train(checkpoint_dir=None, checkpoint_every=999)
    seq_counts = dict(seq_solver._override_counts)
    return {
        "tc": tc,
        "abstraction_path": abstraction_path,
        "seq_metrics": seq_metrics,
        "seq_counts": seq_counts,
    }


def _run_parallel_and_compare(assets, n_groups, label):
    par_solver, par_game_str = _construct_solver(
        assets["tc"], assets["abstraction_path"]
    )
    par_metrics = parallel_train(
        par_solver,
        game_str=par_game_str,
        abstraction_path=assets["abstraction_path"],
        n_workers=n_groups,
        use_processes=True,
        checkpoint_dir=None,
    )
    _compare_metrics(par_metrics, assets["seq_metrics"], label=label)
    assert par_solver._override_counts == assets["seq_counts"], (
        f"{label}: override-count tally mismatch: "
        f"parallel={par_solver._override_counts}, "
        f"sequential={assets['seq_counts']}"
    )


@pytest.mark.parametrize("n_groups", [2, 4])
@pytest.mark.timeout(300)
def test_parallel_tournament_mix0_matches_sequential(
    tournament_mix0_assets, n_groups
):
    """Tournament mode, mix=0: parallel must reproduce sequential bit-for-bit."""
    _run_parallel_and_compare(
        tournament_mix0_assets,
        n_groups=n_groups,
        label=f"tournament mix=0 G={n_groups}",
    )


@pytest.mark.parametrize("n_groups", [2, 4])
@pytest.mark.timeout(300)
def test_parallel_tournament_mix_gt_0_matches_sequential(
    tournament_mix_gt_0_assets, n_groups
):
    """Tournament mode + league_mix=0.30: the diversity-mix endpoint."""
    _run_parallel_and_compare(
        tournament_mix_gt_0_assets,
        n_groups=n_groups,
        label=f"tournament mix=0.30 G={n_groups}",
    )
