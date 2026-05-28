"""GATE 2: parallel == sequential bit-identity at mix>0.

Phase 2 fork extends the worker contract so workers sample their own
overrides deterministically per (seed, iter, t). This test exercises
that path at league_mix=0.30 with a tiny Shanky-only pool (the same
fixture Phase 1 validation used).

Compares per-iter metrics (iter, traverser, adv_loss, strat_loss,
strat_buf, buf_0..buf_5) AND the final override_counts tally between
sequential train() and parallel_train() at G in {2, 4} with
use_processes=True. Bit-identity required.

Pool reconstruction cost test (CheckpointPolicy load doesn't perturb the
worker's main nets stub) is in test_parallel_phase2_worker_isolation.py
or noted N/A if no checkpoint-typed entry is in scope here.
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
SHANKY_FIXTURE = REPO_ROOT / "tests" / "scripted_bots_fixtures" / "littlegreen2.txt"


def _build_test_registry(path):
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


def _build_mix_gt_0_config(tc_kwargs, registry_path):
    """Augment a baseline_seq.yaml-shape TrainConfig6Max kwargs dict with
    league_mix=0.30 + the test registry. Returns a fresh TrainConfig6Max.
    """
    cfg = dict(tc_kwargs)
    cfg["league_mix"] = 0.30
    cfg["league_registry_path"] = str(registry_path)
    cfg["league_sample_strategy"] = "uniform"
    return TrainConfig6Max(**cfg)


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
def mix_gt_0_assets(tmp_path_factory):
    if not BASELINE_CONFIG_PATH.exists():
        pytest.skip("configs/baseline_seq.yaml missing")
    if not SHANKY_FIXTURE.exists():
        pytest.skip(f"Shanky fixture missing at {SHANKY_FIXTURE}")
    cfg = yaml.safe_load(open(BASELINE_CONFIG_PATH))
    abstraction_path = cfg["abstraction_path"]
    tc_kwargs = {
        k: v for k, v in cfg.items()
        if k not in ("tag", "abstraction_path", "checkpoint_every")
    }
    tmp = tmp_path_factory.mktemp("phase2_mix_gt_0")
    registry_path = tmp / "test_registry.json"
    _build_test_registry(registry_path)

    # --- Sequential anchor (mix>0) ---
    tc_seq = _build_mix_gt_0_config(tc_kwargs, registry_path)
    seq_solver, seq_game_str = _construct_solver(tc_seq, abstraction_path)
    seq_metrics = seq_solver.train(checkpoint_dir=None, checkpoint_every=999)
    seq_counts = dict(seq_solver._override_counts)
    return {
        "tc_kwargs": tc_kwargs,
        "abstraction_path": abstraction_path,
        "registry_path": registry_path,
        "seq_metrics": seq_metrics,
        "seq_counts": seq_counts,
    }


@pytest.mark.parametrize("n_groups", [2, 4])
@pytest.mark.timeout(240)
def test_parallel_mix_gt_0_matches_sequential(mix_gt_0_assets, n_groups):
    """Parallel mp.fork at mix=0.30 must match sequential bit-for-bit on
    metrics AND end-of-training override counter tally."""
    tc_parallel = _build_mix_gt_0_config(
        mix_gt_0_assets["tc_kwargs"], mix_gt_0_assets["registry_path"]
    )
    par_solver, par_game_str = _construct_solver(
        tc_parallel, mix_gt_0_assets["abstraction_path"]
    )

    par_metrics = parallel_train(
        par_solver,
        game_str=par_game_str,
        abstraction_path=mix_gt_0_assets["abstraction_path"],
        n_workers=n_groups,
        use_processes=True,
        checkpoint_dir=None,
    )

    _compare_metrics(
        par_metrics,
        mix_gt_0_assets["seq_metrics"],
        label=f"parallel G={n_groups} mp.fork mix>0",
    )

    assert par_solver._override_counts == mix_gt_0_assets["seq_counts"], (
        f"override-count tally mismatch at G={n_groups}: "
        f"parallel={par_solver._override_counts}, "
        f"sequential={mix_gt_0_assets['seq_counts']}"
    )
