"""Full-run bit-identity gate for src.nlhe.parallel.orchestrator.

Stage 1: in-process serial orchestrator (use_processes=False) at G in
{1,2,3,5} — every group count must produce metrics bit-identical to
runs/baseline_fork_A/metrics.json. Partition-independence is part of the
contract (fixed ascending-t merge means group boundaries are invisible).

Stage 2: fork()ed orchestrator (use_processes=True) at G in {2,3,5} —
same gate. Run only after Stage 1 passes (the user gates this manually).

Compared fields (per spec): iter, traverser, adv_loss, strat_loss,
strat_buf, buf_0..buf_5. Times differ (wall-clock) and mini_eval is empty.
Override-count check: at end of training, solver._override_counts must
equal {self_play: T, archetype: 0, league: 0} for the last iter — proves
the orchestrator called _maybe_sample_league_opponent() T times per iter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyspiel
import pytest
import yaml

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import PokerGameConfig
from src.nlhe.networks6 import NUM_SEATS_6MAX
from src.nlhe.parallel.orchestrator import parallel_train
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "baseline_seq.yaml"
REFERENCE_METRICS = REPO_ROOT / "runs" / "baseline_fork_A" / "metrics.json"


# ----------------------------- helpers -----------------------------


def _load_baseline_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_game_str_from_tc(tc: TrainConfig6Max) -> str:
    return PokerGameConfig(
        num_players=6,
        starting_stack=tc.starting_stack,
        big_blind=tc.big_blind,
        small_blind=tc.small_blind,
    ).to_universal_poker_string()


def _floats_exact(a, b) -> bool:
    """NaN-aware exact float equality (== as bits, except NaN==NaN -> True)."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


def _list_exact(a: list, b: list) -> tuple[bool, int]:
    """Return (ok, first_divergence_index) — -1 if equal."""
    if len(a) != len(b):
        return False, min(len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        if not _floats_exact(x, y):
            return False, i
    return True, -1


def _compare_metrics(actual: dict, expected: dict) -> None:
    """Bit-identity check on the gate fields. Raises AssertionError with a
    specific first-divergence pointer if anything differs.
    """
    checked_keys = ["iter", "traverser", "adv_loss", "strat_loss", "strat_buf"]
    for s in range(NUM_SEATS_6MAX):
        checked_keys.append(f"buf_{s}")
    for k in checked_keys:
        assert k in actual, f"actual metrics missing key {k!r}"
        assert k in expected, f"expected metrics missing key {k!r}"
        ok, idx = _list_exact(actual[k], expected[k])
        if not ok:
            ax = actual[k][idx] if idx < len(actual[k]) else "<missing>"
            ex = expected[k][idx] if idx < len(expected[k]) else "<missing>"
            raise AssertionError(
                f"metrics[{k!r}] diverges first at iter index {idx}: "
                f"actual={ax} expected={ex}"
            )


# ----------------------------- fixture -----------------------------


@pytest.fixture(scope="session")
def baseline_assets():
    """One-time setup shared across all parametrizations of this file.
    Loads config, abstraction, and the reference metrics. Solver is NOT
    constructed here — each test builds its own fresh solver.
    """
    cfg = _load_baseline_config()
    abstraction_path = cfg["abstraction_path"]
    cfg_for_tc = {
        k: v for k, v in cfg.items()
        if k not in ("tag", "abstraction_path", "checkpoint_every")
    }
    tc = TrainConfig6Max(**cfg_for_tc)
    game_str = _build_game_str_from_tc(tc)
    abst = Abstraction.load(abstraction_path)
    if not REFERENCE_METRICS.exists():
        pytest.skip(
            f"reference {REFERENCE_METRICS} missing — generate via "
            f"`python -m scripts.train_6max --config configs/baseline_seq.yaml "
            f"--out runs/baseline_fork_A` (committed at a8416b2-ish)"
        )
    with open(REFERENCE_METRICS) as f:
        expected = json.load(f)
    return tc, game_str, abst, abstraction_path, expected


def _run_and_compare(
    baseline_assets,
    *,
    n_groups: int,
    use_processes: bool,
) -> None:
    tc, game_str, abst, abstraction_path, expected = baseline_assets
    game = pyspiel.load_game(game_str)
    solver = DeepCFR6MaxSolver(game=game, abstraction=abst, config=tc)

    actual = parallel_train(
        solver,
        game_str=game_str,
        abstraction_path=abstraction_path,
        n_workers=n_groups,
        use_processes=use_processes,
        checkpoint_dir=None,
    )

    _compare_metrics(actual, expected)

    # Override-count gate: last iter's tally must be self_play=T at mix=0.
    T = tc.traversals_per_iter
    assert solver._override_counts == {
        "archetype": 0, "league": 0, "self_play": T
    }, (
        f"override-count tally mismatch: got {solver._override_counts}, "
        f"expected self_play={T} (archetype=0, league=0) at mix=0"
    )


# ----------------------------- Stage 1: in-process serial -----------------------------


@pytest.mark.parametrize("n_groups", [1, 2, 3, 5])
def test_orchestrator_serial_matches_baseline_fork_A(baseline_assets, n_groups):
    """Stage 1 gate. Partition-independent: any G in {1,2,3,5} produces
    metrics bit-identical to baseline_fork_A."""
    _run_and_compare(baseline_assets, n_groups=n_groups, use_processes=False)


# ----------------------------- Stage 2: fork()ed multiprocessing -----------------------------


@pytest.mark.parametrize("n_groups", [2, 3, 5])
def test_orchestrator_mp_matches_baseline_fork_A(baseline_assets, n_groups):
    """Stage 2 gate. fork() workers via mp.Pool; same bit-identity contract
    as Stage 1. Run only after Stage 1 passes at all G."""
    _run_and_compare(baseline_assets, n_groups=n_groups, use_processes=True)
