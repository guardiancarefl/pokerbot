"""Auto-derive regression test for solver6's encoder max_bucket_dim.

The solver previously hardcoded max_bucket_dim=200, which silently
truncated higher-k abstractions in the one-hot encoder. The fix derives
max_bucket_dim from max(sa.k for sa in abstraction.streets.values()) at
solver init, so the encoder + nets adapt automatically.

Both subtests construct a real DeepCFR6MaxSolver from baseline_seq.yaml
(legacy mode, [32,32] nets — kept tiny so the test runs fast) and
inspect the resulting encoder + nets dims.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pyspiel
import pytest
import yaml

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import PokerGameConfig
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max


REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "baseline_seq.yaml"


def _construct_solver(abstraction_path):
    cfg = yaml.safe_load(open(BASELINE_CONFIG_PATH))
    tc_kwargs = {
        k: v for k, v in cfg.items()
        if k not in ("tag", "abstraction_path", "checkpoint_every")
    }
    tc = TrainConfig6Max(**tc_kwargs)
    game_str = PokerGameConfig(
        num_players=6,
        starting_stack=tc.starting_stack,
        big_blind=tc.big_blind,
        small_blind=tc.small_blind,
    ).to_universal_poker_string()
    game = pyspiel.load_game(game_str)
    abst = Abstraction.load(abstraction_path)
    return DeepCFR6MaxSolver(game=game, abstraction=abst, config=tc)


def test_auto_derive_k200_preserves_legacy_dims():
    """Existing k=200 retrofit abstraction must produce the historical
    feature_dim=236 / input_dim=236 — auto-derive must not perturb the
    k=200 path (this is what keeps all 16 prior parallel gates green)."""
    cfg = yaml.safe_load(open(BASELINE_CONFIG_PATH))
    abstraction_path = cfg["abstraction_path"]
    if not (REPO_ROOT / abstraction_path).exists():
        pytest.skip(f"k=200 abstraction missing at {abstraction_path}")
    solver = _construct_solver(abstraction_path)
    assert solver.encoder.max_bucket_dim == 200, (
        f"k=200 path: max_bucket_dim={solver.encoder.max_bucket_dim}, "
        f"expected 200"
    )
    assert solver.encoder.feature_dim == 236, (
        f"k=200 path: feature_dim={solver.encoder.feature_dim}, expected 236"
    )
    assert solver.policy_nets.input_dim == 236, (
        f"k=200 path: input_dim={solver.policy_nets.input_dim}, expected 236"
    )


def test_auto_derive_k500_adapts_to_larger_abstraction():
    """k=500 abstraction must produce feature_dim=536 / input_dim=536 —
    confirms auto-derive picks up the larger k and no hardcoded 200 leaks
    through. Skipped when no k=500 artifact is present locally (this
    happens in environments where the regen has not been run)."""
    matches = sorted(
        glob.glob(str(REPO_ROOT / "runs" / "abstraction_k500_*" / "abstraction.pkl"))
    )
    if not matches:
        pytest.skip(
            "k=500 abstraction not present locally (regen via "
            "`python scripts/train_abstraction.py --k_postflop 500`)"
        )
    abstraction_path = matches[-1]
    solver = _construct_solver(abstraction_path)
    assert solver.encoder.max_bucket_dim == 500, (
        f"k=500 path: max_bucket_dim={solver.encoder.max_bucket_dim}, "
        f"expected 500"
    )
    assert solver.encoder.feature_dim == 536, (
        f"k=500 path: feature_dim={solver.encoder.feature_dim}, expected 536"
    )
    assert solver.policy_nets.input_dim == 536, (
        f"k=500 path: input_dim={solver.policy_nets.input_dim}, expected 536"
    )
