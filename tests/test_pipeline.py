"""Smoke tests for the Phase 1 Leduc training pipeline.

These tests are intentionally fast (~30 seconds total). Their job is to
catch broken imports, broken file IO, and broken API calls — not to
validate algorithm correctness. Algorithm validation comes from the
exploitability number on the real training run.

Run with: python -m pytest tests/ -v
Or just: python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pyspiel
import torch

# Make src importable when running tests directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.leduc.checkpoint import save_checkpoint, load_checkpoint
from src.leduc.config import TrainConfig
from src.leduc.evaluate import exploitability_mbb
from src.leduc.solver import train


def test_config_yaml_roundtrip():
    """Both default and smoke YAML configs load cleanly."""
    for name in ["leduc_default", "leduc_smoke"]:
        path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        c = TrainConfig.from_yaml(path)
        # Roundtrip through dict.
        c2 = TrainConfig.from_dict(c.to_dict())
        assert c == c2, f"roundtrip mismatch for {name}"
    print("  config roundtrip OK")


def test_config_overrides():
    """CLI override semantics: None passes through, non-None replaces."""
    c = TrainConfig(iterations=100, num_traversals=40)
    c2 = c.merge_overrides({"iterations": 5, "num_traversals": None})
    assert c2.iterations == 5
    assert c2.num_traversals == 40
    print("  override semantics OK")


def test_uniform_policy_is_exploitable():
    """Sanity check: uniform random in Leduc is highly exploitable."""
    game = pyspiel.load_game("leduc_poker")

    def uniform(state):
        actions = state.legal_actions()
        p = 1.0 / len(actions)
        return {a: p for a in actions}

    expl = exploitability_mbb(game, uniform)
    assert expl > 100, f"uniform should be very exploitable, got {expl}"
    print(f"  uniform exploitability: {expl:.1f} mbb/g")


def test_full_pipeline_smoke():
    """End-to-end: load smoke config, train, evaluate, save checkpoint, load it back.

    This is the most important test. If this passes, the whole stack works.
    """
    config_path = PROJECT_ROOT / "configs" / "leduc_smoke.yaml"
    config = TrainConfig.from_yaml(config_path)

    game = pyspiel.load_game("leduc_poker")
    result = train(game, config)

    import math
    assert result.train_seconds > 0, "train_seconds should be positive"
    assert set(result.advantage_losses.keys()) == {0, 1}, \
        f"expected losses for players 0 and 1, got {result.advantage_losses.keys()}"
    for p, losses in result.advantage_losses.items():
        assert len(losses) == config.iterations, \
            f"player {p}: expected {config.iterations} loss entries, got {len(losses)}"
    # policy_loss and individual advantage losses may be NaN under the smoke
    # config (insufficient data to fill a batch). We allow it.
    assert isinstance(result.policy_loss, float), "policy_loss should be a float"

    # Exploitability eval should run without errors and produce a positive number.
    expl = exploitability_mbb(game, result.action_probabilities_fn)
    assert expl > 0, f"exploitability should be positive, got {expl}"
    print(f"  smoke train: {config.iterations} iters in {result.train_seconds:.1f}s, "
          f"final_loss={result.policy_loss:.4f}, expl={expl:.1f} mbb/g")

    # Checkpoint roundtrip.
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "smoke.pt"
        metrics = {"exploitability_mbb_per_game": expl,
                   "final_policy_loss": result.policy_loss}
        save_checkpoint(ckpt_path, result.policy_network,
                        config.to_dict(), metrics)
        loaded = load_checkpoint(ckpt_path)
        assert loaded["config"]["iterations"] == config.iterations
        assert loaded["metrics"]["final_policy_loss"] == result.policy_loss
        assert "policy_network_state_dict" in loaded
        # Companion JSONs should exist.
        assert (Path(tmpdir) / "config.json").exists()
        assert (Path(tmpdir) / "metrics.json").exists()
    print("  checkpoint roundtrip OK")


def run_all():
    """Run all tests sequentially. Returns 0 on success, 1 on failure."""
    tests = [
        ("config_yaml_roundtrip", test_config_yaml_roundtrip),
        ("config_overrides", test_config_overrides),
        ("uniform_policy_is_exploitable", test_uniform_policy_is_exploitable),
        ("full_pipeline_smoke", test_full_pipeline_smoke),
    ]
    failed = []
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            failed.append(name)

    print()
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} — {failed}")
        return 1
    else:
        print(f"ALL {len(tests)} TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(run_all())
