"""Tests for league-play wire-in to solver6 + cfr6.

Validates the trainer integration added on phase4f-league:
  - TrainConfig6Max league_* fields default to no-league behavior
  - DeepCFR6MaxSolver._maybe_sample_league_opponent sampling rate matches
    league_mix probabilistically
  - Validation errors fire on bad config combinations
  - traverse_6max opponent_policy_override threading: when set, the
    override is invoked at NON-traverser nodes; self-play machinery is
    bypassed at those nodes; defaults preserve old behavior.

The cadence/probability tests mirror test_reinit's pure-math pattern:
they re-derive the formula inline so they validate logic independent of
solver state.

The traversal-propagation tests use lightweight mocks for state/Policy/
context so they don't require torch or pyspiel.
"""
from __future__ import annotations

import random
from unittest.mock import MagicMock

import pytest


# ============================================================
# Sampling-rate formula — pure math.
# ============================================================

def _should_sample(rng_value: float, league_mix: float) -> bool:
    """Mirror of formula in DeepCFR6MaxSolver._maybe_sample_league_opponent.

    The solver calls self.rng.random() and samples iff that value is < league_mix.
    """
    if league_mix <= 0.0:
        return False
    return rng_value < league_mix


class TestSamplingRate:
    def test_zero_mix_never_fires(self):
        for v in [0.0, 0.001, 0.5, 0.999]:
            assert _should_sample(v, 0.0) is False

    def test_negative_mix_never_fires(self):
        for v in [0.0, 0.5, 0.999]:
            assert _should_sample(v, -0.1) is False

    def test_one_mix_always_fires(self):
        # rng.random() returns values in [0, 1), so 1.0 covers them all.
        for v in [0.0, 0.001, 0.5, 0.999, 0.9999999]:
            assert _should_sample(v, 1.0) is True

    def test_half_mix_fires_on_lower_half(self):
        assert _should_sample(0.0, 0.5) is True
        assert _should_sample(0.49, 0.5) is True
        assert _should_sample(0.5, 0.5) is False
        assert _should_sample(0.99, 0.5) is False

    def test_quarter_mix_empirical_rate(self):
        # 50k draws against league_mix=0.25 should land near 0.25 within
        # ~3 stderr on a binomial.
        rng = random.Random(42)
        hits = sum(1 for _ in range(50_000) if _should_sample(rng.random(), 0.25))
        rate = hits / 50_000
        assert 0.235 < rate < 0.265  # ~3 stderr window

    def test_thirty_percent_mix_empirical_rate(self):
        # Matches the recommended starting mix in the staged league config.
        rng = random.Random(2026)
        hits = sum(1 for _ in range(50_000) if _should_sample(rng.random(), 0.30))
        rate = hits / 50_000
        assert 0.285 < rate < 0.315


# ============================================================
# TrainConfig6Max defaults.
# ============================================================

class TestConfigDefaults:
    @pytest.fixture
    def cfg(self):
        try:
            from src.nlhe.solver6 import TrainConfig6Max
        except ImportError as e:
            pytest.skip(f"TrainConfig6Max import failed: {e}")
        return TrainConfig6Max()

    def test_league_mix_defaults_to_zero(self, cfg):
        assert cfg.league_mix == 0.0

    def test_league_registry_path_defaults_to_none(self, cfg):
        assert cfg.league_registry_path is None

    def test_league_sample_strategy_defaults_uniform(self, cfg):
        assert cfg.league_sample_strategy == "uniform"

    def test_league_weights_defaults_to_none(self, cfg):
        assert cfg.league_weights is None

    def test_league_tag_filter_defaults_to_none(self, cfg):
        assert cfg.league_tag_filter is None

    def test_defaults_preserve_old_behavior_invariant(self, cfg):
        # The combination of league_mix=0 + league_pool=None is what
        # _maybe_sample_league_opponent uses to short-circuit to None.
        # If this invariant breaks, every old training run changes behavior.
        assert cfg.league_mix == 0.0
        assert cfg.league_registry_path is None


# ============================================================
# Solver-level invariants — no torch needed for the validation path.
# ============================================================

class TestSolverValidation:
    """Validate the early-error paths in __init__ that fire before any
    network construction. These tests would ideally instantiate a real
    solver but that requires torch + pyspiel. We import the relevant
    error-raising logic by reflection where possible.
    """

    def test_mix_without_registry_raises(self):
        """league_mix > 0 with no registry path is a config error.

        We can't construct a real solver without torch, so verify the
        error message text exists in the source as a smoke check that
        the validation branch is wired.
        """
        try:
            import inspect
            from src.nlhe import solver6
        except ImportError as e:
            pytest.skip(f"solver6 import failed: {e}")

        src = inspect.getsource(solver6.DeepCFR6MaxSolver.__init__)
        assert "league_mix > 0 requires league_registry_path" in src

    def test_empty_pool_raises(self):
        """Empty eligible pool should fail loudly, not silently no-op."""
        try:
            import inspect
            from src.nlhe import solver6
        except ImportError as e:
            pytest.skip(f"solver6 import failed: {e}")

        src = inspect.getsource(solver6.DeepCFR6MaxSolver.__init__)
        assert "empty eligible" in src or "empty\n                    eligible" in src


# ============================================================
# traverse_6max opponent_policy_override propagation.
# ============================================================

class TestOverridePropagation:
    """Verify the override threads through every recursive call site
    in cfr6.traverse_6max. This is the key correctness property: if any
    call site forgets to pass it, league opponents would silently revert
    to self-play at deeper nodes.
    """

    def test_signature_has_override_param(self):
        try:
            import inspect
            from src.nlhe.cfr6 import traverse_6max
        except ImportError as e:
            pytest.skip(f"cfr6 import failed: {e}")
        sig = inspect.signature(traverse_6max)
        assert "opponent_policy_override" in sig.parameters
        # Default must be None to preserve old behavior.
        assert sig.parameters["opponent_policy_override"].default is None

    def test_all_recursive_calls_thread_override(self):
        """grep the source: every traverse_6max recursive call must pass
        opponent_policy_override=opponent_policy_override.

        Counting only matters in cfr6.py itself (where the function is
        defined and recurses). We expect exactly 4 recursive call sites:
          1. Chance node
          2. League-override short-circuit branch
          3. Traverser-node enumeration loop
          4. Self-play opponent-node branch
        """
        try:
            from src.nlhe import cfr6
            import inspect
        except ImportError as e:
            pytest.skip(f"cfr6 import failed: {e}")

        src = inspect.getsource(cfr6.traverse_6max)
        # Count keyword-argument occurrences (the signature line uses the
        # bare name; the recursive calls all pass it by keyword).
        kwarg_uses = src.count("opponent_policy_override=opponent_policy_override")
        assert kwarg_uses == 4, (
            f"expected 4 recursive call sites threading the override, "
            f"got {kwarg_uses}. If you added or removed a recursive call, "
            f"update this test."
        )

    def test_short_circuit_check_uses_correct_condition(self):
        """The override short-circuit must check BOTH (override is not
        None) AND (cp != traversing_player). If we got the player check
        wrong, traverser-node regret samples would be missed.
        """
        try:
            from src.nlhe import cfr6
            import inspect
        except ImportError as e:
            pytest.skip(f"cfr6 import failed: {e}")
        src = inspect.getsource(cfr6.traverse_6max)
        assert (
            "opponent_policy_override is not None and cp != traversing_player"
            in src
        )


# ============================================================
# Sampling-rate distribution under a mocked LeaguePool.
# ============================================================

class TestSamplingViaSolverHelper:
    """Exercise DeepCFR6MaxSolver._maybe_sample_league_opponent directly
    via a duck-typed object — avoids constructing a real solver (which
    needs torch + pyspiel) but exercises the actual control flow.
    """

    def _make_fake_solver(self, league_mix, league_pool):
        """Build a minimal object that quacks like DeepCFR6MaxSolver for
        the purposes of _maybe_sample_league_opponent."""
        from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max
        # Bind the method to a SimpleNamespace-like object.
        class Fake:
            pass
        f = Fake()
        f.cfg = TrainConfig6Max(league_mix=league_mix)
        f.league_pool = league_pool
        f.rng = random.Random(2026)
        f._maybe_sample_league_opponent = (
            DeepCFR6MaxSolver._maybe_sample_league_opponent.__get__(f, Fake)
        )
        return f

    def test_pool_none_returns_none(self):
        try:
            f = self._make_fake_solver(league_mix=0.5, league_pool=None)
        except ImportError as e:
            pytest.skip(f"solver6 import failed: {e}")
        for _ in range(100):
            assert f._maybe_sample_league_opponent() is None

    def test_mix_zero_returns_none(self):
        try:
            pool = MagicMock()
            pool.sample_opponent = MagicMock(side_effect=AssertionError(
                "sample_opponent must not be called when league_mix=0"
            ))
            f = self._make_fake_solver(league_mix=0.0, league_pool=pool)
        except ImportError as e:
            pytest.skip(f"solver6 import failed: {e}")
        for _ in range(100):
            assert f._maybe_sample_league_opponent() is None

    def test_mix_one_always_samples(self):
        try:
            pool = MagicMock()
            sentinel = object()
            pool.sample_opponent = MagicMock(return_value=sentinel)
            f = self._make_fake_solver(league_mix=1.0, league_pool=pool)
        except ImportError as e:
            pytest.skip(f"solver6 import failed: {e}")
        for _ in range(50):
            assert f._maybe_sample_league_opponent() is sentinel
        assert pool.sample_opponent.call_count == 50

    def test_mix_half_empirical_rate(self):
        try:
            pool = MagicMock()
            pool.sample_opponent = MagicMock(return_value="opp")
            f = self._make_fake_solver(league_mix=0.5, league_pool=pool)
        except ImportError as e:
            pytest.skip(f"solver6 import failed: {e}")
        N = 10_000
        hits = sum(1 for _ in range(N) if f._maybe_sample_league_opponent() is not None)
        rate = hits / N
        # 3-stderr binomial window for p=0.5, n=10k is ~0.015.
        assert 0.485 < rate < 0.515
