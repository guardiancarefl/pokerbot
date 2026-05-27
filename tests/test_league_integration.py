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
        f.archetype_pool = None  # league-only fixture: no archetype source
        f.rng = random.Random(2026)
        f._maybe_sample_league_opponent = (
            DeepCFR6MaxSolver._maybe_sample_league_opponent.__get__(f, Fake)
        )
        # Dashboard (Step 7-pre): _maybe_sample_league_opponent now calls
        # self._count_override. A real solver always has it + _override_counts
        # (set in __init__); bind them on the fake so the control flow runs.
        # Counting draws no rng, so these tests still validate bit-identity.
        f._override_counts = {"archetype": 0, "league": 0, "self_play": 0}
        f._count_override = DeepCFR6MaxSolver._count_override.__get__(f, Fake)
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


# ============================================================
# Phase 5-A: combined archetype + league three-way override sampling.
# ============================================================

def _classify(r, archetype_mix, league_mix):
    """Pure mirror of the three-way band logic in
    _maybe_sample_league_opponent. Band ordering (load-bearing):
        archetype  [0, archetype_mix)
        league     [archetype_mix, archetype_mix + league_mix)
        self_play  [archetype_mix + league_mix, 1)
    """
    if r < archetype_mix:
        return "archetype"
    if r < archetype_mix + league_mix:
        return "league"
    return "self_play"


class TestCombinedSampling:
    def test_classify_band_ordering(self):
        A, L = 0.3, 0.2
        assert _classify(0.0, A, L) == "archetype"
        assert _classify(0.29, A, L) == "archetype"
        assert _classify(0.30, A, L) == "league"      # boundary: r==A → league band
        assert _classify(0.49, A, L) == "league"
        assert _classify(0.50, A, L) == "self_play"   # boundary: r==A+L → self-play
        assert _classify(0.999, A, L) == "self_play"

    def test_classify_at_specific_band_point(self):
        # Explicit ordering test: at A=0.3, L=0.2, r=0.25 lands in the
        # ARCHETYPE band, not league. Catches a future band-reorder bug.
        assert _classify(0.25, 0.3, 0.2) == "archetype"

    def test_classify_archetype_zero_collapses_to_league_gate(self):
        # archetype_mix=0.0 → empty archetype band → league band is [0, L),
        # exactly the pre-Phase-5-A single-gate condition.
        A, L = 0.0, 0.5
        assert _classify(0.0, A, L) == "league"
        assert _classify(0.49, A, L) == "league"
        assert _classify(0.50, A, L) == "self_play"

    def test_monte_carlo_three_way_proportions(self):
        A, L = 0.3, 0.2
        rng = random.Random(2026)
        N = 10_000
        counts = {"archetype": 0, "league": 0, "self_play": 0}
        for _ in range(N):
            counts[_classify(rng.random(), A, L)] += 1
        # 3-stderr binomial windows (~0.014 for p~0.3 at n=10k); ±0.02 is safe.
        assert abs(counts["archetype"] / N - 0.3) < 0.02
        assert abs(counts["league"] / N - 0.2) < 0.02
        assert abs(counts["self_play"] / N - 0.5) < 0.02


class TestCombinedSamplerBitIdentity:
    """At archetype_mix=0.0 the combined sampler must consume rng and emit
    Policy/None outcomes byte-for-byte identically to the pre-Phase-5-A
    league-only gate. This is the structural guard for the bit-identity smoke.
    """

    def _make_fake(self, archetype_mix, league_mix, league_pool, seed):
        from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max

        class Fake:
            pass

        f = Fake()
        f.cfg = TrainConfig6Max(archetype_mix=archetype_mix, league_mix=league_mix)
        f.league_pool = league_pool
        f.archetype_pool = None  # bit-identity fixture: archetype_mix=0 → no pool
        f.rng = random.Random(seed)
        f._maybe_sample_league_opponent = (
            DeepCFR6MaxSolver._maybe_sample_league_opponent.__get__(f, Fake)
        )
        # Dashboard (Step 7-pre): _maybe_sample_league_opponent now calls
        # self._count_override. A real solver always has it + _override_counts
        # (set in __init__); bind them on the fake so the control flow runs.
        # Counting draws no rng, so these tests still validate bit-identity.
        f._override_counts = {"archetype": 0, "league": 0, "self_play": 0}
        f._count_override = DeepCFR6MaxSolver._count_override.__get__(f, Fake)
        return f

    def test_bit_identity_at_archetype_zero(self):
        try:
            from src.nlhe.solver6 import TrainConfig6Max  # noqa: F401
        except ImportError as e:
            pytest.skip(f"solver6 import failed: {e}")

        L = 0.5
        # A sample_opponent that consumes exactly one rng draw, so old/new
        # streams stay in lockstep when the league band fires.
        def sample(rng):
            return ("league", rng.random())

        # "Old" reference: single self.rng.random() < L gate (pre-Phase-5-A).
        old_rng = random.Random(777)
        old_out = []
        for _ in range(1000):
            if old_rng.random() < L:
                old_out.append(sample(old_rng))
            else:
                old_out.append(None)

        # "New" combined sampler at archetype_mix=0.0, same pool/seed.
        new_pool = MagicMock()
        new_pool.sample_opponent = MagicMock(side_effect=sample)
        f = self._make_fake(archetype_mix=0.0, league_mix=L,
                            league_pool=new_pool, seed=777)
        new_out = [f._maybe_sample_league_opponent() for _ in range(1000)]

        assert new_out == old_out
        # Identical rng state after N calls → no extra/missing draws.
        assert f.rng.getstate() == old_rng.getstate()

    def test_no_source_active_does_not_draw_rng(self):
        # archetype_mix=0, league_mix=0, no pool → short-circuit, NO draw.
        f = self._make_fake(archetype_mix=0.0, league_mix=0.0,
                            league_pool=None, seed=1234)
        before = f.rng.getstate()
        for _ in range(100):
            assert f._maybe_sample_league_opponent() is None
        assert f.rng.getstate() == before  # rng untouched


class TestArchetypeConfig:
    def test_archetype_mix_defaults_zero(self):
        from src.nlhe.solver6 import TrainConfig6Max
        assert TrainConfig6Max().archetype_mix == 0.0

    def test_sum_constraint_raises(self):
        from src.nlhe.solver6 import TrainConfig6Max
        with pytest.raises(ValueError, match=r"archetype_mix \+ league_mix"):
            TrainConfig6Max(archetype_mix=0.6, league_mix=0.6)

    def test_sum_constraint_boundary_ok(self):
        from src.nlhe.solver6 import TrainConfig6Max
        cfg = TrainConfig6Max(archetype_mix=0.5, league_mix=0.5)  # sum == 1.0 OK
        assert cfg.archetype_mix == 0.5 and cfg.league_mix == 0.5

    def test_archetype_mix_out_of_range_raises(self):
        from src.nlhe.solver6 import TrainConfig6Max
        with pytest.raises(ValueError, match="archetype_mix must be in"):
            TrainConfig6Max(archetype_mix=1.5)
