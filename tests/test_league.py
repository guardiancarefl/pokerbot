"""Tests for CheckpointRegistry and LeaguePool.

Coverage:
  CheckpointRegistry — CRUD, idempotent register, conflict detection,
    tag queries (any/all match), JSON save/load round-trip,
    missing-file graceful load, version validation.
  LeaguePool — uniform/weighted/recency sampling distributions,
    tag filter, validation errors, lazy-load cache, clear_cache,
    end-to-end sample_opponent against the production abstraction
    (auto-skips if pickle absent, mirrors test_cfr6 pattern).
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from src.nlhe.checkpoint_registry import (
    CheckpointEntry,
    CheckpointRegistry,
)
from src.nlhe.league_pool import LeaguePool


ABSTRACTION_PATH = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCTURE_PATH = "configs/ignition_double_up_6max_turbo.yaml"


# ============================================================
# CheckpointRegistry
# ============================================================

class TestRegistryBasics:
    def test_empty_registry(self):
        r = CheckpointRegistry()
        assert len(r) == 0
        assert r.names() == []
        assert list(r) == []

    def test_register_one(self):
        r = CheckpointRegistry()
        e = r.register("ckpt-a", "path/a.pt", metadata={"iter": 100}, tags=["dcfr"])
        assert e.name == "ckpt-a"
        assert len(r) == 1
        assert "ckpt-a" in r

    def test_register_idempotent_on_identical(self):
        r = CheckpointRegistry()
        r.register("a", "path/a.pt", metadata={"x": 1}, tags=["t"])
        r.register("a", "path/a.pt", metadata={"x": 1}, tags=["t"])
        assert len(r) == 1

    def test_register_conflict_raises(self):
        r = CheckpointRegistry()
        r.register("a", "path/a.pt")
        with pytest.raises(ValueError, match="already registered"):
            r.register("a", "path/different.pt")

    def test_unregister(self):
        r = CheckpointRegistry()
        r.register("a", "path/a.pt")
        r.unregister("a")
        assert len(r) == 0
        with pytest.raises(KeyError):
            r.unregister("a")

    def test_get_missing_raises(self):
        r = CheckpointRegistry()
        with pytest.raises(KeyError):
            r.get("nonexistent")


class TestRegistryQuery:
    @pytest.fixture
    def populated(self):
        r = CheckpointRegistry()
        r.register("dcfr-100", "p/d100.pt", tags=["dcfr", "shakedown"])
        r.register("dcfr-200", "p/d200.pt", tags=["dcfr", "shakedown"])
        r.register("vanilla-100", "p/v100.pt", tags=["vanilla", "shakedown"])
        r.register("vanilla-overnight", "p/vo.pt", tags=["vanilla", "overnight"])
        return r

    def test_query_no_tags_returns_all(self, populated):
        assert len(populated.query()) == 4
        assert len(populated.query(tags=[])) == 4

    def test_query_any_match(self, populated):
        dcfr_only = populated.query(tags=["dcfr"])
        assert {e.name for e in dcfr_only} == {"dcfr-100", "dcfr-200"}

        either = populated.query(tags=["dcfr", "overnight"], match="any")
        assert {e.name for e in either} == {"dcfr-100", "dcfr-200", "vanilla-overnight"}

    def test_query_all_match(self, populated):
        dcfr_and_shake = populated.query(
            tags=["dcfr", "shakedown"], match="all"
        )
        assert {e.name for e in dcfr_and_shake} == {"dcfr-100", "dcfr-200"}

        none_match = populated.query(
            tags=["dcfr", "overnight"], match="all"
        )
        assert none_match == []

    def test_query_invalid_match_raises(self, populated):
        with pytest.raises(ValueError, match="match must be"):
            populated.query(tags=["dcfr"], match="bogus")


class TestRegistryPersistence:
    def test_roundtrip(self, tmp_path):
        r = CheckpointRegistry()
        r.register("a", "p/a.pt", metadata={"iter": 100, "note": "foo"}, tags=["x", "y"])
        r.register("b", "p/b.pt", tags=["y"])

        path = tmp_path / "reg.json"
        r.save(str(path))

        r2 = CheckpointRegistry.load(str(path))
        assert len(r2) == 2
        assert r2.get("a") == r.get("a")
        assert r2.get("b") == r.get("b")

    def test_load_missing_returns_empty(self, tmp_path):
        r = CheckpointRegistry.load(str(tmp_path / "does_not_exist.json"))
        assert len(r) == 0

    def test_save_creates_parent_dir(self, tmp_path):
        r = CheckpointRegistry()
        r.register("a", "p/a.pt")
        nested = tmp_path / "a" / "b" / "c" / "reg.json"
        r.save(str(nested))
        assert nested.exists()

    def test_load_rejects_unknown_version(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"version": 999, "entries": []}))
        with pytest.raises(ValueError, match="unsupported registry version"):
            CheckpointRegistry.load(str(path))


# ============================================================
# LeaguePool — sampling (no solver loads)
# ============================================================

@pytest.fixture
def reg5():
    r = CheckpointRegistry()
    for i, name in enumerate(["a", "b", "c", "d", "e"]):
        r.register(
            name,
            f"path/{name}.pt",
            tags=["dcfr" if i % 2 == 0 else "vanilla"],
        )
    return r


class TestLeaguePoolUniform:
    def test_eligible_count(self, reg5):
        pool = LeaguePool(reg5, abstraction=None, structure=None)
        assert len(pool) == 5
        assert set(pool.eligible_names()) == {"a", "b", "c", "d", "e"}

    def test_uniform_balanced(self, reg5):
        pool = LeaguePool(reg5, abstraction=None, structure=None, sample_strategy="uniform")
        rng = random.Random(42)
        counts = {}
        for _ in range(10000):
            e = pool.sample_entry(rng)
            counts[e.name] = counts.get(e.name, 0) + 1
        for name, c in counts.items():
            assert 1700 < c < 2300, f"{name}={c} not balanced"


class TestLeaguePoolWeighted:
    def test_weights_respected(self, reg5):
        weights = {"a": 5, "b": 1, "c": 0, "d": 1, "e": 5}
        pool = LeaguePool(reg5, abstraction=None, structure=None,
                          sample_strategy="weighted", weights=weights)
        rng = random.Random(42)
        counts = {n: 0 for n in "abcde"}
        for _ in range(12000):
            counts[pool.sample_entry(rng).name] += 1
        assert 4500 < counts["a"] < 5500
        assert 4500 < counts["e"] < 5500
        assert counts["c"] == 0

    def test_missing_weights_raises(self, reg5):
        with pytest.raises(ValueError, match="requires non-empty weights"):
            LeaguePool(reg5, abstraction=None, structure=None,
                       sample_strategy="weighted")

    def test_unknown_weight_key_raises(self, reg5):
        with pytest.raises(ValueError, match="not in eligible"):
            LeaguePool(reg5, abstraction=None, structure=None,
                       sample_strategy="weighted",
                       weights={"nonexistent": 1.0})

    def test_negative_weight_raises(self, reg5):
        with pytest.raises(ValueError, match="non-negative"):
            LeaguePool(reg5, abstraction=None, structure=None,
                       sample_strategy="weighted",
                       weights={"a": -1.0})

    def test_all_zero_weights_raises_at_sample_time(self, reg5):
        pool = LeaguePool(reg5, abstraction=None, structure=None,
                          sample_strategy="weighted",
                          weights={"a": 0, "b": 0, "c": 0, "d": 0, "e": 0})
        rng = random.Random(42)
        with pytest.raises(RuntimeError, match="all sampling weights are zero"):
            pool.sample_entry(rng)


class TestLeaguePoolRecency:
    def test_monotonic_by_age(self, reg5):
        pool = LeaguePool(reg5, abstraction=None, structure=None,
                          sample_strategy="recency", recency_halflife=2.0)
        rng = random.Random(42)
        counts = {n: 0 for n in "abcde"}
        for _ in range(10000):
            counts[pool.sample_entry(rng).name] += 1
        # Oldest "a" -> newest "e" should be monotonically increasing.
        assert counts["a"] < counts["b"] < counts["c"] < counts["d"] < counts["e"]


class TestLeaguePoolValidation:
    def test_invalid_strategy_raises(self, reg5):
        with pytest.raises(ValueError, match="sample_strategy must be"):
            LeaguePool(reg5, abstraction=None, structure=None,
                       sample_strategy="bogus")

    def test_empty_pool_raises_at_sample(self):
        empty = CheckpointRegistry()
        pool = LeaguePool(empty, abstraction=None, structure=None)
        rng = random.Random(42)
        with pytest.raises(RuntimeError, match="no eligible"):
            pool.sample_entry(rng)

    def test_tag_filter_restricts_pool(self, reg5):
        pool = LeaguePool(reg5, abstraction=None, structure=None,
                          sample_strategy="uniform", tag_filter=["dcfr"])
        assert pool.eligible_names() == ["a", "c", "e"]

    def test_tag_filter_with_no_match_gives_empty_pool(self, reg5):
        pool = LeaguePool(reg5, abstraction=None, structure=None,
                          sample_strategy="uniform", tag_filter=["nonexistent"])
        assert len(pool) == 0


# ============================================================
# LeaguePool — sample_opponent end-to-end (requires real artifacts)
# ============================================================

@pytest.fixture(scope="module")
def real_abstraction():
    if not Path(ABSTRACTION_PATH).exists():
        pytest.skip(f"abstraction.pkl not found at {ABSTRACTION_PATH}")
    from src.nlhe.abstraction import Abstraction
    return Abstraction.load(ABSTRACTION_PATH)


@pytest.fixture(scope="module")
def real_structure():
    if not Path(STRUCTURE_PATH).exists():
        pytest.skip(f"structure not found at {STRUCTURE_PATH}")
    from src.nlhe.game_strings import TournamentStructure
    return TournamentStructure.from_yaml(STRUCTURE_PATH)


@pytest.fixture
def real_ckpt():
    """Use any real checkpoint from a Session 10 run."""
    for candidate in [
        "runs/six_max_20260524_005853_phase4f_dcfr_linear_shakedown/checkpoints/ckpt_iter_0200.pt",
        "runs/six_max_20260523_224646_phase4f_overnight/checkpoints/ckpt_iter_0200.pt",
    ]:
        if Path(candidate).exists():
            return candidate
    pytest.skip("no real checkpoint available for end-to-end test")


class TestLeaguePoolEndToEnd:
    def test_sample_opponent_loads_real_checkpoint(
        self, real_abstraction, real_structure, real_ckpt
    ):
        r = CheckpointRegistry()
        r.register("real", real_ckpt, tags=["test"])
        pool = LeaguePool(r, real_abstraction, real_structure)
        rng = random.Random(42)
        policy = pool.sample_opponent(rng)
        assert policy.name == "real"
        # Required to conform to Policy protocol from eval_pool.py
        assert hasattr(policy, "select_action")

    def test_lazy_load_caches_policy(
        self, real_abstraction, real_structure, real_ckpt
    ):
        r = CheckpointRegistry()
        r.register("real", real_ckpt, tags=["test"])
        pool = LeaguePool(r, real_abstraction, real_structure)
        rng = random.Random(42)
        p1 = pool.sample_opponent(rng)
        p2 = pool.sample_opponent(rng)
        assert p1 is p2, "second sample should return the cached policy"
        info = pool.cache_info()
        assert info["cached_count"] == 1
        assert info["cached_names"] == ["real"]

    def test_clear_cache_empties(
        self, real_abstraction, real_structure, real_ckpt
    ):
        r = CheckpointRegistry()
        r.register("real", real_ckpt, tags=["test"])
        pool = LeaguePool(r, real_abstraction, real_structure)
        rng = random.Random(42)
        pool.sample_opponent(rng)
        assert pool.cache_info()["cached_count"] == 1
        pool.clear_cache()
        assert pool.cache_info()["cached_count"] == 0
