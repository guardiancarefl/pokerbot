"""Tests for scripts/build_league_registry.

The script is the canonical entry point for converting a training run's
checkpoint directory into LeaguePool-ready registry entries. Failures
here would silently break league-overnight launches.

Coverage:
  - File-pattern discovery (ckpt_iter_NNNN.pt match + sort order)
  - Skipping of non-matching files (final.pt, config.json, junk)
  - Iter parsing edge cases (zero-padding, large numbers, no padding)
  - Per-run name prefix derivation
  - Idempotent re-registration (same args twice = no change)
  - Conflict detection (same name, different path -> raise)
  - Multi-run registration in one invocation
  - Config tag pulled from <run_dir>/checkpoints/config.json when present
  - Empty-checkpoint-dir warning path
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.nlhe.checkpoint_registry import CheckpointRegistry
from scripts.build_league_registry import (
    discover_checkpoints,
    read_run_config_tag,
    register_run,
)


# ============================================================
# Fixtures
# ============================================================

def _make_run_dir(tmp_path: Path, name: str, iters: list[int],
                  include_junk: bool = False,
                  config_tag: str | None = None) -> Path:
    """Build a fake run directory with checkpoint files for testing."""
    run_dir = tmp_path / name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for it in iters:
        (ckpt_dir / f"ckpt_iter_{it:04d}.pt").write_bytes(b"\x00")
    if include_junk:
        (ckpt_dir / "final.pt").write_bytes(b"\x00")
        (ckpt_dir / "metrics.json").write_text("{}")
        (ckpt_dir / "config.json").write_text("{}")
        (ckpt_dir / "README.md").write_text("notes")
    if config_tag is not None:
        (ckpt_dir / "config.json").write_text(json.dumps({"tag": config_tag}))
    return run_dir


# ============================================================
# discover_checkpoints
# ============================================================

class TestDiscovery:
    def test_finds_all_matching(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-a", [100, 200, 300])
        found = discover_checkpoints(run_dir)
        assert len(found) == 3
        iters = [it for it, _ in found]
        assert iters == [100, 200, 300]

    def test_sorted_ascending_by_iter(self, tmp_path):
        # On disk in random order; discover_checkpoints must sort.
        run_dir = _make_run_dir(tmp_path, "run-b", [800, 100, 400, 200])
        found = discover_checkpoints(run_dir)
        iters = [it for it, _ in found]
        assert iters == [100, 200, 400, 800]

    def test_skips_junk_files(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-c", [100, 200], include_junk=True)
        found = discover_checkpoints(run_dir)
        iters = [it for it, _ in found]
        assert iters == [100, 200]  # final.pt, metrics.json etc. excluded

    def test_handles_no_zero_padding(self, tmp_path):
        # The trainer uses 4-digit padding, but the regex shouldn't care.
        run_dir = tmp_path / "run-d"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "checkpoints" / "ckpt_iter_5.pt").write_bytes(b"")
        (run_dir / "checkpoints" / "ckpt_iter_50.pt").write_bytes(b"")
        (run_dir / "checkpoints" / "ckpt_iter_500.pt").write_bytes(b"")
        (run_dir / "checkpoints" / "ckpt_iter_5000.pt").write_bytes(b"")
        found = discover_checkpoints(run_dir)
        iters = [it for it, _ in found]
        assert iters == [5, 50, 500, 5000]

    def test_missing_dir_raises(self, tmp_path):
        bogus = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError):
            discover_checkpoints(bogus)

    def test_empty_ckpt_dir_returns_empty(self, tmp_path):
        run_dir = tmp_path / "run-e"
        (run_dir / "checkpoints").mkdir(parents=True)
        found = discover_checkpoints(run_dir)
        assert found == []


# ============================================================
# read_run_config_tag
# ============================================================

class TestConfigTag:
    def test_reads_from_checkpoints_subdir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-f", [100],
                                config_tag="phase4f_dcfr_linear_overnight")
        tag = read_run_config_tag(run_dir)
        assert tag == "phase4f_dcfr_linear_overnight"

    def test_returns_none_when_no_config(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-g", [100])
        assert read_run_config_tag(run_dir) is None

    def test_returns_none_on_malformed_json(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-h", [100])
        (run_dir / "checkpoints" / "config.json").write_text("not json{{{")
        assert read_run_config_tag(run_dir) is None

    def test_returns_none_when_tag_missing(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-i", [100])
        (run_dir / "checkpoints" / "config.json").write_text(
            json.dumps({"n_iterations": 3400})
        )
        assert read_run_config_tag(run_dir) is None


# ============================================================
# register_run
# ============================================================

class TestRegisterRun:
    def test_basic_registration(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-j", [200, 400, 600])
        reg = CheckpointRegistry()
        added, idem = register_run(
            registry=reg,
            run_dir=run_dir,
            name_prefix="dcfr-overnight",
            tags=["dcfr"],
        )
        assert added == 3
        assert idem == 0
        assert len(reg) == 3
        assert "dcfr-overnight-200" in reg
        assert "dcfr-overnight-400" in reg
        assert "dcfr-overnight-600" in reg

    def test_idempotent_re_register(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-k", [200, 400])
        reg = CheckpointRegistry()
        register_run(reg, run_dir, "dcfr-overnight", ["dcfr"])
        # Second pass with same args: all hits idempotent, no adds.
        added, idem = register_run(reg, run_dir, "dcfr-overnight", ["dcfr"])
        assert added == 0
        assert idem == 2

    def test_conflict_same_name_different_path(self, tmp_path):
        run_a = _make_run_dir(tmp_path, "run-a", [200])
        run_b = _make_run_dir(tmp_path, "run-b", [200])
        reg = CheckpointRegistry()
        register_run(reg, run_a, "dcfr", ["dcfr"])
        # run-b's ckpt_iter_0200 has a different path; same prefix means
        # same name -> conflict.
        with pytest.raises(ValueError, match="conflict"):
            register_run(reg, run_b, "dcfr", ["dcfr"])

    def test_different_prefixes_no_conflict(self, tmp_path):
        run_a = _make_run_dir(tmp_path, "run-a", [200])
        run_b = _make_run_dir(tmp_path, "run-b", [200])
        reg = CheckpointRegistry()
        register_run(reg, run_a, "dcfr-overnight", ["dcfr"])
        register_run(reg, run_b, "dcfr-shake", ["dcfr", "shakedown"])
        assert "dcfr-overnight-200" in reg
        assert "dcfr-shake-200" in reg
        assert len(reg) == 2

    def test_metadata_includes_iter_and_run_dir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-m", [400])
        reg = CheckpointRegistry()
        register_run(reg, run_dir, "dcfr", ["dcfr"])
        entry = reg.get("dcfr-400")
        assert entry.metadata["iter"] == 400
        assert entry.metadata["run_dir"] == str(run_dir)

    def test_metadata_includes_config_tag_when_available(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-n", [400],
                                config_tag="phase4f_dcfr_linear_overnight")
        reg = CheckpointRegistry()
        register_run(reg, run_dir, "dcfr", ["dcfr"])
        entry = reg.get("dcfr-400")
        assert entry.metadata["config_tag"] == "phase4f_dcfr_linear_overnight"

    def test_tags_applied(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-o", [200])
        reg = CheckpointRegistry()
        register_run(reg, run_dir, "dcfr", ["dcfr", "overnight"])
        entry = reg.get("dcfr-200")
        assert sorted(entry.tags) == ["dcfr", "overnight"]

    def test_dry_run_does_not_modify(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-p", [200, 400])
        reg = CheckpointRegistry()
        added, idem = register_run(
            reg, run_dir, "dcfr", ["dcfr"], dry_run=True,
        )
        # dry_run: counters stay 0, registry untouched.
        assert added == 0
        assert idem == 0
        assert len(reg) == 0

    def test_ascending_iter_order_preserved(self, tmp_path):
        # LeaguePool's recency sampling depends on registration order
        # reflecting checkpoint age.
        run_dir = _make_run_dir(tmp_path, "run-q", [800, 200, 400, 600])
        reg = CheckpointRegistry()
        register_run(reg, run_dir, "dcfr", ["dcfr"])
        names = reg.names()
        # Names() reflects insertion order; insertion is sorted-by-iter.
        assert names == ["dcfr-200", "dcfr-400", "dcfr-600", "dcfr-800"]


# ============================================================
# End-to-end save/load round-trip
# ============================================================

class TestRoundTrip:
    def test_save_then_load(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "run-r", [200, 400, 600])
        reg = CheckpointRegistry()
        register_run(reg, run_dir, "dcfr-overnight", ["dcfr", "overnight"])
        out = tmp_path / "registry.json"
        reg.save(str(out))

        loaded = CheckpointRegistry.load(str(out))
        assert len(loaded) == 3
        assert sorted(loaded.names()) == sorted(reg.names())
        for name in reg.names():
            assert loaded.get(name) == reg.get(name)

    def test_load_missing_returns_empty(self, tmp_path):
        out = tmp_path / "nothing-here.json"
        loaded = CheckpointRegistry.load(str(out))
        assert len(loaded) == 0
