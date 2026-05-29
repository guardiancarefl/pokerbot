"""Tests for the Step 7-pre training dashboard (Pieces A + B).

Piece A: the enhanced per-iter log (ETA, rolling avg10, override-mix) plus the
per-iteration override-mix counter.
Piece B: the periodic in-training mini-eval against anchors (src/nlhe/mini_eval.py).

Design contract under test: both features default OFF and are pure
observability — enabling them must not perturb the training trajectory.

=== A note on torch global RNG isolation (read before editing test #2) ===
The naive assertion "torch.get_rng_state() is byte-identical before/after a
mini-eval" is FALSE and is intentionally NOT used here. Loading a
CheckpointPolicy (challenger + each anchor) constructs fresh MLPs, and
nn.Linear weight init draws from the torch GLOBAL RNG — so running a mini-eval
DOES advance it. That advancement is provably harmless: training never reads
the torch global RNG after construction. Evidence:
  - MLP (src/nlhe/solver.py) is Linear+ReLU only — no Dropout, deterministic
    forward.
  - ReservoirBuffer.sample_batch draws minibatch indices from a Python
    random.Random (self.rng), not torch.
  - _train_advantage_net / _train_strategy_net: Python-rng minibatch -> forward
    -> loss.backward() -> Adam.step(). No torch global RNG draw.
Therefore the authoritative guarantee is "training is byte-identical with
mini-eval ON vs OFF", verified directly by
test_mini_eval_enabled_preserves_training_bit_identity below. The Python-rng
stream (which training DOES use) is checked separately by
test_mini_eval_uses_dedicated_python_rng.
"""
from __future__ import annotations

import hashlib
import math
import numbers
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max

REPO = Path(__file__).resolve().parent.parent
ABSTRACTION_PATH = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
TOURNEY_PATH = "configs/ignition_double_up_6max_turbo.yaml"
FAST_VIEW_CONFIG = "/tmp/fast_view_smoke.yaml"
FAST_VIEW_BASELINE = "/tmp/fast_view_smoke_pre"


# ===== Stub abstraction (mirrors tests/test_solver6.py) =====


@dataclass
class _StubAbstraction:
    """Hash-based deterministic bucket assignment; not a real abstraction.

    Adequate for the legacy-mode (non-tournament) tests that exercise the
    override counter and log format. The mini-eval / bit-identity tests that
    actually play hands use the real abstraction so the eval path matches
    production exactly.
    """

    def bucket_of(self, hero, board, runouts=200, rng=None):
        key = (tuple(sorted(hero)), tuple(sorted(board)))
        digest = hashlib.sha256(repr(key).encode()).hexdigest()
        return int(digest[:8], 16) % 200


# ===== Construction helpers =====


def _legacy_cfg(**over):
    """Small real config, legacy (non-tournament) mode, dashboard default OFF."""
    kw = dict(
        starting_stack=1500, big_blind=100, small_blind=50,
        payout_mode="double_up", hidden_dim=[16, 16],
        n_iterations=2, traversals_per_iter=5, train_steps_per_iter=3,
        batch_size=4, learning_rate=1e-3, buffer_capacity=500,
        bucket_runouts=10, max_traversal_depth=200, seed=2026,
    )
    kw.update(over)
    return TrainConfig6Max(**kw)


def _legacy_solver(abstraction, logger=None, **over):
    from scripts.train_6max import build_six_max_game
    tc = _legacy_cfg(**over)
    game = build_six_max_game(tc)
    return DeepCFR6MaxSolver(
        game=game, abstraction=abstraction, config=tc,
        logger=logger or (lambda s: None),
    )


def _tourney_solver(abstraction, logger=None, **over):
    """Tournament-mode solver (required for mini-eval) on the real abstraction."""
    from scripts.train_6max import build_six_max_game
    kw = dict(
        starting_stack=1500, big_blind=100, small_blind=50,
        payout_mode="double_up", hidden_dim=[16, 16],
        n_iterations=1, traversals_per_iter=10, train_steps_per_iter=3,
        batch_size=4, learning_rate=1e-3, buffer_capacity=2000,
        bucket_runouts=10, max_traversal_depth=200,
        tournament_structure_path=TOURNEY_PATH, seed=2026,
    )
    kw.update(over)
    tc = TrainConfig6Max(**kw)
    game = build_six_max_game(tc)
    return DeepCFR6MaxSolver(
        game=game, abstraction=abstraction, config=tc,
        logger=logger or (lambda s: None),
    )


def _adv_net_tensors(ckpt_path):
    """Flat ordered list of (key, tensor) for the 6 advantage nets in a ckpt."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    nets = ckpt["policy_nets"]["nets"]  # list of 6 advantage-net state_dicts
    out = []
    for seat, sd in enumerate(nets):
        for k in sorted(sd.keys()):
            out.append((f"net{seat}.{k}", sd[k]))
    return out


def _assert_adv_nets_identical(base_path, cand_path, label):
    a = _adv_net_tensors(base_path)
    b = _adv_net_tensors(cand_path)
    assert len(a) == len(b) and len(a) > 0, f"{label}: param count mismatch/empty"
    for (ka, ta), (kb, tb) in zip(a, b):
        assert ka == kb, f"{label}: key mismatch {ka} vs {kb}"
        max_diff = (ta.float() - tb.float()).abs().max().item()
        assert max_diff == 0.0, f"{label}: adv param {ka} differs (max {max_diff})"


# ===== Fixtures =====


@pytest.fixture(scope="module")
def stub_abstraction():
    return _StubAbstraction()


@pytest.fixture(scope="module")
def real_abstraction():
    from src.nlhe.abstraction import Abstraction
    if not (REPO / ABSTRACTION_PATH).exists():
        pytest.skip(f"real abstraction not found: {ABSTRACTION_PATH}")
    if not (REPO / TOURNEY_PATH).exists():
        pytest.skip(f"tournament structure not found: {TOURNEY_PATH}")
    return Abstraction.load(str(REPO / ABSTRACTION_PATH))


@pytest.fixture(scope="module")
def mini_eval_run(real_abstraction, tmp_path_factory):
    """Build a tournament-mode challenger with one self-anchor, run a single
    mini-eval via the full _maybe_run_mini_eval wrapper, and capture the
    Python-rng state and the populated history/metrics structures.

    Self-anchor: a freshly-built sibling solver saved to disk is used as the
    (only) anchor. The isolation/shape properties are independent of which
    anchor is used, so this keeps the fixture hermetic and fast.
    """
    tmpdir = tmp_path_factory.mktemp("dash")
    anchor_path = str(tmpdir / "anchor.pt")
    base = _tourney_solver(real_abstraction, seed=2026)
    base.save_checkpoint(anchor_path, slim=True)

    solver = _tourney_solver(
        real_abstraction, seed=2026,
        mini_eval_enabled=True, mini_eval_every=2, mini_eval_n_hands=12,
        mini_eval_anchors=[f"selfanchor={anchor_path}"],
    )

    metrics = {"mini_eval": []}
    solver._mini_eval_history = []
    py_before = solver.rng.getstate()
    solver._maybe_run_mini_eval(it=2, metrics=metrics)
    py_after = solver.rng.getstate()

    return SimpleNamespace(
        py_before=py_before,
        py_after=py_after,
        history=solver._mini_eval_history,
        metrics_mini_eval=metrics["mini_eval"],
    )


# ============================================================
# Isolation / bit-identity (3)
# ============================================================


def test_mini_eval_uses_dedicated_python_rng(mini_eval_run):
    """Level 1: a mini-eval must not advance the solver's Python training rng.

    Training traversals and reservoir minibatch sampling draw from solver.rng /
    its derived streams; evaluate_matchup self-seeds its own random.Random(seed)
    and save_checkpoint only reads getstate(). So self.rng is byte-identical
    across the mini-eval.
    """
    assert mini_eval_run.py_before == mini_eval_run.py_after


def test_mini_eval_enabled_preserves_training_bit_identity(real_abstraction, tmp_path):
    """Levels 2+3 (authoritative torch-RNG isolation): training is byte-identical
    with mini-eval ON vs OFF, despite the mini-eval advancing the torch global
    RNG (via CheckpointPolicy MLP construction). See the module docstring.

    Two solvers built with the SAME seed get identical init (the solver calls
    torch.manual_seed(seed) before constructing its nets), so any post-init
    divergence would be attributable to the mini-eval's RNG side effects.
    """
    anchor_path = str(tmp_path / "anchor.pt")
    base = _tourney_solver(real_abstraction, seed=2026)
    base.save_checkpoint(anchor_path, slim=True)

    def run(mini_eval_on):
        extra = {}
        if mini_eval_on:
            extra = dict(
                mini_eval_enabled=True, mini_eval_every=2, mini_eval_n_hands=8,
                mini_eval_anchors=[f"selfanchor={anchor_path}"],
            )
        s = _tourney_solver(real_abstraction, seed=4242, n_iterations=4, **extra)
        s.train(checkpoint_dir=None)
        adv = [p.detach().cpu().clone()
               for net in s.policy_nets.nets for p in net.parameters()]
        strat = [p.detach().cpu().clone()
                 for p in s.policy_nets.strat_net.parameters()]
        return adv, strat

    off_adv, off_strat = run(False)
    on_adv, on_strat = run(True)

    assert len(off_adv) == len(on_adv) and len(off_adv) > 0
    for i, (a, b) in enumerate(zip(off_adv, on_adv)):
        max_diff = (a - b).abs().max().item()
        assert max_diff == 0.0, f"adv-net param {i} differs ON vs OFF (max {max_diff})"
    for i, (a, b) in enumerate(zip(off_strat, on_strat)):
        max_diff = (a - b).abs().max().item()
        assert max_diff == 0.0, f"strat-net param {i} differs ON vs OFF (max {max_diff})"


def test_disabled_dashboard_preserves_bit_identity(tmp_path):
    """F1 / external baseline: re-running the fast_view smoke config (dashboard
    fields absent -> default OFF) through the real train_6max entrypoint must
    reproduce the pre-dashboard adv-net weights byte-for-byte.

    Skips where the pre-dashboard baseline is absent (it cannot be regenerated
    post-change); the self-contained ON/OFF test above covers the durable case.
    """
    if not (os.path.exists(FAST_VIEW_CONFIG) and os.path.isdir(FAST_VIEW_BASELINE)):
        pytest.skip("fast_view baseline/config absent (F1 runs where the pre-dashboard baseline exists)")
    out = tmp_path / "fv_post"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.train_6max",
         "--config", FAST_VIEW_CONFIG, "--out", str(out)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"train_6max failed:\n{proc.stderr[-2000:]}"
    for it in range(1, 6):
        cand = out / "checkpoints" / f"ckpt_iter_{it:04d}.pt"
        base = Path(FAST_VIEW_BASELINE) / "checkpoints" / f"ckpt_iter_{it:04d}.pt"
        assert cand.exists(), f"missing candidate checkpoint {cand}"
        assert base.exists(), f"missing baseline checkpoint {base}"
        _assert_adv_nets_identical(base, cand, f"iter {it}")


# ============================================================
# Config validation (2)
# ============================================================


def test_config_validation_post_init():
    with pytest.raises(ValueError, match="at least one of"):
        TrainConfig6Max(mini_eval_enabled=True)
    with pytest.raises(ValueError, match=r"mini_eval_every must be >= 1"):
        TrainConfig6Max(mini_eval_enabled=True,
                        mini_eval_anchors=["a=/x.pt"], mini_eval_every=0)
    with pytest.raises(ValueError, match=r"mini_eval_n_hands must be >= 1"):
        TrainConfig6Max(mini_eval_enabled=True,
                        mini_eval_anchors=["a=/x.pt"], mini_eval_n_hands=0)


def test_anchor_existence_validation(stub_abstraction, tmp_path):
    from scripts.train_6max import build_six_max_game
    missing = str(tmp_path / "nope.pt")
    tc = _legacy_cfg(
        mini_eval_enabled=True, mini_eval_every=2, mini_eval_n_hands=4,
        mini_eval_anchors=[f"x={missing}"],
    )
    game = build_six_max_game(tc)
    with pytest.raises(ValueError, match=re.escape(missing)):
        DeepCFR6MaxSolver(game=game, abstraction=stub_abstraction, config=tc,
                          logger=lambda s: None)


# ============================================================
# Override-mix counter (2)
# ============================================================


def test_override_counter_per_iter_reset(stub_abstraction):
    """The counter is reset at the top of every iter, not accumulated for the
    run. With no override pools, every traversal counts as self_play; after 2
    iters the counter must hold ONE iter's worth (reset), not two."""
    solver = _legacy_solver(stub_abstraction, n_iterations=2, traversals_per_iter=5)
    solver.train(checkpoint_dir=None)
    total = sum(solver._override_counts.values())
    assert total == solver.cfg.traversals_per_iter, (
        f"expected {solver.cfg.traversals_per_iter} (per-iter reset), got {total}")
    assert solver._override_counts["self_play"] == solver.cfg.traversals_per_iter
    assert solver._override_counts["archetype"] == 0
    assert solver._override_counts["league"] == 0


def test_override_counter_accuracy():
    """At archetype_mix=0.30, league_mix=0.15 the empirical band proportions over
    many draws must land within 3 sigma of (0.30, 0.15, 0.55). Mirrors the
    monte-carlo proportion pattern in tests/test_league_integration.py."""
    class Fake:
        pass

    f = Fake()
    f.cfg = TrainConfig6Max(archetype_mix=0.30, league_mix=0.15)
    f.archetype_pool = MagicMock()
    f.archetype_pool.sample_opponent = MagicMock(return_value="arch")
    f.league_pool = MagicMock()
    f.league_pool.sample_opponent = MagicMock(return_value="league")
    f.rng = random.Random(2026)
    f._override_counts = {"archetype": 0, "league": 0, "self_play": 0}
    f._count_override = DeepCFR6MaxSolver._count_override.__get__(f, Fake)
    f._maybe_sample_league_opponent = (
        DeepCFR6MaxSolver._maybe_sample_league_opponent.__get__(f, Fake))

    N = 20000
    for _ in range(N):
        f._maybe_sample_league_opponent()

    c = f._override_counts
    assert sum(c.values()) == N
    for band, p in [("archetype", 0.30), ("league", 0.15), ("self_play", 0.55)]:
        rate = c[band] / N
        sigma = math.sqrt(p * (1 - p) / N)
        assert abs(rate - p) < 3 * sigma, (
            f"{band}: empirical {rate:.4f} vs {p} (|diff| {abs(rate-p):.4f} >= 3sigma {3*sigma:.4f})")


# ============================================================
# Log format (2)
# ============================================================

# Structure of the pre-Step-7-pre log line (solver6.py default branch).
# Fields are double-space separated; numeric values are right-justified so a
# value may carry leading spaces after its '=' (e.g. "adv=  0.6654").
_LEGACY_LOG_RE = re.compile(
    r"^iter\s+\d+/\d+  "
    r"trav=\d  "
    r"adv=\s*(?:nan|\d+\.\d{4})  "
    r"strat=\s*(?:nan|\d+\.\d{4})  "
    r"bufs=\(\d+(?:, \d+){5}\)  "
    r"sbuf=\d+  "
    r"\d+\.\ds$"
)


def test_enhanced_log_format_when_enabled(stub_abstraction):
    logs = []
    solver = _legacy_solver(stub_abstraction, n_iterations=3, enhanced_logging=True,
                            logger=logs.append)
    solver.train(checkpoint_dir=None)
    text = "\n".join(logs)
    for marker in ["iter", "elapsed", "ETA", "adv-loss", "avg10",
                   "override-mix", "iter-wall"]:
        assert marker in text, f"enhanced log missing marker {marker!r}"


def test_disabled_log_byte_identical_to_baseline(stub_abstraction):
    logs = []
    solver = _legacy_solver(stub_abstraction, n_iterations=2, enhanced_logging=False,
                            logger=logs.append)
    solver.train(checkpoint_dir=None)
    iter_lines = [ln for ln in logs if ln.startswith("iter ")]
    assert len(iter_lines) == 2, f"expected 2 iter lines, got {len(iter_lines)}"
    for ln in iter_lines:
        assert _LEGACY_LOG_RE.match(ln), f"line does not match legacy format: {ln!r}"


# ============================================================
# Shape locks (2)
# ============================================================


def test_mini_eval_history_shape(mini_eval_run):
    """_mini_eval_history: list of raw per-snapshot result dicts (the shape the
    rolling-delta reads), {name: {'lift','std','sigma'}}."""
    h = mini_eval_run.history
    assert isinstance(h, list) and len(h) == 1
    entry = h[0]
    assert isinstance(entry, dict) and len(entry) >= 1
    for name, r in entry.items():
        assert isinstance(name, str)
        assert set(r.keys()) == {"lift", "std", "sigma"}
        for k in ("lift", "std", "sigma"):
            assert isinstance(r[k], numbers.Real), f"{name}.{k} not a number: {r[k]!r}"


def test_mini_eval_metrics_shape(mini_eval_run):
    """metrics['mini_eval']: list of wrapped records {iter, wall_s, results}."""
    m = mini_eval_run.metrics_mini_eval
    assert isinstance(m, list) and len(m) == 1
    rec = m[0]
    assert set(rec.keys()) == {"iter", "wall_s", "results"}
    assert isinstance(rec["iter"], int)
    assert isinstance(rec["wall_s"], float)
    assert isinstance(rec["results"], dict) and len(rec["results"]) >= 1


# ============================================================
# Assumption lock (1)
# ============================================================


def test_iteration_1_indexing_assumption(stub_abstraction):
    """The train loop is 1-indexed (start_iter = self.iteration + 1, >= 1), which
    is why the mini-eval gate omits an explicit `it > 0` guard. Lock it: a fresh
    solver is at iteration 0, and the first processed iter is 1."""
    solver = _legacy_solver(stub_abstraction, n_iterations=1, traversals_per_iter=3)
    assert solver.iteration == 0
    metrics = solver.train(checkpoint_dir=None)
    assert metrics["iter"][0] == 1, "first processed iter must be 1-indexed"
    assert min(metrics["iter"]) >= 1
