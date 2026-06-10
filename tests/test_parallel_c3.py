"""Bit-identity gate for the C3 deltas in the parallel framework (2026-06-10).

The C3 retrain config (237-d eff-BB encoder + empirical hand-start
distribution + tournament mode) must produce IDENTICAL training results
through the G-worker path as through solver6.train()'s sequential path —
the same standard the framework originally passed (see DESIGN.md
"bit-identity gate").

These tests run a tiny-but-real config (full C3 feature set, small nets /
few traversals) and compare metrics, buffer contents, and net parameters
at full precision. The production-scale 10-iter gate record lives in
docs/PARALLEL_TRAINING.md.
"""
from __future__ import annotations

import gzip
import json
import os
import random

import numpy as np
import pyspiel
import pytest
import torch

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import six_max_sng
from src.nlhe.parallel.orchestrator import parallel_train
from src.nlhe.parallel.protocol import WorkerInput
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max

ABSTRACTION_PATH = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCTURE_PATH = "configs/ignition_double_up_6max_turbo.yaml"


@pytest.fixture(scope="module")
def abstraction():
    if not os.path.exists(ABSTRACTION_PATH):
        pytest.skip(f"abstraction not found at {ABSTRACTION_PATH}")
    return Abstraction.load(ABSTRACTION_PATH)


@pytest.fixture()
def dist_artifact(tmp_path):
    rows = [
        {"seed": 1, "hand": 1, "level": 1,
         "stacks": [1500, 1500, 1500, 1500, 1500, 1500], "dealer": 0,
         "n_alive": 6},
        {"seed": 1, "hand": 9, "level": 3,
         "stacks": [700, 2900, 0, 1500, 2400, 1500], "dealer": 4,
         "n_alive": 5},
        {"seed": 2, "hand": 14, "level": 4,
         "stacks": [3100, 4100, 0, 0, 1100, 700], "dealer": 0,
         "n_alive": 4},
        {"seed": 3, "hand": 2, "level": 2,
         "stacks": [2200, 800, 1500, 1500, 1500, 1500], "dealer": 2,
         "n_alive": 6},
    ]
    p = tmp_path / "dist.json.gz"
    p.write_bytes(gzip.compress(json.dumps(
        {"version": "training_dist_test", "n_games": 3,
         "hand_starts": rows}).encode()))
    return str(p)


def _c3_cfg(dist_path, **over):
    base = dict(starting_stack=1500, big_blind=100, small_blind=50,
                hidden_dim=[8], n_iterations=2, traversals_per_iter=6,
                train_steps_per_iter=2, batch_size=2, buffer_capacity=64,
                bucket_runouts=4, seed=3,
                encoder_eff_bb=True,
                ante_convention="real",
                empirical_dist_path=dist_path,
                tournament_structure_path=STRUCTURE_PATH)
    base.update(over)
    return TrainConfig6Max(**base)


def _fresh_solver(abstraction, cfg):
    game = pyspiel.load_game(six_max_sng(starting_stack=cfg.starting_stack))
    return DeepCFR6MaxSolver(game=game, abstraction=abstraction, config=cfg,
                             logger=lambda *_: None)


def _buffer_fingerprint(buf):
    """Exact content tuple for a ReservoirBuffer: every stored array, the
    seen-counter, and the buffer RNG state."""
    feats = [np.asarray(f).tobytes() for f in buf.features]
    targs = [np.asarray(t).tobytes() for t in buf.targets]
    masks = [np.asarray(m).tobytes() for m in buf.legal_masks]
    return (feats, targs, masks, list(buf.iters), buf.n_seen,
            buf.rng.getstate())


def _solver_fingerprint(solver):
    params = {f"{s}/{k}": v.detach().clone()
              for s in range(6)
              for k, v in solver.policy_nets.nets[s].state_dict().items()}
    params.update({f"strat/{k}": v.detach().clone()
                   for k, v in solver.policy_nets.strat_net.state_dict().items()})
    bufs = [_buffer_fingerprint(solver.policy_nets.buffer_for(s))
            for s in range(6)]
    sbuf = _buffer_fingerprint(solver.policy_nets.strat_buffer)
    return params, bufs, sbuf, solver.rng.getstate(), solver._override_counts


def _assert_identical(fp_a, fp_b):
    params_a, bufs_a, sbuf_a, rng_a, oc_a = fp_a
    params_b, bufs_b, sbuf_b, rng_b, oc_b = fp_b
    assert params_a.keys() == params_b.keys()
    for k in params_a:
        assert torch.equal(params_a[k], params_b[k]), f"param mismatch: {k}"
    for s, (ba, bb) in enumerate(zip(bufs_a, bufs_b)):
        assert ba == bb, f"advantage buffer mismatch at seat {s}"
    assert sbuf_a == sbuf_b, "strategy buffer mismatch"
    assert rng_a == rng_b, "solver RNG state mismatch"
    assert oc_a == oc_b, "override counts mismatch"


def test_worker_input_c3_defaults_are_legacy():
    """New fields default OFF: a WorkerInput built by pre-C3 callers is
    bit-compatible with pre-C3 behavior."""
    fields = WorkerInput.__dataclass_fields__
    assert fields["encoder_eff_bb"].default is False
    assert fields["empirical_dist_path"].default is None


@pytest.mark.timeout(300)
def test_c3_parallel_inproc_bit_identical(abstraction, dist_artifact):
    """Sequential solver.train() vs parallel_train(G=3, in-process) on the
    full C3 feature set: metrics, params, buffers, RNG — all identical."""
    cfg = _c3_cfg(dist_artifact)

    seq = _fresh_solver(abstraction, cfg)
    m_seq = seq.train()

    par = _fresh_solver(abstraction, cfg)
    m_par = parallel_train(
        par, game_str=six_max_sng(starting_stack=cfg.starting_stack),
        abstraction_path=ABSTRACTION_PATH, n_workers=3, use_processes=False)

    for key in ("iter", "traverser", "adv_loss", "strat_loss", "strat_buf",
                *[f"buf_{s}" for s in range(6)]):
        assert m_seq[key] == m_par[key], f"metrics[{key}] diverged"
    _assert_identical(_solver_fingerprint(seq), _solver_fingerprint(par))


@pytest.mark.timeout(300)
def test_c3_parallel_mp_fork_bit_identical(abstraction, dist_artifact):
    """Same gate through real fork()ed worker processes (G=2)."""
    cfg = _c3_cfg(dist_artifact)

    seq = _fresh_solver(abstraction, cfg)
    m_seq = seq.train()

    par = _fresh_solver(abstraction, cfg)
    m_par = parallel_train(
        par, game_str=six_max_sng(starting_stack=cfg.starting_stack),
        abstraction_path=ABSTRACTION_PATH, n_workers=2, use_processes=True)

    for key in ("iter", "traverser", "adv_loss", "strat_loss", "strat_buf",
                *[f"buf_{s}" for s in range(6)]):
        assert m_seq[key] == m_par[key], f"metrics[{key}] diverged"
    _assert_identical(_solver_fingerprint(seq), _solver_fingerprint(par))
