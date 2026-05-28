"""Single-traversal equivalence test for src.nlhe.parallel.worker.

Validates that run_traversals(WorkerInput) for one (seed, iteration, t)
produces bit-identical adv_samples and strat_samples to running the same
(seed, iteration, t) sequentially through the real DeepCFR6MaxSolver's
components, with buffer.add() intercepted.

This is the foundation gate for the parallel implementation: if one worker
traversal can't match one sequential traversal, no orchestrator on top of
it can. The unit must be locked before any multiprocessing layer goes in.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyspiel
import pytest
import yaml

from src.nlhe.abstraction import Abstraction
from src.nlhe.cfr6 import CFR6MaxContext, traverse_6max
from src.nlhe.game_strings import PokerGameConfig
from src.nlhe.networks6 import NUM_SEATS_6MAX
from src.nlhe.parallel.protocol import TraversalSample, WorkerInput
from src.nlhe.parallel.worker import _CollectorBuffer, run_traversals
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "baseline_seq.yaml"


def _load_baseline_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_game_str_from_config(tc: TrainConfig6Max) -> str:
    """Mirror scripts/train_6max.py's build_six_max_game string construction."""
    return PokerGameConfig(
        num_players=6,
        starting_stack=tc.starting_stack,
        big_blind=tc.big_blind,
        small_blind=tc.small_blind,
    ).to_universal_poker_string()


def _fork_rng(seed: int, iteration: int, t: int) -> random.Random:
    """Same derivation as solver6.train() and worker._fork_rng."""
    s = (seed * 1_000_003 + iteration * 9_973 + t) & 0x7FFFFFFFFFFFFFFF
    return random.Random(s)


def _assert_samples_equal(seq_list, worker_list, label):
    assert len(seq_list) == len(worker_list), (
        f"{label}: count mismatch sequential={len(seq_list)} "
        f"worker={len(worker_list)}"
    )
    for i, (s, w) in enumerate(zip(seq_list, worker_list)):
        assert isinstance(s, TraversalSample) and isinstance(w, TraversalSample), (
            f"{label}[{i}]: type mismatch"
        )
        assert s.iteration == w.iteration, (
            f"{label}[{i}].iteration: seq={s.iteration} worker={w.iteration}"
        )
        assert np.array_equal(s.feature, w.feature), (
            f"{label}[{i}].feature differs"
        )
        assert np.array_equal(s.target, w.target), (
            f"{label}[{i}].target differs (max abs diff "
            f"{np.abs(s.target - w.target).max() if s.target.shape == w.target.shape else 'shape mismatch'})"
        )
        assert np.array_equal(s.legal_mask, w.legal_mask), (
            f"{label}[{i}].legal_mask differs"
        )


def test_one_traversal_worker_matches_sequential():
    """Run one traversal at (seed=2026, it=1, t=0) two ways; compare samples."""
    cfg = _load_baseline_config()
    abstraction_path = cfg["abstraction_path"]
    cfg_for_tc = {k: v for k, v in cfg.items()
                  if k not in ("tag", "abstraction_path", "checkpoint_every")}
    tc = TrainConfig6Max(**cfg_for_tc)

    game_str = _build_game_str_from_config(tc)
    game = pyspiel.load_game(game_str)
    abst = Abstraction.load(abstraction_path)

    # Build the real solver (this seeds torch and constructs all 6 adv nets +
    # the strat net + the encoder, all deterministic from tc.seed).
    solver = DeepCFR6MaxSolver(game=game, abstraction=abst, config=tc)

    # ---- Sequential path: capture samples via collectors ----
    it = 1
    t = 0
    traverser = (it - 1) % NUM_SEATS_6MAX  # = 0

    seq_adv_collector = _CollectorBuffer()
    seq_strat_collector = _CollectorBuffer()

    # Monkey-patch the policy_nets's add() destinations. predict_advantages
    # stays the real one (we want bit-identical forward passes).
    orig_buffer_for = solver.policy_nets.buffer_for
    orig_strat_buffer = solver.policy_nets.strat_buffer
    solver.policy_nets.buffer_for = lambda seat: seq_adv_collector  # type: ignore[assignment]
    solver.policy_nets.strat_buffer = seq_strat_collector  # type: ignore[assignment]

    try:
        solver.encoder.reset_cache()  # match solver6.train()'s per-iter reset
        ctx_seq = CFR6MaxContext(
            policy_nets=solver.policy_nets,
            encoder=solver.encoder,
            starting_stacks=solver.starting_stacks,
            payouts=solver.payouts,
            iteration=it,
            max_depth=tc.max_traversal_depth,
            num_paid=tc.num_paid,
        )
        rng_seq = _fork_rng(tc.seed, it, t)
        state_seq = game.new_initial_state()
        traverse_6max(
            state_seq,
            traversing_player=traverser,
            ctx=ctx_seq,
            rng=rng_seq,
        )
    finally:
        # Restore solver.policy_nets so the test fixture doesn't leak state.
        solver.policy_nets.buffer_for = orig_buffer_for  # type: ignore[assignment]
        solver.policy_nets.strat_buffer = orig_strat_buffer  # type: ignore[assignment]

    seq_adv = seq_adv_collector.samples
    seq_strat = seq_strat_collector.samples

    # The traversal must have produced SOMETHING for the test to be meaningful.
    assert len(seq_adv) > 0, "sequential traversal produced no adv samples"
    assert len(seq_strat) > 0, "sequential traversal produced no strat samples"

    # ---- Worker path: same inputs, separate process state ----
    adv_state_dicts = [
        {k: v.detach().clone() for k, v in solver.policy_nets.nets[s].state_dict().items()}
        for s in range(NUM_SEATS_6MAX)
    ]

    wi = WorkerInput(
        seed=tc.seed,
        iteration=it,
        traverser=traverser,
        traversal_ids=[t],
        adv_state_dicts=adv_state_dicts,
        input_dim=solver.encoder.feature_dim,
        hidden_dim=list(tc.hidden_dim),
        abstraction_path=abstraction_path,
        encoder_starting_stack=tc.starting_stack,
        encoder_max_bucket_dim=solver.encoder.max_bucket_dim,
        encoder_bucket_runouts=tc.bucket_runouts,
        game_str=game_str,
        starting_stacks=list(solver.starting_stacks),
        payouts=list(solver.payouts),
        max_depth=tc.max_traversal_depth,
        num_paid=tc.num_paid,
        dealer_seat=None,
    )
    outputs = run_traversals(wi)
    assert len(outputs) == 1
    assert outputs[0].traversal_id == t

    # ---- Compare ----
    _assert_samples_equal(seq_adv, outputs[0].adv_samples, "adv_samples")
    _assert_samples_equal(seq_strat, outputs[0].strat_samples, "strat_samples")
