"""Tests for src/nlhe/solver6.py (Phase 4e.3c).

Uses a stub abstraction to keep tests fast and decoupled from the
production EMD pickle. Full integration with the production abstraction
is exercised via tests/test_cfr6.py (already covered by 4e.3b).

Coverage:
  - Payout-mode resolution (double_up vs standard vs invalid)
  - Construction wiring (game/abstraction/config; rejects wrong-N games)
  - One-iteration smoke (no exceptions, metrics dict shape)
  - Traverser cycling: 6 iterations -> each seat traversed once
  - Per-iteration training only affects the traverser's net (the others
    keep their pre-iteration parameters byte-identical)
  - Loss is NaN until any buffer has >= batch_size samples
  - Buffer growth across iterations
  - Bit-identical checkpoint roundtrip: save -> load fresh -> compare all
    network parameters and all buffer contents
  - Resume continues iteration counter correctly
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np
import pyspiel
import pytest
import torch

from src.nlhe.game_strings import six_max_sng
from src.nlhe.networks6 import NUM_SEATS_6MAX
from src.nlhe.solver6 import (
    DeepCFR6MaxSolver,
    TrainConfig6Max,
    _resolve_payouts,
)


# ===== Stub abstraction =====


@dataclass
class _StubAbstraction:
    """Hash-based deterministic bucket assignment; not a real abstraction."""

    def bucket_of(self, hero, board, runouts=200, rng=None):
        key = (tuple(sorted(hero)), tuple(sorted(board)))
        digest = hashlib.sha256(repr(key).encode()).hexdigest()
        return int(digest[:8], 16) % 200


# ===== Fixtures =====


@pytest.fixture
def tiny_config():
    """Small but real config so tests finish in seconds."""
    return TrainConfig6Max(
        starting_stack=1500,
        big_blind=100,
        small_blind=50,
        payout_mode="double_up",
        hidden_dim=[16, 16],
        n_iterations=2,
        traversals_per_iter=5,
        train_steps_per_iter=3,
        batch_size=4,
        learning_rate=1e-3,
        buffer_capacity=500,
        bucket_runouts=10,
        max_traversal_depth=200,
        seed=2026,
    )


@pytest.fixture
def fresh_game():
    return pyspiel.load_game(six_max_sng(1500))


@pytest.fixture
def stub_abstraction():
    return _StubAbstraction()


def _make_solver(game, abstraction, config, logger=None):
    return DeepCFR6MaxSolver(
        game=game, abstraction=abstraction, config=config, logger=logger or (lambda s: None),
    )


# ===== Payout-mode resolution =====


def test_resolve_payouts_double_up():
    p = _resolve_payouts("double_up", buy_in=1.0, first_share=0.65)
    assert p == [2.0, 2.0, 2.0]


def test_resolve_payouts_standard_default():
    p = _resolve_payouts("standard", buy_in=1.0, first_share=0.65)
    assert len(p) == 2
    # 6.0 prize pool * 0.65 = 3.9 first place; 6.0 * 0.35 = 2.1 second.
    assert abs(p[0] - 3.9) < 1e-9
    assert abs(p[1] - 2.1) < 1e-9


def test_resolve_payouts_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown payout_mode"):
        _resolve_payouts("triple_up", buy_in=1.0, first_share=0.65)


# ===== Construction =====


def test_solver_constructs(tiny_config, fresh_game, stub_abstraction):
    solver = _make_solver(fresh_game, stub_abstraction, tiny_config)
    assert solver.iteration == 0
    assert len(solver.policy_nets.nets) == NUM_SEATS_6MAX
    assert solver.payouts == [2.0, 2.0, 2.0]
    # Uniform starting stacks for now (documented simplification).
    assert solver.starting_stacks == [1500] * NUM_SEATS_6MAX


def test_solver_rejects_wrong_player_count(tiny_config, stub_abstraction):
    """HUNL game should be refused."""
    game_hunl = pyspiel.load_game(
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,"
        "blind=50 100,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,"
        "numHoleCards=2,numBoardCards=0 3 1 1,stack=1500 1500,"
        "bettingAbstraction=fullgame)"
    )
    with pytest.raises(ValueError, match="6-player game"):
        _make_solver(game_hunl, stub_abstraction, tiny_config)


def test_standard_payout_mode_resolves(fresh_game, stub_abstraction):
    cfg = TrainConfig6Max(
        starting_stack=1500, hidden_dim=[8, 8], n_iterations=1,
        traversals_per_iter=1, train_steps_per_iter=1, batch_size=2,
        buffer_capacity=100, payout_mode="standard", first_share=0.65,
    )
    solver = _make_solver(fresh_game, stub_abstraction, cfg)
    assert len(solver.payouts) == 2
    assert abs(solver.payouts[0] - 3.9) < 1e-9


# ===== Training: smoke =====


def test_train_one_iteration_smoke(tiny_config, fresh_game, stub_abstraction):
    tiny_config.n_iterations = 1
    solver = _make_solver(fresh_game, stub_abstraction, tiny_config)
    metrics = solver.train()
    assert metrics["iter"] == [1]
    assert metrics["traverser"] == [0]  # (it-1) % 6 = 0 for iter 1
    assert len(metrics["adv_loss"]) == 1
    assert len(metrics["time"]) == 1
    for s in range(NUM_SEATS_6MAX):
        assert f"buf_{s}" in metrics
        assert len(metrics[f"buf_{s}"]) == 1


def test_train_two_iterations(tiny_config, fresh_game, stub_abstraction):
    solver = _make_solver(fresh_game, stub_abstraction, tiny_config)
    metrics = solver.train()
    assert metrics["iter"] == [1, 2]
    assert metrics["traverser"] == [0, 1]
    assert solver.iteration == 2


# ===== Traverser cycling =====


def test_traverser_cycles_through_six_seats(tiny_config, fresh_game, stub_abstraction):
    """6 iterations cycle the traverser through each seat exactly once."""
    tiny_config.n_iterations = 6
    solver = _make_solver(fresh_game, stub_abstraction, tiny_config)
    metrics = solver.train()
    assert metrics["traverser"] == [0, 1, 2, 3, 4, 5]
    # Cycle should repeat in a 7th iteration.
    tiny_config2 = TrainConfig6Max(
        **{**tiny_config.__dict__, "n_iterations": 7, "seed": 99}
    )
    solver2 = _make_solver(fresh_game, stub_abstraction, tiny_config2)
    m2 = solver2.train()
    assert m2["traverser"][-1] == 0  # 7th iter = (7-1) % 6 = 0


def test_each_seat_buffer_grows_in_its_iteration(tiny_config, fresh_game, stub_abstraction):
    """After iter where seat=k traverses, buffer_for(k) is non-empty AND
    its size strictly increased compared to before that iter."""
    tiny_config.n_iterations = 6
    solver = _make_solver(fresh_game, stub_abstraction, tiny_config)
    metrics = solver.train()
    # buf_k[k] (size at the END of iter k+1, where seat k traversed) > buf_k[k-1]
    # (size before that iter). For iter index 0 (it=1, traverser=0), check
    # buf_0[0] > 0.
    for seat in range(NUM_SEATS_6MAX):
        # 0-indexed iteration index where this seat traversed.
        idx = seat
        before = metrics[f"buf_{seat}"][idx - 1] if idx > 0 else 0
        after = metrics[f"buf_{seat}"][idx]
        assert after > before, (
            f"seat {seat} buffer didn't grow during its iter: "
            f"before={before}, after={after}"
        )


# ===== Loss is NaN before buffer reaches batch_size =====


def test_loss_is_nan_with_tiny_buffer_threshold(fresh_game, stub_abstraction):
    """If batch_size is larger than what a single iteration's traversals
    can fill, the first iteration's training returns NaN (buffer < batch)."""
    cfg = TrainConfig6Max(
        starting_stack=1500, hidden_dim=[8, 8], n_iterations=1,
        traversals_per_iter=1,  # one traversal -> ~30 decisions for the traverser
        train_steps_per_iter=1,
        batch_size=1000,  # larger than any plausible single-traversal buffer
        buffer_capacity=2000,
        bucket_runouts=10,
        seed=2026,
    )
    solver = _make_solver(fresh_game, stub_abstraction, cfg)
    metrics = solver.train()
    import math
    assert math.isnan(metrics["adv_loss"][0]), (
        f"expected NaN loss with buffer < batch_size; got {metrics['adv_loss'][0]}"
    )


# ===== Only the traverser's net trains per iteration =====


def _snapshot_seat_params(solver, seat):
    return [p.detach().clone() for p in solver.policy_nets.net_for(seat).parameters()]


def test_only_traverser_net_updates_per_iter(tiny_config, fresh_game, stub_abstraction):
    """Run 1 iteration with traverser=0; verify seats 1..5 nets are byte-identical
    before and after, and only seat 0 changes (assuming buffer fills enough to train)."""
    # Bump traversals + steps so seat 0 actually trains (buffer >= batch_size).
    tiny_config.traversals_per_iter = 30
    tiny_config.train_steps_per_iter = 5
    tiny_config.batch_size = 8
    tiny_config.n_iterations = 1
    solver = _make_solver(fresh_game, stub_abstraction, tiny_config)

    snapshots_before = {s: _snapshot_seat_params(solver, s) for s in range(NUM_SEATS_6MAX)}
    metrics = solver.train()

    # Seat 0 trained (loss should be a real number, not NaN).
    import math
    assert not math.isnan(metrics["adv_loss"][0]), (
        f"seat 0 should have trained — got NaN loss (buffer didn't fill)"
    )
    # Verify seat 0 params changed; seats 1..5 unchanged.
    after_0 = _snapshot_seat_params(solver, 0)
    assert not all(torch.allclose(a, b) for a, b in zip(snapshots_before[0], after_0)), (
        "seat 0 params didn't change — training didn't actually update the net"
    )
    for s in range(1, NUM_SEATS_6MAX):
        after_s = _snapshot_seat_params(solver, s)
        for a, b in zip(snapshots_before[s], after_s):
            assert torch.allclose(a, b), (
                f"seat {s} params changed even though it didn't traverse this iter"
            )


# ===== Checkpoint roundtrip =====


def test_checkpoint_roundtrip_preserves_params(tiny_config, fresh_game, stub_abstraction, tmp_path):
    solver1 = _make_solver(fresh_game, stub_abstraction, tiny_config)
    solver1.train()  # 2 iterations
    ckpt = tmp_path / "test_ckpt.pt"
    solver1.save_checkpoint(ckpt)

    # Fresh solver, same config — it'll have different initial random params.
    cfg2 = TrainConfig6Max(**{**tiny_config.__dict__, "seed": tiny_config.seed + 1000})
    solver2 = _make_solver(fresh_game, stub_abstraction, cfg2)
    # Different seed -> different params before load.
    p1_before = list(solver2.policy_nets.net_for(0).parameters())[0]
    p1_orig = list(solver1.policy_nets.net_for(0).parameters())[0]
    assert not torch.allclose(p1_before, p1_orig), (
        "test premise broken: solver1 and solver2 already have same params"
    )

    solver2.load_checkpoint(ckpt)

    # Bit-identical param check across all 6 nets.
    for seat in range(NUM_SEATS_6MAX):
        p1 = list(solver1.policy_nets.net_for(seat).parameters())
        p2 = list(solver2.policy_nets.net_for(seat).parameters())
        for a, b in zip(p1, p2):
            max_diff = (a - b).abs().max().item()
            assert max_diff == 0.0, (
                f"seat {seat} param diff = {max_diff} (not bit-identical)"
            )


def test_checkpoint_roundtrip_preserves_buffers(tiny_config, fresh_game, stub_abstraction, tmp_path):
    solver1 = _make_solver(fresh_game, stub_abstraction, tiny_config)
    solver1.train()
    ckpt = tmp_path / "ckpt.pt"
    solver1.save_checkpoint(ckpt)

    cfg2 = TrainConfig6Max(**{**tiny_config.__dict__, "seed": tiny_config.seed + 1000})
    solver2 = _make_solver(fresh_game, stub_abstraction, cfg2)
    solver2.load_checkpoint(ckpt)

    for seat in range(NUM_SEATS_6MAX):
        b1 = solver1.policy_nets.buffer_for(seat)
        b2 = solver2.policy_nets.buffer_for(seat)
        assert len(b1) == len(b2), f"seat {seat}: buf sizes {len(b1)} vs {len(b2)}"
        assert b1.n_seen == b2.n_seen, f"seat {seat}: n_seen mismatch"
        for i, (f1, f2) in enumerate(zip(b1.features, b2.features)):
            assert np.array_equal(f1, f2), f"seat {seat} sample {i}: feature mismatch"
        for i, (t1, t2) in enumerate(zip(b1.targets, b2.targets)):
            assert np.array_equal(t1, t2), f"seat {seat} sample {i}: target mismatch"
        for i, (m1, m2) in enumerate(zip(b1.legal_masks, b2.legal_masks)):
            assert np.array_equal(m1, m2), f"seat {seat} sample {i}: mask mismatch"
        assert b1.iters == b2.iters, f"seat {seat}: iters list mismatch"


def test_checkpoint_resume_continues_iteration(tiny_config, fresh_game, stub_abstraction, tmp_path):
    """Save at iter 2, resume into a fresh solver, train 2 more, end at iter 4."""
    solver1 = _make_solver(fresh_game, stub_abstraction, tiny_config)
    solver1.train()
    assert solver1.iteration == 2
    ckpt = tmp_path / "ckpt.pt"
    solver1.save_checkpoint(ckpt)

    cfg2 = TrainConfig6Max(**{**tiny_config.__dict__, "n_iterations": 4})
    solver2 = _make_solver(fresh_game, stub_abstraction, cfg2)
    solver2.load_checkpoint(ckpt)
    assert solver2.iteration == 2
    m = solver2.train()
    # train() begins at start_iter = self.iteration + 1 = 3, runs through n_iterations = 4.
    assert m["iter"] == [3, 4]
    assert solver2.iteration == 4


# ===== Checkpoint dir creation =====


def test_smoke_yaml_parses_into_train_config():
    """The configs/six_max_smoke.yaml must round-trip cleanly into
    TrainConfig6Max — catches YAML keys that don't match config fields."""
    import yaml
    from pathlib import Path
    yaml_path = Path(__file__).parent.parent / "configs" / "six_max_smoke.yaml"
    if not yaml_path.exists():
        pytest.skip(f"smoke config not found at {yaml_path}")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    # Strip script-only keys.
    cfg.pop("tag", None)
    cfg.pop("abstraction_path", None)
    cfg.pop("checkpoint_every", None)
    tc = TrainConfig6Max(**cfg)  # raises TypeError on unknown key
    # Spot-check a few values we know.
    assert tc.starting_stack == 1500
    assert tc.payout_mode == "double_up"
    assert tc.n_iterations >= 1
    assert tc.seed == 2026


def test_script_train_6max_imports_and_helpers_work():
    """The training script's imports and helper functions must work without
    actually invoking training. Validates the script wiring end-to-end up
    to the abstraction-load step."""
    from scripts.train_6max import (
        build_six_max_game,
        load_yaml_config,
    )
    cfg = TrainConfig6Max(starting_stack=1500, big_blind=100, small_blind=50)
    game = build_six_max_game(cfg)
    assert game.num_players() == 6


def test_checkpoint_dir_autocreate_during_train(tiny_config, fresh_game, stub_abstraction, tmp_path):
    """Passing checkpoint_dir creates it and writes a ckpt file at completion."""
    tiny_config.n_iterations = 2
    solver = _make_solver(fresh_game, stub_abstraction, tiny_config)
    ckpt_dir = tmp_path / "checkpoints_subdir" / "deeper"
    assert not ckpt_dir.exists()
    solver.train(checkpoint_dir=ckpt_dir, checkpoint_every=1)
    assert ckpt_dir.exists()
    ckpts = list(ckpt_dir.glob("ckpt_iter_*.pt"))
    assert len(ckpts) == 2, f"expected 2 checkpoints, found {len(ckpts)}: {ckpts}"


# ===== Strategy net (v2 schema, Step D) =====


def test_strategy_loss_descends_over_iterations(tiny_config, fresh_game, stub_abstraction):
    """Over 50 iterations the shared strategy net's KL loss trends down."""
    cfg = TrainConfig6Max(**{**tiny_config.__dict__, "n_iterations": 50})
    solver = _make_solver(fresh_game, stub_abstraction, cfg)
    metrics = solver.train()
    finite = [x for x in metrics["strat_loss"] if not np.isnan(x)]
    assert len(finite) >= 2, "strategy net never trained (buffer never reached batch_size)"
    assert finite[-1] < finite[0], (
        f"strategy loss did not descend: first={finite[0]:.4f} last={finite[-1]:.4f}"
    )


def test_strategy_output_sums_to_one_on_legal_mask(tiny_config, fresh_game, stub_abstraction):
    """The strategy net's masked softmax is a valid distribution over the legal
    actions (sums to 1, zero on illegal) — holds for any weights, no training."""
    from src.nlhe.networks6 import N_DISCRETE_ACTIONS
    solver = _make_solver(fresh_game, stub_abstraction, tiny_config)
    legal = (0, 1, 6)  # FOLD, CALL, ALLIN
    feat = torch.zeros(1, 236)
    mask = torch.zeros(1, N_DISCRETE_ACTIONS)
    for i in legal:
        mask[0, i] = 1.0
    with torch.no_grad():
        logits = solver.policy_nets.strat_net(feat.to(solver.device))
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_l = torch.exp(logits) * mask.to(solver.device)
    probs = (exp_l / exp_l.sum(dim=1, keepdim=True).clamp(min=1e-8)).cpu().numpy()[0]
    assert abs(probs[list(legal)].sum() - 1.0) < 1e-5, f"legal sum {probs[list(legal)].sum()}"
    illegal = [i for i in range(N_DISCRETE_ACTIONS) if i not in legal]
    assert all(probs[i] == 0.0 for i in illegal), "illegal action has nonzero prob"


def test_strategy_output_differs_from_regret_matched(tiny_config, fresh_game, stub_abstraction):
    """The strategy net learns a policy distinct from the current regret-matched
    strategy off the advantage net — otherwise there'd be no point training it."""
    from src.nlhe.solver import _strategy_from_advantages
    cfg = TrainConfig6Max(**{**tiny_config.__dict__, "n_iterations": 50})
    solver = _make_solver(fresh_game, stub_abstraction, cfg)
    solver.train()
    buf = solver.policy_nets.strat_buffer
    assert len(buf) >= 8
    feats, targets, masks, iters = buf.sample_batch(min(32, len(buf)))
    # Strategy-net masked-softmax policy on these infosets.
    with torch.no_grad():
        logits = solver.policy_nets.strat_net(feats.to(solver.device))
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_l = torch.exp(logits) * masks.to(solver.device)
    strat_probs = (exp_l / exp_l.sum(dim=1, keepdim=True).clamp(min=1e-8)).cpu().numpy()
    # Regret-matched CURRENT strategy from the seat-0 advantage net, same infosets.
    feats_np = feats.cpu().numpy()
    masks_np = masks.cpu().numpy()
    rm = np.array([
        _strategy_from_advantages(
            solver.policy_nets.predict_advantages(seat=0, features=feats_np[i]),
            masks_np[i],
        )
        for i in range(feats_np.shape[0])
    ])
    l1 = np.abs(strat_probs - rm).sum(axis=1)
    assert l1.max() > 0.05, (
        f"strategy net output indistinguishable from regret-matched "
        f"(max L1 {l1.max():.4f}) — strategy net may not be learning a distinct policy"
    )


def test_checkpoint_v2_save_load_roundtrip(tiny_config, fresh_game, stub_abstraction, tmp_path):
    """Non-slim v2 checkpoint roundtrips the strategy net params + buffer."""
    solver1 = _make_solver(fresh_game, stub_abstraction, tiny_config)
    solver1.train()
    assert len(solver1.policy_nets.strat_buffer) > 0
    ckpt = tmp_path / "v2.pt"
    solver1.save_checkpoint(ckpt)  # non-slim

    cfg2 = TrainConfig6Max(**{**tiny_config.__dict__, "seed": tiny_config.seed + 1000})
    solver2 = _make_solver(fresh_game, stub_abstraction, cfg2)
    solver2.load_checkpoint(ckpt)

    # Strategy net params bit-identical.
    for a, b in zip(solver1.policy_nets.strat_net.parameters(),
                    solver2.policy_nets.strat_net.parameters()):
        assert (a - b).abs().max().item() == 0.0, "strat_net params not bit-identical"
    # Strategy buffer contents match.
    sb1, sb2 = solver1.policy_nets.strat_buffer, solver2.policy_nets.strat_buffer
    assert len(sb1) == len(sb2) > 0
    assert sb1.n_seen == sb2.n_seen
    for f1, f2 in zip(sb1.features, sb2.features):
        assert np.array_equal(f1, f2)
    for t1, t2 in zip(sb1.targets, sb2.targets):
        assert np.array_equal(t1, t2)
    assert sb1.iters == sb2.iters


def test_checkpoint_slim_drops_strat_buffer(tiny_config, fresh_game, stub_abstraction, tmp_path):
    """Slim checkpoint carries no buffer contents (adv or strat); load is clean."""
    solver = _make_solver(fresh_game, stub_abstraction, tiny_config)
    solver.train()
    assert len(solver.policy_nets.strat_buffer) > 0
    ckpt = tmp_path / "slim.pt"
    solver.save_checkpoint(ckpt, slim=True)

    d = torch.load(ckpt, weights_only=False)
    assert "strat_buffer" not in d
    assert "buffers" not in d

    cfg2 = TrainConfig6Max(**{**tiny_config.__dict__, "seed": tiny_config.seed + 1000})
    solver2 = _make_solver(fresh_game, stub_abstraction, cfg2)
    solver2.load_checkpoint(ckpt)  # must not raise
    assert len(solver2.policy_nets.strat_buffer) == 0


def test_v1_solver_load_succeeds(tiny_config, fresh_game, stub_abstraction, tmp_path):
    """REVISED (Step E): a v1-shape checkpoint (policy_nets payload lacking
    schema_version, and no strategy buffer — a genuine pre-strategy-net save)
    LOADS at the solver level without raising. Advantage nets are restored, the
    container is marked v1, and the strategy net is left fresh. The refusal moved
    from the load boundary to deployment (inference_policy)."""
    solver = _make_solver(fresh_game, stub_abstraction, tiny_config)
    solver.train()
    ckpt = tmp_path / "downgraded_v1.pt"
    solver.save_checkpoint(ckpt)
    # Downgrade to a genuine v1 shape: strip schema_version + the strategy buffer.
    d = torch.load(ckpt, weights_only=False)
    del d["policy_nets"]["schema_version"]
    d.pop("strat_buffer", None)
    torch.save(d, ckpt)

    cfg2 = TrainConfig6Max(**{**tiny_config.__dict__, "seed": tiny_config.seed + 1000})
    solver2 = _make_solver(fresh_game, stub_abstraction, cfg2)
    solver2.load_checkpoint(ckpt)  # must NOT raise
    assert solver2.policy_nets.loaded_schema_version == "v1"
    # Advantage nets restored bit-identically from the v1 checkpoint.
    for seat in range(NUM_SEATS_6MAX):
        for a, b in zip(solver.policy_nets.net_for(seat).parameters(),
                        solver2.policy_nets.net_for(seat).parameters()):
            assert (a - b).abs().max().item() == 0.0, f"seat {seat} adv params not restored"
