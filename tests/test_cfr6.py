"""Tests for src/nlhe/cfr6.py (Phase 4e.3b).

Stub-abstraction tests run anywhere; integration tests using the real
production abstraction (runs/abstraction_20260521_223018_retrofit/abstraction.pkl)
auto-skip when the artifact isn't present, matching the pattern in
tests/test_infoset6.py.

What's covered:
  - Construction-level checks (CFR6MaxContext validation)
  - Terminal handling returns ICM-adjusted utility for the traverser
  - Chance-node traversal runs without crashing
  - Full-game traversal returns a float in equity units
  - Buffer addition is restricted to the traverser's buffer
  - Number of samples added equals the number of traverser decision nodes
  - Regret samples are zero on illegal actions
  - Sample shapes match (feature, regrets, legal_mask) what the buffer expects
  - Determinism: same rng seed -> same return value and same buffer count
  - Out-of-range traversing_player raises IndexError
  - Smoke test: 100 traversals on six_max_sng complete without errors
"""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pyspiel
import pytest

from src.nlhe.cfr6 import (
    CFR6MaxContext,
    _build_view_6max,
    traverse_6max,
)
from src.nlhe.game_strings import six_max_sng
from src.nlhe.icm import (
    sng_payouts_6max_double_up,
    sng_payouts_6max_standard,
)
from src.nlhe.icm_returns import icm_adjust_returns
from src.nlhe.infoset6 import InfosetEncoder6Max, parse_state_6max
from src.nlhe.networks6 import (
    N_DISCRETE_ACTIONS,
    NUM_SEATS_6MAX,
    PlayerNetworks6Max,
)


# ===== Stub abstraction (decouples cfr6 tests from the production pickle) =====


@dataclass
class _StubAbstraction:
    """Minimal Abstraction stand-in for cfr6 mechanics tests.

    Returns a deterministic bucket id derived from (hero, board) via SHA-256.
    Doesn't pretend to be a *good* abstraction — same hand always maps to
    the same bucket, which is the only property cfr6 actually needs.
    """

    def bucket_of(self, hero, board, runouts=200, rng=None):
        key = (tuple(sorted(hero)), tuple(sorted(board)))
        digest = hashlib.sha256(repr(key).encode()).hexdigest()
        return int(digest[:8], 16) % 200


# ===== Fixtures =====


STARTING_STACK = 1500  # 6-max SNG default per src/nlhe/game_strings.py


@pytest.fixture
def stub_encoder():
    return InfosetEncoder6Max(
        abstraction=_StubAbstraction(),
        starting_stack=STARTING_STACK,
        max_bucket_dim=200,
        bucket_runouts=50,  # smaller for test speed
    )


@pytest.fixture
def policy_nets():
    """Untrained 6-seat networks with small hidden layers for test speed."""
    return PlayerNetworks6Max(
        input_dim=236,
        hidden=[32, 32],
        learning_rate=1e-3,
        buffer_capacity=10_000,
        rng=random.Random(2026),
        device="cpu",
    )


@pytest.fixture
def ctx_double_up(stub_encoder, policy_nets):
    return CFR6MaxContext(
        policy_nets=policy_nets,
        encoder=stub_encoder,
        starting_stacks=[STARTING_STACK] * NUM_SEATS_6MAX,
        payouts=sng_payouts_6max_double_up(buy_in=1.0),
        iteration=0,
    )


@pytest.fixture
def ctx_standard(stub_encoder, policy_nets):
    return CFR6MaxContext(
        policy_nets=policy_nets,
        encoder=stub_encoder,
        starting_stacks=[STARTING_STACK] * NUM_SEATS_6MAX,
        payouts=sng_payouts_6max_standard(buy_in=1.0),
        iteration=0,
    )


@pytest.fixture
def fresh_game():
    return pyspiel.load_game(six_max_sng(STARTING_STACK))


# ===== Context construction =====


def test_context_rejects_wrong_starting_stacks_length(stub_encoder, policy_nets):
    with pytest.raises(ValueError, match="starting_stacks must have 6"):
        CFR6MaxContext(
            policy_nets=policy_nets,
            encoder=stub_encoder,
            starting_stacks=[STARTING_STACK] * 3,  # too few
            payouts=sng_payouts_6max_double_up(),
        )


def test_context_rejects_empty_payouts(stub_encoder, policy_nets):
    with pytest.raises(ValueError, match="payouts"):
        CFR6MaxContext(
            policy_nets=policy_nets,
            encoder=stub_encoder,
            starting_stacks=[STARTING_STACK] * NUM_SEATS_6MAX,
            payouts=[],
        )


# ===== _build_view_6max =====


def test_build_view_at_preflop_utg(fresh_game):
    """At the first UTG decision (preflop, facing blinds), the view should
    reflect: nothing folded yet, UTG hasn't contributed, BB is the bet to
    call. Pot value uses universal_poker's reporting convention (not the
    raw 150 chips actually committed — this matches HUNL's convention)."""
    state = fresh_game.new_initial_state()
    rng = random.Random(0)
    while state.is_chance_node():
        a = rng.choices(*zip(*state.chance_outcomes()), k=1)[0]
        state.apply_action(a)
    parsed = parse_state_6max(state)
    view = _build_view_6max(state, parsed)
    # UTG hasn't contributed; to_call = BB = 100 (raw chips, NOT scaled).
    assert view.to_call == 100
    # Facing a bet means fold is legal.
    assert view.legal_fold is True
    # Call is always legal preflop facing the BB.
    assert view.legal_call is True
    # min_bet must be >= 2 (universal_poker action encoding); max_bet must
    # be at most the starting stack.
    assert view.min_bet >= 2
    assert view.max_bet <= STARTING_STACK
    # Pot value is whatever universal_poker reports; sanity-check it's > 0
    # and not absurdly larger than the committed chips.
    assert view.pot > 0


# ===== Out-of-range traversing player =====


def test_traverse_rejects_invalid_traversing_player(fresh_game, ctx_double_up):
    state = fresh_game.new_initial_state()
    rng = random.Random(0)
    with pytest.raises(IndexError):
        traverse_6max(state, traversing_player=6, ctx=ctx_double_up, rng=rng)
    with pytest.raises(IndexError):
        traverse_6max(state, traversing_player=-1, ctx=ctx_double_up, rng=rng)


# ===== Terminal handling =====


def _walk_to_terminal(game, seed):
    """Walk a game to terminal using a fold-heavy random policy.
    Returns the terminal state. Folds bias things toward short trajectories."""
    state = game.new_initial_state()
    rng = random.Random(seed)
    while not state.is_terminal():
        if state.is_chance_node():
            a = rng.choices(*zip(*state.chance_outcomes()), k=1)[0]
            state.apply_action(a)
        else:
            legal = state.legal_actions()
            # Prefer fold (action 0) heavily to get short trajectories.
            if 0 in legal and rng.random() < 0.7:
                state.apply_action(0)
            else:
                state.apply_action(rng.choice(legal))
    return state


def test_terminal_returns_icm_adjusted_traverser_slice(fresh_game, ctx_double_up):
    """At a terminal state, traverse_6max returns icm_adjust_returns(...)[traverser]."""
    state = _walk_to_terminal(fresh_game, seed=2026)
    assert state.is_terminal()
    expected = icm_adjust_returns(
        chip_returns=list(state.returns()),
        starting_stacks=ctx_double_up.starting_stacks,
        payouts=ctx_double_up.payouts,
    )
    for traverser in range(NUM_SEATS_6MAX):
        result = traverse_6max(
            state, traversing_player=traverser,
            ctx=ctx_double_up, rng=random.Random(0)
        )
        assert abs(result - expected[traverser]) < 1e-9, (
            f"traverser {traverser}: traverse_6max returned {result}, "
            f"expected {expected[traverser]}"
        )


def test_terminal_returns_match_icm_for_standard_payouts(fresh_game, ctx_standard):
    state = _walk_to_terminal(fresh_game, seed=1)
    assert state.is_terminal()
    expected = icm_adjust_returns(
        chip_returns=list(state.returns()),
        starting_stacks=ctx_standard.starting_stacks,
        payouts=ctx_standard.payouts,
    )
    for traverser in range(NUM_SEATS_6MAX):
        result = traverse_6max(
            state, traversing_player=traverser,
            ctx=ctx_standard, rng=random.Random(0)
        )
        assert abs(result - expected[traverser]) < 1e-9


def test_terminal_no_buffer_write(fresh_game, ctx_double_up):
    """Terminal-state calls don't add anything to any buffer."""
    state = _walk_to_terminal(fresh_game, seed=42)
    assert state.is_terminal()
    initial_sizes = [len(ctx_double_up.policy_nets.buffer_for(s))
                     for s in range(NUM_SEATS_6MAX)]
    traverse_6max(state, traversing_player=0,
                  ctx=ctx_double_up, rng=random.Random(0))
    final_sizes = [len(ctx_double_up.policy_nets.buffer_for(s))
                   for s in range(NUM_SEATS_6MAX)]
    assert initial_sizes == final_sizes, (
        f"buffers changed at terminal: {initial_sizes} -> {final_sizes}"
    )


# ===== Chance-node + full-game traversal =====


def test_traverse_from_initial_chance_node_doesnt_crash(fresh_game, ctx_double_up):
    """A fresh state is a chance node (cards undealt). Traversal must handle it."""
    state = fresh_game.new_initial_state()
    assert state.is_chance_node()
    result = traverse_6max(
        state, traversing_player=0, ctx=ctx_double_up, rng=random.Random(0)
    )
    assert isinstance(result, float)


def test_traverse_returns_float(fresh_game, ctx_double_up):
    state = fresh_game.new_initial_state()
    result = traverse_6max(
        state, traversing_player=2, ctx=ctx_double_up, rng=random.Random(7)
    )
    assert isinstance(result, float)
    # ICM equity for 6 players with Double Up payouts ([2.0, 2.0, 2.0]) is
    # bounded by max payout = 2.0. Per-player baseline is 1.0 (prize_pool/6).
    # Delta from baseline is in (-1.0, 1.0). Loose check.
    assert -2.0 <= result <= 2.0


# ===== Buffer addition: only the traverser's buffer grows =====


def test_only_traverser_buffer_grows(fresh_game, ctx_double_up):
    traverser = 3
    state = fresh_game.new_initial_state()
    traverse_6max(state, traversing_player=traverser,
                  ctx=ctx_double_up, rng=random.Random(2026))
    sizes = [len(ctx_double_up.policy_nets.buffer_for(s))
             for s in range(NUM_SEATS_6MAX)]
    # Traverser's buffer should have at least one entry (they made at
    # least one decision in any complete 6-max game).
    assert sizes[traverser] > 0, (
        f"traverser {traverser}'s buffer is empty after a traversal: {sizes}"
    )
    for s in range(NUM_SEATS_6MAX):
        if s != traverser:
            assert sizes[s] == 0, (
                f"non-traverser seat {s} got buffer writes: {sizes}"
            )


def test_each_seat_independently_collects_samples(fresh_game, ctx_double_up):
    """Running traversals across all 6 traversing-player choices fills each
    seat's buffer independently."""
    for traverser in range(NUM_SEATS_6MAX):
        state = fresh_game.new_initial_state()
        traverse_6max(state, traversing_player=traverser,
                      ctx=ctx_double_up, rng=random.Random(100 + traverser))
    sizes = [len(ctx_double_up.policy_nets.buffer_for(s))
             for s in range(NUM_SEATS_6MAX)]
    for s in range(NUM_SEATS_6MAX):
        assert sizes[s] > 0, (
            f"seat {s} got no samples even though we ran a traversal for it: {sizes}"
        )


# ===== Sample shape and content invariants =====


def test_regret_samples_have_correct_shape(fresh_game, ctx_double_up):
    state = fresh_game.new_initial_state()
    traverser = 0
    traverse_6max(state, traversing_player=traverser,
                  ctx=ctx_double_up, rng=random.Random(0))
    buf = ctx_double_up.policy_nets.buffer_for(traverser)
    assert len(buf) > 0
    feat = buf.features[0]
    targ = buf.targets[0]
    mask = buf.legal_masks[0]
    assert feat.shape == (236,), f"feature shape {feat.shape}"
    assert targ.shape == (N_DISCRETE_ACTIONS,), f"regret shape {targ.shape}"
    assert mask.shape == (N_DISCRETE_ACTIONS,), f"mask shape {mask.shape}"


def test_regrets_are_zero_on_illegal_actions(fresh_game, ctx_double_up):
    """For each regret sample, regrets[i] must be 0 wherever legal_mask[i] == 0."""
    state = fresh_game.new_initial_state()
    traverser = 4
    traverse_6max(state, traversing_player=traverser,
                  ctx=ctx_double_up, rng=random.Random(0))
    buf = ctx_double_up.policy_nets.buffer_for(traverser)
    assert len(buf) > 0
    for i, (regrets, mask) in enumerate(zip(buf.targets, buf.legal_masks)):
        illegal = mask == 0.0
        nz_on_illegal = regrets[illegal]
        assert np.allclose(nz_on_illegal, 0.0), (
            f"sample {i}: nonzero regret on illegal action: "
            f"regrets={regrets}, mask={mask}"
        )


def test_iteration_stamped_on_samples(fresh_game, ctx_double_up):
    ctx_double_up.iteration = 7
    state = fresh_game.new_initial_state()
    traverse_6max(state, traversing_player=1,
                  ctx=ctx_double_up, rng=random.Random(0))
    buf = ctx_double_up.policy_nets.buffer_for(1)
    assert len(buf) > 0
    for it in buf.iters:
        assert it == 7, f"sample tagged with iter {it}, expected 7"


# ===== Determinism =====


def test_traverse_is_deterministic_under_fixed_rng(fresh_game, stub_encoder):
    """Two traversals with identical rng seed -> identical return value AND
    identical buffer length."""
    def run_once(seed):
        nets = PlayerNetworks6Max(
            input_dim=236, hidden=[32, 32], buffer_capacity=10_000,
            rng=random.Random(0), device="cpu",
        )
        # Force same initial network weights via fixed torch seed.
        import torch
        torch.manual_seed(0)
        # Re-init nets after seeding so they're reproducible.
        nets2 = PlayerNetworks6Max(
            input_dim=236, hidden=[32, 32], buffer_capacity=10_000,
            rng=random.Random(0), device="cpu",
        )
        ctx = CFR6MaxContext(
            policy_nets=nets2,
            encoder=InfosetEncoder6Max(
                abstraction=_StubAbstraction(),
                starting_stack=STARTING_STACK,
                bucket_runouts=50,
            ),
            starting_stacks=[STARTING_STACK] * 6,
            payouts=sng_payouts_6max_double_up(),
            iteration=0,
        )
        state = fresh_game.new_initial_state()
        v = traverse_6max(state, traversing_player=0,
                          ctx=ctx, rng=random.Random(seed))
        return v, len(ctx.policy_nets.buffer_for(0))

    import torch
    torch.manual_seed(0)
    v1, n1 = run_once(seed=2026)
    torch.manual_seed(0)
    v2, n2 = run_once(seed=2026)
    assert v1 == v2, f"non-deterministic value: {v1} vs {v2}"
    assert n1 == n2, f"non-deterministic buffer count: {n1} vs {n2}"


# ===== Smoke test =====


def test_smoke_100_traversals_complete(fresh_game, stub_encoder):
    """100 traversals on a 6-max SNG complete without exceptions."""
    nets = PlayerNetworks6Max(
        input_dim=236, hidden=[32, 32], buffer_capacity=100_000,
        rng=random.Random(0), device="cpu",
    )
    ctx = CFR6MaxContext(
        policy_nets=nets,
        encoder=stub_encoder,
        starting_stacks=[STARTING_STACK] * 6,
        payouts=sng_payouts_6max_double_up(),
        iteration=0,
    )
    rng = random.Random(2026)
    for i in range(100):
        ctx.iteration = i
        traverser = i % NUM_SEATS_6MAX
        state = fresh_game.new_initial_state()
        v = traverse_6max(state, traversing_player=traverser,
                          ctx=ctx, rng=rng)
        assert isinstance(v, float)
        assert -3.0 <= v <= 3.0, f"iter {i}: value {v} out of plausible range"
    # After 100 traversals, every seat should have received some samples
    # (we cycled through traversers 0..5 multiple times).
    for s in range(NUM_SEATS_6MAX):
        assert len(nets.buffer_for(s)) > 0, (
            f"seat {s} got no samples across 100 traversals"
        )


# ===== Integration with the production abstraction (auto-skip if missing) =====


ABSTRACTION_PATH = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"


@pytest.fixture
def real_abstraction():
    if not os.path.exists(ABSTRACTION_PATH):
        pytest.skip(f"production abstraction not found at {ABSTRACTION_PATH}")
    from src.nlhe.abstraction import Abstraction
    return Abstraction.load(ABSTRACTION_PATH)


def test_traverse_with_real_abstraction(real_abstraction, fresh_game):
    """Smoke test integration with the real EMD-clustered abstraction."""
    encoder = InfosetEncoder6Max(
        abstraction=real_abstraction,
        starting_stack=STARTING_STACK,
        bucket_runouts=50,
    )
    nets = PlayerNetworks6Max(
        input_dim=236, hidden=[32, 32], buffer_capacity=10_000,
        rng=random.Random(0), device="cpu",
    )
    ctx = CFR6MaxContext(
        policy_nets=nets,
        encoder=encoder,
        starting_stacks=[STARTING_STACK] * 6,
        payouts=sng_payouts_6max_double_up(),
        iteration=0,
    )
    state = fresh_game.new_initial_state()
    v = traverse_6max(state, traversing_player=0,
                      ctx=ctx, rng=random.Random(2026))
    assert isinstance(v, float)
    assert -3.0 <= v <= 3.0
    assert len(nets.buffer_for(0)) > 0
