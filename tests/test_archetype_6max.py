"""Tests for src/nlhe/archetype6.py — the 6-max archetype Policy adapter.

Covers the Policy contract, the ArchetypePool sampling interface, the binary
in_position mapping + dealer_seat fallback, the strat-buffer zero-write
invariant during archetype hands (DECISIONS.md:216), and an adapter-level
style sanity check.

Heavier tests that need real artifacts (the retrofit abstraction and the
6-max calibration) skip cleanly when those files are absent.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pytest

from src.nlhe.actions import DiscreteAction
from src.nlhe.archetypes import NAMED_ARCHETYPES, ArchetypeName, EquityCalibration
from src.nlhe.archetype6 import ArchetypePolicy, ArchetypePool, VALID_ARCHETYPE_NAMES
from src.nlhe.infoset6 import POSITIONS_6MAX, position_for_seat_with_dealer

CALIB_6MAX = Path("runs/archetype_design/bucket_equity_analysis_6max.json")
ABSTRACTION_6MAX = Path("runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
_BTN = POSITIONS_6MAX.index("BTN")
_BB = POSITIONS_6MAX.index("BB")


def _by_name(name: ArchetypeName):
    for a in NAMED_ARCHETYPES:
        if a.name == name:
            return a
    raise KeyError(name)


# ===== Fixtures =====

@pytest.fixture(scope="module")
def calibration():
    if not CALIB_6MAX.exists():
        pytest.skip(f"6-max calibration not found at {CALIB_6MAX}")
    return EquityCalibration.load(CALIB_6MAX)


@pytest.fixture(scope="module")
def real_abstraction():
    if not ABSTRACTION_6MAX.exists():
        pytest.skip(f"retrofit abstraction not found at {ABSTRACTION_6MAX}")
    from src.nlhe.abstraction import Abstraction
    return Abstraction.load(str(ABSTRACTION_6MAX))


def _advance_to_decision(state, rng):
    """Apply chance outcomes until a player-decision node (or terminal)."""
    steps = 0
    while not state.is_terminal() and state.is_chance_node() and steps < 200:
        outcomes = state.chance_outcomes()
        chosen = rng.choices([o[0] for o in outcomes],
                            weights=[o[1] for o in outcomes], k=1)[0]
        state.apply_action(chosen)
        steps += 1
    return state


class _FixedBucketAbstraction:
    """Returns a fixed bucket id regardless of cards. Street-agnostic, so the
    caller is responsible for choosing an id legal on the queried street."""

    def __init__(self, bucket_id: int):
        self.bucket_id = bucket_id

    def bucket_of(self, hero, board, runouts=200, rng=None):
        return self.bucket_id


# ===== (b) Pool sampling =====

class TestArchetypePool:
    def test_sample_returns_archetype_policy(self, calibration):
        pool = ArchetypePool(
            calibration_path=str(CALIB_6MAX), abstraction=object(),
            profile_names=None, bucket_runouts=30,
        )
        assert len(pool) == 5
        rng = random.Random(0)
        for _ in range(20):
            p = pool.sample_opponent(rng)
            assert isinstance(p, ArchetypePolicy)

    def test_profile_subset_respected(self, calibration):
        pool = ArchetypePool(
            calibration_path=str(CALIB_6MAX), abstraction=object(),
            profile_names=["NIT", "MANIAC"], bucket_runouts=30,
        )
        assert len(pool) == 2
        rng = random.Random(1)
        seen = {pool.sample_opponent(rng).profile.name for _ in range(50)}
        assert seen == {ArchetypeName.NIT, ArchetypeName.MANIAC}

    def test_full_pool_shows_diversity(self, calibration):
        pool = ArchetypePool(
            calibration_path=str(CALIB_6MAX), abstraction=object(),
            profile_names=None, bucket_runouts=30,
        )
        rng = random.Random(2)
        seen = {pool.sample_opponent(rng).profile.name for _ in range(200)}
        assert len(seen) == 5  # all five profiles surface over 200 samples


# ===== (c) in_position mapping =====

class TestInPositionMapping:
    def _policy(self):
        return ArchetypePolicy(
            profile=_by_name(ArchetypeName.NIT),
            abstraction=object(), calibration=object(), bucket_runouts=30,
        )

    def test_postflop_btn_is_in_position(self):
        policy = self._policy()
        for dealer in range(6):
            for seat in range(6):
                pos = position_for_seat_with_dealer(seat, dealer, 6)
                parsed = {"dealer_seat": dealer, "current_player": seat,
                          "street_idx": 1}  # flop
                got = policy._in_position(parsed, street_idx=1)
                assert got == (pos == _BTN), (
                    f"postflop dealer={dealer} seat={seat} pos={pos}: "
                    f"in_position={got}, expected {pos == _BTN}"
                )

    def test_preflop_bb_is_in_position(self):
        policy = self._policy()
        for dealer in range(6):
            for seat in range(6):
                pos = position_for_seat_with_dealer(seat, dealer, 6)
                parsed = {"dealer_seat": dealer, "current_player": seat,
                          "street_idx": 0}  # preflop
                got = policy._in_position(parsed, street_idx=0)
                assert got == (pos == _BB), (
                    f"preflop dealer={dealer} seat={seat} pos={pos}: "
                    f"in_position={got}, expected {pos == _BB}"
                )


# ===== (d) dealer_seat-missing fallback =====

class TestDealerSeatFallback:
    def test_missing_dealer_seat_returns_false_with_warning(self, caplog):
        # Reset the one-time latch so this test sees the warning regardless of
        # test ordering.
        ArchetypePolicy._warned_no_dealer = False
        policy = ArchetypePolicy(
            profile=_by_name(ArchetypeName.NIT),
            abstraction=object(), calibration=object(), bucket_runouts=30,
        )
        parsed = {"current_player": 0, "street_idx": 1}  # no dealer_seat
        with caplog.at_level(logging.WARNING, logger="archetype6"):
            got = policy._in_position(parsed, street_idx=1)
        assert got is False
        assert any("dealer_seat" in r.message for r in caplog.records), \
            "expected a one-time dealer_seat warning"

    def test_warning_fires_only_once(self, caplog):
        ArchetypePolicy._warned_no_dealer = False
        policy = ArchetypePolicy(
            profile=_by_name(ArchetypeName.NIT),
            abstraction=object(), calibration=object(), bucket_runouts=30,
        )
        parsed = {"current_player": 0, "street_idx": 1}
        with caplog.at_level(logging.WARNING, logger="archetype6"):
            for _ in range(5):
                policy._in_position(parsed, street_idx=1)
        warns = [r for r in caplog.records if "dealer_seat" in r.message]
        assert len(warns) == 1, f"expected exactly one warning, got {len(warns)}"


# ===== (a) Policy contract on a real state =====

class TestPolicyContract:
    def test_select_action_returns_legal_chip_action(self, calibration, real_abstraction):
        import pyspiel
        from src.nlhe.game_strings import six_max_sng
        from src.nlhe.infoset6 import parse_state_6max

        game = pyspiel.load_game(six_max_sng(1500))
        state = _advance_to_decision(game.new_initial_state(), random.Random(7))
        assert not state.is_terminal() and not state.is_chance_node()
        parsed = parse_state_6max(state)
        legal = set(state.legal_actions())

        policy = ArchetypePolicy(
            profile=_by_name(ArchetypeName.TAG),
            abstraction=real_abstraction, calibration=calibration,
            bucket_runouts=10,
        )
        chip = policy.select_action(parsed, state, random.Random(123), mode="sample")
        assert chip in legal

    def test_select_action_deterministic_under_seed(self, calibration, real_abstraction):
        import pyspiel
        from src.nlhe.game_strings import six_max_sng
        from src.nlhe.infoset6 import parse_state_6max

        game = pyspiel.load_game(six_max_sng(1500))
        state = _advance_to_decision(game.new_initial_state(), random.Random(7))
        parsed = parse_state_6max(state)
        policy = ArchetypePolicy(
            profile=_by_name(ArchetypeName.LAG),
            abstraction=real_abstraction, calibration=calibration,
            bucket_runouts=10,
        )
        a = policy.select_action(parsed, state, random.Random(999), mode="sample")
        b = policy.select_action(parsed, state, random.Random(999), mode="sample")
        assert a == b


# ===== (f) Style sanity through the adapter =====

class TestStyleThroughAdapter:
    def test_nit_folds_more_buckets_than_maniac(self, calibration):
        """Through the adapter, NIT (tight) argmax-folds on strictly more
        preflop buckets than MANIAC (loose). Iterates all preflop buckets with
        a fixed-bucket abstraction so equity is controlled; mirrors
        test_archetypes.test_nit_strictly_tighter_than_maniac at adapter level.
        """
        import pyspiel
        from src.nlhe.actions import discretize_legal_actions
        from src.nlhe.cfr6 import _build_view_6max
        from src.nlhe.game_strings import six_max_sng
        from src.nlhe.infoset6 import parse_state_6max

        # A real preflop decision node where there is a bet to call (fold legal).
        game = pyspiel.load_game(six_max_sng(1500))
        state = _advance_to_decision(game.new_initial_state(), random.Random(3))
        parsed = parse_state_6max(state)
        view = _build_view_6max(state, parsed)
        discrete = discretize_legal_actions(list(state.legal_actions()), view)
        if DiscreteAction.FOLD not in discrete:
            pytest.skip("first decision node has no legal fold; deal-dependent")
        fold_chip = discrete[DiscreteAction.FOLD]

        def folds_on_bucket(profile_name, bucket_id):
            abst = _FixedBucketAbstraction(bucket_id)
            policy = ArchetypePolicy(_by_name(profile_name), abst, calibration, 10)
            chip = policy.select_action(parsed, state, random.Random(0), "argmax")
            return chip == fold_chip

        n_buckets = len(calibration.bucket_equity["preflop"])
        nit_folds = sum(folds_on_bucket(ArchetypeName.NIT, b) for b in range(n_buckets))
        maniac_folds = sum(folds_on_bucket(ArchetypeName.MANIAC, b) for b in range(n_buckets))
        assert nit_folds > maniac_folds, (
            f"NIT should fold on more buckets than MANIAC; "
            f"NIT={nit_folds}, MANIAC={maniac_folds} of {n_buckets}"
        )


# ===== (e) strat-buffer zero-write invariant during archetype hands =====

class TestStratBufferSuppression:
    """At archetype_mix=1.0 every opponent decision is an archetype, which
    reaches training via the cfr6 short-circuit and never writes to the
    strategy buffer (DECISIONS.md:216). Contrast: archetype_mix=0.0 fills it."""

    def _config(self, archetype_mix):
        from src.nlhe.solver6 import TrainConfig6Max
        return TrainConfig6Max(
            starting_stack=1500, big_blind=100, small_blind=50,
            payout_mode="double_up", hidden_dim=[16, 16],
            n_iterations=2, traversals_per_iter=6, train_steps_per_iter=2,
            batch_size=4, learning_rate=1e-3, buffer_capacity=2000,
            bucket_runouts=10, max_traversal_depth=200, seed=2026,
            archetype_mix=archetype_mix,
            archetype_calibration_path=str(CALIB_6MAX),
        )

    def test_zero_strat_writes_at_full_archetype_mix(self, calibration, real_abstraction):
        import pyspiel
        from src.nlhe.game_strings import six_max_sng
        from src.nlhe.solver6 import DeepCFR6MaxSolver

        game = pyspiel.load_game(six_max_sng(1500))
        solver = DeepCFR6MaxSolver(
            game=game, abstraction=real_abstraction,
            config=self._config(archetype_mix=1.0), logger=lambda s: None,
        )
        solver.train()
        assert len(solver.policy_nets.strat_buffer) == 0, (
            "strat_buffer must stay empty when every opponent is an archetype "
            f"(got {len(solver.policy_nets.strat_buffer)})"
        )

    def test_strat_buffer_fills_at_zero_archetype_mix(self, calibration, real_abstraction):
        # Contrast: pool built (path set) but archetype_mix=0.0 → pure self-play
        # → strat_buffer fills. Proves the zero-write test isn't vacuous.
        import pyspiel
        from src.nlhe.game_strings import six_max_sng
        from src.nlhe.solver6 import DeepCFR6MaxSolver

        game = pyspiel.load_game(six_max_sng(1500))
        solver = DeepCFR6MaxSolver(
            game=game, abstraction=real_abstraction,
            config=self._config(archetype_mix=0.0), logger=lambda s: None,
        )
        solver.train()
        assert len(solver.policy_nets.strat_buffer) > 0
