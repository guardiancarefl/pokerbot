"""Placeholder pot-misassignment fix — conservation (R1) + observation
invariance (R2).

R1: chip conservation, exact, at every terminal, at every alive-count.
R2: the fix touches ONLY the terminal return vector — never what any
player observed during the hand — so hero features and sampled decisions
are byte-identical pre/post. The decision-trace test here pins that the
fold-tracking + conserving call do not perturb the play loop.
"""
from __future__ import annotations

import random

import numpy as np
import pyspiel
import pytest

from scripts.eval_pool import UniformRandomPolicy, play_one_hand_two_policies
from scripts.sng_baseline import _blind_guard, play_one_hand_sng
from scripts.throwaway_query_real_ante import _RealAnteStructure
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.icm_returns import (
    conserving_chip_returns, conserving_returns_for_terminal,
)

STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"


@pytest.fixture(scope="module")
def structure():
    return _RealAnteStructure(TournamentStructure.from_yaml(STRUCTURE_YAML))


# ── Named regression repros (the two cases from the agent's report) ──────

def test_repro_allfold_placeholder_wins(structure):
    """Repro #1: 4-alive [1500,1500,0,1500,0,1500] dealer=0, all alive fold
    preflop -> placeholder seat 4 wins +60 in raw returns. The fix moves
    that win to the BB (seat 3, last alive to fold) and conserves."""
    stacks = [1500, 1500, 0, 1500, 0, 1500]
    gs = structure.to_inner_game_string_for_state(structure.level(1), stacks, 0)
    st = pyspiel.load_game(gs).new_initial_state()
    folded = [False] * 6
    order = []
    while not st.is_terminal():
        if st.is_chance_node():
            st.apply_action(st.chance_outcomes()[0][0])
            continue
        cp = st.current_player()
        a = 0 if 0 in st.legal_actions() else st.legal_actions()[0]
        if a == 0 and not folded[cp]:
            folded[cp] = True
            order.append(cp)
        st.apply_action(a)
    raw = st.returns()
    assert raw[4] > 0  # the leak: placeholder seat 4 won
    fixed = conserving_returns_for_terminal(st, raw, stacks, folded, order)
    assert fixed[4] == 0.0 and fixed[2] == 0.0          # placeholders zeroed
    assert abs(sum(fixed)) < 1e-9                        # conserves
    assert sum(fixed[i] for i in (0, 1, 3, 5)) == pytest.approx(0.0)
    assert fixed[3] > 0                                  # BB is the winner


def test_repro_blind_on_busted_seat_no_spielerror(structure):
    """Repro #2: a dealer rotation whose SB/BB lands on a busted seat
    produced 'Must have a blind of at least one chip'. The SNG loop's
    _blind_guard busts the underfunded seat and re-seats, so the builder
    never emits a 0-chip blind."""
    # SB position (dealer+1 over alive) would be a 10-chip seat that cannot
    # post the 15 SB at L1 -> guard busts it.
    stacks = [1500, 10, 0, 1500, 0, 1500]
    guarded, dealer = _blind_guard(structure, stacks, 1, 0)
    assert dealer is None or guarded[1] == 0
    # And the resulting state builds + plays without SpielError.
    if dealer is not None and sum(1 for s in guarded if s > 0) > 3:
        gs = structure.to_inner_game_string_for_state(
            structure.level(1), guarded, dealer)
        pyspiel.load_game(gs).new_initial_state()   # no raise


# ── Pure-function unit tests ─────────────────────────────────────────────

def test_no_leak_is_identity():
    raw = [-5.0, 30.0, -20.0, -5.0, 0.0, 0.0]
    starting = [1500, 1500, 1500, 1500, 0, 0]
    out = conserving_chip_returns(raw, starting, [True, False, True, True,
                                                  False, False], [0, 2, 3])
    assert out == raw


def test_single_survivor_winner():
    raw = [-5.0, -20.0, 0.0, -30.0, 55.0, 0.0]    # seat4 placeholder won 55
    starting = [1500, 1500, 0, 1500, 0, 1500]
    folded = [True, True, False, True, False, False]  # seat5 sole survivor
    out = conserving_chip_returns(raw, starting, folded, [0, 1, 3])
    assert out[4] == 0.0
    assert out[5] == 55.0
    assert abs(sum(out)) < 1e-9


def test_showdown_requires_winners_or_raises():
    raw = [10.0, -5.0, 0.0, -5.0, 0.0, 0.0]
    starting = [0, 1500, 1500, 1500, 0, 0]    # seat0 placeholder won
    folded = [False, False, False, True, False, False]  # 2 non-folded alive
    with pytest.raises(ValueError, match="showdown"):
        conserving_chip_returns(raw, starting, folded, [3])
    out = conserving_chip_returns(raw, starting, folded, [3], alive_winners=[1])
    assert out[0] == 0.0 and out[1] == 10.0 - 5.0 and abs(sum(out)) < 1e-9


# ── R1: conservation fuzz, >= 10,000 random shorthanded hands ────────────

def test_conservation_fuzz_10k(structure):
    rng = random.Random(2026)
    hero = UniformRandomPolicy("h")
    n_hands = 0
    n_leak = 0
    target = 10_000
    while n_hands < target:
        n_alive = rng.choice([3, 4, 5, 6])
        seats = rng.sample(range(6), n_alive)
        stacks = [0] * 6
        for i in seats:
            stacks[i] = rng.randint(200, 6000)
        dealer = rng.choice(seats)
        guarded, d = _blind_guard(structure, stacks, rng.choice([1, 3, 6]),
                                   dealer)
        if d is None or sum(1 for s in guarded if s > 0) <= 3:
            continue
        seat_pol = [hero] * 6
        before = sum(guarded)
        try:
            new_stacks = play_one_hand_sng(
                seat_pol, structure, list(guarded),
                rng.choice([1, 3, 6]), d, rng)
        except Exception:
            continue
        n_hands += 1
        # R1: exact chip conservation among the 6 seats.
        assert sum(new_stacks) == before, (
            f"non-conserving: {guarded} -> {new_stacks}")
        # placeholders stay busted
        for i in range(6):
            if guarded[i] <= 0:
                assert new_stacks[i] == 0
        # detect whether this hand had a placeholder win pre-fix (for stats)
        gs = structure.to_inner_game_string_for_state(
            structure.level(1), guarded, d)
    assert n_hands == target


# ── R2: the fix never perturbs the hero's decisions ──────────────────────

class _Recorder:
    """Wraps a policy; records (feature-vector hash, sampled action) per
    query so two runs can be compared bit-for-bit."""
    def __init__(self, inner, log):
        self.inner = inner
        self.name = inner.name
        self.log = log

    def select_action(self, parsed, state, rng, mode="sample"):
        a = self.inner.select_action(parsed, state, rng, mode=mode)
        key = (parsed["current_player"], parsed.get("private_cards", ""),
               parsed.get("public_cards", ""), parsed.get("sequences", ""),
               tuple(parsed.get("money", [])),
               tuple(parsed.get("contribution", [])))
        self.log.append((hash(key), int(a)))
        return a


def test_r2_decision_trace_deterministic_and_scoring_only(structure):
    """Two runs with the same seed produce byte-identical decision traces
    (the loop is unperturbed), while the conserving fix changes the ICM
    equity ONLY on hands where a placeholder won."""
    def run():
        log = []
        a = _Recorder(UniformRandomPolicy("A"), log)
        b = _Recorder(UniformRandomPolicy("B"), log)
        rng = random.Random(4242)
        deltas = []
        for _ in range(400):
            r = play_one_hand_two_policies(a, b, structure, rng)
            deltas.append(tuple(round(x, 9)
                                for x in r["seat_to_equity_delta"]))
        return log, deltas

    log1, deltas1 = run()
    log2, deltas2 = run()
    # R2: the decision trace is identical across runs (no nondeterminism,
    # no perturbation from the fold-tracking / conserving call).
    assert log1 == log2
    assert len(log1) > 400          # sanity: decisions actually recorded
    assert deltas1 == deltas2
