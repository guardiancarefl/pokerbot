"""SNG-baseline harness: dispatch, determinism, busts, and the corrected
hand-start n_alive axis (the exact case the A/B logs' money>0 expression
got wrong: mid-hand all-in players are alive; busted placeholders are not).

Cheap policies only — no checkpoint load."""
from __future__ import annotations

import random

import pytest

from scripts.sng_baseline import play_one_hand_sng, play_sng_game
from scripts.throwaway_query_real_ante import _RealAnteStructure
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.scripted_bots.policy import ShankyProfilePolicy

STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"


@pytest.fixture(scope="module")
def structure():
    return _RealAnteStructure(TournamentStructure.from_yaml(STRUCTURE_YAML))


@pytest.fixture(scope="module")
def littlegreen():
    return ShankyProfilePolicy(name="littlegreen",
                                profile_path="data/shanky_profiles/littlegreen.txt",
                                big_blind_chips=100)


class RecorderPolicy:
    """Returns the first legal action (fold/cheapest); records the seat it
    was queried for."""
    def __init__(self, seat):
        self.seat = seat
        self.queried_at = []

    def select_action(self, parsed, state, rng, mode="sample"):
        self.queried_at.append(parsed["current_player"])
        return int(state.legal_actions()[0])


class AllInPolicy:
    """Always takes the largest legal chip action (shove)."""
    def __init__(self):
        self.n_queries = 0

    def select_action(self, parsed, state, rng, mode="sample"):
        self.n_queries += 1
        return int(max(state.legal_actions()))


def test_dispatch_each_seat_queries_its_own_policy(structure):
    stubs = [RecorderPolicy(i) for i in range(6)]
    rng = random.Random(1)
    play_one_hand_sng(stubs, structure, [1500] * 6, level_idx=1,
                       dealer_seat=0, rng=rng)
    queried_any = False
    for stub in stubs:
        for cp in stub.queried_at:
            assert cp == stub.seat, (cp, stub.seat)
            queried_any = True
    assert queried_any


def test_busted_seat_never_acts_and_stays_busted(structure):
    stubs = [RecorderPolicy(i) for i in range(6)]
    stacks = [1500, 1500, 0, 1500, 0, 1500]
    rng = random.Random(2)
    new_stacks = play_one_hand_sng(stubs, structure, stacks, level_idx=1,
                                    dealer_seat=0, rng=rng)
    assert new_stacks[2] == 0 and new_stacks[4] == 0
    assert stubs[2].queried_at == [] and stubs[4].queried_at == []
    # chip conservation across the hand
    assert sum(new_stacks) == sum(stacks)


def test_deterministic_same_seed(structure, littlegreen):
    from scripts.eval_pool import UniformRandomPolicy
    hero = UniformRandomPolicy("hero")
    accs = []
    recs = []
    for _ in range(2):
        acc = {}
        recs.append(play_sng_game(hero, littlegreen, structure, seed=99,
                                   hands_per_level=3, max_hands=60,
                                   stage_acc=acc))
        accs.append(acc)
    assert recs[0] == recs[1]
    assert accs[0] == accs[1]


def test_game_terminates_at_bubble_and_escalates(structure, littlegreen):
    from scripts.eval_pool import UniformRandomPolicy
    hero = UniformRandomPolicy("hero")
    rec = play_sng_game(hero, littlegreen, structure, seed=7,
                         hands_per_level=2, max_hands=120)
    assert not rec["tainted"]
    if not rec["capped"]:
        assert rec["n_alive_end"] <= 3
    if rec["hands"] >= 4:
        assert rec["level_end"] >= 2          # escalation every 2 hands
    assert rec["hero_net"] in (1.0, -1.0) or rec["capped"]


def test_hand_start_n_alive_counts_midhand_allins(structure):
    # 4 seated seats, everyone shoves -> every seated player's money hits 0
    # MID-hand. The hand-start axis must say n_alive=4; the A/B logs'
    # money>0 expression would have reported 0-2 at those decision points
    # (and would count the two stack=1 placeholders once dealt in).
    allin = AllInPolicy()
    acc = {}
    rec = play_sng_game(None, None, structure, seed=11, hands_per_level=5,
                         max_hands=5, stage_acc=acc,
                         starting_stacks=[300, 300, 300, 300, 0, 0],
                         seat_to_policy=[allin] * 6)
    assert allin.n_queries > 0
    n_alive_keys = {k[0] for k in acc}
    assert n_alive_keys == {4}, acc
    assert not rec["tainted"]
    assert rec["n_alive_end"] <= 3            # the multiway shove busted past bubble


def test_tainted_game_is_flagged_with_exception_detail(structure):
    class BoomPolicy:
        def select_action(self, parsed, state, rng, mode="sample"):
            raise ValueError("boom")
    rec = play_sng_game(None, None, structure, seed=3, max_hands=5,
                         starting_stacks=[1500] * 6,
                         seat_to_policy=[BoomPolicy()] * 6)
    assert rec["tainted"] is True
    assert "ValueError" in rec["exception"]
    assert "hero_net" in rec                  # record still well-formed
