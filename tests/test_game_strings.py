"""Tests for src/nlhe/game_strings.py (Phase 4a: parametric game strings)."""
from __future__ import annotations
import random

import pyspiel
import pytest

from src.nlhe.game_strings import (
    PokerGameConfig,
    hunl_200bb,
    hunl_20bb,
    six_max_200bb,
    six_max_sng,
)


# ---- PokerGameConfig validation ----

def test_config_rejects_too_few_players():
    with pytest.raises(ValueError):
        PokerGameConfig(num_players=1)


def test_config_rejects_too_many_players():
    with pytest.raises(ValueError):
        PokerGameConfig(num_players=11)


def test_config_rejects_nonpositive_stack():
    with pytest.raises(ValueError):
        PokerGameConfig(starting_stack=0)
    with pytest.raises(ValueError):
        PokerGameConfig(starting_stack=-100)


def test_config_rejects_bad_blinds():
    with pytest.raises(ValueError):
        PokerGameConfig(small_blind=0)
    with pytest.raises(ValueError):
        PokerGameConfig(small_blind=100, big_blind=100)  # SB must be < BB
    with pytest.raises(ValueError):
        PokerGameConfig(small_blind=200, big_blind=100)  # SB > BB


def test_config_accepts_valid_defaults():
    cfg = PokerGameConfig()
    assert cfg.num_players == 2
    assert cfg.starting_stack == 20000


# ---- game string format ----

def test_hunl_string_contains_expected_pieces():
    s = hunl_200bb()
    assert "numPlayers=2" in s
    assert "stack=20000 20000" in s
    assert "blind=50 100" in s
    assert "bettingAbstraction=fullgame" in s


def test_six_max_string_contains_six_players():
    s = six_max_200bb()
    assert "numPlayers=6" in s
    # Stack repeated six times.
    assert "stack=20000 20000 20000 20000 20000 20000" in s
    # Blind padded with zeros for non-blind seats.
    assert "blind=50 100 0 0 0 0" in s


def test_first_player_differs_between_2_and_6():
    # HUNL: BB acts first preflop (firstPlayer=2 1 1 1)
    assert "firstPlayer=2 1 1 1" in hunl_200bb()
    # 6-max: UTG acts first preflop (firstPlayer=3 1 1 1)
    assert "firstPlayer=3 1 1 1" in six_max_200bb()


# ---- OpenSpiel can actually load each ----

@pytest.mark.parametrize("name,fn", [
    ("hunl_200bb", hunl_200bb),
    ("hunl_20bb", hunl_20bb),
    ("six_max_200bb", six_max_200bb),
    ("six_max_sng_default", six_max_sng),
])
def test_openspiel_loads(name, fn):
    """OpenSpiel can load each of the convenience configs."""
    game = pyspiel.load_game(fn())
    assert game.num_players() == (2 if "hunl" in name else 6)


def test_six_max_game_walks_to_terminal():
    """A 6-max game with random actions completes with zero-sum returns."""
    game = pyspiel.load_game(six_max_200bb())
    state = game.new_initial_state()
    rng = random.Random(2026)
    steps = 0
    while not state.is_terminal():
        steps += 1
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            a = rng.choices(actions, weights=probs, k=1)[0]
        else:
            a = rng.choice(state.legal_actions())
        state.apply_action(a)
        if steps > 5000:
            pytest.fail(f"6-max game did not reach terminal in 5000 steps")
    returns = state.returns()
    assert len(returns) == 6
    # Zero-sum: returns must sum to 0 (within float tolerance)
    assert abs(sum(returns)) < 1e-6, f"returns sum to {sum(returns)}, expected 0"


def test_six_max_sng_starting_stack_configurable():
    """six_max_sng(starting_stack=N) actually uses N."""
    s = six_max_sng(starting_stack=3000)
    assert "stack=3000 3000 3000 3000 3000 3000" in s


def test_three_handed_game_loads():
    """Edge case: 3-handed game (between HUNL and 6-max). Important for late-tournament play."""
    cfg = PokerGameConfig(num_players=3)
    game = pyspiel.load_game(cfg.to_universal_poker_string())
    assert game.num_players() == 3


# ---- BlindLevel validation (Phase 4f) ----

from src.nlhe.game_strings import BlindLevel


def test_blind_level_basic_construction():
    bl = BlindLevel(level=1, small_blind=15, big_blind=25, ante=5)
    assert bl.level == 1
    assert bl.small_blind == 15
    assert bl.big_blind == 25
    assert bl.ante == 5
    assert bl.duration_minutes == 5  # default


def test_blind_level_no_ante_default():
    bl = BlindLevel(level=1, small_blind=50, big_blind=100)
    assert bl.ante == 0


def test_blind_level_rejects_zero_level():
    with pytest.raises(ValueError):
        BlindLevel(level=0, small_blind=15, big_blind=25)


def test_blind_level_rejects_negative_level():
    with pytest.raises(ValueError):
        BlindLevel(level=-1, small_blind=15, big_blind=25)


def test_blind_level_rejects_zero_small_blind():
    with pytest.raises(ValueError):
        BlindLevel(level=1, small_blind=0, big_blind=25)


def test_blind_level_rejects_zero_big_blind():
    with pytest.raises(ValueError):
        BlindLevel(level=1, small_blind=15, big_blind=0)


def test_blind_level_rejects_sb_equals_bb():
    with pytest.raises(ValueError):
        BlindLevel(level=1, small_blind=25, big_blind=25)


def test_blind_level_rejects_sb_greater_than_bb():
    with pytest.raises(ValueError):
        BlindLevel(level=1, small_blind=30, big_blind=25)


def test_blind_level_rejects_negative_ante():
    with pytest.raises(ValueError):
        BlindLevel(level=1, small_blind=15, big_blind=25, ante=-1)


def test_blind_level_rejects_zero_duration():
    with pytest.raises(ValueError):
        BlindLevel(level=1, small_blind=15, big_blind=25, duration_minutes=0)


def test_blind_level_inflated_big_blind_no_ante():
    bl = BlindLevel(level=1, small_blind=50, big_blind=100, ante=0)
    assert bl.inflated_big_blind(6) == 100  # no inflation


def test_blind_level_inflated_big_blind_with_ante():
    # Ignition Double Up Turbo level 1: SB=15, BB=25, ante=5
    bl = BlindLevel(level=1, small_blind=15, big_blind=25, ante=5)
    assert bl.inflated_big_blind(6) == 55  # 25 + 6*5
    assert bl.inflated_big_blind(2) == 35  # 25 + 2*5


def test_blind_level_inflated_big_blind_preserves_total_dead_money():
    # The inflated BB should equal what would actually go in the pot pre-action
    # from blinds + antes if the BB were the only contributor.
    bl = BlindLevel(level=3, small_blind=50, big_blind=100, ante=15)
    n = 6
    # Real Ignition: SB=50, BB=100, 6 antes of 15 each
    real_pot = bl.small_blind + bl.big_blind + n * bl.ante  # 50 + 100 + 90 = 240
    # Inflated approximation: SB=50, BB=190, no antes
    inflated_pot = bl.small_blind + bl.inflated_big_blind(n)  # 50 + 190 = 240
    assert real_pot == inflated_pot


def test_blind_level_inflated_big_blind_rejects_solo_player():
    bl = BlindLevel(level=1, small_blind=15, big_blind=25, ante=5)
    with pytest.raises(ValueError):
        bl.inflated_big_blind(1)


def test_blind_level_is_frozen():
    bl = BlindLevel(level=1, small_blind=15, big_blind=25, ante=5)
    with pytest.raises(Exception):  # FrozenInstanceError in 3.10
        bl.level = 2


# ---- TournamentStructure validation (Phase 4f) ----

from src.nlhe.game_strings import TournamentStructure


def _basic_schedule():
    return (
        BlindLevel(level=1, small_blind=15, big_blind=25, ante=5),
        BlindLevel(level=2, small_blind=25, big_blind=50, ante=10),
        BlindLevel(level=3, small_blind=50, big_blind=100, ante=15),
    )


def _basic_tournament(**overrides):
    """Helper to construct a valid TournamentStructure with optional overrides."""
    defaults = dict(
        format_name='test_format',
        num_players=6,
        starting_chips=1500,
        payout_mode='double_up',
        payouts_dollars=(10.0, 10.0, 10.0),
        buy_in_dollars=5.0,
        level_duration_minutes=5,
        blind_schedule=_basic_schedule(),
        training_weights=((1, 0.5), (2, 0.3), (3, 0.2)),
    )
    defaults.update(overrides)
    return TournamentStructure(**defaults)


def test_tournament_structure_basic_construction():
    ts = _basic_tournament()
    assert ts.format_name == 'test_format'
    assert ts.num_players == 6
    assert ts.starting_chips == 1500
    assert ts.num_paid() == 3
    assert ts.total_chips_in_play() == 9000
    assert ts.buy_in_chips() == 1500


def test_tournament_structure_level_lookup():
    ts = _basic_tournament()
    bl = ts.level(2)
    assert bl.level == 2
    assert bl.small_blind == 25
    assert bl.big_blind == 50
    assert bl.ante == 10


def test_tournament_structure_level_lookup_raises_on_missing():
    ts = _basic_tournament()
    with pytest.raises(KeyError):
        ts.level(99)


def test_tournament_structure_rejects_empty_format_name():
    with pytest.raises(ValueError):
        _basic_tournament(format_name='')


def test_tournament_structure_rejects_too_few_players():
    with pytest.raises(ValueError):
        _basic_tournament(num_players=1)


def test_tournament_structure_rejects_too_many_players():
    with pytest.raises(ValueError):
        _basic_tournament(num_players=11)


def test_tournament_structure_rejects_zero_starting_chips():
    with pytest.raises(ValueError):
        _basic_tournament(starting_chips=0)


def test_tournament_structure_rejects_unknown_payout_mode():
    with pytest.raises(ValueError):
        _basic_tournament(payout_mode='knockout')


def test_tournament_structure_rejects_empty_payouts():
    with pytest.raises(ValueError):
        _basic_tournament(payouts_dollars=())


def test_tournament_structure_rejects_nonpositive_payouts():
    with pytest.raises(ValueError):
        _basic_tournament(payouts_dollars=(10.0, 0.0, 10.0))


def test_tournament_structure_double_up_requires_equal_payouts():
    with pytest.raises(ValueError):
        _basic_tournament(payouts_dollars=(10.0, 5.0, 10.0))


def test_tournament_structure_rejects_zero_buy_in():
    with pytest.raises(ValueError):
        _basic_tournament(buy_in_dollars=0)


def test_tournament_structure_rejects_zero_level_duration():
    with pytest.raises(ValueError):
        _basic_tournament(level_duration_minutes=0)


def test_tournament_structure_rejects_empty_schedule():
    with pytest.raises(ValueError):
        _basic_tournament(blind_schedule=())


def test_tournament_structure_rejects_non_blind_level_entries():
    with pytest.raises(ValueError):
        _basic_tournament(blind_schedule=(1, 2, 3))  # ints, not BlindLevels


def test_tournament_structure_rejects_non_ascending_schedule():
    bad = (
        BlindLevel(level=1, small_blind=15, big_blind=25),
        BlindLevel(level=3, small_blind=50, big_blind=100),
        BlindLevel(level=2, small_blind=25, big_blind=50),  # out of order
    )
    with pytest.raises(ValueError):
        _basic_tournament(blind_schedule=bad)


def test_tournament_structure_rejects_duplicate_levels():
    bad = (
        BlindLevel(level=1, small_blind=15, big_blind=25),
        BlindLevel(level=1, small_blind=25, big_blind=50),  # duplicate
    )
    with pytest.raises(ValueError):
        _basic_tournament(blind_schedule=bad)


def test_tournament_structure_rejects_weights_referencing_missing_level():
    # training_weights references level 99 which isn't in the schedule
    with pytest.raises(ValueError):
        _basic_tournament(training_weights=((1, 0.5), (99, 0.5)))


def test_tournament_structure_rejects_weights_summing_below_1():
    with pytest.raises(ValueError):
        _basic_tournament(training_weights=((1, 0.5), (2, 0.3)))  # sums to 0.8


def test_tournament_structure_rejects_weights_summing_above_1():
    with pytest.raises(ValueError):
        _basic_tournament(training_weights=((1, 0.6), (2, 0.6)))  # sums to 1.2


def test_tournament_structure_rejects_negative_weights():
    with pytest.raises(ValueError):
        _basic_tournament(training_weights=((1, 1.5), (2, -0.5)))  # sums to 1.0 but negative


def test_tournament_structure_accepts_empty_weights():
    # No training_weights is fine — sampler will fall back to uniform
    ts = _basic_tournament(training_weights=())
    assert ts.training_weights == ()


def test_tournament_structure_is_frozen():
    ts = _basic_tournament()
    with pytest.raises(Exception):  # FrozenInstanceError
        ts.format_name = 'something_else'


# ---- TournamentStructure game-string builders (Phase 4f) ----


def test_to_inner_game_string_no_ante():
    """A tournament with no antes should produce a game string with raw blinds."""
    bls = (BlindLevel(level=1, small_blind=50, big_blind=100, ante=0),)
    ts = TournamentStructure(
        format_name='no_ante', num_players=6, starting_chips=1500,
        payout_mode='double_up', payouts_dollars=(10.0, 10.0, 10.0),
        buy_in_dollars=5.0, level_duration_minutes=5,
        blind_schedule=bls, training_weights=(),
    )
    gs = ts.to_inner_game_string(level=1)
    assert "universal_poker" in gs
    assert "blind=50 100 0 0 0 0" in gs  # raw bb=100, no inflation
    assert "numPlayers=6" in gs


def test_to_inner_game_string_with_ante_inflates_bb():
    """Antes should be absorbed into the big blind via inflation."""
    bls = (BlindLevel(level=1, small_blind=15, big_blind=25, ante=5),)
    ts = TournamentStructure(
        format_name='with_ante', num_players=6, starting_chips=1500,
        payout_mode='double_up', payouts_dollars=(10.0, 10.0, 10.0),
        buy_in_dollars=5.0, level_duration_minutes=5,
        blind_schedule=bls, training_weights=(),
    )
    gs = ts.to_inner_game_string(level=1)
    assert "blind=15 55 0 0 0 0" in gs  # 25 + 6*5 = 55


def test_to_inner_game_string_loads_in_pyspiel():
    """The produced inner string should be loadable by OpenSpiel."""
    import pyspiel
    ts = TournamentStructure.from_yaml('configs/ignition_double_up_6max_turbo.yaml')
    gs = ts.to_inner_game_string(level=1)
    game = pyspiel.load_game(gs)
    assert game.num_players() == 6


def test_to_inner_game_string_for_different_levels():
    """Different levels should produce different game strings."""
    ts = TournamentStructure.from_yaml('configs/ignition_double_up_6max_turbo.yaml')
    gs1 = ts.to_inner_game_string(level=1)
    gs5 = ts.to_inner_game_string(level=5)
    assert gs1 != gs5
    assert "15 55" in gs1     # level 1: sb=15, bb=25+30=55
    assert "100 380" in gs5   # level 5: sb=100, bb=200+180=380


def test_to_blind_schedule_string_format():
    """blind_schedule should be ';'-separated <hands>:<sb>/<bb> with antes inflated."""
    bls = (
        BlindLevel(level=1, small_blind=15, big_blind=25, ante=5),
        BlindLevel(level=2, small_blind=25, big_blind=50, ante=10),
    )
    ts = TournamentStructure(
        format_name='x', num_players=6, starting_chips=1500,
        payout_mode='double_up', payouts_dollars=(10.0, 10.0, 10.0),
        buy_in_dollars=5.0, level_duration_minutes=5,
        blind_schedule=bls, training_weights=(),
    )
    sched = ts.to_blind_schedule_string(hands_per_level=10)
    # Expect: 10:15/55;10:25/110;
    assert sched == "10:15/55;10:25/110;"


def test_to_blind_schedule_string_rejects_zero_hands():
    ts = TournamentStructure.from_yaml('configs/ignition_double_up_6max_turbo.yaml')
    with pytest.raises(ValueError):
        ts.to_blind_schedule_string(hands_per_level=0)


def test_to_repeated_poker_string_contains_outer_and_inner():
    ts = TournamentStructure.from_yaml('configs/ignition_double_up_6max_turbo.yaml')
    gs = ts.to_repeated_poker_string(max_num_hands=200, hands_per_level=10)
    assert gs.startswith("repeated_poker(")
    assert "universal_poker(" in gs
    assert "max_num_hands=200" in gs
    assert "rotate_dealer=True" in gs
    assert "reset_stacks=False" in gs


def test_to_repeated_poker_string_loads_in_pyspiel():
    """The full nested string should be loadable by OpenSpiel as a repeated_poker game."""
    import pyspiel
    ts = TournamentStructure.from_yaml('configs/ignition_double_up_6max_turbo.yaml')
    gs = ts.to_repeated_poker_string(max_num_hands=50, hands_per_level=5)
    game = pyspiel.load_game(gs)
    assert game.num_players() == 6
    state = game.new_initial_state()
    assert state.is_chance_node()  # game starts with the deal


def test_to_repeated_poker_string_rejects_zero_hands():
    ts = TournamentStructure.from_yaml('configs/ignition_double_up_6max_turbo.yaml')
    with pytest.raises(ValueError):
        ts.to_repeated_poker_string(max_num_hands=0)


def test_to_repeated_poker_walks_to_terminal():
    """A random-action playthrough should terminate cleanly with chip conservation."""
    import pyspiel
    import random
    rng = random.Random(2026)
    ts = TournamentStructure.from_yaml('configs/ignition_double_up_6max_turbo.yaml')
    gs = ts.to_repeated_poker_string(max_num_hands=30, hands_per_level=5)
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()

    steps = 0
    while not state.is_terminal() and steps < 5000:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            a = rng.choices(
                [o[0] for o in outcomes], weights=[o[1] for o in outcomes]
            )[0]
            state.apply_action(a)
        else:
            legal = state.legal_actions()
            a = rng.choice(legal)
            state.apply_action(a)
        steps += 1

    assert state.is_terminal(), f"failed to terminate in {steps} steps"
    # Chip conservation: returns sum to zero
    returns = state.returns()
    assert abs(sum(returns)) < 1e-6, f"chip non-conservation: returns sum to {sum(returns)}"


# ---- from_yaml loader (Phase 4f) ----


def test_from_yaml_loads_ignition_config():
    ts = TournamentStructure.from_yaml('configs/ignition_double_up_6max_turbo.yaml')
    assert ts.format_name == "ignition_double_up_6max_turbo"
    assert ts.num_players == 6
    assert ts.starting_chips == 1500
    assert ts.payout_mode == "double_up"
    assert ts.payouts_dollars == (10.0, 10.0, 10.0)
    assert ts.buy_in_dollars == 5.0
    assert len(ts.blind_schedule) == 17


def test_from_yaml_blind_schedule_ascending():
    ts = TournamentStructure.from_yaml('configs/ignition_double_up_6max_turbo.yaml')
    levels = [bl.level for bl in ts.blind_schedule]
    assert levels == sorted(levels)
    assert levels == list(range(1, 18))


def test_from_yaml_specific_levels_parsed_correctly():
    ts = TournamentStructure.from_yaml('configs/ignition_double_up_6max_turbo.yaml')
    # Level 1: SB=15, BB=25, ante=5
    l1 = ts.level(1)
    assert l1.small_blind == 15
    assert l1.big_blind == 25
    assert l1.ante == 5
    # Level 9: SB=400, BB=800, ante=120
    l9 = ts.level(9)
    assert l9.small_blind == 400
    assert l9.big_blind == 800
    assert l9.ante == 120


def test_from_yaml_training_weights_sum_to_one():
    ts = TournamentStructure.from_yaml('configs/ignition_double_up_6max_turbo.yaml')
    total = sum(w for _, w in ts.training_weights)
    assert 0.99 <= total <= 1.01, f"weights sum to {total}, not ~1.0"



# ---- to_inner_game_string_for_state tests (Phase 4f) ----

import pyspiel as _pyspiel_state


def test_for_state_full_table_blinds_correct():
    """All 6 alive, dealer=0: SB seat=1, BB seat=2, UTG seat=3."""
    ts = TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml"
    )
    stacks = [1500] * 6
    gs = ts.to_inner_game_string_for_state(
        blind_level=ts.level(1), stacks=stacks, dealer_seat=0,
    )
    # SB=15 on seat 1, BB=55 (inflated) on seat 2
    assert "blind=0 15 55 0 0 0" in gs
    # firstPlayer: UTG=4 (1-indexed=4), postflop=SB=2 (1-indexed=2)
    assert "firstPlayer=4 2 2 2" in gs


def test_for_state_busted_seats_get_placeholder():
    """Busted seats get stack=1 in the produced game string."""
    ts = TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml"
    )
    stacks = [0, 1500, 1500, 0, 1500, 4500]  # seats 0,3 busted
    gs = ts.to_inner_game_string_for_state(
        blind_level=ts.level(1), stacks=stacks, dealer_seat=1,
    )
    # busted seats get stack=1, alive seats keep their values
    assert "stack=1 1500 1500 1 1500 4500" in gs


def test_for_state_sb_bb_rotate_over_alive_seats():
    """With dealer=2 and seat 3 busted: SB should be seat 4 (next alive),
    not seat 3 (would be standard +1 but is busted)."""
    ts = TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml"
    )
    stacks = [1500, 1500, 1500, 0, 1500, 4500]  # seat 3 busted
    gs = ts.to_inner_game_string_for_state(
        blind_level=ts.level(1), stacks=stacks, dealer_seat=2,
    )
    # SB should land on seat 4 (next alive after dealer 2, skipping busted seat 3)
    # BB on seat 5 (next alive after seat 4)
    assert "blind=0 0 0 0 15 55" in gs


def test_for_state_dealer_must_be_alive():
    """dealer_seat with stack=0 should raise ValueError."""
    ts = TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml"
    )
    stacks = [0, 1500, 1500, 1500, 1500, 3500]  # seat 0 busted
    with pytest.raises(ValueError):
        ts.to_inner_game_string_for_state(
            blind_level=ts.level(1), stacks=stacks, dealer_seat=0,
        )


def test_for_state_produces_loadable_game():
    """The game string must be loadable by pyspiel."""
    ts = TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml"
    )
    stacks = [1500] * 6
    gs = ts.to_inner_game_string_for_state(
        blind_level=ts.level(1), stacks=stacks, dealer_seat=2,
    )
    game = _pyspiel_state.load_game(gs)
    assert game.num_players() == 6
    state = game.new_initial_state()
    assert state.is_chance_node()
