"""opponentsattable wiring: live alive-count instead of hardcoded 5.

Before this wiring, build_game_context set opponentsattable = num_players-1
(always 5 in 6-max), so every OpponentsAtTable=2/3/4 predicate in every
Shanky profile was dead and bake-off opponents played their 6-handed rules
at every table size. The wiring derives the seated count from the parsed
state: busted seats in single-hand game strings are stack=1/ante=0
placeholders (money + contribution == 1); seated seats always exceed that.

Behavioral proof uses KillPhilMTT's ACTUAL parsed rules. Note the profile's
4-6-handed blocks (OpponentsAtTable = 3/4/5) are clones of each other, and
the firstcallerposition atom that differentiates their limper sub-blocks is
unset by the bridge (defaults to 0) — so the live behavioral contrast is
between the 3-handed block (OpponentsAtTable = 2: jam Groups 1-10, KQs
included) and the 4-6-handed blocks (jam Groups 1-5 only, KQs folds via the
'when others fold force' catch-all).
"""
from __future__ import annotations

import pytest

from src.nlhe.scripted_bots.policy import build_game_context
from src.nlhe.scripted_bots.parser import ActionKind, parse_profile
from src.nlhe.scripted_bots.runtime import evaluate_profile

PROFILE_PATH = "data/shanky_profiles/KillPhilMTT.txt"


def _parsed_state(n_alive: int, hero_cards: str = "KsQs",
                  stacks_chips: int = 8000) -> dict:
    """Synthetic parse_state_6max output: n_alive seated seats (hero is
    seat 0, to act preflop, unraised), the rest busted placeholders
    (money=1, contribution=0) exactly as the single-hand game strings
    produce them."""
    num_players = 6
    money = []
    contribution = []
    for i in range(num_players):
        if i < n_alive:
            # Seated: ante 5 posted by everyone; blinds at seats 1/2.
            ante = 5
            blind = 15 if i == 1 else (25 if i == 2 else 0)
            money.append(stacks_chips - ante - blind)
            contribution.append(ante + blind)
        else:
            money.append(1)        # busted placeholder
            contribution.append(0)
    return {
        "num_players": num_players,
        "street_idx": 0,
        "current_player": 0,
        "pot": sum(contribution),
        "money": money,
        "contribution": contribution,
        "private_cards": hero_cards,
        "public_cards": "",
        "sequences": "",
        "big_blind": 25,
    }


# ── Wiring: opponentsattable == (seated − 1) at every table size ──────────

@pytest.mark.parametrize("n_alive,expected_oat", [(6, 5), (5, 4), (4, 3),
                                                  (3, 2), (2, 1)])
def test_opponentsattable_tracks_alive_count(n_alive, expected_oat):
    ctx = build_game_context(_parsed_state(n_alive), state=None,
                              big_blind_chips=25)
    assert ctx.opponentsattable == expected_oat


def test_other_predicate_inputs_unperturbed():
    parsed = _parsed_state(5)
    ctx = build_game_context(parsed, state=None, big_blind_chips=25)
    # Untouched by the wiring: chips-derived and action-count inputs.
    assert ctx.stacksize == pytest.approx((8000 - 5) / 25)
    # Pre-existing behavior pinned, not endorsed: `opponents` counts
    # money>0 seats, which includes the busted stack=1 placeholder
    # (5 here, not 4). Flagged in the C1a report; out of wiring scope.
    assert ctx.opponents == 5
    assert ctx.bigblindsize == 1.0     # left as-is (see C1a step-1 record)
    assert ctx.raises == 0 and ctx.calls == 0
    assert ctx.potsize == pytest.approx((5 * 5 + 15 + 25) / 25)


# ── Behavioral: KillPhilMTT's table-size rules actually wake up ───────────

@pytest.fixture(scope="module")
def killphil():
    return parse_profile(open(PROFILE_PATH).read())


def _action_at(killphil, n_alive, hero_cards="KsQs"):
    ctx = build_game_context(_parsed_state(n_alive, hero_cards), state=None,
                              big_blind_chips=25)
    return evaluate_profile(killphil, ctx)


def test_killphil_qts_unraised_jams_3handed_folds_6handed(killphil):
    # QTs sits in the 3-handed jam list ('unraised from any position Move
    # allin with G-1..10', OpponentsAtTable = 2 block: 'QT suited or J9
    # suited or ...') but OUTSIDE every 4-6-handed jam range — so it folds
    # via 'when others fold force' at full tables and jams 3-handed. Under
    # the old hardcoding (opponentsattable always 5) BOTH states folded.
    act3 = _action_at(killphil, n_alive=3, hero_cards="QsTs")
    act6 = _action_at(killphil, n_alive=6, hero_cards="QsTs")
    assert act3.kind == ActionKind.RAISE_MAX, act3
    assert act6.kind == ActionKind.FOLD, act6


def test_killphil_premium_jams_at_every_table_size(killphil):
    # AA is in the jam range of every block — table size must not matter.
    for n_alive in (3, 4, 5, 6):
        act = _action_at(killphil, n_alive, hero_cards="AsAh")
        assert act.kind == ActionKind.RAISE_MAX, (n_alive, act)


def test_killphil_4_to_6_handed_blocks_are_clones(killphil):
    # The profile's OpponentsAtTable = 3/4/5 blocks are mechanical clones
    # (their limper sub-blocks hinge on firstcallerposition, which the
    # bridge leaves unset — pre-existing gap, flagged in the C1a report).
    # Within 4-6 players the wiring therefore changes nothing for this
    # profile: QTs unraised folds at all three sizes, junk folds, and the
    # jam ranges match.
    for n_alive in (4, 5, 6):
        assert _action_at(killphil, n_alive,
                          hero_cards="QsTs").kind == ActionKind.FOLD
        assert _action_at(killphil, n_alive,
                          hero_cards="7h2c").kind == ActionKind.FOLD
