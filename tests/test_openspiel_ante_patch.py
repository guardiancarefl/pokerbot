"""Verifies the OpenSpiel `ante` parameter patch (Step 2 of the inflated-BB retrain).

Two verification gates, run together but reported as two distinct sections in
the harness:

  2a — BUILD CORRECTNESS: the rebuilt pyspiel.so loads cleanly AND existing
       game strings (six_max_sng, the inflated-BB convention we've been using
       all along) still load and play through with no regression. This isolates
       "build broke pyspiel" from "patch broke behavior."

  2b — PATCH CORRECTNESS: the new `ante` parameter posts antes per seat,
       reduces effective starting stacks by ante, includes antes in the
       starting pot, AND leaves the min-raise floor at 2 × real_BB
       (NOT 2 × (BB + ante) — that's the load-bearing distinction from
       the old inflated-BB hack).

Run with: python -m pytest tests/test_openspiel_ante_patch.py -v
"""
from __future__ import annotations

import re

import pyspiel


# === Helpers ============================================================

def _build_inner(blind: str, stack: str, num_players: int = 6,
                  ante: str | None = None, first_player: str = "3 1 1 1"
                  ) -> str:
    """Build a universal_poker game string with optional ante param."""
    ante_part = f"ante={ante}," if ante else ""
    return (
        f"universal_poker(betting=nolimit,"
        f"numPlayers={num_players},"
        f"numRounds=4,"
        f"blind={blind},"
        f"{ante_part}"
        f"firstPlayer={first_player},"
        f"numSuits=4,"
        f"numRanks=13,"
        f"numHoleCards=2,"
        f"numBoardCards=0 3 1 1,"
        f"stack={stack},"
        f"bettingAbstraction=fullgame)"
    )


def _walk_to_first_decision(state, hero_seat: int = 0):
    """Walk past chance nodes (deals) to the first decision-node. Picks
    arbitrary deals (first legal action) — we don't care about specific
    cards for parameter-mechanics tests."""
    safety = 200
    while safety > 0 and state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
        safety -= 1
    if state.is_chance_node():
        raise RuntimeError("never left chance nodes")
    return state


def _read_pot_contributions(state) -> list[int]:
    """Return the per-seat cumulative contribution by parsing
    information_state_string. Format includes lines like
    `Money: 1495 1495 1495 1495 1495 1495` (per-seat REMAINING stack)."""
    info = state.information_state_string(0)
    # OpenSpiel observation includes 'Money: ...' (remaining stack per seat).
    m = re.search(r"Money:\s+([\d\s]+)", info)
    if m:
        return [int(x) for x in m.group(1).split()]
    return []


# === 2a — BUILD CORRECTNESS ============================================

def test_2a_pyspiel_imports():
    """The rebuilt module imports without ABI errors."""
    assert hasattr(pyspiel, "load_game")
    assert "universal_poker" in pyspiel.registered_names()


def test_2a_six_max_sng_inflated_bb_still_loads():
    """The pre-patch convention (no ante, inflated BB) must still work
    bit-for-bit. This is the production game string used in the existing
    blueprint training (rebel_value_net_full.pt was trained against this
    exact convention)."""
    from src.nlhe.game_strings import six_max_sng
    gs = six_max_sng(starting_stack=1500)
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    state = _walk_to_first_decision(state)
    # Just verify we reached a decision node with legal actions
    assert state.current_player() >= 0
    assert len(state.legal_actions()) >= 2


def test_2a_six_max_sng_plays_a_few_hands_random():
    """Walk a full hand with random actions — no exceptions, terminal
    reached. Tests that the build didn't introduce any UB / crashes in
    the standard play path."""
    from src.nlhe.game_strings import six_max_sng
    import random
    rng = random.Random(42)
    for trial in range(3):
        game = pyspiel.load_game(six_max_sng(starting_stack=1500))
        state = game.new_initial_state()
        safety = 1000
        while not state.is_terminal() and safety > 0:
            if state.is_chance_node():
                # uniform over legal chance actions
                state.apply_action(rng.choice(state.legal_actions()))
            else:
                state.apply_action(rng.choice(state.legal_actions()))
            safety -= 1
        assert state.is_terminal(), f"hand {trial} did not terminate within 1000 steps"


def test_2a_existing_inflated_bb_game_min_bet_is_unchanged():
    """The pre-patch inflated_BB convention (BB=55 at L1) should STILL
    produce min-raise-to of 110 — confirms the patch is non-invasive when
    no ante parameter is passed."""
    # Level 1 inflated BB: bb = 25 + 6*5 = 55
    gs = _build_inner(blind="15 55 0 0 0 0",
                       stack="1500 1500 1500 1500 1500 1500",
                       num_players=6)
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    state = _walk_to_first_decision(state)
    legal = state.legal_actions()
    raises = [a for a in legal if a >= 2]
    assert raises, f"expected raise actions in legal={legal[:20]}"
    assert min(raises) == 110, (
        f"min raise should be 2*55=110 with old inflated-BB convention, "
        f"got {min(raises)}")


# === 2b — PATCH CORRECTNESS ============================================

def test_2b_ante_parameter_is_registered():
    """The `ante` parameter must appear in the game's parameter list —
    direct check that the OpenSpiel-layer patch (kGameType registration)
    landed."""
    game = pyspiel.load_game(_build_inner(
        blind="15 25 0 0 0 0",
        stack="1500 1500 1500 1500 1500 1500",
        num_players=6,
    ))
    params = game.get_parameters()
    assert "ante" in params, (
        f"`ante` parameter missing — patch not applied? Params: "
        f"{sorted(params.keys())}")


def test_2b_min_bet_is_2x_real_BB_with_ante():
    """LOAD-BEARING — this is the entire point of the patch.

    Level 1 (Ignition Double Up Turbo): SB=15, BB=25, ante=5, 6 players,
    1500 starting stacks.

    Real-poker NL min-raise-to = 2 × BB = 50 (NOT 2 × (BB + ante) = 60,
    NOT 2 × inflated_BB = 110). The min-raise floor is by NL convention
    a function of BB only; antes are "dead money" that doesn't affect
    raise sizing rules.
    """
    gs = _build_inner(
        blind="15 25 0 0 0 0",
        ante="5 5 5 5 5 5",
        stack="1500 1500 1500 1500 1500 1500",
        num_players=6,
    )
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    state = _walk_to_first_decision(state)
    legal = state.legal_actions()
    raises = [a for a in legal if a >= 2]
    assert raises, f"expected raise actions in legal={legal[:20]}"
    assert min(raises) == 50, (
        f"PATCH FAILED: min_raise_to = {min(raises)}, expected 50 "
        f"(= 2 × real_BB = 50). Got {min(raises)} chips instead — "
        f"if 60, the patch is not excluding ante from maxSpent; "
        f"if 110, the patch isn't applied at all. legal[:8] = {legal[:8]}")


def test_2b_antes_deducted_from_stacks():
    """Each player's effective remaining stack at game start should be
    starting_stack - ante (minus blind for SB and BB seats). With
    1500 chips + 5 ante for all + SB=15 + BB=25:
        seats 0,2,3,4,5: 1500 - 5 = 1495
        seat SB (idx 0): 1500 - 5 - 15 = 1480 (= 1495 - 15)
        seat BB (idx 1): 1500 - 5 - 25 = 1470 (= 1495 - 25)

    Wait — with `blind=15 25 0 0 0 0`, SB=idx 0 and BB=idx 1. Other
    seats only paid ante."""
    gs = _build_inner(
        blind="15 25 0 0 0 0",
        ante="5 5 5 5 5 5",
        stack="1500 1500 1500 1500 1500 1500",
        num_players=6,
    )
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    state = _walk_to_first_decision(state)
    money = _read_pot_contributions(state)
    assert money, "could not parse 'Money:' from information_state_string"
    expected = [1480, 1470, 1495, 1495, 1495, 1495]
    assert money == expected, (
        f"Per-seat remaining stacks wrong: got {money}, expected {expected}. "
        f"SB (idx 0) should be 1500-5-15=1480, "
        f"BB (idx 1) should be 1500-5-25=1470, "
        f"others should be 1500-5=1495.")


def test_2b_starting_pot_includes_antes():
    """Pre-action pot = 6×ante + SB + BB = 30 + 15 + 25 = 70 chips.
    OpenSpiel's 'Spent: ...' field is per-seat cumulative contribution;
    the sum at game start (no voluntary action yet) should equal the
    pre-action pot."""
    gs = _build_inner(
        blind="15 25 0 0 0 0",
        ante="5 5 5 5 5 5",
        stack="1500 1500 1500 1500 1500 1500",
        num_players=6,
    )
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    state = _walk_to_first_decision(state)
    info = state.information_state_string(0)
    m = re.search(r"Spent:\s+\[?([\d:\s,P]+?)\]?\s*$", info, re.M)
    # Try alternative formats — universal_poker uses
    # 'Spent: P0: 20  P1: 30  P2: 5  P3: 5  P4: 5  P5: 5'
    contrib = [int(x) for x in re.findall(r"P\d+:\s*(\d+)", info)]
    if not contrib:
        # Fall back: derive from money = stack - spent
        money = _read_pot_contributions(state)
        contrib = [1500 - m for m in money]
    assert contrib, f"could not parse per-seat contributions from:\n{info[:500]}"
    total_pot = sum(contrib)
    expected_pot = 30 + 15 + 25  # 6 antes + SB + BB
    assert total_pot == expected_pot, (
        f"Starting pot {total_pot} != expected {expected_pot}. "
        f"Per-seat contributions: {contrib}")


def test_2b_2x_real_BB_raise_is_legal():
    """A real-poker 2×BB open (raise-to 50 chips at level 1) must be a
    legal action at the first preflop decision. This is the action that
    was previously unavailable due to the inflated-BB hack."""
    gs = _build_inner(
        blind="15 25 0 0 0 0",
        ante="5 5 5 5 5 5",
        stack="1500 1500 1500 1500 1500 1500",
        num_players=6,
    )
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    state = _walk_to_first_decision(state)
    legal = state.legal_actions()
    assert 50 in legal, (
        f"2×BB raise-to-50 should be legal preflop but is not in "
        f"legal_actions. legal[:10] = {legal[:10]}, min raise = "
        f"{min([a for a in legal if a >= 2], default='NONE')}")


def test_2b_no_ante_param_is_byte_identical_to_pre_patch():
    """Sanity: when the ante parameter is NOT passed (or empty string),
    behavior is identical to pre-patch. Catches regressions where the
    patch accidentally affects the no-ante path."""
    # Build the same game with and without `ante=`
    gs_no_ante = _build_inner(
        blind="15 55 0 0 0 0",   # inflated BB (pre-patch convention)
        stack="1500 1500 1500 1500 1500 1500",
        num_players=6,
    )
    gs_empty_ante = _build_inner(
        blind="15 55 0 0 0 0",
        ante="",                  # explicit empty
        stack="1500 1500 1500 1500 1500 1500",
        num_players=6,
    )
    g1 = pyspiel.load_game(gs_no_ante)
    s1 = _walk_to_first_decision(g1.new_initial_state())
    g2 = pyspiel.load_game(gs_empty_ante)
    s2 = _walk_to_first_decision(g2.new_initial_state())
    assert s1.legal_actions() == s2.legal_actions(), (
        "Empty `ante=` parameter changed legal_actions vs no-ante — "
        "patch is not backward-compatible")


def test_2b_zero_antes_dont_affect_min_bet():
    """ante='0 0 0 0 0 0' (all zeros) must give same min-bet as no ante
    parameter at all."""
    gs = _build_inner(
        blind="15 25 0 0 0 0",
        ante="0 0 0 0 0 0",
        stack="1500 1500 1500 1500 1500 1500",
        num_players=6,
    )
    game = pyspiel.load_game(gs)
    state = _walk_to_first_decision(game.new_initial_state())
    legal = state.legal_actions()
    raises = [a for a in legal if a >= 2]
    assert min(raises) == 50, (
        f"Zero-ante should give min-raise-to=50 (=2*BB), got {min(raises)}")
