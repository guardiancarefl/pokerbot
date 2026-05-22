"""Unit tests for the four top-level helpers in src/nlhe/policy_adapter.py.

These exercise the pure translation logic only. End-to-end adapter behavior
(checkpoint load, replay, network forward, action sampling) is validated by
the live plumbing-test run against Slumbot, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlhe.policy_adapter import (
    _extract_card_from_action_string,
    openspiel_action_to_slumbot_token,
    pick_deck_action,
    slumbot_token_to_openspiel_action,
)


class _FakeChanceState:
    """Minimal stand-in for a pyspiel state at a chance node."""

    def __init__(self, deck: list[str], extra_garbage: list[str] | None = None):
        # deck holds the card tokens we'll synthesize action strings for.
        # extra_garbage holds any non-card action_to_string entries to verify
        # the extractor's tolerance for malformed siblings.
        self._cards = list(deck)
        self._garbage = list(extra_garbage or [])
        self._legal = list(range(len(self._cards) + len(self._garbage)))

    def legal_actions(self) -> list[int]:
        return list(self._legal)

    def action_to_string(self, a: int) -> str:
        if a < len(self._cards):
            return f"player=-1 move=Deal {self._cards[a]}"
        return self._garbage[a - len(self._cards)]


class _FakeDecisionState:
    """Stand-in for a non-chance state — only used for the call/check ambiguity
    in openspiel_action_to_slumbot_token."""

    def __init__(self, legal: list[int]):
        self._legal = list(legal)

    def legal_actions(self) -> list[int]:
        return list(self._legal)


# ----- _extract_card_from_action_string -----

class TestExtractCard:
    def test_extracts_canonical_chance_format(self):
        assert _extract_card_from_action_string("player=-1 move=Deal 2c") == "2c"

    def test_extracts_each_rank(self):
        for rank in "23456789TJQKA":
            for suit in "cdhs":
                tok = rank + suit
                s = f"player=-1 move=Deal {tok}"
                assert _extract_card_from_action_string(s) == tok

    def test_extra_whitespace_tolerated(self):
        assert _extract_card_from_action_string("Deal  Ah ") == "Ah"

    def test_raises_on_garbage(self):
        with pytest.raises(ValueError, match="no 'Deal <card>'"):
            _extract_card_from_action_string("not a deal at all")

    def test_raises_on_malformed_card(self):
        # 'Xz' is not a valid card token.
        with pytest.raises(ValueError):
            _extract_card_from_action_string("player=-1 move=Deal Xz")


# ----- slumbot_token_to_openspiel_action -----

class TestSlumbotToOpenSpiel:
    def test_fold(self):
        assert slumbot_token_to_openspiel_action("f") == 0

    def test_call(self):
        assert slumbot_token_to_openspiel_action("c") == 1

    def test_check(self):
        assert slumbot_token_to_openspiel_action("k") == 1

    def test_bet_target(self):
        assert slumbot_token_to_openspiel_action("b350") == 350
        assert slumbot_token_to_openspiel_action("b2") == 2
        assert slumbot_token_to_openspiel_action("b20000") == 20000

    def test_raises_on_bare_b(self):
        with pytest.raises(ValueError, match="unrecognized"):
            slumbot_token_to_openspiel_action("b")

    def test_raises_on_b_with_non_digits(self):
        with pytest.raises(ValueError):
            slumbot_token_to_openspiel_action("babc")

    def test_raises_on_unknown_token(self):
        with pytest.raises(ValueError):
            slumbot_token_to_openspiel_action("x")
        with pytest.raises(ValueError):
            slumbot_token_to_openspiel_action("")

    def test_raises_on_bet_target_below_two(self):
        # 0 and 1 collide with fold/call; OpenSpiel doesn't represent a bet of 1.
        with pytest.raises(ValueError, match=">= 2"):
            slumbot_token_to_openspiel_action("b1")
        with pytest.raises(ValueError, match=">= 2"):
            slumbot_token_to_openspiel_action("b0")

    def test_b_token_with_prior_commitment(self):
        # Flop b100 with 300 chips committed preflop -> OpenSpiel int 400.
        assert slumbot_token_to_openspiel_action(
            "b100", prior_streets_committed_by_actor=300,
        ) == 400

    def test_b_token_prior_commitment_default_zero(self):
        # Default kwarg keeps preflop semantics unchanged.
        assert slumbot_token_to_openspiel_action("b350") == \
            slumbot_token_to_openspiel_action("b350", prior_streets_committed_by_actor=0)


# ----- openspiel_action_to_slumbot_token -----

class TestOpenSpielToSlumbot:
    def test_fold(self):
        st = _FakeDecisionState(legal=[0, 1])
        assert openspiel_action_to_slumbot_token(0, st) == "f"

    def test_call_when_facing_bet(self):
        # Fold is legal -> we're facing a bet -> 1 means call.
        st = _FakeDecisionState(legal=[0, 1, 200, 400])
        assert openspiel_action_to_slumbot_token(1, st) == "c"

    def test_check_when_not_facing_bet(self):
        # Fold is NOT legal -> nothing to call -> 1 means check.
        st = _FakeDecisionState(legal=[1, 100, 200])
        assert openspiel_action_to_slumbot_token(1, st) == "k"

    def test_bet_passes_through(self):
        st = _FakeDecisionState(legal=[0, 1, 350, 800])
        assert openspiel_action_to_slumbot_token(350, st) == "b350"
        assert openspiel_action_to_slumbot_token(800, st) == "b800"
        assert openspiel_action_to_slumbot_token(2, st) == "b2"

    def test_raises_on_negative_action(self):
        st = _FakeDecisionState(legal=[0, 1])
        with pytest.raises(ValueError):
            openspiel_action_to_slumbot_token(-1, st)

    def test_b_action_with_prior_commitment(self):
        # OpenSpiel int 400 with 300 preflop -> wire 'b100'.
        st = _FakeDecisionState(legal=[1, 400, 500, 600, 1000])
        assert openspiel_action_to_slumbot_token(
            400, st, prior_streets_committed_by_actor=300,
        ) == "b100"

    def test_b_action_prior_commitment_default_zero(self):
        # Default kwarg keeps preflop semantics unchanged.
        st = _FakeDecisionState(legal=[0, 1, 350])
        assert openspiel_action_to_slumbot_token(350, st) == \
            openspiel_action_to_slumbot_token(350, st, prior_streets_committed_by_actor=0)


# ----- pick_deck_action -----

class TestPickDeckAction:
    def test_finds_known_card(self):
        st = _FakeChanceState(deck=["2c", "2d", "Kh", "As"])
        a = pick_deck_action(st, lambda c: c == "Kh", "find Kh")
        assert a == 2  # third in the list

    def test_first_match_wins(self):
        st = _FakeChanceState(deck=["2c", "Kh", "Kh", "As"])
        # Both indices 1 and 2 would match, but we should return the first.
        a = pick_deck_action(st, lambda c: c == "Kh", "find Kh")
        assert a == 1

    def test_any_card_predicate(self):
        st = _FakeChanceState(deck=["2c", "2d", "Kh"])
        a = pick_deck_action(st, lambda c: True, "any card")
        assert a == 0  # first legal

    def test_raises_with_purpose_when_no_match(self):
        st = _FakeChanceState(deck=["2c", "2d", "Kh"])
        with pytest.raises(RuntimeError, match="find unicorn"):
            pick_deck_action(st, lambda c: c == "Js", "find unicorn")

    def test_skips_garbage_action_strings(self):
        # Mix in non-Deal entries; predicate should still find the real card.
        st = _FakeChanceState(
            deck=["2c", "Kh"],
            extra_garbage=["not a deal", "player=-1 move=Pass"],
        )
        a = pick_deck_action(st, lambda c: c == "Kh", "find Kh among garbage")
        assert a == 1

    def test_error_lists_some_available_cards(self):
        st = _FakeChanceState(deck=["2c", "2d", "2h", "Ks"])
        with pytest.raises(RuntimeError) as ei:
            pick_deck_action(st, lambda c: c == "Js", "find Js")
        # Diagnostic should help debugging.
        assert "2c" in str(ei.value)
        assert "Ks" in str(ei.value) or "..." in str(ei.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
