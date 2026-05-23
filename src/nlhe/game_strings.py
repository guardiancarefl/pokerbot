"""Build OpenSpiel universal_poker game strings parametrically.

Phase 4a (docs/PHASE4_PLAN.md). The HUNL configs hardcoded game strings
with numPlayers=2; for 6-max we need the same shape with numPlayers=6.
This module builds the string from named parameters so configs can say
"6-max, 1500 chip starting stack, 100/50 blinds" instead of pasting a
multi-line string with hidden parameters.

NOTE: this module just builds strings. It does NOT validate that the
downstream code (src/nlhe/infoset.py, src/nlhe/solver.py, ...) works
correctly for num_players > 2. The smoke test in
tests/test_6max_game_load.py confirms OpenSpiel can load and walk a
6-max game with random actions — but real training validation comes
later in Phase 4.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PokerGameConfig:
    """Parameters for an OpenSpiel universal_poker game string.

    Defaults reproduce the HUNL Phase 2d game (200bb starting, 50/100 blinds).
    """
    num_players: int = 2
    starting_stack: int = 20000
    big_blind: int = 100
    small_blind: int = 50

    def __post_init__(self) -> None:
        if self.num_players < 2:
            raise ValueError(f"num_players must be >= 2, got {self.num_players}")
        if self.num_players > 10:
            raise ValueError(
                f"num_players must be <= 10 (universal_poker limit), got {self.num_players}"
            )
        if self.starting_stack <= 0:
            raise ValueError(f"starting_stack must be > 0, got {self.starting_stack}")
        if self.big_blind <= 0 or self.small_blind <= 0:
            raise ValueError("blinds must be > 0")
        if self.small_blind >= self.big_blind:
            raise ValueError(
                f"small_blind {self.small_blind} must be < big_blind {self.big_blind}"
            )

    def to_universal_poker_string(self) -> str:
        """Build the OpenSpiel universal_poker game string for this config."""
        n = self.num_players
        # Blind structure: small blind first, then big blind, then 0 for everyone else.
        # Format: "<sb> <bb> 0 0 0 0" for 6-max for example.
        blind_parts = [str(self.small_blind), str(self.big_blind)]
        blind_parts.extend(["0"] * (n - 2))
        blind_str = " ".join(blind_parts)
        # firstPlayer per round: in 6-max, player 3 acts first preflop (UTG),
        # then player 1 (small blind seat) acts first postflop. For HUNL the
        # existing pattern is "2 1 1 1" (BB acts first preflop, SB acts first
        # postflop). For >2 players, universal_poker uses the small-blind seat
        # to act first postflop, and 3 (UTG) preflop.
        if n == 2:
            first_player = "2 1 1 1"
        else:
            # Postflop: small blind acts first. Preflop: UTG (seat 3) acts first.
            first_player = f"3 1 1 1"
        # Stack list: identical for every player.
        stack_str = " ".join([str(self.starting_stack)] * n)
        return (
            f"universal_poker(betting=nolimit,"
            f"numPlayers={n},"
            f"numRounds=4,"
            f"blind={blind_str},"
            f"firstPlayer={first_player},"
            f"numSuits=4,"
            f"numRanks=13,"
            f"numHoleCards=2,"
            f"numBoardCards=0 3 1 1,"
            f"stack={stack_str},"
            f"bettingAbstraction=fullgame)"
        )


# Convenience constructors matching the existing configs.

def hunl_200bb() -> str:
    """The HUNL config from configs/nlhe_200bb.yaml and Phase 2d training."""
    return PokerGameConfig(num_players=2, starting_stack=20000,
                           big_blind=100, small_blind=50).to_universal_poker_string()


def hunl_20bb() -> str:
    """The smoke/phase2b config: 20bb starting (2000 chips, 100/50 blinds)."""
    return PokerGameConfig(num_players=2, starting_stack=2000,
                           big_blind=100, small_blind=50).to_universal_poker_string()


def six_max_200bb() -> str:
    """6-max with 200bb starting stacks (matches HUNL chip scale for comparison)."""
    return PokerGameConfig(num_players=6, starting_stack=20000,
                           big_blind=100, small_blind=50).to_universal_poker_string()


def six_max_sng(starting_stack: int = 1500) -> str:
    """6-max SNG default: 15bb starting (1500 chips, 100/50 blinds).

    Matches typical SNG starting depth where the late game compresses quickly
    to push/fold. Phase 4 training target.
    """
    return PokerGameConfig(num_players=6, starting_stack=starting_stack,
                           big_blind=100, small_blind=50).to_universal_poker_string()
