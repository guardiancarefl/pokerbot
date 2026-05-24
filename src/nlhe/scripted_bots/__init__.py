"""Shanky/WinHoldem profile parser and runtime.

This module loads .txt/.ppl bot-profile files in the Shanky rule language
(https://www.shankybot.com/holdem-bot/strategy-development/) and turns them
into AST objects that can later be executed against game state.

The 36 profiles in the project's archetype set are all in this format. Phase
5 of the architecture (see ARCHITECTURE.md) calls for these to be plugged
in as anonymous training opponents, providing behavioral style diversity
to complement the strength diversity from league play.

Two modules:
  - `parser`: source text → Profile AST. Pure parsing.
  - `runtime`: Profile + GameContext → Action. Predicate evaluation.

The third module (the Policy adapter that bridges Action AST to the discrete
action abstraction used by eval_pool / LeaguePool / OpenSpiel) is added
in a subsequent commit.
"""
from src.nlhe.scripted_bots.parser import (
    parse_profile,
    Profile,
    Section,
    Rule,
    Action,
    ActionKind,
    ParseError,
)
from src.nlhe.scripted_bots.runtime import (
    GameContext,
    Runtime,
    HandFeatures,
    BoardFeatures,
    evaluate_profile,
    compute_hand_features,
    compute_board_features,
    matches_hand_spec,
    matches_board_spec,
    card_to_int,
    int_to_rank_suit,
    card_rank,
    card_suit,
    UnsupportedPredicateError,
)

__all__ = [
    # Parser
    "parse_profile",
    "Profile",
    "Section",
    "Rule",
    "Action",
    "ActionKind",
    "ParseError",
    # Runtime
    "GameContext",
    "Runtime",
    "HandFeatures",
    "BoardFeatures",
    "evaluate_profile",
    "compute_hand_features",
    "compute_board_features",
    "matches_hand_spec",
    "matches_board_spec",
    "card_to_int",
    "int_to_rank_suit",
    "card_rank",
    "card_suit",
    "UnsupportedPredicateError",
]
