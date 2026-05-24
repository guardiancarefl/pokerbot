"""Shanky/WinHoldem profile parser, runtime, and policy adapter.

This module loads .txt profile files in the Shanky rule language and
provides the full chain from source text to drop-in Policy:

  .txt source --[parser]--> Profile AST --[runtime]--> Action
                                                          |
                                                  [policy adapter]
                                                          |
                                                          v
                                          OpenSpiel chip-action int
                                          (matches Policy protocol)

The 36 profiles in the project's archetype set are all supported. Phase
5 of the architecture (see ARCHITECTURE.md) calls for these to be plugged
in as anonymous training opponents, providing behavioral style diversity
to complement the strength diversity from league play.

Three submodules:
  - `parser`: source text → Profile AST. Pure parsing.
  - `runtime`: Profile + GameContext → Action. Predicate evaluation.
  - `policy`: Profile path → Policy (eval_pool / LeaguePool compatible).

The policy module bridges to the existing Policy protocol used by
eval_pool.py and LeaguePool, so a Shanky profile can be used wherever
a CheckpointPolicy is currently used.
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
from src.nlhe.scripted_bots.policy import (
    ShankyProfilePolicy,
    build_game_context,
    shanky_action_to_discrete,
    load_shanky_policies_from_dir,
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
    # Policy adapter
    "ShankyProfilePolicy",
    "build_game_context",
    "shanky_action_to_discrete",
    "load_shanky_policies_from_dir",
]
