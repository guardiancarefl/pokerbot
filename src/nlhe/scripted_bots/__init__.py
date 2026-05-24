"""Shanky/WinHoldem profile parser and runtime.

This module loads .txt/.ppl bot-profile files in the Shanky rule language
(https://www.shankybot.com/holdem-bot/strategy-development/) and turns them
into AST objects that can later be executed against game state.

The 36 profiles in the project's archetype set are all in this format. Phase
5 of the architecture (see ARCHITECTURE.md) calls for these to be plugged
in as anonymous training opponents, providing behavioral style diversity
to complement the strength diversity from league play.

This v1 covers parsing only. The runtime evaluator and the Policy adapter
that bridges to OpenSpiel's action space live in sibling modules added in
later commits.
"""
from src.nlhe.scripted_bots.parser import (
    parse_profile,
    Profile,
    Section,
    Rule,
    ParseError,
)

__all__ = [
    "parse_profile",
    "Profile",
    "Section",
    "Rule",
    "ParseError",
]
