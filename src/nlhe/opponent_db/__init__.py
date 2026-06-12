"""Opponent-data pipeline (track H3).

Parses live dry-run session jsonl logs (logs/live_dryrun_*.jsonl) into a
structured per-hand / per-action opponent database. Read-only consumer of
the live path: imports helpers from src.nlhe.integration.scraper_schema
but never modifies live code.

Opponents are ANONYMOUS by project constraint: keyed by
(session_id, seat) only — never by player identity.
"""
from src.nlhe.opponent_db.parser import (  # noqa: F401
    PARSER_VERSION,
    SessionParse,
    HandRecord,
    ActionEvent,
    parse_session_file,
    segment_hands,
)
