"""Hand-start replay engine for Option B integration.

Phase 1 scope: given a ScraperFrame that satisfies is_hand_start (board
empty, only blinds + antes posted), reconstruct the corresponding OpenSpiel
universal_poker state. Verify the reconstruction matches the scraper's
observable fields via strict invariant check.

Out of Phase 1 scope (deferred until corpus verification of hand-start
replay passes):
  - Forced hole-card dealing
  - Forced board dealing
  - Action-history replay across multiple frames

The hand-start state is the simplest reconstruction target: state is at the
INITIAL chance node (no cards dealt yet) immediately after blinds + antes
have been posted via the inflated_big_blind encoding. The scraper's
observable fields (pot, per-seat stack, per-seat bet, dealer position) all
have known equivalents at this state, modulo OpenSpiel's ante-inflation
convention which we explicitly convert.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pyspiel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent if "src/nlhe/integration" in str(Path(__file__).resolve()) else Path(__file__).resolve().parent.parent))

# Local imports — module is at src/nlhe/integration/replay.py when installed
try:
    from src.nlhe.game_strings import TournamentStructure, BlindLevel
    from src.nlhe.infoset6 import parse_state_6max
    from src.nlhe.integration.scraper_schema import (
        ScraperFrame, BlindsLevel, pre_hand_stacks, is_hand_start,
        NUM_SEATS,
    )
except ImportError:
    # /tmp dev path — import from /tmp
    sys.path.insert(0, "/tmp")
    from src.nlhe.game_strings import TournamentStructure, BlindLevel
    from src.nlhe.infoset6 import parse_state_6max
    from scraper_schema import (
        ScraperFrame, BlindsLevel, pre_hand_stacks, is_hand_start,
        NUM_SEATS,
    )


class ReplayError(Exception):
    """Replay failed for a non-recoverable reason — caller must safe-fallback."""


@dataclass
class HandStartState:
    """Result of a successful hand-start replay."""
    state: Any                          # the OpenSpiel pyspiel.State
    game_str: str                       # game string used to load
    pre_hand_stacks: tuple[int, ...]    # length 6 — what we passed in
    blind_level: BlindLevel             # the matched schedule entry
    sb_seat: int                        # 0..5 — alive-seat-rotation SB position
    bb_seat: int                        # 0..5 — alive-seat-rotation BB position
    n_alive: int                        # number of alive seats


def _find_blind_level(structure: TournamentStructure,
                       blinds: BlindsLevel) -> BlindLevel:
    """Find the BlindLevel entry matching the scraper's parsed (SB, BB, ante).

    Strict match required — no fuzzing. The blind schedule is small (17
    levels for the turbo YAML) so a linear scan is fine.

    Raises ReplayError if no match.
    """
    for bl in structure.blind_schedule:
        if (bl.small_blind == blinds.sb and
                bl.big_blind == blinds.bb and
                bl.ante == blinds.ante):
            return bl
    sched_repr = [(bl.level, bl.small_blind, bl.big_blind, bl.ante)
                  for bl in structure.blind_schedule]
    raise ReplayError(
        f"no blind_schedule entry matches scraper-parsed (SB={blinds.sb}, "
        f"BB={blinds.bb}, ante={blinds.ante}). schedule has: {sched_repr[:5]}..."
    )


def _sb_bb_seats(dealer_seat: int, alive_seats: list[int]
                 ) -> tuple[int, int]:
    """The SB and BB positions per to_inner_game_string_for_state's
    alive-seat rotation (game_strings.py:397-404):
        sb_seat = alive_seats[(dealer_pos + 1) % n_alive]
        bb_seat = alive_seats[(dealer_pos + 2) % n_alive]
    """
    n_alive = len(alive_seats)
    if dealer_seat not in alive_seats:
        raise ReplayError(
            f"dealer_seat {dealer_seat} not in alive seats {alive_seats}"
        )
    if n_alive < 2:
        raise ReplayError(
            f"need >= 2 alive seats, got {n_alive}"
        )
    dpos = alive_seats.index(dealer_seat)
    sb_idx = alive_seats[(dpos + 1) % n_alive]
    bb_idx = alive_seats[(dpos + 2) % n_alive]
    return sb_idx, bb_idx


def replay_hand_start(frame: ScraperFrame, structure: TournamentStructure
                      ) -> HandStartState:
    """Reconstruct the OpenSpiel hand-start state from a parsed ScraperFrame.

    Pre-conditions (raises ReplayError otherwise):
      - frame must be a hand-start (is_hand_start(frame) == True)
      - the parsed (SB, BB, ante) must match an entry in structure.blind_schedule
      - >= 2 alive seats (else no hand can begin)
      - dealer_seat must be alive

    Post-condition: returned HandStartState.state is at a chance node (about
    to deal hole cards), with blinds + antes already encoded in the per-seat
    contribution via OpenSpiel's inflated_big_blind convention.
    """
    if not is_hand_start(frame):
        raise ReplayError(
            "frame is not a hand-start (board non-empty or non-blind bets present)"
        )

    blind_level = _find_blind_level(structure, frame.blinds)

    # Pre-hand stacks: what each seat had at the table before blinds + antes.
    stacks = list(pre_hand_stacks(frame))
    n_alive = sum(1 for x in frame.alive if x)
    alive_seats = [i for i in range(NUM_SEATS) if frame.alive[i]]

    if n_alive < 2:
        raise ReplayError(f"need >= 2 alive seats, got {n_alive}")

    sb_seat, bb_seat = _sb_bb_seats(frame.dealer_seat, alive_seats)

    # Build the game string. to_inner_game_string_for_state requires:
    #   - all alive-seat stacks >= bb_inflated (since those seats can post a blind)
    #   - dealer_seat is alive
    inflated = blind_level.inflated_big_blind(NUM_SEATS)
    if stacks[bb_seat] < inflated:
        raise ReplayError(
            f"BB seat {bb_seat} has pre-hand stack {stacks[bb_seat]} < "
            f"inflated_bb {inflated}; cannot post (real game would auto-allin)"
        )
    if stacks[sb_seat] < blind_level.small_blind:
        raise ReplayError(
            f"SB seat {sb_seat} has pre-hand stack {stacks[sb_seat]} < "
            f"sb {blind_level.small_blind}; cannot post"
        )

    game_str = structure.to_inner_game_string_for_state(
        blind_level=blind_level,
        stacks=stacks,
        dealer_seat=frame.dealer_seat,
    )

    try:
        game = pyspiel.load_game(game_str)
        state = game.new_initial_state()
    except Exception as e:
        raise ReplayError(
            f"OpenSpiel load_game / new_initial_state failed: {e}\n"
            f"game_str (first 300 chars): {game_str[:300]}"
        )

    # The initial state should be at a chance node (hole-card dealing).
    if not state.is_chance_node():
        raise ReplayError(
            f"expected chance node at initial state, got "
            f"current_player={state.current_player()}, is_terminal={state.is_terminal()}"
        )

    return HandStartState(
        state=state,
        game_str=game_str,
        pre_hand_stacks=tuple(stacks),
        blind_level=blind_level,
        sb_seat=sb_seat,
        bb_seat=bb_seat,
        n_alive=n_alive,
    )
