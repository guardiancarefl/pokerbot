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

Phase 2 prerequisite — empirically verified 6-max hole-card deal order:
universal_poker deals all of seat 0's hole cards (both), then all of
seat 1's, ..., then all of seat 5's. Same convention as HUNL
(documented in src/nlhe/policy_adapter.py:464). Verified by
scripts/probe_six_max_deal_order.py; codified in DEAL_ORDER_SEQUENCE
below for use by the Phase 2 forced-card dealer.
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


# OpenSpiel universal_poker 6-max hole-card deal order, verified empirically
# by scripts/probe_six_max_deal_order.py: sequential per seat — seat 0 gets
# both cards, then seat 1 gets both, ..., then seat 5 gets both. Each tuple
# is (seat_idx, card_position) for slots 0..11. The tuple at slot N tells
# the Phase 2 forced-card dealer which (seat, card-slot) to fill when
# OpenSpiel asks for chance action N.
DEAL_ORDER_SEQUENCE: tuple[tuple[int, int], ...] = tuple(
    (seat, card) for seat in range(NUM_SEATS) for card in range(2)
)
# Sanity: 12 hole cards total for 6 seats × 2 cards
assert len(DEAL_ORDER_SEQUENCE) == NUM_SEATS * 2


# --------------------------------------------------------------------------
# Forced-card dealer (Phase 2 Piece 3)
# --------------------------------------------------------------------------
#
# OpenSpiel's universal_poker engine deals hole + board cards via chance
# nodes; left to its own RNG it picks uniform at random. For our replay we
# need DETERMINISTIC card placements:
#   - hero's hole cards: known from scraper, force-deal at hero's slots
#   - opponent hole cards: hidden — pick any plausible card not on the
#     hero's or board's "forbidden" set
#   - board cards: known from scraper as the hand progresses, force-deal
#     at the right chance node when board cards open
#
# These helpers port the proven HUNL pattern from src/nlhe/policy_adapter.py
# (`pick_deck_action`, `_deal_one_card`) to 6-max, indexed via the verified
# DEAL_ORDER_SEQUENCE above.

import re as _re

_DEAL_RE = _re.compile(r"\bDeal\s+([2-9TJQKA][cdhs])\b")
_PRIVATE_RE_REPLAY = _re.compile(r"\[Private:\s+([^\]]*)\]")
_PUBLIC_RE_REPLAY = _re.compile(r"\[Public:\s+([^\]]*)\]")


def _extract_card_from_action_string(s: str) -> str:
    """Pull '2c' out of OpenSpiel chance action_to_string output like
    'player=-1 move=Deal 2c'. Raises ValueError if no card found."""
    m = _DEAL_RE.search(s)
    if not m:
        raise ValueError(f"no deal-card pattern in action string: {s!r}")
    return m.group(1)


def pick_deck_action(state, predicate, purpose: str) -> int:
    """Scan a chance node's legal actions; return the first int whose card
    satisfies predicate. Raises ReplayError if nothing matches.

    Ported from src/nlhe/policy_adapter.py:59 (HUNL version); semantics
    identical, just lifted to the 6-max replay module.
    """
    legal = state.legal_actions()
    available: list[str] = []
    for a in legal:
        try:
            card = _extract_card_from_action_string(state.action_to_string(a))
        except ValueError:
            continue
        available.append(card)
        if predicate(card):
            return int(a)
    preview = ", ".join(available[:8]) + (" ..." if len(available) > 8 else "")
    raise ReplayError(
        f"pick_deck_action({purpose}): no legal card satisfied predicate. "
        f"Available in deck ({len(available)} cards): [{preview}]"
    )


def _private_cards_for(state, seat: int) -> str:
    """Read OpenSpiel's [Private: XXXX] field for the given seat."""
    info = state.information_state_string(seat)
    m = _PRIVATE_RE_REPLAY.search(info)
    return m.group(1) if m else ""


def _public_cards(state) -> str:
    """Read OpenSpiel's [Public: XXXX] field (public info; either seat works)."""
    try:
        info = state.information_state_string(0)
    except Exception:
        # At chance nodes universal_poker may reject info_state for some
        # observers. observation_string is the safe fallback.
        info = state.observation_string(0)
    m = _PUBLIC_RE_REPLAY.search(info)
    return m.group(1) if m else ""


def deal_one_card_6max(state, hero_seat: int,
                        hero_cards: tuple[str, ...],
                        target_board: tuple[str, ...]) -> None:
    """At a chance node, apply the appropriate card-dealing action.

    Universal_poker 6-max deals hole cards per DEAL_ORDER_SEQUENCE (seat 0
    both, seat 1 both, ..., seat 5 both), then board cards one at a time
    at street boundaries (3 on flop, 1 on turn, 1 on river).

    For hero's hole-card slots: place the next hero card not yet placed.
    For opponent hole-card slots: place an ARBITRARY card not in
        forbidden = hero_cards ∪ target_board (avoids consuming a card
        we'll need later).
    For board-card slots: place the next target_board card not yet placed.
    """
    # Determine where we are: count cards already dealt to each seat
    # + how many board cards are out.
    per_seat_counts = [len(_private_cards_for(state, p)) // 2
                       for p in range(NUM_SEATS)]
    board_count = len(_public_cards(state)) // 2
    total_hole_dealt = sum(per_seat_counts)

    if total_hole_dealt < len(DEAL_ORDER_SEQUENCE):
        # Still dealing hole cards; the next slot is DEAL_ORDER_SEQUENCE[total_hole_dealt]
        target_seat, _target_card_pos = DEAL_ORDER_SEQUENCE[total_hole_dealt]
        if target_seat == hero_seat:
            already = set(_cards_in_private(
                _private_cards_for(state, hero_seat)))
            remaining = [c for c in hero_cards if c not in already]
            if not remaining:
                raise ReplayError(
                    f"hero hole-card slot but all hero_cards already placed: "
                    f"hero_priv_now={_private_cards_for(state, hero_seat)!r} "
                    f"hero_cards={hero_cards}")
            target_card = remaining[0]
            action = pick_deck_action(
                state,
                lambda c, t=target_card: c == t,
                f"hero (seat={hero_seat}) hole card {target_card!r}",
            )
        else:
            # Opponent hole card — pick any card NOT reserved for hero or
            # board. Multiple replay calls in one hand: also exclude cards
            # already dealt to OTHER opponents so we don't risk double-deal
            # if pick_deck_action's predicate is greedy. universal_poker's
            # legal_actions naturally excludes already-dealt cards, but the
            # forbidden set is belt-and-suspenders.
            already_dealt: set[str] = set()
            for p in range(NUM_SEATS):
                priv = _private_cards_for(state, p)
                already_dealt.update(_cards_in_private(priv))
            already_dealt.update(_cards_in_public(_public_cards(state)))
            forbidden = (set(hero_cards) | set(target_board)
                          | already_dealt)
            action = pick_deck_action(
                state,
                lambda c, fb=forbidden: c not in fb,
                f"opponent (seat={target_seat}) hole card "
                f"(forbidden={sorted(forbidden)})",
            )
    else:
        # Board card phase. board_count tells us which board card is next.
        if board_count >= len(target_board):
            raise ReplayError(
                f"chance node past hole cards but target_board exhausted: "
                f"board_count={board_count} target_board={target_board}")
        next_board = target_board[board_count]
        action = pick_deck_action(
            state,
            lambda c, t=next_board: c == t,
            f"board card #{board_count + 1}: {next_board!r}",
        )
    state.apply_action(int(action))


def _cards_in_private(priv: str) -> list[str]:
    """Split concatenated private string ('2d2c') into ['2d', '2c']."""
    return [priv[i:i + 2] for i in range(0, len(priv), 2)]


def _cards_in_public(pub: str) -> list[str]:
    """Same split for board string."""
    return [pub[i:i + 2] for i in range(0, len(pub), 2)]


# --------------------------------------------------------------------------
# Full mid-hand replay engine (Phase 2 Piece 4)
# --------------------------------------------------------------------------


@dataclass
class MidHandState:
    """Result of a successful mid-hand replay.

    Extends HandStartState with bookkeeping needed by the generalised
    invariant check (Piece 5): per-seat prior-streets cumulative
    commitment (used to convert OpenSpiel total-commit to scraper
    current-street bet during the diff).
    """
    state: Any                          # OpenSpiel state at hero's decision
    game_str: str
    pre_hand_stacks: tuple[int, ...]
    blind_level: BlindLevel
    sb_seat: int
    bb_seat: int
    n_alive: int
    n_actions_applied: int              # for diagnostics
    final_current_player: int           # should == hero_seat


def replay_to_decision(frame, structure):
    """Walk OpenSpiel from new_initial_state() to the hero's current decision.

    Combines Pieces 1, 2, 3:
      - Piece 1 (DEAL_ORDER_SEQUENCE): drives the dealer's slot lookup
      - Piece 2 (derive_action_sequence): canonical action sequence
      - Piece 3 (deal_one_card_6max): forced chance handling

    Step-by-step:
      1. Build pre-hand stacks via the simple-model chip-conservation
         helper from scraper_schema (same one derive_action_sequence uses).
      2. Build game string via to_inner_game_string_for_state with those
         pre-hand stacks + dealer.
      3. Load + new_initial_state() -> at the first hole-card chance node.
      4. Derive the action sequence via derive_action_sequence(frame).
      5. Walk: at each loop step, if state.is_chance_node() apply a
         forced deal (hole or board card as appropriate); otherwise apply
         the next action int from the sequence. Continue until either
         (a) the action sequence is exhausted AND state.current_player()
         == hero_seat (SUCCESS) or (b) we hit a step that can't be
         applied (ReplayError).
      6. Return a MidHandState ready for the generalised invariant.

    Raises ReplayError if any step inconsistency: action not in legal_actions,
    state goes terminal before reaching hero, hero_seat mismatch at end, etc.
    Callers should treat ReplayError as a soft-drop (ScraperDataQuality
    equivalent for replay-time failures).
    """
    try:
        # Lazy imports — keep top of module light + tolerate /tmp paths.
        from src.nlhe.integration.scraper_schema import (
            _derive_pre_hand_simple_model, derive_action_sequence,
            ActionDerivationError,
        )
    except ImportError:  # pragma: no cover (only the /tmp dev shim path)
        from scraper_schema import (  # type: ignore
            _derive_pre_hand_simple_model, derive_action_sequence,
            ActionDerivationError,
        )

    alive_seats = [i for i in range(NUM_SEATS) if frame.alive[i]]
    n_alive = len(alive_seats)
    if n_alive < 2:
        raise ReplayError(f"n_alive={n_alive} < 2; no hand possible")

    # Match SB/BB by the same alive-seat rotation the library uses
    sb_seat, bb_seat = _sb_bb_seats(frame.dealer_seat, alive_seats)

    blind_level = _find_blind_level(structure, frame.blinds)

    # Pre-hand stacks via the simple-model chip-conservation helper
    pre = _derive_pre_hand_simple_model(frame, sb_seat, bb_seat)
    stacks = list(pre)

    # Game string + initial state
    try:
        game_str = structure.to_inner_game_string_for_state(
            blind_level=blind_level,
            stacks=stacks,
            dealer_seat=frame.dealer_seat,
        )
        game = pyspiel.load_game(game_str)
        state = game.new_initial_state()
    except Exception as e:
        raise ReplayError(
            f"OpenSpiel load_game / new_initial_state failed: {e}\n"
            f"game_str (first 300): {game_str[:300] if 'game_str' in locals() else 'N/A'}"
        )

    # Derive canonical action sequence
    try:
        action_seq = derive_action_sequence(frame)
    except ActionDerivationError as e:
        raise ReplayError(f"action sequence derivation failed: {e}")

    # Walk: interleave chance handling with action application
    action_idx = 0
    max_steps = 200  # safety cap (12 holes + 5 board + ~50 actions << 200)
    n_actions_applied = 0
    for _step in range(max_steps):
        if state.is_terminal():
            raise ReplayError(
                f"state went terminal before hero's decision; "
                f"action_idx={action_idx}/{len(action_seq)} "
                f"n_actions_applied={n_actions_applied}"
            )
        if state.is_chance_node():
            try:
                deal_one_card_6max(
                    state, frame.hero_seat,
                    frame.hero_cards, frame.board,
                )
            except ReplayError:
                raise  # propagate
            except Exception as e:
                raise ReplayError(f"chance deal failed: {e}")
            continue
        # Decision node — check if we've reached hero's decision
        if action_idx >= len(action_seq):
            # Sequence exhausted; verify we landed at hero
            if state.current_player() == frame.hero_seat:
                return MidHandState(
                    state=state,
                    game_str=game_str,
                    pre_hand_stacks=tuple(stacks),
                    blind_level=blind_level,
                    sb_seat=sb_seat,
                    bb_seat=bb_seat,
                    n_alive=n_alive,
                    n_actions_applied=n_actions_applied,
                    final_current_player=state.current_player(),
                )
            raise ReplayError(
                f"action sequence exhausted but current_player="
                f"{state.current_player()}, expected hero_seat="
                f"{frame.hero_seat}; sequence under-emitted"
            )
        # Apply next action — verify seat matches expected
        expected_seat, chip_int = action_seq[action_idx]
        actual_seat = state.current_player()
        if actual_seat != expected_seat:
            raise ReplayError(
                f"action_seq[{action_idx}] expects seat {expected_seat} "
                f"but state.current_player()={actual_seat}; "
                f"action_seq mismatched OpenSpiel's action order"
            )
        legal = state.legal_actions()
        if chip_int not in legal:
            raise ReplayError(
                f"action_seq[{action_idx}]=(seat={expected_seat}, "
                f"chip_int={chip_int}) not in legal_actions={legal[:10]}"
                f"{' ...' if len(legal) > 10 else ''}; "
                f"derive_action_sequence emitted an illegal action"
            )
        try:
            state.apply_action(int(chip_int))
        except Exception as e:
            raise ReplayError(f"apply_action failed: {e}")
        action_idx += 1
        n_actions_applied += 1
    raise ReplayError(
        f"replay exceeded max_steps={max_steps}; "
        f"action_idx={action_idx}/{len(action_seq)} "
        f"likely infinite chance loop"
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
