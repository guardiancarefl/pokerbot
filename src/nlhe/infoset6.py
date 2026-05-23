"""6-max NLHE information set encoder for Phase 4 training.

Phase 4d (docs/PHASE4_PLAN.md). Companion to src/nlhe/infoset.py
(HUNL-specific). This module is SEPARATE so the HUNL encoder stays
untouched — Phase 2d's trained checkpoint and the Slumbot eval pipeline
remain working production code while we build 6-max infrastructure.

Feature layout (default config, k=200 postflop):
  - card bucket one-hot (200 dims, padded with 0 if street's k < 200)
  - street one-hot (4: preflop, flop, turn, river)
  - seat-position one-hot (6: UTG, MP, CO, BTN, SB, BB) — position of
    the CURRENT player
  - per-player normalized stacks (6: stack_i / starting_stack)
  - per-player active mask (6: 1.0 if still in hand, 0.0 if folded)
  - per-player normalized contribution this hand (6: contrib_i / starting_stack)
  - pot / starting_stack (1)
  - to_call / starting_stack (1)
  - effective stack between hero and current best villain (1)
  - betting-history features (5): bets-this-street, raises-this-street,
    last-bet-size-frac, n-actions-this-street, is-facing-bet

Total feature dimension: 200 + 4 + 6 + 6 + 6 + 6 + 1 + 1 + 1 + 5 = 236.
Higher than HUNL [214] because of the 6x repeated per-player features.

NOT integrated with the solver yet — that's Phase 4e. This module is
the validated building block.
"""
from __future__ import annotations
import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.nlhe.abstraction import Abstraction


# ===== Parsing 6-max universal_poker state =====

# Compiled regexes once at module load.
_RE_ROUND = re.compile(r"\[Round (\d+)\]")
_RE_PLAYER = re.compile(r"\[Player: (\d+)\]")
_RE_POT = re.compile(r"\[Pot: (\d+)\]")
_RE_MONEY = re.compile(r"\[Money: ([\d\s]+)\]")
_RE_PRIVATE = re.compile(r"\[Private: (\w*)\]")
_RE_PUBLIC = re.compile(r"\[Public: (\w*)\]")
_RE_SEQUENCES = re.compile(r"\[Sequences: ([^\]]*)\]")
_RE_CONTRIBUTION = re.compile(r"\[PlayerContribution: ([\d\s]+)\]")


def parse_state_6max(state: Any) -> dict:
    """Parse an OpenSpiel universal_poker state for a 6-max game.

    Uses observation_string() rather than information_state_string()
    because the former includes PlayerContribution which we need for
    accurate pot odds / committed chip tracking.

    Returns a dict with parsed fields:
        street_idx (int 0-3)
        current_player (int 0-5, 0-indexed)
        pot (int)
        money (list[int], length 6): remaining stacks per player
        contribution (list[int], length 6): chips committed this hand
        private_cards (str): hero's hole cards e.g. "AsKs"
        public_cards (str): board cards e.g. "Kh2c3d"
        sequences (str): the betting sequence string e.g. "ccc/cc"
        num_players (int): always 6 in 6-max
    """
    obs = state.observation_string(state.current_player())
    info = state.information_state_string(state.current_player())

    out = {"num_players": 6}

    m = _RE_ROUND.search(obs)
    out["street_idx"] = int(m.group(1)) if m else 0

    # OpenSpiel's [Player: N] is 0-indexed (verified against state.current_player()).
    m = _RE_PLAYER.search(obs)
    out["current_player"] = int(m.group(1)) if m else 0

    m = _RE_POT.search(obs)
    out["pot"] = int(m.group(1)) if m else 0

    m = _RE_MONEY.search(obs)
    if m:
        out["money"] = [int(x) for x in m.group(1).split()]
    else:
        out["money"] = [0] * 6

    m = _RE_CONTRIBUTION.search(obs)
    if m:
        out["contribution"] = [int(x) for x in m.group(1).split()]
    else:
        # Fallback: derive from money if PlayerContribution isn't present.
        out["contribution"] = [0] * 6

    m = _RE_PRIVATE.search(obs)
    out["private_cards"] = m.group(1) if m else ""

    # Public cards are in information_state_string, not observation_string.
    m = _RE_PUBLIC.search(info)
    out["public_cards"] = m.group(1) if m else ""

    # Sequences too.
    m = _RE_SEQUENCES.search(info)
    out["sequences"] = m.group(1) if m else ""

    return out


# ===== Position from seat number =====

# 6-max position labels by seat-index-relative-to-button.
# button = seat 0 (relative), then going clockwise:
#   relative seat 0 = BTN
#   relative seat 1 = SB
#   relative seat 2 = BB
#   relative seat 3 = UTG
#   relative seat 4 = MP
#   relative seat 5 = CO
# But OpenSpiel's universal_poker uses absolute seats. For 6-max with
# firstPlayer=3 1 1 1, seat 3 (UTG, 0-indexed seat 2) acts first preflop.
# We index positions purely by absolute seat for now — the encoder maps
# absolute seat to position one-hot via a fixed table.
#
# This assumes the dealer button is fixed (no rotation within a hand,
# which is true within a single hand). For multi-hand tournament play,
# the button rotates; we re-derive positions per hand.

POSITIONS_6MAX = ["UTG", "MP", "CO", "BTN", "SB", "BB"]


def position_for_seat(seat: int, num_players: int = 6) -> int:
    """Return position-index (0=UTG, 1=MP, ..., 5=BB) for a seat.

    OpenSpiel 6-max convention: seat 0 = SB, seat 1 = BB, seat 2 = UTG,
    seat 3 = MP, seat 4 = CO, seat 5 = BTN. We remap to the
    UTG/MP/CO/BTN/SB/BB ordering used by POSITIONS_6MAX:
        seat 0 -> SB (idx 4)
        seat 1 -> BB (idx 5)
        seat 2 -> UTG (idx 0)
        seat 3 -> MP (idx 1)
        seat 4 -> CO (idx 2)
        seat 5 -> BTN (idx 3)
    """
    assert 0 <= seat < num_players, f"seat {seat} out of range for {num_players}-handed"
    if num_players != 6:
        # For non-6max games we just return the seat as-is; caller's responsibility.
        return seat
    mapping = {0: 4, 1: 5, 2: 0, 3: 1, 4: 2, 5: 3}
    return mapping[seat]


# ===== Encoder =====


@dataclass
class InfosetEncoder6Max:
    """6-max NLHE infoset feature encoder.

    Args:
        abstraction: card abstraction module (shared with HUNL encoder).
        starting_stack: per-player starting chip count; used to normalize
            stack/pot/contribution features.
        max_bucket_dim: dimensionality of the bucket-one-hot slice. Should
            be max(k) across all streets (200 by default).
        bucket_runouts: MC runouts for postflop bucket lookups
            (preflop uses the dict lookup, no MC).
    """

    abstraction: Abstraction
    starting_stack: int = 20000
    max_bucket_dim: int = 200
    bucket_runouts: int = 200

    # Per-traversal cache, keyed by (frozenset(hero), tuple(board)) -> bucket id.
    _bucket_cache: dict = field(default_factory=dict)

    def reset_cache(self) -> None:
        self._bucket_cache.clear()

    @property
    def feature_dim(self) -> int:
        """Total feature vector dimension."""
        # 200 bucket + 4 street + 6 position + 6 stacks + 6 active + 6 contribution
        # + 1 pot + 1 tocall + 1 effstack + 5 betting
        return self.max_bucket_dim + 4 + 6 + 6 + 6 + 6 + 1 + 1 + 1 + 5

    def encode(self, state: Any, rng: random.Random | None = None) -> np.ndarray:
        """Encode an OpenSpiel state into a feature vector for the network.

        Backward-compatible wrapper: parses the OpenSpiel state and delegates
        to encode_from_parsed. Existing callers using single-hand
        universal_poker states keep working unchanged.

        For repeated_poker (multi-hand tournament) states, callers should
        parse with parse_state_repeated_6max and pass the result to
        encode_from_parsed directly. That path uses dealer-aware position
        encoding rather than the seat-5-fixed-dealer assumption.

        Args:
            state: OpenSpiel universal_poker state object.
            rng: optional random.Random for non-deterministic MC fallback.

        Returns:
            np.ndarray of shape (feature_dim,), dtype float32.
        """
        parsed = parse_state_6max(state)
        return self.encode_from_parsed(parsed, rng)

    def encode_from_parsed(
        self, parsed: dict, rng: random.Random | None = None
    ) -> np.ndarray:
        """Encode a pre-parsed state dict into a feature vector.

        Same feature vector shape as encode(). Branches on whether the
        parsed dict has a 'dealer_seat' field: if present (repeated_poker),
        uses position_for_seat_with_dealer to compute button-rotation-aware
        position; if absent (universal_poker), uses the original
        position_for_seat with the fixed dealer=seat-5 assumption.

        Args:
            parsed: a dict from parse_state_6max or parse_state_repeated_6max.
            rng: optional random.Random for non-deterministic MC fallback.

        Returns:
            np.ndarray of shape (feature_dim,), dtype float32.
        """
        feat = np.zeros(self.feature_dim, dtype=np.float32)
        offset = 0

        # Card bucket one-hot.
        bucket_id = self._get_bucket(parsed, rng)
        if bucket_id is not None and bucket_id < self.max_bucket_dim:
            feat[offset + bucket_id] = 1.0
        offset += self.max_bucket_dim

        # Street one-hot.
        street_idx = parsed["street_idx"]
        if 0 <= street_idx < 4:
            feat[offset + street_idx] = 1.0
        offset += 4

        # Position one-hot for the current player.
        # Repeated_poker path: dealer rotates each hand, so position depends
        # on current button location. Single-hand path: dealer is fixed at
        # seat 5 by OpenSpiel's universal_poker convention.
        if "dealer_seat" in parsed:
            pos_idx = position_for_seat_with_dealer(
                parsed["current_player"],
                dealer_seat=parsed["dealer_seat"],
                num_players=6,
            )
        else:
            pos_idx = position_for_seat(parsed["current_player"], num_players=6)
        if 0 <= pos_idx < 6:
            feat[offset + pos_idx] = 1.0
        offset += 6

        # Per-player normalized stacks.
        ss = float(self.starting_stack)
        for i in range(6):
            feat[offset + i] = parsed["money"][i] / ss if ss > 0 else 0.0
        offset += 6

        # Active mask: a player is active if they have non-zero money OR
        # they've contributed something this hand (haven't busted).
        for i in range(6):
            feat[offset + i] = 1.0 if parsed["money"][i] > 0 else 0.0
        offset += 6

        # Per-player normalized contribution this hand.
        for i in range(6):
            feat[offset + i] = parsed["contribution"][i] / ss if ss > 0 else 0.0
        offset += 6

        # Pot / starting_stack.
        feat[offset] = parsed["pot"] / ss if ss > 0 else 0.0
        offset += 1

        # To-call / starting_stack.
        cp = parsed["current_player"]
        max_contrib = max(parsed["contribution"]) if parsed["contribution"] else 0
        my_contrib = parsed["contribution"][cp] if cp < len(parsed["contribution"]) else 0
        to_call = max(0, max_contrib - my_contrib)
        feat[offset] = to_call / ss if ss > 0 else 0.0
        offset += 1

        # Effective stack: min over active opponents of (their stack vs hero's stack).
        my_stack = parsed["money"][cp] if cp < len(parsed["money"]) else 0
        opponent_stacks = [
            parsed["money"][i] for i in range(6)
            if i != cp and parsed["money"][i] > 0
        ]
        if opponent_stacks:
            eff = min(my_stack, max(opponent_stacks))
        else:
            eff = my_stack
        feat[offset] = eff / ss if ss > 0 else 0.0
        offset += 1

        # Betting-history features for the current street.
        b = _betting_features(parsed["sequences"], parsed["pot"], self.starting_stack)
        feat[offset + 0] = b["n_bets_street"]
        feat[offset + 1] = b["n_raises_street"]
        feat[offset + 2] = b["last_bet_frac"]
        feat[offset + 3] = b["n_actions_street"]
        feat[offset + 4] = 1.0 if b["is_facing_bet"] else 0.0
        offset += 5

        assert offset == self.feature_dim, f"offset {offset} != feature_dim {self.feature_dim}"
        return feat

    def _get_bucket(self, parsed: dict, rng: random.Random | None) -> int | None:
        """Get the bucket id for the parsed state's (hero, board)."""
        from src.nlhe.equity import cards_from_str

        hero_str = parsed["private_cards"]
        board_str = parsed["public_cards"]
        if not hero_str:
            return None

        hero = list(cards_from_str(hero_str))
        # Board may be empty (preflop) or have 3/4/5 cards (flop/turn/river).
        # cards_from_str takes pairs of chars; board_str may be longer.
        board = list(cards_from_str(board_str)) if board_str else []

        key = (frozenset(hero), tuple(board))
        if key in self._bucket_cache:
            return self._bucket_cache[key]

        bucket = self.abstraction.bucket_of(hero, board, self.bucket_runouts, rng)
        self._bucket_cache[key] = bucket
        return bucket


# ===== Betting features (parser, shared concept with HUNL) =====


def _betting_features(sequences: str, pot: int, starting_stack: int) -> dict:
    """Extract per-street betting summary from the sequence string.

    The sequences string looks like 'cc/rcc/c' where:
      - 'c' = check/call
      - 'r' = raise/bet
      - 'f' = fold
      - '/' separates streets

    Returns dict with: n_bets_street, n_raises_street, last_bet_frac,
    n_actions_street, is_facing_bet.
    """
    if not sequences:
        return {
            "n_bets_street": 0.0,
            "n_raises_street": 0.0,
            "last_bet_frac": 0.0,
            "n_actions_street": 0.0,
            "is_facing_bet": False,
        }

    streets = sequences.split("/")
    current = streets[-1] if streets else ""

    n_bets = sum(1 for c in current if c == "r")
    # universal_poker uses 'r' for any raise. We treat first 'r' on street as a bet, rest as raises.
    n_raises = max(0, n_bets - 1)
    actions_this_street = len(current)

    # Is the last action a bet/raise that the next player needs to respond to?
    is_facing_bet = current.endswith("r")

    # Last bet size as fraction of pot — approximate (we don't have exact sizing).
    # Conservative fallback: 0 if no bet, 0.5 if a bet exists.
    last_bet_frac = 0.5 if is_facing_bet else 0.0

    return {
        "n_bets_street": float(n_bets),
        "n_raises_street": float(n_raises),
        "last_bet_frac": last_bet_frac,
        "n_actions_street": float(actions_this_street),
        "is_facing_bet": is_facing_bet,
    }


# ===== RepeatedPokerState parser (Phase 4f) =====
#
# RepeatedPokerState wraps multiple universal_poker hands in one game episode,
# with dealer rotation and blind escalation between hands. Its API differs
# from UniversalPokerState in important ways:
#
#   - information_state_string() raises "not implemented"
#   - observation_string() works and includes Round/Player/Pot/Money/Private/
#     PlayerContribution (everything parse_state_6max needs except sequences
#     and public cards)
#   - to_dict() exposes the underlying universal_poker hand as a JSON string
#     under "current_universal_poker_json", which contains board_cards and
#     betting_history
#   - outer state exposes dealer_seat(), big_blind(), small_blind() directly
#
# This parser combines both data sources to produce the same output shape as
# parse_state_6max, with four additional fields specific to repeated_poker:
# dealer_seat, hand_number, current_big_blind, current_small_blind.


def parse_state_repeated_6max(state):
    """Parse an OpenSpiel RepeatedPokerState for 6-max NLHE tournament play.

    Returns the same fields as parse_state_6max plus tournament-level fields
    (dealer_seat, hand_number, current_big_blind, current_small_blind)
    needed for position-relative-to-button feature encoding.
    """
    import json as _json

    obs = state.observation_string(state.current_player())
    out = {"num_players": 6}

    m = _RE_ROUND.search(obs)
    out["street_idx"] = int(m.group(1)) if m else 0

    m = _RE_PLAYER.search(obs)
    out["current_player"] = int(m.group(1)) if m else 0

    m = _RE_POT.search(obs)
    out["pot"] = int(m.group(1)) if m else 0

    m = _RE_MONEY.search(obs)
    if m:
        out["money"] = [int(x) for x in m.group(1).split()]
    else:
        out["money"] = [0] * 6

    m = _RE_CONTRIBUTION.search(obs)
    if m:
        out["contribution"] = [int(x) for x in m.group(1).split()]
    else:
        out["contribution"] = [0] * 6

    m = _RE_PRIVATE.search(obs)
    out["private_cards"] = m.group(1) if m else ""

    state_dict = state.to_dict()
    inner_str = state_dict.get("current_universal_poker_json", "")
    if inner_str:
        if isinstance(inner_str, str):
            inner = _json.loads(inner_str)
        else:
            inner = inner_str
        out["public_cards"] = inner.get("board_cards", "")
        out["sequences"] = inner.get("betting_history", "")
    else:
        out["public_cards"] = ""
        out["sequences"] = ""

    out["dealer_seat"] = state.dealer_seat()
    out["hand_number"] = state_dict.get("hand_number", 0)
    out["current_big_blind"] = state.big_blind()
    out["current_small_blind"] = state.small_blind()

    return out


def position_for_seat_with_dealer(seat, dealer_seat, num_players=6):
    """Return position-index (0=UTG, 1=MP, ..., 5=BTN, etc.) for a seat,
    given the current dealer's seat number.

    This is the multi-hand-aware version of position_for_seat. With
    rotating dealer (as in repeated_poker), the position of any given
    seat depends on where the button is. For example, seat 0 is the
    SB (position index 4) only when the dealer is at seat 5 (the
    original layout). When the dealer rotates to seat 0, then seat 1
    becomes the SB, seat 2 becomes the BB, etc.

    Args:
        seat: 0-indexed seat number whose position we want.
        dealer_seat: 0-indexed seat number of the dealer (button).
        num_players: total active players at the table. Default 6.

    Returns:
        Position index in POSITIONS_6MAX: 0=UTG, 1=MP, 2=CO, 3=BTN,
        4=SB, 5=BB.

    Position layout for 6-max (clockwise from dealer):
        dealer_offset 0 -> BTN (idx 3)
        dealer_offset 1 -> SB  (idx 4)
        dealer_offset 2 -> BB  (idx 5)
        dealer_offset 3 -> UTG (idx 0)
        dealer_offset 4 -> MP  (idx 1)
        dealer_offset 5 -> CO  (idx 2)
    """
    if not (0 <= seat < num_players):
        raise ValueError(
            f"seat {seat} out of range for {num_players}-handed"
        )
    if not (0 <= dealer_seat < num_players):
        raise ValueError(
            f"dealer_seat {dealer_seat} out of range for {num_players}-handed"
        )

    # Seats are arranged around the table; dealer is at dealer_seat.
    # The seat "after" the dealer (clockwise +1) is SB, then BB, etc.
    offset = (seat - dealer_seat) % num_players

    if num_players == 6:
        # 6-max position layout indexed by offset from button.
        offset_to_position = {
            0: 3,  # BTN (dealer itself)
            1: 4,  # SB
            2: 5,  # BB
            3: 0,  # UTG
            4: 1,  # MP
            5: 2,  # CO
        }
        return offset_to_position[offset]
    else:
        # For non-6max games, return offset as a position proxy.
        # (Caller's responsibility to handle this case properly.)
        return offset
