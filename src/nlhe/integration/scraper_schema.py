"""Scraper-JSON parser for Option B integration.

Binds to the live9.jsonl schema (288-record capture, 2026-06-04). Parses a
single scraper record into a structured ScraperFrame, applying the schema
tolerances the user spec'd:

  - blinds: raw OCR string ("% 15/25, 5 Ante No Limit Hold'em ... TBL#1") —
    regex-extract SB/BB/ante, ignore leading "%" and table-number junk.
  - pot: variable keys ({} early, {total, main} normally, {total, main, side}
    with side pots). No assumed key set.
  - stacks / bets: int OR null per seat. null = empty seat (stacks) or no bet
    this street (bets).
  - folded: bool per seat.
  - empty: bool per seat — drives the alive-seat list (NOT busted; empty seat
    is one that's seated-out / not dealt in).
  - dealer: "seatN" string.
  - suspect: bool — REJECT frame if true (parser raises ScraperSuspect).

Tournament format pin: STARTING_CHIPS = 1500 (Ignition Double-Up 6-max turbo,
$5 buy-in level; matches the YAML config + the blueprint's training depth).
Sample JSON in the design exchange used a 2000-chip variant for scraper
calibration ONLY; the real-deployment format is 1500.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Deployment-format constants. Locked to the YAML config:
# configs/ignition_double_up_6max_turbo.yaml (starting_chips=1500, 6 seats,
# Double-Up top-3 equal payout, 5-minute turbo blinds, ante from L1).
STARTING_CHIPS: int = 1500
NUM_SEATS: int = 6

# Regex over the blinds OCR string. Tolerates:
#   leading "%" or whitespace,
#   "SB/BB, A Ante" pattern (decimal commas, spaces around slash optional),
#   any trailing OCR noise (game name, table number).
# Examples that must parse:
#   "% 15/25, 5 Ante No Limit Hold'em - Hold'em - TBL#1"
#   "15/25,5 Ante"
#   " 50/100, 15 Ante  ... whatever ..."
_BLINDS_RE = re.compile(
    r"(?P<sb>\d+)\s*/\s*(?P<bb>\d+)\s*,\s*(?P<ante>\d+)\s*[Aa]nte"
)

# Seat-name -> 0-indexed conversion. Scraper uses seat1..seat6.
_SEAT_RE = re.compile(r"^seat([1-6])$")


class ScraperParseError(Exception):
    """The scraper JSON is malformed in a way we can't recover from."""


class ScraperSuspect(Exception):
    """The scraper marked this frame as suspect; never pass to the resolver."""


@dataclass(frozen=True)
class BlindsLevel:
    sb: int
    bb: int
    ante: int

    def inflated_bb(self, n_alive: int) -> int:
        """OpenSpiel's pot-size-preserving inflation: BB carries everyone's ante.

        Matches `BlindLevel.inflated_big_blind` in src/nlhe/game_strings.py:167
        but kept local so the scraper module has no OpenSpiel dependency.
        """
        return self.bb + n_alive * self.ante


@dataclass(frozen=True)
class ScraperFrame:
    """One parsed scraper record, ready for replay + invariant check.

    All chip values are CANONICAL INTS in chips (not BBs). All seat-indexed
    arrays are length-6, with the scraper's seat1..seat6 mapped to 0..5
    (seat1 -> idx 0, seat6 -> idx 5).
    """
    # Identity / audit
    captured_at: str           # raw timestamp string, e.g. "20260604_115713_574"

    # Game-format
    blinds: BlindsLevel
    dealer_seat: int           # 0..5 — seat the button is on
    hero_seat: int             # 0..5 — by config (default 0 = "seat1")

    # Cards
    hero_cards: tuple[str, ...]   # ("6h", "7c"), length 2 if seated, else ()
    board: tuple[str, ...]        # ("Jc", "Th", "3d"), length 0/3/4/5

    # Per-seat state — all length 6, indexed 0..5
    stack: tuple[int, ...]        # chips behind, post-blind/ante posting
    bet: tuple[int, ...]          # chips committed THIS STREET
                                  #   (0 if scraper had null and seat is alive,
                                  #    0 if seat is folded/empty too — caller
                                  #    should check alive[i] before trusting)
    folded: tuple[bool, ...]      # True = seat folded earlier this hand
    empty: tuple[bool, ...]       # True = seat is not seated (no player)
    alive: tuple[bool, ...]       # derived: not empty AND stack-or-bet > 0

    # Pot (the scraper's reported total)
    pot_total: int                # required field; matches sum-of-contributions

    # Hero-to-act controls
    controls_present: bool        # True iff this frame is a hero decision
    hero_facing_bet: bool         # derived from bets: True if any opp bet > hero bet


def _seat_to_idx(name: str) -> int:
    """'seat3' -> 2. Raises ScraperParseError on malformed input."""
    m = _SEAT_RE.match(name)
    if not m:
        raise ScraperParseError(
            f"malformed seat name {name!r}; expected 'seat1'..'seat6'"
        )
    return int(m.group(1)) - 1


def parse_blinds(blinds_str: str) -> BlindsLevel:
    """Extract SB/BB/ante from the raw OCR string.

    Robust to:
      - leading garbage ('%', spaces, etc.)
      - optional whitespace around '/' and ',':  '15/25, 5 Ante'  '50 /100,15 Ante'
      - trailing junk (game name, table number, etc.)

    Raises ScraperParseError if no SB/BB/ante pattern is found.
    """
    if not isinstance(blinds_str, str):
        raise ScraperParseError(f"blinds is not a string: {blinds_str!r}")
    m = _BLINDS_RE.search(blinds_str)
    if not m:
        raise ScraperParseError(
            f"could not extract SB/BB/ante from blinds string: {blinds_str!r}"
        )
    sb = int(m.group("sb"))
    bb = int(m.group("bb"))
    ante = int(m.group("ante"))
    if sb <= 0 or bb <= 0 or ante < 0:
        raise ScraperParseError(
            f"non-positive blinds parsed from {blinds_str!r}: "
            f"sb={sb} bb={bb} ante={ante}"
        )
    if bb < sb:
        raise ScraperParseError(
            f"BB ({bb}) < SB ({sb}) in {blinds_str!r}"
        )
    return BlindsLevel(sb=sb, bb=bb, ante=ante)


def _seat_dict_to_array(d: dict, default, kind: str) -> tuple:
    """Convert {'seat1': v1, ..., 'seat6': v6} to a tuple of length 6,
    indexed 0..5. Missing seats fill with `default`. Tolerates null values
    (kept as None for the caller to coerce per-field semantics)."""
    out = [default] * NUM_SEATS
    for k, v in d.items():
        try:
            idx = _seat_to_idx(k)
        except ScraperParseError:
            # Tolerate unknown keys (e.g. future schema additions); ignore.
            continue
        out[idx] = v
    return tuple(out)


def parse_frame(record: dict, hero_seat_alias: str = "seat1") -> ScraperFrame:
    """Parse a single scraper JSON record into a ScraperFrame.

    Args:
        record: the dict from one line of liveN.jsonl.
        hero_seat_alias: which scraper seat is the hero. Default 'seat1'
            (the standard Ignition UI bottom-center convention). Change if
            the scraper indexes seats differently.

    Raises:
        ScraperSuspect: if record['suspect'] is True. Caller must drop.
        ScraperParseError: schema malformed in a non-recoverable way.
    """
    if not isinstance(record, dict):
        raise ScraperParseError(f"record is not a dict: {type(record).__name__}")

    if record.get("suspect", False):
        raise ScraperSuspect(
            f"frame marked suspect by scraper; dropping. captured_at="
            f"{record.get('captured_at', '<missing>')}"
        )

    captured_at = str(record.get("captured_at", ""))
    blinds = parse_blinds(record.get("blinds", ""))

    hero_seat = _seat_to_idx(hero_seat_alias)

    dealer_str = record.get("dealer", "")
    if not dealer_str:
        raise ScraperParseError("dealer field missing/empty")
    dealer_seat = _seat_to_idx(dealer_str)

    # Cards
    hero_cards_raw = record.get("hero_cards", []) or []
    if not isinstance(hero_cards_raw, list):
        raise ScraperParseError(f"hero_cards is not a list: {hero_cards_raw!r}")
    hero_cards = tuple(str(c) for c in hero_cards_raw)
    board_raw = record.get("board", []) or []
    if not isinstance(board_raw, list):
        raise ScraperParseError(f"board is not a list: {board_raw!r}")
    board = tuple(str(c) for c in board_raw)

    # Pot — only require 'total'; tolerate any extra keys.
    pot_dict = record.get("pot", {}) or {}
    if "total" not in pot_dict:
        # Hand-start frames may have empty pot dict; treat as 0
        pot_total = 0
    else:
        pot_total = int(pot_dict["total"])

    # Per-seat dicts
    stack_raw = _seat_dict_to_array(record.get("stacks", {}) or {}, None, "stack")
    bet_raw = _seat_dict_to_array(record.get("bets", {}) or {}, None, "bet")
    folded_raw = _seat_dict_to_array(record.get("folded", {}) or {}, False, "folded")
    empty_raw = _seat_dict_to_array(record.get("empty", {}) or {}, False, "empty")

    # Coerce per-seat semantics:
    #   stack: null OR 0 means "empty seat / busted" -> 0 chips
    #   bet:   null means "no bet THIS STREET" -> 0
    #   folded/empty: null falls back to default False
    stack = tuple(int(s) if s is not None else 0 for s in stack_raw)
    bet = tuple(int(b) if b is not None else 0 for b in bet_raw)
    folded = tuple(bool(f) if f is not None else False for f in folded_raw)
    empty = tuple(bool(e) if e is not None else False for e in empty_raw)

    # Derived: alive = seated (not empty) AND has chips somewhere
    # (stack or bet — busted seats have both 0)
    alive = tuple(
        (not empty[i]) and (stack[i] > 0 or bet[i] > 0)
        for i in range(NUM_SEATS)
    )

    # Controls
    controls_present = bool(
        (record.get("controls") or {}).get("present", False)
    )

    # Hero-facing-bet derivation: hero's bet vs max opponent bet
    if alive[hero_seat]:
        max_opp_bet = max(
            (bet[i] for i in range(NUM_SEATS) if i != hero_seat and alive[i]),
            default=0,
        )
        hero_facing_bet = max_opp_bet > bet[hero_seat]
    else:
        hero_facing_bet = False

    return ScraperFrame(
        captured_at=captured_at,
        blinds=blinds,
        dealer_seat=dealer_seat,
        hero_seat=hero_seat,
        hero_cards=hero_cards,
        board=board,
        stack=stack,
        bet=bet,
        folded=folded,
        empty=empty,
        alive=alive,
        pot_total=pot_total,
        controls_present=controls_present,
        hero_facing_bet=hero_facing_bet,
    )


def is_hand_start(frame: ScraperFrame) -> bool:
    """True iff this frame is the first frame of a new hand (board empty,
    only blinds posted, no voluntary action yet)."""
    if frame.board:
        return False
    # All bets should be either 0 or exactly the SB/BB amount, with exactly
    # one SB and one BB position (the dealer-next-two-alive-seats).
    sb_chips, bb_chips = frame.blinds.sb, frame.blinds.bb
    # Look for exactly one seat with bet==SB and one with bet==BB.
    n_sb = sum(1 for i in range(NUM_SEATS) if frame.alive[i] and frame.bet[i] == sb_chips)
    n_bb = sum(1 for i in range(NUM_SEATS) if frame.alive[i] and frame.bet[i] == bb_chips)
    n_other = sum(
        1 for i in range(NUM_SEATS)
        if frame.alive[i] and frame.bet[i] not in (0, sb_chips, bb_chips)
    )
    return n_sb == 1 and n_bb == 1 and n_other == 0


def pre_hand_stacks(frame: ScraperFrame) -> tuple[int, ...]:
    """Reconstruct each seat's stack at hand START (before blinds + antes posted).

    For each alive seat:
      pre_hand[i] = current_stack[i] + bet_this_street[i] + ante
                    + (extra street-completion add-backs, but at hand-start there are none)

    Antes are paid by every alive seat at the start of the hand; the scraper
    reports them as deducted from stack (and contributed to pot, not bet).
    For non-hand-start frames, ante reconstruction needs to handle whether
    we're still on the first street (antes already in pot but stacks reflect
    deduction). Hand-start: simple add-back.

    Returns length-6 tuple; non-alive seats stay at 0.
    """
    ante = frame.blinds.ante
    out = []
    for i in range(NUM_SEATS):
        if not frame.alive[i]:
            out.append(0)
        else:
            out.append(frame.stack[i] + frame.bet[i] + ante)
    return tuple(out)


def chip_conservation_total(frame: ScraperFrame) -> int:
    """Total chips at the table = sum of pre-hand stacks across alive seats.

    For verification: this should equal STARTING_CHIPS * (initial_n_seats)
    minus chips removed with busted seats from prior hands.

    At hand 1: == STARTING_CHIPS * NUM_SEATS == 9000.
    Mid-tournament with k busts: == 9000 - sum(busted-seat-final-stacks-at-bust)
    which is a strictly NON-increasing sequence frame-to-frame within a session.
    """
    return sum(pre_hand_stacks(frame))
