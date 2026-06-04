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
    """The scraper JSON is structurally malformed (schema violation, real
    corruption). Raise loudly — these are bugs to investigate.
    """


class ScraperSuspect(Exception):
    """The scraper marked this frame as suspect:true; drop silently."""


class ScraperDataQuality(Exception):
    """The frame is syntactically fine but the scraper's reading is unusable
    for replay (dealer field missing/empty due to OCR miss, dealer button
    apparently on an empty seat, etc.). DROP the frame — do NOT raise to
    the resolver. Counted separately from ScraperParseError so we can see
    scraper-coverage gaps in the corpus stats without conflating them with
    actual schema violations.
    """


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
        # Scraper sometimes misses the dealer button mid-move-animation or
        # when the button graphic is partly off-screen. This is a coverage
        # gap, not a schema violation -> soft drop.
        raise ScraperDataQuality(
            f"dealer field missing/empty; OCR coverage gap. captured_at="
            f"{record.get('captured_at', '<missing>')}"
        )
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

    # Dealer-on-empty: the button can't sit on an empty seat in real poker.
    # When the scraper reports this, it's an OCR drift (typically a
    # transient frame just before/after the button moves). Soft-drop.
    if empty[dealer_seat] or not alive[dealer_seat]:
        raise ScraperDataQuality(
            f"dealer points to seat{dealer_seat+1} but that seat is "
            f"empty/non-alive (alive={alive}, empty={empty}); soft drop. "
            f"captured_at={record.get('captured_at', '<missing>')}"
        )

    # n_alive < 4: out of the model's training distribution AND out of the
    # deployment format's playable range. _sample_alive_count requires
    # alive_count >= num_paid + 1 = 4 (the ship blueprint + rebel value-net
    # have zero exposure to 2/3-alive states), and Double-Up top-3
    # terminates at 3-alive (the match is over; everyone left has cashed).
    # Any frame with n_alive < 4 is therefore either (a) a different
    # tournament format mixed into the corpus, or (b) a transient frame
    # captured between bust and the match-end UI update. Either way the
    # resolver should not be asked to decide on it. Soft-drop.
    n_alive_total = sum(alive)
    if n_alive_total < 4:
        raise ScraperDataQuality(
            f"n_alive={n_alive_total} < 4; below the trained model's "
            f"sample_starting_state lower bound (alive_count >= num_paid+1=4) "
            f"AND below the Double-Up top-3 deployment's playable range "
            f"(match terminates at 3-alive). Soft drop. "
            f"captured_at={record.get('captured_at', '<missing>')}"
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


# --------------------------------------------------------------------------
# Action-sequence derivation (Phase 2 Piece 2)
# --------------------------------------------------------------------------


class ActionDerivationError(Exception):
    """Action-sequence derivation hit an inconsistency; caller drops the frame
    as ScraperDataQuality (the safe-fallback path)."""


def preflop_action_order(dealer_seat: int, alive_seats: list[int]) -> list[int]:
    """Return seats in OpenSpiel preflop action order, ONE FULL LAP.

    Matches `to_inner_game_string_for_state`'s alive-seat rotation
    (game_strings.py:397-423) for ALIVE seats only. EMPTY (non-alive) seats
    are NOT in this list — but the replay engine MUST account for them, since
    OpenSpiel cycles through all seats including the stack=1 placeholders.
    `preflop_action_order_with_empties` is the version that includes empties
    and is the right one for replay sequence emission.

    Use this version for ALGORITHMIC reasoning (UTG/SB/BB identification);
    use the _with_empties version for emitting action sequences fed to
    OpenSpiel's state.apply_action.
    """
    n_alive = len(alive_seats)
    if n_alive < 2:
        return []
    dpos = alive_seats.index(dealer_seat)
    if n_alive == 2:
        bb_seat = alive_seats[(dpos + 2) % n_alive]  # = dealer (library bug)
        sb_seat = alive_seats[(dpos + 1) % n_alive]
        return [bb_seat, sb_seat]
    return [alive_seats[(dpos + 3 + i) % n_alive] for i in range(n_alive)]


def preflop_action_order_with_empties(dealer_seat: int,
                                        alive: tuple[bool, ...]) -> list[int]:
    """Like preflop_action_order but includes EMPTY (non-alive) seats in
    their OpenSpiel cycle position. OpenSpiel rotates through ALL six seats
    (including stack=1 placeholders for empties), so the action sequence
    must emit an action for each empty seat too (it's a forced fold).

    The order starts at UTG-among-alive and walks clockwise through ALL
    seat indices 0..5, in absolute order rotating from UTG. Empty seats
    that fall in the natural clockwise position are kept.
    """
    alive_seats = [i for i in range(NUM_SEATS) if alive[i]]
    n_alive = len(alive_seats)
    if n_alive < 2:
        return []
    if n_alive == 2:
        # Heads-up: only the two alive seats get cycled (we don't model the
        # empties' "forced fold" turns for heads-up since the n_alive<4
        # soft-drop rejects these frames anyway).
        return preflop_action_order(dealer_seat, alive_seats)
    # UTG = (dealer + 3) % NUM_SEATS in ABSOLUTE seat numbering when all
    # seats are alive. Shorthanded: UTG is the third ALIVE seat clockwise
    # from dealer (library convention).
    dpos = alive_seats.index(dealer_seat)
    utg_seat = alive_seats[(dpos + 3) % n_alive]
    # From utg_seat, walk clockwise through ALL 6 seats once
    return [(utg_seat + offset) % NUM_SEATS for offset in range(NUM_SEATS)]


def postflop_action_order(dealer_seat: int, alive_seats: list[int],
                           folded: tuple[bool, ...]) -> list[int]:
    """Return alive-non-folded seats in OpenSpiel postflop action order
    (SB first, then clockwise; folded seats skipped).

    SB is alive_seats[(dpos+1) % n_alive]. We walk clockwise from there and
    skip any seat that's folded. n_alive == 2 case: SB acts first postflop
    (= non-dealer; this matches both real-poker and the library's
    `postflop_actor = sb_seat + 1` line, so no convention-bug here for
    postflop)."""
    n_alive = len(alive_seats)
    if n_alive < 2:
        return []
    dpos = alive_seats.index(dealer_seat)
    sb_alive_idx = (dpos + 1) % n_alive
    order = []
    for offset in range(n_alive):
        seat = alive_seats[(sb_alive_idx + offset) % n_alive]
        if not folded[seat]:
            order.append(seat)
    return order


def _street_idx_from_board(board: tuple) -> int:
    """Map scraper.board length to OpenSpiel street_idx.
    0/3/4/5 cards -> 0/1/2/3 (preflop/flop/turn/river). Anything else: error."""
    n = len(board)
    if n == 0:
        return 0
    if n == 3:
        return 1
    if n == 4:
        return 2
    if n == 5:
        return 3
    raise ActionDerivationError(
        f"invalid board length {n}; expected 0/3/4/5"
    )


def _derive_pre_hand_and_preflop_commit_simple_model(
        frame: ScraperFrame, sb_seat: int, bb_seat: int
        ) -> tuple[tuple[int, ...], int]:
    """Same simple-model chip-conservation derivation as
    _derive_pre_hand_simple_model but ALSO returns preflop_commit_per_alive,
    which Piece 5's mid-hand invariant needs for prior-streets-committed
    bookkeeping when converting cumulative-OpenSpiel-contribution to
    current-street-scraper-bet on postflop frames.

    Returns (pre_hand_stacks: tuple[int,6], preflop_commit_per_alive: int).
    """
    alive_seats = [i for i in range(NUM_SEATS) if frame.alive[i]]
    n_alive = len(alive_seats)
    ante = frame.blinds.ante
    sb = frame.blinds.sb
    bb = frame.blinds.bb

    folded_commit_total = 0
    folded_seats = [i for i in alive_seats if frame.folded[i]]
    for i in folded_seats:
        folded_commit_total += ante
        if i == sb_seat:
            folded_commit_total += sb
        elif i == bb_seat:
            folded_commit_total += bb

    alive_non_folded = [i for i in alive_seats if not frame.folded[i]]
    n_anf = len(alive_non_folded)

    if n_anf == 0:
        preflop_commit_per_alive = 0
    else:
        bet_sum_anf = sum(frame.bet[i] for i in alive_non_folded)
        residual = (frame.pot_total - folded_commit_total
                     - n_anf * ante - bet_sum_anf)
        preflop_commit_per_alive = max(0, residual // n_anf)

    pre = []
    for i in range(NUM_SEATS):
        if not frame.alive[i]:
            pre.append(0)
        elif frame.folded[i]:
            blind_amt = sb if i == sb_seat else bb if i == bb_seat else 0
            pre.append(frame.stack[i] + ante + blind_amt)
        else:
            pre.append(frame.stack[i] + ante
                        + preflop_commit_per_alive
                        + frame.bet[i])
    return tuple(pre), preflop_commit_per_alive


def _derive_pre_hand_simple_model(frame: ScraperFrame,
                                    sb_seat: int, bb_seat: int
                                    ) -> tuple[int, ...]:
    """Pre-hand stacks (chips at hand START) via the simple-model chip-
    conservation derivation. See
    _derive_pre_hand_and_preflop_commit_simple_model for the algorithm + the
    simple-model assumptions. This wrapper drops the preflop_commit return
    value; callers that need it (Piece 5 invariant) use the longer name.
    """
    pre, _ = _derive_pre_hand_and_preflop_commit_simple_model(
        frame, sb_seat, bb_seat)
    return pre


def derive_action_sequence(frame: ScraperFrame
                            ) -> list[tuple[int, int]]:
    """Derive the canonical (seat_idx, openspiel_chip_int) sequence to walk
    OpenSpiel state from new_initial_state() to the hero's current decision.

    Stop condition (caller stops at the hero's decision):
      - On preflop: we walk the preflop order; we BREAK at the hero's slot.
        Whether the hero has already acted (re-raise, second lap) is NOT
        handled here — first-lap only. Multi-lap preflop frames will produce
        an action sequence that lands at the wrong current_player and the
        downstream invariant rejects them as ScraperDataQuality (correct
        safe-fallback). Acceptable for Phase 2 first pass; tighten if the
        corpus shows non-trivial multi-lap incidence.
      - On postflop frames: we emit all preflop actions, then for each
        postflop street up to (but not including) the current street, emit
        a CHECK for every alive non-folded seat in postflop order. On the
        CURRENT postflop street, emit the seat actions implied by frame.bet
        in postflop order; STOP at the hero's slot.

    Simple postflop model: all prior-street commit (preflop + intermediate
    streets) is attributed to PREFLOP — intermediate streets are
    all-checks. Frames where alive seats raised on intermediate streets
    (so their preflop_commit derived from total - current_bet doesn't
    match across alive seats) will fail the downstream invariant. The
    invariant's strictness is the correctness gate — this function is
    permitted to be approximate as long as it's right for the common case.

    Returns: list of (seat_idx, openspiel_chip_int). For each entry,
    apply_action(chip_int) on the OpenSpiel state when state.current_player()
    is at that seat (Piece 4's replay engine handles chance-node
    interleaving and asserts the seat matches).

    Raises:
        ActionDerivationError: structural inconsistency that should make
            the caller treat the frame as ScraperDataQuality (soft drop).
    """
    alive_seats = [i for i in range(NUM_SEATS) if frame.alive[i]]
    n_alive = len(alive_seats)
    if n_alive < 2:
        raise ActionDerivationError(
            f"n_alive={n_alive} < 2; no hand possible")

    dpos = alive_seats.index(frame.dealer_seat)
    sb_seat = alive_seats[(dpos + 1) % n_alive]
    bb_seat = alive_seats[(dpos + 2) % n_alive] if n_alive >= 3 \
        else alive_seats[(dpos + 2) % n_alive]

    street_idx = _street_idx_from_board(frame.board)

    # Per-seat pre-hand stacks via simple-model chip conservation. Works
    # for both hand-start (P_max=0) and postflop frames. See helper docstring
    # for the simple-model assumption and its failure modes.
    pre = _derive_pre_hand_simple_model(frame, sb_seat, bb_seat)

    # In the simple model, for alive seats we ASSUME all prior-street commit
    # was preflop. For a preflop frame, preflop_commit_seat = total_commit_seat
    # (i.e., frame.bet[seat] = preflop_commit for alive seats, and the
    # pre_hand_stacks helper already adds ante back so chip arithmetic closes).
    #
    # For postflop alive seats: preflop_commit = total_committed - current_bet.
    # But the SIMPLE model picks a single preflop_max for all alive
    # non-folded seats. We pick preflop_max = max alive (total_committed -
    # current_bet) — i.e., the highest preflop commitment any alive seat
    # made. If alive seats DIFFER on this metric, they had non-matching
    # prior-street commits (some bet on an intermediate street, others
    # didn't), which the simple model can't represent. We DON'T raise here;
    # we proceed with the max and let the downstream invariant catch any
    # state mismatch.

    if street_idx == 0:
        # PREFLOP frame. preflop_commit[seat] = chips committed THIS STREET
        # (frame.bet[seat], for alive seats). For folded seats the bet field
        # is 0 in our parser, but they may have committed sb/bb if they
        # were the blind seat and folded after just-blinding.
        # In OpenSpiel accounting (un-inflated bb on the integration side),
        # the BB seat's preflop_commit_in_openspiel after blinds posted = bb
        # (which the inflated_bb represents in the buggy library; we already
        # bug-match elsewhere). For action sequence, we emit chip_ints in
        # SCRAPER-EQUIVALENT accounting and the OpenSpiel engine handles the
        # inflated bookkeeping on its side.
        preflop_commit = list(frame.bet)
    else:
        # POSTFLOP: compute per-seat total_committed, derive preflop_commit
        # via simple model (all prior to preflop).
        total_committed = [
            pre[i] - frame.stack[i] if frame.alive[i] else 0
            for i in range(NUM_SEATS)
        ]
        # Subtract ante (which is in "chips put in pot" arithmetic but NOT
        # in OpenSpiel-side seat contributions for non-BB seats; the BB
        # carries everyone's ante via the inflated convention).
        ante = frame.blinds.ante
        preflop_commit = [
            max(0, total_committed[i] - frame.bet[i] - ante)
            if frame.alive[i] else 0
            for i in range(NUM_SEATS)
        ]
        # For folded seats: their total_committed includes ante + whatever
        # they paid before folding. Attribute all of it to preflop.
        for seat in range(NUM_SEATS):
            if not frame.alive[seat]:
                continue
            if frame.folded[seat]:
                preflop_commit[seat] = max(
                    0, total_committed[seat] - ante)

    # PREFLOP action emission — walk WITH-EMPTIES so OpenSpiel's natural
    # cycle (which includes stack=1 placeholders for empty seats) lines up.
    # Empty seats emit forced-fold; alive seats emit per the simple model.
    actions: list[tuple[int, int]] = []
    pf_order = preflop_action_order_with_empties(frame.dealer_seat, frame.alive)
    running_max_pf = frame.blinds.bb  # initial preflop max = BB
    for seat in pf_order:
        # Empty seat: forced fold (OpenSpiel placeholder cycles through them).
        if not frame.alive[seat]:
            actions.append((seat, 0))
            continue
        # Hero stop condition for preflop frame
        if street_idx == 0 and seat == frame.hero_seat:
            break
        seat_commit_pf = preflop_commit[seat]
        if frame.folded[seat]:
            actions.append((seat, 0))  # fold
            # Folded seats don't update running_max
            continue
        if seat_commit_pf == 0:
            # Alive seat with zero commit — usually means hero wasn't reached
            # yet in the order; emit fold defensively (the invariant will
            # catch if this is wrong).
            actions.append((seat, 0))
            continue
        if seat_commit_pf < running_max_pf:
            # Less than current max → they must have folded (committed only
            # blind/partial); already handled above for folded seats; here
            # alive-but-under-max is anomalous, treat as fold defensively.
            actions.append((seat, 0))
        elif seat_commit_pf == running_max_pf:
            # Call/check
            actions.append((seat, 1))
        else:
            # Raise to seat_commit_pf (OpenSpiel chip_int = cumulative
            # voluntary commitment for non-BB; for BB it's also cumulative
            # but starts at inflated_bb). For the bug-matched library, the
            # right chip_int for a non-BB raise equals their scraper-side
            # voluntary bet (= seat_commit_pf for preflop).
            actions.append((seat, seat_commit_pf))
            running_max_pf = seat_commit_pf

    if street_idx == 0:
        return actions

    # POSTFLOP: emit intermediate streets (all-checks) then current street
    pf_alive_post = postflop_action_order(
        frame.dealer_seat, alive_seats, frame.folded)
    # Intermediate streets (1..street_idx-1): all checks
    for _ in range(1, street_idx):
        for seat in pf_alive_post:
            actions.append((seat, 1))  # check
    # Current street: emit bets per frame.bet in postflop order, stop at hero
    running_max_cs = 0  # current-street max (chips IN FRONT this street)
    for seat in pf_alive_post:
        if seat == frame.hero_seat:
            break
        cs_commit = frame.bet[seat]
        if cs_commit == 0:
            actions.append((seat, 1))  # check
        elif cs_commit == running_max_cs:
            actions.append((seat, 1))  # call
        else:
            # Raise this street: chip_int = cumulative across streets =
            # preflop_commit[seat] + cs_commit
            total_int = preflop_commit[seat] + cs_commit
            actions.append((seat, total_int))
            running_max_cs = cs_commit

    return actions
