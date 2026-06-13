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

import dataclasses
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

    # Dead button (D2, flag-gated): True iff the dealer button sits on an
    # empty/eliminated seat — the standard short-handed dead-button
    # rotation (the BB advances exactly one ACTIVE player per hand; SB and
    # button derive from it, so the button can land on the seat vacated by
    # the previous hand's bust). Only ever True when parse_frame ran with
    # dead_button_handling=True; the default keeps every pre-D2
    # constructor call and pickle byte-compatible.
    dealer_dead: bool = False


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


def parse_frame(record: dict, hero_seat_alias: str = "seat1",
                allow_suspect: bool = False,
                dead_button_handling: bool = False) -> ScraperFrame:
    """Parse a single scraper JSON record into a ScraperFrame.

    Args:
        record: the dict from one line of liveN.jsonl.
        hero_seat_alias: which scraper seat is the hero. Default 'seat1'
            (the standard Ignition UI bottom-center convention). Change if
            the scraper indexes seats differently.
        allow_suspect: parse a suspect:true record instead of raising
            ScraperSuspect. ONLY for the single-field recovery path
            (live_loop._attempt_suspect_stack_recovery), which re-validates
            the frame through replay + invariant before any use. Every
            other caller must keep the default — a suspect frame's flagged
            field is known-bad and must never reach the model raw.
        dead_button_handling: D2 dead-button position handling (operator
            directive 2026-06-12; OFF by default). When False, a frame
            whose dealer points at an empty/non-alive seat soft-drops as
            ScraperDataQuality — byte-identical to the pre-D2 bridge.
            When True, such a frame is accepted with dealer_dead=True
            (the standard short-handed dead-button rotation; Windows
            attribution proved these are correctly scraped). The additive
            Windows-side `dealer_dead` schema key is PREFERRED when
            present; when absent, deadness derives from the dealer seat's
            emptiness/stack. SB/BB assignment for a dead-button frame is
            resolved downstream by _derive_blinds_and_action_order,
            validated against the observed blind posts (Predicate 1),
            with the unchanged replay + invariant gate as the final
            correctness bar — an OCR-drift frame (transient button-move
            capture, the other source of dealer-on-empty reads)
            reconstructs inconsistently and still drops there.

    Raises:
        ScraperSuspect: if record['suspect'] is True (and allow_suspect is
            False, the default). Caller must drop.
        ScraperParseError: schema malformed in a non-recoverable way.
    """
    if not isinstance(record, dict):
        raise ScraperParseError(f"record is not a dict: {type(record).__name__}")

    if record.get("suspect", False) and not allow_suspect:
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

    # Dealer-on-empty. Two real-world causes:
    #   (a) DEAD BUTTON — standard short-handed rotation: the BB advances
    #       exactly one ACTIVE player per hand; SB and button derive from
    #       it, so the button legitimately sits on the seat vacated by the
    #       previous hand's bust (Windows attribution 2026-06-12: the
    #       session-5 #1 hand-killer skip class is correctly-scraped dead
    #       buttons, e.g. live seq 912 QdKc / seq 1366 TsAc).
    #   (b) OCR drift — a transient frame just before/after the button
    #       graphic moves.
    # Flag-gated (D2): with dead_button_handling=False (default) BOTH
    # cases soft-drop, byte-identical to the pre-D2 bridge. With the flag
    # ON the frame is accepted and marked dealer_dead; case (b) is caught
    # downstream (post-validated SB/BB + replay + invariant). The additive
    # Windows `dealer_dead` key is preferred when present: an explicit
    # dealer_dead=false against an empty-reading dealer seat is
    # contradictory (drift), so the pre-D2 drop is kept for it.
    dealer_dead = False
    dealer_dead_reported = record.get("dealer_dead", None)
    if dealer_dead_reported is not None:
        dealer_dead_reported = bool(dealer_dead_reported)
    if empty[dealer_seat] or not alive[dealer_seat]:
        if not dead_button_handling or dealer_dead_reported is False:
            raise ScraperDataQuality(
                f"dealer points to seat{dealer_seat+1} but that seat is "
                f"empty/non-alive (alive={alive}, empty={empty}); soft drop. "
                f"captured_at={record.get('captured_at', '<missing>')}"
            )
        dealer_dead = True
    elif dead_button_handling and dealer_dead_reported:
        # Scraper flags a dead button on a seat we read as alive. Trust
        # the explicit marker (it is the higher-information source); for
        # the >= 3-alive states the bridge serves, the SB/BB derivation
        # walks clockwise from dealer+1 either way, so this is
        # annotation-only.
        dealer_dead = True

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

    # Controls — hero is "to act" iff the UI shows controls AND at least
    # one non-FOLD action is available. A frame whose ONLY available button
    # is FOLD is Ignition's persistent muck-anytime UI (visible between
    # actions for already-committed hands); it's not a strategic decision.
    # Real call-or-fold decisions always include at least one CALL/CHECK/
    # BET/RAISE alongside FOLD — verified across the corpus (0 false
    # positives: every real-decision signature has a non-FOLD action).
    controls_block = record.get("controls") or {}
    controls_present = bool(controls_block.get("present", False))
    if controls_present:
        btn_labels = [
            (b.get("label") or "").upper()
            for b in (controls_block.get("action_buttons") or [])
        ]
        if len(btn_labels) == 1 and btn_labels[0] == "FOLD":
            # Non-decision UI state — bot has no strategic choice here.
            # Routed to not_hero_to_act downstream, not invariant_fail.
            controls_present = False

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
        dealer_dead=dealer_dead,
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


def is_preblind_hand_start(frame: ScraperFrame) -> bool:
    """True iff this frame is an ANTE-ONLY hand-start capture: board
    empty, no bets posted yet, and the pot holds exactly the antes
    (n_alive * ante, ante > 0).

    One UI beat earlier than `is_hand_start` (which requires the SB+BB
    posts to be visible). Used ONLY by the flag-gated P2 pre-blind
    anchor capture: when a blind post is displaced into a seat's stack
    (the bet-closure signature), `is_hand_start` never fires for that
    hand, so the ante-only frame is the last clean anchorable view.
    The pot == antes requirement is what excludes between-hands and
    mid-hand frames: once any blind/bet is posted the pot exceeds the
    antes (under Ignition's bets-in-pot convention), and post-hand
    frames read pot == 0."""
    if frame.board:
        return False
    ante = int(frame.blinds.ante)
    if ante <= 0:
        return False
    if any(int(frame.bet[i]) != 0 for i in range(NUM_SEATS)):
        return False
    n_alive = sum(frame.alive)
    return int(frame.pot_total) == n_alive * ante


def preblind_pre_hand_stacks(frame: ScraperFrame) -> tuple[int, ...]:
    """Per-seat hand-start stacks from an ante-only frame
    (`is_preblind_hand_start`): stack + ante per alive seat — only the
    ante has left the stack so far. Non-alive seats stay at 0, mirroring
    `pre_hand_stacks`."""
    ante = int(frame.blinds.ante)
    return tuple(
        frame.stack[i] + ante if frame.alive[i] else 0
        for i in range(NUM_SEATS)
    )


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
# Session state tracker (Option 3 — multi-hand stack accumulation)
# --------------------------------------------------------------------------


class SessionTracker:
    """Track per-hand pre-hand stacks across a sequence of scraper frames.

    Use case: the integration's simple-model pre-hand derivation assumes
    every seat enters each hand with 1500 chips. After hand 1 of a
    tournament, that's false — winners have stacks > 1500, losers < 1500.
    Without per-seat pre-hand stacks, postflop frames in such hands can't
    reconstruct cleanly (chip arithmetic doesn't close), and the simple
    model produces sub-min-raise chip_ints that OpenSpiel rejects.

    SessionTracker observes each frame as it streams through the harness.
    When it sees a hand-start frame (board empty, only blinds posted, no
    voluntary action yet — per `is_hand_start`), it records per-seat
    `pre_hand_stacks(frame)` and the hand's identifying key (dealer +
    level). For subsequent mid-hand frames in the same hand, callers can
    query `pre_hand_for(frame)` to retrieve those per-seat values; they
    pass them as the `pre_hand_override` parameter into
    `derive_action_sequence` / `replay_to_decision`, bypassing the
    simple-model uniform-commit assumption.

    Discipline: returns the tracked pre-hand stacks ONLY when the
    chip-conservation arithmetic still closes for the current frame
    (sum of tracked pre-hand stacks == sum of current frame stacks +
    pot total). Any mismatch signals a hand boundary or a missed
    hand-start capture; the tracker returns None in that case so the
    caller falls back to the simple model (which will then safely reject
    via ActionDerivationError if the frame is irreducibly asymmetric).

    Stateless WRT the rest of the integration — instantiate one per
    session/stream and call `observe` on every frame in order.
    """

    # Chips-in-play ceiling for the anchor guard (2026-06-09,
    # docs/OBSERVE_CEILING_GUARD_DESIGN.md). Derived from the
    # deployment-format pins above (STARTING_CHIPS x NUM_SEATS = 9000):
    # chips never leave a single-table SNG, so the true table total is
    # constant for the whole match. Load-bearing format config that the
    # entire training/eval/replay stack already depends on — NOT a
    # hand-tuned tolerance of the scraper's STACK_IMPOSSIBLE=13000 class.
    # The observed-mode cross-check in observe() covers the remaining
    # failure mode (wrong table format mounted under this config).
    _CHIPS_IN_PLAY_CEILING: int = STARTING_CHIPS * NUM_SEATS

    # Minimum accepted anchors before the observed-mode cross-check is
    # meaningful (early under-read transients would false-alarm it).
    _OOD_MIN_ANCHORS: int = 5

    def __init__(self, anchor_sum_floor: bool = False,
                 bet_closure_recovery: bool = False) -> None:
        self._current_pre_hand: tuple[int, ...] | None = None
        self._current_hand_key: tuple | None = None
        # Accepted-anchor chip sums for the [ANCHOR-OOD] cross-check.
        self._anchor_sums: dict[int, int] = {}
        self._ood_warned: bool = False
        # P1 anchor sum-floor guard (approved 2026-06-11, seq-1363
        # postmortem): when ON, a hand-start anchor whose pre-hand sum
        # differs from chips-in-play while EVERY seat reads alive is
        # refused. With all six stacks visible there is no mis-alive
        # seat to hide chips behind — the sum must close exactly; a
        # short sum is an OCR corruption (dropped digit) that poisons
        # the anchor and inverts the Layer-1 stack recovery (seq 1363:
        # anchored sum 8200 passed the ceiling-only guard, recovery
        # then "derived" seat6=8 against a true 808). The ceiling-only
        # rationale (mis-alive seats legitimately sum BELOW) still
        # applies whenever any seat reads non-alive. OFF by default.
        self._anchor_sum_floor = bool(anchor_sum_floor)
        # P2 bet-closure recovery (approved 2026-06-11, session-3
        # postmortem; built 2026-06-12): when ON, observe() additionally
        # captures a PRE-BLIND hand-start anchor from clean ante-only
        # frames (board empty, all bets 0, pot == n_alive*ante). The
        # displacement signature P2 recovers from — one seat's blind/bet
        # rendered into its stack — corrupts the blinds-posted hand-start
        # too (is_hand_start sees no BB poster), so the REGULAR anchor is
        # systematically absent for exactly the hands P2 needs one
        # (live seq-276, 2026-06-12). The pre-blind capture happens one
        # UI beat earlier, before any blind can be displaced. Stored in a
        # SEPARATE slot served ONLY via bet_closure_anchors_for(): the
        # regular anchor flow (pre_hand_for / anchor_for /
        # corrected_pot_for) is byte-identical with the flag ON.
        # Gated on P1: a pre-blind anchor must clear the same ceiling +
        # sum-floor guards as a regular one (enforced in __init__).
        self._bet_closure_recovery = bool(bet_closure_recovery)
        if self._bet_closure_recovery and not self._anchor_sum_floor:
            raise ValueError(
                "bet_closure_recovery requires anchor_sum_floor (P2 is "
                "gated on P1 — recovery derives chip values from the "
                "anchor, so the anchor must be sum-floor guarded; "
                "seq-1363 poisoned-anchor postmortem)")
        self._preblind_pre_hand: tuple[int, ...] | None = None
        self._preblind_key: tuple | None = None
        self._preblind_refused_key: tuple | None = None
        # Blind-posting evidence for the P2 dead-SB guard (first capture
        # per hand-key wins): "sb_only" / "bb_only" — see
        # blind_posting_evidence_for. Captured in the same flag-gated
        # hook as the pre-blind anchor; never read outside P2.
        self._blind_evidence: str | None = None
        self._blind_evidence_key: tuple | None = None

    def _hand_key(self, frame: ScraperFrame) -> tuple:
        """Identifying tuple for a hand. A change in any component
        indicates the tracker is no longer looking at the same hand.

        Excludes `alive[]` deliberately. Diagnosed live 2026-06-08 (seq=148
        hand-start anchored alive=6; seq=156 still dealer=1 but a player had
        just folded and the scraper now reported alive=5 — old key was
        (dealer, blinds, alive) so it mismatched and `pre_hand_for` returned
        None for every subsequent frame, including the river-trips spot at
        seq=175 → safe-fold on a 3-of-a-kind value bet). Within one hand
        the dealer button cannot move, so (dealer, blinds) uniquely
        identifies the hand. The chip-conservation closure check in
        `pre_hand_for` is the safety net against (extremely unlikely) key
        collision."""
        return (
            int(frame.dealer_seat),
            int(frame.blinds.sb),
            int(frame.blinds.bb),
            int(frame.blinds.ante),
        )

    def observe(self, frame: ScraperFrame) -> bool | None:
        """Update internal state from this frame.

        On a hand-start frame: compute and record pre-hand stacks IFF the
        hand_key has changed (= a new hand began) or no hand is tracked
        yet. Subsequent hand-start frames for the SAME hand (Ignition
        captures the hand-start UI multiple times before action starts)
        are ignored — the first capture is typically the most reliable
        because subsequent re-captures sometimes show transient stale
        stack/pot readings that don't reflect the actual game state.

        Non-hand-start frames don't update state — the tracked pre-hand
        stays valid until a new hand-start replaces it.

        Returns:
            True  — a new anchor was accepted from this frame.
            False — this frame was a new hand-start whose anchor was
                    REFUSED by the chips-in-play ceiling guard (caller
                    may surface this for audit, e.g. LiveDecision.
                    anchor_refused).
            None  — no-op (not a hand-start, same hand already anchored,
                    or internally-inconsistent capture).
        """
        if not is_hand_start(frame):
            # P2 pre-blind anchor capture (flag-gated; see __init__).
            # Never changes observe()'s return value or any regular-
            # anchor state — flag-ON behavior outside the bet-closure
            # path stays byte-identical.
            if self._bet_closure_recovery:
                self._maybe_capture_preblind_anchor(frame)
                self._maybe_capture_blind_evidence(frame)
            return None
        key = self._hand_key(frame)
        if key == self._current_hand_key:
            # Same hand, already recorded. Don't overwrite — keep the
            # first capture which is empirically more reliable.
            return None
        pre = pre_hand_stacks(frame)

        # Chips-in-play ceiling guard (2026-06-09 design, approved).
        # A poisoned hand-start (stuck-digit stack, e.g. seat6=9907 ->
        # sum(pre)=17917) passes the self-consistency check below because
        # both sides of that equation use the same wrong stack — and a
        # poisoned ANCHOR is the one input the recovery/invariant gate
        # cannot independently verify (replayed 154557: 7 poisoned
        # decision_recovered outputs, phantom seat6 stacks 9822/9897/9907).
        # Upper bounds only: 30/178 corpus hand-starts legitimately sum
        # BELOW the ceiling (mis-alive hidden chips) and must keep
        # anchoring. Strict `>`: equality is unreachable — parse_frame
        # rejects n_alive < 4 and every alive seat holds > 0 chips, so a
        # single pre-hand stack is <= total - 3 in any frame that gets
        # here; a seat equal to the full chips-in-play implies the match
        # is over. Per-seat bound is mathematically implied by the sum
        # bound (pre-hand values are non-negative) — kept because it
        # names the offending seat in the audit log and stands on its
        # own if a future change ever makes components signed.
        ceiling = self._CHIPS_IN_PLAY_CEILING
        over_seats = [
            i for i in range(NUM_SEATS) if int(pre[i]) > ceiling
        ]
        if over_seats or sum(pre) > ceiling:
            detail = (
                ", ".join(f"seat{i+1} pre_hand={pre[i]}"
                          for i in over_seats)
                or f"sum(pre_hand)={sum(pre)}"
            )
            print(f"[ANCHOR-REFUSED] hand-start anchor fails "
                  f"chips-in-play ceiling ({ceiling}): {detail}  "
                  f"captured_at={frame.captured_at}", flush=True)
            return False

        # P1 sum-floor guard (flag-gated; see __init__). Placed before
        # the internal-consistency check on purpose: that check passes
        # for a dropped-digit corruption because both sides use the
        # same wrong stack — exactly the seq-1363 poisoning.
        if (self._anchor_sum_floor and sum(pre) != ceiling
                and all(frame.alive)):
            print(f"[ANCHOR-REFUSED] hand-start anchor fails sum-floor "
                  f"guard: sum(pre_hand)={sum(pre)} != chips-in-play "
                  f"{ceiling} with all {NUM_SEATS} seats alive (no "
                  f"mis-alive seat to hide chips) "
                  f"captured_at={frame.captured_at}", flush=True)
            return False

        chip_total = sum(
            int(s) for s, a in zip(frame.stack, frame.alive) if a
        ) + int(frame.pot_total)
        # Reject internally-inconsistent hand-start captures.
        if sum(pre) != chip_total:
            return None
        self._current_pre_hand = pre
        self._current_hand_key = key

        # Observed-mode cross-check (design Q1 option C): if the mode of
        # accepted anchor sums disagrees with the config-derived ceiling,
        # the mounted table's format doesn't match the config — every
        # layer above is OOD, not just this guard. Log once, loudly.
        s = sum(pre)
        self._anchor_sums[s] = self._anchor_sums.get(s, 0) + 1
        if not self._ood_warned and (
                sum(self._anchor_sums.values()) >= self._OOD_MIN_ANCHORS):
            mode = max(self._anchor_sums, key=self._anchor_sums.get)
            if mode != ceiling:
                self._ood_warned = True
                print(f"[ANCHOR-OOD] observed hand-start chip-sum mode "
                      f"({mode}, n={self._anchor_sums[mode]}) != config "
                      f"chips-in-play ({ceiling}) — wrong table format "
                      f"mounted under this config?", flush=True)
        return True

    def pre_hand_for(self, frame: ScraperFrame) -> tuple[int, ...] | None:
        """Return tracked pre-hand stacks for this frame's hand, or None.

        Closure check accepts the anchor when chip totals are consistent
        with ONE of three patterns:

          1. Strict closure:  Σ stack[seats_at_start] + pot == expected
          2. UI-lag:          Σ stack[seats_at_start] + pot + Σ bet == expected
                              (uncollected bets sit in front of seats; pot
                               UI hasn't picked them up yet)
          3. Mis-alive:       deficit is bounded by mis-alive seats' max
                              possible remaining chips (= pre_hand - ante
                              each). Live regression 2026-06-08 surfaced
                              that Ignition mis-reads folded-but-still-
                              seated players as alive=False with stack=0,
                              hiding their actual remaining chips from
                              the visible-stack sum.

        Patterns 2 and 3 can co-occur (UI lag PLUS a mis-alive seat). The
        check folds in the UI-lag bets first, then checks the remaining
        deficit against the mis-alive upper bound.
        """
        if self._current_pre_hand is None:
            return None
        if self._hand_key(frame) != self._current_hand_key:
            return None
        if self._closure_plausible(frame):
            return self._current_pre_hand
        return None

    def anchor_for(self, frame: ScraperFrame) -> tuple[int, ...] | None:
        """Tracked pre-hand stacks for this frame's hand WITHOUT the
        chip-conservation closure check.

        For the suspect-recovery path ONLY: closure cannot hold on a frame
        whose flagged stack field is wrong — that broken closure is the
        thing recovery solves for. The caller must re-validate the
        recovered frame through the full replay + invariant gate before
        any use. Never use this as a general pre_hand_for substitute:
        without the closure check there is no evidence the anchor still
        describes the current hand beyond the (dealer, blinds) key.
        """
        if self._current_pre_hand is None:
            return None
        if self._hand_key(frame) != self._current_hand_key:
            return None
        return self._current_pre_hand

    @property
    def anchor_sum_floor_armed(self) -> bool:
        """True iff the P1 anchor sum-floor guard is armed (P2's
        bet-closure recovery refuses to run without it)."""
        return self._anchor_sum_floor

    def _maybe_capture_preblind_anchor(self, frame: ScraperFrame) -> None:
        """Record a pre-blind hand-start anchor (P2, flag-gated).

        First capture per hand-key wins; a guard refusal is sticky for
        the key (recorded so bet_closure_anchors_for can surface
        "anchor refused" instead of "no anchor"). Guards mirror
        observe()'s regular-anchor guards exactly: per-seat + sum
        chips-in-play ceiling, and the P1 sum-floor (armed by
        construction — __init__ enforces the P1 gating). The regular
        internal-consistency check (sum(pre) == visible stacks + pot)
        is an identity for an ante-only frame, so it adds nothing here.
        """
        key = self._hand_key(frame)
        if key in (self._preblind_key, self._preblind_refused_key):
            return
        if not is_preblind_hand_start(frame):
            return
        pre = preblind_pre_hand_stacks(frame)
        ceiling = self._CHIPS_IN_PLAY_CEILING
        over_seats = [
            i for i in range(NUM_SEATS) if int(pre[i]) > ceiling
        ]
        if over_seats or sum(pre) > ceiling:
            detail = (
                ", ".join(f"seat{i+1} pre_hand={pre[i]}"
                          for i in over_seats)
                or f"sum(pre_hand)={sum(pre)}"
            )
            print(f"[ANCHOR-REFUSED] pre-blind hand-start anchor fails "
                  f"chips-in-play ceiling ({ceiling}): {detail}  "
                  f"captured_at={frame.captured_at}", flush=True)
            self._preblind_refused_key = key
            return
        if sum(pre) != ceiling and all(frame.alive):
            print(f"[ANCHOR-REFUSED] pre-blind hand-start anchor fails "
                  f"sum-floor guard: sum(pre_hand)={sum(pre)} != "
                  f"chips-in-play {ceiling} with all {NUM_SEATS} seats "
                  f"alive (no mis-alive seat to hide chips) "
                  f"captured_at={frame.captured_at}", flush=True)
            self._preblind_refused_key = key
            return
        self._preblind_pre_hand = pre
        self._preblind_key = key

    def _maybe_capture_blind_evidence(self, frame: ScraperFrame) -> None:
        """Record single-blind posting evidence (P2 dead-SB guard,
        flag-gated; first capture per hand-key wins).

        At the post-blinds moment of a hand whose start `is_hand_start`
        cannot anchor, exactly ONE blind is visible. The two patterns
        mean opposite things for the recovery's trust argument:

          "sb_only" — one alive seat shows bet == SB and the pot holds
              antes + SB. The BB post is MISSING from bet/pot/stack
              arithmetic: positive evidence for a displaced BB (Ignition
              never plays a dead BB) — the recon's default blind
              assignment is right and the displacement is real.
          "bb_only" — one alive seat shows bet == BB and the pot holds
              antes + BB. No SB was posted: a DEAD-SB hand (SB seat
              busted the previous hand — live 2026-06-12 seq 270-276,
              seat3 bust -> dead SB, BB on seat4) or a displaced SB.
              Either way the recon's next-alive-after-dealer default
              mis-assigns the blinds whenever per-frame dead-SB
              detection has lost the posting evidence (the BB poster
              raised), and the resulting invariant deltas mimic the
              displacement signature with a phantom BB — recovery must
              REFUSE.
        """
        key = self._hand_key(frame)
        if key == self._blind_evidence_key:
            return
        if frame.board:
            return
        sb, bb = int(frame.blinds.sb), int(frame.blinds.bb)
        ante = int(frame.blinds.ante)
        nonzero = [
            (i, int(frame.bet[i])) for i in range(NUM_SEATS)
            if frame.alive[i] and int(frame.bet[i]) != 0
        ]
        if len(nonzero) != 1:
            return
        _, posted = nonzero[0]
        antes_total = sum(frame.alive) * ante
        evidence = None
        if posted == sb and int(frame.pot_total) == antes_total + sb:
            evidence = "sb_only"
        elif posted == bb and int(frame.pot_total) == antes_total + bb:
            evidence = "bb_only"
        if evidence is not None:
            self._blind_evidence = evidence
            self._blind_evidence_key = key

    def blind_posting_evidence_for(self, frame: ScraperFrame) -> str | None:
        """Single-blind posting evidence for this frame's hand-key
        ("sb_only" / "bb_only"), or None. P2 dead-SB guard only."""
        if self._hand_key(frame) != self._blind_evidence_key:
            return None
        return self._blind_evidence

    def bet_closure_anchors_for(
        self, frame: ScraperFrame,
    ) -> tuple[list[tuple[tuple[int, ...], str]], str | None]:
        """Anchors usable by the P2 bet-closure recovery path ONLY.

        Returns (anchors, refused_why):
          anchors     — [(pre_hand, source), ...] for this frame's hand
                        key: the regular hand-start anchor (if tracked)
                        plus the pre-blind anchor (if captured and
                        distinct). Like anchor_for, NO closure check —
                        broken closure is what recovery solves for; the
                        caller must re-validate through the full replay
                        + invariant gate.
          refused_why — audit string when a pre-blind hand-start for
                        this key was REFUSED by the anchor guards (the
                        P2 "P1 refused the anchor" refusal class),
                        else None.
        """
        key = self._hand_key(frame)
        anchors: list[tuple[tuple[int, ...], str]] = []
        if (self._current_pre_hand is not None
                and key == self._current_hand_key):
            anchors.append((self._current_pre_hand, "anchor"))
        if (self._bet_closure_recovery
                and self._preblind_pre_hand is not None
                and key == self._preblind_key
                and not any(pre == self._preblind_pre_hand
                            for pre, _ in anchors)):
            anchors.append((self._preblind_pre_hand, "preblind_anchor"))
        refused_why = None
        if key == self._preblind_refused_key:
            refused_why = ("pre-blind hand-start anchor was refused by "
                           "the anchor guard (ceiling/sum-floor)")
        return anchors, refused_why

    def _closure_plausible(self, frame: ScraperFrame) -> bool:
        """True iff chip conservation against the anchor is plausible
        under strict (or strict+mis-alive) OR UI-lag (or UI-lag+mis-alive)
        patterns.

        Hand-start convention is `strict`: bets are visually shown in front
        of SB/BB but the chips ARE already in pot_total (verified against
        live_1500 line 2: sum(stack)+pot == 9000 exactly, bet_sum is
        redundant).

        Mid-hand UI-lag is `UI-lag`: a fresh bet sits visibly in front of a
        seat and pot_total hasn't yet picked it up. Adding bet_sum to the
        right-hand side closes the conservation.

        Mis-alive overlay (either pattern): originally-alive seats that
        read alive=False now hold invisible chips (scraper bug). The
        residual deficit is bounded by their max possible remaining chips
        (pre_hand - ante each, since they at least paid ante at hand-start).
        """
        seats_at_start = [
            i for i in range(NUM_SEATS) if self._current_pre_hand[i] > 0
        ]
        visible_stack = sum(
            int(frame.stack[i]) for i in seats_at_start
        )
        pot = int(frame.pot_total)
        bet_sum = sum(int(frame.bet[i]) for i in seats_at_start)
        expected_total = sum(self._current_pre_hand)
        ante = int(frame.blinds.ante)
        mis_alive_max_remaining = sum(
            int(self._current_pre_hand[i]) - ante
            for i in seats_at_start
            if not frame.alive[i]
        )

        # Strict (bets already accounted for in pot, as at hand-start).
        deficit_strict = expected_total - visible_stack - pot
        if 0 <= deficit_strict <= mis_alive_max_remaining:
            return True
        # UI-lag (bets sit uncollected in front of seats).
        deficit_uilag = deficit_strict - bet_sum
        if 0 <= deficit_uilag <= mis_alive_max_remaining:
            return True
        return False

    def corrected_pot_for(self, frame: ScraperFrame) -> int | None:
        """If the frame is in the tracked hand but its pot field is
        UI-stale (a new bet is visible in front of a seat that hasn't
        been collected to the pot box yet), return the inferred
        post-collection pot value. Otherwise None (frame's own pot is
        correct, or frame is out-of-hand and uncorrectable).

        Tolerates a positive residual deficit attributable to mis-alive
        seats (Ignition's folded-seat alive=False mis-read). The
        correction is only emitted when the STRICT closure fails BUT the
        UI-lag closure succeeds — at hand-start, where bets are visible
        but their chips are already in the pot, strict closure already
        balances and no correction is emitted.
        """
        if self._current_pre_hand is None:
            return None
        if self._hand_key(frame) != self._current_hand_key:
            return None
        seats_at_hand_start = [
            i for i in range(NUM_SEATS) if self._current_pre_hand[i] > 0
        ]
        visible_stack = sum(
            int(frame.stack[i]) for i in seats_at_hand_start
        )
        pot = int(frame.pot_total)
        bet_sum = sum(
            int(frame.bet[i]) for i in seats_at_hand_start
        )
        expected_total = sum(self._current_pre_hand)
        ante = int(frame.blinds.ante)
        mis_alive_max_remaining = sum(
            int(self._current_pre_hand[i]) - ante
            for i in seats_at_hand_start
            if not frame.alive[i]
        )

        # Strict closure (bets already in pot): no correction needed.
        deficit_strict = expected_total - visible_stack - pot
        if 0 <= deficit_strict <= mis_alive_max_remaining:
            return None

        # UI-lag closure: bets in front haven't been picked up. Fold them in.
        deficit_uilag = deficit_strict - bet_sum
        if (bet_sum > 0
                and 0 <= deficit_uilag <= mis_alive_max_remaining):
            return pot + bet_sum
        return None


# --------------------------------------------------------------------------
# Suspect-frame single-field stack recovery (Layer 1 — 2026-06-09 blackout
# postmortem). The scraper's SanityChecker is all-or-nothing: one impossible
# stack read (stuck OCR digit, e.g. 1110 -> "11101") marks the whole frame
# suspect even when board/pot/controls/bets all parsed cleanly, and a STABLE
# misread defeats polling redundancy — 12 consecutive frames dropped, 4
# hero-to-act moments lost, 38.4s freeze, hero busted. These helpers let the
# bridge derive the one flagged stack from the clean hand-start anchor via
# chip conservation; live_loop re-validates the result through the full
# replay + invariant gate before it can become a decision. The invariant is
# NOT loosened: recovery only adds one derived candidate that must clear the
# same bar every clean frame clears.
# --------------------------------------------------------------------------


# The scraper's stack-jump suspect reason, e.g. "seat1 stack jump 1260->11101".
# Recovery is offered ONLY for this reason shape — any other suspect reason
# (pot jump, card flicker, multi-field) means the frame's corruption is not
# the single-stack class and the frame must drop as before.
_SUSPECT_STACK_JUMP_RE = re.compile(
    r"^seat([1-6])\s+stack\s+jump\s+\d+\s*->\s*\d+$"
)


def recoverable_suspect_seat(record: dict) -> int | None:
    """If this suspect record is recovery-eligible, return the 0-indexed
    seat whose stack the scraper flagged; else None.

    Eligible iff suspect_reasons is a non-empty list, EVERY reason is a
    stack-jump on the SAME seat (the scraper may emit one per polling
    cycle), and no other reason type appears. Multi-seat or non-stack
    reasons -> None (frame-level corruption; drop as before).
    """
    reasons = record.get("suspect_reasons") or []
    if not isinstance(reasons, list) or not reasons:
        return None
    seats = set()
    for r in reasons:
        m = _SUSPECT_STACK_JUMP_RE.match(str(r).strip())
        if not m:
            return None
        seats.add(int(m.group(1)) - 1)
    if len(seats) != 1:
        return None
    return seats.pop()


def _rebuild_frame_with_stack(frame: ScraperFrame, seat: int,
                              new_stack: int) -> ScraperFrame:
    """Return a copy of `frame` with stack[seat] replaced and the derived
    fields (alive, hero_facing_bet) recomputed with parse_frame's exact
    formulas, so a recovered frame is indistinguishable from a cleanly
    parsed one downstream."""
    stack = list(frame.stack)
    stack[seat] = int(new_stack)
    alive = tuple(
        (not frame.empty[i]) and (stack[i] > 0 or frame.bet[i] > 0)
        for i in range(NUM_SEATS)
    )
    if alive[frame.hero_seat]:
        max_opp_bet = max(
            (frame.bet[i] for i in range(NUM_SEATS)
             if i != frame.hero_seat and alive[i]),
            default=0,
        )
        hero_facing_bet = max_opp_bet > frame.bet[frame.hero_seat]
    else:
        hero_facing_bet = False
    return dataclasses.replace(
        frame, stack=tuple(stack), alive=alive,
        hero_facing_bet=hero_facing_bet)


def build_stack_recovery_candidates(
    frame: ScraperFrame, bad_seat: int, pre_hand: tuple[int, ...],
) -> list[tuple[ScraperFrame, str]]:
    """Solve the chip-conservation closure for the single flagged stack.

    One unknown, one equation: with the anchored pre-hand stacks and every
    OTHER field of the frame taken as observed,
        strict:  stack[bad] = Σpre − Σother_stacks − pot
        uilag:   stack[bad] = Σpre − Σother_stacks − pot − Σbets
    (uilag = fresh bets sit visibly in front of seats, pot hasn't picked
    them up yet — same two conventions SessionTracker._closure_plausible
    accepts). When Σbets == 0 the conventions coincide; only strict is
    emitted. When both are emitted the caller must treat survival of BOTH
    through re-validation as ambiguity and drop the frame.

    Mis-alive seats (pre_hand > 0 but reading alive=False with stack 0) are
    NOT special-cased: a genuinely all-in seat contributes 0 to the stack
    sum and the solve stays exact; a scraper mis-alive hides chips and the
    solved value is wrong by exactly the hidden amount — which the replay +
    invariant re-validation then rejects (per-seat stack equality is exact).

    Returns [] when no candidate lies in the feasible range
    (0, pre_hand[bad] − ante]: a mid-hand stack is positive (an all-in seat
    has no decision UI, so a hero-to-act frame can't be recovering to 0)
    and can't exceed pre-hand minus the posted ante.
    """
    seats_at_start = [
        i for i in range(NUM_SEATS) if int(pre_hand[i]) > 0
    ]
    if bad_seat not in seats_at_start:
        return []
    expected_total = sum(int(pre_hand[i]) for i in seats_at_start)
    others_visible = sum(
        int(frame.stack[i]) for i in seats_at_start if i != bad_seat
    )
    pot = int(frame.pot_total)
    bet_sum = sum(int(frame.bet[i]) for i in seats_at_start)
    upper = int(pre_hand[bad_seat]) - int(frame.blinds.ante)

    strict_cand = expected_total - others_visible - pot
    candidates = [(strict_cand, "strict")]
    if bet_sum > 0:
        candidates.append((strict_cand - bet_sum, "uilag"))

    out = []
    for cand, label in candidates:
        if 0 < cand <= upper:
            out.append((_rebuild_frame_with_stack(frame, bad_seat, cand),
                        label))
    return out


# --------------------------------------------------------------------------
# Bet-closure recovery for the displacement signature (P2 — 2026-06-11
# session-3 postmortem, built 2026-06-12). The scraper sometimes renders a
# seat's posted blind/bet folded into its stack (live seq 276: seat5's BB
# read stack=1222 bet=0 pot=560 against the true 1122/100/660). The frame
# parses cleanly and is NOT suspect-flagged — it fails only the strict
# invariant, with a recognizable signature: the scraper-vs-reconstruction
# deltas CLOSE under a single seat's bet/stack transfer (stack reads X
# high, bet reads X low, pot reads X low — or all three reversed). These
# helpers classify that signature and build the corrected frame, with
# chip conservation against the clean hand-start anchor as an independent
# post-patch check; live_loop re-validates the result through the full
# replay + invariant gate before it can become a decision. The invariant
# is NOT loosened: recovery only adds one derived candidate per anchor
# that must clear the same bar every clean frame clears.
# --------------------------------------------------------------------------


_DELTA_STACK_RE = re.compile(r"^stack\[seat([1-6])\]$")
_DELTA_BET_RE = re.compile(r"^bet\[seat([1-6])\]$")


def classify_displacement_deltas(deltas) -> tuple[str, object]:
    """Classify an invariant-fail delta list against the displacement
    signature. `deltas` is InvariantResult.deltas:
    [(field_name, scraper_val, reconstructed_val), ...].

    Returns one of:
      ("not_signature", None)   — not the displacement family at all (any
                                  non-pot/stack/bet delta, or no seat
                                  stack/bet delta). The caller must leave
                                  the frame's drop byte-identical.
      ("refused", why)          — displacement-family deltas that P2 must
                                  refuse (multi-seat, or amounts that do
                                  not close). Caller drops as before,
                                  reason annotated.
      ("closure", (seat, x))    — single-seat closure: seat (0-indexed)
                                  reads `x` chips too HIGH in stack and
                                  `x` too LOW in bet and pot (x < 0 is
                                  the reverse direction). Transferring x
                                  from stack to bet and adding x to pot
                                  reproduces the reconstruction exactly.
    """
    pot_delta = 0
    seats: dict[int, dict[str, int]] = {}
    for name, scraper_val, recon_val in deltas:
        if name == "pot":
            pot_delta = int(scraper_val) - int(recon_val)
            continue
        m = _DELTA_STACK_RE.match(str(name))
        if m:
            seats.setdefault(int(m.group(1)) - 1, {})["stack"] = (
                int(scraper_val) - int(recon_val))
            continue
        m = _DELTA_BET_RE.match(str(name))
        if m:
            seats.setdefault(int(m.group(1)) - 1, {})["bet"] = (
                int(scraper_val) - int(recon_val))
            continue
        # Any other delta type (current_player, street_idx, cards,
        # scraper_self_consistency, legal_actions) — corruption beyond a
        # bet/stack displacement; not this signature.
        return ("not_signature", None)
    if not seats:
        return ("not_signature", None)
    if len(seats) > 1:
        return ("refused",
                f"closure would touch {len(seats)} seats' bet/stack "
                f"pairs (only single-seat displacement is recoverable)")
    seat, d = next(iter(seats.items()))
    x = d.get("stack", 0)
    if x == 0 or d.get("bet", 0) != -x or pot_delta != -x:
        return ("refused",
                "deltas do not close under a single bet/stack transfer "
                f"(stack {d.get('stack', 0):+d}, bet {d.get('bet', 0):+d},"
                f" pot {pot_delta:+d})")
    return ("closure", (seat, x))


def build_bet_closure_candidate(
    frame: ScraperFrame, seat: int, x: int, pre_hand: tuple[int, ...],
) -> tuple[ScraperFrame | None, str]:
    """Apply the single-seat bet/stack transfer (move `x` chips from
    stack[seat] to bet[seat], add `x` to the pot) and hold the result to
    chip conservation against the anchored pre-hand stacks.

    The conservation check is the independent post-patch verification:
    the corrected stacks + corrected pot must reproduce the anchor's
    chip total exactly under the strict (bets-in-pot) convention. A
    frame whose corruption is anything OTHER than a pure displacement
    (UI-lag pot, mis-alive hidden chips, a second bad field) fails here
    or in the caller's replay + invariant re-validation.

    Returns (frame, "ok") or (None, why)."""
    if int(pre_hand[seat]) <= 0:
        return None, (f"seat{seat + 1} not at hand start per the anchor "
                      f"(pre_hand=0)")
    new_stack = int(frame.stack[seat]) - int(x)
    new_bet = int(frame.bet[seat]) + int(x)
    new_pot = int(frame.pot_total) + int(x)
    upper = int(pre_hand[seat]) - int(frame.blinds.ante)
    if not (0 <= new_stack <= upper):
        return None, (f"corrected stack {new_stack} out of range "
                      f"(0..{upper})")
    if seat == frame.hero_seat and new_stack == 0:
        return None, ("corrected hero stack would be 0 (an all-in hero "
                      "has no decision UI)")
    if new_bet < 0:
        return None, f"corrected bet {new_bet} negative"
    if new_pot <= 0:
        return None, f"corrected pot {new_pot} not positive"
    seats_at_start = [
        i for i in range(NUM_SEATS) if int(pre_hand[i]) > 0
    ]
    corrected_stack_sum = sum(
        (new_stack if i == seat else int(frame.stack[i]))
        for i in seats_at_start
    )
    expected_total = sum(int(pre_hand[i]) for i in seats_at_start)
    if corrected_stack_sum + new_pot != expected_total:
        return None, ("corrected frame fails chip conservation against "
                      f"the anchor ({corrected_stack_sum} + {new_pot} != "
                      f"{expected_total})")
    return _rebuild_frame_with_bet_closure(
        frame, seat, new_stack, new_bet, new_pot), "ok"


def _rebuild_frame_with_bet_closure(
    frame: ScraperFrame, seat: int, new_stack: int, new_bet: int,
    new_pot: int,
) -> ScraperFrame:
    """Copy of `frame` with seat's stack/bet and the pot replaced, derived
    fields (alive, hero_facing_bet) recomputed with parse_frame's exact
    formulas — same contract as _rebuild_frame_with_stack."""
    stack = list(frame.stack)
    bet = list(frame.bet)
    stack[seat] = int(new_stack)
    bet[seat] = int(new_bet)
    alive = tuple(
        (not frame.empty[i]) and (stack[i] > 0 or bet[i] > 0)
        for i in range(NUM_SEATS)
    )
    if alive[frame.hero_seat]:
        max_opp_bet = max(
            (bet[i] for i in range(NUM_SEATS)
             if i != frame.hero_seat and alive[i]),
            default=0,
        )
        hero_facing_bet = max_opp_bet > bet[frame.hero_seat]
    else:
        hero_facing_bet = False
    return dataclasses.replace(
        frame, stack=tuple(stack), bet=tuple(bet), pot_total=int(new_pot),
        alive=alive, hero_facing_bet=hero_facing_bet)


# --------------------------------------------------------------------------
# Action-sequence derivation (Phase 2 Piece 2)
# --------------------------------------------------------------------------


class ActionDerivationError(Exception):
    """Action-sequence derivation hit an inconsistency; caller drops the frame
    as ScraperDataQuality (the safe-fallback path)."""


def _detect_blinds_from_bb_post(
    frame: "ScraperFrame", dealer_seat: int, alive_seats: list[int]
) -> tuple[int | None, int | None]:
    """Predicate 1: detect Ignition's dead-SB rotation from posted-blind
    evidence in the frame's `bet` field.

    Walks alive seats clockwise from (dealer + 1) absolute. Tracks the
    first SB poster (bet == sb_amount) and the first BB poster
    (bet == bb_amount) encountered. Stops at the BB poster.

    Returns (sb_seat, bb_seat). Either may be None:
      - (sb, bb) both ints: live-SB confirmed by visible SB+BB posters.
      - (None, bb): dead-SB — the BB poster was found AND no alive seat
        between dealer and BB posted the SB amount AND the absolute
        (dealer+1) % NUM_SEATS seat is empty/busted.
      - (None, None): signal not clean (no visible BB poster, or an
        alive seat between dealer and BB has a bet that's neither 0,
        sb_amount, nor bb_amount — could be raised SB / mid-hand state).
        Caller falls back to the H1 default (next-alive-after-dealer).

    Strict refusal on any unrecognized intervening bet prevents false-
    positive dead-SB on OCR-noisy frames or mid-hand frames where SB
    has acted past their forced post.
    """
    n = NUM_SEATS
    sb_amt = frame.blinds.sb
    bb_amt = frame.blinds.bb
    sb_poster: int | None = None
    bb_poster: int | None = None
    for off in range(1, n + 1):
        c = (dealer_seat + off) % n
        if c not in alive_seats:
            continue
        b = frame.bet[c]
        if b == sb_amt and sb_poster is None and bb_poster is None:
            sb_poster = c
            continue
        if b == bb_amt:
            bb_poster = c
            break
        if b == 0:
            # Alive seat with no current bet — folded or yet-to-act in
            # mid-hand frame. Continue scanning past.
            continue
        # Any other bet (raised, OCR noise, etc.) — signal not clean.
        return None, None
    if bb_poster is None:
        return None, None
    if sb_poster is not None:
        return sb_poster, bb_poster
    sb_candidate_abs = (dealer_seat + 1) % n
    if sb_candidate_abs not in alive_seats:
        return None, bb_poster  # confirmed dead-SB
    # SB-candidate alive but no SB-amount poster seen → ambiguous
    # (SB may have acted past their forced post). Defer to H1.
    return None, None


def _derive_blinds_and_action_order(
    dealer_seat: int,
    alive_seats: list[int],
    *,
    frame: "ScraperFrame | None" = None,
) -> tuple[int | None, int | None, list[int], list[int], int | None]:
    """SINGLE SOURCE OF TRUTH for SB/BB seat assignment AND action order
    under Ignition's forward-moving-button rule.

    Returns (sb_seat, bb_seat, pf_order_alive, pf_order_with_empties,
              postflop_first_seat).
      sb_seat: int seat index, or None for Ignition dead-SB rotation.
      bb_seat: int seat index. Always set for n_alive >= 2.
      pf_order_alive: alive seats in preflop action order, one full lap
        from UTG (= next alive after BB).
      pf_order_with_empties: NUM_SEATS absolute seat indices clockwise
        from UTG (for OpenSpiel emission, which cycles empties as
        forced-fold stack=1 placeholders).
      postflop_first_seat: first-to-act postflop (= SB if alive, else BB).

    When `frame` is provided, applies Predicate 1
    (_detect_blinds_from_bb_post) to detect dead-SB from the bet field.
    When `frame` is None OR Predicate 1 returns ambiguous, falls back to
    H1 (live-SB default: SB = next alive after dealer, BB = next alive
    after SB).

    TRIPWIRE: this function is the ONLY site that resolves SB/BB from
    positional + frame evidence. ALL downstream consumers
    (derive_action_sequence, to_inner_game_string_for_state via
    replay.py) MUST thread the resolved (sb_seat, bb_seat) through and
    MUST NOT recompute independently. Recomputing at a second site
    re-introduces the dead-SB misclassification one layer down — the
    seq=55-class regression mode rolled back on 2026-06-09.

    DEAD BUTTON (D2, 2026-06-12): dealer_seat may be a vacated seat —
    parse_frame admits dealer-on-empty frames behind dead_button_handling.
    Both Predicate 1's walk and the H1 default below walk ABSOLUTE seat
    indices clockwise from dealer+1 and never require the dealer itself
    to be alive, which implements the governing dead-button invariant for
    the common single-elimination case (the BB advanced one active player;
    the button landed on the bust's seat; the next actives post SB/BB) —
    and the posted blinds in `frame.bet` remain the ground truth via
    Predicate 1 (covering double elimination: dead button + dead SB shows
    a bb_only post pattern and resolves sb_seat=None).

    Upstream-guard tripwire (for the dead-SB branch specifically): this
    function relies on parse_frame's upstream filter rejecting sub-4-alive
    frames (heads-up etc.). If that guard is loosened, the n_alive == 2
    branch below becomes reachable; it still assumes a LIVE dealer
    (heads-up cannot have a dead button under the standard rule — the
    button/SB is always one of the two remaining players — but a scraped
    transition frame could violate that and would need handling here).
    """
    n_alive = len(alive_seats)
    if n_alive < 2:
        return (None, None, [], [], None)

    n = NUM_SEATS

    if n_alive == 2:
        # Heads-up: button posts SB. OpenSpiel preflop order [BB, SB]
        # (library convention; see prior preflop_action_order impl).
        sb_seat = dealer_seat
        bb_seat = next(s for s in alive_seats if s != dealer_seat)
        pf_order_alive = [bb_seat, sb_seat]
        return (sb_seat, bb_seat, pf_order_alive,
                pf_order_alive, sb_seat)

    # Default H1: SB = first alive seat walking clockwise from dealer+1,
    # BB = next alive after SB. The walk is over ABSOLUTE seat indices so
    # it tolerates a DEAD BUTTON (dealer on a vacated seat — D2,
    # parse_frame dead_button_handling); for an alive dealer it is
    # arithmetically identical to the old alive_seats-index lookup
    # (alive_seats[(dpos+1) % n_alive] etc.).
    sb_h1: int | None = None
    bb_h1: int | None = None
    for off in range(1, n + 1):
        cand = (dealer_seat + off) % n
        if cand not in alive_seats:
            continue
        if sb_h1 is None:
            sb_h1 = cand
            continue
        bb_h1 = cand
        break
    sb_seat: int | None = sb_h1
    bb_seat: int = bb_h1

    if frame is not None:
        det_sb, det_bb = _detect_blinds_from_bb_post(
            frame, dealer_seat, alive_seats)
        if det_bb is not None:
            sb_seat = det_sb  # may be None (dead-SB) or int (live-SB)
            bb_seat = det_bb

    # UTG = first alive walking forward (absolute) from bb_seat + 1.
    utg_seat = None
    for offset in range(1, n + 1):
        cand = (bb_seat + offset) % n
        if cand in alive_seats:
            utg_seat = cand
            break

    utg_alive_idx = alive_seats.index(utg_seat)
    pf_order_alive = [
        alive_seats[(utg_alive_idx + i) % n_alive] for i in range(n_alive)
    ]
    pf_order_with_empties = [
        (utg_seat + offset) % n for offset in range(n)
    ]
    postflop_first_seat = sb_seat if sb_seat is not None else bb_seat
    return (sb_seat, bb_seat, pf_order_alive,
            pf_order_with_empties, postflop_first_seat)


def preflop_action_order(dealer_seat: int, alive_seats: list[int],
                          *, frame: "ScraperFrame | None" = None
                          ) -> list[int]:
    """Return alive seats in OpenSpiel preflop action order, ONE FULL LAP.

    Without `frame`: defaults to live-SB rotation (matches existing
    behavior for callers that don't have frame data — primarily tests).
    With `frame`: routes through the centralized
    _derive_blinds_and_action_order which applies Predicate 1 dead-SB
    detection. Used by derive_action_sequence to keep the rotation
    consistent with the resolved SB/BB.
    """
    _, _, pf_alive, _, _ = _derive_blinds_and_action_order(
        dealer_seat, alive_seats, frame=frame)
    return pf_alive


def preflop_action_order_with_empties(dealer_seat: int,
                                        alive: tuple[bool, ...],
                                        *,
                                        frame: "ScraperFrame | None" = None
                                        ) -> list[int]:
    """Like preflop_action_order but includes EMPTY seats in their
    absolute clockwise position. OpenSpiel cycles through ALL six seats
    (stack=1 placeholders for empties), so the action sequence must emit
    an action for each empty seat too (forced fold)."""
    alive_seats = [i for i in range(NUM_SEATS) if alive[i]]
    _, _, _, pf_empties, _ = _derive_blinds_and_action_order(
        dealer_seat, alive_seats, frame=frame)
    return pf_empties


def postflop_action_order(dealer_seat: int, alive_seats: list[int],
                           folded: tuple[bool, ...],
                           *, frame: "ScraperFrame | None" = None
                           ) -> list[int]:
    """Return alive-non-folded seats in OpenSpiel postflop action order
    (first-to-act first, then clockwise; folded seats skipped).

    First-to-act is SB if alive, else BB (dead-SB rotation under Ignition's
    forward-moving button). Heads-up: SB acts first postflop (= non-dealer).
    """
    n_alive = len(alive_seats)
    if n_alive < 2:
        return []
    _, _, _, _, first_seat = _derive_blinds_and_action_order(
        dealer_seat, alive_seats, frame=frame)
    first_idx = alive_seats.index(first_seat)
    order = []
    for offset in range(n_alive):
        seat = alive_seats[(first_idx + offset) % n_alive]
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


def _alive_at_hand_start_mask(
        frame: ScraperFrame,
        pre_hand_override: tuple[int, ...] | None,
        ) -> tuple[bool, ...]:
    """Return a length-6 mask of seats that were ALIVE AT HAND-START.

    When override is supplied: derived from `pre_hand_override[i] > 0` —
    matches the alive_seats `to_inner_game_string_for_state` uses to build
    the game string. A seat that had chips at hand-start but went bust
    mid-hand is INCLUDED (Class A-deeper, live dryrun 2026-06-08).
    Without override: identical to `frame.alive` — the simple-model
    derivation produces a pre that's already aligned with frame.alive.
    """
    if pre_hand_override is None:
        return tuple(frame.alive)
    return tuple(int(pre_hand_override[i]) > 0 for i in range(NUM_SEATS))


def _repair_folded_from_chip_deductions(
        frame: ScraperFrame, sb_seat: int, bb_seat: int,
        pre_hand_override: tuple[int, ...] | None = None,
        ) -> tuple[bool, ...]:
    """Re-derive `folded` from per-seat chip deductions to work around
    Ignition's stale folded-field bug.

    When pre_hand_override is supplied, the chip-equality comparison
    operates on PER-SEAT DEDUCTIONS THIS HAND (= pre_hand[i] - stack[i] -
    bet[i]) rather than on raw (stack[i] + bet[i]). This handles
    multi-hand stack accumulation: seats entering the hand with different
    pre-hand stacks (winners and losers from prior hands) commit
    DIFFERENT amounts to (stack+bet) for the same voluntary action, so
    raw stack+bet equality fails — but deductions-this-hand still equal
    each other for non-folders that all called/checked to the same
    preflop level.

    Ignition's scraper doesn't update the per-seat `folded` flag when seats
    fold preflop (verified on live_1500 corpus 2026-06-04 at ~30% rate on
    postflop nominal hero-to-act frames). However, per-seat stack values
    ARE updated reliably — a preflop folder lost only the ante (or
    ante + own forced blind), whereas a non-folder lost the SAME chips as
    every other non-folder (they all matched at the same preflop level,
    whether that's bb for a limped pot or a higher amount for a raised
    pot). The chip pattern distinguishes folders from non-folders.

    Algorithm — STACK-EQUALITY iteration. Among the alive non-folded set,
    all true non-folders should share the same stack value (= pre_hand
    minus the shared deduction `ante + preflop_commit`). Any seat whose
    stack is HIGHER (smaller deduction) committed less and must have
    folded. Iteratively drop the highest-stack candidate (alive non-
    folded, bet==0, not BB) until either:
      (a) max(stack) - min(stack) == 0 across all non-folded (excluding
          BB) — the surviving set is chip-consistent
      (b) only 2 non-folded remain (heads-up postflop; can't drop further)
      (c) no more drop candidates

    BB is EXCLUDED from the equality check because BB-folded and BB-
    checked-option are chip-indistinguishable (both lose exactly ante+bb).
    The scraper's reported folded state for BB carries through unchanged.

    SB IS included: SB folded loses ante+sb, SB completed (called the
    preflop action) loses ante+bb (or more if raised); the difference of
    at least (bb-sb) chips distinguishes them.

    Cases handled correctly (verified by unit tests):
      - Limped pot with non-blind folders (the DOMINANT live_1500 case)
      - Limped pot with SB folded after just posting
      - Preflop raise with non-blind folders
      - All-checked-around preflop (no drops, no-op)
      - Heads-up postflop (early termination)
      - Pre-existing folded flags (treated as already dropped, augmented
        if more chip-pattern folders are detected)

    Returns a 6-tuple of booleans replacing frame.folded; does not modify
    the frame.
    """
    repaired = list(frame.folded)
    # Class A-deeper: under override, use ALIVE-AT-HAND-START so seats
    # that busted mid-hand are still considered for the chip-pattern
    # repair (they DID act this hand — their pre_hand chips are in the
    # pot — and the repair needs to classify them as folded so action-
    # seq emission excludes them from later laps).
    alive_at_start = _alive_at_hand_start_mask(frame, pre_hand_override)
    alive_seats = [i for i in range(NUM_SEATS) if alive_at_start[i]]

    # `chip_signature(i)` returns a value such that LARGER = LESS
    # committed (= more likely to have folded). For the simple-model
    # 1500-baseline case this is stack + bet (a folder kept more chips
    # behind/in front than a non-folder who matched the running bet).
    # With pre_hand_override, the per-seat baseline differs so we use the
    # negative of deductions-this-hand (deduction = pre_hand - stack -
    # bet; larger deduction = paid more = NOT a folder; smaller
    # deduction = paid less = candidate folder). Either way, the equality
    # check (max - min == 0) and the drop rule (max-signature seat) work
    # consistently.
    if pre_hand_override is None:
        def chip_signature(i):
            return frame.stack[i] + frame.bet[i]
    else:
        def chip_signature(i):
            deduction = (int(pre_hand_override[i])
                          - int(frame.stack[i])
                          - int(frame.bet[i]))
            return -deduction

    while True:
        non_folded = [i for i in alive_seats if not repaired[i]]
        if len(non_folded) <= 2:
            break
        # Check (stack+bet) equality among non-folded (excluding BB which
        # is chip-indistinguishable folded vs checked-option).
        nf_for_check = [i for i in non_folded if i != bb_seat]
        if len(nf_for_check) <= 1:
            break
        sigs = [chip_signature(i) for i in nf_for_check]
        if max(sigs) - min(sigs) == 0:
            break  # all non-folded share the same chip signature → self-consistent
        # Drop the highest-signature (least-committed) candidate. Constraints:
        #   - bet==0 (a seat with a visible bet has non-zero chips in
        #     front and can't have folded)
        #   - not BB (BB folded vs BB-checked-option is chip-
        #     indistinguishable)
        #   - not the hero (hero has controls present and hero_cards
        #     visible — they're definitionally in the hand and must
        #     never be dropped; without this guard the chip-equality
        #     heuristic can spuriously flag hero in later levels where
        #     pre-hand stacks vary across seats)
        candidates = [
            i for i in non_folded
            if i != bb_seat and i != frame.hero_seat
            and frame.bet[i] == 0
        ]
        if not candidates:
            break
        to_drop = max(candidates, key=chip_signature)
        repaired[to_drop] = True

    # With pre_hand_override, the per-seat voluntary commits are
    # KNOWN bit-exactly. The standard repair excludes BB from the
    # equality check because BB-folded vs BB-checked-option are
    # chip-indistinguishable under uniform-commit fudge — but with
    # honest per-seat data we CAN distinguish: BB committed less
    # than the rest of the non-folded set ⇒ BB folded (either
    # option-folded preflop or folded on a later street). Promote BB
    # to folded so the action-sequence emitter doesn't try to keep
    # them in the hand with mismatched chips.
    if pre_hand_override is not None and not repaired[bb_seat]:
        non_folded = [i for i in alive_seats if not repaired[i]]
        if len(non_folded) > 1:
            def voluntary(i):
                return (int(pre_hand_override[i])
                        - int(frame.stack[i])
                        - int(frame.bet[i]))  # ante included; relative compare
            others = [voluntary(i) for i in non_folded if i != bb_seat]
            if others and voluntary(bb_seat) < max(others):
                repaired[bb_seat] = True

    return tuple(repaired)


def _derive_pre_hand_and_preflop_commit_simple_model(
        frame: ScraperFrame, sb_seat: int, bb_seat: int,
        pre_hand_override: tuple[int, ...] | None = None,
        ) -> tuple[tuple[int, ...], int]:
    """Same simple-model chip-conservation derivation as
    _derive_pre_hand_simple_model but ALSO returns preflop_commit_per_alive,
    which Piece 5's mid-hand invariant needs for prior-streets-committed
    bookkeeping when converting cumulative-OpenSpiel-contribution to
    current-street-scraper-bet on postflop frames.

    When pre_hand_override is provided (length-6 tuple of per-seat pre-hand
    stacks, typically supplied by a session tracker that observed a recent
    hand-start frame), the simple-model chip-conservation step is bypassed
    in favor of those per-seat values — this handles the multi-hand stack
    accumulation case where seats enter the hand with stacks differing from
    1500 (winners and losers from prior hands).

    Uses _repair_folded_from_chip_deductions to work around Ignition's stale
    `folded` field — the function operates on the repaired folded array, not
    on frame.folded directly. See helper docstring for the algorithm + the
    known limitation around blind-seat folders.

    Returns (pre_hand_stacks: tuple[int,6], preflop_commit_per_alive: int).
    """
    # Class A-deeper: same rationale as in `_repair_folded_from_chip_deductions`
    # and `derive_action_sequence` — under override, use ALIVE-AT-HAND-START
    # so the chip-arithmetic accounts for seats that busted mid-hand.
    alive_at_start = _alive_at_hand_start_mask(frame, pre_hand_override)
    alive_seats = [i for i in range(NUM_SEATS) if alive_at_start[i]]
    n_alive = len(alive_seats)
    ante = frame.blinds.ante
    sb = frame.blinds.sb
    bb = frame.blinds.bb

    folded = _repair_folded_from_chip_deductions(
        frame, sb_seat, bb_seat,
        pre_hand_override=pre_hand_override)

    folded_commit_total = 0
    folded_seats = [i for i in alive_seats if folded[i]]
    for i in folded_seats:
        folded_commit_total += ante
        if i == sb_seat:
            folded_commit_total += sb
        elif i == bb_seat:
            folded_commit_total += bb

    alive_non_folded = [i for i in alive_seats if not folded[i]]
    n_anf = len(alive_non_folded)

    if pre_hand_override is not None:
        # Session-tracked path. Derive per-seat voluntary commitments
        # from the override; the simple uniform-commit model applies to
        # non-busted seats. BUSTED-MID-HAND seats (alive_at_start AND
        # currently alive=False, e.g. an all-in-for-less BB that lost
        # the side pot) commit a DIFFERENT amount than the uniform
        # caller-set — that's exactly what their bust represents.
        # Exclude them from the equality check; their commitment is
        # accounted for separately via the all-in emission in
        # derive_action_sequence.
        busted_mid_hand = [
            i for i in alive_non_folded
            if int(pre_hand_override[i]) > 0 and not frame.alive[i]
        ]
        equality_check_seats = [
            i for i in alive_non_folded if i not in busted_mid_hand
        ]
        if not equality_check_seats:
            preflop_commit_per_alive = 0
        else:
            per_seat_voluntary = [
                int(pre_hand_override[i]) - int(frame.stack[i])
                - ante - int(frame.bet[i])
                for i in equality_check_seats
            ]
            if min(per_seat_voluntary) != max(per_seat_voluntary):
                raise ActionDerivationError(
                    f"override-derived per-seat voluntary commitments "
                    f"differ across alive non-folded non-busted seats "
                    f"({dict(zip(equality_check_seats, per_seat_voluntary))}) "
                    f"— frame chip-arithmetic is internally inconsistent "
                    f"(likely a scraper stale-folded-flag case)")
            preflop_commit_per_alive = max(0, per_seat_voluntary[0])
        # Override-supplied pre-hand is used directly.
        return tuple(int(x) for x in pre_hand_override), preflop_commit_per_alive

    if n_anf == 0:
        preflop_commit_per_alive = 0
        remainder = 0
    else:
        bet_sum_anf = sum(frame.bet[i] for i in alive_non_folded)
        residual = (frame.pot_total - folded_commit_total
                     - n_anf * ante - bet_sum_anf)
        preflop_commit_per_alive = max(0, residual // n_anf)
        remainder = max(0, residual - preflop_commit_per_alive * n_anf)

    # Non-zero remainder = simple-model can't represent the alive
    # seats' commitments with a single uniform preflop_commit_per_alive
    # value. Distributing +1 chips across seats would produce sub-min-
    # raise chip_ints that OpenSpiel rejects. Reject as data quality:
    # we can't reconstruct this state honestly with our simple model.
    if remainder > 0:
        raise ActionDerivationError(
            f"simple-model preflop_commit remainder {remainder} "
            f"(residual={preflop_commit_per_alive * n_anf + remainder}, "
            f"n_anf={n_anf}) — alive seats committed unequal preflop "
            f"amounts that the uniform-commit model cannot represent")

    pre = []
    for i in range(NUM_SEATS):
        if not frame.alive[i]:
            pre.append(0)
        elif folded[i]:
            # sb_seat may be None under dead-SB rotation. Use explicit
            # None-check rather than relying on `i == None == False`.
            if sb_seat is not None and i == sb_seat:
                blind_amt = sb
            elif i == bb_seat:
                blind_amt = bb
            else:
                blind_amt = 0
            pre.append(frame.stack[i] + ante + blind_amt)
        else:
            pre.append(frame.stack[i] + ante
                        + preflop_commit_per_alive
                        + frame.bet[i])
    return tuple(pre), preflop_commit_per_alive


def _derive_pre_hand_simple_model(frame: ScraperFrame,
                                    sb_seat: int, bb_seat: int,
                                    pre_hand_override: tuple[int, ...] | None = None,
                                    ) -> tuple[int, ...]:
    """Pre-hand stacks (chips at hand START) via the simple-model chip-
    conservation derivation. See
    _derive_pre_hand_and_preflop_commit_simple_model for the algorithm + the
    simple-model assumptions. This wrapper drops the preflop_commit return
    value; callers that need it (Piece 5 invariant) use the longer name.

    pre_hand_override (if given) bypasses the simple-model and uses the
    caller-supplied per-seat pre-hand stacks directly.
    """
    pre, _ = _derive_pre_hand_and_preflop_commit_simple_model(
        frame, sb_seat, bb_seat, pre_hand_override=pre_hand_override)
    return pre


def derive_action_sequence(frame: ScraperFrame,
                            pre_hand_override: tuple[int, ...] | None = None,
                            ) -> list[tuple[int, int]]:
    """Derive the canonical (seat_idx, openspiel_chip_int) sequence to walk
    OpenSpiel state from new_initial_state() to the hero's current decision.

    pre_hand_override (length-6 tuple, optional): caller-supplied per-seat
    pre-hand stacks for the current hand. When provided, bypasses the
    simple-model chip-arithmetic derivation — used by session-aware
    callers that have observed a hand-start frame and recorded per-seat
    pre-hand stacks (handles multi-hand stack accumulation and rounding
    cases the uniform-commit simple model can't represent).

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
    # Class A-deeper: when override is supplied, use the SAME alive_seats
    # definition the game-string builder uses (= pre_hand_override[i] > 0)
    # so SB/BB/UTG and preflop order match OpenSpiel's view. A seat that
    # had chips at hand-start but went bust mid-hand is in the rotation
    # per OpenSpiel — we must include them here too. See
    # replay.py:replay_to_decision for the matching comment.
    if pre_hand_override is not None:
        alive_seats = [
            i for i in range(NUM_SEATS) if int(pre_hand_override[i]) > 0
        ]
    else:
        alive_seats = [i for i in range(NUM_SEATS) if frame.alive[i]]
    n_alive = len(alive_seats)
    if n_alive < 2:
        raise ActionDerivationError(
            f"n_alive={n_alive} < 2; no hand possible")

    # SINGLE SOURCE OF TRUTH for SB/BB: routed through the centralizer
    # which applies Predicate 1 (dead-SB detection from posted-blind
    # evidence in `frame.bet`). sb_seat may be None under Ignition's
    # forward-moving-button dead-SB rotation.
    sb_seat, bb_seat, _, _, _ = _derive_blinds_and_action_order(
        frame.dealer_seat, alive_seats, frame=frame)

    street_idx = _street_idx_from_board(frame.board)

    # Repaired folded array — works around Ignition's stale `folded` field
    # by re-deriving from per-seat stack deductions. See
    # _repair_folded_from_chip_deductions docstring.
    folded = _repair_folded_from_chip_deductions(
        frame, sb_seat, bb_seat,
        pre_hand_override=pre_hand_override)

    # Per-seat pre-hand stacks via simple-model chip conservation. Works
    # for both hand-start (P_max=0) and postflop frames. See helper docstring
    # for the simple-model assumption and its failure modes.
    pre = _derive_pre_hand_simple_model(
        frame, sb_seat, bb_seat, pre_hand_override=pre_hand_override)

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

    ante = frame.blinds.ante
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
            if folded[seat]:
                preflop_commit[seat] = max(
                    0, total_committed[seat] - ante)

    # PREFLOP action emission — multi-turn aware.
    #
    # Walks the OpenSpiel preflop order in cycles, tracking per-seat
    # cur_commit and running_max. Emits actions that bring each seat
    # toward its FINAL target (= frame.bet[seat]). Continues past hero's
    # FIRST turn when subsequent seats re-open action (raise), so the
    # state OpenSpiel ends at correctly represents hero's actual current
    # decision — facing the re-raise, not facing the initial blinds.
    #
    # Stop conditions (emit nothing further; OpenSpiel state at this
    # point IS hero's current decision):
    #   - hero is encountered AND cur_commit[hero] < running_max
    #     (hero faces a bet/raise — to act)
    #   - hero is BB AND cur_commit[BB] == bb_amount AND
    #     running_max == bb_amount AND no voluntary raise happened
    #     (BB option after limp-around)
    #
    # Empty seats: forced fold ONCE on first visit (OpenSpiel cycles
    # through stack=1 placeholders; the fold removes them from the cycle).
    # Folded alive seats: fold ONCE on first visit (chip-deduction repair
    # determines fold status). Cycle ends naturally when all alive
    # non-folded seats have cur_commit == running_max (round closes) or
    # when we exhaust the safety cap.
    actions: list[tuple[int, int]] = []
    # Class A-deeper: pass alive_at_hand_start so the with-empties walk
    # rotates over the SAME seats as OpenSpiel's firstPlayer (which is
    # built from pre_hand_override-derived alive in
    # `to_inner_game_string_for_state`). A seat that busted mid-hand
    # remains in the rotation per OpenSpiel; we must walk over them too.
    alive_at_start = _alive_at_hand_start_mask(frame, pre_hand_override)
    # Pass frame= so the helpers route through Predicate 1 — guaranteeing
    # the action-order uses the SAME (sb_seat, bb_seat) resolved above.
    pf_order_full = preflop_action_order_with_empties(
        frame.dealer_seat, alive_at_start, frame=frame)
    pf_order_alive = preflop_action_order(
        frame.dealer_seat, alive_seats, frame=frame)
    if not pf_order_alive:
        # Heads-up degenerate — already guarded by n_alive < 2 earlier.
        return actions

    bb_amount = frame.blinds.bb
    sb_amount = frame.blinds.sb
    cur_commit = [0] * NUM_SEATS
    # sb_seat is None under Ignition's dead-SB rotation; no seat posted SB.
    if sb_seat is not None:
        cur_commit[sb_seat] = sb_amount
    cur_commit[bb_seat] = bb_amount
    running_max = bb_amount
    raise_above_bb = False  # any voluntary raise above the BB?
    # The MAX chip_int emitted during preflop = OpenSpiel's preflop spent
    # at the end of preflop, identical for all non-folded seats since
    # they all reach the running-max (via call/raise/all-in). For uniform
    # raises this equals preflop_commit_per_alive; for forced all-in
    # raises (busted-mid-hand) it equals the busted seat's pre_hand
    # (which is > preflop_commit_per_alive by exactly the ante because
    # chip_int=pre_hand bakes the ante in). Used by the postflop emission
    # to compute chip_int = preflop_spent + current-street-voluntary;
    # using preflop_commit_per_alive instead would understate by ante.
    preflop_max_chip_int = bb_amount

    # Per-seat preflop target chip-int (final cumulative voluntary
    # commitment ON THE PREFLOP STREET). For preflop frames this is
    # frame.bet[seat]; for postflop frames it's the preflop_commit value
    # the simple model derived above (= total_committed - current_street
    # - ante). For blind seats, ensure the target floors at the blind
    # amount the seat was forced to post.
    target = [0] * NUM_SEATS
    for seat in range(NUM_SEATS):
        if not frame.alive[seat]:
            continue
        commit = int(preflop_commit[seat])
        if seat == sb_seat:
            target[seat] = max(sb_amount, commit)
        elif seat == bb_seat:
            target[seat] = max(bb_amount, commit)
        else:
            target[seat] = commit

    empties_folded: set[int] = set()
    folded_emitted: set[int] = set()
    has_acted_once: set[int] = set()  # voluntary action emitted at least once

    # "Delayed fold" seats: in the repaired folded mask but with a
    # non-zero preflop voluntary commit (= they limped/called preflop,
    # then folded on a later street). For OpenSpiel chip arithmetic to
    # close at hero's decision, we MUST emit their preflop call before
    # the fold — otherwise their contribution falls short by their
    # voluntary commit amount. Treat them as alive non-folded during
    # the preflop emission loop; they'll naturally limp/call to their
    # target then defensive-fold when running_max exceeds their target.
    # Immediate-fold seats (folded with voluntary == 0) keep the
    # original "fold once on first visit" behavior.
    delayed_fold_seats: set[int] = {
        s for s in alive_seats
        if folded[s] and preflop_commit[s] > 0
    }

    # Class A-deeper: BUSTED-MID-HAND seats — seats that were alive at
    # hand-start (in pre_hand_override) but currently read alive=False
    # (Ignition zeroes out the busted seat's stack AND bet). OpenSpiel
    # built them into the rotation with their pre-hand stack value, so we
    # must emit an action that commits their full chips. Emit chip_int =
    # pre[seat] (= full pre-hand stack value) as an all-in raise; that
    # makes OpenSpiel set spent[seat] = pre[seat]. After this one
    # emission they're absorbed and out of the rotation; mark them in
    # folded_emitted so the loop skips them on subsequent visits.
    busted_mid_hand: set[int] = set()
    if pre_hand_override is not None:
        busted_mid_hand = {
            i for i in range(NUM_SEATS)
            if int(pre_hand_override[i]) > 0 and not frame.alive[i]
        }

    def is_active_for_emission(s: int) -> bool:
        """True iff seat is in the rotation AND not yet finalized in
        OpenSpiel (= not in folded_emitted, regardless of folded mask).
        Uses alive_at_start so busted-mid-hand seats are emitted-for
        once before being skipped via folded_emitted."""
        if not alive_at_start[s]:
            return False
        if s in folded_emitted:
            return False
        if folded[s] and s not in delayed_fold_seats:
            return False
        return True

    def round_closed() -> bool:
        # Round closes when every active-for-emission seat has emitted a
        # voluntary action AND their cur_commit matches running_max.
        for s in pf_order_alive:
            if not is_active_for_emission(s):
                continue
            if s not in has_acted_once:
                return False
            if cur_commit[s] != running_max:
                return False
        return True

    # First lap walks the full-with-empties order so OpenSpiel's
    # forced-fold cycle for stack=1 placeholders is respected. After the
    # first lap, subsequent laps cycle through alive non-folded seats
    # only (empties already folded out).
    MAX_VISITS = NUM_SEATS * 4  # 4 laps safety cap
    visit = 0
    on_first_lap = True
    first_lap_idx = 0

    while visit < MAX_VISITS:
        if on_first_lap:
            if first_lap_idx >= len(pf_order_full):
                on_first_lap = False
                continue
            seat = pf_order_full[first_lap_idx]
            first_lap_idx += 1
        else:
            # Subsequent laps: cycle active-for-emission seats only
            # (= alive seats not yet folded in OpenSpiel — includes
            # delayed-fold seats that haven't emitted their fold yet).
            if round_closed():
                break
            alive_remaining = [
                s for s in pf_order_alive
                if is_active_for_emission(s)
            ]
            if not alive_remaining:
                break
            # The visit counter is unique per loop turn; mod into the
            # cycle to pick the next seat.
            seat = alive_remaining[(visit - len(pf_order_full)) %
                                    len(alive_remaining)]
        visit += 1

        # Busted-mid-hand seat (Class A-deeper): emit all-in chip_int =
        # pre[seat] once; OpenSpiel sets spent=pre_hand, absorbing all
        # their committed chips into the pot. The all-in raises running_max
        # (in voluntary terms) to (pre - ante), prompting other still-in
        # seats to call/match on subsequent laps. After this emission the
        # busted seat is out of the rotation; folded_emitted skips them
        # on subsequent visits.
        if seat in busted_mid_hand:
            if seat not in folded_emitted:
                # Discriminator: did the busted seat's all-in act AS a
                # raise (their stack > current running_max ⇒ they raise),
                # or AS a call-for-less (someone already raised above
                # their stack ⇒ they can only call all-in)?
                busted_voluntary = int(pre[seat]) - ante
                if busted_voluntary > running_max:
                    # All-in raise. chip_int = pre_hand (= the only legal
                    # raise when remaining < min-raise increment).
                    actions.append((seat, int(pre[seat])))
                    running_max = busted_voluntary
                    raise_above_bb = True
                    if int(pre[seat]) > preflop_max_chip_int:
                        preflop_max_chip_int = int(pre[seat])
                else:
                    # All-in call for less. chip_int=1 (call); OpenSpiel
                    # caps the seat's spent at their stack.
                    actions.append((seat, 1))
                folded_emitted.add(seat)
                has_acted_once.add(seat)
            continue

        # Empty seat: forced fold once. Uses alive_at_start to
        # distinguish a truly-busted-at-hand-start seat (stack=1
        # placeholder in the game string) from a busted-mid-hand seat
        # (handled above).
        if not alive_at_start[seat]:
            if seat not in empties_folded:
                actions.append((seat, 0))
                empties_folded.add(seat)
            continue

        # Folded alive seat: fold immediately only if their voluntary
        # commit was zero (= just ante, no preflop action). Delayed-fold
        # seats (folded with voluntary > 0) fall through to the regular
        # emission flow; they'll limp/call to their target and
        # defensive-fold when running_max later exceeds it.
        if folded[seat] and seat not in delayed_fold_seats:
            if seat not in folded_emitted:
                actions.append((seat, 0))
                folded_emitted.add(seat)
            continue

        # Hero stop checks (preflop frames only). The discriminator: is
        # hero at their final preflop commit, AND facing a running_max
        # above it? Stop only then. If hero hasn't reached their target
        # yet, emit their action toward target.
        if seat == frame.hero_seat and street_idx == 0:
            t = target[seat]
            cur = cur_commit[seat]
            if seat in has_acted_once:
                # Hero already had a voluntary turn this hand.
                if cur >= t and running_max > cur:
                    # Hero already at target AND faces a raise above it
                    # → to act on re-opened action. Stop.
                    return actions
                # Else: hero hasn't reached target yet OR running_max
                # has not advanced past their target — continue emitting
                # actions toward target.
            else:
                # Hero's first voluntary turn this hand.
                if t <= cur:
                    # No chips to move beyond the auto-posted blind →
                    # hero's first decision is at this point.
                    return actions
                # Else: target > cur, hero acted in this hand and has
                # chips to commit — fall through and emit.

        # Decide the action for this seat this visit.
        t = target[seat]
        # "Needs to act" = either hasn't taken a voluntary action yet,
        # OR has taken one but cur_commit < running_max (re-opened).
        needs_to_act = (seat not in has_acted_once
                        or cur_commit[seat] < running_max)
        if not needs_to_act:
            continue

        if t < running_max:
            # Distinguish (a) truly-folded seat (no voluntary commit at
            # all, or a partial commit below all-in) from (b) all-in-for-
            # less seat that committed every chip they had below the
            # running_max. For (b), emit chip_int=1 (call); OpenSpiel
            # caps the seat's spent at their remaining stack so the
            # committed chips land correctly in the pot. For (a), the
            # original defensive-fold (chip_int=0) stays.
            #
            # Discriminator: pre[s] - ante == t (= they committed all
            # available chips to this target) AND t > 0 (excludes
            # ante-only contributions) AND frame.stack[s] == 0 (no
            # chips behind).
            #
            # NOTE: this is the surgical fix for the all-in-for-less
            # case (seq=246 SB, audit 2026-06-09). The earlier proposal
            # to broaden busted_mid_hand was rejected because it would
            # have re-routed alive-but-all-in raising seats (seq=97,
            # seq=170, seq=428 etc.) through the chip_int=pre_hand
            # convention, regressing the Class B chip_int=bet contract
            # those seats currently satisfy.
            is_all_in_for_less = (
                int(pre[seat]) - ante == int(t)
                and int(t) > 0
                and int(frame.stack[seat]) == 0
            )
            actions.append((seat, 1 if is_all_in_for_less else 0))
            folded_emitted.add(seat)
            continue

        if t > running_max:
            # Potential raise. Defer the raise if either:
            #   (a) a smaller raise above running_max is pending from
            #       an unemitted seat — they raise first (preserves
            #       chronological order on multi-raise rounds);
            #   (b) an unemitted seat AFTER me in pf_order_alive wants
            #       to call/limp at the current running_max (target
            #       equals running_max) AND there is another raiser-
            #       eligible seat after me to fire the raise instead.
            #       This handles "delayed-fold limper" cases: e.g., BTN
            #       limped preflop before SB raised; if I (UTG/hero)
            #       raise immediately, BTN's limp never gets emitted
            #       and chip arithmetic falls short.
            def _comes_after(s_other: int) -> bool:
                idx_me = pf_order_alive.index(seat)
                return s_other in pf_order_alive[idx_me + 1:]

            # Predicate 2 (seq=246, refined post-seq=428 audit 2026-06-09):
            # only defer to a seat whose max possible voluntary commit
            # (= pre[s] - ante) is STRICTLY GREATER than their current
            # target — i.e., they could keep raising past their target
            # if they wanted. A seat with pre-ante == target is fully
            # committed at target (all-in-for-less at that level); their
            # commit is fixed and chronologically subordinate to a
            # larger raise above their target. Deferring to them is
            # wrong because they can't actually raise further.
            #
            # seq=246 SB: pre=630, ante=15, target=615, pre-ante==target
            #   → don't defer (correct; SB is all-in for less below BTN).
            # seq=428 hero: pre=1511, ante=30, target=1200, pre-ante=1481
            #   > target=1200 → defer (correct; hero 3-bet partially, has
            #   281 chips left, their 1200 commit IS chronologically
            #   before UTG's 4-bet to 1938).
            # Legit multi-raise (MP pre=2000, target=200): pre-ante=1990
            #   > 200 → defer (correct; MP's 200 limp/call IS before
            #   the larger raiser's lap-2 re-raise).
            defer_for_smaller_raise = any(
                target[s] > running_max and target[s] < t
                and (int(pre[s]) - ante > int(target[s]))
                for s in pf_order_alive
                if s != seat and is_active_for_emission(s)
                and (s not in has_acted_once
                     or cur_commit[s] < running_max)
            )
            # A "limper" is a seat that voluntarily completed to the BB. The
            # BB sitting at exactly bb_amount before any voluntary raise is
            # NOT a limper — they are still in their forced post. Treating
            # the BB as a limper here causes early-position raisers (UTG,
            # MP) to defer their open and emit chip_int=1 (call) instead of
            # chip_int=raise_target, mis-attributing the raise to a later
            # seat (live_dryrun 2026-06-08 seq=170: UTG opens 100, bridge
            # emitted seat 2 chip_int=1 and seat 4 chip_int=100, off by
            # exactly 50 chips on UTG's contribution).
            #
            # Exclude the BB if their target == bb_amount AND raise_above_bb
            # is False (no voluntary raise has happened yet). SB at exactly
            # sb_amount can never satisfy target[s] == running_max here
            # (sb_amount < running_max = bb_amount in the un-raised case),
            # so SB needs no special exclusion — but symmetric exclusion is
            # added for completeness in case future code paths break that
            # invariant (e.g., heads-up where SB is the dealer).
            def _is_blind_still_at_forced_post(s: int) -> bool:
                if not raise_above_bb:
                    if s == bb_seat and target[s] == bb_amount:
                        return True
                    # sb_seat may be None under dead-SB; no seat posted SB.
                    if (sb_seat is not None
                            and s == sb_seat
                            and target[s] == sb_amount):
                        return True
                return False

            limper_after_me_unemitted = any(
                target[s] == running_max
                and s not in has_acted_once
                and _comes_after(s)
                and not _is_blind_still_at_forced_post(s)
                for s in pf_order_alive
                if s != seat and is_active_for_emission(s)
            )
            raiser_after_me_exists = any(
                target[s] > running_max
                and (s not in has_acted_once
                     or cur_commit[s] < running_max)
                and _comes_after(s)
                for s in pf_order_alive
                if s != seat and is_active_for_emission(s)
            )
            # Class A-deeper: if a busted-mid-hand seat AFTER me in
            # pf_order_alive will all-in to at least my target, they are
            # the implicit raiser — I should call/limp now and let them
            # raise on their turn (preserves chronological order: callers
            # act, then the all-in raise, then callers match the all-in
            # on subsequent laps).
            busted_raiser_after_me = any(
                s in busted_mid_hand
                and (int(pre[s]) - ante) >= t
                and _comes_after(s)
                for s in pf_order_alive
            )
            should_defer = (
                defer_for_smaller_raise
                or (limper_after_me_unemitted and raiser_after_me_exists)
                or busted_raiser_after_me
            )

            if should_defer:
                if seat not in has_acted_once:
                    actions.append((seat, 1))
                    cur_commit[seat] = max(cur_commit[seat], running_max)
                    has_acted_once.add(seat)
                continue
            # Raise now.
            actions.append((seat, t))
            cur_commit[seat] = t
            running_max = t
            raise_above_bb = True
            has_acted_once.add(seat)
            if t > preflop_max_chip_int:
                preflop_max_chip_int = t
            continue

        # t == running_max: call/check.
        actions.append((seat, 1))
        cur_commit[seat] = t
        has_acted_once.add(seat)

    # Preflop emission complete. For PREFLOP frames the function is done.
    if street_idx == 0:
        return actions

    # POSTFLOP: emit intermediate streets (all-checks) then current street
    # with multi-turn aware re-opened-action support, same shape as
    # preflop. For each postflop street we walk a postflop order
    # (SB-first → BTN), cycling until round closes or hero is to act.
    # Use the post-preflop folded set (= original folded mask ∪ any
    # delayed-fold seats that emitted defensive-fold during preflop)
    # so seats that limped-then-folded preflop are correctly excluded
    # from postflop action order.
    post_preflop_folded = tuple(
        bool(folded[i] or i in folded_emitted) for i in range(NUM_SEATS)
    )
    pf_alive_post = postflop_action_order(
        frame.dealer_seat, alive_seats, post_preflop_folded,
        frame=frame)

    # Intermediate streets: each closes with all checks (simple-model).
    for _ in range(1, street_idx):
        for seat in pf_alive_post:
            actions.append((seat, 1))  # check

    # Current postflop street: per-seat current-street targets come from
    # frame.bet directly (this-street commits). Walk in cycles, allowing
    # hero to take a first action and then return for a re-opened raise.
    cs_target = [int(frame.bet[s]) for s in range(NUM_SEATS)]
    cs_cur = [0] * NUM_SEATS
    cs_has_acted: set[int] = set()
    # cs_folded: seats that defensive-folded on the current postflop
    # street. Tracked as a set rather than removed from pf_alive_post so
    # the cycle iteration's modular index stays stable across removals.
    # Mirrors the preflop loop's folded_emitted pattern; mutating
    # pf_alive_post mid-loop shifts indices and causes the iterator to
    # skip past hero (seq=101 hero=BTN regression, 2026-06-09).
    cs_folded: set[int] = set()
    running_max_cs = 0
    # cumulative chip int for OpenSpiel raise = preflop_commit + cs_target
    raise_happened_cs = False

    def cs_round_closed() -> bool:
        # Round closes when every alive non-folded seat (excluding seats
        # that defensive-folded this street) has emitted a voluntary
        # action on this street AND their cs_cur matches running_max_cs.
        for s in pf_alive_post:
            if s in cs_folded:
                continue
            if s not in cs_has_acted:
                return False
            if cs_cur[s] != running_max_cs:
                return False
        return True

    MAX_CS_VISITS = max(1, len(pf_alive_post)) * 4
    cs_visit = 0
    cs_pos = 0
    while (cs_visit < MAX_CS_VISITS and pf_alive_post
           and len(cs_folded) < len(pf_alive_post)):
        if cs_round_closed():
            break
        seat = pf_alive_post[cs_pos % len(pf_alive_post)]
        cs_pos += 1
        cs_visit += 1
        # Skip seats that have folded on this street (cycle position
        # preserved; index modulo doesn't shift).
        if seat in cs_folded:
            continue

        if seat == frame.hero_seat:
            # Hero on the current postflop street.
            if seat in cs_has_acted:
                # Re-opened action.
                if cs_cur[seat] < running_max_cs:
                    return actions
                # Else: matched and waiting; continue cycling.
            else:
                if cs_target[seat] > cs_cur[seat]:
                    # Hero acted (committed chips on this street) but
                    # we haven't emitted that yet → fall through and
                    # emit hero's first action.
                    pass
                else:
                    # Hero hasn't moved chips. STOP iff no LATER seat
                    # will bet — otherwise hero's first action was a
                    # CHECK and a later seat raised (re-opens for hero).
                    later_has_bet = any(
                        cs_target[s] > running_max_cs
                        for s in pf_alive_post
                        if s != seat and s not in cs_folded
                    )
                    if not later_has_bet:
                        # Hero's true first decision — to check/bet.
                        return actions
                    # Else: hero will face a bet later → emit check
                    # now, continue cycling. Hero stops on a later
                    # visit when cs_cur < running_max_cs.

        t = cs_target[seat]
        needs_to_act = (seat not in cs_has_acted
                        or cs_cur[seat] < running_max_cs)
        if not needs_to_act:
            continue

        if t < running_max_cs:
            # Final cs commit below running max → seat folded
            # (defensive). On postflop we treat as a silent fold the
            # scraper didn't catch (e.g., repaired_folded missed a
            # blind-seat fold). Track in cs_folded (NOT remove from
            # pf_alive_post) so the cycle index stays stable.
            actions.append((seat, 0))
            cs_folded.add(seat)
            continue

        if t > running_max_cs:
            # Defer if a LATER unacted seat has a smaller cs_target
            # above running_max (the smaller one raised first).
            unacted_eligible = [
                cs_target[s] for s in pf_alive_post
                if s not in cs_folded
                and (s not in cs_has_acted
                     or cs_cur[s] < running_max_cs)
                and cs_target[s] > running_max_cs
            ]
            if unacted_eligible and t != min(unacted_eligible):
                if seat not in cs_has_acted:
                    # Emit check/call (= match current running_max_cs).
                    actions.append((seat, 1))
                    cs_cur[seat] = max(cs_cur[seat], running_max_cs)
                    cs_has_acted.add(seat)
                continue
            # Raise now. chip_int (= OpenSpiel cumulative spent target) =
            # preflop_spent (= preflop_max_chip_int, the max chip_int
            # emitted during preflop) + current-street voluntary commit t.
            # For uniform preflop raises preflop_max_chip_int ==
            # preflop_commit_per_alive (= preflop_commit[seat]); for
            # forced all-in preflop raises (busted-mid-hand) it is larger
            # by the ante. Using preflop_max_chip_int is correct in both
            # cases.
            actions.append((seat, preflop_max_chip_int + t))
            cs_cur[seat] = t
            running_max_cs = t
            raise_happened_cs = True
            cs_has_acted.add(seat)
            continue

        # t == running_max_cs: call/check
        actions.append((seat, 1))
        cs_cur[seat] = t
        cs_has_acted.add(seat)

    return actions
