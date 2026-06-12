"""Session-jsonl -> structured per-hand records (track H3).

WHAT THE FRAMES ACTUALLY SUPPORT (be honest; mark unknown, don't guess):

The live listener logs one row per scraper frame. Each row carries parsed
summary fields (seq, blinds [sb,bb,ante], level, dealer_seat, hero_cards,
board, pot_total, status, client_action, ...) and — in every session from
2026-06-08 15:27 onward — the full `raw_record` with per-seat
stacks/bets/folded/empty, the dealer string, and the suspect flag.

Frames are SNAPSHOTS sampled every few seconds, not an action feed.
Consequences, all deliberate in this parser:

  * Opponent actions are DERIVED by diffing consecutive snapshots within a
    hand (the same chip-arithmetic primitives the live bridge uses:
    `is_hand_start` / `pre_hand_stacks` / `_derive_blinds_and_action_order`
    are imported from scraper_schema). A bet level that rises between two
    frames is one observed "chips-in" event; multiple physical actions in
    one sampling gap collapse into one event. Coverage is therefore a
    LOWER BOUND on true action frequency.
  * CHECKS are invisible (no chips move) — never recorded.
  * FOLDS come from the scraper `folded` flag transitions, which Ignition
    leaves stale ~30% of the time — folds are undercounted. observed_via
    records the evidence class for every event.
  * OPPONENT HOLE CARDS are not in the scraper schema at all (only
    hero_cards exists). Opponent showdown holdings are structurally
    unobservable; the showdowns table stays empty for opponents and we
    say so rather than inferring.
  * Hand OUTCOME (net chips per seat) is derivable only when this hand AND
    the next hand both have a clean hand-start anchor; otherwise unknown.
  * Sessions logged before raw_record existed (2026-06-08 morning logs)
    ingest as summary-tier: hands segmented from top-level fields, but no
    per-seat opponent actions.

Anonymity: seats are keyed (session_id, seat) only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.nlhe.integration.scraper_schema import (
    NUM_SEATS,
    ScraperDataQuality,
    ScraperParseError,
    ScraperSuspect,
    _derive_blinds_and_action_order,
    is_hand_start,
    parse_frame,
    pre_hand_stacks,
)

PARSER_VERSION = "1.0"

_STREET_BY_BOARD_LEN = {0: 0, 3: 1, 4: 2, 5: 3}
STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}

# Frame rows that are not table snapshots.
_NON_FRAME_RECORD_TYPES = {"session_header", "fallback"}

DECISION_STATUSES = ("decision", "decision_cached")


def parse_captured_at(s) -> Optional[datetime]:
    try:
        return datetime.strptime(str(s), "%Y%m%d_%H%M%S_%f")
    except (ValueError, TypeError):
        return None


# ── jsonl loading ───────────────────────────────────────────────────────


def load_session_records(path: str):
    """Returns (header_or_None, frame_records, fallback_records).

    Tolerates blank lines. A malformed json line raises — session logs are
    append-only single-writer; corruption is a bug to surface, not skip.
    """
    header = None
    frames: list[dict] = []
    fallbacks: list[dict] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rt = rec.get("record_type")
            if rt == "session_header":
                header = rec
            elif rt == "fallback":
                fallbacks.append(rec)
            elif rt in _NON_FRAME_RECORD_TYPES:
                continue
            else:
                frames.append(rec)
    return header, frames, fallbacks


# ── hand segmentation (exact port of scripts/dryrun_triage.py) ─────────
# The triage segmentation is the audited ground truth (37 hands for
# session 20260611_163815). Boundary rules: hero pair changes to a
# different non-empty pair, raw dealer string changes, or board shrinks.
# Pair-less fragments (between-hand dead air) merge into the FOLLOWING
# hand; a trailing fragment merges backward.
#
# One extension for pre-raw_record logs (2026-06-08 morning): when
# raw_record is absent the triage's raw_dealer() returns None and the
# dealer-change boundary never fires; we fall back to the top-level
# dealer_seat int so old sessions still segment. For raw_record logs the
# behavior is bit-identical to triage.


def _hero_pair(rec: dict):
    raw = rec.get("raw_record") or {}
    cards = raw.get("hero_cards") or rec.get("hero_cards") or []
    if len(cards) == 2:
        return tuple(sorted(cards))
    return None


def _raw_dealer(rec: dict):
    raw = rec.get("raw_record") or {}
    d = raw.get("dealer") or None
    if d is not None:
        return d
    if not raw:
        ds = rec.get("dealer_seat")
        if ds is not None:
            return f"seat{int(ds) + 1}"
    return None


def _board_len(rec: dict):
    raw = rec.get("raw_record") or {}
    board = raw.get("board")
    if board is None:
        board = rec.get("board")
    return len(board) if board is not None else None


def segment_hands(records: list[dict]) -> list[dict]:
    """Greedy boundary detection; see module/triage docstring."""
    hands: list[dict] = []
    cur = None
    prev_dealer = None
    prev_board_len = None
    for rec in records:
        pair = _hero_pair(rec)
        dealer = _raw_dealer(rec)
        blen = _board_len(rec)
        boundary = cur is None
        if cur is not None:
            if (pair is not None and cur["pair"] is not None
                    and pair != cur["pair"]):
                boundary = True
            if (dealer is not None and prev_dealer is not None
                    and dealer != prev_dealer):
                boundary = True
            if (blen is not None and prev_board_len is not None
                    and blen < prev_board_len):
                boundary = True
        if boundary:
            cur = {"pair": pair, "frames": [], "first_seq": rec.get("seq"),
                   "last_seq": rec.get("seq")}
            hands.append(cur)
        if cur["pair"] is None and pair is not None:
            cur["pair"] = pair
        cur["frames"].append(rec)
        cur["last_seq"] = rec.get("seq")
        if dealer is not None:
            prev_dealer = dealer
        if blen is not None:
            prev_board_len = blen

    merged: list[dict] = []
    pending: list[dict] = []
    for h in hands:
        if h["pair"] is None:
            pending.append(h)
            continue
        if pending:
            h["frames"] = [f for frag in pending
                           for f in frag["frames"]] + h["frames"]
            h["first_seq"] = pending[0]["first_seq"]
            pending = []
        merged.append(h)
    for frag in pending:
        if merged:
            merged[-1]["frames"].extend(frag["frames"])
            merged[-1]["last_seq"] = frag["last_seq"]
        else:
            merged.append(frag)
    return merged


# ── light per-frame view (diff substrate) ───────────────────────────────


@dataclass
class FrameView:
    """Minimal per-seat snapshot extracted from raw_record for diffing.

    Deliberately LESS strict than scraper_schema.parse_frame: frames that
    parse_frame soft-drops (missing dealer, n_alive < 4) still carry
    usable stack/bet observations for the opponent DB."""
    seq: Optional[int]
    captured_at: str
    street: Optional[int]          # 0..3 from board length; None = unusable
    board: tuple
    pot_total: int
    stacks: tuple                  # len 6, null -> 0
    bets: tuple                    # len 6, null -> 0
    folded: tuple                  # len 6 bool
    empty: tuple                   # len 6 bool
    suspect: bool


def _seat_arr(d, coerce, default):
    out = [default] * NUM_SEATS
    for k, v in (d or {}).items():
        if isinstance(k, str) and k.startswith("seat"):
            try:
                idx = int(k[4:]) - 1
            except ValueError:
                continue
            if 0 <= idx < NUM_SEATS:
                out[idx] = coerce(v) if v is not None else default
    return tuple(out)


def frame_view(rec: dict) -> Optional[FrameView]:
    raw = rec.get("raw_record")
    if not raw:
        return None
    board = tuple(raw.get("board") or ())
    street = _STREET_BY_BOARD_LEN.get(len(board))
    pot = raw.get("pot") or {}
    try:
        pot_total = int(pot.get("total", 0))
    except (TypeError, ValueError):
        pot_total = 0
    return FrameView(
        seq=rec.get("seq"),
        captured_at=str(raw.get("captured_at", rec.get("captured_at", ""))),
        street=street,
        board=board,
        pot_total=pot_total,
        stacks=_seat_arr(raw.get("stacks"), int, 0),
        bets=_seat_arr(raw.get("bets"), int, 0),
        folded=_seat_arr(raw.get("folded"), bool, False),
        empty=_seat_arr(raw.get("empty"), bool, False),
        suspect=bool(raw.get("suspect", False)),
    )


# ── output records ──────────────────────────────────────────────────────


@dataclass
class ActionEvent:
    seat: int
    is_hero: bool
    street: int                    # 0..3
    event_idx: int                 # order within the hand
    kind: str                      # fold / call / bet / raise /
                                   # call_allin_partial / partial_unknown /
                                   # or client_action kind for hero rows
    amount_to: Optional[int]       # bet level reached this street (chips)
    amount_delta: Optional[int]    # chips moved in this event window
    stack_before: Optional[int]    # chips behind before the event
    stack_depth_bb: Optional[float]
    all_in: bool
    facing_allin: bool
    voluntary: bool
    observed_via: str              # frame_diff / folded_flag /
                                   # decision_record / safe_fold_record
    seq: Optional[int]
    captured_at: str


@dataclass
class HandRecord:
    hand_idx: int
    first_seq: Optional[int]
    last_seq: Optional[int]
    n_frames: int
    n_usable_frames: int           # frames with raw_record, not suspect
    captured_first: str
    captured_last: str
    duration_seconds: Optional[float]
    level: Optional[int]
    sb: Optional[int]
    bb: Optional[int]
    ante: Optional[int]
    dealer_seat: Optional[int]
    hero_seat: int
    hero_cards: Optional[str]      # "Kd Ts" or None
    board_final: str
    max_street: Optional[int]
    sb_seat: Optional[int]         # None = dead-SB or unknown
    bb_seat: Optional[int]
    dealt_seats: list = field(default_factory=list)
    start_stacks: Optional[dict] = None    # {seat: chips} from anchor
    anchor_found: bool = False
    net_chips: Optional[dict] = None       # {seat: +-chips}; None = unknown
    outcome_known: bool = False
    actions: list = field(default_factory=list)
    hero_reached_showdown: bool = False    # inferred, see parse docstring
    n_decision_frames: int = 0


@dataclass
class SessionParse:
    header: Optional[dict]
    n_frames: int
    n_fallbacks: int
    has_raw_record: bool
    hands: list = field(default_factory=list)
    statuses: dict = field(default_factory=dict)


# ── per-hand action derivation ──────────────────────────────────────────


def _majority(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    counts: dict = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def _try_parse_frame(raw: dict):
    try:
        return parse_frame(raw)
    except (ScraperSuspect, ScraperParseError, ScraperDataQuality,
            ValueError, TypeError):
        return None


def _derive_hand_actions(hand: HandRecord, views: list[FrameView]):
    """Frame-diff action derivation. Appends ActionEvents to hand.actions.

    Baseline starts as the synthetic blinds-posted state when (sb_seat,
    bb_seat) are known; otherwise the first usable frame itself is the
    baseline and emits no events (honest coverage gap). Within a street a
    seat's bet can only rise; a drop on the same street is pot-collection
    UI lag and never lowers the baseline (prevents double-counting).
    """
    if not views or hand.bb is None:
        return
    bb_amt = hand.bb

    baseline_known = hand.bb_seat is not None
    cur_street = 0
    cur_bets = [0] * NUM_SEATS
    if baseline_known:
        if hand.sb_seat is not None and hand.sb is not None:
            cur_bets[hand.sb_seat] = hand.sb
        cur_bets[hand.bb_seat] = bb_amt
    cur_folded = [False] * NUM_SEATS
    cur_stacks: Optional[list] = None
    initialized = baseline_known

    # Seat visit order approximating action order: clockwise from dealer+1.
    d = hand.dealer_seat if hand.dealer_seat is not None else 0
    seat_order = [(d + 1 + i) % NUM_SEATS for i in range(NUM_SEATS)]

    for v in views:
        if v.street is None:
            continue
        if not initialized:
            # No blinds baseline — adopt this frame silently.
            cur_street = v.street
            cur_bets = list(v.bets)
            cur_folded = list(v.folded)
            cur_stacks = list(v.stacks)
            initialized = True
            continue
        if v.street < cur_street:
            continue  # transient board glitch inside a hand
        if v.street > cur_street:
            cur_street = v.street
            cur_bets = [0] * NUM_SEATS
        if cur_stacks is None:
            cur_stacks = list(v.stacks)

        for seat in seat_order:
            if v.empty[seat]:
                continue
            if v.folded[seat] and not cur_folded[seat]:
                hand.actions.append(ActionEvent(
                    seat=seat, is_hero=(seat == hand.hero_seat),
                    street=cur_street, event_idx=len(hand.actions),
                    kind="fold", amount_to=None, amount_delta=None,
                    stack_before=int(v.stacks[seat]),
                    stack_depth_bb=(round(v.stacks[seat] / bb_amt, 2)
                                    if bb_amt else None),
                    all_in=False,
                    facing_allin=_facing_allin(
                        seat, cur_bets, cur_stacks, cur_folded, v),
                    voluntary=False, observed_via="folded_flag",
                    seq=v.seq, captured_at=v.captured_at))
                cur_folded[seat] = True
                continue
            b0, b1 = int(cur_bets[seat]), int(v.bets[seat])
            if b1 > b0 and not cur_folded[seat]:
                facing = max(
                    (int(cur_bets[j]) for j in range(NUM_SEATS)
                     if j != seat and not cur_folded[j] and not v.empty[j]),
                    default=0)
                delta = b1 - b0
                stack_after = int(v.stacks[seat])
                stack_before = stack_after + delta
                all_in = stack_after == 0
                if b1 > facing:
                    kind = "raise" if facing > 0 else "bet"
                elif b1 == facing:
                    kind = "call"
                else:
                    kind = "call_allin_partial" if all_in else "partial_unknown"
                hand.actions.append(ActionEvent(
                    seat=seat, is_hero=(seat == hand.hero_seat),
                    street=cur_street, event_idx=len(hand.actions),
                    kind=kind, amount_to=b1, amount_delta=delta,
                    stack_before=stack_before,
                    stack_depth_bb=(round(stack_before / bb_amt, 2)
                                    if bb_amt else None),
                    all_in=all_in,
                    facing_allin=_facing_allin(
                        seat, cur_bets, cur_stacks, cur_folded, v),
                    voluntary=True, observed_via="frame_diff",
                    seq=v.seq, captured_at=v.captured_at))
                cur_bets[seat] = b1
            elif b1 > cur_bets[seat]:
                cur_bets[seat] = b1
            # b1 < b0 on same street: pot-collection lag; keep baseline.
        cur_stacks = list(v.stacks)


def _facing_allin(seat, cur_bets, cur_stacks, cur_folded, v: FrameView):
    """An outstanding all-in bet from another seat at event time: chips in
    front (baseline bet > 0) with zero behind."""
    stacks = cur_stacks if cur_stacks is not None else v.stacks
    for j in range(NUM_SEATS):
        if j == seat or v.empty[j] or cur_folded[j]:
            continue
        if int(cur_bets[j]) > 0 and int(stacks[j]) == 0:
            return True
    return False


def _hero_decision_actions(hand: HandRecord, frames: list[dict]):
    """Hero rows from the listener's own decision records. These are the
    bot's CHOSEN action — in log-only dry-runs the human at the table may
    have played differently, so they are stored with a distinct
    observed_via and excluded from observed-table statistics."""
    for rec in frames:
        status = rec.get("status")
        if status in DECISION_STATUSES:
            hand.n_decision_frames += 1
            ca = rec.get("client_action") or {}
            kind = ca.get("kind") or "unknown"
            amount = ca.get("chip_amount")
            street = rec.get("street_idx")
            hand.actions.append(ActionEvent(
                seat=hand.hero_seat, is_hero=True,
                street=int(street) if street is not None else 0,
                event_idx=len(hand.actions),
                kind=kind, amount_to=amount, amount_delta=None,
                stack_before=rec.get("hero_stack"),
                stack_depth_bb=(round(rec["hero_stack"] / hand.bb, 2)
                                if rec.get("hero_stack") is not None
                                and hand.bb else None),
                all_in=(kind == "allin"),
                facing_allin=False,  # not re-derived for hero rows
                voluntary=kind not in ("fold", "check", "no_plan",
                                       "unknown"),
                observed_via="decision_record",
                seq=rec.get("seq"), captured_at=str(rec.get("captured_at"))))
        elif status == "safe_fold":
            hand.actions.append(ActionEvent(
                seat=hand.hero_seat, is_hero=True,
                street=(int(rec["street_idx"])
                        if rec.get("street_idx") is not None else 0),
                event_idx=len(hand.actions),
                kind="safe_fold", amount_to=None, amount_delta=None,
                stack_before=rec.get("hero_stack"), stack_depth_bb=None,
                all_in=False, facing_allin=False, voluntary=False,
                observed_via="safe_fold_record",
                seq=rec.get("seq"), captured_at=str(rec.get("captured_at"))))


def _parse_hand(hand_idx: int, seg: dict, hero_seat: int) -> HandRecord:
    frames = seg["frames"]
    views = [fv for fv in (frame_view(r) for r in frames) if fv is not None]
    usable = [v for v in views if not v.suspect]

    blinds = _majority(tuple(r.get("blinds")) if r.get("blinds") else None
                       for r in frames)
    sb = bb = ante = None
    if blinds and len(blinds) == 3:
        sb, bb, ante = int(blinds[0]), int(blinds[1]), int(blinds[2])
    level = _majority(r.get("level") for r in frames)
    dealer_seat = _majority(r.get("dealer_seat") for r in frames)

    cap_first = frames[0].get("captured_at")
    cap_last = frames[-1].get("captured_at")
    t0, t1 = parse_captured_at(cap_first), parse_captured_at(cap_last)
    duration = (t1 - t0).total_seconds() if t0 and t1 else None

    hero_cards = seg.get("pair")
    boards = [v.board for v in usable if v.board]
    board_final = " ".join(max(boards, key=len)) if boards else ""
    streets = [v.street for v in usable if v.street is not None]
    max_street = max(streets) if streets else None

    hand = HandRecord(
        hand_idx=hand_idx, first_seq=seg.get("first_seq"),
        last_seq=seg.get("last_seq"), n_frames=len(frames),
        n_usable_frames=len(usable),
        captured_first=str(cap_first), captured_last=str(cap_last),
        duration_seconds=duration, level=level, sb=sb, bb=bb, ante=ante,
        dealer_seat=dealer_seat, hero_seat=hero_seat,
        hero_cards=" ".join(hero_cards) if hero_cards else None,
        board_final=board_final, max_street=max_street,
        sb_seat=None, bb_seat=None)

    # Hand-start anchor + blind seats via the bridge's own primitives.
    anchor_pf = None
    first_parsed = None
    for rec, v in zip(frames, views):
        raw = rec.get("raw_record")
        if not raw or v.suspect:
            continue
        pf = _try_parse_frame(raw)
        if pf is None:
            continue
        if first_parsed is None:
            first_parsed = pf
        if v.street == 0 and is_hand_start(pf):
            anchor_pf = pf
            break
    ref = anchor_pf or first_parsed
    if ref is not None:
        alive_seats = [i for i in range(NUM_SEATS) if ref.alive[i]]
        if len(alive_seats) >= 2:
            sb_seat, bb_seat, _, _, _ = _derive_blinds_and_action_order(
                ref.dealer_seat, alive_seats, frame=ref)
            hand.sb_seat, hand.bb_seat = sb_seat, bb_seat
        if hand.dealer_seat is None:
            hand.dealer_seat = ref.dealer_seat
    if anchor_pf is not None:
        pre = pre_hand_stacks(anchor_pf)
        hand.anchor_found = True
        hand.start_stacks = {i: int(pre[i]) for i in range(NUM_SEATS)
                             if pre[i] > 0}
        hand.dealt_seats = sorted(hand.start_stacks)
    elif usable:
        v0 = usable[0]
        hand.dealt_seats = [
            i for i in range(NUM_SEATS)
            if not v0.empty[i] and (v0.stacks[i] > 0 or v0.bets[i] > 0)]

    _derive_hand_actions(hand, usable)
    _hero_decision_actions(hand, frames)

    # Inferred showdown reach for hero (flagged as inference, see README):
    # river seen, >= 2 seats not folded at the last usable frame, hero not
    # folded by flag and no hero fold event/decision in the hand.
    if max_street == 3 and usable:
        last = usable[-1]
        not_folded = [i for i in range(NUM_SEATS)
                      if not last.empty[i] and not last.folded[i]
                      and (last.stacks[i] > 0 or last.bets[i] > 0
                           or hand.anchor_found)]
        hero_folded = last.folded[hero_seat] or any(
            a.is_hero and a.kind in ("fold", "safe_fold")
            for a in hand.actions)
        if len(not_folded) >= 2 and hero_seat in not_folded \
                and not hero_folded:
            hand.hero_reached_showdown = True
    return hand


def _derive_outcomes(hands: list[HandRecord]):
    """Net chips per seat for hand i = start_stacks[i+1] - start_stacks[i],
    when BOTH consecutive hands have anchors. Seats present in hand i but
    gone in hand i+1 busted (net = -start). Accept only when the table
    total conserves (sum of nets == 0); otherwise leave unknown."""
    for a, b in zip(hands, hands[1:]):
        if not (a.anchor_found and b.anchor_found
                and a.start_stacks and b.start_stacks):
            continue
        net = {}
        ok = True
        for seat, chips in a.start_stacks.items():
            if seat in b.start_stacks:
                net[seat] = b.start_stacks[seat] - chips
            else:
                net[seat] = -chips  # busted out
        for seat in b.start_stacks:
            if seat not in a.start_stacks:
                ok = False  # seat appeared from nowhere — mis-segmentation
        if ok and sum(net.values()) == 0:
            a.net_chips = net
            a.outcome_known = True


def parse_session_file(path: str) -> SessionParse:
    header, frames, fallbacks = load_session_records(path)
    has_raw = any(r.get("raw_record") for r in frames)
    statuses: dict = {}
    for r in frames:
        s = r.get("status")
        statuses[s] = statuses.get(s, 0) + 1
    hero_seat = _majority(r.get("hero_seat") for r in frames)
    hero_seat = int(hero_seat) if hero_seat is not None else 0

    segs = segment_hands(frames)
    hands = [_parse_hand(i, seg, hero_seat) for i, seg in enumerate(segs)]
    _derive_outcomes(hands)
    return SessionParse(header=header, n_frames=len(frames),
                        n_fallbacks=len(fallbacks), has_raw_record=has_raw,
                        hands=hands, statuses=statuses)
