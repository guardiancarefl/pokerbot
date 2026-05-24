"""Shanky bot profile runtime — evaluates a parsed Profile against game state.

This is the second module of Phase 5 (the parser is the first). It takes a
parsed `Profile` AST and a `GameContext` representing the current decision
point, walks the AST evaluating predicates, and returns the Action that the
profile's rules dictate.

Design:
  - `GameContext` is the adapter surface — a dataclass containing the
    fields the Shanky predicates can ask about (betting-state counts,
    positions, money, hole cards, board). Constructed by callers (eval_pool,
    league trainer, future deployments) from whatever raw state they have.
  - `HandFeatures` is computed once from hole cards + board and exposes
    boolean attributes for every hand-class predicate (havetoppair, haveset,
    haveflushdraw, etc.). Predicates then read these as simple lookups.
  - `BoardFeatures` is computed once from board alone and exposes the
    board-shape booleans (flushpossible, paironboard, threecardstraightonboard,
    etc.).
  - The predicate dictionary maps Shanky predicate names to callables that
    take (ctx, hand_features, board_features) and return a value (boolean
    for bare predicates, integer/float for compare-LHS predicates, string
    for category predicates like position).
  - `evaluate_profile(profile, ctx)` walks the section rules in order,
    evaluates each rule's condition, returns the first matching rule's
    Action. User-flags are tracked in a per-call set (cleared at each
    new decision).

Card representation matches `src/nlhe/equity.py`: cards as integers 0–51,
encoded as `rank * 4 + suit` where rank 0=2, 1=3, ..., 12=A and
suit 0=c, 1=d, 2=h, 3=s. The runtime converts AST `Card(rank='A', suit='s')`
to the integer encoding when matching hand-spec patterns.

The runtime does NOT discretize Shanky's continuous actions into the bot's
discrete action abstraction; that's the policy adapter's job (third module).
The Action returned here is the raw AST Action, intact.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Set, List, Tuple

from src.nlhe.scripted_bots.parser import (
    Profile, Section, Rule, Action, ActionKind,
    Expr, BoolOp, Not, Compare, PredCall, PositionPred, OthersAtom,
    NumberLit, IdentLit, PercentExpr, HandSpec, BoardSpec, Card,
)


# ============================================================
# Card encoding (matches src/nlhe/equity.py)
# ============================================================

_RANKS = "23456789TJQKA"             # index 0..12
_SUITS = "cdhs"                       # index 0..3
_RANK_TO_INT = {r: i for i, r in enumerate(_RANKS)}
_SUIT_TO_INT = {s: i for i, s in enumerate(_SUITS)}
_INT_TO_RANK = {i: r for r, i in _RANK_TO_INT.items()}
_INT_TO_SUIT = {i: s for s, i in _SUIT_TO_INT.items()}


def card_to_int(rank: str, suit: str) -> int:
    """E.g. ('A','s') -> 51; ('2','c') -> 0."""
    return _RANK_TO_INT[rank.upper()] * 4 + _SUIT_TO_INT[suit.lower()]


def int_to_rank_suit(card: int) -> Tuple[str, str]:
    return _INT_TO_RANK[card // 4], _INT_TO_SUIT[card % 4]


def card_rank(card: int) -> int:
    """Integer rank 0..12 (0=2, 12=A)."""
    return card // 4


def card_suit(card: int) -> int:
    return card % 4


# ============================================================
# Game context (the adapter surface)
# ============================================================

@dataclass
class GameContext:
    """All the state a Shanky predicate might read.

    Constructed by callers from their game representation. Most fields have
    natural defaults; only `hole_cards`, `board`, `street`, and basic betting
    state are typically required.

    Conventions:
      - All chip amounts are in BIG BLINDS (matching how Shanky profiles
        typically write numeric thresholds: `stacksize < 10`, `betsize > 4`).
      - `street`: one of 'preflop', 'flop', 'turn', 'river'.
      - `position`: one of 'first', 'middle', 'last', or specific seat
        names ('button', 'smallblind', 'bigblind').
      - `hole_cards`: list of 2 ints, e.g. [51, 47] for As Ks.
      - `board`: list of 0/3/4/5 ints depending on street.
      - `botslastaction`: 'none' | 'check' | 'call' | 'raise' | 'bet' | 'fold'
                          | 'beep' | 'allin'.
    """
    # Hole cards and board (integer encoding)
    hole_cards: List[int] = field(default_factory=list)
    board: List[int] = field(default_factory=list)
    street: str = "preflop"          # 'preflop' | 'flop' | 'turn' | 'river'

    # Betting state (current street unless noted)
    raises: int = 0
    bets: int = 0
    calls: int = 0
    folds: int = 0
    checks: int = 0
    amounttocall: float = 0.0        # in big blinds
    betsize: float = 0.0             # in big blinds — current bet to face
    potsize: float = 0.0             # in big blinds
    bigblindsize: float = 1.0        # in chips per BB (always 1.0 if everything in BB)
    stacksize: float = 100.0         # hero's effective stack in BBs
    totalinvested: float = 0.0       # hero's chips already in pot, in BBs
    opponents: int = 0               # active opponents in hand
    opponentsattable: int = 0        # opponents at the table (including folded)
    opponentsonflop: int = 0         # opponents at start of flop
    stilltoact: int = 0              # opponents yet to act this street

    # Position
    position: str = "middle"          # 'first' | 'middle' | 'last' or specific seat
    lastraiserposition: Optional[int] = None
    firstcallerposition: Optional[int] = None
    firstraiserposition: Optional[int] = None
    lastcallerposition: Optional[int] = None

    # Bot action history
    botslastaction: str = "none"
    botslastpreflopaction: str = "none"
    botsactionsonthisround: int = 0
    botsactionsonflop: int = 0
    botsactionspreflop: int = 0
    callssincelastraise: int = 0
    raisessincelastplay: int = 0

    # Street-specific raise/bet history
    raisesbeforeflop: int = 0
    raisesonflop: int = 0
    raisesonturn: int = 0
    nobettingonflop_flag: bool = False
    nobettingonturn_flag: bool = False
    botraisedbeforeflop_flag: bool = False
    botraisedonflop_flag: bool = False
    botraisedonturn_flag: bool = False
    calledonflop_flag: bool = False
    calledonturn_flag: bool = False
    botcalledbeforeflop_flag: bool = False
    opponentcalledonturn_flag: bool = False
    botislastraiser_flag: bool = False

    # Opponent state
    opponentisallin_flag: bool = False
    maxopponentstacksize: float = 100.0
    minopponentstacksize: float = 100.0
    maxcurrentopponentstacksize: float = 100.0
    opponentswithhigherstack: int = 0

    # Previous-street hand class memory (used by hadtoppaironflop etc.)
    hadtoppaironflop_flag: bool = False
    hadoverpaironflop_flag: bool = False

    # Random seed for the `random` predicate (0-100 per rule evaluation)
    rng_seed: int = 0


# ============================================================
# Hand features
# ============================================================

# Helpers for hand evaluation.

def _count_ranks(cards: List[int]) -> Dict[int, int]:
    """rank-index -> count, for the given cards."""
    counts: Dict[int, int] = {}
    for c in cards:
        r = card_rank(c)
        counts[r] = counts.get(r, 0) + 1
    return counts


def _count_suits(cards: List[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for c in cards:
        s = card_suit(c)
        counts[s] = counts.get(s, 0) + 1
    return counts


def _is_straight(rank_set: Set[int]) -> bool:
    """True if rank_set contains 5 consecutive ranks, OR contains
    an A-low straight (A,2,3,4,5)."""
    if not rank_set:
        return False
    ranks = sorted(rank_set)
    # Slide a window of 5 over consecutive ranks.
    for i in range(len(ranks) - 4):
        if ranks[i + 4] - ranks[i] == 4:
            return True
    # Wheel straight: A,2,3,4,5 (A=12, 2=0, 3=1, 4=2, 5=3)
    if {12, 0, 1, 2, 3}.issubset(rank_set):
        return True
    return False


def _straight_high(rank_set: Set[int]) -> Optional[int]:
    """Return the high-rank of the best straight in rank_set, or None."""
    if not rank_set:
        return None
    ranks = sorted(rank_set, reverse=True)
    rs = set(rank_set)
    # Try every possible top-rank
    for r in ranks:
        if all((r - i) in rs for i in range(5)):
            return r
    # Wheel
    if {12, 0, 1, 2, 3}.issubset(rs):
        return 3  # 5-high straight, "high" rank is 5 (= index 3)
    return None


def _consecutive_in_set(rank_set: Set[int], n: int) -> bool:
    """True if `rank_set` contains n consecutive ranks (anywhere)."""
    if not rank_set:
        return False
    ranks = sorted(rank_set)
    for i in range(len(ranks) - n + 1):
        if ranks[i + n - 1] - ranks[i] == n - 1:
            return True
    return False


@dataclass
class HandFeatures:
    """Booleans and counts about hero's hand given hole cards + board.

    All fields are computed once at construction and read by predicates.
    """
    # Made-hand classes
    havepair: bool = False
    havetoppair: bool = False
    have2ndtoppair: bool = False
    have3rdtoppair: bool = False
    have4thtoppair: bool = False
    havebottompair: bool = False
    haveunderpair: bool = False
    haveoverpair: bool = False
    have2ndoverpair: bool = False
    have3rdoverpair: bool = False
    have4thoverpair: bool = False
    have5thoverpair: bool = False
    pairinhand: bool = False
    havetwopair: bool = False
    havetoptwopair: bool = False
    haveset: bool = False
    havetopset: bool = False
    havebottomset: bool = False
    havetrips: bool = False
    havestraight: bool = False
    havenutstraight: bool = False
    have2ndnutstraight: bool = False
    haveunderstraight: bool = False
    haveflush: bool = False
    havenutflush: bool = False
    have2ndnutflush: bool = False
    have3rdnutflush: bool = False
    havefullhouse: bool = False
    havequads: bool = False
    havestraightflush: bool = False
    havenutstraightflush: bool = False
    havenuts: bool = False
    havenothing: bool = False
    havetopnonboardpairedpair: bool = False

    # Drawing-hand classes
    havestraightdraw: bool = False             # open-ended (8 outs)
    haveinsidestraightdraw: bool = False       # gutshot (4 outs)
    haveinsidenutstraightdraw: bool = False
    havenutstraightdraw: bool = False
    haveflushdraw: bool = False
    havenutflushdraw: bool = False
    have2ndnutflushdraw: bool = False
    have3rdnutflushdraw: bool = False
    have4thnutflushdraw: bool = False
    have5thnutflushdraw: bool = False
    havebackdoorflushdraw: bool = False
    havebackdoornutflushdraw: bool = False

    # Kicker quality (only meaningful when havepair-like)
    havebestkicker: bool = False
    have2ndbestkicker: bool = False
    have3rdbestkicker: bool = False
    havebestkickerorbetter: bool = False
    have2ndbestkickerorbetter: bool = False
    have3rdbestkickerorbetter: bool = False

    # Counts/quantities
    overcards: int = 0
    overcardsonboard: int = 0
    suitsinhand: int = 1
    suitsonboard: int = 0
    suitsonflop: int = 0


def compute_hand_features(hole_cards: List[int], board: List[int]) -> HandFeatures:
    """Categorize hero's hand into the boolean classes Shanky predicates ask about.

    Args:
        hole_cards: 2 ints (or 0 if preflop and we have no hand info).
        board: 0, 3, 4, or 5 ints.

    Returns:
        Populated HandFeatures.
    """
    f = HandFeatures()

    if len(hole_cards) < 2:
        return f  # nothing to evaluate

    # Hole-card features
    h0, h1 = hole_cards[0], hole_cards[1]
    hr0, hr1 = card_rank(h0), card_rank(h1)
    hs0, hs1 = card_suit(h0), card_suit(h1)
    f.pairinhand = (hr0 == hr1)
    f.suitsinhand = 1 if hs0 == hs1 else 2

    # Board features the runtime cares about for hand classification
    board_ranks = [card_rank(c) for c in board]
    board_rank_set = set(board_ranks)
    board_rank_counts = _count_ranks(board)
    f.suitsonboard = len(_count_suits(board)) if board else 0
    # `suitsonflop` is the suit count of the first 3 board cards
    if len(board) >= 3:
        flop_suit_counts = _count_suits(board[:3])
        f.suitsonflop = len(flop_suit_counts)

    # All cards we hold = hole + board
    all_cards = hole_cards + board
    all_ranks = [card_rank(c) for c in all_cards]
    all_rank_counts = _count_ranks(all_cards)

    # === MADE HAND CLASSES ===

    # Quads, full house, trips, two pair, pair detection
    rank_counts_sorted = sorted(all_rank_counts.values(), reverse=True)
    has_quads = rank_counts_sorted[0] >= 4
    has_fh = (rank_counts_sorted[0] >= 3 and len(rank_counts_sorted) >= 2 and rank_counts_sorted[1] >= 2)
    has_trips = rank_counts_sorted[0] >= 3
    has_two_pair = (rank_counts_sorted[0] >= 2 and len(rank_counts_sorted) >= 2 and rank_counts_sorted[1] >= 2)
    has_one_pair = rank_counts_sorted[0] >= 2

    f.havequads = has_quads
    f.havefullhouse = has_fh
    f.havetwopair = has_two_pair and not has_fh and not has_quads
    f.havepair = has_one_pair

    # Set/trips distinction: SET means pocket pair + 1 on board; TRIPS means
    # board pair + 1 in hand. Both yield three-of-a-kind, but Shanky treats
    # them as separate predicates.
    if has_trips and not has_fh and not has_quads:
        # Find the rank that has 3.
        trip_rank = max(r for r, c in all_rank_counts.items() if c >= 3)
        # SET if hero holds the pair
        if f.pairinhand and hr0 == trip_rank:
            f.haveset = True
        else:
            # TRIPS if board has the pair and hero holds the third
            board_has_pair = board_rank_counts.get(trip_rank, 0) >= 2
            hero_holds_trip = (hr0 == trip_rank or hr1 == trip_rank)
            if board_has_pair and hero_holds_trip:
                f.havetrips = True
            else:
                # Edge case fall-through (e.g., one hero card + board pair).
                f.haveset = True

        # Top/bottom set
        if f.haveset and board_ranks:
            sorted_board = sorted(set(board_ranks), reverse=True)
            if trip_rank > max(board_ranks):
                pass  # overpair set, not really "top set"; leave both False
            elif trip_rank == sorted_board[0]:
                f.havetopset = True
            elif trip_rank == sorted_board[-1]:
                f.havebottomset = True

    # Straight detection
    all_rank_set = set(all_ranks)
    if _is_straight(all_rank_set):
        f.havestraight = True

    # Flush detection — any 5+ cards of same suit
    all_suit_counts = _count_suits(all_cards)
    flush_suit = None
    for suit, count in all_suit_counts.items():
        if count >= 5:
            flush_suit = suit
            break
    if flush_suit is not None:
        f.haveflush = True

    # Straight flush detection
    if flush_suit is not None:
        flush_ranks = {card_rank(c) for c in all_cards if card_suit(c) == flush_suit}
        if _is_straight(flush_ranks):
            f.havestraightflush = True

    # Pair classification on the board: top-pair, 2nd-top, bottom-pair, etc.
    if has_one_pair and board_ranks:
        sorted_board_ranks_desc = sorted(set(board_ranks), reverse=True)
        # Find which rank hero paired
        hero_paired_rank = None
        if f.pairinhand:
            # Pocket pair: check if it matches one of these conditions
            top_board = sorted_board_ranks_desc[0]
            if hr0 > top_board:
                f.haveoverpair = True
                # Compute 2nd/3rd/4th/5th overpair
                # Sort possible "overpair tiers": pairs above the top board card.
                # 2nd-overpair = pocket pair that's above top board but not the absolute top.
                # This is rarely useful; we approximate: hero's pair rank relative to
                # all higher-than-top-board ranks.
                possible_overs = [r for r in range(top_board + 1, 13)]
                if possible_overs:
                    # Rank index in descending order
                    possible_overs_desc = sorted(possible_overs, reverse=True)
                    if hr0 == possible_overs_desc[0]:
                        pass  # nominal "top overpair" — fall through to haveoverpair
                    if len(possible_overs_desc) >= 2 and hr0 == possible_overs_desc[1]:
                        f.have2ndoverpair = True
                    if len(possible_overs_desc) >= 3 and hr0 == possible_overs_desc[2]:
                        f.have3rdoverpair = True
                    if len(possible_overs_desc) >= 4 and hr0 == possible_overs_desc[3]:
                        f.have4thoverpair = True
                    if len(possible_overs_desc) >= 5 and hr0 == possible_overs_desc[4]:
                        f.have5thoverpair = True
            elif hr0 < min(board_ranks):
                f.haveunderpair = True
            # Pocket pair matching a board rank → set (already handled above)
        else:
            # Non-pocket-pair: hero paired a board card with one hole card
            for hole_r in (hr0, hr1):
                if hole_r in board_rank_set:
                    hero_paired_rank = hole_r
                    break
            if hero_paired_rank is not None:
                if hero_paired_rank == sorted_board_ranks_desc[0]:
                    f.havetoppair = True
                elif len(sorted_board_ranks_desc) >= 2 and hero_paired_rank == sorted_board_ranks_desc[1]:
                    f.have2ndtoppair = True
                elif len(sorted_board_ranks_desc) >= 3 and hero_paired_rank == sorted_board_ranks_desc[2]:
                    f.have3rdtoppair = True
                elif len(sorted_board_ranks_desc) >= 4 and hero_paired_rank == sorted_board_ranks_desc[3]:
                    f.have4thtoppair = True
                if hero_paired_rank == sorted_board_ranks_desc[-1]:
                    f.havebottompair = True

    # Top two pair: hero has two pair using the top two board ranks
    if f.havetwopair and len(board_ranks) >= 2:
        sorted_board_ranks_desc = sorted(set(board_ranks), reverse=True)
        if len(sorted_board_ranks_desc) >= 2:
            top_two = {sorted_board_ranks_desc[0], sorted_board_ranks_desc[1]}
            hole_set = {hr0, hr1}
            if top_two.issubset(hole_set):
                f.havetoptwopair = True

    # === DRAWING HANDS (only meaningful on flop/turn) ===

    if 3 <= len(board) <= 4:
        # Flush draw: 4 to a flush total (hero must hold at least 1 of the suit)
        for suit, count in all_suit_counts.items():
            hero_suit_count = sum(1 for c in hole_cards if card_suit(c) == suit)
            if count == 4 and hero_suit_count >= 1:
                f.haveflushdraw = True
                # Nut flush draw: hero holds the A of that suit
                if any(card_rank(c) == 12 and card_suit(c) == suit for c in hole_cards):
                    f.havenutflushdraw = True
                break
            if count == 3 and hero_suit_count >= 1 and len(board) == 3:
                # Backdoor flush draw (only on flop)
                f.havebackdoorflushdraw = True
                if any(card_rank(c) == 12 and card_suit(c) == suit for c in hole_cards):
                    f.havebackdoornutflushdraw = True

        # Straight draws: count how many of hero's cards contribute to a near-straight
        # Open-ended: hero's two hole cards plus 2 board cards form 4 in a row, and
        # filling either end completes a straight.
        # Inside (gutshot): one missing rank in the middle.
        # We check by simulating: for each rank r, would adding r to all_rank_set
        # complete a straight? If exactly one such r exists at either extreme,
        # it's a one-sided open-ender (treat as gutshot, 4 outs).
        if not f.havestraight:
            outs = []
            for r in range(13):
                if r in all_rank_set:
                    continue
                if _is_straight(all_rank_set | {r}):
                    outs.append(r)
            # Hero must contribute to the straight via at least one hole card
            # being among the consecutive 4 ranks.
            if outs and any(hr in all_rank_set for hr in (hr0, hr1)):
                if len(outs) >= 2:
                    f.havestraightdraw = True
                else:
                    f.haveinsidestraightdraw = True

    # === BEST-NUTS CLASSIFICATION ===
    # "Nuts" for runtime purposes: best possible 5-card hand for hero given board.
    # This is a simplification; we approximate via hand-class rank.
    if f.havestraightflush and not (f.havequads or f.havefullhouse):
        # Best straight flush check: hero's straight flush uses the highest possible top.
        f.havenutstraightflush = True
        f.havenuts = True
    elif f.havequads:
        f.havenuts = True
    elif f.havefullhouse:
        # Approximation — true "nut full house" requires checking nobody else has
        # a better full house given the board. Defer the strict check.
        pass
    elif f.haveflush:
        # Nut flush: hero's flush uses the A of the flush suit.
        if flush_suit is not None:
            hero_has_ace_of_suit = any(
                card_rank(c) == 12 and card_suit(c) == flush_suit for c in hole_cards
            )
            if hero_has_ace_of_suit:
                f.havenutflush = True
    elif f.havestraight:
        # Nut straight: hero's straight is the highest possible given the board.
        # Approximation: hero's straight uses ranks at the top of the possible range.
        f.havenutstraight = True   # simplification

    # === KICKER QUALITY ===
    # When hero has a pair (top pair specifically), the kicker is the other hole card.
    if f.havetoppair and not f.pairinhand:
        # Kicker is whichever hole card isn't the paired one
        kicker_rank = hr1 if hr0 in board_rank_set else hr0
        # Best kicker = A. 2nd-best = K (excluding ranks on board). 3rd-best = Q. etc.
        kickers_in_descending_order = sorted(
            (r for r in range(13) if r not in board_rank_set),
            reverse=True,
        )
        if kickers_in_descending_order:
            if kicker_rank == kickers_in_descending_order[0]:
                f.havebestkicker = True
                f.havebestkickerorbetter = True
                f.have2ndbestkickerorbetter = True
                f.have3rdbestkickerorbetter = True
            elif len(kickers_in_descending_order) >= 2 and kicker_rank == kickers_in_descending_order[1]:
                f.have2ndbestkicker = True
                f.have2ndbestkickerorbetter = True
                f.have3rdbestkickerorbetter = True
            elif len(kickers_in_descending_order) >= 3 and kicker_rank == kickers_in_descending_order[2]:
                f.have3rdbestkicker = True
                f.have3rdbestkickerorbetter = True

    # === MISC COUNTS ===

    # Overcards: number of hero hole cards higher than the top board card.
    if board_ranks:
        top_board = max(board_ranks)
        f.overcards = sum(1 for r in (hr0, hr1) if r > top_board)
        # Overcards on board: number of board cards higher than both hero cards.
        max_hero = max(hr0, hr1)
        f.overcardsonboard = sum(1 for r in board_ranks if r > max_hero)

    # Have nothing: no pair, no draws
    f.havenothing = not (
        f.havepair or f.havetwopair or f.haveset or f.havetrips
        or f.havestraight or f.haveflush or f.havefullhouse
        or f.havequads or f.havestraightflush
        or f.havestraightdraw or f.haveinsidestraightdraw
        or f.haveflushdraw
    )

    return f


# ============================================================
# Board features
# ============================================================

@dataclass
class BoardFeatures:
    """Booleans about the board alone (independent of hero's hole cards)."""
    paironboard: bool = False
    twopaironboard: bool = False
    tripsonboard: bool = False
    quadsonboard: bool = False
    fullhouseonboard: bool = False
    straightonboard: bool = False
    flushonboard: bool = False
    threecardstraightonboard: bool = False
    flushpossible: bool = False
    straightpossible: bool = False
    straightflushpossible: bool = False
    straightflushpossiblebyothers: bool = False
    onecardflushpossible: bool = False
    onecardstraightpossible: bool = False
    onecardstraightpossibleonturn: bool = False
    morethanonestraightpossibleonflop: bool = False
    morethanonestraightpossibleonturn: bool = False
    onlyonestraightpossible: bool = False
    flushpossibleonflop: bool = False
    flushpossibleonturn: bool = False
    straightpossibleonflop: bool = False
    straightpossibleonturn: bool = False
    paironflop: bool = False
    paironturn: bool = False
    acepresentonflop: bool = False
    kingpresentonflop: bool = False
    queenpresentonflop: bool = False
    uncoordinatedflop: bool = False
    topflopcardpairedonturn: bool = False
    rivercardisovercardtoboard: bool = False
    nutfullhouseorfourofakind: bool = False        # actually depends on hero too

    # Counts
    suitsonboard: int = 0
    suitsonflop: int = 0


def compute_board_features(board: List[int]) -> BoardFeatures:
    """Compute board-shape booleans from board alone."""
    f = BoardFeatures()

    if not board:
        return f

    board_ranks = [card_rank(c) for c in board]
    board_rank_set = set(board_ranks)
    board_rank_counts = _count_ranks(board)
    suit_counts = _count_suits(board)

    f.suitsonboard = len(suit_counts)

    # Pair / two-pair / trips / quads / full house on board
    counts_sorted = sorted(board_rank_counts.values(), reverse=True)
    if counts_sorted:
        if counts_sorted[0] >= 4:
            f.quadsonboard = True
        if counts_sorted[0] >= 3:
            f.tripsonboard = True
        if counts_sorted[0] >= 3 and len(counts_sorted) >= 2 and counts_sorted[1] >= 2:
            f.fullhouseonboard = True
        if counts_sorted[0] >= 2 and len(counts_sorted) >= 2 and counts_sorted[1] >= 2:
            f.twopaironboard = True
        if counts_sorted[0] >= 2:
            f.paironboard = True

    # Straight on board
    if _is_straight(board_rank_set):
        f.straightonboard = True

    # Flush on board (5+ same suit)
    if any(c >= 5 for c in suit_counts.values()):
        f.flushonboard = True

    # Flush possible: 3+ same suit on board
    f.flushpossible = any(c >= 3 for c in suit_counts.values())
    # Onecard flush possible: 4+ same suit on board (only need one matching card)
    f.onecardflushpossible = any(c >= 4 for c in suit_counts.values())

    # Three-card straight on board: 3 consecutive ranks (or A-2-3 etc.)
    f.threecardstraightonboard = _consecutive_in_set(board_rank_set, 3)
    # Straight possible: 3 within a 5-rank window
    # Onecard straight possible: 4 within a 5-rank window
    for top in range(4, 13):
        window = set(range(top - 4, top + 1))
        n = len(window & board_rank_set)
        if n >= 3:
            f.straightpossible = True
        if n >= 4:
            f.onecardstraightpossible = True
    # Wheel window
    wheel = {12, 0, 1, 2, 3}
    n_wheel = len(wheel & board_rank_set)
    if n_wheel >= 3:
        f.straightpossible = True
    if n_wheel >= 4:
        f.onecardstraightpossible = True

    # Straight flush possible by others: 3+ same suit AND those forming
    # a straight pattern within them
    for suit, count in suit_counts.items():
        if count < 3:
            continue
        suit_ranks = {card_rank(c) for c in board if card_suit(c) == suit}
        if _consecutive_in_set(suit_ranks, 3):
            f.straightflushpossiblebyothers = True
            if count >= 4:
                f.straightflushpossible = True

    # Specific-cards-on-flop predicates (flop is board[:3])
    if len(board) >= 3:
        flop = board[:3]
        flop_ranks = {card_rank(c) for c in flop}
        flop_suit_counts = _count_suits(flop)
        f.suitsonflop = len(flop_suit_counts)
        f.acepresentonflop = 12 in flop_ranks
        f.kingpresentonflop = 11 in flop_ranks
        f.queenpresentonflop = 10 in flop_ranks
        # Pair on flop = same as paironboard when board is exactly 3
        if any(c >= 2 for c in _count_ranks(flop).values()):
            f.paironflop = True
        # Flush possible on flop = 3 same suit
        f.flushpossibleonflop = any(c >= 3 for c in flop_suit_counts.values())
        # Straight possible on flop
        for top in range(4, 13):
            window = set(range(top - 4, top + 1))
            if len(window & flop_ranks) >= 3:
                f.straightpossibleonflop = True
        # Uncoordinated flop: no pair, no flush draw, no straight draw
        f.uncoordinatedflop = (
            not f.paironflop
            and not f.flushpossibleonflop
            and not f.straightpossibleonflop
        )

    # Turn-specific predicates (board[3] is the turn card)
    if len(board) >= 4:
        turn_card = board[3]
        flop_ranks = [card_rank(c) for c in board[:3]]
        flop_suits = [card_suit(c) for c in board[:3]]
        # Pair on turn: turn card matches a flop card
        if card_rank(turn_card) in set(flop_ranks):
            f.paironturn = True
        # Top flop card paired on turn
        top_flop = max(flop_ranks)
        if card_rank(turn_card) == top_flop:
            f.topflopcardpairedonturn = True
        # Flush possible on turn: 3 same suit total (flop + turn)
        turn_suit_counts = _count_suits(board[:4])
        f.flushpossibleonturn = any(c >= 3 for c in turn_suit_counts.values())
        # Straight possible on turn
        turn_ranks = {card_rank(c) for c in board[:4]}
        for top in range(4, 13):
            window = set(range(top - 4, top + 1))
            if len(window & turn_ranks) >= 3:
                f.straightpossibleonturn = True
        # One-card straight possible on turn: 4 in a window
        for top in range(4, 13):
            window = set(range(top - 4, top + 1))
            if len(window & turn_ranks) >= 4:
                f.onecardstraightpossibleonturn = True

        # More than one straight possible on flop / turn
        flop_set = set(flop_ranks)
        flop_straights = 0
        for top in range(4, 13):
            window = set(range(top - 4, top + 1))
            if len(window & flop_set) >= 3:
                flop_straights += 1
        f.morethanonestraightpossibleonflop = flop_straights > 1

        turn_straights = 0
        for top in range(4, 13):
            window = set(range(top - 4, top + 1))
            if len(window & turn_ranks) >= 3:
                turn_straights += 1
        f.morethanonestraightpossibleonturn = turn_straights > 1
        f.onlyonestraightpossible = (flop_straights == 1) if len(board) == 3 else (turn_straights == 1)

    # River-card-is-overcard-to-board: river card higher than all flop+turn cards
    if len(board) == 5:
        river_rank = card_rank(board[4])
        prior_max = max(card_rank(c) for c in board[:4])
        f.rivercardisovercardtoboard = river_rank > prior_max

    return f


# ============================================================
# Hand-spec matching (for `hand = AK suited` etc.)
# ============================================================

def matches_hand_spec(spec: HandSpec, hole_cards: List[int]) -> bool:
    """True if hero's hole cards match the given HandSpec.

    Handles all the forms the parser produces:
      - 2 fully-specified cards: Kd 9d
      - 2 ranks, optional suited/offsuit
      - 1 rank (wildcard): any A-x
    """
    if len(hole_cards) < 2:
        return False

    h0, h1 = hole_cards[0], hole_cards[1]
    hr0, hr1 = card_rank(h0), card_rank(h1)
    hs0, hs1 = card_suit(h0), card_suit(h1)
    hero_is_suited = (hs0 == hs1)

    if len(spec.cards) == 1:
        # Wildcard: any hand containing the specified rank
        spec_rank = _RANK_TO_INT.get(spec.cards[0].rank.upper())
        if spec_rank is None:
            return False
        present = (hr0 == spec_rank or hr1 == spec_rank)
        if not present:
            return False
        if spec.suitedness == "suited" and not hero_is_suited:
            return False
        if spec.suitedness == "offsuit" and hero_is_suited:
            return False
        return True

    if len(spec.cards) == 2:
        c0, c1 = spec.cards
        sr0 = _RANK_TO_INT.get(c0.rank.upper())
        sr1 = _RANK_TO_INT.get(c1.rank.upper())
        if sr0 is None or sr1 is None:
            return False
        hero_ranks = sorted([hr0, hr1], reverse=True)
        spec_ranks = sorted([sr0, sr1], reverse=True)
        if hero_ranks != spec_ranks:
            return False

        # If suit is specified on either card, hero must match exactly
        if c0.suit is not None or c1.suit is not None:
            spec_suit0 = _SUIT_TO_INT.get(c0.suit) if c0.suit else None
            spec_suit1 = _SUIT_TO_INT.get(c1.suit) if c1.suit else None
            hero_suits = sorted([hs0, hs1])
            # Try both orderings
            if spec_suit0 is not None and spec_suit1 is not None:
                spec_suits = sorted([spec_suit0, spec_suit1])
                # The (rank, suit) pairings need to match
                # Hero: (hr0,hs0),(hr1,hs1)
                # Spec: (sr0,spec_suit0),(sr1,spec_suit1)
                # Try ordering 1
                if ((hr0 == sr0 and hs0 == spec_suit0 and hr1 == sr1 and hs1 == spec_suit1)
                        or (hr0 == sr1 and hs0 == spec_suit1 and hr1 == sr0 and hs1 == spec_suit0)):
                    return True
                return False

        # Otherwise check suitedness modifier
        if spec.suitedness == "suited" and not hero_is_suited:
            return False
        if spec.suitedness == "offsuit" and hero_is_suited:
            return False
        return True

    return False


def matches_board_spec(spec: BoardSpec, board: List[int]) -> bool:
    """True if the board's ranks contain all ranks in the BoardSpec.

    Shanky `board = AKQ` means "the board contains an A, a K, and a Q
    (possibly with extra cards)." Order doesn't matter.
    """
    if not board:
        return len(spec.ranks) == 0
    board_ranks = [card_rank(c) for c in board]
    # Each spec rank must be present at least as many times as it appears.
    spec_rank_counts: Dict[int, int] = {}
    for r in spec.ranks:
        ri = _RANK_TO_INT.get(r.upper())
        if ri is None:
            return False
        spec_rank_counts[ri] = spec_rank_counts.get(ri, 0) + 1
    board_rank_counts = _count_ranks(board)
    for ri, need in spec_rank_counts.items():
        if board_rank_counts.get(ri, 0) < need:
            return False
    return True


# ============================================================
# Predicate dictionary
# ============================================================
#
# Compare-LHS lookups: name -> callable(ctx, hf, bf) -> numeric or string value.
# These are the predicates that appear on the left side of `=`, `<`, etc.

def _compare_lhs_dict() -> Dict[str, Callable]:
    return {
        # Direct ctx fields
        "raises":                lambda c, hf, bf: c.raises,
        "bets":                  lambda c, hf, bf: c.bets,
        "calls":                 lambda c, hf, bf: c.calls,
        "folds":                 lambda c, hf, bf: c.folds,
        "checks":                lambda c, hf, bf: c.checks,
        "amounttocall":          lambda c, hf, bf: c.amounttocall,
        "betsize":               lambda c, hf, bf: c.betsize,
        "potsize":               lambda c, hf, bf: c.potsize,
        "stacksize":             lambda c, hf, bf: c.stacksize,
        "bigblindsize":          lambda c, hf, bf: c.bigblindsize,
        "totalinvested":         lambda c, hf, bf: c.totalinvested,
        "opponents":             lambda c, hf, bf: c.opponents,
        "opponentsattable":      lambda c, hf, bf: c.opponentsattable,
        "opponentsonflop":       lambda c, hf, bf: c.opponentsonflop,
        "stilltoact":            lambda c, hf, bf: c.stilltoact,
        "position":              lambda c, hf, bf: c.position,
        "lastraiserposition":    lambda c, hf, bf: c.lastraiserposition or 0,
        "firstcallerposition":   lambda c, hf, bf: c.firstcallerposition or 0,
        "firstraiserposition":   lambda c, hf, bf: c.firstraiserposition or 0,
        "lastcallerposition":    lambda c, hf, bf: c.lastcallerposition or 0,
        "botslastaction":        lambda c, hf, bf: c.botslastaction,
        "botslastpreflopaction": lambda c, hf, bf: c.botslastpreflopaction,
        "botsactionsonthisround": lambda c, hf, bf: c.botsactionsonthisround,
        "botsactionsonflop":     lambda c, hf, bf: c.botsactionsonflop,
        "botsactionspreflop":    lambda c, hf, bf: c.botsactionspreflop,
        "callssincelastraise":   lambda c, hf, bf: c.callssincelastraise,
        "raisessincelastplay":   lambda c, hf, bf: c.raisessincelastplay,
        "raisesbeforeflop":      lambda c, hf, bf: c.raisesbeforeflop,
        "raisesonflop":          lambda c, hf, bf: c.raisesonflop,
        "raisesonturn":          lambda c, hf, bf: c.raisesonturn,

        # Hand-derived counts
        "overcards":             lambda c, hf, bf: hf.overcards,
        "overcardsonboard":      lambda c, hf, bf: hf.overcardsonboard,
        "suitsinhand":           lambda c, hf, bf: hf.suitsinhand,
        "suitsonboard":          lambda c, hf, bf: bf.suitsonboard,
        "suitsonflop":           lambda c, hf, bf: bf.suitsonflop,

        # Opponent stack metrics
        "maxopponentstacksize":         lambda c, hf, bf: c.maxopponentstacksize,
        "maxcurrentopponentstacksize":  lambda c, hf, bf: c.maxcurrentopponentstacksize,
        "minopponentstacksize":         lambda c, hf, bf: c.minopponentstacksize,
        "opponentswithhigherstack":     lambda c, hf, bf: c.opponentswithhigherstack,

        # `random` 0-100, per-rule
        "random":                lambda c, hf, bf: _random.Random(c.rng_seed).randint(0, 100),

        # `nutfullhouseorfourofakind` — used as compare-LHS in some profiles.
        # Treat as 1 if hero has nut FH or quads, 0 otherwise.
        "nutfullhouseorfourofakind": lambda c, hf, bf: 1 if (hf.havequads or (hf.havefullhouse and hf.havenuts)) else 0,
    }


def _bare_predicate_dict() -> Dict[str, Callable]:
    """Bare predicates: name -> callable(ctx, hf, bf) -> bool."""
    return {
        # ----- Hand-class predicates (read from HandFeatures) -----
        "havepair":              lambda c, hf, bf: hf.havepair,
        "havetoppair":           lambda c, hf, bf: hf.havetoppair,
        "have2ndtoppair":        lambda c, hf, bf: hf.have2ndtoppair,
        "have3rdtoppair":        lambda c, hf, bf: hf.have3rdtoppair,
        "have4thtoppair":        lambda c, hf, bf: hf.have4thtoppair,
        "havebottompair":        lambda c, hf, bf: hf.havebottompair,
        "haveunderpair":         lambda c, hf, bf: hf.haveunderpair,
        "haveoverpair":          lambda c, hf, bf: hf.haveoverpair,
        "have2ndoverpair":       lambda c, hf, bf: hf.have2ndoverpair,
        "have3rdoverpair":       lambda c, hf, bf: hf.have3rdoverpair,
        "have4thoverpair":       lambda c, hf, bf: hf.have4thoverpair,
        "have5thoverpair":       lambda c, hf, bf: hf.have5thoverpair,
        "pairinhand":            lambda c, hf, bf: hf.pairinhand,
        "havetwopair":           lambda c, hf, bf: hf.havetwopair,
        "havetoptwopair":        lambda c, hf, bf: hf.havetoptwopair,
        "haveset":               lambda c, hf, bf: hf.haveset,
        "havetopset":            lambda c, hf, bf: hf.havetopset,
        "havebottomset":         lambda c, hf, bf: hf.havebottomset,
        "havetrips":             lambda c, hf, bf: hf.havetrips,
        "havestraight":          lambda c, hf, bf: hf.havestraight,
        "havenutstraight":       lambda c, hf, bf: hf.havenutstraight,
        "have2ndnutstraight":    lambda c, hf, bf: hf.have2ndnutstraight,
        "haveunderstraight":     lambda c, hf, bf: hf.haveunderstraight,
        "haveflush":             lambda c, hf, bf: hf.haveflush,
        "havenutflush":          lambda c, hf, bf: hf.havenutflush,
        "have2ndnutflush":       lambda c, hf, bf: hf.have2ndnutflush,
        "have3rdnutflush":       lambda c, hf, bf: hf.have3rdnutflush,
        "havefullhouse":         lambda c, hf, bf: hf.havefullhouse,
        "havequads":             lambda c, hf, bf: hf.havequads,
        "havestraightflush":     lambda c, hf, bf: hf.havestraightflush,
        "havenutstraightflush":  lambda c, hf, bf: hf.havenutstraightflush,
        "havenuts":              lambda c, hf, bf: hf.havenuts,
        "havenothing":           lambda c, hf, bf: hf.havenothing,
        "havetopnonboardpairedpair": lambda c, hf, bf: hf.havetopnonboardpairedpair,

        # Drawing-hand classes
        "havestraightdraw":          lambda c, hf, bf: hf.havestraightdraw,
        "haveinsidestraightdraw":    lambda c, hf, bf: hf.haveinsidestraightdraw,
        "haveinsidenutstraightdraw": lambda c, hf, bf: hf.haveinsidenutstraightdraw,
        "havenutstraightdraw":       lambda c, hf, bf: hf.havenutstraightdraw,
        "haveflushdraw":             lambda c, hf, bf: hf.haveflushdraw,
        "havenutflushdraw":          lambda c, hf, bf: hf.havenutflushdraw,
        "have2ndnutflushdraw":       lambda c, hf, bf: hf.have2ndnutflushdraw,
        "have3rdnutflushdraw":       lambda c, hf, bf: hf.have3rdnutflushdraw,
        "have4thnutflushdraw":       lambda c, hf, bf: hf.have4thnutflushdraw,
        "have5thnutflushdraw":       lambda c, hf, bf: hf.have5thnutflushdraw,
        "havebackdoorflushdraw":     lambda c, hf, bf: hf.havebackdoorflushdraw,
        "havebackdoornutflushdraw":  lambda c, hf, bf: hf.havebackdoornutflushdraw,

        # Kicker quality
        "havebestkicker":           lambda c, hf, bf: hf.havebestkicker,
        "have2ndbestkicker":        lambda c, hf, bf: hf.have2ndbestkicker,
        "have3rdbestkicker":        lambda c, hf, bf: hf.have3rdbestkicker,
        "havebestkickerorbetter":   lambda c, hf, bf: hf.havebestkickerorbetter,
        "have2ndbestkickerorbetter": lambda c, hf, bf: hf.have2ndbestkickerorbetter,
        "have3rdbestkickerorbetter": lambda c, hf, bf: hf.have3rdbestkickerorbetter,

        # ----- Board-shape predicates (read from BoardFeatures) -----
        "paironboard":           lambda c, hf, bf: bf.paironboard,
        "twopaironboard":        lambda c, hf, bf: bf.twopaironboard,
        "tripsonboard":          lambda c, hf, bf: bf.tripsonboard,
        "quadsonboard":          lambda c, hf, bf: bf.quadsonboard,
        "fullhouseonboard":      lambda c, hf, bf: bf.fullhouseonboard,
        "straightonboard":       lambda c, hf, bf: bf.straightonboard,
        "flushonboard":          lambda c, hf, bf: bf.flushonboard,
        "threecardstraightonboard": lambda c, hf, bf: bf.threecardstraightonboard,
        "flushpossible":         lambda c, hf, bf: bf.flushpossible,
        "straightpossible":      lambda c, hf, bf: bf.straightpossible,
        "straightflushpossible": lambda c, hf, bf: bf.straightflushpossible,
        "straightflushpossiblebyothers": lambda c, hf, bf: bf.straightflushpossiblebyothers,
        "onecardflushpossible":  lambda c, hf, bf: bf.onecardflushpossible,
        "onecardstraightpossible": lambda c, hf, bf: bf.onecardstraightpossible,
        "onecardstraightpossibleonturn": lambda c, hf, bf: bf.onecardstraightpossibleonturn,
        "morethanonestraightpossibleonflop": lambda c, hf, bf: bf.morethanonestraightpossibleonflop,
        "morethanonestraightpossibleonturn": lambda c, hf, bf: bf.morethanonestraightpossibleonturn,
        "onlyonestraightpossible": lambda c, hf, bf: bf.onlyonestraightpossible,
        "flushpossibleonflop":   lambda c, hf, bf: bf.flushpossibleonflop,
        "flushpossibleonturn":   lambda c, hf, bf: bf.flushpossibleonturn,
        "straightpossibleonflop": lambda c, hf, bf: bf.straightpossibleonflop,
        "straightpossibleonturn": lambda c, hf, bf: bf.straightpossibleonturn,
        "paironflop":            lambda c, hf, bf: bf.paironflop,
        "paironturn":            lambda c, hf, bf: bf.paironturn,
        "acepresentonflop":      lambda c, hf, bf: bf.acepresentonflop,
        "kingpresentonflop":     lambda c, hf, bf: bf.kingpresentonflop,
        "queenpresentonflop":    lambda c, hf, bf: bf.queenpresentonflop,
        "uncoordinatedflop":     lambda c, hf, bf: bf.uncoordinatedflop,
        "topflopcardpairedonturn": lambda c, hf, bf: bf.topflopcardpairedonturn,
        "rivercardisovercardtoboard": lambda c, hf, bf: bf.rivercardisovercardtoboard,

        # ----- Player/action-state predicates (read from GameContext) -----
        "opponentisallin":       lambda c, hf, bf: c.opponentisallin_flag,
        "botislastraiser":       lambda c, hf, bf: c.botislastraiser_flag,
        "inbigblind":            lambda c, hf, bf: c.position == "bigblind",
        "insmallblind":          lambda c, hf, bf: c.position == "smallblind",
        "inbutton":              lambda c, hf, bf: c.position == "button",
        "nobettingonflop":       lambda c, hf, bf: c.nobettingonflop_flag,
        "nobettingonturn":       lambda c, hf, bf: c.nobettingonturn_flag,
        "botraisedbeforeflop":   lambda c, hf, bf: c.botraisedbeforeflop_flag,
        "botraisedonflop":       lambda c, hf, bf: c.botraisedonflop_flag,
        "botraisedonturn":       lambda c, hf, bf: c.botraisedonturn_flag,
        "calledonflop":          lambda c, hf, bf: c.calledonflop_flag,
        "calledonturn":          lambda c, hf, bf: c.calledonturn_flag,
        "botcalledbeforeflop":   lambda c, hf, bf: c.botcalledbeforeflop_flag,
        "opponentcalledonturn":  lambda c, hf, bf: c.opponentcalledonturn_flag,

        # Previous-street hand memory
        "hadtoppaironflop":      lambda c, hf, bf: c.hadtoppaironflop_flag,
        "hadoverpaironflop":     lambda c, hf, bf: c.hadoverpaironflop_flag,
    }


_COMPARE_LHS = _compare_lhs_dict()
_BARE_PREDICATES = _bare_predicate_dict()


# ============================================================
# Expression evaluator
# ============================================================

class UnsupportedPredicateError(Exception):
    """Raised when an expression references a predicate not in the dictionary.

    Should never happen in practice if the parser and runtime dictionaries
    are in sync; serves as a tripwire for missing implementations.
    """


class Runtime:
    """Per-decision evaluator.

    Constructed fresh for each decision. Tracks user-flags set by `userfoo`
    actions during this decision so that subsequent rules can test them
    via `userfoo` as a bare predicate.
    """

    def __init__(self, profile: Profile, ctx: GameContext):
        self.profile = profile
        self.ctx = ctx
        self.hand_features = compute_hand_features(ctx.hole_cards, ctx.board)
        self.board_features = compute_board_features(ctx.board)
        self.user_flags: Set[str] = set()
        # `others` evaluates True only when no prior rule in the current
        # section has matched. We track this in iter_section.
        self._any_prior_matched: bool = False

    def evaluate_expression(self, e: Expr) -> bool:
        """Evaluate a boolean expression. Returns True/False."""
        if isinstance(e, BoolOp):
            if e.op == "and":
                return all(self.evaluate_expression(a) for a in e.args)
            if e.op == "or":
                return any(self.evaluate_expression(a) for a in e.args)
            raise ValueError(f"unknown BoolOp: {e.op}")

        if isinstance(e, Not):
            return not self.evaluate_expression(e.arg)

        if isinstance(e, OthersAtom):
            # True iff no prior rule in this section has matched.
            return not self._any_prior_matched

        if isinstance(e, PositionPred):
            return self.ctx.position == e.position

        if isinstance(e, PredCall):
            name = e.name
            # User-flag reference: True iff the flag is set in this decision.
            if name.startswith("user"):
                return name in self.user_flags
            # Look up in bare predicate dict.
            fn = _BARE_PREDICATES.get(name)
            if fn is None:
                raise UnsupportedPredicateError(
                    f"unsupported bare predicate {name!r} "
                    f"(profile: {self.profile.source_name})"
                )
            return bool(fn(self.ctx, self.hand_features, self.board_features))

        if isinstance(e, Compare):
            lhs_name = e.lhs
            # Hand and board specs are handled specially.
            if lhs_name == "hand" and isinstance(e.rhs, HandSpec):
                return _eval_hand_compare(e.op, e.rhs, self.ctx.hole_cards)
            if lhs_name == "board" and isinstance(e.rhs, BoardSpec):
                return _eval_board_compare(e.op, e.rhs, self.ctx.board)
            # Otherwise it's a numeric or string compare.
            lhs_fn = _COMPARE_LHS.get(lhs_name)
            if lhs_fn is None:
                raise UnsupportedPredicateError(
                    f"unsupported compare LHS {lhs_name!r} "
                    f"(profile: {self.profile.source_name})"
                )
            lhs_val = lhs_fn(self.ctx, self.hand_features, self.board_features)
            rhs_val = self._eval_rhs(e.rhs)
            return _compare_values(lhs_val, e.op, rhs_val)

        raise ValueError(f"unknown expression type: {type(e).__name__}")

    def _eval_rhs(self, rhs) -> object:
        """Evaluate the RHS of a Compare into a value."""
        if isinstance(rhs, NumberLit):
            return rhs.value
        if isinstance(rhs, IdentLit):
            return rhs.name
        if isinstance(rhs, PercentExpr):
            # Resolve target to a numeric value (potsize, stacksize, etc.).
            target_fn = _COMPARE_LHS.get(rhs.target)
            if target_fn is None:
                raise UnsupportedPredicateError(
                    f"unknown percent target {rhs.target!r}"
                )
            target_val = target_fn(self.ctx, self.hand_features, self.board_features)
            try:
                return (rhs.pct / 100.0) * float(target_val)
            except (TypeError, ValueError):
                return 0.0
        raise ValueError(f"unknown RHS type: {type(rhs).__name__}")

    def iter_section(self, section: Section) -> Optional[Action]:
        """Walk a section's rules in order. Return the first matching rule's
        action, or None if no rule matched.

        Implements user-flag side effects: a SET_USER_FLAG action sets the
        flag and continues evaluating rules; any other action terminates
        and returns it.
        """
        self._any_prior_matched = False
        for rule in section.rules:
            try:
                matched = self.evaluate_expression(rule.condition)
            except UnsupportedPredicateError:
                # Skip rules with unsupported predicates so the profile can
                # still produce sensible output for the rules it CAN evaluate.
                # In strict mode this would re-raise.
                continue
            if not matched:
                continue
            self._any_prior_matched = True
            if rule.action.kind == ActionKind.SET_USER_FLAG:
                # Side effect: set the flag, continue evaluating.
                if rule.action.user_flag:
                    self.user_flags.add(rule.action.user_flag)
                continue
            return rule.action
        return None


def _compare_values(lhs, op: str, rhs) -> bool:
    """Apply the comparison operator. Strings compared by ==/!= only."""
    if isinstance(lhs, str) or isinstance(rhs, str):
        if op == "=":
            return str(lhs) == str(rhs)
        if op == "<>":
            return str(lhs) != str(rhs)
        return False
    lhs_n = float(lhs)
    rhs_n = float(rhs)
    if op == "=":
        return lhs_n == rhs_n
    if op == "<":
        return lhs_n < rhs_n
    if op == "<=":
        return lhs_n <= rhs_n
    if op == ">":
        return lhs_n > rhs_n
    if op == ">=":
        return lhs_n >= rhs_n
    if op == "<>":
        return lhs_n != rhs_n
    raise ValueError(f"unknown comparison operator: {op}")


def _eval_hand_compare(op: str, spec: HandSpec, hole_cards: List[int]) -> bool:
    """Evaluate `hand = X` (or `hand <> X`)."""
    matched = matches_hand_spec(spec, hole_cards)
    if op == "=":
        return matched
    if op == "<>":
        return not matched
    return False


def _eval_board_compare(op: str, spec: BoardSpec, board: List[int]) -> bool:
    """Evaluate `board = X` (or `board <> X`)."""
    matched = matches_board_spec(spec, board)
    if op == "=":
        return matched
    if op == "<>":
        return not matched
    return False


# ============================================================
# Public entry point
# ============================================================

def evaluate_profile(profile: Profile, ctx: GameContext) -> Optional[Action]:
    """Run the profile against the given game context. Returns the first
    matching rule's action in the section corresponding to ctx.street, or
    None if no rule matched.

    The returned Action is the raw AST node — the caller is responsible
    for translating it into their action space (call policy_adapter.py
    for the discrete-action translation).
    """
    rt = Runtime(profile, ctx)
    # Find the section matching the current street.
    target_section: Optional[Section] = None
    for sec in profile.sections:
        if sec.name == ctx.street:
            target_section = sec
            break
    if target_section is None:
        return None
    return rt.iter_section(target_section)
