"""Token construction for the Leduc adaptive-policy net (arch doc §2, §3, §8).

This module is the LOAD-BEARING CORRECTNESS SURFACE of the whole proof: the
information-leakage invariant (§8) is enforced *here*, by construction, and
verified by tests/test_leduc_token_no_leak.py. The single most important
property:

    At any prediction position, the model's inputs (token stream + opp_stats)
    contain ZERO information about ANY opponent's private (hole) card.

We guarantee this by NEVER READING the opponent's dealt card while building
tokens or stats. The only private-card slot in a token is the HERO's, and it
is filled only at hero decision tokens. Opp's card has no slot anywhere.

OpenSpiel leduc_poker facts this code relies on (verified 2026-05-31):
  - actions: 0=Fold, 1=Call/Check, 2=Raise
  - 6 cards (indices 0..5); rank = card // 2  (0,1->J ; 2,3->Q ; 4,5->K)
  - chance/deal order: deal #0 -> player 0, deal #1 -> player 1,
    deal #2 -> public (revealed for round 2 only)
  - "facing a bet" at a decision  <=>  FOLD is legal there
  - per-street raise cap = 2
  - betting rounds: 1 = preflop, 2 = post-public ("flop")
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pyspiel

FOLD, CALL, RAISE = 0, 1, 2
NUM_ACTIONS = 3

# --------------------------------------------------------------------------
# Token field layout (17 dims). This layout is the SINGLE SOURCE OF TRUTH and
# is hashed by leakage Test A — any silent change to the byte layout fails the
# test loudly. Do not reorder without updating the test's expected manifest.
# --------------------------------------------------------------------------
F_RAW = 17

# Field -> (offset, width). Documented; consumed by TOKEN_MANIFEST below.
FIELDS = (
    ("actor_is_opp", 0, 1),    # 0 = hero, 1 = opponent
    ("street",       1, 2),    # one-hot [preflop, flop]
    ("public_card",  3, 3),    # one-hot [J, Q, K]; zeros in round 1
    ("facing_bet",   6, 1),    # 1 if a bet is currently faced
    ("raises_street", 7, 2),   # one-hot {0,1}; all-zero means >=2 (raise illegal)
    ("hero_card",    9, 3),    # one-hot [J, Q, K]; NON-ZERO ONLY at hero tokens
    ("action_taken", 12, 3),   # one-hot [fold, call, raise]; zero at query token
    ("hand_boundary", 15, 1),  # 1 at the first decision of a (new) hand
    ("pad",          16, 1),   # reserved zero
)

# Stable manifest string the leakage test pins against.
TOKEN_MANIFEST = "leduc-adaptive-token-v1|" + "|".join(
    f"{name}@{off}:{off + w}" for name, off, w in FIELDS
)
TOKEN_MANIFEST_SHA = hashlib.sha256(TOKEN_MANIFEST.encode()).hexdigest()

# --------------------------------------------------------------------------
# opp_stats layout v3 (20 dims) — per-street split.
#
# v1 (6 dims): smoothed facing rates + AF + open_raise + confidence weight.
# v2 (12 dims): + 2 unsmoothed raw fractions + last-action one-hot + no-yet.
# v3 (20 dims): split the 8 action-frequency features by street (preflop vs
# postflop) so a maniac's preflop-raise character isn't smeared with its
# postflop cap-forced calls. The 4-dim last-action snapshot stays
# instantaneous (a moment, not a rate). confidence_weight IS split per
# street so the head can learn "postflop confidence is 0 here, ignore the
# postflop block".
#
# Every new field remains a pure function of (action, facing_bet, street,
# previous_action). cur_round transitions live on the chance-node deal
# schedule (deal_count==2), independent of any card value — §8 invariant
# unchanged; Test B (counterfactual bit-identity) covers the v3 opp_stats
# vector and must stay green.
# --------------------------------------------------------------------------
F_STATS = 20

STATS_FIELDS = (
    # Preflop block (8 dims) — Leduc round 1
    ("preflop_facing_fold_rate_smoothed",   0),
    ("preflop_facing_call_rate_smoothed",   1),
    ("preflop_facing_raise_rate_smoothed",  2),
    ("preflop_aggression_factor_smoothed",  3),
    ("preflop_open_raise_rate_smoothed",    4),
    ("preflop_confidence_weight",           5),
    ("preflop_raw_raise_fraction",          6),
    ("preflop_raw_fold_fraction",           7),
    # Postflop block (8 dims) — Leduc round 2 (post-public-card)
    ("postflop_facing_fold_rate_smoothed",   8),
    ("postflop_facing_call_rate_smoothed",   9),
    ("postflop_facing_raise_rate_smoothed", 10),
    ("postflop_aggression_factor_smoothed", 11),
    ("postflop_open_raise_rate_smoothed",   12),
    ("postflop_confidence_weight",          13),
    ("postflop_raw_raise_fraction",         14),
    ("postflop_raw_fold_fraction",          15),
    # Last-action snapshot (4 dims) — instantaneous, NOT per-street
    ("last_action_is_fold",  16),
    ("last_action_is_call",  17),
    ("last_action_is_raise", 18),
    ("no_actions_yet",       19),
)

STATS_MANIFEST = "leduc-adaptive-opp-stats-v3|" + "|".join(
    f"{name}@{idx}" for name, idx in STATS_FIELDS
)
STATS_MANIFEST_SHA = hashlib.sha256(STATS_MANIFEST.encode()).hexdigest()

_GAME = pyspiel.load_game("leduc_poker")


def game():
    return _GAME


def card_rank(card_index: int) -> int:
    """Leduc rank in {0=J, 1=Q, 2=K} from a card index 0..5."""
    return card_index // 2


# --------------------------------------------------------------------------
# Opponent running-summary statistics (§3). Function of OBSERVED OPP ACTIONS
# ONLY — never the opp's card. Mild Laplace smoothing (+1 / +|legal|).
# --------------------------------------------------------------------------

@dataclass
class _OppCounters:
    # Preflop counts (Leduc round 1)
    p_facing_fold: int = 0
    p_facing_call: int = 0
    p_facing_raise: int = 0
    p_notfacing_call: int = 0
    p_notfacing_raise: int = 0
    p_n: int = 0
    # Postflop counts (Leduc round 2)
    f_facing_fold: int = 0
    f_facing_call: int = 0
    f_facing_raise: int = 0
    f_notfacing_call: int = 0
    f_notfacing_raise: int = 0
    f_n: int = 0
    # Instantaneous last action (across both streets)
    last_action: int = -1     # -1 = no opp action yet; else 0/1/2

    def update(self, action: int, facing_bet: bool, street: int):
        """street: 1 = preflop, 2 = postflop. Routes counts to the correct
        per-street block; v3 split for opponent-character discrimination."""
        self.last_action = action
        if street == 1:
            self.p_n += 1
            if facing_bet:
                if action == FOLD:
                    self.p_facing_fold += 1
                elif action == CALL:
                    self.p_facing_call += 1
                else:
                    self.p_facing_raise += 1
            else:
                if action == RAISE:
                    self.p_notfacing_raise += 1
                else:  # CALL == check
                    self.p_notfacing_call += 1
        else:  # street == 2 (postflop)
            self.f_n += 1
            if facing_bet:
                if action == FOLD:
                    self.f_facing_fold += 1
                elif action == CALL:
                    self.f_facing_call += 1
                else:
                    self.f_facing_raise += 1
            else:
                if action == RAISE:
                    self.f_notfacing_raise += 1
                else:
                    self.f_notfacing_call += 1

    def _street_block(self, facing_fold, facing_call, facing_raise,
                       notfacing_call, notfacing_raise, n):
        """Compute the 8-dim block for ONE street from its raw counts."""
        f = facing_fold + facing_call + facing_raise
        nf = notfacing_call + notfacing_raise
        fold_r = (facing_fold + 1.0) / (f + 3.0)
        call_r = (facing_call + 1.0) / (f + 3.0)
        raise_r = (facing_raise + 1.0) / (f + 3.0)
        all_raise = facing_raise + notfacing_raise
        all_call = facing_call + notfacing_call
        af = (all_raise + 1.0) / (all_raise + all_call + 2.0)
        open_r = (notfacing_raise + 1.0) / (nf + 2.0)
        conf = min(1.0, np.log1p(n) / np.log(101.0))
        raw_raise = (all_raise / n) if n > 0 else 0.0
        raw_fold = (facing_fold / f) if f > 0 else 0.0
        return [fold_r, call_r, raise_r, af, open_r, conf, raw_raise, raw_fold]

    def vector(self) -> np.ndarray:
        # 0-7: preflop block
        p_block = self._street_block(
            self.p_facing_fold, self.p_facing_call, self.p_facing_raise,
            self.p_notfacing_call, self.p_notfacing_raise, self.p_n,
        )
        # 8-15: postflop block
        f_block = self._street_block(
            self.f_facing_fold, self.f_facing_call, self.f_facing_raise,
            self.f_notfacing_call, self.f_notfacing_raise, self.f_n,
        )
        # 16-18: last opp action one-hot (zero if last_action == -1)
        last_is_fold = 1.0 if self.last_action == FOLD else 0.0
        last_is_call = 1.0 if self.last_action == CALL else 0.0
        last_is_raise = 1.0 if self.last_action == RAISE else 0.0
        # 19: no opp actions yet (across BOTH streets)
        no_actions = 1.0 if (self.p_n + self.f_n) == 0 else 0.0
        return np.array(
            [*p_block, *f_block,
             last_is_fold, last_is_call, last_is_raise, no_actions],
            dtype=np.float32,
        )


def empty_opp_stats() -> np.ndarray:
    """opp_stats with no observed opp decisions (smoothed priors)."""
    return _OppCounters().vector()


# --------------------------------------------------------------------------
# Core walk: replay an action history and emit one token per decision node.
# Reads ONLY hero's dealt card + the public card. Never reads opp's card.
# --------------------------------------------------------------------------

def _new_token() -> np.ndarray:
    return np.zeros(F_RAW, dtype=np.float32)


def _fill_public_state(tok: np.ndarray, *, actor_is_hero: bool, cur_round: int,
                       public_rank, facing_bet: bool, raises_street: int,
                       hero_card_rank, hand_boundary: bool):
    tok[0] = 0.0 if actor_is_hero else 1.0
    # street one-hot [preflop, flop]
    tok[1 + (cur_round - 1)] = 1.0
    # public card one-hot (round 2 only)
    if public_rank is not None:
        tok[3 + public_rank] = 1.0
    tok[6] = 1.0 if facing_bet else 0.0
    # raises-this-street one-hot {0,1}; all-zero if >=2
    if raises_street in (0, 1):
        tok[7 + raises_street] = 1.0
    # hero private card one-hot — ONLY at hero tokens (load-bearing, §8)
    if actor_is_hero and hero_card_rank is not None:
        tok[9 + hero_card_rank] = 1.0
    tok[15] = 1.0 if hand_boundary else 0.0
    # tok[16] pad stays 0


def _walk(history, hero_seat: int):
    """Replay `history` (list of OpenSpiel actions incl. chance) for a single
    Leduc hand. Yields a token per decision node and tracks opp stats.

    Returns (tokens, actor_is_hero, legal_masks, counters, ctx) where ctx is
    the live post-replay state plus public bookkeeping needed to build a query
    token for the *current* (unplayed) decision node.

    NOTE: opp's dealt card (the deal to 1-hero_seat) is intentionally never
    read into any token or counter.
    """
    state = _GAME.new_initial_state()
    deal_count = 0
    hero_card_rank = None
    public_rank = None
    cur_round = 1
    raises_street = 0
    first_decision = True

    tokens: list[np.ndarray] = []
    actor_is_hero: list[bool] = []
    legal_masks: list[np.ndarray] = []
    counters = _OppCounters()

    for a in history:
        if state.is_chance_node():
            if deal_count == hero_seat:
                hero_card_rank = card_rank(a)
            elif deal_count == 2:
                public_rank = card_rank(a)
                cur_round = 2
                raises_street = 0
            # deal_count == (1 - hero_seat) is the OPP card: deliberately NOT read.
            deal_count += 1
            state.apply_action(a)
            continue

        # decision node
        player = state.current_player()
        legal = state.legal_actions()
        facing = FOLD in legal
        is_hero = (player == hero_seat)

        tok = _new_token()
        _fill_public_state(
            tok, actor_is_hero=is_hero, cur_round=cur_round, public_rank=public_rank,
            facing_bet=facing, raises_street=raises_street,
            hero_card_rank=hero_card_rank, hand_boundary=first_decision,
        )
        # action_taken one-hot
        tok[12 + a] = 1.0

        tokens.append(tok)
        actor_is_hero.append(is_hero)
        lm = np.zeros(NUM_ACTIONS, dtype=bool)
        for la in legal:
            lm[la] = True
        legal_masks.append(lm)

        if not is_hero:
            counters.update(a, facing, cur_round)
        if a == RAISE:
            raises_street += 1
        first_decision = False
        state.apply_action(a)

    ctx = {
        "state": state,
        "cur_round": cur_round,
        "public_rank": public_rank,
        "raises_street": raises_street,
        "hero_card_rank": hero_card_rank,
        "first_decision": first_decision,
    }
    return tokens, actor_is_hero, legal_masks, counters, ctx


# --------------------------------------------------------------------------
# Public entry points.
# --------------------------------------------------------------------------

def tokenize_full_hand(history, hero_seat: int) -> dict:
    """Tokens for EVERY decision in a complete hand (no query token).

    Used by the leakage tests: every emitted token + the final opp_stats must
    be invariant to substituting the opponent's hole card.
    """
    tokens, actor_is_hero, legal_masks, counters, _ = _walk(history, hero_seat)
    T = len(tokens)
    return {
        "tokens": np.stack(tokens) if T else np.zeros((0, F_RAW), np.float32),
        "actor_is_hero": np.array(actor_is_hero, dtype=bool),
        "legal_masks": np.stack(legal_masks) if T else np.zeros((0, NUM_ACTIONS), bool),
        "opp_stats": counters.vector(),
        "n_tokens": T,
    }


def tokenize_decision(state) -> dict:
    """Prefix tokens (this hand's prior decisions) + a query token for the
    CURRENT decision node `state`. hero = the player to act at `state`.

    The query token is a hero token with the current public state and a ZERO
    action slot (the action is what we predict — §2). Returns everything the
    net needs for one forward pass at this decision.
    """
    if state.is_chance_node() or state.is_terminal():
        raise ValueError("tokenize_decision requires a player decision node")
    hero_seat = state.current_player()
    tokens, actor_is_hero, legal_masks, counters, ctx = _walk(state.history(), hero_seat)

    legal = state.legal_actions()
    facing = FOLD in legal
    query = _new_token()
    _fill_public_state(
        query, actor_is_hero=True, cur_round=ctx["cur_round"],
        public_rank=ctx["public_rank"], facing_bet=facing,
        raises_street=ctx["raises_street"], hero_card_rank=ctx["hero_card_rank"],
        hand_boundary=ctx["first_decision"],
    )
    # query action slot stays zero (masked)
    tokens.append(query)
    actor_is_hero.append(True)
    query_idx = len(tokens) - 1

    lm = np.zeros(NUM_ACTIONS, dtype=bool)
    for la in legal:
        lm[la] = True

    return {
        "tokens": np.stack(tokens).astype(np.float32),
        "query_idx": query_idx,
        "legal_mask": lm,
        "opp_stats": counters.vector(),
        "info_state": state.information_state_string(),
        "legal_actions": legal,
    }
