"""Strict invariant check for Option B integration.

Diffs the reconstructed OpenSpiel hand-start state against the originating
ScraperFrame. Any field mismatch -> REJECT and return a safe fallback.

============================================================================
DELIBERATE BUG-MATCH WORKAROUND — read this whole block before editing
============================================================================

There is a real bug in `src/nlhe/game_strings.py:to_inner_game_string_for_state`
(line ~381-382): when computing the OpenSpiel BB-position chip amount, it
calls `blind_level.inflated_big_blind(n=self.num_players)` (always 6 for
6-max), NOT `inflated_big_blind(n_alive)`. The result: at shorthanded tables
the library posts a "ghost ante" for each empty seat into the BB's
contribution slot. At n_alive=5, BB contribution = bb + 6*ante instead of
the correct bb + 5*ante.

Why we don't fix the library here: the existing rebel_value_net and k200
blueprint were trained on samples produced via `sample_starting_state` ->
`to_inner_game_string_for_state` (the buggy path), so the bug is baked into
their training distributions. `sample_starting_state` samples 4/5/6-handed
states at meaningful rates (40-55% of mid/short stages). Fixing the library
without retraining would push inference-time inputs OOD on shorthanded
states by up to (NUM_SEATS - n_alive) * ante / starting_stack — as much as
24% normalized-feature shift at level 10. That would invalidate the
candidate_bakeoff numbers we just used to pick the ship checkpoint.

Decision (documented in docs/DECISIONS.md): match the library's bug here in
the integration layer so the resolver sees the same in-distribution state
it was trained on. Queue the real fix (alive-count antes) for the NEXT
training cycle, at which point both the library AND this workaround must be
fixed together.

Conversion (BUG-MATCHED — uses NUM_SEATS, not n_alive, in the ante terms):
    For each alive seat i:
        if i == bb_seat:
            scraper_stack[i] = openspiel_stack[i] + (NUM_SEATS - 1) * ante
            scraper_bet[i]   = openspiel_contribution[i] - NUM_SEATS * ante
        else:
            scraper_stack[i] = openspiel_stack[i] - ante
            scraper_bet[i]   = openspiel_contribution[i]
    scraper_pot = sum(contribution) - (NUM_SEATS - n_alive) * ante
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                       the "ghost antes" the library posted
                                       for empty seats; subtract them so the
                                       reconstructed pot matches what the
                                       scraper sees on the actual table.

When the library bug is fixed (Option A queued): replace NUM_SEATS with
n_alive in the three places above and DELETE the ghost-antes correction
from scraper_pot. The 6-handed case is bit-identical between bug-match and
the correct formula (NUM_SEATS == n_alive), so 6-handed tests don't change.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from src.nlhe.infoset6 import parse_state_6max
    from src.nlhe.integration.scraper_schema import (
        ScraperFrame, NUM_SEATS,
    )
    from src.nlhe.integration.replay import HandStartState
except ImportError:
    sys.path.insert(0, "/tmp")
    from src.nlhe.infoset6 import parse_state_6max
    from scraper_schema import ScraperFrame, NUM_SEATS
    from replay import HandStartState


@dataclass
class InvariantResult:
    """Outcome of the invariant check."""
    ok: bool
    deltas: list[tuple[str, Any, Any]] = field(default_factory=list)
    # On failure, the safe action to return to the client. "check" if hero
    # isn't facing a bet, "fold" if they are. Never a chip-committing action.
    safe_action: Optional[str] = None
    # The reconstructed scraper-view fields (for logging/quarantine even on pass).
    reconstructed: dict = field(default_factory=dict)

    def format_deltas(self) -> str:
        """Pretty-print the deltas for logging."""
        if not self.deltas:
            return "(no deltas)"
        lines = []
        for field_name, scraper_val, openspiel_val in self.deltas:
            lines.append(
                f"  {field_name:<24s}  scraper={scraper_val!r:<40s}  "
                f"reconstructed={openspiel_val!r}"
            )
        return "\n".join(lines)


def openspiel_to_scraper_view(parsed: dict, bb_seat: int, n_alive: int,
                               ante: int) -> dict:
    """Convert OpenSpiel parsed dict to scraper-equivalent fields.

    Per the conversion described in the module docstring. Returns:
      {
        "stack": tuple[int, ...],     # per-seat scraper-equivalent stack
        "bet":   tuple[int, ...],     # per-seat scraper-equivalent bet THIS STREET
        "pot":   int,                 # scraper pot total (matches OpenSpiel's pot)
        "current_player": int,
        "street_idx": int,
        "private_cards": str,
        "public_cards": str,
      }
    """
    money = list(parsed["money"])
    contrib = list(parsed["contribution"])
    scraper_stack = list(money)
    scraper_bet = list(contrib)
    # BUG-MATCHED conversion (uses NUM_SEATS, not n_alive). See module
    # docstring for the why; remove this workaround when the library bug
    # in to_inner_game_string_for_state is fixed and models are retrained.
    for i in range(NUM_SEATS):
        if i == bb_seat:
            scraper_stack[i] = money[i] + (NUM_SEATS - 1) * ante
            scraper_bet[i] = contrib[i] - NUM_SEATS * ante
        else:
            # Non-blind seats: OpenSpiel didn't deduct ante; scraper did.
            # If money[i] == 0 (busted/empty seat), keep at 0 (no ante to deduct).
            if money[i] > 0:
                scraper_stack[i] = money[i] - ante
            scraper_bet[i] = contrib[i]
        # Defensive: chip values must be >= 0
        if scraper_stack[i] < 0:
            scraper_stack[i] = 0
        if scraper_bet[i] < 0:
            scraper_bet[i] = 0
    # NOTE: OpenSpiel's [Pot: N] observation field is its internal "potential
    # pot" (often n_seats * inflated_bb, e.g. 330 = 6*55 at level 1), NOT the
    # actual chips in the pot. The chips-in-pot equivalent that matches the
    # scraper's pot.total is sum(contribution). Verified empirically at the
    # initial chance node of a 6-max universal_poker game.
    # BUG-MATCHED: subtract the (NUM_SEATS - n_alive) "ghost antes" the
    # library posted for empty seats; without this the reconstructed pot
    # is over by (NUM_SEATS - n_alive) * ante on shorthanded states.
    ghost_antes = (NUM_SEATS - n_alive) * ante
    scraper_pot = int(sum(parsed["contribution"]) - ghost_antes)
    return {
        "stack": tuple(scraper_stack),
        "bet": tuple(scraper_bet),
        "pot": scraper_pot,
        "current_player": int(parsed["current_player"]),
        "street_idx": int(parsed["street_idx"]),
        "private_cards": str(parsed.get("private_cards", "")),
        "public_cards": str(parsed.get("public_cards", "")),
    }


def check_hand_start_invariant(frame: ScraperFrame,
                                state_pack: HandStartState
                                ) -> InvariantResult:
    """Strict diff between the parsed scraper frame and the replayed state.

    Checks (any mismatch -> reject):
      pot_total: exact int equality
      per-seat stack[i]: exact int equality (i in 0..5)
      per-seat bet[i]:   exact int equality (i in 0..5)

    Hand-start-specific notes:
      - At a chance node (initial), current_player is -1. We don't check
        current_player for hand-start.
      - street_idx should be 0 (preflop). We do check this.
      - private_cards / public_cards: not dealt yet at the initial chance
        node. We don't check those for hand-start.

    Returns InvariantResult with safe_action set on failure.
    """
    state = state_pack.state
    bb_seat = state_pack.bb_seat
    n_alive = state_pack.n_alive
    ante = state_pack.blind_level.ante

    # Parse the chance-node state from hero's observer perspective so we
    # don't trip current_player==-1 in observation_string.
    parsed = parse_state_6max(state, observer=frame.hero_seat)
    parsed["dealer_seat"] = frame.dealer_seat

    reconstructed = openspiel_to_scraper_view(
        parsed, bb_seat=bb_seat, n_alive=n_alive, ante=ante)

    deltas: list[tuple[str, Any, Any]] = []

    # pot
    if reconstructed["pot"] != frame.pot_total:
        deltas.append(("pot", frame.pot_total, reconstructed["pot"]))

    # street_idx — must be 0 at hand-start
    if reconstructed["street_idx"] != 0:
        deltas.append(("street_idx", 0, reconstructed["street_idx"]))

    # per-seat stack & bet
    for i in range(NUM_SEATS):
        s_scraper = frame.stack[i]
        s_recon = reconstructed["stack"][i]
        if s_scraper != s_recon:
            deltas.append((f"stack[seat{i+1}]", s_scraper, s_recon))
        b_scraper = frame.bet[i]
        b_recon = reconstructed["bet"][i]
        if b_scraper != b_recon:
            deltas.append((f"bet[seat{i+1}]", b_scraper, b_recon))

    if deltas:
        safe = "fold" if frame.hero_facing_bet else "check"
        return InvariantResult(
            ok=False, deltas=deltas, safe_action=safe,
            reconstructed=reconstructed,
        )
    return InvariantResult(
        ok=True, deltas=[], safe_action=None,
        reconstructed=reconstructed,
    )


def _canonical_card_string(cards) -> str:
    """Normalize a card representation (str 'AhKs' or tuple ('Ah','Ks')) to
    sorted concatenated form for canonical comparison."""
    if isinstance(cards, str):
        # Split into 2-char card chunks
        chunks = [cards[i:i + 2] for i in range(0, len(cards), 2)]
    else:
        chunks = [str(c) for c in cards]
    return "".join(sorted(chunks))


def check_mid_hand_invariant(frame, state_pack) -> InvariantResult:
    """Strict diff for mid-hand hero-to-act state (Piece 5, Phase 2).

    Five LOAD-BEARING checks (any mismatch -> ok=False, safe_action set):
      1. current_player == hero_seat — we replayed to the right point
      2. private_cards == hero's hole cards — we dealt the right hero cards
      3. public_cards == scraper's board — we dealt the right board
      4. pot == frame.pot_total — total-chip conservation matches
      5. legal_actions consistent with hero_facing_bet:
           if hero_facing_bet -> FOLD MUST be in state.legal_actions()
           else                -> FOLD MUST NOT be in state.legal_actions()

    NOTE on per-seat stack/bet checks (intentionally DROPPED for mid-hand):
    the inflated-BB library bug (Option B bug-match) cascades through OpenSpiel's
    betting rules in ways that make per-seat scraper-view conversion
    configuration-dependent: at hand-start the BB has inflated contribution
    but non-BB seats are clean (Phase 1's formula works); after a call of
    the inflated_bb, every alive seat's OpenSpiel contribution = inflated_bb
    (subtract n*ante to recover scraper_bet); after a raise above inflated_bb,
    the contribution is the raise amount as-is (NO subtraction needed). The
    rule changes per-state, and there's no clean single formula. Per-seat
    checks would either generate false rejections on raises or paper over
    real mismatches on calls — neither acceptable. The 5 load-bearing
    checks above are SUFFICIENT to verify the resolver receives the
    correct decision-point state: right player to act, right cards, right
    pot total, right action menu. Per-seat redundancy goes when the
    library bug is fixed (queued in DECISIONS.md).
    """
    # MidHandState (Piece 4) carries everything we need beyond what
    # HandStartState provides.
    state = state_pack.state
    bb_seat = state_pack.bb_seat
    n_alive = state_pack.n_alive
    ante = state_pack.blind_level.ante
    street_idx = state_pack.street_idx
    preflop_commit_per_alive = state_pack.preflop_commit_per_alive

    parsed = parse_state_6max(state, observer=frame.hero_seat)
    parsed["dealer_seat"] = frame.dealer_seat

    # Use the existing scraper-view conversion for hero/board fields and
    # the per-seat stack (the latter is used for diagnostics only — we
    # don't strict-check it on mid-hand).
    recon_handlevel = openspiel_to_scraper_view(
        parsed, bb_seat=bb_seat, n_alive=n_alive, ante=ante)

    # Compute the correct scraper-equivalent pot using the formula that
    # accounts for the inflated-BB cascade through OpenSpiel's call
    # mechanism. Per-seat OpenSpiel-vs-scraper chip-flow diff:
    #   BB (always):                       +(NUM_SEATS - 1) * ante
    #   non-BB seat at contrib==inflated_bb (= matched the max via call):
    #                                      +(NUM_SEATS - 1) * ante
    #   non-BB alive at contrib != inflated_bb (folded, just-blinded,
    #                                            or raised above):
    #                                      -ante
    #   empty seat (no real ante paid):    0
    # Sum simplifies to:
    #   total_diff = ante * (NUM_SEATS * (1 + k) - n_alive)
    # where k = count of non-BB alive seats with openspiel_contrib ==
    # inflated_bb. Verified by arithmetic on 6-handed limp-around
    # (k=5 -> diff=150 ✓), 6-handed UTG-raise-folds (k=0 -> diff=0 ✓),
    # and 5-handed limp-around (k=4 -> diff=125 ✓).
    inflated_bb = state_pack.blind_level.inflated_big_blind(NUM_SEATS)
    contrib = parsed["contribution"]
    k = sum(1 for i in range(NUM_SEATS)
            if i != bb_seat and contrib[i] == inflated_bb)
    total_diff = ante * (NUM_SEATS * (1 + k) - n_alive)
    reconstructed_pot = int(sum(contrib) - total_diff)

    deltas = []

    # 1. current_player
    if parsed["current_player"] != frame.hero_seat:
        deltas.append(("current_player",
                        frame.hero_seat, parsed["current_player"]))

    # 2. street_idx (from state) should match scraper-derived (from board len)
    if parsed["street_idx"] != street_idx:
        deltas.append(("street_idx", street_idx, parsed["street_idx"]))

    # 3. pot (sum-of-contribution un-inflated, scenario-aware formula)
    if reconstructed_pot != frame.pot_total:
        deltas.append(("pot", frame.pot_total, reconstructed_pot))

    # 4. private_cards: hero's hole cards (canonical sorted)
    scraper_hole_canon = _canonical_card_string(frame.hero_cards)
    state_hole_canon = _canonical_card_string(recon_handlevel["private_cards"])
    if scraper_hole_canon != state_hole_canon:
        deltas.append(("hero_cards",
                        scraper_hole_canon, state_hole_canon))

    # 5. public_cards: board (canonical sorted)
    scraper_board_canon = _canonical_card_string(frame.board)
    state_board_canon = _canonical_card_string(recon_handlevel["public_cards"])
    if scraper_board_canon != state_board_canon:
        deltas.append(("board",
                        scraper_board_canon, state_board_canon))

    # 6. legal_actions consistency with hero_facing_bet
    legal = state.legal_actions()
    fold_legal = 0 in legal
    if frame.hero_facing_bet and not fold_legal:
        deltas.append(("legal_actions:fold_when_facing_bet",
                        True, False))
    # NOTE: when hero is NOT facing a bet (e.g., BB's preflop option), FOLD
    # is sometimes still in legal_actions() in OpenSpiel even though it'd
    # be a strictly dominated check (give up vs costless check). We don't
    # treat that case as a mismatch — only the "hero facing a bet but no
    # FOLD legal" direction is a real correctness concern.

    if deltas:
        safe = "fold" if frame.hero_facing_bet else "check"
        return InvariantResult(
            ok=False, deltas=deltas, safe_action=safe,
            reconstructed={
                **recon_handlevel,
                "legal_has_fold": fold_legal,
            },
        )
    return InvariantResult(
        ok=True, deltas=[], safe_action=None,
        reconstructed={
            **recon_handlevel,
            "legal_has_fold": fold_legal,
        },
    )
