"""Strict invariant check for Option B integration.

Diffs the reconstructed OpenSpiel hand-start state against the originating
ScraperFrame. Any field mismatch -> REJECT and return a safe fallback.

Critical translation: OpenSpiel uses the "inflated_big_blind" convention
where all antes are folded into the BB's contribution slot (the SB and
other seats see no ante deduction; the BB sees bb + n_alive*ante in its
contribution). The scraper sees the on-screen reality: every seat's stack
is deducted by its own ante, the BB's "bet in front" is just bb (not
inflated), and antes appear only in the pot. The invariant converts
OpenSpiel-view -> scraper-view before diffing.

Conversion (validated against the sample frame in the prior turn's
arithmetic check):
    n_alive = count of alive seats (== NUM_SEATS at full table)
    For each alive seat i:
        if i == bb_seat:
            scraper_stack[i] = openspiel_stack[i] + (n_alive - 1) * ante
            scraper_bet[i]   = openspiel_contribution[i] - n_alive * ante
        else:
            scraper_stack[i] = openspiel_stack[i] - ante
            scraper_bet[i]   = openspiel_contribution[i]   # 0 for non-blinds, sb for SB
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
    for i in range(NUM_SEATS):
        if i == bb_seat:
            scraper_stack[i] = money[i] + (n_alive - 1) * ante
            scraper_bet[i] = contrib[i] - n_alive * ante
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
    # actual chips in the pot. The chips-in-pot equivalent matching the
    # scraper's pot.total is sum(contribution). Verified empirically at the
    # initial chance node of a 6-max universal_poker game.
    scraper_pot = int(sum(parsed["contribution"]))
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
