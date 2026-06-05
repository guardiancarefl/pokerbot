"""Real-chip ↔ OpenSpiel-chip translation (Phase 2 bridge layer).

The Option-B integration matches the library's inflated-BB convention on
the scraper-view side. But on the ACTION-replay side, OpenSpiel's min-raise
constraint (2 × inflated_bb) systematically rejects real-poker raise sizes
that fall below it (e.g., a real 2x-BB raise to 50 is illegal in OpenSpiel
when inflated_bb=55 → min-raise=110 at level 1).

This module bridges the two action spaces. Forward translation maps a real
chip_int from the scraper's action sequence to an OpenSpiel-legal chip_int
(bumping raises up to the OpenSpiel minimum when needed). Reverse maps the
resolver's chosen chip_int back to a real-table chip amount the client
enters.

Trade-off (documented in docs/DECISIONS.md → "Phase 2 bridge"):
  - fold (0) and call/check (1) translate exactly: same semantic in both
    spaces.
  - raise (chip_int >= 2): real → OpenSpiel uses max(real, openspiel_min);
    raises below openspiel_min collapse to the min (a known resolution
    loss the trained model also had during training).
  - reverse for raises: clamp to the real-table legal range and pass the
    OpenSpiel chip_int through. The model thinks in inflated chip space;
    the client acts in real chips. Bet-sizing DRIFT of up to ~2x at
    level 1 (less at higher levels) is the documented cost. Decisions
    qualitatively correct (inflation is ratio-preserving for pot
    odds / equity comparisons); raise SIZING shifted upward.

This is a measurement step: log-only deployment with the bridge will tell
us whether the drift is tolerable for double-up format. Library rewrite +
retrain is queued as the clean fallback if drift proves problematic.
"""
from __future__ import annotations


def real_to_openspiel_action(real_chip_int: int, legal_actions: list[int]
                              ) -> int:
    """Map a real-poker chip_int (derived from scraper) to an OpenSpiel-
    legal chip_int.

    Args:
        real_chip_int: chip_int the scraper-derived action sequence wants
            to apply. 0 = fold, 1 = call/check, N >= 2 = raise-to-N.
        legal_actions: state.legal_actions() at the current decision node.

    Returns:
        An OpenSpiel-legal chip_int. For fold/call/check, passes through.
        For raises below OpenSpiel's min-raise, bumps up to the min-raise.
        Raises above min are passed through unchanged.

    Raises:
        ValueError if neither fold nor any raise is in legal_actions (would
        indicate a degenerate state — caller should treat as ReplayError).
    """
    if real_chip_int == 0:
        if 0 in legal_actions:
            return 0
        # Defensive: fold not legal (rare — degenerate state). Fall back
        # to call/check.
        if 1 in legal_actions:
            return 1
        raise ValueError(
            f"fold (0) requested but not legal_actions={legal_actions[:10]}; "
            f"no call (1) available either")
    if real_chip_int == 1:
        if 1 in legal_actions:
            return 1
        # Defensive: call not legal — shouldn't happen at a decision node.
        raise ValueError(
            f"call (1) requested but not legal_actions={legal_actions[:10]}")
    # Raise: chip_int >= 2.
    # Find the OpenSpiel min-raise (smallest legal raise int >= 2).
    raise_options = [a for a in legal_actions if a >= 2]
    if not raise_options:
        # No raise legal here. Caller may want to fall back to call.
        # Raise an error so the caller decides explicitly.
        raise ValueError(
            f"raise to {real_chip_int} requested but no raise actions legal "
            f"(legal_actions={legal_actions[:10]})")
    openspiel_min_raise = min(raise_options)
    if real_chip_int >= openspiel_min_raise:
        # Real raise is already legal (above OpenSpiel's min). Pass through.
        # Note: also need to be <= max raise (hero's stack). OpenSpiel's
        # legal_actions caps at the hero's all-in amount. Clamp.
        openspiel_max_raise = max(raise_options)
        return min(real_chip_int, openspiel_max_raise)
    # Real raise is below OpenSpiel's min-raise. Bump up to the min.
    # This is the bet-sizing drift — model will see "min-raise" instead of
    # the real raise size.
    return openspiel_min_raise


def openspiel_to_real_action(openspiel_chip_int: int,
                              scraper_min_raise: int,
                              scraper_max_raise: int,
                              scraper_facing_bet: bool) -> dict:
    """Map the resolver's chosen OpenSpiel chip_int to a real-table action
    the client enters.

    Args:
        openspiel_chip_int: chip_int the resolver returned.
        scraper_min_raise: minimum legal raise amount in the real game
            (= 2 × scraper.BB at preflop, or 2 × current-street-max-bet
            postflop).
        scraper_max_raise: hero's real-stack cap (= scraper.stack[hero] +
            scraper.bet[hero]; can't raise more than they have).
        scraper_facing_bet: True if hero faces a bet (= FOLD is legal in
            real terms). Determines whether chip_int=1 means call or check.

    Returns:
        dict with:
          kind:         "fold" | "call" | "check" | "raise_to"
          chip_amount:  int (the chip amount to enter into the bet box;
                        None for fold/call/check)
          raw_openspiel_chip_int: int (the resolver's original output, for
                        logging/audit)
    """
    if openspiel_chip_int == 0:
        return {
            "kind": "fold",
            "chip_amount": None,
            "raw_openspiel_chip_int": 0,
        }
    if openspiel_chip_int == 1:
        return {
            "kind": "call" if scraper_facing_bet else "check",
            "chip_amount": None,
            "raw_openspiel_chip_int": 1,
        }
    # Raise: clamp to real-table legal range. The model's chip_int lives
    # in inflated space; we use it directly as the real chip amount,
    # subject to the real-game legal bounds. This is the bet-sizing drift:
    # at level 1, the model's "min-raise" = 110 chips in inflated terms,
    # which we'd enter as 110 real chips (= 4.4x real BB instead of 2x
    # min-raise). The drift narrows at higher levels (smaller relative
    # ante share).
    raise_amount = max(scraper_min_raise,
                       min(openspiel_chip_int, scraper_max_raise))
    return {
        "kind": "raise_to",
        "chip_amount": int(raise_amount),
        "raw_openspiel_chip_int": int(openspiel_chip_int),
    }
