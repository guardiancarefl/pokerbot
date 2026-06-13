"""Strict invariant check for Option B integration.

Diffs the reconstructed OpenSpiel hand-start state against the originating
ScraperFrame. Any field mismatch -> REJECT and return a safe fallback.

Real-ante convention: the library (`game_strings.to_inner_game_string_for_state`)
emits a native per-seat `ante=...` array — each alive seat contributes its
own ante; the BB seat contributes `bb + ante`; non-BB alive seats contribute
just `ante`; busted seats contribute 0. The conversion below is a 1:1
identity — no inflation, no ghost-antes correction. Mid-hand bit-exact
chip equality is restored because no chip ints get translated anywhere.
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


def compute_ante_absorbed_mask(
    frame,
    still_in_hand,
    street_idx: int,
    sb_seat: int,
    bb_seat: int,
    sb_amount: int,
    bb_amount: int,
    pre_hand: tuple[int, ...] | None = None,
    ante: int | None = None,
):
    """Per-seat: has the patched OpenSpiel absorbed this seat's ante out
    of `contribution[i]` (and credited it back into `money[i]`)?

    Absorption happens the moment a seat takes its first voluntary
    action (CALL or RAISE) — not from blind posting. Once absorbed,
    persists for the rest of the hand, INCLUDING through a subsequent
    fold ("delayed-fold" seats: limped/called preflop then folded
    later → ante was credited back to money at the moment of their
    voluntary action; the later fold doesn't undo that).

    The standard detection uses `still_in_hand` (= absorbed on postflop;
    blind-threshold check on preflop). With `pre_hand` AND `ante`
    provided, the more accurate per-seat voluntary-commit check kicks
    in: absorbed[i] = True iff seat i committed any chips beyond the
    forced ante + their forced blind. This catches delayed-fold seats
    that the still_in_hand heuristic misses.
    """
    absorbed = [False] * NUM_SEATS
    # When pre_hand + ante are supplied, also catch DELAYED-FOLD seats
    # (folded by the current frame but committed chips before folding
    # — their ante was absorbed at the moment of their voluntary
    # action and stays absorbed through the subsequent fold). The
    # still_in_hand-based detection alone misses these.
    detect_delayed_folds = pre_hand is not None and ante is not None
    if street_idx > 0:
        for i in range(NUM_SEATS):
            if still_in_hand[i]:
                absorbed[i] = True
                continue
            if detect_delayed_folds and frame.alive[i]:
                deduction = int(pre_hand[i]) - int(frame.stack[i])
                forced = int(ante)
                if i == sb_seat:
                    forced += sb_amount
                elif i == bb_seat:
                    forced += bb_amount
                absorbed[i] = (deduction - forced) > 0
        return tuple(absorbed)
    # Preflop: per-seat heuristic against the relevant blind threshold.
    for i in range(NUM_SEATS):
        if not still_in_hand[i]:
            continue
        b = int(frame.bet[i])
        if i == sb_seat:
            absorbed[i] = b > sb_amount
        elif i == bb_seat:
            absorbed[i] = b > bb_amount
        else:
            absorbed[i] = b > 0
    return tuple(absorbed)


def openspiel_to_scraper_view(
    parsed: dict,
    *,
    frame_alive,
    still_in_hand,
    absorbed,
    ante: int,
    preflop_commit_per_alive: int = 0,
    busted_mid_hand_exists: bool = False,
    preflop_max_chip_int: int = 0,
) -> dict:
    """Convert OpenSpiel parsed dict to scraper-equivalent fields under
    the real-ante convention.

    LIBRARY QUIRK — load-bearing context (verified empirically against
    the patched pyspiel). The native-ante implementation does not keep
    antes visible in mid-state observations consistently:

      | seat status                            | contribution[i]                | money[i]                |
      | -------------------------------------- | ------------------------------ | ----------------------- |
      | alive, still in hand, ante NOT absorbed| lifetime commit *includes* ante| start − lifetime (incl ante) |
      | alive, still in hand, ante absorbed    | lifetime commit *minus* ante   | start − lifetime + ante |
      | alive, folded                          | lifetime commit *includes* ante| start − lifetime (incl ante) |
      | busted (alive=False)                   | 0                              | 1 (placeholder)         |

    "Absorbed" = the seat has voluntarily acted at least once this hand
    (CALL or RAISE; folding does not trigger). Once absorbed, persists.
    See compute_ante_absorbed_mask above for the per-seat trigger.

    Chip-conservation at terminal (state.returns()) IS correct — the
    discrepancy is observation-only. The model isn't affected (training
    used the same observations as inference). Only this conversion needs
    to know the quirk so the strict-equality invariant compares apples
    to apples against the scraper UI.

    The scraper's "bet" field (per Ignition's UI semantics) is "chips
    committed THIS STREET, EXCLUDING ante" — antes go straight to pot,
    not through the bet display. The scraper's "stack" reflects all
    deductions (blinds + antes + actions).

    Args:
      parsed: output of parse_state_6max(state).
      frame_alive: length-6 bool, True for seats that posted an ante at
        hand-start. Skips busted seats.
      still_in_hand: length-6 bool, True for seats that posted ante AND
        have not folded.
      absorbed: length-6 bool from compute_ante_absorbed_mask. Drives
        ante-credit-back undo on stack and current-street-commit
        derivation on bet.
      ante: per-seat ante for the current level.
      preflop_commit_per_alive: simple-model uniform-preflop-commit
        value (from state_pack); used to subtract preflop carry-over
        from absorbed seats' contribution on postflop streets.

    Returns:
      {
        "stack": tuple[int, ...],     # per-seat scraper-equivalent stack
        "bet":   tuple[int, ...],     # per-seat scraper-equivalent bet THIS STREET
        "pot":   int,                 # scraper pot total (true chips-in-pot)
        "current_player": int,
        "street_idx": int,
        "private_cards": str,
        "public_cards": str,
      }
    """
    money = list(parsed["money"])
    contrib = list(parsed["contribution"])
    street_idx = int(parsed["street_idx"])
    preflop = (street_idx == 0)

    scraper_stack = [0] * NUM_SEATS
    scraper_bet = [0] * NUM_SEATS

    # Option A gated PER-SEAT (Class A-deeper followup, live dryrun
    # 2026-06-08 busted-mid-hand cases). Empirically (probed against
    # live_20260607 line 163 vs line 156):
    #
    #   - chip_int=N for a RAISE sets spent[seat] = N (replaces the
    #     ante-only init). So spent[raiser] does NOT include ante for a
    #     voluntary-convention raise (chip_int = voluntary-in-front).
    #   - chip_int=1 for CALL sets spent[caller] = maxSpent (the chip_int
    #     of the last raise, or bb_voluntary if no raise yet).
    #
    # When a busted seat raises to chip_int=pre_hand, maxSpent jumps to
    # pre_hand. Subsequent callers reach spent=pre_hand — and that value
    # equals (their ante + their full voluntary), since pre_hand IS the
    # busted's total stack. For these callers, money = pre - pre_hand =
    # the scraper's stack EXACTLY. Subtracting ante (as the standard
    # branch does) over-subtracts.
    #
    # For seats that DIDN'T call the all-in (e.g. they limped at bb
    # earlier and the frame captured before they re-decided), their
    # spent stays at the bb-voluntary level (NOT including ante). The
    # standard formula applies.
    #
    # So the gate is PER-SEAT: a seat's spent[i] >= preflop_max_chip_int
    # (= the busted seat's pre_hand, = max chip_int of any raise in
    # preflop) means they matched the all-in convention. Use the
    # no-ante-subtract formula for them. For other seats, standard
    # formula. Strictly additive: when busted_mid_hand_exists is False,
    # preflop_max_chip_int stays 0 and no seat satisfies spent>=0+1,
    # so all seats take the standard branch (= live_1500 unchanged).
    # Track which seats matched the busted's all-in chip_int (= their
    # OpenSpiel spent reached preflop_max_chip_int). For those seats use
    # the no-ante-subtract formula. preflop_max_chip_int defaults to 0
    # when no busted-mid-hand exists, so the matched_all_in check is
    # never true → standard branch for every seat.
    matched_all_in = [False] * NUM_SEATS
    if busted_mid_hand_exists and preflop_max_chip_int > 0:
        for i in range(NUM_SEATS):
            # contrib[i] is OpenSpiel's spent for the seat (= chip_int
            # they reached). If >= preflop_max_chip_int, they're either
            # the busted raiser themselves or a caller who matched.
            if int(contrib[i]) >= preflop_max_chip_int:
                matched_all_in[i] = True
    # All-in-for-less detection (seq=246, audit 2026-06-09): a seat that
    # committed every chip they had via a CALL action (chip_int=1, capped
    # at remaining stack) has money[i] == 0. For these seats the OpenSpiel
    # patch does NOT credit the ante back into money (verified empirically:
    # call all-in keeps lifetime=pre AND money=0, whereas raise all-in
    # keeps lifetime=pre AND money=ante via the absorption credit).
    # Their contrib therefore includes ante and the no-ante-subtract
    # matched_all_in branch produces the correct scraper_bet. This trigger
    # only fires on call-for-less because raise-all-in seats always have
    # money[i] >= ante (audit: seq=97/106/170/427/428 all have money>=ante).
    for i in range(NUM_SEATS):
        if frame_alive[i] and int(money[i]) == 0:
            matched_all_in[i] = True

    for i in range(NUM_SEATS):
        if not frame_alive[i]:
            # Busted / empty seat: scraper shows 0 chips, 0 bet.
            scraper_stack[i] = 0
            scraper_bet[i] = 0
            continue
        if absorbed[i]:
            if matched_all_in[i]:
                # chip_int=pre_hand convention: spent already includes ante.
                # money = pre - spent IS the scraper-side stack (no further
                # ante subtraction).
                scraper_stack[i] = max(0, int(money[i]))
                if preflop:
                    # spent (= contrib) = ante + voluntary in front. Subtract
                    # ante to recover voluntary chips visible to scraper.
                    scraper_bet[i] = max(0, int(contrib[i]) - ante)
                else:
                    # POSTFLOP all-in via chip_int=pre_hand convention. spent
                    # = ante + preflop_carry + current-street_voluntary
                    # (the convention OVERWRITES spent with pre_hand, so the
                    # ante that was previously credited-back via absorption
                    # is re-introduced). Recover current-street voluntary
                    # by subtracting the preflop carry AND the ante. NOT
                    # preflop_max_chip_int — that's the BUSTED seat's
                    # pre_hand value, which for a postflop all-in equals
                    # (ante + preflop_carry + current-street voluntary)
                    # and would over-subtract the current-street voluntary
                    # to 0 (live_dryrun 2026-06-08 seq=192: BB all-in for
                    # 1015 on the turn, prior code reported bet[BB]=0).
                    scraper_bet[i] = max(
                        0, int(contrib[i]) - preflop_commit_per_alive
                            - ante)
                continue
            # Standard absorbed seat (chip_int=voluntary convention).
            # Ante absorbed: money has ante credited back; contrib has
            # ante subtracted. Undo both for scraper view.
            scraper_stack[i] = max(0, int(money[i]) - ante)
            if preflop:
                # contrib is already the current-street voluntary commit
                # in scraper units (ante removed). Use as-is.
                scraper_bet[i] = max(0, int(contrib[i]))
            else:
                # Postflop: contrib = preflop carry + current-street.
                # Subtract simple-model preflop commit to isolate current.
                scraper_bet[i] = max(
                    0, int(contrib[i]) - preflop_commit_per_alive)
            continue
        if still_in_hand[i]:
            # Still in hand, not absorbed: only possible on preflop —
            # seat is pre-action (only blind/ante posted, no voluntary
            # commit yet). ante stays in contrib; money has ante deducted.
            scraper_stack[i] = max(0, int(money[i]))
            scraper_bet[i] = max(0, int(contrib[i]) - ante)
            continue
        # Folded: OpenSpiel keeps ante in lifetime contribution; money
        # already reflects the ante deduction.
        scraper_stack[i] = max(0, int(money[i]))
        scraper_bet[i] = 0  # no current-street commit; folded

    # Pot calc: for seats that matched the all-in, contrib already
    # includes ante (no need to add back). For seats that didn't match,
    # the standard rule applies (add n_absorbed_unmatched * ante).
    n_absorbed_unmatched = sum(
        1 for i in range(NUM_SEATS)
        if absorbed[i] and not matched_all_in[i]
    )
    scraper_pot = int(sum(contrib)) + n_absorbed_unmatched * ante

    return {
        "stack": tuple(scraper_stack),
        "bet": tuple(scraper_bet),
        "pot": scraper_pot,
        "current_player": int(parsed["current_player"]),
        "street_idx": street_idx,
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
    ante = state_pack.blind_level.ante
    sb_seat = state_pack.sb_seat
    bb_seat = state_pack.bb_seat
    sb_amount = state_pack.blind_level.small_blind
    bb_amount = state_pack.blind_level.big_blind

    # Parse the chance-node state from hero's observer perspective so we
    # don't trip current_player==-1 in observation_string.
    parsed = parse_state_6max(state, observer=frame.hero_seat)
    parsed["dealer_seat"] = frame.dealer_seat

    # At hand-start, every alive seat is still in hand (no folds yet) and
    # no voluntary actions have happened (blinds/antes are forced posts,
    # not voluntary). Absorbed mask is therefore all-False.
    still_in_hand = tuple(frame.alive)
    absorbed = compute_ante_absorbed_mask(
        frame, still_in_hand, street_idx=0,
        sb_seat=sb_seat, bb_seat=bb_seat,
        sb_amount=sb_amount, bb_amount=bb_amount,
    )
    reconstructed = openspiel_to_scraper_view(
        parsed,
        frame_alive=frame.alive,
        still_in_hand=still_in_hand,
        absorbed=absorbed,
        ante=ante,
        preflop_commit_per_alive=0,
    )

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
    """Strict invariant check for mid-hand hero-to-act state.

    Real-ante convention: chip ints flow scraper → OpenSpiel → scraper with
    no translation. Bit-exact per-seat stack + bet + pot equality is the
    correctness contract; any deviation is a reconstruction failure and the
    frame is rejected (safe-fold or safe-check, never a chip-committing
    guess).

    LOAD-BEARING CHECKS (any mismatch -> ok=False, safe_action set):
      1. current_player == hero_seat — replayed to the right player to act
      2. street_idx (from state) == scraper-derived (from board len)
      3. per-seat stack[i] EXACT equality vs reconstructed stack
      4. per-seat bet[i]   EXACT equality vs reconstructed bet
      5. pot_total         EXACT equality vs reconstructed pot
      6. private_cards == hero's hole cards (canonical sorted)
      7. public_cards == scraper's board (canonical sorted)
      8. legal_actions consistent with hero_facing_bet:
           if hero_facing_bet -> FOLD MUST be in state.legal_actions()

    PLUS SCRAPER SELF-CONSISTENCY (chip-conservation against the frame's
    own fields, independent of the OpenSpiel reconstruction):
      9. scraper.pot_total ≈ sum_of_chips_implied_by_per-seat_arithmetic
         (catches scraper OCR errors. Uses the REPAIRED folded array —
         workaround for Ignition's stale folded field, see
         _repair_folded_from_chip_deductions. Tolerance = max(1, n_anf-1)
         to absorb simple-model integer-divide remainder.)
    """
    # MidHandState (Piece 4) carries everything we need beyond what
    # HandStartState provides.
    state = state_pack.state
    ante = state_pack.blind_level.ante
    street_idx = state_pack.street_idx
    preflop_commit_per_alive = state_pack.preflop_commit_per_alive

    parsed = parse_state_6max(state, observer=frame.hero_seat)
    parsed["dealer_seat"] = frame.dealer_seat

    # Compute the still-in-hand + absorbed masks up-front so
    # openspiel_to_scraper_view can apply the per-seat ante-bookkeeping
    # correctly. Uses the repaired folded array (workaround for
    # Ignition's stale `folded` field — orthogonal to the inflated-BB
    # removal). The mid-hand self-consistency block below reuses these.
    from src.nlhe.integration.scraper_schema import (
        _repair_folded_from_chip_deductions,
    )
    sb_seat = state_pack.sb_seat
    sb = state_pack.blind_level.small_blind
    bb = state_pack.blind_level.big_blind
    bb_seat = state_pack.bb_seat
    # Use the same pre_hand_override as the replay (state_pack carries
    # the per-seat pre-hand stacks) so multi-hand stack accumulation
    # doesn't desync the invariant's folded mask from the replay's.
    folded = _repair_folded_from_chip_deductions(
        frame, sb_seat=sb_seat, bb_seat=bb_seat,
        pre_hand_override=state_pack.pre_hand_stacks)
    still_in_hand = tuple(
        frame.alive[i] and not folded[i] for i in range(NUM_SEATS))
    absorbed = compute_ante_absorbed_mask(
        frame, still_in_hand, street_idx=street_idx,
        sb_seat=sb_seat, bb_seat=bb_seat,
        sb_amount=sb, bb_amount=bb,
        pre_hand=state_pack.pre_hand_stacks,
        ante=ante,
    )

    # Class A-deeper / Option A gated: when ANY seat is all-in via the
    # chip_int=pre_hand convention (replay_to_decision's Class B
    # fallback OR derive_action_sequence's busted_mid_hand emission),
    # their OpenSpiel spent equals pre_hand (= ante + voluntary).
    # Subsequent callers reach the same spent. For these seats the
    # standard view formula `money - ante` over-subtracts the ante (it
    # was already included in spent).
    #
    # All-in seats are:
    #   - busted-mid-hand: pre_hand > 0 AND frame.alive=False (scraper
    #     zeroes stack + bet)
    #   - voluntarily all-in (alive=True, stack=0, bet>0): they shoved
    #     and the Class B fallback emitted chip_int=pre_hand
    #
    # Strictly additive: when no all-in seat exists, preflop_max_chip_int
    # stays 0; no seat's contrib satisfies the matched_all_in check;
    # standard branch for every seat (= live_1500 100% preserved).
    all_in_seats_chip_ints = []
    for i in range(NUM_SEATS):
        pre_i = int(state_pack.pre_hand_stacks[i])
        if pre_i <= 0:
            continue
        if not frame.alive[i]:
            # Busted-mid-hand
            all_in_seats_chip_ints.append(pre_i)
        elif int(frame.stack[i]) == 0 and int(frame.bet[i]) > 0:
            # Voluntarily all-in (Class B fallback territory)
            all_in_seats_chip_ints.append(pre_i)
    busted_mid_hand_exists = len(all_in_seats_chip_ints) > 0
    preflop_max_chip_int = (
        max(all_in_seats_chip_ints) if all_in_seats_chip_ints else 0
    )

    # Real-ante view with ante-aware mapping (per-seat absorption mask
    # + simple-model preflop carry).
    recon_handlevel = openspiel_to_scraper_view(
        parsed,
        frame_alive=frame.alive,
        still_in_hand=still_in_hand,
        absorbed=absorbed,
        ante=ante,
        preflop_commit_per_alive=preflop_commit_per_alive,
        busted_mid_hand_exists=busted_mid_hand_exists,
        preflop_max_chip_int=preflop_max_chip_int,
    )

    deltas = []

    # 1. current_player
    if parsed["current_player"] != frame.hero_seat:
        deltas.append(("current_player",
                        frame.hero_seat, parsed["current_player"]))

    # 2. street_idx (from state) should match scraper-derived (from board len)
    if parsed["street_idx"] != street_idx:
        deltas.append(("street_idx", street_idx, parsed["street_idx"]))

    # 3 + 4 + 5. EXACT per-seat stack / bet / pot equality (re-tightened
    # post-bridge: the inflated-BB shim is gone, so chips round-trip 1:1).
    if frame.pot_total != recon_handlevel["pot"]:
        deltas.append(("pot", frame.pot_total, recon_handlevel["pot"]))
    for i in range(NUM_SEATS):
        s_scraper = frame.stack[i]
        s_recon = recon_handlevel["stack"][i]
        if s_scraper != s_recon:
            deltas.append((f"stack[seat{i+1}]", s_scraper, s_recon))
        b_scraper = frame.bet[i]
        b_recon = recon_handlevel["bet"][i]
        if b_scraper != b_recon:
            deltas.append((f"bet[seat{i+1}]", b_scraper, b_recon))

    # 9. SCRAPER SELF-CONSISTENCY (chip conservation against frame's own
    # fields). The scraper pot_total should equal: SB + BB + n_alive*ante
    # + sum_of_per-seat_voluntary_commits_implied_by_pre_hand_arithmetic.
    # In the simple model: voluntary commits per alive non-folded seat =
    # preflop_commit_per_alive + frame.bet[i]; per folded seat = blind-only
    # (sb if sb_seat, bb if bb_seat, 0 else).
    #
    # DELAYED-FOLD ENHANCEMENT: when state_pack carries an override
    # pre_hand (session-tracked), folded seats that committed chips
    # voluntarily before folding (= limped/called preflop, then folded
    # later) contribute MORE than just their blind. Per-seat voluntary
    # = (pre_hand - stack - ante - blind_if_blind) — anything > 0 is
    # the delayed-fold's pre-fold commitment that needs to be in the
    # pot sum.
    pre_hand = state_pack.pre_hand_stacks
    if busted_mid_hand_exists:
        # Class A-deeper gated branch: with busted-mid-hand seats, the
        # simple-model uniform-commit assumption breaks (busted seat's
        # voluntary differs from non-busted's). Use the direct
        # chip-conservation identity instead: every seat alive at
        # hand-start contributed (pre_hand[i] - stack[i]) chips to pot
        # (where stack[i] = 0 for busted seats). This is the strongest
        # check possible — no tolerance, exact equality required.
        implied_pot = sum(
            int(pre_hand[i]) - int(frame.stack[i])
            for i in range(NUM_SEATS)
            if int(pre_hand[i]) > 0
        )
        if implied_pot != int(frame.pot_total):
            deltas.append(("scraper_self_consistency:pot",
                            frame.pot_total, implied_pot))
    else:
        n_alive_total = sum(frame.alive)
        implied_pot = n_alive_total * ante  # antes
        for i in range(NUM_SEATS):
            if not frame.alive[i]:
                continue
            if folded[i]:
                # Folded seat — start with blind only.
                if i == sb_seat:
                    implied_pot += sb
                elif i == bb_seat:
                    implied_pot += bb
                # Plus any DELAYED-FOLD voluntary commit (chips committed
                # via call/raise BEFORE the fold; pre-hand-derived).
                deduction = int(pre_hand[i]) - int(frame.stack[i])
                forced = ante + (sb if i == sb_seat else
                                  bb if i == bb_seat else 0)
                voluntary = deduction - forced
                if voluntary > 0:
                    implied_pot += voluntary
            else:
                # Alive non-folded: preflop_commit + current-street bet
                implied_pot += preflop_commit_per_alive + frame.bet[i]
        # Tolerance: the simple-model integer-divides `residual // n_anf`,
        # which loses up to (n_anf - 1) chips of remainder.
        n_anf = sum(1 for i in range(NUM_SEATS)
                    if frame.alive[i] and not folded[i])
        self_consistency_tolerance = max(1, n_anf - 1)
        if abs(implied_pot - frame.pot_total) > self_consistency_tolerance:
            deltas.append(("scraper_self_consistency:pot",
                            frame.pot_total, implied_pot))

    # 6. private_cards: hero's hole cards (canonical sorted)
    scraper_hole_canon = _canonical_card_string(frame.hero_cards)
    state_hole_canon = _canonical_card_string(recon_handlevel["private_cards"])
    if scraper_hole_canon != state_hole_canon:
        deltas.append(("hero_cards",
                        scraper_hole_canon, state_hole_canon))

    # 7. public_cards: board (canonical sorted)
    scraper_board_canon = _canonical_card_string(frame.board)
    state_board_canon = _canonical_card_string(recon_handlevel["public_cards"])
    if scraper_board_canon != state_board_canon:
        deltas.append(("board",
                        scraper_board_canon, state_board_canon))

    # 8. legal_actions consistency with hero_facing_bet
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
