"""Single-record decision pipeline for the staged auto-clicker.

Stage 1 (DRY-RUN): the entry-point script feeds scraper records into
`make_decision()` one at a time. Each call returns a `LiveDecision`
describing what the bot READ, what it would DO (under sample-mode
policy), and the click PLAN it would execute (Stage 1 displays only;
never clicks).

Pure consumer logic. Owns:
  - SessionTracker integration (per-hand pre-hand stack tracking +
    UI-lag pot correction)
  - Override path with simple-model fallback (same shape as
    scripts/test_integration_midhand.py)
  - Strict invariant check + safe-fold short-circuit
  - Sample-mode policy query via the existing solver helper
  - OpenSpiel chip_int → real-table client_action translation
  - click_target.compute_click_target

The function NEVER raises: any internal error is caught and surfaced
as a `safe_fold` LiveDecision with a `skip_reason` so the loop keeps
running on the next frame.
"""
from __future__ import annotations

import dataclasses
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LiveDecision:
    """Result of processing one scraper record through the live loop."""
    # Source / ordering
    seq: int | None
    captured_at: str
    processed_at_ts: float = field(default_factory=time.time)

    # Frame summary (for the READ panel)
    level: int | None = None
    blinds: tuple[int, int, int] | None = None  # (sb, bb, ante)
    hero_seat: int | None = None
    dealer_seat: int | None = None
    hero_cards: tuple[str, ...] = ()
    board: tuple[str, ...] = ()
    pot_total: int | None = None
    pot_corrected: bool = False  # True if SessionTracker fixed a UI-lag pot
    hero_stack: int | None = None
    n_alive: int | None = None
    street_idx: int | None = None
    facing_bet: bool | None = None

    # Status: one of
    #   "decision"               — full pipeline succeeded, bot freshly sampled
    #   "decision_cached"        — same decision identity already sampled this
    #                              hand/street/bet-state; returning the locked-
    #                              in action to prevent the per-frame re-sample
    #                              lottery (CFR mixed strategies → sample once)
    #   "decision_recovered"     — decision on a repaired frame, re-validated
    #                              through the full replay+invariant gate:
    #                              either a scraper-suspect frame whose single
    #                              flagged stack was derived from the clean
    #                              hand-start anchor via chip conservation
    #                              (Layer 1, 2026-06-09 blackout postmortem)
    #                              or a flag-gated P2 bet-closure recovery of
    #                              an invariant-failed displacement-signature
    #                              frame (2026-06-11 postmortem; field labels
    #                              "(bet_closure...)" in recovered_fields)
    #   "decision_recovered_cached" — ditto, action came from DecisionCache
    #   "safe_fold"              — invariant fail / replay error → no plan
    #   "skip_not_hero_to_act"   — frame parsed, hero not to act
    #   "skip_data_quality"      — parse_frame raised or hero-cards missing
    status: str = "..."
    skip_reason: str | None = None
    pre_hand_override_used: bool = False
    # On status decision_recovered*: audit trail of derived fields, e.g.
    # ["stack.seat1=1110 (strict)"]. None on every other path.
    recovered_fields: list[str] | None = None
    # True iff this frame was a new hand-start whose anchor the
    # SessionTracker REFUSED under the chips-in-play ceiling guard
    # (poisoned hand-start, e.g. stuck-digit 9907 -> sum 17917). The
    # frame's own decision output is unaffected (hand-start frames are
    # not hero-to-act); this flag is the audit trail for why later
    # frames of the hand have no anchor.
    anchor_refused: bool = False

    # On status == "decision" or "decision_cached"
    client_action: dict | None = None
    client_action_bb_mult: float | None = None
    client_action_pot_frac: float | None = None
    client_action_stack_frac: float | None = None
    resolver_raw_openspiel_chip_int: int | None = None
    # Tuple identifying this decision (dealer, blinds, alive, hero_cards,
    # board, hero_seat, max_opp_bet). Same identity across frames = same
    # decision; cached action is reused. Populated on decision/decision_cached.
    decision_identity: tuple | None = None

    # On status == "safe_fold" (invariant_fail path)
    invariant_deltas: list | None = None

    # Always populated (no_op on skip / safe_fold)
    click_plan: Any | None = None  # ClickPlan (deferred import)


class DecisionCache:
    """Sample-once-per-decision cache.

    A pure-CFR mixed-strategy policy returns DIFFERENT samples on each call
    for the same infoset (that's what mixed means). The scraper sends many
    frames per real decision; sampling on every frame produces shove-vs-call
    drift across polling jitter (real symptom from live dryrun 2026-06-08:
    seq=222→2182-chip shove, seq=223→6×BB raise — same hand, same spot).

    DecisionCache locks in one sample per "actual decision" so subsequent
    frames of the SAME decision return the SAME action. Re-decides only
    when the situation genuinely changes (new hand / new street / new bet
    to face).

    Identity tuple — these together define ONE actual decision:
      - dealer_seat           — changes when the button moves (new hand)
      - blinds (sb, bb, ante) — changes at level transitions
      - alive[]               — changes when a seat busts
      - hero_cards            — changes when a new hand is dealt
      - board                 — changes at street transitions
      - hero_seat             — sanity: should be constant per session
      - max_opp_bet           — max bet by any non-hero seat this street;
                                a re-raise that puts hero back in the
                                decision raises this and forces re-sample.
                                Crucially: a CALL by an opponent (pot grows
                                but max bet unchanged) is NOT a new decision
                                until action returns to hero, at which point
                                either max_opp_bet has changed (raise) or
                                the action lap is structurally different.

    NOT included on purpose:
      - pot_total: UI-lag and intra-street collections cause pot to drift
        on the same actual decision. SessionTracker corrects pot drift;
        max_opp_bet is the right signal for "new bet to face".
      - facing_bet: derived from max_opp_bet > hero_bet — redundant.
      - hero_stack: changes due to UI lag too; max_opp_bet captures the
        real action.
    """

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._action: dict | None = None
        self._chip_int: int | None = None

    def identity(self, frame) -> tuple:
        """Compute the decision-identity tuple for a parsed ScraperFrame."""
        max_opp_bet = 0
        for i, b in enumerate(frame.bet):
            if i == frame.hero_seat:
                continue
            if b is None:
                continue
            if int(b) > max_opp_bet:
                max_opp_bet = int(b)
        return (
            int(frame.dealer_seat),
            (int(frame.blinds.sb), int(frame.blinds.bb), int(frame.blinds.ante)),
            tuple(bool(a) for a in frame.alive),
            tuple(frame.hero_cards),
            tuple(frame.board),
            int(frame.hero_seat),
            int(max_opp_bet),
        )

    def get(self, key: tuple) -> tuple[dict, int] | None:
        """Return (cached_action_dict, cached_chip_int) if the key matches
        the locked-in identity; None otherwise."""
        if key == self._key and self._action is not None:
            return self._action, int(self._chip_int)
        return None

    def put(self, key: tuple, action: dict, chip_int: int) -> None:
        """Lock in a fresh sample for this identity. Overwrites any prior
        identity — the cache holds only the most recent decision (one
        decision active at a time per session)."""
        self._key = key
        self._action = action
        self._chip_int = int(chip_int)


# --------------------------------------------------------------------------
# AA/KK preflop FOLD floor (deployment-only policy patch)
# --------------------------------------------------------------------------
#
# The validated blueprint plays a CFR-converged mixed strategy on every
# infoset, including a small (~5%) FOLD tail on AA and (~8%) on KK at deep
# stacks (verified in runs/k200_real_ante_20260605_225847/convergence_log.csv
# at iter_1500, and matched by the live seq=227 fold on 2026-06-08).
# Folding AA or KK preflop is CATEGORICALLY never +EV — no opponent range
# makes folding the highest-EV action — so the tail is pure error and safe
# to mask without disturbing the rest of the validated distribution.
#
# Strict scope: street_idx == 0 AND both hero_cards' ranks are 'A' or both
# are 'K'. Rank check is on the literal hero_cards string ('As', 'Ah', ...)
# — NOT on the abstraction bucket id, which conflates AA with KK and (at
# other streets / boards) may include non-premium hands.
#
# This filter is wired into make_decision only. Training (cfr6) and eval
# (eval_pool, eval_6max_self_play, ablations) do not use it; they continue
# to see the unmodified policy.

def _ranks_are_both(rank: str, hero_cards: tuple) -> bool:
    if len(hero_cards) != 2:
        return False
    c0, c1 = hero_cards[0], hero_cards[1]
    if not (isinstance(c0, str) and isinstance(c1, str)):
        return False
    if len(c0) < 1 or len(c1) < 1:
        return False
    return c0[0] == rank and c1[0] == rank


def apply_aa_kk_preflop_floor(policy, legal_mask, parsed, state):
    """Mask FOLD on the policy distribution if and only if hero holds AA
    or KK preflop. Renormalize over the remaining legal mass.

    Args:
      policy:     length-N np.array, sums to 1 on the legal mask.
      legal_mask: length-N np.array, 1.0 for legal slots, 0.0 otherwise.
      parsed:     parse_state_6max dict; uses 'private_cards' (string like
                  'AsAh') and 'street_idx' if present, else 'public_cards'
                  length to infer street.
      state:      OpenSpiel state (unused; kept for hook signature parity).

    Returns:
      A new np.array (or the same `policy` reference if the gate doesn't
      fire). Caller MUST treat the return value as the post-filter policy.
    """
    import numpy as np
    from src.nlhe.actions import DiscreteAction

    # Gate 1: hero is preflop. parse_state_6max stores street as the
    # length of `public_cards` (0 chars = preflop, 6 = flop, 8 = turn,
    # 10 = river). Use that — robust whether or not the optional
    # 'street_idx' key exists in the parsed dict.
    pub = parsed.get("public_cards", "") or ""
    is_preflop = (len(pub) == 0)
    if not is_preflop:
        return policy

    # Gate 2: hero holds AA or KK. private_cards is the concatenated 2-card
    # string ('AsAh'), so split into 2-char chunks for the rank check.
    priv = parsed.get("private_cards", "") or ""
    if len(priv) != 4:
        return policy
    cards = (priv[0:2], priv[2:4])
    is_aa = _ranks_are_both("A", cards)
    is_kk = _ranks_are_both("K", cards)
    if not (is_aa or is_kk):
        return policy

    # Gate 3: FOLD must be legal at all (always true at a real decision
    # node, but defensive against zero-legal-FOLD edge cases).
    fold_idx = int(DiscreteAction.FOLD)
    if legal_mask[fold_idx] == 0.0:
        return policy
    if float(policy[fold_idx]) == 0.0:
        return policy

    # Mask + renormalize. The remaining mass across all other legal
    # actions is (1 - policy[FOLD]); each kept action keeps its
    # relative share, so we divide by that remainder.
    new_policy = policy.copy()
    fold_mass = float(new_policy[fold_idx])
    new_policy[fold_idx] = 0.0
    remainder = 1.0 - fold_mass
    if remainder <= 0.0:
        # Degenerate (policy was P(FOLD)=1 with no other legal mass —
        # impossible at a real decision node, but be safe). Fall back
        # to uniform on legal non-FOLD.
        keep_mask = legal_mask.copy()
        keep_mask[fold_idx] = 0.0
        n = float(keep_mask.sum())
        if n <= 0.0:
            return policy
        return (keep_mask / n).astype(policy.dtype)
    new_policy = new_policy / remainder
    return new_policy.astype(policy.dtype)


# --------------------------------------------------------------------------
# Check-when-free FOLD floor (deployment-only policy patch)
# --------------------------------------------------------------------------
#
# Folding when CHECK is available (to_call == 0 with CALL legal) is
# CATEGORICALLY never +EV — you give up free continued equity for zero
# gain. Universal_poker keeps FOLD in legal_actions() even at check
# spots (so the trained model can assign mass there; the depth-invariance
# probe found ~55% FOLD on 92o BB-check at seq=48). Strictly dominated,
# safe to mask everywhere.
#
# Scope: any street, any depth, any hand — fires whenever (to_call == 0
# AND CALL legal). Strictly additive: when CHECK isn't free, returns
# the input policy reference unchanged.

def apply_check_when_free_floor(policy, legal_mask, parsed, state):
    """Mask FOLD when CHECK is available. Strictly dominated action."""
    import numpy as np
    from src.nlhe.actions import DiscreteAction

    cp = parsed["current_player"]
    contribs = parsed.get("contribution", [])
    if not contribs:
        return policy
    to_call = int(max(contribs)) - int(contribs[cp])
    if to_call > 0:
        return policy

    fold_idx = int(DiscreteAction.FOLD)
    call_idx = int(DiscreteAction.CALL)
    if legal_mask[call_idx] == 0.0:
        return policy   # no CHECK option
    if legal_mask[fold_idx] == 0.0 or float(policy[fold_idx]) == 0.0:
        return policy   # nothing to mask

    new_policy = policy.copy()
    fold_mass = float(new_policy[fold_idx])
    new_policy[fold_idx] = 0.0
    remainder = 1.0 - fold_mass
    if remainder <= 0.0:
        keep = legal_mask.copy()
        keep[fold_idx] = 0.0
        n = float(keep.sum())
        if n <= 0.0:
            return policy
        return (keep / n).astype(policy.dtype)
    return (new_policy / remainder).astype(policy.dtype)


# --------------------------------------------------------------------------
# Short-stack floor (deployment-only policy patch)
# --------------------------------------------------------------------------
#
# At ≤ threshold_bb effective stack, the depth-confused model still emits
# meaningful mass on intermediate bet sizes (min-raise, half-pot, pot)
# that are strategically incoherent at push/fold depth (depth-invariance
# probe TV ratio 1.5-3.2 favoring chip-magnitude over BB-depth; seq=461
# was a 5.3BB SB facing action picking BET_100 as a min-raise).
#
# Floored decision set: {FOLD, CALL, ALLIN}. Bet/raise sizes (BET_33,
# BET_50, BET_66, BET_100, BET_150, BET_200) are masked out and their
# mass is redistributed proportionally onto the kept set.
#
# Off-line A/B (hpl=5, live-matched escalation, 24000 paired games):
# paired ICM delta V1−V0 = +0.0100 ± 0.00255 (z=3.92). Per-firing
# (diverged games only) +0.256 ± 0.065 (z=3.95). Floor fires on 7.53%
# of decisions in the live-matched arm.

_DEFAULT_SHORT_STACK_FLOOR_BB = 6.0


def _hero_eff_bb_from_parsed(parsed) -> tuple[float, int]:
    """Return (eff-stack-in-BB, BB-amount). 0.0 / 0 if undetermined."""
    bb = int(parsed.get("big_blind", 0))
    if bb <= 0:
        return 0.0, 0
    cp = parsed["current_player"]
    money = parsed.get("money", [])
    if cp >= len(money):
        return 0.0, bb
    my_stack = int(money[cp])
    opps = [int(money[i]) for i in range(len(money))
            if i != cp and money[i] > 0]
    eff = min(my_stack, max(opps)) if opps else my_stack
    return float(eff) / bb, bb


def apply_short_stack_floor(policy, legal_mask, parsed, state,
                              threshold_bb: float = _DEFAULT_SHORT_STACK_FLOOR_BB):
    """At hero eff-stack ≤ threshold_bb:
       - facing action (to_call > 0)        → keep {FOLD, CALL, ALLIN}
       - to_call==0 with CHECK legal        → keep {CALL, ALLIN}
       - to_call==0 with CHECK illegal      → keep {FOLD, ALLIN}
    Renormalize over kept legal mass. Strict no-op above threshold."""
    import numpy as np
    from src.nlhe.actions import DiscreteAction

    eff_bb, bb = _hero_eff_bb_from_parsed(parsed)
    if bb <= 0 or eff_bb > threshold_bb:
        return policy

    cp = parsed["current_player"]
    contribs = parsed.get("contribution", [])
    if not contribs:
        return policy
    to_call = int(max(contribs)) - int(contribs[cp])
    facing = to_call > 0

    fold_idx = int(DiscreteAction.FOLD)
    call_idx = int(DiscreteAction.CALL)
    allin_idx = int(DiscreteAction.ALLIN)
    if facing:
        keep_idxs = (fold_idx, call_idx, allin_idx)
    elif legal_mask[call_idx] > 0:
        keep_idxs = (call_idx, allin_idx)
    else:
        keep_idxs = (fold_idx, allin_idx)

    keep_mask = np.zeros_like(policy)
    for a in keep_idxs:
        if legal_mask[a] > 0:
            keep_mask[a] = 1.0
    if keep_mask.sum() == 0:
        return policy   # no legal kept action — fall back

    new = policy * keep_mask
    s = float(new.sum())
    if s > 1e-12:
        new = (new / s).astype(policy.dtype)
    else:
        new = (keep_mask / float(keep_mask.sum())).astype(policy.dtype)
    return new


# --------------------------------------------------------------------------
# Commitment-scaled tail floor (H1; deployment-only policy patch, flag-OFF)
# --------------------------------------------------------------------------
#
# Sample mode draws from the full mixed strategy, so individual low-mass
# draws can commit the whole stack (session-1 seq-315: 2d5c open-shove
# sampled from a 7.5% ALLIN tail). Hypothesis (H1, binding spec at
# docs/research_program/H1_TAIL_FLOOR_SPEC.md): pruning low-probability
# actions *in proportion to how much they commit* gains EV against the
# live field without opening an exploitable leak.
#
# Threshold scales with commitment: tau(a) = tau_max * commit_frac(a),
# where commit_frac(a) = chips the action would ADD / hero stack. Prune
# every action with 0 < policy[a] < tau(a); renormalize. Consequences:
# FOLD and free CHECK are never pruned (commit 0); a 7.5% ALLIN is pruned
# at tau_max >= 0.075; a 7.5% quarter-stack bet needs tau_max >= 0.30.
# Calling off a shove is treated identically to jamming (commit ~= 1).
#
# Wiring is flag-gated OFF: make_live_policy_filter only invokes this
# function when tail_floor_tau is not None, so the OFF chain is
# byte-identical to the pre-H1 build (TG1 gate).

def apply_commitment_tail_floor(policy, legal_mask, parsed, state,
                                  tau_max: float,
                                  discrete_to_chip: dict | None = None):
    """Prune legal actions whose mass is positive but below
    tau_max * (chips-the-action-would-add / hero_stack); renormalize.

    Commitment source (adversarial-review finding, EXP_H1): when
    `discrete_to_chip` is supplied (the discretize map already built at
    the sampling call site), chips-added is EXACT: bet-class chip ints
    are whole-hand totals, so added = chip - contribution[cp]. The
    fallback formula approximates pot from sum(contribution), which
    UNDERSTATES universal_poker's parsed pot and misclassified 28/1579
    corpus bets in the dangerous (commit-underestimating) direction —
    kept only for callers without a chip map.

    Returns the SAME `policy` reference when nothing fires (identity
    short-circuit contract shared by all deployment floors)."""
    import numpy as np
    from src.nlhe.actions import DiscreteAction, BET_FRACTIONS

    cp = parsed.get("current_player")
    money = parsed.get("money", [])
    contribs = parsed.get("contribution", [])
    if cp is None or not money or not contribs:
        return policy
    if cp >= len(money) or cp >= len(contribs):
        return policy
    stack = float(money[cp])
    if stack <= 0:
        return policy
    to_call = float(max(contribs)) - float(contribs[cp])
    pot = float(sum(contribs))
    my_contrib = float(contribs[cp])

    def _commit(da: DiscreteAction) -> float:
        """Chips this action would ADD (CHECK == CALL at to_call=0 =>
        commit 0). FOLD/CALL/ALLIN are exact in both modes; bet sizes
        are exact iff discrete_to_chip is present."""
        if da == DiscreteAction.FOLD:
            return 0.0
        if da == DiscreteAction.CALL:
            return min(to_call, stack)
        if da == DiscreteAction.ALLIN:
            return stack
        if discrete_to_chip is not None:
            chip = discrete_to_chip.get(da)
            if chip is None:
                chip = discrete_to_chip.get(int(da))
            if chip is not None and chip not in (0, 1):
                return max(0.0, min(float(chip) - my_contrib, stack))
        f = BET_FRACTIONS[da]
        return min(to_call + f * (pot + to_call), stack)

    prune_idxs = []
    for da in DiscreteAction:
        idx = int(da)
        if legal_mask[idx] == 0.0:
            continue
        mass = float(policy[idx])
        if mass <= 0.0:
            continue
        tau = float(tau_max) * (_commit(da) / stack)
        if mass < tau:
            prune_idxs.append(idx)

    if not prune_idxs:
        return policy

    new_policy = policy.copy()
    for idx in prune_idxs:
        new_policy[idx] = 0.0
    s = float(new_policy.sum())
    if s <= 1e-12:
        # Degenerate: pruning would zero out the whole distribution.
        return policy
    return (new_policy / s).astype(policy.dtype)


# --------------------------------------------------------------------------
# Live policy filter — composes all deployment-time floors with logging
# --------------------------------------------------------------------------

def make_live_policy_filter(short_stack_threshold_bb: float = _DEFAULT_SHORT_STACK_FLOOR_BB,
                             *, log_prefix: str = "[FLOOR]",
                             tail_floor_tau: "float | None" = None):
    """Build the composed deployment-only policy filter.

    Order: AA/KK preflop → check-when-free → short-stack → tail floor.
    Each filter is identity-short-circuited when its gate doesn't fire
    (returns the same `policy` reference), so the chain's net cost when
    nothing fires is a few reference-equality checks.

    The H1 commitment-scaled tail floor runs LAST and ONLY when
    `tail_floor_tau` is not None — the default OFF chain never calls it
    and is byte-identical to the pre-H1 build.

    Logs to stdout each time any floor changes the action distribution.
    """
    import numpy as np
    from src.nlhe.actions import DiscreteAction

    def composed_filter(policy, legal_mask, parsed, state,
                        discrete_to_chip=None):
        p1 = apply_aa_kk_preflop_floor(policy, legal_mask, parsed, state)
        aa_kk_fired = (p1 is not policy)
        p2 = apply_check_when_free_floor(p1, legal_mask, parsed, state)
        check_free_fired = (p2 is not p1)
        p3 = apply_short_stack_floor(
            p2, legal_mask, parsed, state,
            threshold_bb=short_stack_threshold_bb)
        ss_fired = (p3 is not p2)
        if tail_floor_tau is not None:
            p4 = apply_commitment_tail_floor(
                p3, legal_mask, parsed, state, tau_max=tail_floor_tau,
                discrete_to_chip=discrete_to_chip)
        else:
            p4 = p3
        tail_fired = (p4 is not p3)

        if aa_kk_fired or check_free_fired or ss_fired or tail_fired:
            fires = []
            if aa_kk_fired: fires.append("AA/KK")
            if check_free_fired: fires.append("check-free")
            if ss_fired: fires.append("short-stack")
            if tail_fired: fires.append("tail")
            eff_bb, _ = _hero_eff_bb_from_parsed(parsed)
            cp = parsed.get("current_player", -1)
            # Best-effort argmax for pre/post audit
            masked_pre = policy * legal_mask
            masked_post = p4 * legal_mask
            pre_a = int(np.argmax(masked_pre)) if masked_pre.sum() > 0 else -1
            post_a = int(np.argmax(masked_post)) if masked_post.sum() > 0 else -1
            pre_name = DiscreteAction(pre_a).name if pre_a >= 0 else "?"
            post_name = DiscreteAction(post_a).name if post_a >= 0 else "?"
            street = parsed.get("street_idx", -1)
            tail_note = ""
            if tail_fired:
                pruned = [f"{DiscreteAction(i).name}:{float(p3[i]):.4f}"
                          for i in range(len(p3))
                          if float(p3[i]) > 0.0 and float(p4[i]) == 0.0]
                tail_note = f"  tail_pruned=[{','.join(pruned)}]"
            print(f"{log_prefix} fired=[{','.join(fires)}]  "
                  f"eff_bb={eff_bb:.2f}  cp={cp}  street={street}  "
                  f"pre_argmax={pre_name}  post_argmax={post_name}"
                  f"{tail_note}",
                  flush=True)
        return p4

    # Sampling call sites that have the discretize map check this marker
    # and pass discrete_to_chip= so the tail floor sees exact chip costs.
    # Callers unaware of it use the legacy 4-arg call — fully compatible.
    composed_filter.accepts_d2c = True
    return composed_filter


def _client_action_for_chip_int(chip_int: int, frame) -> dict:
    """Translate an OpenSpiel chip_int to a real-table client_action dict.
    Same shape as the inline dispatch in run_logonly_resolver.py."""
    if chip_int == 0:
        return {"kind": "fold", "chip_amount": None,
                "raw_openspiel_chip_int": 0}
    if chip_int == 1:
        kind = "call" if frame.hero_facing_bet else "check"
        return {"kind": kind, "chip_amount": None,
                "raw_openspiel_chip_int": 1}
    scraper_min_raise = max(
        frame.blinds.bb,
        2 * max((b for b in frame.bet), default=0))
    scraper_max_raise = (frame.stack[frame.hero_seat]
                          + frame.bet[frame.hero_seat])
    raise_amount = max(scraper_min_raise,
                        min(int(chip_int), scraper_max_raise))
    return {"kind": "raise_to",
            "chip_amount": int(raise_amount),
            "raw_openspiel_chip_int": int(chip_int)}


def _attempt_suspect_stack_recovery(record: dict, structure, tracker):
    """Layer-1 single-field recovery for scraper-suspect frames
    (2026-06-09 blackout postmortem: a stable stuck-digit hero-stack OCR
    suspect-flagged 12 consecutive otherwise-clean frames; 4 hero-to-act
    moments dropped; hero busted on the freeze).

    Returns (frame, recovered_fields, why):
      frame is None  -> recovery declined; `why` is the audit string the
                        caller appends to the unchanged safe-fold reason.
      frame is set   -> a re-validated recovered ScraperFrame;
                        recovered_fields like ["stack.seat1=1110 (strict)"].

    Gates — ALL must hold, else decline (caller drops the frame exactly as
    before this path existed):
      1. suspect_reasons name exactly ONE seat, stack-jump reasons only
      2. the record parses cleanly apart from the suspect flag
      3. the frame is a hero-to-act decision (recovery exists to prevent
         dropped decisions; non-decision suspect frames stay dropped)
      4. hero cards visible, pot > 0 (mirrors make_decision's own guards)
      5. a clean-frame pre-hand anchor exists for this hand-key
      6. the closure solve yields a candidate in (0, pre_hand − ante]
      7. the candidate frame passes replay_to_decision WITH the anchor
         (no simple-model fallback — the derived value's trust argument
         rests on the anchor) AND check_mid_hand_invariant
      8. if strict and UI-lag candidates BOTH survive 7 → ambiguous → decline

    The invariant is not loosened anywhere: the recovered frame clears the
    same replay+invariant bar as every clean frame, with the per-seat
    exactness of the five other stacks, all bets, pot, current_player,
    street and cards carrying the independent verification (the recovered
    seat's own stack equality and the pot-conservation check are satisfied
    by construction — documented power loss, see SESSION_LOG 2026-06-09).
    """
    from src.nlhe.integration.scraper_schema import (
        parse_frame, recoverable_suspect_seat,
        build_stack_recovery_candidates,
        ScraperParseError, ScraperDataQuality,
    )
    from src.nlhe.integration.replay import replay_to_decision, ReplayError
    from src.nlhe.integration.invariant import check_mid_hand_invariant

    bad_seat = recoverable_suspect_seat(record)
    if bad_seat is None:
        return None, None, "suspect_reasons not a single-seat stack jump"
    try:
        frame = parse_frame(record, allow_suspect=True)
    except (ScraperParseError, ScraperDataQuality) as e:
        return None, None, (f"suspect frame failed clean parse: "
                            f"{type(e).__name__}")
    if not frame.controls_present:
        return None, None, "not a hero-to-act frame"
    if frame.alive[frame.hero_seat] and not frame.hero_cards:
        return None, None, "hero cards missing"
    if frame.pot_total <= 0:
        return None, None, "pot_total <= 0"
    anchor = tracker.anchor_for(frame)
    if anchor is None:
        return None, None, "no clean pre-hand anchor for this hand-key"
    candidates = build_stack_recovery_candidates(frame, bad_seat, anchor)
    if not candidates:
        return None, None, "no in-range closure candidate"

    validated = []
    for cand_frame, label in candidates:
        # Mirror make_decision's pot handling exactly so the surviving
        # candidate behaves identically when it re-runs the main pipeline.
        eff = cand_frame
        corrected = tracker.corrected_pot_for(cand_frame)
        if corrected is not None:
            eff = dataclasses.replace(cand_frame, pot_total=int(corrected))
        try:
            pack = replay_to_decision(
                eff, structure, pre_hand_override=anchor)
        except ReplayError:
            continue
        inv = check_mid_hand_invariant(eff, pack)
        if not inv.ok:
            continue
        validated.append((cand_frame, label))

    if not validated:
        return None, None, "no candidate passed replay+invariant re-validation"
    if len(validated) > 1:
        return None, None, ("ambiguous: strict and uilag candidates both "
                            "reconstruct with different stacks")
    cand_frame, label = validated[0]
    recovered = [
        f"stack.seat{bad_seat + 1}={int(cand_frame.stack[bad_seat])} ({label})"
    ]
    return cand_frame, recovered, "ok"


def _attempt_bet_closure_recovery(frame, inv, structure, tracker):
    """P2 bet-closure recovery for invariant-failed frames with the
    displacement signature (2026-06-11 session-3 postmortem; live seq 276
    2026-06-12: seat5's posted BB rendered into its stack — scraper read
    stack=1222 bet=0 pot=560 vs true 1122/100/660; the frame parsed
    cleanly, was never suspect-flagged, and dropped as a 3-delta
    invariant_fail on a real AcKc decision).

    Returns (frame, pack, recovered_fields, why):
      frame None, why None  -> NOT the displacement signature; the caller
                               must drop the frame byte-identically to the
                               pre-P2 path (no annotation).
      frame None, why set   -> signature family, recovery REFUSED; caller
                               drops exactly as before plus the audit
                               annotation.
      frame set             -> re-validated corrected ScraperFrame + its
                               replay pack; recovered_fields like
                               ["stack.seat5=1122 (bet_closure)", ...].

    Gates — ALL must hold, else refuse:
      1. inv.deltas match the displacement signature: deltas confined to
         {pot, stack[K], bet[K]} for exactly ONE seat K, closing under a
         single bet/stack transfer of x chips (multi-seat or non-closing
         deltas refuse; any other delta type is not the signature)
      2. P1's anchor sum-floor guard is armed (P2 is gated on P1 — the
         derivation trusts the anchor, seq-1363 poisoned-anchor lesson)
      3. a clean hand-start anchor exists for this hand-key (regular or
         flag-gated pre-blind); an anchor REFUSED by the P1/ceiling
         guard is surfaced as its own refusal, never used
      3b. dead-SB guard: positive blind-structure evidence exists — a
         regular anchor (both blinds seen posted) or sb_only posting
         evidence; bb_only posting evidence always refuses (the live
         seq-276 ground-truth lesson: a dead-SB hand mimics the
         displacement signature with a phantom BB)
      4. the corrected frame passes chip conservation against the anchor
         (the independent post-patch check, inside
         build_bet_closure_candidate) and the in-range bounds
      5. the corrected frame passes replay_to_decision WITH the anchor
         (no simple-model fallback — the derived values' trust argument
         rests on the anchor) AND check_mid_hand_invariant
      6. if more than one DISTINCT anchor survives re-validation ->
         more than one candidate closure -> ambiguous -> refuse

    The invariant is not loosened anywhere: the corrected frame clears
    the same replay+invariant bar as every clean frame, with the five
    other seats' stacks, the other bets, current_player, street and
    cards carrying the independent verification (the patched seat's
    stack/bet and the pot equality are satisfied by construction —
    same documented power loss as the Layer-1 suspect recovery)."""
    from src.nlhe.integration.scraper_schema import (
        classify_displacement_deltas, build_bet_closure_candidate,
    )
    from src.nlhe.integration.replay import replay_to_decision, ReplayError
    from src.nlhe.integration.invariant import check_mid_hand_invariant

    kind, payload = classify_displacement_deltas(inv.deltas)
    if kind == "not_signature":
        return None, None, None, None
    if kind == "refused":
        return None, None, None, payload
    seat, x = payload

    if not getattr(tracker, "anchor_sum_floor_armed", False):
        return None, None, None, ("anchor sum-floor guard not armed "
                                  "(P2 is gated on P1)")
    anchors, refused_why = tracker.bet_closure_anchors_for(frame)
    if not anchors:
        if refused_why is not None:
            return None, None, None, refused_why
        return None, None, None, ("no clean hand-start anchor for this "
                                  "hand-key (regular or pre-blind)")

    # Dead-SB guard (live 2026-06-12 ground-truth disproof of the seq-276
    # "displacement"): a dead-SB hand — SB seat busted the previous hand,
    # BB falls one seat later — reproduces the displacement signature
    # EXACTLY once the BB poster's bet changes (per-frame dead-SB
    # detection loses its evidence, the replay falls back to the
    # next-alive-after-dealer default and reconstructs a phantom BB on
    # the wrong seat). Chip conservation cannot distinguish the two
    # (a displacement is sum-invariant everywhere), so recovery demands
    # positive evidence that the recon's blind assignment is right:
    # either a REGULAR anchor (is_hand_start saw both blinds posted) or
    # captured "sb_only" posting evidence (the BB was missing at the
    # post moment — a real displaced BB, never a dead one). "bb_only"
    # evidence is the dead-SB/displaced-SB pattern: always refuse.
    evidence = tracker.blind_posting_evidence_for(frame)
    has_regular_anchor = any(src == "anchor" for _, src in anchors)
    if evidence == "bb_only":
        return None, None, None, (
            "hand shows BB-only posting (dead-SB hand or displaced SB) "
            "— the reconstruction's blind assignment is unverifiable "
            "and the deltas may be a phantom-BB artifact")
    if not has_regular_anchor and evidence != "sb_only":
        return None, None, None, (
            "no blind-structure evidence: neither a regular hand-start "
            "anchor (both blinds seen posted) nor sb_only posting "
            "evidence exists for this hand-key")

    validated = []
    decline_why = None
    for pre, source in anchors:
        cand_frame, why = build_bet_closure_candidate(frame, seat, x, pre)
        if cand_frame is None:
            decline_why = why
            continue
        # Mirror make_decision's pot handling exactly so the surviving
        # candidate behaves identically when it re-runs the main
        # pipeline. (For a conservation-passing candidate the strict
        # closure holds, so this is a no-op by construction; kept for
        # exactness with the Layer-1 recovery path.)
        eff = cand_frame
        corrected = tracker.corrected_pot_for(cand_frame)
        if corrected is not None:
            eff = dataclasses.replace(cand_frame, pot_total=int(corrected))
        try:
            pack = replay_to_decision(
                eff, structure, pre_hand_override=pre)
        except ReplayError as e:
            decline_why = f"replay rejected corrected frame: {str(e)[:80]}"
            continue
        inv2 = check_mid_hand_invariant(eff, pack)
        if not inv2.ok:
            decline_why = ("corrected frame still fails the invariant "
                           f"({len(inv2.deltas)} deltas)")
            continue
        # No dedupe here: bet_closure_anchors_for already collapses
        # identical anchors, so two survivors mean two DISTINCT pre-hand
        # vectors both reconstruct cleanly — that's ambiguity (the packs
        # differ even when the corrected frame is the same), and the
        # arbitration below must refuse rather than pick one.
        validated.append((eff, pack, source))

    if not validated:
        return None, None, None, (
            "no candidate passed replay+invariant re-validation"
            + (f" ({decline_why})" if decline_why else ""))
    if len(validated) > 1:
        return None, None, None, (
            "ambiguous: multiple distinct anchors both re-validate the "
            "closure (regular vs pre-blind disagree on pre-hand stacks)")
    eff, pack, source = validated[0]
    recovered = [
        f"stack.seat{seat + 1}={int(eff.stack[seat])} (bet_closure)",
        f"bet.seat{seat + 1}={int(eff.bet[seat])} (bet_closure)",
        f"pot={int(eff.pot_total)} (bet_closure:{source})",
    ]
    return eff, pack, recovered, "ok"


def make_decision(
    record: dict,
    structure,
    solver,
    tracker,
    rng: random.Random,
    mode: str = "sample",
    seq: int | None = None,
    decision_cache: "DecisionCache | None" = None,
    short_stack_floor_bb: float = _DEFAULT_SHORT_STACK_FLOOR_BB,
    extended_click_plans: bool = False,
    tail_floor_tau: "float | None" = None,
    bet_closure_recovery: bool = False,
) -> LiveDecision:
    """Process one scraper record. Returns a LiveDecision.

    Args:
        record: a scraper record (same shape as one line of
            data/live_1500.jsonl).
        structure: TournamentStructure for this format.
        solver: a loaded DeepCFR6MaxSolver (e.g. via
            scripts/eval_6max_self_play._load_solver).
        tracker: a SessionTracker; .observe() is called on every record,
            .pre_hand_for() and .corrected_pot_for() are queried.
        rng: random.Random for sample-mode policy.
        mode: "sample" (production default) or "argmax" (audit).
        seq: optional monotonic counter from the socket protocol.
        decision_cache: optional DecisionCache for sample-once-per-decision
            behaviour. When supplied, the first frame of a new decision
            samples and stores the action; subsequent frames with the same
            decision identity return the cached action with
            status="decision_cached". When None, every frame samples
            independently (the pre-fix per-frame-lottery behaviour).
        tail_floor_tau: H1 commitment-scaled tail floor tau_max. Default
            None = OFF (the composed policy filter never calls the tail
            floor; live path byte-identical to pre-H1).
        bet_closure_recovery: P2 displacement-signature recovery on
            invariant-failed frames (see _attempt_bet_closure_recovery).
            Default False = OFF (invariant_fail drops byte-identical to
            pre-P2). Requires a tracker built with
            SessionTracker(anchor_sum_floor=True,
            bet_closure_recovery=True) — gated on P1.

    Returns:
        A LiveDecision. Never raises; internal errors surface as a
        `safe_fold` or `skip_*` status with `skip_reason` populated.
    """
    # Deferred imports — keep this module light at import time and
    # tolerate the pokerbot venv path.
    from src.nlhe.integration.scraper_schema import (
        parse_frame,
        ScraperParseError, ScraperSuspect, ScraperDataQuality,
    )
    from src.nlhe.integration.replay import replay_to_decision, ReplayError
    from src.nlhe.integration.invariant import check_mid_hand_invariant
    from src.nlhe.integration.click_target import (
        compute_click_target, click_plan_for_safe_fold,
    )

    captured_at = record.get("captured_at", "")
    out = LiveDecision(seq=seq, captured_at=captured_at)

    # 1. Parse the record. Soft drops surface as skip status.
    try:
        frame = parse_frame(record)
    except ScraperSuspect as e:
        # 1b. Layer-1 single-field recovery. A suspect frame whose ONLY
        # flagged problem is one seat's stack jump may be recoverable by
        # deriving that stack from the clean hand-start anchor via chip
        # conservation, then re-validating through the full replay +
        # invariant gate. Every decline path below is byte-for-byte the
        # pre-recovery drop, plus an audit note.
        frame, recovered, why = _attempt_suspect_stack_recovery(
            record, structure, tracker)
        if frame is None:
            out.status = "skip_data_quality"
            out.skip_reason = (f"{type(e).__name__}: {str(e)[:160]} "
                                f"(recovery declined: {why})")
            out.click_plan = click_plan_for_safe_fold(out.skip_reason)
            return out
        out.recovered_fields = recovered
    except (ScraperParseError, ScraperDataQuality) as e:
        out.status = "skip_data_quality"
        out.skip_reason = f"{type(e).__name__}: {str(e)[:160]}"
        out.click_plan = click_plan_for_safe_fold(out.skip_reason)
        return out

    # 2. Update tracker; apply UI-lag pot correction if the chip math
    # closes via (sum_stacks + pot + sum_bets == expected).
    # Recovered frames must NOT feed the tracker: pre-hand anchors come
    # from clean frames exclusively (a derived value anchoring future
    # derivations would be circular trust).
    if out.recovered_fields is None:
        out.anchor_refused = tracker.observe(frame) is False
    corrected_pot = tracker.corrected_pot_for(frame)
    if corrected_pot is not None:
        frame = dataclasses.replace(frame, pot_total=int(corrected_pot))
        out.pot_corrected = True

    # 3. Populate the READ summary.
    out.blinds = (int(frame.blinds.sb), int(frame.blinds.bb),
                   int(frame.blinds.ante))
    out.hero_seat = int(frame.hero_seat)
    out.dealer_seat = int(frame.dealer_seat)
    out.hero_cards = tuple(frame.hero_cards)
    out.board = tuple(frame.board)
    out.pot_total = int(frame.pot_total)
    out.hero_stack = int(frame.stack[frame.hero_seat]) \
        if frame.alive[frame.hero_seat] else 0
    out.n_alive = int(sum(frame.alive))
    out.facing_bet = bool(frame.hero_facing_bet)
    for bl in structure.blind_schedule:
        if (bl.small_blind == frame.blinds.sb
                and bl.big_blind == frame.blinds.bb
                and bl.ante == frame.blinds.ante):
            out.level = int(bl.level)
            break

    # OOD-warning: training_weights cover L1-L9; min stack-BB sampled
    # ≥ ~2BB. Live drifting past those bounds is silent → log it.
    if out.level is not None and (
        out.level >= 8 or
        (frame.blinds.bb > 0 and out.hero_stack and
         out.hero_stack / frame.blinds.bb < 2.0)
    ):
        print(f"[OOD-WARN] live out-of-training-support: "
              f"level={out.level} hero_stack={out.hero_stack} "
              f"({(out.hero_stack or 0)/max(1,frame.blinds.bb):.1f} BB) "
              f"captured_at={frame.captured_at}", flush=True)

    # 4. Hero-to-act gate. Hand-start frames + FOLD-only UI captures
    # have controls_present=False after the parse_frame filter.
    if not frame.controls_present:
        out.status = "skip_not_hero_to_act"
        out.skip_reason = ("controls.present=False (not a hero-to-act "
                            "decision: hand-start, muck-only UI, or "
                            "between actions)")
        out.click_plan = click_plan_for_safe_fold(out.skip_reason)
        return out
    if frame.alive[frame.hero_seat] and not frame.hero_cards:
        out.status = "skip_data_quality"
        out.skip_reason = ("hero alive but no cards visible "
                            "(between-hands or mid-deal capture)")
        out.click_plan = click_plan_for_safe_fold(out.skip_reason)
        return out
    if frame.pot_total <= 0:
        out.status = "skip_data_quality"
        out.skip_reason = "pot_total <= 0 (scraper failed to extract pot)"
        out.click_plan = click_plan_for_safe_fold(out.skip_reason)
        return out

    # 5. Replay to hero's decision via the real-ante bridge. Try
    # override path first; fall back to simple model on
    # asymmetric-commit rejection (same shape as
    # scripts/test_integration_midhand.py).
    pre_hand_override = tracker.pre_hand_for(frame)
    out.pre_hand_override_used = pre_hand_override is not None
    pack = None
    try:
        pack = replay_to_decision(
            frame, structure, pre_hand_override=pre_hand_override)
    except ReplayError as e:
        if pre_hand_override is not None:
            try:
                pack = replay_to_decision(
                    frame, structure, pre_hand_override=None)
                out.pre_hand_override_used = False
            except ReplayError as e2:
                out.status = "safe_fold"
                out.skip_reason = (f"replay_error (with + without override): "
                                    f"{str(e2)[:160]}")
                out.click_plan = click_plan_for_safe_fold(out.skip_reason)
                return out
        else:
            out.status = "safe_fold"
            out.skip_reason = f"replay_error: {str(e)[:160]}"
            out.click_plan = click_plan_for_safe_fold(out.skip_reason)
            return out

    out.street_idx = int(pack.street_idx)

    # 6. Strict invariant check.
    inv = check_mid_hand_invariant(frame, pack)
    if not inv.ok:
        # 6b. P2 bet-closure recovery (flag-gated). A frame whose ONLY
        # invariant failure is the displacement signature — one seat's
        # bet/stack split misrendered, deltas closing under a single
        # transfer — may be recoverable by deriving the corrected split
        # against the clean hand-start anchor and re-validating through
        # the full replay + invariant gate. Every refusal path below is
        # byte-for-byte the pre-P2 drop, plus an audit note; frames not
        # matching the signature drop with no annotation at all.
        bc_frame = bc_pack = bc_recovered = bc_why = None
        if bet_closure_recovery:
            bc_frame, bc_pack, bc_recovered, bc_why = (
                _attempt_bet_closure_recovery(frame, inv, structure,
                                              tracker))
        if bc_frame is None:
            out.status = "safe_fold"
            out.skip_reason = f"invariant_fail ({len(inv.deltas)} deltas)"
            if bc_why is not None:
                out.skip_reason += (f" (bet-closure recovery declined: "
                                     f"{bc_why})")
            out.invariant_deltas = [list(d) for d in inv.deltas]
            out.click_plan = click_plan_for_safe_fold(out.skip_reason)
            return out
        # Recovered: continue the pipeline on the corrected frame + pack,
        # refreshing every READ-summary field the correction can touch.
        frame = bc_frame
        pack = bc_pack
        out.recovered_fields = (out.recovered_fields or []) + bc_recovered
        out.pot_total = int(frame.pot_total)
        out.hero_stack = int(frame.stack[frame.hero_seat]) \
            if frame.alive[frame.hero_seat] else 0
        out.n_alive = int(sum(frame.alive))
        out.facing_bet = bool(frame.hero_facing_bet)
        out.street_idx = int(pack.street_idx)

    # 7. Decide-once gate. Compute the decision-identity from the frame;
    # if the cache holds an action for that identity (same actual decision,
    # different polling frame), reuse it. Otherwise sample a fresh action
    # and lock it into the cache. Skipping the policy sample on cache hit
    # is what locks in ONE action across the many frames of a single live
    # decision — without this, a mixed strategy re-draws on every frame
    # and the bot would click whichever sample fired (shove-vs-call lottery).
    cache_key = None
    cached_from_cache = False
    if decision_cache is not None:
        cache_key = decision_cache.identity(frame)
        out.decision_identity = cache_key
        cached = decision_cache.get(cache_key)
        if cached is not None:
            client_action_cached, chip_int = cached
            cached_from_cache = True

    if not cached_from_cache:
        from src.nlhe.infoset6 import parse_state_6max
        from scripts.eval_6max_self_play import _sample_action_from_policy
        parsed = parse_state_6max(pack.state, observer=frame.hero_seat)
        parsed["dealer_seat"] = frame.dealer_seat
        try:
            chip_int = _sample_action_from_policy(
                solver, parsed, pack.state, rng, mode=mode,
                policy_filter=make_live_policy_filter(
                    short_stack_threshold_bb=short_stack_floor_bb,
                    tail_floor_tau=tail_floor_tau),
            )
        except Exception as e:  # pragma: no cover (defensive)
            out.status = "safe_fold"
            out.skip_reason = f"policy_error: {type(e).__name__}: {str(e)[:120]}"
            out.click_plan = click_plan_for_safe_fold(out.skip_reason)
            return out

    # 8. Translate to client_action + populate sizing fields.
    if cached_from_cache:
        client_action = client_action_cached
    else:
        client_action = _client_action_for_chip_int(int(chip_int), frame)
        if decision_cache is not None and cache_key is not None:
            decision_cache.put(cache_key, client_action, int(chip_int))
    out.client_action = client_action
    out.resolver_raw_openspiel_chip_int = int(chip_int)
    if client_action.get("chip_amount") is not None:
        ch = int(client_action["chip_amount"])
        out.client_action_pot_frac = ch / max(1, frame.pot_total)
        out.client_action_bb_mult = ch / max(1, frame.blinds.bb)
        out.client_action_stack_frac = ch / max(1, out.hero_stack or 1)

    # 9. Compute the click plan (always recomputed: the controls' on-screen
    # coordinates can shift between frames even within a single decision).
    if extended_click_plans:
        hero_max_commit = (int(frame.stack[frame.hero_seat])
                           + int(frame.bet[frame.hero_seat]))
        out.click_plan = compute_click_target(
            client_action, record.get("controls") or {},
            extended=True, hero_max_commit=hero_max_commit)
    else:
        out.click_plan = compute_click_target(
            client_action, record.get("controls") or {})

    if out.recovered_fields is not None:
        out.status = ("decision_recovered_cached" if cached_from_cache
                       else "decision_recovered")
    else:
        out.status = "decision_cached" if cached_from_cache else "decision"
    return out
