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
    #                              or a flag-gated CR commit reconciliation:
    #                              the UNCHANGED frame decided on a replay
    #                              rebuilt from anchor-implied commits
    #                              (2026-06-13; labels
    #                              "(commit_reconciliation...)")
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
# Slate-2a shove-defense floor (deployment-only policy patch, flag-OFF)
# --------------------------------------------------------------------------
#
# Ported VERBATIM from scripts/shove_defense_probe.make_shove_defense_floor
# (the probe-only harness; src/nlhe/integration/live_loop.py was untouched
# at probe time). M-A re-grade (evals/a2a_shove_floor_probe_20260612):
# m1=0.00346 vs 0.0867 baseline (96% loss reduction), call-mass 0.051,
# 7497/8112 spots fired, confusion fired_oracle_fold=7464 / fired_oracle_call=33.
# M-B CRN-paired self-play A/B was non-negative (z=+4.7 pooled, slate-2a).
#
# THE FLOOR (registered tau=0, the pure break-even gate): at a preflop
# facing-all-in node with hero eff-stack in [5,15] BB and exactly ONE
# all-in raiser (call commits hero's whole stack), compute hero hand-class
# equity vs the FROZEN killphil shove range of the NEAREST battery cell
# (evals/h2_battery/battery_v1.json "ranges") and the Malmuth-Harville ICM
# break-even call equity. If equity < break-even + tau, move all non-FOLD
# legal mass (== CALL/ALLIN mass at these nodes) to FOLD; else identity.
#
# CELL MAPPING for off-grid nodes (verbatim from the probe registration):
#   depth  = hero hand-start stack in BB = (hero money + hero contribution)
#            / real BB; gated to [5,15]; nearest of {5,8,11,15} (ties->lower).
#   level  = blind level from the structure schedule; nearest of {3,5,7}
#            (ties->lower; levels 1-2 -> 3, levels >= 8 -> 7).
#   shover = "SB" when the shover seat posted a blind this hand (SB or BB
#            seat per the game-string blind array), else "UTG". The battery
#            froze only UTG and SB shover cells; UTG is the nearest
#            (tightest) cell for any non-blind open-shover.
#
# SCOPE GUARDS (strict no-op unless ALL hold): street==preflop; real BB
# known; to_call > 0; to_call >= hero remaining stack; exactly one opponent
# at max contribution and that opponent is ALL-IN (money == 0); every other
# non-hero seat's contribution <= BB + max ante; hero hole cards visible;
# FOLD legal.
#
# Wiring is flag-gated OFF: make_live_policy_filter only invokes this floor
# when shove_defense_floor is not None, so the OFF chain is byte-identical
# to the pre-slate-2a build (Gate 1).

_SHOVE_DEPTH_GRID = (5, 8, 11, 15)
_SHOVE_LEVEL_GRID = (3, 5, 7)
_SHOVE_DEFAULT_TAU = 0.0
_SHOVE_PAYOUTS = (2.0, 2.0, 2.0)
_SHOVE_DEFAULT_RANGE_TABLE = "evals/h2_battery/battery_v1.json"


def _shove_nearest_grid(x: float, grid) -> int:
    """Nearest grid point; ties resolve to the LOWER value. Verbatim from
    scripts/shove_defense_probe._nearest_grid."""
    return min(grid, key=lambda g: (abs(g - x), g))


def _shove_blind_info(state):
    """(sb_seat, bb_seat, ante_max) from the game-string params. Verbatim
    from scripts/shove_defense_probe._blind_info."""
    try:
        params = state.get_game().get_parameters()
        blind_str = str(params.get("blind", ""))
        vals = [int(x) for x in blind_str.split()]
    except Exception:
        return None, None, 0
    if not vals or max(vals) <= 0:
        return None, None, 0
    bb = max(vals)
    bb_seat = vals.index(bb)
    sub = [(v, i) for i, v in enumerate(vals) if 0 < v < bb]
    sb_seat = sub[0][1] if len(sub) == 1 else None
    ante_max = 0
    try:
        ante_str = str(params.get("ante", ""))
        avals = [int(x) for x in ante_str.split()]
        if avals:
            ante_max = max(avals)
    except Exception:
        ante_max = 0
    return sb_seat, bb_seat, ante_max


def _shove_icm_for(stacks, hero):
    """Hero ICM equity. Verbatim from scripts/shove_defense_probe._icm_for."""
    from src.nlhe.icm import icm_equity
    eligible = [i for i in range(len(stacks)) if stacks[i] > 0]
    if hero not in eligible:
        return 0.0
    return float(icm_equity(stacks, list(_SHOVE_PAYOUTS), eligible=eligible)[hero])


def _shove_qualify(parsed, state, bb_to_level):
    """Return (cell_key, hero_cards, ev_fold, ev_win, ev_lose, depth_bb) if
    the node qualifies for the shove-defense floor, else None. Ported
    verbatim from scripts/shove_defense_probe._qualify (the fold/win/lose
    ICM arithmetic mirrors fold_vs_shove_battery.oracle_ev)."""
    if int(parsed.get("street_idx", -1)) != 0:
        return None
    bb = int(parsed.get("big_blind", 0))
    if bb <= 0:
        return None
    cp = parsed.get("current_player")
    m = parsed.get("money") or []
    c = parsed.get("contribution") or []
    n = len(m)
    if cp is None or n == 0 or len(c) != n:
        return None
    mx = max(c)
    to_call = mx - c[cp]
    if to_call <= 0 or m[cp] <= 0 or to_call < m[cp]:
        return None                          # call must commit hero's stack
    shovers = [j for j in range(n) if j != cp and c[j] == mx]
    if len(shovers) != 1:
        return None
    j = shovers[0]
    if m[j] != 0:
        return None                           # the raiser must be all-in
    sb_seat, bb_seat, ante_max = _shove_blind_info(state)
    for k in range(n):
        if k in (cp, j):
            continue
        if c[k] > bb + ante_max:
            return None                       # caller behind -> disqualify
    depth_bb = float(m[cp] + c[cp]) / bb
    if not (5.0 <= depth_bb <= 15.0):
        return None                           # registration window
    lvl_actual = bb_to_level.get(bb)
    if lvl_actual is None:
        return None
    d = _shove_nearest_grid(depth_bb, _SHOVE_DEPTH_GRID)
    lvl = _shove_nearest_grid(lvl_actual, _SHOVE_LEVEL_GRID)
    pos = "SB" if (j == sb_seat or j == bb_seat) else "UTG"
    cell_key = f"{pos}|{d}|{lvl}"
    priv = parsed.get("private_cards", "") or ""
    if len(priv) != 4:
        return None
    hero_cards = (priv[0:2], priv[2:4])

    # Scenario stacks. Busted-seat placeholders (stack<=1, no post) -> 0.
    base = [0 if (m[k] + c[k]) <= 1 else int(m[k]) for k in range(n)]
    pot = int(sum(c))
    hero_total = int(c[cp] + m[cp])
    dead = int(sum(c[k] for k in range(n) if k not in (cp, j)))
    stacks_fold = list(base)
    stacks_fold[cp] = int(m[cp])
    stacks_fold[j] = pot                      # m[j] == 0; shover scoops
    stacks_win = list(base)
    stacks_win[cp] = 2 * hero_total + dead
    stacks_win[j] = int(c[j]) - hero_total    # refund (c[j] >= hero_total)
    stacks_lose = list(base)
    stacks_lose[cp] = 0
    stacks_lose[j] = int(c[j]) + hero_total + dead
    ev_fold = _shove_icm_for(stacks_fold, cp)
    ev_win = _shove_icm_for(stacks_win, cp)
    ev_lose = _shove_icm_for(stacks_lose, cp)
    return cell_key, hero_cards, ev_fold, ev_win, ev_lose, depth_bb


class ShoveDefenseFloor:
    """Loaded-once shove-defense floor state + the apply hook.

    The range table (evals/h2_battery/battery_v1.json "ranges") and the
    BB->level map are loaded ONCE at construction (filter-build time), never
    per-decision. The equity-vs-range Monte Carlo is memoized per
    (hero_cards, cell_key) in `eq_cache` exactly as the probe does.

    `apply(...)` is the deployment hook: same identity short-circuit
    contract as every other floor (returns the SAME `policy` reference when
    the gate doesn't fire)."""

    def __init__(self, structure, ranges: dict, tau: float = _SHOVE_DEFAULT_TAU,
                 *, range_table_path: str | None = None,
                 stats: dict | None = None, eq_cache: dict | None = None):
        self.ranges = ranges
        self.tau = float(tau)
        self.range_table_path = range_table_path
        self.bb_to_level = {int(bl.big_blind): int(bl.level)
                            for bl in structure.blind_schedule}
        if stats is None:
            stats = {}
        for k in ("n_calls", "n_qualify", "n_fired"):
            stats.setdefault(k, 0)
        stats.setdefault("cells", {})
        self.stats = stats
        self.eq_cache = {} if eq_cache is None else eq_cache

    def apply(self, policy, legal_mask, parsed, state, discrete_to_chip=None):
        import numpy as np
        from src.nlhe.actions import DiscreteAction
        from scripts.fold_vs_shove_battery import _equity_vs_labels

        self.stats["n_calls"] += 1
        q = _shove_qualify(parsed, state, self.bb_to_level)
        if q is None:
            return policy
        cell_key, hero_cards, ev_fold, ev_win, ev_lose, _depth = q
        labels = self.ranges.get(cell_key)
        if not labels:
            return policy
        cstat = self.stats["cells"].setdefault(cell_key, [0, 0])
        cstat[0] += 1                         # qualify-count for this cell
        self.stats["n_qualify"] += 1
        denom = ev_win - ev_lose
        if denom <= 1e-12:
            return policy
        eq_star = (ev_fold - ev_lose) / denom
        ck = (hero_cards, cell_key)
        eq = self.eq_cache.get(ck)
        if eq is None:
            eq = _equity_vs_labels(hero_cards, set(labels))
            self.eq_cache[ck] = eq
        if eq >= eq_star + self.tau:
            return policy                     # equity clears the gate
        a_fold = int(DiscreteAction.FOLD)
        if legal_mask[a_fold] <= 0:
            return policy
        # Move ALL non-FOLD legal mass (== CALL/ALLIN at these nodes) to FOLD.
        legal_mass = float(sum(float(policy[i]) for i in range(len(policy))
                               if legal_mask[i] > 0))
        if legal_mass <= 0:
            return policy
        new = np.zeros_like(np.asarray(policy, dtype=np.float64))
        new[a_fold] = 1.0
        self.stats["n_fired"] += 1
        cstat[1] += 1
        return new.astype(np.asarray(policy).dtype)


def load_shove_defense_floor(range_table_path: str = _SHOVE_DEFAULT_RANGE_TABLE,
                             tau: float = _SHOVE_DEFAULT_TAU,
                             structure=None) -> ShoveDefenseFloor:
    """Build a ShoveDefenseFloor from the battery range table on disk.

    The table is read ONCE here (filter-construction time). Refuses with a
    clear error if the table is missing or carries no "ranges" key. When
    `structure` is None the battery's own structure YAML is loaded so the
    BB->level map matches the frozen cells exactly."""
    from pathlib import Path
    import json
    from src.nlhe.actions import DiscreteAction  # noqa: F401 (import parity)

    p = Path(range_table_path)
    if not p.exists():
        raise FileNotFoundError(
            f"shove-defense range table not found: {range_table_path} "
            f"(the frozen killphil shove battery; expected at "
            f"{_SHOVE_DEFAULT_RANGE_TABLE})")
    table = json.loads(p.read_text())
    ranges = table.get("ranges")
    if not ranges:
        raise ValueError(
            f"shove-defense range table {range_table_path} has no non-empty "
            f"'ranges' key — refusing to arm a no-op floor")
    if structure is None:
        from src.nlhe.game_strings import TournamentStructure
        structure = TournamentStructure.from_yaml(table["structure"])
    return ShoveDefenseFloor(structure, ranges, tau=tau,
                             range_table_path=range_table_path)


# --------------------------------------------------------------------------
# Live policy filter — composes all deployment-time floors with logging
# --------------------------------------------------------------------------

def make_live_policy_filter(short_stack_threshold_bb: float = _DEFAULT_SHORT_STACK_FLOOR_BB,
                             *, log_prefix: str = "[FLOOR]",
                             tail_floor_tau: "float | None" = None,
                             shove_defense_floor: "ShoveDefenseFloor | None" = None):
    """Build the composed deployment-only policy filter.

    Order: AA/KK preflop → check-when-free → short-stack → shove-defense
    → tail floor. Each filter is identity-short-circuited when its gate
    doesn't fire (returns the same `policy` reference), so the chain's net
    cost when nothing fires is a few reference-equality checks.

    The slate-2a shove-defense floor runs AFTER short-stack and ONLY when
    `shove_defense_floor` is not None (a loaded ShoveDefenseFloor). It
    composes with the H1 tail floor — both can fire on the same decision
    (shove-defense moves CALL/ALLIN mass to FOLD; the tail floor then
    sees the post-shove-defense distribution).

    The H1 commitment-scaled tail floor runs LAST and ONLY when
    `tail_floor_tau` is not None — the default OFF chain never calls it
    and is byte-identical to the pre-H1 build. The default OFF chain
    (shove_defense_floor=None, tail_floor_tau=None) is byte-identical to
    the pre-slate-2a build (Gate 1).

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
        if shove_defense_floor is not None:
            p3b = shove_defense_floor.apply(
                p3, legal_mask, parsed, state,
                discrete_to_chip=discrete_to_chip)
        else:
            p3b = p3
        shove_fired = (p3b is not p3)
        if tail_floor_tau is not None:
            p4 = apply_commitment_tail_floor(
                p3b, legal_mask, parsed, state, tau_max=tail_floor_tau,
                discrete_to_chip=discrete_to_chip)
        else:
            p4 = p3b
        tail_fired = (p4 is not p3b)

        if aa_kk_fired or check_free_fired or ss_fired or shove_fired or tail_fired:
            fires = []
            if aa_kk_fired: fires.append("AA/KK")
            if check_free_fired: fires.append("check-free")
            if ss_fired: fires.append("short-stack")
            if shove_fired: fires.append("shove-defense")
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
                pruned = [f"{DiscreteAction(i).name}:{float(p3b[i]):.4f}"
                          for i in range(len(p3b))
                          if float(p3b[i]) > 0.0 and float(p4[i]) == 0.0]
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


def _attempt_suspect_stack_recovery(record: dict, structure, tracker,
                                    dead_button_handling: bool = False,
                                    allin_zero_stack: bool = False):
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
        frame = parse_frame(record, allow_suspect=True,
                            dead_button_handling=dead_button_handling,
                            allin_zero_stack=allin_zero_stack)
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


def _attempt_commit_reconciliation(frame, inv, structure, tracker):
    """Anchored commit reconciliation for invariant-failed frames whose
    deltas are confined to pot/stack/bet fields (CR — 2026-06-13 session-5
    build; pre-registered in evals/p2_session5_validation_20260613/
    REPORT.txt). The MIRROR family of P2: the SCRAPER frame is correct
    (anchor-conserving; outcome ground truth sided with it 3/3 on
    2026-06-12) and the RECONSTRUCTION under-counts non-hero commits —
    swept folded blinds (live seq 780), limp-then-fold (seq 1086),
    ante-bookkeeping on an all-in-for-less hand (seq 1635).

    Returns (pack, recovered_fields, why):
      pack None, why None  -> NOT the commit-family signature; the caller
                              must drop the frame byte-identically to the
                              pre-CR path (no annotation).
      pack None, why set   -> family, reconciliation REFUSED; caller
                              drops exactly as before plus the audit
                              annotation.
      pack set             -> re-validated replay pack for the UNCHANGED
                              scraper frame; recovered_fields like
                              ["commit.seat4=50 (commit_reconciliation)",
                               ...]. The caller continues the pipeline on
                              the ORIGINAL frame + this pack — CR never
                              patches the frame (that is P2's direction,
                              proven backwards for this class).

    Gates — ALL must hold, else refuse:
      1. inv.deltas are all pot/stack[K]/bet[K] fields (any other delta
         type is not the signature — byte-identical drop)
      2. P1's anchor sum-floor guard is armed (CR derives per-seat
         commits from the anchor; seq-1363 poisoned-anchor lesson)
      3. a clean REGULAR hand-start anchor exists for this hand-key.
         The regular anchor doubles as the positive blind-structure
         evidence the rebuild's blind assignment needs: is_hand_start
         fires only when BOTH blinds were seen posted (live-SB hand), so
         a dead-SB hand (the seq-276 fixture family, bb_only posting)
         can never serve one and always refuses here. Pre-blind anchors
         are NOT used: they carry no blind evidence.
      4. the frame passes strict chip conservation against the anchor
         and every per-seat implied commit is consistent (inside
         build_commit_reconciliation_shadow; see its gate list)
      5. the rebuilt action sequence replays (replay_to_decision WITH
         the anchor — no simple-model fallback; the rebuild's trust
         argument rests on the anchor) AND the UNCHANGED scraper frame
         passes check_mid_hand_invariant against the rebuilt state, with
         the gate's ante-bookkeeping HEURISTICS replaced by the
         rebuild's exact emission knowledge (ReconciledEmission — every
         equality check is untouched)

    Determinism note: derive_action_sequence is deterministic given
    (shadow, anchor), and the invariant's exact per-seat equalities pin
    every externally observable chip of the resulting state to the
    scraper frame — the "multiple legal sequences with different hero
    game states" ambiguity class is excluded by construction: any
    sequence the gate would accept reproduces the same stacks, bets,
    pot, street, board and current_player the model decides on."""
    from src.nlhe.integration.scraper_schema import (
        classify_commit_reconciliation_deltas,
        build_commit_reconciliation_shadow,
        derive_action_sequence, ActionDerivationError,
    )
    from src.nlhe.integration.replay import replay_to_decision, ReplayError
    from src.nlhe.integration.invariant import (
        check_mid_hand_invariant, ReconciledEmission,
    )
    from src.nlhe.infoset6 import parse_state_6max

    if classify_commit_reconciliation_deltas(inv.deltas) != "candidate":
        return None, None, None

    if not getattr(tracker, "anchor_sum_floor_armed", False):
        return None, None, ("anchor sum-floor guard not armed "
                            "(commit reconciliation is gated on P1)")
    anchor = tracker.anchor_for(frame)
    if anchor is None:
        return None, None, (
            "no clean REGULAR hand-start anchor for this hand-key — the "
            "regular anchor is also the positive blind-structure evidence "
            "(both blinds seen posted at hand start); refused/absent "
            "anchors and dead-SB-family hands stop here")

    shadow, restored, why = build_commit_reconciliation_shadow(
        frame, anchor)
    if shadow is None:
        return None, None, why

    # Rebuild + replay. derive_action_sequence runs on the SHADOW (the
    # restored folded bets re-enter the blind-detection and delayed-fold
    # emission paths); the emitted sequence is recomputed here only to
    # extract exact emission knowledge — replay_to_decision re-derives
    # the identical sequence internally (deterministic, same inputs).
    try:
        action_seq = derive_action_sequence(
            shadow, pre_hand_override=anchor)
    except ActionDerivationError as e:
        return None, None, (f"rebuilt action derivation failed: "
                            f"{str(e)[:120]}")
    try:
        pack = replay_to_decision(
            shadow, structure, pre_hand_override=anchor)
    except ReplayError as e:
        return None, None, f"rebuilt replay failed: {str(e)[:120]}"

    # Exact emission knowledge for the gate's ante bookkeeping
    # (ReconciledEmission): pre-hand-convention all-in RAISES present in
    # the replayed state (emitted raise whose post-replay money == 0 —
    # catches replay.py's Class B chip-int replacement too, since the
    # convention is read off the STATE), and preflop limp/raise-then-fold
    # seats (a voluntary action AND a fold emitted for the same seat).
    parsed = parse_state_6max(pack.state, observer=frame.hero_seat)
    money = parsed["money"]
    contrib = parsed["contribution"]
    raise_ints = tuple(sorted(
        int(contrib[s]) for (s, ci) in action_seq
        if ci not in (0, 1) and int(money[s]) == 0
    ))
    voluntary_seats = {s for (s, ci) in action_seq if ci != 0}
    fold_seats = {s for (s, ci) in action_seq if ci == 0}
    delayed = tuple(sorted(
        s for s in (voluntary_seats & fold_seats) if frame.alive[s]
    ))

    inv2 = check_mid_hand_invariant(
        frame, pack,
        reconciled_emission=ReconciledEmission(
            all_in_raise_chip_ints=raise_ints,
            preflop_delayed_fold_seats=delayed,
        ))
    if not inv2.ok:
        return None, None, (
            f"rebuilt replay still fails the invariant "
            f"({len(inv2.deltas)} deltas)")

    recovered = [
        f"commit.seat{s + 1}={int(c)} (commit_reconciliation)"
        for s, c in sorted(restored.items())
    ]
    recovered.append(
        f"replay=rebuilt_from_anchor_commits "
        f"(commit_reconciliation:{len(restored)} seats restored)")
    return pack, recovered, "ok"


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
    shove_defense_floor: "ShoveDefenseFloor | None" = None,
    bet_closure_recovery: bool = False,
    dead_button_handling: bool = False,
    commit_reconciliation: bool = False,
    allin_zero_stack: bool = False,
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
        shove_defense_floor: slate-2a shove-defense floor — a loaded
            ShoveDefenseFloor (range table + tau, built ONCE via
            load_shove_defense_floor) or None. Default None = OFF (the
            composed policy filter never calls it; live path byte-identical
            to pre-slate-2a). Runs after short-stack and composes with the
            tail floor (both can fire). At a facing-all-in 5-15bb node,
            moves CALL/ALLIN mass to FOLD when hero equity vs the frozen
            killphil shove range is below the ICM break-even + tau.
        bet_closure_recovery: P2 displacement-signature recovery on
            invariant-failed frames (see _attempt_bet_closure_recovery).
            Default False = OFF (invariant_fail drops byte-identical to
            pre-P2). Requires a tracker built with
            SessionTracker(anchor_sum_floor=True,
            bet_closure_recovery=True) — gated on P1.
        dead_button_handling: D2 dead-button position handling (see
            scraper_schema.parse_frame). Default False = OFF
            (dealer-on-empty frames soft-drop byte-identical to pre-D2).
            When True, dead-button frames parse with dealer_dead=True and
            flow through the unchanged SB/BB post-validation + replay +
            invariant chain.
        commit_reconciliation: CR anchored commit reconciliation on
            invariant-failed frames (see _attempt_commit_reconciliation)
            — the MIRROR family of P2: scraper right, reconstruction
            under-counts non-hero commits. Default False = OFF
            (invariant_fail drops byte-identical to pre-CR). Requires a
            tracker built with SessionTracker(anchor_sum_floor=True) —
            gated on P1. Runs AFTER bet_closure_recovery when both are
            armed (disjoint classes; P2's re-validation refuses
            mirror-class frames, proven live 2026-06-12).
        allin_zero_stack: F2 zero-stack all-in handling (see
            scraper_schema.parse_frame). Default False = OFF (a 0-stack
            bet-0 seat reads non-alive and a dealer pointing there
            soft-drops byte-identical to pre-F2). When True, a
            committed (pot-arithmetic-evidenced) explicit-0 seat is a
            valid all-in: the frame parses (no dealer-on-dead drop, no
            dead-button misclassification, counts toward n_alive>=4)
            and flows through the unchanged anchor + replay + invariant
            chain, where the busted-mid-hand machinery models it.

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
        frame = parse_frame(record,
                            dead_button_handling=dead_button_handling,
                            allin_zero_stack=allin_zero_stack)
    except ScraperSuspect as e:
        # 1b. Layer-1 single-field recovery. A suspect frame whose ONLY
        # flagged problem is one seat's stack jump may be recoverable by
        # deriving that stack from the clean hand-start anchor via chip
        # conservation, then re-validating through the full replay +
        # invariant gate. Every decline path below is byte-for-byte the
        # pre-recovery drop, plus an audit note.
        frame, recovered, why = _attempt_suspect_stack_recovery(
            record, structure, tracker,
            dead_button_handling=dead_button_handling,
            allin_zero_stack=allin_zero_stack)
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
    # F2: committed zero-stack all-in seats are in the hand — the READ
    # summary counts them (flag OFF: allin_seats all-False, identical).
    out.n_alive = int(sum(frame.alive)) + int(sum(frame.allin_seats))
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
        # 6c. CR anchored commit reconciliation (flag-gated; the MIRROR
        # family of P2 — scraper right, recon under-counts non-hero
        # commits). Consulted only when P2 did not recover. The frame is
        # NEVER patched: on success the pipeline continues on the
        # ORIGINAL frame with the rebuilt replay pack. Refusals annotate;
        # non-signature frames drop with no annotation at all.
        cr_pack = cr_recovered = cr_why = None
        if commit_reconciliation and bc_frame is None:
            cr_pack, cr_recovered, cr_why = (
                _attempt_commit_reconciliation(frame, inv, structure,
                                               tracker))
        if bc_frame is None and cr_pack is None:
            out.status = "safe_fold"
            out.skip_reason = f"invariant_fail ({len(inv.deltas)} deltas)"
            if bc_why is not None:
                out.skip_reason += (f" (bet-closure recovery declined: "
                                     f"{bc_why})")
            if cr_why is not None:
                out.skip_reason += (f" (commit reconciliation declined: "
                                     f"{cr_why})")
            out.invariant_deltas = [list(d) for d in inv.deltas]
            out.click_plan = click_plan_for_safe_fold(out.skip_reason)
            return out
        if bc_frame is not None:
            # Recovered (P2): continue the pipeline on the corrected
            # frame + pack, refreshing every READ-summary field the
            # correction can touch.
            frame = bc_frame
            pack = bc_pack
            out.recovered_fields = (out.recovered_fields or []) + bc_recovered
            out.pot_total = int(frame.pot_total)
            out.hero_stack = int(frame.stack[frame.hero_seat]) \
                if frame.alive[frame.hero_seat] else 0
            out.n_alive = (int(sum(frame.alive))
                           + int(sum(frame.allin_seats)))
            out.facing_bet = bool(frame.hero_facing_bet)
            out.street_idx = int(pack.street_idx)
        else:
            # Recovered (CR): the SCRAPER frame is trusted unchanged —
            # only the believed replay was rebuilt. READ-summary fields
            # already reflect the (correct) frame; refresh the
            # pack-derived street only.
            pack = cr_pack
            out.recovered_fields = (out.recovered_fields or []) + cr_recovered
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
                    tail_floor_tau=tail_floor_tau,
                    shove_defense_floor=shove_defense_floor),
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
