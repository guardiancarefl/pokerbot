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
    #   "safe_fold"              — invariant fail / replay error → no plan
    #   "skip_not_hero_to_act"   — frame parsed, hero not to act
    #   "skip_data_quality"      — parse_frame raised or hero-cards missing
    status: str = "..."
    skip_reason: str | None = None
    pre_hand_override_used: bool = False

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


def make_decision(
    record: dict,
    structure,
    solver,
    tracker,
    rng: random.Random,
    mode: str = "sample",
    seq: int | None = None,
    decision_cache: "DecisionCache | None" = None,
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
    except (ScraperParseError, ScraperSuspect, ScraperDataQuality) as e:
        out.status = "skip_data_quality"
        out.skip_reason = f"{type(e).__name__}: {str(e)[:160]}"
        out.click_plan = click_plan_for_safe_fold(out.skip_reason)
        return out

    # 2. Update tracker; apply UI-lag pot correction if the chip math
    # closes via (sum_stacks + pot + sum_bets == expected).
    tracker.observe(frame)
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
        out.status = "safe_fold"
        out.skip_reason = f"invariant_fail ({len(inv.deltas)} deltas)"
        out.invariant_deltas = [list(d) for d in inv.deltas]
        out.click_plan = click_plan_for_safe_fold(out.skip_reason)
        return out

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
                policy_filter=apply_aa_kk_preflop_floor,
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
    out.click_plan = compute_click_target(
        client_action, record.get("controls") or {})

    out.status = "decision_cached" if cached_from_cache else "decision"
    return out
