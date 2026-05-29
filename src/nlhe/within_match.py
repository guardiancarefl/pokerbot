"""Layer 4 / C1a — within-match opponent observation.

Pure data layer. Observes public actions and showdowns, accumulates per-seat
stats with NO cross-match state retention and NO module-level globals.

Anti-pattern to avoid: archetype6.py:78's `_warned_no_dealer` class-level latch.
All mutable state MUST live on the MatchObserver instance, not on the class.

Locked design: docs/scratch/session_handoff/layer4_decisions_locked.md.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# ---- Module-level CONSTANTS (immutable; per the anti-latch rule, no module-
# level mutable state is allowed). ----

# Confidence ramp boundaries (n_actions). Locked from C1_PLAN §"Decay and confidence".
_CONF_RAMP_START = 20       # below this: confidence = 0
_CONF_RAMP_MID = 100        # at this: confidence = 0.5
_CONF_RAMP_END = 300        # at or above this: confidence = 1.0

# DiscreteAction integer values (kept as local constants so this module does NOT
# import src.nlhe.actions at module load — the lazy import lives inside update()).
_A_FOLD = 0
_A_CALL = 1
_A_BET_33 = 2
_A_BET_66 = 3
_A_BET_100 = 4
_A_BET_200 = 5
_A_ALLIN = 6
_AGGRESSIVE_ACTIONS = frozenset({_A_BET_33, _A_BET_66, _A_BET_100, _A_BET_200, _A_ALLIN})


@dataclass
class SeatStats:
    """Raw counters for one seat. Consumers compute ratios from these on the fly."""
    seat: int
    n_actions: int = 0                       # total public decisions observed
    # Preflop
    n_preflop_decisions: int = 0
    n_preflop_voluntary: int = 0             # VPIP numerator (call above blinds OR any raise)
    n_preflop_raises: int = 0                # PFR numerator
    # Postflop aggression — index by street_idx (1=flop, 2=turn, 3=river); slot 0 unused.
    n_postflop_decisions: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    n_postflop_aggressive: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    # Fold-to-bet — by street_idx 0..3 (preflop included).
    n_facing_bet: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    n_folds_facing_bet: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    # Showdown (outcome-only, per §6-Q2 lock)
    n_showdowns: int = 0
    n_showdown_wins: int = 0
    # Bet sizing
    sum_bet_size_over_pot: float = 0.0       # running sum of (bet_size/pot) for each bet
    n_bet_size_samples: int = 0


class MatchObserver:
    """Per-match opponent observation accumulator. Wipe at match boundaries.

    ALL mutable state lives on the instance. NO class-level latches, NO module-
    level mutable globals. wipe() must restore the instance to behavior identical
    to a freshly-constructed observer.
    """

    def __init__(self, num_seats: int = 6):
        self._num_seats = int(num_seats)
        self._stats: list[SeatStats] = [SeatStats(seat=i) for i in range(self._num_seats)]

    # ---- Observation ----

    def update(self, state: Any, parsed: dict, action: int, seat: int) -> None:
        """Record one public decision by `seat`.

        Parameters
        ----------
        state : universal_poker state PRIOR to this action being applied (used
            only for bet-size extraction; may be None in tests — bet-size sample
            is then skipped, other counters still update).
        parsed : output of parse_state_6max / parse_state_repeated_6max for the
            same state. Must contain at minimum street_idx and contribution.
        action : DiscreteAction integer in [0, 6] that `seat` just played.
        seat : acting seat index in [0, num_seats).
        """
        if not (0 <= seat < self._num_seats):
            return  # silently ignore out-of-range seats; not a hard error
        stats = self._stats[seat]
        stats.n_actions += 1

        street_idx = int(parsed.get("street_idx", 0))
        contribution = parsed.get("contribution", [0] * self._num_seats)
        # facing_bet: this seat has committed strictly fewer chips than the max
        # committed by any seat on the table. Pure parsed-dict signal; works across
        # both parse_state_6max and parse_state_repeated_6max outputs.
        try:
            seat_contrib = int(contribution[seat])
            max_contrib = int(max(contribution)) if contribution else 0
        except (TypeError, ValueError, IndexError):
            seat_contrib = 0
            max_contrib = 0
        facing_bet = seat_contrib < max_contrib

        is_fold = (action == _A_FOLD)
        is_call = (action == _A_CALL)
        is_aggressive = action in _AGGRESSIVE_ACTIONS

        # Per-street facing-bet counters (preflop included).
        if facing_bet and 0 <= street_idx <= 3:
            stats.n_facing_bet[street_idx] += 1
            if is_fold:
                stats.n_folds_facing_bet[street_idx] += 1

        # Preflop-specific counters.
        if street_idx == 0:
            stats.n_preflop_decisions += 1
            if is_aggressive:
                stats.n_preflop_raises += 1
                stats.n_preflop_voluntary += 1
            elif is_call and facing_bet:
                # CALL above the blind = VPIP. CALL as BB-option-check (no bet
                # to face) is not voluntary; the facing_bet guard filters it.
                stats.n_preflop_voluntary += 1
        else:
            # Postflop aggression (slot 0 unused; only 1..3 written).
            if 1 <= street_idx <= 3:
                stats.n_postflop_decisions[street_idx] += 1
                if is_aggressive:
                    stats.n_postflop_aggressive[street_idx] += 1

        # Bet sizing — best-effort. If state is None or any helper raises,
        # skip the sample (other counters already updated above).
        if is_aggressive and state is not None:
            try:
                from src.nlhe.actions import DiscreteAction, discretize_legal_actions
                from src.nlhe.cfr6 import _build_view_6max
                view = _build_view_6max(state, parsed)
                discrete_to_chip = discretize_legal_actions(list(state.legal_actions()), view)
                chip_amount = discrete_to_chip.get(DiscreteAction(action))
                pot_pre = float(parsed.get("pot", 0) or 0)
                if chip_amount is not None and pot_pre > 0:
                    stats.sum_bet_size_over_pot += float(chip_amount) / pot_pre
                    stats.n_bet_size_samples += 1
            except Exception:
                # TODO(c1b): tighten bet-size extraction once we identify what
                # the real failure modes are in production eval.
                pass

    def note_showdown(self, seats_reaching_showdown: list[int],
                      winner_seats: list[int]) -> None:
        """Update showdown counters (outcome-only, §6-Q2 lock).

        seats_reaching_showdown : seats with non-folded hole cards at river end.
        winner_seats : subset of seats_reaching_showdown that won (≥1 for any
            non-degenerate showdown). Seats not in seats_reaching_showdown are
            untouched.
        """
        winners = set(int(s) for s in winner_seats)
        for s in seats_reaching_showdown:
            s = int(s)
            if not (0 <= s < self._num_seats):
                continue
            stats = self._stats[s]
            stats.n_showdowns += 1
            if s in winners:
                stats.n_showdown_wins += 1

    # ---- Read API ----

    def get_stats(self, seat: int) -> SeatStats:
        """Return a deepcopy of `seat`'s SeatStats. Caller mutations must NOT
        affect observer state, so a fresh copy is returned every call."""
        if not (0 <= seat < self._num_seats):
            return SeatStats(seat=seat)
        return copy.deepcopy(self._stats[seat])

    def confidence(self, seat: int) -> float:
        """Per-seat confidence in [0, 1] from n_actions.

        Ramp (locked from C1_PLAN §"Decay and confidence"):
            n < 20           → 0.0
            20 ≤ n < 100     → linear 0.0 → 0.5
            100 ≤ n < 300    → linear 0.5 → 1.0
            n ≥ 300          → 1.0
        """
        if not (0 <= seat < self._num_seats):
            return 0.0
        n = self._stats[seat].n_actions
        if n < _CONF_RAMP_START:
            return 0.0
        if n < _CONF_RAMP_MID:
            return (n - _CONF_RAMP_START) / float(_CONF_RAMP_MID - _CONF_RAMP_START) * 0.5
        if n < _CONF_RAMP_END:
            return 0.5 + (n - _CONF_RAMP_MID) / float(_CONF_RAMP_END - _CONF_RAMP_MID) * 0.5
        return 1.0

    # ---- Lifecycle ----

    def wipe(self) -> None:
        """Reset all per-seat state to fresh zero. Idempotent.

        After wipe(), the observer behaves IDENTICALLY to a freshly-constructed
        MatchObserver(num_seats). No latches, no caches, no 'already warned'
        flags survive.
        """
        self._stats = [SeatStats(seat=i) for i in range(self._num_seats)]

    def match_started(self) -> None:
        """Eval-loop convenience hook. Equivalent to wipe() — present for
        symmetry with match_ended() so callers can bracket matches cleanly."""
        self.wipe()

    def match_ended(self) -> None:
        """Eval-loop convenience hook. Equivalent to wipe()."""
        self.wipe()
