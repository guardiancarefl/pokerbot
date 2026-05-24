"""Action abstraction for HUNL.

Converts between:
  - Our discrete policy action space: {fold, call, 0.33pot, 0.66pot, 1pot, 2pot, allin}
  - OpenSpiel universal_poker's integer-chip action space (~20000 actions per node)

Two-way translation:
  - policy_to_game_action: discrete action -> integer chip bet (for our policy's choices)
  - game_to_policy_action: integer chip bet -> probability over discrete actions
    (for translating opponent moves; uses pseudo-harmonic mapping for off-tree sizes)

Discrete bet sizes are expressed as multiples of the current pot.

References:
  - Ganzfried & Sandholm, "Action Translation in Extensive-Form Games with
    Large Action Spaces", IJCAI 2013 (pseudo-harmonic mapping)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


class DiscreteAction(IntEnum):
    FOLD = 0
    CALL = 1
    BET_33 = 2     # bet sized to 0.33x current pot
    BET_66 = 3     # bet sized to 0.66x current pot
    BET_100 = 4    # bet sized to 1.0x current pot ("pot")
    BET_200 = 5    # bet sized to 2.0x current pot ("overbet")
    ALLIN = 6

    @property
    def label(self) -> str:
        return {
            DiscreteAction.FOLD: "fold",
            DiscreteAction.CALL: "call",
            DiscreteAction.BET_33: "0.33pot",
            DiscreteAction.BET_66: "0.66pot",
            DiscreteAction.BET_100: "pot",
            DiscreteAction.BET_200: "2xpot",
            DiscreteAction.ALLIN: "allin",
        }[self]


# Ordered list of bet actions for translation lookup, ascending in pot-fraction.
BET_ACTIONS_IN_ORDER = (
    DiscreteAction.BET_33,
    DiscreteAction.BET_66,
    DiscreteAction.BET_100,
    DiscreteAction.BET_200,
    DiscreteAction.ALLIN,
)

# Pot-fraction for each bet action (allin has no fixed fraction, handled separately).
BET_FRACTIONS: dict[DiscreteAction, float] = {
    DiscreteAction.BET_33: 0.33,
    DiscreteAction.BET_66: 0.66,
    DiscreteAction.BET_100: 1.00,
    DiscreteAction.BET_200: 2.00,
}

# Treat opponent bets at or above this fraction of effective stack as ALLIN.
ALLIN_TRANSLATE_THRESHOLD = 0.90


@dataclass(frozen=True)
class GameStateView:
    """Minimal subset of state needed to translate actions.

    Designed so callers can populate this from any backend (OpenSpiel,
    custom engine) without coupling the action module to a specific API.
    """
    pot: int                    # total chips in the pot right now (including any committed this street)
    to_call: int                # chips the actor needs to add to call
    effective_stack: int        # remaining chips behind for the actor with the smaller stack
    min_bet: int                # legal minimum total bet (OpenSpiel "Bet X" X value)
    max_bet: int                # legal maximum total bet (typically actor's all-in chip count)
    legal_fold: bool            # is fold a legal action right now (false if no bet to fold to)
    legal_call: bool            # is call a legal action (false if can already check for free)


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def policy_to_game_action(action: DiscreteAction, view: GameStateView) -> int | None:
    """Convert a discrete policy action to an OpenSpiel integer game action.

    Returns:
        The integer action to pass to state.apply_action(), or None if the
        discrete action is not legal in this state.

    OpenSpiel universal_poker action encoding:
        0       = fold
        1       = call/check
        N >= 2  = bet to N total chips this street (the X in "Bet X")
    """
    if action == DiscreteAction.FOLD:
        return 0 if view.legal_fold else None

    if action == DiscreteAction.CALL:
        # "Call" in OpenSpiel is also the check action when there's nothing to call.
        # Always action index 1 when legal.
        return 1

    if action == DiscreteAction.ALLIN:
        # Bet for the actor's full remaining stack. min_bet/max_bet are total-bet
        # values; max_bet equals the all-in size.
        return view.max_bet

    if action in BET_FRACTIONS:
        # Total bet = pot * fraction. But OpenSpiel's "Bet X" means total chips
        # in the bet, which for the first bet of a street is the bet itself, and
        # for a raise is the raise-to amount.
        target = int(round(view.pot * BET_FRACTIONS[action]))
        # If our intended target is below the legal minimum bet, this discrete
        # action is unavailable in this state (e.g. 0.33pot bet on a 200-chip
        # pot when min_bet=200). Return None so downstream code can exclude it
        # from the policy's softmax rather than aliasing several actions to the
        # same chip count.
        if target < view.min_bet:
            return None
        target = _clamp(target, view.min_bet, view.max_bet)
        return target

    raise ValueError(f"unknown action: {action}")


def _legal_discrete_bet_sizes(view: GameStateView) -> list[tuple[DiscreteAction, int]]:
    """Return discrete bet actions paired with their integer-chip size, in ascending order.

    Skips bet sizes that fall below min_bet (those discrete actions are
    unavailable in this state). For ties at the same chip count from clipping
    at max_bet, keeps only the largest discrete label (e.g., if both BET_200
    and ALLIN would equal max_bet, ALLIN is the meaningful label).

    KNOWN ALIAS (facing all-in, no re-raise room): when no raise is legal
    (hero faces a shove and lacks chips to re-raise), the state's legal
    actions are just {fold=0, call=1} and _build_view_6max sets
    min_bet == max_bet == 0. Every bet fraction then clamps to 0 and ALLIN
    takes max_bet == 0, so this function returns [(ALLIN, 0)]. Because chip 0
    is also FOLD, discretize_legal_actions emits {FOLD: 0, CALL: 1, ALLIN: 0}
    and policy_to_game_action(ALLIN, view) returns 0. CONSEQUENCE: selecting
    DiscreteAction.ALLIN in this state translates to chip 0, which
    state.apply_action() executes as a FOLD, not an all-in. The
    semantically-correct all-in-equivalent here is CALL (1) — calling a shove
    for one's entire stack IS the all-in. This is a label-only alias (no real
    all-in action exists in legal_actions); behavior is intentionally left
    unchanged here. Callers that translate ALLIN back to a game action when
    facing a shove (e.g. SubgamePolicy, sub-step 5) must account for it.
    """
    out: list[tuple[DiscreteAction, int]] = []
    seen_chips: dict[int, DiscreteAction] = {}
    for action in BET_ACTIONS_IN_ORDER:
        if action == DiscreteAction.ALLIN:
            chips = view.max_bet
        else:
            chips = int(round(view.pot * BET_FRACTIONS[action]))
            if chips < view.min_bet:
                # This discrete action is below the legal minimum; not playable here.
                continue
            chips = _clamp(chips, view.min_bet, view.max_bet)
        # Later (larger) actions overwrite earlier ones at the same chip count,
        # so the dict ends up holding the largest label for any tie.
        seen_chips[chips] = action
    # Emit in ascending chip-count order so downstream callers iterate from
    # smallest to largest bet.
    for chips in sorted(seen_chips.keys()):
        out.append((seen_chips[chips], chips))
    return out


def _pseudo_harmonic_weight(x: float, a: float, b: float) -> float:
    """Probability of mapping observed bet x to lower discrete size a.

    Used for opponent bet translation. From Ganzfried & Sandholm 2013.

    Args:
        x: observed bet size (pot fraction)
        a: lower discrete size (pot fraction)
        b: upper discrete size (pot fraction)

    Returns:
        Probability in [0,1] of mapping x to size a. (1 - this) maps to b.

    Mathematical property: returns 1 if x == a, 0 if x == b, monotone between.
    Approximates the optimal randomized translation strategy.
    """
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    # Pseudo-harmonic: P(map to a) = ((b - x) * (a + x)) / ((b - a) * (a + 2x))
    num = (b - x) * (a + x)
    denom = (b - a) * (a + 2 * x)
    if denom == 0:
        return 0.5
    return num / denom


def game_to_policy_action(
    chip_action: int,
    view: GameStateView,
) -> dict[DiscreteAction, float]:
    """Translate an OpenSpiel integer action to a probability distribution over discrete actions.

    Returns a dict mapping DiscreteAction -> probability. The distribution always
    sums to 1.0. For most actions only one DiscreteAction has weight 1.0; for
    off-tree bet sizes the distribution is split between the two neighboring
    discrete sizes via pseudo-harmonic translation.

    Args:
        chip_action: integer action from state.legal_actions() (0=fold, 1=call, N=bet to N)
        view: current game state view

    Returns:
        {DiscreteAction: probability} dict, summing to 1.0.
    """
    if chip_action == 0:
        return {DiscreteAction.FOLD: 1.0}
    if chip_action == 1:
        return {DiscreteAction.CALL: 1.0}

    # Bet/raise action. chip_action is the total bet target.
    # First check: is this effectively all-in?
    if view.effective_stack > 0:
        stack_fraction = chip_action / max(view.max_bet, 1)
        if stack_fraction >= ALLIN_TRANSLATE_THRESHOLD or chip_action >= view.max_bet:
            return {DiscreteAction.ALLIN: 1.0}

    # Compute observed bet as a fraction of pot.
    if view.pot <= 0:
        # Degenerate state — shouldn't happen mid-hand, but be defensive.
        return {DiscreteAction.BET_100: 1.0}
    x_frac = chip_action / view.pot

    # Find the two discrete sizes the observed bet falls between.
    # Treat ALLIN as covered by the threshold check above; here we only need to
    # locate x_frac within BET_FRACTIONS.
    sorted_fracs = sorted(BET_FRACTIONS.items(), key=lambda kv: kv[1])
    # If smaller than smallest, map fully to smallest.
    if x_frac <= sorted_fracs[0][1]:
        return {sorted_fracs[0][0]: 1.0}
    # If larger than largest (and not all-in), map fully to largest non-allin.
    if x_frac >= sorted_fracs[-1][1]:
        return {sorted_fracs[-1][0]: 1.0}

    # Find bracketing pair.
    for i in range(len(sorted_fracs) - 1):
        a_action, a_frac = sorted_fracs[i]
        b_action, b_frac = sorted_fracs[i + 1]
        if a_frac <= x_frac <= b_frac:
            p_a = _pseudo_harmonic_weight(x_frac, a_frac, b_frac)
            return {a_action: p_a, b_action: 1.0 - p_a}

    # Should be unreachable given the boundary checks above.
    raise RuntimeError(f"bet translation failed for x_frac={x_frac}")


# ----- Helpers for working with OpenSpiel legal_actions() -----

def discretize_legal_actions(
    legal_chip_actions: Sequence[int],
    view: GameStateView,
) -> dict[DiscreteAction, int]:
    """Return mapping of legal DiscreteAction -> the chip action that implements it.

    Used at decision time: query this dict to know which discrete actions are
    actually playable, then map our policy's softmax over discrete actions onto
    only the legal subset.
    """
    legal_set = set(legal_chip_actions)
    out: dict[DiscreteAction, int] = {}

    if 0 in legal_set:
        out[DiscreteAction.FOLD] = 0
    if 1 in legal_set:
        out[DiscreteAction.CALL] = 1

    # For each discrete bet action, compute its target and check it's in legal_set.
    for action, chips in _legal_discrete_bet_sizes(view):
        if chips in legal_set:
            out[action] = chips
    return out
