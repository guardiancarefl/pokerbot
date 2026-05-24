"""Shanky bot profile policy adapter — bridges Shanky profiles to the
Policy protocol used by eval_pool.py and LeaguePool.

This is the third module of Phase 5 (parser + runtime + policy adapter).
With it, any of the 36 Shanky archetype profiles can be dropped into
eval_pool.py or used as a sampled opponent in league training, exactly
like a CheckpointPolicy wrapping a trained CFR model.

Three responsibilities:

1. **GameContext construction**: read the parsed OpenSpiel state dict and
   build a `runtime.GameContext` with all the predicate-needed fields
   (hole cards, board, street, betting state, position, stacks in BBs).

2. **Action translation**: convert Shanky's Action (continuous-ish, e.g.
   `raise 80% potsize`, `raisemax`, `betpot`) to the bot's discrete action
   space (DiscreteAction enum: FOLD, CALL, BET_33, BET_66, BET_100, BET_200,
   ALLIN).

3. **Policy protocol matching**: expose `select_action(parsed, state, rng,
   mode)` returning an integer OpenSpiel chip-action.

The action mapping is intentionally lossy: Shanky has more sizing freedom
than our discrete buckets. The adapter picks the closest discrete bet and
falls back gracefully when the chosen action isn't legal (e.g., the profile
says "raise" but only check/call are available).
"""
from __future__ import annotations

import logging
import random
from typing import Any, Optional

from src.nlhe.actions import (
    DiscreteAction,
    GameStateView,
    discretize_legal_actions,
)
# _build_view_6max is lazy-imported inside select_action() to avoid pulling
# in the full ML stack (treys, torch, etc.) when this module is just being
# imported for its translation helpers (e.g., in unit tests).
from src.nlhe.scripted_bots.parser import (
    Action as ShankyAction,
    ActionKind,
    Profile,
    parse_profile,
)
from src.nlhe.scripted_bots.runtime import (
    GameContext,
    card_to_int,
    evaluate_profile,
)


log = logging.getLogger("shanky_policy")


# ============================================================
# Card-string parsing
# ============================================================

def _parse_cards_string(s: str) -> list[int]:
    """Parse OpenSpiel's card string (e.g. 'AsKh' or 'Kh2c3d') to a list of ints.

    OpenSpiel writes 2-char cards: rank (2-9TJQKA) + suit (cdhs), no separators.
    Empty string yields empty list.
    """
    if not s:
        return []
    out: list[int] = []
    # Step through 2 chars at a time.
    for i in range(0, len(s), 2):
        if i + 1 >= len(s):
            break
        rank = s[i]
        suit = s[i + 1].lower()
        try:
            out.append(card_to_int(rank, suit))
        except KeyError:
            # Skip malformed; let downstream handle empty hands.
            continue
    return out


# ============================================================
# Street / position derivation
# ============================================================

_STREET_NAMES = ("preflop", "flop", "turn", "river")


def _street_name(street_idx: int) -> str:
    """street_idx 0..3 -> 'preflop' .. 'river'."""
    if 0 <= street_idx < len(_STREET_NAMES):
        return _STREET_NAMES[street_idx]
    return "preflop"


def _position_name(current_player: int, num_players: int, button_seat: int = 0) -> str:
    """Map current player's seat to a position name Shanky predicates expect.

    For 6-max with the button at `button_seat`:
      - button seat: 'button'
      - button+1 (SB): 'smallblind'
      - button+2 (BB): 'bigblind'
      - earlier seats: 'first' (UTG, MP)
      - later seats: 'last' (CO)
      - middle: 'middle'
    """
    # Relative position (0 = button)
    rel = (current_player - button_seat) % num_players
    if num_players == 6:
        if rel == 0:
            return "button"
        if rel == 1:
            return "smallblind"
        if rel == 2:
            return "bigblind"
        if rel == 3:
            return "first"      # UTG
        if rel == 4:
            return "middle"     # MP
        if rel == 5:
            return "last"       # CO
    # Generic fallback
    if rel == 0:
        return "button"
    if rel == 1:
        return "smallblind"
    if rel == 2:
        return "bigblind"
    return "middle"


# ============================================================
# Sequence parsing
# ============================================================

def _count_actions_on_current_street(sequences: str) -> dict:
    """Parse the sequences string to count actions on the current street.

    OpenSpiel universal_poker sequences encoding:
      'c' = check or call
      'r<N>' = raise/bet to N chips
      'f' = fold
      '/' = street separator

    Returns {raises, bets, calls, checks, folds, last_action_char}.

    Note: 'c' represents both checks and calls — OpenSpiel doesn't distinguish.
    We treat any 'c' following a 'c' or street-start as a check; following
    an 'r' as a call. This is a heuristic but matches typical Shanky usage.
    """
    if not sequences:
        return {"raises": 0, "bets": 0, "calls": 0, "checks": 0, "folds": 0,
                "last_action": "none"}

    # Take only the current (last) street's sequence
    streets = sequences.split("/")
    current = streets[-1] if streets else ""

    raises = 0
    bets = 0
    calls = 0
    checks = 0
    folds = 0
    last = "none"
    has_raise = False     # tracks within this street whether a raise/bet has occurred

    i = 0
    while i < len(current):
        ch = current[i]
        if ch == "r":
            # raise/bet — skip past the digits
            if has_raise:
                raises += 1
                last = "raise"
            else:
                bets += 1
                has_raise = True
                last = "raise"
            i += 1
            while i < len(current) and current[i].isdigit():
                i += 1
        elif ch == "c":
            if has_raise:
                calls += 1
                last = "call"
            else:
                checks += 1
                last = "check"
            i += 1
        elif ch == "f":
            folds += 1
            last = "fold"
            i += 1
        else:
            i += 1

    return {
        "raises": raises,
        "bets": bets,
        "calls": calls,
        "checks": checks,
        "folds": folds,
        "last_action": last,
    }


# ============================================================
# Building GameContext from parsed state
# ============================================================

def build_game_context(parsed: dict, state: Any, big_blind_chips: int = 100) -> GameContext:
    """Construct a Shanky GameContext from OpenSpiel parse + state.

    Args:
        parsed: output of `parse_state_6max(state)` — dict with keys
            street_idx, current_player, pot, money, contribution,
            private_cards, public_cards, sequences, num_players.
        state: OpenSpiel state (used for legal_actions and chance/terminal checks).
        big_blind_chips: chips per BB in this game. Used to convert money to BBs.
            Defaults to 100 (common in OpenSpiel universal_poker default config).

    Returns:
        GameContext with hole cards, board, street, position, betting state,
        all chip amounts converted to BBs.
    """
    cp = parsed["current_player"]
    num_players = parsed.get("num_players", 6)
    money = parsed.get("money", [0] * num_players)
    contribution = parsed.get("contribution", [0] * num_players)
    pot = parsed.get("pot", 0)

    # Card strings → integer encoding
    hole_cards = _parse_cards_string(parsed.get("private_cards", ""))
    board = _parse_cards_string(parsed.get("public_cards", ""))

    # Street
    street = _street_name(parsed.get("street_idx", 0))

    # Chip-to-BB conversion (Shanky predicates expect BB units)
    def to_bb(chips: int) -> float:
        if big_blind_chips <= 0:
            return float(chips)
        return chips / big_blind_chips

    # Hero state
    my_money = money[cp] if cp < len(money) else 0
    max_contrib = max(contribution) if contribution else 0
    my_contrib = contribution[cp] if cp < len(contribution) else 0
    to_call_chips = max(0, max_contrib - my_contrib)

    # Effective stack vs largest active opponent
    opp_stacks = [money[i] for i in range(len(money)) if i != cp and money[i] > 0]
    effective_stack = min(my_money, max(opp_stacks)) if opp_stacks else my_money

    # Action counts (current street)
    action_counts = _count_actions_on_current_street(parsed.get("sequences", ""))

    # Opponents alive
    opponents_active = sum(1 for i, m in enumerate(money) if i != cp and m > 0)

    # Position name
    position = _position_name(cp, num_players)

    # Bot's last action (we ourselves; in eval contexts the bot=hero is the Shanky bot)
    # Reading the bot's last action requires knowing the last token in `sequences`
    # contributed by this player. We approximate as the most recent action token
    # of any kind, which works as long as the bot just acted (which is when we're
    # called — at a decision node).
    botslastaction = action_counts.get("last_action", "none")

    # Construct context
    ctx = GameContext(
        hole_cards=hole_cards,
        board=board,
        street=street,
        raises=action_counts["raises"],
        bets=action_counts["bets"],
        calls=action_counts["calls"],
        folds=action_counts["folds"],
        checks=action_counts["checks"],
        amounttocall=to_bb(to_call_chips),
        betsize=to_bb(max_contrib),
        potsize=to_bb(pot),
        stacksize=to_bb(my_money),
        bigblindsize=1.0,        # always 1.0 since chip amounts already in BBs
        totalinvested=to_bb(my_contrib),
        opponents=opponents_active,
        opponentsattable=num_players - 1,    # any seated, including folded
        opponentsonflop=opponents_active,    # approximation; deserves dedicated tracking
        stilltoact=0,                         # OpenSpiel doesn't expose; safe default
        position=position,
        botslastaction=botslastaction,
        opponentisallin_flag=any(m == 0 for i, m in enumerate(money) if i != cp),
        maxopponentstacksize=to_bb(max(opp_stacks)) if opp_stacks else 0.0,
        minopponentstacksize=to_bb(min(opp_stacks)) if opp_stacks else 0.0,
    )

    return ctx


# ============================================================
# Action translation: Shanky Action → DiscreteAction
# ============================================================

def shanky_action_to_discrete(
    action: ShankyAction,
    discrete_to_chip: dict,
    view: GameStateView,
) -> Optional[DiscreteAction]:
    """Translate a Shanky Action into a legal DiscreteAction.

    Args:
        action: the Shanky Action returned by `evaluate_profile`.
        discrete_to_chip: the output of `discretize_legal_actions` — a dict
            mapping DiscreteAction enum members to chip-action ints. Only
            these are legal.
        view: the GameStateView for the current decision.

    Returns:
        A DiscreteAction from the keys of discrete_to_chip, or None if no
        sensible translation exists (caller falls back to fold/check).
    """
    if action is None:
        return None

    legal_set = set(discrete_to_chip.keys())
    if not legal_set:
        return None

    kind = action.kind

    # Direct mappings
    if kind == ActionKind.FOLD:
        # Fold if legal, else check (DiscreteAction.CALL covers check when no bet to face)
        return DiscreteAction.FOLD if DiscreteAction.FOLD in legal_set else (
            DiscreteAction.CALL if DiscreteAction.CALL in legal_set else None
        )

    if kind == ActionKind.CALL:
        return DiscreteAction.CALL if DiscreteAction.CALL in legal_set else None

    if kind == ActionKind.CHECK:
        # CHECK is represented by CALL in our enum (you "call" 0)
        return DiscreteAction.CALL if DiscreteAction.CALL in legal_set else None

    if kind == ActionKind.RAISE_MAX or kind == ActionKind.BET_MAX:
        return DiscreteAction.ALLIN if DiscreteAction.ALLIN in legal_set else _largest_legal_bet(legal_set)

    if kind == ActionKind.RAISE_POT or kind == ActionKind.BET_POT:
        return _nearest_legal_bet(DiscreteAction.BET_100, legal_set)

    if kind == ActionKind.RAISE_MIN or kind == ActionKind.BET_MIN:
        return _smallest_legal_bet(legal_set)

    if kind in (ActionKind.RAISE_PERCENT, ActionKind.BET_PERCENT):
        # Pot-percent bets
        amt = action.amount or 0
        target = action.amount_target or "potsize"
        if target == "potsize":
            pct = amt / 100.0
        elif target == "stacksize":
            # Convert stack-fraction to pot-fraction approximately
            if view.pot > 0:
                pct = (amt / 100.0) * view.effective_stack / view.pot
            else:
                pct = amt / 100.0
        else:
            pct = amt / 100.0
        return _pot_fraction_to_discrete(pct, legal_set, view)

    if kind == ActionKind.RAISE_AMOUNT or kind == ActionKind.BET_AMOUNT:
        # Bare amount (in BB by Shanky convention)
        # If amount is None → engine default → treat as 2/3 pot
        if action.amount is None:
            return _nearest_legal_bet(DiscreteAction.BET_66, legal_set)
        # Translate BB count to pot fraction
        bb_to_chips = view.pot / max(1, view.pot)  # gives 1.0; we trust amount as BB already
        # Compare to pot: amount_in_chips ≈ amount * 1 BB
        # Simpler heuristic: small raises → BET_33/BET_66, larger → BET_100/200, very large → ALLIN
        if view.pot > 0:
            pct = action.amount / max(1.0, view.pot)
        else:
            pct = 1.0
        return _pot_fraction_to_discrete(pct, legal_set, view)

    if kind in (ActionKind.SITOUT, ActionKind.BEEP):
        # No equivalent in our discrete action space; sit out is effectively a fold.
        # BEEP is a test-bot debug action; fold is the safe interpretation.
        return DiscreteAction.FOLD if DiscreteAction.FOLD in legal_set else (
            DiscreteAction.CALL if DiscreteAction.CALL in legal_set else None
        )

    if kind == ActionKind.SECTION_DEFAULT:
        # Should be resolved before reaching here.
        return DiscreteAction.FOLD if DiscreteAction.FOLD in legal_set else None

    if kind == ActionKind.SET_USER_FLAG:
        # Handled in runtime; should never reach the adapter.
        return None

    return None


def _largest_legal_bet(legal_set: set) -> Optional[DiscreteAction]:
    """Return the largest-sized legal bet action (closest to all-in)."""
    for da in (DiscreteAction.ALLIN, DiscreteAction.BET_200, DiscreteAction.BET_100,
               DiscreteAction.BET_66, DiscreteAction.BET_33):
        if da in legal_set:
            return da
    return None


def _smallest_legal_bet(legal_set: set) -> Optional[DiscreteAction]:
    """Return the smallest-sized legal bet action."""
    for da in (DiscreteAction.BET_33, DiscreteAction.BET_66, DiscreteAction.BET_100,
               DiscreteAction.BET_200, DiscreteAction.ALLIN):
        if da in legal_set:
            return da
    return None


def _nearest_legal_bet(target: DiscreteAction, legal_set: set) -> Optional[DiscreteAction]:
    """Return the legal bet action closest to `target` in bet-size ordering."""
    if target in legal_set:
        return target
    # Bet actions in size order
    ordered = [DiscreteAction.BET_33, DiscreteAction.BET_66, DiscreteAction.BET_100,
               DiscreteAction.BET_200, DiscreteAction.ALLIN]
    if target not in ordered:
        return _smallest_legal_bet(legal_set)
    target_idx = ordered.index(target)
    # Walk outward from target
    for delta in range(1, len(ordered)):
        for direction in (-1, 1):
            idx = target_idx + delta * direction
            if 0 <= idx < len(ordered) and ordered[idx] in legal_set:
                return ordered[idx]
    return None


def _pot_fraction_to_discrete(
    pct: float, legal_set: set, view: GameStateView,
) -> Optional[DiscreteAction]:
    """Map a pot fraction (0.0–N) to the closest legal DiscreteAction.

    Buckets:
        pct < 0.5      → BET_33
        0.5 ≤ pct < 0.85 → BET_66
        0.85 ≤ pct < 1.5 → BET_100
        1.5 ≤ pct < 0.9*effective_stack/pot → BET_200
        pct ≥ that → ALLIN
    """
    if pct >= 0.90 * (view.effective_stack / max(1, view.pot)):
        target = DiscreteAction.ALLIN
    elif pct < 0.50:
        target = DiscreteAction.BET_33
    elif pct < 0.85:
        target = DiscreteAction.BET_66
    elif pct < 1.50:
        target = DiscreteAction.BET_100
    else:
        target = DiscreteAction.BET_200
    return _nearest_legal_bet(target, legal_set)


# ============================================================
# Policy class
# ============================================================

class ShankyProfilePolicy:
    """Adapter that wraps a Shanky bot profile as a Policy.

    Conforms to the `Policy` protocol used by eval_pool.py:
        name: str
        select_action(parsed, state, rng, mode) -> int

    Profile is loaded once at construction; subsequent select_action calls
    reuse the cached AST.

    Usage:
        policy = ShankyProfilePolicy(
            name="gushansenmtt",
            profile_path="/path/to/GusHansenMTT.txt",
        )
        # Then use exactly like CheckpointPolicy / UniformRandomPolicy
        # in eval_pool.py:
        eval_pool.play_one_hand_two_policies(
            policy_a=policy, policy_b=other, ...
        )
    """

    def __init__(
        self,
        name: str,
        profile_path: str,
        big_blind_chips: int = 100,
        fallback_action: DiscreteAction = DiscreteAction.FOLD,
    ):
        """Construct from a profile file on disk.

        Args:
            name: short identifier for logging and eval-result attribution.
            profile_path: path to a .txt or .ppl profile file (UTF-8 text).
            big_blind_chips: chips per big blind in the games this policy will
                play. Defaults to 100 (OpenSpiel universal_poker default).
            fallback_action: which DiscreteAction to default to when the
                profile returns None (no rule matched) or when the chosen
                action isn't legal. Defaults to FOLD.
        """
        self.name = name
        self.profile_path = profile_path
        self.big_blind_chips = big_blind_chips
        self.fallback_action = fallback_action
        with open(profile_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        self.profile: Profile = parse_profile(source, source_name=name)

    def select_action(
        self,
        parsed: dict,
        state: Any,
        rng: random.Random,
        mode: str = "sample",
    ) -> int:
        """Pick a chip-action for the current decision.

        Returns an OpenSpiel chip-action int from `state.legal_actions()`.
        """
        # Lazy import to avoid pulling in the full ML stack (treys, torch, etc.)
        # at module import time. _build_view_6max only needs OpenSpiel state.
        from src.nlhe.cfr6 import _build_view_6max

        # Build view + discretization
        legal_chip = list(state.legal_actions())
        view = _build_view_6max(state, parsed)
        discrete_to_chip = discretize_legal_actions(legal_chip, view)

        if not discrete_to_chip:
            # No discrete actions available; fall back to any legal chip action.
            return rng.choice(legal_chip)

        # Build context and evaluate profile
        ctx = build_game_context(parsed, state, big_blind_chips=self.big_blind_chips)
        # Seed the random predicate via the RNG so multiple evaluations
        # within the same hand can still produce varied randomness.
        ctx.rng_seed = rng.randint(0, 2 ** 31 - 1)

        shanky_action = evaluate_profile(self.profile, ctx)

        # Translate to DiscreteAction
        da = shanky_action_to_discrete(shanky_action, discrete_to_chip, view)

        # Fallback chain
        if da is None or da not in discrete_to_chip:
            # First fallback: configured default if legal
            if self.fallback_action in discrete_to_chip:
                da = self.fallback_action
            elif DiscreteAction.CALL in discrete_to_chip:
                # Second fallback: call (check if free)
                da = DiscreteAction.CALL
            elif DiscreteAction.FOLD in discrete_to_chip:
                da = DiscreteAction.FOLD
            else:
                # Last resort: pick any legal discrete
                da = next(iter(discrete_to_chip.keys()))

        chip_action = discrete_to_chip.get(da)
        if chip_action is None:
            # Belt-and-suspenders fallback
            chip_action = rng.choice(list(discrete_to_chip.values()))
        return int(chip_action)


# ============================================================
# Bulk loader convenience
# ============================================================

def load_shanky_policies_from_dir(
    directory: str,
    big_blind_chips: int = 100,
    fallback_action: DiscreteAction = DiscreteAction.FOLD,
    suffix: str = ".txt",
) -> list[ShankyProfilePolicy]:
    """Load every .txt profile in a directory as a ShankyProfilePolicy.

    Convenience for league training opponent-pool population.

    Args:
        directory: path containing profile files.
        big_blind_chips: chips per BB to pass to each policy.
        fallback_action: default action for each policy.
        suffix: file extension to load (default ".txt"; .ppl files are
            encrypted binary and not supported).

    Returns:
        List of ShankyProfilePolicy instances, one per matching file. The
        profile name is derived from the filename (stem, lowercased).
    """
    import os
    out = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(suffix):
            continue
        path = os.path.join(directory, fname)
        name = os.path.splitext(fname)[0].lower()
        try:
            policy = ShankyProfilePolicy(
                name=name,
                profile_path=path,
                big_blind_chips=big_blind_chips,
                fallback_action=fallback_action,
            )
            out.append(policy)
        except Exception as e:
            log.warning(f"Failed to load {fname}: {e}")
            continue
    return out
