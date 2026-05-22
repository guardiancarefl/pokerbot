"""PolicyAdapter: trained DeepCFRSolver -> Slumbot wire-token policy.

choose_action(SlumbotState) -> str reconstructs an OpenSpiel state by
replaying the hand from new_initial_state() (forcing hero's hole cards,
dealing arbitrary opponent hole cards, forcing board cards, walking the
Slumbot action history), encodes the resulting infoset via the same
InfosetEncoder used during training, runs the strategy network for the
hero seat, samples a DiscreteAction over legal actions, and translates
that back to a Slumbot wire token.

Stack mismatch: the checkpoint stores its train-time starting_stack in
config_dict; the OpenSpiel game_str specifies the in-play stack. Both
are exposed as instance attributes (train_starting_stack,
game_starting_stack). When they disagree the network sees out-of-
distribution normalized features and its output is meaningless; the
adapter still routes a syntactically valid action so state translation
can be plumbing-tested independently of strategy quality.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import pyspiel
import torch

from src.nlhe.abstraction import Abstraction
from src.nlhe.actions import (
    DiscreteAction,
    discretize_legal_actions,
    policy_to_game_action,
)
from src.nlhe.slumbot_client import SlumbotState
from src.nlhe.solver import DeepCFRSolver, TrainConfig, _build_game_state_view


_DEAL_RE = re.compile(r"\bDeal\s+([2-9TJQKA][cdhs])\b")
_CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
_PRIVATE_RE = re.compile(r"\[Private:\s+([^\]]*)\]")
_PUBLIC_RE = re.compile(r"\[Public:\s+([^\]]*)\]")
_BET_TOKEN_RE = re.compile(r"^b(\d+)$")


# ----- Top-level helpers (unit-testable) -----

def _extract_card_from_action_string(s: str) -> str:
    """Pull the '2c' out of 'player=-1 move=Deal 2c'.

    Raises ValueError if the string doesn't contain a 'Deal <card>' fragment.
    """
    m = _DEAL_RE.search(s)
    if m is None:
        raise ValueError(f"no 'Deal <card>' found in action string: {s!r}")
    return m.group(1)


def pick_deck_action(
    state: Any,
    predicate: Callable[[str], bool],
    purpose: str,
) -> int:
    """Scan a chance node's legal actions; return the first int whose card
    satisfies predicate. Raises with `purpose` if nothing matches.
    """
    legal = state.legal_actions()
    available: list[str] = []
    for a in legal:
        try:
            card = _extract_card_from_action_string(state.action_to_string(a))
        except ValueError:
            continue
        available.append(card)
        if predicate(card):
            return int(a)
    preview = ", ".join(available[:8]) + (" ..." if len(available) > 8 else "")
    raise RuntimeError(
        f"pick_deck_action({purpose}): no legal card satisfied predicate. "
        f"Available in deck ({len(available)} cards): [{preview}]"
    )


def slumbot_token_to_openspiel_action(
    token: str,
    prior_streets_committed_by_actor: int = 0,
) -> int:
    """Translate a Slumbot wire token to an OpenSpiel chip-action int.

    Mapping:
        'f'      -> 0  (fold)
        'c'/'k'  -> 1  (call/check)
        'b<N>'   -> N + prior_streets_committed_by_actor

    Slumbot's N is chips committed by the actor on the CURRENT STREET.
    OpenSpiel's bet integer is chips committed by the actor across the
    WHOLE HAND. Default 0 keeps preflop callers correct.
    """
    if not isinstance(token, str) or not token:
        raise ValueError(f"empty or non-string Slumbot token: {token!r}")
    if token == "f":
        return 0
    if token == "c" or token == "k":
        return 1
    m = _BET_TOKEN_RE.match(token)
    if m is None:
        raise ValueError(f"unrecognized Slumbot token: {token!r}")
    n = int(m.group(1))
    if n < 2:
        # OpenSpiel reserves 0/1 for fold/call. A bet-to-N must be >= 2.
        raise ValueError(f"bet target must be >= 2 chips, got {token!r}")
    return n + int(prior_streets_committed_by_actor)


def openspiel_action_to_slumbot_token(
    action: int,
    state: Any,
    prior_streets_committed_by_actor: int = 0,
) -> str:
    """Inverse of slumbot_token_to_openspiel_action.

    For action=1, returns 'c' if 0 is in state.legal_actions() (i.e., we're
    facing a bet so call is the live semantics) otherwise 'k' (free check).
    For bets, subtracts the actor's prior-streets commitment so the wire
    token reflects chips committed on the current street only.
    """
    if action == 0:
        return "f"
    if action == 1:
        return "c" if 0 in state.legal_actions() else "k"
    if action >= 2:
        return f"b{int(action) - int(prior_streets_committed_by_actor)}"
    raise ValueError(f"unrepresentable OpenSpiel action: {action!r}")


_MONEY_RE = re.compile(r"\[Money:\s+(\d+)\s+(\d+)\]")


def _refresh_prior_streets_committed(state: Any, starting_stack: int) -> dict[int, int]:
    """Snapshot each player's total committed chips as of the current node.

    Called at the start of a postflop street (after the board cards are
    dealt, before any betting on that street). At that moment, no chips
    have entered the pot on the current street yet, so the snapshot is
    exactly 'committed across prior streets'.
    """
    info = state.information_state_string(0)
    m = _MONEY_RE.search(info)
    if not m:
        raise RuntimeError(f"could not parse [Money: X Y] from info_state: {info!r}")
    return {
        0: int(starting_stack) - int(m.group(1)),
        1: int(starting_stack) - int(m.group(2)),
    }


# ----- Internal helpers (used by PolicyAdapter only) -----

def _parse_action_tokens(action_str: str) -> list[str]:
    """Flatten a Slumbot action string ('b200c/b300c/') into a token list.

    Street boundaries ('/') are dropped — the OpenSpiel state knows what
    street it's on so we don't need them for replay.
    """
    if not action_str:
        return []
    tokens: list[str] = []
    for seg in action_str.split("/"):
        if not seg:
            continue
        i = 0
        while i < len(seg):
            ch = seg[i]
            if ch in ("c", "f", "k"):
                tokens.append(ch)
                i += 1
            elif ch == "b":
                m = re.match(r"b(\d+)", seg[i:])
                if m is None:
                    raise ValueError(f"malformed bet at offset {i} in segment {seg!r}")
                tokens.append(m.group(0))
                i += len(m.group(0))
            else:
                raise ValueError(f"unrecognized char {ch!r} at offset {i} in segment {seg!r}")
    return tokens


def _private_for(state: Any, player: int) -> str:
    """Read OpenSpiel's [Private: XXXX] field for the given player."""
    m = _PRIVATE_RE.search(state.information_state_string(player))
    return m.group(1) if m else ""


def _public_field(state: Any) -> str:
    """Read OpenSpiel's [Public: XXXX] field (public info; either seat works)."""
    m = _PUBLIC_RE.search(state.information_state_string(0))
    return m.group(1) if m else ""


def _cards_in_private(priv: str) -> list[str]:
    """Split a concatenated private string ('2d2c') into ['2d', '2c']."""
    return [priv[i:i + 2] for i in range(0, len(priv), 2)]


class PolicyAdapter:
    """Wraps a trained DeepCFRSolver into a Slumbot-compatible policy.

    Attributes:
        train_starting_stack: starting_stack from the checkpoint's config_dict.
        game_starting_stack: stack=N parsed from the OpenSpiel game_str.
        mode: 'sample' (categorical) or 'argmax' (deterministic).
    """

    def __init__(
        self,
        checkpoint_path: str,
        abstraction_path: str,
        game_str: str,
        mode: str = "sample",
        seed: int = 2026,
    ):
        if mode not in ("sample", "argmax"):
            raise ValueError(f"mode must be 'sample' or 'argmax', got {mode!r}")
        self.mode = mode
        self.game_str = game_str

        # Load checkpoint to discover its train-time starting_stack.
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        if "config_dict" not in ckpt:
            raise ValueError(
                f"checkpoint {checkpoint_path} missing config_dict; cannot "
                f"determine train_starting_stack."
            )
        train_cfg_dict = dict(ckpt["config_dict"])
        if "starting_stack" not in train_cfg_dict:
            raise ValueError(
                f"checkpoint config_dict missing starting_stack: keys="
                f"{list(train_cfg_dict.keys())}"
            )
        self.train_starting_stack = int(train_cfg_dict["starting_stack"])

        # Parse the game_str's stack value.
        m = re.search(r"stack=(\d+)\s+(\d+)", game_str)
        if m is None:
            raise ValueError(f"could not parse stack=N N from game_str: {game_str!r}")
        s0, s1 = int(m.group(1)), int(m.group(2))
        if s0 != s1:
            raise ValueError(f"asymmetric stacks in game_str not supported: {s0} vs {s1}")
        self.game_starting_stack = s0

        # Load abstraction and OpenSpiel game.
        self.abstraction = Abstraction.load(abstraction_path)
        self.game = pyspiel.load_game(game_str)

        # Build solver for inference, load weights.
        tc = TrainConfig(**train_cfg_dict)
        self.solver = DeepCFRSolver(
            self.game, self.abstraction, tc, logger=lambda _msg: None,
        )
        self.solver.load_checkpoint(checkpoint_path)
        for net in self.solver.strat_nets:
            net.eval()
        self.encoder = self.solver.encoder

        self.rng = np.random.default_rng(seed)

    # ----- Public entry point -----

    def choose_action(self, state: SlumbotState) -> str:
        # Entry validation — these are loud failures so callers can't pass us
        # mid-flight corrupted state without noticing.
        if len(state.hole_cards) != 2:
            raise AssertionError(
                f"PolicyAdapter: expected exactly 2 hole cards, got "
                f"{state.hole_cards!r}"
            )
        for c in state.hole_cards:
            if not _CARD_RE.match(c):
                raise AssertionError(f"PolicyAdapter: malformed hole card {c!r}")
        if len(state.board) not in (0, 3, 4, 5):
            raise AssertionError(
                f"PolicyAdapter: board length {len(state.board)} not in {{0,3,4,5}}; "
                f"board={state.board}"
            )
        for c in state.board:
            if not _CARD_RE.match(c):
                raise AssertionError(f"PolicyAdapter: malformed board card {c!r}")

        # Reconstruct via replay.
        action_tokens = _parse_action_tokens(state.action)
        os_state = self.game.new_initial_state()
        try:
            prior_streets_committed = self._replay(
                os_state,
                hero_seat=state.client_pos,
                hero_cards=list(state.hole_cards),
                target_board=list(state.board),
                action_tokens=action_tokens,
            )
        except (RuntimeError, AssertionError, ValueError) as e:
            raise RuntimeError(
                "PolicyAdapter replay failed.\n"
                f"  SlumbotState:\n"
                f"    client_pos={state.client_pos}\n"
                f"    hole_cards={state.hole_cards}\n"
                f"    board={state.board}\n"
                f"    action={state.action!r}\n"
                f"  action_tokens={action_tokens}\n"
                f"  underlying: {type(e).__name__}: {e}"
            ) from e

        if os_state.current_player() != state.client_pos:
            raise RuntimeError(
                "PolicyAdapter: after replay, current_player mismatches client_pos.\n"
                f"  os_state.current_player()={os_state.current_player()}\n"
                f"  state.client_pos={state.client_pos}\n"
                f"  hole_cards={state.hole_cards}  board={state.board}\n"
                f"  action={state.action!r}\n"
                f"  os info-state (hero view):\n"
                f"    {os_state.information_state_string(state.client_pos)}"
            )

        # Encode + forward.
        self.encoder.reset_cache()
        feat = self.encoder.encode(os_state)
        view = _build_game_state_view(os_state, self.train_starting_stack)
        legal_chip = os_state.legal_actions()
        discrete_to_chip = discretize_legal_actions(legal_chip, view)
        hero_prior = prior_streets_committed[state.client_pos]
        if not discrete_to_chip:
            legal_tokens = [
                openspiel_action_to_slumbot_token(
                    a, os_state, prior_streets_committed_by_actor=hero_prior,
                )
                for a in legal_chip
            ]
            raise RuntimeError(
                "PolicyAdapter: no legal discrete actions at replayed state.\n"
                f"  legal OpenSpiel actions: {legal_chip}\n"
                f"  legal Slumbot tokens: {legal_tokens}\n"
                f"  view: {view}"
            )

        legal_mask = np.zeros(len(DiscreteAction), dtype=np.float32)
        for da in discrete_to_chip:
            legal_mask[int(da)] = 1.0

        hero_seat = state.client_pos
        with torch.no_grad():
            logits = (
                self.solver.strat_nets[hero_seat](
                    torch.from_numpy(feat).float().unsqueeze(0).to(self.solver.device)
                )
                .numpy()[0]
            )
        logits = logits - logits.max()
        exp_l = np.exp(logits) * legal_mask
        denom = float(exp_l.sum())
        if denom > 0:
            probs = exp_l / denom
        else:
            # Network output collapsed to -inf on all legal actions; fall back
            # to uniform over legal. This shouldn't crash; just routes one move.
            probs = legal_mask / float(legal_mask.sum())

        if self.mode == "argmax":
            chosen_idx = int(np.argmax(probs))
        else:
            chosen_idx = int(self.rng.choice(len(probs), p=probs))
        da = DiscreteAction(chosen_idx)
        if da not in discrete_to_chip:
            # Defensive: if masking somehow let an illegal idx through, fall
            # back to any legal discrete action.
            da = next(iter(discrete_to_chip))

        chip = policy_to_game_action(da, view)
        if chip is None:
            # discretize_legal_actions said this was legal; policy_to_game_action
            # said no. Bail to the first legal option.
            da, chip = next(iter(discrete_to_chip.items()))
        return openspiel_action_to_slumbot_token(
            int(chip), os_state, prior_streets_committed_by_actor=hero_prior,
        )

    # ----- Replay -----

    def _replay(
        self,
        state: Any,
        hero_seat: int,
        hero_cards: list[str],
        target_board: list[str],
        action_tokens: list[str],
    ) -> dict[int, int]:
        """Walk from new_initial_state to the current decision point.

        Returns the per-player prior_streets_committed snapshot valid at
        the decision point (so callers can translate back to Slumbot wire
        format correctly).
        """
        token_idx = 0
        prior_streets_committed: dict[int, int] = {0: 0, 1: 0}
        last_refresh_board_count = 0
        # Safety cap: hole dealing (4) + max board (5) + max plausible action
        # sequence (~30 in HUNL). 200 is wildly conservative.
        for _step in range(200):
            if state.is_terminal():
                raise RuntimeError(
                    f"hand became terminal during replay at token_idx={token_idx} "
                    f"of {len(action_tokens)}"
                )
            if state.is_chance_node():
                self._deal_one_card(state, hero_seat, hero_cards, target_board)
                continue
            # Decision node. If we just finished dealing a postflop street's
            # board cards, snapshot prior-streets commitments before applying
            # any betting on the new street.
            pub = _public_field(state)
            board_count = len(pub) // 2
            if board_count >= 3 and board_count != last_refresh_board_count:
                prior_streets_committed = _refresh_prior_streets_committed(
                    state, self.game_starting_stack
                )
                last_refresh_board_count = board_count
            if token_idx >= len(action_tokens):
                return prior_streets_committed  # decision point reached
            tok = action_tokens[token_idx]
            actor = state.current_player()
            os_action = slumbot_token_to_openspiel_action(
                tok, prior_streets_committed_by_actor=prior_streets_committed[actor],
            )
            legal = state.legal_actions()
            if os_action not in legal:
                legal_tokens = [
                    openspiel_action_to_slumbot_token(
                        a, state,
                        prior_streets_committed_by_actor=prior_streets_committed[actor],
                    )
                    for a in legal
                ]
                raise RuntimeError(
                    f"token {tok!r} -> OpenSpiel {os_action} not in legal_actions.\n"
                    f"  token_idx={token_idx} of {len(action_tokens)}\n"
                    f"  legal OpenSpiel ints: {legal[:20]}{' ...' if len(legal) > 20 else ''}\n"
                    f"  legal Slumbot tokens: {legal_tokens[:20]}{' ...' if len(legal_tokens) > 20 else ''}\n"
                    f"  current_player={actor}\n"
                    f"  prior_streets_committed={prior_streets_committed}\n"
                    f"  info-state: {state.information_state_string(actor)}"
                )
            state.apply_action(int(os_action))
            token_idx += 1
        raise RuntimeError(f"replay exceeded 200 steps; token_idx={token_idx}")

    def _deal_one_card(
        self,
        state: Any,
        hero_seat: int,
        hero_cards: list[str],
        target_board: list[str],
    ) -> None:
        """At a chance node, apply the appropriate card-dealing action.

        Universal_poker (verified empirically) deals hole cards in this order:
        p0's first card, p0's second card, p1's first card, p1's second card.
        Then board cards one at a time at street boundaries.
        """
        p0_priv = _private_for(state, 0)
        p1_priv = _private_for(state, 1)
        pub = _public_field(state)
        p0_count = len(p0_priv) // 2
        p1_count = len(p1_priv) // 2
        board_count = len(pub) // 2

        if p0_count < 2:
            target_player = 0
        elif p1_count < 2:
            target_player = 1
        else:
            target_player = None  # board card

        if target_player is not None:
            if target_player == hero_seat:
                # Deal next hero hole card. Pick the first hero target not yet placed.
                already = set(_cards_in_private(_private_for(state, hero_seat)))
                remaining = [c for c in hero_cards if c not in already]
                if not remaining:
                    raise RuntimeError(
                        f"hero hole-card phase but all hero cards already dealt: "
                        f"hero_priv_now={_private_for(state, hero_seat)!r}, "
                        f"hero_cards={hero_cards}"
                    )
                target_card = remaining[0]
                action = pick_deck_action(
                    state,
                    lambda c, t=target_card: c == t,
                    f"hero (seat={hero_seat}) hole card {target_card!r}",
                )
            else:
                # Opponent hole card — pick any card NOT in our hero targets
                # or board targets, so we don't accidentally consume one of
                # those before its turn.
                forbidden = set(hero_cards) | set(target_board)
                action = pick_deck_action(
                    state,
                    lambda c, fb=forbidden: c not in fb,
                    f"opponent (seat={1 - hero_seat}) hole card",
                )
        else:
            # Board card.
            if board_count >= len(target_board):
                raise RuntimeError(
                    f"chance node past hole cards but board already full: "
                    f"board_count={board_count}, target_board={target_board}"
                )
            next_board = target_board[board_count]
            action = pick_deck_action(
                state,
                lambda c, t=next_board: c == t,
                f"board card #{board_count + 1}: {next_board!r}",
            )

        state.apply_action(int(action))
