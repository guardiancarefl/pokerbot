"""HTTP client for the Slumbot 2017 API.

Slumbot is heads-up no-limit Texas hold'em at 200bb (stack 20000, blinds 50/100).
Public benchmark used in academic Deep CFR literature.

API endpoints:
    POST /api/new_hand    -> start a new hand
    POST /api/act         -> submit an action ('c', 'f', or 'b<N>')

The Slumbot server auto-advances chance nodes (deals) and Slumbot's own
decisions, returning to the client only when it's the client's turn to act
or the hand has ended.

Action language (in the action string):
    c       check or call
    f       fold
    b<N>    bet/raise to N total chips
    /       street boundary (returned after a street completes)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

log = logging.getLogger("slumbot_client")

BASE_URL = "https://slumbot.com/api"
TIMEOUT_S = 10
SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 20000  # chips per player at hand start (200bb)


@dataclass
class SlumbotState:
    """Parsed Slumbot response. Mirrors what the API returns plus a couple of
    derived booleans for terminal detection."""
    token: str
    client_pos: int                       # 0 or 1
    hole_cards: list[str]                 # ["As", "Kh"]
    board: list[str]                      # ["Jd", "9h", "2s"] (0/3/4/5 cards)
    action: str                           # full sequence: "b200c/b300c/"
    old_action: str                       # sequence before the most recent client action
    raw: dict = field(default_factory=dict)  # full server response for debugging

    # Terminal-hand fields (only populated when the hand has ended)
    winnings: Optional[int] = None
    bot_hole_cards: Optional[list[str]] = None
    won_pot: Optional[int] = None
    # Slumbot's built-in variance reduction signal. The server reports the
    # difference between our action's EV and a baseline policy's EV at every
    # decision point, summed for the hand. session_baseline_total / num_hands
    # gives a much lower-variance estimate of our edge than raw winnings.
    baseline_winnings: Optional[float] = None
    session_num_hands: Optional[int] = None
    session_total: Optional[int] = None
    session_baseline_total: Optional[float] = None

    @property
    def is_terminal(self) -> bool:
        return self.winnings is not None


class SlumbotClient:
    """Stateful client for one Slumbot session.

    Usage:
        client = SlumbotClient()
        state = client.new_hand()
        while not state.is_terminal:
            action = my_policy(state)
            state = client.act(action)
        # state.winnings is the client's win/loss in chips for this hand
    """

    def __init__(self, base_url: str = BASE_URL, timeout_s: float = TIMEOUT_S):
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.token: Optional[str] = None

    # ----- Low-level HTTP -----

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.error(f"Slumbot {endpoint} failed: {e}")
            raise

    def _parse(self, resp: dict) -> SlumbotState:
        # Slumbot returns error responses with an "error_msg" key and otherwise
        # empty fields. Detect this and surface it as an exception rather than
        # silently producing a bogus state.
        if "error_msg" in resp:
            raise RuntimeError(f"Slumbot API error: {resp['error_msg']}")
        st = SlumbotState(
            token=resp.get("token", self.token or ""),
            client_pos=int(resp.get("client_pos", 0)),
            hole_cards=list(resp.get("hole_cards", [])),
            board=list(resp.get("board", [])),
            action=resp.get("action", ""),
            old_action=resp.get("old_action", ""),
            raw=resp,
            winnings=resp.get("winnings"),  # None if mid-hand
            bot_hole_cards=resp.get("bot_hole_cards"),
            won_pot=resp.get("won_pot"),
            baseline_winnings=resp.get("baseline_winnings"),
            session_num_hands=resp.get("session_num_hands"),
            session_total=resp.get("session_total"),
            session_baseline_total=resp.get("session_baseline_total"),
        )
        if st.token != self.token and st.token:
            self.token = st.token
        return st

    # ----- Public API -----

    def new_hand(self) -> SlumbotState:
        resp = self._post("new_hand", {"token": self.token or ""})
        return self._parse(resp)

    def act(self, incr: str) -> SlumbotState:
        if self.token is None:
            raise RuntimeError("must call new_hand() before act()")
        resp = self._post("act", {"token": self.token, "incr": incr})
        return self._parse(resp)


# ----- Action sequence parsing -----

def parse_action_sequence(action_str: str) -> list[list[str]]:
    """Parse a Slumbot action string into a list-of-lists by street.

    Examples:
        ""                  -> [[]]                       (no actions yet, preflop)
        "b200c/"            -> [["b200", "c"]]            (preflop done, on flop)
        "b200c/b300c/"      -> [["b200", "c"], ["b300", "c"]]   (preflop and flop done, on turn)
        "b200c/b300b900f"   -> [["b200", "c"], ["b300", "b900", "f"]]
    """
    # Split into street segments. Trailing '/' produces an empty final segment we drop.
    segments = action_str.split("/")
    if segments and segments[-1] == "":
        segments = segments[:-1]

    streets = []
    for seg in segments:
        actions = []
        i = 0
        while i < len(seg):
            ch = seg[i]
            if ch in ("c", "f", "k"):  # 'k' = check, but Slumbot uses 'c' uniformly
                actions.append(ch)
                i += 1
            elif ch == "b":
                # Read 'b' followed by digits
                m = re.match(r"b(\d+)", seg[i:])
                if not m:
                    raise ValueError(f"malformed bet action at offset {i}: {seg!r}")
                actions.append(m.group(0))
                i += len(m.group(0))
            else:
                raise ValueError(f"unrecognized char {ch!r} at offset {i} in segment {seg!r}")
        streets.append(actions)
    return streets


def current_street_idx(action_str: str, board_size: int) -> int:
    """Return 0=preflop, 1=flop, 2=turn, 3=river based on action and board state.

    The board size is the authoritative signal — Slumbot returns 0/3/4/5 board cards.
    """
    return {0: 0, 3: 1, 4: 2, 5: 3}.get(board_size, 0)


# ----- Convenience: a no-op random-policy client -----

class RandomPolicy:
    """Uniform random legal action. Used to validate the eval harness end-to-end
    before plugging in a trained policy. Expected bb/100: ~-200 (random play
    loses to a competent opponent)."""

    def __init__(self, seed: int = 0):
        import random
        self.rng = random.Random(seed)

    def choose_action(self, state: SlumbotState) -> str:
        """Return a Slumbot action string ('c', 'f', or 'b<N>').

        For a random policy we choose between {fold, call, min-bet, allin}
        with uniform probability over those that are legal-ish (fold only
        legal when facing a bet).
        """
        streets = parse_action_sequence(state.action)
        current = streets[-1] if streets else []
        # Facing a bet means the LAST action this street was a bet/raise that
        # the actor hasn't yet responded to. A 'b' earlier in the sequence
        # that's been called (e.g., 'b400c') is closed — we are not facing it.
        #
        # Special case: on PREFLOP with no recorded actions, the SB/button is
        # facing the unposted big blind. Slumbot doesn't include the blind in
        # the action string, but the BB is functionally an unmatched bet of
        # 100 chips. So we must call/raise/fold, not check.
        is_preflop = len(streets) <= 1
        if not current and is_preflop:
            facing_bet = True
        else:
            facing_bet = bool(current) and current[-1].startswith("b")

        # Choose uniformly among a small action set.
        # Slumbot protocol: 'k' = check (no bet to match), 'c' = call (match bet),
        # 'f' = fold, 'b<N>' = bet/raise to N. Check and fold are illegal when
        # not facing a bet; call is illegal when there's nothing to call.
        if facing_bet:
            options = ["c", "f"]
        else:
            options = ["k"]
        # Always include a min-bet/raise option.
        # For a min-raise after a bet: raise to 2x the last bet.
        last_bet = None
        for a in reversed(current):
            if a.startswith("b"):
                last_bet = int(a[1:])
                break
        if last_bet is not None:
            min_raise = min(STARTING_STACK, last_bet * 2)
            options.append(f"b{min_raise}")
        else:
            # No bet yet on this street; we open at 2x BB = 200 (preflop) or pot-like.
            options.append("b200")
        # Note: we intentionally skip all-in for the random policy because computing
        # the legal max bet from the action string is fiddly and not worth doing here.
        # The random policy is only for harness validation; trained policies will
        # use proper legal-action computation.

        return self.rng.choice(options)
