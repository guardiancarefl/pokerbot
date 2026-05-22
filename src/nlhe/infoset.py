"""Custom information state encoder for HUNL Deep CFR.

Converts OpenSpiel universal_poker states into the feature vector our policy
and advantage networks consume.

Encoding (default config, k=200 postflop):
  - bucket one-hot (200 dims — sized to max k across streets, padded with 0)
  - street one-hot (4: preflop, flop, turn, river)
  - position one-hot (2: button/SB, BB)
  - pot / starting_stack (1)
  - to_call / starting_stack (1)
  - effective_stack / starting_stack (1)
  - betting history features (5: bets-this-street, raises-this-street, last-bet-size-frac, n-actions-this-street, is-facing-bet)
  Total: ~214 dims with k=200 postflop.

Caching: bucket_of() is the expensive primitive (~60ms per call). Within a single
traversal we hit the same (hero, board) many times — once per decision node and
again on backups. The encoder maintains a per-traversal cache keyed by
(hero_frozenset, board_tuple) to amortize that cost.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.nlhe.abstraction import Abstraction


# ----- Parsing OpenSpiel state -----

def _parse_universal_poker_state(state: Any) -> dict:
    """Extract HUNL-relevant fields from a universal_poker state.

    OpenSpiel's information_state_string() returns something like:
      "[Round 0][Player: 1][Pot: 200][Money: 19900 19950][Private: KcTs]
       [Public: ][Sequences: ]"

    We parse it rather than reach into the C++ object's internals.
    """
    s = state.information_state_string(state.current_player())
    out = {}

    # Round (street): 0=preflop, 1=flop, 2=turn, 3=river
    import re
    m = re.search(r"\[Round (\d+)\]", s)
    out["street_idx"] = int(m.group(1)) if m else 0

    m = re.search(r"\[Player: (\d+)\]", s)
    out["player"] = int(m.group(1)) if m else 0

    m = re.search(r"\[Pot: (\d+)\]", s)
    out["pot"] = int(m.group(1)) if m else 0

    m = re.search(r"\[Money:\s+(\d+)\s+(\d+)\]", s)
    if m:
        out["money"] = [int(m.group(1)), int(m.group(2))]
    else:
        out["money"] = [0, 0]

    m = re.search(r"\[Private:\s+([\w]+)\]", s)
    out["private"] = m.group(1) if m else ""

    m = re.search(r"\[Public:\s+([\w]*)\]", s)
    out["public"] = m.group(1) if m else ""

    m = re.search(r"\[Sequences:\s+([^\]]*)\]", s)
    out["sequences"] = m.group(1).strip() if m else ""

    return out


# ----- Betting history features -----

def _betting_features(sequences: str, pot: int, starting_stack: int) -> dict:
    """Compute betting-history features from the sequences string.

    OpenSpiel encodes the betting history as a string like 'r400c/r800f' where
    'r<X>' = raise to X, 'c' = call/check, 'f' = fold, '/' = street boundary.
    """
    # Get just the current street's segment (everything after the last '/').
    if "/" in sequences:
        this_street = sequences.rsplit("/", 1)[1]
    else:
        this_street = sequences

    # Count actions and raises this street.
    raises_this_street = this_street.count("r")
    actions_this_street = (
        this_street.count("r") + this_street.count("c") + this_street.count("f")
    )

    # Last bet size: extract the last 'rN' if any.
    last_bet_size_frac = 0.0
    import re
    raise_matches = re.findall(r"r(\d+)", this_street)
    if raise_matches:
        last_bet_size = int(raise_matches[-1])
        last_bet_size_frac = min(1.0, last_bet_size / max(starting_stack, 1))

    is_facing_bet = 1.0 if raise_matches else 0.0

    return {
        "raises_this_street": raises_this_street,
        "actions_this_street": actions_this_street,
        "last_bet_size_frac": last_bet_size_frac,
        "is_facing_bet": is_facing_bet,
    }


# ----- Encoder -----

@dataclass
class InfosetEncoder:
    """Encodes an OpenSpiel HUNL state into a flat feature vector.

    Args:
        abstraction: trained Abstraction artifact for bucket lookup.
        max_buckets: width of the bucket one-hot region. Should be >= max k
            across streets (default 200).
        starting_stack: chips each player started with, used to normalize.
        bucket_runouts: number of MC trials per bucket_of() call.
    """
    abstraction: Abstraction
    max_buckets: int = 200
    starting_stack: int = 20000
    bucket_runouts: int = 100  # smaller than abstraction training MC since it's hot path

    # Per-traversal cache for (hero, board) -> bucket lookups.
    _bucket_cache: dict = field(default_factory=dict)

    def reset_cache(self) -> None:
        """Call at the start of each traversal."""
        self._bucket_cache.clear()

    @property
    def feature_dim(self) -> int:
        # bucket one-hot + street(4) + position(2) + pot/tocall/effstack(3) + betting(4 features below + 1 = 5)
        return self.max_buckets + 4 + 2 + 3 + 5

    def encode(self, state: Any, rng: random.Random | None = None) -> np.ndarray:
        """Encode the current state from the current player's perspective."""
        parsed = _parse_universal_poker_state(state)

        vec = np.zeros(self.feature_dim, dtype=np.float32)
        idx = 0

        # Bucket one-hot.
        hero_str = parsed["private"]
        board_str = parsed["public"]
        bucket = self._get_bucket(hero_str, board_str, parsed["street_idx"], rng)
        if 0 <= bucket < self.max_buckets:
            vec[bucket] = 1.0
        idx += self.max_buckets

        # Street one-hot.
        street = max(0, min(3, parsed["street_idx"]))
        vec[idx + street] = 1.0
        idx += 4

        # Position one-hot (player 0 or 1).
        player = max(0, min(1, parsed["player"]))
        vec[idx + player] = 1.0
        idx += 2

        # Pot / to_call / effective_stack, normalized.
        pot = parsed["pot"]
        money = parsed["money"]
        my_money = money[player] if player < len(money) else 0
        opp_money = money[1 - player] if (1 - player) < len(money) else 0
        # The committed-this-hand for each player is starting_stack - money[i].
        my_committed = self.starting_stack - my_money
        opp_committed = self.starting_stack - opp_money
        to_call = max(0, opp_committed - my_committed)
        effective_stack = min(my_money, opp_money)
        vec[idx] = min(1.0, pot / max(self.starting_stack * 2, 1))
        vec[idx + 1] = min(1.0, to_call / max(self.starting_stack, 1))
        vec[idx + 2] = min(1.0, effective_stack / max(self.starting_stack, 1))
        idx += 3

        # Betting features.
        bf = _betting_features(parsed["sequences"], pot, self.starting_stack)
        vec[idx] = min(1.0, bf["raises_this_street"] / 4.0)  # rare to have >4 raises
        vec[idx + 1] = min(1.0, bf["actions_this_street"] / 8.0)
        vec[idx + 2] = bf["last_bet_size_frac"]
        vec[idx + 3] = bf["is_facing_bet"]
        vec[idx + 4] = 1.0 if parsed["sequences"] else 0.0  # any action yet this hand
        idx += 5

        assert idx == self.feature_dim, f"idx={idx} != feature_dim={self.feature_dim}"
        return vec

    def _get_bucket(
        self,
        hero_str: str,
        board_str: str,
        street_idx: int,
        rng: random.Random | None,
    ) -> int:
        """Look up the bucket for (hero, board), with caching."""
        # Cache key: canonicalize hero (order-insensitive) + board (order matters).
        # In treys-int form ordering matters; in card-string form we sort.
        hero_key = "".join(sorted(hero_str[i:i+2] for i in range(0, len(hero_str), 2)))
        cache_key = (hero_key, board_str, street_idx)
        if cache_key in self._bucket_cache:
            return self._bucket_cache[cache_key]

        # Convert to treys-int format.
        from src.nlhe.equity import cards_from_str
        hero_cards = cards_from_str(hero_str)
        board_cards = cards_from_str(board_str) if board_str else []

        # Defensive: if street doesn't match board length, fall back to street-implied size.
        expected_board_len = {0: 0, 1: 3, 2: 4, 3: 5}[street_idx]
        if len(board_cards) != expected_board_len:
            # Skip MC and assign bucket 0 if state is inconsistent.
            self._bucket_cache[cache_key] = 0
            return 0

        bucket = self.abstraction.bucket_of(
            hero_cards, board_cards,
            runouts=self.bucket_runouts,
            rng=rng,
        )
        self._bucket_cache[cache_key] = bucket
        return bucket
