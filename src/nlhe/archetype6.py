"""6-max adapter: archetype profiles as opponent-override Policies.

Phase 5-B of Scenario 3 step 5. Wraps the game-agnostic HUNL archetype
machinery (`src/nlhe/archetypes.py` — ArchetypeProfile, archetype_policy,
EquityCalibration) as a 6-max Policy that implements the override protocol
used by cfr6.traverse_6max and the league pool:

    select_action(parsed, state, rng, mode) -> int   (an OpenSpiel chip action)

The underlying `archetype_policy` is reused UNMODIFIED (Option 1 — wrap, not
port). It carries no 2-player assumptions: it is purely parametric over
(bucket_id, in_position, pot_odds, stack_to_pot, legal_mask, facing_bet). The
only HUNL-specific inputs — bucket lookup and position derivation — are
computed here in the adapter against the 6-max abstraction and the 6-max
position helper, exactly as HUNL's solver._archetype_strategy computes them
against its own.

Strategy-buffer suppression is automatic: archetype opponents reach the bot's
training loop through the cfr6 NON-traverser short-circuit (cfr6.py:299), which
returns before strat_buffer.add (cfr6.py:381). So archetype decisions are never
written to the strategy buffer — the bot learns its own policy, not the
archetype's (DECISIONS.md:207/216). No explicit skip is needed (unlike HUNL's
solver, which dispatches at the opponent node and must skip the write there).

Calibration note: postflop bucket lookup uses MC runouts (cfg.bucket_runouts),
producing the same small per-query noise the 6-max calibration's postflop
quantiles carry (Phase 5-pre). This is by-design — the archetype framework
groups hands as play/don't-play and is tolerant to noise at this scale
(archetypes.py docstring; DECISIONS.md Phase-5-pre audit entry).

EquityCalibration: load runs/archetype_design/bucket_equity_analysis_6max.json
(commit d093abd), the 6-max-abstraction calibration.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Optional, Sequence

import numpy as np

from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.archetypes import (
    ArchetypeProfile,
    EquityCalibration,
    NAMED_ARCHETYPES,
    N_DISCRETE_ACTIONS,
    archetype_policy,
)
from src.nlhe.equity import cards_from_str
from src.nlhe.infoset6 import POSITIONS_6MAX, position_for_seat_with_dealer

log = logging.getLogger("archetype6")

# Position indices in POSITIONS_6MAX = [UTG, MP, CO, BTN, SB, BB].
_BTN = POSITIONS_6MAX.index("BTN")  # 3 — last to act postflop → in position
_BB = POSITIONS_6MAX.index("BB")    # 5 — last to act preflop  → in position

# Expected board-card count per street (used for the defensive bucket guard,
# mirroring HUNL solver._archetype_strategy).
_EXPECTED_BOARD_LEN = {0: 0, 1: 3, 2: 4, 3: 5}

# Map archetype name string → profile (for config-driven subset selection).
_NAME_TO_PROFILE = {p.name.name: p for p in NAMED_ARCHETYPES}
VALID_ARCHETYPE_NAMES = tuple(_NAME_TO_PROFILE.keys())  # NIT, TAG, LAG, STATION, MANIAC


class ArchetypePolicy:
    """One archetype profile, wrapped as a 6-max override Policy.

    Conforms to the Policy protocol (name + select_action) used by
    cfr6.traverse_6max's override short-circuit, eval_pool, and LeaguePool —
    identical to CheckpointPolicy / ShankyProfilePolicy.
    """

    # Class-level latch so the "missing dealer_seat" warning fires once per
    # process, not once per decision (it would otherwise flood the log).
    _warned_no_dealer = False

    def __init__(
        self,
        profile: ArchetypeProfile,
        abstraction: Any,
        calibration: EquityCalibration,
        bucket_runouts: int = 30,
    ):
        self.profile = profile
        self.abstraction = abstraction
        self.calibration = calibration
        self.bucket_runouts = bucket_runouts
        self.name = f"archetype-{profile.name.name.lower()}"

    def _bucket_id(self, parsed: dict, street_idx: int, rng: random.Random) -> int:
        """Bucket the hero's (hole, board) via the 6-max abstraction.

        Mirrors HUNL solver._archetype_strategy: extract cards from the parsed
        observation, guard the board length, and fall back to bucket 0 on any
        shape mismatch (defensive — matches encoder behavior).
        """
        hero_str = parsed.get("private_cards", "") or ""
        board_str = parsed.get("public_cards", "") or ""
        hero_cards = cards_from_str(hero_str) if hero_str else []
        board_cards = cards_from_str(board_str) if board_str else []
        expected = _EXPECTED_BOARD_LEN.get(street_idx, 0)
        if len(hero_cards) != 2 or len(board_cards) != expected:
            return 0
        return self.abstraction.bucket_of(
            hero_cards, board_cards, runouts=self.bucket_runouts, rng=rng
        )

    def _in_position(self, parsed: dict, street_idx: int) -> bool:
        """Binary in/out-of-position for the acting seat.

        Mirrors HUNL's "acts last on this street" semantics, lifted to 6-max:
        BTN is last to act postflop; BB is last to act preflop. Approximate
        (ignores folded-button edge cases) but appropriate — in_position drives
        only a 10% betting nudge in archetype_policy.

        Defensive fallback: if dealer_seat is absent (a non-tournament-mode
        caller — the legacy single-hand parse omits it), return False and warn
        ONCE so the unexpected invocation surfaces.
        """
        dealer_seat = parsed.get("dealer_seat")
        if dealer_seat is None:
            if not ArchetypePolicy._warned_no_dealer:
                ArchetypePolicy._warned_no_dealer = True
                log.warning(
                    "ArchetypePolicy: parsed dict has no 'dealer_seat' "
                    "(non-tournament-mode caller?); falling back to "
                    "in_position=False. This warning fires once per process."
                )
            return False
        seat = parsed.get("current_player", 0)
        position = position_for_seat_with_dealer(seat, dealer_seat, num_players=6)
        if street_idx > 0:
            return position == _BTN
        return position == _BB

    def select_action(
        self,
        parsed: dict,
        state: Any,
        rng: random.Random,
        mode: str = "sample",
    ) -> int:
        """Pick a chip-action for the current decision (override protocol).

        Returns an OpenSpiel chip-action int from `state.legal_actions()`.
        """
        # Lazy import (matches ShankyProfilePolicy): keeps the ML stack out of
        # the import path for callers that only need the translation helpers.
        from src.nlhe.cfr6 import _build_view_6max

        legal_chip = list(state.legal_actions())
        view = _build_view_6max(state, parsed)
        discrete_to_chip = discretize_legal_actions(legal_chip, view)
        if not discrete_to_chip:
            # No discrete actions mappable; fall back to any legal chip action.
            return rng.choice(legal_chip)

        street_idx = parsed.get("street_idx", 0)
        bucket_id = self._bucket_id(parsed, street_idx, rng)
        in_position = self._in_position(parsed, street_idx)

        facing_bet = view.to_call > 0
        pot_odds = view.to_call / max(view.pot + view.to_call, 1)
        stack_to_pot = view.effective_stack / max(view.pot, 1)

        legal_mask = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
        for da in discrete_to_chip:
            legal_mask[int(da)] = 1.0

        dist = archetype_policy(
            archetype=self.profile,
            calibration=self.calibration,
            street_idx=street_idx,
            bucket_id=bucket_id,
            in_position=in_position,
            pot_odds=pot_odds,
            stack_to_pot=stack_to_pot,
            legal_mask=legal_mask,
            facing_bet=facing_bet,
            rng=rng,
        )

        if mode == "argmax":
            chosen = int(np.argmax(dist))
        else:
            chosen = rng.choices(
                range(N_DISCRETE_ACTIONS), weights=dist.tolist(), k=1
            )[0]

        da = DiscreteAction(chosen)
        chip_action = discrete_to_chip.get(da)
        if chip_action is None:
            # Defensive: dist is masked to legal actions, so the chosen index
            # should always be legal. If not (e.g. a degenerate all-zero dist
            # the archetype fell back to), pick uniformly from the legal map.
            da, chip_action = rng.choice(list(discrete_to_chip.items()))
        return int(chip_action)


class ArchetypePool:
    """Sampling pool over archetype profiles, returning override Policies.

    Mirrors LeaguePool's contract: sample_opponent(rng) -> Policy, with NO
    internal mix gate. The archetype/league/self-play decision is made by the
    three-way roll in DeepCFR6MaxSolver._maybe_sample_league_opponent BEFORE
    this is called — re-checking a mix here would double-roll.
    """

    def __init__(
        self,
        calibration_path: str,
        abstraction: Any,
        profile_names: Optional[Sequence[str]] = None,
        bucket_runouts: int = 30,
    ):
        self.calibration = EquityCalibration.load(calibration_path)
        self.abstraction = abstraction
        self.bucket_runouts = bucket_runouts
        if profile_names is None:
            self.profiles = list(NAMED_ARCHETYPES)
        else:
            # profile_names is validated against VALID_ARCHETYPE_NAMES at
            # config-load time (TrainConfig6Max.__post_init__); guard anyway.
            self.profiles = [_NAME_TO_PROFILE[n] for n in profile_names]
        if not self.profiles:
            raise ValueError("ArchetypePool requires at least one profile")

    def __len__(self) -> int:
        return len(self.profiles)

    def sample_opponent(self, rng: random.Random) -> ArchetypePolicy:
        """Pick a profile uniformly and return it wrapped as a Policy."""
        profile = rng.choice(self.profiles)
        return ArchetypePolicy(
            profile=profile,
            abstraction=self.abstraction,
            calibration=self.calibration,
            bucket_runouts=self.bucket_runouts,
        )
