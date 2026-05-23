"""Build OpenSpiel universal_poker game strings parametrically.

Phase 4a (docs/PHASE4_PLAN.md). The HUNL configs hardcoded game strings
with numPlayers=2; for 6-max we need the same shape with numPlayers=6.
This module builds the string from named parameters so configs can say
"6-max, 1500 chip starting stack, 100/50 blinds" instead of pasting a
multi-line string with hidden parameters.

NOTE: this module just builds strings. It does NOT validate that the
downstream code (src/nlhe/infoset.py, src/nlhe/solver.py, ...) works
correctly for num_players > 2. The smoke test in
tests/test_6max_game_load.py confirms OpenSpiel can load and walk a
6-max game with random actions — but real training validation comes
later in Phase 4.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PokerGameConfig:
    """Parameters for an OpenSpiel universal_poker game string.

    Defaults reproduce the HUNL Phase 2d game (200bb starting, 50/100 blinds).
    """
    num_players: int = 2
    starting_stack: int = 20000
    big_blind: int = 100
    small_blind: int = 50

    def __post_init__(self) -> None:
        if self.num_players < 2:
            raise ValueError(f"num_players must be >= 2, got {self.num_players}")
        if self.num_players > 10:
            raise ValueError(
                f"num_players must be <= 10 (universal_poker limit), got {self.num_players}"
            )
        if self.starting_stack <= 0:
            raise ValueError(f"starting_stack must be > 0, got {self.starting_stack}")
        if self.big_blind <= 0 or self.small_blind <= 0:
            raise ValueError("blinds must be > 0")
        if self.small_blind >= self.big_blind:
            raise ValueError(
                f"small_blind {self.small_blind} must be < big_blind {self.big_blind}"
            )

    def to_universal_poker_string(self) -> str:
        """Build the OpenSpiel universal_poker game string for this config."""
        n = self.num_players
        # Blind structure: small blind first, then big blind, then 0 for everyone else.
        # Format: "<sb> <bb> 0 0 0 0" for 6-max for example.
        blind_parts = [str(self.small_blind), str(self.big_blind)]
        blind_parts.extend(["0"] * (n - 2))
        blind_str = " ".join(blind_parts)
        # firstPlayer per round: in 6-max, player 3 acts first preflop (UTG),
        # then player 1 (small blind seat) acts first postflop. For HUNL the
        # existing pattern is "2 1 1 1" (BB acts first preflop, SB acts first
        # postflop). For >2 players, universal_poker uses the small-blind seat
        # to act first postflop, and 3 (UTG) preflop.
        if n == 2:
            first_player = "2 1 1 1"
        else:
            # Postflop: small blind acts first. Preflop: UTG (seat 3) acts first.
            first_player = f"3 1 1 1"
        # Stack list: identical for every player.
        stack_str = " ".join([str(self.starting_stack)] * n)
        return (
            f"universal_poker(betting=nolimit,"
            f"numPlayers={n},"
            f"numRounds=4,"
            f"blind={blind_str},"
            f"firstPlayer={first_player},"
            f"numSuits=4,"
            f"numRanks=13,"
            f"numHoleCards=2,"
            f"numBoardCards=0 3 1 1,"
            f"stack={stack_str},"
            f"bettingAbstraction=fullgame)"
        )


# Convenience constructors matching the existing configs.

def hunl_200bb() -> str:
    """The HUNL config from configs/nlhe_200bb.yaml and Phase 2d training."""
    return PokerGameConfig(num_players=2, starting_stack=20000,
                           big_blind=100, small_blind=50).to_universal_poker_string()


def hunl_20bb() -> str:
    """The smoke/phase2b config: 20bb starting (2000 chips, 100/50 blinds)."""
    return PokerGameConfig(num_players=2, starting_stack=2000,
                           big_blind=100, small_blind=50).to_universal_poker_string()


def six_max_200bb() -> str:
    """6-max with 200bb starting stacks (matches HUNL chip scale for comparison)."""
    return PokerGameConfig(num_players=6, starting_stack=20000,
                           big_blind=100, small_blind=50).to_universal_poker_string()


def six_max_sng(starting_stack: int = 1500) -> str:
    """6-max SNG default: 15bb starting (1500 chips, 100/50 blinds).

    Matches typical SNG starting depth where the late game compresses quickly
    to push/fold. Phase 4 training target.
    """
    return PokerGameConfig(num_players=6, starting_stack=starting_stack,
                           big_blind=100, small_blind=50).to_universal_poker_string()


# ===== Tournament structures (Phase 4f) =====
#
# repeated_poker integration. The existing PokerGameConfig builds single-hand
# universal_poker game strings. For SNG training we need multi-hand
# tournaments with escalating blinds, button rotation, and chip carryover.
# OpenSpiel's repeated_poker (1.6.11+) wraps universal_poker for that purpose.
#
# Antes: universal_poker does not support an `ante` parameter (see
# tests/test_game_strings.py for the probe that confirms this). We approximate antes
# by inflating the big blind: total dead money per hand = blinds + N*ante;
# we encode that as a single big blind of `bb + N*ante` and post no ante.
# This preserves the pre-action pot size exactly but misattributes who paid
# what (only the BB has skin-in-the-game pre-action). Expected EV leak:
# 1-3 bb/100 of BB-defense range distortion. Documented in DECISIONS.md.


@dataclass(frozen=True)
class BlindLevel:
    """One row of a tournament's blind schedule.

    Fields match Ignition's in-client Tourney Info display.

    Args:
        level: 1-indexed level number.
        small_blind: small-blind amount in chips.
        big_blind: big-blind amount in chips. Must exceed small_blind.
        ante: per-player ante in chips. 0 if no antes at this level.
        duration_minutes: minutes spent at this level before escalating.
            Used only for documentation; OpenSpiel's blind_schedule
            string measures level lengths in HANDS, not minutes.
    """
    level: int
    small_blind: int
    big_blind: int
    ante: int = 0
    duration_minutes: int = 5

    def __post_init__(self) -> None:
        if self.level < 1:
            raise ValueError(f"level must be >= 1, got {self.level}")
        if self.small_blind <= 0:
            raise ValueError(f"small_blind must be > 0, got {self.small_blind}")
        if self.big_blind <= 0:
            raise ValueError(f"big_blind must be > 0, got {self.big_blind}")
        if self.small_blind >= self.big_blind:
            raise ValueError(
                f"small_blind {self.small_blind} must be < big_blind {self.big_blind}"
            )
        if self.ante < 0:
            raise ValueError(f"ante must be >= 0, got {self.ante}")
        if self.duration_minutes <= 0:
            raise ValueError(
                f"duration_minutes must be > 0, got {self.duration_minutes}"
            )

    def inflated_big_blind(self, num_players: int) -> int:
        """Approximate antes by inflating the big blind.

        Returns big_blind + num_players * ante, which preserves the
        total pre-action pot size for any num_players >= 2.

        Args:
            num_players: number of players at the table this hand.

        Returns:
            Inflated big-blind value (chips). Equal to big_blind when
            ante=0 (no inflation needed).
        """
        if num_players < 2:
            raise ValueError(
                f"num_players must be >= 2 for inflated_big_blind, got {num_players}"
            )
        return self.big_blind + num_players * self.ante


@dataclass(frozen=True)
class TournamentStructure:
    """Full tournament definition: schedule, payouts, sizing.

    Loaded from a YAML config (see configs/ignition_double_up_6max_turbo.yaml).
    This is the single source of truth for any specific tournament format.
    Training configs reference a TournamentStructure by path rather than
    duplicating blind/payout data.

    Args:
        format_name: human-readable identifier, e.g. "ignition_double_up_6max_turbo".
        num_players: number of seats at the table.
        starting_chips: per-seat chip count at tournament start.
        payout_mode: "double_up" for top-N-equal-pay (used by ICM math).
        payouts_dollars: tuple of dollar payouts per finishing position.
        buy_in_dollars: cost to enter (entry fee excluded).
        level_duration_minutes: fallback for levels without explicit duration.
        blind_schedule: ordered tuple of BlindLevels. Levels unique, ascending.
        training_weights: tuple of (level_int, weight_float). Sum to ~1.0.
    """
    format_name: str
    num_players: int
    starting_chips: int
    payout_mode: str
    payouts_dollars: tuple
    buy_in_dollars: float
    level_duration_minutes: int
    blind_schedule: tuple
    training_weights: tuple

    def __post_init__(self) -> None:
        if not self.format_name:
            raise ValueError("format_name must be non-empty")
        if self.num_players < 2:
            raise ValueError(f"num_players must be >= 2, got {self.num_players}")
        if self.num_players > 10:
            raise ValueError(
                f"num_players must be <= 10 (universal_poker limit), got {self.num_players}"
            )
        if self.starting_chips <= 0:
            raise ValueError(f"starting_chips must be > 0, got {self.starting_chips}")
        if self.payout_mode not in ("double_up",):
            raise ValueError(
                f"unknown payout_mode {self.payout_mode!r}; supported: double_up"
            )
        if not self.payouts_dollars:
            raise ValueError("payouts_dollars must be non-empty")
        if any(p <= 0 for p in self.payouts_dollars):
            raise ValueError("all payouts_dollars must be > 0")
        if self.payout_mode == "double_up":
            first = self.payouts_dollars[0]
            if not all(abs(p - first) < 1e-9 for p in self.payouts_dollars):
                raise ValueError(
                    f"double_up requires equal payouts, got {self.payouts_dollars}"
                )
        if self.buy_in_dollars <= 0:
            raise ValueError(f"buy_in_dollars must be > 0, got {self.buy_in_dollars}")
        if self.level_duration_minutes <= 0:
            raise ValueError(
                f"level_duration_minutes must be > 0, got {self.level_duration_minutes}"
            )
        if not self.blind_schedule:
            raise ValueError("blind_schedule must be non-empty")
        for entry in self.blind_schedule:
            if not isinstance(entry, BlindLevel):
                raise ValueError(
                    f"blind_schedule entries must be BlindLevel, got {type(entry).__name__}"
                )
        levels = [bl.level for bl in self.blind_schedule]
        if levels != sorted(set(levels)):
            raise ValueError(
                f"blind_schedule levels must be unique and ascending, got {levels}"
            )
        scheduled_levels = set(levels)
        for level_n, _weight in self.training_weights:
            if level_n not in scheduled_levels:
                raise ValueError(
                    f"training_weights references level {level_n} not in blind_schedule"
                )
        if self.training_weights:
            total = sum(w for _, w in self.training_weights)
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"training_weights must sum to ~1.0, got {total:.4f}")
            if any(w < 0 for _, w in self.training_weights):
                raise ValueError("training_weights values must be >= 0")

    def level(self, n: int) -> BlindLevel:
        """Look up a blind level by 1-indexed level number."""
        for bl in self.blind_schedule:
            if bl.level == n:
                return bl
        raise KeyError(
            f"level {n} not in blind_schedule "
            f"(available: {[bl.level for bl in self.blind_schedule]})"
        )

    def total_chips_in_play(self) -> int:
        """Total chips across all seats at tournament start."""
        return self.starting_chips * self.num_players

    def num_paid(self) -> int:
        """Number of finishing positions that receive a payout."""
        return len(self.payouts_dollars)

    def buy_in_chips(self) -> int:
        """Equivalent of one buy-in in chips. Used to normalize ICM returns."""
        return self.starting_chips


    # ===== Game-string builders =====

    def to_inner_game_string(self, level: int = 1) -> str:
        """Build the inner universal_poker game string for a specific blind level.

        Args:
            level: 1-indexed blind level to use for blinds/antes.

        Returns:
            A universal_poker(...) game string ready to pass to pyspiel.load_game.
            If the level has ante > 0, the big blind is inflated to
            big_blind + num_players * ante to absorb the total pre-action
            ante pot (see module docstring on the antes approximation).
        """
        bl = self.level(level)
        n = self.num_players
        sb = bl.small_blind
        bb = bl.inflated_big_blind(n)  # absorbs antes via inflation

        blind_parts = [str(sb), str(bb)]
        blind_parts.extend(["0"] * (n - 2))
        blind_str = " ".join(blind_parts)

        # firstPlayer rounds: preflop=UTG (seat 3 for >2 players),
        # postflop=SB. For 2-player, BB acts first preflop.
        if n == 2:
            first_player = "2 1 1 1"
        else:
            first_player = "3 1 1 1"

        stack_str = " ".join([str(self.starting_chips)] * n)

        return (
            f"universal_poker(betting=nolimit,"
            f"numPlayers={n},"
            f"numRounds=4,"
            f"blind={blind_str},"
            f"firstPlayer={first_player},"
            f"numSuits=4,"
            f"numRanks=13,"
            f"numHoleCards=2,"
            f"numBoardCards=0 3 1 1,"
            f"stack={stack_str},"
            f"bettingAbstraction=fullgame)"
        )

    def to_inner_game_string_for_state(
        self,
        blind_level,
        stacks,
        dealer_seat,
    ):
        """Build a single-hand universal_poker game string for a sampled state.

        Used by stack_sampler integration: rotates blinds, firstPlayer, and
        stacks to put the dealer at a specific seat with specific chip counts.

        Args:
            blind_level: BlindLevel for blinds/antes (ante absorbed via
                inflation, same as to_inner_game_string).
            stacks: per-seat chip counts, length num_players. Busted seats
                must be 0; alive seats must be >= bb_inflated.
            dealer_seat: 0-indexed seat that holds the button. Small blind
                is at (dealer+1) % n, big blind at (dealer+2) % n.

        Returns:
            A universal_poker(...) game string with the dealer at dealer_seat.

        Caveats:
            universal_poker does not support eliminated seats; if a "busted"
            seat's stack is 0 we'd hit the same assertion failure that broke
            repeated_poker integration. This method REQUIRES stacks[i] >= 1
            for all seats. Stack_sampler guarantees this since it always
            allocates at least bb_inflated to alive seats.

            For busted seats this method passes stack=1 as a placeholder
            (the player will fold/auto-busts on first action; CFR ignores
            them via the active mask).
        """
        n = self.num_players
        if len(stacks) != n:
            raise ValueError(f"stacks length {len(stacks)} != num_players {n}")
        if not (0 <= dealer_seat < n):
            raise ValueError(f"dealer_seat {dealer_seat} out of range [0,{n})")

        sb = blind_level.small_blind
        bb = blind_level.inflated_big_blind(n)

        # Identify alive seats (stack > 0). At a contracted table, blinds
        # and firstPlayer must rotate over ALIVE seats only; (dealer+1) % n
        # may land on a busted seat in real tournament states.
        alive_seats = [i for i, s in enumerate(stacks) if s > 0]
        if dealer_seat not in alive_seats:
            raise ValueError(
                f"dealer_seat {dealer_seat} is busted (stack=0); "
                f"alive seats are {alive_seats}"
            )
        n_alive = len(alive_seats)
        if n_alive < 2:
            raise ValueError(f"need >= 2 alive seats, got {n_alive}")

        # Position of dealer within the alive-seats list.
        dealer_pos = alive_seats.index(dealer_seat)

        # SB and BB are the next 2 alive seats clockwise from dealer.
        sb_seat_alive_pos = (dealer_pos + 1) % n_alive
        bb_seat_alive_pos = (dealer_pos + 2) % n_alive
        sb_seat = alive_seats[sb_seat_alive_pos]
        bb_seat = alive_seats[bb_seat_alive_pos]

        # Assign blinds in original 6-seat indexing.
        blind_array = [0] * n
        blind_array[sb_seat] = sb
        blind_array[bb_seat] = bb
        blind_str = " ".join(str(b) for b in blind_array)

        # firstPlayer (1-indexed): preflop = UTG (3 alive-positions past
        # dealer for full table; in HU it's the BB), postflop = SB.
        if n_alive == 2:
            # Heads-up: BB acts first preflop, SB first postflop.
            preflop_actor = bb_seat + 1
            postflop_actor = sb_seat + 1
        else:
            utg_alive_pos = (dealer_pos + 3) % n_alive
            utg_seat = alive_seats[utg_alive_pos]
            preflop_actor = utg_seat + 1
            postflop_actor = sb_seat + 1
        first_player = f"{preflop_actor} {postflop_actor} {postflop_actor} {postflop_actor}"

        # Stacks: busted seats need a placeholder (universal_poker requires
        # stack > 0). Use 1 chip. They never act because they're not in
        # firstPlayer rotation; the placeholder is just to satisfy the parser.
        stack_safe = [max(1, s) for s in stacks]
        stack_str = " ".join(str(s) for s in stack_safe)

        return (
            f"universal_poker(betting=nolimit,"
            f"numPlayers={n},"
            f"numRounds=4,"
            f"blind={blind_str},"
            f"firstPlayer={first_player},"
            f"numSuits=4,"
            f"numRanks=13,"
            f"numHoleCards=2,"
            f"numBoardCards=0 3 1 1,"
            f"stack={stack_str},"
            f"bettingAbstraction=fullgame)"
        )

    def to_blind_schedule_string(self, hands_per_level: int = 10) -> str:
        """Build the repeated_poker `blind_schedule` parameter string.

        Format expected by OpenSpiel:
            "<num_hands>:<sb>/<bb>;<num_hands>:<sb>/<bb>;..."

        Args:
            hands_per_level: number of hands to spend at each level before
                escalating. 10 is reasonable for training (slightly
                over-samples each level vs real Ignition's ~3 hands per
                5-min Turbo level), giving the network enough exposure
                to each stage.

        Returns:
            A semicolon-terminated blind-schedule string.
        """
        if hands_per_level < 1:
            raise ValueError(
                f"hands_per_level must be >= 1, got {hands_per_level}"
            )
        parts = []
        for bl in self.blind_schedule:
            bb_inflated = bl.inflated_big_blind(self.num_players)
            parts.append(f"{hands_per_level}:{bl.small_blind}/{bb_inflated}")
        return ";".join(parts) + ";"

    def to_repeated_poker_string(
        self,
        max_num_hands: int = 200,
        hands_per_level: int = 10,
        reset_stacks: bool = False,
        rotate_dealer: bool = True,
    ) -> str:
        """Build the full repeated_poker game string with this tournament's structure.

        Args:
            max_num_hands: safety cap on number of hands. For top-3-equal-pay
                Double Up at 6-max, the tournament can't last past about
                level 10 due to chip-math constraints (see structure config
                docstring), so 200 is a very generous cap. Set conservatively.
            hands_per_level: passed through to to_blind_schedule_string.
            reset_stacks: False for tournament (chips carry over hand to hand).
                True only for cash-game training. Default False.
            rotate_dealer: True for tournament (button moves clockwise).
                Default True.

        Returns:
            A repeated_poker(...) game string ready to pass to pyspiel.load_game.
            Uses level 1's blinds as the inner universal_poker starting state;
            the blind_schedule parameter handles escalation across hands.
        """
        if max_num_hands < 1:
            raise ValueError(
                f"max_num_hands must be >= 1, got {max_num_hands}"
            )
        inner = self.to_inner_game_string(level=1)
        sched = self.to_blind_schedule_string(hands_per_level=hands_per_level)
        return (
            f"repeated_poker("
            f"max_num_hands={max_num_hands},"
            f"reset_stacks={str(reset_stacks)},"
            f"rotate_dealer={str(rotate_dealer)},"
            f"blind_schedule={sched},"
            f"universal_poker_game_string={inner}"
            f")"
        )

    @classmethod
    def from_yaml(cls, path: str) -> "TournamentStructure":
        """Load a TournamentStructure from a YAML config file.

        Expected YAML schema (see configs/ignition_double_up_6max_turbo.yaml):
            format_name: string
            num_players: int
            starting_chips: int
            payout_mode: string
            payouts_dollars: list of floats
            buy_in_dollars: float
            level_duration_minutes: int
            blind_schedule: list of dicts with level/small_blind/big_blind/ante
            training_weights: dict of "level_N" -> float

        Args:
            path: path to the YAML config file.

        Returns:
            A fully-validated TournamentStructure instance.
        """
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse blind_schedule into BlindLevel tuple
        schedule_data = data.get("blind_schedule", [])
        bls = tuple(
            BlindLevel(
                level=row["level"],
                small_blind=row["small_blind"],
                big_blind=row["big_blind"],
                ante=row.get("ante", 0),
                duration_minutes=row.get(
                    "duration_minutes",
                    data.get("level_duration_minutes", 5),
                ),
            )
            for row in schedule_data
        )

        # Parse training_weights from "level_N: weight" dict into tuple of (int, float)
        weights_data = data.get("training_weights", {}) or {}
        weights = tuple(
            (int(k.replace("level_", "")), float(v))
            for k, v in weights_data.items()
        )
        # Sort by level for determinism
        weights = tuple(sorted(weights))

        return cls(
            format_name=data["format_name"],
            num_players=data["num_players"],
            starting_chips=data["starting_chips"],
            payout_mode=data["payout_mode"],
            payouts_dollars=tuple(data["payouts_dollars"]),
            buy_in_dollars=float(data["buy_in_dollars"]),
            level_duration_minutes=data["level_duration_minutes"],
            blind_schedule=bls,
            training_weights=weights,
        )
