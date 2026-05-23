"""Sample starting states for single-hand-per-trajectory tournament training.

Phase 4f. For Choice 1 training (single-hand CFR with ICM-projected
terminal values), each training trajectory starts at a sampled state
representing some moment in the tournament:

  - blind level (early/mid/late escalation)
  - per-seat chip distribution
  - dealer seat (rotation across trajectories)

The sampler\'s job is to produce a distribution of starting states that
matches what the bot will face at deployment. The bot at deployment sees:
  - Hand 1 of a fresh tournament (6 alive, equal stacks 1500 each)
  - Mid-tournament with varied stack distributions and some bust history
  - Late-game / bubble pressure with short stacks

A well-trained bot must perform across that whole distribution. The
sampler weights each stage by the structure\'s training_weights and
samples plausible stack distributions for each stage.

Constraints enforced:
  - Chip-pool conservation: sum(stacks) == num_players * starting_chips
  - Each alive stack >= inflated_big_blind (avoids the OpenSpiel
    "blind > stack" assertion failure)
  - Number of alive >= num_paid + 1 (we don\'t sample past-bubble states;
    those are CFR terminals via is_tournament_terminal)
"""
from __future__ import annotations
import random
from typing import Sequence

from src.nlhe.game_strings import TournamentStructure, BlindLevel


def sample_starting_state(
    structure: TournamentStructure,
    rng: random.Random,
    num_paid: int = 3,
):
    """Sample a starting state for a single-hand training trajectory.

    Args:
        structure: the TournamentStructure to sample from. The
            training_weights field selects blind level; the rest of
            the sampler uses heuristics keyed off that level.
        rng: random source.
        num_paid: number of paid positions (3 for Double Up). Used to
            constrain alive count to > num_paid (we don\'t train on
            past-bubble states).

    Returns:
        Dict with keys:
            blind_level: a BlindLevel from the structure
            stacks: list of length num_players, alive players have
                stack >= bb_inflated, busted players have 0
            dealer_seat: integer in [0, num_players)
            alive_count: number of alive players (>= num_paid + 1)
    """
    n = structure.num_players
    starting_chips = structure.starting_chips
    total_chips = structure.total_chips_in_play()

    blind_level = _sample_blind_level(structure, rng)
    bb_inflated = blind_level.inflated_big_blind(n)

    # Stage is implied by level: early = level 1-2, mid = 3-5, late = 6+.
    # We sample alive_count and stack distribution conditional on stage.
    stage = _stage_of_level(blind_level.level)
    alive_count = _sample_alive_count(
        stage, n, num_paid, total_chips, bb_inflated, rng
    )
    stacks = _sample_stack_distribution(
        n=n,
        alive_count=alive_count,
        total_chips=total_chips,
        bb_inflated=bb_inflated,
        starting_chips=starting_chips,
        stage=stage,
        rng=rng,
    )

    # Dealer seat: uniformly random over the alive seats. Spreading dealer
    # across trajectories ensures each network seat trains on each position.
    alive_seats = [i for i, s in enumerate(stacks) if s > 0]
    dealer_seat = rng.choice(alive_seats)

    return {
        "blind_level": blind_level,
        "stacks": stacks,
        "dealer_seat": dealer_seat,
        "alive_count": alive_count,
    }


def _sample_blind_level(
    structure: TournamentStructure, rng: random.Random
) -> BlindLevel:
    """Sample a blind level using training_weights from the structure.

    If training_weights is empty, falls back to uniform over schedule.
    """
    if structure.training_weights:
        levels = [t[0] for t in structure.training_weights]
        weights = [t[1] for t in structure.training_weights]
        chosen_level = rng.choices(levels, weights=weights, k=1)[0]
    else:
        levels = [bl.level for bl in structure.blind_schedule]
        chosen_level = rng.choice(levels)
    return structure.level(chosen_level)


def _stage_of_level(level: int) -> str:
    """Map blind level to tournament stage.

    Heuristic only — different tournaments hit bubble at different
    levels. For Ignition Double Up 6-max Turbo: bubble typically at L4-L7.
    """
    if level <= 2:
        return "deep"      # 30-60 BB effective, real postflop play
    elif level <= 5:
        return "mid"       # 8-25 BB, push/fold creeping in
    else:
        return "short"     # <10 BB, mostly push/fold


def _sample_alive_count(
    stage: str,
    n: int,
    num_paid: int,
    total_chips: int,
    bb_inflated: int,
    rng: random.Random,
) -> int:
    """Sample number of alive players given stage and chip-pool feasibility.

    Constraints:
      - alive_count >= num_paid + 1 (we don\'t train past-bubble)
      - alive_count * bb_inflated <= total_chips (every alive player
        must be able to post the blind)
    """
    # Stage priors:
    #   Deep stage: rarely any busts.
    #   Mid stage: typically 4-6 alive.
    #   Short stage: typically 4-5 alive (close to bubble).
    if stage == "deep":
        weights = {6: 0.85, 5: 0.15}
    elif stage == "mid":
        weights = {6: 0.35, 5: 0.40, 4: 0.25}
    else:  # short
        weights = {5: 0.45, 4: 0.55}

    # Feasibility filter: each alive must afford BB.
    max_feasible_alive = total_chips // bb_inflated if bb_inflated > 0 else n
    valid = {
        k: v for k, v in weights.items()
        if k > num_paid and k <= max_feasible_alive
    }

    if not valid:
        # No feasible alive_count at this level: fall back to the smallest
        # above-bubble count that satisfies the chip constraint. If even
        # that is infeasible (e.g., bb_inflated > total_chips), raise.
        min_required = num_paid + 1
        if min_required <= max_feasible_alive:
            return min_required
        raise RuntimeError(
            f"no feasible alive_count at stage={stage}, bb={bb_inflated}, "
            f"total={total_chips}; max_feasible={max_feasible_alive}, "
            f"min_required={min_required}"
        )

    options = list(valid.keys())
    probs = list(valid.values())
    total = sum(probs)
    probs = [p / total for p in probs]
    return rng.choices(options, weights=probs, k=1)[0]


def _sample_stack_distribution(
    n: int,
    alive_count: int,
    total_chips: int,
    bb_inflated: int,
    starting_chips: int,
    stage: str,
    rng: random.Random,
) -> list:
    """Sample plausible per-seat stacks.

    Returns length-n list. Busted seats are 0. Alive seats sum to
    total_chips and each alive stack is >= bb_inflated.

    Strategy:
      - Pick which seats are busted (sampled uniformly from n choose
        n-alive_count).
      - For alive seats, sample relative weights (shape varies by stage),
        normalize so sum == total_chips, then floor at bb_inflated and
        adjust to preserve total.
    """
    # Pick which seats are busted.
    n_busted = n - alive_count
    all_seats = list(range(n))
    busted_seats = rng.sample(all_seats, n_busted) if n_busted > 0 else []
    busted_set = set(busted_seats)
    alive_seats = [i for i in range(n) if i not in busted_set]

    # Pre-allocate bb_inflated to each alive seat, then distribute the
    # REMAINING chips according to sampled shape weights. This is provably
    # correct: each alive stack >= bb_inflated by construction, and the
    # total is preserved exactly.
    remaining_chips = total_chips - alive_count * bb_inflated
    if remaining_chips < 0:
        raise RuntimeError(
            f"infeasible: alive={alive_count}, bb_inflated={bb_inflated}, "
            f"total={total_chips}; need {alive_count * bb_inflated} but have {total_chips}"
        )

    # Sample shape weights for distributing the *remaining* (above-bb) chips.
    # Deep stage: stacks above bb are nearly equal.
    # Mid stage: moderate variance.
    # Short stage: high variance, one chip leader emerging.
    if stage == "deep":
        weights = [1.0 + rng.gauss(0, 0.05) for _ in range(alive_count)]
    elif stage == "mid":
        weights = [rng.lognormvariate(0.0, 0.4) for _ in range(alive_count)]
    else:  # short
        weights = [rng.lognormvariate(0.0, 0.7) for _ in range(alive_count)]
    # Ensure all positive
    weights = [max(w, 0.01) for w in weights]
    w_sum = sum(weights)

    # Above-bb extras per seat
    extras_float = [w / w_sum * remaining_chips for w in weights]
    extras_int = [int(round(e)) for e in extras_float]

    # Correct rounding error on the last seat
    diff = remaining_chips - sum(extras_int)
    if extras_int:
        extras_int[-1] += diff

    # Final per-alive-seat stacks = bb_inflated + extras
    int_stacks = [bb_inflated + e for e in extras_int]

    # Sanity check (these are invariants of the algorithm, should never fail)
    assert sum(int_stacks) == total_chips, (
        f"chip pool violated: sum={sum(int_stacks)} != {total_chips}"
    )
    for s in int_stacks:
        assert s >= bb_inflated, (
            f"bb floor violated: stack {s} < bb_inflated {bb_inflated}"
        )

    # Assemble length-n stacks list with busted seats zeroed.
    stacks = [0] * n
    for seat, chip in zip(alive_seats, int_stacks):
        stacks[seat] = chip

    return stacks
