"""6-max self-play evaluator: head-to-head between two checkpoints.

Compares two trained DeepCFR6MaxSolver checkpoints by playing N hands where
each hand has seats randomly assigned to policy A or policy B. The scoring
metric is per-hand ICM-equity-delta (the same value CFR optimizes).

Usage:
    python -m scripts.eval_6max_self_play \\
        --ckpt-a runs/.../ckpt_iter_0100.pt \\
        --ckpt-b runs/.../ckpt_iter_0010.pt \\
        --abstraction runs/abstraction_*/abstraction.pkl \\
        --structure configs/ignition_double_up_6max_turbo.yaml \\
        --hands 200

Output: avg ICM equity per hand for policy A, with stderr from N samples.
Positive value means A is stronger than B per hand by that many equity units.

Use cases:
  - Eval during training: compare iter_N vs iter_baseline checkpoints
  - Eval after training: compare different seeds, hyperparams, etc.
"""
from __future__ import annotations
import argparse
import logging
import math
import random
import time
from pathlib import Path
from typing import Optional

import pyspiel
import torch

from src.nlhe.abstraction import Abstraction
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max
from src.nlhe.game_strings import TournamentStructure, six_max_sng
from src.nlhe.stack_sampler import sample_starting_state
from src.nlhe.cfr6 import (
    NUM_SEATS_6MAX,
    CFR6MaxContext,
    is_tournament_terminal,
    compute_icm_payouts,
)
from src.nlhe.infoset6 import (
    InfosetEncoder6Max,
    parse_state_6max,
    parse_state_repeated_6max,
)
from src.nlhe.networks6 import N_DISCRETE_ACTIONS
from src.nlhe.actions import (
    DiscreteAction,
    discretize_legal_actions,
)
from src.nlhe.cfr6 import _build_view_6max
from src.nlhe.icm_returns import icm_adjust_returns


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_6max_self_play")


def _load_solver(
    checkpoint_path: str,
    abstraction: Abstraction,
    structure: TournamentStructure,
) -> DeepCFR6MaxSolver:
    """Construct a solver matching the checkpoint's saved hyperparameters,
    then load weights into it.

    Reads config_dict from the checkpoint so hidden_dim/buffer_capacity/etc.
    match the network shape that was saved. Otherwise load_state_dict would
    raise a size-mismatch error.
    """
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    saved = ckpt.get("config_dict", {})

    cfg = TrainConfig6Max(
        starting_stack=saved.get("starting_stack", 1500),
        big_blind=saved.get("big_blind", 100),
        small_blind=saved.get("small_blind", 50),
        payout_mode=saved.get("payout_mode", "double_up"),
        buy_in=saved.get("buy_in", 1.0),
        first_share=saved.get("first_share", 0.65),
        hidden_dim=list(saved.get("hidden_dim", [64, 64])),
        n_iterations=1,
        traversals_per_iter=1,
        train_steps_per_iter=1,
        batch_size=1,
        buffer_capacity=saved.get("buffer_capacity", 100),
        bucket_runouts=saved.get("bucket_runouts", 20),
        seed=0,
        tournament_structure_path=None,
        num_paid=saved.get("num_paid", 3),
    )
    game = pyspiel.load_game(six_max_sng(starting_stack=cfg.starting_stack))
    solver = DeepCFR6MaxSolver(game=game, abstraction=abstraction, config=cfg)
    solver.load_checkpoint(checkpoint_path)
    return solver


def _sample_action_from_policy(
    solver: DeepCFR6MaxSolver,
    parsed: dict,
    state,
    rng: random.Random,
    mode: str = "sample",
):
    """Use solver\'s 7-dim advantage network to pick an abstract action,
    then map it to a concrete chip action.

    This mirrors the opponent-sampling block in cfr6.py\'s traverse_6max:
      1. discretize_legal_actions gives us {DiscreteAction: chip_action}
      2. build a 7-dim legal_mask, query advantages
      3. RM+ strategy = positive_advantages / sum, masked to legal
      4. sample or argmax from strategy
      5. map chosen DiscreteAction back to chip_action via the map
    """
    import numpy as np

    cp = parsed["current_player"]
    legal_chip = list(state.legal_actions())
    view = _build_view_6max(state, parsed)
    discrete_to_chip = discretize_legal_actions(legal_chip, view)

    if not discrete_to_chip:
        # Defensive: pick any legal chip action
        return rng.choice(legal_chip)

    legal_mask = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
    for da in discrete_to_chip:
        legal_mask[int(da)] = 1.0

    encoded = solver.encoder.encode_from_parsed(parsed, rng=rng)
    features = np.asarray(encoded, dtype=np.float32)

    # Deployment policy distribution (masked, sums to 1 on legal). The schema
    # dispatch lives in inference_policy: v2 checkpoints play the shared strategy
    # net (average policy); v1 checkpoints fall back to the regret-matched
    # current strategy off the advantage net (the legacy behavior). For the
    # common v1 weighted case this is the identical distribution + rng.choices
    # call as before, so sample-mode behavior is preserved.
    policy = solver.policy_nets.inference_policy(cp, features, legal_mask)

    if mode == "argmax":
        chosen_da_idx = int(np.argmax(policy))
    else:
        chosen_da_idx = rng.choices(
            range(N_DISCRETE_ACTIONS),
            weights=policy.tolist(),
            k=1,
        )[0]

    da = DiscreteAction(chosen_da_idx)
    chip_action = discrete_to_chip.get(da)
    if chip_action is None:
        # Belt-and-suspenders
        chip_action = rng.choice(list(discrete_to_chip.values()))
    return int(chip_action)


def play_one_hand(
    seat_to_solver: list,
    structure: TournamentStructure,
    rng: random.Random,
    num_paid: int = 3,
    mode: str = "sample",
):
    """Play one hand from a sampled tournament state.

    Args:
        seat_to_solver: length-6 list where each entry is the solver that
            controls that seat. (Two solvers, distributed across seats.)
        structure: tournament structure for sampling starting state.
        rng: random source.
        num_paid: ICM payout count.
        mode: 'sample' or 'argmax' for action selection.

    Returns:
        dict with: seat_to_equity_delta (length 6), sampled_blind_level, alive_count
    """
    # Sample a starting state.
    sampled = sample_starting_state(structure, rng, num_paid=num_paid)
    gs = structure.to_inner_game_string_for_state(
        blind_level=sampled["blind_level"],
        stacks=sampled["stacks"],
        dealer_seat=sampled["dealer_seat"],
    )
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    starting_stacks = list(sampled["stacks"])

    # Play to terminal using each seat's assigned solver.
    max_steps = 500  # safety cap
    for _ in range(max_steps):
        if state.is_terminal():
            break
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            chosen = rng.choices(
                [o[0] for o in outcomes], weights=[o[1] for o in outcomes], k=1
            )[0]
            state.apply_action(chosen)
            continue
        # Decision node
        if hasattr(state, "dealer_seat"):
            parsed = parse_state_repeated_6max(state)
        else:
            parsed = parse_state_6max(state)
        cp = parsed["current_player"]
        solver = seat_to_solver[cp]
        a = _sample_action_from_policy(solver, parsed, state, rng, mode=mode)
        state.apply_action(a)

    if not state.is_terminal():
        # Hand exceeded safety cap. Treat as a wash.
        return {
            "seat_to_equity_delta": [0.0] * NUM_SEATS_6MAX,
            "sampled_blind_level": sampled["blind_level"].level,
            "alive_count": sampled["alive_count"],
            "exceeded_cap": True,
        }

    # Convert chip-returns to ICM-equity-delta against the sampled starting stacks.
    chip_returns = state.returns()
    payouts = [2.0] * num_paid  # Double Up
    equity_delta = icm_adjust_returns(
        chip_returns=chip_returns,
        starting_stacks=starting_stacks,
        payouts=payouts,
    )
    return {
        "seat_to_equity_delta": equity_delta,
        "sampled_blind_level": sampled["blind_level"].level,
        "alive_count": sampled["alive_count"],
        "exceeded_cap": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a", required=True, help="checkpoint A ('trained')")
    p.add_argument("--ckpt-b", required=True, help="checkpoint B ('baseline')")
    p.add_argument("--abstraction", required=True, help="abstraction pkl path")
    p.add_argument("--structure", required=True, help="tournament structure YAML")
    p.add_argument("--hands", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="sample", choices=["sample", "argmax"])
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    log.info(f"Eval: {args.hands} hands, A=ckpt-a, B=ckpt-b, mode={args.mode}")

    abstr = Abstraction.load(args.abstraction)
    structure = TournamentStructure.from_yaml(args.structure)

    log.info("loading checkpoint A...")
    solver_a = _load_solver(args.ckpt_a, abstr, structure)
    log.info("loading checkpoint B...")
    solver_b = _load_solver(args.ckpt_b, abstr, structure)

    rng = random.Random(args.seed)

    # Accumulate per-policy total equity delta across hands.
    a_total = 0.0
    a_squared = 0.0  # for stderr
    a_seat_counts = [0] * 6
    b_total = 0.0
    b_squared = 0.0
    b_seat_counts = [0] * 6
    n_hands = 0
    n_capped = 0

    t0 = time.time()
    for h in range(1, args.hands + 1):
        # Randomly assign each seat to A or B. (~half each in expectation.)
        seat_assignment = [rng.choice(['A', 'B']) for _ in range(6)]
        seat_to_solver = [
            solver_a if c == 'A' else solver_b
            for c in seat_assignment
        ]
        for s, c in enumerate(seat_assignment):
            if c == 'A':
                a_seat_counts[s] += 1
            else:
                b_seat_counts[s] += 1

        result = play_one_hand(seat_to_solver, structure, rng, mode=args.mode)
        if result["exceeded_cap"]:
            n_capped += 1
            continue

        # Sum equity for A's seats and B's seats.
        a_hand_total = sum(
            result["seat_to_equity_delta"][s]
            for s, c in enumerate(seat_assignment) if c == 'A'
        )
        b_hand_total = sum(
            result["seat_to_equity_delta"][s]
            for s, c in enumerate(seat_assignment) if c == 'B'
        )
        n_a_seats = seat_assignment.count('A')
        n_b_seats = seat_assignment.count('B')

        # Per-seat avg for this hand: a_hand_total / n_a_seats
        if n_a_seats > 0:
            per_seat_a = a_hand_total / n_a_seats
            a_total += per_seat_a
            a_squared += per_seat_a ** 2
        if n_b_seats > 0:
            per_seat_b = b_hand_total / n_b_seats
            b_total += per_seat_b
            b_squared += per_seat_b ** 2

        n_hands += 1

        if h % args.log_every == 0 or h == args.hands:
            avg_a = a_total / n_hands if n_hands else 0.0
            avg_b = b_total / n_hands if n_hands else 0.0
            # stderr of A
            if n_hands > 1:
                var_a = (a_squared / n_hands) - avg_a ** 2
                stderr_a = math.sqrt(max(0.0, var_a) / n_hands)
            else:
                stderr_a = 0.0
            elapsed = time.time() - t0
            log.info(
                f"  hand {h:>5}/{args.hands}  "
                f"avg_a={avg_a:>+7.4f} +/- {stderr_a:.4f}  "
                f"avg_b={avg_b:>+7.4f}  "
                f"capped={n_capped}  "
                f"[{elapsed:.1f}s, {h/elapsed:.1f} h/s]"
            )

    log.info("=" * 60)
    log.info(f"Done. {n_hands} hands counted, {n_capped} capped.")
    if n_hands > 0:
        avg_a = a_total / n_hands
        avg_b = b_total / n_hands
        var_a = (a_squared / n_hands) - avg_a ** 2
        stderr_a = math.sqrt(max(0.0, var_a) / n_hands)
        diff = avg_a - avg_b
        log.info(f"Policy A avg per-seat per-hand equity delta: {avg_a:>+8.4f} +/- {stderr_a:.4f}")
        log.info(f"Policy B avg per-seat per-hand equity delta: {avg_b:>+8.4f}")
        log.info(f"Diff (A - B): {diff:>+8.4f}")
        log.info(f"  positive = A stronger; negative = B stronger")
        # Note: zero-sum means avg_a + avg_b should be ~0 after seat assignments balance.
        log.info(f"  (sanity: avg_a + avg_b = {avg_a + avg_b:+.4f}, should be near 0)")


if __name__ == "__main__":
    main()
