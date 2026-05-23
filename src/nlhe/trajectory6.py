"""6-max trajectory walker for Phase 4e training.

Phase 4e.1 (docs/PHASE4_PLAN.md). Pure trajectory-generation primitive:
takes a game and a policy, walks one 6-max NLHE game to terminal,
returns the structured trajectory.

NO solver. NO training. NO ICM (that's 4e.2). NO advantage networks
(that's 4e.3). This module is the data-generation foundation that
later 4e layers build on.

A 'policy' here is a callable:
    policy(state, current_player) -> list[float]
where the returned list is a probability distribution over
state.legal_actions(). The walker samples from it, applies the
sampled action, and continues until terminal.

Chance nodes (card dealing) are handled by sampling from the
chance distribution provided by OpenSpiel — the policy callback is
NOT consulted at chance nodes.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# A policy function signature for type clarity.
# Takes (state, current_player_idx) and returns action probabilities over legal_actions.
PolicyFn = Callable[[Any, int], "list[float]"]


@dataclass
class TrajectoryStep:
    """One decision point in a trajectory.

    Includes the info needed by downstream CFR (Phase 4e.3): which
    player acted, what action they took, what policy they sampled from.
    State is captured BEFORE the action is applied.
    """
    player: int                  # Which player acted (0..5).
    legal_actions: list[int]     # Legal actions at this state.
    action_taken: int            # The sampled action.
    policy: list[float]          # The probability distribution sampled from.
    # NOTE: we don't capture the state object itself because OpenSpiel
    # states are mutable and the trajectory walker advances them. The
    # caller can replay the trajectory by replaying the actions if
    # state-at-step-N is needed.


@dataclass
class Trajectory:
    """A complete game trajectory from initial state to terminal.

    Captures every decision point (skipping chance nodes — those are
    sampled automatically by the walker) and the terminal returns.

    Fields:
        steps: list of TrajectoryStep, one per decision point.
        terminal_returns: per-player final returns (length 6 for 6-max).
            These are the chip-EV returns from OpenSpiel. ICM-EV
            transformation happens in 4e.2.
        num_players: number of players in this game.
        n_chance_nodes: number of chance-node samples we made (useful
            for verifying mechanics).
        terminated_at_max_steps: True if we bailed out at max_steps
            without reaching a terminal. Should always be False in
            healthy games; True means a bug.
    """
    steps: list[TrajectoryStep] = field(default_factory=list)
    terminal_returns: list[float] = field(default_factory=list)
    num_players: int = 0
    n_chance_nodes: int = 0
    terminated_at_max_steps: bool = False

    def __len__(self) -> int:
        return len(self.steps)


def walk_game(
    game: Any,
    policy: PolicyFn,
    rng: random.Random,
    max_steps: int = 5000,
) -> Trajectory:
    """Walk one game to terminal, sampling actions from `policy`.

    Args:
        game: OpenSpiel game object (from pyspiel.load_game).
        policy: callable returning action probabilities at each
            non-chance decision node.
        rng: random source for sampling actions and chance outcomes.
        max_steps: safety cap to detect infinite loops. Healthy games
            never hit this; if we do, the trajectory is flagged.

    Returns:
        Trajectory object capturing the decision sequence and terminal
        returns.
    """
    state = game.new_initial_state()
    traj = Trajectory(num_players=game.num_players())

    step = 0
    while not state.is_terminal():
        step += 1
        if step > max_steps:
            traj.terminated_at_max_steps = True
            break

        if state.is_chance_node():
            # Chance node: sample from OpenSpiel's chance distribution
            actions, probs = zip(*state.chance_outcomes())
            sampled = rng.choices(actions, weights=probs, k=1)[0]
            state.apply_action(sampled)
            traj.n_chance_nodes += 1
        else:
            # Decision node: ask policy for action distribution
            cp = state.current_player()
            legal = list(state.legal_actions())
            probs = policy(state, cp)

            # Validate the policy returned a valid distribution.
            if len(probs) != len(legal):
                raise ValueError(
                    f"policy returned {len(probs)} probs but legal_actions has "
                    f"{len(legal)} actions (state.current_player={cp}, step={step})"
                )
            if abs(sum(probs) - 1.0) > 1e-5:
                # Renormalize (tolerate small numerical drift but flag big).
                total = sum(probs)
                if total <= 0:
                    raise ValueError(
                        f"policy returned zero-sum probs at step {step}, player {cp}"
                    )
                probs = [p / total for p in probs]

            # Sample.
            action = rng.choices(legal, weights=probs, k=1)[0]
            traj.steps.append(TrajectoryStep(
                player=cp,
                legal_actions=legal,
                action_taken=action,
                policy=list(probs),
            ))
            state.apply_action(action)

    # Capture terminal returns.
    traj.terminal_returns = list(state.returns())
    return traj


# ===== Standard policies for testing and baseline use =====


def uniform_random_policy(state: Any, current_player: int) -> list[float]:
    """Uniform random over legal actions. Useful for smoke tests."""
    legal = state.legal_actions()
    if not legal:
        return []
    p = 1.0 / len(legal)
    return [p] * len(legal)


def always_fold_policy(state: Any, current_player: int) -> list[float]:
    """Always fold if possible; otherwise pick the first legal action.

    Useful baseline for verifying mechanics: a folder should never have
    positive returns (they lose blinds + posted chips on every hand).
    """
    legal = state.legal_actions()
    if not legal:
        return []
    # In OpenSpiel universal_poker action 0 is the lowest-cost action
    # (fold when facing a bet, check when not). For 'always fold'
    # semantics we just pick action 0 with probability 1.
    probs = [0.0] * len(legal)
    probs[0] = 1.0
    return probs
