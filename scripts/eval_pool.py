"""Multi-baseline 6-max evaluator.

Pits one 'challenger' checkpoint against a list of opponent policies.
Per matchup, plays N hands with seats randomly assigned to challenger or
opponent, reports per-hand ICM-equity-delta with stderr.

Foundation eval: every future training change (DCFR, reinit, league play,
wider abstraction, subgame solving) gets validated by running the new
checkpoint through this script against a fixed opponent pool.

Usage:
    python -m scripts.eval_pool \\
        --challenger dcfr-200=runs/.../ckpt_iter_0200.pt \\
        --opponents \\
            vanilla-200=runs/.../ckpt_iter_0200.pt \\
            vanilla-400=runs/.../ckpt_iter_0400.pt \\
            random=__RANDOM__ \\
        --abstraction runs/abstraction_*/abstraction.pkl \\
        --structure configs/ignition_double_up_6max_turbo.yaml \\
        --hands 5000 \\
        --output evals/dcfr-200_pool.json

Opponent spec format: name=path. Use path '__RANDOM__' for uniform random.
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Protocol

import pyspiel

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.stack_sampler import sample_starting_state
from src.nlhe.cfr6 import NUM_SEATS_6MAX, _build_view_6max
from src.nlhe.infoset6 import parse_state_6max, parse_state_repeated_6max
from src.nlhe.actions import discretize_legal_actions
from src.nlhe.icm_returns import icm_adjust_returns

# Reuse from existing eval so we don't drift.
from scripts.eval_6max_self_play import (
    _load_solver,
    _sample_action_from_policy as _sample_action_from_solver,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_pool")


# ---------------- Policy types ----------------

class Policy(Protocol):
    name: str
    def select_action(self, parsed, state, rng, mode: str) -> int: ...


class CheckpointPolicy:
    """Wraps a saved DeepCFR6MaxSolver checkpoint."""
    def __init__(self, name: str, ckpt_path: str, abstraction, structure):
        self.name = name
        self.ckpt_path = ckpt_path
        self.solver = _load_solver(ckpt_path, abstraction, structure)

    def select_action(self, parsed, state, rng, mode: str = "sample") -> int:
        return _sample_action_from_solver(self.solver, parsed, state, rng, mode=mode)


class UniformRandomPolicy:
    """Uniform over the same discrete action space the trained bots use.

    Falls back to uniform over raw legal chip actions if the discretizer
    returns nothing (e.g. at chance nodes — though we shouldn't reach this
    here since chance nodes are handled separately in the rollout).
    """
    def __init__(self, name: str = "random"):
        self.name = name

    def select_action(self, parsed, state, rng, mode: str = "sample") -> int:
        legal_chip = list(state.legal_actions())
        view = _build_view_6max(state, parsed)
        discrete_to_chip = discretize_legal_actions(legal_chip, view)
        if not discrete_to_chip:
            return rng.choice(legal_chip)
        chosen_da = rng.choice(list(discrete_to_chip.keys()))
        return int(discrete_to_chip[chosen_da])


# ---------------- Rollout ----------------

def play_one_hand_two_policies(
    policy_a: Policy,
    policy_b: Policy,
    structure: TournamentStructure,
    rng: random.Random,
    num_paid: int = 3,
    mode: str = "sample",
) -> dict:
    """Play one hand with each seat independently assigned to A or B."""
    sampled = sample_starting_state(structure, rng, num_paid=num_paid)
    gs = structure.to_inner_game_string_for_state(
        blind_level=sampled["blind_level"],
        stacks=sampled["stacks"],
        dealer_seat=sampled["dealer_seat"],
    )
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    starting_stacks = list(sampled["stacks"])

    seat_assignment = [rng.choice(["A", "B"]) for _ in range(NUM_SEATS_6MAX)]
    seat_to_policy = [policy_a if c == "A" else policy_b for c in seat_assignment]

    max_steps = 500
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
        if hasattr(state, "dealer_seat"):
            parsed = parse_state_repeated_6max(state)
        else:
            parsed = parse_state_6max(state)
        cp = parsed["current_player"]
        a = seat_to_policy[cp].select_action(parsed, state, rng, mode=mode)
        state.apply_action(a)

    if not state.is_terminal():
        return {
            "seat_assignment": seat_assignment,
            "seat_to_equity_delta": [0.0] * NUM_SEATS_6MAX,
            "exceeded_cap": True,
        }

    chip_returns = state.returns()
    payouts = [2.0] * num_paid
    equity_delta = icm_adjust_returns(
        chip_returns=chip_returns,
        starting_stacks=starting_stacks,
        payouts=payouts,
    )
    return {
        "seat_assignment": seat_assignment,
        "seat_to_equity_delta": equity_delta,
        "exceeded_cap": False,
    }


# ---------------- Matchup eval ----------------

def evaluate_matchup(
    challenger: Policy,
    opponent: Policy,
    structure: TournamentStructure,
    hands: int,
    seed: int,
    mode: str = "sample",
    log_every: int = 500,
) -> dict:
    rng = random.Random(seed)
    a_total = 0.0
    a_squared = 0.0
    b_total = 0.0
    n_hands = 0
    n_capped = 0
    t0 = time.time()

    for h in range(1, hands + 1):
        result = play_one_hand_two_policies(
            challenger, opponent, structure, rng, mode=mode
        )
        if result["exceeded_cap"]:
            n_capped += 1
            continue
        seat_assignment = result["seat_assignment"]
        equity = result["seat_to_equity_delta"]
        n_a = seat_assignment.count("A")
        n_b = seat_assignment.count("B")
        a_hand = sum(equity[s] for s, c in enumerate(seat_assignment) if c == "A")
        b_hand = sum(equity[s] for s, c in enumerate(seat_assignment) if c == "B")
        if n_a > 0:
            per_seat_a = a_hand / n_a
            a_total += per_seat_a
            a_squared += per_seat_a ** 2
        if n_b > 0:
            b_total += b_hand / n_b
        n_hands += 1

        if h % log_every == 0 or h == hands:
            avg_a = a_total / n_hands if n_hands else 0.0
            avg_b = b_total / n_hands if n_hands else 0.0
            if n_hands > 1:
                var_a = (a_squared / n_hands) - avg_a ** 2
                stderr_a = math.sqrt(max(0.0, var_a) / n_hands)
            else:
                stderr_a = 0.0
            elapsed = time.time() - t0
            log.info(
                f"  [{challenger.name} vs {opponent.name}] "
                f"{h:>5}/{hands}  diff={avg_a - avg_b:>+7.4f} +/- {stderr_a:.4f}  "
                f"capped={n_capped}  [{elapsed:.1f}s, {h/max(elapsed,0.001):.1f} h/s]"
            )

    avg_a = a_total / n_hands if n_hands else 0.0
    avg_b = b_total / n_hands if n_hands else 0.0
    var_a = (a_squared / n_hands) - avg_a ** 2 if n_hands > 1 else 0.0
    stderr_a = math.sqrt(max(0.0, var_a) / n_hands) if n_hands > 1 else 0.0
    diff = avg_a - avg_b
    sigma = abs(diff) / stderr_a if stderr_a > 0 else float("nan")
    return {
        "opponent": opponent.name,
        "n_hands": n_hands,
        "n_capped": n_capped,
        "avg_challenger": avg_a,
        "avg_opponent": avg_b,
        "diff": diff,
        "stderr": stderr_a,
        "sigma": sigma,
    }


# ---------------- CLI ----------------

def _parse_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name, path
    p = Path(spec)
    name = f"{p.parent.parent.name}/{p.stem}"
    return name, spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--challenger", required=True, help="name=path")
    p.add_argument("--opponents", nargs="+", required=True,
                   help="list of name=path; use path '__RANDOM__' for uniform random")
    p.add_argument("--abstraction", required=True)
    p.add_argument("--structure", required=True)
    p.add_argument("--hands", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="sample", choices=["sample", "argmax"])
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--output", default=None, help="JSON results path")
    args = p.parse_args()

    abstr = Abstraction.load(args.abstraction)
    structure = TournamentStructure.from_yaml(args.structure)

    challenger_name, challenger_path = _parse_spec(args.challenger)
    log.info(f"Loading challenger: {challenger_name} from {challenger_path}")
    challenger = CheckpointPolicy(challenger_name, challenger_path, abstr, structure)

    opponents: list[Policy] = []
    for spec in args.opponents:
        name, path = _parse_spec(spec)
        if path == "__RANDOM__":
            log.info(f"Opponent: {name} (uniform random)")
            opponents.append(UniformRandomPolicy(name))
        else:
            log.info(f"Loading opponent: {name} from {path}")
            opponents.append(CheckpointPolicy(name, path, abstr, structure))

    log.info(
        f"Pool eval: challenger={challenger_name}, {len(opponents)} opponents, "
        f"{args.hands} hands/matchup, mode={args.mode}"
    )

    results = []
    for i, opp in enumerate(opponents):
        log.info("=" * 60)
        log.info(f"Matchup: {challenger.name} vs {opp.name}")
        r = evaluate_matchup(
            challenger, opp, structure, args.hands,
            seed=args.seed + i * 1000,
            mode=args.mode,
            log_every=args.log_every,
        )
        results.append(r)

    log.info("=" * 60)
    log.info(f"SUMMARY: {challenger_name} vs pool ({args.hands} hands each, mode={args.mode})")
    log.info(f"  {'opponent':<28} {'diff':>10} {'stderr':>8} {'sigma':>6} {'capped':>7}")
    for r in results:
        sigma_str = f"{r['sigma']:>6.1f}" if math.isfinite(r['sigma']) else "   nan"
        log.info(
            f"  {r['opponent']:<28} {r['diff']:>+10.4f} "
            f"{r['stderr']:>8.4f} {sigma_str} {r['n_capped']:>7}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "challenger": {"name": challenger_name, "path": challenger_path},
            "n_hands_per_matchup": args.hands,
            "mode": args.mode,
            "base_seed": args.seed,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        log.info(f"Saved JSON results to {out_path}")


if __name__ == "__main__":
    main()
