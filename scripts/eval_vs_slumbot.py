"""Evaluate a policy against Slumbot over N hands.

Usage:
    python -m scripts.eval_vs_slumbot --policy random --hands 100
    python -m scripts.eval_vs_slumbot --policy random --hands 1000 --seed 42

Reports both raw bb/100 (high variance) and baseline-adjusted bb/100 (Slumbot's
built-in variance reduction; the headline number).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from src.nlhe.slumbot_client import (
    BIG_BLIND,
    SlumbotClient,
    RandomPolicy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_vs_slumbot")


def play_one_hand(client: SlumbotClient, policy, max_steps: int = 50) -> dict:
    """Play one hand to completion. Returns the final state's summary fields."""
    state = client.new_hand()
    for _ in range(max_steps):
        if state.is_terminal:
            break
        action = policy.choose_action(state)
        state = client.act(action)
    return {
        "winnings": state.winnings,
        "baseline_winnings": state.baseline_winnings,
        "session_total": state.session_total,
        "session_baseline_total": state.session_baseline_total,
        "session_num_hands": state.session_num_hands,
        "final_action": state.action,
    }


def make_policy(name: str, seed: int):
    if name == "random":
        return RandomPolicy(seed=seed)
    raise ValueError(f"unknown policy: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="random", choices=["random"], help="policy to evaluate")
    p.add_argument("--hands", type=int, default=100, help="number of hands to play")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-every", type=int, default=25,
        help="print a running summary every N hands",
    )
    args = p.parse_args()

    log.info(f"Evaluating policy={args.policy} for {args.hands} hands (seed={args.seed})")
    client = SlumbotClient()
    policy = make_policy(args.policy, args.seed)

    t0 = time.time()
    winnings_sum = 0
    baseline_sum = 0.0
    n_completed = 0
    n_errors = 0

    for h in range(1, args.hands + 1):
        try:
            result = play_one_hand(client, policy)
            n_completed += 1
            if result["winnings"] is not None:
                winnings_sum += result["winnings"]
            if result["baseline_winnings"] is not None:
                baseline_sum += result["baseline_winnings"]
        except Exception as e:
            n_errors += 1
            log.warning(f"hand {h}: error {type(e).__name__}: {e}")
            # Reset the session token so the next hand starts cleanly
            client.token = None
            continue

        if h % args.log_every == 0 or h == args.hands:
            elapsed = time.time() - t0
            raw_bb100 = (winnings_sum / max(n_completed, 1)) / BIG_BLIND * 100
            delta = winnings_sum - baseline_sum
            adj_bb100 = (delta / max(n_completed, 1)) / BIG_BLIND * 100
            log.info(
                f"  hand {h:>5}/{args.hands}  "
                f"completed={n_completed} errors={n_errors}  "
                f"raw_bb100={raw_bb100:>+8.2f}  "
                f"baseline_adj_bb100={adj_bb100:>+8.2f}  "
                f"[{elapsed:.1f}s, {h/elapsed:.1f} hands/s]"
            )

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"Done. {n_completed} hands completed in {elapsed:.1f}s ({n_completed/elapsed:.1f} hands/s)")
    log.info(f"  errors: {n_errors}")
    if n_completed > 0:
        raw_bb100 = (winnings_sum / n_completed) / BIG_BLIND * 100
        adj_bb100 = ((winnings_sum - baseline_sum) / n_completed) / BIG_BLIND * 100
        log.info(f"  total winnings: {winnings_sum} chips")
        log.info(f"  total baseline: {baseline_sum:.1f} chips")
        log.info(f"  raw bb/100:               {raw_bb100:>+8.2f}  (high variance)")
        log.info(f"  baseline-adjusted bb/100: {adj_bb100:>+8.2f}  (headline metric)")
        # Sanity check expectations
        log.info("")
        if args.policy == "random":
            log.info("Expected for random policy: raw ~-200 bb/100, baseline-adjusted ~similar.")
            log.info("If both are near 0 or strongly positive, something's wrong in the harness.")


if __name__ == "__main__":
    main()
