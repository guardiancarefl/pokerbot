"""cProfile the DCFR 6-max solver.

Runs a small CPU-forced training loop under cProfile, dumps the .prof
file, and prints two views:
  - Top-30 by cumulative time (where time is spent, walking down the call tree)
  - Top-20 by self time (where work actually happens, excluding callees)

Cumulative tells you the chain leading to expensive work.
Self time tells you which functions to actually rewrite.

CPU-forced so we measure function costs, not CUDA sync waits. Re-profile
on GPU later if we want absolute speedup numbers.

Usage:
    python -m scripts.profile_solver --config configs/six_max_profile.yaml
"""
from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pstats
import time
from pathlib import Path

# Force CPU BEFORE importing torch-using modules.
import torch
_orig_is_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False

import yaml

from src.nlhe.abstraction import Abstraction
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("profile_solver")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-prof", default="/tmp/solver6.prof")
    parser.add_argument("--cumulative-top", type=int, default=30)
    parser.add_argument("--self-top", type=int, default=20)
    args = parser.parse_args()

    log.info(f"CPU forced: torch.cuda.is_available() = {torch.cuda.is_available()}")
    assert not torch.cuda.is_available(), "CPU patch failed"

    cfg_dict = yaml.safe_load(open(args.config))
    cfg_dict.pop("tag", None)
    abstraction_path = cfg_dict.pop("abstraction_path")
    cfg_dict.pop("checkpoint_every", None)
    tc = TrainConfig6Max(**cfg_dict)
    log.info(f"config: {tc}")

    log.info(f"loading abstraction: {abstraction_path}")
    abstr = Abstraction.load(abstraction_path)

    # Build game the same way train_6max does.
    import pyspiel
    from src.nlhe.game_strings import PokerGameConfig
    game = pyspiel.load_game(PokerGameConfig(
        num_players=6,
        starting_stack=tc.starting_stack,
        big_blind=tc.big_blind,
        small_blind=tc.small_blind,
    ).to_universal_poker_string())

    log.info("constructing solver (CPU)...")
    solver = DeepCFR6MaxSolver(game=game, abstraction=abstr, config=tc)

    log.info(f"profiling {tc.n_iterations} iters under cProfile...")
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    solver.train()
    pr.disable()
    elapsed = time.time() - t0
    log.info(f"profile complete: {elapsed:.1f}s for {tc.n_iterations} iters "
             f"({elapsed/tc.n_iterations:.2f}s/iter)")

    # Dump raw .prof for later analysis (snakeviz, etc.)
    Path(args.out_prof).parent.mkdir(parents=True, exist_ok=True)
    pr.dump_stats(args.out_prof)
    log.info(f"raw profile dumped to {args.out_prof}")
    log.info(f"  visualize: snakeviz {args.out_prof}  (if installed)")

    # Print top-N by cumulative time.
    print()
    print("=" * 80)
    print(f"Top {args.cumulative_top} by CUMULATIVE time")
    print("  (time spent in fn + everything it called; high cumtime = chain leading to expensive work)")
    print("=" * 80)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
    ps.print_stats(args.cumulative_top)
    print(s.getvalue())

    # Print top-N by self time (tottime).
    print("=" * 80)
    print(f"Top {args.self_top} by SELF time (tottime)")
    print("  (time spent IN fn, excluding callees; high tottime = function to actually rewrite)")
    print("=" * 80)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime")
    ps.print_stats(args.self_top)
    print(s.getvalue())

    print("=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("  Look for:")
    print("    - Traversal/CFR functions dominating cumtime -> parallel rollouts is the win")
    print("    - Adv-net training functions dominating cumtime -> batch/network tuning is the win")
    print("    - Encoder/abstraction functions in top-10 tottime -> overhead bug to fix BEFORE more compute")
    print("    - OpenSpiel state copies in top-10 tottime -> structural issue, hard to fix")
    print("    - torch primitives in top-10 tottime -> normal, focus on what calls them")


if __name__ == "__main__":
    main()
