"""Resume a k200 real-ante run from a checkpoint, training through a
target iteration. Production-grade (full traversals_per_iter from
the config, NOT throttled). Used to recover from the O_APPEND-fix
restart on 2026-06-06 without losing the iter_100 work.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyspiel
import yaml

from src.nlhe.game_strings import TournamentStructure, PokerGameConfig
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("continue_k200_real_ante")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/six_max_phase4f_dcfr_candC_k200.yaml")
    ap.add_argument("--resume", required=True, help="path to ckpt to resume from")
    ap.add_argument("--target-iter", type=int, default=2000)
    ap.add_argument("--out", default=None,
                    help="run dir to write new ckpts to (default: same dir as --resume)")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.pop("tag", None)
    abstraction_path = cfg_dict.pop("abstraction_path")
    cfg_dict["n_iterations"] = args.target_iter
    cfg_dict.pop("checkpoint_every", None)
    cfg_dict.pop("parallel_groups", None)
    cfg_dict.pop("parallel_use_processes", None)

    cfg = TrainConfig6Max(**cfg_dict)
    log.info(f"Resuming from: {args.resume}")
    log.info(f"Target iter:   {args.target_iter}")
    log.info(f"Config: traversals_per_iter={cfg.traversals_per_iter}, "
             f"train_steps_per_iter={cfg.train_steps_per_iter}, "
             f"hidden_dim={cfg.hidden_dim}")

    out_dir = Path(args.out) if args.out else Path(args.resume).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {out_dir}")

    log.info("Loading abstraction...")
    with open(abstraction_path, "rb") as f:
        abstraction = pickle.load(f)

    structure = TournamentStructure.from_yaml(cfg.tournament_structure_path)

    game = pyspiel.load_game(
        PokerGameConfig(num_players=6, starting_stack=cfg.starting_stack,
                        big_blind=cfg.big_blind, small_blind=cfg.small_blind
                        ).to_universal_poker_string())
    solver = DeepCFR6MaxSolver(game=game, abstraction=abstraction, config=cfg)
    solver.tournament_structure = structure

    log.info(f"Loading checkpoint state...")
    solver.load_checkpoint(args.resume)
    log.info(f"Resumed at iteration {solver.iteration} "
             f"(will train through {args.target_iter}). "
             f"Encoder feature_dim={solver.encoder.feature_dim} (expected 236).")

    t0 = time.time()
    metrics = solver.train(checkpoint_dir=out_dir,
                            checkpoint_every=args.checkpoint_every)
    elapsed = time.time() - t0
    log.info(f"Training done. wall_time={elapsed:.0f}s ({elapsed/3600:.2f}h)")
    n_done = len(metrics.get("iter", [])) if metrics else 0
    log.info(f"Iterations completed in this resume: {n_done}")


if __name__ == "__main__":
    main()
