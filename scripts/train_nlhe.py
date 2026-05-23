"""Train Deep CFR on HUNL with card and action abstraction.

Usage:
    python -m scripts.train_nlhe --config configs/nlhe_smoke.yaml
    python -m scripts.train_nlhe --config configs/nlhe_phase2b.yaml --resume runs/.../ckpt_iter_0050.pt

Output structure (mirrors Leduc):
    runs/nlhe_<timestamp>_<tag>/
        config.json
        metrics.json (updated each iteration)
        checkpoints/
            ckpt_iter_XXXX.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pyspiel
import yaml

from src.nlhe.abstraction import Abstraction
from src.nlhe.solver import DeepCFRSolver, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_nlhe")


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_game(
    game_str: str | None,
    num_players: int | None = None,
    starting_stack: int | None = None,
    big_blind: int | None = None,
    small_blind: int | None = None,
) -> "pyspiel.Game":
    """Load a universal_poker game.

    Two ways to specify the game:
      1. Structured params (num_players + starting_stack + big_blind + small_blind):
         preferred path. Uses PokerGameConfig from src.nlhe.game_strings.
      2. Legacy game_str: a raw OpenSpiel game string. Backward compat for
         configs that pre-date the structured fields.

    If both are provided, structured params win and game_str is ignored.
    If neither is provided, defaults to HUNL 200bb (Phase 2d shape).

    Note: starting_stack here is the GAME's per-player starting chip count
    (same numeric value as TrainConfig.starting_stack used by the solver).
    """
    if num_players is not None:
        from src.nlhe.game_strings import PokerGameConfig
        cfg = PokerGameConfig(
            num_players=num_players,
            starting_stack=starting_stack if starting_stack is not None else 20000,
            big_blind=big_blind if big_blind is not None else 100,
            small_blind=small_blind if small_blind is not None else 50,
        )
        game_str = cfg.to_universal_poker_string()
    elif game_str is None:
        game_str = (
            "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 50,"
            "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,"
            "numBoardCards=0 3 1 1,stack=20000 20000,bettingAbstraction=fullgame)"
        )
    return pyspiel.load_game(game_str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--resume", default=None, help="path to checkpoint .pt to resume from")
    parser.add_argument("--out", default=None, help="output run dir (default runs/nlhe_<timestamp>_<tag>)")
    args = parser.parse_args()

    cfg_dict = load_yaml_config(args.config)
    tag = cfg_dict.pop("tag", Path(args.config).stem)
    abstraction_path = cfg_dict.pop("abstraction_path")
    game_str = cfg_dict.pop("game_str", None)
    checkpoint_every = cfg_dict.pop("checkpoint_every", 5)

    # Remaining keys go to TrainConfig.
    tc = TrainConfig(**cfg_dict)
    log.info(f"TrainConfig: {tc}")

    if args.out:
        run_dir = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f"runs/nlhe_{ts}_{tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    log.info(f"output dir: {run_dir}")

    # Save effective config alongside the run.
    save_cfg = {**tc.__dict__, "abstraction_path": abstraction_path, "game_str": game_str,
                "num_players": num_players, "big_blind": big_blind, "small_blind": small_blind,
                "checkpoint_every": checkpoint_every, "tag": tag}
    with open(run_dir / "config.json", "w") as f:
        json.dump(save_cfg, f, indent=2)

    log.info("loading abstraction...")
    abst = Abstraction.load(abstraction_path)
    for street, sa in abst.streets.items():
        log.info(f"  {street}: k={sa.k}, bins={sa.bins}")

    log.info("loading game...")
    game = build_game(game_str, num_players=num_players, starting_stack=starting_stack_game,
                      big_blind=big_blind, small_blind=small_blind)
    log.info(f"  max_game_length={game.max_game_length()}")

    solver = DeepCFRSolver(game=game, abstraction=abst, config=tc, logger=log.info)
    log.info(f"  feature_dim={solver.encoder.feature_dim}")

    if args.resume:
        log.info(f"resuming from checkpoint: {args.resume}")
        solver.load_checkpoint(args.resume)
        log.info(f"  resumed at iteration {solver.iteration}")

    t0 = time.time()
    metrics = solver.train(checkpoint_dir=ckpt_dir, checkpoint_every=checkpoint_every)
    total = time.time() - t0
    log.info(f"=== Total training time: {total/60:.1f} min ===")

    # Persist metrics.
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"saved: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
