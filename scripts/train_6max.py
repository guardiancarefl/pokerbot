"""Train Deep CFR on 6-max NLHE with card and action abstraction.

Usage:
    python -m scripts.train_6max --config configs/six_max_smoke.yaml
    python -m scripts.train_6max --config configs/six_max_blueprint.yaml \\
        --resume runs/six_max_<ts>_blueprint/checkpoints/ckpt_iter_0050.pt

Output structure (mirrors train_nlhe.py):
    runs/six_max_<timestamp>_<tag>/
        config.json         (effective config, saved at start)
        metrics.json        (per-iteration metrics, written on completion)
        checkpoints/
            ckpt_iter_XXXX.pt

The script intentionally fails loudly if --config references a missing
abstraction artifact. Build the abstraction with scripts/train_abstraction.py
first (it's reused unchanged from the HUNL pipeline — same pickle works
for 6-max because the per-street card abstraction is a function of (hero
cards, board cards), independent of player count).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pyspiel
import yaml

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import PokerGameConfig
from src.nlhe.parallel.orchestrator import parallel_train
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_6max")


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_six_max_game(cfg: TrainConfig6Max) -> "pyspiel.Game":
    """Build the OpenSpiel 6-max universal_poker game from a TrainConfig6Max."""
    return pyspiel.load_game(_six_max_game_str(cfg))


def _six_max_game_str(cfg: TrainConfig6Max) -> str:
    """The universal_poker string for a TrainConfig6Max — needed by
    parallel_train so workers can re-load pyspiel.Game locally (Game does
    not pickle)."""
    game_cfg = PokerGameConfig(
        num_players=6,
        starting_stack=cfg.starting_stack,
        big_blind=cfg.big_blind,
        small_blind=cfg.small_blind,
    )
    return game_cfg.to_universal_poker_string()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--resume", default=None, help="path to ckpt .pt to resume from")
    parser.add_argument("--out", default=None, help="output run dir (default runs/six_max_<ts>_<tag>)")
    args = parser.parse_args()

    cfg_dict = load_yaml_config(args.config)
    tag = cfg_dict.pop("tag", Path(args.config).stem)
    abstraction_path = cfg_dict.pop("abstraction_path")
    checkpoint_every = cfg_dict.pop("checkpoint_every", 10)

    # Remaining keys go into TrainConfig6Max. This will raise a clear
    # TypeError if the YAML has an unknown key.
    tc = TrainConfig6Max(**cfg_dict)
    log.info(f"TrainConfig6Max: {tc}")

    if args.out:
        run_dir = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f"runs/six_max_{ts}_{tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    log.info(f"output dir: {run_dir}")

    # Save effective config alongside the run for traceability.
    save_cfg = {
        **asdict(tc),
        "abstraction_path": abstraction_path,
        "checkpoint_every": checkpoint_every,
        "tag": tag,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(save_cfg, f, indent=2)
    log.info(f"saved: {run_dir / 'config.json'}")

    log.info("loading abstraction...")
    abst = Abstraction.load(abstraction_path)
    for street, sa in abst.streets.items():
        log.info(f"  {street}: k={sa.k}, bins={sa.bins}")

    log.info("building 6-max game...")
    game_str = _six_max_game_str(tc)
    game = pyspiel.load_game(game_str)
    log.info(f"  num_players={game.num_players()}  max_length={game.max_game_length()}")

    solver = DeepCFR6MaxSolver(
        game=game, abstraction=abst, config=tc, logger=log.info,
    )

    if args.resume:
        log.info(f"resuming from checkpoint: {args.resume}")
        solver.load_checkpoint(args.resume)
        log.info(f"  resumed at iteration {solver.iteration}")

    t0 = time.time()
    interrupted = False
    metrics = None
    try:
        if tc.parallel_groups > 0:
            log.info(
                f"parallel mode: G={tc.parallel_groups} "
                f"use_processes={tc.parallel_use_processes}"
            )
            metrics = parallel_train(
                solver,
                game_str=game_str,
                abstraction_path=abstraction_path,
                n_workers=tc.parallel_groups,
                use_processes=tc.parallel_use_processes,
                checkpoint_dir=ckpt_dir,
                checkpoint_every=checkpoint_every,
            )
        else:
            metrics = solver.train(
                checkpoint_dir=ckpt_dir, checkpoint_every=checkpoint_every
            )
    except KeyboardInterrupt:
        # Ctrl+C from the tmux session (or any other SIGINT). solver.train()
        # internal state is reachable via solver.iteration etc., so we can
        # checkpoint at whatever iter we were on and exit cleanly. Subsequent
        # work (loading the checkpoint, re-running eval, etc.) treats this run
        # as ending at solver.iteration.
        interrupted = True
        log.info(f"=== INTERRUPTED at iter {solver.iteration} — saving final checkpoint ===")
        final_ckpt = ckpt_dir / f"ckpt_iter_{solver.iteration:04d}.pt"
        try:
            solver.save_checkpoint(final_ckpt, slim=True)
            log.info(f"  saved final checkpoint: {final_ckpt}")
        except Exception as e:
            log.warning(f"  final checkpoint save FAILED: {e}")
    total = time.time() - t0
    log.info(
        f"=== Total training time: {total/60:.1f} min "
        f"({'INTERRUPTED' if interrupted else 'completed'}) ==="
    )

    # Always write a metrics.json so downstream tooling can see how far the
    # run got. On interrupt, metrics is None (solver.train() didn't return);
    # write a stub with interrupted flag so a partial run is distinguishable
    # from a successful one without reading checkpoint files.
    if metrics is None:
        metrics = {
            "iter": [], "time": [], "traverser": [],
            "adv_loss": [], "strat_loss": [], "strat_buf": [],
            "mini_eval": [],
        }
        for s in range(6):
            metrics[f"buf_{s}"] = []
    metrics["interrupted"] = interrupted
    metrics["last_iteration"] = solver.iteration
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"saved: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
