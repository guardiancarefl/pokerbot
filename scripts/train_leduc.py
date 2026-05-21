"""Phase 1 entry point: Deep CFR on Leduc Poker.

Loads a YAML config, applies CLI overrides, runs training, evaluates
exploitability, and writes everything to a timestamped run directory.

Usage:
    python scripts/train_leduc.py
        # Uses configs/leduc_default.yaml.

    python scripts/train_leduc.py --config configs/leduc_smoke.yaml
        # Fast smoke test.

    python scripts/train_leduc.py --iterations 200 --learning-rate 0.0005
        # Override individual fields from CLI.

    python scripts/train_leduc.py --config configs/leduc_default.yaml \\
        --run-name big_experiment
        # Override and add a name suffix to the run dir.

Output goes to runs/leduc_<timestamp>[_<run_name>]/ with:
    config.json              (effective config used)
    metrics.json             (final metrics summary)
    training.log             (stdout-mirrored log)
    advantage_losses.csv     (per-iteration, per-player advantage losses)
    checkpoints/final.pt     (trained policy network + config + metrics)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import sys
from pathlib import Path

# Make src importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pyspiel  # noqa: E402
import torch  # noqa: E402

from src.leduc.checkpoint import save_checkpoint  # noqa: E402
from src.leduc.config import TrainConfig  # noqa: E402
from src.leduc.evaluate import exploitability_mbb  # noqa: E402
from src.leduc.solver import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: Deep CFR on Leduc Poker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path,
                   default=PROJECT_ROOT / "configs" / "leduc_default.yaml",
                   help="Path to YAML config file.")

    # Optional overrides for every TrainConfig field. None = use YAML value.
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--num-traversals", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--memory-capacity", type=int, default=None)
    p.add_argument("--advantage-network-layers", type=int, nargs="+", default=None)
    p.add_argument("--policy-network-layers", type=int, nargs="+", default=None)
    p.add_argument("--advantage-train-steps", type=int, default=None)
    p.add_argument("--policy-train-steps", type=int, default=None)
    p.add_argument("--batch-size-advantage", type=int, default=None)
    p.add_argument("--batch-size-strategy", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--skip-exploitability", action="store_true", default=None,
                   help="If set, skip the (slow) exploitability eval at end.")
    p.add_argument("--run-name", type=str, default=None,
                   help="Suffix appended to timestamped run directory name.")

    return p.parse_args()


def cli_overrides(args: argparse.Namespace) -> dict:
    """Extract non-None CLI args as a TrainConfig override dict.

    argparse uses underscores; TrainConfig fields use underscores too,
    so the mapping is direct (no hyphen translation needed).
    """
    override_keys = [
        "iterations", "num_traversals", "learning_rate", "memory_capacity",
        "advantage_network_layers", "policy_network_layers",
        "advantage_train_steps", "policy_train_steps",
        "batch_size_advantage", "batch_size_strategy",
        "seed", "skip_exploitability", "run_name",
    ]
    return {k: getattr(args, k) for k in override_keys}


def make_run_dir(run_name: str | None) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"leduc_{ts}" if not run_name else f"leduc_{ts}_{run_name}"
    run_dir = PROJECT_ROOT / "runs" / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logger = logging.getLogger("leduc_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for h in [
        logging.FileHandler(run_dir / "training.log"),
        logging.StreamHandler(sys.stdout),
    ]:
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
    return logger


def write_advantage_losses_csv(path: Path, advantage_losses: dict) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "player", "advantage_loss"])
        for player_idx, losses in advantage_losses.items():
            for it_idx, loss in enumerate(losses, start=1):
                w.writerow([it_idx, player_idx, f"{loss:.6f}"])


def main() -> int:
    args = parse_args()

    # Load YAML config, apply CLI overrides.
    base_config = TrainConfig.from_yaml(args.config)
    overrides = cli_overrides(args)
    config = base_config.merge_overrides(overrides)

    run_dir = make_run_dir(config.run_name)
    logger = setup_logging(run_dir)

    logger.info(f"run dir: {run_dir}")
    logger.info(f"config source: {args.config}")
    logger.info(f"effective config: {config.to_dict()}")
    logger.info(f"torch: {torch.__version__}")

    game = pyspiel.load_game("leduc_poker")
    logger.info(f"game: leduc_poker, players={game.num_players()}, "
                f"max_length={game.max_game_length()}")

    logger.info(f"starting solve(): {config.iterations} iterations x "
                f"{config.num_traversals} traversals/player")
    result = train(game, config)
    logger.info(f"solve() complete in {result.train_seconds/60:.2f} min "
                f"({result.train_seconds:.1f}s)")
    logger.info(f"final policy loss: {result.policy_loss:.6f}")

    # Advantage loss trajectory summary per player.
    for p_idx, losses in result.advantage_losses.items():
        if losses:
            logger.info(
                f"  player {p_idx}: adv_loss first={losses[0]:.4f}, "
                f"last={losses[-1]:.4f}, min={min(losses):.4f}, "
                f"n_iters={len(losses)}"
            )

    write_advantage_losses_csv(run_dir / "advantage_losses.csv",
                               result.advantage_losses)

    # Exploitability evaluation.
    metrics: dict = {
        "iterations": config.iterations,
        "num_traversals": config.num_traversals,
        "final_policy_loss": result.policy_loss,
        "train_seconds": result.train_seconds,
        "exploitability_mbb_per_game": None,
    }

    if config.skip_exploitability:
        logger.info("skipping exploitability eval (skip_exploitability=true)")
    else:
        logger.info("computing exploitability (full tabular game-tree traversal)...")
        import time
        t0 = time.time()
        expl = exploitability_mbb(game, result.action_probabilities_fn)
        eval_secs = time.time() - t0
        metrics["exploitability_mbb_per_game"] = expl
        metrics["exploitability_seconds"] = eval_secs
        logger.info(f"exploitability: {expl:.3f} mbb/g (eval took {eval_secs:.1f}s)")

    # Save checkpoint + config.json + metrics.json (companions written automatically).
    ckpt_path = run_dir / "checkpoints" / "final.pt"
    save_checkpoint(ckpt_path, result.policy_network, config.to_dict(), metrics)
    logger.info(f"checkpoint saved: {ckpt_path}")
    logger.info(f"run dir: {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
