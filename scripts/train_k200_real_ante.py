"""Production k200 retrain on the real-ante convention.

Recipe = the proven k200 recipe, changed ONLY by the real-ante convention.
The 8-cycle depth-distinction investigation resolved to two real fixes:
the inflated_BB action-set patch (already in the patched pyspiel) and
the argmax→sample deployment default. Everything else (sampler
rebalances, depth feature, capacity scare) was iter_500 under-convergence
artifacts.

Specifically NOT in this recipe (intentionally reverted to original):
  - NO sampler rebalance (training_weights + σ values at original)
  - NO depth feature (encoder back to 236 dims)
  - NO L1=0.30 oversampling
The only meaningful change from the validated k200 pipeline is the
real-ante game-string convention via the _RealAnteStructure wrapper —
which is what makes the patched-pyspiel min_bet = 2*real_BB available
instead of 2*inflated_BB.

Saves checkpoints every 100 iters so the saturation-point probe (~800)
has a target.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_k200_real_ante")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/six_max_phase4f_dcfr_candC_k200.yaml",
                    help="production k200 config (validated recipe)")
    ap.add_argument("--iterations", type=int, default=2000,
                    help="target iterations (matches the validated k200 = 2000)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.pop("tag", None)
    abstraction_path = cfg_dict.pop("abstraction_path")
    cfg_dict["n_iterations"] = args.iterations
    cfg_dict.pop("checkpoint_every", None)
    cfg_dict.pop("parallel_groups", None)
    cfg_dict.pop("parallel_use_processes", None)

    cfg = TrainConfig6Max(**cfg_dict)
    log.info(f"Loaded config: traversals_per_iter={cfg.traversals_per_iter}, "
             f"train_steps_per_iter={cfg.train_steps_per_iter}, "
             f"hidden_dim={cfg.hidden_dim}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(
        f"runs/k200_real_ante_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        f"# Production k200 retrain on real-ante convention\n"
        f"# Recipe: original distribution + 236-dim encoder + real-ante game strings\n"
        f"config: {cfg_dict}\n"
        f"abstraction_path: {abstraction_path}\n"
        f"checkpoint_every: {args.checkpoint_every}\n"
        f"started: {datetime.now().isoformat()}\n")
    log.info(f"Output dir: {out_dir}")

    log.info("Loading abstraction (reused from k200; ante-invariant)...")
    with open(abstraction_path, "rb") as f:
        abstraction = pickle.load(f)

    log.info(f"Loading TournamentStructure from {cfg.tournament_structure_path}")
    structure = TournamentStructure.from_yaml(cfg.tournament_structure_path)
    log.info("Loaded TournamentStructure (canonical real-ante game-string emitter)")

    # Sanity: emit one game string + verify min_bet = 50 at L1 with real ante.
    test_gs = structure.to_inner_game_string_for_state(
        blind_level=structure.level(1), stacks=(1500,)*6, dealer_seat=0)
    log.info(f"Sample L1 real-ante game string: {test_gs[:120]}...")
    test_game = pyspiel.load_game(test_gs)
    test_state = test_game.new_initial_state()
    while test_state.is_chance_node():
        test_state.apply_action(test_state.legal_actions()[0])
    test_legal = [a for a in test_state.legal_actions() if a >= 2]
    min_raise = min(test_legal) if test_legal else 0
    log.info(f"Pre-train sanity: min_raise at L1 = {min_raise} chips (expected 50 = 2 × real_BB)")
    if min_raise != 50:
        log.error(f"PRE-TRAIN SANITY FAILED: min_raise = {min_raise}, expected 50. Aborting.")
        sys.exit(1)
    log.info("Pre-train sanity: PASSED. Building solver...")

    game = pyspiel.load_game(
        PokerGameConfig(num_players=6, starting_stack=cfg.starting_stack,
                        big_blind=cfg.big_blind, small_blind=cfg.small_blind
                        ).to_universal_poker_string())
    solver = DeepCFR6MaxSolver(game=game, abstraction=abstraction, config=cfg)
    solver.tournament_structure = structure
    log.info(f"Solver ready. Encoder feature_dim={solver.encoder.feature_dim} "
             f"(expected 236).")
    log.info(f"Training {args.iterations} iterations, "
             f"checkpoint every {args.checkpoint_every}...")

    t0 = time.time()
    metrics = solver.train(
        checkpoint_dir=out_dir,
        checkpoint_every=args.checkpoint_every,
    )
    elapsed = time.time() - t0
    log.info(f"Training done. wall_time={elapsed:.0f}s ({elapsed/3600:.2f}h)")
    n_done = len(metrics.get("iter", [])) if metrics else 0
    log.info(f"Iterations completed: {n_done}")
    log.info(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
