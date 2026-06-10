"""C3 retrain launcher — 237-d encoder + empirical distribution + clean ante.

Recipe = the deployed k200_real_ante recipe with exactly the three
operator-approved C3 deltas (see configs/c3_retrain_k200.yaml). Fresh
start, 2000-iteration cap, checkpoint every 100.

PRE-TRAIN GATES (abort on any failure, mirroring train_k200_real_ante's
real-ante check):
  G1. L1 min-raise: the first decision of a fresh L1 hand must offer
      min raise-to = 2 x real_BB = 50 chips.
  G2. Ante correctness under the convention, full-ring AND shorthanded:
      per-seat ante arrays in the emitted game strings must charge every
      alive seat exactly the level ante and busted seats NOTHING, at
      every level of the schedule.
  G3. Empirical artifact: version stamp matches, its recorded H1 gate is
      PASS, and every row's n_alive >= num_paid + 1.
  G4. Encoder: feature_dim must be 237 with the eff-BB channel live.

Monitoring (external, read-only — start after launch):
  - scripts/monitor_k200_convergence.py --run-dir <out> --train-log <log>
      (per-checkpoint premium-fold / jam-share / depth-gap CSV)
  - scripts/c3_premium_fold_alert.py --csv <out>/convergence.csv
      (ALERT if AA/KK fold% > 10% past iter 700)
  - scripts/slope_h2h_k200.py for the 5,000-hand paired slope evals and
      the iter-1500-style stop call (DECISIONS.md "k200_real_ante
      iter-1500 stop").
"""
from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyspiel
import yaml

from src.nlhe.game_strings import PokerGameConfig, TournamentStructure
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_c3")

_RE_BLIND = re.compile(r"blind=([\d ]+),")
_RE_ANTE = re.compile(r"ante=([\d ]+),")


def gate_min_raise(structure) -> None:
    gs = structure.to_inner_game_string_for_state(
        blind_level=structure.level(1), stacks=(1500,) * 6, dealer_seat=0)
    state = pyspiel.load_game(gs).new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    raises = [a for a in state.legal_actions() if a >= 2]
    expected = 2 * structure.level(1).big_blind
    if not raises or min(raises) != expected or expected != 50:
        log.error(f"GATE G1 FAILED: L1 min_raise={min(raises) if raises else None}, "
                  f"expected 50 = 2 x real_BB. Aborting.")
        sys.exit(1)
    log.info("GATE G1 PASSED: L1 min_raise = 50 = 2 x real_BB")


def gate_ante_correctness(structure) -> None:
    cases = [
        ([100_000] * 6, 6),                                  # full ring
        ([100_000, 100_000, 0, 100_000, 100_000, 0], 4),     # shorthanded
        ([100_000, 0, 100_000, 100_000, 100_000, 0], 4),
    ]
    for bl in structure.blind_schedule:
        for stacks, n_alive in cases:
            gs = structure.to_inner_game_string_for_state(
                blind_level=bl, stacks=stacks, dealer_seat=0)
            blinds = [int(x) for x in _RE_BLIND.search(gs).group(1).split()]
            antes = [int(x) for x in _RE_ANTE.search(gs).group(1).split()]
            for i in range(6):
                want = bl.ante if stacks[i] > 0 else 0
                if antes[i] != want:
                    log.error(f"GATE G2 FAILED: L{bl.level} stacks={stacks} "
                              f"seat {i} ante={antes[i]}, want {want}")
                    sys.exit(1)
            dead = sum(blinds) + sum(antes)
            want_dead = bl.small_blind + bl.big_blind + n_alive * bl.ante
            if dead != want_dead:
                log.error(f"GATE G2 FAILED: L{bl.level} dead money {dead} != "
                          f"{want_dead}")
                sys.exit(1)
    log.info(f"GATE G2 PASSED: per-seat antes correct at all "
             f"{len(structure.blind_schedule)} levels x {len(cases)} "
             f"alive-patterns (busted seats post nothing)")


def gate_artifact(solver, num_paid: int) -> None:
    import gzip, json
    p = solver.cfg.empirical_dist_path
    raw = gzip.open(p, "rb").read() if p.endswith(".gz") else open(p, "rb").read()
    art = json.loads(raw)
    if not str(art.get("version", "")).startswith("training_dist_v"):
        log.error(f"GATE G3 FAILED: unversioned artifact {p}")
        sys.exit(1)
    h1 = art.get("h1_gate", {})
    if not h1.get("pass", False):
        log.error(f"GATE G3 FAILED: artifact H1 gate not PASS: {h1}")
        sys.exit(1)
    bad = sum(1 for r in solver.empirical_rows
              if r["n_alive"] < num_paid + 1)
    if bad:
        log.error(f"GATE G3 FAILED: {bad} rows with n_alive < {num_paid + 1}")
        sys.exit(1)
    log.info(f"GATE G3 PASSED: {art['version']}, "
             f"{len(solver.empirical_rows)} rows, H1 "
             f"L6+={100 * h1['l6plus_mass']:.2f}% (pass), all rows "
             f"n_alive >= {num_paid + 1}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/c3_retrain_k200.yaml")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override n_iterations (benchmark: --iterations 1)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.pop("tag", None)
    abstraction_path = cfg_dict.pop("abstraction_path")
    checkpoint_every = (args.checkpoint_every
                        if args.checkpoint_every is not None
                        else cfg_dict.pop("checkpoint_every", 100))
    cfg_dict.pop("checkpoint_every", None)
    if args.iterations is not None:
        cfg_dict["n_iterations"] = args.iterations

    cfg = TrainConfig6Max(**cfg_dict)
    if not cfg.encoder_eff_bb or cfg.ante_convention != "real" \
            or not cfg.empirical_dist_path:
        log.error("config must set the three C3 deltas "
                  "(encoder_eff_bb / ante_convention=real / "
                  "empirical_dist_path). Aborting.")
        sys.exit(1)
    log.info(f"C3 config: iters={cfg.n_iterations} "
             f"traversals={cfg.traversals_per_iter} "
             f"train_steps={cfg.train_steps_per_iter} "
             f"hidden={cfg.hidden_dim} ckpt_every={checkpoint_every}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(f"runs/c3_retrain_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        f"# C3 retrain: 237-d encoder + empirical distribution + clean ante\n"
        f"config: {cfg_dict}\n"
        f"abstraction_path: {abstraction_path}\n"
        f"checkpoint_every: {checkpoint_every}\n"
        f"started: {datetime.now().isoformat()}\n")
    log.info(f"Output dir: {out_dir}")

    log.info("Loading abstraction (ante-invariant, reused from k200)...")
    with open(abstraction_path, "rb") as f:
        abstraction = pickle.load(f)
    structure = TournamentStructure.from_yaml(cfg.tournament_structure_path)

    # ---- Pre-train gates (abort on failure) ----
    gate_min_raise(structure)
    gate_ante_correctness(structure)

    game = pyspiel.load_game(
        PokerGameConfig(num_players=6, starting_stack=cfg.starting_stack,
                        big_blind=cfg.big_blind, small_blind=cfg.small_blind
                        ).to_universal_poker_string())
    solver = DeepCFR6MaxSolver(game=game, abstraction=abstraction, config=cfg)
    solver.tournament_structure = structure

    gate_artifact(solver, cfg.num_paid)
    if solver.encoder.feature_dim != 237 or not solver.encoder.include_eff_bb:
        log.error(f"GATE G4 FAILED: feature_dim={solver.encoder.feature_dim}, "
                  f"include_eff_bb={solver.encoder.include_eff_bb}. Aborting.")
        sys.exit(1)
    log.info("GATE G4 PASSED: encoder feature_dim=237, eff-BB channel live")
    log.info("ALL PRE-TRAIN GATES PASSED.")

    log.info(f"Training {cfg.n_iterations} iterations, "
             f"checkpoint every {checkpoint_every}...")
    t0 = time.time()
    metrics = solver.train(checkpoint_dir=out_dir,
                           checkpoint_every=checkpoint_every)
    elapsed = time.time() - t0
    n_done = len(metrics.get("iter", [])) if metrics else 0
    log.info(f"Training done. wall_time={elapsed:.0f}s "
             f"({elapsed / 3600:.2f}h), iters={n_done}, "
             f"sec/iter={elapsed / max(1, n_done):.1f}")
    log.info(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
