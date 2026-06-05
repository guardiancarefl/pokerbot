"""THROWAWAY — Step 3 diagnosis confirmation training.

Trains a SHORT k200 blueprint using the new real-ante OpenSpiel patch
(real `ante` parameter, NOT the inflated_BB hack). Reuses the existing
abstraction (EMD-on-equity is ante-invariant — verified empirically:
abstraction.py + equity.py have zero references to ante/blind/chip
amounts).

Doesn't modify game_strings.py (the bridge depends on its existing
methods). Monkey-patches the TournamentStructure instance with a real-
ante variant of `to_inner_game_string_for_state` for the duration of
this script only. Bridge code on disk is untouched.

This is a SANITY check: does the new action set move the trained
model's preflop behavior away from 2.4% BTN / 0% CO open rates?
If yes → action-set distortion confirmed as root cause, full retrain
authorized.
If no → other factors at play, need more diagnosis before retrain.

Output: a single checkpoint at runs/throwaway_real_ante/<ts>/ckpt.pt
that the companion query script reads.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import types
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyspiel
import yaml

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("throwaway_train")


# ---------------------------------------------------------------------
# Real-ante game-string emitter — drop-in replacement for
# TournamentStructure.to_inner_game_string_for_state. Same signature,
# emits a game string using the new `ante` parameter rather than the
# inflated_BB encoding.
# ---------------------------------------------------------------------
def to_inner_game_string_for_state_real_ante(
        self, blind_level, stacks, dealer_seat) -> str:
    """Same contract as TournamentStructure.to_inner_game_string_for_state
    but emits a real-ante universal_poker game string (uses the patched
    pyspiel's native `ante=...` parameter, NOT bb + N×ante inflation).
    """
    n = self.num_players
    if len(stacks) != n:
        raise ValueError(f"stacks length {len(stacks)} != num_players {n}")
    if not (0 <= dealer_seat < n):
        raise ValueError(f"dealer_seat {dealer_seat} out of range [0,{n})")

    sb = blind_level.small_blind
    bb = blind_level.big_blind        # NOT inflated
    ante = blind_level.ante

    # Identify alive seats (stack > 0).
    alive_seats = [i for i, s in enumerate(stacks) if s > 0]
    if dealer_seat not in alive_seats:
        raise ValueError(
            f"dealer_seat {dealer_seat} is busted; alive={alive_seats}")
    n_alive = len(alive_seats)
    if n_alive < 2:
        raise ValueError(f"need >= 2 alive seats, got {n_alive}")

    dealer_pos = alive_seats.index(dealer_seat)
    sb_seat = alive_seats[(dealer_pos + 1) % n_alive]
    bb_seat = alive_seats[(dealer_pos + 2) % n_alive]

    # blinds: per-seat sb/bb at the chosen positions, 0 elsewhere
    blind_array = [0] * n
    blind_array[sb_seat] = sb
    blind_array[bb_seat] = bb

    # antes: per-alive seat ante, 0 for busted (which carry stack=1 placeholder).
    # Busted seats getting 0 ante satisfies ACPC's blind+ante <= stack guard.
    ante_array = [ante if stacks[i] > 0 else 0 for i in range(n)]

    blind_str = " ".join(str(b) for b in blind_array)
    ante_str = " ".join(str(a) for a in ante_array)

    if n_alive == 2:
        preflop_actor = bb_seat + 1
        postflop_actor = sb_seat + 1
    else:
        utg_alive_pos = (dealer_pos + 3) % n_alive
        utg_seat = alive_seats[utg_alive_pos]
        preflop_actor = utg_seat + 1
        postflop_actor = sb_seat + 1
    first_player = (f"{preflop_actor} {postflop_actor} "
                    f"{postflop_actor} {postflop_actor}")

    stack_safe = [max(1, s) for s in stacks]
    stack_str = " ".join(str(s) for s in stack_safe)

    return (
        f"universal_poker(betting=nolimit,"
        f"numPlayers={n},"
        f"numRounds=4,"
        f"blind={blind_str},"
        f"ante={ante_str},"
        f"firstPlayer={first_player},"
        f"numSuits=4,"
        f"numRanks=13,"
        f"numHoleCards=2,"
        f"numBoardCards=0 3 1 1,"
        f"stack={stack_str},"
        f"bettingAbstraction=fullgame)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/six_max_phase4f_dcfr_linear_shakedown.yaml")
    ap.add_argument("--iterations", type=int, default=200,
                    help="iterations cap for the throwaway")
    ap.add_argument("--out", default=None)
    ap.add_argument("--time-budget-sec", type=int, default=7200,
                    help="hard wall-clock cap (default 2h)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.pop("tag", None)
    abstraction_path = cfg_dict.pop("abstraction_path")
    cfg_dict["n_iterations"] = args.iterations
    # Save checkpoints frequently so we have something even if killed.
    # checkpoint_every is a solver.train() arg, NOT a TrainConfig6Max field.
    ckpt_every = max(20, args.iterations // 5)
    cfg_dict.pop("checkpoint_every", None)
    cfg_dict.pop("parallel_groups", None)
    cfg_dict.pop("parallel_use_processes", None)
    # Drop GPU-tuned counts that bloat CPU wall-time on Contabo
    cfg_dict["traversals_per_iter"] = min(
        cfg_dict.get("traversals_per_iter", 150), 40)
    cfg_dict["train_steps_per_iter"] = min(
        cfg_dict.get("train_steps_per_iter", 200), 40)

    cfg = TrainConfig6Max(**cfg_dict)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(
        f"runs/throwaway_real_ante_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        f"# Real-ante throwaway, iters={args.iterations}\n"
        f"config: {cfg_dict}\nabstraction_path: {abstraction_path}\n")
    log.info(f"Output dir: {out_dir}")

    log.info("Loading abstraction (reused, ante-invariant)...")
    import pickle
    with open(abstraction_path, "rb") as f:
        abstraction = pickle.load(f)

    log.info(f"Loading TournamentStructure from {cfg.tournament_structure_path}")
    base = TournamentStructure.from_yaml(cfg.tournament_structure_path)
    # TournamentStructure is a frozen dataclass — can't monkey-patch.
    # Use a duck-typed wrapper that forwards everything except the one
    # method we need to override.
    class _RealAnteStructure:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)
        def __getattr__(self, name):
            return getattr(self._inner, name)
        def to_inner_game_string_for_state(self, blind_level, stacks,
                                            dealer_seat):
            return to_inner_game_string_for_state_real_ante(
                self._inner, blind_level, stacks, dealer_seat)
    structure = _RealAnteStructure(base)
    log.info("Wrapped TournamentStructure -> real-ante variant for "
             "to_inner_game_string_for_state")

    # Sanity: emit one game string + load it to verify the patched pyspiel
    # accepts the new ante parameter end-to-end before training starts.
    test_stacks = (cfg.starting_stack,) * 6
    test_gs = structure.to_inner_game_string_for_state(
        blind_level=structure.level(1), stacks=test_stacks, dealer_seat=0)
    log.info(f"Sample real-ante game-string (level 1, dealer=0):\n  {test_gs}")
    g = pyspiel.load_game(test_gs)
    s = g.new_initial_state()
    while s.is_chance_node():
        s.apply_action(s.legal_actions()[0])
    legal_raises = [a for a in s.legal_actions() if a >= 2]
    log.info(f"At first decision: min raise-to = {min(legal_raises)} "
             f"(should be 50 = 2*real_BB at level 1)")
    if min(legal_raises) != 50:
        log.error(f"PRE-TRAIN SANITY FAILED: min-raise = {min(legal_raises)} "
                  f"!= 50. Aborting throwaway train.")
        sys.exit(1)
    log.info("Pre-train sanity: min-raise=50 confirmed. Building solver...")

    # Build OpenSpiel game (needed for solver init — uses static
    # PokerGameConfig, NOT the per-state game string; this is the
    # action-abstraction/encoder game).
    from src.nlhe.game_strings import PokerGameConfig
    game = pyspiel.load_game(
        PokerGameConfig(num_players=6,
                        starting_stack=cfg.starting_stack,
                        big_blind=cfg.big_blind,
                        small_blind=cfg.small_blind).to_universal_poker_string())

    solver = DeepCFR6MaxSolver(game=game, abstraction=abstraction, config=cfg)
    # Overwrite the structure the solver loaded internally with our monkey-
    # patched version, so the training-loop game-string call uses real ante.
    solver.tournament_structure = structure
    log.info(f"Solver ready. Training {args.iterations} iterations "
             f"(time budget {args.time_budget_sec}s = "
             f"{args.time_budget_sec/3600:.1f}h)...")

    t0 = time.time()
    # solver.train() is bounded by cfg.n_iterations (= args.iterations).
    # Wall-clock budget is enforced externally by the caller's timeout
    # (background process killed if it overruns) — periodic checkpoints
    # ensure we have something even on abort.
    metrics = solver.train(
        checkpoint_dir=out_dir,
        checkpoint_every=ckpt_every,
    )
    elapsed = time.time() - t0
    log.info(f"Training done. wall_time={elapsed:.0f}s ({elapsed/60:.1f}m)")
    n_done = len(metrics.get("iter", [])) if metrics else 0
    log.info(f"Iterations completed: {n_done}")
    # List checkpoints
    ckpts = sorted(out_dir.glob("ckpt_iter_*.pt"))
    log.info(f"Checkpoints: {[c.name for c in ckpts]}")
    if ckpts:
        log.info(f"Latest ckpt: {ckpts[-1]}")
    return out_dir


if __name__ == "__main__":
    main()
