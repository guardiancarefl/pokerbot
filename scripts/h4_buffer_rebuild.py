"""H4 P3 — buffer rebuild (spec §3 Option A, the H2-fragility-safe step).

Resume the SLIM deployed champion checkpoint (no reservoir buffers), refill the
advantage/strategy reservoirs from on-policy traversals, then save a FULL
(buffer-inclusive) checkpoint. The field-mix probe (P4) resumes from THIS full
checkpoint — never from the slim one — which is what makes H4 H2-compliant: no
fine-tuning from empty reservoirs with off-policy opponents (the H2 collapse
signature, self-anchor z=-9.9).

REFILL METHOD (--populate-only, REQUIRED — 2026-06-13 resolution):
The original FREE-SELF-PLAY refill is RETIRED for this purpose. It retrained the
adv+strat nets on a from-empty reservoir, re-averaging the strategy and moving
the player: the post-rebuild self-anchor regressed 0.15/game (z=-3.1, plateaued
across iters 1600-1900). See EXPERIMENT_LOG 2026-06-13. With --populate-only the
solver COLLECTS samples via FROZEN-net traversals but SKIPS both gradient steps,
so the nets never move and the post-rebuild self-anchor is 0 BY CONSTRUCTION
(verified bit-identical: 901,439 params, max|Δ|=0). Run WITH --populate-only.

Encoder discipline (spec §3): champion-lineage 236-d legacy encoder ONLY
(config `six_max_phase4f_dcfr_candC_k200.yaml`, encoder_eff_bb=False, no
empirical_dist). NOT the C3 237-d eff_bb recipe — that would (a) violate
one-delta discipline and (b) fail to load the 236-d champion. Verified the
champion is 236-d on load (solver prints feature_dim=236).

Sanity gate (spec §3 Option A): after the rebuild, the self-anchor row
(hero=rebuilt vs all-champion, |z|<2 of 0) must hold BEFORE any field mix.
If it fails, the rebuild drifted and the probe must NOT proceed on it (fall
back to Option-B miniature, spec §3). Run that check separately via
`h4_progress_monitor --once` on the full checkpoint.

`solver.train()` saves SLIM checkpoints during the run; this script saves an
EXPLICIT FULL checkpoint (slim=False) at the end — without it the rebuild is
pointless (the buffers would not carry forward).

Usage:
  # benchmark first (project rule): 2 iters, timing + buffer-fill, no full save
  python -m scripts.h4_buffer_rebuild --benchmark
  # full rebuild (Contabo-only; ~1.8h at G=8):
  python -m scripts.h4_buffer_rebuild --iters 400 --out runs/h4_buffer_rebuild_<ts>
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pyspiel  # noqa: E402
import yaml  # noqa: E402

from src.nlhe.game_strings import TournamentStructure, PokerGameConfig  # noqa: E402
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("h4_buffer_rebuild")

CHAMPION = "runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt"
CHAMPION_CONFIG = "configs/six_max_phase4f_dcfr_candC_k200.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CHAMPION_CONFIG)
    ap.add_argument("--resume", default=CHAMPION)
    ap.add_argument("--iters", type=int, default=400,
                    help="self-play rebuild iters (spec §3: 300-500)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--parallel-groups", type=int, default=8)
    ap.add_argument("--benchmark", action="store_true",
                    help="run only 2 iters, report timing + buffer fill, no full save")
    ap.add_argument("--populate-only", action="store_true",
                    help="GENERATE-ONLY refill (REQUIRED): collect samples via "
                         "frozen-net traversals, SKIP both gradient steps so the "
                         "player never moves (self-anchor=0 by construction). The "
                         "free-self-play refill is retired (EXPERIMENT_LOG 2026-06-13).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.pop("tag", None)
    abstraction_path = cfg_dict.pop("abstraction_path")
    cfg_dict.pop("checkpoint_every", None)
    # ONE-DELTA DISCIPLINE + H2 SAFETY: pure self-play, champion 236-d recipe.
    cfg_dict["league_mix"] = 0.0
    cfg_dict["archetype_mix"] = 0.0
    cfg_dict["parallel_groups"] = args.parallel_groups
    cfg_dict["parallel_use_processes"] = True
    # GENERATE-ONLY refill: frozen nets, skip both gradient steps (the player
    # never moves). The free-self-play refill is retired for this purpose.
    cfg_dict["populate_only"] = args.populate_only

    # resume iteration is read from the checkpoint; target = resume + iters.
    resume_iter = _ckpt_iter(args.resume)
    n_iters = resume_iter + (2 if args.benchmark else args.iters)
    cfg_dict["n_iterations"] = n_iters

    cfg = TrainConfig6Max(**cfg_dict)
    assert cfg.league_mix == 0.0 and cfg.archetype_mix == 0.0, "must be pure self-play"
    assert not cfg.encoder_eff_bb, "must be champion 236-d encoder (one-delta discipline)"
    if not cfg.populate_only:
        log.warning("RUNNING WITHOUT --populate-only: this is the RETIRED free-self-play "
                    "refill that regressed the anchor 0.15/game (EXPERIMENT_LOG 2026-06-13). "
                    "Pass --populate-only unless you are intentionally reproducing that result.")

    out_dir = Path(args.out) if args.out else Path(
        "runs/h4_buffer_rebuild_benchmark" if args.benchmark
        else "runs/h4_buffer_rebuild")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"resume={args.resume} (iter {resume_iter}) -> target iter {n_iters}  "
             f"[{'BENCHMARK' if args.benchmark else 'FULL'}], G={cfg.parallel_groups}")
    log.info(f"league_mix={cfg.league_mix} archetype_mix={cfg.archetype_mix} "
             f"(pure self-play); out={out_dir}")

    with open(abstraction_path, "rb") as f:
        abstraction = pickle.load(f)
    structure = TournamentStructure.from_yaml(cfg.tournament_structure_path)
    game = pyspiel.load_game(
        PokerGameConfig(num_players=6, starting_stack=cfg.starting_stack,
                        big_blind=cfg.big_blind, small_blind=cfg.small_blind
                        ).to_universal_poker_string())
    solver = DeepCFR6MaxSolver(game=game, abstraction=abstraction, config=cfg)
    solver.tournament_structure = structure

    log.info("loading SLIM champion (expect empty buffers, feature_dim=236) ...")
    solver.load_checkpoint(args.resume)
    assert solver.encoder.feature_dim == 236, \
        f"feature_dim {solver.encoder.feature_dim} != 236 — wrong encoder lineage!"
    log.info(f"resumed at iter {solver.iteration}; "
             f"buffers (should be ~empty): "
             f"{[solver.policy_nets.buffer_for(i).n_seen for i in range(6)]}")

    t0 = time.time()
    if cfg.parallel_groups > 0:
        # Parallelize the traversal phase across G workers (the dominant cost
        # under populate_only, which does no training). SAFE for generate-only:
        # the nets are frozen, so worker-threading differences change only which
        # champion-policy samples land in the buffers, never the player. (Was a
        # latent no-op before — solver.train() ignored parallel_groups.)
        from src.nlhe.parallel.orchestrator import parallel_train
        game_str = PokerGameConfig(
            num_players=6, starting_stack=cfg.starting_stack,
            big_blind=cfg.big_blind, small_blind=cfg.small_blind
        ).to_universal_poker_string()
        log.info(f"parallel mode: G={cfg.parallel_groups} "
                 f"use_processes={cfg.parallel_use_processes}")
        parallel_train(
            solver, game_str=game_str, abstraction_path=abstraction_path,
            n_workers=cfg.parallel_groups, use_processes=cfg.parallel_use_processes,
            checkpoint_dir=out_dir, checkpoint_every=args.checkpoint_every,
        )
    else:
        solver.train(checkpoint_dir=out_dir, checkpoint_every=args.checkpoint_every)
    elapsed = time.time() - t0
    n_done = solver.iteration - resume_iter
    per_iter = elapsed / max(n_done, 1)
    bufs = [solver.policy_nets.buffer_for(i).n_seen for i in range(6)]
    sbuf = solver.policy_nets.strat_buffer.n_seen
    log.info(f"trained {n_done} iters in {elapsed:.0f}s ({per_iter:.1f}s/iter); "
             f"adv buffers n_seen={bufs}, strat n_seen={sbuf}")

    if args.benchmark:
        proj_h = per_iter * args.iters / 3600.0
        log.info(f"BENCHMARK: {per_iter:.1f}s/iter -> {args.iters} iters "
                 f"≈ {proj_h:.2f}h. Buffers filling: {'YES' if min(bufs) > 0 else 'NO — INVESTIGATE'}.")
        log.info("benchmark only — no full checkpoint saved.")
        return

    # THE point of P3: an explicit FULL (buffer-inclusive) checkpoint.
    full_path = out_dir / "ckpt_full_rebuilt.pt"
    solver.save_checkpoint(full_path, slim=False)
    sz = full_path.stat().st_size / 1e6
    log.info(f"saved FULL checkpoint: {full_path}  ({sz:.1f} MB)")
    meta = {
        "resume_from": args.resume, "resume_iter": resume_iter,
        "rebuild_iters": args.iters, "final_iter": solver.iteration,
        "full_checkpoint": str(full_path), "full_mb": round(sz, 1),
        "adv_buffer_n_seen": bufs, "strat_buffer_n_seen": sbuf,
        "league_mix": cfg.league_mix, "feature_dim": solver.encoder.feature_dim,
        "populate_only": cfg.populate_only,
        "refill_method": "generate-only" if cfg.populate_only else "free-self-play(RETIRED)",
        "per_iter_s": round(per_iter, 1), "elapsed_s": round(elapsed, 1),
        "next": "run sanity self-anchor (h4_progress_monitor --once on this "
                "full ckpt vs champion; require |z|<2) BEFORE the field-mix probe.",
    }
    (out_dir / "rebuild_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(f"meta -> {out_dir/'rebuild_meta.json'}")
    log.info("NEXT: sanity self-anchor must pass (|z|<2) before P4 field mix.")


def _ckpt_iter(path):
    import re
    m = re.search(r"ckpt_iter_(\d+)\.pt", Path(path).name)
    return int(m.group(1)) if m else 0


if __name__ == "__main__":
    main()
