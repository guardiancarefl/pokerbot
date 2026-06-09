"""Real-ante bake-off runner.

Two usage modes:

  --mode pool: challenger ckpt + abstraction vs every Shanky profile
  --mode h2h:  challenger ckpt + abstraction vs ONE DCFR ckpt loaded
               with its own (potentially different) abstraction

Both modes wrap the structure with _RealAnteStructure so the game
is played under the real-ante convention regardless of how the
ckpts were trained.

Output: one JSON per invocation with per-matchup diff/stderr/sigma.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bake_off")


def build_shanky_pool(shanky_dir: str, big_blind_chips: int = 100):
    from src.nlhe.scripted_bots.policy import ShankyProfilePolicy
    out = []
    for fname in sorted(os.listdir(shanky_dir)):
        if not fname.endswith(".txt"):
            continue
        stem = os.path.splitext(fname)[0].lower()
        normalized = stem.replace("__1_", "").replace("_v_", "_v").strip("_")
        path = os.path.join(shanky_dir, fname)
        try:
            policy = ShankyProfilePolicy(
                name=normalized,
                profile_path=path,
                big_blind_chips=big_blind_chips,
            )
            out.append(policy)
            log.info(f"loaded shanky:{normalized}")
        except Exception as e:
            log.warning(f"failed to load {fname}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pool", "h2h"), required=True)
    ap.add_argument("--challenger-name", required=True)
    ap.add_argument("--challenger-ckpt", required=True)
    ap.add_argument("--challenger-abstraction", required=True)
    ap.add_argument("--opponent-name", default=None,
                    help="(h2h only) display name for opponent")
    ap.add_argument("--opponent-ckpt", default=None,
                    help="(h2h only) opponent DCFR checkpoint path")
    ap.add_argument("--opponent-abstraction", default=None,
                    help="(h2h only) opponent's OWN abstraction path "
                         "(different from challenger's by design)")
    ap.add_argument("--shanky-dir",
                    default="data/shanky_profiles",
                    help="(pool only) directory with .txt profile files")
    ap.add_argument("--structure", required=True)
    ap.add_argument("--hands", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mode-action", default="sample",
                    choices=("sample", "argmax"))
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    log.info("loading dependencies...")
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from scripts.eval_pool import CheckpointPolicy, evaluate_matchup
    from scripts.throwaway_query_real_ante import _RealAnteStructure

    log.info(f"loading structure from {args.structure}")
    base_struct = TournamentStructure.from_yaml(args.structure)
    structure = _RealAnteStructure(base_struct)
    log.info("  wrapped with _RealAnteStructure")

    log.info(f"loading challenger abstraction: {args.challenger_abstraction}")
    challenger_abstr = Abstraction.load(args.challenger_abstraction)
    log.info(f"  challenger postflop k = {challenger_abstr.streets['flop'].k}")

    log.info(f"loading challenger: {args.challenger_name} from {args.challenger_ckpt}")
    challenger = CheckpointPolicy(
        name=args.challenger_name,
        ckpt_path=args.challenger_ckpt,
        abstraction=challenger_abstr,
        structure=structure,
    )

    opponents = []
    if args.mode == "pool":
        log.info(f"loading shanky pool from {args.shanky_dir}")
        opponents = build_shanky_pool(args.shanky_dir)
        log.info(f"  -> {len(opponents)} shanky profiles loaded")
    else:
        assert args.opponent_ckpt and args.opponent_abstraction, \
            "h2h mode requires --opponent-ckpt and --opponent-abstraction"
        log.info(f"loading opponent abstraction: {args.opponent_abstraction}")
        opp_abstr = Abstraction.load(args.opponent_abstraction)
        log.info(f"  opponent postflop k = {opp_abstr.streets['flop'].k}")
        opp_name = args.opponent_name or "opponent"
        log.info(f"loading opponent: {opp_name} from {args.opponent_ckpt}")
        opp = CheckpointPolicy(
            name=opp_name,
            ckpt_path=args.opponent_ckpt,
            abstraction=opp_abstr,
            structure=structure,
        )
        opponents = [opp]

    log.info(
        f"Bake-off: challenger={challenger.name}, "
        f"{len(opponents)} opponent(s), {args.hands} hands/matchup, "
        f"mode={args.mode_action}"
    )

    results = []
    t_start = time.time()

    for i, opp in enumerate(opponents):
        log.info("=" * 60)
        log.info(f"[{i + 1}/{len(opponents)}] "
                 f"Matchup: {challenger.name} vs {opp.name}")
        seed = args.seed + i * 1000
        try:
            r = evaluate_matchup(
                challenger=challenger,
                opponent=opp,
                structure=structure,
                hands=args.hands,
                seed=seed,
                mode=args.mode_action,
                log_every=args.log_every,
            )
            # Tag opponent name w/ kind prefix for downstream parsing
            prefix = "shanky:" if args.mode == "pool" else "dcfr:"
            r["opponent"] = f"{prefix}{opp.name}"
            r["seed"] = seed
            results.append(r)
        except Exception as e:
            log.exception(f"  matchup errored: {type(e).__name__}: {e}")

    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info(
        f"SUMMARY: {challenger.name} vs {len(results)} opponent(s) "
        f"(elapsed {elapsed / 60:.1f} min)"
    )
    log.info(f"  {'opponent':<35} {'diff':>10} {'stderr':>8} {'sigma':>6}  {'capped':>6}")
    log.info(f"  {'-' * 35} {'-' * 10} {'-' * 8} {'-' * 6}  {'-' * 6}")
    # Sort ascending so toughest opponent appears first
    results_sorted = sorted(results, key=lambda r: r["diff"])
    for r in results_sorted:
        sigma_val = r["sigma"]
        sigma_str = f"{sigma_val:.1f}" if sigma_val == sigma_val else "nan"
        log.info(
            f"  {r['opponent']:<35} "
            f"{r['diff']:+10.4f} {r['stderr']:8.4f} {sigma_str:>6}  "
            f"{r['n_capped']:6d}"
        )
    if results:
        agg = sum(r["diff"] for r in results) / len(results)
        log.info(f"  Aggregate diff (mean): {agg:+.4f}")

    output = {
        "mode": args.mode,
        "challenger_name": args.challenger_name,
        "challenger_ckpt": args.challenger_ckpt,
        "challenger_abstraction": args.challenger_abstraction,
        "structure": args.structure,
        "real_ante": True,
        "n_hands_per_matchup": args.hands,
        "base_seed": args.seed,
        "action_mode": args.mode_action,
        "elapsed_min": elapsed / 60,
        "results": results,
    }
    if args.mode == "h2h":
        output["opponent_name"] = args.opponent_name
        output["opponent_ckpt"] = args.opponent_ckpt
        output["opponent_abstraction"] = args.opponent_abstraction
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    log.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
