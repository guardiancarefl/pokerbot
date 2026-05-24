"""Tournament eval: Shanky profiles vs DCFR checkpoint.

Pits every Shanky profile (or a selected subset) head-to-head against a
DCFR challenger checkpoint. Reuses scripts.eval_pool.evaluate_matchup()
for the rollout — same seating, ICM diff, and per-seat statistics. The
only difference is that the opponent is a ShankyProfilePolicy instead
of a CheckpointPolicy.

Output JSON has the same schema as eval_pool's, so it slots into existing
comparison tooling.

Example:
    python -m scripts.eval_shanky_vs_dcfr \\
        --challenger-ckpt runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/ckpt_iter_3000.pt \\
        --challenger-name dcfr-3000 \\
        --shanky-dir /workspace/pokerbot/data/shanky_profiles \\
        --abstraction runs/abstraction_20260521_223018_retrofit/abstraction.pkl \\
        --structure configs/ignition_double_up_6max_turbo.yaml \\
        --hands 2500 \\
        --output evals/shanky_vs_dcfr-3000_2500.json

Answers three questions:
  1. Are any Shanky profiles competitive with DCFR-3000?
  2. Which Shanky profiles are STRONGEST (smallest positive diff for DCFR)?
  3. What's the typical strength gap between DCFR and scripted bots?

Decision rule for league v2 inclusion: pick Shanky profiles where DCFR
beats them by < 0.05 ICM. Those are the meaningful style opponents.
Profiles crushed by 0.10+ are pushovers that would weaken training.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import List

logging.basicConfig(
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("eval_shanky_vs_dcfr")


def _build_shanky_policies(
    shanky_dir: str,
    only: List[str] = None,
    big_blind_chips: int = 100,
):
    """Load Shanky profiles from a directory.

    Args:
        shanky_dir: directory containing .txt Shanky profile files.
        only: if given, only load profiles whose stem (filename without
            .txt) matches. Comparison normalizes common suffixes like
            '__1_' that some downloads add.
        big_blind_chips: chips per BB for ShankyProfilePolicy construction.

    Returns:
        list of ShankyProfilePolicy instances, ordered by filename.
    """
    from src.nlhe.scripted_bots.policy import ShankyProfilePolicy

    if not os.path.isdir(shanky_dir):
        raise FileNotFoundError(f"shanky_dir not found: {shanky_dir}")

    out = []
    for fname in sorted(os.listdir(shanky_dir)):
        if not fname.endswith(".txt"):
            continue
        stem = os.path.splitext(fname)[0].lower()
        # Strip suffixes added by some download sources
        normalized = stem.replace("__1_", "").replace("_v_", "_v").strip("_")
        if only is not None and normalized not in only and stem not in only:
            continue
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
    if only is not None:
        loaded = {p.name for p in out}
        missing = [n for n in only if n not in loaded]
        if missing:
            log.warning(f"requested but not loaded: {missing}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--challenger-ckpt",
        required=True,
        help="path to DCFR .pt checkpoint (the strength benchmark)",
    )
    ap.add_argument(
        "--challenger-name",
        default="dcfr-3000",
        help="display name for the challenger in output",
    )
    ap.add_argument(
        "--shanky-dir",
        required=True,
        help="directory containing .txt Shanky profile files",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="if given, only test these Shanky profiles (by stem name)",
    )
    ap.add_argument("--abstraction", required=True, help="path to abstraction.pkl")
    ap.add_argument("--structure", required=True, help="path to tournament structure YAML")
    ap.add_argument(
        "--hands",
        type=int,
        default=2500,
        help="hands per matchup (default 2500)",
    )
    ap.add_argument("--seed", type=int, default=2026, help="base seed")
    ap.add_argument(
        "--mode",
        choices=("sample", "argmax"),
        default="sample",
        help="action sampling mode for DCFR challenger (default 'sample')",
    )
    ap.add_argument(
        "--log-every", type=int, default=500,
        help="log progress every N hands within a matchup",
    )
    ap.add_argument("--output", required=True, help="output JSON path")
    args = ap.parse_args()

    log.info("loading dependencies...")
    from scripts.eval_pool import CheckpointPolicy, evaluate_matchup
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.stack_sampler import TournamentStructure

    log.info(f"loading abstraction from {args.abstraction}")
    abstr = Abstraction.load(args.abstraction)
    log.info(f"loading structure from {args.structure}")
    structure = TournamentStructure.from_yaml(args.structure)

    log.info(f"loading challenger: {args.challenger_name} from {args.challenger_ckpt}")
    challenger = CheckpointPolicy(
        name=args.challenger_name,
        ckpt_path=args.challenger_ckpt,
        abstraction=abstr,
        structure=structure,
    )

    log.info(f"loading Shanky profiles from {args.shanky_dir}")
    shanky_policies = _build_shanky_policies(args.shanky_dir, only=args.only)
    log.info(f"  -> {len(shanky_policies)} Shanky profiles loaded")

    if not shanky_policies:
        log.error("no Shanky profiles loaded; aborting")
        sys.exit(1)

    log.info(
        f"Tournament eval: challenger={challenger.name}, "
        f"{len(shanky_policies)} Shanky opponents, "
        f"{args.hands} hands/matchup, mode={args.mode}"
    )

    results = []
    t_start = time.time()

    for i, opp in enumerate(shanky_policies):
        log.info("=" * 60)
        log.info(
            f"[{i + 1}/{len(shanky_policies)}] "
            f"Matchup: {challenger.name} vs shanky:{opp.name}"
        )
        # Different seed per matchup so they're statistically independent
        seed = args.seed + i * 1000
        try:
            r = evaluate_matchup(
                challenger=challenger,
                opponent=opp,
                structure=structure,
                hands=args.hands,
                seed=seed,
                mode=args.mode,
                log_every=args.log_every,
            )
            # Tag opponent name with shanky: prefix
            r["opponent"] = f"shanky:{opp.name}"
            r["seed"] = seed
            results.append(r)
        except Exception as e:
            log.exception(f"  matchup errored: {type(e).__name__}: {e}")
            continue

    # Summary
    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info(
        f"SUMMARY: {challenger.name} vs {len(results)} Shanky profiles "
        f"(elapsed {elapsed / 60:.1f} min)"
    )
    log.info("")
    log.info(
        f"  {'opponent':<35} {'diff':>10} {'stderr':>8} "
        f"{'sigma':>6}  {'capped':>6}"
    )
    log.info(f"  {'-' * 35} {'-' * 10} {'-' * 8} {'-' * 6}  {'-' * 6}")

    # Sort by diff ASCENDING (smallest positive = strongest Shanky opponent)
    results.sort(key=lambda r: r["diff"])

    for r in results:
        sigma_val = r["sigma"]
        sigma_str = f"{sigma_val:.1f}" if sigma_val == sigma_val else "nan"
        log.info(
            f"  {r['opponent']:<35} "
            f"{r['diff']:+10.4f} {r['stderr']:8.4f} {sigma_str:>6}  "
            f"{r['n_capped']:6d}"
        )

    if results:
        agg_diff = sum(r["diff"] for r in results) / len(results)
        log.info("")
        log.info(f"  Aggregate diff (mean): {agg_diff:+.4f}")
        log.info(f"  (positive = challenger DCFR beats Shanky on average)")
        log.info("")

        # Decision-rule output: which profiles are "competitive"
        competitive = [r for r in results if r["diff"] < 0.05]
        log.info(f"  Competitive profiles (DCFR wins by <0.05 ICM): {len(competitive)}")
        for r in competitive:
            log.info(
                f"    {r['opponent']:<35} "
                f"diff={r['diff']:+.4f}  sigma={r['sigma']:.1f}"
            )

    # Persist
    output = {
        "challenger": challenger.name,
        "challenger_ckpt": args.challenger_ckpt,
        "n_hands_per_matchup": args.hands,
        "mode": args.mode,
        "base_seed": args.seed,
        "shanky_dir": args.shanky_dir,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved JSON results to {args.output}")


if __name__ == "__main__":
    main()
