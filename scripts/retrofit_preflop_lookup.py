#!/usr/bin/env python3
"""Retrofit a deterministic preflop_lookup onto an existing trained abstraction.

Background: bucket_of() preflop was historically non-deterministic — same
(hero, []) call returned different bucket IDs because of MC sampling
variance (see DECISIONS.md). The preflop_lookup field on StreetAbstraction
fixes this at the lookup layer.

For abstractions trained BEFORE the trainer was updated to populate
preflop_lookup (commit ae2a1e7), the field is None. Re-running the trainer
would produce a NEW abstraction with different bucket IDs than the original
(verified empirically — Session 7.5 Eval B regressed from +15.05 to -16.75
because the bot was indexed by the original's bucket IDs).

This script retrofits the lookup table onto an existing abstraction by:
  1. Loading the original abstraction (with its existing medoid_histograms).
  2. For each of the 169 canonical preflop HoleClasses, calling bucket_of()
     N times with different RNG seeds.
  3. Taking the MODAL bucket assignment as the "real" bucket id for that hand.
  4. Writing that mapping back into the abstraction's preflop_lookup field.
  5. Saving the result to a new path.

Result: an abstraction with the SAME bucket IDs the original training used
(modulo the small fraction of hands where MC noise actually flipped the
nearest-medoid answer at training time too), but with a deterministic
lookup path at query time. Phase 2d's trained network can use this without
the preflop bucket-remap regression we saw in Eval B.

Usage:
    python scripts/retrofit_preflop_lookup.py \
        --in  runs/abstraction_20260521_223018/abstraction.pkl \
        --out runs/abstraction_20260521_223018_retrofit/abstraction.pkl \
        --trials 11 --runouts 5000
"""
from __future__ import annotations
import argparse
import logging
import pickle
import random
import time
from collections import Counter
from pathlib import Path

from src.nlhe.abstraction import Abstraction
from src.nlhe.equity import all_hole_classes, hole_class_to_cards

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retrofit")


def modal_bucket(abst: Abstraction, hero: list[int], trials: int, runouts: int) -> tuple[int, int]:
    """Call bucket_of() multiple times with different RNG seeds, return modal answer.

    Returns (modal_bucket_id, agreement_count). The agreement count tells us how
    confident we are: trials/trials is unambiguous, trials/2 is a coin flip.
    """
    votes = []
    for trial in range(trials):
        rng = random.Random(trial * 7919 + 1)  # 7919 = prime, well-spread seeds
        b = abst.bucket_of(hero, [], runouts=runouts, rng=rng)
        votes.append(b)
    counts = Counter(votes)
    modal_id, modal_count = counts.most_common(1)[0]
    return modal_id, modal_count


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", required=True,
                        help="path to the input abstraction.pkl (will not be modified)")
    parser.add_argument("--out", dest="out_path", required=True,
                        help="path to write the retrofit abstraction.pkl")
    parser.add_argument("--trials", type=int, default=11,
                        help="MC trials per canonical hand (odd number recommended for "
                             "unambiguous modal); default 11")
    parser.add_argument("--runouts", type=int, default=5000,
                        help="MC runouts per trial; default 5000 (high stability)")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"loading source abstraction from {in_path}")
    abst = Abstraction.load(in_path)
    if "preflop" not in abst.streets:
        raise SystemExit("ERROR: abstraction has no preflop street")
    sa = abst.streets["preflop"]
    if sa.preflop_lookup is not None:
        log.warning(f"  source abstraction already has preflop_lookup populated "
                    f"({len(sa.preflop_lookup)} entries). Will be REPLACED.")
    log.info(f"  preflop k = {sa.k}")
    log.info(f"  computing modal bucket for each of 169 canonical HoleClasses")
    log.info(f"  (trials={args.trials}, runouts={args.runouts} per call)")

    t_start = time.time()
    lookup = {}
    unstable = []  # (canonical_str, agreement, total)
    canonical_classes = list(all_hole_classes())
    for i, hc in enumerate(canonical_classes):
        hero = list(hole_class_to_cards(hc))
        modal_id, count = modal_bucket(abst, hero, args.trials, args.runouts)
        lookup[str(hc)] = int(modal_id)
        if count < args.trials:
            unstable.append((str(hc), count, args.trials))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (169 - i - 1) / rate
            log.info(f"  progress: {i+1}/169  rate={rate:.1f}/s  ETA={eta:.0f}s")

    total_elapsed = time.time() - t_start
    log.info(f"=== retrofit done in {total_elapsed:.1f}s ===")
    log.info(f"  unambiguous (modal agreement = {args.trials}): {169 - len(unstable)}/169")
    log.info(f"  ambiguous (modal disagreement): {len(unstable)}/169")
    if unstable:
        log.info(f"  ambiguous hands:")
        for s, c, t in unstable[:20]:
            log.info(f"    {s:5s}: modal won {c}/{t} trials")
        if len(unstable) > 20:
            log.info(f"    ... and {len(unstable) - 20} more")

    # Verify completeness
    if len(lookup) != 169:
        raise SystemExit(f"ERROR: lookup has {len(lookup)} entries, expected 169")
    if len(set(lookup.keys())) != 169:
        raise SystemExit(f"ERROR: lookup has duplicate keys")

    # Inject and save
    sa.preflop_lookup = lookup
    log.info(f"writing retrofit abstraction to {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(abst, f)
    log.info(f"OK. retrofit complete.")


if __name__ == "__main__":
    main()
