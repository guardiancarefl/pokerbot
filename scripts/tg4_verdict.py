"""TG4 (EXP_H1) verdict: merge sharded attacker_extraction_eval outputs and
apply the pre-committed F-TG4 kill bar.

F-TG4 (EXPERIMENT_LOG 2026-06-12 registration): extraction with the tail
floor armed WORSE than the champion B4/B5 floors-ON baselines by > 2 sigma
=> kill the tail floor regardless of TG3.

Extraction metric (matches evals/attacker_ext_20260611 REPORT.md):
  standard: hero_net (neutral 0 by symmetry).
  bubble:   (hero_net + 1) - hero start Malmuth-Harville ICM equity, with
            the start state re-derived from the seed schedule
            (seed = master + 7919*g; row_rng = Random(seed ^ 0xB0BB1E)).

Also reports the CRN-paired per-game delta vs the archived baseline
records (same seed schedule => same games) as a higher-power diagnostic;
the REGISTERED bar stays the unpaired 2-sigma comparison.

Usage:
  python -m scripts.tg4_verdict --mode standard \
      --new "evals/.../tg4/std_shard*.jsonl" \
      --baseline-glob "evals/attacker_ext_20260611_pergame/attacker_ext_20260611/b4_std_on_shard*.jsonl" \
      --baseline-mean -0.0400 --baseline-se 0.0158 \
      --out evals/.../tg4/tg4_standard_verdict.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nlhe.icm import icm_equity  # noqa: E402
from scripts.sng_baseline import N_SEATS, PAYOUTS  # noqa: E402


def load_records(pattern):
    recs = {}
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                assert r["game"] not in recs, f"dup game {r['game']} in {p}"
                recs[r["game"]] = r
    return recs


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def extraction(rec, mode, bubble_rows, master_seed):
    if rec["tainted"]:
        return None
    if mode == "standard":
        return rec["hero_net"]
    seed = master_seed + 7919 * rec["game"]
    row_rng = random.Random(seed ^ 0xB0BB1E)
    row = row_rng.choice(bubble_rows)
    alive = [i for i in range(N_SEATS) if row["stacks"][i] > 0]
    hero_seat = row_rng.choice(alive)
    assert hero_seat == rec["hero_seat"], \
        f"seed-schedule mismatch at game {rec['game']}"
    start_eq = icm_equity(row["stacks"], PAYOUTS, eligible=alive)[hero_seat]
    return (rec["hero_net"] + 1.0) - start_eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("standard", "bubble"), required=True)
    ap.add_argument("--new", required=True, help="glob of new shard jsonls")
    ap.add_argument("--baseline-glob", default=None,
                    help="archived baseline per-game jsonls (CRN-paired diag)")
    ap.add_argument("--baseline-mean", type=float, required=True)
    ap.add_argument("--baseline-se", type=float, required=True)
    ap.add_argument("--bubble-artifact",
                    default="data/training_dist_v1_bubble4.json.gz")
    ap.add_argument("--master-seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bubble_rows = None
    if args.mode == "bubble":
        bubble_rows = json.load(
            gzip.open(args.bubble_artifact, "rt"))["hand_starts"]

    new = load_records(args.new)
    exts = {g: extraction(r, args.mode, bubble_rows, args.master_seed)
            for g, r in new.items()}
    vals = [v for v in exts.values() if v is not None]
    n_tainted = sum(1 for v in exts.values() if v is None)
    m, se = mean_se(vals)

    z_vs_base = (m - args.baseline_mean) / math.sqrt(
        se ** 2 + args.baseline_se ** 2)
    # "Worse than baseline" in F-TG4 means worse FOR THE CHAMPION: the
    # attacker extracts MORE with the tail floor armed (z > +2). Lower
    # extraction = the floor increased robustness = the good direction.
    kill = z_vs_base > 2.0

    paired = None
    if args.baseline_glob:
        base = load_records(args.baseline_glob)
        common = sorted(set(new) & set(base))
        deltas = []
        for g in common:
            a = extraction(new[g], args.mode, bubble_rows, args.master_seed)
            b = extraction(base[g], args.mode, bubble_rows, args.master_seed)
            if a is not None and b is not None:
                deltas.append(a - b)
        dm, dse = mean_se(deltas)
        paired = {"n_pairs": len(deltas), "mean": dm, "se": dse,
                  "z": dm / dse if dse > 0 else None,
                  "n_identical": sum(1 for d in deltas if d == 0.0)}

    out = {
        "registration": "EXP_H1 H1.2 / TG4",
        "mode": args.mode,
        "games": len(vals), "n_tainted": n_tainted,
        "extraction_mean": m, "extraction_se": se,
        "raw_net_mean": sum(r["hero_net"] for r in new.values()
                            if not r["tainted"]) / len(vals),
        "baseline": {"mean": args.baseline_mean, "se": args.baseline_se},
        "z_vs_baseline": z_vs_base,
        "kill_bar_attacker_gain_z_gt_2": kill,
        "verdict": "KILL" if kill else "PASS",
        "paired_vs_baseline_records": paired,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
