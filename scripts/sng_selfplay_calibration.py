"""Self-play calibration row for the SNG yardstick (harness bias proof).

Hero checkpoint occupies ALL SIX seats. By symmetry the hero seat's
expected net/game is exactly 0 — any significant deviation measures
harness bias (seat asymmetry, dealer-rotation imbalance, scoring-path
leak), not skill. PASS: |net/game| < 2*stderr.

Reuses scripts/sng_baseline.play_sng_game via its seat_to_policy seam,
so the game loop, blind guard, taint gate and scoring path are byte-for-
byte the v1 yardstick. Per-seat action sampling draws from independent
random.Random streams derived from the per-game seed (seed*1_000_003 +
seat); the game-level rng (chance outcomes, initial dealer draw, v1 seed
schedule master + 7919*g) is untouched, so the dealt universe matches v1.

Run --check-dealer-only first: it reproduces the initial-dealer draw for
every game seed and chi-square tests uniformity over the 6 seats. The
calibration run refuses to start if that check fails.

Shardable via --game-start/--game-count; merge shard summaries with
scripts/sng_selfplay_calibration.py --merge <dir>.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.sng_baseline import (N_SEATS, _git_identity, _sha256_of_file,
                                  play_sng_game)

SEAT_SEED_STRIDE = 1_000_003  # prime; decorrelates seat streams per game


class SeatRNGPolicy:
    """Wraps a policy so its sampling draws come from a seat-private RNG.

    The harness-level rng (chance outcomes, dealer draw) passed into
    select_action is ignored; this seat draws from its own stream seeded
    from the per-game seed. Keeps seat decisions independent of the order
    in which other seats happened to consume shared-rng draws."""

    def __init__(self, inner, seat_seed: int):
        self.inner = inner
        self.name = inner.name
        self.rng = random.Random(seat_seed)

    def select_action(self, parsed, state, rng, mode: str = "sample") -> int:
        return self.inner.select_action(parsed, state, self.rng, mode=mode)


def dealer_balance(master_seed: int, n_games: int):
    """Reproduce play_sng_game's initial-dealer draw for every game seed
    and chi-square test uniformity over the 6 seats."""
    counts = [0] * N_SEATS
    for g in range(n_games):
        seed = master_seed + 7919 * g
        counts[random.Random(seed).randrange(N_SEATS)] += 1
    expected = n_games / N_SEATS
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, N_SEATS - 1))
    except ImportError:
        p = float("nan")
    return {"counts": counts, "expected_per_seat": expected,
            "chi2": chi2, "df": N_SEATS - 1, "p_value": p,
            "balanced": p > 0.01 if p == p else chi2 < 15.09}  # 0.01 crit


def merge(out_dir: Path):
    nets, hands, taints, caps = [], [], 0, 0
    shards = sorted(out_dir.glob("calib_w*/summary.json"))
    for sp in shards:
        s = json.loads(sp.read_text())
        taints += s["n_tainted"]
        caps += s["n_capped"]
    for jp in sorted(out_dir.glob("calib_w*/games.jsonl")):
        for line in jp.read_text().splitlines():
            rec = json.loads(line)
            if rec["tainted"]:
                continue
            nets.append(rec["hero_net"])
            hands.append(rec["hands"])
    n = len(nets)
    mean = sum(nets) / n
    var = sum(x * x for x in nets) / n - mean * mean
    stderr = math.sqrt(var / n)
    sigma = abs(mean) / stderr if stderr > 0 else float("nan")
    verdict = "PASS" if sigma < 2.0 else "FAIL"
    out = {
        "record_type": "selfplay_calibration_merged",
        "n_shards": len(shards),
        "n_scored": n,
        "n_tainted": taints,
        "n_capped": caps,
        "hero_net_per_game": mean,
        "stderr": stderr,
        "sigma_from_zero": sigma,
        "mean_hands_per_game": sum(hands) / n,
        "criterion": "|net/game| < 2*stderr",
        "verdict": verdict,
    }
    (out_dir / "calibration_merged.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if verdict == "PASS" else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt")
    ap.add_argument("--abstraction")
    ap.add_argument("--structure",
                     default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--games", type=int, default=2000,
                     help="total games in the calibration row (seed schedule)")
    ap.add_argument("--game-start", type=int, default=0)
    ap.add_argument("--game-count", type=int, default=None,
                     help="games this shard plays (default: all)")
    ap.add_argument("--master-seed", type=int, default=2026)
    ap.add_argument("--hands-per-level", type=int, default=5)
    ap.add_argument("--max-hands", type=int, default=200)
    ap.add_argument("--mode", default="sample", choices=["sample", "argmax"])
    ap.add_argument("--out-dir")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--check-dealer-only", action="store_true")
    ap.add_argument("--merge", metavar="DIR",
                     help="merge calib_w*/ shard outputs under DIR and exit")
    args = ap.parse_args()

    if args.merge:
        return merge(Path(args.merge))

    bal = dealer_balance(args.master_seed, args.games)
    print(f"dealer balance over {args.games} game seeds: counts={bal['counts']} "
          f"chi2={bal['chi2']:.2f} (df={bal['df']}) p={bal['p_value']:.4f} "
          f"-> {'BALANCED' if bal['balanced'] else 'IMBALANCED'}", flush=True)
    if args.check_dealer_only:
        return 0 if bal["balanced"] else 1
    if not bal["balanced"]:
        print("REFUSING to run: dealer rotation imbalanced", flush=True)
        return 1
    if not args.ckpt or not args.abstraction or not args.out_dir:
        raise SystemExit("--ckpt/--abstraction/--out-dir required for a run")

    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from scripts.eval_pool import CheckpointPolicy
    from scripts.throwaway_query_real_ante import _RealAnteStructure

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    head, dirty = _git_identity()
    g0 = args.game_start
    g1 = g0 + (args.game_count if args.game_count is not None
               else args.games - g0)
    header = {
        "record_type": "run_header",
        "kind": "selfplay_calibration",
        "ckpt_path": str(Path(args.ckpt).resolve()),
        "ckpt_sha256": _sha256_of_file(args.ckpt),
        "abstraction_sha256": _sha256_of_file(args.abstraction),
        "config": {k: v for k, v in vars(args).items()},
        "game_range": [g0, g1],
        "dealer_balance": bal,
        "git_head": head,
        "git_dirty": dirty,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(f"run header: {json.dumps(header)}", flush=True)

    structure = _RealAnteStructure(TournamentStructure.from_yaml(args.structure))
    abstr = Abstraction.load(args.abstraction)
    hero = CheckpointPolicy(name="k200", ckpt_path=args.ckpt,
                             abstraction=abstr, structure=structure)

    nets, hands_counts = [], []
    n_tainted = n_capped = 0
    t0 = time.time()
    with open(out_dir / "games.jsonl", "w") as fh:
        for g in range(g0, g1):
            seed = args.master_seed + 7919 * g     # v1 CRN seed schedule
            seat_to_policy = [
                SeatRNGPolicy(hero, seed * SEAT_SEED_STRIDE + i)
                for i in range(N_SEATS)
            ]
            rec = play_sng_game(hero, hero, structure, seed=seed,
                                 hands_per_level=args.hands_per_level,
                                 max_hands=args.max_hands, mode=args.mode,
                                 seat_to_policy=seat_to_policy)
            rec["game"] = g
            fh.write(json.dumps(rec) + "\n")
            if rec["tainted"]:
                n_tainted += 1
                print(f"  [EXCEPTION] game {g}: {rec['exception']}", flush=True)
                continue
            if rec["capped"]:
                n_capped += 1
            nets.append(rec["hero_net"])
            hands_counts.append(rec["hands"])
            done = g - g0 + 1
            if done % args.log_every == 0:
                m = sum(nets) / len(nets)
                print(f"  {done}/{g1 - g0} games  net/game={m:+.4f}  "
                      f"[{time.time() - t0:.0f}s]", flush=True)

    n = len(nets)
    mean = sum(nets) / n if n else float("nan")
    var = (sum(x * x for x in nets) / n - mean * mean) if n > 1 else float("nan")
    stderr = math.sqrt(max(0.0, var) / n) if n > 1 else float("nan")
    summary = dict(header)
    summary["record_type"] = "summary"
    summary.update({
        "n_scored": n, "n_tainted": n_tainted, "n_capped": n_capped,
        "hero_net_per_game": mean, "stderr": stderr,
        "sigma_from_zero": abs(mean) / stderr if stderr and stderr > 0
                            else float("nan"),
        "mean_hands_per_game": sum(hands_counts) / n if n else float("nan"),
        "elapsed_s": time.time() - t0,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"shard done: n={n} net/game={mean:+.4f} +/- {stderr:.4f} "
          f"tainted={n_tainted} [{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
