"""C3 prep — empirical training-distribution harvest from self-play SNGs.

Plays full self-play tournaments in the calibration-row configuration
(hero checkpoint in all six seats, per-seat RNG streams from the
per-game seed, hpl=5, v1 CRN seed schedule master + 7919*g) and records
the joint hand-start state (blind_level, n_alive, stack vector, dealer
seat) for every hand played. The merged, versioned artifact becomes the
C3 trainer's empirical sampling distribution.

The game loop is the replay-verified rider-1 probe loop (exact
sng_baseline RNG consumption order; hero_net cross-checked 0/2000
mismatches against v1 jsonls) with one added observation per hand.

Shardable via --game-start/--game-count. Merge with --merge, which
writes the artifact + a summary table and evaluates PRE-GATE H1
(L6+ hand mass must be <= --h1-threshold, default 10%).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.sng_baseline import (N_SEATS, _blind_guard, _git_identity,
                                  _sha256_of_file, play_one_hand_sng)
from scripts.sng_selfplay_calibration import SEAT_SEED_STRIDE, SeatRNGPolicy

ARTIFACT_VERSION = "training_dist_v1"


def play_harvest_game(seat_to_policy, structure, *, seed, out_rows,
                       starting_stack=1500, hands_per_level=5,
                       max_hands=200, mode="sample"):
    """Calibration-row game loop + one hand-start record per hand."""
    rng = random.Random(seed)
    stacks = [starting_stack] * N_SEATS
    max_level = max(bl.level for bl in structure.blind_schedule)
    level = 1
    dealer = rng.randrange(N_SEATS)
    hands_played = 0
    hands_in_level = 0
    tainted = False
    while True:
        n_alive = sum(1 for s in stacks if s > 0)
        if n_alive <= 3:
            break
        if hands_played >= max_hands:
            break
        if hands_in_level >= hands_per_level:
            level = min(level + 1, max_level)
            hands_in_level = 0
        stacks, dealer = _blind_guard(structure, stacks, level, dealer)
        if dealer is None or sum(1 for s in stacks if s > 0) <= 3:
            break
        # ---- the harvested observation: the post-blind-guard hand start
        out_rows.append({
            "seed": seed,
            "hand": hands_played + 1,
            "level": level,
            "n_alive": sum(1 for s in stacks if s > 0),
            "stacks": list(stacks),
            "dealer": dealer,
        })
        try:
            stacks = play_one_hand_sng(seat_to_policy, structure, stacks,
                                        level, dealer, rng, mode=mode)
        except Exception:
            tainted = True
            break
        hands_played += 1
        hands_in_level += 1
        dealer = (dealer + 1) % N_SEATS
        guard = 0
        while stacks[dealer] == 0 and guard < N_SEATS:
            dealer = (dealer + 1) % N_SEATS
            guard += 1
    return hands_played, tainted


def merge(out_dir: Path, artifact_path: Path, h1_threshold: float):
    rows = []
    for jp in sorted(out_dir.glob("harvest_w*/hand_starts.jsonl")):
        for line in jp.read_text().splitlines():
            rows.append(json.loads(line))
    metas = [json.loads(p.read_text())
             for p in sorted(out_dir.glob("harvest_w*/summary.json"))]
    n_games = sum(m["n_games"] for m in metas)
    n_tainted = sum(m["n_tainted"] for m in metas)

    lvl = {}
    alive = {}
    for r in rows:
        lvl[r["level"]] = lvl.get(r["level"], 0) + 1
        alive[r["n_alive"]] = alive.get(r["n_alive"], 0) + 1
    tot = len(rows)
    l6plus = sum(c for l, c in lvl.items() if l >= 6) / tot
    h1_pass = l6plus <= h1_threshold

    artifact = {
        "version": ARTIFACT_VERSION,
        "record_type": "training_distribution",
        "source": "self-play harvest (calibration-row config, hpl=5)",
        "git_head": _git_identity()[0],
        "ckpt_sha256": metas[0]["ckpt_sha256"],
        "n_games": n_games,
        "n_tainted_games": n_tainted,
        "n_hand_starts": tot,
        "level_hist": {str(k): v for k, v in sorted(lvl.items())},
        "n_alive_hist": {str(k): v for k, v in sorted(alive.items())},
        "h1_gate": {"l6plus_mass": l6plus, "threshold": h1_threshold,
                     "pass": h1_pass},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hand_starts": rows,
    }
    raw = json.dumps(artifact).encode()
    if str(artifact_path).endswith(".gz"):
        artifact_path.write_bytes(gzip.compress(raw))
    else:
        artifact_path.write_bytes(raw)

    print(f"artifact: {artifact_path}  ({len(raw)/1e6:.1f} MB raw, "
          f"{tot} hand starts from {n_games} games, {n_tainted} tainted)")
    print(f"\nlevel distribution (hand starts):")
    for l in sorted(lvl):
        bar = "#" * int(60 * lvl[l] / max(lvl.values()))
        print(f"  L{l:<2} {lvl[l]:>7}  {100*lvl[l]/tot:5.1f}%  {bar}")
    print(f"\nn_alive distribution:")
    for a in sorted(alive, reverse=True):
        print(f"  {a}-handed {alive[a]:>7}  {100*alive[a]/tot:5.1f}%")
    print(f"\nPRE-GATE H1: L6+ mass = {100*l6plus:.2f}%  "
          f"(threshold {100*h1_threshold:.0f}%)  -> "
          f"{'PASS' if h1_pass else 'FAIL'}")
    return 0 if h1_pass else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--abstraction")
    ap.add_argument("--structure",
                     default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--game-start", type=int, default=0)
    ap.add_argument("--game-count", type=int, default=None)
    ap.add_argument("--master-seed", type=int, default=2026)
    ap.add_argument("--hands-per-level", type=int, default=5)
    ap.add_argument("--max-hands", type=int, default=200)
    ap.add_argument("--out-dir")
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--merge", metavar="DIR")
    ap.add_argument("--artifact", default="data/training_dist_v1.json.gz")
    ap.add_argument("--h1-threshold", type=float, default=0.10)
    args = ap.parse_args()

    if args.merge:
        return merge(Path(args.merge), Path(args.artifact), args.h1_threshold)
    if not args.ckpt or not args.abstraction or not args.out_dir:
        raise SystemExit("--ckpt/--abstraction/--out-dir required")

    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from scripts.eval_pool import CheckpointPolicy
    from scripts.throwaway_query_real_ante import _RealAnteStructure

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g0 = args.game_start
    g1 = g0 + (args.game_count if args.game_count is not None
               else args.games - g0)

    structure = _RealAnteStructure(TournamentStructure.from_yaml(args.structure))
    abstr = Abstraction.load(args.abstraction)
    hero = CheckpointPolicy(name="k200", ckpt_path=args.ckpt,
                             abstraction=abstr, structure=structure)
    print(f"harvest shard games [{g0},{g1}) master_seed={args.master_seed}",
          flush=True)

    n_tainted = 0
    n_hands = 0
    t0 = time.time()
    with open(out_dir / "hand_starts.jsonl", "w") as fh:
        for g in range(g0, g1):
            seed = args.master_seed + 7919 * g
            seat_to_policy = [SeatRNGPolicy(hero, seed * SEAT_SEED_STRIDE + i)
                              for i in range(N_SEATS)]
            rows = []
            hands, tainted = play_harvest_game(
                seat_to_policy, structure, seed=seed, out_rows=rows,
                hands_per_level=args.hands_per_level,
                max_hands=args.max_hands)
            if tainted:
                n_tainted += 1
                continue            # taint discipline: exclude whole game
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            n_hands += len(rows)
            done = g - g0 + 1
            if done % args.log_every == 0:
                print(f"  {done}/{g1-g0} games  {n_hands} hand starts  "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    summary = {
        "record_type": "harvest_shard_summary",
        "ckpt_sha256": _sha256_of_file(args.ckpt),
        "game_range": [g0, g1],
        "n_games": g1 - g0,
        "n_tainted": n_tainted,
        "n_hand_starts": n_hands,
        "master_seed": args.master_seed,
        "hands_per_level": args.hands_per_level,
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"shard done: {n_hands} hand starts, tainted={n_tainted} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
