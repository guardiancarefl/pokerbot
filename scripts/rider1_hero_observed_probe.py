"""Rider-1 probe: hero-OBSERVED game length vs full-table game length.

The v1 yardstick's hands/game counts hands until the TABLE reaches 3
alive — the loop keeps simulating after the hero busts. The live corpus
counts hands until the HERO stops observing (bust or cash). This probe
replays exact v1 seeds (CRN: master + 7919*g) with the same loop and
records the hand at which the hero's outcome was decided, verifying
hero_net per seed against the v1 jsonl as a replay-correctness check.

The loop body is copied verbatim from scripts.sng_baseline.play_sng_game
(same RNG consumption order) with one added observation: hero_decided_hand.
Analysis-only; v1 code and artifacts untouched.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.sng_baseline import (N_SEATS, _blind_guard, play_one_hand_sng)


def play_probe_game(seat_to_policy, structure, *, seed, hero_seat=0,
                     starting_stack=1500, hands_per_level=5, max_hands=200,
                     mode="sample"):
    rng = random.Random(seed)
    stacks = [starting_stack] * N_SEATS
    max_level = max(bl.level for bl in structure.blind_schedule)
    level = 1
    dealer = rng.randrange(N_SEATS)
    hands_played = 0
    hands_in_level = 0
    rec = {"seed": seed, "tainted": False, "capped": False}
    hero_decided_hand = None        # hand idx when hero bust or table hit 3

    while True:
        n_alive = sum(1 for s in stacks if s > 0)
        if n_alive <= 3:
            break
        if hands_played >= max_hands:
            rec["capped"] = True
            break
        if hands_in_level >= hands_per_level:
            level = min(level + 1, max_level)
            hands_in_level = 0
        stacks, dealer = _blind_guard(structure, stacks, level, dealer)
        if dealer is None or sum(1 for s in stacks if s > 0) <= 3:
            break
        try:
            stacks = play_one_hand_sng(seat_to_policy, structure, stacks,
                                        level, dealer, rng, mode=mode)
        except Exception:
            rec["tainted"] = True
            break
        hands_played += 1
        hands_in_level += 1
        if hero_decided_hand is None and stacks[hero_seat] <= 0:
            hero_decided_hand = hands_played      # hero busted here
        dealer = (dealer + 1) % N_SEATS
        guard = 0
        while stacks[dealer] == 0 and guard < N_SEATS:
            dealer = (dealer + 1) % N_SEATS
            guard += 1

    if hero_decided_hand is None:
        hero_decided_hand = hands_played          # hero observed to the end
    rec.update({
        "hands": hands_played,
        "hero_observed_hands": hero_decided_hand,
        "level_end": level,
        "hero_net": (1.0 if stacks[hero_seat] > 0 else -1.0)
                    if not rec["capped"] else None,
    })
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--abstraction", required=True)
    ap.add_argument("--structure",
                     default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--shanky-dir", default="data/shanky_profiles")
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--master-seed", type=int, default=2026)
    ap.add_argument("--v1-dir", default="evals/sng_baseline_20260610",
                     help="v1 run dir for per-seed hero_net cross-check")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from scripts.eval_pool import CheckpointPolicy
    from scripts.bake_off_real_ante import build_shanky_pool
    from scripts.throwaway_query_real_ante import _RealAnteStructure

    structure = _RealAnteStructure(TournamentStructure.from_yaml(args.structure))
    abstr = Abstraction.load(args.abstraction)
    hero = CheckpointPolicy(name="k200", ckpt_path=args.ckpt,
                             abstraction=abstr, structure=structure)
    wanted = [p.strip() for p in args.profiles.split(",")]
    pool = {p.name: p for p in build_shanky_pool(args.shanky_dir)
            if p.name in wanted}

    # v1 per-seed hero_net for the replay cross-check
    v1_net = {}
    for name in wanted:
        for jp in Path(args.v1_dir).glob(f"w*/games_{name}.jsonl"):
            for line in open(jp):
                r = json.loads(line)
                v1_net[(name, r["seed"])] = r["hero_net"]

    out = {}
    for name in wanted:
        opp = pool[name]
        seat_to_policy = [hero] + [opp] * (N_SEATS - 1)
        rows, mismatches = [], 0
        for g in range(args.games):
            seed = args.master_seed + 7919 * g
            rec = play_probe_game(seat_to_policy, structure, seed=seed)
            ref = v1_net.get((name, seed))
            rec["v1_match"] = (ref is None or rec["hero_net"] == ref)
            if not rec["v1_match"]:
                mismatches += 1
            rows.append(rec)
        obs = [r["hero_observed_hands"] for r in rows if not r["tainted"]]
        tab = [r["hands"] for r in rows if not r["tainted"]]
        out[name] = {
            "n": len(rows), "replay_mismatches": mismatches,
            "table_hands_mean": sum(tab) / len(tab),
            "hero_observed_hands_mean": sum(obs) / len(obs),
            "hero_observed_median": sorted(obs)[len(obs) // 2],
            "rows": rows,
        }
        print(f"{name}: table={out[name]['table_hands_mean']:.1f} "
              f"hero-observed={out[name]['hero_observed_hands_mean']:.1f} "
              f"(median {out[name]['hero_observed_median']}) "
              f"replay_mismatches={mismatches}", flush=True)
    Path(args.out).write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
