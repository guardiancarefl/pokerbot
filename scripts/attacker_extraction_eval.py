"""Tier-0 exploitability measurement: trained attacker vs the frozen champion
on the SNG harness (tournament terms — the currency that matters).

Modes:
  - standard: full SNGs from 6x1500 (the v1 yardstick game loop, byte-for-
    byte via scripts/sng_baseline.play_sng_game's seat_to_policy seam).
  - bubble: games START at harvested n_alive=4 states (stacks + blind level
    drawn per game from the bubble artifact); hero is a uniformly-random
    alive seat. The exploitability MAP entry for where the EV lives.

Floors: --floors on wraps every CHAMPION seat with the live composed
policy filter (AA/KK + check-free + short-stack) — the armor layer's
adversarial value (B4). Default off = raw blueprint.

Per-game seed schedule: master_seed + 7919*g (the v1 yardstick schedule).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nlhe.game_strings import TournamentStructure
from scripts.eval_pool import CheckpointPolicy
from scripts.eval_6max_self_play import _sample_action_from_policy
from scripts.sng_baseline import (N_SEATS, _sha256_of_file, play_sng_game)
from scripts.throwaway_query_real_ante import _RealAnteStructure


class FlooredCheckpointPolicy(CheckpointPolicy):
    """CheckpointPolicy with the live composed deployment floors applied
    after inference_policy, exactly as the live consumer does.

    tail_tau=None keeps the chain byte-identical to the original B4/B5
    instrument (make_live_policy_filter never calls the tail floor when
    tail_floor_tau is None); a float arms the H1 commitment-scaled tail
    floor LAST in the chain — the TG4 re-measure configuration."""

    tail_tau: "float | None" = None

    def __init__(self, name, ckpt_path, abstraction, structure):
        super().__init__(name, ckpt_path, abstraction, structure)
        from src.nlhe.integration.live_loop import make_live_policy_filter
        self._filter = make_live_policy_filter(
            log_prefix="[FLOOR:champ]", tail_floor_tau=self.tail_tau)

    def select_action(self, parsed, state, rng, mode: str = "sample") -> int:
        return _sample_action_from_policy(
            self.solver, parsed, state, rng, mode=mode,
            policy_filter=self._filter)


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacker-ckpt", required=True)
    ap.add_argument("--champion-ckpt",
                    default="runs/k200_real_ante_20260605_225847_PRESERVED/"
                            "ckpt_iter_1500.pt")
    ap.add_argument("--abstraction",
                    default="runs/abstraction_20260521_223018_retrofit/"
                            "abstraction.pkl")
    ap.add_argument("--structure",
                    default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--master-seed", type=int, default=2026)
    ap.add_argument("--hands-per-level", type=int, default=5)
    ap.add_argument("--max-hands", type=int, default=200)
    ap.add_argument("--floors", choices=("on", "off"), default="off")
    ap.add_argument("--tail-tau", type=float, default=None,
                    help="arm the H1 tail floor in the champion's chain "
                         "(requires --floors on); None = original B4/B5 "
                         "instrument byte-identical")
    ap.add_argument("--mode", choices=("standard", "bubble"),
                    default="standard")
    ap.add_argument("--bubble-artifact",
                    default="data/training_dist_v1_bubble4.json.gz")
    ap.add_argument("--game-start", type=int, default=0)
    ap.add_argument("--game-count", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=400)
    args = ap.parse_args()

    print(f"[args] {vars(args)}", flush=True)
    print(f"[id] attacker sha256 = {_sha256_of_file(args.attacker_ckpt)}")
    print(f"[id] champion sha256 = {_sha256_of_file(args.champion_ckpt)}")

    structure = _RealAnteStructure(
        TournamentStructure.from_yaml(args.structure))
    from src.nlhe.abstraction import Abstraction
    abstr = Abstraction.load(args.abstraction)

    attacker = CheckpointPolicy("attacker", args.attacker_ckpt, abstr,
                                structure)
    if args.tail_tau is not None and args.floors != "on":
        ap.error("--tail-tau requires --floors on")
    champ_cls = (FlooredCheckpointPolicy if args.floors == "on"
                 else CheckpointPolicy)
    if args.floors == "on":
        FlooredCheckpointPolicy.tail_tau = args.tail_tau
    champion = champ_cls("champion", args.champion_ckpt, abstr, structure)

    bubble_rows = None
    if args.mode == "bubble":
        artifact = json.load(gzip.open(args.bubble_artifact, "rt"))
        bubble_rows = artifact["hand_starts"]
        print(f"[bubble] {len(bubble_rows)} n_alive=4 rows "
              f"({artifact.get('version')})")

    g_lo = args.game_start
    g_hi = (g_lo + args.game_count) if args.game_count else args.games
    nets, hands_counts = [], []
    stage_acc = {}
    n_tainted = 0
    t0 = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for g in range(g_lo, g_hi):
            seed = args.master_seed + 7919 * g
            kw = {}
            hero_seat = 0
            if bubble_rows is not None:
                row_rng = random.Random(seed ^ 0xB0BB1E)
                row = row_rng.choice(bubble_rows)
                alive = [i for i in range(N_SEATS) if row["stacks"][i] > 0]
                hero_seat = row_rng.choice(alive)
                kw = {"starting_stacks": list(row["stacks"]),
                      "starting_level": row["level"],
                      "starting_dealer": row["dealer"]}
            seat_to_policy = [attacker if i == hero_seat else champion
                              for i in range(N_SEATS)]
            rec = play_sng_game(
                attacker, champion, structure, seed=seed,
                hero_seat=hero_seat,
                hands_per_level=args.hands_per_level,
                max_hands=args.max_hands, mode="sample",
                stage_acc=stage_acc, seat_to_policy=seat_to_policy, **kw)
            rec["game"] = g
            rec["hero_seat"] = hero_seat
            fh.write(json.dumps(rec) + "\n")
            if rec["tainted"]:
                n_tainted += 1
                print(f"  [EXCEPTION] game {g}: {rec['exception']}",
                      flush=True)
                continue
            nets.append(rec["hero_net"])
            hands_counts.append(rec["hands"])
            if (g + 1 - g_lo) % args.log_every == 0:
                m, se = mean_se(nets)
                el = time.time() - t0
                print(f"  {g + 1 - g_lo}/{g_hi - g_lo}  "
                      f"net/game={m:+.4f}±{se:.4f}  [{el:.0f}s]",
                      flush=True)

    m, se = mean_se(nets)
    summary = {
        "attacker_ckpt": args.attacker_ckpt,
        "attacker_sha256": _sha256_of_file(args.attacker_ckpt),
        "champion_sha256": _sha256_of_file(args.champion_ckpt),
        "mode": args.mode, "floors": args.floors,
        "tail_tau": args.tail_tau,
        "games": len(nets), "n_tainted": n_tainted,
        "net_per_game": m, "stderr": se,
        "mean_hands_per_game": (sum(hands_counts) / len(hands_counts)
                                if hands_counts else float("nan")),
        "stage_acc": {f"{k[0]}|{k[1]}": v
                      for k, v in sorted(stage_acc.items())},
    }
    sp = out_path.with_suffix(".summary.json")
    sp.write_text(json.dumps(summary, indent=2))
    print(f"\n[RESULT mode={args.mode} floors={args.floors}] "
          f"net/game = {m:+.4f} ± {se:.4f}  (n={len(nets)}, "
          f"tainted={n_tainted})")
    print(f"wrote {sp}")


if __name__ == "__main__":
    main()
