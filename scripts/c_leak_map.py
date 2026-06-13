"""PROBE C — leak map: decompose the champion's field-battery EV-loss.

R6 already answered "is there headroom" (champion 0.0866/spot >> 0.02 -> YES).
This decomposes WHERE that headroom concentrates and confirms its NATURE
(defensive call/fold-vs-shove, not offensive) so A/B know what to target and we
understand why global-RNR's over-aggression was the wrong adjustment.

Read-only, Contabo, $0 training. Per-spot loss = oracle_EV - champion_EV over the
frozen jam-wall battery, grouped by position / depth / oracle-direction.

Usage:
  python -m scripts.c_leak_map --battery evals/h4_field_battery/battery_v1.json \
      --champion runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt \
      --out evals/c_leak_map_20260613.json
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.actions import DiscreteAction
from scripts.depth_invariance_probe import query_policy, HERO_SEAT
from scripts.fold_vs_shove_battery import build_facing_shove_spot
from scripts.field_battery import STRUCTURE_YAML


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="evals/h4_field_battery/battery_v1.json")
    ap.add_argument("--champion",
                    default="runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt")
    ap.add_argument("--out", default="evals/c_leak_map_20260613.json")
    args = ap.parse_args()

    from src.nlhe.abstraction import Abstraction
    from scripts.eval_pool import CheckpointPolicy
    from scripts.throwaway_query_real_ante import _RealAnteStructure

    battery = json.loads(Path(args.battery).read_text())
    structure = _RealAnteStructure(TournamentStructure.from_yaml(battery["structure"]))
    abstr = Abstraction.load("runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    pol = CheckpointPolicy("champion", args.champion, abstr, structure)
    fold_i = int(DiscreteAction.FOLD)

    rows = []
    for s in battery["spots"]:
        state, dealer, _bb, _sh = build_facing_shove_spot(
            structure, s["level"], s["hero_pos"], s["opener_pos"],
            tuple(s["hero_cards"]), s["depth_bb"])
        policy, mask, _p, _d, _f = query_policy(pol.solver, state, HERO_SEAT,
                                                dealer, rng_seed=0)
        p_fold = policy[fold_i] if mask[fold_i] else 0.0
        legal_mass = float(sum(policy[i] for i in range(len(policy)) if mask[i]))
        p_call = max(0.0, legal_mass - p_fold)
        tot = p_fold + p_call
        if tot <= 0:
            continue
        p_call /= tot
        ev_policy = p_call * s["ev_call"] + (1 - p_call) * s["ev_fold"]
        ev_oracle = max(s["ev_call"], s["ev_fold"])
        loss = ev_oracle - ev_policy
        rows.append({"pos": s["hero_pos"], "depth": s["depth_bb"], "level": s["level"],
                     "oracle": s["oracle"], "label": s["hero_label"],
                     "loss": loss, "p_call": p_call,
                     "oracle_call": 1.0 if s["oracle"] == "call" else 0.0})

    n = len(rows)
    total = sum(r["loss"] for r in rows)
    mean = total / n

    def group(key):
        agg = defaultdict(lambda: [0.0, 0])
        for r in rows:
            agg[r[key]][0] += r["loss"]; agg[r[key]][1] += 1
        # rank by TOTAL loss contribution (where the 0.087 concentrates)
        out = [{key: k, "mean_loss": v[0] / v[1], "n": v[1],
                "loss_share": v[0] / total} for k, v in agg.items()]
        return sorted(out, key=lambda d: -d["loss_share"])

    # defensive vs offensive: when oracle=call, is champion UNDER-calling (over-folding
    # = leaving DEFENSIVE headroom)? mean (oracle_call - p_call) over oracle=call spots.
    call_spots = [r for r in rows if r["oracle"] == "call"]
    underdef = (sum(1.0 - r["p_call"] for r in call_spots) / len(call_spots)
                if call_spots else float("nan"))
    fold_spots = [r for r in rows if r["oracle"] == "fold"]
    overcall = (sum(r["p_call"] for r in fold_spots) / len(fold_spots)
                if fold_spots else float("nan"))

    result = {
        "champion": args.champion, "n_spots": n,
        "ev_loss_mean": mean, "ev_loss_total": total,
        "by_position": group("pos"), "by_depth": group("depth"),
        "by_oracle": group("oracle"),
        "nature": {
            "oracle_call_share_of_loss": sum(r["loss"] for r in call_spots) / total,
            "mean_underdefense_on_call_oracle": underdef,
            "mean_overcall_on_fold_oracle": overcall,
            "reading": ("DEFENSIVE leak = champion over-folds vs shoves where the "
                        "oracle says call (under-defense); capturing it is a "
                        "call/fold threshold fix, NOT offensive aggression."),
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[c-leak-map] n={n} mean_loss={mean:.4f} (R6 ref 0.0866)")
    print(f"[c-leak-map] TOP positions by loss-share: "
          + ", ".join(f"{d['pos']}={d['loss_share']:.0%}(mean {d['mean_loss']:.3f})"
                      for d in result['by_position'][:4]))
    print(f"[c-leak-map] TOP depths by loss-share: "
          + ", ".join(f"{d['depth']}bb={d['loss_share']:.0%}" for d in result['by_depth'][:4]))
    print(f"[c-leak-map] oracle=call holds {result['nature']['oracle_call_share_of_loss']:.0%} "
          f"of loss; mean under-defense on call spots = {underdef:.2f} "
          f"(champion folds this fraction where oracle says call)")
    print(f"[c-leak-map] -> {args.out}")


if __name__ == "__main__":
    main()
