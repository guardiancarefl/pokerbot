"""Pre-ship candidate bake-off: shared-CRN sweep across 4 checkpoints.

Each candidate sees the SAME starting state per hand index and the SAME
opponent-RNG seed. ICM/hand columns are directly comparable across
candidates (subject to the standard CRN caveat: action divergence at
decision points causes opponent-RNG drift, but the environment/cards
remain shared up to divergence).

Candidates:
  (1) k200_iter_2000   — bare k=200 blueprint (canonical, sha=b1981e3e14a9)
  (2) k200_iter_2800   — late-iter k=200 blueprint (sha=5297ead797f2)
  (3) k1000_iter_0527  — partial k=1000 blueprint (UNDERTRAINED — interrupted)
  (4) rebel_d3k150     — value-only ReBeL d=3 k=150 resolver
                           (uses k200_iter_2000 as blueprint warm-start)

Matchups (opponents):
  KillPhilMTT (tight shanky), NIT (tight archetype),
  STATION (loose archetype), MANIAC (loose archetype),
  LAG (mid archetype)

500 paired-CRN hands per (candidate, matchup) = 2500 hands per candidate
= 10,000 hand-evals total.

Output: stdout table + JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.nlhe.abstraction import Abstraction  # noqa: E402
from src.nlhe.archetypes import EquityCalibration  # noqa: E402
from src.nlhe.game_strings import TournamentStructure  # noqa: E402
from src.nlhe.icm import sng_payouts_6max_double_up  # noqa: E402
from src.nlhe.stack_sampler import sample_starting_state  # noqa: E402
from scripts.eval_6max_self_play import _load_solver  # noqa: E402
from scripts.rebel_gate2 import (  # noqa: E402
    BlueprintHero,
    ReBeLHero,
    build_opponent_factory,
    mlp,
    play_hand,
)

STRUCT = "configs/ignition_double_up_6max_turbo.yaml"
CALIB = "runs/archetype_design/bucket_equity_analysis_6max.json"
ABSTR_K200 = "runs/k200_abstraction.pkl"
ABSTR_K1000 = "runs/abstraction_k1000_retrofit_20260530_221744/abstraction.pkl"

CKPT_K200_2000 = "runs/six_max_20260530_034023_phase4f_dcfr_candC_k200/checkpoints/ckpt_iter_2000.pt"
CKPT_K200_2800 = "runs/six_max_20260530_034023_phase4f_dcfr_candC_k200/checkpoints/ckpt_iter_2800.pt"
CKPT_K1000_527 = "runs/six_max_20260602_135657_candC_k1000_launch/checkpoints/ckpt_iter_0527.pt"
REBEL_VALUE_NET = "/home/quant/pokerbot/rebel_value_net_full.pt"


def build_candidates(structure: TournamentStructure, payouts):
    """Returns dict: label -> (hero, kind, sha1_12, abstraction_used_for_opps)."""
    print("[load] k200 abstraction")
    abs_k200 = Abstraction.load(ABSTR_K200)
    print("[load] k1000 abstraction")
    abs_k1000 = Abstraction.load(ABSTR_K1000)

    print(f"[load] k200 iter_2000 solver  ({CKPT_K200_2000})")
    solver_k200_2000 = _load_solver(CKPT_K200_2000, abs_k200, structure)
    print(f"[load] k200 iter_2800 solver  ({CKPT_K200_2800})")
    solver_k200_2800 = _load_solver(CKPT_K200_2800, abs_k200, structure)
    print(f"[load] k1000 iter_0527 solver ({CKPT_K1000_527})")
    solver_k1000_527 = _load_solver(CKPT_K1000_527, abs_k1000, structure)

    print(f"[load] rebel value net ({REBEL_VALUE_NET})")
    cck = torch.load(REBEL_VALUE_NET, map_location="cpu", weights_only=False)
    cnet = mlp(cck["in_dim"], tuple(cck["hidden"]))
    cnet.load_state_dict(
        {kk.replace("net.", "", 1): vv for kk, vv in cck["state_dict"].items()}
    )
    cnet.eval()

    candidates = {
        "k200_iter2000": (BlueprintHero(solver_k200_2000), "blueprint_k200", abs_k200),
        "k200_iter2800": (BlueprintHero(solver_k200_2800), "blueprint_k200", abs_k200),
        "k1000_iter0527": (BlueprintHero(solver_k1000_527), "blueprint_k1000", abs_k1000),
        "rebel_d3k150": (
            ReBeLHero(solver_k200_2000, abs_k200, cnet, cck, payouts,
                       depth=3, n_iters=150, weighting="linear",
                       warm_start=True, belief="uniform"),
            "rebel_d3k150",
            abs_k200,
        ),
    }
    return candidates


PANEL = [
    ("KillPhilMTT", ("shanky", "KillPhilMTT.txt")),
    ("NIT",         ("archetype", "NIT")),
    ("STATION",     ("archetype", "STATION")),
    ("MANIAC",      ("archetype", "MANIAC")),
    ("LAG",         ("archetype", "LAG")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-json",
                    default="runs/phase1_d128_repro/candidate_bakeoff.json")
    args = ap.parse_args()

    t0 = time.time()
    structure = TournamentStructure.from_yaml(STRUCT)
    calib = EquityCalibration.load(CALIB)
    payouts = list(sng_payouts_6max_double_up())

    candidates = build_candidates(structure, payouts)
    cand_order = ["k200_iter2000", "k200_iter2800", "k1000_iter0527", "rebel_d3k150"]
    # Opponent factories are built ONCE per matchup (against k200 abstraction —
    # the calibration is k200-aligned; archetype opps reference k200 buckets
    # regardless of which candidate hero is acting).
    abs_for_opps = candidates["k200_iter2000"][2]
    factories = {lbl: build_opponent_factory(spec, abs_for_opps, calib, structure)
                 for lbl, spec in PANEL}

    print(f"\nBAKE-OFF — {args.hands} shared-CRN hands per (candidate, matchup)\n")
    print(f"{'matchup':<14s}  " + "  ".join(f"{c:>16s}" for c in cand_order),
          flush=True)
    print("-" * (14 + 4 + len(cand_order) * 18), flush=True)

    # results[matchup][candidate] = list of per-hand ICM
    results: dict[str, dict[str, list[float]]] = {
        lbl: {c: [] for c in cand_order} for lbl, _ in PANEL
    }

    for lbl, spec in PANEL:
        t_m = time.time()
        factory = factories[lbl]
        for i in range(args.hands):
            base = args.seed + i * 7919
            sm = sample_starting_state(
                structure, random.Random(base * 2 + 1), num_paid=3)
            alive = [s for s in range(6) if sm["stacks"][s] > 0]
            hero_seat = random.Random(base * 3 + 5).choice(alive)
            dealer = sm["dealer_seat"]
            gs = structure.to_inner_game_string_for_state(
                blind_level=sm["blind_level"], stacks=sm["stacks"],
                dealer_seat=dealer)
            opps = factory(sm["blind_level"].level)
            bb_chips = sm["blind_level"].inflated_big_blind(6)

            for cname in cand_order:
                hero, _, _ = candidates[cname]
                hero.new_hand(dealer, bb=bb_chips)
                v = play_hand(gs, sm, hero_seat, hero, opps, dealer, base)
                if v is not None:
                    results[lbl][cname].append(v)
        wall_m = time.time() - t_m
        # Per-matchup row.
        row_vals = []
        for cname in cand_order:
            arr = results[lbl][cname]
            mean = statistics.mean(arr) if arr else 0.0
            se = (statistics.pstdev(arr) / (len(arr) ** 0.5)) if len(arr) > 1 else 0.0
            row_vals.append(f"{mean:>+8.4f} ±{2*se:.4f}")
        print(f"{lbl:<14s}  " + "  ".join(f"{v:>16s}" for v in row_vals)
              + f"   [{wall_m/60:.1f}min]", flush=True)

    # Panel summary per candidate
    print()
    print("=== Panel summary per candidate ===")
    print(f"{'metric':<22s}  " + "  ".join(f"{c:>16s}" for c in cand_order))
    panel_means = {}
    panel_worsts = {}
    for cname in cand_order:
        per_match = [statistics.mean(results[lbl][cname]) if results[lbl][cname] else 0.0
                     for lbl, _ in PANEL]
        panel_means[cname] = statistics.mean(per_match)
        panel_worsts[cname] = min(per_match)
    print(f"{'panel_mean ICM/hand':<22s}  " +
          "  ".join(f"{panel_means[c]:>+16.4f}" for c in cand_order))
    print(f"{'worst_matchup ICM/hand':<22s}  " +
          "  ".join(f"{panel_worsts[c]:>+16.4f}" for c in cand_order))

    # ReBeL head-to-head vs k200_iter2000
    print()
    print("=== Head-to-head: rebel_d3k150  vs  k200_iter2000 (Δ = rebel - blueprint) ===")
    h2h = {}
    for lbl, _ in PANEL:
        ra = results[lbl]["rebel_d3k150"]
        ka = results[lbl]["k200_iter2000"]
        n = min(len(ra), len(ka))
        if n == 0:
            continue
        d = [ra[i] - ka[i] for i in range(n)]
        md = statistics.mean(d)
        se = (statistics.pstdev(d) / (n ** 0.5)) if n > 1 else 0.0
        verdict = ("rebel+" if md > 2 * se
                   else "blueprint+" if md < -2 * se else "~tie")
        h2h[lbl] = {"delta": md, "se": se, "verdict": verdict}
        print(f"  {lbl:<14s}  Δ={md:>+8.4f} ±{2*se:.4f}   verdict={verdict}")

    payload = {
        "config": {
            "n_hands": int(args.hands),
            "seed": int(args.seed),
            "candidates": cand_order,
            "matchups": [lbl for lbl, _ in PANEL],
        },
        "per_matchup": {
            lbl: {
                cname: {
                    "n": len(results[lbl][cname]),
                    "mean_icm_per_hand": (statistics.mean(results[lbl][cname])
                                           if results[lbl][cname] else None),
                    "se_icm": (statistics.pstdev(results[lbl][cname])
                                / (len(results[lbl][cname]) ** 0.5)
                                if len(results[lbl][cname]) > 1 else 0.0),
                    "samples": results[lbl][cname],
                }
                for cname in cand_order
            }
            for lbl, _ in PANEL
        },
        "panel_summary": {
            cname: {
                "panel_mean": panel_means[cname],
                "worst_matchup": panel_worsts[cname],
            }
            for cname in cand_order
        },
        "rebel_vs_k200_head_to_head": h2h,
        "wall_seconds": float(time.time() - t0),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[saved] {out}")
    print(f"[done] total wall = {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
