"""Aggregate paired A/B shards. Headline-interpretable report.

Reports:
  H1. Total hero decisions across V0 (the universe denominator).
  H2. Count + % of decisions meeting the floor condition (firing rate).
  H3. Count + % of GAMES where V1 ≠ V0 (arms diverged at least once).
  H4. Paired ICM delta V1-V0 over ALL games + SE.
  H5. Paired ICM delta V1-V0 restricted to ONLY diverged games + SE.
       ← the real per-firing effect
  H6. V0 incoherence rate among floored decisions (mass on intermediate
       bet sizes — the seq=461-class error rate at field scale).
  H7. Outcome correlation: ICM × # incoherent ≤BB-floored decisions.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

# DiscreteAction enum (live order):
#   FOLD=0, CALL=1, BET_33=2, BET_66=3, BET_100=4, BET_200=5, ALLIN=6,
#   BET_50=7, BET_150=8
COHERENT_IDXS = (0, 1, 6)               # FOLD, CALL, ALLIN
INTERMEDIATE_IDXS = (2, 3, 4, 5, 7, 8)  # BET_33/66/100/200/50/150
DA_NAMES = ["FOLD", "CALL", "BET_33", "BET_66", "BET_100", "BET_200",
            "ALLIN", "BET_50", "BET_150"]


def load_shards(shards_glob):
    shard_dirs = sorted(glob.glob(shards_glob))
    games = []
    v0_decisions_path = []
    v1_decisions_path = []
    for s in shard_dirs:
        gp = Path(s) / "games.jsonl"
        if not gp.exists():
            print(f"  WARN: {gp} missing")
            continue
        with open(gp) as f:
            for line in f:
                games.append(json.loads(line))
        v0_decisions_path.append(Path(s) / "decisions_v0.jsonl")
        v1_decisions_path.append(Path(s) / "decisions_v1.jsonl")
    return shard_dirs, games, v0_decisions_path, v1_decisions_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-glob", default="evals/short_stack_floor_ab/shard_*")
    ap.add_argument("--threshold-bb", type=float, default=6.0)
    ap.add_argument("--label", type=str, default=None,
                     help="Label for the headline (e.g. 'hpl=3' or 'hpl=5')")
    args = ap.parse_args()

    label = args.label or args.shards_glob
    print(f"\n{'='*72}")
    print(f"  PAIRED A/B AGGREGATE — {label}")
    print(f"  shards-glob: {args.shards_glob}")
    print(f"  threshold_bb: {args.threshold_bb}")
    print(f"{'='*72}")

    shard_dirs, games, v0_paths, v1_paths = load_shards(args.shards_glob)
    if not games:
        print(" no games found")
        return
    N = len(games)
    print(f"\n  shards: {len(shard_dirs)}   paired games: {N}")

    # ---------- Decision-level scan (one streaming pass per shard) ----------
    # We want:
    #   - total V0 decisions
    #   - V0 ≤thr-BB count + facing/check/unopened breakdown
    #   - V1 floor-fired count
    #   - V1 floor-fired BY REGIME
    #   - V0 incoherence on the spots where V1 would fire (= V0 ≤thr-BB facing)
    #   - per-game V0 ICM × incoherent-decision count for outcome correlation
    n_v0_decisions = 0
    n_v0_short_total = 0
    n_v0_short_facing = 0
    n_v0_short_check = 0
    n_v0_short_unopened = 0
    incoh_v0_at_fire = []  # raw_policy mass on intermediate-bets at V0 ≤thr-BB facing
    per_game_incoh_count = {}  # game_id -> count of decisions with intermediate-mass > 0.20
    for p in v0_paths:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                n_v0_decisions += 1
                eff_bb = float(r["hero_eff_bb"])
                if eff_bb > args.threshold_bb:
                    continue
                n_v0_short_total += 1
                rawp = np.asarray(r["raw_policy"], dtype=np.float64)
                legal = np.asarray(r["legal_mask"], dtype=np.float64)
                coh = float(sum(rawp[i] for i in COHERENT_IDXS if legal[i] > 0))
                incoh = float(sum(
                    rawp[i] for i in INTERMEDIATE_IDXS if legal[i] > 0))
                facing = bool(r["facing_action"])
                if facing:
                    n_v0_short_facing += 1
                    incoh_v0_at_fire.append(incoh)
                    if incoh > 0.20:
                        gid = int(r["game"])
                        per_game_incoh_count[gid] = per_game_incoh_count.get(gid, 0) + 1
                else:
                    if legal[1] > 0:
                        n_v0_short_check += 1
                    else:
                        n_v0_short_unopened += 1

    n_v1_decisions = 0
    n_v1_fires = 0
    fires_by_regime = {"facing": 0, "check": 0, "unopened": 0}
    for p in v1_paths:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                n_v1_decisions += 1
                if r["floor_fired"]:
                    n_v1_fires += 1
                    reg = r.get("floor_regime", "?")
                    fires_by_regime[reg] = fires_by_regime.get(reg, 0) + 1

    # ---------- H1, H2 ----------
    print(f"\n[H1] DECISIONS")
    print(f"  total V0 hero decisions:  {n_v0_decisions:>8d}")
    print(f"  total V1 hero decisions:  {n_v1_decisions:>8d}")
    print(f"  (note: V1 path may diverge → different decision count)")

    print(f"\n[H2] FLOOR-FIRING RATE  (decisions where V1 would mask)")
    print(f"  V0 decisions ≤{args.threshold_bb}BB:                "
          f"{n_v0_short_total:>6d} ({100*n_v0_short_total/max(1,n_v0_decisions):.2f}% of V0 decisions)")
    print(f"   - facing action (→ FOLD/CALL/ALLIN):  "
          f"{n_v0_short_facing:>6d}")
    print(f"   - check option (→ CALL/ALLIN):        "
          f"{n_v0_short_check:>6d}")
    print(f"   - unopened (→ FOLD/ALLIN):            "
          f"{n_v0_short_unopened:>6d}")
    print(f"  V1 actually-floored decisions:        "
          f"{n_v1_fires:>6d} ({100*n_v1_fires/max(1,n_v1_decisions):.2f}% of V1 decisions)")
    print(f"   by regime:  {fires_by_regime}")

    # ---------- H3 ----------
    print(f"\n[H3] ARM DIVERGENCE  (games where V1 ≠ V0)")
    deltas = np.array([g["delta"] for g in games], dtype=np.float64)
    nz = int((deltas != 0).sum())
    print(f"  total games:                       {N:>5d}")
    print(f"  games with V1 ≠ V0 (any divergence): "
          f"{nz:>5d}  ({100*nz/N:.2f}%)")
    print(f"  delta distribution:")
    for v in sorted(set(deltas.tolist())):
        print(f"     {v:+.1f}:  {int((deltas == v).sum()):>5d}  "
              f"({100*(deltas == v).mean():.2f}%)")

    # ---------- H4, H5 ----------
    v0_arr = np.array([g["v0_icm"] for g in games], dtype=np.float64)
    v1_arr = np.array([g["v1_icm"] for g in games], dtype=np.float64)
    mean_all = float(deltas.mean())
    sd_all = float(deltas.std(ddof=1)) if N > 1 else 0.0
    se_all = sd_all / math.sqrt(N) if N > 0 else 0.0

    div = deltas[deltas != 0]
    if div.size > 0:
        mean_div = float(div.mean())
        sd_div = float(div.std(ddof=1)) if div.size > 1 else 0.0
        se_div = sd_div / math.sqrt(div.size) if div.size > 0 else 0.0
    else:
        mean_div = sd_div = se_div = 0.0

    print(f"\n[H4] PAIRED ICM DELTA V1−V0  — ALL GAMES")
    print(f"  N games:           {N}")
    print(f"  V0 mean ICM:       {v0_arr.mean():+.4f}   "
          f"P(cash) = {(v0_arr > 0).mean():.4f}")
    print(f"  V1 mean ICM:       {v1_arr.mean():+.4f}   "
          f"P(cash) = {(v1_arr > 0).mean():.4f}")
    print(f"  paired mean delta: {mean_all:+.5f}")
    print(f"  paired SD:         {sd_all:.4f}")
    print(f"  paired SE:         {se_all:.5f}")
    if se_all > 0:
        z = mean_all / se_all
        ci_lo = mean_all - 1.96 * se_all
        ci_hi = mean_all + 1.96 * se_all
        print(f"  z = mean/SE:       {z:+.2f}")
        print(f"  95% CI:            [{ci_lo:+.5f}, {ci_hi:+.5f}]")

    print(f"\n[H5] PAIRED ICM DELTA V1−V0  — DIVERGED GAMES ONLY")
    print(f"  (the per-firing effect: among games where the floor changed an action,")
    print(f"   did the change help or hurt?)")
    print(f"  N diverged:        {div.size}")
    if div.size > 0:
        print(f"  mean delta:        {mean_div:+.5f}")
        print(f"  SD:                {sd_div:.4f}")
        print(f"  SE:                {se_div:.5f}")
        if se_div > 0:
            z = mean_div / se_div
            ci_lo = mean_div - 1.96 * se_div
            ci_hi = mean_div + 1.96 * se_div
            print(f"  z = mean/SE:       {z:+.2f}")
            print(f"  95% CI:            [{ci_lo:+.5f}, {ci_hi:+.5f}]")

    # ---------- H6 ----------
    print(f"\n[H6] V0 INCOHERENCE  (mass on intermediate-bet sizes at floor-fire spots)")
    if incoh_v0_at_fire:
        ic = np.asarray(incoh_v0_at_fire)
        print(f"  N decisions (V0 ≤{args.threshold_bb}BB facing action): {len(ic)}")
        print(f"  mean intermediate-bet mass:  {ic.mean():.4f}")
        print(f"  median:                      {np.median(ic):.4f}")
        print(f"  percentiles 25/75/90/95:     "
              f"{np.percentile(ic,25):.4f} / "
              f"{np.percentile(ic,75):.4f} / "
              f"{np.percentile(ic,90):.4f} / "
              f"{np.percentile(ic,95):.4f}")
        gt20 = int((ic > 0.20).sum())
        gt40 = int((ic > 0.40).sum())
        gt60 = int((ic > 0.60).sum())
        print(f"  >20% intermediate mass (seq=461-class):  "
              f"{gt20:>6d}  ({100*gt20/len(ic):.2f}%)")
        print(f"  >40% intermediate mass (strong):         "
              f"{gt40:>6d}  ({100*gt40/len(ic):.2f}%)")
        print(f"  >60% intermediate mass (severe):         "
              f"{gt60:>6d}  ({100*gt60/len(ic):.2f}%)")
    else:
        print(f"  no V0 ≤{args.threshold_bb}BB facing-action decisions observed")

    # ---------- H7 ----------
    print(f"\n[H7] OUTCOME CORRELATION  (V0 ICM × # incoherent ≤{args.threshold_bb}BB-facing-action V0 decisions)")
    n_inc_per_g = []
    v0_per_g = []
    for g in games:
        gid = g["game"]
        n_inc_per_g.append(per_game_incoh_count.get(gid, 0))
        v0_per_g.append(g["v0_icm"])
    if len(n_inc_per_g) > 1:
        a = np.asarray(n_inc_per_g, dtype=np.float64)
        b = np.asarray(v0_per_g, dtype=np.float64)
        if a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a, b)[0, 1])
            print(f"  Pearson r:  {r:+.4f}")
        else:
            print(f"  (zero variance — skip Pearson r)")
        buckets = {}
        for ni, v in zip(n_inc_per_g, v0_per_g):
            k = ni if ni < 4 else 4
            buckets.setdefault(k, []).append(v)
        print(f"  mean V0 ICM by # incoherent decisions in game:")
        print(f"    {'bucket':<10s}  {'n_games':>8s}  {'mean V0 ICM':>12s}  "
              f"{'P(cash)':>9s}")
        for k in sorted(buckets):
            arr = np.asarray(buckets[k])
            lbl = f"≥{k}" if k == 4 else str(k)
            print(f"    {lbl:<10s}  {len(arr):>8d}  {arr.mean():>+12.4f}  "
              f"{(arr > 0).mean():>9.4f}")

    # ---------- Per-game floor-fire conditional delta ----------
    n_fire_per_g = {}
    for p in v1_paths:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r["floor_fired"]:
                    gid = int(r["game"])
                    n_fire_per_g[gid] = n_fire_per_g.get(gid, 0) + 1
    print(f"\n[H7b] CONDITIONAL DELTA  (V1−V0 | # V1 floor-fires in this game)")
    fires_arr = np.asarray([n_fire_per_g.get(g["game"], 0) for g in games])
    buckets = {}
    for ni, d in zip(fires_arr.tolist(), deltas.tolist()):
        k = ni if ni < 4 else 4
        buckets.setdefault(k, []).append(d)
    print(f"    {'#fires':<10s}  {'n_games':>8s}  {'mean delta':>12s}  "
          f"{'SE':>8s}  {'p_diverge':>10s}")
    for k in sorted(buckets):
        arr = np.asarray(buckets[k])
        sd = arr.std(ddof=1) if len(arr) > 1 else 0
        se_k = sd / math.sqrt(len(arr)) if len(arr) > 0 else 0
        lbl = f"≥{k}" if k == 4 else str(k)
        nz = int((arr != 0).sum())
        print(f"    {lbl:<10s}  {len(arr):>8d}  {arr.mean():>+12.5f}  "
              f"{se_k:>8.5f}  {100*nz/max(1,len(arr)):>9.2f}%")


if __name__ == "__main__":
    main()
