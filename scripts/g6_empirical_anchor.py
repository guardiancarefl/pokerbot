"""G6 empirical-anchor probe — Phase-2 premise check.

Re-runs the G6 opponent-separation test against an EMPIRICAL blueprint anchor
(the head's own mean output for blueprint-vs-blueprint matches) instead of
the ground-truth SeatStats reference used in the original G6 probe.

Premise: the original G6 used tendency_blueprint = ground-truth SeatStats from
blueprint self-play. But the tendency head was never trained on
blueprint-as-opp samples (pool gen used gus/kill/loose only). At probe time
the head regresses toward its archetype-trained output for blueprint inputs,
which is far from the ground-truth measurement — making the d_bar ratio test
falsely report "WEAK". The empirical anchor (head's own blueprint-conditioned
output) is the correct reference, by the same logic that justifies an
empirical anchor in the Leduc head-derived read (gate(0)=0 by construction
must use the head's natural zero point, not an external one).

Inference-only; ~1 CPU-min. Loads runs/phase1_d128_repro/smoke_net.pt.

Output: prints empirical anchor, per-opp d_bar, separation ratios, verdict;
writes JSON to runs/phase1_d128_repro/g6_empirical_anchor.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Lift reusable harness from the smoke pipeline
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nlhe.adaptive.model import (  # noqa: E402
    Adaptive6MaxNet, K_TENDENCY, collate, tendency_l2_read,
)
from scripts.six_max_adaptive_smoke import (  # noqa: E402
    NUM_SEATS, POOL_DEFAULTS, _BlueprintAsOpp, load_blueprint,
    make_opp_policy, run_match_collect,
)


def collect_tendency_preds(net, solver, opp_policy, *, n_matches: int,
                           hand_target: int, base_seed: int
                           ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run n_matches matches with `opp_policy` as opponent; for each hero
    decision, run the net and collect the K=10 tendency_pred vector. Returns
    (all_preds [N_decisions_total, K], per_match_preds: list of [T_i, K])."""
    net.eval()
    per_match = []
    for m in range(n_matches):
        hero_seat = m % NUM_SEATS
        designated_opp_seat = (hero_seat + 1) % NUM_SEATS
        result = run_match_collect(
            solver=solver, opp_policy=opp_policy, hero_seat=hero_seat,
            hand_target=hand_target, epsilon=0.0,
            designated_opp_seat=designated_opp_seat,
            seed=base_seed + m * 17, hero_encoder_rng_seed=m + 1000)
        if not result.tuples:
            continue
        with torch.no_grad():
            packed = collate(result.tuples, device="cpu")
            _, tendency_pred, _ = net(
                packed["tokens"], packed["pad_mask"],
                packed["query_idx"], packed["legal_mask"],
                packed["opp_stats"])
        per_match.append(tendency_pred.cpu().numpy())
    if not per_match:
        return np.zeros((0, K_TENDENCY), dtype=np.float32), []
    return np.concatenate(per_match, axis=0), per_match


def late_match_preds(per_match: list[np.ndarray], last_k: int) -> np.ndarray:
    """Take the last `last_k` decisions from each match and stack. This is
    the gate-relevant regime: match_conf ramps from 0 at n_actions<20 to 1
    at n>=300, so the gate fires mainly on late-match decisions where
    opp_stats has accumulated."""
    chunks = [m[-last_k:] for m in per_match if len(m) > 0]
    if not chunks:
        return np.zeros((0, K_TENDENCY), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phase1_d128_repro/smoke_net.pt")
    ap.add_argument("--anchor-dir",
                    default="runs/six_max_20260530_034023_phase4f_dcfr_candC_k200")
    ap.add_argument("--abstraction-pkl",
                    default="runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    ap.add_argument("--n-bp-matches", type=int, default=20,
                    help="Blueprint-vs-blueprint matches for the empirical anchor.")
    ap.add_argument("--n-probe-matches", type=int, default=16,
                    help="Probe matches per archetype opponent.")
    ap.add_argument("--hand-target", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--last-k", type=int, default=10,
                    help="Last-K decisions per match for the late-match "
                         "(gate-relevant) view; must satisfy K << hand_target.")
    ap.add_argument("--out", default="runs/phase1_d128_repro/g6_empirical_anchor.json")
    args = ap.parse_args()

    t_total = time.time()
    print(f"[load] solver from {args.anchor_dir}")
    solver = load_blueprint(Path(args.anchor_dir), Path(args.abstraction_pkl))

    print(f"[load] net from {args.ckpt}")
    d = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    cfg = d["config"]
    net = Adaptive6MaxNet(d_model=cfg["d_model"], num_layers=cfg["num_layers"],
                          nhead=cfg["nhead"], dim_ff=cfg["dim_ff"])
    net.load_state_dict(d["state_dict"])
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"        loaded: {n_params:,} params, config={cfg}")

    # 1) Empirical blueprint anchor — head output on blueprint-vs-blueprint
    print(f"\n=== Empirical anchor: {args.n_bp_matches} blueprint-vs-blueprint matches ===")
    t = time.time()
    bp_opp = _BlueprintAsOpp(solver)
    bp_preds, bp_per_match = collect_tendency_preds(
        net, solver, bp_opp, n_matches=args.n_bp_matches,
        hand_target=args.hand_target, base_seed=args.seed + 9001)
    tendency_blueprint_empirical = bp_preds.mean(axis=0).astype(np.float32)
    tendency_blueprint_empirical_std = bp_preds.std(axis=0)
    bp_preds_late = late_match_preds(bp_per_match, args.last_k)
    tendency_blueprint_empirical_late = (
        bp_preds_late.mean(axis=0).astype(np.float32)
        if len(bp_preds_late) else tendency_blueprint_empirical)
    print(f"        n_decisions_all  = {len(bp_preds)}  "
          f"n_decisions_lateK = {len(bp_preds_late)}  "
          f"wall = {time.time()-t:.1f}s")
    print(f"        empirical anchor (mean, ALL decisions):")
    print(f"          {[f'{v:.4f}' for v in tendency_blueprint_empirical.tolist()]}")
    print(f"        empirical anchor (mean, LAST-{args.last_k} per match):")
    print(f"          {[f'{v:.4f}' for v in tendency_blueprint_empirical_late.tolist()]}")
    print(f"        per-dim std across decisions (ALL):")
    print(f"          {[f'{v:.3f}' for v in tendency_blueprint_empirical_std.tolist()]}")

    # 2) Per-archetype probes
    opp_specs = {
        "gushansenmtt":   {"path": POOL_DEFAULTS["gushansenmtt"]["path"]},
        "killphilmtt":    {"path": POOL_DEFAULTS["killphilmtt"]["path"]},
        "loosenluckymtt": {"path": POOL_DEFAULTS["loosenluckymtt"]["path"]},
    }
    print(f"\n=== Archetype probes: {args.n_probe_matches} matches per opp ===")
    per_opp = {}                  # name -> all-decision preds [N, K]
    per_opp_late = {}             # name -> last-K preds [N_late, K]
    per_opp_match_means = {}      # name -> [n_matches, K] per-match-mean
    per_opp_match_means_late = {} # name -> [n_matches, K] per-match-mean(last-K)

    def _per_match_means(per_match_list):
        if not per_match_list:
            return np.zeros((0, K_TENDENCY), dtype=np.float32)
        return np.stack([m.mean(axis=0) for m in per_match_list if len(m) > 0])

    def _per_match_means_late(per_match_list, k):
        if not per_match_list:
            return np.zeros((0, K_TENDENCY), dtype=np.float32)
        return np.stack([m[-k:].mean(axis=0) for m in per_match_list
                         if len(m) > 0])

    blueprint_probe_preds, bp_probe_per_match = collect_tendency_preds(
        net, solver, bp_opp, n_matches=args.n_probe_matches,
        hand_target=args.hand_target, base_seed=args.seed + 77777)
    per_opp["blueprint"] = blueprint_probe_preds
    per_opp_late["blueprint"] = late_match_preds(bp_probe_per_match, args.last_k)
    per_opp_match_means["blueprint"] = _per_match_means(bp_probe_per_match)
    per_opp_match_means_late["blueprint"] = _per_match_means_late(
        bp_probe_per_match, args.last_k)

    for name, info in opp_specs.items():
        t = time.time()
        opp_pol = make_opp_policy(name, info, solver)
        preds, per_match = collect_tendency_preds(
            net, solver, opp_pol, n_matches=args.n_probe_matches,
            hand_target=args.hand_target,
            base_seed=args.seed + abs(hash(name)) % 100000)
        per_opp[name] = preds
        per_opp_late[name] = late_match_preds(per_match, args.last_k)
        per_opp_match_means[name] = _per_match_means(per_match)
        per_opp_match_means_late[name] = _per_match_means_late(
            per_match, args.last_k)
        print(f"  [{name:<18s}] n_decisions={len(preds)}  "
              f"n_late={len(per_opp_late[name])}  "
              f"n_match_means={len(per_opp_match_means[name])}  "
              f"wall={time.time()-t:.1f}s")

    # 3) Compute d_bar against EMPIRICAL anchor — two views
    def _d_bar_view(preds_dict, anchor, label):
        print(f"\n=== G6 with EMPIRICAL anchor ({label}): "
              f"d_bar = ||head(opp) - anchor||_2 ===")
        view = {}
        for name, preds in preds_dict.items():
            if len(preds) == 0:
                view[name] = {"n_decisions": 0, "mean_d_bar": float("nan"),
                              "median_d_bar": float("nan"),
                              "mean_tendency_pred": [float("nan")] * K_TENDENCY}
                continue
            d_bars = [tendency_l2_read(p, anchor) for p in preds]
            view[name] = {
                "n_decisions": int(len(d_bars)),
                "mean_d_bar": float(np.mean(d_bars)),
                "median_d_bar": float(np.median(d_bars)),
                "mean_tendency_pred": preds.mean(axis=0).tolist(),
            }
        bp_d_bar = view["blueprint"]["mean_d_bar"]
        print(f"  blueprint (probe, vs empirical anchor) mean_d_bar = {bp_d_bar:.4f}")
        print(f"  archetype       mean_d_bar     ratio_vs_blueprint")
        ratios = {}
        for name in ("gushansenmtt", "killphilmtt", "loosenluckymtt"):
            m = view[name]["mean_d_bar"]
            r = m / max(bp_d_bar, 1e-12)
            ratios[name] = r
            print(f"  {name:<18s} {m:.4f}         {r:.2f}×")
        return view, bp_d_bar, ratios

    results_all, bp_d_bar_all, ratios_all = _d_bar_view(
        per_opp, tendency_blueprint_empirical, "ALL decisions")
    results_late, bp_d_bar_late, ratios_late = _d_bar_view(
        per_opp_late, tendency_blueprint_empirical_late,
        f"LAST-{args.last_k} per match (gate-relevant regime)")

    # Per-match-mean view: anchor = mean across blueprint matches of
    # per-match-mean tendency_pred; d_bar per archetype-match = ||match_mean -
    # anchor||_2. This is the grain a match-level EMA gate would operate on
    # (one read per match, not per decision).
    bp_match_means_late_anchor = (
        per_opp_match_means_late["blueprint"].mean(axis=0).astype(np.float32))
    print(f"\n=== Per-match-mean (LAST-{args.last_k}) d_bar view "
          f"(match-EMA gate regime) ===")
    print(f"  anchor (mean of blueprint per-match-means, LAST-{args.last_k}):")
    print(f"    {[f'{v:.4f}' for v in bp_match_means_late_anchor.tolist()]}")

    results_match = {}
    for name in ("blueprint", "gushansenmtt", "killphilmtt", "loosenluckymtt"):
        mm = per_opp_match_means_late[name]
        if len(mm) == 0:
            results_match[name] = {"n_matches": 0,
                                    "mean_d_bar": float("nan"),
                                    "per_match_d_bars": [],
                                    "match_means_std": [float("nan")]*K_TENDENCY}
            continue
        per_match_d_bars = [
            float(np.linalg.norm(mm[i] - bp_match_means_late_anchor))
            for i in range(len(mm))]
        results_match[name] = {
            "n_matches": int(len(mm)),
            "mean_d_bar": float(np.mean(per_match_d_bars)),
            "median_d_bar": float(np.median(per_match_d_bars)),
            "std_d_bar": float(np.std(per_match_d_bars)),
            "per_match_d_bars": per_match_d_bars,
            "match_means_std":
                np.std(mm, axis=0).astype(float).tolist(),
            "match_means_mean":
                np.mean(mm, axis=0).astype(float).tolist(),
        }

    bp_d_bar_match = results_match["blueprint"]["mean_d_bar"]
    print(f"  blueprint    per-match mean_d_bar = {bp_d_bar_match:.4f}  "
          f"std = {results_match['blueprint']['std_d_bar']:.4f}")
    print(f"  archetype       mean_d_bar   std_d_bar   ratio_vs_blueprint")
    ratios_match = {}
    for name in ("gushansenmtt", "killphilmtt", "loosenluckymtt"):
        m = results_match[name]["mean_d_bar"]
        sd = results_match[name]["std_d_bar"]
        r = m / max(bp_d_bar_match, 1e-12)
        ratios_match[name] = r
        print(f"  {name:<18s} {m:.4f}      {sd:.4f}      {r:.2f}×")

    # Headline: late-K view (the gate operates after match_conf ramps in)
    results = results_late
    bp_d_bar = bp_d_bar_late
    archetype_ratios = ratios_late

    # 4) Pairwise L2 between mean prediction vectors (independent diagnostic)
    # Use the LATE-K vectors as the gate-relevant view.
    print(f"\n=== Pairwise L2 between per-opp MEAN tendency_pred vectors "
          f"(LAST-{args.last_k} per match) ===")
    names = ["blueprint", "gushansenmtt", "killphilmtt", "loosenluckymtt"]
    mean_vecs = {n: np.array(results_late[n]["mean_tendency_pred"])
                 for n in names}
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dist = float(np.linalg.norm(mean_vecs[a] - mean_vecs[b]))
            pairwise[f"{a}__{b}"] = dist
            print(f"  ||{a:<18s} - {b:<18s}||_2 = {dist:.4f}")

    # 5) Verdict
    G6_blueprint_low = bp_d_bar <= 0.05
    G6_separation_ok = all(r >= 2.0 for r in archetype_ratios.values())
    G6_match_separation_ok = all(r >= 2.0 for r in ratios_match.values())
    print(f"\n=== G6 EMPIRICAL-ANCHOR verdict ===")
    print(f"  PER-DECISION (gate fires on per-decision d_bar):")
    print(f"    blueprint_d_bar      = {bp_d_bar:.4f}  (≤ 0.05 expected) → "
          f"{'OK' if G6_blueprint_low else 'LOOSE'}")
    print(f"    separation ratios    = "
          f"{ {k: f'{v:.2f}x' for k, v in archetype_ratios.items()} }")
    print(f"    separation_ok (all archetype ratios ≥ 2.0) → "
          f"{'OK' if G6_separation_ok else 'WEAK'}")
    print(f"  PER-MATCH-MEAN (gate fires on match-EMA d_bar):")
    print(f"    blueprint_d_bar      = {bp_d_bar_match:.4f}")
    print(f"    separation ratios    = "
          f"{ {k: f'{v:.2f}x' for k, v in ratios_match.items()} }")
    print(f"    separation_ok (all archetype ratios ≥ 2.0) → "
          f"{'OK' if G6_match_separation_ok else 'WEAK'}")

    payload = {
        "ckpt": args.ckpt,
        "n_params": int(n_params),
        "config": cfg,
        "n_bp_anchor_matches": int(args.n_bp_matches),
        "n_probe_matches_per_opp": int(args.n_probe_matches),
        "hand_target": int(args.hand_target),
        "last_k": int(args.last_k),
        "seed": int(args.seed),
        "tendency_blueprint_empirical_all":
            tendency_blueprint_empirical.tolist(),
        "tendency_blueprint_empirical_late":
            tendency_blueprint_empirical_late.tolist(),
        "tendency_blueprint_empirical_per_dim_std_all":
            tendency_blueprint_empirical_std.tolist(),
        "n_bp_anchor_decisions_all": int(len(bp_preds)),
        "n_bp_anchor_decisions_late": int(len(bp_preds_late)),
        "per_opp_all_decisions": results_all,
        "per_opp_late_decisions": results_late,
        "per_opp_match_means_late": results_match,
        "bp_match_means_late_anchor": bp_match_means_late_anchor.tolist(),
        "archetype_ratios_all": ratios_all,
        "archetype_ratios_late": ratios_late,
        "archetype_ratios_match_late": ratios_match,
        "bp_d_bar_all": float(bp_d_bar_all),
        "bp_d_bar_late": float(bp_d_bar_late),
        "bp_d_bar_match_late": float(bp_d_bar_match),
        "pairwise_l2_mean_pred_late": pairwise,
        "G6_blueprint_low_late": bool(G6_blueprint_low),
        "G6_blueprint_low_threshold": 0.05,
        "G6_separation_ok_per_decision_late": bool(G6_separation_ok),
        "G6_separation_ok_per_match_late": bool(G6_match_separation_ok),
        "G6_separation_threshold": 2.0,
        "headline_view": f"LAST-{args.last_k} per match",
        "wall_seconds": float(time.time() - t_total),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {out_path}")
    print(f"[done] wall = {payload['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
