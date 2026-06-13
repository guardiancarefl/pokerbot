"""PROBE C2 — reachability / self-anchor-cost map (FREE, read-only).

C found the field-EV headroom is DEFENSIVE: the champion over-calls the field's
shoves; the exploit is fold-MORE in the BB at 11-15bb. But folding more vs the
CHAMPION'S OWN (wider) shove range LOSES — that is the exact H4 trap. C2 measures
that price: for each jam-wall spot, the EV cost of playing the FIELD-oracle action
against the CHAMPION'S shove range vs the champion-optimal action.

  self_anchor_cost(spot) = max(ev_call_champ, ev_fold)
                           - (ev_call_champ if field_oracle=='call' else ev_fold)

Headline: total weighted self_anchor_cost vs the field headroom (0.0866/spot).
  cost << 0.087  -> fold-more is nearly free vs champion -> B/F safe, exploit worth
                    it; proceed confidently.
  cost ~ or > 0.087 -> ENTANGLED -> the context GATE does all the work (apply only
                    vs field); un-gated methods (H4) collapse. Net value = the
                    field gain captured ONLY when gated.

Reuses the field battery's equity/ICM math; swaps the opener shove range from the
pool to the champion. Read-only, CPU, no pod.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from src.nlhe.game_strings import TournamentStructure
from scripts.depth_invariance_probe import query_policy, HERO_SEAT, enumerate_169_hands
from scripts.fold_vs_shove_battery import _build_unopened_spot
from scripts.field_battery import _hero_equity, _spot_icm, STRUCTURE_YAML


def champion_jam_density(champ, structure, level, opener_pos, depth, hands):
    """Champion's soft open-jam range at this cell: {label: P(shove first-in)}.
    Shove mass = policy mass on discrete actions whose chip action is all-in."""
    density = {}
    for c1, c2, label, _kind in hands:
        try:
            state, dealer, _bb = _build_unopened_spot(
                structure, level, opener_pos, (c1, c2), depth)
        except (ValueError, RuntimeError):
            continue
        policy, mask, _p, d2c, _f = query_policy(champ.solver, state, HERO_SEAT,
                                                 dealer, rng_seed=0)
        legal_chip = [v for v in d2c.values() if v is not None]
        if not legal_chip:
            continue
        allin_chip = 0.9 * max(legal_chip)
        shove = sum(policy[i] for i, ch in d2c.items()
                    if ch is not None and ch >= allin_chip and ch > 1 and mask[i])
        if shove > 0.02:
            density[label] = float(shove)
    return density


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="evals/h4_field_battery/battery_v1.json")
    ap.add_argument("--champion",
                    default="runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt")
    ap.add_argument("--out", default="evals/c2_reachability_20260613.json")
    args = ap.parse_args()

    from src.nlhe.abstraction import Abstraction
    from scripts.eval_pool import CheckpointPolicy
    from scripts.throwaway_query_real_ante import _RealAnteStructure

    battery = json.loads(Path(args.battery).read_text())
    structure = _RealAnteStructure(TournamentStructure.from_yaml(battery["structure"]))
    abstr = Abstraction.load("runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    champ = CheckpointPolicy("champion", args.champion, abstr, structure)
    hands = enumerate_169_hands()

    # champion shove range per (opener_pos, depth, level) — cache across hero pos
    champ_range = {}
    cells = sorted({(s["opener_pos"], s["depth_bb"], s["level"]) for s in battery["spots"]})
    for i, (op, dp, lv) in enumerate(cells):
        champ_range[(op, dp, lv)] = champion_jam_density(champ, structure, lv, op, dp, hands)
        if (i + 1) % 5 == 0:
            print(f"[c2] champ range {i+1}/{len(cells)}", flush=True)

    eqc, rows = {}, []
    for s in battery["spots"]:
        rk = (s["opener_pos"], s["depth_bb"], s["level"])
        dens = champ_range[rk]
        ev_fold, icm_win, icm_lose = _spot_icm(
            structure, s["level"], s["hero_pos"], s["opener_pos"], s["depth_bb"])
        if not dens:
            ev_call_champ = ev_fold  # champion never shoves here -> calling ~ neutral
        else:
            ek = (s["hero_label"], rk)
            if ek not in eqc:
                eqc[ek] = _hero_equity(tuple(s["hero_cards"]), dens)
            ev_call_champ = eqc[ek] * icm_win + (1 - eqc[ek]) * icm_lose
        field_oracle = s["oracle"]
        ev_field_action = ev_call_champ if field_oracle == "call" else ev_fold
        champ_oracle = "call" if ev_call_champ > ev_fold else "fold"
        cost = max(ev_call_champ, ev_fold) - ev_field_action
        rows.append({"pos": s["hero_pos"], "depth": s["depth_bb"],
                     "field_oracle": field_oracle, "champ_oracle": champ_oracle,
                     "cost": cost, "disagree": field_oracle != champ_oracle})

    n = len(rows)
    total_cost = sum(r["cost"] for r in rows)
    mean_cost = total_cost / n
    disagree_rate = sum(r["disagree"] for r in rows) / n
    FIELD_HEADROOM = 0.0866

    def by(key):
        agg = defaultdict(lambda: [0.0, 0, 0])
        for r in rows:
            agg[r[key]][0] += r["cost"]; agg[r[key]][1] += 1; agg[r[key]][2] += r["disagree"]
        return {str(k): {"mean_cost": v[0]/v[1], "disagree_rate": v[2]/v[1], "n": v[1]}
                for k, v in sorted(agg.items(), key=lambda kv: -kv[1][0])}

    verdict = ("REACHABLE/CHEAP — fold-more is nearly free vs champion; B/F safe"
               if mean_cost < 0.25 * FIELD_HEADROOM else
               "ENTANGLED — fold-more costs vs champion; the GATE is load-bearing "
               "(net value captured only when gated; un-gated = H4 collapse)")
    result = {
        "champion": args.champion, "n_spots": n,
        "field_headroom_per_spot": FIELD_HEADROOM,
        "self_anchor_cost_mean": mean_cost,
        "cost_as_frac_of_headroom": mean_cost / FIELD_HEADROOM,
        "oracle_disagree_rate": disagree_rate,
        "by_position": by("pos"), "by_depth": by("depth"),
        "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[c2] field headroom/spot   = +{FIELD_HEADROOM:.4f}  (the gain vs FIELD)")
    print(f"[c2] self-anchor cost/spot = {mean_cost:+.4f}  "
          f"({mean_cost/FIELD_HEADROOM:.0%} of headroom)  (the price vs CHAMPION)")
    print(f"[c2] oracle disagreement (field vs champ) = {disagree_rate:.0%} of spots")
    print(f"[c2] VERDICT: {verdict}")
    print(f"[c2] -> {args.out}")


if __name__ == "__main__":
    main()
