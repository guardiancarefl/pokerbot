"""PROBE F — un-gated oracle overlay (NO training). Apply the field best-response
call/fold (already in battery_v1.json) at jam-wall spots; champion everywhere else.

C2 said the exploit is barely entangled (fold-more costs only 0.004/spot vs the
champion). F tests whether the SIMPLEST possible capture — a zero-training lookup
overlay applied UN-GATED (vs all opponents) — clears the canonical gates:
  - self-anchor within ±0.10 / |z|<2  (does the 0.004/spot cost compound past the band?)
  - real ΔFIELD (captures the +0.087/spot headroom in /game terms)
  - flat aggression (F only changes fold/call -> Δaggr ~ 0 by construction)

Honest approximation: the overlay is POSITION-MARGINALIZED — keyed by
(hand_label, depth_bucket) aggregated over the battery's positions/openers/levels
(live SNG states don't carry the battery's exact position keys robustly). It
captures the dominant fold-more-at-short-stacks signal; a per-position version
(or B's learned head) is sharper. Read-only/eval, CPU, no pod.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.cfr6 import _build_view_6max, _INTERMEDIATE_RAISE_IDX, N_DISCRETE_ACTIONS
from src.nlhe.infoset6 import parse_state_6max

_RANK = "23456789TJQKA"
DEPTH_BUCKETS = [5, 8, 11, 15]


def card_label(two: str) -> str:
    """'AsKd' -> 'AKo'; 'AsAh' -> 'AA'; 'As9s' -> 'A9s'."""
    r1, s1, r2, s2 = two[0], two[1], two[2], two[3]
    if r1 == r2:
        return r1 + r2
    hi, lo = (r1, r2) if _RANK.index(r1) > _RANK.index(r2) else (r2, r1)
    return hi + lo + ("s" if s1 == s2 else "o")


def build_overlay_table(battery):
    """(label, depth_bucket) -> 'call'|'fold', EV-marginalized over positions/
    openers/levels (best fixed action vs the field at that hand+depth)."""
    agg = defaultdict(lambda: [0.0, 0.0])  # [sum ev_call, sum ev_fold]
    for s in battery["spots"]:
        k = (s["hero_label"], s["depth_bb"])
        agg[k][0] += s["ev_call"]; agg[k][1] += s["ev_fold"]
    return {k: ("call" if c >= f else "fold") for k, (c, f) in agg.items()}


def nearest_depth(bb_depth):
    return min(DEPTH_BUCKETS, key=lambda d: abs(d - bb_depth))


class FOverlayPolicy:
    """Champion everywhere; at jam-wall (facing-shove) spots, override fold/call
    with the marginalized field oracle."""

    def __init__(self, name, champ, table):
        self.name = name
        self.champ = champ
        self.table = table
        self.n_override = 0
        self.n_jam = 0

    def select_action(self, parsed, state, rng, mode="sample") -> int:
        view = _build_view_6max(state, parsed)
        legal_chip = list(state.legal_actions())
        d2c = discretize_legal_actions(legal_chip, view)
        lm = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
        for da in d2c:
            if da is not None:
                lm[int(da)] = 1.0
        is_jam = (view.to_call > 0) and not any(lm[i] for i in _INTERMEDIATE_RAISE_IDX)
        if is_jam:
            self.n_jam += 1
            cp = parsed["current_player"]
            label = card_label(parsed["private_cards"])
            depth = nearest_depth(
                (parsed["money"][cp] + parsed["contribution"][cp]) / parsed["big_blind"])
            oracle = self.table.get((label, depth))
            fold_chip = d2c.get(int(DiscreteAction.FOLD))
            call_chip = d2c.get(int(DiscreteAction.CALL))
            if oracle == "fold" and fold_chip is not None:
                self.n_override += 1
                return int(fold_chip)
            if oracle == "call" and call_chip is not None:
                self.n_override += 1
                return int(call_chip)
        return int(self.champ.select_action(parsed, state, rng, mode=mode))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="evals/h4_field_battery/battery_v1.json")
    ap.add_argument("--champion",
                    default="runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt")
    ap.add_argument("--registry", default="configs/league/registry_h4_field.json")
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--out", default="evals/probe_F_20260613.json")
    args = ap.parse_args()

    from src.nlhe.abstraction import Abstraction
    from scripts.eval_pool import CheckpointPolicy
    from scripts.throwaway_query_real_ante import _RealAnteStructure
    from scripts.field_battery import field_mixture_row
    from scripts.sng_baseline import evaluate_profile

    battery = json.loads(Path(args.battery).read_text())
    structure = _RealAnteStructure(TournamentStructure.from_yaml(battery["structure"]))
    abstr = Abstraction.load("runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    champ = CheckpointPolicy("champion", args.champion, abstr, structure)
    table = build_overlay_table(battery)
    F = FOverlayPolicy("F_overlay", champ, table)

    CHAMP_FIELD = 0.746  # champion_baselines.json F_B2 baseline
    print(f"[F] overlay table: {len(table)} (label,depth) keys", flush=True)

    # GATE: ΔFIELD — F vs the weighted pool
    print(f"[F] field-mixture row ({args.games} games)...", flush=True)
    fr = field_mixture_row(F, args.registry, abstr, structure,
                           n_games=args.games, master_seed=2026, log=print)
    dfield = fr["net"] - CHAMP_FIELD

    # GATE: self-anchor — F vs champion
    print(f"[F] self-anchor (F vs champion, {args.games} games)...", flush=True)
    sa = evaluate_profile(hero_policy=F, opp_policy=champ, structure=structure,
                          n_games=args.games, master_seed=2026, hands_per_level=5,
                          max_hands=200, mode="sample", log=lambda *a, **k: None)
    sa_net, sa_se = sa["hero_net_per_game"], sa["stderr"]
    sa_z = sa_net / sa_se if sa_se > 0 else float("nan")

    sa_pass = abs(sa_net) <= 0.10 and abs(sa_z) < 2.0
    field_pass = dfield >= 0.05
    result = {
        "n_games": args.games, "override_rate": F.n_override / max(F.n_jam, 1),
        "n_jam_spots": F.n_jam, "n_overrides": F.n_override,
        "self_anchor": {"net": sa_net, "se": sa_se, "z": sa_z, "pass": sa_pass},
        "delta_field": {"F_net": fr["net"], "champ_baseline": CHAMP_FIELD,
                        "delta": dfield, "pass": field_pass},
        "aggression": "flat by construction (F only changes fold/call, adds no raises)",
        "verdict": ("F PASSES — zero-training overlay captures the edge within gates"
                    if (sa_pass and field_pass) else
                    "F FAILS a gate — " + ("self-anchor " if not sa_pass else "")
                    + ("ΔFIELD" if not field_pass else "") + " (case for B's gated head)"),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[F] self-anchor: {sa_net:+.4f} ± {sa_se:.4f} (z={sa_z:+.1f})  "
          f"{'PASS' if sa_pass else 'FAIL'} (gate |net|<=0.10, |z|<2)")
    print(f"[F] ΔFIELD: {dfield:+.4f}  (F {fr['net']:.3f} - champ {CHAMP_FIELD})  "
          f"{'PASS' if field_pass else 'FAIL'} (gate >=+0.05)")
    print(f"[F] override rate at jam spots: {result['override_rate']:.0%} "
          f"({F.n_override}/{F.n_jam})")
    print(f"[F] VERDICT: {result['verdict']}")
    print(f"[F] -> {args.out}")


if __name__ == "__main__":
    main()
