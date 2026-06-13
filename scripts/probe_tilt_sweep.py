"""TILT SWEEP — the one SOUND free live-exploitability probe (no battery reference).

Eval champion DIRECTIONAL perturbations vs the live field. If any crude tilt beats
the champion's +0.746, a live edge exists in that direction. If none does, the
champion is locally robust vs this field on the tested directions (suggestive, not
proof — a subtle state-conditioned edge could still exist, which only A would find).

Read WITH the artifact lens: vs a WEAK PASSIVE pool, aggro tilts may "win" by
over-attacking pool passivity (the limp-gap artifact, NOT a real edge — it cost H4
its self-anchor). A tilt that beats the champion is a lead to investigate, not a
confirmed edge. Read-only, CPU, no pod.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.cfr6 import _build_view_6max, N_DISCRETE_ACTIONS


class TiltPolicy:
    def __init__(self, name, champ, direction, delta, rng_seed=0):
        self.name = name; self.champ = champ; self.direction = direction; self.delta = delta
        self.n_fire = 0
        import random as _r; self._rng = _r.Random(rng_seed)

    def select_action(self, parsed, state, rng, mode="sample") -> int:
        a = int(self.champ.select_action(parsed, state, rng, mode=mode))
        if self._rng.random() >= self.delta:
            return a
        view = _build_view_6max(state, parsed)
        d2c = discretize_legal_actions(list(state.legal_actions()), view)
        fold_c = d2c.get(int(DiscreteAction.FOLD))
        call_c = d2c.get(int(DiscreteAction.CALL))
        allin_c = d2c.get(int(DiscreteAction.ALLIN))
        bets = [d2c[int(b)] for b in (DiscreteAction.BET_33, DiscreteAction.BET_66,
                DiscreteAction.BET_100, DiscreteAction.BET_200) if d2c.get(int(b)) is not None]
        biggest = (allin_c if allin_c is not None else (max(bets) if bets else None))
        preflop = parsed.get("street_idx", 0) == 0
        d = self.direction
        if d == "aggro_more" and a == call_c and biggest is not None and biggest != a:
            self.n_fire += 1; return int(biggest)
        if d == "call_more" and a == fold_c and call_c is not None:
            self.n_fire += 1; return int(call_c)
        if d == "fold_more" and a == call_c and fold_c is not None:
            self.n_fire += 1; return int(fold_c)
        if d == "aggro_less" and (a == allin_c or a in bets) and call_c is not None:
            self.n_fire += 1; return int(call_c)
        if d == "tight_pre" and preflop and a == call_c and fold_c is not None:
            self.n_fire += 1; return int(fold_c)
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion",
                    default="runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt")
    ap.add_argument("--registry", default="configs/league/registry_h4_field.json")
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--out", default="evals/tilt_sweep_20260613.json")
    args = ap.parse_args()

    from src.nlhe.abstraction import Abstraction
    from scripts.eval_pool import CheckpointPolicy
    from scripts.throwaway_query_real_ante import _RealAnteStructure
    from scripts.field_battery import field_mixture_row

    structure = _RealAnteStructure(TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml"))
    abstr = Abstraction.load("runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    champ = CheckpointPolicy("champion", args.champion, abstr, structure)
    CHAMP = 0.746

    rows = []
    for d in ["aggro_more", "call_more", "fold_more", "aggro_less", "tight_pre"]:
        pol = TiltPolicy(f"tilt_{d}", champ, d, args.delta)
        fr = field_mixture_row(pol, args.registry, abstr, structure,
                               n_games=args.games, master_seed=2026, log=lambda *a, **k: None)
        delta = fr["net"] - CHAMP
        beats = delta > 0.05
        rows.append({"direction": d, "net": fr["net"], "delta_vs_champ": delta,
                     "fires": pol.n_fire, "beats_champ_by_0.05": beats})
        print(f"[tilt] {d:12s}: net {fr['net']:.3f}  Δvs champ {delta:+.4f}  "
              f"{'<<< BEATS' if beats else ''}", flush=True)

    any_beat = any(r["beats_champ_by_0.05"] for r in rows)
    out = {"champion_baseline": CHAMP, "delta": args.delta, "games": args.games,
           "tilts": rows,
           "verdict": ("a tilt BEATS champion -> live edge lead (check artifact lens)"
                       if any_beat else
                       "NO crude tilt beats champion -> locally robust vs this field "
                       "(suggestive of near-optimal on tested directions)")}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[tilt] VERDICT: {out['verdict']}")
    print(f"[tilt] -> {args.out}")


if __name__ == "__main__":
    main()
