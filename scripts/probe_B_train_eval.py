"""PROBE B — context-gated exploit head (LEARNED, position-aware). Trainer + eval.

F proved the un-gated overlay is SAFE but too crude (position-marginalized washes
out C's 59%-BB-specific fold-more). B fixes that: a small head trained to predict
the per-spot field oracle from the FULL 236-d feature vector (position, depth,
to_call, hand bucket all included), applied ONLY at gated jam-wall spots.

Safety BY CONSTRUCTION (the F structure, unchanged):
  - gate = 1 only on (field-context AND jam-wall). vs the champion (self-anchor
    eval) the context is NOT field -> gate = 0 -> composed == champion EXACTLY
    -> self-anchor = 0 by construction (no need to tune).
  - off jam-wall -> champion. So leakage onto non-field / non-jam play = 0.
The only empirical question (gate-2a): does the learned head capture the
BB-specific fold-more -> ΔFIELD >= +0.05 with flat aggression (it adds no raises).

Distillation target = the battery oracle (call/fold). Supervised, CPU, minutes,
no pod. Composes with C2 (cost 0.004/spot) and C (headroom 0.087/spot).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.cfr6 import _build_view_6max, _INTERMEDIATE_RAISE_IDX, N_DISCRETE_ACTIONS
from scripts.depth_invariance_probe import query_policy, HERO_SEAT
from scripts.fold_vs_shove_battery import build_facing_shove_spot

torch.manual_seed(0)


class JamHead(nn.Module):
    """236-d features -> fold/call logits (the position-aware oracle)."""
    def __init__(self, in_dim=236, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def collect_training_data(champ, structure, battery):
    """(features, oracle_label) for every facing-shove battery spot.
    oracle_label: 1=call, 0=fold. Features = the deployed 236-d encoding."""
    X, y = [], []
    for i, s in enumerate(battery["spots"]):
        try:
            state, dealer, _bb, _sh = build_facing_shove_spot(
                structure, s["level"], s["hero_pos"], s["opener_pos"],
                tuple(s["hero_cards"]), s["depth_bb"])
        except (ValueError, RuntimeError):
            continue
        _pol, _m, _p, _d, feat = query_policy(champ.solver, state, HERO_SEAT,
                                              dealer, rng_seed=0)
        X.append(np.asarray(feat, dtype=np.float32))
        y.append(1 if s["oracle"] == "call" else 0)
        if (i + 1) % 4000 == 0:
            print(f"[B] collected {i+1}/{len(battery['spots'])}", flush=True)
    return np.stack(X), np.asarray(y, dtype=np.int64)


def train_head(X, y, epochs=60, lr=1e-3, bs=256):
    head = JamHead(in_dim=X.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    # class weights (oracle is ~91% fold) so 'call' isn't ignored.
    w = torch.tensor([1.0, max(1.0, (y == 0).sum() / max((y == 1).sum(), 1))],
                     dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for j in range(0, n, bs):
            idx = perm[j:j+bs]
            opt.zero_grad()
            loss = lossf(head(Xt[idx]), yt[idx])
            loss.backward(); opt.step(); tot += float(loss) * len(idx)
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                acc = (head(Xt).argmax(1) == yt).float().mean()
            print(f"[B] epoch {ep+1}: loss {tot/n:.4f} oracle-fit acc {float(acc):.3f}",
                  flush=True)
    return head


class BHeadPolicy:
    """Champion everywhere; at gated (field-context) jam-wall spots, play the
    learned head's fold/call. gate=0 (e.g. vs champion) -> champion exactly."""
    def __init__(self, name, champ, head, field_context: bool):
        self.name = name; self.champ = champ; self.head = head
        self.gate = 1.0 if field_context else 0.0
        self.n_jam = 0; self.n_override = 0

    def select_action(self, parsed, state, rng, mode="sample") -> int:
        if self.gate == 0.0:
            return int(self.champ.select_action(parsed, state, rng, mode=mode))
        view = _build_view_6max(state, parsed)
        d2c = discretize_legal_actions(list(state.legal_actions()), view)
        lm = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
        for da in d2c:
            if da is not None:
                lm[int(da)] = 1.0
        is_jam = (view.to_call > 0) and not any(lm[i] for i in _INTERMEDIATE_RAISE_IDX)
        if is_jam:
            self.n_jam += 1
            feat = self.champ.solver.encoder.encode_from_parsed(parsed, rng=rng)
            with torch.no_grad():
                logit = self.head(torch.from_numpy(np.asarray(feat, np.float32))[None])[0]
            want_call = bool(logit[1] > logit[0])
            fold_chip = d2c.get(int(DiscreteAction.FOLD))
            call_chip = d2c.get(int(DiscreteAction.CALL))
            if want_call and call_chip is not None:
                self.n_override += 1; return int(call_chip)
            if (not want_call) and fold_chip is not None:
                self.n_override += 1; return int(fold_chip)
        return int(self.champ.select_action(parsed, state, rng, mode=mode))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="evals/h4_field_battery/battery_v1.json")
    ap.add_argument("--champion",
                    default="runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt")
    ap.add_argument("--registry", default="configs/league/registry_h4_field.json")
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--out", default="evals/probe_B_20260613.json")
    ap.add_argument("--head-out", default="evals/probe_B_head_20260613.pt")
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

    print("[B] collecting distillation data (features + oracle)...", flush=True)
    X, y = collect_training_data(champ, structure, battery)
    print(f"[B] {len(X)} samples, oracle call-rate {float((y==1).mean()):.3f}", flush=True)
    head = train_head(X, y)
    torch.save(head.state_dict(), args.head_out)

    CHAMP_FIELD = 0.746
    B_field = BHeadPolicy("B_field", champ, head, field_context=True)
    print(f"[B] ΔFIELD eval ({args.games} games)...", flush=True)
    fr = field_mixture_row(B_field, args.registry, abstr, structure,
                           n_games=args.games, master_seed=2026, log=print)
    dfield = fr["net"] - CHAMP_FIELD

    # self-anchor: gate=0 vs champion -> composed == champion -> 0 by construction.
    B_self = BHeadPolicy("B_self", champ, head, field_context=False)
    print(f"[B] self-anchor eval ({args.games} games, gate=0 -> expect ~0)...", flush=True)
    sa = evaluate_profile(hero_policy=B_self, opp_policy=champ, structure=structure,
                          n_games=args.games, master_seed=2026, hands_per_level=5,
                          max_hands=200, mode="sample", log=lambda *a, **k: None)
    sa_net, sa_se = sa["hero_net_per_game"], sa["stderr"]
    sa_z = sa_net / sa_se if sa_se > 0 else float("nan")

    sa_pass = abs(sa_net) <= 0.10 and abs(sa_z) < 2.0
    field_pass = dfield >= 0.05
    result = {
        "n_games": args.games,
        "field_override_rate": B_field.n_override / max(B_field.n_jam, 1),
        "self_anchor": {"net": sa_net, "se": sa_se, "z": sa_z, "pass": sa_pass,
                        "note": "gate=0 vs champion -> composed==champion; ~0 by construction"},
        "delta_field": {"B_net": fr["net"], "champ_baseline": CHAMP_FIELD,
                        "delta": dfield, "pass": field_pass},
        "vs_F": "F (marginalized) got ΔFIELD ~+0.014; B is position-aware",
        "verdict": ("B PASSES gate-2a — position-aware head captures the BB fold-more"
                    if (sa_pass and field_pass) else
                    "B short of gate — " + ("self-anchor " if not sa_pass else "")
                    + ("ΔFIELD<+0.05" if not field_pass else "")),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[B] self-anchor: {sa_net:+.4f} ± {sa_se:.4f} (z={sa_z:+.1f})  "
          f"{'PASS' if sa_pass else 'FAIL'}")
    print(f"[B] ΔFIELD: {dfield:+.4f}  (B {fr['net']:.3f} - champ {CHAMP_FIELD})  "
          f"{'PASS' if field_pass else 'FAIL'} (gate >=+0.05)  [F was ~+0.014]")
    print(f"[B] VERDICT: {result['verdict']}")
    print(f"[B] -> {args.out}")


if __name__ == "__main__":
    main()
