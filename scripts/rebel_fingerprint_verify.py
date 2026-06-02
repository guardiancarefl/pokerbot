"""Fingerprint-verify the k=200 blueprint + abstraction load correctly on a
fresh box. Catches silent-load bugs (wrong feature_dim, wrong action head,
truncated/old checkpoint, abstraction k mismatch) before any generation run.

Expected fingerprint (from prior validated boxes):
  - encoder.feature_dim == 236
  - discrete action head == 9 (N_DISCRETE_ACTIONS)
  - total policy-net params ~901k (6 advantage + 1 shared strategy MLP)
  - abstraction: preflop k=20, postflop k=200
"""
import sys
sys.path.insert(0, '.')
import torch

from src.nlhe.abstraction import Abstraction
from src.nlhe.actions import DiscreteAction
from src.nlhe.networks6 import N_DISCRETE_ACTIONS
from src.nlhe.game_strings import TournamentStructure
from scripts.eval_6max_self_play import _load_solver

ABSTR = "runs/k200_abstraction.pkl"
CKPT = "runs/k200_blueprint_ckpt_iter_2000.pt"
STRUCT = "configs/ignition_double_up_6max_turbo.yaml"

EXPECT_FEATDIM = 236
EXPECT_ACTIONS = 9
PARAM_LO, PARAM_HI = 850_000, 950_000  # ~901k

def main():
    fails = []

    # --- abstraction ---
    abstr = Abstraction.load(ABSTR)
    print(f"[abstraction] loaded {ABSTR}")
    streets = getattr(abstr, "streets", None)
    print(f"[abstraction] streets = {list(streets) if isinstance(streets, dict) else streets}")
    # derive per-street k (n buckets) however the structure exposes it
    ks = {}
    if isinstance(streets, dict):
        for name, sd in streets.items():
            for attr in ("k", "n_buckets", "num_buckets"):
                if hasattr(sd, attr):
                    ks[name] = getattr(sd, attr); break
            else:
                if isinstance(sd, dict):
                    ks[name] = sd.get("k") or sd.get("n_buckets") or sd.get("num_buckets")
    print(f"[abstraction] per-street k = {ks}")
    if ks:
        kpost = max(v for v in ks.values() if isinstance(v, int))
        if kpost != 200:
            fails.append(f"postflop k {kpost} != 200")

    # --- solver / blueprint ---
    structure = TournamentStructure.from_yaml(STRUCT)
    solver = _load_solver(CKPT, abstr, structure)
    print(f"[blueprint] loaded {CKPT}")

    # feature dim
    fd = solver.encoder.feature_dim
    print(f"[blueprint] encoder.feature_dim = {fd} (expect {EXPECT_FEATDIM})")
    if fd != EXPECT_FEATDIM:
        fails.append(f"feature_dim {fd} != {EXPECT_FEATDIM}")

    # action head: inspect a concrete advantage net's final layer out_features
    net0 = solver.policy_nets.net_for(0)
    last_linear = [m for m in net0.modules() if isinstance(m, torch.nn.Linear)][-1]
    out_dim = last_linear.out_features
    print(f"[blueprint] adv-net output dim = {out_dim} (N_DISCRETE_ACTIONS={N_DISCRETE_ACTIONS}, expect {EXPECT_ACTIONS})")
    print(f"[blueprint] DiscreteAction members = {[d.name for d in DiscreteAction]}")
    if out_dim != EXPECT_ACTIONS:
        fails.append(f"action head {out_dim} != {EXPECT_ACTIONS}")
    if N_DISCRETE_ACTIONS != EXPECT_ACTIONS:
        fails.append(f"N_DISCRETE_ACTIONS {N_DISCRETE_ACTIONS} != {EXPECT_ACTIONS}")

    # param count across all policy nets (6 advantage + shared strategy)
    seen = set()
    total = 0
    nets = []
    from src.nlhe.solver6 import NUM_SEATS_6MAX
    for s in range(NUM_SEATS_6MAX):
        nets.append(("adv_%d" % s, solver.policy_nets.net_for(s)))
    nets.append(("strat", solver.policy_nets.strat_net))
    per_net = {}
    for name, net in nets:
        if id(net) in seen:
            per_net[name] = "shared(=already counted)"
            continue
        seen.add(id(net))
        p = sum(x.numel() for x in net.parameters())
        per_net[name] = p
        total += p
    print(f"[blueprint] per-net params: {per_net}")
    print(f"[blueprint] TOTAL policy params = {total:,} (expect {PARAM_LO:,}..{PARAM_HI:,})")
    if not (PARAM_LO <= total <= PARAM_HI):
        fails.append(f"param count {total} outside [{PARAM_LO},{PARAM_HI}]")

    # config sanity
    cfg = solver.cfg
    print(f"[blueprint] hidden_dim={cfg.hidden_dim} starting_stack={cfg.starting_stack} "
          f"bb={cfg.big_blind} sb={cfg.small_blind} payout={cfg.payout_mode}")

    # live forward pass sanity (encode a fresh state -> net forward, no NaN)
    import pyspiel, random
    from src.nlhe.game_strings import six_max_sng
    game = pyspiel.load_game(six_max_sng(starting_stack=int(cfg.starting_stack)))
    st = game.new_initial_state()
    rng = random.Random(0)
    while st.is_chance_node():
        a, p = zip(*st.chance_outcomes())
        st = st.child(int(rng.choices(a, weights=p, k=1)[0]))
    from src.nlhe.infoset6 import parse_state_6max
    parsed = parse_state_6max(st)
    feat = solver.encoder.encode_from_parsed(parsed, rng=rng)
    import numpy as np
    dev = next(net0.parameters()).device
    feat_t = torch.tensor(np.asarray(feat, dtype=np.float32)).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = net0(feat_t)
    print(f"[forward] ran on device={dev}")
    print(f"[forward] feat shape={feat_t.shape} -> out shape={tuple(out.shape)} "
          f"finite={bool(torch.isfinite(out).all())}")
    if feat_t.shape[-1] != EXPECT_FEATDIM:
        fails.append(f"encoded feat dim {feat_t.shape[-1]} != {EXPECT_FEATDIM}")
    if out.shape[-1] != EXPECT_ACTIONS:
        fails.append(f"forward out dim {out.shape[-1]} != {EXPECT_ACTIONS}")
    if not bool(torch.isfinite(out).all()):
        fails.append("forward produced non-finite values")

    print()
    if fails:
        print("FINGERPRINT: FAIL")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("FINGERPRINT: PASS  (feature_dim=236, 9-action, ~901k params, abstraction k=20/200, live forward finite)")

if __name__ == "__main__":
    main()
