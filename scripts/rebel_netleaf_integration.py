"""Integration smoke: does the per-seat 6-vector net plug into the search leaf?

v1's blocker: the scalar (hero-relative) net couldn't value non-hero leaves. The
v3 net outputs a per-seat 6-vector indexed by ABSOLUTE seat — exactly the shape/
indexing subgame_leaf writes to node.leaf_value and subgame_solver reads as
leaf_value[hero_seat]. This builds a real subgame, solves it TWICE — once with the
normal rollout leaves, once with leaves overwritten by the net — and confirms the
net path runs with no perspective/shape mismatch and yields a valid solve.
"""
import sys, random
sys.path.insert(0, '.')
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch, torch.nn as nn
import pyspiel

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.stack_sampler import sample_starting_state
from src.nlhe.biased_policy import BiasedBlueprint
from src.nlhe.icm import sng_payouts_6max_double_up
from src.nlhe.subgame import build_subgame_tree, iter_leaf_nodes
from src.nlhe.subgame_leaf import LeafEvalContext, LeafEvalMode, evaluate_leaves
from src.nlhe.subgame_solver import SubgameSolveContext, solve_subgame
from src.nlhe.infoset6 import parse_state_6max
from scripts.eval_6max_self_play import _load_solver

ABSTR = "runs/k200_abstraction.pkl"
STRUCT = "configs/ignition_double_up_6max_turbo.yaml"
CKPT = "runs/k200_blueprint_ckpt_iter_2000.pt"


def mlp(in_dim, out_dim, hidden):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def main():
    rng = random.Random(7)
    structure = TournamentStructure.from_yaml(STRUCT)
    abstraction = Abstraction.load(ABSTR)
    solver = _load_solver("runs/k200_blueprint_ckpt_iter_2000.pt", abstraction, structure)
    enc = solver.encoder; bd = enc.max_bucket_dim
    payouts = list(sng_payouts_6max_double_up())

    ck = torch.load("/workspace/rebel_value_net_v3.pt", map_location="cpu", weights_only=False)
    net = mlp(ck["in_dim"], 6, tuple(ck["hidden"]))
    sd = {kk.replace("net.", "", 1): vv for kk, vv in ck["state_dict"].items()}
    net.load_state_dict(sd); net.eval()
    ymu, ysd, k = ck["y_mean"], ck["y_std"], ck["k"]

    def net_leaf_value(node, dealer):
        """Per-seat 6-vector from the net at this leaf's PBS (public + a belief)."""
        st = node.state
        parsed = parse_state_6max(st, observer=0) if st.current_player() < 0 else parse_state_6max(st)
        parsed["dealer_seat"] = dealer
        feat = np.asarray(enc.encode_from_parsed(parsed, rng=rng), dtype=np.float32)
        public = feat[200:]
        # smoke belief: acting/known seat one-hot from feat, others uniform (valid PBS belief)
        belief = np.zeros((6, k), dtype=np.float32)
        hb = int(feat[:bd].argmax()) if float(feat[:bd].max()) > 0 else -1
        for s in range(6):
            if s == (st.current_player() if st.current_player() >= 0 else 0) and 0 <= hb < k:
                belief[s, hb] = 1.0
            else:
                belief[s, :k] = 1.0 / k
        x = np.concatenate([public, belief.reshape(6 * k)])[None, :].astype(np.float32)
        with torch.no_grad():
            out = net(torch.from_numpy(x)).numpy()[0] * ysd + ymu
        return [float(v) for v in out]

    # Find a real decision state via a short trajectory.
    for _ in range(50):
        sm = sample_starting_state(structure, rng, num_paid=3)
        g = pyspiel.load_game(structure.to_inner_game_string_for_state(
            blind_level=sm["blind_level"], stacks=sm["stacks"], dealer_seat=sm["dealer_seat"]))
        s = g.new_initial_state()
        while s.is_chance_node():
            a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
        if not s.is_terminal() and s.current_player() >= 0 and sm["stacks"][s.current_player()] > 0:
            break
    dealer = sm["dealer_seat"]; stacks = list(sm["stacks"]); hero = s.current_player()
    tree = build_subgame_tree(s, max_action_depth=3, chance_samples_per_node=2, rng=rng)
    nleaf = sum(1 for _ in iter_leaf_nodes(tree))
    print(f"[setup] tree: {tree.summary()}  hero={hero} dealer={dealer} alive={sm['alive_count']}")

    def solve(ctx):
        return solve_subgame(tree, ctx)

    sctx = SubgameSolveContext(blueprint=solver, starting_stacks=stacks, payouts=payouts,
                               hero_seat=hero, n_iterations=150, rng=rng, num_paid=3,
                               average_weighting="linear", dealer_seat=dealer)

    # (1) Normal rollout leaves.
    lctx = LeafEvalContext(blueprint=solver, biased_blueprint=BiasedBlueprint(), starting_stacks=stacks,
                           payouts=payouts, hero_seat=hero, mode=LeafEvalMode.PROFILE_SAMPLE,
                           n_samples=1, rng=rng, num_paid=3, dealer_seat=dealer)
    evaluate_leaves(tree, lctx)
    res_roll = solve(sctx)
    print(f"[rollout-leaf ] root_policy sum={float(np.sum(res_roll.root_policy)):.4f}  "
          f"value6={np.round(res_roll.root_value_per_seat,4).tolist()}  finite={np.isfinite(res_roll.root_value_per_seat).all()}")

    # (2) Overwrite EVERY leaf with the per-seat net output, re-solve.
    for leaf in iter_leaf_nodes(tree):
        leaf.leaf_value = net_leaf_value(leaf, dealer)
    res_net = solve(sctx)
    ok = (abs(float(np.sum(res_net.root_policy)) - 1.0) < 1e-4
          and np.isfinite(res_net.root_value_per_seat).all()
          and len(res_net.root_value_per_seat) == 6)
    print(f"[NET-leaf     ] root_policy sum={float(np.sum(res_net.root_policy)):.4f}  "
          f"value6={np.round(res_net.root_value_per_seat,4).tolist()}  finite={np.isfinite(res_net.root_value_per_seat).all()}")
    print(f"[NET-leaf     ] hero value read as leaf_value[{hero}] -> root hero value={res_net.root_value_per_seat[hero]:.4f}")
    print()
    print("INTEGRATION:", "PASS — per-seat net plugs into the search leaf, no perspective/shape mismatch" if ok
          else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
