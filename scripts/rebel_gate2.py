"""GATE 2 (directional): does ReBeL (search + value net) beat bare k=200?

Hero A = bare k=200 blueprint.   Hero B = ReBeL: depth-limited search + the trained
v3 value net at the leaves (per-seat 6-vector) + k=200 as the policy prior (the
solver warm-start). Paired, common-random-numbers by seed: identical sampled
tournament state + hole-card deal + opponent rng, only the hero policy differs.

Metric = hero ICM-equity-delta per hand (the Double-Up cash-equity proxy; the unit
the pilot used). Opponent panel built-in (Shanky tight bots load the same way once
data/shanky_profiles is present): NIT, TAG, LAG (tight->loose), a MIXED table, and
LOOSE (STATION). The decisive row is the TIGHT bots (NIT/TAG) — the over-folding
leak ReBeL targets.

CAVEATS (directional, not the verdict): rough M=1 net (R²=0.39); net-leaf uses a
uniform-opponent belief prior at leaves (no card peeking — hero row is hero's known
bucket, opponents are the max-entropy range); per-hand ICM, not full-SNG-to-bubble.
"""
import sys, os, random, argparse, statistics
sys.path.insert(0, '.')
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch, torch.nn as nn
import pyspiel

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.stack_sampler import sample_starting_state
from src.nlhe.icm import sng_payouts_6max_double_up
from src.nlhe.icm_returns import icm_adjust_returns
from src.nlhe.equity import cards_from_str
from src.nlhe.subgame import build_subgame_tree, iter_leaf_nodes
from src.nlhe.subgame_solver import SubgameSolveContext, solve_subgame, extract_action
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.actions import discretize_legal_actions
from src.nlhe.cfr6 import _build_view_6max
from src.nlhe.networks6 import SCHEMA_VERSION as _PN_SCHEMA
from src.nlhe.archetype6 import ArchetypePolicy
from src.nlhe.archetypes import NAMED_ARCHETYPES, EquityCalibration
from src.nlhe.scripted_bots import ShankyProfilePolicy
from scripts.eval_6max_self_play import _load_solver, _sample_action_from_policy

SHANKY_DIR = "data/shanky_profiles"

ABSTR = "runs/k200_abstraction.pkl"
STRUCT = "configs/ignition_double_up_6max_turbo.yaml"
CKPT = "runs/k200_blueprint_ckpt_iter_2000.pt"
CALIB = "runs/archetype_design/bucket_equity_analysis_6max.json"
NET = "/workspace/rebel_value_net_full.pt"
NUM_SEATS = 6
_NAME = {p.name.name: p for p in NAMED_ARCHETYPES}


def mlp(in_dim, hidden):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers += [nn.Linear(d, 6)]
    return nn.Sequential(*layers)


class BlueprintHero:
    name = "k200"
    def __init__(self, solver):
        self.solver = solver; self.dealer = None
    def new_hand(self, dealer): self.dealer = dealer
    def select_action(self, parsed, state, rng, mode="sample"):
        parsed = dict(parsed); parsed["dealer_seat"] = self.dealer
        return _sample_action_from_policy(self.solver, parsed, state, rng, mode=mode)


class ReBeLHero:
    name = "rebel"
    def __init__(self, solver, abstraction, net, ck, payouts, depth=3, n_iters=150, num_paid=3,
                 weighting="linear", warm_start=True, belief="uniform"):
        self.solver = solver; self.abs = abstraction; self.net = net
        self.ymu, self.ysd, self.k = ck["y_mean"], ck["y_std"], ck["k"]
        self.bd = solver.encoder.max_bucket_dim; self.enc = solver.encoder
        self.payouts = payouts; self.depth = depth; self.n_iters = n_iters; self.num_paid = num_paid
        self.weighting = weighting; self.warm_start = warm_start; self.belief = belief
        self.pn = solver.policy_nets
        self.is_v2 = (getattr(self.pn, "loaded_schema_version", None) == _PN_SCHEMA)
        self.dealer = None; self.kstreet = {0: 20, 1: 200, 2: 200, 3: 200}
        self._opp_belief = None  # precomputed per-decision opponent reach-belief (reach mode)
    def new_hand(self, dealer): self.dealer = dealer

    def _starting(self, parsed):
        m, c = parsed["money"], parsed["contribution"]
        return [int(m[i]) + int(c[i]) for i in range(NUM_SEATS)]

    def _batched_policy(self, seat, public, mask, kk):
        """Blueprint policy over all kk bucket-variants at one infoset (for reach)."""
        feats = np.zeros((kk, self.enc.feature_dim), dtype=np.float32)
        feats[:, self.bd:] = public
        feats[np.arange(kk), np.arange(kk)] = 1.0
        x = torch.from_numpy(feats)
        with torch.no_grad():
            net = self.pn.strat_net if self.is_v2 else self.pn.nets[seat]
            net.eval(); out = net(x).numpy()
        m = mask[None, :]
        ex = (np.exp(out - out.max(1, keepdims=True)) if self.is_v2 else np.maximum(out, 0.0)) * m
        den = ex.sum(1, keepdims=True)
        return np.where(den > 0, ex / np.maximum(den, 1e-12), m / max(float(mask.sum()), 1.0))

    def _root_belief(self, state, hero, starting):
        """Per-opponent reach-POSTERIOR over root-street buckets from the public
        betting history (the resolver's precise-belief mode). Held across the
        depth-limited subgame's leaves (opponent ranges ~constant within it)."""
        try:
            init = state.get_game().new_initial_state()
            recs = []; s = init
            for act in state.history():
                if s.is_chance_node():
                    s.apply_action(act); continue
                cp = s.current_player()
                p = parse_state_6max(s); p["dealer_seat"] = self.dealer
                feat = np.asarray(self.enc.encode_from_parsed(p, rng=None), dtype=np.float32)
                view = _build_view_6max(s, p)
                d2c = discretize_legal_actions(list(s.legal_actions()), view)
                da = next((int(d) for d, c in d2c.items() if c == act), None)
                if da is not None:
                    mask = np.zeros(9, dtype=np.float32)
                    for dd in d2c:
                        mask[int(dd)] = 1.0
                    recs.append({"seat": cp, "public": feat[self.bd:].copy(), "mask": mask,
                                 "action": da, "street": int(p["street_idx"])})
                s.apply_action(act)
            root_street = int(parse_state_6max(state)["street_idx"])
            kk = self.kstreet.get(root_street, self.k)
            folded = {r["seat"] for r in recs if r["action"] == 0}
            bel = np.zeros((6, self.k), dtype=np.float32)
            for sd in range(6):
                if sd == hero or starting[sd] <= 0 or sd in folded:
                    continue
                past = [r for r in recs if r["seat"] == sd and r["street"] == root_street]
                if not past:
                    bel[sd, :kk] = 1.0 / kk; continue
                logr = np.zeros(kk)
                for r in past:
                    pol = self._batched_policy(sd, r["public"], r["mask"], kk)
                    logr += np.log(pol[:, r["action"]] + 1e-9)
                logr -= logr.max(); rr = np.exp(logr); rr /= max(rr.sum(), 1e-12)
                bel[sd, :kk] = rr
            return bel
        except Exception:
            return None

    def _leaf_val(self, leaf, hero, hero_cards, hbucket_cache):
        st = leaf.state
        p = parse_state_6max(st, observer=0) if st.current_player() < 0 else parse_state_6max(st)
        p["dealer_seat"] = self.dealer
        # We only use the PUBLIC block (feat[200:]); blank the private cards so the
        # encoder skips its expensive per-leaf bucket Monte-Carlo (the bucket one-hot
        # is discarded anyway — opponent buckets would be perfect info regardless).
        p_pub = dict(p); p_pub["private_cards"] = ""
        feat = np.asarray(self.enc.encode_from_parsed(p_pub, rng=None), dtype=np.float32)
        public = feat[200:]
        bkey = p.get("public_cards", "") or ""
        if bkey in hbucket_cache:
            hb = hbucket_cache[bkey]
        else:
            board = cards_from_str(bkey)
            hb = int(self.abs.bucket_of(hero_cards, board, runouts=20, rng=random.Random(0))) \
                if len(hero_cards) == 2 else -1
            hbucket_cache[bkey] = hb
        kk = self.kstreet.get(int(p["street_idx"]), self.k)
        belief = np.zeros((6, self.k), dtype=np.float32)
        if 0 <= hb < self.k:
            belief[hero, hb] = 1.0
        starting = self._starting(p)
        if self.belief == "reach" and self._opp_belief is not None:
            # precise: opponents = the root reach-posterior (held across leaves)
            for s in range(6):
                if s != hero and starting[s] > 0:
                    belief[s] = self._opp_belief[s]
        else:
            for s in range(6):
                if s != hero and starting[s] > 0:
                    belief[s, :kk] = 1.0 / kk
        x = np.concatenate([public, belief.reshape(6 * self.k)])[None, :].astype(np.float32)
        with torch.no_grad():
            out = self.net(torch.from_numpy(x)).numpy()[0] * self.ysd + self.ymu
        return [float(v) for v in out]

    def select_action(self, parsed, state, rng, mode="sample"):
        cp = parsed["current_player"]
        starting = self._starting(parsed)
        try:
            hero_cards = cards_from_str(parse_state_6max(state, observer=cp).get("private_cards", "") or "")
            self._opp_belief = self._root_belief(state, cp, starting) if self.belief == "reach" else None
            tree = build_subgame_tree(state, max_action_depth=self.depth,
                                      chance_samples_per_node=2, rng=rng)
            cache = {}
            for leaf in iter_leaf_nodes(tree):
                leaf.leaf_value = self._leaf_val(leaf, cp, hero_cards, cache)
            res = solve_subgame(tree, SubgameSolveContext(
                blueprint=self.solver, starting_stacks=starting, payouts=self.payouts,
                hero_seat=cp, n_iterations=self.n_iters, rng=rng, num_paid=self.num_paid,
                average_weighting=self.weighting, warm_start=self.warm_start, dealer_seat=self.dealer))
            if res.degraded:
                raise RuntimeError("degraded")
            return extract_action(res, state, rng, mode)
        except Exception:
            # Blueprint fall-through (the resolver's safety net).
            p = dict(parsed); p["dealer_seat"] = self.dealer
            return _sample_action_from_policy(self.solver, p, state, rng, mode=mode)


def build_opponent_factory(spec, abstraction, calib, structure):
    """spec = ('archetype'|'mixed'|'shanky', key). Returns level -> [6 policies].

    Archetypes are level-independent. Shanky bots read stacks-in-BB via a fixed
    big_blind_chips, so we pre-build one instance per level with that level's
    INFLATED big blind (which is the actual bb of our ante-folded game string)."""
    kind, key = spec
    def ap(prof):
        return ArchetypePolicy(profile=prof, abstraction=abstraction, calibration=calib, bucket_runouts=20)
    if kind == "archetype":
        pols = [ap(_NAME[key]) for _ in range(6)]
        return lambda lvl: pols
    if kind == "mixed":
        order = ["NIT", "TAG", "LAG", "STATION", "MANIAC", "TAG"]
        pols = [ap(_NAME[order[i]]) for i in range(6)]
        return lambda lvl: pols
    if kind == "shanky":
        path = os.path.join(SHANKY_DIR, key)
        by_level = {}
        for bl in structure.blind_schedule:
            bb = bl.inflated_big_blind(6)
            by_level[bl.level] = ShankyProfilePolicy(name=key.replace(".txt", ""),
                                                     profile_path=path, big_blind_chips=bb)
        return lambda lvl: [by_level[lvl]] * 6
    raise ValueError(spec)


def play_hand(gs, sm, hero_seat, hero_policy, opp_policies, dealer, seed):
    """Play one hand: hero_policy at hero_seat, opp_policies elsewhere. Dealer-aware
    parse for all (positions correct). Returns hero's ICM-equity-delta."""
    rng = random.Random(seed)
    game = pyspiel.load_game(gs); state = game.new_initial_state()
    starting = list(sm["stacks"]); payouts = [2.0] * 3
    for _ in range(500):
        if state.is_terminal():
            break
        if state.is_chance_node():
            o = state.chance_outcomes()
            state.apply_action(int(rng.choices([x[0] for x in o], weights=[x[1] for x in o], k=1)[0]))
            continue
        parsed = parse_state_6max(state); parsed["dealer_seat"] = dealer
        cp = parsed["current_player"]
        pol = hero_policy if cp == hero_seat else opp_policies[cp]
        state.apply_action(int(pol.select_action(parsed, state, rng, mode="sample")))
    if not state.is_terminal():
        return None
    return icm_adjust_returns(chip_returns=state.returns(), starting_stacks=starting, payouts=payouts)[hero_seat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--net", default=NET)
    ap.add_argument("--net-a", default="k200", help="'k200' or a value-net .pt (hero A)")
    ap.add_argument("--net-b", default=None, help="'k200' or a value-net .pt (hero B); default=--net")
    ap.add_argument("--label", default="")
    ap.add_argument("--depth-a", type=int, default=3)
    ap.add_argument("--depth-b", type=int, default=3)
    ap.add_argument("--kiters-a", type=int, default=150)
    ap.add_argument("--kiters-b", type=int, default=150)
    ap.add_argument("--weighting-a", default="linear")
    ap.add_argument("--weighting-b", default="linear")
    ap.add_argument("--warmstart-a", type=int, default=1)
    ap.add_argument("--warmstart-b", type=int, default=1)
    ap.add_argument("--belief-a", default="uniform")
    ap.add_argument("--belief-b", default="uniform")
    a = ap.parse_args()
    if a.net_b is None:
        a.net_b = a.net
    # (label, kind, key) — tight bots first (the decisive over-folding rows).
    PANEL = [
        ("KillPhilMTT", ("shanky", "KillPhilMTT.txt")),
        ("timidtom",    ("shanky", "timidtom.txt")),
        ("NIT",         ("archetype", "NIT")),
        ("TAG",         ("archetype", "TAG")),
        ("LAG",         ("archetype", "LAG")),
        ("mixed",       ("mixed", None)),
        ("STATION",     ("archetype", "STATION")),
    ]

    structure = TournamentStructure.from_yaml(STRUCT)
    abstraction = Abstraction.load(ABSTR)
    solver = _load_solver(CKPT, abstraction, structure)
    calib = EquityCalibration.load(CALIB)
    payouts = list(sng_payouts_6max_double_up())
    def make_hero(spec, depth=3, kiters=150, weighting="linear", warm=1, belief="uniform"):
        if spec == "k200":
            return BlueprintHero(solver), "k200"
        cck = torch.load(spec, map_location="cpu", weights_only=False)
        cnet = mlp(cck["in_dim"], tuple(cck["hidden"]))
        cnet.load_state_dict({kk.replace("net.", "", 1): vv for kk, vv in cck["state_dict"].items()})
        cnet.eval()
        lbl = os.path.basename(spec).replace(".pt","")+f"_d{depth}k{kiters}{weighting[0]}"+("" if warm else "_nowarm")+("" if belief=="uniform" else "_rb")
        return ReBeLHero(solver, abstraction, cnet, cck, payouts, depth=depth, n_iters=kiters, weighting=weighting, warm_start=bool(warm), belief=belief), lbl

    heroA, labelA = make_hero(a.net_a, a.depth_a, a.kiters_a, a.weighting_a, a.warmstart_a, a.belief_a)
    heroB, labelB = make_hero(a.net_b, a.depth_b, a.kiters_b, a.weighting_b, a.warmstart_b, a.belief_b)

    print(f"GATE 2 — B={labelB} vs A={labelA}, {a.hands} paired hands/matchup, ICM-equity-delta/hand", flush=True)
    print(f"opponents: Shanky tight bots + built-in archetypes\n", flush=True)
    print(f"{'matchup':<12} {labelA[:10]:>10} {labelB[:10]:>10} {'Δ (B-A)':>10} {'noise±':>8} {'verdict':>10}", flush=True)
    print("-" * 62, flush=True)

    for label, spec in PANEL:
        factory = build_opponent_factory(spec, abstraction, calib, structure)
        As, Bs = [], []
        for i in range(a.hands):
            base = a.seed + i * 7919
            sm = sample_starting_state(structure, random.Random(base * 2 + 1), num_paid=3)
            alive = [s for s in range(6) if sm["stacks"][s] > 0]
            hero_seat = random.Random(base * 3 + 5).choice(alive)
            dealer = sm["dealer_seat"]
            gs = structure.to_inner_game_string_for_state(
                blind_level=sm["blind_level"], stacks=sm["stacks"], dealer_seat=dealer)
            opps = factory(sm["blind_level"].level)
            heroA.new_hand(dealer); heroB.new_hand(dealer)
            va = play_hand(gs, sm, hero_seat, heroA, opps, dealer, base)
            vb = play_hand(gs, sm, hero_seat, heroB, opps, dealer, base)
            if va is not None and vb is not None:
                As.append(va); Bs.append(vb)
        n = len(As)
        mA, mB = statistics.mean(As), statistics.mean(Bs)
        d = [b - x for b, x in zip(Bs, As)]
        md = statistics.mean(d)
        se = (statistics.pstdev(d) / (n ** 0.5)) if n > 1 else 0.0
        verdict = "B+" if md > 2 * se else ("A+" if md < -2 * se else "~tie")
        print(f"{label:<12} {mA:>10.4f} {mB:>10.4f} {md:>+10.4f} {2*se:>8.4f} {verdict:>10}", flush=True)

    print(f"\n(Δ>2·SE = B beats A beyond noise. B={labelB} A={labelA}. TIGHT rows NIT/TAG/KillPhil are decisive.)", flush=True)


if __name__ == "__main__":
    main()
