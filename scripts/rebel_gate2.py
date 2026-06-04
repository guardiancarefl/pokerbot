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
from src.nlhe.equity import cards_from_str, equity_vs_range
from treys import Card as _TreysCard
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


# ====== Option A: short-stack call-vs-shove override =========================
# Diagnostic finding (evals/diag_killphil_1k.json): 93% of ICM loss vs KillPhilMTT
# is on the river, 81% in <10bb starts, 88% when hero's last action was check/call.
# The leak: our resolver assumes a generic (wider) shove range than KillPhil's
# actual tight push range, so we call too wide in short-stack call-vs-shove spots.
# Fix: bypass the abstraction for those decisions — compute hero's hand equity vs
# a hardcoded tight shove range (~16% of hands), compare to ICM-tightened pot
# odds. The override fires ONLY at start_stack < 10bb and to_call >= 70% of remaining
# (i.e., "calling this commits me effectively all-in"), so wider-range spots are
# untouched. Range tuned to a tight 6-max push range; expected to keep us
# calling correctly vs loose bots (their shoves are wider, so our equity-vs-tight
# still passes — we just FOLD MORE vs actually-tight shovers).
_SHOVE_CLASSES = [
    "22","33","44","55","66","77","88","99","TT","JJ","QQ","KK","AA",
    "A2s","A3s","A4s","A5s","A6s","A7s","A8s","A9s","ATs","AJs","AQs","AKs",
    "ATo","AJo","AQo","AKo",
    "KTs","KJs","KQs",
    "KJo","KQo",
    "QJs","JTs",
]
# KillPhilMTT's literal shove ("raisemax force") range — read from
# data/shanky_profiles/KillPhilMTT.txt at <18bb stacks.
_KILLPHIL_SHOVE_CLASSES = [
    "AA","KK","QQ","JJ","TT","99","88","77","66",
    "AKs","AKo","AQs","AQo","AJs","ATs",
]
_SUITS = "shdc"


def _expand_hand_class(name):
    """Expand 'AA', 'AKs', 'AKo' into list of (treys-int, treys-int) tuples."""
    r1, r2 = name[0], name[1]
    if r1 == r2:
        return [(_TreysCard.new(r1 + _SUITS[i]), _TreysCard.new(r1 + _SUITS[j]))
                for i in range(4) for j in range(i + 1, 4)]
    suited = (len(name) == 3 and name[2] == "s")
    if suited:
        return [(_TreysCard.new(r1 + s), _TreysCard.new(r2 + s)) for s in _SUITS]
    return [(_TreysCard.new(r1 + s1), _TreysCard.new(r2 + s2))
            for s1 in _SUITS for s2 in _SUITS if s1 != s2]


TIGHT_SHOVE_COMBOS = []
for _cls in _SHOVE_CLASSES:
    TIGHT_SHOVE_COMBOS.extend(_expand_hand_class(_cls))

KILLPHIL_SHOVE_COMBOS = []
for _cls in _KILLPHIL_SHOVE_CLASSES:
    KILLPHIL_SHOVE_COMBOS.extend(_expand_hand_class(_cls))

SHOVE_RANGE_TABLE = {
    "tight16": TIGHT_SHOVE_COMBOS,
    "killphil7": KILLPHIL_SHOVE_COMBOS,
}


def mlp(in_dim, hidden, out_dim=6):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def _load_policy_net(path):
    """Load a ReBeL policy-net checkpoint (out_dim=9, masked-softmax target).

    Same MLP body as the v3 value net; the only difference is the final layer
    width. Returns (net, ck) where ck retains the metadata (k, in_dim, ...)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out_dim = int(ck.get("out_dim", 9))
    net = mlp(ck["in_dim"], tuple(ck["hidden"]), out_dim=out_dim)
    net.load_state_dict({kk.replace("net.", "", 1): vv for kk, vv in ck["state_dict"].items()})
    net.eval()
    return net, ck


class BlueprintHero:
    name = "k200"
    def __init__(self, solver):
        self.solver = solver; self.dealer = None; self.bb_chips = None
    def new_hand(self, dealer, bb=None):
        self.dealer = dealer; self.bb_chips = bb
    def select_action(self, parsed, state, rng, mode="sample"):
        parsed = dict(parsed); parsed["dealer_seat"] = self.dealer
        return _sample_action_from_policy(self.solver, parsed, state, rng, mode=mode)


class ReBeLHero:
    name = "rebel"
    def __init__(self, solver, abstraction, net, ck, payouts, depth=3, n_iters=150, num_paid=3,
                 weighting="linear", warm_start=True, belief="uniform",
                 policy_net=None, policy_ck=None,
                 short_stack_fix=False, ss_bb_thresh=10.0, ss_commit_thresh=0.7,
                 ss_icm_tighten=0.05, ss_eq_trials=400, ss_range="tight16"):
        self.solver = solver; self.abs = abstraction; self.net = net
        self.ymu, self.ysd, self.k = ck["y_mean"], ck["y_std"], ck["k"]
        self.bd = solver.encoder.max_bucket_dim; self.enc = solver.encoder
        self.payouts = payouts; self.depth = depth; self.n_iters = n_iters; self.num_paid = num_paid
        self.weighting = weighting; self.warm_start = warm_start; self.belief = belief
        self.pn = solver.policy_nets
        self.is_v2 = (getattr(self.pn, "loaded_schema_version", None) == _PN_SCHEMA)
        self.dealer = None; self.bb_chips = None
        self.kstreet = {0: 20, 1: 200, 2: 200, 3: 200}
        self._opp_belief = None  # precomputed per-decision opponent reach-belief (reach mode)
        # ReBeL policy-net warm-start (Brown et al. 2020). When set, the policy net
        # supplies a per-decision σ̂ that warm-starts the CFR solve at the hero root
        # (replaces the blueprint advantage warm-start). Same input shape as the
        # value net (public36 ⊕ belief6×k); output is a masked softmax over 9 slots.
        self.policy_net = policy_net
        if policy_ck is not None:
            assert int(policy_ck.get("k", self.k)) == self.k, \
                f"policy-net k={policy_ck['k']} != value-net k={self.k}"
        # Option A: short-stack call-vs-shove override (see TIGHT_SHOVE_COMBOS comment).
        self.short_stack_fix = short_stack_fix
        self.ss_bb_thresh = ss_bb_thresh
        self.ss_commit_thresh = ss_commit_thresh
        self.ss_icm_tighten = ss_icm_tighten
        self.ss_eq_trials = ss_eq_trials
        if ss_range not in SHOVE_RANGE_TABLE:
            raise ValueError(f"unknown ss_range={ss_range}; "
                             f"valid: {sorted(SHOVE_RANGE_TABLE)}")
        self.ss_range_name = ss_range
        self.ss_range_combos = SHOVE_RANGE_TABLE[ss_range]
        # Per-run counters (for diagnostics; reset by new_hand).
        self._ss_calls = 0; self._ss_folds = 0; self._ss_eqs = []
    def new_hand(self, dealer, bb=None):
        self.dealer = dealer; self.bb_chips = bb

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

    def _policy_prior(self, state, hero, hero_cards, starting):
        """ReBeL policy-net prediction at the hero ROOT: masked softmax (9,)
        over DiscreteAction slots. Input identical to the value net's leaf-eval
        input (public36 ⊕ belief6×k) so the same belief pipeline feeds both heads.
        Returns None if the policy net is not loaded — the resolver then falls back
        to the blueprint warm-start (legacy path)."""
        if self.policy_net is None:
            return None
        # public block + legal mask at the hero root.
        p = parse_state_6max(state); p["dealer_seat"] = self.dealer
        p_pub = dict(p); p_pub["private_cards"] = ""
        feat = np.asarray(self.enc.encode_from_parsed(p_pub, rng=None), dtype=np.float32)
        public = feat[200:]
        view = _build_view_6max(state, p)
        d2c = discretize_legal_actions(list(state.legal_actions()), view)
        mask = np.zeros(9, dtype=np.float32)
        for dd in d2c:
            mask[int(dd)] = 1.0
        # Belief: hero one-hot on its bucket; opponents = reach posterior if precomputed
        # (belief="reach"), else uniform-over-legal-buckets.
        bkey = p.get("public_cards", "") or ""
        board = cards_from_str(bkey)
        hb = int(self.abs.bucket_of(hero_cards, board, runouts=20, rng=random.Random(0))) \
            if len(hero_cards) == 2 else -1
        kk = self.kstreet.get(int(p["street_idx"]), self.k)
        belief = np.zeros((6, self.k), dtype=np.float32)
        if 0 <= hb < self.k:
            belief[hero, hb] = 1.0
        if self.belief == "reach" and self._opp_belief is not None:
            for s in range(6):
                if s != hero and starting[s] > 0:
                    belief[s] = self._opp_belief[s]
        else:
            for s in range(6):
                if s != hero and starting[s] > 0:
                    belief[s, :kk] = 1.0 / kk
        x = np.concatenate([public, belief.reshape(6 * self.k)])[None, :].astype(np.float32)
        with torch.no_grad():
            logits = self.policy_net(torch.from_numpy(x)).numpy()[0]
        # Masked softmax over legal slots.
        logits = np.where(mask > 0, logits, -1e30)
        logits -= logits.max()
        ex = np.exp(logits) * mask
        s = ex.sum()
        if s <= 0:
            return mask / max(float(mask.sum()), 1.0)
        return (ex / s).astype(np.float64)

    def _short_stack_override(self, parsed, state, rng):
        """Option A: bypass the resolver for <10bb call-vs-shove decisions.

        Trigger (all required):
          * `self.short_stack_fix` enabled.
          * `self.bb_chips` known (passed via `new_hand(bb=...)`).
          * Hero's start-of-hand stack < `ss_bb_thresh` big blinds.
          * `to_call > 0` (there is a bet to call).
          * `to_call >= ss_commit_thresh * remaining` (calling commits hero
            effectively all-in — the spot where the resolver overcalls).

        Decision: equity(hero, TIGHT_SHOVE_COMBOS, board) vs chip-EV pot-odds
        threshold `to_call / (pot + to_call)` plus `ss_icm_tighten` ICM padding.
        Returns the OpenSpiel action int (fold or call) or None to defer.
        """
        if not self.short_stack_fix or not self.bb_chips:
            return None
        hero = parsed["current_player"]
        contribs = parsed["contribution"]; monies = parsed["money"]
        to_call = int(max(contribs)) - int(contribs[hero])
        if to_call <= 0:
            return None
        remaining = int(monies[hero])
        if remaining <= 0:
            return None
        start_chips = remaining + int(contribs[hero])
        if start_chips / float(self.bb_chips) >= self.ss_bb_thresh:
            return None
        if to_call < self.ss_commit_thresh * remaining:
            return None
        # Get the legal mapping discrete -> game action; need at least fold or call.
        view = _build_view_6max(state, parsed)
        d2c = discretize_legal_actions(list(state.legal_actions()), view)
        if 0 not in d2c and 1 not in d2c:
            return None  # not a call/fold decision
        # Hero hole cards + board.
        try:
            ph = parse_state_6max(state, observer=hero) if state.current_player() < 0 \
                else parse_state_6max(state)
            hero_cards = cards_from_str(ph.get("private_cards", "") or "")
            board = cards_from_str(ph.get("public_cards", "") or "")
            if len(hero_cards) != 2:
                return None
        except Exception:
            return None
        try:
            eq = float(equity_vs_range(hero_cards, self.ss_range_combos,
                                       board=board, trials=self.ss_eq_trials, rng=rng))
        except Exception:
            return None
        pot = int(sum(contribs))
        threshold = (to_call / float(pot + to_call)) + self.ss_icm_tighten
        self._ss_eqs.append((eq, threshold))
        if eq < threshold and 0 in d2c:
            self._ss_folds += 1
            return int(d2c[0])  # fold
        if 1 in d2c:
            self._ss_calls += 1
            return int(d2c[1])  # call
        return None

    def select_action(self, parsed, state, rng, mode="sample"):
        cp = parsed["current_player"]
        starting = self._starting(parsed)
        # Option A: short-stack call-vs-shove override (bypasses the resolver
        # in the exact spots the KillPhil-leak diagnostic identified).
        ssfix = self._short_stack_override(parsed, state, rng)
        if ssfix is not None:
            return ssfix
        try:
            hero_cards = cards_from_str(parse_state_6max(state, observer=cp).get("private_cards", "") or "")
            self._opp_belief = self._root_belief(state, cp, starting) if self.belief == "reach" else None
            tree = build_subgame_tree(state, max_action_depth=self.depth,
                                      chance_samples_per_node=2, rng=rng)
            cache = {}
            for leaf in iter_leaf_nodes(tree):
                leaf.leaf_value = self._leaf_val(leaf, cp, hero_cards, cache)
            policy_prior = self._policy_prior(state, cp, hero_cards, starting)
            res = solve_subgame(tree, SubgameSolveContext(
                blueprint=self.solver, starting_stacks=starting, payouts=self.payouts,
                hero_seat=cp, n_iterations=self.n_iters, rng=rng, num_paid=self.num_paid,
                average_weighting=self.weighting, warm_start=self.warm_start, dealer_seat=self.dealer,
                root_policy_prior=policy_prior))
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
    ap.add_argument("--policy-net-a", default=None,
                    help="path to ReBeL policy-net .pt for hero A (warm-start CFR at root)")
    ap.add_argument("--policy-net-b", default=None,
                    help="path to ReBeL policy-net .pt for hero B (warm-start CFR at root)")
    ap.add_argument("--matchups", default=None,
                    help="comma-separated subset of panel labels to run (e.g. KillPhilMTT,NIT,TAG); default=all")
    # Option A short-stack call-vs-shove override (per-hero).
    ap.add_argument("--short-stack-fix-a", type=int, default=0,
                    help="1 = enable <10bb tight-shove-equity call/fold override on hero A")
    ap.add_argument("--short-stack-fix-b", type=int, default=0,
                    help="1 = enable <10bb tight-shove-equity call/fold override on hero B")
    ap.add_argument("--ss-bb-thresh", type=float, default=10.0,
                    help="start-stack threshold (bb) for override trigger")
    ap.add_argument("--ss-commit-thresh", type=float, default=0.7,
                    help="commit threshold (to_call / remaining) for override trigger")
    ap.add_argument("--ss-icm-tighten", type=float, default=0.05,
                    help="ICM-tightening added to chip-EV equity threshold")
    ap.add_argument("--ss-range", default="tight16",
                    help="assumed shove range: 'tight16' (~16% 6-max push) or 'killphil7' (KP allin range)")
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
    if a.matchups:
        keep = set(s.strip() for s in a.matchups.split(",") if s.strip())
        PANEL = [(lbl, spec) for (lbl, spec) in PANEL if lbl in keep]
        if not PANEL:
            raise SystemExit(f"--matchups filtered out all rows; valid labels are: KillPhilMTT,timidtom,NIT,TAG,LAG,mixed,STATION")

    structure = TournamentStructure.from_yaml(STRUCT)
    abstraction = Abstraction.load(ABSTR)
    solver = _load_solver(CKPT, abstraction, structure)
    calib = EquityCalibration.load(CALIB)
    payouts = list(sng_payouts_6max_double_up())
    def make_hero(spec, depth=3, kiters=150, weighting="linear", warm=1, belief="uniform",
                  policy_path=None, ss_fix=False):
        if spec == "k200":
            return BlueprintHero(solver), "k200"
        cck = torch.load(spec, map_location="cpu", weights_only=False)
        cnet = mlp(cck["in_dim"], tuple(cck["hidden"]))
        cnet.load_state_dict({kk.replace("net.", "", 1): vv for kk, vv in cck["state_dict"].items()})
        cnet.eval()
        pnet = pck = None
        if policy_path:
            pnet, pck = _load_policy_net(policy_path)
        lbl = os.path.basename(spec).replace(".pt","")+f"_d{depth}k{kiters}{weighting[0]}" \
              + ("" if warm else "_nowarm") + ("" if belief=="uniform" else "_rb") \
              + ("_pn" if pnet is not None else "") \
              + ("_ssf" if ss_fix else "")
        return ReBeLHero(solver, abstraction, cnet, cck, payouts, depth=depth, n_iters=kiters,
                         weighting=weighting, warm_start=bool(warm), belief=belief,
                         policy_net=pnet, policy_ck=pck,
                         short_stack_fix=ss_fix,
                         ss_bb_thresh=a.ss_bb_thresh,
                         ss_commit_thresh=a.ss_commit_thresh,
                         ss_icm_tighten=a.ss_icm_tighten,
                         ss_range=a.ss_range), lbl

    heroA, labelA = make_hero(a.net_a, a.depth_a, a.kiters_a, a.weighting_a, a.warmstart_a, a.belief_a,
                              policy_path=a.policy_net_a, ss_fix=bool(a.short_stack_fix_a))
    heroB, labelB = make_hero(a.net_b, a.depth_b, a.kiters_b, a.weighting_b, a.warmstart_b, a.belief_b,
                              policy_path=a.policy_net_b, ss_fix=bool(a.short_stack_fix_b))

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
            bb_chips = sm["blind_level"].inflated_big_blind(6)
            heroA.new_hand(dealer, bb=bb_chips); heroB.new_hand(dealer, bb=bb_chips)
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

    # Short-stack-fix telemetry: how often did the override fire on each hero?
    for hh, hl in [(heroA, "A"), (heroB, "B")]:
        if isinstance(hh, ReBeLHero) and hh.short_stack_fix:
            n_fires = hh._ss_calls + hh._ss_folds
            if n_fires:
                eq_mean = statistics.mean(e for e, _ in hh._ss_eqs)
                th_mean = statistics.mean(t for _, t in hh._ss_eqs)
                print(f"[ss-fix {hl}] fires={n_fires}  calls={hh._ss_calls}  folds={hh._ss_folds}  "
                      f"mean_eq={eq_mean:.3f}  mean_threshold={th_mean:.3f}", flush=True)
            else:
                print(f"[ss-fix {hl}] enabled but never fired (no <10bb shove-call spots)", flush=True)

    print(f"\n(Δ>2·SE = B beats A beyond noise. B={labelB} A={labelA}. TIGHT rows NIT/TAG/KillPhil are decisive.)", flush=True)


if __name__ == "__main__":
    main()
