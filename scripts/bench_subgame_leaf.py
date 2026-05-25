"""Benchmark harness for the depth-limited subgame LEAF evaluator (Q13).

Grounds every numerical claim in docs/STAGE_E_BUDGET_REDERIVATION.md with a
fresh, attributed measurement: per-config wall-clock plus a per-component
decomposition that reports, for each bottleneck candidate, PER-CALL cost AND
CALL FREQUENCY AND (for the encoder) POST-CACHE MISS frequency — the
measurement-discipline sub-rule promoted in design doc Q4.5.

Instrumentation points (wrapped in place for one evaluate_leaves call):
  - encoder.encode_from_parsed   -> encoder calls + total time (incl. bucket-MC)
  - abstraction.bucket_of        -> cache MISSES (called only on miss) + time
  - policy_nets.predict_advantages -> network forwards + time
  - subgame_leaf.parse_state_6max  -> parse calls + time
  - subgame_leaf.fast_view_and_discretize -> view calls + time
  glue = wall-clock - (encode + network[outside encode? no] + parse + view)
  NOTE network forward is OUTSIDE encode (separate call), bucket-MC is INSIDE
  encode, so encode_rest = encode_total - bucket_time and
  glue = total - parse - view - encode_total - network.

Usage:
  python -m scripts.bench_subgame_leaf            # all configs
  python -m scripts.bench_subgame_leaf --parallel # parallelism check only
"""
from __future__ import annotations
import argparse
import random
import time
from dataclasses import dataclass

import pyspiel

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import TournamentStructure, six_max_sng
from src.nlhe.biased_policy import BiasedBlueprint
from src.nlhe.icm import sng_payouts_6max_double_up
from src.nlhe.subgame import build_subgame_tree, iter_leaf_nodes
from src.nlhe import subgame_leaf as SL
from src.nlhe.subgame_leaf import LeafEvalContext, LeafEvalMode, evaluate_leaves
from scripts.eval_6max_self_play import _load_solver

ABSTR = "runs/abstraction_20260521_223018/abstraction.pkl"
CKPT = "runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/ckpt_iter_3000.pt"
STRUCT = "configs/ignition_double_up_6max_turbo.yaml"


@dataclass
class Counter:
    calls: int = 0
    t: float = 0.0


class Instr:
    """Wraps the bottleneck-candidate callables for one measured run."""
    def __init__(self, solver):
        self.solver = solver
        self.encode = Counter()
        self.bucket = Counter()   # cache MISSES (bucket_of is called only on miss)
        self.net = Counter()
        self.parse = Counter()
        self.view = Counter()
        self._orig = {}

    def __enter__(self):
        s = self.solver
        enc, buck, net, par, vie = (self.encode, self.bucket, self.net,
                                    self.parse, self.view)

        o_encode = s.encoder.encode_from_parsed
        def w_encode(parsed, rng=None):
            t0 = time.perf_counter(); r = o_encode(parsed, rng=rng)
            enc.t += time.perf_counter() - t0; enc.calls += 1; return r
        s.encoder.encode_from_parsed = w_encode
        self._orig['encode'] = (s.encoder, 'encode_from_parsed', o_encode)

        o_bucket = s.abstraction.bucket_of
        def w_bucket(hero, board, runouts, rng):
            t0 = time.perf_counter(); r = o_bucket(hero, board, runouts, rng)
            buck.t += time.perf_counter() - t0; buck.calls += 1; return r
        s.abstraction.bucket_of = w_bucket
        self._orig['bucket'] = (s.abstraction, 'bucket_of', o_bucket)

        o_net = s.policy_nets.predict_advantages
        def w_net(seat, features):
            t0 = time.perf_counter(); r = o_net(seat, features)
            net.t += time.perf_counter() - t0; net.calls += 1; return r
        s.policy_nets.predict_advantages = w_net
        self._orig['net'] = (s.policy_nets, 'predict_advantages', o_net)

        o_parse = SL.parse_state_6max
        def w_parse(state):
            t0 = time.perf_counter(); r = o_parse(state)
            par.t += time.perf_counter() - t0; par.calls += 1; return r
        SL.parse_state_6max = w_parse
        self._orig['parse'] = (SL, 'parse_state_6max', o_parse)

        o_view = SL.fast_view_and_discretize
        def w_view(state, parsed):
            t0 = time.perf_counter(); r = o_view(state, parsed)
            vie.t += time.perf_counter() - t0; vie.calls += 1; return r
        SL.fast_view_and_discretize = w_view
        self._orig['view'] = (SL, 'fast_view_and_discretize', o_view)
        return self

    def __exit__(self, *a):
        for obj, name, orig in self._orig.values():
            setattr(obj, name, orig)


def _load():
    abstr = Abstraction.load(ABSTR)
    structure = TournamentStructure.from_yaml(STRUCT)
    solver = _load_solver(CKPT, abstr, structure)
    stack = int(solver.encoder.starting_stack)
    game = pyspiel.load_game(six_max_sng(starting_stack=stack))
    return solver, structure, stack, game, BiasedBlueprint()


def _walk_to_decision(game, seed, postflop=True):
    rng = random.Random(seed)
    s = game.new_initial_state()
    while s.is_chance_node():
        a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
    if not postflop:
        return s
    guard = 0
    while not s.child(1).is_chance_node():
        s = s.child(1); guard += 1; assert guard < 20
    s = s.child(1)
    while s.is_chance_node():
        a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
    return s


def build_tree(game, depth, seed=11, chance_samples=2):
    st = _walk_to_decision(game, seed=seed, postflop=True)
    return build_subgame_tree(st, max_action_depth=depth,
                              chance_samples_per_node=chance_samples,
                              rng=random.Random(seed))


def make_ctx(solver, stack, biased, tree, mode, M, seed=101):
    return LeafEvalContext(
        blueprint=solver, biased_blueprint=biased,
        starting_stacks=[stack] * 6, payouts=list(sng_payouts_6max_double_up()),
        hero_seat=tree.root.current_player, mode=mode, n_samples=M,
        rng=random.Random(seed))


def run_config(solver, stack, biased, tree, mode, M, label):
    n_leaves = len(list(iter_leaf_nodes(tree)))
    ctx = make_ctx(solver, stack, biased, tree, mode, M)
    instr = Instr(solver)
    t0 = time.perf_counter()
    with instr:
        res = evaluate_leaves(tree, ctx)
    wall = time.perf_counter() - t0

    enc_rest = instr.encode.t - instr.bucket.t
    glue = wall - instr.parse.t - instr.view.t - instr.encode.t - instr.net.t
    pct = lambda x: 100.0 * x / wall if wall > 0 else 0.0
    print(f"\n=== {label} | leaves={n_leaves} evaluated={res.n_evaluated} ===")
    print(f"  WALL: {wall:.3f} s")
    def line(name, c: Counter, t):
        per = (1e3 * c.t / c.calls) if c.calls else 0.0
        print(f"  {name:<16} {pct(t):5.1f}%  t={t:7.3f}s  calls={c.calls:>7}  "
              f"per-call={per:7.4f} ms")
    print(f"  {'network':<16} {pct(instr.net.t):5.1f}%  t={instr.net.t:7.3f}s  "
          f"calls={instr.net.calls:>7}  per-call={1e3*instr.net.t/max(1,instr.net.calls):7.4f} ms")
    print(f"  {'bucket-MC(miss)':<16} {pct(instr.bucket.t):5.1f}%  t={instr.bucket.t:7.3f}s  "
          f"calls={instr.bucket.calls:>7}  per-call={1e3*instr.bucket.t/max(1,instr.bucket.calls):7.4f} ms")
    print(f"  {'encode(rest)':<16} {pct(enc_rest):5.1f}%  t={enc_rest:7.3f}s  "
          f"calls={instr.encode.calls:>7}  (encode_total={instr.encode.t:.3f}s)")
    line("parse", instr.parse, instr.parse.t)
    line("view+disc", instr.view, instr.view.t)
    print(f"  {'glue(rest)':<16} {pct(glue):5.1f}%  t={glue:7.3f}s")
    hit = 1.0 - instr.bucket.calls / max(1, instr.encode.calls)
    print(f"  encoder: {instr.encode.calls} calls, {instr.bucket.calls} misses "
          f"({100*hit:.1f}% hit), network: {instr.net.calls} forwards")
    return {"label": label, "wall": wall, "leaves": n_leaves,
            "net_calls": instr.net.calls, "net_t": instr.net.t,
            "bucket_calls": instr.bucket.calls, "bucket_t": instr.bucket.t,
            "encode_calls": instr.encode.calls, "encode_t": instr.encode.t,
            "parse_t": instr.parse.t, "view_t": instr.view.t, "glue": glue}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parallel", action="store_true")
    args = p.parse_args()

    solver, structure, stack, game, biased = _load()
    print(f"Loaded blueprint {CKPT}")
    print(f"stack={stack}")

    tree3 = build_tree(game, depth=3)
    tree4 = build_tree(game, depth=4)
    print(f"depth-3 leaves={len(list(iter_leaf_nodes(tree3)))}  "
          f"depth-4 leaves={len(list(iter_leaf_nodes(tree4)))}")

    results = []
    # warmup (JIT/torch graph + fill caches once on a throwaway tree)
    _ = run_config(solver, stack, biased, build_tree(game, depth=2),
                   LeafEvalMode.PROFILE_SAMPLE, 4, "WARMUP (ignore)")

    results.append(run_config(solver, stack, biased, tree3,
                              LeafEvalMode.PROFILE_SAMPLE, 8, "PROFILE M=8 depth-3"))
    results.append(run_config(solver, stack, biased, tree3,
                              LeafEvalMode.BEST_RESPONSE, 5, "BR M=5 depth-3"))
    results.append(run_config(solver, stack, biased, tree3,
                              LeafEvalMode.BEST_RESPONSE, 8, "BR M=8 depth-3"))
    results.append(run_config(solver, stack, biased, tree4,
                              LeafEvalMode.BEST_RESPONSE, 5, "BR M=5 depth-4"))

    print("\n\n===== SUMMARY =====")
    for r in results:
        print(f"{r['label']:<22} wall={r['wall']:7.2f}s leaves={r['leaves']:>4} "
              f"net={r['net_calls']:>6}/{1e3*r['net_t']/max(1,r['net_calls']):.3f}ms "
              f"bucketmiss={r['bucket_calls']:>5} "
              f"net%={100*r['net_t']/r['wall']:4.1f} bucket%={100*r['bucket_t']/r['wall']:4.1f} "
              f"glue%={100*r['glue']/r['wall']:4.1f}")


if __name__ == "__main__":
    main()
