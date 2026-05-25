"""Parallelism check for evaluate_leaves (Q13 Measurement set 3).

Question: the box has 128 CPU cores + 1 GPU. Does evaluate_leaves parallelize,
or does the single GPU serialize the network forward (45% of BR cost)? And is
N CPU workers (own model each) faster than N GPU workers sharing one device?

Method: spawn W worker processes, each loads its own solver, builds the SAME
depth-3 tree, waits on a Barrier (so all start compute simultaneously), then
times ONE BR M=5 evaluate_leaves call. The per-worker COMPUTE time (load
excluded) under contention vs solo isolates the contention slowdown:
  slowdown = mean(parallel_compute) / solo_compute
  parallel_speedup = W / slowdown        (W if perfectly parallel, 1 if serial)
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import random
import time

ABSTR = "runs/abstraction_20260521_223018/abstraction.pkl"
CKPT = "runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/ckpt_iter_3000.pt"
STRUCT = "configs/ignition_double_up_6max_turbo.yaml"


def _worker(wid, device, depth, M, barrier, ret, threads):
    import torch
    torch.set_num_threads(threads)
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import pyspiel
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure, six_max_sng
    from src.nlhe.biased_policy import BiasedBlueprint
    from src.nlhe.icm import sng_payouts_6max_double_up
    from src.nlhe.subgame import build_subgame_tree
    from src.nlhe.subgame_leaf import LeafEvalContext, LeafEvalMode, evaluate_leaves
    from scripts.eval_6max_self_play import _load_solver

    solver = _load_solver(CKPT, Abstraction.load(ABSTR),
                          TournamentStructure.from_yaml(STRUCT))
    stack = int(solver.encoder.starting_stack)
    game = pyspiel.load_game(six_max_sng(starting_stack=stack))
    rng = random.Random(11); s = game.new_initial_state()
    while s.is_chance_node():
        a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
    g = 0
    while not s.child(1).is_chance_node():
        s = s.child(1); g += 1; assert g < 20
    s = s.child(1)
    while s.is_chance_node():
        a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
    tree = build_subgame_tree(s, max_action_depth=depth, chance_samples_per_node=2,
                              rng=random.Random(11))
    ctx = LeafEvalContext(
        blueprint=solver, biased_blueprint=BiasedBlueprint(),
        starting_stacks=[stack] * 6, payouts=list(sng_payouts_6max_double_up()),
        hero_seat=tree.root.current_player, mode=LeafEvalMode.BEST_RESPONSE,
        n_samples=M, rng=random.Random(101 + wid))
    barrier.wait()
    t0 = time.perf_counter()
    evaluate_leaves(tree, ctx)
    ret[wid] = time.perf_counter() - t0


def run(device, W, depth, M, threads):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(W)
    ret = ctx.Manager().dict()
    procs = [ctx.Process(target=_worker,
                         args=(i, device, depth, M, barrier, ret, threads))
             for i in range(W)]
    wall0 = time.perf_counter()
    for p in procs: p.start()
    for p in procs: p.join()
    wall = time.perf_counter() - wall0
    times = [ret[i] for i in range(W)]
    return times, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--devices", default="cuda,cpu")
    args = ap.parse_args()
    W = args.workers

    print(f"Parallelism check: BR M={args.M} depth-{args.depth}, W={W} workers\n")
    _devmap = {"cuda": 4, "cpu": 1}
    for device in args.devices.split(","):
        threads = _devmap[device]
        solo, _ = run(device, 1, args.depth, args.M, threads)
        par, wall = run(device, W, args.depth, args.M, threads)
        solo_t = solo[0]
        mean_par = sum(par) / len(par)
        slowdown = mean_par / solo_t
        speedup = W / slowdown
        print(f"[{device}] threads/worker={threads}")
        print(f"  solo compute:     {solo_t:6.2f} s")
        print(f"  parallel compute: mean={mean_par:6.2f} s  min={min(par):.2f} max={max(par):.2f}  (W={W})")
        print(f"  contention slowdown: {slowdown:.2f}x   => parallel speedup: {speedup:.2f}x of {W}")
        print(f"  effective throughput vs 1 solo worker: {speedup:.2f}x")
        print(f"  (wall incl. load: {wall:.1f}s)\n")


if __name__ == "__main__":
    main()
