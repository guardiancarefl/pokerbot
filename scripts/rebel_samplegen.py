"""ReBeL Phase-2 sample generation (bootstrap round) — multi-worker, multi-box.

Generalises scripts/rebel_samplegen_pilot.py from a *rate measurement* into a
real generator that persists every solved decision as a (PBS, target) sample via
src/rebel/sample_io.ShardWriter. Embarrassingly parallel: each worker is a CPU
process writing its own host+worker-prefixed shards, so N boxes pointed at their
own /workspace merge losslessly (see sample_io.merge_shards).

Per sample (docs/REBEL_PBS_6MAX.md §5):
  self-play to a hero decision PBS -> build_subgame_tree -> evaluate_leaves
  (PROFILE_SAMPLE bootstrap leaf) -> solve_subgame (K-iter CFR) -> persist
  (root PBS encoding, root value + policy + q).

Stop condition: --target samples (across this box's workers) OR --seconds wall.
Checkpointing: ShardWriter flushes an atomic shard every --flush-every samples,
so a disconnect loses at most one partial buffer.

Single box (this RTX box, ~27 cores):
  .venv/bin/python scripts/rebel_samplegen.py --workers 26 --target 1000000 \
      --out /workspace/rebel_samples
Second box (e.g. Contabo, 12 cores) — SAME command, fewer workers; the host
prefix keeps shards distinct so both /workspace dirs merge:
  .venv/bin/python scripts/rebel_samplegen.py --workers 11 --target 1500000 \
      --out /workspace/rebel_samples
"""
import sys, os, time, random, argparse, socket
import multiprocessing as mp
sys.path.insert(0, '.')

ABSTR = "runs/k200_abstraction.pkl"
CKPT = "runs/k200_blueprint_ckpt_iter_2000.pt"
STRUCT = "configs/ignition_double_up_6max_turbo.yaml"


def worker(wid, target_per_worker, seconds, depth, M, n_iters, out_dir,
           flush_every, ret):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU per worker (bootstrap leaf is CPU rollout)
    import numpy as np
    import torch; torch.set_num_threads(1)
    import pyspiel
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure, six_max_sng
    from src.nlhe.biased_policy import BiasedBlueprint
    from src.nlhe.icm import sng_payouts_6max_double_up
    from src.nlhe.subgame import build_subgame_tree, tree_depth
    from src.nlhe.subgame_leaf import LeafEvalContext, LeafEvalMode, evaluate_leaves
    from src.nlhe.subgame_solver import SubgameSolveContext, solve_subgame
    from src.nlhe.infoset6 import parse_state_6max
    from src.rebel.sample_io import ShardWriter, RebelSample, file_sha1
    from scripts.eval_6max_self_play import _load_solver

    solver = _load_solver(CKPT, Abstraction.load(ABSTR), TournamentStructure.from_yaml(STRUCT))
    encoder = solver.encoder
    bucket_dim = encoder.max_bucket_dim
    stack = int(solver.encoder.starting_stack)
    game = pyspiel.load_game(six_max_sng(starting_stack=stack))
    payouts = list(sng_payouts_6max_double_up())
    rng = random.Random(1000 + wid)

    meta = {
        "blueprint_sha": file_sha1(CKPT),
        "abstraction_sha": file_sha1(ABSTR),
        "k_postflop": int(bucket_dim),
        "feat_dim": int(encoder.feature_dim),
        "n_actions": 9,
        "starting_stack": stack,
        "depth": depth, "M": M, "n_iterations": n_iters,
        "leaf_mode": "PROFILE_SAMPLE",
        "round": "bootstrap",
    }
    writer = ShardWriter(root_dir=out_dir, worker=wid, flush_every=flush_every, meta=meta)

    def a_decision():
        s = game.new_initial_state()
        while s.is_chance_node():
            a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
        g = 0
        while not s.is_terminal() and not s.child(1).is_chance_node():
            s = s.child(1); g += 1
            if g >= 18: break
        if s.is_terminal(): return None
        s = s.child(1)
        while s.is_chance_node():
            a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
        return s if not s.is_terminal() else None

    n = 0
    t_gen_start = time.perf_counter()   # after all load; steady-state clock
    t_end = t_gen_start + seconds if seconds else float("inf")
    while n < target_per_worker and time.perf_counter() < t_end:
        s = a_decision()
        if s is None:
            continue
        try:
            tree = build_subgame_tree(s, max_action_depth=depth, chance_samples_per_node=2, rng=rng)
            hero = tree.root.current_player
            lctx = LeafEvalContext(blueprint=solver, biased_blueprint=BiasedBlueprint(),
                                   starting_stacks=[stack] * 6, payouts=payouts, hero_seat=hero,
                                   mode=LeafEvalMode.PROFILE_SAMPLE, n_samples=M, rng=rng)
            evaluate_leaves(tree, lctx)
            sctx = SubgameSolveContext(blueprint=solver, starting_stacks=[stack] * 6, payouts=payouts,
                                       hero_seat=hero, n_iterations=n_iters, rng=rng, num_paid=3,
                                       average_weighting="linear")
            res = solve_subgame(tree, sctx)

            # --- PBS encoding (root) + target (search output) ---
            parsed = parse_state_6max(tree.root.state)
            feat = np.asarray(encoder.encode_from_parsed(parsed, rng=rng), dtype=np.float32)
            onehot = feat[:bucket_dim]
            bucket = int(onehot.argmax()) if float(onehot.max()) > 0 else -1
            policy = np.asarray(res.root_policy, dtype=np.float32)
            qvals = np.asarray(res.root_q_values, dtype=np.float32)
            mask = np.asarray(res.legal_mask, dtype=np.int8)
            root_value = float((policy * qvals * (mask > 0)).sum())

            writer.add(RebelSample(
                feat=feat, bucket=bucket, hero_seat=int(hero),
                street=int(parsed["street_idx"]), depth=int(tree_depth(tree)),
                n_iterations=int(res.n_iterations), degraded=bool(res.degraded),
                root_value=root_value, root_policy=policy, root_q_values=qvals,
                legal_mask=mask,
            ))
            n += 1
            if n % 200 == 0:
                ret[wid] = n  # live progress heartbeat
        except Exception as e:
            ret['err_%d' % wid] = repr(e)[:160]

    writer.close()
    ret['gen_%d' % wid] = time.perf_counter() - t_gen_start  # steady-state seconds (load excluded)
    ret[wid] = n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--target", type=int, default=1_000_000, help="total samples for THIS box")
    ap.add_argument("--seconds", type=int, default=0, help="optional wall-clock cap (0 = none)")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--n-iters", type=int, default=150)
    ap.add_argument("--out", default="/workspace/rebel_samples")
    ap.add_argument("--flush-every", type=int, default=256)
    a = ap.parse_args()

    per_worker = (a.target + a.workers - 1) // a.workers
    os.makedirs(a.out, exist_ok=True)
    host = socket.gethostname()
    print(f"[samplegen] host={host} workers={a.workers} target={a.target} "
          f"(~{per_worker}/worker) depth={a.depth} M={a.M} K={a.n_iters} out={a.out}", flush=True)

    ctx = mp.get_context("spawn")
    ret = ctx.Manager().dict()
    procs = [ctx.Process(target=worker,
                         args=(w, per_worker, a.seconds, a.depth, a.M, a.n_iters,
                               a.out, a.flush_every, ret))
             for w in range(a.workers)]
    t0 = time.perf_counter()
    for p in procs: p.start()
    for p in procs: p.join()
    wall = time.perf_counter() - t0

    total = sum(v for k, v in ret.items() if isinstance(k, int))
    gen_secs = [v for k, v in ret.items() if isinstance(k, str) and k.startswith("gen_")]
    errs = [v for k, v in ret.items() if isinstance(k, str) and not k.startswith("gen_")]
    # Steady-state rate = samples / the workers' parallel gen window (load excluded);
    # total/wall under-reports on short runs where spawn+import dominates, but equals
    # steady-state on a long run where startup amortizes to nothing.
    gen_window = max(gen_secs) if gen_secs else wall
    print(f"[samplegen] DONE host={host} wall={wall:.1f}s gen_window={gen_window:.1f}s "
          f"total_samples={total}", flush=True)
    print(f"[samplegen] steady-state: {total/gen_window:.2f} samples/s = "
          f"{total/gen_window*3600:.0f} samples/hour  "
          f"(wall-incl-startup: {total/wall*3600:.0f}/hr)", flush=True)
    if errs:
        print("[samplegen] errors (sample):", errs[:3], flush=True)
