"""k=1000 abstraction test — generate POSTFLOP value samples, dual-encoded.

k=1000 only refines POSTFLOP (preflop is k=20 in both abstractions), so this
isolates postflop spots and encodes each in BOTH k=200 and k=1000 — identical
spot, identical target (from the k=1000 search/blueprint), only the input
resolution differs. Train a net on each and compare R²: does finer input let the
net break the k=200 0.39 ceiling?

Compact storage (sidesteps the 16KB/sample disk wall): we store the two feature
vectors + the per-seat value target. No dense belief block — opponents are
implicit-uniform (the belief ablation was +0.033; the hero bucket is the lever),
so the value-net input is just the feat (public + hero bucket).
"""
import sys, os, time, random, argparse, socket
import multiprocessing as mp
sys.path.insert(0, '.')

STRUCT = "configs/ignition_double_up_6max_turbo.yaml"
K1000_AB = "/workspace/k1000_artifacts/abstraction_k1000_retrofit_20260530_221744/abstraction.pkl"
K1000_CK = "/workspace/k1000_artifacts/six_max_20260602_135657_candC_k1000_launch/checkpoints/ckpt_iter_0527.pt"
K200_AB = "runs/k200_abstraction.pkl"
_NUM_SEATS = 6
# Heavy postflop weighting: we only care about postflop spots here (where k=1000
# differs). Preflop decisions are kept rare just for trajectory realism.
_STREET_W = {0: 0.04, 1: 1.0, 2: 1.0, 3: 1.0}


def worker(wid, target_per_worker, seconds, depth, M, n_iters, num_paid, out_dir, flush_every, ret):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
    import numpy as np
    import torch; torch.set_num_threads(1)
    import pyspiel
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from src.nlhe.stack_sampler import sample_starting_state
    from src.nlhe.biased_policy import BiasedBlueprint
    from src.nlhe.icm import sng_payouts_6max_double_up
    from src.nlhe.subgame import build_subgame_tree, tree_depth
    from src.nlhe.subgame_leaf import LeafEvalContext, LeafEvalMode, evaluate_leaves
    from src.nlhe.subgame_solver import SubgameSolveContext, solve_subgame
    from src.nlhe.infoset6 import parse_state_6max, InfosetEncoder6Max
    from scripts.eval_6max_self_play import _load_solver, _sample_action_from_policy

    structure = TournamentStructure.from_yaml(STRUCT)
    ab1000 = Abstraction.load(K1000_AB)
    ab200 = Abstraction.load(K200_AB)
    solver = _load_solver(K1000_CK, ab1000, structure)   # k=1000 search + k=1000 encoder
    enc1000 = solver.encoder
    enc200 = InfosetEncoder6Max(abstraction=ab200, starting_stack=int(enc1000.starting_stack))
    payouts = list(sng_payouts_6max_double_up())
    rng = random.Random(4000 + wid)

    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    host = socket.gethostname()
    buf = {"feat1000": [], "feat200": [], "value6": [], "hero": [], "street": [], "blind": [], "alive": []}
    seq = [0]; written = [0]

    def flush():
        if not buf["value6"]:
            return
        cols = {k: np.asarray(v) for k, v in buf.items()}
        name = f"k1shard-{host}-w{wid:02d}-{ts}-{seq[0]:05d}.npz"
        path = os.path.join(out_dir, name); tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **cols); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        seq[0] += 1; written[0] += len(cols["value6"])
        for v in buf.values():
            v.clear()

    def walk_root():
        sm = sample_starting_state(structure, rng, num_paid=num_paid)
        gs = structure.to_inner_game_string_for_state(blind_level=sm["blind_level"], stacks=sm["stacks"], dealer_seat=sm["dealer_seat"])
        g = pyspiel.load_game(gs); s = g.new_initial_state(); dealer = sm["dealer_seat"]; stacks = sm["stacks"]
        cand = []
        steps = 0
        while not s.is_terminal() and steps < 80:
            steps += 1
            if s.is_chance_node():
                a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0])); continue
            cp = s.current_player()
            if cp >= 0 and stacks[cp] > 0:
                st_idx = int(parse_state_6max(s)["street_idx"])
                cand.append((s.clone(), st_idx))
            parsed = parse_state_6max(s); parsed["dealer_seat"] = dealer
            s = s.child(int(_sample_action_from_policy(solver, parsed, s, rng, "sample")))
        if not cand:
            return None
        ws = [_STREET_W.get(st, 1.0) for _, st in cand]
        i = rng.choices(range(len(cand)), weights=ws, k=1)[0]
        return cand[i][0], sm

    n = 0
    t0 = time.perf_counter(); t_end = t0 + seconds if seconds else float("inf")
    while n < target_per_worker and time.perf_counter() < t_end:
        try:
            r = walk_root()
            if r is None:
                continue
            root, sm = r; stacks = list(sm["stacks"]); dealer = sm["dealer_seat"]
            hero = root.current_player()
            if hero < 0 or stacks[hero] <= 0:
                continue
            tree = build_subgame_tree(root, max_action_depth=depth, chance_samples_per_node=2, rng=rng)
            evaluate_leaves(tree, LeafEvalContext(blueprint=solver, biased_blueprint=BiasedBlueprint(),
                            starting_stacks=stacks, payouts=payouts, hero_seat=hero, mode=LeafEvalMode.PROFILE_SAMPLE,
                            n_samples=M, rng=rng, num_paid=num_paid, dealer_seat=dealer))
            res = solve_subgame(tree, SubgameSolveContext(blueprint=solver, starting_stacks=stacks, payouts=payouts,
                            hero_seat=hero, n_iterations=n_iters, rng=rng, num_paid=num_paid,
                            average_weighting="linear", dealer_seat=dealer))
            if res.root_value_per_seat is None:
                continue
            parsed = parse_state_6max(root); parsed["dealer_seat"] = dealer
            f1000 = np.asarray(enc1000.encode_from_parsed(parsed, rng=rng), dtype=np.float32)
            f200 = np.asarray(enc200.encode_from_parsed(parsed, rng=rng), dtype=np.float32)
            buf["feat1000"].append(f1000); buf["feat200"].append(f200)
            buf["value6"].append(np.asarray(res.root_value_per_seat, dtype=np.float32))
            buf["hero"].append(hero); buf["street"].append(int(parsed["street_idx"]))
            buf["blind"].append(int(sm["blind_level"].level)); buf["alive"].append(int(sm["alive_count"]))
            n += 1
            if len(buf["value6"]) >= flush_every:
                flush()
            if n % 200 == 0:
                ret[wid] = n
        except Exception as e:
            ret['err_%d' % wid] = repr(e)[:200]
    flush()
    ret['gen_%d' % wid] = time.perf_counter() - t0
    ret[wid] = written[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--target", type=int, default=1_000_000)
    ap.add_argument("--seconds", type=int, default=0)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--n-iters", type=int, default=150)
    ap.add_argument("--num-paid", type=int, default=3)
    ap.add_argument("--out", default="/workspace/rebel_k1000")
    ap.add_argument("--flush-every", type=int, default=256)
    a = ap.parse_args()
    per = (a.target + a.workers - 1) // a.workers
    os.makedirs(a.out, exist_ok=True)
    print(f"[k1000gen] workers={a.workers} target={a.target} depth={a.depth} M={a.M} K={a.n_iters} "
          f"out={a.out} (postflop-weighted, dual-encode k200+k1000)", flush=True)
    ctx = mp.get_context("spawn"); ret = ctx.Manager().dict()
    procs = [ctx.Process(target=worker, args=(w, per, a.seconds, a.depth, a.M, a.n_iters, a.num_paid, a.out, a.flush_every, ret)) for w in range(a.workers)]
    t0 = time.perf_counter()
    for p in procs: p.start()
    for p in procs: p.join()
    wall = time.perf_counter() - t0
    total = sum(v for k, v in ret.items() if isinstance(k, int))
    gens = [v for k, v in ret.items() if isinstance(k, str) and k.startswith("gen_")]
    errs = [v for k, v in ret.items() if isinstance(k, str) and not k.startswith("gen_")]
    gw = max(gens) if gens else wall
    print(f"[k1000gen] DONE wall={wall:.1f}s gen_window={gw:.1f}s total={total} "
          f"=> {total/gw:.1f}/s = {total/gw*3600:.0f}/hr", flush=True)
    if errs: print("[k1000gen] errors:", errs[:3], flush=True)
