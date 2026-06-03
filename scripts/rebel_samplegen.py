"""ReBeL Phase-2 sample generation v2 — TRAJECTORY-BASED, full-coverage sampler.

The v1 sampler (new_initial_state + a scripted call-line) collapsed to ONE spot
(seat 0, flop, 6-handed, 14bb, pot 600). This v2 sampler covers the FULL reachable
Double-Up SNG state space by:

  1. `stack_sampler.sample_starting_state(structure)` → a tournament moment:
     blind level (escalation-weighted), per-seat stacks (varied/asymmetric, the
     ICM-relevant configs), dealer seat (→ all positions), alive_count
     (6/5/4-handed; 4 = the bubble — over-bubble {4,5,6} is the reachable set for
     top-3 Double-Up, since at 3 alive the survivors have cashed = terminal).
  2. Build the rotated single-hand game for that state and DEAL.
  3. Walk the hand with the blueprint, collecting every decision state, and pick
     ONE uniformly → realistic street/position/betting distribution.
  4. build_subgame_tree → evaluate_leaves → solve_subgame at the SAMPLED stacks
     (so ICM/Malmuth-Harville finish equity is correct near the bubble).
  5. Persist the v2 sample: PER-SEAT 6-vector value target (`value6`, so the net
     values any leaf regardless of whose turn it is) + belief block (`seat_buckets`,
     the 6-seat joint config → 6×k at train time) + tournament context.

Tournament-mode states don't expose dealer_seat(), and the k=200 blueprint was
trained dealer-aware, so `dealer_seat` is injected into every blueprint encode
(walk + solver warmup + leaf rollout) for correct positions.

Single box:  .venv/bin/python scripts/rebel_samplegen.py --workers 26 --target 1000000 --out /workspace/rebel_samples
Second box:  same command (host-prefixed shards merge losslessly).
"""
import sys, os, time, random, argparse, socket
import multiprocessing as mp
sys.path.insert(0, '.')

ABSTR = "runs/k200_abstraction.pkl"
CKPT = "runs/k200_blueprint_ckpt_iter_2000.pt"
STRUCT = "configs/ignition_double_up_6max_turbo.yaml"
_NUM_SEATS = 6


# Street upweight for root selection (Option-3 mild stratification): realistic
# preflop-dominant, but flop/turn/river kept above a usable floor. Tuned so the
# marginal lands ~preflop 85 / flop 8 / turn 4 / river 3.
_STREET_W = {0: 1.0, 1: 10.0, 2: 60.0, 3: 200.0}
_FOLD = 0  # DiscreteAction.FOLD


def worker(wid, target_per_worker, seconds, depth, M, n_iters, num_paid,
           out_dir, flush_every, ret):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU per worker
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
    from src.nlhe.infoset6 import parse_state_6max
    from src.nlhe.cfr6 import _build_view_6max
    from src.nlhe.actions import discretize_legal_actions, DiscreteAction
    from src.nlhe.networks6 import SCHEMA_VERSION as PN_SCHEMA
    from src.rebel.sample_io import ShardWriter, RebelSample, file_sha1
    from scripts.eval_6max_self_play import _load_solver

    structure = TournamentStructure.from_yaml(STRUCT)
    abstraction = Abstraction.load(ABSTR)
    solver = _load_solver(CKPT, abstraction, structure)
    encoder = solver.encoder
    pn = solver.policy_nets
    dev = pn.device
    bucket_dim = encoder.max_bucket_dim
    payouts = list(sng_payouts_6max_double_up())
    rng = random.Random(1000 + wid)
    # k per street (preflop=20, postflop=200) from the abstraction.
    K_STREET = {}
    for name, sd in abstraction.streets.items():
        idx = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}.get(name)
        if idx is not None:
            K_STREET[idx] = int(getattr(sd, "medoid_histograms").shape[0])
    is_v2 = (getattr(pn, "loaded_schema_version", None) == PN_SCHEMA)

    meta = {
        "blueprint_sha": file_sha1(CKPT), "abstraction_sha": file_sha1(ABSTR),
        "k_postflop": int(bucket_dim), "feat_dim": int(encoder.feature_dim),
        "n_actions": 9, "structure": structure.format_name,
        "depth": depth, "M": M, "n_iterations": n_iters, "num_paid": num_paid,
        "leaf_mode": "PROFILE_SAMPLE", "round": "bootstrap_v2",
        "sampler": "trajectory_stack_sampler", "belief": "reach_posterior",
        "street_upweight": _STREET_W,
    }
    writer = ShardWriter(root_dir=out_dir, worker=wid, flush_every=flush_every, meta=meta)

    def batched_policy(seat, public_part, mask, kk):
        """Blueprint policy over ALL kk bucket-variants at one decision: returns
        (kk, n_actions). Same masked-softmax (v2) / RM+ (v1) as inference_policy,
        batched over the bucket one-hot — for the opponent reach-belief."""
        feats = np.zeros((kk, encoder.feature_dim), dtype=np.float32)
        feats[:, bucket_dim:] = public_part
        feats[np.arange(kk), np.arange(kk)] = 1.0
        x = torch.from_numpy(feats).to(dev)
        with torch.no_grad():
            net = pn.strat_net if is_v2 else pn.nets[seat]
            net.eval()
            out = net(x).cpu().numpy()
        m = mask[None, :]
        if is_v2:
            out = out - out.max(1, keepdims=True)
            ex = np.exp(out) * m
        else:
            ex = np.maximum(out, 0.0) * m
        denom = ex.sum(1, keepdims=True)
        unif = m / max(float(mask.sum()), 1.0)
        return np.where(denom > 0, ex / np.maximum(denom, 1e-12), unif)

    def blueprint_step(s, dealer):
        """Sample one blueprint action; return (chip, record) where record carries
        the public feature, legal mask, discrete action, seat, street — the bits
        needed later to reconstruct this seat's reach-belief over buckets."""
        cp = s.current_player()
        parsed = parse_state_6max(s)
        parsed["dealer_seat"] = dealer
        feat = np.asarray(encoder.encode_from_parsed(parsed, rng=rng), dtype=np.float32)
        view = _build_view_6max(s, parsed)
        d2c = discretize_legal_actions(list(s.legal_actions()), view)
        if not d2c:
            return None, None
        mask = np.zeros(9, dtype=np.float32)
        for da in d2c:
            mask[int(da)] = 1.0
        policy = np.asarray(pn.inference_policy(cp, feat, mask), dtype=np.float64)
        idx = int(rng.choices(range(9), weights=policy.tolist(), k=1)[0])
        chip = d2c.get(DiscreteAction(idx))
        if chip is None:
            da, chip = rng.choice(list(d2c.items()))
            idx = int(da)
        rec = {"seat": cp, "public": feat[bucket_dim:].copy(), "mask": mask,
               "action": idx, "street": int(parsed["street_idx"])}
        return int(chip), rec

    def compute_belief(history_before, root_feat, hero, alive_stacks, root_street):
        """Per-seat belief (6, bucket_dim). hero = one-hot(known bucket); each active
        opponent = reach-posterior over root-street buckets from its blueprint actions
        (uniform if it has no same-street action yet); folded/busted = 0."""
        belief = np.zeros((_NUM_SEATS, bucket_dim), dtype=np.float32)
        kk = K_STREET.get(root_street, bucket_dim)
        # Hero knows its hand.
        hb = int(root_feat[:bucket_dim].argmax()) if float(root_feat[:bucket_dim].max()) > 0 else -1
        if hb >= 0:
            belief[hero, hb] = 1.0
        # Folded-before-root seats contribute no range.
        folded = {h["seat"] for h in history_before if h["action"] == _FOLD}
        for s in range(_NUM_SEATS):
            if s == hero or alive_stacks[s] <= 0 or s in folded:
                continue
            past = [h for h in history_before if h["seat"] == s and h["street"] == root_street]
            if not past:
                belief[s, :kk] = 1.0 / kk   # no same-street info → uniform prior
                continue
            logr = np.zeros(kk)
            for h in past:
                pol = batched_policy(s, h["public"], h["mask"], kk)  # (kk, 9)
                logr += np.log(pol[:, h["action"]] + 1e-9)
            logr -= logr.max()
            r = np.exp(logr); r /= max(r.sum(), 1e-12)
            belief[s, :kk] = r
        return belief

    def one_trajectory():
        """Sample a tournament moment, deal, walk the blueprint recording history;
        return (history, root_idx, sampled) with root chosen by street-weighted
        sampling over ALIVE-hero decisions — or None on a dead trajectory."""
        sampled = sample_starting_state(structure, rng, num_paid=num_paid)
        gs = structure.to_inner_game_string_for_state(
            blind_level=sampled["blind_level"], stacks=sampled["stacks"],
            dealer_seat=sampled["dealer_seat"])
        game = pyspiel.load_game(gs)
        s = game.new_initial_state()
        dealer = sampled["dealer_seat"]
        stacks = sampled["stacks"]
        history = []          # ordered decision records (with state clone)
        steps = 0
        while (not s.is_terminal()) and steps < 80:
            steps += 1
            if s.is_chance_node():
                a, p = zip(*s.chance_outcomes())
                s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
                continue
            chip, rec = blueprint_step(s, dealer)
            if rec is None:
                break
            rec["state"] = s.clone()
            rec["alive_hero"] = (rec["seat"] >= 0 and stacks[rec["seat"]] > 0)
            history.append(rec)
            s = s.child(int(chip))
        # Candidate roots = alive-hero decisions, weighted by street (Option-3).
        cand = [(i, _STREET_W.get(history[i]["street"], 1.0))
                for i in range(len(history)) if history[i]["alive_hero"]]
        if not cand:
            return None
        idxs = [c[0] for c in cand]; ws = [c[1] for c in cand]
        root_idx = rng.choices(idxs, weights=ws, k=1)[0]
        return history, root_idx, sampled

    n = 0
    t_gen_start = time.perf_counter()
    t_end = t_gen_start + seconds if seconds else float("inf")
    while n < target_per_worker and time.perf_counter() < t_end:
        try:
            traj = one_trajectory()
            if traj is None:
                continue
            history, root_idx, sampled = traj
            root_state = history[root_idx]["state"]
            stacks = list(sampled["stacks"])
            dealer = sampled["dealer_seat"]
            hero = root_state.current_player()
            if hero < 0 or stacks[hero] <= 0:
                continue

            tree = build_subgame_tree(root_state, max_action_depth=depth,
                                      chance_samples_per_node=2, rng=rng)
            lctx = LeafEvalContext(blueprint=solver, biased_blueprint=BiasedBlueprint(),
                                   starting_stacks=stacks, payouts=payouts, hero_seat=hero,
                                   mode=LeafEvalMode.PROFILE_SAMPLE, n_samples=M, rng=rng,
                                   num_paid=num_paid, dealer_seat=dealer)
            evaluate_leaves(tree, lctx)
            sctx = SubgameSolveContext(blueprint=solver, starting_stacks=stacks, payouts=payouts,
                                       hero_seat=hero, n_iterations=n_iters, rng=rng,
                                       num_paid=num_paid, average_weighting="linear",
                                       dealer_seat=dealer)
            res = solve_subgame(tree, sctx)
            if res.root_value_per_seat is None:
                continue  # K<=1 path doesn't produce the 6-vector; generator uses K>1

            # --- PBS encoding (dealer-aware) + reach-belief + per-seat target ---
            parsed = parse_state_6max(root_state)
            parsed["dealer_seat"] = dealer
            feat = np.asarray(encoder.encode_from_parsed(parsed, rng=rng), dtype=np.float32)
            onehot = feat[:bucket_dim]
            bucket = int(onehot.argmax()) if float(onehot.max()) > 0 else -1
            root_street = int(parsed["street_idx"])
            belief = compute_belief(history[:root_idx], feat, hero, stacks, root_street
                                    ).astype(np.float16)

            value6 = np.asarray(res.root_value_per_seat, dtype=np.float32)
            policy = np.asarray(res.root_policy, dtype=np.float32)
            qvals = np.asarray(res.root_q_values, dtype=np.float32)
            mask = np.asarray(res.legal_mask, dtype=np.int8)

            writer.add(RebelSample(
                feat=feat, bucket=bucket, belief=belief,
                hero_seat=int(hero), street=root_street,
                depth=int(tree_depth(tree)), n_iterations=int(res.n_iterations),
                degraded=bool(res.degraded),
                blind_level=int(sampled["blind_level"].level),
                alive_count=int(sampled["alive_count"]), dealer_seat=int(dealer),
                value6=value6, root_value=float(value6[hero]),
                root_policy=policy, root_q_values=qvals, legal_mask=mask,
            ))
            n += 1
            if n % 200 == 0:
                ret[wid] = n
        except Exception as e:
            ret['err_%d' % wid] = repr(e)[:200]

    writer.close()
    ret['gen_%d' % wid] = time.perf_counter() - t_gen_start
    ret[wid] = n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--target", type=int, default=1_000_000, help="total samples for THIS box")
    ap.add_argument("--seconds", type=int, default=0, help="optional wall cap (0 = none)")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--n-iters", type=int, default=150)
    ap.add_argument("--num-paid", type=int, default=3)
    ap.add_argument("--out", default="/workspace/rebel_samples")
    ap.add_argument("--flush-every", type=int, default=256)
    a = ap.parse_args()

    per_worker = (a.target + a.workers - 1) // a.workers
    os.makedirs(a.out, exist_ok=True)
    host = socket.gethostname()
    print(f"[samplegen v2] host={host} workers={a.workers} target={a.target} "
          f"(~{per_worker}/worker) depth={a.depth} M={a.M} K={a.n_iters} out={a.out}", flush=True)

    ctx = mp.get_context("spawn")
    ret = ctx.Manager().dict()
    procs = [ctx.Process(target=worker,
                         args=(w, per_worker, a.seconds, a.depth, a.M, a.n_iters,
                               a.num_paid, a.out, a.flush_every, ret))
             for w in range(a.workers)]
    t0 = time.perf_counter()
    for p in procs: p.start()
    for p in procs: p.join()
    wall = time.perf_counter() - t0

    total = sum(v for k, v in ret.items() if isinstance(k, int))
    gen_secs = [v for k, v in ret.items() if isinstance(k, str) and k.startswith("gen_")]
    errs = [v for k, v in ret.items() if isinstance(k, str) and not k.startswith("gen_")]
    gen_window = max(gen_secs) if gen_secs else wall
    print(f"[samplegen v2] DONE host={host} wall={wall:.1f}s gen_window={gen_window:.1f}s "
          f"total_samples={total}", flush=True)
    print(f"[samplegen v2] steady-state: {total/gen_window:.2f} samples/s = "
          f"{total/gen_window*3600:.0f} samples/hour  (wall-incl-startup: {total/wall*3600:.0f}/hr)", flush=True)
    if errs:
        print("[samplegen v2] errors (sample):", errs[:3], flush=True)
