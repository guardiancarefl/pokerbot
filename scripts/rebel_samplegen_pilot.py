import sys, os, time, random, argparse
import multiprocessing as mp
sys.path.insert(0,'.')
ABSTR="runs/k200_abstraction.pkl"; CKPT="runs/k200_blueprint_ckpt_iter_2000.pt"; STRUCT="configs/ignition_double_up_6max_turbo.yaml"

def worker(wid, seconds, depth, M, ret):
    os.environ["CUDA_VISIBLE_DEVICES"]=""   # CPU per worker (avoid GPU serialization)
    import torch; torch.set_num_threads(1)
    import pyspiel
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure, six_max_sng
    from src.nlhe.biased_policy import BiasedBlueprint
    from src.nlhe.icm import sng_payouts_6max_double_up
    from src.nlhe.subgame import build_subgame_tree
    from src.nlhe.subgame_leaf import LeafEvalContext, LeafEvalMode, evaluate_leaves
    from src.nlhe.subgame_solver import SubgameSolveContext, solve_subgame
    from src.nlhe.infoset6 import InfosetEncoder6Max
    from scripts.eval_6max_self_play import _load_solver
    solver=_load_solver(CKPT, Abstraction.load(ABSTR), TournamentStructure.from_yaml(STRUCT))
    stack=int(solver.encoder.starting_stack)
    game=pyspiel.load_game(six_max_sng(starting_stack=stack))
    payouts=list(sng_payouts_6max_double_up()); rng=random.Random(1000+wid)
    def a_decision():
        s=game.new_initial_state()
        while s.is_chance_node():
            a,p=zip(*s.chance_outcomes()); s=s.child(int(rng.choices(a,weights=p,k=1)[0]))
        g=0
        while not s.is_terminal() and not s.child(1).is_chance_node():
            s=s.child(1); g+=1
            if g>=18: break
        if s.is_terminal(): return None
        s=s.child(1)
        while s.is_chance_node():
            a,p=zip(*s.chance_outcomes()); s=s.child(int(rng.choices(a,weights=p,k=1)[0]))
        return s if not s.is_terminal() else None
    n=0; t_end=time.perf_counter()+seconds
    while time.perf_counter()<t_end:
        s=a_decision()
        if s is None: continue
        try:
            tree=build_subgame_tree(s, max_action_depth=depth, chance_samples_per_node=2, rng=rng)
            lctx=LeafEvalContext(blueprint=solver, biased_blueprint=BiasedBlueprint(), starting_stacks=[stack]*6,
                payouts=payouts, hero_seat=tree.root.current_player, mode=LeafEvalMode.PROFILE_SAMPLE, n_samples=M, rng=rng)
            evaluate_leaves(tree,lctx)
            sctx=SubgameSolveContext(blueprint=solver, starting_stacks=[stack]*6, payouts=payouts,
                hero_seat=tree.root.current_player, n_iterations=150, rng=rng, num_paid=3, average_weighting="linear")
            solve_subgame(tree,sctx)
            n+=1
        except Exception as e:
            ret['err_%d'%wid]=repr(e)[:120]
    ret[wid]=n

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=13); ap.add_argument("--seconds",type=int,default=60)
    ap.add_argument("--depth",type=int,default=3); ap.add_argument("--M",type=int,default=1); a=ap.parse_args()
    ctx=mp.get_context("spawn"); ret=ctx.Manager().dict()
    procs=[ctx.Process(target=worker,args=(w,a.seconds,a.depth,a.M,ret)) for w in range(a.workers)]
    t0=time.perf_counter()
    for p in procs: p.start()
    for p in procs: p.join()
    wall=time.perf_counter()-t0
    total=sum(v for k,v in ret.items() if isinstance(k,int))
    errs=[v for k,v in ret.items() if isinstance(k,str)]
    print(f"workers={a.workers} depth={a.depth} M={a.M} PROFILE_SAMPLE")
    print(f"wall={wall:.1f}s  total_samples={total}  => {total/wall:.2f} samples/s  = {total/wall*3600:.0f} samples/hour")
    if errs: print("errors (sample):", errs[:2])
