"""Resolver gate-widening experiment: blueprint+resolver vs the legit Shanky pool.

Sweep contract (carried by run_resolver_sweep.sh — committed alongside this file):
  hands per shard: 600
  default leaf mode: profile (LeafEvalMode.PROFILE_SAMPLE) — see --leaf-mode to switch
  default max_action_depth: 3 — see --max-action-depth to override

  Conditions (gate kwargs threaded into SubgamePolicy ctor):
    A_defaults     max_blueprint_prob=0.95   min_legal_actions=3
    B_moderate     max_blueprint_prob=0.99   min_legal_actions=2
    C_aggressive   max_blueprint_prob=0.995  min_legal_actions=2

  CRN seed map (seed = base + i*1000 where i is the profile's index in the sweep
  driver's PROFILES tuple — NOT the post-filter shanky list. The sweep driver
  passes the right seed via --seed when --only is one profile, which is the
  per-shard invocation form):
    killphilmtt          base + 0*1000  -> 2026
    ticketmaster         base + 1*1000  -> 3026
    thefixersng          base + 2*1000  -> 4026
    minestackermttv.7.3  base + 3*1000  -> 5026
    gushansenmtt         base + 4*1000  -> 6026

  No-flag invocation post-shim is byte-identical to the original pre-shim hardcode:
  --leaf-mode profile + --max-action-depth 3 reproduces SubgamePolicy(leaf_mode=
  PROFILE_SAMPLE) with the SubgamePolicy default max_action_depth=3.

Built for the "gate-widening" experiment (resolver_experiment_results.md). Unlike
scripts.eval_shanky_vs_dcfr (which pits a PLAIN CheckpointPolicy challenger against
the Shanky pool), this harness uses a SubgamePolicy *resolver* as the challenger and
sweeps the gate threshold via solve_kw -- there is no YAML key for the gate, the
thresholds are SubgamePolicy ctor kwargs (min_legal_actions, max_blueprint_prob).

For every (gate condition x Shanky profile) matchup it records, in addition to the
ICM-equity-delta that eval_pool.evaluate_matchup already returns:
  - actual gate fire-rate f (overall + per-street), measured on REAL matchup traffic
  - per-decision action time: mean, p95, and fraction exceeding the 15s budget
  - solve-only action time (the >0 tail), mean/p95/max
Results are appended to a JSONL one line per (condition, profile) so a long run can
be interrupted without losing completed matchups.

The challenger seat decisions are intercepted by a thin timing wrapper around
SubgamePolicy.select_action; SubgamePolicy itself is unmodified. The wrapper reads
the gate outcome from the n_gated_solve counter delta (the gate increments it iff it
decided to SOLVE this decision) and the street from parsed["street_idx"].

Resolver leaf mode is PROFILE_SAMPLE -- the production choice per the eval_pool_ablation
verdict ("use PROFILE for production"). Solver params (n_samples/n_iterations/depth)
are left at SubgamePolicy defaults, where the ~6.7s/solve cost model was measured.

Usage (smoke -- 1 condition, few hands, 2 profiles):
    python -m scripts.eval_resolver_vs_shanky \\
        --ckpt runs/six_max_20260530_034023_phase4f_dcfr_candC_k200/checkpoints/ckpt_iter_2000.pt \\
        --abstraction runs/abstraction_20260521_223018_retrofit/abstraction.pkl \\
        --structure configs/ignition_double_up_6max_turbo.yaml \\
        --shanky-dir data/shanky_profiles \\
        --conditions A --only killphilmtt timidtom --hands 50 \\
        --out evals/resolver_smoke.jsonl

Usage (full sweep -- 3 conditions x legit pool):
    python -m scripts.eval_resolver_vs_shanky \\
        --ckpt .../ckpt_iter_2000.pt --abstraction .../abstraction.pkl \\
        --structure configs/ignition_double_up_6max_turbo.yaml \\
        --shanky-dir data/shanky_profiles --hands 800 \\
        --out evals/resolver_sweep.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict

import numpy as np

logging.basicConfig(
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger("resolver_vs_shanky")

_STREET = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
_BUDGET_S = 15.0

# Gate conditions (the one variable in this experiment).
CONDITIONS = {
    "A": {"label": "A_defaults",   "max_blueprint_prob": 0.95,  "min_legal_actions": 3},
    "B": {"label": "B_moderate",   "max_blueprint_prob": 0.99,  "min_legal_actions": 2},
    "C": {"label": "C_aggressive", "max_blueprint_prob": 0.995, "min_legal_actions": 2},
}

# Profiles excluded from the "legit" pool (non-6max / degenerate styles).
DEFAULT_EXCLUDE = ("fixedlimitheadsup", "modernmikecash", "lucky1_6max")


class TimedResolver:
    """Thin wrapper around SubgamePolicy that times each challenger decision and
    tallies per-street gate fires, without modifying SubgamePolicy. Buffers are
    reset per matchup via reset()."""

    def __init__(self, sg):
        self.sg = sg
        self.reset()

    @property
    def name(self):
        return self.sg.name

    def reset(self):
        self.times = []            # wall time of every challenger decision (s)
        self.solve_times = []      # wall time of decisions that SOLVED (s)
        self.street_tot = defaultdict(int)
        self.street_solve = defaultdict(int)

    def select_action(self, parsed, state, rng, mode: str = "sample") -> int:
        street = int(parsed.get("street_idx", 0))
        before = self.sg.n_gated_solve
        t0 = time.perf_counter()
        a = self.sg.select_action(parsed, state, rng, mode=mode)
        dt = time.perf_counter() - t0
        solved = (self.sg.n_gated_solve - before) == 1
        self.times.append(dt)
        self.street_tot[street] += 1
        if solved:
            self.street_solve[street] += 1
            self.solve_times.append(dt)
        return a

    def timing_summary(self) -> dict:
        t = np.asarray(self.times, dtype=float)
        st = np.asarray(self.solve_times, dtype=float)
        n = int(t.size)
        n_solve = int(sum(self.street_solve.values()))
        per_street = {}
        for s_idx in (0, 1, 2, 3):
            tot = self.street_tot.get(s_idx, 0)
            sol = self.street_solve.get(s_idx, 0)
            per_street[_STREET[s_idx]] = {
                "solve": sol, "total": tot,
                "rate": (sol / tot) if tot else 0.0,
            }
        return {
            "n_decisions": n,
            "n_solve": n_solve,
            "f_overall": (n_solve / n) if n else 0.0,
            "per_street": per_street,
            "action_time_mean": float(t.mean()) if n else 0.0,
            "action_time_p95": float(np.percentile(t, 95)) if n else 0.0,
            "action_time_max": float(t.max()) if n else 0.0,
            "frac_over_15s": float((t > _BUDGET_S).mean()) if n else 0.0,
            "n_over_15s": int((t > _BUDGET_S).sum()) if n else 0,
            "solve_time_mean": float(st.mean()) if st.size else 0.0,
            "solve_time_p95": float(np.percentile(st, 95)) if st.size else 0.0,
            "solve_time_max": float(st.max()) if st.size else 0.0,
        }


_LEAF_MODE_MAP = {
    "profile": "PROFILE_SAMPLE",
    "br":      "BEST_RESPONSE",
}


def _build_resolver(name, ckpt, abstraction, structure, gate_kw,
                    leaf_mode: str = "profile", max_action_depth: int = 3,
                    n_samples: int = 8):
    from src.nlhe.subgame_policy import SubgamePolicy
    from src.nlhe.subgame_leaf import LeafEvalMode
    lm_enum = getattr(LeafEvalMode, _LEAF_MODE_MAP[leaf_mode])
    return SubgamePolicy(name, ckpt, abstraction, structure,
                         leaf_mode=lm_enum, n_samples=n_samples,
                         max_action_depth=max_action_depth, **gate_kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="blueprint .pt (fixed across conditions)")
    ap.add_argument("--abstraction", required=True)
    ap.add_argument("--structure", required=True)
    ap.add_argument("--shanky-dir", required=True)
    ap.add_argument("--conditions", nargs="*", default=["A", "B", "C"],
                    choices=list(CONDITIONS), help="gate conditions to run (default all)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these Shanky profiles (normalized stems); for smoke")
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                    help="profiles to drop from the legit pool")
    ap.add_argument("--hands", type=int, default=800, help="hands per matchup")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mode", choices=("sample", "argmax"), default="sample")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--leaf-mode", choices=("profile", "br"), default="profile",
                    help="subgame leaf evaluation mode "
                         "(profile=LeafEvalMode.PROFILE_SAMPLE [default, pre-shim "
                         "hardcode], br=LeafEvalMode.BEST_RESPONSE)")
    ap.add_argument("--max-action-depth", type=int, default=3,
                    help="SubgamePolicy max_action_depth (default 3 = SubgamePolicy "
                         "ctor default, pre-shim behavior)")
    ap.add_argument("--n-samples", type=int, default=8,
                    help="LeafEvalContext n_samples (M); SubgamePolicy ctor default 8. "
                         "X6 oracle-leaf probe uses 32 (4x production precision).")
    ap.add_argument("--no-icm-short-circuit", action="store_true",
                    help="Disable the leaf-eval ITM Option-A short-circuit so EVERY "
                         "leaf rolls out (oracle-leaf precision). Implemented by "
                         "monkeypatching subgame_leaf.is_itm -> False for this run "
                         "only (same pattern as --players-remaining); src untouched. "
                         "is_itm is used ONLY in the two short-circuit guards "
                         "(subgame_leaf.py:564,623), so this is exact and side-effect-free.")
    ap.add_argument("--players-remaining", type=int, default=None,
                    help="bubble-slice ablation: force alive_count to this value "
                         "(typical 4 for the 3-paid bubble). Default None = "
                         "byte-identical sampling. Monkey-patches "
                         "stack_sampler.sample_starting_state for this run only.")
    ap.add_argument("--no-resolver", action="store_true",
                    help="Use plain CheckpointPolicy (blueprint-alone) instead of "
                         "SubgamePolicy as the challenger. Skips gate / SubgamePolicy "
                         "construction entirely. Used for the bubble-vs-bubble baseline "
                         "(--players-remaining 4 --no-resolver). gate / leaf-mode / "
                         "depth / solver timing fields will be zero in the JSONL.")
    ap.add_argument("--per-hand", action="store_true",
                    help="Force per-hand record logging (eff_stack_bb, blind_level, "
                         "diff). Auto-on when --players-remaining is set; this flag "
                         "lets natural runs capture the regime split too.")
    ap.add_argument("--out", required=True, help="output JSONL path (appended)")
    args = ap.parse_args()

    from scripts.eval_pool import evaluate_matchup
    from scripts.eval_shanky_vs_dcfr import _build_shanky_policies
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.stack_sampler import TournamentStructure

    # Bubble-slice override. Monkey-patches sample_starting_state for this run
    # only — chosen over threading a force_alive kwarg through eval_pool because
    # it keeps eval_pool unchanged. With --players-remaining absent, both module
    # bindings retain their import-time targets -> byte-identical sampling.
    if args.players_remaining is not None:
        from src.nlhe import stack_sampler as _ss
        from scripts import eval_pool as _ep
        _orig_sampler = _ss.sample_starting_state
        def _forced_sampler(structure, rng, num_paid=3):
            return _orig_sampler(structure, rng, num_paid=num_paid,
                                 force_alive=args.players_remaining)
        _ss.sample_starting_state = _forced_sampler
        _ep.sample_starting_state = _forced_sampler  # eval_pool imports by name
        log.info(f"BUBBLE SLICE: forcing alive_count = {args.players_remaining}")

    # Oracle-leaf precision: disable the ITM short-circuit so every leaf rolls out.
    # Monkeypatch subgame_leaf.is_itm -> False (it is consumed ONLY in the two
    # short-circuit guards, subgame_leaf.py:564,623; terminal ICM uses
    # icm_adjust_returns, not is_itm), so this exactly realizes
    # LeafEvalContext.icm_short_circuit=False without touching src.
    if args.no_icm_short_circuit:
        from src.nlhe import subgame_leaf as _sl
        _sl.is_itm = lambda *a, **k: False
        log.info("ORACLE-LEAF: ITM short-circuit DISABLED (is_itm patched -> False); "
                 "every leaf rolls out")

    log.info(f"loading abstraction {args.abstraction}")
    abstr = Abstraction.load(args.abstraction)
    structure = TournamentStructure.from_yaml(args.structure)

    log.info(f"loading Shanky profiles from {args.shanky_dir}")
    shanky = _build_shanky_policies(args.shanky_dir, only=args.only)
    exclude = {e.lower() for e in args.exclude}
    shanky = [p for p in shanky if p.name not in exclude]
    if not shanky:
        log.error("no Shanky profiles after exclude/only filter; aborting")
        raise SystemExit(1)
    log.info(f"legit pool ({len(shanky)}): {sorted(p.name for p in shanky)}")
    log.info(f"excluded: {sorted(exclude)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_matchups = len(args.conditions) * len(shanky)
    log.info(f"running {n_matchups} matchups "
             f"({len(args.conditions)} conditions x {len(shanky)} profiles), "
             f"{args.hands} hands each, mode={args.mode}")

    done = 0
    t_start = time.time()
    with open(args.out, "a") as fout:
        for cond_key in args.conditions:
            cond = CONDITIONS[cond_key]
            gate_kw = {"max_blueprint_prob": cond["max_blueprint_prob"],
                       "min_legal_actions": cond["min_legal_actions"]}
            log.info("=" * 70)
            log.info(f"CONDITION {cond_key} ({cond['label']}): {gate_kw}")
            if args.no_resolver:
                # Blueprint-alone challenger (bubble-vs-bubble baseline). Plain
                # CheckpointPolicy — no SubgamePolicy, no gate. TimedResolver's
                # n_gated_solve probe is absent here, so it tallies zero solves
                # (timing_summary returns f=0, all solve_time fields zero).
                from scripts.eval_pool import CheckpointPolicy
                resolver = CheckpointPolicy(f"blueprint-{cond_key}", args.ckpt,
                                            abstr, structure)
                # Adapter: CheckpointPolicy doesn't have n_gated_solve. Stub.
                resolver.n_gated_solve = 0
                log.info(f"  --no-resolver: using CheckpointPolicy (blueprint-alone)")
            else:
                # Fresh resolver per condition (reloads the same blueprint with new gate).
                resolver = _build_resolver(f"resolver-{cond_key}", args.ckpt,
                                           abstr, structure, gate_kw,
                                           leaf_mode=args.leaf_mode,
                                           max_action_depth=args.max_action_depth,
                                           n_samples=args.n_samples)
            wrapped = TimedResolver(resolver)

            for i, opp in enumerate(shanky):
                wrapped.reset()
                seed = args.seed + i * 1000
                t0 = time.perf_counter()
                log.info("-" * 60)
                log.info(f"[{done + 1}/{n_matchups}] cond={cond_key} "
                         f"resolver vs shanky:{opp.name}  (seed={seed})")
                # Per-hand eff_stack logged when running a bubble slice (the regime-split
                # consumer) OR when --per-hand is explicitly set. For all other invocations,
                # record_per_hand=False -> identical evaluate_matchup return shape.
                record_per_hand = args.per_hand or (args.players_remaining is not None)
                try:
                    r = evaluate_matchup(
                        challenger=wrapped, opponent=opp, structure=structure,
                        hands=args.hands, seed=seed, mode=args.mode,
                        log_every=args.log_every,
                        record_per_hand=record_per_hand,
                    )
                except Exception as e:
                    log.exception(f"  matchup errored: {type(e).__name__}: {e}")
                    done += 1
                    continue
                wall = time.perf_counter() - t0
                tsum = wrapped.timing_summary()
                rec = {
                    "condition": cond_key,
                    "condition_label": cond["label"],
                    "challenger": ("blueprint" if args.no_resolver else "subgame"),
                    "gate": (None if args.no_resolver else gate_kw),
                    "leaf_mode": (None if args.no_resolver else args.leaf_mode),
                    "max_action_depth": (None if args.no_resolver else args.max_action_depth),
                    "n_samples": (None if args.no_resolver else args.n_samples),
                    "icm_short_circuit": (None if args.no_resolver
                                          else (not args.no_icm_short_circuit)),
                    "players_remaining": args.players_remaining,
                    "opponent": f"shanky:{opp.name}",
                    "hands": args.hands,
                    "seed": seed,
                    "icm_diff": r["diff"],
                    "stderr": r["stderr"],
                    "sigma": r["sigma"],
                    "n_capped": r["n_capped"],
                    "avg_challenger": r["avg_challenger"],
                    "avg_opponent": r["avg_opponent"],
                    "matchup_wall_s": wall,
                    "per_hand": r.get("per_hand"),  # None unless record_per_hand
                    **tsum,
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                done += 1
                log.info(
                    f"  -> diff={r['diff']:+.4f} +/-{r['stderr']:.4f} "
                    f"(sigma={r['sigma']:.2f}) | f={tsum['f_overall']:.3f} "
                    f"| act mean={tsum['action_time_mean']:.2f}s "
                    f"p95={tsum['action_time_p95']:.2f}s "
                    f">15s={tsum['frac_over_15s']*100:.1f}% "
                    f"| solve p95={tsum['solve_time_p95']:.2f}s "
                    f"| wall={wall/60:.1f}m"
                )

    log.info("=" * 70)
    log.info(f"DONE: {done} matchups in {(time.time() - t_start)/60:.1f} min "
             f"-> {args.out}")


if __name__ == "__main__":
    main()
