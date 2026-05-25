"""Parallel pool-ablation harness for B1c sub-step 6 (Level-3 strength go/no-go).

Wraps `eval_pool`'s per-hand logic with (a) deterministic per-hand SHA256 seeding,
(b) hand-level multiprocessing, and (c) CRN pairing across challengers — so the
cross-challenger LIFT (subgame-BR − blueprint, etc.) is a paired, low-variance
comparison. Full design + the pre-committed verdict: `docs/SUBSTEP_6_DESIGN.md`.

CRN via split RNG (the variance reducer): each hand seed drives a `chance_rng` (deal,
board cards, seat assignment) that is SHARED across all challengers for that hand —
so the starting conditions are identical — and a separate `policy_rng` for action
selection (which diverges once challengers act differently; play divergence is
expected, the starting conditions are what we pair on).

Determinism: a hand's seed is `SHA256(base_seed:opp_idx:hand_idx) mod 2**31`, computed
from indices only — so the aggregate is bit-identical regardless of worker count and
of execution order. Worker exceptions FAIL THE RUN (no silent shard drop, which would
bias the aggregate).

STATUS: STAGE 6-A — parallel wrapper + per-hand seeding + CRN + reduce + aggregation
(per-matchup diff/stderr/sigma + the CRN-paired lift). The four-branch verdict() lands
in Stage 6-B; the real run is Stage 6-D.
"""
from __future__ import annotations

import hashlib
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import pyspiel

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.stack_sampler import sample_starting_state
from src.nlhe.cfr6 import NUM_SEATS_6MAX
from src.nlhe.infoset6 import parse_state_6max, parse_state_repeated_6max
from src.nlhe.icm_returns import icm_adjust_returns

_MAX_STEPS = 500


# ============================================================
# Per-hand seeding (deterministic, order-independent)
# ============================================================

def hand_seed(base_seed: int, opp_idx: int, hand_idx: int) -> int:
    """Deterministic per-hand seed in [0, 2**31). Index-derived -> order-independent."""
    h = hashlib.sha256(f"{base_seed}:{opp_idx}:{hand_idx}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31)


# ============================================================
# Policy specs (picklable; policies constructed inside workers)
# ============================================================

@dataclass
class PolicySpec:
    """A picklable description of a policy; `build()` constructs it inside a worker
    (torch models can't cross the process boundary)."""
    name: str
    kind: str                      # "checkpoint" | "subgame_br" | "subgame_profile" | "random"
    ckpt: Optional[str] = None
    solve_kw: dict = field(default_factory=dict)  # SubgamePolicy overrides (e.g. small smoke params)

    def build(self, abstraction, structure):
        if self.kind == "random":
            from scripts.eval_pool import UniformRandomPolicy
            return UniformRandomPolicy(self.name)
        if self.kind == "checkpoint":
            from scripts.eval_pool import CheckpointPolicy
            return CheckpointPolicy(self.name, self.ckpt, abstraction, structure)
        if self.kind in ("subgame_br", "subgame_profile"):
            from src.nlhe.subgame_policy import SubgamePolicy
            from src.nlhe.subgame_leaf import LeafEvalMode
            mode = (LeafEvalMode.BEST_RESPONSE if self.kind == "subgame_br"
                    else LeafEvalMode.PROFILE_SAMPLE)
            return SubgamePolicy(self.name, self.ckpt, abstraction, structure,
                                 leaf_mode=mode, **self.solve_kw)
        raise ValueError(f"unknown PolicySpec kind: {self.kind!r}")


# ============================================================
# One hand, split RNG (CRN-paired starting conditions)
# ============================================================

def _play_one_hand(challenger, opponent, structure, seed: int, mode: str,
                   num_paid: int = 3) -> dict:
    """One hand with the challenger in the 'A' seats and the opponent in 'B'.

    `chance_rng = Random(seed)` drives the deal, board cards and seat assignment (so
    these are IDENTICAL across challengers for the same seed — the CRN pairing);
    `policy_rng` drives action selection. Returns the per-seat ICM-equity-delta plus
    the seat assignment and a cap flag (mirrors eval_pool.play_one_hand_two_policies).
    """
    chance_rng = random.Random(seed)
    policy_rng = random.Random((seed ^ 0x5DEECE66D) & 0x7FFFFFFF)

    sampled = sample_starting_state(structure, chance_rng, num_paid=num_paid)
    gs = structure.to_inner_game_string_for_state(
        blind_level=sampled["blind_level"], stacks=sampled["stacks"],
        dealer_seat=sampled["dealer_seat"])
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    starting_stacks = list(sampled["stacks"])

    seat_assignment = [chance_rng.choice(["A", "B"]) for _ in range(NUM_SEATS_6MAX)]
    seat_to_policy = [challenger if c == "A" else opponent for c in seat_assignment]

    for _ in range(_MAX_STEPS):
        if state.is_terminal():
            break
        if state.is_chance_node():
            outs = state.chance_outcomes()
            a = chance_rng.choices([o[0] for o in outs],
                                   weights=[o[1] for o in outs], k=1)[0]
            state.apply_action(int(a))
            continue
        parsed = (parse_state_repeated_6max(state)
                  if hasattr(state, "dealer_seat") else parse_state_6max(state))
        cp = parsed["current_player"]
        state.apply_action(int(seat_to_policy[cp].select_action(
            parsed, state, policy_rng, mode=mode)))

    if not state.is_terminal():
        return {"seat_assignment": seat_assignment,
                "equity": [0.0] * NUM_SEATS_6MAX, "exceeded_cap": True}
    equity = icm_adjust_returns(chip_returns=list(state.returns()),
                                starting_stacks=starting_stacks,
                                payouts=[2.0] * num_paid)
    return {"seat_assignment": seat_assignment, "equity": list(equity),
            "exceeded_cap": False}


def _contribution(result) -> Optional[float]:
    """Per-seat (challenger − opponent) ICM-equity-delta for one hand; None if capped
    or a side has no seats. Matches eval_pool's per-hand diff contribution."""
    if result["exceeded_cap"]:
        return None
    sa, eq = result["seat_assignment"], result["equity"]
    n_a = sa.count("A")
    n_b = sa.count("B")
    if n_a == 0 or n_b == 0:
        return None
    a = sum(eq[s] for s in range(NUM_SEATS_6MAX) if sa[s] == "A") / n_a
    b = sum(eq[s] for s in range(NUM_SEATS_6MAX) if sa[s] == "B") / n_b
    return a - b


# ============================================================
# Worker: process a shard of (opp_idx, hand_idx), play ALL challengers per hand
# ============================================================

def _worker(args) -> dict:
    """Run one shard. Constructs its own policies (per-process), plays every (opp,
    hand) in the shard with each challenger (CRN: same hand seed), and returns the
    per-hand challenger contributions + per-challenger SubgamePolicy.stats() counters.
    A raised exception propagates to the parent -> the run fails (no silent drop)."""
    (shard, challenger_specs, opponent_specs, abstr_path, struct_path,
     base_seed, mode, num_paid) = args
    # Abstraction is only needed by checkpoint/subgame policies; random policies don't
    # use it, so load lazily (lets abstraction-free runs pass abstr_path=None).
    abstraction = Abstraction.load(abstr_path) if abstr_path is not None else None
    structure = TournamentStructure.from_yaml(struct_path)
    challengers = {s.name: s.build(abstraction, structure) for s in challenger_specs}
    opponents = {i: spec.build(abstraction, structure)
                 for i, spec in enumerate(opponent_specs)}

    # records[opp_idx][hand_idx] = {challenger_name: contribution_or_None}
    records: dict = {}
    for (opp_idx, hand_idx) in shard:
        seed = hand_seed(base_seed, opp_idx, hand_idx)
        per_hand = {}
        for cname, cpol in challengers.items():
            res = _play_one_hand(cpol, opponents[opp_idx], structure, seed, mode, num_paid)
            per_hand[cname] = _contribution(res)
        records.setdefault(opp_idx, {})[hand_idx] = per_hand

    stats = {cname: (pol.stats() if hasattr(pol, "stats") else None)
             for cname, pol in challengers.items()}
    return {"records": records, "stats": stats}


def _merge_stats(a, b):
    if a is None:
        return dict(b) if b is not None else None
    if b is None:
        return a
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)
            if not k.endswith("_rate")}


# ============================================================
# Aggregation: per-matchup diff/stderr/sigma + CRN-paired lift
# ============================================================

def _mean_stderr_sigma(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0, float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    sig = (abs(m) / se) if se > 0 else float("nan")
    return m, se, n, sig


def aggregate(records: dict, challenger_names: list, opponent_names: list,
              lift_pairs=None) -> dict:
    """records[opp_idx][hand_idx] = {challenger: contribution|None}.

    Per (challenger, opponent): diff/stderr/sigma over non-None contributions.
    Per lift pair (A_minus_B): CRN-paired per-hand (contrib_A − contrib_B) pooled over
    all opponents (only hands where BOTH are non-None), with paired stderr/sigma.
    """
    out = {"per_matchup": {}, "lifts": {}}
    # Iterate hands in sorted hand_idx order so summation is canonical -> the aggregate
    # is bit-identical regardless of worker count / shard order (not just value-equal).
    for ci, cname in enumerate(challenger_names):
        for oi, oname in enumerate(opponent_names):
            opp_recs = records.get(oi, {})
            xs = [opp_recs[h][cname] for h in sorted(opp_recs)
                  if opp_recs[h].get(cname) is not None]
            m, se, n, sig = _mean_stderr_sigma(xs)
            out["per_matchup"][f"{cname}|{oname}"] = {
                "challenger": cname, "opponent": oname, "n_hands": n,
                "diff": m, "stderr": se, "sigma": sig}

    lift_pairs = lift_pairs or []
    for (a, b) in lift_pairs:
        paired = []
        per_opp_mean = []
        for oi, oname in enumerate(opponent_names):
            opp_recs = records.get(oi, {})
            opp_paired = [opp_recs[h][a] - opp_recs[h][b] for h in sorted(opp_recs)
                          if opp_recs[h].get(a) is not None and opp_recs[h].get(b) is not None]
            paired.extend(opp_paired)
            mo, _, _, _ = _mean_stderr_sigma(opp_paired)
            per_opp_mean.append((oname, mo))
        m, se, n, sig = _mean_stderr_sigma(paired)
        out["lifts"][f"{a}_minus_{b}"] = {
            "lift": m, "stderr": se, "sigma": sig, "n_hands": n,
            "per_opponent": [{"opponent": o, "lift": v} for o, v in per_opp_mean],
            "n_opponents_positive": sum(1 for _, v in per_opp_mean
                                        if isinstance(v, float) and v > 0)}
    return out


# ============================================================
# Driver
# ============================================================

def run_ablation(challenger_specs, opponent_specs, abstraction_path, structure_path,
                 hands: int, base_seed: int = 2026, workers: Optional[int] = None,
                 mode: str = "sample", num_paid: int = 3, lift_pairs=None) -> dict:
    """Parallel ablation. Returns the aggregate dict (per_matchup + lifts + stats)."""
    import multiprocessing
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 2)

    units = [(oi, h) for oi in range(len(opponent_specs)) for h in range(hands)]
    workers = min(workers, max(1, len(units)))
    # stride-shard so each worker spans the index space evenly (order-independent)
    shards = [units[w::workers] for w in range(workers)]
    shards = [s for s in shards if s]

    base_args = (challenger_specs, opponent_specs, abstraction_path, structure_path,
                 base_seed, mode, num_paid)
    args_list = [(shard, *base_args) for shard in shards]

    merged_records: dict = {}
    merged_stats: dict = {}
    if workers == 1:
        results = [_worker(args_list[0])]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_worker, args_list))  # exception -> propagates (fail-loud)

    for part in results:
        for oi, hands_map in part["records"].items():
            merged_records.setdefault(oi, {}).update(hands_map)
        for cname, st in part["stats"].items():
            merged_stats[cname] = _merge_stats(merged_stats.get(cname), st)

    cnames = [s.name for s in challenger_specs]
    onames = [s.name for s in opponent_specs]
    agg = aggregate(merged_records, cnames, onames, lift_pairs=lift_pairs)
    # recompute rates on the merged counters
    for cname, st in merged_stats.items():
        if st and st.get("n_decisions_total"):
            n = st["n_decisions_total"]
            st["gate_skip_rate"] = st.get("n_gated_skip", 0) / n
            st["gate_solve_rate"] = st.get("n_gated_solve", 0) / n
            st["degraded_rate"] = st.get("n_degraded", 0) / max(1, st.get("n_gated_solve", 0))
    agg["stats"] = merged_stats
    agg["config"] = {"hands": hands, "base_seed": base_seed, "workers": workers,
                     "mode": mode, "n_opponents": len(opponent_specs)}
    return agg


# ============================================================
# Verdict (Stage 6-B) — the LOCKED four-branch + revert gate (docs/SUBSTEP_6_DESIGN.md)
# ============================================================

# Locked thresholds (ICM-equity-delta + sigma). Do not retune to results.
PASS_L, PASS_SIGMA = 0.005, 2.0
SUBSTANTIVE_L, SUBSTANTIVE_SIGMA = 0.002, 1.5
BRVSP_SIGMA = 1.5            # BR must be statistically distinguishable from PROFILE
SUBSTANTIVE_OPP_FRACTION = 0.8   # >= 4/5 opponents positive


def compute_verdict(L, sigma_L, ordering_ok, n_opp_positive, n_opp,
                    L_brvsp, sigma_brvsp) -> dict:
    """The pre-committed four-branch verdict + revert gate. Returns
    {status, recommendation, ...inputs}. `status` is one of:
    PASS / SUBSTANTIVE_PASS / PASS_BR_EQUIVALENT_TO_PROFILE / AMBIGUOUS / FAIL."""
    import math as _m
    need_pos = _m.ceil(SUBSTANTIVE_OPP_FRACTION * n_opp)
    br_distinguishable = (sigma_brvsp >= BRVSP_SIGMA and L_brvsp > 0)
    profile_sig_better = (sigma_brvsp >= BRVSP_SIGMA and L_brvsp < 0)

    def pack(status, rec):
        return {"status": status, "recommendation": rec,
                "L_br_vs_blueprint": L, "sigma_L": sigma_L, "ordering_ok": ordering_ok,
                "n_opponents_positive": n_opp_positive, "n_opponents": n_opp,
                "L_br_vs_profile": L_brvsp, "sigma_br_vs_profile": sigma_brvsp,
                "thresholds": {"PASS_L": PASS_L, "PASS_SIGMA": PASS_SIGMA,
                               "SUBSTANTIVE_L": SUBSTANTIVE_L,
                               "SUBSTANTIVE_SIGMA": SUBSTANTIVE_SIGMA,
                               "BRVSP_SIGMA": BRVSP_SIGMA}}

    # FAIL: no lift, or significantly the wrong way.
    if L <= 0 or (sigma_L >= PASS_SIGMA and L < 0):
        return pack("FAIL", "subgame solving does not lift strength as measured; "
                            "do not lock — surface for diagnosis (Path A)")

    meets_strict = (L >= PASS_L and sigma_L >= PASS_SIGMA and ordering_ok)
    meets_subst = (L >= SUBSTANTIVE_L and sigma_L >= SUBSTANTIVE_SIGMA
                   and n_opp_positive >= need_pos)

    if meets_strict or meets_subst:
        base = "PASS" if meets_strict else "SUBSTANTIVE_PASS"
        if br_distinguishable:
            return pack(base, "lock BR (subgame-BR is the production form)")
        rec = ("use PROFILE for production: PROFILE significantly beats BR — BR's v×k "
               "cost is counterproductive" if profile_sig_better else
               "use PROFILE for production: BR is NOT statistically distinguishable "
               "from PROFILE at this sample size, so BR's v×k complexity is not "
               "justified by the data (not a confident 'PROFILE is better')")
        return pack("PASS_BR_EQUIVALENT_TO_PROFILE", rec)

    # Positive but undermeasured / sub-threshold.
    return pack("AMBIGUOUS", "directionally positive but undermeasured; re-run with "
                             "more hands (variance is tight, 2-4x is feasible) before "
                             "deciding")


# ============================================================
# 3-challenger orchestration + JSON output + CLI (Stage 6-B)
# ============================================================

BLUEPRINT, SG_PROFILE, SG_BR = "blueprint", "sg-profile", "sg-br"
_LIFT_PAIRS = [(SG_BR, BLUEPRINT), (SG_PROFILE, BLUEPRINT), (SG_BR, SG_PROFILE)]


def _file_md5(path):
    import hashlib as _h
    h = _h.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run_three_way(blueprint_ckpt, opponent_specs, abstraction_path, structure_path,
                  hands, base_seed=2026, workers=None, mode="sample", solve_kw=None):
    """Run the 3-way ablation (blueprint / sg-profile / sg-br) over the pool and
    compute the pooled per-challenger diffs, lifts, and the locked verdict."""
    solve_kw = solve_kw or {}
    challenger_specs = [
        PolicySpec(BLUEPRINT, "checkpoint", ckpt=blueprint_ckpt),
        PolicySpec(SG_PROFILE, "subgame_profile", ckpt=blueprint_ckpt, solve_kw=solve_kw),
        PolicySpec(SG_BR, "subgame_br", ckpt=blueprint_ckpt, solve_kw=solve_kw),
    ]
    agg = run_ablation(challenger_specs, opponent_specs, abstraction_path,
                       structure_path, hands=hands, base_seed=base_seed,
                       workers=workers, mode=mode, lift_pairs=_LIFT_PAIRS)

    onames = [s.name for s in opponent_specs]
    pooled = {}
    for c in (BLUEPRINT, SG_PROFILE, SG_BR):
        ds = [agg["per_matchup"][f"{c}|{o}"]["diff"] for o in onames
              if math.isfinite(agg["per_matchup"][f"{c}|{o}"]["diff"])]
        pooled[c] = (sum(ds) / len(ds)) if ds else float("nan")
    ordering_ok = (pooled[SG_BR] >= pooled[SG_PROFILE] >= pooled[BLUEPRINT])

    lbb = agg["lifts"]["sg-br_minus_blueprint"]
    lbp = agg["lifts"]["sg-br_minus_sg-profile"]
    verdict = compute_verdict(
        L=lbb["lift"], sigma_L=lbb["sigma"], ordering_ok=ordering_ok,
        n_opp_positive=lbb["n_opponents_positive"], n_opp=len(onames),
        L_brvsp=lbp["lift"], sigma_brvsp=lbp["sigma"])

    agg["pooled_diff"] = pooled
    agg["verdict"] = verdict
    return agg


def _build_output(agg, blueprint_ckpt, abstraction_path, structure_path,
                  wall_clock_s):
    pm = [v for v in agg["per_matchup"].values()]
    return {
        "ablation": "B1c sub-step 6 — Level-3 pool ablation (3-way)",
        "blueprint": {"path": blueprint_ckpt, "md5": _file_md5(blueprint_ckpt)},
        "abstraction": abstraction_path, "structure": structure_path,
        "challengers": [BLUEPRINT, SG_PROFILE, SG_BR],
        "config": agg["config"], "git_rev": _git_rev(),
        "wall_clock_s": round(wall_clock_s, 1),
        "per_matchup": pm,
        "pooled_diff": agg["pooled_diff"],
        "lifts": agg["lifts"],
        "stats": agg["stats"],
        "verdict": agg["verdict"],
    }


def main():
    import argparse
    import json
    import time
    ap = argparse.ArgumentParser(description="B1c sub-step 6 — 3-way pool ablation.")
    ap.add_argument("--blueprint-ckpt", required=True)
    ap.add_argument("--abstraction", required=True)
    ap.add_argument("--structure", default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--opponents", nargs="+", required=True,
                    help="name=path (path '__RANDOM__' for uniform random)")
    ap.add_argument("--base-seed", type=int, default=2026)
    ap.add_argument("--hands", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--mode", default="sample", choices=["sample", "argmax"])
    ap.add_argument("--output", default=None)
    ap.add_argument("--smoke", action="store_true", help="50 hands x first opponent only")
    # subgame solve overrides (defaults = SubgamePolicy production: M=8 depth-3 K=1000)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    args = ap.parse_args()

    opp_specs = []
    for spec in args.opponents:
        name, path = spec.split("=", 1)
        opp_specs.append(PolicySpec(name, "random") if path == "__RANDOM__"
                         else PolicySpec(name, "checkpoint", ckpt=path))
    hands = args.hands
    if args.smoke:
        hands, opp_specs = 50, opp_specs[:1]

    solve_kw = {}
    if args.n_samples is not None:
        solve_kw["n_samples"] = args.n_samples
    if args.depth is not None:
        solve_kw["max_action_depth"] = args.depth
    if args.iterations is not None:
        solve_kw["n_iterations"] = args.iterations

    print(f"3-way ablation | blueprint={args.blueprint_ckpt}\n"
          f"  {len(opp_specs)} opponents x {hands} hands, seed={args.base_seed}, "
          f"workers={args.workers}, smoke={args.smoke}")
    t0 = time.perf_counter()
    agg = run_three_way(args.blueprint_ckpt, opp_specs, args.abstraction, args.structure,
                        hands=hands, base_seed=args.base_seed, workers=args.workers,
                        mode=args.mode, solve_kw=solve_kw)
    wall = time.perf_counter() - t0
    out = _build_output(agg, args.blueprint_ckpt, args.abstraction, args.structure, wall)

    print(f"\nVERDICT: {out['verdict']['status']}  — {out['verdict']['recommendation']}")
    print(f"  L(BR-blueprint)={out['verdict']['L_br_vs_blueprint']:+.4f} "
          f"sigma={out['verdict']['sigma_L']:.2f} | "
          f"L(BR-PROFILE)={out['verdict']['L_br_vs_profile']:+.4f} "
          f"sigma={out['verdict']['sigma_br_vs_profile']:.2f} | "
          f"ordering_ok={out['verdict']['ordering_ok']}")
    print(f"  pooled diff: {', '.join(f'{k}={v:+.4f}' for k, v in out['pooled_diff'].items())}")
    print(f"  wall-clock: {wall:.0f}s")

    out_path = args.output or f"evals/subgame_ablation_seed{args.base_seed}_{hands}h.json"
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {out_path}")
    return out


if __name__ == "__main__":
    main()

