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
