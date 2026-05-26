"""scripts/subgame_6d_diag_v1.py — BR-vs-PROFILE leaf-eval diagnostic spike (Stage 6-D, Path A).

Diagnoses the Stage-6-D calibration finding (sg-br significantly WORSE than sg-profile:
L(BR-PROFILE)=-0.0426, sigma=2.05 at 200 hands). Evaluates the SAME subgame trees under
BOTH LeafEvalMode.PROFILE_SAMPLE and LeafEvalMode.BEST_RESPONSE at high M (default 128) to
suppress rollout MC noise, and captures per-solve leaf values + root policies side by side
for offline analysis. Output: evals/subgame_6d_diag_v1.json.

READ-ONLY on production code: this script imports and re-uses SubgamePolicy / subgame_leaf /
subgame_solver / eval_pool_ablation, and NEVER modifies them. The only departure from
production is the n_samples knob (M=128 vs the M=8 production default).

Design decisions carried from the session-22 recon (see NEXT_SESSION / the spike thread):
  D2  evaluate_leaves mutates node.leaf_value IN PLACE and the batch summary carries no
      per-leaf detail, so the two modes are interleaved on ONE tree with a LOCKED ordering:
      eval(PROFILE) -> read leaf values -> solve(PROFILE) -> eval(BR) [overwrites] -> read -> solve(BR).
  D3  BR-selected biases are not exposed by any public path (LeafEvalResult lacks them and
      evaluate_leaves returns only a summary) -> NOT captured in v1.
  D4  There is no single "decision sequence the calibration saw" (each challenger's policy_rng
      diverges after its first solve). The trajectory here is BLUEPRINT-DRIVEN — mode-independent
      by construction — so BR and PROFILE are compared on identical states. This is NOT a replay
      of any one calibration challenger's path.
  D5  M=128 is ~16x the M=8 calibration cost, so the run is parallelized across hands.
  RNG Noise control is HIGH M, not seed-CRN between modes — their rng consumption order differs
      and evaluate_leaves shares the bucket cache, so seed-CRN would be only approximate. Per-
      decision capture RNGs are derived deterministically from the hand seed purely for
      reproducibility; the trajectory uses the same split-RNG scheme as eval_pool_ablation.

Determinism: each hand is seeded by SHA256(base_seed:opp_idx:hand_idx) (the harness's hand_seed);
all capture RNGs derive from it. Output is bit-identical regardless of --workers.
"""
from __future__ import annotations

import os
# CPU-forced (Q13: single-row forward is launch-bound, CPU beats GPU for this evaluator).
# setdefault so an explicit launch-time CUDA_VISIBLE_DEVICES still wins; must precede torch import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import pyspiel

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.stack_sampler import sample_starting_state
from src.nlhe.cfr6 import NUM_SEATS_6MAX
from src.nlhe.infoset6 import parse_state_6max, parse_state_repeated_6max
from src.nlhe.subgame import build_subgame_tree, iter_leaf_nodes
from src.nlhe.subgame_leaf import LeafEvalContext, LeafEvalMode, evaluate_leaves
from src.nlhe.subgame_solver import (SubgameSolveContext, solve_subgame,
                                     summarize_solve_result)
# Re-use the harness's exact PolicySpec.build / hand_seed / step cap for production fidelity.
from scripts.eval_pool_ablation import PolicySpec, hand_seed, _MAX_STEPS

# Defaults mirror the Stage-6-D calibration run (base_seed=7, dcfr-shake-200, single opp).
_DEF_BLUEPRINT = "runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/ckpt_iter_3000.pt"
_DEF_ABSTRACTION = "runs/abstraction_20260521_223018/abstraction.pkl"
_DEF_OPPONENT = "runs/six_max_20260524_005853_phase4f_dcfr_linear_shakedown/checkpoints/ckpt_iter_0200.pt"
_DEF_STRUCTURE = "configs/ignition_double_up_6max_turbo.yaml"

_STREETS = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
# Conservative lower bound on gate-solve decisions per hand (calibration saw ~0.90: 173/193).
# Used only to bound how many hands to process so we comfortably clear --target-solves.
_PLAN_RATE = 0.7


# ============================================================
# Helpers
# ============================================================

def _parse(state):
    """Mirror eval_pool_ablation._play_one_hand's parse dispatch."""
    return (parse_state_repeated_6max(state)
            if hasattr(state, "dealer_seat") else parse_state_6max(state))


def _derive_rng(seed: int, decision_idx: int, tag: str) -> random.Random:
    """Deterministic per-(hand-seed, hero-decision, purpose) RNG — reproducibility only."""
    h = hashlib.sha256(f"{seed}:{decision_idx}:{tag}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _leaf_hero_values(tree, hero_seat: int) -> list:
    """Hero-seat component of each leaf's value, in iter_leaf_nodes order (stable across the
    two passes — the tree structure is unchanged, only node.leaf_value is overwritten). The
    solver consumes exactly this hero component at terminal/leaf nodes. None for any leaf left
    unevaluated (should not happen without a time budget; guarded anyway)."""
    out = []
    for node in iter_leaf_nodes(tree):
        lv = node.leaf_value
        out.append(None if lv is None else float(lv[hero_seat]))
    return out


def _l1(a: list, b: list) -> float:
    """L1 over index-aligned lists, skipping any None entries."""
    return float(sum(abs(x - y) for x, y in zip(a, b) if x is not None and y is not None))


def _max_abs(a: list, b: list) -> float:
    diffs = [abs(x - y) for x, y in zip(a, b) if x is not None and y is not None]
    return float(max(diffs)) if diffs else float("nan")


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


# ============================================================
# Per-solve capture (D2 locked ordering)
# ============================================================

def _capture_solve(diag, parsed, state, seed, decision_idx, hand_idx, gate, n_samples):
    """Build ONE tree at this hero decision, evaluate leaves under PROFILE then BR (high M),
    solve each, and return the per-solve record. PROFILE solve runs BEFORE the BR eval
    overwrites node.leaf_value (D2)."""
    cp = parsed["current_player"]
    starting_stacks = diag._reconstruct_starting_stacks(parsed)

    tree = build_subgame_tree(
        state, max_action_depth=diag.max_action_depth,
        chance_samples_per_node=diag.chance_samples_per_node,
        rng=_derive_rng(seed, decision_idx, "tree"))
    n_chance_leaves = sum(1 for lf in iter_leaf_nodes(tree)
                          if lf.state is not None and lf.state.is_chance_node())

    def _ctx(mode, tag):
        # Mirrors SubgamePolicy._solve_action's LeafEvalContext exactly, except n_samples.
        return LeafEvalContext(
            blueprint=diag.solver, biased_blueprint=diag.biased,
            starting_stacks=starting_stacks, payouts=diag.payouts, hero_seat=cp,
            mode=mode, n_samples=n_samples, rng=_derive_rng(seed, decision_idx, tag),
            num_paid=diag.num_paid)

    def _solve_ctx(tag):
        return SubgameSolveContext(
            blueprint=diag.solver, starting_stacks=starting_stacks, payouts=diag.payouts,
            hero_seat=cp, n_iterations=diag.n_iterations,
            rng=_derive_rng(seed, decision_idx, tag), num_paid=diag.num_paid)

    # ---- PROFILE pass (eval -> read -> solve), then BR pass (overwrites) ----
    evaluate_leaves(tree, _ctx(LeafEvalMode.PROFILE_SAMPLE, "eval_profile"))
    lv_profile = _leaf_hero_values(tree, cp)
    sp = summarize_solve_result(solve_subgame(tree, _solve_ctx("solve_profile")))

    evaluate_leaves(tree, _ctx(LeafEvalMode.BEST_RESPONSE, "eval_br"))
    lv_br = _leaf_hero_values(tree, cp)
    sb = summarize_solve_result(solve_subgame(tree, _solve_ctx("solve_br")))

    # "active" has no flag in parsed; use seats holding a stake (money behind + committed > 0).
    n_active = sum(1 for i in range(NUM_SEATS_6MAX)
                   if (int(parsed["money"][i]) + int(parsed["contribution"][i])) > 0)

    return {
        "hand_idx": hand_idx,
        "decision_idx_in_hand": decision_idx,
        "hero_seat": cp,
        "street": _STREETS.get(int(gate["street_idx"]), str(gate["street_idx"])),
        "n_active_seats": n_active,  # seats with a stake (money+contribution>0); see module note
        "gate": dict(gate),
        "tree": {
            "n_leaf_nodes": tree.n_leaf_nodes,
            "n_decision_nodes": tree.n_decision_nodes,
            "n_chance_nodes": tree.n_chance_nodes,
            "n_chance_leaves": n_chance_leaves,
        },
        "profile": {
            "leaf_values": lv_profile,
            "root_policy": sp["root_policy"],
            "root_blueprint": sp["root_blueprint"],   # mode-independent; kept once here
            "policy_shift_l1": sp["policy_shift_l1"],
            "convergence_tail": sp["converged_l1_tail"],
            "n_iterations_run": sp["n_iterations_run"],
            "degraded": sp["degraded"],
        },
        "br": {
            "leaf_values": lv_br,
            "root_policy": sb["root_policy"],          # root_blueprint skipped (== profile's)
            "policy_shift_l1": sb["policy_shift_l1"],
            "convergence_tail": sb["converged_l1_tail"],
            "n_iterations_run": sb["n_iterations_run"],
            "degraded": sb["degraded"],
        },
        "derived": {
            "leaf_value_l1_br_vs_profile": _l1(lv_br, lv_profile),
            "root_policy_l1_br_vs_profile": _l1(sb["root_policy"], sp["root_policy"]),
            "leaf_value_max_abs_diff": _max_abs(lv_br, lv_profile),
        },
    }


# ============================================================
# Per-hand replay (blueprint-driven trajectory, D4)
# ============================================================

def _play_and_capture(hand_idx, seed, diag, opponent, structure, n_samples, num_paid):
    """Play one hand; capture a per-solve record at every gated-SOLVE hero decision. Hero
    (A-seat) decisions advance the trajectory with the BLUEPRINT action (mode-independent),
    opponent (B-seat) decisions use the opponent ckpt — both as in _play_one_hand. The same
    split-RNG scheme (chance_rng / policy_rng) is used so deals match the calibration regime."""
    chance_rng = random.Random(seed)
    policy_rng = random.Random((seed ^ 0x5DEECE66D) & 0x7FFFFFFF)

    sampled = sample_starting_state(structure, chance_rng, num_paid=num_paid)
    gs = structure.to_inner_game_string_for_state(
        blind_level=sampled["blind_level"], stacks=sampled["stacks"],
        dealer_seat=sampled["dealer_seat"])
    state = pyspiel.load_game(gs).new_initial_state()

    seat_assignment = [chance_rng.choice(["A", "B"]) for _ in range(NUM_SEATS_6MAX)]

    solves = []
    decision_idx = 0          # hero-decision ordinal within the hand (solve OR skip)
    n_solves = n_skips = 0
    truncated = False

    for _ in range(_MAX_STEPS):
        if state.is_terminal():
            break
        if state.is_chance_node():
            outs = state.chance_outcomes()
            a = chance_rng.choices([o[0] for o in outs],
                                   weights=[o[1] for o in outs], k=1)[0]
            state.apply_action(int(a))
            continue

        parsed = _parse(state)
        cp = parsed["current_player"]
        if seat_assignment[cp] == "B":
            state.apply_action(int(opponent.select_action(
                parsed, state, policy_rng, mode="sample")))
            continue

        # Hero (A-seat) decision. Gate is mode-independent (blueprint-only); reuse for both.
        gate = diag._evaluate_gate(parsed, state, policy_rng)
        if gate["solve"]:
            solves.append(_capture_solve(
                diag, parsed, state, seed, decision_idx, hand_idx, gate, n_samples))
            n_solves += 1
        else:
            n_skips += 1
        # Advance with the blueprint action (D4): identical policy_rng consumption to a
        # production SKIP, so the trajectory does not depend on solve/skip.
        state.apply_action(int(diag._blueprint_action(parsed, state, policy_rng, "sample")))
        decision_idx += 1
    else:
        truncated = True  # hit _MAX_STEPS without terminal

    summary = {"hand_idx": hand_idx, "n_hero_decisions": decision_idx,
               "n_solves": n_solves, "n_skips": n_skips, "truncated": truncated}
    return solves, summary


# ============================================================
# Worker (one shard of hand indices)
# ============================================================

def _worker(args):
    # Pin intra-op threads to 1: with `workers` processes on a many-core box, torch's
    # default unbounded intra-op pool oversubscribes (workers x ~ncores threads), and
    # the contention dominates wall-clock. One thread/worker = no oversubscription.
    import torch
    torch.set_num_threads(1)
    (hand_idxs, blueprint_ckpt, opponent_ckpt, abstr_path, struct_path,
     base_seed, opp_idx, n_samples, num_paid) = args

    abstraction = Abstraction.load(abstr_path)
    structure = TournamentStructure.from_yaml(struct_path)
    # ONE SubgamePolicy serves both modes — we build our own LeafEvalContexts with explicit
    # mode, so the policy's leaf_mode is never used (kind chosen arbitrarily). It supplies the
    # loaded blueprint solver, the bias menu, payouts, tree depths, gate, and stack reconstruction.
    diag = PolicySpec("diag", "subgame_profile", ckpt=blueprint_ckpt).build(abstraction, structure)
    opponent = PolicySpec("opp", "checkpoint", ckpt=opponent_ckpt).build(abstraction, structure)

    solves, summaries = [], []
    for hand_idx in hand_idxs:
        seed = hand_seed(base_seed, opp_idx, hand_idx)
        s, summ = _play_and_capture(
            hand_idx, seed, diag, opponent, structure, n_samples, num_paid)
        solves.extend(s)
        summaries.append(summ)
    return {"solves": solves, "hand_summaries": summaries}


# ============================================================
# CLI / orchestration
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="BR-vs-PROFILE leaf-eval diagnostic spike (Stage 6-D, Path A). "
                    "Same trees, both leaf modes, high M to suppress rollout noise.")
    ap.add_argument("--blueprint-ckpt", default=_DEF_BLUEPRINT)
    ap.add_argument("--abstraction", default=_DEF_ABSTRACTION)
    ap.add_argument("--opponent-ckpt", default=_DEF_OPPONENT)
    ap.add_argument("--structure", default=_DEF_STRUCTURE)
    ap.add_argument("--base-seed", type=int, default=7)
    ap.add_argument("--opp-idx", type=int, default=0)
    ap.add_argument("--target-solves", type=int, default=50)
    ap.add_argument("--max-hands", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=128, help="leaf-eval M (production default 8)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--num-paid", type=int, default=3)
    ap.add_argument("--output", default="evals/subgame_6d_diag_v1.json")
    args = ap.parse_args()

    # Surface missing assets loudly rather than failing deep in a worker.
    for label, path in (("structure", args.structure), ("abstraction", args.abstraction),
                        ("blueprint-ckpt", args.blueprint_ckpt),
                        ("opponent-ckpt", args.opponent_ckpt)):
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: --{label} not found: {path}")

    # Bound the hands processed so we comfortably clear target-solves without over-computing
    # (each BR solve at M=128 is expensive). Process [0, n_planned), collect all solves,
    # then trim deterministically to target-solves in (hand_idx, decision_idx) order.
    n_planned = min(args.max_hands, math.ceil(args.target_solves / _PLAN_RATE))
    hand_idxs = list(range(n_planned))
    nw = max(1, min(args.workers, n_planned))
    shards = [hand_idxs[i::nw] for i in range(nw)]
    worker_args = [
        (shard, args.blueprint_ckpt, args.opponent_ckpt, args.abstraction, args.structure,
         args.base_seed, args.opp_idx, args.n_samples, args.num_paid)
        for shard in shards if shard
    ]

    print(f"diag spike | blueprint={args.blueprint_ckpt}")
    print(f"  opponent={args.opponent_ckpt}  M={args.n_samples}  base_seed={args.base_seed} "
          f"opp_idx={args.opp_idx}")
    print(f"  planning {n_planned} hands across {len(worker_args)} workers, "
          f"target {args.target_solves} solves")

    t0 = time.time()
    all_solves, all_summaries = [], []
    with ProcessPoolExecutor(max_workers=len(worker_args)) as ex:
        for r in ex.map(_worker, worker_args):
            all_solves.extend(r["solves"])
            all_summaries.extend(r["hand_summaries"])
    wall = time.time() - t0

    # Deterministic ordering, then trim to target.
    all_solves.sort(key=lambda s: (s["hand_idx"], s["decision_idx_in_hand"]))
    all_summaries.sort(key=lambda s: s["hand_idx"])
    captured = len(all_solves)
    trimmed = all_solves[:args.target_solves]

    if captured >= args.target_solves:
        stop_reason = "target_solves_reached"
    elif n_planned >= args.max_hands:
        stop_reason = "max_hands_reached"
    else:
        stop_reason = "plan_exhausted_below_target"  # planned estimate too low; rerun w/ more hands

    out = {
        "metadata": {
            "base_seed": args.base_seed,
            "opp_idx": args.opp_idx,
            "n_samples": args.n_samples,
            "workers": len(worker_args),
            "blueprint_ckpt": args.blueprint_ckpt,
            "opponent_ckpt": args.opponent_ckpt,
            "abstraction": args.abstraction,
            "structure": args.structure,
            "n_hands_planned": n_planned,
            "n_hands_processed": len(all_summaries),
            "n_solves_captured": captured,
            "n_solves_written": len(trimmed),
            "git_rev": _git_rev(),
            "wall_clock_s": round(wall, 1),
        },
        "solves": trimmed,
        "hand_summaries": all_summaries,
        "stop_reason": stop_reason,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\ncaptured {captured} solves over {len(all_summaries)} hands "
          f"(wrote {len(trimmed)}); stop_reason={stop_reason}; wall={wall:.1f}s")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
