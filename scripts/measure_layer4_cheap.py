"""C1d-cheap — standalone Layer 4 measurement (throwaway-grade infrastructure).

Plays N matches × 3 conditions × match_len hands:
  - baseline:           SubgamePolicy(bias_factory=None)
  - layer4_raw:         SubgamePolicy(bias_factory=make_bias_factory(observer, 'raw'))
  - layer4_archetype:   SubgamePolicy(bias_factory=make_bias_factory(observer, 'archetype', ...))

Each match runs against a fixed archetype scripted opponent (rotated through
NIT/TAG/LAG/STATION/MANIAC) seated at the 5 non-hero seats. Hero rotates
through all 6 seats across matches.

CRN per match: same RNG seed → identical opponent identity, identical hand
deals, identical seat assignment across the 3 conditions for a given
match_idx. Different match_idx → different CRN seeds. This pairs the noise
between conditions and lets paired t-tests detect smaller lift than the
unpaired between-match variance would.

Match boundary semantics: each "match" is match_len independent sampled hands.
Stacks are NOT persistent across hands (single-hand sampled starting state per
hand, matching the existing eval driver pattern). The Layer-4 observer DOES
persist across the match_len hands within a match; it wipes at match end.

Metric: per-match ICM-equity-delta total (sum across match_len hands) at
hero's seat. ICM-equity-delta is the same value Deep CFR optimizes against;
it's the natural unit for SNG strength comparison.

ARCHETYPE PATH CAVEAT (known limitation): the leaf_context_resolver in this
script provides STATIC per-seat context (median bucket_id, in_position=False)
because the resolver fires inside the subgame solver and doesn't know which
leaf infoset triggered it. This degrades the archetype path's mixture into a
leaf-agnostic posterior-weighted average. The raw path doesn't have this
limitation. The §6-Q3 sub-decision on bucket-marginalization is what would
improve this — out of scope here.

Verdict rule:
  - LAYER4_LIFTS:        ≥ one path has lift > 0 with p < 0.05 (one-sided).
  - NO_LIFT_DETECTED:    both paths have p > 0.20 AND point estimate < 0.005
                         ICM-equity-delta per hand (~0.5% per-hand lift).
  - INCONCLUSIVE:        anything else.

Output: evals/layer4_cheap_<UTC-timestamp>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyspiel

from src.nlhe.abstraction import Abstraction
from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.archetype6 import ArchetypePolicy as ArchetypeOppPolicy
from src.nlhe.archetypes import NAMED_ARCHETYPES, EquityCalibration
from src.nlhe.cfr6 import _build_view_6max
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.icm_returns import icm_adjust_returns
from src.nlhe.infoset6 import parse_state_6max, parse_state_repeated_6max
from src.nlhe.layer4_factory import make_bias_factory
from src.nlhe.stack_sampler import sample_starting_state
from src.nlhe.subgame_leaf import LeafEvalMode
from src.nlhe.subgame_policy import SubgamePolicy
from src.nlhe.within_match import MatchObserver

NUM_SEATS = 6
NUM_PAID = 3
PAYOUTS_DOUBLE_UP = [2.0] * NUM_PAID
_MAX_STEPS_PER_HAND = 500


# ============================================================
# One-hand play loop (mirrors eval_6max_self_play.play_one_hand but accepts
# arbitrary Policy objects with .select_action, not DeepCFR solvers)
# ============================================================

def _play_one_hand(seat_to_policy, structure, rng: random.Random,
                   observer: Optional[MatchObserver] = None) -> list[float]:
    """Play one hand from a sampled tournament state and return per-seat
    ICM-equity-delta (length-6 list). Mirrors the C1a observer hook in
    scripts/eval_6max_self_play.py."""
    sampled = sample_starting_state(structure, rng, num_paid=NUM_PAID)
    gs = structure.to_inner_game_string_for_state(
        blind_level=sampled["blind_level"],
        stacks=sampled["stacks"],
        dealer_seat=sampled["dealer_seat"],
    )
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    starting_stacks = list(sampled["stacks"])

    for _ in range(_MAX_STEPS_PER_HAND):
        if state.is_terminal():
            break
        if state.is_chance_node():
            outs = state.chance_outcomes()
            a = rng.choices([o[0] for o in outs],
                            weights=[o[1] for o in outs], k=1)[0]
            state.apply_action(int(a))
            continue
        if hasattr(state, "dealer_seat"):
            parsed = parse_state_repeated_6max(state)
        else:
            parsed = parse_state_6max(state)
        cp = parsed["current_player"]
        a = seat_to_policy[cp].select_action(parsed, state, rng, mode="sample")
        if observer is not None:
            view = _build_view_6max(state, parsed)
            discrete_to_chip = discretize_legal_actions(list(state.legal_actions()), view)
            da_int = next((int(d) for d, c in discrete_to_chip.items()
                           if c == a), None)
            if da_int is not None:
                observer.update(state, parsed, da_int, cp)
        state.apply_action(int(a))

    if not state.is_terminal():
        return [0.0] * NUM_SEATS  # safety cap; treat as wash

    chip_returns = state.returns()
    return list(icm_adjust_returns(
        chip_returns=chip_returns,
        starting_stacks=starting_stacks,
        payouts=PAYOUTS_DOUBLE_UP,
    ))


# ============================================================
# Match (match_len independent hands, observer persists, wipe at end)
# ============================================================

def _play_one_match(hero_seat: int, hero_policy, opp_policy,
                    structure, match_len: int, rng: random.Random,
                    observer: MatchObserver) -> float:
    """Play match_len hands; return hero's accumulated ICM-equity-delta.

    `hero_policy` is the SubgamePolicy (its bias_factory determines the
    condition). `opp_policy` is the ArchetypeOppPolicy for the 5 non-hero
    seats. The observer wipes at start and end.
    """
    observer.match_started()
    seat_to_policy = [opp_policy] * NUM_SEATS
    seat_to_policy[hero_seat] = hero_policy

    total = 0.0
    for _ in range(match_len):
        per_seat = _play_one_hand(seat_to_policy, structure, rng,
                                  observer=observer)
        total += float(per_seat[hero_seat])
    observer.match_ended()
    return total


# ============================================================
# Per-condition bias_factory wiring
# ============================================================

def _make_archetype_resolver(calibration: EquityCalibration,
                             median_bucket_id: int = 100):
    """Static per-seat resolver for the archetype path.

    CAVEAT (script-level): the resolver fires inside the subgame solver per
    leaf, but we don't surface leaf context here. We hand back a fixed
    median bucket and in_position=False so the archetype-policy lookup is
    deterministic across leaves. C1b's archetype path therefore evaluates
    archetype distributions at a leaf-AGNOSTIC bucket. Degraded vs the
    intended per-leaf lookup; documented as a measurement limitation.
    """
    static_parsed = {
        "street_idx": 1,  # flop — any postflop street is fine for the mixture
        "current_player": 0,  # ignored by archetype_policy
        "contribution": [0] * NUM_SEATS,
        "money": [1000] * NUM_SEATS,
        "pot": 100,
        "legal_mask": np.ones(7, dtype=np.float32),
    }

    def resolver(seat: int) -> dict:
        return {
            "parsed": static_parsed,
            "state": None,
            "bucket_id": median_bucket_id,
            "in_position": False,
            "calibration": calibration,
        }
    return resolver


def _apply_condition(hero_policy: SubgamePolicy, condition: str,
                     observer: MatchObserver,
                     calibration: Optional[EquityCalibration]) -> None:
    """Mutate `hero_policy.bias_factory` in place per condition. The
    SubgamePolicy is constructed once at script start (loading the solver
    is expensive); mutating bias_factory between matches is bit-identity-safe
    by C1c's locked invariant."""
    if condition == "baseline":
        hero_policy.bias_factory = None
    elif condition == "raw":
        hero_policy.bias_factory = make_bias_factory(observer, "raw")
    elif condition == "archetype":
        if calibration is None:
            raise RuntimeError(
                "archetype path requires a loaded EquityCalibration "
                "(missing runs/archetype_design/bucket_equity_analysis_6max.json?)"
            )
        resolver = _make_archetype_resolver(calibration)
        hero_policy.bias_factory = make_bias_factory(
            observer, "archetype", leaf_context_resolver=resolver)
    else:
        raise ValueError(f"unknown condition: {condition!r}")


# ============================================================
# Stats / verdict
# ============================================================

@dataclass
class CondStats:
    name: str
    outcomes: list[float]

    def mean(self) -> float:
        return float(np.mean(self.outcomes)) if self.outcomes else 0.0

    def std(self) -> float:
        return float(np.std(self.outcomes, ddof=1)) if len(self.outcomes) > 1 else 0.0


def _paired_t_one_sided_greater(diffs: list[float]) -> tuple[float, float]:
    """Return (t_statistic, p_value) for H1: mean(diffs) > 0.

    Uses the survival function of the standard normal as a t→p approximation
    (n is large enough — N=200 — that the t and normal are indistinguishable
    at the report-precision needed). Avoids scipy dependency."""
    if len(diffs) < 2:
        return 0.0, 1.0
    arr = np.asarray(diffs, dtype=np.float64)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0:
        return (float("inf"), 0.0) if mu > 0 else (0.0, 1.0)
    se = sd / math.sqrt(len(arr))
    t = mu / se
    # One-sided greater: p = 1 - Phi(t) = 0.5 * erfc(t / sqrt(2)).
    p = 0.5 * math.erfc(t / math.sqrt(2.0))
    return t, p


def _decide_verdict(raw_stats: dict, arch_stats: dict) -> str:
    """LAYER4_LIFTS / NO_LIFT_DETECTED / INCONCLUSIVE per the spec rule."""
    if raw_stats["p_value"] < 0.05 and raw_stats["lift_per_hand"] > 0:
        return "LAYER4_LIFTS"
    if arch_stats["p_value"] < 0.05 and arch_stats["lift_per_hand"] > 0:
        return "LAYER4_LIFTS"
    no_lift = (
        raw_stats["p_value"] > 0.20 and arch_stats["p_value"] > 0.20
        and abs(raw_stats["lift_per_hand"]) < 0.005
        and abs(arch_stats["lift_per_hand"]) < 0.005
    )
    if no_lift:
        return "NO_LIFT_DETECTED"
    return "INCONCLUSIVE"


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_matches", type=int, default=200)
    p.add_argument("--match_len", type=int, default=40)
    p.add_argument("--base_seed", type=int, default=12345)
    p.add_argument("--blueprint_ckpt", required=True,
                   help="path to baseline DeepCFR6MaxSolver checkpoint")
    p.add_argument("--abstraction", required=True,
                   help="path to abstraction .pkl matching the blueprint")
    p.add_argument("--structure",
                   default="configs/ignition_double_up_6max_turbo.yaml")
    p.add_argument("--calibration",
                   default="runs/archetype_design/bucket_equity_analysis_6max.json",
                   help="EquityCalibration JSON for archetypes")
    p.add_argument("--out_dir", default="evals/")
    p.add_argument("--alpha", type=float, default=2.0,
                   help="C1 bias clip bound (locked §5-Q1 default = 2.0)")
    p.add_argument("--n_iterations", type=int, default=1000,
                   help="subgame solver K (SUBSTEP_5_DESIGN default = 1000)")
    args = p.parse_args()

    t0 = time.time()
    print(f"[c1d-cheap] start: {args.n_matches} matches × {args.match_len} hands × 3 conditions")
    print(f"[c1d-cheap] blueprint: {args.blueprint_ckpt}")
    print(f"[c1d-cheap] abstraction: {args.abstraction}")

    abstraction = Abstraction.load(args.abstraction)
    structure = TournamentStructure.from_yaml(args.structure)

    cal_path = Path(args.calibration)
    if cal_path.exists():
        calibration = EquityCalibration.load(cal_path)
    else:
        calibration = None
        print(f"[c1d-cheap] WARN: {cal_path} missing — archetype condition will fail")

    print(f"[c1d-cheap] constructing hero SubgamePolicy (loads checkpoint)...")
    hero_policy = SubgamePolicy(
        name="c1d_hero",
        ckpt_path=args.blueprint_ckpt,
        abstraction=abstraction,
        structure=structure,
        leaf_mode=LeafEvalMode.BEST_RESPONSE,
        n_iterations=args.n_iterations,
    )
    print(f"[c1d-cheap] hero ready; gate min_legal={hero_policy.min_legal_actions}, "
          f"max_prob={hero_policy.max_blueprint_prob}")

    conditions = ["baseline", "raw", "archetype"]
    cond_outcomes: dict[str, list[float]] = {c: [] for c in conditions}
    per_match_records: list[dict] = []

    for match_idx in range(args.n_matches):
        match_seed = args.base_seed + match_idx
        hero_seat = match_idx % NUM_SEATS
        archetype_profile = NAMED_ARCHETYPES[match_idx % len(NAMED_ARCHETYPES)]
        opp_policy = ArchetypeOppPolicy(
            profile=archetype_profile,
            abstraction=abstraction,
            calibration=(calibration if calibration is not None
                         else EquityCalibration(
                             bucket_equity={s: np.array([0.5]) for s in
                                            ("preflop", "flop", "turn", "river")},
                             quantile_thresholds={s: {"q05": 0.4, "q50": 0.5, "q95": 0.6}
                                                  for s in
                                                  ("preflop", "flop", "turn", "river")},
                         )),
        )

        match_record = {
            "match_idx": match_idx,
            "hero_seat": hero_seat,
            "opp_archetype": archetype_profile.name.name,
            "outcomes": {},
        }
        for condition in conditions:
            # CRN: identical RNG state per (match, condition).
            rng = random.Random(match_seed)
            observer = MatchObserver(num_seats=NUM_SEATS)
            _apply_condition(hero_policy, condition, observer, calibration)
            outcome = _play_one_match(
                hero_seat=hero_seat, hero_policy=hero_policy, opp_policy=opp_policy,
                structure=structure, match_len=args.match_len, rng=rng,
                observer=observer,
            )
            cond_outcomes[condition].append(outcome)
            match_record["outcomes"][condition] = outcome

        per_match_records.append(match_record)
        if (match_idx + 1) % max(1, args.n_matches // 20) == 0 or match_idx == args.n_matches - 1:
            elapsed = time.time() - t0
            done = match_idx + 1
            base_mean = float(np.mean(cond_outcomes["baseline"]))
            raw_mean = float(np.mean(cond_outcomes["raw"]))
            arch_mean = float(np.mean(cond_outcomes["archetype"]))
            print(f"[c1d-cheap] match {done:>4}/{args.n_matches}  "
                  f"base={base_mean:+.4f}  raw={raw_mean:+.4f}  arch={arch_mean:+.4f}  "
                  f"[{elapsed:.1f}s, {elapsed/done:.2f}s/match]")

    # Aggregate.
    base = CondStats("baseline", cond_outcomes["baseline"])
    raw = CondStats("raw", cond_outcomes["raw"])
    arch = CondStats("archetype", cond_outcomes["archetype"])

    raw_diffs = [r - b for r, b in zip(cond_outcomes["raw"], cond_outcomes["baseline"])]
    arch_diffs = [a - b for a, b in zip(cond_outcomes["archetype"], cond_outcomes["baseline"])]
    raw_t, raw_p = _paired_t_one_sided_greater(raw_diffs)
    arch_t, arch_p = _paired_t_one_sided_greater(arch_diffs)
    raw_lift = float(np.mean(raw_diffs)) if raw_diffs else 0.0
    arch_lift = float(np.mean(arch_diffs)) if arch_diffs else 0.0

    raw_summary = {
        "mean_match": raw.mean(), "std_match": raw.std(), "n": len(raw.outcomes),
        "lift_vs_baseline_match": raw_lift,
        "lift_per_hand": raw_lift / args.match_len,
        "paired_t": raw_t, "p_value": raw_p,
    }
    arch_summary = {
        "mean_match": arch.mean(), "std_match": arch.std(), "n": len(arch.outcomes),
        "lift_vs_baseline_match": arch_lift,
        "lift_per_hand": arch_lift / args.match_len,
        "paired_t": arch_t, "p_value": arch_p,
    }
    base_summary = {
        "mean_match": base.mean(), "std_match": base.std(), "n": len(base.outcomes),
    }
    raw_vs_arch_diffs = [r - a for r, a in zip(cond_outcomes["raw"], cond_outcomes["archetype"])]
    rva_t, rva_p = _paired_t_one_sided_greater(raw_vs_arch_diffs)
    raw_vs_arch_summary = {
        "lift_match": float(np.mean(raw_vs_arch_diffs)) if raw_vs_arch_diffs else 0.0,
        "paired_t": rva_t, "p_value": rva_p,
    }

    verdict = _decide_verdict(raw_summary, arch_summary)
    wall_s = time.time() - t0

    out = {
        "config": {
            "n_matches": args.n_matches,
            "match_len": args.match_len,
            "base_seed": args.base_seed,
            "blueprint_ckpt": args.blueprint_ckpt,
            "abstraction": args.abstraction,
            "structure": args.structure,
            "calibration": args.calibration,
            "alpha": args.alpha,
            "n_iterations": args.n_iterations,
        },
        "git_rev": _git_rev(),
        "wall_clock_s": wall_s,
        "per_match": per_match_records,
        "summary": {
            "baseline": base_summary,
            "raw": raw_summary,
            "archetype": arch_summary,
            "raw_vs_archetype": raw_vs_arch_summary,
        },
        "verdict": verdict,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"layer4_cheap_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[c1d-cheap] done in {wall_s:.1f}s. wrote {out_path}")
    print(f"[c1d-cheap] verdict: {verdict}")
    print(f"[c1d-cheap]   baseline:  {base_summary['mean_match']:+.4f} ± {base_summary['std_match']:.4f}")
    print(f"[c1d-cheap]   raw:       {raw_summary['mean_match']:+.4f} ± {raw_summary['std_match']:.4f}  "
          f"(lift/hand {raw_summary['lift_per_hand']:+.5f}, p={raw_summary['p_value']:.4f})")
    print(f"[c1d-cheap]   archetype: {arch_summary['mean_match']:+.4f} ± {arch_summary['std_match']:.4f}  "
          f"(lift/hand {arch_summary['lift_per_hand']:+.5f}, p={arch_summary['p_value']:.4f})")


if __name__ == "__main__":
    main()
