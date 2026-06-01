"""ICM-panel floor v2 — 2-arm × 6-seat PAIRED.

Supersedes sixmax_icm_panel.py (single-arm, random A/B seats). This version
implements the floor gate's actual claim: student-zero-context >= blueprint,
measured as a PAIRED delta per panel member, with the blueprint arm run on
the SAME anchor (candC_k200) and the SAME dealt situations.

Design (per human direction, 2026-06-01):
  - 2 arms:  student-zero-context  and  blueprint  (both as hero)
  - 5 members: {blueprint anchor, 2 worst Shanky vs dcfr-3000, maniac, nit}
  - 6 fixed hero seats (rotation; per-seat reported so one bad seat shows)
  - N = 1000 hands per (member, arm, seat) -> pooled N=6000/member/arm
  - ICM via icm_adjust_returns(payouts=[2.0,2.0,2.0]) on sampled tournament
    states (double-up structure) -- deployment currency.

PAIRED CRN (the delta is truly paired):
  For hand h at seat s of member m, hand_seed = base(m,s)+h. BOTH arms reset
  their RNG to hand_seed before playing -> identical sample_starting_state
  (stacks/level/dealer) AND identical hole-card + early board deals for all 6
  seats. The two arms diverge only once the hero takes its first (differing)
  action. Same seat, same N, same structure. Per-hand paired delta =
  student_icm[h] - blueprint_icm[h].

Surface guarantees (recon-locked, unchanged):
  - StudentZeroContextPolicy = Test-D6 forward surface (tokens [1,1,236],
    pad_mask [1,1]=False, query_idx=[0], opp_stats [1,12]=zeros) -> trunk +
    policy_head directly. NOT the blend(g_total=0) anchor passthrough.
  - blueprint arm = sm._BlueprintAsOpp(candC_k200 solver) -- identical anchor.

Pre-committed read (human owns the gate; this script only emits numbers):
  student >= blueprint within |sigma|<2 on pooled delta vs EVERY member AND
  no single seat materially negative -> floor signed off. Any member pooled
  <= -2 sigma OR any seat materially negative -> localized breach; stop.

Diffs to /tmp first; copied to the run dir + committed once spec-complete.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, "/home/quant/pokerbot")
sys.path.insert(0, "/home/quant/pokerbot/runs/sixmax_smoke_20260601_020033")

# Triggers leakage preflight + threading setup before heavy imports.
from scripts import six_max_adaptive_smoke as sm  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import pyspiel  # noqa: E402

from src.nlhe.adaptive.model import Adaptive6MaxNet  # noqa: E402
from src.nlhe.archetype6 import ArchetypePolicy  # noqa: E402
from src.nlhe.archetypes import (  # noqa: E402
    ArchetypeName, EquityCalibration, NAMED_ARCHETYPES,
)
from src.nlhe.game_strings import TournamentStructure  # noqa: E402
from src.nlhe.icm_returns import icm_adjust_returns  # noqa: E402
from src.nlhe.infoset6 import parse_state_6max, parse_state_repeated_6max  # noqa: E402
from src.nlhe.scripted_bots.policy import ShankyProfilePolicy  # noqa: E402
from src.nlhe.stack_sampler import sample_starting_state  # noqa: E402

from sixmax_zero_context_ev import (  # noqa: E402
    ABSTRACTION_PKL, ANCHOR_DIR, STUDENT_CKPT, STUDENT_CONFIG,
    StudentZeroContextPolicy,
)

STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"
SHANKY_DIR = "data/shanky_profiles"
ARCHETYPE_CALIB_PATH = "runs/archetype_design/bucket_equity_analysis_6max.json"
NUM_SEATS = 6
NUM_PAID = 3
PAYOUTS = [2.0] * NUM_PAID
MAX_STEPS = 500
_SEAT_STRIDE = 1_000_000  # disjoint hand_seed ranges across seats

# (member_name, kind, base_seed) — base_seed disjoint per member.
PANEL_SPEC = [
    ("blueprint",                  "blueprint", 70_000_000),
    ("shanky:minestackermttv_7_3", "shanky",    71_000_000),
    ("shanky:killphilmtt",         "shanky",    72_000_000),
    ("archetype-maniac",           "archetype", 73_000_000),
    ("archetype-nit",              "archetype", 74_000_000),
]
SHANKY_FILE = {
    "shanky:minestackermttv_7_3": "MinestackerMTTv.7.3.txt",
    "shanky:killphilmtt":         "KillPhilMTT.txt",
}


def build_panel(solver, abstraction, calibration, big_blind_chips=100):
    out = []
    for name, kind, base_seed in PANEL_SPEC:
        if kind == "blueprint":
            pol = sm._BlueprintAsOpp(solver)
            pol.name = name
        elif kind == "shanky":
            pol = ShankyProfilePolicy(
                name=name,
                profile_path=os.path.join(SHANKY_DIR, SHANKY_FILE[name]),
                big_blind_chips=big_blind_chips,
            )
        elif kind == "archetype":
            arche = name.split("-", 1)[1].upper()
            profile = next(p for p in NAMED_ARCHETYPES
                           if p.name == ArchetypeName[arche])
            pol = ArchetypePolicy(profile, abstraction, calibration)
            pol.name = name
        else:
            raise ValueError(f"unknown kind {kind!r}")
        out.append((name, pol, base_seed))
    return out


def play_one_hand_fixed_seat(hero_policy, opp_policy, hero_seat, structure, rng):
    """One sampled tournament hand; hero in hero_seat, all other seats =
    opp_policy. Returns hero's ICM-equity-delta (float), or None if capped."""
    sampled = sample_starting_state(structure, rng, num_paid=NUM_PAID)
    gs = structure.to_inner_game_string_for_state(
        blind_level=sampled["blind_level"],
        stacks=sampled["stacks"],
        dealer_seat=sampled["dealer_seat"],
    )
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    starting_stacks = list(sampled["stacks"])

    for _ in range(MAX_STEPS):
        if state.is_terminal():
            break
        if state.is_chance_node():
            outs = state.chance_outcomes()
            a = rng.choices([o for o, _ in outs],
                            weights=[p for _, p in outs], k=1)[0]
            state.apply_action(int(a))
            continue
        parsed = (parse_state_repeated_6max(state)
                  if hasattr(state, "dealer_seat") else parse_state_6max(state))
        cp = parsed["current_player"]
        pol = hero_policy if cp == hero_seat else opp_policy
        chip = pol.select_action(parsed, state, rng, mode="sample")
        state.apply_action(int(chip))

    if not state.is_terminal():
        return None
    icm = icm_adjust_returns(
        chip_returns=state.returns(),
        starting_stacks=starting_stacks,
        payouts=PAYOUTS,
    )
    return float(icm[hero_seat])


def _stats(deltas):
    a = np.asarray(deltas, dtype=np.float64)
    n = len(a)
    mean = float(a.mean()) if n else 0.0
    stderr = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return n, mean, stderr


def run_member(name, opp_pol, base_seed, student, blueprint, structure,
               n_per_seat, log_every=250):
    """Returns the per-member result dict with pooled + per-seat numbers."""
    per_seat = []
    pooled_student, pooled_bp, pooled_pair = [], [], []
    t_member = time.time()

    for seat in range(NUM_SEATS):
        s_d, b_d, pair_d = [], [], []
        n_capped = 0
        seat_base = base_seed + seat * _SEAT_STRIDE
        for h in range(n_per_seat):
            hand_seed = seat_base + h
            sd = play_one_hand_fixed_seat(
                student, opp_pol, seat, structure, random.Random(hand_seed))
            bd = play_one_hand_fixed_seat(
                blueprint, opp_pol, seat, structure, random.Random(hand_seed))
            if sd is None or bd is None:
                n_capped += 1
                continue
            s_d.append(sd); b_d.append(bd); pair_d.append(sd - bd)
            if (h + 1) % log_every == 0:
                _, m, se = _stats(pair_d)
                print(f"    [{name} seat{seat}] {h+1}/{n_per_seat}  "
                      f"delta={m:+.5f}+/-{se:.5f}  capped={n_capped}",
                      flush=True)
        n, dmean, dse = _stats(pair_d)
        _, smean, _ = _stats(s_d)
        _, bmean, _ = _stats(b_d)
        dsigma = abs(dmean) / dse if dse > 0 else float("nan")
        per_seat.append({
            "seat": seat, "n": n, "n_capped": n_capped,
            "student_icm": smean, "blueprint_icm": bmean,
            "delta": dmean, "delta_stderr": dse, "delta_sigma": dsigma,
        })
        pooled_student += s_d; pooled_bp += b_d; pooled_pair += pair_d
        print(f"  [{name} seat{seat}] n={n} student={smean:+.5f} "
              f"blueprint={bmean:+.5f} delta={dmean:+.5f}+/-{dse:.5f} "
              f"sigma={dsigma:.2f}", flush=True)

    n, dmean, dse = _stats(pooled_pair)
    _, smean, _ = _stats(pooled_student)
    _, bmean, _ = _stats(pooled_bp)
    dsigma = abs(dmean) / dse if dse > 0 else float("nan")
    wall = time.time() - t_member
    print(f"==> [{name}] POOLED n={n}  student={smean:+.5f} "
          f"blueprint={bmean:+.5f}  DELTA={dmean:+.5f} +/- {dse:.5f}  "
          f"sigma={dsigma:.2f}  ({wall:.1f}s)", flush=True)
    return {
        "member": name,
        "pooled": {
            "n": n, "student_icm": smean, "blueprint_icm": bmean,
            "delta": dmean, "delta_stderr": dse, "delta_sigma": dsigma,
        },
        "per_seat": per_seat,
        "wall_seconds": wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands-per-seat", type=int, default=1000)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated member names (default all 5)")
    ap.add_argument("--seats", type=str, default=None,
                    help="comma-separated seat ids for smoke (default 0-5)")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--log-every", type=int, default=250)
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    seats = ([int(x) for x in args.seats.split(",")] if args.seats
             else list(range(NUM_SEATS)))

    print("=" * 78)
    print(f"ICM-panel floor v2 (2-arm x 6-seat paired)  "
          f"N/seat={args.hands_per_seat}  seats={seats}  "
          f"only={only or 'ALL 5'}")
    print(f"  student={STUDENT_CKPT}")
    print(f"  anchor ={ANCHOR_DIR}")
    print("=" * 78, flush=True)

    t = time.time()
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    solver = sm.load_blueprint(Path(ANCHOR_DIR), Path(ABSTRACTION_PKL))
    abstraction = solver.abstraction
    calibration = EquityCalibration.load(ARCHETYPE_CALIB_PATH)
    print(f"[load] structure+blueprint+abstraction+calib ({time.time()-t:.1f}s)",
          flush=True)

    t = time.time()
    student_net = Adaptive6MaxNet(**STUDENT_CONFIG)
    ckpt = torch.load(STUDENT_CKPT, weights_only=False, map_location="cpu")
    student_net.load_state_dict(ckpt["state_dict"])
    student_net.eval()
    student = StudentZeroContextPolicy(student_net, solver.encoder)
    blueprint = sm._BlueprintAsOpp(solver)
    blueprint.name = "blueprint_hero"
    print(f"[load] student ({sum(p.numel() for p in student_net.parameters()):,} "
          f"params) ({time.time()-t:.1f}s)", flush=True)

    panel = build_panel(solver, abstraction, calibration)
    if only is not None:
        panel = [(n, p, s) for (n, p, s) in panel if n in only]
        if not panel:
            raise SystemExit(f"no members matched --only={only}")

    results = []
    t_total = time.time()
    for name, opp, base_seed in panel:
        print(f"\n--- member: {name}  (N/seat={args.hands_per_seat}) ---",
              flush=True)
        if seats != list(range(NUM_SEATS)):
            r = run_member_seats(name, opp, base_seed, student, blueprint,
                                 structure, args.hands_per_seat, seats,
                                 args.log_every)
        else:
            r = run_member(name, opp, base_seed, student, blueprint,
                           structure, args.hands_per_seat, args.log_every)
        results.append(r)

    total_wall = time.time() - t_total
    out = {
        "challenger": "student_zero_context",
        "comparator": "blueprint(candC_k200)_paired",
        "student_ckpt": STUDENT_CKPT,
        "anchor_dir": ANCHOR_DIR,
        "abstraction_pkl": ABSTRACTION_PKL,
        "structure_yaml": STRUCTURE_YAML,
        "payouts": PAYOUTS,
        "hands_per_seat": args.hands_per_seat,
        "seats": seats,
        "pairing": "per-hand CRN: both arms reset RNG to same hand_seed",
        "total_wall_seconds": total_wall,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== DONE total_wall={total_wall:.1f}s output={args.output} ===",
          flush=True)


def run_member_seats(name, opp_pol, base_seed, student, blueprint, structure,
                     n_per_seat, seats, log_every):
    """Smoke variant of run_member that iterates only `seats`."""
    per_seat = []
    pooled_student, pooled_bp, pooled_pair = [], [], []
    t_member = time.time()
    for seat in seats:
        s_d, b_d, pair_d = [], [], []
        n_capped = 0
        seat_base = base_seed + seat * _SEAT_STRIDE
        for h in range(n_per_seat):
            hand_seed = seat_base + h
            sd = play_one_hand_fixed_seat(
                student, opp_pol, seat, structure, random.Random(hand_seed))
            bd = play_one_hand_fixed_seat(
                blueprint, opp_pol, seat, structure, random.Random(hand_seed))
            if sd is None or bd is None:
                n_capped += 1
                continue
            s_d.append(sd); b_d.append(bd); pair_d.append(sd - bd)
        n, dmean, dse = _stats(pair_d)
        _, smean, _ = _stats(s_d)
        _, bmean, _ = _stats(b_d)
        dsigma = abs(dmean) / dse if dse > 0 else float("nan")
        per_seat.append({
            "seat": seat, "n": n, "n_capped": n_capped,
            "student_icm": smean, "blueprint_icm": bmean,
            "delta": dmean, "delta_stderr": dse, "delta_sigma": dsigma,
        })
        pooled_student += s_d; pooled_bp += b_d; pooled_pair += pair_d
        print(f"  [{name} seat{seat}] n={n} student={smean:+.5f} "
              f"blueprint={bmean:+.5f} delta={dmean:+.5f}+/-{dse:.5f} "
              f"sigma={dsigma:.2f}", flush=True)
    n, dmean, dse = _stats(pooled_pair)
    _, smean, _ = _stats(pooled_student)
    _, bmean, _ = _stats(pooled_bp)
    dsigma = abs(dmean) / dse if dse > 0 else float("nan")
    wall = time.time() - t_member
    print(f"==> [{name}] POOLED(seats={seats}) n={n} student={smean:+.5f} "
          f"blueprint={bmean:+.5f} DELTA={dmean:+.5f}+/-{dse:.5f} "
          f"sigma={dsigma:.2f} ({wall:.1f}s)", flush=True)
    return {"member": name,
            "pooled": {"n": n, "student_icm": smean, "blueprint_icm": bmean,
                       "delta": dmean, "delta_stderr": dse,
                       "delta_sigma": dsigma},
            "per_seat": per_seat, "wall_seconds": wall}


if __name__ == "__main__":
    main()
