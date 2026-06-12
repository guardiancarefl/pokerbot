"""TG3 harness (EXP_H1, H1.2 re-registration): CRN-paired A/B of the
DEPLOYED floor chain WITH vs WITHOUT the commitment-scaled tail floor.

  V0: full deployed floor chain via
      make_live_policy_filter(short_stack_threshold_bb=6.0)
      (AA/KK preflop + check-when-free + short-stack; tail floor OFF)
  V1: identical chain with tail_floor_tau=TAU (default 0.10)

This differs from scripts/short_stack_floor_ab.py, whose V0 arm is the RAW
policy (no live floors) and whose V1 floor is a script-local short-stack
mask. Here BOTH arms run the full deployed chain; the tail floor is the
only delta (review disposition A3 / H1.2 registration).

Reuse vs reimplementation:
  - REUSED by import from scripts.short_stack_floor_ab (ssfab):
    load_solver (ckpt/abstraction shas + feature_dim assert), play_match /
    play_one_hand (game loop, blind escalation incl. hands-per-level,
    dealer rotation, bust handling, ICM terminal accounting), seeding
    scheme (per-game seed = base_seed + game_idx, same seed both arms),
    paired-delta structure of play_paired_game.
  - NEW here: the decision function. ssfab._decide samples with its own
    inline code (script-local floor, deterministic per-decision encoder
    rng). The H1.2 registration requires the deployed sampling path, so
    we temporarily swap ssfab._decide (save/restore contextmanager) for
    one that calls scripts.eval_6max_self_play._sample_action_from_policy
    with policy_filter=make_live_policy_filter(...) — the literal live
    code path, including the `accepts_d2c` discrete_to_chip dispatch that
    gives the tail floor exact chip commitment costs.
  - NEW here: tail-firing stats (chain run with tail OFF and ON per hero
    decision; pure functions, outputs differ iff tail fired), per-game
    divergence detection (first hero sampled chip-action difference),
    all-games / diverged-only mean±SE+z, --shards/--shard support.

CLI (per H1.2):
  python -m scripts.tail_floor_ab --games 24000 --base-seed 1 --hpl 5 \
      --tau 0.10 --out evals/h1_tail_floor_20260612/tg3_shardK.json \
      --shards 8 --shard K

--tau <= 0 disables the tail floor in BOTH arms (V0-identity gate: every
paired delta must be exactly 0.0 and zero games may diverge).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import scripts.short_stack_floor_ab as ssfab
from scripts.eval_6max_self_play import _sample_action_from_policy
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.live_loop import make_live_policy_filter

HERO_SEAT = ssfab.HERO_SEAT_DEFAULT

# The composed live filter prints "[FLOOR] fired=..." on every fire; at
# hpl=5 the short-stack floor fires constantly — swallow the chatter.
_DEVNULL = open(os.devnull, "w")


# --------------------------------------------------------------------------
# Hero filter with tail-firing stats
# --------------------------------------------------------------------------

def make_stat_filter(threshold_bb: float, tau, stats: dict):
    """Wrap the deployed composed filter; count any-floor / tail-floor fires.

    All four floors are pure deterministic functions of (policy, legal_mask,
    parsed, state, d2c), so running the chain with tail OFF and tail ON on
    the same inputs isolates the tail floor: outputs differ iff it fired.
    The returned (sampled-from) policy is always the tail-ON chain output
    when tau is set, so V1 behavior is exactly the deployed armed chain.
    """
    chain_off = make_live_policy_filter(
        short_stack_threshold_bb=threshold_bb, tail_floor_tau=None)
    chain_on = (make_live_policy_filter(
        short_stack_threshold_bb=threshold_bb, tail_floor_tau=tau)
        if tau is not None else None)

    def filt(policy, legal_mask, parsed, state, discrete_to_chip=None):
        with contextlib.redirect_stdout(_DEVNULL):
            p_off = chain_off(policy, legal_mask, parsed, state,
                              discrete_to_chip=discrete_to_chip)
            if chain_on is None:
                p_final = p_off
                tail_fired = False
            else:
                p_final = chain_on(policy, legal_mask, parsed, state,
                                   discrete_to_chip=discrete_to_chip)
                tail_fired = not np.array_equal(np.asarray(p_off),
                                                np.asarray(p_final))
        stats["n_calls"] += 1
        if p_final is not policy:
            stats["any_fired"] += 1
        if tail_fired:
            stats["tail_fired"] += 1
        return p_final

    filt.accepts_d2c = True   # opt into exact-commit d2c dispatch
    return filt


# --------------------------------------------------------------------------
# Deployed-path decide, patched into ssfab's game loop
# --------------------------------------------------------------------------

def _make_decide(hero_filter):
    """Build a drop-in replacement for ssfab._decide that samples through
    the deployed path. Opponents: raw policy (no filter), identical in
    both arms. Hero: the live composed chain. log_sink records the hero
    sampled chip ints for CRN divergence detection."""
    def _decide(solver, parsed, state, rng, hero_seat, threshold_bb,
                log_sink, game_idx, hand_idx, blind_level_idx, dealer_seat):
        cp = parsed["current_player"]
        pf = hero_filter if cp == hero_seat else None
        chip = _sample_action_from_policy(
            solver, parsed, state, rng, mode="sample", policy_filter=pf)
        if cp == hero_seat and log_sink is not None:
            log_sink.append({"hand": int(hand_idx), "chip": int(chip)})
        return int(chip)
    return _decide


@contextlib.contextmanager
def _patched_decide(decide_fn):
    orig = ssfab._decide
    ssfab._decide = decide_fn
    try:
        yield
    finally:
        ssfab._decide = orig


# --------------------------------------------------------------------------
# Paired game
# --------------------------------------------------------------------------

def play_paired_game_tail(solver, structure, *, seed: int, tau,
                          threshold_bb: float, starting_stack: int,
                          hands_per_level: int, max_hands: int,
                          game_idx: int) -> dict:
    """One CRN pair: same seed both arms; V0 tail OFF, V1 tail=tau."""
    results = {}
    for arm, arm_tau in (("v0", None), ("v1", tau)):
        stats = {"n_calls": 0, "any_fired": 0, "tail_fired": 0}
        log: list = []
        filt = make_stat_filter(threshold_bb, arm_tau, stats)
        with _patched_decide(_make_decide(filt)):
            m = ssfab.play_match(
                solver, structure, seed=seed, hero_seat=HERO_SEAT,
                threshold_bb=None,   # ssfab's own floor unused (our _decide ignores it)
                starting_stack=starting_stack,
                hands_per_level=hands_per_level, max_hands=max_hands,
                log_sink=log, game_idx=game_idx)
        results[arm] = (m, log, stats)

    m0, log0, s0 = results["v0"]
    m1, log1, s1 = results["v1"]

    # Diverged-game = first hero sampled-action (chip int) difference.
    div_idx = None
    for i, (a, b) in enumerate(zip(log0, log1)):
        if a["chip"] != b["chip"]:
            div_idx = i
            break
    if div_idx is None and len(log0) != len(log1):
        # Defensive: shouldn't happen without a chip difference (identical
        # action streams ⇒ identical games), but count it as divergence.
        div_idx = min(len(log0), len(log1))
    diverged = div_idx is not None
    div_hand = None
    if diverged:
        src = log1 if div_idx < len(log1) else log0
        if div_idx < len(src):
            div_hand = src[div_idx]["hand"]

    return {
        "game": int(game_idx), "seed": int(seed), "hero_seat": HERO_SEAT,
        "v0_icm": float(m0["hero_icm"]), "v1_icm": float(m1["hero_icm"]),
        "delta": float(m1["hero_icm"] - m0["hero_icm"]),
        "diverged": bool(diverged),
        "div_idx": div_idx, "div_hand": div_hand,
        "v0_hero_decisions": len(log0), "v1_hero_decisions": len(log1),
        "v0_any_fired": s0["any_fired"],
        "v1_any_fired": s1["any_fired"],
        "v1_tail_fired": s1["tail_fired"],
        "v0_hands": m0["hands_played"], "v1_hands": m1["hands_played"],
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _mean_se_z(xs: list) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "se": None, "z": None}
    m = float(np.mean(xs))
    se = float(np.std(xs, ddof=1) / math.sqrt(n)) if n > 1 else None
    z = (m / se) if (se is not None and se > 0) else None
    return {"n": n, "mean": m, "se": se, "z": z}


def summarize(records: list, args, elapsed_s: float) -> dict:
    deltas = [r["delta"] for r in records]
    div = [r for r in records if r["diverged"]]
    n_dec_v1 = sum(r["v1_hero_decisions"] for r in records)
    n_tail = sum(r["v1_tail_fired"] for r in records)
    n_any_v1 = sum(r["v1_any_fired"] for r in records)
    return {
        "harness": "tail_floor_ab",
        "registration": "EXP_H1 H1.2 / TG3",
        "config": {
            "games_requested": args.games, "base_seed": args.base_seed,
            "shards": args.shards, "shard": args.shard,
            "tau": (args.tau if args.tau > 0 else None),
            "threshold_bb": args.threshold_bb, "hpl": args.hpl,
            "max_hands": args.max_hands,
            "starting_stack": args.starting_stack,
            "checkpoint": ssfab.CHECKPOINT,
            "abstraction": ssfab.ABSTRACTION,
            "structure": ssfab.STRUCTURE_YAML,
        },
        "games_completed": len(records),
        "all_games": _mean_se_z(deltas),
        "diverged_only": _mean_se_z([r["delta"] for r in div]),
        "n_diverged": len(div),
        "diverged_frac": (len(div) / len(records)) if records else None,
        "firing": {
            "v1_hero_decisions": n_dec_v1,
            "v1_tail_fired_decisions": n_tail,
            "v1_tail_firing_rate": (n_tail / n_dec_v1) if n_dec_v1 else None,
            "v1_any_floor_fired_decisions": n_any_v1,
            "games_with_tail_fire":
                sum(1 for r in records if r["v1_tail_fired"] > 0),
        },
        "elapsed_s": elapsed_s,
        "sec_per_paired_game": (elapsed_s / len(records)) if records else None,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=24000,
                    help="total games across all shards (seeds base..base+N-1)")
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--hpl", type=int, default=5,
                    help="hands per blind level (live-matched escalation = 5)")
    ap.add_argument("--tau", type=float, default=0.10,
                    help="tail floor tau_max for V1; <=0 disables it in "
                         "BOTH arms (V0-identity gate)")
    ap.add_argument("--out", type=str, required=True,
                    help="summary JSON path; per-game records go to "
                         "<out>.games.jsonl")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-hands", type=int, default=200)
    ap.add_argument("--starting-stack", type=int, default=1500)
    ap.add_argument("--threshold-bb", type=float, default=6.0)
    args = ap.parse_args()
    assert 0 <= args.shard < args.shards

    print(f"[args] {vars(args)}", flush=True)
    tau = args.tau if args.tau > 0 else None
    if tau is None:
        print("[mode] tau disabled — V0-IDENTITY GATE (both arms tail OFF)",
              flush=True)

    structure = TournamentStructure.from_yaml(ssfab.STRUCTURE_YAML)
    solver = ssfab.load_solver(structure)

    game_idxs = [i for i in range(args.games) if i % args.shards == args.shard]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    games_path = Path(str(out_path) + ".games.jsonl")

    print(f"[run] shard {args.shard}/{args.shards}: {len(game_idxs)} paired "
          f"games  tau={tau}  hpl={args.hpl}  out={out_path}", flush=True)
    records = []
    t0 = time.time()
    with open(games_path, "w") as fg:
        for k, i in enumerate(game_idxs):
            seed = args.base_seed + i
            try:
                rec = play_paired_game_tail(
                    solver, structure, seed=seed, tau=tau,
                    threshold_bb=args.threshold_bb,
                    starting_stack=args.starting_stack,
                    hands_per_level=args.hpl, max_hands=args.max_hands,
                    game_idx=i)
            except Exception as e:
                print(f"  game {i}: FAIL {type(e).__name__}: {str(e)[:80]}",
                      flush=True)
                continue
            records.append(rec)
            fg.write(json.dumps(rec) + "\n")
            if (k + 1) % 50 == 0:
                fg.flush()
                el = time.time() - t0
                rate = (k + 1) / el
                eta = (len(game_idxs) - k - 1) / max(rate, 1e-9)
                print(f"  game {k+1}/{len(game_idxs)}  elapsed={el/60:.1f}m  "
                      f"rate={rate:.2f}g/s  eta={eta/60:.1f}m", flush=True)
    elapsed = time.time() - t0

    summary = summarize(records, args, elapsed)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    ag = summary["all_games"]
    dg = summary["diverged_only"]
    fr = summary["firing"]
    print(f"\n[summary] games={summary['games_completed']}  "
          f"diverged={summary['n_diverged']} "
          f"({100.0*(summary['diverged_frac'] or 0):.1f}%)", flush=True)
    print(f"  all-games   delta = {ag['mean']:+.5f} ± "
          f"{(ag['se'] if ag['se'] is not None else float('nan')):.5f}  "
          f"z={(ag['z'] if ag['z'] is not None else float('nan')):+.2f}",
          flush=True)
    if dg["n"] > 0:
        print(f"  diverged-only delta = {dg['mean']:+.5f} ± "
              f"{(dg['se'] if dg['se'] is not None else float('nan')):.5f}  "
              f"z={(dg['z'] if dg['z'] is not None else float('nan')):+.2f}",
              flush=True)
    print(f"  tail fired {fr['v1_tail_fired_decisions']}/"
          f"{fr['v1_hero_decisions']} hero decisions "
          f"({100.0*(fr['v1_tail_firing_rate'] or 0):.2f}%)  "
          f"games-with-fire={fr['games_with_tail_fire']}", flush=True)
    print(f"  wall {elapsed/60:.1f}m  "
          f"{summary['sec_per_paired_game']:.2f}s/paired-game", flush=True)
    print(f"[out] {out_path}\n[out] {games_path}", flush=True)


if __name__ == "__main__":
    main()
