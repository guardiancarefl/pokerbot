"""Match-length probe: simulate Ignition 6-max Double-Up Turbo to natural
3-left bubble termination. Measures hands-to-first-bust and hands-to-3-left.

JOB 1 of the Phase-2 go/no-go gate. The corpus's 150-hands-per-match is a
GENERATION parameter (one-hand-at-a-time with persistent observer, stacks do
NOT carry over), not a real match length. This probe simulates the real
turbo blind escalation and stack dynamics.

Setup:
  - configs/ignition_double_up_6max_turbo.yaml — turbo blind schedule
  - starting_stack = 1500 chips (15bb at level 1's bb=25)
  - hands_per_level = 3 (real Ignition turbo, per game_strings.py:454 comment)
  - dealer rotates over alive seats each hand
  - All six seats play blueprint (self-play)
  - Match terminates when only 3 players remain (Ignition Double-Up bubble)

Per-match metrics:
  - hands_until_first_bust  (6 -> 5 players)
  - hands_until_three_left  (match end)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyspiel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nlhe.actions import DiscreteAction  # noqa: E402
from src.nlhe.game_strings import TournamentStructure  # noqa: E402
from src.nlhe.infoset6 import parse_state_6max  # noqa: E402
from src.nlhe.cfr6 import _build_view_6max, discretize_legal_actions  # noqa: E402

from scripts.six_max_adaptive_smoke import (  # noqa: E402
    blueprint_policy, load_blueprint, NUM_SEATS,
)


def play_one_hand(solver, structure: TournamentStructure, stacks: list[int],
                  level: int, dealer_seat: int, rng: random.Random,
                  policy: str = "blueprint"
                  ) -> tuple[list[int], list[int], list[int]]:
    """Play one hand. Returns (new_stacks, chip_deltas, blinded_out_seats).

    Busted seats (stack=0) get a stack=1 placeholder per
    to_inner_game_string_for_state's contract; they sit dealt cards with
    zero meaningful action. We force-fold any decision they might be asked
    to make.

    Pre-hand: any alive seat whose stack < small_blind, OR whose stack
    happens to fall on the BB position but stack < bb_inflated, is force-
    eliminated. (Real poker would auto-all-in for less than blind; OpenSpiel
    universal_poker rejects stack < blind so we approximate by elimination.
    Stack < SB is functionally dead in turbo blinds anyway.)"""
    blind_level = structure.level(level)
    n = structure.num_players
    sb = blind_level.small_blind
    bb_inflated = blind_level.inflated_big_blind(n)

    stacks = list(stacks)
    blinded_out = [i for i in range(n) if 0 < stacks[i] < sb]
    for i in blinded_out:
        stacks[i] = 0

    alive_seats = [i for i, s in enumerate(stacks) if s > 0]
    if len(alive_seats) <= 3:
        return stacks, [0] * n, blinded_out
    if dealer_seat not in alive_seats:
        i = dealer_seat
        while i not in alive_seats:
            i = (i + 1) % n
        dealer_seat = i

    # The BB seat is determined by alive-seat rotation
    # (to_inner_game_string_for_state line 401-404). Iterate: any time the
    # BB or SB position can't cover their required blind, kill them and
    # recompute alive seats / dealer. Loop until both blind positions can
    # cover.
    def sb_bb_seats_for_dealer(d, alive):
        dpos = alive.index(d)
        return alive[(dpos + 1) % len(alive)], alive[(dpos + 2) % len(alive)]
    guard_iter = 0
    while len(alive_seats) > 3 and guard_iter < n:
        sb_s, bb_s = sb_bb_seats_for_dealer(dealer_seat, alive_seats)
        if stacks[sb_s] >= sb and stacks[bb_s] >= bb_inflated:
            break
        if stacks[sb_s] < sb:
            stacks[sb_s] = 0
            blinded_out.append(sb_s)
        if stacks[bb_s] < bb_inflated:
            stacks[bb_s] = 0
            blinded_out.append(bb_s)
        alive_seats = [i for i, s in enumerate(stacks) if s > 0]
        if len(alive_seats) <= 3:
            return stacks, [0] * n, blinded_out
        if dealer_seat not in alive_seats:
            i = dealer_seat
            while i not in alive_seats:
                i = (i + 1) % n
            dealer_seat = i
        guard_iter += 1

    game_str = structure.to_inner_game_string_for_state(
        blind_level, stacks, dealer_seat)
    g = pyspiel.load_game(game_str)
    state = g.new_initial_state()

    # play to terminal
    while not state.is_terminal():
        if state.is_chance_node():
            outs = state.chance_outcomes()
            a = rng.choices([o for o, _ in outs],
                            weights=[p for _, p in outs], k=1)[0]
            state.apply_action(int(a))
            continue
        cur = state.current_player()
        if cur < 0:
            break
        # Busted seat with stack=1 placeholder: force fold (action 0)
        if stacks[cur] <= 0:
            legal = state.legal_actions()
            # FOLD is the safest choice when available; otherwise call min
            state.apply_action(int(legal[0]))
            continue
        if policy == "random":
            # Uniform random over OpenSpiel's raw legal actions
            legal = state.legal_actions()
            state.apply_action(int(rng.choice(legal)))
            continue
        # blueprint policy
        parsed = parse_state_6max(state)
        probs, legal_mask, d_to_chip = blueprint_policy(solver, parsed, state)
        keys = np.flatnonzero(legal_mask).tolist()
        ws = [float(probs[k]) for k in keys]
        total = sum(ws)
        if total <= 0:
            choice = rng.choice(keys)
        else:
            choice = rng.choices(keys, weights=ws, k=1)[0]
        chip = d_to_chip.get(DiscreteAction(int(choice)))
        if chip is None:
            chip = next(iter(d_to_chip.values()))
        state.apply_action(int(chip))

    returns = list(state.returns())
    new_stacks = list(stacks)
    for i in range(n):
        new_stacks[i] = max(0, stacks[i] + int(returns[i]))
    # Any seat that started this hand busted (stack=0) gets the placeholder
    # chip 1 stripped — they should stay at 0
    for i in range(n):
        if stacks[i] == 0:
            new_stacks[i] = 0
    return new_stacks, [int(returns[i]) for i in range(n)], blinded_out


def play_match(solver, structure: TournamentStructure, *,
               starting_stack: int, hands_per_level: int,
               max_hands: int, seed: int, policy: str = "blueprint"
               ) -> dict:
    """Play one match to 3-left or max_hands fallback.
    Returns dict with hands_to_first_bust, hands_to_three_left,
    n_alive_at_end, level_at_end, terminated_by_cap."""
    rng = random.Random(seed)
    n = structure.num_players
    stacks = [starting_stack] * n
    level = 1
    max_level = max(bl.level for bl in structure.blind_schedule)
    dealer = rng.randrange(n)
    hands_played = 0
    hands_in_level = 0
    first_bust_hand = None

    while True:
        n_alive = sum(1 for s in stacks if s > 0)
        if n_alive <= 3:
            return {
                "hands_to_first_bust": first_bust_hand,
                "hands_to_three_left": hands_played,
                "n_alive_at_end": n_alive,
                "level_at_end": level,
                "terminated_by_cap": False,
                "final_stacks": stacks,
            }
        if hands_played >= max_hands:
            return {
                "hands_to_first_bust": first_bust_hand,
                "hands_to_three_left": None,
                "n_alive_at_end": n_alive,
                "level_at_end": level,
                "terminated_by_cap": True,
                "final_stacks": stacks,
            }
        if hands_in_level >= hands_per_level:
            level = min(level + 1, max_level)
            hands_in_level = 0

        try:
            stacks, _deltas, _blinded = play_one_hand(
                solver, structure, stacks, level, dealer, rng, policy=policy)
        except Exception as e:
            # Defensive: should not happen after the blinded-out guards,
            # but if universal_poker rejects, treat all sub-bb players as
            # blinded out and continue.
            bl = structure.level(level)
            for i in range(n):
                if stacks[i] > 0 and stacks[i] < bl.inflated_big_blind(n):
                    stacks[i] = 0
            print(f"  [warn] hand {hands_played+1} engine error: {e}; "
                  f"flushed sub-BB stacks", flush=True)
        hands_played += 1
        hands_in_level += 1

        n_alive_after = sum(1 for s in stacks if s > 0)
        if first_bust_hand is None and n_alive_after < n:
            first_bust_hand = hands_played

        # rotate dealer to next alive seat
        dealer = (dealer + 1) % n
        while stacks[dealer] == 0:
            dealer = (dealer + 1) % n


def histogram_5(values: list[int]) -> dict[str, int]:
    """5-hand-wide histogram. Returns dict like '0-4': 12, '5-9': 7, ..."""
    if not values:
        return {}
    h = Counter()
    for v in values:
        bucket = (v // 5) * 5
        h[f"{bucket}-{bucket + 4}"] = h.get(f"{bucket}-{bucket + 4}", 0) + 1
    # sort by bucket start
    return dict(sorted(h.items(), key=lambda kv: int(kv[0].split("-")[0])))


def summarize(values, label):
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0:
        return {"n": 0, "label": label}
    return {
        "label": label,
        "n": int(len(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p10": float(np.percentile(arr, 10)),
        "p5": float(np.percentile(arr, 5)),
        "p1": float(np.percentile(arr, 1)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "histogram_5_hand_buckets": histogram_5(list(arr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-matches", type=int, default=200)
    ap.add_argument("--starting-stack", type=int, default=1500)
    ap.add_argument("--hands-per-level", type=int, default=3,
                    help="Real Ignition turbo per game_strings.py:454 comment")
    ap.add_argument("--max-hands", type=int, default=500,
                    help="Safety fallback (well above natural 3-left)")
    ap.add_argument("--structure-yaml",
                    default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--anchor-dir",
                    default="runs/six_max_20260530_034023_phase4f_dcfr_candC_k200")
    ap.add_argument("--abstraction-pkl",
                    default="runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--policy", choices=["blueprint", "random"],
                    default="blueprint",
                    help="random = uniform over OpenSpiel legal_actions "
                         "(policy-independent match-length floor)")
    ap.add_argument("--out-json",
                    default="runs/phase1_d128_repro/match_length_probe.json")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] tournament structure {args.structure_yaml}")
    structure = TournamentStructure.from_yaml(args.structure_yaml)
    print(f"        num_players={structure.num_players}  "
          f"starting_chips={structure.starting_chips}  "
          f"num_paid={structure.num_paid()}  "
          f"levels={len(structure.blind_schedule)}")

    if args.policy == "blueprint":
        print(f"[load] blueprint solver")
        solver = load_blueprint(Path(args.anchor_dir),
                                 Path(args.abstraction_pkl))
    else:
        print(f"[policy] random-uniform (no blueprint solver loaded)")
        solver = None

    print(f"[sim] running {args.n_matches} matches  "
          f"policy={args.policy}  "
          f"starting_stack={args.starting_stack}  "
          f"hands_per_level={args.hands_per_level}  "
          f"max_hands={args.max_hands}")

    per_match = []
    t_sim = time.time()
    for m in range(args.n_matches):
        t_m = time.time()
        r = play_match(solver, structure,
                       starting_stack=args.starting_stack,
                       hands_per_level=args.hands_per_level,
                       max_hands=args.max_hands,
                       seed=args.seed + m * 1009,
                       policy=args.policy)
        r["match_idx"] = m
        r["wall_seconds"] = float(time.time() - t_m)
        per_match.append(r)
        if (m + 1) % 5 == 0:
            elapsed = time.time() - t_sim
            est_total = elapsed * args.n_matches / (m + 1)
            print(f"  [{m+1}/{args.n_matches}] "
                  f"first_bust={r['hands_to_first_bust']} "
                  f"three_left={r['hands_to_three_left']} "
                  f"level={r['level_at_end']} "
                  f"alive={r['n_alive_at_end']} "
                  f"wall_match={r['wall_seconds']:.1f}s  "
                  f"elapsed={elapsed/60:.1f}min  est_total={est_total/60:.1f}min")

    first_bust = [r["hands_to_first_bust"] for r in per_match]
    three_left = [r["hands_to_three_left"] for r in per_match]

    n_capped = sum(1 for r in per_match if r["terminated_by_cap"])
    print(f"\n[stats] n_matches={args.n_matches}  "
          f"capped_at_max_hands={n_capped}  "
          f"total_wall={(time.time()-t0)/60:.1f}min")

    s_fb = summarize(first_bust, "hands_to_first_bust")
    s_tl = summarize(three_left, "hands_to_three_left")

    for s in (s_fb, s_tl):
        print(f"\n=== {s['label']} (n={s['n']}) ===")
        if s["n"] == 0:
            continue
        print(f"  median = {s['median']}  p25 = {s['p25']}  "
              f"p10 = {s['p10']}  p5 = {s['p5']}  p1 = {s['p1']}")
        print(f"  min = {s['min']}  max = {s['max']}  mean = {s['mean']:.2f}")
        print(f"  histogram (5-hand buckets):")
        for k, v in s["histogram_5_hand_buckets"].items():
            bar = "#" * v
            print(f"    {k:>10s}  {v:>4d}  {bar}")

    payload = {
        "config": {
            "n_matches": int(args.n_matches),
            "policy": args.policy,
            "starting_stack": int(args.starting_stack),
            "hands_per_level": int(args.hands_per_level),
            "max_hands": int(args.max_hands),
            "structure_yaml": args.structure_yaml,
            "anchor_dir": args.anchor_dir,
            "abstraction_pkl": args.abstraction_pkl,
            "seed": int(args.seed),
        },
        "n_capped_at_max_hands": int(n_capped),
        "wall_seconds_total": float(time.time() - t0),
        "summary_hands_to_first_bust": s_fb,
        "summary_hands_to_three_left": s_tl,
        "per_match": [{k: v for k, v in r.items() if k != "final_stacks"}
                      for r in per_match],
        "level_at_end_dist": dict(Counter(r["level_at_end"]
                                          for r in per_match)),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[saved] {out}")
    print(f"[done] total wall = {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
