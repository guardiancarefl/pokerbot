"""SNG baseline: full-tournament ICM vs Shanky fields.

The official yardstick replacing the single-hand bake-off: hero (checkpoint
solver) + 5 seats of one Shanky profile play FULL double-up SNGs — blinds
escalate per the Ignition turbo schedule, busts carry between hands, the
game terminates at 3 alive (everyone left has cashed) — and the per-game
hero outcome is the ICM payout (+1 cashed / -1 busted, buy-in units).

Reuses: the A/B game loop shape (scripts/short_stack_floor_ab.play_match),
eval_pool's per-seat policy dispatch (select_action protocol shared by
CheckpointPolicy and ShankyProfilePolicy), bake_off_real_ante's pool
loading/reporting, B3's run-identity stamping discipline.

Scoring-path rules (high scrutiny):
- Per-hand records use HAND-START alive count from the carried match
  stacks (stacks[i] > 0). The per-decision money>0 expression from the
  A/B logs (which dropped mid-hand all-ins and counted busted stack=1
  placeholders) does not exist anywhere in this path.
- A hand that raises an exception taints its game: the game is logged
  with the exception (game id, hand, type, message), counted, and
  EXCLUDED from scoring. No stack mutation, no silent inclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pyspiel

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.icm import icm_equity
from src.nlhe.icm_returns import conserving_returns_for_terminal
from src.nlhe.infoset6 import parse_state_6max

N_SEATS = 6
PAYOUTS = [2.0, 2.0, 2.0]  # double-up: top-3 equal, gross buy-in units


# ── Run identity (B3 discipline) ────────────────────────────────────────

def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_identity() -> tuple[str, bool]:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True).strip()
        porcelain = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain",
             "--untracked-files=no"], text=True)
        return head, bool(porcelain.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


# ── One hand at a sampled tournament state ─────────────────────────────

def _integerize_conserving(returns):
    """Round a float return vector to integers preserving the exact total
    (largest-remainder method). universal_poker can emit half-chip returns
    on odd-chip all-in pot splits; naive int()/round() breaks chip
    conservation by +/-1. Residual whole chips go to the largest fractional
    remainders (ICM-neutral; mirrors the real-poker odd-chip rule loosely)."""
    floors = [math.floor(x) for x in returns]
    residual = int(round(sum(returns))) - sum(floors)
    out = list(floors)
    if residual:
        order = sorted(range(len(returns)),
                       key=lambda i: returns[i] - floors[i], reverse=True)
        for k in range(residual):
            out[order[k]] += 1
    return out


def _blind_guard(structure, stacks, level_idx, dealer_seat):
    """Bust seats that cannot post their forced blind (mirrors the A/B loop),
    so to_inner_game_string_for_state never produces a 0-chip blind (which
    universal_poker rejects: 'Must have a blind of at least one chip')."""
    n = N_SEATS
    bl = structure.level(level_idx)
    bb_inflated = bl.inflated_big_blind(n)
    stacks = list(stacks)
    alive = [i for i in range(n) if stacks[i] > 0]
    guard = 0
    while len(alive) > 3 and guard < n:
        d = alive.index(dealer_seat)
        sb_s = alive[(d + 1) % len(alive)]
        bb_s = alive[(d + 2) % len(alive)]
        if stacks[sb_s] >= bl.small_blind and stacks[bb_s] >= bb_inflated:
            break
        if stacks[sb_s] < bl.small_blind:
            stacks[sb_s] = 0
        if stacks[bb_s] < bb_inflated:
            stacks[bb_s] = 0
        alive = [i for i in range(n) if stacks[i] > 0]
        if len(alive) <= 3 or dealer_seat not in alive:
            return stacks, None
        guard += 1
    return stacks, dealer_seat


def play_one_hand_sng(seat_to_policy, structure, stacks, level_idx,
                       dealer_seat, rng, mode="sample"):
    """Play one hand; return new stacks. Busted seats (stack 0) become
    stack=1/ante=0 placeholders in the game string and never act. The
    terminal return vector is corrected for placeholder pot-misassignment
    (conserving_returns_for_terminal) so chips conserve among real seats."""
    blind_level = structure.level(level_idx)
    game_str = structure.to_inner_game_string_for_state(
        blind_level, stacks, dealer_seat)
    state = pyspiel.load_game(game_str).new_initial_state()

    folded = [False] * N_SEATS
    fold_order: list[int] = []
    max_steps = 500
    for _ in range(max_steps):
        if state.is_terminal():
            break
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            chosen = rng.choices([o[0] for o in outcomes],
                                  weights=[o[1] for o in outcomes], k=1)[0]
            state.apply_action(int(chosen))
            continue
        cp = state.current_player()
        if cp < 0 or stacks[cp] <= 0:
            state.apply_action(int(state.legal_actions()[0]))
            continue
        parsed = parse_state_6max(state)
        a = int(seat_to_policy[cp].select_action(parsed, state, rng, mode=mode))
        if a == 0 and not folded[cp]:
            folded[cp] = True
            fold_order.append(cp)
        state.apply_action(a)
    if not state.is_terminal():
        raise RuntimeError("hand exceeded max_steps without terminating")

    returns = conserving_returns_for_terminal(
        state, state.returns(), stacks, folded, fold_order)
    int_returns = _integerize_conserving(returns)
    new_stacks = []
    for i in range(N_SEATS):
        if stacks[i] <= 0:
            new_stacks.append(0)   # placeholder stays busted (return is 0)
        else:
            new_stacks.append(stacks[i] + int_returns[i])
    return new_stacks


# ── One full SNG game ──────────────────────────────────────────────────

def play_sng_game(hero_policy, opp_policy, structure, *, seed, hero_seat=0,
                   starting_stack=1500, hands_per_level=5, max_hands=200,
                   mode="sample", stage_acc=None, starting_stacks=None,
                   seat_to_policy=None, starting_level=1,
                   starting_dealer=None):
    """Play one full double-up SNG. Returns a per-game record dict.

    stage_acc: optional dict accumulating per-hand hero ICM-equity deltas
    keyed by (hand-start alive count, level): {key: [count, sum, sumsq]}.
    starting_stacks / seat_to_policy: test seams; production callers leave
    them None (fresh 6x starting_stack, hero + 5x opponent).
    """
    rng = random.Random(seed)
    if seat_to_policy is None:
        seat_to_policy = [hero_policy if i == hero_seat else opp_policy
                          for i in range(N_SEATS)]
    stacks = (list(starting_stacks) if starting_stacks is not None
              else [starting_stack] * N_SEATS)
    max_level = max(bl.level for bl in structure.blind_schedule)
    # starting_level: test/measurement seam (Tier-0 bubble battery starts
    # games at harvested mid-tournament states). Default 1 = v1 yardstick
    # behavior bit-for-bit.
    level = min(starting_level, max_level)
    # starting_dealer: measurement seam paired with starting_stacks — a
    # harvested mid-tournament state carries its own button. The rng draw
    # still happens unconditionally so the deal/runout stream is unchanged
    # vs the default path (seed schedule discipline). If the given dealer
    # is busted in starting_stacks, rotate forward to the next alive seat
    # (the game loop's own rotation rule).
    dealer = rng.randrange(N_SEATS)
    if starting_dealer is not None:
        dealer = int(starting_dealer)
        guard = 0
        while stacks[dealer] == 0 and guard < N_SEATS:
            dealer = (dealer + 1) % N_SEATS
            guard += 1
    hands_played = 0
    hands_in_level = 0
    rec = {"seed": seed, "tainted": False, "exception": None, "capped": False}

    while True:
        n_alive = sum(1 for s in stacks if s > 0)   # HAND-START alive count
        if n_alive <= 3:
            break
        if hands_played >= max_hands:
            rec["capped"] = True
            break
        if hands_in_level >= hands_per_level:
            level = min(level + 1, max_level)
            hands_in_level = 0

        # Bust seats that can't post their blind, then re-seat the button if
        # it landed on a now-busted seat. Prevents the 0-chip-blind SpielError.
        stacks, dealer = _blind_guard(structure, stacks, level, dealer)
        if dealer is None or sum(1 for s in stacks if s > 0) <= 3:
            break
        eligible = [i for i in range(N_SEATS) if stacks[i] > 0]
        e_before = icm_equity(stacks, PAYOUTS, eligible=eligible)
        try:
            stacks = play_one_hand_sng(seat_to_policy, structure, stacks,
                                        level, dealer, rng, mode=mode)
        except Exception as e:
            # HARD GATE: taint + log + exclude from scoring. No mutation.
            rec["tainted"] = True
            rec["exception"] = (f"hand={hands_played + 1} "
                                f"{type(e).__name__}: {str(e)[:120]}")
            break
        e_after = icm_equity(stacks, PAYOUTS, eligible=eligible)
        if stage_acc is not None:
            delta = float(e_after[hero_seat] - e_before[hero_seat])
            c = stage_acc.setdefault((n_alive, level), [0, 0.0, 0.0])
            c[0] += 1
            c[1] += delta
            c[2] += delta * delta

        hands_played += 1
        hands_in_level += 1
        dealer = (dealer + 1) % N_SEATS
        guard = 0
        while stacks[dealer] == 0 and guard < N_SEATS:
            dealer = (dealer + 1) % N_SEATS
            guard += 1

    # Per-game hero outcome, net buy-in units: +1 cashed / -1 busted at the
    # 3-alive terminal; ICM expectation (gross - 1) for capped games.
    if rec["capped"]:
        eligible = [i for i in range(N_SEATS) if stacks[i] > 0]
        hero_net = float(
            icm_equity(stacks, PAYOUTS, eligible=eligible)[hero_seat]) - 1.0
    else:
        hero_net = 1.0 if stacks[hero_seat] > 0 else -1.0

    rec.update({
        "hands": hands_played,
        "level_end": level,
        "n_alive_end": sum(1 for s in stacks if s > 0),
        "hero_net": hero_net,
    })
    return rec


# ── Per-profile evaluation ─────────────────────────────────────────────

def evaluate_profile(hero_policy, opp_policy, structure, *, n_games,
                      master_seed, hands_per_level, max_hands, mode,
                      games_fh=None, log_every=200, log=print):
    stage_acc = {}
    nets, hands_counts = [], []
    n_tainted = n_capped = 0
    exceptions = []
    t0 = time.time()
    for g in range(n_games):
        seed = master_seed + 7919 * g          # identical across profiles (CRN)
        rec = play_sng_game(hero_policy, opp_policy, structure, seed=seed,
                             hands_per_level=hands_per_level,
                             max_hands=max_hands, mode=mode,
                             stage_acc=stage_acc)
        rec["game"] = g
        if games_fh is not None:
            games_fh.write(json.dumps(rec) + "\n")
        if rec["tainted"]:
            n_tainted += 1
            exceptions.append({"game": g, "detail": rec["exception"]})
            log(f"    [EXCEPTION] game {g}: {rec['exception']}")
            continue                            # EXCLUDED from scoring
        if rec["capped"]:
            n_capped += 1
        nets.append(rec["hero_net"])
        hands_counts.append(rec["hands"])
        if (g + 1) % log_every == 0:
            m = sum(nets) / len(nets)
            log(f"    {g + 1}/{n_games} games  net/game={m:+.4f}  "
                f"[{time.time() - t0:.0f}s]")

    n = len(nets)
    mean = sum(nets) / n if n else float("nan")
    var = (sum(x * x for x in nets) / n - mean * mean) if n > 1 else float("nan")
    stderr = math.sqrt(max(0.0, var) / n) if n > 1 else float("nan")
    return {
        "n_games": n_games,
        "n_scored": n,
        "n_tainted": n_tainted,
        "n_capped": n_capped,
        "exceptions": exceptions,
        "hero_net_per_game": mean,
        "stderr": stderr,
        "sigma": abs(mean) / stderr if stderr and stderr > 0 else float("nan"),
        "mean_hands_per_game": sum(hands_counts) / n if n else float("nan"),
        "stage_acc": {f"{k[0]}|{k[1]}": v for k, v in sorted(stage_acc.items())},
        "elapsed_s": time.time() - t0,
    }


# ── Driver ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SNG baseline: full-tournament "
                                              "ICM vs Shanky fields.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--abstraction", required=True)
    ap.add_argument("--structure",
                     default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--shanky-dir", default="data/shanky_profiles")
    ap.add_argument("--profiles", default="all",
                     help="comma-separated profile names (normalized) or 'all'")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--master-seed", type=int, default=2026)
    ap.add_argument("--hands-per-level", type=int, default=5,
                     help="live-matched Ignition turbo cadence")
    ap.add_argument("--max-hands", type=int, default=200)
    ap.add_argument("--mode", default="sample", choices=["sample", "argmax"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.nlhe.abstraction import Abstraction
    from scripts.eval_pool import CheckpointPolicy
    from scripts.bake_off_real_ante import build_shanky_pool
    from scripts.throwaway_query_real_ante import _RealAnteStructure

    def log(msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)

    head, dirty = _git_identity()
    header = {
        "record_type": "run_header",
        "ckpt_path": str(Path(args.ckpt).resolve()),
        "ckpt_sha256": _sha256_of_file(args.ckpt),
        "abstraction_sha256": _sha256_of_file(args.abstraction),
        "config": {k: v for k, v in vars(args).items()},
        "master_seed": args.master_seed,
        "git_head": head,
        "git_dirty": dirty,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    log(f"run header: {json.dumps(header)}")

    structure = _RealAnteStructure(TournamentStructure.from_yaml(args.structure))
    abstr = Abstraction.load(args.abstraction)
    hero = CheckpointPolicy(name="k200", ckpt_path=args.ckpt,
                             abstraction=abstr, structure=structure)
    pool = build_shanky_pool(args.shanky_dir)
    if args.profiles != "all":
        wanted = {p.strip() for p in args.profiles.split(",")}
        pool = [p for p in pool if p.name in wanted]
        missing = wanted - {p.name for p in pool}
        if missing:
            raise SystemExit(f"unknown profiles: {sorted(missing)}")
    log(f"{len(pool)} profile(s), {args.games} games each, "
        f"hpl={args.hands_per_level}, master_seed={args.master_seed}")

    results = []
    for i, opp in enumerate(pool):
        log(f"[{i + 1}/{len(pool)}] {opp.name}")
        games_path = out_dir / f"games_{opp.name}.jsonl"
        with open(games_path, "w") as fh:
            r = evaluate_profile(hero, opp, structure, n_games=args.games,
                                  master_seed=args.master_seed,
                                  hands_per_level=args.hands_per_level,
                                  max_hands=args.max_hands, mode=args.mode,
                                  games_fh=fh, log_every=args.log_every,
                                  log=log)
        r["opponent"] = f"shanky:{opp.name}"
        results.append(r)
        log(f"  {opp.name}: net/game={r['hero_net_per_game']:+.4f} "
            f"+/- {r['stderr']:.4f}  hands/game={r['mean_hands_per_game']:.1f} "
            f"tainted={r['n_tainted']} capped={r['n_capped']} "
            f"[{r['elapsed_s']:.0f}s]")

    summary = dict(header)
    summary["record_type"] = "summary"
    summary["results"] = results
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_path}")

    log("SUMMARY (ascending hero net/game):")
    for r in sorted(results, key=lambda r: r["hero_net_per_game"]):
        log(f"  {r['opponent']:<32s} {r['hero_net_per_game']:+8.4f} "
            f"+/- {r['stderr']:.4f}  hands={r['mean_hands_per_game']:.1f} "
            f"tainted={r['n_tainted']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
