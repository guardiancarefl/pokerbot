"""Slate-2a shove-defense floor probe — EXPERIMENTAL, probe-only harness.

Pre-registration (BINDING): evals/a1_blur_map_20260612/BLUR_MAP_REPORT.txt,
final section "SLATE-2A DELIVERABLE — cheapest intervention probe for the
worst bucket (pre-registration sketch, 2026-06-12)". One scalar knob:
threshold margin tau, pre-registered tau = 0 (the pure break-even gate).
Measurements M-A, M-B, M-C in order; stop on first falsification.

THE FLOOR (deployment-time, composed-filter family; NOT wired into any
live-path file — src/nlhe/integration/live_loop.py is untouched):

  At a qualifying node — preflop, hero facing exactly ONE all-in raiser,
  to_call >= hero remaining stack, hero hand-start depth in [5, 15] bb —
  compute hero hand equity vs the FROZEN killphil shove range of the
  nearest battery cell (evals/h2_battery/battery_v1.json "ranges",
  "<shover_pos>|<depth>|<level>"), and the ICM break-even call equity
  from Malmuth-Harville fold/call EVs (the same arithmetic as
  scripts/fold_vs_shove_battery.oracle_ev, generalized to the actual
  stacks/contributions of the node). If equity < break-even + tau:
  move the policy's CALL/ALLIN mass (== all non-FOLD legal mass; at
  to_call >= stack every non-fold discrete action commits the stack)
  to FOLD. Else: leave the policy untouched (identity reference, the
  short-circuit contract shared by the deployed floor family).

CELL MAPPING for off-grid nodes (documented per the registration):
  depth  = (hero money + hero contribution) / real BB — i.e. hero
           hand-start stack in BB. Gated to the registration window
           [5, 15]; mapped to the nearest of {5, 8, 11, 15} (ties ->
           the lower depth).
  level  = blind level recovered from the real BB via the structure
           schedule; mapped to the nearest of {3, 5, 7} (ties -> the
           lower level; levels 1-2 map to 3, levels >= 8 map to 7).
  shover = "SB" when the shover seat posted a blind this hand (SB or
           BB seat per the game-string blind array), else "UTG". The
           battery froze only UTG and SB shover cells; UTG is the
           nearest (tightest) cell for any non-blind open-shover.

SCOPE GUARDS (floor is a strict no-op unless ALL hold):
  street == preflop; real BB known; to_call > 0; to_call >= hero
  remaining stack (call commits everything); exactly one opponent at
  max contribution and that opponent is ALL-IN (money == 0); every
  other non-hero seat's contribution <= BB + max ante (blind/ante
  posts only — a caller behind the shove disqualifies the node);
  hero hole cards visible; FOLD legal.

MEASUREMENTS (registration, verbatim bars in the REPORT):
  M-A  re-grade the frozen H2 battery with the floor applied to the
       policy distribution before scoring (fold_vs_shove_battery grade
       mechanics; champion ckpt b79e82dd).
  M-B  CRN-paired self-play A/B, 4,000 games, per-game seed
       2026 + 7919*g. BOTH arms run the full deployed floor chain
       (make_live_policy_filter, short_stack_threshold_bb=6.0, tail
       floor OFF = deployed default); the probe floor is the ONLY
       delta (V1 runs it after the chain). Harness pattern copied from
       scripts/tail_floor_ab.py.
  M-C  (secondary) sng_baseline killphilmtt row, 2000 games, master
       seed 2026, hero = raw checkpoint + probe floor (the baseline
       instrument's hero is the raw checkpoint, so the probe floor is
       the only delta vs the e2 post-fix row).

Usage (run from repo root, venv active):
  python -m scripts.shove_defense_probe ma --out-dir evals/a2a_shove_floor_probe_20260612
  python -m scripts.shove_defense_probe mb --games 4000 --shards 6 --shard 0 \
      --out evals/a2a_shove_floor_probe_20260612/mb_shard0.json
  python -m scripts.shove_defense_probe mc --out-dir evals/a2a_shove_floor_probe_20260612/mc
  python -m scripts.shove_defense_probe mb-combine \
      --out evals/a2a_shove_floor_probe_20260612/mb_combined.json \
      evals/a2a_shove_floor_probe_20260612/mb_shard*.json.games.jsonl
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

from src.nlhe.actions import DiscreteAction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.icm import icm_equity

BATTERY = "evals/h2_battery/battery_v1.json"
CHECKPOINT = "runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt"
ABSTRACTION = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"

PAYOUTS = [2.0, 2.0, 2.0]
A_FOLD = int(DiscreteAction.FOLD)
N_ACT = len(DiscreteAction)
DEPTH_GRID = (5, 8, 11, 15)
LEVEL_GRID = (3, 5, 7)
TAU_DEFAULT = 0.0           # pre-registered: the pure break-even gate

_DEVNULL = open(os.devnull, "w")


# --------------------------------------------------------------------------
# The probe floor
# --------------------------------------------------------------------------

def _nearest_grid(x: float, grid) -> int:
    """Nearest grid point; ties resolve to the LOWER value."""
    return min(grid, key=lambda g: (abs(g - x), g))


def _blind_info(state):
    """(sb_seat, bb_seat, ante_max) from the game-string params.
    Real-ante convention: `blind` is per-seat ("50 100 0 0 0 0" rotated),
    `ante` is a separate per-seat array. Dead-SB hands have no SB post ->
    sb_seat None."""
    try:
        params = state.get_game().get_parameters()
        blind_str = str(params.get("blind", ""))
        vals = [int(x) for x in blind_str.split()]
    except Exception:
        return None, None, 0
    if not vals or max(vals) <= 0:
        return None, None, 0
    bb = max(vals)
    bb_seat = vals.index(bb)
    sub = [(v, i) for i, v in enumerate(vals) if 0 < v < bb]
    sb_seat = sub[0][1] if len(sub) == 1 else None
    ante_max = 0
    try:
        ante_str = str(params.get("ante", ""))
        avals = [int(x) for x in ante_str.split()]
        if avals:
            ante_max = max(avals)
    except Exception:
        ante_max = 0
    return sb_seat, bb_seat, ante_max


def _icm_for(stacks, hero):
    """Mirror of fold_vs_shove_battery._hero_icm, parametrized on seat."""
    eligible = [i for i in range(len(stacks)) if stacks[i] > 0]
    if hero not in eligible:
        return 0.0
    return float(icm_equity(stacks, PAYOUTS, eligible=eligible)[hero])


def _qualify(parsed: dict, state, bb_to_level: dict):
    """Return (cell_key, hero_cards, ev_fold, ev_win, ev_lose, depth_bb)
    if the node qualifies for the shove-defense floor, else None.

    The fold/win/lose ICM arithmetic mirrors fold_vs_shove_battery.oracle_ev
    (ties folded into equity at half weight; folded-blind dead money goes
    to the winner), generalized from the battery's equal-stack probe
    construction to the node's actual stacks and contributions. On the
    battery's own spots it reproduces oracle_ev exactly.
    """
    if int(parsed.get("street_idx", -1)) != 0:
        return None
    bb = int(parsed.get("big_blind", 0))
    if bb <= 0:
        return None
    cp = parsed.get("current_player")
    m = parsed.get("money") or []
    c = parsed.get("contribution") or []
    n = len(m)
    if cp is None or n == 0 or len(c) != n:
        return None
    mx = max(c)
    to_call = mx - c[cp]
    if to_call <= 0 or m[cp] <= 0 or to_call < m[cp]:
        return None                          # call must commit hero's stack
    shovers = [j for j in range(n) if j != cp and c[j] == mx]
    if len(shovers) != 1:
        return None
    j = shovers[0]
    if m[j] != 0:
        return None                           # the raiser must be all-in
    sb_seat, bb_seat, ante_max = _blind_info(state)
    for k in range(n):
        if k in (cp, j):
            continue
        if c[k] > bb + ante_max:
            return None                       # caller behind -> disqualify
    depth_bb = float(m[cp] + c[cp]) / bb
    if not (5.0 <= depth_bb <= 15.0):
        return None                           # registration window
    lvl_actual = bb_to_level.get(bb)
    if lvl_actual is None:
        return None
    d = _nearest_grid(depth_bb, DEPTH_GRID)
    lvl = _nearest_grid(lvl_actual, LEVEL_GRID)
    pos = "SB" if (j == sb_seat or j == bb_seat) else "UTG"
    cell_key = f"{pos}|{d}|{lvl}"
    priv = parsed.get("private_cards", "") or ""
    if len(priv) != 4:
        return None
    hero_cards = (priv[0:2], priv[2:4])

    # Scenario stacks. Busted-seat placeholders (stack<=1, no post) -> 0.
    base = [0 if (m[k] + c[k]) <= 1 else int(m[k]) for k in range(n)]
    pot = int(sum(c))
    hero_total = int(c[cp] + m[cp])
    dead = int(sum(c[k] for k in range(n) if k not in (cp, j)))
    stacks_fold = list(base)
    stacks_fold[cp] = int(m[cp])
    stacks_fold[j] = pot                      # m[j] == 0; shover scoops
    stacks_win = list(base)
    stacks_win[cp] = 2 * hero_total + dead
    stacks_win[j] = int(c[j]) - hero_total    # refund (c[j] >= hero_total)
    stacks_lose = list(base)
    stacks_lose[cp] = 0
    stacks_lose[j] = int(c[j]) + hero_total + dead
    ev_fold = _icm_for(stacks_fold, cp)
    ev_win = _icm_for(stacks_win, cp)
    ev_lose = _icm_for(stacks_lose, cp)
    return cell_key, hero_cards, ev_fold, ev_win, ev_lose, depth_bb


def make_shove_defense_floor(structure, ranges: dict, tau: float = TAU_DEFAULT,
                             stats: dict | None = None,
                             eq_cache: dict | None = None):
    """Build the probe floor (same call signature + identity contract as
    the deployed floor family; accepts_d2c for chain compatibility)."""
    from scripts.fold_vs_shove_battery import _equity_vs_labels

    bb_to_level = {int(bl.big_blind): int(bl.level)
                   for bl in structure.blind_schedule}
    if stats is None:
        stats = {}
    for k in ("n_calls", "n_qualify", "n_fired"):
        stats.setdefault(k, 0)
    stats.setdefault("cells", {})
    if eq_cache is None:
        eq_cache = {}

    def floor(policy, legal_mask, parsed, state, discrete_to_chip=None):
        stats["n_calls"] += 1
        q = _qualify(parsed, state, bb_to_level)
        if q is None:
            return policy
        cell_key, hero_cards, ev_fold, ev_win, ev_lose, _depth = q
        labels = ranges.get(cell_key)
        if not labels:
            return policy
        stats["n_qualify"] += 1
        denom = ev_win - ev_lose
        if denom <= 1e-12:
            return policy
        eq_star = (ev_fold - ev_lose) / denom
        ck = (hero_cards, cell_key)
        eq = eq_cache.get(ck)
        if eq is None:
            eq = _equity_vs_labels(hero_cards, set(labels))
            eq_cache[ck] = eq
        if eq >= eq_star + tau:
            return policy                     # equity clears the gate
        if legal_mask[A_FOLD] <= 0:
            return policy
        # Move ALL non-FOLD legal mass (== CALL/ALLIN mass at these
        # nodes) to FOLD; renormalize over legal mass.
        new = np.zeros_like(np.asarray(policy, dtype=np.float64))
        legal_mass = float(sum(float(policy[i]) for i in range(len(policy))
                               if legal_mask[i] > 0))
        if legal_mass <= 0:
            return policy
        new[A_FOLD] = 1.0
        stats["n_fired"] += 1
        cstat = stats["cells"].setdefault(cell_key, [0, 0])
        cstat[1] += 1
        return new.astype(np.asarray(policy).dtype)

    # qualify-count per cell even when not fired
    _orig_qualify_bump = None  # (cell tallies updated below via wrapper)

    def floor_with_celltally(policy, legal_mask, parsed, state,
                             discrete_to_chip=None):
        q = _qualify(parsed, state, bb_to_level)
        if q is not None and ranges.get(q[0]):
            cstat = stats["cells"].setdefault(q[0], [0, 0])
            cstat[0] += 1
        return floor(policy, legal_mask, parsed, state,
                     discrete_to_chip=discrete_to_chip)

    floor_with_celltally.accepts_d2c = True
    floor_with_celltally.stats = stats
    floor_with_celltally.eq_cache = eq_cache
    return floor_with_celltally


# --------------------------------------------------------------------------
# M-A — re-grade the frozen battery with the floor ON
# --------------------------------------------------------------------------

def run_ma(out_dir: str, tau: float, limit: int | None = None):
    from src.nlhe.abstraction import Abstraction
    from scripts.eval_pool import CheckpointPolicy
    from scripts.throwaway_query_real_ante import _RealAnteStructure
    from scripts.fold_vs_shove_battery import build_facing_shove_spot
    from scripts.depth_invariance_probe import query_policy, HERO_SEAT

    battery = json.loads(Path(BATTERY).read_text())
    structure = _RealAnteStructure(
        TournamentStructure.from_yaml(battery["structure"]))
    abstr = Abstraction.load(ABSTRACTION)
    pol = CheckpointPolicy("graded", CHECKPOINT, abstr, structure)
    stats = {}
    floor = make_shove_defense_floor(structure, battery["ranges"], tau=tau,
                                     stats=stats)

    spots = battery["spots"][:limit] if limit else battery["spots"]
    losses, n_oracle_call, call_mass_sum = [], 0, 0.0
    n_fired = 0
    conf = {"fired_oracle_fold": 0, "fired_oracle_call": 0,
            "nofire_oracle_fold": 0, "nofire_oracle_call": 0,
            "not_qualified": 0}
    t0 = time.time()
    for idx, s in enumerate(spots):
        state, dealer, _bb, _sh = build_facing_shove_spot(
            structure, s["level"], s["hero_pos"], s["shover_pos"],
            tuple(s["hero_cards"]), s["depth_bb"])
        policy, mask, parsed, d2c, _f = query_policy(
            pol.solver, state, HERO_SEAT, dealer, rng_seed=0)
        before = stats["n_fired"]
        before_q = stats["n_qualify"]
        policy2 = floor(policy, mask, parsed, state, discrete_to_chip=d2c)
        fired = stats["n_fired"] > before
        qualified = stats["n_qualify"] > before_q
        if not qualified:
            conf["not_qualified"] += 1
        if fired:
            n_fired += 1
            conf["fired_oracle_call" if s["oracle"] == "call"
                 else "fired_oracle_fold"] += 1
        else:
            conf["nofire_oracle_call" if s["oracle"] == "call"
                 else "nofire_oracle_fold"] += 1
        # grade mechanics, verbatim from fold_vs_shove_battery.grade
        fold_i = A_FOLD
        p_fold = policy2[fold_i] if mask[fold_i] else 0.0
        legal_mass = float(sum(policy2[i] for i in range(len(policy2))
                               if mask[i]))
        p_call = max(0.0, legal_mass - p_fold)
        tot = p_fold + p_call
        if tot <= 0:
            continue
        p_call /= tot
        ev_policy = p_call * s["ev_call"] + (1 - p_call) * s["ev_fold"]
        ev_oracle = max(s["ev_call"], s["ev_fold"])
        losses.append(ev_oracle - ev_policy)
        n_oracle_call += (s["oracle"] == "call")
        call_mass_sum += p_call
        if (idx + 1) % 500 == 0:
            print(f"  [ma] {idx + 1}/{len(spots)} spots  "
                  f"m1so far={np.mean(losses):.4f}  fired={n_fired}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    m1 = float(np.mean(losses))
    call_mass = call_mass_sum / len(losses)
    out = {
        "measurement": "M-A",
        "registration": "slate-2a shove-defense floor probe (BLUR_MAP_REPORT)",
        "battery": battery["version"], "ckpt": CHECKPOINT, "tau": tau,
        "n_spots_graded": len(losses),
        "m1_mean_ev_loss": m1,
        "m1_se": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
        "oracle_call_rate": n_oracle_call / len(losses),
        "policy_call_mass_mean": call_mass,
        "baseline_m1": 0.08673686258402144,
        "baseline_call_mass": 0.3591451695208801,
        "m1_reduction_frac": 1.0 - m1 / 0.08673686258402144,
        "floor": {"n_fired": n_fired, "stats": {
            k: v for k, v in stats.items() if k != "cells"},
            "cells_qualified_fired": stats["cells"],
            "confusion_vs_oracle": conf},
        "bars": {
            "F1_pass": bool(m1 <= 0.0434 and 0.05 <= call_mass <= 0.15),
            "F1_m1_bar": 0.0434,
            "F1_call_mass_window": [0.05, 0.15],
            "FALSIFIED_m1_gt_0.065": bool(m1 > 0.065),
            "FALSIFIED_call_mass_gt_0.20": bool(call_mass > 0.20),
        },
        "elapsed_s": time.time() - t0,
    }
    p = Path(out_dir) / "ma_grade.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in (
        "m1_mean_ev_loss", "m1_se", "policy_call_mass_mean",
        "m1_reduction_frac", "bars")}, indent=2))
    print(f"[ma] wrote {p}")
    return out


# --------------------------------------------------------------------------
# M-B — CRN-paired self-play A/B (deployed chain both arms; probe = delta)
# --------------------------------------------------------------------------

def _make_arm_filter(chain, probe_floor, stats: dict):
    """Compose deployed chain (+ optional probe floor). Counts fires."""
    def filt(policy, legal_mask, parsed, state, discrete_to_chip=None):
        with contextlib.redirect_stdout(_DEVNULL):
            p = chain(policy, legal_mask, parsed, state,
                      discrete_to_chip=discrete_to_chip)
            if probe_floor is not None:
                p2 = probe_floor(p, legal_mask, parsed, state,
                                 discrete_to_chip=discrete_to_chip)
            else:
                p2 = p
        stats["n_calls"] += 1
        if p is not policy:
            stats["chain_fired"] += 1
        if p2 is not p:
            stats["probe_fired"] += 1
        return p2
    filt.accepts_d2c = True
    return filt


def _make_decide(hero_filter, hero_seat):
    """Drop-in for ssfab._decide sampling through the deployed path
    (pattern: scripts/tail_floor_ab.py). Opponents: raw policy."""
    from scripts.eval_6max_self_play import _sample_action_from_policy

    def _decide(solver, parsed, state, rng, hseat, threshold_bb,
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
def _patched_decide(ssfab, decide_fn):
    orig = ssfab._decide
    ssfab._decide = decide_fn
    try:
        yield
    finally:
        ssfab._decide = orig


def _mean_se_z(xs: list) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "se": None, "z": None}
    m = float(np.mean(xs))
    se = float(np.std(xs, ddof=1) / math.sqrt(n)) if n > 1 else None
    z = (m / se) if (se is not None and se > 0) else None
    return {"n": n, "mean": m, "se": se, "z": z}


def run_mb(out_path: str, tau: float, games: int, shards: int, shard: int,
           hpl: int = 5, max_hands: int = 200, starting_stack: int = 1500,
           threshold_bb: float = 6.0):
    import scripts.short_stack_floor_ab as ssfab
    from src.nlhe.integration.live_loop import make_live_policy_filter

    battery = json.loads(Path(BATTERY).read_text())
    structure = TournamentStructure.from_yaml(ssfab.STRUCTURE_YAML)
    solver = ssfab.load_solver(structure)
    hero_seat = ssfab.HERO_SEAT_DEFAULT
    eq_cache: dict = {}

    game_idxs = [i for i in range(games) if i % shards == shard]
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    games_path = Path(str(out_p) + ".games.jsonl")
    print(f"[mb] shard {shard}/{shards}: {len(game_idxs)} paired games  "
          f"tau={tau}  seeds 2026+7919*g", flush=True)

    records = []
    t0 = time.time()
    with open(games_path, "w") as fg:
        for k, g in enumerate(game_idxs):
            seed = 2026 + 7919 * g            # registration seed schedule
            arm_out = {}
            try:
                for arm in ("v0", "v1"):
                    stats = {"n_calls": 0, "chain_fired": 0,
                             "probe_fired": 0}
                    chain = make_live_policy_filter(
                        short_stack_threshold_bb=threshold_bb,
                        tail_floor_tau=None)
                    probe = (make_shove_defense_floor(
                        structure, battery["ranges"], tau=tau,
                        eq_cache=eq_cache) if arm == "v1" else None)
                    filt = _make_arm_filter(chain, probe, stats)
                    log: list = []
                    with _patched_decide(ssfab,
                                         _make_decide(filt, hero_seat)):
                        m = ssfab.play_match(
                            solver, structure, seed=seed,
                            hero_seat=hero_seat, threshold_bb=None,
                            starting_stack=starting_stack,
                            hands_per_level=hpl, max_hands=max_hands,
                            log_sink=log, game_idx=g)
                    pq = (probe.stats["n_qualify"] if probe is not None
                          else 0)
                    arm_out[arm] = (m, log, stats, pq)
            except Exception as e:
                print(f"  game {g}: FAIL {type(e).__name__}: "
                      f"{str(e)[:80]}", flush=True)
                continue
            m0, log0, s0, _ = arm_out["v0"]
            m1, log1, s1, q1 = arm_out["v1"]
            div_idx = None
            for i, (a, b) in enumerate(zip(log0, log1)):
                if a["chip"] != b["chip"]:
                    div_idx = i
                    break
            if div_idx is None and len(log0) != len(log1):
                div_idx = min(len(log0), len(log1))
            rec = {
                "game": int(g), "seed": int(seed),
                "v0_icm": float(m0["hero_icm"]),
                "v1_icm": float(m1["hero_icm"]),
                "delta": float(m1["hero_icm"] - m0["hero_icm"]),
                "diverged": div_idx is not None, "div_idx": div_idx,
                "v0_hero_decisions": len(log0),
                "v1_hero_decisions": len(log1),
                "v1_chain_fired": s1["chain_fired"],
                "v1_probe_qualified": q1,
                "v1_probe_fired": s1["probe_fired"],
                "v0_hands": m0["hands_played"],
                "v1_hands": m1["hands_played"],
            }
            records.append(rec)
            fg.write(json.dumps(rec) + "\n")
            if (k + 1) % 25 == 0:
                fg.flush()
                el = time.time() - t0
                rate = (k + 1) / el
                eta = (len(game_idxs) - k - 1) / max(rate, 1e-9)
                print(f"  [mb s{shard}] {k + 1}/{len(game_idxs)}  "
                      f"elapsed={el / 60:.1f}m  eta={eta / 60:.1f}m",
                      flush=True)
    elapsed = time.time() - t0
    summary = _mb_summary(records, tau, games, shards, shard, elapsed)
    out_p.write_text(json.dumps(summary, indent=2))
    ag = summary["all_games"]
    print(f"[mb s{shard}] done  n={len(records)}  "
          f"delta={ag['mean']:+.5f}±{ag['se'] or float('nan'):.5f}  "
          f"probe_fired_games={summary['firing']['games_with_probe_fire']}",
          flush=True)
    print(f"[out] {out_p}\n[out] {games_path}")
    return summary


def _mb_summary(records, tau, games, shards, shard, elapsed):
    deltas = [r["delta"] for r in records]
    div = [r for r in records if r["diverged"]]
    n_fired = sum(r["v1_probe_fired"] for r in records)
    n_q = sum(r["v1_probe_qualified"] for r in records)
    n_dec = sum(r["v1_hero_decisions"] for r in records)
    ag = _mean_se_z(deltas)
    return {
        "measurement": "M-B",
        "harness": "shove_defense_probe (tail_floor_ab CRN pattern)",
        "config": {"tau": tau, "games_requested": games,
                   "seed_schedule": "2026 + 7919*g",
                   "shards": shards, "shard": shard,
                   "checkpoint": CHECKPOINT,
                   "deployed_chain": "make_live_policy_filter("
                                     "short_stack_threshold_bb=6.0, "
                                     "tail_floor_tau=None) BOTH arms"},
        "games_completed": len(records),
        "all_games": ag,
        "diverged_only": _mean_se_z([r["delta"] for r in div]),
        "n_diverged": len(div),
        "firing": {
            "v1_hero_decisions": n_dec,
            "v1_probe_qualified": n_q,
            "v1_probe_fired": n_fired,
            "games_with_probe_fire":
                sum(1 for r in records if r["v1_probe_fired"] > 0),
        },
        "bars": {
            "F2_pass": (bool(ag["mean"] is not None
                             and ag["mean"] >= -0.005
                             and (ag["z"] is None or ag["z"] > -2))),
            "FALSIFIED_delta": (bool(ag["mean"] is not None
                                     and ag["mean"] < -0.01
                                     and ag["z"] is not None
                                     and ag["z"] < -2)),
        },
        "elapsed_s": elapsed,
        "sec_per_paired_game": (elapsed / len(records)) if records else None,
    }


def run_mb_combine(out_path: str, games_jsonls: list[str]):
    records = []
    for p in games_jsonls:
        with open(p) as f:
            for line in f:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["game"])
    summary = _mb_summary(records, TAU_DEFAULT, len(records), 1, 0, 0.0)
    summary["combined_from"] = games_jsonls
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("games_completed", "all_games", "diverged_only",
                       "n_diverged", "firing", "bars")}, indent=2))
    print(f"[mb-combine] wrote {out_path}")
    return summary


# --------------------------------------------------------------------------
# M-C — killphilmtt 5-seat extraction yardstick, floor ON
# --------------------------------------------------------------------------

def run_mc(out_dir: str, tau: float, games: int = 2000,
           master_seed: int = 2026):
    from src.nlhe.abstraction import Abstraction
    from scripts.eval_pool import CheckpointPolicy
    from scripts.bake_off_real_ante import build_shanky_pool
    from scripts.throwaway_query_real_ante import _RealAnteStructure
    from scripts.eval_6max_self_play import _sample_action_from_policy
    from scripts.sng_baseline import evaluate_profile

    battery = json.loads(Path(BATTERY).read_text())
    structure = _RealAnteStructure(
        TournamentStructure.from_yaml(STRUCTURE_YAML))
    abstr = Abstraction.load(ABSTRACTION)
    base_hero = CheckpointPolicy("k200", ckpt_path=CHECKPOINT,
                                 abstraction=abstr, structure=structure)
    stats = {}
    floor = make_shove_defense_floor(structure, battery["ranges"], tau=tau,
                                     stats=stats)

    class FlooredHero:
        """Baseline instrument hero (raw checkpoint sample path) with the
        probe floor as the ONLY delta vs the e2 rebaseline row."""
        name = "k200+shove_floor"

        def select_action(self, parsed, state, rng, mode="sample"):
            return _sample_action_from_policy(
                base_hero.solver, parsed, state, rng, mode=mode,
                policy_filter=floor)

    pool = build_shanky_pool("data/shanky_profiles")
    opp = next(p for p in pool if p.name == "killphilmtt")
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    games_path = out_p / "games_killphilmtt.jsonl"
    print(f"[mc] killphilmtt x {games} games  master_seed={master_seed}  "
          f"tau={tau}", flush=True)
    with open(games_path, "w") as fh:
        r = evaluate_profile(FlooredHero(), opp, structure, n_games=games,
                             master_seed=master_seed, hands_per_level=5,
                             max_hands=200, mode="sample", games_fh=fh)
    hero_net = r["hero_net_per_game"]
    extraction = -hero_net
    out = {
        "measurement": "M-C",
        "instrument": "scripts/sng_baseline.py killphilmtt row "
                      "(hero = raw ckpt + probe floor; floor is the only "
                      "delta vs the e2 post-fix baseline row)",
        "ckpt": CHECKPOINT, "tau": tau,
        "games": games, "master_seed": master_seed,
        "result": {k: r[k] for k in
                   ("n_games", "n_scored", "n_tainted", "n_capped",
                    "hero_net_per_game", "stderr", "mean_hands_per_game",
                    "elapsed_s")},
        "floor_stats": {k: v for k, v in stats.items() if k != "cells"},
        "floor_cells": stats["cells"],
        "killphil_extraction_per_game": extraction,
        "baselines": {
            "registration_standing_0.174_PREFIX": 0.174,
            "post_fix_e2_rebaseline_hero_net": -0.0800,
            "post_fix_e2_rebaseline_extraction": 0.0800,
            "post_fix_e2_rebaseline_se": 0.0223,
            "note": "registration's 0.174 is the PRE-adapter-fix row; "
                    "the post-fix e2 row (-0.0800±0.0223 hero net) is "
                    "authoritative (evals/e2_rebaseline_20260612).",
        },
        "bars": {
            "F3_pass_vs_registration_0.174": bool(extraction < 0.174),
            "F3_pass_vs_postfix_0.0800": bool(extraction < 0.0800),
            "F3_blocking": False,
        },
    }
    p = out_p / "mc_result.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("result", "killphil_extraction_per_game", "bars",
                       "floor_stats")}, indent=2))
    print(f"[mc] wrote {p}")
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ma")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--tau", type=float, default=TAU_DEFAULT)
    a.add_argument("--limit", type=int, default=None,
                   help="grade only the first N spots (smoke test)")

    b = sub.add_parser("mb")
    b.add_argument("--out", required=True)
    b.add_argument("--tau", type=float, default=TAU_DEFAULT)
    b.add_argument("--games", type=int, default=4000)
    b.add_argument("--shards", type=int, default=1)
    b.add_argument("--shard", type=int, default=0)
    b.add_argument("--hpl", type=int, default=5)
    b.add_argument("--max-hands", type=int, default=200)
    b.add_argument("--starting-stack", type=int, default=1500)

    bc = sub.add_parser("mb-combine")
    bc.add_argument("--out", required=True)
    bc.add_argument("games_jsonls", nargs="+")

    c = sub.add_parser("mc")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--tau", type=float, default=TAU_DEFAULT)
    c.add_argument("--games", type=int, default=2000)
    c.add_argument("--master-seed", type=int, default=2026)

    args = ap.parse_args()
    if args.cmd == "ma":
        run_ma(args.out_dir, args.tau, args.limit)
    elif args.cmd == "mb":
        assert 0 <= args.shard < args.shards
        run_mb(args.out, args.tau, args.games, args.shards, args.shard,
               hpl=args.hpl, max_hands=args.max_hands,
               starting_stack=args.starting_stack)
    elif args.cmd == "mb-combine":
        run_mb_combine(args.out, args.games_jsonls)
    elif args.cmd == "mc":
        run_mc(args.out_dir, args.tau, games=args.games,
               master_seed=args.master_seed)


if __name__ == "__main__":
    main()
