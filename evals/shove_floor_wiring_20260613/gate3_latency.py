"""Gate 3 — per-decision latency when the floor FIRES.

Measures wall time of ShoveDefenseFloor.apply on real facing-shove decision
nodes — the exact deployment hook the live filter calls. The dominant cost is
the equity-vs-range Monte Carlo, memoized per (hero_cards, cell_key). We time:

  - COLD : the first apply for a fresh (hero_cards, cell) — runs the MC. This
           is the live worst case (a hero hand never seen this cell before).
  - WARM : a repeat apply for an already-cached (hero_cards, cell) — the
           memoized path that dominates a real session once a cell warms up.

Per-decision wall time = the full apply() (qualify + ICM + equity + collapse),
not just the MC, because that is what sits on the live decision path. We sample
distinct battery spots so each COLD timing pays a real, uncached MC.

GATE: p99 of the COLD path < 1.0 s (the live per-decision budget). This is the
number that gates live use — a single decision must clear the click deadline
even on a cache miss.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.nlhe.actions import DiscreteAction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.integration.live_loop import load_shove_defense_floor
from scripts.fold_vs_shove_battery import build_facing_shove_spot
from scripts.depth_invariance_probe import HERO_SEAT
from scripts.throwaway_query_real_ante import _RealAnteStructure

BATTERY = REPO / "evals/h2_battery/battery_v1.json"
N_ACT = len(DiscreteAction)
A_FOLD = int(DiscreteAction.FOLD)
A_CALL = int(DiscreteAction.CALL)
A_ALLIN = int(DiscreteAction.ALLIN)
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def _pm():
    p = np.zeros(N_ACT, dtype=np.float32)
    p[A_FOLD], p[A_CALL], p[A_ALLIN] = 0.30, 0.30, 0.40
    m = np.zeros(N_ACT, dtype=np.float32)
    m[A_FOLD] = m[A_CALL] = m[A_ALLIN] = 1.0
    return p, m


def _pct(xs, q):
    return float(np.percentile(np.array(xs), q)) if xs else float("nan")


def main():
    battery = json.loads(BATTERY.read_text())
    structure = _RealAnteStructure(
        TournamentStructure.from_yaml(battery["structure"]))
    # Fresh floor (empty eq_cache) so the first apply per cell is genuinely
    # cold. A single shared floor across the session is exactly the live wiring
    # (loaded once); we reuse it so the WARM timings hit the live cache.
    floor = load_shove_defense_floor(range_table_path=str(BATTERY), tau=0.0,
                                     structure=floor_struct(structure))

    spots = battery["spots"][:N_SAMPLE]
    # Pre-build states so build cost is OUT of the timed window (build is a
    # test harness artifact; live frames arrive already built from the
    # scraper replay).
    built = []
    for s in spots:
        state, _d, _bb, _sh = build_facing_shove_spot(
            structure, s["level"], s["hero_pos"], s["shover_pos"],
            tuple(s["hero_cards"]), s["depth_bb"])
        parsed = parse_state_6max(state, observer=HERO_SEAT)
        built.append((state, parsed))

    cold_ms, warm_ms, fired = [], [], 0
    for (state, parsed) in built:
        p, m = _pm()
        seen_before = len(floor.eq_cache)
        t0 = time.perf_counter()
        out = floor.apply(p.copy(), m, parsed, state)
        dt = (time.perf_counter() - t0) * 1000.0
        cold = len(floor.eq_cache) > seen_before  # a new MC was computed
        if not np.array_equal(out, p):
            fired += 1
        if cold:
            cold_ms.append(dt)
        # Immediately re-apply: now memoized -> WARM timing for the same node.
        t1 = time.perf_counter()
        floor.apply(p.copy(), m, parsed, state)
        warm_ms.append((time.perf_counter() - t1) * 1000.0)

    print("GATE 3 — per-decision latency (ShoveDefenseFloor.apply)")
    print(f"battery               : {BATTERY}")
    print(f"spots timed           : {len(built)}  (fired: {fired})")
    print(f"cold MC decisions     : {len(cold_ms)}")
    print()
    print(f"{'path':<10}{'n':>6}{'p50_ms':>12}{'p99_ms':>12}"
          f"{'max_ms':>12}{'mean_ms':>12}")
    for name, xs in (("COLD", cold_ms), ("WARM", warm_ms)):
        if not xs:
            continue
        print(f"{name:<10}{len(xs):>6}{_pct(xs,50):>12.3f}"
              f"{_pct(xs,99):>12.3f}{max(xs):>12.3f}"
              f"{float(np.mean(xs)):>12.3f}")
    print()
    p99_cold = _pct(cold_ms, 99)
    p50_cold = _pct(cold_ms, 50)
    print(f"GATE (cold p99 < 1000 ms / 1 s live budget):")
    print(f"  cold p50 = {p50_cold:.3f} ms   cold p99 = {p99_cold:.3f} ms")
    if p99_cold < 1000.0:
        print(f"GATE 3 PASS: cold p99 = {p99_cold:.3f} ms < 1000 ms.")
        return 0
    print(f"GATE 3 FAIL: cold p99 = {p99_cold:.3f} ms >= 1000 ms.")
    return 1


def floor_struct(s):
    return s


if __name__ == "__main__":
    sys.exit(main())
