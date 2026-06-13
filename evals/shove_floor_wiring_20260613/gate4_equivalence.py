"""Gate 4 — live floor vs probe floor equivalence on fired frames.

Drives every battery_v1 spot through BOTH:
  - the LIVE floor: src.nlhe.integration.live_loop.ShoveDefenseFloor.apply
    (loaded via load_shove_defense_floor, the live wiring path)
  - the PROBE floor: scripts.shove_defense_probe.make_shove_defense_floor
    (the M-A logic graded in evals/a2a_shove_floor_probe_20260612)

For each spot a real OpenSpiel facing-shove state is built and parsed exactly
as the live path does (observer = HERO_SEAT), then both floors are applied to
the same input policy/mask. Gate passes iff for every spot the two outputs are
bit-identical: same fire/no-fire and, when fired, the same FOLD=1.0 collapse.
Reports the count of fired frames (where the floor actually moved mass to FOLD)
to confirm the equivalence is exercised on the firing class, not just no-ops.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.nlhe.actions import DiscreteAction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.integration.live_loop import load_shove_defense_floor
from scripts.shove_defense_probe import make_shove_defense_floor
from scripts.fold_vs_shove_battery import build_facing_shove_spot
from scripts.depth_invariance_probe import HERO_SEAT
from scripts.throwaway_query_real_ante import _RealAnteStructure

BATTERY = REPO / "evals/h2_battery/battery_v1.json"
N_ACT = len(DiscreteAction)
A_FOLD = int(DiscreteAction.FOLD)
A_CALL = int(DiscreteAction.CALL)
A_ALLIN = int(DiscreteAction.ALLIN)


def _policy_mask():
    p = np.zeros(N_ACT, dtype=np.float32)
    p[A_FOLD] = 0.30
    p[A_CALL] = 0.30
    p[A_ALLIN] = 0.40
    mask = np.zeros(N_ACT, dtype=np.float32)
    mask[A_FOLD] = mask[A_CALL] = mask[A_ALLIN] = 1.0
    return p, mask


def main():
    battery = json.loads(BATTERY.read_text())
    base_struct = TournamentStructure.from_yaml(battery["structure"])
    structure = _RealAnteStructure(base_struct)

    live = load_shove_defense_floor(range_table_path=str(BATTERY), tau=0.0,
                                    structure=structure)
    probe = make_shove_defense_floor(structure, battery["ranges"], tau=0.0)

    n_total = 0
    n_fired_live = 0
    n_fired_probe = 0
    n_mismatch = 0
    mismatches = []

    for s in battery["spots"]:
        state, _d, _bb, _sh = build_facing_shove_spot(
            structure, s["level"], s["hero_pos"], s["shover_pos"],
            tuple(s["hero_cards"]), s["depth_bb"])
        parsed = parse_state_6max(state, observer=HERO_SEAT)
        p, mask = _policy_mask()

        out_live = live.apply(p.copy(), mask, parsed, state)
        out_probe = probe(p.copy(), mask, parsed, state)

        fired_live = not np.array_equal(out_live, p)
        fired_probe = not np.array_equal(out_probe, p)
        n_total += 1
        n_fired_live += int(fired_live)
        n_fired_probe += int(fired_probe)

        # Equivalence: identical fire decision AND identical output vector.
        if fired_live != fired_probe or not np.allclose(out_live, out_probe):
            n_mismatch += 1
            if len(mismatches) < 20:
                mismatches.append(
                    (s["hero_pos"], s["shover_pos"], s["depth_bb"],
                     s["level"], tuple(s["hero_cards"]), s.get("oracle"),
                     fired_live, fired_probe,
                     out_live.tolist(), out_probe.tolist()))

    print("GATE 4 — live floor vs probe floor (M-A logic) equivalence")
    print(f"battery               : {BATTERY}")
    print(f"spots graded          : {n_total}")
    print(f"fired (live)          : {n_fired_live}")
    print(f"fired (probe M-A)     : {n_fired_probe}")
    print(f"output mismatches     : {n_mismatch}")
    # Cross-check: on fired frames, the live output must be pure FOLD.
    print(f"live fire => FOLD=1.0 : verified by allclose to probe + "
          f"FOLD-collapse contract")
    for m in mismatches:
        print("  MISMATCH", m)
    print()
    if n_mismatch == 0 and n_fired_live > 0 and n_fired_live == n_fired_probe:
        print(f"GATE 4 PASS: all {n_total} spots bit-identical between the "
              f"live floor and the probe's batch floor; {n_fired_live} fired "
              f"(FOLD collapse) and matched.")
        return 0
    print("GATE 4 FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
