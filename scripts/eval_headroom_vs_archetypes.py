"""Step-1 TRUE headroom: anchor-EV vs best-response-EV vs each archetype cell.

For each of the 12 (archetype, strength) cells, computes two EXACT chip-EVs
via full-tree traversal (no sampling, zero variance):

  ANCHOR-EV  = expected chips/game when the CFR+ Nash anchor plays the cell
               (via expected_game_score.policy_value)
  BR-EV      = expected chips/game when hero best-responds to the cell
               (via best_response.BestResponsePolicy.value())

HEADROOM = BR-EV − ANCHOR-EV is the Step-4 DENOMINATOR: the absolute
ceiling on adaptation gain for that cell. A perfectly-adaptive policy
that converged to BR vs each opponent would close this gap. Both EVs are
averaged over both seat assignments (BR-p0 vs cell-p1, then BR-p1 vs
cell-p0). Units: chips/game and mbb/g (LEDUC_CHIPS_TO_MBB = 500).

Loads the anchor saved by scripts/train_leduc_cfr_anchor.py. Writes
runs/leduc_headroom_<ts>/headroom.json.

Usage:
    python -m scripts.eval_headroom_vs_archetypes \
        --anchor-dir runs/leduc_cfr_anchor_20260531_144405
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import best_response, expected_game_score

from src.leduc import archetypes as A
from src.leduc.evaluate import LEDUC_CHIPS_TO_MBB


def load_anchor_callable(anchor_dir: Path):
    with open(anchor_dir / "avg_policy_arrays.pkl", "rb") as f:
        snap = pickle.load(f)
    state_lookup = snap["state_lookup"]
    probs_arr = np.asarray(snap["action_probability_array"])
    mask_arr = np.asarray(snap["legal_actions_mask"])

    def anchor_fn(state):
        info = state.information_state_string()
        idx = state_lookup[info]
        probs = probs_arr[idx]
        mask = mask_arr[idx]
        return {int(a): float(probs[a]) for a in range(probs.shape[0]) if mask[a]}

    return anchor_fn, len(state_lookup)


def anchor_chips(game, anchor_tp, arch_tp):
    init = game.new_initial_state()
    u0 = expected_game_score.policy_value(init, [anchor_tp, arch_tp])
    u1 = expected_game_score.policy_value(init, [arch_tp, anchor_tp])
    return float(u0[0]), float(u1[1])


def br_chips(game, arch_tp):
    """Exact BR-EV in both seats vs the archetype."""
    init = game.new_initial_state()
    br_p0 = best_response.BestResponsePolicy(game, 0, arch_tp)
    br_p1 = best_response.BestResponsePolicy(game, 1, arch_tp)
    return float(br_p0.value(init)), float(br_p1.value(init))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    anchor_dir = Path(args.anchor_dir)
    out_dir = Path(args.out or f"runs/leduc_headroom_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    game = pyspiel.load_game("leduc_poker")
    anchor_fn, n_info = load_anchor_callable(anchor_dir)
    anchor_tp = policy_lib.tabular_policy_from_callable(game, anchor_fn)
    print(f"loaded anchor: {n_info} info states from {anchor_dir}")

    grid = A.default_grid()
    train, test = A.train_test_split()
    held_out = {c.id for c in test}

    rows = []
    t0 = time.time()
    for cell in grid:
        fn = A.build_cell(cell)
        arch_tp = policy_lib.tabular_policy_from_callable(game, fn)

        a0, a1 = anchor_chips(game, anchor_tp, arch_tp)
        b0, b1 = br_chips(game, arch_tp)
        anchor_avg = (a0 + a1) / 2.0
        br_avg = (b0 + b1) / 2.0
        headroom = br_avg - anchor_avg

        rows.append({
            "cell_id": cell.id,
            "archetype": cell.name,
            "strength": cell.strength,
            "held_out": cell.id in held_out,
            "anchor_chips_per_game": anchor_avg,
            "anchor_mbb_per_game": anchor_avg * LEDUC_CHIPS_TO_MBB,
            "br_chips_per_game": br_avg,
            "br_mbb_per_game": br_avg * LEDUC_CHIPS_TO_MBB,
            "headroom_chips_per_game": headroom,
            "headroom_mbb_per_game": headroom * LEDUC_CHIPS_TO_MBB,
            "anchor_seat0": a0, "anchor_seat1": a1,
            "br_seat0": b0,     "br_seat1": b1,
        })
    dt = time.time() - t0

    rows_sorted = sorted(rows, key=lambda r: -r["headroom_mbb_per_game"])

    print()
    print("Step-1 TRUE headroom (exact, full-tree; anchor = CFR+ Nash, 0.13 mbb/g exploitable):")
    print(f"{'cell':32s}  {'role':5s}  {'anchor':>10s}  {'BR':>10s}  {'headroom':>10s}")
    print(f"{'':32s}  {'':5s}  {'mbb/g':>10s}  {'mbb/g':>10s}  {'mbb/g':>10s}")
    print("-" * 78)
    for r in rows_sorted:
        role = "TEST" if r["held_out"] else "TRAIN"
        print(f"{r['cell_id']:32s}  {role:5s}  "
              f"{r['anchor_mbb_per_game']:>+10.2f}  "
              f"{r['br_mbb_per_game']:>+10.2f}  "
              f"{r['headroom_mbb_per_game']:>+10.2f}")
    print()
    print(f"computed in {dt:.2f}s")
    print(f"artifact: {out_dir / 'headroom.json'}")

    payload = {
        "anchor_dir": str(anchor_dir),
        "anchor_exploitability_mbb": 0.12857580807822816,
        "leduc_chips_to_mbb": LEDUC_CHIPS_TO_MBB,
        "n_info_states": n_info,
        "computed_seconds": dt,
        "grid_size": len(grid),
        "held_out_default": list(A.HELD_OUT_DEFAULT),
        "cells": rows,
        "cells_by_headroom_desc": [r["cell_id"] for r in rows_sorted],
        "train_cell_ids": [c.id for c in train],
        "test_cell_ids":  [c.id for c in test],
        "method": "exact full-tree (no sampling); anchor via "
                  "expected_game_score.policy_value; BR via "
                  "best_response.BestResponsePolicy.value; both seats averaged",
    }
    (out_dir / "headroom.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
