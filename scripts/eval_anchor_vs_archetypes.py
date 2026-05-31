"""Step-1 baseline: exact chip-EV of the CFR+ Nash anchor vs each archetype cell.

For each of the 12 (archetype, strength) cells in the default grid, compute the
EXACT expected utility of the anchor playing against the archetype, averaged
over both seat assignments. "Exact" = full game-tree traversal via OpenSpiel
expected_game_score.policy_value — no sampling, zero variance.

This is the baseline the Step-3 adaptive policy must IMPROVE on:
  - cells where the anchor already crushes -> small headroom for adaptation.
  - cells where the anchor only ties or barely wins -> biggest expected gain.

Loads the anchor saved by scripts/train_leduc_cfr_anchor.py (arrays pickle).
Writes runs/leduc_anchor_vs_archetypes_<ts>/baseline.json with per-cell
expected returns in chips/game AND mbb/g (LEDUC_CHIPS_TO_MBB = 500).

Usage:
    python -m scripts.eval_anchor_vs_archetypes \
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
from open_spiel.python.algorithms import expected_game_score

from src.leduc import archetypes as A
from src.leduc.evaluate import LEDUC_CHIPS_TO_MBB


def load_anchor_callable(anchor_dir: Path):
    """Return an action_probabilities(state) callable backed by the saved
    average policy arrays."""
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


def anchor_chips_per_game(game, anchor_tp, archetype_tp) -> tuple[float, float, float]:
    """Exact expected chips/game for the anchor against the archetype,
    averaged over both seat assignments. Returns (anchor_avg, anchor_seat0,
    anchor_seat1) in chips."""
    init = game.new_initial_state()
    # Anchor seat 0, archetype seat 1
    u0 = expected_game_score.policy_value(init, [anchor_tp, archetype_tp])
    # Anchor seat 1, archetype seat 0
    u1 = expected_game_score.policy_value(init, [archetype_tp, anchor_tp])
    anchor_seat0 = float(u0[0])
    anchor_seat1 = float(u1[1])
    return (anchor_seat0 + anchor_seat1) / 2.0, anchor_seat0, anchor_seat1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor-dir", required=True,
                    help="Directory containing avg_policy_arrays.pkl")
    ap.add_argument("--out", default=None,
                    help="output dir (default runs/leduc_anchor_vs_archetypes_<ts>)")
    args = ap.parse_args()

    anchor_dir = Path(args.anchor_dir)
    out_dir = Path(args.out or f"runs/leduc_anchor_vs_archetypes_{time.strftime('%Y%m%d_%H%M%S')}")
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
        ev_chips, seat0_chips, seat1_chips = anchor_chips_per_game(game, anchor_tp, arch_tp)
        ev_mbb = ev_chips * LEDUC_CHIPS_TO_MBB
        rows.append({
            "cell_id": cell.id,
            "archetype": cell.name,
            "strength": cell.strength,
            "held_out": cell.id in held_out,
            "anchor_chips_per_game": ev_chips,
            "anchor_mbb_per_game": ev_mbb,
            "anchor_seat0_chips": seat0_chips,
            "anchor_seat1_chips": seat1_chips,
        })
    dt = time.time() - t0

    # Print a baseline table.
    print()
    print("Anchor (CFR+ Nash, 0.13 mbb/g exploitability) vs each archetype cell:")
    print(f"{'cell':32s}  {'role':6s}  {'chips/g':>10s}  {'mbb/g':>10s}  notes")
    print("-" * 78)
    for r in rows:
        role = "TEST" if r["held_out"] else "TRAIN"
        notes = []
        if r["anchor_mbb_per_game"] < 50:
            notes.append("LOW HEADROOM" if r["anchor_mbb_per_game"] > 0 else "ANCHOR LOSES")
        if abs(r["anchor_seat0_chips"] - r["anchor_seat1_chips"]) > 0.05:
            notes.append(f"seat-asym dlt={r['anchor_seat0_chips']-r['anchor_seat1_chips']:+.3f}")
        print(f"{r['cell_id']:32s}  {role:6s}  {r['anchor_chips_per_game']:>+10.4f}  "
              f"{r['anchor_mbb_per_game']:>+10.2f}  {' / '.join(notes)}")
    print()
    print(f"computed in {dt:.2f}s")
    print(f"artifact dir: {out_dir}")

    payload = {
        "anchor_dir": str(anchor_dir),
        "anchor_exploitability_mbb": 0.12857580807822816,
        "leduc_chips_to_mbb": LEDUC_CHIPS_TO_MBB,
        "n_info_states": n_info,
        "computed_seconds": dt,
        "grid_size": len(grid),
        "held_out_default": list(A.HELD_OUT_DEFAULT),
        "cells": rows,
        "train_cell_ids": [c.id for c in train],
        "test_cell_ids":  [c.id for c in test],
    }
    (out_dir / "baseline.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
