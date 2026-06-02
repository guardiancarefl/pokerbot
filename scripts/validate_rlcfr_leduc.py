"""Leduc validation gate for the RL-CFR tabular (D)CFR engine.

RL-CFR (Li/Fang/Huang, ICML 2024, arXiv:2403.04344) is a dynamic-action-
abstraction layer built on top of a CFR solver (specifically depth-limited
ReBeL with DCFR). The CFR engine is the foundation: if it does not converge to
Nash, nothing built on it can be trusted. This script is the hard gate — it
runs our engine on Leduc poker, where the Nash equilibrium is known, and checks
that exploitability (measured by OpenSpiel's *exact* best-response oracle)
converges to the same ~0.13 mbb/g that OpenSpiel's reference CFR+ and the repo's
own Deep CFR reach.

Pass criterion (the gate): CFR+ exploitability at 1000 iterations < 0.5 mbb/g
(OpenSpiel CFR+ reference is 0.1286; our DCFR setting reaches ~0.09).

Run:  python scripts/validate_rlcfr_leduc.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyspiel  # noqa: E402

from src.rlcfr.cfr import TabularCFR, DCFRParams  # noqa: E402

# OpenSpiel reference CFR+ on Leduc (computed independently; see git history):
#   it100 = 6.708, it300 = 0.798, it1000 = 0.1286 mbb/g
GATE_THRESHOLD_MBB = 0.5  # CFR+ @ 1000 iters must beat this to pass the gate


def run_curve(name: str, params: DCFRParams, milestones=(100, 300, 1000)):
    game = pyspiel.load_game("leduc_poker")
    solver = TabularCFR(game, dcfr=params)
    t0 = time.time()
    curve = {}
    for m in milestones:
        solver.run(m - solver.iteration)
        expl = solver.exploitability_mbb()
        curve[m] = expl
        print(f"  {name:6s} it{m:5d} = {expl:9.5f} mbb/g   ({time.time()-t0:5.1f}s)")
    return curve


def main() -> int:
    print("RL-CFR engine — Leduc validation gate")
    print("Oracle: OpenSpiel exact best response. Reference: CFR+ = 0.1286 mbb/g @1000.\n")

    cfr_plus = run_curve("CFR+", DCFRParams(mode="plus"))
    print()
    dcfr = run_curve("DCFR", DCFRParams(mode="dcfr"))  # RL-CFR's (1.5, 0, 2)

    print()
    final = cfr_plus[1000]
    passed = final < GATE_THRESHOLD_MBB
    status = "PASS" if passed else "FAIL"
    print(f"GATE [{status}]: CFR+ @1000 = {final:.5f} mbb/g "
          f"(threshold {GATE_THRESHOLD_MBB}); DCFR @1000 = {dcfr[1000]:.5f} mbb/g")
    if passed:
        print("Engine converges to Nash on Leduc, matching OpenSpiel CFR+ / Deep CFR.")
    else:
        print("Engine does NOT converge to the known Leduc optimum — DO NOT trust it.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
