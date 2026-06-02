"""GATE 1 — validate the ReBeL depth-limited search logic on Leduc.

The search machinery is built in layers; each layer is checked against the exact
Leduc solution before the next is trusted. A buggy search invalidates everything
downstream (value net, NLHE), so this script is a HARD gate: any layer failing
means STOP and fix.

  Layer 1  plumbing      DepthLimitedCFR(leaf_predicate=None) must be byte-for-
                         byte identical to the validated TabularCFR engine.
  Layer 2  beliefs       PBS public-state partition + belief vectors are
                         consistent with the engine's infoset set.   (pbs.py)
  Layer 3  oracle leaf   exact subgame re-solve as the leaf value fn.
  Layer 4  GATE          depth-limited re-solving agent's full-game
                         exploitability ≈ Nash (not blown up).

Run:  python scripts/validate_rebel_leduc.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pyspiel  # noqa: E402

from src.rlcfr.cfr import DCFRParams, TabularCFR  # noqa: E402
from src.rebel.subgame import DepthLimitedCFR  # noqa: E402


def layer1_plumbing(iters: int = 100) -> bool:
    """DepthLimitedCFR with no leaf hook == the validated engine, exactly."""
    print("Layer 1 — plumbing: DepthLimitedCFR(leaf_predicate=None) == TabularCFR")
    game = pyspiel.load_game("leduc_poker")
    t0 = time.time()

    base = TabularCFR(game, dcfr=DCFRParams(mode="plus"))
    base.run(iters)
    e_base = base.exploitability_mbb()

    dl = DepthLimitedCFR(game, dcfr=DCFRParams(mode="plus"), leaf_predicate=None)
    dl.run(iters)
    e_dl = dl.exploitability_mbb()

    same_expl = np.isclose(e_base, e_dl, atol=1e-12, rtol=0.0)
    # Stronger: every infoset's average strategy must match bit-for-bit.
    keys = set(base.nodes) | set(dl.nodes)
    max_strat_diff = 0.0
    for k in keys:
        a = base.average_strategy(k)
        b = dl.average_strategy(k)
        if a is None or b is None:
            max_strat_diff = np.inf
            break
        max_strat_diff = max(max_strat_diff, float(np.max(np.abs(a - b))))
    same_nodes = len(base.nodes) == len(dl.nodes)

    ok = bool(same_expl and same_nodes and max_strat_diff < 1e-12)
    print(f"  engine    @{iters} = {e_base:.6f} mbb/g")
    print(f"  depthlim  @{iters} = {e_dl:.6f} mbb/g")
    print(f"  #infosets engine={len(base.nodes)} depthlim={len(dl.nodes)} "
          f"(match={same_nodes})")
    print(f"  max avg-strategy diff over all infosets = {max_strat_diff:.2e}")
    print(f"  leaf substitutions (must be 0): {dl.leaf_hits}")
    print(f"  Layer 1 [{'PASS' if ok else 'FAIL'}]   ({time.time()-t0:.1f}s)\n")
    return ok


def main() -> int:
    print("=" * 64)
    print("GATE 1 — ReBeL depth-limited search validation on Leduc")
    print("=" * 64 + "\n")
    results = {}
    results["layer1"] = layer1_plumbing()
    # Layers 2-4 are appended as they are built.

    print("-" * 64)
    allpass = all(results.values())
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"GATE 1 so far: {'PASS' if allpass else 'FAIL'}")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
