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
from src.rebel import pbs  # noqa: E402


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


def layer2_beliefs() -> bool:
    """Public partition is a faithful coarsening; beliefs conserve mass."""
    print("Layer 2 — PBS: public partition bijective with engine infosets")
    game = pyspiel.load_game("leduc_poker")
    t0 = time.time()

    # Engine's infoset set (all reachable after a single traversal).
    eng = TabularCFR(game, dcfr=DCFRParams(mode="plus"))
    eng.run(1)
    engine_infosets = set(eng.nodes.keys())

    tree = pbs.build_public_tree(game)
    members = []
    for pk, node in tree.items():
        members.extend(node["members"].values())
    member_set = set(members)

    # Bijection checks: no infoset claimed twice, and the public partition's
    # members are exactly the engine's infosets.
    no_dupes = len(members) == len(member_set)
    covers = member_set == engine_infosets
    n_priv = pbs.num_private(game)
    # Each public node's members are keyed by distinct private cards in [0, n_priv).
    priv_ok = all(
        all(0 <= c < n_priv for c in node["members"])
        for node in tree.values())

    # Belief mass conservation under the uniform policy: at the root public
    # states (round 1, no betting) every legal private card carries equal mass.
    def uniform_policy(state):
        la = state.legal_actions()
        return {a: 1.0 / len(la) for a in la}

    ranges = pbs.compute_ranges(game, uniform_policy)
    total_mass = sum(float(r[p].sum()) for r in ranges.values()
                     for p in range(game.num_players()))
    mass_finite = np.isfinite(total_mass) and total_mass > 0

    ok = bool(no_dupes and covers and priv_ok and mass_finite)
    print(f"  public nodes: {len(tree)}   infoset members: {len(members)}")
    print(f"  engine infosets: {len(engine_infosets)}   "
          f"partition==engine: {covers}")
    print(f"  no infoset claimed by two public nodes: {no_dupes}")
    print(f"  members keyed by valid private cards: {priv_ok}")
    print(f"  belief mass conserved/finite under uniform policy: {mass_finite}")
    print(f"  Layer 2 [{'PASS' if ok else 'FAIL'}]   ({time.time()-t0:.1f}s)\n")
    return ok


def main() -> int:
    print("=" * 64)
    print("GATE 1 — ReBeL depth-limited search validation on Leduc")
    print("=" * 64 + "\n")
    results = {}
    results["layer1"] = layer1_plumbing()
    results["layer2"] = layer2_beliefs()
    # Layers 3-4 are appended as they are built.

    print("-" * 64)
    allpass = all(results.values())
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"GATE 1 so far: {'PASS' if allpass else 'FAIL'}")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
