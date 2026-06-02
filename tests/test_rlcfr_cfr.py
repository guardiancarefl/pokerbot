"""Fast correctness tests for the RL-CFR tabular (D)CFR engine.

These guard the two real bugs found during the Leduc validation build:

  1. Per-iteration strategy must be *frozen*: an infoset is visited once per
     history (many times per traversal), so mutating regret mid-traversal lets
     later visits see a half-updated regret and corrupts the solve. Signature:
     after 1 iteration with uniform strategies, regrets at a 2-action node must
     be exactly symmetric (regret(a0) == -regret(a1)).

  2. The engine must actually converge to Nash. We use Kuhn (tiny, ~1s) so this
     stays a fast test; the full Leduc gate lives in
     scripts/validate_rlcfr_leduc.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyspiel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rlcfr.cfr import TabularCFR, DCFRParams  # noqa: E402


def test_one_iteration_regret_symmetry():
    """Frozen-strategy invariant: 2-action nodes are symmetric after 1 iter."""
    game = pyspiel.load_game("kuhn_poker")
    solver = TabularCFR(game, dcfr=DCFRParams(vanilla=True))
    solver.alternating = False  # simultaneous: clean uniform-strategy snapshot
    solver.run(1)
    for key, node in solver.nodes.items():
        if len(node.regret) == 2:
            assert np.isclose(node.regret[0], -node.regret[1], atol=1e-12), (
                f"asymmetric regret at {key}: {node.regret}")


def test_cfr_plus_converges_on_kuhn():
    """CFR+ drives Kuhn exploitability near zero (Nash is ~0)."""
    game = pyspiel.load_game("kuhn_poker")
    solver = TabularCFR(game, dcfr=DCFRParams(mode="plus"))
    solver.run(300)
    expl = solver.exploitability_mbb()
    assert expl < 1.0, f"CFR+ Kuhn exploitability too high: {expl} mbb/g"


def test_dcfr_converges_on_kuhn():
    """DCFR(1.5,0,2) — RL-CFR's setting — also converges on Kuhn."""
    game = pyspiel.load_game("kuhn_poker")
    solver = TabularCFR(game, dcfr=DCFRParams(mode="dcfr"))
    solver.run(300)
    expl = solver.exploitability_mbb()
    assert expl < 1.0, f"DCFR Kuhn exploitability too high: {expl} mbb/g"


def test_variant_ordering():
    """CFR+ and DCFR must both beat vanilla at equal iterations."""
    game = pyspiel.load_game("kuhn_poker")
    res = {}
    for name, params in [("plus", DCFRParams(mode="plus")),
                         ("dcfr", DCFRParams(mode="dcfr")),
                         ("vanilla", DCFRParams(vanilla=True))]:
        s = TabularCFR(game, dcfr=params)
        s.run(200)
        res[name] = s.exploitability_mbb()
    assert res["plus"] < res["vanilla"], res
    assert res["dcfr"] < res["vanilla"], res


if __name__ == "__main__":
    test_one_iteration_regret_symmetry()
    test_cfr_plus_converges_on_kuhn()
    test_dcfr_converges_on_kuhn()
    test_variant_ordering()
    print("all rlcfr cfr tests passed")
