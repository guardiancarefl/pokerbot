"""ReBeL depth-limited subgame solving on top of the validated DCFR engine.

This package implements the search half of ReBeL (Brown et al., NeurIPS 2020):
depth-limited subgame solving where the part of the tree beyond the depth limit
is replaced by a learned *public-belief-state* (PBS) value function. The CFR
solver underneath is the Leduc-validated `src.rlcfr.cfr.TabularCFR` (commit
34aa725); nothing here mutates that engine — `DepthLimitedCFR` subclasses it and
only inserts a leaf-substitution hook, so the engine's test guarantees carry
over.

Build/validation order (each gated on Leduc against the exact solution):

  1. DepthLimitedCFR with leaf_predicate=None  == full-tree engine  (plumbing).
  2. PBS representation + belief propagation for Leduc.
  3. An *oracle* leaf value function (exact subgame re-solve) — stands in for
     the not-yet-trained value net, to isolate "is the search correct?" from
     "is the value net good?".
  4. Full depth-limited re-solving agent + exploitability ≈ Nash  (GATE 1).
"""

from src.rebel.subgame import DepthLimitedCFR, LeafPredicate, LeafValueFn

__all__ = ["DepthLimitedCFR", "LeafPredicate", "LeafValueFn"]
