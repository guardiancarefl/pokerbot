"""Depth-limited subgame LEAF evaluator (Track B1c, sub-step 2).

Assigns each `SubgameNode` of kind LEAF a per-seat ICM-equity-delta 6-vector — the
same units `cfr6.traverse_6max` backs up at true terminals — so sub-step 3's CFR
loop can treat LEAF and TERMINAL nodes uniformly. Full design and rationale:
`docs/SUBGAME_LEAF_DESIGN.md` (the contract this module conforms to).

STATUS: STAGE B — SCAFFOLD ONLY. This file defines the Q9 interface contracts
(LeafEvalMode, BlueprintProvider, LeafEvalContext, evaluate_leaf,
evaluate_leaves) with NO behavior. The evaluation bodies raise
NotImplementedError and land in later stages:
  - Stage C: PROFILE_SAMPLE mode (prior-weighted average, the simpler path).
  - Stage D: BEST_RESPONSE mode (opponent best-response, the production path).
  - Stage E: evaluate_leaves batch path (lockstep batching across leaves).

Cost model (validated provenance, for future readers — not buried in the doc):
  Per-decision-step cost ≈ 0.4 ms CPU / 0.13 ms GPU INCLUDING the network
  forward, validated by Stage A: `fast_view` + discretize run at 0.084 ms/step
  end-to-end EXCLUDING the network (commit 994b587; gate ≤0.30 ms cleared ~3.5×).
  The earlier "~0.9 ms parse floor" was a mis-attribution — corrected to
  view/discretize over the ~9,803-element fullgame legal_actions (commit dc09617).
  Stage E's *batched* cost is the gate that actually closes the full Z = 1.5 s
  leaf-eval budget; this scaffold ships at the design's M = 8 (LeafEvalContext
  default) with a measured fallback to M = 5 per Stage E results (see n_samples).

This module reuses, rather than re-implements:
  - `biased_policy.BiasedBlueprint` — the k=4 continuation strategies (Q2).
  - `fast_view.fast_view_and_discretize` — the rollout-hot-loop view/discretize
    (Stage A); rollouts in Stages C-D run through it, not the canonical path.
  - `icm_returns.icm_adjust_returns` — terminal chip-returns → ICM-equity delta.
  - `icm.icm_equity` / `icm.is_itm` — the Option-A ITM short-circuit (Q5).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from src.nlhe.subgame import SubgameNode, SubgameTree


# ============================================================
# Mode (Q9)
# ============================================================

class LeafEvalMode(Enum):
    """Which leaf-value semantics to use.

    BEST_RESPONSE is the production / GPU target (Q3, Q6): each live opponent
    independently best-responds among its k biases against hero's blueprint
    (Brown/Sandholm 2018 single-pass approximation). PROFILE_SAMPLE is the
    cheaper, non-robust comparison baseline (prior-weighted average over biases),
    used for the Q11 ablation and CPU debugging.
    """
    BEST_RESPONSE = "best_response"    # default, production / GPU target
    PROFILE_SAMPLE = "profile_sample"  # CPU fallback / ablation (Q3, Q11)


# ============================================================
# Blueprint provider protocol (Q9)
# ============================================================

@runtime_checkable
class BlueprintProvider(Protocol):
    """The minimal blueprint surface the leaf evaluator needs.

    Defined as a Protocol (not an import of the concrete `DeepCFR6MaxSolver`) so
    `subgame_leaf` stays decoupled from the solver, matching the `Policy`-protocol
    style of `scripts/eval_pool.py`. A `DeepCFR6MaxSolver` already satisfies it.

    Required attributes (exercised exactly as `eval_6max_self_play._sample_action_
    from_policy` does, lines 137-143):
      - `encoder`: has `encode_from_parsed(parsed: dict, rng) -> np.ndarray`
        (236-dim feature vector for the acting seat).
      - `policy_nets`: has `predict_advantages(seat: int, features) -> np.ndarray`
        (7-dim advantages; RM+ over the legal mask yields the blueprint probs the
        BiasedBlueprint is then applied to).
    """
    encoder: Any
    policy_nets: Any


# ============================================================
# Eval context (Q9)
# ============================================================

@dataclass
class LeafEvalContext:
    """Fixed dependencies for one decision's leaf-eval pass.

    Bundled (same pattern as `cfr6.CFR6MaxContext`) so `evaluate_leaf` /
    `evaluate_leaves` keep short signatures. Every field here is fixed for the
    whole subgame solve of a decision — which is exactly why leaf values are
    cacheable-exact across CFR iterations (Q6).

    Required:
        blueprint: the BlueprintProvider (encoder + policy_nets).
        biased_blueprint: `biased_policy.BiasedBlueprint` (the k=4 bias configs).
        starting_stacks: per-seat chips at the START of the hand (length 6);
            the `icm_adjust_returns` baseline at terminals (Q1/Q5).
        payouts: tournament prize structure (e.g. `sng_payouts_6max_double_up()`).
        hero_seat: the seat whose decision this subgame is being solved for;
            hero plays blueprint (bias index 0), opponents best-respond / are
            sampled (Q3).

    Optional (defaults are the design assumptions):
        mode: BEST_RESPONSE (default, production) or PROFILE_SAMPLE (Q9).
        opponent_prior: per-(seat, bias) array, shape (k,) or (NUM_SEATS_6MAX, k);
            None → uniform full menu. Meaning is mode-dependent (Q7): in
            BEST_RESPONSE it is the best-response MENU (0 excludes a bias, positive
            includes + tilts the argmax); in PROFILE_SAMPLE it is the sampling
            distribution. Track C1 nudges it; the evaluator never hardcodes it.
        n_samples: MC rollouts per (leaf, strategy) — the design's M.
            DEFAULT = 8 is PROVISIONAL: it is the optimistic-batching assumption
            pending Stage E's measured batched cost. If Stage E shows M=8 does not
            fit Z = 1.5 s, the default drops to M = 5 (still within Q4's stderr
            bounds). The knob is live; do NOT fix it to 5 prematurely — keep 8 as
            the default and let Stage E's measurement choose.
        rng: random source for chance/bias sampling. None → a fresh
            `random.Random()` is created at eval time (reproducibility tests pass
            a seeded one).
        icm_short_circuit: if True, leaves where `is_itm()` holds skip the rollout
            and use the Option-A ICM estimate directly (Q5).
        time_budget_s: wall-clock guard. None → unbounded. When exceeded, the
            evaluator degrades (fewer samples, then Option-A) rather than
            overrunning, setting a degraded flag (Q4/Q8).
        num_paid: paid finishing positions (Double Up 6-max = 3).
    """
    blueprint: BlueprintProvider
    biased_blueprint: Any
    starting_stacks: Sequence[int]
    payouts: Sequence[float]
    hero_seat: int

    mode: LeafEvalMode = LeafEvalMode.BEST_RESPONSE
    opponent_prior: Optional[Any] = None
    n_samples: int = 8  # design M; provisional pending Stage E (see docstring)
    rng: Optional[random.Random] = None
    icm_short_circuit: bool = True
    time_budget_s: Optional[float] = None
    num_paid: int = 3


# ============================================================
# Entry points (Q9) — bodies land in Stages C/D/E
# ============================================================

def evaluate_leaf(node: SubgameNode, ctx: LeafEvalContext) -> tuple[float, ...]:
    """Return the per-seat ICM-equity-delta 6-vector for a single LEAF node.

    The testable inner call. Dispatches on `ctx.mode`:
      - PROFILE_SAMPLE: prior-weighted average over opponents' biases (Stage C).
      - BEST_RESPONSE: opponents independently best-respond among biases (Stage D).

    Returns a length-`NUM_SEATS_6MAX` tuple of floats (Q1).

    NOTE: scaffold only — body lands in Stages C (PROFILE_SAMPLE) and
    D (BEST_RESPONSE).
    """
    raise NotImplementedError(
        "evaluate_leaf: leaf evaluation lands in Stage C (PROFILE_SAMPLE) "
        "and Stage D (BEST_RESPONSE). This is the Stage B scaffold."
    )


def evaluate_leaves(tree: SubgameTree, ctx: LeafEvalContext) -> None:
    """Populate `node.leaf_value` for every LEAF node in `tree`, in place.

    The batch entry point sub-step 3 calls ONCE before the CFR loop. Iterates
    `iter_leaf_nodes(tree)` and, where possible, batches the per-step network
    forward across leaves in lockstep (the Q4 batching argument). Writes the
    `evaluate_leaf` result onto each leaf's `leaf_value` field.

    NOTE: scaffold only — body lands in Stage E.
    """
    raise NotImplementedError(
        "evaluate_leaves: the batch path lands in Stage E. This is the "
        "Stage B scaffold."
    )
