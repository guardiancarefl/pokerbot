"""Depth-limited subgame CFR solver (Track B1c, sub-step 3).

This is the real multi-iteration solver that turns the depth-limited subgame tree
(`subgame.build_subgame_tree`, sub-step 1.5) plus the validated leaf values
(`subgame_leaf.evaluate_leaves`, sub-step 2 — Stage F/G closed) into hero's
**refined root policy**. It is the piece Pluribus called "continual resolving";
the design and the seven pre-committed decisions live in `docs/SUBSTEP_3_DESIGN.md`.

Relationship to the other B1c modules:
  - LEAF EVALUATOR (`subgame_leaf.py`): the solver READS `node.leaf_value` and
    `node.terminal_returns`; it never calls the evaluator. `evaluate_leaves(tree,
    leaf_ctx)` must run BEFORE `solve_subgame` so every LEAF carries its 6-vector.
    Because the BEST_RESPONSE leaf value is computed against hero's *blueprint*
    (the Brown/Sandholm 2018 single-pass approximation, design Q6), it is fixed for
    the whole decision — so it is cached exactly across all CFR iterations and the
    iteration loop does zero rollouts and zero leaf re-evaluation.
  - STAGE-G STUB (`scripts/ablation_decision_level.py:stub_root_policy`): the stub
    was a depth-1, one-iteration, hero-only regret update at the root. This module
    is its generalization to depth-K, multi-iteration, full-tree traversal. The
    stub is the seed; this is the plant.

Pre-committed algorithm (design Decisions 2 & 4, both signed off):
  - VANILLA weighted traversal over the small finite built tree — NOT solve-time
    Monte-Carlo external sampling. Chance children are weighted by their
    renormalized `chance_prob` (subgame.py:114-117), opponent children by the
    opponent's FIXED blueprint strategy, hero children enumerated for regret.
  - Only HERO accumulates regret; opponents and chance are fixed (resolve against
    the blueprint, not against itself). Safety is realized by the BEST_RESPONSE
    leaf mode (the bias menu discretizes opponent deviations), not a HUNL CFV
    gadget — see ARCHITECTURE.md Layer 3 and design Decision 6.

K progression (n_iterations):
  - K = 0  → NO refinement: return the blueprint's own masked action distribution
            at the root, `RM+(adv_root)`. A real, valid result ("don't refine").
            *** Stage 3-A ships this path. ***
  - K = 1  → a single root regret update; reduces bit-identically to the Stage-G
            stub `stub_root_policy`. (Stage 3-B.)
  - K > 1  → the full vanilla weighted CFR loop with the running average strategy.
            (Stage 3-C/3-D.)

STATUS: STAGE 3-A — scaffold + warm-up caching + K=0 path. `solve_subgame` raises
NotImplementedError for n_iterations >= 1 (Stages 3-B/3-C land that).
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.nlhe.actions import DiscreteAction
from src.nlhe.infoset6 import parse_state_6max, parse_state_repeated_6max
from src.nlhe.solver import _strategy_from_advantages
from src.nlhe.subgame import SubgameTree, iter_decision_nodes
# Reuse the leaf evaluator's blueprint protocol rather than redefining it — a
# DeepCFR6MaxSolver already satisfies it (encoder + policy_nets).
from src.nlhe.subgame_leaf import BlueprintProvider

log = logging.getLogger("subgame_solver")

_N_ACTIONS = len(DiscreteAction)  # 7
_NUM_SEATS = 6


# ============================================================
# Context / Result (mirrors CFR6MaxContext / LeafEvalContext)
# ============================================================

@dataclass
class SubgameSolveContext:
    """Fixed dependencies for one subgame solve (one hero decision).

    Bundled like `cfr6.CFR6MaxContext` / `subgame_leaf.LeafEvalContext` so
    `solve_subgame` keeps a short signature. Every field is fixed for the whole
    solve, which is what makes the warm-up cache and the leaf cache valid across
    all CFR iterations (design Q6 / Decision 1).

    Required:
        blueprint: the BlueprintProvider (encoder + policy_nets). Queried only in
            warm-up to populate per-node advantages / strategies; the iteration
            loop never touches it.
        starting_stacks: per-seat chips at the START of the hand (length 6); the
            `icm_adjust_returns` baseline at terminal nodes (matches the leaf
            evaluator's `LeafEvalContext.starting_stacks`).
        payouts: tournament prize structure (e.g. `sng_payouts_6max_double_up()`).
        hero_seat: the seat whose decision this subgame refines. MUST equal
            `tree.root.current_player` (the root is hero's decision).

    Optional:
        n_iterations: K (design Decision 1). K=0 → blueprint passthrough (Stage
            3-A). K>=1 lands in Stage 3-B/3-C. Default 1000 (the design baseline;
            the real number is set by the Stage 3-E convergence curve).
        rng: random source. Vanilla CFR itself is deterministic; the rng exists for
            any future tie-break / sampling and to seed the warm-up encoder. None →
            a fresh `random.Random()`.
        num_paid: paid finishing positions (Double Up 6-max = 3).
        average_weighting: "linear" (LCFR, weight iteration t by t; Pluribus's
            choice, faster convergence) or "uniform". Unused until K>=1.
    """
    blueprint: BlueprintProvider
    starting_stacks: Sequence[int]
    payouts: Sequence[float]
    hero_seat: int

    n_iterations: int = 1000
    rng: Optional[random.Random] = None
    num_paid: int = 3
    average_weighting: str = "linear"

    def __post_init__(self) -> None:
        if len(self.starting_stacks) != _NUM_SEATS:
            raise ValueError(
                f"starting_stacks must have {_NUM_SEATS} entries, "
                f"got {len(self.starting_stacks)}"
            )
        if not self.payouts:
            raise ValueError("payouts must be non-empty")
        if not (0 <= self.hero_seat < _NUM_SEATS):
            raise ValueError(
                f"hero_seat {self.hero_seat} out of range [0, {_NUM_SEATS})"
            )
        if self.n_iterations < 0:
            raise ValueError(f"n_iterations must be >= 0, got {self.n_iterations}")
        if self.average_weighting not in ("linear", "uniform"):
            raise ValueError(
                f"average_weighting must be 'linear' or 'uniform', "
                f"got {self.average_weighting!r}"
            )


@dataclass
class SubgameSolveResult:
    """The refined root policy plus diagnostics.

    Attributes:
        root_policy: hero's refined action distribution at the root — a length-7
            masked probability vector (sums to 1 over legal actions, 0 on illegal).
            This is the CONTRACT consumed by sub-step 4 (policy extraction). For
            K=0 it equals `root_blueprint`.
        root_blueprint: the unrefined blueprint strategy at the root, σ0 =
            RM+(adv_root). Kept for diagnostics / the policy-shift measurement.
        legal_mask: the root's length-7 legal mask (1 on legal DiscreteActions).
        hero_seat: echo of the seat solved for.
        n_iterations: CFR iterations actually run (0 for the K=0 passthrough).
        n_decision_nodes_cached: warm-up coverage (decision nodes the blueprint was
            queried at). Diagnostic — confirms warm-up walked the whole tree.
        degraded: True if any reached LEAF lacked a `leaf_value` (e.g. a
            budget-truncated batch). Lets sub-step 5 choose a blueprint fallback.
            Always False on the K=0 path (no leaves are read).
        converged_l1_tail: mean per-iteration root-policy L1 change over the last
            10% of iterations (the Stage-3-E convergence metric). NaN until K>=1.
    """
    root_policy: np.ndarray
    root_blueprint: np.ndarray
    legal_mask: np.ndarray
    hero_seat: int
    n_iterations: int
    n_decision_nodes_cached: int
    degraded: bool = False
    converged_l1_tail: float = float("nan")


# ============================================================
# Warm-up caching (Stage 3-A) — hoisted from the Stage-G stub pattern
# ============================================================

@dataclass
class _WarmupCache:
    """Per-decision-node blueprint quantities, keyed by `id(node)`.

    Populated once before the CFR loop; read (never recomputed) during iterations.
    All blueprint network forwards happen here, which is why the iteration loop is
    pure arithmetic (design Decision 1 / cost model §D).

        adv[id]   : (7,) float32 blueprint advantages at the node's acting seat.
        mask[id]  : (7,) float32 legal mask, built from the node's children.
        sigma[id] : (7,) float32 blueprint strategy RM+(adv) — the FIXED strategy
                    opponents play, and σ0 at the hero root.
    """
    adv: dict = field(default_factory=dict)
    mask: dict = field(default_factory=dict)
    sigma: dict = field(default_factory=dict)


def _parse(state):
    """Dispatch the parser exactly as cfr6.traverse_6max and subgame do
    (repeated_poker exposes dealer_seat; single-hand universal_poker does not)."""
    if hasattr(state, "dealer_seat"):
        return parse_state_repeated_6max(state)
    return parse_state_6max(state)


def _mask_from_children(node) -> np.ndarray:
    """Legal mask from a decision node's children.

    Decision-node `action_at_child` holds DiscreteAction ints (subgame.py:396) —
    the builder created a child only for each legal discrete action, so this is
    exactly the legal mask the blueprint's RM+ should use (matches the Stage-G stub
    at ablation_decision_level.py:307-309)."""
    m = np.zeros(_N_ACTIONS, dtype=np.float32)
    for a in node.action_at_child:
        m[int(a)] = 1.0
    return m


def _blueprint_adv(node, ctx: SubgameSolveContext, rng) -> np.ndarray:
    """Blueprint advantages at a decision node's acting seat.

    Mirrors the Stage-G stub (ablation_decision_level.py:295-297) and
    eval_6max_self_play._sample_action_from_policy (lines 137-143): parse → encode
    → predict_advantages for the node's current_player."""
    parsed = _parse(node.state)
    feat = np.asarray(
        ctx.blueprint.encoder.encode_from_parsed(parsed, rng=rng), dtype=np.float32
    )
    cp = node.current_player
    adv = np.asarray(
        ctx.blueprint.policy_nets.predict_advantages(cp, feat), dtype=np.float32
    )
    return adv


def _build_warmup(tree: SubgameTree, ctx: SubgameSolveContext, rng) -> _WarmupCache:
    """Populate per-decision-node adv / mask / sigma from the blueprint.

    Resets the encoder bucket cache once at entry so a freshly seeded rng yields
    reproducible bucket draws (the Stage-G stub / Stage-E pattern); `bucket_of` is
    deterministic, so the reset has no correctness effect, only perf hygiene."""
    try:
        ctx.blueprint.encoder.reset_cache()
    except Exception:
        pass
    cache = _WarmupCache()
    for node in iter_decision_nodes(tree):
        adv = _blueprint_adv(node, ctx, rng)
        mask = _mask_from_children(node)
        nid = id(node)
        cache.adv[nid] = adv
        cache.mask[nid] = mask
        cache.sigma[nid] = _strategy_from_advantages(adv, mask)
    return cache


# ============================================================
# Entry point (Stage 3-A: warm-up + K=0; K>=1 lands later)
# ============================================================

def solve_subgame(tree: SubgameTree, ctx: SubgameSolveContext) -> SubgameSolveResult:
    """Solve the depth-limited subgame; return hero's refined root policy.

    Pre-condition: `subgame_leaf.evaluate_leaves(tree, leaf_ctx)` has populated
    every LEAF's `leaf_value` (the K>=1 loop reads them; the K=0 path does not).

    Stage 3-A scope: builds the warm-up cache and serves the K=0 passthrough
    (return the blueprint's masked root policy unchanged). K>=1 raises
    NotImplementedError until Stage 3-B/3-C.
    """
    root = tree.root
    if root is None or not root.is_decision:
        raise ValueError("subgame root must be a DECISION node (hero's decision)")
    if root.current_player != ctx.hero_seat:
        raise ValueError(
            f"root current_player {root.current_player} != "
            f"ctx.hero_seat {ctx.hero_seat}"
        )

    rng = ctx.rng if ctx.rng is not None else random.Random()
    cache = _build_warmup(tree, ctx, rng)

    nid = id(root)
    root_mask = cache.mask[nid]
    root_sigma0 = cache.sigma[nid]  # blueprint masked policy at the root

    if ctx.n_iterations == 0:
        # K=0 — no refinement: hero plays the blueprint's own action distribution.
        return SubgameSolveResult(
            root_policy=root_sigma0.copy(),
            root_blueprint=root_sigma0.copy(),
            legal_mask=root_mask.copy(),
            hero_seat=ctx.hero_seat,
            n_iterations=0,
            n_decision_nodes_cached=len(cache.adv),
            degraded=False,
        )

    raise NotImplementedError(
        "solve_subgame with n_iterations >= 1 lands in Stage 3-B (K=1, stub "
        "bit-identity) and Stage 3-C (K>1, full vanilla weighted CFR loop). "
        "Stage 3-A ships warm-up caching + the K=0 blueprint-passthrough path only."
    )
