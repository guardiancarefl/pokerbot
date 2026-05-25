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

STATUS: STAGE 3-D — full solver + diagnostic enrichment. K=0 passthrough, K=1
single-iteration regret update (bit-identical to the Stage-G stub), and K>1 the
multi-iteration vanilla weighted CFR loop (`_run_cfr`) with deeper tree descent and
linear-LCFR average-strategy output. SubgameSolveResult carries diagnostic fields
(convergence_history, root_q_values, blueprint/refined advantages) and
`summarize_solve_result` produces a JSON-serializable summary for sub-step 5/6.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.nlhe.actions import DiscreteAction
from src.nlhe.icm_returns import icm_adjust_returns
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
        n_iterations: K (design Decision 1). K=0 → blueprint passthrough; K=1 →
            single regret update (Stage-G-stub-identical); K>1 → the CFR loop.
            Production default chosen in Stage 3-E based on Stage 3-C measurements:
            converged_l1_tail at K=1000 is ~4.4e-8 (far below any stabilization
            threshold), and the CFR loop costs ~0.1s at this K (negligible against
            the ~10-20s leaf-eval cost per decision, Q13). K=1000 was selected for
            comfortable convergence margin; the loop scales linearly in K so
            escalation is essentially free if ever needed (K=10000 ≈ 1s loop).
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
        converged_l1_tail: L1 change in the AVERAGE (output) root policy between the
            last two iterations — the Stage-3-E convergence metric (design doc D
            "root-policy L1 change"). NaN for K<2. Not the current-strategy
            consecutive L1, which collapses to 0 in ~1-2 iterations here (the subgame
            is a best response to fixed opponents; see _run_cfr).

    Stage-3-D diagnostic fields (supplementary to root_policy; for sub-step 5/6
    introspection — see `summarize_solve_result`). Interpretation, normal-vs-
    pathological:
        n_iterations_run: CFR iterations actually executed. Equals `n_iterations`
            today (no early-stop); a future stage may stop early at convergence.
        convergence_history: tuple of (iteration, avg-policy-L1) pairs sampled every
            10 iterations (and the last). NORMAL: values decrease toward 0
            (monotone-ish, design doc D). PATHOLOGICAL: flat or rising → the solve is
            not converging (check leaf values / regret accumulation).
        root_q_values: hero's per-action Q at the root from the FINAL iteration (the
            q that drove the last regret update). NORMAL: spread across actions, the
            argmax aligns with where root_policy puts mass. All-equal q ⇒ no signal
            (bias-inactive root) ⇒ root_policy ≈ root_blueprint. NaN/garbage ⇒ a
            degraded leaf leaked in (`degraded` should then be True).
        root_advantages_blueprint: the warm-up advantage vector RM+ turns into
            root_blueprint (the network's raw O(1) advantages, the warm-start). Can
            be negative.
        root_advantages_refined: the hero root's cumulative regret accumulator after
            K iterations (RM+-clamped, >= 0). NORMAL: mass concentrates on the
            best-response action(s); argmax aligns with root_q_values' argmax. NOTE:
            this GROWS ~linearly with K (it is accumulated regret, not a one-shot
            advantage), so it is on a DIFFERENT scale than root_advantages_blueprint
            — read the *direction* (which actions gained mass), and use root_policy
            vs root_blueprint (or summarize's policy_shift_l1) for the *magnitude* of
            the pull off blueprint.
    """
    root_policy: np.ndarray
    root_blueprint: np.ndarray
    legal_mask: np.ndarray
    hero_seat: int
    n_iterations: int
    n_decision_nodes_cached: int
    degraded: bool = False
    converged_l1_tail: float = float("nan")
    n_iterations_run: int = 0
    convergence_history: tuple = ()
    root_q_values: Optional[np.ndarray] = None
    root_advantages_blueprint: Optional[np.ndarray] = None
    root_advantages_refined: Optional[np.ndarray] = None


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
# Regret update + hero action values (Stage 3-B: the K=1 case)
# ============================================================

def _regret_matched_policy(adv: np.ndarray, q: np.ndarray,
                           mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-iteration root regret update; returns (sigma_m, sigma0).

    DELIBERATELY mirrors `scripts.ablation_decision_level.stub_root_policy`
    line-for-line — the Stage-G validated stub. We copy the math rather than import
    it so `src/nlhe/` does not depend on a `scripts/` ablation harness (which would
    also pull torch-heavy modules at import). The Stage-3-B bit-identity test
    (`test_k1_bit_identical_to_stub`) is the guard that keeps the two in lockstep.

        sigma0 = RM+(adv)              # blueprint masked policy
        ev     = Σ_a sigma0[a]·q[a]    # blueprint-mix value under the leaf values
        r[a]   = (q[a] − ev)·mask[a]   # instantaneous regret (cfr6.py:378)
        sigma_m= RM+(adv + r)          # blueprint regret + one fresh iteration

    The float32 round-trips inside `_strategy_from_advantages` and the float64
    ev/r arithmetic match the stub exactly, so K=1 is bit-identical to it.
    """
    adv = np.asarray(adv, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    sigma0 = _strategy_from_advantages(adv.astype(np.float32), mask.astype(np.float32))
    ev = float((sigma0 * q * mask).sum())
    r = (q - ev) * mask
    sigma_m = _strategy_from_advantages((adv + r).astype(np.float32),
                                        mask.astype(np.float32))
    return sigma_m, sigma0


def _hero_action_values(root, ctx: SubgameSolveContext) -> tuple[np.ndarray, bool]:
    """Hero's per-action value q[a] from the root's immediate children (depth-1).

    LEAF child   → `leaf_value[hero]` (populated by `evaluate_leaves`; design Q1/Q6).
    TERMINAL child → `icm_adjust_returns(terminal_returns, ...)[hero]` (cfr6.py:278-285).
    Mirrors the Stage-G stub's per-child q extraction (ablation_decision_level.py:307-326),
    but READS the cached `leaf_value` instead of re-running rollouts. Returns
    (q: (7,) float64, degraded), where `degraded` is True if any reached LEAF lacks
    a `leaf_value` (e.g. a budget-truncated batch) — that action contributes q=0 and
    sub-step 5 can fall back to blueprint.

    K=1 uses ONLY the root's children (the depth-1 subgame); deeper descent is the
    Stage-3-C multi-iteration loop.
    """
    q = np.zeros(_N_ACTIONS, dtype=np.float64)
    degraded = False
    for child in root.children:
        a = int(child.action_from_parent)
        if child.is_terminal:
            icm = icm_adjust_returns(list(child.terminal_returns),
                                     list(ctx.starting_stacks), list(ctx.payouts))
            q[a] = float(icm[ctx.hero_seat])
        elif child.is_leaf:
            if child.leaf_value is None:
                degraded = True
                q[a] = 0.0
            else:
                q[a] = float(child.leaf_value[ctx.hero_seat])
        else:
            # A depth-1 root child is always LEAF or TERMINAL (subgame.py:267-293):
            # the terminal check precedes the depth cut, and a chance node at the
            # depth limit becomes a LEAF. A DECISION/CHANCE child here means the tree
            # was not built at max_action_depth=1 — degrade rather than guess.
            degraded = True
            q[a] = 0.0
    return q, degraded


# ============================================================
# Multi-iteration vanilla weighted CFR loop (Stage 3-C: the K>1 case)
# ============================================================

def _leaf_terminal_hero_values(tree: SubgameTree,
                               ctx: SubgameSolveContext) -> tuple[dict, bool]:
    """Precompute each LEAF/TERMINAL node's hero scalar value ONCE.

    These are iteration-invariant (design Q1/Q6: leaf values are fixed for the whole
    solve), so the CFR loop must not recompute `icm_adjust_returns` per iteration —
    that would put the dominant cost back inside the loop. LEAF → leaf_value[hero];
    TERMINAL → icm_adjust_returns(terminal_returns, ...)[hero] (cfr6.py:278-285).
    Returns (values: {id(node): float}, degraded), degraded=True if any LEAF lacks a
    leaf_value (budget-truncated batch) — that node contributes 0 and sub-step 5 can
    fall back to blueprint."""
    hero = ctx.hero_seat
    stacks = list(ctx.starting_stacks)
    payouts = list(ctx.payouts)
    values: dict = {}
    degraded = False
    for node in tree.all_nodes:
        if node.is_terminal:
            icm = icm_adjust_returns(list(node.terminal_returns), stacks, payouts)
            values[id(node)] = float(icm[hero])
        elif node.is_leaf:
            if node.leaf_value is None:
                degraded = True
                values[id(node)] = 0.0
            else:
                values[id(node)] = float(node.leaf_value[hero])
    return values, degraded


def _run_cfr(tree: SubgameTree, ctx: SubgameSolveContext,
             cache: _WarmupCache) -> dict:
    """The K-iteration vanilla weighted CFR loop (Decisions 2 & 4).

    Vanilla, NOT external sampling: every iteration visits every node, weighting
    chance children by `chance_prob` (subgame.py:114-117) and opponent children by
    the FIXED blueprint strategy (Decision 4). Only HERO accumulates regret. The
    per-node hero counterfactual value is a scalar (hero's slice) — we never need the
    other seats' values because opponents do not update.

    Per iteration t (weight w_t = t, linear LCFR averaging):
      hero node I → σ_pre = RM+(R[I]); q[a] = value(child_a); ev = Σ σ_pre·q;
                    R[I] = max(R[I] + (q−ev)·mask, 0)  (RM+ clamp);
                    σ_post = RM+(R[I]); S[I] += w_t · σ_post; return ev.
    Warm-start R[I] = blueprint advantages (cache.adv[I]), so K=1 reduces to the
    Stage-G stub (RM+(adv + r)); K>1 accumulates further iterations.

    Returns a dict of diagnostics: root_policy ((7,) float32 average strategy),
    converged_l1_tail (L1 between the AVERAGE root policy at iters K-1 and K, NaN if
    K<2 — see the SubgameSolveResult field note on why the average, not the current
    strategy), degraded, n_run, convergence_history (sampled (t, l1) pairs),
    root_q_values (hero's per-action Q at the root from the final iteration), and
    root_advantages_refined (the hero root's cumulative regret accumulator).

    Per-node regret/strategy tables are keyed by id(node): this single-deal subgame's
    hero infosets are singletons (subgame.py:158-162 — chance subsampling keeps
    sampled-subgame decision nodes distinct; sub-step 3 is told not to merge them).
    """
    hero = ctx.hero_seat
    K = ctx.n_iterations
    values, degraded = _leaf_terminal_hero_values(tree, ctx)

    # Hero regret (R) + linearly-weighted average-strategy (S) accumulators.
    R: dict = {}
    S: dict = {}
    for node in iter_decision_nodes(tree):
        if node.current_player == hero:
            nid = id(node)
            R[nid] = cache.adv[nid].astype(np.float64)
            S[nid] = np.zeros(_N_ACTIONS, dtype=np.float64)

    root_id = id(tree.root)
    root_q_holder = [np.zeros(_N_ACTIONS, dtype=np.float64)]  # last-iter root Q

    def traverse(node, w: float) -> float:
        if node.is_leaf or node.is_terminal:
            return values[id(node)]
        if node.is_chance:
            tot = 0.0
            for child in node.children:
                p = child.chance_prob
                if p:
                    tot += p * traverse(child, w)
            return tot
        nid = id(node)
        if node.current_player != hero:
            # Opponent: fixed blueprint strategy weights the children (Decision 4).
            sig = cache.sigma[nid]
            tot = 0.0
            for child in node.children:
                p = float(sig[int(child.action_from_parent)])
                if p != 0.0:
                    tot += p * traverse(child, w)
            return tot
        # Hero decision node.
        mask = cache.mask[nid]
        Rn = R[nid]
        sigma_pre = _strategy_from_advantages(Rn.astype(np.float32), mask)
        q = np.zeros(_N_ACTIONS, dtype=np.float64)
        for child in node.children:
            q[int(child.action_from_parent)] = traverse(child, w)
        ev = float((sigma_pre * q * mask).sum())
        if nid == root_id:
            root_q_holder[0] = q  # capture the final iteration's root Q (overwritten each iter)
        R[nid] = np.maximum(Rn + (q - ev) * mask, 0.0)  # RM+ clamp
        sigma_post = _strategy_from_advantages(R[nid].astype(np.float32), mask)
        S[nid] += w * sigma_post
        return ev

    # converged_l1_tail tracks the AVERAGE (output) root-policy movement between the
    # last two iterations — the design-doc D "root-policy L1 change", and the signal
    # Stage 3-E needs to pick K. NOT the CURRENT-strategy consecutive L1: with
    # opponents fixed (Decision 4) the subgame is a best-response problem, and RM+
    # drives the current strategy to its (pure) BR in ~1-2 iterations, so its
    # consecutive L1 collapses to 0 immediately and cannot inform K. The average
    # output keeps moving ~1/K and is what "has the solve converged" actually means.
    prev_avg = None
    l1_tail = float("nan")
    history: list = []
    for t in range(1, K + 1):
        traverse(tree.root, float(t))
        avg_t = S[root_id] / (t * (t + 1) / 2.0)  # running linear-weighted average
        if prev_avg is not None:
            l1_tail = float(np.abs(avg_t - prev_avg).sum())
            if t % 10 == 0 or t == K:
                history.append((t, l1_tail))
        prev_avg = avg_t

    return {
        "root_policy": prev_avg.astype(np.float32),
        "converged_l1_tail": l1_tail,
        "degraded": degraded,
        "n_run": K,
        "convergence_history": tuple(history),
        "root_q_values": root_q_holder[0].astype(np.float64),
        "root_advantages_refined": R[root_id].astype(np.float64),
    }


# ============================================================
# Entry point (Stage 3-A: K=0; Stage 3-B: K=1; Stage 3-C: K>1)
# ============================================================

def solve_subgame(tree: SubgameTree, ctx: SubgameSolveContext) -> SubgameSolveResult:
    """Solve the depth-limited subgame; return hero's refined root policy.

    Pre-condition: `subgame_leaf.evaluate_leaves(tree, leaf_ctx)` has populated
    every LEAF's `leaf_value` (the K>=1 loop reads them; the K=0 path does not).

    Builds the warm-up cache, then: K=0 returns the blueprint's masked root policy
    unchanged; K=1 applies one regret update (bit-identical to the Stage-G stub);
    K>1 runs the multi-iteration vanilla weighted CFR loop. All paths populate the
    Stage-3-D diagnostic fields (see SubgameSolveResult / `summarize_solve_result`).
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

    root_adv = cache.adv[nid]

    if ctx.n_iterations == 0:
        # K=0 — no refinement: hero plays the blueprint's own action distribution.
        # No leaves are read, so root_q_values is not meaningful (reported as zeros)
        # and the refined advantages equal the blueprint warm-start.
        return SubgameSolveResult(
            root_policy=root_sigma0.copy(),
            root_blueprint=root_sigma0.copy(),
            legal_mask=root_mask.copy(),
            hero_seat=ctx.hero_seat,
            n_iterations=0,
            n_decision_nodes_cached=len(cache.adv),
            degraded=False,
            n_iterations_run=0,
            convergence_history=(),
            root_q_values=np.zeros(_N_ACTIONS, dtype=np.float64),
            root_advantages_blueprint=root_adv.copy(),
            root_advantages_refined=root_adv.astype(np.float64),
        )

    if ctx.n_iterations == 1:
        # K=1 — a single root regret update over the depth-1 subgame. Bit-identical
        # to the Stage-G stub (the seed-to-plant continuity gate). Uses only the
        # root's immediate children; no deeper traversal (that is Stage 3-C).
        q, degraded = _hero_action_values(root, ctx)
        sigma_m, sigma0 = _regret_matched_policy(root_adv, q, root_mask)
        # Diagnostics: the refined accumulator after one RM+ update (matches _run_cfr
        # R[root] at K=1): max(adv + (q - ev)*mask, 0), ev under the blueprint σ0.
        ev = float((sigma0 * q * root_mask).sum())
        refined = np.maximum(root_adv.astype(np.float64) + (q - ev) * root_mask, 0.0)
        return SubgameSolveResult(
            root_policy=sigma_m,
            root_blueprint=root_sigma0.copy(),
            legal_mask=root_mask.copy(),
            hero_seat=ctx.hero_seat,
            n_iterations=1,
            n_decision_nodes_cached=len(cache.adv),
            degraded=degraded,
            n_iterations_run=1,
            convergence_history=(),
            root_q_values=q,
            root_advantages_blueprint=root_adv.copy(),
            root_advantages_refined=refined,
        )

    # K>1 — the full multi-iteration vanilla weighted CFR loop (Stage 3-C). Deeper
    # tree descent: opponents fixed at blueprint, chance weighted by chance_prob,
    # hero accumulates regret; output is the linearly-weighted average root strategy.
    out = _run_cfr(tree, ctx, cache)
    return SubgameSolveResult(
        root_policy=out["root_policy"],
        root_blueprint=root_sigma0.copy(),
        legal_mask=root_mask.copy(),
        hero_seat=ctx.hero_seat,
        n_iterations=ctx.n_iterations,
        n_decision_nodes_cached=len(cache.adv),
        degraded=out["degraded"],
        converged_l1_tail=out["converged_l1_tail"],
        n_iterations_run=out["n_run"],
        convergence_history=out["convergence_history"],
        root_q_values=out["root_q_values"],
        root_advantages_blueprint=root_adv.copy(),
        root_advantages_refined=out["root_advantages_refined"],
    )


# ============================================================
# Diagnostic summary (Stage 3-D) — for logging / sub-step 6 eval output
# ============================================================

def summarize_solve_result(result: SubgameSolveResult) -> dict:
    """JSON-serializable diagnostic summary of a solve (logging / sub-step 6 eval).

    6 decimal places on policies / advantages / Q-values (not load-bearing math),
    finer (9) on the tiny convergence L1s. NaN/None-safe: converged_l1_tail is NaN
    for K<2 → serialized as null. Adds `policy_shift_l1` = L1(root_policy,
    root_blueprint) — the clean "how far CFR pulled hero off the blueprint" MAGNITUDE
    (read root_advantages_refined for the direction; it is on a K-scaled magnitude).
    """
    def r6(arr):
        return None if arr is None else [round(float(x), 6)
                                         for x in np.asarray(arr).ravel()]

    def fnum(x, nd):
        return (round(float(x), nd)
                if (x is not None and math.isfinite(float(x))) else None)

    policy_shift = float(np.abs(np.asarray(result.root_policy)
                                - np.asarray(result.root_blueprint)).sum())
    return {
        "hero_seat": int(result.hero_seat),
        "n_iterations": int(result.n_iterations),
        "n_iterations_run": int(result.n_iterations_run),
        "n_decision_nodes_cached": int(result.n_decision_nodes_cached),
        "degraded": bool(result.degraded),
        "converged_l1_tail": fnum(result.converged_l1_tail, 9),
        "policy_shift_l1": round(policy_shift, 6),
        "root_policy": r6(result.root_policy),
        "root_blueprint": r6(result.root_blueprint),
        "legal_mask": [int(x) for x in np.asarray(result.legal_mask).ravel()],
        "root_q_values": r6(result.root_q_values),
        "root_advantages_blueprint": r6(result.root_advantages_blueprint),
        "root_advantages_refined": r6(result.root_advantages_refined),
        "convergence_history": [[int(t), fnum(l, 9)]
                                for t, l in result.convergence_history],
    }
