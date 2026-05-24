"""Depth-limited subgame LEAF evaluator (Track B1c, sub-step 2).

Assigns each `SubgameNode` of kind LEAF a per-seat ICM-equity-delta 6-vector — the
same units `cfr6.traverse_6max` backs up at true terminals — so sub-step 3's CFR
loop can treat LEAF and TERMINAL nodes uniformly. Full design and rationale:
`docs/SUBGAME_LEAF_DESIGN.md` (the contract this module conforms to).

STATUS: STAGE C — PROFILE_SAMPLE mode implemented. BEST_RESPONSE (Stage D) and the
`evaluate_leaves` batch path (Stage E) are not yet implemented and raise
NotImplementedError with a message naming the stage that adds them.

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
    (Stage A); rollouts run through it, not the canonical path.
  - `icm_returns.icm_adjust_returns` — terminal chip-returns → ICM-equity delta.
  - `icm.icm_equity` / `icm.is_itm` — the Option-A ITM short-circuit (Q5).

Return-type note (deviation from Q9's literal `-> tuple[float, ...]`):
  `evaluate_leaf` returns a `LeafEvalResult` carrying the 6-vector AND the
  `degraded` flag. Q9 sketched a bare tuple, but Q8 / the Stage-C spec require a
  `leaf_eval_degraded` flag on the result (budget breach, all-samples-failed,
  Option-A fallback). A bare tuple cannot carry it, so the result is a small
  record; the 6-vector is `result.value`, which is what `evaluate_leaves` writes
  to `node.leaf_value`.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from src.nlhe.actions import DiscreteAction
from src.nlhe.icm import icm_equity, is_itm
from src.nlhe.icm_returns import icm_adjust_returns
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.fast_view import fast_view_and_discretize
from src.nlhe.subgame import SubgameNode, SubgameTree

# 6-max; kept local so importing this module does not pull torch (networks6).
_NUM_SEATS = 6
_N_ACTIONS = len(DiscreteAction)  # 7
# Safety cap on rollout length, mirroring eval_pool.py:122. Healthy hands
# terminate far below this; the cap only catches a pathological non-terminating
# state (treated as a failed sample, Q8).
_MAX_ROLLOUT_STEPS = 500


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
        (236-dim feature vector for the acting seat) and `reset_cache()`.
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
            distribution over biases. Track C1 nudges it; never hardcoded.
        n_samples: MC rollouts per (leaf, strategy) — the design's M.
            DEFAULT = 8 is PROVISIONAL: it is the optimistic-batching assumption
            pending Stage E's measured batched cost. If Stage E shows M=8 does not
            fit Z = 1.5 s, the default drops to M = 5 (still within Q4's stderr
            bounds). The knob is live; do NOT fix it to 5 prematurely — keep 8 as
            the default and let Stage E's measurement choose.
        rng: random source for chance/bias/action sampling. None → a fresh
            `random.Random()` is created at eval time. NOTE: reproducibility
            (same seed → identical result) holds because `evaluate_leaf` resets
            the encoder bucket cache at the start of each call; pass a freshly
            seeded `random.Random(seed)` per call to reproduce.
        icm_short_circuit: if True, leaves where `is_itm(starting_stacks, num_paid)`
            holds skip the rollout and use the Option-A ICM estimate directly (Q5).
        time_budget_s: wall-clock guard. None → unbounded. When exceeded, the
            evaluator stops sampling between rollouts, returns the mean of the
            completed samples, and sets `degraded=True` (Q4/Q8). Checked between
            rollouts (a rollout is not interrupted mid-flight).
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
# Result (Q8 degraded flag — see module return-type note)
# ============================================================

@dataclass
class LeafEvalResult:
    """A leaf's evaluated value plus diagnostics.

    Attributes:
        value: per-seat ICM-equity-delta 6-vector (length NUM_SEATS_6MAX, Q1).
            This is what gets written to `SubgameNode.leaf_value`.
        degraded: the `leaf_eval_degraded` flag (Q8). True when the value is a
            fallback — budget breach (partial samples), all samples failed, or
            Option-A could not be computed (then `value` is the zero-vector).
        n_completed: MC rollouts that actually completed (0 for short-circuit /
            total failure). Diagnostic.
        short_circuited: True if the ITM Option-A short-circuit (Q5) was taken
            (NOT a degradation — a deliberate, exact-in-the-ITM-regime shortcut).
    """
    value: tuple
    degraded: bool = False
    n_completed: int = 0
    short_circuited: bool = False


# ============================================================
# Internal rollout machinery
# ============================================================

def _legal_mask(discrete_to_chip: dict) -> np.ndarray:
    m = np.zeros(_N_ACTIONS, dtype=np.float32)
    for da in discrete_to_chip:
        m[int(da)] = 1.0
    return m


def _blueprint_probs(ctx: LeafEvalContext, parsed: dict, cp: int,
                     discrete_to_chip: dict, rng) -> Optional[np.ndarray]:
    """RM+ masked blueprint 7-vector for seat `cp`, or None on NaN (Q8).

    Mirrors `eval_6max_self_play._sample_action_from_policy` lines 137-143:
    encode → predict_advantages → RM+ (positive part, masked, normalized);
    all-zero advantages fall back to uniform-over-legal.
    """
    feat = ctx.blueprint.encoder.encode_from_parsed(parsed, rng=rng)
    features = np.asarray(feat, dtype=np.float32)
    adv = np.asarray(ctx.blueprint.policy_nets.predict_advantages(cp, features),
                     dtype=np.float32)
    if not np.all(np.isfinite(adv)):
        return None
    mask = _legal_mask(discrete_to_chip)
    positive = np.maximum(adv, 0.0) * mask
    total = float(positive.sum())
    if total > 0.0:
        probs = positive / total
    else:
        s = float(mask.sum())
        if s <= 0.0:
            return None
        probs = mask / s
    if not np.all(np.isfinite(probs)):
        return None
    return probs


def _sample_action(dist7: np.ndarray, discrete_to_chip: dict, rng) -> int:
    """Sample a chip action from a 7-dim distribution masked to legal actions."""
    idx = rng.choices(range(_N_ACTIONS), weights=dist7.tolist(), k=1)[0]
    da = DiscreteAction(idx)
    chip = discrete_to_chip.get(da)
    if chip is None:  # belt-and-suspenders (dist7 is masked to legal)
        da = rng.choice(list(discrete_to_chip.keys()))
        chip = discrete_to_chip[da]
    return int(chip)


def _rollout_once(state, ctx: LeafEvalContext, action_dist_fn, rng):
    """One MC rollout from `state` (a CLONE, mutated in place) to terminal.

    `action_dist_fn(blueprint_probs7, legal_mask7, current_player) -> dist7`
    supplies the acting policy at each decision (PROFILE_SAMPLE passes a biased
    blueprint; tests can pass a raw-blueprint fn). Chance is sampled from
    `state.chance_outcomes()`.

    Returns the per-seat ICM-equity-delta 6-list at the natural terminal, or
    None on failure (NaN probs, degenerate dist, non-termination within the cap,
    or non-finite ICM) — the caller drops None samples (Q8).
    """
    steps = 0
    while (not state.is_terminal()) and steps < _MAX_ROLLOUT_STEPS:
        steps += 1
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions = [o[0] for o in outcomes]
            probs = [o[1] for o in outcomes]
            state.apply_action(int(rng.choices(actions, weights=probs, k=1)[0]))
            continue
        cp = state.current_player()
        parsed = parse_state_6max(state)
        view, discrete_to_chip, _legal = fast_view_and_discretize(state, parsed)
        if not discrete_to_chip:
            return None
        probs7 = _blueprint_probs(ctx, parsed, cp, discrete_to_chip, rng)
        if probs7 is None:
            return None
        dist7 = action_dist_fn(probs7, _legal_mask(discrete_to_chip), cp)
        dist7 = np.asarray(dist7, dtype=np.float64)
        if dist7.shape != (_N_ACTIONS,) or not np.all(np.isfinite(dist7)) \
                or float(dist7.sum()) <= 0.0:
            return None
        chip = _sample_action(dist7, discrete_to_chip, rng)
        state.apply_action(int(chip))

    if not state.is_terminal():
        return None
    chip_returns = list(state.returns())
    icm = icm_adjust_returns(chip_returns, list(ctx.starting_stacks),
                             list(ctx.payouts))
    if not all(math.isfinite(x) for x in icm):
        return None
    return icm


def _seat_prior_weights(prior, seat: int, k: int) -> list:
    """Per-seat sampling weights over the k biases (PROFILE_SAMPLE). None → uniform."""
    if prior is None:
        return [1.0 / k] * k
    arr = np.asarray(prior, dtype=float)
    row = arr if arr.ndim == 1 else arr[seat]
    w = [float(x) for x in row.tolist()]
    if len(w) != k or sum(w) <= 0.0:
        return [1.0 / k] * k
    return w


def _draw_opp_biases(ctx: LeafEvalContext, rng) -> dict:
    """Draw one bias index per opponent seat from the prior (PROFILE_SAMPLE).

    Drawn for ALL non-hero seats; folded/all-in seats never act, so their drawn
    bias is simply unused (the Q3 live-opponent reduction is implicit here — no
    need to enumerate live seats for the averaged form).
    """
    k = ctx.biased_blueprint.k
    out = {}
    for seat in range(_NUM_SEATS):
        if seat == ctx.hero_seat:
            continue
        w = _seat_prior_weights(ctx.opponent_prior, seat, k)
        out[seat] = rng.choices(range(k), weights=w, k=1)[0]
    return out


def _option_a(node: SubgameNode, ctx: LeafEvalContext):
    """Option-A ICM estimate at the leaf (Q5): icm_equity(current) - icm_equity(start).

    `current` stacks = chips behind at the leaf (parsed 'money'). Used by the ITM
    short-circuit and as the all-samples-failed fallback. Returns (value_list, ok).
    """
    try:
        parsed = parse_state_6max(node.state)
        current = list(parsed["money"])
        e_cur = icm_equity(current, list(ctx.payouts))
        e_start = icm_equity(list(ctx.starting_stacks), list(ctx.payouts))
        val = [float(e_cur[i] - e_start[i]) for i in range(len(e_cur))]
        if len(val) != _NUM_SEATS or not all(math.isfinite(x) for x in val):
            return [0.0] * _NUM_SEATS, False
        return val, True
    except Exception:
        return [0.0] * _NUM_SEATS, False


def _evaluate_profile_sample(node: SubgameNode, ctx: LeafEvalContext,
                             rng) -> LeafEvalResult:
    """PROFILE_SAMPLE: prior-weighted average over opponents' biases (Q3 fallback form).

    Per rollout, each opponent seat's bias is drawn from the prior; hero plays
    blueprint (bias 0). Averaging M such rollouts is an unbiased estimate of the
    prior-weighted leaf value. ITM short-circuit, wall-clock guard, and per-sample
    failure handling per Q5/Q8.
    """
    t0 = time.perf_counter()

    # Option-A ITM short-circuit (Q5): deliberate, not a degradation.
    if ctx.icm_short_circuit and is_itm(list(ctx.starting_stacks), ctx.num_paid):
        val, ok = _option_a(node, ctx)
        return LeafEvalResult(value=tuple(val), degraded=(not ok),
                              n_completed=0, short_circuited=True)

    acc = [0.0] * _NUM_SEATS
    completed = 0
    degraded = False
    for _ in range(ctx.n_samples):
        if ctx.time_budget_s is not None and \
                (time.perf_counter() - t0) > ctx.time_budget_s:
            degraded = True
            break
        opp = _draw_opp_biases(ctx, rng)

        def dist_fn(probs7, mask7, cp, _opp=opp):
            bias_idx = 0 if cp == ctx.hero_seat else _opp.get(cp, 0)
            return ctx.biased_blueprint.action_probs(probs7, mask7, bias_idx)

        vec = _rollout_once(node.state.clone(), ctx, dist_fn, rng)
        if vec is None:
            continue  # drop failed/NaN sample (Q8)
        for i in range(_NUM_SEATS):
            acc[i] += vec[i]
        completed += 1

    if completed == 0:
        # all samples failed (or budget hit before any completed) → Option-A,
        # else zero-vector. Either way degraded (Q8).
        val, _ok = _option_a(node, ctx)
        return LeafEvalResult(value=tuple(val), degraded=True, n_completed=0)

    mean = tuple(acc[i] / completed for i in range(_NUM_SEATS))
    return LeafEvalResult(value=mean, degraded=degraded, n_completed=completed)


# ============================================================
# Entry points (Q9)
# ============================================================

def evaluate_leaf(node: SubgameNode, ctx: LeafEvalContext) -> LeafEvalResult:
    """Evaluate a single LEAF node → `LeafEvalResult` (value 6-vector + degraded).

    Dispatches on `ctx.mode`:
      - PROFILE_SAMPLE: prior-weighted average over opponents' biases (Stage C).
      - BEST_RESPONSE: opponents best-respond among biases (Stage D — not yet).

    Resets the blueprint encoder's bucket cache at the start so a freshly seeded
    `ctx.rng` yields bit-identical bucket draws across calls (the cache otherwise
    persists and would desync rng usage); the cache still speeds up the M rollouts
    WITHIN this call.

    See the module return-type note for why this returns a `LeafEvalResult` rather
    than Q9's literal bare tuple.
    """
    rng = ctx.rng if ctx.rng is not None else random.Random()
    try:
        ctx.blueprint.encoder.reset_cache()
    except Exception:
        pass

    if ctx.mode == LeafEvalMode.PROFILE_SAMPLE:
        return _evaluate_profile_sample(node, ctx, rng)
    if ctx.mode == LeafEvalMode.BEST_RESPONSE:
        raise NotImplementedError(
            "evaluate_leaf: BEST_RESPONSE mode lands in Stage D. Stage C "
            "implements PROFILE_SAMPLE only."
        )
    raise ValueError(f"unknown LeafEvalMode: {ctx.mode!r}")


def evaluate_leaves(tree: SubgameTree, ctx: LeafEvalContext) -> None:
    """Populate `node.leaf_value` for every LEAF node in `tree`, IN PLACE.

    Side effect: mutates each leaf node — sets `node.leaf_value` to the evaluated
    6-vector (`evaluate_leaf(node, ctx).value`). Returns None. The batch entry
    point sub-step 3 calls ONCE before the CFR loop; where possible it batches the
    per-step network forward across leaves in lockstep (the Q4 batching argument).

    NOTE: NOT YET IMPLEMENTED — the batch path (and its lockstep batching) lands in
    Stage E. Sub-step 3 must NOT rely on `evaluate_leaves` being callable until
    then; call `evaluate_leaf` per leaf in the meantime if needed.
    """
    raise NotImplementedError(
        "evaluate_leaves: the batch path (lockstep-batched forward across leaves) "
        "lands in Stage E. Until then, mutation semantics are fixed (sets "
        "node.leaf_value in place, returns None); use evaluate_leaf per leaf."
    )
