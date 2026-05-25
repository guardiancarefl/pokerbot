"""Depth-limited subgame LEAF evaluator (Track B1c, sub-step 2).

Assigns each `SubgameNode` of kind LEAF a per-seat ICM-equity-delta 6-vector — the
same units `cfr6.traverse_6max` backs up at true terminals — so sub-step 3's CFR
loop can treat LEAF and TERMINAL nodes uniformly. Full design and rationale:
`docs/SUBGAME_LEAF_DESIGN.md` (the contract this module conforms to).

STATUS: STAGE E — PROFILE_SAMPLE and BEST_RESPONSE modes plus the
`evaluate_leaves` batch path (shared-cache) are implemented. The design budget
Z=1.5 s is NOT yet closed: the dominant cost is the encoder bucket-MC, not the
network forward (see evaluate_leaves and design doc Q4/Q12). The budget-closing
encoder bucket-MC precompute is the Stage E.5 follow-up.

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

import logging
import math
import random
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from src.nlhe.actions import DiscreteAction
from src.nlhe.icm import icm_equity, is_itm
from src.nlhe.icm_returns import icm_adjust_returns
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.fast_view import fast_view_and_discretize
from src.nlhe.subgame import SubgameNode, SubgameTree, iter_leaf_nodes

log = logging.getLogger("subgame_leaf")

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
            DEFAULT = 8 (the original design value). M is a REAL budget lever per
            the session-16 Q13 measurement: network forwards dominate BR cost
            (~45%) and scale linearly in M, so M directly multiplies the dominant
            cost (BR M=8 d3 ≈ 20.5 s vs M=5 ≈ 13.8 s per decision). The session-14
            drop to 5 was made under the wrong attribution (assumed bucket-MC
            dominated and M wouldn't help) and is reverted. Default 8 is chosen for
            MEASUREMENT QUALITY — M=5 carries ~26 % more MC stderr (sqrt(8/5)),
            which hurts when sub-step 6 measures small bb/100 deltas. The budget is
            closed by PARALLELISM, not sample-count: sub-step 6 is throughput-bound
            and parallelizes (Q13 / docs/STAGE_E_BUDGET_REDERIVATION.md), so M=8 at
            Y≈24 is ~0.85 s effective vs the ~27 s/decision budget. M=5 remains
            available as a cost-saving knob (e.g. real-time deployment latency).
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
        manage_cache_externally: cache-lifecycle control. False (default,
            standalone evaluate_leaf): evaluate_leaf resets the blueprint encoder's
            bucket cache at entry, and BR evaluation resets it per CRN sample —
            giving per-call reproducibility and clean (exact-tie) common random
            numbers. True (set by evaluate_leaves via dataclasses.replace): NO
            internal resets — the BATCH owns the cache (one reset for the whole
            tree), so the expensive encoder bucket-MC (~10.77 ms/cache-miss) is
            shared across all leaves and samples. The tradeoff: BR's per-sample
            CRN becomes approximate (the shared cache desyncs the bucket-MC rng
            between bias passes), trading a little argmax-ranking variance for the
            ~3-10x batch speedup. See evaluate_leaves.
    """
    blueprint: BlueprintProvider
    biased_blueprint: Any
    starting_stacks: Sequence[int]
    payouts: Sequence[float]
    hero_seat: int

    mode: LeafEvalMode = LeafEvalMode.BEST_RESPONSE
    opponent_prior: Optional[Any] = None
    n_samples: int = 8  # design M; restored 5->8 per session-16 Q13 (see docstring)
    rng: Optional[random.Random] = None
    icm_short_circuit: bool = True
    time_budget_s: Optional[float] = None
    num_paid: int = 3
    manage_cache_externally: bool = False


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


@dataclass
class LeafBatchResult:
    """Summary returned by `evaluate_leaves` (which also mutates the tree in place).

    Attributes:
        n_leaves: total leaf nodes in the tree.
        n_evaluated: leaf nodes whose `leaf_value` was populated this call.
        partial_eval_degraded: True if the wall-clock budget cut the batch off
            before all leaves were evaluated — the un-evaluated leaves keep
            `leaf_value = None`. (Naming note: the per-leaf result uses
            `degraded` for a fallback VALUE; this flag is the distinct
            batch-level concept of an INCOMPLETE pass. `eval_pool.py` uses
            `exceeded_cap` for an analogous budget breach.)
    """
    n_leaves: int
    n_evaluated: int
    partial_eval_degraded: bool = False


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
        starting = list(ctx.starting_stacks)
        # Exclude pre-busted seats (entered at stack 0) from both ICM endpoints,
        # consistent with icm_adjust_returns; otherwise busted-seat placeholders
        # in `current` would draw spurious equity (see icm.py `eligible`).
        eligible = [i for i in range(len(starting)) if starting[i] > 0]
        e_cur = icm_equity(current, list(ctx.payouts), eligible=eligible)
        e_start = icm_equity(starting, list(ctx.payouts), eligible=eligible)
        val = [float(e_cur[i] - e_start[i]) for i in range(len(e_cur))]
        if len(val) != _NUM_SEATS or not all(math.isfinite(x) for x in val):
            return [0.0] * _NUM_SEATS, False
        return val, True
    except Exception:
        return [0.0] * _NUM_SEATS, False


_TIE_EPS = 1e-9


def _reset_cache(ctx: LeafEvalContext) -> None:
    try:
        ctx.blueprint.encoder.reset_cache()
    except Exception:
        pass


def _past(deadline) -> bool:
    return deadline is not None and time.perf_counter() > deadline


def _bias_dist_fn(ctx: LeafEvalContext, biases: dict):
    """action_dist_fn for `_rollout_once`: hero plays bias 0 (blueprint); each
    opponent seat plays `biases.get(seat, 0)` (0 = blueprint default)."""
    hero = ctx.hero_seat
    bb = ctx.biased_blueprint

    def fn(probs7, mask7, cp):
        bias_idx = 0 if cp == hero else int(biases.get(cp, 0))
        return bb.action_probs(probs7, mask7, bias_idx)
    return fn


def _menu_biases(prior, seat: int, k: int) -> list:
    """BEST_RESPONSE menu (Q7): the biases the opponent may choose from.

    prior None → full menu [0, k). Otherwise a bias is IN the menu iff its weight
    is > 0 (weight 0 EXCLUDES it). The additive value-tilt Q7 also describes is
    DEFERRED to Track C1 — Stage D implements menu MEMBERSHIP only. So a positive
    non-uniform prior behaves like a uniform full menu in Stage D; only zeros
    matter. (Documented; revisit when C1 needs tilt.)
    """
    if prior is None:
        return list(range(k))
    arr = np.asarray(prior, dtype=float)
    row = arr if arr.ndim == 1 else arr[seat]
    menu = [b for b in range(k) if b < len(row) and float(row[b]) > 0.0]
    return menu if menu else list(range(k))


def _opponent_bias_values(node, ctx, o, menu, rng, deadline=None):
    """CRN evaluation rollouts for opponent `o` (Addition 1).

    For each bias b in `menu`, run ctx.n_samples rollouts with opponent o playing
    b, ALL OTHER opponents playing blueprint (bias 0), hero playing blueprint;
    collect opponent o's OWN ICM-equity-delta per sample. Common-random-numbers:
    the SAME per-sample seeds (with a reset cache) are reused across biases, so a
    bias-invariant policy yields identical samples (EXACT ties → clean lowest-index
    tie-break) and the bias comparison has reduced variance.

    NOTE: 'other opponents = blueprint' here follows Q3's single-pass approximation
    ('others held at blueprint'). The prompt's Addition 1 wording said 'other
    opponents playing their priors'; under prior-as-MENU (Q7) there is no sampling
    distribution to draw from in BR mode, so the blueprint baseline is used. (Flagged.)

    Returns ({bias: [per-sample o-values]}, budget_hit).
    """
    eval_seeds = [rng.randrange(2 ** 31) for _ in range(ctx.n_samples)]
    out = {b: [] for b in menu}
    for b in menu:
        dist_fn = _bias_dist_fn(ctx, {o: b})
        for m in range(ctx.n_samples):
            if _past(deadline):
                return out, True
            r = random.Random(eval_seeds[m])
            if not ctx.manage_cache_externally:
                # Per-sample reset gives clean CRN (exact ties) for standalone
                # evaluate_leaf. Under evaluate_leaves the batch owns the cache
                # (shared across samples), so we skip it — CRN is then approximate.
                _reset_cache(ctx)
            vec = _rollout_once(node.state.clone(), ctx, dist_fn, r)
            if vec is not None:
                out[b].append(vec[o])
    return out, False


def _best_response_biases(node, ctx, rng, deadline=None):
    """Per-opponent independent best-response bias selection (Q3, Addition 1).

    Live opponents = non-hero seats with chips behind (money > 0): excludes
    all-in / busted seats (Q3). Folded-but-chipped seats stay in the loop but
    never act, so their per-bias values tie and they resolve to bias 0 (harmless;
    pruning folded seats is a Stage-E perf optimization, not a Stage-D concern).

    For each live opponent, argmax over its menu of mean opponent-value, with a
    deterministic LOWEST-INDEX tie-break (Q8). Returns (br_dict, budget_hit).
    """
    parsed = parse_state_6max(node.state)
    money = parsed["money"]
    live = [o for o in range(_NUM_SEATS)
            if o != ctx.hero_seat and o < len(money) and money[o] > 0]
    k = ctx.biased_blueprint.k
    br = {}
    for o in live:
        menu = _menu_biases(ctx.opponent_prior, o, k)
        vals, hit = _opponent_bias_values(node, ctx, o, menu, rng, deadline)
        if hit:
            return br, True
        means = {b: (sum(v) / len(v) if v else float("-inf"))
                 for b, v in vals.items()}
        max_v = max(means.values())
        br[o] = min(b for b in menu if means[b] >= max_v - _TIE_EPS)
    return br, False


def _evaluate_best_response(node: SubgameNode, ctx: LeafEvalContext,
                           rng) -> LeafEvalResult:
    """BEST_RESPONSE: opponents independently best-respond among biases (Q3).

    Phase 1 selects each live opponent's BR bias (CRN eval rollouts, others=
    blueprint). Phase 2 ('value-collecting rollouts') runs ctx.n_samples rollouts
    with every opponent at its selected BR bias and hero at blueprint; the mean
    ICM-equity-delta 6-vector is the leaf value. ITM short-circuit, wall-clock
    guard, and per-sample failure handling match PROFILE_SAMPLE.

    Cost per leaf: v×k×M_eval evaluation rollouts + M_value value rollouts. Stage D
    keeps M_eval = M_value = ctx.n_samples (Addition 1); Stage E may split them.
    """
    t0 = time.perf_counter()
    deadline = (t0 + ctx.time_budget_s) if ctx.time_budget_s is not None else None

    if ctx.icm_short_circuit and is_itm(list(ctx.starting_stacks), ctx.num_paid):
        val, ok = _option_a(node, ctx)
        return LeafEvalResult(value=tuple(val), degraded=(not ok),
                              n_completed=0, short_circuited=True)
    if _past(deadline):
        val, _ok = _option_a(node, ctx)
        return LeafEvalResult(value=tuple(val), degraded=True, n_completed=0)

    br, hit = _best_response_biases(node, ctx, rng, deadline)
    if hit:
        val, _ok = _option_a(node, ctx)
        return LeafEvalResult(value=tuple(val), degraded=True, n_completed=0)

    # Phase 2: value-collecting rollouts under the composed BR profile.
    # Reset the bucket cache ONLY when this call owns it (standalone evaluate_leaf,
    # for clean per-call CRN). Under evaluate_leaves (manage_cache_externally=True)
    # the BATCH owns the cache and resetting here would defeat the Stage-E shared
    # cache once per leaf. The reset has no correctness role: bucket_of is fully
    # deterministic (seeded from sha256(sorted(hero,board)), rng discarded), so a
    # warm cache returns the same buckets a cold one would, and the cache path
    # never advances ctx.rng — so skipping the reset leaves leaf values bit-identical
    # while cutting redundant bucket-MC. Measured: BR M=5 on the 64-leaf depth-3
    # tree 16.7s -> 13.0s, bucket-MC misses 417 -> 83, all 64 leaf values identical.
    if not ctx.manage_cache_externally:
        _reset_cache(ctx)
    biases = {o: br.get(o, 0) for o in range(_NUM_SEATS) if o != ctx.hero_seat}
    dist_fn = _bias_dist_fn(ctx, biases)
    acc = [0.0] * _NUM_SEATS
    n = 0
    degraded = False
    for _ in range(ctx.n_samples):
        if _past(deadline):
            degraded = True
            break
        vec = _rollout_once(node.state.clone(), ctx, dist_fn, rng)
        if vec is None:
            continue
        for i in range(_NUM_SEATS):
            acc[i] += vec[i]
        n += 1
    if n == 0:
        val, _ok = _option_a(node, ctx)
        return LeafEvalResult(value=tuple(val), degraded=True, n_completed=0)
    mean = tuple(acc[i] / n for i in range(_NUM_SEATS))
    return LeafEvalResult(value=mean, degraded=degraded, n_completed=n)


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
        vec = _rollout_once(node.state.clone(), ctx, _bias_dist_fn(ctx, opp), rng)
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

    Cache lifecycle: when `ctx.manage_cache_externally` is False (default,
    standalone use), resets the blueprint encoder's bucket cache at the start so a
    freshly seeded `ctx.rng` yields reproducible bucket draws across calls (the
    cache otherwise persists and would desync rng usage); the cache still speeds up
    the M rollouts WITHIN this call. When True (called by evaluate_leaves), the
    batch owns the cache and this reset is skipped so it is shared across leaves.

    See the module return-type note for why this returns a `LeafEvalResult` rather
    than Q9's literal bare tuple.
    """
    rng = ctx.rng if ctx.rng is not None else random.Random()
    if not ctx.manage_cache_externally:
        try:
            ctx.blueprint.encoder.reset_cache()
        except Exception:
            pass

    if ctx.mode == LeafEvalMode.PROFILE_SAMPLE:
        return _evaluate_profile_sample(node, ctx, rng)
    if ctx.mode == LeafEvalMode.BEST_RESPONSE:
        return _evaluate_best_response(node, ctx, rng)
    raise ValueError(f"unknown LeafEvalMode: {ctx.mode!r}")


def evaluate_leaves(tree: SubgameTree, ctx: LeafEvalContext) -> LeafBatchResult:
    """Evaluate every LEAF in `tree`, MUTATING `node.leaf_value` in place.

    The batch entry point sub-step 3 calls ONCE before the CFR loop. Side effect:
    sets `node.leaf_value = list(evaluate_leaf(node, ...).value)` for each leaf
    from `iter_leaf_nodes(tree)`. Also returns a `LeafBatchResult` summary (Q9
    sketched a None return; the summary carries the `partial_eval_degraded` flag
    the budget guard needs — leaf values are still delivered via the in-place
    mutation, so sub-step 3's `node.leaf_value` read is unchanged).

    Cache lifecycle (Stage E — the measured budget lever): resets the blueprint
    encoder's bucket cache exactly ONCE here, then runs all leaves with
    `manage_cache_externally=True` so NO further resets happen. The dominant
    leaf-eval cost is `encode_from_parsed`'s bucket-MC (~10.77 ms per distinct
    (hero, board) cache-miss); sharing the cache across all leaves and samples
    turns repeated boards into ~0.025 ms hits — the 3x (PROFILE) / ~10x (BR
    single-leaf) speedup measured in Stage E. NOTE: the design budget Z=1.5 s is
    NOT closed by this alone (see design doc Q4/Q12); the bucket-MC precompute
    (Stage E.5) is the budget-closing follow-up. Lockstep NETWORK batching from
    the original Q4 plan optimizes a non-bottleneck (network forward is 0.25 ms
    single / 0.45 us/row batched, vs the 10.77 ms bucket-MC) and is not done here.

    Budget guard: `ctx.time_budget_s` bounds the WHOLE batch, checked between
    leaves (a leaf in progress is not interrupted). On exhaustion, remaining
    leaves keep `leaf_value=None`, `partial_eval_degraded=True` is returned, and a
    warning is logged.
    """
    leaves = list(iter_leaf_nodes(tree))
    n = len(leaves)
    if n == 0:
        return LeafBatchResult(n_leaves=0, n_evaluated=0, partial_eval_degraded=False)

    # ONE cache reset for the whole batch; children run cache-managed-externally so
    # they neither reset at entry nor per BR sample (shared cache = the speedup).
    _reset_cache(ctx)
    batch_ctx = replace(
        ctx,
        manage_cache_externally=True,
        time_budget_s=None,  # batch enforces a tree-wide deadline between leaves
        rng=ctx.rng if ctx.rng is not None else random.Random(),
    )

    t0 = time.perf_counter()
    deadline = (t0 + ctx.time_budget_s) if ctx.time_budget_s is not None else None
    n_eval = 0
    for leaf in leaves:
        if _past(deadline):
            break
        res = evaluate_leaf(leaf, batch_ctx)
        leaf.leaf_value = list(res.value)
        n_eval += 1

    partial = n_eval < n
    if partial:
        log.warning(
            "evaluate_leaves: time budget (%.4fs) exhausted after %d/%d leaves; "
            "remaining leaves keep leaf_value=None (partial_eval_degraded).",
            ctx.time_budget_s if ctx.time_budget_s is not None else -1.0, n_eval, n,
        )
    return LeafBatchResult(n_leaves=n, n_evaluated=n_eval, partial_eval_degraded=partial)
