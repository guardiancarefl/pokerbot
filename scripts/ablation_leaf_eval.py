"""Stage F — Q11 Level 1 leaf-level ablation (Track B1c, sub-step 2 closure gate).

Directional confirmation that BEST_RESPONSE leaf evaluation produces meaningfully
different, correct-direction leaf values vs PROFILE_SAMPLE(uniform) and a
pure-blueprint baseline — BEFORE investing in sub-step 3's real CFR loop.

WHAT Q11 LEVEL 1 ASKS (design doc Q11 + Q10 #9): on ~50 leaf states spanning
streets / stack-depths / live-opponent counts, confirm the ordering
  per live opponent i:  max_b V_i(b)  >=  mean_b V_i(b)      (BR >= uniform)
  hero:                 V_hero(BR)    <=  V_hero(uniform)
and quantify the per-leaf hero-value gap.

MEASUREMENT DISCIPLINE / KNOWN NOISE (session-14 Stage-D finding, recorded in
tests/test_subgame_leaf.py::test_biases_produce_different_rollouts): the RAW
per-leaf hero-value directional signal is tiny (~0.05) and washes out against
per-hand ICM variance (~2.0) even at M>=120 — a 15bb preflop hero often folds
immediately, so opponent style barely moves hero value at a single leaf. So we do
NOT gate on a per-leaf hero MC delta. Instead:

  PRIMARY (robust) signals — the SPLIT METRIC that gates (session-17 revision; the
  prior single "frac BR selects non-blueprint >= 20%" gate conflated sampling
  resolution, blueprint convergence quality, and the mechanism itself):
    (1) RESOLUTION rate: per (leaf, opp) pair, is the most blueprint-DEVIATING bias
        (either direction) detectably different from blueprint under CRN PAIRING
        (best_bias[m] - bias0[m]; pairing cancels the shared per-deal variance that
        otherwise swamps the small bias signal — session-14)? > 3σ on the paired
        diff = "resolved". A LOW resolution rate means M was too small to see bias
        effects — a measurement-budget limit, not an architecture failure.
    (2) DIFFERENTIATION rate: among RESOLVED pairs only, does BR's argmax select a
        NON-blueprint bias? High = BR picks biases that beat the blueprint. Low (at
        high resolution) = the blueprint is already near opponent-self-optimal here
        — a valid finding, distinct from a collapsed mechanism.
  SECONDARY (noisy, reported with caveats, NOT gating):
    (3) Hero-value direction: aggregate mean over leaves of (V_hero(PROFILE) -
        V_hero(BR)) should be >= 0 (BR opponents hurt hero more than uniform-mixed).
        Per-leaf this is noise-dominated; the AGGREGATE SIGN across ~50 leaves and
        CRN-paired deltas are what carry signal.

LEAF SELECTION (documented deviation from Q11's "hand-built"): we sample reachable
leaf states from the PRODUCTION game (seeded sample_starting_state -> walk to a
decision at a target street via calls), spanning preflop/flop/turn/river and the
sampler's stack-depth / alive-count distribution. Rationale: real reachable states,
fully reproducible, avoids hand-construction error, honors the "tests use the
production game" discipline, and spans the same axes Q11 names. We filter to
non-ITM leaves (is_itm short-circuits to a deterministic Option-A value carrying no
BR signal) with >= 1 live opponent (else BR == baseline trivially).

WHAT IF IT FAILS (Path A discipline): the split metric distinguishes the failure
modes. LOW resolution => M too small (re-run heavier, not an architecture finding).
HIGH resolution + LOW differentiation, below the resolution>=90% pass clause =>
AMBIGUOUS (re-run at higher M/N). A genuine FAIL is the mechanism producing no
detectable, blueprint-beating bias selection where resolution says it should. On
any FAIL/AMBIGUOUS the correct response is to STOP and surface to the human for a
diagnostic round — NOT to rationalize the result or proceed to Stage G. The whole
point of Path A is that F/G catch problems before sub-step 3's heavier work; that
gate only works if honored.

Usage:
    python -m scripts.ablation_leaf_eval               # N=50, M=8 (the gate run)
    python -m scripts.ablation_leaf_eval --smoke       # N=3, M=4 (fast correctness)
    python -m scripts.ablation_leaf_eval --n 50 --samples 8 --seed 17
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyspiel

from src.nlhe.abstraction import Abstraction
from src.nlhe.biased_policy import BiasedBlueprint
from src.nlhe.game_strings import TournamentStructure, six_max_sng
from src.nlhe.icm import is_itm, sng_payouts_6max_double_up
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.stack_sampler import sample_starting_state
from src.nlhe.subgame import SubgameNode, NodeKind
from src.nlhe.subgame_leaf import (
    LeafEvalContext, LeafEvalMode,
    _best_response_biases, _opponent_bias_values, _menu_biases,
    _bias_dist_fn, _draw_opp_biases, _rollout_once, _reset_cache,
)
from scripts.eval_6max_self_play import _load_solver
from scripts.bench_subgame_leaf import Instr

ABSTR = "runs/abstraction_20260521_223018/abstraction.pkl"
CKPT = "runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/ckpt_iter_3000.pt"
STRUCT = "configs/ignition_double_up_6max_turbo.yaml"
NUM_PAID = 3
N_SEATS = 6
_EPS = 1e-9


# ----------------------------------------------------------------------------
# Leaf-state battery (sampled from the production game)
# ----------------------------------------------------------------------------

def _sample_chance(state, rng):
    a, p = zip(*state.chance_outcomes())
    return state.child(int(rng.choices(a, weights=p, k=1)[0]))


def _walk_to_street_decision(state, rng, target_street):
    """Advance to a decision on `target_street` (0=preflop..3=river) via calls.

    Deals the initial chance, then for each street to advance: everyone
    calls/checks until the betting round closes (a chance node), deal it, repeat.
    Returns the decision state, or None if the line terminated / no decision.
    """
    s = state
    while s.is_chance_node():
        s = _sample_chance(s, rng)
    done = 0
    while done < target_street:
        guard = 0
        while not s.is_chance_node():
            if s.is_terminal():
                return None
            legal = s.legal_actions()
            s = s.child(1 if 1 in legal else legal[0])  # 1 = call/check
            guard += 1
            if guard > 30:
                return None
        while s.is_chance_node():
            s = _sample_chance(s, rng)
        done += 1
    if s.is_terminal() or s.is_chance_node():
        return None
    return s


def _try_sample_leaf(structure, master_rng, target):
    """One attempt to sample a valid leaf at `target` street (0=preflop..3=river).

    Returns a leaf dict (node, starting_stacks, hero, meta) or None if the sample
    is ITM (short-circuits, no BR signal), has no live opponent (BR == baseline
    trivially), or the line can't reach a decision at `target`."""
    sampled = sample_starting_state(structure, master_rng, num_paid=NUM_PAID)
    stacks = list(sampled["stacks"])
    if is_itm(stacks, NUM_PAID):
        return None
    gs = structure.to_inner_game_string_for_state(
        blind_level=sampled["blind_level"], stacks=stacks,
        dealer_seat=sampled["dealer_seat"])
    game = pyspiel.load_game(gs)
    seed = master_rng.randrange(2 ** 31)
    st = _walk_to_street_decision(game.new_initial_state(), random.Random(seed), target)
    if st is None:
        return None
    hero = st.current_player()
    if hero < 0:
        return None
    parsed = parse_state_6max(st)
    money = parsed["money"]
    live = [o for o in range(N_SEATS) if o != hero and o < len(money) and money[o] > 0]
    if not live:
        return None
    node = SubgameNode(kind=NodeKind.LEAF, state=st, depth=4, current_player=hero)
    return {
        "node": node, "starting_stacks": stacks, "hero": hero,
        "meta": {"target_street": target,
                 "street_idx": int(parsed["street_idx"]),
                 "n_live_opp": len(live),
                 "alive_count": int(sampled["alive_count"]),
                 "blind_level": int(sampled["blind_level"].level),
                 "eff_stack": int(min(money[hero], max(money[o] for o in live)))},
    }


def build_battery(structure, master_rng, n, min_late_leaves=10):
    """Sample n valid (non-ITM, >=1 live opp) leaf states spanning streets.

    Pass 1 round-robins target street over {preflop, flop, turn, river} for an even
    natural spread. Pass 2 (Item 2 — street-coverage stratification) then TOPS UP
    late-street leaves (street_idx >= 2: turn/river) to at least `min_late_leaves`
    by replacing early-street leaves with freshly sampled late ones (total stays n).
    Late-game leaves have distinct bias-sensitivity (fewer decisions per opponent,
    tighter action menus) that the gate needs to measure, and shallow turbo stacks
    make them under-sample naturally (many lines end before the turn). Returns the
    leaf list; the caller logs the realized per-street distribution.

    min_late_leaves=0 disables the top-up (used by --smoke, a fast correctness
    check where coverage is not the point). Capped at n internally."""
    cap = n * 60
    leaves, attempts = [], 0
    street_cycle = [0, 1, 2, 3]
    # Pass 1: even round-robin (the target street is "sticky" until a sample lands,
    # so a hard-to-reach late street keeps retrying rather than skewing the spread).
    while len(leaves) < n and attempts < cap:
        attempts += 1
        target = street_cycle[len(leaves) % len(street_cycle)]
        leaf = _try_sample_leaf(structure, master_rng, target)
        if leaf is not None:
            leaves.append(leaf)

    # Pass 2: guarantee late-street coverage without changing n.
    want_late = min(min_late_leaves, len(leaves))
    n_late = lambda ls: sum(1 for e in ls if e["meta"]["street_idx"] >= 2)
    late_cycle, topup = [2, 3], 0
    while n_late(leaves) < want_late and topup < cap:
        topup += 1
        leaf = _try_sample_leaf(structure, master_rng, late_cycle[topup % 2])
        if leaf is None or leaf["meta"]["street_idx"] < 2:
            continue
        early_idx = next((i for i, e in enumerate(leaves)
                          if e["meta"]["street_idx"] < 2), None)
        if early_idx is None:
            break  # everything is already late-street
        leaves[early_idx] = leaf
    return leaves


# ----------------------------------------------------------------------------
# Per-condition sampling (orchestrates the evaluator's own rollout primitives)
# ----------------------------------------------------------------------------

def _make_ctx(solver, biased, starting_stacks, hero, mode):
    return LeafEvalContext(
        blueprint=solver, biased_blueprint=biased,
        starting_stacks=starting_stacks, payouts=list(sng_payouts_6max_double_up()),
        hero_seat=hero, mode=mode, n_samples=8, icm_short_circuit=True, num_paid=NUM_PAID)


def _collect_samples(node, ctx, dist_fn, seeds, per_seed_biases=None):
    """Run one rollout per shared seed (CRN), returning per-sample 6-vectors.

    dist_fn: fixed action_dist_fn (BR profile / baseline). If per_seed_biases is
    given (PROFILE), it maps seed->biases drawn from a SEPARATE rng so the rollout
    rng (Random(seed)) stays aligned across conditions (approx CRN on the deal).

    Cache: reset ONCE at the start (mirrors a single evaluate_leaf call — one reset,
    then the M samples share the deterministic bucket cache). bucket_of is
    order-independent and discards rng, so cache state never changes a value; the
    single reset gives this condition a cold-cache cost measurement comparable to
    a standalone evaluate_leaf, while still amortizing repeated boards within the M
    samples (the regime the per-mode cost split below reports)."""
    try:
        ctx.blueprint.encoder.reset_cache()
    except Exception:
        pass
    out = []
    for s in seeds:
        fn = dist_fn if per_seed_biases is None else _bias_dist_fn(ctx, per_seed_biases(s))
        r = random.Random(s)
        vec = _rollout_once(node.state.clone(), ctx, fn, r)
        out.append(vec)  # may be None (failed sample)
    return out


def _mean_stderr(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0, len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var / len(xs)), len(xs)


def _opponent_bias_values_aligned(node, ctx, o, menu, rng):
    """Index-aligned CRN variant of subgame_leaf._opponent_bias_values (Item 1).

    Returns {bias: [v_0 .. v_{M-1}]} with None for a failed sample, PRESERVING
    per-sample-index alignment across biases so the bias effect can be measured
    CRN-PAIRED (same deal m under each bias). Mirrors the canonical seeding (one
    shared `eval_seeds` list) and per-sample cache reset (standalone CRN) exactly,
    so the per-bias MEANS — and therefore the argmax `br_bias` — match the canonical
    `_opponent_bias_values`. The canonical compacts failures out (losing alignment);
    this keeps the index. subgame_leaf is unchanged — Item 1 is criterion-only.
    """
    eval_seeds = [rng.randrange(2 ** 31) for _ in range(ctx.n_samples)]
    out = {b: [None] * ctx.n_samples for b in menu}
    for b in menu:
        dist_fn = _bias_dist_fn(ctx, {o: b})
        for m in range(ctx.n_samples):
            r = random.Random(eval_seeds[m])
            if not ctx.manage_cache_externally:
                _reset_cache(ctx)
            vec = _rollout_once(node.state.clone(), ctx, dist_fn, r)
            if vec is not None:
                out[b][m] = vec[o]
    return out


def _aligned_mean(vs):
    good = [x for x in vs if x is not None]
    return (sum(good) / len(good)) if good else float("-inf")


def _resolution(vals, means, finite, v0):
    """CRN-paired RESOLUTION test for one (leaf, opp) pair (Item 1).

    Picks the bias whose mean opp-value deviates MOST from blueprint (bias 0) in
    EITHER direction, then tests that deviation against MC noise using the
    CRN-PAIRED per-sample difference (best_bias[m] - bias0[m]). Pairing cancels the
    shared per-deal variance that, per the session-14 finding, otherwise swamps the
    small bias signal — it is the only way the effect clears noise at feasible M.

    ABSOLUTE deviation (not the directional 'best bias beats blueprint') on purpose:
    a bias that makes the opponent detectably WORSE still proves M is adequate to
    SEE bias effects. That is exactly the high-resolution / low-differentiation
    regime (blueprint already opponent-optimal) the revised gate must be able to
    distinguish from a genuine mechanism collapse; a directional definition would
    make differentiation-among-resolved ~tautological.

    Returns (resolved, gap, stderr, bias, n_paired). resolved iff
    |mean(paired)| > 3 * stderr(paired) with >= 2 paired samples (a zero-variance
    non-zero difference resolves; an all-tied pair, gap 0, does not).
    """
    if not math.isfinite(v0):
        return False, float("nan"), float("nan"), 0, 0
    devs = {b: abs(means[b] - v0) for b in finite if b != 0}
    if not devs:
        return False, float("nan"), float("nan"), 0, 0
    res_bias = max(devs, key=lambda b: devs[b])
    paired = [vals[res_bias][m] - vals[0][m] for m in range(len(vals[0]))
              if vals[res_bias][m] is not None and vals[0][m] is not None]
    dmean, dstderr, dn = _mean_stderr(paired)
    if dn < 2:
        return False, float("nan"), float("nan"), res_bias, dn
    return (abs(dmean) > 3.0 * dstderr), abs(dmean), dstderr, res_bias, dn


def evaluate_leaf_three_conditions(entry, solver, biased, samples, seed):
    """Evaluate one leaf under BR / PROFILE(uniform) / baseline(blueprint).

    Returns a per-leaf record. Uses CRN: the same `seeds` drive the rollout rng
    across all three conditions; opponent-bias draws (PROFILE) use a separate rng.
    """
    node, starting_stacks, hero = entry["node"], entry["starting_stacks"], entry["hero"]
    ctx_br = _make_ctx(solver, biased, starting_stacks, hero, LeafEvalMode.BEST_RESPONSE)
    ctx_pr = _make_ctx(solver, biased, starting_stacks, hero, LeafEvalMode.PROFILE_SAMPLE)
    k = biased.k
    parsed = parse_state_6max(node.state)
    money = parsed["money"]
    live = [o for o in range(N_SEATS) if o != hero and o < len(money) and money[o] > 0]

    leaf_rng = random.Random(seed)
    crn_seeds = [leaf_rng.randrange(2 ** 31) for _ in range(samples)]

    # --- mechanism: per-opponent BR selection + CRN-paired RESOLUTION (Item 1) ---
    mech_seed = leaf_rng.randrange(2 ** 31)
    ctx_br.n_samples = samples
    opp_records = []
    br_profile = {}
    for o in live:
        menu = _menu_biases(None, o, k)
        # Index-aligned per-sample values so the bias effect can be tested
        # CRN-paired. Means equal the canonical _opponent_bias_values' (same
        # seeds/reset), so br_bias selection is unchanged.
        vals = _opponent_bias_values_aligned(node, ctx_br, o, menu,
                                              random.Random(mech_seed + o))
        means = {b: _aligned_mean(vals[b]) for b in menu}
        finite = {b: m for b, m in means.items() if math.isfinite(m)}
        if not finite:
            continue
        max_v = max(finite.values())
        mean_v = sum(finite.values()) / len(finite)
        # BR selects the value-MAXIMIZING bias (directional), lowest-index tie-break.
        br_bias = min(b for b in finite if finite[b] >= max_v - _EPS)
        br_profile[o] = br_bias
        # RESOLUTION: most blueprint-deviating bias detectable under CRN pairing?
        v0 = means.get(0, float("nan"))
        resolved, res_gap, res_stderr, res_bias, res_n = _resolution(
            vals, means, finite, v0)
        opp_records.append({
            "opp": o, "menu": menu,
            "bias_means": {int(b): float(m) for b, m in finite.items()},
            "br_bias": int(br_bias), "max_minus_mean": float(max_v - mean_v),
            "max_ge_mean": bool(max_v >= mean_v - _EPS),
            "br_is_nonblueprint": bool(br_bias != 0),
            "resolved": bool(resolved),
            "resolution_bias": int(res_bias),
            "resolution_gap_crn": float(res_gap),
            "resolution_stderr_crn": float(res_stderr),
            "resolution_n_paired": int(res_n),
            "differentiated": bool(resolved and br_bias != 0),
        })

    # --- hero-value conditions under CRN (fixed BR profile = single-pass) ---
    br_fn = _bias_dist_fn(ctx_br, {o: br_profile.get(o, 0) for o in range(N_SEATS) if o != hero})
    base_fn = _bias_dist_fn(ctx_pr, {o: 0 for o in range(N_SEATS) if o != hero})

    def profile_draw(s):
        return _draw_opp_biases(ctx_pr, random.Random((s * 2654435761) & 0x7FFFFFFF))

    t0 = time.perf_counter()
    with Instr(solver) as instr_br:
        br_samp = _collect_samples(node, ctx_br, br_fn, crn_seeds)
    t_br = time.perf_counter() - t0
    t0 = time.perf_counter()
    with Instr(solver) as instr_pr:
        pr_samp = _collect_samples(node, ctx_pr, None, crn_seeds, per_seed_biases=profile_draw)
    t_pr = time.perf_counter() - t0
    t0 = time.perf_counter()
    base_samp = _collect_samples(node, ctx_pr, base_fn, crn_seeds)
    t_base = time.perf_counter() - t0

    def hero_vals(samp):
        return [v[hero] if v is not None else None for v in samp]

    def mean_vec(samp):
        good = [v for v in samp if v is not None]
        if not good:
            return [float("nan")] * N_SEATS
        return [sum(v[i] for v in good) / len(good) for i in range(N_SEATS)]

    br_h, pr_h, base_h = hero_vals(br_samp), hero_vals(pr_samp), hero_vals(base_samp)
    br_hm, br_he, br_n = _mean_stderr(br_h)
    pr_hm, pr_he, pr_n = _mean_stderr(pr_h)
    base_hm, base_he, base_n = _mean_stderr(base_h)

    # CRN-paired hero deltas (positive = BR hurts hero, the expected direction)
    paired_pr = [(pr_h[i] - br_h[i]) for i in range(samples)
                 if br_h[i] is not None and pr_h[i] is not None]
    paired_base = [(base_h[i] - br_h[i]) for i in range(samples)
                   if br_h[i] is not None and base_h[i] is not None]
    d_pr_m, d_pr_e, _ = _mean_stderr(paired_pr)
    d_base_m, d_base_e, _ = _mean_stderr(paired_base)

    degraded = (br_n == 0 or pr_n == 0 or base_n == 0)
    return {
        "meta": entry["meta"], "hero": hero, "n_live_opp": len(live),
        "br_value": mean_vec(br_samp), "profile_value": mean_vec(pr_samp),
        "baseline_value": mean_vec(base_samp),
        "br_hero": br_hm, "br_hero_stderr": br_he,
        "profile_hero": pr_hm, "profile_hero_stderr": pr_he,
        "baseline_hero": base_hm, "baseline_hero_stderr": base_he,
        "hero_delta_profile_minus_br": d_pr_m, "hero_delta_profile_minus_br_stderr": d_pr_e,
        "hero_delta_baseline_minus_br": d_base_m, "hero_delta_baseline_minus_br_stderr": d_base_e,
        "n_completed": {"br": br_n, "profile": pr_n, "baseline": base_n},
        "wallclock_s": {"br": t_br, "profile": t_pr, "baseline": t_base},
        "degraded": degraded,
        "opponents": opp_records,
        "cost": {
            "br": {"net_calls": instr_br.net.calls, "net_t": instr_br.net.t,
                   "bucket_miss": instr_br.bucket.calls, "bucket_t": instr_br.bucket.t,
                   "encode_calls": instr_br.encode.calls},
            "profile": {"net_calls": instr_pr.net.calls, "net_t": instr_pr.net.t,
                        "bucket_miss": instr_pr.bucket.calls, "bucket_t": instr_pr.bucket.t,
                        "encode_calls": instr_pr.encode.calls},
        },
    }


# ----------------------------------------------------------------------------
# Aggregation + verdict
# ----------------------------------------------------------------------------

def _m_projection(opp_pairs, m_current, sigma_floor=1.0, target_sigma=3.0):
    """Project the M needed for the MEDIAN near-resolved pair to clear 3σ (Item 4).

    Paired stderr ~ 1/sqrt(M), so the per-pair significance sigma = gap/stderr ~
    sqrt(M). A pair measured at sigma0 (= resolution_gap_crn / resolution_stderr_crn)
    under m_current samples needs M ≈ m_current * (target_sigma / sigma0)^2 to reach
    target_sigma. We take the MEDIAN sigma0 over NEAR-RESOLVED pairs (sigma0 >=
    sigma_floor) — restricting to pairs with at least sigma_floor of real signal so
    the all-tied pairs (gap ~ 0, sigma ~ 0) don't dominate the median and make M
    look unboundedly large. One number, `m_needed_for_gate`, picks the next
    escalation step (M=16 vs 32 vs ...) without trial-and-error.

    If NO pair reaches sigma_floor, the gap is absent (not merely unresolved) at
    these leaves: escalating M won't help — surfaced via m_needed_for_gate=None.
    """
    sigmas = []
    for o in opp_pairs:
        gap, se, n = (o.get("resolution_gap_crn"), o.get("resolution_stderr_crn"),
                      o.get("resolution_n_paired", 0))
        if n < 2 or not isinstance(gap, float) or not math.isfinite(gap) or gap <= 0:
            continue
        if se == 0:
            sig = float("inf")          # zero-variance, already resolved
        elif math.isfinite(se):
            sig = gap / se
        else:
            continue
        if sig >= sigma_floor:
            sigmas.append(sig)
    if not sigmas:
        return {"m_current": m_current, "candidate_pairs": 0,
                "median_sigma_current": float("nan"),
                "median_stderr_to_gap_ratio": float("nan"),
                "m_needed_for_gate": None,
                "note": ("no (leaf,opp) pair reaches >=1σ of bias signal — the gap is "
                         "absent, not merely unresolved; escalating M is unlikely to "
                         "help. Revisit leaf selection / bias menu rather than M.")}
    sigmas.sort()
    mid = len(sigmas) // 2
    med = sigmas[mid] if len(sigmas) % 2 else 0.5 * (sigmas[mid - 1] + sigmas[mid])
    if not math.isfinite(med):
        return {"m_current": m_current, "candidate_pairs": len(sigmas),
                "median_sigma_current": float("inf"),
                "median_stderr_to_gap_ratio": 0.0,
                "m_needed_for_gate": int(m_current),
                "note": "median near-resolved pair already resolves (zero-variance)."}
    m_needed = math.ceil(m_current * (target_sigma / med) ** 2)
    return {"m_current": m_current, "candidate_pairs": len(sigmas),
            "median_sigma_current": med,
            "median_stderr_to_gap_ratio": 1.0 / med,
            "m_needed_for_gate": int(m_needed),
            "note": (f"median near-resolved pair is at {med:.2f}σ at M={m_current}; "
                     f"~M={m_needed} projected to clear {target_sigma:.0f}σ "
                     f"(stderr ~ 1/sqrt(M)).")}


def aggregate(records, m_current=None):
    n = len(records)
    usable = [r for r in records if not r["degraded"]]
    # opponent-level mechanism
    opp_pairs = [o for r in usable for o in r["opponents"]]
    n_pairs = len(opp_pairs)
    frac_max_ge_mean = (sum(o["max_ge_mean"] for o in opp_pairs) / n_pairs) if n_pairs else float("nan")
    frac_br_nonbp = (sum(o["br_is_nonblueprint"] for o in opp_pairs) / n_pairs) if n_pairs else float("nan")
    gaps = [o["max_minus_mean"] for o in opp_pairs]
    mean_gap = (sum(gaps) / len(gaps)) if gaps else float("nan")
    # --- split metric (Item 1): resolution + differentiation ---
    n_resolved = sum(1 for o in opp_pairs if o.get("resolved"))
    resolution_rate = (n_resolved / n_pairs) if n_pairs else float("nan")
    n_diff = sum(1 for o in opp_pairs if o.get("resolved") and o["br_is_nonblueprint"])
    differentiation_rate = (n_diff / n_resolved) if n_resolved else float("nan")
    res_gaps = [o["resolution_gap_crn"] for o in opp_pairs
                if o.get("resolved") and math.isfinite(o.get("resolution_gap_crn", float("nan")))]
    mean_res_gap = (sum(res_gaps) / len(res_gaps)) if res_gaps else float("nan")
    # hero direction (aggregate across leaves of the per-leaf CRN-paired delta mean)
    d_pr = [r["hero_delta_profile_minus_br"] for r in usable
            if math.isfinite(r["hero_delta_profile_minus_br"])]
    d_base = [r["hero_delta_baseline_minus_br"] for r in usable
              if math.isfinite(r["hero_delta_baseline_minus_br"])]
    dpr_m, dpr_e, _ = _mean_stderr(d_pr)
    dbase_m, dbase_e, _ = _mean_stderr(d_base)
    frac_dir_correct = (sum(1 for x in d_pr if x > 0) / len(d_pr)) if d_pr else float("nan")
    # condition mean hero values
    brh = _mean_stderr([r["br_hero"] for r in usable if math.isfinite(r["br_hero"])])
    prh = _mean_stderr([r["profile_hero"] for r in usable if math.isfinite(r["profile_hero"])])
    bah = _mean_stderr([r["baseline_hero"] for r in usable if math.isfinite(r["baseline_hero"])])
    # mode-dependent cost (sum across leaves)
    def cost_split(key):
        nt = sum(r["cost"][key]["net_t"] for r in usable)
        bt = sum(r["cost"][key]["bucket_t"] for r in usable)
        nc = sum(r["cost"][key]["net_calls"] for r in usable)
        bm = sum(r["cost"][key]["bucket_miss"] for r in usable)
        tot = nt + bt
        return {"net_calls": nc, "bucket_miss": bm, "net_t": nt, "bucket_t": bt,
                "net_frac_of_net_plus_bucket": (nt / tot if tot else float("nan")),
                "bucket_frac_of_net_plus_bucket": (bt / tot if tot else float("nan"))}
    street_dist = {str(s): sum(1 for r in records if r["meta"]["street_idx"] == s)
                   for s in range(4)}
    return {
        "n_leaves": n, "n_usable": len(usable), "n_degraded": n - len(usable),
        "street_distribution": street_dist,
        "n_late_leaves_ge2": sum(v for s, v in street_dist.items() if int(s) >= 2),
        "n_opp_pairs": n_pairs,
        "n_resolved_pairs": n_resolved,
        "resolution_rate": resolution_rate,
        "differentiation_rate": differentiation_rate,
        "mean_resolved_gap_crn": mean_res_gap,
        "m_projection": _m_projection(opp_pairs, m_current),
        "frac_max_ge_mean": frac_max_ge_mean,
        "frac_br_selects_nonblueprint": frac_br_nonbp,
        "mean_opp_gap_max_minus_mean": mean_gap,
        "hero_delta_profile_minus_br": {"mean": dpr_m, "stderr": dpr_e,
                                        "sigma": (dpr_m / dpr_e if dpr_e else float("nan")),
                                        "frac_leaves_correct_sign": frac_dir_correct},
        "hero_delta_baseline_minus_br": {"mean": dbase_m, "stderr": dbase_e,
                                         "sigma": (dbase_m / dbase_e if dbase_e else float("nan"))},
        "mean_hero_value": {"br": brh[0], "profile": prh[0], "baseline": bah[0]},
        "cost_by_mode": {"br": cost_split("br"), "profile": cost_split("profile")},
    }


def verdict(agg):
    """Split-metric verdict (Item 1 revision). status in {PASS, FAIL}.

    Replaces the single frac_br_selects_nonblueprint>=0.20 gate, which conflated
    three phenomena — sampling resolution, blueprint convergence quality, and the
    BR mechanism. The revised gate decomposes them:

      resolution_rate     = fraction of (leaf,opp) pairs where the most blueprint-
                            DEVIATING bias differs from blueprint by >3x its
                            CRN-paired stderr (is M adequate to SEE bias effects?).
      differentiation_rate = among RESOLVED pairs only, fraction where BR's argmax
                            selects a non-blueprint bias (does BR pick biases that
                            BEAT the blueprint, or is the blueprint already
                            opponent-self-optimal here?).

    Gate: PASS iff resolution_rate >= 0.50 AND
                   (differentiation_rate >= 0.30 OR resolution_rate >= 0.90).
    The second clause passes the high-resolution / low-differentiation case: if M
    is clearly adequate (resolution >= 90%) yet BR rarely beats blueprint, that is
    evidence the blueprint is well-converged, not that the mechanism is broken.
    frac_br_selects_nonblueprint is still reported (legacy) but no longer gates.
    """
    rr = agg["resolution_rate"]
    dr = agg["differentiation_rate"]
    fb = agg["frac_br_selects_nonblueprint"]

    # Sanity guard (near-tautological bug detector, retained from the prior gate).
    if math.isfinite(agg["frac_max_ge_mean"]) and agg["frac_max_ge_mean"] < 0.99:
        return "FAIL", [
            f"SANITY VIOLATION: max_b V_i(b) >= mean_b holds on only "
            f"{agg['frac_max_ge_mean']:.1%} of pairs (expect ~100%) — an argmax/"
            f"aggregation bug, NOT an architecture finding. Fix before interpreting."]

    passed = (math.isfinite(rr) and rr >= 0.50
              and ((math.isfinite(dr) and dr >= 0.30) or rr >= 0.90))

    if not math.isfinite(rr) or rr < 0.50:
        mp = agg.get("m_projection", {})
        m_needed = mp.get("m_needed_for_gate")
        proj = (f" Projected M for the median near-resolved pair to clear 3σ: "
                f"~{m_needed} (median {mp.get('median_sigma_current', float('nan')):.2f}σ "
                f"at M={mp.get('m_current')})." if m_needed is not None
                else f" {mp.get('note', '')}")
        scenario = ("LOW RESOLUTION: M is too small to detect bias effects at these "
                    "leaves (most pairs tie within MC noise). A measurement-budget "
                    "limit, NOT an architecture failure — re-run at higher M (and/or "
                    "more leaves) before any architectural conclusion." + proj)
    elif math.isfinite(dr) and dr >= 0.30:
        scenario = ("HIGH RESOLUTION + HIGH DIFFERENTIATION: bias effects are "
                    "detectable AND BR frequently picks a bias that beats the "
                    "blueprint — the mechanism does real work; the blueprint is not "
                    "already opponent-optimal at these leaves.")
    elif rr >= 0.90:
        scenario = ("HIGH RESOLUTION + LOW DIFFERENTIATION: bias effects are clearly "
                    "detectable but BR rarely beats the blueprint — evidence the "
                    "blueprint is well-converged (near opponent-self-optimal here). "
                    "The mechanism is sound; it would add more vs weaker blueprints. "
                    "Passes on resolution >= 90%.")
    else:
        scenario = ("AMBIGUOUS (moderate resolution, low differentiation): neither "
                    "clearly measurement-limited nor clearly blueprint-optimal — "
                    "re-run at higher M or N to disambiguate.")

    lines = [
        f"resolution_rate = {rr:.1%}  [>3σ CRN-paired bias-vs-blueprint on "
        f"{agg['n_resolved_pairs']}/{agg['n_opp_pairs']} pairs; high => M adequate "
        f"to detect bias effects].",
        f"differentiation_rate = {dr:.1%}  [among resolved pairs, BR picks a "
        f"non-blueprint bias; high => BR selects biases that beat the blueprint].",
        f"frac_br_selects_nonblueprint = {fb:.1%}  [legacy all-pairs metric; "
        f"reported for continuity, no longer the gate driver].",
        scenario,
    ]
    return ("PASS" if passed else "FAIL"), lines


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def _file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--samples", type=int, default=8, help="M per condition (default 8)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--smoke", action="store_true", help="tiny run: N=3, M=4")
    ap.add_argument("--min-late-leaves", type=int, default=10,
                    help="min leaves with street_idx>=2 (turn/river); 0 disables (Item 2)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.n, args.samples, args.min_late_leaves = 3, 4, 0

    abstr = Abstraction.load(ABSTR)
    structure = TournamentStructure.from_yaml(STRUCT)
    solver = _load_solver(CKPT, abstr, structure)
    biased = BiasedBlueprint()
    stack = int(solver.encoder.starting_stack)
    print(f"Stage F leaf-eval ablation | blueprint={CKPT}")
    print(f"N={args.n} leaves, M={args.samples} samples/condition, seed={args.seed}, stack={stack}")

    master_rng = random.Random(args.seed)
    t_build = time.perf_counter()
    battery = build_battery(structure, master_rng, args.n,
                            min_late_leaves=args.min_late_leaves)
    street_hist = {s: sum(1 for e in battery if e["meta"]["street_idx"] == s)
                   for s in range(4)}
    n_late = sum(v for s, v in street_hist.items() if s >= 2)
    print(f"Built {len(battery)} leaf states in {time.perf_counter()-t_build:.1f}s "
          f"(per-street {street_hist}; late[>=2]={n_late}, min={args.min_late_leaves})")
    if args.min_late_leaves and n_late < args.min_late_leaves:
        print(f"  WARNING: only {n_late} late-street leaves (< {args.min_late_leaves} "
              f"requested) — late-game bias-sensitivity will be under-measured.")

    records = []
    t0 = time.perf_counter()
    for i, entry in enumerate(battery):
        rec = evaluate_leaf_three_conditions(entry, solver, biased, args.samples,
                                             seed=master_rng.randrange(2 ** 31))
        records.append(rec)
        if (i + 1) % 10 == 0 or i + 1 == len(battery):
            print(f"  evaluated {i+1}/{len(battery)} leaves "
                  f"[{time.perf_counter()-t0:.1f}s]")
    wall = time.perf_counter() - t0

    agg = aggregate(records, m_current=args.samples)
    status, vlines = verdict(agg)

    # ---- human-readable summary ----
    print("\n" + "=" * 70)
    print("STAGE F — Q11 LEVEL 1 LEAF-EVAL ABLATION — SUMMARY")
    print("=" * 70)
    print(f"leaves: {agg['n_leaves']} ({agg['n_usable']} usable, {agg['n_degraded']} degraded); "
          f"{agg['n_opp_pairs']} (leaf,opp) pairs")
    print(f"\nSPLIT-METRIC GATE (Item 1 — the verdict drivers):")
    print(f"  resolution rate:        {agg['resolution_rate']:.1%}  "
          f"({agg['n_resolved_pairs']}/{agg['n_opp_pairs']} pairs, >3σ CRN-paired)")
    print(f"  differentiation rate:   {agg['differentiation_rate']:.1%}  "
          f"(of resolved pairs; BR picks non-blueprint)")
    print(f"  mean resolved gap (CRN):{agg['mean_resolved_gap_crn']:+.4f}")
    mp = agg["m_projection"]
    if mp["m_needed_for_gate"] is not None:
        print(f"  M projection:           median near-resolved pair "
              f"{mp['median_sigma_current']:.2f}σ at M={mp['m_current']} "
              f"({mp['candidate_pairs']} candidates) => ~M={mp['m_needed_for_gate']} for 3σ")
    else:
        print(f"  M projection:           {mp['note']}")
    print(f"\nLEGACY / SANITY SIGNALS (reported, not gating):")
    print(f"  frac BR non-blueprint:  {agg['frac_br_selects_nonblueprint']:.1%} of all pairs")
    print(f"  max_b V_i(b) >= mean_b: {agg['frac_max_ge_mean']:.1%} of pairs (expect ~100%; sanity)")
    print(f"  mean opp gap (max-mean):{agg['mean_opp_gap_max_minus_mean']:+.4f}")
    print(f"\nHERO-VALUE DIRECTION (noisy per session-14; aggregate sign is the signal):")
    hd = agg["hero_delta_profile_minus_br"]
    print(f"  aggregate (PROFILE - BR) hero:    {hd['mean']:+.4f} +/- {hd['stderr']:.4f} "
          f"({hd['sigma']:+.1f}σ)   [+ = BR hurts hero, expected]")
    print(f"  per-leaf correct-sign fraction:   {hd['frac_leaves_correct_sign']:.1%}")
    hb = agg["hero_delta_baseline_minus_br"]
    print(f"  aggregate (BASELINE - BR) hero:   {hb['mean']:+.4f} +/- {hb['stderr']:.4f} "
          f"({hb['sigma']:+.1f}σ)")
    mh = agg["mean_hero_value"]
    print(f"  mean hero value:  BR={mh['br']:+.4f}  PROFILE={mh['profile']:+.4f}  "
          f"BASELINE={mh['baseline']:+.4f}   (expect BR <= PROFILE <= BASELINE)")
    print(f"\nMODE-DEPENDENT COST (verify Q13 split holds at this scale):")
    for mode in ("br", "profile"):
        c = agg["cost_by_mode"][mode]
        print(f"  {mode.upper():<8} net={c['net_calls']:>7} calls / bucket-miss={c['bucket_miss']:>5}"
              f"  | net%={c['net_frac_of_net_plus_bucket']:.0%} "
              f"bucket%={c['bucket_frac_of_net_plus_bucket']:.0%} (of net+bucket)")
    print(f"\nwall-clock: {wall:.1f}s for {len(battery)} leaves x 3 conditions x M={args.samples}")
    print("\n" + "-" * 70)
    print(f"VERDICT: {status}")
    for ln in vlines:
        print(f"  - {ln}")
    if status == "PASS":
        print("  => Architecture mechanism is firing correctly. Stage G is the natural next stage.")
    else:
        print("  => STOP and surface to the human (Path A discipline). Do NOT proceed to Stage G.")
    print("-" * 70)

    # ---- JSON ----
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.output) if args.output else Path(
        f"evals/leaf_eval_ablation_session17_{ts}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "F", "q11_level": 1, "session": 17,
        "verdict": status, "verdict_notes": vlines,
        "repro": {
            "git_rev": _git_rev(), "blueprint_ckpt": CKPT,
            "blueprint_md5": _file_md5(CKPT), "abstraction": ABSTR,
            "structure": STRUCT, "seed": args.seed, "n_leaves": args.n,
            "samples_per_condition": args.samples, "num_paid": NUM_PAID,
            "min_late_leaves": args.min_late_leaves,
            "starting_stack": stack, "python": sys.version.split()[0],
            "numpy": np.__version__, "timestamp_utc": ts,
        },
        "aggregate": agg,
        "per_leaf": records,
    }
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nWrote {out}")
    return status


if __name__ == "__main__":
    main()
