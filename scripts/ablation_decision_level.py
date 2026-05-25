"""Stage G — Q11 Level 2 decision-level ablation (Track B1c, sub-step 2 closure gate).

Does BEST_RESPONSE leaf evaluation, fed through a one-iteration root regret update,
produce a meaningfully DIFFERENT hero ROOT POLICY than PROFILE_SAMPLE? Stage F (Q11
Level 1) confirmed the leaf mechanism via aggregate signal; Stage G confirms the
mechanism moves hero's *decision*. Full design: docs/STAGE_G_DESIGN.md.

THE STUB SOLVER (design "one-iteration root regret update at the root infoset only"):
  Build a depth-1 subgame tree at hero's decision (subgame.build_subgame_tree,
  max_action_depth=1) → each child is a LEAF/TERMINAL (no traversal/backup needed).
  For each legal action a, q_M[a] = hero's value of the child under leaf-eval mode M.
  Blueprint root policy  σ0 = RM+(adv).
  ev_M  = Σ_a σ0[a]·q_M[a]
  r_M[a]= (q_M[a] − ev_M)·legal_mask[a]        (cfr6.py:378 regret, instantaneous)
  σ_M   = RM+(adv + r_M)                         (solver._strategy_from_advantages)
  Properties: r_M=0 (q flat) ⇒ σ_M=σ0 (reduces to blueprint); q_BR≡q_PROFILE
  (bias-inactive root) ⇒ σ_BR≡σ_PROFILE (cross-mode identity). adv and r_M are
  scale-matched: the advantage net regresses instantaneous O(1) ICM-unit regrets.

SAMPLING (G-A): stratified-by-street with late-street oversampling, behind a
"real-decision" inclusion filter — hero has >=3 discrete legal actions (mixing
latitude, not fold/shove), >=1 live opponent, non-ITM root. Part-2 finding: ~85%+ of
roots are bias-active (NOT the 55%-dead-pairs wall Stage F hit at the per-pair
level), but effect magnitude concentrates late-street, so we oversample turn/river.

GATE (G-C, split-metric mirroring Stage F; PRE-COMMITTED, applied mechanically):
  resolution_rate     = frac roots where the most mode-deviating action's CRN-paired
                        q_PROFILE[a]-q_BR[a] clears 3σ (is the BR-vs-PROFILE effect
                        detectable at this root?).
  differentiation_rate= among resolved roots, frac where the BR-vs-PROFILE policy
                        shift is MATERIAL (L1(σ_BR,σ_PROFILE) >= L1_FLOOR). NOTE:
                        direction-AGNOSTIC by design (Addition 2) — it is NOT the
                        "BR flatter" test, so the load-bearing gate never hinges on
                        the over-specific flatness prediction. (It is also NOT
                        KL-to-blueprint: the depth-1 stub's r frequently dominates
                        adv so σ_M sits far from σ0 — see kl_br_blueprint, a
                        stub-crudeness diagnostic reported but not gated.)
  value_suppression   = aggregate CRN-paired (q_PROFILE-q_BR) over roots (BR opponents
                        hurt hero — continuity with Stage F's +3.4σ). LOAD-BEARING.
  policy_divergence   = aggregate mean L1(σ_BR,σ_PROFILE) vs a per-deal permutation
                        null (mode-label exchange). LOAD-BEARING — the core Stage-G
                        claim that BR MOVES the policy.
  entropy_delta       = aggregate H(σ_BR)−H(σ_PROFILE) (predicted >0: BR flatter).
                        SOFT / reported only (Addition 2): the flatness prediction is
                        a Pluribus-style theoretical expectation that may not hold
                        uniformly; a non-significant or reversed entropy direction is
                        surfaced for interpretation, NOT a FAIL, provided the
                        load-bearing measures pass.
  n_distinct_shifted  = distinct most-shifted actions across resolved roots (W2
                        non-degeneracy guard).
  filter_rate (Addition 1) = frac of otherwise-valid sampled roots excluded SOLELY
                        for having <3 legal actions (degenerate fold/shove). >30% is
                        itself surfaced as a finding about the game.

  PASS (strict): resolution_rate>=0.50 AND (differentiation_rate>=0.60 OR
                 resolution_rate>=0.90) AND value_suppression>=3σ.
  SUBSTANTIVE_PASS_AGGREGATE: resolution_rate<0.50 AND value_suppression>=3σ AND
                 policy_divergence significant (>99.7pct of null) AND
                 differentiation_rate>=0.55 AND n_distinct_shifted>=2. (entropy is
                 reported, never gates.)
  FAIL: otherwise → STOP and surface to the human (Path A). Do NOT proceed to
        sub-step 3 without consultation.

Usage:
    python -m scripts.ablation_decision_level                 # N=50, M=16 (gate)
    python -m scripts.ablation_decision_level --smoke         # N=5, M=8 (correctness)
    python -m scripts.ablation_decision_level --n 50 --samples 16 --seed 18
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyspiel

from src.nlhe.abstraction import Abstraction
from src.nlhe.actions import DiscreteAction
from src.nlhe.biased_policy import BiasedBlueprint
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.icm import is_itm, sng_payouts_6max_double_up
from src.nlhe.icm_returns import icm_adjust_returns
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.solver import _strategy_from_advantages
from src.nlhe.stack_sampler import sample_starting_state
from src.nlhe.subgame import build_subgame_tree, NodeKind, _discretize_at_decision
from src.nlhe.subgame_leaf import (
    LeafEvalMode, _best_response_biases, _bias_dist_fn, _draw_opp_biases,
)
from scripts.ablation_leaf_eval import (
    ABSTR, CKPT, STRUCT, NUM_PAID, N_SEATS,
    _make_ctx, _collect_samples, _mean_stderr, _walk_to_street_decision,
)
from scripts.eval_6max_self_play import _load_solver

_N_ACTIONS = len(DiscreteAction)  # 7
_EPS = 1e-9
L1_FLOOR = 0.05           # material BR-vs-PROFILE policy shift (differentiation)
MIN_ACTIONS = 3           # real-decision filter (G-A #1)
PERM_B = 400              # permutation resamples for the policy-divergence null


# ----------------------------------------------------------------------------
# The stub solver (pure function — unit-tested directly)
# ----------------------------------------------------------------------------

def rm_plus_masked(vec: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """RM+ over the legal mask — delegates to the production helper (solver.py:143)."""
    return _strategy_from_advantages(np.asarray(vec, dtype=np.float32),
                                     np.asarray(legal_mask, dtype=np.float32))


def stub_root_policy(adv: np.ndarray, q: np.ndarray, legal_mask: np.ndarray):
    """One-iteration root regret update. Returns (sigma_M, sigma0).

    adv : 7-dim advantage-net output at the root (blueprint regrets).
    q   : 7-dim hero action values under a leaf-eval mode (illegal entries ignored).
    σ0 = RM+(adv); ev = Σ σ0·q over legal; r = (q-ev)·mask; σ_M = RM+(adv + r).
    """
    adv = np.asarray(adv, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = np.asarray(legal_mask, dtype=np.float64)
    sigma0 = rm_plus_masked(adv, mask)
    ev = float((sigma0 * q * mask).sum())
    r = (q - ev) * mask
    sigma_m = rm_plus_masked(adv + r, mask)
    return sigma_m, sigma0


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p‖q) in nats, with q smoothed to avoid inf where p>0, q=0."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    qs = q + 1e-9
    qs = qs / qs.sum()
    out = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            out += p[i] * math.log(p[i] / qs[i])
    return float(out)


# ----------------------------------------------------------------------------
# Root battery (sampled from the production game) + filter-rate tracking
# ----------------------------------------------------------------------------

def _try_sample_root(structure, master_rng, target):
    """One attempt to sample a valid ROOT decision at `target` street.

    Returns (root_dict | None, reason). reason in {None,'itm','no_decision',
    'no_live','few_actions'} so the caller can tally the >=3-action filter rate
    (Addition 1) over roots that pass the OTHER filters."""
    sampled = sample_starting_state(structure, master_rng, num_paid=NUM_PAID)
    stacks = list(sampled["stacks"])
    if is_itm(stacks, NUM_PAID):
        return None, "itm"
    gs = structure.to_inner_game_string_for_state(
        blind_level=sampled["blind_level"], stacks=stacks,
        dealer_seat=sampled["dealer_seat"])
    game = pyspiel.load_game(gs)
    seed = master_rng.randrange(2 ** 31)
    st = _walk_to_street_decision(game.new_initial_state(), random.Random(seed), target)
    if st is None:
        return None, "no_decision"
    hero = st.current_player()
    if hero < 0:
        return None, "no_decision"
    parsed = parse_state_6max(st)
    money = parsed["money"]
    live = [o for o in range(N_SEATS) if o != hero and o < len(money) and money[o] > 0]
    if not live:
        return None, "no_live"
    discrete_to_chip = _discretize_at_decision(st)
    n_actions = len(discrete_to_chip)
    if n_actions < MIN_ACTIONS:        # the real-decision filter (G-A #1 / Addition 1)
        return None, "few_actions"
    return ({
        "state": st, "starting_stacks": stacks, "hero": hero,
        "n_actions": n_actions,
        "meta": {"target_street": target, "street_idx": int(parsed["street_idx"]),
                 "n_live_opp": len(live), "alive_count": int(sampled["alive_count"]),
                 "blind_level": int(sampled["blind_level"].level), "n_actions": n_actions,
                 "eff_stack": int(min(money[hero], max(money[o] for o in live)))},
    }, None)


def build_root_battery(structure, master_rng, n, min_late_roots=20):
    """Sample n valid roots (>=3 actions, >=1 live opp, non-ITM) spanning streets.

    Pass 1 round-robins street; pass 2 tops up turn/river to min_late_roots. Tracks
    the >=3-action filter rate over roots that pass the other filters (Addition 1).
    Returns (roots, filter_stats)."""
    cap = n * 120
    roots, attempts = [], 0
    kept_ge3 = excluded_few = 0
    street_cycle = [0, 1, 2, 3]

    def _count(reason):
        nonlocal kept_ge3, excluded_few
        if reason is None:
            kept_ge3 += 1
        elif reason == "few_actions":
            excluded_few += 1

    while len(roots) < n and attempts < cap:
        attempts += 1
        target = street_cycle[len(roots) % len(street_cycle)]
        root, reason = _try_sample_root(structure, master_rng, target)
        _count(reason)
        if root is not None:
            roots.append(root)

    want_late = min(min_late_roots, len(roots))
    n_late = lambda rs: sum(1 for e in rs if e["meta"]["street_idx"] >= 2)
    topup = 0
    while n_late(roots) < want_late and topup < cap:
        topup += 1
        root, reason = _try_sample_root(structure, master_rng, [2, 3][topup % 2])
        _count(reason)
        if root is None or root["meta"]["street_idx"] < 2:
            continue
        early = next((i for i, e in enumerate(roots) if e["meta"]["street_idx"] < 2), None)
        if early is None:
            break
        roots[early] = root

    total_valid = kept_ge3 + excluded_few
    filter_stats = {
        "roots_passing_other_filters": total_valid,
        "excluded_few_actions": excluded_few,
        "kept_ge3_actions": kept_ge3,
        "filter_rate": (excluded_few / total_valid) if total_valid else float("nan"),
    }
    return roots, filter_stats


# ----------------------------------------------------------------------------
# Per-root evaluation
# ----------------------------------------------------------------------------

def _hero_persamples_for_child(child_state, solver, biased, starting_stacks, hero,
                               crn_seeds, mech_seed, samples):
    """Per-sample hero values for one action-child under BR / PROFILE / baseline.

    Reuses Stage F's CRN rollout primitives (no ITM short-circuit — _collect_samples
    rolls every leaf out, matching Stage F's hero-value path). Returns three lists of
    length `samples` (None for a failed rollout)."""
    from src.nlhe.subgame import SubgameNode  # local: avoid top-level torch pull order
    node = SubgameNode(kind=NodeKind.LEAF, state=child_state, depth=1, current_player=None)
    ctx_br = _make_ctx(solver, biased, starting_stacks, hero, LeafEvalMode.BEST_RESPONSE)
    ctx_pr = _make_ctx(solver, biased, starting_stacks, hero, LeafEvalMode.PROFILE_SAMPLE)
    ctx_br.n_samples = samples
    ctx_pr.n_samples = samples

    br_profile, _hit = _best_response_biases(node, ctx_br, random.Random(mech_seed))
    br_fn = _bias_dist_fn(ctx_br, {o: br_profile.get(o, 0)
                                   for o in range(N_SEATS) if o != hero})
    base_fn = _bias_dist_fn(ctx_pr, {o: 0 for o in range(N_SEATS) if o != hero})

    def profile_draw(s):
        return _draw_opp_biases(ctx_pr, random.Random((s * 2654435761) & 0x7FFFFFFF))

    br = _collect_samples(node, ctx_br, br_fn, crn_seeds)
    pr = _collect_samples(node, ctx_pr, None, crn_seeds, per_seed_biases=profile_draw)
    base = _collect_samples(node, ctx_pr, base_fn, crn_seeds)
    h = lambda samp: [v[hero] if v is not None else None for v in samp]
    return h(br), h(pr), h(base)


def evaluate_root(entry, solver, biased, samples, seed):
    """Stub-solve one root under BR / PROFILE / baseline; return a per-root record."""
    state, starting_stacks, hero = entry["state"], entry["starting_stacks"], entry["hero"]
    root_rng = random.Random(seed)
    crn_seeds = [root_rng.randrange(2 ** 31) for _ in range(samples)]

    # Blueprint advantages at the root (deterministic given features).
    parsed = parse_state_6max(state)
    try:
        solver.encoder.reset_cache()
    except Exception:
        pass
    feat = np.asarray(solver.encoder.encode_from_parsed(parsed, rng=random.Random(seed)),
                      dtype=np.float32)
    adv = np.asarray(solver.policy_nets.predict_advantages(hero, feat), dtype=np.float32)

    # Depth-1 subgame: one child per legal discrete action (all LEAF/TERMINAL).
    tree = build_subgame_tree(state, max_action_depth=1, rng=random.Random(seed + 1))
    legal_mask = np.zeros(_N_ACTIONS, dtype=np.float64)
    payouts = list(sng_payouts_6max_double_up())

    # per-action per-sample hero values (length `samples`, None for failed)
    qs = {"br": {}, "profile": {}, "baseline": {}}
    degraded = False
    for child in tree.root.children:
        a = int(child.action_from_parent)
        legal_mask[a] = 1.0
        if child.kind == NodeKind.TERMINAL:
            icm = icm_adjust_returns(list(child.terminal_returns),
                                     list(starting_stacks), payouts)
            val = float(icm[hero]) if all(math.isfinite(x) for x in icm) else None
            if val is None:
                degraded = True
            row = [val] * samples
            qs["br"][a], qs["profile"][a], qs["baseline"][a] = row, list(row), list(row)
        else:
            mech_seed = root_rng.randrange(2 ** 31)
            br, pr, base = _hero_persamples_for_child(
                child.state, solver, biased, starting_stacks, hero,
                crn_seeds, mech_seed, samples)
            qs["br"][a], qs["profile"][a], qs["baseline"][a] = br, pr, base
            if not [x for x in br if x is not None] or not [x for x in pr if x is not None]:
                degraded = True

    legal = [a for a in range(_N_ACTIONS) if legal_mask[a] > 0]

    def qbar(mode):
        out = np.zeros(_N_ACTIONS, dtype=np.float64)
        for a in legal:
            xs = [x for x in qs[mode][a] if x is not None]
            out[a] = (sum(xs) / len(xs)) if xs else 0.0
        return out

    q_br, q_pr, q_base = qbar("br"), qbar("profile"), qbar("baseline")
    sig_br, sig0 = stub_root_policy(adv, q_br, legal_mask)
    sig_pr, _ = stub_root_policy(adv, q_pr, legal_mask)
    sig_base, _ = stub_root_policy(adv, q_base, legal_mask)

    # --- per-root resolution: most mode-deviating action, CRN-paired 3σ ---
    best_a, best_absmean, best_se, best_n = None, -1.0, float("nan"), 0
    for a in legal:
        paired = [qs["profile"][a][m] - qs["br"][a][m] for m in range(samples)
                  if qs["profile"][a][m] is not None and qs["br"][a][m] is not None]
        dm, de, dn = _mean_stderr(paired)
        if dn >= 2 and abs(dm) > best_absmean:
            best_a, best_absmean, best_se, best_n = a, abs(dm), de, dn
    resolved = bool(best_a is not None and best_n >= 2 and best_absmean > 3.0 * best_se)

    # --- per-(root,action) value-suppression: CRN-paired (PROFILE - BR) ---
    sup_per_action = []
    for a in legal:
        paired = [qs["profile"][a][m] - qs["br"][a][m] for m in range(samples)
                  if qs["profile"][a][m] is not None and qs["br"][a][m] is not None]
        if paired:
            sup_per_action.append(sum(paired) / len(paired))
    value_suppression_root = (sum(sup_per_action) / len(sup_per_action)
                              if sup_per_action else float("nan"))

    l1 = float(np.abs(sig_br - sig_pr).sum())
    shifted_action = int(np.argmax(np.abs(sig_br - sig_pr))) if legal else -1
    h_br, h_pr = _entropy(sig_br), _entropy(sig_pr)
    kl_br_bp = _kl(sig_br, sig0)

    return {
        "meta": entry["meta"], "hero": hero, "legal_actions": legal,
        "degraded": degraded,
        "sigma0": sig0.tolist(), "sigma_br": sig_br.tolist(),
        "sigma_profile": sig_pr.tolist(), "sigma_baseline": sig_base.tolist(),
        "q_br": q_br.tolist(), "q_profile": q_pr.tolist(), "q_baseline": q_base.tolist(),
        "l1_br_profile": l1, "shifted_action": shifted_action,
        "entropy_br": h_br, "entropy_profile": h_pr, "entropy_delta": h_br - h_pr,
        "kl_br_blueprint": kl_br_bp,
        "resolved": resolved, "resolution_action": best_a,
        "resolution_absmean_crn": best_absmean if best_a is not None else float("nan"),
        "resolution_stderr_crn": best_se, "resolution_n_paired": best_n,
        # differentiation: among resolved roots, does the detected value-difference
        # move the policy MATERIALLY? Measured on the BR-vs-PROFILE shift (L1), NOT
        # on KL-to-blueprint (which is ~tautologically large — the depth-1 stub's r
        # frequently dominates adv, so σ_M departs far from σ0; see kl_br_blueprint,
        # reported as a stub-crudeness diagnostic, not a gate).
        "differentiated": bool(resolved and l1 >= L1_FLOOR),
        "value_suppression_root": value_suppression_root,
        # raw per-sample hero values retained for the permutation null
        "_persamp": {m: {a: [qs[m][a][i] for i in range(samples)] for a in legal}
                     for m in ("br", "profile")},
        "_adv": adv.tolist(), "_mask": legal_mask.tolist(),
    }


# ----------------------------------------------------------------------------
# Aggregation + verdict
# ----------------------------------------------------------------------------

def _permutation_null_l1(records, seed=12345):
    """Per-deal mode-label exchange null for aggregate mean L1 (policy_divergence).

    Under H0 (opponent mode doesn't matter) BR/PROFILE per-deal hero values are
    exchangeable; permuting the label per deal (consistently across actions) and
    recomputing both policies gives the null distribution of the aggregate mean L1.
    Returns (observed_mean_l1, null_p997, frac_null_ge_obs, B)."""
    rng = np.random.default_rng(seed)
    usable = [r for r in records if not r["degraded"]]
    obs = float(np.mean([r["l1_br_profile"] for r in usable])) if usable else float("nan")
    null_means = []
    for _ in range(PERM_B):
        l1s = []
        for r in usable:
            legal = r["legal_actions"]
            adv = np.asarray(r["_adv"]); mask = np.asarray(r["_mask"])
            ps = r["_persamp"]
            M = len(next(iter(ps["br"].values())))
            flip = rng.random(M) < 0.5
            qb = np.zeros(_N_ACTIONS); qp = np.zeros(_N_ACTIONS)
            for a in legal:
                br = ps["br"][a]; pr = ps["profile"][a]
                bvals, pvals = [], []
                for m in range(M):
                    x, y = (pr[m], br[m]) if flip[m] else (br[m], pr[m])
                    if x is not None:
                        bvals.append(x)
                    if y is not None:
                        pvals.append(y)
                qb[a] = (sum(bvals) / len(bvals)) if bvals else 0.0
                qp[a] = (sum(pvals) / len(pvals)) if pvals else 0.0
            sb, _ = stub_root_policy(adv, qb, mask)
            sp, _ = stub_root_policy(adv, qp, mask)
            l1s.append(float(np.abs(sb - sp).sum()))
        if l1s:
            null_means.append(float(np.mean(l1s)))
    if not null_means:
        return obs, float("nan"), float("nan"), 0
    null_means.sort()
    p997 = null_means[min(len(null_means) - 1, int(0.997 * len(null_means)))]
    frac_ge = sum(1 for x in null_means if x >= obs) / len(null_means)
    return obs, p997, frac_ge, len(null_means)


def aggregate(records):
    n = len(records)
    usable = [r for r in records if not r["degraded"]]
    nu = len(usable)
    n_resolved = sum(1 for r in usable if r["resolved"])
    resolution_rate = (n_resolved / nu) if nu else float("nan")
    n_diff = sum(1 for r in usable if r["differentiated"])
    differentiation_rate = (n_diff / n_resolved) if n_resolved else float("nan")
    # value suppression (per-root mean over actions, then across roots)
    sup = [r["value_suppression_root"] for r in usable
           if math.isfinite(r["value_suppression_root"])]
    sup_m, sup_e, _ = _mean_stderr(sup)
    # entropy delta (SOFT)
    ed = [r["entropy_delta"] for r in usable if math.isfinite(r["entropy_delta"])]
    ed_m, ed_e, _ = _mean_stderr(ed)
    frac_resolved_br_flatter = (
        sum(1 for r in usable if r["resolved"] and r["entropy_delta"] > 0) / n_resolved
        if n_resolved else float("nan"))
    # policy divergence
    obs_l1, null_p997, frac_null_ge, perm_b = _permutation_null_l1(records)
    policy_divergence_significant = bool(
        math.isfinite(obs_l1) and math.isfinite(null_p997) and obs_l1 > null_p997)
    # non-degeneracy
    from collections import Counter
    shifted_hist = Counter(r["shifted_action"] for r in usable if r["resolved"])
    n_distinct_shifted = len(shifted_hist)
    # mean policies / entropies for the report
    mean_h_br = float(np.mean([r["entropy_br"] for r in usable])) if usable else float("nan")
    mean_h_pr = float(np.mean([r["entropy_profile"] for r in usable])) if usable else float("nan")
    mean_kl = float(np.mean([r["kl_br_blueprint"] for r in usable])) if usable else float("nan")
    street_dist = {str(s): sum(1 for r in records if r["meta"]["street_idx"] == s)
                   for s in range(4)}
    return {
        "n_roots": n, "n_usable": nu, "n_degraded": n - nu,
        "street_distribution": street_dist,
        "n_late_roots_ge2": sum(v for s, v in street_dist.items() if int(s) >= 2),
        "n_resolved": n_resolved, "resolution_rate": resolution_rate,
        "differentiation_rate": differentiation_rate,
        "value_suppression": {"mean": sup_m, "stderr": sup_e,
                              "sigma": (sup_m / sup_e if sup_e else float("nan"))},
        "entropy_delta": {"mean": ed_m, "stderr": ed_e,
                          "sigma": (ed_m / ed_e if ed_e else float("nan")),
                          "frac_resolved_br_flatter": frac_resolved_br_flatter},
        "policy_divergence": {"observed_mean_l1": obs_l1, "null_p997": null_p997,
                              "frac_null_ge_obs": frac_null_ge, "perm_b": perm_b,
                              "significant": policy_divergence_significant},
        "n_distinct_shifted_actions": n_distinct_shifted,
        "shifted_action_hist": {int(a): int(c) for a, c in shifted_hist.items()},
        "mean_entropy": {"br": mean_h_br, "profile": mean_h_pr},
        "mean_kl_br_blueprint": mean_kl,
    }


def verdict(agg, filter_stats):
    """Pre-committed split-metric verdict. status in {PASS, SUBSTANTIVE_PASS_AGGREGATE, FAIL}.

    Entropy direction is SOFT (Addition 2): it never gates. The load-bearing measures
    are value_suppression (mechanism fires), policy_divergence (BR moves the policy),
    differentiation (BR refines off blueprint — direction-agnostic), and non-degeneracy."""
    rr = agg["resolution_rate"]
    dr = agg["differentiation_rate"]
    vs = agg["value_suppression"]
    pd = agg["policy_divergence"]
    ed = agg["entropy_delta"]
    nds = agg["n_distinct_shifted_actions"]

    vs_ok = math.isfinite(vs["mean"]) and vs["mean"] > 0 and \
        math.isfinite(vs["sigma"]) and vs["sigma"] >= 3.0
    # entropy: SOFT — reported, never a FAIL on its own
    ent_sig = math.isfinite(ed["sigma"]) and ed["sigma"] >= 2.0 and ed["mean"] > 0
    ent_note = ("entropy direction CONFIRMS predicted flatness "
                f"(+{ed['sigma']:.1f}σ)." if ent_sig else
                f"entropy direction NOT significant in predicted (flatter) direction "
                f"({ed['mean']:+.4f}, {ed['sigma']:+.1f}σ) — surfacing for "
                f"interpretation; the flatness prediction was an over-specific "
                f"theoretical assumption, NOT a load-bearing gate (Addition 2).")

    lines = [
        f"resolution_rate = {rr:.1%}  [most mode-deviating action >3σ CRN-paired on "
        f"{agg['n_resolved']}/{agg['n_usable']} usable roots].",
        f"differentiation_rate = {dr:.1%}  [among resolved, BR-vs-PROFILE policy shift "
        f"L1 >= {L1_FLOOR} (material); direction-AGNOSTIC].",
        f"value_suppression = {vs['mean']:+.4f} ± {vs['stderr']:.4f} ({vs['sigma']:+.1f}σ)  "
        f"[BR opponents hurt hero — continuity with Stage F]. LOAD-BEARING.",
        f"policy_divergence = mean L1 {pd['observed_mean_l1']:.4f} vs null p99.7 "
        f"{pd['null_p997']:.4f} (significant={pd['significant']}). LOAD-BEARING.",
        f"n_distinct_shifted_actions = {nds}  [non-degeneracy guard].",
        f"entropy_delta (SOFT): {ent_note}",
        f"filter_rate (Addition 1) = {filter_stats['filter_rate']:.1%} "
        f"({filter_stats['excluded_few_actions']}/{filter_stats['roots_passing_other_filters']} "
        f"valid roots excluded for <{MIN_ACTIONS} legal actions).",
    ]

    strict = (math.isfinite(rr) and rr >= 0.50
              and ((math.isfinite(dr) and dr >= 0.60) or rr >= 0.90)
              and vs_ok)
    if strict:
        return "PASS", lines

    substantive = (math.isfinite(rr) and rr < 0.50 and vs_ok
                   and pd["significant"]
                   and math.isfinite(dr) and dr >= 0.55
                   and nds >= 2)
    if substantive:
        return "SUBSTANTIVE_PASS_AGGREGATE", lines

    return "FAIL", lines


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


def _strip_internal(records):
    """Drop the bulky per-sample arrays from the JSON payload (kept only for the
    in-memory permutation null)."""
    out = []
    for r in records:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        out.append(rr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--samples", type=int, default=16, help="M per condition (default 16)")
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--smoke", action="store_true", help="tiny run: N=5, M=8")
    ap.add_argument("--min-late-roots", type=int, default=20)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.n, args.samples, args.min_late_roots = 5, 8, 2

    abstr = Abstraction.load(ABSTR)
    structure = TournamentStructure.from_yaml(STRUCT)
    solver = _load_solver(CKPT, abstr, structure)
    biased = BiasedBlueprint()
    stack = int(solver.encoder.starting_stack)
    print(f"Stage G decision-level ablation | blueprint={CKPT}")
    print(f"N={args.n} roots, M={args.samples}/condition, seed={args.seed}, stack={stack}")

    master_rng = random.Random(args.seed)
    t_build = time.perf_counter()
    battery, filter_stats = build_root_battery(structure, master_rng, args.n,
                                               min_late_roots=args.min_late_roots)
    street_hist = {s: sum(1 for e in battery if e["meta"]["street_idx"] == s)
                   for s in range(4)}
    n_late = sum(v for s, v in street_hist.items() if s >= 2)
    print(f"Built {len(battery)} roots in {time.perf_counter()-t_build:.1f}s "
          f"(per-street {street_hist}; late[>=2]={n_late})")
    print(f"FILTER RATE (Addition 1): {filter_stats['filter_rate']:.1%} "
          f"({filter_stats['excluded_few_actions']}/{filter_stats['roots_passing_other_filters']} "
          f"valid roots had <{MIN_ACTIONS} legal actions)")
    if math.isfinite(filter_stats["filter_rate"]) and filter_stats["filter_rate"] > 0.30:
        print(f"  *** filter_rate > 30%: a substantial fraction of real decisions are "
              f"degenerate fold/shove; the architecture is measured on the "
              f"'interesting' subset (itself a finding about the game). ***")

    records = []
    t0 = time.perf_counter()
    for i, entry in enumerate(battery):
        rec = evaluate_root(entry, solver, biased, args.samples,
                            seed=master_rng.randrange(2 ** 31))
        records.append(rec)
        if (i + 1) % 5 == 0 or i + 1 == len(battery):
            print(f"  evaluated {i+1}/{len(battery)} roots [{time.perf_counter()-t0:.1f}s]")
    wall = time.perf_counter() - t0

    agg = aggregate(records)
    status, vlines = verdict(agg, filter_stats)

    print("\n" + "=" * 70)
    print("STAGE G — Q11 LEVEL 2 DECISION-LEVEL ABLATION — SUMMARY")
    print("=" * 70)
    print(f"roots: {agg['n_roots']} ({agg['n_usable']} usable, {agg['n_degraded']} degraded); "
          f"per-street {agg['street_distribution']}, late={agg['n_late_roots_ge2']}")
    print(f"\nSPLIT-METRIC GATE:")
    for ln in vlines:
        print(f"  - {ln}")
    print(f"\nREPORTED:")
    print(f"  mean entropy:  BR={agg['mean_entropy']['br']:.3f}  "
          f"PROFILE={agg['mean_entropy']['profile']:.3f}  (nats; higher=flatter)")
    print(f"  frac resolved roots BR-flatter: {agg['entropy_delta']['frac_resolved_br_flatter']:.1%}")
    print(f"  mean KL(BR‖blueprint): {agg['mean_kl_br_blueprint']:.4f}")
    print(f"  shifted-action hist: {agg['shifted_action_hist']}")
    print(f"\nwall-clock: {wall:.1f}s for {len(battery)} roots x ~children x 3 modes x M={args.samples}")
    print("\n" + "-" * 70)
    print(f"VERDICT: {status}")
    if status == "PASS":
        print("  => Stage G strict PASS. Stage F+G close sub-step 2; next is sub-step 3.")
    elif status == "SUBSTANTIVE_PASS_AGGREGATE":
        print("  => Strict PASS narrowly missed (per-root resolution intractable), but the "
              "load-bearing aggregate signals confirm the architecture. Sub-step 2 closes.")
    else:
        print("  => STOP and surface to the human (Path A). Do NOT proceed to sub-step 3.")
    print("-" * 70)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "smoke" if args.smoke else "production"
    out = Path(args.output) if args.output else Path(
        f"evals/decision_level_ablation_session18_{tag}_{ts}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "G", "q11_level": 2, "session": 18, "verdict": status,
        "verdict_notes": vlines,
        "repro": {
            "git_rev": _git_rev(), "blueprint_ckpt": CKPT, "blueprint_md5": _file_md5(CKPT),
            "abstraction": ABSTR, "structure": STRUCT, "seed": args.seed,
            "n_roots": args.n, "samples_per_condition": args.samples, "num_paid": NUM_PAID,
            "min_late_roots": args.min_late_roots, "min_actions_filter": MIN_ACTIONS,
            "l1_floor": L1_FLOOR, "perm_b": PERM_B, "starting_stack": stack,
            "python": sys.version.split()[0], "numpy": np.__version__, "timestamp_utc": ts,
        },
        "filter_stats": filter_stats,
        "aggregate": agg,
        "per_root": _strip_internal(records),
    }
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nWrote {out}")
    return status


if __name__ == "__main__":
    main()
