"""Layer 4 / C1b — observation-driven BiasConfig builders.

Two paths per locked decision §6-Q3:
  - stats_to_bias_configs_raw: §4 mapping table with z-scored stat deviations
    against BLUEPRINT_REF, exp-of-linear with clip to [1/alpha, alpha], k=4
    menu of progressively-scaled perturbations.
  - stats_to_bias_configs_archetype: Bayesian posterior over NAMED_ARCHETYPES
    from aggregate-stat likelihoods, posterior-weighted action-distribution
    mixture (k=0) plus top-3 pure-archetype BR configs (k=1..3).

Both functions return list[BiasConfig] of length k (default 4). At
confidence == 0 BOTH paths return all-ones multipliers — the bit-identity
gate that C1c relies on to preserve the SUBSTEP_6_DESIGN BR/PROFILE/blueprint
verdict when C1 is disabled.

α_C1 default = 2.0 (locked §5-Q1). Distinct from the leaf-eval menu's
alpha=3.0 in biased_policy.standard_bias_configs.

Semantic note (hero-defensive): the per-action multipliers here represent
hero's preferred bias under the observation — e.g. when the opponent's
observed aggression_freq is high, the FOLD multiplier rises (hero is more
inclined to fold against an aggressor). C1c integration decides how to
project this onto the leaf-eval surface; C1b just produces the configs.

The empirical bake-off between the raw and archetype paths is deferred to
C1d. Bucket_id is REQUIRED for the archetype path at confidence > 0
(deferred §6-Q3 sub-decision: marginalization vs abstraction-mean
conditioning is a C1d empirical call).
"""
from __future__ import annotations

import random as _random
from typing import Optional

import numpy as np

from src.nlhe.within_match import SeatStats
from src.nlhe.biased_policy import BiasConfig
from src.nlhe.archetypes import NAMED_ARCHETYPES, archetype_policy

ALPHA_C1_DEFAULT = 2.0
N_ACTIONS = 7  # FOLD, CALL, BET_33, BET_66, BET_100, BET_200, ALLIN

# Blueprint-population reference values for raw-path stat z-scoring.
# (mean, sd) per stat. Conservative midpoints; refine in C1d if calibration
# data improves.
BLUEPRINT_REF: dict[str, tuple[float, float]] = {
    "vpip": (0.30, 0.10),
    "pfr": (0.18, 0.08),
    "aggression_freq": (0.45, 0.15),
    "fold_to_bet": (0.55, 0.15),
    "avg_bet_over_pot": (0.65, 0.20),
}

# Per-stat per-direction mapping matrix W[stat][direction] is a shape (7,)
# vector. Each entry is small (~0.1–0.3) so exp(confidence × Σ |z| × W) stays
# in a reasonable range before clipping to [1/alpha, alpha]. Semantics are
# hero-defensive throughout: positive entries on FOLD when the observation
# argues for tighter hero play (e.g. high opp aggression), etc.
RAW_STATS_W: dict[str, dict[str, np.ndarray]] = {
    "vpip": {
        # opp loose preflop → wider opp range → hero plays more pots
        "high": np.array([-0.20, +0.20, +0.10, +0.10, +0.10, +0.05, +0.00]),
        # opp tight preflop → tight range → fold marginals, don't c-bet bluffs
        "low":  np.array([+0.25, -0.10, -0.10, -0.10, -0.10, -0.05, +0.00]),
    },
    "pfr": {
        # opp PFR high → polar 3-bets → hero defends slightly tighter
        "high": np.array([+0.10, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05]),
        # opp PFR low → limpers → hero exploits with raises
        "low":  np.array([0.0, +0.10, +0.10, +0.10, +0.10, +0.05, 0.0]),
    },
    "aggression_freq": {
        # opp aggressive postflop → hero defends: fold more, call less, bluff
        # bet less (test 5 lock: FOLD multiplier > 1 in this case)
        "high": np.array([+0.30, -0.20, -0.10, -0.10, -0.10, -0.05, +0.05]),
        # opp passive → hero bets more freely (no resistance)
        "low":  np.array([-0.10, +0.10, +0.15, +0.20, +0.20, +0.15, +0.05]),
    },
    "fold_to_bet": {
        # opp folds easily → hero exploits fold equity with more bets
        "high": np.array([-0.10, -0.05, +0.15, +0.20, +0.20, +0.15, +0.10]),
        # opp calls down → hero only value-bets
        "low":  np.array([+0.05, +0.20, -0.10, -0.10, -0.05, -0.05, -0.05]),
    },
    "avg_bet_over_pot": {
        # opp bets large → hero defends tighter, marginals fold
        "high": np.array([+0.20, -0.10, -0.05, 0.0, +0.05, +0.10, +0.05]),
        # opp bets small → hero calls cheap, bets smaller in response
        "low":  np.array([-0.10, +0.20, +0.05, 0.0, -0.05, -0.05, -0.05]),
    },
}

# Per-archetype expected stat ratios for the archetype-path likelihood
# computation. Derived from NAMED_ARCHETYPES' play_quantile_by_street +
# aggression scalars: VPIP ≈ 1 - q_preflop, PFR ≈ VPIP × aggression,
# aggression_freq ≈ aggression, fold_to_bet ≈ mean(q_postflop),
# avg_bet_over_pot eyeballed from aggression intensity.
ARCHETYPE_REF: dict[str, dict[str, float]] = {
    "NIT":     {"vpip": 0.15, "pfr": 0.04, "aggression_freq": 0.25, "fold_to_bet": 0.68, "avg_bet_over_pot": 0.40},
    "TAG":     {"vpip": 0.30, "pfr": 0.20, "aggression_freq": 0.65, "fold_to_bet": 0.50, "avg_bet_over_pot": 0.65},
    "LAG":     {"vpip": 0.60, "pfr": 0.51, "aggression_freq": 0.85, "fold_to_bet": 0.35, "avg_bet_over_pot": 0.85},
    "STATION": {"vpip": 0.75, "pfr": 0.15, "aggression_freq": 0.20, "fold_to_bet": 0.10, "avg_bet_over_pot": 0.40},
    "MANIAC":  {"vpip": 0.95, "pfr": 0.90, "aggression_freq": 0.95, "fold_to_bet": 0.20, "avg_bet_over_pot": 1.20},
}

# Gaussian likelihood σ per stat for the archetype posterior. Loose so noisy
# early-match readings don't lock in.
ARCHETYPE_LIK_SIGMA: dict[str, float] = {
    "vpip": 0.15,
    "pfr": 0.15,
    "aggression_freq": 0.20,
    "fold_to_bet": 0.20,
    "avg_bet_over_pot": 0.30,
}

# k=4 menu scaling factors for the raw path. k=0 is the canonical bias; k=1..3
# attenuate the same direction so the leaf-eval BR has a graded menu and the
# four entries are distinct even under moderate deviations (test 6 lock).
_RAW_K_SCALES = (1.0, 0.7, 0.5, 0.25)


def compute_seat_ratios(stats: SeatStats) -> dict[str, Optional[float]]:
    """Convert raw SeatStats counters to ratio summary.

    Returns a dict with keys {vpip, pfr, aggression_freq, fold_to_bet,
    avg_bet_over_pot}; each value is the computed ratio or None when the
    denominator is zero (no observations yet)."""
    out: dict[str, Optional[float]] = {}
    out["vpip"] = (stats.n_preflop_voluntary / stats.n_preflop_decisions
                  if stats.n_preflop_decisions > 0 else None)
    out["pfr"] = (stats.n_preflop_raises / stats.n_preflop_decisions
                 if stats.n_preflop_decisions > 0 else None)
    total_post_dec = sum(stats.n_postflop_decisions)
    total_post_agg = sum(stats.n_postflop_aggressive)
    out["aggression_freq"] = (total_post_agg / total_post_dec
                              if total_post_dec > 0 else None)
    total_facing = sum(stats.n_facing_bet)
    total_folds = sum(stats.n_folds_facing_bet)
    out["fold_to_bet"] = (total_folds / total_facing
                          if total_facing > 0 else None)
    out["avg_bet_over_pot"] = (stats.sum_bet_size_over_pot / stats.n_bet_size_samples
                               if stats.n_bet_size_samples > 0 else None)
    return out


def _raw_log_mult(stats: SeatStats) -> np.ndarray:
    """Build the unscaled log-multiplier vector from observed deviations.

    Sum over stats with defined ratios: |z| × W[stat][direction].
    Returns shape (7,) float64. Returns all-zeros if no ratios are defined.
    """
    log_mult = np.zeros(N_ACTIONS, dtype=np.float64)
    ratios = compute_seat_ratios(stats)
    for stat_name, ref in BLUEPRINT_REF.items():
        ratio = ratios.get(stat_name)
        if ratio is None:
            continue
        mean, sd = ref
        if sd <= 0:
            continue
        z = (ratio - mean) / sd
        direction = "high" if z >= 0 else "low"
        w = RAW_STATS_W[stat_name][direction]
        log_mult += abs(z) * w
    return log_mult


def stats_to_bias_configs_raw(
    stats: SeatStats,
    confidence: float,
    alpha: float = ALPHA_C1_DEFAULT,
    k: int = 4,
) -> list[BiasConfig]:
    """Raw-stats z-score path.

    At confidence == 0 returns k all-ones BiasConfigs (blueprint-recovery
    bit-identity for C1c). At confidence > 0 the k entries are progressively-
    attenuated scalings of the base log-multiplier vector, each clipped to
    [1/alpha, alpha].

    Names: "c1_raw_k0", "c1_raw_k1", ..., "c1_raw_k{k-1}".
    """
    if k <= 0:
        return []
    if confidence <= 0.0:
        return [
            BiasConfig(name=f"c1_raw_k{i}",
                       multipliers=np.ones(N_ACTIONS, dtype=np.float64))
            for i in range(k)
        ]

    base = _raw_log_mult(stats) * float(confidence)
    scales = _RAW_K_SCALES if k == len(_RAW_K_SCALES) else tuple(
        # fallback for non-default k: spread linearly between 1.0 and 0.25
        1.0 - 0.75 * (i / max(k - 1, 1)) for i in range(k)
    )
    lo, hi = 1.0 / float(alpha), float(alpha)
    configs: list[BiasConfig] = []
    for i in range(k):
        log_mult = base * scales[i]
        mults = np.clip(np.exp(log_mult), lo, hi).astype(np.float64)
        configs.append(BiasConfig(name=f"c1_raw_k{i}", multipliers=mults))
    return configs


def _compute_archetype_posterior(stats: SeatStats) -> np.ndarray:
    """Bayesian posterior over NAMED_ARCHETYPES from aggregate stats.

    Likelihood is a per-stat Gaussian on (observed_ratio - archetype_ref) /
    sigma; log-likelihoods sum across stats with defined ratios. Uniform
    prior. Returns shape (5,) probability vector ordered like
    NAMED_ARCHETYPES.

    If NO ratio is defined (fresh stats), returns uniform 1/5 — no info,
    no preference.
    """
    ratios = compute_seat_ratios(stats)
    n_arch = len(NAMED_ARCHETYPES)
    log_likelihoods = np.zeros(n_arch, dtype=np.float64)
    any_defined = False
    for arch_idx, arch in enumerate(NAMED_ARCHETYPES):
        arch_name = arch.name.name
        ref = ARCHETYPE_REF.get(arch_name)
        if ref is None:
            continue
        ll = 0.0
        for stat_name, obs_value in ratios.items():
            if obs_value is None:
                continue
            ref_value = ref.get(stat_name)
            sigma = ARCHETYPE_LIK_SIGMA.get(stat_name)
            if ref_value is None or sigma is None or sigma <= 0:
                continue
            any_defined = True
            z = (obs_value - ref_value) / sigma
            ll += -0.5 * z * z
        log_likelihoods[arch_idx] = ll
    if not any_defined:
        return np.full(n_arch, 1.0 / n_arch, dtype=np.float64)
    # Softmax with the standard max-subtract for numerical stability.
    log_likelihoods -= log_likelihoods.max()
    liks = np.exp(log_likelihoods)
    return liks / liks.sum()


def _derive_leaf_context(parsed: dict, state) -> dict:
    """Pull street/legal/pot/facing context for the leaf infoset.

    Best-effort: prefers explicit fields in `parsed` (test friendliness),
    falls back to deriving from state when present, finally defaults.
    """
    out: dict = {}
    out["street_idx"] = int(parsed.get("street_idx", 0))

    legal_mask = parsed.get("legal_mask")
    if legal_mask is None and state is not None:
        try:
            from src.nlhe.cfr6 import _build_view_6max
            from src.nlhe.actions import discretize_legal_actions
            view = _build_view_6max(state, parsed)
            discrete_to_chip = discretize_legal_actions(list(state.legal_actions()), view)
            lm = np.zeros(N_ACTIONS, dtype=np.float32)
            for da in discrete_to_chip:
                lm[int(da)] = 1.0
            legal_mask = lm
        except Exception:
            legal_mask = np.ones(N_ACTIONS, dtype=np.float32)
    elif legal_mask is None:
        legal_mask = np.ones(N_ACTIONS, dtype=np.float32)
    out["legal_mask"] = np.asarray(legal_mask, dtype=np.float32)

    contribution = parsed.get("contribution", [0] * 6)
    cp = int(parsed.get("current_player", 0))
    try:
        cp_contrib = int(contribution[cp])
        max_contrib = int(max(contribution)) if contribution else 0
    except (TypeError, ValueError, IndexError):
        cp_contrib = 0
        max_contrib = 0
    out["facing_bet"] = cp_contrib < max_contrib
    out["to_call"] = max(max_contrib - cp_contrib, 0)

    out["pot"] = float(parsed.get("pot", 0) or 0)
    money = parsed.get("money", [0] * 6)
    try:
        eff_stack = float(money[cp])
    except (TypeError, ValueError, IndexError):
        eff_stack = 0.0
    out["eff_stack"] = eff_stack

    denom_po = out["pot"] + out["to_call"]
    out["pot_odds"] = (out["to_call"] / denom_po) if denom_po > 0 else 0.0
    out["stack_to_pot"] = (out["eff_stack"] / out["pot"]) if out["pot"] > 0 else 10.0
    return out


def _multipliers_from_dist(dist: np.ndarray, legal_mask: np.ndarray,
                           confidence: float, alpha: float) -> np.ndarray:
    """Convert an opponent-action distribution to per-action multipliers.

    Multipliers scale the blueprint dist toward `dist`: action slots where
    `dist` puts more mass than uniform-over-legal get mult > 1; slots with
    less mass get mult < 1. Illegal slots get mult = 1 (no effect). Scaled
    by `confidence` and clipped to [1/alpha, alpha].
    """
    legal_bool = legal_mask > 0
    n_legal = float(legal_bool.sum())
    if n_legal <= 0:
        return np.ones(N_ACTIONS, dtype=np.float64)
    uniform_legal = 1.0 / n_legal
    safe_dist = np.maximum(dist, 1e-9)
    log_mult = float(confidence) * np.log(safe_dist / uniform_legal)
    log_mult = np.where(legal_bool, log_mult, 0.0)
    lo, hi = 1.0 / float(alpha), float(alpha)
    return np.clip(np.exp(log_mult), lo, hi).astype(np.float64)


def stats_to_bias_configs_archetype(
    stats: SeatStats,
    confidence: float,
    parsed: dict,
    state,
    bucket_id: Optional[int],
    in_position: bool,
    *,
    calibration=None,
    alpha: float = ALPHA_C1_DEFAULT,
    k: int = 4,
) -> list[BiasConfig]:
    """Archetype-Bayesian posterior path.

    Algorithm:
      1. Compute posterior over the 5 NAMED_ARCHETYPES from aggregate stats
         via Gaussian likelihood (_compute_archetype_posterior).
      2. Evaluate each archetype's action distribution at the leaf via
         archetypes.archetype_policy (using `bucket_id`, `in_position`, and
         per-leaf derived context: street_idx, pot_odds, stack_to_pot,
         legal_mask, facing_bet).
      3. Posterior-weighted mixture distribution → BiasConfig multipliers
         (_multipliers_from_dist).
      4. k=4 menu: k=0 is the mixture-derived config; k=1..k-1 are
         pure-archetype configs for the top-3 archetypes in posterior order.

    Bucket_id handling (§6-Q3 sub-decision deferred to C1d): bucket_id is
    REQUIRED at confidence > 0; passing None raises ValueError. C1d will
    decide between caller-supplied bucket vs uniform marginalization vs
    abstraction-mean conditioning.

    Names: "c1_arch_k0", "c1_arch_k1_<top_arch>", "c1_arch_k2_<2nd_arch>",
    "c1_arch_k3_<3rd_arch>".
    """
    if k <= 0:
        return []
    if confidence <= 0.0:
        # Blueprint-recovery short-circuit. No bucket/calibration needed.
        return [
            BiasConfig(name=f"c1_arch_k{i}",
                       multipliers=np.ones(N_ACTIONS, dtype=np.float64))
            for i in range(k)
        ]

    if bucket_id is None:
        raise ValueError(
            "stats_to_bias_configs_archetype: bucket_id is required when "
            "confidence > 0 (§6-Q3 sub-decision deferred to C1d — caller "
            "must supply the opponent's bucket)."
        )
    if calibration is None:
        raise ValueError(
            "stats_to_bias_configs_archetype: calibration is required when "
            "confidence > 0 (caller supplies the EquityCalibration matching "
            "the abstraction in use)."
        )

    ctx = _derive_leaf_context(parsed, state)
    legal_mask = ctx["legal_mask"]
    rng_dummy = _random.Random(0)  # archetype_policy is deterministic; rng unused

    archetype_dists = np.zeros((len(NAMED_ARCHETYPES), N_ACTIONS), dtype=np.float64)
    for arch_idx, arch in enumerate(NAMED_ARCHETYPES):
        dist = archetype_policy(
            archetype=arch,
            calibration=calibration,
            street_idx=ctx["street_idx"],
            bucket_id=int(bucket_id),
            in_position=bool(in_position),
            pot_odds=ctx["pot_odds"],
            stack_to_pot=ctx["stack_to_pot"],
            legal_mask=legal_mask.astype(np.float32),
            facing_bet=ctx["facing_bet"],
            rng=rng_dummy,
        )
        archetype_dists[arch_idx] = np.asarray(dist, dtype=np.float64)

    posterior = _compute_archetype_posterior(stats)
    mix_dist = posterior @ archetype_dists  # shape (7,)

    configs: list[BiasConfig] = []
    # k=0: mixture-derived multipliers.
    mults_k0 = _multipliers_from_dist(mix_dist, legal_mask, confidence, alpha)
    configs.append(BiasConfig(name="c1_arch_k0", multipliers=mults_k0))

    # k=1..k-1: top archetypes by posterior (cycle if k > 5).
    top_indices = np.argsort(-posterior)
    for slot in range(1, k):
        arch_idx = int(top_indices[(slot - 1) % len(NAMED_ARCHETYPES)])
        arch_name = NAMED_ARCHETYPES[arch_idx].name.name
        mults = _multipliers_from_dist(
            archetype_dists[arch_idx], legal_mask, confidence, alpha)
        configs.append(BiasConfig(name=f"c1_arch_k{slot}_{arch_name}",
                                  multipliers=mults))
    return configs
