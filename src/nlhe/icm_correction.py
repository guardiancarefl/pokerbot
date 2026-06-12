"""Bubble-cell ICM correction v1 — standalone opt-in layer over MH ICM.

Registration: docs/research_program/ICM_CORRECTION_SPEC.md (frozen
2026-06-12). Model M1: per-seat additive correction with conservation
projection,

    Delta_i = g(depth_bb_i) * h(CV(s))

where g is continuous piecewise-linear with knots at the registered
depth-band edges {0, 5, 10, 20} bb (constant beyond 20), and h is
linear in alive-stack CV, anchored to 0 at the t1-median CV (exact
anchor form: h(CV) = max(0, (CV - cv0) / (cv1 - cv0)), cv0 = t1 fit-set
median CV, cv1 = t3 fit-set median CV — a scale convention only; the
scale is absorbed by g). Corrected probabilities:

    p_corr = clip(q + Delta, 0, 1), then conservation projection so
    that sum over alive seats = n_paid (=3), implemented as iterative
    redistribution of the excess proportional to |Delta_i| (seats with
    Delta = 0 move minimally; if all Delta = 0 the state is untouched
    because MH conserves exactly).

Monotone in q by construction: Delta does not depend on q.

SCOPE (registered): the correction applies ONLY when the state is a
4-alive (bubble) hand-start under a 3-paid equal-payout structure with
no newly-busted eligible seat. Everywhere else the functions return the
plain Malmuth-Harville values (identity). Tier-1 evidence covers only
n_alive=4, levels 3-4; extension needs Tier-2 (separate registration).

This module is NOT wired into src/nlhe/icm.py or any consumer.
Consumers adopt it only via their own gates, behind explicit flags,
per spec sections 3-4. Fitted coefficients live in
data/icm_correction_v1.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from src.nlhe.icm import icm_equity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ARTIFACT = REPO_ROOT / "data" / "icm_correction_v1.json"

KNOTS = (0.0, 5.0, 10.0, 20.0)   # registered depth-band edges, bb
N_PAID = 3


# ── Pure functional core (used by the fit script pre-artifact) ─────────

def g_basis(depth_bb: float) -> list[float]:
    """Hat-function basis for the continuous piecewise-linear g at the
    registered knots; constant extrapolation beyond the last knot."""
    d = max(0.0, float(depth_bb))
    b = [0.0] * len(KNOTS)
    if d >= KNOTS[-1]:
        b[-1] = 1.0
        return b
    for k in range(len(KNOTS) - 1):
        lo, hi = KNOTS[k], KNOTS[k + 1]
        if lo <= d < hi:
            t = (d - lo) / (hi - lo)
            b[k] = 1.0 - t
            b[k + 1] = t
            return b
    b[0] = 1.0   # d < KNOTS[0] (unreachable: d clipped at 0)
    return b


def g_eval(depth_bb: float, g_coef: Sequence[float]) -> float:
    return sum(c * b for c, b in zip(g_coef, g_basis(depth_bb)))


def h_eval(cv: float, cv0: float, cv1: float) -> float:
    """Anchored-linear CV term: 0 at/below cv0 (t1 median), 1 at cv1
    (t3 median), linear elsewhere (no upper clip)."""
    if cv1 <= cv0:
        return 1.0
    return max(0.0, (cv - cv0) / (cv1 - cv0))


def cv_of(stacks: Sequence[float]) -> float:
    """Population CV of alive-seat chip fractions (matches the c1
    instrument's definition exactly)."""
    alive = [s for s in stacks if s > 0]
    tot = float(sum(alive))
    fr = [s / tot for s in alive]
    mu = sum(fr) / len(fr)
    var = sum((f - mu) ** 2 for f in fr) / len(fr)
    return math.sqrt(var) / mu


def project_conservation(q: Sequence[float], delta: Sequence[float],
                         target: float = float(N_PAID),
                         max_iter: int = 50) -> list[float]:
    """clip(q + delta, 0, 1) renormalized to sum = target by shrinking
    the corrections proportionally to |delta_i| (not by scaling p), so
    seats with delta = 0 move minimally."""
    d = [float(x) for x in delta]
    p = [min(1.0, max(0.0, qi + di)) for qi, di in zip(q, d)]
    for _ in range(max_iter):
        excess = sum(p) - target
        if abs(excess) < 1e-12:
            break
        mag = [abs(x) for x in d]
        tot = sum(mag)
        if tot < 1e-15:
            break   # delta == 0 everywhere: MH already conserves
        d = [di - excess * mi / tot for di, mi in zip(d, mag)]
        p = [min(1.0, max(0.0, qi + di)) for qi, di in zip(q, d)]
    return p


# ── Artifact-backed correction ─────────────────────────────────────────

class IcmCorrection:
    """Fitted bubble-cell correction. Construct from the serialized
    artifact (`IcmCorrection.load()`) or from a params dict.

    Two registered model families (spec 1.2):
      - M1: additive Delta = g(depth)*h(CV) (+ rank dummies) with the
        conservation projection.
      - M2 (registered fallback, ADOPTED for v1 after M1 failed holdout
        gate G3): per-bin (CV-tertile x depth-band) isotonic regression
        of p_hat on q, piecewise-constant step prediction, no
        conservation (registered M2 semantics).
    """

    def __init__(self, params: dict):
        self.params = params
        self.model = params.get("model", "M1")
        if self.model == "M1":
            self.g_coef = [float(x) for x in params["g_coef"]]
            self.cv0 = float(params["cv_anchor"]["cv0"])
            self.cv1 = float(params["cv_anchor"]["cv1"])
            self.variant = params.get("variant", "depth_x_cv")
            self.rank_coef = params.get("rank_coef")   # variant M1c
            assert tuple(params.get("knots", KNOTS)) == KNOTS
        elif self.model == "M2":
            self.tertile_cuts = [float(c) for c in params["tertile_cuts"]]
            self.depth_bands = [(float(lo), float("inf") if hi is None
                                 else float(hi))
                                for lo, hi in params["depth_bands"]]
            self.tables = params["tables"]
        else:
            raise ValueError(f"unknown model {self.model!r}")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT) -> "IcmCorrection":
        with open(path) as f:
            return cls(json.load(f))

    # -- scope predicate (registered: bubble cell only) --
    def in_scope(self, stacks: Sequence[float],
                 payouts: Sequence[float],
                 eligible: Sequence[int] | None) -> bool:
        n = len(stacks)
        elig = list(range(n)) if eligible is None else list(eligible)
        if any(stacks[i] <= 0 for i in elig):
            return False   # newly-busted seat: terminal semantics, not
                           # a hand-start; out of Tier-1 support
        alive = [i for i in elig if stacks[i] > 0]
        if len(alive) != 4:
            return False
        if len(payouts) != N_PAID:
            return False
        if any(abs(p - payouts[0]) > 1e-9 for p in payouts):
            return False   # only the top-3-equal format was measured
        return True

    def delta(self, stacks: Sequence[float], big_blind: float,
              eligible: Sequence[int]) -> dict[int, float]:
        """Raw (pre-projection) per-seat correction in prob units.

        Variants (frozen nested set, spec 1.2):
          depth_only:            Delta_i = g(depth_i)
          depth_x_cv:            Delta_i = g(depth_i) * h(CV)
          depth_x_cv_plus_rank:  Delta_i = (g(depth_i) + r_rank(i)) * h(CV)
        Rank dummies (shortest=rank 1..3; rank 4 = leader is the zero
        reference) are also anchored by h so all variants are identity
        at low CV except depth_only.
        """
        cv = cv_of([stacks[i] for i in eligible])
        h = 1.0 if self.variant == "depth_only" \
            else h_eval(cv, self.cv0, self.cv1)
        ranked = sorted(eligible, key=lambda i: (stacks[i], i))
        out = {}
        for i in eligible:
            d = g_eval(stacks[i] / big_blind, self.g_coef)
            if self.rank_coef is not None:
                rk = ranked.index(i)            # 0 = shortest stack
                if rk < len(self.rank_coef):
                    d += float(self.rank_coef[rk])
            out[i] = d * h
        return out

    # -- M2 step lookup --
    def _m2_predict(self, cv: float, depth_bb: float, q: float) -> float:
        import bisect
        cell = ("t1" if cv < self.tertile_cuts[0]
                else "t2" if cv < self.tertile_cuts[1] else "t3")
        band = None
        for lo, hi in self.depth_bands:
            if lo <= depth_bb < hi:
                band = (f"[{int(lo)},{int(hi)})" if hi != float("inf")
                        else f"{int(lo)}+")
                break
        tab = self.tables.get(f"{cell}|{band}")
        if tab is None:
            return q                  # empty/unseen bin: identity
        k = max(0, bisect.bisect_right(tab["q_lo"], q) - 1)
        return min(1.0, max(0.0, float(tab["value"][k])))

    def corrected_itm_probs(self, stacks: Sequence[float],
                            payouts: Sequence[float],
                            eligible: Sequence[int] | None = None,
                            *, big_blind: float) -> list[float]:
        """Per-seat P(finish in the money). Identity (= MH q) outside
        the registered scope. `big_blind` is required because the fitted
        feature is depth in current big blinds (spec 1.2); this is the
        one deliberate deviation from the spec's proposed signature.
        """
        n = len(stacks)
        elig = list(range(n)) if eligible is None else list(eligible)
        eq = icm_equity(stacks, payouts, eligible=elig)
        unit = payouts[0] if payouts else 1.0
        q_full = [e / unit if unit else 0.0 for e in eq]
        if not self.in_scope(stacks, payouts, eligible):
            return q_full
        alive = [i for i in elig if stacks[i] > 0]
        out = list(q_full)
        if self.model == "M2":
            cv = cv_of([stacks[i] for i in alive])
            for i in alive:
                out[i] = self._m2_predict(cv, stacks[i] / big_blind,
                                          q_full[i])
            return out
        d = self.delta(stacks, big_blind, alive)
        q_alive = [q_full[i] for i in alive]
        p_alive = project_conservation(q_alive, [d[i] for i in alive])
        for i, p in zip(alive, p_alive):
            out[i] = p
        return out

    def corrected_icm_equity(self, stacks: Sequence[float],
                             payouts: Sequence[float],
                             eligible: Sequence[int] | None = None,
                             *, big_blind: float) -> list[float]:
        """Drop-in equity wrapper: corrected probs x per-payout unit.
        Equals icm_equity(...) exactly outside the registered scope."""
        probs = self.corrected_itm_probs(stacks, payouts, eligible,
                                         big_blind=big_blind)
        unit = payouts[0] if payouts else 0.0
        return [p * unit for p in probs]
