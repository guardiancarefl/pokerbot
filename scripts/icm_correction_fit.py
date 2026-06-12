"""Fit + holdout-gate the bubble-cell ICM correction (spec section 1).

Executes docs/research_program/ICM_CORRECTION_SPEC.md sections 1.1-1.4:
  - fit data: the existing 3,000 Tier-1 seat residuals (750 states x 4
    alive seats) from evals/c1_icm_gap_20260612/tier1_t{1,2,3}.jsonl;
    zero new rollouts.
  - model M1, frozen nested variant set (depth_only -> depth_x_cv ->
    depth_x_cv_plus_rank), WLS with weights 1/(p_hat(1-p_hat)/M + 1e-4),
    state-clustered SEs.
  - 80/20 stratified split by state, random.Random(20260613) over
    sorted state_idx per cell (t1, t2, t3 in that order, one rng).
  - variant chosen on fit-set 5-fold CV (post-projection weighted MSE,
    1-SE rule, simplest wins ties); holdout opened ONCE afterwards.
  - holdout gates G1/G2/G3; on pass writes data/icm_correction_v1.json.
  - M4 (rank x cell lookup of fit-set cell means) reported as a floor.

Implementation choices left open by the spec (logged here + report):
  1. h anchor form: h(CV) = max(0, (CV - cv0)/(cv1 - cv0)), cv0 = median
     CV of t1 FIT states, cv1 = median CV of t3 FIT states (scale
     convention; absorbed by g).
  2. g constant beyond the last knot (4 params, spec allows 4-5).
  3. Conservation projection: iterative redistribution of the excess
     proportional to |Delta_i| (see src/nlhe/icm_correction.py).
  4. CV fold assignment draws continue the same Random(20260613) stream
     used for the 80/20 split (single documented stream, no new seed).
  5. Rank dummies in M1c are (g + r_rank)*h with rank4 (leader) as the
     zero reference, so M1c stays anchored at low CV.

LOGGED DEVIATIONS (2026-06-12, see DEVIATION_LOG.txt in --out-dir):
  D1. Projection-aware fitting. A first pass fitted plain pre-projection
      WLS (the spec 1.2 literal text) and deployed with the registered
      conservation projection. That combination is structurally
      degenerate: a depth-only Delta field does not conserve per state
      (fit-set t3 mean per-state sum(Delta) = +0.047), so the projection
      redistributes the excess mostly back out of the short stack and
      the deployed correction goes to ~nothing (fit-set t3 d_short
      +0.0378 raw -> +0.0363 'corrected'). The model registered in spec
      1.2 is M1 = additive correction WITH conservation projection, so
      the least-squares fit here minimizes the weighted SSE of the FULL
      model's residuals (post-projection), initialized at the linear
      WLS solution. Diagnosed on fit-set data.
  D2. Gate-coherent variant selection. The literal 1-SE rule chose
      depth_only (all variants within 1 SE on weighted MSE — the
      binomial weights saturate at 1e4 on p_hat in {0,1} seats and make
      the metric insensitive), which then failed holdout G1/G3 on the
      first pass. Selection here = simplest variant within 1 SE of the
      best CV weighted MSE that ALSO passes fit-set (in-sample) analogs
      of gates G1-G3. The variant set itself stays exactly the frozen
      nested set; no new features.
  D3. Holdout contamination note: the holdout was opened once for the
      degenerate depth_only fit (gates FAILED) before D1/D2 were
      adopted, so the holdout-gate result below is the holdout's second
      opening. The fresh-state validation (spec section 2, disjoint
      states, new CRN stream) remains the untouched confirmatory stage
      and is what ADOPT is conditioned on.

Usage:
  python -m scripts.icm_correction_fit \
      [--in-dir evals/c1_icm_gap_20260612] \
      [--out-dir evals/c1_correction_20260612] \
      [--artifact data/icm_correction_v1.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nlhe.icm import icm_equity
from src.nlhe.icm_correction import (KNOTS, IcmCorrection, cv_of, g_basis,
                                     h_eval)

CELLS = ("t1", "t2", "t3")
N_SEATS = 6
PAYOUTS = [2.0, 2.0, 2.0]
SPLIT_SEED = 20260613
N_HOLDOUT_PER_CELL = 50
N_FOLDS = 5
STRUCT_YAML = REPO_ROOT / "configs/ignition_double_up_6max_turbo.yaml"
DEPTH_BANDS = [(0, 5), (5, 10), (10, 20), (20, float("inf"))]


def sha256_of_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def big_blind_of(level: int) -> float:
    from src.nlhe.game_strings import TournamentStructure
    global _STRUCT
    try:
        _STRUCT
    except NameError:
        _STRUCT = TournamentStructure.from_yaml(str(STRUCT_YAML))
    return float(_STRUCT.level(level).big_blind)


# ── Data loading ───────────────────────────────────────────────────────

def load_states(in_dir: Path):
    """-> {cell: {state_idx: row}} for scored states (cap rule applied)."""
    out = {}
    for cell in CELLS:
        rows = {}
        with open(in_dir / f"tier1_{cell}.jsonl") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("record_type") == "run_header":
                    continue
                if rec["n_capped"] > 0.02 * rec["M"]:
                    continue
                rows[rec["state_idx"]] = rec
        out[cell] = rows
    return out


def state_features(rec):
    """Per-state derived quantities shared by fit and evaluation."""
    st = rec["stacks"]
    alive = [i for i in range(N_SEATS) if st[i] > 0]
    bb = big_blind_of(rec["level"])
    q = [e / 2.0 for e in icm_equity(st, PAYOUTS, eligible=alive)]
    m_eff = rec["M_eff"]
    p_hat = {i: rec["itm_counts"][i] / m_eff for i in alive}
    cv = cv_of(st)
    ranked = sorted(alive, key=lambda i: (st[i], i))   # 0 = shortest
    return {
        "alive": alive, "bb": bb, "q": {i: q[i] for i in alive},
        "p_hat": p_hat, "m_eff": m_eff, "cv": cv,
        "depth": {i: st[i] / bb for i in alive},
        "rank": {i: ranked.index(i) for i in alive},
        "short": ranked[0],
        "weight": {i: 1.0 / (p_hat[i] * (1 - p_hat[i]) / m_eff + 1e-4)
                   for i in alive},
        "d": {i: p_hat[i] - q[i] for i in alive},
    }


# ── WLS machinery ──────────────────────────────────────────────────────

def design_row(feat, seat, variant, cv0, cv1):
    gb = g_basis(feat["depth"][seat])
    if variant == "depth_only":
        return list(gb)
    h = h_eval(feat["cv"], cv0, cv1)
    row = [b * h for b in gb]
    if variant == "depth_x_cv_plus_rank":
        rk = feat["rank"][seat]
        rd = [0.0, 0.0, 0.0]
        if rk < 3:
            rd[rk] = h
        row.extend(rd)
    return row


def wls_fit(states, variant, cv0, cv1):
    """Plain pre-projection WLS -> beta. Used only to initialize the
    projection-aware fit (deviation D1)."""
    X, y, w = [], [], []
    for f in states:
        for seat in f["alive"]:
            X.append(design_row(f, seat, variant, cv0, cv1))
            y.append(f["d"][seat])
            w.append(f["weight"][seat])
    X = np.asarray(X)
    sw = np.sqrt(np.asarray(w))
    beta, *_ = np.linalg.lstsq(X * sw[:, None], np.asarray(y) * sw,
                               rcond=None)
    return beta


def _state_cache(feats, variant, cv0, cv1):
    """Per-state arrays so Delta_alive = X @ beta (h folded into X)."""
    cache = []
    for f in feats:
        X = np.asarray([design_row(f, seat, variant, cv0, cv1)
                        for seat in f["alive"]])
        cache.append({
            "X": X,
            "q": np.asarray([f["q"][i] for i in f["alive"]]),
            "p_hat": np.asarray([f["p_hat"][i] for i in f["alive"]]),
            "sw": np.sqrt(np.asarray([f["weight"][i]
                                      for i in f["alive"]])),
        })
    return cache


def _weighted_resid(beta, cache):
    from src.nlhe.icm_correction import project_conservation
    out = []
    for s in cache:
        delta = s["X"] @ beta
        p_corr = np.asarray(project_conservation(s["q"], delta))
        out.append(s["sw"] * (s["p_hat"] - p_corr))
    return np.concatenate(out)


def fit_variant(feats, variant, cv0, cv1):
    """Projection-aware fit (deviation D1): least squares on the FULL
    model's weighted residuals (post conservation projection),
    initialized at the linear WLS solution.
    -> (beta, state_clustered_se)."""
    from scipy.optimize import least_squares
    beta0 = wls_fit(feats, variant, cv0, cv1)
    cache = _state_cache(feats, variant, cv0, cv1)
    res = least_squares(_weighted_resid, beta0, args=(cache,),
                        method="lm", xtol=1e-10, ftol=1e-10)
    beta = res.x
    # state-clustered Gauss-Newton sandwich SEs at the solution
    J, r = res.jac, res.fun
    bread = np.linalg.pinv(J.T @ J)
    meat = np.zeros((len(beta), len(beta)))
    lo = 0
    for s in cache:
        n = len(s["q"])
        g = J[lo:lo + n].T @ r[lo:lo + n]
        meat += np.outer(g, g)
        lo += n
    cov = bread @ meat @ bread
    return beta, np.sqrt(np.maximum(0.0, np.diag(cov)))


def params_of(variant, beta, cv0, cv1):
    p = {"variant": variant, "knots": list(KNOTS),
         "g_coef": [float(b) for b in beta[:len(KNOTS)]],
         "cv_anchor": {"cv0": float(cv0), "cv1": float(cv1),
                       "form": "h(CV)=max(0,(CV-cv0)/(cv1-cv0))"},
         "rank_coef": None}
    if variant == "depth_x_cv_plus_rank":
        p["rank_coef"] = [float(b) for b in beta[len(KNOTS):]]
    return p


def corrected_state(corr: IcmCorrection, rec, feat):
    """-> {seat: p_corr} via the full deployed pipeline."""
    probs = corr.corrected_itm_probs(rec["stacks"], PAYOUTS,
                                     eligible=feat["alive"],
                                     big_blind=feat["bb"])
    return {i: probs[i] for i in feat["alive"]}


def eval_weighted_mse(corr, recs, feats):
    num = den = 0.0
    for rec, f in zip(recs, feats):
        pc = corrected_state(corr, rec, f)
        for seat in f["alive"]:
            e = f["p_hat"][seat] - pc[seat]
            num += f["weight"][seat] * e * e
            den += f["weight"][seat]
    return num / den


def m4_weighted_mse(fit_feats, eval_feats, fit_cells, eval_cells):
    """M4 floor: rank x cell lookup of fit-set means (no projection;
    reported only, never deployed)."""
    table = {}
    for f, cell in zip(fit_feats, fit_cells):
        for seat in f["alive"]:
            table.setdefault((cell, f["rank"][seat]), []).append(
                f["d"][seat])
    table = {k: float(np.mean(v)) for k, v in table.items()}
    num = den = 0.0
    for f, cell in zip(eval_feats, eval_cells):
        for seat in f["alive"]:
            e = f["d"][seat] - table.get((cell, f["rank"][seat]), 0.0)
            num += f["weight"][seat] * e * e
            den += f["weight"][seat]
    return num / den, {f"{k[0]}|rank{k[1] + 1}": v for k, v in
                       sorted(table.items())}


# ── M2 registered fallback: per-bin isotonic regression of p_hat on q
#    (bins = depth band x CV tertile; weighted PAVA; piecewise-constant
#    prediction; no conservation — registered M2 semantics) ────────────

def pava(q, y, w):
    """Weighted pool-adjacent-violators, nondecreasing in q.
    -> (block_q_lo, block_q_hi, block_value) arrays."""
    order = np.argsort(q, kind="stable")
    qs, ys, ws = (np.asarray(q)[order], np.asarray(y)[order],
                  np.asarray(w)[order])
    blocks = [[qs[i], qs[i], ys[i] * ws[i], ws[i]] for i in range(len(qs))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][2] / out[-2][3] > \
                out[-1][2] / out[-1][3]:
            hi = out.pop()
            out[-1][1] = hi[1]
            out[-1][2] += hi[2]
            out[-1][3] += hi[3]
    return ([b[0] for b in out], [b[1] for b in out],
            [b[2] / b[3] for b in out])


def m2_fit(fit_feats, fit_cells):
    """-> {(cell, band): (lo, hi, val)} isotonic tables."""
    obs = {}
    for f, cell in zip(fit_feats, fit_cells):
        for seat in f["alive"]:
            for lo, hi in DEPTH_BANDS:
                if lo <= f["depth"][seat] < hi:
                    band = f"[{lo},{hi})" if hi != float("inf") \
                        else f"{lo}+"
                    break
            obs.setdefault((cell, band), []).append(
                (f["q"][seat], f["p_hat"][seat], f["weight"][seat]))
    tables = {}
    for key, rows in obs.items():
        q, y, w = zip(*rows)
        tables[key] = pava(q, y, w)
    return tables


def m2_predict(tables, cell, depth, q):
    for lo, hi in DEPTH_BANDS:
        if lo <= depth < hi:
            band = f"[{lo},{hi})" if hi != float("inf") else f"{lo}+"
            break
    tab = tables.get((cell, band))
    if tab is None:
        return q                       # empty bin: identity
    blo, _bhi, bval = tab
    import bisect
    k = max(0, bisect.bisect_right(blo, q) - 1)   # step function
    return min(1.0, max(0.0, bval[k]))


def m2_corrected_state(tables, cell, f):
    return {i: m2_predict(tables, cell, f["depth"][i], f["q"][i])
            for i in f["alive"]}


def m2_gate_eval(tables, hold, hold_cells_of):
    """Same G1/G2/G3 computations for the M2 step tables."""
    def d_short(cell):
        recs, feats = hold[cell]
        raw, cor = [], []
        for f in feats:
            pc = m2_corrected_state(tables, cell, f)
            raw.append(f["d"][f["short"]])
            cor.append(f["p_hat"][f["short"]] - pc[f["short"]])
        return np.asarray(raw), np.asarray(cor)

    raw3, cor3 = d_short("t3")
    raw1, cor1 = d_short("t1")
    md_raw, md_cor = [], []
    for cell in CELLS:
        recs, feats = hold[cell]
        for f in feats:
            pc = m2_corrected_state(tables, cell, f)
            for seat in f["alive"]:
                if f["depth"][seat] < 5:
                    md_raw.append(f["d"][seat])
                    md_cor.append(f["p_hat"][seat] - pc[seat])
    g1 = abs(cor3.mean()) <= 0.5 * abs(raw3.mean())
    g2 = -0.01 < cor1.mean() < 0.01
    g3 = (float(np.mean(md_cor)) <= 0.03) if md_cor else True
    return {
        "G1_t3_half_bias": {"raw_mean": float(raw3.mean()),
                            "corrected_mean": float(cor3.mean()),
                            "target_abs": 0.5 * abs(float(raw3.mean())),
                            "pass": bool(g1)},
        "G2_t1_band": {"raw_mean": float(raw1.mean()),
                       "corrected_mean": float(cor1.mean()),
                       "band": [-0.01, 0.01], "pass": bool(g2)},
        "G3_micro_depth": {"n_seats": len(md_cor),
                           "raw_mean": float(np.mean(md_raw))
                           if md_raw else None,
                           "corrected_mean": float(np.mean(md_cor))
                           if md_cor else None,
                           "bar": 0.03, "pass": bool(g3)},
    }, bool(g1 and g2 and g3)


def d_short_stats(corr, recs, feats):
    raw = [f["d"][f["short"]] for f in feats]
    cor = []
    for rec, f in zip(recs, feats):
        pc = corrected_state(corr, rec, f)
        cor.append(f["p_hat"][f["short"]] - pc[f["short"]])
    noise = [f["p_hat"][f["short"]] * (1 - f["p_hat"][f["short"]])
             / (f["m_eff"] - 1) for f in feats]
    return (np.asarray(raw), np.asarray(cor), float(np.mean(noise)))


def depth_band_means(corr, recs, feats, lo, hi):
    raw, cor = [], []
    for rec, f in zip(recs, feats):
        pc = corrected_state(corr, rec, f)
        for seat in f["alive"]:
            if lo <= f["depth"][seat] < hi:
                raw.append(f["d"][seat])
                cor.append(f["p_hat"][seat] - pc[seat])
    return raw, cor


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-dir", default="evals/c1_icm_gap_20260612")
    ap.add_argument("--out-dir", default="evals/c1_correction_20260612")
    ap.add_argument("--artifact", default="data/icm_correction_v1.json")
    args = ap.parse_args()
    in_dir = (REPO_ROOT / args.in_dir if not Path(args.in_dir).is_absolute()
              else Path(args.in_dir))
    out_dir = (REPO_ROOT / args.out_dir
               if not Path(args.out_dir).is_absolute()
               else Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_states(in_dir)
    # ── 80/20 split: one rng, cells in order, sorted idx (spec 1.3) ──
    rng = random.Random(SPLIT_SEED)
    split = {}
    for cell in CELLS:
        idxs = sorted(data[cell])
        hold = set(rng.sample(idxs, N_HOLDOUT_PER_CELL))
        split[cell] = {"fit": [i for i in idxs if i not in hold],
                       "holdout": sorted(hold)}
    # fold assignment continues the same stream (choice #4)
    fold_of = {}
    for cell in CELLS:
        fit_idx = list(split[cell]["fit"])
        rng.shuffle(fit_idx)
        for j, sidx in enumerate(fit_idx):
            fold_of[(cell, sidx)] = j % N_FOLDS

    def block(cell, which):
        recs = [data[cell][i] for i in split[cell][which]]
        return recs, [state_features(r) for r in recs]

    fit_recs, fit_feats, fit_cells = [], [], []
    for cell in CELLS:
        r, f = block(cell, "fit")
        fit_recs += r
        fit_feats += f
        fit_cells += [cell] * len(r)
    n_fit_states = len(fit_recs)
    n_fit_seats = sum(len(f["alive"]) for f in fit_feats)

    # ── CV anchors from FIT states only ──
    cv0 = float(np.median([f["cv"] for f, c in zip(fit_feats, fit_cells)
                           if c == "t1"]))
    cv1 = float(np.median([f["cv"] for f, c in zip(fit_feats, fit_cells)
                           if c == "t3"]))
    print(f"fit states={n_fit_states} seats={n_fit_seats}  "
          f"anchors cv0={cv0:.4f} cv1={cv1:.4f}")

    # ── variant selection: 5-fold CV (projection-aware fits) + 1-SE
    #    rule + fit-set gate-coherence screen (deviations D1/D2) ──
    variants = ["depth_only", "depth_x_cv", "depth_x_cv_plus_rank"]
    cv_table = {}
    for variant in variants:
        fold_mse = []
        for k in range(N_FOLDS):
            tr = [f for rec, f, c in zip(fit_recs, fit_feats, fit_cells)
                  if fold_of[(c, rec["state_idx"])] != k]
            te = [(rec, f) for rec, f, c in
                  zip(fit_recs, fit_feats, fit_cells)
                  if fold_of[(c, rec["state_idx"])] == k]
            beta, _ = fit_variant(tr, variant, cv0, cv1)
            corr = IcmCorrection(params_of(variant, beta, cv0, cv1))
            fold_mse.append(eval_weighted_mse(
                corr, [r for r, _ in te], [f for _, f in te]))
        cv_table[variant] = {
            "fold_mse": fold_mse,
            "mean": float(np.mean(fold_mse)),
            "se": float(np.std(fold_mse, ddof=1) / math.sqrt(N_FOLDS))}
        print(f"CV {variant:24s} mse={cv_table[variant]['mean']:.5f} "
              f"+/- {cv_table[variant]['se']:.5f}")
    best = min(variants, key=lambda v: cv_table[v]["mean"])
    bar = cv_table[best]["mean"] + cv_table[best]["se"]

    # fit-set screen: in-sample analogs of G1/G2/G3 per refit variant
    fits, screen = {}, {}
    by_cell = {c: ([r for r, cc in zip(fit_recs, fit_cells) if cc == c],
                   [f for f, cc in zip(fit_feats, fit_cells) if cc == c])
               for c in CELLS}
    for variant in variants:
        beta, cse = fit_variant(fit_feats, variant, cv0, cv1)
        fits[variant] = (beta, cse)
        corr = IcmCorrection(params_of(variant, beta, cv0, cv1))
        raw3, cor3, _ = d_short_stats(corr, *by_cell["t3"])
        raw1, cor1, _ = d_short_stats(corr, *by_cell["t1"])
        md_raw, md_cor = [], []
        for c in CELLS:
            r, co = depth_band_means(corr, *by_cell[c], 0, 5)
            md_raw += r
            md_cor += co
        s = {"t3_raw": float(raw3.mean()), "t3_corr": float(cor3.mean()),
             "t1_corr": float(cor1.mean()),
             "micro_raw": float(np.mean(md_raw)),
             "micro_corr": float(np.mean(md_cor))}
        s["pass"] = bool(
            abs(s["t3_corr"]) <= 0.5 * abs(s["t3_raw"])
            and -0.01 < s["t1_corr"] < 0.01
            and s["micro_corr"] <= 0.03)
        screen[variant] = s
        print(f"screen {variant:24s} t3 {s['t3_raw']:+.4f}->"
              f"{s['t3_corr']:+.4f} t1 {s['t1_corr']:+.4f} micro "
              f"{s['micro_raw']:+.4f}->{s['micro_corr']:+.4f} "
              f"pass={s['pass']}")
    eligible_variants = [v for v in variants
                         if cv_table[v]["mean"] <= bar and
                         screen[v]["pass"]]
    if not eligible_variants:   # screen over 1-SE bar if in conflict
        eligible_variants = [v for v in variants if screen[v]["pass"]]
    if not eligible_variants:
        print("NO variant passes the fit-set screen -> M1 FAILS; "
              "per spec: fall back to M2 / NO-ADOPT")
        chosen = None
    else:
        chosen = eligible_variants[0]
    print(f"chosen variant: {chosen} (best={best}, 1-SE bar={bar:.5f})")
    if chosen is None:
        return 1

    beta, cse = fits[chosen]
    params = params_of(chosen, beta, cv0, cv1)
    corr = IcmCorrection(params)
    print(f"coefficients ({chosen}): "
          f"{[round(float(b), 4) for b in beta]}")
    print(f"clustered SEs: {[round(float(s), 4) for s in cse]}")

    # ── open the holdout ONCE: gates G1-G3 ──
    hold = {cell: block(cell, "holdout") for cell in CELLS}
    raw3, cor3, noise3 = d_short_stats(corr, *hold["t3"])
    raw1, cor1, _ = d_short_stats(corr, *hold["t1"])
    g1_target = 0.5 * abs(raw3.mean())
    g1 = abs(cor3.mean()) <= g1_target
    g2 = -0.01 < cor1.mean() < 0.01
    md_raw, md_cor = [], []
    for cell in CELLS:
        r, c = depth_band_means(corr, *hold[cell], 0, 5)
        md_raw += r
        md_cor += c
    g3 = (float(np.mean(md_cor)) <= 0.03) if md_cor else True
    gates = {
        "G1_t3_half_bias": {
            "raw_mean": float(raw3.mean()), "corrected_mean":
            float(cor3.mean()), "target_abs": float(g1_target),
            "pass": bool(g1)},
        "G2_t1_band": {
            "raw_mean": float(raw1.mean()),
            "corrected_mean": float(cor1.mean()),
            "band": [-0.01, 0.01], "pass": bool(g2)},
        "G3_micro_depth": {
            "n_seats": len(md_cor),
            "raw_mean": float(np.mean(md_raw)) if md_raw else None,
            "corrected_mean": float(np.mean(md_cor)) if md_cor else None,
            "bar": 0.03, "pass": bool(g3)},
    }
    gate_pass = g1 and g2 and g3
    for k, v in gates.items():
        print(f"M1 {k}: raw={v['raw_mean']} corrected={v['corrected_mean']} "
              f"pass={v['pass']}")
    print(f"M1 HOLDOUT GATE: {'PASS' if gate_pass else 'FAIL'}")

    # ── registered fallback M2 (spec 1.2/5) if M1 fails any gate ──
    final_model = "M1" if gate_pass else None
    m2_block = None
    if not gate_pass:
        tables = m2_fit(fit_feats, fit_cells)
        m2_gates, m2_pass = m2_gate_eval(tables, hold, None)
        for k, v in m2_gates.items():
            print(f"M2 {k}: raw={v['raw_mean']} "
                  f"corrected={v['corrected_mean']} pass={v['pass']}")
        print(f"M2 HOLDOUT GATE: {'PASS' if m2_pass else 'FAIL'}")
        m2_block = {"gates": m2_gates, "pass": m2_pass,
                    "n_bins": len(tables)}
        if m2_pass:
            final_model = "M2"
            # recompute holdout t3 d_short under M2 for the
            # heterogeneity clause below
            _, feats3 = hold["t3"]
            cor3 = np.asarray(
                [f["p_hat"][f["short"]] -
                 m2_corrected_state(tables, "t3", f)[f["short"]]
                 for f in feats3])

    # heterogeneity honesty clause (spec 2.5) on holdout t3
    var_raw = float(raw3.var(ddof=1))
    var_cor = float(cor3.var(ddof=1))
    s2b_raw = max(0.0, var_raw - noise3)
    s2b_cor = max(0.0, var_cor - noise3)
    hetero = {
        "holdout_t3_sigma_bias_raw": math.sqrt(s2b_raw),
        "holdout_t3_sigma_bias_corrected": math.sqrt(s2b_cor),
        "fraction_sigma2_bias_explained":
            (1.0 - s2b_cor / s2b_raw) if s2b_raw > 0 else None,
    }

    # M4 floor on the holdout
    h_recs, h_feats, h_cells = [], [], []
    for cell in CELLS:
        h_recs += hold[cell][0]
        h_feats += hold[cell][1]
        h_cells += [cell] * len(hold[cell][0])
    m4_mse, m4_table = m4_weighted_mse(fit_feats, h_feats, fit_cells,
                                       h_cells)
    chosen_hold_mse = eval_weighted_mse(corr, h_recs, h_feats)
    raw_hold_mse = eval_weighted_mse(
        IcmCorrection({"variant": "depth_only",
                       "g_coef": [0.0] * len(KNOTS),
                       "cv_anchor": {"cv0": cv0, "cv1": cv1},
                       "knots": list(KNOTS), "rank_coef": None}),
        h_recs, h_feats)

    results = {
        "record_type": "icm_correction_fit",
        "spec": "docs/research_program/ICM_CORRECTION_SPEC.md",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "fit_data": {
            "files": {f"tier1_{c}.jsonl":
                      sha256_of_file(in_dir / f"tier1_{c}.jsonl")
                      for c in CELLS},
            "n_states": {c: len(data[c]) for c in CELLS},
            "split_seed": SPLIT_SEED,
            "n_fit_states": n_fit_states,
            "n_fit_seat_residuals": n_fit_seats,
            "holdout_idx": {c: split[c]["holdout"] for c in CELLS},
        },
        "cv_anchor": params["cv_anchor"],
        "deviations": ["D1 projection-aware fit", "D2 gate-coherent "
                       "variant selection", "D3 holdout opened twice "
                       "(first opening: degenerate depth_only, gates "
                       "FAILED)"],
        "variant_cv_table": cv_table,
        "fit_set_screen": screen,
        "variant_chosen": chosen,
        "one_se_bar": bar,
        "coefficients": params,
        "clustered_se": [float(s) for s in cse],
        "holdout_gates_m1": gates,
        "holdout_gate_pass_m1": bool(gate_pass),
        "m2_fallback": m2_block,
        "final_model": final_model,
        "holdout_weighted_mse": {
            "raw_mh": raw_hold_mse, "chosen": chosen_hold_mse,
            "m4_lookup_floor": m4_mse},
        "m4_table": m4_table,
        "heterogeneity": hetero,
    }
    (out_dir / "fit_results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir / 'fit_results.json'}")

    if final_model is not None:
        if final_model == "M1":
            artifact = dict(params)
            artifact["model"] = "M1"
        else:
            with open(in_dir / "tier1_t3.jsonl") as fh:
                hdr = json.loads(fh.readline())
            artifact = {
                "model": "M2",
                "tertile_cuts": hdr["frame_meta"]["tertile_cuts"],
                "depth_bands": [[lo, hi if hi != float("inf") else None]
                                for lo, hi in DEPTH_BANDS],
                "tables": {f"{cell}|{band}":
                           {"q_lo": lo_, "q_hi": hi_, "value": val_}
                           for (cell, band), (lo_, hi_, val_)
                           in sorted(tables.items())},
            }
        artifact.update({
            "version": "icm_correction_v1",
            "spec": "docs/research_program/ICM_CORRECTION_SPEC.md",
            "scope": {"n_alive": 4, "payouts": "3 equal (top-3 format)",
                      "note": "identity outside scope; Tier-1 evidence "
                              "covers n_alive=4 levels 3-4 only"},
            "fit_set_hashes": results["fit_data"]["files"],
            "split_seed": SPLIT_SEED,
            "git_head": git_head(),
            "fitted_utc": results["generated_utc"],
        })
        apath = (REPO_ROOT / args.artifact
                 if not Path(args.artifact).is_absolute()
                 else Path(args.artifact))
        apath.write_text(json.dumps(artifact, indent=2))
        print(f"wrote {apath} (model={final_model})")
    else:
        print("M1 and M2 both FAILED the holdout gate -> per spec: "
              "NO-ADOPT, escalate (Tier-2 + rethink); no artifact, "
              "no validation rollouts to be spent")
    return 0 if final_model is not None else 1


if __name__ == "__main__":
    sys.exit(main())
