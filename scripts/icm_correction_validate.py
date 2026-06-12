"""Fresh-state validation of ICM correction v1 (spec section 2).

Pre-registered in docs/research_program/ICM_CORRECTION_SPEC.md:
  - fresh states from the same frame (data/training_dist_v1.json.gz),
    same cell definitions/tertile cuts as Tier-1, EXCLUDING all 750
    Tier-1 tuples (fit + holdout). State-sampling seed 20260614
    (one random.Random(20260614) per cell over the sorted eligible
    pool, mirroring Tier-1's per-cell convention; logged).
  - cells: fresh t3 (N=150), fresh t1 (N=150), V3 = 100 states from the
    t2-union-t3 CV range filtered to shortest-stack depth < 5bb.
  - rollout CRN seed = sha256("2026|{cell}|val|{state_idx}|{m}")
    (implemented by passing cell_id "{cell}|val" to the c1 instrument's
    rollout_seed; disjoint from Tier-1's stream).
  - same instrument: scripts/icm_gap_probe.py rollout semantics,
    corrected ITM definition (D2), champion ckpt, M=100, caps/taints
    rules unchanged.
  - bars V1/V2/V3 frozen in spec 2.2; lambda-shrink retry only if V2
    fails alone.

Usage:
  python -m scripts.icm_correction_validate --phase sample   # dry: list
  python -m scripts.icm_correction_validate --phase run --cell t3 [--resume]
  python -m scripts.icm_correction_validate --phase report
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.icm_gap_probe import (DEFAULT_ABSTR, DEFAULT_CKPT,
                                   DEFAULT_DIST, DEFAULT_STRUCT, M_TIER1,
                                   N_SEATS, N_TIER1, PAYOUTS,
                                   _init_worker, _run_state_task,
                                   _struct_level, load_primary_frame,
                                   log, q_vector, run_header,
                                   sample_cell_states, t_ci)

VAL_SEED = 20260614
VAL_CELLS = {"t1": 150, "t3": 150, "v3": 100}
DEFAULT_OUT = "evals/c1_correction_20260612"
ARTIFACT = "data/icm_correction_v1.json"


def tier1_used_tuples(distinct):
    """The 750 Tier-1 tuples (fit + holdout) — both are excluded."""
    used = set()
    for cell in ("t1", "t2", "t3"):
        for s in sample_cell_states(distinct, cell, N_TIER1):
            used.add((tuple(s["stacks"]), s["level"], s["dealer"]))
    return used


def shortest_depth_bb(stacks, level):
    bl = _struct_level(level)
    alive = [s for s in stacks if s > 0]
    return min(alive) / bl.big_blind


def sample_validation_states(dist_path):
    distinct, meta = load_primary_frame(dist_path)
    used = tier1_used_tuples(distinct)
    out = {}
    pools = {
        "t1": [t for t in distinct["t1"] if t not in used],
        "t3": [t for t in distinct["t3"] if t not in used],
        "v3": [t for t in sorted(set(distinct["t2"]) | set(distinct["t3"]))
               if t not in used
               and shortest_depth_bb(t[0], t[1]) < 5.0],
    }
    for cell, n in VAL_CELLS.items():
        rng = random.Random(VAL_SEED)
        picks = rng.sample(pools[cell], min(n, len(pools[cell])))
        out[cell] = [{"stacks": list(st), "level": lv, "dealer": dl}
                     for (st, lv, dl) in picks]
    return out, meta, {k: len(v) for k, v in pools.items()}


def phase_run(args, out_dir):
    import multiprocessing as mp
    states_by_cell, meta, pool_sizes = sample_validation_states(args.dist)
    cell = args.cell
    states = states_by_cell[cell]
    header = run_header(args)
    header.update({
        "record_type": "run_header",
        "spec": "docs/research_program/ICM_CORRECTION_SPEC.md",
        "phase": f"validation_{cell}",
        "val_seed": VAL_SEED,
        "crn_stream": f"sha256('2026|{cell}|val|<state_idx>|<m>')",
        "frame_meta": meta,
        "eligible_pool_size": pool_sizes[cell],
        "n_states": len(states),
        "tier1_tuples_excluded": 750,
    })
    tasks = [{"cell": f"{cell}|val", "state_idx": i, "stacks": s["stacks"],
              "level": s["level"], "dealer": s["dealer"], "M": M_TIER1}
             for i, s in enumerate(states)]
    jsonl_path = out_dir / f"val_{cell}.jsonl"
    if args.resume and jsonl_path.exists():
        done_idx = set()
        with open(jsonl_path) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("record_type") == "run_header":
                    continue
                i = rec["state_idx"]
                assert rec["stacks"] == tasks[i]["stacks"], \
                    f"resume mismatch at state_idx {i}"
                done_idx.add(i)
        tasks = [t for t in tasks if t["state_idx"] not in done_idx]
        log(f"resume: {len(done_idx)} done, {len(tasks)} remaining")
    if args.limit:
        tasks = tasks[:args.limit]
    t0 = time.time()
    with open(jsonl_path, "a") as fh:
        fh.write(json.dumps(header) + "\n")
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(args.ckpt, args.abstraction,
                               args.structure)) as p:
            done = 0
            for row in p.imap_unordered(_run_state_task, tasks):
                done += 1
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if done % 10 == 0 or done == len(tasks):
                    el = time.time() - t0
                    log(f"val_{cell} {done}/{len(tasks)} states "
                        f"[{el:.0f}s, {el / done:.1f} s/state, "
                        f"eta {(len(tasks) - done) * el / done / 60:.0f}m]")
                if row["n_tainted"]:
                    log(f"  [TAINT] state {row['state_idx']}: "
                        f"{row['taints']}")
    log(f"val_{cell} complete -> {jsonl_path}")
    return 0


def load_cell_rows(out_dir, cell):
    rows, header = [], None
    with open(out_dir / f"val_{cell}.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("record_type") == "run_header":
                header = rec
            else:
                rows.append(rec)
    return header, rows


def analyze(rows, corr):
    """Per-state raw + corrected d_short and per-seat micro residuals."""
    import numpy as np
    raw_s, cor_s, noise = [], [], []
    micro_raw, micro_cor = [], []
    dropped = taints = 0
    for r in rows:
        taints += r["n_tainted"]
        if r["n_capped"] > 0.02 * r["M"]:
            dropped += 1
            continue
        st = r["stacks"]
        alive = [i for i in range(N_SEATS) if st[i] > 0]
        bl = _struct_level(r["level"])
        q = q_vector(st)
        p_corr = corr.corrected_itm_probs(st, PAYOUTS, eligible=alive,
                                          big_blind=bl.big_blind)
        m_eff = r["M_eff"]
        p_hat = {i: r["itm_counts"][i] / m_eff for i in alive}
        short = min(alive, key=lambda i: (st[i], i))
        raw_s.append(p_hat[short] - q[short])
        cor_s.append(p_hat[short] - p_corr[short])
        noise.append(p_hat[short] * (1 - p_hat[short]) / (m_eff - 1))
        for i in alive:
            if st[i] / bl.big_blind < 5.0:
                micro_raw.append(p_hat[i] - q[i])
                micro_cor.append(p_hat[i] - p_corr[i])
    import numpy as np
    raw = np.asarray(raw_s)
    cor = np.asarray(cor_s)
    rm, rse, rlo, rhi = t_ci(raw, 0.95)
    cm, cse, clo, chi = t_ci(cor, 0.95)
    var_noise = float(np.mean(noise))
    s2b_raw = max(0.0, float(raw.var(ddof=1)) - var_noise)
    s2b_cor = max(0.0, float(cor.var(ddof=1)) - var_noise)
    return {
        "n_states_scored": len(raw_s), "n_dropped_cap": dropped,
        "n_tainted_rollouts": taints,
        "raw_d_short": {"mean": rm, "se": rse, "ci95": [rlo, rhi]},
        "corrected_d_short": {"mean": cm, "se": cse, "ci95": [clo, chi]},
        "sigma_bias_raw": math.sqrt(s2b_raw),
        "sigma_bias_corrected": math.sqrt(s2b_cor),
        "fraction_sigma2_bias_explained":
            (1.0 - s2b_cor / s2b_raw) if s2b_raw > 0 else None,
        "micro_depth_seats": {
            "n": len(micro_raw),
            "raw_mean": float(np.mean(micro_raw)) if micro_raw else None,
            "corrected_mean": float(np.mean(micro_cor))
            if micro_cor else None,
        },
    }


def phase_report(args, out_dir):
    from src.nlhe.icm_correction import IcmCorrection
    corr = IcmCorrection.load(REPO_ROOT / ARTIFACT)
    cells, headers = {}, {}
    for cell in VAL_CELLS:
        headers[cell], rows = load_cell_rows(out_dir, cell)
        cells[cell] = analyze(rows, corr)
        log(f"{cell}: n={cells[cell]['n_states_scored']} raw="
            f"{cells[cell]['raw_d_short']['mean']:+.4f} corrected="
            f"{cells[cell]['corrected_d_short']['mean']:+.4f}")
    t3, t1, v3 = cells["t3"], cells["t1"], cells["v3"]
    # V1 (primary, t3)
    half = abs(t3["corrected_d_short"]["mean"]) <= \
        0.5 * abs(t3["raw_d_short"]["mean"])
    clo, chi = t3["corrected_d_short"]["ci95"]
    v1 = half and (clo <= 0.0 <= chi
                   or abs(t3["corrected_d_short"]["mean"]) <= 0.0206)
    # V2 (no-degradation, t1)
    lo1, hi1 = t1["corrected_d_short"]["ci95"]
    v2 = (-0.015 < lo1 and hi1 < 0.015) and \
        (abs(t1["corrected_d_short"]["mean"]) <=
         abs(t1["raw_d_short"]["mean"]) + 0.005)
    # V3 (micro-depth oversample cell)
    v3_mean = v3["micro_depth_seats"]["corrected_mean"]
    v3_pass = v3_mean is not None and v3_mean <= 0.03
    if v1 and v2 and v3_pass:
        verdict = "ADOPT (V1+V2+V3 pass)"
    elif v1 and v3_pass:
        verdict = ("V2 FAILS ALONE -> one pre-registered lambda-shrink "
                   "retry permitted on the same states")
    else:
        verdict = "NO-ADOPT (V1 or V3 fail) -> report, Tier-2 escalation"
    results = {
        "record_type": "icm_correction_validation",
        "spec": "docs/research_program/ICM_CORRECTION_SPEC.md",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": ARTIFACT,
        "model": corr.model,
        "headers": headers,
        "cells": cells,
        "bars": {
            "V1_t3": {"half_bias": bool(half),
                      "ci_contains_0_or_small": bool(
                          clo <= 0.0 <= chi or
                          abs(t3["corrected_d_short"]["mean"]) <= 0.0206),
                      "raw_replication_check_vs_+0.0412":
                          t3["raw_d_short"]["mean"],
                      "pass": bool(v1)},
            "V2_t1": {"ci95_in_band": bool(-0.015 < lo1 and hi1 < 0.015),
                      "no_degradation": bool(
                          abs(t1["corrected_d_short"]["mean"]) <=
                          abs(t1["raw_d_short"]["mean"]) + 0.005),
                      "pass": bool(v2)},
            "V3_micro": {"corrected_mean": v3_mean, "bar": 0.03,
                         "pass": bool(v3_pass)},
        },
        "verdict": verdict,
    }
    (out_dir / "validation_results.json").write_text(
        json.dumps(results, indent=2))
    log(f"wrote {out_dir / 'validation_results.json'}")
    log(f"VALIDATION VERDICT: {verdict}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True,
                    choices=["sample", "run", "report"])
    ap.add_argument("--cell", choices=list(VAL_CELLS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--abstraction", default=DEFAULT_ABSTR)
    ap.add_argument("--structure", default=DEFAULT_STRUCT)
    ap.add_argument("--dist", default=DEFAULT_DIST)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "sample":
        states, meta, pools = sample_validation_states(args.dist)
        for cell, st in states.items():
            log(f"{cell}: {len(st)} states (pool {pools[cell]})")
        return 0
    if args.phase == "run":
        if not args.cell:
            raise SystemExit("--cell required")
        return phase_run(args, out_dir)
    return phase_report(args, out_dir)


if __name__ == "__main__":
    sys.exit(main())
