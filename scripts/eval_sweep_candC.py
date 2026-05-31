"""Read-only eval sweep over existing Cand C checkpoints.

Produces two curves for runs/six_max_*_phase4f_dcfr_candC_k200:
  STEP 1 self-improvement (vs past self) -> self_improvement.jsonl
  STEP 2 lift vs the Shanky pool          -> shanky_lift.jsonl

Every eval is launched as a subprocess under `nice -n 15` so the live
training run (8 fork() workers) keeps priority. Evals run strictly
serially -- at most one extra process competes with training at a time.

Resumable: a (newer,older,kind) pair or a shanky iter already present in
the target jsonl is skipped, so the sweep can be re-run after interruption.

This is a driver only; it does not touch training and writes nothing into
the checkpoints dir. Rendering lives in scripts/render_learning_curves.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

RUN = "runs/six_max_20260530_034023_phase4f_dcfr_candC_k200"
ABST = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCT = "configs/ignition_double_up_6max_turbo.yaml"
HANDS = 600
NICE = "15"


def log(msg: str) -> None:
    print(f"[sweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_iters(run: str) -> list[int]:
    iters = []
    for p in glob.glob(os.path.join(run, "checkpoints", "ckpt_iter_*.pt")):
        m = re.search(r"ckpt_iter_(\d+)\.pt$", p)
        if m:
            iters.append(int(m.group(1)))
    return sorted(set(iters))


def ckpt(run: str, it: int) -> str:
    return os.path.join(run, "checkpoints", f"ckpt_iter_{it:04d}.pt")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_subproc(cmd: list[str]) -> int:
    """Run a command under nice -n NICE, streaming output. Returns rc."""
    full = ["nice", "-n", NICE] + cmd
    log("$ " + " ".join(full))
    proc = subprocess.run(full)
    return proc.returncode


# ---------------- STEP 1 ----------------

def run_eval_pool_pair(newer: int, older: int) -> dict | None:
    """Run eval_pool challenger=newer vs opponent=older. Return results[0]."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    cmd = [
        sys.executable, "-m", "scripts.eval_pool",
        "--challenger", f"iter{newer}={ckpt(RUN, newer)}",
        "--opponents", f"iter{older}={ckpt(RUN, older)}",
        "--abstraction", ABST, "--structure", STRUCT,
        "--hands", str(HANDS), "--log-every", "300",
        "--output", out,
    ]
    rc = run_subproc(cmd)
    if rc != 0:
        log(f"  eval_pool {newer}v{older} FAILED rc={rc}")
        return None
    payload = json.loads(open(out).read())
    os.unlink(out)
    return payload["results"][0]


def step1(jsonl_path: str) -> None:
    iters = list_iters(RUN)
    log(f"STEP 1 checkpoints: {iters}")
    existing = read_jsonl(jsonl_path)
    seen = {(r["newer_iter"], r["older_iter"], r["kind"]) for r in existing}

    # Build requested (newer, older, kind) list.
    requested: list[tuple[int, int, str]] = []
    for i in range(1, len(iters)):
        requested.append((iters[i], iters[i - 1], "step"))
    base = iters[0]
    for i in range(1, len(iters)):  # cumulative vs base, including base+1
        requested.append((iters[i], base, "cumulative"))

    # Cache eval_pool results by (newer, older) so a pair shared by step and
    # cumulative (e.g. 400v200) is only computed once.
    cache: dict[tuple[int, int], dict] = {}
    for newer, older, kind in requested:
        if (newer, older, kind) in seen:
            log(f"  skip {newer}v{older} [{kind}] (already recorded)")
            continue
        key = (newer, older)
        if key not in cache:
            res = run_eval_pool_pair(newer, older)
            if res is None:
                continue
            cache[key] = res
        res = cache[key]
        rec = {
            "newer_iter": newer,
            "older_iter": older,
            "kind": kind,
            "diff": res["diff"],
            "stderr": res["stderr"],
            "sigma": res["sigma"],
            "n_hands": res["n_hands"],
        }
        append_jsonl(jsonl_path, rec)
        log(f"  recorded {newer}v{older} [{kind}] diff={res['diff']:+.4f} "
            f"sigma={res['sigma']:.1f}")


# ---------------- STEP 2 ----------------

def run_eval_shanky(it: int) -> dict | None:
    out = f"/tmp/shanky_iter{it}.json"
    cmd = [
        sys.executable, "-m", "scripts.eval_shanky_vs_dcfr",
        "--challenger-ckpt", ckpt(RUN, it),
        "--challenger-name", f"candC-iter{it}",
        "--shanky-dir", "data/shanky_profiles",
        "--abstraction", ABST, "--structure", STRUCT,
        "--hands", str(HANDS), "--log-every", "300",
        "--output", out,
    ]
    rc = run_subproc(cmd)
    if rc != 0:
        log(f"  eval_shanky iter{it} FAILED rc={rc}")
        return None
    return json.loads(open(out).read())


def aggregate_shanky(payload: dict) -> dict:
    """Mean lift across all profiles + sigma of that mean.

    Each profile reports diff (per-seat ICM, challenger - opponent) and its
    own stderr. The mean lift is the simple average of per-profile diffs;
    its standard error combines the per-profile stderrs in quadrature
    (profiles are independent matchups), so
        se_mean = sqrt(sum(stderr_i^2)) / N
        sigma   = |mean_diff| / se_mean
    n_hands is the total hands played across all matchups.
    """
    results = payload["results"]
    n = len(results)
    diffs = [r["diff"] for r in results]
    mean_diff = sum(diffs) / n if n else float("nan")
    se_mean = math.sqrt(sum(r["stderr"] ** 2 for r in results)) / n if n else 0.0
    sigma = abs(mean_diff) / se_mean if se_mean > 0 else float("nan")
    total_hands = sum(r["n_hands"] for r in results)
    return {
        "mean_lift_vs_shanky": mean_diff,
        "sigma": sigma,
        "n_hands": total_hands,
        "n_profiles": n,
        "per_profile": [
            {"opponent": r["opponent"], "diff": r["diff"],
             "stderr": r["stderr"], "sigma": r["sigma"]}
            for r in results
        ],
    }


def step2(jsonl_path: str) -> None:
    iters = list_iters(RUN)
    log(f"STEP 2 checkpoints: {iters}")
    existing = read_jsonl(jsonl_path)
    seen = {r["iter"] for r in existing}
    for it in iters:
        if it in seen:
            log(f"  skip shanky iter{it} (already recorded)")
            continue
        payload = run_eval_shanky(it)
        if payload is None:
            continue
        agg = aggregate_shanky(payload)
        rec = {"iter": it, **agg}
        append_jsonl(jsonl_path, rec)
        log(f"  recorded shanky iter{it} mean_lift={agg['mean_lift_vs_shanky']:+.4f} "
            f"sigma={agg['sigma']:.1f} (n_profiles={agg['n_profiles']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="1,2", help="comma list of steps to run")
    args = ap.parse_args()
    steps = {s.strip() for s in args.steps.split(",")}

    self_path = os.path.join(RUN, "self_improvement.jsonl")
    shanky_path = os.path.join(RUN, "shanky_lift.jsonl")

    t0 = time.time()
    if "1" in steps:
        log("=== STEP 1: self-improvement (eval_pool) ===")
        step1(self_path)
    if "2" in steps:
        log("=== STEP 2: shanky-lift (eval_shanky_vs_dcfr) ===")
        step2(shanky_path)
    log(f"=== sweep done in {(time.time() - t0) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
