"""C3 checkpoint dashboard — read-only observability for a live retrain.

Watches a training run dir (local, or remote over scp) and, on each new
ckpt_iter_NNNN.pt, appends one row to a dashboard CSV with:

  - iter, adv_loss, strat_loss        (losses parsed from the train log)
  - premium-fold rates: AA/KK fold% at 60bb and 10bb (sample-mode probe,
    same machinery as monitor_k200_convergence.behavioral_probe)
  - ANCHOR EVALS: paired-CRN matches (default 5000 pairs = 10000 hands)
    vs two FIXED anchors:
      (a) the deployed ckpt_iter_1500 (sha b79e82dd..., refused on mismatch)
      (b) the run's own iter-100 snapshot (blank until it exists)
    reported as margin +/- stderr per anchor (positive = candidate beats
    the anchor; ICM-equity units per paired hand).

Anchor seeds are FIXED across checkpoints, so every checkpoint plays the
exact same deal sequence vs each anchor — trajectory comparisons are CRN
variance-reduced.

DO NOT read adjacent-checkpoint deltas as signal: per-checkpoint noise
sigma is ~1-2.5 (DECISIONS.md, k200_real_ante iter-1500 stop discipline).
Trajectories only. The rendered table repeats this warning.

Outputs:
  - CSV (append-only, idempotent: processed iters are skipped on restart)
  - a plain-text table of the latest 10 rows, atomically refreshed, for
    `watch cat <table.txt>`

Read-only on checkpoints; never touches the trainer. Anchor evals run in
a fork() process pool sized to the available cores (cgroup-aware).

Local usage (sibling tmux window on the training box):
  tmux new-session -d -s c3_dash \
    "cd ~/pokerbot && source .venv/bin/activate && \
     PYTHONUNBUFFERED=1 python scripts/c3_dashboard.py \
       --run-dir runs/c3_retrain_v1 \
       --train-log runs/c3_retrain_v1_train.log \
       --table runs/c3_retrain_v1/dashboard.txt --watch 300"

Remote usage (this pod watching Contabo's live run over scp):
  PYTHONUNBUFFERED=1 python scripts/c3_dashboard.py \
    --run-dir quant@80.241.219.63:~/pokerbot/runs/c3_retrain_v1 \
    --train-log quant@80.241.219.63:~/pokerbot/runs/c3_retrain_v1_train.log \
    --cache-dir runs/c3_remote_cache --watch 600

Calibration / acceptance check (deployed ckpt vs itself must be ~0):
  python scripts/c3_dashboard.py --demo-calibration --pairs 2000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import pickle
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.nlhe.game_strings import TournamentStructure
from scripts.throwaway_query_real_ante import _RealAnteStructure
from scripts.monitor_k200_convergence import (
    behavioral_probe, parse_losses_at_iter)
from scripts.eval_6max_self_play import _load_solver, play_one_hand

DEPLOYED_CKPT = "runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt"
DEPLOYED_SHA = "b79e82dd0ce9e78e4eb666b7379df953dadbf2a6e026c6bd4b6eec695e9b1b11"
ABSTRACTION = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCTURE = "configs/ignition_double_up_6max_turbo.yaml"

# Fixed, disjoint seed families (also disjoint from the monitor's
# seed_base+1000 family and slope_h2h's 20260607).
SEED_BASE_DEPLOYED = 20260610
SEED_BASE_ITER100 = 20270610

CSV_FIELDS = [
    "iter", "timestamp", "adv_loss", "strat_loss",
    "AA_fold_60bb", "AA_fold_10bb", "KK_fold_60bb", "KK_fold_10bb",
    "premium_fold_60bb", "premium_fold_10bb",
    "vs_deployed_margin", "vs_deployed_sem", "vs_deployed_pairs",
    "vs_iter100_margin", "vs_iter100_sem", "vs_iter100_pairs",
]

_RE_ITER = re.compile(r"ckpt_iter_(\d+)\.pt$")


def cgroup_cores() -> int:
    """Dedicated cores from the cgroup CPU quota (v1 then v2), falling
    back to os.cpu_count(). On shared hosts nproc lies; the quota doesn't."""
    try:  # v1
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return max(1, q // p)
    except OSError:
        pass
    try:  # v2
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return max(1, int(parts[0]) // int(parts[1]))
    except OSError:
        pass
    return os.cpu_count() or 1


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Paired-CRN anchor eval, fork-parallel ───────────────────────────────
# Globals inherited by fork()ed pool workers (set in parent pre-Pool).
_G: dict = {}


def _eval_pair_range(span) -> tuple[list[float], int]:
    """Worker: play pairs [lo, hi) of candidate-vs-anchor, return margins.
    Identical pairing scheme to slope_h2h_k200.paired_crn_margins."""
    lo, hi, seed_base = span
    cand, anchor, structure = _G["cand"], _G["anchor"], _G["structure"]
    margins, n_failed = [], 0
    for i in range(lo, hi):
        seed = seed_base + i
        seats_A = [cand, anchor, cand, anchor, cand, anchor]
        seats_B = [anchor, cand, anchor, cand, anchor, cand]
        try:
            rng = random.Random(seed)
            res_A = play_one_hand(seats_A, structure, rng,
                                  num_paid=3, mode="sample")
            delta_A = sum(res_A["seat_to_equity_delta"][s] for s in (0, 2, 4))
            rng = random.Random(seed)
            res_B = play_one_hand(seats_B, structure, rng,
                                  num_paid=3, mode="sample")
            delta_B = sum(res_B["seat_to_equity_delta"][s] for s in (1, 3, 5))
        except Exception:
            n_failed += 1
            continue
        margins.append((delta_A + delta_B) / 2.0)
    return margins, n_failed


def anchor_eval(cand, anchor, structure, n_pairs, seed_base, workers,
                log=print):
    """Margin +/- sem of candidate vs anchor over n_pairs paired-CRN hands,
    fanned out over a fork pool. Positive margin = candidate beats anchor."""
    _G["cand"], _G["anchor"], _G["structure"] = cand, anchor, structure
    chunk = max(25, (n_pairs + workers * 4 - 1) // (workers * 4))
    spans = [(lo, min(lo + chunk, n_pairs), seed_base)
             for lo in range(0, n_pairs, chunk)]
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(_eval_pair_range, spans)
    margins = [m for ms, _ in results for m in ms]
    n_failed = sum(nf for _, nf in results)
    el = time.time() - t0
    n = len(margins)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(margins))
    sem = float(np.std(margins, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    log(f"    {n} pairs in {el:.0f}s ({n / el:.0f} pairs/s, "
        f"{workers} workers, {n_failed} failed)  "
        f"margin={mean:+.4f} +/- {sem:.4f}")
    return mean, sem, n


# ── Remote (scp) support ────────────────────────────────────────────────

def is_remote(spec: str) -> bool:
    return ":" in spec and not Path(spec.split(":", 1)[0]).exists()


def remote_list_ckpts(spec: str) -> list[str]:
    host, path = spec.split(":", 1)
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"ls {path}/ckpt_iter_*.pt"],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        return []
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def remote_pull(host_path: str, dest: Path) -> bool:
    r = subprocess.run(["scp", "-q", host_path, str(dest)], timeout=600)
    return r.returncode == 0


# ── CSV / table rendering ───────────────────────────────────────────────

def processed_iters(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {int(row["iter"]) for row in csv.DictReader(f)}


def append_row(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def fmt(v, spec="{:+.3f}") -> str:
    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return spec.format(float(v))


def render_table(csv_path: Path, table_path: Path) -> None:
    """Latest 10 rows as a fixed-width table, refreshed atomically so the
    operator can `watch cat <table_path>`."""
    if not csv_path.exists():
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["iter"]))
    last = rows[-10:]
    lines = [
        f"C3 DASHBOARD  (refreshed {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
        f"{len(rows)} checkpoints total, latest 10 shown)",
        "margins: paired-CRN ICM-equity/hand, positive = candidate beats anchor",
        "NOTE: per-checkpoint noise sigma ~1-2.5 — read TRAJECTORIES, never",
        "      adjacent-checkpoint deltas (DECISIONS.md iter-1500 stop rule).",
        "",
        f"{'iter':>5}  {'adv':>7} {'strat':>7}  "
        f"{'AAf60':>5} {'AAf10':>5} {'KKf60':>5} {'KKf10':>5}  "
        f"{'vs_deployed':>16}  {'vs_iter100':>16}",
    ]
    lines.append("-" * len(lines[-1]))
    for r in last:
        vd = (f"{fmt(r['vs_deployed_margin'])}+/-"
              f"{fmt(r['vs_deployed_sem'], '{:.3f}')}")
        v1 = (f"{fmt(r['vs_iter100_margin'])}+/-"
              f"{fmt(r['vs_iter100_sem'], '{:.3f}')}")
        lines.append(
            f"{r['iter']:>5}  {fmt(r['adv_loss'], '{:.4f}'):>7} "
            f"{fmt(r['strat_loss'], '{:.4f}'):>7}  "
            f"{fmt(r['AA_fold_60bb'], '{:.0f}'):>4}% "
            f"{fmt(r['AA_fold_10bb'], '{:.0f}'):>4}% "
            f"{fmt(r['KK_fold_60bb'], '{:.0f}'):>4}% "
            f"{fmt(r['KK_fold_10bb'], '{:.0f}'):>4}%  "
            f"{vd:>16}  {v1:>16}")
    tmp = table_path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(table_path)


# ── Per-checkpoint processing ───────────────────────────────────────────

def process_checkpoint(ckpt_path: Path, it: int, *, abstraction, structure,
                       deployed_solver, iter100_path: Path | None,
                       train_log: str | None, n_pairs: int, workers: int,
                       probe_seed: int = 20260601) -> dict:
    print(f"[iter {it}] loading candidate {ckpt_path.name}...", flush=True)
    cand = _load_solver(str(ckpt_path), abstraction, structure)
    cand.tournament_structure = structure

    behav = behavioral_probe(cand, structure, random.Random(probe_seed + it))
    adv_loss, strat_loss = (parse_losses_at_iter(train_log, it)
                            if train_log else (None, None))

    print(f"[iter {it}] anchor eval vs deployed ckpt_1500 "
          f"({n_pairs} pairs)...", flush=True)
    m_dep, sem_dep, n_dep = anchor_eval(
        cand, deployed_solver, structure, n_pairs, SEED_BASE_DEPLOYED, workers)

    m_100 = sem_100 = ""
    n_100 = 0
    if iter100_path is not None and iter100_path.exists():
        print(f"[iter {it}] anchor eval vs own iter-100 snapshot...",
              flush=True)
        anchor100 = _load_solver(str(iter100_path), abstraction, structure)
        anchor100.tournament_structure = structure
        m_100, sem_100, n_100 = anchor_eval(
            cand, anchor100, structure, n_pairs, SEED_BASE_ITER100, workers)

    return {
        "iter": it,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "adv_loss": "" if adv_loss is None else f"{adv_loss:.4f}",
        "strat_loss": "" if strat_loss is None else f"{strat_loss:.4f}",
        "AA_fold_60bb": behav["AA_60bb_fold"],
        "AA_fold_10bb": behav["AA_10bb_fold"],
        "KK_fold_60bb": behav["KK_60bb_fold"],
        "KK_fold_10bb": behav["KK_10bb_fold"],
        "premium_fold_60bb": (behav["AA_60bb_fold"] + behav["KK_60bb_fold"]) / 2,
        "premium_fold_10bb": (behav["AA_10bb_fold"] + behav["KK_10bb_fold"]) / 2,
        "vs_deployed_margin": f"{m_dep:.4f}",
        "vs_deployed_sem": f"{sem_dep:.4f}",
        "vs_deployed_pairs": n_dep,
        "vs_iter100_margin": m_100 if m_100 == "" else f"{m_100:.4f}",
        "vs_iter100_sem": sem_100 if sem_100 == "" else f"{sem_100:.4f}",
        "vs_iter100_pairs": n_100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="training run dir; local path or user@host:path")
    ap.add_argument("--train-log", default=None,
                    help="trainer log for loss parsing; local or remote")
    ap.add_argument("--cache-dir", default="runs/c3_dashboard_cache",
                    help="local cache for remote checkpoints")
    ap.add_argument("--deployed-ckpt", default=DEPLOYED_CKPT)
    ap.add_argument("--deployed-sha", default=DEPLOYED_SHA)
    ap.add_argument("--abstraction", default=ABSTRACTION)
    ap.add_argument("--structure", default=STRUCTURE)
    ap.add_argument("--pairs", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=None,
                    help="anchor-eval processes (default: cgroup cores - 1)")
    ap.add_argument("--csv", default=None,
                    help="dashboard CSV (default <run-dir-or-cache>/dashboard.csv)")
    ap.add_argument("--table", default=None,
                    help="rendered table txt (default alongside the CSV)")
    ap.add_argument("--watch", type=int, default=300,
                    help="poll seconds; 0 = single pass and exit")
    ap.add_argument("--demo-calibration", action="store_true",
                    help="play the deployed ckpt vs ITSELF and exit; the "
                         "margin must be ~0 (plumbing/scoring calibration)")
    args = ap.parse_args()

    workers = args.workers or max(1, cgroup_cores() - 1)
    print(f"workers={workers} (cgroup cores={cgroup_cores()})", flush=True)

    # Deployed anchor: refuse a wrong artifact outright.
    got = sha256_file(args.deployed_ckpt)
    if got != args.deployed_sha:
        sys.exit(f"FATAL: {args.deployed_ckpt} sha256={got[:12]}... does not "
                 f"match the deployed model {args.deployed_sha[:12]}...")
    print(f"deployed anchor sha256 OK ({got[:12]}...)", flush=True)

    with open(args.abstraction, "rb") as f:
        abstraction = pickle.load(f)
    structure = _RealAnteStructure(TournamentStructure.from_yaml(args.structure))
    deployed = _load_solver(args.deployed_ckpt, abstraction, structure)
    deployed.tournament_structure = structure

    if args.demo_calibration:
        print(f"CALIBRATION: deployed vs itself, {args.pairs} pairs "
              f"(expected margin ~0)...", flush=True)
        m, sem, n = anchor_eval(deployed, deployed, structure,
                                args.pairs, SEED_BASE_DEPLOYED, workers)
        ok = abs(m) < max(0.01, 3 * (sem if not np.isnan(sem) else 0))
        print(f"CALIBRATION RESULT: margin={m:+.5f} +/- {sem:.5f} "
              f"over {n} pairs -> {'PASS (~0)' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    if not args.run_dir:
        sys.exit("--run-dir is required (or use --demo-calibration)")

    remote = is_remote(args.run_dir)
    cache = Path(args.cache_dir)
    local_dir = cache if remote else Path(args.run_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else local_dir / "dashboard.csv"
    table_path = Path(args.table) if args.table else local_dir / "dashboard.txt"
    train_log_local = args.train_log
    if args.train_log and is_remote(args.train_log):
        train_log_local = str(cache / "train.log")

    print(f"run-dir={args.run_dir} (remote={remote})  csv={csv_path}  "
          f"table={table_path}", flush=True)

    while True:
        if remote:
            for hp in remote_list_ckpts(args.run_dir):
                name = Path(hp).name
                if not (cache / name).exists():
                    host = args.run_dir.split(":", 1)[0]
                    print(f"pulling {name} from {host}...", flush=True)
                    remote_pull(f"{host}:{hp}", cache / name)
            if args.train_log and is_remote(args.train_log):
                remote_pull(args.train_log, Path(train_log_local))

        done = processed_iters(csv_path)
        ckpts = sorted(
            ((int(_RE_ITER.search(p.name).group(1)), p)
             for p in local_dir.glob("ckpt_iter_*.pt")),
            key=lambda t: t[0])
        iter100 = next((p for i, p in ckpts if i == 100), None)
        new = [(i, p) for i, p in ckpts if i not in done]
        for it, p in new:
            row = process_checkpoint(
                p, it, abstraction=abstraction, structure=structure,
                deployed_solver=deployed, iter100_path=iter100,
                train_log=train_log_local, n_pairs=args.pairs,
                workers=workers)
            append_row(csv_path, row)
            render_table(csv_path, table_path)
            print(f"[iter {it}] row appended; table refreshed.", flush=True)
        if not new:
            render_table(csv_path, table_path)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
