"""ICM gap probe — Tier 0 + Tier 1 of the pre-registered c1 design.

Executes docs/research_program/ICM_GAP_STUDY_DESIGN.md exactly:
  - Tier 0: 50-rollout rate benchmark (gate) + C0 negative control
    (symmetric start, M=2000; seat-averaged error != 0 => INSTRUMENT
    FAILURE, halt).
  - Tier 1: 3 primary cells = dispersion (CV) tertiles of
    (n_alive=4, levels 3-4), N=250 states x M=100 rollouts, champion in
    all 6 seats, mode="sample", hands_per_level=5, max_hands=200.
  - Estimator: per-state paired difference d_short(s) = p_hat - q for
    the shortest alive stack; t-interval over states; Bonferroni 0.05/3
    on the primary cells; TOST 90% CIs for equivalence; noise/bias
    variance decomposition; BCa bootstrap robustness check.

Implementation choices left open by the design (documented here and in
TIER1_REPORT.txt):
  1. Script is named icm_gap_probe.py (task mandate) instead of the
     design's placeholder name icm_gap_study.py. Same role.
  2. play_sng_game returns only the hero seat's outcome; the study needs
     all six survival flags. Under the new-files-only rule we mirror its
     game loop verbatim here (importing _blind_guard / play_one_hand_sng
     from scripts.sng_baseline), preserving the rng draw schedule
     including the unconditional dealer randrange. The stage_acc
     icm_equity calls are skipped: they never touch the rng stream.
  3. "Stable hash" rollout seed = first 8 bytes of
     sha256("2026|{cell_id}|{state_idx}|{m}") as a big-endian int.
  4. CV = population std (ddof=0) of alive-seat chip fractions divided
     by their mean. Tertile cuts are recomputed exactly from the
     artifact within the (4, levels 3-4) cell via numpy quantile
     (linear interpolation) over RECORDS (the design's table values are
     rounded versions of the same computation).
  5. Capped/tainted rollouts are excluded from p_hat (M_eff = M minus
     exclusions); a state with >2% capped rollouts is dropped and
     logged, with NO replacement state drawn (effective N shrinks).
  6. Depth-in-BB secondary bands: [0,5), [5,10), [10,20), [20,inf),
     depth = stack / raw big_blind at the state's level. Exploratory.
  7. C0 instrument test scalar: per-rollout e_m = (sum_i Y_im)/6 - 0.5
     (identically 0 unless the harness caps, taints, or terminates
     below 3 alive); t-CI over rollouts; CI excluding 0 => HALT.

LOGGED DEVIATION (2026-06-12, before any Tier-1 inference — see
DEVIATION_LOG.txt in the output dir): the design's outcome definition
"Y=1 iff seat has chips when the loop breaks at n_alive<=3" FAILED its
own C0 negative control (seat-averaged error -0.0101, t ~ -10.8; cause:
115/2000 games end with <3 chip-holders via simultaneous busts, so the
raw definition pays fewer than 3 seats while the real format always
pays exactly 3). Per the design's instrument-failure branch (halt +
debug before inference), Y is corrected to "seat finishes in the top
3" with simultaneous busts ranked by hand-start stack (bigger stack
finishes higher; ties -> lower seat index finishes worse), the standard
tournament rule and the rule MH ICM itself assumes. Both definitions
are recorded per state; the corrected one is primary, the raw one is
reported as a sensitivity line.

Usage (phases run in design order; each writes into --out-dir):
  python scripts/icm_gap_probe.py --phase bench
  python scripts/icm_gap_probe.py --phase c0
  python scripts/icm_gap_probe.py --phase tier1 --cell t1 [--n 250]
  python scripts/icm_gap_probe.py --phase report
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MASTER_SEED = 2026
DEFAULT_CKPT = ("runs/k200_real_ante_20260605_225847_PRESERVED/"
                "ckpt_iter_1500.pt")
DEFAULT_ABSTR = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
DEFAULT_STRUCT = "configs/ignition_double_up_6max_turbo.yaml"
DEFAULT_DIST = "data/training_dist_v1.json.gz"
DEFAULT_OUT = "evals/c1_icm_gap_20260612"
N_SEATS = 6
PAYOUTS = [2.0, 2.0, 2.0]
HANDS_PER_LEVEL = 5
MAX_HANDS = 200
MODE = "sample"
M_TIER1 = 100
N_TIER1 = 250
M_C0 = 2000
PRIMARY_CELLS = ["t1", "t2", "t3"]
DEPTH_BANDS = [(0, 5), (5, 10), (10, 20), (20, float("inf"))]


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')}  {msg}", flush=True)


def sha256_of_file(path: str) -> str:
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


def rollout_seed(cell_id: str, state_idx: int, m: int) -> int:
    raw = f"{MASTER_SEED}|{cell_id}|{state_idx}|{m}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


# ── Rollout: mirror of sng_baseline.play_sng_game returning all seats ──

def rollout_survivors(policy, structure, stacks0, level0, dealer0, seed):
    """One champion-self-play continuation. Mirrors play_sng_game's loop
    (same rng schedule) but returns per-seat survival flags."""
    from scripts.sng_baseline import _blind_guard, play_one_hand_sng

    rng = random.Random(seed)
    seat_to_policy = [policy] * N_SEATS
    stacks = list(stacks0)
    max_level = max(bl.level for bl in structure.blind_schedule)
    level = min(level0, max_level)
    dealer = rng.randrange(N_SEATS)        # unconditional draw, as in seam
    dealer = int(dealer0)
    guard = 0
    while stacks[dealer] == 0 and guard < N_SEATS:
        dealer = (dealer + 1) % N_SEATS
        guard += 1
    hands_played = 0
    hands_in_level = 0
    capped = False
    tainted = False
    exception = None
    # Elimination order for the corrected ITM definition: chronological,
    # within-iteration simultaneous busts appended ascending by
    # (hand-start stack, seat) so a LATER index = a BETTER finish.
    elim_order: list[int] = []

    while True:
        n_alive = sum(1 for s in stacks if s > 0)
        if n_alive <= 3:
            break
        if hands_played >= MAX_HANDS:
            capped = True
            break
        if hands_in_level >= HANDS_PER_LEVEL:
            level = min(level + 1, max_level)
            hands_in_level = 0
        pre_hand = list(stacks)
        stacks, dealer = _blind_guard(structure, stacks, level, dealer)
        if dealer is None or sum(1 for s in stacks if s > 0) <= 3:
            busted = [i for i in range(N_SEATS)
                      if pre_hand[i] > 0 and stacks[i] == 0]
            elim_order.extend(sorted(busted, key=lambda i: (pre_hand[i], i)))
            break
        try:
            stacks = play_one_hand_sng(seat_to_policy, structure, stacks,
                                       level, dealer, rng, mode=MODE)
        except Exception as e:  # taint: logged + excluded, never scored
            tainted = True
            exception = (f"hand={hands_played + 1} "
                         f"{type(e).__name__}: {str(e)[:120]}")
            break
        busted = [i for i in range(N_SEATS)
                  if pre_hand[i] > 0 and stacks[i] == 0]
        elim_order.extend(sorted(busted, key=lambda i: (pre_hand[i], i)))
        hands_played += 1
        hands_in_level += 1
        dealer = (dealer + 1) % N_SEATS
        guard = 0
        while stacks[dealer] == 0 and guard < N_SEATS:
            dealer = (dealer + 1) % N_SEATS
            guard += 1

    survived = [1 if s > 0 else 0 for s in stacks]
    n_alive_end = sum(survived)
    # Corrected ITM (logged deviation): exactly the top-3 finishers.
    itm = list(survived)
    fill = 3 - n_alive_end
    for seat in reversed(elim_order):
        if fill <= 0:
            break
        itm[seat] = 1
        fill -= 1
    return {
        "survived": survived,
        "itm": itm,
        "capped": capped,
        "tainted": tainted,
        "exception": exception,
        "hands": hands_played,
        "n_alive_end": n_alive_end,
    }


# ── Worker pool plumbing ───────────────────────────────────────────────

_POLICY = None
_STRUCTURE = None


def _init_worker(ckpt_path, abstr_path, struct_path):
    global _POLICY, _STRUCTURE
    import torch
    torch.set_num_threads(1)
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from scripts.eval_pool import CheckpointPolicy
    from scripts.throwaway_query_real_ante import _RealAnteStructure
    _STRUCTURE = _RealAnteStructure(TournamentStructure.from_yaml(struct_path))
    abstr = Abstraction.load(abstr_path)
    _POLICY = CheckpointPolicy(name="champ", ckpt_path=ckpt_path,
                               abstraction=abstr, structure=_STRUCTURE)


def _run_state_task(task):
    """M rollouts of one state; returns the per-state JSONL row."""
    cell, sidx = task["cell"], task["state_idx"]
    stacks, level, dealer = task["stacks"], task["level"], task["dealer"]
    M = task["M"]
    counts = [0] * N_SEATS
    itm_counts = [0] * N_SEATS
    n_capped = n_tainted = 0
    hands_sum = 0
    sub3 = 0
    taints = []
    t0 = time.time()
    for m in range(M):
        r = rollout_survivors(_POLICY, _STRUCTURE, stacks, level, dealer,
                              rollout_seed(cell, sidx, m))
        if r["tainted"]:
            n_tainted += 1
            taints.append({"m": m, "detail": r["exception"]})
            continue
        if r["capped"]:
            n_capped += 1
            continue
        for i in range(N_SEATS):
            counts[i] += r["survived"][i]
            itm_counts[i] += r["itm"][i]
        hands_sum += r["hands"]
        if r["n_alive_end"] < 3:
            sub3 += 1
    m_eff = M - n_capped - n_tainted
    return {
        "cell": cell, "state_idx": sidx, "stacks": stacks, "level": level,
        "dealer": dealer, "total_chips": sum(stacks),
        "survive_counts": counts, "itm_counts": itm_counts,
        "M": M, "M_eff": m_eff,
        "n_capped": n_capped, "n_tainted": n_tainted, "taints": taints,
        "n_sub3_terminals": sub3,
        "mean_hands": hands_sum / m_eff if m_eff else float("nan"),
        "elapsed_s": round(time.time() - t0, 2),
    }


# ── State frame: cells, CV, q, positions ───────────────────────────────

def cv_of(stacks):
    alive = [s for s in stacks if s > 0]
    tot = float(sum(alive))
    fr = [s / tot for s in alive]
    mu = sum(fr) / len(fr)
    var = sum((f - mu) ** 2 for f in fr) / len(fr)   # population (ddof=0)
    return math.sqrt(var) / mu


def q_vector(stacks):
    from src.nlhe.icm import icm_equity
    alive = [i for i in range(N_SEATS) if stacks[i] > 0]
    eq = icm_equity(stacks, PAYOUTS, eligible=alive)
    return [e / 2.0 for e in eq]   # probability units


def rotate_dealer(stacks, dealer):
    d = int(dealer)
    guard = 0
    while stacks[d] == 0 and guard < N_SEATS:
        d = (d + 1) % N_SEATS
        guard += 1
    return d


def seat_positions(stacks, dealer):
    """Hand-start position label per alive seat: BTN/SB/BB/OTH."""
    d = rotate_dealer(stacks, dealer)
    alive = [i for i in range(N_SEATS) if stacks[i] > 0]
    k = alive.index(d)
    order = alive[k:] + alive[:k]
    labels = {}
    for j, seat in enumerate(order):
        labels[seat] = ("BTN", "SB", "BB")[j] if j < 3 else "OTH"
    return labels


def load_primary_frame(dist_path):
    """Records of the (n_alive=4, levels 3-4) cell + recomputed tertile
    cuts + sorted distinct tuples per tertile."""
    import numpy as np
    with gzip.open(dist_path, "rt") as f:
        art = json.load(f)
    recs = [r for r in art["hand_starts"]
            if r["n_alive"] == 4 and r["level"] in (3, 4)]
    cvs = np.array([cv_of(r["stacks"]) for r in recs])
    t1_cut, t2_cut = np.quantile(cvs, [1 / 3, 2 / 3])
    cells = {"t1": set(), "t2": set(), "t3": set()}
    for r, c in zip(recs, cvs):
        key = (tuple(r["stacks"]), r["level"], r["dealer"])
        if c < t1_cut:
            cells["t1"].add(key)
        elif c < t2_cut:
            cells["t2"].add(key)
        else:
            cells["t3"].add(key)
    distinct = {k: sorted(v) for k, v in cells.items()}
    meta = {"n_records": len(recs),
            "tertile_cuts": [float(t1_cut), float(t2_cut)],
            "distinct_tuples": {k: len(v) for k, v in distinct.items()}}
    return distinct, meta


def sample_cell_states(distinct, cell, n):
    pool = distinct[cell]
    rng = random.Random(MASTER_SEED)
    n_eff = min(n, len(pool))
    picks = rng.sample(pool, n_eff)
    return [{"stacks": list(st), "level": lv, "dealer": dl}
            for (st, lv, dl) in picks]


# ── Phases ─────────────────────────────────────────────────────────────

def run_header(args):
    return {
        "record_type": "run_header",
        "design": "docs/research_program/ICM_GAP_STUDY_DESIGN.md",
        "phase": args.phase,
        "ckpt_path": str(Path(args.ckpt).resolve()),
        "ckpt_sha256": sha256_of_file(args.ckpt),
        "abstraction_sha256": sha256_of_file(args.abstraction),
        "dist_sha256": sha256_of_file(args.dist),
        "master_seed": MASTER_SEED,
        "git_head": git_head(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }


def phase_bench(args, out_dir):
    """Tier-0 gate: 1 worker x 50 rollouts from a bubble start and a
    6-alive start. Bracket [0.15, 0.8] s/rollout (design 5.1)."""
    _init_worker(args.ckpt, args.abstraction, args.structure)
    distinct, meta = load_primary_frame(args.dist)
    bub = sample_cell_states(distinct, "t2", 1)[0]   # modal-dispersion cell
    starts = {
        "bubble_4alive_l34": ("BENCH4", bub["stacks"], bub["level"],
                              bub["dealer"]),
        "sixalive_l1_full": ("BENCH6", [1500] * N_SEATS, 1, 0),
    }
    out = {"record_type": "rate_benchmark", "n_rollouts": 50,
           "frame_meta": meta, "bench_states": {}}
    for name, (cid, st, lv, dl) in starts.items():
        t0 = time.time()
        hands = 0
        for m in range(50):
            r = rollout_survivors(_POLICY, _STRUCTURE, st, lv, dl,
                                  rollout_seed(cid, 0, m))
            hands += r["hands"]
        dt = time.time() - t0
        out["bench_states"][name] = {
            "stacks": st, "level": lv, "dealer": dl,
            "s_per_rollout": dt / 50, "mean_hands": hands / 50,
            "elapsed_s": dt}
        log(f"bench {name}: {dt / 50:.3f} s/rollout, "
            f"{hands / 50:.1f} hands/rollout")
    rate = out["bench_states"]["bubble_4alive_l34"]["s_per_rollout"]
    tier1_core_h = 3 * N_TIER1 * M_TIER1 * rate / 3600
    out["bracket"] = [0.15, 0.8]
    out["in_bracket"] = bool(0.15 <= rate <= 0.8)
    out["below_bracket"] = bool(rate < 0.15)
    out["tier1_core_h_proj"] = tier1_core_h
    out["tier1_wall_h_proj_4cores"] = tier1_core_h / 4
    (out_dir / "rate_benchmark.json").write_text(json.dumps(out, indent=2))
    log(f"GATE: bubble rate {rate:.3f} s/rollout, bracket {out['bracket']}, "
        f"in_bracket={out['in_bracket']}; Tier-1 projected "
        f"{tier1_core_h:.1f} core-h ({tier1_core_h / 4:.1f} h on 4 cores)")
    return 0


def phase_c0(args, out_dir):
    """C0 negative control: symmetric start, M=2000 full games."""
    import multiprocessing as mp
    header = run_header(args)
    chunk = 50
    tasks = []
    for k, lo in enumerate(range(0, M_C0, chunk)):
        tasks.append({"cell": "C0", "state_idx": 0, "m_lo": lo,
                      "m_hi": min(lo + chunk, M_C0)})
    counts = [0] * N_SEATS
    itm_counts = [0] * N_SEATS
    surv_hist = {}
    itm_hist = {}
    n_capped = n_tainted = 0
    taints = []
    hands_sum = 0
    m_eff = 0
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.ckpt, args.abstraction, args.structure)) as p:
        done = 0
        for res in p.imap_unordered(_c0_chunk, tasks):
            done += 1
            n_capped += res["n_capped"]
            n_tainted += res["n_tainted"]
            taints.extend(res["taints"])
            hands_sum += res["hands_sum"]
            m_eff += res["m_eff"]
            for i in range(N_SEATS):
                counts[i] += res["counts"][i]
                itm_counts[i] += res["itm_counts"][i]
            for k, v in res["surv_hist"].items():
                surv_hist[k] = surv_hist.get(k, 0) + v
            for k, v in res["itm_hist"].items():
                itm_hist[k] = itm_hist.get(k, 0) + v
            if done % 8 == 0 or done == len(tasks):
                log(f"C0 {done}/{len(tasks)} chunks "
                    f"[{time.time() - t0:.0f}s]")
    out = dict(header)
    out["record_type"] = "c0_control"
    out.update({
        "stacks": [1500] * N_SEATS, "level": 1, "dealer": 0, "M": M_C0,
        "M_eff": m_eff, "n_capped": n_capped, "n_tainted": n_tainted,
        "taints": taints, "survive_counts": counts,
        "itm_counts": itm_counts,
        "survivor_sum_hist": surv_hist,
        "itm_sum_hist": itm_hist,
        "mean_hands": hands_sum / m_eff if m_eff else float("nan"),
        "elapsed_s": time.time() - t0,
    })
    (out_dir / "c0_control.json").write_text(json.dumps(out, indent=2))
    log(f"C0 done: itm_counts={itm_counts} raw_counts={counts} "
        f"capped={n_capped} tainted={n_tainted} "
        f"surv_hist={surv_hist} itm_hist={itm_hist}")
    return 0


def _c0_chunk(task):
    counts = [0] * N_SEATS
    itm_counts = [0] * N_SEATS
    surv_hist = {}
    itm_hist = {}
    n_capped = n_tainted = 0
    taints = []
    hands_sum = 0
    m_eff = 0
    for m in range(task["m_lo"], task["m_hi"]):
        r = rollout_survivors(_POLICY, _STRUCTURE, [1500] * N_SEATS, 1, 0,
                              rollout_seed("C0", 0, m))
        if r["tainted"]:
            n_tainted += 1
            taints.append({"m": m, "detail": r["exception"]})
            continue
        if r["capped"]:
            n_capped += 1
            continue
        m_eff += 1
        hands_sum += r["hands"]
        ssum = sum(r["survived"])
        surv_hist[str(ssum)] = surv_hist.get(str(ssum), 0) + 1
        isum = sum(r["itm"])
        itm_hist[str(isum)] = itm_hist.get(str(isum), 0) + 1
        for i in range(N_SEATS):
            counts[i] += r["survived"][i]
            itm_counts[i] += r["itm"][i]
    return {"counts": counts, "itm_counts": itm_counts,
            "surv_hist": surv_hist, "itm_hist": itm_hist,
            "n_capped": n_capped,
            "n_tainted": n_tainted, "taints": taints, "hands_sum": hands_sum,
            "m_eff": m_eff}


def phase_tier1(args, out_dir):
    """One primary cell: N states x M=100 rollouts, sharded over workers."""
    import multiprocessing as mp
    assert args.cell in PRIMARY_CELLS
    header = run_header(args)
    header["cell"] = args.cell
    distinct, meta = load_primary_frame(args.dist)
    header["frame_meta"] = meta
    states = sample_cell_states(distinct, args.cell, args.n)
    log(f"cell {args.cell}: {len(states)} states sampled "
        f"(distinct pool {meta['distinct_tuples'][args.cell]}), M={M_TIER1}")
    tasks = [{"cell": args.cell, "state_idx": i, "stacks": s["stacks"],
              "level": s["level"], "dealer": s["dealer"], "M": M_TIER1}
             for i, s in enumerate(states)]
    jsonl_path = out_dir / f"tier1_{args.cell}.jsonl"
    if args.resume and jsonl_path.exists():
        # CRN-deterministic design: rollout seed = f(cell, state_idx, m),
        # so skipping already-completed states is exact. State sampling is
        # itself deterministic (Random(MASTER_SEED) over sorted tuples),
        # so state_idx -> state is stable across invocations; verify.
        done_idx = set()
        with open(jsonl_path) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("record_type") == "run_header":
                    continue
                i = rec["state_idx"]
                assert rec["stacks"] == tasks[i]["stacks"] and \
                    rec["level"] == tasks[i]["level"] and \
                    rec["dealer"] == tasks[i]["dealer"], \
                    f"resume mismatch at state_idx {i}"
                done_idx.add(i)
        tasks = [t for t in tasks if t["state_idx"] not in done_idx]
        log(f"resume: {len(done_idx)} states already on disk, "
            f"{len(tasks)} remaining")
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
                    log(f"{args.cell} {done}/{len(tasks)} states "
                        f"[{el:.0f}s, {el / done:.1f} s/state, "
                        f"eta {(len(tasks) - done) * el / done / 60:.0f}m]")
                if row["n_tainted"]:
                    log(f"  [TAINT] {args.cell} state {row['state_idx']}: "
                        f"{row['taints']}")
    log(f"cell {args.cell} complete -> {jsonl_path}")
    return 0


# ── Analysis / report ──────────────────────────────────────────────────

def t_ci(d, conf):
    import numpy as np
    from scipy import stats
    d = np.asarray(d, dtype=float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    tcrit = stats.t.ppf(0.5 + conf / 2, n - 1)
    return float(m), float(se), float(m - tcrit * se), float(m + tcrit * se)


def analyze_cell(rows, bonf_alpha):
    """Primary + secondary scalars for one cell's state rows."""
    import numpy as np
    from scipy import stats
    d_short, noise_terms = [], []
    d_short_raw = []          # design's original Y definition (sensitivity)
    rank_d = {}
    pos_d = {}
    depth_d = {}
    dropped = []
    taint_total = 0
    for r in rows:
        taint_total += r["n_tainted"]
        if r["n_capped"] > 0.02 * r["M"]:
            dropped.append({"state_idx": r["state_idx"],
                            "n_capped": r["n_capped"]})
            continue
        st = r["stacks"]
        q = q_vector(st)
        m_eff = r["M_eff"]
        p_hat = [c / m_eff for c in r["itm_counts"]]
        p_hat_raw = [c / m_eff for c in r["survive_counts"]]
        alive = [i for i in range(N_SEATS) if st[i] > 0]
        short = min(alive, key=lambda i: (st[i], i))
        d_short.append(p_hat[short] - q[short])
        d_short_raw.append(p_hat_raw[short] - q[short])
        noise_terms.append(p_hat[short] * (1 - p_hat[short]) / (m_eff - 1))
        by_rank = sorted(alive, key=lambda i: (st[i], i))
        for rk, seat in enumerate(by_rank, start=1):
            rank_d.setdefault(rk, []).append(p_hat[seat] - q[seat])
        pos = seat_positions(st, r["dealer"])
        for seat in alive:
            pos_d.setdefault(pos[seat], []).append(p_hat[seat] - q[seat])
        bl = _struct_level(r["level"])
        for seat in alive:
            depth = st[seat] / bl.big_blind
            for lo, hi in DEPTH_BANDS:
                if lo <= depth < hi:
                    key = f"[{lo},{hi})" if hi != float("inf") else f"{lo}+"
                    depth_d.setdefault(key, []).append(p_hat[seat] - q[seat])
                    break
    n = len(d_short)
    arr = np.asarray(d_short)
    mean, se, lo95, hi95 = t_ci(arr, 0.95)
    _, _, lob, hib = t_ci(arr, 1 - bonf_alpha)
    _, _, lo90, hi90 = t_ci(arr, 0.90)          # TOST
    var_total = float(arr.var(ddof=1))
    var_noise = float(np.mean(noise_terms))
    var_bias = max(0.0, var_total - var_noise)
    boot = stats.bootstrap((arr,), np.mean, n_resamples=10000,
                           confidence_level=1 - bonf_alpha, method="BCa",
                           random_state=np.random.default_rng(MASTER_SEED))
    pval = float(stats.ttest_1samp(arr, 0.0).pvalue)

    def smap(dd):
        out = {}
        for k, v in sorted(dd.items(), key=lambda kv: str(kv[0])):
            v = np.asarray(v)
            out[str(k)] = {
                "n": int(len(v)), "mean": float(v.mean()),
                "se": float(v.std(ddof=1) / math.sqrt(len(v)))
                      if len(v) > 1 else float("nan")}
        return out

    raw_arr = np.asarray(d_short_raw)
    return {
        "n_states_scored": n, "n_states_dropped_cap": dropped,
        "n_tainted_rollouts": taint_total,
        "sensitivity_d_short_raw_definition": {
            "mean": float(raw_arr.mean()),
            "se": float(raw_arr.std(ddof=1) / math.sqrt(n)),
        },
        "primary_d_short": {
            "mean": mean, "se": se, "ci95": [lo95, hi95],
            "ci_bonferroni": [lob, hib],
            "ci_bonferroni_level": 1 - bonf_alpha,
            "ci90_tost": [lo90, hi90],
            "bca_ci_bonferroni": [float(boot.confidence_interval.low),
                                  float(boot.confidence_interval.high)],
            "p_value_t": pval,
        },
        "variance_decomposition": {
            "sigma2_total": var_total, "sigma2_noise": var_noise,
            "sigma2_bias": var_bias, "sigma_bias": math.sqrt(var_bias),
        },
        "secondary_by_stack_rank": smap(rank_d),
        "secondary_by_position": smap(pos_d),
        "secondary_by_depth_bb": smap(depth_d),
        "d_short_values": [float(x) for x in arr],
    }


_STRUCT_CACHE = None


def _struct_level(level):
    global _STRUCT_CACHE
    if _STRUCT_CACHE is None:
        from src.nlhe.game_strings import TournamentStructure
        from scripts.throwaway_query_real_ante import _RealAnteStructure
        _STRUCT_CACHE = _RealAnteStructure(
            TournamentStructure.from_yaml(DEFAULT_STRUCT))
    return _STRUCT_CACHE.level(level)


def analyze_c0(c0):
    """Instrument test + position effect."""
    import numpy as np
    from scipy import stats
    m_eff = c0["M_eff"]

    def seat_avg_test(hist):
        e_vals = []
        for ssum, cnt in hist.items():
            e_vals.extend([int(ssum) / 6 - 0.5] * cnt)
        e = np.asarray(e_vals, dtype=float)
        if e.std(ddof=1) == 0:
            m = float(e.mean())
            return {"mean": m, "se": 0.0, "ci95": [m, m],
                    "p_value": 1.0 if m == 0 else 0.0}
        mean, se, lo, hi = t_ci(e, 0.95)
        return {"mean": mean, "se": se, "ci95": [lo, hi],
                "p_value": float(stats.ttest_1samp(e, 0.0).pvalue)}

    itm_test = seat_avg_test(c0["itm_sum_hist"])
    raw_test = seat_avg_test(c0["survivor_sum_hist"])
    lo, hi = itm_test["ci95"]
    instrument_ok = (lo <= 0 <= hi and c0["n_tainted"] == 0
                     and c0["n_capped"] <= 0.02 * c0["M"])
    pos_names = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO"}
    pos_effect = {}
    for i in range(N_SEATS):
        p = c0["itm_counts"][i] / m_eff
        se_i = math.sqrt(p * (1 - p) / m_eff)
        pos_effect[pos_names[i]] = {
            "p_hat": p, "dev_from_0.5": p - 0.5, "binom_se": se_i,
            "z": (p - 0.5) / se_i if se_i else float("nan")}
    return {
        "M": c0["M"], "M_eff": m_eff, "n_capped": c0["n_capped"],
        "n_tainted": c0["n_tainted"],
        "seat_avg_error": itm_test,
        "seat_avg_error_raw_definition": raw_test,
        "instrument_ok": bool(instrument_ok),
        "survivor_sum_hist": c0["survivor_sum_hist"],
        "itm_sum_hist": c0["itm_sum_hist"],
        "mean_hands": c0["mean_hands"],
        "position_effect": pos_effect,
    }


def phase_report(args, out_dir):
    bench = json.loads((out_dir / "rate_benchmark.json").read_text())
    c0 = json.loads((out_dir / "c0_control.json").read_text())
    c0_res = analyze_c0(c0)
    bonf_alpha = 0.05 / len(PRIMARY_CELLS)
    cells = {}
    headers = {}
    for cell in PRIMARY_CELLS:
        path = out_dir / f"tier1_{cell}.jsonl"
        rows = []
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("record_type") == "run_header":
                    headers[cell] = rec
                else:
                    rows.append(rec)
        cells[cell] = analyze_cell(rows, bonf_alpha)
        log(f"analyzed {cell}: n={cells[cell]['n_states_scored']} "
            f"mean={cells[cell]['primary_d_short']['mean']:+.4f}")

    # Verdict per design section 6/7 (flip rate is Tier 2b -> pending).
    bstar_cell = max(PRIMARY_CELLS,
                     key=lambda c: abs(cells[c]["primary_d_short"]["mean"]))
    bstar = cells[bstar_cell]["primary_d_short"]["mean"]
    def _outside_band(c):
        lo, hi = cells[c]["primary_d_short"]["ci_bonferroni"]
        return lo > 0.01 or hi < -0.01

    material = any(abs(cells[c]["primary_d_short"]["mean"]) >= 0.02
                   and _outside_band(c) for c in PRIMARY_CELLS)
    falsified = any(_outside_band(c) for c in PRIMARY_CELLS)
    tost_pass = all(
        -0.01 < cells[c]["primary_d_short"]["ci90_tost"][0]
        and cells[c]["primary_d_short"]["ci90_tost"][1] < 0.01
        for c in PRIMARY_CELLS)
    sigma_bias_max = max(cells[c]["variance_decomposition"]["sigma_bias"]
                         for c in PRIMARY_CELLS)
    heterogeneous = sigma_bias_max > 0.05
    marginal = (0.01 <= abs(bstar) < 0.02) or heterogeneous
    if not c0_res["instrument_ok"]:
        verdict = "INSTRUMENT FAILURE — HALT (C0 seat-averaged error CI excludes 0)"
    elif material:
        verdict = "MATERIAL (B* >= 0.02, Bonferroni CI excludes +/-0.01)"
    elif falsified:
        verdict = ("FALSIFIED at +/-0.01 (a Bonferroni CI lies outside the "
                   "band) — MARGINAL action branch")
    elif tost_pass:
        verdict = ("EQUIVALENCE CONFIRMED (TOST: all 90% CIs inside "
                   "+/-0.01) — NULL/VALIDATION branch, pending Tier-2b "
                   "flip-rate <= 2% for full closure")
    elif marginal:
        verdict = "MARGINAL (0.01 <= B* < 0.02 or sigma_bias > 0.05)"
    else:
        verdict = "INCONCLUSIVE (CIs straddle a bound) — adaptive rule governs"

    results = {
        "record_type": "tier1_results",
        "design": "docs/research_program/ICM_GAP_STUDY_DESIGN.md",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "headers": headers,
        "rate_benchmark": bench,
        "c0_control": c0_res,
        "bonferroni_alpha_per_cell": bonf_alpha,
        "cells": cells,
        "B_star": {"cell": bstar_cell, "value": bstar},
        "sigma_bias_max": sigma_bias_max,
        "verdict": verdict,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    log(f"wrote {out_dir / 'results.json'}")
    log(f"VERDICT: {verdict}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True,
                    choices=["bench", "c0", "tier1", "report"])
    ap.add_argument("--cell", choices=PRIMARY_CELLS)
    ap.add_argument("--n", type=int, default=N_TIER1)
    ap.add_argument("--resume", action="store_true",
                    help="tier1: skip state_idx already present in the "
                         "cell's jsonl (CRN makes this exact)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--abstraction", default=DEFAULT_ABSTR)
    ap.add_argument("--structure", default=DEFAULT_STRUCT)
    ap.add_argument("--dist", default=DEFAULT_DIST)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "bench":
        return phase_bench(args, out_dir)
    if args.phase == "c0":
        return phase_c0(args, out_dir)
    if args.phase == "tier1":
        if not args.cell:
            raise SystemExit("--cell required for tier1")
        return phase_tier1(args, out_dir)
    return phase_report(args, out_dir)


if __name__ == "__main__":
    sys.exit(main())
