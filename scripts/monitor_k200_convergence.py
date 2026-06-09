"""Convergence-monitoring harness for k200 real-ante retrain.

External process. Watches a training run's output dir, probes every new
checkpoint, and appends rows to a single convergence CSV. Does not
modify training; pure observability.

What it logs per checkpoint (every 100 iters from training):
  - Behavioral: sample-mode AA/KK/QQ at 60bb and 10bb
    (200 samples each → fold%, jam-share, nr-share)
  - Aggregates: premium-fold (avg AA/KK at each depth),
    jam-share at each depth, depth-gap (10bb − 60bb jam-share)
  - Training losses: adv_loss + strat_loss parsed from the training log
    at the checkpoint iteration
  - Convergence proxy (only at multiples of 200, comparing vs iter/2):
    play 100 paired-CRN hands of 6-max NLHE between current ckpt and
    half-iter ckpt, log the win margin in BB/100 for the later ckpt

CSV row appended atomically (file open in 'a' mode per row).

Usage:
    python scripts/monitor_k200_convergence.py \\
        --run-dir runs/k200_real_ante_20260605_224706 \\
        --train-log /tmp/k200_real_ante_train.log

Persistent: stays running until killed. Idempotent: if rerun, skips
checkpoints already in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import pickle
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyspiel

from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.networks6 import N_DISCRETE_ACTIONS
from src.nlhe.cfr6 import _build_view_6max
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.integration.replay import deal_one_card_6max
from scripts.throwaway_query_real_ante import _RealAnteStructure

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  monitor  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("monitor")

NORMAL = {int(DiscreteAction.BET_33), int(DiscreteAction.BET_50),
          int(DiscreteAction.BET_66), int(DiscreteAction.BET_100),
          int(DiscreteAction.BET_150), int(DiscreteAction.BET_200)}
ALLIN = int(DiscreteAction.ALLIN)
FOLD = int(DiscreteAction.FOLD)
CALL = int(DiscreteAction.CALL)

HANDS = [("AA", ("As", "Ah")), ("KK", ("Ks", "Kh")), ("QQ", ("Qs", "Qh"))]
DEPTHS = [(1, "60bb"), (4, "10bb")]
SAMPLES_PER_PROBE = 200

CSV_FIELDS = [
    "iter", "timestamp",
    # Per-hand-per-depth sample-mode behavior — absolute percentages
    *[f"{h}_{d}_{m}" for h in ("AA","KK","QQ") for d in ("60bb","10bb")
       for m in ("fold","call","raise","jam","nr_share","jam_share")],
    # Aggregates (premium = AA+KK averaged)
    "premium_fold_60bb", "premium_fold_10bb",
    "jam_share_60bb", "jam_share_10bb", "depth_gap_pp",
    # Training losses (from log)
    "adv_loss_at_iter", "strat_loss_at_iter",
    # Convergence proxy
    "proxy_vs_iter", "proxy_margin_chips_per_hand",
]


def build_state_at_btn(structure, level, cards, hero_seat=5):
    gs = structure.to_inner_game_string(level=level)
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    for _ in range(80):
        if state.is_chance_node():
            deal_one_card_6max(state, hero_seat, cards, ())
            continue
        if state.current_player() == hero_seat:
            return state
        state.apply_action(0)
    raise RuntimeError("setup exhausted in build_state_at_btn")


def behavioral_probe(solver, structure, rng):
    """Probe AA/KK/QQ at 60bb and 10bb. Returns dict of per-hand-per-depth
    absolute action percentages (fold/call/raise/jam) AND share metrics
    (nr_share/jam_share) computed from 200 sample-mode action draws."""
    out = {}
    for hand_name, cards in HANDS:
        for level, depth_label in DEPTHS:
            state = build_state_at_btn(structure, level, cards)
            parsed = parse_state_6max(state, observer=5)
            parsed["dealer_seat"] = 5
            legal = list(state.legal_actions())
            view = _build_view_6max(state, parsed)
            d2c = discretize_legal_actions(legal, view)
            mask = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
            for da in d2c:
                mask[int(da)] = 1.0
            feats = np.asarray(
                solver.encoder.encode_from_parsed(parsed, rng=rng),
                dtype=np.float32)
            p = np.asarray(
                solver.policy_nets.inference_policy(5, feats, mask),
                dtype=np.float64)
            samples = rng.choices(range(N_DISCRETE_ACTIONS),
                                   weights=p.tolist(),
                                   k=SAMPLES_PER_PROBE)
            n_fold = sum(1 for a in samples if a == FOLD)
            n_call = sum(1 for a in samples if a == CALL)
            n_normal = sum(1 for a in samples if a in NORMAL)
            n_allin = sum(1 for a in samples if a == ALLIN)
            n_raise = n_normal + n_allin
            out[f"{hand_name}_{depth_label}_fold"] = 100.0 * n_fold / SAMPLES_PER_PROBE
            out[f"{hand_name}_{depth_label}_call"] = 100.0 * n_call / SAMPLES_PER_PROBE
            out[f"{hand_name}_{depth_label}_raise"] = 100.0 * n_normal / SAMPLES_PER_PROBE
            out[f"{hand_name}_{depth_label}_jam"] = 100.0 * n_allin / SAMPLES_PER_PROBE
            out[f"{hand_name}_{depth_label}_jam_share"] = (
                100.0 * n_allin / n_raise) if n_raise > 0 else 0.0
            out[f"{hand_name}_{depth_label}_nr_share"] = (
                100.0 * n_normal / n_raise) if n_raise > 0 else 0.0
    return out


def format_probe_block(iter_n, behav, premium_fold_60bb, premium_fold_10bb,
                        jam_60, jam_10, depth_gap,
                        proxy_vs_iter, proxy_margin,
                        adv_loss, strat_loss):
    """Format the multi-line probe block the user reads via tail -f.
    Single write() — appended atomically to the main log."""
    lines = []
    lines.append(f"===== CONVERGENCE PROBE @ iter_{iter_n:04d} =====")
    for hand in ("AA", "KK", "QQ"):
        for depth in ("60bb", "10bb"):
            f = behav[f"{hand}_{depth}_fold"]
            c = behav[f"{hand}_{depth}_call"]
            r = behav[f"{hand}_{depth}_raise"]
            j = behav[f"{hand}_{depth}_jam"]
            lines.append(
                f"{hand}  {depth}: fold {f:>4.0f}%  raise {r:>4.0f}%  "
                f"jam {j:>4.0f}%  call {c:>4.0f}%")
    lines.append(
        f"premium-fold avg: 60bb {premium_fold_60bb:.0f}% / "
        f"10bb {premium_fold_10bb:.0f}%   (target: 3-7%)")
    lines.append(
        f"depth-gap (10bb-60bb jam-share): {depth_gap:+.0f}pp")
    if adv_loss is not None and strat_loss is not None:
        lines.append(
            f"losses at iter: adv={adv_loss:.4f}  strat={strat_loss:.4f}")
    if proxy_margin is not None and proxy_vs_iter:
        lines.append(
            f"vs-past-self (iter_{iter_n:04d} vs iter_{proxy_vs_iter:04d}): "
            f"{proxy_margin:+.3f} chips/hand")
    lines.append("=" * 40)
    return "\n".join(lines) + "\n"


def parse_losses_at_iter(train_log_path: str, target_iter: int):
    """Walk the training log for the LAST adv= and strat= values at the
    given iter. Each iter logs once per traversed seat, so we average
    across the seats touched at this iter (6 traversals)."""
    if not Path(train_log_path).exists():
        return None, None
    # Look for "iter  N/M  trav=K  adv=  X.XXX  strat=  Y.YYY"
    pat = re.compile(
        rf"iter\s+{target_iter}/\d+\s+trav=\d+\s+adv=\s+([\d.]+)\s+strat=\s+([\d.]+)")
    advs, strats = [], []
    try:
        with open(train_log_path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    advs.append(float(m.group(1)))
                    strats.append(float(m.group(2)))
    except Exception:
        return None, None
    if advs:
        return float(np.mean(advs)), float(np.mean(strats))
    return None, None


def play_paired_crn_match(solver_a, solver_b, structure, n_pairs=100,
                          seed_base=42):
    """Play n_pairs PAIRED CRN hands of 6-max NLHE between two solvers.
    In each pair: same dealer/cards (same rng seed), with solvers swapped
    across seats. Returns solver_a's mean equity delta in BB/100.

    Implementation note: 'paired CRN' here means we replay the exact same
    rng-driven trajectory twice with swapped seat assignments. Variance
    reduction from controlling deal-luck.
    """
    from scripts.eval_6max_self_play import play_one_hand
    margins = []
    # Each pair = 2 games (A on even seats, B on odd; then mirrored).
    # We track solver_a's mean chip delta in BB/100.
    for i in range(n_pairs):
        seed = seed_base + i
        # Game A: solver_a at even seats {0,2,4}, solver_b at odd {1,3,5}
        seats_A = [solver_a, solver_b, solver_a, solver_b, solver_a, solver_b]
        rng = random.Random(seed)
        try:
            result_A = play_one_hand(seats_A, structure, rng,
                                      num_paid=3, mode="sample")
        except Exception:
            continue
        # Sum solver_a's chip delta across its 3 seats
        delta_A = sum(result_A["seat_to_equity_delta"][s]
                      for s in (0, 2, 4))
        # Game B: solver_a at odd seats, solver_b at even (mirrored, same seed)
        seats_B = [solver_b, solver_a, solver_b, solver_a, solver_b, solver_a]
        rng = random.Random(seed)
        try:
            result_B = play_one_hand(seats_B, structure, rng,
                                      num_paid=3, mode="sample")
        except Exception:
            continue
        delta_B = sum(result_B["seat_to_equity_delta"][s]
                      for s in (1, 3, 5))
        # Paired mean = (A + B) / 2 — variance reduced
        paired_delta = (delta_A + delta_B) / 2.0
        margins.append(paired_delta)
    if not margins:
        return float("nan")
    # Convert chip margin to BB/100. Level-weighted average BB ≈ structure
    # level 3 ≈ 100 chips. We just report mean chips/hand × 100 / level_3_bb.
    mean_per_hand = float(np.mean(margins))
    # Use a stable BB reference — small_blind*2 from production = 100.
    return mean_per_hand


def load_solver_real_ante(ckpt_path, abstraction, structure):
    """Load a 236-dim solver. Wraps _load_solver, no override needed
    because the production run is 236-dim by design."""
    from scripts.eval_6max_self_play import _load_solver
    solver = _load_solver(ckpt_path, abstraction, structure)
    solver.tournament_structure = structure
    return solver


def get_processed_iters(csv_path: str):
    if not Path(csv_path).exists():
        return set()
    processed = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                processed.add(int(row["iter"]))
            except (KeyError, ValueError):
                pass
    return processed


def append_row(csv_path: str, row: dict):
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        # Round floats for readability
        clean = {}
        for k, v in row.items():
            if isinstance(v, float):
                clean[k] = round(v, 3)
            else:
                clean[k] = v
        writer.writerow(clean)


def probe_ckpt(ckpt_path: str, abstraction, structure, train_log,
               main_log: str, run_dir: str, rng_seed: int):
    """Probe one checkpoint. Writes CSV row AND appends formatted probe
    block to main_log (the training log file)."""
    name = Path(ckpt_path).name
    m = re.search(r"ckpt_iter_(\d+)\.pt", name)
    if not m:
        return None
    cur_iter = int(m.group(1))

    rng = random.Random(rng_seed)

    log.info(f"Probing {name} (iter={cur_iter})...")
    solver = load_solver_real_ante(ckpt_path, abstraction, structure)
    behav = behavioral_probe(solver, structure, rng)

    premium_fold_60bb = (behav["AA_60bb_fold"] + behav["KK_60bb_fold"]) / 2
    premium_fold_10bb = (behav["AA_10bb_fold"] + behav["KK_10bb_fold"]) / 2
    jam_60bb = (behav["AA_60bb_jam_share"] + behav["KK_60bb_jam_share"]) / 2
    jam_10bb = (behav["AA_10bb_jam_share"] + behav["KK_10bb_jam_share"]) / 2
    depth_gap = jam_10bb - jam_60bb

    adv_loss, strat_loss = parse_losses_at_iter(train_log, cur_iter)

    proxy_vs_iter = None
    proxy_margin = None
    if cur_iter >= 200 and cur_iter % 200 == 0:
        half = cur_iter // 2
        half_path = f"{run_dir}/ckpt_iter_{half:04d}.pt"
        if Path(half_path).exists():
            log.info(f"  convergence proxy: {cur_iter} vs {half}")
            try:
                solver_half = load_solver_real_ante(
                    half_path, abstraction, structure)
                proxy_margin = play_paired_crn_match(
                    solver, solver_half, structure, n_pairs=100,
                    seed_base=rng_seed + 1000)
                proxy_vs_iter = half
            except Exception as e:
                log.warning(f"  proxy failed: {e}")

    # Print formatted block to stdout — gets appended to the main log by
    # the shell's `>>` redirect. Single write so it's atomic.
    block = format_probe_block(
        cur_iter, behav, premium_fold_60bb, premium_fold_10bb,
        jam_60bb, jam_10bb, depth_gap,
        proxy_vs_iter, proxy_margin, adv_loss, strat_loss)
    sys.stdout.write(block)
    sys.stdout.flush()

    # Also write directly to main_log file via append — belt-and-suspenders
    # in case the redirect chain is misconfigured. POSIX O_APPEND is atomic
    # for writes under PIPE_BUF (~4KB), and our block is ~600 bytes.
    if main_log and main_log != "stdout":
        try:
            with open(main_log, "a") as f:
                f.write(block)
                f.flush()
        except Exception as e:
            log.warning(f"could not append probe block to {main_log}: {e}")

    row = {
        "iter": cur_iter,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **behav,
        "premium_fold_60bb": premium_fold_60bb,
        "premium_fold_10bb": premium_fold_10bb,
        "jam_share_60bb": jam_60bb,
        "jam_share_10bb": jam_10bb,
        "depth_gap_pp": depth_gap,
        "adv_loss_at_iter": adv_loss if adv_loss is not None else "",
        "strat_loss_at_iter": strat_loss if strat_loss is not None else "",
        "proxy_vs_iter": proxy_vs_iter if proxy_vs_iter else "",
        "proxy_margin_chips_per_hand": proxy_margin if proxy_margin is not None else "",
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-log", required=True,
                    help="training log file (read for loss values)")
    ap.add_argument("--main-log", default="",
                    help="if set, append probe blocks here directly (in "
                         "addition to stdout); useful when monitor stdout "
                         "is being read by something other than the main log")
    ap.add_argument("--abstraction",
                    default="runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    ap.add_argument("--structure",
                    default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--poll-secs", type=int, default=30)
    ap.add_argument("--once", action="store_true",
                    help="probe whatever ckpts exist now and exit (smoke test)")
    args = ap.parse_args()

    run_dir = args.run_dir
    csv_path = f"{run_dir}/convergence_log.csv"
    log.info(f"Run dir: {run_dir}")
    log.info(f"CSV    : {csv_path}")
    log.info(f"Train log: {args.train_log}")

    with open(args.abstraction, "rb") as f:
        abstraction = pickle.load(f)
    base = TournamentStructure.from_yaml(args.structure)
    structure = _RealAnteStructure(base)

    rng_seed = 2026

    while True:
        processed = get_processed_iters(csv_path)
        # Find all ckpts
        all_ckpts = sorted(Path(run_dir).glob("ckpt_iter_*.pt"),
                            key=lambda p: int(
                                re.search(r"ckpt_iter_(\d+)\.pt", p.name).group(1)))
        for ck in all_ckpts:
            m = re.search(r"ckpt_iter_(\d+)\.pt", ck.name)
            it = int(m.group(1))
            if it in processed:
                continue
            try:
                row = probe_ckpt(str(ck), abstraction, structure,
                                  args.train_log, args.main_log,
                                  run_dir, rng_seed)
                if row:
                    append_row(csv_path, row)
            except Exception as e:
                log.error(f"  probe failed for {ck.name}: {e}")
                import traceback; traceback.print_exc()
        if args.once:
            break
        time.sleep(args.poll_secs)


if __name__ == "__main__":
    main()
