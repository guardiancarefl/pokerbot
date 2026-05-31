"""D-pivot probe: re-evaluate an EXISTING co-train checkpoint under the
argmax-disagreement read primitive (no retraining).

Why this script exists (vs `--read-primitive argmax` on the cotrain script):
the read primitive is inference-only and lives outside the training loss, so
the A+B checkpoint can be probed with the new primitive in seconds. This
script also computes the diagnostic the cotrain probe doesn't:
  - conditional accuracy on choice-revealing decisions (|legal|>=2 AND raise
    legal AND pi_eq[raise]<1.0) — the revised condition (iii) bar
  - per-cell breakdown of disagreement events into lopsided GTO spots
    (gto_margin >= 0.3) vs near-tied GTO spots (gto_margin < 0.3) — checks
    whether the ratio denominator is being inflated by spurious near-tie firing

Usage:
  cd ~/pokerbot && .venv/bin/python -m scripts.leduc_d_pivot_probe \
      --ckpt runs/leduc_phase1cotrain_20260531_222313/phase1_cotrain.pt \
      --anchor-dir runs/leduc_cfr_anchor_20260531_144405
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pyspiel
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from src.leduc import archetypes as A
from src.leduc.adaptive import tokens as TT
from src.leduc.adaptive.model import (
    AdaptivePolicyNet, collate, read_terms_argmax, running_read,
)

GAME = pyspiel.load_game("leduc_poker")
NEG_INF = -1e9

# Bars (revised separation gate per D-pivot)
SEP_RATIO_MIN = 2.0
MANIAC_G_MIN = 0.40
MANIAC_COND_ACC_MIN = 0.85
E0_BAR = 5.0
RECAL_G_MANIAC = 0.50
RECAL_G_UNIFORM_MAX = 0.10
GTO_LOPSIDED_THRESHOLD = 0.30  # gto_margin >= 0.30 -> lopsided


def _log(msg=""):
    print(msg, flush=True)


def load_anchor(anchor_dir: Path):
    with open(anchor_dir / "avg_policy_arrays.pkl", "rb") as f:
        snap = pickle.load(f)
    state_lookup = snap["state_lookup"]
    probs_arr = np.asarray(snap["action_probability_array"])
    mask_arr = np.asarray(snap["legal_actions_mask"])

    def anchor_fn(state):
        info = state.information_state_string()
        idx = state_lookup[info]
        probs = probs_arr[idx]
        mask = mask_arr[idx]
        return {int(a): float(probs[a]) for a in range(len(probs)) if mask[a]}

    return anchor_fn


def sample_from_dict(probs: dict, rng) -> int:
    keys = list(probs.keys())
    ps = np.array(list(probs.values()), dtype=np.float64)
    ps = ps / ps.sum()
    return int(rng.choice(keys, p=ps))


def run_hand_collect(net, anchor_fn, opp_fn, hero_seat, *, w0, device, rng):
    """Run one hand. At each opp decision, compute argmax-disagreement read
    terms AND record diagnostic fields (q_t, pi_eq, legal_mask, observed
    action). Returns (D_bar, n_opp_decisions, records[]) where each record is
    a dict with the fields per_cell needs."""
    state = GAME.new_initial_state()
    d_list, w_list = [], []
    cached_opp_logits = None
    records = []
    while not state.is_terminal():
        if state.is_chance_node():
            outs = state.chance_outcomes()
            acts = [a for a, _ in outs]
            ps = np.array([p for _, p in outs])
            state.apply_action(int(rng.choice(acts, p=ps / ps.sum())))
            continue
        cur = state.current_player()
        if cur == hero_seat:
            toks = TT.tokenize_decision(state)
            packed = collate([toks], device=device)
            with torch.no_grad():
                policy_raw, opp_logits, _ = net(
                    packed["tokens"], packed["pad_mask"], packed["query_idx"],
                    packed["legal_mask"], packed["opp_stats"],
                )
            cached_opp_logits = opp_logits[0].cpu().numpy()
            legal = toks["legal_actions"]
            net_probs = policy_raw[0].cpu().numpy()
            ps = np.array([net_probs[a] for a in legal])
            ps = ps / ps.sum()
            state.apply_action(int(rng.choice(legal, p=ps)))
        else:
            legal = state.legal_actions()
            lm = np.zeros(TT.NUM_ACTIONS, dtype=bool)
            for la in legal:
                lm[la] = True
            if cached_opp_logits is not None:
                lgt = cached_opp_logits.astype(np.float64).copy()
                lgt[~lm] = NEG_INF
                lgt -= lgt.max()
                e = np.exp(lgt) * lm
                q_t = e / e.sum()
            else:
                q_t = lm.astype(np.float64) / lm.sum()
            ap = anchor_fn(state)
            pi_eq = np.zeros(TT.NUM_ACTIONS, dtype=np.float64)
            for a, p in ap.items():
                pi_eq[a] = p
            d_t, w_t = read_terms_argmax(q_t, pi_eq, lm)
            d_list.append(d_t)
            w_list.append(w_t)
            op = opp_fn(state)
            a = sample_from_dict(op, rng)
            records.append({
                "q": q_t.copy(),
                "pi_eq": pi_eq.copy(),
                "legal_mask": lm.copy(),
                "observed_action": int(a),
                "d_t": float(d_t),
                "w_t": float(w_t),
            })
            state.apply_action(a)
    return running_read(d_list, w_list, w0=w0), len(d_list), records


def _gto_margin(pi_eq, legal_mask) -> float:
    legal = legal_mask.astype(bool)
    n = int(legal.sum())
    if n <= 1:
        return 1.0  # degenerate; cap-forced spots aren't disagreement-eligible
    p = pi_eq[legal].astype(np.float64)
    p = p / max(p.sum(), 1e-12)
    p_sorted = np.sort(p)[::-1]
    return float(p_sorted[0] - p_sorted[1])


def _argmax_set(pi_eq, legal_mask, tol=1e-9):
    legal = legal_mask.astype(bool)
    if legal.sum() == 0:
        return set()
    p = pi_eq.astype(np.float64).copy()
    p[~legal] = -np.inf
    pmax = p[legal].max()
    return {int(a) for a in range(len(p)) if legal[a] and (pmax - p[a]) <= tol}


def cell_metrics(records, cell_id, dbars, kappa):
    n_dec = len(records)
    if n_dec == 0:
        return {
            "cell_id": cell_id, "n_matches": len(dbars), "n_opp_decisions": 0,
            "mean_dbar": 0.0, "median_dbar": 0.0,
            "mean_g_at_kappa": 0.0, "median_g_at_kappa": 0.0,
            "overall_acc": float("nan"), "conditional_acc": float("nan"),
            "choice_reveal_acc": float("nan"), "n_choice_reveal": 0,
            "n_disagreement": 0,
            "n_disagreement_lopsided": 0, "n_disagreement_near_tied": 0,
            "frac_disagreement_lopsided": float("nan"),
        }
    mean_dbar = float(np.mean(dbars))
    median_dbar = float(np.median(dbars))
    preds = np.stack([r["q"] for r in records])
    acts = np.array([r["observed_action"] for r in records])
    legal_counts = np.array([int(r["legal_mask"].sum()) for r in records])
    A_RAISE = 2  # leduc encoding: 0=fold,1=call,2=raise (from tokens.NUM_ACTIONS layout)
    pi_raise = np.array([float(r["pi_eq"][A_RAISE]) for r in records])
    raise_legal = np.array([bool(r["legal_mask"][A_RAISE]) for r in records])

    overall_acc = float((preds.argmax(axis=-1) == acts).mean())

    cond_mask = legal_counts >= 2
    n_cond = int(cond_mask.sum())
    cond_acc = (float((preds[cond_mask].argmax(axis=-1) == acts[cond_mask]).mean())
                if n_cond > 0 else float("nan"))

    cr_mask = cond_mask & raise_legal & (pi_raise < 1.0 - 1e-9)
    n_cr = int(cr_mask.sum())
    cr_acc = (float((preds[cr_mask].argmax(axis=-1) == acts[cr_mask]).mean())
              if n_cr > 0 else float("nan"))

    n_disagree = 0
    n_disagree_lop = 0
    n_disagree_near = 0
    for r in records:
        if int(r["legal_mask"].sum()) <= 1:
            continue
        q_argmax = int(np.argmax(np.where(r["legal_mask"], r["q"], -np.inf)))
        if q_argmax not in _argmax_set(r["pi_eq"], r["legal_mask"]):
            n_disagree += 1
            margin = _gto_margin(r["pi_eq"], r["legal_mask"])
            if margin >= GTO_LOPSIDED_THRESHOLD:
                n_disagree_lop += 1
            else:
                n_disagree_near += 1
    frac_lop = (n_disagree_lop / n_disagree) if n_disagree > 0 else float("nan")

    return {
        "cell_id": cell_id,
        "n_matches": len(dbars),
        "n_opp_decisions": n_dec,
        "mean_dbar": mean_dbar,
        "median_dbar": median_dbar,
        "mean_g_at_kappa": float(np.tanh(kappa * mean_dbar)),
        "median_g_at_kappa": float(np.tanh(kappa * median_dbar)),
        "overall_acc": overall_acc,
        "conditional_acc": cond_acc, "n_conditional": n_cond,
        "choice_reveal_acc": cr_acc, "n_choice_reveal": n_cr,
        "n_disagreement": int(n_disagree),
        "n_disagreement_lopsided": int(n_disagree_lop),
        "n_disagreement_near_tied": int(n_disagree_near),
        "frac_disagreement_lopsided": frac_lop,
    }


def revised_separation_gate(cell_table, e0_inherited, *, kappa):
    maniac = next(c for c in cell_table if c["cell_id"] == "always_raise@s1.00")
    uniform_max = max(
        (c for c in cell_table if c["cell_id"].startswith("random_uniform")),
        key=lambda c: c["mean_dbar"])
    Dm = maniac["mean_dbar"]
    Du = uniform_max["mean_dbar"]
    sep_ratio = (Dm / Du) if Du > 0 else math.inf
    g_maniac = math.tanh(kappa * Dm)
    cond_acc_maniac = maniac["choice_reveal_acc"]

    ok_sep = sep_ratio >= SEP_RATIO_MIN
    if g_maniac >= MANIAC_G_MIN:
        ok_g = True
        kappa_final = kappa
        recal_verdict = "kappa_stands"
    else:
        kappa_a = math.atanh(min(RECAL_G_MANIAC, 0.999)) / Dm if Dm > 0 else math.inf
        kappa_b_cap = (math.atanh(min(RECAL_G_UNIFORM_MAX, 0.999)) / Du
                       if Du > 0 else math.inf)
        if kappa_a <= kappa_b_cap:
            ok_g = True
            kappa_final = kappa_a
            recal_verdict = "kappa_recalibrated"
        else:
            ok_g = False
            kappa_final = None
            recal_verdict = "no_valid_kappa"
    ok_cond_acc = (not math.isnan(cond_acc_maniac)
                   and cond_acc_maniac >= MANIAC_COND_ACC_MIN)
    ok_e0 = e0_inherited <= E0_BAR
    overall = ok_sep and ok_g and ok_cond_acc and ok_e0
    return {
        "verdict": "PASS" if overall else "FAIL",
        "E0_mbb_inherited": e0_inherited, "E0_bar": E0_BAR, "E0_ok": ok_e0,
        "D_maniac": Dm, "D_uniform_max_cell": uniform_max["cell_id"],
        "D_uniform_max": Du,
        "separation_ratio": sep_ratio,
        "separation_ratio_min": SEP_RATIO_MIN,
        "separation_ok": ok_sep,
        "maniac_g_at_kappa": g_maniac, "maniac_g_min": MANIAC_G_MIN,
        "kappa_initial": kappa, "kappa_final": kappa_final,
        "kappa_recal_verdict": recal_verdict, "g_ok": ok_g,
        "maniac_choice_reveal_acc": cond_acc_maniac,
        "maniac_choice_reveal_acc_min": MANIAC_COND_ACC_MIN,
        "cond_acc_ok": ok_cond_acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--n-hands-probe", type=int, default=50)
    ap.add_argument("--w0", type=float, default=2.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    net = AdaptivePolicyNet().to(device)
    net.load_state_dict(ckpt["state_dict"])
    n_params = sum(p.numel() for p in net.parameters())
    _log(f"[ckpt] loaded {args.ckpt}")
    _log(f"[ckpt] config: {ckpt.get('config', {})}")
    _log(f"[net]  AdaptivePolicyNet ({n_params} params, F_RAW={TT.F_RAW}, "
         f"F_STATS={TT.F_STATS})")

    anchor_fn = load_anchor(Path(args.anchor_dir))
    train_cells, test_cells = A.train_test_split()
    _log(f"[cells] train={len(train_cells)} test={len(test_cells)} "
         f"(held-out: {[c.id for c in test_cells]})")
    _log(f"[primitive] argmax-disagreement (D-pivot)")
    _log(f"[probe] n_hands={args.n_hands_probe}, w0={args.w0}, "
         f"kappa={args.kappa}, seed={args.seed}")
    _log("")

    rng = np.random.default_rng(args.seed + 1)
    cell_table = []
    t0_all = time.time()
    for ci, cell in enumerate(train_cells):
        t_cell = time.time()
        opp_fn = A.build_cell(cell)
        dbars = []
        recs = []
        for _ in range(args.n_hands_probe):
            for hero_seat in (0, 1):
                dbar, _, r = run_hand_collect(
                    net, anchor_fn, opp_fn, hero_seat,
                    w0=args.w0, device=device, rng=rng)
                dbars.append(dbar)
                recs.extend(r)
        m = cell_metrics(recs, cell.id, dbars, kappa=args.kappa)
        cell_table.append(m)
        _log(f"  [probe {ci+1}/{len(train_cells)}] {cell.id:<28} "
             f"D̄'={m['mean_dbar']:.4f} g={m['mean_g_at_kappa']:.4f} "
             f"acc_overall={m['overall_acc']:.3f} "
             f"acc_cond={m['conditional_acc']:.3f}(n={m['n_conditional']}) "
             f"acc_cr={m['choice_reveal_acc']:.3f}(n={m['n_choice_reveal']}) "
             f"disagrees={m['n_disagreement']} "
             f"(lop={m['n_disagreement_lopsided']}, near={m['n_disagreement_near_tied']}) "
             f"({time.time()-t_cell:.1f}s)")
    _log(f"[probe] done in {time.time()-t0_all:.1f}s")

    # Inherit E0 from the original cotrain metrics.json (unchanged by
    # read-primitive substitution).
    ckpt_dir = Path(args.ckpt).parent
    metrics_path = ckpt_dir / "metrics.json"
    if metrics_path.exists():
        m_prev = json.loads(metrics_path.read_text())
        e0_inherited = float(m_prev.get("E0_mbb_per_game", float("nan")))
    else:
        e0_inherited = float("nan")
    _log(f"[E0 inherited] {e0_inherited:.4f} mbb/g (unchanged — read-primitive "
         f"swap is inference-only, model weights identical)")

    gate = revised_separation_gate(cell_table, e0_inherited, kappa=args.kappa)
    _log("")
    _log("=== Revised separation gate (D-pivot, argmax primitive) ===")
    _log(f"  (i)   D̄'_maniac={gate['D_maniac']:.4f} vs "
         f"D̄'_uniform_max={gate['D_uniform_max']:.4f} "
         f"(@{gate['D_uniform_max_cell']}) "
         f"=> ratio {gate['separation_ratio']:.2f} "
         f"(>= {gate['separation_ratio_min']}) -> "
         f"{'OK' if gate['separation_ok'] else 'FAIL'}")
    _log(f"  (ii)  maniac g(κ={args.kappa})={gate['maniac_g_at_kappa']:.4f} "
         f"(>= {gate['maniac_g_min']}) -> verdict={gate['kappa_recal_verdict']} "
         f"(κ_final={gate['kappa_final']}) -> "
         f"{'OK' if gate['g_ok'] else 'FAIL'}")
    _log(f"  (iii) maniac choice-reveal acc"
         f"={gate['maniac_choice_reveal_acc']:.3f} "
         f"(>= {gate['maniac_choice_reveal_acc_min']}) -> "
         f"{'OK' if gate['cond_acc_ok'] else 'FAIL'}")
    _log(f"  (iv)  E0={gate['E0_mbb_inherited']:.4f} mbb/g (<= {gate['E0_bar']}) "
         f"-> {'OK' if gate['E0_ok'] else 'FAIL'}")
    _log(f"  OVERALL: {gate['verdict']}")

    out = {
        "ckpt": args.ckpt, "anchor_dir": args.anchor_dir,
        "primitive": "argmax_disagreement",
        "params": {"n_hands_probe": args.n_hands_probe, "w0": args.w0,
                    "kappa": args.kappa, "seed": args.seed,
                    "gto_lopsided_threshold": GTO_LOPSIDED_THRESHOLD},
        "E0_mbb_inherited": e0_inherited,
        "per_cell_table": cell_table,
        "separation_gate": gate,
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        _log(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
