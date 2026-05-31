"""S2(a) co-train: archetype-classifier opp-head — the fork-collapsing
experiment.

Sequence vs the canonical phase1 cotrain:
  - Architecture: AdaptivePolicyNetS2a (3 heads: policy, opp_action, opp_cell).
  - Training: 3 losses
       L = L_distill + lam_cell · L_cell + lam_action · L_action
    cell-CE provides identity-discriminative pressure on the shared trunk;
    action-CE (carryover from A+B) keeps the action head calibrated as the
    inference-time read source.
  - Read primitive (HEAD-DERIVED, no oracle):
       q_action_t = softmax(action_logits_t)[legal]
       d_t        = KL(q_action_t || pi_GTO_t)
       w_t        = 1 - H(q_action_t) / log(|legal|)
       D̄         = Σ(w_t · d_t) / (w0 + Σ w_t)        (same aggregator)
       g          = tanh(κ · D̄)
    No archetype policy and no δ table touch the read path.
  - Separation gate (iii) replaced: maniac CELL-classification accuracy ≥ 0.85
    (direct measure of identity supervision working).
  - Additional reported diagnostic: 9×9 confusion matrix (true vs predicted
    cell, MAP at hand-end) + maniac misclassification breakdown.

Run:
  cd ~/pokerbot && setsid .venv/bin/python -m scripts.leduc_phase1_s2a_cotrain \
      --anchor-dir runs/leduc_cfr_anchor_20260531_144405 \
      > /tmp/leduc_s2a_cotrain.log 2>&1 < /dev/null & disown

§8 leakage preflight A/B/C/D must be GREEN — HARD STOP.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pyspiel
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# --------------------------------------------------------------------------
# §8 preflight (HARD STOP)
# --------------------------------------------------------------------------

def gate_on_leakage_tests(test_file: str = "tests/test_leduc_token_no_leak.py"):
    """Run §8 A/B/C/D leakage tests as preflight. Hard fail if any fails."""
    print(f"[preflight] running pytest {test_file}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", test_file,
         "--tb=short", "--no-header"],
        capture_output=True, text=True,
    )
    summary = proc.stdout.strip().split("\n")[-1]
    print(summary, flush=True)
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
        print(proc.stderr, file=sys.stderr, flush=True)
        sys.exit(f"[preflight] §8 leakage tests FAILED — refusing to train.")
    print("[preflight] leakage tests GREEN — proceeding to model import",
          flush=True)


gate_on_leakage_tests()


# Safe to import model machinery now.
from src.leduc import archetypes as A
from src.leduc.adaptive import tokens as TT
from src.leduc.adaptive.model import (
    AdaptivePolicyNetS2a, collate, running_read,
)
from src.leduc.evaluate import exploitability_mbb

GAME = pyspiel.load_game("leduc_poker")
NEG_INF = -1e9


def _log(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Anchor loader
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Distill source — full-tree enumeration
# --------------------------------------------------------------------------

def enumerate_decision_examples(anchor_fn):
    examples = []

    def rec(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a in state.legal_actions():
                child = state.clone()
                child.apply_action(a)
                rec(child)
            return
        toks = TT.tokenize_decision(state)
        target = np.zeros(TT.NUM_ACTIONS, dtype=np.float32)
        for a, p in anchor_fn(state).items():
            target[a] = p
        examples.append({
            "tokens": toks["tokens"],
            "query_idx": toks["query_idx"],
            "legal_mask": toks["legal_mask"],
            "opp_stats": toks["opp_stats"],
            "target": target,
        })
        for a in state.legal_actions():
            child = state.clone()
            child.apply_action(a)
            rec(child)

    rec(GAME.new_initial_state())
    return examples


# --------------------------------------------------------------------------
# Aux source — noised-anchor vs archetype rollout pool
# --------------------------------------------------------------------------

def noised_anchor_action(anchor_fn, state, epsilon, rng):
    legal = state.legal_actions()
    n = len(legal)
    a_probs = anchor_fn(state)
    p = np.zeros(TT.NUM_ACTIONS, dtype=np.float64)
    for a, q in a_probs.items():
        p[a] = q
    p_explore = np.zeros(TT.NUM_ACTIONS, dtype=np.float64)
    for a in legal:
        p_explore[a] = (1.0 - epsilon) * p[a] + epsilon / n
    p_explore = p_explore / p_explore.sum()
    return int(rng.choice(TT.NUM_ACTIONS, p=p_explore))


def sample_from_dict(probs: dict, rng) -> int:
    keys = list(probs.keys())
    ps = np.array(list(probs.values()), dtype=np.float64)
    ps = ps / ps.sum()
    return int(rng.choice(keys, p=ps))


def generate_aux_pool(anchor_fn, train_cells, *, n_hands, epsilon, seed,
                      cell_sampling_weights=None, cell_to_idx=None):
    rng = np.random.default_rng(seed)
    cells = list(train_cells)
    held_out_ids = {f"{n}@s{s:.2f}" for n, s in A.HELD_OUT_DEFAULT}
    w = np.array([cell_sampling_weights.get(c.id, 1.0)
                  if cell_sampling_weights else 1.0
                  for c in cells], dtype=np.float64)
    w = w / w.sum()
    pool = []
    cell_counts = {c.id: 0 for c in cells}
    for h in range(n_hands):
        ci = int(rng.choice(len(cells), p=w))
        cell = cells[ci]
        cell_counts[cell.id] += 1
        assert cell.id not in held_out_ids, f"held-out cell leaked: {cell.id}"
        opp_fn = A.build_cell(cell)
        hero_seat = int(rng.integers(2))
        state = GAME.new_initial_state()
        latest_hero_query = None
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
                latest_hero_query = {
                    "tokens": toks["tokens"].copy(),
                    "query_idx": int(toks["query_idx"]),
                    "legal_mask": toks["legal_mask"].copy(),
                    "opp_stats": toks["opp_stats"].copy(),
                }
                a = noised_anchor_action(anchor_fn, state, epsilon, rng)
                state.apply_action(a)
            else:
                legal = state.legal_actions()
                opp_lm = np.zeros(TT.NUM_ACTIONS, dtype=bool)
                for la in legal:
                    opp_lm[la] = True
                op = opp_fn(state)
                a = sample_from_dict(op, rng)
                if latest_hero_query is not None:
                    pool.append({
                        **latest_hero_query,
                        "opp_legal_mask": opp_lm.copy(),
                        "observed_opp_action": a,
                        "cell_idx": int(cell_to_idx[cell.id]),
                        "cell_id": cell.id,
                    })
                state.apply_action(a)
    return pool, cell_counts


# --------------------------------------------------------------------------
# Co-train loop (3 losses)
# --------------------------------------------------------------------------

def co_train(net, distill_examples, aux_pool, *,
             epochs, steps_per_epoch, batch_distill, batch_aux,
             lr, lam_cell, lam_action,
             device, seed, ckpt_dir, ckpt_every):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * steps_per_epoch)
    rng = np.random.default_rng(seed)
    n_distill = len(distill_examples)
    n_aux = len(aux_pool)
    distill_targets = np.stack([e["target"] for e in distill_examples])
    aux_actions = np.array([p["observed_opp_action"] for p in aux_pool],
                           dtype=np.int64)
    aux_legal = np.stack([p["opp_legal_mask"] for p in aux_pool])
    aux_cell_idx = np.array([p["cell_idx"] for p in aux_pool], dtype=np.int64)
    log = []
    net.train()
    step = 0
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        ep_l_d = 0.0
        ep_l_cell = 0.0
        ep_l_action = 0.0
        ep_acc_cell = 0.0
        ep_acc_action = 0.0
        ep_count = 0
        for _ in range(steps_per_epoch):
            # Distill mini-batch
            di = rng.choice(n_distill, size=batch_distill, replace=False)
            db = [distill_examples[i] for i in di]
            packed_d = collate(db, device=device)
            t_d = torch.tensor(distill_targets[di], device=device)
            policy_raw, _, _, _ = net(
                packed_d["tokens"], packed_d["pad_mask"], packed_d["query_idx"],
                packed_d["legal_mask"], packed_d["opp_stats"],
            )
            log_p = torch.log(policy_raw + 1e-12)
            L_distill = -(t_d * log_p).sum(dim=-1).mean()

            # Aux mini-batch — feeds BOTH cell-CE and action-CE.
            ai = rng.choice(n_aux, size=batch_aux, replace=False)
            ab = [aux_pool[i] for i in ai]
            packed_a = collate(ab, device=device)
            _, cell_logits, action_logits, _ = net(
                packed_a["tokens"], packed_a["pad_mask"], packed_a["query_idx"],
                packed_a["legal_mask"], packed_a["opp_stats"],
            )
            # Cell-CE: no class weighting (recipe-A pool oversamples maniac).
            cell_t = torch.tensor(aux_cell_idx[ai], dtype=torch.long,
                                  device=device)
            L_cell = F.cross_entropy(cell_logits, cell_t)
            # Action-CE: masked over opp legal; no class weighting (recipe B off
            # for the cell target; for action carryover we use uniform CE to
            # match the head-derived read's expectation).
            opp_lm = torch.tensor(aux_legal[ai], dtype=torch.bool, device=device)
            action_logits_masked = action_logits.masked_fill(~opp_lm, NEG_INF)
            a_t = torch.tensor(aux_actions[ai], dtype=torch.long, device=device)
            L_action = F.cross_entropy(action_logits_masked, a_t)
            with torch.no_grad():
                pred_cell = cell_logits.argmax(dim=-1)
                pred_action = action_logits_masked.argmax(dim=-1)
                acc_cell = float((pred_cell == cell_t).float().mean())
                acc_action = float((pred_action == a_t).float().mean())

            L = L_distill + lam_cell * L_cell + lam_action * L_action
            opt.zero_grad()
            L.backward()
            opt.step()
            sched.step()

            ep_l_d += float(L_distill.detach())
            ep_l_cell += float(L_cell.detach())
            ep_l_action += float(L_action.detach())
            ep_acc_cell += acc_cell
            ep_acc_action += acc_action
            ep_count += 1
            step += 1
        dt = time.time() - t_ep
        rec = {"epoch": ep,
               "L_distill": ep_l_d / ep_count,
               "L_cell": ep_l_cell / ep_count,
               "L_action": ep_l_action / ep_count,
               "acc_cell": ep_acc_cell / ep_count,
               "acc_action": ep_acc_action / ep_count,
               "seconds": dt,
               "lr": sched.get_last_lr()[0]}
        log.append(rec)
        if ep == 1 or ep % max(1, epochs // 20) == 0 or ep == epochs:
            _log(f"  [s2a ep {ep:>3d}/{epochs}] L_d={rec['L_distill']:.4f} "
                 f"L_c={rec['L_cell']:.4f} L_a={rec['L_action']:.4f} "
                 f"acc_c={rec['acc_cell']:.3f} acc_a={rec['acc_action']:.3f} "
                 f"lr={rec['lr']:.2e} ({dt:.1f}s)")
        if ep % ckpt_every == 0:
            torch.save({"state_dict": net.state_dict(), "epoch": ep,
                        "phase": "phase1_s2a_inprogress"},
                       ckpt_dir / f"cotrain_ckpt_ep{ep:04d}.pt")
    return log


# --------------------------------------------------------------------------
# Read primitive + probe (HEAD-DERIVED, no oracle)
# --------------------------------------------------------------------------

def make_net_policy_callable(net, device):
    net.eval()
    def fn(state):
        toks = TT.tokenize_decision(state)
        packed = collate([toks], device=device)
        with torch.no_grad():
            policy_raw, _, _, _ = net(
                packed["tokens"], packed["pad_mask"], packed["query_idx"],
                packed["legal_mask"], packed["opp_stats"],
            )
        probs = policy_raw[0].cpu().numpy()
        return {int(a): float(probs[a]) for a in toks["legal_actions"]}
    return fn


def read_terms_head(q_action: np.ndarray, pi_eq: np.ndarray, legal_mask: np.ndarray):
    """KL(q_action || pi_eq) with entropy-sharpness weight — head-derived.
    No oracle path: q_action comes from the model's opp_head_action; pi_eq
    is the universal anchor (not opp-specific)."""
    eps = 1e-12
    legal = legal_mask.astype(bool)
    q = np.clip(q_action[legal], eps, 1.0)
    p = np.clip(pi_eq[legal], eps, 1.0)
    q = q / q.sum()
    p = p / p.sum()
    d_t = float(np.sum(q * (np.log(q) - np.log(p))))
    n_legal = int(legal.sum())
    if n_legal <= 1:
        w_t = 1.0
    else:
        H = -float(np.sum(q * np.log(q)))
        w_t = 1.0 - H / np.log(n_legal)
    return d_t, max(0.0, min(1.0, w_t))


def run_hand_s2a(net, anchor_fn, opp_fn, hero_seat, *, w0, device, rng):
    """Run one hand. At each opp decision, derive read from action_logits
    (head-derived) and snapshot cell_logits for classification accuracy."""
    state = GAME.new_initial_state()
    d_list, w_list = [], []
    cached_action_logits = None
    cached_cell_logits = None
    opp_records = []
    last_cell_logits = None  # for hand-end MAP cell classification
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
                policy_raw, cell_logits, action_logits, _ = net(
                    packed["tokens"], packed["pad_mask"], packed["query_idx"],
                    packed["legal_mask"], packed["opp_stats"],
                )
            cached_action_logits = action_logits[0].cpu().numpy()
            cached_cell_logits = cell_logits[0].cpu().numpy()
            last_cell_logits = cached_cell_logits.copy()
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
            if cached_action_logits is not None:
                lgt = cached_action_logits.astype(np.float64).copy()
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
            d_t, w_t = read_terms_head(q_t, pi_eq, lm)
            d_list.append(d_t)
            w_list.append(w_t)
            op = opp_fn(state)
            a = sample_from_dict(op, rng)
            # Record q (for action-acc / disagreement diagnostics), observed
            # action, and cell_logits snapshot at the most recent hero query.
            opp_records.append({
                "q": q_t.copy(),
                "pi_eq": pi_eq.copy(),
                "legal_mask": lm.copy(),
                "observed_action": int(a),
                "cell_logits": (cached_cell_logits.copy()
                                if cached_cell_logits is not None else None),
            })
            state.apply_action(a)
    return (running_read(d_list, w_list, w0=w0), len(d_list),
            opp_records, last_cell_logits)


def per_cell_table_s2a(net, anchor_fn, train_cells, cell_to_idx, *, w0, kappa,
                       n_hands, device, seed):
    rng = np.random.default_rng(seed)
    out = []
    net.eval()
    n_cells = len(train_cells)
    # 9x9 confusion: rows = true cell, cols = predicted (MAP at hand end)
    confusion = np.zeros((n_cells, n_cells), dtype=np.int64)
    for ci, cell in enumerate(train_cells):
        t_cell = time.time()
        opp_fn = A.build_cell(cell)
        true_idx = cell_to_idx[cell.id]
        dbars = []
        all_records = []
        # per-cell cell-classification accuracy across ALL opp decisions
        cell_correct_decision_level = 0
        cell_total_decision_level = 0
        # per-hand MAP correctness (end-of-hand cell head argmax)
        hand_map_correct = 0
        hand_map_total = 0
        for _ in range(n_hands):
            for hero_seat in (0, 1):
                dbar, _, recs, last_cl = run_hand_s2a(
                    net, anchor_fn, opp_fn, hero_seat,
                    w0=w0, device=device, rng=rng)
                dbars.append(dbar)
                all_records.extend(recs)
                # decision-level cell acc: at every opp decision, was the
                # cached cell head MAP correct?
                for r in recs:
                    if r["cell_logits"] is None:
                        continue
                    pred = int(np.argmax(r["cell_logits"]))
                    cell_total_decision_level += 1
                    if pred == true_idx:
                        cell_correct_decision_level += 1
                # hand-end MAP
                if last_cl is not None:
                    pred_hand = int(np.argmax(last_cl))
                    hand_map_total += 1
                    confusion[true_idx, pred_hand] += 1
                    if pred_hand == true_idx:
                        hand_map_correct += 1
        # action-head diagnostics (action acc; unused as a gate but reported)
        if all_records:
            preds_q = np.stack([r["q"] for r in all_records])
            acts = np.array([r["observed_action"] for r in all_records])
            action_acc = float((preds_q.argmax(axis=-1) == acts).mean())
        else:
            action_acc = float("nan")
        cell_acc_decisions = (cell_correct_decision_level
                              / max(1, cell_total_decision_level))
        cell_acc_hand_end = hand_map_correct / max(1, hand_map_total)
        out.append({
            "cell_id": cell.id,
            "archetype": cell.name,
            "strength": cell.strength,
            "n_matches": len(dbars),
            "n_opp_decisions": len(all_records),
            "mean_dbar": float(np.mean(dbars)),
            "median_dbar": float(np.median(dbars)),
            "mean_g_at_kappa": float(np.tanh(kappa * np.mean(dbars))),
            "median_g_at_kappa": float(np.tanh(kappa * np.median(dbars))),
            "cell_acc_decisions": float(cell_acc_decisions),
            "cell_acc_hand_end": float(cell_acc_hand_end),
            "action_acc_decisions": action_acc,
        })
        _log(f"  [probe {ci+1}/{n_cells}] {cell.id:<28} "
             f"D̄={np.mean(dbars):.4f} g={np.tanh(kappa*np.mean(dbars)):.4f} "
             f"cell_acc_dec={cell_acc_decisions:.3f} "
             f"cell_acc_end={cell_acc_hand_end:.3f} "
             f"act_acc={action_acc:.3f} ({time.time()-t_cell:.1f}s)")
    return out, confusion


def separation_gate(cell_table, e0, *, kappa, e0_bar, sep_ratio_min,
                    maniac_g_min, maniac_cell_acc_min,
                    recal_g_maniac, recal_g_uniform_max):
    maniac = next(c for c in cell_table if c["cell_id"] == "always_raise@s1.00")
    uniform_max = max(
        (c for c in cell_table if c["archetype"] == "random_uniform"),
        key=lambda c: c["mean_dbar"])
    Dm = maniac["mean_dbar"]
    Du = uniform_max["mean_dbar"]
    sep_ratio = (Dm / Du) if Du > 0 else math.inf
    g_maniac = math.tanh(kappa * Dm)
    maniac_cell_acc = maniac["cell_acc_hand_end"]

    ok_sep = sep_ratio >= sep_ratio_min
    if g_maniac >= maniac_g_min:
        ok_g = True
        kappa_final = kappa
        recal_verdict = "kappa_stands"
    else:
        kappa_a = (math.atanh(min(recal_g_maniac, 0.999)) / Dm
                   if Dm > 0 else math.inf)
        kappa_b_cap = (math.atanh(min(recal_g_uniform_max, 0.999)) / Du
                       if Du > 0 else math.inf)
        if kappa_a <= kappa_b_cap:
            ok_g = True
            kappa_final = kappa_a
            recal_verdict = "kappa_recalibrated"
        else:
            ok_g = False
            kappa_final = None
            recal_verdict = "no_valid_kappa"
    ok_cell_acc = maniac_cell_acc >= maniac_cell_acc_min
    ok_e0 = e0 <= e0_bar
    overall = ok_sep and ok_g and ok_cell_acc and ok_e0
    return {
        "verdict": "PASS" if overall else "FAIL",
        "E0_mbb": e0, "E0_bar": e0_bar, "E0_ok": ok_e0,
        "D_maniac": Dm, "D_uniform_max_cell": uniform_max["cell_id"],
        "D_uniform_max": Du,
        "separation_ratio": sep_ratio, "separation_ratio_min": sep_ratio_min,
        "separation_ok": ok_sep,
        "maniac_g_at_kappa": g_maniac, "maniac_g_min": maniac_g_min,
        "kappa_initial": kappa, "kappa_final": kappa_final,
        "kappa_recal_verdict": recal_verdict, "g_ok": ok_g,
        "maniac_cell_acc": maniac_cell_acc,
        "maniac_cell_acc_min": maniac_cell_acc_min,
        "cell_acc_ok": ok_cell_acc,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps-per-epoch", type=int, default=15)
    ap.add_argument("--batch-distill", type=int, default=256)
    ap.add_argument("--batch-aux", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-cell", type=float, default=0.3,
                    help="weight on cell-CE loss (primary identity signal).")
    ap.add_argument("--lam-action", type=float, default=0.3,
                    help="weight on action-CE loss (carryover from A+B; the "
                         "first lever if E0 breaks the floor).")
    ap.add_argument("--pool-hands", type=int, default=2560)
    ap.add_argument("--epsilon", type=float, default=0.20,
                    help="noised-anchor ε-mix with uniform.")
    ap.add_argument("--weight-maniac", type=float, default=3.0,
                    help="recipe-A cell-sampling weight for always_raise@s1.00.")
    ap.add_argument("--weight-raise-mid", type=float, default=2.0,
                    help="recipe-A cell-sampling weight for always_raise@s0.50.")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--w0", type=float, default=2.0)
    ap.add_argument("--n-hands-probe", type=int, default=50)
    ap.add_argument("--e0-bar", type=float, default=5.0)
    ap.add_argument("--sep-ratio-min", type=float, default=2.0)
    ap.add_argument("--maniac-g-min", type=float, default=0.40)
    ap.add_argument("--maniac-cell-acc-min", type=float, default=0.85)
    ap.add_argument("--recal-g-maniac", type=float, default=0.50)
    ap.add_argument("--recal-g-uniform-max", type=float, default=0.10)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"runs/leduc_phase1s2a_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir = {out}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    anchor_fn = load_anchor(Path(args.anchor_dir))

    # 1. Distill source.
    t0 = time.time()
    distill_examples = enumerate_decision_examples(anchor_fn)
    _log(f"[distill] {len(distill_examples)} examples in {time.time()-t0:.1f}s")

    # 2. Aux pool — noised-anchor vs archetype.
    train_cells, test_cells = A.train_test_split()
    assert len(train_cells) == 9 and len(test_cells) == 3
    cell_to_idx = {c.id: i for i, c in enumerate(train_cells)}
    idx_to_cell = {i: cid for cid, i in cell_to_idx.items()}
    _log(f"[cells] train={len(train_cells)} test={len(test_cells)} "
         f"(held-out: {[c.id for c in test_cells]})")
    _log(f"[cells] train cell ordering (for cell head):")
    for i, c in enumerate(train_cells):
        _log(f"    idx {i}: {c.id}")

    t0 = time.time()
    cell_sampling_weights = {
        "always_raise@s1.00": args.weight_maniac,
        "always_raise@s0.50": args.weight_raise_mid,
    }
    _log(f"[recipe A] cell sampling weights: {cell_sampling_weights} "
         "(others default to 1.0)")
    aux_pool, cell_counts = generate_aux_pool(
        anchor_fn, train_cells,
        n_hands=args.pool_hands, epsilon=args.epsilon,
        seed=args.seed + 7, cell_sampling_weights=cell_sampling_weights,
        cell_to_idx=cell_to_idx,
    )
    _log(f"[aux pool] {len(aux_pool)} tuples from {args.pool_hands} hands "
         f"(ε={args.epsilon}) in {time.time()-t0:.1f}s")
    for cid, n in sorted(cell_counts.items(), key=lambda x: -x[1]):
        _log(f"    cell hand count: {cid:<28} {n:>4d} "
             f"({100.0*n/args.pool_hands:.1f}%)")

    # Recipe-B cell class-balance is OFF (rationale: recipe-A oversamples
    # maniac; cell-balancing would undo that). Note for record.
    actions_arr = np.array([p["observed_opp_action"] for p in aux_pool],
                           dtype=np.int64)
    counts = np.bincount(actions_arr, minlength=TT.NUM_ACTIONS).astype(np.float64)
    cell_idx_arr = np.array([p["cell_idx"] for p in aux_pool], dtype=np.int64)
    cell_pool_counts = np.bincount(cell_idx_arr, minlength=len(train_cells))
    _log(f"[recipe B] aux action counts: fold={int(counts[0])} "
         f"call={int(counts[1])} raise={int(counts[2])} "
         f"(total={int(counts.sum())})")
    _log(f"[recipe B] cell class-balance OFF (recipe-A maniac oversample "
         f"preserved)")
    _log(f"[aux pool cell-tuple distribution]:")
    for i, n in enumerate(cell_pool_counts):
        _log(f"    idx {i} ({idx_to_cell[i]:<28}): {n} tuples "
             f"({100*n/len(aux_pool):.1f}%)")

    # 3. Fresh S2(a) net.
    net = AdaptivePolicyNetS2a(n_cells=len(train_cells)).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    _log(f"[net] AdaptivePolicyNetS2a ({n_params} params, "
         f"F_RAW={TT.F_RAW}, F_STATS={TT.F_STATS}, n_cells={len(train_cells)})")
    _log(f"[losses] L = L_distill + {args.lam_cell}·L_cell + "
         f"{args.lam_action}·L_action")

    # 4. Co-train.
    t0 = time.time()
    train_log = co_train(net, distill_examples, aux_pool,
                          epochs=args.epochs,
                          steps_per_epoch=args.steps_per_epoch,
                          batch_distill=args.batch_distill,
                          batch_aux=args.batch_aux,
                          lr=args.lr,
                          lam_cell=args.lam_cell,
                          lam_action=args.lam_action,
                          device=device, seed=args.seed,
                          ckpt_dir=out, ckpt_every=args.ckpt_every)
    t_train = time.time() - t0
    _log(f"[s2a cotrain] {args.epochs} epochs × {args.steps_per_epoch} steps "
         f"in {t_train:.1f}s")

    # 5a. E0 (PROMINENT).
    t0 = time.time()
    e0 = exploitability_mbb(GAME, make_net_policy_callable(net, device))
    _log("")
    _log(f"[E0] {e0:.4f} mbb/g (bar ≤ {args.e0_bar}) in {time.time()-t0:.1f}s")
    if e0 > args.e0_bar:
        _log(f"[E0 WARN] floor broken (E0 > {args.e0_bar}); first lever per "
             f"plan is lowering --lam-action (currently {args.lam_action}); "
             f"not auto-acting — reporting verbatim.")

    # 5b. Per-cell probe + confusion matrix.
    t0 = time.time()
    cell_table, confusion = per_cell_table_s2a(
        net, anchor_fn, train_cells, cell_to_idx,
        w0=args.w0, kappa=args.kappa, n_hands=args.n_hands_probe,
        device=device, seed=args.seed + 1)
    _log(f"[probe] done in {time.time()-t0:.1f}s")

    # 6. Confusion matrix + maniac misclass.
    _log("")
    _log("=== 9×9 cell confusion (rows=true, cols=predicted; MAP @ hand-end) ===")
    col_labels = [idx_to_cell[i] for i in range(len(train_cells))]
    header = "  true \\ pred           " + "".join(f"{i:>5d}" for i in range(len(train_cells)))
    _log(header)
    for i in range(len(train_cells)):
        row = "  " + f"[{i}] {idx_to_cell[i]:<22}"
        for j in range(len(train_cells)):
            row += f"{confusion[i, j]:>5d}"
        _log(row)
    maniac_idx = cell_to_idx["always_raise@s1.00"]
    maniac_row = confusion[maniac_idx]
    total = int(maniac_row.sum())
    if total > 0:
        rank = np.argsort(-maniac_row)
        _log("")
        _log(f"=== Maniac (always_raise@s1.00) misclassification breakdown ===")
        _log(f"  total maniac hands = {total}")
        for j in rank[:5]:
            j = int(j)
            cnt = int(maniac_row[j])
            if cnt == 0:
                continue
            tag = " (← true)" if j == maniac_idx else ""
            _log(f"    pred = {idx_to_cell[j]:<28} {cnt:>4d} "
                 f"({100*cnt/total:.1f}%){tag}")

    # 7. Separation gate.
    gate = separation_gate(cell_table, e0,
                            kappa=args.kappa, e0_bar=args.e0_bar,
                            sep_ratio_min=args.sep_ratio_min,
                            maniac_g_min=args.maniac_g_min,
                            maniac_cell_acc_min=args.maniac_cell_acc_min,
                            recal_g_maniac=args.recal_g_maniac,
                            recal_g_uniform_max=args.recal_g_uniform_max)
    _log("")
    _log("=== S2(a) Separation gate ===")
    _log(f"  (i)   D̄_maniac={gate['D_maniac']:.4f} vs "
         f"D̄_uniform_max={gate['D_uniform_max']:.4f} "
         f"(@{gate['D_uniform_max_cell']}) "
         f"=> ratio {gate['separation_ratio']:.2f} "
         f"(>= {gate['separation_ratio_min']}) -> "
         f"{'OK' if gate['separation_ok'] else 'FAIL'}")
    _log(f"  (ii)  maniac g(κ={args.kappa})={gate['maniac_g_at_kappa']:.4f} "
         f"(>= {gate['maniac_g_min']}) -> verdict={gate['kappa_recal_verdict']} "
         f"(κ_final={gate['kappa_final']}) -> "
         f"{'OK' if gate['g_ok'] else 'FAIL'}")
    _log(f"  (iii) maniac CELL-classification acc={gate['maniac_cell_acc']:.3f} "
         f"(>= {gate['maniac_cell_acc_min']}) -> "
         f"{'OK' if gate['cell_acc_ok'] else 'FAIL'}")
    _log(f"  (iv)  E0={gate['E0_mbb']:.4f} mbb/g (<= {gate['E0_bar']}) -> "
         f"{'OK' if gate['E0_ok'] else 'FAIL'}")
    _log(f"  OVERALL: {gate['verdict']}")

    # 8. Decision branch (named plainly).
    _log("")
    _log("=== DECISION BRANCH ===")
    if gate["verdict"] == "PASS":
        _log("  PASS — training-target was the fix; identity-supervised trunk "
             "drives a head-derived read that separates per-hand.")
        _log("  → GREEN to scale to 6-max with corrected (cell-classifier) "
             "head. Mechanism transfers by construction (no oracle in read "
             "path).")
    else:
        _log("  FAIL — S1 CONFIRMED.")
        _log("  Even with the most direct identity supervision (cell-CE) on "
             "the trunk and a head-derived read, Leduc cannot per-hand-"
             "separate the maniac from uniform within the (i)/(ii)/(iii) "
             "bars.")
        _log("  → STOP proving on Leduc. Move to 6-max with this S2(a) "
             "negative as design input.")

    # 9. Save.
    torch.save({"state_dict": net.state_dict(),
                "config": {"d_model": 64, "nhead": 4, "num_layers": 2,
                            "dim_ff": 128, "max_seq": 128,
                            "n_cells": len(train_cells),
                            "phase": "phase1_s2a_complete",
                            "anchor_dir": str(args.anchor_dir),
                            "epochs": args.epochs,
                            "epsilon_pool": args.epsilon,
                            "lam_cell": args.lam_cell,
                            "lam_action": args.lam_action,
                            "cell_to_idx": cell_to_idx},
                "separation_gate": gate},
               out / "phase1_s2a.pt")
    with open(out / "aux_pool.pkl", "wb") as f:
        pickle.dump({"epsilon": args.epsilon, "n_hands": args.pool_hands,
                     "n_tuples": len(aux_pool), "seed": args.seed + 7,
                     "cell_to_idx": cell_to_idx}, f)

    def _jsafe(x):
        if isinstance(x, dict):
            return {k: _jsafe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_jsafe(v) for v in x]
        if isinstance(x, np.ndarray):
            return _jsafe(x.tolist())
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x

    metrics = {
        "phase": "phase1_s2a",
        "anchor_dir": str(args.anchor_dir),
        "anchor_exploitability_mbb": 0.12857580807822816,
        "schema": {"F_RAW": TT.F_RAW, "F_STATS": TT.F_STATS,
                    "stats_manifest_sha": TT.STATS_MANIFEST_SHA,
                    "token_manifest_sha": TT.TOKEN_MANIFEST_SHA},
        "training": {
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_distill": args.batch_distill,
            "batch_aux": args.batch_aux,
            "lr": args.lr,
            "lam_cell": args.lam_cell, "lam_action": args.lam_action,
            "pool_hands": args.pool_hands,
            "epsilon_noised_anchor": args.epsilon,
            "n_distill_examples": len(distill_examples),
            "n_aux_tuples": len(aux_pool),
            "recipe_A_cell_weights": cell_sampling_weights,
            "recipe_A_realized_cell_counts": cell_counts,
            "recipe_B_cell_class_balance_on": False,
            "cell_to_idx": cell_to_idx,
            "n_params": n_params,
        },
        "E0_mbb_per_game": e0,
        "read_primitive": "head_derived_kl_action",
        "per_cell_table": cell_table,
        "cell_confusion": confusion.tolist(),
        "separation_gate": gate,
        "seconds": {"train": t_train},
    }
    (out / "metrics.json").write_text(json.dumps(_jsafe(metrics), indent=2))
    (out / "train_log.json").write_text(json.dumps(_jsafe(train_log), indent=2))
    _log(f"\n[saved] {out}/phase1_s2a.pt + metrics.json + train_log.json "
         f"+ aux_pool.pkl")
    _log(f"[STOP] decision branch reported. STOP for user review.")


if __name__ == "__main__":
    main()
