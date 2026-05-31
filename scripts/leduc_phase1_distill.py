"""Phase 1 of the Leduc adaptive-policy proof — distill to anchor + exit gate.

Per arch doc §12.1 + §12.6 (HARD STOP #2):

  1. Enumerate every decision node of leduc_poker; tokenize via the §2 token
     layout; target = the CFR+ anchor's policy at the info state.
  2. Train AdaptivePolicyNet's policy head with forward KL
        L = E[ -Σ_a legal target[a] · log net_probs[a] ]
     Opp-model aux is OFF (λ = 0) per §12.1 — self-play vs the anchor gives no
     informative opp signal.
  3. Compute E0 = exact best-response exploitability (mbb/g) of the distilled
     net's policy callable. EXIT GATE: E0 ≤ τ_distill (default 5 mbb/g).
  4. Per-cell D̄ probe (§12.6 #1): for each of the 9 train cells, run hands
     with hero = distilled net and opp = archetype. q_t = softmax(opp_head
     logits) from the most recent hero forward pass; π_eq = anchor at opp
     state; D̄ = Σ w_t d_t / (w0 + Σ w_t); g = tanh(κ · D̄).
  5. Save checkpoint + metrics; STOP for user before Phase 2.

Run:
  cd ~/pokerbot && .venv/bin/python -m scripts.leduc_phase1_distill \
      --anchor-dir runs/leduc_cfr_anchor_20260531_144405
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Single-thread BEFORE importing torch — small transformer + small batches
# get DESTROYED by intra-op threading on CPU; previous run did ~100 min of
# nothing at 9 cores. Pin to 1 thread for predictable speed.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pyspiel
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def _log(msg):
    print(msg, flush=True)

from src.leduc import archetypes as A
from src.leduc.adaptive import tokens as TT
from src.leduc.adaptive.model import (
    AdaptivePolicyNet,
    collate,
    read_terms,
    running_read,
)
from src.leduc.evaluate import exploitability_mbb

GAME = pyspiel.load_game("leduc_poker")


# --------------------------------------------------------------------------
# Anchor loader (matches scripts/eval_anchor_vs_archetypes.py's interface)
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
# Step 1 — enumerate every decision node.
# --------------------------------------------------------------------------

def enumerate_decision_examples(anchor_fn):
    """Walk Leduc's full game tree. Emit one (tokens, target) example per
    (chance-history, decision-node) pair. §8 guarantees tokens are identical
    across paths differing only in opp's hole card — duplicates are kept;
    they give the model uniform learning signal across all reach paths."""
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
            "info_state": toks["info_state"],
            "current_player": state.current_player(),
        })
        for a in state.legal_actions():
            child = state.clone()
            child.apply_action(a)
            rec(child)

    rec(GAME.new_initial_state())
    return examples


# --------------------------------------------------------------------------
# Step 2 — forward-KL distillation.
# --------------------------------------------------------------------------

def distill(net, examples, *, epochs, batch_size, lr, device, seed=0):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)
    n = len(examples)
    targets_all = np.stack([e["target"] for e in examples])
    log = []
    net.train()
    for ep in range(epochs):
        idx = rng.permutation(n)
        ep_loss = 0.0
        ep_count = 0
        for s in range(0, n, batch_size):
            bi = idx[s:s + batch_size]
            batch = [examples[i] for i in bi]
            packed = collate(batch, device=device)
            t = torch.tensor(targets_all[bi], device=device)
            policy_raw, _, _ = net(
                packed["tokens"], packed["pad_mask"], packed["query_idx"],
                packed["legal_mask"], packed["opp_stats"],
            )
            log_p = torch.log(policy_raw + 1e-12)
            loss = -(t * log_p).sum(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach()) * len(batch)
            ep_count += len(batch)
        sched.step()
        avg = ep_loss / ep_count
        log.append({"epoch": ep, "loss": avg, "lr": sched.get_last_lr()[0]})
        if ep == 0 or (ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1:
            _log(f"  [distill ep {ep+1:>4d}/{epochs}] loss={avg:.6f} "
                 f"lr={sched.get_last_lr()[0]:.2e}")
    return log


# --------------------------------------------------------------------------
# Step 3 — E0 (BR exploitability of distilled net).
# --------------------------------------------------------------------------

def make_net_policy_callable(net, device):
    """Callable(state) -> {a: prob} backed by the net's policy head (with the
    legal-action mask already applied via the masked softmax)."""
    net.eval()

    def fn(state):
        toks = TT.tokenize_decision(state)
        packed = collate([toks], device=device)
        with torch.no_grad():
            policy_raw, _, _ = net(
                packed["tokens"], packed["pad_mask"], packed["query_idx"],
                packed["legal_mask"], packed["opp_stats"],
            )
        probs = policy_raw[0].cpu().numpy()
        return {int(a): float(probs[a]) for a in toks["legal_actions"]}

    return fn


# --------------------------------------------------------------------------
# Step 4 — per-cell D̄ probe (§12.6 #1).
# --------------------------------------------------------------------------

def run_hand_dbar(net, anchor_fn, opp_fn, hero_seat, *, w0, device, rng):
    """Hero = distilled net (no gate, no blend — Phase-1 policy head is the
    distilled approximation of the anchor). Opp = archetype. We read opp_head
    on every hero forward pass and accumulate the §12.3 D̄."""
    state = GAME.new_initial_state()
    d_list, w_list = [], []
    cached_opp_logits = None
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
                lgt[~lm] = -1e9
                lgt -= lgt.max()
                e = np.exp(lgt) * lm
                q_t = e / e.sum()
            else:
                q_t = lm.astype(np.float64) / lm.sum()
            ap = anchor_fn(state)
            pi_eq = np.zeros(TT.NUM_ACTIONS, dtype=np.float64)
            for a, p in ap.items():
                pi_eq[a] = p
            d_t, w_t = read_terms(q_t, pi_eq, lm)
            d_list.append(d_t)
            w_list.append(w_t)
            op = opp_fn(state)
            acts = list(op.keys())
            ps = np.array(list(op.values()))
            state.apply_action(int(rng.choice(acts, p=ps / ps.sum())))
    return running_read(d_list, w_list, w0=w0), len(d_list), d_list, w_list


def per_cell_dbar(net, anchor_fn, train_cells, *, w0, kappa, n_hands, device, seed):
    rng = np.random.default_rng(seed)
    out = []
    net.eval()
    for ci, cell in enumerate(train_cells):
        t_cell = time.time()
        opp_fn = A.build_cell(cell)
        dbars = []
        n_dec = []
        d_pool = []
        w_pool = []
        for _ in range(n_hands):
            for hero_seat in (0, 1):
                dbar, nd, dl, wl = run_hand_dbar(
                    net, anchor_fn, opp_fn, hero_seat,
                    w0=w0, device=device, rng=rng,
                )
                dbars.append(dbar)
                n_dec.append(nd)
                d_pool.extend(dl)
                w_pool.extend(wl)
        _log(f"  [D̄ cell {ci+1}/{len(train_cells)}] {cell.id:<28} "
             f"in {time.time()-t_cell:.1f}s, mean_D̄={float(np.mean(dbars)):.4f}")
        mean_dbar = float(np.mean(dbars))
        med_dbar = float(np.median(dbars))
        out.append({
            "cell_id": cell.id,
            "archetype": cell.name,
            "strength": cell.strength,
            "n_matches": len(dbars),
            "mean_dbar": mean_dbar,
            "median_dbar": med_dbar,
            "mean_g_at_kappa": float(np.tanh(kappa * mean_dbar)),
            "median_g_at_kappa": float(np.tanh(kappa * med_dbar)),
            "mean_opp_decisions_per_match": float(np.mean(n_dec)),
            "raw_d_mean": float(np.mean(d_pool)) if d_pool else 0.0,
            "raw_w_mean": float(np.mean(w_pool)) if w_pool else 0.0,
        })
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-hands", type=int, default=200,
                    help="hands per (cell, seat) for D̄")
    ap.add_argument("--w0", type=float, default=2.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--tau-distill", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"runs/leduc_phase1_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir = {out}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    anchor_fn = load_anchor(Path(args.anchor_dir))

    t0 = time.time()
    examples = enumerate_decision_examples(anchor_fn)
    t_enum = time.time() - t0
    n_unique_info = len({e["info_state"] for e in examples})
    _log(f"[enum] {len(examples)} decision examples "
          f"({n_unique_info} unique info states) in {t_enum:.1f}s")

    net = AdaptivePolicyNet().to(device)
    t0 = time.time()
    train_log = distill(net, examples,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, device=device, seed=args.seed)
    t_train = time.time() - t0
    final_loss = train_log[-1]["loss"]
    _log(f"[distill] {args.epochs} epochs in {t_train:.1f}s, "
          f"final loss = {final_loss:.6f}")

    t0 = time.time()
    e0 = exploitability_mbb(GAME, make_net_policy_callable(net, device))
    t_e0 = time.time() - t0
    e0_passed = e0 <= args.tau_distill
    _log(f"[E0] {e0:.4f} mbb/g (vs anchor 0.1286, τ_distill {args.tau_distill}) "
          f"in {t_e0:.1f}s — {'PASS' if e0_passed else 'FAIL'}")

    train_cells, test_cells = A.train_test_split()
    _log(f"[D̄] {len(train_cells)} train cells × {args.n_hands} hands × 2 seats; "
          f"κ={args.kappa}, w0={args.w0}")
    t0 = time.time()
    cell_table = per_cell_dbar(net, anchor_fn, train_cells,
                               w0=args.w0, kappa=args.kappa,
                               n_hands=args.n_hands, device=device,
                               seed=args.seed + 1)
    t_dbar = time.time() - t0
    _log(f"[D̄] computed in {t_dbar:.1f}s")
    _log("    cell                        mean_D̄    median_D̄  mean_g    median_g")
    for c in sorted(cell_table, key=lambda x: -x["mean_dbar"]):
        _log(f"    {c['cell_id']:<28} {c['mean_dbar']:>8.4f}  "
              f"{c['median_dbar']:>8.4f}  {c['mean_g_at_kappa']:>7.4f}  "
              f"{c['median_g_at_kappa']:>7.4f}")
    maniac = next(c for c in cell_table if c["cell_id"] == "always_raise@s1.00")
    maniac_lit = maniac["mean_g_at_kappa"] >= 0.1  # lit ≈ tanh(0.1) read magnitude
    _log(f"[gate-scale] maniac always_raise@s1.00: "
          f"mean D̄={maniac['mean_dbar']:.4f}, mean g={maniac['mean_g_at_kappa']:.4f} "
          f"— {'GATE LIFTS OFF ZERO' if maniac_lit else 'GATE STUCK NEAR ZERO'}")

    torch.save({
        "state_dict": net.state_dict(),
        "config": {
            "d_model": 64, "nhead": 4, "num_layers": 2,
            "dim_ff": 128, "max_seq": 128,
            "phase": "phase1_distill", "anchor_dir": str(args.anchor_dir),
            "epochs": args.epochs, "lr": args.lr,
        },
    }, out / "phase1_distilled.pt")
    metrics = {
        "anchor_dir": str(args.anchor_dir),
        "anchor_exploitability_mbb": 0.12857580807822816,
        "n_examples": len(examples),
        "n_unique_info_states": n_unique_info,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_distill_loss": final_loss,
        "E0_mbb_per_game": e0,
        "tau_distill_mbb": args.tau_distill,
        "exit_gate_passed": bool(e0_passed),
        "n_hands_dbar_per_cell_per_seat": args.n_hands,
        "w0": args.w0,
        "kappa": args.kappa,
        "per_cell_dbar": cell_table,
        "maniac_gate_lifts_off_zero": bool(maniac_lit),
        "seconds": {"enum": t_enum, "train": t_train, "E0": t_e0, "dbar": t_dbar},
        "opp_head_status": (
            "UNTRAINED — Phase-1 distillation trains policy_head only; "
            "opp-aux λ=0 per arch §12.1 because self-play anchor-vs-anchor "
            "gives no informative opp signal. D̄ values below are reads from "
            "an effectively random opp_head."
        ),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "train_log.json").write_text(json.dumps(train_log, indent=2))
    _log(f"[saved] {out}/phase1_distilled.pt + metrics.json + train_log.json")


if __name__ == "__main__":
    main()
