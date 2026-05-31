"""Phase 1 (co-train) for the Leduc adaptive-policy proof.

Replaces the original distill-then-warmup sequence. Distillation + opp-aux
are trained TOGETHER from scratch on a shared trunk, so the trunk learns
cell-distinguishing representation alongside GTO reproduction.

Sequence:
  0. Pre-flight: pytest tests/test_leduc_token_no_leak.py must exit 0.
     (HARD BLOCKER — schema-change re-verification of §8.)
  1. Build distill set via full-tree enumeration (3780 examples, anchor target).
  2. Pre-generate aux rollout pool (default 2560 hands):
       hero plays NOISED ANCHOR (ε=0.20 uniform mix), opp plays archetype
       (cell ~ Uniform({9 train cells}); seats 50/50).
       At each opp decision following >=1 hero decision, record the inputs
       the net SAW at the most recent hero forward, plus the observed opp
       action and the opp legal mask. (Combined is NOT cached — recomputed
       at training time so the trunk's gradient flow is honest.)
  3. Initialize fresh AdaptivePolicyNet (v2 stats schema).
  4. Co-train K=200 epochs × S steps/epoch. Per step:
       distill_mb (256) -> L_distill = forward-KL
       aux_mb     (256) -> L_aux     = masked CE(opp_head, observed)
       L_total = L_distill + λ · L_aux,  λ = 0.3 (signed).
  5. E0(gate=0) + per-cell D̄/g/accuracy probe (same as Phase-2 warmup probe).
  6. Apply separation gate:
       (i)   D̄_maniac > 2× D̄_uniform_max
       (ii)  maniac g(κ=1.0) ≥ 0.40 OR valid two-condition κ_new
       (iii) maniac opp_head accuracy ≥ 0.85
       (iv)  E0 ≤ 5 mbb/g
     STOP for user review regardless of outcome.

Run:
  cd ~/pokerbot && setsid .venv/bin/python -m scripts.leduc_phase1_cotrain \
      --anchor-dir runs/leduc_cfr_anchor_20260531_144405 \
      > /tmp/leduc_phase1_cotrain.log 2>&1 < /dev/null & disown
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


def _log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Pre-flight: leakage tests must pass before importing the model.
# --------------------------------------------------------------------------

def gate_on_leakage_tests(test_file: str = "tests/test_leduc_token_no_leak.py"):
    _log(f"[preflight] running pytest {test_file}")
    res = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-q",
         "--tb=short", "--no-header"],
        cwd=str(Path.cwd()),
        capture_output=True, text=True,
    )
    _log(res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "(no output)")
    if res.returncode != 0:
        _log("[preflight] LEAKAGE TESTS FAILED — ABORTING")
        _log(res.stdout)
        _log(res.stderr)
        sys.exit(2)
    _log("[preflight] leakage tests GREEN — proceeding to model import")


gate_on_leakage_tests()


# Safe to import model machinery now.
from src.leduc import archetypes as A
from src.leduc.adaptive import tokens as TT
from src.leduc.adaptive.model import (
    AdaptivePolicyNet,
    collate,
    read_terms,
    read_terms_argmax,
    running_read,
)

READ_PRIMITIVES = {"kl": read_terms, "argmax": read_terms_argmax}
from src.leduc.evaluate import exploitability_mbb

GAME = pyspiel.load_game("leduc_poker")
NEG_INF = -1e9


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
# Distill source — full-tree enumeration (same as Phase 1)
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
    # ε-mix with uniform over legal.
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
                      cell_sampling_weights=None):
    """Roll out n_hands of (noised-anchor hero, archetype opp). At each opp
    decision following >=1 hero decision, record (hero_query_inputs,
    observed_opp_action, opp_legal_mask) for later aux loss recomputation.

    `cell_sampling_weights` is an optional dict {cell.id: weight}. Cells not
    in the dict default to 1.0. Used for recipe-A archetype oversampling
    (boost raise-heavy cells so raise actions aren't drowned by cap-hit calls)."""
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
                    })
                state.apply_action(a)
    return pool, cell_counts


# --------------------------------------------------------------------------
# Co-train loop
# --------------------------------------------------------------------------

def co_train(net, distill_examples, aux_pool, *,
             epochs, steps_per_epoch, batch_distill, batch_aux, lr, lam,
             device, seed, ckpt_dir, ckpt_every, class_weights=None):
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
    log = []
    net.train()
    step = 0
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        ep_l_distill = 0.0
        ep_l_aux = 0.0
        ep_acc_aux = 0.0
        ep_count = 0
        for s in range(steps_per_epoch):
            # Distill mini-batch
            di = rng.choice(n_distill, size=batch_distill, replace=False)
            db = [distill_examples[i] for i in di]
            packed_d = collate(db, device=device)
            t_d = torch.tensor(distill_targets[di], device=device)
            policy_raw, _, _ = net(
                packed_d["tokens"], packed_d["pad_mask"], packed_d["query_idx"],
                packed_d["legal_mask"], packed_d["opp_stats"],
            )
            log_p = torch.log(policy_raw + 1e-12)
            L_distill = -(t_d * log_p).sum(dim=-1).mean()

            # Aux mini-batch
            ai = rng.choice(n_aux, size=batch_aux, replace=False)
            ab = [aux_pool[i] for i in ai]
            packed_a = collate(ab, device=device)
            _, opp_logits, _ = net(
                packed_a["tokens"], packed_a["pad_mask"], packed_a["query_idx"],
                packed_a["legal_mask"], packed_a["opp_stats"],
            )
            opp_lm = torch.tensor(aux_legal[ai], dtype=torch.bool, device=device)
            opp_logits = opp_logits.masked_fill(~opp_lm, NEG_INF)
            a_t = torch.tensor(aux_actions[ai], dtype=torch.long, device=device)
            L_aux = F.cross_entropy(opp_logits, a_t, weight=class_weights)
            with torch.no_grad():
                pred = opp_logits.argmax(dim=-1)
                acc = float((pred == a_t).float().mean())

            L = L_distill + lam * L_aux
            opt.zero_grad()
            L.backward()
            opt.step()
            sched.step()

            ep_l_distill += float(L_distill.detach())
            ep_l_aux += float(L_aux.detach())
            ep_acc_aux += acc
            ep_count += 1
            step += 1
        dt = time.time() - t_ep
        avg_d = ep_l_distill / ep_count
        avg_a = ep_l_aux / ep_count
        avg_acc = ep_acc_aux / ep_count
        log.append({"epoch": ep, "L_distill": avg_d, "L_aux": avg_a,
                    "acc_aux": avg_acc, "seconds": dt,
                    "lr": sched.get_last_lr()[0]})
        if ep == 1 or ep % max(1, epochs // 20) == 0 or ep == epochs:
            _log(f"  [cotrain ep {ep:>3d}/{epochs}] L_d={avg_d:.4f} "
                 f"L_a={avg_a:.4f} acc_a={avg_acc:.3f} "
                 f"lr={sched.get_last_lr()[0]:.2e} ({dt:.1f}s)")
        if ep % ckpt_every == 0:
            torch.save({"state_dict": net.state_dict(), "epoch": ep,
                        "phase": "phase1_cotrain_inprogress"},
                       ckpt_dir / f"cotrain_ckpt_ep{ep:04d}.pt")
    return log


# --------------------------------------------------------------------------
# E0 + D̄ probe (same shapes as Phase 1 / Phase 2 warmup)
# --------------------------------------------------------------------------

def make_net_policy_callable(net, device):
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


def run_hand_dbar(net, anchor_fn, opp_fn, hero_seat, *, w0, device, rng,
                  read_fn=read_terms):
    state = GAME.new_initial_state()
    d_list, w_list = [], []
    cached_opp_logits = None
    opp_records = []
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
            d_t, w_t = read_fn(q_t, pi_eq, lm)
            d_list.append(d_t)
            w_list.append(w_t)
            op = opp_fn(state)
            a = sample_from_dict(op, rng)
            opp_records.append((q_t.copy(), a, lm.copy()))
            state.apply_action(a)
    return running_read(d_list, w_list, w0=w0), len(d_list), opp_records


def per_cell_table(net, anchor_fn, train_cells, *, w0, kappa, n_hands,
                   device, seed, read_fn=read_terms):
    rng = np.random.default_rng(seed)
    out = []
    net.eval()
    for ci, cell in enumerate(train_cells):
        t_cell = time.time()
        opp_fn = A.build_cell(cell)
        dbars = []
        all_records = []
        for _ in range(n_hands):
            for hero_seat in (0, 1):
                dbar, _, recs = run_hand_dbar(
                    net, anchor_fn, opp_fn, hero_seat,
                    w0=w0, device=device, rng=rng, read_fn=read_fn)
                dbars.append(dbar)
                all_records.extend(recs)
        preds = np.stack([r[0] for r in all_records])
        acts = np.array([r[1] for r in all_records])
        acc = float((preds.argmax(axis=-1) == acts).mean())
        ce = float(-np.mean(np.log(np.clip(
            preds[np.arange(len(acts)), acts], 1e-12, 1.0))))
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
            "opp_head_accuracy": acc,
            "opp_head_ce": ce,
        })
        _log(f"  [probe {ci+1}/{len(train_cells)}] {cell.id:<28} "
             f"D̄_mean={np.mean(dbars):.4f} g={np.tanh(kappa*np.mean(dbars)):.4f} "
             f"acc={acc:.3f} ce={ce:.4f} ({time.time()-t_cell:.1f}s)")
    return out


# --------------------------------------------------------------------------
# Separation gate
# --------------------------------------------------------------------------

def separation_gate(cell_table, e0, *, kappa, e0_bar, sep_ratio_min,
                    maniac_g_min, maniac_acc_min,
                    recal_g_maniac, recal_g_uniform_max):
    maniac = next(c for c in cell_table if c["cell_id"] == "always_raise@s1.00")
    uniform_max = max((c for c in cell_table if c["archetype"] == "random_uniform"),
                      key=lambda c: c["mean_dbar"])
    Dm = maniac["mean_dbar"]
    Du = uniform_max["mean_dbar"]
    sep_ratio = (Dm / Du) if Du > 0 else math.inf
    g_maniac = math.tanh(kappa * Dm)
    acc_maniac = maniac["opp_head_accuracy"]

    # Condition (i)
    ok_sep = sep_ratio >= sep_ratio_min
    # Condition (ii) — direct or via recalibration
    if g_maniac >= maniac_g_min:
        ok_g = True
        kappa_final = kappa
        recal_verdict = "kappa_stands"
    else:
        kappa_a = math.atanh(recal_g_maniac) / Dm if Dm > 0 else math.inf
        kappa_b_cap = (math.atanh(recal_g_uniform_max) / Du
                       if Du > 0 else math.inf)
        if kappa_a <= kappa_b_cap:
            ok_g = True
            kappa_final = kappa_a
            recal_verdict = "kappa_recalibrated"
        else:
            ok_g = False
            kappa_final = None
            recal_verdict = "no_valid_kappa"
    # Condition (iii)
    ok_acc = acc_maniac >= maniac_acc_min
    # Condition (iv)
    ok_e0 = e0 <= e0_bar

    overall = ok_sep and ok_g and ok_acc and ok_e0

    return {
        "verdict": "PASS" if overall else "FAIL",
        "E0_mbb": e0,
        "E0_bar": e0_bar,
        "E0_ok": ok_e0,
        "D_maniac": Dm,
        "D_uniform_max_cell": uniform_max["cell_id"],
        "D_uniform_max": Du,
        "separation_ratio": sep_ratio,
        "separation_ratio_min": sep_ratio_min,
        "separation_ok": ok_sep,
        "maniac_g_at_kappa": g_maniac,
        "maniac_g_min": maniac_g_min,
        "kappa_initial": kappa,
        "kappa_final": kappa_final,
        "kappa_recal_verdict": recal_verdict,
        "g_ok": ok_g,
        "maniac_accuracy": acc_maniac,
        "maniac_accuracy_min": maniac_acc_min,
        "acc_ok": ok_acc,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps-per-epoch", type=int, default=15)
    ap.add_argument("--batch-distill", type=int, default=256)
    ap.add_argument("--batch-aux", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--pool-hands", type=int, default=2560)
    ap.add_argument("--epsilon", type=float, default=0.20,
                    help="hero noised-anchor uniform mix")
    # Recipe A: archetype oversampling (raise-heavy cells get boosted weight).
    ap.add_argument("--weight-maniac", type=float, default=3.0,
                    help="cell sampling weight for always_raise@s1.00")
    ap.add_argument("--weight-raise-mid", type=float, default=2.0,
                    help="cell sampling weight for always_raise@s0.50")
    # Recipe B: inverse-frequency class-balanced CE on the aux loss.
    ap.add_argument("--class-balance", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="weight aux CE by inverse action frequency")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--w0", type=float, default=2.0)
    ap.add_argument("--n-hands-probe", type=int, default=50)
    ap.add_argument("--e0-bar", type=float, default=5.0)
    ap.add_argument("--sep-ratio-min", type=float, default=2.0)
    ap.add_argument("--maniac-g-min", type=float, default=0.40)
    ap.add_argument("--maniac-acc-min", type=float, default=0.85)
    ap.add_argument("--recal-g-maniac", type=float, default=0.50)
    ap.add_argument("--recal-g-uniform-max", type=float, default=0.10)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--read-primitive", choices=sorted(READ_PRIMITIVES.keys()),
                    default="kl",
                    help="probe/inference read primitive. 'kl' = forward-KL with "
                         "entropy sharpness (Phase-1/2 default). 'argmax' = "
                         "binary argmax-disagreement with sharpness-margin (D-pivot).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"runs/leduc_phase1cotrain_{time.strftime('%Y%m%d_%H%M%S')}")
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
    _log(f"[cells] train={len(train_cells)} test={len(test_cells)} "
         f"(held-out: {[c.id for c in test_cells]})")
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
    )
    _log(f"[aux pool] {len(aux_pool)} tuples from {args.pool_hands} hands "
         f"(ε={args.epsilon}) in {time.time()-t0:.1f}s")
    for cid, n in sorted(cell_counts.items(), key=lambda x: -x[1]):
        _log(f"    cell hand count: {cid:<28} {n:>4d} "
             f"({100.0*n/args.pool_hands:.1f}%)")

    # Recipe B: inverse-frequency class weights from realized pool actions.
    actions_arr = np.array([p["observed_opp_action"] for p in aux_pool],
                           dtype=np.int64)
    counts = np.bincount(actions_arr, minlength=TT.NUM_ACTIONS).astype(np.float64)
    total = counts.sum()
    if args.class_balance:
        # sklearn-style: weight = N / (n_classes * n_a); avoid div0.
        cw = total / (TT.NUM_ACTIONS * np.maximum(counts, 1.0))
        class_weights_t = torch.tensor(cw, dtype=torch.float32, device=device)
    else:
        cw = np.ones(TT.NUM_ACTIONS, dtype=np.float64)
        class_weights_t = None
    _log(f"[recipe B] aux action counts: fold={int(counts[0])} "
         f"call={int(counts[1])} raise={int(counts[2])} "
         f"(total={int(total)})")
    _log(f"[recipe B] class weights: fold={cw[0]:.3f} call={cw[1]:.3f} "
         f"raise={cw[2]:.3f}  (class_balance={args.class_balance})")
    # Sanity: pool covers all 9 train cells but no held-out.
    # (Hard to assert per-cell since cell isn't kept; we already assert at gen.)

    # 3. Fresh net (v2 schema).
    net = AdaptivePolicyNet().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    _log(f"[net] AdaptivePolicyNet v2 ({n_params} params, "
         f"F_RAW={TT.F_RAW}, F_STATS={TT.F_STATS})")

    # 4. Co-train.
    t0 = time.time()
    train_log = co_train(net, distill_examples, aux_pool,
                          epochs=args.epochs,
                          steps_per_epoch=args.steps_per_epoch,
                          batch_distill=args.batch_distill,
                          batch_aux=args.batch_aux,
                          lr=args.lr, lam=args.lam, device=device,
                          seed=args.seed, ckpt_dir=out,
                          ckpt_every=args.ckpt_every,
                          class_weights=class_weights_t)
    t_train = time.time() - t0
    _log(f"[cotrain] {args.epochs} epochs × {args.steps_per_epoch} steps "
         f"in {t_train:.1f}s")

    # 5a. E0.
    t0 = time.time()
    e0 = exploitability_mbb(GAME, make_net_policy_callable(net, device))
    _log(f"[E0] {e0:.4f} mbb/g (bar ≤ {args.e0_bar}) in {time.time()-t0:.1f}s")

    # 5b. Per-cell probe.
    t0 = time.time()
    read_fn = READ_PRIMITIVES[args.read_primitive]
    _log(f"[probe] read primitive = {args.read_primitive}")
    cell_table = per_cell_table(net, anchor_fn, train_cells,
                                 w0=args.w0, kappa=args.kappa,
                                 n_hands=args.n_hands_probe,
                                 device=device, seed=args.seed + 1,
                                 read_fn=read_fn)
    _log(f"[probe] done in {time.time()-t0:.1f}s")

    # 6. Separation gate.
    gate = separation_gate(cell_table, e0,
                            kappa=args.kappa, e0_bar=args.e0_bar,
                            sep_ratio_min=args.sep_ratio_min,
                            maniac_g_min=args.maniac_g_min,
                            maniac_acc_min=args.maniac_acc_min,
                            recal_g_maniac=args.recal_g_maniac,
                            recal_g_uniform_max=args.recal_g_uniform_max)
    _log("")
    _log("=== Separation gate ===")
    _log(f"  (i)   D̄_maniac={gate['D_maniac']:.4f} vs "
         f"D̄_uniform_max={gate['D_uniform_max']:.4f} "
         f"(@{gate['D_uniform_max_cell']}) "
         f"=> ratio {gate['separation_ratio']:.2f} "
         f"(>= {gate['separation_ratio_min']}) -> "
         f"{'OK' if gate['separation_ok'] else 'FAIL'}")
    _log(f"  (ii)  maniac g(κ=1.0)={gate['maniac_g_at_kappa']:.4f} "
         f"(>= {gate['maniac_g_min']}) -> verdict={gate['kappa_recal_verdict']} "
         f"(κ_final={gate['kappa_final']}) -> "
         f"{'OK' if gate['g_ok'] else 'FAIL'}")
    _log(f"  (iii) maniac acc={gate['maniac_accuracy']:.3f} "
         f"(>= {gate['maniac_accuracy_min']}) -> "
         f"{'OK' if gate['acc_ok'] else 'FAIL'}")
    _log(f"  (iv)  E0={gate['E0_mbb']:.4f} mbb/g (<= {gate['E0_bar']}) -> "
         f"{'OK' if gate['E0_ok'] else 'FAIL'}")
    _log(f"  OVERALL: {gate['verdict']}")

    # JSON-safe coercion: numpy scalars / bools sometimes slip into the gate
    # dict via comparisons and break json.dumps with a cryptic message.
    def _jsafe(x):
        if isinstance(x, dict):
            return {k: _jsafe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_jsafe(v) for v in x]
        if hasattr(x, "item") and not isinstance(x, (str, bytes)):
            try:
                return x.item()
            except Exception:
                pass
        return x

    # Save.
    torch.save({"state_dict": net.state_dict(),
                "config": {"d_model": 64, "nhead": 4, "num_layers": 2,
                            "dim_ff": 128, "max_seq": 128,
                            "phase": "phase1_cotrain_complete",
                            "anchor_dir": str(args.anchor_dir),
                            "epochs": args.epochs,
                            "epsilon_pool": args.epsilon,
                            "lam": args.lam},
                "separation_gate": gate},
               out / "phase1_cotrain.pt")
    with open(out / "aux_pool.pkl", "wb") as f:
        pickle.dump({"epsilon": args.epsilon, "n_hands": args.pool_hands,
                     "n_tuples": len(aux_pool), "seed": args.seed + 7}, f)
    metrics = {
        "phase": "phase1_cotrain",
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
            "lr": args.lr, "lam": args.lam,
            "pool_hands": args.pool_hands,
            "epsilon_noised_anchor": args.epsilon,
            "n_distill_examples": len(distill_examples),
            "n_aux_tuples": len(aux_pool),
            "recipe_A_cell_weights": cell_sampling_weights,
            "recipe_A_realized_cell_counts": cell_counts,
            "recipe_B_class_balance_on": bool(args.class_balance),
            "recipe_B_action_counts": {"fold": int(counts[0]),
                                        "call": int(counts[1]),
                                        "raise": int(counts[2])},
            "recipe_B_class_weights": {"fold": float(cw[0]),
                                        "call": float(cw[1]),
                                        "raise": float(cw[2])},
        },
        "E0_mbb_per_game": e0,
        "read_primitive": args.read_primitive,
        "per_cell_table": cell_table,
        "separation_gate": gate,
        "seconds": {"train": t_train},
    }
    (out / "metrics.json").write_text(json.dumps(_jsafe(metrics), indent=2))
    (out / "train_log.json").write_text(json.dumps(_jsafe(train_log), indent=2))
    _log(f"\n[saved] {out}/phase1_cotrain.pt + metrics.json + train_log.json "
         f"+ aux_pool.pkl")
    _log("[STOP] separation gate reported. STOP for user review.")


if __name__ == "__main__":
    main()
