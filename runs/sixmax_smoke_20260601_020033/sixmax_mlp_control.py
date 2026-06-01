"""D1 — MLP control: train an MLP student with the same architectural shape
as the blueprint (hidden=[256, 256]) on the SAME distill pipeline / data /
mask / loss / LR / epochs as the d=64 smoke. lam_aux=0 (pure distill).

If MLP reaches near-zero KL → the smoke's transformer-over-history is the
bottleneck (S1/S4 architectural). If MLP also stalls at ~0.10-0.15 → NOT
architecture; it's a data/target/loss/mask issue (S3 or label-gen).

The MLP consumes ONLY tokens[query_idx] — the per-state feat vector at the
hero's current decision — not the sequence. This matches the blueprint's
input surface exactly.

Read-only on prod code; runs in /tmp.
"""
from __future__ import annotations
import os, sys, json, pickle, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, "/home/quant/pokerbot")

# Import smoke module via the proper package path; this triggers the
# leakage preflight and torch threading config.
from scripts import six_max_adaptive_smoke as sm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nlhe.adaptive.model import F_TOKEN, NUM_ACTIONS

GAME = sm.GAME
ANCHOR_DIR = "runs/six_max_20260530_034023_phase4f_dcfr_candC_k200"
ABSTRACTION_PKL = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
SEED = 2026
NEG_INF = -1e9


class MLPStudent(nn.Module):
    """Same architectural shape as the blueprint strategy net: in_dim 236,
    hidden [256, 256], out_dim 9. ReLU activations, masked-softmax on the
    output (matches the smoke's policy_head pathway)."""
    def __init__(self, in_dim=236, hidden=(256, 256), n_actions=9):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, feat, legal_mask):
        logits = self.net(feat)
        logits = logits.masked_fill(~legal_mask, NEG_INF)
        return torch.softmax(logits, dim=-1)


def main():
    print("=" * 70)
    print("D1 — MLP control (blueprint-shape: hidden=[256,256])")
    print("    SAME distill loss, data, mask, LR=3e-4, epochs=25, lam_aux=0")
    print("=" * 70)

    # Load blueprint
    t0 = time.time()
    solver = sm.load_blueprint(
        __import__("pathlib").Path(ANCHOR_DIR),
        __import__("pathlib").Path(ABSTRACTION_PKL))
    print(f"[blueprint] loaded in {time.time()-t0:.1f}s")

    # Build the eval set deterministically (same seed as smoke + diagnostics)
    t0 = time.time()
    eval_set = sm.build_eval_set(solver, n_states=1000, seed=SEED + 12345)
    print(f"[eval set] {len(eval_set)} states in {time.time()-t0:.1f}s")

    # Build the SAME pool the smoke uses (same seed → bit-identical)
    print("[pool gen] regenerating same-seed pool ...")
    t0 = time.time()
    opp_specs = {k: {"path": sm.POOL_DEFAULTS[k]["path"],
                     "n_matches": sm.POOL_DEFAULTS[k]["n_matches"]}
                 for k in sm.POOL_DEFAULTS}
    pool, _, pool_stats = sm.generate_pool(
        solver=solver, opp_specs=opp_specs,
        hand_target=40, epsilon=0.20,
        base_seed=SEED, verbose=False)
    print(f"[pool gen] {pool_stats['n_pool_tuples']} tuples in "
          f"{pool_stats['wall_seconds']:.1f}s")

    # Build the MLP and train
    net = MLPStudent(in_dim=F_TOKEN, hidden=(256, 256), n_actions=NUM_ACTIONS)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[mlp] hidden=[256,256], {n_params:,} params")

    # Training params (match d=64 smoke at LR=3e-4)
    epochs = 25
    steps_per_epoch = 15
    batch_size = 256
    lr = 3e-4
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    n_pool = len(pool)

    # Pre-extract per-tuple feat and legal_mask + distill_target
    feats = np.stack([t["tokens"][t["query_idx"]] for t in pool])
    legals = np.stack([t["legal_mask"] for t in pool])
    targets = np.stack([t["distill_target"] for t in pool])
    print(f"[pool flat] feats {feats.shape}, legals {legals.shape}, "
          f"targets {targets.shape}")

    # Pre-train floor sanity
    def floor_check_mlp(net, eval_set):
        net.eval()
        kls = []; ces = []
        with torch.no_grad():
            for i in range(0, len(eval_set), 64):
                batch = eval_set[i:i+64]
                fe = torch.from_numpy(np.stack(
                    [t["tokens"][t["query_idx"]] for t in batch]))
                lm = torch.from_numpy(np.stack(
                    [t["legal_mask"] for t in batch]))
                out = net(fe, lm).cpu().numpy()
                for j, t in enumerate(batch):
                    bp = t["blueprint_probs"]; mask = t["legal_mask"]
                    p = np.clip(bp[mask], 1e-12, 1.0); p /= p.sum()
                    q = np.clip(out[j][mask], 1e-12, 1.0); q /= q.sum()
                    kls.append(float(np.sum(p * (np.log(p) - np.log(q)))))
                    # CE
                    ces.append(float(-np.sum(bp * np.log(np.maximum(out[j], 1e-12)))))
        return (float(np.mean(kls)), float(np.quantile(kls, 0.95)),
                float(np.mean(ces)))

    pre_mean_kl, pre_p95_kl, pre_ce = floor_check_mlp(net, eval_set)
    print(f"[pre-train] mean_KL={pre_mean_kl:.4f}  p95_KL={pre_p95_kl:.4f}  "
          f"CE={pre_ce:.4f}")

    # Train
    print("\n=== TRAINING ===")
    traj_ce = []
    traj_kl = []
    net.train()
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        ep_L = 0.0
        n_steps = 0
        for s in range(steps_per_epoch):
            idx = rng.choice(n_pool, size=batch_size, replace=False)
            fe = torch.from_numpy(feats[idx])
            lm = torch.from_numpy(legals[idx])
            tg = torch.from_numpy(targets[idx])
            policy_raw = net(fe, lm)
            log_p = torch.log(policy_raw + 1e-12)
            L = -(tg * log_p).sum(dim=-1).mean()
            opt.zero_grad(); L.backward(); opt.step()
            ep_L += float(L.detach())
            n_steps += 1
        avg_ce = ep_L / n_steps
        traj_ce.append(avg_ce)
        # Floor check every epoch (cheap on MLP)
        m_kl, p95_kl, ce = floor_check_mlp(net, eval_set)
        traj_kl.append(m_kl)
        net.train()
        dt = time.time() - t_ep
        if ep == 1 or ep % 2 == 0 or ep == epochs:
            print(f"  [ep {ep:>3d}/{epochs}] train_CE={avg_ce:.4f}  "
                  f"eval mean_KL={m_kl:.4f}  p95_KL={p95_kl:.4f}  "
                  f"({dt:.1f}s)")

    # Final
    final_mean_kl, final_p95_kl, final_ce = floor_check_mlp(net, eval_set)
    H_eval = 0.5539  # from D0
    print("")
    print("=== D1 VERDICT ===")
    print(f"  MLP final mean_KL = {final_mean_kl:.4f}  (bar 0.02)")
    print(f"  MLP final p95_KL  = {final_p95_kl:.4f}  (bar 0.05)")
    print(f"  MLP final CE (eval) = {final_ce:.4f}  (H_eval = {H_eval:.4f}, "
          f"CE - H = {final_ce - H_eval:.4f})")
    print("")
    print("Comparison vs transformer student at d=128, ep25 (from D0+D2):")
    print(f"  d=128 mean_KL     = 0.1023")
    print(f"  d=128 p95_KL      = 0.4475")
    print(f"  d=128 CE          = 0.6563")
    print("")
    print("Read:")
    if final_mean_kl <= 0.05:
        print(f"  MLP reached mean_KL = {final_mean_kl:.4f} <= 0.05")
        print("  → Architecture (transformer sequence handling on the "
              "redundant 236-dim surface) is the bottleneck.")
        print("  S1/S4 candidate confirmed; refactor earned.")
    elif final_mean_kl <= 0.5 * 0.1023:
        print(f"  MLP at {final_mean_kl:.4f} is much better than d=128's "
              "0.1023.")
        print("  → Architecture contributes substantially; not the whole "
              "story but a real factor.")
    else:
        print(f"  MLP at {final_mean_kl:.4f} stalls in same neighborhood as "
              "d=128's 0.1023.")
        print("  → Architecture is NOT the issue. Look at data / target / "
              "mask / label-gen.")
        print("  S3 (mask pathology) or label-distribution issue most likely.")

    # Save the trajectory for the commit
    np.save("/tmp/mlp_traj_ce.npy", np.array(traj_ce))
    np.save("/tmp/mlp_traj_kl.npy", np.array(traj_kl))
    with open("/tmp/mlp_final.json", "w") as f:
        json.dump({
            "final_mean_kl": final_mean_kl,
            "final_p95_kl": final_p95_kl,
            "final_ce": final_ce,
            "traj_ce": traj_ce,
            "traj_kl": traj_kl,
        }, f, indent=2)
    print("\nSaved /tmp/mlp_final.json, /tmp/mlp_traj_ce.npy, "
          "/tmp/mlp_traj_kl.npy")


if __name__ == "__main__":
    main()
