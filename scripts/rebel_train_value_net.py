"""Train the ROUGH ReBeL PBS value net on the bootstrap dataset (Gate-2 preview).

Input  = the persisted PBS encoding (the 236-dim InfosetEncoder6Max root feature:
         public block + acting-seat bucket one-hot — sample_io `feat`).
Target = the depth-limited search's own solved root value (`root_value`, a scalar
         hero-relative value in ICM-equity-delta units — sample_io target, the
         ReBeL bootstrapped regression target, docs/REBEL_PBS_6MAX.md §5).

Net is sized to the bootstrap data (~550k samples): a small MLP, ~39k params
(~14 samples/param). 10% held out for validation. Trains on the GPU; generation
is CPU-bound (don't touch it).

Reports train/val loss and held-out value-prediction error (MAE, RMSE, R^2) in
the original value units, with a predict-the-mean baseline for context. Saves the
net + normalization stats to --out for the Gate-2 peek.
"""
import sys, time, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/rebel_bootstrap_merged.npz")
    ap.add_argument("--out", default="/workspace/rebel_value_net_rough.pt")
    ap.add_argument("--hidden", default="128,64")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    d = np.load(a.data, allow_pickle=False)
    X = d["feat"].astype(np.float32)          # (N, 236)
    y = d["root_value"].astype(np.float32)    # (N,)
    N, in_dim = X.shape
    print(f"[data] {N} samples, feat_dim={in_dim}, "
          f"target mean={y.mean():.4f} std={y.std():.4f} "
          f"min={y.min():.3f} max={y.max():.3f}", flush=True)

    # Split.
    idx = np.random.permutation(N)
    n_val = int(N * a.val_frac)
    vi, ti = idx[:n_val], idx[n_val:]
    Xtr, ytr, Xva, yva = X[ti], y[ti], X[vi], y[vi]
    print(f"[split] train={len(ti)} val={len(vi)}", flush=True)

    # Standardize the target for training stability; report errors in original units.
    ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-8)

    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor((ytr - ymu) / ysd, device=dev)
    Xva_t = torch.tensor(Xva, device=dev)
    yva_t = torch.tensor(yva, device=dev)

    hidden = tuple(int(h) for h in a.hidden.split(","))
    net = ValueNet(in_dim, hidden).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[net] hidden={hidden} params={n_params:,} "
          f"({len(ti)/n_params:.1f} samples/param) device={dev}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    loss_fn = nn.MSELoss()
    ntr = len(ti)

    def val_metrics():
        net.eval()
        with torch.no_grad():
            pred = net(Xva_t) * ysd + ymu       # back to original units
            err = pred - yva_t
            mae = float(err.abs().mean())
            rmse = float((err ** 2).mean() ** 0.5)
            ss_res = float((err ** 2).sum())
            ss_tot = float(((yva_t - yva_t.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        net.train()
        return mae, rmse, r2

    t0 = time.perf_counter()
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(ntr, device=dev)
        tot = 0.0
        for i in range(0, ntr, a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad()
            out = net(Xtr_t[b])
            loss = loss_fn(out, ytr_t[b])
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if ep % 5 == 0 or ep == 1 or ep == a.epochs:
            mae, rmse, r2 = val_metrics()
            print(f"[ep {ep:3d}] train_mse(std)={tot/ntr:.4f}  "
                  f"val_MAE={mae:.4f}  val_RMSE={rmse:.4f}  val_R2={r2:.4f}", flush=True)

    # Baselines for context (held-out).
    base_rmse = float(((yva - ytr.mean()) ** 2).mean() ** 0.5)
    base_mae = float(np.abs(yva - ytr.mean()).mean())
    mae, rmse, r2 = val_metrics()
    print(f"[FINAL] held-out: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}  "
          f"| predict-mean baseline: MAE={base_mae:.4f} RMSE={base_rmse:.4f}", flush=True)
    print(f"[FINAL] value units = ICM-equity-delta (target std={ysd:.4f}); "
          f"RMSE/std = {rmse/ysd:.3f}  (lower is better; <1 beats the mean)", flush=True)
    print(f"[time] {time.perf_counter()-t0:.1f}s on {dev}", flush=True)

    torch.save({
        "state_dict": net.state_dict(),
        "in_dim": in_dim, "hidden": hidden,
        "y_mean": ymu, "y_std": ysd,
        "n_train": len(ti), "n_val": len(vi), "n_params": n_params,
        "val_mae": mae, "val_rmse": rmse, "val_r2": r2,
        "schema": 1, "target": "root_value_scalar_hero_relative",
        "input": "infoset6_236_public+actingbucket",
    }, a.out)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
