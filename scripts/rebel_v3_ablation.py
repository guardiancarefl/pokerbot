"""Controlled ablation on v3 data: does the BELIEF block add predictive signal?

v1's 0.24 was on one-spot data — not comparable to full-coverage v3. The clean
question is: on the SAME v3 data + SAME target + SAME net, does adding the belief
block beat public-only? If yes, the reach-belief fix is doing real work.

Three configs (small net, weight decay, early-stop best val R²):
  A. hero-scalar target, input = public(36)                  [baseline]
  B. hero-scalar target, input = public(36) + belief(6xk)    [+belief]
  C. per-seat 6-vec target, input = public + belief          [full v3]
"""
import sys, argparse
sys.path.insert(0, '.')
import numpy as np
import torch, torch.nn as nn


def mlp(in_dim, out_dim, hidden=(64, 32)):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def run(name, X, Y, hero, dev, epochs=80, bs=2048, wd=1e-4):
    N = X.shape[0]; out_dim = Y.shape[1]
    rng = np.random.RandomState(0); idx = rng.permutation(N); nv = N // 10
    vi, ti = idx[:nv], idx[nv:]
    ymu = Y[ti].mean(0); ysd = Y[ti].std(0) + 1e-8
    Xtr = torch.tensor(X[ti], device=dev); Ytr = torch.tensor((Y[ti]-ymu)/ysd, device=dev)
    Xva = torch.tensor(X[vi], device=dev); Yva = torch.tensor(Y[vi], device=dev)
    hva = hero[vi]
    net = mlp(X.shape[1], out_dim).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd); lf = nn.MSELoss()
    best = -9; ntr = len(ti)
    for ep in range(1, epochs+1):
        perm = torch.randperm(ntr, device=dev)
        for i in range(0, ntr, bs):
            b = perm[i:i+bs]; opt.zero_grad(); lf(net(Xtr[b]), Ytr[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = (net(Xva).cpu().numpy())*ysd + ymu
        net.train()
        # R^2 on the hero value (the meaningful quantity in every config)
        if out_dim == 1:
            ph, th = pred[:,0], Y[vi][:,0]
        else:
            ph = pred[np.arange(len(vi)), hva]; th = Y[vi][np.arange(len(vi)), hva]
        r2 = 1 - ((ph-th)**2).sum()/(((th-th.mean())**2).sum()+1e-12)
        best = max(best, r2)
    print(f"  {name:32s} params={npar:6d}  BEST val R2(hero)={best:.4f}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="/workspace/rebel_v3_smoke.npz")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.data, allow_pickle=False)
    feat = d["feat"].astype(np.float32); belief = d["belief"].astype(np.float32)
    v6 = d["value6"].astype(np.float32); hero = d["hero_seat"].astype(np.int64)
    N, k = feat.shape[0], belief.shape[2]
    public = feat[:, 200:]                                  # (N,36)
    pub_bel = np.concatenate([public, belief.reshape(N, 6*k)], 1)  # (N,1236)
    yh = v6[np.arange(N), hero][:, None]                    # hero-scalar target (N,1)
    print(f"[ablation] N={N}  hero-value R2 (early-stop best), small net + weight decay:", flush=True)
    rA = run("A public-only -> hero",        public,  yh, hero, dev)
    rB = run("B public+BELIEF -> hero",      pub_bel, yh, hero, dev)
    rC = run("C public+belief -> 6vec",      pub_bel, v6, hero, dev)
    print(f"\n[ablation] belief lift on hero-value R2: A={rA:.4f} -> B={rB:.4f}  (delta {rB-rA:+.4f})", flush=True)
    print(f"[ablation] full v3 (6-vec head) hero R2 = {rC:.4f}", flush=True)
    print("[ablation] B>A => the reach-belief block adds real predictive signal (the fix works).", flush=True)


if __name__ == "__main__":
    main()
