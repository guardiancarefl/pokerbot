"""Train the ReBeL v3 value net (per-seat 6-vector target + belief-block input).

v3 input  = public block (feat[200:], 36) ⊕ belief block (6×k reach-ranges flat).
v3 target = value6, the PER-SEAT 6-vector value (lets the net value any leaf).

Sized to avoid the smoke-test overfit (350k params / 277k samples): the hidden
width is auto-chosen for ~--samples-per-param against the snapshot. Early stopping
on val R² (saves the BEST-val checkpoint, not the last), weight decay on. Per-epoch
train+val R² (6-vector AND hero-scalar) streamed to --logfile (flushed each epoch).
"""
import sys, time, argparse, json
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn as nn


class ValueNet6(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 6)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def pick_hidden(in_dim, n_train, target_spp):
    """Choose (h1, h2=h1//2) so total params ≈ n_train/target_spp. Params are
    dominated by in_dim*h1, so h1 ≈ budget/in_dim. Clamped to a sane range."""
    budget = max(n_train / max(target_spp, 1), 40_000)
    h1 = int(budget / (in_dim + 1))
    h1 = max(32, min(h1, 256))
    return (h1, max(16, h1 // 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/rebel_v3_full.npz")
    ap.add_argument("--out", default="/workspace/rebel_value_net_full.pt")
    ap.add_argument("--logfile", default="/tmp/train_full.log")
    ap.add_argument("--hidden", default="auto")
    ap.add_argument("--samples-per-param", type=float, default=15.0)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--max-samples", type=int, default=0, help="subsample to this many rows (0=all); for size-matched controls")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    lf = open(a.logfile, "w")
    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n"); lf.flush()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    d = np.load(a.data, allow_pickle=False)
    feat = d["feat"].astype(np.float32)
    belief = d["belief"].astype(np.float32)
    y = d["value6"].astype(np.float32)
    hero = d["hero_seat"].astype(np.int64)
    if a.max_samples and a.max_samples < len(feat):
        sub = np.random.RandomState(a.seed).permutation(len(feat))[:a.max_samples]
        feat, belief, y, hero = feat[sub], belief[sub], y[sub], hero[sub]
    N, k = feat.shape[0], belief.shape[2]
    X = np.concatenate([feat[:, 200:], belief.reshape(N, 6 * k)], axis=1)
    in_dim = X.shape[1]

    idx = np.random.permutation(N); nv = int(N * a.val_frac)
    vi, ti = idx[:nv], idx[nv:]
    ntr = len(ti)
    hidden = pick_hidden(in_dim, ntr, a.samples_per_param) if a.hidden == "auto" \
        else tuple(int(h) for h in a.hidden.split(","))

    ymu, ysd = float(y[ti].mean()), float(y[ti].std() + 1e-8)
    Xtr = torch.tensor(X[ti], device=dev); Ytr = torch.tensor((y[ti]-ymu)/ysd, device=dev)
    Xva = torch.tensor(X[vi], device=dev); Yva = torch.tensor(y[vi], device=dev)
    hva = torch.tensor(hero[vi], device=dev)
    Ytr_raw = torch.tensor(y[ti], device=dev)

    net = ValueNet6(in_dim, hidden).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    lossf = nn.MSELoss()

    log(f"[data] N={N} snapshot  input={in_dim} (public36 + belief 6x{k})  target=value6(6)")
    log(f"[data] value6 std={y.std():.4f}  zero-sum rowsum|mean|={np.abs(y.sum(1)).mean():.2e}")
    log(f"[net] hidden={hidden} params={npar:,}  samples/param={ntr/npar:.1f}  "
        f"train={ntr} val={nv} dev={dev} wd={a.weight_decay}")
    log(f"[earlystop] max_epochs={a.max_epochs} patience={a.patience} (save BEST-val 6vec R2)")

    def r2(pred, tgt, idxgather=None):
        if idxgather is not None:
            pred = pred.gather(1, idxgather[:, None]).squeeze(1)
            tgt = tgt.gather(1, idxgather[:, None]).squeeze(1)
        ss_res = float(((pred - tgt) ** 2).sum())
        ss_tot = float(((tgt - tgt.mean()) ** 2).sum())
        return 1 - ss_res / (ss_tot + 1e-12)

    def evaluate():
        net.eval()
        with torch.no_grad():
            pv = net(Xva) * ysd + ymu
            pt = net(Xtr) * ysd + ymu
            r2_6 = r2(pv, Yva); r2_h = r2(pv, Yva, hva)
            tr_6 = r2(pt, Ytr_raw)
        net.train()
        return r2_6, r2_h, tr_6

    best_r2 = -9.0; best_state = None; best_ep = 0; since = 0
    t0 = time.perf_counter()
    for ep in range(1, a.max_epochs + 1):
        perm = torch.randperm(ntr, device=dev); tot = 0.0
        for i in range(0, ntr, a.batch):
            b = perm[i:i+a.batch]; opt.zero_grad()
            loss = lossf(net(Xtr[b]), Ytr[b]); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        v6, vh, t6 = evaluate()
        flag = ""
        if v6 > best_r2 + 1e-4:
            best_r2 = v6; best_ep = ep; since = 0
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in net.state_dict().items()}
            flag = "  <- best"
        else:
            since += 1
        log(f"[ep {ep:3d}] train_mse={tot/ntr:.4f}  train_R2(6vec)={t6:.4f}  "
            f"val_R2(6vec)={v6:.4f}  val_R2(hero)={vh:.4f}{flag}")
        if since >= a.patience:
            log(f"[earlystop] no val improvement for {a.patience} epochs -> stop at ep {ep}")
            break

    # Restore + save BEST.
    net.load_state_dict(best_state)
    v6, vh, t6 = evaluate()
    log(f"\n[BEST] epoch {best_ep}: val_R2(6vec)={v6:.4f}  val_R2(hero)={vh:.4f}  (saved)")
    log(f"[time] {time.perf_counter()-t0:.1f}s on {dev}")
    torch.save({"state_dict": net.state_dict(), "in_dim": in_dim, "hidden": list(hidden),
                "y_mean": ymu, "y_std": ysd, "k": k, "n_train": ntr, "n_params": npar,
                "best_epoch": best_ep, "val_r2_6vec": v6, "val_r2_hero": vh,
                "input": "public36+belief6xk", "target": "value6_perseat"}, a.out)
    log(f"[saved] {a.out}")
    lf.close()


if __name__ == "__main__":
    main()
