"""Train the ReBeL policy net — the warm-start arm of ReBeL self-play.

Counterpart to `rebel_train_value_net_v3.py` (which fits the depth-limited search's
PER-SEAT 6-vector root VALUE). This trainer fits the search's REFINED ROOT POLICY
at the hero's decision — the σ̂ that, fed back as the hero-root warm-start to the
CFR resolver, completes the ReBeL value+policy loop (Brown et al., NeurIPS 2020).

Input  = public block (feat[200:], 36) ⊕ belief block (6×k reach-ranges, flat).
         IDENTICAL feature construction to the value net so the same belief-pipe
         feeds both heads, and `policy_net(x)` at inference uses the resolver's
         already-computed (public, belief) tensor with no extra work.
Target = `root_policy` (length-9 distribution over DiscreteAction), masked by the
         per-sample `legal_mask`. We also drop K<2 (no real CFR loop ran) and any
         `degraded` row (a leaf went missing → root_policy unreliable).

Loss  = masked KL(target ‖ pred) per row, mean over rows. Pred is a masked
        softmax over 9 logits (illegal slots forced to −inf). Validation reports
        the same KL plus an L1 (Σ|p−q|) so the gate has a magnitude scale.

Checkpoint shape mirrors the v3 value net so the existing `mlp()` loader in
`rebel_gate2.py` / `subgame_solver` can reuse it (same `state_dict` keys with
`net.<i>.<weight|bias>`, plus `in_dim`, `hidden`, `k`, `n_train`, `n_params`).
Out dim is recorded as `out_dim=9` so a loader can distinguish a value (6) from
a policy (9) checkpoint without sniffing parameter shapes.
"""
import sys, os, time, argparse, json, glob
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_N_ACTIONS = 9
_NUM_SEATS = 6


class PolicyNet(nn.Module):
    """Same MLP body as the value net, output dim = 9 (DiscreteAction slots)."""
    def __init__(self, in_dim, hidden):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, _N_ACTIONS)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def pick_hidden(in_dim, n_train, target_spp):
    """Same auto-sizer as the v3 value net (h1 ≈ budget/in_dim; clamped)."""
    budget = max(n_train / max(target_spp, 1), 40_000)
    h1 = int(budget / (in_dim + 1))
    h1 = max(32, min(h1, 256))
    return (h1, max(16, h1 // 2))


def load_dataset(roots, drop_degraded=True, min_k=2, max_rows=0, seed=0):
    """Stream shards under `roots`, stack columns we need, drop rows the trainer
    cannot use. Returns (X, P, M, meta) where X is (N, 36+6k), P is (N, 9) the
    masked target, M is (N, 9) the legal mask, meta has shapes / counters."""
    from src.rebel.sample_io import iter_shards
    paths = list(iter_shards(roots))
    if not paths:
        raise FileNotFoundError(f"no shards under {list(roots)}")
    feats, beliefs, policies, masks = [], [], [], []
    n_seen = n_v2 = n_kept = n_degraded = n_low_k = 0
    k = None
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            if "_meta_json" not in z.files:
                continue
            m = json.loads(str(z["_meta_json"]))
            if m.get("schema", 0) < 2:
                continue  # v1 shards lack belief / value6 / non-noisy targets
            n = int(z["feat"].shape[0])
            n_seen += n; n_v2 += n
            keep = np.ones(n, dtype=bool)
            if drop_degraded and "degraded" in z.files:
                deg = z["degraded"].astype(bool)
                n_degraded += int(deg.sum())
                keep &= ~deg
            if min_k > 0 and "n_iterations" in z.files:
                low = z["n_iterations"].astype(np.int64) < min_k
                n_low_k += int(low.sum())
                keep &= ~low
            if not keep.any():
                continue
            feat = z["feat"][keep].astype(np.float32)  # (m, 236)
            bel = z["belief"][keep].astype(np.float32)  # (m, 6, k)
            pol = z["root_policy"][keep].astype(np.float32)  # (m, 9)
            msk = z["legal_mask"][keep].astype(np.float32)  # (m, 9)
            if k is None:
                k = bel.shape[2]
            elif bel.shape[2] != k:
                raise ValueError(f"belief k drift: {p} has {bel.shape[2]}, expected {k}")
            feats.append(feat[:, 200:].copy())  # public block only
            beliefs.append(bel.reshape(len(feat), _NUM_SEATS * k))
            policies.append(pol)
            masks.append(msk)
            n_kept += len(feat)
    if not feats:
        raise RuntimeError("after filtering, no usable rows")
    X = np.concatenate([np.concatenate(feats, axis=0),
                        np.concatenate(beliefs, axis=0)], axis=1)
    P = np.concatenate(policies, axis=0)
    M = np.concatenate(masks, axis=0)
    if max_rows and max_rows < len(X):
        sub = np.random.RandomState(seed).permutation(len(X))[:max_rows]
        X, P, M = X[sub], P[sub], M[sub]
    meta = dict(n_seen=n_seen, n_v2=n_v2, n_kept=n_kept,
                n_degraded=n_degraded, n_low_k=n_low_k, k=k)
    return X, P, M, meta


def masked_kl(logits, target, mask, eps=1e-9):
    """KL( target ‖ softmax(logits over legal) ), mean over batch.

    target rows already sum to 1 over legal actions (root_policy was masked by the
    solver). Illegal slots get logits = −inf before softmax; targets at those
    slots are 0 and contribute nothing to the cross-entropy or to log target.
    Returns (kl, ce, ent) — KL = CE − ent(target); we minimize KL = CE up to a
    target-only constant (ent doesn't depend on params).
    """
    neg_inf = torch.full_like(logits, float("-inf"))
    masked_logits = torch.where(mask > 0, logits, neg_inf)
    log_pred = F.log_softmax(masked_logits, dim=1)
    # CE = -Σ target · log_pred over LEGAL slots. At illegal slots log_pred=-inf,
    # target=0 → 0·(-inf)=NaN in float; mask the product to keep it well-defined.
    zero = torch.zeros_like(log_pred)
    ce = -torch.where(mask > 0, target * log_pred, zero).sum(dim=1)
    # Entropy of target (legal-only): −Σ p log p, with 0 log 0 = 0.
    safe_t = torch.clamp(target, min=eps)
    ent = -torch.where(mask > 0, target * torch.log(safe_t), zero).sum(dim=1)
    kl = ce - ent
    return kl.mean(), ce.mean(), ent.mean()


def masked_l1(logits, target, mask):
    """L1(target − softmax(logits over legal)), mean. Magnitude scale for the gate."""
    neg_inf = torch.full_like(logits, float("-inf"))
    masked_logits = torch.where(mask > 0, logits, neg_inf)
    pred = F.softmax(masked_logits, dim=1)
    return (pred - target).abs().sum(dim=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["/home/quant/rebel_samples"],
                    help="one or more shard root dirs (merged across hosts)")
    ap.add_argument("--out", default="rebel_policy_net.pt")
    ap.add_argument("--logfile", default="/tmp/policy_train.log")
    ap.add_argument("--hidden", default="auto")
    ap.add_argument("--samples-per-param", type=float, default=15.0)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    lf = open(a.logfile, "w")
    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n"); lf.flush()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    t_load = time.perf_counter()
    X, P, M, dmeta = load_dataset(a.roots, max_rows=a.max_samples, seed=a.seed)
    log(f"[data] roots={a.roots} loaded in {time.perf_counter()-t_load:.1f}s "
        f"n_seen={dmeta['n_seen']} n_v2={dmeta['n_v2']} "
        f"n_degraded={dmeta['n_degraded']} n_low_k={dmeta['n_low_k']} "
        f"n_kept={dmeta['n_kept']} k={dmeta['k']}")
    N, in_dim = X.shape
    log(f"[data] X={X.shape} P={P.shape} M={M.shape}  in_dim={in_dim}")
    # Sanity: rows of P sum to ~1 over legal mask.
    rowsum = (P * M).sum(1)
    log(f"[data] root_policy rowsum (over legal): mean={rowsum.mean():.4f} "
        f"min={rowsum.min():.4f} max={rowsum.max():.4f}")

    idx = np.random.permutation(N); nv = int(N * a.val_frac)
    vi, ti = idx[:nv], idx[nv:]
    ntr = len(ti)
    hidden = pick_hidden(in_dim, ntr, a.samples_per_param) if a.hidden == "auto" \
        else tuple(int(h) for h in a.hidden.split(","))

    Xtr = torch.tensor(X[ti], device=dev); Ptr = torch.tensor(P[ti], device=dev)
    Mtr = torch.tensor(M[ti], device=dev)
    Xva = torch.tensor(X[vi], device=dev); Pva = torch.tensor(P[vi], device=dev)
    Mva = torch.tensor(M[vi], device=dev)

    net = PolicyNet(in_dim, hidden).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    log(f"[net] hidden={hidden} params={npar:,} samples/param={ntr/npar:.1f}  "
        f"train={ntr} val={nv} dev={dev} wd={a.weight_decay}")
    log(f"[earlystop] max_epochs={a.max_epochs} patience={a.patience} (save BEST-val KL)")

    def evaluate():
        net.eval()
        with torch.no_grad():
            lo_v = net(Xva)
            kl_v, ce_v, ent_v = masked_kl(lo_v, Pva, Mva)
            l1_v = masked_l1(lo_v, Pva, Mva)
            lo_t = net(Xtr)
            kl_t, _, _ = masked_kl(lo_t, Ptr, Mtr)
        net.train()
        return float(kl_v), float(l1_v), float(ent_v), float(kl_t)

    best_kl = float("inf"); best_state = None; best_ep = 0; since = 0
    t0 = time.perf_counter()
    for ep in range(1, a.max_epochs + 1):
        perm = torch.randperm(ntr, device=dev); tot = 0.0
        for i in range(0, ntr, a.batch):
            b = perm[i:i+a.batch]; opt.zero_grad()
            lo = net(Xtr[b])
            kl, _, _ = masked_kl(lo, Ptr[b], Mtr[b])
            kl.backward(); opt.step()
            tot += float(kl) * len(b)
        kl_v, l1_v, ent_v, kl_t = evaluate()
        flag = ""
        if kl_v < best_kl - 1e-5:
            best_kl = kl_v; best_ep = ep; since = 0
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in net.state_dict().items()}
            flag = "  <- best"
            # Persist on improvement so an in-progress run is usable for wiring tests.
            torch.save({"state_dict": best_state, "in_dim": in_dim, "hidden": list(hidden),
                        "out_dim": _N_ACTIONS, "k": dmeta["k"], "n_train": ntr, "n_params": npar,
                        "best_epoch": ep, "val_kl": kl_v, "val_l1": l1_v,
                        "input": "public36+belief6xk", "target": "root_policy_masked",
                        "loss": "masked_kl"}, a.out)
        else:
            since += 1
        log(f"[ep {ep:3d}] train_KL={tot/ntr:.4f}  val_KL={kl_v:.4f}  "
            f"val_L1={l1_v:.4f}  val_ent(target)={ent_v:.4f}{flag}")
        if since >= a.patience:
            log(f"[earlystop] no val improvement for {a.patience} epochs -> stop at ep {ep}")
            break

    net.load_state_dict(best_state)
    kl_v, l1_v, ent_v, kl_t = evaluate()
    log(f"\n[BEST] epoch {best_ep}: val_KL={kl_v:.4f}  val_L1={l1_v:.4f}  (saved)")
    log(f"[time] {time.perf_counter()-t0:.1f}s on {dev}")
    torch.save({"state_dict": net.state_dict(), "in_dim": in_dim, "hidden": list(hidden),
                "out_dim": _N_ACTIONS, "k": dmeta["k"], "n_train": ntr, "n_params": npar,
                "best_epoch": best_ep, "val_kl": kl_v, "val_l1": l1_v,
                "input": "public36+belief6xk", "target": "root_policy_masked",
                "loss": "masked_kl"}, a.out)
    log(f"[saved] {a.out}")
    lf.close()


if __name__ == "__main__":
    main()
