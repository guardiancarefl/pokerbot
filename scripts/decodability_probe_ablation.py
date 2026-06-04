"""JOB 2: opp_stats ablation on the FAITHFUL feature.

The FAITHFUL feature in decodability_probe.py is `combined` = concat(
  trunk_repr_at_query [128], opp_stats_proj(opp_stats) [32]). It scored
held-out binary AUC 0.907, but `opp_stats[:12]` alone (REFERENCE) scored
0.923 — suggesting the held-out signal might be entirely carried by the
opp_stats input via opp_stats_proj.

This probe re-runs the FAITHFUL path with opp_stats=ZEROS at the model
input. The 128-d trunk repr_ is UNCHANGED (it doesn't read opp_stats);
the 32-d opp_stats_proj output collapses to a constant bias term. So the
LR classifier sees the same trunk-only signal plus a useless constant.
Mathematically: this is equivalent to using only the 128-d trunk_repr.

Question: does the trunk's transformer-pooled-at-query representation
carry ANY held-out opp signal independent of opp_stats, or does it
collapse to ~0.46 like PRIMARY tokens?

To avoid re-running the entire forward pass (took ~50 min in the original
probe), we extract ONLY `repr_` (128-d, the slice of `combined` that
doesn't depend on opp_stats). The opp_stats=0 case equals
[repr_, opp_stats_proj.bias_vector] which is linearly equivalent to repr_
alone for a Logistic Regression classifier.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nlhe.adaptive.model import (  # noqa: E402
    Adaptive6MaxNet, F_STATS, F_TOKEN, K_TENDENCY, NUM_ACTIONS, collate,
)
from scripts.decodability_probe import (  # noqa: E402
    iter_decisions, feat_opp_stats, make_binary_mapping, WINDOW,
    HANDS_PER_MATCH,
)


def batch_repr(net, samples: list[dict], device: str = "cpu") -> np.ndarray:
    """Extract repr_ (128-d), the trunk transformer output at query_idx.
    Identical to combined[:, :d_model] — see model.py:133. Independent
    of opp_stats."""
    packed = collate(samples, device=device)
    with torch.no_grad():
        tokens = packed["tokens"]
        pad_mask = packed["pad_mask"]
        query_idx = packed["query_idx"]
        B = tokens.shape[0]
        x = net.token_proj(tokens) + net.pos_embed[:tokens.shape[1]].unsqueeze(0)
        h = net.encoder(x, src_key_padding_mask=pad_mask)
        repr_ = h[torch.arange(B, device=tokens.device), query_idx]
    return repr_.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="l4_corpus")
    ap.add_argument("--phase1-ckpt",
                    default="runs/phase1_d128_repro/smoke_net.pt")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-iter-lr", type=int, default=1000)
    ap.add_argument("--batch-extract", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-json",
                    default="runs/phase1_d128_repro/decodability_ablation.json")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    print("[load] Phase-1 net")
    d = torch.load(args.phase1_ckpt, weights_only=False, map_location="cpu")
    cfg = d["config"]
    net = Adaptive6MaxNet(d_model=cfg["d_model"], num_layers=cfg["num_layers"],
                          nhead=cfg["nhead"], dim_ff=cfg["dim_ff"])
    net.load_state_dict(d["state_dict"])
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"        {n_params:,} params; config={cfg}")

    corpus_dir = Path(args.corpus)
    print(f"[load] streaming corpus from {corpus_dir}")
    train_samples = []
    held_samples = []
    t = time.time()
    for s in iter_decisions(corpus_dir, "train"):
        train_samples.append(s)
    for s in iter_decisions(corpus_dir, "heldout"):
        held_samples.append(s)
    print(f"        train n_decisions = {len(train_samples)}  "
          f"heldout n_decisions = {len(held_samples)}  "
          f"wall = {time.time()-t:.1f}s")

    # Match-level split for train (same scheme/seed as decodability_probe)
    train_match_ids = sorted({s["match_id"] for s in train_samples})
    shuf_idx = rng.permutation(len(train_match_ids))
    n_val = int(args.val_frac * len(train_match_ids))
    val_set = {train_match_ids[i] for i in shuf_idx[:n_val]}
    print(f"[split] train_matches = {len(train_match_ids) - n_val}  "
          f"val_matches = {n_val}")

    # Build binary mapping (same procedure as original probe)
    tt_by_opp_match = defaultdict(dict)
    for s in train_samples:
        tt_by_opp_match[s["opp_name"]].setdefault(
            s["match_id"], s["tendency_target"])
    per_opp_mean_tt_train = {
        n: np.stack(list(md.values())).mean(axis=0)
        for n, md in tt_by_opp_match.items()
    }
    tt_by_opp_match_h = defaultdict(dict)
    for s in held_samples:
        tt_by_opp_match_h[s["opp_name"]].setdefault(
            s["match_id"], s["tendency_target"])
    per_opp_mean_tt_held = {
        n: np.stack(list(md.values())).mean(axis=0)
        for n, md in tt_by_opp_match_h.items()
    }
    with open(corpus_dir / "l4_corpus_train" / "manifest.json") as f:
        train_manifest = json.load(f)
    bp_ref = np.asarray(train_manifest["tendency_blueprint_ref"], dtype=np.float32)
    threshold = float(np.median(sorted(
        np.linalg.norm(v - bp_ref) for v in per_opp_mean_tt_train.values())))
    bin_labels_train, _ = make_binary_mapping(per_opp_mean_tt_train, bp_ref,
                                              threshold)
    bin_labels_held, _ = make_binary_mapping(per_opp_mean_tt_held, bp_ref,
                                              threshold)
    print(f"[label] binary threshold = {threshold:.4f}")

    # Extract repr_ (128-d) for both splits — this is the FAITHFUL feature
    # with opp_stats=0 (the opp_stats_proj contribution collapses to a constant
    # bias term, which adds no information to the LR).
    print(f"[feat] extracting repr_ (128-d) from frozen trunk, train, "
          f"batch={args.batch_extract}")
    t = time.time()
    X_train = np.empty((len(train_samples), net.d_model), dtype=np.float32)
    for i in range(0, len(train_samples), args.batch_extract):
        batch = train_samples[i:i + args.batch_extract]
        X_train[i:i + len(batch)] = batch_repr(net, batch)
        if i // args.batch_extract % 50 == 0:
            print(f"        train batch {i:>7d}/{len(train_samples)}")
    print(f"        X_train {X_train.shape}  wall = {time.time()-t:.1f}s")

    print(f"[feat] extracting repr_ for heldout")
    t = time.time()
    X_held = np.empty((len(held_samples), net.d_model), dtype=np.float32)
    for i in range(0, len(held_samples), args.batch_extract):
        batch = held_samples[i:i + args.batch_extract]
        X_held[i:i + len(batch)] = batch_repr(net, batch)
    print(f"        X_held {X_held.shape}  wall = {time.time()-t:.1f}s")

    # Build labels + history-index
    y_bin_train = np.array(
        [bin_labels_train[s["opp_name"]] for s in train_samples], dtype=np.int64)
    y_bin_held = np.array(
        [bin_labels_held[s["opp_name"]] for s in held_samples], dtype=np.int64)
    hi_train = np.array(
        [s["hand_index_in_match"] for s in train_samples], dtype=np.int64)
    hi_held = np.array(
        [s["hand_index_in_match"] for s in held_samples], dtype=np.int64)
    mid_train = [s["match_id"] for s in train_samples]
    is_val = np.array([m in val_set for m in mid_train], dtype=bool)

    Xtr = X_train[~is_val]
    Xva = X_train[is_val]
    ytr = y_bin_train[~is_val]
    yva = y_bin_train[is_val]
    hi_va = hi_train[is_val]

    print(f"[label] y_bin_train: GTO={(y_bin_train==0).sum()}  "
          f"exploitable={(y_bin_train==1).sum()}")
    print(f"[label] y_bin_held:  GTO={(y_bin_held==0).sum()}  "
          f"exploitable={(y_bin_held==1).sum()}")

    # Standardize
    print(f"[fit] standardize + LR fit (max_iter={args.max_iter_lr})")
    t = time.time()
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xhd_s = scaler.transform(X_held)
    lr = LogisticRegression(solver="lbfgs", max_iter=args.max_iter_lr,
                            n_jobs=-1, random_state=args.seed)
    lr.fit(Xtr_s, ytr)
    print(f"        fit wall = {time.time()-t:.1f}s")

    p_val = lr.predict_proba(Xva_s)[:, 1]
    p_held = lr.predict_proba(Xhd_s)[:, 1]
    auc_val = float(roc_auc_score(yva, p_val))
    auc_held = float(roc_auc_score(y_bin_held, p_held))

    print(f"\n=== FAITHFUL ABLATED (opp_stats zeroed) — binary AUC ===")
    print(f"  val AUC      = {auc_val:.4f}")
    print(f"  heldout AUC  = {auc_held:.4f}")
    print(f"  reference points (from decodability_probe.json):")
    print(f"    FAITHFUL non-ablated val      = 0.9220")
    print(f"    FAITHFUL non-ablated heldout  = 0.9074")
    print(f"    PRIMARY tokens val            = 0.7606")
    print(f"    PRIMARY tokens heldout        = 0.4637  (chance baseline)")

    # History-length buckets
    buckets = [(1, lambda hi: hi <= 1),
               (3, lambda hi: (hi >= 2) & (hi <= 3)),
               (5, lambda hi: (hi >= 4) & (hi <= 5)),
               (10, lambda hi: (hi >= 6) & (hi <= 10)),
               (20, lambda hi: hi >= 20)]

    print(f"\n=== History-length AUC (binary) — FAITHFUL ABLATED ===")
    per_bucket = {}
    for tag, predicate in buckets:
        mask_v = predicate(hi_va)
        mask_h = predicate(hi_held)
        try:
            auc_b_v = float(roc_auc_score(yva[mask_v], p_val[mask_v]))
        except Exception:
            auc_b_v = None
        try:
            auc_b_h = float(roc_auc_score(y_bin_held[mask_h], p_held[mask_h]))
        except Exception:
            auc_b_h = None
        per_bucket[tag] = {
            "val_auc": auc_b_v, "val_n": int(mask_v.sum()),
            "held_auc": auc_b_h, "held_n": int(mask_h.sum()),
        }
        v_str = f"{auc_b_v:.4f}" if auc_b_v is not None else "N/A"
        h_str = f"{auc_b_h:.4f}" if auc_b_h is not None else "N/A"
        print(f"  bucket@{tag:>3}h_obs : val auc={v_str} "
              f"(n={int(mask_v.sum())})  "
              f"held auc={h_str} (n={int(mask_h.sum())})")

    payload = {
        "ablation": "opp_stats input zeroed; equivalent to using repr_ (128-d trunk-only) since opp_stats_proj(0) is a constant bias",
        "feature_dim": int(net.d_model),
        "n_train_decisions": int(len(train_samples)),
        "n_held_decisions": int(len(held_samples)),
        "n_val_matches": int(n_val),
        "binary_threshold_train_median": float(threshold),
        "binary_label_train": bin_labels_train,
        "binary_label_held": bin_labels_held,
        "val_AUC": auc_val,
        "heldout_AUC": auc_held,
        "history_length_buckets": per_bucket,
        "reference_decodability_probe_results": {
            "FAITHFUL_non_ablated_val": 0.9220,
            "FAITHFUL_non_ablated_heldout": 0.9074,
            "REFERENCE_opp_stats_val": 0.9203,
            "REFERENCE_opp_stats_heldout": 0.9231,
            "PRIMARY_tokens_val": 0.7606,
            "PRIMARY_tokens_heldout": 0.4637,
        },
        "wall_seconds": float(time.time() - t0),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[saved] {out}")
    print(f"[done] total wall = {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
