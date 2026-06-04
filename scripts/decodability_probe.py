"""Phase-2 premise: decodability probe — necessary-condition screen.

If opponent identity isn't decodable from observable within-match history at
realistic match length — especially on held-out opponents — Phase-2 has
nothing to exploit. We test three feature sources:

  (1) PRIMARY / verdict: mean+std pool over tokens[≤50, 236] → 472-d.
  (2) REFERENCE / not-counted: opp_stats[:12]. Tendency-derived upper bound.
      Causal (not leaky from the future), but circular for the binary verdict
      because the binary label is also a function of per-opp tendency.
  (3) FAITHFUL: `combined` (160-d) from the frozen Phase-1 d128 backbone
      (`runs/phase1_d128_repro/smoke_net.pt`). This is the representation the
      blend's policy_head and opp_head_tendency BOTH read from
      (src/nlhe/adaptive/model.py:135). False-red guard: if (1) decodes but
      (3) doesn't, the trunk has discarded the signal — Phase-2 RL can't
      recover that without unfreezing the backbone.

Splits by MATCH (match_id reconstructed from worker stream + 150-contiguous
grouping), not by tuple — prevents intra-match leakage.

Output:
  - /tmp/decodability_probe_results.json (full numerical results)
  - docs/DECODABILITY_PROBE.md (markdown report, written by caller)
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
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

WINDOW = 50
HANDS_PER_MATCH = 150


# ----------------------------------------------------------------------
# Corpus loader: inject match_id + hand_index_in_match via 150-contiguous
# grouping within each worker stream.
# ----------------------------------------------------------------------

def iter_decisions(corpus_dir: Path, split: str):
    """Yield per-decision dicts from a corpus split, injecting match_id and
    hand_index_in_match. Match_id = (worker_id, match_within_worker_idx);
    hand_index_in_match = 0..149. Decisions inherit hand-level fields plus
    decision-level fields, with tokens sliced per expand_hand's window=50
    rule and query_idx made relative to the slice."""
    sub = corpus_dir / f"l4_corpus_{split}"
    shards_by_worker: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for p in sub.glob("shard_w*.pkl.gz"):
        # "shard_w{worker}_{idx}.pkl.gz"
        stem = p.name[: -len(".pkl.gz")]
        _, wpart, ipart = stem.split("_")
        wid = int(wpart[1:])  # drop leading 'w'
        sidx = int(ipart)
        shards_by_worker[wid].append((sidx, p))
    for wid in shards_by_worker:
        shards_by_worker[wid].sort()

    for wid in sorted(shards_by_worker.keys()):
        hand_i_in_worker = 0
        for _, path in shards_by_worker[wid]:
            with gzip.open(path, "rb") as f:
                hands = pickle.load(f)
            for hand in hands:
                hand_index_in_match = hand_i_in_worker % HANDS_PER_MATCH
                match_within_worker = hand_i_in_worker // HANDS_PER_MATCH
                match_id = (wid, match_within_worker)
                hand_i_in_worker += 1
                toks = hand["tokens"]
                for d in hand["decisions"]:
                    qi = int(d["query_idx"])
                    lo = max(0, qi - WINDOW + 1)
                    yield {
                        "tokens": toks[lo:qi + 1].astype(np.float32),
                        "query_idx": qi - lo,
                        "legal_mask": d["legal_mask"],
                        "opp_stats": np.asarray(d["opp_stats"], dtype=np.float32),
                        "match_conf": float(d["match_conf"]),
                        "opp_id": int(hand["opp_id"]),
                        "opp_name": str(hand["opp_name"]),
                        "hero_seat": int(hand["hero_seat"]),
                        "tendency_target": np.asarray(
                            hand["tendency_target"], dtype=np.float32),
                        "hand_index_in_match": hand_index_in_match,
                        "match_id": match_id,
                        "worker_id": wid,
                    }


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------

def feat_tokens_pool(sample) -> np.ndarray:
    """(1) PRIMARY: mean+std pool over tokens[≤50, 236] → 472-d."""
    toks = sample["tokens"]  # [T, 236]
    mean = toks.mean(axis=0)
    std = toks.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


def feat_opp_stats(sample) -> np.ndarray:
    """(2) REFERENCE: opp_stats[:12]."""
    return sample["opp_stats"][:12].astype(np.float32)


def batch_combined(net, samples: list[dict], device: str = "cpu") -> np.ndarray:
    """(3) FAITHFUL: `combined` (160-d) from frozen Phase-1 net."""
    packed = collate(samples, device=device)
    with torch.no_grad():
        _, _, combined = net(
            packed["tokens"], packed["pad_mask"], packed["query_idx"],
            packed["legal_mask"], packed["opp_stats"])
    return combined.cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------
# Binary label: GTO-like (0) vs exploitable (1), thresholded on
# per-opp mean tendency_target distance from manifest's blueprint reference.
# ----------------------------------------------------------------------

def make_binary_mapping(per_opp_mean_tendency: dict[str, np.ndarray],
                        bp_ref: np.ndarray, threshold: float
                        ) -> tuple[dict[str, int], dict[str, float]]:
    """Returns (opp_name -> binary_label, opp_name -> L2_distance).
    binary_label = 0 (GTO-like) if dist < threshold, else 1 (exploitable).
    Blueprint is FORCED to 0 (the universal GTO reference)."""
    dists = {n: float(np.linalg.norm(v - bp_ref))
             for n, v in per_opp_mean_tendency.items()}
    labels = {}
    for n, d in dists.items():
        if n == "blueprint":
            labels[n] = 0
        else:
            labels[n] = 0 if d < threshold else 1
    return labels, dists


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="l4_corpus")
    ap.add_argument("--phase1-ckpt",
                    default="runs/phase1_d128_repro/smoke_net.pt")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-iter-lr", type=int, default=500)
    ap.add_argument("--batch-combined", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-json",
                    default="runs/phase1_d128_repro/decodability_probe.json")
    args = ap.parse_args()

    t0_all = time.time()
    rng = np.random.default_rng(args.seed)

    print("[load] Phase-1 net for FAITHFUL feature (3)")
    d = torch.load(args.phase1_ckpt, weights_only=False, map_location="cpu")
    cfg = d["config"]
    net = Adaptive6MaxNet(d_model=cfg["d_model"], num_layers=cfg["num_layers"],
                          nhead=cfg["nhead"], dim_ff=cfg["dim_ff"])
    net.load_state_dict(d["state_dict"])
    net.eval()
    print(f"        {sum(p.numel() for p in net.parameters()):,} params; "
          f"config={cfg}")

    # ------ Load both splits ------
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

    # ------ Match-level split for train ------
    train_match_ids = sorted({s["match_id"] for s in train_samples})
    n_train_matches = len(train_match_ids)
    print(f"[split] train matches = {n_train_matches}")
    shuf_idx = rng.permutation(n_train_matches)
    n_val = int(args.val_frac * n_train_matches)
    val_set = {train_match_ids[i] for i in shuf_idx[:n_val]}
    train_set = {train_match_ids[i] for i in shuf_idx[n_val:]}
    print(f"        train_matches = {len(train_set)}  "
          f"val_matches = {len(val_set)}")

    # ------ Per-opp mean tendency_target (uses tendency_target which is
    # PER-MATCH; collapse to per-opp by averaging match-level vectors) ------
    # train: get per-(opp_name, match_id) tendency_target, then per-opp mean
    tt_by_opp_match: dict[str, dict[tuple, np.ndarray]] = defaultdict(dict)
    for s in train_samples:
        tt_by_opp_match[s["opp_name"]].setdefault(
            s["match_id"], s["tendency_target"])
    per_opp_mean_tt_train = {
        n: np.stack(list(md.values())).mean(axis=0)
        for n, md in tt_by_opp_match.items()
    }
    tt_by_opp_match_h: dict[str, dict[tuple, np.ndarray]] = defaultdict(dict)
    for s in held_samples:
        tt_by_opp_match_h[s["opp_name"]].setdefault(
            s["match_id"], s["tendency_target"])
    per_opp_mean_tt_held = {
        n: np.stack(list(md.values())).mean(axis=0)
        for n, md in tt_by_opp_match_h.items()
    }

    # Read blueprint ref from manifest
    with open(corpus_dir / "l4_corpus_train" / "manifest.json") as f:
        train_manifest = json.load(f)
    bp_ref = np.asarray(train_manifest["tendency_blueprint_ref"], dtype=np.float32)

    # Pick threshold: median train opp distance (gives ~balanced binary)
    train_dists = {
        n: float(np.linalg.norm(v - bp_ref))
        for n, v in per_opp_mean_tt_train.items()}
    train_dist_values = sorted(train_dists.values())
    threshold = float(np.median(train_dist_values))
    bin_labels_train, _ = make_binary_mapping(per_opp_mean_tt_train,
                                              bp_ref, threshold)
    bin_labels_held, _ = make_binary_mapping(per_opp_mean_tt_held,
                                              bp_ref, threshold)
    print("[label] binary mapping (0=GTO-like, 1=exploitable):")
    print("  TRAIN:")
    for n in sorted(per_opp_mean_tt_train):
        d_ = train_dists[n]
        print(f"    {n:<24s} dist_to_bp_ref = {d_:.4f}  label = "
              f"{bin_labels_train[n]}")
    print(f"    [threshold = median train dist = {threshold:.4f}]")
    print("  HELD-OUT:")
    held_dists = {
        n: float(np.linalg.norm(v - bp_ref))
        for n, v in per_opp_mean_tt_held.items()}
    for n in sorted(per_opp_mean_tt_held):
        d_ = held_dists[n]
        print(f"    {n:<24s} dist_to_bp_ref = {d_:.4f}  label = "
              f"{bin_labels_held[n]}  (held-out)")

    # ------ Multiclass opp_id label mapping (12 train classes) ------
    train_opp_names = sorted({s["opp_name"] for s in train_samples})
    opp_to_idx = {n: i for i, n in enumerate(train_opp_names)}
    n_classes_train = len(train_opp_names)
    print(f"[label] multiclass: {n_classes_train} train opps -> idx")
    for n, i in opp_to_idx.items():
        print(f"    {i:>2d}: {n}")

    # ------ Match length distribution ------
    match_to_n_hands = defaultdict(set)
    for s in train_samples:
        match_to_n_hands[s["match_id"]].add(s["hand_index_in_match"])
    n_hands_per_match = sorted(len(v) for v in match_to_n_hands.values())
    median_match_len = int(np.median(n_hands_per_match)) if n_hands_per_match else 0
    print(f"[stats] median train-match length = {median_match_len} hands "
          f"(min={n_hands_per_match[0]} max={n_hands_per_match[-1]} "
          f"n_matches={len(n_hands_per_match)})")

    # ------ Build feature matrices ------
    def build_static_features(samples):
        X1 = np.empty((len(samples), 2 * F_TOKEN), dtype=np.float32)
        X2 = np.empty((len(samples), F_STATS), dtype=np.float32)
        for i, s in enumerate(samples):
            X1[i] = feat_tokens_pool(s)
            X2[i] = feat_opp_stats(s)
        return X1, X2

    def build_combined_features(samples, batch_size):
        X = np.empty((len(samples), net.d_model + net.d_stats_proj),
                     dtype=np.float32)
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            X[i:i + len(batch)] = batch_combined(net, batch)
            if i // batch_size % 50 == 0:
                print(f"        combined batch {i:>7d}/{len(samples)}")
        return X

    print("[feat] building train static features (1)+(2)")
    t = time.time()
    X1_train, X2_train = build_static_features(train_samples)
    print(f"        X1 {X1_train.shape}  X2 {X2_train.shape}  "
          f"wall = {time.time()-t:.1f}s")

    print("[feat] building heldout static features (1)+(2)")
    t = time.time()
    X1_held, X2_held = build_static_features(held_samples)
    print(f"        X1_held {X1_held.shape}  X2_held {X2_held.shape}  "
          f"wall = {time.time()-t:.1f}s")

    print(f"[feat] building train FAITHFUL (3) combined (batch={args.batch_combined})")
    t = time.time()
    X3_train = build_combined_features(train_samples, args.batch_combined)
    print(f"        X3 {X3_train.shape}  wall = {time.time()-t:.1f}s")

    print(f"[feat] building heldout FAITHFUL (3) combined")
    t = time.time()
    X3_held = build_combined_features(held_samples, args.batch_combined)
    print(f"        X3_held {X3_held.shape}  wall = {time.time()-t:.1f}s")

    # ------ Build labels ------
    y_multi_train = np.array(
        [opp_to_idx[s["opp_name"]] for s in train_samples], dtype=np.int64)
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

    print(f"[label] y_multi_train classes: "
          f"{Counter(y_multi_train.tolist()).most_common()}")
    print(f"[label] y_bin_train: GTO={int((y_bin_train==0).sum())}  "
          f"exploitable={int((y_bin_train==1).sum())}")
    print(f"[label] y_bin_held: GTO={int((y_bin_held==0).sum())}  "
          f"exploitable={int((y_bin_held==1).sum())}")

    # ------ Train+evaluate each feature ------
    results: dict = {
        "feature_names": {
            "1": "PRIMARY tokens_pool [mean,std] 472-d",
            "2": "REFERENCE opp_stats[:12] (circular for binary verdict)",
            "3": "FAITHFUL combined 160-d from frozen Phase-1 d128 backbone",
        },
        "n_train_decisions": int(len(train_samples)),
        "n_held_decisions": int(len(held_samples)),
        "n_train_matches": int(n_train_matches),
        "n_val_matches": int(len(val_set)),
        "n_classes_train": int(n_classes_train),
        "opp_to_idx": opp_to_idx,
        "binary_threshold_train_median": float(threshold),
        "binary_label_train": bin_labels_train,
        "binary_label_held": bin_labels_held,
        "train_dist_to_bp_ref": train_dists,
        "held_dist_to_bp_ref": held_dists,
        "blueprint_ref": bp_ref.tolist(),
        "median_train_match_length": int(median_match_len),
        "match_length_min": int(n_hands_per_match[0]) if n_hands_per_match else 0,
        "match_length_max": int(n_hands_per_match[-1]) if n_hands_per_match else 0,
        "by_feature": {},
    }

    history_buckets = [(1, lambda hi: hi <= 1),
                       (3, lambda hi: (hi >= 2) & (hi <= 3)),
                       (5, lambda hi: (hi >= 4) & (hi <= 5)),
                       (10, lambda hi: (hi >= 6) & (hi <= 10)),
                       (20, lambda hi: hi >= 20)]

    for fname, Xtr_full, Xhd_full in [
        ("1", X1_train, X1_held),
        ("2", X2_train, X2_held),
        ("3", X3_train, X3_held),
    ]:
        print(f"\n=== Feature ({fname}) — {results['feature_names'][fname]} ===")
        Xtr = Xtr_full[~is_val]
        Xva = Xtr_full[is_val]
        ymt_tr = y_multi_train[~is_val]
        ymt_va = y_multi_train[is_val]
        ybi_tr = y_bin_train[~is_val]
        ybi_va = y_bin_train[is_val]
        hi_va = hi_train[is_val]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xhd_s = scaler.transform(Xhd_full)

        # multiclass
        t = time.time()
        lr_m = LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            max_iter=args.max_iter_lr, n_jobs=-1,
            random_state=args.seed)
        lr_m.fit(Xtr_s, ymt_tr)
        top1 = float(accuracy_score(ymt_va, lr_m.predict(Xva_s)))
        chance = float(1.0 / n_classes_train)
        print(f"  multiclass top-1 val = {top1:.4f}  vs chance {chance:.4f}  "
              f"wall = {time.time()-t:.1f}s")

        # binary train+val
        t = time.time()
        lr_b = LogisticRegression(
            solver="lbfgs", max_iter=args.max_iter_lr, n_jobs=-1,
            random_state=args.seed)
        lr_b.fit(Xtr_s, ybi_tr)
        p_val = lr_b.predict_proba(Xva_s)[:, 1]
        auc_val = float(roc_auc_score(ybi_va, p_val))
        print(f"  binary AUC val  = {auc_val:.4f}  "
              f"(n_pos={int(ybi_va.sum())} n_neg={int((1-ybi_va).sum())})  "
              f"wall = {time.time()-t:.1f}s")

        # binary held-out
        if (y_bin_held == 0).any() and (y_bin_held == 1).any():
            p_held = lr_b.predict_proba(Xhd_s)[:, 1]
            auc_held = float(roc_auc_score(y_bin_held, p_held))
            print(f"  binary AUC heldout = {auc_held:.4f}  "
                  f"(n_pos={int(y_bin_held.sum())} "
                  f"n_neg={int((1-y_bin_held).sum())})")
        else:
            auc_held = None
            print(f"  binary AUC heldout = N/A (only one class present: "
                  f"GTO={int((y_bin_held==0).sum())} "
                  f"exploitable={int((y_bin_held==1).sum())})")

        # history-length curve (binary AUC)
        print(f"  history-length AUC (binary):")
        per_bucket = {}
        for tag, predicate in history_buckets:
            mask_v = predicate(hi_va)
            n_pos_v = int(ybi_va[mask_v].sum())
            n_neg_v = int((1 - ybi_va[mask_v]).sum())
            mask_h = predicate(hi_held)
            n_pos_h = int(y_bin_held[mask_h].sum())
            n_neg_h = int((1 - y_bin_held[mask_h]).sum())
            try:
                auc_b_v = float(roc_auc_score(ybi_va[mask_v], p_val[mask_v]))
            except Exception:
                auc_b_v = None
            if auc_held is not None:
                try:
                    auc_b_h = float(roc_auc_score(
                        y_bin_held[mask_h], p_held[mask_h]))
                except Exception:
                    auc_b_h = None
            else:
                auc_b_h = None
            per_bucket[tag] = {
                "val_auc": auc_b_v, "val_n_pos": n_pos_v, "val_n_neg": n_neg_v,
                "held_auc": auc_b_h, "held_n_pos": n_pos_h, "held_n_neg": n_neg_h,
            }
            print(f"    bucket@{tag:>3}h_obs : val auc={auc_b_v if auc_b_v is None else f'{auc_b_v:.4f}'} "
                  f"(n={int(mask_v.sum())})  "
                  f"held auc={auc_b_h if auc_b_h is None else f'{auc_b_h:.4f}'} "
                  f"(n={int(mask_h.sum())})")

        results["by_feature"][fname] = {
            "multiclass_top1_val": top1,
            "multiclass_chance": chance,
            "binary_auc_val": auc_val,
            "binary_auc_heldout": auc_held,
            "history_length_buckets": per_bucket,
        }

    # ------ save ------
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[saved] {out_path}")
    print(f"[done] total wall = {time.time()-t0_all:.1f}s")


if __name__ == "__main__":
    main()
