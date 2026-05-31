"""Phase-2 warmup for the Leduc adaptive-policy proof — opp_head only.

Per the approved structure + HARD STOP #3 protocol:

  - Load the Phase-1 distilled checkpoint.
  - Freeze trunk + policy_head; only opp_head is trainable.
  - K=200 updates. Each update:
      * B=128 hands. cell ~ Uniform(9 train cells); seats 50/50.
      * Hero plays distilled net (policy_head); opp plays archetype.
      * At every hero decision, cache `combined` under no_grad.
      * At every opp decision following ≥1 hero decision, record
        (cached_combined, observed_opp_action, legal_mask).
      * Aux loss = λ · CE(masked opp_head(combined), action), λ=0.3.
      * Adam step (only opp_head params have requires_grad=True).
  - At end-of-warmup: per-cell D̄ + g(κ=1.0), opp-head accuracy + CE per cell,
    trunk/policy_head bit-identity sanity, E0 sanity.
  - Apply user-approved two-condition κ decision rule:
      * lights bar:  always_raise@s1.00 mean g >= 0.40 at κ=1.0 → κ stands.
      * recalibration target (BOTH must hold):
          (a) always_raise@s1.00 g ≈ 0.50, and
          (b) random_uniform@{0.5,1.0} g ≲ 0.10.
        If max(D̄_uniform) ≥ atanh(0.5)/(atanh(0.1)/D̄_uniform) · D̄_maniac
        → no κ satisfies both → HEAD-QUALITY finding, not κ.
  - STOP — write metrics + warmed checkpoint, exit for user review.

Run:
  cd ~/pokerbot && setsid .venv/bin/python -m scripts.leduc_phase2_warmup \
      --phase1-ckpt runs/leduc_phase1_20260531_210331/phase1_distilled.pt \
      --anchor-dir runs/leduc_cfr_anchor_20260531_144405 \
      > /tmp/leduc_phase2_warmup.log 2>&1 < /dev/null & disown
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
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
NEG_INF = -1e9


def _log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Anchor loader (same shape as Phase-1)
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
# Helpers
# --------------------------------------------------------------------------

def sample_from_dict(probs: dict, rng) -> int:
    keys = list(probs.keys())
    ps = np.array(list(probs.values()), dtype=np.float64)
    ps = ps / ps.sum()
    return int(rng.choice(keys, p=ps))


def hero_forward_combined(net, state, device):
    """One-sample forward at a hero decision. Returns
    (action sampled from policy_head over legal mask, combined detached)."""
    toks = TT.tokenize_decision(state)
    packed = collate([toks], device=device)
    with torch.no_grad():
        policy_raw, _, combined = net(
            packed["tokens"], packed["pad_mask"], packed["query_idx"],
            packed["legal_mask"], packed["opp_stats"],
        )
    probs = policy_raw[0].cpu().numpy()
    legal = toks["legal_actions"]
    lm = np.zeros(TT.NUM_ACTIONS, dtype=bool)
    for la in legal:
        lm[la] = True
    return combined[0].detach(), probs, lm


# --------------------------------------------------------------------------
# Warmup rollout: one hand
# --------------------------------------------------------------------------

def warmup_rollout_one_hand(net, opp_fn, hero_seat, device, rng):
    """Roll out one hand. Hero plays distilled net's policy_head; opp plays
    archetype. Returns list of (combined_tensor, observed_opp_action,
    legal_mask_numpy) tuples to feed the aux loss."""
    state = GAME.new_initial_state()
    pairs = []
    cached_combined = None
    while not state.is_terminal():
        if state.is_chance_node():
            outs = state.chance_outcomes()
            acts = [a for a, _ in outs]
            ps = np.array([p for _, p in outs])
            state.apply_action(int(rng.choice(acts, p=ps / ps.sum())))
            continue
        cur = state.current_player()
        if cur == hero_seat:
            combined, probs, lm = hero_forward_combined(net, state, device)
            cached_combined = combined
            # Sample from policy_head over legal mask.
            legal_probs = probs[lm]
            legal_probs = legal_probs / legal_probs.sum()
            legal_actions = np.where(lm)[0]
            a = int(rng.choice(legal_actions, p=legal_probs))
            state.apply_action(a)
        else:
            legal = state.legal_actions()
            lm = np.zeros(TT.NUM_ACTIONS, dtype=bool)
            for la in legal:
                lm[la] = True
            # Opp acts (deterministic / mixture per archetype).
            opp_probs = opp_fn(state)
            a = sample_from_dict(opp_probs, rng)
            # Record (combined, action) for aux loss IF hero has acted at least once.
            if cached_combined is not None:
                pairs.append((cached_combined, a, lm))
            state.apply_action(a)
    return pairs


# --------------------------------------------------------------------------
# K=200 warmup
# --------------------------------------------------------------------------

def run_warmup(net, train_cells, *, K, B, lr, lam, device, seed,
               ckpt_dir, ckpt_every):
    # Freeze trunk + policy_head; ONLY opp_head trainable.
    for p in net.parameters():
        p.requires_grad = False
    for p in net.opp_head.parameters():
        p.requires_grad = True

    trainable = [p for p in net.parameters() if p.requires_grad]
    assert all(p.shape[0] in (3,) or (p.dim() == 2 and p.shape[1] == 96)
               for p in trainable), "unexpected trainable shape"
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in net.parameters())
    _log(f"[warmup] trainable params (opp_head only) = {n_trainable} / {n_total}")

    opt = torch.optim.Adam(trainable, lr=lr)
    rng = np.random.default_rng(seed)
    cells = list(train_cells)
    n_cells = len(cells)
    log = []
    for upd in range(1, K + 1):
        t0 = time.time()
        pairs = []
        for _ in range(B):
            ci = int(rng.integers(n_cells))
            cell = cells[ci]
            assert (cell.name, cell.strength) not in set(A.HELD_OUT_DEFAULT), (
                f"held-out cell leaked into warmup rollout: {cell.id}"
            )
            opp_fn = A.build_cell(cell)
            hero_seat = int(rng.integers(2))
            pairs.extend(warmup_rollout_one_hand(net, opp_fn, hero_seat,
                                                  device=device, rng=rng))
        # Aux loss step.
        if pairs:
            combined_batch = torch.stack([p[0] for p in pairs]).to(device)  # [N, 96]
            action_batch = torch.tensor([p[1] for p in pairs], dtype=torch.long,
                                        device=device)
            legal_batch = torch.tensor(np.stack([p[2] for p in pairs]),
                                       dtype=torch.bool, device=device)
            opp_logits = net.opp_head(combined_batch)
            opp_logits = opp_logits.masked_fill(~legal_batch, NEG_INF)
            loss = lam * F.cross_entropy(opp_logits, action_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Accuracy + CE for logging.
            with torch.no_grad():
                pred = opp_logits.argmax(dim=-1)
                acc = float((pred == action_batch).float().mean())
                ce = float(F.cross_entropy(opp_logits, action_batch))
        else:
            acc = float("nan"); ce = float("nan"); loss = torch.tensor(0.0)
        dt = time.time() - t0
        log.append({"update": upd, "n_pairs": len(pairs),
                    "loss": float(loss.detach()), "acc": acc, "ce": ce,
                    "seconds": dt})
        if upd == 1 or upd % 20 == 0 or upd == K:
            _log(f"  [warmup upd {upd:>3d}/{K}] n_pairs={len(pairs):>4d} "
                 f"loss={float(loss.detach()):.4f} acc={acc:.3f} ce={ce:.4f} "
                 f"({dt:.1f}s)")
        if upd % ckpt_every == 0:
            torch.save({"state_dict": net.state_dict(),
                        "update": upd,
                        "phase": "phase2_warmup_inprogress"},
                       ckpt_dir / f"warmup_ckpt_{upd:04d}.pt")
    return log


# --------------------------------------------------------------------------
# D̄ probe (same as Phase 1 but with trained opp_head)
# --------------------------------------------------------------------------

def run_hand_dbar(net, anchor_fn, opp_fn, hero_seat, *, w0, device, rng):
    state = GAME.new_initial_state()
    d_list, w_list = [], []
    cached_opp_logits = None
    opp_decision_records = []  # for per-cell accuracy / CE
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
            d_t, w_t = read_terms(q_t, pi_eq, lm)
            d_list.append(d_t)
            w_list.append(w_t)
            op = opp_fn(state)
            a = sample_from_dict(op, rng)
            # Record for accuracy/CE: pred dist q_t vs observed action a.
            opp_decision_records.append((q_t.copy(), a, lm.copy()))
            state.apply_action(a)
    return running_read(d_list, w_list, w0=w0), len(d_list), opp_decision_records


def per_cell_table(net, anchor_fn, train_cells, *, w0, kappa, n_hands,
                   device, seed):
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
                    w0=w0, device=device, rng=rng,
                )
                dbars.append(dbar)
                all_records.extend(recs)
        # Accuracy + CE on opp_head predictions vs observed actions.
        if all_records:
            preds = np.stack([r[0] for r in all_records])  # [N, 3]
            acts = np.array([r[1] for r in all_records])    # [N]
            acc = float((preds.argmax(axis=-1) == acts).mean())
            ce = float(-np.mean(np.log(np.clip(preds[np.arange(len(acts)), acts],
                                                1e-12, 1.0))))
        else:
            acc = float("nan"); ce = float("nan")
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
# κ decision rule
# --------------------------------------------------------------------------

def kappa_decision(cell_table, *, kappa, lights_bar, recal_g_maniac,
                   recal_g_uniform_max):
    maniac = next(c for c in cell_table if c["cell_id"] == "always_raise@s1.00")
    uniform_cells = [c for c in cell_table if c["archetype"] == "random_uniform"]
    Dm = maniac["mean_dbar"]
    Du_max = max(c["mean_dbar"] for c in uniform_cells)
    g_maniac_at_kappa = math.tanh(kappa * Dm)

    if g_maniac_at_kappa >= lights_bar:
        return {
            "verdict": "kappa_stands",
            "kappa_final": kappa,
            "g_maniac_at_kappa": g_maniac_at_kappa,
            "g_uniform_max_at_kappa": math.tanh(kappa * Du_max),
            "rationale": (f"g_maniac={g_maniac_at_kappa:.4f} >= lights_bar="
                          f"{lights_bar} at κ={kappa}"),
        }
    # Recalibration required. Solve for κ_new such that:
    #   (a) tanh(κ_new · D̄_maniac) = recal_g_maniac (e.g., 0.50)
    #   (b) tanh(κ_new · D̄_uniform_max) ≤ recal_g_uniform_max (e.g., 0.10)
    kappa_a = math.atanh(recal_g_maniac) / Dm if Dm > 0 else math.inf
    kappa_b_cap = math.atanh(recal_g_uniform_max) / Du_max if Du_max > 0 else math.inf
    if kappa_a <= kappa_b_cap:
        return {
            "verdict": "kappa_recalibrated",
            "kappa_final": kappa_a,
            "g_maniac_at_kappa": math.tanh(kappa_a * Dm),
            "g_uniform_max_at_kappa": math.tanh(kappa_a * Du_max),
            "kappa_b_cap": kappa_b_cap,
            "rationale": (f"κ_new={kappa_a:.4f} hits g_maniac={recal_g_maniac} "
                          f"and stays below g_uniform={recal_g_uniform_max}"),
        }
    # No single κ satisfies both. Head-quality failure.
    separation_ratio = Dm / Du_max if Du_max > 0 else math.inf
    required_ratio = math.atanh(recal_g_maniac) / math.atanh(recal_g_uniform_max)
    return {
        "verdict": "head_quality_failure",
        "kappa_final": None,
        "g_maniac_at_kappa": math.tanh(kappa_a * Dm),
        "g_uniform_max_at_kappa": math.tanh(kappa_a * Du_max),
        "kappa_a_for_maniac": kappa_a,
        "kappa_b_cap_for_uniform": kappa_b_cap,
        "separation_ratio_dmaniac_over_duniform": separation_ratio,
        "required_separation_ratio": required_ratio,
        "rationale": (f"No single κ satisfies both: D̄_maniac/D̄_uniform="
                      f"{separation_ratio:.2f} < required {required_ratio:.2f}. "
                      "opp_head's reads do not separate the maniac from noise."),
    }


# --------------------------------------------------------------------------
# Sanity: trunk + policy_head bit-identical
# --------------------------------------------------------------------------

def trunk_signature(net):
    """SHA-256 of the concatenated bytes of trunk + policy_head params."""
    import hashlib
    h = hashlib.sha256()
    for name, p in net.named_parameters():
        if not name.startswith("opp_head"):
            h.update(name.encode())
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1-ckpt", required=True)
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--K", type=int, default=200, help="warmup updates")
    ap.add_argument("--B", type=int, default=128, help="hands per update")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.3, help="opp-aux λ")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--w0", type=float, default=2.0)
    ap.add_argument("--n-hands-probe", type=int, default=50)
    ap.add_argument("--lights-bar", type=float, default=0.40)
    ap.add_argument("--recal-g-maniac", type=float, default=0.50)
    ap.add_argument("--recal-g-uniform-max", type=float, default=0.10)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"runs/leduc_phase2_warmup_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out
    _log(f"out_dir = {out}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Load Phase-1 checkpoint.
    ckpt = torch.load(args.phase1_ckpt, map_location=device, weights_only=False)
    net = AdaptivePolicyNet().to(device)
    net.load_state_dict(ckpt["state_dict"])
    _log(f"[load] Phase-1 distilled checkpoint from {args.phase1_ckpt}")

    # Capture pre-warmup trunk + policy_head signature.
    sig_before = trunk_signature(net)
    _log(f"[sanity] trunk+policy_head SHA-256 (pre-warmup) = {sig_before[:16]}...")

    # E0 sanity (pre-warmup).
    anchor_fn = load_anchor(Path(args.anchor_dir))
    t0 = time.time()
    e0_pre = exploitability_mbb(GAME, make_net_policy_callable(net, device))
    _log(f"[sanity] E0_pre = {e0_pre:.4f} mbb/g ({time.time()-t0:.1f}s)")

    # Train/test split — assert we only see train cells.
    train_cells, test_cells = A.train_test_split()
    assert len(train_cells) == 9 and len(test_cells) == 3
    _log(f"[cells] train={len(train_cells)} test={len(test_cells)} "
         f"(held-out blocked: {[c.id for c in test_cells]})")

    # Warmup.
    t0 = time.time()
    warmup_log = run_warmup(net, train_cells, K=args.K, B=args.B, lr=args.lr,
                            lam=args.lam, device=device, seed=args.seed,
                            ckpt_dir=ckpt_dir, ckpt_every=args.ckpt_every)
    t_warmup = time.time() - t0
    _log(f"[warmup] {args.K} updates in {t_warmup:.1f}s")

    # Trunk + policy_head bit-identity check.
    sig_after = trunk_signature(net)
    sig_match = (sig_before == sig_after)
    _log(f"[sanity] trunk+policy_head SHA-256 (post-warmup) = {sig_after[:16]}...")
    _log(f"[sanity] trunk+policy_head bit-identical = {sig_match}")
    assert sig_match, "trunk or policy_head modified during warmup — freeze leaked"

    # E0 sanity (post-warmup).
    t0 = time.time()
    e0_post = exploitability_mbb(GAME, make_net_policy_callable(net, device))
    _log(f"[sanity] E0_post = {e0_post:.4f} mbb/g ({time.time()-t0:.1f}s)")
    assert abs(e0_post - e0_pre) < 1e-6, "E0 changed despite trunk freeze"

    # Per-cell D̄ probe.
    _log(f"[probe] running κ-check probe: {len(train_cells)} train cells × "
         f"{args.n_hands_probe} hands × 2 seats")
    t0 = time.time()
    cell_table = per_cell_table(net, anchor_fn, train_cells,
                                w0=args.w0, kappa=args.kappa,
                                n_hands=args.n_hands_probe,
                                device=device, seed=args.seed + 1)
    t_probe = time.time() - t0
    _log(f"[probe] done in {t_probe:.1f}s")

    # κ decision.
    decision = kappa_decision(cell_table,
                              kappa=args.kappa,
                              lights_bar=args.lights_bar,
                              recal_g_maniac=args.recal_g_maniac,
                              recal_g_uniform_max=args.recal_g_uniform_max)
    _log("")
    _log(f"=== κ decision rule (lights_bar={args.lights_bar}) ===")
    _log(f"  verdict: {decision['verdict']}")
    _log(f"  rationale: {decision['rationale']}")
    if decision["kappa_final"] is not None:
        _log(f"  κ_final = {decision['kappa_final']:.4f}")
        _log(f"  g_maniac at κ_final = {math.tanh(decision['kappa_final'] * cell_table[[c['cell_id'] for c in cell_table].index('always_raise@s1.00')]['mean_dbar']):.4f}")

    # If κ recalibrated, re-report table at κ_new.
    final_kappa = decision["kappa_final"] if decision["kappa_final"] is not None else args.kappa
    if decision["verdict"] == "kappa_recalibrated":
        _log(f"\n=== Re-reported table at κ_new = {final_kappa:.4f} ===")
        for c in cell_table:
            c["mean_g_at_kappa_new"] = float(np.tanh(final_kappa * c["mean_dbar"]))
            c["median_g_at_kappa_new"] = float(np.tanh(final_kappa * c["median_dbar"]))
            _log(f"  {c['cell_id']:<28} D̄_mean={c['mean_dbar']:.4f} "
                 f"g(κ=1.0)={c['mean_g_at_kappa']:.4f} "
                 f"g(κ_new)={c['mean_g_at_kappa_new']:.4f}")

    # Save artifacts.
    torch.save({"state_dict": net.state_dict(),
                "config": {**ckpt["config"], "phase": "phase2_warmup_complete"},
                "warmup_K": args.K, "warmup_B": args.B, "warmup_lam": args.lam,
                "kappa_decision": decision},
               out / "phase2_warmed.pt")
    metrics = {
        "phase1_ckpt": str(args.phase1_ckpt),
        "anchor_dir": str(args.anchor_dir),
        "K": args.K, "B": args.B, "lr": args.lr, "lam": args.lam,
        "kappa": args.kappa, "w0": args.w0,
        "n_hands_probe": args.n_hands_probe,
        "lights_bar_maniac_g": args.lights_bar,
        "recal_g_maniac": args.recal_g_maniac,
        "recal_g_uniform_max": args.recal_g_uniform_max,
        "trunk_policy_head_bit_identical": sig_match,
        "trunk_signature_before": sig_before,
        "trunk_signature_after": sig_after,
        "E0_pre_warmup_mbb": e0_pre,
        "E0_post_warmup_mbb": e0_post,
        "per_cell_table": cell_table,
        "kappa_decision": decision,
        "seconds": {"warmup": t_warmup, "probe": t_probe},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "warmup_log.json").write_text(json.dumps(warmup_log, indent=2))
    _log(f"\n[saved] {out}/phase2_warmed.pt + metrics.json + warmup_log.json")
    _log("[STOP] HARD STOP #3 reached. Awaiting user review before policy release.")


if __name__ == "__main__":
    main()
