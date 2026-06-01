"""D0 + D2 diagnostics: confirm L_distill = CE (not KL), measure irreducible
floor, per-state KL histogram on the existing d=128 checkpoint.

D0: confirm L_distill is forward cross-entropy by computing it with the
    student replaced by the teacher (= H(teacher)). Sanity: KL(t||t) = 0.
    Verify CE = H + KL on the eval set.
D2: load runs/sixmax_smoke_20260601_020033/smoke_net.pt (the d=128 ckpt),
    compute per-state forward-KL on the floor eval set, stratify the
    worst 5% by (a) top-mass bucket, (b) #legal actions, (c) street.

D1 (MLP control) is a separate file (runs longer; trains a fresh MLP).

Read-only on prod code; runs in /tmp.
"""
from __future__ import annotations
import os, sys, json, pickle, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)

sys.path.insert(0, "/home/quant/pokerbot")

from src.nlhe.game_strings import six_max_sng
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max
from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.cfr6 import _build_view_6max
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.adaptive.model import (
    Adaptive6MaxNet, F_TOKEN, F_STATS, NUM_ACTIONS, collate,
)
import pyspiel

GAME = pyspiel.load_game(six_max_sng())
ANCHOR_DIR = "runs/six_max_20260530_034023_phase4f_dcfr_candC_k200"
ABSTRACTION_PKL = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
D128_CKPT = "runs/sixmax_smoke_20260601_020033/smoke_net.pt"
SEED = 2026
EVAL_SET_SEED = SEED + 12345  # matches the smoke script's build_eval_set
N_EVAL = 1000


def load_blueprint():
    with open(ABSTRACTION_PKL, "rb") as f:
        absn = pickle.load(f)
    with open(f"{ANCHOR_DIR}/config.json") as f:
        cfg_d = json.load(f)
    cfg = TrainConfig6Max(
        starting_stack=int(cfg_d["starting_stack"]),
        big_blind=int(cfg_d["big_blind"]),
        small_blind=int(cfg_d["small_blind"]),
        payout_mode=cfg_d["payout_mode"],
        buy_in=float(cfg_d["buy_in"]),
        first_share=float(cfg_d["first_share"]),
        hidden_dim=list(cfg_d["hidden_dim"]),
        bucket_runouts=int(cfg_d["bucket_runouts"]),
        tournament_structure_path=None,
    )
    solver = DeepCFR6MaxSolver(game=GAME, abstraction=absn, config=cfg)
    ckpt = torch.load(f"{ANCHOR_DIR}/checkpoints/ckpt_iter_2000.pt",
                       weights_only=False, map_location="cpu")
    solver.policy_nets.load_state_dict(ckpt["policy_nets"])
    return solver


def build_eval_set(solver, n_states=N_EVAL, seed=EVAL_SET_SEED):
    """Same as scripts.six_max_adaptive_smoke.build_eval_set."""
    py_rng = random.Random(seed)
    enc_rng = random.Random(seed + 1)
    eval_tuples = []
    while len(eval_tuples) < n_states:
        hero_seat = py_rng.randrange(6)
        state = GAME.new_initial_state()
        hand_buffer = []
        while not state.is_terminal() and len(eval_tuples) < n_states:
            if state.is_chance_node():
                outs = state.chance_outcomes()
                a = py_rng.choices(
                    [o for o, _ in outs],
                    weights=[p for _, p in outs], k=1)[0]
                state.apply_action(int(a))
                continue
            parsed_hero = parse_state_6max(state, observer=hero_seat)
            solver.encoder.reset_cache()
            tok = solver.encoder.encode_from_parsed(
                parsed_hero, rng=enc_rng).astype(np.float32)
            hand_buffer.append(tok)
            if len(hand_buffer) > 50:
                hand_buffer = hand_buffer[-50:]
            cur = state.current_player()
            if cur == hero_seat:
                parsed_cur = parse_state_6max(state)
                feat = solver.encoder.encode_from_parsed(
                    parsed_cur, rng=random.Random(cur * 100003 + 7))
                view = _build_view_6max(state, parsed_cur)
                d_to_chip = discretize_legal_actions(
                    list(state.legal_actions()), view)
                lm = np.zeros(NUM_ACTIONS, dtype=bool)
                for d in d_to_chip:
                    lm[int(d)] = True
                probs = solver.policy_nets.inference_policy(cur, feat, lm)
                tokens = np.stack(hand_buffer).astype(np.float32)
                # street_idx for stratification
                street_idx = int(parsed_cur.get("street_idx", 0))
                eval_tuples.append({
                    "tokens": tokens,
                    "query_idx": int(tokens.shape[0] - 1),
                    "legal_mask": lm.copy(),
                    "opp_stats": np.zeros(F_STATS, dtype=np.float32),
                    "blueprint_probs": probs.astype(np.float32),
                    "street_idx": street_idx,
                })
                chip = py_rng.choice(list(d_to_chip.values()))
                state.apply_action(int(chip))
            else:
                chip = py_rng.choice(state.legal_actions())
                state.apply_action(int(chip))
    return eval_tuples


def loss_distill_fn(student_probs, teacher_probs):
    """Reproduce the smoke's L_distill formula exactly:
       L = -sum(target * log(student + 1e-12)).mean()
    where target = teacher_probs."""
    eps = 1e-12
    return float(-(teacher_probs * np.log(np.maximum(student_probs, eps))
                   ).sum(axis=-1).mean())


def teacher_entropy_per_state(probs, legal_mask):
    """H(teacher) on the LEGAL subset (zeros on illegal don't contribute)."""
    eps = 1e-12
    p = np.where(legal_mask, np.clip(probs, eps, 1.0), 0.0)
    # since teacher mass on illegal is 0, p*log(p) has 0 contribution there
    p_legal = p[legal_mask]
    p_legal = p_legal / max(p_legal.sum(), eps)
    return float(-np.sum(p_legal * np.log(p_legal + eps)))


def kl_forward_per_state(teacher_probs, student_probs, legal_mask):
    """KL(teacher || student) on legal subset."""
    eps = 1e-12
    legal = legal_mask
    p = np.clip(teacher_probs[legal], eps, 1.0); p /= p.sum()
    q = np.clip(student_probs[legal], eps, 1.0); q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


# ========================================================================
# D0 — what is L_distill, and what's the irreducible floor?
# ========================================================================

print("=" * 70)
print("D0 — verify L_distill is forward cross-entropy (not KL)")
print("=" * 70)
print("\nLoss expression in scripts/six_max_adaptive_smoke.py:709:")
print("    log_p = torch.log(policy_raw + 1e-12)")
print("    L_distill = -(distill_target * log_p).sum(dim=-1).mean()")
print("")
print("Math: this is forward-CE under the teacher distribution:")
print("    CE(t||s) = -E_t[log s] = H(t) + KL_forward(t||s)")
print("    NOT KL alone.")
print("")
print("Test: replace the student with the teacher itself. Then:")
print("    L_distill should equal H(teacher) (the irreducible floor),")
print("    and KL(teacher||teacher) should be exactly 0.")
print("")

solver = load_blueprint()
t0 = time.time()
eval_set = build_eval_set(solver, n_states=N_EVAL, seed=EVAL_SET_SEED)
print(f"[eval set] {len(eval_set)} states built in {time.time()-t0:.1f}s\n")

# Teacher-as-student baseline
ce_self = []
H_per_state = []
kl_self = []
for t in eval_set:
    p = t["blueprint_probs"]; lm = t["legal_mask"]
    # L_distill formula with student=teacher: identical mass on legal,
    # zero on illegal.
    ce_self.append(loss_distill_fn(p, p))
    H_per_state.append(teacher_entropy_per_state(p, lm))
    kl_self.append(kl_forward_per_state(p, p, lm))

ce_self = np.array(ce_self); H_per_state = np.array(H_per_state); kl_self = np.array(kl_self)
print(f"L_distill (student = teacher) mean:        {ce_self.mean():.4f} nats")
print(f"H(teacher) mean   (legal-subset entropy):  {H_per_state.mean():.4f} nats")
print(f"KL(teacher||teacher) mean (sanity = 0):    {kl_self.mean():.6e}")
print(f"|L_distill_self - H_mean| (should be ~0): {abs(ce_self.mean()-H_per_state.mean()):.4f}")

# Real student: load the d=128 checkpoint and compute its L_distill + KL
print("\n" + "-" * 70)
print("Now compute L_distill and KL on the eval set using the d=128 student.")
ckpt = torch.load(D128_CKPT, weights_only=False, map_location="cpu")
net = Adaptive6MaxNet(d_model=128, num_layers=4, nhead=4, dim_ff=512)
net.load_state_dict(ckpt["state_dict"]); net.eval()

ce_student = []
kl_student = []
batch_size = 64
NEG_INF = -1e9
with torch.no_grad():
    for i in range(0, len(eval_set), batch_size):
        batch = eval_set[i:i+batch_size]
        packed = collate(batch, device="cpu")
        policy_raw, _, _ = net(
            packed["tokens"], packed["pad_mask"], packed["query_idx"],
            packed["legal_mask"], packed["opp_stats"])
        student_probs = policy_raw.cpu().numpy()
        for j, t in enumerate(batch):
            p_t = t["blueprint_probs"]; lm = t["legal_mask"]
            p_s = student_probs[j]
            ce_student.append(loss_distill_fn(p_s, p_t))
            kl_student.append(kl_forward_per_state(p_t, p_s, lm))
ce_student = np.array(ce_student); kl_student = np.array(kl_student)
print(f"L_distill (student = d=128 net) mean:      {ce_student.mean():.4f}")
print(f"KL(teacher||student) mean:                 {kl_student.mean():.4f}")
print(f"H(teacher) mean (irreducible):             {H_per_state.mean():.4f}")
print(f"Verify CE = H + KL: {H_per_state.mean():.4f} + {kl_student.mean():.4f}"
      f" = {H_per_state.mean()+kl_student.mean():.4f}  (vs CE {ce_student.mean():.4f})")

# Final D0 reading
gap_KL = kl_student.mean()
target_KL = 0.02
print("\nD0 VERDICT:")
print(f"  Irreducible floor (H(teacher) mean) = {H_per_state.mean():.4f} nats")
print(f"  L_distill plateau (d=128, ep25)     = {ce_student.mean():.4f} nats")
print(f"  Real gap above floor (mean KL)      = {gap_KL:.4f} nats")
print(f"  Floor bar (mean KL ≤ 0.02)          = {target_KL} nats")
print(f"  Ratio: gap/bar                      = {gap_KL/target_KL:.1f}x")

# ========================================================================
# D2 — per-state KL histogram + stratification
# ========================================================================
print("\n" + "=" * 70)
print("D2 — per-state KL histogram on the existing d=128 checkpoint")
print("=" * 70)
kls = np.array(kl_student)
print(f"\nN states: {len(kls)}")
print(f"KL stats: min={kls.min():.4f}, mean={kls.mean():.4f}, "
      f"median={np.median(kls):.4f}, p75={np.quantile(kls, 0.75):.4f}, "
      f"p95={np.quantile(kls, 0.95):.4f}, p99={np.quantile(kls, 0.99):.4f}, "
      f"max={kls.max():.4f}")

# Stratify by (a) top-mass bucket, (b) #legal actions, (c) street
print("\nStratification by TOP-MASS of teacher:")
top_mass = np.array([t["blueprint_probs"].max() for t in eval_set])
for lo, hi, label in [(0.0, 0.5, "[0.0, 0.5)"),
                       (0.5, 0.7, "[0.5, 0.7)"),
                       (0.7, 0.9, "[0.7, 0.9)"),
                       (0.9, 1.01, "[0.9, 1.0]")]:
    mask = (top_mass >= lo) & (top_mass < hi)
    n = int(mask.sum())
    if n == 0:
        continue
    print(f"  top_mass {label}  n={n:>4d}  "
          f"mean_KL={kls[mask].mean():.4f}  "
          f"p95_KL={np.quantile(kls[mask], 0.95):.4f}  "
          f"max_KL={kls[mask].max():.4f}")

print("\nStratification by #LEGAL actions:")
n_legal = np.array([int(t["legal_mask"].sum()) for t in eval_set])
for nl in sorted(set(n_legal.tolist())):
    mask = (n_legal == nl)
    n = int(mask.sum())
    if n == 0: continue
    print(f"  n_legal = {nl}  n={n:>4d}  "
          f"mean_KL={kls[mask].mean():.4f}  "
          f"p95_KL={np.quantile(kls[mask], 0.95) if n > 1 else kls[mask][0]:.4f}  "
          f"max_KL={kls[mask].max():.4f}")

print("\nStratification by STREET:")
streets = np.array([t["street_idx"] for t in eval_set])
street_names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
for s in (0, 1, 2, 3):
    mask = (streets == s)
    n = int(mask.sum())
    if n == 0: continue
    print(f"  {street_names[s]:<8}  n={n:>4d}  "
          f"mean_KL={kls[mask].mean():.4f}  "
          f"p95_KL={np.quantile(kls[mask], 0.95) if n > 1 else kls[mask][0]:.4f}  "
          f"max_KL={kls[mask].max():.4f}")

# Worst 5% — print 10 examples
print("\nWORST 5% — sample 10 worst states:")
worst_idx = np.argsort(-kls)[:10]
for k, i in enumerate(worst_idx):
    t = eval_set[int(i)]
    p = t["blueprint_probs"]; lm = t["legal_mask"]
    s = streets[int(i)]
    nl = int(lm.sum())
    top = float(p.max())
    print(f"  #{k+1}: KL={kls[i]:.4f}  street={street_names[s]:<7}  "
          f"n_legal={nl}  top_mass={top:.3f}  "
          f"top_action={int(np.argmax(p))}  legal_idx={np.flatnonzero(lm).tolist()}")

print("\nDONE D0+D2.")
