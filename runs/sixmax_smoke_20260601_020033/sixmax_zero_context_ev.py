"""Floor-as-exploitability measurement.

Per user direction: KL was the wrong floor proxy (units error + median 0.019
≈ bar + soft-teacher tail is indifference-point noise). Re-anchor the floor
to head-to-head ICM-EV / chip-EV between the student's zero-context policy
and the blueprint, with a blueprint-vs-blueprint symmetry baseline as the
second benchmark.

The student's zero-context policy:
  - opp_stats = zeros (no observed opp stats)
  - tokens = [single current-state encoder vector], T=1
  - query_idx = 0
  - pad_mask = all False (no padding at T=1)
  - legal_mask from discretize_legal_actions
  - Returns: net.forward(...) -> policy_raw (the student's distilled
                                                policy_head output)
This is the Test-D6-verified path: forward accepts only the 5 declared
inputs; passing zeros for opp_stats is allowed and routes through the
trunk + policy_head exactly as designed for "no-context" inference.
Critically: this measures the STUDENT'S policy_head, NOT the anchor
(blueprint) -- which is what the user's "g_total=0 → policy_head only"
intent requires. (The blend at g_total=0 in my model.py would give the
blueprint passthrough; that is NOT what we want.)

Two benchmarks (per user's "two-benchmark minimum"):
  (1) Blueprint vs blueprint, hero in seat 0.   (symmetry check)
  (2) Student-zero-context vs blueprint, seat 0. (the measurement)

Both arms use the SAME hero seat (= 0) to isolate the student's policy
delta from positional bias (SB pays 50 chips/hand by structure).

Reported metric: chip-EV per hand (chips returned by state.returns() at
each terminal hand, for the hero seat). Convert to BB/hand and BB/100.
Stderr from N hands.

Read (pre-committed by user):
  Student EV >= Blueprint EV within noise -> floor SATISFIED.
  Student EV meaningfully WORSE than blueprint -> real undertraining;
                                                  then localize via pool
                                                  KL stratification.

Diffs to /tmp only. Read-only on prod code.
"""
from __future__ import annotations
import os, sys, time, random, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, "/home/quant/pokerbot")
from scripts import six_max_adaptive_smoke as sm  # triggers preflight + threading

import numpy as np
import torch
from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.cfr6 import _build_view_6max
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.adaptive.model import (
    Adaptive6MaxNet, F_TOKEN, F_STATS, NUM_ACTIONS,
)
import pyspiel

ANCHOR_DIR = "runs/six_max_20260530_034023_phase4f_dcfr_candC_k200"
ABSTRACTION_PKL = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STUDENT_CKPT = "runs/sixmax_smoke_20260601_020033/smoke_net.pt"
STUDENT_CONFIG = dict(d_model=128, num_layers=4, nhead=4, dim_ff=512)

HERO_SEAT = 0       # held constant in both benchmarks
N_HANDS = 3000      # per arm
HAND_TARGET = 1     # one hand per "match" — independent evaluation
BIG_BLIND = 100


class StudentZeroContextPolicy:
    """Wraps the trained Adaptive6MaxNet as a single-hand Policy at
    zero opponent context. Forward signature matches the smoke's
    blueprint_policy() call surface."""
    name = "student_zero_context"

    def __init__(self, net, encoder):
        self.net = net
        self.encoder = encoder
        self.net.eval()

    def select_action(self, parsed, state, rng, mode="sample") -> int:
        cur = int(parsed["current_player"])
        # Encode the current state from this seat's POV (deterministic
        # rng for postflop MC fallback to mirror the smoke's blueprint_policy)
        feat = self.encoder.encode_from_parsed(
            parsed, rng=random.Random(cur * 100003 + 7))
        view = _build_view_6max(state, parsed)
        d_to_chip = discretize_legal_actions(list(state.legal_actions()), view)
        legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for d in d_to_chip:
            legal_mask[int(d)] = True
        # Zero-context forward: T=1, opp_stats=zero
        tokens = torch.from_numpy(
            feat[None, None, :].astype(np.float32))      # [1, 1, 236]
        pad_mask = torch.zeros(1, 1, dtype=torch.bool)
        query_idx = torch.zeros(1, dtype=torch.long)
        legal_mask_t = torch.from_numpy(legal_mask[None, :])
        opp_stats_t = torch.zeros(1, F_STATS, dtype=torch.float32)
        with torch.no_grad():
            policy_raw, _, _ = self.net(
                tokens, pad_mask, query_idx, legal_mask_t, opp_stats_t)
        probs = policy_raw[0].cpu().numpy()
        legal_keys = list(d_to_chip.keys())
        ws = [float(probs[int(k)]) for k in legal_keys]
        total = sum(ws)
        if total > 0:
            choice = rng.choices(legal_keys, weights=ws, k=1)[0]
        else:
            choice = rng.choice(legal_keys)
        chip = d_to_chip.get(DiscreteAction(int(choice)))
        if chip is None:
            chip = next(iter(d_to_chip.values()))
        return int(chip)


def run_eval_hands(game, hero_policy, opp_policy, hero_seat: int,
                   n_hands: int, seed: int):
    """Run n_hands independent 6-max hands. hero plays hero_policy in
    hero_seat; all other seats play opp_policy. Returns per-hand chip
    deltas (state.returns() for hero_seat at terminal)."""
    py_rng = random.Random(seed)
    deltas = []
    t_start = time.time()
    for h in range(n_hands):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outs = state.chance_outcomes()
                a = py_rng.choices(
                    [o for o, _ in outs],
                    weights=[p for _, p in outs], k=1)[0]
                state.apply_action(int(a))
                continue
            cur = state.current_player()
            parsed = parse_state_6max(state)
            if cur == hero_seat:
                chip = hero_policy.select_action(parsed, state, py_rng)
            else:
                chip = opp_policy.select_action(parsed, state, py_rng)
            state.apply_action(int(chip))
        rets = state.returns()
        deltas.append(float(rets[hero_seat]))
        if (h + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (h + 1) / elapsed
            print(f"    [{h+1}/{n_hands} hands] {elapsed:.1f}s  "
                  f"rate={rate:.1f} hand/s  "
                  f"running_mean={np.mean(deltas):+.2f} chips/hand",
                  flush=True)
    return np.array(deltas)


def report(label, deltas, baseline_mean=None):
    n = len(deltas)
    mean_chips = float(np.mean(deltas))
    stderr_chips = float(np.std(deltas, ddof=1) / np.sqrt(n))
    mean_bb_hand = mean_chips / BIG_BLIND
    mean_bb100 = mean_chips  # since BB = 100 chips, mean_chips/100 * 100 = mean_chips
    sigma_from_zero = mean_chips / stderr_chips if stderr_chips > 0 else 0.0
    msg = (f"  [{label}] n={n}  "
           f"mean={mean_chips:+.3f} chips/hand "
           f"({mean_bb_hand:+.4f} BB/hand,  {mean_bb100:+.2f} BB/100)  "
           f"stderr={stderr_chips:.3f} chips  "
           f"sigma_vs_zero={sigma_from_zero:+.2f}")
    print(msg)
    if baseline_mean is not None:
        delta = mean_chips - baseline_mean
        delta_se = stderr_chips  # baseline stderr ignored in this rough comparison
        sigma_vs_baseline = (delta / delta_se) if delta_se > 0 else 0.0
        print(f"       delta_vs_baseline={delta:+.3f} chips/hand  "
              f"(sigma vs baseline ~{sigma_vs_baseline:+.2f})")
    return {
        "label": label, "n": n,
        "mean_chips": mean_chips, "stderr_chips": stderr_chips,
        "mean_bb_per_hand": mean_bb_hand,
        "mean_bb_per_100": mean_bb100,
        "sigma_vs_zero": sigma_from_zero,
    }


def main():
    from pathlib import Path
    print("=" * 70)
    print("Floor-as-exploitability measurement (head-to-head fallback)")
    print(f"  hero_seat={HERO_SEAT}  N_HANDS={N_HANDS} per arm  BB={BIG_BLIND}")
    print("=" * 70)

    t0 = time.time()
    solver = sm.load_blueprint(Path(ANCHOR_DIR), Path(ABSTRACTION_PKL))
    print(f"[blueprint] loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    student_net = Adaptive6MaxNet(**STUDENT_CONFIG)
    ckpt = torch.load(STUDENT_CKPT, weights_only=False, map_location="cpu")
    student_net.load_state_dict(ckpt["state_dict"])
    student_net.eval()
    n_params = sum(p.numel() for p in student_net.parameters())
    print(f"[student] loaded {STUDENT_CKPT}  ({n_params:,} params)  "
          f"in {time.time()-t0:.1f}s")

    bp_pol = sm._BlueprintAsOpp(solver)
    student_pol = StudentZeroContextPolicy(student_net, solver.encoder)

    # Benchmark 1: blueprint vs blueprint, hero in seat 0
    print("")
    print(f"--- Benchmark (1): BLUEPRINT vs BLUEPRINT  (hero seat {HERO_SEAT}) ---")
    t0 = time.time()
    deltas_bp = run_eval_hands(
        game=sm.GAME, hero_policy=bp_pol, opp_policy=bp_pol,
        hero_seat=HERO_SEAT, n_hands=N_HANDS, seed=2026)
    print(f"  (wall {time.time()-t0:.1f}s)")
    bp_result = report("blueprint_vs_blueprint", deltas_bp)

    # Benchmark 2: student-zero-context vs blueprint
    print("")
    print(f"--- Benchmark (2): STUDENT-zero-context vs BLUEPRINT  "
          f"(hero seat {HERO_SEAT}) ---")
    t0 = time.time()
    deltas_st = run_eval_hands(
        game=sm.GAME, hero_policy=student_pol, opp_policy=bp_pol,
        hero_seat=HERO_SEAT, n_hands=N_HANDS, seed=4052)
    print(f"  (wall {time.time()-t0:.1f}s)")
    st_result = report("student_vs_blueprint", deltas_st,
                        baseline_mean=bp_result["mean_chips"])

    # Verdict
    print("")
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    delta = st_result["mean_chips"] - bp_result["mean_chips"]
    # Combined stderr (independent samples, sigma=sqrt(se_a^2 + se_b^2))
    combined_se = float(np.sqrt(
        st_result["stderr_chips"] ** 2 + bp_result["stderr_chips"] ** 2))
    sigma = delta / combined_se if combined_se > 0 else 0.0
    print(f"  delta (student_mean - blueprint_mean) = {delta:+.3f} chips/hand "
          f"({delta/BIG_BLIND:+.4f} BB/hand,  {delta:+.2f} BB/100)")
    print(f"  combined stderr = {combined_se:.3f} chips/hand")
    print(f"  sigma (delta / combined_se) = {sigma:+.2f}")
    print("")
    if abs(sigma) < 2.0:
        print("  -> Student exploitability is WITHIN MEASUREMENT NOISE of blueprint.")
        print("     Floor SATISFIED. The KL tail is indifference-point noise; "
              "re-anchor the floor metric to head-to-head EV.")
    elif sigma < -2.0:
        print(f"  -> Student is meaningfully WORSE than blueprint "
              f"(sigma {sigma:.1f}).")
        print("     Real undertraining. Localize via pool KL stratification + "
              "matched-distribution eval.")
    else:
        print(f"  -> Student is meaningfully BETTER than blueprint "
              f"(sigma {sigma:.1f}). Unexpected -- investigate.")

    # Save
    out = {
        "hero_seat": HERO_SEAT,
        "n_hands_per_arm": N_HANDS,
        "big_blind": BIG_BLIND,
        "blueprint_vs_blueprint": bp_result,
        "student_vs_blueprint": st_result,
        "delta_chips": delta,
        "combined_stderr_chips": combined_se,
        "sigma": sigma,
    }
    with open("/tmp/sixmax_zero_context_ev.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved /tmp/sixmax_zero_context_ev.json")


if __name__ == "__main__":
    main()
