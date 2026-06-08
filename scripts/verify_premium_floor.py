"""End-to-end verification of the AA/KK preflop FOLD floor.

Three checks:
  (A) seq=227 AsAh-at-MP-facing-BB infoset: full policy distribution
      BEFORE vs AFTER the floor — fold mass should move to 0 and the
      remaining mass should renormalize correctly.
  (B) Stress-sample the floor-on AA infoset many times — fold count
      must be 0 (the live consumer's deployment sampler should never
      draw FOLD once the floor is applied).
  (C) Replay data/live_1500.jsonl through the full live pipeline
      BOTH with the floor and without it (by toggling the filter).
      Diff: only spots where hero holds AA/KK at street_idx=0 may
      produce a different chip action.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyspiel

from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.cfr6 import _build_view_6max
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.integration.invariant import check_mid_hand_invariant
from src.nlhe.integration.live_loop import apply_aa_kk_preflop_floor
from src.nlhe.integration.replay import deal_one_card_6max, replay_to_decision
from src.nlhe.integration.scraper_schema import (
    ScraperDataQuality, ScraperParseError, ScraperSuspect,
    SessionTracker, parse_frame,
)


CHECKPOINT = "runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt"
ABSTRACTION = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCTURE = "configs/ignition_double_up_6max_turbo.yaml"


def load_solver():
    with open(ABSTRACTION, "rb") as f:
        abstr = pickle.load(f)
    structure = TournamentStructure.from_yaml(STRUCTURE)
    from scripts.eval_6max_self_play import _load_solver
    solver = _load_solver(CHECKPOINT, abstr, structure)
    return solver, abstr, structure


def build_seq227_state(structure):
    """Reconstruct seq=227 infoset: AsAh, dealer=2, hero seat 0 (MP),
    UTG (seat 5) folded, level 1, 1500 stacks all."""
    blind_level = structure.level(1)
    stacks = [1500] * 6
    dealer_seat = 2
    gs = structure.to_inner_game_string_for_state(
        blind_level=blind_level, stacks=stacks, dealer_seat=dealer_seat,
    )
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    hero_seat = 0
    hero_cards = ("As", "Ah")
    folds_before = [5]
    fold_idx = 0
    for _ in range(80):
        if state.is_terminal():
            raise RuntimeError("terminal")
        if state.is_chance_node():
            deal_one_card_6max(state, hero_seat=hero_seat,
                                hero_cards=hero_cards, target_board=())
            continue
        cp = state.current_player()
        if cp == hero_seat:
            return state, hero_seat, dealer_seat
        assert cp == folds_before[fold_idx], (cp, fold_idx)
        state.apply_action(0)
        fold_idx += 1
    raise RuntimeError("exhausted")


def get_policy(solver, state, hero_seat, dealer_seat, rng_seed=0):
    parsed = parse_state_6max(state, observer=hero_seat)
    parsed["dealer_seat"] = dealer_seat
    legal_chip = list(state.legal_actions())
    view = _build_view_6max(state, parsed)
    discrete_to_chip = discretize_legal_actions(legal_chip, view)
    legal_mask = np.zeros(len(DiscreteAction), dtype=np.float32)
    for da in discrete_to_chip:
        legal_mask[int(da)] = 1.0
    rng = random.Random(rng_seed)
    encoded = solver.encoder.encode_from_parsed(parsed, rng=rng)
    features = np.asarray(encoded, dtype=np.float32)
    policy = solver.policy_nets.inference_policy(
        hero_seat, features, legal_mask)
    return policy, legal_mask, parsed, discrete_to_chip


def check_a_seq227_distribution(solver, structure):
    state, hero_seat, dealer_seat = build_seq227_state(structure)
    p_raw, mask, parsed, d2c = get_policy(
        solver, state, hero_seat, dealer_seat, rng_seed=0)
    # Build a fresh state for the floored path so no state mutates
    state2, _, _ = build_seq227_state(structure)
    p_raw2, mask2, parsed2, d2c2 = get_policy(
        solver, state2, hero_seat, dealer_seat, rng_seed=0)
    p_floor = apply_aa_kk_preflop_floor(p_raw2, mask2, parsed2, state2)

    return {
        "no_floor": {
            DiscreteAction(i).name: float(p_raw[i])
            for i in range(len(DiscreteAction)) if mask[i] > 0
        },
        "with_floor": {
            DiscreteAction(i).name: float(p_floor[i])
            for i in range(len(DiscreteAction)) if mask2[i] > 0
        },
        "fold_no_floor": float(p_raw[int(DiscreteAction.FOLD)]),
        "fold_with_floor": float(p_floor[int(DiscreteAction.FOLD)]),
        "sum_no_floor":    float(p_raw.sum()),
        "sum_with_floor":  float(p_floor.sum()),
    }


def check_b_stress_sample_floor(solver, structure, n=20000):
    """Sample n times from the floored distribution. FOLD count must be 0."""
    state, hero_seat, dealer_seat = build_seq227_state(structure)
    p_raw, mask, parsed, d2c = get_policy(
        solver, state, hero_seat, dealer_seat, rng_seed=0)
    p = apply_aa_kk_preflop_floor(p_raw, mask, parsed, state)

    rng = random.Random(12345)
    counts = {DiscreteAction(i).name: 0 for i in range(len(DiscreteAction))
              if mask[i] > 0}
    for _ in range(n):
        idx = rng.choices(
            range(len(DiscreteAction)),
            weights=p.tolist(), k=1)[0]
        counts[DiscreteAction(idx).name] += 1
    return {
        "n_samples": n,
        "counts": counts,
        "fold_count": counts.get("FOLD", 0),
    }


def replay_corpus(solver, structure, corpus, with_floor: bool, seed=0):
    from scripts.eval_6max_self_play import _sample_action_from_policy
    pol_filter = apply_aa_kk_preflop_floor if with_floor else None

    tracker = SessionTracker()
    rng = random.Random(seed)
    out = []
    with open(corpus) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                frame = parse_frame(record)
            except (json.JSONDecodeError, ScraperParseError,
                    ScraperSuspect, ScraperDataQuality):
                continue
            tracker.observe(frame)
            if not frame.controls_present:
                continue
            if frame.alive[frame.hero_seat] and not frame.hero_cards:
                continue
            if frame.pot_total <= 0:
                continue
            pre_hand_override = tracker.pre_hand_for(frame)
            try:
                pack = replay_to_decision(
                    frame, structure, pre_hand_override=pre_hand_override)
            except Exception:
                if pre_hand_override is not None:
                    try:
                        pack = replay_to_decision(
                            frame, structure, pre_hand_override=None)
                    except Exception:
                        continue
                else:
                    continue
            inv = check_mid_hand_invariant(frame, pack)
            if not inv.ok:
                continue
            try:
                parsed = parse_state_6max(pack.state, observer=frame.hero_seat)
                parsed["dealer_seat"] = frame.dealer_seat
                chip_int = _sample_action_from_policy(
                    solver, parsed, pack.state, rng, mode="sample",
                    policy_filter=pol_filter,
                )
            except Exception:
                continue
            out.append({
                "line_no":     line_no,
                "captured_at": frame.captured_at,
                "hero_cards":  list(frame.hero_cards),
                "street_idx":  pack.street_idx,
                "chip_int":    int(chip_int),
            })
    return out


def check_c_corpus_diff(solver, structure, corpus_path):
    """Replay corpus with-floor and without-floor (same seed). Diff
    must be limited to (AA-or-KK at street_idx == 0) decisions."""
    no_floor = replay_corpus(solver, structure, corpus_path,
                             with_floor=False, seed=0)
    with_floor = replay_corpus(solver, structure, corpus_path,
                               with_floor=True, seed=0)

    # Key by (line_no, captured_at) for alignment
    nf_idx = {(r["line_no"], r["captured_at"]): r for r in no_floor}
    wf_idx = {(r["line_no"], r["captured_at"]): r for r in with_floor}

    common_keys = set(nf_idx) & set(wf_idx)

    diffs = []
    unaligned_changes = 0
    for k in sorted(common_keys):
        a = nf_idx[k]
        b = wf_idx[k]
        if a["chip_int"] != b["chip_int"]:
            is_premium = (
                a["street_idx"] == 0 and
                len(a["hero_cards"]) == 2 and
                a["hero_cards"][0][0] == a["hero_cards"][1][0] and
                a["hero_cards"][0][0] in ("A", "K")
            )
            diffs.append({
                "line_no": a["line_no"],
                "captured_at": a["captured_at"],
                "hero_cards": a["hero_cards"],
                "street_idx": a["street_idx"],
                "no_floor_chip": a["chip_int"],
                "with_floor_chip": b["chip_int"],
                "is_aa_kk_preflop": is_premium,
            })
            if not is_premium:
                unaligned_changes += 1

    return {
        "n_decisions_no_floor":   len(no_floor),
        "n_decisions_with_floor": len(with_floor),
        "n_common":               len(common_keys),
        "n_diffs":                len(diffs),
        "n_diffs_non_premium":    unaligned_changes,
        "diffs":                  diffs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/verify_floor.json")
    ap.add_argument("--corpus", default="data/live_1500.jsonl")
    ap.add_argument("--stress-n", type=int, default=20000)
    args = ap.parse_args()

    print("Loading solver...", file=sys.stderr)
    solver, abstr, structure = load_solver()

    print("[A] seq=227 distribution before/after floor...", file=sys.stderr)
    ra = check_a_seq227_distribution(solver, structure)

    print(f"[B] Stress-sampling floored AA n={args.stress_n}...",
          file=sys.stderr)
    rb = check_b_stress_sample_floor(solver, structure, n=args.stress_n)

    print(f"[C] Corpus diff (with-floor vs without) on {args.corpus}...",
          file=sys.stderr)
    rc = check_c_corpus_diff(solver, structure, args.corpus)

    report = {"check_A_seq227": ra, "check_B_stress": rb, "check_C_diff": rc}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport: {args.out}\n", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("(A) seq=227 distribution", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  P(fold) no_floor:   {ra['fold_no_floor']:.4f}", file=sys.stderr)
    print(f"  P(fold) with_floor: {ra['fold_with_floor']:.4f}",
          file=sys.stderr)
    print(f"  sum no_floor:       {ra['sum_no_floor']:.6f}",
          file=sys.stderr)
    print(f"  sum with_floor:     {ra['sum_with_floor']:.6f}",
          file=sys.stderr)
    print("  per-action probs:", file=sys.stderr)
    print(f"    {'action':<12s}  {'no_floor':>10s}  {'with_floor':>10s}  "
          f"{'scale':>8s}", file=sys.stderr)
    for k in ra["no_floor"]:
        a = ra["no_floor"][k]
        b = ra["with_floor"][k]
        scale = (b / a) if a > 0 else float("inf")
        print(f"    {k:<12s}  {a:>10.4f}  {b:>10.4f}  {scale:>8.4f}",
              file=sys.stderr)

    print("=" * 70, file=sys.stderr)
    print("(B) Stress sample", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  n={rb['n_samples']}  fold_count={rb['fold_count']}  "
          f"(must be 0)", file=sys.stderr)
    for k, v in sorted(rb["counts"].items(), key=lambda kv: -kv[1]):
        pct = 100 * v / rb["n_samples"]
        print(f"    {k:<12s}  {v:>6d}  {pct:5.2f}%", file=sys.stderr)

    print("=" * 70, file=sys.stderr)
    print(f"(C) Corpus diff (corpus={args.corpus})", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  n_decisions_no_floor:   {rc['n_decisions_no_floor']}",
          file=sys.stderr)
    print(f"  n_decisions_with_floor: {rc['n_decisions_with_floor']}",
          file=sys.stderr)
    print(f"  n_common:               {rc['n_common']}", file=sys.stderr)
    print(f"  n_diffs:                {rc['n_diffs']}", file=sys.stderr)
    print(f"  n_diffs_non_premium:    {rc['n_diffs_non_premium']}",
          file=sys.stderr)
    if rc["diffs"]:
        for d in rc["diffs"]:
            tag = "AA/KK preflop ✓" if d["is_aa_kk_preflop"] else "*** OTHER ***"
            print(f"    line {d['line_no']:<5d}  {d['hero_cards']}  "
                  f"street={d['street_idx']}  "
                  f"chip: {d['no_floor_chip']} -> {d['with_floor_chip']}  "
                  f"{tag}", file=sys.stderr)

    # Pass / fail summary
    print("\n" + "=" * 70, file=sys.stderr)
    print("VERDICT", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    ok_A = (abs(ra["fold_with_floor"]) < 1e-6 and
            abs(ra["sum_with_floor"] - 1.0) < 1e-5)
    ok_B = (rb["fold_count"] == 0)
    ok_C = (rc["n_diffs_non_premium"] == 0)
    print(f"  [A] P(fold)=0 + sum=1:        {'PASS' if ok_A else 'FAIL'}",
          file=sys.stderr)
    print(f"  [B] zero folds across stress: {'PASS' if ok_B else 'FAIL'}",
          file=sys.stderr)
    print(f"  [C] only AA/KK preflop diff:  {'PASS' if ok_C else 'FAIL'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
