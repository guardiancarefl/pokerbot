"""Verification of the validated checkpoint's behavior on Session B frames.

Covers the five checks the user asked for after the live dryrun
(logs/live_dryrun_20260608_051036.jsonl) showed an AA fold at seq=227
plus other suspicious decisions:

  1. Checkpoint identity     — confirm md5 / config match
  2. Determinism             — query the policy net twice on the same
                                infoset → distributions identical
  3. Live-vs-replay          — replay the seq=227 spot through the same
                                pipeline the live consumer used and
                                compare the action to the logged FOLD
  4. Model-still-good        — re-run the live_1500 corpus through the
                                same pipeline and dump action histograms
  5. Aces (seq=227)          — full P(fold)/P(call)/P(raise) distribution
                                on the AsAh spot at MP-facing-BB, plus
                                the abstraction bucket the encoder lands
                                AsAh in at this exact infoset

Output: JSON to stdout + human-readable summary to stderr.

Usage:
    python scripts/verify_live_session_b.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyspiel

from src.nlhe.actions import (
    DiscreteAction, discretize_legal_actions,
)
from src.nlhe.cfr6 import _build_view_6max
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.integration.replay import deal_one_card_6max, replay_to_decision
from src.nlhe.integration.invariant import check_mid_hand_invariant
from src.nlhe.integration.scraper_schema import (
    ScraperDataQuality, ScraperParseError, ScraperSuspect,
    SessionTracker, parse_frame,
)


CHECKPOINT = "runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt"
ABSTRACTION = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCTURE = "configs/ignition_double_up_6max_turbo.yaml"
ALT_BAKEOFF_CKPT = "runs/k200_blueprint_ckpt_iter_2000.pt"  # the old inflated-BB one


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_solver():
    with open(ABSTRACTION, "rb") as f:
        abstr = pickle.load(f)
    structure = TournamentStructure.from_yaml(STRUCTURE)
    from scripts.eval_6max_self_play import _load_solver
    solver = _load_solver(CHECKPOINT, abstr, structure)
    return solver, abstr, structure


def policy_distribution(solver, state, hero_seat, dealer_seat, rng_seed=0):
    """Return policy distribution over the 7 discrete actions for the
    current decision at `state` (must be hero's turn).

    Returns (policy_vector_7, legal_mask_7, discrete_to_chip)."""
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
        hero_seat, features, legal_mask
    )
    return policy, legal_mask, discrete_to_chip, parsed


def build_seq227_state(structure):
    """Build the seq=227 infoset cleanly: 6-player, level 1, 1500 stacks,
    dealer=2, hero=0 (MP) holds AsAh, UTG (seat 5) folded, hero facing BB.

    Returns (state, hero_seat=0, dealer_seat=2)."""
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
    # Walk: deal all hole cards (12 chance nodes), then UTG (seat 5) folds,
    # then hero (seat 0) to act.
    max_steps = 80
    fold_idx = 0
    folds_before = [5]  # UTG (seat 5) folds; hero is MP (seat 0) next
    for _ in range(max_steps):
        if state.is_terminal():
            raise RuntimeError("state went terminal during setup")
        if state.is_chance_node():
            deal_one_card_6max(state, hero_seat=hero_seat,
                                hero_cards=hero_cards, target_board=())
            continue
        cp = state.current_player()
        if cp == hero_seat:
            return state, hero_seat, dealer_seat
        if fold_idx >= len(folds_before):
            raise RuntimeError(
                f"unexpected decision: cp={cp}, hero={hero_seat}, "
                f"folds_before exhausted")
        expected = folds_before[fold_idx]
        if cp != expected:
            raise RuntimeError(
                f"expected cp={expected} (fold #{fold_idx}), got {cp}")
        state.apply_action(0)  # fold
        fold_idx += 1
    raise RuntimeError("max_steps exhausted")


def replay_corpus(solver, structure, corpus_path, seed=0):
    """Run the full bridge+resolver pipeline on a raw scraper corpus and
    return per-decision results."""
    from scripts.eval_6max_self_play import _sample_action_from_policy

    tracker = SessionTracker()
    rng = random.Random(seed)
    out = []
    n_records = 0
    with open(corpus_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n_records += 1
            try:
                record = json.loads(line)
                frame = parse_frame(record)
            except (json.JSONDecodeError,
                    ScraperParseError, ScraperSuspect, ScraperDataQuality):
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
                    solver, parsed, pack.state, rng, mode="sample")
            except Exception:
                continue
            out.append({
                "line_no": line_no,
                "captured_at": frame.captured_at,
                "hero_cards": list(frame.hero_cards),
                "board": list(frame.board),
                "hero_stack": frame.stack[frame.hero_seat],
                "facing_bet": frame.hero_facing_bet,
                "street_idx": pack.street_idx,
                "chip_int": int(chip_int),
            })
    return out, n_records


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_1_checkpoint_identity():
    res = {
        "preserved_md5": md5_file(CHECKPOINT),
        "config_path":   CHECKPOINT.replace("ckpt_iter_1500.pt", "config.json"),
    }
    with open(res["config_path"]) as f:
        cfg_text = f.read()
    res["config_excerpt"] = cfg_text.strip()
    res["abstraction_path"] = ABSTRACTION
    res["abstraction_md5"] = md5_file(ABSTRACTION)
    # Compare against the alt candidate (the OLD inflated-BB blueprint
    # mentioned in earlier launch commands) to rule out a swap.
    try:
        res["alt_blueprint_md5"] = md5_file(ALT_BAKEOFF_CKPT)
    except FileNotFoundError:
        res["alt_blueprint_md5"] = None
    # Convergence log at iter_1500 — the validation signal
    conv_csv = ("runs/k200_real_ante_20260605_225847/convergence_log.csv")
    try:
        with open(conv_csv) as f:
            header = f.readline().strip().split(",")
            for line in f:
                row = line.strip().split(",")
                if row and row[0] == "1500":
                    res["convergence_iter1500"] = dict(zip(header, row))
                    break
    except FileNotFoundError:
        res["convergence_iter1500"] = None
    return res


def check_2_determinism(solver, structure):
    """Query the policy net on the seq=227 state twice with the SAME
    rng seed; the result must be bit-identical."""
    state, hero_seat, dealer_seat = build_seq227_state(structure)
    p1, m1, d1, parsed1 = policy_distribution(
        solver, state, hero_seat, dealer_seat, rng_seed=0)
    # Rebuild from scratch to make sure no state leaks between queries
    state2, _, _ = build_seq227_state(structure)
    p2, m2, d2, parsed2 = policy_distribution(
        solver, state2, hero_seat, dealer_seat, rng_seed=0)
    delta = float(np.max(np.abs(p1 - p2)))
    return {
        "policy_run1": [float(x) for x in p1.tolist()],
        "policy_run2": [float(x) for x in p2.tolist()],
        "max_abs_diff": delta,
        "bit_identical": bool(np.array_equal(p1, p2)),
        "legal_mask": [int(x) for x in m1.tolist()],
        "discrete_to_chip": {DiscreteAction(int(k)).name: int(v)
                              for k, v in d1.items()},
    }


def check_5_aces(solver, structure):
    """Full AsAh distribution at MP-facing-BB, ~deep stack, level 1.
    Also dumps the abstraction bucket id for AsAh at this infoset."""
    state, hero_seat, dealer_seat = build_seq227_state(structure)
    p, m, d, parsed = policy_distribution(
        solver, state, hero_seat, dealer_seat, rng_seed=0)
    # Bucket id from the parsed infoset (whatever the encoder sees)
    bucket = parsed.get("bucket_id")
    # Recover label per action
    actions = []
    for i in range(len(DiscreteAction)):
        if m[i] > 0:
            actions.append({
                "name":    DiscreteAction(i).name,
                "label":   DiscreteAction(i).label,
                "chip":    int(d.get(DiscreteAction(i), -1)),
                "prob":    float(p[i]),
            })
    actions.sort(key=lambda x: -x["prob"])
    fold_prob = float(p[int(DiscreteAction.FOLD)])
    call_prob = float(p[int(DiscreteAction.CALL)])
    any_raise_prob = float(sum(p[int(DiscreteAction(i))]
                                for i in range(len(DiscreteAction))
                                if DiscreteAction(i) not in (
                                    DiscreteAction.FOLD,
                                    DiscreteAction.CALL,
                                ) and m[int(DiscreteAction(i))] > 0))
    # Also re-query with a DIFFERENT rng seed to see whether the
    # encoder's MC equity sampling moves the bucket / distribution.
    state_alt, _, _ = build_seq227_state(structure)
    p_alt, m_alt, d_alt, parsed_alt = policy_distribution(
        solver, state_alt, hero_seat, dealer_seat, rng_seed=99)
    bucket_alt = parsed_alt.get("bucket_id")
    return {
        "bucket_id":              bucket,
        "bucket_id_alt_rng":      bucket_alt,
        "fold_prob":              fold_prob,
        "call_prob":              call_prob,
        "raise_total_prob":       any_raise_prob,
        "distribution":           actions,
        "hero_pot":               int(state.legal_actions()[0]) if state.legal_actions() else None,
        # Also report what argmax would have picked (the OLD deployment mode)
        "argmax_action":          DiscreteAction(int(np.argmax(p))).name,
        "argmax_chip":            int(d.get(DiscreteAction(int(np.argmax(p))), -1)),
    }


def check_4_model_still_good(solver, structure, corpus_path):
    """Re-run the live_1500 corpus through the SAME pipeline the live
    consumer uses; report action histograms."""
    results, n_records = replay_corpus(solver, structure, corpus_path, seed=0)

    # Action histograms
    histo = Counter()
    fold_aa = 0
    n_aa = 0
    fold_premium = 0
    n_premium = 0
    PREMIUM = {("As","Ah"),("As","Kh"),("Ah","Ks"),("Ks","Kh"),("Qs","Qh"),
               ("As","Ks")}  # small canonical set; treated unordered
    def is_premium(cards):
        s = tuple(sorted(cards))
        return s in {tuple(sorted(p)) for p in PREMIUM}
    def is_aa(cards):
        return tuple(sorted(cards)) == tuple(sorted(("As","Ah")))
    fold_calls_per_pos = []
    for r in results:
        chip = r["chip_int"]
        kind = ("fold" if chip == 0 else
                ("call" if chip == 1 else "raise"))
        histo[kind] += 1
        cards = tuple(r["hero_cards"])
        if r["street_idx"] == 0 and len(cards) == 2:
            if is_aa(cards):
                n_aa += 1
                if kind == "fold":
                    fold_aa += 1
            if is_premium(cards):
                n_premium += 1
                if kind == "fold":
                    fold_premium += 1
    return {
        "corpus":      corpus_path,
        "n_records":   n_records,
        "n_decisions": len(results),
        "histo":       dict(histo),
        "n_AA_preflop":      n_aa,
        "n_AA_folds":        fold_aa,
        "n_premium_preflop": n_premium,
        "n_premium_folds":   fold_premium,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/verify_session_b.json")
    ap.add_argument("--corpus", default="data/live_1500.jsonl",
                    help="corpus for check 4 (model-still-good)")
    args = ap.parse_args()

    print("[1/5] Checkpoint identity...", file=sys.stderr)
    r1 = check_1_checkpoint_identity()

    print("[load] Loading solver (live model)...", file=sys.stderr)
    solver, abstr, structure = load_solver()

    print("[5/5] Aces (seq=227) full distribution...", file=sys.stderr)
    r5 = check_5_aces(solver, structure)

    print("[2/5] Determinism on seq=227 infoset...", file=sys.stderr)
    r2 = check_2_determinism(solver, structure)

    print(f"[4/5] Re-running pipeline on {args.corpus}...", file=sys.stderr)
    r4 = check_4_model_still_good(solver, structure, args.corpus)

    # Check 3 deferred — Session B raw input frames were streamed over
    # TCP socket and not retained on disk, so we cannot replay them
    # 1-for-1. We can replay the equivalent INFOSET (check 5) and
    # demonstrate the live consumer's translation is faithful (no extra
    # transform between policy output and logged action).
    r3_note = (
        "DEFERRED: Session B raw frames came in via TCP socket and were "
        "not captured to disk. Live consumer's translation path "
        "(src/nlhe/integration/live_loop.py:_client_action_for_chip_int) "
        "is a 3-case dispatch with no model query — chip_int 0 → fold "
        "verbatim. If check 5 shows P(fold) > 0 on AsAh, the live FOLD "
        "is consistent with the model's distribution. To run a true "
        "live-vs-replay, the Windows scraper must record raw frames or "
        "the consumer must log the raw record alongside its decision."
    )

    report = {
        "checkpoint_used_by_live_run":      CHECKPOINT,
        "abstraction_used_by_live_run":     ABSTRACTION,
        "check_1_checkpoint_identity":      r1,
        "check_2_determinism":              r2,
        "check_3_live_vs_replay_note":      r3_note,
        "check_4_model_still_good":         r4,
        "check_5_aces_distribution":        r5,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.out}\n", file=sys.stderr)

    # Human summary
    print("=" * 70, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"[1] live ckpt   md5: {r1['preserved_md5']}", file=sys.stderr)
    print(f"    alt iter2000 md5: {r1['alt_blueprint_md5']}  "
          f"(different = good; we did NOT load the old inflated-BB ckpt)",
          file=sys.stderr)
    print(f"    abstraction  md5: {r1['abstraction_md5']}", file=sys.stderr)
    if r1.get("convergence_iter1500"):
        c = r1["convergence_iter1500"]
        print(f"    iter_1500 convergence row:", file=sys.stderr)
        print(f"      AA@60bb fold={c.get('AA_60bb_fold')}  "
              f"call={c.get('AA_60bb_call')}  "
              f"raise={c.get('AA_60bb_raise')}  "
              f"jam={c.get('AA_60bb_jam')}",
              file=sys.stderr)
    print(f"[2] determinism: max_abs_diff={r2['max_abs_diff']:.6e}  "
          f"bit_identical={r2['bit_identical']}", file=sys.stderr)
    print(f"[3] (deferred — Session B raw frames not on disk)",
          file=sys.stderr)
    print(f"[4] {r4['corpus']}: {r4['n_decisions']} decisions",
          file=sys.stderr)
    print(f"    histo={r4['histo']}", file=sys.stderr)
    print(f"    AA preflop folds: {r4['n_AA_folds']}/{r4['n_AA_preflop']}",
          file=sys.stderr)
    print(f"    Premium preflop folds: "
          f"{r4['n_premium_folds']}/{r4['n_premium_preflop']}",
          file=sys.stderr)
    print(f"[5] AsAh @ MP-facing-BB, deep:", file=sys.stderr)
    print(f"    fold={r5['fold_prob']:.4f}  call={r5['call_prob']:.4f}  "
          f"raise_total={r5['raise_total_prob']:.4f}", file=sys.stderr)
    print(f"    bucket_id={r5['bucket_id']} (alt rng → {r5['bucket_id_alt_rng']})",
          file=sys.stderr)
    print(f"    argmax: {r5['argmax_action']}  chip={r5['argmax_chip']}",
          file=sys.stderr)
    print(f"    distribution (descending prob):", file=sys.stderr)
    for a in r5["distribution"]:
        bar = "#" * int(a["prob"] * 50)
        print(f"      {a['name']:<14s}  chip={a['chip']:<6d}  "
              f"p={a['prob']:.4f}  {bar}", file=sys.stderr)


if __name__ == "__main__":
    main()
