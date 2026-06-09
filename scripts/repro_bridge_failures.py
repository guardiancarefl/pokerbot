"""Reproduce seq=170 + seq=192 reconstruction failures and seq=48
check-when-free FOLD anomaly from logs/live_dryrun_20260608_152756.jsonl.

Replays the full session through SessionTracker so per-hand override
matches what live_dryrun saw; at the target seqs, dumps the derived
action sequence, the OpenSpiel state, and the scraper view side by
side. For seq=48 also queries the policy distribution.
"""
from __future__ import annotations

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
from src.nlhe.integration.invariant import (
    check_mid_hand_invariant, openspiel_to_scraper_view,
    compute_ante_absorbed_mask,
)
from src.nlhe.integration.replay import replay_to_decision, ReplayError
from src.nlhe.integration.scraper_schema import (
    ScraperDataQuality, ScraperParseError, ScraperSuspect,
    SessionTracker, derive_action_sequence, parse_frame,
    _derive_pre_hand_and_preflop_commit_simple_model,
    _repair_folded_from_chip_deductions, _alive_at_hand_start_mask,
    _street_idx_from_board,
)

LOG = "logs/live_dryrun_20260608_152756.jsonl"
STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"
TARGET_SEQS = {48, 170, 192}


def dump_frame(frame, label):
    print(f"  -- ScraperFrame [{label}] --")
    print(f"    dealer_seat={frame.dealer_seat}  hero_seat={frame.hero_seat}  "
          f"hero_cards={frame.hero_cards}")
    print(f"    blinds: sb={frame.blinds.sb} bb={frame.blinds.bb} "
          f"ante={frame.blinds.ante}")
    print(f"    board={frame.board}  street_idx={_street_idx_from_board(frame.board)}")
    print(f"    alive    : {list(frame.alive)}")
    print(f"    folded   : {list(frame.folded)}")
    print(f"    stack    : {list(frame.stack)}")
    print(f"    bet      : {list(frame.bet)}")
    print(f"    pot_total={frame.pot_total}  controls_present={frame.controls_present}  "
          f"facing_bet={frame.hero_facing_bet}")


def dump_action_seq(action_seq, pack, pre, frame):
    print("  -- derive_action_sequence emission --")
    print(f"    n_emissions={len(action_seq)}")
    for i, (seat, chip) in enumerate(action_seq):
        print(f"      [{i:>2d}] seat={seat}  chip_int={chip}")
    print(f"    pre_hand_stacks (used to build game_str): {pack.pre_hand_stacks}")
    print(f"    sb_seat={pack.sb_seat}  bb_seat={pack.bb_seat}  "
          f"n_alive={pack.n_alive}  "
          f"preflop_commit_per_alive={pack.preflop_commit_per_alive}")


def dump_state_vs_scraper(frame, pack):
    parsed = parse_state_6max(pack.state, observer=frame.hero_seat)
    parsed["dealer_seat"] = frame.dealer_seat
    ante = pack.blind_level.ante
    alive_at_start = _alive_at_hand_start_mask(frame, pack.pre_hand_stacks)
    folded_rep = _repair_folded_from_chip_deductions(
        frame, pack.sb_seat, pack.bb_seat,
        pre_hand_override=pack.pre_hand_stacks,
    )
    still = tuple(alive_at_start[i] and not folded_rep[i] for i in range(6))
    still_in_hand = tuple(alive_at_start[i] and not folded_rep[i] for i in range(6))
    absorbed = compute_ante_absorbed_mask(
        frame, still_in_hand=still_in_hand,
        street_idx=_street_idx_from_board(frame.board),
        sb_seat=pack.sb_seat, bb_seat=pack.bb_seat,
        sb_amount=pack.blind_level.small_blind,
        bb_amount=pack.blind_level.big_blind,
        pre_hand=pack.pre_hand_stacks, ante=ante,
    )
    busted = any(alive_at_start[i] and not frame.alive[i] for i in range(6))
    preflop_max_chip_int = pack.blind_level.big_blind
    view = openspiel_to_scraper_view(
        parsed,
        frame_alive=tuple(alive_at_start),
        still_in_hand=still,
        absorbed=absorbed,
        ante=ante,
        preflop_commit_per_alive=pack.preflop_commit_per_alive,
        busted_mid_hand_exists=busted,
        preflop_max_chip_int=preflop_max_chip_int,
    )
    print("  -- OpenSpiel-view (recon) vs Scraper (truth) --")
    print(f"    {'seat':<4s} {'alive':<5s} {'folded_rep':<10s} "
          f"{'absorbed':<8s} {'stk_recon':>9s} {'stk_truth':>9s} "
          f"{'bet_recon':>9s} {'bet_truth':>9s}")
    for i in range(6):
        print(f"    {i:<4d} {str(bool(alive_at_start[i])):<5s} "
              f"{str(bool(folded_rep[i])):<10s} {str(bool(absorbed[i])):<8s} "
              f"{view['stack'][i]:>9d} {frame.stack[i]:>9d} "
              f"{view['bet'][i]:>9d} {frame.bet[i]:>9d}")
    print(f"    pot recon={view['pot']}  scraper={frame.pot_total}")
    print(f"    current_player recon={view['current_player']}  hero={frame.hero_seat}")


def run_target(rec_by_seq, structure, solver, target_seq):
    print(f"\n{'='*72}")
    print(f"FRAME seq={target_seq}")
    print(f"{'='*72}")
    rec = rec_by_seq[target_seq]
    raw = rec["raw_record"]
    try:
        frame = parse_frame(raw)
    except Exception as e:
        print(f"  parse_frame failed: {type(e).__name__}: {e}")
        return
    dump_frame(frame, f"seq={target_seq}")

    # Replay all earlier frames into a fresh tracker
    tracker = SessionTracker()
    for seq in sorted(rec_by_seq):
        if seq > target_seq:
            break
        try:
            tf = parse_frame(rec_by_seq[seq]["raw_record"])
            tracker.observe(tf)
        except Exception:
            continue
    pre_hand_override = tracker.pre_hand_for(frame)
    print(f"\n  tracker.pre_hand_override = {pre_hand_override}")

    # Run replay+invariant
    try:
        pack = replay_to_decision(frame, structure,
                                   pre_hand_override=pre_hand_override)
    except ReplayError as e:
        print(f"  replay_to_decision RAISED ReplayError: {e}")
        # Try without override
        try:
            pack = replay_to_decision(frame, structure,
                                       pre_hand_override=None)
            print("  (succeeded without override)")
        except ReplayError as e2:
            print(f"  also failed without override: {e2}")
            return
    print(f"\n  replay_to_decision SUCCESS  street_idx={pack.street_idx}  "
          f"current_player={pack.final_current_player}")

    # Re-derive action sequence ourselves so we can dump it
    pre = pack.pre_hand_stacks
    try:
        action_seq = derive_action_sequence(
            frame, pre_hand_override=pre_hand_override)
    except Exception as e:
        print(f"  derive_action_sequence raised {type(e).__name__}: {e}")
        action_seq = []
    dump_action_seq(action_seq, pack, pre, frame)

    # Invariant check
    inv = check_mid_hand_invariant(frame, pack)
    print(f"\n  invariant.ok={inv.ok}  n_diffs={len(inv.deltas)}")
    for d in inv.deltas:
        print(f"    delta: {d}")

    ante = pack.blind_level.ante
    # Raw OpenSpiel-side parsed for direct comparison
    raw_parsed = parse_state_6max(pack.state, observer=frame.hero_seat)
    print(f"\n  -- OpenSpiel parsed (raw) --")
    print(f"    contribution = {list(raw_parsed['contribution'])}")
    print(f"    money        = {list(raw_parsed['money'])}")
    print(f"    street_idx   = {raw_parsed['street_idx']}")
    print(f"    current_player = {raw_parsed['current_player']}")

    # Replicate invariant's exact computation
    sb_seat_inv = pack.sb_seat
    bb_seat_inv = pack.bb_seat
    folded_inv = _repair_folded_from_chip_deductions(
        frame, sb_seat=sb_seat_inv, bb_seat=bb_seat_inv,
        pre_hand_override=pack.pre_hand_stacks)
    still_inv = tuple(frame.alive[i] and not folded_inv[i] for i in range(6))
    absorbed_inv = compute_ante_absorbed_mask(
        frame, still_inv, street_idx=pack.street_idx,
        sb_seat=sb_seat_inv, bb_seat=bb_seat_inv,
        sb_amount=pack.blind_level.small_blind,
        bb_amount=pack.blind_level.big_blind,
        pre_hand=pack.pre_hand_stacks, ante=ante,
    )
    all_in_chips = []
    for i in range(6):
        pre_i = int(pack.pre_hand_stacks[i])
        if pre_i <= 0:
            continue
        if not frame.alive[i]:
            all_in_chips.append(pre_i)
        elif int(frame.stack[i]) == 0 and int(frame.bet[i]) > 0:
            all_in_chips.append(pre_i)
    busted_exists = len(all_in_chips) > 0
    pmci = max(all_in_chips) if all_in_chips else 0
    print(f"\n  -- INVARIANT-EXACT VIEW --")
    print(f"    folded_repaired      = {folded_inv}")
    print(f"    still_in_hand        = {still_inv}")
    print(f"    absorbed             = {absorbed_inv}")
    print(f"    all_in_chip_ints     = {all_in_chips}")
    print(f"    busted_mid_hand_ex.  = {busted_exists}")
    print(f"    preflop_max_chip_int = {pmci}")
    view_inv = openspiel_to_scraper_view(
        raw_parsed,
        frame_alive=frame.alive,
        still_in_hand=still_inv,
        absorbed=absorbed_inv,
        ante=ante,
        preflop_commit_per_alive=pack.preflop_commit_per_alive,
        busted_mid_hand_exists=busted_exists,
        preflop_max_chip_int=pmci,
    )
    print(f"    view stack = {list(view_inv['stack'])}")
    print(f"    view bet   = {list(view_inv['bet'])}")
    print(f"    view pot   = {view_inv['pot']}")

    # State-vs-scraper full diff
    dump_state_vs_scraper(frame, pack)

    # Per the user's special ask: for seq=48 query the policy distribution
    if target_seq == 48 and solver is not None:
        print("\n  -- POLICY DISTRIBUTION (seq=48 check-was-free FOLD) --")
        parsed = parse_state_6max(pack.state, observer=frame.hero_seat)
        parsed["dealer_seat"] = frame.dealer_seat
        legal_chip = list(pack.state.legal_actions())
        print(f"    OpenSpiel legal_actions: {legal_chip}")
        view = _build_view_6max(pack.state, parsed)
        d2c = discretize_legal_actions(legal_chip, view)
        print(f"    discretize_legal_actions: { {da.name: chip for da, chip in d2c.items()} }")
        N = len(DiscreteAction)
        mask = np.zeros(N, dtype=np.float32)
        for da in d2c:
            mask[int(da)] = 1.0
        rng = random.Random(0)
        encoded = solver.encoder.encode_from_parsed(parsed, rng=rng)
        features = np.asarray(encoded, dtype=np.float32)
        policy = solver.policy_nets.inference_policy(
            frame.hero_seat, features, mask)
        print(f"    policy:")
        for i in range(N):
            if mask[i] > 0:
                print(f"      {DiscreteAction(i).name:<10s}  "
                      f"p={float(policy[i]):.4f}")


def main():
    print(f"Loading structure from {STRUCTURE_YAML}...")
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)

    # Load log
    print(f"Loading log {LOG}...")
    rec_by_seq = {}
    with open(LOG) as f:
        for line in f:
            r = json.loads(line)
            rec_by_seq[int(r["seq"])] = r
    print(f"  {len(rec_by_seq)} records")

    # Lazy-load solver (only needed for seq=48 policy query)
    print(f"Loading solver (for seq=48 policy query)...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.eval_6max_self_play import _load_solver
    abstr_path = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
    ckpt_path = "runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt"
    with open(abstr_path, "rb") as af:
        abstr = pickle.load(af)
    solver = _load_solver(ckpt_path, abstr, structure)
    print("  done.")

    for seq in (170, 192, 48):
        run_target(rec_by_seq, structure, solver, seq)

    print(f"\n{'='*72}")
    print("done.")


if __name__ == "__main__":
    main()
