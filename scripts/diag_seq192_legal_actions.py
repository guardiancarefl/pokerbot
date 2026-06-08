"""Diagnose seq=192 forced-all-in fallback. Walk replay up to BB's turn
decision; dump OpenSpiel's legal_actions(); report which BB-side
chip_int the engine actually accepts: 1515 (voluntary-only) or 1525
(includes ante) or both.

If 1515 IS legal → the derive emission is correct, fallback should
not fire, defect is in fallback detection.
If 1515 NOT legal but 1525 IS → OpenSpiel's chip_int convention here
includes the ante for the absorbed BB seat; fallback inflation is
correct but the view branch needs to account for it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pyspiel

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.integration.replay import deal_one_card_6max
from src.nlhe.integration.scraper_schema import (
    SessionTracker, derive_action_sequence, parse_frame,
    _derive_pre_hand_and_preflop_commit_simple_model,
)


LOG = "logs/live_dryrun_20260608_152756.jsonl"
STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"


def main():
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    rec_by_seq = {}
    with open(LOG) as f:
        for line in f:
            r = json.loads(line)
            rec_by_seq[int(r["seq"])] = r

    # Replay all earlier frames to populate SessionTracker, then get
    # the override for seq=192.
    tracker = SessionTracker()
    for seq in sorted(rec_by_seq):
        if seq > 192:
            break
        try:
            tf = parse_frame(rec_by_seq[seq]["raw_record"])
            tracker.observe(tf)
        except Exception:
            continue
    rec192 = rec_by_seq[192]
    frame192 = parse_frame(rec192["raw_record"])
    pre_hand_override = tracker.pre_hand_for(frame192)
    print(f"seq=192 frame:")
    print(f"  hero_seat={frame192.hero_seat}  dealer_seat={frame192.dealer_seat}")
    print(f"  board={frame192.board}")
    print(f"  stacks: {list(frame192.stack)}")
    print(f"  bets:   {list(frame192.bet)}")
    print(f"  blinds: sb={frame192.blinds.sb} bb={frame192.blinds.bb} "
          f"ante={frame192.blinds.ante}")
    print(f"  pre_hand_override (session tracker): {pre_hand_override}")

    # Derive sequence
    pre, preflop_commit_per_alive = (
        _derive_pre_hand_and_preflop_commit_simple_model(
            frame192, sb_seat=frame192.hero_seat,
            bb_seat=(frame192.hero_seat + 1) % 6,
            pre_hand_override=pre_hand_override))
    print(f"  preflop_commit_per_alive: {preflop_commit_per_alive}")
    print()

    action_seq = derive_action_sequence(
        frame192, pre_hand_override=pre_hand_override)
    print(f"derived action sequence ({len(action_seq)} entries):")
    for i, (seat, chip) in enumerate(action_seq):
        print(f"  [{i:>2d}] seat={seat}  chip_int={chip}")
    print()

    # Find BB's turn-emission entry (the all-in)
    bb_seat = (frame192.dealer_seat + 2) % 6
    bb_turn_idx = None
    for i, (seat, chip) in enumerate(action_seq):
        if seat == bb_seat and chip > 100:  # heuristic: large chip_int = the all-in raise
            bb_turn_idx = i
    print(f"BB (seat {bb_seat}) all-in emission index: {bb_turn_idx}  "
          f"chip_int={action_seq[bb_turn_idx][1]}")
    print()

    # Build OpenSpiel state, walk to BB's pre-action moment
    blind_level = structure.level(2)
    game_str = structure.to_inner_game_string_for_state(
        blind_level=blind_level, stacks=list(pre),
        dealer_seat=frame192.dealer_seat)
    print(f"game_str: {game_str[:200]}...")
    game = pyspiel.load_game(game_str)
    state = game.new_initial_state()

    # Walk action_seq up to BB's turn emission (NOT applying it)
    action_idx = 0
    safety = 200
    while action_idx < bb_turn_idx and safety > 0:
        safety -= 1
        if state.is_chance_node():
            deal_one_card_6max(state, frame192.hero_seat,
                                 frame192.hero_cards, frame192.board)
            continue
        if state.is_terminal():
            print(f"  terminal before BB's turn emission!")
            return
        expected_seat, chip = action_seq[action_idx]
        cp = state.current_player()
        if cp != expected_seat:
            print(f"  step {action_idx}: expected seat {expected_seat}, "
                  f"got cp={cp}")
            break
        legal = state.legal_actions()
        if chip not in legal:
            print(f"  step {action_idx} would trigger fallback: "
                  f"chip={chip} not in legal={legal[:10]}...")
            # Apply fallback (same as replay.py)
            chip = int(pre[expected_seat])
        state.apply_action(chip)
        action_idx += 1

    # Now drain any chance nodes to reach BB's turn decision
    while state.is_chance_node():
        deal_one_card_6max(state, frame192.hero_seat,
                             frame192.hero_cards, frame192.board)

    cp = state.current_player()
    parsed = parse_state_6max(state, observer=frame192.hero_seat)
    print(f"\nState at BB's turn-decision moment:")
    print(f"  current_player: {cp}  (BB seat = {bb_seat})")
    print(f"  parsed.contribution: {list(parsed['contribution'])}")
    print(f"  parsed.money:        {list(parsed['money'])}")
    print(f"  street_idx:          {parsed['street_idx']}")

    legal_actions = state.legal_actions()
    print(f"  legal_actions count: {len(legal_actions)}")
    if len(legal_actions) < 30:
        print(f"  legal_actions:       {legal_actions}")
    else:
        print(f"  legal_actions[:30]:  {legal_actions[:30]}")
        print(f"  legal_actions[-10:]: {legal_actions[-10:]}")
    print()
    print(f"  1515 in legal? {1515 in legal_actions}")
    print(f"  1525 in legal? {1525 in legal_actions}")
    derived_chip = action_seq[bb_turn_idx][1]
    print(f"  derived chip_int = {derived_chip}; in legal? "
          f"{derived_chip in legal_actions}")
    print(f"  pre_hand[BB] = {pre[bb_seat]}; in legal? "
          f"{pre[bb_seat] in legal_actions}")
    # Also test surrounding values to see what's accepted
    for offset in (-15, -10, -5, 0, 5, 10, 15):
        v = derived_chip + offset
        print(f"    chip_int={v:>5d}  in legal? "
              f"{v in legal_actions}")


if __name__ == "__main__":
    main()
