"""Piece 7 — first real resolver call on a mid-hand reconstructed state.

Picks the first PASSING postflop mid-hand frame from data/live9.jsonl,
reconstructs the OpenSpiel state via replay_to_decision, verifies the
bridge-aware invariant, then runs the trained blueprint policy net on
the resulting state and reverse-translates the chip_int through the
bridge to produce a real-table client action.

LOG-ONLY. Does NOT click. Output is for the user to inspect — specifically
whether the bet-sizing drift documented in DECISIONS.md "Phase 2 bridge"
is showing up in the open and is qualitatively reasonable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.invariant import check_mid_hand_invariant
from src.nlhe.integration.replay import replay_to_decision
from src.nlhe.integration.scraper_schema import parse_frame
from src.nlhe.integration.translate import openspiel_to_real_action


def find_first_passing_mid_hand(jsonl_path: str, structure):
    """Return (line_no, frame, pack, inv) for the first passing postflop
    hero-to-act frame, or raise if none found."""
    with open(jsonl_path) as f:
        for line_no, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                frame = parse_frame(record)
            except Exception:
                continue
            if frame is None:
                continue
            if not frame.controls_present:
                continue
            if not frame.hero_cards:
                continue
            if sum(frame.alive) < 4:
                continue
            try:
                pack = replay_to_decision(frame, structure)
            except Exception:
                continue
            inv = check_mid_hand_invariant(frame, pack)
            if not inv.ok:
                continue
            if pack.street_idx == 0:
                continue  # want postflop (more interesting first call)
            return line_no, frame, pack, inv
    raise RuntimeError("no passing postflop mid-hand frame found")


def main():
    structure_path = "configs/ignition_double_up_6max_turbo.yaml"
    jsonl_path = "data/live9.jsonl"
    abstraction_path = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
    checkpoint_path = "runs/k200_blueprint_ckpt_iter_2000.pt"

    print(f"=== Piece 7: first real resolver call (LOG-ONLY, NO CLICK) ===\n")
    print(f"checkpoint:   {checkpoint_path}")
    print(f"abstraction:  {abstraction_path}")
    print(f"structure:    {structure_path}")
    print(f"corpus:       {jsonl_path}\n")

    structure = TournamentStructure.from_yaml(structure_path)

    # --- Find a frame ---
    print("Finding first passing postflop hero-to-act frame...")
    line_no, frame, pack, inv = find_first_passing_mid_hand(
        jsonl_path, structure)
    print(f"\n--- FRAME (line {line_no}, captured_at={frame.captured_at}) ---")
    print(f"  blinds       sb={frame.blinds.sb} bb={frame.blinds.bb} "
          f"ante={frame.blinds.ante}")
    print(f"  n_alive      {sum(frame.alive)}   dealer=seat{frame.dealer_seat+1}"
          f"   hero=seat{frame.hero_seat+1}")
    print(f"  hero_cards   {frame.hero_cards}")
    print(f"  board        {frame.board}   (street_idx={pack.street_idx})")
    print(f"  pot (scrapr) {frame.pot_total}")
    print(f"  per-seat stk {frame.stack}")
    print(f"  per-seat bet {frame.bet}")
    print(f"  facing_bet   {frame.hero_facing_bet}")

    legal = pack.state.legal_actions()
    raise_options = [a for a in legal if a >= 2]
    opie_min_raise = min(raise_options) if raise_options else None
    opie_max_raise = max(raise_options) if raise_options else None
    print(f"  legal[:8]    {legal[:8]}...   n_legal={len(legal)}")
    print(f"  openspiel    min_raise={opie_min_raise}   "
          f"max_raise={opie_max_raise}")

    # --- Load solver + abstraction ---
    print("\nLoading solver + abstraction...")
    import pickle
    with open(abstraction_path, "rb") as f:
        abstraction = pickle.load(f)
    from scripts.eval_6max_self_play import (
        _load_solver, _sample_action_from_policy,
    )
    solver = _load_solver(checkpoint_path, abstraction, structure)
    print(f"  loaded solver: type={type(solver).__name__}")

    # --- Resolver call ---
    import random
    rng = random.Random(0)  # deterministic for the demo

    from src.nlhe.infoset6 import parse_state_6max
    parsed = parse_state_6max(pack.state, observer=frame.hero_seat)
    parsed["dealer_seat"] = frame.dealer_seat

    print(f"\nRunning blueprint policy net (mode=argmax, deterministic)...")
    chosen_chip_int = _sample_action_from_policy(
        solver, parsed, pack.state, rng, mode="argmax",
    )
    print(f"  raw openspiel chip_int = {chosen_chip_int}")

    # --- Reverse-translate ---
    # scraper_min/max_raise: derive from scraper-view conservatively.
    # Real-poker min-raise = 2 × bb (preflop) or 2 × max_current_street_bet
    # (postflop). On a postflop check-around at level 1, min_bet = bb = 25.
    scraper_min_raise = max(frame.blinds.bb,
                             2 * max((b for b in frame.bet), default=0))
    scraper_max_raise = frame.stack[frame.hero_seat] + frame.bet[frame.hero_seat]
    client_action = openspiel_to_real_action(
        chosen_chip_int,
        scraper_min_raise=scraper_min_raise,
        scraper_max_raise=scraper_max_raise,
        scraper_facing_bet=frame.hero_facing_bet,
    )

    # --- Log it ---
    print(f"\n=== CLIENT ACTION (LOG-ONLY — NO CLICK) ===")
    print(f"  kind              {client_action['kind']}")
    if client_action.get("chip_amount") is not None:
        chip_amt = client_action["chip_amount"]
        print(f"  chip_amount       {chip_amt} "
              f"(real-poker chips to enter into bet box)")
        # Useful framings:
        pot_frac = chip_amt / max(1, frame.pot_total)
        bb_mult = chip_amt / max(1, frame.blinds.bb)
        stk_frac = chip_amt / max(1, frame.stack[frame.hero_seat])
        print(f"                    = {pot_frac:.2f} × pot")
        print(f"                    = {bb_mult:.2f} × BB")
        print(f"                    = {stk_frac:.2f} × hero_stack")
    print(f"  raw_openspiel     {client_action['raw_openspiel_chip_int']}")
    print(f"  scraper_min_raise {scraper_min_raise}")
    print(f"  scraper_max_raise {scraper_max_raise}")

    # --- Watch-list reminder ---
    print(f"\n--- WATCH-LIST (from DECISIONS.md) ---")
    print("  Track preflop open-fold-frequency-when-called at levels 1-3.")
    print("  Above frame is postflop, so it's NOT in the watch-list bucket")
    print("  itself, but the same MECHANISM applies: model thinks in inflated")
    print("  space, client acts in real chips, bet sizing is the drift channel.")
    print()
    print("STILL NO CLICKING. Log-only.")


if __name__ == "__main__":
    main()
