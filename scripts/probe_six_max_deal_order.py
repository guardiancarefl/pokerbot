"""6-max universal_poker hole-card deal-order empirical probe (Phase 2 prereq).

Determine which OpenSpiel chance-node slot corresponds to which seat's
first/second hole card. Two candidate orders:

  Order A (round-robin):  p0c1, p1c1, p2c1, p3c1, p4c1, p5c1,
                          p0c2, p1c2, p2c2, p3c2, p4c2, p5c2
  Order B (sequential):   p0c1, p0c2, p1c1, p1c2, p2c1, p2c2,
                          p3c1, p3c2, p4c1, p4c2, p5c1, p5c2

The HUNL PolicyAdapter docstring (policy_adapter.py:464) says HUNL uses
Order B ("p0's first card, p0's second card, p1's first card, p1's second
card"). Whether 6-max also uses B is empirical — the universal_poker engine
isn't documented to extrapolate.

Method: load a 6-max game, scan each chance node ONE AT A TIME, apply ANY
legal deal, then diff the per-seat [Private:] field across seats. The seat
whose private cards GREW is the one that received the card.

Output: the deal order as a list of (seat_idx, card_position) tuples for
verification + a printable summary. We then assert the discovered order
matches one of A/B in a unit test so this never silently regresses.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pyspiel

sys.path.insert(0, "/home/quant/pokerbot")

from src.nlhe.game_strings import TournamentStructure  # noqa: E402

_PRIVATE_RE = re.compile(r"\[Private:\s+([^\]]*)\]")


def _private_cards_for(state, seat: int) -> str:
    """Read OpenSpiel's [Private: XXXX] field for the given seat."""
    info = state.information_state_string(seat)
    m = _PRIVATE_RE.search(info)
    return m.group(1) if m else ""


def probe_deal_order() -> list[tuple[int, int]]:
    """Walk chance nodes one at a time at the start of a 6-max hand.
    Returns the ordered list of (seat_idx, card_position) tuples for each
    hole-card deal (12 entries: 6 seats × 2 cards). Stops when state is no
    longer a chance node (= first decision).
    """
    structure = TournamentStructure.from_yaml(
        "configs/ignition_double_up_6max_turbo.yaml")
    bl = structure.level(1)
    stacks = [1500] * 6
    gs = structure.to_inner_game_string_for_state(
        blind_level=bl, stacks=stacks, dealer_seat=0)
    game = pyspiel.load_game(gs)
    state = game.new_initial_state()
    assert state.is_chance_node(), (
        f"expected chance node at init; got is_chance={state.is_chance_node()}, "
        f"is_terminal={state.is_terminal()}, current_player={state.current_player()}"
    )
    deal_order: list[tuple[int, int]] = []
    step = 0
    while state.is_chance_node() and step < 30:
        before = [_private_cards_for(state, p) for p in range(6)]
        legal = state.legal_actions()
        if not legal:
            break
        # Apply ANY legal deal — we just want to advance to see who got it.
        action = legal[0]
        state.apply_action(int(action))
        after = [_private_cards_for(state, p) for p in range(6)]
        # Find the seat whose private string grew.
        changed_seat = None
        new_position = None
        for p in range(6):
            if len(after[p]) > len(before[p]):
                changed_seat = p
                new_position = (len(after[p]) // 2) - 1
                break
        if changed_seat is None:
            # This was a board card (or another deal type). Stop hole-card scan.
            print(f"  step {step}: no seat private grew — likely a board card "
                  f"or other chance; stopping hole-card probe")
            break
        deal_order.append((changed_seat, new_position))
        step += 1
        if len(deal_order) == 12:
            break
    return deal_order


def classify_order(order: list[tuple[int, int]]) -> str:
    """Match against the two candidate orders. Returns 'A' / 'B' / 'OTHER'."""
    order_A = [(s, c) for c in range(2) for s in range(6)]      # round-robin
    order_B = [(s, c) for s in range(6) for c in range(2)]      # sequential
    if order == order_A:
        return "A (round-robin: p0c1, p1c1, ..., p5c1, p0c2, p1c2, ..., p5c2)"
    if order == order_B:
        return "B (sequential: p0c1, p0c2, p1c1, p1c2, ..., p5c2)"
    return f"OTHER (neither A nor B): {order}"


def main():
    print("=== 6-max universal_poker hole-card deal-order probe ===\n")
    order = probe_deal_order()
    print(f"Observed deal sequence ({len(order)} entries):")
    for i, (seat, pos) in enumerate(order):
        print(f"  slot {i:>2d}: seat {seat}  card {pos+1}/2")
    print()
    classification = classify_order(order)
    print(f"Classification: {classification}")
    print()
    if classification.startswith("OTHER"):
        print("UNEXPECTED order — Phase 2 build BLOCKED until this is understood.")
        return 1
    print(f"Order locked in — codify in src/nlhe/integration/replay.py + unit test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
