"""Offline preflop opening-range query for the trained blueprint.

Constructs a clean OpenSpiel state at level 1 (15/25 + 5 ante), 6-max,
1500 starting stacks, hero on BTN (or CO) with action folded around to
hero. For each of the 169 starting hands, force-deals hero those cards,
queries the policy net in argmax mode, and reports the action.

Output: 13×13 grid showing OPEN (raise / call / fold) for each hand,
plus the chip amount when the model opens. Pure model behavior — no
scraper, no bridge artifacts, no real-game frame variability.

Answers: "is the trained blueprint itself ultra-tight?"

Usage:
    python scripts/query_model_preflop_range.py
    python scripts/query_model_preflop_range.py --hero-pos CO
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.replay import deal_one_card_6max

import pyspiel


RANKS = "AKQJT98765432"  # ace-high → 2 (standard chart order)


def all_169_hands() -> list[tuple[str, tuple[str, str]]]:
    """Yield (label, (card1, card2)) for each of the 169 starting hands.

    Canonical representatives:
      Pair XX  → 'XX'   (e.g., 'AA') with cards 'X♠ X♥'
      Suited   → 'XYs'  with both spades  ('X♠ Y♠')
      Offsuit  → 'XYo'  with mixed suits  ('X♠ Y♥')
    """
    out = []
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i == j:
                # Pair
                out.append((f"{r1}{r1}", (f"{r1}s", f"{r1}h")))
            elif i < j:
                # r1 is higher → suited (upper-right triangle convention)
                out.append((f"{r1}{r2}s", (f"{r1}s", f"{r2}s")))
            else:
                # r2 is higher → offsuit (lower-left triangle convention)
                # Use r2-first to match the higher-rank-first label
                out.append((f"{r2}{r1}o", (f"{r2}s", f"{r1}h")))
    return out


def build_state_with_hero_at(structure, hero_pos: str, hero_cards: tuple[str, str]):
    """Build a level-1 6-max state at hero's preflop decision with action
    folded around to hero.

    hero_pos: "BTN" or "CO".
      - BTN: 3 folds (UTG, MP, CO) before hero at idx 0.
      - CO:  2 folds (UTG, MP) before hero at idx 5.

    Layout (dealer=idx0): SB=1, BB=2, UTG=3, MP=4, CO=5, BTN=0.
    Preflop action order: 3 → 4 → 5 → 0 → 1 → 2.
    """
    game_str = structure.to_inner_game_string(level=1)
    game = pyspiel.load_game(game_str)
    state = game.new_initial_state()

    # `to_inner_game_string(level=1)` hardcodes blinds at "15 25 0 0 0 0"
    # (SB=idx 0, BB=idx 1) and firstPlayer=3 (UTG=idx 2). So the 6-max
    # layout is: SB=0, BB=1, UTG=2, MP=3, CO=4, BTN=5. Preflop action
    # order: idx 2 → 3 → 4 → 5 → 0 → 1.
    if hero_pos == "BTN":
        hero_seat = 5
        folds_before = [2, 3, 4]   # UTG, MP, CO fold to hero on BTN
    elif hero_pos == "CO":
        hero_seat = 4
        folds_before = [2, 3]      # UTG, MP fold to hero on CO
    else:
        raise ValueError(f"hero_pos must be BTN or CO, got {hero_pos!r}")

    # Empty target board (we're at preflop)
    target_board: tuple[str, ...] = ()

    # Walk chance + action nodes
    max_steps = 80
    fold_idx = 0
    for _ in range(max_steps):
        if state.is_terminal():
            raise RuntimeError(
                f"state went terminal during setup; fold_idx={fold_idx}")
        if state.is_chance_node():
            deal_one_card_6max(state, hero_seat=hero_seat,
                               hero_cards=hero_cards,
                               target_board=target_board)
            continue
        # Decision node — verify current_player matches the expected actor
        cp = state.current_player()
        if cp == hero_seat:
            # We've reached hero's decision. Done setting up.
            return state, hero_seat, game_str
        # Otherwise, expect the next expected folder
        if fold_idx >= len(folds_before):
            raise RuntimeError(
                f"reached unexpected decision: current_player={cp}, "
                f"hero={hero_seat}, fold_idx exhausted at {fold_idx}")
        expected = folds_before[fold_idx]
        if cp != expected:
            raise RuntimeError(
                f"expected current_player={expected} (fold #{fold_idx}), "
                f"got {cp}")
        # Apply fold (action 0)
        legal = state.legal_actions()
        if 0 not in legal:
            raise RuntimeError(
                f"fold (0) not in legal_actions={legal[:10]} for "
                f"current_player={cp}")
        state.apply_action(0)
        fold_idx += 1
    raise RuntimeError("max_steps exhausted during setup")


def query_action(solver, state, hero_seat: int, rng) -> tuple[str, int | None, dict]:
    """Run the blueprint policy net and report the chosen action.

    Returns (kind, chip_amount, policy_dist).
      kind: "fold" | "call" | "check" | "raise"
      chip_amount: int if raise, else None
      policy_dist: dict of DiscreteAction-name -> probability
    """
    from src.nlhe.infoset6 import parse_state_6max
    from src.nlhe.actions import (
        DiscreteAction, BET_ACTIONS_IN_ORDER, BET_FRACTIONS,
        discretize_legal_actions,
    )
    from src.nlhe.cfr6 import _build_view_6max
    import numpy as np

    parsed = parse_state_6max(state, observer=hero_seat)
    # With to_inner_game_string(level=1)'s convention (SB=0, BB=1), BTN
    # is at idx 5. parse_state_6max needs dealer_seat set explicitly.
    parsed["dealer_seat"] = 5
    legal_chip = list(state.legal_actions())
    view = _build_view_6max(state, parsed)
    discrete_to_chip = discretize_legal_actions(legal_chip, view)

    legal_mask = np.zeros(len(DiscreteAction), dtype=np.float32)
    for da in discrete_to_chip:
        legal_mask[int(da)] = 1.0

    encoded = solver.encoder.encode_from_parsed(parsed, rng=rng)
    features = np.asarray(encoded, dtype=np.float32)
    policy = solver.policy_nets.inference_policy(
        hero_seat, features, legal_mask)

    chosen_idx = int(np.argmax(policy))
    da = DiscreteAction(chosen_idx)
    chip = discrete_to_chip.get(da)

    if da == DiscreteAction.FOLD:
        kind = "fold"
    elif da == DiscreteAction.CALL:
        # In a fold-around-to-hero state, CALL = limp (call BB)
        kind = "call"
    else:
        kind = f"raise:{da.label}"

    dist = {DiscreteAction(i).label: float(policy[i])
             for i in range(len(DiscreteAction))
             if legal_mask[i] > 0}
    return kind, (int(chip) if chip is not None else None), dist


def render_range_grid(results: dict[str, tuple[str, int | None]]) -> str:
    """Render a 13×13 preflop range grid.

    Convention: rows = higher rank, cols = lower rank. Upper-right
    triangle = suited (XYs). Lower-left triangle = offsuit (XYo). Diagonal
    = pairs.

    Symbol legend per cell:
      F  = fold
      L  = limp (call BB)
      R  = raise/open (any size — chip amount in parenthetical legend)
    """
    lines = []
    header = "    " + " ".join(f"{r:>4}" for r in RANKS)
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))
    for i, r1 in enumerate(RANKS):
        cells = []
        for j, r2 in enumerate(RANKS):
            if i == j:
                label = f"{r1}{r1}"
            elif i < j:
                label = f"{r1}{r2}s"
            else:
                label = f"{r2}{r1}o"
            kind, chip = results.get(label, ("?", None))
            if kind == "fold":
                cell = "  F "
            elif kind == "call":
                cell = "  L "
            elif kind and kind.startswith("raise"):
                cell = "  R "
            else:
                cell = "  ? "
            cells.append(cell)
        lines.append(f"{r1:>3} " + " ".join(cells))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="runs/k200_blueprint_ckpt_iter_2000.pt")
    ap.add_argument("--abstraction",
                    default="runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    ap.add_argument("--structure",
                    default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--hero-pos", default="BTN", choices=["BTN", "CO"])
    args = ap.parse_args()

    structure = TournamentStructure.from_yaml(args.structure)
    print(f"=== Offline preflop range query ===")
    print(f"  structure:    {args.structure} "
          f"(starting={structure.starting_chips})")
    print(f"  checkpoint:   {args.checkpoint}")
    print(f"  level:        1 (sb={structure.level(1).small_blind} "
          f"bb={structure.level(1).big_blind} "
          f"ante={structure.level(1).ante})")
    print(f"  hero_pos:     {args.hero_pos}")
    print(f"  action:       folded around to hero (clean open spot)\n")

    print("Loading solver...")
    with open(args.abstraction, "rb") as f:
        abstraction = pickle.load(f)
    from scripts.eval_6max_self_play import _load_solver
    solver = _load_solver(args.checkpoint, abstraction, structure)

    rng = random.Random(0)
    results: dict[str, tuple[str, int | None]] = {}
    raise_sizes: list[tuple[str, int]] = []
    distribution_examples: list[tuple[str, dict]] = []

    hands = all_169_hands()
    print(f"Querying model on {len(hands)} starting hands...\n")
    for label, cards in hands:
        try:
            state, hero_seat, _ = build_state_with_hero_at(
                structure, args.hero_pos, cards)
        except Exception as e:
            print(f"  setup failed for {label}: {e}")
            results[label] = ("error", None)
            continue
        try:
            kind, chip, dist = query_action(solver, state, hero_seat, rng)
        except Exception as e:
            print(f"  query failed for {label}: {e}")
            results[label] = ("error", None)
            continue
        results[label] = (kind, chip)
        if kind and kind.startswith("raise"):
            raise_sizes.append((label, chip))
        # Collect a few interesting hands' full distributions
        if label in ("AA", "AKs", "AKo", "KJs", "KJo", "ATo", "T9s", "22", "72o"):
            distribution_examples.append((label, dist))

    # --- Reports ---
    print(f"=== Preflop opening range, hero on {args.hero_pos}, level 1 ===")
    print("    Legend: F=fold  L=limp(call BB)  R=raise/open\n")
    print(render_range_grid(results))

    # Summary stats
    fold_count = sum(1 for k, _ in results.values() if k == "fold")
    call_count = sum(1 for k, _ in results.values() if k == "call")
    raise_count = sum(1 for k, _ in results.values() if k and k.startswith("raise"))
    err_count = sum(1 for k, _ in results.values() if k == "error")
    total = len(results)
    print(f"\n=== Summary ===")
    print(f"  total hands:  {total}")
    print(f"  fold:         {fold_count}  ({100*fold_count/total:.1f}%)")
    print(f"  limp (call):  {call_count}  ({100*call_count/total:.1f}%)")
    print(f"  open (raise): {raise_count}  ({100*raise_count/total:.1f}%)")
    if err_count:
        print(f"  errors:       {err_count}")

    if raise_sizes:
        print(f"\n=== Open-raise hands & sizing ===")
        for label, chip in raise_sizes:
            print(f"  {label}: raise to {chip} chips "
                  f"({chip / structure.level(1).big_blind:.1f}xBB, "
                  f"{chip / 1500:.2f}x stack)")

    if distribution_examples:
        print(f"\n=== Full policy distribution on interesting hands ===")
        for label, dist in distribution_examples:
            print(f"  {label}:")
            for act, prob in sorted(dist.items(), key=lambda kv: -kv[1]):
                bar = "█" * int(prob * 40)
                print(f"    {act:<10s}  {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()
