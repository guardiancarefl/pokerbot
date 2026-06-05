"""Step 3 diagnosis-confirmation query — preflop range for the throwaway
real-ante blueprint.

Wraps query_model_preflop_range's logic but substitutes a real-ante
TournamentStructure wrapper so the queried game state uses the new
ante=N N N N N N convention (NOT the inflated_BB hack the original
query script uses by default).

Reports BTN + CO open rates against pre-committed thresholds:
  BTN: target >= 30% (was 2.4% on pre-patch model)
  CO:  target >= 20% (was 0% on pre-patch model)

Movement of either toward the target = diagnosis confirmed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlhe.game_strings import TournamentStructure
# Reuse the throwaway training's real-ante emitter to keep them in sync.
from scripts.throwaway_train_real_ante import (
    to_inner_game_string_for_state_real_ante,
)


def to_inner_game_string_real_ante(self, level: int = 1) -> str:
    """Real-ante version of to_inner_game_string(level=1).

    Matches the existing method's hardcoded convention (SB=idx 0, BB=idx 1,
    BTN=idx 5, full 6-player ring with 1500 stacks), but uses native ante
    parameter instead of folding antes into BB.
    """
    bl = self.level(level)
    n = self.num_players
    sb = bl.small_blind
    bb = bl.big_blind          # NOT inflated
    ante = bl.ante

    blind_parts = [str(sb), str(bb)]
    blind_parts.extend(["0"] * (n - 2))
    blind_str = " ".join(blind_parts)

    ante_parts = [str(ante)] * n
    ante_str = " ".join(ante_parts)

    if n == 2:
        first_player = "2 1 1 1"
    else:
        first_player = "3 1 1 1"

    stack_str = " ".join([str(self.starting_chips)] * n)

    return (
        f"universal_poker(betting=nolimit,"
        f"numPlayers={n},"
        f"numRounds=4,"
        f"blind={blind_str},"
        f"ante={ante_str},"
        f"firstPlayer={first_player},"
        f"numSuits=4,"
        f"numRanks=13,"
        f"numHoleCards=2,"
        f"numBoardCards=0 3 1 1,"
        f"stack={stack_str},"
        f"bettingAbstraction=fullgame)"
    )


class _RealAnteStructure:
    """Duck-typed TournamentStructure with both game-string methods
    overridden to emit real-ante variants."""
    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
    def __getattr__(self, name):
        return getattr(self._inner, name)
    def to_inner_game_string(self, level=1):
        return to_inner_game_string_real_ante(self._inner, level)
    def to_inner_game_string_for_state(self, blind_level, stacks, dealer_seat):
        return to_inner_game_string_for_state_real_ante(
            self._inner, blind_level, stacks, dealer_seat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="path to the throwaway-trained checkpoint .pt")
    ap.add_argument("--abstraction",
                    default="runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    ap.add_argument("--structure",
                    default="configs/ignition_double_up_6max_turbo.yaml")
    args = ap.parse_args()

    # Monkey-patch TournamentStructure.from_yaml so query_model_preflop_range
    # (which calls it internally) gets our wrapped instance.
    from src.nlhe import game_strings as gs_module
    original_from_yaml = TournamentStructure.from_yaml

    def patched_from_yaml(path):
        return _RealAnteStructure(original_from_yaml(path))
    # Bind on the class — affects all calls in this process for duration.
    gs_module.TournamentStructure.from_yaml = staticmethod(patched_from_yaml)

    # Import the query script's main with our patched from_yaml in place.
    import importlib
    import scripts.query_model_preflop_range as q
    importlib.reload(q)

    # Run the query with the throwaway checkpoint, both positions.
    print("\n" + "="*70)
    print("STEP 3 — diagnosis confirmation query")
    print("="*70)
    for pos in ("BTN", "CO"):
        print(f"\n>>> Position: {pos}\n")
        sys.argv = [
            "throwaway_query_real_ante.py",
            "--checkpoint", args.checkpoint,
            "--abstraction", args.abstraction,
            "--structure", args.structure,
            "--hero-pos", pos,
        ]
        try:
            q.main()
        except SystemExit:
            pass


if __name__ == "__main__":
    main()
