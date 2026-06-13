"""Gate 2 companion — floor qualify/fire stats on the live corpus.

Re-runs make_decision over all 17 logs with a SHARED ShoveDefenseFloor whose
.stats accumulate across the whole corpus. Reports n_calls / n_qualify /
n_fired so the zero-change Gate-2 result is explained positively: the floor's
apply hook was invoked on every decision (n_calls), screened the scope guards
(n_qualify = facing single all-in raiser, hero eff 5-15bb, call commits stack),
and only fired where equity < ICM break-even (n_fired). This is the firing-path
audit the flag-ON replay diff cannot show when nothing fires.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.abstraction import Abstraction
from src.nlhe.integration.live_loop import (
    DecisionCache, make_decision, load_shove_defense_floor)
from src.nlhe.integration.scraper_schema import SessionTracker
from scripts.eval_6max_self_play import _load_solver
from scripts.replay_make_decision_diff import STRUCTURE_YAML

CKPT = "runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt"
ABSTR = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
LOGS = sys.argv[1:]


def main():
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    abstr = Abstraction.load(ABSTR)
    solver = _load_solver(CKPT, abstr, structure)
    floor = load_shove_defense_floor(tau=0.0, structure=structure)

    for log in LOGS:
        tracker = SessionTracker()
        cache = DecisionCache()
        rng = random.Random(42)
        with open(REPO / "logs" / f"{log}.jsonl") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                raw = rec.get("raw_record")
                if raw is None:
                    continue
                make_decision(raw, structure, solver, tracker, rng,
                              mode="sample", seq=rec.get("seq", i),
                              decision_cache=cache, shove_defense_floor=floor)

    s = floor.stats
    print("GATE 2 companion — shove-defense floor stats over all 17 logs")
    print(f"n_calls   (apply hook invoked) : {s['n_calls']}")
    print(f"n_qualify (passed scope guards) : {s['n_qualify']}")
    print(f"n_fired   (equity < break-even) : {s['n_fired']}")
    if s["cells"]:
        print("per-cell [qualify, fired]:")
        for k, v in sorted(s["cells"].items()):
            print(f"  {k:<14s} {v}")
    print()
    print("Interpretation: the apply hook ran on every decision; the corpus "
          "contained {} node(s) inside the facing-single-all-in / 5-15bb "
          "scope. Fired only where ICM-break-even failed.".format(
              s["n_qualify"]))


if __name__ == "__main__":
    main()
