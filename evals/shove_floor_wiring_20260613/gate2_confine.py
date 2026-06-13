"""Gate 2 — flag-ON confinement.

Diffs the flag-ON replay (--shove-defense-floor) against the flag-OFF post
replay for all 17 logs, keyed by captured_at (the diff() keying that survives
seq collisions). Every behavioral change is classified:

  - "shove_fold"  : preflop (street_idx==0), facing a bet (facing_bet truthy),
                    and the action changed to FOLD (chip_int 0 / client_action
                    kind 'fold') from a non-fold ON the OFF arm. This is the
                    floor's ONLY behavior (collapse CALL/ALLIN mass to FOLD on
                    a facing-all-in 5-15bb preflop node).
  - "class_other" : any behavioral change NOT matching the above profile.

Gate passes iff class_other == 0 (all changes are the in-scope shove-fold).
The 5-15bb depth gate is enforced inside the floor itself (the off-scope
guard), so a flag-ON change can only occur at a node the floor qualified;
the street/facing/→FOLD profile is the externally observable confinement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BEHAVIORAL_KEYS = (
    "status", "skip_reason", "hero_stack", "pot_total", "pot_corrected",
    "pre_hand_override_used", "street_idx", "facing_bet",
    "client_action", "chip_int", "decision_identity",
    "invariant_deltas", "click_plan", "recovered_fields", "anchor_refused",
)

OFF_DIR = Path(sys.argv[1])   # flag-OFF post replays (g1_post)
ON_DIR = Path(sys.argv[2])    # flag-ON replays (g2_on)
LOGS = sys.argv[3:]


def _load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        key = r.get("captured_at") or f"seq:{r['seq']}"
        out[key] = r
    return out


def _is_fold(row):
    if row.get("chip_int") == 0:
        return True
    ca = row.get("client_action")
    return isinstance(ca, dict) and ca.get("kind") == "fold"


def main():
    n_shove_fold = 0
    n_other = 0
    shove_examples = []
    other_examples = []
    total_changed = 0

    for log in LOGS:
        off = _load(OFF_DIR / f"{log}.jsonl")
        on = _load(ON_DIR / f"{log}.jsonl")
        assert set(off) == set(on), f"frame sets differ for {log}"
        for k in sorted(off):
            a, b = off[k], on[k]
            diffs = {kk for kk in BEHAVIORAL_KEYS if a.get(kk) != b.get(kk)}
            if not diffs:
                continue
            total_changed += 1
            # In-scope shove-fold profile: preflop, facing a bet on the OFF
            # arm, ON arm action is FOLD, OFF arm action was NOT fold.
            in_scope = (
                int(a.get("street_idx") or -1) == 0
                and bool(a.get("facing_bet"))
                and _is_fold(b)
                and not _is_fold(a)
            )
            if in_scope:
                n_shove_fold += 1
                if len(shove_examples) < 25:
                    shove_examples.append(
                        (log, k, a.get("chip_int"), b.get("chip_int"),
                         a.get("hero_stack")))
            else:
                n_other += 1
                if len(other_examples) < 25:
                    other_examples.append((log, k, sorted(diffs),
                                           a.get("street_idx"),
                                           a.get("facing_bet"),
                                           a.get("chip_int"),
                                           b.get("chip_int")))

    print("GATE 2 — flag-ON confinement (ON vs flag-OFF post, all 17 logs)")
    print(f"behavioral changes total : {total_changed}")
    print(f"class_shove_fold         : {n_shove_fold}  "
          f"(preflop, facing-bet, CALL/ALLIN -> FOLD)")
    print(f"class_other              : {n_other}")
    print()
    print("shove-fold changes (log, captured_at, off_chip->on_chip, hero_stack):")
    for e in shove_examples:
        print(f"  {e}")
    if n_other:
        print("\nclass_other changes (UNEXPECTED):")
        for e in other_examples:
            print(f"  {e}")
    print()
    if n_other == 0 and n_shove_fold > 0:
        print(f"GATE 2 PASS: all {n_shove_fold} behavioral changes are "
              f"in-scope shove-folds; class_other=0.")
        return 0
    if n_shove_fold == 0:
        print("GATE 2 WARN: floor produced ZERO changes across the corpus "
              "(no preflop facing-all-in 5-15bb node fired).")
        return 0 if n_other == 0 else 1
    print("GATE 2 FAIL: class_other != 0.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
