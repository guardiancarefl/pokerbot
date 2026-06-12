"""Extract stale-`folded`-flag forensics candidates from the opponent DB.

Ground truth: a NON-BLIND dealt seat whose net for the hand is exactly
-ante (outcome_known via chip-conservation closure) folded preflop —
no other line loses exactly the ante. If the hand reached postflop
(max_street >= 1) the scraper had whole streets to set `folded`; a seat
with NO folded_flag transition in the hand is a STALE-FLAG instance.

Output jsonl, one row per (hand, seat): session file, hand_idx,
first/last seq, captured_at range (the Windows PNG-archive key), seat,
max_street — plus a summary line on stdout with the staleness rate
(stale / all chip-proven preflop folds in postflop-reaching hands).

Usage:
  python -m scripts.extract_stale_folded_candidates \
      --db data/opponent_db/opponent_db.sqlite \
      --out tools/scraper_folded_showdown_task/stale_folded_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/opponent_db/opponent_db.sqlite")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT h.session_id, s.file_name, h.hand_idx, h.first_seq,
               h.last_seq, h.captured_first, h.captured_last, h.ante,
               h.bb, h.hero_seat, h.sb_seat, h.bb_seat,
               h.net_chips_json, h.max_street, h.dealt_seats_json
        FROM hands h JOIN sessions s USING(session_id)
        WHERE h.outcome_known = 1 AND h.ante IS NOT NULL
          AND h.ante > 0 AND h.max_street >= 1
    """).fetchall()

    stale, flagged = [], 0
    for r in rows:
        net = json.loads(r["net_chips_json"])
        for seat in json.loads(r["dealt_seats_json"] or "[]"):
            if seat in (r["hero_seat"], r["sb_seat"], r["bb_seat"]):
                continue
            n = net.get(str(seat)) if isinstance(net, dict) else (
                net[seat] if seat < len(net) else None)
            if n != -r["ante"]:
                continue
            has_flag = db.execute(
                """SELECT COUNT(*) FROM actions
                   WHERE session_id=? AND hand_idx=? AND seat=?
                     AND observed_via='folded_flag'""",
                (r["session_id"], r["hand_idx"], seat)).fetchone()[0]
            if has_flag:
                flagged += 1
            else:
                stale.append({
                    "file": r["file_name"], "hand_idx": r["hand_idx"],
                    "first_seq": r["first_seq"], "last_seq": r["last_seq"],
                    "captured_first": r["captured_first"],
                    "captured_last": r["captured_last"],
                    "seat": seat, "max_street": r["max_street"],
                    "ante": r["ante"], "bb": r["bb"],
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in stale:
            f.write(json.dumps(row) + "\n")

    total = len(stale) + flagged
    rate = (len(stale) / total) if total else float("nan")
    print(f"chip-proven preflop folds (non-blind, postflop-reaching, "
          f"outcome-known hands): {total}")
    print(f"  folded_flag transition seen : {flagged}")
    print(f"  STALE (no transition)       : {len(stale)}  "
          f"({100 * rate:.1f}% stale rate)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
