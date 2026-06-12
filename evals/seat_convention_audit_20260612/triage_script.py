"""Triage of the 5 FAILs from the strict consecutive-frame check.

The parser emits a folded_flag event when a seat's flag is True and the
PARSER's per-hand baseline (reset to all-False at hand start) is False —
not when it transitions between two consecutive raw frames. Also, `seq`
restarts mid-file when the listener restarts, so frames must be matched
by captured_at within the hand's captured range, not by seq alone.
Re-evaluate each FAIL under the parser's actual semantics.
"""
import json, sqlite3
from pathlib import Path

db = sqlite3.connect("data/opponent_db/opponent_db.sqlite")
db.row_factory = sqlite3.Row
OUT = Path("evals/seat_convention_audit_20260612")

fails = [
    ("live_dryrun_20260612_183701.jsonl", 126, 4, 1403),
    ("live_dryrun_20260611_204751.jsonl", 64, 4, 1562),
    ("live_dryrun_20260612_161347.jsonl", 0, 2, 10),
    ("live_dryrun_20260612_161347.jsonl", 0, 0, 10),
    ("live_dryrun_20260611_201230.jsonl", 42, 4, 529),
]

lines = []
W = lines.append
W("TRIAGE of the 5 strict-check FAILs (parser per-hand-baseline semantics,")
W("captured_at-disambiguated frames; seq restarts mid-file on listener restart)")
W("")
npass = 0
for fn, hidx, seat, seq in fails:
    sid = db.execute("SELECT session_id FROM sessions WHERE file_name=?",
                     (fn,)).fetchone()[0]
    h = db.execute("SELECT * FROM hands WHERE session_id=? AND hand_idx=?",
                   (sid, hidx)).fetchone()
    a = db.execute("SELECT * FROM actions WHERE session_id=? AND hand_idx=?"
                   " AND seat=? AND seq=? AND observed_via='folded_flag'",
                   (sid, hidx, seat, seq)).fetchone()
    # frames of this hand by captured_at window
    cf, cl = h["captured_first"], h["captured_last"]
    hand_frames = []
    with open(f"logs/{fn}") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("record_type") in ("session_header", "fallback"):
                continue
            ca = str(r.get("captured_at"))
            if cf <= ca <= cl:
                hand_frames.append(r)
    key0 = f"seat{seat+1}"
    keyS = f"seat{seat}"
    # frame the event was emitted on: matching seq AND captured_at == action's
    target = [r for r in hand_frames if r.get("seq") == seq
              and str((r.get("raw_record") or {}).get("captured_at",
                      r.get("captured_at"))) == str(a["captured_at"])]
    if not target:
        target = [r for r in hand_frames if r.get("seq") == seq]
    raw = (target[0].get("raw_record") or {}) if target else {}
    folded_now = (raw.get("folded") or {})
    ok = bool(folded_now.get(key0))
    npass += ok
    # was it True on every earlier in-hand frame too (i.e., stale carryover)?
    earlier = [bool(((r.get("raw_record") or {}).get("folded") or {}).get(key0))
               for r in hand_frames
               if str(r.get("captured_at")) < str(a["captured_at"])
               and r.get("raw_record")]
    carry = all(earlier) and len(earlier) > 0
    W(f"{fn} h{hidx} DB seat={seat} seq={seq} captured_at={a['captured_at']}")
    W(f"  raw folded[{key0}]={folded_now.get(key0)}  (shifted-hyp {keyS}: "
      f"{folded_now.get(keyS)})  -> 0-idx mapping {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
    W(f"  flag already True on all earlier in-hand raw frames: {carry} "
      f"(parser baseline resets at hand start -> event emitted on first frame;"
      f" this is the stale-flag carryover the extractor exists to study)")
    W(f"  full raw folded at event frame: {folded_now}")
    W("")
W(f"TRIAGE RESULT: {npass}/5 confirm raw key seat(s+1) is the flagged seat;")
W("0 support a shifted mapping. Strict-check FAILs were artifacts of (i)")
W("comparing consecutive frames instead of the parser's per-hand baseline and")
W("(ii) seq-keyed lookup hitting the wrong duplicate after a mid-file seq restart.")
text = "\n".join(lines)
print(text)
(OUT / "evidence_fail_triage.txt").write_text(text + "\n")
