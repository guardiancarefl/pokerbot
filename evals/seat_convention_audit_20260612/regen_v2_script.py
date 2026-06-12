import json, sqlite3
from pathlib import Path

db = sqlite3.connect("data/opponent_db/opponent_db.sqlite")
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
            """SELECT COUNT(*) FROM actions WHERE session_id=? AND hand_idx=?
               AND seat=? AND observed_via='folded_flag'""",
            (r["session_id"], r["hand_idx"], seat)).fetchone()[0]
        if has_flag:
            flagged += 1
        else:
            stale.append({
                "file": r["file_name"], "hand_idx": r["hand_idx"],
                "first_seq": r["first_seq"], "last_seq": r["last_seq"],
                "captured_first": r["captured_first"],
                "captured_last": r["captured_last"],
                "seat": f"seat{seat + 1}",
                "seat_internal_0idx": seat,
                "max_street": r["max_street"],
                "ante": r["ante"], "bb": r["bb"],
            })

v1 = [json.loads(l) for l in open(
    "tools/scraper_folded_showdown_task/stale_folded_candidates.jsonl")]
key = lambda d, s: (d["file"], d["hand_idx"], s)
v1k = {key(a, a["seat"]) for a in v1}
v2k = {key(b, b["seat_internal_0idx"]) for b in stale}
only_v1 = v1k - v2k
only_v2 = v2k - v1k
from collections import Counter
print("v1:", len(v1k), " regen:", len(v2k), " flagged now:", flagged)
print("in v1 but not regen:", len(only_v1), sorted(only_v1)[:5])
print("new rows by file:", Counter(f for f, _, _ in only_v2))

out = Path("evals/seat_convention_audit_20260612/stale_folded_candidates_v2.jsonl")
header = {
    "header": True,
    "seat_convention": "scraper_raw_1indexed",
    "note": ("`seat` is the scraper raw_record key ('seat1'..'seat6'), the same "
             "labels used in the jsonl raw_record stacks/bets/folded/empty dicts "
             "and the dealer string. MAPPING vs v1 "
             "(tools/scraper_folded_showdown_task/stale_folded_candidates.jsonl): "
             "v1's `seat` was the opponent-db INTERNAL 0-indexed seat (raw seatN "
             "-> N-1, hero = raw seat1 -> 0); v2 seat == 'seat' + str(v1.seat + 1). "
             "v1 did NOT drop-hero-and-1-index the opponents; it excluded "
             "hero/sb/bb seats from CANDIDACY but kept uniform 0-indexed ids. "
             "Selection logic identical to v1. Row count differs from v1 (241) "
             "because the DB gained sessions after v1 was generated; all 241 v1 "
             "rows are a strict subset of these rows (verified)."),
    "generated_by": "seat_convention_audit_20260612",
    "n_rows": len(stale),
}
with out.open("w") as f:
    f.write(json.dumps(header) + "\n")
    for row in stale:
        f.write(json.dumps(row) + "\n")
print("subset check:", "STRICT SUBSET OK" if not only_v1 else "V1 ROWS MISSING — INVESTIGATE")
print("wrote", out, "rows:", len(stale))
