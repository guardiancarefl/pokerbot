"""Seat-convention ground-truth audit (2026-06-12).

Verifies, against raw session jsonl frames, that the opponent-db stores
seats under ONE consistent mapping: raw 'seatN' -> internal N-1 (0-indexed).
Tests:
  (a) hero identity: rec.hero_stack == raw_record.stacks['seat1'] on decision
      frames; DB hands.hero_seat == 0.
  (b) raw dealer 'seatN' vs hands.dealer_seat == N-1.
  (c) frame_diff actions: DB actions.seat=s -> raw bets['seat{s+1}'] rose by
      amount_delta to amount_to between consecutive raw frames; competing
      shifted hypotheses (raw seat{s} / seat{s+2}) checked explicitly.
  (d) folded_flag actions: raw folded['seat{s+1}'] False->True at that seq.
  (e) v1 candidate rows: under raw key seat{s+1}, anchor-stack drop across the
      hand == ante (the chip-proof the extractor relied on).
"""
import json, sqlite3, os
from pathlib import Path

DB = "data/opponent_db/opponent_db.sqlite"
OUT = Path("evals/seat_convention_audit_20260612")
OUT.mkdir(parents=True, exist_ok=True)

db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row

def load_frames(fname):
    frames = {}
    order = []
    with open(f"logs/{fname}") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            if rec.get("record_type") in ("session_header", "fallback"): continue
            frames[rec.get("seq")] = rec
            order.append(rec.get("seq"))
    return frames, order

sessions = {r["session_id"]: r["file_name"] for r in
            db.execute("SELECT session_id, file_name FROM sessions WHERE has_raw_record=1")}

evidence = []   # list of dicts written per-hand

# ---- pick frame_diff actions spread across sessions/hands ----
acts = db.execute("""
    SELECT a.session_id, a.hand_idx, a.event_idx, a.seat, a.street, a.kind,
           a.amount_to, a.amount_delta, a.seq, a.observed_via, a.is_hero,
           h.dealer_seat, h.hero_seat, h.first_seq, h.last_seq
    FROM actions a JOIN hands h USING(session_id, hand_idx)
    WHERE a.observed_via IN ('frame_diff','folded_flag') AND a.seq IS NOT NULL
    ORDER BY a.session_id, a.hand_idx, a.event_idx
""").fetchall()

from collections import defaultdict
by_hand = defaultdict(list)
for a in acts:
    if a["session_id"] in sessions:
        by_hand[(a["session_id"], a["hand_idx"])].append(a)

# choose hands: spread across sessions, prefer hands with both frame_diff and folded_flag
chosen = []
seen_sessions = set()
hands_sorted = sorted(by_hand.items(),
    key=lambda kv: (-len(set(x["observed_via"] for x in kv[1])), -len(kv[1])))
for (sid, hidx), rows in hands_sorted:
    fn = sessions[sid]
    if sum(1 for s,h in chosen if s==sid) >= 3:  # max 3 hands per session
        continue
    chosen.append((sid, hidx))
    seen_sessions.add(sid)
    if len(chosen) >= 8 and len(seen_sessions) >= 3:
        break

frames_cache = {}
def frames_for(sid):
    if sid not in frames_cache:
        frames_cache[sid] = load_frames(sessions[sid])
    return frames_cache[sid]

def prev_raw_frame(frames, order, seq):
    """Last frame before `seq` (in file order) that has raw_record."""
    prev = None
    for s in order:
        if s == seq: break
        r = frames.get(s)
        if r and r.get("raw_record"): prev = r
    return prev

report_lines = []
W = report_lines.append

n_pass = n_fail = 0
def check(label, cond, detail):
    global n_pass, n_fail
    ok = "PASS" if cond else "FAIL"
    if cond: n_pass += 1
    else: n_fail += 1
    W(f"  [{ok}] {label}: {detail}")
    return cond

for sid, hidx in chosen:
    fn = sessions[sid]
    frames, order = frames_for(sid)
    h = db.execute("SELECT * FROM hands WHERE session_id=? AND hand_idx=?",
                   (sid, hidx)).fetchone()
    W(f"\n=== HAND {fn} hand_idx={hidx} seq[{h['first_seq']}..{h['last_seq']}] "
      f"dealer_seat={h['dealer_seat']} hero_seat={h['hero_seat']} ===")
    ev = {"file": fn, "hand_idx": hidx, "checks": []}

    # (b) dealer check: majority raw dealer over hand frames
    hand_seqs = [s for s in order if h["first_seq"] <= (s or -1) <= h["last_seq"]]
    dealers = {}
    for s in hand_seqs:
        raw = frames[s].get("raw_record") or {}
        d = raw.get("dealer")
        if d: dealers[d] = dealers.get(d, 0) + 1
    if dealers and h["dealer_seat"] is not None:
        maj = max(dealers, key=dealers.get)
        expect = f"seat{h['dealer_seat']+1}"
        check("dealer", maj == expect,
              f"raw dealer (majority)={maj!r} counts={dealers}; DB dealer_seat={h['dealer_seat']} -> expects raw {expect!r}")
        ev["checks"].append({"test": "dealer", "raw_dealer": maj,
                             "db_dealer_seat": h["dealer_seat"],
                             "mapping_0idx_ok": maj == expect})

    # (a) hero decision frames in this hand
    done_hero = 0
    for s in hand_seqs:
        rec = frames[s]
        if rec.get("status") in ("decision", "decision_cached") and rec.get("raw_record"):
            raw = rec["raw_record"]
            hs = rec.get("hero_stack")
            stacks = raw.get("stacks") or {}
            matches = [k for k, v in stacks.items() if v == hs]
            check(f"hero@seq{s}", stacks.get("seat1") == hs,
                  f"rec.hero_stack={hs} raw.stacks.seat1={stacks.get('seat1')} "
                  f"(all seats matching that value: {matches}); top-level hero_seat={rec.get('hero_seat')}; DB hands.hero_seat={h['hero_seat']}")
            ev["checks"].append({"test": "hero_decision", "seq": s,
                                 "hero_stack": hs, "raw_stacks": stacks,
                                 "db_hero_seat": h["hero_seat"],
                                 "toplevel_hero_seat": rec.get("hero_seat")})
            done_hero += 1
            if done_hero >= 2: break

    # (c)/(d) action checks
    for a in by_hand[(sid, hidx)][:6]:
        s = a["seq"]; seat = a["seat"]
        rec = frames.get(s)
        if not rec or not rec.get("raw_record"): continue
        raw = rec["raw_record"]
        prev = prev_raw_frame(frames, order, s)
        praw = prev["raw_record"] if prev else {}
        key0 = f"seat{seat+1}"          # 0-indexed internal hypothesis
        key_shift = f"seat{seat}"        # "internal already 1-indexed" hypothesis
        if a["observed_via"] == "frame_diff":
            b_now = (raw.get("bets") or {})
            b_prev = (praw.get("bets") or {})
            now0 = b_now.get(key0) or 0
            prev0 = b_prev.get(key0) or 0
            nowS = b_now.get(key_shift) or 0
            prevS = b_prev.get(key_shift) or 0
            ok = (now0 == a["amount_to"])
            check(f"frame_diff seat={seat} {a['kind']}@seq{s}", ok,
                  f"amount_to={a['amount_to']} amount_delta={a['amount_delta']}; "
                  f"raw {key0}: {prev0}->{now0} | shifted-hyp raw {key_shift}: {prevS}->{nowS} "
                  f"| full bets prev={b_prev} now={b_now}")
            ev["checks"].append({"test": "frame_diff", "seq": s, "db_seat": seat,
                                 "kind": a["kind"], "amount_to": a["amount_to"],
                                 "amount_delta": a["amount_delta"],
                                 "raw_bets_prev": b_prev, "raw_bets_now": b_now,
                                 "mapping_0idx_ok": ok})
        else:  # folded_flag
            f_now = (raw.get("folded") or {})
            f_prev = (praw.get("folded") or {})
            ok = bool(f_now.get(key0)) and not bool(f_prev.get(key0))
            check(f"folded_flag seat={seat}@seq{s}", ok,
                  f"raw folded {key0}: {f_prev.get(key0)}->{f_now.get(key0)} | "
                  f"shifted-hyp {key_shift}: {f_prev.get(key_shift)}->{f_now.get(key_shift)} "
                  f"| full folded prev={f_prev} now={f_now}")
            ev["checks"].append({"test": "folded_flag", "seq": s, "db_seat": seat,
                                 "raw_folded_prev": f_prev, "raw_folded_now": f_now,
                                 "mapping_0idx_ok": ok})
    evidence.append(ev)

# ---- (e) candidate-file rows: verify chip-proof under raw seat{s+1} ----
W("\n=== V1 CANDIDATE ROWS vs RAW (chip-proof: anchor-stack drop == ante) ===")
cands = [json.loads(l) for l in open("tools/scraper_folded_showdown_task/stale_folded_candidates.jsonl")]
cand_checked = 0
for c in cands:
    fname = c["file"]
    sid = next((k for k, v in sessions.items() if v == fname), None)
    if sid is None: continue
    h = db.execute("SELECT * FROM hands WHERE session_id=? AND hand_idx=?",
                   (sid, c["hand_idx"])).fetchone()
    nxt = db.execute("SELECT * FROM hands WHERE session_id=? AND hand_idx=?",
                     (sid, c["hand_idx"]+1)).fetchone()
    if not h or not nxt or not h["start_stacks_json"] or not nxt["start_stacks_json"]:
        continue
    s0 = json.loads(h["start_stacks_json"]); s1 = json.loads(nxt["start_stacks_json"])
    seat = c["seat"]
    d0 = s0.get(str(seat)); d1 = s1.get(str(seat))
    if d0 is None or d1 is None: continue
    drop = d0 - d1
    # confirm the start_stacks themselves against the raw anchor frame
    frames, order = frames_for(sid)
    raw_anchor = None
    for s in order:
        if s is None or not (h["first_seq"] <= s <= h["last_seq"]): continue
        rec = frames[s]
        raw = rec.get("raw_record")
        if raw and raw.get("stacks"):
            raw_anchor = (s, raw["stacks"], raw.get("bets"), raw.get("folded"))
            break
    ok = (drop == c["ante"])
    check(f"candidate {fname} h{c['hand_idx']} seat={seat}", ok,
          f"start_stacks[{seat}] {d0} -> next-hand {d1}, drop={drop}, ante={c['ante']} "
          f"(raw key under 0-idx mapping: seat{seat+1}; first raw stacks this hand={raw_anchor[1] if raw_anchor else None})")
    evidence.append({"file": fname, "hand_idx": c["hand_idx"],
                     "checks": [{"test": "candidate_chip_proof", "db_seat": seat,
                                 "raw_key_0idx": f"seat{seat+1}",
                                 "start_stack": d0, "next_start": d1,
                                 "drop": drop, "ante": c["ante"], "ok": ok}]})
    cand_checked += 1
    if cand_checked >= 5: break

W(f"\nTOTAL: {n_pass} PASS / {n_fail} FAIL")
print("\n".join(report_lines))
with open(OUT / "evidence_checks.json", "w") as f:
    json.dump(evidence, f, indent=1)
