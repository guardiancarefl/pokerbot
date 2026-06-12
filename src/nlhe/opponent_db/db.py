"""SQLite archive + field-statistics report for the opponent DB (H3).

Idempotency contract: a session is keyed by the sha256 of the jsonl file
content (session_id == file sha256). Re-ingesting the same content — from
logs/ or from an archive copy — REPLACES the session's rows atomically;
it can never duplicate. Schema documented in data/opponent_db/README.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.nlhe.opponent_db.parser import (
    DECISION_STATUSES,
    PARSER_VERSION,
    SessionParse,
    parse_session_file,
)

DEFAULT_DB_PATH = "data/opponent_db/opponent_db.sqlite"

# H4 unlocks at >= 500 opponent-observed hands (docs/research_program/
# PROGRAM.md section H4).
H4_THRESHOLD = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,   -- sha256 of the jsonl file
    file_name         TEXT NOT NULL,
    file_path         TEXT NOT NULL,
    started_utc       TEXT,
    mode              TEXT,
    seed              INTEGER,
    git_head          TEXT,
    n_frames          INTEGER NOT NULL,
    n_hands           INTEGER NOT NULL,
    n_decision_frames INTEGER NOT NULL,
    n_safe_folds      INTEGER NOT NULL,
    n_fallbacks       INTEGER NOT NULL,
    has_raw_record    INTEGER NOT NULL,   -- 0 = summary-tier old format
    statuses_json     TEXT,
    parser_version    TEXT NOT NULL,
    ingested_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hands (
    session_id        TEXT NOT NULL REFERENCES sessions(session_id),
    hand_idx          INTEGER NOT NULL,
    first_seq         INTEGER,
    last_seq          INTEGER,
    n_frames          INTEGER,
    n_usable_frames   INTEGER,
    captured_first    TEXT,
    captured_last     TEXT,
    duration_seconds  REAL,
    level             INTEGER,
    sb                INTEGER,
    bb                INTEGER,
    ante              INTEGER,
    dealer_seat       INTEGER,
    hero_seat         INTEGER,
    sb_seat           INTEGER,            -- NULL = dead-SB or unknown
    bb_seat           INTEGER,
    hero_cards        TEXT,
    board_final       TEXT,
    max_street        INTEGER,
    dealt_seats_json  TEXT,
    start_stacks_json TEXT,               -- NULL = no hand-start anchor
    anchor_found      INTEGER NOT NULL,
    net_chips_json    TEXT,               -- NULL = outcome unknown
    outcome_known     INTEGER NOT NULL,
    hero_reached_showdown INTEGER NOT NULL,
    n_decision_frames INTEGER NOT NULL,
    n_actions         INTEGER NOT NULL,
    n_opp_voluntary   INTEGER NOT NULL,
    opp_observed      INTEGER NOT NULL,   -- >=1 non-hero voluntary action
    PRIMARY KEY (session_id, hand_idx)
);
CREATE TABLE IF NOT EXISTS actions (
    action_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL,
    hand_idx       INTEGER NOT NULL,
    event_idx      INTEGER NOT NULL,
    seat           INTEGER NOT NULL,      -- anonymous: (session_id, seat)
    is_hero        INTEGER NOT NULL,
    street         INTEGER NOT NULL,      -- 0=preflop..3=river
    kind           TEXT NOT NULL,
    amount_to      INTEGER,
    amount_delta   INTEGER,
    stack_before   INTEGER,
    stack_depth_bb REAL,
    all_in         INTEGER NOT NULL,
    facing_allin   INTEGER NOT NULL,
    voluntary      INTEGER NOT NULL,
    observed_via   TEXT NOT NULL,
    seq            INTEGER,
    captured_at    TEXT,
    FOREIGN KEY (session_id, hand_idx) REFERENCES hands(session_id, hand_idx)
);
CREATE INDEX IF NOT EXISTS idx_actions_session_seat
    ON actions(session_id, seat);
CREATE TABLE IF NOT EXISTS showdowns (
    session_id  TEXT NOT NULL,
    hand_idx    INTEGER NOT NULL,
    seat        INTEGER NOT NULL,
    is_hero     INTEGER NOT NULL,
    cards       TEXT,
    source      TEXT NOT NULL,
    PRIMARY KEY (session_id, hand_idx, seat)
);
"""


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def is_recently_modified(path: str, min_age_seconds: float = 600.0) -> bool:
    """True if the file was modified within min_age_seconds — i.e. a
    session log that may still be written by the LIVE listener. Such
    files must be skipped and re-ingested later."""
    age = datetime.now().timestamp() - os.path.getmtime(path)
    return age < min_age_seconds


def ingest_session(conn: sqlite3.Connection, path: str,
                   parse: SessionParse | None = None) -> dict:
    """Idempotent ingest of one session jsonl. Returns a summary dict."""
    sha = file_sha256(path)
    if parse is None:
        parse = parse_session_file(path)
    header = parse.header or {}
    n_dec = sum(parse.statuses.get(s, 0) for s in DECISION_STATUSES)
    n_safe = parse.statuses.get("safe_fold", 0)
    now = datetime.now(timezone.utc).isoformat()

    cur = conn.cursor()
    existed = cur.execute(
        "SELECT 1 FROM sessions WHERE session_id=?", (sha,)).fetchone()
    # Replace, never duplicate.
    cur.execute("DELETE FROM actions WHERE session_id=?", (sha,))
    cur.execute("DELETE FROM showdowns WHERE session_id=?", (sha,))
    cur.execute("DELETE FROM hands WHERE session_id=?", (sha,))
    cur.execute("DELETE FROM sessions WHERE session_id=?", (sha,))

    cur.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sha, Path(path).name, str(Path(path).resolve()),
         header.get("started_utc"), header.get("mode"), header.get("seed"),
         header.get("git_head"), parse.n_frames, len(parse.hands),
         n_dec, n_safe, parse.n_fallbacks, int(parse.has_raw_record),
         json.dumps(parse.statuses), PARSER_VERSION, now))

    for h in parse.hands:
        n_opp_vol = sum(1 for a in h.actions
                        if not a.is_hero and a.voluntary
                        and a.observed_via == "frame_diff")
        opp_observed = int(n_opp_vol > 0)
        cur.execute(
            "INSERT INTO hands VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
            "?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sha, h.hand_idx, h.first_seq, h.last_seq, h.n_frames,
             h.n_usable_frames, h.captured_first, h.captured_last,
             h.duration_seconds, h.level, h.sb, h.bb, h.ante,
             h.dealer_seat, h.hero_seat, h.sb_seat, h.bb_seat,
             h.hero_cards, h.board_final, h.max_street,
             json.dumps(h.dealt_seats),
             json.dumps(h.start_stacks) if h.start_stacks else None,
             int(h.anchor_found),
             json.dumps(h.net_chips) if h.net_chips else None,
             int(h.outcome_known), int(h.hero_reached_showdown),
             h.n_decision_frames, len(h.actions), n_opp_vol, opp_observed))
        for a in h.actions:
            cur.execute(
                "INSERT INTO actions (session_id, hand_idx, event_idx,"
                " seat, is_hero, street, kind, amount_to, amount_delta,"
                " stack_before, stack_depth_bb, all_in, facing_allin,"
                " voluntary, observed_via, seq, captured_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sha, h.hand_idx, a.event_idx, a.seat, int(a.is_hero),
                 a.street, a.kind, a.amount_to, a.amount_delta,
                 a.stack_before, a.stack_depth_bb, int(a.all_in),
                 int(a.facing_allin), int(a.voluntary), a.observed_via,
                 a.seq, a.captured_at))
        if h.hero_reached_showdown and h.hero_cards:
            cur.execute(
                "INSERT INTO showdowns VALUES (?,?,?,?,?,?)",
                (sha, h.hand_idx, h.hero_seat, 1, h.hero_cards,
                 "hero_inferred_river_reach"))
    conn.commit()
    return {
        "session_id": sha, "file": Path(path).name,
        "replaced": bool(existed), "hands": len(parse.hands),
        "decision_frames": n_dec,
        "actions": sum(len(h.actions) for h in parse.hands),
        "opp_voluntary": sum(
            1 for h in parse.hands for a in h.actions
            if not a.is_hero and a.voluntary
            and a.observed_via == "frame_diff"),
        "opp_observed_hands": sum(
            1 for h in parse.hands
            if any(not a.is_hero and a.voluntary
                   and a.observed_via == "frame_diff" for a in h.actions)),
        "showdown_rows": sum(1 for h in parse.hands
                             if h.hero_reached_showdown and h.hero_cards),
        "has_raw_record": parse.has_raw_record,
    }


# ── field-statistics report ─────────────────────────────────────────────


def _q(conn, sql, args=()):
    return conn.execute(sql, args).fetchall()


def field_report(conn: sqlite3.Connection) -> str:
    """Per-session + field-aggregate statistics. All opponent statistics
    are computed from OBSERVED table evidence only (frame_diff /
    folded_flag rows; never the hero's decision_record rows) and are
    lower bounds — see data/opponent_db/README.md for caveats."""
    lines = []
    W = lines.append
    W("=" * 76)
    W("OPPONENT-DB FIELD REPORT (H3)")
    W(f"generated {datetime.now(timezone.utc).isoformat()}  "
      f"parser v{PARSER_VERSION}")
    W("=" * 76)
    W("")
    W("Per-session  (opp_* = non-hero seats, observed frame evidence only)")
    W("")
    hdr = (f"{'session file':42s} {'hands':>5s} {'oppObs':>6s} "
           f"{'dec':>4s} {'acts':>5s} {'oppVol':>6s} {'sd':>3s} "
           f"{'VPIP':>6s} {'foldVshove':>10s}")
    W(hdr)
    W("-" * len(hdr))

    sessions = _q(conn, "SELECT session_id, file_name, n_hands,"
                        " n_decision_frames, has_raw_record FROM sessions"
                        " ORDER BY file_name")
    agg = {"hands": 0, "opp_obs": 0, "vpip_n": 0, "vpip_d": 0,
           "shove_fold": 0, "shove_opp": 0, "sd": 0, "acts": 0,
           "opp_vol": 0, "dec": 0}
    for sid, fname, n_hands, n_dec, has_raw in sessions:
        opp_obs = _q(conn, "SELECT COUNT(*) FROM hands WHERE session_id=?"
                           " AND opp_observed=1", (sid,))[0][0]
        n_acts = _q(conn, "SELECT COUNT(*) FROM actions WHERE session_id=?",
                    (sid,))[0][0]
        n_opp_vol = _q(conn, "SELECT COUNT(*) FROM actions WHERE"
                             " session_id=? AND is_hero=0 AND voluntary=1"
                             " AND observed_via='frame_diff'", (sid,))[0][0]
        n_sd = _q(conn, "SELECT COUNT(*) FROM showdowns WHERE session_id=?",
                  (sid,))[0][0]
        # VPIP: per (session, seat): dealt hands vs hands with >=1
        # voluntary preflop observed action. Summed across non-hero seats.
        vpip_n, vpip_d = _session_vpip(conn, sid)
        # fold-vs-shove: observed non-hero events taken while facing an
        # outstanding all-in.
        sh = _q(conn, "SELECT SUM(CASE WHEN kind='fold' THEN 1 ELSE 0 END),"
                      " COUNT(*) FROM actions WHERE session_id=? AND"
                      " is_hero=0 AND facing_allin=1 AND"
                      " observed_via IN ('frame_diff','folded_flag')",
                (sid,))[0]
        sh_f, sh_t = (sh[0] or 0), (sh[1] or 0)
        vpip_s = f"{vpip_n}/{vpip_d}" if vpip_d else "-"
        shove_s = f"{sh_f}/{sh_t}" if sh_t else "-"
        tag = "" if has_raw else "  [summary-tier: no raw_record]"
        W(f"{fname:42s} {n_hands:5d} {opp_obs:6d} {n_dec:4d} "
          f"{n_acts:5d} {n_opp_vol:6d} {n_sd:3d} {vpip_s:>6s} "
          f"{shove_s:>10s}{tag}")
        agg["hands"] += n_hands; agg["opp_obs"] += opp_obs
        agg["vpip_n"] += vpip_n; agg["vpip_d"] += vpip_d
        agg["shove_fold"] += sh_f; agg["shove_opp"] += sh_t
        agg["sd"] += n_sd; agg["acts"] += n_acts
        agg["opp_vol"] += n_opp_vol; agg["dec"] += n_dec

    W("")
    W("FIELD AGGREGATE (all sessions, non-hero observed evidence)")
    W(f"  sessions ingested:           {len(sessions)}")
    W(f"  hands total:                 {agg['hands']}"
      f"  (avg {agg['hands']/len(sessions):.1f}/session)"
      if sessions else "  hands total: 0")
    W(f"  hero decision frames:        {agg['dec']}")
    W(f"  action rows total:           {agg['acts']}")
    W(f"  opp voluntary actions:       {agg['opp_vol']}")
    vp = (100.0 * agg["vpip_n"] / agg["vpip_d"]) if agg["vpip_d"] else 0.0
    W(f"  opp VPIP (observed, lower bound): {agg['vpip_n']}/{agg['vpip_d']}"
      f" seat-hands = {vp:.1f}%")
    fs = (100.0 * agg["shove_fold"] / agg["shove_opp"]) \
        if agg["shove_opp"] else 0.0
    W(f"  opp fold-vs-shove:           {agg['shove_fold']}/"
      f"{agg['shove_opp']} observed facing-allin events = {fs:.1f}%"
      f"  (folds undercounted: stale folded flags)")
    W(f"  showdown rows:               {agg['sd']} (hero only, inferred "
      f"river-reach; opponent holdings: 0 — scraper has no opponent "
      f"hole-card field)")
    W("")
    counter = agg["opp_obs"]
    state = "UNLOCKED" if counter >= H4_THRESHOLD else "LOCKED"
    W(f"  H4 UNLOCK COUNTER: {counter} opponent-observed hands "
      f"(hands with >=1 observed non-hero voluntary action)")
    W(f"  H4 threshold {H4_THRESHOLD} -> {state}"
      + ("" if counter >= H4_THRESHOLD else
         f"  ({H4_THRESHOLD - counter} more needed)"))
    W("")
    return "\n".join(lines)


def _session_vpip(conn, sid):
    """(numerator, denominator) over non-hero seats: seat-hands dealt vs
    seat-hands with >=1 voluntary preflop observed action."""
    rows = _q(conn, "SELECT hand_idx, hero_seat, dealt_seats_json"
                    " FROM hands WHERE session_id=?", (sid,))
    vol = set((r[0], r[1]) for r in _q(
        conn, "SELECT DISTINCT hand_idx, seat FROM actions WHERE"
              " session_id=? AND is_hero=0 AND voluntary=1 AND street=0"
              " AND observed_via='frame_diff'", (sid,)))
    num = den = 0
    for hand_idx, hero_seat, dealt_json in rows:
        dealt = json.loads(dealt_json or "[]")
        for seat in dealt:
            if seat == hero_seat:
                continue
            den += 1
            if (hand_idx, seat) in vol:
                num += 1
    return num, den


def h4_counter(conn: sqlite3.Connection) -> int:
    return _q(conn, "SELECT COUNT(*) FROM hands WHERE opp_observed=1")[0][0]


# ── parquet export ──────────────────────────────────────────────────────


def export_parquet(conn: sqlite3.Connection,
                   out_dir: str = "data/opponent_db/parquet") -> str:
    """One parquet file per table via pandas. If pandas/pyarrow are not in
    the venv (they are not, as of 2026-06-12), writes SKIPPED.txt and
    returns a message instead of failing the ingest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # noqa
    except ImportError as e:
        msg = (f"parquet export skipped: {e}. The venv (python 3.10, "
               f"requirements.txt) does not include pandas/pyarrow. "
               f"Install them and re-run "
               f"`python -m scripts.ingest_session --all --export-parquet` "
               f"to produce parquet files; the sqlite archive is complete "
               f"without them.")
        (out / "SKIPPED.txt").write_text(msg + "\n")
        return msg
    try:
        import pandas as pd
        for table in ("sessions", "hands", "actions", "showdowns"):
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            df.to_parquet(out / f"{table}.parquet", index=False)
        skipped = out / "SKIPPED.txt"
        if skipped.exists():
            skipped.unlink()
        return f"parquet export: 4 tables -> {out}/"
    except Exception as e:  # pyarrow missing despite pandas, etc.
        msg = f"parquet export skipped: {e}"
        (out / "SKIPPED.txt").write_text(msg + "\n")
        return msg
