"""Unit tests for the opponent-data pipeline (track H3).

Focus: hand-boundary detection (against the audited 163815 ground truth
and a fixture cut from that log) and idempotent re-ingest.

Fixture: tests/fixtures/oppdb_fixture_session.jsonl = session header +
frames seq 1..60 of logs/live_dryrun_20260611_163815.jsonl (the first 4
complete hands of the session-2 audit ground truth: 37 hands / 61
decision frames).
"""
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nlhe.opponent_db import db as odb
from src.nlhe.opponent_db.parser import parse_session_file

FIXTURE = REPO_ROOT / "tests/fixtures/oppdb_fixture_session.jsonl"
FULL_LOG = REPO_ROOT / "logs/live_dryrun_20260611_163815.jsonl"


# ── hand-boundary detection ─────────────────────────────────────────────


def test_fixture_hand_boundaries():
    p = parse_session_file(str(FIXTURE))
    assert len(p.hands) == 4
    # (first_seq, last_seq, hero_cards) per hand — cut from the audited
    # session; boundary logic must reproduce these exactly.
    expected = [
        (1, 22, "Kd Ts"),
        (23, 36, "7c Jh"),
        (37, 46, "7c Jh"),   # same pair, dealer moved (board shrink)
        (47, 60, "5c 8c"),
    ]
    got = [(h.first_seq, h.last_seq, h.hero_cards) for h in p.hands]
    assert got == expected


def test_fixture_anchor_and_blinds():
    p = parse_session_file(str(FIXTURE))
    h0 = p.hands[0]
    assert (h0.sb, h0.bb, h0.ante) == (15, 25, 5)
    assert h0.anchor_found
    # Single-table SNG: anchored pre-hand stacks conserve 6 x 1500 chips.
    assert sum(h0.start_stacks.values()) == 9000
    # Outcome derivable when the NEXT hand is anchored too.
    assert h0.outcome_known
    assert sum(h0.net_chips.values()) == 0


def test_fixture_decision_frames_attributed():
    p = parse_session_file(str(FIXTURE))
    # statuses in the cut: decision + decision_cached must all be
    # attributed to exactly one hand each.
    n_dec_status = sum(p.statuses.get(s, 0)
                       for s in ("decision", "decision_cached"))
    n_dec_hands = sum(h.n_decision_frames for h in p.hands)
    assert n_dec_status == n_dec_hands > 0


def test_full_session_ground_truth():
    """Audited ground truth for session 20260611_163815: 37 hands, 61
    hero decision frames (decision_audit 2026-06-11)."""
    if not FULL_LOG.exists():
        import pytest
        pytest.skip("full session log not present")
    p = parse_session_file(str(FULL_LOG))
    assert len(p.hands) == 37
    assert sum(h.n_decision_frames for h in p.hands) == 61


# ── idempotent re-ingest ───────────────────────────────────────────────


def _counts(conn):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("sessions", "hands", "actions", "showdowns")}


def test_idempotent_reingest(tmp_path):
    db_path = str(tmp_path / "test.sqlite")
    conn = odb.connect(db_path)
    s1 = odb.ingest_session(conn, str(FIXTURE))
    c1 = _counts(conn)
    assert c1["sessions"] == 1
    assert c1["hands"] == 4
    assert c1["actions"] > 0
    assert not s1["replaced"]

    s2 = odb.ingest_session(conn, str(FIXTURE))
    c2 = _counts(conn)
    assert s2["replaced"]
    assert c1 == c2, "re-ingest must replace, never duplicate"
    conn.close()


def test_same_content_different_path_dedupes(tmp_path):
    """Archive copies are sha-identical to logs/ copies — they must map
    to the same session row."""
    db_path = str(tmp_path / "test.sqlite")
    copy = tmp_path / "copied_session.jsonl"
    copy.write_bytes(FIXTURE.read_bytes())
    conn = odb.connect(db_path)
    odb.ingest_session(conn, str(FIXTURE))
    odb.ingest_session(conn, str(copy))
    assert _counts(conn)["sessions"] == 1
    assert _counts(conn)["hands"] == 4
    conn.close()


def test_live_session_guard(tmp_path):
    """A jsonl modified <10 min ago must be flagged as possibly live."""
    f = tmp_path / "fresh.jsonl"
    f.write_text("{}\n")
    assert odb.is_recently_modified(str(f), 600.0)
    old = time.time() - 3600
    os.utime(f, (old, old))
    assert not odb.is_recently_modified(str(f), 600.0)


def test_anonymity_no_identity_columns(tmp_path):
    """Opponents keyed by (session_id, seat) only — schema must not grow
    name/identity columns (project core constraint)."""
    conn = odb.connect(str(tmp_path / "t.sqlite"))
    for table in ("hands", "actions", "showdowns"):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        for forbidden in ("player", "name", "username", "identity"):
            assert not any(forbidden in c.lower() for c in cols)
    conn.close()
