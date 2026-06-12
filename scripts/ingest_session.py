"""Ingest live dry-run session jsonl logs into the opponent DB (H3).

Usage (always as module, repo root, venv active):

    python -m scripts.ingest_session logs/live_dryrun_<TS>.jsonl
    python -m scripts.ingest_session --all          # sweep logs/ + archive/
    python -m scripts.ingest_session --stats        # field report only

Safety / idempotency:
  * IDEMPOTENT: sessions keyed by file-content sha256; re-ingest replaces,
    never duplicates. logs/ and archive copies of the same session dedupe.
  * LIVE-SESSION GUARD: any jsonl modified within the last 10 minutes is
    assumed to be an in-flight live session and is SKIPPED (re-run after
    the session ends).
  * Read-only on logs/; writes only data/opponent_db/.

--all sweeps logs/live_dryrun_*.jsonl plus any live_dryrun_*.jsonl members
of archive/live_logs_archive_*.tar.gz (extracted to a temp dir; archive
copies are sha-identical to logs/ so they dedupe via the session key).
"""
from __future__ import annotations

import argparse
import glob
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nlhe.opponent_db import db as odb  # noqa: E402

FIELD_REPORT_PATH = REPO_ROOT / "data/opponent_db/FIELD_REPORT.txt"
MIN_AGE_SECONDS = 600.0


def _eligible(path: str) -> bool:
    if odb.is_recently_modified(path, MIN_AGE_SECONDS):
        print(f"  SKIP (modified <10 min ago — likely a LIVE session, "
              f"re-run later): {path}")
        return False
    return True


def _sweep_paths() -> list[str]:
    paths = sorted(glob.glob(str(REPO_ROOT / "logs/live_dryrun_*.jsonl")))
    return paths


def _archive_members(tmpdir: str) -> list[str]:
    """Extract live_dryrun_*.jsonl members from archive tarballs."""
    out: list[str] = []
    for tar_path in sorted(glob.glob(
            str(REPO_ROOT / "archive/live_logs_archive_*.tar.gz"))):
        with tarfile.open(tar_path) as tf:
            members = [m for m in tf.getmembers()
                       if m.isfile()
                       and Path(m.name).name.startswith("live_dryrun_")
                       and m.name.endswith(".jsonl")]
            for m in members:
                tf.extract(m, tmpdir)
                out.append(str(Path(tmpdir) / m.name))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Ingest live session jsonl into the opponent DB (H3)")
    ap.add_argument("paths", nargs="*", help="session jsonl file(s)")
    ap.add_argument("--all", action="store_true",
                    help="sweep logs/live_dryrun_*.jsonl + archive/ tarballs")
    ap.add_argument("--stats", action="store_true",
                    help="print the field report (and write "
                         "data/opponent_db/FIELD_REPORT.txt)")
    ap.add_argument("--db", default=str(REPO_ROOT / odb.DEFAULT_DB_PATH))
    ap.add_argument("--export-parquet", action="store_true", default=None,
                    help="export parquet after ingest (default: on when "
                         "anything was ingested)")
    args = ap.parse_args(argv)

    if not args.paths and not args.all and not args.stats:
        ap.error("give session path(s), --all, or --stats")

    conn = odb.connect(args.db)
    ingested = []
    skipped = 0

    targets = list(args.paths)
    tmpdir_ctx = None
    if args.all:
        targets += _sweep_paths()
        tmpdir_ctx = tempfile.TemporaryDirectory(prefix="oppdb_archive_")
        targets += _archive_members(tmpdir_ctx.name)

    seen_sha = set()
    for path in targets:
        if not Path(path).exists():
            print(f"  MISSING: {path}")
            continue
        if not _eligible(path):
            skipped += 1
            continue
        sha = odb.file_sha256(path)
        if sha in seen_sha:
            print(f"  DEDUP (same content already ingested this run): "
                  f"{Path(path).name}")
            continue
        seen_sha.add(sha)
        summary = odb.ingest_session(conn, path)
        ingested.append(summary)
        tag = " (replaced)" if summary["replaced"] else ""
        tier = "" if summary["has_raw_record"] else " [summary-tier]"
        print(f"  ingested {summary['file']}{tag}{tier}: "
              f"hands={summary['hands']} actions={summary['actions']} "
              f"opp_vol={summary['opp_voluntary']} "
              f"opp_obs_hands={summary['opp_observed_hands']} "
              f"showdowns={summary['showdown_rows']}")

    if tmpdir_ctx is not None:
        tmpdir_ctx.cleanup()

    if ingested and args.export_parquet is not False:
        print(odb.export_parquet(conn))

    if args.stats or ingested:
        report = odb.field_report(conn)
        FIELD_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIELD_REPORT_PATH.write_text(report + "\n")
        if args.stats:
            print(report)
        print(f"field report written: {FIELD_REPORT_PATH}")
        print(f"H4 UNLOCK COUNTER: {odb.h4_counter(conn)} / "
              f"{odb.H4_THRESHOLD}")
    if skipped:
        print(f"{skipped} file(s) skipped as possibly-live; re-run "
              f"`python -m scripts.ingest_session --all` after the session.")
    conn.close()


if __name__ == "__main__":
    main()
