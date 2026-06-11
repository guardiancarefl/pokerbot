# Triage sample outputs — 2026-06-11

Output of `scripts/dryrun_triage.py` over every historical dry-run log
(the `live_dryrun_*.jsonl` corpus in gitignored `logs/`), committed as
the tool's acceptance evidence.

- `triage_<ts>.txt` — the eight historical sessions as recorded
  (pre-`aa2f505`, so all carry the loud "no session_header" warning).
  Validated against the documented postmortems:
  - 152756: skip sub-causes 17 suspect / 17 dealer-missing / 4
    dealer-non-alive / 3 blinds-string — matches STATUS exactly; the one
    OCR-lost hand is the known 8dKc hand; seq-170 and seq-192 are
    classified into their named reconstruction-signature classes; the
    pre-floor seq=48 FOLD-facing-0 is flagged loudly.
  - 154557 (the worst session: 571 frames, 510 skips, 247 data-quality):
    14 OCR-lost hands, matching the corrected 2026-06-09 re-analysis
    (incl. the AdKd suspect-killed hand and both safe-fold-only hands).
  - verify1: the seq 83-94 blackout shows up as
    `hero-to-act frames skipped = 10`.
- `triage_demo_154557_replay.txt` — modern-format demo: 154557's raw
  records replayed through `run_live_dryrun.py --replay-from-jsonl`
  under the deployed checkpoint at HEAD, so the log carries a
  session_header (mode/seed/sha asserts PASS), the 6 layer-1 recoveries,
  2 ANCHOR-REFUSED red flags (the stuck-digit 9907 poisoned anchors),
  and a floor column joined from the `.stdout` companion.

Regenerate any of these with:

```bash
.venv/bin/python scripts/dryrun_triage.py logs/live_dryrun_<ts>.jsonl
```
