# OPERATOR QUEUE — decisions awaiting the operator

Rule (Addendum 2): a track that hits an operator boundary parks its request
HERE and the program continues elsewhere. Check for stale entries at every
resume; move resolved entries to the Resolved section with the decision.

## Open

(none)

## Resolved

### OQ-1 (filed 2026-06-12, RESOLVED 2026-06-12) — Windows scraper: folded-flag fix + showdown capture
- **Decision (operator):** APPROVED, scope EXPANDED, priority RAISED.
  (1) The stale `folded` flags are the priority half — they corrupt
  fold-vs-shove counting, which is both H3's field stats AND H2's probe
  metric. Fix first. (2) Showdown hole-card capture approved as specced
  (optional schema field, bridge ignores, ingester reads).
- **Action taken:** full forensics-grade Windows-task brief written at
  `docs/WINDOWS_TASK_SCRAPER_FOLDED_SHOWDOWN.md` (PNG-evidence-first,
  fix proposal review, offline replay proof over archived sessions
  before live use; bridge provably untouched via replay + injection
  gates; H3 ingester gains the new fields behind `schema_version: 2`).
  Candidate forensic frames packaged at
  `tools/scraper_folded_showdown_task/stale_folded_candidates.jsonl`
  (extractor: `scripts/extract_stale_folded_candidates.py`).
- **New evidence found while packaging:** staleness is near-TOTAL per
  fold event, not ~30% — 241 chip-proven non-blind preflop folds in
  postflop-reaching outcome-known hands, **0/241** got a `folded`
  transition; the whole 432-hand corpus has only 16 transitions.
- **Awaiting:** operator routes the brief to a Windows CC session.
