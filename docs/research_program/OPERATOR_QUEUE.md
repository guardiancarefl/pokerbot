# OPERATOR QUEUE — decisions awaiting the operator

Rule (Addendum 2): a track that hits an operator boundary parks its request
HERE and the program continues elsewhere. Check for stale entries at every
resume; move resolved entries to the Resolved section with the decision.

## Open

### OQ-1 (2026-06-12) — Windows scraper: showdown hole-card capture (route to Windows CC)
- **Decision needed:** add opponent shown-card OCR at showdown to the scraper
  frame schema (new optional field; bridge ignores it, only the H3 ingester
  reads it).
- **Evidence:** H3 pipeline build found the schema has no opponent hole-card
  field — 0 opponent holdings observable across 432 archived hands; also
  `folded` flags are stale (~30%), so fold-vs-shove is systematically
  undercounted (0/23 events). `data/opponent_db/README.md` documents both.
- **Recommendation:** queue it with the existing Layer-2 scraper work
  (SanityChecker reference decay); not urgent — H4 can proceed on
  action-frequency tendencies alone, but showdown capture upgrades the RNR
  target from frequencies to ranges.
- **Unblocks:** higher-fidelity H4 (range-conditioned RNR); does NOT block
  H4's 500-hand unlock (240/500 as of today).
- **Track state:** H3 continues unblocked (per-session ingest now in
  DRYRUN_CHECKLIST).

## Resolved

(none yet)
