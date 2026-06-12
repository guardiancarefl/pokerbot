# OPERATOR QUEUE — decisions awaiting the operator

Rule (Addendum 2): a track that hits an operator boundary parks its request
HERE and the program continues elsewhere. Check for stale entries at every
resume; move resolved entries to the Resolved section with the decision.

## Open

### OQ-2 (2026-06-12) — SHIP RECOMMENDATION: arm the H1 tail floor (`--tail-floor-tau 0.10`)
- **Decision needed:** arm the commitment-scaled tail floor in the live
  dry-run config (add `--tail-floor-tau 0.10` to the listener launch
  line). Build is already deployed and OFF by default; arming is a
  flag, not a code change.
- **Evidence (full pre-registered chain, EXPERIMENT_LOG 2026-06-12):**
  TG1 byte-identity 16/16 logs; TG2 exact-commit all bars at tau=0.10
  (seq-315 caught, 0 argmax changes, altered 4.37%); **TG3 24k paired:
  all-games ICM delta +0.1227 +/- 0.0073 (z=16.8)**, diverged-only
  +0.1597 +/- 0.0095; **TG4 attacker re-measure: kill bar not
  triggered** — standard extraction IMPROVED 0.10/game vs B4 (paired
  z=-4.8), bubble -0.0295 +/- 0.0151 vs B5on (robustness direction).
- **Caveats already registered:** self-play-vs-field generalization
  (H1.2 narrowed claim), stale-attacker TG4 instrument (trained vs the
  no-tail champion; a tail-aware attacker is future work).
- **Recommendation:** ARM for the next live dry-run session; audit
  `[FLOOR] fired=[tail]` lines in the session triage (dryrun_triage
  already joins floor firings).

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
