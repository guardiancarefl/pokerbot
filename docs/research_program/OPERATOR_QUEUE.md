# OPERATOR QUEUE — decisions awaiting the operator

Rule (Addendum 2): a track that hits an operator boundary parks its request
HERE and the program continues elsewhere. Check for stale entries at every
resume; move resolved entries to the Resolved section with the decision.

## Status (morning read — updated 2026-06-12 ~16:00)

H1 COMPLETE + ARMED (tail floor standard config per your OQ-2 approval;
first armed session will be auto-triaged per Addendum 4.6 watcher).
H2 probe IMMINENT: spec registered, battery frozen (champion
over-calls shoves: 35.9% call mass vs 7.7% oracle), adapter fixed —
**killphil gate row is now −0.080 ± 0.022, not −0.174: ~half the
believed extraction was adapter artifact** (e2 fix, commit a8933ea).
H2 probe interrupted at ckpt_1800 for your live window (resumes
automatically when the listener exits; verdict ~1.5 h later). H1 live:
2 armed sessions, both in-band, 0 tail-caused argmax changes. **P2
bet-closure recovery: APPROVED TO BUILD (your line, 2026-06-12) —
queued immediately after the H2 verdict chain, full gate treatment per
the original session-3 spec; today's AcKc safe-fold becomes a test
fixture.** OQ-1 Windows brief still awaits
routing (today's 49 suspect-frames/session says it matters). Nothing
else needs you.

## POD_CASE (standing one-liner — Addendum 5.5, update on every state change)

RIGHT NOW a 27-core pod would unlock: nothing yet decision-grade — the
H2 probe verdict lands ~17:00 today on this box. **IF the probe PASSES:
the bbnorm-carrying league retrain per BBNORM_TRANSPLANT_SPEC.md is the
immediate case — measured ~6-8 pod-hours vs ~18-26 Contabo-hours**
(two-phase: 1500 self-play + 500 league iters; bbnorm port from the
v2gate bundle is a blocking P0 either way), plus a pod frees all 12
local cores for eval batteries.

## Open

(none)

## Resolved

### OQ-2 (filed 2026-06-12, RESOLVED 2026-06-12) — H1 tail floor ARMED as standard config
- **Decision (operator, verbatim):** "APPROVED. Arm --tail-floor-tau 0.10
  in the dry-run checklist's listener line as standard config."
- **Action taken:** `docs/DRYRUN_CHECKLIST.md` listener launch line now
  includes `--tail-floor-tau 0.10` with the ARMED-banner check
  (`ARMED: tau_max=0.100`), expected firing magnitudes (distribution
  adjusted ~70–75% of decisions, sampled action changes ~4.4% per TG2),
  and the post-session triage eyeball step.
- **Evidence basis:** EXP_H1 TG1–TG4 all PASS
  (`reports/EXP_H1_tail_floor.md` §7). Registered caveats stand
  (self-play evidence, stale-attacker TG4): first armed live session
  observations go under H1 in EXPERIMENT_LOG.

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
