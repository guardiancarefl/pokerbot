# OPERATOR QUEUE — decisions awaiting the operator

Rule (Addendum 2): a track that hits an operator boundary parks its request
HERE and the program continues elsewhere. Check for stale entries at every
resume; move resolved entries to the Resolved section with the decision.

## Status (morning read — updated 2026-06-12 ~21:30)

**Cycle 1 closed; PROGRAM_REVIEW.md #1 is the read.** Three experiments
in ~18 h: H1 PASS (tail floor armed; 3 live sessions in-band, 0
tail-caused argmax changes incl. 104 firings in your 140-hand session);
H2 FAIL at probe (league fine-tune collapsed — mechanism confounded by
the slim checkpoint; H2b possible later, sequenced after cheaper
items); c1 MATERIAL (MH ICM underprices short-stack survival +0.065
below 5bb in high-dispersion bubble states — correction fit + consumer
re-price audit queued). P2 is BUILT+GATED-OFF and your AcKc turned out
to be a dead-SB reconstruction failure, not a displacement — the real
fix (carry blind structure into replay) is specced for the next build
slot. H4 at 383/500 with the spec pre-drafted. **Decisions needed:
P2 arming; OQ-1 Windows routing. No pod case stands.**

## POD_CASE (standing one-liner — Addendum 5.5, update on every state change)

RIGHT NOW: **no active pod case — H2 FAILED at probe** (all bars,
report: reports/EXP_H2_killphil_league.md; collapse predates the
interruption, relaunch clause rejected on evidence). The
bbnorm-transplant pod case is SUSPENDED pending the refill-pass verdict
on H2b (slim-ckpt fine-tuning fragility is the live suspect — any
successor must solve buffer continuity first). Next pod-relevant
decision point: H2b registration or ensemble-probe results.

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
