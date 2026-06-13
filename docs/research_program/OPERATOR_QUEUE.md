# OPERATOR QUEUE — decisions awaiting the operator

Rule (Addendum 2): a track that hits an operator boundary parks its request
HERE and the program continues elsewhere. Check for stale entries at every
resume; move resolved entries to the Resolved section with the decision.

## Status (morning read — updated 2026-06-13 ~03:30)

**UNLOCK WON** (hero 6975 double-up; all-time 27 played, 10W/13L).
First fully-armed session: the session-5 package WORKED — its target
classes (dealer-burst, commit-recon) were absent or correctly declined;
0 floor-gaps. **Strategic shift: the binding deployment constraint is
now the Windows scraper SanityChecker (OCR-suspect frames) — 9 of 10
lost hands this session were OCR gaps, not bridge logic.** Tail floor:
6 armed sessions, still 0 argmax changes. Stage-2 gate streak 0/5
(scraper quality, not bridge, is what's gating it now).

YOUR OPEN ITEMS: (1) shove-floor live-wiring is building its gates incl.
the p99-latency check — I'll bring it for arming only when green; you've
already approved the floor itself. (2) Windows: P1/P2 SanityChecker is
now top-priority (it's THE Stage-2 blocker); OQ-1 Part-B, OQ-3 F1,
D3 also queued your side. (3) F2 stays held until Windows F1.

## POD_CASE (standing — Addendum 5.5, updated 2026-06-13 ~03:35)

**H4 UNLOCKED (517/500) — ingest committed 2026-06-13; spec FROZEN (56edb84).**
The H4 field-RNR retrain is the first training-scale job since H2 and the pod
restart trigger. **Full case: `POD_CASE_H4.md`** — pod ask ~6–8 pod-hours
(4.7–5.8h retrain + ~1.5–2h gates) vs 23–31 Contabo-hours; probe is
Contabo-only (~9–12h, no pod) and must PASS before any pod spend; operator go
required. Key decisions now locked: **legacy 236-d encoder, train fresh — NO
bbnorm organ** (one-delta discipline; bbnorm spec void); **p = 0.25** (OQ-1
fold data sub-grade); **H2-fragility-safe** (full-buffer / league-from-iter-0,
no slim-ckpt fine-tune); gate battery is the full unweakened program (PG1–PG4
+ ATT) **plus the shoviest panel** (exploiter must not lose to known
shovers). The bbnorm-transplant variant could ride a LATER pod run as its own
registered experiment. Lower-tier pod candidate: ICM Tier-2 (~3–10 core-h,
Contabo-fine).

## Open

### OQ-4 (2026-06-13) — Windows scraper: WINDOW-HANDLE capture + felt-collapse auto-pause — URGENT (operator routing)
- **Root cause, session-6 (proven, not hypothesized).** The "recognition
  collapse" was NOT a tunnel drop and NOT OCR failure: it was the scraper
  capturing the **wrong window**. Frames arrived continuously (median 1.49s,
  max gap 35s, ZERO >60s; arrival-gaps ≡ capture-gaps to 2 decimals → no
  delivery starvation, no backlog; one TCP reconnect, at the very start).
  25 frames had `felt_fraction < 0.10` + `table_rendered=False`, and their
  OCR text reads back the scraper's OWN environment: `"_read_table.py"`,
  `"127.0.0.1:9000"`, and the Contabo listener's own stdout
  (`"controls.present=False (not a hero-to-act decision)"`). The capture
  grabbed terminals/SSH consoles, worsening 2→9→14 across session thirds as
  the operator interacted with terminals during the network blip. The
  operator's network event hit the **CC/SSH terminal, not the frame
  pipeline.** Evidence: this session's audit (`logs/live_dryrun_20260613_023826`).
- **Fix #1 (the real fix — WGC, now URGENT; tonight is its proof case):**
  pin capture to the poker-table **window handle** (Windows Graphics Capture
  / per-HWND capture), NOT foreground/screen-region capture, so operator
  terminal activity and focus changes can never bleed into frames.
- **Fix #2 (cheap interim guard before WGC lands):** auto-pause capture /
  stop sending when `felt_fraction` collapses below threshold for N
  consecutive frames — stop feeding the bridge terminal screenshots. (felt
  median stayed ~0.79 on good frames all session, so a low threshold ≈ 0.10
  cleanly separates table from non-table; pick N≈3.)
- Operator will route to Windows CC. Independent of OQ-3 (that is OCR on
  rendered tables; this is wrong-window capture of non-tables) — both needed.
- **Not a blocker for the H4 probe** (offline/file batteries), but a
  precondition for trusting any live dry-run of the shipped exploiter.

### OQ-3 (2026-06-13) — Windows scraper: pointed-seat stack OCR (dealer-burst recurrence) — MUST-LAND
- Session-5 postmortem: the dealer-burst hand-killer RECURRED with a
  NEW mechanism — the dealer-pointed seat's stack OCR reads None/0 on
  ALIVE seats (seats 2/3/5; 54 frames, 4 hands killed incl. KQo/ATo
  fallbacks). The 2026-06-09 hero-button fix still holds; this is a
  different field. Re-prioritized MUST-LAND alongside the P1/P2
  SanityChecker work and OQ-1 Part-B routing.
- D2 dead-button handling (bridge-side, building) covers the
  legitimately-dead-button subset; this OQ covers the OCR-false-dead
  subset — both needed.

### Operator confirmations (RESOLVED 2026-06-13)
- Stage-2 gate **CONFIRMED at N=5** (zero never-decided + zero required
  interventions, consecutive). Ledger header updated.
- Game count: **detector confirmed right — 4 games (L,L,L,W)**; the
  all-time table (26 played, 9W/13L/4U) stands as computed.

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
