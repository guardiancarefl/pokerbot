# NEXT_RESUME — updated 2026-06-12 ~18:30 (H2 CLOSED-FAIL; P2 build + night shift active)

**Read first:** PROGRAM.md (Addenda 1–5), EXPERIMENT_LOG.md (last four
entries), OPERATOR_QUEUE.md, COMPUTE_QUEUE.md, then this. Cross-check
`git log --oneline -10`.

## State

- **H1: DEPLOYED + ARMED** (OQ-2). Two armed live sessions, both
  in-band (0 tail-caused argmax changes). Post-session pipeline is
  automatic (Addendum 4.6; tmux watcher + staleness proxy — process
  watchers DON'T work in background tasks, file signals only).
- **H2: CLOSED — FAIL at probe** (`reports/EXP_H2_killphil_league.md`).
  All bars broken; collapse predates the interruption (relaunch clause
  rejected on evidence). Open mechanism: slim-ckpt fine-tuning
  fragility vs league-teaching failure. **H2b = refill candidate** (must
  solve buffer continuity: full-buffer ckpt or league-from-iter-0).
  Surviving instruments: fold-vs-shove battery, post-fix baselines
  (killphil −0.0800 ± 0.0223), b2 stability (spread 0.022), league infra.
- **P2 bet-closure recovery: OPERATOR-APPROVED, BUILD STARTING**
  (task #7; agent-built, manager-gated). Fixture: session 161347
  seq 276 AcKc invariant_fail. Gates: flag-off byte-identity over ALL
  raw-record logs + counterfactual evidence; arming = operator line.
- **Night shift (Addendum 5.4):** ICM gap tier-1 (designed, own gates)
  + ensemble probe (pre-registered) launching on the free cores.
- **H4: 296/500.** e2 re-baseline complete (9 rows). Blur map shards
  done (corrected proxy); aggregation report pending from the agent.
- **OQ-1 Windows brief: still awaiting operator routing.**

## Exact next actions (priority order)

1. Gate-check + commit the P2 build when the agent delivers.
2. Read ICM tier-1 + ensemble probe results against their
   pre-registered bars; log verdicts.
3. Blur report → commit + slate-2a intervention probe to refill.
4. REFILL PASS due (Addendum 4.4): candidates = H2b (buffer-continuity
   variant), slate-2a intervention probe, ReBeL P1 (zero-compute,
   gate R²≥0.5), triage-taxonomy trend report. Pre-register before
   running.
5. **Addendum 1.4 program review (PROGRAM_REVIEW.md) after the next
   completed experiment** (H1, H2 done = 2; review at 3).
6. Post-session pipeline on any live session (automatic).

## Standing rules

All of Addenda 1–5 + : file-signal watchers only; pre/post-fix
instrument tags never mix; probes from slim ckpts are a known
fragility — flag in any new registration.
