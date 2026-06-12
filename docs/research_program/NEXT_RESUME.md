# NEXT_RESUME — updated 2026-06-12 ~06:10 (H1 phase-1 + H3 complete)

**Read first:** PROGRAM.md (incl. Addenda 1–3), EXPERIMENT_LOG.md (last two
entries), OPERATOR_QUEUE.md (check stale), then this. Cross-check
`git log --oneline -10`.

## State

- **H1:** build + TG1 (byte-identity, 2 runs, final 16/16 logs) + TG2
  (exact-commit, ALL BARS PASS at τ=0.10) complete; adversarial review
  incorporated; **H1.2 re-registered** (report §4: τ=0.10, 24k paired games,
  full floor chain both arms). Full record: `reports/EXP_H1_tail_floor.md`.
  **TG3/TG4 DEFERRED until no live dry-run is active**
  (`pgrep -f run_live_dryrun` — was active since 04:20 today).
- **H3:** pipeline built, 15 sessions ingested, H4 counter **240/500
  LOCKED**. Post-session ingest line is in DRYRUN_CHECKLIST. OQ-1 filed
  (scraper showdown capture, non-blocking).
- **H2:** not started (next in queue).
- Everything committed on `track-policy` (see git log for the H1/H3 commits).

## Exact next actions (priority order)

1. **H1 TG3 harness build (runs anytime, light):** adapt
   `scripts/short_stack_floor_ab.py` per H1.2 registration — both arms run
   the full deployed floor chain via `make_live_policy_filter`, arm-1 adds
   `tail_floor_tau=0.10`, exact-commit d2c path; V0-identity check (adapted
   harness with tail OFF must reproduce the un-adapted harness byte-for-byte
   on ~100 games) BEFORE any 24k launch.
2. **H1 TG3 RUN (ONLY when `pgrep -f run_live_dryrun` is empty AND
   `tmux ls | grep dryrun` shows no live session):**
   24k paired games, seeds 1..24000, hpl=5, nice -n 19 taskset -c 0-7.
   Ship bars in H1.2 (log entry 2026-06-12 item 8). Then TG4: re-measure
   B4/B5-style batteries floors-ON+tail using
   `runs/attacker_v1/ckpt_iter_1000.pt` + `runs/attacker_bubble_v1/ckpt_iter_0600.pt`
   via `scripts/attacker_extraction_eval.py`. Log verdicts; if both pass →
   OPERATOR_QUEUE ship recommendation (arming is operator-only).
3. **H2 spec** (`H2_KILLPHIL_LEAGUE_SPEC.md`): numeric probe falsification
   thresholds BEFORE any probe (fold-vs-shove at 5–15BB toward
   killphil-optimal + panel EV holds). FIRST: audit killphil-class profiles
   for the dead `stilltoact` predicate
   (`src/nlhe/scripted_bots/policy.py:308`) — e2 in RESEARCH_MAP.
4. **After tonight's session ends:** ingest the new log
   (`python -m scripts.ingest_session logs/live_dryrun_20260612_042058.jsonl`
   + any later ones), update H4 counter in RESEARCH_MAP d1.
5. Idle-time exploratory (pre-approved, RESEARCH_MAP backlog):
   tail-concentration map (~2 h, substrate = TG2 JSON) → blur map → v2 EV
   decomposition (a3) → field CI-vs-n (d2).
6. After 3rd completed experiment (H1 counts as 1 when TG3/TG4 verdict
   lands): write PROGRAM_REVIEW.md entry per Addendum 1.4.

## Standing rules (do not drop)

nice -n 19 + taskset -c 0-7 everything; >30min evals only with no dry-run
active; no live-path arming without operator approval line; gates never
weakened; falsification criteria pre-committed BEFORE experiments; update
EXPERIMENT_LOG + this file after every step; operator boundaries → 
OPERATOR_QUEUE and continue (Addendum 2).
