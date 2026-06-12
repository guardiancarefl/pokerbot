# NEXT_RESUME — updated 2026-06-12 ~13:30 (H1 COMPLETE; OQ-2 ship rec filed)

**Read first:** PROGRAM.md (incl. Addenda 1–3), EXPERIMENT_LOG.md (last
three entries), OPERATOR_QUEUE.md (OQ-2 OPEN — ship recommendation),
then this. Cross-check `git log --oneline -10`.

## State

- **H1: EXPERIMENT COMPLETE — TG1 ✓ TG2 ✓ TG3 ✓ TG4 ✓.** TG3 24k:
  all-games +0.1227 ± 0.0073 (z=16.8). TG4: kill bar not triggered;
  standard-game robustness IMPROVED 0.10/game vs B4 (paired z=−4.8).
  Full record: `reports/EXP_H1_tail_floor.md` §7. **Ship rec = OQ-2;
  arming (`--tail-floor-tau 0.10`) is operator-only; flag stays OFF.**
  Registered caveats stand: self-play (not live-field) evidence;
  stale-attacker TG4 (fresh attacker retrain = pod-class, only if it
  matters post-arming).
- **TG4 instrument recovered** (was runpod-env-branch-only): ckpts at
  `mirrors/tier0_20260611/runs/` (sha-verified), script + sng_baseline
  bubble seam extracted from the bundle, bubble4 artifact regenerated +
  byte-identity-proven. `evals/h1_tail_floor_20260612/tg4/`.
- **OQ-1 RESOLVED** (operator, 2026-06-12): Windows brief written —
  `docs/WINDOWS_TASK_SCRAPER_FOLDED_SHOWDOWN.md` (folded-flag fix
  prioritized; showdown capture; bridge-untouched gates pre-committed).
  Awaiting operator routing to a Windows CC session. New forensic
  headline: folded flag ~100% stale per fold event (0/241).
- **H3:** pipeline live; **H4 counter 259/500 LOCKED** (16 sessions /
  467 hands ingested incl. both 2026-06-12 night sessions).
- **H2:** not started — NOW NEXT IN QUEUE.

## Exact next actions (priority order)

1. **H2 spec** (`H2_KILLPHIL_LEAGUE_SPEC.md`): numeric probe
   falsification thresholds BEFORE any probe (fold-vs-shove at 5–15BB
   toward killphil-optimal + panel EV holds). FIRST: audit
   killphil-class profiles for the dead `stilltoact` predicate
   (`src/nlhe/scripted_bots/policy.py:308`) — e2 in RESEARCH_MAP.
2. **After any live session:** ingest the log
   (`python -m scripts.ingest_session logs/live_dryrun_<TS>.jsonl`),
   update H4 counter in RESEARCH_MAP d1. If the operator armed the tail
   floor (OQ-2), audit `[FLOOR] fired=[tail]` lines in triage and log
   first-live-session observations under H1 in EXPERIMENT_LOG.
3. **When Windows CC delivers the OQ-1 scraper patch:** run the
   pre-committed Contabo gates (WINDOWS_TASK doc §5: G1 replay +
   injection, G2 ingester/schema_version, G3 first-live-session stale
   rate ≤5%).
4. Idle-time exploratory (pre-approved, RESEARCH_MAP backlog):
   tail-concentration map (~2 h, substrate = TG2 JSON) → blur map → v2
   EV decomposition (a3) → field CI-vs-n (d2).
5. **After the 3rd completed experiment** (H1 is #1; H3 was
   infrastructure, not an EV experiment — judgement call recorded:
   review triggers after H2's verdict): write PROGRAM_REVIEW.md per
   Addendum 1.4.

## Standing rules (do not drop)

nice -n 19 + taskset -c 0-7 everything; >30min evals only with no dry-run
active (`pgrep -f run_live_dryrun` + `tmux ls | grep dryrun` BOTH clear);
no live-path arming without operator approval line; gates never weakened;
falsification criteria pre-committed BEFORE experiments; update
EXPERIMENT_LOG + this file after every step; operator boundaries →
OPERATOR_QUEUE and continue (Addendum 2).
