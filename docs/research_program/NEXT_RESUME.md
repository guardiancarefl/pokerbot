# NEXT_RESUME — updated 2026-06-12 ~14:30 (H1 DEPLOYED-config; H2 spec drafted; e2 fix in)

**Read first:** PROGRAM.md (incl. Addenda 1–3), EXPERIMENT_LOG.md (last
three entries), OPERATOR_QUEUE.md (both items Resolved), then this. Cross-check `git log --oneline -10`.

## State

- **H1: EXPERIMENT COMPLETE — TG1 ✓ TG2 ✓ TG3 ✓ TG4 ✓.** TG3 24k:
  all-games +0.1227 ± 0.0073 (z=16.8). TG4: kill bar not triggered;
  standard-game robustness IMPROVED 0.10/game vs B4 (paired z=−4.8).
  Full record: `reports/EXP_H1_tail_floor.md` §7. **OQ-2 APPROVED
  2026-06-12: `--tail-floor-tau 0.10` is STANDARD CONFIG in the
  DRYRUN_CHECKLIST listener line.**
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

## SESSION_HANDOFF (~15:00 — context insurance per Addendum 4.3)

Three tracks in flight, each with a watcher/agent that re-invokes the
manager: (1) e2 re-baseline, 7 profiles, tmux `e2_rebaseline`, watcher
on *.DONE count; (2) H2 fold-vs-shove battery freeze, tmux `h2_battery`,
watcher on MAKE.DONE — output evals/h2_battery/battery_v1.json; (3)
tail-concentration map exploratory via background agent (writes
evals/h1_tail_floor_20260612/tail_concentration_map.{txt,json}; manager
commits). On resume after any landing: fill H2 spec §3 from the
re-baseline (incl. tighttom-vs-trickytom divergence check + new
killphil gate row), run champion M1 grade
(`fold_vs_shove_battery grade --ckpt <champion>`), FREEZE the spec, then
launch the H2 probe per spec §4 (benchmark 1 iter first). Addendum 4 is
now in PROGRAM.md — perpetual operation, queue never empties.

## In flight RIGHT NOW (2026-06-12 ~14:30)

- **e2 ADAPTER FIX LANDED (a8933ea):** stilltoact derived (PPL preflop
  semantics) + preflop BB-is-live-bet counter fix (limps were checks,
  opens were bets). 172 tests green. **7-profile re-baseline RUNNING**
  in tmux `e2_rebaseline` → `evals/e2_rebaseline_20260612/` (killphil
  gate row + heavy stilltoact users + tighttom/trickytom divergence
  check); watcher armed. Old gate row −0.1740 ± 0.0220.
- **H2 spec DRAFTED** (`H2_KILLPHIL_LEAGUE_SPEC.md`): thresholds
  pre-committed (F-M1 ≥25% EV-loss drop, F-M2a killphil +0.05 @2σ,
  F-M2b panel holds); §3 baselines TBF from the re-baseline → THEN
  freeze. Next build: `scripts/fold_vs_shove_battery.py` (harness +
  frozen oracle battery) per spec §2.

## Exact next actions (priority order)

1. **Fill H2 spec §3 from the re-baseline, freeze, build the
   fold-vs-shove harness, run champion M1 baseline, then launch the
   probe** (tmux + watcher; benchmark 1 iter first).
2. **After any live session:** ingest the log
   (`python -m scripts.ingest_session logs/live_dryrun_<TS>.jsonl`),
   update H4 counter in RESEARCH_MAP d1. The tail floor is ARMED (OQ-2 approved): audit `[FLOOR] fired=[tail]` lines in triage and log
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

## Operational lesson (2026-06-12 ~15:10)

Background watchers CANNOT reliably see other processes (ps/pgrep view
is inconsistent in detached tasks — two blur watchers false-fired, two
listener Monitors died). **Watchers must key on FILE signals only**:
.DONE flags, log completion lines, output file existence. The probe and
b2 watchers are file-based (safe). Listener detection: the tmux-based
Monitor (tmux ls works) + checklist convention that listeners run in
tmux `dryrun`.
