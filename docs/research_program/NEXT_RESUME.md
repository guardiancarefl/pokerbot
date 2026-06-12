# NEXT_RESUME — updated 2026-06-12 ~04:30 (program init)

**Read first:** PROGRAM.md (gates/constraints/queue), EXPERIMENT_LOG.md (last
entry), then this. Cross-check `git log --oneline -10` per CLAUDE.md.

## State right now

- Program initialized. H1 spec frozen (`H1_TAIL_FLOOR_SPEC.md`), falsification
  criteria pre-committed in EXPERIMENT_LOG.
- **Live dry-run ACTIVE** as of 04:20 (`pgrep -f run_live_dryrun` to re-check)
  ⇒ H1 TG3/TG4 and any >30min eval DEFERRED until it ends.
- H1 build: implementation subagent dispatched (floor + flag plumbing + tests
  + TG2 counterfactual script). NOT yet committed.
- H3 build: background subagent dispatched (extractor + sqlite/parquet archive
  + ingest command + field-stats report). NOT yet committed.

## Exact next actions (in order)

1. If H1 build finished: adversarial-review it, run tests
   (`source .venv/bin/activate && python -m pytest tests/test_tail_floor.py -v`),
   then **TG1**: `nice -n 19 taskset -c 0-7 python scripts/replay_make_decision_diff.py`
   over ALL raw-record dry-run logs (flag OFF) — bar: 0 behavioral diffs.
2. **TG2**: `nice -n 19 taskset -c 0-7 python scripts/tail_floor_counterfactual.py`
   (all logs, τ_max ∈ {0.05,0.10,0.15}) → evidence to
   `evals/h1_tail_floor_20260612/`. Check F-TG2 bars (seq-315 caught at
   0.10/0.15; zero argmax changes; ≤5% altered at 0.10). Fix τ_max from TG2.
3. Log TG1/TG2 verdicts in EXPERIMENT_LOG; commit H1 build + evidence
   (validation status in commit message); update this file.
4. **When no dry-run active** (check first!): TG3 2k-game CRN-paired panel
   floor-on(τ from TG2) vs floor-off, champion ckpt both arms, nice/taskset
   — then TG4 attacker re-measure reusing `runs/attacker_v1/ckpt_iter_1000.pt`
   + `runs/attacker_bubble_v1/ckpt_iter_0600.pt` (no retraining). Ship bar in
   EXPERIMENT_LOG F-TG3/F-TG4.
5. H3: review extractor output reconciliation (session-1 = 37 hands / 61
   decision frames known cross-check), add ingest line to
   `docs/DRYRUN_CHECKLIST.md`, ingest ALL logs incl. tonight's
   `live_dryrun_20260612_042058.jsonl` after the session ends, report H4
   unlock counter (bar: ≥500 opponent-observed hands).
6. Then H2: write `H2_KILLPHIL_LEAGUE_SPEC.md` with NUMERIC probe
   falsification thresholds (fold-vs-shove at 5–15BB toward killphil-optimal;
   panel EV holds) BEFORE launching the 500-iter low-G probe. Audit
   killphil-class Shanky profiles for the dead `stilltoact` predicate
   (`src/nlhe/scripted_bots/policy.py:308`) first.

## Standing rules (do not drop)

nice -n 19 + taskset -c 0-7 everything; >30min evals only with no dry-run
active; no live-path arming without operator approval line; gates never
weakened; update EXPERIMENT_LOG + this file after every step.
