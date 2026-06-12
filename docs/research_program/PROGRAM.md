# BEAT THE CHAMPION — autonomous research program

**Started:** 2026-06-12. **Role model:** research manager (Claude session) +
specialist subagents (design / implementation / measurement / adversarial
review). Program state lives in THIS directory and survives session death:

- `PROGRAM.md` — this file: objective, gates, constraints, hypothesis queue.
- `EXPERIMENT_LOG.md` — append-only. Falsification criteria logged BEFORE each
  experiment starts. Verdicts logged with evidence paths.
- `NEXT_RESUME.md` — updated after EVERY completed step. The operator pastes
  "RESUME research program" into a fresh session; that file must be sufficient.
- `H*_<NAME>_SPEC.md` — per-hypothesis specs.

## Objective

Produce a candidate that beats the deployed champion through the FULL gate
battery. Ship = all gates green. **No gate may be weakened, ever.** Operator
decides deployment.

**Champion:** `runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt`
sha256 `b79e82dd0ce9e78e4eb666b7379df953dadbf2a6e026c6bd4b6eec695e9b1b11`
(+ abstraction `0fc20800…`). Currently live behind
`scripts/run_live_dryrun.py` with Stage-2 flags.

## The gate battery (program gates — distinct from train_c3.py's pre-train G1-G4)

| Gate | Definition | Champion baseline | Bar |
|---|---|---|---|
| PG1 yardstick | CRN-paired panel (`scripts/eval_icm_panel_floor.py`), killphilmtt row | **−0.174** net/game | candidate ≥ **−0.124** |
| PG1 panel mean | same harness, 24-member panel | +0.509 ± 0.076 | ≥ +0.365; paired delta vs champion NOT significantly negative (z > −2) |
| PG2 bubble edge | bubble Δ/decision | (C3 ref bar) | > +0.003691 |
| PG3 calibration | `scripts/sng_selfplay_calibration.py`, hero all 6 seats, 2k games | −0.019 ± 0.0224 (0.85σ) | \|net/game\| < 2σ |
| PG4 bridge replay | `scripts/replay_make_decision_diff.py` over ALL raw-record dry-run logs | 0 diffs | 0 unexplained behavioral diffs; live-servable convention |
| ATT attacker | trained-BR extraction (`scripts/attacker_extraction_eval.py`), reuse `runs/attacker_v1` (1000it) + `runs/attacker_bubble_v1` (600it) | B3 −0.0035±0.0158 · B4 −0.0400±0.0158 · B5off −0.0783±0.0132 · B5on −0.0998±0.0133 | extraction no worse than champion baselines (beyond 2σ) |

Evidence anchors: `evals/c3_gate_20260611_pergame/`,
`evals/attacker_ext_20260611_pergame/attacker_ext_20260611/REPORT.md`.

## Standing constraints (verbatim intent)

1. **This box runs the live listener.** ALL research processes `nice -n 19` +
   `taskset -c 0-7` (8-core hard cap). Any eval >30 min: check
   `tmux ls | grep dryrun` AND `pgrep -f run_live_dryrun` first — if a live
   dry-run is active, DEFER heavy work.
2. **Full retrains do not run here.** When earned: write `POD_REQUEST.md`
   (config, expected hours, gates) and STOP that track. RUNPOD_SETUP.md is the
   rebuild doc; cgroup check first; tar/scp, no rsync.
3. **No live-path code changes without an explicit operator approval line.**
   Flag-gated OFF-by-default builds with byte-identity proof follow the
   Stage-2 precedent; ARMING is always an operator call.
4. Pre-committed falsification criteria in EXPERIMENT_LOG **before** each
   experiment starts. Two-benchmark minimum for measurement verdicts. Every
   commit notes validation status.
5. Budget discipline: estimated cost (hours) per experiment in its log entry;
   kill criteria honored without sentiment.

## Hypothesis queue

**Dead (falsified — may NOT re-run without new evidence):** encoder depth,
training distribution, more iterations, k=1000 abstraction.

### H1 — TAIL FLOOR (commitment-scaled) — ACTIVE
Spec: `H1_TAIL_FLOOR_SPEC.md`. Deployment-time policy filter pruning
low-probability high-commitment tail actions before sample-mode draw
(motivating incident: live session-1 seq-315, 2d5c shove drawn from a 7.5%
ALLIN tail). Approved as Stage-2 CAN-RIDE ("tail floor (policy call)",
SESSION_LOG 2026-06-11).
**Gates:** TG1 byte-identity flag-off · TG2 counterfactual at τ ∈
{0.05, 0.10, 0.15} · TG3 2k-game CRN-paired panel floor-on vs floor-off ·
TG4 attacker re-check. **Evidence decides; no paired EV gain ⇒ no ship.**
**Est cost:** build+TG1+TG2 ≈ 4–6 h (runs now, light); TG3 ≈ 4–8 h CPU
(DEFERRED while dry-run active); TG4 ≈ 6–10 h CPU (measurement only — reuses
trained attackers; DEFERRED likewise).

### H2 — KILLPHIL-LEAGUE PROBE — QUEUED (design may start; probe after spec)
Hypothesis: self-play monoculture never taught shove-defense (killphilmtt
−0.174 is the champion's worst panel row). Design a league mix: champion-style
self-play + scripted exploiter styles from `data/shanky_profiles` (incl.
killphil-class shove bots) + checkpointed past selves, injected via the
existing league/archetype override path (`configs/league/`).
**Pre-committed probe BEFORE any full retrain:** short CPU-sized run (500
iters, low G, here) measuring ONE thing — does fold-vs-shove frequency at
5–15 BB move toward the killphil-optimal response while self-play panel EV
holds? Numeric falsification thresholds fixed in H2 spec BEFORE launch.
Probe pass ⇒ POD_REQUEST.md for full retrain, track STOPS for operator.
**Known hazard (from SNG-field-v2 work):** `stilltoact` hardcoded 0 in
`src/nlhe/scripted_bots/policy.py:308` — all `stilltoact>=k` Shanky rules are
dead. Audit killphill-class profiles for dependence on it before trusting the
league mix.
**Est cost:** design+spec 2 h; probe ≈ 6–10 h CPU + 2 h measurement harness.

### H3 — OPPONENT-DATA PIPELINE — ACTIVE (independent, runs regardless)
Extractor parsing ALL live session jsonl (`logs/live_dryrun_*.jsonl`) into a
structured hand/opponent database: actions by seat/street/stack-depth,
showdown holdings, timing. Deliverables: growing sqlite+parquet archive at
`data/opponent_db/`; per-session ingest command added to
`docs/DRYRUN_CHECKLIST.md` post-session list; field-statistics report
(VPIP / fold-to-shove / showdown tendencies). Substrate for H4.
**Est cost:** build 3–5 h; per-session ingest minutes.

### H4 — RNR FIELD-EXPLOITATION RETRAIN — LOCKED
Unlocks when the H3 archive holds ≥ 500 opponent-observed hands. Restricted
Nash response vs measured field tendencies, anchored to champion. Design doc
may be drafted early; NO training until unlock + probe + operator pod
approval.

## Cadence

End every working block: update EXPERIMENT_LOG (what ran, verdict, next) and
NEXT_RESUME.md (exact next action). Adversarial-review subagent attacks every
conclusion before it enters the log.

## ADDENDUM 1 (2026-06-12) — Scientific rigor & reporting standard (permanent)

1. **Full experiment reports, not summaries.** Every experiment produces
   `reports/EXP_<id>_<name>.md` with: Background & motivation (prior
   evidence), Pre-registered design (hypothesis, method, falsification
   criteria, cost — written BEFORE running), Methods (exact configs, seeds,
   commands — reproducible by a stranger), Results (full tables, CIs on
   everything, raw-data paths), Discussion (what it means, what it does NOT
   mean, threats to validity, adversarial-reviewer objections + disposition),
   Future Work (specific follow-ups with cost + expected information value).
2. **Negative results are first-class.** A falsified hypothesis gets the same
   full report. The program's value is the map, not just the treasure.
3. **`RESEARCH_MAP.md`** — living document, every open question by pillar:
   (a) representation, (b) training dynamics, (c) the tournament game itself,
   (d) opponent modeling & exploitation, (e) measurement methodology,
   (f) frontier/proprietary. Per question: evidence state, decisive
   experiment, cost, expected value of information. Update after every
   experiment.
4. **Periodic synthesis:** after every 3 completed experiments or weekly
   (whichever first), append to `PROGRAM_REVIEW.md` — what the evidence now
   says, how priorities changed, what a well-resourced team would do next.
5. **Moderate-depth exploratory studies licensed:** cheap read-only analyses
   of existing artifacts (tail-concentration map, field-tendency pre-analysis,
   bucket decision-entropy "blur map", etc.). Mini-report each; feeds the
   RESEARCH_MAP. Fills idle time between gated experiments, never displaces
   them.

## ADDENDUM 2 (2026-06-12) — Non-blocking operation (overnight rule, permanent)

Operator gates halt one TRACK, never the program. At any
decision/approval/pod boundary: write the request to `OPERATOR_QUEUE.md`
(decision needed, evidence, recommendation, what it unblocks), mark the track
BLOCKED-ON-OPERATOR in the log, and IMMEDIATELY continue with next-highest-
value unblocked work. Priority cascade when blocked: next queue hypothesis →
H3 pipeline work → exploratory studies from the RESEARCH_MAP (blur map, tail
concentration, field-tendency pre-analysis, decision-entropy map are
pre-approved) → RESEARCH_MAP / synthesis writing. There is ALWAYS legal work;
an idle program is a bug. Nothing deploys, arms, or trains-on-pod without the
operator — but nothing waits on the operator either. Check OPERATOR_QUEUE.md
for stale entries at every resume.

## ADDENDUM 3 (2026-06-12) — Frontier research license (permanent)

The program may pursue NOVEL proprietary architectures, training processes,
and hybrid systems (style-specialist ensembles + selector/mixer, hybrid
blueprint+override-region policies, distillation chains, tournament-value
architectures, "frankenstein" compositions of validated artifacts — champion,
v2's depth-cured bbnorm encoder, the trained attackers, the 31 scripted
styles, league machinery, the live archive; failed candidates' components are
organ banks). Spine:
1. **Evidence-first:** a frontier idea enters the queue only with a written
   rationale citing OUR measurements; adversarial reviewer attacks the
   rationale before any build.
2. **Probe-before-program:** cheapest-possible falsification probe (hours,
   not days) before real investment. Most frontier ideas should die at probe
   stage — that is the system working.
3. **Same gates forever:** full battery (PG1–PG4 + attacker), no novelty
   discounts; hybrids/ensembles additionally must meet live-path constraints
   (sub-second single-decision latency, deterministic replay, floor
   compatibility) or they are research results, not candidates.
4. Logged in RESEARCH_MAP pillar (f) with the full-report standard.
Frontier work fills idle/exploratory cycles on expected-information-value;
it never displaces a gated experiment mid-run.

## ADDENDUM 4 (2026-06-12) — Perpetual operation (permanent)

Operator directive, recorded verbatim in intent:

1. **Never end a turn while unblocked work exists.** After completing any
   item, immediately select the next per the priority cascade (gated
   queue → specs → exploratory → synthesis → RESEARCH_MAP grooming) and
   continue. "Report and stop" is a violation unless EVERY track is
   BLOCKED-ON-OPERATOR.
2. **Long computations launch in their own tmux sessions** with
   completion-watchers that re-invoke the manager; while one cooks, work
   the next track — never idle-wait on a run.
3. **Context self-management:** when context grows long, proactively
   write a complete handoff (NEXT_RESUME.md + a one-paragraph
   SESSION_HANDOFF note) BEFORE quality degrades, then continue — the
   handoff is insurance, not a stopping point.
4. **The queue never empties:** whenever fewer than 3 actionable items
   remain, run a PIPELINE REFILL pass — mine the RESEARCH_MAP, the
   latest live-session data, and all experiment reports for the
   next-highest expected-information-value experiments; pre-register
   their falsification criteria; add them. Synthesizing
   novel/proprietary directions per Addendum 3 is part of refill.
5. **Every completed experiment triggers:** full report, RESEARCH_MAP
   update, refill check, and a one-line OPERATOR_QUEUE status note so
   the operator's morning read always reflects reality.
6. **Post-session auto-pipeline (added 2026-06-12):** when a live
   dry-run session ends (the dryrun tmux session disappears), auto-run
   the post-session pipeline if the operator hasn't: `dryrun_triage` +
   `ingest_session` on the new log, update the H4 counter (RESEARCH_MAP
   d1), and note anomalies in EXPERIMENT_LOG — specifically tail-floor
   firing rates vs the expected band (distribution adjusted ~70–75% of
   decisions; sampled-action changes ~4.4% per TG2;
   most-argmaxes-changed is anomalous). Implemented in-session as a
   persistent tmux watcher that re-invokes the manager on the
   appear→disappear transition.

## ADDENDUM 5 (2026-06-12) — Laboratory intensification: single-site max utilization (permanent)

Operator directive, recorded in intent:
1. **Core ledger:** the active heavy job owns its taskset; every
   remaining core runs something at all times. Idle cores are a bug.
   Live-play deference unchanged (full stand-down when a listener is up).
2. **Analysis-heavy parallelism:** background specialist agents spawn
   freely for file-only work (no core cost). Standing slate (a-f:
   blur-map mining, bbnorm-transplant spec, ensemble-probe design,
   field dossier, ICM-gap study design, ReBeL-class design memo); each
   produces a mini-report into RESEARCH_MAP; refill per Addendum 4.4
   when fewer than 3 remain.
3. **COMPUTE_QUEUE.md:** every runnable experiment pre-registered,
   sorted by EVoI per core-hour; next job launches within minutes of
   any heavy-job completion, watcher-armed. H2 chain keeps priority.
4. **Night shift:** overnight (no operator, no live play) is 100%
   compute — longest queued items run then automatically.
5. **POD_CASE:** a one-line standing entry in OPERATOR_QUEUE stating
   what a 27-core pod would currently unlock and its time savings.
   If the H2 probe PASSES: full league retrain carrying the bbnorm
   organ — ~3 h pod vs ~28 h here.
