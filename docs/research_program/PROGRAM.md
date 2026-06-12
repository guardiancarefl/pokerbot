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
