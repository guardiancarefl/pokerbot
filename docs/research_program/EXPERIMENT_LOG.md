# EXPERIMENT LOG — append-only

Falsification criteria are logged BEFORE an experiment starts. Verdicts cite
evidence paths. Two-benchmark minimum for measurement verdicts.

---

## 2026-06-12 — PROGRAM INIT

- Program files created. Champion = b79e82dd (`ckpt_iter_1500.pt`,
  k200_real_ante). Gate battery + attacker baselines recovered from
  `evals/c3_gate_20260611_pergame/` and
  `evals/attacker_ext_20260611_pergame/attacker_ext_20260611/REPORT.md`;
  tabulated in PROGRAM.md.
- Finding: no on-disk spec existed for the commitment-scaled tail floor (only
  the approved CAN-RIDE line, SESSION_LOG 2026-06-11 + tail-accounting
  evidence). Spec reconstructed and frozen at `H1_TAIL_FLOOR_SPEC.md`.
- Environment: live dry-run ACTIVE (run_live_dryrun.py since 04:20,
  `logs/live_dryrun_20260612_042058.jsonl`, Stage-2 flags armed) ⇒ all >30min
  evals (H1 TG3/TG4) DEFERRED until it ends.

## 2026-06-12 — H1 TAIL FLOOR — falsification criteria (PRE-COMMITTED, before build)

Estimated cost: build+TG1+TG2 ≈ 4–6 h now; TG3 ≈ 4–8 h CPU deferred;
TG4 ≈ 6–10 h CPU deferred (measurement-only, reuses trained attackers).

- **F-TG1:** any behavioral diff with flag OFF across ALL raw-record dry-run
  logs ⇒ implementation rejected (fix or abandon; gate not weakened).
- **F-TG2:** (a) the session-1 seq-315 7.5% ALLIN tail must be caught at
  τ_max ∈ {0.10, 0.15}; (b) zero argmax changes at any τ; (c) altered-decision
  rate at τ_max=0.10 ≤ 5% of decision frames. Any failure ⇒ STOP before TG3.
- **F-TG3 (ship bar):** paired all-games delta z > −2 AND diverged-games
  per-firing delta > 0 with z ≥ 2, at the single τ_max fixed by TG2 BEFORE
  launch. No EV gain ⇒ does not ship. No seed re-rolls, no post-hoc τ shopping.
- **F-TG4:** extraction with tail floor armed worse than champion B4/B5
  floors-ON baselines by > 2σ ⇒ kill regardless of TG3.

Status: build starting (implementation subagent). Verdict: pending.

## 2026-06-12 — H3 OPPONENT-DATA PIPELINE — start

Estimated cost: 3–5 h build. Not an EV experiment — falsification n/a;
acceptance = extractor round-trips ALL existing live session logs with hand /
action / showdown counts that reconcile against the audited postmortems
(session-1: 37 hands / 61 decision frames is the known cross-check), ingest
command is idempotent (re-ingesting a session does not duplicate rows), and
the field-stats report runs. H4 unlock counter = opponent-observed hands
(hands with ≥1 non-hero action observed), threshold ≥ 500.

Status: build starting (background subagent). Verdict: pending.

## 2026-06-12 — H1 TG1+TG2 COMPLETE (incl. adversarial review + instrument fix)

Full report: `reports/EXP_H1_tail_floor.md`. Chain of verdicts:

1. **Build** landed (floor + flag plumbing + 18 tests + TG2 instrument);
   regression suites green (57 floor/bridge tests; header test extended for
   the new `tail_floor_tau` header key).
2. **TG1 #1 PASS:** 5,369 frames / 11 frame-bearing logs byte-identical
   (HEAD-with-diff vs 28ce614 worktree, cmp).
3. **TG2 #1:** bars (a),(b) PASS; **(c) FAIL — 5.10% (clean 5.30%) > 5%**
   ⇒ STOP honored, TG3 not launched under first registration.
4. **Adversarial review** (verdicts recorded in full in the report):
   A1 metric operationalization-dependent (TV 4.36% / hist-draw 5.10% /
   realized 9.45% at τ=0.10) — all reported; A2 instrument bug UPHELD
   (commit formula understated pot; 28/1,579 bets misclassified dangerous-
   direction); A4: τ→0.05 amendment REJECTED, "proceed at ~5%" REJECTED;
   A5/A6 attacks on TG1 and instrument REJECTED. Prescription: fix commit
   semantics, re-run gates, re-register on corrected numbers.
5. **Fix adopted:** exact chip-map commitment via `discrete_to_chip`
   pass-through (`accepts_d2c` marker; `eval_6max_self_play.py` dispatch).
   23 tail-floor tests green.
6. **TG1 #2 PASS** on the final diff (16/16 logs identical).
7. **TG2 #2 (exact commit): ALL BARS PASS at τ=0.10** — altered 4.37%
   (clean 4.64%), seq-315 caught, 0 argmax changes.
8. **H1.2 re-registered BEFORE TG3** (report §4): τ=0.10, 24k paired games,
   full floor chain both arms, seeds 1..24000 fixed, ship bars unchanged,
   narrowed self-play claim + stale-attacker TG4 caveat recorded.

**Verdict: H1 build phase COMPLETE; TG3/TG4 pending a dry-run-free window**
(live session active all night). Next H1 action: adapt the A/B harness per
H1.2 (build + V0-identity check can run anytime; the 24k eval cannot).
Evidence root: `evals/h1_tail_floor_20260612/`.

## 2026-06-12 — H3 PIPELINE BUILT + ALL LOGS INGESTED

Acceptance met: session-1 reconciliation EXACT (37 hands / 61 decision
frames); idempotent sha-keyed ingest; 8/8 tests. 15 sessions / 432 hands /
1,466 actions / 506 opponent voluntary actions. **H4 unlock: 240/500 —
LOCKED.** Structural findings (RESEARCH_MAP d3, OPERATOR_QUEUE OQ-1):
opponent hole cards unobservable in the scraper schema; stale folded flags
⇒ fold-vs-shove undercounted. Ingest command added to DRYRUN_CHECKLIST.
Artifacts: `src/nlhe/opponent_db/`, `scripts/ingest_session.py`,
`data/opponent_db/` (sqlite + FIELD_REPORT.txt).

## 2026-06-12 — H1 TG3 HARNESS BUILT + GATED (run deferred)

`scripts/tail_floor_ab.py`: deployed floor chain BOTH arms via
make_live_policy_filter, tail floor (tau=0.10, exact-commit d2c path,
dispatch verified 46/46) the only delta. V0-identity gate PASS (100 games,
tail off both arms: all deltas exactly 0, 0 diverged); shard-split
reproduces unsharded records exactly. Smoke 200 paired games: 0.52 s/pair
(24k ~ 3.5 h single-process / ~1 h 8-way sharded), tail fired 73.9% of hero
decisions (consistent w/ TG2 71.1%), 161/200 games diverged; direction
positive (+0.200 +/- 0.076 all-games) — RECORDED ONLY, underpowered, per
H1.2 no conclusion before the registered 24k run. Evidence:
evals/h1_tail_floor_20260612/tg3_harness_gates.txt. 24k launch awaits a
dry-run-free window; command in NEXT_RESUME.

## 2026-06-12 ~11:25 — H1 TG3 24k LAUNCHED + night-session ingest

- Dry-run gate VERIFIED CLEAR before launch: `pgrep -f run_live_dryrun`
  empty (only the checking shell matched), no dryrun tmux session, box
  idle (15-min load 0.15).
- **TG3 24k launched** per H1.2 registration: 8-way sharded
  `scripts.tail_floor_ab --games 24000 --base-seed 1 --hpl 5 --tau 0.10
  --shards 8 --shard K`, nice 19, one core per shard (0-7). Ckpt/abstr
  shas confirmed in every shard log (b79e82dd… / 0fc20800…). Rate at
  launch ~1.7 g/s/shard, ETA ~30 min. Out:
  `evals/h1_tail_floor_20260612/tg3_24k_shard{0..7}.json[.games.jsonl]`
  (+ .log). Aggregation = concat games.jsonl (shard-invariance gate in
  tg3_harness_gates.txt).
- **H3 ingest (NEXT_RESUME item 4):** `live_dryrun_20260612_042058.jsonl`
  ingested NEW (35 hands / 106 actions / 36 opp-voluntary / 19
  opp-observed / 1 showdown); `035454` re-ingest idempotent-replaced
  (already counted). **H4 unlock counter 240 → 259/500 — still LOCKED.**
  RESEARCH_MAP d1 updated.

Verdict: pending (TG3 ship bars: all-games z > −2 AND diverged-only
per-firing > 0 with z ≥ 2 at τ=0.10).

## 2026-06-12 ~12:00 — OQ-1 RESOLVED → WINDOWS_TASK brief written

Operator approved OQ-1 with expanded scope and raised priority (folded
flags first; showdown capture as specced). Delivered:

- `docs/WINDOWS_TASK_SCRAPER_FOLDED_SHOWDOWN.md` — forensics-grade brief
  for the Windows CC session: PNG-evidence-first (A-1), fix-proposal
  review (A-2), offline replay proof over >=2 archived sessions with a
  ZERO-new-false-positive hard bar (A-3 — false `folded=true` is
  live-path dangerous because `_repair_folded_from_chip_deductions`
  trusts pre-existing flags); showdown capture as optional
  `shown_cards` + `schema_version: 2` (B); Contabo gates pre-committed
  (G1 replay+injection bridge-untouched, G2 ingester behind
  parser_version bump w/ FIELD_REPORT idempotence, G3 first-live-session
  stale rate <= 5%).
- `scripts/extract_stale_folded_candidates.py` +
  `tools/scraper_folded_showdown_task/stale_folded_candidates.jsonl`:
  **new forensic headline — 0/241 chip-proven non-blind preflop folds
  (postflop-reaching, outcome-known hands) ever got a `folded`
  transition (100% stale per fold event; whole corpus has 16
  transitions across 432 hands).** The remembered ~30% was a
  per-frame, different-denominator measure.
- OPERATOR_QUEUE OQ-1 moved to Resolved; RESEARCH_MAP d3 updated.

Not an EV experiment — falsification n/a; acceptance = the Windows-side
gates in the brief, pre-committed before any scraper change.

## 2026-06-12 ~12:40 — TG4 instrument RECOVERED + GATED (commit 6351dae)

Resume-blocking find: the TG4 instrument (attacker eval script, both
attacker ckpts, bubble artifact) was absent from the working tree — built
on the unmerged runpod-env branch. Recovered: ckpts in
`mirrors/tier0_20260611/runs/` (shas match the archived shard logs),
script + sng_baseline bubble seam extracted from the bundle, bubble4
artifact regenerated (96,749 rows). `--tail-tau` added (the TG4 delta).
Gates ALL PASS: B4 + B5 shard00 repro at HEAD byte-identical to the
2026-06-11 archives (200/200 each, tail code present at tau=None);
tg4_verdict.py self-test paired delta exactly 0 both modes. Full record:
`evals/h1_tail_floor_20260612/tg4/tg4_instrument_gates.txt`. F-TG4 kill
bar unchanged. TG4 launches after the TG3 verdict.

## 2026-06-12 ~12:55 — H1 TG3 24k VERDICT: PASS (both ship bars, z=16.8)

Registered run completed exactly as specified (24,000 paired games, seeds
1..24000, tau=0.10, full deployed floor chain both arms, 8 shards, 28.7 m
wall). Aggregation coverage gate: 24000/24000 games exactly once.

- **All-games paired ICM delta V1-V0 = +0.1227 +/- 0.0073 (z = +16.76)**
  — bar was z > -2 (non-inferiority); measured massively positive.
- **Diverged-only = +0.1597 +/- 0.0095 (z = +16.78, n = 18,436)** — bar
  was mean > 0 with z >= 2.
- Divergence 76.8% of games; tail fired 74.0% of 473,915 hero decisions
  (TG2 71.1%, smoke 73.9% — instrument consistent).
- vs the smoke's +0.200 +/- 0.076: the 24k point sits 1.0 sigma below the
  smoke point — consistent, the smoke was simply underpowered as recorded.

**F-TG3: PASS. No seed re-rolls, no tau shopping (single pre-registered
tau=0.10).** Evidence: `evals/h1_tail_floor_20260612/tg3_24k.json`
(+ .games.jsonl, 8 shard files, logs). Remaining before ship
recommendation: TG4 (kill bar F-TG4).

## 2026-06-12 ~13:15 — H1 TG4 VERDICT: PASS both batteries → H1 EXPERIMENT COMPLETE

Batteries (4,000 games each, 0 tainted, seeds = the archived B4/B5
schedule, champion floors ON + tail tau=0.10; attacker ckpts sha-matched
to the archived instrument):

- **TG4-standard vs B4 (-0.0400 +/- 0.0158):** extraction
  **-0.1405 +/- 0.0157** — the attacker extracts LESS with the tail
  armed; unpaired z = -4.52, CRN-paired delta -0.1005 +/- 0.0210
  (z = -4.78, 2219/4000 games identical). The tail floor INCREASED
  adversarial robustness in the standard game.
- **TG4-bubble vs B5on (-0.0998 +/- 0.0133):** extraction
  **-0.1293 +/- 0.0136**; unpaired z = -1.55, paired -0.0295 +/- 0.0151
  (z = -1.96, 3089/4000 identical). Robustness direction, not
  individually significant — and nowhere near the kill region.

**F-TG4 kill bar: NOT triggered** ("extraction worse than the B4/B5
floors-ON baselines by > 2 sigma" = worse FOR THE CHAMPION = attacker
extraction HIGHER by > 2 sigma; both batteries moved the OTHER way).

TOOLING DISCLOSURE (gates unchanged, tooling corrected): the first run
of scripts/tg4_verdict.py had the kill-bar SIGN inverted (flagged
attacker-does-WORSE as a kill). Caught on direction review against the
registered F-TG4 wording + the B-report's "armor" framing before any
verdict was logged; fixed in the script (kill = z > +2), both verdicts
re-emitted. The registered criterion text was never edited. Numbers
were identical in both runs; only the verdict label changed.

**EXP_H1 chain: TG1 PASS, TG2 PASS (exact-commit), TG3 PASS (z=16.8),
TG4 PASS ⇒ ship recommendation filed (OQ-2). Arming is operator-only;
flag stays OFF until the operator arms `--tail-floor-tau 0.10`.**
Evidence: `evals/h1_tail_floor_20260612/tg4/` (+ tg3_24k.json).

## 2026-06-12 ~13:40 — H2 pre-work: e2 stilltoact audit (killphil half) DONE

KillPhilMTT uses `stilltoact` in 17/629 rules — position-tiered
unopened-push ranges. With the adapter's hardcoded 0: the 9 `>=k` + 3
`=k` early/mid-position (tighter) tiers are dead; the 5 `<=1`
late-position (loosest) tiers fire from every seat. Adapter-killphil
therefore shoves looser than real killphil in unopened pots. Verdict
for H2: the spec MUST resolve this before defining "killphil-optimal"
— either fix stilltoact (derive yet-to-act count from the live view)
and re-baseline, or pre-commit the probe to adapter-as-played
semantics. Details + field-wide usage ranking in RESEARCH_MAP e2.

## 2026-06-12 ~13:55 — OQ-2 RESOLVED: tail floor ARMED as standard config

Operator approval line received ("Arm --tail-floor-tau 0.10 in the
dry-run checklist's listener line as standard config"). DRYRUN_CHECKLIST
listener line updated (+ ARMED-banner check + expected firing magnitudes
+ post-session triage step). OQ-2 → Resolved. **H1 is now DEPLOYED
pending the next live session; first armed-session observations to be
logged here under H1.** No code change — the flag was built, gated, and
OFF since c9a2b14; this is configuration only.
