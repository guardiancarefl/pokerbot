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
