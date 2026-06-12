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

## 2026-06-12 ~15:10 — H2 battery FROZEN + champion M1 baseline; tail-concentration map (exploratory)

- **Battery frozen:** `evals/h2_battery/battery_v1.json` — 8,112 spots
  (24 nonempty range cells), oracle gradients sane (BB>SB>BTN call
  rates; 5bb 17-18% vs 8bb+ ~5-6%; killphil ranges 73/169 @5bb,
  45/169 @8bb+ through the POST-FIX adapter). Double-up equal-stack ICM
  is bubble-harsh: overall oracle call rate 7.7%.
- **Champion M1 = 0.0867 ± 0.0007 ICM-loss/spot.** Diagnostic: champion
  call mass 35.9% vs oracle 7.7% — the shove-defense gap is dominantly
  OVER-CALLING. F-M1 bar for the probe: M1 ≤ 0.0650 (-25% relative).
- **Tail-concentration map (RESEARCH_MAP backlog #1) DONE** (subagent):
  83% of pruned mass preflop; ALLIN 46% + big bets 33% of pruned mass;
  firing falls monotonically with commitment (77.8% at 0-5% committed →
  25.0% at 30%+) — the commitment scaling behaves as designed; near
  silent <6bb where the short-stack floor owns the space. 57
  seq-315-class spots cataloged. Substrate caveat: per-decision
  context regenerated via the instrument's replay path, census-matched
  to TG2 exactly. `evals/h1_tail_floor_20260612/tail_concentration_map.txt`.

## 2026-06-12 ~16:20 — d2 CI-vs-n (exploratory) DONE; H2 probe benchmark GREEN

- **d2 (RESEARCH_MAP):** field VPIP lower bound 20.1% ± 1.9pp (Wilson) —
  already decision-grade; 5-15BB field-pooled 20.4% ± 4.1pp. Per-opponent
  reads: 51% of opponent-keys end a session below ±15pp but only from
  hand ~26-39 — **per-opponent precision is capped by session length,
  not archive size; field-pooling is the only viable estimator for
  H2's regime.** Fold-vs-shove 0/13 structurally uninformative until the
  OQ-1 scraper fix (now the binding constraint for H4 fidelity).
  H4 unlock projected ~11-13 more sessions.
  Evidence: `evals/d2_ci_vs_n_20260612/`.
- **H2 probe wiring benchmark (spec §4 rule):** 2 iters via train_6max
  resume — 15.0 s/iter at G=8 contended (~2.1 h for 500), league pool 3
  eligible mix 0.300, resume at 1500, strat loss 0.76 (champion-level).
  Finding: deployed ckpt is SLIM (no buffers) → probe rebuilds
  reservoirs; recorded in spec §4. Probe launches on §3 freeze
  (awaiting gushansen + millennium re-baseline rows).

## 2026-06-12 ~12:55 (system clock) — LIVE SESSION STARTED, FULLY ARMED — compute stood down

- Session `logs/live_dryrun_20260612_125441.jsonl` started; banner shows
  **H1 tail floor ARMED tau_max=0.100 (FIRST armed live session)** plus
  the full Stage-2 set armed/enforced (P1 anchor sum-floor, extended
  click plans, session-abort ENFORCED, watchdog-v2 fallback N=7.0s).
- Standing rule enforced: the two remaining e2 re-baseline rows
  (gushansenmtt, millenniummttv.49) were KILLED mid-run for CPU
  clearance (SIGSTOP did not stick; killed via tmux). CRN seeds make
  the relaunch reproduce identical rows post-session. H2 probe launch
  HELD for the same reason — spec §3 freeze now waits on both.
- Addendum 4.6 watcher armed: on session end → triage + ingest + H4
  counter + tail-floor anomaly check (expected band: distribution
  adjusted ~70-75% of decisions; sampled-action changes ~4.4%).

## 2026-06-12 ~13:45 — POST-SESSION PIPELINE (Addendum 4.6) — first armed-session H1 read: IN-BAND

Live window had 3 listener starts; sessions 125441 + 130846 ended (a
third listener is up idle, waiting for the sender — heavy evals remain
HELD). Pipeline results:

- **H1 first armed live observations (the headline): IN-BAND.** Main
  session 130846: 10/12 fresh decisions tail-fired (~83% vs ~70-75%
  self-play band at n=12), **0 argmax changes** (TG2 predicted 0), one
  check-when-free FOLD→CALL save. Triage: 233 frames / 22 decisions /
  0 red flags / 0 hands lost.
- **Session 125441 (the operator's restart explains itself):** 84
  frames, 88.1% skips, 7 red-flag safe-folds, 3 hands lost. NEW failure
  class observed: `derive_action_sequence emitted an illegal action`
  (chip_int=30 at 15/25 blinds — between BB and min-raise; smells like
  a bet-field OCR misread, e.g. 130→30). Bridge behaved correctly
  (safe-fold, no wrong decision). FLAGGED for the Windows scraper
  task's orbit (bet-field OCR); not yet a spec change — n=1 session.
- **Ingest:** both sessions; **H4 counter 259 → 272/500.** RESEARCH_MAP
  d1 updated.

## 2026-06-12 ~14:05 — a3 EV DECOMPOSITION (exploratory) DONE — three program-relevant verdicts

`evals/a3_v2_decomposition_20260612/` (terminal-depth proxy; per-hand
depth not recorded in the gate artifacts — harness fix documented):
1. **The bbnorm depth cure paid EV at shallow:** v2-vs-champion paired
   delta +0.130/game (z=+11.8) at <=6bb — exactly where the champion's
   depth confusion lives. Encoder = keepable organ (f-pillar).
2. **The killphil hole is depth-flat** (every bucket |z|<=0.61) — a
   shove-defense POLICY defect, not representation. Strengthens H2's
   monoculture premise independently of the adapter-artifact discount.
3. **v2's gate failure was an independent broad-spectrum over-folding
   regression** (folds-facing-action elevated at every depth) — travels
   with the run, not the encoder.

## 2026-06-12 ~14:35 — H2 SPEC FROZEN + PROBE LAUNCHED

§3 filled and frozen (commit aab1f52): killphil −0.0800 ± 0.0223,
holds ticketmaster +0.5400 / sng +0.8360 / tighttom +0.7780, champion
M1 0.0867 ± 0.0007, divergence check PASS (59/2000 outcome-differ).
Derived bars: probe M1 ≤ 0.0650; killphil row ≥ −0.0300 @ ≥2σ_diff.
Probe launched on freeze: tmux `h2_probe`, 500 iters league-mix
continuation (train_6max resume, G=8, cores 4-11), run dir
`runs/h2_probe_league_v1`. First iter reproduces the benchmark
deterministically (adv 0.5446 / strat 0.7625); 17.5 s/iter under
contention → ~2.4 h. Watcher armed (completion or Traceback).
On completion: grade ckpt_2000 on the frozen battery (F-M1), then
CRN killphil + hold rows + self-anchor (F-M2a/b), verdict per §5.
e2-record rows (gushansen, millennium) still completing on cores 0-3.

## 2026-06-12 ~15:00 — e2 re-baseline COMPLETE (9 rows); FIELD_DOSSIER landed; b2 launched

- **e2 record complete** (`evals/e2_rebaseline_20260612/REPORT.txt`):
  killphil −0.0800 (gate, ~54% of old extraction was artifact); all 9
  post-fix rows tabled. Standing rule: pre-2026-06-12 panel rows are
  PRE-FIX instrument readings — never mix without the tag.
- **FIELD_DOSSIER.md (slate 2d):** decision-grade field constants —
  flat positional VPIP (22.7–24.5% EP→BTN), 2.0x min-raise mode (55%
  of opens), jam-regime break at 10–15bb, 5–15bb jam rate 8.5%
  [6.1,11.8]. Timing structurally unextractable (2.5 s frame cadence).
- **b2 LAUNCHED (cores 0-1, tmux b2_variance):** killphil row at
  champion ckpt_1300 + ckpt_1400 (2000 games CRN each) — calibrates the
  gate metric's checkpoint-to-checkpoint swing BEFORE the H2 verdict
  reads its +0.05 bar against checkpoint noise.
- Core ledger: 0-1 b2, 2-3 blur map (agent), 4-11 probe (~16:40),
  slate agents 2b/2c/2e/2f rolling.
