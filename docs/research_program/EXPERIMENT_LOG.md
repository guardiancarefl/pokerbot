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

## 2026-06-12 ~15:40 — b2 (killphil half): gate metric is CHECKPOINT-STABLE

killphil row across adjacent champion checkpoints (2000 games CRN each):
ckpt_1300 −0.1020 ± 0.0222 · ckpt_1400 −0.0830 ± 0.0223 · ckpt_1500
−0.0800 ± 0.0223 (gate baseline). Max spread 0.022 < 1σ_diff (0.031).
**F-M2a's +0.05 bar is ~2.3× the observed checkpoint swing — an H2
probe pass cannot be checkpoint-selection noise.** RESEARCH_MAP b2
updated by this entry (ticketmaster complement still running, shared
cores). Probe note: iter ~1700/2000, slowed to ~28 s/iter under full
ledger (ETA ~17:45) — acceptable per Addendum 5 throughput-first.

## 2026-06-12 ~16:15 — LIVE SESSION #2 (armed) — probe interrupted at ckpt_1800

Session `logs/live_dryrun_20260612_161347.jsonl` started, tail floor +
Stage-2 set armed (banner verified). Stand-down per standing rule:
**H2 probe killed at iter ~1800 (ckpt_iter_1800.pt on disk — resume
path: train_6max --resume ckpt_iter_1800, 200 iters ≈ 50 min
post-session; recorded as an infrastructure interruption per spec §5's
relaunch clause, resume-from-checkpoint variant; checkpoint
slim-vs-full buffer caveat to be checked at resume and recorded)**;
b2-tm killed (CRN relaunch later). Blur shards had finished pre-session
(corrected-proxy rerun complete, 414k+413k decisions, 0 tainted);
aggregation is file-only and proceeds. Verdict battery launcher staged
(scripts/h2_verdict_battery.sh) — fires after the post-session probe
resume completes.

## 2026-06-12 ~16:50 — Session-2 main log triaged (tail floor IN-BAND again); listener restarted (window continues)

- `161347` (391 frames / 51 decisions / 30 hands): **tail floor 19
  firings, 0 tail-caused argmax changes** (2 argmax changes were
  check-when-free saves co-firing — that floor's job). H1 live evidence
  now 2 sessions, both in-band.
- Silence profile worse than session 1: 31 hero-to-act skips, 3 hands
  lost, dominant cause 49 scraper-suspect frames — the Windows P1/P2
  SanityChecker fix (routed, pending) remains root cause. RED FLAG:
  AcKc preflop safe-fold on a transient invariant_fail (pot/stack
  deltas of exactly 100) — exactly the approved-can-ride P2
  bet-closure-recovery class (NOT YET BUILT; operator visibility via
  this entry + morning read).
- Ingest deferred (mtime guard, possibly-live) — re-run post-window.
- Operator restarted a listener OUTSIDE tmux again (header-only log
  163114 waiting): window ACTIVE, compute stays down. Probe resume +
  battery queue on true window end.

## 2026-06-12 ~16:55 — window over; ingest done (H4 296/500); probe RESUMED 1800→2000

- Window-end verified (0 listeners; 163114 stayed header-only). Session
  161347 ingested: +24 opp-observed → **H4 296/500**.
- **Probe resumed** from ckpt_iter_1800 (tmux h2_probe, run dir
  `runs/h2_probe_league_v1_resume`, league pool 3/0.300 confirmed,
  20.8 s/iter → ~70 min). DEVIATION RECORDED: ckpt_1800 is SLIM
  (10.9 MB, no buffers) — the 1800-2000 segment trains on a
  fresh-rebuilt reservoir, the run's second buffer rebuild
  (interruption-induced). Outcome-based falsification unaffected;
  noted so the verdict reads the training path honestly.
- b2-tm relaunched (cores 0-1). Verdict battery fires on
  runs/H2_PROBE.DONE via the staged launcher.

## 2026-06-12 ~18:25 — EXP_H2 VERDICT: FAIL AT PROBE — H2 CLOSED

Full report: `reports/EXP_H2_killphil_league.md`. All load-bearing bars
broken (M1 0.1375 vs ≤0.0650; killphil −0.2360 vs ≥−0.0300; anchor
z=−9.9; ticketmaster −5.2σ). ckpt_1800 diagnostic killed the
interruption excuse (collapse pre-dates it; relaunch clause rejected as
results-motivated). Mechanism left open per the report: league-teaching
failure vs slim-ckpt fine-tuning fragility (evidence leans fragility).
RESEARCH_MAP b1 closed; POD_CASE suspended; battery + baselines +
league infra survive as instruments. **H2 is experiment #2 complete
(H1 #1) — Addendum 1.4 program review triggers after the NEXT completed
experiment.** Next per operator sequencing: P2 build (task #7), refill
pass for the experiment queue.

## 2026-06-12 ~19:05 — LIVE SESSION #3 starting — night shift stood down

12 compute processes killed (ICM t1 rollouts, ensemble baseline rows).
Both are CRN-deterministic per their designs — relaunch reproduces.
Agents will find the listener on their next check and hold per their
stand-down rules. P2 build agent (file/tests only) continues. Pipeline
queues on session end per Addendum 4.6.

## 2026-06-12 ~19:50 — P2 BUILT + GATED (commit 3bf5d3f) — and the AcKc fixture flipped the diagnosis

P2 bet-closure recovery landed flag-gated OFF (requires P1), full gate
treatment: flag-off byte-identity 6077/6077 frames over all 14 raw
logs; flag-on diffs are 9 annotation-only refusals; 23/23 new tests.
**Build finding: seq=276 (AcKc) is NOT a displacement** — dead-SB hand,
phantom recon BB; recovery would have been a wrong-state decision
(seq-1363 class). Dead-SB guard added (positive blind-structure
evidence required); the synthetic live-SB twin proves the mechanism.
**Root-cause follow-up filed: carry blind structure into replay so
dead-SB hands reconstruct — that is the real fix for the AcKc class**
(bridge-side, candidate for the next build slot). Arming = operator
call; flag table update for DRYRUN_CHECKLIST queued with it.

## 2026-06-12 ~20:00 — Session-3 pipeline + night shift resumed

- Session `183701` (largest yet: 1519 frames / 194 decisions / 140
  hands): **tail floor 104 firings, 0 tail-caused argmax changes —
  third consecutive in-band armed session, now at meaningful n.**
  2 layer-1 recoveries fired live (first live firings). 83 to-act
  skips / 6 hands lost / 8 red flags — scraper-suspect class still
  dominant; full triage at logs/triage_20260612_183701.txt. Ingest
  queued behind the mtime guard (+H4 update with it).
- Night shift resumed: ensemble Arm B (cores 0-3, fresh agent with the
  prior agent's state), ICM tier-1 (cores 4-7, same). P2 build landed
  earlier (3bf5d3f).

## 2026-06-12 ~20:20 — session-3 ingested: H4 383/500 (+87 in one session)

140 hands / 411 actions / 87 opp-observed / 9 showdowns (hero-inferred).
Unlock ~1-2 sessions away — **H4 spec drafting promoted to the refill
slate** (draft before unlock so the experiment starts the day the
counter crosses). RESEARCH_MAP d1 updated.

## 2026-06-12 ~21:10 — EXP_c1 (ICM gap Tier-1) VERDICT: MATERIAL — re-price

Third completed program experiment (H1 PASS, H2 FAIL, c1 MATERIAL).
MH ICM bias B* = +0.0412 [+0.0286, +0.0539] Bonferroni 98.33% in
high-dispersion bubble states — H_c1 falsified, "re-price" branch of
the pre-registered decision matrix. Structure: MH underprices
short-stack survival (depth <5bb error +0.065), overprices mid stacks;
equal-stack states ≈ clean (t1 −0.0034 n.s.) so TG3/battery
equal-stack metrics are not invalidated wholesale — the bias
concentrates exactly where the program's bubble interest lives.
σ_bias(t3)=0.068 → HETEROGENEOUS (a correction must be state-dependent).
C0 instrument-halt fired + fixed + re-run clean (DEVIATION_LOG).
Follow-ups queued to refill: correction fit (depth_bb first feature);
consumer re-price audit (icm_adjust_returns, subgame leaves, H2-battery
re-label, H1 floor re-check — all expectations: small at equal stacks).
**Addendum 1.4 PROGRAM REVIEW now due (3 completed experiments).**

## 2026-06-12 ~22:00 — Windows Part-A PROVEN; cross-machine seat off-by-one routed to audit

- **OQ-1 Part A (folded-flag fix) replay-proven on Windows: 0/241 →
  153/158 folds detected, 0 false positives, bridge untouched.** Patch
  holds for live deployment pending the seat-convention reconciliation
  below.
- **Windows CC root-caused a seat off-by-one in OUR candidates file**
  (extract_stale_folded_candidates.py: candidate seat i = scraper seat
  i+1). HIGH-PRIORITY audit launched: does the H3 ingester itself
  misattribute per-seat data (Case B would contaminate FIELD_DOSSIER
  positional/per-opponent claims and H4's training target)? Ground-truth
  verification against raw records; candidates v2 with explicit
  convention header. FIELD_DOSSIER consumers ON HOLD for per-seat
  claims until the audit reports.

## 2026-06-12 ~22:25 — Seat audit: CASE A (cosmetic) — loop CLOSED, Part-A clear to deploy

52 ground-truth checks (9 hands / 6 sessions): ONE uniform mapping (raw
seatN → internal N−1) everywhere in the H3 DB — **FIELD_DOSSIER and H4
inputs UNCONTAMINATED, no re-ingest.** The off-by-one lived only in the
v1 candidates EXPORT (internal ints leaked out). Fixed at the source:
extractor now emits "seatN" + seat_convention field, byte-equivalent to
the audit's v2 (98682fa). Side finding: actions.seq is unique per
listener run, not per file — captured_at disambiguates. **Both seat
conventions reconciled → the Windows Part-A patch (153/158 folds, 0 FP)
is CLEAR for live deployment.** Dossier hold lifted.

## 2026-06-12 ~22:45 — f1 ensemble probe: FALSIFIED (K2) — frontier item closed at probe cost

All bars in the agent result; headline: no cheap organ improves the
bubble (best candidate WORSENS it, z=-2.54, plus control harm). f1
closed; RESEARCH_MAP updated. Cores 0-3 roll to the ICM correction fit
(spec FROZEN as drafted — gates G1-G3 + validation bars pre-committed).

## 2026-06-12 ~23:05 — REGISTRATION SUPPLEMENT (operator/CEO) — logged on receipt

Received BEFORE M-C was read (slate-2a agent's last report: M-C still
running, no number observed by manager or agent-final-message at
receipt time — the supplement is clean pre-registration):

1. **M-C "degrades" DEFINED:** point estimate worse than the −0.08
   killphil baseline by more than 1 SE ⇒ degrades; within 1 SE ⇒
   holds. Tightening a registered bar after an undesired result is
   prohibited the same as loosening one.
2. **SHOVIEST-ROWS PANEL (conditional on M-C holds), registered now:**
   the 3 highest shove-frequency profiles from the bake-off pool +
   killphilmtt's nearest stylistic neighbor; 2000 CRN-paired games per
   row; **tail+shove config vs tail-only config**; bars: no row
   significantly negative (z ≤ −2), pooled delta ≥ 0 within noise.
   Profile names to be logged BEFORE launch (selection method: measured
   open-shove range size at canonical short-stack spots through the
   fixed adapter; neighbor by range-overlap vs killphil — selection
   computed next, logged below before any panel launch).
3. **Interaction-check gray zone z ∈ (−2, 0) reaffirmed as registered
   PASS.**

## 2026-06-12 ~23:15 — Shoviest-rows panel SELECTION logged (pre-launch, per supplement §2)

Method (as registered): open-shove range size through the fixed
adapter at the canonical short-stack spots (UTG + SB first-in, 8bb,
L5 — the battery cells), union of the two; neighbor by mean Jaccard
overlap of (UTG, SB) shove sets vs killphilmtt.

- 3 highest shove-frequency: **lionmttv.10** (union 123/169),
  **modernmikemtt** (63), **itmstrikea** (58; itmstrikec ties at 58 —
  same family, tie broken to the A-variant, logged here).
- killphil nearest stylistic neighbor: **minestackermttv.7.3**
  (Jaccard 0.717; runners-up thefixersng/sng 0.712).

PANEL = {lionmttv.10, modernmikemtt, itmstrikea, minestackermttv.7.3},
2000 CRN-paired games per row, tail+shove vs tail-only config, bars per
supplement §2. LAUNCH CONDITION: M-C holds (within 1 SE of −0.08).
M-C still unread at this log entry.

## 2026-06-12 ~21:35 — c1 item #2: ICM correction v1 — VERDICT: ADOPT (via registered fallback M2)

Spec executed: `ICM_CORRECTION_SPEC.md` (fit on the existing 3,000
Tier-1 seat residuals; holdout gates BEFORE rollouts; pre-registered
40k-rollout fresh-state validation; consumer-audit v1).

1. **M1 failed holdout G3** (micro-depth 0.0721 → 0.0352 vs bar 0.030;
   G1/G2 passed) after two logged methodology deviations (D1
   projection-aware fit, D2 gate-coherent variant selection — the
   literal pre-projection WLS + 1-SE rule is structurally degenerate
   under the registered conservation projection; full trail in
   `evals/c1_correction_20260612/DEVIATION_LOG.txt`, incl. D3 holdout
   double-opening).
2. **M2 (registered fallback) passed G1–G3** (t3 d_short 0.0549 →
   0.0173; micro 0.0721 → 0.0203) ⇒ validation launched.
3. **Fresh validation (400 states × M=100, caps=0, taints=0): V1+V2+V3
   ALL PASS.** t3 raw +0.0380 (replicates +0.0412) → corrected +0.0078
   CI95 [−0.0025,+0.0182]; t1 corrected +0.0050 CI ⊂ ±0.015; micro
   oversample +0.0671 → +0.0142 ≤ 0.03. No λ-shrink needed.
   Heterogeneity (§2.5): fresh-t3 σ_bias 0.0531 → 0.0431 (34% of
   σ²_bias explained) — 0.043 bounds corrected per-decision accuracy.
4. **Consumer-audit v1:** battery 0/8112 flips ASSERTED (max ICM diff
   0.0); TG3/TG4 re-reads hold (TG3 scorer verified exact cash, not
   MH); sng_baseline capped count 0/66,000 games; floor-panel re-price
   NOT EXECUTABLE (no per-hand records persisted — manager flag);
   MH-pinning icm tests 48/48 green, `icm.py` untouched.
5. Artifacts: `src/nlhe/icm_correction.py` (standalone, NOT wired into
   any consumer), `data/icm_correction_v1.json`,
   `evals/c1_correction_20260612/{REPORT.txt,results.json,…}`.
   UNCOMMITTED per task mandate. Adoption limitations (M2 steps ⇒ no
   gradient consumption pending Tier-2b; non-conservation ΣP up to
   ~3.03; bubble-cell scope only) listed in REPORT.txt §4. Manager
   actions on ADOPT: commit + DECISIONS.md + RESEARCH_MAP c1; queue
   Tier-2 / Tier-2b / retrain-flag pod item.

## 2026-06-12 ~23:40 — slate-2a M-C VERDICT: HOLDS (improves) — panel condition MET

M-C (floor-on killphil row, 2000 CRN games): hero net **−0.0300 ±
0.0224** vs post-fix baseline −0.0800 ± 0.0223 — within-1-SE bar not
just met but beaten by ~2.2σ_diff toward improvement; extraction
halved. F3 passes vs both baselines (pre-fix 0.174 noted, post-fix
authoritative). Floor fired 3,134/3,410 qualifying nodes (6.8% of all
decisions qualify). With M-A F1 PASS and M-B F2 PASS: **slate-2a probe
PASSES wholesale.** Supplement §2 condition met → SHOVIEST-ROWS PANEL
LAUNCHING with the pre-logged selection (lionmttv.10, modernmikemtt,
itmstrikea, minestackermttv.7.3; tail+shove vs tail-only; 2000 CRN
paired games/row; bars: no row z ≤ −2, pooled ≥ 0 within noise; gray
zone z ∈ (−2,0) = registered PASS per supplement §3).

## 2026-06-13 ~00:25 — session-4 pipeline complete; H4 410/500

Session 220101: 36 hands / 53 decisions / 0 lost / tail floor 25
firings 0 argmax changes (4th consecutive in-band armed session).
Ingested +27 opp-observed → H4 410/500. RT-1 exhibit from this session
registered separately (RED_TEAM_LOG). Panel relaunched post-window.

## 2026-06-13 ~01:30 — SESSION-5 BUILD SET launched (operator directives)

Session 220101→230149 audit directives in flight, four agents:
1. **P2 validation vs tonight's casualties** (P2 already BUILT 3bf5d3f
   — directive executes as targeted counterfactual: 8c8d/AcAh/5hTs
   recover-or-refuse + arming recommendation; AcAh has operator-played
   ground truth).
2. **Click-plan RAISE→CHECK root-cause** (Stage-2 blocker; 2 exhibits:
   seq194 RAISE_TO 610 tonight + prior 716-raise).
3. **D2 dead-button position handling** (operator: Windows attribution
   proved tonight's dealer-bursts were CORRECT dead-button scrapes —
   bridge gap, #1 hand-killer; BB-advances-one-active invariant +
   posts-as-ground-truth derivation; flag-gated, full chain; QdKc/TsAc
   counterfactual targets; Windows adding additive dealer_dead flag).
4. **Postmortem + Stage-2 ledger (N=5 gate proposed) + games-played
   counter** (calibration: operator played 3, won the last).
Session-5 tail floor: 78 firings, 0 argmax changes — in-band ×5.
Panel: 5/8 shards survived the live window; 3 relaunched.

## 2026-06-13 ~02:50 — Operator answers logged

Stage-2 gate CONFIRMED N=5 (ledger updated — streak 0/5). Game-count
detector CONFIRMED correct (4 games L,L,L,W; operator had forgotten the
13-min first bust — detector calibration validated against operator
memory and won). OQ-3 routed to Windows CC by operator alongside D3.

## 2026-06-13 ~03:05 — SLATE-2A COMPLETE: shove-defense floor passes everything

M-A F1 PASS · M-B F2 PASS · M-C holds-improves (extraction halved) ·
shoviest panel ALL ROWS POSITIVE, pooled +0.0108 ± 0.0023 (z=+4.7).
Cycle-2's #1 EVoI item delivered: zero training, all frozen
instruments, gray-zone never needed. DEPLOYMENT PACKAGE now assembles
for the operator: shove floor arming + P2 arming + click-plan fix
(committed) + D2 + commit-reconciliation + abort-counter fix — the
session-5 set. Caveat carried: oracle is killphil-range-specific and
MH-based at 6-alive equal-stack cells (ICM Tier-2 will bound it);
self-play and panel evidence is the load-bearing EV case.

## 2026-06-13 ~03:20 — OQ-3 root-caused (Windows): all-in zeros, not button overlap — F2 routed

Windows causal test DISPROVED button overlap; the pointed-seat kills
were ALL-IN seats displaying "0" + an ocr_int truthiness bug dropping
consensus zeros. Windows ships F1 (ocr_int, replay-gated). Bridge F2
(0-stack in-hand seat = ALL-IN, valid state) registered as task #11,
sequenced after D2 (same scraper_schema region, avoid agent
collision); counterfactual targets = session-5 KQo/ATo 54-frame set.
OQ-3 updated: mechanism resolved, F1 Windows-side + F2 bridge-side.

## 2026-06-13 ~03:35 — H4 UNLOCKED (517/500) + unlock-session WIN

Session 023826 (70 hands, +32 opp-observed) crossed the H4 threshold.
The H3/H4 pipeline's gating goal is met. H4_RNR_FIELD_SPEC (drafted at
383) now enters its freeze checklist: (1) recompute FIELD_DOSSIER
constants at n≥500 + d2 CI re-grade (running, file-only); (2) OQ-1
folded-flag check (Windows Part-A proven but not yet in the archive —
fold-vs-shove still undercounted, dossier must flag it); (3) registration
freeze. The EXPERIMENT is a field-RNR retrain = training-scale →
POD-relevant + H2-fragility-binding (buffer continuity, no slim-ckpt
fine-tune). Surfaced to POD_CASE. Not auto-launching: freeze first,
operator decides pod.
