# FIELD DOSSIER — live opponent archive, deep-cut analysis (Addendum 5, slate 2d)

Generated 2026-06-12 from `data/opponent_db/opponent_db.sqlite` (sqlite-only analysis,
no simulations). This is the document the eventual H4 spec consumes.

**Substrate snapshot:** 18 sessions (13 raw-record, 5 summary-tier), 483 hands,
1806 non-hero dealt seat-hands, 581 opponent voluntary frame_diff actions,
62 opponent keys `(session_id, seat)`. **H4 unlock counter: 272/500.**
Note: the d2 CI report (`evals/d2_ci_vs_n_20260612/REPORT.txt`) was generated against a
slightly earlier snapshot (16 sessions / 467 hands / 259 opp-obs); two short sessions
ingested since. All cross-checked numbers below remain inside d2's CIs — no conclusion
changes.

**Grading rubric** (inherits d2's conventions):
- `[GRADE: decision]` — Wilson 95% half-width ≤ ±5pp for field-level reads (±15pp for
  per-opponent reads), AND not structurally invalidated by a bias flag.
- `[GRADE: provisional]` — half-width above threshold, or a bias flag materially bites,
  or the read is qualitative on small n. Usable as a prior, not as a spec constant.
- `[GRADE: anecdote]` — n too small for any frequency claim; shape/illustration only.

**Inherited bias flags** (every number in this file carries them; see d2 REPORT for detail):
- **[B1]** All frequencies are **observed lower bounds**: ~2.5s frame sampling loses
  actions between frames; checks are invisible (no chips move).
- **[B2]** Folds are ~100% missed (stale `folded` flag, RESEARCH_MAP d3). **Any
  fold-based statistic is uncomputable, not noisy.** Only chip-moving frequencies
  (VPIP, raise, call, all-in) are trustworthy lower bounds.
- **[B3]** Wilson CIs around [B1]-suppressed numerators look deceptively narrow near
  p̂=0; the true rate may sit above the upper limit.

All opponent statistics below use non-hero `frame_diff` rows only (`decision_record`
rows excluded per `data/opponent_db/README.md`); raw-record sessions only.

---

## 1. Positional tendencies (VPIP lower bound by relative position)

Position derived per seat-hand from `dealer_seat`/`sb_seat`/`bb_seat` + `dealt_seats_json`;
non-blind non-button seats ordered from first-to-act: EP / MP / CO. Numerator = ≥1
voluntary preflop frame_diff action.

| pos | VPIP-lb | Wilson 95% | half-width | grade |
|---|---|---|---|---|
| EP  | 60/262 = 22.9% | [18.2, 28.4] | ±5.1pp | provisional (grazes ±5pp) |
| MP  | 41/181 = 22.7% | [17.2, 29.3] | ±6.1pp | provisional |
| CO  | 76/322 = 23.6% | [19.3, 28.5] | ±4.6pp | decision [B1] |
| BTN | 79/323 = 24.5% | [20.1, 29.4] | ±4.7pp | decision [B1] |
| SB  | 87/313 = 27.8% | [23.1, 33.0] | ±4.9pp | decision [B1] |
| BB  | 29/317 =  9.1% | [ 6.4, 12.8] | ±3.2pp | provisional — see below |
| unclassifiable | 0/88 | [0.0, 4.2] | — | data-quality artifact |

- **The field shows essentially NO positional discipline: EP ≈ MP ≈ CO ≈ BTN
  (22.7–24.5%), all four CIs overlapping almost completely.** `[GRADE: decision,
  n=262–323 per cell; the qualitative claim "no EP→BTN widening detectable at ±5pp"
  is robust]`. A positionally-aware population would show BTN well above EP; this
  field's open range is position-invariant — the classic recreational-pool
  fingerprint. [B1] applies equally across positions, so the *flatness* survives the
  undercount even though levels are lower bounds.
- SB elevated (27.8%) — includes SB completes, which are cheap and chip-moving
  (hence visible). `[GRADE: decision on the observed rate, n=313; provisional as a
  "looser SB" interpretation since completes ≠ opens]`
- **BB 9.1% is NOT a VPIP** in the usual sense: BB voluntary chips only register when
  the BB calls a raise or re-raises; BB checks are invisible [B1] and BB folds
  unrecorded [B2]. Read it as "BB puts in extra chips ≥9.1% of the time", i.e. a
  defend-frequency floor. `[GRADE: provisional, n=317, CI [6.4,12.8] — structurally
  a different quantity than the other rows]`
- The 88 unclassifiable seat-hands come from 36 hands with `dealer_seat` NULL
  (degraded frames); they show **zero** observed voluntary actions, so including them
  in the global denominator slightly deflates field VPIP. `[GRADE: anecdote —
  data-quality note, not a behavior read]`

## 2. Sizing tells (385 opponent bet/raise frame_diff events)

### 2a. Preflop open sizing (first raise of the hand; n=191, of which 44 all-in)

- **All-in share of first-raises: 23.0% [17.6, 29.5]** `[GRADE: decision, n=191]` —
  nearly a quarter of all observed opens are open-shoves (blind-pressured SNG field).
- Non-all-in opens (n=147), in xBB (0.5x bins):
  `2.0x: 76 | 2.5x: 7 | 3.0x: 24 | 3.5x: 8 | 4.0x: 10 | 5.0x: 13 | tail to 14x: 4 | sub-min noise ≤1.5x: 5`
- **The distribution is multimodal: a dominant min-raise mode at exactly 2.0x
  (≤2.0x = 55.1% [47.0, 62.9]), a secondary mode at 3.0x (16%), and a 5x+ "I have
  a hand" tail (~10%).** `[GRADE: decision on the 2x-mode dominance, n=147;
  provisional on the exact mode weights]`
- Open size is **depth-invariant**: median 2.0x in every actor-depth band
  (5–15bb n=26, 15–25bb n=28, 25bb+ n=93). `[GRADE: provisional, per-band n small]`
- Per-session open-size medians split cleanly into a 2.0x cluster and a 3.0–3.5x
  cluster (see §5) — sizing is a **table-type fingerprint**, not just a player one.
  `[GRADE: provisional, 13 sessions]`
- Caveat on "first raise": frame collapsing [B1] can merge limp-then-raise sequences,
  so a few entries are raises-over-limpers misread as opens.

### 2b. Preflop calls: limp-dominant

Of 150 opponent preflop voluntary calls, **95 (63%) are limps/completes to exactly
1bb**. `[GRADE: decision on the observed split, n=150, Wilson [55.3, 70.6]; B1 makes
both components lower bounds]` Combined with §2a: the field's preflop entry mix is
limp-heavy and min-raise-heavy — almost no standard 2.2–2.5x sizing exists (5% of
non-all-in opens).

### 2c. Postflop pot-fraction (amount_to / reconstructed pot)

The actions table has no pot column; pot-before-action was reconstructed from blinds +
antes + per-street max observed commitment per seat. **Missed actions [B1] make the
reconstructed pot a lower bound, so the fractions below are biased HIGH.** Treat shapes,
not point values.

| street | n | p25 | median | p75 | ≤1/4 pot | 1/4–0.45 | ~1/2 | 2/3–0.85 | ~pot | >pot | all-in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flop  | 94 | 0.27 | 0.44 | 0.77 | 24% | 29% | 14% | 10% | 10% | 14% | 6 |
| turn  | 48 | 0.18 | 0.51 | 0.81 | 40% | 10% | 17% | 12% |  6% | 15% | 5 |
| river | 26 | 0.38 | 0.66 | 0.91 | 19% |  8% | 19% | 19% | 19% | 15% | 3 |

- **Bimodal flop/turn profile: a small-bet mode (≤0.45 pot, ~50% of flop bets) and an
  overbet tail (>pot, 14–15% on every street).** `[GRADE: provisional — flop n=94 is
  the only street near sample adequacy, and the pot-reconstruction bias is
  unquantified]`
- River shape (drift toward 2/3–pot) `[GRADE: anecdote, n=26]`.

## 3. Depth behavior (H2's 5–15bb regime)

### 3a. Per-action: opponent voluntary actions by actor `stack_depth_bb` at event time

| band (bb) | n | all-in % [CI] | bet/raise % [CI] | grade |
|---|---|---|---|---|
| 0–5   |  27 | 77.8 [59.2, 89.4] | 66.7 [47.8, 81.4] | provisional (n) |
| 5–10  |  42 | 59.5 [44.5, 73.0] | 76.2 [61.5, 86.5] | provisional (n) |
| 10–15 |  67 | 13.4 [ 7.2, 23.6] | 68.7 [56.8, 78.5] | provisional |
| 15–25 |  97 | 10.3 [ 5.7, 17.9] | 71.1 [61.4, 79.2] | provisional |
| 25–50 | 155 |  7.1 [ 4.0, 12.3] | 64.5 [56.7, 71.6] | decision [B1] |
| 50+   | 193 |  1.6 [ 0.5,  4.5] | 62.2 [55.2, 68.7] | decision [B1] |

- **The all-in gradient is steep and monotone, with the regime break between 10 and
  15bb: below 10bb the majority of voluntary actions are jams (60–78%); at 10–15bb it
  collapses to ~13%.** `[GRADE: decision on monotonicity and on the <10bb-jam-regime
  claim — CIs of 5–10 vs 10–15 are disjoint; provisional on per-band levels]`
- The bet/raise share column is NOT an aggression factor: its denominator excludes
  invisible checks [B1] and unrecorded folds [B2], so it is inflated by construction.
  Do not consume it as AF. `[GRADE: anecdote as a level; the depth-invariance of the
  share is mildly informative]`

### 3b. Per-seat-hand: pre-hand depth (anchored `start_stacks_json`) → preflop behavior

Denominator = non-hero dealt seat-hands with anchored pre-hand stack (anchored hands
only — mild selection toward clean-frame hands, per d2).

| band (bb) | seat-hands | VPIP-lb [CI] | PFR-lb [CI] | preflop all-in [CI] |
|---|---|---|---|---|
| 0–5   |  54 | 29.6 [19.1, 42.8] | 18.5 [10.4, 30.8] | 22.2 [13.2, 34.9] |
| 5–10  | 178 | 19.1 [14.0, 25.5] | 14.0 [ 9.7, 19.9] | 11.8 [ 7.8, 17.4] |
| 10–15 | 198 | 21.2 [16.1, 27.4] | 14.6 [10.4, 20.2] |  5.6 [ 3.1,  9.7] |
| 15–25 | 233 | 23.2 [18.2, 29.0] | 12.0 [ 8.4, 16.8] |  2.1 [ 0.9,  4.9] |
| 25–50 | 338 | 28.1 [23.6, 33.1] | 16.3 [12.7, 20.6] |  2.1 [ 1.0,  4.2] |
| 50+   | 426 | 20.9 [17.3, 25.0] | 10.3 [ 7.8, 13.6] |  0.2 [ 0.0,  1.3] |

- **Combined 5–15bb (H2 regime): VPIP-lb 76/376 = 20.2% [16.5, 24.6] (±4.0pp) —
  decision-grade at ±5pp** `[GRADE: decision, B1 lower bound]`; matches d2 Q3
  (20.4% on the earlier snapshot). Short stacks do NOT tighten up relative to the
  field's 20.6% baseline.
- **Combined 5–15bb preflop all-in rate per seat-hand: 32/376 = 8.5% [6.1, 11.8]
  (±2.8pp) — decision-grade.** `[GRADE: decision; all-ins are large chip moves and
  the least [B1]-suppressed event class]` This is the single most H2-relevant field
  constant in the archive: a 5–15bb opponent jams preflop roughly 1 hand in 12, and
  at 5–10bb specifically 11.8% [7.8, 17.4] `[GRADE: provisional per sub-band]`.
- 0–5bb: jam rate 22.2% [13.2, 34.9] `[GRADE: provisional, n=54]`.
- Opponent preflop re-raises (3-bets) are rare in-archive: 26 events total, of which
  13 all-in (50% [32, 68]). `[GRADE: anecdote — directionally "3-bets are polar/jammy",
  nothing more]`
- **Fold-vs-shove in or out of band: 0/25 observed facing-allin events folded —
  STRUCTURALLY UNINFORMATIVE [B2].** Not a read; do not consume. `[GRADE: n/a —
  uncomputable until OQ-1]`

## 4. Timing patterns — NOT extractable. Full stop.

- `captured_at` timestamps are the **scraper's frame-detection times, not player action
  times**: median per-hand frame cadence is 2.5s (p10 1.4s, p90 3.8s, from
  `duration_seconds/(n_frames-1)`, n=405 hands), and an action is stamped at the frame
  where its chip diff was first seen — the physical action occurred anywhere in the
  preceding cadence window.
- Consecutive within-hand action deltas (n=421): median 8.8s, p25 4.7s, p75 13.7s.
  Each delta confounds (a) player think time, (b) ±1–2 frames of sampling quantization
  (≈ ±2.5–5s — same order as the think times themselves), and (c) multi-action
  collapse (several actions landing in one frame produce delta≈0; 17% of deltas ≤3s).
- **Verdict: timing tells are structurally absent from this corpus.** `[GRADE: n/a]`
  Extracting them requires event-driven or sub-second capture on the Windows scraper
  side — queue alongside the OQ-1 folded-flag fix if ever wanted; do not spec H4
  features on timing.

## 5. Per-session field composition (table-type archetypes)

13 raw-record sessions, sorted by seat-hands. `allin/100sh` = opponent voluntary
all-in events per 100 seat-hands; `aggr%` is observability-inflated (see §3a caveat).

| session | hands | seatH | VPIP-lb [CI] | open med xBB (n) | allin/100sh |
|---|---|---|---|---|---|
| 20260611_222532 | 80 | 380 | 22.9 [19, 27] | 2.0 (37) | 3.2 |
| 20260611_204751 | 76 | 306 | 23.5 [19, 29] | 2.0 (29) | 7.2 |
| 20260611_201230 | 62 | 256 | 18.4 [14, 24] | 2.0 (19) | 6.2 |
| 20260609_154557 | 48 | 173 | 14.5 [10, 20] | 2.0 (9)  | 2.9 |
| 20260609_192543 | 44 | 166 | 13.3 [ 9, 19] | 3.0 (7)  | 3.0 |
| 20260611_163815 | 37 | 140 | 22.9 [17, 30] | 3.0 (15) | 5.0 |
| 20260612_042058 | 35 | 126 | 19.0 [13, 27] | 3.0 (10) | 1.6 |
| 20260612_035454 | 15 |  74 | 20.3 [13, 31] | 3.0 (5)  | 2.7 |
| 20260608_152756 | 11 |  53 | 22.6 [13, 36] | 2.8 (4)  | 5.7 |
| 20260612_130846 | 11 |  52 | 38.5 [26, 52] | 3.5 (6)  | 7.7 |
| 20260611_181249 |  6 |  30 | 26.7 [14, 44] | 5.0 (4)  | 3.3 |
| verify1_20260609 |  6 |  30 | 16.7 [ 7, 34] | 2.0 (3)  | 0.0 |
| 20260612_125441 |  5 |  20 | 15.0 [ 5, 36] | 2.0 (2)  | 0.0 |

Qualitative read (n=13 sessions — this is a sketch, not a classifier):
- A **tighter cluster** (~13–15% VPIP-lb: both 2026-06-09 long sessions) and a
  **looser-standard cluster** (~18–24%: the 2026-06-11 block), with one **loose
  outlier** (38.5% [26, 52], 11 hands). The two clusters' CIs barely separate
  (e.g. 14.5 [10,20] vs 22.9 [19,27]) — real but marginal at current n.
  `[GRADE: provisional — between-session spread exceeds sampling noise for the
  extreme pairs only]`
- The **open-median fingerprint (2.0x vs 3.0x tables) tracks the date blocks more than
  VPIP does** — consistent with table-type (or lobby-time) regimes rather than one
  homogeneous pool. `[GRADE: anecdote, 13 sessions, per-session open n=2–37]`
- Implication for H4: a single field prior is justified today; a two-archetype mixture
  (tight-3x vs standard-minraise) is the first refinement to test once ~30+ raw-record
  sessions exist. `[GRADE: provisional]`

## 6. Cross-check against the d2 CI framework

- Field VPIP-lb here: 372/1806 = 20.6%, vs d2's 349/1734 = 20.1% [18.3, 22.1] — inside
  d2's CI; the two snapshots agree. ✔
- Depth 5–15bb VPIP-lb 20.2% [16.5, 24.6] vs d2 Q3 20.4% [16.6, 24.9]. ✔
- d2's verdicts adopted wholesale: per-opponent VPIP is a late-session signal only
  (≥26–39 observed hands, ~half of keys, long sessions only); **no per-opponent
  depth-conditioned read ever becomes decision-grade within a session** — field
  pooling is the only viable 5–15bb estimator; sampling error is already an order of
  magnitude below the [B1]/[B2] observability bias, so **more hands no longer improve
  read quality — the scraper fix does**. Every per-band number in §1–§3 above was
  graded against d2's ±5pp (field) / ±15pp (per-opponent) bars, with i.i.d. caveat:
  true CIs are somewhat wider (within-session opponent repetition), so borderline
  "decision" grades (EP positional, flop sizing mode) should be read conservatively.

---

## What H4 can use today

1. **Field VPIP-lb 20.6% (±1.9pp)** as the population activity prior; true VPIP ≥ this. [decision]
2. **Positional flatness**: model field open ranges as position-invariant EP→BTN
   (22.7–24.5%); do not import blueprint-style positional widening into the field
   prior. SB completes elevated; BB defend-floor ≥9%. [decision on flatness]
3. **Sizing vocabulary for the response abstraction**: preflop mass sits at
   {limp, 2.0x, 3.0x, jam} — 2.2–2.5x is ~5% of opens. 23% of first-raises are
   open-jams. Off-tree translation (Ganzfried-Sandholm in `actions.py`) should expect
   a hard 2.0x mode. [decision]
4. **H2-regime constants**: 5–15bb field VPIP-lb 20.2% (±4.0pp); **preflop jam rate
   8.5% (±2.8pp) per seat-hand in band** (11.8% at 5–10bb, 22.2% at 0–5bb,
   provisional per sub-band); jam-regime boundary sits between 10 and 15bb. [decision
   on combined band + boundary]
5. **Estimator policy** (from d2): field-pool everything depth-conditioned; allow at
   most a coarse per-opponent overall-VPIP offset, late-session, long matches only. [decision]
6. Directional only: limp-heavy entry mix (63% of calls are limps), small-bet+overbet
   bimodal flop sizing, polar/jammy 3-bets, possible tight-vs-standard table
   archetypes. [provisional — priors, not constants]

## What must wait for n=500 / the scraper fix

- **Everything fold-based** (fold-vs-shove, fold-to-3bet, steal success, any
  continuation-frequency denominator): uncomputable until the Windows folded-flag fix
  (OQ-1) lands [B2]. No amount of n helps.
- **True (not lower-bound) VPIP, limp rate, check-derived stats**: need faster or
  event-driven capture [B1]; n=500 will not move these.
- **Timing tells**: structurally absent (§4); requires a scraper redesign, not data.
- **Opponent showdown holdings / range fitting**: the scraper schema has no opponent
  hole-card field — 0 rows exist and 0 will exist until the schema changes.
- **3-bet frequency & sizing distributions**: 26 events total; needs roughly the n=500
  unlock corpus (and ideally the fold fix for the denominator).
- **Per-band depth levels at ±5pp** (0–5, 5–10 separately) and **postflop turn/river
  sizing shapes**: more hands genuinely help here; revisit at unlock.
- **Session-archetype classifier** (tight-3x vs standard-minraise mixture): needs
  ~30+ raw-record sessions; today it is a sketch.
- **Per-opponent depth-conditioned reads**: never arrive at live data rates (d2 Q3/Q4)
  — design H4 around their absence rather than waiting.
