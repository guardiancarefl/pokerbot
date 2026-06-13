# H4 — RNR FIELD-EXPLOITATION RETRAIN — registration

**Status: FROZEN 2026-06-13 at counter 517 (≥ 500).** Drafted 2026-06-12 at
383; the §5 data-refresh checklist ran at unlock
(`evals/h4_freeze_refresh_20260613/`) and every §2.2 target T1–T8 reproduced
INSIDE its dossier Wilson CI — no constant moved outside CI, so the moment
targets are REFINED (not re-stated) and `registry_h4_field.json` does NOT
require re-solving. Substrate: 23 sessions / 18 raw-record / 3426 non-hero
seat-hands / 1041 opp voluntary frame_diff actions. §4 bars unchanged
(relative to champion baselines re-recorded at launch). The refreshed T1–T8
values are tabled in the refresh report + `refreshed_constants.json`; the
within-CI drift (T1 20.6→20.1%, T5 23.0→25.5% jammier, T7 8.5→8.4%, T8
boundary intact) supersedes the `[DRAFT@…]` stamps below for execution.
**OQ-1 fold-fix status (material):** the fold-flag fix HAS landed — 3 post-fix
sessions capture folds densely; fold-vs-shove is now COMPUTABLE (66.1%) but
NOT decision-grade (±8.2pp). Per §5, the cap is NOT lifted: **p stays at 0.25**
for the probe (fold dimension measurable-but-sub-grade, carried by the anchor),
lifting to p=0.30 + a fold-vs-shove target at the NEXT freeze once post-fix
fold data reaches ±5pp (~6 post-fix sessions). This re-opens §7 R2 (PATCH 5 in
the refresh diff). Per Addendum 1: now frozen, thresholds may not be weakened.

> Freeze provenance: `evals/h4_freeze_refresh_20260613/{REFRESH_REPORT.txt,
> refreshed_constants.json, PROPOSED_SPEC_DIFF.txt}`. The `[DRAFT@…]` stamps
> in the body are retained for history; the freeze block above governs.

Evidence base (all cited; nothing new asserted):
- `docs/research_program/PROGRAM.md` §H4 — original framing: restricted Nash
  response vs measured field tendencies, anchored to champion; unlock ≥ 500
  opp-observed hands; NO training until unlock + probe + operator pod approval.
- `docs/research_program/FIELD_DOSSIER.md` — graded field constants (consumed
  in §2). Generated at counter 272; §5 refresh is mandatory.
- `evals/d2_ci_vs_n_20260612/REPORT.txt` — **BINDING estimator policy:
  field-pooling is the only viable estimator** for depth-conditioned reads;
  per-opponent reads are session-length-capped and never decision-grade
  in-band; fold-based statistics are uncomputable until OQ-1 lands [B2].
- `data/opponent_db/README.md` — observability limits: frequencies are lower
  bounds [B1], folds ~100% missed [B2], opponent showdown holdings
  structurally unobservable, timing structurally absent (dossier §4).
- `reports/EXP_H2_killphil_league.md` — **BINDING LESSON: fine-tuning from a
  SLIM checkpoint (empty reservoirs) with 30% off-policy opponents collapsed
  the policy** (self-anchor z=−9.9, broad regression, partial healing as the
  buffer matured). §3 of this spec treats buffer continuity as a design
  REQUIREMENT, not a footnote.
- `docs/research_program/BBNORM_TRANSPLANT_SPEC.md` §5.3 — the numeric
  fold-rate regression-guard pattern, reused verbatim in §4.4. (That spec is
  otherwise VOID — its precondition, H2 probe PASS, failed.)
- `docs/research_program/ENSEMBLE_PROBE_DESIGN.md` +
  `evals/f1_ensemble_probe_20260612/REPORT.txt` — what the ensemble track has
  falsified: NO zero-training-cost bubble specialist exists (S1 dead:
  ckpt_1300 −0.058, ckpt_1400 −0.019, attacker_bubble_v1 −0.078, all FAIL
  ≥ +0.040 @ z≥2); selector seam P0-validated bit-identical. Consequence for
  H4: exploitation gains must come from TRAINING against a target, not from
  recombining existing checkpoints — and the validated EnsemblePolicy seam is
  available later if an H4 artifact wants override-region composition.

---

## 1. Hypothesis (falsifiable)

> The champion was trained and gate-tested exclusively against self-play and
> Shanky-profile opponents. The LIVE field, as measured by the H3 archive, is
> a different animal: position-invariant ~23% VPIP, limp-heavy/min-raise-heavy
> entry (no standard 2.2–2.5x sizing), a hard jam regime below 10–15bb with an
> 8.5% per-seat-hand 5–15bb jam rate, and a 23% open-jam share. A restricted
> Nash response trained against a field model matching these measured
> marginals — anchored to the champion by the self-play mass and a
> self-anchor gate — earns EV against the field model that transfers to
> field-like opponents WITHOUT degrading the champion's panel, self-play, or
> attacker-resistance properties.

Two separable sub-claims (mirroring f1's S1/S2 discipline):
- **S1 (learnable):** field-mix training moves the policy's responses toward
  the field-oracle on a frozen battery while the self-anchor holds. Tested at
  probe scale (§4).
- **S2 (it transfers and is safe):** the trained candidate beats the champion
  against the field mixture AND passes the full unweakened program battery
  (PG1–PG4 + ATT). Tested only after S1 passes, at full-retrain scale.

---

## 2. The RNR target — the FIELD model

### 2.1 Why a field model and not per-opponent models (BINDING, from d2)

d2's verdict is adopted wholesale: per-opponent VPIP is decision-grade
(±15pp) for ~half of opponent keys only after ~26–39 observed hands — a
late-session signal on long matches; **no per-opponent depth-conditioned read
ever becomes decision-grade within a session at live data rates** (best key
in the archive barely grazes ±15pp once); n-per-key is capped by session
length, not archive size, so waiting does not help. The anonymity constraint
forbids cross-session linkage. Therefore the H4 opponent is a SINGLE FIELD
DISTRIBUTION; per-opponent adaptation remains Layer-4's (separate) job, and
this spec deliberately contains no per-opponent machinery.

### 2.2 Target marginals (from FIELD_DOSSIER; refresh per §5 before freeze)

Only **chip-moving, decision-grade-or-better** marginals are targets. Nothing
fold-based, check-based, timing-based, or showdown-range-based may be a
target (d2 [B1]/[B2], dossier §4, d3) — those are structurally unobserved,
not noisy.

| id | marginal | target value `[DRAFT@272]` | dossier grade |
|---|---|---|---|
| T1 | field VPIP (lower bound) | 20.6% (±1.9pp) | decision [B1] |
| T2 | positional shape EP→BTN | FLAT, 22.7–24.5% (no widening) | decision (flatness) |
| T3 | SB elevated / BB defend-floor | SB 27.8% / BB ≥ 9.1% | decision / provisional |
| T4 | open-size mix (non-allin opens) | 2.0x mode 55.1%; 3.0x ~16%; 5x+ tail ~10%; 2.2–2.5x ~5% | decision (2x dominance) |
| T5 | open-jam share of first-raises | 23.0% | decision |
| T6 | preflop entry mix | limps = 63% of preflop calls | decision (observed split) |
| T7 | 5–15bb preflop jam rate / seat-hand | 8.5% (±2.8pp); 11.8% @5–10, 22.2% @0–5 | decision (band) / provisional (sub-bands) |
| T8 | jam-regime break | between 10 and 15bb (jam share of voluntary actions 60–78% below 10bb → ~13% at 10–15) | decision (boundary) |

Priors, NOT match targets (provisional grade — used only to sanity-check the
implemented model, never moment-matched): small-bet+overbet bimodal flop
sizing, polar/jammy 3-bets (26 events), tight-3x vs standard-minraise session
clusters.

### 2.3 Implementation: REWEIGHTED LEAGUE POOL (chosen), not a scripted synthetic

Two candidate implementations were considered:

**(a) Scripted synthetic field profile** — a new parametric scripted policy
playing the T1–T8 frequencies directly (position-flat VPIP, {limp, 2.0x,
3.0x, jam} sizing mix, depth-banded jam rates).
*For:* hits the measured marginals exactly by construction.
*Against — decisive:* the marginals do not determine a policy. A synthetic
must invent (i) WHICH hands fill each frequency (range composition —
showdown holdings are structurally unobservable, d3), (ii) ALL
fold/continuation behavior (uncomputable until OQ-1 [B2]), and (iii) all
postflop play beyond a provisional flop histogram. Deep CFR will
best-respond to precisely the invented parts — RNR vs a fiction, maximally
exploiting dimensions we never measured. This is the highest-risk failure
mode of the whole hypothesis and (a) walks straight into it.

**(b) Reweighted league pool (CHOSEN)** — select a subset of the 31 Shanky
profiles (`data/shanky_profiles/`, post-a8933ea adapter) and assign sampling
weights so the POOL'S aggregate behavior matches the T1–T8 marginals;
inject via the existing, H2-surviving league machinery
(`league_mix`/`league_registry_path` in `solver6.py`; scripted decisions
never enter the strategy buffer).
*For:* every pool member is a coherent full policy — its fold, continuation,
and postflop behavior is internally consistent rather than invented slot by
slot, so the unmeasured dimensions are filled with *plausible whole players*
instead of fabrications; a mixture-of-players also matches what the field IS
(dossier §5: heterogeneous table types), where a single averaged bot would
be a player who does not exist; zero new opponent-engine code (the pool,
registry, and uniform/weighted sampling seam exist; H2's infra survives as
instrument).
*Against (accepted, mitigated):* match quality is limited by the pool's
span — no profile may individually be position-flat-23%-min-raise-limpy;
mitigation is moment-matching at the POOL level (§2.4) with a pre-stated
residual tolerance and a thin-overlay fallback. Pool members are
"profile-as-adapter-plays-it" (e2 caveat) — accepted: the M1 oracle (§4.2)
is defined against the pool AS IMPLEMENTED, so target and oracle cannot
disagree about adapter semantics.

### 2.4 Pool construction procedure (P1 build item; runs before the probe)

1. **Measure** each candidate profile's own marginals with a synthetic-context
   sweep of the profile runtime (`evaluate_profile` over the 169 canonical
   classes × seeded (position, depth, level) cells — the exact method the H2
   battery oracle used) plus a short instrumented self-play row where the
   sweep cannot reach a marginal (entry mix, open-size histogram). File-only
   where possible; any simulation queues per the core ledger.
2. **Solve** for weights `w` over a candidate subset minimizing weighted L1
   distance to T1–T8 (decision-grade rows weighted 1.0, provisional rows 0.3),
   subject to: ≥ 4 profiles in support; max single weight 0.40 (no
   single-profile RNR — that was H2, and it is not this experiment); at least
   one limp-heavy member, one min-raise member, one jam-regime member
   (killphil-class) so every target dimension has a carrier.
3. **Acceptance tolerance (pre-stated):** pool aggregate within the dossier's
   own Wilson 95% CI for every decision-grade target T1–T8. If infeasible
   within the 31-profile span: add ONE thin scripted overlay member carrying
   only the unreachable marginal (e.g. a pure position-flat min-raiser),
   weight ≤ 0.25, with its unmeasured dimensions delegated to fold=never /
   call-down=profile-median documented explicitly — and the §7 fiction attack
   re-run against it before freeze.
4. **Freeze** the registry (`configs/league/registry_h4_field.json`) with
   member shas + weights + the measured pool-aggregate table. Requires the
   small `league_sample_strategy: weighted` extension (uniform is what
   exists; build item, ~1 h + tests, bit-identity at weights=uniform).

### 2.5 The RNR knob

`league_mix` IS the RNR p: with probability p per traversal all
non-traverser seats play the field pool; with 1−p, self-play. The anchor is
the self-play mass plus the §4 self-anchor gate. **`[DRAFT@383]` p = 0.30**
(the only value with measured infrastructure precedent — H2 probe and the
bbnorm spec both used 0.30). Pre-registered contingency: if OQ-1's fold fix
has NOT landed at freeze time, cap p at **0.25** — the field model's fold
dimension is then entirely profile-inherent rather than measured, and the
anchor must carry more of the safety burden (§7 attack 2). No mid-run p
changes; p is frozen with the spec.

---

## 3. Training design — H2-fragility mitigation as a DESIGN REQUIREMENT

**REQUIREMENT (from EXP_H2, binding):** no H4 training run may fine-tune
from a slim (buffer-less) checkpoint with off-policy opponents in the mix.
The deployed champion checkpoint `b79e82dd` is SLIM (10.9 MB, no reservoir
buffers — H2 spec §4 noted deviation); the H2 collapse signature
(self-anchor z=−9.9, M1 worsening 1800→2000 healing as the buffer matured,
v2's same-shaped independent failure) points at exactly this. Every H4 arm
must use one of:

**Option A — full-buffer checkpoint creation first (probe path).**
Rebuild the champion's reservoirs before any field exposure: resume from
ckpt_1500 with `league_mix=0`, run ~300–500 pure self-play iterations so
the advantage/strategy reservoirs refill from on-policy traversals, save a
FULL checkpoint (the checkpointer already serializes all buffers — verified
bit-identical resume), and only then switch the field mix on, resuming from
that full checkpoint.
*Cost:* rebuild ~1.3–2.4 h Contabo (300–500 iters @ 15.0–17.5 s/iter, G=8,
H2-measured) + ~1 h harness/verification. *Residual risk:* the rebuilt
buffer reflects the FROZEN champion's play, not the original training
trajectory — a continuation, not a bit-continuation; recorded, and the
self-anchor bar is the arbiter of whether it was good enough.

**Option B — league-from-iter-0 / two-phase from scratch (full-retrain path).**
Phase 1: pure self-play from iter 0 to ~1500 (champion recipe, k200_real_ante
config shape, NEW seed), producing own FULL checkpoints. Phase 2: resume own
ckpt_1500 (buffers intact) with `league_mix=p` + the §2.4 registry, 500–1000
iters under a stop rule (bbnorm §2 pattern: per-100-ckpt battery + field-row
eval, stop on 3 consecutive non-improvements, hard cap 2500).
*Cost:* ~8.3–12.2 h Contabo (2000–2500 iters) or **~4.7–5.8 h pod** (G=12,
8.37 s/iter measured) — pod-class per standing constraint 2.
*Residual risk:* phase 1 re-rolls the self-play dice (a new champion-recipe
run at a new seed may not reproduce champion strength); mitigated by the
phase-1 endpoint check (panel-neutral vs champion before phase 2 may start).

**Decision `[DRAFT@383]`:** probe uses Option A (cheapest H2-compliant
continuation; also the clean re-test of "does league exposure teach at all"
with the buffer confound removed). Full retrain, if the probe passes, uses
Option B on the pod (POD_REQUEST.md; track STOPS for operator per
Addendum 2). If Option A's rebuild itself fails its sanity check (post-
rebuild self-anchor row outside |z|<2 BEFORE any field mix), the probe falls
back to a from-scratch miniature (Option B at reduced iters) and the failure
is reported — that result alone would sharpen the H2 mechanism question.

**Encoder discipline:** champion-lineage 236-d legacy encoder ONLY. The
bbnorm transplant spec is void (precondition failed) and one-delta-per-
experiment discipline holds: H4 tests the field-target delta, nothing else.
Combining bbnorm + field-mix would require its own registration.

---

## 4. Falsification plan — probe-before-program (Addendum 3.2)

Structure mirrors H2 §2–§5: frozen battery metric + CRN panel rows +
self-anchor + numeric regression guard. ALL bars `[DRAFT@383]`; the freeze
block (§5) replaces them with values computed from refreshed baselines
BEFORE the probe launches. No seed re-rolls, no threshold shopping, one
probe run; one infrastructure relaunch allowed, recorded.

### 4.1 Instruments and baselines to freeze

- **Field-exploitation battery (BUILD, ~2 h + ~1 h compute):**
  `scripts/field_battery.py`, same architecture as the proven
  `fold_vs_shove_battery.py`. A FIXED file of ≥ 2,000 preflop decision spots
  drawn from the field model's characteristic action classes: (a) hero
  facing a single 2.0x open (the 55% mode), (b) hero in blinds behind 1–2
  limpers (T6), (c) hero facing a 5–15bb open-jam (T7 — reuses the existing
  battery's spot generator), (d) hero first-in vs position-flat callers,
  across levels L3–L7, all 169 hand classes × seeded (depth, position,
  level) cells. **Oracle = argmax_{action} ICM-EV against the POOL AS
  IMPLEMENTED** at the §2.4 frozen weights — pool members' ranges enumerated
  by querying the profile runtimes over the 169 classes (H2's oracle
  method), mixture-weighted. Battery + oracle labels written to disk ONCE
  and frozen before the probe; graded against the file, not regenerated.
- **Field-mixture row:** `sng_baseline`-format row where all 5 opponent
  seats sample from the frozen weighted pool per game (CRN master seed 2026,
  2000 games). Champion baseline on this row recorded pre-probe.
- **CRN panel hold rows:** killphilmtt, ticketmaster, sng, tighttom (2000
  games each, post-a8933ea baselines per `evals/e2_rebaseline_20260612/`:
  −0.0800 ± 0.0223 / +0.5400 ± 0.0188 / +0.8360 ± 0.0123 / +0.7780 ±
  0.0140). At least TWO hold rows must be profiles OUTSIDE the H4 pool
  support (out-of-model robustness); if killphil-class enters the pool, add
  replacement out-of-pool holds at freeze.
- **Self-anchor row:** 2000 games, hero = candidate, all opponents =
  unmodified champion `b79e82dd` — the RNR safety property made measurable.
- **H2 fold-vs-shove battery (frozen file):** retained as a guard metric;
  champion M1 = 0.0867 ± 0.0007.

### 4.2 Probe (Option A) — pre-registered bars `[DRAFT@383]`

Probe = buffer rebuild (§3.A) + **500 iterations** at `league_mix=p`, pool =
frozen §2.4 registry, G=8, Contabo, tmux + watcher, benchmark 1–2 iters
before committing (standing rule). Grades on probe ckpt_0500 (post-rebuild
numbering):

| id | metric | PASS requires | else |
|---|---|---|---|
| F-B1 | field-battery EV-loss vs frozen oracle file | ≥ **25% relative drop** vs champion baseline (H2's F-M1 bar shape) | probe FAIL |
| F-B2 | field-mixture row, CRN 2026 | ≥ **+0.05 net/game** over champion baseline AND ≥ 2σ_diff | probe FAIL |
| F-H1 | each hold row (4 rows incl. ≥2 out-of-pool) | degrades ≤ 2σ_diff vs e2 baseline | probe FAIL |
| F-H2 | self-anchor row | \|net\| < 2σ of 0 (H2's exact bar — the one H2 broke at z=−9.9) | probe FAIL |
| F-H3 | fold-vs-shove battery M1 | ≤ champion 0.0867 + 0.0020 (no shove-defense regression) | probe FAIL |
| F-G | §4.4 regression guard, reduced 6,000-game floored A/B at ckpt_0250 and _0500 | no bin cap exceeded by >1.5× | tripwire kill |
| F-iter | losses finite; premium-fold alert silent | sanity | kill |

Any FAIL ⇒ H4 dies at probe; full negative report (Addendum 1.2); the
battery, pool registry, and field-mixture row survive as permanent
instruments. ALL PASS ⇒ `POD_REQUEST.md` (Option B full retrain, §3) and the
track STOPS for the operator.

### 4.3 Full-retrain gates (S2; only after probe PASS + pod approval)

The candidate faces the FULL unweakened program battery: PG1 yardstick +
24-panel mean, PG2 bubble edge, PG3 calibration, PG4 bridge replay
(live-servable convention, floor-chain composition incl. armed tail floor),
ATT attacker re-extraction — bars exactly as PROGRAM.md, frozen baselines
re-recorded post-fix at freeze time — PLUS §4.2's F-B/F-H rows at full scale
and §4.4 in full (24,000-game floored A/B). Same gates forever, no novelty
discount.

### 4.4 Numeric regression guard (reused from BBNORM_TRANSPLANT_SPEC §5.3)

Instrument: candidate floored arm, per-decision logs, facing-action
decisions binned by `hero_eff_bb`; reference = deployed floored arm as
computed in `evals/a3_v2_decomposition_20260612/analysis.json`
(fold/allin reference table reproduced there; n ≥ 13k/bin). **FAIL if ANY
of** (caps copied verbatim — they were set before any candidate existed):
- Δfold(15–25bb) > +0.04 or Δfold(>25bb) > +0.04
- Δfold(6–10bb) > +0.08 or Δfold(10–15bb) > +0.08
- Δfold(0–4bb) > +0.10 or Δfold(4–6bb) > +0.10
- Δallin(10–15bb) < −0.04 or Δallin(15–25bb) < −0.04

**H4-specific additions `[DRAFT@383]`** (the field is limp-heavy and
passive; the symmetric failure mode here is AGGRESSION inflation — learning
to attack limpers indiscriminately, which over-fits the pool's passivity):
- Δallin(15–25bb) > **+0.04** or Δallin(>25bb) > **+0.04** ⇒ FAIL
- Δraise-mass(>25bb, facing no action) > **+0.10** ⇒ FAIL
Plus the EV-level mid-depth guard (a3 method): pooled paired-Δ z at 6–12bb
AND 12–25bb each > −3.0 vs the refreshed panel baseline.

---

## 5. Data-refresh checklist at freeze time (mandatory, in order)

- [ ] Counter ≥ 500 confirmed in `data/opponent_db/FIELD_REPORT.txt`.
- [ ] **Recompute every dossier constant** behind T1–T8 on the full corpus
      (sqlite-only, dossier methodology unchanged); regenerate the marginals
      table in §2.2 with new Wilson CIs and grades per the d2 rubric
      (±5pp field bar). d2 projects ±1.4pp on field VPIP at unlock.
- [ ] **CI check per d2:** every decision-grade target must remain
      decision-grade; any constant that moved OUTSIDE its dossier CI ⇒
      re-run §2.4 moment-matching and re-freeze the registry before
      proceeding. (Per d2, sampling error is already below [B1]/[B2] bias —
      large moves indicate composition shift, not noise.)
- [ ] **3-bet block:** at ~500 the 3-bet sample (~26 events @272) may
      graduate from anecdote; if a 3-bet frequency reaches provisional-or-
      better, add it to the moment-match as a 0.3-weight row, else leave out.
- [ ] **OQ-1 status check:** if the Windows folded-flag fix has landed and
      ≥ 3 post-fix sessions exist, compute fold-vs-shove / fold-to-steal on
      post-fix data; if decision-grade, add as targets and lift the p=0.25
      cap (§2.5); a MATERIAL fold-target addition re-opens §7 review before
      freeze. If not landed: p capped, [B2] disclaimer carried.
- [ ] Session-archetype re-check (dossier §5): if the tight-3x vs
      standard-minraise clusters have separated at ≥ 30 raw-record sessions,
      record the two-mixture refinement as FUTURE WORK — this spec still
      trains vs the single pooled field (first refinement, not this run).
- [ ] Champion identity unchanged (sha `b79e82dd…`); e2 hold-row baselines
      still authoritative (re-record only if any adapter/harness commit
      touched them since 2026-06-12).
- [x] Freeze refresh ran 2026-06-13 (n=517): §2.2 regenerated, all T1-T8
      inside dossier CI; OQ-1 landed-but-subgrade (p stays 0.25); registry
      not re-frozen. Evidence: evals/h4_freeze_refresh_20260613/.
- [x] Replace every `[DRAFT@…]` stamp; write the freeze block (date, counter,
      registry sha, battery sha, bar table); append registration to
      EXPERIMENT_LOG **before** the probe launches.

## 6. Cost table

Anchors: H2 probe 15.0–17.5 s/iter (G=8, contended Contabo); pod 8.37 s/iter
(G=12, 27-core, measured); 2000-game CRN row ≈ 7–8 min/row hot (yardstick);
gate battery envelope from BBNORM §6.

| item | Contabo (12 vCPU, contended, nice 19 / cores per ledger) | 27-core pod |
|---|---|---|
| P1 pool measurement + moment-match + registry freeze | ~2–3 h (mostly file-only; sweep + 1 short instrumented row) | — |
| P2 field battery build + oracle + champion baselines (battery, field row, holds reuse e2) | ~2 h dev + ~1.5 h compute | — |
| P3 buffer rebuild (Option A, 300–500 iters) + sanity anchor row | ~1.5–2.5 h + ~0.5 h | — |
| P4 probe (500 iters @ p=0.30) + per-ckpt tripwires | ~2.1–2.4 h + ~1 h evals | — |
| P5 probe measurement (battery + 6 rows) + report | ~1.5–2 h + 1 h analysis | — |
| **Probe total** | **~9–12 h** (fits a night shift) | not pod-class |
| Full retrain (Option B, 2000–2500 iters, only after probe PASS + approval) | 8.3–12.2 h (occupies the box — discouraged) | **4.7–5.8 h** |
| Full gate battery (PG1–PG4 + ATT + §4 rows + 24k A/B) | ~5.5–7 h | ~1.5–2 h |
| **Full-program total** | ~23–31 h | **~6–8 pod-h** + ~12 h Contabo (probe + analysis) |

## 7. Adversarial review (self-conducted, per Addendum 1)

- **R1 — "The field model is a fiction averaging incompatible opponents."**
  Dossier §5 shows real heterogeneity (tight-3x vs standard-minraise
  clusters, one loose outlier); a best response to the AVERAGE can be wrong
  against EVERY actual table. DISPOSITION: three mitigations are structural,
  not hopeful. (i) The target is a MIXTURE of whole coherent players (§2.3b),
  not one averaged bot — traversals see individual members, so the response
  is trained against the spread, not the centroid. (ii) The RNR anchor
  (1−p = 0.70 self-play mass + the F-H2 self-anchor bar) bounds the price of
  modeling error by construction — that is the entire reason this is RNR and
  not a best response. (iii) Hold rows include out-of-pool profiles spanning
  styles. Residual risk accepted and stated: if the field is bimodal enough,
  the single-prior RNR under-exploits both modes; the two-mixture refinement
  is registered FUTURE WORK gated on ~30+ raw-record sessions (§5).
- **R2 — "Frequencies without fold-stats bias the target" (OQ-1 pending).**
  Correct and unfixable by more hands [B2]: the field's fold behavior is
  unmeasured, and steal-EV vs a folding field is plausibly the LARGEST
  exploitation term — the model's fold dimension is currently whatever the
  Shanky members happen to do. DISPOSITION: (i) the pool choice (§2.3) means
  fold behavior is at least internally coherent per member rather than an
  invented constant; (ii) p is capped at 0.25 if OQ-1 hasn't landed at
  freeze (§2.5) — the anchor explicitly carries the unmeasured dimension;
  (iii) the §4.4 aggression-inflation caps directly fence the failure mode
  of over-learning steals against pool-fold artifacts; (iv) the §5 checklist
  upgrades the target the moment post-fix fold data is decision-grade.
  [B1] residual: all VPIP-class targets are lower bounds — the true field is
  LOOSER than the model; direction noted: it biases H4 toward
  under-exploitation, which the anchor tolerates, not toward unsafety.
- **R3 — "This is just H2's league mix with more profiles — and H2 FAILED."**
  The distinction must be earned, not asserted. (i) Target provenance: H2's
  pool was motivated by the worst panel row (one exploiter); H4's pool is
  moment-matched to measured live-field marginals with graded CIs —
  exploit-the-field, not patch-a-row. (ii) The RNR property: H2 had no
  self-anchor DESIGN element (only a gate that failed); H4 makes the anchor
  load-bearing (0.70 self-play mass, anchor bar at probe AND retrain,
  fragility requirement §3). (iii) H2's mechanism question was left open —
  league-teaching failure vs slim-buffer fragility, evidence leaning
  fragility. H4's probe is the clean re-test: buffer-safe by design, so a
  same-shaped collapse would falsify league-teaching generally (informative
  either way), while a pass localizes H2's failure to the buffer. HONESTY
  CLAUSE: if the H4 probe fails F-H2 (anchor) despite Option A, the
  successor claim "league training works if buffers are handled" is DEAD for
  this stack and may not be re-run without new evidence.
- **R4 — "The battery oracle is circular: it grades the candidate against
  the pool we trained it on."** True by design (the oracle is "field-model-
  optimal", not "field-optimal") — the SAME scoping H2 used for its
  killphil-oracle (H2 §7 A1). The transfer question is carried by F-B2
  (mixture row, EV not oracle-agreement), the out-of-pool holds, and
  ultimately the live dry-run after any ship. The battery answers
  "did it learn the target", nothing more; the spec never claims more.
- **R5 — "Adapter-fidelity: pool members are profiles-as-adapter-plays-them"
  (e2: stilltoact heavy users GusHansen/Millennium/itmstrike).** Accepted
  and made consistent rather than fixed: target, oracle, and training all
  use the SAME post-a8933ea adapter semantics, so there is no instrument/
  target mismatch inside the experiment. Moment-matching (§2.4) measures
  members AS PLAYED, so the pool aggregate matches the dossier regardless of
  PPL-vs-adapter drift. Residual: "as-played" may differ from the live
  field's actual humans — that is R1/R2's risk, already dispositioned.
- **R6 — "Champion may already be field-proof; the whole experiment buys
  nothing."** Possible: the panel contains loose/limpy profiles and the
  champion crushes most rows. This is exactly what the CHEAP pre-probe
  numbers answer: the champion's baseline on the field-mixture row and the
  field battery (P2, ~1.5 h) are measured BEFORE the probe is committed —
  if the champion's field-battery loss is already near-oracle and the
  mixture row near its panel mean, H4 stops at P2 for the cost of an
  evening, reported as a (valuable) negative. Pre-registered no-go:
  champion battery EV-loss < **0.02/spot** `[DRAFT@383]` ⇒ do not launch
  the probe; re-scope.

## 8. Registration checklist (fill at execution, in order)

- [ ] §5 data-refresh checklist complete; freeze block written; bars frozen
- [ ] EXPERIMENT_LOG registration entry BEFORE probe launch
- [ ] P1 registry frozen (`configs/league/registry_h4_field.json` + weights + sha)
- [ ] `league_sample_strategy: weighted` extension landed, bit-identity at uniform
- [ ] P2 battery + oracle frozen to disk; champion baselines recorded; R6 no-go checked
- [ ] P3 buffer rebuild + full checkpoint + anchor sanity row
- [ ] P4 probe launch (config sha + git head in run header; benchmark first; tmux + watcher)
- [ ] P5 verdict vs §4.2 bars; full report `reports/EXP_H4_rnr_field.md`; RESEARCH_MAP d1 update
- [ ] If PASS: POD_REQUEST.md (Option B); track STOPS for operator (Addendum 2)
