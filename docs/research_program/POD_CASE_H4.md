# POD_CASE — H4 RNR Field-Exploitation Retrain

**STATUS 2026-06-13 (SUPERSEDED): the H4 probe FAILED at F-H2 (self-anchor
collapse −0.18/z−3.7; `reports/EXP_H4_probe_FH2_collapse.md`). The global-RNR
Option-B program described below is NOT triggered.** Root cause = a METHOD bug:
the "RNR" had no explicit restriction to the frozen champion, so the self-play
anchor co-drifted and the policy walked off the blueprint. **OUR DATA IS VALID;
OUR METHOD OF APPLYING IT WAS WRONG.** The pod case is no longer "run the
global RNR full retrain"; it is now **"probe the competing data-application
methods (C/A/B/D) and let the survivor earn the program"** — top of RESEARCH_MAP
(Q0). The buffer-rebuild requirement below stands and is RESOLVED (generate-only;
see [[EXPERIMENT_LOG]] 2026-06-13). Everything from here down is the original
(now-superseded) global-RNR case, retained for the record.

---

**Status: POD-RELEVANT, awaiting freeze-complete + operator go.** Written
2026-06-13 at H4 counter **517/500 (UNLOCKED, ingest committed to
`data/opponent_db/opponent_db.sqlite`: 23 sessions / 883 hands)**. This is the
**first training-scale job since H2** and the **pod restart trigger** — the
review's standing rule (COMPUTE_QUEUE §5 / OPERATOR_QUEUE Addendum 5.5) is to
surface a POD_CASE the moment a track needs training-scale compute. H4 now does.

Authoritative spec: `H4_RNR_FIELD_SPEC.md` (FROZEN 2026-06-13, commit 56edb84).
This document is the *case for the pod* (why, how much, what it must clear); the
actual launch ask (Option B) is `POD_REQUEST.md`, written only **after** the
probe passes (§4.2) and only with operator approval.

---

## 1. Why pod, why now

H3/H4's gating goal is met: ≥ 500 opponent-observed hands, so the field model is
measured (not projected) and the RNR target is freezable. The hypothesis (spec
§1) is that the champion `b79e82dd` was tuned against self-play + Shanky
profiles, while the **live field is measurably different** — position-invariant
~20% VPIP, limp/min-raise entry (no 2.2–2.5x standard open), a hard sub-10bb jam
regime, ~25% open-jam share — and a restricted-Nash response trained against a
field model matching those marginals earns transferable EV **without** degrading
the champion's panel / self-play / attacker-resistance.

The full retrain is **training-scale**: 2000–2500 Deep-CFR iterations. On
contended Contabo that is 8.3–12.2 h occupying the whole box; on a 27-core pod it
is **4.7–5.8 h at the measured 8.37 s/iter (G=12)**. That gap is the pod case.

**Not auto-launching.** Two gates precede any pod spend: (a) the **probe**
(§3 below) runs on Contabo and must PASS all of F-B1/F-B2/F-H1/F-H2/F-H3; (b)
operator go. The track STOPS for the operator after the probe (Addendum 2).

---

## 2. Cost / pod-hours

Anchors: H2 probe 15.0–17.5 s/iter (G=8, contended Contabo); pod 8.37 s/iter
(G=12, 27-core, measured); 2000-game CRN row ≈ 7–8 min hot.

| stage | where | cost |
|---|---|---|
| P1 pool measure + moment-match + registry freeze | Contabo (file-only mostly) | ~2–3 h |
| P2 field battery build + oracle + champion baselines | Contabo | ~2 h dev + ~1.5 h compute |
| P3 buffer rebuild (Option A, 300–500 iters) + anchor sanity row | Contabo | ~1.5–2.5 h + ~0.5 h |
| P4 **probe** (500 iters @ p=0.25) + per-ckpt tripwires | Contabo | ~2.1–2.4 h + ~1 h evals |
| P5 probe measurement (battery + 6 rows) + report | Contabo | ~1.5–2 h + 1 h |
| **Probe total (no pod)** | Contabo | **~9–12 h — fits a night shift** |
| **Full retrain (Option B, 2000–2500 iters)** — *only after probe PASS + go* | **POD** | **4.7–5.8 pod-h** (vs 8.3–12.2 h Contabo) |
| Full gate battery (PG1–PG4 + ATT + §4 rows + 24k A/B) | POD | ~1.5–2 pod-h |
| **Full-program pod total** | POD | **~6–8 pod-h** + ~12 h Contabo (probe + analysis) |

**Pod ask, when it comes, is ~6–8 pod-hours** (≈ $2–3 at RTX 4090
$0.34/hr-class, or the chosen 27-core provider). The probe is deliberately
**Contabo-only** so the pod is spent only on a hypothesis that already passed.

---

## 3. The bbnorm-organ decision — DECIDED: legacy 236-d encoder, train fresh

**H4-v1 does NOT carry the depth-cured (bbnorm) encoder. It uses the
champion-lineage 236-d legacy encoder, period.** (Spec §3, "Encoder
discipline.") Three reasons, all binding:

1. **One-delta-per-experiment.** H4 tests exactly one delta — the field-RNR
   target. Stacking the bbnorm transplant on top would confound the result;
   bbnorm + field-mix "would require its own registration" (§3).
2. **The bbnorm transplant spec is VOID** — its precondition (H2 probe PASS)
   failed, so there is no validated depth-cured organ to carry.
3. **The champion is a 236-d model** (`feature_dim=236`, confirmed in the live
   listener banner) and the entire frozen instrument suite — e2 hold-row
   baselines, the H2 fold-vs-shove battery, the §4.4 a3 regression reference —
   is recorded against 236-d. Changing the encoder would invalidate every
   baseline the gates compare against.

The bbnorm variant *could* ride a later pod run as its **own** registered
experiment; it is explicitly out of scope for H4-v1.

**Training path (H2-fragility-binding, spec §3):** no H4 arm may fine-tune from
a slim (buffer-less) checkpoint with off-policy opponents — that is precisely
what collapsed H2 (self-anchor z = −9.9). The deployed champion `b79e82dd` IS
slim (10.9 MB, no reservoirs). So:

- **Probe → Option A:** resume `ckpt_1500`, refill the advantage/strategy
  reservoirs, save a FULL checkpoint, *then* switch the field mix on. Cheapest
  H2-compliant path and a clean re-test of "does league exposure teach at all"
  with the buffer confound removed.
  - **REFILL MUST BE GENERATE-ONLY (2026-06-13, binding).** The original
    free-self-play refill is RETIRED for this purpose: it retrained the nets on
    a from-empty reservoir, re-averaging the strategy and moving the player —
    the post-rebuild self-anchor regressed 0.15/game (z=−3.1, plateaued), which
    is the H2 buffer-fragility mechanism manifesting *in the rebuild itself*,
    upstream of any league. Use `h4_buffer_rebuild.py --populate-only`: collect
    samples via FROZEN-net traversals, skip both gradient steps, so the player
    is provably unchanged (verified bit-identical, 901,439 params, max|Δ|=0) and
    the self-anchor gate passes by construction. See EXPERIMENT_LOG 2026-06-13.
    This sharpens the H2 finding: "league training works if buffers are handled"
    requires buffers reconstructed *policy-preservingly*, not via free self-play.
- **Full retrain → Option B (the pod job):** Phase 1 = pure self-play from
  iter 0 to ~1500 (champion recipe, k200_real_ante config shape, NEW seed),
  producing own FULL checkpoints; Phase 2 = resume own `ckpt_1500` (buffers
  intact) with `league_mix=0.25` + the §2.4 registry, 500–1000 iters under the
  bbnorm-§2 stop rule (per-100-ckpt battery + field row; stop on 3 consecutive
  non-improvements; hard cap 2500).

---

## 4. The moment-matched reweighted-pool target (frozen)

**Implementation: reweighted league pool, NOT a scripted synthetic** (spec §2.3,
decisive). A scripted profile would force us to invent range composition, all
fold/continuation behavior, and postflop play — exactly the unmeasured
dimensions Deep CFR would then best-respond to (RNR vs a fiction). The pool
fills the unmeasured dimensions with *plausible whole players* instead.

**Construction (§2.4):** select ≥ 4 of the 31 post-a8933ea Shanky profiles
(`data/shanky_profiles/`), solve sampling weights `w` minimizing weighted L1
distance to the frozen T1–T8 marginals (decision rows weight 1.0, provisional
0.3), subject to: max single weight 0.40 (no single-profile RNR — that was H2),
and ≥ 1 limp-heavy + 1 min-raise + 1 jam-regime (killphil-class) member so every
target dimension has a carrier. Freeze to `configs/league/registry_h4_field.json`
with member shas + weights + measured pool aggregate. Needs the small
`league_sample_strategy: weighted` extension (uniform exists; ~1 h + bit-identity
test at weights=uniform).

**The RNR knob (§2.5): p = 0.25 (FROZEN).** With probability p each traversal,
all non-traverser seats play the field pool; with 1−p = 0.75, self-play. The
anchor is the 0.75 self-play mass + the F-H2 self-anchor gate. p was capped at
0.25 (not the draft's 0.30) because OQ-1's fold fix landed but the post-fix
fold-vs-shove marginal (66.1%, ±8.2pp, 3 sessions) is **not yet decision-grade**
— so the fold dimension is profile-inherent and the anchor carries it. Lifts to
0.30 + a fold target at the next freeze once post-fix fold data reaches ±5pp
(~6 post-fix sessions).

### Frozen field target — what the retrain trains against (FROZEN@517)

| id | marginal | frozen value (n / Wilson hw) | grade |
|---|---|---|---|
| T1 | field VPIP (lower bound) | **20.1%** (689/3426, ±1.3pp) | decision [B1] |
| T2 | positional shape EP→BTN | **FLAT** — EP 20.8 / MP 21.4 / CO 26.7 / BTN 22.7 (CIs overlap; no EP→BTN widening) | decision (flatness) |
| T3 | SB elevated / BB defend-floor | **SB 27.9% / BB ≥ 9.0%** | decision / provisional |
| T4 | open-size mix (non-allin) | **≤2.0x = 57.8%** (159 at 2.0x); 3.0x 16.3%; 5x+ tail 10.4%; 2.2–2.5x ~3% | decision (2x dominance) |
| T5 | open-jam share of first-raises | **25.5%** (99/388, ±4.3pp) | decision |
| T6 | preflop entry mix | **limps = 63.7% of calls** (156/245, ±6.0pp) | decision (split) |
| T7 | 5–15bb jam rate / seat-hand | **8.4%** (68/812, ±1.9pp); 11.4% @5–10; 19.4% @0–5 | decision (band) |
| T8 | jam-regime break | **between 10–15bb** — jam share 59.3% @5–10 → 20.3% @10–15 (CIs disjoint) | decision (boundary) |

Field portrait, plainly: a **passive, limp-heavy, position-flat ~20% field that
min-raises (2x) when it does open and flips to a hard jam regime under 10bb** —
about a quarter of all first-raises are outright open-jams. The exploiter learns
to attack the limp/min-raise entry and to defend correctly against the sub-10bb
jam wall. **Every T1–T8 reproduced inside its dossier CI at n=517** (no
re-statement, no registry re-freeze); the within-CI drift since the 272-snapshot
is if anything *jammier* (T5 +2.5pp, T8 10–15 level up) — which strengthens, not
weakens, the §4.4 aggression-inflation guards.

The accepted residual (R2, [B2]): the field's true fold behavior is
sub-decision-grade, so steal-EV against a folding field — plausibly the largest
exploitation term — is carried by the pool members' own fold behavior, not a
measured target. The p ≤ 0.25 anchor and the §4.4 aggression caps fence the
matching failure mode (over-learning steals against pool-fold artifacts).

---

## 5. The full gate battery it must clear

The exploiter is held to **the same unweakened program bars, no novelty
discount** (spec §4.3). Two stages:

**Probe gate (Contabo, ckpt_0500, ALL must PASS or H4 dies at probe):**

| id | metric | PASS |
|---|---|---|
| F-B1 | field-battery EV-loss vs frozen oracle | ≥ 25% relative drop vs champion |
| F-B2 | field-mixture row (CRN 2026, 2000 games) | ≥ +0.05 net/game over champion AND ≥ 2σ_diff |
| F-H1 | 4 hold rows (≥ 2 OUT-of-pool) | each degrades ≤ 2σ_diff vs e2 baseline |
| F-H2 | **self-anchor row** (opps = champion) | \|net\| < 2σ of 0 — *the bar H2 broke at z=−9.9* |
| F-H3 | H2 fold-vs-shove battery M1 | ≤ champion 0.0867 + 0.0020 (no shove-defense regression) |
| F-G | §4.4 regression guard (6k floored A/B @ _0250/_0500) | no bin cap exceeded > 1.5× |

**Full-retrain gate (pod, after probe PASS + go) — the complete program:**

- **PG1** yardstick + 24-panel mean
- **PG2** bubble edge
- **PG3** calibration
- **PG4** bridge replay — live-servable convention, full floor-chain composition
  **including the armed tail floor** (H1 τ=0.10)
- **ATT — attacker re-extraction.** The exploiter must NOT have opened a new
  attack surface; re-run the attacker batteries against the candidate.
- **Shove-defense / shoviest panel (SLATE-2A, just-passed champion baseline:
  ALL ROWS POSITIVE, pooled +0.0108 ± 0.0023, z = +4.7).** The exploiter must
  still **not lose to known opponents** — a field-RNR that crushes limpers but
  bleeds to shovers is a net regression. This panel is the load-bearing
  "still-safe-vs-the-aggressive-tail" check.
- **§4.2 F-B/F-H rows at full scale** + **§4.4 numeric regression guard in full
  (24,000-game floored A/B)**, including the H4-specific aggression-inflation
  caps (Δallin(15–25bb) > +0.04 ⇒ FAIL; Δraise-mass(>25bb, no action) > +0.10 ⇒
  FAIL) and the mid-depth EV guards.

Baselines (panel, attacker, fold-vs-shove M1, e2 hold rows) are **re-recorded
post-fix at freeze/launch time** so the n≥500 refresh does not move the bars; the
bars themselves are champion-relative and unchanged.

---

## 6. Pre-registered no-go (don't even probe)

If the cheap pre-probe numbers (P2, ~1.5 h) show the champion is **already
field-proof** — field-battery EV-loss < 0.02/spot and the field-mixture row near
its panel mean (R6) — H4 stops at P2 for the cost of an evening, reported as a
valuable negative. The probe is committed only if the champion leaves
exploitation EV on the table.

---

## 7. Sequencing / dependencies before the pod

1. **Freeze: DONE** (spec FROZEN 56edb84; ingest committed this session).
2. **P1 registry** (`registry_h4_field.json`) + `league_sample_strategy:
   weighted` extension — Contabo, ~3–4 h, bit-identity gated.
3. **P2 battery + oracle + champion baselines** — Contabo, ~3.5 h; R6 no-go
   checked here.
4. **P3+P4+P5 probe** — Contabo night shift, ~9–12 h. Track STOPS for operator.
5. **Only on probe PASS + operator go → `POD_REQUEST.md` (Option B).**

The bridge/scraper capture fix (OQ-4, window-handle capture + felt-collapse
auto-pause) is **not a blocker** for the probe (probe is offline/self-play +
file batteries), but IS a precondition for trusting any *live* dry-run of the
shipped exploiter afterward.
