# RESEARCH MAP — open questions by pillar

Living document (Addendum 1.3). Per question: evidence state → decisive
experiment → cost → expected value of information (EVoI). Updated after every
experiment. Initialized 2026-06-12.

## (a) Representation

**a1. Where is the k=200 abstraction ceiling?**
Evidence: anchor run plateaued iter 500–1000 at k=200 (Session 5);
k=1000 abstraction FALSIFIED as a fix (dead hypothesis — gates
`evals/k1000_gate_iter*.json`); champion still has the killphilmtt hole.
The plateau is real but its cause (card abstraction vs action abstraction vs
training dynamics) is not isolated.
Decisive experiment: bucket decision-entropy "blur map" (exploratory,
pre-approved): which buckets/streets/depths carry the highest policy entropy
and the heaviest EV swings; cross-reference with where attacker extraction
concentrates. Cost: ~3–5 h read-only. EVoI: HIGH — tells us whether
representation is even the binding constraint before any rebuild talk.

**a2. What would a ReBeL-class (belief-state + search) rebuild require for
THIS format (6-max SNG, ICM, CPU live serving)?**
Evidence: real-time resolver track CLOSED net-negative (Floor lock,
2026-05-31) — but that was leaf-value precision on a depth-limited resolve,
not a learned belief-value architecture. No current assessment exists.
Decisive step: literature-grounded design assessment (paper study + sizing:
belief-state dimension for 6-max, value-net training compute, live-latency
budget vs the sub-second constraint). Cost: ~4–6 h reading/writing, no
compute. EVoI: MEDIUM now, HIGH if a/b experiments show representation binds.

**a3. Why didn't the depth cure translate to EV?** v2's bbnorm encoder PASSED
the depth-invariance probe (median TV ratio 0.004 vs deployed 1.55–3.16) yet
FAILED killphilmtt (−0.198), calibration (2.55σ), and the short-stack-floor
A/B collapse test. Either depth confusion wasn't the EV-binding defect, or
v2's training run had an independent regression.
Decisive experiment: per-depth-bucket paired EV decomposition of v2 vs
champion from the existing gate artifacts (read-only). Cost: ~2–4 h.
EVoI: MEDIUM-HIGH — decides whether the bbnorm encoder is an organ worth
transplanting (frontier pillar) despite v2's failure.
**ANSWERED 2026-06-12 (`evals/a3_v2_decomposition_20260612/`):** the
depth cure DID pay EV where the confusion lives — paired delta at
<=6bb terminal depth **+0.130/game (z=+11.8, 15/19 profiles positive)**;
mid-depth negative. The killphil hole is **depth-flat** (all |z|<=0.61)
— a policy defect, not representation (strengthens H2's premise). v2's
gate failures are an independent broad-spectrum over-folding regression
(folds-facing-action elevated at EVERY depth; mid-depth losses vs
pressure profiles). **f-pillar: bbnorm encoder upgraded to "economically
signed at shallow" — keep as transplant organ** (one-run confound
noted). Caveat: terminal-depth proxy (per-hand depth not recorded in
G2/G3 artifacts; cheapest harness fix documented in the report).

## (b) Training dynamics

**b1. Is the killphilmtt hole a self-play monoculture artifact?** =
**H2 — CLOSED AT PROBE, FAIL (2026-06-12,
`reports/EXP_H2_killphil_league.md`).** The 500-iter league probe
collapsed broadly (self-anchor −0.216, all bars broken); collapse
predates the live-window interruption (ckpt_1800 worse than final).
UNRESOLVED MECHANISM: league-teaching-failure vs slim-checkpoint
fine-tuning fragility (evidence leans fragility: M1 healed 1800→2000
as the reservoir matured; v2 showed the same broad-regression
signature). Successor H2b (full-buffer or from-iter-0 league run) is a
refill candidate on EVoI. The hole itself is REAL and depth-flat (a3);
the post-fix gate row is −0.0800 ± 0.0223. ORIGINAL ENTRY: Evidence: C3's deltas (encoder channel,
distribution fix) did NOT close it (C3 candidate −0.215); more iterations
falsified; the hole is specifically shove-defense at 5–15 BB.
Decisive experiment: H2 league probe (500 iters, league mix incl.
killphil-class exploiters). Cost: ~8–12 h CPU. EVoI: HIGH — directly gates
the only retrain hypothesis still alive.

**b2. Checkpoint-selection variance.** Gate verdicts ride single checkpoints
(iter 1500). Unknown: how much do gate metrics swing across adjacent
checkpoints of the same run? Cost: ~4 h (paired panel on 2–3 champion-run
checkpoints, deferred-class). EVoI: MEDIUM — calibrates how big a candidate's
edge must be to be real.

## (c) The tournament game itself

**c1. How wrong is Malmuth-Harville ICM for this format (top-3 equal
payout)?** Evidence: none measured — ICM is assumed throughout training and
eval. The known failure modes (ignores position/blinds/skill, equal-payout
flattening near the bubble) are exactly where our gates concentrate
(bubble edge).
Decisive experiment: empirical tournament-value probe (run 2026-06-12).
**ANSWERED — MATERIAL (`evals/c1_icm_gap_20260612/`):** B*=+0.0412
[+0.0286,+0.0539] in high-dispersion bubble states; MH underprices
short-stack survival (<5bb error +0.065), overprices mid stacks;
equal-stack states clean. HETEROGENEOUS (σ_bias 0.068) → correction
must be state-dependent (depth_bb first feature). Consumers flagged:
icm_adjust_returns, subgame leaves, H2-battery oracle, H1 floor evals
(equal-stack metrics expected robust). Follow-ups in refill: correction
fit + consumer re-price audit.

## (d) Opponent modeling & exploitation

**d1. Field tendencies (H3/H4 track).** ACTIVE — pipeline BUILT 2026-06-12
(`src/nlhe/opponent_db/`, `scripts/ingest_session.py`,
`data/opponent_db/`). 16 sessions / 467 hands ingested (2026-06-12: night
session 042058 added 35 hands / 19 opp-observed; 035454 re-ingest was an
idempotent replace); reconciliation vs the audited session-1 ground truth
exact (37 hands / 61 decision frames).
**H4 unlock counter: 383/500 opponent-observed hands — LOCKED but
~1-2 sessions out** (session 183701 alone: +87 opp-observed across 140
hands — far above the d2 per-session projection; 19 sessions total).
Per-opponent reads stay session-length-capped (d2). PREP: H4's spec
should be drafted BEFORE unlock — refill-pass candidate.
**d2. Pre-threshold field pre-analysis** (exploratory, pre-approved): first
cut exists in `data/opponent_db/FIELD_REPORT.txt` (opponent VPIP lower bound
20.2%, 506 voluntary actions). Remaining: CI-vs-n analysis for
decision-grade thresholds. Cost: ~1–2 h. EVoI: MEDIUM.
**d3. NEW — structural observability limits of the live archive
(2026-06-12, H3 build finding):** the scraper schema has NO opponent
hole-card field — showdown holdings are unobservable, so range inference
from showdowns is impossible without a Windows-scraper change; and the
stale `folded` flags make fold-vs-shove a systematic undercount (0/23
observed). Consequence: H4's RNR target must be built from
action-frequency tendencies (VPIP, raise sizes, all-in rates by depth),
NOT showdown-conditioned ranges, unless the scraper grows a showdown
capture. **2026-06-12: OQ-1 RESOLVED — operator approved both halves,
folded-fix prioritized; Windows brief at
`docs/WINDOWS_TASK_SCRAPER_FOLDED_SHOWDOWN.md`. Forensic re-measure
while packaging: stale rate is ~100% per fold event (0/241 chip-proven
preflop folds flagged), not the earlier ~30%-of-frames figure.** EVoI
of the scraper fix: HIGH for H4 fidelity (and H2's probe metric).

## (e) Measurement methodology

**e1. The attacker is abstraction-blind.** Both trained attackers use the
same k=200 abstraction as the champion → extraction ≤ 0 proves safety only
within the shared abstraction. A fine-card-abstraction attacker (k=1000
buckets or raw-ish features, attacker-only) would see exploits our instrument
can't. Cost: attacker retrain ~pod-class or long CPU. EVoI: HIGH for any
ship decision's confidence statement; currently our "Tier-0 exploitability"
claim carries this caveat.
**e2. Scripted-adapter fidelity:** `stilltoact` hardcoded 0
(`src/nlhe/scripted_bots/policy.py:308`) kills all `stilltoact>=k` rules in
every Shanky profile — panel rows measure profile-as-adapter-plays-it.
Matters for H2's league realism. Fix cost: ~2–4 h + re-baseline affected
rows. EVoI: MEDIUM-HIGH before H2's probe (audit first; fix if killphil-class
profiles depend on it).
**AUDIT DONE 2026-06-12 (killphil half): KillPhilMTT DOES depend on it** —
17/629 `when` rules use `stilltoact`, structured as position-tiered
unopened-push ranges: 9 `>=k` + 3 `=k` rules (early/mid-position TIGHTER
ranges) are DEAD; the 5 `<=1` rules (late-position LOOSEST ranges) fire
from EVERY position. Net: adapter-killphil shoves systematically looser
than real killphil in unopened pots — biases both the −0.174 gate row
(adapter-killphil is likely WEAKER than real) and any "killphil-optimal"
probe target. **H2 spec must either fix stilltoact first (derive from
dealt/folded/acted state — the adapter has the live view) or define
killphil-optimal against the adapter-as-played semantics explicitly.**
Heaviest users field-wide: GusHansen 2370 rules, Millennium 253,
itmstrike A/C ~107 — re-baseline those rows after any fix.
**e3. Harness seat-bias bound:** self-play calibration marginal-PASS
(−0.042 ± 0.0223, 1.88σ). Any conclusion under ~0.045/game is inside the
instrument's noise floor. Standing caveat on every panel verdict.

## (f) Frontier / proprietary

**f1. Shove-defense specialist override region** (hybrid policy): if H2's
probe shows a league-trained model fixes 5–15 BB shove-defense but costs EV
elsewhere, compose champion + specialist gated by (eff_bb ∈ [5,15] ∧ facing
all-in) — the floor-chain machinery is the natural injection point, and the
gate battery already measures both halves. Entry condition: H2 probe result.
Cost: probe-dependent. EVoI: HIGH conditional on H2.
**f2. Organ bank inventory:** v2 bbnorm encoder (depth-cured, a3 decides);
trained attackers as league sparring partners (B3/B5 attackers are exactly
"trained exploiters" for H2's mix); 31 scripted styles; league/PSRO
machinery; live archive (d1). No build without an evidence-cited rationale +
probe (Addendum 3).

## Exploratory backlog (pre-approved, idle-time)

1. **Tail-concentration map** — where do the champion's low-mass aggressive
   tails concentrate (street/depth/position)? Feeds H1 discussion directly
   (TG2 artifacts are the substrate). ~2 h.
2. **Blur map** (a1). ~3–5 h.
3. **Field-tendency pre-analysis** (d2). ~1–2 h after H3 ingest.
4. **v2 EV decomposition** (a3). ~2–4 h.
