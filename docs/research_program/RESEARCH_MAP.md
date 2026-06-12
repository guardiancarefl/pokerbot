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

## (b) Training dynamics

**b1. Is the killphilmtt hole (champion −0.174) a self-play monoculture
artifact?** = **H2**, ACTIVE QUEUE. Evidence: C3's deltas (encoder channel,
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
Decisive experiment: empirical tournament-value probe — from N sampled
mid-tournament states, compare ICM-predicted finish-distribution vs
self-play rollout finish-distribution (the harness exists: self-play games
with fixed starts). Gap map by stack-config/level. Cost: ~6–10 h CPU
(deferred-class). EVoI: HIGH if the gap is material — it re-prices every
ICM-adjusted return in training; LOW if ICM holds (still worth a report).

## (d) Opponent modeling & exploitation

**d1. Field tendencies (H3/H4 track).** ACTIVE — pipeline BUILT 2026-06-12
(`src/nlhe/opponent_db/`, `scripts/ingest_session.py`,
`data/opponent_db/`). 16 sessions / 467 hands ingested (2026-06-12: night
session 042058 added 35 hands / 19 opp-observed; 035454 re-ingest was an
idempotent replace); reconciliation vs the audited session-1 ground truth
exact (37 hands / 61 decision frames).
**H4 unlock counter: 259/500 opponent-observed hands — LOCKED** (~4–6 more
sessions at current rates).
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
