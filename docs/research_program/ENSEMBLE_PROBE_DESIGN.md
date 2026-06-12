# ENSEMBLE PROBE — champion + bubble-specialist + trivial selector

**Pillar:** (f) frontier/proprietary, Addendum 5 slate item (c).
**Status:** DESIGN (pre-registration). Nothing in this document has run.
**License:** Addendum 3 — evidence-first, probe-before-program, same gates
forever. This is the cheapest falsifiable probe; the expected outcome is
falsification of the cheap-organ version (that is the system working).
**Author note:** file-only design pass 2026-06-12; all numbers below are
recomputed from committed artifacts, paths given inline.

---

## 0. Hypothesis (falsifiable)

> A composite policy — deployed champion everywhere, a bubble-specialist
> checkpoint on hands that start with exactly 4 players alive, selected by a
> zero-parameter deterministic rule — improves bubble-stage EV against the
> Shanky field without degrading EV anywhere else.

Two separable sub-claims, each independently falsifiable:

- **S1 (specialist exists):** some existing checkpoint plays n_alive=4
  hands better than `champion-b79e82dd` (ckpt_1500).
- **S2 (composition transfers):** swapping that checkpoint in only at the
  bubble improves yardstick rows where the champion's bubble bleed lives,
  without harming rows where the champion's bubble edge is positive.

If S1 fails for every zero-training-cost candidate, S2 is never run and the
residual question ("would a purpose-trained bubble specialist serve?")
becomes a *separate* pre-registered probe gated on the H2 chain — no
training happens inside this probe.

---

## 1. The trivial selector — spec and justification

### 1.1 Spec

```
selector(hand) := SPECIALIST  if hand-start n_alive == 4
                  CHAMPION    otherwise
```

Zero parameters, zero learned state, deterministic, evaluates in one integer
comparison. Both members share the champion's abstraction
(`runs/abstraction_20260521_223018_retrofit/abstraction.pkl`, sha
`0fc20800…`) and the 236-d encoder convention, so the selector switches
*checkpoints*, never representations.

**n_alive sources, per environment:**

- **Live path:** `make_decision()` already computes
  `out.n_alive = int(sum(frame.alive))` from the scraper's alive vector
  (`src/nlhe/integration/live_loop.py:809`) and logs it on every record —
  exact, replayable.
- **Eval harness:** the selector lives entirely inside a `Policy`-protocol
  wrapper (`EnsemblePolicy.select_action`), the same seam
  `CheckpointPolicy` / `FlooredCheckpointPolicy` / Shanky adapters use —
  **no harness modification**. It infers hand-start alive count from the
  parsed public state: `n_alive = #{i : money[i] + contribution[i] > 1}`.
  Busted seats are stack=1/ante=0 placeholders in the game string
  (`scripts/sng_baseline.play_one_hand_sng`), so their money+contribution
  total is exactly 1; live seats total their real hand-start stack; a
  mid-hand all-in real seat has money=0 but contribution>1. The count is
  invariant within a hand.
  - *Known collision:* a real seat entering a hand with a total of exactly
    1 chip (possible post-ante) is indistinguishable from a placeholder →
    alive UNDERCOUNT → specialist may fire on a true-5-alive hand
    containing a 1-chip stub. Direction is benign (such hands are
    bubble-imminent); P0 characterizes it with constructed
    `starting_stacks` via the test seam. The LIVE selector does not have
    this collision (scraper alive[] is ground truth).

### 1.2 Why n_alive == 4 — from our own measurements

Recomputed from the v1 yardstick stage decomposition
(`evals/sng_baseline_20260610/summary_merged.json`, champion b79e82dd,
24 profiles × 2,000 games, master_seed 2026, `stage_acc` keyed by
hand-start alive count — the scoring path documented in
`scripts/sng_baseline.py`):

| hand-start n_alive | hands | hero ICM Δ/hand | z | share of total edge |
|---|---|---|---|---|
| 6 | 936,931 | **+0.020305** ± 0.000135 | +150.0 | 82.6 % |
| 5 | 284,809 | **+0.011718** ± 0.000306 | +38.3 | 14.5 % |
| 4 (bubble) | 177,789 | **+0.003691** ± 0.000391 | +9.4 | 2.9 % |

The champion's per-hand edge at the bubble is **5.5× thinner** than at
6-alive, and the PG2 bar (+0.003691) *is* this measured mean — the bubble is
the champion's weakest stage by direct measurement. Per-profile, the bubble
stage is outright **negative against 6 of 24 panel members**:

| profile | net/game (row) | bubble Δ/hand | bubble hands |
|---|---|---|---|
| fixedlimitheadsup | +0.111 | **−0.00751** ± 0.00324 | 6,417 |
| loosenluckymtt | +0.278 | **−0.00489** ± 0.00230 | 6,668 |
| killphilmtt (worst row, PG1) | **−0.174** | **−0.00471** ± 0.00144 | 16,547 |
| modernmikecash | +0.729 | **−0.00450** ± 0.00390 | 1,245 |
| itmstrikea | +0.033 | **−0.00190** ± 0.00137 | 14,798 |
| itmstrikec | +0.044 | **−0.00163** ± 0.00137 | 14,831 |

Why not a wider or finer trigger:

- **Not n_alive ∈ {4,5}:** the 5-alive stage is strongly positive
  (+0.0117/hand, z=38) — replacing the champion there risks 14.5 % of the
  total edge to fix nothing.
- **Not (eff_bb ∈ [5,15] ∧ facing all-in):** that is f1's shove-defense
  override region, explicitly gated on the H2 probe outcome
  (RESEARCH_MAP f1); a3 showed the killphil hole is depth-flat policy, so
  the finer trigger needs H2's league evidence first. n_alive==4 is the
  cleanest boundary our instruments already measure (PG2 slice, bubble
  harvest artifact, B5 battery all use it), and a bubble-start game
  consists *only* of n_alive=4 hands — so existing bubble-mode evidence
  reads directly as ensemble-bubble behavior.

---

## 2. Specialist candidates, ranked

### Rank 1 — champion-run sibling checkpoints (ckpt_1300, ckpt_1400)

`runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_{1300,1400}.pt`
(only these survive besides 1500).

- **For:** zero training cost; identical conventions; RESEARCH_MAP b2
  (checkpoint-selection variance) is an open question — gate verdicts ride
  single checkpoints, and nothing has ever measured *bubble-stage* metric
  swing across adjacent checkpoints. The probe answers b2 for free at the
  stage that matters.
- **Against:** weak prior — same run, same training distribution, 100–200
  iterations apart; any bubble difference is plateau noise, not a "style".
  Expected outcome: no significant bubble edge → falsified cheaply.

### Rank 2 — attacker_bubble_v1 as-is

`mirrors/tier0_20260611/runs/attacker_bubble_v1/ckpt_iter_0600.pt`
(600 iters, every traversal from a harvested n_alive=4 start,
`training_dist_v1_bubble4`, 96,749 rows).

- **For:** exists (zero marginal training cost); trained *exclusively* on
  bubble states; same encoder/abstraction, loads through the identical
  `CheckpointPolicy` path (proven in B5).
- **Against — and this is heavy:**
  1. **It already failed its own specialty head-to-head.** B5
     (`evals/attacker_ext_20260611_pergame/attacker_ext_20260611/REPORT.md`):
     as bubble hero vs five champion seats its extraction is
     **−0.0783 ± 0.0132 (z=−5.9)** floors-off — significantly *below*
     champion-symmetric par. 600 BR iterations produced a bubble player
     measurably weaker than the champion at the bubble.
  2. **Wrong objective.** It is a best-response *exploiter of the
     champion* — `league_mix=1.0`, every opponent seat short-circuited to
     the frozen champion. The ensemble needs a general bubble player vs
     the *field* (killphil/itmstrike-class shove pressure, per §1.2); the
     attacker never saw a single non-champion opponent. Monoculture
     squared, against H2's own premise.
  3. 600 iters of Deep CFR is far from convergence even for the BR task
     (the REPORT's own caveat 1).
- **Residual reason to test it at all:** the transfer hypothesis
  ("BR-to-champion bubble play happens to handle field bubble pressure")
  is logically distinct from B5's verdict and costs one P2 arm to kill.
  It rides *conditionally* (§3, P2 arm B) — and only because the prompt of
  this design is exactly "cheapest falsifiable".

### Rank 3 — bubble-league-trained variant (does not exist)

The *clean* concept: resume from champion ckpt_1500, traversals drawn from
the bubble-start artifact, **self-play/league mix** (champion frozen seats +
killphil/itmstrike-class scripted styles via `configs/league/` — the H2
machinery) rather than pure BR — i.e., "what H2 is to shove-defense, this
is to the bubble stage."

- **For:** directly targets the measured defect (§1.2); all machinery
  exists (h2_probe is running this exact pattern right now); a3 says
  policy-not-representation, so a policy-level fix is the right class.
- **Against:** it is a **training job** (~500 iters ≈ 2.5–4 h CPU at
  h2_probe's measured 17.5 s/iter under contention) — not "cheapest", not
  file-only, competes with the active H2 chain for cores, and inherits
  e2's stilltoact caveat for any killphil-class league member.
- **Disposition:** NOT part of this probe. It becomes a separately
  pre-registered probe **only if** this probe falsifies all Rank-1/2
  candidates *and* P0 validates the selector seam (so the ensemble shell
  is known-good and waiting for a working organ). Entry also waits on the
  H2 probe verdict — if H2's league mix already lifts bubble play, the
  specialist may be a free by-product.

---

## 3. The probe — full pre-registration

**Battery uses existing harnesses only:** `scripts/sng_baseline.py`
(CRN per-game rows, `seat_to_policy` seam, `stage_acc`) and
`scripts/attacker_extraction_eval.py` (bubble mode, harvested n_alive=4
starts, per-game seed `2026 + 7919·g`). The only new code is an
`EnsemblePolicy` wrapper (Policy-protocol, ~30 lines, selector of §1.1) and
a thin driver that calls `sng_baseline.evaluate_profile` with
`hero_policy=EnsemblePolicy(...)` — no game-loop, scoring, or seeding code
is touched. Identity stamping per B3 discipline (sha256 of BOTH member
checkpoints + abstraction in the run header).

All runs: `nice -n 19`, `taskset` within cores the core ledger assigns
(8-core cap, live-listener stand-down rule applies; h2_probe currently owns
cores 4–11 — this probe queues behind it in COMPUTE_QUEUE.md or runs night
shift).

### P0 — selector-mechanics validation (≈ 15 min, 1 core)

Run `EnsemblePolicy(champion, champion)` (specialist := the champion itself)
for 50 yardstick games vs killphilmtt, master_seed 2026.

- **Bar (exact):** per-game records **bit-identical** to the same 50 games
  with plain `CheckpointPolicy` champion. The selector consumes no RNG
  draws and both branches load the same weights, so any diff is a seam
  bug. Also: constructed-`starting_stacks` cases via the documented test
  seam (incl. a 1-chip live seat) to characterize the §1.1 collision; log
  inferred n_alive per hand.
- **Fail ⇒ fix or kill (kill criterion K1).**

### P1 — S1 screen: do sibling checkpoints carry a bubble edge? (≈ 1.5 h wall on 8 cores)

`attacker_extraction_eval.py --mode bubble --floors off`, hero =
ckpt_1300 (arm 1) / ckpt_1400 (arm 2), champion = ckpt_1500, **2,000 games
each**, master_seed 2026 (CRN-paired across arms and against B5's first
2,000 games). Metric: extraction/game (ICM-start-relative; neutral = 0 by
symmetry, per the B5 baseline note).

- attacker_bubble_v1 needs **no P1 run** — B5 *is* its P1 (4,000 games,
  −0.0783 ± 0.0132): already FAILED this screen.
- **S1 pass bar (per candidate, fixed now):** extraction/game ≥ **+0.040**
  with z ≥ 2 (SE at n=2,000 ≈ 0.020 by B5 scaling). Below that, a bubble
  "edge" cannot survive PG-battery noise floors (e3: instrument noise
  ~0.045/game) and is not worth an ensemble.
- **Expected SEs:** ±0.020/game per arm; paired-vs-1500 tighter.

### P2 — S2 test: ensemble vs the field (CRN yardstick rows) (≈ 2–3 h wall on 8 cores)

Arms (run only what P1 licenses):

- **Arm A:** `Ensemble(champion, best P1 passer)` — only if some Rank-1
  candidate passes P1.
- **Arm B:** `Ensemble(champion, attacker_bubble_v1)` — the transfer
  hypothesis; runs regardless of P1 (it is the named organ this design was
  asked to adjudicate), UNLESS the budget kill (K4) has fired.

Each arm: `sng_baseline` rows vs **8 profiles** — the 6 bubble-negative
rows (§1.2 table) + 2 positive-bubble controls (**thefixersng** +0.0121/hand
bubble, **littlegreen** +0.0125/hand bubble; chosen for strongest controls at
lowest per-row cost, 439 s/385 s per 2,000 games in the yardstick run) —
**2,000 games each, master_seed 2026, hpl 5, mode sample** = exactly the
v1 yardstick configuration, so every game pairs by game-id against the
existing champion per-game records
(`evals/sng_baseline_20260610/w*/games_<profile>.jsonl`, seeds verified
`2026 + 7919·g`). The champion arm costs **zero new compute**. Because the
selector consumes no RNG and both members consume one draw per decision,
ensemble games are byte-identical to champion games until the first
bubble-hand action divergence — pairing cancels all pre-bubble variance.

**Falsification bars — fixed NOW, before any run:**

| id | metric | PASS requires | else |
|---|---|---|---|
| B1 | aggregate bubble-stage Δ/hand (ensemble − champion, paired, pooled over the 6 target rows; from `stage_acc` n_alive=4) | > 0 with z ≥ 2 | arm falsified |
| B2 | each of 2 control rows, paired net/game delta | not significantly negative (z > −2) | arm falsified (harm) |
| B3 | killphilmtt row net/game | ≥ champion's −0.174 − 2·SE_paired (no worsening of the PG1 worst row) | arm falsified |
| B4 | pooled 6-row paired net/game delta | > 0 (sign check; B1 is the powered test — bubble stage is only ~13 % of hands, so game-level significance is not required at probe scale) | note, not kill |

An arm must pass B1 ∧ B2 ∧ B3. Passing ⇒ the ensemble is promoted to a
**candidate** and faces the full unweakened program battery (PG1 full
24-panel + PG2 + PG3 + PG4 + ATT with the ensemble as target) before any
deployment talk — same gates forever, no novelty discount (Addendum 3.3).
The probe itself ships nothing.

### Cost (pre-committed)

Timings scale from the yardstick run's own elapsed_s (8-way parallel,
contended box) and B5's bubble-game shape:

| step | wall (8 cores, nice 19) | core-h |
|---|---|---|
| P0 | ~0.25 h | ~0.3 |
| P1 (2 × 2,000 bubble games) | ~1.5 h | ~4 |
| P2 (per arm: 8 rows × 2,000 games ≈ 4,800 s hot-row sum) | ~2–3 h | ~8 |
| analysis + report | ~0.5 h | ~0.5 |
| **total (worst case, both P2 arms)** | **≤ 6 h** | **~21** |

If wall-clock projects past 6 h after P1 (contention), drop P2 to one arm
(priority: P1 passer > attacker_bubble_v1) — pre-committed de-scope, not an
ad-hoc one.

---

## 4. Live-path feasibility (Addendum 3.3 constraints)

- **Memory:** each checkpoint is 11 MB on disk (~tens of MB loaded); two
  resident solvers are noise against 48 GB. Loading both at session start
  adds one `_load_solver` call (~seconds, once).
- **Latency:** selector = one integer compare on a field `make_decision`
  already computes (`out.n_alive`, live_loop.py:809); exactly one network
  forward per decision either way → sub-second budget untouched.
- **Determinism / PG4:** n_alive is logged on every LiveDecision record,
  so `replay_make_decision_diff.py` replays the selector exactly;
  `DecisionCache` identity already includes `alive[]`, so a selector flip
  forces a fresh cached decision — no cross-member cache leakage.
- **Floor compatibility:** `make_live_policy_filter` is policy-agnostic
  (operates on the policy vector + parsed/state, see
  `FlooredCheckpointPolicy` in attacker_extraction_eval.py); the same
  composed chain (AA/KK + check-free + short-stack [+ tail floor]) wraps
  specialist output unchanged. P2 runs floors-off (raw-policy comparison,
  the yardstick convention); the full battery re-measures floors-on.
- **Wiring (NOT part of this probe):** `make_decision` takes a single
  `solver`; production would add an optional `bubble_solver=None` param —
  flag-gated OFF by default, byte-identity proof flag-off per the Stage-2
  precedent, ARMING operator-only (standing constraint 3). No live-path
  code changes within this probe.

---

## 5. Kill criteria (explicit, honored without sentiment)

- **K1 — seam:** P0 bit-identity fails and is not fixed within 1 h of
  debugging ⇒ kill the probe, log the seam bug, report.
- **K2 — S1 dead:** no Rank-1 candidate passes P1 (≥ +0.040, z ≥ 2) AND
  Arm B fails B1 ⇒ the cheap-organ ensemble is **falsified**. Write the
  negative report (first-class, Addendum 1.2); the only registered
  follow-up is the Rank-3 bubble-league specialist probe, gated on the H2
  verdict and its own pre-registration. Do not re-run without new evidence.
- **K3 — harm:** any P2 arm fails B2 or B3 ⇒ that arm is falsified even if
  B1 passes (a specialist that buys bubble EV by selling control rows is
  exactly the failure mode the selector was supposed to prevent).
- **K4 — budget:** projected wall > 6 h ⇒ de-scope per §3; if still > 6 h,
  kill and reschedule night shift. Kill on *type-of-problem* change
  (wrong config, no visibility, live listener appears ⇒ immediate
  stand-down), never on impatience.
- **K5 — staleness:** if the H2 probe (running now) PASSES before P2
  launches, re-evaluate: an H2-retrained champion changes the baseline this
  whole design measures against; the probe re-registers against the new
  champion or dies.

**Verdict logging:** falsification criteria above are the pre-registration;
EXPERIMENT_LOG gets the entry before P0 starts; full report to
`reports/EXP_F_ENSEMBLE_PROBE.md` either way; RESEARCH_MAP pillar (f) gets
a new f3 entry with the outcome (and b2 gets P1's checkpoint-variance
answer for free).
