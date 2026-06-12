# EXP_H1 — Commitment-scaled tail floor

**Status:** build + TG1 + TG2 complete (2026-06-12); TG3/TG4 re-registered,
deferred for a dry-run-free window. **Verdict so far:** TG2-as-first-
registered FAILED its over-aggression guard; adversarial review traced part
of the failure to an instrument bug (commitment formula), fix adopted,
gates re-run — see Results. Nothing ships from this experiment without
TG3+TG4 and operator sign-off.

## 1. Background & motivation

The deployed champion (b79e82dd) serves sample-mode draws from its full
mixed strategy. Cumulative tail accounting across live sessions 1–3 is
statistically clean (z=+0.49..+0.57 — no mechanical defect), but individual
low-mass draws commit the whole stack: session-1 seq-315 sampled a 2d5c
open-shove from a 7.46% ALLIN tail. Operator approved "tail floor (policy
call)" as Stage-2 CAN-RIDE (SESSION_LOG 2026-06-11). The mixed strategy's
tail value is theoretical (balance vs adapting opponents); the realized cost
in a 6-max ICM SNG is concrete. Related prior: the short-stack floor's
paired A/B showed masking strategically-incoherent actions at deployment can
be worth +0.0100 ± 0.0026 ICM/game.

## 2. Pre-registered design (frozen 2026-06-12 before build)

Spec: `../H1_TAIL_FLOOR_SPEC.md`. Floor: τ(a) = τ_max · commit_frac(a),
prune 0 < mass < τ(a), renormalize; flag-gated OFF (`--tail-floor-tau`).
Falsification criteria (EXPERIMENT_LOG, pre-committed): F-TG1 any flag-off
behavioral diff ⇒ reject; F-TG2 (a) seq-315 caught at τ∈{0.10,0.15},
(b) zero argmax changes, (c) altered-decision rate at τ=0.10 ≤ 5% else STOP;
F-TG3 ship bar paired z > −2 AND per-firing diverged delta z ≥ +2;
F-TG4 extraction no worse than B4/B5 floors-ON baselines beyond 2σ.
Cost estimate: build+TG1+TG2 4–6 h (actual ≈ 5 h incl. review + fix);
TG3 4–8 h CPU; TG4 6–10 h CPU.

## 3. Methods

- Implementation: `apply_commitment_tail_floor` in
  `src/nlhe/integration/live_loop.py`, composed LAST after the three
  existing floors; OFF path structurally never calls it. Flag plumbing:
  `make_decision(tail_floor_tau=)`, `run_live_dryrun.py --tail-floor-tau`,
  header echo. Tests: `tests/test_tail_floor.py` (18) +
  `tests/test_session_header.py` (header key).
- **TG1:** `scripts/replay_make_decision_diff.py run` per log, two arms:
  HEAD-with-H1-diff vs pre-H1 commit `28ce614` (clean git worktree,
  per-arm imports verified by review), champion ckpt b79e82dd + retrofit
  abstraction, seed 42; `cmp` byte-identity per log. Corpus: every
  `logs/live_dryrun_*.jsonl` not modified in the last 10 min.
- **TG2:** `scripts/tail_floor_counterfactual.py` — replays each log
  through `make_decision` with decision_audit capture hooks (per-frame
  fidelity check vs the logged session), applies the floor at
  τ∈{0.05,0.10,0.15} to each fresh-decision distribution.
  Outputs: `evals/h1_tail_floor_20260612/`.
- Commit-semantics fix (post-review): exact chips-added from the
  sampling site's `discrete_to_chip` map (bet chip ints are whole-hand
  totals ⇒ added = chip − contribution[cp]; FOLD/CALL/ALLIN exact in both
  modes), threaded via `accepts_d2c` filter attribute in
  `scripts/eval_6max_self_play.py`. Pre-fix artifacts preserved at
  `tail_floor_counterfactual_BIASED_COMMIT.{json,txt}`.

## 4. Results

### TG1 (first run, biased-commit build) — PASS
5,369 frames over **11 frame-bearing logs** (5 logs have 0 raw_record
frames — pre-raw-record format), all byte-identical HEAD vs 28ce614.
Stdout side-channels also identical (review-verified). Evidence:
`evals/h1_tail_floor_20260612/tg1/`.

### TG2 (first run, biased-commit floor) — bars (a),(b) PASS; (c) FAIL
412 fresh decision frames, 11 logs; 4 pre-06-11 headerless logs carry 112
replay-fidelity mismatches (all traced to known code evolution: 7d47e86
suspect recovery, 2c0b71b seq-101, f1a412b seq-170 + RNG cascade) — their
distributions are reconstructions; clean subset = 302 frames.

| τ_max | fired | altered (hist. sample excluded) | clean-subset altered | argmax changes |
|---|---|---|---|---|
| 0.05 | 51.2% | 7/412 = 1.70% | 5/302 = 1.66% | 0 |
| 0.10 | 74.0% | 21/412 = **5.10%** | 16/302 = **5.30%** | 0 |
| 0.15 | 84.0% | 41/412 = 9.95% | 31/302 = 10.26% | 0 |

seq-315 (2d5c 7.46% ALLIN) caught at 0.10 and 0.15. Also caught at 0.10:
seq-443 9d3h ALLIN 9.3%, 5d3h ALLIN 1.7%, Js6h ALLIN 4.7% (junk-jam class)
— and Ks7c ALLIN 9.9%, 5dAd ALLIN 9.8%, AdAs BET_200 6.1% (plausibly
correct mixed aggression; EV question = TG3's job).
**Bar (c) verdict: FAIL at 5.10% (5.30% clean) vs ≤5% — STOP held; TG3 did
not launch under the first registration.**

### Adversarial review (full text in EXPERIMENT_LOG entry)
- **A1 UPHELD:** "altered-decision rate" is operationalization-dependent.
  At τ=0.10: E[pruned mass] (TV) 4.36%; historical-draw metric 5.10%;
  realized `rng.choices`-coupling divergence **9.45%**. All three now
  reported; the registered bright-line verdict (FAIL) stands under every
  reading.
- **A2 UPHELD (instrument bug):** `_commit`'s pot=sum(contribution)
  understates universal_poker's parsed pot; 28/1,579 corpus bets
  misclassified in the dangerous (commit-underestimating) direction.
  With exact chip-map commitment, reviewer's quantification: τ=0.10
  altered rate 4.37% (clean 4.64%) — passes the guard, still catches
  seq-315, 0 argmax changes. **Fix adopted** (Methods).
- **A3 PARTIAL:** TG3 harness swap to `short_stack_floor_ab.py`-style 24k
  paired games is pre-authorized by the spec, but narrows the claim to
  blueprint-self-play EV (conservative junk-detection test, not
  "vs live field"); recorded. TG4 blind spot recorded: reused attackers
  were trained vs the un-floored champion — a measurement-only re-check
  cannot find a NEW leak the floor opens.
- **A4 PARTIAL:** τ→0.05 amendment REJECTED as drafted (abandons the
  motivating incident; skips the dominant investigate-outcome = fix the
  commit bug). "5.10%≈5%, proceed anyway" also REJECTED — bright lines
  stay bright.
- **A5 REJECTED (TG1 stands)** with caveats: scope is make_decision-level
  behavioral projection; `run_live_dryrun.py` OFF-path delta
  (flag plumbing + header key) asserted by inspection, not replay-gated;
  log count corrected to 11 frame-bearing.
- **A6 REJECTED (instrument not tainted):** mismatches = known code
  evolution; clean-subset reporting is the correct handling.

### TG1 re-run (exact-commit build, includes eval_6max_self_play.py dispatch) — PASS
Same two-arm protocol over the post-fix diff (now also touching the
`accepts_d2c` dispatch in `scripts/eval_6max_self_play.py`): all 16 logs
(11 frame-bearing, 5,369 frames) byte-identical. F-TG1 satisfied on the
final build. Evidence: `evals/h1_tail_floor_20260612/tg1/sweep.log`.

### TG2 re-run (exact-commit floor) — ALL BARS PASS at τ=0.10
412 frames (302 clean), same instrument, exact chip-map commitment:

| τ_max | fired | altered (full) | altered (clean subset) | argmax changes |
|---|---|---|---|---|
| 0.05 | 50.5% | 8/412 = 1.94% | 6/302 = 1.99% | 0 |
| **0.10** | 71.1% | **18/412 = 4.37%** | **14/302 = 4.64%** | **0** |
| 0.15 | 82.0% | 34/412 = 8.25% | 24/302 = 7.95% | 0 |

Bars: (a) seq-315 caught at 0.10/0.15 ✓; (b) 0 argmax changes ✓;
(c) 4.37% ≤ 5% (clean 4.64%) ✓. Reviewer-computed realized
(`rng.choices`-coupling) divergence at corrected τ=0.10 ≈ 8.7%/decision —
recorded per A1 as the honest deployment-divergence figure.
The 18 altered draws at τ=0.10: 10 junk-aggression (2d5c/9d3h/4s3c/Td2h/
Js6h/9d5c/Qh2d/Qc9d/8dJs/5d3h), 4 marginal A-x/K-x jams (6cAc/4hKs/Ks7c/
5dAd ~5–10% mass), plus QsJh BET_150 1.3%, 5sAh BET_100 1.3%, 6hAs BET_200
4.7%, Qs6d CALL 1.3%. EV separation of those classes is exactly TG3's job.
Evidence: `evals/h1_tail_floor_20260612/tail_floor_counterfactual.{json,txt}`
(biased-commit originals preserved as `*_BIASED_COMMIT.*`).

### H1.2 re-registration (pre-committed 2026-06-12, BEFORE TG3)
- **Arm:** τ_max = 0.10, exact-commit floor (the value that passes the
  guard AND covers the motivating incident — reviewer A4 disposition).
- **TG3:** 24,000 CRN-paired games (seeds 1..24000, hpl=5 live-matched
  escalation), `short_stack_floor_ab.py`-style harness adapted per review
  A3: BOTH arms run the full deployed floor chain, tail floor the only
  delta; new-arm V0-identity check before launch; diverged-game = first
  hero sampled-action difference. Claim scope: blueprint self-play
  (conservative junk-detection), NOT live-field EV.
- **Ship bars UNCHANGED:** F-TG3 paired all-games z > −2 AND diverged-only
  per-firing delta z ≥ +2; F-TG4 extraction no worse than B4/B5 floors-ON
  baselines beyond 2σ (with the recorded stale-attacker caveat).

## 5. Discussion

What this means: the floor mechanism is built, OFF-path-safe, and its
counterfactual behavior is now measured with exact commitment costs. What
it does NOT mean: nothing here demonstrates an EV gain — TG2's altered set
visibly mixes junk jams with plausibly-correct mixed aggression, and only
the paired EV gate (TG3) can separate them. Threats to validity: (i) the
historical-draw counterfactual metric is one of three defensible
operationalizations (all reported per A1); (ii) 110/412 frames come from
reconstructed (non-faithful) logs — clean-subset numbers are headline;
(iii) TG3 as re-registered tests blueprint self-play, not the live field;
(iv) TG4's reused attackers cannot expose floor-opened leaks (A3).
Alternative explanation kept open: low-mass aggressive tails may be
load-bearing balance, in which case TG3 should show the floor ≤ 0 and H1
dies — that is a fully acceptable outcome under F-TG3.

## 6. Future work

- TG3 (re-registered; deferred for dry-run-free window): 24k paired games,
  full floor chain both arms, tail floor only delta, seeds 1..24000,
  hpl=5 live-matched. ~4–8 h CPU. Decides ship/no-ship with TG4.
- TG4 attacker re-measure (B4/B5-style, floors ON incl. tail). ~6–10 h CPU.
- Exploratory (RESEARCH_MAP backlog #1): tail-concentration map from the
  TG2 JSON (street/depth/position of pruned tails). ~2 h, feeds pillar (a).
- If TG3 passes and ships: live arming proposal goes to OPERATOR_QUEUE
  (never armed without an approval line).
- Fresh-attacker retrain vs the floored champion (closes the A3/TG4 blind
  spot) — pod-class cost; only if H1 ships and matters.
