# EXP_H2 — KILLPHIL-LEAGUE PROBE — registration (2026-06-12)

Status: **FROZEN 2026-06-12 ~14:35** (§3 filled from
`evals/e2_rebaseline_20260612/` + `evals/h2_battery/m1_champion.json`).
Per Addendum 1: falsification thresholds are pre-committed and may not
be weakened; adversarial review of the rationale is §7. Probe launched
on freeze (run dir `runs/h2_probe_league_v1`).

## 1. Hypothesis and rationale (evidence-cited)

**H2: the champion's shove-defense gap is a self-play monoculture
artifact, and a small league-mix exposure to killphil-class shove bots
during continued training moves 5–15 BB fold-vs-shove decisions toward
the optimal response without degrading self-play or panel EV.**

Evidence base:
- killphilmtt is the champion's worst panel row — v1 yardstick
  **−0.1740 ± 0.0220** (2000 games, seed schedule 2026+7919g); the sole
  C3 gate metric (`evals/sng_baseline_20260610/REPORT.txt`).
- The champion was trained pure self-play (k200_real_ante recipe);
  nothing in training ever shoved wide at it. The depth-confusion work
  (STATUS 2026-06-08) independently shows weak short-stack play.
- ADAPTER CAVEAT (now closed — and it BITES): pre-2026-06-12 the
  killphil row was measured against a corrupted adapter (stilltoact
  dead + preflop raises/limps miscounted — RESEARCH_MAP e2). The fix
  (commit a8933ea) changes killphil's as-played behavior materially:
  the re-baseline killphil row moved from −0.1740 ± 0.0220 toward
  ≈ −0.07 (final number in §3) — **a sizable share of the believed
  extraction was instrument artifact** (the broken counter kept
  `raises = 0` true facing opens, so adapter-killphil 3-bet-shoved its
  entire first-in range over the champion's opens). The hypothesis
  survives at reduced magnitude: the row is still the champion's worst
  and still negative, and the champion's M1 diagnostic (call mass 35.9%
  vs oracle 7.7% on the frozen battery) independently shows a real
  shove-defense gap — dominantly OVER-calling. **All H2 numbers use the
  POST-FIX baselines** from `evals/e2_rebaseline_20260612/`.
- INDEPENDENT SUPPORT (a3 decomposition, 2026-06-12): the killphil
  deficit is **depth-flat** across terminal-depth buckets (every
  |z| ≤ 0.61 in the v2-vs-champion artifacts) — consistent with a
  shove-defense POLICY defect rather than depth representation, which
  is exactly what league exposure (not encoder work) should fix.
  `evals/a3_v2_decomposition_20260612/`.

## 2. Instruments

- **Adapter:** post-a8933ea `ShankyProfilePolicy` (PPL preflop
  semantics, stilltoact derived). 172 tests green.
- **Trainer:** `scripts/continue_k200_real_ante.py`-style continuation
  from the deployed champion `ckpt_iter_1500.pt` (sha `b79e82dd…`),
  with the solver's existing override band
  (`solver6._sample_override_opponent`): `archetype_mix=0.0`,
  **`league_mix=0.30`**, league pool = **{killphilmtt (shanky),
  champion ckpt_iter_1000, champion ckpt_iter_0500}** sampled uniformly
  (one pool draw per traversal, all non-traverser seats — the existing
  semantics; scripted decisions never enter the strategy buffer).
- **Fold-vs-shove harness (BUILD, ~2 h):**
  `scripts/fold_vs_shove_battery.py` — a FIXED battery of preflop
  facing-all-in decision states: hero eff-stack 5–15 BB, exactly one
  all-in raiser, hero in {SB, BB, BTN}, blind levels L3–L7, all 169
  canonical hand classes × a seeded sample of (depth, position, level)
  cells, ≥ 2,000 spots total. For each spot the ORACLE response is
  computed directly: killphil's shove range at that (depth, position)
  is enumerated by querying the fixed profile runtime over all 169
  classes (`evaluate_profile` on synthetic contexts); oracle =
  argmax_{call,fold} ICM-EV given that range (equity via
  `equity_vs_range`, ICM via `icm_equity`). Battery + oracle labels are
  WRITTEN TO DISK ONCE and frozen before the probe (the probe is graded
  against a file, not regenerated labels).
- **Primary metric M1:** mean ICM-EV loss per battery spot of the
  policy's sampled-distribution response vs the oracle response
  (expected loss under the policy's call/fold mass, 0 when matching
  oracle). Lower = better shove defense.
- **Hold metrics M2:** (a) killphilmtt row, 2000 games CRN seed 2026;
  (b) panel holds: ticketmaster, sng, tighttom rows, 2000 games CRN;
  (c) self-anchor: 2000-game self-play row vs the unmodified champion
  (hero=probe ckpt, opponents=champion).

## 3. Baselines (FROZEN before probe launch)

Filled from `evals/e2_rebaseline_20260612/` (post-fix adapter, 2000
games, master seed 2026) + champion battery M1 run:

| quantity | value |
|---|---|
| killphilmtt row (post-fix) | **−0.0800 ± 0.0223** (was −0.1740 ± 0.0220 pre-fix) |
| ticketmaster row (post-fix, M2b hold) | +0.5400 ± 0.0188 |
| sng row (post-fix, M2b hold) | +0.8360 ± 0.0123 |
| tighttom row (post-fix, M2b hold) | +0.7780 ± 0.0140 |
| champion M1 EV-loss on the frozen battery | **0.0867 ± 0.0007** (call mass 35.9% vs oracle 7.7%) |
| tighttom-vs-trickytom divergence check | PASS — 59/2000 games outcome-differ (pre-fix: 0, byte-identical) |

Derived F-M2a target: killphil row ≥ **−0.0300** with improvement
≥ 2σ_diff (σ_diff ≈ 0.0315 for two 2000-game rows). Derived F-M1
target: probe M1 ≤ **0.0650**. (e2-record rows gushansenmtt /
millenniummttv.49 complete separately; they are not §3 quantities.)

## 4. Probe (pre-registered; ONLY after §3 freeze)

500 iterations continued training on Contabo via
`scripts/train_6max.py --config configs/h2_probe_league.yaml --resume
<champion ckpt_iter_1500>` (continue_k200_real_ante strips
parallel_groups — not used). **Benchmark DONE (2026-06-12): 15.0 s/iter
at G=8 on a contended box → ~2.1 h projected**; league pool loads (3
eligible, mix 0.300), resume at iter 1500 confirmed, losses at
champion-level (strat 0.76). Checkpoint every 100; tmux + watcher per
Addendum 4.2.

NOTED DEVIATION (benchmark finding): the deployed champion checkpoint
is SLIM (10.9 MB — nets + optimizers, no reservoir buffers), so the
probe rebuilds reservoirs from fresh league-mix traversals during the
first iterations rather than continuing the champion's buffer state.
Outcome-based falsification (M1/M2) is unaffected; recorded so the
probe is not misread as literal bit-continuation.

## 5. Falsification (pre-committed; any failure ⇒ H2 dies at probe)

- **F-M1 (the hypothesis):** probe ckpt_0500's M1 EV-loss must drop by
  **≥ 25% relative** vs the champion's frozen-battery baseline. A drop
  < 25% (or any increase) ⇒ FAIL — the league mix did not teach
  shove-defense at probe scale.
- **F-M2a (it transfers):** killphilmtt row must improve by **≥ +0.05
  net/game** over the §3 baseline with the improvement ≥ 2σ_diff
  (σ_diff = sqrt(se₁² + se₂²), CRN-paired games).
- **F-M2b (no collapse):** each hold row (ticketmaster, sng, tighttom)
  degrades by no more than 2σ_diff; the self-anchor row stays within
  |net| < 2σ of 0.
- **F-iter (sanity):** strat/adv losses stay finite and the
  premium-fold alert tooling (C3 battery) fires nothing.
- No seed re-rolls; no threshold shopping; one probe run. If the probe
  is killed mid-run for an infrastructure reason (not results), it may
  be relaunched once from iter 0 with the same seeds, recorded.

## 6. Outcomes

- **PASS (all of §5):** write `POD_REQUEST.md` (full-scale retrain
  proposal with measured probe evidence); **track STOPS for operator**
  (Addendum 2 boundary — pod spend).
- **FAIL any:** full report per Addendum 1, RESEARCH_MAP update, H2
  closed at probe stage; the battery + oracle harness remains as a
  permanent panel instrument.

## 7. Adversarial review (pre-launch, self-conducted per Addendum 1)

- **A1 "Oracle is killphil-specific":** the M1 oracle assumes the
  all-in raiser plays killphil's range. Against other shovers the
  optimal response differs. DISPOSITION: accepted limitation, scoped —
  M1 is named "killphil-optimal", not "optimal"; M2b holds guard
  against overfitting the defense to one range.
- **A2 "League mix could just clone killphil":** the traverser only
  learns RESPONSES (scripted seats never write to the strategy buffer);
  cloning is structurally impossible through this path. REJECTED.
- **A3 "Probe scale too small to move anything":** possible — that is
  what F-M1's 25% bar tests; a null result at 500 iters with
  league_mix=0.30 is informative (the C3 retrain moved premium-fold
  behavior within 500 iters at comparable scale). ACCEPTED RISK,
  pre-committed.
- **A4 "Adapter fix changed the target mid-program":** that is why §3
  freezes POST-fix baselines before launch and why the re-baseline runs
  first. RESOLVED by ordering.
- **A5 "CRN row comparisons are not truly paired after training
  changes hero":** correct — hero plays different actions, games
  diverge; CRN still removes draw-order variance. σ_diff bars stated
  for the unpaired-after-divergence case (conservative). NOTED.

## 8. Cost ledger (est.)

harness build 2 h · battery freeze + champion M1 ~1 h · probe 6–10 h
CPU (tmux, non-blocking) · post-probe measurement ~2 h · report 1 h.
