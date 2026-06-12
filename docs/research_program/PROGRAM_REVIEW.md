# PROGRAM REVIEW — Addendum 1.4 synthesis

Append-only. Review #1 below; future reviews append after every 3 completed
experiments or weekly, whichever first.

---

# REVIEW #1 — 2026-06-12, after experiments 1–3 (H1, H2, c1)

Inputs: `EXPERIMENT_LOG.md` (full chronology), `reports/EXP_H1_tail_floor.md`,
`reports/EXP_H2_killphil_league.md`, `evals/c1_icm_gap_20260612/TIER1_REPORT.txt`,
`RESEARCH_MAP.md`, `OPERATOR_QUEUE.md`, slate outputs (FIELD_DOSSIER,
ENSEMBLE_PROBE_DESIGN + f1 REPORT, ICM_GAP_STUDY_DESIGN, REBEL_CLASS_DESIGN_MEMO,
BBNORM_TRANSPLANT_SPEC, H4_RNR_FIELD_SPEC), `evals/a1_blur_map_20260612/`.

## 0. Operator executive summary

1. Three experiments closed in ~18 h: **H1 PASS** (tail floor: +0.123/game
   self-play, z=16.8; armed; 3 live sessions in-band, 0 tail-caused argmax
   changes), **H2 FAIL at probe** (league fine-tune collapsed, all bars broken;
   mechanism confounded by a slim checkpoint), **c1 MATERIAL** (MH ICM is
   wrong by +0.041 P(ITM) on short stacks in high-dispersion bubble states).
2. No candidate beats the champion yet. The cycle hardened the champion,
   killed one retrain path, and re-priced the value function under everything.
3. The killphil hole is real but **−0.080, not −0.174** — half the believed
   extraction was our own adapter bug. It is over-calling, depth-flat, and
   three independent instruments now point at the same 5–15bb facing-shove cell.
4. Next cycle's top item is the **slate-2a shove-defense floor probe** — zero
   training, attacks that cell directly; then the ICM correction fit/re-price.
5. Decisions needed from you: P2 arming (built+gated, OFF), OQ-1 Windows brief
   routing (binding constraint on H4 fidelity). No pod case stands today.

## 1. Scoreboard

| # | exp | verdict | headline number | what it changed |
|---|---|---|---|---|
| 1 | H1 tail floor | **PASS → DEPLOYED-ARMED** | TG3 paired self-play ICM Δ **+0.1227 ± 0.0073/game (z=+16.76)**, 24k CRN games; TG4: trained-BR extraction *fell* (standard paired −0.1005 ± 0.0210, z=−4.78) | `--tail-floor-tau 0.10` is standard listener config (OQ-2). First program artifact to reach live. 3 armed sessions in-band: 130846 10/12 decisions fired, 0 argmax changes; 161347 19 firings, 0 tail-caused argmax; 183701 104 firings, 0 tail-caused argmax (firing band ~70–83% vs ~70–75% predicted). |
| 2 | H2 killphil league | **FAIL-closed at probe** | M1 0.1375 ± 0.0008 vs bar ≤ 0.0650 (champion 0.0867); killphil row −0.2360 vs bar ≥ −0.0300; self-anchor z=−9.9; battery call mass 35.9% → **52.1%** (worse) | b1 closed; POD_CASE suspended; BBNORM_TRANSPLANT_SPEC void (precondition failed). Survives: fold-vs-shove battery + oracle, post-fix baselines, b2 stability calibration, league infra. **Honest mechanism question stays open:** the probe cannot distinguish "league exposure doesn't teach shove-defense" from "fine-tuning from a slim (bufferless) checkpoint with 30% scripted traversals collapses the policy." Evidence leans fragility: collapse is far broader than the treatment (anchor −0.216), M1 *healed* 1800→2000 as the reservoir matured (0.1565→0.1375), and v2 showed the same broad-regression signature under a different treatment. H2's titled hypothesis is therefore NOT cleanly falsified — only this implementation of it. |
| 3 | c1 ICM gap (Tier-1) | **MATERIAL — re-price** | B* = **+0.0412** P(ITM), Bonferroni 98.33% CI [+0.0286, +0.0539], high-dispersion bubble (t3); micro-stack (<5bb) error +0.065; equal-stack t1 −0.0034 n.s. | "Re-price" branch of the pre-registered matrix: every consumer of `icm_equity` (training regrets, subgame leaves, H2-battery oracle labels, panel pricing) carries a state-dependent bias. Correction fit + consumer audit queued. Equal-stack metrics (TG3, calibration) are not invalidated wholesale — the bias concentrates where stack dispersion is high, exactly the program's bubble interest. Caveat: t1 equivalence not formally established either (TOST 90% CI touches −0.01). |

Program objective check: the goal is a candidate that beats the champion
through the full battery. After 3 experiments the candidate count is zero;
the deployed champion is instead measurably harder to exploit (TG4) and the
measurement stack under any future candidate is more honest (e2, c1, b2).

## 2. What the program now believes (vs 24 h ago)

- **Champion robustness.** Was: low-mass aggressive tails presumed
  load-bearing balance (theoretical). Now: pruning the commitment-scaled
  tail at τ=0.10 is worth +0.12/game in self-play and *reduces* trained-BR
  extraction — the "tails are balance" alternative is refuted at this τ in
  self-play. Standing caveats: blueprint-vs-blueprint evidence, and the
  reused attackers were trained vs the un-floored champion, so floor-opened
  leaks are invisible to TG4 (A3).
- **The killphil hole's true size is −0.080 ± 0.022, not −0.174.** ~54% of
  the believed worst-row extraction was the `stilltoact`-hardcoded-0 adapter
  artifact (fix a8933ea, e2 re-baseline, 9 rows). The hole is real,
  checkpoint-stable (b2 spread 0.022 < 1σ), and dominantly OVER-CALLING
  (call mass 35.9% vs oracle 7.7%, M1 0.0867 ± 0.0007). Consequence not yet
  acted on: PROGRAM.md's PG1 baselines (−0.174 / bar −0.124) are PRE-FIX
  instrument readings and need re-statement before the next candidate gate.
- **The defect is policy, not representation, and not depth.** a3: the hole
  is depth-flat (every bucket |z| ≤ 0.61), while v2's bbnorm depth cure paid
  +0.130/game (z=+11.8) at ≤6bb — encoder kept as a transplant organ; v2's
  gate failure was an independent broad over-folding regression.
- **The blur map independently re-finds the same cell.** Facing-allin
  entropy excess peaks at 6–20bb (+0.066/+0.073 over ordinary nodes at
  80–113bb stake); near-coin-flip mixing where the oracle says fold ~92% —
  indecision, not equilibrium texture. Same-tree BR attackers extract ≤ 0
  everywhere; only off-tree shove pressure demonstrates the leak. Three
  instruments (battery, blur, a3) now triangulate one repair target.
- **MH ICM bias has structure.** Underprices micro-stack survival (<5bb
  +0.065), taxes mid stacks (rank2 −0.023, rank3 −0.022), chip leader about
  right; heterogeneous (σ_bias(t3)=0.068) so any correction must be
  state-dependent, depth_bb first feature. Position effect (~+2pp late, SB
  worst) confirmed measurable and MH-invisible.
- **Zero-training-cost bubble specialists do not exist** (f1 P1, S1
  falsified): ckpt_1300 −0.0583 (z=−3.17), ckpt_1400 −0.0193,
  attacker_bubble_v1 −0.0783 (z=−5.93) vs bar ≥ +0.040 @ z≥2. The selector
  seam itself is P0-validated bit-identical and banked. P2 Arm B (transfer
  hypothesis) is SUSPENDED on live-window stand-down, not falsified — K2
  remains open.
- **Field constants are now decision-grade** (dossier, 18 sessions/483
  hands): VPIP-lb 20.6% ± 1.9pp; positional flatness EP→BTN 22.7–24.5% (no
  widening); open sizing mass at {limp, 2.0x (55%), 3.0x, jam}, 2.2–2.5x ≈
  5%; open-jam share 23.0%; 5–15bb jam rate 8.5% [6.1, 11.8] with the
  jam-regime break at 10–15bb. Structural limits are equally settled: folds
  ~100% unobserved (0/241), showdown holdings and timing structurally
  absent — scraper work, not sample size, is the binding constraint.

## 3. Methodology audit

### 3.1 What the pre-registration discipline caught

- **τ-shopping avoided.** TG2 #1 failed its own bright line (5.10% > 5%);
  the STOP held, the reviewer rejected both the τ→0.05 amendment and
  "5.10%≈5%, proceed," the instrument bug (commitment formula, 28/1,579
  bets misclassified dangerous-direction) was fixed, and TG3 ran once at a
  single pre-registered τ with seeds 1..24000 fixed. The PASS is clean
  because the FAIL was honored first.
- **Relaunch clause rejected on evidence.** H2's interruption excuse was
  killed by the ckpt_1800 diagnostic (pre-interruption M1 0.1565 worse than
  final 0.1375) — invoking the clause would have been results-motivated,
  which the clause's own text forbids.
- **C0 halt fired and worked.** c1's negative control failed under the
  registered outcome definition (seat-avg −0.0101, t≈−10.8; 115/2000
  simultaneous-bust games); the instrument-failure halt fired, the
  definitional fix was logged in DEVIATION_LOG *before* Tier 1, and the
  re-run was clean (itm_sum ≡ 3, 2000/2000).
- **Instrument-fix re-baselining held the line.** After a8933ea, the
  pre/post-fix tag rule was enforced twice under temptation: e2 re-baselined
  all 9 affected rows before H2 froze its bars, and f1 caught its own
  registered pairing against pre-fix records and re-paired against post-fix
  rows (bit-identity verified) before running.
- Near-miss handled correctly: the TG4 verdict script's inverted kill-bar
  sign was caught on direction review against the registered wording before
  any verdict was logged; numbers identical both runs, disclosure recorded.

### 3.2 Where the process leaked

- **Slim-checkpoint fragility was observed but not flagged as a threat
  pre-H2.** The wiring benchmark (15:00–16:20 entries) recorded "deployed
  ckpt is SLIM → probe rebuilds reservoirs" as a logistics note in spec §4,
  not as a confound in §5's falsification design. Cost: ~2.6 h of probe
  compute bought a mechanism-confounded verdict, and the program's only
  retrain hypothesis closed conditionally instead of cleanly. Propagated
  fix: H4 spec makes buffer continuity BINDING; NEXT_RESUME carries the
  standing rule.
- **seq=276 (AcKc) was misclassified in operator-facing docs before
  forensics.** The 16:50 log entry and the P2 queue items labeled it
  "exactly the approved-can-ride P2 bet-closure-recovery class"; the P2
  build's forensics showed a dead-SB hand with a phantom reconstructed BB —
  recovery would have produced a *wrong-state decision* (seq-1363 class).
  The operator read a wrong diagnosis for ~3 h. Mitigations landed: dead-SB
  guard (positive blind-structure evidence required) + root-cause follow-up
  filed (carry blind structure into replay). Lesson: incident classification
  is a forensic claim and should carry an evidence tag or "provisional."
- **Background-watcher process-view failures.** Process-level watchers do
  not work from background tasks (Addendum 4.6 watcher re-implemented on
  file signals); SIGSTOP failed to stick on the e2 rows (killed via tmux);
  operator listeners started outside tmux produced header-only logs (163114)
  that the window-end logic had to special-case. Cost: stand-down/relaunch
  churn and one held ingest. The file-signal-only rule is now standing, but
  the listener-detection contract (tmux presence vs pgrep vs log mtime) is
  still three heuristics rather than one interface.

## 4. EVoI ranking — next cycle

Ranked; all require pre-registration before running (Addendum 1).

1. **slate-2a shove-defense floor probe** — zero-training deployment-time
   equity gate on preflop facing-allin 5–15bb; attacks the program's largest
   demonstrated leak (M1 0.0867, call 35.9% vs 7.7%) where three instruments
   agree; bars pre-sketched (M1 ≤ 0.0434 + call mass ∈ [5,15]%; paired
   self-play z > −2); H1 just proved this artifact class can pass gates and
   ship. <1 day, frozen instruments only.
2. **ICM correction fit + consumer re-price audit** (c1 MATERIAL actions) —
   fit p_corr(q, depth_bb, …) on the existing 750 Tier-1 states (20%
   holdout), re-label the 8,112 battery oracles, audit icm_adjust_returns /
   subgame leaves / H1 panel pricing. Cheap (file + minutes), and it gates
   honest reading of every future bubble verdict — including slate-2a's M-A
   grade, which reads through MH fold-vs-call EVs (battery is equal-stack-
   adjacent, where t1 ≈ clean, so expect small label shifts — but verify,
   don't assume).
3. **f1 P2 Arm B resume** — ~40–50 min closes K2 on a pre-registered probe
   (baseline rows ~10 min + 8 CRN rows + analyze); low pass-prior, but
   near-zero cost and it retires the last open ensemble claim.
4. **H4 freeze-on-unlock** — counter 383/500, ~1–2 sessions out; spec
   DRAFT@383 exists; refresh §2 constants + §4 bars at ≥500 and freeze. The
   only remaining retrain hypothesis with live-data grounding; fidelity is
   capped by OQ-1 (fold stats uncomputable) until the Windows fix lands.
5. **H2b buffer-continuity probe** — resolves H2's open mechanism
   (full-buffer continuation or league-from-iter-0); determines whether ANY
   fine-tune-based retrain (including H4's continuation shape) is viable on
   this stack. ~3 h+ compute; sequence AFTER slate-2a's verdict — if a
   deployment floor closes the hole, H2b's EVoI drops to methodology-only.
6. **Dead-SB replay fix** (bridge-side) — root fix for the AcKc class;
   prevents wrong-state recoveries and removes a forensics trap; small
   build, full gate treatment, operator-gated arming.
7. **P2 arming + OQ-1 routing** (operator items, zero compute) — P2 is
   built+gated OFF (3bf5d3f); OQ-1 is the single highest-leverage data
   action: it converts fold-vs-shove from uncomputable to measurable for
   both H4's target and any field-facing verdict.
8. **ReBeL-P1** — idle-time only, zero new samples, hard gate R² ≥ 0.5
   (kill < 0.4 closes a2 permanently); the memo's own verdict says the
   binding leak is opponent-range-dependent, so this stays bottom unless
   a1/H4 evidence shows the representation ceiling binds.

What a well-resourced team would add (not currently affordable): fresh
attacker retrain vs the floored champion (closes the TG4 blind spot — the
only structural hole in the H1 ship evidence), a fine-abstraction attacker
(e1: our exploitability claim is abstraction-blind), event-driven scraper
capture (kills [B1]/[B2] at the source), and the Tier-2 27-cell ICM map.

## 5. The standing pod question

**No pod case stands today.** The H2-pass case (full league retrain carrying
the bbnorm organ, ~3 h pod vs ~28 h Contabo) died with the probe;
BBNORM_TRANSPLANT_SPEC is void on its own precondition. Everything ranked
above runs inside the 8-core nice-19 budget (slate-2a < 1 day; correction
fit minutes; f1 resume ~1 h). Next pod triggers, in likelihood order:
(a) H2b registration *if* the chosen variant is league-from-iter-0 with full
buffers (full-retrain class: ~28 h here vs ~3 h pod); (b) H4 S1 probe pass →
full RNR retrain (pod-mandatory per PROGRAM §H4); (c) ReBeL P2 regen (~$5–10,
~8 h) only if P1 clears R² ≥ 0.5. Recommendation: keep POD_CASE suspended;
re-state it at H2b registration or H4 unlock, whichever first.
