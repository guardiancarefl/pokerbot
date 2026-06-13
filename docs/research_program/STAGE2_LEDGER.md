# STAGE-2 READINESS LEDGER

Generated 2026-06-13 00:29 UTC by `python scripts/stage2_ledger.py` (re-runnable; the post-session pipeline regenerates this file in full — do not hand-edit rows).

## Pre-registered STAGE-2 GATE

**GATE: 5 consecutive sessions with ZERO never-decided to-act hands AND ZERO required manual interventions.**

- N=5 is **PROPOSED, awaiting operator confirmation** (pre-registered 2026-06-13, before any session satisfied it).
- *Never-decided to-act hand* = triage hand-accounting row "hands LOST to skips": raw_record shows hero action buttons in >=1 frame of the hand and the pipeline produced zero decisions for it. A fired watchdog fallback does NOT clear the hand — fallbacks are guaranteed actions, not decisions.
- *Required manual intervention* = operator had to overrule a pipeline output to avoid a clear EV disaster (pin lists in EXPERIMENT_LOG; mirrored in `MANUAL_INTERVENTIONS` in the script). Sessions with no recorded pin list count as UNKNOWN, not zero — they cannot contribute to the gate streak.
- Fallback fires and safe-folds are tracked as trend columns but do not gate (they are the safety net working as designed).

## Per-session ledger

| session | op-session | frames | hands | decisions | never-decided hands | to-act frames skipped | fallback fires | aborts tripped | safe-folds | manual interv. | red flags | gate-clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260608_004502 |  | 374 | 18 | 33 | 0 | 0 | n/a | n/a | 10 (4x invariant_fail; 2x replay terminal-before-hero; 2x replay-derivation; 2x replay_error) | unknown | 16 | no (interv. unknown) |
| 20260608_045752 |  | 93 | 4 | 4 | 0 | 0 | n/a | n/a | 0 (0) | unknown | 1 | no (interv. unknown) |
| 20260608_050354 |  | 79 | 4 | 5 | 0 | 0 | n/a | n/a | 1 (1x invariant_fail) | unknown | 3 | no (interv. unknown) |
| 20260608_051036 |  | 358 | 20 | 59 | 0 | 0 | n/a | n/a | 3 (2x replay terminal-before-hero; 1x replay-derivation) | unknown | 4 | no (interv. unknown) |
| 20260608_152756 |  | 200 | 11 | 31 | 1 | 5 | n/a | n/a | 2 (2x invariant_fail) | unknown | 6 | no |
| 20260609_154557 |  | 571 | 48 | 59 | 14 | 53 | n/a | n/a | 2 (1x invariant_fail; 1x replay terminal-before-hero) | unknown | 4 | no |
| 20260609_192543 |  | 402 | 44 | 65 | 2 | 51 | 0 | 0 | 1 (1x replay-derivation) | unknown | 2 | no |
| 20260611_163815 |  | 486 | 37 | 61 | 4 | 34 | 0 | 0 | 4 (1x invariant_fail; 2x replay-derivation; 1x replay_error) | unknown | 4 | no |
| 20260611_181249 |  | 54 | 6 | 7 | 0 | 0 | 0 | 0 | 0 (0) | unknown | 0 | no (interv. unknown) |
| 20260611_201230 | op-session 3 (G1) | 733 | 62 | 92 | 3 | 24 | 4 | 0 | 5 (2x invariant_fail; 1x replay-derivation; 2x replay_error) | unknown | 10 | no |
| 20260611_204751 | op-session 3 (G2-3) | 950 | 76 | 141 | 1 | 67 | 2 | 0 | 1 (1x replay_error) | unknown | 1 | no |
| 20260611_222532 |  | 1326 | 80 | 155 | 4 | 42 | 4 | 0 | 6 (4x invariant_fail; 2x replay-derivation) | unknown | 11 | no |
| 20260612_035454 |  | 218 | 15 | 27 | 0 | 18 | 3 | 0 | 1 (1x invariant_fail) | unknown | 1 | no (interv. unknown) |
| 20260612_042058 |  | 316 | 35 | 43 | 5 | 17 | 5 | 4 | 11 (7x invariant_fail; 3x replay terminal-before-hero; 1x replay_error) | unknown | 18 | no |
| 20260612_125441 |  | 84 | 5 | 3 | 3 | 3 | 3 | 2 | 7 (4x replay-derivation; 3x replay_error) | unknown | 7 | no |
| 20260612_130846 |  | 233 | 11 | 22 | 0 | 2 | 1 | 0 | 0 (0) | unknown | 0 | no (interv. unknown) |
| 20260612_133056 |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (0) | unknown | 0 | empty |
| 20260612_161347 |  | 391 | 30 | 51 | 3 | 31 | 1 | 0 | 1 (1x invariant_fail) | unknown | 1 | no |
| 20260612_163114 |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (0) | unknown | 0 | empty |
| 20260612_183701 |  | 1519 | 140 | 194 | 6 | 83 | 7 | 0 | 6 (2x invariant_fail; 2x replay terminal-before-hero; 2x replay-derivation) | unknown | 8 | no |
| 20260612_220101 |  | 505 | 36 | 53 | 0 | 10 | 0 | 0 | 1 (1x replay_error) | unknown | 1 | no (interv. unknown) |
| 20260612_230149 | op-session 5 | 1649 | 124 | 215 | 8 | 86 | 8 | 0 | 14 (4x invariant_fail; 4x replay terminal-before-hero; 6x replay-derivation) | 1 | 16 | no |
| verify1_20260609_202137 |  | 90 | 6 | 14 | 0 | 10 | 0 | 0 | 0 (0) | unknown | 1 | no (interv. unknown) |

## Trend

- Sessions on record: 21 (+2 empty logs).
- Never-decided hands per session (chronological): [0, 0, 0, 0, 1, 14, 2, 4, 0, 3, 1, 4, 0, 5, 3, 0, 3, 6, 0, 8, 0] — latest = 0.
- Fallback fires (sessions with stdout): [0, 0, 0, 4, 2, 4, 3, 5, 3, 1, 1, 7, 0, 8, 0].
- **Current gate streak: 0 / 5.** No session yet combines zero never-decided hands with a recorded zero-intervention pin list.

## Games played (all-time, detector + operator reconciliation)

Game boundary = stack reset to ~1500 for >=5 seats at L1 blinds (15/25) after the previous game escalated past L1. Outcome: L = hero bust observed (trailing seat1=0 reads); W = 2x-start terminal (last reliable hero read >= 2700); ? = end hidden in transition OCR garbage. Games begun between listener runs are not counted, and pre-aa2f505 logs (no raw_record: the four 2026-06-08 logs) cannot be segmented at all (coverage caveat).

| session | games | W | L | ? | per-game detail (first-last seq, max BB, hero final, outcome) |
|---|---|---|---|---|---|
| 20260608_152756 | 1 | 0 | 1 | 0 | G1 1-200 bb50 hero=0 L |
| 20260609_154557 | 1 | 1 | 0 | 0 | G1 8-450 bb200 hero=3252 W |
| 20260609_192543 | 1 | 1 | 0 | 0 | G1 1-402 bb150 hero=5275 W |
| 20260611_163815 | 1 | 1 | 0 | 0 | G1 1-486 bb200 hero=5354 W |
| 20260611_181249 | 1 | 0 | 1 | 0 | G1 1-54 bb0 hero=0 L |
| 20260611_201230 | 1 | 1 | 0 | 0 | G1 1-733 bb300 hero=3621 W |
| 20260611_204751 | 2 | 2 | 0 | 0 | G1 743-1140 bb100 hero=5965 W; G2 1140-1715 bb300 hero=5887 W |
| 20260611_222532 | 4 | 0 | 4 | 0 | G1 1717-2034 bb100 hero=0 L; G2 2034-2414 bb150 hero=0 L; G3 2414-2672 bb100 hero=0 L; G4 2672-3042 bb100 hero=0 L |
| 20260612_035454 | 1 | 0 | 1 | 0 | G1 3046-192 bb50 hero=0 L |
| 20260612_042058 | 1 | 1 | 0 | 0 | G1 263-578 bb100 hero=5915 W |
| 20260612_125441 | 1 | 0 | 0 | 1 | G1 1-58 bb0 hero=1170 ? |
| 20260612_130846 | 1 | 0 | 1 | 0 | G1 66-298 bb100 hero=0 L |
| 20260612_161347 | 1 | 0 | 0 | 1 | G1 8-382 bb150 hero=2630 ? |
| 20260612_183701 | 3 | 1 | 1 | 1 | G1 1-395 bb200 hero=4785 W; G2 395-941 bb200 hero=0 L; G3 941-1519 bb300 hero=4477 ? |
| 20260612_220101 | 1 | 0 | 1 | 0 | G1 8-512 bb200 hero=0 L |
| 20260612_230149 | 4 | 1 | 3 | 0 | G1 3-173 bb50 hero=0 L; G2 173-383 bb50 hero=0 L; G3 383-976 bb150 hero=3997 L*; G4 976-1651 bb150 hero=3617 W |
| verify1_20260609_202137 | 1 | 0 | 0 | 1 | G1 8-97 bb0 hero=0 ? |

**All-time totals: 26 tournaments — 9 W / 13 L / 4 unknown.**  (`*` = outcome set by operator statement, see OPERATOR_GAME_OVERRIDES.)

### Calibration / reconciliation notes

- 2026-06-12 op-session 5 (230149): operator ground truth = "played 3 games, won the last". Detector finds **4** games; the extra one is seqs 1-172 (~19:02-19:15 local): 7 hero decision spots (8h6d, 2cQc, 9dAc, Qh8d, 6h9s, Qc8s, 3cQh), hero short by L2 and seat1 reads 0 at seqs 160-162, fresh 1500-stack table at seq 173. Evidence says the operator count omitted this first quick bust-out — **awaiting operator confirmation**. Outcomes agree: every game before the last was a loss; the last was the win (hero ~3617 at L4).
- op-session 3 (204751): SESSION_LOG records "won G1+G3" (G2 a loss), but the log shows hero ending BOTH 204751 games in the final 3 (5965 and 5887 chips at 3-handed terminal) — either a G-numbering mismatch (an unlogged game in the 20:32-20:47 listener gap) or a SESSION_LOG error. Flagged, not silently corrected.

