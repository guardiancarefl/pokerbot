# Q0 PROBE BURST — data-application methods (pre-registration)

**Question (RESEARCH_MAP Q0):** how do we correctly convert 517 hands of field
data into a winning policy? The H4 probe proved global RNR-against-the-synthetic-
pool FAILS (F-H2 collapse; `reports/EXP_H4_probe_FH2_collapse.md`). This burst
probes the competing methods, cheapest-first, under the IDENTICAL gates that
caught H4. **No quiet softening — the bars below are the H4 bars verbatim.**

## CANONICAL GATES (identical for every probe — operator-verify no softening)

These are the exact bars that caught H4. Any probe that needs a softer bar is
REJECTED, not accommodated.

- **G1 — self-anchor (F-H2), THE load-bearing bar.** 400-game CRN-paired net-EV
  of candidate vs the FROZEN champion `b79e82dd`, per checkpoint.
  PASS = net within ±0.10 of 0 AND |z|<2, no monotone decline.
  FAIL (collapse-watch, verbatim from H4) = **net < −0.10 AND z ≤ −2**, or a
  monotone decline x3, or non-finite. *(H4 failed here: −0.18/z−3.7.)*
- **G2 — ΔFIELD (F-B2).** Candidate−champion field-mixture EV, CRN-paired.
  A *learning* signal needs a real positive TREND toward **+0.05/game** across
  checkpoints — NOT a single point (noise floor ≈ ±0.05/SE at 400 games).
- **G3 — aggression (§4.4), the artifact guard.** Δaggr vs champion. >0.08 = red
  flag. **ΔFIELD up WITH Δaggr inflated (>+0.04) = the limp-gap artifact = NOT a
  pass** (this is exactly how H4's fake +0.065 was unmasked). A real win is
  ΔFIELD up with FLAT aggression.
- **G4 — full battery (survivor only).** The registered §4.2/§4.3 battery on the
  final checkpoint: F-B2 field EV + **≥2 out-of-pool holds** + a clean §4.4.
  Transfer is carried by these, never by the in-pool proxy alone (R4 circularity).

A probe PASSES its cheap stage only if it clears **G1 AND (G2 with G3 flat)**.
Clearing G2 by violating G3 is an artifact, scored FAIL.

## EXECUTION ORDER (operator-approved): C → D → A → B

1. **C** (`C_data_as_targeting.md`) — read-only, Contabo, **$0 training**. Run
   first/always; it sharpens A/B by saying WHERE the field is exploitable.
2. **D** (`D_jamwall_only.md`) — cheapest control; isolates the imperfect-pool factor.
3. **A** (`A_explicit_restriction_rnr.md`) — the direct fix for H4's root cause.
4. **B** (`B_exploit_residual_head.md`) — most novel, highest build; the
   structurally-self-anchor-safe method (operator's structural favorite).

C runs on Contabo now (free). D/A/B run as ONE parallel pod burst on a
snapshot-restored pod (skip the build; re-smoke first). Coordinator caps
concurrent G-sum ≤ 24, each probe its own run dir + verdict JSON.

## POD ECONOMICS (this burst)

Re-rent from snapshot (§11/§12 RUNPOD_SETUP): ~minutes restore + 2-min smoke
(NOT the 1h build). D/A/B parallel ≈ 3–5 h ≈ **$1.5–2.5 compute**. Release the
instant the burst drains. Standing-cluster idle burn (~$8/day) is the failure
mode to avoid; burst-from-snapshot is strictly dominant below ~16 h/day use.

## COORDINATOR / SPECIALIST MODEL

One coordinator (Contabo, persistent) owns the queue + gates + operator
interface. Specialists = sub-agents, one probe each, worktree-isolated, each
emitting a falsifier-stamped report + verdict JSON to disk. Session-death-
recoverable: the burst rebuilds from these files. Throughput = PASSED/FAILED
verdicts, not runs launched.

## OPERATOR GATES (the irreversible buttons — coordinator may NEVER cross)

- **pod-spend** (rent / extend / snapshot storage)
- **escalate probe → program** (big compute)
- **ship / arm / deploy** any model live
- **weaken or re-scope any gate** (FORBIDDEN — re-registration only)

The coordinator autonomously: runs probes within an approved budget, writes
reports, proposes. It decides nothing that touches the bankroll.
