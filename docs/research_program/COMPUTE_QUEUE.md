# COMPUTE QUEUE — runnable experiments by EVoI per core-hour

Rule (Addendum 5.3): heavy-job completion → next item launches within
minutes, watcher-armed, tmux per Addendum 4.2. H2 chain keeps priority.
Live-listener stand-down overrides everything.

## Core ledger (update on every launch/completion)

| cores | owner | ETA |
|---|---|---|
| 4-11 | H2 probe (tmux h2_probe, iters 1500→2000) | ~16:40 |
| 0-1 | e2 re-baseline final row (gushansenmtt) | ~14:55 |
| 2-3 | a1 blur map (agent-run, build+measure+run) | ~18:00 |
| (8-9 light) | 5 file-only slate agents (2b/2c/2d/2e/2f) | rolling |

## Queue (next-up first)

1. **H2 verdict battery** (on probe completion; owns probe's cores):
   M1 grade of ckpt_iter_2000 vs frozen battery (~10 min, 2 cores) +
   CRN F-M2 rows: killphilmtt / ticketmaster / sng / tighttom 2000
   games each + self-anchor (6 jobs, 1-2 cores each, ~25 min wall
   parallel). Pre-registered: H2 spec §5. EVoI: gates the only live
   retrain hypothesis.
2. **a1 blur map — LAUNCHED on cores 2-3** (agent abc07b…; ~3 h run
   after harness build). Slate 2a mining happens inside its report.
3. **f1 ensemble probe — PRE-REGISTERED, READY**
   (`ENSEMBLE_PROBE_DESIGN.md`): P0 bit-identity smoke → P1 bubble
   screen (bar ≥+0.040 z≥2) → P2 CRN yardstick paired vs existing
   per-game jsonls (zero champion cost). ≤6 h wall / ~21 core-h.
   Launches post-H2-verdict (selector candidates partly answered by
   the running b2).
4. **c1 ICM-gap probe — DESIGNED, NIGHT SHIFT**
   (`ICM_GAP_STUDY_DESIGN.md`): tiered 17-43 core-h, 50-rollout rate
   benchmark gate first, MDE80 0.0093 at N=250×M=100. Tonight's
   primary night-shift item if cores free.
5. **b2 checkpoint-selection variance — RUNNING (cores 0-1)**:
   killphil row at ckpt_1300/1400. Doubles as ensemble-candidate
   screen (2c design).

## Night shift (Addendum 5.4)

Tonight's candidates in order: whatever of 1-2 remains, then 4 if its
design (slate 2e) has landed, else 5.
