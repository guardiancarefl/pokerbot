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
3. **f1 ensemble probe** (post-H2-verdict; per slate 2c's spec once
   written — runs only with its pre-registered criteria committed).
4. **c1 ICM-gap probe** (NIGHT SHIFT class, ~6-10 h: design lands via
   slate 2e first; runs overnight when designed + no live play).
5. **b2 checkpoint-selection variance** (NIGHT SHIFT class, ~4 h:
   paired panel on 2-3 champion-run checkpoints).

## Night shift (Addendum 5.4)

Tonight's candidates in order: whatever of 1-2 remains, then 4 if its
design (slate 2e) has landed, else 5.
