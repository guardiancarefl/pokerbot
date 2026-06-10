# Parallel training for the C3 recipe — wiring, gate, scaling

**Date:** 2026-06-10. **Branch:** `runpod-env`. **Status:** ARMED BUT
UNUSED — the live `runs/c3_retrain_v1` run continues single-process;
this mode is for the next launch.

## What changed

The G-worker mp.fork framework (`src/nlhe/parallel/`, validated at
G=10 on Contabo) now supports the full C3 recipe. Three deltas:

1. `WorkerInput` grew `encoder_eff_bb: bool = False` and
   `empirical_dist_path: Optional[str] = None` (defaults preserve
   pre-C3 behavior bit-for-bit).
2. `worker.py` constructs the encoder with the eff-BB channel and, in
   tournament mode with an empirical artifact, draws the per-traversal
   hand start via `rng_stack_t.choice(rows)` — statement-for-statement
   the same branch solver6's sequential loop runs, on the same
   `STACK_SAMPLE_SALT` RNG stream, so the draw is identical across the
   fork boundary. Rows are adopted into a module-level cache by the
   orchestrator BEFORE the first fork (`preload_empirical_rows`), so
   the 322k-row artifact crosses by fork CoW — workers never re-parse
   the gz.
3. The `TrainConfig6Max` guard that refused `parallel_groups > 0` with
   the C3 deltas (old `solver6.py:302`) is removed.
   `scripts/train_c3.py` gained `--parallel-groups G`; absent/0 routes
   to the identical `solver.train()` call as before (the live run's
   path — see the gate below for the proof of equivalence; the
   sequential code path itself was not edited).

Tests: `tests/test_parallel_c3.py` (WorkerInput defaults; in-process
G=3 and mp.fork G=2 bit-identity on the full C3 feature set);
`tests/test_c3_bundle.py::test_empirical_parallel_now_allowed` replaces
the old guard test. Full parallel-suite regression: 8 passed / 7
skipped (the known `baseline_fork_A` artifact skips), no new failures.

## Bit-identity gate — PASS (the original framework standard)

Production config (`configs/c3_retrain_k200.yaml`), 10 iterations, same
seed: sequential vs `G=8 use_processes=True`, run 2026-06-10 on the
RunPod 27-core box (`/tmp/gate10.py`, scratch dirs
`runs/parallel_validation_{A,B}`). Verbatim:

```
=== COMPARISON ===
metrics[      iter]: IDENTICAL
metrics[ traverser]: IDENTICAL
metrics[  adv_loss]: IDENTICAL
metrics[strat_loss]: IDENTICAL
metrics[ strat_buf]: IDENTICAL
metrics[     buf_0]: IDENTICAL
metrics[     buf_1]: IDENTICAL
metrics[     buf_2]: IDENTICAL
metrics[     buf_3]: IDENTICAL
metrics[     buf_4]: IDENTICAL
metrics[     buf_5]: IDENTICAL
6 adv-net param sets     : IDENTICAL
strategy-net params      : IDENTICAL
adv buffer seat 0 (contents+n_seen+rng): IDENTICAL
adv buffer seat 1 (contents+n_seen+rng): IDENTICAL
adv buffer seat 2 (contents+n_seen+rng): IDENTICAL
adv buffer seat 3 (contents+n_seen+rng): IDENTICAL
adv buffer seat 4 (contents+n_seen+rng): IDENTICAL
adv buffer seat 5 (contents+n_seen+rng): IDENTICAL
strategy buffer (contents+n_seen+rng): IDENTICAL
solver python RNG state  : IDENTICAL
torch RNG state          : IDENTICAL
override counts          : IDENTICAL {'archetype': 0, 'league': 0, 'self_play': 150}
GATE: PASS — BIT-IDENTICAL
```

The two FULL checkpoint files hash differently
(`b26a0d1b…` vs `c6731a9e…`) for exactly one reason, verified by
key-by-key diff: the checkpoint records its config, and
`config_dict.parallel_groups` is 0 vs 8. Every tensor, every buffer
array, every RNG state is equal. No state divergence exists.

## Scaling (27.2-core cgroup, live trainer co-resident)

Protocol: every leg resumes the SAME steady-state checkpoint (Contabo
c3 iter-200 weights — the 21.8 s/iter single-process benchmark
protocol), 10 iterations (201–210), steady = mean of the last 8.
`OMP/MKL/OPENBLAS_NUM_THREADS=1` pinned in all workers (torch is
already pinned by solver6). Measured legs capped at G=12 — the live
production trainer shares this box and its pace was re-verified at its
intrinsic trajectory (ratio ~0.43–0.47 of Contabo's same-iteration
times) before and after each batch; G>12 is extrapolated, not measured.

| G | s/iter | speedup | note |
|---|---|---|---|
| seq (G=0) | 23.57 | 1.00x | sequential path, same-batch reference |
| 1 | 24.17 | 0.98x | orchestrator overhead ≈ 2.5% |
| 4 | 8.73 | 2.70x | |
| 8 | 6.67 | 3.53x | **knee starts here** |
| 12 | 5.70 | 4.13x | best measured |
| 16 (extrap.) | ~5.2 | ~4.5x | Amdahl fit, see below |
| 20 (extrap.) | ~4.9 | ~4.8x | |
| 26 (extrap.) | ~4.7 | ~5.0x | ceiling |

Amdahl fit from the G=8/G=12 pair: parallelizable ≈ 23.3 s, serial
floor ≈ 3.8–4.2 s/iter (advantage+strategy train steps, sample merge,
per-iter fork/ship). **The knee is G≈8–12**: doubling workers beyond
G=12 buys < 1 s/iter. Recommended launch value: **G=12** (leaves 14+
cores for dashboard/mirror/OS on this box; G=20+ is not worth the
contention even on an idle box).

Full-retrain projection at G=12: 2000 iters at the fixed-workload ratio
(4.13x) → **≈ 3.2 h** against the 21.8 s/iter-equivalent baseline
(measured-protocol single-process ≈ 12 h). Caveat: per-iter cost is
iteration-dependent (both boxes show a ~2x hump around iters ~90–130
that recedes by ~200), so treat absolute hours as ±50%; the SPEEDUP
ratio is the stable number.

## Launch command for a future parallel C3 retrain

```bash
tmux new-session -d -s c3_train \
  "cd ~/pokerbot && source .venv/bin/activate && \
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
   PYTHONUNBUFFERED=1 python -m scripts.train_c3 \
     --config configs/c3_retrain_k200.yaml \
     --parallel-groups 12 \
     --out runs/<run_name> \
   > runs/<run_name>_train.log 2>&1"
```

Checkpoint cadence, gates G1–G4, dashboard, and mirror are unchanged —
checkpoints from a parallel run are bit-identical to sequential ones,
so every downstream consumer (dashboard anchors, conventions gate,
bridge) is unaffected. Resume works across modes for the same reason:
a sequential run can resume a parallel checkpoint and vice versa.

## Live-run protection record (2026-06-10)

All validation ran in `runs/parallel_validation_*` scratch dirs, ≤ 13
cores at peak (gate G=8: 9; sweep max G=12: 13), thread-pinned. The
live trainer's sec/iter was sampled idle before/after each batch and
compared against Contabo's logs at the SAME iterations (the workload
has an intrinsic mid-run hump, so raw sec/iter is not a valid
degradation signal): ratios stayed 0.41–0.49 throughout, identical to
the pre-test baseline. One early unpinned sweep batch (G=16/20 legs
pending) was killed on operator instruction and re-run pinned at ≤G=12.
