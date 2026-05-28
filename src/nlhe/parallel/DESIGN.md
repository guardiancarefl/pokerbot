# Parallel traversal for 6-max Deep CFR — design spec

## Goal
Parallelize the per-iteration traversal loop in DeepCFR6MaxSolver.train()
across processes, with results BIT-IDENTICAL to sequential forked training.
End goal: make experiments cheap enough to run the self-play-vs-diversity-mix
comparison. Speed is the means; the architecture experiment is the point.

## Non-negotiable: bit-identity gate
A parallel run MUST reproduce runs/baseline_fork_A/metrics.json bit-for-bit
(iter, traverser, adv_loss, strat_loss, strat_buf, buf_0..buf_5) at full
IEEE-754 precision. Gate also checks: override counts (self_play/archetype/
league tallies) match sequential. Not just the metrics.json loss/buffer arrays.
This is the correctness gate. Anything less is a bug.

## What is parallelized
ONLY the `for t in range(traversals_per_iter)` loop. Each traversal is
independent: own forked rng_t = Random(seed*1_000_003 + it*9_973 + t),
read-only nets (training happens after the loop), and write-only buffer
access (deferred — see merge).

## What stays sequential (orchestrator-side)
- Network training (_train_advantage_net, _train_strategy_net)
- Buffer add() calls — consume orchestrator-side n_seen + buffer rng
  (seed+1 / seed+100); workers NEVER call add()
- Checkpointing, metrics, encoder cache lifecycle (reset per iter)

## Per-iter process model (v1): fork, re-partition each iter
- Each iter: fork N workers (multiprocessing), partition the T traversals
  across them, collect results, join.
- Use multiprocessing.get_context('fork') on Linux/CPU (copy-on-write
  parent memory, no torch reinit per worker). NOT 'spawn' (only needed
  for CUDA/Windows; neither applies).
- INVARIANT: workers MUST NOT rely on inherited global RNG or torch global
  state — ALL randomness comes from the explicit per-traversal fork
  rng_t = Random(seed*1_000_003 + it*9_973 + t). predict_advantages is a
  deterministic forward pass (no RNG). This invariant is what makes fork
  safe for bit-identity.
- Rationale: dead-simple determinism (fresh processes, no carried state, no
  weight-broadcast staleness). Optimize to persistent pool LATER, only if
  speedup measurement at a realistic config shows spawn overhead dominates.
- Worker count: configurable, default min(N_traversals, n_cores-ish). On the
  Contabo box, account for ~/research daemons consuming cores.

## Worker contract
Input (picklable, sent to each worker):
- The 6 ADVANTAGE-net state_dicts ONLY. Worker rebuilds a minimal nets stub
  exposing predict_advantages(seat, features). The full PlayerNetworks6Max
  (optimizers, buffers, strat_net, rng) does NOT cross the process boundary
  — strat_net is the average policy, never read during traversal.
- abstraction_path. Worker reloads the abstraction from abstraction_path
  (not a pickled Abstraction object) and constructs its OWN fresh
  InfosetEncoder6Max with empty _bucket_cache. Encoders are never shared
  across workers; bucket lookups are pure functions of (hero, board) so
  cold cache is correct.
- Encoder construction params: starting_stack, max_bucket_dim, bucket_runouts.
- ctx scalar fields: starting_stacks, payouts, iteration, max_depth,
  num_paid, dealer_seat.
- The game STRING. Worker calls pyspiel.load_game() locally; pyspiel.Game
  objects are never pickled.
- (seed, iteration, assigned t-list) for deterministic rng_t fork derivation
  and the worker's traversal assignment.

Process: for each t in assigned t-list, build rng_t deterministically, build
a fresh game state, run traverse_6max with buffer writes REDIRECTED to local
lists (not buffer.add()).

Output (picklable): for each t, (t, adv_samples: list[tuple], strat_samples:
list[tuple]) — tagged with t so the orchestrator can order them.

Confirmed complete worker input set (no hidden globals in the traversal hot
path): 6 adv-net state_dicts; abstraction_path; encoder construction params
(starting_stack, max_bucket_dim, bucket_runouts); ctx scalars
(starting_stacks, payouts, iteration, max_depth, num_paid, dealer_seat);
game string; and (seed, iteration, assigned t-list).

## The bit-identity mechanism: fixed-order merge
Orchestrator collects all workers' tagged samples, then replays them into the
buffers in STRICT ASCENDING t ORDER (t=0's samples, then t=1, ...), regardless
of worker completion order. Because buffer add() consumes only orchestrator-
side state, fixed-order replay reproduces the sequential buffer state exactly.
Wall-clock completion order is discarded at merge.

The orchestrator ALSO calls _maybe_sample_league_opponent() T times in STRICT
ASCENDING t ORDER during the merge phase, interleaved correctly with sample
replay so override bookkeeping matches sequential. At mix=0 this is a pure
_count_override('self_play') increment with zero rng draws — but it MUST
still happen T times or the override counter (and _log_enhanced output)
diverges from sequential. At mix>0 (out of scope v1) it draws from self.rng
and needs its own fork.

THIS is what makes parallel == sequential.

## Determinism prerequisites (verified)
- Per-traversal rng fork: committed a8416b2.
- Forked sequential is run-to-run bit-identical: verified (baseline_fork_A==B).
- Encoder bucket lookup is a pure function of (hero, board) — rng discarded
  (abstraction.bucket_of: del rng); cache is memoization-only, order-independent.
- Worker rng MUST use explicit integer arithmetic, NOT hash() (hash is salted
  per-process; would break cross-process determinism).

## KNOWN LIMITATION — mix>0 not yet bit-identical-safe
At league_mix=0 AND archetype_mix=0 (baseline_seq.yaml), the
_maybe_sample_league_opponent() override is a no-op (zero rng draws). At mix>0
it DRAWS from self.rng between traversals — which the per-traversal fork does
NOT yet cover. The self-play-vs-diversity-mix EXPERIMENT requires mix>0, so
before that experiment can run parallel, the override-sampling RNG must also be
forked deterministically per-traversal. Scoped as a FOLLOW-UP, not v1. v1 proves
bit-identity at mix=0.

## Validation plan
1. Correctness: parallel run at baseline_seq.yaml == baseline_fork_A bit-for-bit
   (metrics + override counts).
2. Speedup: measured at a REALISTIC config (hundreds of traversals, larger nets),
   NOT the smoke config — smoke traversals are too short to amortize spawn cost.
   Report honest numbers; do not claim smoke-config speedup.
3. Decision: if spawn overhead dominates at realistic scale, implement persistent
   pool (broadcast weights per iter) as v2, gated bit-identical against v1.

## Implementation files
Planned files (all under src/nlhe/parallel/):
- protocol.py — picklable input/output dataclasses (worker request/response).
- worker.py — worker entry: build rng_t, fresh encoder, adv-net stub, run
  traverse_6max with buffer writes redirected to local lists, return tagged
  samples.
- orchestrator.py — per-iter fork/partition/collect/fixed-order-merge harness.
- runner.py — scripts/train_6max_parallel.py entry, mirrors train_6max.py.

The existing solver6.py train() is NOT modified for v1; the orchestrator drives
the parallel loop using solver6's components.

## File boundary
All parallel code lives in src/nlhe/parallel/. Existing src/nlhe/ files are NOT
edited for v1 (the fork they depend on is already committed). The eventual
mix>0 override fork WILL touch solver6.py — surface for sign-off when we reach it.
