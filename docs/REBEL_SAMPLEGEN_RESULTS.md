# ReBeL Phase-2 sample-gen — this-box rate + multi-box generation + persistence

Companion to `docs/REBEL_SAMPLEGEN_PILOT.md` (which measured the pilot on a
13.6-core slice at **~52,500 samples/hour**). This doc records the production
generation setup: the persistence layer, the measured rate on the RTX PRO 4500
box, and the multi-box (this box + Contabo) generation + merge plan.

## Persistence layer (`src/rebel/sample_io.py`)

Each solved hero decision is persisted as a `(PBS encoding, target)` sample
(docs/REBEL_PBS_6MAX.md §5):

- **PBS encoding** — the validated 236-dim `InfosetEncoder6Max` root feature
  (`feat`), whose layout is `[bucket one-hot (200)] ++ [public block (36)]`; the
  acting-seat bucket is stored separately as `bucket`. The pure public block is
  `feat[200:]`. (The soft 6×k per-seat belief block is the Phase-3 net-input
  assembly step, pre-registered in PBS doc §3–4; the shards carry everything
  needed to build it — many samples over the same public node ARE the empirical
  belief.)
- **Target** — `root_value` (hero root EV = Σ_a policy[a]·q[a]), plus the full
  `root_policy`, `root_q_values`, `legal_mask` (9-dim each) for the richer
  per-bucket head (B) later.
- **Provenance** — `hero_seat, street, depth, n_iterations, degraded` per sample;
  `host, worker, schema, blueprint_sha, abstraction_sha, k_postflop, feat_dim,
  n_actions` per shard.

Shards are `.npz` (numpy arrays only — no pickled custom classes, so a shard
written on Contabo loads byte-identically on the GPU box). Named
`shard-<host>-w<worker>-<ts>-<seq>.npz` → cross-box/worker collision-free; writes
are atomic (`tmp` + `os.replace`) and flushed every `--flush-every` samples, so a
disconnect loses at most one partial buffer, never a finished shard.

## Measured rate on the RTX PRO 4500 box

- Real CPU budget: **27.2 effective cores** (`cpu.max` = 2720000/100000; the host
  reports 128 — cgroup-limited, NOT host cores).
- 26 CPU workers, depth 3, PROFILE_SAMPLE M=1, K=150, persistence ON:

      33.65 samples/s  =  ~121,000 samples/hour   (steady-state)

  Measured over a 101s gen window (two probes agree: 34.2/s and 33.6/s). NOTE:
  the rate is samples / the workers' parallel *gen window*, NOT samples / total
  wall — the first ~160s of a run is one-time startup (26 spawn processes each
  importing torch-cu130 + loading the 10.9 MB checkpoint), which deflates a
  naive total/wall to ~47k/hr on a short probe but amortizes to nothing over a
  multi-hour run. ~2.3× the 13.6-core pilot's 52.5k/hr (2× cores + faster per-
  core), consistent with generation being linear in cores. Bootstrap-round
  leaf-eval is the CPU rollout; the GPU is reserved for Phase-3 value-net training.

  1M ETA on this box alone: 1,000,000 / 121,000 ≈ **8.3 h**.

## Multi-box generation

Both boxes run the SAME script against their own `/workspace`; the host prefix
keeps shards distinct, so the union of both shard dirs merges losslessly.

This box (RTX PRO 4500, ~27 cores):

    cd /workspace/pokerbot && export CUDA_VISIBLE_DEVICES=""   # workers force CPU anyway
    .venv/bin/python scripts/rebel_samplegen.py \
        --workers 26 --target 1000000 --out /workspace/rebel_samples --flush-every 256

Contabo (~12 cores) — fire simultaneously:

    cd /workspace/pokerbot && git pull origin rebel-search   # get sample_io + samplegen
    .venv/bin/python scripts/rebel_samplegen.py \
        --workers 11 --target 1500000 --out /workspace/rebel_samples --flush-every 256

Split the ~2–3M target by each box's core share (RTX ~27 cores, Contabo ~12 →
~69% / 31%). At ~39 cores combined the bootstrap-round rate is ~140k/hr, so
~2–2.5M is ~a day.

## Merge plan

`scp` Contabo's shards into a shared dir alongside this box's (any layout — the
merger globs `shard-*.npz` recursively per root), then:

    # inspect / count across boxes
    .venv/bin/python -m src.rebel.sample_io /workspace/rebel_samples /workspace/rebel_samples_contabo
    # materialise one merged training set for the value net
    .venv/bin/python -m src.rebel.sample_io \
        /workspace/rebel_samples /workspace/rebel_samples_contabo --out /workspace/rebel_bootstrap_merged.npz

The merger validates `feat_dim`/`n_actions` consistency across every shard and
raises on drift (the cross-box silent-dim-mismatch guard). `blueprint_sha` /
`abstraction_sha` in each shard's meta confirm both boxes generated against the
identical k=200 artifacts.

## Scale path (bootstrap-then-accelerate)

Bootstrap ~1–2M (this round, ~a day) → train a ROUGH value net (sized to the
data, ~hundreds-k–1M params) → **GATE 2 EARLY**: does search + rough net beat
k=200 on tournament cash-rate vs tight archetypes? If yes → use the net's ~100×
faster GPU leaf-eval to generate tens of millions cheaply and train toward a
paper-grade (~18M param) net, iterating the self-play loop; held-out accuracy
guides net size each round. If the rough net fails Gate 2, scaling to 18M is
unlikely to fix it (search isn't the lever for the fixed-archetype case) — stop
before the ~18-day full-scale investment.
