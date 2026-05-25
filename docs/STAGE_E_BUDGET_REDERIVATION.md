# Q13 — Subgame leaf-evaluator budget re-derivation

**Session 16, 2026-05-25.** Supersedes the arbitrary `Z = 1.5 s` real-time budget
(design doc Q4/Q12) with a measurement-grounded target derived from what sub-step 6
actually needs. The short answer: **the current evaluator (post-reset-fix, BR M=5
depth-3) is already adequate for sub-step 6; no further optimization is required
before sub-step 3. Stage E.5/E.6 stays shelved.** The long-form reasoning,
measurements, and the surprises that fell out of measuring follow.

> **Status of every number below:** measured fresh this session via
> `scripts/bench_subgame_leaf.py` (decomposition) and
> `scripts/bench_subgame_parallel.py` (parallelism), per the Q4.5
> measurement-discipline rule. Each bottleneck candidate is reported as
> **per-call cost AND call frequency AND post-cache-miss frequency** (the Q4.5
> sub-rule). Absolute wall-clock is not comparable across sessions (box load
> varies ~10×); the *attributions and ratios* are the durable facts.

## Measurement provenance

- **Box:** 128 vCPU + 1× NVIDIA RTX PRO 4000 Blackwell GPU (this measurement
  host — **NOT** the CLAUDE.md Contabo VPS, which has no GPU and 12 vCPU, nor the
  planned RunPod RTX 4090). The hardware conclusion below is re-checked against
  the 12-vCPU Contabo case explicitly.
- **Blueprint:** `runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight/checkpoints/ckpt_iter_3000.pt`
  (`dcfr-overnight-3000`), `starting_stack=1500`, `hidden_dim=[64,64]`,
  `bucket_runouts=20`.
- **Tree:** production game `six_max_sng(starting_stack=1500)`, post-flop decision,
  `chance_samples_per_node=2`. depth-3 → **64 leaves** (the established baseline);
  depth-4 → 205 leaves.

---

## Measurement set 1 — current evaluator decomposition (GPU box)

| config | wall | leaves | network calls | net/call | **net %** | bucket-MC misses | bucket/call | **bucket %** | **glue %** | encoder hit |
|---|---|---|---|---|---|---|---|---|---|---|
| PROFILE M=8 d3 | 2.07 s | 64 | 2,503 | 0.206 ms | 25.0% | 104 | 10.40 ms | **52.2%** | 14.6% | 95.8% |
| BR M=5 d3 | 13.81 s | 64 | 29,846 | 0.207 ms | **44.7%** | 197 | 10.42 ms | 14.9% | 25.8% | 99.3% |
| BR M=8 d3 | 20.50 s | 64 | 47,600 | 0.201 ms | 46.7% | 216 | 10.39 ms | 10.9% | 27.0% | 99.5% |
| BR M=5 d4 | 31.92 s | 205 | 75,969 | 0.197 ms | 47.0% | 275 | 10.15 ms | 8.7% | 29.0% | 99.6% |

(remaining % = parse ~4.5% + view/discretize ~4.1% + encode-rest ~6.5%.)

**The bottleneck is mode-dependent — both prior framings were partial:**

- **BR mode (production):** the **network forward dominates (~45%)**, glue is
  second (~26%), bucket-MC is only ~15%. This confirms session-15's Q12 inversion:
  per-call, bucket-MC (10.4 ms) ≫ network (0.21 ms), but the **call counts invert
  it** — 197 bucket misses vs 29,846 network forwards. The 99.3% encoder hit rate
  means the Stage-E shared cache already does what an E.5 precompute would.
- **PROFILE mode:** bucket-MC genuinely **dominates (52%)** — but only because
  PROFILE makes ~12× *fewer* network calls (no `v×k` best-response eval phase), so
  the same 104 misses are now the largest slice. PROFILE is cheap in absolute
  terms (2 s).

**Dropping M 8→5 was a real win** (BR d3 20.5 s → 13.8 s, −33%): network calls
scale linearly in M (47,600 → 29,846), and network is the dominant cost, so M is a
real lever — contra the session-14 claim that "M is not the budget lever." It is
*a* lever; it just was not the *only* one.

**Glue (~26% in BR) is a newly-attributed cost** — `state.clone()`,
`apply_action`, chance sampling, `icm_adjust_returns`, action sampling, numpy
masking, and BR orchestration. It is second-largest after the network and would
become the bottleneck if the network were ever batched away.

### CPU is faster than GPU here (surprise — see §Surprises)

| config | GPU wall | **CPU wall** |
|---|---|---|
| BR M=5 d3 (64 leaves) | 13.8 s | **10.0 s** |
| BR M=5 d2 (19 leaves) | — | 3.7 s |
| PROFILE M=8 d3 | 2.1 s | 1.8 s |

The single-row `[64,64]` MLP forward is kernel-launch/host-transfer-bound on the
GPU; a CPU matmul of that size is faster. **The GPU is the wrong device for this
evaluator as written** (it only wins when forwards are *batched* — which the
rollout loop does not do).

---

## Measurement set 2 — throughput requirement for sub-step 6

**Sub-step 3 calls `evaluate_leaves` once per decision, cached across all CFR
iterations** — confirmed in design doc Q6 ("Compute leaf values once per decision,
cache across all CFR iterations within that decision … Z stays a one-time cost,
not Z×W"). The Brown/Sandholm leaf value depends only on `(leaf state, k biases,
menu, hero blueprint)`, none of which the CFR loop mutates. So **per-decision cost
= one `evaluate_leaves` call**, and the sub-step-3 CFR loop adds only cheap
root-region regret-matching over cached leaf values. *Confirmed, not ambiguous —
proceeded.*

Sub-step 6 = Q11 Level-3 pool ablation:
- **Pool: 5 opponents** (`evals/league-v2-600_vs_pool_5000.json`: vanilla-200,
  vanilla-400, dcfr-shake-100, dcfr-shake-200, dcfr-overnight-600).
- **5,000 hands/matchup** (pinned in that file — *not* underspecified).
- **3 challengers:** (1) pure blueprint — *no* subgame solve, ~free; (2) subgame +
  PROFILE_SAMPLE; (3) subgame + BEST_RESPONSE. Only (2) and (3) invoke the leaf
  evaluator.
- **Decisions per hand:** measured 6.08 total decisions/hand across the turbo-SNG
  sampling distribution (15bb Double-Up — mostly preflop push/fold; alive-count
  mix 6:52% / 5:28% / 4:20%). The challenger is assigned a random ~half of seats,
  so **~3.04 challenger decisions/hand**.

**Decisions per subgame challenger** = 5 × 5,000 × 3.04 ≈ **76,000 subgame solves.**

**Sequential wall-clock (CPU per-decision cost):**

| challenger | per-decision | × 76,000 sequential |
|---|---|---|
| BR M=5 d3 | 10.0 s | 211 h ≈ **8.8 days** |
| PROFILE M=8 d3 | 1.8 s | 38 h ≈ 1.6 days |
| blueprint | ~0.5 ms | ~minutes |

BR is the binding constraint. Sequential, the full ablation is ~10.5 days — which
is why parallelism is the whole game.

---

## Measurement set 3 — parallelism (the decisive measurement)

Barrier-synced worker processes, each its own solver, one BR M=5 depth-3
`evaluate_leaves` call; contention slowdown = mean(parallel compute) / solo
compute; parallel speedup = W / slowdown.

| config | solo | parallel (mean) | slowdown | **speedup** |
|---|---|---|---|---|
| GPU shared, W=4, 4 thr | 13.85 s | 18.57 s | 1.34× | **2.98× of 4** |
| CPU, W=4, 1 thr | 9.95 s | 9.80 s | 0.98× | **4.06× of 4** (linear) |
| CPU, W=32, 1 thr | 9.75 s | 13.15 s | 1.35× | **23.7× of 32** (~74% eff) |

- **The shared GPU partially serializes** (4 workers → ~3× throughput): one device,
  many tiny forwards queued.
- **CPU with 1-thread workers scales near-linearly** (4.06× of 4; 23.7× of 32) AND
  is faster per call. **The right hardware for sub-step 6 is the many-core CPU box
  with N single-threaded worker processes, not the GPU.**

**Parallelism factor used below: Y = 24** (directly measured at W=32 on this
128-core box; scales further toward ~50–64 with more workers at declining
efficiency, but 24 is the conservatively-measured anchor). On the 12-vCPU Contabo,
Y ≈ 10.

---

## Q13-A — What does sub-step 6 actually need?

| quantity | value |
|---|---|
| matchups (per subgame challenger) | 5 opponents |
| hands / matchup | 5,000 |
| challenger decisions / hand | ~3.04 |
| **total decisions (BR challenger)** | **~76,000** |
| acceptable total wall-clock | **1 day** (comfortably overnight; weekend = generous) |
| parallelism factor Y | **24** (measured, 128-core CPU; ~10 on Contabo) |

Required cost-per-decision = (acceptable_wall × Y) / total_decisions
= (86,400 s × 24) / 76,000 ≈ **27.3 s/decision**.

> **Required cost-per-decision for sub-step 6 to be feasible: ~27 s/decision, at
> parallelism factor Y = 24** (1-day budget). At a weekend budget (~2.5 days) the
> bar relaxes to ~68 s/decision.

---

## Q13-B — Does the current evaluator meet that target?

**Yes, with margin.** Current BR M=5 depth-3 = **10.0 s/decision** (CPU) vs the
**27.3 s/decision** required → **2.7× headroom** at the 1-day budget (6.8× at a
weekend budget).

Projected sub-step 6 wall-clock at Y = 24:
- BR challenger: 76,000 × 10.0 s / 24 ≈ **8.8 h**
- PROFILE challenger: 76,000 × 1.8 s / 24 ≈ **1.6 h**
- blueprint challenger: ~minutes
- **Full Level-3 ablation ≈ 10.5 h — fits comfortably overnight.**

Even on the 12-vCPU Contabo (Y ≈ 10): BR ≈ 21 h, PROFILE ≈ 3.8 h — a weekend job,
still feasible.

**Conclusion: the evaluator is already adequate for sub-step 6. Stage E.5/E.6 work
is shelved. The next implementation work is sub-step 3 (the subgame CFR loop).**

---

## Q13-C — What does deployment (sub-step 5 real-time play) need?

Different use case, different metric. Deployment is **single-decision latency**
(decisions within a hand are sequential — *not* parallelizable across the 128
cores the way the ablation is). The project targets a 6-max SNG bot; for online
SNG play, **5–30 s/decision is socially tolerable, 1–3 s comfortable.**

Current single-decision latency:
- BR M=5 depth-3: **~10 s** — within the tolerable band, at its high end.
- BR M=5 depth-2: **~3.7 s** — comfortable.
- PROFILE M=8 depth-3: **~1.8 s** — comfortable.

**Deployment hits a comfortable latency by parameter selection, not new
optimization:** drop to depth-2, or lower M, or PROFILE mode. The deployment
evaluator and the ablation evaluator are the **same code path** with different
`LeafEvalContext` parameters — ablation maximizes throughput (full depth-3 BR M=5,
massively parallel); deployment minimizes latency (depth-2 / lower M as needed).
They do not need the same optimization, and neither needs a new stage.

> Note: the design doc's "BEST_RESPONSE is the production / GPU target" framing
> (LeafEvalMode docstring, Q4) is now wrong on the hardware: production should run
> on **CPU**, and the GPU offers no benefit at single-row forward sizes.

---

## Q13-D — What's the right optimization, if any?

**Outcome: no further optimization needed.** Current evaluator (post-reset-fix)
meets the sub-step 6 throughput target (2.7×–6.8× headroom) and the deployment
latency target (via parameter selection). **Stage E.5/E.6 is shelved.** This is
triply confirmed:

1. Bucket-MC (the E.5 target) is only ~15% of BR cost; the Stage-E shared cache
   already achieves 99.3% hit, leaving ~0 for a precompute to recover.
2. The dominant cost (network forward, ~45%) is *already cheaper on CPU* than GPU,
   and CPU parallelism is near-linear — so throughput scales with cores for free.
3. The `Z = 1.5 s` real-time premise was never the sub-step-6 constraint: the
   ablation is throughput-bound and parallelizes to an overnight job.

**If a future real-time deployment ever demands sub-3 s BR at depth-3** (not a
current requirement), the measured highest-value lever is **network-forward
batching** via lockstep rollouts: the network is ~45% of BR cost at 0.21 ms/call
single-row, and the batched path is 0.45 µs/row (~440× per-row headroom). The
second lever would then be **glue** (~26%), which batching would expose as the new
bottleneck. This is recorded as a *contingent* Stage E.6, **not scoped as required
work** — open it only against a concrete latency requirement, and re-measure
first (it is a hypothesis until then).

**Budget closure check:** the only "optimization" sub-step 6 needs is *running on
the CPU box in parallel*, which is a deployment choice, not a code change. With it,
cost-per-decision (effective) = 10.0 s / 24 ≈ 0.42 s, against the 27.3 s/decision
budget — closed by ~65×. The gap from Q13-B is **already closed**.

---

## Surprises (the fifth instance of the discipline pattern)

The measurement-discipline note predicts each re-measurement surfaces new findings.
This session's:

1. **CPU beats GPU for this evaluator** (BR M=5 d3: 10.0 s CPU vs 13.8 s GPU). The
   single-row `[64,64]` forward is launch/transfer-bound on the GPU. Every "GPU
   production target" assertion in the design doc is wrong on the hardware.
2. **The shared GPU partially serializes** (2.98× of 4), while **CPU 1-thread
   workers scale near-linearly** (23.7× of 32). The deployment hardware question
   answers decisively in favor of many-core CPU.
3. **The bottleneck is mode-dependent**: network dominates BR (~45%), bucket-MC
   dominates PROFILE (~52%). Neither the original Q4 framing nor the session-15
   correction was complete — *both* modes had to be measured.
4. **Glue is ~26% of BR cost** — previously unattributed; it is the second-largest
   slice and the next bottleneck behind any network speedup.
5. **M *is* a budget lever after all** (8→5 cut BR d3 by 33%), because network
   forwards scale linearly in M and the network dominates BR — directly contra the
   session-14 "M is not the lever" note. The session-14 claim was made when
   bucket-MC was (wrongly) believed to dominate; under the correct attribution, M
   moves the dominant cost.
