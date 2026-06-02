# RL-CFR (`src/rlcfr/`)

Investigation into **RL-CFR** (Li, Fang, Huang — ICML 2024, arXiv:2403.04344):
dynamic action abstraction for imperfect-information games. Reported +64–84
mbb/hand over fixed-abstraction baselines (ReBeL replication, Slumbot) on HUNL.

## Status (2026-06-02)

**Foundation built and Leduc-validated. Full RL-CFR NOT built (held — see scope).**

What RL-CFR actually is, from the paper: a two-phase method.
1. **CFR backbone** = ReBeL (depth-limited subgame solving with DCFR(1.5,0,2),
   T=250, leaf values from an ~18M-param PBS value network trained by self-play).
2. **RL layer** = an actor-critic (DDPG-style, ~20K params) whose MDP has
   state = public state, action = a 2K-dim vector selecting up to K bet sizes
   (`f(x,y)=clip(2.5(x+1)×pot)`, y<0 drops the action), reward = CFR PBS-value
   of the chosen abstraction minus a default abstraction.

It is **not** a standalone blueprint solver — it is a search/resolving method.

### `cfr.py` — tabular (Discounted) CFR engine — VALIDATED

The CFR backbone, isolated and certified on Leduc (where Nash is known), since a
buggy solver invalidates everything above it. Full-tree, exact, deterministic;
supports vanilla / CFR+ / Linear / DCFR(α,β,γ); alternating (Gauss-Seidel) or
simultaneous updates; abstraction-aware (`abstraction_fn` hook for the dynamic
bet sizes RL-CFR will choose).

**Leduc gate (exact OpenSpiel best-response oracle):**

| variant | it100 | it300 | it1000 |
|---------|-------|-------|--------|
| CFR+    | 6.708 | 1.152 | **0.1234** |
| DCFR(1.5,0,2) | 4.068 | 0.596 | **0.0879** |

OpenSpiel reference CFR+ = 0.1286 @1000; repo Deep CFR reference ~0.13. **Match.**
Reproduce: `python scripts/validate_rlcfr_leduc.py`. Fast unit tests:
`pytest tests/test_rlcfr_cfr.py`.

Two real bugs were found and fixed during validation (both guarded by tests):
- **Frozen strategy / deferred commit:** an infoset is visited once per history,
  so regret must not be mutated mid-traversal — else later visits read a
  half-updated regret. Symptom was asymmetric 1-iteration regrets and CFR+
  converging *slower* than vanilla.
- **Alternating updates:** simultaneous updates converge ~1/√T; alternating
  (Gauss-Seidel) gives the ~1/T that matches OpenSpiel and reaches 0.13 @1000.

## What a faithful RL-CFR for our 6-max NLHE would take (scope)

The engine above is the cheap, reusable part. The rest is large:

- **PBS value network (the bulk):** ReBeL-quality leaf evaluation. The paper used
  ~6×10⁷ self-play PBS samples, an 18M-param net, **8 GPUs**. On CPU this is
  infeasible at quality (weeks+). On 1 modern GPU, realistically multi-day to
  ~1–2 weeks of self-play to a usable (not paper-grade) net — and our game is
  **6-max**, not heads-up, which enlarges the PBS (joint belief over more
  opponents) and the value-net target substantially beyond the 2-player paper.
- **Depth-limited subgame solver + safe/nested resolving** wiring around the net.
- **Actor-critic RL layer:** comparatively cheap (~30% of value-net cost per the
  paper) once the value net exists.

Bottom line: a faithful 6-max RL-CFR is a **GPU-scale, multi-week** build, gated
on the value net. Decision on whether to pursue it should follow the M2
(lightweight real-time-search) result — M2 is the cheap test of the same
search-helps hypothesis that justifies the heavy ReBeL/RL-CFR build.
