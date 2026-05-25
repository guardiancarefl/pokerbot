# Session 16 — 2026-05-25

Focus: **Q13 — the subgame leaf-evaluator budget re-derivation.** A strategic
(non-code) session: measure what sub-step 6 actually needs, decide whether any
further evaluator optimization is required before sub-step 3, and reconcile the
record. Conclusion: **no optimization needed — the current evaluator is adequate;
sub-step 3 is next.** Plus three cleanup commits restoring/correcting decisions
the Q13 measurements overturned.

(There is no `session_15_summary.md`: session 15 was the two correction commits
`3109fb0` + `90a3492` only — captured in those commit messages and the inline
session-14 corrections, not a separate summary file.)

## What was done

- **Q13 budget re-derivation** (`b092480`). New `docs/STAGE_E_BUDGET_REDERIVATION.md`
  answering Q13-A..D, plus a Q13 section + Q4.5 sub-rule in
  `docs/SUBGAME_LEAF_DESIGN.md`, plus two reproducible benchmark harnesses
  (`scripts/bench_subgame_leaf.py` decomposition, `scripts/bench_subgame_parallel.py`
  parallelism). Every number measured fresh with per-call cost **and** call
  frequency **and** cache-miss frequency.
- **Restore `n_samples` default 5 → 8** (`db89145`). The session-14 drop to 5 was
  made under the wrong cost attribution; M is a real budget lever (Q13). Docstring
  rewritten; `test_constructs_with_defaults` updated; full leaf suite green
  (24 passed, 10 subtests).
- **Session-14 corrections** (`23e3d8b`). Two inline notes (session-15 pattern):
  "M is NOT the lever; Stage E.5 is" is wrong on both clauses; "GPU available but
  irrelevant" understates it — GPU is *slower* than CPU here.
- **Runtime-state note** (`b12d471`). `NEXT_SESSION.md`: Contabo is nominal,
  RunPod instances are the live dev box; Q13 conclusion holds on both.
  `ARCHITECTURE.md`/`DECISIONS.md` deliberately left untouched.

## What was decided

- **Stage E.5/E.6 is shelved.** Not "deferred until sub-step 6" (the session-14
  framing) — *shelved*. The current post-reset-fix evaluator meets sub-step 6's
  throughput target with headroom; no budget-closing optimization is needed.
- **Next implementation work is sub-step 3** (the subgame CFR loop), not an
  optimization stage. Stages F/G (Q11 Level-1/2 ablations) and sub-step 3 are the
  open choice for next session (see State at close).
- **Production runs on CPU, not GPU.** A contingent network-batching stage is
  recorded for *real-time deployment latency only*, to be scoped against a concrete
  requirement and re-measured first — not now.
- Decisions recorded inline in `docs/STAGE_E_BUDGET_REDERIVATION.md` (Q13-A..D) and
  design-doc Q13; no new `DECISIONS.md` entry (this overturns/grounds prior
  in-flight choices rather than locking a new architectural one).

## What was learned / measured

Sub-step 6 = Q11 Level-3 ablation: 2 subgame challengers × **5 opponents** ×
**5,000 hands** (both pinned in `evals/league-v2-600_vs_pool_5000.json`) ×
**~3.04 challenger decisions/hand** (measured) ≈ **76,000 subgame solves per
challenger**. Per Q6, one `evaluate_leaves` call per decision, cached across CFR
iterations (confirmed, not ambiguous).

Decomposition (per-call **and** call frequency — the Q4.5 sub-rule), BR M=5 d3
64-leaf: network forward **44.7%** (29,846 calls × 0.21 ms), glue **25.8%**,
bucket-MC **14.9%** (197 misses × 10.4 ms, 99.3% cache hit), parse/view ~8%.
PROFILE inverts (bucket-MC 52%) because it makes ~12× fewer network calls.

Required cost-per-decision ≈ **27 s** (1-day budget, parallelism Y=24); current
BR M=5 d3 = **10 s on CPU** → adequate with 2.7× headroom; projected ablation
≈ **10.5 h** (~21 h on Contabo at Y≈10). Budget closed by parallelism, not by
any code change.

**Five findings (the fifth-opportunity prediction held):**
1. **CPU beats GPU** for this evaluator (10.0 s vs 13.8 s, BR M=5 d3) — single-row
   `[64,64]` forward is launch/transfer-bound on the GPU.
2. **Shared GPU partially serializes** (2.98× of 4); **CPU 1-thread workers scale
   near-linearly** (4.06× of 4, 23.7× of 32). Deploy on many-core CPU.
3. **Bottleneck is mode-dependent** — network dominates BR (~45%), bucket-MC
   dominates PROFILE (~52%); both prior framings were partial.
4. **Glue is ~26% of BR** — previously unattributed; the next bottleneck behind
   any network speedup.
5. **M *is* a budget lever** (8→5 = −33%), directly contra session-14 — that note
   was written under the wrong (bucket-MC-dominant) attribution.

Surfaced/contextual: the measurement box was a 128-core + RTX PRO 4000 Blackwell
RunPod instance (not CLAUDE.md's Contabo). The Q13 conclusion is hardware-robust.

## State at close

- **Done:** Q13 resolved; Stage E.5/E.6 shelved; `n_samples` default restored to 8;
  session-14 + runtime-state records reconciled. Leaf suite green.
- **Open (sub-step 2 closure):** Q11 **Stage F** (Level-1 leaf-only ablation) and
  **Stage G** (Level-2 decision-level stub-solver ablation) still remain — they
  gate the formal sub-step 2 close.
- **Also still queued** (unchanged by Q13): fold `fast_view` into the canonical
  `_build_view_6max` (with the `dcfr-overnight-3000` fp-repro gate); decide the
  `dcfr-overnight-3000` ICM-retrain after sub-step 6 measures the busted-seat bias.

## Next session opens with

A **pace decision, to be made at the start of next session**, not now:
- **Option A (confirmation-first):** run Stages F (Q11 Level-1) and possibly G
  (Q11 Level-2 stub-solver) for directional confirmation that BR moves hero's
  policy as hypothesized, *before* building sub-step 3.
- **Option B (push pace):** go straight to **sub-step 3 design (the real CFR
  loop)** and defer F/G as confirmation ablations that run after sub-step 3 lands.

Decide based on whether directional confidence in the BR choice is wanted before
committing to sub-step 3's implementation. Either way, **no evaluator optimization
work precedes it** — that is the durable Q13 result.
