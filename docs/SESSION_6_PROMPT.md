# Session 6 starter prompt

Paste this verbatim at the start of the next Claude conversation.

---

Continuing the pokerbot project. Session 6.

**Status:** Session 5 closed with a strategic pivot. Phase 4f (parallel training + Shanky diversity-mix experiment) produced a working parallel framework (G=10 at ~4.5x speedup, BLAS-pinned) and a completed DCFR self-play anchor at k=200, iter 2000 — but the anchor's lift trajectory revealed the bot plateaus around iter 500-1000 at this abstraction level (strat_loss refines but head-to-head strength doesn't improve). The bot loses to 15 of 19 sampled commercial Shanky scripted bots. Honest conclusion: k=200 is the bottleneck, not training iters. Diversity-mix experiment shelved. See docs/DECISIONS.md for full reasoning.

**Session 6 focus: Layer 3 — real-time subgame solving.**

This is the architectural keystone of the project. The 4-layer stack in ARCHITECTURE.md places it as Phase 3+, but Session 5's findings promoted it to immediate priority. Layer 3 is the difference between "Nash within an abstraction" (what we have) and Pluribus-level play (Nash + real-time refinement at decision time).

**Session 6 deliverables (design phase, no code yet):**

1. **Literature recon.** Read the relevant published work on subgame solving for CFR-based poker bots. Key references: Pluribus (Brown & Sandholm 2019) for 6-max continual resolving and safe subgame solving; Libratus (HUNL precedent); DeepStack (continual resolving variant). Identify which variant best fits our constraints (CPU-only on Contabo, 6-max not HUNL, ICM-adjusted value function).

2. **Integration design.** Concrete plan for how Layer 3 plugs into the existing blueprint:
   - Where in the play loop the subgame solver runs (at every decision, only certain streets, only when stack depth exceeds threshold?)
   - How blueprint values are used at subgame boundaries (the "anchor" for the subgame)
   - Action abstraction during resolve (finer than blueprint? same?)
   - Card abstraction during resolve (finer? bucket-pinned to blueprint?)
   - Computational budget per decision (target: sub-second; willing to spend more during evaluation)

3. **Build vs blueprint-first decision.** The question is whether to:
   - (a) Build Layer 3 on top of the current k=200 blueprint immediately (faster start, weaker base)
   - (b) Train a k=500 blueprint first (we have the abstraction artifact already, ~6-8h compute) then build Layer 3
   - (c) Train at even higher k (e.g. k=2000) for the production blueprint, then build Layer 3
   The choice depends on Layer 3's expected lift over blueprint alone, which the literature recon should help estimate.

4. **Implementation skeleton.** Once the design is settled, sketch the code structure: new module locations (src/nlhe/subgame/?), key classes, integration with existing solver and policy adapter.

**Workflow rules carrying over from previous sessions:**
- Diff-on-copy validation discipline (write to /tmp, validate, apply for real, only after gates green)
- Two-SSH-session workflow on Contabo (one for long-running compute, one for editing/committing)
- Verify file writes with wc/head/tail/grep rather than trusting terminal echo
- Benchmark one iteration / one decision before committing to a long run
- All commits include validation status; no "trust me" commits
- When exact file contents are given between BEGIN/END markers, write them verbatim; concerns are questions before writing, not silent edits

**Available infrastructure (don't rebuild):**
- Contabo VPS at quant@80.241.219.63, project at ~/pokerbot/, repo github.com/guardiancarefl/pokerbot branch phase4f-league
- Python 3.10 venv with PyTorch 2.12 CPU, OpenSpiel 1.6.11
- Parallel training framework (parallel_train in src/nlhe/parallel/orchestrator.py), G=10 production operating point
- k=200 blueprint with iter 2000 checkpoint at runs/dcfr_anchor_2000/checkpoints/ckpt_iter_2000.pt
- k=500 abstraction artifact (untrained) at runs/abstraction_k500_20260529_013537/abstraction.pkl
- 24-profile Shanky league pool at data/shanky_profiles/ + registry at configs/league/registry_experiment.json
- Production runner: scripts/run_training.sh
- Mini-eval dashboard with colored lift readings (lift.log.ansi)
- All 25 gates passing on HEAD (b2571aa or later)

**First step of Session 6:** read docs/STATUS.md, docs/DECISIONS.md (specifically the last entry about Layer 3 promotion), docs/ARCHITECTURE.md (Layer 3 description), and confirm state matches before starting design work.
