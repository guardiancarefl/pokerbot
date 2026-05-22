# Prompt for Session 6 (paste this verbatim at the start of the next conversation)

---

Continuing the pokerbot project. Session 6.

**Status:** Phase 2d closed in Session 5. The pipeline scaled from CPU to GPU successfully. GPU-trained ckpt_iter_0100 evaluated against Slumbot at **+31.45 baseline-adjusted bb/100** (1000 hands, seed 2026, 200bb), a +46 bb/100 improvement over the CPU baseline of -14.8. Bigger buffer was the lever, not bigger network — strategy loss plateau dropped from ~0.95 (100K buffer) to ~0.85 (500K buffer) at the same [512,512] network.

**Major shift in Session 5:** project goal sharpened from "specialized SNG bot" to "**strongest publicly-known 6-max NLHE SNG bot with correct ICM**." Targets include beating Slumbot in HUNL, decisive wins against the 42 bought bots, 70%+ top-3 finish rate in SNG simulations, sub-second decisions via subgame solving, and plausibly beating Pluribus head-to-head in 6-max cash. Architecture revised: subgame solving moves up from Phase 5-6 to Phase 3 as a parallel track. Within-match opponent modeling (continuous archetype + Bayesian updating) moves up from Phase 6 to Phase 3 as another parallel track. Estimated 6-10 weeks of focused work, $500-2000 in compute.

**Runtime:** Contabo VPS, Ubuntu 24.04, 12 vCPU AMD EPYC, 48GB RAM, no GPU. Project at `~/pokerbot/` on Contabo. Python 3.10 venv with PyTorch 2.12 CPU + OpenSpiel 1.6.11 + treys. Repo at github.com/guardiancarefl/pokerbot (private).

**Active resources:**
- RunPod RTX PRO 4000 Blackwell pod ($0.57/hr Secure Cloud) at 213.173.107.11:46595 may still be running ckpt_iter_0100+. Check status with `ssh root@213.173.107.11 -p 46595 -i ~/.ssh/id_ed25519 'tail -5 /tmp/v2_run.log'` from Contabo.
- ~$10 RunPod credit remaining
- ~$93 Vultr credit reserved for Phase 4

**Session 6 priorities (from docs/PHASE3_PLAN.md):**

1. **Decide whether the Phase 2d GPU training is worth continuing.**
   - Check latest checkpoint on pod (probably ckpt_iter_0200 or 0300 by morning)
   - If interesting, eval that checkpoint against Slumbot to chart the bb/100 trajectory
   - Either way, terminate the pod once we have what we need — Phase 3 work is on Contabo CPU

2. **Begin Track A1: Implement DCFR (Linear / Discounted CFR).**
   - Add per-buffer-entry timestamp/iteration tracking to ReservoirBuffer
   - Weight strategy training loss by iter index (linear) or configurable exponent (discounted)
   - Validate on Leduc first (where exploitability is computable), then on HUNL smoke
   - Expected outcome: 1.5-3x faster convergence to comparable Slumbot bb/100

3. **Begin Track A2: Hand-engineered archetype framework design.**
   - 2D parameterization (tightness × aggression)
   - Behavioral rule sets per region of the 2D space
   - Plan integration into training opponent pool

4. **Begin Track B1: Subgame extractor design.**
   - OpenSpiel API research for subgame extraction
   - Define depth-limit policy
   - Sketch the data flow (state → subgame → solver → policy)

5. **Begin Track C1: Continuous archetype belief representation design.**
   - Gaussian or particle representation over tightness × aggression
   - Action likelihood model (P(action | archetype, infoset))
   - Bayesian update math

Items 2-5 are design + first implementation. Don't try to complete all four in Session 6 — sketch each, pick one to develop in depth.

**Key files to reference:**
- `docs/STATUS.md` — current state
- `docs/PHASE3_PLAN.md` — full Phase 3 plan (THE working document for next 6-10 weeks)
- `docs/DECISIONS.md` — the goal-sharpening, three-track plan, and buffer-vs-network findings from Session 5
- `docs/SESSION_LOG.md` — Session 5 entry has the full GPU validation story
- `runs/gpu_phase2d_artifacts/` — preserved eval log and v2 config from Phase 2d GPU run
- `src/nlhe/solver.py` — Deep CFR solver, now with GPU device support and RNG state try/except (touch carefully for DCFR)
- `src/nlhe/policy_adapter.py` — Slumbot eval entry point with the device-aware `.cpu()` fix

**Workflow rules carrying over from previous sessions:**
- Single-quoted heredoc delimiters with distinctive names for file writes
- Verify file writes with wc -l / grep / head / tail
- Verify each patch landed before moving to the next step
- Benchmark before committing to long runs
- Kill criterion: kill when the type of problem changes, not when "taking longer than hoped"
- Probe APIs with curl/manual tests before writing client code
- Files in /mnt/project/ are stale; trust live repo on Contabo
- tmux for any long-running session
- When patches touch device handling, smoke test the GPU inference path before committing — CPU unit tests don't catch GPU-only issues (Session 5 lesson)
- One-liner status checks from Contabo (`ssh ... 'tail -5 /tmp/log'`) cheaper than full SSH attach

Please read the project knowledge docs (STATUS.md as entry point, then DECISIONS.md for the new decisions, then PHASE3_PLAN.md for the working plan, then SESSION_LOG.md Session 5 entry for the recent context). Confirm state matches what STATUS says. Then start with item 1: check the pod status and decide whether to continue Phase 2d GPU training.
