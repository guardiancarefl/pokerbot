# Track A3: Card Abstraction Comparison Harness

**Started:** Session 7.5 (2026-05-22 night).
**Estimated:** 3-4 weeks of focused work across multiple sessions.
**Goal:** Replace the current EMD-with-k=20-preflop abstraction with the strongest
publicly-known method, validated against alternatives with measured data
rather than guessed-at theoretical advantages.

## Scope

Implement three card abstractions, train HUNL Deep CFR on each at matched
compute budget, evaluate against Slumbot, pick the winner.

### Option 0 (baseline): current EMD, k=20/k=200
Already exists at `runs/abstraction_20260521_223018/`. The Phase 2d GPU
result (+31.45 bb/100 vs Slumbot at ckpt_iter_0100) is our reference point.
No new work, just reused.

### Option 1: EMD with more buckets
Same EMD-histogram algorithm and code path, just bigger k:
- Preflop: k=169 (lossless — every starting hand its own bucket)
- Flop/turn/river: k=500

Status: trainer at `scripts/train_abstraction_a3_option1.py`, started Session 7.5.
Output will live at `runs/abstraction_a3_option1_<timestamp>/`.

### Option 2 (deferred): river OCHS overlay on Option 1
Adds river-only OCHS feature per Johanson 2013 IR-KE-KO. Postponed until
Option 1 baseline result is in hand — if Option 1 alone fixes the AA/TT
collision and produces a meaningfully better bot, OCHS overlay is incremental
polish rather than a separate experiment.

### Option 4: KrwEmd (Fu et al. 2025)
State-of-the-art per the November 2025 paper. Incorporates historical
information that imperfect-recall abstractions discard.

**Algorithm:**

For each hand at phase r, compute the k-recall winrate feature (KRWF):
rf(r,k)(psi) = [wo(r)(psi); wo(r-1)(psi); ...; wo(r-k)(psi)]
where each `wo(j)(psi)` is the 3-dim vector `(P(lose), P(tie), P(win))` at phase j.

For two hands psi and psi-prime, define the distance as:
D(rf(psi), rf(psi-prime)) = sum_{j=0..k} w_j * EMD(pf(r-j)(psi), pf(r-j)(psi-prime))
The EMD here is the standard earth-mover's distance with the ground distance matrix:
ground = [[0, 1, 2],
[1, 0, 1],
[2, 1, 0]]
Loss-to-loss = 0, loss-to-win = 2, etc.

Then cluster with k-means++ using this distance.

**Recommended weights** (from paper, exponentially decreasing):
- Phase 3 (river) with k=2: `(w0, w1, w2) = (16, 4, 1)`
- Phase 2 (turn) with k=2: `(w0, w1, w2) = (16, 4, 1)` or `(4, 1)` for k=1
- Phase 1 (flop) with k=1: `(w0, w1) = (4, 1)`
- Phase 0 (preflop) k=0: just the current winrate

**Honest caveat:** the paper validated KrwEmd only on "Numeral211 hold'em",
a simplified custom 3-phase poker variant the authors designed. No published
HUNL CFR experiments. We would be doing the first real-HUNL validation.

## Comparison harness

After all three abstractions are trained, the harness produces:

1. **Bucket-equity analysis** for each abstraction (rerun
   `scripts/analyze_bucket_equity.py` with the new abstraction path). Confirms
   the collision pathology is fixed (or not).
2. **HUNL Deep CFR training run** at matched compute budget for each. Same
   network architecture, same hyperparameters, same iteration count, same
   seed. Only the underlying abstraction differs.
3. **Slumbot evaluation** at the same checkpoint iteration. bb/100 over
   sufficient hands (5000+) for statistical significance.
4. **Optional: head-to-head between the three** at fixed compute. Pairwise
   crosstable.

The winner becomes the project's production abstraction. The losers are
preserved as artifacts at `runs/abstraction_*/`.

## Open design questions

These need locking before Option 4 implementation:

1. **Memory budget.** KrwEmd at k=2 for the turn produces 37M signal observation
   infosets per the paper's HUNL table. We can't enumerate that many on Contabo
   RAM. Need to either (a) sample a representative subset, (b) limit k more
   aggressively, or (c) use the same 5000-hand sampling we use for EMD.
2. **Computational cost.** KrwEmd's distance is a weighted sum of EMDs over
   k+1 phases. For k=2 that's 3x the per-pair EMD cost of plain EMD. Plus the
   feature construction requires equity rollouts per hand per phase, which we
   already do. Net: maybe 5-10x slower abstraction training than Option 1.
   On Contabo CPU, the Option 1 flop step is taking ~10 minutes (preliminary
   observation). KrwEmd's flop+turn+river could be 1-3 hours total. Acceptable.
3. **EMD ground distance for win/tie/lose**. Paper gives the 3x3 matrix above.
   We use it as-is; no reason to deviate.
4. **Weight tuning.** Paper recommends exponentially decreasing. We use that
   as default. If results are weak we revisit.

## Multi-session plan

**Session 7.5 (current, partial):**
- (DONE) Decision locked: Option 4 with comparison harness.
- (DONE) Paper read and algorithm extracted.
- (RUNNING) Option 1 training: `scripts/train_abstraction_a3_option1.py`.
- (THIS DOC) Design plan committed.

**Session 8:**
- Verify Option 1 training completed; rerun
  `scripts/analyze_bucket_equity.py` against it. Confirm AA/TT/QQ no longer
  collide.
- Sketch the KrwEmd module: `src/nlhe/krwemd.py`. Decide memory and sampling
  strategy. Lock the API surface.
- Begin implementation of the k-recall winrate feature computation.

**Session 9:**
- Finish KrwEmd feature computation + distance metric. Tests on
  hand-computable examples.
- Begin clustering integration.

**Session 10:**
- Finish clustering. Run end-to-end KrwEmd training on Contabo.
- Output a new Abstraction artifact.

**Session 11:**
- Build the comparison harness driver:
  `scripts/run_a3_comparison.py`. Train three Deep CFR runs at matched compute.
  Eval each vs Slumbot.

**Session 12:**
- Crunch results. Pick winner. Document. Commit. Decide whether Option 2
  (river OCHS overlay) is worth chasing or whether we move on to Track B1.

This is 5-6 focused sessions. Comfortable inside the "3-4 weeks" budget.

## Decision recording

The reasoning behind picking Option 4 with comparison harness over Option 1
alone or B1 (subgame solver) is in `docs/DECISIONS.md` — added in commit
during Session 7.5.

## Reference

- Johanson, Burch, Valenzano, Bowling (2013). *Evaluating State-Space
  Abstractions in Extensive-Form Games.* AAMAS-13. Canonical OCHS paper.
- Brown, Sandholm (2019). *Superhuman AI for multiplayer poker.* Science.
  Pluribus paper, describes PAAEMD blueprint abstraction.
- Fu, Xu, Bai, Zhao, Huang (2025). *Signal Observation Models and Historical
  Information Integration in Poker Hand Abstraction.* arXiv:2403.11486v3.
  KrwEmd theoretical foundation.
- Fu, Yin, Liu, Xu, Huang (2025). *KrwEmd: Revising the Imperfect-Recall
  Abstraction from Forgetting Everything.* arXiv:2511.12089. Practical
  KrwEmd algorithm.
