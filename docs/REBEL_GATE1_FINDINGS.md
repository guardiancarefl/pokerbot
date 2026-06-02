# ReBeL GATE 1 (Leduc search validation) — findings

**Status: GATE 1 NOT PASSED — stopped per the failed-gate norm.**
Root cause is a *known, expected* property (unsafe subgame solving), not a
plumbing bug. Every search-plumbing layer is byte-exact validated; the faithful
re-solving step is unsafe and must use safe subgame solving before any
NLHE / value-net work proceeds.

Branch: `rebel-search` (off `34aa725`). Code: `src/rebel/`, gate:
`scripts/validate_rebel_leduc.py`.

## What passed (byte-exact on Leduc)

| Layer | Claim | Result |
|-------|-------|--------|
| 1 plumbing | `DepthLimitedCFR(leaf_predicate=None)` ≡ validated engine | PASS — max avg-strategy diff `0.00e+00`, expl identical, 936/936 infosets |
| 2 PBS | public-state partition is bijective with engine infosets | PASS — 186 public nodes tile all 936 infosets; mass conserved |
| 3a subgame solver | `SubgameCFR` (single root = game root) ≡ engine | PASS — max diff `0.00e+00` |
| 3a' multi-root | `SubgameCFR` rooted post-deal, reach=ones, no cut | PASS — reproduces Nash round-2 to `0.014` |

The search recursion, the leaf-substitution hook, the PBS public partition, and
the PBS-rooted multi-root subgame solver are all correct.

## What failed — and why it is NOT a bug

**Layer 3b (negative control, EXPECTED to fail): fixed leaf values are unsafe.**
Feeding the depth-limited round-1 solve the *exact Nash continuation values* as
**static** leaves yields exploitability **219 mbb/g** (vs Nash 0.123). Pinning
round-2 to fixed values removes the opponent's ability to best-respond in round
2, so the trunk strategy becomes exploitable. This reproduces, on a solvable
game, the repo's prior "leaf-value precision/bias vs unsafe solve" finding
(M1.5/M2 resolvers, retired Session 6).

**Layer 4 (the gate): belief-conditioned re-solving is *also* unsafe without a
safety gadget.** Re-solving the round-2 subgame for the current belief (so the
opponent's round-2 adapts) was the expected fix. It is not sufficient:

- Acid test — combine the **true Nash round-1** with the **belief-conditioned
  re-solved round-2**: exploitability **54.6 mbb/g** (should be ≈0.12 if safe).
- The belief fed to the re-solve is **exactly** the true Nash round-1 reach
  (verified: `max|belief − true_reach| = 0.0`).
- Two independent re-solve implementations (belief-seeded `SubgameCFR`, and
  frozen-round-1 `FrozenTrunkCFR`) produce the **identical** wrong result
  (54.595 mbb/g, strat max-diff 0.4754) — a shared cause, not a typo.
- The split/recombine + exploitability machinery is proven correct (full's own
  r1+r2 recombined → 1.1523 = engine, keys/values identical).

**Diagnosis — unsafe subgame solving.** Even with *exact* beliefs, a subgame
solved in isolation best-responds only to that fixed range. Exploitability lets
the opponent play *any* round-1 strategy, arriving at round 2 with an off-Nash
range; the isolated re-solve is a best response only to the Nash range and is
exploitable against the induced off-Nash range. The full *joint* solve is safe
because round-1 and round-2 are optimised together. This is the standard
unsafe-resolving result (Burch/Johanson/Bowling CFR-D 2014; Brown–Sandholm safe
& nested subgame solving 2017), and it is exactly the failure mode this Leduc
gate exists to catch before NLHE.

## Path forward (needs a direction decision)

Pass Gate 1 by making the re-solve **safe**. Options, in rough order of effort:

1. **CFR-D / re-solving gadget (recommended):** at the subgame root give the
   constrained player a per-infoset "opt-out" valued at their blueprint
   counterfactual value, so the re-solved strategy cannot be exploited by range
   deviation. Standard, well-specified; the right primitive for NLHE re-solving.
2. **Max-margin / Reach subgame solving (Brown–Sandholm 2017):** stronger
   safety/exploitability bounds, more involved.
3. **Full ReBeL PBS-value self-play:** solve the subgame rooted at the PBS with
   the value net at leaves and update beliefs each CFR iteration (ReBeL Alg. 1);
   safety comes from the PBS-value construction + accurate values. This is the
   eventual target anyway, but validating it on Leduc still needs the safe
   re-solve primitive at the leaf.

## Gadget attempt #1 (FAILED) — 2026-06-02

First CFR-D gadget implementation (`src/rebel/gadget.py`) used a **non-standard
symmetric outer-loop-on-ranges** formulation: both players get opt-outs, ranges
rescaled by per-hand FOLLOW probabilities between warm-started inner re-solves.

Acid test (blueprint = Nash, exploitability of Nash-r1 + gadget-safe-r2):
**410.6 mbb/g** — *worse* than the unsafe 54.6, and 405s/run.

Likely causes (untested): (a) symmetric both-player gadget over-constrains /
is the wrong game vs the textbook **one-sided** gadget (fix re-solver range,
gadget only the opponent, run once per player); (b) the outer-loop-on-ranges is
an unreliable approximation of the gadget — the correct method builds an
**augmented game tree** (per-hand FOLLOW/TERMINATE nodes) and solves it with a
single CFR; (c) warm-started inner solver averages across changing ranges,
corrupting the CFVs the gadget regret consumes. The proper one-sided
augmented-tree gadget remains untried and is a meaningful additional build.

**Decision point (per the failed-gate / time-box norm):** invest in the proper
one-sided augmented-tree gadget, or reconsider scope. See session report.

## Original recommendation
**Recommendation:** implement the CFR-D gadget (option 1) as the safe re-solve
primitive, re-run Layer 4 on Leduc (target: assembled exploitability → ≈Nash),
and only then proceed to the 6-max PBS value-net + Phase 2. Do **not** start
NLHE sample-gen until Gate 1 is green — an unsafe search would invalidate every
downstream number.
