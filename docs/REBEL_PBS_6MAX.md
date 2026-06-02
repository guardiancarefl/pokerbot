# 6-max Public Belief State (PBS) representation for ReBeL

The PBS is the object ReBeL's search and value net operate on. In heads-up
(the paper) it is `(public state, range_p0, range_p1)`. For 6-max it is
materially bigger — a **joint belief over up to 5 opponents** — which is the key
design piece this doc specifies. It builds on the repo's existing, validated
components rather than inventing new ones:

  - card abstraction: `runs/k200_abstraction.pkl` (preflop k=20, postflop k=200).
  - public/infoset features: `src/nlhe/infoset6.InfosetEncoder6Max` (236-dim).
  - subgame structure: `src/nlhe/subgame.build_subgame_tree` (depth-limited).
  - leaf value units: per-seat ICM-equity-delta 6-vector (the units
    `cfr6.traverse_6max` and `subgame_leaf.evaluate_leaves` already back up).

## 1. What is public vs private

**Public (observed by all):** board cards; per-street betting sequences;
pot; per-seat committed chips; per-seat stacks; per-seat active/folded mask;
button/position; tournament state (blinds level, ICM-relevant stack vector).
This is essentially what `parse_state_6max` already extracts.

**Private:** each still-active seat's 2 hole cards — abstracted to a **bucket**
in `{0..k-1}` for the current street (k=20 preflop, 200 postflop) via
`Abstraction.bucket_of`.

## 2. The belief (range) factorisation

A seat's **range** is a distribution over its abstraction buckets:
`r_seat ∈ Δ^{k}` (k = 20 or 200 by street). The **joint belief** is taken as the
**product of per-seat ranges** (the standard PBS independence factorisation —
exact under public observation given independent deals; the abstraction already
collapses card-removal effects into buckets):

    PBS = ( public_state , { r_seat : seat active } )

So a 6-max postflop PBS carries up to 6 range vectors of length 200. Folded
seats contribute no range. The hero's own range is degenerate during play (it
knows its hand) but is a full distribution from opponents' perspective — the
search maintains all of them because CFR needs every seat's counterfactual reach.

Belief propagation: as in the validated Leduc PBS (`src/rebel/pbs.py`), a seat's
range is its **reach** into the public node = product of that seat's blueprint/
solved action probabilities along the public betting, normalised per street.
`build_subgame_tree` already aggregates states into infosets; the range is the
reach-weight over the buckets within the root public node.

## 3. Value-net input encoding

Input = **public features ⊕ per-seat ranges**:

  - public block: the non-bucket part of `InfosetEncoder6Max` — street one-hot
    (4), seat-position one-hot (6), per-seat normalised stacks (6), per-seat
    active mask (6), per-seat normalised contributions (6), pot/to_call/eff-stack
    (3), betting features (5) ≈ **~36 dims**. (The 236-dim encoder = this block +
    a 200-dim bucket one-hot for ONE seat; the value net replaces that single
    one-hot with the full belief.)
  - belief block: 6 × k range vectors. Postflop = 6 × 200 = **1200 dims**;
    preflop = 6 × 20 = 120. Folded seats zero-filled + masked.

Total value-net input ≈ **~36 + 1200 ≈ 1.25k dims** postflop (vs the paper's
heads-up ~2× a few-hundred). Bigger but well within the 4090.

## 4. Value-net output

Per ReBeL, the net predicts the **counterfactual values the search needs at a
leaf PBS**. Two viable heads (decision deferred to Phase 3, pre-registered here):

  - **(A) per-seat value 6-vector** in ICM-equity-delta units — matches exactly
    what `subgame_leaf` returns and what `solve_subgame` backs up; simplest drop-in
    leaf replacement. Output dim 6.
  - **(B) per-bucket CFV for the acting seat** (200-vector) — richer (gives the
    search per-hand leaf CFVs, closer to the paper) but heavier to train/label.

**Recommendation:** start with (A) — it is the minimal change that makes
`evaluate_leaves` a net forward instead of a 7–20 s rollout (the measured
bottleneck), and it is the unit the existing solver consumes unmodified. Revisit
(B) only if (A) underfits the tight-matchup spots at GATE 2.

## 5. Targets / sample labels (Phase 2)

A training sample = `(PBS encoding, target)`. The target is the **CFR-solved
value at the root PBS** from `solve_subgame` (the per-seat root value derived
from `root_policy · root_q_values`), i.e. the depth-limited search's own output
— ReBeL's bootstrapped regression target. Round-1 bootstrap uses a cheap leaf
(PROFILE_SAMPLE off the k200 blueprint); later rounds use the trained net at
leaves, so generation accelerates ~100× once the net exists (leaf-eval is the
entire cost — CFR solve is ~0.05 s).

## 6. 6-max-specific caveats (carry into every gate)

  - **No Nash/safety theorem.** 6-max CFR converges empirically (3p Kuhn NashConv
    ~3e-5; see `src/rebel/multiplayer.py`) but not provably; the 2-player safe-
    resolve gadget (GATE 1) is reserved for the heads-up/3-handed endgame subgames
    Double-Up collapses to, where it directly applies. General 6-max relies on the
    value net for correctness, gated empirically on cash-rate vs k=200 (GATE 2).
  - **Independence factorisation is an approximation** (ignores cross-seat card
    removal beyond the bucketing). Acceptable given the abstraction; flagged.
  - **Belief dimensionality** (6×200) makes the value net the memory/throughput
    driver, not the CFR solve. Size the net for this input on the 4090.
